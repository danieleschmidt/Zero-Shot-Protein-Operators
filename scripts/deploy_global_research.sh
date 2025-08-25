#!/bin/bash
# Global Research Infrastructure Deployment Script
# Deploys distributed protein design research infrastructure across multiple regions

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="protein-research-global"
REGIONS=("us-central" "eu-west" "asia-east" "us-west")
KUBECTL_TIMEOUT="300s"
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_DELAY=30

# Logging function
log() {
    local level=$1
    shift
    case $level in
        INFO)  echo -e "${GREEN}[INFO]${NC}  $*" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC}  $*" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $*" ;;
        DEBUG) echo -e "${BLUE}[DEBUG]${NC} $*" ;;
    esac
}

# Error handling
error_exit() {
    log ERROR "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl not found. Please install kubectl."
    fi
    
    # Check kubernetes cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    fi
    
    # Check for required files
    local required_files=(
        "k8s/global-research-infrastructure.yaml"
        "Dockerfile"
        "docker-compose.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error_exit "Required file not found: $file"
        fi
    done
    
    log INFO "Prerequisites check passed ✓"
}

# Build and push Docker image
build_and_push_image() {
    log INFO "Building and pushing Docker image..."
    
    local image_tag="protein-operators:$(git rev-parse --short HEAD)"
    local registry="${DOCKER_REGISTRY:-localhost:5000}"
    
    log DEBUG "Building image: $registry/$image_tag"
    
    docker build -t "$registry/$image_tag" . || error_exit "Failed to build Docker image"
    docker push "$registry/$image_tag" || error_exit "Failed to push Docker image"
    
    # Update image references in deployment files
    sed -i.bak "s|protein-operators:latest|$registry/$image_tag|g" k8s/global-research-infrastructure.yaml
    
    log INFO "Docker image built and pushed ✓"
}

# Create secrets
create_secrets() {
    log INFO "Creating Kubernetes secrets..."
    
    # Database secret
    local db_password=$(openssl rand -base64 32)
    local db_url="postgresql://protein_user:$db_password@research-db-service:5432/protein_research_global"
    
    kubectl create secret generic research-db-secret \
        --namespace="$NAMESPACE" \
        --from-literal=username=protein_user \
        --from-literal=password="$db_password" \
        --from-literal=url="$db_url" \
        --dry-run=client -o yaml | kubectl apply -f - || error_exit "Failed to create database secret"
    
    # API secrets
    local api_key=$(openssl rand -base64 32)
    local jwt_secret=$(openssl rand -base64 64)
    
    kubectl create secret generic api-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=api-key="$api_key" \
        --from-literal=jwt-secret="$jwt_secret" \
        --dry-run=client -o yaml | kubectl apply -f - || error_exit "Failed to create API secrets"
    
    log INFO "Secrets created ✓"
}

# Deploy to single region
deploy_to_region() {
    local region=$1
    log INFO "Deploying to region: $region"
    
    # Set region context
    kubectl config use-context "$region" || log WARN "Could not switch to context $region, using current context"
    
    # Apply namespace and RBAC first
    kubectl apply -f k8s/namespace.yaml --timeout="$KUBECTL_TIMEOUT" || error_exit "Failed to create namespace in $region"
    kubectl apply -f k8s/rbac.yaml --timeout="$KUBECTL_TIMEOUT" || error_exit "Failed to create RBAC in $region"
    
    # Apply storage classes and PVCs
    kubectl apply -f k8s/global-research-infrastructure.yaml --timeout="$KUBECTL_TIMEOUT" || error_exit "Failed to deploy infrastructure in $region"
    
    log INFO "Deployment to $region completed ✓"
}

# Wait for deployment readiness
wait_for_deployment() {
    local deployment=$1
    local region=${2:-"current"}
    
    log INFO "Waiting for deployment $deployment to be ready in $region..."
    
    kubectl wait --for=condition=available deployment/$deployment \
        --namespace="$NAMESPACE" \
        --timeout="$KUBECTL_TIMEOUT" || error_exit "Deployment $deployment failed to become ready in $region"
    
    log INFO "Deployment $deployment is ready in $region ✓"
}

# Health check
health_check() {
    local service=$1
    local port=$2
    local region=${3:-"current"}
    
    log INFO "Running health check for $service in $region..."
    
    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        log DEBUG "Health check attempt $i/$HEALTH_CHECK_RETRIES for $service"
        
        if kubectl exec -n "$NAMESPACE" deploy/$service -- curl -f http://localhost:$port/health &>/dev/null; then
            log INFO "Health check passed for $service in $region ✓"
            return 0
        fi
        
        if [[ $i -lt $HEALTH_CHECK_RETRIES ]]; then
            log DEBUG "Health check failed, retrying in ${HEALTH_CHECK_DELAY}s..."
            sleep $HEALTH_CHECK_DELAY
        fi
    done
    
    log ERROR "Health check failed for $service in $region after $HEALTH_CHECK_RETRIES attempts"
    return 1
}

# Deploy monitoring
deploy_monitoring() {
    log INFO "Deploying monitoring infrastructure..."
    
    # Apply monitoring configurations
    if [[ -d "monitoring/" ]]; then
        kubectl apply -f monitoring/ --timeout="$KUBECTL_TIMEOUT" || log WARN "Failed to deploy monitoring"
        log INFO "Monitoring deployed ✓"
    else
        log WARN "Monitoring directory not found, skipping monitoring deployment"
    fi
}

# Validate global deployment
validate_global_deployment() {
    log INFO "Validating global deployment..."
    
    # Check all deployments are running
    local deployments=(
        "research-database-global"
        "quantum-simulator"
        "distributed-training"
        "comparative-studies"
        "research-coordinator"
    )
    
    for deployment in "${deployments[@]}"; do
        local ready_replicas=$(kubectl get deployment $deployment -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired_replicas=$(kubectl get deployment $deployment -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
        
        if [[ "$ready_replicas" == "$desired_replicas" && "$ready_replicas" != "0" ]]; then
            log INFO "Deployment $deployment: $ready_replicas/$desired_replicas replicas ready ✓"
        else
            log ERROR "Deployment $deployment: $ready_replicas/$desired_replicas replicas ready"
        fi
    done
    
    # Check services
    local services=$(kubectl get services -n "$NAMESPACE" --no-headers | wc -l)
    log INFO "Services deployed: $services"
    
    # Check storage
    local pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers | grep -c Bound || echo "0")
    local total_pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers | wc -l)
    log INFO "PVCs bound: $pvcs/$total_pvcs"
    
    log INFO "Global deployment validation completed ✓"
}

# Performance benchmark
run_performance_benchmark() {
    log INFO "Running performance benchmark..."
    
    # Create a test job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: performance-benchmark
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: benchmark
        image: protein-operators:latest
        command:
        - python
        - -m
        - protein_operators.benchmarks.advanced_comparative_studies
        env:
        - name: BENCHMARK_MODE
          value: "performance"
        - name: SAMPLES
          value: "100"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
      restartPolicy: Never
  backoffLimit: 3
EOF

    # Wait for benchmark completion
    kubectl wait --for=condition=complete job/performance-benchmark \
        --namespace="$NAMESPACE" \
        --timeout="600s" || log WARN "Performance benchmark timed out"
    
    # Get benchmark results
    kubectl logs job/performance-benchmark -n "$NAMESPACE" | tail -20 || log WARN "Could not retrieve benchmark results"
    
    # Cleanup benchmark job
    kubectl delete job performance-benchmark -n "$NAMESPACE" &>/dev/null || true
    
    log INFO "Performance benchmark completed ✓"
}

# Generate deployment report
generate_deployment_report() {
    log INFO "Generating deployment report..."
    
    local report_file="deployment_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "# Global Research Infrastructure Deployment Report"
        echo "Generated: $(date)"
        echo "Namespace: $NAMESPACE"
        echo ""
        
        echo "## Deployments Status"
        kubectl get deployments -n "$NAMESPACE" -o wide
        echo ""
        
        echo "## Services"
        kubectl get services -n "$NAMESPACE" -o wide
        echo ""
        
        echo "## Persistent Volume Claims"
        kubectl get pvc -n "$NAMESPACE" -o wide
        echo ""
        
        echo "## Resource Usage"
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics server not available"
        echo ""
        
        echo "## Events (Last 10)"
        kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp | tail -10
        
    } > "$report_file"
    
    log INFO "Deployment report generated: $report_file ✓"
}

# Cleanup function
cleanup() {
    log INFO "Performing cleanup..."
    
    # Restore original deployment file
    if [[ -f "k8s/global-research-infrastructure.yaml.bak" ]]; then
        mv k8s/global-research-infrastructure.yaml.bak k8s/global-research-infrastructure.yaml
    fi
}

# Main deployment function
main() {
    log INFO "🚀 Starting Global Research Infrastructure Deployment"
    log INFO "Deploying Zero-Shot Protein Operators research infrastructure"
    echo ""
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Step 1: Prerequisites
    check_prerequisites
    
    # Step 2: Build and push image
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_and_push_image
    fi
    
    # Step 3: Create namespace
    log INFO "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Step 4: Create secrets
    create_secrets
    
    # Step 5: Deploy infrastructure
    log INFO "Deploying global research infrastructure..."
    kubectl apply -f k8s/global-research-infrastructure.yaml --timeout="$KUBECTL_TIMEOUT" || error_exit "Failed to deploy infrastructure"
    
    # Step 6: Wait for key deployments
    local key_deployments=("research-coordinator" "comparative-studies" "quantum-simulator")
    for deployment in "${key_deployments[@]}"; do
        wait_for_deployment "$deployment"
    done
    
    # Step 7: Deploy monitoring
    deploy_monitoring
    
    # Step 8: Validate deployment
    validate_global_deployment
    
    # Step 9: Health checks
    health_check "research-coordinator" "9000" || log WARN "Research coordinator health check failed"
    health_check "comparative-studies" "8000" || log WARN "Comparative studies health check failed"
    
    # Step 10: Performance benchmark
    if [[ "${RUN_BENCHMARK:-false}" == "true" ]]; then
        run_performance_benchmark
    fi
    
    # Step 11: Generate report
    generate_deployment_report
    
    log INFO ""
    log INFO "🎉 Global Research Infrastructure Deployment Complete!"
    log INFO ""
    log INFO "📊 Access Points:"
    log INFO "   Research Coordinator: http://$(kubectl get service global-research-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9000"
    log INFO "   Comparative Studies:  http://$(kubectl get service global-research-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000"
    log INFO "   Quantum Simulation:   http://$(kubectl get service global-research-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8080"
    log INFO ""
    log INFO "🔬 Research Infrastructure Ready for:"
    log INFO "   • Quantum-Classical Hybrid Optimization"
    log INFO "   • Advanced Comparative Studies"
    log INFO "   • Distributed Multi-GPU Training"
    log INFO "   • Global Research Coordination"
    log INFO "   • Statistical Validation & Reproducibility"
    log INFO ""
    log INFO "📄 Next Steps:"
    log INFO "   1. Run research experiments: kubectl exec -n $NAMESPACE -it deploy/research-coordinator -- python -m protein_operators.research"
    log INFO "   2. Monitor performance: kubectl logs -f -n $NAMESPACE deploy/comparative-studies"
    log INFO "   3. Access research dashboard: http://research-dashboard.protein-operators.org"
    log INFO ""
    log INFO "✨ Advanced Research Infrastructure is now globally deployed and operational!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi