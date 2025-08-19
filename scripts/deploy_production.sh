#!/bin/bash

# Autonomous SDLC Production Deployment Script
# Deploys the complete protein operators system to production Kubernetes cluster

set -euo pipefail

# Configuration
PROJECT_NAME="protein-operators"
NAMESPACE="protein-operators-prod"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-your-registry.com}"
KUBECONFIG="${KUBECONFIG:-~/.kube/config}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
trap 'log_error "Deployment failed on line $LINENO. Exit code: $?"' ERR

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed (for building)
    if [[ "$SKIP_BUILD" != "true" ]] && ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed. Some features may not be available."
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended for deployment"
    fi
    
    log_success "Prerequisites check passed"
}

# Function to run quality gates
run_quality_gates() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping quality gates as requested"
        return 0
    fi
    
    log_info "Running quality gates..."
    
    # Run the comprehensive quality gates
    if ! python scripts/run_quality_gates.py --fail-on-warning; then
        log_error "Quality gates failed. Deployment aborted."
        exit 1
    fi
    
    log_success "Quality gates passed"
}

# Function to build and push Docker image
build_and_push_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping image build as requested"
        return 0
    fi
    
    log_info "Building and pushing Docker image..."
    
    local image_name="$CONTAINER_REGISTRY/$PROJECT_NAME:$IMAGE_TAG"
    
    # Build the Docker image
    log_info "Building image: $image_name"
    docker build -t "$image_name" -f Dockerfile .
    
    # Tag as latest if this is a release
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        docker tag "$image_name" "$CONTAINER_REGISTRY/$PROJECT_NAME:latest"
    fi
    
    # Push to registry
    log_info "Pushing image to registry..."
    docker push "$image_name"
    
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        docker push "$CONTAINER_REGISTRY/$PROJECT_NAME:latest"
    fi
    
    log_success "Image built and pushed successfully"
}

# Function to create namespace
create_namespace() {
    log_info "Creating namespace if not exists..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
    
    # Label the namespace
    kubectl label namespace "$NAMESPACE" app="$PROJECT_NAME" environment=production --overwrite
}

# Function to setup secrets
setup_secrets() {
    log_info "Setting up secrets..."
    
    # Check if secrets already exist
    if kubectl get secret protein-operators-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist. Skipping secret creation."
        log_warning "To update secrets, delete them first: kubectl delete secret protein-operators-secrets -n $NAMESPACE"
        return 0
    fi
    
    # Generate secure passwords if not provided
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    JWT_SECRET="${JWT_SECRET:-$(openssl rand -base64 64)}"
    API_KEY="${API_KEY:-$(openssl rand -base64 32)}"
    
    # Create secrets
    kubectl create secret generic protein-operators-secrets \
        --from-literal=POSTGRES_USER=protein_operators \
        --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --from-literal=REDIS_PASSWORD="$REDIS_PASSWORD" \
        --from-literal=JWT_SECRET="$JWT_SECRET" \
        --from-literal=API_KEY="$API_KEY" \
        -n "$NAMESPACE"
    
    log_success "Secrets created successfully"
    log_warning "Save these credentials securely:"
    log_warning "POSTGRES_PASSWORD: $POSTGRES_PASSWORD"
    log_warning "REDIS_PASSWORD: $REDIS_PASSWORD"
    log_warning "JWT_SECRET: $JWT_SECRET"
    log_warning "API_KEY: $API_KEY"
}

# Function to deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL..."
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: protein_operators
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: protein-operators-secrets
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: protein-operators-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
EOF
    
    # Deploy Redis
    log_info "Deploying Redis..."
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: protein-operators-secrets
              key: REDIS_PASSWORD
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF
    
    log_success "Infrastructure components deployed"
}

# Function to wait for infrastructure to be ready
wait_for_infrastructure() {
    log_info "Waiting for infrastructure to be ready..."
    
    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL..."
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n "$NAMESPACE"
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n "$NAMESPACE"
    
    # Give them a bit more time to fully start
    sleep 30
    
    log_success "Infrastructure is ready"
}

# Function to deploy the main application
deploy_application() {
    log_info "Deploying main application..."
    
    # Update the image in the deployment manifest
    local temp_manifest="/tmp/deployment-${RANDOM}.yaml"
    cp k8s/autonomous-sdlc-production.yaml "$temp_manifest"
    
    # Replace image references
    sed -i "s|image: protein-operators:latest|image: $CONTAINER_REGISTRY/$PROJECT_NAME:$IMAGE_TAG|g" "$temp_manifest"
    
    # Apply the manifest
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would apply the following manifest:"
        kubectl apply -f "$temp_manifest" --dry-run=client
    else
        kubectl apply -f "$temp_manifest"
    fi
    
    # Clean up temp file
    rm -f "$temp_manifest"
    
    log_success "Application deployed"
}

# Function to wait for application deployment
wait_for_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Skipping deployment wait"
        return 0
    fi
    
    log_info "Waiting for application deployments to be ready..."
    
    # Wait for API deployment
    log_info "Waiting for API deployment..."
    kubectl wait --for=condition=available --timeout=600s deployment/protein-operators-api -n "$NAMESPACE"
    
    # Wait for worker deployment
    log_info "Waiting for worker deployment..."
    kubectl wait --for=condition=available --timeout=600s deployment/protein-operators-worker -n "$NAMESPACE"
    
    # Wait for coordinator deployment
    log_info "Waiting for coordinator deployment..."
    kubectl wait --for=condition=available --timeout=600s deployment/protein-operators-coordinator -n "$NAMESPACE"
    
    log_success "All deployments are ready"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping post-deployment tests as requested"
        return 0
    fi
    
    log_info "Running post-deployment tests..."
    
    # Get the API service endpoint
    local api_service_ip
    api_service_ip=$(kubectl get service protein-operators-api-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test API health endpoint
    log_info "Testing API health endpoint..."
    kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f "http://$api_service_ip/health" || {
        log_error "API health check failed"
        return 1
    }
    
    # Test API ready endpoint
    log_info "Testing API ready endpoint..."
    kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f "http://$api_service_ip/ready" || {
        log_error "API readiness check failed"
        return 1
    }
    
    log_success "Post-deployment tests passed"
}

# Function to setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Deploy Prometheus if not exists
    if ! kubectl get deployment prometheus -n monitoring &> /dev/null; then
        log_info "Deploying Prometheus..."
        
        # Create monitoring namespace
        kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
        
        # Apply monitoring stack
        kubectl apply -f monitoring/ -n monitoring || log_warning "Failed to apply monitoring stack"
    else
        log_info "Prometheus already deployed"
    fi
    
    log_success "Monitoring setup complete"
}

# Function to display deployment status
show_deployment_status() {
    log_info "Deployment Status:"
    
    echo -e "\n${BLUE}Namespace:${NC} $NAMESPACE"
    echo -e "${BLUE}Image:${NC} $CONTAINER_REGISTRY/$PROJECT_NAME:$IMAGE_TAG"
    
    echo -e "\n${BLUE}Deployments:${NC}"
    kubectl get deployments -n "$NAMESPACE" -o wide
    
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n "$NAMESPACE" -o wide
    
    echo -e "\n${BLUE}Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n "$NAMESPACE" -o wide
    
    echo -e "\n${BLUE}HPA:${NC}"
    kubectl get hpa -n "$NAMESPACE" -o wide
    
    # Show resource usage
    echo -e "\n${BLUE}Resource Usage:${NC}"
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || log_warning "Metrics server not available"
}

# Function to show access information
show_access_info() {
    log_info "Access Information:"
    
    # Get ingress info
    local ingress_ip
    ingress_ip=$(kubectl get ingress protein-operators-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "<pending>")
    
    echo -e "\n${GREEN}API Endpoints:${NC}"
    echo -e "  External: https://api.protein-operators.com (IP: $ingress_ip)"
    echo -e "  Internal: http://protein-operators-api-service.$NAMESPACE.svc.cluster.local"
    
    echo -e "\n${GREEN}Monitoring:${NC}"
    echo -e "  Prometheus: http://prometheus.monitoring.svc.cluster.local:9090"
    echo -e "  Grafana: http://grafana.monitoring.svc.cluster.local:3000"
    
    echo -e "\n${GREEN}Useful Commands:${NC}"
    echo -e "  View logs: kubectl logs -f deployment/protein-operators-api -n $NAMESPACE"
    echo -e "  Scale API: kubectl scale deployment protein-operators-api --replicas=5 -n $NAMESPACE"
    echo -e "  Port forward: kubectl port-forward service/protein-operators-api-service 8080:80 -n $NAMESPACE"
    echo -e "  Shell access: kubectl exec -it deployment/protein-operators-api -n $NAMESPACE -- /bin/bash"
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed. Cleaning up..."
    
    # Optional: Rollback to previous version
    if [[ "${ROLLBACK_ON_FAILURE:-false}" == "true" ]]; then
        log_info "Rolling back deployments..."
        kubectl rollout undo deployment/protein-operators-api -n "$NAMESPACE" || true
        kubectl rollout undo deployment/protein-operators-worker -n "$NAMESPACE" || true
        kubectl rollout undo deployment/protein-operators-coordinator -n "$NAMESPACE" || true
    fi
    
    # Show failure diagnostics
    echo -e "\n${RED}Failure Diagnostics:${NC}"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
    
    exit 1
}

# Main deployment function
main() {
    echo -e "${GREEN}"
    echo "================================================================="
    echo "   🚀 Autonomous SDLC Production Deployment"
    echo "   🧬 Protein Operators System"
    echo "================================================================="
    echo -e "${NC}"
    
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  Project: $PROJECT_NAME"
    echo -e "  Namespace: $NAMESPACE"
    echo -e "  Image Tag: $IMAGE_TAG"
    echo -e "  Registry: $CONTAINER_REGISTRY"
    echo -e "  Dry Run: $DRY_RUN"
    echo -e "  Skip Tests: $SKIP_TESTS"
    echo -e "  Skip Build: $SKIP_BUILD"
    echo ""
    
    # Confirm deployment
    if [[ "${CONFIRM_DEPLOY:-true}" == "true" && "$DRY_RUN" != "true" ]]; then
        read -p "Do you want to proceed with production deployment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Execute deployment steps
    check_prerequisites
    run_quality_gates
    build_and_push_image
    create_namespace
    setup_secrets
    deploy_infrastructure
    wait_for_infrastructure
    deploy_application
    wait_for_deployment
    run_post_deployment_tests
    setup_monitoring
    
    # Show results
    show_deployment_status
    show_access_info
    
    log_success "🎉 Production deployment completed successfully!"
    
    echo -e "\n${GREEN}Next Steps:${NC}"
    echo -e "  1. Configure DNS to point to the ingress IP"
    echo -e "  2. Set up SSL certificates (cert-manager)"
    echo -e "  3. Configure monitoring alerts"
    echo -e "  4. Set up backup procedures"
    echo -e "  5. Review and update security policies"
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            CONTAINER_REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Run deployment in dry-run mode"
            echo "  --skip-tests      Skip quality gates and post-deployment tests"
            echo "  --skip-build      Skip Docker image build and push"
            echo "  --image-tag TAG   Docker image tag to deploy (default: latest)"
            echo "  --registry REG    Container registry URL"
            echo "  --namespace NS    Kubernetes namespace (default: protein-operators-prod)"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  IMAGE_TAG         Docker image tag"
            echo "  CONTAINER_REGISTRY Container registry URL"
            echo "  KUBECONFIG        Path to kubeconfig file"
            echo "  DRY_RUN           Run in dry-run mode (true/false)"
            echo "  SKIP_TESTS        Skip tests (true/false)"
            echo "  SKIP_BUILD        Skip build (true/false)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function with error handling
trap cleanup_on_failure ERR
main
