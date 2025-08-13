#!/bin/bash

# Production Deployment Script for Protein Operators
# Usage: ./deploy.sh [environment] [version]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENVIRONMENT="${1:-production}"
VERSION="${2:-latest}"
NAMESPACE="protein-operators"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-docker.io/terragon}"
IMAGE_NAME="protein-operators"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    # Check kubectl cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "${PROJECT_ROOT}"
    
    # Build production image
    docker build -t "${IMAGE_REGISTRY}/${IMAGE_NAME}:${VERSION}" \
                 -t "${IMAGE_REGISTRY}/${IMAGE_NAME}:latest" \
                 --target production \
                 .
    
    # Build GPU image
    docker build -t "${IMAGE_REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}" \
                 -t "${IMAGE_REGISTRY}/${IMAGE_NAME}:gpu-latest" \
                 --target gpu \
                 .
    
    log_success "Docker images built successfully"
}

# Push Docker image
push_image() {
    log_info "Pushing Docker images to registry..."
    
    docker push "${IMAGE_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker push "${IMAGE_REGISTRY}/${IMAGE_NAME}:latest"
    docker push "${IMAGE_REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"
    docker push "${IMAGE_REGISTRY}/${IMAGE_NAME}:gpu-latest"
    
    log_success "Docker images pushed successfully"
}

# Create namespace if not exists
create_namespace() {
    log_info "Creating namespace if not exists..."
    
    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_info "Namespace ${NAMESPACE} already exists"
    else
        kubectl apply -f "${PROJECT_ROOT}/k8s/namespace.yaml"
        log_success "Namespace ${NAMESPACE} created"
    fi
}

# Apply Kubernetes manifests
apply_manifests() {
    log_info "Applying Kubernetes manifests..."
    
    cd "${PROJECT_ROOT}"
    
    # Apply in order
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/rbac.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/pvc.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/protein-operators-models -n "${NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=Bound pvc/protein-operators-logs -n "${NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=Bound pvc/protein-operators-data -n "${NAMESPACE}" --timeout=300s
    
    # Apply services
    kubectl apply -f k8s/service.yaml
    
    # Apply deployments
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/gpu-deployment.yaml
    
    # Apply HPA
    kubectl apply -f k8s/hpa.yaml
    
    # Apply ingress
    kubectl apply -f k8s/ingress.yaml
    
    log_success "Kubernetes manifests applied successfully"
}

# Wait for deployments to be ready
wait_for_deployment() {
    log_info "Waiting for deployments to be ready..."
    
    kubectl rollout status deployment/protein-operators -n "${NAMESPACE}" --timeout=600s
    kubectl rollout status deployment/protein-operators-gpu -n "${NAMESPACE}" --timeout=600s
    
    log_success "Deployments are ready"
}

# Run health checks
health_check() {
    log_info "Running health checks..."
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service protein-operators-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "${service_ip}" ]]; then
        log_warning "LoadBalancer IP not yet assigned, using port-forward for health check"
        kubectl port-forward service/protein-operators-service 8080:80 -n "${NAMESPACE}" &
        local port_forward_pid=$!
        sleep 5
        service_ip="localhost:8080"
    fi
    
    # Health check
    local retry_count=0
    local max_retries=30
    
    while [[ ${retry_count} -lt ${max_retries} ]]; do
        if curl -f "http://${service_ip}/health" &> /dev/null; then
            log_success "Health check passed"
            if [[ -n "${port_forward_pid:-}" ]]; then
                kill "${port_forward_pid}" 2>/dev/null || true
            fi
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        log_info "Health check failed (attempt ${retry_count}/${max_retries}), retrying..."
        sleep 10
    done
    
    log_error "Health check failed after ${max_retries} attempts"
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill "${port_forward_pid}" 2>/dev/null || true
    fi
    return 1
}

# Cleanup function
cleanup() {
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill "${port_forward_pid}" 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Main deployment function
main() {
    log_info "Starting deployment of Protein Operators ${VERSION} to ${ENVIRONMENT}..."
    
    check_prerequisites
    build_image
    push_image
    create_namespace
    apply_manifests
    wait_for_deployment
    health_check
    
    log_success "Deployment completed successfully!"
    log_info "Application is available at the LoadBalancer IP"
    
    # Display useful information
    echo
    log_info "Useful commands:"
    echo "  View pods: kubectl get pods -n ${NAMESPACE}"
    echo "  View services: kubectl get services -n ${NAMESPACE}"
    echo "  View logs: kubectl logs -f deployment/protein-operators -n ${NAMESPACE}"
    echo "  Scale deployment: kubectl scale deployment protein-operators --replicas=5 -n ${NAMESPACE}"
}

# Run main function
main "$@"