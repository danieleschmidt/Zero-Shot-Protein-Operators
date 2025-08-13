#!/bin/bash

# Scaling Script for Protein Operators
# Usage: ./scale.sh [replicas] [deployment]

set -euo pipefail

# Configuration
NAMESPACE="protein-operators"
REPLICAS="${1:-3}"
DEPLOYMENT="${2:-protein-operators}"

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

# Validate inputs
validate_inputs() {
    if ! [[ "${REPLICAS}" =~ ^[0-9]+$ ]]; then
        log_error "Replicas must be a positive integer"
        exit 1
    fi
    
    if [[ "${REPLICAS}" -gt 20 ]]; then
        log_warning "Scaling to more than 20 replicas. This may consume significant resources."
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Scaling cancelled"
            exit 0
        fi
    fi
    
    if [[ "${DEPLOYMENT}" != "protein-operators" && "${DEPLOYMENT}" != "protein-operators-gpu" ]]; then
        log_error "Invalid deployment name. Must be 'protein-operators' or 'protein-operators-gpu'"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Namespace ${NAMESPACE} does not exist."
        exit 1
    fi
    
    if ! kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" &> /dev/null; then
        log_error "Deployment ${DEPLOYMENT} does not exist in namespace ${NAMESPACE}."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Show current status
show_current_status() {
    log_info "Current deployment status:"
    kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" -o wide
    
    echo
    log_info "Current pods:"
    kubectl get pods -n "${NAMESPACE}" -l app="${DEPLOYMENT}" -o wide
}

# Scale deployment
scale_deployment() {
    log_info "Scaling ${DEPLOYMENT} to ${REPLICAS} replicas..."
    
    kubectl scale deployment "${DEPLOYMENT}" --replicas="${REPLICAS}" -n "${NAMESPACE}"
    
    log_info "Waiting for scaling to complete..."
    kubectl rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=600s
    
    log_success "Scaling completed successfully"
}

# Verify scaling
verify_scaling() {
    log_info "Verifying scaling..."
    
    local current_replicas
    current_replicas=$(kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
    
    if [[ "${current_replicas}" -eq "${REPLICAS}" ]]; then
        log_success "Scaling verified: ${current_replicas} replicas are ready"
    else
        log_warning "Scaling may not be complete: ${current_replicas}/${REPLICAS} replicas are ready"
    fi
    
    # Show final status
    echo
    log_info "Final deployment status:"
    kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" -o wide
}

# Show resource usage
show_resource_usage() {
    log_info "Resource usage after scaling:"
    
    # Show node resource usage
    kubectl top nodes 2>/dev/null || log_warning "Metrics server not available for node resource usage"
    
    echo
    # Show pod resource usage
    kubectl top pods -n "${NAMESPACE}" -l app="${DEPLOYMENT}" 2>/dev/null || log_warning "Metrics server not available for pod resource usage"
}

# Main scaling function
main() {
    log_info "Starting scaling operation for ${DEPLOYMENT} to ${REPLICAS} replicas..."
    
    validate_inputs
    check_prerequisites
    
    # Show current status
    show_current_status
    
    # Confirm scaling
    echo
    read -p "Scale ${DEPLOYMENT} to ${REPLICAS} replicas? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Scaling cancelled"
        exit 0
    fi
    
    scale_deployment
    verify_scaling
    show_resource_usage
    
    log_success "Scaling operation completed successfully!"
    
    # Display useful information
    echo
    log_info "Useful commands:"
    echo "  View pods: kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYMENT}"
    echo "  View logs: kubectl logs -f deployment/${DEPLOYMENT} -n ${NAMESPACE}"
    echo "  Scale again: kubectl scale deployment ${DEPLOYMENT} --replicas=<number> -n ${NAMESPACE}"
}

# Show usage if help requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [replicas] [deployment]"
    echo
    echo "Scale Protein Operators deployment to specified number of replicas."
    echo
    echo "Options:"
    echo "  replicas     Number of replicas to scale to (default: 3)"
    echo "  deployment   Deployment name: protein-operators or protein-operators-gpu (default: protein-operators)"
    echo "  -h, --help   Show this help message"
    echo
    echo "Examples:"
    echo "  $0 5                           # Scale protein-operators to 5 replicas"
    echo "  $0 2 protein-operators-gpu     # Scale GPU deployment to 2 replicas"
    echo "  $0 0 protein-operators         # Scale down to 0 (maintenance mode)"
    exit 0
fi

# Run main function
main "$@"