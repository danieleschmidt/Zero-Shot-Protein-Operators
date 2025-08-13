#!/bin/bash

# Rollback Script for Protein Operators
# Usage: ./rollback.sh [revision]

set -euo pipefail

# Configuration
NAMESPACE="protein-operators"
REVISION="${1:-}"

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
    
    log_success "Prerequisites check passed"
}

# Show rollout history
show_history() {
    log_info "Deployment history for protein-operators:"
    kubectl rollout history deployment/protein-operators -n "${NAMESPACE}"
    
    echo
    log_info "Deployment history for protein-operators-gpu:"
    kubectl rollout history deployment/protein-operators-gpu -n "${NAMESPACE}"
}

# Perform rollback
perform_rollback() {
    local revision_flag=""
    
    if [[ -n "${REVISION}" ]]; then
        revision_flag="--to-revision=${REVISION}"
        log_info "Rolling back to revision ${REVISION}..."
    else
        log_info "Rolling back to previous revision..."
    fi
    
    # Rollback main deployment
    log_info "Rolling back protein-operators deployment..."
    kubectl rollout undo deployment/protein-operators ${revision_flag} -n "${NAMESPACE}"
    
    # Rollback GPU deployment
    log_info "Rolling back protein-operators-gpu deployment..."
    kubectl rollout undo deployment/protein-operators-gpu ${revision_flag} -n "${NAMESPACE}"
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/protein-operators -n "${NAMESPACE}" --timeout=600s
    kubectl rollout status deployment/protein-operators-gpu -n "${NAMESPACE}" --timeout=600s
    
    log_success "Rollback completed successfully"
}

# Verify rollback
verify_rollback() {
    log_info "Verifying rollback..."
    
    # Check pod status
    local ready_pods
    ready_pods=$(kubectl get pods -n "${NAMESPACE}" -l app=protein-operators --field-selector=status.phase=Running | grep -c Running || true)
    
    if [[ ${ready_pods} -gt 0 ]]; then
        log_success "Rollback verified: ${ready_pods} pods are running"
    else
        log_error "Rollback verification failed: No running pods found"
        return 1
    fi
    
    # Basic health check
    log_info "Performing basic health check..."
    if kubectl exec -n "${NAMESPACE}" deployment/protein-operators -- python -c "import sys; sys.path.insert(0, 'src'); from protein_operators import ProteinDesigner; print('Health check passed')" &> /dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed - application may still be starting"
    fi
}

# Main rollback function
main() {
    if [[ -n "${REVISION}" ]]; then
        log_info "Starting rollback to revision ${REVISION}..."
    else
        log_info "Starting rollback to previous revision..."
    fi
    
    check_prerequisites
    
    # Show current history
    show_history
    
    # Confirm rollback
    echo
    read -p "Are you sure you want to proceed with the rollback? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    perform_rollback
    verify_rollback
    
    log_success "Rollback completed successfully!"
    
    # Display useful information
    echo
    log_info "Useful commands:"
    echo "  View pods: kubectl get pods -n ${NAMESPACE}"
    echo "  View logs: kubectl logs -f deployment/protein-operators -n ${NAMESPACE}"
    echo "  Check history: kubectl rollout history deployment/protein-operators -n ${NAMESPACE}"
}

# Show usage if help requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [revision]"
    echo
    echo "Rollback Protein Operators deployment to a previous revision."
    echo
    echo "Options:"
    echo "  revision    Specific revision number to rollback to (optional)"
    echo "  -h, --help  Show this help message"
    echo
    echo "Examples:"
    echo "  $0           # Rollback to previous revision"
    echo "  $0 3         # Rollback to revision 3"
    exit 0
fi

# Run main function
main "$@"