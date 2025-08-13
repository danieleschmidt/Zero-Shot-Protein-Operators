#!/bin/bash

# Deploy Monitoring Stack for Protein Operators
# Usage: ./deploy-monitoring.sh

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="protein-operators"

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
        log_error "Namespace ${NAMESPACE} does not exist. Please deploy the application first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy monitoring components
deploy_rbac() {
    log_info "Deploying RBAC for monitoring..."
    kubectl apply -f "${SCRIPT_DIR}/rbac.yaml"
    log_success "RBAC deployed"
}

deploy_prometheus() {
    log_info "Deploying Prometheus..."
    kubectl apply -f "${SCRIPT_DIR}/prometheus.yaml"
    
    log_info "Waiting for Prometheus to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n "${NAMESPACE}"
    log_success "Prometheus deployed and ready"
}

deploy_alertmanager() {
    log_info "Deploying Alertmanager..."
    kubectl apply -f "${SCRIPT_DIR}/alertmanager.yaml"
    
    log_info "Waiting for Alertmanager to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/alertmanager -n "${NAMESPACE}"
    log_success "Alertmanager deployed and ready"
}

deploy_grafana() {
    log_info "Deploying Grafana..."
    kubectl apply -f "${SCRIPT_DIR}/grafana.yaml"
    
    log_info "Waiting for Grafana to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n "${NAMESPACE}"
    log_success "Grafana deployed and ready"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying monitoring deployment..."
    
    # Check if all services are running
    local prometheus_status
    local alertmanager_status
    local grafana_status
    
    prometheus_status=$(kubectl get deployment prometheus -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
    alertmanager_status=$(kubectl get deployment alertmanager -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
    grafana_status=$(kubectl get deployment grafana -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
    
    if [[ "${prometheus_status}" -ge 1 && "${alertmanager_status}" -ge 1 && "${grafana_status}" -ge 1 ]]; then
        log_success "All monitoring components are running"
        return 0
    else
        log_error "Some monitoring components are not ready"
        return 1
    fi
}

# Show access information
show_access_info() {
    log_info "Monitoring stack deployed successfully!"
    echo
    log_info "Access Information:"
    echo
    
    # Prometheus
    echo "Prometheus:"
    echo "  URL: http://localhost:9090"
    echo "  Command: kubectl port-forward service/prometheus 9090:9090 -n ${NAMESPACE}"
    echo
    
    # Alertmanager
    echo "Alertmanager:"
    echo "  URL: http://localhost:9093"
    echo "  Command: kubectl port-forward service/alertmanager 9093:9093 -n ${NAMESPACE}"
    echo
    
    # Grafana
    echo "Grafana:"
    echo "  URL: http://localhost:3000"
    echo "  Command: kubectl port-forward service/grafana 3000:3000 -n ${NAMESPACE}"
    echo "  Username: admin"
    echo "  Password: password (change this!)"
    echo
    
    log_info "To access services, run the port-forward commands above in separate terminals."
    log_warning "Remember to change the default Grafana password!"
}

# Main deployment function
main() {
    log_info "Starting monitoring stack deployment..."
    
    check_prerequisites
    deploy_rbac
    deploy_prometheus
    deploy_alertmanager
    deploy_grafana
    verify_deployment
    show_access_info
    
    log_success "Monitoring stack deployment completed!"
}

# Show usage if help requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo
    echo "Deploy monitoring stack (Prometheus, Alertmanager, Grafana) for Protein Operators."
    echo
    echo "Prerequisites:"
    echo "  - kubectl installed and configured"
    echo "  - Protein Operators application already deployed"
    echo "  - Sufficient cluster resources"
    echo
    echo "Components deployed:"
    echo "  - Prometheus (metrics collection)"
    echo "  - Alertmanager (alerting)"
    echo "  - Grafana (visualization)"
    echo
    exit 0
fi

# Run main function
main "$@"