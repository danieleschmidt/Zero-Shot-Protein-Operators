# Zero-Shot Protein Operators - Production Deployment Guide

This guide provides comprehensive instructions for deploying the Zero-Shot Protein Operators system in production environments.

## 🏗️ Architecture Overview

The system is designed for scalable, production-ready deployment with the following components:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Gateway │────│  Protein Design │
│                 │    │               │    │    Services     │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                       │                    │
         ▼                       ▼                    ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Caching     │    │   GPU Compute   │
│   & Alerting    │    │   Layer       │    │    Cluster     │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

1. **Clone and setup environment:**
```bash
git clone https://github.com/danieleschmidt/Zero-Shot-Protein-Operators
cd Zero-Shot-Protein-Operators
cp .env.example .env  # Configure environment variables
```

2. **Start production stack:**
```bash
# Build and start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose -f docker-compose.prod.yml ps
```

3. **Verify deployment:**
```bash
# Test API endpoint
curl http://localhost:8000/health

# Check monitoring dashboard
open http://localhost:3000  # Grafana dashboard
```

### Container Images

Our production images are optimized for performance and security:

- **Base Image**: `python:3.11-slim` (security updates, minimal attack surface)
- **GPU Image**: `nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04` (GPU acceleration)
- **Size**: ~1.2GB base, ~3.5GB GPU variant
- **Security**: Non-root user, minimal dependencies, security scanning

### Environment Configuration

Create a `.env` file with production settings:

```env
# Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://user:password@postgres:5432/proteindb
REDIS_URL=redis://redis:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-256-bit-secret-key-here
JWT_SECRET=your-jwt-secret-here
ALLOWED_HOSTS=your-domain.com,api.your-domain.com

# Performance
MAX_WORKERS=8
CACHE_SIZE=10000
CACHE_MEMORY_MB=2048

# GPU Configuration (if available)
ENABLE_GPU=true
GPU_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1

# Monitoring
MONITORING_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Auto-scaling
AUTO_SCALING_ENABLED=true
MIN_WORKERS=2
MAX_WORKERS=16
SCALE_UP_THRESHOLD=0.8
SCALE_DOWN_THRESHOLD=0.2
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Our CI/CD pipeline includes:

1. **Code Quality Gates**
   - Linting and formatting
   - Security vulnerability scanning
   - Unit and integration tests
   - Performance benchmarks

2. **Build and Test**
   - Multi-stage Docker builds
   - Image security scanning
   - Integration testing

3. **Deployment**
   - Blue-green deployments
   - Health checks
   - Rollback capabilities

### Pipeline Configuration

```yaml
# .github/workflows/deploy.yml
name: Production Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Run Quality Gates
        run: |
          python scripts/comprehensive_quality_gates.py --strict
      
      - name: Upload Quality Report
        uses: actions/upload-artifact@v3
        with:
          name: quality-gates-report
          path: quality_gates_report.json

  build-and-deploy:
    needs: quality-gates
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Production Image
        run: |
          docker build --target production -t protein-operators:latest .
      
      - name: Security Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'protein-operators:latest'
      
      - name: Deploy to Production
        run: |
          # Deployment logic here
          echo "Deploying to production..."
```

## ☸️ Kubernetes Deployment

### Namespace and Resources

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: protein-operators
  labels:
    app: protein-operators
    environment: production
```

### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: protein-operators-api
  namespace: protein-operators
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: protein-operators-api
  template:
    metadata:
      labels:
        app: protein-operators-api
    spec:
      containers:
      - name: api
        image: protein-operators:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090  # Metrics
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: protein-operators-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
```

### GPU Workloads

```yaml
# k8s/gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: protein-operators-gpu
  namespace: protein-operators
spec:
  replicas: 2
  selector:
    matchLabels:
      app: protein-operators-gpu
  template:
    metadata:
      labels:
        app: protein-operators-gpu
    spec:
      containers:
      - name: gpu-worker
        image: protein-operators:gpu-latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2000m"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4000m"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: protein-operators-service
  namespace: protein-operators
spec:
  selector:
    app: protein-operators-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: protein-operators-ingress
  namespace: protein-operators
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.protein-operators.org
    secretName: protein-operators-tls
  rules:
  - host: api.protein-operators.org
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: protein-operators-service
            port:
              number: 80
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'protein-operators'
    static_configs:
      - targets: ['protein-operators-service:9090']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: protein_operators_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(protein_operators_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighMemoryUsage
        expr: protein_operators_memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      - alert: GPUUtilizationLow
        expr: protein_operators_gpu_utilization < 10
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Low GPU utilization"
          description: "GPU utilization is {{ $value }}%"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Protein Operators Production",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(protein_operators_requests_total[5m])",
            "legend": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, protein_operators_response_time_histogram)",
            "legend": "95th percentile"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "protein_operators_cpu_usage_percent",
            "legend": "CPU %"
          },
          {
            "expr": "protein_operators_memory_usage_percent", 
            "legend": "Memory %"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### Network Security

1. **TLS Encryption**
   - All API endpoints use HTTPS
   - TLS 1.3 minimum
   - Certificate auto-renewal with cert-manager

2. **Network Policies**
   - Restricted inter-pod communication
   - Egress filtering
   - No direct internet access for workers

3. **API Security**
   - JWT authentication
   - Rate limiting (100 req/min per client)
   - Input validation and sanitization
   - CORS configuration

### Container Security

```dockerfile
# Security-optimized Dockerfile
FROM python:3.11-slim as base

# Create non-root user
RUN groupadd -r protein && useradd -r -g protein protein

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application
COPY --chown=protein:protein src/ /app/src/
COPY --chown=protein:protein requirements.txt /app/

# Install dependencies
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Security: run as non-root
USER protein

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000
CMD ["python", "-m", "protein_operators.api.app"]
```

### Secrets Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: protein-operators-secrets
  namespace: protein-operators
type: Opaque
stringData:
  database-url: postgresql://user:password@postgres:5432/proteindb
  jwt-secret: your-jwt-secret-here
  api-key: your-api-key-here
```

## 🚀 Performance Optimization

### Auto-scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: protein-operators-hpa
  namespace: protein-operators
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: protein-operators-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Caching Strategy

1. **Redis Configuration**
   ```yaml
   # Redis cluster for high availability
   apiVersion: apps/v1
   kind: StatefulSet
   metadata:
     name: redis-cluster
   spec:
     serviceName: redis-headless
     replicas: 6
     template:
       spec:
         containers:
         - name: redis
           image: redis:7-alpine
           resources:
             requests:
               memory: "1Gi"
               cpu: "500m"
             limits:
               memory: "2Gi"
               cpu: "1000m"
   ```

2. **Cache Policies**
   - Protein structures: 1 hour TTL
   - Constraint encodings: 30 minutes TTL
   - Validation results: 15 minutes TTL
   - Performance metrics: 5 minutes TTL

## 📈 Load Testing and Capacity Planning

### Load Testing with Artillery

```yaml
# load-test/config.yml
config:
  target: 'https://api.protein-operators.org'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 300
      arrivalRate: 50
      name: "Sustained load"
    - duration: 120
      arrivalRate: 100
      name: "Peak load"

scenarios:
  - name: "Protein Design API"
    weight: 70
    flow:
      - get:
          url: "/health"
      - post:
          url: "/api/v1/design"
          json:
            constraints:
              binding_sites:
                - residues: [10, 20, 30]
                  ligand: "ATP"
              length: 150
            operator_type: "deeponet"

  - name: "Validation API"
    weight: 30
    flow:
      - post:
          url: "/api/v1/validate"
          json:
            structure_data: "sample_structure"
            validation_level: "standard"
```

### Capacity Planning

**Recommended Production Specs:**

- **Small Deployment (< 100 req/min)**
  - 3x API pods: 2 CPU, 4GB RAM each
  - 1x GPU worker: 4 CPU, 16GB RAM, 1x V100
  - 1x Redis: 2 CPU, 4GB RAM
  - 1x PostgreSQL: 2 CPU, 8GB RAM

- **Medium Deployment (100-1000 req/min)**
  - 8x API pods: 2 CPU, 4GB RAM each
  - 3x GPU workers: 4 CPU, 16GB RAM, 1x V100 each
  - 3x Redis cluster: 2 CPU, 4GB RAM each
  - 1x PostgreSQL (HA): 4 CPU, 16GB RAM

- **Large Deployment (> 1000 req/min)**
  - 20x API pods: 2 CPU, 4GB RAM each
  - 8x GPU workers: 8 CPU, 32GB RAM, 2x V100 each
  - 6x Redis cluster: 4 CPU, 8GB RAM each
  - PostgreSQL cluster: 8 CPU, 32GB RAM

## 🔧 Troubleshooting Guide

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage per pod
   kubectl top pods -n protein-operators
   
   # Increase memory limits
   kubectl patch deployment protein-operators-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```

2. **GPU Not Detected**
   ```bash
   # Verify GPU nodes
   kubectl get nodes -l accelerator=nvidia-tesla-v100
   
   # Check GPU device plugin
   kubectl get pods -n kube-system | grep nvidia-device-plugin
   ```

3. **Database Connection Issues**
   ```bash
   # Check database connectivity
   kubectl exec -it deployment/protein-operators-api -- python -c "
   import os
   import psycopg2
   conn = psycopg2.connect(os.environ['DATABASE_URL'])
   print('Database connection successful')
   "
   ```

### Performance Debugging

1. **Enable Debug Logging**
   ```bash
   kubectl set env deployment/protein-operators-api LOG_LEVEL=DEBUG
   ```

2. **Profile API Endpoints**
   ```bash
   # Install profiling tools
   kubectl exec -it deployment/protein-operators-api -- pip install py-spy
   
   # Profile running process
   kubectl exec -it deployment/protein-operators-api -- py-spy record -o profile.svg -d 60 -p 1
   ```

3. **Monitor Resource Usage**
   ```bash
   # Watch resource usage
   watch kubectl top pods -n protein-operators
   
   # Get detailed metrics
   kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/protein-operators/pods
   ```

## 📋 Production Checklist

Before going live, ensure:

- [ ] All quality gates pass
- [ ] Security scan completed with no critical vulnerabilities
- [ ] Load testing completed successfully
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] Documentation updated
- [ ] Team trained on production procedures
- [ ] On-call rotation established
- [ ] Incident response procedures defined
- [ ] Performance baselines established

## 📞 Support

For production deployment support:

- **Documentation**: https://protein-operators.readthedocs.io
- **Issues**: https://github.com/danieleschmidt/Zero-Shot-Protein-Operators/issues  
- **Security**: security@protein-operators.org
- **Status Page**: https://status.protein-operators.org

---

*This deployment guide is maintained by the Protein Operators team and updated regularly with best practices and lessons learned from production deployments.*