# Deployment Guide for Zero-Shot Protein-Operators

## CI/CD Setup Instructions

Due to GitHub App permissions, the CI/CD workflow file cannot be automatically created. Please follow these steps to set up the complete CI/CD pipeline:

### 1. Create the GitHub Actions Workflow

1. **Copy the workflow template**:
   ```bash
   mkdir -p .github/workflows
   cp ci-cd-template.yml .github/workflows/ci.yml
   ```

2. **Commit the workflow file manually**:
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "feat(ci): add comprehensive CI/CD pipeline"
   git push
   ```

### 2. Configure Repository Secrets

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

#### Required Secrets:
- `CODECOV_TOKEN`: For code coverage reporting
- `PYPI_API_TOKEN`: For package publishing to PyPI
- `SLACK_WEBHOOK_URL`: For build notifications (optional)

#### AWS Secrets (for deployment):
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

### 3. Enable GitHub Actions Permissions

1. Go to Settings → Actions → General
2. Set "Workflow permissions" to "Read and write permissions"
3. Check "Allow GitHub Actions to create and approve pull requests"

### 4. Quality Gates Configuration

The quality gates are automatically enforced via the `scripts/quality_gates.py` script with these thresholds:

- **Test Coverage**: ≥85%
- **Code Quality**: Black formatting, Flake8 linting, MyPy type checking
- **Security**: Zero high vulnerabilities, ≤5 medium vulnerabilities
- **Performance**: Tests complete within 300 seconds
- **Documentation**: ≥80% docstring coverage

### 5. Local Development Setup

```bash
# Install development dependencies
pip install -e ".[dev,experiments]"

# Install pre-commit hooks (optional)
pre-commit install

# Run quality gates locally
python scripts/quality_gates.py

# Run specific quality gate
python scripts/quality_gates.py --gate coverage
```

### 6. Performance Monitoring

The system includes comprehensive performance monitoring:

- **Real-time metrics**: CPU, memory, GPU utilization
- **Performance benchmarks**: Automated regression testing
- **Optimization recommendations**: Automatic performance tuning

To enable performance monitoring:

```python
from protein_operators.utils.performance import configure_performance_monitoring

# Configure monitoring
monitor = configure_performance_monitoring(
    collect_system_metrics=True,
    collect_gpu_metrics=True,
    auto_optimize=True
)
```

### 7. Production Deployment

#### Docker Deployment:
```bash
# Build production image
docker build --target production -t protein-operators:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 protein-operators:latest
```

#### Environment Variables:
```bash
# Core settings
export PROTEIN_OPERATORS_LOG_LEVEL=INFO
export PROTEIN_OPERATORS_CACHE_TYPE=redis
export PROTEIN_OPERATORS_GPU_ENABLED=true

# Database
export DATABASE_URL=postgresql://user:pass@host:5432/protein_operators

# API settings
export API_PORT=8000
export API_HOST=0.0.0.0
export API_WORKERS=4
```

### 8. Monitoring and Observability

#### Health Checks:
- API: `GET /health`
- System metrics: Available via performance monitor
- Cache statistics: Available via cache manager

#### Logging:
- Structured JSON logging in production
- Colored console logging in development
- Automatic log rotation and cleanup

#### Metrics Collection:
```python
from protein_operators.utils.performance import record_metric

# Record custom metrics
record_metric("design_success_rate", 0.95, "ratio")
record_metric("processing_time", 12.5, "seconds")
```

### 9. Testing Strategy

The project includes comprehensive testing:

- **Unit Tests**: `tests/unit/` - Fast, isolated tests
- **Integration Tests**: `tests/integration/` - End-to-end workflows  
- **Performance Tests**: `tests/benchmarks/` - Performance regression testing
- **API Tests**: Automated API endpoint testing

Run tests:
```bash
# All tests
pytest

# Specific test categories
pytest tests/unit -v
pytest tests/integration -v  
pytest tests/benchmarks --benchmark-only
```

### 10. Troubleshooting

#### Common Issues:

1. **GitHub Actions Workflow Not Running**:
   - Ensure workflow file is in `.github/workflows/ci.yml`
   - Check repository permissions and secrets

2. **GPU Tests Failing**:
   - Verify CUDA installation and GPU availability
   - Check PyTorch CUDA compatibility

3. **Cache Issues**:
   - Clear cache: `python -c "from protein_operators.cache import get_cache_manager; get_cache_manager().clear()"`
   - Check disk space for file-based caching

4. **Performance Regression**:
   - Review performance benchmarks in CI/CD reports
   - Check system resource usage
   - Verify caching effectiveness

### 11. Scaling Considerations

#### Horizontal Scaling:
- Deploy multiple API instances behind load balancer
- Use Redis for distributed caching
- Implement database connection pooling

#### Vertical Scaling:
- GPU acceleration for neural operator inference
- Memory optimization for large protein structures
- Multi-threading for concurrent operations

#### Resource Requirements:
- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8+ CPU cores, 16GB+ RAM, GPU with 8GB+ VRAM
- **Storage**: SSD recommended for caching and database

---

## Success Metrics

After deployment, monitor these key metrics:

✅ **API Response Time**: <200ms for design requests  
✅ **System Uptime**: >99.9% availability  
✅ **Test Coverage**: >85% maintained  
✅ **Security**: Zero high vulnerabilities  
✅ **Performance**: No regressions in benchmarks  

The Zero-Shot Protein-Operators system is now production-ready with enterprise-grade CI/CD, monitoring, and deployment capabilities.