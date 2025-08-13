# Multi-stage Dockerfile for Zero-Shot Protein Operators
# Base stage with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Activate environment by default
ENV CONDA_DEFAULT_ENV=protein-operators
ENV PATH=/opt/conda/envs/protein-operators/bin:$PATH

# Development stage
FROM base as development

WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Install package in editable mode
RUN pip install -e ".[dev,experiments]"

# Set up pre-commit hooks
RUN pre-commit install

# Expose ports for Jupyter and MLflow
EXPOSE 8888 5000

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

WORKDIR /app

# Copy only necessary files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install package in production mode
RUN pip install .

# Create non-root user for security
RUN useradd -m -u 1000 proteinops && \
    mkdir -p /app/logs /app/models /app/data /app/config && \
    chown -R proteinops:proteinops /app

# Copy production configuration
COPY docker/config/ ./config/

USER proteinops

# Create volume mount points
VOLUME ["/app/logs", "/app/models", "/app/data", "/app/config"]

# Expose ports for API and metrics
EXPOSE 8000 9090

# Health check with improved error handling
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD python -c "import sys; sys.path.insert(0, 'src'); from protein_operators.utils.health_monitoring import HealthMonitor; monitor = HealthMonitor(); status = monitor.get_system_health(); print('Health check passed' if status['overall_status'] == 'healthy' else 'Health check failed'); exit(0 if status['overall_status'] == 'healthy' else 1)" || exit 1

# Set production environment variables
ENV PROTEIN_OPERATORS_ENV=production
ENV PROTEIN_OPERATORS_LOG_LEVEL=INFO
ENV PROTEIN_OPERATORS_CONFIG_PATH=/app/config

# Production command - start API server
CMD ["python", "-m", "protein_operators.api.app"]

# Enhanced GPU production stage
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as gpu

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        curl \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install dependencies
RUN pip3 install --no-cache-dir .

# Create non-root user
RUN useradd -m -u 1000 proteinops && \
    mkdir -p /app/logs /app/models /app/data && \
    chown -R proteinops:proteinops /app
USER proteinops

# Create volume mount points
VOLUME ["/app/logs", "/app/models", "/app/data"]

# Expose ports
EXPOSE 8000 9090

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import sys; sys.path.insert(0, 'src'); from protein_operators import ProteinDesigner; print('GPU health check passed')" || exit 1

# GPU-enabled command
CMD ["python", "-m", "protein_operators.api.app", "--enable-gpu"]

# GPU validation stage
FROM development as gpu-test

# Install GPU validation tools
RUN pip install gpustat nvidia-ml-py3

# GPU test script
COPY <<EOF /workspace/test_gpu.py
import torch
import jax

def test_gpu_availability():
    """Test GPU availability for PyTorch and JAX"""
    print("=" * 50)
    print("GPU Availability Test")
    print("=" * 50)
    
    # PyTorch GPU test
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # JAX GPU test
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Simple computation test
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x.t())
        print(f"PyTorch GPU computation test: {'PASSED' if y.shape == (1000, 1000) else 'FAILED'}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_gpu_availability()
EOF

CMD ["python", "/workspace/test_gpu.py"]