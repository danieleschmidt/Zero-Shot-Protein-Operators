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

# Install package in production mode
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 proteinops
USER proteinops

# Default command for production
CMD ["python", "-m", "protein_operators.cli"]

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