#!/bin/bash
set -e

echo "🚀 Setting up Zero-Shot Protein-Operators development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git-lfs \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libfuse2 \
    wget \
    curl \
    vim \
    htop \
    tree

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "🐍 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    rm /tmp/miniconda.sh
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
    export PATH="/opt/conda/bin:$PATH"
fi

# Initialize conda
conda init bash
source ~/.bashrc

# Create conda environment
echo "🔬 Creating protein-operators conda environment..."
if ! conda env list | grep -q protein-operators; then
    conda env create -f environment.yml
fi

# Activate environment
conda activate protein-operators

# Install additional development tools
echo "🛠️ Installing development tools..."
pip install --upgrade pip
pip install \
    pre-commit \
    black \
    isort \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    pytest-xvfb \
    jupyter \
    jupyterlab \
    ipywidgets \
    nbconvert \
    nbstripout \
    mlflow \
    wandb \
    tensorboard \
    seaborn \
    plotly \
    bokeh

# Install molecular visualization tools
echo "🧬 Installing molecular visualization tools..."
pip install \
    py3Dmol \
    nglview \
    MDAnalysis \
    biotite

# Install package in development mode
echo "🔧 Installing protein-operators in development mode..."
pip install -e ".[dev,experiments]"

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

# Setup git LFS
echo "📁 Setting up Git LFS..."
git lfs install

# Create data directories
echo "📂 Creating data directories..."
mkdir -p data/{raw,processed,models,experiments,benchmarks}
mkdir -p logs
mkdir -p notebooks
mkdir -p results

# Setup Jupyter extensions
echo "📓 Setting up Jupyter extensions..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install nglview-js-widgets

# Create useful aliases
echo "⚡ Setting up useful aliases..."
cat >> ~/.bashrc << 'EOL'

# Protein Operators aliases
alias po='python -m protein_operators'
alias po-train='python scripts/train.py'
alias po-design='python scripts/design.py'
alias po-validate='python scripts/validate.py'
alias jupyter-lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias tensorboard='tensorboard --host=0.0.0.0 --port=6006'
alias mlflow-ui='mlflow ui --host=0.0.0.0 --port=5000'

# Development helpers
alias test='pytest tests/ -v'
alias test-cov='pytest tests/ --cov=protein_operators --cov-report=html'
alias lint='pylint src/protein_operators'
alias format='black src/ tests/ && isort src/ tests/'
alias typecheck='mypy src/protein_operators'

# GPU monitoring
alias nvidia-mon='watch -n 1 nvidia-smi'
alias gpu-temp='nvidia-smi -q -d TEMPERATURE'
alias gpu-mem='nvidia-smi -q -d MEMORY'

EOL

# Setup environment variables
echo "🌍 Setting up environment variables..."
cat >> ~/.bashrc << 'EOL'

# Protein Operators environment
export PROTEIN_OPERATORS_ROOT="/workspaces/Zero-Shot-Protein-Operators"
export PROTEIN_OPERATORS_DATA="/workspaces/data"
export PROTEIN_OPERATORS_MODELS="/workspaces/data/models"
export PYTHONPATH="${PROTEIN_OPERATORS_ROOT}/src:${PYTHONPATH}"

# CUDA configuration
export CUDA_VISIBLE_DEVICES="0"
export CUDA_CACHE_DISABLE=1

# PyTorch configuration
export TORCH_HOME="/workspaces/data/models/torch"
export TRANSFORMERS_CACHE="/workspaces/data/models/transformers"

# OpenMM configuration
export OPENMM_PLUGIN_DIR="/opt/conda/lib/plugins"

# Molecular dynamics configuration
export GROMACS_DIR="/opt/conda"

# Jupyter configuration
export JUPYTER_CONFIG_DIR="/workspaces/.jupyter"
export JUPYTER_DATA_DIR="/workspaces/.jupyter"

EOL

# Create Jupyter config
echo "📝 Creating Jupyter configuration..."
mkdir -p /workspaces/.jupyter
cat > /workspaces/.jupyter/jupyter_lab_config.py << 'EOL'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.notebook_dir = '/workspaces/Zero-Shot-Protein-Operators'
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True
c.LabApp.check_for_updates_class = 'jupyterlab.NeverCheckForUpdate'
EOL

# Setup MLflow tracking
echo "📊 Setting up MLflow tracking..."
mkdir -p /workspaces/mlruns
export MLFLOW_TRACKING_URI="file:///workspaces/mlruns"

# Create sample notebooks
echo "📚 Creating sample notebooks..."
mkdir -p notebooks/tutorials
cat > notebooks/tutorials/01_getting_started.ipynb << 'EOL'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Getting Started with Zero-Shot Protein-Operators\n",
    "\n",
    "This notebook demonstrates basic usage of the protein design framework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from protein_operators import ProteinDesigner, Constraints\n",
    "\n",
    "print(\"🧬 Zero-Shot Protein-Operators Setup Complete!\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOL

# Set correct permissions
chmod +x .devcontainer/setup.sh
chown -R vscode:vscode /workspaces 2>/dev/null || true

echo "✅ Development environment setup complete!"
echo "🚀 Ready to design proteins with neural operators!"
echo ""
echo "Quick start commands:"
echo "  conda activate protein-operators"
echo "  jupyter-lab"
echo "  po --help"
echo ""
echo "Access points:"
echo "  Jupyter Lab: http://localhost:8888"
echo "  MLflow UI: http://localhost:5000"
echo "  TensorBoard: http://localhost:6006"