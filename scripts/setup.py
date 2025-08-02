#!/usr/bin/env python3
"""
Setup script for protein operators development environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"🔧 {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False


def check_requirements():
    """Check if basic requirements are available."""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check conda
    if not run_command("conda --version", "Checking conda"):
        print("❌ Conda not found. Please install Miniconda or Anaconda.")
        return False
    
    # Check git
    if not run_command("git --version", "Checking git"):
        print("❌ Git not found.")
        return False
    
    # Check NVIDIA GPU (optional)
    gpu_available = run_command("nvidia-smi", "Checking NVIDIA GPU")
    if gpu_available:
        print("✅ NVIDIA GPU detected")
    else:
        print("⚠️  No NVIDIA GPU detected (CPU-only mode)")
    
    return True


def setup_conda_environment():
    """Set up conda environment."""
    print("\n📦 Setting up conda environment...")
    
    env_name = "protein-operators"
    
    # Check if environment exists
    result = subprocess.run(f"conda env list | grep {env_name}", 
                          shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"⚠️  Environment '{env_name}' already exists")
        response = input("Do you want to update it? (y/N): ")
        if response.lower() == 'y':
            run_command(f"conda env update -f environment.yml", 
                       f"Updating {env_name} environment")
    else:
        run_command(f"conda env create -f environment.yml", 
                   f"Creating {env_name} environment")
    
    print(f"\n✅ Environment setup complete!")
    print(f"To activate: conda activate {env_name}")


def setup_pre_commit():
    """Set up pre-commit hooks."""
    print("\n🔗 Setting up pre-commit hooks...")
    
    # Create .pre-commit-config.yaml if it doesn't exist
    pre_commit_config = Path(".pre-commit-config.yaml")
    if not pre_commit_config.exists():
        config_content = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
"""
        with open(pre_commit_config, 'w') as f:
            f.write(config_content.strip())
    
    # Install pre-commit in the environment
    run_command("conda run -n protein-operators pre-commit install", 
               "Installing pre-commit hooks")


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "experiments/notebooks",
        "experiments/configs", 
        "experiments/results",
        "experiments/baselines",
        "experiments/analysis",
        "data/raw",
        "data/processed",
        "data/external",
        "models/checkpoints",
        "models/pretrained",
        "logs",
        "outputs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created {directory}/")
    
    # Create .gitkeep files for empty directories
    gitkeep_dirs = ["data/raw", "data/processed", "logs", "outputs"]
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()


def setup_jupyter():
    """Set up Jupyter Lab extensions and kernels."""
    print("\n📓 Setting up Jupyter Lab...")
    
    # Install jupyter lab extensions
    extensions = [
        "@jupyter-widgets/jupyterlab-manager",
        "jupyterlab-plotly",
        "@bokeh/jupyter_bokeh",
    ]
    
    for ext in extensions:
        run_command(f"conda run -n protein-operators jupyter labextension install {ext}", 
                   f"Installing {ext}")
    
    # Register conda environment as jupyter kernel
    run_command("conda run -n protein-operators python -m ipykernel install --user --name protein-operators --display-name 'Protein Operators'", 
               "Registering Jupyter kernel")


def create_example_notebook():
    """Create an example notebook."""
    print("\n📝 Creating example notebook...")
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Protein Operators - Getting Started\\n",
    "\\n",
    "This notebook demonstrates basic usage of the protein operators framework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "\\n",
    "# Import protein operators\\n",
    "from protein_operators import ProteinDesigner, Constraints\\n",
    "\\n",
    "print(f\\"PyTorch version: {torch.__version__}\\")\\n",
    "print(f\\"CUDA available: {torch.cuda.is_available()}\\")\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f\\"GPU: {torch.cuda.get_device_name(0)}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize protein designer\\n",
    "designer = ProteinDesigner(\\n",
    "    operator_type=\\"deeponet\\",\\n",
    "    device=\\"auto\\"\\n",
    ")\\n",
    "\\n",
    "print(f\\"Designer initialized with {designer.operator_type} on {designer.device}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create simple constraints\\n",
    "constraints = Constraints()\\n",
    "# TODO: Add specific constraints\\n",
    "\\n",
    "print(f\\"Created constraints: {constraints}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate protein structure\\n",
    "structure = designer.generate(\\n",
    "    constraints=constraints,\\n",
    "    length=100,\\n",
    "    num_samples=1\\n",
    ")\\n",
    "\\n",
    "print(f\\"Generated structure with {structure.num_residues} residues\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Protein Operators",
   "language": "python",
   "name": "protein-operators"
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
}'''
    
    notebook_path = Path("experiments/notebooks/01_getting_started.ipynb")
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"   Created {notebook_path}")


def main():
    """Main setup function."""
    print("🧬 Protein Operators Development Environment Setup")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ Requirements check failed. Please install missing dependencies.")
        sys.exit(1)
    
    setup_conda_environment()
    setup_directories() 
    setup_pre_commit()
    setup_jupyter()
    create_example_notebook()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. conda activate protein-operators")
    print("2. pip install -e .")
    print("3. jupyter lab")
    print("4. Open experiments/notebooks/01_getting_started.ipynb")
    print("\nFor development:")
    print("- Run tests: pytest")
    print("- Format code: black .")
    print("- Check types: mypy src/")
    print("- Pre-commit: pre-commit run --all-files")


if __name__ == "__main__":
    main()