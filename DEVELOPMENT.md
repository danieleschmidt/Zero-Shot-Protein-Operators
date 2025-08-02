# Development Guide

This guide covers development workflows, best practices, and contribution guidelines for the Zero-Shot Protein Operators project.

## Development Environment Setup

### Quick Start
```bash
# Clone and setup
git clone https://github.com/danieleschmidt/Zero-Shot-Protein-Operators
cd Zero-Shot-Protein-Operators
python scripts/setup.py

# Activate environment
conda activate protein-operators

# Install in development mode
pip install -e ".[dev,experiments]"
```

### Directory Structure
```
Zero-Shot-Protein-Operators/
├── src/protein_operators/     # Main package source code
│   ├── models/               # Neural operator implementations
│   ├── constraints/          # Constraint specification system
│   ├── pde/                 # PDE formulations
│   ├── training/            # Training utilities
│   ├── validation/          # Validation and testing
│   └── visualization/       # Visualization tools
├── experiments/             # Research experiments
│   ├── notebooks/          # Jupyter notebooks
│   ├── configs/           # Experiment configurations
│   └── results/           # Experimental results
├── tests/                  # Test suite
├── scripts/               # Utility scripts
├── data/                  # Data storage
├── models/                # Model checkpoints
└── docs/                  # Documentation
```

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/ -v
black src/ tests/
isort src/ tests/
mypy src/

# Commit with descriptive message
git commit -m "feat: add neural operator feature

- Implement DeepONet architecture
- Add constraint encoding system
- Include unit tests and documentation"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=protein_operators --cov-report=html

# Run specific test categories
pytest tests/test_models.py -v
pytest -k "test_deeponet" -v

# Run tests on GPU (if available)
pytest tests/ -m gpu
```

### 3. Code Quality
```bash
# Format code
black src/ tests/ experiments/
isort src/ tests/ experiments/

# Check code quality
flake8 src/ tests/
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

### 4. Experiments
```bash
# Start Jupyter Lab
jupyter lab

# Run experiment script
python experiments/train_deeponet.py --config configs/deeponet_base.yaml

# Track experiments with MLflow
mlflow ui  # View at http://localhost:5000
```

## Code Standards

### Python Style
- Follow PEP 8 with line length 88 (Black default)
- Use type hints for all public functions
- Document all public classes and methods with docstrings
- Prefer descriptive variable names over comments

### Documentation Style
```python
def generate_protein(
    constraints: Constraints,
    length: int,
    temperature: float = 300.0
) -> ProteinStructure:
    """
    Generate protein structure from constraints.
    
    Args:
        constraints: Design constraints specification
        length: Target protein length in residues
        temperature: Simulation temperature in Kelvin
        
    Returns:
        Generated protein structure
        
    Raises:
        ValueError: If constraints are invalid
        RuntimeError: If generation fails
        
    Examples:
        >>> constraints = Constraints()
        >>> constraints.add_binding_site(ligand="ATP")
        >>> structure = generate_protein(constraints, length=100)
    """
```

### Testing Standards
- Aim for >80% test coverage
- Write unit tests for all public functions
- Include integration tests for workflows
- Use fixtures for complex test data
- Test both CPU and GPU code paths

```python
import pytest
import torch
from protein_operators import ProteinDesigner

class TestProteinDesigner:
    """Test protein design functionality."""
    
    @pytest.fixture
    def designer(self):
        """Create test designer instance."""
        return ProteinDesigner(operator_type="deeponet", device="cpu")
    
    def test_basic_generation(self, designer):
        """Test basic protein generation."""
        constraints = Constraints()
        structure = designer.generate(constraints, length=50)
        assert structure.num_residues == 50
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_generation(self):
        """Test generation on GPU."""
        designer = ProteinDesigner(device="cuda")
        # ... test GPU-specific functionality
```

## Research Best Practices

### Experiment Organization
- Use descriptive experiment names: `YYYY-MM-DD_experiment_description`
- Store configurations in `experiments/configs/`
- Track all experiments with MLflow or Weights & Biases
- Document results in experiment notebooks

### Reproducibility
- Set random seeds in all experiments
- Pin dependency versions in environment files
- Save exact configurations with results
- Use version control for code and DVC for data

### Data Management
```bash
# Initialize DVC (if using)
dvc init

# Track large datasets
dvc add data/protein_structures.h5
git add data/protein_structures.h5.dv data/.gitignore

# Version data changes
dvc add data/processed_features.pkl
git add data/processed_features.pkl.dv
git commit -m "data: add processed features v2.0"
```

## Neural Operator Development

### Adding New Architectures
1. Inherit from `BaseNeuralOperator`
2. Implement required abstract methods
3. Add configuration to model factory
4. Include comprehensive tests
5. Document architecture choices in ADR

```python
class NewOperator(BaseNeuralOperator):
    """New neural operator architecture."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize architecture
    
    def encode_constraints(self, constraints: torch.Tensor) -> torch.Tensor:
        # Implement constraint encoding
        pass
    
    def encode_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        # Implement coordinate encoding
        pass
    
    def operator_forward(self, constraint_encoding, coordinate_encoding):
        # Implement main computation
        pass
```

### Physics Integration
- Always validate physics consistency
- Include energy conservation tests
- Document PDE formulations clearly
- Test against analytical solutions where possible

## Performance Optimization

### Profiling
```bash
# Profile training script
python -m cProfile -o profile.out experiments/train_model.py
python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(20)"

# Memory profiling
python -m memory_profiler experiments/train_model.py

# GPU profiling (NVIDIA)
nsys profile --trace cuda,nvtx python experiments/train_model.py
```

### Optimization Guidelines
- Use mixed precision training when possible
- Implement gradient checkpointing for large models
- Optimize data loading with multiple workers
- Use appropriate batch sizes for hardware
- Profile regularly to identify bottlenecks

## Deployment

### Model Checkpoints
- Save models with metadata and configuration
- Include validation metrics with checkpoints
- Use semantic versioning for model releases
- Provide model cards documenting capabilities

### API Development
- Follow REST API best practices
- Include comprehensive input validation
- Provide clear error messages
- Document API endpoints with OpenAPI/Swagger

## Contributing

### Pull Request Process
1. Fork the repository
2. Create feature branch from `main`
3. Make changes with tests and documentation
4. Run full test suite
5. Update CHANGELOG.md
6. Submit PR with clear description

### Review Criteria
- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact assessed
- [ ] Scientific validity confirmed

### Issue Guidelines
- Use issue templates for bugs/features
- Include minimal reproducible examples
- Provide environment information
- Search existing issues first

## Community

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Documentation: In-code docs and external guides

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn the codebase
- Acknowledge contributions appropriately

## Resources

### Learning Materials
- [Neural Operators Theory](docs/theory/neural_operators.md)
- [Protein Folding PDEs](docs/theory/protein_pdes.md)
- [Architecture Decision Records](docs/adr/)

### External Dependencies
- [PyTorch Documentation](https://pytorch.org/docs/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [OpenMM User Guide](http://docs.openmm.org/)
- [BioPython Tutorial](https://biopython.org/DIST/docs/tutorial/Tutorial.html)

### Debugging Tips
- Use `torch.autograd.detect_anomaly()` for gradient issues
- Enable CUDA error checking: `CUDA_LAUNCH_BLOCKING=1`
- Use `pdb` or `ipdb` for interactive debugging
- Check tensor shapes and devices frequently
- Validate physics constraints in unit tests