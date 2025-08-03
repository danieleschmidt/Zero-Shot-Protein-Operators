# Contributing to Zero-Shot Protein-Operators

We welcome contributions from the computational biology and machine learning communities! This guide helps you get started.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following our guidelines
4. Run tests: `pytest tests/`
5. Submit a pull request

## Development Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional)
- Git LFS for model weights

### Installation
```bash
git clone https://github.com/your-username/Zero-Shot-Protein-Operators
cd Zero-Shot-Protein-Operators
conda env create -f environment.yml
conda activate protein-operators
pip install -e ".[dev,experiments]"
pre-commit install
```

## Contribution Guidelines

### Neural Operator Implementation

#### Code Standards
- **Type Hints**: All functions must include type annotations
- **Docstrings**: Use Google-style docstrings with examples
- **Error Handling**: Include proper exception handling
- **Testing**: 90%+ test coverage for new code

#### Neural Operator Best Practices
```python
class ProteinOperator(nn.Module):
    """Base class for protein neural operators.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output coordinate dimension (typically 3)
        
    Example:
        >>> operator = ProteinOperator(256, 3)
        >>> coords = operator(constraints, positions)
    """
    
    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(
        self, 
        constraints: torch.Tensor, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through operator.
        
        Args:
            constraints: Encoded constraints [batch, constraint_dim]
            positions: Spatial positions [batch, n_points, 3]
            
        Returns:
            Generated coordinates [batch, n_points, 3]
        """
        raise NotImplementedError
```

### Protein Validation Protocols

#### Required Validation Checks
1. **Stereochemistry**: Bond lengths, angles, chirality
2. **Clash Detection**: Van der Waals overlaps
3. **Ramachandran Analysis**: Backbone torsion angles
4. **Energy Validation**: Force field energy scoring
5. **Physics Consistency**: Conservation laws

#### Validation Implementation
```python
def validate_protein_structure(structure: ProteinStructure) -> ValidationResult:
    """Comprehensive protein structure validation.
    
    Args:
        structure: Generated protein structure
        
    Returns:
        Validation results with scores and diagnostics
    """
    results = ValidationResult()
    
    # Stereochemistry validation
    results.stereochemistry = validate_stereochemistry(structure)
    
    # Clash detection
    results.clash_score = detect_atomic_clashes(structure)
    
    # Ramachandran analysis
    results.ramachandran = analyze_backbone_torsions(structure)
    
    # Overall quality score
    results.overall_score = compute_overall_score(results)
    
    return results
```

### Benchmark Dataset Curation

#### Dataset Requirements
- **Diversity**: Represent all major protein folds
- **Quality**: High-resolution structures (< 2.5 Å)
- **Annotations**: Functional and structural metadata
- **Licensing**: Open access with clear usage rights

#### Dataset Structure
```
datasets/
├── training/
│   ├── pdb_files/
│   ├── constraints/
│   └── metadata.json
├── validation/
├── benchmarks/
│   ├── casp_targets/
│   ├── designed_proteins/
│   └── experimental_validation/
└── documentation/
    ├── dataset_description.md
    ├── curation_protocol.md
    └── usage_guidelines.md
```

## Code Review Process

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Performance benchmarks included
- [ ] Security considerations addressed

### Review Criteria
1. **Scientific Accuracy**: Validates against known physics
2. **Code Quality**: Clean, readable, maintainable
3. **Performance**: Efficient algorithms and implementations
4. **Testing**: Comprehensive test coverage
5. **Documentation**: Clear and complete documentation

## Testing Guidelines

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **Physics Tests**: Conservation law verification
4. **Benchmark Tests**: Performance regression testing
5. **End-to-End Tests**: Complete pipeline validation

### Test Implementation
```python
import pytest
import torch
from protein_operators import ProteinDesigner

class TestProteinDesigner:
    """Test suite for ProteinDesigner class."""
    
    @pytest.fixture
    def designer(self):
        return ProteinDesigner(operator_type="deeponet")
    
    def test_constraint_encoding(self, designer):
        """Test constraint encoding functionality."""
        constraints = Constraints()
        constraints.add_binding_site([10, 20, 30], "ATP")
        
        encoding = designer._encode_constraints(constraints)
        
        assert encoding.shape[-1] == 256  # Expected encoding dim
        assert torch.isfinite(encoding).all()  # No NaN/Inf values
    
    def test_physics_conservation(self, designer):
        """Test energy conservation in generated structures."""
        constraints = Constraints()
        structure = designer.generate(constraints, length=50)
        
        # Verify energy conservation
        initial_energy = structure.compute_energy()
        refined_structure = designer.optimize(structure)
        final_energy = refined_structure.compute_energy()
        
        assert final_energy <= initial_energy  # Energy should decrease
```

## Performance Guidelines

### Optimization Targets
- **Inference Speed**: < 1 second per 100-residue protein
- **Memory Usage**: < 8GB GPU memory for largest models
- **Batch Processing**: Support 32+ concurrent designs
- **Scalability**: Linear scaling with protein length

### Profiling Requirements
```python
# Always profile new implementations
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    structure = designer.generate(constraints, length=200)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Documentation Standards

### API Documentation
- Use Google-style docstrings
- Include type hints
- Provide usage examples
- Document exceptions

### Tutorial Requirements
- Step-by-step explanations
- Runnable code examples
- Expected outputs
- Common troubleshooting

## Community Guidelines

### Code of Conduct
We are committed to providing a welcoming and inclusive environment:
- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge contributions

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community chat
- **Discord**: Real-time community support
- **Mailing List**: Development announcements

## Recognition

We value all contributions and provide recognition through:
- Contributor acknowledgments in releases
- Co-authorship opportunities for significant contributions
- Conference presentation opportunities
- Access to beta features and datasets

## Getting Help

### Resources
- Documentation: https://protein-operators.readthedocs.io
- Tutorials: `notebooks/tutorials/`
- Examples: `examples/`
- FAQ: `docs/faq.md`

### Support Channels
1. Check existing GitHub issues
2. Search documentation
3. Ask in community discussions
4. Join our Discord server
5. Contact maintainers directly

Thank you for contributing to advancing computational protein design! 🧬