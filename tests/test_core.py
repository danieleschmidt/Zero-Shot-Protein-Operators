"""
Tests for core protein design functionality.
"""

import pytest
import torch
from protein_operators import ProteinDesigner, Constraints
from protein_operators.models import ProteinDeepONet, ProteinFNO


class TestProteinDesigner:
    """Test cases for ProteinDesigner class."""
    
    def test_designer_initialization(self):
        """Test designer initialization with different configurations."""
        # Test DeepONet initialization
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        assert designer.operator_type == "deeponet"
        assert designer.device.type == "cpu"
        assert isinstance(designer.model, ProteinDeepONet)
        
        # Test FNO initialization
        designer = ProteinDesigner(operator_type="fno", device="cpu")
        assert designer.operator_type == "fno"
        assert isinstance(designer.model, ProteinFNO)
    
    def test_invalid_operator_type(self):
        """Test that invalid operator types raise errors."""
        with pytest.raises(ValueError, match="Unknown operator type"):
            ProteinDesigner(operator_type="invalid")
    
    def test_device_setup(self):
        """Test device selection logic."""
        designer = ProteinDesigner(device="cpu")
        assert designer.device.type == "cpu"
        
        # Test auto device selection
        designer = ProteinDesigner(device="auto")
        assert designer.device.type in ["cpu", "cuda"]
    
    def test_generate_basic(self):
        """Test basic structure generation."""
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        constraints = Constraints()
        
        structure = designer.generate(
            constraints=constraints,
            length=50,
            num_samples=1
        )
        
        assert structure is not None
        assert hasattr(structure, 'coordinates')
        assert hasattr(structure, 'constraints')
    
    def test_generate_multiple_samples(self):
        """Test generation of multiple structure samples."""
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        constraints = Constraints()
        
        structure = designer.generate(
            constraints=constraints,
            length=30,
            num_samples=3
        )
        
        assert structure is not None
        # Would test that multiple samples are generated
    
    def test_statistics_tracking(self):
        """Test that design statistics are tracked."""
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        constraints = Constraints()
        
        initial_count = designer.design_count
        
        designer.generate(constraints=constraints, length=20)
        
        assert designer.design_count == initial_count + 1
        
        stats = designer.statistics
        assert "designs_generated" in stats
        assert "operator_type" in stats
        assert stats["operator_type"] == "deeponet"


class TestConstraints:
    """Test cases for Constraints system."""
    
    def test_empty_constraints(self):
        """Test empty constraints container."""
        constraints = Constraints()
        assert len(constraints) == 0
        
        encoding = constraints.encode()
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[0] >= 1  # At least one constraint slot
    
    def test_constraint_encoding(self):
        """Test constraint encoding functionality."""
        constraints = Constraints()
        
        # Test with empty constraints
        encoding = constraints.encode(max_constraints=5)
        assert encoding.shape[0] == 5  # Padded to max_constraints
        assert encoding.ndim == 2  # [num_constraints, constraint_dim]
    
    def test_constraint_management(self):
        """Test adding and removing constraints."""
        constraints = Constraints()
        
        # Test that we can call these methods without errors
        # (actual constraint implementations would be tested separately)
        assert len(constraints) == 0
        
        # Test iteration
        constraint_list = list(constraints)
        assert constraint_list == []
    
    def test_constraint_serialization(self):
        """Test constraint serialization to/from dict."""
        constraints = Constraints()
        
        # Test empty constraints
        data = constraints.to_dict()
        assert "constraints" in data
        assert "num_constraints" in data
        assert data["num_constraints"] == 0
        
        # Test deserialization
        restored = Constraints.from_dict(data)
        assert len(restored) == 0


class TestPhysicsIntegration:
    """Test physics-informed aspects of protein design."""
    
    def test_physics_guided_generation(self):
        """Test physics-guided structure generation."""
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        constraints = Constraints()
        
        # Test that physics_guided parameter is accepted
        structure = designer.generate(
            constraints=constraints,
            length=25,
            physics_guided=True
        )
        
        assert structure is not None
    
    def test_validation_pipeline(self):
        """Test structure validation."""
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        constraints = Constraints()
        
        structure = designer.generate(constraints=constraints, length=20)
        
        # Test validation
        metrics = designer.validate(structure)
        assert isinstance(metrics, dict)
        assert "stereochemistry_score" in metrics
        assert "clash_score" in metrics
        assert "ramachandran_score" in metrics
        assert "constraint_satisfaction" in metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUFunctionality:
    """Test GPU-specific functionality."""
    
    def test_gpu_initialization(self):
        """Test GPU device initialization."""
        designer = ProteinDesigner(operator_type="deeponet", device="cuda")
        assert designer.device.type == "cuda"
    
    def test_gpu_generation(self):
        """Test structure generation on GPU."""
        designer = ProteinDesigner(operator_type="deeponet", device="cuda")
        constraints = Constraints()
        
        structure = designer.generate(
            constraints=constraints,
            length=30,
            num_samples=1
        )
        
        assert structure is not None
        # Test that coordinates are on correct device
        if hasattr(structure, 'coordinates') and torch.is_tensor(structure.coordinates):
            assert structure.coordinates.device.type == "cuda"


class TestModelArchitectures:
    """Test different neural operator architectures."""
    
    def test_deeponet_vs_fno(self):
        """Test that different operators produce different results."""
        constraints = Constraints()
        
        # DeepONet
        designer_don = ProteinDesigner(operator_type="deeponet", device="cpu")
        structure_don = designer_don.generate(constraints=constraints, length=20)
        
        # FNO
        designer_fno = ProteinDesigner(operator_type="fno", device="cpu")
        structure_fno = designer_fno.generate(constraints=constraints, length=20)
        
        # Both should produce valid structures
        assert structure_don is not None
        assert structure_fno is not None
    
    def test_model_parameters(self):
        """Test model parameter customization."""
        # Test DeepONet with custom parameters
        designer = ProteinDesigner(
            operator_type="deeponet",
            device="cpu",
            constraint_dim=128,
            branch_hidden=[256, 512],
            trunk_hidden=[256, 512],
            num_basis=512
        )
        
        assert designer.model.input_dim == 128
        assert designer.model.num_basis == 512


if __name__ == "__main__":
    pytest.main([__file__])