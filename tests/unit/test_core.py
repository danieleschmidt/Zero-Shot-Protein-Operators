"""
Unit tests for core protein design functionality.

Tests the ProteinDesigner class and core design workflows.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from protein_operators.core import ProteinDesigner
from protein_operators.constraints import Constraints
from protein_operators.structure import ProteinStructure


class TestProteinDesigner:
    """Test ProteinDesigner class."""
    
    def test_initialization(self):
        """Test designer initialization."""
        designer = ProteinDesigner(
            operator_type="deeponet",
            device="cpu"
        )
        
        assert designer.operator_type == "deeponet"
        assert designer.device.type == "cpu"
        assert designer.design_count == 0
        assert designer.success_rate == 0.0
    
    def test_initialization_with_checkpoint(self, tmp_path):
        """Test initialization with model checkpoint."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "test_model.pt"
        dummy_state = {"layer.weight": torch.randn(10, 5)}
        torch.save(dummy_state, checkpoint_path)
        
        with patch("protein_operators.core.ProteinDeepONet") as mock_model:
            mock_instance = Mock()
            mock_model.return_value = mock_instance
            
            designer = ProteinDesigner(
                operator_type="deeponet",
                checkpoint=str(checkpoint_path),
                device="cpu"
            )
            
            # Verify model was loaded
            mock_instance.load_state_dict.assert_called_once()
    
    def test_device_setup(self):
        """Test device setup logic."""
        # Auto device selection
        designer1 = ProteinDesigner(device="auto")
        assert designer1.device is not None
        
        # Explicit CPU
        designer2 = ProteinDesigner(device="cpu")
        assert designer2.device.type == "cpu"
        
        # None should default to auto
        designer3 = ProteinDesigner(device=None)
        assert designer3.device is not None
    
    def test_constraint_validation(self, sample_constraints):
        """Test constraint validation."""
        designer = ProteinDesigner(device="cpu")
        
        # Valid constraints
        designer._validate_constraints(sample_constraints, 100)
        
        # Invalid length
        with pytest.raises(ValueError, match="must be positive"):
            designer._validate_constraints(sample_constraints, 0)
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            designer._validate_constraints(sample_constraints, 1500)
    
    def test_constraint_encoding(self, sample_constraints):
        """Test constraint encoding."""
        designer = ProteinDesigner(device="cpu")
        
        encoding = designer._encode_constraints(sample_constraints)
        
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape == (1, 256)
        assert torch.isfinite(encoding).all()
    
    def test_coordinate_generation(self, sample_constraints):
        """Test coordinate generation."""
        with patch.object(ProteinDesigner, "_physics_based_generation") as mock_physics:
            mock_physics.return_value = torch.randn(50, 3)
            
            designer = ProteinDesigner(device="cpu")
            
            constraint_encoding = torch.randn(1, 256)
            coordinates = designer._generate_coordinates(
                constraint_encoding, 50, 1
            )
            
            assert coordinates.shape == (1, 50, 3)
            assert torch.isfinite(coordinates).all()
    
    def test_physics_refinement(self):
        """Test physics-based refinement."""
        designer = ProteinDesigner(device="cpu")
        
        initial_coords = torch.randn(2, 20, 3)
        refined_coords = designer._refine_with_physics(initial_coords)
        
        assert refined_coords.shape == initial_coords.shape
        assert torch.isfinite(refined_coords).all()
    
    def test_structure_creation(self, sample_constraints):
        """Test structure creation."""
        designer = ProteinDesigner(device="cpu")
        
        coordinates = torch.randn(2, 50, 3)
        structure = designer._create_structure(coordinates, sample_constraints)
        
        assert isinstance(structure, ProteinStructure)
        assert structure.coordinates.shape == (50, 3)
    
    def test_generate_basic(self, sample_constraints):
        """Test basic protein generation."""
        with patch.object(ProteinDesigner, "_physics_based_generation") as mock_physics:
            mock_physics.return_value = torch.randn(50, 3)
            
            designer = ProteinDesigner(device="cpu")
            
            structure = designer.generate(
                constraints=sample_constraints,
                length=50,
                num_samples=1
            )
            
            assert isinstance(structure, ProteinStructure)
            assert structure.coordinates.shape == (50, 3)
            assert designer.design_count == 1
    
    def test_generate_multiple_samples(self, sample_constraints):
        """Test generation with multiple samples."""
        with patch.object(ProteinDesigner, "_physics_based_generation") as mock_physics:
            mock_physics.return_value = torch.randn(50, 3)
            
            designer = ProteinDesigner(device="cpu")
            
            structure = designer.generate(
                constraints=sample_constraints,
                length=50,
                num_samples=5
            )
            
            assert isinstance(structure, ProteinStructure)
            assert structure.coordinates.shape == (50, 3)
    
    def test_generate_with_physics(self, sample_constraints):
        """Test generation with physics guidance."""
        from protein_operators.pde import FoldingPDE
        
        pde = Mock(spec=FoldingPDE)
        designer = ProteinDesigner(device="cpu", pde=pde)
        
        with patch.object(designer, "_refine_with_physics") as mock_refine:
            mock_refine.return_value = torch.randn(1, 50, 3)
            
            structure = designer.generate(
                constraints=sample_constraints,
                length=50,
                physics_guided=True
            )
            
            mock_refine.assert_called_once()
            assert isinstance(structure, ProteinStructure)
    
    def test_validation(self, sample_protein_structure):
        """Test structure validation."""
        designer = ProteinDesigner(device="cpu")
        
        metrics = designer.validate(sample_protein_structure)
        
        assert isinstance(metrics, dict)
        assert "stereochemistry_score" in metrics
        assert "clash_score" in metrics
        assert "ramachandran_score" in metrics
        assert "constraint_satisfaction" in metrics
        
        # All scores should be between 0 and 1
        for score in metrics.values():
            assert 0 <= score <= 1
    
    def test_optimization(self, sample_protein_structure):
        """Test structure optimization."""
        designer = ProteinDesigner(device="cpu")
        
        optimized = designer.optimize(sample_protein_structure, iterations=10)
        
        assert isinstance(optimized, ProteinStructure)
        assert optimized.coordinates.shape == sample_protein_structure.coordinates.shape
    
    def test_statistics(self, sample_constraints):
        """Test statistics tracking."""
        designer = ProteinDesigner(device="cpu")
        
        initial_stats = designer.statistics
        assert initial_stats["designs_generated"] == 0
        
        # Generate a design
        with patch.object(ProteinDesigner, "_physics_based_generation") as mock_physics:
            mock_physics.return_value = torch.randn(50, 3)
            
            designer.generate(sample_constraints, 50)
            
            updated_stats = designer.statistics
            assert updated_stats["designs_generated"] == 1
    
    def test_physics_based_generation(self, sample_constraints):
        """Test fallback physics-based generation."""
        designer = ProteinDesigner(device="cpu")
        
        constraint_encoding = torch.randn(1, 256)
        coords = designer._physics_based_generation(constraint_encoding, 30)
        
        assert coords.shape == (30, 3)
        assert torch.isfinite(coords).all()
    
    def test_physics_energy_computation(self):
        """Test physics energy computation."""
        designer = ProteinDesigner(device="cpu")
        
        coordinates = torch.randn(1, 20, 3)
        energy = designer._compute_physics_energy(coordinates)
        
        assert isinstance(energy, torch.Tensor)
        assert energy.dim() == 0  # Scalar
        assert energy >= 0  # Energy should be non-negative
    
    def test_structure_scoring(self):
        """Test structure scoring."""
        designer = ProteinDesigner(device="cpu")
        
        coordinates = torch.randn(50, 3)
        score = designer._score_structure(coordinates)
        
        assert isinstance(score, torch.Tensor)
        assert score.dim() == 0  # Scalar
        assert score > 0  # Score should be positive
    
    def test_validation_methods(self, sample_protein_structure):
        """Test individual validation methods."""
        designer = ProteinDesigner(device="cpu")
        coords = sample_protein_structure.coordinates
        
        # Test stereochemistry validation
        stereo_score = designer._validate_stereochemistry(coords)
        assert 0 <= stereo_score <= 1
        
        # Test clash validation
        clash_score = designer._validate_clashes(coords)
        assert 0 <= clash_score <= 1
        
        # Test Ramachandran validation
        rama_score = designer._validate_ramachandran(coords)
        assert 0 <= rama_score <= 1
        
        # Test constraint satisfaction
        constraint_score = designer._validate_constraints_satisfaction(sample_protein_structure)
        assert 0 <= constraint_score <= 1
    
    def test_total_energy_computation(self, sample_protein_structure):
        """Test total energy computation with constraints."""
        designer = ProteinDesigner(device="cpu")
        
        coords = sample_protein_structure.coordinates
        constraints = sample_protein_structure.constraints
        
        energy = designer._compute_total_energy(coords, constraints)
        
        assert isinstance(energy, torch.Tensor)
        assert energy.dim() == 0  # Scalar
        assert energy >= 0
    
    def test_error_handling(self, sample_constraints):
        """Test error handling in generation."""
        designer = ProteinDesigner(device="cpu")
        
        # Test with invalid constraint
        invalid_constraints = Constraints()
        invalid_constraints.add_binding_site([200, 300], "ligand")  # Beyond protein length
        
        with pytest.raises(ValueError):
            designer.generate(invalid_constraints, 50)
    
    @pytest.mark.slow
    def test_large_protein_generation(self, sample_constraints):
        """Test generation of large proteins."""
        with patch.object(ProteinDesigner, "_physics_based_generation") as mock_physics:
            mock_physics.return_value = torch.randn(500, 3)
            
            designer = ProteinDesigner(device="cpu")
            
            structure = designer.generate(
                constraints=sample_constraints,
                length=500,  # Large protein
                num_samples=1
            )
            
            assert structure.coordinates.shape == (500, 3)
    
    def test_operator_type_validation(self):
        """Test operator type validation."""
        # Valid operator type
        designer = ProteinDesigner(operator_type="deeponet", device="cpu")
        assert designer.operator_type == "deeponet"
        
        # Invalid operator type should raise error
        with pytest.raises(ValueError, match="Unknown operator type"):
            ProteinDesigner(operator_type="invalid_type", device="cpu")
    
    def test_model_loading_error(self, tmp_path):
        """Test error handling in model loading."""
        # Non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            ProteinDesigner(
                operator_type="deeponet",
                checkpoint="non_existent.pt",
                device="cpu"
            )
    
    def test_concurrent_generation(self, sample_constraints):
        """Test thread safety of generation."""
        import threading
        
        designer = ProteinDesigner(device="cpu")
        results = []
        
        def generate_protein():
            with patch.object(ProteinDesigner, "_physics_based_generation") as mock_physics:
                mock_physics.return_value = torch.randn(50, 3)
                structure = designer.generate(sample_constraints, 50)
                results.append(structure)
        
        # Run multiple threads
        threads = [threading.Thread(target=generate_protein) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(results) == 3
        assert all(isinstance(s, ProteinStructure) for s in results)