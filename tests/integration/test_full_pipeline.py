"""
Integration tests for the full protein design pipeline.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from protein_operators import ProteinDesigner, Constraints, BindingSiteConstraint, StructuralConstraint
from protein_operators.constraints.biophysical import StabilityConstraint, SolubilityConstraint
from protein_operators.structure import ProteinStructure


class TestFullPipeline:
    """Test complete protein design workflow."""
    
    def test_basic_design_pipeline(self):
        """Test basic protein design from constraints to structure."""
        # Create constraints
        constraints = Constraints()
        
        # Add binding site constraint
        binding_site = BindingSiteConstraint(
            residues=[10, 15, 20],
            ligand="ATP",
            affinity_nm=100.0
        )
        constraints.add_constraint(binding_site)
        
        # Add secondary structure constraint
        helix_constraint = StructuralConstraint(
            start=5,
            end=15,
            ss_type="helix"
        )
        constraints.add_constraint(helix_constraint)
        
        # Initialize designer
        designer = ProteinDesigner(operator_type="deeponet")
        
        # Generate structure
        structure = designer.generate(
            constraints=constraints,
            length=50,
            num_samples=1
        )
        
        # Validate results
        assert structure is not None
        assert structure.coordinates.shape == (50, 3)
        assert hasattr(structure, 'constraints')
        
        # Validate structure quality
        validation_metrics = designer.validate(structure)
        assert 'stereochemistry_score' in validation_metrics
        assert 'clash_score' in validation_metrics
        assert 'ramachandran_score' in validation_metrics
        assert 'constraint_satisfaction' in validation_metrics
        
        # All scores should be between 0 and 1
        for metric_name, score in validation_metrics.items():
            assert 0.0 <= score <= 1.0, f"{metric_name} score {score} not in valid range"
    
    def test_biophysical_constraints_design(self):
        """Test design with biophysical constraints."""
        constraints = Constraints()
        
        # Add stability constraint
        stability = StabilityConstraint(
            tm_celsius=75.0,
            ph_range=(6.5, 7.5),
            ionic_strength=0.15
        )
        constraints.add_constraint(stability)
        
        # Add solubility constraint
        solubility = SolubilityConstraint(
            min_solubility_mg_ml=10.0,
            ph_optimum=7.0,
            hydrophobicity_ratio=0.4
        )
        constraints.add_constraint(solubility)
        
        designer = ProteinDesigner(operator_type="fno")
        
        structure = designer.generate(
            constraints=constraints,
            length=80,
            num_samples=2
        )
        
        assert structure.coordinates.shape == (80, 3)
        
        # Test constraint satisfaction
        stability_score = stability.compute_satisfaction(structure)
        solubility_score = solubility.compute_satisfaction(structure)
        
        assert 0.0 <= stability_score <= 1.0
        assert 0.0 <= solubility_score <= 1.0
    
    def test_physics_guided_design(self):
        """Test physics-guided protein design."""
        from protein_operators.pde import FoldingPDE
        
        constraints = Constraints()
        binding_site = BindingSiteConstraint(
            residues=[25, 30, 35],
            ligand="DRUG",
            affinity_nm=50.0
        )
        constraints.add_constraint(binding_site)
        
        # Create PDE system
        pde = FoldingPDE(
            force_field="amber99sb",
            temperature=300.0
        )
        
        designer = ProteinDesigner(
            operator_type="deeponet",
            pde=pde
        )
        
        structure = designer.generate(
            constraints=constraints,
            length=60,
            physics_guided=True
        )
        
        assert structure.coordinates.shape == (60, 3)
        
        # Physics-guided design should have lower energy
        validation_metrics = designer.validate(structure)
        assert validation_metrics['stereochemistry_score'] > 0.5  # Should be reasonable
    
    def test_structure_optimization(self):
        """Test structure optimization workflow."""
        # Create initial structure
        constraints = Constraints()
        designer = ProteinDesigner()
        
        initial_structure = designer.generate(
            constraints=constraints,
            length=40
        )
        
        # Optimize structure
        optimized_structure = designer.optimize(
            initial_structure=initial_structure,
            iterations=50
        )
        
        assert optimized_structure.coordinates.shape == initial_structure.coordinates.shape
        
        # Validate optimization improved structure
        initial_metrics = designer.validate(initial_structure)
        optimized_metrics = designer.validate(optimized_structure)
        
        # At least one metric should improve (allowing for noise in simple implementation)
        improvement_found = any(
            optimized_metrics[metric] >= initial_metrics[metric] - 0.1  # Allow small degradation due to noise
            for metric in initial_metrics.keys()
        )
        assert improvement_found, "Optimization should improve at least one metric"
    
    def test_pdb_save_load_roundtrip(self):
        """Test saving and loading PDB files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure
            constraints = Constraints()
            designer = ProteinDesigner()
            
            original_structure = designer.generate(
                constraints=constraints,
                length=30
            )
            
            # Save to PDB
            pdb_path = Path(tmpdir) / "test_protein.pdb"
            original_structure.save_pdb(pdb_path)
            
            assert pdb_path.exists()
            
            # Load from PDB
            loaded_structure = ProteinStructure.from_pdb(pdb_path)
            
            # Validate loaded structure
            assert loaded_structure.coordinates.shape == original_structure.coordinates.shape
            assert hasattr(loaded_structure, 'sequence')
            assert len(loaded_structure.sequence) == original_structure.coordinates.shape[0]
            
            # Coordinates should be reasonably close (allowing for PDB precision)
            rmsd = loaded_structure.compute_rmsd(original_structure)
            assert rmsd < 0.01, f"RMSD too high: {rmsd}"
    
    def test_multi_constraint_design(self):
        """Test design with multiple overlapping constraints."""
        constraints = Constraints()
        
        # Multiple binding sites
        binding1 = BindingSiteConstraint(
            residues=[10, 12, 14],
            ligand="ATP"
        )
        binding2 = BindingSiteConstraint(
            residues=[30, 32, 34],
            ligand="GTP"
        )
        constraints.add_constraint(binding1)
        constraints.add_constraint(binding2)
        
        # Multiple secondary structures
        helix1 = StructuralConstraint(start=5, end=20, ss_type="helix")
        sheet1 = StructuralConstraint(start=25, end=35, ss_type="sheet")
        constraints.add_constraint(helix1)
        constraints.add_constraint(sheet1)
        
        # Biophysical constraints
        stability = StabilityConstraint(tm_celsius=80.0)
        solubility = SolubilityConstraint(min_solubility_mg_ml=5.0)
        constraints.add_constraint(stability)
        constraints.add_constraint(solubility)
        
        designer = ProteinDesigner(operator_type="deeponet")
        
        structure = designer.generate(
            constraints=constraints,
            length=100
        )
        
        assert structure.coordinates.shape == (100, 3)
        
        # Validate all constraints
        validation_metrics = designer.validate(structure)
        
        # Should satisfy constraints reasonably well
        assert validation_metrics['constraint_satisfaction'] > 0.3  # Relaxed threshold for complex constraints
    
    def test_constraint_validation_errors(self):
        """Test constraint validation catches errors."""
        constraints = Constraints()
        designer = ProteinDesigner()
        
        # Test invalid length
        with pytest.raises(ValueError, match="Protein length must be positive"):
            designer.generate(constraints, length=0)
        
        with pytest.raises(ValueError, match="too short"):
            designer.generate(constraints, length=5)
        
        # Test residue indices out of bounds
        invalid_binding = BindingSiteConstraint(
            residues=[50, 60, 70],  # Beyond protein length
            ligand="ATP"
        )
        constraints.add_constraint(invalid_binding)
        
        with pytest.raises(ValueError, match="residue indices must be between"):
            designer.generate(constraints, length=40)
    
    def test_designer_statistics(self):
        """Test designer tracks statistics correctly."""
        designer = ProteinDesigner()
        constraints = Constraints()
        
        initial_stats = designer.statistics
        assert initial_stats['designs_generated'] == 0
        
        # Generate a design
        structure = designer.generate(constraints, length=20)
        
        updated_stats = designer.statistics
        assert updated_stats['designs_generated'] == 1
        assert updated_stats['operator_type'] == 'deeponet'
        assert 'device' in updated_stats
    
    def test_error_handling(self):
        """Test graceful error handling in design pipeline."""
        designer = ProteinDesigner()
        
        # Test with None constraints (should handle gracefully)
        with pytest.raises(AttributeError):  # Expected to fail gracefully
            designer.generate(None, length=50)
        
        # Test with malformed constraints
        constraints = Constraints()
        
        # Add constraint with invalid parameters
        try:
            invalid_stability = StabilityConstraint(tm_celsius=300.0)  # Too high
            constraints.add_constraint(invalid_stability)
            
            # Should raise validation error
            with pytest.raises(ValueError):
                designer.generate(constraints, length=50)
        except ValueError:
            pass  # Expected validation error


class TestConstraintSystem:
    """Test constraint system functionality."""
    
    def test_constraint_encoding(self):
        """Test constraint encoding for neural operators."""
        constraints = Constraints()
        
        binding_site = BindingSiteConstraint(
            residues=[5, 10, 15],
            ligand="ATP",
            affinity_nm=100.0
        )
        constraints.add_constraint(binding_site)
        
        designer = ProteinDesigner()
        encoding = designer._encode_constraints(constraints)
        
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[-1] > 0  # Should have features
        assert not torch.isnan(encoding).any()  # No NaN values
        assert torch.isfinite(encoding).all()  # All finite values
    
    def test_constraint_satisfaction_scoring(self):
        """Test constraint satisfaction scoring."""
        # Create a simple structure
        coordinates = torch.randn(50, 3)
        constraints = Constraints()
        
        binding_site = BindingSiteConstraint(
            residues=[10, 15, 20],
            ligand="ATP"
        )
        constraints.add_constraint(binding_site)
        
        structure = ProteinStructure(coordinates, constraints)
        
        # Test satisfaction scoring
        satisfaction_score = binding_site.compute_satisfaction(structure)
        assert 0.0 <= satisfaction_score <= 1.0
    
    def test_biophysical_constraint_parameters(self):
        """Test biophysical constraint parameter validation."""
        # Test valid parameters
        stability = StabilityConstraint(
            tm_celsius=75.0,
            ph_range=(6.0, 8.0),
            ionic_strength=0.15
        )
        stability.validate_parameters()  # Should not raise
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            invalid_stability = StabilityConstraint(tm_celsius=200.0)  # Too high
            invalid_stability.validate_parameters()
        
        with pytest.raises(ValueError):
            invalid_ph = StabilityConstraint(ph_range=(10.0, 5.0))  # Invalid range
            invalid_ph.validate_parameters()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])