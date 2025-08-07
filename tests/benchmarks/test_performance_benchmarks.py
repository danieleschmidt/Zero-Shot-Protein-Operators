"""
Performance benchmarks for protein operators.

These benchmarks track performance regressions and improvements across releases.
"""

import pytest
import torch
import numpy as np
import time
from pathlib import Path
import tempfile

from protein_operators import ProteinDesigner, Constraints, BindingSiteConstraint, StructuralConstraint
from protein_operators.constraints.biophysical import StabilityConstraint, SolubilityConstraint
from protein_operators.structure import ProteinStructure
from protein_operators.utils.performance import get_performance_monitor, performance_timer
from protein_operators.cache.cache_manager import get_cache_manager


class TestDesignPerformance:
    """Benchmark protein design operations."""
    
    def test_basic_design_speed(self, benchmark):
        """Benchmark basic protein design speed."""
        def design_protein():
            constraints = Constraints()
            binding_site = BindingSiteConstraint(
                residues=[10, 15, 20],
                ligand="ATP"
            )
            constraints.add_constraint(binding_site)
            
            designer = ProteinDesigner(operator_type="deeponet")
            structure = designer.generate(
                constraints=constraints,
                length=50,
                num_samples=1
            )
            return structure
        
        result = benchmark(design_protein)
        assert result is not None
        assert result.coordinates.shape == (50, 3)
    
    def test_large_protein_design(self, benchmark):
        """Benchmark design of large proteins."""
        def design_large_protein():
            constraints = Constraints()
            
            # Multiple constraints for complexity
            for i in range(0, 200, 50):
                binding_site = BindingSiteConstraint(
                    residues=[i+10, i+15, i+20],
                    ligand=f"LIGAND_{i}"
                )
                constraints.add_constraint(binding_site)
            
            designer = ProteinDesigner(operator_type="deeponet")
            structure = designer.generate(
                constraints=constraints,
                length=200,
                num_samples=1
            )
            return structure
        
        result = benchmark(design_large_protein)
        assert result is not None
        assert result.coordinates.shape == (200, 3)
    
    def test_multi_sample_design(self, benchmark):
        """Benchmark multi-sample protein design."""
        def design_multi_sample():
            constraints = Constraints()
            binding_site = BindingSiteConstraint(
                residues=[10, 15, 20],
                ligand="ATP"
            )
            constraints.add_constraint(binding_site)
            
            designer = ProteinDesigner(operator_type="deeponet")
            structure = designer.generate(
                constraints=constraints,
                length=100,
                num_samples=5
            )
            return structure
        
        result = benchmark(design_multi_sample)
        assert result is not None
        assert result.coordinates.shape == (100, 3)
    
    def test_complex_constraints_design(self, benchmark):
        """Benchmark design with complex constraints."""
        def design_complex_constraints():
            constraints = Constraints()
            
            # Multiple binding sites
            constraints.add_constraint(BindingSiteConstraint(
                residues=[10, 15, 20], ligand="ATP"
            ))
            constraints.add_constraint(BindingSiteConstraint(
                residues=[30, 35, 40], ligand="GTP"
            ))
            
            # Secondary structure constraints
            constraints.add_constraint(StructuralConstraint(
                start=5, end=25, ss_type="helix"
            ))
            constraints.add_constraint(StructuralConstraint(
                start=45, end=65, ss_type="sheet"
            ))
            
            # Biophysical constraints
            constraints.add_constraint(StabilityConstraint(
                tm_celsius=75.0,
                ph_range=(6.5, 8.0)
            ))
            constraints.add_constraint(SolubilityConstraint(
                min_solubility_mg_ml=10.0
            ))
            
            designer = ProteinDesigner(operator_type="deeponet")
            structure = designer.generate(
                constraints=constraints,
                length=120,
                num_samples=1
            )
            return structure
        
        result = benchmark(design_complex_constraints)
        assert result is not None
        assert result.coordinates.shape == (120, 3)
    
    def test_physics_guided_design(self, benchmark):
        """Benchmark physics-guided design."""
        def design_physics_guided():
            from protein_operators.pde import FoldingPDE
            
            constraints = Constraints()
            constraints.add_constraint(BindingSiteConstraint(
                residues=[20, 25, 30], ligand="DRUG"
            ))
            
            pde = FoldingPDE(force_field="amber99sb")
            designer = ProteinDesigner(
                operator_type="deeponet",
                pde=pde
            )
            
            structure = designer.generate(
                constraints=constraints,
                length=80,
                physics_guided=True
            )
            return structure
        
        result = benchmark(design_physics_guided)
        assert result is not None
        assert result.coordinates.shape == (80, 3)


class TestOptimizationPerformance:
    """Benchmark structure optimization operations."""
    
    def test_structure_optimization_speed(self, benchmark):
        """Benchmark structure optimization speed."""
        def optimize_structure():
            # Create initial structure
            coordinates = torch.randn(60, 3) * 10  # Random coordinates
            constraints = Constraints()
            constraints.add_constraint(BindingSiteConstraint(
                residues=[15, 20, 25], ligand="ATP"
            ))
            
            structure = ProteinStructure(coordinates, constraints)
            
            designer = ProteinDesigner()
            optimized = designer.optimize(
                initial_structure=structure,
                iterations=100
            )
            return optimized
        
        result = benchmark(optimize_structure)
        assert result is not None
        assert result.coordinates.shape == (60, 3)
    
    def test_large_structure_optimization(self, benchmark):
        """Benchmark optimization of large structures."""
        def optimize_large_structure():
            coordinates = torch.randn(300, 3) * 15  # Large random structure
            constraints = Constraints()
            
            # Add multiple constraints
            for i in range(0, 300, 100):
                constraints.add_constraint(BindingSiteConstraint(
                    residues=[i+10, i+20, i+30], ligand=f"LIG_{i}"
                ))
            
            structure = ProteinStructure(coordinates, constraints)
            
            designer = ProteinDesigner()
            optimized = designer.optimize(
                initial_structure=structure,
                iterations=50  # Fewer iterations for large structures
            )
            return optimized
        
        result = benchmark(optimize_large_structure)
        assert result is not None
        assert result.coordinates.shape == (300, 3)


class TestValidationPerformance:
    """Benchmark validation operations."""
    
    def test_structure_validation_speed(self, benchmark):
        """Benchmark structure validation speed."""
        def validate_structure():
            coordinates = torch.randn(100, 3) * 8
            constraints = Constraints()
            structure = ProteinStructure(coordinates, constraints)
            
            designer = ProteinDesigner()
            metrics = designer.validate(structure)
            return metrics
        
        result = benchmark(validate_structure)
        assert isinstance(result, dict)
        assert 'stereochemistry_score' in result
        assert 'clash_score' in result
    
    def test_large_structure_validation(self, benchmark):
        """Benchmark validation of large structures."""
        def validate_large_structure():
            coordinates = torch.randn(500, 3) * 12
            constraints = Constraints()
            structure = ProteinStructure(coordinates, constraints)
            
            designer = ProteinDesigner()
            metrics = designer.validate(structure)
            return metrics
        
        result = benchmark(validate_large_structure)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestIOPerformance:
    """Benchmark I/O operations."""
    
    def test_pdb_save_speed(self, benchmark):
        """Benchmark PDB file saving speed."""
        def save_pdb():
            coordinates = torch.randn(200, 3) * 10
            structure = ProteinStructure(coordinates)
            structure.sequence = 'A' * 200
            
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=True) as f:
                pdb_path = Path(f.name)
                structure.save_pdb(pdb_path)
                return pdb_path.stat().st_size
        
        result = benchmark(save_pdb)
        assert result > 0  # File should have content
    
    def test_pdb_load_speed(self, benchmark):
        """Benchmark PDB file loading speed."""
        # First create a PDB file to load
        coordinates = torch.randn(150, 3) * 10
        structure = ProteinStructure(coordinates)
        structure.sequence = 'A' * 150
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            pdb_path = Path(f.name)
        
        try:
            structure.save_pdb(pdb_path)
            
            def load_pdb():
                loaded_structure = ProteinStructure.from_pdb(pdb_path)
                return loaded_structure
            
            result = benchmark(load_pdb)
            assert result is not None
            assert result.coordinates.shape == (150, 3)
            
        finally:
            pdb_path.unlink()
    
    def test_large_pdb_operations(self, benchmark):
        """Benchmark large PDB file operations."""
        def large_pdb_roundtrip():
            # Create large structure
            coordinates = torch.randn(1000, 3) * 15
            structure = ProteinStructure(coordinates)
            structure.sequence = 'A' * 1000
            
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=True) as f:
                pdb_path = Path(f.name)
                
                # Save
                structure.save_pdb(pdb_path)
                
                # Load
                loaded = ProteinStructure.from_pdb(pdb_path)
                
                return loaded
        
        result = benchmark(large_pdb_roundtrip)
        assert result is not None
        assert result.coordinates.shape == (1000, 3)


class TestCachePerformance:
    """Benchmark caching operations."""
    
    def test_cache_write_speed(self, benchmark):
        """Benchmark cache write performance."""
        cache_manager = get_cache_manager()
        
        def write_to_cache():
            for i in range(100):
                key = f"test_key_{i}"
                value = {
                    'data': torch.randn(50, 3).tolist(),
                    'metadata': {'iteration': i, 'timestamp': time.time()}
                }
                cache_manager.set(key, value)
            return 100
        
        result = benchmark(write_to_cache)
        assert result == 100
    
    def test_cache_read_speed(self, benchmark):
        """Benchmark cache read performance."""
        cache_manager = get_cache_manager()
        
        # Populate cache
        test_data = {}
        for i in range(100):
            key = f"read_test_key_{i}"
            value = {
                'data': torch.randn(50, 3).tolist(),
                'metadata': {'iteration': i}
            }
            cache_manager.set(key, value)
            test_data[key] = value
        
        def read_from_cache():
            retrieved = 0
            for key in test_data.keys():
                if cache_manager.get(key) is not None:
                    retrieved += 1
            return retrieved
        
        result = benchmark(read_from_cache)
        assert result == 100


class TestMemoryPerformance:
    """Benchmark memory usage."""
    
    def test_memory_efficient_design(self, benchmark):
        """Test memory efficiency of design operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        def memory_efficient_design():
            initial_memory = process.memory_info().rss
            
            constraints = Constraints()
            constraints.add_constraint(BindingSiteConstraint(
                residues=[10, 15, 20], ligand="ATP"
            ))
            
            designer = ProteinDesigner(operator_type="deeponet")
            
            # Generate multiple structures
            structures = []
            for _ in range(5):
                structure = designer.generate(
                    constraints=constraints,
                    length=100
                )
                structures.append(structure)
            
            peak_memory = process.memory_info().rss
            memory_usage_mb = (peak_memory - initial_memory) / (1024 * 1024)
            
            return memory_usage_mb
        
        result = benchmark(memory_efficient_design)
        # Memory usage should be reasonable (less than 500MB for this test)
        assert result < 500
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_efficiency(self, benchmark):
        """Test GPU memory efficiency."""
        def gpu_memory_test():
            initial_gpu_memory = torch.cuda.memory_allocated()
            
            constraints = Constraints()
            constraints.add_constraint(BindingSiteConstraint(
                residues=[10, 15, 20], ligand="ATP"
            ))
            
            designer = ProteinDesigner(
                operator_type="deeponet",
                device="cuda"
            )
            
            # Generate structure on GPU
            structure = designer.generate(
                constraints=constraints,
                length=100
            )
            
            peak_gpu_memory = torch.cuda.max_memory_allocated()
            gpu_memory_used_mb = (peak_gpu_memory - initial_gpu_memory) / (1024 * 1024)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            return gpu_memory_used_mb
        
        result = benchmark(gpu_memory_test)
        # GPU memory usage should be reasonable
        assert result < 1000  # Less than 1GB


class TestConcurrentPerformance:
    """Benchmark concurrent operations."""
    
    def test_concurrent_design(self, benchmark):
        """Benchmark concurrent protein design."""
        import threading
        import concurrent.futures
        
        def design_single_protein(protein_id):
            constraints = Constraints()
            constraints.add_constraint(BindingSiteConstraint(
                residues=[10, 15, 20], ligand=f"LIG_{protein_id}"
            ))
            
            designer = ProteinDesigner(operator_type="deeponet")
            structure = designer.generate(
                constraints=constraints,
                length=50
            )
            return protein_id, structure
        
        def concurrent_design():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(design_single_protein, i)
                    for i in range(10)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    protein_id, structure = future.result()
                    results.append((protein_id, structure))
                
                return len(results)
        
        result = benchmark(concurrent_design)
        assert result == 10


# Performance regression tests
class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_design_performance_baseline(self, benchmark):
        """Establish baseline performance for design operations."""
        def baseline_design():
            constraints = Constraints()
            constraints.add_constraint(BindingSiteConstraint(
                residues=[10, 15, 20], ligand="ATP"
            ))
            
            designer = ProteinDesigner(operator_type="deeponet")
            start_time = time.time()
            
            structure = designer.generate(
                constraints=constraints,
                length=100
            )
            
            end_time = time.time()
            return end_time - start_time
        
        duration = benchmark(baseline_design)
        
        # Performance baseline: should complete in under 30 seconds
        assert duration < 30.0, f"Design took {duration:.2f}s, expected < 30s"
    
    def test_optimization_performance_baseline(self, benchmark):
        """Establish baseline performance for optimization."""
        def baseline_optimization():
            coordinates = torch.randn(100, 3) * 10
            constraints = Constraints()
            structure = ProteinStructure(coordinates, constraints)
            
            designer = ProteinDesigner()
            start_time = time.time()
            
            optimized = designer.optimize(
                initial_structure=structure,
                iterations=100
            )
            
            end_time = time.time()
            return end_time - start_time
        
        duration = benchmark(baseline_optimization)
        
        # Performance baseline: should complete in under 60 seconds
        assert duration < 60.0, f"Optimization took {duration:.2f}s, expected < 60s"


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "--benchmark-only", "--benchmark-sort=mean", "-v"])