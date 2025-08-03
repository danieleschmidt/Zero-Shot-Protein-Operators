"""
Pytest configuration and fixtures for protein operators testing.

Provides shared fixtures, test configuration, and utilities for
comprehensive testing of the protein design framework.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import json

import pytest
import torch
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from protein_operators.database.connection import Base, DatabaseManager
from protein_operators.database.models import ProteinDesign, Experiment, Constraint
from protein_operators.constraints import Constraints, BindingSiteConstraint
from protein_operators.core import ProteinDesigner
from protein_operators.structure import ProteinStructure


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="protein_operators_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_database() -> Generator[DatabaseManager, None, None]:
    """Create in-memory test database."""
    db = DatabaseManager(
        database_url="sqlite:///:memory:",
        echo=False
    )
    
    # Create all tables
    db.create_tables()
    
    yield db
    
    # Cleanup
    db.close()


@pytest.fixture
def db_session(test_database):
    """Provide database session for testing."""
    with test_database.session() as session:
        yield session


@pytest.fixture
def sample_protein_coordinates() -> torch.Tensor:
    """Generate sample protein coordinates."""
    # Create a simple linear chain of 50 CA atoms
    coords = torch.zeros(50, 3)
    for i in range(50):
        coords[i, 0] = i * 3.8  # CA-CA distance
        coords[i, 1] = 2.0 * torch.sin(torch.tensor(i * 0.2))  # Slight curve
        coords[i, 2] = 1.0 * torch.cos(torch.tensor(i * 0.3))  # Twist
    
    return coords


@pytest.fixture
def sample_constraints() -> Constraints:
    """Create sample constraints for testing."""
    constraints = Constraints()
    
    # Add binding site constraint
    constraints.add_constraint(BindingSiteConstraint(
        name="ATP_binding_site",
        residues=[10, 15, 20],
        ligand="ATP",
        affinity_nm=100.0,
        weight=1.0
    ))
    
    # Add secondary structure constraint
    from protein_operators.constraints.structural import SecondaryStructureConstraint
    constraints.add_constraint(SecondaryStructureConstraint(
        name="helix_1",
        start=5,
        end=15,
        ss_type="helix",
        confidence=0.9,
        weight=0.8
    ))
    
    return constraints


@pytest.fixture
def sample_protein_structure(sample_protein_coordinates, sample_constraints) -> ProteinStructure:
    """Create sample protein structure."""
    return ProteinStructure(
        coordinates=sample_protein_coordinates,
        constraints=sample_constraints,
        sequence="A" * 50,  # Poly-alanine
        metadata={"source": "test"}
    )


@pytest.fixture
def sample_experiment(db_session) -> Experiment:
    """Create sample experiment in database."""
    experiment = Experiment(
        name="Test Experiment",
        description="Test experiment for unit tests",
        objective="Test protein design",
        parameters={"operator_type": "deeponet", "hidden_dim": 256},
        status="running"
    )
    
    db_session.add(experiment)
    db_session.commit()
    
    return experiment


@pytest.fixture
def sample_protein_design(db_session, sample_experiment, sample_protein_coordinates) -> ProteinDesign:
    """Create sample protein design in database."""
    design = ProteinDesign(
        name="Test Design",
        description="Test protein design",
        sequence="A" * 50,
        length=50,
        operator_type="deeponet",
        coordinates=sample_protein_coordinates.tolist(),
        experiment_id=sample_experiment.id,
        status="generated"
    )
    
    db_session.add(design)
    db_session.commit()
    
    return design


@pytest.fixture
def mock_designer() -> ProteinDesigner:
    """Create mock protein designer for testing."""
    return ProteinDesigner(
        operator_type="deeponet",
        device="cpu"  # Use CPU for testing
    )


@pytest.fixture
def sample_pdb_content() -> str:
    """Sample PDB file content for testing."""
    return '''HEADER    TEST PROTEIN                            01-JAN-25   TEST            
TITLE     TEST PROTEIN STRUCTURE                                          
ATOM      1  CA  ALA A   1      -5.123   2.456   1.789  1.00 20.00           C  
ATOM      2  CA  ALA A   2      -1.323   2.456   1.789  1.00 20.00           C  
ATOM      3  CA  ALA A   3       2.477   2.456   1.789  1.00 20.00           C  
ATOM      4  CA  ALA A   4       6.277   2.456   1.789  1.00 20.00           C  
ATOM      5  CA  ALA A   5      10.077   2.456   1.789  1.00 20.00           C  
END                                                                             
'''


@pytest.fixture
def temp_pdb_file(test_data_dir, sample_pdb_content) -> Path:
    """Create temporary PDB file for testing."""
    pdb_file = test_data_dir / "test_protein.pdb"
    pdb_file.write_text(sample_pdb_content)
    return pdb_file


@pytest.fixture
def sample_design_config() -> Dict[str, Any]:
    """Sample design configuration."""
    return {
        "operator_type": "deeponet",
        "hidden_dim": 256,
        "num_basis": 512,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "max_length": 200,
        "physics_weight": 0.1
    }


@pytest.fixture
def sample_validation_results() -> Dict[str, float]:
    """Sample validation results."""
    return {
        "stereochemistry_score": 0.85,
        "clash_score": 0.92,
        "ramachandran_score": 0.78,
        "constraint_satisfaction": 0.89,
        "overall_score": 0.86
    }


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may require GPU or long computation)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "experimental: marks experimental tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    # Add slow marker to tests that run for more than a certain duration
    # This would need custom implementation to detect slow tests
    
    # Skip GPU tests if CUDA is not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip network tests if running offline
    if os.getenv("PYTEST_OFFLINE", "false").lower() == "true":
        skip_network = pytest.mark.skip(reason="Running in offline mode")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_random_coordinates(length: int, seed: int = 42) -> torch.Tensor:
        """Generate random but reasonable protein coordinates."""
        torch.manual_seed(seed)
        
        # Start with extended chain
        coords = torch.zeros(length, 3)
        for i in range(length):
            coords[i, 0] = i * 3.8
        
        # Add random perturbations
        noise = torch.randn(length, 3) * 0.5
        coords += noise
        
        return coords
    
    @staticmethod
    def generate_test_sequence(length: int, composition: Dict[str, float] = None) -> str:
        """Generate test amino acid sequence."""
        if composition is None:
            composition = {"A": 0.2, "G": 0.2, "L": 0.15, "V": 0.15, "S": 0.1, "T": 0.1, "F": 0.1}
        
        amino_acids = list(composition.keys())
        weights = list(composition.values())
        
        np.random.seed(42)
        sequence = np.random.choice(amino_acids, size=length, p=weights)
        
        return "".join(sequence)
    
    @staticmethod
    def create_test_constraints(protein_length: int) -> Constraints:
        """Create realistic test constraints."""
        constraints = Constraints()
        
        # Binding site in middle third
        binding_start = protein_length // 3
        binding_residues = list(range(binding_start, binding_start + 5))
        constraints.add_constraint(BindingSiteConstraint(
            name="test_binding_site",
            residues=binding_residues,
            ligand="test_ligand",
            affinity_nm=50.0
        ))
        
        # Secondary structure elements
        from protein_operators.constraints.structural import SecondaryStructureConstraint
        constraints.add_constraint(SecondaryStructureConstraint(
            name="test_helix",
            start=5,
            end=20,
            ss_type="helix",
            confidence=0.8
        ))
        
        constraints.add_constraint(SecondaryStructureConstraint(
            name="test_sheet",
            start=protein_length - 15,
            end=protein_length - 5,
            ss_type="sheet",
            confidence=0.7
        ))
        
        return constraints


# Export test utilities
@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator."""
    return TestDataGenerator()


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    import psutil
    import torch
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_gpu_memory = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if torch.cuda.is_available():
                self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results = {
                "elapsed_time": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory
            }
            
            if torch.cuda.is_available() and self.start_gpu_memory is not None:
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                results["gpu_memory_delta"] = end_gpu_memory - self.start_gpu_memory
            
            return results
    
    return PerformanceMonitor()


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set environment variables for testing
    os.environ["APP_ENV"] = "testing"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise in tests
    
    # Set PyTorch to use single thread for deterministic results
    torch.set_num_threads(1)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    yield
    
    # Cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()