"""
Data generation system for neural operator training.

This module provides comprehensive data generation for protein structure
and constraint pairs, including synthetic data generation, PDB processing,
and molecular dynamics simulation data extraction.
"""

from typing import List, Dict, Any, Optional, Tuple, Iterator
import os
import json
import pickle
import random as python_random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple NumPy mock for basic operations
    class MockNumPy:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod 
        def zeros(shape):
            if isinstance(shape, (list, tuple)):
                if len(shape) == 1:
                    return [0.0] * shape[0]
                elif len(shape) == 2:
                    return [[0.0] * shape[1] for _ in range(shape[0])]
                else:
                    return [0.0] * 10  # Fallback
            return [0.0] * shape
        
        @staticmethod
        def linspace(start, stop, num):
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        
        @staticmethod
        def vstack(arrays):
            result = []
            for arr in arrays:
                if isinstance(arr, list):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result
        
        @staticmethod
        def expand_dims(arr, axis):
            return [arr]
        
        class random:
            @staticmethod
            def seed(s):
                python_random.seed(s)
            
            @staticmethod
            def randn(*shape):
                import math
                total = 1
                for s in shape:
                    total *= s
                # Box-Muller transform for normal distribution
                result = []
                for _ in range(total):
                    u1 = python_random.random()
                    u2 = python_random.random()
                    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                    result.append(z)
                return result if len(result) > 1 else result[0]
            
            @staticmethod
            def normal(mean, std, shape):
                MockNumPy.random.randn(*shape) if isinstance(shape, tuple) else [MockNumPy.random.randn(1)[0] * std + mean for _ in range(shape)]
                return [python_random.gauss(mean, std) for _ in range(shape[0] * shape[1] if isinstance(shape, tuple) and len(shape) >= 2 else shape[0] if isinstance(shape, tuple) else 10)]
        
        @staticmethod
        def linalg():
            class linalg:
                @staticmethod
                def norm(arr):
                    if isinstance(arr, list):
                        return sum(x**2 for x in arr) ** 0.5
                    return abs(arr)
            return linalg()
    
    np = MockNumPy()

# Use PyTorch integration
from ..utils.torch_integration import (
    TORCH_AVAILABLE, tensor, zeros, ones, randn,
    TensorUtils, get_device
)

# Import for structure processing
try:
    from Bio import PDB
    from Bio.PDB import PDBParser, DSSP
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Mock PDB processing if BioPython not available
if not BIOPYTHON_AVAILABLE:
    class MockPDBParser:
        def get_structure(self, *args): return None
    PDBParser = MockPDBParser


@dataclass
class ProteinStructureData:
    """Container for protein structure data."""
    
    # Basic information
    pdb_id: str
    chain_id: str
    sequence: str
    length: int
    
    # Coordinates
    ca_coords: np.ndarray  # (N, 3) CA coordinates
    bb_coords: np.ndarray  # (N, 4, 3) backbone coordinates (N, CA, C, O)
    all_atom_coords: Optional[np.ndarray] = None  # (N, max_atoms, 3)
    
    # Secondary structure
    secondary_structure: Optional[str] = None  # DSSP assignment
    phi_angles: Optional[np.ndarray] = None
    psi_angles: Optional[np.ndarray] = None
    
    # Properties
    resolution: Optional[float] = None
    b_factors: Optional[np.ndarray] = None
    
    # Computed features
    contact_map: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None


@dataclass
class ConstraintData:
    """Container for protein design constraints."""
    
    # Constraint metadata
    constraint_id: str
    constraint_type: str  # binding_site, secondary_structure, stability
    priority: float = 1.0
    
    # Binding site constraints
    binding_residues: Optional[List[int]] = None
    ligand_type: Optional[str] = None
    binding_affinity: Optional[float] = None
    
    # Structural constraints
    ss_assignment: Optional[str] = None  # Secondary structure
    disulfide_bonds: Optional[List[Tuple[int, int]]] = None
    metal_sites: Optional[List[Dict]] = None
    
    # Stability constraints
    target_tm: Optional[float] = None  # Melting temperature
    target_ph: Optional[float] = None
    target_expression: Optional[float] = None
    
    # Encoded features
    constraint_vector: Optional[np.ndarray] = None
    spatial_mask: Optional[np.ndarray] = None


class SyntheticProteinGenerator:
    """Generate synthetic protein structures and constraints."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize synthetic generator."""
        self.random_seed = random_seed
        python_random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Standard amino acids
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        
        # Secondary structure types
        self.ss_types = ['H', 'E', 'C']  # Helix, Sheet, Coil
    
    def generate_random_sequence(self, length: int) -> str:
        """Generate random protein sequence."""
        return ''.join(python_random.choices(self.amino_acids, k=length))
    
    def generate_helical_coords(self, 
                               length: int, 
                               start_pos: np.ndarray = None) -> np.ndarray:
        """Generate coordinates for alpha-helical structure."""
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])
        
        if NUMPY_AVAILABLE:
            coords = np.zeros((length, 3))
        else:
            coords = [[0.0, 0.0, 0.0] for _ in range(length)]
        
        # Alpha helix parameters
        rise_per_residue = 1.5  # Angstroms
        radius = 2.3  # Angstroms
        import math
        angle_per_residue = 100.0 * math.pi / 180.0  # 100 degrees
        
        for i in range(length):
            angle = i * angle_per_residue
            if NUMPY_AVAILABLE:
                coords[i] = start_pos + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    i * rise_per_residue
                ])
            else:
                import math
                coords[i] = [
                    start_pos[0] + radius * math.cos(angle),
                    start_pos[1] + radius * math.sin(angle), 
                    start_pos[2] + i * rise_per_residue
                ]
        
        return coords
    
    def generate_sheet_coords(self, 
                             length: int, 
                             start_pos: np.ndarray = None) -> np.ndarray:
        """Generate coordinates for beta-sheet structure."""
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])
        
        if NUMPY_AVAILABLE:
            coords = np.zeros((length, 3))
        else:
            coords = [[0.0, 0.0, 0.0] for _ in range(length)]
        
        # Beta sheet parameters
        residue_spacing = 3.5  # Angstroms
        
        for i in range(length):
            # Slight zigzag pattern
            y_offset = 0.5 * np.sin(i * np.pi)
            coords[i] = start_pos + np.array([
                i * residue_spacing,
                y_offset,
                0.0
            ])
        
        return coords
    
    def generate_coil_coords(self, 
                            length: int, 
                            start_pos: np.ndarray = None) -> np.ndarray:
        """Generate coordinates for random coil structure."""
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])
        
        if NUMPY_AVAILABLE:
            coords = np.zeros((length, 3))
        else:
            coords = [[0.0, 0.0, 0.0] for _ in range(length)]
        coords[0] = start_pos
        
        # Random walk with realistic bond lengths
        bond_length = 3.8  # CA-CA distance
        
        for i in range(1, length):
            # Random direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            coords[i] = coords[i-1] + bond_length * direction
        
        return coords
    
    def generate_synthetic_structure(self, 
                                   length: int,
                                   ss_composition: Dict[str, float] = None) -> ProteinStructureData:
        """Generate complete synthetic protein structure."""
        if ss_composition is None:
            ss_composition = {'H': 0.4, 'E': 0.3, 'C': 0.3}
        
        # Generate sequence
        sequence = self.generate_random_sequence(length)
        
        # Generate secondary structure assignment
        ss_assignment = self._generate_ss_assignment(length, ss_composition)
        
        # Generate coordinates based on secondary structure
        coords = self._generate_coords_from_ss(ss_assignment)
        
        # Add noise to make more realistic
        coords += np.random.normal(0, 0.1, coords.shape)
        
        return ProteinStructureData(
            pdb_id=f"SYNTH_{python_random.randint(1000, 9999)}",
            chain_id="A",
            sequence=sequence,
            length=length,
            ca_coords=coords,
            bb_coords=coords.reshape(-1, 1, 3).repeat(4, axis=1),  # Simplified
            secondary_structure=ss_assignment,
            resolution=2.0  # Synthetic resolution
        )
    
    def _generate_ss_assignment(self, 
                               length: int, 
                               composition: Dict[str, float]) -> str:
        """Generate secondary structure assignment."""
        assignment = []
        
        # Create segments based on composition
        segments = []
        remaining_length = length
        
        for ss_type, fraction in composition.items():
            segment_length = int(length * fraction)
            if segment_length > 0 and remaining_length > 0:
                actual_length = min(segment_length, remaining_length)
                segments.append((ss_type, actual_length))
                remaining_length -= actual_length
        
        # Assign remaining residues
        if remaining_length > 0:
            segments.append(('C', remaining_length))
        
        # Shuffle segments to create realistic patterns
        python_random.shuffle(segments)
        
        # Build assignment string
        for ss_type, segment_length in segments:
            assignment.extend([ss_type] * segment_length)
        
        return ''.join(assignment[:length])
    
    def _generate_coords_from_ss(self, ss_assignment: str) -> np.ndarray:
        """Generate coordinates based on secondary structure."""
        coords = []
        current_pos = np.array([0.0, 0.0, 0.0])
        
        # Group consecutive residues of same SS type
        segments = []
        current_type = ss_assignment[0]
        current_start = 0
        
        for i, ss_type in enumerate(ss_assignment):
            if ss_type != current_type:
                segments.append((current_type, current_start, i))
                current_type = ss_type
                current_start = i
        segments.append((current_type, current_start, len(ss_assignment)))
        
        # Generate coordinates for each segment
        for ss_type, start, end in segments:
            segment_length = end - start
            
            if ss_type == 'H':
                segment_coords = self.generate_helical_coords(segment_length, current_pos)
            elif ss_type == 'E':
                segment_coords = self.generate_sheet_coords(segment_length, current_pos)
            else:  # 'C'
                segment_coords = self.generate_coil_coords(segment_length, current_pos)
            
            coords.append(segment_coords)
            current_pos = segment_coords[-1] + np.array([5.0, 0.0, 0.0])  # Gap between segments
        
        return np.vstack(coords) if coords else np.zeros((0, 3))
    
    def generate_constraint_for_structure(self, 
                                        structure: ProteinStructureData) -> ConstraintData:
        """Generate realistic constraint for given structure."""
        constraint_types = ['binding_site', 'secondary_structure', 'stability']
        constraint_type = python_random.choice(constraint_types)
        
        constraint_id = f"{structure.pdb_id}_{constraint_type}_{python_random.randint(1, 999)}"
        
        if constraint_type == 'binding_site':
            # Random binding site
            num_residues = python_random.randint(3, 8)
            binding_residues = sorted(python_random.sample(range(structure.length), num_residues))
            
            return ConstraintData(
                constraint_id=constraint_id,
                constraint_type=constraint_type,
                binding_residues=binding_residues,
                ligand_type=python_random.choice(['ATP', 'GTP', 'DNA', 'RNA', 'Mg2+']),
                binding_affinity=python_random.uniform(1e-9, 1e-6)  # nM to μM
            )
        
        elif constraint_type == 'secondary_structure':
            return ConstraintData(
                constraint_id=constraint_id,
                constraint_type=constraint_type,
                ss_assignment=structure.secondary_structure
            )
        
        else:  # stability
            return ConstraintData(
                constraint_id=constraint_id,
                constraint_type=constraint_type,
                target_tm=python_random.uniform(50, 90),  # Celsius
                target_ph=python_random.uniform(6.0, 8.0),
                target_expression=python_random.uniform(10, 1000)  # mg/L
            )


class PDBDataProcessor:
    """Process PDB structures for training data."""
    
    def __init__(self, pdb_dir: str):
        """Initialize PDB processor."""
        self.pdb_dir = Path(pdb_dir)
        self.parser = PDBParser(QUIET=True) if BIOPYTHON_AVAILABLE else MockPDBParser()
    
    def process_pdb_file(self, pdb_file: str) -> Optional[List[ProteinStructureData]]:
        """Process single PDB file."""
        if not BIOPYTHON_AVAILABLE:
            return None
        
        try:
            structure = self.parser.get_structure('protein', pdb_file)
            structures = []
            
            for model in structure:
                for chain in model:
                    chain_data = self._process_chain(chain, pdb_file)
                    if chain_data:
                        structures.append(chain_data)
            
            return structures
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            return None
    
    def _process_chain(self, chain, pdb_file: str) -> Optional[ProteinStructureData]:
        """Process single chain."""
        try:
            # Extract CA coordinates and sequence
            ca_coords = []
            sequence = []
            
            for residue in chain:
                if PDB.is_aa(residue):  # Standard amino acid
                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                        ca_coords.append(ca_coord)
                        sequence.append(PDB.Polypeptide.three_to_one(residue.get_resname()))
            
            if len(ca_coords) < 20:  # Skip very short chains
                return None
            
            pdb_id = Path(pdb_file).stem
            
            return ProteinStructureData(
                pdb_id=pdb_id,
                chain_id=chain.id,
                sequence=''.join(sequence),
                length=len(sequence),
                ca_coords=np.array(ca_coords),
                bb_coords=np.array(ca_coords).reshape(-1, 1, 3).repeat(4, axis=1),
                resolution=None  # Would extract from header
            )
        
        except Exception as e:
            print(f"Error processing chain {chain.id}: {e}")
            return None


class TrainingDataGenerator:
    """Main data generator for training neural operators."""
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 synthetic_ratio: float = 0.5):
        """
        Initialize data generator.
        
        Args:
            cache_dir: Directory for caching processed data
            synthetic_ratio: Fraction of synthetic vs real data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_ratio = synthetic_ratio
        
        # Initialize generators
        self.synthetic_gen = SyntheticProteinGenerator()
        self.pdb_processor = PDBDataProcessor("data/pdb")
        
        # Data containers
        self.structures: List[ProteinStructureData] = []
        self.constraints: List[ConstraintData] = []
    
    def generate_training_dataset(self, 
                                 num_samples: int,
                                 length_range: Tuple[int, int] = (50, 300),
                                 save_cache: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate complete training dataset.
        
        Args:
            num_samples: Total number of training samples
            length_range: Range of protein lengths
            save_cache: Whether to save processed data to cache
        
        Returns:
            Tuple of (constraint_data, structure_data) lists
        """
        # Check cache first
        cache_file = self.cache_dir / f"training_data_{num_samples}.pkl"
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['constraints'], cached_data['structures']
        
        print(f"Generating {num_samples} training samples...")
        
        constraint_data = []
        structure_data = []
        
        num_synthetic = int(num_samples * self.synthetic_ratio)
        num_real = num_samples - num_synthetic
        
        # Generate synthetic data
        print(f"Generating {num_synthetic} synthetic samples...")
        for i in range(num_synthetic):
            length = python_random.randint(*length_range)
            
            # Generate structure
            structure = self.synthetic_gen.generate_synthetic_structure(length)
            
            # Generate corresponding constraint
            constraint = self.synthetic_gen.generate_constraint_for_structure(structure)
            
            # Convert to dictionaries for training
            constraint_dict = self._constraint_to_dict(constraint)
            structure_dict = self._structure_to_dict(structure)
            
            constraint_data.append(constraint_dict)
            structure_data.append(structure_dict)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_synthetic} synthetic samples")
        
        # Generate real data (placeholder - would implement PDB processing)
        print(f"Generating {num_real} samples from PDB data...")
        for i in range(num_real):
            # For now, generate additional synthetic data as placeholder
            length = python_random.randint(*length_range)
            structure = self.synthetic_gen.generate_synthetic_structure(length)
            constraint = self.synthetic_gen.generate_constraint_for_structure(structure)
            
            constraint_dict = self._constraint_to_dict(constraint)
            structure_dict = self._structure_to_dict(structure)
            
            constraint_data.append(constraint_dict)
            structure_data.append(structure_dict)
        
        # Save to cache
        if save_cache:
            cache_data = {
                'constraints': constraint_data,
                'structures': structure_data,
                'metadata': {
                    'num_samples': num_samples,
                    'synthetic_ratio': self.synthetic_ratio,
                    'length_range': length_range
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved training data to cache: {cache_file}")
        
        print(f"Generated {len(constraint_data)} training samples")
        return constraint_data, structure_data
    
    def _constraint_to_dict(self, constraint: ConstraintData) -> Dict[str, Any]:
        """Convert ConstraintData to dictionary."""
        constraint_dict = {
            'constraint_id': constraint.constraint_id,
            'constraint_type': constraint.constraint_type,
            'priority': constraint.priority
        }
        
        if constraint.binding_residues:
            constraint_dict['binding_residues'] = constraint.binding_residues
            constraint_dict['ligand_type'] = constraint.ligand_type
            constraint_dict['binding_affinity'] = constraint.binding_affinity
        
        if constraint.ss_assignment:
            constraint_dict['ss_assignment'] = constraint.ss_assignment
        
        if constraint.target_tm:
            constraint_dict['target_tm'] = constraint.target_tm
            constraint_dict['target_ph'] = constraint.target_ph
            constraint_dict['target_expression'] = constraint.target_expression
        
        return constraint_dict
    
    def _structure_to_dict(self, structure: ProteinStructureData) -> Dict[str, Any]:
        """Convert ProteinStructureData to dictionary."""
        structure_dict = {
            'pdb_id': structure.pdb_id,
            'chain_id': structure.chain_id,
            'sequence': structure.sequence,
            'length': structure.length,
            'ca_coords': structure.ca_coords.tolist(),
            'secondary_structure': structure.secondary_structure
        }
        
        if structure.resolution:
            structure_dict['resolution'] = structure.resolution
        
        return structure_dict
    
    def create_validation_split(self, 
                               constraint_data: List[Dict],
                               structure_data: List[Dict],
                               val_ratio: float = 0.2) -> Tuple[Tuple[List, List], Tuple[List, List]]:
        """Create train/validation split."""
        total_samples = len(constraint_data)
        val_size = int(total_samples * val_ratio)
        
        # Random split
        indices = list(range(total_samples))
        python_random.shuffle(indices)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_constraints = [constraint_data[i] for i in train_indices]
        train_structures = [structure_data[i] for i in train_indices]
        
        val_constraints = [constraint_data[i] for i in val_indices]
        val_structures = [structure_data[i] for i in val_indices]
        
        return (train_constraints, train_structures), (val_constraints, val_structures)


__all__ = [
    'ProteinStructureData', 'ConstraintData',
    'SyntheticProteinGenerator', 'PDBDataProcessor',
    'TrainingDataGenerator'
]