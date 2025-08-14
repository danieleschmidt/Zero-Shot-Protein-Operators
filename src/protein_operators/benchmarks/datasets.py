"""
Benchmark datasets for protein neural operator evaluation.

This module provides standardized datasets for evaluating neural operator
performance on protein folding and design tasks.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    import mock_torch as torch
    Dataset = object
    DataLoader = object


@dataclass
class ProteinStructure:
    """Container for protein structure data."""
    sequence: str
    coordinates: torch.Tensor  # [N, 3] atomic coordinates
    secondary_structure: Optional[str] = None
    pdb_id: Optional[str] = None
    chain_id: Optional[str] = None
    resolution: Optional[float] = None
    organism: Optional[str] = None
    classification: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.sequence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sequence': self.sequence,
            'coordinates': self.coordinates.tolist() if isinstance(self.coordinates, torch.Tensor) else self.coordinates,
            'secondary_structure': self.secondary_structure,
            'pdb_id': self.pdb_id,
            'chain_id': self.chain_id,
            'resolution': self.resolution,
            'organism': self.organism,
            'classification': self.classification
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProteinStructure':
        """Create from dictionary."""
        coords = data['coordinates']
        if isinstance(coords, list):
            coords = torch.tensor(coords, dtype=torch.float32)
        
        return cls(
            sequence=data['sequence'],
            coordinates=coords,
            secondary_structure=data.get('secondary_structure'),
            pdb_id=data.get('pdb_id'),
            chain_id=data.get('chain_id'),
            resolution=data.get('resolution'),
            organism=data.get('organism'),
            classification=data.get('classification')
        )


class SyntheticProteinDataset(Dataset):
    """
    Synthetic protein dataset for controlled benchmarking.
    
    Generates synthetic protein structures with known properties
    for testing neural operator capabilities.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        min_length: int = 50,
        max_length: int = 300,
        structure_types: List[str] = None,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            n_samples: Number of synthetic structures
            min_length: Minimum protein length
            max_length: Maximum protein length
            structure_types: Types of structures to generate
            noise_level: Amount of noise to add to coordinates
            seed: Random seed
        """
        self.n_samples = n_samples
        self.min_length = min_length
        self.max_length = max_length
        self.structure_types = structure_types or ['helix', 'sheet', 'coil', 'mixed']
        self.noise_level = noise_level
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.structures = self._generate_structures()
    
    def _generate_structures(self) -> List[ProteinStructure]:
        """Generate synthetic protein structures."""
        structures = []
        
        for i in range(self.n_samples):
            # Random length
            length = np.random.randint(self.min_length, self.max_length + 1)
            
            # Random structure type
            structure_type = np.random.choice(self.structure_types)
            
            # Generate sequence
            sequence = self._generate_sequence(length)
            
            # Generate coordinates based on structure type
            coordinates = self._generate_coordinates(length, structure_type)
            
            # Add noise
            coordinates += torch.randn_like(coordinates) * self.noise_level
            
            # Generate secondary structure
            ss = self._generate_secondary_structure(length, structure_type)
            
            structure = ProteinStructure(
                sequence=sequence,
                coordinates=coordinates,
                secondary_structure=ss,
                pdb_id=f"SYN_{i:04d}",
                classification=structure_type
            )
            
            structures.append(structure)
        
        return structures
    
    def _generate_sequence(self, length: int) -> str:
        """Generate random amino acid sequence."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequence = ''.join(np.random.choice(list(amino_acids), length))
        return sequence
    
    def _generate_coordinates(self, length: int, structure_type: str) -> torch.Tensor:
        """Generate coordinates based on structure type."""
        if structure_type == 'helix':
            return self._generate_helix(length)
        elif structure_type == 'sheet':
            return self._generate_sheet(length)
        elif structure_type == 'coil':
            return self._generate_coil(length)
        elif structure_type == 'mixed':
            return self._generate_mixed_structure(length)
        else:
            return self._generate_coil(length)
    
    def _generate_helix(self, length: int) -> torch.Tensor:
        """Generate alpha-helix coordinates."""
        # Alpha helix parameters
        radius = 2.3  # Angstroms
        pitch = 1.5   # Angstroms per residue
        angle_per_residue = 100 * np.pi / 180  # degrees
        
        coordinates = torch.zeros(length, 3)
        
        for i in range(length):
            angle = i * angle_per_residue
            z = i * pitch
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            coordinates[i] = torch.tensor([x, y, z], dtype=torch.float32)
        
        return coordinates
    
    def _generate_sheet(self, length: int) -> torch.Tensor:
        """Generate beta-sheet coordinates."""
        # Beta sheet parameters
        strand_length = 3.2  # Angstroms between residues
        
        coordinates = torch.zeros(length, 3)
        
        # Zigzag pattern for extended conformation
        for i in range(length):
            x = i * strand_length
            y = 0.5 * np.sin(i * np.pi) if i % 2 == 0 else -0.5 * np.sin(i * np.pi)
            z = 0
            
            coordinates[i] = torch.tensor([x, y, z], dtype=torch.float32)
        
        return coordinates
    
    def _generate_coil(self, length: int) -> torch.Tensor:
        """Generate random coil coordinates."""
        # Random walk with constrained bond lengths
        bond_length = 3.8  # Angstroms
        
        coordinates = torch.zeros(length, 3)
        
        for i in range(1, length):
            # Random direction
            direction = torch.randn(3)
            direction = direction / torch.norm(direction)
            
            # Next position
            coordinates[i] = coordinates[i-1] + bond_length * direction
        
        return coordinates
    
    def _generate_mixed_structure(self, length: int) -> torch.Tensor:
        """Generate mixed secondary structure."""
        coordinates = torch.zeros(length, 3)
        
        # Divide into segments
        n_segments = np.random.randint(2, 5)
        segment_lengths = np.random.multinomial(length, [1/n_segments] * n_segments)
        
        current_pos = 0
        last_coord = torch.zeros(3)
        
        for i, seg_len in enumerate(segment_lengths):
            if seg_len == 0:
                continue
            
            # Random structure type for segment
            seg_type = np.random.choice(['helix', 'sheet', 'coil'])
            
            # Generate segment
            if seg_type == 'helix':
                seg_coords = self._generate_helix(seg_len)
            elif seg_type == 'sheet':
                seg_coords = self._generate_sheet(seg_len)
            else:
                seg_coords = self._generate_coil(seg_len)
            
            # Connect to previous segment
            if current_pos > 0:
                offset = last_coord - seg_coords[0]
                seg_coords += offset
            
            # Add to full structure
            end_pos = current_pos + seg_len
            coordinates[current_pos:end_pos] = seg_coords
            
            last_coord = coordinates[end_pos - 1]
            current_pos = end_pos
        
        return coordinates
    
    def _generate_secondary_structure(self, length: int, structure_type: str) -> str:
        """Generate secondary structure annotation."""
        if structure_type == 'helix':
            return 'H' * length
        elif structure_type == 'sheet':
            return 'E' * length
        elif structure_type == 'coil':
            return 'C' * length
        elif structure_type == 'mixed':
            # Random mix
            ss_chars = ['H', 'E', 'C']
            return ''.join(np.random.choice(ss_chars, length))
        else:
            return 'C' * length
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        structure = self.structures[idx]
        
        # Convert sequence to one-hot encoding
        sequence_encoded = self._encode_sequence(structure.sequence)
        
        return {
            'input': sequence_encoded,
            'target': structure.coordinates,
            'sequence': structure.sequence,
            'secondary_structure': structure.secondary_structure,
            'pdb_id': structure.pdb_id,
            'metadata': {
                'length': len(structure),
                'classification': structure.classification
            }
        }
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode amino acid sequence as one-hot."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        encoding = torch.zeros(len(sequence), len(amino_acids))
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoding[i, aa_to_idx[aa]] = 1.0
        
        return encoding


class CATHDataset(Dataset):
    """
    CATH database subset for protein structure benchmarking.
    
    Uses CATH (Class, Architecture, Topology, Homology) classification
    for systematic evaluation across protein families.
    """
    
    def __init__(
        self,
        data_dir: str = "data/cath",
        subset: str = "representative",
        max_resolution: float = 2.5,
        min_length: int = 30,
        max_length: int = 500
    ):
        """
        Initialize CATH dataset.
        
        Args:
            data_dir: Directory containing CATH data
            subset: Dataset subset ('representative', 'full', 'test')
            max_resolution: Maximum X-ray resolution
            min_length: Minimum protein length
            max_length: Maximum protein length
        """
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.max_resolution = max_resolution
        self.min_length = min_length
        self.max_length = max_length
        
        self.structures = self._load_structures()
    
    def _load_structures(self) -> List[ProteinStructure]:
        """Load CATH structures from data directory."""
        structures = []
        
        # Load from JSON file if available
        json_file = self.data_dir / f"cath_{self.subset}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                structure = ProteinStructure.from_dict(item)
                
                # Apply filters
                if (self.min_length <= len(structure) <= self.max_length and
                    (structure.resolution is None or structure.resolution <= self.max_resolution)):
                    structures.append(structure)
        else:
            # Generate synthetic CATH-like data for demonstration
            structures = self._generate_cath_like_data()
        
        return structures
    
    def _generate_cath_like_data(self) -> List[ProteinStructure]:
        """Generate CATH-like synthetic data for demonstration."""
        # CATH fold families (simplified)
        cath_folds = [
            ('1.10.8.10', 'Immunoglobulin-like'),
            ('2.60.40.10', 'NAD(P)-binding Rossmann-like Domain'),
            ('3.40.50.720', 'TIM barrel'),
            ('3.30.70.330', 'Ferredoxin fold'),
            ('1.20.58.10', 'Four Helix Bundle'),
            ('2.40.50.140', 'Nucleotide-binding domain'),
        ]
        
        structures = []
        n_per_fold = 20
        
        for fold_id, fold_desc in cath_folds:
            for i in range(n_per_fold):
                # Generate structure based on fold type
                if 'Helix' in fold_desc:
                    length = np.random.randint(80, 200)
                    structure_type = 'helix'
                elif 'barrel' in fold_desc.lower() or 'tim' in fold_desc.lower():
                    length = np.random.randint(200, 400)
                    structure_type = 'mixed'
                elif 'sheet' in fold_desc.lower() or 'domain' in fold_desc.lower():
                    length = np.random.randint(100, 250)
                    structure_type = 'sheet'
                else:
                    length = np.random.randint(50, 300)
                    structure_type = 'mixed'
                
                # Generate sequence and coordinates
                sequence = self._generate_sequence(length)
                coordinates = self._generate_coordinates_by_type(length, structure_type)
                
                structure = ProteinStructure(
                    sequence=sequence,
                    coordinates=coordinates,
                    pdb_id=f"CATH_{fold_id.replace('.', '_')}_{i:02d}",
                    classification=fold_desc,
                    resolution=np.random.uniform(1.0, self.max_resolution),
                    organism='synthetic'
                )
                
                structures.append(structure)
        
        return structures
    
    def _generate_sequence(self, length: int) -> str:
        """Generate amino acid sequence."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        return ''.join(np.random.choice(list(amino_acids), length))
    
    def _generate_coordinates_by_type(self, length: int, structure_type: str) -> torch.Tensor:
        """Generate coordinates based on structure type."""
        # Simplified coordinate generation
        if structure_type == 'helix':
            # Multiple helices
            n_helices = np.random.randint(3, 7)
            helix_lengths = np.random.multinomial(length, [1/n_helices] * n_helices)
            
            coordinates = torch.zeros(length, 3)
            current_pos = 0
            
            for i, h_len in enumerate(helix_lengths):
                if h_len == 0:
                    continue
                
                # Generate helix
                helix_coords = self._generate_helix_segment(h_len)
                
                # Random orientation and position
                rotation = torch.randn(3, 3)
                rotation, _ = torch.qr(rotation)  # Orthogonalize
                translation = torch.randn(3) * 10
                
                helix_coords = torch.matmul(helix_coords, rotation.T) + translation
                
                end_pos = current_pos + h_len
                coordinates[current_pos:end_pos] = helix_coords
                current_pos = end_pos
        
        elif structure_type == 'sheet':
            # Beta barrel or sheet
            coordinates = self._generate_sheet_structure(length)
        
        else:  # mixed
            coordinates = self._generate_mixed_fold(length)
        
        return coordinates
    
    def _generate_helix_segment(self, length: int) -> torch.Tensor:
        """Generate single helix segment."""
        radius = 2.3
        pitch = 1.5
        angle_per_residue = 100 * np.pi / 180
        
        coordinates = torch.zeros(length, 3)
        
        for i in range(length):
            angle = i * angle_per_residue
            z = i * pitch
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            coordinates[i] = torch.tensor([x, y, z], dtype=torch.float32)
        
        return coordinates
    
    def _generate_sheet_structure(self, length: int) -> torch.Tensor:
        """Generate beta-sheet structure."""
        # Simplified beta barrel
        n_strands = 8
        strand_length = length // n_strands
        barrel_radius = 15.0
        
        coordinates = torch.zeros(length, 3)
        current_pos = 0
        
        for strand in range(n_strands):
            angle = strand * 2 * np.pi / n_strands
            
            for res in range(strand_length):
                if current_pos >= length:
                    break
                
                x = barrel_radius * np.cos(angle)
                y = barrel_radius * np.sin(angle)
                z = res * 3.2 - strand_length * 1.6  # Center around z=0
                
                coordinates[current_pos] = torch.tensor([x, y, z], dtype=torch.float32)
                current_pos += 1
        
        return coordinates
    
    def _generate_mixed_fold(self, length: int) -> torch.Tensor:
        """Generate mixed secondary structure fold."""
        # TIM barrel-like structure
        n_units = 8  # 8 alpha/beta units
        unit_length = length // n_units
        
        coordinates = torch.zeros(length, 3)
        current_pos = 0
        
        for unit in range(n_units):
            angle = unit * 2 * np.pi / n_units
            
            # Beta strand
            strand_len = unit_length // 2
            for i in range(strand_len):
                if current_pos >= length:
                    break
                
                radius = 12.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = i * 3.2 - 5
                
                coordinates[current_pos] = torch.tensor([x, y, z], dtype=torch.float32)
                current_pos += 1
            
            # Alpha helix
            helix_len = unit_length - strand_len
            for i in range(helix_len):
                if current_pos >= length:
                    break
                
                helix_radius = 2.3
                helix_angle = angle + i * 100 * np.pi / 180
                helix_x = (12.0 + 5.0) * np.cos(angle) + helix_radius * np.cos(helix_angle)
                helix_y = (12.0 + 5.0) * np.sin(angle) + helix_radius * np.sin(helix_angle)
                helix_z = i * 1.5
                
                coordinates[current_pos] = torch.tensor([helix_x, helix_y, helix_z], dtype=torch.float32)
                current_pos += 1
        
        return coordinates
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        structure = self.structures[idx]
        
        # Encode sequence
        sequence_encoded = self._encode_sequence(structure.sequence)
        
        return {
            'input': sequence_encoded,
            'target': structure.coordinates,
            'sequence': structure.sequence,
            'pdb_id': structure.pdb_id,
            'metadata': {
                'length': len(structure),
                'classification': structure.classification,
                'resolution': structure.resolution,
                'organism': structure.organism
            }
        }
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode amino acid sequence."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        encoding = torch.zeros(len(sequence), len(amino_acids))
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoding[i, aa_to_idx[aa]] = 1.0
        
        return encoding


class ProteinBenchmarkDatasets:
    """
    Manager for protein benchmark datasets.
    
    Provides unified interface for loading and managing different
    protein structure datasets for benchmarking.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, name: str, **kwargs) -> Dataset:
        """
        Load a specific dataset.
        
        Args:
            name: Dataset name ('synthetic', 'cath', 'casp', etc.)
            **kwargs: Additional arguments for dataset
            
        Returns:
            Dataset instance
        """
        if name in self.datasets:
            return self.datasets[name]
        
        if name == 'synthetic':
            dataset = SyntheticProteinDataset(**kwargs)
        elif name == 'cath':
            dataset = CATHDataset(data_dir=str(self.data_dir / "cath"), **kwargs)
        elif name == 'casp':
            dataset = self._load_casp_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        self.datasets[name] = dataset
        return dataset
    
    def _load_casp_dataset(self, **kwargs) -> Dataset:
        """Load CASP (Critical Assessment of Structure Prediction) dataset."""
        # For demonstration, return synthetic data
        # In practice, this would load real CASP targets
        return SyntheticProteinDataset(
            n_samples=100,
            min_length=100,
            max_length=500,
            structure_types=['mixed'],
            **kwargs
        )
    
    def get_dataloader(
        self,
        dataset_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        **dataset_kwargs
    ) -> DataLoader:
        """
        Get DataLoader for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **dataset_kwargs: Additional dataset arguments
            
        Returns:
            DataLoader instance
        """
        dataset = self.load_dataset(dataset_name, **dataset_kwargs)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for variable-length proteins."""
        # Pad sequences to same length
        max_length = max(item['input'].shape[0] for item in batch)
        
        padded_inputs = []
        padded_targets = []
        sequences = []
        metadata = []
        
        for item in batch:
            input_seq = item['input']
            target_coords = item['target']
            
            # Pad input
            padding_input = torch.zeros(max_length - input_seq.shape[0], input_seq.shape[1])
            padded_input = torch.cat([input_seq, padding_input], dim=0)
            padded_inputs.append(padded_input)
            
            # Pad target
            padding_target = torch.zeros(max_length - target_coords.shape[0], target_coords.shape[1])
            padded_target = torch.cat([target_coords, padding_target], dim=0)
            padded_targets.append(padded_target)
            
            sequences.append(item['sequence'])
            metadata.append(item['metadata'])
        
        return {
            'input': torch.stack(padded_inputs),
            'target': torch.stack(padded_targets),
            'sequences': sequences,
            'metadata': metadata
        }
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        dataset = self.load_dataset(name)
        
        info = {
            'name': name,
            'size': len(dataset),
            'type': type(dataset).__name__
        }
        
        # Sample a few items to get statistics
        sample_indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
        lengths = []
        
        for idx in sample_indices:
            item = dataset[idx]
            lengths.append(item['input'].shape[0])
        
        info.update({
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'std_length': np.std(lengths)
        })
        
        return info