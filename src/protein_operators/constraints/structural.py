"""
Structural constraints for protein design.

This module implements constraints related to protein structure,
such as secondary structure, fold topology, and geometric features.
"""

from typing import List, Optional, Dict, Any, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
import numpy as np

from .base import BaseConstraint


class SecondaryStructureConstraint(BaseConstraint):
    """
    Constraint for secondary structure elements.
    
    Specifies regions that should adopt specific secondary structure types.
    """
    
    CONSTRAINT_TYPE_ID = 4
    
    def __init__(
        self,
        name: str,
        start: int,
        end: int,
        ss_type: str,
        confidence: float = 1.0,
        **kwargs
    ):
        """
        Initialize secondary structure constraint.
        
        Args:
            name: Constraint name
            start: Starting residue index (1-based)
            end: Ending residue index (1-based)
            ss_type: Secondary structure type ('helix', 'sheet', 'loop')
            confidence: Confidence level for this constraint (0-1)
        """
        super().__init__(name, **kwargs)
        self.start = start
        self.end = end
        self.ss_type = ss_type
        self.confidence = confidence
        
        if start >= end:
            raise ValueError("Start position must be less than end position")
        if ss_type not in ["helix", "sheet", "loop", "turn"]:
            raise ValueError(f"Unknown secondary structure type: {ss_type}")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def length(self) -> int:
        return self.end - self.start + 1
    
    def encode(self) -> torch.Tensor:
        """Encode secondary structure constraint as tensor."""
        encoding = torch.zeros(10)
        
        # Encode start and end positions (normalized)
        encoding[0] = float(self.start) / 1000.0
        encoding[1] = float(self.end) / 1000.0
        
        # Encode structure type
        ss_type_map = {"helix": 0.2, "sheet": 0.5, "loop": 0.8, "turn": 0.9}
        encoding[2] = ss_type_map.get(self.ss_type, 0.5)
        
        # Encode confidence
        encoding[3] = self.confidence
        
        # Encode length
        encoding[4] = min(float(self.length) / 50.0, 1.0)  # Normalize by max length
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate secondary structure constraint against structure."""
        # Check bounds
        if self.end > structure.coordinates.shape[0]:
            return False
        
        # Get coordinates for this region
        region_coords = structure.coordinates[self.start-1:self.end]  # 0-indexed
        
        if region_coords.shape[0] < 3:
            return True  # Too short to validate
        
        # Simple geometric validation based on structure type
        if self.ss_type == "helix":
            return self._validate_helix_geometry(region_coords)
        elif self.ss_type == "sheet":
            return self._validate_sheet_geometry(region_coords)
        else:
            return True  # Loop/turn - no specific geometry requirements
    
    def _validate_helix_geometry(self, coords: torch.Tensor) -> bool:
        """Validate helical geometry."""
        if coords.shape[0] < 4:
            return True
        
        # Check for roughly helical rise and twist
        # CA-CA distances should be ~3.8 Å
        ca_distances = torch.norm(coords[1:] - coords[:-1], dim=1)
        avg_distance = torch.mean(ca_distances)
        
        # Helical CA-CA distance is typically 3.6-4.0 Å
        return 3.2 <= avg_distance <= 4.4
    
    def _validate_sheet_geometry(self, coords: torch.Tensor) -> bool:
        """Validate sheet geometry."""
        if coords.shape[0] < 3:
            return True
        
        # Check for roughly extended conformation
        # CA-CA distances should be ~3.8 Å
        ca_distances = torch.norm(coords[1:] - coords[:-1], dim=1)
        avg_distance = torch.mean(ca_distances)
        
        # Sheet CA-CA distance is typically 3.7-3.9 Å
        return 3.5 <= avg_distance <= 4.1
    
    def satisfaction_score(self, structure) -> float:
        """Compute secondary structure satisfaction score."""
        if not self.validate(structure):
            return 0.0
        
        # Get coordinates for this region
        region_coords = structure.coordinates[self.start-1:self.end]
        
        if region_coords.shape[0] < 3:
            return 1.0
        
        # Score based on geometry consistency
        if self.ss_type == "helix":
            score = self._score_helix_geometry(region_coords)
        elif self.ss_type == "sheet":
            score = self._score_sheet_geometry(region_coords)
        else:
            score = 1.0  # Loop/turn - always satisfied
        
        return float(score * self.confidence)
    
    def _score_helix_geometry(self, coords: torch.Tensor) -> torch.Tensor:
        """Score helical geometry quality."""
        ca_distances = torch.norm(coords[1:] - coords[:-1], dim=1)
        avg_distance = torch.mean(ca_distances)
        
        # Ideal helical distance is ~3.8 Å
        ideal_distance = 3.8
        distance_score = torch.exp(-((avg_distance - ideal_distance) / 0.3)**2)
        
        return distance_score
    
    def _score_sheet_geometry(self, coords: torch.Tensor) -> torch.Tensor:
        """Score sheet geometry quality."""
        ca_distances = torch.norm(coords[1:] - coords[:-1], dim=1)
        avg_distance = torch.mean(ca_distances)
        
        # Ideal sheet distance is ~3.8 Å
        ideal_distance = 3.8
        distance_score = torch.exp(-((avg_distance - ideal_distance) / 0.2)**2)
        
        return distance_score
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class DisulfideBondConstraint(BaseConstraint):
    """
    Constraint for disulfide bonds between cysteine residues.
    """
    
    CONSTRAINT_TYPE_ID = 5
    
    def __init__(
        self,
        name: str,
        cys1: int,
        cys2: int,
        **kwargs
    ):
        """
        Initialize disulfide bond constraint.
        
        Args:
            name: Constraint name
            cys1: First cysteine residue index
            cys2: Second cysteine residue index
        """
        super().__init__(name, **kwargs)
        self.cys1 = min(cys1, cys2)
        self.cys2 = max(cys1, cys2)
        
        if cys1 == cys2:
            raise ValueError("Disulfide bond requires two different cysteines")
    
    def encode(self) -> torch.Tensor:
        """Encode disulfide bond constraint as tensor."""
        encoding = torch.zeros(5)
        
        # Encode cysteine positions
        encoding[0] = float(self.cys1) / 1000.0
        encoding[1] = float(self.cys2) / 1000.0
        
        # Encode separation
        separation = self.cys2 - self.cys1
        encoding[2] = min(float(separation) / 100.0, 1.0)
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate disulfide bond constraint against structure."""
        # Check bounds
        if max(self.cys1, self.cys2) > structure.coordinates.shape[0]:
            return False
        
        # Check distance between cysteines
        coord1 = structure.coordinates[self.cys1 - 1]  # 0-indexed
        coord2 = structure.coordinates[self.cys2 - 1]
        
        distance = torch.norm(coord1 - coord2)
        
        # Disulfide bond distance should be ~2.0 Å (allow 1.8-2.2 Å)
        return 1.8 <= distance <= 2.2
    
    def satisfaction_score(self, structure) -> float:
        """Compute disulfide bond satisfaction score."""
        if max(self.cys1, self.cys2) > structure.coordinates.shape[0]:
            return 0.0
        
        coord1 = structure.coordinates[self.cys1 - 1]
        coord2 = structure.coordinates[self.cys2 - 1]
        
        distance = torch.norm(coord1 - coord2)
        
        # Ideal disulfide distance is 2.0 Å
        ideal_distance = 2.0
        score = torch.exp(-((distance - ideal_distance) / 0.2)**2)
        
        return float(score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class MetalSiteConstraint(BaseConstraint):
    """
    Constraint for metal coordination sites.
    """
    
    CONSTRAINT_TYPE_ID = 6
    
    def __init__(
        self,
        name: str,
        metal: str,
        coordinating_residues: List[int],
        geometry: str = "tetrahedral",
        **kwargs
    ):
        """
        Initialize metal site constraint.
        
        Args:
            name: Constraint name
            metal: Metal ion type (Zn2+, Fe2+, etc.)
            coordinating_residues: Residues coordinating the metal
            geometry: Coordination geometry
        """
        super().__init__(name, **kwargs)
        self.metal = metal
        self.coordinating_residues = coordinating_residues
        self.geometry = geometry
        
        if len(coordinating_residues) < 2:
            raise ValueError("Metal site requires at least 2 coordinating residues")
    
    def encode(self) -> torch.Tensor:
        """Encode metal site constraint as tensor."""
        encoding = torch.zeros(15)
        
        # Encode metal type (simplified hash)
        metal_hash = hash(self.metal) % 10
        encoding[0] = float(metal_hash) / 10.0
        
        # Encode coordinating residues
        for i, res_idx in enumerate(self.coordinating_residues[:8]):
            encoding[i + 1] = float(res_idx) / 1000.0
        
        # Encode geometry
        geom_map = {"tetrahedral": 0.2, "octahedral": 0.5, "square_planar": 0.8}
        encoding[9] = geom_map.get(self.geometry, 0.2)
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate metal site constraint against structure."""
        # Check bounds
        max_residue = max(self.coordinating_residues)
        if max_residue > structure.coordinates.shape[0]:
            return False
        
        # Check that coordinating residues are reasonably close
        coord_coords = structure.coordinates[torch.tensor(self.coordinating_residues) - 1]
        center = torch.mean(coord_coords, dim=0)
        distances = torch.norm(coord_coords - center, dim=1)
        
        # All coordinating residues should be within 5 Å of center
        return torch.all(distances < 5.0)
    
    def satisfaction_score(self, structure) -> float:
        """Compute metal site satisfaction score."""
        if not self.validate(structure):
            return 0.0
        
        coord_coords = structure.coordinates[torch.tensor(self.coordinating_residues) - 1]
        center = torch.mean(coord_coords, dim=0)
        distances = torch.norm(coord_coords - center, dim=1)
        
        # Score based on compactness of coordination sphere
        avg_distance = torch.mean(distances)
        
        # Ideal coordination distance is ~2.5 Å
        ideal_distance = 2.5
        score = torch.exp(-((avg_distance - ideal_distance) / 1.0)**2)
        
        return float(score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class StructuralConstraint(BaseConstraint):
    """
    General structural constraint for protein design.
    
    This is an alias for FoldConstraint to maintain backward compatibility.
    """
    
    CONSTRAINT_TYPE_ID = 8
    
    def __init__(
        self,
        name: str,
        fold_type: Optional[str] = None,
        disulfide_bonds: Optional[List[Tuple[int, int]]] = None,
        metal_sites: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.fold_type = fold_type
        self.disulfide_bonds = disulfide_bonds or []
        self.metal_sites = metal_sites or []
    
    def encode(self) -> torch.Tensor:
        encoding = torch.zeros(20)
        if self.fold_type:
            fold_hash = hash(self.fold_type) % 100
            encoding[0] = float(fold_hash) / 100.0
        return encoding
    
    def validate(self, structure) -> bool:
        return True
    
    def satisfaction_score(self, structure) -> float:
        return 1.0
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class FoldConstraint(BaseConstraint):
    """
    Constraint for overall protein fold topology.
    """
    
    CONSTRAINT_TYPE_ID = 7
    
    def __init__(
        self,
        name: str,
        fold_family: str,
        reference_structure: Optional[str] = None,
        max_rmsd: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize fold constraint.
        
        Args:
            name: Constraint name
            fold_family: Fold family (e.g., "immunoglobulin", "rossmann")
            reference_structure: Reference structure identifier
            max_rmsd: Maximum allowed RMSD from reference
        """
        super().__init__(name, **kwargs)
        self.fold_family = fold_family
        self.reference_structure = reference_structure
        self.max_rmsd = max_rmsd
    
    def encode(self) -> torch.Tensor:
        """Encode fold constraint as tensor."""
        encoding = torch.zeros(10)
        
        # Encode fold family (simplified hash)
        fold_hash = hash(self.fold_family) % 100
        encoding[0] = float(fold_hash) / 100.0
        
        # Encode RMSD tolerance
        if self.max_rmsd is not None:
            encoding[1] = min(self.max_rmsd / 10.0, 1.0)
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate fold constraint against structure."""
        # For now, always return True since fold validation
        # requires comparison with reference structures
        return True
    
    def satisfaction_score(self, structure) -> float:
        """Compute fold satisfaction score."""
        # Simplified scoring based on compactness
        coords = structure.coordinates
        center = torch.mean(coords, dim=0)
        distances = torch.norm(coords - center, dim=1)
        
        # More compact structures generally score higher
        avg_distance = torch.mean(distances)
        score = torch.exp(-avg_distance / coords.shape[0])
        
        return float(score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID