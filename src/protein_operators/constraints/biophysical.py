"""
Biophysical constraints for protein design.

This module implements constraints related to protein biophysical properties,
including stability, solubility, expression, and immunogenicity.
"""

from typing import Optional, Tuple, List, Union
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
# Standard imports for biophysical constraints
from .base import BaseConstraint


class StabilityConstraint(BaseConstraint):
    """
    Constraint for protein thermodynamic stability.
    
    Specifies requirements for protein stability including melting temperature,
    pH stability, and resistance to denaturation.
    """
    
    CONSTRAINT_TYPE_ID = 9
    
    def __init__(
        self,
        name: str,
        tm_celsius: Optional[float] = None,
        ph_range: Optional[Tuple[float, float]] = None,
        ionic_strength: Optional[float] = None,
        denaturant_resistance: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize stability constraint.
        
        Args:
            name: Constraint name
            tm_celsius: Target melting temperature in Celsius
            ph_range: Optimal pH range as (min, max)
            ionic_strength: Ionic strength in M
            denaturant_resistance: Resistance to denaturants (0-1 scale)
        """
        super().__init__(name, **kwargs)
        self.tm_celsius = tm_celsius
        self.ph_range = ph_range or (6.0, 8.0)
        self.ionic_strength = ionic_strength or 0.15
        self.denaturant_resistance = denaturant_resistance
        
    def validate_parameters(self) -> None:
        """Validate constraint parameters."""
        if self.tm_celsius is not None and (self.tm_celsius < -20 or self.tm_celsius > 150):
            raise ValueError("Tm must be between -20°C and 150°C")
            
        if self.ph_range[0] < 0 or self.ph_range[1] > 14 or self.ph_range[0] >= self.ph_range[1]:
            raise ValueError("pH range must be valid (0-14) with min < max")
            
        if self.ionic_strength is not None and (self.ionic_strength < 0 or self.ionic_strength > 5.0):
            raise ValueError("Ionic strength must be between 0 and 5.0 M")
            
        if self.denaturant_resistance is not None and (self.denaturant_resistance < 0 or self.denaturant_resistance > 1):
            raise ValueError("Denaturant resistance must be between 0 and 1")

    def encode(self) -> torch.Tensor:
        """Encode stability constraint as tensor."""
        encoding = torch.zeros(8)
        
        # Tm encoding (normalized)
        if self.tm_celsius is not None:
            encoding[0] = self.tm_celsius / 100.0
        
        # pH range encoding
        encoding[1] = self.ph_range[0] / 14.0
        encoding[2] = self.ph_range[1] / 14.0
        
        # Ionic strength encoding
        if self.ionic_strength is not None:
            encoding[3] = min(self.ionic_strength / 2.0, 1.0)
        
        # Denaturant resistance
        if self.denaturant_resistance is not None:
            encoding[4] = self.denaturant_resistance
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate stability constraint."""
        return hasattr(structure, 'coordinates') and structure.coordinates.numel() > 0
    
    def satisfaction_score(self, structure) -> float:
        """Compute how well structure satisfies stability constraint."""
        if not hasattr(structure, 'coordinates'):
            return 0.0
            
        coords = structure.coordinates
        if coords.numel() == 0:
            return 0.0
        
        # Simple stability metric based on compactness
        center = torch.mean(coords, dim=0)
        distances = torch.norm(coords - center, dim=1)
        compactness = 1.0 / (1.0 + torch.std(distances))
        
        # Penalize very extended structures
        max_distance = torch.max(distances)
        extension_penalty = torch.exp(-max_distance / coords.shape[0])
        
        stability_score = 0.7 * compactness + 0.3 * extension_penalty
        return float(stability_score.clamp(0, 1))
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class SolubilityConstraint(BaseConstraint):
    """
    Constraint for protein solubility in aqueous solution.
    
    Specifies requirements for protein solubility including minimum
    concentrations and aggregation resistance.
    """
    
    CONSTRAINT_TYPE_ID = 10
    
    def __init__(
        self,
        name: str,
        min_solubility_mg_ml: Optional[float] = None,
        ph_optimum: Optional[float] = None,
        hydrophobicity_ratio: Optional[float] = None,
        aggregation_resistance: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize solubility constraint.
        
        Args:
            name: Constraint name
            min_solubility_mg_ml: Minimum solubility in mg/mL
            ph_optimum: pH for optimal solubility
            hydrophobicity_ratio: Target hydrophobic/hydrophilic ratio
            aggregation_resistance: Resistance to aggregation (0-1 scale)
        """
        super().__init__(name, **kwargs)
        self.min_solubility_mg_ml = min_solubility_mg_ml
        self.ph_optimum = ph_optimum or 7.4
        self.hydrophobicity_ratio = hydrophobicity_ratio
        self.aggregation_resistance = aggregation_resistance
        
    def validate_parameters(self) -> None:
        """Validate constraint parameters."""
        if self.min_solubility_mg_ml is not None and self.min_solubility_mg_ml < 0:
            raise ValueError("Minimum solubility must be non-negative")
            
        if self.ph_optimum < 0 or self.ph_optimum > 14:
            raise ValueError("pH optimum must be between 0 and 14")
            
        if self.hydrophobicity_ratio is not None and self.hydrophobicity_ratio < 0:
            raise ValueError("Hydrophobicity ratio must be non-negative")
            
        if self.aggregation_resistance is not None and (self.aggregation_resistance < 0 or self.aggregation_resistance > 1):
            raise ValueError("Aggregation resistance must be between 0 and 1")

    def encode(self) -> torch.Tensor:
        """Encode solubility constraint as tensor."""
        encoding = torch.zeros(6)
        
        # Minimum solubility (log-scaled and normalized)
        if self.min_solubility_mg_ml is not None:
            encoding[0] = min(torch.log10(torch.tensor(self.min_solubility_mg_ml + 1e-6)) / 3.0, 1.0)
        
        # pH optimum
        encoding[1] = self.ph_optimum / 14.0
        
        # Hydrophobicity ratio
        if self.hydrophobicity_ratio is not None:
            encoding[2] = min(self.hydrophobicity_ratio / 2.0, 1.0)
        
        # Aggregation resistance
        if self.aggregation_resistance is not None:
            encoding[3] = self.aggregation_resistance
            
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate solubility constraint."""
        return hasattr(structure, 'coordinates') and structure.coordinates.numel() > 0
    
    def satisfaction_score(self, structure) -> float:
        """Compute how well structure satisfies solubility constraint."""
        if not hasattr(structure, 'coordinates'):
            return 0.0
            
        coords = structure.coordinates
        if coords.numel() == 0:
            return 0.0
        
        # Estimate surface exposure (simplified)
        center = torch.mean(coords, dim=0)
        distances_from_center = torch.norm(coords - center, dim=1)
        
        # Surface residues are those farther from center
        surface_threshold = torch.quantile(distances_from_center, 0.7)
        surface_mask = distances_from_center > surface_threshold
        surface_ratio = torch.sum(surface_mask.float()) / coords.shape[0]
        
        # Higher surface exposure generally correlates with better solubility
        solubility_score = surface_ratio.clamp(0, 1)
        
        return float(solubility_score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID