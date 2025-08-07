"""
Biophysical constraints for protein design.
"""

from typing import Optional, Tuple, List, Union
import torch
from pydantic import BaseModel, Field
from .base import Constraint


class StabilityConstraint(Constraint):
    """
    Constraint for protein thermodynamic stability.
    
    Attributes:
        tm_celsius: Target melting temperature in Celsius
        ph_range: Optimal pH range as (min, max)
        ionic_strength: Ionic strength in M
        denaturant_resistance: Resistance to denaturants (0-1 scale)
    """
    
    def __init__(
        self,
        tm_celsius: Optional[float] = None,
        ph_range: Optional[Tuple[float, float]] = None,
        ionic_strength: Optional[float] = None,
        denaturant_resistance: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
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

    def encode(self, max_length: int) -> torch.Tensor:
        """Encode stability constraint as tensor."""
        encoding = torch.zeros(8)
        
        # Constraint type identifier
        encoding[0] = 3.0  # Stability constraint type
        
        # Tm encoding (normalized)
        if self.tm_celsius is not None:
            encoding[1] = self.tm_celsius / 100.0
        
        # pH range encoding
        encoding[2] = self.ph_range[0] / 14.0
        encoding[3] = self.ph_range[1] / 14.0
        
        # Ionic strength encoding
        if self.ionic_strength is not None:
            encoding[4] = min(self.ionic_strength / 2.0, 1.0)
        
        # Denaturant resistance
        if self.denaturant_resistance is not None:
            encoding[5] = self.denaturant_resistance
        
        return encoding
    
    def compute_satisfaction(self, structure) -> float:
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


class SolubilityConstraint(Constraint):
    """
    Constraint for protein solubility in aqueous solution.
    
    Attributes:
        min_solubility_mg_ml: Minimum solubility in mg/mL
        ph_optimum: pH for optimal solubility
        hydrophobicity_ratio: Target hydrophobic/hydrophilic ratio
        aggregation_resistance: Resistance to aggregation (0-1 scale)
    """
    
    def __init__(
        self,
        min_solubility_mg_ml: Optional[float] = None,
        ph_optimum: Optional[float] = None,
        hydrophobicity_ratio: Optional[float] = None,
        aggregation_resistance: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
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

    def encode(self, max_length: int) -> torch.Tensor:
        """Encode solubility constraint as tensor."""
        encoding = torch.zeros(6)
        
        # Constraint type identifier
        encoding[0] = 4.0  # Solubility constraint type
        
        # Minimum solubility (log-scaled and normalized)
        if self.min_solubility_mg_ml is not None:
            encoding[1] = min(torch.log10(torch.tensor(self.min_solubility_mg_ml + 1e-6)) / 3.0, 1.0)
        
        # pH optimum
        encoding[2] = self.ph_optimum / 14.0
        
        # Hydrophobicity ratio
        if self.hydrophobicity_ratio is not None:
            encoding[3] = min(self.hydrophobicity_ratio / 2.0, 1.0)
        
        # Aggregation resistance
        if self.aggregation_resistance is not None:
            encoding[4] = self.aggregation_resistance
            
        return encoding
    
    def compute_satisfaction(self, structure) -> float:
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


class ExpressionConstraint(Constraint):
    """
    Constraint for protein expression in recombinant systems.
    
    Attributes:
        expression_system: Target expression system ('ecoli', 'yeast', 'mammalian')
        min_yield_mg_l: Minimum expression yield in mg/L
        codon_optimization: Whether to optimize codons for expression system
        avoid_toxic_sequences: Avoid sequences toxic to expression host
    """
    
    def __init__(
        self,
        expression_system: str = 'ecoli',
        min_yield_mg_l: Optional[float] = None,
        codon_optimization: bool = True,
        avoid_toxic_sequences: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.expression_system = expression_system.lower()
        self.min_yield_mg_l = min_yield_mg_l
        self.codon_optimization = codon_optimization
        self.avoid_toxic_sequences = avoid_toxic_sequences
        
        # Validate expression system
        valid_systems = {'ecoli', 'yeast', 'mammalian', 'insect', 'plant'}
        if self.expression_system not in valid_systems:
            raise ValueError(f"Expression system must be one of {valid_systems}")
        
    def validate_parameters(self) -> None:
        """Validate constraint parameters."""
        if self.min_yield_mg_l is not None and self.min_yield_mg_l < 0:
            raise ValueError("Minimum yield must be non-negative")

    def encode(self, max_length: int) -> torch.Tensor:
        """Encode expression constraint as tensor."""
        encoding = torch.zeros(6)
        
        # Constraint type identifier
        encoding[0] = 5.0  # Expression constraint type
        
        # Expression system encoding
        system_map = {'ecoli': 0.2, 'yeast': 0.4, 'mammalian': 0.6, 'insect': 0.8, 'plant': 1.0}
        encoding[1] = system_map.get(self.expression_system, 0.2)
        
        # Minimum yield (log-scaled)
        if self.min_yield_mg_l is not None:
            encoding[2] = min(torch.log10(torch.tensor(self.min_yield_mg_l + 1e-6)) / 4.0, 1.0)
        
        # Boolean flags
        encoding[3] = float(self.codon_optimization)
        encoding[4] = float(self.avoid_toxic_sequences)
        
        return encoding
    
    def compute_satisfaction(self, structure) -> float:
        """Compute how well structure satisfies expression constraint."""
        if not hasattr(structure, 'coordinates'):
            return 0.0
            
        # Simplified expression compatibility score
        # In practice, this would analyze sequence features
        coords = structure.coordinates
        if coords.numel() == 0:
            return 0.0
        
        # Assume moderate complexity structures express better
        complexity = self._estimate_complexity(coords)
        
        # Optimal complexity range for expression
        if 0.3 <= complexity <= 0.7:
            return 0.9
        elif 0.2 <= complexity <= 0.8:
            return 0.7
        else:
            return 0.4
    
    def _estimate_complexity(self, coords: torch.Tensor) -> float:
        """Estimate structural complexity."""
        if coords.shape[0] < 2:
            return 0.0
        
        # Measure structural variation
        center = torch.mean(coords, dim=0)
        distances = torch.norm(coords - center, dim=1)
        normalized_complexity = torch.std(distances) / torch.mean(distances)
        
        return float(normalized_complexity.clamp(0, 1))


class ImmunogenicityConstraint(Constraint):
    """
    Constraint to minimize immunogenicity for therapeutic proteins.
    
    Attributes:
        max_immunogenicity_score: Maximum allowed immunogenicity score (0-1)
        species: Target species ('human', 'mouse', 'monkey')
        avoid_t_cell_epitopes: Whether to avoid known T-cell epitopes
        humanization_level: Level of humanization required (0-1)
    """
    
    def __init__(
        self,
        max_immunogenicity_score: float = 0.3,
        species: str = 'human',
        avoid_t_cell_epitopes: bool = True,
        humanization_level: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_immunogenicity_score = max_immunogenicity_score
        self.species = species.lower()
        self.avoid_t_cell_epitopes = avoid_t_cell_epitopes
        self.humanization_level = humanization_level
        
        # Validate species
        valid_species = {'human', 'mouse', 'rat', 'monkey', 'rabbit'}
        if self.species not in valid_species:
            raise ValueError(f"Species must be one of {valid_species}")
        
    def validate_parameters(self) -> None:
        """Validate constraint parameters."""
        if not 0 <= self.max_immunogenicity_score <= 1:
            raise ValueError("Max immunogenicity score must be between 0 and 1")
            
        if self.humanization_level is not None and not 0 <= self.humanization_level <= 1:
            raise ValueError("Humanization level must be between 0 and 1")

    def encode(self, max_length: int) -> torch.Tensor:
        """Encode immunogenicity constraint as tensor."""
        encoding = torch.zeros(6)
        
        # Constraint type identifier
        encoding[0] = 6.0  # Immunogenicity constraint type
        
        # Max immunogenicity score
        encoding[1] = self.max_immunogenicity_score
        
        # Species encoding
        species_map = {'human': 0.8, 'monkey': 0.6, 'mouse': 0.4, 'rat': 0.3, 'rabbit': 0.2}
        encoding[2] = species_map.get(self.species, 0.5)
        
        # Boolean flags
        encoding[3] = float(self.avoid_t_cell_epitopes)
        
        # Humanization level
        if self.humanization_level is not None:
            encoding[4] = self.humanization_level
        
        return encoding
    
    def compute_satisfaction(self, structure) -> float:
        """Compute how well structure satisfies immunogenicity constraint."""
        if not hasattr(structure, 'coordinates'):
            return 0.0
            
        # Simplified immunogenicity assessment
        # In practice, this would use epitope prediction algorithms
        coords = structure.coordinates
        if coords.numel() == 0:
            return 0.0
        
        # Assume more compact, less exposed structures are less immunogenic
        center = torch.mean(coords, dim=0)
        distances = torch.norm(coords - center, dim=1)
        surface_exposure = torch.max(distances) / coords.shape[0]
        
        # Lower surface exposure = lower immunogenicity
        immunogenicity_score = surface_exposure.clamp(0, 1)
        satisfaction = 1.0 - immunogenicity_score
        
        return float(satisfaction)