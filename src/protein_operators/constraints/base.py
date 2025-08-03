"""
Base constraint classes for protein design.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ConstraintConfig:
    """Configuration for a constraint."""
    constraint_type: str
    weight: float = 1.0
    tolerance: float = 0.1
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseConstraint(ABC):
    """
    Abstract base class for all protein design constraints.
    
    All constraint types must inherit from this class and implement
    the required methods for encoding, validation, and satisfaction scoring.
    """
    
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        tolerance: float = 0.1,
        required: bool = True,
        **kwargs
    ):
        """
        Initialize base constraint.
        
        Args:
            name: Human-readable constraint name
            weight: Importance weight (higher = more important)
            tolerance: Tolerance for constraint satisfaction
            required: Whether this constraint must be satisfied
            **kwargs: Additional constraint-specific parameters
        """
        self.name = name
        self.weight = weight
        self.tolerance = tolerance
        self.required = required
        self.config = kwargs
        
    @abstractmethod
    def encode(self) -> torch.Tensor:
        """
        Encode constraint into neural network input format.
        
        Returns:
            Constraint encoding tensor
        """
        pass
    
    @abstractmethod
    def validate(self, structure: "ProteinStructure") -> bool:
        """
        Check if constraint is satisfied by a protein structure.
        
        Args:
            structure: Protein structure to validate
            
        Returns:
            True if constraint is satisfied within tolerance
        """
        pass
    
    @abstractmethod
    def satisfaction_score(self, structure: "ProteinStructure") -> float:
        """
        Compute continuous satisfaction score for the constraint.
        
        Args:
            structure: Protein structure to score
            
        Returns:
            Score in [0, 1] where 1 = perfectly satisfied
        """
        pass
    
    @abstractmethod
    def get_constraint_type_id(self) -> int:
        """
        Get unique integer ID for this constraint type.
        
        Returns:
            Integer type ID for neural network encoding
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert constraint to dictionary representation."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "weight": self.weight,
            "tolerance": self.tolerance,
            "required": self.required,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConstraint":
        """Create constraint from dictionary representation."""
        config = data.get("config", {})
        return cls(
            name=data["name"],
            weight=data.get("weight", 1.0),
            tolerance=data.get("tolerance", 0.1),
            required=data.get("required", True),
            **config
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"


class Constraints:
    """
    Container for managing multiple protein design constraints.
    
    This class provides methods for adding, removing, and encoding
    multiple constraints for use in neural operator models.
    
    Examples:
        >>> constraints = Constraints()
        >>> constraints.add_binding_site(
        ...     residues=[45, 67, 89],
        ...     ligand="ATP",
        ...     affinity_nm=100
        ... )
        >>> constraints.add_secondary_structure(
        ...     regions=[(10, 25, "helix"), (30, 40, "sheet")]
        ... )
        >>> encoding = constraints.encode()
    """
    
    def __init__(self):
        self.constraints: List[BaseConstraint] = []
        self._constraint_types = {}  # Cache for constraint type mappings
        
    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Add a constraint to the collection."""
        self.constraints.append(constraint)
    
    def remove_constraint(self, name: str) -> bool:
        """
        Remove constraint by name.
        
        Args:
            name: Name of constraint to remove
            
        Returns:
            True if constraint was found and removed
        """
        for i, constraint in enumerate(self.constraints):
            if constraint.name == name:
                del self.constraints[i]
                return True
        return False
    
    def get_constraint(self, name: str) -> Optional[BaseConstraint]:
        """Get constraint by name."""
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint
        return None
    
    def encode(self, max_constraints: int = 20) -> torch.Tensor:
        """
        Encode all constraints for neural operator input.
        
        Args:
            max_constraints: Maximum number of constraints to encode
            
        Returns:
            Constraint tensor [num_constraints, constraint_dim]
            Format: [type_id, weight, tolerance, ...constraint_specific_params]
        """
        if not self.constraints:
            # Return zero constraint tensor
            return torch.zeros(1, 10)  # Default constraint dimension
        
        constraint_encodings = []
        
        for constraint in self.constraints[:max_constraints]:
            # Get type ID and basic parameters
            type_id = constraint.get_constraint_type_id()
            basic_params = torch.tensor([
                type_id,
                constraint.weight,
                constraint.tolerance,
                1.0 if constraint.required else 0.0
            ])
            
            # Get constraint-specific encoding
            specific_encoding = constraint.encode()
            
            # Combine basic and specific parameters
            full_encoding = torch.cat([basic_params, specific_encoding])
            constraint_encodings.append(full_encoding)
        
        # Pad to consistent length
        if constraint_encodings:
            max_dim = max(enc.size(0) for enc in constraint_encodings)
            padded_encodings = []
            
            for enc in constraint_encodings:
                if enc.size(0) < max_dim:
                    padding = torch.zeros(max_dim - enc.size(0))
                    enc = torch.cat([enc, padding])
                padded_encodings.append(enc)
            
            result = torch.stack(padded_encodings)
        else:
            result = torch.zeros(1, 10)
        
        # Pad number of constraints if needed
        if result.size(0) < max_constraints:
            padding = torch.zeros(max_constraints - result.size(0), result.size(1))
            result = torch.cat([result, padding], dim=0)
        
        return result
    
    def validate_all(self, structure: "ProteinStructure") -> Dict[str, bool]:
        """
        Validate all constraints against a structure.
        
        Args:
            structure: Protein structure to validate
            
        Returns:
            Dictionary mapping constraint names to satisfaction status
        """
        results = {}
        for constraint in self.constraints:
            results[constraint.name] = constraint.validate(structure)
        return results
    
    def satisfaction_scores(self, structure: "ProteinStructure") -> Dict[str, float]:
        """
        Compute satisfaction scores for all constraints.
        
        Args:
            structure: Protein structure to score
            
        Returns:
            Dictionary mapping constraint names to satisfaction scores
        """
        scores = {}
        for constraint in self.constraints:
            scores[constraint.name] = constraint.satisfaction_score(structure)
        return scores
    
    def overall_satisfaction(self, structure: "ProteinStructure") -> float:
        """
        Compute weighted overall satisfaction score.
        
        Args:
            structure: Protein structure to score
            
        Returns:
            Overall satisfaction score in [0, 1]
        """
        if not self.constraints:
            return 1.0
        
        total_weight = sum(c.weight for c in self.constraints)
        if total_weight == 0:
            return 1.0
        
        weighted_score = 0.0
        for constraint in self.constraints:
            score = constraint.satisfaction_score(structure)
            weighted_score += constraint.weight * score
        
        return weighted_score / total_weight
    
    def get_required_constraints(self) -> List[BaseConstraint]:
        """Get list of required constraints."""
        return [c for c in self.constraints if c.required]
    
    def get_optional_constraints(self) -> List[BaseConstraint]:
        """Get list of optional constraints."""
        return [c for c in self.constraints if not c.required]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary representation."""
        return {
            "constraints": [c.to_dict() for c in self.constraints],
            "num_constraints": len(self.constraints),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraints":
        """Create constraints from dictionary representation."""
        constraints = cls()
        
        for constraint_data in data.get("constraints", []):
            # Import constraint class dynamically
            constraint_type = constraint_data["type"]
            
            # This would need a proper constraint registry in practice
            if constraint_type == "StructuralConstraint":
                from .structural import StructuralConstraint
                constraint = StructuralConstraint.from_dict(constraint_data)
            elif constraint_type == "BindingSiteConstraint":
                from .functional import BindingSiteConstraint
                constraint = BindingSiteConstraint.from_dict(constraint_data)
            # Add more constraint types as needed
            else:
                continue  # Skip unknown constraint types
                
            constraints.add_constraint(constraint)
        
        return constraints
    
    def __len__(self) -> int:
        return len(self.constraints)
    
    def __iter__(self):
        return iter(self.constraints)
    
    # Convenience methods for common constraint types
    @property
    def binding_sites(self):
        """Get binding site constraints (for backward compatibility)."""
        from .functional import BindingSiteConstraint
        return [c for c in self.constraints if isinstance(c, BindingSiteConstraint)]
    
    @property 
    def secondary_structure(self):
        """Get secondary structure constraints (for backward compatibility)."""
        from .structural import SecondaryStructureConstraint
        return [c for c in self.constraints if isinstance(c, SecondaryStructureConstraint)]
    
    def add_binding_site(self, residues, ligand, affinity_nm=None, **kwargs):
        """Add a binding site constraint."""
        from .functional import BindingSiteConstraint
        constraint = BindingSiteConstraint(
            name=f"binding_site_{len(self.binding_sites) + 1}",
            residues=residues,
            ligand=ligand,
            affinity_nm=affinity_nm,
            **kwargs
        )
        self.add_constraint(constraint)
    
    def add_secondary_structure(self, start, end, ss_type, confidence=1.0, **kwargs):
        """Add a secondary structure constraint."""
        from .structural import SecondaryStructureConstraint
        constraint = SecondaryStructureConstraint(
            name=f"ss_{ss_type}_{start}_{end}",
            start=start,
            end=end,
            ss_type=ss_type,
            confidence=confidence,
            **kwargs
        )
        self.add_constraint(constraint)
        
    def __repr__(self) -> str:
        return f"Constraints({len(self.constraints)} constraints)"