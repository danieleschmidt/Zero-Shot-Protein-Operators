"""
Functional constraints for protein design.

This module implements constraints related to protein function,
such as binding sites, catalytic activity, and allosteric regulation.
"""

from typing import List, Optional, Dict, Any
import torch
import numpy as np

from .base import BaseConstraint


class BindingSiteConstraint(BaseConstraint):
    """
    Constraint for protein-ligand binding sites.
    
    Specifies residues that should form a binding site for a particular
    ligand with desired affinity and selectivity properties.
    """
    
    CONSTRAINT_TYPE_ID = 1
    
    def __init__(
        self,
        name: str,
        residues: List[int],
        ligand: str,
        affinity_nm: Optional[float] = None,
        selectivity_fold: Optional[float] = None,
        binding_mode: str = "competitive",
        **kwargs
    ):
        """
        Initialize binding site constraint.
        
        Args:
            name: Constraint name
            residues: List of residue indices forming the binding site
            ligand: Ligand identifier (name, SMILES, etc.)
            affinity_nm: Target binding affinity in nanomolar
            selectivity_fold: Required selectivity over off-targets
            binding_mode: Type of binding (competitive, allosteric, etc.)
        """
        super().__init__(name, **kwargs)
        self.residues = residues
        self.ligand = ligand
        self.affinity_nm = affinity_nm
        self.selectivity_fold = selectivity_fold
        self.binding_mode = binding_mode
        
        if not residues:
            raise ValueError("Binding site must contain at least one residue")
        if any(res < 1 for res in residues):
            raise ValueError("Residue indices must be positive")
    
    def encode(self) -> torch.Tensor:
        """Encode binding site constraint as tensor."""
        encoding = torch.zeros(20)  # Fixed size encoding
        
        # Encode residue positions (up to 10 residues)
        for i, res_idx in enumerate(self.residues[:10]):
            encoding[i] = float(res_idx) / 1000.0  # Normalize
        
        # Encode ligand (simplified hash)
        ligand_hash = hash(self.ligand) % 100
        encoding[10] = float(ligand_hash) / 100.0
        
        # Encode affinity
        if self.affinity_nm is not None:
            encoding[11] = min(self.affinity_nm / 1000.0, 1.0)
        
        # Encode selectivity
        if self.selectivity_fold is not None:
            encoding[12] = min(np.log10(self.selectivity_fold) / 4.0, 1.0)
        
        # Encode binding mode
        mode_map = {"competitive": 0.2, "allosteric": 0.5, "covalent": 0.8}
        encoding[13] = mode_map.get(self.binding_mode, 0.2)
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate binding site constraint against structure."""
        # Check if all residues are within structure bounds
        max_residue = max(self.residues)
        if max_residue > structure.coordinates.shape[0]:
            return False
        
        # Check if binding site residues are spatially clustered
        site_coords = structure.coordinates[torch.tensor(self.residues) - 1]  # 0-indexed
        center = torch.mean(site_coords, dim=0)
        distances = torch.norm(site_coords - center, dim=1)
        
        # Binding site should be compact (all residues within 15 Å of center)
        return torch.all(distances < 15.0)
    
    def satisfaction_score(self, structure) -> float:
        """Compute binding site satisfaction score."""
        if not self.validate(structure):
            return 0.0
        
        # Score based on spatial clustering of binding site residues
        site_coords = structure.coordinates[torch.tensor(self.residues) - 1]
        center = torch.mean(site_coords, dim=0)
        distances = torch.norm(site_coords - center, dim=1)
        
        # Better score for more compact binding sites
        avg_distance = torch.mean(distances)
        score = torch.exp(-avg_distance / 10.0)  # Decay with distance
        
        return float(score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class CatalyticConstraint(BaseConstraint):
    """
    Constraint for enzymatic catalytic activity.
    
    Specifies requirements for catalyzing specific chemical reactions.
    """
    
    CONSTRAINT_TYPE_ID = 2
    
    def __init__(
        self,
        name: str,
        reaction_type: str,
        substrate: str,
        catalytic_residues: Optional[List[int]] = None,
        kcat_per_km: Optional[float] = None,
        ph_optimum: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize catalytic constraint.
        
        Args:
            name: Constraint name
            reaction_type: Type of reaction (hydrolysis, oxidation, etc.)
            substrate: Substrate specification
            catalytic_residues: Residues forming the catalytic site
            kcat_per_km: Target catalytic efficiency (M^-1 s^-1)
            ph_optimum: Optimal pH for activity
        """
        super().__init__(name, **kwargs)
        self.reaction_type = reaction_type
        self.substrate = substrate
        self.catalytic_residues = catalytic_residues or []
        self.kcat_per_km = kcat_per_km
        self.ph_optimum = ph_optimum
    
    def encode(self) -> torch.Tensor:
        """Encode catalytic constraint as tensor."""
        encoding = torch.zeros(15)
        
        # Encode reaction type (simplified)
        reaction_map = {
            "hydrolysis": 0.1, "oxidation": 0.3, "reduction": 0.5,
            "transfer": 0.7, "isomerization": 0.9
        }
        encoding[0] = reaction_map.get(self.reaction_type, 0.1)
        
        # Encode catalytic residues
        for i, res_idx in enumerate(self.catalytic_residues[:5]):
            encoding[i + 1] = float(res_idx) / 1000.0
        
        # Encode kinetic parameters
        if self.kcat_per_km is not None:
            encoding[6] = min(np.log10(self.kcat_per_km) / 8.0, 1.0)  # Up to 10^8
        
        if self.ph_optimum is not None:
            encoding[7] = self.ph_optimum / 14.0
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate catalytic constraint against structure."""
        if not self.catalytic_residues:
            return True  # No specific residue requirements
        
        # Check if catalytic residues are within bounds
        max_residue = max(self.catalytic_residues)
        if max_residue > structure.coordinates.shape[0]:
            return False
        
        # Check if catalytic residues are reasonably close
        cat_coords = structure.coordinates[torch.tensor(self.catalytic_residues) - 1]
        pairwise_distances = torch.cdist(cat_coords, cat_coords)
        
        # All catalytic residues should be within 20 Å of each other
        return torch.all(pairwise_distances[pairwise_distances > 0] < 20.0)
    
    def satisfaction_score(self, structure) -> float:
        """Compute catalytic satisfaction score."""
        if not self.validate(structure):
            return 0.0
        
        if not self.catalytic_residues:
            return 1.0  # No specific requirements
        
        # Score based on optimal geometry of catalytic residues
        cat_coords = structure.coordinates[torch.tensor(self.catalytic_residues) - 1]
        center = torch.mean(cat_coords, dim=0)
        distances = torch.norm(cat_coords - center, dim=1)
        
        # Prefer catalytic sites that are compact but not too tight
        avg_distance = torch.mean(distances)
        optimal_distance = 8.0  # Å
        score = torch.exp(-((avg_distance - optimal_distance) / 5.0)**2)
        
        return float(score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID


class AllostericConstraint(BaseConstraint):
    """
    Constraint for allosteric regulation sites.
    
    Specifies sites that should respond to allosteric effectors.
    """
    
    CONSTRAINT_TYPE_ID = 3
    
    def __init__(
        self,
        name: str,
        allosteric_site: List[int],
        active_site: List[int],
        effector_type: str = "activator",
        coupling_strength: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize allosteric constraint.
        
        Args:
            name: Constraint name
            allosteric_site: Residues forming the allosteric site
            active_site: Residues forming the active site
            effector_type: Type of allosteric effect (activator/inhibitor)
            coupling_strength: Strength of allosteric coupling
        """
        super().__init__(name, **kwargs)
        self.allosteric_site = allosteric_site
        self.active_site = active_site
        self.effector_type = effector_type
        self.coupling_strength = coupling_strength
    
    def encode(self) -> torch.Tensor:
        """Encode allosteric constraint as tensor."""
        encoding = torch.zeros(20)
        
        # Encode allosteric site residues
        for i, res_idx in enumerate(self.allosteric_site[:5]):
            encoding[i] = float(res_idx) / 1000.0
        
        # Encode active site residues
        for i, res_idx in enumerate(self.active_site[:5]):
            encoding[i + 5] = float(res_idx) / 1000.0
        
        # Encode effector type
        encoding[10] = 0.2 if self.effector_type == "activator" else 0.8
        
        # Encode coupling strength
        if self.coupling_strength is not None:
            encoding[11] = min(self.coupling_strength, 1.0)
        
        return encoding
    
    def validate(self, structure) -> bool:
        """Validate allosteric constraint against structure."""
        max_allo = max(self.allosteric_site)
        max_active = max(self.active_site)
        
        if max_allo > structure.coordinates.shape[0]:
            return False
        if max_active > structure.coordinates.shape[0]:
            return False
        
        # Check that allosteric and active sites are separated
        allo_coords = structure.coordinates[torch.tensor(self.allosteric_site) - 1]
        active_coords = structure.coordinates[torch.tensor(self.active_site) - 1]
        
        allo_center = torch.mean(allo_coords, dim=0)
        active_center = torch.mean(active_coords, dim=0)
        
        separation = torch.norm(allo_center - active_center)
        
        # Sites should be separated by at least 15 Å
        return separation > 15.0
    
    def satisfaction_score(self, structure) -> float:
        """Compute allosteric satisfaction score."""
        if not self.validate(structure):
            return 0.0
        
        allo_coords = structure.coordinates[torch.tensor(self.allosteric_site) - 1]
        active_coords = structure.coordinates[torch.tensor(self.active_site) - 1]
        
        allo_center = torch.mean(allo_coords, dim=0)
        active_center = torch.mean(active_coords, dim=0)
        
        separation = torch.norm(allo_center - active_center)
        
        # Optimal separation is around 25 Å
        optimal_separation = 25.0
        score = torch.exp(-((separation - optimal_separation) / 10.0)**2)
        
        return float(score)
    
    def get_constraint_type_id(self) -> int:
        return self.CONSTRAINT_TYPE_ID