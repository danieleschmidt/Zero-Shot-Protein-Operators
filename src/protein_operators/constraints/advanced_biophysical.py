"""
Advanced biophysical constraint embedding for neural operators.

This module provides sophisticated constraint embedding mechanisms for
incorporating complex biophysical properties into neural operator architectures.
These constraints encode fundamental physical and chemical principles that
govern protein structure and function.

Key innovations:
- Hierarchical constraint encoding from quantum to system scales
- Evolutionary conservation-aware constraint weighting
- Thermodynamic stability constraint embedding
- Allosteric regulation constraint modeling
- Protein-protein interaction interface constraints
"""

import os
import sys
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional


class ConstraintType(Enum):
    """Enumeration of constraint types."""
    THERMODYNAMIC = "thermodynamic"
    EVOLUTIONARY = "evolutionary"
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    ALLOSTERIC = "allosteric"
    INTERFACE = "interface"
    QUANTUM = "quantum"
    KINETIC = "kinetic"


@dataclass
class BiophysicalConstraint:
    """
    Container for biophysical constraint specification.
    """
    constraint_type: ConstraintType
    target_residues: List[int]
    constraint_value: float
    weight: float = 1.0
    tolerance: float = 0.1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ThermodynamicConstraintEncoder(nn.Module):
    """
    Encoder for thermodynamic stability constraints.
    
    Incorporates:
    - Free energy minimization
    - Entropy considerations
    - Temperature dependence
    - pH stability
    - Solvent effects
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        temperature_range: Tuple[float, float] = (273.15, 373.15),
        ph_range: Tuple[float, float] = (5.0, 9.0)
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature_range = temperature_range
        self.ph_range = ph_range
        
        # Free energy encoder
        self.free_energy_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Entropy encoder
        self.entropy_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive entropy
        )
        
        # Temperature dependence
        self.temperature_embedding = nn.Embedding(100, 32)  # Discretized temperature
        self.temp_modulation = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # pH dependence
        self.ph_embedding = nn.Embedding(50, 16)  # Discretized pH
        self.ph_modulation = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Solvent effect encoder
        self.solvent_encoder = nn.Sequential(
            nn.Linear(input_dim + 48, hidden_dim),  # +48 for temp and pH embeddings
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final constraint embedding
        self.constraint_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, hidden_dim),  # 3 encoders + free energy + entropy
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 298.15,
        ph: float = 7.0,
        solvent_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode thermodynamic constraints.
        
        Args:
            features: Input features [batch, input_dim]
            temperature: Temperature in Kelvin
            ph: pH value
            solvent_data: Optional solvent properties
            
        Returns:
            Encoded thermodynamic constraints [batch, input_dim]
        """
        batch_size = features.shape[0]
        
        # Normalize temperature and pH to embedding indices
        temp_normalized = (temperature - self.temperature_range[0]) / (
            self.temperature_range[1] - self.temperature_range[0]
        )
        temp_idx = int(torch.clamp(temp_normalized * 99, 0, 99))
        
        ph_normalized = (ph - self.ph_range[0]) / (self.ph_range[1] - self.ph_range[0])
        ph_idx = int(torch.clamp(ph_normalized * 49, 0, 49))
        
        # Get embeddings
        temp_emb = self.temperature_embedding(torch.tensor(temp_idx, device=features.device))
        ph_emb = self.ph_embedding(torch.tensor(ph_idx, device=features.device))
        
        # Broadcast to batch size
        temp_emb = temp_emb.unsqueeze(0).expand(batch_size, -1)
        ph_emb = ph_emb.unsqueeze(0).expand(batch_size, -1)
        
        # Encode free energy and entropy
        free_energy = self.free_energy_encoder(features)
        entropy = self.entropy_encoder(features)
        
        # Temperature modulation
        temp_mod = self.temp_modulation(temp_emb)
        temp_modulated_features = features * temp_mod
        
        # pH modulation
        ph_mod = self.ph_modulation(ph_emb)
        ph_modulated_features = features * ph_mod
        
        # Solvent effects
        if solvent_data is None:
            solvent_features = torch.cat([features, temp_emb, ph_emb], dim=1)
        else:
            solvent_features = torch.cat([features, solvent_data, temp_emb, ph_emb], dim=1)
        
        solvent_encoding = self.solvent_encoder(solvent_features)
        
        # Fuse all constraints
        combined_features = torch.cat([
            temp_modulated_features,
            ph_modulated_features, 
            solvent_encoding,
            free_energy,
            entropy
        ], dim=1)
        
        constraint_embedding = self.constraint_fusion(combined_features)
        
        return constraint_embedding


class EvolutionaryConstraintEncoder(nn.Module):
    """
    Encoder for evolutionary conservation constraints.
    
    Incorporates:
    - Conservation scores
    - Coevolution patterns
    - Phylogenetic relationships
    - Functional site conservation
    - Compensatory mutations
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        n_species: int = 1000,
        max_sequence_length: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_species = n_species
        self.max_sequence_length = max_sequence_length
        
        # Conservation score encoder
        self.conservation_encoder = nn.Sequential(
            nn.Linear(21, 64),  # 20 amino acids + gap
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )
        
        # Coevolution encoder using attention
        self.coevolution_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Phylogenetic relationship encoder
        self.phylo_encoder = nn.Sequential(
            nn.Linear(n_species, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Functional site encoder
        self.functional_site_encoder = nn.Sequential(
            nn.Linear(input_dim + 32, hidden_dim),  # +32 for conservation scores
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Compensatory mutation encoder
        self.compensatory_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Final constraint fusion
        self.constraint_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        conservation_scores: torch.Tensor,
        phylogenetic_weights: Optional[torch.Tensor] = None,
        functional_sites: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode evolutionary constraints.
        
        Args:
            features: Input features [batch, seq_len, input_dim]
            conservation_scores: Conservation scores [batch, seq_len, 21]
            phylogenetic_weights: Phylogenetic tree weights [batch, n_species]
            functional_sites: Functional site annotations [batch, seq_len]
            
        Returns:
            Encoded evolutionary constraints [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Encode conservation scores
        conservation_emb = self.conservation_encoder(conservation_scores)
        
        # Coevolution modeling through self-attention
        coevolution_features, _ = self.coevolution_attention(features, features, features)
        
        # Phylogenetic relationships
        if phylogenetic_weights is not None:
            phylo_emb = self.phylo_encoder(phylogenetic_weights)
            phylo_emb = phylo_emb.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            phylo_emb = torch.zeros(batch_size, seq_len, self.hidden_dim, device=features.device)
        
        # Functional site conservation
        functional_features = torch.cat([features, conservation_emb], dim=-1)
        functional_encoding = self.functional_site_encoder(functional_features)
        
        # Compensatory mutation detection
        # Use pairwise features to detect compensatory patterns
        pairwise_features = torch.cat([
            coevolution_features.unsqueeze(2).expand(-1, -1, seq_len, -1),
            coevolution_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
        ], dim=-1)
        
        # Average over sequence pairs
        compensatory_scores = self.compensatory_encoder(
            pairwise_features.view(batch_size, seq_len * seq_len, -1)
        ).view(batch_size, seq_len, seq_len)
        
        # Aggregate compensatory information
        compensatory_info = torch.mean(compensatory_scores, dim=-1, keepdim=True)
        
        # Fuse all evolutionary constraints
        combined_features = torch.cat([
            coevolution_features,
            phylo_emb,
            functional_encoding,
            compensatory_info
        ], dim=-1)
        
        constraint_embedding = self.constraint_fusion(combined_features)
        
        return constraint_embedding


class AllostericConstraintEncoder(nn.Module):
    """
    Encoder for allosteric regulation constraints.
    
    Incorporates:
    - Allosteric site identification
    - Signal propagation pathways
    - Conformational coupling
    - Regulatory binding sites
    - Dynamic network analysis
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        max_path_length: int = 20
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        
        # Allosteric site encoder
        self.allosteric_site_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Signal propagation encoder using graph neural network
        self.signal_propagation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            for _ in range(3)  # 3 propagation layers
        ])
        
        # Conformational coupling encoder
        self.coupling_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Coupling can be positive or negative
        )
        
        # Dynamic network encoder
        self.dynamic_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Path importance weighting
        self.path_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final constraint integration
        self.constraint_integration = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        allosteric_sites: Optional[torch.Tensor] = None,
        regulatory_sites: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode allosteric constraints.
        
        Args:
            features: Input features [batch, seq_len, input_dim]
            adjacency_matrix: Residue contact adjacency [batch, seq_len, seq_len]
            allosteric_sites: Known allosteric sites [batch, seq_len]
            regulatory_sites: Regulatory binding sites [batch, seq_len]
            
        Returns:
            Encoded allosteric constraints [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Project features to hidden dimension
        hidden_features = torch.matmul(features, torch.randn(self.input_dim, self.hidden_dim, device=features.device))
        
        # Identify allosteric sites
        allosteric_scores = self.allosteric_site_encoder(features).squeeze(-1)
        
        # Signal propagation through residue network
        propagated_features = hidden_features
        for layer in self.signal_propagation:
            # Graph convolution operation
            neighbor_features = torch.matmul(adjacency_matrix, propagated_features)
            propagated_features = layer(neighbor_features + propagated_features)
        
        # Conformational coupling analysis
        # Compute pairwise coupling strengths
        features_i = features.unsqueeze(2).expand(-1, -1, seq_len, -1)
        features_j = features.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairwise_features = torch.cat([features_i, features_j], dim=-1)
        
        coupling_matrix = self.coupling_encoder(pairwise_features).squeeze(-1)
        
        # Weight by distance/adjacency
        coupling_matrix = coupling_matrix * adjacency_matrix
        
        # Aggregate coupling information
        coupling_features = torch.matmul(coupling_matrix, hidden_features)
        
        # Dynamic network analysis
        dynamic_features, _ = self.dynamic_encoder(hidden_features)
        dynamic_features = dynamic_features[:, :, :self.hidden_dim] + dynamic_features[:, :, self.hidden_dim:]
        
        # Path-based attention weighting
        attended_features, _ = self.path_attention(
            dynamic_features, dynamic_features, dynamic_features
        )
        
        # Integrate all allosteric constraints
        combined_features = torch.cat([
            propagated_features,
            coupling_features,
            attended_features,
            allosteric_scores.unsqueeze(-1)
        ], dim=-1)
        
        constraint_embedding = self.constraint_integration(combined_features)
        
        return constraint_embedding


class QuantumConstraintEncoder(nn.Module):
    """
    Encoder for quantum mechanical constraints.
    
    Incorporates:
    - Electronic structure properties
    - Bond formation energies
    - Quantum tunneling effects
    - Spin states
    - Molecular orbital interactions
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        n_orbitals: int = 50
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_orbitals = n_orbitals
        
        # Electronic structure encoder
        self.electronic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_orbitals),
            nn.Softmax(dim=-1)  # Orbital occupation probabilities
        )
        
        # Bond energy calculator
        self.bond_energy_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Quantum tunneling effect encoder
        self.tunneling_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Molecular orbital interaction encoder
        self.orbital_interaction = nn.MultiheadAttention(
            embed_dim=n_orbitals,
            num_heads=5,
            batch_first=True
        )
        
        # Constraint fusion
        self.quantum_fusion = nn.Sequential(
            nn.Linear(n_orbitals + hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        atomic_numbers: Optional[torch.Tensor] = None,
        bond_orders: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode quantum mechanical constraints.
        
        Args:
            features: Input features [batch, seq_len, input_dim]
            atomic_numbers: Atomic numbers [batch, seq_len]
            bond_orders: Bond order matrix [batch, seq_len, seq_len]
            
        Returns:
            Encoded quantum constraints [batch, seq_len, input_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Electronic structure
        orbital_occupations = self.electronic_encoder(features)
        
        # Orbital interactions
        interacting_orbitals, _ = self.orbital_interaction(
            orbital_occupations, orbital_occupations, orbital_occupations
        )
        
        # Bond energies (simplified pairwise calculation)
        bond_energies = torch.zeros(batch_size, seq_len, device=features.device)
        
        for i in range(min(seq_len - 1, 5)):  # Limit computation for efficiency
            pair_features = torch.cat([features[:, :-i-1], features[:, i+1:]], dim=-1)
            pair_energies = self.bond_energy_encoder(pair_features).squeeze(-1)
            bond_energies[:, :-i-1] += pair_energies
        
        # Quantum tunneling effects
        tunneling_probs = self.tunneling_encoder(features).squeeze(-1)
        
        # Combine quantum effects
        combined_features = torch.cat([
            interacting_orbitals,
            features,  # Original features
            bond_energies.unsqueeze(-1),
            tunneling_probs.unsqueeze(-1)
        ], dim=-1)
        
        quantum_constraints = self.quantum_fusion(combined_features)
        
        return quantum_constraints


class AdvancedBiophysicalConstraintEmbedder(nn.Module):
    """
    Advanced biophysical constraint embedding system.
    
    This module integrates multiple types of biophysical constraints
    into a unified embedding that can guide neural operator training
    and inference for protein structure prediction and design.
    
    Features:
    - Multi-scale constraint integration
    - Adaptive constraint weighting
    - Cross-constraint interactions
    - Uncertainty-aware constraint encoding
    - Dynamic constraint activation
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        constraint_types: List[str] = None,
        adaptive_weighting: bool = True,
        uncertainty_quantification: bool = True
    ):
        """
        Initialize advanced constraint embedder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for encoders
            constraint_types: Types of constraints to include
            adaptive_weighting: Enable adaptive constraint weighting
            uncertainty_quantification: Enable uncertainty estimation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.adaptive_weighting = adaptive_weighting
        self.uncertainty_quantification = uncertainty_quantification
        
        if constraint_types is None:
            constraint_types = ['thermodynamic', 'evolutionary', 'allosteric', 'quantum']
        
        self.constraint_types = constraint_types
        
        # Initialize constraint encoders
        self.encoders = nn.ModuleDict()
        
        if 'thermodynamic' in constraint_types:
            self.encoders['thermodynamic'] = ThermodynamicConstraintEncoder(
                input_dim, hidden_dim
            )
        
        if 'evolutionary' in constraint_types:
            self.encoders['evolutionary'] = EvolutionaryConstraintEncoder(
                input_dim, hidden_dim
            )
        
        if 'allosteric' in constraint_types:
            self.encoders['allosteric'] = AllostericConstraintEncoder(
                input_dim, hidden_dim
            )
        
        if 'quantum' in constraint_types:
            self.encoders['quantum'] = QuantumConstraintEncoder(
                input_dim, hidden_dim
            )
        
        # Adaptive weighting system
        if adaptive_weighting:
            self.weight_predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(constraint_types)),
                nn.Softmax(dim=-1)
            )
        
        # Cross-constraint interaction modeling
        self.constraint_interaction = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Uncertainty estimation
        if uncertainty_quantification:
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(input_dim * len(constraint_types), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Softplus()
            )
        
        # Final constraint fusion
        self.constraint_fusion = nn.Sequential(
            nn.Linear(input_dim * len(constraint_types), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        constraint_data: Dict[str, Any],
        return_uncertainty: bool = False,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through constraint embedder.
        
        Args:
            features: Input features [batch, seq_len, input_dim]
            constraint_data: Dictionary of constraint-specific data
            return_uncertainty: Whether to return uncertainty estimates
            return_weights: Whether to return constraint weights
            
        Returns:
            Constraint embedding and optional uncertainty/weights
        """
        batch_size, seq_len, _ = features.shape
        
        # Encode each constraint type
        constraint_embeddings = []
        
        for constraint_type in self.constraint_types:
            if constraint_type in self.encoders:
                encoder = self.encoders[constraint_type]
                
                # Get constraint-specific data
                constraint_specific_data = constraint_data.get(constraint_type, {})
                
                # Encode constraint
                if constraint_type == 'thermodynamic':
                    embedding = encoder(
                        features.view(-1, self.input_dim),
                        temperature=constraint_specific_data.get('temperature', 298.15),
                        ph=constraint_specific_data.get('ph', 7.0),
                        solvent_data=constraint_specific_data.get('solvent_data')
                    ).view(batch_size, seq_len, -1)
                
                elif constraint_type == 'evolutionary':
                    embedding = encoder(
                        features,
                        conservation_scores=constraint_specific_data.get('conservation_scores'),
                        phylogenetic_weights=constraint_specific_data.get('phylogenetic_weights'),
                        functional_sites=constraint_specific_data.get('functional_sites')
                    )
                
                elif constraint_type == 'allosteric':
                    embedding = encoder(
                        features,
                        adjacency_matrix=constraint_specific_data.get('adjacency_matrix'),
                        allosteric_sites=constraint_specific_data.get('allosteric_sites'),
                        regulatory_sites=constraint_specific_data.get('regulatory_sites')
                    )
                
                elif constraint_type == 'quantum':
                    embedding = encoder(
                        features,
                        atomic_numbers=constraint_specific_data.get('atomic_numbers'),
                        bond_orders=constraint_specific_data.get('bond_orders')
                    )
                
                else:
                    # Generic encoding
                    embedding = features
                
                constraint_embeddings.append(embedding)
        
        # Stack constraint embeddings
        stacked_embeddings = torch.stack(constraint_embeddings, dim=-2)  # [batch, seq_len, n_constraints, input_dim]
        
        # Adaptive weighting
        if self.adaptive_weighting:
            # Compute weights based on input features
            mean_features = torch.mean(features, dim=1)  # [batch, input_dim]
            constraint_weights = self.weight_predictor(mean_features)  # [batch, n_constraints]
            
            # Apply weights
            weighted_embeddings = stacked_embeddings * constraint_weights.unsqueeze(1).unsqueeze(-1)
        else:
            weighted_embeddings = stacked_embeddings
            constraint_weights = torch.ones(batch_size, len(self.constraint_types), device=features.device) / len(self.constraint_types)
        
        # Cross-constraint interactions
        # Flatten for attention
        flat_embeddings = weighted_embeddings.view(batch_size, seq_len * len(self.constraint_types), self.input_dim)
        
        interacted_embeddings, _ = self.constraint_interaction(
            flat_embeddings, flat_embeddings, flat_embeddings
        )
        
        # Reshape back
        interacted_embeddings = interacted_embeddings.view(batch_size, seq_len, len(self.constraint_types), self.input_dim)
        
        # Final fusion
        concatenated_embeddings = interacted_embeddings.view(batch_size, seq_len, -1)
        final_embedding = self.constraint_fusion(concatenated_embeddings)
        
        results = [final_embedding]
        
        # Uncertainty estimation
        if return_uncertainty and self.uncertainty_quantification:
            uncertainty = self.uncertainty_estimator(concatenated_embeddings)
            results.append(uncertainty)
        
        # Return weights
        if return_weights:
            results.append(constraint_weights)
        
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
    
    def get_constraint_importance(
        self,
        features: torch.Tensor,
        constraint_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Analyze the importance of different constraint types.
        
        Args:
            features: Input features
            constraint_data: Constraint data
            
        Returns:
            Dictionary of constraint importance scores
        """
        if not self.adaptive_weighting:
            # Equal importance if no adaptive weighting
            importance = 1.0 / len(self.constraint_types)
            return {ct: importance for ct in self.constraint_types}
        
        # Compute adaptive weights
        mean_features = torch.mean(features, dim=(0, 1))  # Average over batch and sequence
        constraint_weights = self.weight_predictor(mean_features.unsqueeze(0))
        
        importance_dict = {}
        for i, constraint_type in enumerate(self.constraint_types):
            importance_dict[constraint_type] = constraint_weights[0, i].item()
        
        return importance_dict