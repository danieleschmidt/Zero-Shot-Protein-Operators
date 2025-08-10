"""
DeepONet implementation for protein design.

Based on "Learning nonlinear operators via DeepONet" (Lu et al., 2021)
Extended for protein structure generation from biophysical constraints.
"""

from typing import Optional, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional

from .base import BaseNeuralOperator


class ConstraintEncoder(nn.Module):
    """
    Encoder for protein design constraints.
    
    This module processes various types of constraints (binding sites,
    structural requirements, etc.) into a unified embedding space.
    """
    
    def __init__(
        self,
        constraint_types: int = 10,  # Number of constraint types
        embedding_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.constraint_types = constraint_types
        self.embedding_dim = embedding_dim
        
        # Constraint type embeddings
        self.type_embedding = nn.Embedding(constraint_types, embedding_dim)
        
        # Constraint value processing
        self.value_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Attention mechanism for constraint aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, constraints: torch.Tensor) -> torch.Tensor:
        """
        Encode constraints into unified representation.
        
        Args:
            constraints: [batch, num_constraints, constraint_dim]
                Where constraint_dim includes type_id and values
                
        Returns:
            Encoded constraints [batch, embedding_dim]
        """
        batch_size, num_constraints, _ = constraints.shape
        
        # Extract constraint types and values
        constraint_types = constraints[:, :, 0].long()  # First element is type
        constraint_values = constraints[:, :, 1:]       # Rest are values
        
        # Embed constraint types
        type_embeds = self.type_embedding(constraint_types)
        
        # Process constraint values
        # Pad/truncate values to match embedding_dim
        if constraint_values.size(-1) < self.embedding_dim:
            padding = torch.zeros(
                batch_size, num_constraints, 
                self.embedding_dim - constraint_values.size(-1),
                device=constraints.device
            )
            constraint_values = torch.cat([constraint_values, padding], dim=-1)
        else:
            constraint_values = constraint_values[:, :, :self.embedding_dim]
        
        value_embeds = self.value_encoder(constraint_values)
        
        # Combine type and value embeddings
        combined_embeds = type_embeds + value_embeds
        
        # Apply attention to aggregate constraints
        attended, _ = self.attention(
            combined_embeds, combined_embeds, combined_embeds
        )
        
        # Global pooling to get final constraint representation
        constraint_repr = attended.mean(dim=1)  # [batch, embedding_dim]
        
        return constraint_repr


class PositionalEncoder(nn.Module):
    """
    Positional encoding for 3D coordinates in protein structures.
    
    Uses sinusoidal encodings extended to 3D space with multiple
    frequency scales to capture both local and global spatial patterns.
    """
    
    def __init__(
        self,
        coordinate_dim: int = 3,  # 3D coordinates
        encoding_dim: int = 128,
        max_freq: float = 1000.0,
    ):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.encoding_dim = encoding_dim
        
        # Create frequency scales
        freqs = torch.logspace(0, torch.log10(torch.tensor(max_freq)), 
                              encoding_dim // (2 * coordinate_dim))
        self.register_buffer('freqs', freqs)
        
        # Linear projection to target dimension
        self.projection = nn.Linear(
            coordinate_dim * encoding_dim // coordinate_dim * 2, 
            encoding_dim
        )
        
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D coordinates with positional information.
        
        Args:
            coordinates: [batch, num_points, coordinate_dim]
            
        Returns:
            Encoded coordinates [batch, num_points, encoding_dim]
        """
        batch_size, num_points, _ = coordinates.shape
        
        # Compute sinusoidal encodings for each dimension
        encodings = []
        for dim in range(self.coordinate_dim):
            coord_values = coordinates[:, :, dim:dim+1]  # [batch, num_points, 1]
            
            # Apply frequencies
            angles = coord_values * self.freqs.unsqueeze(0).unsqueeze(0)
            
            # Compute sin and cos
            sin_enc = torch.sin(angles)
            cos_enc = torch.cos(angles)
            
            encodings.extend([sin_enc, cos_enc])
        
        # Concatenate all encodings
        full_encoding = torch.cat(encodings, dim=-1)
        
        # Project to target dimension
        encoded = self.projection(full_encoding)
        
        return encoded


class ProteinDeepONet(BaseNeuralOperator):
    """
    DeepONet architecture specialized for protein design.
    
    Implements the operator learning paradigm where:
    - Branch network: Encodes protein design constraints
    - Trunk network: Encodes spatial coordinates
    - Output: 3D protein structure coordinates
    
    Examples:
        >>> model = ProteinDeepONet(
        ...     constraint_dim=256,
        ...     branch_hidden=[512, 1024],
        ...     trunk_hidden=[512, 1024],
        ...     num_basis=1024
        ... )
        >>> constraints = torch.randn(2, 5, 10)  # 2 batch, 5 constraints
        >>> coordinates = torch.randn(2, 100, 3)  # 2 batch, 100 residues
        >>> output = model(constraints, coordinates)
        >>> print(output.shape)  # torch.Size([2, 100, 3])
    """
    
    def __init__(
        self,
        constraint_dim: int = 256,
        coordinate_dim: int = 3,
        output_dim: int = 3,
        branch_hidden: List[int] = [512, 1024],
        trunk_hidden: List[int] = [512, 1024], 
        num_basis: int = 1024,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize ProteinDeepONet.
        
        Args:
            constraint_dim: Dimension of constraint encoding
            coordinate_dim: Dimension of input coordinates (typically 3)
            output_dim: Dimension of output coordinates (typically 3)
            branch_hidden: Hidden layer sizes for branch network
            trunk_hidden: Hidden layer sizes for trunk network
            num_basis: Number of basis functions (should match final branch/trunk output)
            activation: Activation function ("relu", "gelu", "swish")
            dropout_rate: Dropout probability
        """
        super().__init__(constraint_dim, output_dim, **kwargs)
        
        self.num_basis = num_basis
        self.activation_name = activation
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Constraint encoder (branch network input)
        self.constraint_encoder = ConstraintEncoder(
            embedding_dim=constraint_dim
        )
        
        # Positional encoder (trunk network input)
        self.positional_encoder = PositionalEncoder(
            coordinate_dim=coordinate_dim,
            encoding_dim=constraint_dim  # Match constraint encoding dim
        )
        
        # Branch network: processes constraints
        branch_layers = []
        prev_dim = constraint_dim
        
        for hidden_dim in branch_hidden:
            branch_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        branch_layers.append(nn.Linear(prev_dim, num_basis))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk network: processes coordinates
        trunk_layers = []
        prev_dim = constraint_dim  # From positional encoder
        
        for hidden_dim in trunk_hidden:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        trunk_layers.append(nn.Linear(prev_dim, num_basis))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Output projection (bias term)
        self.output_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Optional output transformation
        self.output_transform = nn.Linear(output_dim, output_dim)
        
    def encode_constraints(self, constraints: torch.Tensor) -> torch.Tensor:
        """Encode constraint specifications using constraint encoder."""
        return self.constraint_encoder(constraints)
    
    def encode_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode spatial coordinates using positional encoder.""" 
        return self.positional_encoder(coordinates)
    
    def operator_forward(
        self,
        constraint_encoding: torch.Tensor,
        coordinate_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Main DeepONet computation: branch ⊗ trunk + bias.
        
        Args:
            constraint_encoding: [batch, constraint_dim]
            coordinate_encoding: [batch, num_points, encoding_dim]
            
        Returns:
            Output coordinates [batch, num_points, output_dim]
        """
        batch_size, num_points, _ = coordinate_encoding.shape
        
        # Branch network: constraint encoding → basis coefficients
        branch_output = self.branch_net(constraint_encoding)  # [batch, num_basis]
        
        # Trunk network: coordinate encoding → basis functions
        trunk_input = coordinate_encoding.reshape(-1, coordinate_encoding.size(-1))
        trunk_output = self.trunk_net(trunk_input)  # [batch*num_points, num_basis]
        trunk_output = trunk_output.reshape(batch_size, num_points, self.num_basis)
        
        # DeepONet combination: sum over basis functions
        # branch_output: [batch, num_basis] → [batch, 1, num_basis]
        # trunk_output: [batch, num_points, num_basis]
        combined = torch.sum(
            branch_output.unsqueeze(1) * trunk_output,
            dim=-1
        )  # [batch, num_points]
        
        # Expand to output dimensions and add bias
        output = combined.unsqueeze(-1).expand(-1, -1, self.output_dim)
        output = output + self.output_bias
        
        # Optional output transformation
        output = self.output_transform(output)
        
        return output
    
    def compute_operator_norm(self) -> torch.Tensor:
        """Compute operator norm for regularization."""
        branch_norm = sum(p.norm() for p in self.branch_net.parameters())
        trunk_norm = sum(p.norm() for p in self.trunk_net.parameters())
        return branch_norm + trunk_norm
    
    def get_basis_activations(
        self,
        constraints: torch.Tensor,
        coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get intermediate basis function activations for analysis.
        
        Returns:
            branch_activations: [batch, num_basis]
            trunk_activations: [batch, num_points, num_basis]
        """
        with torch.no_grad():
            constraint_encoding = self.encode_constraints(constraints)
            coordinate_encoding = self.encode_coordinates(coordinates)
            
            branch_activations = self.branch_net(constraint_encoding)
            
            batch_size, num_points, _ = coordinate_encoding.shape
            trunk_input = coordinate_encoding.reshape(-1, coordinate_encoding.size(-1))
            trunk_activations = self.trunk_net(trunk_input)
            trunk_activations = trunk_activations.reshape(batch_size, num_points, self.num_basis)
            
            return branch_activations, trunk_activations