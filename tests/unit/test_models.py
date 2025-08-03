"""
Unit tests for neural operator models.

Tests the DeepONet and other neural operator implementations
for correctness, performance, and edge cases.
"""

import pytest
import torch
import numpy as np

from protein_operators.models import ProteinDeepONet, BaseNeuralOperator
from protein_operators.models.deeponet import PositionalEncoding


class TestBaseNeuralOperator:
    """Test base neural operator functionality."""
    
    def test_initialization(self):
        """Test base operator initialization."""
        # Create a simple subclass for testing
        class SimpleOperator(BaseNeuralOperator):
            def encode_constraints(self, constraints):
                return torch.randn(constraints.shape[0], 256)
            
            def encode_coordinates(self, coordinates):
                return torch.randn(coordinates.shape[0], coordinates.shape[1], 256)
            
            def operator_forward(self, constraint_encoding, coordinate_encoding):
                return torch.randn(coordinate_encoding.shape[0], coordinate_encoding.shape[1], 3)
        
        operator = SimpleOperator(input_dim=128, output_dim=3)
        
        assert operator.input_dim == 128
        assert operator.output_dim == 3
    
    def test_physics_loss_computation(self):
        """Test physics loss computation."""
        class SimpleOperator(BaseNeuralOperator):
            def encode_constraints(self, constraints):
                return torch.randn(constraints.shape[0], 256)
            
            def encode_coordinates(self, coordinates):
                return torch.randn(coordinates.shape[0], coordinates.shape[1], 256)
            
            def operator_forward(self, constraint_encoding, coordinate_encoding):
                return torch.randn(coordinate_encoding.shape[0], coordinate_encoding.shape[1], 3)
        
        operator = SimpleOperator(input_dim=128)
        
        # Test with simple coordinates
        coords = torch.randn(2, 10, 3)
        constraints = torch.randn(2, 128)
        
        loss = operator.compute_physics_loss(coords, constraints)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss >= 0  # Loss should be non-negative
    
    def test_model_info(self):
        """Test model information retrieval."""
        class SimpleOperator(BaseNeuralOperator):
            def encode_constraints(self, constraints):
                return torch.randn(constraints.shape[0], 256)
            
            def encode_coordinates(self, coordinates):
                return torch.randn(coordinates.shape[0], coordinates.shape[1], 256)
            
            def operator_forward(self, constraint_encoding, coordinate_encoding):
                return torch.randn(coordinate_encoding.shape[0], coordinate_encoding.shape[1], 3)
        
        operator = SimpleOperator(input_dim=128, output_dim=3, test_param="test")
        info = operator.get_model_info()
        
        assert "model_type" in info
        assert "input_dim" in info
        assert "output_dim" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["input_dim"] == 128
        assert info["output_dim"] == 3


class TestProteinDeepONet:
    """Test ProteinDeepONet implementation."""
    
    def test_initialization(self):
        """Test DeepONet initialization."""
        model = ProteinDeepONet(
            constraint_dim=256,
            coordinate_dim=3,
            output_dim=3,
            branch_hidden=[512, 1024],
            trunk_hidden=[512, 1024],
            num_basis=1024
        )
        
        assert model.num_basis == 1024
        assert model.activation_name == "relu"
        assert hasattr(model, "constraint_encoder")
        assert hasattr(model, "positional_encoder")
        assert hasattr(model, "branch_net")
        assert hasattr(model, "trunk_net")
    
    def test_forward_pass(self):
        """Test forward pass through DeepONet."""
        model = ProteinDeepONet(
            constraint_dim=256,
            num_basis=512,
            branch_hidden=[256],
            trunk_hidden=[256]
        )
        
        # Create test inputs
        batch_size = 2
        num_constraints = 3
        num_points = 20
        
        constraints = torch.randn(batch_size, num_constraints, 10)
        coordinates = torch.randn(batch_size, num_points, 3)
        
        # Forward pass
        output = model(constraints, coordinates)
        
        assert output.shape == (batch_size, num_points, 3)
        assert torch.isfinite(output).all()
    
    def test_constraint_encoding(self):
        """Test constraint encoding."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        constraints = torch.randn(2, 5, 10)
        encoding = model.encode_constraints(constraints)
        
        assert encoding.shape == (2, 256)
        assert torch.isfinite(encoding).all()
    
    def test_coordinate_encoding(self):
        """Test coordinate encoding."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        coordinates = torch.randn(2, 15, 3)
        encoding = model.encode_coordinates(coordinates)
        
        assert encoding.shape == (2, 15, 256)
        assert torch.isfinite(encoding).all()
    
    def test_operator_forward(self):
        """Test operator forward computation."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        constraint_encoding = torch.randn(2, 256)
        coordinate_encoding = torch.randn(2, 20, 256)
        
        output = model.operator_forward(constraint_encoding, coordinate_encoding)
        
        assert output.shape == (2, 20, 3)
        assert torch.isfinite(output).all()
    
    def test_basis_activations(self):
        """Test basis activation retrieval."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        constraints = torch.randn(1, 3, 10)
        coordinates = torch.randn(1, 10, 3)
        
        branch_act, trunk_act = model.get_basis_activations(constraints, coordinates)
        
        assert branch_act.shape == (1, 512)
        assert trunk_act.shape == (1, 10, 512)
        assert torch.isfinite(branch_act).all()
        assert torch.isfinite(trunk_act).all()
    
    def test_operator_norm(self):
        """Test operator norm computation."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        norm = model.compute_operator_norm()
        
        assert isinstance(norm, torch.Tensor)
        assert norm.dim() == 0  # Scalar
        assert norm >= 0
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["relu", "gelu", "swish"]
        
        for activation in activations:
            model = ProteinDeepONet(
                constraint_dim=128,
                num_basis=256,
                activation=activation
            )
            
            constraints = torch.randn(1, 2, 8)
            coordinates = torch.randn(1, 5, 3)
            
            output = model(constraints, coordinates)
            
            assert output.shape == (1, 5, 3)
            assert torch.isfinite(output).all()
    
    def test_variable_input_sizes(self):
        """Test model with variable input sizes."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        # Test different protein lengths
        lengths = [5, 10, 50, 100]
        
        for length in lengths:
            constraints = torch.randn(1, 2, 10)
            coordinates = torch.randn(1, length, 3)
            
            output = model(constraints, coordinates)
            
            assert output.shape == (1, length, 3)
            assert torch.isfinite(output).all()
    
    def test_batch_processing(self):
        """Test batch processing."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            constraints = torch.randn(batch_size, 3, 10)
            coordinates = torch.randn(batch_size, 20, 3)
            
            output = model(constraints, coordinates)
            
            assert output.shape == (batch_size, 20, 3)
            assert torch.isfinite(output).all()


class TestPositionalEncoding:
    """Test positional encoding component."""
    
    def test_initialization(self):
        """Test positional encoding initialization."""
        pos_enc = PositionalEncoding(input_dim=3, encoding_dim=128)
        
        assert pos_enc.input_dim == 3
        assert pos_enc.encoding_dim == 128
        assert hasattr(pos_enc, "freqs")
    
    def test_forward_pass(self):
        """Test positional encoding forward pass."""
        pos_enc = PositionalEncoding(input_dim=3, encoding_dim=128)
        
        coordinates = torch.randn(10, 3)
        encoded = pos_enc(coordinates)
        
        assert encoded.shape[0] == 10
        assert torch.isfinite(encoded).all()
    
    def test_different_dimensions(self):
        """Test with different input/output dimensions."""
        test_cases = [
            (1, 64),
            (2, 128),
            (3, 256),
            (4, 512)
        ]
        
        for input_dim, encoding_dim in test_cases:
            pos_enc = PositionalEncoding(input_dim=input_dim, encoding_dim=encoding_dim)
            
            coordinates = torch.randn(5, input_dim)
            encoded = pos_enc(coordinates)
            
            assert encoded.shape[0] == 5
            assert torch.isfinite(encoded).all()
    
    def test_deterministic_output(self):
        """Test that encoding is deterministic."""
        pos_enc = PositionalEncoding(input_dim=3, encoding_dim=128)
        
        coordinates = torch.randn(5, 3)
        
        encoded1 = pos_enc(coordinates)
        encoded2 = pos_enc(coordinates)
        
        assert torch.allclose(encoded1, encoded2)


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_gradient_flow(self):
        """Test gradient flow through model."""
        model = ProteinDeepONet(constraint_dim=128, num_basis=256)
        
        constraints = torch.randn(2, 2, 8, requires_grad=True)
        coordinates = torch.randn(2, 10, 3, requires_grad=True)
        
        output = model(constraints, coordinates)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert constraints.grad is not None
        assert coordinates.grad is not None
        assert torch.isfinite(constraints.grad).all()
        assert torch.isfinite(coordinates.grad).all()
    
    def test_training_mode(self):
        """Test model behavior in training vs eval mode."""
        model = ProteinDeepONet(constraint_dim=128, num_basis=256, dropout_rate=0.1)
        
        constraints = torch.randn(1, 2, 8)
        coordinates = torch.randn(1, 10, 3)
        
        # Training mode
        model.train()
        output_train = model(constraints, coordinates)
        
        # Evaluation mode
        model.eval()
        output_eval = model(constraints, coordinates)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test memory efficiency with large inputs."""
        model = ProteinDeepONet(constraint_dim=256, num_basis=512)
        
        # Large protein (1000 residues)
        constraints = torch.randn(1, 5, 20)
        coordinates = torch.randn(1, 1000, 3)
        
        # Should not cause memory issues
        with torch.no_grad():
            output = model(constraints, coordinates)
        
        assert output.shape == (1, 1000, 3)
        assert torch.isfinite(output).all()
    
    def test_device_compatibility(self):
        """Test model works on different devices."""
        model = ProteinDeepONet(constraint_dim=128, num_basis=256)
        
        # CPU test
        constraints = torch.randn(1, 2, 8)
        coordinates = torch.randn(1, 10, 3)
        
        output_cpu = model(constraints, coordinates)
        assert output_cpu.device.type == "cpu"
        
        # GPU test (if available)
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            constraints_gpu = constraints.cuda()
            coordinates_gpu = coordinates.cuda()
            
            output_gpu = model_gpu(constraints_gpu, coordinates_gpu)
            assert output_gpu.device.type == "cuda"
            
            # Results should be similar (within numerical precision)
            assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-5)