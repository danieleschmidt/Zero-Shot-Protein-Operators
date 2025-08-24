"""
Real PyTorch integration for protein operators.

This module provides the actual PyTorch implementation to replace mock dependencies,
enabling real neural network operations and GPU acceleration.
"""

import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None
    torch = None
    nn = None
    F = None
    optim = None
    warnings.warn(
        "PyTorch not available. Falling back to mock implementation. "
        "Install PyTorch for full functionality: pip install torch"
    )

# Type hints that work whether torch is available or not
if TYPE_CHECKING and TORCH_AVAILABLE:
    TorchDevice = torch.device
    TorchTensor = torch.Tensor
    TorchDtype = torch.dtype
else:
    TorchDevice = Any
    TorchTensor = Any
    TorchDtype = Any

# GPU detection and configuration
CUDA_AVAILABLE = torch.cuda.is_available() if TORCH_AVAILABLE else False
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu') if TORCH_AVAILABLE else 'cpu'

# MPS (Apple Silicon) detection
MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if TORCH_AVAILABLE else False
if MPS_AVAILABLE and not CUDA_AVAILABLE:
    DEVICE = torch.device('mps')


def _get_mock_torch():
    """Helper function to import mock_torch."""
    import sys
    import os
    mock_torch_path = os.path.join(os.path.dirname(__file__), '../../../mock_torch.py')
    if os.path.dirname(mock_torch_path) not in sys.path:
        sys.path.insert(0, os.path.dirname(mock_torch_path))
    import mock_torch
    return mock_torch


def get_device() -> Union[str, TorchDevice]:
    """Get the best available device for computation."""
    if not TORCH_AVAILABLE:
        return 'cpu'
    
    if CUDA_AVAILABLE:
        return torch.device('cuda')
    elif MPS_AVAILABLE:
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    info = {
        'torch_available': TORCH_AVAILABLE,
        'torch_version': TORCH_VERSION,
        'device': str(get_device()),
        'cuda_available': CUDA_AVAILABLE,
        'mps_available': MPS_AVAILABLE,
        'gpu_count': 0,
        'gpu_memory': []
    }
    
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        info['gpu_count'] = torch.cuda.device_count()
        for i in range(info['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            info['gpu_memory'].append({
                'device_id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / 1024**3,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            })
    
    return info


class TensorUtils:
    """Utilities for tensor operations with fallback support."""
    
    @staticmethod
    def create_tensor(data: Any, dtype: Optional[TorchDtype] = None, 
                     device: Optional[Union[str, TorchDevice]] = None,
                     requires_grad: bool = False) -> Union[TorchTensor, Any]:
        """Create a tensor with fallback to mock if PyTorch unavailable."""
        if not TORCH_AVAILABLE:
            mock_torch = _get_mock_torch()
            return mock_torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        
        device = device or get_device()
        return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def zeros(*shape, dtype: Optional[TorchDtype] = None, 
             device: Optional[Union[str, TorchDevice]] = None) -> Union[TorchTensor, Any]:
        """Create zero tensor with fallback."""
        if not TORCH_AVAILABLE:
            mock_torch = _get_mock_torch()
            return mock_torch.zeros(*shape, dtype=dtype, device=device)
        
        device = device or get_device()
        return torch.zeros(*shape, dtype=dtype, device=device)
    
    @staticmethod
    def ones(*shape, dtype: Optional[TorchDtype] = None,
            device: Optional[Union[str, TorchDevice]] = None) -> Union[TorchTensor, Any]:
        """Create ones tensor with fallback."""
        if not TORCH_AVAILABLE:
            from ..mock_torch import ones
            return ones(*shape, dtype=dtype, device=device)
        
        device = device or get_device()
        return torch.ones(*shape, dtype=dtype, device=device)
    
    @staticmethod
    def randn(*shape, dtype: Optional[TorchDtype] = None,
             device: Optional[Union[str, TorchDevice]] = None) -> Union[TorchTensor, Any]:
        """Create random normal tensor with fallback."""
        if not TORCH_AVAILABLE:
            from ..mock_torch import randn
            return randn(*shape, dtype=dtype, device=device)
        
        device = device or get_device()
        return torch.randn(*shape, dtype=dtype, device=device)
    
    @staticmethod
    def to_device(tensor: Any, device: Optional[Union[str, TorchDevice]] = None) -> Any:
        """Move tensor to device with fallback."""
        if not TORCH_AVAILABLE:
            return tensor  # Mock tensors don't need device movement
        
        device = device or get_device()
        if hasattr(tensor, 'to'):
            return tensor.to(device)
        return tensor
    
    @staticmethod
    def compute_pairwise_distances(x1: Any, x2: Any) -> Any:
        """Compute pairwise distances between tensor rows."""
        if not TORCH_AVAILABLE:
            from ..mock_torch import cdist
            return cdist(x1, x2)
        
        return torch.cdist(x1, x2)
    
    @staticmethod
    def matrix_multiply(a: Any, b: Any) -> Any:
        """Matrix multiplication with fallback."""
        if not TORCH_AVAILABLE:
            from ..mock_torch import matmul
            return matmul(a, b)
        
        return torch.matmul(a, b)


class NetworkUtils:
    """Neural network utilities with fallback support."""
    
    @staticmethod
    def get_activation(name: str) -> Any:
        """Get activation function by name."""
        if not TORCH_AVAILABLE:
            mock_torch = _get_mock_torch()
            mock_nn = mock_torch.nn
            activations = {
                'relu': mock_nn.ReLU(),
                'gelu': mock_nn.GELU(),
                'sigmoid': lambda x: 1 / (1 + (-x).exp()),
                'tanh': lambda x: x.tanh() if hasattr(x, 'tanh') else x,
            }
            return activations.get(name.lower(), mock_nn.ReLU())
        
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    @staticmethod
    def get_optimizer(name: str, parameters, **kwargs) -> Any:
        """Get optimizer by name with fallback."""
        if not TORCH_AVAILABLE:
            mock_torch = _get_mock_torch()
            mock_optim = mock_torch.optim
            optimizers = {
                'adam': mock_optim.Adam,
                'sgd': mock_optim.Optimizer,
                'lbfgs': mock_optim.LBFGS,
            }
            OptimizerClass = optimizers.get(name.lower(), mock_optim.Adam)
            return OptimizerClass(parameters, **kwargs)
        
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'lbfgs': optim.LBFGS,
            'adagrad': optim.Adagrad,
        }
        OptimizerClass = optimizers.get(name.lower(), optim.Adam)
        return OptimizerClass(parameters, **kwargs)
    
    @staticmethod
    def initialize_weights(module: Any, method: str = 'xavier_uniform') -> None:
        """Initialize network weights."""
        if not TORCH_AVAILABLE:
            return  # Skip initialization for mock modules
        
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(m.weight, std=0.02)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def count_parameters(model: Any) -> Dict[str, int]:
        """Count model parameters."""
        if not TORCH_AVAILABLE:
            return {'total': 0, 'trainable': 0}
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }


class ModelManager:
    """Manage model loading, saving, and checkpointing."""
    
    @staticmethod
    def save_model(model: Any, optimizer: Any, path: str, 
                  epoch: int = 0, loss: float = 0.0, metrics: Optional[Dict] = None) -> None:
        """Save model checkpoint."""
        if not TORCH_AVAILABLE:
            import pickle
            checkpoint = {
                'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else {},
                'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else {},
                'epoch': epoch,
                'loss': loss,
                'metrics': metrics or {}
            }
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
            return
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics or {},
            'torch_version': TORCH_VERSION
        }
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_model(model: Any, path: str, 
                  optimizer: Optional[Any] = None,
                  device: Optional[Union[str, TorchDevice]] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        device = device or get_device()
        
        if not TORCH_AVAILABLE:
            import pickle
            try:
                with open(path, 'rb') as f:
                    checkpoint = pickle.load(f)
                if hasattr(model, 'load_state_dict'):
                    model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer and hasattr(optimizer, 'load_state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                return checkpoint
            except:
                return {'epoch': 0, 'loss': float('inf'), 'metrics': {}}
        
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    @staticmethod
    def model_to_device(model: Any, device: Optional[Union[str, TorchDevice]] = None) -> Any:
        """Move model to specified device."""
        if not TORCH_AVAILABLE:
            return model
        
        device = device or get_device()
        return model.to(device)


class LossUtils:
    """Loss function utilities."""
    
    @staticmethod
    def get_loss_function(name: str, **kwargs) -> Any:
        """Get loss function by name."""
        if not TORCH_AVAILABLE:
            from ..mock_torch import nn as mock_nn
            # Return a simple mock loss
            return lambda x, y: mock_nn.functional.mse_loss(x, y)
        
        losses = {
            'mse': nn.MSELoss(**kwargs),
            'l1': nn.L1Loss(**kwargs),
            'huber': nn.HuberLoss(**kwargs),
            'cross_entropy': nn.CrossEntropyLoss(**kwargs),
            'bce': nn.BCELoss(**kwargs),
            'bce_with_logits': nn.BCEWithLogitsLoss(**kwargs),
            'nll': nn.NLLLoss(**kwargs),
            'kl_div': nn.KLDivLoss(**kwargs),
        }
        return losses.get(name.lower(), nn.MSELoss(**kwargs))
    
    @staticmethod
    def physics_informed_loss(predictions: Any, targets: Any, 
                            physics_constraints: Optional[Any] = None,
                            physics_weight: float = 0.1) -> Any:
        """Compute physics-informed loss."""
        # Data fitting loss
        data_loss = F.mse_loss(predictions, targets) if TORCH_AVAILABLE else predictions - targets
        
        # Physics constraint loss
        physics_loss = TensorUtils.zeros(1)
        if physics_constraints is not None:
            # This would be implemented based on specific physics constraints
            # For now, return zero constraint loss
            pass
        
        return data_loss + physics_weight * physics_loss


def configure_torch_settings() -> None:
    """Configure PyTorch settings for optimal performance."""
    if not TORCH_AVAILABLE:
        return
    
    # Set number of threads for CPU operations
    if torch.get_num_threads() < 4:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    
    # Enable optimized attention if available
    if hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    np.random.seed(42)


def print_system_info() -> None:
    """Print comprehensive system information."""
    info = get_device_info()
    
    print("=== PyTorch Integration Status ===")
    print(f"PyTorch Available: {info['torch_available']}")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"Device: {info['device']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")
    print(f"GPU Count: {info['gpu_count']}")
    
    if info['gpu_memory']:
        print("\nGPU Information:")
        for gpu in info['gpu_memory']:
            print(f"  GPU {gpu['device_id']}: {gpu['name']}")
            print(f"    Memory: {gpu['total_memory_gb']:.1f} GB")
            print(f"    Compute: {gpu['major']}.{gpu['minor']}")
            print(f"    Multiprocessors: {gpu['multi_processor_count']}")
    
    print("=" * 35)


# Initialize PyTorch settings on import
configure_torch_settings()

# Make commonly used functions available at module level  
def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create tensor at module level."""
    return TensorUtils.create_tensor(data, dtype, device, requires_grad)

def zeros(*shape, dtype=None, device=None):
    """Create zeros tensor at module level."""
    return TensorUtils.zeros(*shape, dtype=dtype, device=device)

def ones(*shape, dtype=None, device=None):
    """Create ones tensor at module level."""
    return TensorUtils.ones(*shape, dtype=dtype, device=device)

def randn(*shape, dtype=None, device=None):
    """Create randn tensor at module level."""
    return TensorUtils.randn(*shape, dtype=dtype, device=device)

def to_device(tensor_obj, device=None):
    """Move tensor to device at module level."""
    return TensorUtils.to_device(tensor_obj, device)

__all__ = [
    'TORCH_AVAILABLE', 'CUDA_AVAILABLE', 'MPS_AVAILABLE', 'DEVICE',
    'get_device', 'get_device_info', 'configure_torch_settings', 'print_system_info',
    'TensorUtils', 'NetworkUtils', 'ModelManager', 'LossUtils',
    'tensor', 'zeros', 'ones', 'randn', 'to_device'
]