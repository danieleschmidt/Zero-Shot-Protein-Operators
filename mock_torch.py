"""
Mock PyTorch implementation for testing without full PyTorch installation.
This allows the code structure to be validated and basic functionality tested.
"""

try:
    import numpy as np
except ImportError:
    import mock_numpy as np
from typing import Any, Optional, Tuple, Union, List


class MockTensor:
    """Mock PyTorch tensor implementation."""
    
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype or np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype or np.float32)
        elif isinstance(data, MockTensor):
            self.data = data.data.copy()
        else:
            self.data = np.array(data, dtype=dtype or np.float32)
        
        self.dtype = dtype or np.float32
        self.device = device or 'cpu'
        self.requires_grad = requires_grad
        self.grad = None
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.shape
    
    def size(self, dim=None):
        if dim is None:
            return self.data.shape
        return self.data.shape[dim]
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data)
        return MockTensor(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data - other.data)
        return MockTensor(self.data - other)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data)
        return MockTensor(self.data * other)
    
    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data / other.data)
        return MockTensor(self.data / other)
    
    def __pow__(self, other):
        return MockTensor(self.data ** other)
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def __setitem__(self, key, value):
        if isinstance(value, MockTensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    def dim(self):
        return len(self.data.shape)
    
    def numel(self):
        return self.data.size
    
    def item(self):
        return self.data.item()
    
    def tolist(self):
        return self.data.tolist()
    
    def numpy(self):
        return self.data
    
    def detach(self):
        result = MockTensor(self.data)
        result.requires_grad = False
        return result
    
    def cpu(self):
        result = MockTensor(self.data)
        result.device = 'cpu'
        return result
    
    def cuda(self):
        result = MockTensor(self.data)
        result.device = 'cuda'
        return result
    
    def clone(self):
        return MockTensor(self.data.copy())
    
    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self.data, axis=dim))
    
    def squeeze(self, dim=None):
        return MockTensor(np.squeeze(self.data, axis=dim))
    
    def reshape(self, *shape):
        return MockTensor(self.data.reshape(shape))
    
    def expand(self, *shape):
        return MockTensor(np.broadcast_to(self.data, shape))
    
    def mean(self, dim=None, keepdim=False):
        return MockTensor(np.mean(self.data, axis=dim, keepdims=keepdim))
    
    def sum(self, dim=None, keepdim=False):
        return MockTensor(np.sum(self.data, axis=dim, keepdims=keepdim))
    
    def std(self, dim=None, keepdim=False):
        return MockTensor(np.std(self.data, axis=dim, keepdims=keepdim))
    
    def max(self, dim=None, keepdim=False):
        if dim is None:
            return MockTensor(np.max(self.data))
        result = np.max(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result)
    
    def min(self, dim=None, keepdim=False):
        if dim is None:
            return MockTensor(np.min(self.data))
        result = np.min(self.data, axis=dim, keepdims=keepdims)
        return MockTensor(result)
    
    def norm(self, dim=None):
        return MockTensor(np.linalg.norm(self.data, axis=dim))
    
    def clamp_(self, min_val, max_val):
        self.data = np.clip(self.data, min_val, max_val)
        return self
    
    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self
    
    def backward(self):
        # Mock backward pass
        pass


def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a mock tensor."""
    return MockTensor(data, dtype, device, requires_grad)


def zeros(*shape, dtype=None, device=None):
    """Create a tensor filled with zeros."""
    return MockTensor(np.zeros(shape), dtype, device)


def ones(*shape, dtype=None, device=None):
    """Create a tensor filled with ones."""
    return MockTensor(np.ones(shape), dtype, device)


def randn(*shape, dtype=None, device=None):
    """Create a tensor filled with random normal values."""
    return MockTensor(np.random.randn(*shape), dtype, device)


def rand(*shape, dtype=None, device=None):
    """Create a tensor filled with random uniform values."""
    return MockTensor(np.random.rand(*shape), dtype, device)


def arange(start, end=None, step=1, dtype=None, device=None):
    """Create a tensor with a range of values."""
    if end is None:
        end = start
        start = 0
    return MockTensor(np.arange(start, end, step), dtype, device)


def linspace(start, end, steps, dtype=None, device=None):
    """Create a tensor with linearly spaced values."""
    return MockTensor(np.linspace(start, end, steps), dtype, device)


def logspace(start, end, steps, base=10.0, dtype=None, device=None):
    """Create a tensor with logarithmically spaced values."""
    return MockTensor(np.logspace(start, end, steps, base=base), dtype, device)


def cat(tensors, dim=0):
    """Concatenate tensors."""
    arrays = [t.data for t in tensors]
    return MockTensor(np.concatenate(arrays, axis=dim))


def stack(tensors, dim=0):
    """Stack tensors."""
    arrays = [t.data for t in tensors]
    return MockTensor(np.stack(arrays, axis=dim))


def cdist(x1, x2):
    """Compute pairwise distances."""
    try:
        from scipy.spatial.distance import cdist as scipy_cdist
        return MockTensor(scipy_cdist(x1.data, x2.data))
    except ImportError:
        # Fallback implementation
        data1 = x1.data if isinstance(x1, MockTensor) else x1
        data2 = x2.data if isinstance(x2, MockTensor) else x2
        
        # Simple pairwise distance for 1D case
        if not isinstance(data1[0], (list, tuple)):
            data1 = [[x] for x in data1]
        if not isinstance(data2[0], (list, tuple)):
            data2 = [[x] for x in data2]
        
        distances = []
        for p1 in data1:
            row = []
            for p2 in data2:
                dist = sum((a - b)**2 for a, b in zip(p1, p2))**0.5
                row.append(dist)
            distances.append(row)
        
        return MockTensor(distances)


def matmul(input, other):
    """Matrix multiplication."""
    return MockTensor(np.matmul(input.data, other.data))


def dot(input, other):
    """Dot product."""
    return MockTensor(np.dot(input.data, other.data))


def norm(input, dim=None):
    """Compute norm."""
    return MockTensor(np.linalg.norm(input.data, axis=dim))


def exp(input):
    """Element-wise exponential."""
    return MockTensor(np.exp(input.data))


def log(input):
    """Element-wise logarithm."""
    return MockTensor(np.log(input.data))


def log10(input):
    """Element-wise base-10 logarithm."""
    return MockTensor(np.log10(input.data))


def sin(input):
    """Element-wise sine."""
    return MockTensor(np.sin(input.data))


def cos(input):
    """Element-wise cosine."""
    return MockTensor(np.cos(input.data))


def sqrt(input):
    """Element-wise square root."""
    return MockTensor(np.sqrt(input.data))


def argmax(input, dim=None):
    """Argmax operation."""
    return MockTensor(np.argmax(input.data, axis=dim))


def argmin(input, dim=None):
    """Argmin operation.""" 
    return MockTensor(np.argmin(input.data, axis=dim))


def all(input, dim=None):
    """All operation."""
    return MockTensor(np.all(input.data, axis=dim))


def any(input, dim=None):
    """Any operation."""
    return MockTensor(np.any(input.data, axis=dim))


def relu(input):
    """ReLU activation."""
    return MockTensor(np.maximum(0, input.data))


def sigmoid(input):
    """Sigmoid activation."""
    return MockTensor(1 / (1 + np.exp(-input.data)))


def svd(input):
    """Singular value decomposition."""
    U, S, Vt = np.linalg.svd(input.data, full_matrices=False)
    return MockTensor(U), MockTensor(S), MockTensor(Vt.T)


def det(input):
    """Determinant."""
    return MockTensor(np.linalg.det(input.data))


class cuda:
    """Mock CUDA functionality."""
    @staticmethod
    def is_available():
        return False


class device:
    """Mock device class."""
    def __init__(self, device_str):
        self.type = device_str
    
    def __str__(self):
        return self.type


def load(path, map_location=None):
    """Mock model loading."""
    import pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return {}


def save(obj, path):
    """Mock model saving."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# Mock nn module
class nn:
    class Module:
        def __init__(self):
            self._parameters = {}
            self._modules = {}
        
        def parameters(self):
            for param in self._parameters.values():
                yield param
        
        def state_dict(self):
            return self._parameters.copy()
        
        def load_state_dict(self, state_dict):
            self._parameters.update(state_dict)
        
        def to(self, device):
            return self
        
        def forward(self, *args, **kwargs):
            raise NotImplementedError
        
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
    
    class Parameter(MockTensor):
        def __init__(self, data):
            super().__init__(data, requires_grad=True)
    
    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(randn(out_features, in_features))
            self.bias = nn.Parameter(zeros(out_features)) if bias else None
        
        def forward(self, input):
            output = matmul(input, self.weight.T)
            if self.bias is not None:
                output = output + self.bias
            return output
    
    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self.layers = layers
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    class ReLU(Module):
        def forward(self, x):
            return relu(x)
    
    class GELU(Module):
        def forward(self, x):
            return MockTensor(0.5 * x.data * (1 + np.tanh(np.sqrt(2/np.pi) * (x.data + 0.044715 * x.data**3))))
    
    class Dropout(Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p
        
        def forward(self, x):
            return x  # No dropout in evaluation mode
    
    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = nn.Parameter(randn(num_embeddings, embedding_dim))
        
        def forward(self, input):
            return MockTensor(self.weight.data[input.data.astype(int)])
    
    class MultiheadAttention(Module):
        def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.batch_first = batch_first
        
        def forward(self, query, key, value):
            # Simplified attention - just return query
            return query, MockTensor(np.ones((query.size(0), query.size(1))))
    
    class functional:
        @staticmethod
        def mse_loss(input, target):
            return MockTensor(np.mean((input.data - target.data) ** 2))
        
        @staticmethod
        def normalize(input, dim=1):
            norms = np.linalg.norm(input.data, axis=dim, keepdims=True)
            return MockTensor(input.data / (norms + 1e-8))
        
        @staticmethod
        def relu(input):
            return relu(input)


# Mock optim module  
class optim:
    class Optimizer:
        def __init__(self, params, lr=1e-3):
            self.param_groups = [{'params': list(params), 'lr': lr}]
        
        def zero_grad(self):
            pass
        
        def step(self):
            pass
    
    class Adam(Optimizer):
        pass
    
    class LBFGS(Optimizer):
        def __init__(self, params, lr=1, max_iter=20):
            super().__init__(params, lr)
            self.max_iter = max_iter
        
        def step(self, closure):
            if closure:
                return closure()


# Aliases for type hints
Tensor = MockTensor
Size = tuple

# Make this module act like torch
import sys
sys.modules['torch'] = sys.modules[__name__]
sys.modules['torch.nn'] = nn
sys.modules['torch.nn.functional'] = nn.functional
sys.modules['torch.optim'] = optim