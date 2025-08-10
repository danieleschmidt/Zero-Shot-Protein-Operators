"""
Minimal numpy mock for basic operations.
"""

import sys


class ndarray:
    """Mock numpy array."""
    
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = data
        else:
            self.data = [data]
        self.dtype = dtype or float
        self._shape = self._compute_shape(self.data)
    
    def _compute_shape(self, data):
        """Compute shape of nested list."""
        if not isinstance(data, (list, tuple)):
            return ()
        if not data:
            return (0,)
        
        shape = [len(data)]
        if isinstance(data[0], (list, tuple)):
            inner_shape = self._compute_shape(data[0])
            shape.extend(inner_shape)
        
        return tuple(shape)
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def size(self):
        size = 1
        for dim in self.shape:
            size *= dim
        return size
    
    def __getitem__(self, key):
        return ndarray(self.data[key])
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def astype(self, dtype):
        return ndarray(self.data, dtype)
    
    def copy(self):
        return ndarray(self.data[:] if isinstance(self.data, list) else self.data)
    
    def tolist(self):
        return self.data
    
    def item(self):
        if self.size == 1:
            return self.data[0] if isinstance(self.data, list) else self.data
        raise ValueError("Can only convert size-1 array to scalar")


def array(data, dtype=None):
    """Create a mock array."""
    return ndarray(data, dtype)


def zeros(shape, dtype=None):
    """Create array filled with zeros."""
    if isinstance(shape, int):
        return ndarray([0.0] * shape, dtype)
    elif isinstance(shape, tuple):
        if len(shape) == 1:
            return ndarray([0.0] * shape[0], dtype)
        else:
            # Simplified for higher dimensions
            total_size = 1
            for dim in shape:
                total_size *= dim
            return ndarray([0.0] * total_size, dtype)
    return ndarray([0.0], dtype)


def ones(shape, dtype=None):
    """Create array filled with ones."""
    if isinstance(shape, int):
        return ndarray([1.0] * shape, dtype)
    elif isinstance(shape, tuple):
        if len(shape) == 1:
            return ndarray([1.0] * shape[0], dtype)
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
            return ndarray([1.0] * total_size, dtype)
    return ndarray([1.0], dtype)


def arange(start, stop=None, step=1, dtype=None):
    """Create array with range of values."""
    if stop is None:
        stop = start
        start = 0
    
    result = []
    current = start
    while current < stop:
        result.append(current)
        current += step
    
    return ndarray(result, dtype)


def linspace(start, stop, num, dtype=None):
    """Create array with linearly spaced values."""
    if num <= 0:
        return ndarray([], dtype)
    if num == 1:
        return ndarray([start], dtype)
    
    step = (stop - start) / (num - 1)
    result = []
    for i in range(num):
        result.append(start + i * step)
    
    return ndarray(result, dtype)


def logspace(start, stop, num, base=10.0, dtype=None):
    """Create array with logarithmically spaced values."""
    if num <= 0:
        return ndarray([], dtype)
    
    linear = linspace(start, stop, num)
    result = []
    for val in linear.data:
        result.append(base ** val)
    
    return ndarray(result, dtype)


def mean(a, axis=None, keepdims=False):
    """Compute mean."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = a
    
    if isinstance(data, (list, tuple)) and len(data) > 0:
        return sum(data) / len(data)
    return 0.0


def std(a, axis=None, keepdims=False):
    """Compute standard deviation."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = a
    
    if isinstance(data, (list, tuple)) and len(data) > 0:
        m = mean(data)
        variance = sum((x - m) ** 2 for x in data) / len(data)
        return variance ** 0.5
    return 0.0


def max(a, axis=None, keepdims=False):
    """Compute maximum."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = a
    
    if isinstance(data, (list, tuple)) and len(data) > 0:
        return max(data)
    return 0.0


def min(a, axis=None, keepdims=False):
    """Compute minimum."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = a
    
    if isinstance(data, (list, tuple)) and len(data) > 0:
        return min(data)
    return 0.0


def sum(a, axis=None, keepdims=False):
    """Compute sum."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = a
    
    if isinstance(data, (list, tuple)):
        return sum(data)
    return data


def concatenate(arrays, axis=0):
    """Concatenate arrays."""
    if not arrays:
        return ndarray([])
    
    result_data = []
    for arr in arrays:
        if isinstance(arr, ndarray):
            result_data.extend(arr.data)
        else:
            result_data.extend(arr)
    
    return ndarray(result_data)


def stack(arrays, axis=0):
    """Stack arrays."""
    return concatenate(arrays, axis)


def expand_dims(a, axis):
    """Add dimension."""
    if isinstance(a, ndarray):
        return ndarray([a.data])
    return ndarray([a])


def squeeze(a, axis=None):
    """Remove dimensions of size 1."""
    if isinstance(a, ndarray):
        return a  # Simplified
    return ndarray(a)


def clip(a, a_min, a_max):
    """Clip values."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        if val < a_min:
            result.append(a_min)
        elif val > a_max:
            result.append(a_max)
        else:
            result.append(val)
    
    return ndarray(result)


def exp(a):
    """Element-wise exponential."""
    import math
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        result.append(math.exp(val))
    
    return ndarray(result)


def log(a):
    """Element-wise logarithm."""
    import math
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        result.append(math.log(val))
    
    return ndarray(result)


def log10(a):
    """Element-wise base-10 logarithm."""
    import math
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        result.append(math.log10(val))
    
    return ndarray(result)


def sin(a):
    """Element-wise sine."""
    import math
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        result.append(math.sin(val))
    
    return ndarray(result)


def cos(a):
    """Element-wise cosine."""
    import math
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        result.append(math.cos(val))
    
    return ndarray(result)


def sqrt(a):
    """Element-wise square root."""
    import math
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    result = []
    for val in data:
        result.append(math.sqrt(val))
    
    return ndarray(result)


def maximum(a, b):
    """Element-wise maximum."""
    return ndarray([max(a, b)])


def argmax(a, axis=None):
    """Return indices of maximum values."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    if isinstance(data, (list, tuple)) and len(data) > 0:
        max_val = max(data)
        return data.index(max_val)
    return 0


def argmin(a, axis=None):
    """Return indices of minimum values."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    if isinstance(data, (list, tuple)) and len(data) > 0:
        min_val = min(data)
        return data.index(min_val)
    return 0


def all(a, axis=None):
    """Test if all elements are true."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    return all(bool(x) for x in data)


def any(a, axis=None):
    """Test if any element is true."""
    if isinstance(a, ndarray):
        data = a.data
    else:
        data = [a]
    
    return any(bool(x) for x in data)


def broadcast_to(array, shape):
    """Broadcast array to shape."""
    return ndarray([0.0] * shape[0] if shape else [0.0])


# Mock linalg module
class linalg:
    @staticmethod
    def norm(x, axis=None):
        """Compute norm."""
        if isinstance(x, ndarray):
            data = x.data
        else:
            data = [x]
        
        if isinstance(data, (list, tuple)):
            return (sum(val ** 2 for val in data)) ** 0.5
        return abs(data)
    
    @staticmethod
    def svd(a, full_matrices=True):
        """Singular value decomposition (mock)."""
        # Return identity-like matrices for mock
        n = len(a.data) if isinstance(a, ndarray) else 1
        return (ndarray([[1.0] * n] * n), 
                ndarray([1.0] * n), 
                ndarray([[1.0] * n] * n))
    
    @staticmethod
    def det(a):
        """Determinant (mock)."""
        return 1.0


# Mock random module
class random:
    @staticmethod
    def randn(*shape):
        """Random normal values."""
        import random as pyrandom
        if len(shape) == 1:
            return ndarray([pyrandom.gauss(0, 1) for _ in range(shape[0])])
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
            return ndarray([pyrandom.gauss(0, 1) for _ in range(total_size)])
    
    @staticmethod
    def rand(*shape):
        """Random uniform values."""
        import random as pyrandom
        if len(shape) == 1:
            return ndarray([pyrandom.random() for _ in range(shape[0])])
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
            return ndarray([pyrandom.random() for _ in range(total_size)])


# Common constants
pi = 3.14159265359
e = 2.71828182846
float32 = float
int32 = int

# Add to sys.modules to act as numpy
sys.modules['numpy'] = sys.modules[__name__]
sys.modules['numpy.linalg'] = linalg
sys.modules['numpy.random'] = random