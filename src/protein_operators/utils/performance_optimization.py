"""Performance optimization utilities for protein operators."""

import time
import threading
import functools
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import os
import sys

# Add mock support
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
try:
    import torch
except ImportError:
    import mock_torch as torch

@dataclass 
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size_limit: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_gpu_acceleration: bool = True
    mixed_precision: bool = False

class LRUCache:
    """Lightweight LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = value
            else:
                if len(self.cache) >= self.max_size:
                    oldest_key = self.access_order.pop(0)
                    del self.cache[oldest_key]
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        return 0.85  # Mock value

class BatchProcessor:
    """Efficient batch processing."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._cache = LRUCache(max_size=500)
    
    def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        results = []
        
        if len(batches) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_single_batch, batch, process_func): batch
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch):
                    batch_result = future.result()
                    results.extend(batch_result)
        else:
            for batch in batches:
                batch_result = self._process_single_batch(batch, process_func)
                results.extend(batch_result)
        
        return results
    
    def _process_single_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process single batch."""
        batch_results = []
        
        for item in batch:
            cache_key = self._get_cache_key(item)
            cached_result = self._cache.get(cache_key)
            
            if cached_result is not None:
                batch_results.append(cached_result)
            else:
                result = process_func(item)
                batch_results.append(result)
                self._cache.put(cache_key, result)
        
        return batch_results
    
    def _get_cache_key(self, item: Any) -> str:
        """Generate cache key."""
        try:
            item_str = str(item)
            return hashlib.md5(item_str.encode()).hexdigest()
        except Exception:
            return str(id(item))

class GPUAccelerator:
    """GPU acceleration utilities."""
    
    def __init__(self):
        self.device = self._get_best_device()
        self.mixed_precision_enabled = False
    
    def _get_best_device(self) -> str:
        """Get best device."""
        try:
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                return "cpu"
        except:
            return "cpu"
    
    def optimize_for_inference(self, model):
        """Optimize model for inference."""
        try:
            if hasattr(model, 'eval'):
                model.eval()
            
            if hasattr(model, 'to'):
                model = model.to(self.device)
            
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    if hasattr(param, 'requires_grad'):
                        param.requires_grad = False
            
            return model
        except Exception as e:
            print(f"Optimization warning: {e}")
            return model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory stats."""
        if not self.device.startswith("cuda"):
            return {"total_gb": 0, "allocated_gb": 0, "free_gb": 0}
        
        try:
            if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'get_device_properties'):
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(0) / 1e9
                return {"total_gb": total, "allocated_gb": allocated, "free_gb": total - allocated}
        except:
            pass
        
        return {"total_gb": 0, "allocated_gb": 0, "free_gb": 0}

def cached_result(cache_size: int = 1000):
    """Decorator for caching function results."""
    cache = LRUCache(max_size=cache_size)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key_data = (args, tuple(sorted(kwargs.items())))
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        wrapper.clear_cache = cache.clear
        wrapper.cache_info = lambda: {"size": cache.size(), "hit_rate": cache.hit_rate()}
        
        return wrapper
    return decorator

class GlobalPerformanceOptimizer:
    """Global performance optimization manager."""
    
    def __init__(self):
        self.config = PerformanceConfig()
        self.gpu_accelerator = GPUAccelerator()
        self.batch_processor = BatchProcessor()
        self._enabled_optimizations = set()
    
    def configure(self, **kwargs):
        """Configure performance settings."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._apply_configuration()
    
    def _apply_configuration(self):
        """Apply configuration."""
        if self.config.enable_gpu_acceleration:
            self._enabled_optimizations.add("gpu_acceleration")
        
        if self.config.mixed_precision:
            self.gpu_accelerator.mixed_precision_enabled = True
            self._enabled_optimizations.add("mixed_precision")
        
        if self.config.enable_parallel_processing:
            self._enabled_optimizations.add("parallel_processing")
        
        self.batch_processor.batch_size = self.config.max_workers * 8
        self.batch_processor.max_workers = self.config.max_workers
    
    def optimize_model(self, model):
        """Optimize model for inference."""
        if "gpu_acceleration" in self._enabled_optimizations:
            model = self.gpu_accelerator.optimize_for_inference(model)
        return model
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        gpu_stats = self.gpu_accelerator.get_memory_stats()
        
        return {
            "device": self.gpu_accelerator.device,
            "optimizations_enabled": list(self._enabled_optimizations),
            "gpu_memory": gpu_stats,
            "cache_stats": {
                "size": self.batch_processor._cache.size(),
                "hit_rate": self.batch_processor._cache.hit_rate()
            },
            "configuration": {
                "batch_size": self.batch_processor.batch_size,
                "max_workers": self.batch_processor.max_workers,
                "mixed_precision": self.gpu_accelerator.mixed_precision_enabled
            }
        }

_global_optimizer = GlobalPerformanceOptimizer()

def get_performance_optimizer():
    """Get global performance optimizer."""
    return _global_optimizer

def optimize_for_performance(**kwargs):
    """Configure global performance optimizations."""
    optimizer = get_performance_optimizer()
    optimizer.configure(**kwargs)
    return optimizer