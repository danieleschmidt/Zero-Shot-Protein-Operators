"""
Advanced performance optimization system for protein design.

This module provides:
- Intelligent caching with eviction policies
- Concurrent processing and parallelization
- Memory pool management
- Computational optimization
- Auto-scaling capabilities
"""

import asyncio
import concurrent.futures
import hashlib
import pickle
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from functools import wraps, lru_cache
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
    import torch.multiprocessing as mp
except ImportError:
    import mock_torch as torch
    import multiprocessing as mp


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ComputeDevice(Enum):
    """Compute device types."""
    CPU = "cpu"
    GPU = "gpu"
    MULTI_GPU = "multi_gpu"
    AUTO = "auto"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    access_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None


@dataclass
class ComputeTask:
    """Compute task definition."""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    device_preference: ComputeDevice = ComputeDevice.AUTO
    memory_requirement: int = 0  # bytes
    estimated_duration: float = 0  # seconds


@dataclass
class ResourcePool:
    """Resource pool configuration."""
    max_cpu_workers: int = 4
    max_gpu_workers: int = 1
    max_memory_per_worker: int = 1024 * 1024 * 1024  # 1GB
    worker_timeout: float = 300.0  # 5 minutes


class IntelligentCache:
    """
    High-performance cache with multiple eviction policies and optimization.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 100 * 1024 * 1024,  # 100MB
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE,
        ttl_default: Optional[float] = None,
        enable_compression: bool = True
    ):
        """
        Initialize intelligent cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory: Maximum memory usage in bytes
            eviction_policy: Eviction policy to use
            ttl_default: Default TTL in seconds
            enable_compression: Whether to compress large entries
        """
        self.max_size = max_size
        self.max_memory = max_memory
        self.eviction_policy = eviction_policy
        self.ttl_default = ttl_default
        self.enable_compression = enable_compression
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._frequency_count: Dict[str, int] = {}  # For LFU
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory = 0
    
    def _generate_key(self, key_data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(key_data, str):
            return key_data
        
        # Hash complex objects
        try:
            key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(key_str).hexdigest()
        except Exception:
            return str(hash(key_data))
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return sys.getsizeof(value)
    
    def _compress_value(self, value: Any) -> Any:
        """Compress value if beneficial."""
        if not self.enable_compression:
            return value
        
        try:
            import gzip
            serialized = pickle.dumps(value)
            if len(serialized) > 1024:  # Only compress large objects
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # 20% compression minimum
                    return ('compressed', compressed)
        except Exception:
            pass
        
        return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Decompress value if needed."""
        if isinstance(value, tuple) and len(value) == 2 and value[0] == 'compressed':
            try:
                import gzip
                return pickle.loads(gzip.decompress(value[1]))
            except Exception:
                pass
        
        return value
    
    def _should_evict_entry(self, entry: CacheEntry) -> bool:
        """Check if entry should be evicted based on TTL."""
        if entry.ttl is None:
            return False
        
        return time.time() - entry.creation_time > entry.ttl
    
    def _evict_entries(self, target_size: int = None, target_memory: int = None):
        """Evict entries based on policy."""
        if not self._cache:
            return
        
        target_size = target_size or self.max_size - 1
        target_memory = target_memory or self.max_memory
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._should_evict_entry(entry)
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        # Check if we still need to evict
        if len(self._cache) <= target_size and self.current_memory <= target_memory:
            return
        
        # Apply eviction policy
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            self._evict_lru(target_size, target_memory)
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            self._evict_lfu(target_size, target_memory)
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            self._evict_fifo(target_size, target_memory)
        elif self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            self._evict_adaptive(target_size, target_memory)
    
    def _evict_lru(self, target_size: int, target_memory: int):
        """Evict least recently used entries."""
        while (len(self._cache) > target_size or self.current_memory > target_memory) and self._access_order:
            key = self._access_order[0]
            self._remove_entry(key)
    
    def _evict_lfu(self, target_size: int, target_memory: int):
        """Evict least frequently used entries."""
        while len(self._cache) > target_size or self.current_memory > target_memory:
            if not self._cache:
                break
            
            # Find least frequently used
            min_freq = min(self._frequency_count[key] for key in self._cache.keys())
            lfu_keys = [key for key in self._cache.keys() if self._frequency_count[key] == min_freq]
            
            # Among LFU entries, remove oldest
            oldest_key = min(lfu_keys, key=lambda k: self._cache[k].creation_time)
            self._remove_entry(oldest_key)
    
    def _evict_fifo(self, target_size: int, target_memory: int):
        """Evict first-in-first-out entries."""
        while len(self._cache) > target_size or self.current_memory > target_memory:
            if not self._cache:
                break
            
            # Find oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].creation_time)
            self._remove_entry(oldest_key)
    
    def _evict_adaptive(self, target_size: int, target_memory: int):
        """Adaptive eviction combining multiple factors."""
        while len(self._cache) > target_size or self.current_memory > target_memory:
            if not self._cache:
                break
            
            # Score entries based on multiple factors
            scores = {}
            current_time = time.time()
            
            for key, entry in self._cache.items():
                age_factor = current_time - entry.creation_time
                access_factor = 1.0 / max(entry.access_count, 1)
                recency_factor = current_time - entry.last_access
                size_factor = entry.size_bytes / max(self.current_memory, 1)
                
                # Higher score = more likely to evict
                scores[key] = (age_factor * 0.3 + 
                              access_factor * 0.4 + 
                              recency_factor * 0.2 + 
                              size_factor * 0.1)
            
            # Remove entry with highest score
            victim_key = max(scores.keys(), key=lambda k: scores[k])
            self._remove_entry(victim_key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self.current_memory -= entry.size_bytes
            del self._cache[key]
            
            if key in self._access_order:
                self._access_order.remove(key)
            if key in self._frequency_count:
                del self._frequency_count[key]
            
            self.evictions += 1
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get value from cache."""
        key_str = self._generate_key(key)
        
        with self._lock:
            if key_str in self._cache:
                entry = self._cache[key_str]
                
                # Check TTL
                if self._should_evict_entry(entry):
                    self._remove_entry(key_str)
                    self.misses += 1
                    return default
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                self._frequency_count[key_str] = entry.access_count
                
                # Update LRU order
                if key_str in self._access_order:
                    self._access_order.remove(key_str)
                self._access_order.append(key_str)
                
                self.hits += 1
                return self._decompress_value(entry.value)
            else:
                self.misses += 1
                return default
    
    def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        key_str = self._generate_key(key)
        
        with self._lock:
            # Compress value if beneficial
            compressed_value = self._compress_value(value)
            size_bytes = self._estimate_size(compressed_value)
            
            # Check if value is too large
            if size_bytes > self.max_memory:
                return False
            
            # Evict if necessary
            self._evict_entries(
                target_memory=self.max_memory - size_bytes
            )
            
            # Remove existing entry if present
            if key_str in self._cache:
                self._remove_entry(key_str)
            
            # Create new entry
            entry = CacheEntry(
                value=compressed_value,
                size_bytes=size_bytes,
                ttl=ttl or self.ttl_default
            )
            
            self._cache[key_str] = entry
            self._access_order.append(key_str)
            self._frequency_count[key_str] = 1
            self.current_memory += size_bytes
            
            # Final eviction check
            self._evict_entries()
            
            return True
    
    def invalidate(self, key: Any) -> bool:
        """Remove key from cache."""
        key_str = self._generate_key(key)
        
        with self._lock:
            if key_str in self._cache:
                self._remove_entry(key_str)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_count.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / max(total_accesses, 1)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage": self.current_memory,
                "max_memory": self.max_memory,
                "memory_usage_percent": (self.current_memory / self.max_memory) * 100,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "eviction_policy": self.eviction_policy.value,
            }


class ConcurrentExecutor:
    """
    Advanced concurrent execution system with resource management.
    """
    
    def __init__(
        self,
        pool_config: ResourcePool = None,
        enable_gpu: bool = True,
        enable_auto_scaling: bool = True
    ):
        """
        Initialize concurrent executor.
        
        Args:
            pool_config: Resource pool configuration
            enable_gpu: Whether to enable GPU processing
            enable_auto_scaling: Whether to enable automatic scaling
        """
        self.pool_config = pool_config or ResourcePool()
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.enable_auto_scaling = enable_auto_scaling
        
        # Thread pools
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.pool_config.max_cpu_workers,
            thread_name_prefix="protein_cpu"
        )
        
        if self.enable_gpu:
            self.gpu_executor = ThreadPoolExecutor(
                max_workers=self.pool_config.max_gpu_workers,
                thread_name_prefix="protein_gpu"
            )
        else:
            self.gpu_executor = None
        
        # Process pool for CPU-intensive tasks
        self.process_executor = ProcessPoolExecutor(
            max_workers=min(mp.cpu_count(), self.pool_config.max_cpu_workers)
        )
        
        # Task queue and management
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Performance tracking
        self.task_history = []
        self.resource_usage = {
            "cpu_utilization": 0.0,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
        }
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.2
        self.min_workers = 1
        self.max_workers = mp.cpu_count() * 2
        
    def submit_task(
        self,
        task_id: str,
        function: Callable,
        *args,
        device: ComputeDevice = ComputeDevice.AUTO,
        priority: int = 0,
        memory_requirement: int = 0,
        **kwargs
    ) -> concurrent.futures.Future:
        """
        Submit a task for execution.
        
        Args:
            task_id: Unique task identifier
            function: Function to execute
            *args: Function arguments
            device: Preferred compute device
            priority: Task priority (higher = more important)
            memory_requirement: Memory requirement in bytes
            **kwargs: Function keyword arguments
            
        Returns:
            Future representing the task result
        """
        task = ComputeTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            device_preference=device,
            memory_requirement=memory_requirement
        )
        
        # Choose appropriate executor
        executor = self._choose_executor(task)
        
        # Submit to executor
        future = executor.submit(self._execute_task_wrapper, task)
        
        # Track active task
        self.active_tasks[task_id] = task
        
        return future
    
    def submit_batch(
        self,
        tasks: List[Tuple[str, Callable, tuple, dict]],
        device: ComputeDevice = ComputeDevice.AUTO
    ) -> List[concurrent.futures.Future]:
        """
        Submit batch of tasks for parallel execution.
        
        Args:
            tasks: List of (task_id, function, args, kwargs) tuples
            device: Preferred compute device for all tasks
            
        Returns:
            List of futures for task results
        """
        futures = []
        
        for task_id, function, args, kwargs in tasks:
            future = self.submit_task(
                task_id, function, *args, device=device, **kwargs
            )
            futures.append(future)
        
        return futures
    
    def _choose_executor(self, task: ComputeTask) -> ThreadPoolExecutor:
        """Choose appropriate executor for task."""
        if task.device_preference == ComputeDevice.GPU and self.gpu_executor:
            return self.gpu_executor
        elif task.device_preference == ComputeDevice.CPU:
            return self.cpu_executor
        else:
            # Auto-choose based on task characteristics
            if (task.memory_requirement > 100 * 1024 * 1024 and  # > 100MB
                self.enable_gpu and self.gpu_executor):
                return self.gpu_executor
            else:
                return self.cpu_executor
    
    def _execute_task_wrapper(self, task: ComputeTask) -> Any:
        """Wrapper for task execution with monitoring."""
        start_time = time.time()
        
        try:
            # Set device context if GPU task
            if (task.device_preference in [ComputeDevice.GPU, ComputeDevice.MULTI_GPU] 
                and self.enable_gpu):
                with torch.cuda.device(0):  # Use first GPU
                    result = task.function(*task.args, **task.kwargs)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            # Record successful completion
            duration = time.time() - start_time
            self._record_task_completion(task, duration, True)
            
            return result
            
        except Exception as e:
            # Record failed completion
            duration = time.time() - start_time
            self._record_task_completion(task, duration, False)
            raise e
        finally:
            # Clean up active task tracking
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def _record_task_completion(self, task: ComputeTask, duration: float, success: bool):
        """Record task completion statistics."""
        completion_record = {
            "task_id": task.task_id,
            "duration": duration,
            "success": success,
            "device": task.device_preference.value,
            "memory_requirement": task.memory_requirement,
            "timestamp": time.time(),
        }
        
        self.task_history.append(completion_record)
        
        # Keep only recent history
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-500:]
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get information about active tasks."""
        return [
            {
                "task_id": task.task_id,
                "device": task.device_preference.value,
                "memory_requirement": task.memory_requirement,
                "priority": task.priority,
            }
            for task in self.active_tasks.values()
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.task_history:
            return {"total_tasks": 0, "avg_duration": 0.0, "success_rate": 1.0}
        
        recent_tasks = [t for t in self.task_history if time.time() - t["timestamp"] < 3600]  # Last hour
        
        if not recent_tasks:
            recent_tasks = self.task_history[-100:]  # Last 100 tasks
        
        total_tasks = len(recent_tasks)
        avg_duration = sum(t["duration"] for t in recent_tasks) / total_tasks
        success_rate = sum(1 for t in recent_tasks if t["success"]) / total_tasks
        
        # Device usage statistics
        device_stats = {}
        for task in recent_tasks:
            device = task["device"]
            if device not in device_stats:
                device_stats[device] = {"count": 0, "avg_duration": 0.0}
            device_stats[device]["count"] += 1
        
        for device in device_stats:
            device_tasks = [t for t in recent_tasks if t["device"] == device]
            device_stats[device]["avg_duration"] = sum(t["duration"] for t in device_tasks) / len(device_tasks)
        
        return {
            "total_tasks": total_tasks,
            "avg_duration": avg_duration,
            "success_rate": success_rate,
            "device_stats": device_stats,
            "active_tasks": len(self.active_tasks),
            "cpu_workers": self.cpu_executor._max_workers,
            "gpu_workers": self.gpu_executor._max_workers if self.gpu_executor else 0,
        }
    
    def scale_resources(self, cpu_workers: Optional[int] = None, gpu_workers: Optional[int] = None):
        """Scale executor resources."""
        if cpu_workers is not None and cpu_workers != self.cpu_executor._max_workers:
            # Create new CPU executor with different worker count
            old_executor = self.cpu_executor
            self.cpu_executor = ThreadPoolExecutor(
                max_workers=cpu_workers,
                thread_name_prefix="protein_cpu"
            )
            old_executor.shutdown(wait=False)
        
        if (gpu_workers is not None and self.gpu_executor and 
            gpu_workers != self.gpu_executor._max_workers):
            # Create new GPU executor with different worker count
            old_executor = self.gpu_executor
            self.gpu_executor = ThreadPoolExecutor(
                max_workers=gpu_workers,
                thread_name_prefix="protein_gpu"
            )
            old_executor.shutdown(wait=False)
    
    def shutdown(self, wait: bool = True):
        """Shutdown all executors."""
        self.cpu_executor.shutdown(wait=wait)
        if self.gpu_executor:
            self.gpu_executor.shutdown(wait=wait)
        self.process_executor.shutdown(wait=wait)


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_memory: int = 100 * 1024 * 1024,  # 100MB
        enable_gpu: bool = True,
        enable_auto_scaling: bool = True,
        pool_config: ResourcePool = None
    ):
        """
        Initialize performance optimizer.
        
        Args:
            cache_size: Maximum cache entries
            cache_memory: Maximum cache memory in bytes
            enable_gpu: Whether to enable GPU processing
            enable_auto_scaling: Whether to enable auto-scaling
            pool_config: Resource pool configuration
        """
        # Initialize cache
        self.cache = IntelligentCache(
            max_size=cache_size,
            max_memory=cache_memory,
            eviction_policy=CacheEvictionPolicy.ADAPTIVE
        )
        
        # Initialize concurrent executor
        self.executor = ConcurrentExecutor(
            pool_config=pool_config,
            enable_gpu=enable_gpu,
            enable_auto_scaling=enable_auto_scaling
        )
        
        # Performance monitoring
        self.optimization_stats = {
            "cache_saves": 0,
            "parallel_speedup": 0.0,
            "memory_efficiency": 0.0,
        }
    
    def cached_compute(
        self,
        function: Callable,
        *args,
        cache_key: Any = None,
        ttl: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with intelligent caching.
        
        Args:
            function: Function to execute
            *args: Function arguments
            cache_key: Custom cache key (auto-generated if None)
            ttl: Time-to-live for cache entry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (cached or computed)
        """
        # Generate cache key
        if cache_key is None:
            cache_key = (function.__name__, args, tuple(sorted(kwargs.items())))
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.optimization_stats["cache_saves"] += 1
            return cached_result
        
        # Compute result
        result = function(*args, **kwargs)
        
        # Cache result
        self.cache.set(cache_key, result, ttl=ttl)
        
        return result
    
    def parallel_compute(
        self,
        function: Callable,
        input_batches: List[Any],
        device: ComputeDevice = ComputeDevice.AUTO,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Execute function in parallel across input batches.
        
        Args:
            function: Function to execute
            input_batches: List of input batches
            device: Preferred compute device
            batch_size: Optional batch size for chunking
            
        Returns:
            List of results
        """
        if batch_size and len(input_batches) > batch_size:
            # Chunk inputs into batches
            chunks = [
                input_batches[i:i + batch_size]
                for i in range(0, len(input_batches), batch_size)
            ]
        else:
            chunks = [input_batches]
        
        # Submit tasks
        futures = []
        for i, chunk in enumerate(chunks):
            task_id = f"parallel_compute_{i}"
            future = self.executor.submit_task(
                task_id, function, chunk, device=device
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        
        return results
    
    def optimize_memory_usage(self, target_reduction: float = 0.2) -> Dict[str, Any]:
        """
        Optimize memory usage across all components.
        
        Args:
            target_reduction: Target memory reduction (0.0-1.0)
            
        Returns:
            Optimization report
        """
        initial_cache_memory = self.cache.current_memory
        
        # Reduce cache size
        new_cache_memory = int(self.cache.max_memory * (1.0 - target_reduction))
        self.cache.max_memory = new_cache_memory
        self.cache._evict_entries(target_memory=new_cache_memory)
        
        final_cache_memory = self.cache.current_memory
        memory_saved = initial_cache_memory - final_cache_memory
        
        return {
            "initial_cache_memory": initial_cache_memory,
            "final_cache_memory": final_cache_memory,
            "memory_saved": memory_saved,
            "reduction_achieved": memory_saved / max(initial_cache_memory, 1),
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "cache_stats": self.cache.get_stats(),
            "executor_stats": self.executor.get_performance_stats(),
            "optimization_stats": self.optimization_stats,
            "active_tasks": self.executor.get_active_tasks(),
        }
    
    def shutdown(self):
        """Shutdown optimizer components."""
        self.executor.shutdown()
        self.cache.clear()


# Decorators for easy optimization
def cached_operation(
    cache_key: Optional[Any] = None,
    ttl: Optional[float] = None,
    optimizer: Optional[PerformanceOptimizer] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache_key: Custom cache key function or value
        ttl: Time-to-live for cache entry
        optimizer: Performance optimizer instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            opt = optimizer or _get_global_optimizer()
            
            # Generate cache key
            if callable(cache_key):
                key = cache_key(*args, **kwargs)
            elif cache_key is not None:
                key = cache_key
            else:
                key = None
            
            return opt.cached_compute(func, *args, cache_key=key, ttl=ttl, **kwargs)
        
        return wrapper
    return decorator


def parallel_operation(
    batch_size: Optional[int] = None,
    device: ComputeDevice = ComputeDevice.AUTO,
    optimizer: Optional[PerformanceOptimizer] = None
):
    """
    Decorator for parallel execution of functions.
    
    Args:
        batch_size: Batch size for chunking
        device: Preferred compute device
        optimizer: Performance optimizer instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(input_batches: List[Any], **kwargs):
            opt = optimizer or _get_global_optimizer()
            return opt.parallel_compute(
                func, input_batches, device=device, batch_size=batch_size
            )
        
        return wrapper
    return decorator


# Global optimizer instance
_global_optimizer = None


def set_global_optimizer(optimizer: PerformanceOptimizer):
    """Set global performance optimizer."""
    global _global_optimizer
    _global_optimizer = optimizer


def _get_global_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def get_global_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    return _get_global_optimizer()