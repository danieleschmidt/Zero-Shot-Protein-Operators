"""
Performance optimization framework for protein design.

This module provides comprehensive performance optimization including
caching, parallelization, memory management, and computational efficiency.
"""

import time
import functools
import hashlib
import pickle
import os
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import json
import weakref

# Configure logging
logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class CacheStrategy(Enum):
    """Cache strategies for different use cases."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    PERSISTENT = "persistent"
    MEMORY_AWARE = "memory_aware"


class ComputeStrategy(Enum):
    """Compute strategies for parallel processing."""
    SEQUENTIAL = "sequential"
    THREAD_PARALLEL = "thread_parallel"
    PROCESS_PARALLEL = "process_parallel"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_tasks: int = 0
    optimization_level: str = "none"
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BaseCache(ABC):
    """Abstract base class for caching implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class LRUCache(BaseCache):
    """
    Least Recently Used cache implementation.
    
    Thread-safe LRU cache with configurable size limits.
    """
    
    def __init__(self, max_size: int = 128):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = value
            else:
                # Add new key
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = self.access_order.pop(0)
                    del self.cache[oldest_key]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0,
                "utilization": len(self.cache) / self.max_size
            }


class MemoryAwareCache(BaseCache):
    """
    Memory-aware cache that adapts to system memory pressure.
    
    Automatically adjusts cache size based on available memory.
    """
    
    def __init__(
        self,
        max_memory_mb: float = 256,
        memory_check_interval: float = 10.0
    ):
        """
        Initialize memory-aware cache.
        
        Args:
            max_memory_mb: Maximum memory to use for cache (MB)
            memory_check_interval: How often to check memory usage (seconds)
        """
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self.cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self.last_memory_check = 0
        self.lock = threading.RLock()
    
    def _check_memory_pressure(self) -> None:
        """Check and respond to memory pressure."""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return
        
        self.last_memory_check = current_time
        
        if HAS_PSUTIL:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > 80:  # High memory usage
                # Clear cache under memory pressure
                logger.warning("High memory usage detected, clearing cache")
                self.clear()
            elif memory.percent > 70:  # Moderate memory pressure
                # Reduce cache size
                target_size = len(self.cache) // 2
                keys_to_remove = list(self.cache.keys())[target_size:]
                for key in keys_to_remove:
                    del self.cache[key]
                logger.info(f"Reduced cache size to {len(self.cache)} items")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self._check_memory_pressure()
            
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            self._check_memory_pressure()
            
            # Estimate memory usage
            if HAS_PSUTIL:
                try:
                    value_size = len(pickle.dumps(value))
                    estimated_mb = value_size / (1024 * 1024)
                    
                    if estimated_mb > self.max_memory_mb:
                        logger.warning(f"Value too large for cache ({estimated_mb:.1f} MB)")
                        return
                except Exception:
                    # If we can't serialize, don't cache
                    return
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            stats = {
                "size": len(self.cache),
                "max_memory_mb": self.max_memory_mb,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0
            }
            
            if HAS_PSUTIL:
                memory = psutil.virtual_memory()
                stats.update({
                    "system_memory_percent": memory.percent,
                    "system_memory_available_gb": memory.available / (1024**3)
                })
            
            return stats


class PersistentCache(BaseCache):
    """
    Persistent cache that survives process restarts.
    
    Stores cache data on disk with integrity checking.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size: int = 1000,
        compression: bool = True
    ):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of items to store
            compression: Whether to compress cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.compression = compression
        
        # Index file tracks all cache entries
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        # Load existing index
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded cache index with {len(self.index)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            self.index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to create valid filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"cache_{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.index:
                self.misses += 1
                return None
            
            cache_path = self._get_cache_path(key)
            
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access time
                self.index[key]["last_accessed"] = time.time()
                self.hits += 1
                return value
                
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                # Remove corrupted entry
                if key in self.index:
                    del self.index[key]
                if cache_path.exists():
                    cache_path.unlink()
                
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            # Check cache size limit
            if key not in self.index and len(self.index) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(
                    self.index.keys(),
                    key=lambda k: self.index[k]["last_accessed"]
                )
                self._remove_entry(oldest_key)
            
            cache_path = self._get_cache_path(key)
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update index
                self.index[key] = {
                    "created": time.time(),
                    "last_accessed": time.time(),
                    "path": str(cache_path),
                    "ttl": ttl
                }
                
                # Periodically save index
                if len(self.index) % 10 == 0:
                    self._save_index()
                
            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key in self.index:
            cache_path = Path(self.index[key]["path"])
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_path}: {e}")
            
            del self.index[key]
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            # Remove all cache files
            for key in list(self.index.keys()):
                self._remove_entry(key)
            
            self.index.clear()
            self.hits = 0
            self.misses = 0
            self._save_index()
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.index.items():
                if entry.get("ttl") and (current_time - entry["created"]) > entry["ttl"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self._save_index()
                logger.info(f"Removed {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            
            # Calculate disk usage
            disk_usage = 0
            for cache_file in self.cache_dir.glob("cache_*.pkl"):
                try:
                    disk_usage += cache_file.stat().st_size
                except Exception:
                    pass
            
            return {
                "size": len(self.index),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0,
                "disk_usage_mb": disk_usage / (1024 * 1024),
                "cache_dir": str(self.cache_dir)
            }


class CacheManager:
    """
    Unified cache manager supporting multiple caching strategies.
    
    Provides a consistent interface across different cache implementations
    with automatic strategy selection based on usage patterns.
    """
    
    def __init__(self, default_strategy: CacheStrategy = CacheStrategy.LRU):
        """
        Initialize cache manager.
        
        Args:
            default_strategy: Default caching strategy
        """
        self.default_strategy = default_strategy
        self.caches: Dict[str, BaseCache] = {}
        self.global_stats = {
            "total_operations": 0,
            "total_hits": 0,
            "total_misses": 0
        }
    
    def get_cache(
        self,
        name: str,
        strategy: Optional[CacheStrategy] = None,
        **kwargs
    ) -> BaseCache:
        """
        Get or create a cache instance.
        
        Args:
            name: Cache name
            strategy: Caching strategy to use
            **kwargs: Cache-specific parameters
            
        Returns:
            Cache instance
        """
        if name not in self.caches:
            strategy = strategy or self.default_strategy
            
            if strategy == CacheStrategy.LRU:
                max_size = kwargs.get('max_size', 128)
                self.caches[name] = LRUCache(max_size)
                
            elif strategy == CacheStrategy.MEMORY_AWARE:
                max_memory_mb = kwargs.get('max_memory_mb', 256)
                self.caches[name] = MemoryAwareCache(max_memory_mb)
                
            elif strategy == CacheStrategy.PERSISTENT:
                cache_dir = kwargs.get('cache_dir', f'/tmp/protein_cache_{name}')
                max_size = kwargs.get('max_size', 1000)
                self.caches[name] = PersistentCache(cache_dir, max_size)
                
            else:
                # Default to LRU
                self.caches[name] = LRUCache()
            
            logger.info(f"Created {strategy.value} cache: {name}")
        
        return self.caches[name]
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        total_hits = 0
        total_misses = 0
        cache_stats = {}
        
        for name, cache in self.caches.items():
            stats = cache.get_stats()
            cache_stats[name] = stats
            total_hits += stats.get('hits', 0)
            total_misses += stats.get('misses', 0)
        
        total_operations = total_hits + total_misses
        
        return {
            "total_operations": total_operations,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "global_hit_rate": total_hits / total_operations if total_operations > 0 else 0.0,
            "cache_count": len(self.caches),
            "cache_stats": cache_stats
        }
    
    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("Cleared all caches")


class ParallelProcessor:
    """
    Parallel processing manager for compute-intensive operations.
    
    Provides intelligent parallelization with automatic strategy selection
    based on task characteristics and system resources.
    """
    
    def __init__(
        self,
        default_strategy: ComputeStrategy = ComputeStrategy.ADAPTIVE,
        max_workers: Optional[int] = None
    ):
        """
        Initialize parallel processor.
        
        Args:
            default_strategy: Default compute strategy
            max_workers: Maximum number of worker processes/threads
        """
        self.default_strategy = default_strategy
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
        # System resource info
        self.cpu_count = os.cpu_count() or 1
        self.system_info = self._get_system_info()
        
        logger.info(f"Initialized parallel processor with {self.max_workers} max workers")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information."""
        info = {
            "cpu_count": self.cpu_count,
            "max_workers": self.max_workers
        }
        
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            info.update({
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "memory_percent": memory.percent
            })
        
        return info
    
    def process_batch(
        self,
        func: Callable,
        items: List[Any],
        strategy: Optional[ComputeStrategy] = None,
        chunk_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process a batch of items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            strategy: Compute strategy to use
            chunk_size: Size of chunks for batch processing
            progress_callback: Optional progress callback
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        strategy = strategy or self.default_strategy
        
        # Adaptive strategy selection
        if strategy == ComputeStrategy.ADAPTIVE:
            strategy = self._select_optimal_strategy(len(items), func)
        
        start_time = time.time()
        
        if strategy == ComputeStrategy.SEQUENTIAL:
            results = self._process_sequential(func, items, progress_callback)
        elif strategy == ComputeStrategy.THREAD_PARALLEL:
            results = self._process_thread_parallel(func, items, progress_callback)
        elif strategy == ComputeStrategy.PROCESS_PARALLEL:
            results = self._process_process_parallel(func, items, chunk_size, progress_callback)
        else:
            # Fallback to sequential
            results = self._process_sequential(func, items, progress_callback)
        
        execution_time = time.time() - start_time
        
        logger.info(
            f"Processed {len(items)} items using {strategy.value} "
            f"in {execution_time:.2f}s ({len(items)/execution_time:.1f} items/s)"
        )
        
        return results
    
    def _select_optimal_strategy(
        self,
        num_items: int,
        func: Callable
    ) -> ComputeStrategy:
        """Select optimal compute strategy based on task characteristics."""
        
        # Small batches - use sequential
        if num_items <= 10:
            return ComputeStrategy.SEQUENTIAL
        
        # Check if function is likely CPU-bound or I/O-bound
        # This is a heuristic - in practice you'd profile the function
        
        # For CPU-bound tasks with many items, use process parallelism
        if num_items > 100:
            return ComputeStrategy.PROCESS_PARALLEL
        
        # Medium batches - use thread parallelism
        return ComputeStrategy.THREAD_PARALLEL
    
    def _process_sequential(
        self,
        func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items sequentially."""
        results = []
        for i, item in enumerate(items):
            result = func(item)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(items))
        
        return results
    
    def _process_thread_parallel(
        self,
        func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items using thread parallelism."""
        results = [None] * len(items)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Task {index} failed: {e}")
                    results[index] = None
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))
        
        return results
    
    def _process_process_parallel(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items using process parallelism."""
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))
        
        # Split items into chunks
        chunks = [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        
        results = []
        completed_chunks = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, func, chunk): chunk
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    # Add None for failed chunk
                    chunk_size_actual = len(future_to_chunk[future])
                    results.extend([None] * chunk_size_actual)
                
                completed_chunks += 1
                if progress_callback:
                    # Estimate progress based on completed chunks
                    progress = (completed_chunks * len(items)) // len(chunks)
                    progress_callback(min(progress, len(items)), len(items))
        
        return results
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items (used by process parallelism)."""
        return [func(item) for item in chunk]


class PerformanceProfiler:
    """
    Performance profiler for tracking and optimizing system performance.
    
    Monitors execution times, memory usage, and resource utilization.
    """
    
    def __init__(self):
        """Initialize performance profiler."""
        self.profiles: Dict[str, List[PerformanceMetrics]] = {}
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def start_profile(self, operation_name: str) -> str:
        """
        Start profiling an operation.
        
        Args:
            operation_name: Name of the operation to profile
            
        Returns:
            Profile ID for tracking
        """
        profile_id = f"{operation_name}_{time.time()}"
        
        with self.lock:
            self.active_profiles[profile_id] = {
                "operation_name": operation_name,
                "start_time": time.time(),
                "start_memory": self._get_memory_usage()
            }
        
        return profile_id
    
    def end_profile(
        self,
        profile_id: str,
        cache_hits: int = 0,
        cache_misses: int = 0,
        parallel_tasks: int = 0
    ) -> PerformanceMetrics:
        """
        End profiling and record metrics.
        
        Args:
            profile_id: Profile ID from start_profile
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            parallel_tasks: Number of parallel tasks executed
            
        Returns:
            Performance metrics
        """
        with self.lock:
            if profile_id not in self.active_profiles:
                raise ValueError(f"Unknown profile ID: {profile_id}")
            
            profile_data = self.active_profiles[profile_id]
            operation_name = profile_data["operation_name"]
            
            # Calculate metrics
            execution_time = time.time() - profile_data["start_time"]
            current_memory = self._get_memory_usage()
            memory_usage = current_memory - profile_data["start_memory"]
            cpu_usage = self._get_cpu_usage()
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                parallel_tasks=parallel_tasks
            )
            
            # Store metrics
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            self.profiles[operation_name].append(metrics)
            
            # Clean up
            del self.active_profiles[profile_id]
            
            return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if HAS_PSUTIL:
            return psutil.cpu_percent(interval=0.1)
        return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific operation."""
        with self.lock:
            if operation_name not in self.profiles:
                return None
            
            metrics_list = self.profiles[operation_name]
            if not metrics_list:
                return None
            
            # Calculate statistics
            execution_times = [m.execution_time for m in metrics_list]
            memory_usages = [m.memory_usage_mb for m in metrics_list]
            cache_hit_rates = [m.cache_hit_rate for m in metrics_list]
            
            return {
                "operation_name": operation_name,
                "total_calls": len(metrics_list),
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
                "max_memory_usage_mb": max(memory_usages),
                "avg_cache_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates),
                "total_parallel_tasks": sum(m.parallel_tasks for m in metrics_list)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations."""
        with self.lock:
            return {
                operation_name: self.get_operation_stats(operation_name)
                for operation_name in self.profiles.keys()
            }
    
    def clear_profiles(self) -> None:
        """Clear all stored profiles."""
        with self.lock:
            self.profiles.clear()
            logger.info("Cleared all performance profiles")


# Global instances
_cache_manager = CacheManager()
_parallel_processor = ParallelProcessor()
_performance_profiler = PerformanceProfiler()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager


def get_parallel_processor() -> ParallelProcessor:
    """Get the global parallel processor instance."""
    return _parallel_processor


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    return _performance_profiler


# Decorator functions for easy integration
def cached(
    cache_name: str,
    strategy: CacheStrategy = CacheStrategy.LRU,
    ttl: Optional[float] = None,
    **cache_kwargs
):
    """
    Decorator to add caching to functions.
    
    Args:
        cache_name: Name of the cache to use
        strategy: Caching strategy
        ttl: Time-to-live for cached values
        **cache_kwargs: Additional cache configuration
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        cache = _cache_manager.get_cache(cache_name, strategy, **cache_kwargs)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": sorted(kwargs.items())
            }
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def parallel_batch(
    strategy: ComputeStrategy = ComputeStrategy.ADAPTIVE,
    chunk_size: Optional[int] = None
):
    """
    Decorator to add parallel processing to batch functions.
    
    Args:
        strategy: Compute strategy to use
        chunk_size: Chunk size for batch processing
        
    Returns:
        Decorated function with parallel processing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            # Create a partial function with additional arguments
            if args or kwargs:
                process_func = lambda item: func(item, *args, **kwargs)
            else:
                process_func = func
            
            # Process in parallel
            return _parallel_processor.process_batch(
                process_func,
                items,
                strategy=strategy,
                chunk_size=chunk_size
            )
        
        return wrapper
    return decorator


def profiled(operation_name: Optional[str] = None):
    """
    Decorator to add performance profiling to functions.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        
    Returns:
        Decorated function with profiling
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profile_id = _performance_profiler.start_profile(op_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = _performance_profiler.end_profile(profile_id)
                logger.debug(f"Profile {op_name}: {metrics.execution_time:.3f}s")
        
        return wrapper
    return decorator