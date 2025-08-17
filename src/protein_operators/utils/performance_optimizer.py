"""
Performance optimization framework for protein design.

This module provides comprehensive performance optimization including
caching, parallelization, memory management, computational efficiency,
distributed processing, and auto-scaling capabilities.
"""

import time
import functools
import hashlib
import pickle
import os
import sys
import multiprocessing as mp
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
import queue
import socket
import subprocess
from collections import defaultdict, deque
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

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

try:
    import torch
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    import mock_torch as torch
    dist = None
    HAS_TORCH = False


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


@dataclass
class ResourceMetrics:
    """Resource utilization metrics for auto-scaling decisions."""
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    queue_length: int
    throughput_per_second: float
    error_rate_percent: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    target_workers: int
    current_workers: int
    confidence: float
    reasoning: str
    estimated_improvement: float
    timestamp: float


class AdaptiveAutoScaler:
    """
    Adaptive auto-scaling system for computational workloads.
    
    Monitors resource utilization and automatically adjusts worker counts
    to optimize performance while managing costs.
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 64,
        target_cpu_usage: float = 0.7,
        target_memory_usage: float = 0.8,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_seconds: float = 300,
        monitoring_window_seconds: float = 300
    ):
        """
        Initialize adaptive auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_cpu_usage: Target CPU utilization (0-1)
            target_memory_usage: Target memory utilization (0-1)
            scale_up_threshold: CPU/memory threshold for scaling up
            scale_down_threshold: CPU/memory threshold for scaling down
            cooldown_seconds: Cooldown period between scaling actions
            monitoring_window_seconds: Window for collecting metrics
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self.monitoring_window_seconds = monitoring_window_seconds
        
        # Current state
        self.current_workers = min_workers
        self.last_scaling_action = 0.0
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_history: List[ScalingDecision] = []
        
        # Performance tracking
        self.performance_history: Dict[int, List[float]] = defaultdict(list)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized auto-scaler: {min_workers}-{max_workers} workers")
    
    def record_metrics(self, metrics: ResourceMetrics) -> None:
        """Record resource metrics for scaling decisions."""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Record performance for this worker count
            self.performance_history[self.current_workers].append(
                metrics.throughput_per_second
            )
    
    def should_scale(self) -> Optional[ScalingDecision]:
        """
        Determine if scaling action is needed.
        
        Returns:
            Scaling decision or None if no action needed
        """
        with self.lock:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_scaling_action < self.cooldown_seconds:
                return None
            
            # Need sufficient metrics for decision
            if len(self.metrics_history) < 3:
                return None
            
            # Analyze recent metrics
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
            
            avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics]) / 100.0
            avg_memory = np.mean([m.memory_usage_percent for m in recent_metrics]) / 100.0
            avg_queue_length = np.mean([m.queue_length for m in recent_metrics])
            avg_throughput = np.mean([m.throughput_per_second for m in recent_metrics])
            avg_error_rate = np.mean([m.error_rate_percent for m in recent_metrics]) / 100.0
            
            # Scaling decision logic
            decision = self._make_scaling_decision(
                avg_cpu, avg_memory, avg_queue_length, avg_throughput, avg_error_rate
            )
            
            if decision and decision.action != 'maintain':
                self.last_scaling_action = current_time
                self.scaling_history.append(decision)
                
                # Update worker count
                if decision.action == 'scale_up':
                    self.current_workers = min(self.max_workers, decision.target_workers)
                elif decision.action == 'scale_down':
                    self.current_workers = max(self.min_workers, decision.target_workers)
                
                logger.info(f"Scaling decision: {decision.action} to {self.current_workers} workers - {decision.reasoning}")
            
            return decision
    
    def _make_scaling_decision(
        self,
        cpu_usage: float,
        memory_usage: float,
        queue_length: float,
        throughput: float,
        error_rate: float
    ) -> ScalingDecision:
        """Make scaling decision based on metrics."""
        
        current_time = time.time()
        reasoning_parts = []
        confidence = 0.5
        
        # Primary scaling factors
        resource_pressure = max(cpu_usage, memory_usage)
        queue_pressure = min(1.0, queue_length / 100.0)  # Normalize queue length
        
        # Scale up conditions
        scale_up_score = 0.0
        if resource_pressure > self.scale_up_threshold:
            scale_up_score += 0.4
            reasoning_parts.append(f"high resource usage ({resource_pressure:.1%})")
        
        if queue_length > 10:
            scale_up_score += 0.3
            reasoning_parts.append(f"queue backlog ({queue_length:.0f} items)")
        
        if error_rate > 0.05:  # 5% error rate
            scale_up_score += 0.2
            reasoning_parts.append(f"elevated error rate ({error_rate:.1%})")
        
        if throughput < self._get_expected_throughput():
            scale_up_score += 0.1
            reasoning_parts.append("below expected throughput")
        
        # Scale down conditions
        scale_down_score = 0.0
        if resource_pressure < self.scale_down_threshold:
            scale_down_score += 0.4
            reasoning_parts.append(f"low resource usage ({resource_pressure:.1%})")
        
        if queue_length == 0:
            scale_down_score += 0.2
            reasoning_parts.append("empty queue")
        
        if self._is_overprovisioned():
            scale_down_score += 0.3
            reasoning_parts.append("overprovisioned capacity")
        
        # Make decision
        if scale_up_score > 0.5 and self.current_workers < self.max_workers:
            # Scale up
            multiplier = min(2.0, 1.0 + scale_up_score)
            target_workers = min(
                self.max_workers,
                max(self.current_workers + 1, int(self.current_workers * multiplier))
            )
            
            confidence = min(0.9, scale_up_score)
            estimated_improvement = self._estimate_improvement('scale_up', target_workers)
            
            return ScalingDecision(
                action='scale_up',
                target_workers=target_workers,
                current_workers=self.current_workers,
                confidence=confidence,
                reasoning=f"Scale up due to: {', '.join(reasoning_parts)}",
                estimated_improvement=estimated_improvement,
                timestamp=current_time
            )
        
        elif scale_down_score > 0.5 and self.current_workers > self.min_workers:
            # Scale down
            target_workers = max(
                self.min_workers,
                self.current_workers - 1
            )
            
            confidence = min(0.9, scale_down_score)
            estimated_improvement = self._estimate_improvement('scale_down', target_workers)
            
            return ScalingDecision(
                action='scale_down',
                target_workers=target_workers,
                current_workers=self.current_workers,
                confidence=confidence,
                reasoning=f"Scale down due to: {', '.join(reasoning_parts)}",
                estimated_improvement=estimated_improvement,
                timestamp=current_time
            )
        
        else:
            # Maintain current scale
            return ScalingDecision(
                action='maintain',
                target_workers=self.current_workers,
                current_workers=self.current_workers,
                confidence=0.7,
                reasoning="Resource usage within target range",
                estimated_improvement=0.0,
                timestamp=current_time
            )
    
    def _get_expected_throughput(self) -> float:
        """Estimate expected throughput for current worker count."""
        if self.current_workers in self.performance_history:
            recent_throughputs = self.performance_history[self.current_workers][-5:]
            if recent_throughputs:
                return np.mean(recent_throughputs)
        
        # Fallback: assume linear scaling
        base_throughput = 10.0  # items per second per worker
        return base_throughput * self.current_workers
    
    def _is_overprovisioned(self) -> bool:
        """Check if current capacity is overprovisioned."""
        if len(self.metrics_history) < 10:
            return False
        
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics]) / 100.0
        avg_memory = np.mean([m.memory_usage_percent for m in recent_metrics]) / 100.0
        
        # Consider overprovisioned if consistently low utilization
        return max(avg_cpu, avg_memory) < 0.2 and self.current_workers > self.min_workers
    
    def _estimate_improvement(self, action: str, target_workers: int) -> float:
        """Estimate performance improvement from scaling action."""
        if action == 'scale_up':
            # Estimate improvement from additional workers
            improvement_factor = target_workers / self.current_workers
            # Diminishing returns
            return min(0.5, (improvement_factor - 1.0) * 0.8)
        
        elif action == 'scale_down':
            # Estimate cost savings (negative improvement for performance)
            cost_savings = (self.current_workers - target_workers) / self.current_workers
            return cost_savings * 0.3  # Positive value for cost savings
        
        return 0.0
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get current scaling recommendations and analysis."""
        with self.lock:
            recent_decisions = self.scaling_history[-10:] if self.scaling_history else []
            
            analysis = {
                'current_workers': self.current_workers,
                'recommended_range': self._get_recommended_range(),
                'recent_decisions': [asdict(d) for d in recent_decisions],
                'performance_trends': self._analyze_performance_trends(),
                'efficiency_metrics': self._calculate_efficiency_metrics()
            }
            
            return analysis
    
    def _get_recommended_range(self) -> Tuple[int, int]:
        """Get recommended worker range based on historical performance."""
        if not self.performance_history:
            return (self.min_workers, self.max_workers)
        
        # Find optimal worker count based on throughput per worker
        efficiency_scores = {}
        for worker_count, throughputs in self.performance_history.items():
            if throughputs:
                avg_throughput = np.mean(throughputs)
                efficiency = avg_throughput / worker_count  # Throughput per worker
                efficiency_scores[worker_count] = efficiency
        
        if not efficiency_scores:
            return (self.min_workers, self.max_workers)
        
        # Find sweet spot (highest efficiency)
        optimal_workers = max(efficiency_scores.items(), key=lambda x: x[1])[0]
        
        # Recommend range around optimal
        min_recommended = max(self.min_workers, optimal_workers - 2)
        max_recommended = min(self.max_workers, optimal_workers + 4)
        
        return (min_recommended, max_recommended)
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across different worker counts."""
        trends = {}
        
        for worker_count, throughputs in self.performance_history.items():
            if len(throughputs) >= 3:
                recent_avg = np.mean(throughputs[-3:])
                older_avg = np.mean(throughputs[:-3]) if len(throughputs) > 3 else recent_avg
                
                trend = 'improving' if recent_avg > older_avg * 1.05 else \
                       'declining' if recent_avg < older_avg * 0.95 else 'stable'
                
                trends[worker_count] = {
                    'trend': trend,
                    'recent_avg_throughput': recent_avg,
                    'total_measurements': len(throughputs)
                }
        
        return trends
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics for current configuration."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_percent for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_per_second for m in recent_metrics])
        
        return {
            'cpu_efficiency': avg_cpu / 100.0,
            'memory_efficiency': avg_memory / 100.0,
            'throughput_per_worker': avg_throughput / self.current_workers,
            'resource_utilization_score': min(avg_cpu, avg_memory) / 100.0
        }


class DistributedWorkloadManager:
    """
    Distributed workload manager for coordinating computation across multiple nodes.
    
    Handles task distribution, load balancing, fault tolerance, and result aggregation
    in distributed environments.
    """
    
    def __init__(
        self,
        node_id: str,
        coordinator_address: Optional[str] = None,
        worker_port: int = 0,  # 0 for auto-assignment
        heartbeat_interval: float = 30.0,
        task_timeout: float = 3600.0
    ):
        """
        Initialize distributed workload manager.
        
        Args:
            node_id: Unique identifier for this node
            coordinator_address: Address of coordinator node (None for coordinator)
            worker_port: Port for worker communication
            heartbeat_interval: Interval for heartbeat messages
            task_timeout: Timeout for individual tasks
        """
        self.node_id = node_id
        self.coordinator_address = coordinator_address
        self.worker_port = worker_port
        self.heartbeat_interval = heartbeat_interval
        self.task_timeout = task_timeout
        
        # Node state
        self.is_coordinator = coordinator_address is None
        self.is_running = False
        
        # Task management
        self.pending_tasks: queue.Queue = queue.Queue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Any] = {}
        
        # Worker nodes (coordinator only)
        self.worker_nodes: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        self.server_socket = None
        self.worker_threads: List[threading.Thread] = []
        
        # Auto-scaler integration
        self.auto_scaler = AdaptiveAutoScaler()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized {'coordinator' if self.is_coordinator else 'worker'} node: {node_id}")
    
    def start(self) -> None:
        """Start the distributed workload manager."""
        with self.lock:
            if self.is_running:
                return
            
            self.is_running = True
            
            if self.is_coordinator:
                self._start_coordinator()
            else:
                self._start_worker()
    
    def stop(self) -> None:
        """Stop the distributed workload manager."""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Stop worker threads
            for thread in self.worker_threads:
                thread.join(timeout=5.0)
            
            # Close server socket
            if self.server_socket:
                self.server_socket.close()
                
            logger.info(f"Stopped node: {self.node_id}")
    
    def _start_coordinator(self) -> None:
        """Start coordinator node."""
        # Start server for worker connections
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.worker_port))
        self.server_socket.listen(10)
        
        # Start worker management thread
        worker_thread = threading.Thread(target=self._handle_worker_connections)
        worker_thread.daemon = True
        worker_thread.start()
        self.worker_threads.append(worker_thread)
        
        # Start task distribution thread
        task_thread = threading.Thread(target=self._distribute_tasks)
        task_thread.daemon = True
        task_thread.start()
        self.worker_threads.append(task_thread)
        
        # Start auto-scaling monitoring
        scaling_thread = threading.Thread(target=self._monitor_scaling)
        scaling_thread.daemon = True
        scaling_thread.start()
        self.worker_threads.append(scaling_thread)
        
        logger.info(f"Coordinator started on port {self.server_socket.getsockname()[1]}")
    
    def _start_worker(self) -> None:
        """Start worker node."""
        # Connect to coordinator
        worker_thread = threading.Thread(target=self._worker_main_loop)
        worker_thread.daemon = True
        worker_thread.start()
        self.worker_threads.append(worker_thread)
        
        logger.info(f"Worker connected to coordinator: {self.coordinator_address}")
    
    def submit_task(
        self,
        task_id: str,
        task_function: str,
        task_args: List[Any],
        task_kwargs: Dict[str, Any],
        priority: int = 0
    ) -> None:
        """
        Submit a task for distributed execution.
        
        Args:
            task_id: Unique task identifier
            task_function: Serializable function name/path
            task_args: Task arguments
            task_kwargs: Task keyword arguments
            priority: Task priority (higher = more important)
        """
        if not self.is_coordinator:
            raise RuntimeError("Only coordinator can submit tasks")
        
        task = {
            'task_id': task_id,
            'function': task_function,
            'args': task_args,
            'kwargs': task_kwargs,
            'priority': priority,
            'submitted_at': time.time(),
            'retries': 0
        }
        
        with self.lock:
            self.pending_tasks.put((priority, task))
            logger.debug(f"Submitted task: {task_id}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        with self.lock:
            if task_id in self.completed_tasks:
                return {'status': 'completed', 'result': self.completed_tasks[task_id]}
            elif task_id in self.failed_tasks:
                return {'status': 'failed', 'error': self.failed_tasks[task_id]}
            elif task_id in self.active_tasks:
                return {'status': 'running', 'worker': self.active_tasks[task_id].get('worker_id')}
            else:
                return {'status': 'pending'}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        with self.lock:
            return {
                'node_id': self.node_id,
                'is_coordinator': self.is_coordinator,
                'worker_count': len(self.worker_nodes),
                'pending_tasks': self.pending_tasks.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'worker_nodes': list(self.worker_nodes.keys()),
                'auto_scaler_recommendation': self.auto_scaler.get_scaling_recommendations()
            }
    
    def _handle_worker_connections(self) -> None:
        """Handle incoming worker connections (coordinator only)."""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                
                # Handle worker in separate thread
                worker_handler = threading.Thread(
                    target=self._handle_worker,
                    args=(client_socket, address)
                )
                worker_handler.daemon = True
                worker_handler.start()
                
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting worker connection: {e}")
    
    def _handle_worker(self, client_socket: socket.socket, address: Tuple[str, int]) -> None:
        """Handle communication with a specific worker."""
        worker_id = f"worker_{address[0]}_{address[1]}"
        
        with self.lock:
            self.worker_nodes[worker_id] = {
                'address': address,
                'socket': client_socket,
                'last_heartbeat': time.time(),
                'active_tasks': 0,
                'completed_tasks': 0
            }
        
        logger.info(f"Worker connected: {worker_id}")
        
        try:
            while self.is_running:
                # Handle worker messages
                message = self._receive_message(client_socket)
                if message:
                    self._process_worker_message(worker_id, message)
                else:
                    break
        except Exception as e:
            logger.error(f"Error communicating with worker {worker_id}: {e}")
        finally:
            with self.lock:
                if worker_id in self.worker_nodes:
                    del self.worker_nodes[worker_id]
            client_socket.close()
            logger.info(f"Worker disconnected: {worker_id}")
    
    def _distribute_tasks(self) -> None:
        """Distribute tasks to available workers (coordinator only)."""
        while self.is_running:
            try:
                # Get highest priority task
                try:
                    priority, task = self.pending_tasks.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Find available worker
                available_worker = self._find_available_worker()
                if available_worker:
                    self._assign_task_to_worker(available_worker, task)
                else:
                    # No workers available, put task back
                    self.pending_tasks.put((priority, task))
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error distributing tasks: {e}")
    
    def _find_available_worker(self) -> Optional[str]:
        """Find an available worker for task assignment."""
        with self.lock:
            # Simple round-robin selection of workers with lowest load
            if not self.worker_nodes:
                return None
            
            available_workers = [
                (worker_id, worker_info['active_tasks'])
                for worker_id, worker_info in self.worker_nodes.items()
                if time.time() - worker_info['last_heartbeat'] < 60.0  # Recent heartbeat
            ]
            
            if not available_workers:
                return None
            
            # Select worker with lowest load
            return min(available_workers, key=lambda x: x[1])[0]
    
    def _assign_task_to_worker(self, worker_id: str, task: Dict[str, Any]) -> None:
        """Assign a task to a specific worker."""
        with self.lock:
            if worker_id not in self.worker_nodes:
                return
            
            task_id = task['task_id']
            self.active_tasks[task_id] = {
                **task,
                'worker_id': worker_id,
                'assigned_at': time.time()
            }
            
            self.worker_nodes[worker_id]['active_tasks'] += 1
            
            # Send task to worker
            message = {
                'type': 'task_assignment',
                'task': task
            }
            
            self._send_message(self.worker_nodes[worker_id]['socket'], message)
            logger.debug(f"Assigned task {task_id} to worker {worker_id}")
    
    def _monitor_scaling(self) -> None:
        """Monitor resource usage and make scaling decisions."""
        while self.is_running:
            try:
                # Collect metrics
                metrics = self._collect_cluster_metrics()
                self.auto_scaler.record_metrics(metrics)
                
                # Check for scaling decision
                decision = self.auto_scaler.should_scale()
                if decision and decision.action != 'maintain':
                    self._execute_scaling_decision(decision)
                
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling monitor: {e}")
                time.sleep(30.0)
    
    def _collect_cluster_metrics(self) -> ResourceMetrics:
        """Collect cluster-wide resource metrics."""
        current_time = time.time()
        
        # Calculate cluster metrics
        with self.lock:
            total_workers = len(self.worker_nodes)
            total_active_tasks = len(self.active_tasks)
            queue_length = self.pending_tasks.qsize()
            
            # Estimate throughput (tasks completed in last minute)
            recent_completions = sum(
                1 for task_result in self.completed_tasks.values()
                if isinstance(task_result, dict) and 
                   task_result.get('completed_at', 0) > current_time - 60
            )
            
            # System resource usage (simplified)
            cpu_usage = min(100.0, (total_active_tasks / max(1, total_workers)) * 80.0)
            memory_usage = min(100.0, cpu_usage * 0.8)  # Estimate based on CPU
            gpu_usage = 0.0  # Would query actual GPU usage in practice
            
            error_rate = len(self.failed_tasks) / max(1, len(self.completed_tasks) + len(self.failed_tasks)) * 100
        
        return ResourceMetrics(
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            gpu_usage_percent=gpu_usage,
            queue_length=queue_length,
            throughput_per_second=recent_completions / 60.0,
            error_rate_percent=error_rate,
            timestamp=current_time
        )
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision."""
        logger.info(f"Executing scaling decision: {decision.action} to {decision.target_workers} workers")
        
        if decision.action == 'scale_up':
            # In a real implementation, this would trigger worker node provisioning
            # For now, we just log the decision
            logger.info(f"Would provision {decision.target_workers - decision.current_workers} additional workers")
        
        elif decision.action == 'scale_down':
            # Gracefully shut down excess workers
            with self.lock:
                workers_to_remove = decision.current_workers - decision.target_workers
                worker_ids = list(self.worker_nodes.keys())[:workers_to_remove]
                
                for worker_id in worker_ids:
                    self._send_shutdown_signal(worker_id)
    
    def _send_shutdown_signal(self, worker_id: str) -> None:
        """Send shutdown signal to a worker."""
        with self.lock:
            if worker_id in self.worker_nodes:
                message = {'type': 'shutdown'}
                self._send_message(self.worker_nodes[worker_id]['socket'], message)
                logger.info(f"Sent shutdown signal to worker: {worker_id}")
    
    def _worker_main_loop(self) -> None:
        """Main loop for worker nodes."""
        # Implementation would connect to coordinator and process tasks
        # This is a simplified placeholder
        while self.is_running:
            time.sleep(1.0)
    
    def _send_message(self, sock: socket.socket, message: Dict[str, Any]) -> None:
        """Send a message over socket."""
        try:
            serialized = json.dumps(message).encode('utf-8')
            length = len(serialized)
            sock.sendall(length.to_bytes(4, byteorder='big'))
            sock.sendall(serialized)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    def _receive_message(self, sock: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive a message from socket."""
        try:
            # Read message length
            length_bytes = sock.recv(4)
            if len(length_bytes) != 4:
                return None
            
            length = int.from_bytes(length_bytes, byteorder='big')
            
            # Read message data
            data = b''
            while len(data) < length:
                chunk = sock.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk
            
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    def _process_worker_message(self, worker_id: str, message: Dict[str, Any]) -> None:
        """Process a message from a worker."""
        message_type = message.get('type')
        
        if message_type == 'heartbeat':
            with self.lock:
                if worker_id in self.worker_nodes:
                    self.worker_nodes[worker_id]['last_heartbeat'] = time.time()
        
        elif message_type == 'task_completed':
            self._handle_task_completion(worker_id, message)
        
        elif message_type == 'task_failed':
            self._handle_task_failure(worker_id, message)


# Global distributed manager instance
_distributed_manager: Optional[DistributedWorkloadManager] = None


def get_distributed_manager() -> Optional[DistributedWorkloadManager]:
    """Get the global distributed workload manager."""
    return _distributed_manager


def initialize_distributed_manager(
    node_id: str,
    coordinator_address: Optional[str] = None,
    **kwargs
) -> DistributedWorkloadManager:
    """Initialize the global distributed workload manager."""
    global _distributed_manager
    _distributed_manager = DistributedWorkloadManager(
        node_id=node_id,
        coordinator_address=coordinator_address,
        **kwargs
    )
    return _distributed_manager