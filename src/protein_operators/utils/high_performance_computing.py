"""
High-Performance Computing framework for protein operators.

This module provides advanced performance optimization, caching,
distributed processing, and auto-scaling capabilities for 
production-scale protein design operations.
"""

import asyncio
import threading
import multiprocessing
import time
import hashlib
import pickle
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
from queue import Queue, PriorityQueue
import sys
import os

# Handle import compatibility
try:
    import torch
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    import mock_torch as torch
    import multiprocessing as mp
    TORCH_AVAILABLE = False

import numpy as np


@dataclass
class ComputeResource:
    """Representation of a compute resource."""
    resource_id: str
    resource_type: str  # cpu, gpu, distributed
    cores: int
    memory_gb: float
    gpu_memory_gb: Optional[float] = None
    utilization: float = 0.0
    available: bool = True
    last_used: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeTask:
    """Compute task for processing queue."""
    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 5  # 1 = highest, 10 = lowest
    estimated_duration: float = 0.0
    memory_requirement: float = 0.0
    gpu_required: bool = False
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __lt__(self, other):
        # For priority queue ordering
        return self.priority < other.priority


class IntelligentCache:
    """
    Intelligent multi-level caching system with adaptive strategies.
    
    Features:
    - Multi-level cache (memory, disk, distributed)
    - Adaptive eviction policies
    - Cache warming and prefetching
    - Performance-based cache sizing
    """
    
    def __init__(self, max_memory_mb: int = 1024, max_disk_gb: int = 10):
        self.max_memory_size = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.max_disk_size = max_disk_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Memory cache (L1)
        self.memory_cache: Dict[str, Any] = {}
        self.memory_usage = 0
        self.memory_access_times: Dict[str, float] = {}
        self.memory_access_counts: Dict[str, int] = defaultdict(int)
        
        # Disk cache (L2) - simplified for demo
        self.disk_cache_dir = "/tmp/protein_operators_cache"
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        self.disk_cache_index: Dict[str, Dict[str, Any]] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_size': 0,
            'disk_size': 0
        }
        
        # Performance tracking
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.prefetch_queue: deque = deque(maxlen=1000)
        
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent access tracking."""
        with self._lock:
            current_time = time.time()
            
            # Try memory cache first (L1)
            if key in self.memory_cache:
                self.memory_access_times[key] = current_time
                self.memory_access_counts[key] += 1
                self.stats['hits'] += 1
                self._record_access_pattern(key, current_time)
                return self.memory_cache[key]
            
            # Try disk cache (L2)
            if key in self.disk_cache_index:
                try:
                    filepath = os.path.join(self.disk_cache_dir, f"{key}.pkl")
                    with open(filepath, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Promote to memory cache
                    self._put_memory(key, value)
                    
                    self.stats['hits'] += 1
                    self._record_access_pattern(key, current_time)
                    return value
                    
                except Exception:
                    # Remove corrupted cache entry
                    del self.disk_cache_index[key]
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, expire_after: Optional[float] = None) -> None:
        """Put value in cache with intelligent placement."""
        with self._lock:
            value_size = self._estimate_size(value)
            current_time = time.time()
            
            # Determine optimal cache level
            if value_size < self.max_memory_size * 0.1:  # Small objects go to memory
                self._put_memory(key, value, expire_after)
            else:  # Large objects go to disk
                self._put_disk(key, value, expire_after)
            
            self._record_access_pattern(key, current_time)
    
    def _put_memory(self, key: str, value: Any, expire_after: Optional[float] = None):
        """Put value in memory cache."""
        value_size = self._estimate_size(value)
        current_time = time.time()
        
        # Check if we need to evict
        while (self.memory_usage + value_size > self.max_memory_size and 
               len(self.memory_cache) > 0):
            self._evict_memory()
        
        # Store in memory
        self.memory_cache[key] = value
        self.memory_usage += value_size
        self.memory_access_times[key] = current_time
        self.memory_access_counts[key] += 1
        
        self.stats['memory_size'] = self.memory_usage
    
    def _put_disk(self, key: str, value: Any, expire_after: Optional[float] = None):
        """Put value in disk cache."""
        try:
            filepath = os.path.join(self.disk_cache_dir, f"{key}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)
            
            file_size = os.path.getsize(filepath)
            
            # Update index
            self.disk_cache_index[key] = {
                'filepath': filepath,
                'size': file_size,
                'created_at': time.time(),
                'expire_after': expire_after
            }
            
            # Check disk space
            total_disk_size = sum(info['size'] for info in self.disk_cache_index.values())
            while total_disk_size > self.max_disk_size and len(self.disk_cache_index) > 0:
                self._evict_disk()
                total_disk_size = sum(info['size'] for info in self.disk_cache_index.values())
            
            self.stats['disk_size'] = total_disk_size
            
        except Exception as e:
            logging.warning(f"Failed to cache to disk: {e}")
    
    def _evict_memory(self):
        """Evict least valuable item from memory cache."""
        if not self.memory_cache:
            return
        
        # LFU + LRU hybrid eviction
        current_time = time.time()
        scores = {}
        
        for key in self.memory_cache:
            frequency = self.memory_access_counts[key]
            recency = current_time - self.memory_access_times.get(key, 0)
            
            # Higher score = more valuable (keep longer)
            scores[key] = frequency / (1 + recency / 3600)  # Favor recent access
        
        # Evict lowest scoring item
        victim_key = min(scores.keys(), key=lambda k: scores[k])
        
        value_size = self._estimate_size(self.memory_cache[victim_key])
        del self.memory_cache[victim_key]
        del self.memory_access_times[victim_key]
        del self.memory_access_counts[victim_key]
        
        self.memory_usage -= value_size
        self.stats['evictions'] += 1
    
    def _evict_disk(self):
        """Evict least valuable item from disk cache."""
        if not self.disk_cache_index:
            return
        
        # Simple LRU eviction for disk
        oldest_key = min(
            self.disk_cache_index.keys(),
            key=lambda k: self.disk_cache_index[k]['created_at']
        )
        
        try:
            info = self.disk_cache_index[oldest_key]
            os.remove(info['filepath'])
            del self.disk_cache_index[oldest_key]
            self.stats['evictions'] += 1
        except Exception:
            pass  # File might already be gone
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default estimate
    
    def _record_access_pattern(self, key: str, timestamp: float):
        """Record access pattern for predictive caching."""
        self.access_patterns[key].append(timestamp)
        
        # Keep only recent accesses
        cutoff = timestamp - 3600  # Last hour
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
    
    def prefetch_likely_accesses(self):
        """Prefetch items likely to be accessed soon based on patterns."""
        current_time = time.time()
        
        for key, access_times in self.access_patterns.items():
            if len(access_times) < 3:
                continue
            
            # Simple pattern: if accessed regularly, prefetch
            intervals = [access_times[i] - access_times[i-1] 
                        for i in range(1, len(access_times))]
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                last_access = access_times[-1]
                
                # If we expect access soon and item not in memory
                if (current_time - last_access > avg_interval * 0.8 and 
                    key not in self.memory_cache):
                    
                    # Try to load from disk to memory
                    if key in self.disk_cache_index:
                        value = self.get(key)  # This will promote to memory
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_items': len(self.memory_cache),
                'disk_items': len(self.disk_cache_index),
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'disk_usage_gb': self.stats['disk_size'] / (1024 * 1024 * 1024)
            }


class ResourceManager:
    """
    Intelligent compute resource management and auto-scaling.
    
    Features:
    - Dynamic resource allocation
    - Load balancing across resources
    - Auto-scaling based on demand
    - Resource utilization optimization
    """
    
    def __init__(self):
        self.resources: Dict[str, ComputeResource] = {}
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Executors for different resource types
        self.cpu_executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.gpu_executor = ThreadPoolExecutor(max_workers=2)  # Assume 2 GPUs max
        
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_compute_time': 0.0,
            'avg_queue_time': 0.0,
            'avg_execution_time': 0.0
        }
        
        self._lock = threading.RLock()
        self._scheduler_thread = None
        self._running = False
        
        # Initialize resources
        self._discover_resources()
    
    def _discover_resources(self):
        """Discover available compute resources."""
        # CPU resources
        cpu_count = multiprocessing.cpu_count()
        memory_gb = 8.0  # Simplified - would query actual memory
        
        cpu_resource = ComputeResource(
            resource_id="cpu_pool",
            resource_type="cpu",
            cores=cpu_count,
            memory_gb=memory_gb
        )
        self.resources["cpu_pool"] = cpu_resource
        
        # GPU resources
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = props.total_memory / (1024**3)
                
                gpu_resource = ComputeResource(
                    resource_id=f"gpu_{i}",
                    resource_type="gpu",
                    cores=1,  # GPU cores handled differently
                    memory_gb=2.0,  # Host memory for GPU tasks
                    gpu_memory_gb=gpu_memory_gb
                )
                self.resources[f"gpu_{i}"] = gpu_resource
    
    def start_scheduler(self):
        """Start the task scheduler."""
        if not self._running:
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
    
    def submit_task(self, task: ComputeTask) -> str:
        """Submit a task for execution."""
        with self._lock:
            self.task_queue.put(task)
            self.stats['tasks_submitted'] += 1
            return task.task_id
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Get next task (blocks with timeout)
                try:
                    task = self.task_queue.get(timeout=1.0)
                except:
                    continue
                
                # Find best resource for task
                resource = self._select_resource(task)
                
                if resource:
                    self._execute_task(task, resource)
                else:
                    # No resources available, put back in queue
                    self.task_queue.put(task)
                    time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(1.0)
    
    def _select_resource(self, task: ComputeTask) -> Optional[ComputeResource]:
        """Select best resource for a task."""
        with self._lock:
            candidates = []
            
            for resource in self.resources.values():
                if not resource.available:
                    continue
                
                # Check resource requirements
                if task.gpu_required and resource.resource_type != "gpu":
                    continue
                
                if task.memory_requirement > resource.memory_gb:
                    continue
                
                # Calculate suitability score
                score = self._calculate_resource_score(task, resource)
                candidates.append((score, resource))
            
            if candidates:
                # Return best resource (highest score)
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
            
            return None
    
    def _calculate_resource_score(self, task: ComputeTask, resource: ComputeResource) -> float:
        """Calculate resource suitability score for a task."""
        score = 0.0
        
        # Prefer less utilized resources
        score += (1.0 - resource.utilization) * 10
        
        # Prefer resources that match requirements
        if task.gpu_required and resource.resource_type == "gpu":
            score += 20
        elif not task.gpu_required and resource.resource_type == "cpu":
            score += 10
        
        # Prefer resources with adequate memory
        memory_ratio = task.memory_requirement / resource.memory_gb
        if memory_ratio <= 0.5:
            score += 5
        elif memory_ratio <= 0.8:
            score += 2
        
        # Prefer recently unused resources (to balance load)
        time_since_use = time.time() - resource.last_used
        score += min(time_since_use / 60.0, 5.0)  # Max 5 points for 5+ min idle
        
        return score
    
    def _execute_task(self, task: ComputeTask, resource: ComputeResource):
        """Execute a task on a resource."""
        with self._lock:
            # Mark resource as in use
            resource.available = False
            resource.utilization = 1.0  # Simplified
            resource.last_used = time.time()
            
            # Track active task
            task.started_at = time.time()
            self.active_tasks[task.task_id] = task
        
        # Select appropriate executor
        if resource.resource_type == "gpu":
            executor = self.gpu_executor
        else:
            executor = self.cpu_executor
        
        # Submit to executor
        future = executor.submit(self._run_task, task, resource)
        future.add_done_callback(lambda f: self._task_completed(task, resource, f))
    
    def _run_task(self, task: ComputeTask, resource: ComputeResource) -> Any:
        """Run the actual task function."""
        try:
            return task.function(*task.args, **task.kwargs)
        except Exception as e:
            logging.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def _task_completed(self, task: ComputeTask, resource: ComputeResource, future):
        """Handle task completion."""
        with self._lock:
            # Update task timing
            task.completed_at = time.time()
            execution_time = task.completed_at - (task.started_at or task.completed_at)
            queue_time = (task.started_at or task.completed_at) - task.created_at
            
            # Update statistics
            if future.exception():
                self.stats['tasks_failed'] += 1
            else:
                self.stats['tasks_completed'] += 1
                self.stats['total_compute_time'] += execution_time
                
                # Update running averages
                completed_count = self.stats['tasks_completed']
                self.stats['avg_execution_time'] = (
                    (self.stats['avg_execution_time'] * (completed_count - 1) + execution_time) 
                    / completed_count
                )
                self.stats['avg_queue_time'] = (
                    (self.stats['avg_queue_time'] * (completed_count - 1) + queue_time) 
                    / completed_count
                )
            
            # Release resource
            resource.available = True
            resource.utilization = 0.0
            
            # Move to completed tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            # Execute callback if provided
            if task.callback:
                try:
                    if future.exception():
                        task.callback(None, future.exception())
                    else:
                        task.callback(future.result(), None)
                except Exception as e:
                    logging.error(f"Task callback failed: {e}")
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        with self._lock:
            return {
                resource_id: resource.utilization 
                for resource_id, resource in self.resources.items()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            return {
                **self.stats,
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'available_resources': sum(
                    1 for r in self.resources.values() if r.available
                ),
                'total_resources': len(self.resources)
            }
    
    def auto_scale(self):
        """Auto-scale resources based on current load."""
        stats = self.get_performance_stats()
        queue_size = stats['queue_size']
        active_tasks = stats['active_tasks']
        available_resources = stats['available_resources']
        
        # Simple auto-scaling logic
        if queue_size > 10 and available_resources == 0:
            # High load, consider scaling up
            self._scale_up()
        elif queue_size == 0 and active_tasks < len(self.resources) * 0.3:
            # Low load, consider scaling down  
            self._scale_down()
    
    def _scale_up(self):
        """Scale up resources (simplified implementation)."""
        # In real implementation, this would:
        # - Request additional cloud instances
        # - Add new worker processes
        # - Increase thread pool sizes
        logging.info("Auto-scaling: Would scale up resources")
    
    def _scale_down(self):
        """Scale down resources (simplified implementation)."""
        # In real implementation, this would:
        # - Terminate idle cloud instances
        # - Reduce worker processes
        # - Decrease thread pool sizes
        logging.info("Auto-scaling: Would scale down resources")


class PerformanceOptimizer:
    """
    Advanced performance optimization system.
    
    Features:
    - Adaptive batch sizing
    - Model compilation and optimization
    - Memory usage optimization
    - Performance profiling and tuning
    """
    
    def __init__(self):
        self.cache = IntelligentCache()
        self.resource_manager = ResourceManager()
        self.performance_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.memory_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.batch_sizes: Dict[str, int] = defaultdict(lambda: 1)
        
        self._lock = threading.RLock()
    
    def optimize_function(self, func: Callable, operation_name: str = None) -> Callable:
        """Optimize a function with caching and performance tracking."""
        if operation_name is None:
            operation_name = func.__name__
        
        @wraps(func)
        def optimized_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(operation_name, args, kwargs)
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute with performance tracking
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Track performance
                execution_time = time.time() - start_time
                memory_used = self._get_memory_usage() - start_memory
                
                with self._lock:
                    self.execution_times[operation_name].append(execution_time)
                    self.memory_usage[operation_name].append(memory_used)
                
                # Cache result
                self.cache.put(cache_key, result)
                
                # Update performance profile
                self._update_performance_profile(operation_name, execution_time, memory_used)
                
                return result
                
            except Exception as e:
                # Don't cache errors, but still track performance
                execution_time = time.time() - start_time
                with self._lock:
                    self.execution_times[operation_name].append(execution_time)
                raise
        
        return optimized_wrapper
    
    def optimize_batch_processing(self, func: Callable, operation_name: str = None) -> Callable:
        """Optimize function for batch processing with adaptive batch sizes."""
        if operation_name is None:
            operation_name = func.__name__
        
        @wraps(func)
        def batch_optimized_wrapper(batch_data, *args, **kwargs):
            current_batch_size = self.batch_sizes[operation_name]
            
            # Split into optimal batch sizes
            results = []
            for i in range(0, len(batch_data), current_batch_size):
                batch = batch_data[i:i + current_batch_size]
                
                start_time = time.time()
                batch_result = func(batch, *args, **kwargs)
                execution_time = time.time() - start_time
                
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                
                # Adapt batch size based on performance
                self._adapt_batch_size(operation_name, len(batch), execution_time)
            
            return results
        
        return batch_optimized_wrapper
    
    def async_optimize(self, func: Callable, operation_name: str = None) -> Callable:
        """Optimize function for asynchronous execution."""
        if operation_name is None:
            operation_name = func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(operation_name, args, kwargs)
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute asynchronously
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            # Cache result
            self.cache.put(cache_key, result)
            
            return result
        
        return async_wrapper
    
    def distributed_optimize(self, func: Callable, operation_name: str = None) -> Callable:
        """Optimize function for distributed execution."""
        if operation_name is None:
            operation_name = func.__name__
        
        @wraps(func)
        def distributed_wrapper(*args, **kwargs):
            # Create compute task
            task = ComputeTask(
                task_id=f"{operation_name}_{int(time.time() * 1000)}",
                function=func,
                args=args,
                kwargs=kwargs,
                priority=kwargs.get('priority', 5),
                estimated_duration=self._estimate_duration(operation_name),
                memory_requirement=self._estimate_memory(operation_name),
                gpu_required=kwargs.get('gpu_required', False)
            )
            
            # Submit to resource manager
            task_id = self.resource_manager.submit_task(task)
            
            # Wait for completion (simplified - real implementation would be async)
            while task.completed_at is None:
                time.sleep(0.1)
            
            return task.function(*task.args, **task.kwargs)
        
        return distributed_wrapper
    
    def _generate_cache_key(self, operation_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function call."""
        # Create a deterministic key from function signature
        key_data = {
            'operation': operation_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0  # Mock for testing
    
    def _update_performance_profile(self, operation_name: str, execution_time: float, memory_used: float):
        """Update performance profile for an operation."""
        with self._lock:
            if operation_name not in self.performance_profiles:
                self.performance_profiles[operation_name] = {
                    'avg_execution_time': 0.0,
                    'avg_memory_usage': 0.0,
                    'sample_count': 0,
                    'last_updated': time.time()
                }
            
            profile = self.performance_profiles[operation_name]
            count = profile['sample_count']
            
            # Update running averages
            profile['avg_execution_time'] = (
                (profile['avg_execution_time'] * count + execution_time) / (count + 1)
            )
            profile['avg_memory_usage'] = (
                (profile['avg_memory_usage'] * count + memory_used) / (count + 1)
            )
            profile['sample_count'] = count + 1
            profile['last_updated'] = time.time()
    
    def _adapt_batch_size(self, operation_name: str, batch_size: int, execution_time: float):
        """Adapt batch size based on performance."""
        with self._lock:
            current_batch_size = self.batch_sizes[operation_name]
            
            # Simple adaptive logic
            throughput = batch_size / execution_time  # items per second
            
            # If throughput is good and execution time reasonable, try larger batches
            if execution_time < 1.0 and throughput > 10:
                self.batch_sizes[operation_name] = min(current_batch_size * 2, 1000)
            # If taking too long, reduce batch size
            elif execution_time > 5.0:
                self.batch_sizes[operation_name] = max(current_batch_size // 2, 1)
    
    def _estimate_duration(self, operation_name: str) -> float:
        """Estimate execution duration for an operation."""
        if operation_name in self.performance_profiles:
            return self.performance_profiles[operation_name]['avg_execution_time']
        return 1.0  # Default estimate
    
    def _estimate_memory(self, operation_name: str) -> float:
        """Estimate memory requirement for an operation."""
        if operation_name in self.performance_profiles:
            return self.performance_profiles[operation_name]['avg_memory_usage'] / 1024  # GB
        return 0.1  # Default estimate
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        with self._lock:
            cache_stats = self.cache.get_stats()
            resource_stats = self.resource_manager.get_performance_stats()
            
            return {
                'cache_performance': cache_stats,
                'resource_utilization': resource_stats,
                'performance_profiles': self.performance_profiles,
                'optimization_recommendations': self._generate_recommendations()
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Cache recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("Consider increasing cache size or improving cache key generation")
        
        # Resource recommendations
        resource_stats = self.resource_manager.get_performance_stats()
        if resource_stats['queue_size'] > 10:
            recommendations.append("High task queue - consider scaling up resources")
        
        if resource_stats['avg_execution_time'] > 10.0:
            recommendations.append("Long execution times - consider optimizing algorithms or using faster resources")
        
        # Performance profile recommendations
        for op_name, profile in self.performance_profiles.items():
            if profile['avg_memory_usage'] > 1000:  # MB
                recommendations.append(f"Operation {op_name} uses high memory - consider optimization")
        
        return recommendations


# Convenience functions and decorators

def cached(cache_instance: Optional[IntelligentCache] = None):
    """Decorator for automatic caching."""
    def decorator(func):
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = IntelligentCache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {'func': func.__name__, 'args': str(args), 'kwargs': str(kwargs)}
            key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try cache
            result = cache_instance.get(key)
            if result is not None:
                return result
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache_instance.put(key, result)
            return result
        
        return wrapper
    return decorator


def performance_monitored(optimizer: Optional[PerformanceOptimizer] = None):
    """Decorator for performance monitoring."""
    def decorator(func):
        nonlocal optimizer
        if optimizer is None:
            optimizer = PerformanceOptimizer()
        
        return optimizer.optimize_function(func)
    return decorator


def distributed_task(resource_manager: Optional[ResourceManager] = None, 
                    priority: int = 5, gpu_required: bool = False):
    """Decorator for distributed task execution."""
    def decorator(func):
        nonlocal resource_manager
        if resource_manager is None:
            resource_manager = ResourceManager()
            resource_manager.start_scheduler()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            task = ComputeTask(
                task_id=f"{func.__name__}_{int(time.time() * 1000)}",
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                gpu_required=gpu_required
            )
            
            return resource_manager.submit_task(task)
        
        return wrapper
    return decorator


# Global instances
_global_optimizer = PerformanceOptimizer()
_global_cache = IntelligentCache()
_global_resource_manager = ResourceManager()


def get_global_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    return _global_optimizer


def get_global_cache() -> IntelligentCache:
    """Get global cache instance."""
    return _global_cache


def get_global_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    return _global_resource_manager


def start_global_optimization():
    """Start global optimization services."""
    _global_resource_manager.start_scheduler()


def stop_global_optimization():
    """Stop global optimization services."""
    _global_resource_manager.stop_scheduler()