"""Distributed processing utilities for scaling protein operations."""

import multiprocessing as mp
import threading
import queue
import time
import os
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import hashlib
from pathlib import Path

@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""
    enable_multiprocessing: bool = True
    max_processes: int = mp.cpu_count()
    max_threads_per_process: int = 2
    chunk_size: int = 100
    enable_shared_memory: bool = False
    result_caching: bool = True
    load_balancing: str = "round_robin"  # round_robin, work_stealing
    
class WorkerStats:
    """Statistics for worker processes/threads."""
    
    def __init__(self):
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        self.memory_usage_mb = 0.0
        self.start_time = time.time()
    
    def record_task(self, processing_time: float, success: bool = True):
        """Record a completed task."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_processing_time += processing_time
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.avg_processing_time = self.total_processing_time / total_tasks
    
    def get_throughput(self) -> float:
        """Get tasks per second."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.tasks_completed / elapsed
        return 0.0

class TaskResult:
    """Result of a distributed task."""
    
    def __init__(self, task_id: str, result: Any, success: bool = True, error: str = None):
        self.task_id = task_id
        self.result = result
        self.success = success
        self.error = error
        self.timestamp = time.time()

class DistributedTaskManager:
    """Manages distributed task execution."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()
        self._workers = []
        self._running = False
        self._shutdown_event = threading.Event()
        
    def start_workers(self):
        """Start worker processes/threads."""
        self._running = True
        
        if self.config.enable_multiprocessing:
            # Use process pool
            for i in range(self.config.max_processes):
                worker_id = f"process_{i}"
                self.worker_stats[worker_id] = WorkerStats()
        else:
            # Use thread pool
            for i in range(self.config.max_processes):  # Reuse max_processes as thread count
                worker_id = f"thread_{i}"
                self.worker_stats[worker_id] = WorkerStats()
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    daemon=True
                )
                worker.start()
                self._workers.append(worker)
    
    def _worker_loop(self, worker_id: str):
        """Main worker loop for thread-based workers."""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Process task
                start_time = time.time()
                try:
                    result = self._process_task(task)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                processing_time = time.time() - start_time
                
                # Record stats
                self.worker_stats[worker_id].record_task(processing_time, success)
                
                # Put result
                task_result = TaskResult(task['id'], result, success, error)
                self.result_queue.put(task_result)
                
                # Mark task done
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def _process_task(self, task: Dict[str, Any]) -> Any:
        """Process a single task."""
        func = task['function']
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})
        
        return func(*args, **kwargs)
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """Submit a task for processing."""
        task = {
            'id': task_id,
            'function': func,
            'args': args,
            'kwargs': kwargs
        }
        
        self.task_queue.put(task)
        return task_id
    
    def get_results(self, timeout: float = None) -> List[TaskResult]:
        """Get completed task results."""
        results = []
        end_time = time.time() + (timeout or float('inf'))
        
        while time.time() < end_time:
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
            except queue.Empty:
                if timeout is None:
                    break
                continue
        
        return results
    
    def wait_for_completion(self, timeout: float = None):
        """Wait for all tasks to complete."""
        self.task_queue.join()
    
    def shutdown(self):
        """Shutdown the task manager."""
        self._running = False
        self._shutdown_event.set()
        
        # Send shutdown signals to workers
        for _ in self._workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self._workers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        total_completed = sum(stats.tasks_completed for stats in self.worker_stats.values())
        total_failed = sum(stats.tasks_failed for stats in self.worker_stats.values())
        avg_throughput = sum(stats.get_throughput() for stats in self.worker_stats.values())
        
        return {
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "average_throughput": avg_throughput / len(self.worker_stats) if self.worker_stats else 0,
            "worker_count": len(self.worker_stats),
            "queue_size": self.task_queue.qsize(),
            "worker_stats": {
                worker_id: {
                    "completed": stats.tasks_completed,
                    "failed": stats.tasks_failed,
                    "throughput": stats.get_throughput(),
                    "avg_time": stats.avg_processing_time
                }
                for worker_id, stats in self.worker_stats.items()
            }
        }

class ProteinBatchProcessor:
    """Specialized batch processor for protein operations."""
    
    def __init__(self, config: DistributedConfig = None):
        self.config = config or DistributedConfig()
        self.task_manager = DistributedTaskManager(self.config)
        self._result_cache: Dict[str, Any] = {}
    
    def process_proteins(
        self, 
        protein_data: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process a list of protein data in parallel."""
        
        if not protein_data:
            return []
        
        # Start workers
        self.task_manager.start_workers()
        
        try:
            # Submit tasks
            task_ids = []
            for i, protein in enumerate(protein_data):
                # Check cache first
                cache_key = self._get_cache_key(protein)
                if self.config.result_caching and cache_key in self._result_cache:
                    continue
                
                task_id = f"protein_task_{i}"
                self.task_manager.submit_task(task_id, process_func, protein)
                task_ids.append((task_id, cache_key, i))
            
            # Collect results
            results = [None] * len(protein_data)
            completed_tasks = 0
            total_tasks = len(task_ids)
            
            while completed_tasks < total_tasks:
                task_results = self.task_manager.get_results(timeout=1.0)
                
                for task_result in task_results:
                    if task_result.success:
                        # Find original index
                        for task_id, cache_key, original_idx in task_ids:
                            if task_result.task_id == task_id:
                                results[original_idx] = task_result.result
                                
                                # Cache result
                                if self.config.result_caching:
                                    self._result_cache[cache_key] = task_result.result
                                
                                completed_tasks += 1
                                break
                    else:
                        print(f"Task {task_result.task_id} failed: {task_result.error}")
                        completed_tasks += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed_tasks, total_tasks)
            
            # Fill cached results
            for i, protein in enumerate(protein_data):
                if results[i] is None:
                    cache_key = self._get_cache_key(protein)
                    if cache_key in self._result_cache:
                        results[i] = self._result_cache[cache_key]
            
            return [r for r in results if r is not None]
            
        finally:
            self.task_manager.shutdown()
    
    def _get_cache_key(self, protein_data: Any) -> str:
        """Generate cache key for protein data."""
        try:
            data_str = str(protein_data)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return str(id(protein_data))
    
    def clear_cache(self):
        """Clear result cache."""
        self._result_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._result_cache),
            "cache_enabled": self.config.result_caching
        }

class ProcessPoolWrapper:
    """Wrapper for process pool with enhanced features."""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 100):
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self._executor = None
    
    def __enter__(self):
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def map_chunked(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Map function over iterable with chunking."""
        if not self._executor:
            raise RuntimeError("ProcessPoolWrapper not entered")
        
        # Split into chunks
        chunks = [iterable[i:i + self.chunk_size] for i in range(0, len(iterable), self.chunk_size)]
        
        # Submit chunk processing jobs
        futures = []
        for chunk in chunks:
            future = self._executor.submit(self._process_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
        
        return results
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]

def parallelize_protein_processing(
    protein_list: List[Any],
    process_func: Callable,
    max_workers: int = None,
    chunk_size: int = 100,
    use_processes: bool = True
) -> List[Any]:
    """High-level function to parallelize protein processing."""
    
    if not protein_list:
        return []
    
    max_workers = max_workers or (mp.cpu_count() if use_processes else 4)
    
    if use_processes and len(protein_list) > chunk_size:
        # Use process pool for large datasets
        with ProcessPoolWrapper(max_workers=max_workers, chunk_size=chunk_size) as pool:
            return pool.map_chunked(process_func, protein_list)
    else:
        # Use thread pool for smaller datasets or I/O bound tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, protein) for protein in protein_list]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Task failed: {e}")
                    results.append(None)
            
            return [r for r in results if r is not None]

# Global distributed processor
_global_processor: Optional[ProteinBatchProcessor] = None

def get_distributed_processor(config: DistributedConfig = None) -> ProteinBatchProcessor:
    """Get global distributed processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = ProteinBatchProcessor(config)
    return _global_processor

def configure_distributed_processing(**kwargs) -> DistributedConfig:
    """Configure distributed processing settings."""
    config = DistributedConfig(**kwargs)
    global _global_processor
    _global_processor = ProteinBatchProcessor(config)
    return config

def process_proteins_distributed(
    protein_data: List[Any],
    process_func: Callable,
    **config_kwargs
) -> List[Any]:
    """Process proteins using distributed processing."""
    if config_kwargs:
        config = DistributedConfig(**config_kwargs)
        processor = ProteinBatchProcessor(config)
    else:
        processor = get_distributed_processor()
    
    return processor.process_proteins(protein_data, process_func)