"""
Auto-scaling and load balancing utilities for protein operators.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import multiprocessing
from queue import Queue, Empty
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
    import torch.multiprocessing as mp
except ImportError:
    import mock_torch as torch
    import multiprocessing as mp


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_BASED = "queue_based"
    LATENCY_BASED = "latency_based"
    HYBRID = "hybrid"


class WorkerStatus(Enum):
    """Worker process/thread status."""
    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class WorkerInfo:
    """Information about a worker."""
    worker_id: str
    process_id: Optional[int]
    status: WorkerStatus
    start_time: float
    last_activity: float
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def update_task_completion(self, task_duration: float, success: bool = True) -> None:
        """Update worker stats after task completion."""
        self.last_activity = time.time()
        
        if success:
            self.completed_tasks += 1
            # Update average task time using exponential moving average
            alpha = 0.1  # Smoothing factor
            self.avg_task_time = alpha * task_duration + (1 - alpha) * self.avg_task_time
        else:
            self.failed_tasks += 1
    
    @property
    def total_tasks(self) -> int:
        """Total number of tasks processed."""
        return self.completed_tasks + self.failed_tasks
    
    @property
    def success_rate(self) -> float:
        """Task success rate."""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_tasks / self.total_tasks
    
    @property
    def uptime(self) -> float:
        """Worker uptime in seconds."""
        return time.time() - self.start_time


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    current_workers: int
    target_workers: int
    queue_size: int
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_latency: float
    total_throughput: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


class AutoScaler:
    """
    Intelligent auto-scaling system for protein operations.
    
    Features:
    - Dynamic worker scaling based on load
    - Multiple scaling strategies
    - Resource monitoring and optimization
    - Load balancing across workers
    - Health monitoring and recovery
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = None,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        scale_up_cooldown: float = 60.0,
        scale_down_cooldown: float = 300.0,
        worker_type: str = "process"  # "process" or "thread"
    ):
        """
        Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (default: CPU count)
            scale_up_threshold: Threshold for scaling up (0-1)
            scale_down_threshold: Threshold for scaling down (0-1)
            scaling_strategy: Strategy for scaling decisions
            scale_up_cooldown: Minimum time between scale-up operations
            scale_down_cooldown: Minimum time between scale-down operations
            worker_type: Type of worker ("process" or "thread")
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_strategy = scaling_strategy
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.worker_type = worker_type
        
        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.worker_processes: Dict[str, Any] = {}
        
        # Monitoring and metrics
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        
        # Control flags
        self._running = False
        self._monitor_thread = None
        self._scaling_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> None:
        """Start the auto-scaler."""
        if self._running:
            self.logger.warning("Auto-scaler already running")
            return
        
        self._running = True
        
        # Start initial workers
        for i in range(self.min_workers):
            self._start_worker()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info(f"Auto-scaler started with {self.min_workers} workers")
    
    def stop(self) -> None:
        """Stop the auto-scaler."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all workers
        self._stop_all_workers()
        
        # Stop monitoring
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaler stopped")
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for processing.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        task = {
            'id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'submit_time': time.time()
        }
        
        self.task_queue.put(task)
        return task_id
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a completed task result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _start_worker(self) -> str:
        """Start a new worker."""
        import uuid
        worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        
        worker_info = WorkerInfo(
            worker_id=worker_id,
            process_id=None,
            status=WorkerStatus.STARTING,
            start_time=time.time(),
            last_activity=time.time()
        )
        
        if self.worker_type == "process":
            # Start worker process
            process = mp.Process(
                target=self._worker_process,
                args=(worker_id, self.task_queue, self.result_queue)
            )
            process.daemon = True
            process.start()
            worker_info.process_id = process.pid
            self.worker_processes[worker_id] = process
        else:
            # Start worker thread
            thread = threading.Thread(
                target=self._worker_thread,
                args=(worker_id, self.task_queue, self.result_queue),
                daemon=True
            )
            thread.start()
            self.worker_processes[worker_id] = thread
        
        worker_info.status = WorkerStatus.IDLE
        self.workers[worker_id] = worker_info
        
        self.logger.info(f"Started worker {worker_id}")
        return worker_id
    
    def _stop_worker(self, worker_id: str) -> None:
        """Stop a specific worker."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        worker.status = WorkerStatus.STOPPING
        
        # Terminate process/thread
        if worker_id in self.worker_processes:
            worker_process = self.worker_processes[worker_id]
            
            if self.worker_type == "process" and hasattr(worker_process, 'terminate'):
                worker_process.terminate()
                worker_process.join(timeout=5.0)
                if worker_process.is_alive():
                    worker_process.kill()
            elif hasattr(worker_process, 'join'):
                # For threads, we can't force terminate, so we rely on daemon flag
                pass
        
        # Clean up
        del self.workers[worker_id]
        if worker_id in self.worker_processes:
            del self.worker_processes[worker_id]
        
        self.logger.info(f"Stopped worker {worker_id}")
    
    def _stop_all_workers(self) -> None:
        """Stop all workers."""
        worker_ids = list(self.workers.keys())
        for worker_id in worker_ids:
            self._stop_worker(worker_id)
    
    def _worker_process(self, worker_id: str, task_queue: Queue, result_queue: Queue) -> None:
        """Worker process main loop."""
        self.logger.info(f"Worker process {worker_id} started")
        
        while self._running:
            try:
                # Get task from queue
                task = task_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Execute task
                start_time = time.time()
                try:
                    result = task['func'](*task['args'], **task['kwargs'])
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                duration = time.time() - start_time
                
                # Send result back
                result_data = {
                    'task_id': task['id'],
                    'worker_id': worker_id,
                    'result': result,
                    'success': success,
                    'error': error,
                    'duration': duration,
                    'submit_time': task['submit_time'],
                    'complete_time': time.time()
                }
                
                result_queue.put(result_data)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
        
        self.logger.info(f"Worker process {worker_id} stopped")
    
    def _worker_thread(self, worker_id: str, task_queue: Queue, result_queue: Queue) -> None:
        """Worker thread main loop (same as process but for threads)."""
        self._worker_process(worker_id, task_queue, result_queue)
    
    def _monitor_loop(self) -> None:
        """Main monitoring and scaling loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep limited history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Make scaling decisions
                self._make_scaling_decision(metrics)
                
                # Clean up failed workers
                self._cleanup_failed_workers()
                
                # Update worker metrics
                self._update_worker_metrics()
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10.0)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Worker metrics
        active_workers = [w for w in self.workers.values() 
                         if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]]
        
        avg_cpu = 0.0
        avg_memory = 0.0
        total_throughput = 0.0
        total_errors = 0
        total_tasks = 0
        
        for worker in active_workers:
            avg_cpu += worker.cpu_usage_percent
            avg_memory += worker.memory_usage_mb
            total_throughput += worker.completed_tasks
            total_errors += worker.failed_tasks
            total_tasks += worker.total_tasks
        
        if active_workers:
            avg_cpu /= len(active_workers)
            avg_memory /= len(active_workers)
        
        error_rate = (total_errors / total_tasks) if total_tasks > 0 else 0.0
        
        # Calculate average latency from recent results
        avg_latency = self._calculate_average_latency()
        
        return ScalingMetrics(
            current_workers=len(active_workers),
            target_workers=len(active_workers),  # Will be updated by scaling logic
            queue_size=self.task_queue.qsize(),
            avg_cpu_usage=avg_cpu,
            avg_memory_usage=avg_memory,
            avg_latency=avg_latency,
            total_throughput=total_throughput,
            error_rate=error_rate,
            timestamp=current_time
        )
    
    def _calculate_average_latency(self) -> float:
        """Calculate average task latency."""
        # This is a simplified version - in practice, you'd track this more precisely
        active_workers = [w for w in self.workers.values() 
                         if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]]
        
        if not active_workers:
            return 0.0
        
        avg_latency = sum(w.avg_task_time for w in active_workers) / len(active_workers)
        return avg_latency
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> None:
        """Make scaling decisions based on metrics."""
        current_time = time.time()
        
        with self._scaling_lock:
            if self.scaling_strategy == ScalingStrategy.QUEUE_BASED:
                target_workers = self._queue_based_scaling(metrics)
            elif self.scaling_strategy == ScalingStrategy.CPU_BASED:
                target_workers = self._cpu_based_scaling(metrics)
            elif self.scaling_strategy == ScalingStrategy.MEMORY_BASED:
                target_workers = self._memory_based_scaling(metrics)
            elif self.scaling_strategy == ScalingStrategy.LATENCY_BASED:
                target_workers = self._latency_based_scaling(metrics)
            else:  # HYBRID
                target_workers = self._hybrid_scaling(metrics)
            
            # Apply constraints
            target_workers = max(self.min_workers, min(self.max_workers, target_workers))
            metrics.target_workers = target_workers
            
            # Scale up
            if target_workers > metrics.current_workers:
                if current_time - self.last_scale_up >= self.scale_up_cooldown:
                    workers_to_add = target_workers - metrics.current_workers
                    for _ in range(workers_to_add):
                        self._start_worker()
                    self.last_scale_up = current_time
                    self.logger.info(f"Scaled up to {target_workers} workers")
            
            # Scale down
            elif target_workers < metrics.current_workers:
                if current_time - self.last_scale_down >= self.scale_down_cooldown:
                    workers_to_remove = metrics.current_workers - target_workers
                    # Remove least productive workers
                    workers_to_stop = self._select_workers_for_removal(workers_to_remove)
                    for worker_id in workers_to_stop:
                        self._stop_worker(worker_id)
                    self.last_scale_down = current_time
                    self.logger.info(f"Scaled down to {target_workers} workers")
    
    def _queue_based_scaling(self, metrics: ScalingMetrics) -> int:
        """Queue-based scaling strategy."""
        # Scale based on queue size
        if metrics.queue_size > metrics.current_workers * 2:
            return metrics.current_workers + 1
        elif metrics.queue_size == 0 and metrics.current_workers > self.min_workers:
            return metrics.current_workers - 1
        return metrics.current_workers
    
    def _cpu_based_scaling(self, metrics: ScalingMetrics) -> int:
        """CPU-based scaling strategy."""
        if metrics.avg_cpu_usage > self.scale_up_threshold * 100:
            return metrics.current_workers + 1
        elif metrics.avg_cpu_usage < self.scale_down_threshold * 100:
            return metrics.current_workers - 1
        return metrics.current_workers
    
    def _memory_based_scaling(self, metrics: ScalingMetrics) -> int:
        """Memory-based scaling strategy."""
        # This is simplified - in practice, you'd have memory thresholds
        if metrics.avg_memory_usage > 1000:  # 1GB per worker
            return metrics.current_workers - 1
        elif metrics.avg_memory_usage < 500:  # 500MB per worker
            return metrics.current_workers + 1
        return metrics.current_workers
    
    def _latency_based_scaling(self, metrics: ScalingMetrics) -> int:
        """Latency-based scaling strategy."""
        target_latency = 5.0  # 5 seconds target
        
        if metrics.avg_latency > target_latency * 1.5:
            return metrics.current_workers + 1
        elif metrics.avg_latency < target_latency * 0.5:
            return metrics.current_workers - 1
        return metrics.current_workers
    
    def _hybrid_scaling(self, metrics: ScalingMetrics) -> int:
        """Hybrid scaling strategy combining multiple factors."""
        # Weighted scoring system
        score = 0.0
        
        # Queue factor (30% weight)
        if metrics.queue_size > metrics.current_workers:
            score += 0.3
        elif metrics.queue_size == 0:
            score -= 0.3
        
        # CPU factor (25% weight)
        cpu_factor = (metrics.avg_cpu_usage / 100.0 - self.scale_up_threshold) / (1.0 - self.scale_up_threshold)
        score += 0.25 * cpu_factor
        
        # Latency factor (25% weight)
        target_latency = 5.0
        if metrics.avg_latency > target_latency:
            latency_factor = min(1.0, (metrics.avg_latency - target_latency) / target_latency)
            score += 0.25 * latency_factor
        
        # Error rate factor (20% weight)
        if metrics.error_rate > 0.1:  # 10% error threshold
            score += 0.2
        
        # Make scaling decision
        if score > 0.5:
            return metrics.current_workers + 1
        elif score < -0.5:
            return metrics.current_workers - 1
        else:
            return metrics.current_workers
    
    def _select_workers_for_removal(self, count: int) -> List[str]:
        """Select workers for removal based on performance."""
        # Sort workers by productivity (least productive first)
        workers = list(self.workers.values())
        workers.sort(key=lambda w: (w.success_rate, w.completed_tasks))
        
        # Select the least productive workers
        workers_to_remove = workers[:count]
        return [w.worker_id for w in workers_to_remove]
    
    def _cleanup_failed_workers(self) -> None:
        """Clean up failed or unresponsive workers."""
        current_time = time.time()
        failed_workers = []
        
        for worker_id, worker in self.workers.items():
            # Check if worker is unresponsive
            if (current_time - worker.last_activity > 300 and  # 5 minutes
                worker.status == WorkerStatus.BUSY):
                worker.status = WorkerStatus.FAILED
                failed_workers.append(worker_id)
            
            # Check if process is still alive
            if worker_id in self.worker_processes:
                process = self.worker_processes[worker_id]
                if hasattr(process, 'is_alive') and not process.is_alive():
                    worker.status = WorkerStatus.FAILED
                    failed_workers.append(worker_id)
        
        # Remove failed workers and start replacements
        for worker_id in failed_workers:
            self.logger.warning(f"Removing failed worker {worker_id}")
            self._stop_worker(worker_id)
            
            # Start replacement if needed
            if len(self.workers) < self.min_workers:
                self._start_worker()
    
    def _update_worker_metrics(self) -> None:
        """Update worker performance metrics."""
        # Process completed results to update worker stats
        while True:
            try:
                result = self.result_queue.get_nowait()
                worker_id = result['worker_id']
                
                if worker_id in self.workers:
                    worker = self.workers[worker_id]
                    worker.update_task_completion(
                        result['duration'],
                        result['success']
                    )
                    
            except Empty:
                break
            except Exception as e:
                self.logger.error(f"Error updating worker metrics: {str(e)}")
                break
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status."""
        with self._scaling_lock:
            active_workers = [w for w in self.workers.values() 
                             if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]]
            
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                "running": self._running,
                "worker_count": len(active_workers),
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "queue_size": self.task_queue.qsize(),
                "scaling_strategy": self.scaling_strategy.value,
                "latest_metrics": latest_metrics.__dict__ if latest_metrics else None,
                "workers": {
                    worker_id: {
                        "status": worker.status.value,
                        "uptime": worker.uptime,
                        "completed_tasks": worker.completed_tasks,
                        "success_rate": worker.success_rate,
                        "avg_task_time": worker.avg_task_time
                    }
                    for worker_id, worker in self.workers.items()
                }
            }


# Global auto-scaler instance
_global_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler() -> AutoScaler:
    """Get the global auto-scaler instance."""
    global _global_auto_scaler
    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler()
    return _global_auto_scaler


def configure_auto_scaling(**kwargs) -> AutoScaler:
    """Configure global auto-scaling."""
    global _global_auto_scaler
    _global_auto_scaler = AutoScaler(**kwargs)
    return _global_auto_scaler


def start_auto_scaling() -> None:
    """Start global auto-scaling."""
    scaler = get_auto_scaler()
    scaler.start()


def stop_auto_scaling() -> None:
    """Stop global auto-scaling."""
    global _global_auto_scaler
    if _global_auto_scaler:
        _global_auto_scaler.stop()


def submit_protein_task(task_func: Callable, *args, **kwargs) -> str:
    """Submit a protein processing task to the auto-scaler."""
    scaler = get_auto_scaler()
    return scaler.submit_task(task_func, *args, **kwargs)


def get_task_result(timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Get a completed task result from the auto-scaler."""
    scaler = get_auto_scaler()
    return scaler.get_result(timeout)