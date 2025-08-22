"""
Scaling Framework for Autonomous Protein Design - Generation 3
High-performance, concurrent, and scalable protein design system.
"""

from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
import sys
import os
import json
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import torch
    import torch.nn as nn
    import torch.multiprocessing as mp
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    mp = None

try:
    import numpy as np
except ImportError:
    import mock_numpy as np


@dataclass
class DesignRequest:
    """Protein design request with priority and metadata."""
    id: str
    constraints: Any
    params: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low
    timestamp: float = field(default_factory=time.time)
    client_id: str = "default"
    timeout_seconds: float = 300.0
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        return self.priority < other.priority


class ScalableProteinDesigner:
    """
    High-performance scalable protein designer.
    
    Features:
    - Concurrent processing with thread/process pools
    - Adaptive load balancing
    - Intelligent caching
    - Batch optimization
    - Resource pooling
    - Auto-scaling triggers
    - Performance optimization
    """
    
    def __init__(
        self,
        base_designer: Any,
        config: Optional[Dict] = None,
        enable_caching: bool = True,
        enable_batching: bool = True,
        enable_distributed: bool = False
    ):
        """Initialize scalable framework."""
        self.base_designer = base_designer
        self.config = config or self._default_config()
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.enable_distributed = enable_distributed
        
        # Initialize components
        self.load_balancer = LoadBalancer(self.config)
        self.cache_manager = CacheManager(self.config) if enable_caching else None
        self.batch_processor = BatchProcessor(self.config) if enable_batching else None
        self.resource_pool = ResourcePool(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.auto_scaler = AutoScaler(self.config)
        
        # Processing infrastructure
        self.request_queue = PriorityQueue()
        self.result_cache = {}
        self.active_workers = {}
        
        # Thread pools for different workloads
        max_workers = self.config["scaling"]["max_workers"]
        self.cpu_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Process pool for CPU-intensive tasks
        if mp:
            try:
                self.process_pool = ProcessPoolExecutor(max_workers=max_workers//2)
            except:
                self.process_pool = None
        else:
            self.process_pool = None
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "processed_requests": 0,
            "queued_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_efficiency": 0.0,
            "avg_processing_time": 0.0,
            "throughput_per_second": 0.0,
            "resource_utilization": 0.0
        }
        
        # Start background workers
        self._start_workers()
        
    def _default_config(self) -> Dict:
        """Default scaling configuration."""
        return {
            "scaling": {
                "max_workers": 8,
                "max_queue_size": 1000,
                "worker_timeout": 300,
                "auto_scaling_enabled": True,
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3
            },
            "caching": {
                "enabled": True,
                "max_cache_size": 10000,
                "ttl_seconds": 3600,
                "cache_strategy": "lru",
                "persistence_enabled": True
            },
            "batching": {
                "enabled": True,
                "max_batch_size": 32,
                "batch_timeout_ms": 100,
                "adaptive_batching": True
            },
            "optimization": {
                "enable_jit": True,
                "enable_mixed_precision": True,
                "gradient_checkpointing": True,
                "memory_optimization": True
            },
            "load_balancing": {
                "strategy": "round_robin",  # round_robin, least_loaded, weighted
                "health_check_interval": 30,
                "circuit_breaker_enabled": True
            }
        }
    
    def _start_workers(self) -> None:
        """Start background worker threads."""
        # Request processor worker
        self.request_processor_thread = threading.Thread(
            target=self._process_requests,
            daemon=True
        )
        self.request_processor_thread.start()
        
        # Metrics collector worker
        self.metrics_thread = threading.Thread(
            target=self._collect_metrics,
            daemon=True
        )
        self.metrics_thread.start()
        
        # Auto-scaler worker
        if self.config["scaling"]["auto_scaling_enabled"]:
            self.auto_scale_thread = threading.Thread(
                target=self._auto_scale_worker,
                daemon=True
            )
            self.auto_scale_thread.start()
    
    async def design_async(
        self,
        constraints: Any,
        priority: int = 1,
        timeout: float = 300.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Asynchronous protein design with priority queuing."""
        request_id = self._generate_request_id(constraints, kwargs)
        
        # Check cache first
        if self.cache_manager:
            cached_result = self.cache_manager.get(request_id)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return {
                    "success": True,
                    "result": cached_result,
                    "from_cache": True,
                    "request_id": request_id
                }
        
        self.metrics["cache_misses"] += 1
        
        # Create request
        request = DesignRequest(
            id=request_id,
            constraints=constraints,
            params=kwargs,
            priority=priority,
            timeout_seconds=timeout
        )
        
        # Queue request
        if self.request_queue.qsize() >= self.config["scaling"]["max_queue_size"]:
            return {
                "success": False,
                "error": "Queue full - system overloaded",
                "request_id": request_id
            }
        
        self.request_queue.put(request)
        self.metrics["queued_requests"] += 1
        
        # Wait for result (simplified for demo)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                return result
            await asyncio.sleep(0.1)
        
        return {
            "success": False,
            "error": "Request timeout",
            "request_id": request_id
        }
    
    def design_batch(
        self,
        batch_requests: List[Dict[str, Any]],
        optimize_batch: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple design requests as a batch for efficiency."""
        if not self.batch_processor:
            # Fall back to sequential processing
            return [self.design_sync(**req) for req in batch_requests]
        
        return self.batch_processor.process_batch(
            batch_requests,
            self.base_designer,
            optimize_batch
        )
    
    def design_sync(self, **kwargs) -> Dict[str, Any]:
        """Synchronous design (blocking)."""
        request_id = self._generate_request_id(kwargs.get("constraints"), kwargs)
        
        # Check cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(request_id)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return {
                    "success": True,
                    "result": cached_result,
                    "from_cache": True,
                    "request_id": request_id
                }
        
        # Execute with optimization
        start_time = time.time()
        
        try:
            # Apply performance optimizations
            optimized_kwargs = self.performance_optimizer.optimize_request(kwargs)
            
            # Execute design
            result = self.base_designer.generate(**optimized_kwargs)
            
            # Cache result
            if self.cache_manager:
                self.cache_manager.set(request_id, result)
            
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "result": result,
                "from_cache": False,
                "request_id": request_id,
                "response_time": response_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
                "response_time": time.time() - start_time
            }
    
    def _process_requests(self) -> None:
        """Background worker to process queued requests."""
        while True:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=1.0)
                
                # Check if request has timed out
                if time.time() - request.timestamp > request.timeout_seconds:
                    self.result_cache[request.id] = {
                        "success": False,
                        "error": "Request expired",
                        "request_id": request.id
                    }
                    continue
                
                # Process request
                start_time = time.time()
                
                try:
                    # Check cache again (might have been cached by another worker)
                    if self.cache_manager:
                        cached_result = self.cache_manager.get(request.id)
                        if cached_result:
                            self.result_cache[request.id] = {
                                "success": True,
                                "result": cached_result,
                                "from_cache": True,
                                "request_id": request.id
                            }
                            self.metrics["cache_hits"] += 1
                            continue
                    
                    # Apply optimizations
                    optimized_params = self.performance_optimizer.optimize_request(request.params)
                    
                    # Execute design
                    result = self.base_designer.generate(
                        constraints=request.constraints,
                        **optimized_params
                    )
                    
                    # Cache result
                    if self.cache_manager:
                        self.cache_manager.set(request.id, result)
                    
                    response_time = time.time() - start_time
                    
                    self.result_cache[request.id] = {
                        "success": True,
                        "result": result,
                        "from_cache": False,
                        "request_id": request.id,
                        "response_time": response_time
                    }
                    
                    self.metrics["processed_requests"] += 1
                    
                except Exception as e:
                    self.result_cache[request.id] = {
                        "success": False,
                        "error": str(e),
                        "request_id": request.id,
                        "response_time": time.time() - start_time
                    }
                
            except:
                # Queue timeout - continue
                pass
    
    def _collect_metrics(self) -> None:
        """Background worker to collect performance metrics."""
        last_processed = 0
        last_time = time.time()
        
        while True:
            time.sleep(10)  # Collect metrics every 10 seconds
            
            current_time = time.time()
            current_processed = self.metrics["processed_requests"]
            
            # Calculate throughput
            time_delta = current_time - last_time
            processed_delta = current_processed - last_processed
            
            if time_delta > 0:
                self.metrics["throughput_per_second"] = processed_delta / time_delta
            
            # Update resource utilization
            self.metrics["resource_utilization"] = self.resource_pool.get_utilization()
            
            # Update queue metrics
            self.metrics["queued_requests"] = self.request_queue.qsize()
            
            last_processed = current_processed
            last_time = current_time
    
    def _auto_scale_worker(self) -> None:
        """Background worker for auto-scaling decisions."""
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            try:
                scale_decision = self.auto_scaler.should_scale(self.metrics)
                
                if scale_decision["action"] == "scale_up":
                    self._scale_up(scale_decision["target_workers"])
                elif scale_decision["action"] == "scale_down":
                    self._scale_down(scale_decision["target_workers"])
                    
            except Exception as e:
                # Log error but continue
                pass
    
    def _scale_up(self, target_workers: int) -> None:
        """Scale up worker capacity."""
        current_workers = self.cpu_pool._max_workers
        if target_workers > current_workers:
            # Create new thread pool with more workers
            old_pool = self.cpu_pool
            self.cpu_pool = ThreadPoolExecutor(max_workers=target_workers)
            # In production, would gracefully shutdown old pool
    
    def _scale_down(self, target_workers: int) -> None:
        """Scale down worker capacity."""
        current_workers = self.cpu_pool._max_workers
        if target_workers < current_workers:
            # Create new thread pool with fewer workers
            old_pool = self.cpu_pool
            self.cpu_pool = ThreadPoolExecutor(max_workers=target_workers)
            # In production, would gracefully shutdown old pool
    
    def _generate_request_id(self, constraints: Any, params: Dict) -> str:
        """Generate unique request ID for caching."""
        # Create deterministic hash from constraints and parameters
        content = f"{constraints}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "metrics": self.metrics.copy(),
            "cache_stats": self.cache_manager.get_stats() if self.cache_manager else {},
            "resource_stats": self.resource_pool.get_stats(),
            "load_balancer_stats": self.load_balancer.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def optimize_for_workload(self, workload_type: str) -> None:
        """Optimize system for specific workload patterns."""
        optimizations = {
            "high_throughput": {
                "batch_size": 64,
                "cache_strategy": "aggressive",
                "worker_count": 16
            },
            "low_latency": {
                "batch_size": 1,
                "cache_strategy": "minimal",
                "worker_count": 8
            },
            "memory_constrained": {
                "batch_size": 8,
                "cache_strategy": "compact",
                "worker_count": 4
            }
        }
        
        if workload_type in optimizations:
            config = optimizations[workload_type]
            self.performance_optimizer.apply_configuration(config)


class LoadBalancer:
    """Intelligent load balancing for distributed processing."""
    
    def __init__(self, config: Dict):
        self.config = config["load_balancing"]
        self.workers = []
        self.current_worker = 0
        self.worker_stats = {}
        
    def add_worker(self, worker_id: str, capacity: float = 1.0) -> None:
        """Add a worker to the load balancer."""
        self.workers.append({
            "id": worker_id,
            "capacity": capacity,
            "active_requests": 0,
            "total_requests": 0,
            "avg_response_time": 0.0,
            "health_status": "healthy"
        })
    
    def get_next_worker(self) -> Optional[str]:
        """Get next worker using configured strategy."""
        if not self.workers:
            return None
        
        strategy = self.config["strategy"]
        
        if strategy == "round_robin":
            worker = self.workers[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.workers)
            return worker["id"]
        
        elif strategy == "least_loaded":
            available_workers = [w for w in self.workers if w["health_status"] == "healthy"]
            if not available_workers:
                return None
            
            least_loaded = min(available_workers, key=lambda w: w["active_requests"])
            return least_loaded["id"]
        
        elif strategy == "weighted":
            # Weighted round-robin based on capacity
            available_workers = [w for w in self.workers if w["health_status"] == "healthy"]
            if not available_workers:
                return None
            
            # Simple weighted selection (in production, use proper weighted round-robin)
            weights = [w["capacity"] for w in available_workers]
            total_weight = sum(weights)
            import random
            r = random.uniform(0, total_weight)
            
            current_weight = 0
            for i, worker in enumerate(available_workers):
                current_weight += weights[i]
                if r <= current_weight:
                    return worker["id"]
        
        return self.workers[0]["id"]  # Fallback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "total_workers": len(self.workers),
            "healthy_workers": len([w for w in self.workers if w["health_status"] == "healthy"]),
            "strategy": self.config["strategy"],
            "worker_stats": self.workers.copy()
        }


class CacheManager:
    """Intelligent caching system with multiple strategies."""
    
    def __init__(self, config: Dict):
        self.config = config["caching"]
        self.cache = {}
        self.access_counts = {}
        self.access_times = {}
        self.max_size = self.config["max_cache_size"]
        self.ttl = self.config["ttl_seconds"]
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.access_times[key] > self.ttl:
            self._evict(key)
            return None
        
        # Update access statistics
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.access_times[key] = time.time()
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        # Check if cache is full and evict if necessary
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_counts[key] = 1
        self.access_times[key] = time.time()
    
    def _evict(self, key: str) -> None:
        """Evict specific key."""
        if key in self.cache:
            del self.cache[key]
            del self.access_counts[key]
            del self.access_times[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        strategy = self.config["cache_strategy"]
        
        if strategy == "lru":
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            self._evict(oldest_key)
        elif strategy == "lfu":
            least_used_key = min(self.access_counts.keys(), key=self.access_counts.get)
            self._evict(least_used_key)
        else:
            # Random eviction
            import random
            key_to_evict = random.choice(list(self.cache.keys()))
            self._evict(key_to_evict)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_ratio": self._calculate_hit_ratio(),
            "strategy": self.config["cache_strategy"]
        }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            return 0.0
        
        # Simplified hit ratio calculation
        return min(1.0, len(self.cache) / max(1, total_accesses))


class BatchProcessor:
    """Intelligent batch processing for efficiency."""
    
    def __init__(self, config: Dict):
        self.config = config["batching"]
        self.pending_requests = []
        self.last_batch_time = time.time()
        
    def process_batch(
        self,
        requests: List[Dict[str, Any]],
        designer: Any,
        optimize: bool = True
    ) -> List[Dict[str, Any]]:
        """Process batch of requests efficiently."""
        if not requests:
            return []
        
        batch_size = min(len(requests), self.config["max_batch_size"])
        
        if optimize:
            # Group similar requests for better batching
            grouped_requests = self._group_similar_requests(requests)
            results = []
            
            for group in grouped_requests:
                if len(group) == 1:
                    # Single request - process normally
                    result = self._process_single(group[0], designer)
                    results.append(result)
                else:
                    # Batch process similar requests
                    batch_results = self._process_batch_group(group, designer)
                    results.extend(batch_results)
            
            return results
        else:
            # Simple sequential processing
            return [self._process_single(req, designer) for req in requests]
    
    def _group_similar_requests(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar requests for efficient batching."""
        groups = {}
        
        for req in requests:
            # Create similarity key based on parameters
            length = req.get("length", 100)
            num_samples = req.get("num_samples", 1)
            
            # Group by similar length ranges
            length_group = (length // 50) * 50  # Group by 50-residue ranges
            similarity_key = f"{length_group}_{num_samples}"
            
            if similarity_key not in groups:
                groups[similarity_key] = []
            groups[similarity_key].append(req)
        
        return list(groups.values())
    
    def _process_single(self, request: Dict[str, Any], designer: Any) -> Dict[str, Any]:
        """Process single request."""
        try:
            result = designer.generate(**request)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_batch_group(self, group: List[Dict[str, Any]], designer: Any) -> List[Dict[str, Any]]:
        """Process a group of similar requests efficiently."""
        # For demo, process sequentially
        # In production, implement true batch processing
        return [self._process_single(req, designer) for req in group]


class ResourcePool:
    """Resource pooling and management."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gpu_pool = []
        self.memory_pool = []
        self.allocated_resources = {}
        
    def allocate_resources(self, request_id: str, requirements: Dict) -> Dict[str, Any]:
        """Allocate resources for a request."""
        allocation = {
            "gpu_id": self._allocate_gpu(requirements.get("gpu_memory", 0)),
            "memory_mb": self._allocate_memory(requirements.get("memory", 100)),
            "cpu_cores": self._allocate_cpu(requirements.get("cpu_cores", 1))
        }
        
        self.allocated_resources[request_id] = allocation
        return allocation
    
    def release_resources(self, request_id: str) -> None:
        """Release resources for a request."""
        if request_id in self.allocated_resources:
            # In production, properly release GPU memory, etc.
            del self.allocated_resources[request_id]
    
    def _allocate_gpu(self, memory_mb: float) -> Optional[int]:
        """Allocate GPU resources."""
        # Simplified GPU allocation
        return 0 if memory_mb > 0 else None
    
    def _allocate_memory(self, memory_mb: float) -> float:
        """Allocate memory resources."""
        # Simplified memory allocation
        return min(memory_mb, 1024)  # Cap at 1GB
    
    def _allocate_cpu(self, cores: int) -> int:
        """Allocate CPU cores."""
        # Simplified CPU allocation
        return min(cores, 4)  # Cap at 4 cores
    
    def get_utilization(self) -> float:
        """Get current resource utilization."""
        # Simplified utilization calculation
        return len(self.allocated_resources) / max(1, 10)  # Assume max 10 concurrent requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        return {
            "active_allocations": len(self.allocated_resources),
            "utilization": self.get_utilization(),
            "available_memory": 4096,  # Mock values
            "available_gpu_memory": 8192
        }


class PerformanceOptimizer:
    """Performance optimization strategies."""
    
    def __init__(self, config: Dict):
        self.config = config["optimization"]
        self.optimization_cache = {}
        
    def optimize_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize request parameters for performance."""
        optimized = params.copy()
        
        # Memory optimization
        if self.config["memory_optimization"]:
            if "length" in optimized and optimized["length"] > 500:
                # For very long proteins, reduce precision
                optimized["precision"] = "mixed"
        
        # Batch size optimization
        if "num_samples" in optimized:
            samples = optimized["num_samples"]
            if samples > 1:
                # Optimize batch size for throughput
                optimal_batch = self._calculate_optimal_batch_size(samples)
                optimized["batch_size"] = optimal_batch
        
        return optimized
    
    def _calculate_optimal_batch_size(self, total_samples: int) -> int:
        """Calculate optimal batch size for given number of samples."""
        # Simple heuristic - in production use performance profiling
        if total_samples <= 4:
            return total_samples
        elif total_samples <= 16:
            return 4
        else:
            return 8
    
    def apply_configuration(self, config: Dict[str, Any]) -> None:
        """Apply performance configuration."""
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value


class AutoScaler:
    """Automatic scaling based on metrics."""
    
    def __init__(self, config: Dict):
        self.config = config["scaling"]
        self.scaling_history = []
        
    def should_scale(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if scaling action is needed."""
        current_utilization = metrics.get("resource_utilization", 0.0)
        queue_size = metrics.get("queued_requests", 0)
        
        scale_up_threshold = self.config["scale_up_threshold"]
        scale_down_threshold = self.config["scale_down_threshold"]
        
        if current_utilization > scale_up_threshold or queue_size > 10:
            return {
                "action": "scale_up",
                "reason": f"High utilization ({current_utilization:.1%}) or queue backlog ({queue_size})",
                "target_workers": min(16, self.config["max_workers"] + 2)
            }
        elif current_utilization < scale_down_threshold and queue_size == 0:
            return {
                "action": "scale_down", 
                "reason": f"Low utilization ({current_utilization:.1%})",
                "target_workers": max(2, self.config["max_workers"] - 2)
            }
        else:
            return {
                "action": "no_change",
                "reason": "Utilization within normal range"
            }
    
    def record_scaling_event(self, event: Dict[str, Any]) -> None:
        """Record scaling event for analysis."""
        event["timestamp"] = time.time()
        self.scaling_history.append(event)
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]