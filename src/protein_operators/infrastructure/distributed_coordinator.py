"""
Distributed coordination system for large-scale protein design operations.

Features:
- Multi-node coordination
- Load balancing and resource allocation
- Fault tolerance and recovery
- Dynamic scaling
- Inter-node communication
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError:
    import mock_torch as torch
    dist = None
    mp = None

import asyncio
import time
import threading
import queue
import pickle
import uuid
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path
import socket
import hashlib
from concurrent.futures import ThreadPoolExecutor, Future

from ..utils.advanced_logger import AdvancedLogger
from ..utils.advanced_monitoring import AdvancedMonitoringSystem


class NodeRole(Enum):
    """Roles for distributed nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    BACKUP_COORDINATOR = "backup_coordinator"


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Status of network nodes."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class DistributedTask:
    """Task to be executed in distributed environment."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    estimated_duration: float = 60.0  # seconds
    required_resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class NodeInfo:
    """Information about a network node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: NodeRole
    status: NodeStatus = NodeStatus.ONLINE
    capabilities: Dict[str, Any] = field(default_factory=dict)
    current_load: float = 0.0
    max_concurrent_tasks: int = 4
    active_tasks: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0


class LoadBalancer:
    """
    Intelligent load balancing system for distributed protein design tasks.
    
    Uses multiple strategies to optimize task distribution across nodes
    based on current load, capabilities, and historical performance.
    """
    
    def __init__(self):
        self.logger = AdvancedLogger(__name__)
        self.node_performance_history = defaultdict(deque)
        self.task_execution_history = defaultdict(list)
        
    def select_node(
        self,
        task: DistributedTask,
        available_nodes: List[NodeInfo],
        strategy: str = "adaptive"
    ) -> Optional[NodeInfo]:
        """Select the best node for executing a task."""
        if not available_nodes:
            return None
        
        # Filter nodes that can handle the task
        capable_nodes = self._filter_capable_nodes(task, available_nodes)
        if not capable_nodes:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_selection(capable_nodes)
        elif strategy == "least_loaded":
            return self._least_loaded_selection(capable_nodes)
        elif strategy == "performance_based":
            return self._performance_based_selection(task, capable_nodes)
        elif strategy == "adaptive":
            return self._adaptive_selection(task, capable_nodes)
        else:
            return capable_nodes[0]  # Default: first available
    
    def _filter_capable_nodes(
        self,
        task: DistributedTask,
        nodes: List[NodeInfo]
    ) -> List[NodeInfo]:
        """Filter nodes that can handle the task requirements."""
        capable_nodes = []
        
        for node in nodes:
            # Check if node is available
            if node.status not in [NodeStatus.ONLINE, NodeStatus.BUSY]:
                continue
            
            # Check if node has capacity
            if len(node.active_tasks) >= node.max_concurrent_tasks:
                continue
            
            # Check task-specific requirements
            can_handle = True
            
            # GPU requirement
            if task.required_resources.get('gpu', False):
                if not node.capabilities.get('has_gpu', False):
                    can_handle = False
            
            # Memory requirement
            required_memory = task.required_resources.get('memory_gb', 0)
            available_memory = node.capabilities.get('memory_gb', 0)
            if required_memory > available_memory:
                can_handle = False
            
            # Task type capability
            supported_tasks = node.capabilities.get('supported_task_types', [])
            if supported_tasks and task.task_type not in supported_tasks:
                can_handle = False
            
            if can_handle:
                capable_nodes.append(node)
        
        return capable_nodes
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Simple round-robin selection."""
        # Sort by last assignment time (would need to track this)
        return min(nodes, key=lambda n: len(n.active_tasks))
    
    def _least_loaded_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with least current load."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _performance_based_selection(
        self,
        task: DistributedTask,
        nodes: List[NodeInfo]
    ) -> NodeInfo:
        """Select node based on historical performance for task type."""
        node_scores = {}
        
        for node in nodes:
            # Get historical performance for this task type
            task_history = self.task_execution_history.get(
                f"{node.node_id}:{task.task_type}", []
            )
            
            if not task_history:
                # No history, use default score based on capabilities
                score = 1.0
            else:
                # Calculate average execution time and success rate
                successful_tasks = [t for t in task_history if t['success']]
                if successful_tasks:
                    avg_duration = sum(t['duration'] for t in successful_tasks) / len(successful_tasks)
                    success_rate = len(successful_tasks) / len(task_history)
                    
                    # Lower duration and higher success rate = better score
                    score = success_rate / (avg_duration + 1e-6)
                else:
                    score = 0.1  # Low score for nodes with only failures
            
            # Adjust for current load
            load_factor = 1.0 - node.current_load
            node_scores[node.node_id] = score * load_factor
        
        # Select node with highest score
        best_node_id = max(node_scores.keys(), key=lambda k: node_scores[k])
        return next(n for n in nodes if n.node_id == best_node_id)
    
    def _adaptive_selection(
        self,
        task: DistributedTask,
        nodes: List[NodeInfo]
    ) -> NodeInfo:
        """Adaptive selection combining multiple factors."""
        node_scores = {}
        
        for node in nodes:
            score = 0.0
            
            # Factor 1: Current load (weight: 0.3)
            load_score = 1.0 - node.current_load
            score += 0.3 * load_score
            
            # Factor 2: Queue length (weight: 0.2)
            queue_score = 1.0 - (len(node.active_tasks) / node.max_concurrent_tasks)
            score += 0.2 * queue_score
            
            # Factor 3: Historical performance (weight: 0.3)
            perf_history = self.node_performance_history[node.node_id]
            if perf_history:
                avg_performance = sum(perf_history) / len(perf_history)
                perf_score = min(1.0, avg_performance)
            else:
                perf_score = 0.5  # Neutral score for new nodes
            score += 0.3 * perf_score
            
            # Factor 4: Success rate (weight: 0.2)
            total_tasks = node.total_tasks_completed + node.total_tasks_failed
            if total_tasks > 0:
                success_rate = node.total_tasks_completed / total_tasks
            else:
                success_rate = 1.0  # Optimistic for new nodes
            score += 0.2 * success_rate
            
            node_scores[node.node_id] = score
        
        # Select node with highest score
        best_node_id = max(node_scores.keys(), key=lambda k: node_scores[k])
        return next(n for n in nodes if n.node_id == best_node_id)
    
    def update_performance(
        self,
        node_id: str,
        task_type: str,
        duration: float,
        success: bool
    ):
        """Update performance history for a node."""
        # Update general performance history
        performance_score = 1.0 / (duration + 1e-6) if success else 0.0
        self.node_performance_history[node_id].append(performance_score)
        
        # Keep only recent history
        if len(self.node_performance_history[node_id]) > 100:
            self.node_performance_history[node_id].popleft()
        
        # Update task-specific history
        task_key = f"{node_id}:{task_type}"
        self.task_execution_history[task_key].append({
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only recent task history
        if len(self.task_execution_history[task_key]) > 50:
            self.task_execution_history[task_key] = self.task_execution_history[task_key][-50:]


class FaultTolerance:
    """
    Fault tolerance and recovery system for distributed operations.
    
    Handles node failures, task recovery, and system resilience.
    """
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.logger = AdvancedLogger(__name__)
        
        # Failure detection
        self.node_failure_detector = NodeFailureDetector()
        self.task_timeout_monitor = TaskTimeoutMonitor()
        
        # Recovery strategies
        self.recovery_strategies = {
            'task_failure': self._handle_task_failure,
            'node_failure': self._handle_node_failure,
            'coordinator_failure': self._handle_coordinator_failure
        }
    
    def detect_failures(self, nodes: Dict[str, NodeInfo], tasks: Dict[str, DistributedTask]):
        """Detect various types of failures in the system."""
        # Detect node failures
        failed_nodes = self.node_failure_detector.check_node_health(nodes)
        for node_id in failed_nodes:
            self._handle_node_failure(node_id, nodes, tasks)
        
        # Detect task timeouts
        timed_out_tasks = self.task_timeout_monitor.check_task_timeouts(tasks)
        for task_id in timed_out_tasks:
            self._handle_task_timeout(task_id, tasks)
    
    def _handle_task_failure(
        self,
        task_id: str,
        error: str,
        tasks: Dict[str, DistributedTask]
    ):
        """Handle individual task failure."""
        if task_id not in tasks:
            return
        
        task = tasks[task_id]
        task.error = error
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            # Retry task on different node
            self.logger.warning(f"Retrying failed task {task_id} (attempt {task.retry_count})")
            task.status = TaskStatus.PENDING
            task.assigned_node = None
        else:
            # Mark as permanently failed
            self.logger.error(f"Task {task_id} failed permanently after {task.retry_count} attempts")
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
    
    def _handle_node_failure(
        self,
        node_id: str,
        nodes: Dict[str, NodeInfo],
        tasks: Dict[str, DistributedTask]
    ):
        """Handle node failure."""
        if node_id not in nodes:
            return
        
        node = nodes[node_id]
        self.logger.error(f"Node {node_id} failed, recovering tasks")
        
        # Mark node as failed
        node.status = NodeStatus.FAILED
        
        # Reschedule all tasks assigned to failed node
        failed_tasks = [
            task for task in tasks.values()
            if task.assigned_node == node_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]
        ]
        
        for task in failed_tasks:
            self._handle_task_failure(
                task.task_id,
                f"Node {node_id} failed",
                tasks
            )
    
    def _handle_coordinator_failure(self):
        """Handle coordinator node failure (for backup coordinators)."""
        self.logger.critical("Coordinator failure detected, initiating recovery")
        # Implementation would involve coordinator election/failover
        pass
    
    def _handle_task_timeout(
        self,
        task_id: str,
        tasks: Dict[str, DistributedTask]
    ):
        """Handle task timeout."""
        if task_id not in tasks:
            return
        
        task = tasks[task_id]
        self.logger.warning(f"Task {task_id} timed out on node {task.assigned_node}")
        
        self._handle_task_failure(
            task_id,
            "Task execution timeout",
            tasks
        )


class NodeFailureDetector:
    """Detects node failures based on heartbeat and health metrics."""
    
    def __init__(self, heartbeat_timeout: float = 60.0):
        self.heartbeat_timeout = heartbeat_timeout
        self.logger = AdvancedLogger(__name__)
    
    def check_node_health(self, nodes: Dict[str, NodeInfo]) -> List[str]:
        """Check health of all nodes and return list of failed nodes."""
        current_time = time.time()
        failed_nodes = []
        
        for node_id, node in nodes.items():
            if node.status == NodeStatus.FAILED:
                continue  # Already marked as failed
            
            # Check heartbeat timeout
            time_since_heartbeat = current_time - node.last_heartbeat
            if time_since_heartbeat > self.heartbeat_timeout:
                self.logger.warning(f"Node {node_id} heartbeat timeout ({time_since_heartbeat:.1f}s)")
                failed_nodes.append(node_id)
        
        return failed_nodes


class TaskTimeoutMonitor:
    """Monitors task execution timeouts."""
    
    def __init__(self, timeout_multiplier: float = 2.0):
        self.timeout_multiplier = timeout_multiplier
        self.logger = AdvancedLogger(__name__)
    
    def check_task_timeouts(self, tasks: Dict[str, DistributedTask]) -> List[str]:
        """Check for timed out tasks."""
        current_time = time.time()
        timed_out_tasks = []
        
        for task_id, task in tasks.items():
            if task.status != TaskStatus.RUNNING or task.started_at is None:
                continue
            
            # Calculate timeout based on estimated duration
            timeout = task.estimated_duration * self.timeout_multiplier
            execution_time = current_time - task.started_at
            
            if execution_time > timeout:
                self.logger.warning(f"Task {task_id} timed out ({execution_time:.1f}s > {timeout:.1f}s)")
                timed_out_tasks.append(task_id)
        
        return timed_out_tasks


class DistributedCoordinator:
    """
    Main coordinator for distributed protein design operations.
    
    Manages task distribution, node coordination, load balancing,
    and fault tolerance across a network of computing nodes.
    """
    
    def __init__(
        self,
        coordinator_id: str,
        port: int = 8080,
        max_concurrent_tasks: int = 100,
        enable_monitoring: bool = True
    ):
        self.coordinator_id = coordinator_id
        self.port = port
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.logger = AdvancedLogger(__name__)
        
        # Core components
        self.load_balancer = LoadBalancer()
        self.fault_tolerance = FaultTolerance(self)
        
        # State management
        self.nodes = {}  # node_id -> NodeInfo
        self.tasks = {}  # task_id -> DistributedTask
        self.task_queue = queue.PriorityQueue()
        
        # Coordination
        self.coordinator_lock = threading.RLock()
        self.task_assignment_thread = None
        self.health_monitoring_thread = None
        self.running = False
        
        # Network communication
        self.communication_server = None
        self.client_connections = {}  # node_id -> connection
        
        # Monitoring
        if enable_monitoring:
            self.monitoring = AdvancedMonitoringSystem()
        else:
            self.monitoring = None
        
        # Performance tracking
        self.task_completion_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        self.logger.info(f"Distributed Coordinator {coordinator_id} initialized")
    
    async def start(self):
        """Start the distributed coordinator."""
        self.logger.info("Starting Distributed Coordinator")
        
        self.running = True
        
        # Start monitoring if enabled
        if self.monitoring:
            self.monitoring.start()
        
        # Start background threads
        self._start_background_threads()
        
        # Start communication server
        await self._start_communication_server()
        
        self.logger.info("Distributed Coordinator started successfully")
    
    async def stop(self):
        """Stop the distributed coordinator."""
        self.logger.info("Stopping Distributed Coordinator")
        
        self.running = False
        
        # Stop background threads
        if self.task_assignment_thread:
            self.task_assignment_thread.join(timeout=5.0)
        
        if self.health_monitoring_thread:
            self.health_monitoring_thread.join(timeout=5.0)
        
        # Stop monitoring
        if self.monitoring:
            self.monitoring.stop()
        
        # Close communication server
        if self.communication_server:
            self.communication_server.close()
        
        self.logger.info("Distributed Coordinator stopped")
    
    def register_node(
        self,
        node_id: str,
        hostname: str,
        ip_address: str,
        port: int,
        capabilities: Dict[str, Any]
    ) -> bool:
        """Register a new worker node."""
        with self.coordinator_lock:
            if node_id in self.nodes:
                self.logger.warning(f"Node {node_id} already registered, updating info")
            
            node_info = NodeInfo(
                node_id=node_id,
                hostname=hostname,
                ip_address=ip_address,
                port=port,
                role=NodeRole.WORKER,
                capabilities=capabilities
            )
            
            self.nodes[node_id] = node_info
            self.logger.info(f"Registered node {node_id} ({hostname}:{port})")
            
            # Record monitoring event
            if self.monitoring:
                self.monitoring.record_metric(
                    "nodes_registered",
                    len(self.nodes),
                    tags={"event": "node_registration"}
                )
        
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a worker node."""
        with self.coordinator_lock:
            if node_id not in self.nodes:
                self.logger.warning(f"Attempted to unregister unknown node {node_id}")
                return False
            
            node = self.nodes[node_id]
            
            # Reschedule any active tasks
            active_task_ids = node.active_tasks.copy()
            for task_id in active_task_ids:
                if task_id in self.tasks:
                    self.fault_tolerance._handle_task_failure(
                        task_id,
                        f"Node {node_id} unregistered",
                        self.tasks
                    )
            
            del self.nodes[node_id]
            self.logger.info(f"Unregistered node {node_id}")
        
        return True
    
    def submit_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 1,
        estimated_duration: float = 60.0,
        required_resources: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a new task for distributed execution."""
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            estimated_duration=estimated_duration,
            required_resources=required_resources or {}
        )
        
        with self.coordinator_lock:
            self.tasks[task_id] = task
            # Add to priority queue (negative priority for max-heap behavior)
            self.task_queue.put((-priority, time.time(), task_id))
        
        self.logger.info(f"Submitted task {task_id} ({task_type})")
        
        # Record monitoring event
        if self.monitoring:
            self.monitoring.record_metric(
                "tasks_submitted",
                len(self.tasks),
                tags={"task_type": task_type}
            )
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self.coordinator_lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            return {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'assigned_node': task.assigned_node,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'retry_count': task.retry_count,
                'error': task.error
            }
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task."""
        with self.coordinator_lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
            else:
                return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.coordinator_lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False  # Cannot cancel finished tasks
            
            # Mark as cancelled
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            # Remove from assigned node if applicable
            if task.assigned_node and task.assigned_node in self.nodes:
                node = self.nodes[task.assigned_node]
                if task_id in node.active_tasks:
                    node.active_tasks.remove(task_id)
            
            self.logger.info(f"Cancelled task {task_id}")
        
        return True
    
    def _start_background_threads(self):
        """Start background processing threads."""
        # Task assignment thread
        self.task_assignment_thread = threading.Thread(
            target=self._task_assignment_loop,
            daemon=True
        )
        self.task_assignment_thread.start()
        
        # Health monitoring thread
        self.health_monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.health_monitoring_thread.start()
    
    def _task_assignment_loop(self):
        """Main loop for assigning tasks to nodes."""
        while self.running:
            try:
                # Get next task from queue (with timeout)
                try:
                    priority, timestamp, task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                with self.coordinator_lock:
                    if task_id not in self.tasks:
                        continue  # Task was cancelled
                    
                    task = self.tasks[task_id]
                    
                    if task.status != TaskStatus.PENDING:
                        continue  # Task already assigned or completed
                    
                    # Get available nodes
                    available_nodes = [
                        node for node in self.nodes.values()
                        if node.status in [NodeStatus.ONLINE, NodeStatus.BUSY]
                    ]
                    
                    if not available_nodes:
                        # No nodes available, put task back in queue
                        self.task_queue.put((priority, timestamp, task_id))
                        time.sleep(1.0)
                        continue
                    
                    # Select best node for task
                    selected_node = self.load_balancer.select_node(
                        task, available_nodes, strategy="adaptive"
                    )
                    
                    if selected_node is None:
                        # No suitable node found, put task back in queue
                        self.task_queue.put((priority, timestamp, task_id))
                        time.sleep(1.0)
                        continue
                    
                    # Assign task to node
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_node = selected_node.node_id
                    
                    selected_node.active_tasks.append(task_id)
                    selected_node.current_load = len(selected_node.active_tasks) / selected_node.max_concurrent_tasks
                    
                    # Update node status
                    if len(selected_node.active_tasks) >= selected_node.max_concurrent_tasks:
                        selected_node.status = NodeStatus.BUSY
                    
                    self.logger.info(f"Assigned task {task_id} to node {selected_node.node_id}")
                    
                    # Send task to node (simplified - would use actual network communication)
                    self._send_task_to_node(task, selected_node)
            
            except Exception as e:
                self.logger.error(f"Error in task assignment loop: {e}")
                time.sleep(1.0)
    
    def _health_monitoring_loop(self):
        """Main loop for monitoring node and task health."""
        while self.running:
            try:
                with self.coordinator_lock:
                    # Check for node and task failures
                    self.fault_tolerance.detect_failures(self.nodes, self.tasks)
                    
                    # Update throughput metrics
                    self._update_throughput_metrics()
                
                time.sleep(10.0)  # Check every 10 seconds
            
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(10.0)
    
    def _send_task_to_node(self, task: DistributedTask, node: NodeInfo):
        """Send task to worker node for execution."""
        # Simplified implementation - in practice would use network communication
        # Here we just mark the task as running
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        # Simulate task completion (for demonstration)
        def simulate_task_completion():
            time.sleep(min(5.0, task.estimated_duration / 10))  # Simulate work
            
            with self.coordinator_lock:
                if task.status == TaskStatus.RUNNING:
                    # Simulate random success/failure
                    import random
                    if random.random() > 0.1:  # 90% success rate
                        task.status = TaskStatus.COMPLETED
                        task.result = {"success": True, "output": "Mock result"}
                        node.total_tasks_completed += 1
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = "Simulated failure"
                        node.total_tasks_failed += 1
                    
                    task.completed_at = time.time()
                    
                    # Remove from node's active tasks
                    if task.task_id in node.active_tasks:
                        node.active_tasks.remove(task.task_id)
                    
                    # Update node status
                    node.current_load = len(node.active_tasks) / node.max_concurrent_tasks
                    if node.status == NodeStatus.BUSY and len(node.active_tasks) < node.max_concurrent_tasks:
                        node.status = NodeStatus.ONLINE
                    
                    # Update performance metrics
                    if task.completed_at and task.started_at:
                        duration = task.completed_at - task.started_at
                        self.load_balancer.update_performance(
                            node.node_id,
                            task.task_type,
                            duration,
                            task.status == TaskStatus.COMPLETED
                        )
                        
                        self.task_completion_times.append(duration)
        
        # Start simulation in background
        threading.Thread(target=simulate_task_completion, daemon=True).start()
    
    def _update_throughput_metrics(self):
        """Update system throughput metrics."""
        current_time = time.time()
        
        # Count completed tasks in last minute
        recent_completions = [
            task for task in self.tasks.values()
            if task.completed_at and (current_time - task.completed_at) < 60
        ]
        
        throughput = len(recent_completions)
        self.throughput_history.append(throughput)
        
        if self.monitoring:
            self.monitoring.record_metric(
                "tasks_per_minute",
                throughput,
                tags={"window": "1min"}
            )
    
    async def _start_communication_server(self):
        """Start network communication server."""
        # Simplified - in practice would implement actual network server
        self.logger.info(f"Communication server started on port {self.port}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self.coordinator_lock:
            # Node statistics
            total_nodes = len(self.nodes)
            online_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE])
            busy_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.BUSY])
            failed_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
            
            # Task statistics
            total_tasks = len(self.tasks)
            pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
            running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
            completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
            failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
            
            # Performance metrics
            avg_completion_time = 0.0
            if self.task_completion_times:
                avg_completion_time = sum(self.task_completion_times) / len(self.task_completion_times)
            
            avg_throughput = 0.0
            if self.throughput_history:
                avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
            
            return {
                'coordinator_id': self.coordinator_id,
                'cluster_health': {
                    'total_nodes': total_nodes,
                    'online_nodes': online_nodes,
                    'busy_nodes': busy_nodes,
                    'failed_nodes': failed_nodes
                },
                'task_statistics': {
                    'total_tasks': total_tasks,
                    'pending_tasks': pending_tasks,
                    'running_tasks': running_tasks,
                    'completed_tasks': completed_tasks,
                    'failed_tasks': failed_tasks,
                    'queue_size': self.task_queue.qsize()
                },
                'performance': {
                    'avg_completion_time_seconds': avg_completion_time,
                    'avg_throughput_tasks_per_minute': avg_throughput,
                    'total_capacity': sum(n.max_concurrent_tasks for n in self.nodes.values()),
                    'current_utilization': sum(len(n.active_tasks) for n in self.nodes.values())
                },
                'timestamp': time.time()
            }
