"""
Auto-scaling and load balancing system for protein design operations.

This module provides:
- Dynamic resource scaling based on load
- Load balancing across compute resources
- Predictive scaling using machine learning
- Resource optimization strategies
"""

import asyncio
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch

from .performance_optimization import PerformanceOptimizer, ResourcePool, ComputeDevice
from .monitoring_system import MonitoringSystem


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    INTELLIGENT = "intelligent"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    queue_depth: int = 0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingAction:
    """Scaling action definition."""
    action_type: str  # "scale_up", "scale_down", "rebalance"
    resource_type: str  # "cpu", "gpu", "memory"
    target_value: int
    reason: str
    confidence: float = 1.0
    estimated_impact: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoadBalancerNode:
    """Load balancer node representation."""
    node_id: str
    weight: float = 1.0
    current_connections: int = 0
    cpu_capacity: float = 1.0
    memory_capacity: float = 1.0
    gpu_capacity: float = 0.0
    health_score: float = 1.0
    last_response_time: float = 0.0
    error_count: int = 0


class PredictiveScaler:
    """
    Predictive scaling system using time series analysis.
    """
    
    def __init__(self, history_window: int = 100, prediction_horizon: int = 10):
        """
        Initialize predictive scaler.
        
        Args:
            history_window: Number of historical data points to consider
            prediction_horizon: Number of time steps to predict ahead
        """
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        
        # Historical data
        self.metrics_history: deque = deque(maxlen=history_window)
        
        # Simple trend analysis (could be replaced with ML models)
        self.trend_weights = [0.5, 0.3, 0.2]  # Recent, medium, older
        
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to historical data."""
        self.metrics_history.append(metrics)
    
    def predict_load(self) -> Dict[str, float]:
        """
        Predict future load based on historical patterns.
        
        Returns:
            Predicted metrics for next time period
        """
        if len(self.metrics_history) < 3:
            # Not enough data for prediction
            if self.metrics_history:
                latest = self.metrics_history[-1]
                return {
                    "cpu_utilization": latest.cpu_utilization,
                    "memory_utilization": latest.memory_utilization,
                    "queue_depth": float(latest.queue_depth),
                    "response_time": latest.response_time_p95,
                }
            else:
                return {
                    "cpu_utilization": 0.0,
                    "memory_utilization": 0.0,
                    "queue_depth": 0.0,
                    "response_time": 0.0,
                }
        
        # Simple weighted average trend analysis
        recent_metrics = list(self.metrics_history)[-3:]
        
        predictions = {}
        
        # CPU utilization trend
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        predictions["cpu_utilization"] = self._weighted_trend(cpu_values)
        
        # Memory utilization trend
        memory_values = [m.memory_utilization for m in recent_metrics]
        predictions["memory_utilization"] = self._weighted_trend(memory_values)
        
        # Queue depth trend
        queue_values = [float(m.queue_depth) for m in recent_metrics]
        predictions["queue_depth"] = max(0.0, self._weighted_trend(queue_values))
        
        # Response time trend
        response_values = [m.response_time_p95 for m in recent_metrics]
        predictions["response_time"] = max(0.0, self._weighted_trend(response_values))
        
        return predictions
    
    def _weighted_trend(self, values: List[float]) -> float:
        """Calculate weighted trend prediction."""
        if len(values) < 2:
            return values[0] if values else 0.0
        
        # Calculate trend
        if len(values) >= 3:
            trend = (values[-1] - values[-3]) / 2.0
        else:
            trend = values[-1] - values[-2]
        
        # Predict next value
        predicted = values[-1] + trend
        
        # Apply bounds
        return max(0.0, min(1.0, predicted))
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect patterns in historical data."""
        if len(self.metrics_history) < 10:
            return {"patterns": [], "confidence": 0.0}
        
        patterns = []
        
        # Detect periodic patterns (simple peak detection)
        cpu_values = [m.cpu_utilization for m in self.metrics_history]
        peaks = self._find_peaks(cpu_values)
        
        if len(peaks) > 1:
            # Calculate average interval between peaks
            intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            patterns.append({
                "type": "periodic_load",
                "interval": avg_interval,
                "amplitude": max(cpu_values) - min(cpu_values),
                "confidence": 0.7
            })
        
        # Detect trend patterns
        if len(cpu_values) >= 5:
            recent_trend = (cpu_values[-1] - cpu_values[-5]) / 5.0
            if abs(recent_trend) > 0.05:  # 5% change
                patterns.append({
                    "type": "trend",
                    "direction": "increasing" if recent_trend > 0 else "decreasing",
                    "rate": recent_trend,
                    "confidence": 0.6
                })
        
        return {
            "patterns": patterns,
            "confidence": sum(p["confidence"] for p in patterns) / max(len(patterns), 1)
        }
    
    def _find_peaks(self, values: List[float], min_distance: int = 3) -> List[int]:
        """Simple peak detection algorithm."""
        peaks = []
        
        for i in range(min_distance, len(values) - min_distance):
            # Check if current value is higher than neighbors
            is_peak = True
            for j in range(1, min_distance + 1):
                if values[i] <= values[i - j] or values[i] <= values[i + j]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks


class LoadBalancer:
    """
    Intelligent load balancer for distributing protein design tasks.
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT,
        health_check_interval: float = 30.0
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Interval between health checks (seconds)
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        self.nodes: Dict[str, LoadBalancerNode] = {}
        self.round_robin_index = 0
        
        # Health checking
        self._health_check_thread = None
        self._running = False
        
        # Statistics
        self.request_count = 0
        self.total_response_time = 0.0
        
    def add_node(
        self,
        node_id: str,
        weight: float = 1.0,
        cpu_capacity: float = 1.0,
        memory_capacity: float = 1.0,
        gpu_capacity: float = 0.0
    ):
        """Add a node to the load balancer."""
        node = LoadBalancerNode(
            node_id=node_id,
            weight=weight,
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
            gpu_capacity=gpu_capacity
        )
        
        self.nodes[node_id] = node
        
    def remove_node(self, node_id: str):
        """Remove a node from the load balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
    
    def select_node(self, task_requirements: Dict[str, float] = None) -> Optional[str]:
        """
        Select the best node for a task.
        
        Args:
            task_requirements: Task resource requirements
            
        Returns:
            Selected node ID or None if no suitable node
        """
        if not self.nodes:
            return None
        
        # Filter healthy nodes
        healthy_nodes = {
            node_id: node for node_id, node in self.nodes.items()
            if node.health_score > 0.5
        }
        
        if not healthy_nodes:
            # Use all nodes if none are healthy
            healthy_nodes = self.nodes
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(healthy_nodes, task_requirements)
        else:  # INTELLIGENT
            return self._intelligent_selection(healthy_nodes, task_requirements)
    
    def _round_robin_selection(self, nodes: Dict[str, LoadBalancerNode]) -> str:
        """Round-robin node selection."""
        node_list = list(nodes.keys())
        selected = node_list[self.round_robin_index % len(node_list)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, nodes: Dict[str, LoadBalancerNode]) -> str:
        """Select node with least connections."""
        return min(nodes.keys(), key=lambda nid: nodes[nid].current_connections)
    
    def _weighted_round_robin_selection(self, nodes: Dict[str, LoadBalancerNode]) -> str:
        """Weighted round-robin selection."""
        # Simple implementation: repeat nodes based on weight
        weighted_list = []
        for node_id, node in nodes.items():
            weight_count = max(1, int(node.weight * 10))
            weighted_list.extend([node_id] * weight_count)
        
        if weighted_list:
            selected = weighted_list[self.round_robin_index % len(weighted_list)]
            self.round_robin_index += 1
            return selected
        
        return list(nodes.keys())[0]
    
    def _resource_aware_selection(
        self, 
        nodes: Dict[str, LoadBalancerNode],
        requirements: Dict[str, float] = None
    ) -> str:
        """Select node based on resource availability."""
        if not requirements:
            return self._least_connections_selection(nodes)
        
        best_node = None
        best_score = -1.0
        
        for node_id, node in nodes.items():
            # Calculate resource utilization score
            cpu_available = max(0.0, node.cpu_capacity - node.current_connections * 0.1)
            memory_available = node.memory_capacity
            
            # Simple scoring based on available resources
            score = (cpu_available + memory_available) * node.health_score
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        return best_node or list(nodes.keys())[0]
    
    def _intelligent_selection(
        self,
        nodes: Dict[str, LoadBalancerNode],
        requirements: Dict[str, float] = None
    ) -> str:
        """Intelligent selection combining multiple factors."""
        if not requirements:
            requirements = {}
        
        best_node = None
        best_score = -1.0
        
        for node_id, node in nodes.items():
            # Multi-factor scoring
            connection_score = 1.0 / (1.0 + node.current_connections * 0.1)
            response_time_score = 1.0 / (1.0 + node.last_response_time * 0.01)
            error_score = 1.0 / (1.0 + node.error_count * 0.1)
            health_score = node.health_score
            
            # Resource matching score
            resource_score = 1.0
            if "cpu" in requirements:
                resource_score *= min(1.0, node.cpu_capacity / max(requirements["cpu"], 0.1))
            if "memory" in requirements:
                resource_score *= min(1.0, node.memory_capacity / max(requirements["memory"], 0.1))
            
            # Combined score
            total_score = (
                connection_score * 0.3 +
                response_time_score * 0.2 +
                error_score * 0.2 +
                health_score * 0.2 +
                resource_score * 0.1
            ) * node.weight
            
            if total_score > best_score:
                best_score = total_score
                best_node = node_id
        
        return best_node or list(nodes.keys())[0]
    
    def record_request(self, node_id: str, response_time: float, success: bool):
        """Record request completion."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_response_time = response_time
            
            if not success:
                node.error_count += 1
            
            self.request_count += 1
            self.total_response_time += response_time
    
    def start_health_checks(self):
        """Start health checking thread."""
        if not self._running:
            self._running = True
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self._health_check_thread.start()
    
    def stop_health_checks(self):
        """Stop health checking thread."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=1.0)
    
    def _health_check_loop(self):
        """Health check loop."""
        while self._running:
            try:
                for node_id, node in self.nodes.items():
                    # Simple health scoring based on errors and response time
                    error_penalty = min(0.5, node.error_count * 0.01)
                    response_penalty = min(0.3, node.last_response_time * 0.001)
                    
                    node.health_score = max(0.0, 1.0 - error_penalty - response_penalty)
                    
                    # Reset error count periodically
                    node.error_count = max(0, node.error_count - 1)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logging.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_connections = sum(node.current_connections for node in self.nodes.values())
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        
        node_stats = {}
        for node_id, node in self.nodes.items():
            node_stats[node_id] = {
                "connections": node.current_connections,
                "health_score": node.health_score,
                "error_count": node.error_count,
                "last_response_time": node.last_response_time,
            }
        
        return {
            "total_nodes": len(self.nodes),
            "total_connections": total_connections,
            "total_requests": self.request_count,
            "avg_response_time": avg_response_time,
            "strategy": self.strategy.value,
            "nodes": node_stats,
        }


class AutoScaler:
    """
    Main auto-scaling system coordinator.
    """
    
    def __init__(
        self,
        optimizer: PerformanceOptimizer,
        monitoring: MonitoringSystem,
        scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        min_workers: int = 1,
        max_workers: int = 16,
        scaling_interval: float = 60.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            optimizer: Performance optimizer instance
            monitoring: Monitoring system instance
            scaling_policy: Auto-scaling policy
            scale_up_threshold: Threshold for scaling up (0.0-1.0)
            scale_down_threshold: Threshold for scaling down (0.0-1.0)
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scaling_interval: Interval between scaling decisions (seconds)
        """
        self.optimizer = optimizer
        self.monitoring = monitoring
        self.scaling_policy = scaling_policy
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_interval = scaling_interval
        
        # Components
        self.predictor = PredictiveScaler()
        self.load_balancer = LoadBalancer()
        
        # State
        self._running = False
        self._scaling_thread = None
        self.scaling_history: List[ScalingAction] = []
        
        # Add default nodes to load balancer
        self._initialize_load_balancer()
    
    def _initialize_load_balancer(self):
        """Initialize load balancer with default nodes."""
        # Add CPU nodes
        for i in range(self.min_workers):
            self.load_balancer.add_node(
                f"cpu_worker_{i}",
                weight=1.0,
                cpu_capacity=1.0,
                memory_capacity=1.0
            )
        
        # Add GPU nodes if available
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(min(gpu_count, 2)):  # Max 2 GPU workers initially
                self.load_balancer.add_node(
                    f"gpu_worker_{i}",
                    weight=2.0,  # Higher weight for GPU
                    cpu_capacity=0.5,
                    memory_capacity=2.0,
                    gpu_capacity=1.0
                )
    
    def start(self):
        """Start auto-scaling system."""
        if not self._running:
            self._running = True
            self._scaling_thread = threading.Thread(
                target=self._scaling_loop,
                daemon=True
            )
            self._scaling_thread.start()
            self.load_balancer.start_health_checks()
    
    def stop(self):
        """Stop auto-scaling system."""
        self._running = False
        if self._scaling_thread:
            self._scaling_thread.join(timeout=2.0)
        self.load_balancer.stop_health_checks()
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self._running:
            try:
                # Collect current metrics
                status = self.monitoring.get_status()
                metrics = self._extract_metrics(status)
                
                # Add to predictor
                self.predictor.add_metrics(metrics)
                
                # Make scaling decision
                actions = self._make_scaling_decision(metrics)
                
                # Execute actions
                for action in actions:
                    self._execute_scaling_action(action)
                    self.scaling_history.append(action)
                
                # Keep history manageable
                if len(self.scaling_history) > 100:
                    self.scaling_history = self.scaling_history[-50:]
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                logging.error(f"Auto-scaling error: {e}")
                time.sleep(self.scaling_interval)
    
    def _extract_metrics(self, status: Dict[str, Any]) -> ScalingMetrics:
        """Extract scaling metrics from monitoring status."""
        metrics_data = status.get("metrics", {})
        
        return ScalingMetrics(
            cpu_utilization=metrics_data.get("system_cpu_usage_percent", {}).get("value", 0.0) / 100.0,
            memory_utilization=metrics_data.get("system_memory_usage_percent", {}).get("value", 0.0) / 100.0,
            queue_depth=len(status.get("active_operations", [])),
            response_time_p95=0.0,  # Would need histogram data
            error_rate=0.0,  # Would need error metrics
            throughput=0.0,  # Would need throughput calculation
        )
    
    def _make_scaling_decision(self, current_metrics: ScalingMetrics) -> List[ScalingAction]:
        """Make scaling decisions based on current metrics and policy."""
        actions = []
        
        if self.scaling_policy in [ScalingPolicy.REACTIVE, ScalingPolicy.HYBRID]:
            actions.extend(self._reactive_scaling(current_metrics))
        
        if self.scaling_policy in [ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID]:
            predicted_metrics = self.predictor.predict_load()
            actions.extend(self._predictive_scaling(predicted_metrics))
        
        return actions
    
    def _reactive_scaling(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Reactive scaling based on current metrics."""
        actions = []
        current_workers = len([n for n in self.load_balancer.nodes.keys() if "cpu" in n])
        
        # Scale up conditions
        if (metrics.cpu_utilization > self.scale_up_threshold or
            metrics.queue_depth > 10 or
            metrics.memory_utilization > self.scale_up_threshold):
            
            if current_workers < self.max_workers:
                target_workers = min(self.max_workers, current_workers + 1)
                actions.append(ScalingAction(
                    action_type="scale_up",
                    resource_type="cpu",
                    target_value=target_workers,
                    reason=f"High utilization: CPU={metrics.cpu_utilization:.1%}, Queue={metrics.queue_depth}",
                    confidence=0.8
                ))
        
        # Scale down conditions
        elif (metrics.cpu_utilization < self.scale_down_threshold and
              metrics.queue_depth < 2 and
              metrics.memory_utilization < self.scale_down_threshold):
            
            if current_workers > self.min_workers:
                target_workers = max(self.min_workers, current_workers - 1)
                actions.append(ScalingAction(
                    action_type="scale_down",
                    resource_type="cpu",
                    target_value=target_workers,
                    reason=f"Low utilization: CPU={metrics.cpu_utilization:.1%}, Queue={metrics.queue_depth}",
                    confidence=0.6
                ))
        
        return actions
    
    def _predictive_scaling(self, predicted_metrics: Dict[str, float]) -> List[ScalingAction]:
        """Predictive scaling based on forecasted metrics."""
        actions = []
        current_workers = len([n for n in self.load_balancer.nodes.keys() if "cpu" in n])
        
        predicted_cpu = predicted_metrics.get("cpu_utilization", 0.0)
        predicted_queue = predicted_metrics.get("queue_depth", 0.0)
        
        # Predictive scale up
        if predicted_cpu > self.scale_up_threshold and current_workers < self.max_workers:
            target_workers = min(self.max_workers, current_workers + 1)
            actions.append(ScalingAction(
                action_type="scale_up",
                resource_type="cpu",
                target_value=target_workers,
                reason=f"Predicted high CPU: {predicted_cpu:.1%}",
                confidence=0.6
            ))
        
        # Predictive scale down  
        elif predicted_cpu < self.scale_down_threshold and current_workers > self.min_workers:
            target_workers = max(self.min_workers, current_workers - 1)
            actions.append(ScalingAction(
                action_type="scale_down",
                resource_type="cpu",
                target_value=target_workers,
                reason=f"Predicted low CPU: {predicted_cpu:.1%}",
                confidence=0.4
            ))
        
        return actions
    
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action."""
        logging.info(f"Executing scaling action: {action.action_type} {action.resource_type} to {action.target_value}")
        
        if action.resource_type == "cpu":
            if action.action_type == "scale_up":
                self._scale_up_cpu_workers(action.target_value)
            elif action.action_type == "scale_down":
                self._scale_down_cpu_workers(action.target_value)
        
        # Could extend for GPU scaling, memory optimization, etc.
    
    def _scale_up_cpu_workers(self, target_workers: int):
        """Scale up CPU workers."""
        current_cpu_nodes = [n for n in self.load_balancer.nodes.keys() if "cpu" in n]
        current_count = len(current_cpu_nodes)
        
        for i in range(current_count, target_workers):
            node_id = f"cpu_worker_{i}"
            self.load_balancer.add_node(node_id, cpu_capacity=1.0, memory_capacity=1.0)
        
        # Update executor
        self.optimizer.executor.scale_resources(cpu_workers=target_workers)
    
    def _scale_down_cpu_workers(self, target_workers: int):
        """Scale down CPU workers."""
        current_cpu_nodes = [n for n in self.load_balancer.nodes.keys() if "cpu" in n]
        current_count = len(current_cpu_nodes)
        
        # Remove excess nodes
        for i in range(target_workers, current_count):
            node_id = f"cpu_worker_{i}"
            self.load_balancer.remove_node(node_id)
        
        # Update executor
        self.optimizer.executor.scale_resources(cpu_workers=target_workers)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        recent_actions = [a for a in self.scaling_history if time.time() - a.timestamp < 3600]
        
        action_counts = {}
        for action in recent_actions:
            key = f"{action.action_type}_{action.resource_type}"
            action_counts[key] = action_counts.get(key, 0) + 1
        
        return {
            "scaling_policy": self.scaling_policy.value,
            "thresholds": {
                "scale_up": self.scale_up_threshold,
                "scale_down": self.scale_down_threshold,
            },
            "worker_limits": {
                "min": self.min_workers,
                "max": self.max_workers,
            },
            "recent_actions": action_counts,
            "total_actions": len(self.scaling_history),
            "load_balancer_stats": self.load_balancer.get_stats(),
        }


# Integration functions
def create_optimized_system(
    cache_size: int = 1000,
    max_workers: int = 8,
    enable_auto_scaling: bool = True
) -> Tuple[PerformanceOptimizer, AutoScaler]:
    """
    Create an optimized system with auto-scaling capabilities.
    
    Args:
        cache_size: Cache size for performance optimizer
        max_workers: Maximum number of workers
        enable_auto_scaling: Whether to enable auto-scaling
        
    Returns:
        Tuple of (PerformanceOptimizer, AutoScaler)
    """
    from .monitoring_system import MonitoringSystem
    
    # Create components
    optimizer = PerformanceOptimizer(cache_size=cache_size)
    monitoring = MonitoringSystem()
    
    if enable_auto_scaling:
        auto_scaler = AutoScaler(
            optimizer=optimizer,
            monitoring=monitoring,
            max_workers=max_workers
        )
    else:
        auto_scaler = None
    
    return optimizer, auto_scaler