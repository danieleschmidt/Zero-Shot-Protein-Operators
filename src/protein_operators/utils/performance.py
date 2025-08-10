"""
Performance monitoring and optimization utilities for protein operators.
"""

import time
import logging
import threading
import psutil
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import functools
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceCategory(Enum):
    """Categories for performance metrics."""
    COMPUTATION = "computation"
    IO = "io"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    CACHE = "cache"


@dataclass
class PerformanceMetric:
    """Single performance metric."""
    name: str
    value: float
    unit: str
    category: PerformanceCategory
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'category': self.category.value,
            'timestamp': self.timestamp,
            'context': self.context,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection
    - System resource monitoring
    - GPU utilization tracking
    - Performance profiling
    - Automatic optimization recommendations
    """
    
    def __init__(
        self,
        collect_system_metrics: bool = True,
        collect_gpu_metrics: bool = True,
        metric_history_size: int = 10000,
        auto_optimize: bool = True
    ):
        """
        Initialize performance monitor.
        
        Args:
            collect_system_metrics: Enable system resource monitoring
            collect_gpu_metrics: Enable GPU metrics collection
            metric_history_size: Maximum number of metrics to store
            auto_optimize: Enable automatic optimization recommendations
        """
        self.collect_system_metrics = collect_system_metrics
        self.collect_gpu_metrics = collect_gpu_metrics
        self.metric_history_size = metric_history_size
        self.auto_optimize = auto_optimize
        
        # Metric storage
        self.metrics: List[PerformanceMetric] = []
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()
        
        # System monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_interval = 5.0  # seconds
        
        # Performance optimization
        self.optimization_rules: List[Callable] = []
        self.recommendations: List[Dict[str, Any]] = []
        
        # Initialize monitoring
        if self.collect_system_metrics:
            self.start_system_monitoring()
        
        logger.info("PerformanceMonitor initialized")
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        category: PerformanceCategory = PerformanceCategory.COMPUTATION,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            category: Metric category
            context: Additional context information
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            category=category,
            timestamp=time.time(),
            context=context or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            
            # Maintain history size limit
            if len(self.metrics) > self.metric_history_size:
                self.metrics = self.metrics[-self.metric_history_size:]
            
            # Update aggregated metrics
            self._update_aggregated_metrics(metric)
        
        # Auto-optimize if enabled
        if self.auto_optimize:
            self._check_optimization_rules(metric)
    
    def _update_aggregated_metrics(self, metric: PerformanceMetric) -> None:
        """Update aggregated metric statistics."""
        key = f"{metric.category.value}_{metric.name}"
        
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0.0,
                'recent_values': []
            }
        
        stats = self.aggregated_metrics[key]
        stats['count'] += 1
        stats['sum'] += metric.value
        stats['min'] = min(stats['min'], metric.value)
        stats['max'] = max(stats['max'], metric.value)
        stats['avg'] = stats['sum'] / stats['count']
        
        # Keep recent values for trend analysis
        stats['recent_values'].append(metric.value)
        if len(stats['recent_values']) > 100:
            stats['recent_values'] = stats['recent_values'][-100:]
    
    def start_system_monitoring(self) -> None:
        """Start background system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True,
            name="performance-monitor"
        )
        self._monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_system_monitoring(self) -> None:
        """Stop background system monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        logger.info("System monitoring stopped")
    
    def _system_monitoring_loop(self) -> None:
        """Background loop for system monitoring."""
        while self._monitoring_active:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1.0)
                self.record_metric(
                    "cpu_usage",
                    cpu_percent,
                    "percent",
                    PerformanceCategory.COMPUTATION
                )
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_metric(
                    "memory_usage",
                    memory.percent,
                    "percent",
                    PerformanceCategory.MEMORY
                )
                self.record_metric(
                    "memory_available",
                    memory.available / (1024**3),
                    "GB",
                    PerformanceCategory.MEMORY
                )
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.record_metric(
                        "disk_read_rate",
                        disk_io.read_bytes / (1024**2),  # MB
                        "MB",
                        PerformanceCategory.IO
                    )
                    self.record_metric(
                        "disk_write_rate",
                        disk_io.write_bytes / (1024**2),  # MB
                        "MB",
                        PerformanceCategory.IO
                    )
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.record_metric(
                        "network_sent",
                        network_io.bytes_sent / (1024**2),  # MB
                        "MB",
                        PerformanceCategory.NETWORK
                    )
                    self.record_metric(
                        "network_recv",
                        network_io.bytes_recv / (1024**2),  # MB
                        "MB",
                        PerformanceCategory.NETWORK
                    )
                
                # GPU metrics if available
                if self.collect_gpu_metrics and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        # GPU utilization
                        gpu_utilization = torch.cuda.utilization(i)
                        self.record_metric(
                            f"gpu_{i}_utilization",
                            gpu_utilization,
                            "percent",
                            PerformanceCategory.GPU,
                            {"gpu_id": i}
                        )
                        
                        # GPU memory
                        memory_info = torch.cuda.mem_get_info(i)
                        memory_used = (memory_info[1] - memory_info[0]) / (1024**3)
                        memory_total = memory_info[1] / (1024**3)
                        memory_percent = (memory_used / memory_total) * 100
                        
                        self.record_metric(
                            f"gpu_{i}_memory_usage",
                            memory_percent,
                            "percent",
                            PerformanceCategory.GPU,
                            {"gpu_id": i, "used_gb": memory_used, "total_gb": memory_total}
                        )
                
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self._monitoring_interval)
    
    def get_metrics(
        self,
        category: Optional[PerformanceCategory] = None,
        name_pattern: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[PerformanceMetric]:
        """
        Get metrics with optional filtering.
        
        Args:
            category: Filter by category
            name_pattern: Filter by name pattern
            since: Only return metrics after this timestamp
            
        Returns:
            List of filtered metrics
        """
        with self._lock:
            filtered_metrics = []
            
            for metric in self.metrics:
                # Category filter
                if category and metric.category != category:
                    continue
                
                # Name pattern filter
                if name_pattern and name_pattern not in metric.name:
                    continue
                
                # Time filter
                if since and metric.timestamp < since:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary performance statistics."""
        with self._lock:
            current_time = time.time()
            recent_threshold = current_time - 300  # Last 5 minutes
            
            summary = {
                'total_metrics': len(self.metrics),
                'monitoring_active': self._monitoring_active,
                'categories': {},
                'recent_performance': {},
                'system_health': self._assess_system_health(),
                'recommendations': self.recommendations[-10:],  # Last 10 recommendations
            }
            
            # Aggregate by category
            for metric in self.metrics:
                if metric.timestamp < recent_threshold:
                    continue
                
                category_name = metric.category.value
                if category_name not in summary['categories']:
                    summary['categories'][category_name] = {
                        'count': 0,
                        'metrics': set()
                    }
                
                summary['categories'][category_name]['count'] += 1
                summary['categories'][category_name]['metrics'].add(metric.name)
            
            # Convert sets to lists for JSON serialization
            for category_data in summary['categories'].values():
                category_data['metrics'] = list(category_data['metrics'])
            
            # Recent performance trends
            for key, stats in self.aggregated_metrics.items():
                if stats['recent_values'] and len(stats['recent_values']) >= 10:
                    recent_values = stats['recent_values'][-20:]  # Last 20 values
                    trend = self._calculate_trend(recent_values)
                    
                    summary['recent_performance'][key] = {
                        'current': recent_values[-1] if recent_values else 0,
                        'avg': stats['avg'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'trend': trend,
                        'count': stats['count']
                    }
            
            return summary
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        health = {
            'status': 'good',
            'issues': [],
            'warnings': []
        }
        
        # Check recent CPU usage
        cpu_metrics = self.get_metrics(
            PerformanceCategory.COMPUTATION,
            name_pattern="cpu_usage",
            since=time.time() - 300  # Last 5 minutes
        )
        
        if cpu_metrics:
            avg_cpu = sum(m.value for m in cpu_metrics) / len(cpu_metrics)
            if avg_cpu > 90:
                health['issues'].append("High CPU usage detected")
                health['status'] = 'critical'
            elif avg_cpu > 75:
                health['warnings'].append("Elevated CPU usage")
                if health['status'] == 'good':
                    health['status'] = 'warning'
        
        # Check memory usage
        memory_metrics = self.get_metrics(
            PerformanceCategory.MEMORY,
            name_pattern="memory_usage",
            since=time.time() - 300
        )
        
        if memory_metrics:
            avg_memory = sum(m.value for m in memory_metrics) / len(memory_metrics)
            if avg_memory > 95:
                health['issues'].append("Critical memory usage")
                health['status'] = 'critical'
            elif avg_memory > 80:
                health['warnings'].append("High memory usage")
                if health['status'] == 'good':
                    health['status'] = 'warning'
        
        # Check GPU health if available
        if self.collect_gpu_metrics and torch.cuda.is_available():
            gpu_metrics = self.get_metrics(
                PerformanceCategory.GPU,
                since=time.time() - 300
            )
            
            gpu_memory_usage = [
                m.value for m in gpu_metrics 
                if "memory_usage" in m.name
            ]
            
            if gpu_memory_usage:
                avg_gpu_memory = sum(gpu_memory_usage) / len(gpu_memory_usage)
                if avg_gpu_memory > 95:
                    health['issues'].append("GPU memory critical")
                    health['status'] = 'critical'
                elif avg_gpu_memory > 85:
                    health['warnings'].append("GPU memory high")
                    if health['status'] == 'good':
                        health['status'] = 'warning'
        
        return health
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from recent values."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple trend analysis using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing" if slope > 1 else "slowly_increasing"
        else:
            return "decreasing" if slope < -1 else "slowly_decreasing"
    
    def add_optimization_rule(self, rule_func: Callable[[PerformanceMetric], Optional[str]]) -> None:
        """
        Add optimization rule.
        
        Args:
            rule_func: Function that takes a metric and returns optimization recommendation
        """
        self.optimization_rules.append(rule_func)
        logger.info(f"Added optimization rule: {rule_func.__name__}")
    
    def _check_optimization_rules(self, metric: PerformanceMetric) -> None:
        """Check optimization rules against new metric."""
        for rule_func in self.optimization_rules:
            try:
                recommendation = rule_func(metric)
                if recommendation:
                    self.recommendations.append({
                        'timestamp': time.time(),
                        'metric': metric.name,
                        'category': metric.category.value,
                        'recommendation': recommendation,
                        'metric_value': metric.value,
                        'context': metric.context
                    })
                    
                    # Limit recommendation history
                    if len(self.recommendations) > 1000:
                        self.recommendations = self.recommendations[-1000:]
                    
                    logger.info(f"Optimization recommendation: {recommendation}")
                    
            except Exception as e:
                logger.error(f"Error in optimization rule {rule_func.__name__}: {e}")
    
    def export_metrics(self, filepath: Path, format: str = "json") -> None:
        """Export metrics to file."""
        with self._lock:
            data = {
                'export_timestamp': time.time(),
                'total_metrics': len(self.metrics),
                'aggregated_metrics': self.aggregated_metrics,
                'recommendations': self.recommendations,
                'metrics': [metric.to_dict() for metric in self.metrics]
            }
            
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")


@contextmanager
def performance_timer(
    monitor: PerformanceMonitor,
    operation_name: str,
    category: PerformanceCategory = PerformanceCategory.COMPUTATION,
    context: Optional[Dict[str, Any]] = None
):
    """
    Context manager for timing operations.
    
    Args:
        monitor: Performance monitor instance
        operation_name: Name of the operation being timed
        category: Performance category
        context: Additional context information
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        monitor.record_metric(
            f"{operation_name}_duration",
            duration,
            "seconds",
            category,
            context
        )


def performance_profiler(
    monitor: Optional[PerformanceMonitor] = None,
    category: PerformanceCategory = PerformanceCategory.COMPUTATION,
    include_args: bool = False
):
    """
    Decorator for automatic performance profiling.
    
    Args:
        monitor: Performance monitor instance
        category: Performance category
        include_args: Whether to include function arguments in context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal monitor
            if monitor is None:
                monitor = get_performance_monitor()
            
            operation_name = f"{func.__module__}.{func.__name__}"
            context = {}
            
            if include_args:
                context.update({
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })
            
            with performance_timer(monitor, operation_name, category, context):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """
    Automatic performance optimization system.
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Add default optimization rules
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default optimization rules."""
        
        def high_cpu_rule(metric: PerformanceMetric) -> Optional[str]:
            """Rule for high CPU usage."""
            if metric.name == "cpu_usage" and metric.value > 85:
                return "Consider reducing batch size or enabling parallel processing"
            return None
        
        def high_memory_rule(metric: PerformanceMetric) -> Optional[str]:
            """Rule for high memory usage."""
            if metric.name == "memory_usage" and metric.value > 90:
                return "Consider reducing data batch size or enabling data streaming"
            return None
        
        def slow_operation_rule(metric: PerformanceMetric) -> Optional[str]:
            """Rule for slow operations."""
            if "duration" in metric.name and metric.value > 60:  # 60 seconds
                return f"Operation {metric.name} is taking too long. Consider optimization or caching"
            return None
        
        def gpu_memory_rule(metric: PerformanceMetric) -> Optional[str]:
            """Rule for GPU memory usage."""
            if "gpu_" in metric.name and "memory_usage" in metric.name and metric.value > 90:
                gpu_id = metric.context.get('gpu_id', 'unknown')
                return f"GPU {gpu_id} memory usage critical. Consider reducing model size or batch size"
            return None
        
        self.monitor.add_optimization_rule(high_cpu_rule)
        self.monitor.add_optimization_rule(high_memory_rule)
        self.monitor.add_optimization_rule(slow_operation_rule)
        self.monitor.add_optimization_rule(gpu_memory_rule)


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer(_global_performance_monitor)
    
    return _global_performance_monitor


def configure_performance_monitoring(**kwargs) -> PerformanceMonitor:
    """Configure global performance monitoring."""
    global _global_performance_monitor
    _global_performance_monitor = PerformanceMonitor(**kwargs)
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(_global_performance_monitor)
    
    return _global_performance_monitor


# Convenient shortcuts
def record_metric(name: str, value: float, unit: str = "", **kwargs) -> None:
    """Record a performance metric using the global monitor."""
    monitor = get_performance_monitor()
    monitor.record_metric(name, value, unit, **kwargs)


def time_operation(name: str, **kwargs):
    """Context manager for timing operations using the global monitor."""
    monitor = get_performance_monitor()
    return performance_timer(monitor, name, **kwargs)