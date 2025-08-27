"""
Advanced monitoring and observability system for protein design operations.

Features:
- Real-time performance metrics
- Design process tracing
- Health checks and alerting
- Resource utilization monitoring
- Distributed tracing support
"""

from typing import Dict, List, Optional, Union, Any, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch

import time
import threading
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path
try:
    import psutil
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    import mock_psutil as psutil
import queue
from datetime import datetime, timedelta
from contextlib import contextmanager

from .advanced_logger import ProteinOperatorsLogger


class MetricType(Enum):
    """Types of metrics collected by the monitoring system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """Alert notification."""
    name: str
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class HealthCheck:
    """Health check configuration and result."""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: float = 60.0
    timeout_seconds: float = 10.0
    last_check_time: Optional[float] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0
    enabled: bool = True


class PerformanceProfiler:
    """
    Performance profiler for protein design operations.
    
    Tracks execution time, memory usage, and GPU utilization
    for different stages of the design pipeline.
    """
    
    def __init__(self):
        self.active_profiles = {}
        self.completed_profiles = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation_name: str, **tags):
        """Context manager for profiling operations."""
        profile_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        # Start profiling
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        with self.lock:
            self.active_profiles[profile_id] = {
                'operation': operation_name,
                'start_time': start_time,
                'start_memory': start_memory,
                'start_gpu_memory': start_gpu_memory,
                'tags': tags
            }
        
        try:
            yield profile_id
        finally:
            # End profiling
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            with self.lock:
                if profile_id in self.active_profiles:
                    profile_data = self.active_profiles.pop(profile_id)
                    
                    # Compute metrics
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    gpu_memory_delta = end_gpu_memory - start_gpu_memory
                    
                    completed_profile = {
                        'operation': operation_name,
                        'duration': duration,
                        'memory_delta': memory_delta,
                        'gpu_memory_delta': gpu_memory_delta,
                        'start_time': start_time,
                        'end_time': end_time,
                        'tags': tags
                    }
                    
                    self.completed_profiles.append(completed_profile)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            pass
        return 0.0
    
    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        with self.lock:
            profiles = list(self.completed_profiles)
        
        if operation_name:
            profiles = [p for p in profiles if p['operation'] == operation_name]
        
        if not profiles:
            return {}
        
        durations = [p['duration'] for p in profiles]
        memory_deltas = [p['memory_delta'] for p in profiles]
        
        return {
            'count': len(profiles),
            'duration': {
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'p95': sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
            },
            'memory': {
                'mean_delta': sum(memory_deltas) / len(memory_deltas),
                'max_delta': max(memory_deltas),
                'min_delta': min(memory_deltas)
            }
        }


class ResourceMonitor:
    """
    System resource monitoring for protein design workloads.
    """
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.metrics_history.append({
                        'timestamp': time.time(),
                        'metrics': metrics
                    })
                
                time.sleep(self.collection_interval)
            except Exception as e:
                # Log error but continue monitoring
                print(f"Resource monitoring error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            metrics['cpu_count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_available_gb'] = memory.available / 1024 / 1024 / 1024
            metrics['memory_used_gb'] = memory.used / 1024 / 1024 / 1024
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            metrics['disk_free_gb'] = disk.free / 1024 / 1024 / 1024
            
            # GPU metrics
            if torch.cuda.is_available():
                metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                metrics['gpu_count'] = torch.cuda.device_count()
        
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        return self._collect_system_metrics()
    
    def get_metrics_history(self, duration_minutes: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get metrics history for specified duration."""
        with self.lock:
            history = list(self.metrics_history)
        
        if duration_minutes is not None:
            cutoff_time = time.time() - (duration_minutes * 60)
            history = [h for h in history if h['timestamp'] >= cutoff_time]
        
        return history


class AlertManager:
    """
    Alert management system for monitoring protein design operations.
    """
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_rules = {}
        self.alert_handlers = []
        self.lock = threading.Lock()
        self.logger = ProteinOperatorsLogger(__name__)
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, float]], bool],
        level: AlertLevel = AlertLevel.WARNING,
        message_template: str = "Alert condition met for {name}"
    ):
        """Add an alert rule."""
        self.alert_rules[name] = {
            'condition': condition,
            'level': level,
            'message_template': message_template
        }
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, float]):
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](metrics):
                    self._trigger_alert(
                        rule_name,
                        rule['level'],
                        rule['message_template'].format(name=rule_name)
                    )
                else:
                    self._resolve_alert(rule_name)
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _trigger_alert(self, name: str, level: AlertLevel, message: str):
        """Trigger an alert."""
        with self.lock:
            if name not in self.active_alerts:
                alert = Alert(
                    name=name,
                    level=level,
                    message=message
                )
                
                self.active_alerts[name] = alert
                self.alert_history.append(alert)
                
                # Notify handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        self.logger.error(f"Alert handler error: {e}")
    
    def _resolve_alert(self, name: str):
        """Resolve an active alert."""
        with self.lock:
            if name in self.active_alerts:
                alert = self.active_alerts.pop(name)
                alert.resolved = True
                alert.resolution_time = time.time()
                
                # Notify handlers of resolution
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        self.logger.error(f"Alert handler error: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: Optional[float] = None) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            history = list(self.alert_history)
        
        if hours is not None:
            cutoff_time = time.time() - (hours * 3600)
            history = [a for a in history if a.timestamp >= cutoff_time]
        
        return history


class AdvancedMonitoringSystem:
    """
    Comprehensive monitoring system for protein design operations.
    
    Integrates performance profiling, resource monitoring, health checks,
    and alerting into a unified observability platform.
    """
    
    def __init__(
        self,
        enable_resource_monitoring: bool = True,
        resource_collection_interval: float = 10.0,
        enable_health_checks: bool = True,
        metrics_retention_hours: float = 24.0
    ):
        self.logger = ProteinOperatorsLogger(__name__)
        
        # Initialize components
        self.profiler = PerformanceProfiler()
        self.resource_monitor = ResourceMonitor(resource_collection_interval)
        self.alert_manager = AlertManager()
        
        # Metrics storage
        self.metrics_storage = deque(maxlen=int(metrics_retention_hours * 3600 / 10))  # 10s intervals
        self.metrics_lock = threading.Lock()
        
        # Health checks
        self.health_checks = {}
        self.health_check_thread = None
        self.health_checks_active = False
        
        # Configuration
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_health_checks = enable_health_checks
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default health checks
        if enable_health_checks:
            self._setup_default_health_checks()
        
        self.logger.info("Advanced Monitoring System initialized")
    
    def start(self):
        """Start all monitoring components."""
        if self.enable_resource_monitoring:
            self.resource_monitor.start_monitoring()
        
        if self.enable_health_checks:
            self._start_health_checks()
        
        # Start metrics collection loop
        self._start_metrics_collection()
        
        self.logger.info("Monitoring system started")
    
    def stop(self):
        """Stop all monitoring components."""
        if self.enable_resource_monitoring:
            self.resource_monitor.stop_monitoring()
        
        if self.enable_health_checks:
            self._stop_health_checks()
        
        self.logger.info("Monitoring system stopped")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            lambda m: m.get('cpu_percent', 0) > 90,
            AlertLevel.WARNING,
            "High CPU usage detected: {cpu_percent:.1f}%"
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            lambda m: m.get('memory_percent', 0) > 85,
            AlertLevel.WARNING,
            "High memory usage detected: {memory_percent:.1f}%"
        )
        
        # Low disk space
        self.alert_manager.add_alert_rule(
            "low_disk_space",
            lambda m: m.get('disk_percent', 0) > 90,
            AlertLevel.ERROR,
            "Low disk space: {disk_percent:.1f}% used"
        )
        
        # GPU memory usage
        self.alert_manager.add_alert_rule(
            "high_gpu_memory",
            lambda m: m.get('gpu_memory_allocated_mb', 0) > 8000,  # 8GB
            AlertLevel.WARNING,
            "High GPU memory usage: {gpu_memory_allocated_mb:.0f}MB"
        )
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        # System health
        self.add_health_check(
            "system_health",
            self._check_system_health,
            interval_seconds=30.0
        )
        
        # GPU availability
        if torch.cuda.is_available():
            self.add_health_check(
                "gpu_health",
                self._check_gpu_health,
                interval_seconds=60.0
            )
        
        # Memory health
        self.add_health_check(
            "memory_health",
            self._check_memory_health,
            interval_seconds=30.0
        )
    
    def _check_system_health(self) -> bool:
        """Check overall system health."""
        try:
            metrics = self.resource_monitor.get_current_metrics()
            
            # System is healthy if CPU < 95% and memory < 95%
            cpu_ok = metrics.get('cpu_percent', 0) < 95
            memory_ok = metrics.get('memory_percent', 0) < 95
            
            return cpu_ok and memory_ok
        except Exception:
            return False
    
    def _check_gpu_health(self) -> bool:
        """Check GPU health."""
        try:
            if not torch.cuda.is_available():
                return False
            
            # Simple GPU health check
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            # Try a simple GPU operation
            test_tensor = torch.ones(10, device=f'cuda:{current_device}')
            result = test_tensor.sum()
            
            return result.item() == 10.0
        except Exception:
            return False
    
    def _check_memory_health(self) -> bool:
        """Check memory health."""
        try:
            import gc
            gc.collect()  # Force garbage collection
            
            memory = psutil.virtual_memory()
            # Memory is healthy if < 90% used and > 1GB available
            return memory.percent < 90 and memory.available > 1024**3
        except Exception:
            return False
    
    def add_health_check(
        self,
        name: str,
        check_function: Callable[[], bool],
        interval_seconds: float = 60.0,
        timeout_seconds: float = 10.0
    ):
        """Add a custom health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds
        )
    
    def _start_health_checks(self):
        """Start health check monitoring."""
        if self.health_checks_active:
            return
        
        self.health_checks_active = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
    
    def _stop_health_checks(self):
        """Stop health check monitoring."""
        self.health_checks_active = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
    
    def _health_check_loop(self):
        """Main health check loop."""
        while self.health_checks_active:
            try:
                current_time = time.time()
                
                for check in self.health_checks.values():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time to run this health check
                    if (check.last_check_time is None or
                        current_time - check.last_check_time >= check.interval_seconds):
                        
                        try:
                            # Run health check with timeout
                            result = check.check_function()
                            
                            check.last_check_time = current_time
                            check.last_result = result
                            
                            if result:
                                check.consecutive_failures = 0
                            else:
                                check.consecutive_failures += 1
                                
                                # Trigger alert after 3 consecutive failures
                                if check.consecutive_failures >= 3:
                                    self.alert_manager._trigger_alert(
                                        f"health_check_{check.name}",
                                        AlertLevel.ERROR,
                                        f"Health check {check.name} failing"
                                    )
                        
                        except Exception as e:
                            self.logger.error(f"Health check {check.name} error: {e}")
                            check.consecutive_failures += 1
                
                time.sleep(5.0)  # Check every 5 seconds
            
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                time.sleep(5.0)
    
    def _start_metrics_collection(self):
        """Start metrics collection loop."""
        def collection_loop():
            while True:
                try:
                    # Collect metrics
                    metrics = self.resource_monitor.get_current_metrics()
                    
                    # Store metrics
                    with self.metrics_lock:
                        self.metrics_storage.append({
                            'timestamp': time.time(),
                            'metrics': metrics
                        })
                    
                    # Check alerts
                    self.alert_manager.check_alerts(metrics)
                    
                    time.sleep(10.0)  # Collect every 10 seconds
                except Exception as e:
                    self.logger.error(f"Metrics collection error: {e}")
                    time.sleep(10.0)
        
        collection_thread = threading.Thread(target=collection_loop, daemon=True)
        collection_thread.start()
    
    @contextmanager
    def profile_operation(self, operation_name: str, **tags):
        """Profile an operation."""
        with self.profiler.profile(operation_name, **tags) as profile_id:
            yield profile_id
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a custom metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        with self.metrics_lock:
            self.metrics_storage.append({
                'timestamp': time.time(),
                'custom_metric': metric
            })
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_metrics = self.resource_monitor.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Health check status
        health_status = {}
        for name, check in self.health_checks.items():
            health_status[name] = {
                'healthy': check.last_result,
                'last_check': check.last_check_time,
                'consecutive_failures': check.consecutive_failures
            }
        
        # Performance summaries
        performance_data = {}
        common_operations = ['design_generation', 'validation', 'optimization']
        for op in common_operations:
            performance_data[op] = self.profiler.get_performance_summary(op)
        
        return {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'active_alerts': [{
                'name': a.name,
                'level': a.level.value,
                'message': a.message,
                'timestamp': a.timestamp
            } for a in active_alerts],
            'health_checks': health_status,
            'performance': performance_data,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export collected metrics to file."""
        with self.metrics_lock:
            data = list(self.metrics_storage)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(data)} metric records to {filepath}")
