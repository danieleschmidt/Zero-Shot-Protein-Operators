"""
Comprehensive monitoring and observability system for protein design.

This module provides:
- Performance monitoring
- Resource usage tracking
- Health checks
- Metrics collection and reporting
- Alerting system
"""

import time
import threading
try:
    import psutil
except ImportError:
    psutil = None
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """A metric value with timestamp."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    check_time: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Alert definition."""
    name: str
    level: AlertLevel
    message: str
    metric: str
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """
    Collector for various types of metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metric values to store
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric."""
        with self._lock:
            metric_value = MetricValue(value, datetime.now(), labels or {})
            self.metrics[name].append(metric_value)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric."""
        with self._lock:
            metric_value = MetricValue(value, datetime.now(), labels or {})
            self.metrics[name].append(metric_value)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        with self._lock:
            metric_value = MetricValue(value, datetime.now(), labels or {})
            self.metrics[name].append(metric_value)
    
    def get_latest(self, name: str) -> Optional[MetricValue]:
        """Get latest value for a metric."""
        with self._lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
        return None
    
    def get_history(self, name: str, minutes: int = 60) -> List[MetricValue]:
        """Get metric history for the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            return [mv for mv in self.metrics[name] if mv.timestamp > cutoff]
    
    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """Get all current metrics in serializable format."""
        with self._lock:
            result = {}
            for name, values in self.metrics.items():
                if values:
                    latest = values[-1]
                    result[name] = {
                        "value": latest.value,
                        "timestamp": latest.timestamp.isoformat(),
                        "labels": latest.labels
                    }
            return result


class PerformanceMonitor:
    """
    Monitor for performance metrics.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._active_operations = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start monitoring an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        with self._lock:
            self._active_operations[operation_id] = {
                "name": operation_name,
                "start_time": time.time(),
                "metadata": metadata or {}
            }
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, metadata: Dict[str, Any] = None):
        """End monitoring an operation."""
        end_time = time.time()
        
        with self._lock:
            if operation_id in self._active_operations:
                op_info = self._active_operations[operation_id]
                duration = end_time - op_info["start_time"]
                
                # Record metrics
                labels = {
                    "operation": op_info["name"],
                    "success": str(success)
                }
                
                self.collector.record_histogram("operation_duration_seconds", duration, labels)
                self.collector.record_counter("operations_total", 1.0, labels)
                
                if not success:
                    self.collector.record_counter("operation_errors_total", 1.0, {"operation": op_info["name"]})
                
                # Remove from active operations
                del self._active_operations[operation_id]
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get list of currently active operations."""
        current_time = time.time()
        
        with self._lock:
            active = []
            for op_id, op_info in self._active_operations.items():
                duration = current_time - op_info["start_time"]
                active.append({
                    "id": op_id,
                    "name": op_info["name"],
                    "duration_seconds": duration,
                    "metadata": op_info["metadata"]
                })
            
            return active


class ResourceMonitor:
    """
    Monitor for system resource usage.
    """
    
    def __init__(self, collector: MetricsCollector, update_interval: float = 5.0):
        self.collector = collector
        self.update_interval = update_interval
        self._running = False
        self._thread = None
    
    def start(self):
        """Start resource monitoring."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        if psutil is None:
            # Mock metrics when psutil is not available
            self.collector.record_gauge("system_cpu_usage_percent", 25.0)
            self.collector.record_gauge("system_memory_usage_percent", 45.0)
            self.collector.record_gauge("system_memory_used_bytes", 4000000000)
            self.collector.record_gauge("system_memory_available_bytes", 4000000000)
            self.collector.record_gauge("system_disk_usage_percent", 60.0)
            self.collector.record_gauge("process_memory_rss_bytes", 500000000)
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.collector.record_gauge("system_cpu_usage_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.collector.record_gauge("system_memory_usage_percent", memory.percent)
        self.collector.record_gauge("system_memory_used_bytes", memory.used)
        self.collector.record_gauge("system_memory_available_bytes", memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.collector.record_gauge("system_disk_usage_percent", 
                                   (disk.used / disk.total) * 100)
        
        # Network I/O
        network = psutil.net_io_counters()
        self.collector.record_gauge("system_network_bytes_sent", network.bytes_sent)
        self.collector.record_gauge("system_network_bytes_recv", network.bytes_recv)
        
        # Process-specific metrics
        process = psutil.Process()
        proc_memory = process.memory_info()
        self.collector.record_gauge("process_memory_rss_bytes", proc_memory.rss)
        self.collector.record_gauge("process_memory_vms_bytes", proc_memory.vms)
        self.collector.record_gauge("process_cpu_percent", process.cpu_percent())
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_cached = torch.cuda.memory_reserved(i)
                    
                    labels = {"device": str(i)}
                    self.collector.record_gauge("gpu_memory_allocated_bytes", 
                                               memory_allocated, labels)
                    self.collector.record_gauge("gpu_memory_cached_bytes", 
                                               memory_cached, labels)
                    
                    # GPU utilization (mock for now)
                    self.collector.record_gauge("gpu_utilization_percent", 0.0, labels)
        
        except Exception as e:
            logging.debug(f"GPU metrics collection failed: {e}")


class HealthChecker:
    """
    Health check system.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
    
    def register_check(self, name: str, check_fn: Callable[[], HealthCheckResult]):
        """Register a health check."""
        self.checks[name] = check_fn
    
    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name in self.checks:
            try:
                return self.checks[name]()
            except Exception as e:
                return HealthCheckResult(
                    name=name,
                    healthy=False,
                    message=f"Health check failed: {str(e)}"
                )
        return None
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        for name in self.checks:
            result = self.run_check(name)
            if result:
                results.append(result)
        return results
    
    def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health."""
        all_results = self.run_all_checks()
        
        if not all_results:
            return HealthCheckResult(
                name="overall",
                healthy=True,
                message="No health checks configured"
            )
        
        unhealthy_checks = [r for r in all_results if not r.healthy]
        
        if not unhealthy_checks:
            return HealthCheckResult(
                name="overall",
                healthy=True,
                message=f"All {len(all_results)} health checks passing",
                details={"total_checks": len(all_results), "failed_checks": 0}
            )
        else:
            return HealthCheckResult(
                name="overall",
                healthy=False,
                message=f"{len(unhealthy_checks)}/{len(all_results)} health checks failing",
                details={
                    "total_checks": len(all_results),
                    "failed_checks": len(unhealthy_checks),
                    "failed_check_names": [r.name for r in unhealthy_checks]
                }
            )


class AlertManager:
    """
    Alert management system.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._running = False
        self._thread = None
    
    def add_alert(self, alert: Alert):
        """Add an alert rule."""
        self.alerts[alert.name] = alert
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def start(self, check_interval: float = 10.0):
        """Start alert monitoring."""
        if not self._running:
            self._running = True
            self._check_interval = check_interval
            self._thread = threading.Thread(target=self._alert_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        """Stop alert monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _alert_loop(self):
        """Main alert checking loop."""
        while self._running:
            try:
                self._check_alerts()
                time.sleep(self._check_interval)
            except Exception as e:
                logging.error(f"Alert checking error: {e}")
                time.sleep(self._check_interval)
    
    def _check_alerts(self):
        """Check all alert conditions."""
        for alert in self.alerts.values():
            try:
                current_value = self.collector.get_latest(alert.metric)
                if current_value is None:
                    continue
                
                should_trigger = self._evaluate_condition(
                    current_value.value, alert.threshold, alert.comparison
                )
                
                if should_trigger and not alert.active:
                    # Trigger alert
                    alert.active = True
                    alert.triggered_at = datetime.now()
                    alert.resolved_at = None
                    
                    logging.warning(f"Alert triggered: {alert.name} - {alert.message}")
                    
                    # Notify handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logging.error(f"Alert handler error: {e}")
                
                elif not should_trigger and alert.active:
                    # Resolve alert
                    alert.active = False
                    alert.resolved_at = datetime.now()
                    
                    logging.info(f"Alert resolved: {alert.name}")
                    
                    # Could notify handlers about resolution too
            
            except Exception as e:
                logging.error(f"Error checking alert {alert.name}: {e}")
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition."""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "eq":
            return abs(value - threshold) < 0.001  # Floating point comparison
        else:
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [alert for alert in self.alerts.values() if alert.active]


class MonitoringSystem:
    """
    Comprehensive monitoring system combining all monitoring components.
    """
    
    def __init__(
        self,
        resource_update_interval: float = 5.0,
        alert_check_interval: float = 10.0,
        metrics_history_size: int = 1000
    ):
        """
        Initialize monitoring system.
        
        Args:
            resource_update_interval: How often to collect resource metrics (seconds)
            alert_check_interval: How often to check alert conditions (seconds)
            metrics_history_size: Maximum number of metric values to store
        """
        # Core components
        self.collector = MetricsCollector(max_history=metrics_history_size)
        self.performance = PerformanceMonitor(self.collector)
        self.resources = ResourceMonitor(self.collector, resource_update_interval)
        self.health = HealthChecker()
        self.alerts = AlertManager(self.collector)
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Alert check interval
        self.alert_check_interval = alert_check_interval
    
    def start(self):
        """Start all monitoring components."""
        logging.info("Starting monitoring system...")
        
        self.resources.start()
        self.alerts.start(self.alert_check_interval)
        
        logging.info("Monitoring system started")
    
    def stop(self):
        """Stop all monitoring components."""
        logging.info("Stopping monitoring system...")
        
        self.resources.stop()
        self.alerts.stop()
        
        logging.info("Monitoring system stopped")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                     labels: Dict[str, str] = None):
        """Record a metric."""
        if metric_type == MetricType.COUNTER:
            self.collector.record_counter(name, value, labels)
        elif metric_type == MetricType.GAUGE:
            self.collector.record_gauge(name, value, labels)
        elif metric_type == MetricType.HISTOGRAM:
            self.collector.record_histogram(name, value, labels)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_results = self.health.run_all_checks()
        overall_health = self.health.get_overall_health()
        active_alerts = self.alerts.get_active_alerts()
        active_operations = self.performance.get_active_operations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": overall_health.healthy,
            "health_checks": [
                {
                    "name": hr.name,
                    "healthy": hr.healthy,
                    "message": hr.message,
                    "details": hr.details
                }
                for hr in health_results
            ],
            "active_alerts": [
                {
                    "name": alert.name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None
                }
                for alert in active_alerts
            ],
            "active_operations": active_operations,
            "metrics": self.collector.get_all_metrics()
        }
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        def check_memory():
            if psutil is None:
                return HealthCheckResult(
                    name="memory",
                    healthy=True,
                    message="Memory check unavailable (psutil not installed)",
                    details={"usage_percent": 0}
                )
            
            memory = psutil.virtual_memory()
            healthy = memory.percent < 90.0
            return HealthCheckResult(
                name="memory",
                healthy=healthy,
                message=f"Memory usage: {memory.percent:.1f}%",
                details={"usage_percent": memory.percent}
            )
        
        def check_disk():
            if psutil is None:
                return HealthCheckResult(
                    name="disk",
                    healthy=True,
                    message="Disk check unavailable (psutil not installed)",
                    details={"usage_percent": 0}
                )
            
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            healthy = usage_percent < 90.0
            return HealthCheckResult(
                name="disk",
                healthy=healthy,
                message=f"Disk usage: {usage_percent:.1f}%",
                details={"usage_percent": usage_percent}
            )
        
        def check_gpu():
            try:
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    return HealthCheckResult(
                        name="gpu",
                        healthy=True,
                        message=f"GPU available: {device_count} device(s)",
                        details={"device_count": device_count}
                    )
                else:
                    return HealthCheckResult(
                        name="gpu",
                        healthy=True,
                        message="GPU not available (CPU mode)",
                        details={"device_count": 0}
                    )
            except Exception as e:
                return HealthCheckResult(
                    name="gpu",
                    healthy=False,
                    message=f"GPU check failed: {str(e)}"
                )
        
        self.health.register_check("memory", check_memory)
        self.health.register_check("disk", check_disk)
        self.health.register_check("gpu", check_gpu)
    
    def _setup_default_alerts(self):
        """Setup default alerts."""
        
        # High memory usage alert
        memory_alert = Alert(
            name="high_memory_usage",
            level=AlertLevel.WARNING,
            message="Memory usage is high",
            metric="system_memory_usage_percent",
            threshold=85.0,
            comparison="gt"
        )
        
        # High CPU usage alert
        cpu_alert = Alert(
            name="high_cpu_usage",
            level=AlertLevel.WARNING,
            message="CPU usage is high",
            metric="system_cpu_usage_percent",
            threshold=90.0,
            comparison="gt"
        )
        
        # High error rate alert
        error_alert = Alert(
            name="high_error_rate",
            level=AlertLevel.CRITICAL,
            message="Error rate is high",
            metric="operation_errors_total",
            threshold=10.0,
            comparison="gt"
        )
        
        self.alerts.add_alert(memory_alert)
        self.alerts.add_alert(cpu_alert)
        self.alerts.add_alert(error_alert)


# Performance monitoring decorators
def monitor_performance(operation_name: str, monitoring_system: Optional[MonitoringSystem] = None):
    """
    Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation for metrics
        monitoring_system: Monitoring system instance (optional)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            system = monitoring_system or _get_global_monitoring_system()
            
            operation_id = system.performance.start_operation(operation_name)
            
            try:
                result = func(*args, **kwargs)
                system.performance.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                system.performance.end_operation(operation_id, success=False)
                raise e
        
        return wrapper
    return decorator


def time_operation(metric_name: str, monitoring_system: Optional[MonitoringSystem] = None):
    """
    Decorator to time an operation and record as histogram.
    
    Args:
        metric_name: Name of the metric to record
        monitoring_system: Monitoring system instance (optional)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            system = monitoring_system or _get_global_monitoring_system()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                system.record_metric(metric_name, duration, MetricType.HISTOGRAM)
                return result
            except Exception as e:
                duration = time.time() - start_time
                system.record_metric(f"{metric_name}_error", duration, MetricType.HISTOGRAM)
                raise e
        
        return wrapper
    return decorator


# Global monitoring system
_global_monitoring_system = None


def set_global_monitoring_system(system: MonitoringSystem):
    """Set the global monitoring system."""
    global _global_monitoring_system
    _global_monitoring_system = system


def _get_global_monitoring_system() -> MonitoringSystem:
    """Get or create global monitoring system."""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = MonitoringSystem()
    return _global_monitoring_system


def get_global_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system."""
    return _get_global_monitoring_system()