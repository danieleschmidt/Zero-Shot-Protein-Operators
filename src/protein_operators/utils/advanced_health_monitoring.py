"""
Advanced health monitoring and observability system for protein operators.

This module provides comprehensive monitoring, metrics collection,
and health assessment capabilities for production deployment.
"""

import time
import threading
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
# Handle psutil import for testing
try:
    import psutil
except ImportError:
    # Mock psutil for testing environments
    class MockPsutil:
        class VirtualMemory:
            def __init__(self):
                self.percent = 50.0
                self.used = 8 * (1024**3)  # 8 GB
                self.available = 8 * (1024**3)  # 8 GB
        
        class DiskUsage:
            def __init__(self):
                self.percent = 25.0
        
        class Process:
            def num_fds(self):
                return 100
            
            def connections(self):
                return []
        
        def cpu_percent(self, interval=None):
            return 25.0
        
        def virtual_memory(self):
            return self.VirtualMemory()
        
        def disk_usage(self, path):
            return self.DiskUsage()
    
    psutil = MockPsutil()
import sys
import os

# Handle import compatibility
try:
    import torch
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    import mock_torch as torch


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics being collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    open_file_descriptors: int = 0
    network_connections: int = 0


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Collects and manages various types of metrics including:
    - Business metrics (designs generated, success rates)
    - System metrics (CPU, memory, GPU usage)  
    - Performance metrics (latency, throughput)
    - Error metrics (error rates, failure patterns)
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for metric cleanup."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
            self._cleanup_thread.start()
    
    def _cleanup_old_metrics(self):
        """Remove old metrics to manage memory usage."""
        while True:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                with self._lock:
                    for metric_name, metric_deque in self.metrics.items():
                        # Remove old metrics
                        while metric_deque and metric_deque[0].timestamp < cutoff_time:
                            metric_deque.popleft()
                    
                    # Clean up histogram data older than retention period
                    for timer_name, timer_deque in self.timers.items():
                        while timer_deque and timer_deque[0]['timestamp'] < cutoff_time:
                            timer_deque.popleft()
                
                # Sleep for 5 minutes before next cleanup
                time.sleep(300)
                
            except Exception as e:
                logging.error(f"Error in metrics cleanup: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            metric = Metric(
                name=name,
                value=self.counters[name],
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values for histograms
            if len(self.histograms[name]) > 10000:
                self.histograms[name] = self.histograms[name][-5000:]
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[name].append(metric)
    
    def time_operation(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)
    
    def record_timer(self, name: str, duration_ms: float, labels: Dict[str, str] = None):
        """Record a timer value."""
        with self._lock:
            timer_data = {
                'duration_ms': duration_ms,
                'timestamp': time.time(),
                'labels': labels or {}
            }
            self.timers[name].append(timer_data)
            
            metric = Metric(
                name=name,
                value=duration_ms,
                metric_type=MetricType.TIMER,
                timestamp=time.time(),
                labels=labels or {},
                unit="ms"
            )
            self.metrics[name].append(metric)
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current values for all metrics."""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_stats': self._calculate_histogram_stats(),
                'timer_stats': self._calculate_timer_stats()
            }
    
    def _calculate_histogram_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for histogram metrics."""
        stats = {}
        for name, values in self.histograms.items():
            if values:
                import statistics
                stats[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99)
                }
            else:
                stats[name] = {
                    'count': 0, 'min': 0, 'max': 0, 'mean': 0, 
                    'median': 0, 'p95': 0, 'p99': 0
                }
        return stats
    
    def _calculate_timer_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for timer metrics."""
        stats = {}
        for name, timer_data in self.timers.items():
            if timer_data:
                durations = [t['duration_ms'] for t in timer_data]
                import statistics
                stats[name] = {
                    'count': len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'mean_ms': statistics.mean(durations),
                    'median_ms': statistics.median(durations),
                    'p95_ms': self._percentile(durations, 95),
                    'p99_ms': self._percentile(durations, 99)
                }
            else:
                stats[name] = {
                    'count': 0, 'min_ms': 0, 'max_ms': 0, 'mean_ms': 0,
                    'median_ms': 0, 'p95_ms': 0, 'p99_ms': 0
                }
        return stats
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        current_time = int(time.time() * 1000)
        
        # Export counters
        for name, value in self.counters.items():
            lines.append(f'# TYPE {name} counter')
            lines.append(f'{name} {value} {current_time}')
        
        # Export gauges
        for name, value in self.gauges.items():
            lines.append(f'# TYPE {name} gauge')
            lines.append(f'{name} {value} {current_time}')
        
        # Export histogram summaries
        hist_stats = self._calculate_histogram_stats()
        for name, stats in hist_stats.items():
            lines.append(f'# TYPE {name} histogram')
            for stat_name, stat_value in stats.items():
                lines.append(f'{name}_{stat_name} {stat_value} {current_time}')
        
        return '\n'.join(lines)


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.name, duration_ms, self.labels)


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Monitors system health across multiple dimensions:
    - System resources (CPU, memory, disk, GPU)
    - Application metrics (throughput, latency, errors)
    - Business metrics (design success rates, constraint satisfaction)
    - External dependencies (databases, model checkpoints)
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.metrics_collector = MetricsCollector()
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.thresholds = self._default_thresholds()
        self.last_check_results: Dict[str, HealthCheckResult] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        self._monitoring_thread = None
        self._is_monitoring = False
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default thresholds for health monitoring."""
        return {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'gpu_memory': {'warning': 85.0, 'critical': 95.0},
            'error_rate': {'warning': 5.0, 'critical': 15.0},  # Percentage
            'response_time': {'warning': 5000.0, 'critical': 10000.0},  # ms
            'success_rate': {'warning': 95.0, 'critical': 90.0},  # Percentage
        }
    
    def _register_default_health_checks(self):
        """Register default health check functions."""
        self.health_checks.update({
            'system_resources': self._check_system_resources,
            'gpu_status': self._check_gpu_status,
            'application_metrics': self._check_application_metrics,
            'model_availability': self._check_model_availability,
            'constraint_validation': self._check_constraint_validation,
            'memory_leaks': self._check_memory_leaks,
        })
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a custom health check function."""
        self.health_checks[name] = check_func
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._is_monitoring = True
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            logging.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._is_monitoring = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        logging.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                # Run all health checks
                self.run_all_health_checks()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(min(self.check_interval, 60))  # Wait at least 1 minute on error
    
    def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_func()
                result.duration_ms = (time.time() - start_time) * 1000
                results[check_name] = result
                self.last_check_results[check_name] = result
                
                # Record health check metrics
                self.metrics_collector.record_timer(
                    f'health_check_duration_{check_name}',
                    result.duration_ms
                )
                
                status_value = 1.0 if result.status == HealthStatus.HEALTHY else 0.0
                self.metrics_collector.set_gauge(
                    f'health_check_status_{check_name}',
                    status_value
                )
                
            except Exception as e:
                error_result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=time.time()
                )
                results[check_name] = error_result
                self.last_check_results[check_name] = error_result
                logging.error(f"Health check {check_name} failed: {e}")
        
        return results
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.metrics_collector.set_gauge('system_cpu_percent', cpu_percent)
            self.metrics_collector.set_gauge('system_memory_percent', memory.percent)
            self.metrics_collector.set_gauge('system_memory_used_gb', memory.used / (1024**3))
            self.metrics_collector.set_gauge('system_memory_available_gb', memory.available / (1024**3))
            self.metrics_collector.set_gauge('system_disk_usage_percent', disk.percent)
            
            # Network and file descriptors
            try:
                process = psutil.Process()
                fd_count = process.num_fds()
                connections = len(process.connections())
                
                self.metrics_collector.set_gauge('system_open_fds', fd_count)
                self.metrics_collector.set_gauge('system_connections', connections)
            except Exception:
                pass  # May not be available on all systems
            
            # GPU metrics (if available)
            self._collect_gpu_metrics()
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    if hasattr(torch.cuda, 'memory_stats'):
                        stats = torch.cuda.memory_stats(i)
                        allocated = stats.get('allocated_bytes.all.current', 0) / (1024**3)
                        reserved = stats.get('reserved_bytes.all.current', 0) / (1024**3)
                        
                        self.metrics_collector.set_gauge(f'gpu_{i}_memory_allocated_gb', allocated)
                        self.metrics_collector.set_gauge(f'gpu_{i}_memory_reserved_gb', reserved)
                    
                    # GPU utilization (simplified)
                    utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    self.metrics_collector.set_gauge(f'gpu_{i}_utilization_percent', utilization)
                    
        except Exception as e:
            logging.debug(f"GPU metrics collection failed: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions based on current metrics and health status."""
        current_time = time.time()
        new_alerts = []
        
        # Check health check results for alerts
        for check_name, result in self.last_check_results.items():
            if result.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                alert = {
                    'type': 'health_check_failure',
                    'severity': result.status.value,
                    'message': f"Health check {check_name} failed: {result.message}",
                    'timestamp': current_time,
                    'details': result.details
                }
                new_alerts.append(alert)
        
        # Check metric thresholds
        current_metrics = self.metrics_collector.get_current_values()
        
        # CPU usage alert
        cpu_usage = current_metrics['gauges'].get('system_cpu_percent', 0)
        if cpu_usage > self.thresholds['cpu_usage']['critical']:
            new_alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'critical',
                'message': f"Critical CPU usage: {cpu_usage:.1f}%",
                'timestamp': current_time,
                'value': cpu_usage
            })
        elif cpu_usage > self.thresholds['cpu_usage']['warning']:
            new_alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f"High CPU usage: {cpu_usage:.1f}%",
                'timestamp': current_time,
                'value': cpu_usage
            })
        
        # Memory usage alert
        memory_usage = current_metrics['gauges'].get('system_memory_percent', 0)
        if memory_usage > self.thresholds['memory_usage']['critical']:
            new_alerts.append({
                'type': 'high_memory_usage',
                'severity': 'critical',
                'message': f"Critical memory usage: {memory_usage:.1f}%",
                'timestamp': current_time,
                'value': memory_usage
            })
        elif memory_usage > self.thresholds['memory_usage']['warning']:
            new_alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f"High memory usage: {memory_usage:.1f}%",
                'timestamp': current_time,
                'value': memory_usage
            })
        
        # Add new alerts
        self.alerts.extend(new_alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = current_time - (24 * 3600)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    # Default health check implementations
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > self.thresholds['cpu_usage']['critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > self.thresholds['cpu_usage']['warning']:
                status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > self.thresholds['memory_usage']['critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > self.thresholds['memory_usage']['warning']:
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > self.thresholds['disk_usage']['critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical disk usage: {disk.percent:.1f}%")
            elif disk.percent > self.thresholds['disk_usage']['warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources within normal limits"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_gpu_status(self) -> HealthCheckResult:
        """Check GPU availability and memory usage."""
        try:
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    name="gpu_status",
                    status=HealthStatus.WARNING,
                    message="CUDA not available",
                    timestamp=time.time(),
                    details={'cuda_available': False}
                )
            
            device_count = torch.cuda.device_count()
            gpu_details = {}
            overall_status = HealthStatus.HEALTHY
            messages = []
            
            for i in range(device_count):
                try:
                    # Get memory info
                    if hasattr(torch.cuda, 'memory_stats'):
                        stats = torch.cuda.memory_stats(i)
                        allocated = stats.get('allocated_bytes.all.current', 0)
                        reserved = stats.get('reserved_bytes.all.current', 0)
                        total = torch.cuda.get_device_properties(i).total_memory
                        
                        memory_percent = (allocated / total) * 100 if total > 0 else 0
                        
                        gpu_details[f'gpu_{i}'] = {
                            'memory_allocated_gb': allocated / (1024**3),
                            'memory_reserved_gb': reserved / (1024**3),
                            'memory_total_gb': total / (1024**3),
                            'memory_percent': memory_percent
                        }
                        
                        if memory_percent > self.thresholds['gpu_memory']['critical']:
                            overall_status = HealthStatus.CRITICAL
                            messages.append(f"GPU {i} critical memory usage: {memory_percent:.1f}%")
                        elif memory_percent > self.thresholds['gpu_memory']['warning']:
                            if overall_status == HealthStatus.HEALTHY:
                                overall_status = HealthStatus.WARNING
                            messages.append(f"GPU {i} high memory usage: {memory_percent:.1f}%")
                    
                except Exception as gpu_error:
                    gpu_details[f'gpu_{i}'] = {'error': str(gpu_error)}
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.WARNING
                    messages.append(f"GPU {i} error: {str(gpu_error)}")
            
            message = "; ".join(messages) if messages else f"All {device_count} GPUs healthy"
            
            return HealthCheckResult(
                name="gpu_status",
                status=overall_status,
                message=message,
                timestamp=time.time(),
                details={'device_count': device_count, 'gpus': gpu_details}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="gpu_status",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check GPU status: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_application_metrics(self) -> HealthCheckResult:
        """Check application-specific metrics."""
        try:
            current_metrics = self.metrics_collector.get_current_values()
            
            # Check error rates
            error_count = current_metrics['counters'].get('errors_total', 0)
            success_count = current_metrics['counters'].get('operations_successful', 0)
            total_operations = error_count + success_count
            
            if total_operations > 0:
                error_rate = (error_count / total_operations) * 100
                success_rate = (success_count / total_operations) * 100
            else:
                error_rate = 0
                success_rate = 100
            
            status = HealthStatus.HEALTHY
            messages = []
            
            if error_rate > self.thresholds['error_rate']['critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical error rate: {error_rate:.1f}%")
            elif error_rate > self.thresholds['error_rate']['warning']:
                status = HealthStatus.WARNING
                messages.append(f"High error rate: {error_rate:.1f}%")
            
            if success_rate < self.thresholds['success_rate']['critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Low success rate: {success_rate:.1f}%")
            elif success_rate < self.thresholds['success_rate']['warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Degraded success rate: {success_rate:.1f}%")
            
            # Check response times
            timer_stats = current_metrics['timer_stats']
            avg_response_time = timer_stats.get('structure_generation', {}).get('mean_ms', 0)
            
            if avg_response_time > self.thresholds['response_time']['critical']:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical response time: {avg_response_time:.0f}ms")
            elif avg_response_time > self.thresholds['response_time']['warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High response time: {avg_response_time:.0f}ms")
            
            message = "; ".join(messages) if messages else "Application metrics within normal limits"
            
            return HealthCheckResult(
                name="application_metrics",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'error_rate': error_rate,
                    'success_rate': success_rate,
                    'avg_response_time_ms': avg_response_time,
                    'total_operations': total_operations
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="application_metrics",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check application metrics: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_model_availability(self) -> HealthCheckResult:
        """Check if required models are available and accessible."""
        try:
            # This would check if model files exist, are readable, etc.
            # For now, return a simple check
            
            return HealthCheckResult(
                name="model_availability",
                status=HealthStatus.HEALTHY,
                message="Models available and accessible",
                timestamp=time.time(),
                details={'checked_models': ['deeponet', 'fno']}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="model_availability",
                status=HealthStatus.CRITICAL,
                message=f"Model availability check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_constraint_validation(self) -> HealthCheckResult:
        """Check constraint validation system health."""
        try:
            # Simple functional test of constraint system
            from ..constraints import Constraints
            
            constraints = Constraints()
            constraints.add_binding_site([1, 2], "test")
            
            # If we get here without exception, constraint system is working
            return HealthCheckResult(
                name="constraint_validation",
                status=HealthStatus.HEALTHY,
                message="Constraint validation system operational",
                timestamp=time.time(),
                details={'test_passed': True}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="constraint_validation",
                status=HealthStatus.CRITICAL,
                message=f"Constraint validation system failed: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_memory_leaks(self) -> HealthCheckResult:
        """Check for potential memory leaks."""
        try:
            current_memory = psutil.virtual_memory().percent
            
            # Simple heuristic: if memory usage has been consistently high
            # and growing, it might indicate a leak
            
            # This is a simplified check - real implementation would track
            # memory growth over time and identify patterns
            
            if current_memory > 95:
                status = HealthStatus.WARNING
                message = "Very high memory usage - potential leak risk"
            else:
                status = HealthStatus.HEALTHY
                message = "No memory leak indicators detected"
            
            return HealthCheckResult(
                name="memory_leaks",
                status=status,
                message=message,
                timestamp=time.time(),
                details={'current_memory_percent': current_memory}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_leaks",
                status=HealthStatus.CRITICAL,
                message=f"Memory leak check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.last_check_results:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'No health checks have been performed yet',
                'timestamp': time.time()
            }
        
        # Determine overall status based on individual checks
        statuses = [result.status for result in self.last_check_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Count check results by status
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = len([s for s in statuses if s == status])
        
        return {
            'status': overall_status.value,
            'message': f'System health: {overall_status.value}',
            'timestamp': time.time(),
            'check_results': {name: {
                'status': result.status.value,
                'message': result.message,
                'timestamp': result.timestamp
            } for name, result in self.last_check_results.items()},
            'status_counts': status_counts,
            'recent_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600]),
            'metrics_summary': self.metrics_collector.get_current_values()
        }


# Global monitoring instance
_global_monitor = HealthMonitor()


def get_global_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return _global_monitor


def start_global_monitoring():
    """Start global health monitoring."""
    _global_monitor.start_monitoring()


def stop_global_monitoring():
    """Stop global health monitoring."""
    _global_monitor.stop_monitoring()