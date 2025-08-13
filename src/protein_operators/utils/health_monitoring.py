"""
Health monitoring and system diagnostics for protein operators.
"""

import time
import threading

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from enum import Enum

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Single health metric."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""

@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_percent: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    designs_completed: int = 0
    validation_errors: int = 0
    average_design_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    active_sessions: int = 0
    queue_size: int = 0
    error_rate_per_hour: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

class HealthMonitor:
    """Health monitoring system for protein operators."""
    
    def __init__(
        self,
        check_interval: int = 30,  # seconds
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.app_metrics_history: List[ApplicationMetrics] = []
        self.health_metrics: Dict[str, HealthMetric] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Health checks registry
        self.health_checks: Dict[str, Callable[[], HealthMetric]] = {}
        
        # Application counters
        self._app_counters = {
            'designs_completed': 0,
            'validation_errors': 0,
            'design_times': [],
            'active_sessions': set(),
            'errors_by_hour': {}
        }
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        
        def check_cpu_usage() -> HealthMetric:
            if not HAS_PSUTIL:
                return HealthMetric(
                    name="cpu_usage",
                    value=0.0,
                    unit="percent",
                    status=HealthStatus.UNKNOWN,
                    description="psutil not available for CPU monitoring"
                )
            
            cpu_percent = psutil.cpu_percent(interval=1)
            status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 70:
                status = HealthStatus.WARNING
            
            return HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                status=status,
                threshold_warning=70.0,
                threshold_critical=90.0,
                description="CPU utilization percentage"
            )
        
        def check_memory_usage() -> HealthMetric:
            if not HAS_PSUTIL:
                return HealthMetric(
                    name="memory_usage",
                    value=0.0,
                    unit="percent",
                    status=HealthStatus.UNKNOWN,
                    description="psutil not available for memory monitoring"
                )
            
            memory = psutil.virtual_memory()
            status = HealthStatus.HEALTHY
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
            elif memory.percent > 80:
                status = HealthStatus.WARNING
            
            return HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                status=status,
                threshold_warning=80.0,
                threshold_critical=90.0,
                description="System memory usage percentage"
            )
        
        def check_disk_space() -> HealthMetric:
            if not HAS_PSUTIL:
                return HealthMetric(
                    name="disk_usage",
                    value=0.0,
                    unit="percent",
                    status=HealthStatus.UNKNOWN,
                    description="psutil not available for disk monitoring"
                )
            
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            status = HealthStatus.HEALTHY
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
            elif disk_percent > 85:
                status = HealthStatus.WARNING
            
            return HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="percent",
                status=status,
                threshold_warning=85.0,
                threshold_critical=95.0,
                description="Disk space usage percentage"
            )
        
        def check_process_memory() -> HealthMetric:
            if not HAS_PSUTIL:
                return HealthMetric(
                    name="process_memory",
                    value=0.0,
                    unit="MB",
                    status=HealthStatus.UNKNOWN,
                    description="psutil not available for process monitoring"
                )
            
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            status = HealthStatus.HEALTHY
            if memory_mb > 8192:  # 8GB
                status = HealthStatus.CRITICAL
            elif memory_mb > 4096:  # 4GB
                status = HealthStatus.WARNING
            
            return HealthMetric(
                name="process_memory",
                value=memory_mb,
                unit="MB",
                status=status,
                threshold_warning=4096.0,
                threshold_critical=8192.0,
                description="Process memory usage in MB"
            )
        
        # Register default checks
        self.register_health_check("cpu_usage", check_cpu_usage)
        self.register_health_check("memory_usage", check_memory_usage)
        self.register_health_check("disk_usage", check_disk_space)
        self.register_health_check("process_memory", check_process_memory)
        
        # GPU health check if available
        if self.enable_gpu_monitoring:
            def check_gpu_memory() -> HealthMetric:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        status = HealthStatus.HEALTHY
                        if gpu_memory > 90:
                            status = HealthStatus.CRITICAL
                        elif gpu_memory > 80:
                            status = HealthStatus.WARNING
                        
                        return HealthMetric(
                            name="gpu_memory",
                            value=gpu_memory,
                            unit="percent",
                            status=status,
                            threshold_warning=80.0,
                            threshold_critical=90.0,
                            description="GPU memory usage percentage"
                        )
                except ImportError:
                    pass
                
                return HealthMetric(
                    name="gpu_memory",
                    value=0.0,
                    unit="percent",
                    status=HealthStatus.UNKNOWN,
                    description="GPU not available"
                )
            
            self.register_health_check("gpu_memory", check_gpu_memory)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthMetric]):
        """Register a custom health check."""
        self.health_checks[name] = check_func
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Store metrics
                with self._lock:
                    self.system_metrics_history.append(system_metrics)
                    self.app_metrics_history.append(app_metrics)
                    
                    # Limit history size
                    if len(self.system_metrics_history) > self.history_size:
                        self.system_metrics_history.pop(0)
                    if len(self.app_metrics_history) > self.history_size:
                        self.app_metrics_history.pop(0)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                # Log error but continue monitoring
                print(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        if not HAS_PSUTIL:
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                process_memory_mb=0.0,
                process_cpu_percent=0.0
            )
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        process = psutil.Process()
        
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / 1024 / 1024 / 1024,
            disk_usage_percent=(disk.used / disk.total) * 100,
            process_memory_mb=process.memory_info().rss / 1024 / 1024,
            process_cpu_percent=process.cpu_percent()
        )
        
        # GPU metrics if available
        if self.enable_gpu_monitoring:
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.gpu_memory_percent = (
                        torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    )
                    # Note: GPU utilization requires nvidia-ml-py
            except ImportError:
                pass
        
        return metrics
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        with self._lock:
            # Calculate average design time
            design_times = self._app_counters['design_times']
            avg_design_time = sum(design_times) / len(design_times) if design_times else 0.0
            
            # Calculate error rate
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            error_rate = self._app_counters['errors_by_hour'].get(current_hour, 0)
            
            return ApplicationMetrics(
                designs_completed=self._app_counters['designs_completed'],
                validation_errors=self._app_counters['validation_errors'],
                average_design_time_ms=avg_design_time,
                active_sessions=len(self._app_counters['active_sessions']),
                error_rate_per_hour=error_rate
            )
    
    def _run_health_checks(self):
        """Run all registered health checks."""
        with self._lock:
            for name, check_func in self.health_checks.items():
                try:
                    metric = check_func()
                    self.health_metrics[name] = metric
                except Exception as e:
                    # Create error metric
                    self.health_metrics[name] = HealthMetric(
                        name=name,
                        value=0.0,
                        unit="error",
                        status=HealthStatus.CRITICAL,
                        description=f"Health check failed: {str(e)}"
                    )
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            # Overall health status
            statuses = [metric.status for metric in self.health_metrics.values()]
            if HealthStatus.CRITICAL in statuses:
                overall_status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                overall_status = HealthStatus.WARNING
            elif statuses:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
            
            # Current metrics
            current_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            current_app = self.app_metrics_history[-1] if self.app_metrics_history else None
            
            return {
                "overall_status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "health_metrics": {
                    name: {
                        "value": metric.value,
                        "unit": metric.unit,
                        "status": metric.status.value,
                        "description": metric.description
                    }
                    for name, metric in self.health_metrics.items()
                },
                "system_metrics": current_system.__dict__ if current_system else None,
                "application_metrics": current_app.__dict__ if current_app else None
            }
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            # Filter recent metrics
            recent_system = [
                m for m in self.system_metrics_history
                if m.timestamp >= cutoff_time
            ]
            recent_app = [
                m for m in self.app_metrics_history
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_system:
                return {"error": "No recent metrics available"}
            
            # Calculate trends
            return {
                "time_period_hours": hours,
                "system_trends": {
                    "cpu_usage": {
                        "avg": sum(m.cpu_percent for m in recent_system) / len(recent_system),
                        "max": max(m.cpu_percent for m in recent_system),
                        "min": min(m.cpu_percent for m in recent_system)
                    },
                    "memory_usage": {
                        "avg": sum(m.memory_percent for m in recent_system) / len(recent_system),
                        "max": max(m.memory_percent for m in recent_system),
                        "min": min(m.memory_percent for m in recent_system)
                    }
                },
                "application_trends": {
                    "total_designs": sum(m.designs_completed for m in recent_app),
                    "total_errors": sum(m.validation_errors for m in recent_app),
                    "avg_design_time": (
                        sum(m.average_design_time_ms for m in recent_app) / len(recent_app)
                        if recent_app else 0
                    )
                }
            }
    
    def record_design_completion(self, duration_ms: float):
        """Record a completed protein design."""
        with self._lock:
            self._app_counters['designs_completed'] += 1
            self._app_counters['design_times'].append(duration_ms)
            
            # Keep only recent design times (last 100)
            if len(self._app_counters['design_times']) > 100:
                self._app_counters['design_times'].pop(0)
    
    def record_validation_error(self):
        """Record a validation error."""
        with self._lock:
            self._app_counters['validation_errors'] += 1
            
            # Track errors by hour
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            self._app_counters['errors_by_hour'][current_hour] = (
                self._app_counters['errors_by_hour'].get(current_hour, 0) + 1
            )
    
    def record_session_start(self, session_id: str):
        """Record the start of a user session."""
        with self._lock:
            self._app_counters['active_sessions'].add(session_id)
    
    def record_session_end(self, session_id: str):
        """Record the end of a user session."""
        with self._lock:
            self._app_counters['active_sessions'].discard(session_id)
    
    def export_metrics(self, filepath: Path, format: str = "json"):
        """Export metrics to file."""
        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "current_health": self.get_current_health(),
            "trends_24h": self.get_health_trends(24),
            "system_metrics_history": [
                m.__dict__ for m in self.system_metrics_history
            ],
            "application_metrics_history": [
                m.__dict__ for m in self.app_metrics_history
            ]
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_health_report(self) -> str:
        """Create a human-readable health report."""
        health = self.get_current_health()
        trends = self.get_health_trends(24)
        
        report = []
        report.append("=== Protein Operators Health Report ===")
        report.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append(f"Overall Status: {health['overall_status'].upper()}")
        report.append("")
        
        # Health metrics
        report.append("Health Metrics:")
        for name, metric in health['health_metrics'].items():
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌',
                'unknown': '❓'
            }.get(metric['status'], '❓')
            
            report.append(f"  {status_emoji} {name}: {metric['value']:.1f}{metric['unit']} ({metric['status']})")
        
        report.append("")
        
        # System metrics
        if health['system_metrics']:
            sm = health['system_metrics']
            report.append("Current System Metrics:")
            report.append(f"  CPU Usage: {sm['cpu_percent']:.1f}%")
            report.append(f"  Memory Usage: {sm['memory_percent']:.1f}% ({sm['memory_available_gb']:.1f}GB available)")
            report.append(f"  Disk Usage: {sm['disk_usage_percent']:.1f}%")
            report.append(f"  Process Memory: {sm['process_memory_mb']:.1f}MB")
            if sm.get('gpu_memory_percent'):
                report.append(f"  GPU Memory: {sm['gpu_memory_percent']:.1f}%")
        
        report.append("")
        
        # Application metrics
        if health['application_metrics']:
            am = health['application_metrics']
            report.append("Application Metrics:")
            report.append(f"  Designs Completed: {am['designs_completed']}")
            report.append(f"  Validation Errors: {am['validation_errors']}")
            report.append(f"  Average Design Time: {am['average_design_time_ms']:.1f}ms")
            report.append(f"  Active Sessions: {am['active_sessions']}")
        
        report.append("")
        
        # 24-hour trends
        if 'system_trends' in trends:
            st = trends['system_trends']
            report.append("24-Hour Trends:")
            report.append(f"  CPU Usage: {st['cpu_usage']['avg']:.1f}% avg, {st['cpu_usage']['max']:.1f}% max")
            report.append(f"  Memory Usage: {st['memory_usage']['avg']:.1f}% avg, {st['memory_usage']['max']:.1f}% max")
        
        return "\\n".join(report)

# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor

def start_health_monitoring():
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring()

def stop_health_monitoring():
    """Stop global health monitoring."""
    monitor = get_health_monitor()
    monitor.stop_monitoring()

def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    monitor = get_health_monitor()
    return monitor.get_current_health()