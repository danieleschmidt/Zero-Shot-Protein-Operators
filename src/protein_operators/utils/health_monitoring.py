"""
Enhanced health monitoring and system diagnostics for protein operators.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class HealthMetric:
    """Individual health metric data."""
    name: str
    value: float
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResources:
    """System resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    gpu_memory_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_gb': self.memory_available_gb,
            'disk_percent': self.disk_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_utilization': self.gpu_utilization
        }


class HealthMonitor:
    """
    Comprehensive health monitoring system for protein operators.
    
    Monitors system resources, operation performance, error rates,
    and provides intelligent health assessments.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Health metrics storage
        self.metrics: Dict[str, List[HealthMetric]] = {}
        self.current_status = HealthStatus.HEALTHY
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        
        # Resource tracking
        self.resource_history: List[SystemResources] = []
        
        # Health check functions
        self.health_checks: Dict[str, Callable[[], HealthMetric]] = {}
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health check functions."""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("gpu_health", self._check_gpu_health)
        self.register_health_check("operation_performance", self._check_operation_performance)
        self.register_health_check("error_rates", self._check_error_rates)
        self.register_health_check("memory_leaks", self._check_memory_leaks)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthMetric]) -> None:
        """
        Register a custom health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns a HealthMetric
        """
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring:
            self.logger.warning("Health monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.check_interval)
    
    def run_health_checks(self) -> Dict[str, HealthMetric]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check_func in self.health_checks.items():
            try:
                metric = check_func()
                results[check_name] = metric
                
                # Update overall status
                if metric.status.value == "critical":
                    overall_status = HealthStatus.CRITICAL
                elif metric.status.value == "warning" and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
                
                # Store metric
                with self._lock:
                    if check_name not in self.metrics:
                        self.metrics[check_name] = []
                    self.metrics[check_name].append(metric)
                    
                    # Keep only recent metrics (last 1000)
                    if len(self.metrics[check_name]) > 1000:
                        self.metrics[check_name] = self.metrics[check_name][-1000:]
                
            except Exception as e:
                self.logger.error(f"Health check '{check_name}' failed: {str(e)}")
                error_metric = HealthMetric(
                    name=check_name,
                    value=0.0,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )
                results[check_name] = error_metric
                overall_status = HealthStatus.CRITICAL
        
        self.current_status = overall_status
        return results
    
    def _check_system_resources(self) -> HealthMetric:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # GPU info
            gpu_memory_percent = None
            gpu_utilization = None
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated()
                    gpu_memory_total = torch.cuda.max_memory_allocated()
                    if gpu_memory_total > 0:
                        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                except:
                    pass
            
            # Store resource snapshot
            resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_percent=disk_percent,
                gpu_memory_percent=gpu_memory_percent,
                gpu_utilization=gpu_utilization
            )
            
            with self._lock:
                self.resource_history.append(resources)
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory_percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > 85:
                status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 90:
                status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            if gpu_memory_percent and gpu_memory_percent > 95:
                status = HealthStatus.CRITICAL
                issues.append(f"GPU memory critical: {gpu_memory_percent:.1f}%")
            
            message = "System resources normal" if not issues else "; ".join(issues)
            
            return HealthMetric(
                name="system_resources",
                value=max(cpu_percent, memory_percent, disk_percent),
                status=status,
                message=message,
                metadata=resources.to_dict()
            )
            
        except Exception as e:
            return HealthMetric(
                name="system_resources",
                value=0.0,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}"
            )
    
    def _check_gpu_health(self) -> HealthMetric:
        """Check GPU health and availability."""
        try:
            if not torch.cuda.is_available():
                return HealthMetric(
                    name="gpu_health",
                    value=0.0,
                    status=HealthStatus.WARNING,
                    message="CUDA not available - running on CPU only"
                )
            
            # Check GPU accessibility
            try:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                # Simple GPU test
                test_tensor = torch.randn(1000, 1000, device='cuda')
                result = torch.matmul(test_tensor, test_tensor.T)
                del test_tensor, result
                torch.cuda.empty_cache()
                
                return HealthMetric(
                    name="gpu_health",
                    value=100.0,
                    status=HealthStatus.HEALTHY,
                    message=f"GPU healthy: {device_name}",
                    metadata={
                        "device_count": device_count,
                        "current_device": current_device,
                        "device_name": device_name
                    }
                )
                
            except Exception as gpu_error:
                return HealthMetric(
                    name="gpu_health",
                    value=0.0,
                    status=HealthStatus.CRITICAL,
                    message=f"GPU test failed: {str(gpu_error)}"
                )
                
        except Exception as e:
            return HealthMetric(
                name="gpu_health",
                value=0.0,
                status=HealthStatus.CRITICAL,
                message=f"GPU health check failed: {str(e)}"
            )
    
    def _check_operation_performance(self) -> HealthMetric:
        """Check operation performance trends."""
        with self._lock:
            if not self.operation_times:
                return HealthMetric(
                    name="operation_performance",
                    value=100.0,
                    status=HealthStatus.HEALTHY,
                    message="No operations recorded yet"
                )
            
            # Calculate average operation times
            avg_times = {}
            for op_name, times in self.operation_times.items():
                if times:
                    avg_times[op_name] = sum(times) / len(times)
            
            # Check for performance degradation
            status = HealthStatus.HEALTHY
            issues = []
            
            for op_name, avg_time in avg_times.items():
                if avg_time > 300:  # 5 minutes
                    status = HealthStatus.CRITICAL
                    issues.append(f"{op_name}: {avg_time:.1f}s")
                elif avg_time > 60:  # 1 minute
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
                    issues.append(f"{op_name}: {avg_time:.1f}s")
            
            overall_avg = sum(avg_times.values()) / len(avg_times) if avg_times else 0
            message = "Performance normal" if not issues else f"Slow operations: {'; '.join(issues)}"
            
            return HealthMetric(
                name="operation_performance",
                value=100.0 - min(100.0, overall_avg / 10.0),  # Normalize to 0-100
                status=status,
                message=message,
                metadata={"average_times": avg_times}
            )
    
    def _check_error_rates(self) -> HealthMetric:
        """Check error rates across operations."""
        with self._lock:
            total_operations = sum(self.success_counts.values()) + sum(self.error_counts.values())
            total_errors = sum(self.error_counts.values())
            
            if total_operations == 0:
                return HealthMetric(
                    name="error_rates",
                    value=100.0,
                    status=HealthStatus.HEALTHY,
                    message="No operations recorded"
                )
            
            error_rate = (total_errors / total_operations) * 100
            
            status = HealthStatus.HEALTHY
            if error_rate > 50:
                status = HealthStatus.CRITICAL
            elif error_rate > 20:
                status = HealthStatus.WARNING
            
            message = f"Error rate: {error_rate:.1f}% ({total_errors}/{total_operations})"
            
            return HealthMetric(
                name="error_rates",
                value=100.0 - error_rate,
                status=status,
                message=message,
                metadata={
                    "total_operations": total_operations,
                    "total_errors": total_errors,
                    "error_rate": error_rate
                }
            )
    
    def _check_memory_leaks(self) -> HealthMetric:
        """Check for potential memory leaks."""
        try:
            # Check memory trend over time
            if len(self.resource_history) < 10:
                return HealthMetric(
                    name="memory_leaks",
                    value=100.0,
                    status=HealthStatus.HEALTHY,
                    message="Insufficient data for leak detection"
                )
            
            # Calculate memory trend (last 10 measurements)
            recent_memory = [r.memory_percent for r in self.resource_history[-10:]]
            memory_trend = recent_memory[-1] - recent_memory[0]
            
            status = HealthStatus.HEALTHY
            message = f"Memory trend: {memory_trend:+.1f}%"
            
            if memory_trend > 20:  # 20% increase
                status = HealthStatus.CRITICAL
                message = f"Potential memory leak detected: {memory_trend:+.1f}% increase"
            elif memory_trend > 10:  # 10% increase
                status = HealthStatus.WARNING
                message = f"Memory usage increasing: {memory_trend:+.1f}%"
            
            return HealthMetric(
                name="memory_leaks",
                value=100.0 - max(0, memory_trend * 5),  # Normalize
                status=status,
                message=message,
                metadata={"memory_trend": memory_trend}
            )
            
        except Exception as e:
            return HealthMetric(
                name="memory_leaks",
                value=0.0,
                status=HealthStatus.CRITICAL,
                message=f"Memory leak check failed: {str(e)}"
            )
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True) -> None:
        """
        Record operation performance data.
        
        Args:
            operation_name: Name of the operation
            duration: Operation duration in seconds
            success: Whether the operation succeeded
        """
        with self._lock:
            # Record timing
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            self.operation_times[operation_name].append(duration)
            
            # Keep only recent timings
            if len(self.operation_times[operation_name]) > 100:
                self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
            
            # Record success/failure
            if success:
                self.success_counts[operation_name] = self.success_counts.get(operation_name, 0) + 1
            else:
                self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary.
        
        Returns:
            Dictionary with health status and metrics
        """
        with self._lock:
            latest_metrics = {}
            for check_name, metrics in self.metrics.items():
                if metrics:
                    latest_metrics[check_name] = metrics[-1]
            
            return {
                "overall_status": self.current_status.value,
                "last_check": time.time(),
                "metrics": {name: metric.__dict__ for name, metric in latest_metrics.items()},
                "system_resources": self.resource_history[-1].to_dict() if self.resource_history else None,
                "operation_stats": {
                    "total_success": sum(self.success_counts.values()),
                    "total_errors": sum(self.error_counts.values()),
                    "operations_tracked": len(self.operation_times)
                }
            }
    
    def get_recommendations(self) -> List[str]:
        """
        Get health improvement recommendations.
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        try:
            summary = self.get_health_summary()
            
            # Check resource issues
            if summary["system_resources"]:
                resources = summary["system_resources"]
                
                if resources["memory_percent"] > 85:
                    recommendations.append("Consider increasing system memory or optimizing memory usage")
                
                if resources["cpu_percent"] > 80:
                    recommendations.append("CPU usage is high - consider load balancing or optimization")
                
                if resources["disk_percent"] > 90:
                    recommendations.append("Disk space is low - clean up temporary files or increase storage")
            
            # Check error rates
            stats = summary["operation_stats"]
            total_ops = stats["total_success"] + stats["total_errors"]
            if total_ops > 0:
                error_rate = (stats["total_errors"] / total_ops) * 100
                if error_rate > 20:
                    recommendations.append("High error rate detected - review error logs and improve error handling")
            
            # Check GPU status
            metrics = summary["metrics"]
            if "gpu_health" in metrics:
                gpu_metric = metrics["gpu_health"]
                if gpu_metric["status"] == "warning":
                    recommendations.append("GPU not available - consider enabling CUDA support for better performance")
                elif gpu_metric["status"] == "critical":
                    recommendations.append("GPU issues detected - check CUDA installation and GPU drivers")
            
            if not recommendations:
                recommendations.append("System is running optimally - no recommendations at this time")
                
        except Exception as e:
            recommendations.append(f"Unable to generate recommendations due to error: {str(e)}")
        
        return recommendations


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def start_health_monitoring() -> None:
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop global health monitoring."""
    monitor = get_health_monitor()
    monitor.stop_monitoring()


def record_operation_performance(operation_name: str, duration: float, success: bool = True) -> None:
    """Record operation performance in global monitor."""
    monitor = get_health_monitor()
    monitor.record_operation(operation_name, duration, success)