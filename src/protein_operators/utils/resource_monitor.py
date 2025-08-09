"""
Resource monitoring and auto-scaling for protein design operations.

This module provides comprehensive system resource monitoring, automatic
scaling decisions, and workload management for optimal performance.
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, NamedTuple, Tuple
from dataclasses import dataclass, asdict
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
import json
import os

# Configure logging
logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False


class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class ScalingAction(Enum):
    """Possible scaling actions."""
    SCALE_UP = auto()
    SCALE_DOWN = auto()
    MAINTAIN = auto()
    THROTTLE = auto()


class ResourceThreshold(NamedTuple):
    """Resource threshold definition."""
    low: float
    high: float
    critical: float


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    
    # Network metrics (if available)
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    
    # GPU metrics (if available)
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    
    # Process-specific metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_thread_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def is_under_pressure(self) -> bool:
        """Check if system is under resource pressure."""
        return (
            self.cpu_percent > 80 or
            self.memory_percent > 85 or
            self.disk_usage_percent > 90
        )
    
    @property
    def pressure_score(self) -> float:
        """Calculate overall resource pressure score (0-1)."""
        scores = [
            self.cpu_percent / 100,
            self.memory_percent / 100,
            self.disk_usage_percent / 100
        ]
        
        if self.gpu_utilization > 0:
            scores.append(self.gpu_utilization / 100)
        
        return max(scores)


@dataclass
class ScalingEvent:
    """Scaling decision event."""
    timestamp: float
    resource_type: ResourceType
    current_utilization: float
    threshold_breached: str
    action_taken: ScalingAction
    parameters_changed: Dict[str, Any]
    reason: str


class BaseResourceCollector(ABC):
    """Abstract base class for resource collectors."""
    
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect resource metrics."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this collector can gather metrics."""
        pass


class SystemResourceCollector(BaseResourceCollector):
    """Collects system-wide resource metrics."""
    
    def __init__(self):
        """Initialize system resource collector."""
        self.process = None
        if HAS_PSUTIL:
            try:
                self.process = psutil.Process(os.getpid())
            except Exception as e:
                logger.warning(f"Could not initialize process monitoring: {e}")
    
    def is_available(self) -> bool:
        """Check if psutil is available."""
        return HAS_PSUTIL
    
    def collect(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        if not HAS_PSUTIL:
            return self._get_fallback_metrics()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
            }
            
            # Process-specific metrics
            if self.process:
                try:
                    proc_cpu = self.process.cpu_percent()
                    proc_memory = self.process.memory_info()
                    
                    metrics.update({
                        "process_cpu_percent": proc_cpu,
                        "process_memory_mb": proc_memory.rss / (1024**2),
                        "process_thread_count": self.process.num_threads()
                    })
                except Exception as e:
                    logger.debug(f"Could not collect process metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return self._get_fallback_metrics()
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Get basic metrics without psutil."""
        return {
            "cpu_percent": 0.0,
            "cpu_count": os.cpu_count() or 1,
            "memory_total_gb": 8.0,  # Assume 8GB
            "memory_available_gb": 4.0,  # Assume 4GB available
            "memory_percent": 50.0,
            "disk_total_gb": 100.0,
            "disk_free_gb": 50.0,
            "disk_usage_percent": 50.0,
            "network_bytes_sent": 0,
            "network_bytes_recv": 0,
            "process_cpu_percent": 0.0,
            "process_memory_mb": 100.0,
            "process_thread_count": 1
        }


class GPUResourceCollector(BaseResourceCollector):
    """Collects GPU resource metrics."""
    
    def __init__(self):
        """Initialize GPU resource collector."""
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            # Try importing GPU libraries
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Check if GPU metrics can be collected."""
        return self.gpu_available
    
    def collect(self) -> Dict[str, Any]:
        """Collect GPU metrics."""
        if not self.gpu_available:
            return {}
        
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_metrics = []
                
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        util = float(parts[0])
                        mem_used = float(parts[1])
                        mem_total = float(parts[2])
                        mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        
                        gpu_metrics.append({
                            "utilization": util,
                            "memory_used_mb": mem_used,
                            "memory_total_mb": mem_total,
                            "memory_percent": mem_percent
                        })
                
                return {
                    "gpu_count": len(gpu_metrics),
                    "gpu_metrics": gpu_metrics,
                    "gpu_utilization": max((g["utilization"] for g in gpu_metrics), default=0),
                    "gpu_memory_percent": max((g["memory_percent"] for g in gpu_metrics), default=0)
                }
        
        except Exception as e:
            logger.debug(f"Could not collect GPU metrics: {e}")
        
        return {}


class ResourceMonitor:
    """
    Comprehensive resource monitoring system.
    
    Monitors system resources and provides alerts when thresholds are exceeded.
    """
    
    def __init__(
        self,
        collection_interval: float = 5.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True
    ):
        """
        Initialize resource monitor.
        
        Args:
            collection_interval: How often to collect metrics (seconds)
            history_size: Number of metrics to keep in history
            enable_gpu_monitoring: Whether to monitor GPU resources
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Resource collectors
        self.collectors: List[BaseResourceCollector] = []
        self.collectors.append(SystemResourceCollector())
        
        if enable_gpu_monitoring:
            gpu_collector = GPUResourceCollector()
            if gpu_collector.is_available():
                self.collectors.append(gpu_collector)
                logger.info("GPU monitoring enabled")
            else:
                logger.info("GPU monitoring not available")
        
        # Data storage
        self.metrics_history: List[ResourceMetrics] = []
        self.current_metrics: Optional[ResourceMetrics] = None
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.metrics_lock = threading.RLock()
        
        # Thresholds
        self.thresholds = {
            ResourceType.CPU: ResourceThreshold(low=20, high=70, critical=90),
            ResourceType.MEMORY: ResourceThreshold(low=30, high=80, critical=95),
            ResourceType.DISK: ResourceThreshold(low=40, high=85, critical=98),
            ResourceType.GPU: ResourceThreshold(low=20, high=75, critical=95)
        }
        
        # Callbacks
        self.threshold_callbacks: Dict[ResourceType, List[Callable]] = {
            resource_type: [] for resource_type in ResourceType
        }
    
    def start(self) -> None:
        """Start resource monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Resource monitoring already running")
            return
        
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self) -> None:
        """Stop resource monitoring."""
        if self.monitoring_thread:
            self.stop_event.set()
            self.monitoring_thread.join(timeout=10)
            if self.monitoring_thread.is_alive():
                logger.warning("Resource monitoring thread did not stop cleanly")
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect metrics from all collectors
                combined_metrics = {}
                for collector in self.collectors:
                    if collector.is_available():
                        metrics = collector.collect()
                        combined_metrics.update(metrics)
                
                # Create ResourceMetrics object
                current_time = time.time()
                metrics = ResourceMetrics(
                    timestamp=current_time,
                    cpu_percent=combined_metrics.get('cpu_percent', 0.0),
                    memory_percent=combined_metrics.get('memory_percent', 0.0),
                    memory_available_gb=combined_metrics.get('memory_available_gb', 0.0),
                    disk_usage_percent=combined_metrics.get('disk_usage_percent', 0.0),
                    disk_free_gb=combined_metrics.get('disk_free_gb', 0.0),
                    network_bytes_sent=combined_metrics.get('network_bytes_sent', 0),
                    network_bytes_recv=combined_metrics.get('network_bytes_recv', 0),
                    gpu_utilization=combined_metrics.get('gpu_utilization', 0.0),
                    gpu_memory_percent=combined_metrics.get('gpu_memory_percent', 0.0),
                    process_cpu_percent=combined_metrics.get('process_cpu_percent', 0.0),
                    process_memory_mb=combined_metrics.get('process_memory_mb', 0.0),
                    process_thread_count=combined_metrics.get('process_thread_count', 0)
                )
                
                # Store metrics
                with self.metrics_lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    
                    # Maintain history size limit
                    if len(self.metrics_history) > self.history_size:
                        self.metrics_history = self.metrics_history[-self.history_size:]
                
                # Check thresholds and trigger callbacks
                self._check_thresholds(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next collection interval
            self.stop_event.wait(self.collection_interval)
    
    def _check_thresholds(self, metrics: ResourceMetrics) -> None:
        """Check resource thresholds and trigger callbacks."""
        threshold_checks = [
            (ResourceType.CPU, metrics.cpu_percent, self.thresholds[ResourceType.CPU]),
            (ResourceType.MEMORY, metrics.memory_percent, self.thresholds[ResourceType.MEMORY]),
            (ResourceType.DISK, metrics.disk_usage_percent, self.thresholds[ResourceType.DISK]),
        ]
        
        if metrics.gpu_utilization > 0:
            threshold_checks.append(
                (ResourceType.GPU, metrics.gpu_utilization, self.thresholds[ResourceType.GPU])
            )
        
        for resource_type, current_value, threshold in threshold_checks:
            level_breached = None
            
            if current_value >= threshold.critical:
                level_breached = "critical"
            elif current_value >= threshold.high:
                level_breached = "high"
            elif current_value <= threshold.low:
                level_breached = "low"
            
            if level_breached:
                # Trigger callbacks
                for callback in self.threshold_callbacks[resource_type]:
                    try:
                        callback(resource_type, current_value, level_breached, metrics)
                    except Exception as e:
                        logger.error(f"Error in threshold callback: {e}")
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics."""
        with self.metrics_lock:
            return self.current_metrics
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[ResourceMetrics]:
        """Get metrics history."""
        with self.metrics_lock:
            history = self.metrics_history.copy()
            if limit:
                history = history[-limit:]
            return history
    
    def get_average_metrics(self, duration_minutes: float = 5.0) -> Optional[ResourceMetrics]:
        """Get average metrics over specified duration."""
        with self.metrics_lock:
            current_time = time.time()
            cutoff_time = current_time - (duration_minutes * 60)
            
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            # Calculate averages
            count = len(recent_metrics)
            return ResourceMetrics(
                timestamp=current_time,
                cpu_percent=sum(m.cpu_percent for m in recent_metrics) / count,
                memory_percent=sum(m.memory_percent for m in recent_metrics) / count,
                memory_available_gb=sum(m.memory_available_gb for m in recent_metrics) / count,
                disk_usage_percent=sum(m.disk_usage_percent for m in recent_metrics) / count,
                disk_free_gb=sum(m.disk_free_gb for m in recent_metrics) / count,
                gpu_utilization=sum(m.gpu_utilization for m in recent_metrics) / count,
                gpu_memory_percent=sum(m.gpu_memory_percent for m in recent_metrics) / count,
                process_cpu_percent=sum(m.process_cpu_percent for m in recent_metrics) / count,
                process_memory_mb=sum(m.process_memory_mb for m in recent_metrics) / count,
                process_thread_count=int(sum(m.process_thread_count for m in recent_metrics) / count)
            )
    
    def register_threshold_callback(
        self,
        resource_type: ResourceType,
        callback: Callable[[ResourceType, float, str, ResourceMetrics], None]
    ) -> None:
        """Register a callback for threshold events."""
        self.threshold_callbacks[resource_type].append(callback)
    
    def set_thresholds(self, resource_type: ResourceType, thresholds: ResourceThreshold) -> None:
        """Set resource thresholds."""
        self.thresholds[resource_type] = thresholds
        logger.info(f"Updated thresholds for {resource_type.value}: {thresholds}")


class AutoScaler:
    """
    Automatic scaling system that adjusts system behavior based on resource utilization.
    
    Makes intelligent decisions about parallelism, caching, and resource allocation.
    """
    
    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        min_workers: int = 1,
        max_workers: int = 32,
        scale_up_threshold: float = 70.0,
        scale_down_threshold: float = 30.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            resource_monitor: Resource monitor instance
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Resource utilization threshold for scaling up
            scale_down_threshold: Resource utilization threshold for scaling down
        """
        self.resource_monitor = resource_monitor
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Current scaling parameters
        self.current_workers = min_workers
        self.current_cache_size = 128
        self.throttling_enabled = False
        
        # Scaling history
        self.scaling_events: List[ScalingEvent] = []
        self.lock = threading.RLock()
        
        # Register threshold callbacks
        self.resource_monitor.register_threshold_callback(
            ResourceType.CPU, self._cpu_threshold_callback
        )
        self.resource_monitor.register_threshold_callback(
            ResourceType.MEMORY, self._memory_threshold_callback
        )
    
    def _cpu_threshold_callback(
        self,
        resource_type: ResourceType,
        current_value: float,
        level: str,
        metrics: ResourceMetrics
    ) -> None:
        """Handle CPU threshold events."""
        if level == "critical" and current_value > 90:
            self._apply_scaling_decision(ScalingAction.THROTTLE, resource_type, current_value, level)
        elif level == "high" and current_value > self.scale_up_threshold:
            if not self.throttling_enabled:
                self._apply_scaling_decision(ScalingAction.SCALE_DOWN, resource_type, current_value, level)
        elif level == "low" and current_value < self.scale_down_threshold:
            if self.throttling_enabled:
                self._apply_scaling_decision(ScalingAction.SCALE_UP, resource_type, current_value, level)
    
    def _memory_threshold_callback(
        self,
        resource_type: ResourceType,
        current_value: float,
        level: str,
        metrics: ResourceMetrics
    ) -> None:
        """Handle memory threshold events."""
        if level == "critical":
            self._apply_scaling_decision(ScalingAction.THROTTLE, resource_type, current_value, level)
        elif level == "high":
            # Reduce cache size under memory pressure
            new_cache_size = max(32, self.current_cache_size // 2)
            if new_cache_size != self.current_cache_size:
                self.current_cache_size = new_cache_size
                self._record_scaling_event(
                    resource_type, current_value, level, ScalingAction.SCALE_DOWN,
                    {"cache_size": new_cache_size}, "Reduced cache size due to memory pressure"
                )
    
    def _apply_scaling_decision(
        self,
        action: ScalingAction,
        resource_type: ResourceType,
        current_value: float,
        level: str
    ) -> None:
        """Apply scaling decision."""
        with self.lock:
            changes = {}
            reason = ""
            
            if action == ScalingAction.SCALE_UP:
                if self.current_workers < self.max_workers:
                    new_workers = min(self.max_workers, self.current_workers + 2)
                    changes["workers"] = new_workers
                    self.current_workers = new_workers
                    reason = f"Increased workers to {new_workers}"
                
                if self.throttling_enabled:
                    self.throttling_enabled = False
                    changes["throttling"] = False
                    reason += "; Disabled throttling"
            
            elif action == ScalingAction.SCALE_DOWN:
                if self.current_workers > self.min_workers:
                    new_workers = max(self.min_workers, self.current_workers - 1)
                    changes["workers"] = new_workers
                    self.current_workers = new_workers
                    reason = f"Decreased workers to {new_workers}"
            
            elif action == ScalingAction.THROTTLE:
                if not self.throttling_enabled:
                    self.throttling_enabled = True
                    changes["throttling"] = True
                    reason = "Enabled throttling due to resource pressure"
                
                # Also reduce workers under extreme pressure
                if self.current_workers > self.min_workers:
                    new_workers = max(self.min_workers, self.current_workers // 2)
                    changes["workers"] = new_workers
                    self.current_workers = new_workers
                    reason += f"; Reduced workers to {new_workers}"
            
            if changes:
                self._record_scaling_event(
                    resource_type, current_value, level, action, changes, reason
                )
                logger.info(f"Auto-scaling: {reason}")
    
    def _record_scaling_event(
        self,
        resource_type: ResourceType,
        current_value: float,
        level: str,
        action: ScalingAction,
        changes: Dict[str, Any],
        reason: str
    ) -> None:
        """Record a scaling event."""
        event = ScalingEvent(
            timestamp=time.time(),
            resource_type=resource_type,
            current_utilization=current_value,
            threshold_breached=level,
            action_taken=action,
            parameters_changed=changes,
            reason=reason
        )
        
        self.scaling_events.append(event)
        
        # Keep only recent events
        if len(self.scaling_events) > 100:
            self.scaling_events = self.scaling_events[-100:]
    
    def get_current_scaling_params(self) -> Dict[str, Any]:
        """Get current scaling parameters."""
        with self.lock:
            return {
                "workers": self.current_workers,
                "cache_size": self.current_cache_size,
                "throttling_enabled": self.throttling_enabled,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers
            }
    
    def get_scaling_history(self, limit: int = 20) -> List[ScalingEvent]:
        """Get recent scaling events."""
        with self.lock:
            return self.scaling_events[-limit:] if self.scaling_events else []
    
    def force_scaling_action(
        self,
        action: ScalingAction,
        reason: str = "Manual override"
    ) -> None:
        """Manually force a scaling action."""
        current_metrics = self.resource_monitor.get_current_metrics()
        if current_metrics:
            self._apply_scaling_decision(
                action, ResourceType.CPU, current_metrics.cpu_percent, "manual"
            )
        
        logger.info(f"Forced scaling action: {action.name} - {reason}")


# Global instances
_resource_monitor = None
_auto_scaler = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
        _resource_monitor.start()
        logger.info("Started global resource monitoring")
    return _resource_monitor


def get_auto_scaler() -> AutoScaler:
    """Get the global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(get_resource_monitor())
        logger.info("Initialized global auto-scaler")
    return _auto_scaler


def initialize_monitoring(
    collection_interval: float = 5.0,
    enable_auto_scaling: bool = True,
    auto_start: bool = True
) -> Tuple[ResourceMonitor, Optional[AutoScaler]]:
    """
    Initialize resource monitoring and auto-scaling.
    
    Args:
        collection_interval: How often to collect metrics (seconds)
        enable_auto_scaling: Whether to enable auto-scaling
        auto_start: Whether to start monitoring automatically
        
    Returns:
        Tuple of (resource_monitor, auto_scaler)
    """
    global _resource_monitor, _auto_scaler
    
    # Initialize resource monitor
    _resource_monitor = ResourceMonitor(collection_interval=collection_interval)
    
    if auto_start:
        _resource_monitor.start()
    
    # Initialize auto-scaler if enabled
    if enable_auto_scaling:
        _auto_scaler = AutoScaler(_resource_monitor)
    
    logger.info(
        f"Initialized monitoring (interval: {collection_interval}s, "
        f"auto-scaling: {enable_auto_scaling}, auto-start: {auto_start})"
    )
    
    return _resource_monitor, _auto_scaler


def shutdown_monitoring() -> None:
    """Shutdown resource monitoring."""
    global _resource_monitor, _auto_scaler
    
    if _resource_monitor:
        _resource_monitor.stop()
        _resource_monitor = None
    
    _auto_scaler = None
    logger.info("Shutdown resource monitoring")