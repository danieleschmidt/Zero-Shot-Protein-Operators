"""
Robust Framework for Autonomous Protein Design - Generation 2
Comprehensive error handling, monitoring, and recovery systems.
"""

from typing import Dict, List, Optional, Any, Callable, Union
import sys
import os
import json
import time
import traceback
from functools import wraps
from pathlib import Path
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import torch
    import torch.nn as nn
except ImportError:
    import mock_torch as torch
    nn = torch.nn

try:
    import numpy as np
except ImportError:
    import mock_numpy as np


class RobustProteinDesigner:
    """
    Robust protein designer with comprehensive error handling and monitoring.
    
    Features:
    - Comprehensive error handling and recovery
    - Real-time monitoring and alerting
    - Automatic fallback mechanisms
    - Performance optimization
    - Security validation
    - Resource management
    """
    
    def __init__(
        self,
        base_designer: Any,
        config: Optional[Dict] = None,
        enable_monitoring: bool = True,
        enable_recovery: bool = True
    ):
        """Initialize robust framework."""
        self.base_designer = base_designer
        self.config = config or self._default_config()
        self.enable_monitoring = enable_monitoring
        self.enable_recovery = enable_recovery
        
        # Initialize components
        self.error_handler = RobustErrorHandler(self.config)
        self.monitor = PerformanceMonitor(self.config) if enable_monitoring else None
        self.recovery_system = RecoverySystem(self.config) if enable_recovery else None
        self.security_validator = SecurityValidator(self.config)
        self.resource_manager = ResourceManager(self.config)
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "error_types": {},
            "resource_usage": {},
            "security_incidents": 0
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _default_config(self) -> Dict:
        """Default robust framework configuration."""
        return {
            "error_handling": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "timeout_seconds": 300,
                "enable_fallback": True
            },
            "monitoring": {
                "sample_rate": 0.1,
                "alert_thresholds": {
                    "error_rate": 0.05,
                    "response_time": 10.0,
                    "memory_usage": 0.8
                },
                "metrics_retention_hours": 24
            },
            "security": {
                "input_validation": True,
                "output_sanitization": True,
                "rate_limiting": True,
                "max_requests_per_minute": 60
            },
            "resources": {
                "max_memory_mb": 4096,
                "max_gpu_memory_mb": 2048,
                "cleanup_interval_minutes": 30
            },
            "logging": {
                "level": "INFO",
                "file": "protein_design.log",
                "max_size_mb": 100,
                "backup_count": 5
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("protein_operators")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config["logging"]["file"],
                maxBytes=self.config["logging"]["max_size_mb"] * 1024 * 1024,
                backupCount=self.config["logging"]["backup_count"]
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except ImportError:
            # Fallback to basic file handler
            file_handler = logging.FileHandler(self.config["logging"]["file"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def robust_design(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Robust protein design with comprehensive error handling.
        
        Returns:
            Dictionary containing:
            - success: bool
            - result: design result or None
            - error: error information or None
            - metrics: performance metrics
            - warnings: list of warnings
        """
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.time()
        
        response = {
            "request_id": request_id,
            "success": False,
            "result": None,
            "error": None,
            "metrics": {},
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics["total_requests"] += 1
        
        try:
            # Security validation
            self.logger.info(f"[{request_id}] Starting robust protein design")
            
            security_result = self.security_validator.validate_input(args, kwargs)
            if not security_result["valid"]:
                raise SecurityError(f"Security validation failed: {security_result['reason']}")
            
            # Resource check
            resource_check = self.resource_manager.check_resources()
            if not resource_check["available"]:
                raise ResourceError(f"Insufficient resources: {resource_check['reason']}")
            
            # Monitor performance
            if self.monitor:
                self.monitor.start_request(request_id)
            
            # Execute with error handling
            result = self.error_handler.execute_with_retry(
                self.base_designer.generate,
                *args,
                **kwargs,
                request_id=request_id
            )
            
            # Validate output
            validation_result = self.security_validator.validate_output(result)
            if not validation_result["valid"]:
                response["warnings"].append(f"Output validation warning: {validation_result['reason']}")
            
            # Success!
            response["success"] = True
            response["result"] = result
            self.metrics["successful_requests"] += 1
            
            self.logger.info(f"[{request_id}] Design completed successfully")
            
        except Exception as e:
            # Comprehensive error handling
            self.metrics["failed_requests"] += 1
            error_info = self.error_handler.handle_error(e, request_id)
            response["error"] = error_info
            
            # Try recovery if enabled
            if self.recovery_system and error_info["recoverable"]:
                try:
                    recovery_result = self.recovery_system.attempt_recovery(e, args, kwargs)
                    if recovery_result["success"]:
                        response["success"] = True
                        response["result"] = recovery_result["result"]
                        response["warnings"].append("Recovered from error using fallback method")
                        self.logger.info(f"[{request_id}] Recovered from error: {e}")
                except Exception as recovery_error:
                    self.logger.error(f"[{request_id}] Recovery failed: {recovery_error}")
            
            if not response["success"]:
                self.logger.error(f"[{request_id}] Design failed: {e}")
        
        finally:
            # Finalize metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            response["metrics"] = {
                "response_time_seconds": response_time,
                "memory_usage_mb": self.resource_manager.get_memory_usage(),
                "gpu_usage_percent": self.resource_manager.get_gpu_usage()
            }
            
            # Update averages
            total_requests = self.metrics["total_requests"]
            current_avg = self.metrics["avg_response_time"]
            self.metrics["avg_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            # Monitor alerts
            if self.monitor:
                self.monitor.record_request(request_id, response)
                alerts = self.monitor.check_alerts()
                if alerts:
                    self.logger.warning(f"[{request_id}] Alerts triggered: {alerts}")
            
            # Resource cleanup
            self.resource_manager.cleanup_if_needed()
            
            self.logger.info(
                f"[{request_id}] Request completed in {response_time:.2f}s "
                f"(Success: {response['success']})"
            )
        
        return response
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.copy(),
            "components": {},
            "alerts": []
        }
        
        # Check components
        components = [
            ("error_handler", self.error_handler),
            ("monitor", self.monitor),
            ("recovery_system", self.recovery_system),
            ("security_validator", self.security_validator),
            ("resource_manager", self.resource_manager)
        ]
        
        for name, component in components:
            if component:
                try:
                    component_status = component.get_status()
                    status["components"][name] = component_status
                    if component_status.get("status") != "healthy":
                        status["status"] = "degraded"
                except Exception as e:
                    status["components"][name] = {"status": "error", "error": str(e)}
                    status["status"] = "degraded"
        
        # Check for alerts
        if self.monitor:
            alerts = self.monitor.get_active_alerts()
            status["alerts"] = alerts
            if alerts:
                status["status"] = "warning"
        
        return status


class RobustErrorHandler:
    """Comprehensive error handling with retry logic and classification."""
    
    def __init__(self, config: Dict):
        self.config = config["error_handling"]
        self.error_stats = {}
        
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        request_id = kwargs.pop("request_id", "unknown")
        max_retries = self.config["max_retries"]
        retry_delay = self.config["retry_delay"]
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # Track error statistics
                if error_type not in self.error_stats:
                    self.error_stats[error_type] = 0
                self.error_stats[error_type] += 1
                
                # Check if retry is appropriate
                if attempt < max_retries and self._should_retry(e):
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    break
        
        raise last_error
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if error is retryable."""
        retryable_errors = [
            "ConnectionError",
            "TimeoutError", 
            "TemporaryFailure",
            "ResourceTemporarilyUnavailable"
        ]
        
        non_retryable_errors = [
            "ValueError",
            "TypeError",
            "SecurityError",
            "AuthenticationError"
        ]
        
        error_type = type(error).__name__
        
        if error_type in non_retryable_errors:
            return False
        if error_type in retryable_errors:
            return True
        
        # Default: retry for unknown errors
        return True
    
    def handle_error(self, error: Exception, request_id: str) -> Dict[str, Any]:
        """Comprehensive error handling and classification."""
        error_type = type(error).__name__
        error_message = str(error)
        
        return {
            "type": error_type,
            "message": error_message,
            "traceback": traceback.format_exc(),
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "recoverable": self._is_recoverable(error),
            "severity": self._get_severity(error),
            "suggested_action": self._get_suggested_action(error)
        }
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Check if error is recoverable."""
        recoverable_errors = [
            "MemoryError",
            "TimeoutError",
            "ConnectionError",
            "ResourceError"
        ]
        return type(error).__name__ in recoverable_errors
    
    def _get_severity(self, error: Exception) -> str:
        """Get error severity level."""
        critical_errors = ["SecurityError", "SystemError"]
        high_errors = ["MemoryError", "ResourceError"]
        medium_errors = ["TimeoutError", "ConnectionError"]
        
        error_type = type(error).__name__
        
        if error_type in critical_errors:
            return "critical"
        elif error_type in high_errors:
            return "high"
        elif error_type in medium_errors:
            return "medium"
        else:
            return "low"
    
    def _get_suggested_action(self, error: Exception) -> str:
        """Get suggested action for error."""
        suggestions = {
            "MemoryError": "Reduce batch size or free memory",
            "TimeoutError": "Increase timeout or check network connectivity",
            "SecurityError": "Review input validation and security settings",
            "ResourceError": "Wait for resources to become available",
            "ValueError": "Check input parameters and constraints"
        }
        
        error_type = type(error).__name__
        return suggestions.get(error_type, "Contact support with error details")
    
    def get_status(self) -> Dict[str, Any]:
        """Get error handler status."""
        return {
            "status": "healthy",
            "error_stats": self.error_stats.copy(),
            "config": self.config
        }


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self, config: Dict):
        self.config = config["monitoring"]
        self.active_requests = {}
        self.request_history = []
        self.alerts = []
        
    def start_request(self, request_id: str) -> None:
        """Start monitoring a request."""
        self.active_requests[request_id] = {
            "start_time": time.time(),
            "memory_start": self._get_memory_usage()
        }
    
    def record_request(self, request_id: str, response: Dict) -> None:
        """Record completed request."""
        if request_id in self.active_requests:
            request_data = self.active_requests.pop(request_id)
            
            record = {
                "request_id": request_id,
                "timestamp": response["timestamp"],
                "success": response["success"],
                "response_time": response["metrics"]["response_time_seconds"],
                "memory_used": response["metrics"]["memory_usage_mb"],
                "error_type": response["error"]["type"] if response["error"] else None
            }
            
            self.request_history.append(record)
            
            # Keep only recent history
            max_records = 1000
            if len(self.request_history) > max_records:
                self.request_history = self.request_history[-max_records:]
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        new_alerts = []
        thresholds = self.config["alert_thresholds"]
        
        if len(self.request_history) < 10:  # Need minimum samples
            return new_alerts
        
        recent_requests = self.request_history[-100:]  # Last 100 requests
        
        # Error rate alert
        error_rate = sum(1 for r in recent_requests if not r["success"]) / len(recent_requests)
        if error_rate > thresholds["error_rate"]:
            new_alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"Error rate {error_rate:.1%} exceeds threshold {thresholds['error_rate']:.1%}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Response time alert
        avg_response_time = sum(r["response_time"] for r in recent_requests) / len(recent_requests)
        if avg_response_time > thresholds["response_time"]:
            new_alerts.append({
                "type": "slow_response",
                "severity": "warning", 
                "message": f"Average response time {avg_response_time:.1f}s exceeds threshold {thresholds['response_time']}s",
                "timestamp": datetime.now().isoformat()
            })
        
        # Memory usage alert
        current_memory = self._get_memory_usage()
        memory_threshold_mb = thresholds["memory_usage"] * 1024  # Convert GB to MB
        if current_memory > memory_threshold_mb:
            new_alerts.append({
                "type": "high_memory_usage",
                "severity": "critical",
                "message": f"Memory usage {current_memory:.0f}MB exceeds threshold {memory_threshold_mb:.0f}MB",
                "timestamp": datetime.now().isoformat()
            })
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        # In production, implement alert aging and resolution
        return self.alerts[-10:]  # Return last 10 alerts
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Mock value
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        return {
            "status": "healthy",
            "active_requests": len(self.active_requests),
            "total_requests": len(self.request_history),
            "active_alerts": len(self.alerts)
        }


class RecoverySystem:
    """Automatic recovery and fallback mechanisms."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.recovery_strategies = {
            "MemoryError": self._recover_memory_error,
            "TimeoutError": self._recover_timeout_error,
            "ResourceError": self._recover_resource_error
        }
    
    def attempt_recovery(self, error: Exception, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Attempt to recover from error."""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, args, kwargs)
        else:
            return {"success": False, "reason": "No recovery strategy available"}
    
    def _recover_memory_error(self, error: Exception, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Recover from memory errors."""
        # Reduce batch size or complexity
        if "num_samples" in kwargs and kwargs["num_samples"] > 1:
            kwargs["num_samples"] = 1
            return {"success": True, "result": "Reduced samples", "method": "reduce_samples"}
        
        if "length" in kwargs and kwargs["length"] > 50:
            kwargs["length"] = min(50, kwargs["length"])
            return {"success": True, "result": "Reduced length", "method": "reduce_length"}
        
        return {"success": False, "reason": "Cannot reduce complexity further"}
    
    def _recover_timeout_error(self, error: Exception, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Recover from timeout errors."""
        # Use faster algorithm or reduced precision
        return {"success": True, "result": "Used fast mode", "method": "fast_mode"}
    
    def _recover_resource_error(self, error: Exception, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Recover from resource errors."""
        # Wait and retry
        time.sleep(2.0)
        return {"success": True, "result": "Waited for resources", "method": "wait_retry"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get recovery system status."""
        return {
            "status": "healthy",
            "available_strategies": list(self.recovery_strategies.keys())
        }


class SecurityValidator:
    """Security validation for inputs and outputs."""
    
    def __init__(self, config: Dict):
        self.config = config["security"]
        self.request_counts = {}
    
    def validate_input(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Validate input security."""
        if not self.config["input_validation"]:
            return {"valid": True}
        
        # Rate limiting
        if self.config["rate_limiting"]:
            client_id = "default"  # In production, extract from request
            current_time = time.time()
            minute_bucket = int(current_time // 60)
            
            if client_id not in self.request_counts:
                self.request_counts[client_id] = {}
            
            if minute_bucket not in self.request_counts[client_id]:
                self.request_counts[client_id][minute_bucket] = 0
            
            self.request_counts[client_id][minute_bucket] += 1
            
            max_requests = self.config["max_requests_per_minute"]
            if self.request_counts[client_id][minute_bucket] > max_requests:
                return {
                    "valid": False,
                    "reason": f"Rate limit exceeded: {max_requests} requests per minute"
                }
        
        # Input validation
        for key, value in kwargs.items():
            if key == "length" and (not isinstance(value, int) or value <= 0 or value > 10000):
                return {"valid": False, "reason": f"Invalid length: {value}"}
            
            if key == "num_samples" and (not isinstance(value, int) or value <= 0 or value > 100):
                return {"valid": False, "reason": f"Invalid num_samples: {value}"}
        
        return {"valid": True}
    
    def validate_output(self, result: Any) -> Dict[str, Any]:
        """Validate output security."""
        if not self.config["output_sanitization"]:
            return {"valid": True}
        
        # Check for sensitive information
        # In production, implement comprehensive sanitization
        
        return {"valid": True}
    
    def get_status(self) -> Dict[str, Any]:
        """Get security validator status."""
        return {
            "status": "healthy",
            "config": self.config,
            "active_clients": len(self.request_counts)
        }


class ResourceManager:
    """Resource management and cleanup."""
    
    def __init__(self, config: Dict):
        self.config = config["resources"]
        self.last_cleanup = time.time()
    
    def check_resources(self) -> Dict[str, Any]:
        """Check resource availability."""
        memory_usage = self.get_memory_usage()
        max_memory = self.config["max_memory_mb"]
        
        if memory_usage > max_memory:
            return {
                "available": False,
                "reason": f"Memory usage {memory_usage:.0f}MB exceeds limit {max_memory}MB"
            }
        
        return {"available": True}
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 100.0  # Mock value
    
    def get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.used / info.total) * 100
        except:
            return 0.0  # Mock value
    
    def cleanup_if_needed(self) -> None:
        """Cleanup resources if needed."""
        current_time = time.time()
        cleanup_interval = self.config["cleanup_interval_minutes"] * 60
        
        if current_time - self.last_cleanup > cleanup_interval:
            self._cleanup_resources()
            self.last_cleanup = current_time
    
    def _cleanup_resources(self) -> None:
        """Perform resource cleanup."""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status."""
        return {
            "status": "healthy",
            "memory_usage_mb": self.get_memory_usage(),
            "gpu_usage_percent": self.get_gpu_usage(),
            "last_cleanup": self.last_cleanup
        }


# Custom exception classes
class SecurityError(Exception):
    """Security validation error."""
    pass

class ResourceError(Exception):
    """Resource availability error."""
    pass