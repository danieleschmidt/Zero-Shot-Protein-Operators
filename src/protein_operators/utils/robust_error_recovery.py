"""
Robust error recovery and resilience system for protein design.

This module provides advanced error handling, recovery mechanisms,
and fault tolerance for the protein operators framework.
"""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Callable, Union, Type
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Handle import compatibility
try:
    import torch
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    import mock_torch as torch


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    NOTIFY_AND_CONTINUE = "notify_continue"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    inputs: Dict[str, Any]
    timestamp: float
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProteinOperatorError(Exception):
    """Base exception for protein operator operations."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.timestamp = time.time()


class ConstraintValidationError(ProteinOperatorError):
    """Raised when constraint validation fails."""
    pass


class StructureGenerationError(ProteinOperatorError):
    """Raised when structure generation fails.""" 
    pass


class PhysicsValidationError(ProteinOperatorError):
    """Raised when physics validation fails."""
    pass


class ModelLoadError(ProteinOperatorError):
    """Raised when model loading fails."""
    pass


class RobustErrorHandler:
    """
    Advanced error handling system with recovery strategies.
    
    Provides intelligent error recovery, retry logic, fallback mechanisms,
    and comprehensive error tracking for production deployment.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.error_history: List[ErrorContext] = []
        self.recovery_functions: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[Type[Exception], Callable] = {}
        self.max_error_history = 1000
        
        # Performance tracking
        self.operation_stats = {}
        self.error_rates = {}
        
        # Register default recovery functions
        self._register_default_recovery_functions()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated error handling logger."""
        logger = logging.getLogger('protein_operators.error_handler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _register_default_recovery_functions(self):
        """Register default recovery functions for common operations."""
        self.recovery_functions.update({
            'constraint_validation': self._recover_constraint_validation,
            'structure_generation': self._recover_structure_generation,
            'physics_validation': self._recover_physics_validation,
            'model_loading': self._recover_model_loading,
        })
    
    def register_recovery_function(self, operation: str, func: Callable):
        """Register a custom recovery function for an operation."""
        self.recovery_functions[operation] = func
    
    def register_fallback_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register a fallback handler for a specific exception type."""
        self.fallback_handlers[exception_type] = handler
    
    def robust_execute(
        self,
        operation: str,
        func: Callable,
        *args,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        max_retries: int = 3,
        fallback_result: Any = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with robust error handling and recovery.
        
        Args:
            operation: Name of the operation being performed
            func: Function to execute
            *args: Arguments for the function
            severity: Error severity level
            recovery_strategy: Strategy for error recovery
            max_retries: Maximum number of retry attempts
            fallback_result: Result to return if all recovery fails
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result or fallback result
            
        Raises:
            ProteinOperatorError: If recovery fails and strategy is ABORT
        """
        context = ErrorContext(
            operation=operation,
            inputs={'args': args, 'kwargs': kwargs},
            timestamp=time.time(),
            severity=severity,
            recovery_strategy=recovery_strategy,
            max_retries=max_retries
        )
        
        start_time = time.time()
        
        while context.retry_count <= max_retries:
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Track successful operation
                self._track_operation_success(operation, time.time() - start_time)
                
                return result
                
            except Exception as e:
                context.retry_count += 1
                self._log_error(e, context)
                
                # Add to error history
                self.error_history.append(context)
                if len(self.error_history) > self.max_error_history:
                    self.error_history.pop(0)
                
                # Track error rate
                self._track_error_rate(operation)
                
                # Determine recovery action
                if context.retry_count <= max_retries:
                    recovery_result = self._attempt_recovery(e, context)
                    if recovery_result is not None:
                        return recovery_result
                else:
                    # Max retries exceeded
                    return self._handle_final_failure(e, context, fallback_result)
        
        return fallback_result
    
    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Attempt to recover from an error using registered strategies."""
        
        if context.recovery_strategy == RecoveryStrategy.SKIP:
            self.logger.warning(f"Skipping operation {context.operation} due to error")
            return None
        
        if context.recovery_strategy == RecoveryStrategy.NOTIFY_AND_CONTINUE:
            self.logger.error(f"Error in {context.operation}, continuing with degraded functionality")
            return None
        
        # Try operation-specific recovery
        if context.operation in self.recovery_functions:
            try:
                recovery_result = self.recovery_functions[context.operation](error, context)
                if recovery_result is not None:
                    self.logger.info(f"Successfully recovered from error in {context.operation}")
                    return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Recovery function failed for {context.operation}: {recovery_error}")
        
        # Try exception-type specific fallback
        for exc_type, handler in self.fallback_handlers.items():
            if isinstance(error, exc_type):
                try:
                    fallback_result = handler(error, context)
                    self.logger.info(f"Fallback handler resolved error for {context.operation}")
                    return fallback_result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback handler failed: {fallback_error}")
        
        # Default retry behavior
        if context.recovery_strategy == RecoveryStrategy.RETRY:
            wait_time = min(2 ** context.retry_count, 30)  # Exponential backoff, max 30s
            self.logger.info(f"Retrying {context.operation} in {wait_time} seconds (attempt {context.retry_count})")
            time.sleep(wait_time)
        
        return None
    
    def _handle_final_failure(self, error: Exception, context: ErrorContext, fallback_result: Any) -> Any:
        """Handle final failure after all recovery attempts."""
        
        if context.recovery_strategy == RecoveryStrategy.ABORT:
            raise ProteinOperatorError(
                f"Operation {context.operation} failed after {context.max_retries} retries",
                severity=context.severity,
                context=context,
                original_error=error
            )
        elif context.recovery_strategy == RecoveryStrategy.FALLBACK:
            self.logger.warning(
                f"Using fallback result for {context.operation} after {context.max_retries} retries"
            )
            return fallback_result
        else:
            self.logger.error(
                f"All recovery attempts failed for {context.operation}, returning fallback"
            )
            return fallback_result
    
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with appropriate level and context."""
        
        error_msg = f"Error in {context.operation} (attempt {context.retry_count}): {str(error)}"
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg)
        else:
            self.logger.info(error_msg)
        
        # Log stack trace for high severity errors
        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Stack trace for {context.operation}:\n{traceback.format_exc()}")
    
    def _track_operation_success(self, operation: str, duration: float):
        """Track successful operation metrics."""
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                'success_count': 0,
                'total_duration': 0,
                'avg_duration': 0
            }
        
        stats = self.operation_stats[operation]
        stats['success_count'] += 1
        stats['total_duration'] += duration
        stats['avg_duration'] = stats['total_duration'] / stats['success_count']
    
    def _track_error_rate(self, operation: str):
        """Track error rates for operations."""
        if operation not in self.error_rates:
            self.error_rates[operation] = {
                'error_count': 0,
                'last_error_time': 0,
                'error_rate_per_minute': 0
            }
        
        current_time = time.time()
        error_data = self.error_rates[operation]
        error_data['error_count'] += 1
        error_data['last_error_time'] = current_time
        
        # Calculate errors per minute (simple sliding window)
        recent_errors = sum(
            1 for ctx in self.error_history 
            if ctx.operation == operation and (current_time - ctx.timestamp) < 60
        )
        error_data['error_rate_per_minute'] = recent_errors
    
    # Default recovery functions
    
    def _recover_constraint_validation(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recover from constraint validation errors."""
        self.logger.info("Attempting constraint validation recovery")
        
        # Try with relaxed validation
        try:
            args, kwargs = context.inputs['args'], context.inputs['kwargs']
            if 'strict' in kwargs:
                kwargs = kwargs.copy()
                kwargs['strict'] = False
                self.logger.info("Retrying constraint validation with relaxed mode")
                # Would call the original function with relaxed parameters
                return None  # Placeholder for actual recovery
        except Exception:
            pass
        
        return None
    
    def _recover_structure_generation(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recover from structure generation errors."""
        self.logger.info("Attempting structure generation recovery")
        
        # Try with simpler parameters
        try:
            args, kwargs = context.inputs['args'], context.inputs['kwargs']
            if 'physics_guided' in kwargs:
                kwargs = kwargs.copy()
                kwargs['physics_guided'] = False
                self.logger.info("Retrying structure generation without physics guidance")
                return None  # Placeholder for actual recovery
        except Exception:
            pass
        
        return None
    
    def _recover_physics_validation(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recover from physics validation errors."""
        self.logger.info("Attempting physics validation recovery")
        
        # Return simplified validation results
        return {
            'physics_score': 0.5,
            'energy': 0.0,
            'forces': torch.zeros(3),
            'recovered': True
        }
    
    def _recover_model_loading(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recover from model loading errors."""
        self.logger.info("Attempting model loading recovery")
        
        # Try alternative model paths or default initialization
        try:
            # Could try different checkpoint paths, fallback models, etc.
            self.logger.info("Using default model initialization")
            return None  # Placeholder - would return a default model
        except Exception:
            pass
        
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history and statistics."""
        
        current_time = time.time()
        recent_errors = [
            ctx for ctx in self.error_history 
            if (current_time - ctx.timestamp) < 3600  # Last hour
        ]
        
        error_by_operation = {}
        error_by_severity = {}
        
        for error_ctx in recent_errors:
            # By operation
            op = error_ctx.operation
            if op not in error_by_operation:
                error_by_operation[op] = 0
            error_by_operation[op] += 1
            
            # By severity
            sev = error_ctx.severity.value
            if sev not in error_by_severity:
                error_by_severity[sev] = 0
            error_by_severity[sev] += 1
        
        return {
            'total_errors_last_hour': len(recent_errors),
            'errors_by_operation': error_by_operation,
            'errors_by_severity': error_by_severity,
            'operation_stats': self.operation_stats,
            'error_rates': self.error_rates,
            'top_failing_operations': sorted(
                error_by_operation.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the error handling system."""
        
        current_time = time.time()
        recent_errors = [
            ctx for ctx in self.error_history 
            if (current_time - ctx.timestamp) < 300  # Last 5 minutes
        ]
        
        critical_errors = [
            ctx for ctx in recent_errors 
            if ctx.severity == ErrorSeverity.CRITICAL
        ]
        
        high_error_rate_ops = [
            op for op, data in self.error_rates.items()
            if data['error_rate_per_minute'] > 10
        ]
        
        health_status = "healthy"
        if critical_errors:
            health_status = "critical"
        elif high_error_rate_ops:
            health_status = "degraded"
        elif len(recent_errors) > 50:
            health_status = "warning"
        
        return {
            'status': health_status,
            'recent_errors': len(recent_errors),
            'critical_errors': len(critical_errors),
            'high_error_rate_operations': high_error_rate_ops,
            'system_health_score': self._calculate_health_score(),
            'recommendations': self._get_health_recommendations()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        
        if not self.operation_stats:
            return 1.0  # No operations yet
        
        total_operations = sum(stats['success_count'] for stats in self.operation_stats.values())
        total_errors = len(self.error_history)
        
        if total_operations == 0:
            return 1.0
        
        success_rate = total_operations / (total_operations + total_errors)
        
        # Adjust for recent error rate
        recent_error_penalty = min(len([
            ctx for ctx in self.error_history 
            if (time.time() - ctx.timestamp) < 300
        ]) / 100.0, 0.5)
        
        return max(0.0, success_rate - recent_error_penalty)
    
    def _get_health_recommendations(self) -> List[str]:
        """Get recommendations for improving system health."""
        
        recommendations = []
        
        # Check error rates
        for op, data in self.error_rates.items():
            if data['error_rate_per_minute'] > 5:
                recommendations.append(f"High error rate for {op} - consider optimization")
        
        # Check critical errors
        recent_critical = [
            ctx for ctx in self.error_history[-10:]
            if ctx.severity == ErrorSeverity.CRITICAL
        ]
        if recent_critical:
            recommendations.append("Critical errors detected - immediate attention required")
        
        # Check operation performance
        slow_operations = [
            op for op, stats in self.operation_stats.items()
            if stats['avg_duration'] > 10.0
        ]
        if slow_operations:
            recommendations.append(f"Slow operations detected: {', '.join(slow_operations)}")
        
        return recommendations


def robust_operation(
    operation_name: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_retries: int = 3,
    fallback_result: Any = None
):
    """
    Decorator for making operations robust with error handling.
    
    Usage:
        @robust_operation("structure_generation", severity=ErrorSeverity.HIGH)
        def generate_structure(constraints, length):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create error handler from context
            error_handler = getattr(wrapper, '_error_handler', None)
            if error_handler is None:
                error_handler = RobustErrorHandler()
                wrapper._error_handler = error_handler
            
            return error_handler.robust_execute(
                operation=operation_name,
                func=func,
                *args,
                severity=severity,
                recovery_strategy=recovery_strategy,
                max_retries=max_retries,
                fallback_result=fallback_result,
                **kwargs
            )
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = RobustErrorHandler()


def get_global_error_handler() -> RobustErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


def set_global_error_handler(handler: RobustErrorHandler):
    """Set the global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler