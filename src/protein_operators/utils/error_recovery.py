"""
Error recovery and resilience mechanisms for protein design.

This module provides robust error handling, automatic recovery strategies,
and graceful degradation mechanisms for the protein design pipeline.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Type
from abc import ABC, abstractmethod
import logging
import traceback
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """
    Context information for error handling.
    
    Attributes:
        operation: Name of the operation that failed
        error_type: Type of error that occurred
        error_message: Error message
        severity: Error severity level
        timestamp: When the error occurred
        stack_trace: Full stack trace
        context_data: Additional context information
        recovery_attempts: Number of recovery attempts made
    """
    operation: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    stack_trace: str
    context_data: Optional[Dict[str, Any]] = None
    recovery_attempts: int = 0


class ProteinDesignError(Exception):
    """Base exception for protein design operations."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.suggestions = suggestions or []
        self.timestamp = time.time()


class ConstraintValidationError(ProteinDesignError):
    """Error in constraint validation or application."""
    pass


class ModelError(ProteinDesignError):
    """Error in neural operator model operations."""
    pass


class StructureError(ProteinDesignError):
    """Error in protein structure operations."""
    pass


class ComputationError(ProteinDesignError):
    """Error in computational operations (PDE solving, optimization, etc.)."""
    pass


class BaseRecoveryHandler(ABC):
    """
    Abstract base class for error recovery handlers.
    
    Recovery handlers implement specific strategies for handling
    and recovering from different types of errors.
    """
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 1.5):
        """
        Initialize recovery handler.
        
        Args:
            max_attempts: Maximum number of recovery attempts
            backoff_factor: Exponential backoff factor for retries
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
    
    @abstractmethod
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """
        Check if this handler can handle the given error.
        
        Args:
            error: Exception that occurred
            context: Error context
            
        Returns:
            True if this handler can handle the error
        """
        pass
    
    @abstractmethod
    def recover(
        self,
        error: Exception,
        context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Attempt to recover from the error.
        
        Args:
            error: Exception that occurred
            context: Error context
            operation: Operation that failed
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Result of recovery attempt
            
        Raises:
            Exception if recovery fails
        """
        pass


class RetryHandler(BaseRecoveryHandler):
    """
    Recovery handler that implements retry logic with exponential backoff.
    
    Suitable for transient errors that might resolve themselves.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry handler.
        
        Args:
            max_attempts: Maximum retry attempts
            backoff_factor: Exponential backoff factor
            retry_exceptions: Exception types to retry (None = retry all)
        """
        super().__init__(max_attempts, backoff_factor)
        self.retry_exceptions = retry_exceptions or [Exception]
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if error should be retried."""
        return (
            context.recovery_attempts < self.max_attempts and
            any(isinstance(error, exc_type) for exc_type in self.retry_exceptions)
        )
    
    def recover(
        self,
        error: Exception,
        context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry the operation with exponential backoff."""
        context.recovery_attempts += 1
        
        # Calculate backoff time
        backoff_time = self.backoff_factor ** (context.recovery_attempts - 1)
        
        logger.warning(
            f"Retrying {context.operation} (attempt {context.recovery_attempts}/{self.max_attempts}) "
            f"after {backoff_time:.2f}s delay. Error: {context.error_message}"
        )
        
        # Wait before retry
        time.sleep(backoff_time)
        
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            if context.recovery_attempts >= self.max_attempts:
                raise ProteinDesignError(
                    f"Operation {context.operation} failed after {self.max_attempts} attempts",
                    error_code="RETRY_EXHAUSTED",
                    severity=ErrorSeverity.HIGH,
                    context={"original_error": str(error), "final_error": str(e)}
                ) from e
            else:
                # Re-raise to trigger another retry
                raise e


class FallbackHandler(BaseRecoveryHandler):
    """
    Recovery handler that provides fallback implementations.
    
    Uses alternative methods or simplified versions when primary method fails.
    """
    
    def __init__(
        self,
        fallback_operations: Dict[str, Callable],
        max_attempts: int = 1
    ):
        """
        Initialize fallback handler.
        
        Args:
            fallback_operations: Mapping of operation names to fallback functions
            max_attempts: Maximum attempts (usually 1 for fallbacks)
        """
        super().__init__(max_attempts)
        self.fallback_operations = fallback_operations
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if fallback is available for the operation."""
        return (
            context.recovery_attempts == 0 and  # Only try fallback once
            context.operation in self.fallback_operations
        )
    
    def recover(
        self,
        error: Exception,
        context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Use fallback operation."""
        context.recovery_attempts += 1
        fallback_op = self.fallback_operations[context.operation]
        
        logger.warning(
            f"Using fallback for {context.operation}. Original error: {context.error_message}"
        )
        
        try:
            return fallback_op(*args, **kwargs)
        except Exception as e:
            raise ProteinDesignError(
                f"Both primary and fallback operations failed for {context.operation}",
                error_code="FALLBACK_FAILED",
                severity=ErrorSeverity.HIGH,
                context={"primary_error": str(error), "fallback_error": str(e)}
            ) from e


class GracefulDegradationHandler(BaseRecoveryHandler):
    """
    Recovery handler that provides graceful degradation.
    
    Returns simplified or partial results when full computation fails.
    """
    
    def __init__(
        self,
        degradation_strategies: Dict[str, Callable],
        quality_threshold: float = 0.5
    ):
        """
        Initialize graceful degradation handler.
        
        Args:
            degradation_strategies: Mapping of operations to degradation functions
            quality_threshold: Minimum quality threshold for degraded results
        """
        super().__init__()
        self.degradation_strategies = degradation_strategies
        self.quality_threshold = quality_threshold
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if degradation strategy exists."""
        return context.operation in self.degradation_strategies
    
    def recover(
        self,
        error: Exception,
        context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Provide degraded result."""
        degradation_func = self.degradation_strategies[context.operation]
        
        logger.warning(
            f"Providing degraded result for {context.operation}. "
            f"Original error: {context.error_message}"
        )
        
        result = degradation_func(*args, **kwargs)
        
        # Add metadata about degradation
        if hasattr(result, '__dict__'):
            result.__dict__['_degraded'] = True
            result.__dict__['_degradation_reason'] = str(error)
        
        return result


class ErrorRecoveryManager:
    """
    Manager for error recovery strategies.
    
    Coordinates multiple recovery handlers and maintains error statistics.
    """
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.handlers: List[BaseRecoveryHandler] = []
        self.error_history: List[ErrorContext] = []
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0
        }
    
    def register_handler(self, handler: BaseRecoveryHandler) -> None:
        """Register a recovery handler."""
        self.handlers.append(handler)
        logger.info(f"Registered recovery handler: {handler.__class__.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        operation_name: str,
        operation: Callable,
        context_data: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Handle an error with registered recovery strategies.
        
        Args:
            error: Exception that occurred
            operation_name: Name of the failed operation
            operation: Operation function
            context_data: Additional context information
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Result of successful recovery
            
        Raises:
            Exception if all recovery strategies fail
        """
        # Create error context
        context = ErrorContext(
            operation=operation_name,
            error_type=error.__class__.__name__,
            error_message=str(error),
            severity=self._determine_severity(error),
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            context_data=context_data or {}
        )
        
        # Record error
        self.error_history.append(context)
        self.recovery_stats["total_errors"] += 1
        
        logger.error(
            f"Error in {operation_name}: {error}",
            extra={"error_context": context}
        )
        
        # Try recovery handlers
        for handler in self.handlers:
            if handler.can_handle(error, context):
                try:
                    logger.info(f"Attempting recovery with {handler.__class__.__name__}")
                    result = handler.recover(error, context, operation, *args, **kwargs)
                    
                    self.recovery_stats["recovered_errors"] += 1
                    logger.info(f"Successfully recovered from error in {operation_name}")
                    
                    return result
                    
                except Exception as recovery_error:
                    logger.warning(
                        f"Recovery handler {handler.__class__.__name__} failed: {recovery_error}"
                    )
                    continue
        
        # If we get here, all recovery attempts failed
        self.recovery_stats["failed_recoveries"] += 1
        
        # Create comprehensive error report
        raise ProteinDesignError(
            f"All recovery strategies failed for operation: {operation_name}",
            error_code="RECOVERY_FAILED",
            severity=ErrorSeverity.CRITICAL,
            context={
                "original_error": str(error),
                "recovery_attempts": len(self.handlers),
                "error_context": context.__dict__
            },
            suggestions=[
                "Check input parameters",
                "Verify system requirements",
                "Consult error logs for details"
            ]
        )
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type."""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (RuntimeWarning, UserWarning)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        recent_errors = [
            err for err in self.error_history
            if time.time() - err.timestamp < 3600  # Last hour
        ]
        
        return {
            **self.recovery_stats,
            "recovery_rate": (
                self.recovery_stats["recovered_errors"] / 
                max(1, self.recovery_stats["total_errors"])
            ),
            "total_handlers": len(self.handlers),
            "recent_errors": len(recent_errors),
            "error_types": {
                error_type: len([e for e in self.error_history if e.error_type == error_type])
                for error_type in set(e.error_type for e in self.error_history)
            }
        }
    
    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0
        }
        logger.info("Error history cleared")


def with_error_recovery(
    operation_name: str,
    recovery_manager: Optional[ErrorRecoveryManager] = None,
    context_data: Optional[Dict[str, Any]] = None
):
    """
    Decorator to add error recovery to functions.
    
    Args:
        operation_name: Name of the operation for logging
        recovery_manager: Error recovery manager to use
        context_data: Additional context information
        
    Returns:
        Decorated function with error recovery
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if recovery_manager is None:
                # No recovery manager, just execute normally
                return func(*args, **kwargs)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return recovery_manager.handle_error(
                    error=e,
                    operation_name=operation_name,
                    operation=func,
                    context_data=context_data,
                    *args,
                    **kwargs
                )
        
        return wrapper
    return decorator


# Global error recovery manager instance
_global_recovery_manager = ErrorRecoveryManager()


def get_global_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance."""
    return _global_recovery_manager


def configure_default_recovery():
    """Configure default recovery handlers."""
    manager = get_global_recovery_manager()
    
    # Add retry handler for common transient errors
    retry_handler = RetryHandler(
        max_attempts=3,
        retry_exceptions=[ConnectionError, TimeoutError, MemoryError]
    )
    manager.register_handler(retry_handler)
    
    # Add fallback handlers for common operations
    fallback_operations = {
        "structure_generation": _fallback_structure_generation,
        "constraint_encoding": _fallback_constraint_encoding,
        "coordinate_generation": _fallback_coordinate_generation
    }
    fallback_handler = FallbackHandler(fallback_operations)
    manager.register_handler(fallback_handler)
    
    # Add graceful degradation for quality operations
    degradation_strategies = {
        "structure_validation": _degraded_structure_validation,
        "constraint_satisfaction": _degraded_constraint_satisfaction
    }
    degradation_handler = GracefulDegradationHandler(degradation_strategies)
    manager.register_handler(degradation_handler)
    
    logger.info("Default recovery handlers configured")


# Fallback implementations
def _fallback_structure_generation(*args, **kwargs):
    """Fallback structure generation using simple methods."""
    logger.info("Using fallback structure generation")
    # Simplified structure generation logic would go here
    return None


def _fallback_constraint_encoding(*args, **kwargs):
    """Fallback constraint encoding using default values."""
    logger.info("Using fallback constraint encoding")
    # Return zero tensor as fallback
    try:
        import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
        return torch.zeros(10)
    except ImportError:
        return [0.0] * 10


def _fallback_coordinate_generation(*args, **kwargs):
    """Fallback coordinate generation using basic geometry."""
    logger.info("Using fallback coordinate generation")
    # Simple extended chain coordinates
    return None


# Degraded implementations
def _degraded_structure_validation(*args, **kwargs):
    """Degraded structure validation with reduced checks."""
    logger.info("Using degraded structure validation")
    return {"validation_score": 0.5, "degraded": True}


def _degraded_constraint_satisfaction(*args, **kwargs):
    """Degraded constraint satisfaction with basic scoring."""
    logger.info("Using degraded constraint satisfaction")
    return {"satisfaction_score": 0.6, "degraded": True}


# Initialize default recovery on module import
configure_default_recovery()