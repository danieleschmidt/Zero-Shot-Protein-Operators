"""
Comprehensive error handling utilities for protein operators.
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Type
from enum import Enum
import sys
from pathlib import Path


class ErrorCategory(Enum):
    """Categories of errors for better classification."""
    VALIDATION_ERROR = "validation_error"
    COMPUTATION_ERROR = "computation_error"
    IO_ERROR = "io_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    DATA_ERROR = "data_error"
    SYSTEM_ERROR = "system_error"


class ProteinOperatorError(Exception):
    """Base exception class for protein operators."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize protein operator error.
        
        Args:
            message: Human-readable error message
            category: Error category for classification
            details: Additional error details
            cause: Underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.cause = cause
        
        # Add traceback information
        self.traceback_info = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'category': self.category.value,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None,
            'traceback': self.traceback_info
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        base_msg = f"[{self.category.value}] {self.message}"
        if self.cause:
            base_msg += f" (caused by: {self.cause})"
        return base_msg


class ValidationError(ProteinOperatorError):
    """Error raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION_ERROR, **kwargs)
        self.field = field
        self.value = value
        if field:
            self.details['field'] = field
        if value is not None:
            self.details['invalid_value'] = str(value)


class ComputationError(ProteinOperatorError):
    """Error raised during protein computation or optimization."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.COMPUTATION_ERROR, **kwargs)
        self.operation = operation
        if operation:
            self.details['operation'] = operation


class ConfigurationError(ProteinOperatorError):
    """Error raised due to invalid configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION_ERROR, **kwargs)
        self.config_key = config_key
        if config_key:
            self.details['config_key'] = config_key


class ResourceError(ProteinOperatorError):
    """Error raised due to resource limitations."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.RESOURCE_ERROR, **kwargs)
        self.resource_type = resource_type
        if resource_type:
            self.details['resource_type'] = resource_type


class DataError(ProteinOperatorError):
    """Error raised due to data quality or format issues."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.DATA_ERROR, **kwargs)
        self.data_source = data_source
        if data_source:
            self.details['data_source'] = data_source


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
    
    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        strategy: Callable[[Exception], Any]
    ) -> None:
        """
        Register a recovery strategy for a specific error type.
        
        Args:
            error_type: Type of exception to handle
            strategy: Function to call for recovery
        """
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None,
        reraise: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: The exception to handle
            context: Additional context information
            operation: Name of the operation that failed
            reraise: Whether to reraise the exception after handling
            
        Returns:
            Recovery result if recovery was successful
            
        Raises:
            The original exception if reraise=True and no recovery
        """
        # Count error occurrences
        error_key = f"{error.__class__.__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Prepare error information
        error_info = {
            'error_type': error.__class__.__name__,
            'error_message': str(error),
            'operation': operation,
            'context': context or {},
            'occurrence_count': self.error_counts[error_key]
        }
        
        # Convert to ProteinOperatorError if not already
        if not isinstance(error, ProteinOperatorError):
            if isinstance(error, ValueError):
                wrapped_error = ValidationError(str(error), cause=error)
            elif isinstance(error, FileNotFoundError):
                wrapped_error = DataError(str(error), cause=error)
            elif isinstance(error, MemoryError):
                wrapped_error = ResourceError(str(error), resource_type="memory", cause=error)
            elif isinstance(error, RuntimeError):
                wrapped_error = ComputationError(str(error), operation=operation, cause=error)
            else:
                wrapped_error = ProteinOperatorError(str(error), cause=error)
        else:
            wrapped_error = error
        
        # Add context to wrapped error
        if context:
            wrapped_error.details.update(context)
        
        # Log the error
        self.logger.error(
            f"Error in {operation or 'unknown operation'}: {wrapped_error}",
            extra=error_info,
            exc_info=True
        )
        
        # Attempt recovery
        recovery_result = None
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    self.logger.info(f"Attempting recovery for {error_type.__name__}")
                    recovery_result = strategy(error)
                    self.logger.info("Recovery successful")
                    return recovery_result
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery failed: {recovery_error}")
        
        # Reraise if requested and no recovery
        if reraise:
            raise wrapped_error from error
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error occurrence statistics."""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            'registered_recovery_strategies': len(self.recovery_strategies)
        }
    
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()
        self.logger.info("Error statistics reset")


def robust_operation(
    retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    operation_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to make operations robust with retry logic.
    
    Args:
        retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        operation_name: Custom operation name for logging
        logger: Logger instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            last_exception = None
            current_delay = delay
            
            for attempt in range(retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt} for {op_name}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"Successfully completed {op_name} after {attempt} retries")
                    
                    return result
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {op_name}: {str(e)}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {retries + 1} attempts failed for {op_name}")
                        raise
            
            # This should not be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def safe_operation(
    default_return: Any = None,
    catch_exceptions: tuple = (Exception,),
    operation_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.ERROR
):
    """
    Decorator to make operations safe by catching exceptions and returning defaults.
    
    Args:
        default_return: Default value to return on exception
        catch_exceptions: Tuple of exceptions to catch
        operation_name: Custom operation name for logging
        logger: Logger instance
        log_level: Log level for caught exceptions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                return func(*args, **kwargs)
            except catch_exceptions as e:
                logger.log(
                    log_level,
                    f"Safe operation {op_name} failed, returning default: {str(e)}",
                    exc_info=True
                )
                return default_return
        
        return wrapper
    return decorator


def validate_input(
    validators: Dict[str, Callable[[Any], bool]],
    error_messages: Optional[Dict[str, str]] = None
):
    """
    Decorator to validate function inputs.
    
    Args:
        validators: Dictionary mapping parameter names to validation functions
        error_messages: Custom error messages for validation failures
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        error_msg = error_messages.get(param_name, f"Invalid value for {param_name}: {value}")
                        raise ValidationError(error_msg, field=param_name, value=value)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """
    Circuit breaker pattern to prevent cascading failures.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Type of exception that triggers circuit breaker
    """
    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = 0
        state = "closed"  # closed, open, half-open
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, state
            
            import time
            current_time = time.time()
            
            # Check if we should attempt recovery
            if state == "open" and current_time - last_failure_time > recovery_timeout:
                state = "half-open"
                failure_count = 0
            
            # Reject calls if circuit is open
            if state == "open":
                raise ComputationError(
                    f"Circuit breaker is open for {func.__name__}. "
                    f"Too many failures ({failure_count}). "
                    f"Retry after {recovery_timeout}s."
                )
            
            try:
                result = func(*args, **kwargs)
                
                # Success in half-open state closes the circuit
                if state == "half-open":
                    state = "closed"
                    failure_count = 0
                
                return result
                
            except expected_exception as e:
                failure_count += 1
                last_failure_time = current_time
                
                # Open circuit if threshold exceeded
                if failure_count >= failure_threshold:
                    state = "open"
                
                raise
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, **kwargs) -> None:
    """Handle an error using the global error handler."""
    handler = get_error_handler()
    handler.handle_error(error, **kwargs)


# Common validation functions
def is_positive_number(value: Any) -> bool:
    """Check if value is a positive number."""
    try:
        return isinstance(value, (int, float)) and value > 0
    except:
        return False


def is_valid_coordinates(value: Any) -> bool:
    """Check if value is valid protein coordinates."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.dim() == 2 and value.shape[1] == 3 and not torch.isnan(value).any()
        elif isinstance(value, (list, tuple)):
            return all(len(coord) == 3 for coord in value)
        return False
    except:
        return False


def is_valid_sequence(value: Any) -> bool:
    """Check if value is a valid amino acid sequence."""
    try:
        if not isinstance(value, str):
            return False
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        return all(aa.upper() in valid_aa for aa in value)
    except:
        return False


def is_valid_length(min_val: int = 1, max_val: int = 2000) -> Callable[[Any], bool]:
    """Create a validator for protein length."""
    def validator(value: Any) -> bool:
        try:
            return isinstance(value, int) and min_val <= value <= max_val
        except:
            return False
    return validator