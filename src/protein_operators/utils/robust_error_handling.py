"""
Robust error handling and recovery system for protein operators.
"""

import traceback
import sys
import time
from typing import Dict, Any, Optional, List, Callable, Union, Type
from functools import wraps
from contextlib import contextmanager
import threading
from dataclasses import dataclass, field
from enum import Enum
import logging

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    IO = "io"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"
    SYSTEM = "system"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    protein_length: Optional[int] = None
    constraint_count: Optional[int] = None
    operator_type: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback_str: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0

class ProteinOperatorsError(Exception):
    """Base exception for protein operators."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext(operation="unknown")
        self.original_error = original_error

class ValidationError(ProteinOperatorsError):
    """Error in protein structure or constraint validation."""
    
    def __init__(self, message: str, validation_details: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        self.validation_details = validation_details or {}

class ComputationError(ProteinOperatorsError):
    """Error in protein design computation."""
    
    def __init__(self, message: str, computation_stage: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.COMPUTATION, **kwargs)
        self.computation_stage = computation_stage

class ConfigurationError(ProteinOperatorsError):
    """Error in system or model configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        self.config_key = config_key

class ResourceError(ProteinOperatorsError):
    """Error related to system resources (memory, GPU, etc.)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.MEMORY, **kwargs)
        self.resource_type = resource_type

class RetryableError(ProteinOperatorsError):
    """Error that can be retried with backoff."""
    
    def __init__(self, message: str, max_retries: int = 3, **kwargs):
        super().__init__(message, **kwargs)
        self.max_retries = max_retries

class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_records: List[ErrorRecord] = []
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.error_callbacks: List[Callable[[ErrorRecord], None]] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_recovery_strategy(
        self, 
        error_type: Type[Exception], 
        strategy: Callable[[Exception, ErrorContext], Any]
    ):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    def add_error_callback(self, callback: Callable[[ErrorRecord], None]):
        """Add a callback to be called when errors occur."""
        self.error_callbacks.append(callback)
    
    def handle_error(
        self, 
        error: Exception, 
        context: ErrorContext,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle an error with optional recovery."""
        
        # Classify error
        severity = self._classify_severity(error)
        category = self._classify_category(error)
        
        # Create error record
        error_record = ErrorRecord(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            traceback_str=traceback.format_exc()
        )
        
        # Store error record
        with self._lock:
            self.error_records.append(error_record)
        
        # Log error
        self.logger.error(
            f"Error in {context.operation}: {str(error)}",
            extra={
                'error_type': error_record.error_type,
                'severity': severity.value,
                'category': category.value,
                'context': context.__dict__
            }
        )
        
        # Attempt recovery if enabled
        recovery_result = None
        if attempt_recovery:
            recovery_result = self._attempt_recovery(error, context, error_record)
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_record)
            except Exception as callback_error:
                self.logger.error(f"Error in error callback: {callback_error}")
        
        # Re-raise if recovery failed or not attempted
        if not error_record.recovery_successful:
            raise error
        
        return recovery_result
    
    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        if isinstance(error, ProteinOperatorsError):
            return error.severity
        elif isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (FileNotFoundError, ConnectionError)):
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _classify_category(self, error: Exception) -> ErrorCategory:
        """Classify error category."""
        if isinstance(error, ProteinOperatorsError):
            return error.category
        elif isinstance(error, (FileNotFoundError, IOError)):
            return ErrorCategory.IO
        elif isinstance(error, MemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.USER_INPUT
        elif isinstance(error, ConnectionError):
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.SYSTEM
    
    def _attempt_recovery(
        self, 
        error: Exception, 
        context: ErrorContext, 
        error_record: ErrorRecord
    ) -> Optional[Any]:
        """Attempt to recover from an error."""
        error_record.recovery_attempted = True
        
        # Check for specific recovery strategy
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    result = strategy(error, context)
                    error_record.recovery_successful = True
                    self.logger.info(f"Successfully recovered from {type(error).__name__}")
                    return result
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
                    break
        
        # General recovery strategies
        if isinstance(error, MemoryError):
            return self._recover_from_memory_error(context, error_record)
        elif isinstance(error, ValueError) and "constraint" in str(error).lower():
            return self._recover_from_constraint_error(error, context, error_record)
        elif isinstance(error, FileNotFoundError):
            return self._recover_from_file_error(error, context, error_record)
        
        return None
    
    def _recover_from_memory_error(self, context: ErrorContext, error_record: ErrorRecord) -> Optional[Any]:
        """Attempt recovery from memory errors."""
        try:
            # Clear potential memory leaks
            import gc
            gc.collect()
            
            # Try with reduced parameters
            if context.protein_length and context.protein_length > 50:
                self.logger.info("Attempting memory recovery by reducing protein length")
                error_record.recovery_successful = True
                return {"suggested_length": min(context.protein_length, 100)}
            
        except Exception:
            pass
        
        return None
    
    def _recover_from_constraint_error(
        self, 
        error: Exception, 
        context: ErrorContext, 
        error_record: ErrorRecord
    ) -> Optional[Any]:
        """Attempt recovery from constraint validation errors."""
        try:
            if "residue" in str(error).lower() and context.protein_length:
                # Adjust constraint positions
                self.logger.info("Attempting constraint recovery by adjusting residue positions")
                error_record.recovery_successful = True
                return {"action": "adjust_constraints", "max_residue": context.protein_length}
        except Exception:
            pass
        
        return None
    
    def _recover_from_file_error(
        self, 
        error: Exception, 
        context: ErrorContext, 
        error_record: ErrorRecord
    ) -> Optional[Any]:
        """Attempt recovery from file errors."""
        try:
            # Create missing directories
            file_path = str(error).split("'")[1] if "'" in str(error) else None
            if file_path:
                from pathlib import Path
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                error_record.recovery_successful = True
                return {"action": "created_directory", "path": file_path}
        except Exception:
            pass
        
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors."""
        with self._lock:
            if not self.error_records:
                return {"total_errors": 0}
            
            summary = {
                "total_errors": len(self.error_records),
                "by_severity": {},
                "by_category": {},
                "by_type": {},
                "recovery_rate": 0.0
            }
            
            for record in self.error_records:
                # Count by severity
                severity = record.severity.value
                summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
                
                # Count by category
                category = record.category.value
                summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
                
                # Count by type
                error_type = record.error_type
                summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1
            
            # Calculate recovery rate
            recovered = sum(1 for r in self.error_records if r.recovery_successful)
            attempted = sum(1 for r in self.error_records if r.recovery_attempted)
            if attempted > 0:
                summary["recovery_rate"] = recovered / attempted
            
            return summary

# Global error handler
_global_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def robust_operation(
    operation_name: str,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    recoverable_errors: Optional[List[Type[Exception]]] = None
):
    """Decorator for robust operation execution with retry logic."""
    recoverable_errors = recoverable_errors or [RetryableError, ConnectionError]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            context = ErrorContext(operation=operation_name)
            
            # Extract context from arguments
            if args and hasattr(args[0], '__class__'):
                context.additional_data['class'] = args[0].__class__.__name__
            
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    
                    # Check if error is recoverable
                    is_recoverable = any(isinstance(e, err_type) for err_type in recoverable_errors)
                    
                    if attempt < max_retries and is_recoverable:
                        wait_time = backoff_factor ** attempt
                        time.sleep(wait_time)
                        continue
                    
                    # Handle error (including non-recoverable ones)
                    try:
                        return error_handler.handle_error(e, context)
                    except Exception:
                        # If error handling also fails, raise original error
                        raise e
            
            # If all retries failed, raise the last error
            if last_error:
                raise last_error
        
        return wrapper
    return decorator

@contextmanager
def error_context(
    operation: str,
    **context_data
):
    """Context manager for error handling."""
    error_handler = get_error_handler()
    context = ErrorContext(operation=operation, **context_data)
    
    try:
        yield context
    except Exception as e:
        error_handler.handle_error(e, context)
        raise

def validate_inputs(**validators):
    """Decorator for input validation with detailed error messages."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate arguments
            func_args = {}
            if hasattr(func, '__code__'):
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                func_args = dict(zip(arg_names, args))
            func_args.update(kwargs)
            
            for param_name, validator in validators.items():
                if param_name in func_args:
                    value = func_args[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Invalid value for parameter '{param_name}': {value}",
                            validation_details={param_name: value}
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Common validators
def positive_integer(value) -> bool:
    """Validate that value is a positive integer."""
    return isinstance(value, int) and value > 0

def non_negative_float(value) -> bool:
    """Validate that value is a non-negative float."""
    return isinstance(value, (int, float)) and value >= 0

def valid_protein_length(value) -> bool:
    """Validate protein length."""
    return isinstance(value, int) and 5 <= value <= 5000

def non_empty_string(value) -> bool:
    """Validate that value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0

# Setup default recovery strategies
def setup_default_recovery_strategies():
    """Setup default recovery strategies for common errors."""
    error_handler = get_error_handler()
    
    def memory_recovery(error: MemoryError, context: ErrorContext):
        """Recovery strategy for memory errors."""
        import gc
        gc.collect()
        
        # Suggest reduced parameters
        suggestions = {}
        if context.protein_length and context.protein_length > 100:
            suggestions['protein_length'] = min(100, context.protein_length)
        
        return suggestions
    
    def validation_recovery(error: ValidationError, context: ErrorContext):
        """Recovery strategy for validation errors."""
        if error.validation_details:
            # Attempt to fix common validation issues
            return {"suggested_fixes": error.validation_details}
        return None
    
    error_handler.register_recovery_strategy(MemoryError, memory_recovery)
    error_handler.register_recovery_strategy(ValidationError, validation_recovery)

# Initialize default strategies
setup_default_recovery_strategies()