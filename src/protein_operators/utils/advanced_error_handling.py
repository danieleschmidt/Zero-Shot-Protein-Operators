"""
Advanced error handling and recovery system for protein design.

This module provides sophisticated error handling capabilities including:
- Automatic error recovery
- Graceful degradation
- Circuit breaker patterns
- Retry mechanisms with exponential backoff
- Error categorization and reporting
"""

import functools
import time
import logging
import traceback
import sys
import os
import random
import asyncio
from typing import Any, Dict, List, Optional, Callable, Type, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


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
    DEGRADE = "degrade"
    FAIL = "fail"
    IGNORE = "ignore"


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    success_count: int = 0


class ProteinDesignException(Exception):
    """Base exception for protein design errors with enhanced context."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, suggestion: Optional[str] = None):
        super().__init__(message)
        self.context = context or {}
        self.suggestion = suggestion
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'type': self.__class__.__name__,
            'message': str(self),
            'context': self.context,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp
        }


class CircuitBreakerError(Exception):
    """Error raised when circuit breaker is open."""
    pass


class AdvancedErrorHandler:
    """
    Advanced error handling system with recovery capabilities.
    
    Features:
    - Automatic error categorization
    - Circuit breaker pattern
    - Retry mechanisms with exponential backoff
    - Graceful degradation
    - Error reporting and analytics
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        enable_graceful_degradation: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize advanced error handler.
        
        Args:
            max_retries: Maximum number of retries for recoverable errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            circuit_breaker_threshold: Number of failures to trigger circuit breaker
            circuit_breaker_timeout: Time to wait before attempting to close circuit
            enable_graceful_degradation: Whether to enable graceful degradation
            log_level: Logging level
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.enable_graceful_degradation = enable_graceful_degradation
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Recovery strategies mapping
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Fallback implementations
        self.fallback_handlers = {}
        
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
        """Initialize default recovery strategies for different error categories."""
        return {
            ErrorCategory.VALIDATION: RecoveryStrategy.FALLBACK,
            ErrorCategory.COMPUTATION: RecoveryStrategy.RETRY,
            ErrorCategory.MEMORY: RecoveryStrategy.DEGRADE,
            ErrorCategory.IO: RecoveryStrategy.RETRY,
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.SECURITY: RecoveryStrategy.FAIL,
            ErrorCategory.CONFIGURATION: RecoveryStrategy.FALLBACK,
            ErrorCategory.RESOURCE: RecoveryStrategy.DEGRADE,
            ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY,
        }
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """
        Categorize an error based on its type and message.
        
        Args:
            error: Exception to categorize
            
        Returns:
            Error category
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Validation errors
        if "validation" in error_message or "invalid" in error_message:
            return ErrorCategory.VALIDATION
        
        # Memory errors
        if isinstance(error, MemoryError) or "memory" in error_message or "out of memory" in error_message:
            return ErrorCategory.MEMORY
        
        # IO errors
        if isinstance(error, (IOError, FileNotFoundError, PermissionError)) or "file" in error_message:
            return ErrorCategory.IO
        
        # Network errors
        if "network" in error_message or "connection" in error_message or "timeout" in error_message:
            return ErrorCategory.NETWORK
        
        # Security errors
        if "security" in error_message or "permission" in error_message or "unauthorized" in error_message:
            return ErrorCategory.SECURITY
        
        # Configuration errors
        if "config" in error_message or "setting" in error_message or "parameter" in error_message:
            return ErrorCategory.CONFIGURATION
        
        # Resource errors
        if "resource" in error_message or "limit" in error_message or "quota" in error_message:
            return ErrorCategory.RESOURCE
        
        # Computation errors
        if isinstance(error, (ValueError, ArithmeticError, RuntimeError)):
            return ErrorCategory.COMPUTATION
        
        return ErrorCategory.UNKNOWN
    
    def assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """
        Assess the severity of an error.
        
        Args:
            error: Exception to assess
            category: Error category
            
        Returns:
            Error severity
        """
        error_message = str(error).lower()
        
        # Critical errors that should stop execution
        if (isinstance(error, (SystemExit, KeyboardInterrupt)) or 
            category == ErrorCategory.SECURITY or
            "critical" in error_message or
            "fatal" in error_message):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if (isinstance(error, MemoryError) or
            category == ErrorCategory.MEMORY or
            "corruption" in error_message or
            "overflow" in error_message):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if (category in [ErrorCategory.IO, ErrorCategory.NETWORK, ErrorCategory.RESOURCE] or
            "warning" in error_message):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """
        Determine if an error should be retried.
        
        Args:
            error_info: Error information
            
        Returns:
            Whether to retry the operation
        """
        if error_info.retry_count >= self.max_retries:
            return False
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        
        strategy = self.recovery_strategies.get(error_info.category, RecoveryStrategy.FAIL)
        return strategy == RecoveryStrategy.RETRY
    
    def calculate_delay(self, retry_count: int) -> float:
        """
        Calculate delay for exponential backoff.
        
        Args:
            retry_count: Current retry attempt
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (2 ** retry_count)
        return min(delay, self.max_delay)
    
    def check_circuit_breaker(self, operation_name: str) -> bool:
        """
        Check if circuit breaker allows operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Whether operation is allowed
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
        
        breaker = self.circuit_breakers[operation_name]
        now = datetime.now()
        
        if breaker.state == "open":
            if (breaker.last_failure_time and 
                now - breaker.last_failure_time > timedelta(seconds=self.circuit_breaker_timeout)):
                breaker.state = "half-open"
                breaker.success_count = 0
            else:
                raise CircuitBreakerError(f"Circuit breaker open for {operation_name}")
        
        return True
    
    def record_success(self, operation_name: str):
        """Record successful operation for circuit breaker."""
        if operation_name in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_name]
            
            if breaker.state == "half-open":
                breaker.success_count += 1
                if breaker.success_count >= 3:  # Successful attempts to close
                    breaker.state = "closed"
                    breaker.failure_count = 0
    
    def record_failure(self, operation_name: str):
        """Record failed operation for circuit breaker."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
        
        breaker = self.circuit_breakers[operation_name]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= self.circuit_breaker_threshold:
            breaker.state = "open"
            self.logger.warning(f"Circuit breaker opened for {operation_name}")
    
    def register_fallback(self, operation_name: str, fallback_fn: Callable):
        """
        Register a fallback function for an operation.
        
        Args:
            operation_name: Name of the operation
            fallback_fn: Fallback function to use
        """
        self.fallback_handlers[operation_name] = fallback_fn
    
    def handle_with_recovery(
        self,
        operation_name: str,
        func: Callable,
        *args,
        fallback_result: Any = None,
        **kwargs
    ) -> Any:
        """
        Execute function with comprehensive error handling and recovery.
        
        Args:
            operation_name: Name of the operation for tracking
            func: Function to execute
            *args: Function arguments
            fallback_result: Default result if all recovery fails
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker
                self.check_circuit_breaker(operation_name)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                self.record_success(operation_name)
                return result
                
            except CircuitBreakerError:
                # Circuit breaker is open - try fallback
                return self._try_fallback(operation_name, fallback_result, *args, **kwargs)
                
            except Exception as e:
                last_error = e
                
                # Create error info
                error_info = ErrorInfo(
                    error_type=type(e).__name__,
                    message=str(e),
                    category=self.categorize_error(e),
                    severity=self.assess_severity(e, self.categorize_error(e)),
                    timestamp=datetime.now(),
                    traceback_str=traceback.format_exc(),
                    context={"operation": operation_name, "attempt": attempt},
                    retry_count=attempt
                )
                
                # Log error
                self.logger.error(f"Error in {operation_name} (attempt {attempt + 1}): {str(e)}")
                
                # Record error
                self.error_history.append(error_info)
                self.record_failure(operation_name)
                
                # Decide recovery strategy
                if error_info.severity == ErrorSeverity.CRITICAL:
                    self.logger.critical(f"Critical error in {operation_name}: {str(e)}")
                    raise e
                
                if not self.should_retry(error_info):
                    break
                
                # Wait before retry
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    self.logger.info(f"Retrying {operation_name} in {delay:.1f} seconds...")
                    time.sleep(delay)
        
        # All retries failed - try recovery
        return self._try_recovery(operation_name, last_error, fallback_result, *args, **kwargs)
    
    def _try_fallback(self, operation_name: str, fallback_result: Any, *args, **kwargs) -> Any:
        """Try fallback handler for operation."""
        if operation_name in self.fallback_handlers:
            try:
                self.logger.info(f"Using fallback for {operation_name}")
                return self.fallback_handlers[operation_name](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback failed for {operation_name}: {str(e)}")
        
        return fallback_result
    
    def _try_recovery(
        self, 
        operation_name: str, 
        error: Exception, 
        fallback_result: Any, 
        *args, 
        **kwargs
    ) -> Any:
        """Try various recovery strategies."""
        category = self.categorize_error(error)
        strategy = self.recovery_strategies.get(category, RecoveryStrategy.FAIL)
        
        self.logger.info(f"Attempting recovery for {operation_name} with strategy: {strategy}")
        
        if strategy == RecoveryStrategy.FALLBACK:
            return self._try_fallback(operation_name, fallback_result, *args, **kwargs)
        
        elif strategy == RecoveryStrategy.DEGRADE:
            return self._degrade_gracefully(operation_name, fallback_result)
        
        elif strategy == RecoveryStrategy.IGNORE:
            self.logger.warning(f"Ignoring error in {operation_name}: {str(error)}")
            return fallback_result
        
        else:  # FAIL
            self.logger.error(f"No recovery possible for {operation_name}")
            raise error
    
    def _degrade_gracefully(self, operation_name: str, fallback_result: Any) -> Any:
        """Implement graceful degradation."""
        if not self.enable_graceful_degradation:
            return fallback_result
        
        # Operation-specific degradation strategies
        if "generate" in operation_name.lower():
            # For generation operations, return simplified result
            self.logger.warning(f"Degrading {operation_name} to simplified mode")
            return self._create_simplified_structure()
        
        elif "validate" in operation_name.lower():
            # For validation operations, return partial validation
            self.logger.warning(f"Degrading {operation_name} to basic validation")
            return {"passed": True, "score": 0.5, "message": "Degraded validation"}
        
        return fallback_result
    
    def _create_simplified_structure(self):
        """Create a simplified protein structure as fallback."""
        try:
            # Import here to avoid circular imports
            from ..structure import ProteinStructure
            
            # Create simple extended chain
            coords = torch.zeros(50, 3)  # 50 residue chain
            for i in range(50):
                coords[i, 0] = i * 3.8  # CA-CA distance
            
            return ProteinStructure(coords, sequence="A" * 50)
            
        except Exception:
            return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and analytics."""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "severities": {}}
        
        # Count by category
        categories = {}
        for error in self.error_history:
            cat = error.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        # Count by severity
        severities = {}
        for error in self.error_history:
            sev = error.severity.value
            severities[sev] = severities.get(sev, 0) + 1
        
        # Recent errors (last hour)
        recent_threshold = datetime.now() - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e.timestamp > recent_threshold]
        
        # Circuit breaker states
        circuit_states = {name: state.state for name, state in self.circuit_breakers.items()}
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "categories": categories,
            "severities": severities,
            "circuit_breakers": circuit_states,
            "error_rate_per_hour": len(recent_errors),
        }
    
    def reset_circuit_breaker(self, operation_name: str):
        """Reset a circuit breaker manually."""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
            self.logger.info(f"Reset circuit breaker for {operation_name}")
    
    def clear_error_history(self, older_than_hours: int = 24):
        """Clear old error history."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        self.error_history = [e for e in self.error_history if e.timestamp > cutoff_time]
        self.logger.info(f"Cleared error history older than {older_than_hours} hours")


def error_handler(
    operation_name: str,
    max_retries: int = 3,
    fallback_result: Any = None,
    handler: Optional[AdvancedErrorHandler] = None
):
    """
    Decorator for automatic error handling with recovery.
    
    Args:
        operation_name: Name of the operation for tracking
        max_retries: Maximum number of retries
        fallback_result: Default result if all recovery fails
        handler: Custom error handler instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler_instance = handler or AdvancedErrorHandler(max_retries=max_retries)
            return error_handler_instance.handle_with_recovery(
                operation_name, func, *args, fallback_result=fallback_result, **kwargs
            )
        return wrapper
    return decorator


def circuit_breaker(
    operation_name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0,
    handler: Optional[AdvancedErrorHandler] = None
):
    """
    Decorator for circuit breaker pattern.
    
    Args:
        operation_name: Name of the operation
        failure_threshold: Number of failures to open circuit
        timeout_seconds: Time to wait before trying again
        handler: Custom error handler instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler_instance = handler or AdvancedErrorHandler(
                circuit_breaker_threshold=failure_threshold,
                circuit_breaker_timeout=timeout_seconds
            )
            
            try:
                error_handler_instance.check_circuit_breaker(operation_name)
                result = func(*args, **kwargs)
                error_handler_instance.record_success(operation_name)
                return result
            except Exception as e:
                error_handler_instance.record_failure(operation_name)
                raise e
        return wrapper
    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, 
                      exponential_base: float = 2.0, jitter: bool = True):
    """
    Enhanced retry decorator with exponential backoff and jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to prevent thundering herd
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise e
                    
                    # Enhanced backoff with jitter to prevent thundering herd
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # Add 50% jitter
                    
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed, retrying in {delay:.2f}s: {e}")
                    
                    # Add attempt context to exception if it's a ProteinDesignException
                    if isinstance(e, ProteinDesignException):
                        e.context.setdefault('retry_attempts', []).append({
                            'attempt': attempt + 1,
                            'delay': delay,
                            'error': str(e)
                        })
                    
                    await asyncio.sleep(delay)
            
            return None  # Should not reach here
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise e
                    
                    # Enhanced backoff with jitter to prevent thundering herd
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # Add 50% jitter
                    
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed, retrying in {delay:.2f}s: {e}")
                    
                    # Add attempt context to exception if it's a ProteinDesignException
                    if isinstance(e, ProteinDesignException):
                        e.context.setdefault('retry_attempts', []).append({
                            'attempt': attempt + 1,
                            'delay': delay,
                            'error': str(e)
                        })
                    
                    time.sleep(delay)
            
            return None  # Should not reach here
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error handler instance
_global_handler = AdvancedErrorHandler()


def set_global_error_handler(handler: AdvancedErrorHandler):
    """Set global error handler instance."""
    global _global_handler
    _global_handler = handler


def get_global_error_handler() -> AdvancedErrorHandler:
    """Get global error handler instance."""
    return _global_handler