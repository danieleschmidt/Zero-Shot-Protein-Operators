"""
Advanced logging system for protein operators with structured logging and performance tracking.
"""

import logging
import logging.handlers
import json
import time
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import os
from datetime import datetime
from functools import wraps
import threading
from contextlib import contextmanager

class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter with enhanced context."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add context information if available
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration
        if hasattr(record, 'protein_length'):
            log_entry['protein_length'] = record.protein_length
        if hasattr(record, 'constraint_count'):
            log_entry['constraint_count'] = record.constraint_count
        if hasattr(record, 'validation_score'):
            log_entry['validation_score'] = record.validation_score
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)

class PerformanceLogger:
    """Performance tracking and logging for protein design operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.operation_stats: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def track_operation(self, operation_name: str, **context):
        """Context manager to track operation performance."""
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Starting operation: {operation_name}",
                extra={'operation': operation_name, **context}
            )
            yield
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.logger.error(
                f"Operation failed: {operation_name}",
                extra={
                    'operation': operation_name,
                    'duration': duration,
                    'error': str(e),
                    **context
                },
                exc_info=True
            )
            raise
            
        else:
            duration = (time.time() - start_time) * 1000
            
            # Record performance statistics
            with self._lock:
                if operation_name not in self.operation_stats:
                    self.operation_stats[operation_name] = []
                self.operation_stats[operation_name].append(duration)
            
            self.logger.info(
                f"Operation completed: {operation_name}",
                extra={
                    'operation': operation_name,
                    'duration': duration,
                    **context
                }
            )
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all tracked operations."""
        with self._lock:
            stats = {}
            for operation, durations in self.operation_stats.items():
                if durations:
                    stats[operation] = {
                        'count': len(durations),
                        'mean_ms': sum(durations) / len(durations),
                        'min_ms': min(durations),
                        'max_ms': max(durations),
                        'total_ms': sum(durations)
                    }
            return stats
    
    def log_performance_summary(self):
        """Log a summary of all performance statistics."""
        stats = self.get_performance_stats()
        self.logger.info(
            "Performance summary",
            extra={'performance_stats': stats}
        )

class ProteinOperatorsLogger:
    """Main logging system for protein operators."""
    
    def __init__(
        self, 
        name: str = "protein_operators",
        level: str = "INFO",
        log_dir: Optional[Path] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_structured: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            if enable_structured:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            
            if enable_structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(file_handler)
        
        # Performance tracking
        self.performance = PerformanceLogger(self.logger)
        
        # Session context
        self.session_id = self._generate_session_id()
        self.user_id = None
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def set_user_context(self, user_id: str):
        """Set user context for logging."""
        self.user_id = user_id
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add session context to log extras."""
        context = {
            'session_id': self.session_id,
            **extra
        }
        if self.user_id:
            context['user_id'] = self.user_id
        return context
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, extra=self._add_context(kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, extra=self._add_context(kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, extra=self._add_context(kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with context and optional exception."""
        extra = self._add_context(kwargs)
        if exception:
            self.logger.error(message, extra=extra, exc_info=exception)
        else:
            self.logger.error(message, extra=extra)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with context and optional exception."""
        extra = self._add_context(kwargs)
        if exception:
            self.logger.critical(message, extra=extra, exc_info=exception)
        else:
            self.logger.critical(message, extra=extra)

# Global logger instance
_global_logger: Optional[ProteinOperatorsLogger] = None

def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = True
) -> ProteinOperatorsLogger:
    """Setup global logging configuration."""
    global _global_logger
    
    log_dir_path = Path(log_dir) if log_dir else None
    _global_logger = ProteinOperatorsLogger(
        level=level,
        log_dir=log_dir_path,
        enable_console=enable_console,
        enable_structured=enable_structured
    )
    
    return _global_logger

def get_logger() -> ProteinOperatorsLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger

def log_operation(operation_name: str):
    """Decorator to automatically log operation performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # Extract context from arguments
            context = {}
            if args and hasattr(args[0], '__class__'):
                context['class'] = args[0].__class__.__name__
            
            # Try to extract protein-specific context
            for arg in args:
                if hasattr(arg, 'num_residues'):
                    context['protein_length'] = arg.num_residues
                elif hasattr(arg, '__len__') and hasattr(arg, 'constraints'):
                    context['constraint_count'] = len(arg)
            
            with logger.performance.track_operation(operation_name, **context):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def log_validation_result(score: float, details: Dict[str, Any]):
    """Log validation results with structured data."""
    logger = get_logger()
    logger.info(
        "Structure validation completed",
        validation_score=score,
        validation_details=details
    )

def log_design_parameters(length: int, constraints: int, operator_type: str):
    """Log protein design parameters."""
    logger = get_logger()
    logger.info(
        "Protein design initiated",
        protein_length=length,
        constraint_count=constraints,
        operator_type=operator_type
    )

def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """Log error with additional context information."""
    logger = get_logger()
    logger.error(
        f"Error occurred: {str(error)}",
        exception=error,
        **context
    )

# Context managers for specific operations
@contextmanager
def log_protein_generation(length: int, operator_type: str):
    """Context manager for protein generation logging."""
    logger = get_logger()
    with logger.performance.track_operation(
        "protein_generation",
        protein_length=length,
        operator_type=operator_type
    ):
        yield

@contextmanager
def log_structure_validation():
    """Context manager for structure validation logging."""
    logger = get_logger()
    with logger.performance.track_operation("structure_validation"):
        yield

@contextmanager
def log_constraint_processing(constraint_count: int):
    """Context manager for constraint processing logging."""
    logger = get_logger()
    with logger.performance.track_operation(
        "constraint_processing",
        constraint_count=constraint_count
    ):
        yield