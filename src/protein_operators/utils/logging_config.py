"""
Comprehensive logging configuration for protein operators.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for production logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored formatter for console output in development.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Format the message
        formatted = super().format(record)
        
        # Apply color
        return f"{color}{formatted}{reset}"


class ProteinOperatorsLogger:
    """
    Centralized logging configuration for protein operators.
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: Optional[Path] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_structured: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize logging configuration.
        
        Args:
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            log_dir: Directory for log files (default: logs/)
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_structured: Use structured JSON logging
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup log files to keep
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir or Path("logs")
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_structured = enable_structured
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        if self.enable_file:
            self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self._configure_logging()
    
    def _configure_logging(self) -> None:
        """Configure logging with appropriate handlers and formatters."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.enable_structured:
                console_formatter = StructuredFormatter()
            else:
                console_formatter = ColoredConsoleFormatter(
                    fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file:
            # Main application log
            app_log_path = self.log_dir / "protein_operators.log"
            app_handler = logging.handlers.RotatingFileHandler(
                app_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            app_handler.setLevel(self.log_level)
            
            # Error log (ERROR and CRITICAL only)
            error_log_path = self.log_dir / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            # Use structured format for file logs
            if self.enable_structured:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(
                    fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            app_handler.setFormatter(file_formatter)
            error_handler.setFormatter(file_formatter)
            
            root_logger.addHandler(app_handler)
            root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger for a specific module.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    def log_performance(self, logger: logging.Logger, operation: str, duration: float, **kwargs):
        """
        Log performance metrics in a structured way.
        
        Args:
            logger: Logger instance
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        metrics = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_log': True,
            **kwargs
        }
        
        logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra=metrics)
    
    def log_design_metrics(self, logger: logging.Logger, structure_id: str, metrics: Dict[str, Any]):
        """
        Log protein design metrics.
        
        Args:
            logger: Logger instance
            structure_id: Unique structure identifier
            metrics: Design metrics dictionary
        """
        log_data = {
            'structure_id': structure_id,
            'design_metrics': True,
            **metrics
        }
        
        logger.info(f"Design metrics for {structure_id}", extra=log_data)
    
    def log_optimization_progress(
        self,
        logger: logging.Logger,
        iteration: int,
        energy: float,
        gradient_norm: float,
        **kwargs
    ):
        """
        Log optimization progress.
        
        Args:
            logger: Logger instance
            iteration: Current iteration
            energy: Current energy value
            gradient_norm: Current gradient norm
            **kwargs: Additional optimization metrics
        """
        progress_data = {
            'optimization_progress': True,
            'iteration': iteration,
            'energy': energy,
            'gradient_norm': gradient_norm,
            **kwargs
        }
        
        logger.debug(f"Optimization iteration {iteration}: energy={energy:.6f}", extra=progress_data)
    
    def log_api_request(
        self,
        logger: logging.Logger,
        request_id: str,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        **kwargs
    ):
        """
        Log API request information.
        
        Args:
            logger: Logger instance
            request_id: Unique request identifier
            endpoint: API endpoint
            method: HTTP method
            response_time: Response time in seconds
            status_code: HTTP status code
            **kwargs: Additional request data
        """
        request_data = {
            'api_request': True,
            'request_id': request_id,
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'status_code': status_code,
            **kwargs
        }
        
        logger.info(
            f"API {method} {endpoint} - {status_code} ({response_time:.3f}s)",
            extra=request_data
        )


class LoggingContextManager:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize logging context.
        
        Args:
            logger: Logger instance
            **context: Context variables to add to log messages
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context and add context variables."""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original factory."""
        logging.setLogRecordFactory(self.old_factory)


def setup_logging(
    log_level: str = None,
    log_dir: str = None,
    structured: bool = None
) -> ProteinOperatorsLogger:
    """
    Setup logging with environment variable overrides.
    
    Args:
        log_level: Override log level
        log_dir: Override log directory
        structured: Override structured logging
        
    Returns:
        Configured logger instance
    """
    # Environment variable overrides
    env_log_level = os.getenv('PROTEIN_OPERATORS_LOG_LEVEL', 'INFO')
    env_log_dir = os.getenv('PROTEIN_OPERATORS_LOG_DIR', 'logs')
    env_structured = os.getenv('PROTEIN_OPERATORS_STRUCTURED_LOGS', 'false').lower() == 'true'
    env_console = os.getenv('PROTEIN_OPERATORS_CONSOLE_LOGS', 'true').lower() == 'true'
    env_file = os.getenv('PROTEIN_OPERATORS_FILE_LOGS', 'true').lower() == 'true'
    
    # Use provided values or environment defaults
    final_log_level = log_level or env_log_level
    final_log_dir = Path(log_dir or env_log_dir)
    final_structured = structured if structured is not None else env_structured
    
    logger_config = ProteinOperatorsLogger(
        log_level=final_log_level,
        log_dir=final_log_dir,
        enable_console=env_console,
        enable_file=env_file,
        enable_structured=final_structured
    )
    
    # Log configuration
    root_logger = logger_config.get_logger(__name__)
    root_logger.info(f"Logging configured: level={final_log_level}, "
                     f"structured={final_structured}, dir={final_log_dir}")
    
    return logger_config


# Performance monitoring decorator
def log_performance(operation_name: str = None):
    """
    Decorator to automatically log function performance.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            import functools
            
            logger = logging.getLogger(func.__module__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract relevant metrics from args/kwargs
                metrics = {}
                if args and hasattr(args[0], '__class__'):
                    metrics['class'] = args[0].__class__.__name__
                
                logger.info(f"Performance: {op_name} completed in {duration:.3f}s", 
                           extra={'operation': op_name, 'duration': duration, **metrics})
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Performance: {op_name} failed after {duration:.3f}s: {str(e)}",
                            extra={'operation': op_name, 'duration': duration, 'error': str(e)})
                raise
        
        return wrapper
    return decorator


# Global logger instance
_global_logger_config: Optional[ProteinOperatorsLogger] = None


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with global configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    global _global_logger_config
    
    if _global_logger_config is None:
        _global_logger_config = setup_logging()
    
    return _global_logger_config.get_logger(name)


def configure_global_logging(**kwargs) -> None:
    """
    Configure global logging settings.
    
    Args:
        **kwargs: Logging configuration parameters
    """
    global _global_logger_config
    _global_logger_config = ProteinOperatorsLogger(**kwargs)