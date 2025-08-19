"""
Security management system for protein design operations.

Features:
- Input validation and sanitization
- Access control and authentication
- Secure data handling
- Audit logging
- Rate limiting
"""

from typing import Dict, List, Optional, Union, Any, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch

import hashlib
import hmac
import secrets
import time
import threading
import re
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

from .advanced_logger import AdvancedLogger


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVILEGED = "privileged"
    ADMIN = "admin"


class AuditEventType(Enum):
    """Types of audit events."""
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DESIGN_GENERATED = "design_generated"
    VALIDATION_PERFORMED = "validation_performed"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class AuditEvent:
    """Audit log event."""
    event_type: AuditEventType
    user_id: Optional[str]
    operation: str
    resource: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    security_level: SecurityLevel
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[float] = None
    permissions: List[str] = field(default_factory=list)


class InputValidator:
    """
    Input validation and sanitization for protein design parameters.
    """
    
    def __init__(self):
        self.logger = AdvancedLogger(__name__)
        
        # Validation patterns
        self.safe_filename_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
        self.safe_identifier_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')
        
        # Limits
        self.max_protein_length = 2000
        self.max_constraints_count = 50
        self.max_string_length = 1000
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    def validate_protein_length(self, length: int) -> bool:
        """Validate protein length parameter."""
        if not isinstance(length, int):
            raise ValueError("Protein length must be an integer")
        
        if length <= 0:
            raise ValueError("Protein length must be positive")
        
        if length > self.max_protein_length:
            raise ValueError(f"Protein length exceeds maximum ({self.max_protein_length})")
        
        return True
    
    def validate_constraint_count(self, constraints) -> bool:
        """Validate number of constraints."""
        if hasattr(constraints, 'all_constraints'):
            count = len(constraints.all_constraints())
        elif isinstance(constraints, (list, tuple)):
            count = len(constraints)
        else:
            count = 1
        
        if count > self.max_constraints_count:
            raise ValueError(f"Too many constraints ({count}). Maximum: {self.max_constraints_count}")
        
        return True
    
    def validate_tensor_input(self, tensor) -> bool:
        """Validate tensor inputs for safety."""
        if not hasattr(tensor, 'shape'):
            raise ValueError("Input must be a tensor-like object")
        
        # Check for reasonable dimensions
        if len(tensor.shape) > 4:
            raise ValueError("Tensor has too many dimensions (max 4)")
        
        # Check for reasonable size
        total_elements = 1
        for dim in tensor.shape:
            total_elements *= dim
        
        if total_elements > 10**8:  # 100M elements
            raise ValueError("Tensor is too large")
        
        # Check for NaN or Inf values
        try:
            if hasattr(tensor, 'isnan') and tensor.isnan().any():
                raise ValueError("Tensor contains NaN values")
            
            if hasattr(tensor, 'isinf') and tensor.isinf().any():
                raise ValueError("Tensor contains infinite values")
        except Exception:
            # Handle mock tensors that don't support these operations
            pass
        
        return True
    
    def sanitize_string(self, text: str, max_length: Optional[int] = None) -> str:
        """Sanitize string inputs."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        max_len = max_length or self.max_string_length
        if len(text) > max_len:
            raise ValueError(f"String too long (max {max_len} characters)")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'\\\\/]', '', text)
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized.strip()
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for safety."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        
        if len(filename) > 255:
            raise ValueError("Filename too long")
        
        if not self.safe_filename_pattern.match(filename):
            raise ValueError("Filename contains invalid characters")
        
        # Prevent directory traversal
        if '..' in filename or filename.startswith('/') or ':' in filename:
            raise ValueError("Invalid filename")
        
        return True
    
    def validate_design_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize design parameters."""
        validated = {}
        
        for key, value in parameters.items():
            # Validate key
            if not self.safe_identifier_pattern.match(key):
                raise ValueError(f"Invalid parameter name: {key}")
            
            # Validate value based on type
            if isinstance(value, str):
                validated[key] = self.sanitize_string(value)
            elif isinstance(value, (int, float)):
                if abs(value) > 1e10:  # Reasonable numeric limit
                    raise ValueError(f"Numeric value too large: {value}")
                validated[key] = value
            elif isinstance(value, bool):
                validated[key] = value
            elif isinstance(value, (list, tuple)):
                if len(value) > 1000:  # Reasonable list size
                    raise ValueError("List parameter too long")
                validated[key] = value
            else:
                # For other types, convert to string and sanitize
                validated[key] = self.sanitize_string(str(value))
        
        return validated


class RateLimiter:
    """
    Rate limiting system to prevent abuse.
    """
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            'design_generation': (10, 3600),  # 10 requests per hour
            'validation': (100, 3600),        # 100 requests per hour
            'api_call': (1000, 3600),         # 1000 API calls per hour
            'file_upload': (5, 300),          # 5 uploads per 5 minutes
        }
        self.lock = threading.Lock()
    
    def is_allowed(self, user_id: str, operation: str) -> bool:
        """Check if operation is allowed for user."""
        if operation not in self.limits:
            return True  # No limit defined
        
        max_requests, window_seconds = self.limits[operation]
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            user_requests = self.requests[f"{user_id}:{operation}"]
            
            # Remove old requests
            while user_requests and user_requests[0] < cutoff_time:
                user_requests.popleft()
            
            # Check if limit exceeded
            if len(user_requests) >= max_requests:
                return False
            
            # Add current request
            user_requests.append(current_time)
            return True
    
    def get_remaining_requests(self, user_id: str, operation: str) -> int:
        """Get remaining requests for user and operation."""
        if operation not in self.limits:
            return float('inf')
        
        max_requests, window_seconds = self.limits[operation]
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            user_requests = self.requests[f"{user_id}:{operation}"]
            
            # Remove old requests
            while user_requests and user_requests[0] < cutoff_time:
                user_requests.popleft()
            
            return max(0, max_requests - len(user_requests))
    
    def reset_user_limits(self, user_id: str):
        """Reset all limits for a user (admin function)."""
        with self.lock:
            keys_to_remove = [key for key in self.requests.keys() if key.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self.requests[key]


class AccessControl:
    """
    Access control system for protein design operations.
    """
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.permissions = {
            'design.generate': SecurityLevel.AUTHENTICATED,
            'design.validate': SecurityLevel.PUBLIC,
            'design.optimize': SecurityLevel.AUTHENTICATED,
            'admin.users': SecurityLevel.ADMIN,
            'admin.config': SecurityLevel.ADMIN,
            'data.export': SecurityLevel.PRIVILEGED,
            'model.train': SecurityLevel.PRIVILEGED,
        }
        self.lock = threading.Lock()
        self.logger = AdvancedLogger(__name__)
    
    def create_user(
        self,
        username: str,
        email: str,
        security_level: SecurityLevel = SecurityLevel.AUTHENTICATED,
        permissions: Optional[List[str]] = None
    ) -> str:
        """Create a new user account."""
        user_id = secrets.token_urlsafe(16)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            security_level=security_level,
            permissions=permissions or []
        )
        
        with self.lock:
            self.users[user_id] = user
        
        self.logger.info(f"Created user {username} with ID {user_id}")
        return user_id
    
    def authenticate_user(self, user_id: str) -> Optional[User]:
        """Authenticate a user and return user object."""
        with self.lock:
            user = self.users.get(user_id)
        
        if user is None:
            return None
        
        if not user.is_active:
            return None
        
        # Check if account is locked
        if user.locked_until and time.time() < user.locked_until:
            return None
        
        # Update last login
        user.last_login = time.time()
        user.failed_login_attempts = 0
        
        return user
    
    def check_permission(
        self,
        user: Optional[User],
        operation: str
    ) -> bool:
        """Check if user has permission for operation."""
        required_level = self.permissions.get(operation, SecurityLevel.ADMIN)
        
        # Public operations
        if required_level == SecurityLevel.PUBLIC:
            return True
        
        # Authenticated operations
        if user is None:
            return False
        
        if required_level == SecurityLevel.AUTHENTICATED:
            return True
        
        # Check user security level
        user_level_value = list(SecurityLevel).index(user.security_level)
        required_level_value = list(SecurityLevel).index(required_level)
        
        if user_level_value >= required_level_value:
            return True
        
        # Check specific permissions
        if operation in user.permissions:
            return True
        
        return False
    
    def create_session(self, user_id: str) -> Optional[str]:
        """Create a new session for user."""
        user = self.authenticate_user(user_id)
        if user is None:
            return None
        
        session_token = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': time.time() + 3600,  # 1 hour
        }
        
        with self.lock:
            self.sessions[session_token] = session_data
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[User]:
        """Validate session and return user."""
        with self.lock:
            session_data = self.sessions.get(session_token)
        
        if session_data is None:
            return None
        
        # Check expiration
        if time.time() > session_data['expires_at']:
            with self.lock:
                del self.sessions[session_token]
            return None
        
        # Get user
        user = self.authenticate_user(session_data['user_id'])
        return user
    
    def revoke_session(self, session_token: str):
        """Revoke a session."""
        with self.lock:
            if session_token in self.sessions:
                del self.sessions[session_token]


class AuditLogger:
    """
    Audit logging system for security events.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.audit_events = deque(maxlen=10000)
        self.log_file = log_file
        self.lock = threading.Lock()
        self.logger = AdvancedLogger(__name__)
    
    def log_event(
        self,
        event_type: AuditEventType,
        operation: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            operation=operation,
            resource=resource,
            success=success,
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with self.lock:
            self.audit_events.append(event)
        
        # Log to file if configured
        if self.log_file:
            self._write_to_file(event)
        
        # Log to standard logger for important events
        if event_type in [AuditEventType.ACCESS_DENIED, AuditEventType.ERROR_OCCURRED]:
            self.logger.warning(f"Audit: {event_type.value} - {operation} by {user_id}")
    
    def _write_to_file(self, event: AuditEvent):
        """Write audit event to file."""
        try:
            with open(self.log_file, 'a') as f:
                event_data = {
                    'timestamp': event.timestamp,
                    'event_type': event.event_type.value,
                    'user_id': event.user_id,
                    'operation': event.operation,
                    'resource': event.resource,
                    'success': event.success,
                    'metadata': event.metadata,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent
                }
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def get_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        hours: Optional[float] = None
    ) -> List[AuditEvent]:
        """Get audit events with filtering."""
        with self.lock:
            events = list(self.audit_events)
        
        # Filter by user
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        # Filter by event type
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Filter by time
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            events = [e for e in events if e.timestamp >= cutoff_time]
        
        return events


class SecurityManager:
    """
    Comprehensive security management system for protein design operations.
    
    Integrates input validation, access control, rate limiting,
    and audit logging into a unified security framework.
    """
    
    def __init__(
        self,
        enable_rate_limiting: bool = True,
        enable_audit_logging: bool = True,
        audit_log_file: Optional[str] = None
    ):
        self.logger = AdvancedLogger(__name__)
        
        # Initialize components
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.access_control = AccessControl()
        self.audit_logger = AuditLogger(audit_log_file) if enable_audit_logging else None
        
        # Configuration
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_logging = enable_audit_logging
        
        self.logger.info("Security Manager initialized")
    
    def secure_operation(self, operation: str, required_level: SecurityLevel = SecurityLevel.AUTHENTICATED):
        """Decorator for securing operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user context (simplified - in real implementation would come from request context)
                user_id = kwargs.pop('user_id', None)
                session_token = kwargs.pop('session_token', None)
                
                # Authenticate user
                user = None
                if session_token:
                    user = self.access_control.validate_session(session_token)
                elif user_id:
                    user = self.access_control.authenticate_user(user_id)
                
                # Check permissions
                if not self.access_control.check_permission(user, operation):
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            AuditEventType.ACCESS_DENIED,
                            operation,
                            user_id=user_id,
                            success=False
                        )
                    raise PermissionError(f"Access denied for operation: {operation}")
                
                # Check rate limits
                if self.rate_limiter and user:
                    if not self.rate_limiter.is_allowed(user.user_id, operation):
                        if self.audit_logger:
                            self.audit_logger.log_event(
                                AuditEventType.ACCESS_DENIED,
                                operation,
                                user_id=user.user_id,
                                success=False,
                                metadata={'reason': 'rate_limit_exceeded'}
                            )
                        raise RuntimeError("Rate limit exceeded")
                
                # Validate inputs
                try:
                    validated_kwargs = self._validate_operation_inputs(operation, kwargs)
                except Exception as e:
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            AuditEventType.ERROR_OCCURRED,
                            operation,
                            user_id=user.user_id if user else None,
                            success=False,
                            metadata={'error': str(e)}
                        )
                    raise
                
                # Execute operation
                try:
                    result = func(*args, **validated_kwargs)
                    
                    # Log successful operation
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            AuditEventType.DATA_ACCESS if operation.startswith('get') else AuditEventType.DATA_MODIFICATION,
                            operation,
                            user_id=user.user_id if user else None,
                            success=True
                        )
                    
                    return result
                
                except Exception as e:
                    # Log failed operation
                    if self.audit_logger:
                        self.audit_logger.log_event(
                            AuditEventType.ERROR_OCCURRED,
                            operation,
                            user_id=user.user_id if user else None,
                            success=False,
                            metadata={'error': str(e)}
                        )
                    raise
            
            return wrapper
        return decorator
    
    def _validate_operation_inputs(self, operation: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs for specific operations."""
        validated = {}
        
        for key, value in kwargs.items():
            if key.endswith('_length') and isinstance(value, int):
                self.validator.validate_protein_length(value)
                validated[key] = value
            elif key == 'constraints':
                self.validator.validate_constraint_count(value)
                validated[key] = value
            elif hasattr(value, 'shape'):  # Tensor-like
                self.validator.validate_tensor_input(value)
                validated[key] = value
            elif isinstance(value, str):
                validated[key] = self.validator.sanitize_string(value)
            elif isinstance(value, dict):
                validated[key] = self.validator.validate_design_parameters(value)
            else:
                validated[key] = value
        
        return validated
    
    def create_user(
        self,
        username: str,
        email: str,
        security_level: SecurityLevel = SecurityLevel.AUTHENTICATED
    ) -> str:
        """Create a new user account."""
        user_id = self.access_control.create_user(username, email, security_level)
        
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.DATA_MODIFICATION,
                'create_user',
                metadata={'username': username, 'email': email}
            )
        
        return user_id
    
    def login_user(self, user_id: str) -> Optional[str]:
        """Login user and create session."""
        session_token = self.access_control.create_session(user_id)
        
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEventType.ACCESS_GRANTED if session_token else AuditEventType.ACCESS_DENIED,
                'login',
                user_id=user_id,
                success=session_token is not None
            )
        
        return session_token
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        dashboard = {
            'timestamp': time.time(),
            'total_users': len(self.access_control.users),
            'active_sessions': len(self.access_control.sessions),
        }
        
        if self.audit_logger:
            recent_events = self.audit_logger.get_events(hours=24)
            dashboard.update({
                'recent_events_24h': len(recent_events),
                'failed_operations_24h': len([e for e in recent_events if not e.success]),
                'access_denied_24h': len([e for e in recent_events if e.event_type == AuditEventType.ACCESS_DENIED])
            })
        
        if self.rate_limiter:
            # Rate limiting stats would be more complex in real implementation
            dashboard['rate_limiting_enabled'] = True
        
        return dashboard
