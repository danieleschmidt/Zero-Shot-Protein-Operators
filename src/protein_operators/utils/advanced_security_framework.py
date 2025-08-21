"""
Advanced security framework for protein operators.

This module provides comprehensive security measures including:
- Input validation and sanitization
- Access control and authentication
- Security monitoring and threat detection
- Data protection and encryption
- Audit logging and compliance
"""

import hashlib
import hmac
import time
import logging
import re
import json
import secrets
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import base64
import sys
import os

# Handle import compatibility
try:
    import torch
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    import mock_torch as torch


class SecurityLevel(Enum):
    """Security levels for operations and data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    permissions: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    threat_level: ThreatLevel
    description: str
    context: SecurityContext
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Input validation rule."""
    field_name: str
    rule_type: str  # regex, length, type, range, custom
    rule_value: Any
    error_message: str
    required: bool = True


class SecurityError(Exception):
    """Base security exception."""
    pass


class ValidationError(SecurityError):
    """Input validation error."""
    pass


class AuthenticationError(SecurityError):
    """Authentication error."""
    pass


class AuthorizationError(SecurityError):
    """Authorization error."""
    pass


class InputSanitizer:
    """
    Advanced input sanitization and validation system.
    
    Provides comprehensive protection against various injection attacks
    and malformed input data for protein design operations.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.whitelist_patterns: Dict[str, re.Pattern] = {}
        self.blacklist_patterns: List[re.Pattern] = []
        
        # Setup default patterns
        self._setup_default_patterns()
        self._setup_default_validation_rules()
    
    def _setup_default_patterns(self):
        """Setup default whitelist and blacklist patterns."""
        
        # Whitelist patterns for common fields
        self.whitelist_patterns.update({
            'protein_sequence': re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$', re.IGNORECASE),
            'residue_indices': re.compile(r'^\d+(,\s*\d+)*$'),
            'ligand_name': re.compile(r'^[a-zA-Z0-9_\-\s]+$'),
            'constraint_name': re.compile(r'^[a-zA-Z0-9_\-\s]{1,100}$'),
            'file_path': re.compile(r'^[a-zA-Z0-9_\-\.\/]+$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'uuid': re.compile(r'^[a-fA-F0-9\-]{36}$'),
            'float_value': re.compile(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$'),
            'integer_value': re.compile(r'^-?\d+$'),
        })
        
        # Blacklist patterns for common attacks
        self.blacklist_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),  # XSS
            re.compile(r'javascript:', re.IGNORECASE),  # JavaScript injection
            re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
            re.compile(r'(union|select|insert|update|delete|drop|create|alter)\s', re.IGNORECASE),  # SQL injection
            re.compile(r'[;&|`]', re.IGNORECASE),  # Command injection
            re.compile(r'\.\./', re.IGNORECASE),  # Path traversal
            re.compile(r'\\x[0-9a-fA-F]{2}'),  # Hex encoding
            re.compile(r'%[0-9a-fA-F]{2}'),  # URL encoding of suspicious chars
        ]
    
    def _setup_default_validation_rules(self):
        """Setup default validation rules for common protein design parameters."""
        
        # Protein sequence validation
        self.add_validation_rule(
            'protein_sequence',
            ValidationRule(
                field_name='sequence',
                rule_type='regex',
                rule_value=self.whitelist_patterns['protein_sequence'],
                error_message='Invalid protein sequence - only standard amino acid codes allowed',
                required=False
            )
        )
        
        # Residue indices validation
        self.add_validation_rule(
            'residue_indices',
            ValidationRule(
                field_name='residues',
                rule_type='custom',
                rule_value=self._validate_residue_indices,
                error_message='Invalid residue indices format',
                required=True
            )
        )
        
        # Constraint parameters
        self.add_validation_rule(
            'constraint_weight',
            ValidationRule(
                field_name='weight',
                rule_type='range',
                rule_value=(0.0, 10.0),
                error_message='Constraint weight must be between 0.0 and 10.0',
                required=False
            )
        )
        
        # Affinity values
        self.add_validation_rule(
            'affinity_nm',
            ValidationRule(
                field_name='affinity_nm',
                rule_type='range',
                rule_value=(0.001, 1000000.0),
                error_message='Affinity must be between 0.001 nM and 1,000,000 nM',
                required=False
            )
        )
        
        # Protein length
        self.add_validation_rule(
            'protein_length',
            ValidationRule(
                field_name='length',
                rule_type='range',
                rule_value=(1, 5000),
                error_message='Protein length must be between 1 and 5000 residues',
                required=True
            )
        )
    
    def add_validation_rule(self, context: str, rule: ValidationRule):
        """Add a validation rule for a specific context."""
        if context not in self.validation_rules:
            self.validation_rules[context] = []
        self.validation_rules[context].append(rule)
    
    def validate_input(self, context: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against registered rules.
        
        Args:
            context: Validation context (e.g., 'constraint_creation')
            data: Input data to validate
            
        Returns:
            Sanitized and validated data
            
        Raises:
            ValidationError: If validation fails
        """
        validated_data = {}
        rules = self.validation_rules.get(context, [])
        
        # Check for malicious patterns first
        self._check_malicious_patterns(data)
        
        # Apply validation rules
        for rule in rules:
            field_value = data.get(rule.field_name)
            
            # Check if required field is missing
            if rule.required and field_value is None:
                raise ValidationError(f"Required field '{rule.field_name}' is missing")
            
            # Skip validation if field is not required and not present
            if not rule.required and field_value is None:
                continue
            
            # Apply specific validation
            try:
                validated_value = self._apply_validation_rule(rule, field_value)
                validated_data[rule.field_name] = validated_value
            except Exception as e:
                raise ValidationError(f"{rule.error_message}: {str(e)}")
        
        # Copy over non-validated fields (with basic sanitization)
        for key, value in data.items():
            if key not in validated_data:
                validated_data[key] = self._basic_sanitize(value)
        
        return validated_data
    
    def _check_malicious_patterns(self, data: Dict[str, Any]):
        """Check for malicious patterns in input data."""
        def check_value(value):
            if isinstance(value, str):
                for pattern in self.blacklist_patterns:
                    if pattern.search(value):
                        raise ValidationError(f"Potentially malicious input detected: {pattern.pattern}")
            elif isinstance(value, dict):
                for v in value.values():
                    check_value(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    check_value(v)
        
        check_value(data)
    
    def _apply_validation_rule(self, rule: ValidationRule, value: Any) -> Any:
        """Apply a specific validation rule to a value."""
        
        if rule.rule_type == 'regex':
            if not isinstance(value, str):
                raise ValueError("Regex validation requires string input")
            if not rule.rule_value.match(str(value)):
                raise ValueError("Value does not match required pattern")
            return value
        
        elif rule.rule_type == 'length':
            if hasattr(value, '__len__'):
                length = len(value)
                min_len, max_len = rule.rule_value
                if not (min_len <= length <= max_len):
                    raise ValueError(f"Length {length} not in range [{min_len}, {max_len}]")
            return value
        
        elif rule.rule_type == 'type':
            expected_type = rule.rule_value
            if not isinstance(value, expected_type):
                raise ValueError(f"Expected type {expected_type.__name__}, got {type(value).__name__}")
            return value
        
        elif rule.rule_type == 'range':
            if not isinstance(value, (int, float)):
                raise ValueError("Range validation requires numeric input")
            min_val, max_val = rule.rule_value
            if not (min_val <= value <= max_val):
                raise ValueError(f"Value {value} not in range [{min_val}, {max_val}]")
            return value
        
        elif rule.rule_type == 'custom':
            return rule.rule_value(value)
        
        else:
            raise ValueError(f"Unknown validation rule type: {rule.rule_type}")
    
    def _validate_residue_indices(self, value: Any) -> List[int]:
        """Custom validation for residue indices."""
        if isinstance(value, (list, tuple)):
            residues = []
            for item in value:
                if not isinstance(item, int) or item < 1:
                    raise ValueError("Residue indices must be positive integers")
                residues.append(item)
            return residues
        elif isinstance(value, str):
            # Parse comma-separated string
            try:
                residues = [int(x.strip()) for x in value.split(',')]
                for r in residues:
                    if r < 1:
                        raise ValueError("Residue indices must be positive")
                return residues
            except ValueError as e:
                raise ValueError(f"Invalid residue indices format: {e}")
        else:
            raise ValueError("Residue indices must be a list or comma-separated string")
    
    def _basic_sanitize(self, value: Any) -> Any:
        """Apply basic sanitization to values."""
        if isinstance(value, str):
            # Remove null bytes and control characters
            sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
            # Limit length
            return sanitized[:10000]  # Max 10k characters
        return value
    
    def sanitize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for safe logging (remove sensitive information)."""
        sanitized = {}
        sensitive_fields = {'password', 'token', 'api_key', 'secret', 'private_key'}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + '...[TRUNCATED]'
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_for_logging(value)
            else:
                sanitized[key] = value
        
        return sanitized


class SecurityManager:
    """
    Advanced security management system.
    
    Provides authentication, authorization, audit logging,
    and security monitoring capabilities.
    """
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.security_events: List[SecurityEvent] = []
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        self.permissions: Dict[str, Set[str]] = {}
        self.sessions: Dict[str, SecurityContext] = {}
        
        # Security configuration
        self.config = {
            'rate_limit_window': 300,  # 5 minutes
            'max_requests_per_window': 100,
            'session_timeout': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'audit_log_retention_days': 90,
        }
        
        self.logger = self._setup_security_logger()
    
    def _setup_security_logger(self) -> logging.Logger:
        """Setup dedicated security logger."""
        logger = logging.getLogger('protein_operators.security')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new secure session."""
        session_id = secrets.token_urlsafe(32)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=self.permissions.get(user_id, set())
        )
        
        self.sessions[session_id] = context
        
        self._log_security_event(
            event_type="session_created",
            threat_level=ThreatLevel.LOW,
            description=f"Session created for user {user_id}",
            context=context
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate and return session context."""
        context = self.sessions.get(session_id)
        
        if context is None:
            return None
        
        # Check session timeout
        if time.time() - context.timestamp > self.config['session_timeout']:
            self.invalidate_session(session_id)
            return None
        
        return context
    
    def invalidate_session(self, session_id: str):
        """Invalidate a session."""
        context = self.sessions.pop(session_id, None)
        if context:
            self._log_security_event(
                event_type="session_invalidated",
                threat_level=ThreatLevel.LOW,
                description=f"Session invalidated for user {context.user_id}",
                context=context
            )
    
    def check_permissions(self, context: SecurityContext, required_permission: str) -> bool:
        """Check if context has required permission."""
        return required_permission in context.permissions
    
    def check_rate_limit(self, identifier: str, max_requests: Optional[int] = None) -> bool:
        """Check rate limiting for an identifier (IP, user, etc.)."""
        max_requests = max_requests or self.config['max_requests_per_window']
        window = self.config['rate_limit_window']
        current_time = time.time()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {
                'requests': [],
                'blocked_until': 0
            }
        
        rate_data = self.rate_limits[identifier]
        
        # Check if still blocked
        if current_time < rate_data['blocked_until']:
            return False
        
        # Clean old requests outside window
        rate_data['requests'] = [
            req_time for req_time in rate_data['requests']
            if current_time - req_time < window
        ]
        
        # Check if limit exceeded
        if len(rate_data['requests']) >= max_requests:
            rate_data['blocked_until'] = current_time + self.config['lockout_duration']
            self._log_security_event(
                event_type="rate_limit_exceeded",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Rate limit exceeded for {identifier}",
                context=SecurityContext(
                    user_id="system",
                    session_id="",
                    ip_address=identifier,
                    user_agent=""
                ),
                details={'requests_in_window': len(rate_data['requests'])}
            )
            return False
        
        # Record this request
        rate_data['requests'].append(current_time)
        return True
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str, reason: str):
        """Block an IP address."""
        self.blocked_ips.add(ip_address)
        self._log_security_event(
            event_type="ip_blocked",
            threat_level=ThreatLevel.HIGH,
            description=f"IP {ip_address} blocked: {reason}",
            context=SecurityContext(
                user_id="system",
                session_id="",
                ip_address=ip_address,
                user_agent=""
            )
        )
    
    def validate_and_sanitize_input(self, context: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input data."""
        try:
            return self.sanitizer.validate_input(context, data)
        except ValidationError as e:
            self._log_security_event(
                event_type="input_validation_failed",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Input validation failed: {str(e)}",
                context=SecurityContext(
                    user_id="unknown",
                    session_id="",
                    ip_address="unknown",
                    user_agent=""
                ),
                details={'validation_context': context, 'error': str(e)}
            )
            raise
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, 
                           description: str, context: SecurityContext,
                           details: Dict[str, Any] = None):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            context=context,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Log to security logger
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }[threat_level]
        
        sanitized_details = self.sanitizer.sanitize_for_logging(event.details)
        self.logger.log(
            log_level,
            f"{event_type}: {description} | User: {context.user_id} | "
            f"IP: {context.ip_address} | Details: {json.dumps(sanitized_details)}"
        )
        
        # Cleanup old events
        cutoff_time = time.time() - (self.config['audit_log_retention_days'] * 24 * 3600)
        self.security_events = [
            e for e in self.security_events if e.timestamp > cutoff_time
        ]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary and statistics."""
        recent_events = [
            e for e in self.security_events
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        event_counts = {}
        threat_counts = {}
        
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            threat_counts[event.threat_level.value] = threat_counts.get(event.threat_level.value, 0) + 1
        
        return {
            'active_sessions': len(self.sessions),
            'blocked_ips': len(self.blocked_ips),
            'rate_limited_entities': len(self.rate_limits),
            'events_last_hour': len(recent_events),
            'event_types': event_counts,
            'threat_levels': threat_counts,
            'high_threat_events': [
                {
                    'type': e.event_type,
                    'description': e.description,
                    'timestamp': e.timestamp
                }
                for e in recent_events
                if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
        }


def secure_endpoint(
    required_permission: str = None,
    security_level: SecurityLevel = SecurityLevel.INTERNAL,
    rate_limit: int = None,
    validate_input: str = None
):
    """
    Decorator for securing API endpoints.
    
    Args:
        required_permission: Permission required to access endpoint
        security_level: Minimum security level required
        rate_limit: Custom rate limit for this endpoint
        validate_input: Input validation context to apply
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get security manager (would be injected in real implementation)
            security_manager = get_global_security_manager()
            
            # Extract security context from request (simplified)
            context = kwargs.get('security_context')
            if not context:
                raise AuthenticationError("No security context provided")
            
            # Validate session
            if not security_manager.validate_session(context.session_id):
                raise AuthenticationError("Invalid or expired session")
            
            # Check IP blocking
            if security_manager.is_ip_blocked(context.ip_address):
                raise AuthorizationError("IP address is blocked")
            
            # Check rate limiting
            rate_limit_key = f"{context.user_id}:{func.__name__}"
            if not security_manager.check_rate_limit(rate_limit_key, rate_limit):
                raise AuthorizationError("Rate limit exceeded")
            
            # Check permissions
            if required_permission and not security_manager.check_permissions(context, required_permission):
                raise AuthorizationError(f"Missing required permission: {required_permission}")
            
            # Validate input if specified
            if validate_input and 'data' in kwargs:
                kwargs['data'] = security_manager.validate_and_sanitize_input(
                    validate_input, kwargs['data']
                )
            
            # Log access
            security_manager._log_security_event(
                event_type="endpoint_access",
                threat_level=ThreatLevel.LOW,
                description=f"Endpoint {func.__name__} accessed",
                context=context
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_protein_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate protein design input."""
    security_manager = get_global_security_manager()
    return security_manager.validate_and_sanitize_input('protein_design', input_data)


def validate_constraint_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate constraint input."""
    security_manager = get_global_security_manager()
    return security_manager.validate_and_sanitize_input('constraint_creation', input_data)


# Global security manager instance
_global_security_manager = SecurityManager()


def get_global_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    return _global_security_manager


def set_global_security_manager(manager: SecurityManager):
    """Set the global security manager instance."""
    global _global_security_manager
    _global_security_manager = manager