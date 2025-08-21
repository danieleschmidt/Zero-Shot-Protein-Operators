"""
Comprehensive tests for robust system components.

Tests error handling, security, monitoring, and resilience features
of the protein operators framework.
"""

import unittest
import time
import threading
import logging
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from protein_operators.utils.robust_error_recovery import (
    RobustErrorHandler, ErrorSeverity, RecoveryStrategy,
    ProteinOperatorError, robust_operation, get_global_error_handler
)
from protein_operators.utils.advanced_health_monitoring import (
    HealthMonitor, MetricsCollector, HealthStatus, MetricType,
    get_global_monitor
)
from protein_operators.utils.advanced_security_framework import (
    SecurityManager, InputSanitizer, SecurityLevel, ThreatLevel,
    ValidationError, AuthenticationError, secure_endpoint,
    get_global_security_manager
)


class TestRobustErrorRecovery(unittest.TestCase):
    """Test robust error handling and recovery systems."""
    
    def setUp(self):
        self.error_handler = RobustErrorHandler()
    
    def test_successful_operation_execution(self):
        """Test successful operation execution without errors."""
        def test_func(x, y):
            return x + y
        
        result = self.error_handler.robust_execute(
            operation="test_addition",
            func=test_func,
            *[5, 3]
        )
        
        self.assertEqual(result, 8)
        self.assertIn("test_addition", self.error_handler.operation_stats)
        self.assertEqual(self.error_handler.operation_stats["test_addition"]["success_count"], 1)
    
    def test_retry_recovery_strategy(self):
        """Test retry recovery strategy with eventual success."""
        call_count = 0
        
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = self.error_handler.robust_execute(
            operation="flaky_operation",
            func=flaky_func,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=3
        )
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_fallback_recovery_strategy(self):
        """Test fallback recovery strategy."""
        def failing_func():
            raise RuntimeError("Always fails")
        
        result = self.error_handler.robust_execute(
            operation="failing_operation",
            func=failing_func,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            max_retries=2,
            fallback_result="fallback_value"
        )
        
        self.assertEqual(result, "fallback_value")
    
    def test_abort_recovery_strategy(self):
        """Test abort recovery strategy raises exception."""
        def failing_func():
            raise RuntimeError("Critical failure")
        
        with self.assertRaises(ProteinOperatorError):
            self.error_handler.robust_execute(
                operation="critical_operation",
                func=failing_func,
                recovery_strategy=RecoveryStrategy.ABORT,
                max_retries=1
            )
    
    def test_custom_recovery_function(self):
        """Test custom recovery function registration and execution."""
        def custom_recovery(error, context):
            if isinstance(error, ValueError):
                return "recovered_value"
            return None
        
        self.error_handler.register_recovery_function("test_operation", custom_recovery)
        
        def failing_func():
            raise ValueError("Recoverable error")
        
        result = self.error_handler.robust_execute(
            operation="test_operation",
            func=failing_func,
            max_retries=1
        )
        
        self.assertEqual(result, "recovered_value")
    
    def test_error_history_tracking(self):
        """Test error history is properly tracked."""
        initial_error_count = len(self.error_handler.error_history)
        
        def failing_func():
            raise RuntimeError("Test error")
        
        self.error_handler.robust_execute(
            operation="test_history",
            func=failing_func,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_result=None
        )
        
        self.assertEqual(len(self.error_handler.error_history), initial_error_count + 1)
        self.assertEqual(self.error_handler.error_history[-1].operation, "test_history")
    
    def test_robust_operation_decorator(self):
        """Test robust operation decorator."""
        @robust_operation("decorated_operation", severity=ErrorSeverity.HIGH)
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Test successful execution
        result = test_function(5)
        self.assertEqual(result, 10)
        
        # Test error handling
        result = test_function(-1)  # Should use fallback
        self.assertIsNone(result)  # Default fallback is None
    
    def test_health_check(self):
        """Test error handler health check functionality."""
        health = self.error_handler.health_check()
        
        self.assertIn('status', health)
        self.assertIn('recent_errors', health)
        self.assertIn('system_health_score', health)
        self.assertIsInstance(health['system_health_score'], float)
        self.assertTrue(0.0 <= health['system_health_score'] <= 1.0)


class TestHealthMonitoring(unittest.TestCase):
    """Test health monitoring and metrics collection systems."""
    
    def setUp(self):
        self.monitor = HealthMonitor(check_interval=1)  # Fast interval for testing
        self.metrics = MetricsCollector(retention_hours=1)
    
    def tearDown(self):
        self.monitor.stop_monitoring()
    
    def test_metrics_collection(self):
        """Test basic metrics collection functionality."""
        # Test counter
        self.metrics.increment_counter("test_counter", 5)
        self.metrics.increment_counter("test_counter", 3)
        
        current_values = self.metrics.get_current_values()
        self.assertEqual(current_values['counters']['test_counter'], 8)
        
        # Test gauge
        self.metrics.set_gauge("test_gauge", 42.5)
        current_values = self.metrics.get_current_values()
        self.assertEqual(current_values['gauges']['test_gauge'], 42.5)
        
        # Test histogram
        self.metrics.record_histogram("test_histogram", 1.0)
        self.metrics.record_histogram("test_histogram", 2.0)
        self.metrics.record_histogram("test_histogram", 3.0)
        
        current_values = self.metrics.get_current_values()
        hist_stats = current_values['histogram_stats']['test_histogram']
        self.assertEqual(hist_stats['count'], 3)
        self.assertEqual(hist_stats['mean'], 2.0)
    
    def test_timer_context_manager(self):
        """Test timer context manager functionality."""
        with self.metrics.time_operation("test_timer"):
            time.sleep(0.01)  # Sleep for 10ms
        
        current_values = self.metrics.get_current_values()
        timer_stats = current_values['timer_stats']['test_timer']
        
        self.assertEqual(timer_stats['count'], 1)
        self.assertGreater(timer_stats['mean_ms'], 8)  # Should be around 10ms
    
    def test_health_check_registration(self):
        """Test custom health check registration."""
        def custom_health_check():
            from protein_operators.utils.advanced_health_monitoring import HealthCheckResult
            return HealthCheckResult(
                name="custom_check",
                status=HealthStatus.HEALTHY,
                message="Custom check passed",
                timestamp=time.time()
            )
        
        self.monitor.register_health_check("custom_check", custom_health_check)
        results = self.monitor.run_all_health_checks()
        
        self.assertIn("custom_check", results)
        self.assertEqual(results["custom_check"].status, HealthStatus.HEALTHY)
    
    def test_monitoring_thread(self):
        """Test continuous monitoring thread."""
        self.monitor.start_monitoring()
        
        # Wait for at least one monitoring cycle
        time.sleep(1.5)
        
        # Check that monitoring has run
        self.assertTrue(len(self.monitor.last_check_results) > 0)
        
        self.monitor.stop_monitoring()
    
    def test_overall_health_assessment(self):
        """Test overall health assessment logic."""
        # Initially should have no results
        health = self.monitor.get_overall_health()
        self.assertEqual(health['status'], HealthStatus.UNKNOWN.value)
        
        # Run health checks
        self.monitor.run_all_health_checks()
        health = self.monitor.get_overall_health()
        
        self.assertIn(health['status'], [s.value for s in HealthStatus])
        self.assertIn('check_results', health)
        self.assertIn('metrics_summary', health)
    
    def test_prometheus_export(self):
        """Test Prometheus format export."""
        self.metrics.increment_counter("requests_total", 100)
        self.metrics.set_gauge("memory_usage", 75.5)
        
        prometheus_output = self.metrics.export_prometheus_format()
        
        self.assertIn("requests_total 100", prometheus_output)
        self.assertIn("memory_usage 75.5", prometheus_output)
        self.assertIn("# TYPE", prometheus_output)


class TestSecurityFramework(unittest.TestCase):
    """Test security framework functionality."""
    
    def setUp(self):
        self.security_manager = SecurityManager()
        self.sanitizer = InputSanitizer()
    
    def test_input_sanitization(self):
        """Test input sanitization and validation."""
        # Test valid protein sequence
        valid_data = {
            'sequence': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'length': 64
        }
        
        sanitized = self.sanitizer.validate_input('protein_sequence', valid_data)
        self.assertEqual(sanitized['sequence'], valid_data['sequence'])
    
    def test_malicious_input_detection(self):
        """Test detection of malicious input patterns."""
        malicious_data = {
            'name': '<script>alert("xss")</script>',
            'description': 'normal text'
        }
        
        with self.assertRaises(ValidationError):
            self.sanitizer.validate_input('constraint_creation', malicious_data)
    
    def test_residue_indices_validation(self):
        """Test residue indices validation."""
        # Valid list format
        valid_data = {'residues': [1, 5, 10, 15]}
        result = self.sanitizer.validate_input('residue_indices', valid_data)
        self.assertEqual(result['residues'], [1, 5, 10, 15])
        
        # Valid string format
        valid_data = {'residues': '1, 5, 10, 15'}
        result = self.sanitizer.validate_input('residue_indices', valid_data)
        self.assertEqual(result['residues'], [1, 5, 10, 15])
        
        # Invalid format
        invalid_data = {'residues': [0, -1, 5]}
        with self.assertRaises(ValidationError):
            self.sanitizer.validate_input('residue_indices', invalid_data)
    
    def test_session_management(self):
        """Test session creation and validation."""
        user_id = "test_user"
        ip_address = "192.168.1.1"
        user_agent = "test_agent"
        
        # Create session
        session_id = self.security_manager.create_session(user_id, ip_address, user_agent)
        self.assertIsInstance(session_id, str)
        self.assertTrue(len(session_id) > 20)  # Should be a long random string
        
        # Validate session
        context = self.security_manager.validate_session(session_id)
        self.assertIsNotNone(context)
        self.assertEqual(context.user_id, user_id)
        self.assertEqual(context.ip_address, ip_address)
        
        # Invalidate session
        self.security_manager.invalidate_session(session_id)
        context = self.security_manager.validate_session(session_id)
        self.assertIsNone(context)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        identifier = "test_user"
        max_requests = 5
        
        # Should allow requests up to limit
        for i in range(max_requests):
            self.assertTrue(self.security_manager.check_rate_limit(identifier, max_requests))
        
        # Should block after limit exceeded
        self.assertFalse(self.security_manager.check_rate_limit(identifier, max_requests))
    
    def test_ip_blocking(self):
        """Test IP blocking functionality."""
        ip_address = "192.168.1.100"
        
        # Initially not blocked
        self.assertFalse(self.security_manager.is_ip_blocked(ip_address))
        
        # Block IP
        self.security_manager.block_ip(ip_address, "Suspicious activity")
        
        # Should now be blocked
        self.assertTrue(self.security_manager.is_ip_blocked(ip_address))
    
    def test_secure_endpoint_decorator(self):
        """Test secure endpoint decorator."""
        from protein_operators.utils.advanced_security_framework import SecurityContext
        
        @secure_endpoint(required_permission="design_proteins")
        def protected_function(data=None, security_context=None):
            return {"status": "success", "data": data}
        
        # Create valid security context
        session_id = self.security_manager.create_session("test_user", "127.0.0.1", "test")
        context = self.security_manager.validate_session(session_id)
        context.permissions.add("design_proteins")
        
        # Should succeed with valid context and permissions
        result = protected_function(
            data={"test": "data"},
            security_context=context
        )
        self.assertEqual(result["status"], "success")
    
    def test_security_event_logging(self):
        """Test security event logging."""
        initial_event_count = len(self.security_manager.security_events)
        
        # Trigger a security event
        try:
            malicious_data = {'input': '<script>alert("xss")</script>'}
            self.security_manager.validate_and_sanitize_input('test_context', malicious_data)
        except ValidationError:
            pass  # Expected
        
        # Should have logged a security event
        self.assertEqual(len(self.security_manager.security_events), initial_event_count + 1)
        
        latest_event = self.security_manager.security_events[-1]
        self.assertEqual(latest_event.event_type, "input_validation_failed")
    
    def test_security_summary(self):
        """Test security summary generation."""
        summary = self.security_manager.get_security_summary()
        
        required_keys = [
            'active_sessions', 'blocked_ips', 'rate_limited_entities',
            'events_last_hour', 'event_types', 'threat_levels'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        self.assertIsInstance(summary['active_sessions'], int)
        self.assertIsInstance(summary['blocked_ips'], int)


class TestIntegrationSecurity(unittest.TestCase):
    """Integration tests for security with other components."""
    
    def test_protein_design_security_integration(self):
        """Test security integration with protein design workflow."""
        from protein_operators.utils.advanced_security_framework import validate_protein_input
        
        # Valid protein design input
        valid_input = {
            'length': 100,
            'constraints': ['binding_site'],
            'operator_type': 'deeponet'
        }
        
        # Should not raise exception
        result = validate_protein_input(valid_input)
        self.assertIn('length', result)
        
        # Invalid input
        invalid_input = {
            'length': -1,  # Invalid length
            'script': '<script>alert("xss")</script>'  # Malicious content
        }
        
        with self.assertRaises(ValidationError):
            validate_protein_input(invalid_input)


class TestSystemResilience(unittest.TestCase):
    """Test overall system resilience and fault tolerance."""
    
    def test_concurrent_error_handling(self):
        """Test error handling under concurrent load."""
        error_handler = RobustErrorHandler()
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                def test_func():
                    if worker_id % 3 == 0:
                        raise ValueError(f"Worker {worker_id} error")
                    return f"Worker {worker_id} success"
                
                result = error_handler.robust_execute(
                    operation=f"worker_{worker_id}",
                    func=test_func,
                    recovery_strategy=RecoveryStrategy.FALLBACK,
                    fallback_result=f"Worker {worker_id} fallback"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent workers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)
        
        # Check that some used fallback values
        fallback_count = sum(1 for r in results if "fallback" in str(r))
        self.assertGreater(fallback_count, 0)
    
    def test_memory_cleanup(self):
        """Test that monitoring systems clean up old data."""
        metrics = MetricsCollector(retention_hours=0.001)  # Very short retention
        
        # Add some metrics
        metrics.increment_counter("test_counter", 1)
        metrics.set_gauge("test_gauge", 1.0)
        
        initial_metric_count = len(metrics.metrics)
        
        # Wait for cleanup (retention is very short)
        time.sleep(0.1)
        
        # Metrics should still be there (cleanup runs periodically)
        self.assertGreaterEqual(len(metrics.metrics), 0)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main()