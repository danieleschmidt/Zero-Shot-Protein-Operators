#!/usr/bin/env python3
"""
Standalone validation tests without torch dependencies.

Tests core validation logic independently.
"""

import sys
import traceback
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """
    Result of a validation check.
    """
    passed: bool
    severity: ValidationSeverity
    message: str
    code: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} [{self.severity.value.upper()}] {self.message}"


class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
    
    @abstractmethod
    def validate(self, target: Any) -> List[ValidationResult]:
        """Validate a target object."""
        pass
    
    def add_result(
        self,
        passed: bool,
        severity: ValidationSeverity,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ) -> None:
        """Add a validation result."""
        result = ValidationResult(
            passed=passed,
            severity=severity,
            message=message,
            code=code,
            details=details,
            suggestions=suggestions
        )
        self.results.append(result)
    
    def has_errors(self) -> bool:
        """Check if any validation errors were found."""
        error_severities = {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
        return any(r.severity in error_severities for r in self.results)
    
    def has_warnings(self) -> bool:
        """Check if any validation warnings were found."""
        return any(r.severity == ValidationSeverity.WARNING for r in self.results)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "errors": sum(1 for r in self.results if r.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}),
            "warnings": sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING),
            "overall_success": not self.has_errors()
        }


class SimpleConstraintValidator(BaseValidator):
    """Simple constraint validator for testing."""
    
    def validate(self, constraints) -> List[ValidationResult]:
        """Validate constraint collection."""
        self.results = []
        
        if not hasattr(constraints, 'constraints'):
            self.add_result(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="Invalid constraints object: missing 'constraints' attribute",
                code="CONST_001"
            )
            return self.results
        
        # Check if constraints are empty
        if not constraints.constraints:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="No constraints specified",
                code="CONST_002",
                suggestions=["Add at least one constraint"]
            )
        else:
            # Validate individual constraints
            for i, constraint in enumerate(constraints.constraints):
                self._validate_constraint(constraint, i)
        
        return self.results
    
    def _validate_constraint(self, constraint, index: int) -> None:
        """Validate individual constraint."""
        # Check required methods
        required_methods = ['encode', 'validate', 'satisfaction_score']
        for method in required_methods:
            if hasattr(constraint, method):
                self.add_result(
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Constraint {index}: has method '{method}'",
                    code="CONST_003"
                )
            else:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Constraint {index}: missing method '{method}'",
                    code="CONST_004"
                )


# Mock classes for testing
class MockTensor:
    """Mock tensor for testing."""
    
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        self.shape = (len(self.data),)
    
    def numel(self):
        return len(self.data)


class MockConstraint:
    """Mock constraint for testing."""
    
    def __init__(self, has_methods=True):
        self.name = "test_constraint"
        self.has_methods = has_methods
        
        if not has_methods:
            # Remove methods to test error handling
            if hasattr(self, 'encode'):
                delattr(self, 'encode')
    
    def encode(self):
        return MockTensor([1, 2, 3])
    
    def validate(self, structure):
        return True
    
    def satisfaction_score(self, structure):
        return 0.8


class MockConstraints:
    """Mock constraints container."""
    
    def __init__(self, constraints=None):
        self.constraints = constraints or []


# Test classes
class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Test passed",
            code="TEST_001"
        )
        
        assert result.passed is True
        assert result.severity == ValidationSeverity.INFO
        assert result.message == "Test passed"
        assert result.code == "TEST_001"
        print("✓ ValidationResult creation test passed")
    
    def test_validation_result_string(self):
        """Test string representation."""
        result = ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All good",
            code="TEST_002"
        )
        
        result_str = str(result)
        assert "✅ PASS" in result_str
        assert "[INFO]" in result_str
        assert "All good" in result_str
        print("✓ ValidationResult string representation test passed")


class TestConstraintValidator:
    """Test constraint validator."""
    
    def test_validator_creation(self):
        """Test creating validator."""
        validator = SimpleConstraintValidator()
        assert validator.strict_mode is False
        assert validator.results == []
        print("✓ Validator creation test passed")
    
    def test_validate_empty_constraints(self):
        """Test empty constraints."""
        validator = SimpleConstraintValidator()
        constraints = MockConstraints([])
        
        results = validator.validate(constraints)
        
        # Should have warning about no constraints
        warning_results = [r for r in results if r.severity == ValidationSeverity.WARNING]
        assert len(warning_results) >= 1
        print("✓ Empty constraints test passed")
    
    def test_validate_valid_constraint(self):
        """Test valid constraint."""
        validator = SimpleConstraintValidator()
        constraint = MockConstraint(has_methods=True)
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have some successful validations
        success_results = [r for r in results if r.passed]
        assert len(success_results) > 0
        print("✓ Valid constraint test passed")
    
    def test_validate_constraint_missing_methods(self):
        """Test constraint missing methods."""
        validator = SimpleConstraintValidator()
        
        # Create constraint without methods
        constraint = object()  # Plain object without methods
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have errors for missing methods
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0
        print("✓ Missing methods test passed")
    
    def test_validator_utilities(self):
        """Test validator utility methods."""
        validator = SimpleConstraintValidator()
        
        # Add some results
        validator.add_result(
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Info message",
            code="INFO_001"
        )
        
        validator.add_result(
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Warning message",
            code="WARN_001"
        )
        
        validator.add_result(
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="Error message",
            code="ERR_001"
        )
        
        # Test utility methods
        assert len(validator.results) == 3
        assert validator.has_warnings()
        assert validator.has_errors()
        
        summary = validator.get_summary()
        assert summary["total_checks"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2
        assert summary["errors"] == 1
        assert summary["warnings"] == 1
        assert summary["overall_success"] is False
        
        print("✓ Validator utilities test passed")


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def run_test(self, test_func, test_name):
        """Run a single test."""
        self.tests_run += 1
        try:
            test_func()
            self.tests_passed += 1
            print(f"✅ {test_name}")
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
    
    def run_all_tests(self, test_classes):
        """Run all tests."""
        for test_class in test_classes:
            instance = test_class()
            class_name = test_class.__name__
            print(f"\n--- Running {class_name} ---")
            
            # Find test methods
            test_methods = [method for method in dir(instance) 
                          if method.startswith('test_')]
            
            for method_name in test_methods:
                test_func = getattr(instance, method_name)
                full_name = f"{class_name}.{method_name}"
                self.run_test(test_func, full_name)
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*50}")
        print(f"VALIDATION SYSTEM TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\nFAILURES:")
            for name, error in self.failures:
                print(f"  {name}: {error}")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        return self.tests_failed == 0


def main():
    """Main test execution."""
    print("🧪 Running Standalone Validation Tests")
    print("=" * 50)
    
    # Test classes to run
    test_classes = [
        TestValidationResult,
        TestConstraintValidator,
    ]
    
    # Create test runner and run tests
    runner = TestRunner()
    runner.run_all_tests(test_classes)
    
    # Print summary
    success = runner.print_summary()
    
    if success:
        print("\n🎉 All tests passed! Validation system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the failures above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())