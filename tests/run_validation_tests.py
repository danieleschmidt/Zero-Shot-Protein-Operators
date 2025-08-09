#!/usr/bin/env python3
"""
Simple test runner for validation system without pytest dependency.

This runner executes all validation tests and reports results.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import test classes
try:
    from protein_operators.utils.validation import (
        ValidationResult, ValidationSeverity, BaseValidator,
        ConstraintValidator, StructureValidator, ModelValidator,
        validate_design_pipeline
    )
    from protein_operators.utils.error_recovery import (
        ProteinDesignError, ConstraintValidationError,
        ErrorRecoveryManager, RetryHandler, FallbackHandler
    )
    from protein_operators.utils.config_manager import (
        ProteinOperatorConfig, ModelConfig, ValidationConfig,
        ConfigManager, ModelConfigValidator
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class SimpleTestRunner:
    """Simple test runner without external dependencies."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        self.tests_run += 1
        try:
            test_func()
            self.tests_passed += 1
            print(f"✅ {test_name}")
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e), traceback.format_exc()))
            print(f"❌ {test_name}: {e}")
    
    def run_all_tests(self, test_classes):
        """Run all tests in test classes."""
        for test_class in test_classes:
            instance = test_class()
            class_name = test_class.__name__
            print(f"\n--- Running {class_name} ---")
            
            # Find all test methods
            test_methods = [method for method in dir(instance) 
                          if method.startswith('test_')]
            
            for method_name in test_methods:
                test_func = getattr(instance, method_name)
                full_name = f"{class_name}.{method_name}"
                self.run_test(test_func, full_name)
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\nFAILURES:")
            for name, error, trace in self.failures:
                print(f"\n{name}: {error}")
                if "--verbose" in sys.argv:
                    print(trace)
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        return self.tests_failed == 0


# Mock classes for testing (simplified versions)
class MockTensor:
    """Mock tensor for testing."""
    
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        self.shape = (len(self.data),)
    
    def numel(self):
        return len(self.data)
    
    def isnan(self):
        return MockTensor([False] * len(self.data))
    
    def isinf(self):
        return MockTensor([False] * len(self.data))
    
    def any(self):
        return any(self.data) if self.data else False
    
    def abs(self):
        return MockTensor([abs(x) for x in self.data])
    
    def max(self):
        return max(self.data) if self.data else 0


class MockConstraint:
    """Mock constraint for testing."""
    
    def __init__(self, has_methods=True, valid_params=True, encoding_works=True):
        self.has_methods = has_methods
        self.valid_params = valid_params
        self.encoding_works = encoding_works
        self.name = "test_constraint"
        
        if not has_methods:
            # Remove some methods to test error handling
            if hasattr(self, 'encode'):
                delattr(self, 'encode')
    
    def encode(self):
        if not self.encoding_works:
            raise ValueError("Encoding failed")
        return MockTensor([1, 2, 3])
    
    def validate(self, structure):
        return True
    
    def satisfaction_score(self, structure):
        return 0.8
    
    def get_constraint_type_id(self):
        return 1
    
    def validate_parameters(self):
        if not self.valid_params:
            raise ValueError("Invalid parameters")


class MockConstraints:
    """Mock constraints container."""
    
    def __init__(self, constraints=None):
        self.constraints = constraints or []


class MockStructure:
    """Mock structure for testing."""
    
    def __init__(self, coordinates=None, has_coordinates=True):
        if has_coordinates:
            self.coordinates = coordinates or MockTensor([1, 2, 3])


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, has_methods=True):
        self.has_methods = has_methods
        if has_methods:
            self.forward = lambda: None
            self.encode_constraints = lambda: None
            self.encode_coordinates = lambda: None


# Test classes (simplified versions)
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
    
    def test_validation_result_string_representation(self):
        """Test string representation."""
        result = ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All good",
            code="TEST_003"
        )
        
        result_str = str(result)
        assert "✅ PASS" in result_str
        assert "[INFO]" in result_str
        assert "All good" in result_str


class TestConstraintValidator:
    """Test ConstraintValidator class."""
    
    def test_validator_creation(self):
        """Test creating constraint validator."""
        validator = ConstraintValidator()
        assert validator.strict_mode is False
        assert validator.results == []
    
    def test_validate_empty_constraints(self):
        """Test validating empty constraints."""
        validator = ConstraintValidator()
        constraints = MockConstraints([])
        
        results = validator.validate(constraints)
        
        # Should have warning about no constraints
        warning_results = [r for r in results if r.severity == ValidationSeverity.WARNING]
        assert len(warning_results) >= 1
    
    def test_validate_valid_constraint(self):
        """Test validating valid constraint."""
        validator = ConstraintValidator()
        constraint = MockConstraint()
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have some results
        assert len(results) > 0
    
    def test_validate_constraint_invalid_params(self):
        """Test constraint with invalid parameters."""
        validator = ConstraintValidator()
        constraint = MockConstraint(valid_params=False)
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have some error results
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0


class TestStructureValidator:
    """Test StructureValidator class."""
    
    def test_validator_creation(self):
        """Test creating structure validator."""
        validator = StructureValidator()
        assert validator.strict_mode is False
    
    def test_validate_structure_missing_coordinates(self):
        """Test structure without coordinates."""
        validator = StructureValidator()
        structure = object()  # Object without coordinates
        
        results = validator.validate(structure)
        
        # Should have critical error
        critical_results = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical_results) > 0
    
    def test_validate_valid_structure(self):
        """Test valid structure."""
        validator = StructureValidator()
        structure = MockStructure()
        
        results = validator.validate(structure)
        
        # Should have some results
        assert len(results) > 0


class TestModelValidator:
    """Test ModelValidator class."""
    
    def test_validator_creation(self):
        """Test creating model validator."""
        validator = ModelValidator()
        assert validator.strict_mode is False
    
    def test_validate_model_with_methods(self):
        """Test model with required methods."""
        validator = ModelValidator()
        model = MockModel(has_methods=True)
        
        results = validator.validate(model)
        
        # Should have some successful validations
        assert len(results) > 0
    
    def test_validate_model_missing_methods(self):
        """Test model with missing methods."""
        validator = ModelValidator()
        model = MockModel(has_methods=False)
        
        results = validator.validate(model)
        
        # Should have error results
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0


class TestValidationPipeline:
    """Test complete validation pipeline."""
    
    def test_validate_design_pipeline_complete(self):
        """Test complete pipeline validation."""
        constraints = MockConstraints([MockConstraint()])
        model = MockModel(has_methods=True)
        structure = MockStructure()
        
        results = validate_design_pipeline(
            constraints=constraints,
            model=model,
            structure=structure
        )
        
        # Should have results for all components
        assert "constraints" in results
        assert "model" in results
        assert "structure" in results
        assert "overall" in results
    
    def test_validate_design_pipeline_no_structure(self):
        """Test pipeline without structure."""
        constraints = MockConstraints([MockConstraint()])
        model = MockModel(has_methods=True)
        
        results = validate_design_pipeline(
            constraints=constraints,
            model=model
        )
        
        # Should have empty structure results
        assert results["structure"] == []


def main():
    """Main test execution."""
    print("🧪 Running Protein Operators Validation Tests")
    print("=" * 50)
    
    # Test classes to run
    test_classes = [
        TestValidationResult,
        TestConstraintValidator,
        TestStructureValidator,
        TestModelValidator,
        TestValidationPipeline,
    ]
    
    # Create test runner and run tests
    runner = SimpleTestRunner()
    runner.run_all_tests(test_classes)
    
    # Print summary and return success status
    success = runner.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())