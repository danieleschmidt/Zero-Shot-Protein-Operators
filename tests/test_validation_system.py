"""
Comprehensive tests for the validation system.

Tests all validation components including constraint validation,
structure validation, and model validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
        assert result.details is None
        assert result.suggestions is None
    
    def test_validation_result_with_details(self):
        """Test validation result with details and suggestions."""
        details = {"key": "value"}
        suggestions = ["Fix this", "Try that"]
        
        result = ValidationResult(
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="Test failed",
            code="TEST_002",
            details=details,
            suggestions=suggestions
        )
        
        assert result.details == details
        assert result.suggestions == suggestions
    
    def test_validation_result_string_representation(self):
        """Test string representation of validation results."""
        result = ValidationResult(
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All good",
            code="TEST_003"
        )
        
        assert "✅ PASS" in str(result)
        assert "[INFO]" in str(result)
        assert "All good" in str(result)
        
        result.passed = False
        result.severity = ValidationSeverity.ERROR
        
        assert "❌ FAIL" in str(result)
        assert "[ERROR]" in str(result)


class MockConstraints:
    """Mock constraints object for testing."""
    
    def __init__(self, constraints=None):
        self.constraints = constraints or []


class MockConstraint:
    """Mock constraint for testing."""
    
    def __init__(self, has_methods=True, valid_params=True, encoding_works=True):
        self.has_methods = has_methods
        self.valid_params = valid_params
        self.encoding_works = encoding_works
        self.name = "test_constraint"
    
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


class MockTensor:
    """Mock tensor for testing."""
    
    def __init__(self, data):
        self.data = data
        self.shape = (len(data),) if isinstance(data, list) else data.shape
    
    def numel(self):
        return len(self.data) if isinstance(self.data, list) else self.data.numel()


class MockStructure:
    """Mock structure for testing."""
    
    def __init__(self, coordinates=None, has_coordinates=True):
        if has_coordinates:
            self.coordinates = coordinates or MockTensor([1, 2, 3])
        else:
            pass  # No coordinates attribute
    
    def compute_radius_of_gyration(self):
        return 10.0


class TestConstraintValidator:
    """Test ConstraintValidator class."""
    
    def test_validator_creation(self):
        """Test creating constraint validator."""
        validator = ConstraintValidator()
        assert validator.strict_mode is False
        assert validator.results == []
        
        validator = ConstraintValidator(strict_mode=True)
        assert validator.strict_mode is True
    
    def test_validate_empty_constraints(self):
        """Test validating empty constraints."""
        validator = ConstraintValidator()
        constraints = MockConstraints([])
        
        results = validator.validate(constraints)
        
        # Should have warning about no constraints
        warning_results = [r for r in results if r.severity == ValidationSeverity.WARNING]
        assert len(warning_results) >= 1
        assert any("No constraints" in r.message for r in warning_results)
    
    def test_validate_valid_constraint(self):
        """Test validating valid constraint."""
        validator = ConstraintValidator()
        constraint = MockConstraint()
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have some successful validations
        success_results = [r for r in results if r.passed]
        assert len(success_results) > 0
    
    def test_validate_constraint_missing_methods(self):
        """Test validating constraint with missing methods."""
        validator = ConstraintValidator()
        constraint = Mock()  # Mock without required methods
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have errors for missing methods
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0
        assert any("missing required method" in r.message for r in error_results)
    
    def test_validate_constraint_invalid_params(self):
        """Test validating constraint with invalid parameters."""
        validator = ConstraintValidator()
        constraint = MockConstraint(valid_params=False)
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have error for invalid parameters
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert any("invalid parameters" in r.message for r in error_results)
    
    def test_validate_constraint_encoding_failure(self):
        """Test validating constraint with encoding failure."""
        validator = ConstraintValidator()
        constraint = MockConstraint(encoding_works=False)
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should have error for encoding failure
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert any("encoding failed" in r.message for r in error_results)
    
    def test_validate_invalid_constraints_object(self):
        """Test validating invalid constraints object."""
        validator = ConstraintValidator()
        constraints = Mock()  # Mock without constraints attribute
        
        results = validator.validate(constraints)
        
        # Should have critical error
        critical_results = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical_results) > 0
        assert any("Invalid constraints object" in r.message for r in critical_results)


class TestStructureValidator:
    """Test StructureValidator class."""
    
    def test_validator_creation(self):
        """Test creating structure validator."""
        validator = StructureValidator()
        assert validator.strict_mode is False
        assert validator.results == []
    
    def test_validate_structure_missing_coordinates(self):
        """Test validating structure without coordinates."""
        validator = StructureValidator()
        structure = MockStructure(has_coordinates=False)
        
        results = validator.validate(structure)
        
        # Should have critical error
        critical_results = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical_results) > 0
        assert any("missing coordinates" in r.message for r in critical_results)
    
    def test_validate_valid_structure(self):
        """Test validating valid structure."""
        validator = StructureValidator()
        structure = MockStructure()
        
        results = validator.validate(structure)
        
        # Should have some successful validations
        success_results = [r for r in results if r.passed]
        assert len(success_results) > 0


class TestModelValidator:
    """Test ModelValidator class."""
    
    def test_validator_creation(self):
        """Test creating model validator."""
        validator = ModelValidator()
        assert validator.strict_mode is False
        assert validator.results == []
    
    def test_validate_model_with_methods(self):
        """Test validating model with required methods."""
        validator = ModelValidator()
        model = Mock()
        model.forward = Mock()
        model.encode_constraints = Mock()
        model.encode_coordinates = Mock()
        
        results = validator.validate(model)
        
        # Should have successful validations for each method
        success_results = [r for r in results if r.passed]
        assert len(success_results) >= 3  # One for each required method
    
    def test_validate_model_missing_methods(self):
        """Test validating model with missing methods."""
        validator = ModelValidator()
        model = Mock()  # Mock without required methods
        
        results = validator.validate(model)
        
        # Should have errors for missing methods
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) >= 3  # One for each missing method


class TestValidationPipeline:
    """Test complete validation pipeline."""
    
    def test_validate_design_pipeline_complete(self):
        """Test validating complete design pipeline."""
        # Create mock objects
        constraints = MockConstraints([MockConstraint()])
        model = Mock()
        model.forward = Mock()
        model.encode_constraints = Mock()
        model.encode_coordinates = Mock()
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
        
        # Overall summary should be present
        overall = results["overall"]
        assert "total_checks" in overall
        assert "passed" in overall
        assert "failed" in overall
        assert "success" in overall
    
    def test_validate_design_pipeline_no_structure(self):
        """Test validating pipeline without structure."""
        constraints = MockConstraints([MockConstraint()])
        model = Mock()
        model.forward = Mock()
        model.encode_constraints = Mock()
        model.encode_coordinates = Mock()
        
        results = validate_design_pipeline(
            constraints=constraints,
            model=model
        )
        
        # Should have empty structure results
        assert results["structure"] == []


class TestErrorHandling:
    """Test error handling in validation."""
    
    def test_constraint_validator_handles_exceptions(self):
        """Test that constraint validator handles exceptions gracefully."""
        validator = ConstraintValidator()
        
        # Create constraint that raises exception
        constraint = Mock()
        constraint.encode.side_effect = RuntimeError("Test error")
        constraints = MockConstraints([constraint])
        
        results = validator.validate(constraints)
        
        # Should handle exception gracefully
        assert len(results) > 0
        # Should have captured the error
        error_results = [r for r in results if not r.passed]
        assert len(error_results) > 0


class TestValidatorUtilities:
    """Test validator utility methods."""
    
    def test_base_validator_add_result(self):
        """Test adding validation results."""
        validator = ConstraintValidator()
        
        validator.add_result(
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Test message",
            code="TEST_001"
        )
        
        assert len(validator.results) == 1
        result = validator.results[0]
        assert result.passed is True
        assert result.message == "Test message"
    
    def test_base_validator_has_errors(self):
        """Test checking for errors."""
        validator = ConstraintValidator()
        
        # No errors initially
        assert not validator.has_errors()
        
        # Add warning - should not count as error
        validator.add_result(
            passed=False,
            severity=ValidationSeverity.WARNING,
            message="Warning",
            code="W001"
        )
        assert not validator.has_errors()
        
        # Add error - should count
        validator.add_result(
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="Error",
            code="E001"
        )
        assert validator.has_errors()
    
    def test_base_validator_get_summary(self):
        """Test getting validation summary."""
        validator = ConstraintValidator()
        
        # Add various results
        validator.add_result(True, ValidationSeverity.INFO, "Info", "I001")
        validator.add_result(False, ValidationSeverity.WARNING, "Warning", "W001")
        validator.add_result(False, ValidationSeverity.ERROR, "Error", "E001")
        
        summary = validator.get_summary()
        
        assert summary["total_checks"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2
        assert summary["errors"] == 1
        assert summary["warnings"] == 1
        assert summary["overall_success"] is False


if __name__ == "__main__":
    # Run specific test if provided as argument
    if len(sys.argv) > 1:
        pytest.main([__file__ + "::" + sys.argv[1], "-v"])
    else:
        pytest.main([__file__, "-v"])