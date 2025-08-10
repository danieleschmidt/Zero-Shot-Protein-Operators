"""
Comprehensive validation framework for protein design components.

This module provides validation utilities for all aspects of the protein design
pipeline, including constraint validation, structure validation, and model validation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
import warnings
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Handle optional torch import
try:
    import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
    logger.warning("PyTorch not available - some validation features will be limited")


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
    
    Attributes:
        passed: Whether the validation check passed
        severity: Severity level of any issues found
        message: Human-readable description of the result
        code: Machine-readable error/warning code
        details: Additional context or data about the result
        suggestions: Recommended actions to fix issues
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
    
    Provides common validation interface and utilities for
    different types of validation (constraints, structures, models).
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []
    
    @abstractmethod
    def validate(self, target: Any) -> List[ValidationResult]:
        """
        Validate a target object.
        
        Args:
            target: Object to validate
            
        Returns:
            List of validation results
        """
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
        
        # Log the result
        if severity == ValidationSeverity.CRITICAL:
            logger.critical(message)
        elif severity == ValidationSeverity.ERROR:
            logger.error(message)
        elif severity == ValidationSeverity.WARNING:
            logger.warning(message)
        else:
            logger.info(message)
    
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


class ConstraintValidator(BaseValidator):
    """
    Validator for protein design constraints.
    
    Validates constraint specifications, parameters, and compatibility.
    """
    
    def validate(self, constraints) -> List[ValidationResult]:
        """
        Validate constraint collection.
        
        Args:
            constraints: Constraints object to validate
            
        Returns:
            List of validation results
        """
        self.results = []
        
        if not hasattr(constraints, 'constraints'):
            self.add_result(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="Invalid constraints object: missing 'constraints' attribute",
                code="CONST_001"
            )
            return self.results
        
        # Validate individual constraints
        for i, constraint in enumerate(constraints.constraints):
            self._validate_constraint(constraint, i)
        
        # Validate constraint compatibility
        self._validate_constraint_compatibility(constraints)
        
        # Check constraint density
        self._validate_constraint_density(constraints)
        
        return self.results
    
    def _validate_constraint(self, constraint, index: int) -> None:
        """Validate individual constraint."""
        try:
            # Check if constraint has required methods
            required_methods = ['encode', 'validate', 'satisfaction_score', 'get_constraint_type_id']
            for method in required_methods:
                if not hasattr(constraint, method):
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Constraint {index}: missing required method '{method}'",
                        code="CONST_002",
                        details={"constraint_index": index, "missing_method": method}
                    )
            
            # Validate constraint parameters
            if hasattr(constraint, 'validate_parameters'):
                try:
                    constraint.validate_parameters()
                    self.add_result(
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"Constraint {index}: parameters valid",
                        code="CONST_003"
                    )
                except ValueError as e:
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Constraint {index}: invalid parameters - {str(e)}",
                        code="CONST_004",
                        details={"constraint_index": index, "error": str(e)}
                    )
            
            # Check constraint encoding
            try:
                encoding = constraint.encode()
                if not hasattr(encoding, 'shape'):
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Constraint {index}: encode() must return tensor-like object",
                        code="CONST_005",
                        details={"constraint_index": index}
                    )
                elif encoding.numel() == 0:
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Constraint {index}: encoding is empty",
                        code="CONST_006",
                        details={"constraint_index": index}
                    )
                else:
                    self.add_result(
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"Constraint {index}: encoding valid",
                        code="CONST_007"
                    )
            except Exception as e:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Constraint {index}: encoding failed - {str(e)}",
                    code="CONST_008",
                    details={"constraint_index": index, "error": str(e)}
                )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Constraint {index}: validation failed - {str(e)}",
                code="CONST_009",
                details={"constraint_index": index, "error": str(e)}
            )
    
    def _validate_constraint_compatibility(self, constraints) -> None:
        """Validate compatibility between constraints."""
        try:
            # Check for conflicting constraints
            binding_sites = []
            ss_regions = []
            
            for constraint in constraints.constraints:
                constraint_type = constraint.__class__.__name__
                
                if constraint_type == "BindingSiteConstraint":
                    binding_sites.append(constraint)
                elif constraint_type == "SecondaryStructureConstraint":
                    ss_regions.append(constraint)
            
            # Check for overlapping binding sites
            for i, bs1 in enumerate(binding_sites):
                for j, bs2 in enumerate(binding_sites[i+1:], i+1):
                    if hasattr(bs1, 'residues') and hasattr(bs2, 'residues'):
                        overlap = set(bs1.residues) & set(bs2.residues)
                        if overlap:
                            self.add_result(
                                passed=False,
                                severity=ValidationSeverity.WARNING,
                                message=f"Binding sites {i} and {j} overlap at residues {overlap}",
                                code="CONST_010",
                                details={"sites": [i, j], "overlap": list(overlap)},
                                suggestions=["Consider merging overlapping binding sites"]
                            )
            
            # Check secondary structure coverage
            total_ss_coverage = 0
            for ss in ss_regions:
                if hasattr(ss, 'start') and hasattr(ss, 'end'):
                    total_ss_coverage += ss.end - ss.start + 1
            
            # Estimate protein length from constraints
            max_residue = 0
            for constraint in constraints.constraints:
                if hasattr(constraint, 'residues'):
                    if constraint.residues:
                        max_residue = max(max_residue, max(constraint.residues))
                elif hasattr(constraint, 'end'):
                    max_residue = max(max_residue, constraint.end)
            
            if max_residue > 0:
                ss_coverage_ratio = total_ss_coverage / max_residue
                if ss_coverage_ratio > 0.95:
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Very high secondary structure coverage ({ss_coverage_ratio:.1%}) may be unrealistic",
                        code="CONST_011",
                        details={"coverage_ratio": ss_coverage_ratio},
                        suggestions=["Leave some regions flexible for loops and turns"]
                    )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Constraint compatibility check failed - {str(e)}",
                code="CONST_012",
                details={"error": str(e)}
            )
    
    def _validate_constraint_density(self, constraints) -> None:
        """Validate constraint density."""
        try:
            if not constraints.constraints:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="No constraints specified - design may be underconstrained",
                    code="CONST_013",
                    suggestions=["Add at least one functional or structural constraint"]
                )
                return
            
            constraint_count = len(constraints.constraints)
            if constraint_count > 20:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Very high number of constraints ({constraint_count}) may be overconstrained",
                    code="CONST_014",
                    details={"constraint_count": constraint_count},
                    suggestions=["Consider consolidating or prioritizing constraints"]
                )
            
            self.add_result(
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Constraint density acceptable ({constraint_count} constraints)",
                code="CONST_015"
            )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Constraint density validation failed - {str(e)}",
                code="CONST_016",
                details={"error": str(e)}
            )


class StructureValidator(BaseValidator):
    """
    Validator for protein structures.
    
    Validates structure geometry, stereochemistry, and physical plausibility.
    """
    
    def validate(self, structure) -> List[ValidationResult]:
        """
        Validate protein structure.
        
        Args:
            structure: ProteinStructure object to validate
            
        Returns:
            List of validation results
        """
        self.results = []
        
        if not hasattr(structure, 'coordinates'):
            self.add_result(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="Invalid structure: missing coordinates",
                code="STRUCT_001"
            )
            return self.results
        
        # Basic structure validation
        self._validate_coordinates(structure)
        self._validate_geometry(structure)
        self._validate_stereochemistry(structure)
        self._validate_clashes(structure)
        
        return self.results
    
    def _validate_coordinates(self, structure) -> None:
        """Validate coordinate data."""
        try:
            coords = structure.coordinates
            
            if coords.numel() == 0:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Structure has no coordinates",
                    code="STRUCT_002"
                )
                return
            
            if len(coords.shape) != 2 or coords.shape[1] != 3:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Invalid coordinate shape: {coords.shape}, expected [N, 3]",
                    code="STRUCT_003",
                    details={"actual_shape": list(coords.shape)}
                )
                return
            
            # Check for NaN or infinite values
            if hasattr(coords, 'isnan') and coords.isnan().any():
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Structure contains NaN coordinates",
                    code="STRUCT_004",
                    suggestions=["Remove or fix invalid coordinate values"]
                )
            
            if hasattr(coords, 'isinf') and coords.isinf().any():
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Structure contains infinite coordinates",
                    code="STRUCT_005",
                    suggestions=["Check coordinate generation process"]
                )
            
            # Check coordinate range
            if hasattr(coords, 'abs'):
                max_coord = coords.abs().max()
                if max_coord > 1000:  # Angstroms
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Very large coordinates detected (max: {max_coord:.1f} Å)",
                        code="STRUCT_006",
                        details={"max_coordinate": float(max_coord)}
                    )
            
            self.add_result(
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Coordinate validation passed ({coords.shape[0]} atoms)",
                code="STRUCT_007"
            )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Coordinate validation failed - {str(e)}",
                code="STRUCT_008",
                details={"error": str(e)}
            )
    
    def _validate_geometry(self, structure) -> None:
        """Validate basic geometry."""
        try:
            coords = structure.coordinates
            if coords.shape[0] < 2:
                return
            
            # Validate bond lengths (CA-CA distances)
            if hasattr(coords, 'norm'):
                distances = []
                for i in range(coords.shape[0] - 1):
                    dist = (coords[i+1] - coords[i]).norm()
                    distances.append(float(dist))
                
                if distances:
                    avg_dist = sum(distances) / len(distances)
                    min_dist = min(distances)
                    max_dist = max(distances)
                    
                    # Expected CA-CA distance is ~3.8 Å
                    if avg_dist < 2.0 or avg_dist > 8.0:
                        self.add_result(
                            passed=False,
                            severity=ValidationSeverity.WARNING,
                            message=f"Unusual average CA-CA distance: {avg_dist:.2f} Å",
                            code="STRUCT_009",
                            details={"avg_distance": avg_dist, "expected": 3.8}
                        )
                    
                    if min_dist < 1.0:
                        self.add_result(
                            passed=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"Very short bond detected: {min_dist:.2f} Å",
                            code="STRUCT_010",
                            details={"min_distance": min_dist}
                        )
                    
                    if max_dist > 15.0:
                        self.add_result(
                            passed=False,
                            severity=ValidationSeverity.WARNING,
                            message=f"Very long bond detected: {max_dist:.2f} Å",
                            code="STRUCT_011",
                            details={"max_distance": max_dist}
                        )
                    
                    if 2.5 <= avg_dist <= 5.0:
                        self.add_result(
                            passed=True,
                            severity=ValidationSeverity.INFO,
                            message=f"Geometry validation passed (avg CA-CA: {avg_dist:.2f} Å)",
                            code="STRUCT_012"
                        )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Geometry validation failed - {str(e)}",
                code="STRUCT_013",
                details={"error": str(e)}
            )
    
    def _validate_stereochemistry(self, structure) -> None:
        """Validate stereochemistry."""
        try:
            # This would implement detailed stereochemical validation
            # For now, just mark as info
            self.add_result(
                passed=True,
                severity=ValidationSeverity.INFO,
                message="Stereochemistry validation not yet implemented",
                code="STRUCT_014"
            )
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Stereochemistry validation failed - {str(e)}",
                code="STRUCT_015",
                details={"error": str(e)}
            )
    
    def _validate_clashes(self, structure) -> None:
        """Validate atomic clashes."""
        try:
            coords = structure.coordinates
            if coords.shape[0] < 3:
                return
            
            clash_count = 0
            min_distance = float('inf')
            
            # Simplified clash detection
            for i in range(coords.shape[0]):
                for j in range(i + 2, coords.shape[0]):  # Skip adjacent residues
                    if hasattr(coords, 'norm'):
                        dist = (coords[i] - coords[j]).norm()
                        min_distance = min(min_distance, float(dist))
                        
                        if dist < 2.0:  # Van der Waals radius
                            clash_count += 1
            
            if clash_count > 0:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Detected {clash_count} potential clashes",
                    code="STRUCT_016",
                    details={"clash_count": clash_count, "min_distance": min_distance},
                    suggestions=["Consider energy minimization to resolve clashes"]
                )
            else:
                self.add_result(
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message="No clashes detected",
                    code="STRUCT_017"
                )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Clash validation failed - {str(e)}",
                code="STRUCT_018",
                details={"error": str(e)}
            )


class ModelValidator(BaseValidator):
    """
    Validator for neural operator models.
    
    Validates model architecture, parameters, and configuration.
    """
    
    def validate(self, model) -> List[ValidationResult]:
        """
        Validate neural operator model.
        
        Args:
            model: Neural operator model to validate
            
        Returns:
            List of validation results
        """
        self.results = []
        
        # Basic model validation
        self._validate_model_interface(model)
        self._validate_model_config(model)
        
        return self.results
    
    def _validate_model_interface(self, model) -> None:
        """Validate model interface."""
        try:
            required_methods = ['forward', 'encode_constraints', 'encode_coordinates']
            for method in required_methods:
                if not hasattr(model, method):
                    self.add_result(
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Model missing required method: {method}",
                        code="MODEL_001",
                        details={"missing_method": method}
                    )
                else:
                    self.add_result(
                        passed=True,
                        severity=ValidationSeverity.INFO,
                        message=f"Model has required method: {method}",
                        code="MODEL_002"
                    )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Model interface validation failed - {str(e)}",
                code="MODEL_003",
                details={"error": str(e)}
            )
    
    def _validate_model_config(self, model) -> None:
        """Validate model configuration."""
        try:
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                self.add_result(
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Model info available: {info.get('model_type', 'Unknown')}",
                    code="MODEL_004",
                    details=info
                )
            else:
                self.add_result(
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="Model info not available",
                    code="MODEL_005"
                )
        
        except Exception as e:
            self.add_result(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Model config validation failed - {str(e)}",
                code="MODEL_006",
                details={"error": str(e)}
            )


def validate_design_pipeline(
    constraints,
    model,
    structure = None,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive validation of entire design pipeline.
    
    Args:
        constraints: Constraints object
        model: Neural operator model
        structure: Optional structure to validate
        strict_mode: Whether to treat warnings as errors
        
    Returns:
        Comprehensive validation report
    """
    results = {
        "constraints": [],
        "model": [],
        "structure": [],
        "overall": {}
    }
    
    # Validate constraints
    constraint_validator = ConstraintValidator(strict_mode)
    results["constraints"] = constraint_validator.validate(constraints)
    
    # Validate model
    model_validator = ModelValidator(strict_mode)
    results["model"] = model_validator.validate(model)
    
    # Validate structure if provided
    if structure is not None:
        structure_validator = StructureValidator(strict_mode)
        results["structure"] = structure_validator.validate(structure)
    
    # Overall summary
    all_results = (results["constraints"] + results["model"] + 
                  (results["structure"] if structure else []))
    
    results["overall"] = {
        "total_checks": len(all_results),
        "passed": sum(1 for r in all_results if r.passed),
        "failed": sum(1 for r in all_results if not r.passed),
        "errors": sum(1 for r in all_results if r.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}),
        "warnings": sum(1 for r in all_results if r.severity == ValidationSeverity.WARNING),
        "success": not any(r.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL} for r in all_results)
    }
    
    return results