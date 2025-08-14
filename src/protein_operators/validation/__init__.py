"""
Experimental validation framework for neural operator protein design.

This module provides comprehensive experimental validation capabilities
for neural operator-based protein structure prediction and design,
including uncertainty quantification and experimental comparison protocols.
"""

from .validation_framework import (
    ExperimentalValidationFramework,
    ValidationResult,
    UncertaintyQuantifier
)
from .experimental_protocols import (
    StructuralValidationProtocol,
    FunctionalValidationProtocol,
    ThermodynamicValidationProtocol
)
from .uncertainty_estimation import (
    EnsembleUncertainty,
    BayesianUncertainty,
    DropoutUncertainty,
    CalibrationAnalyzer
)
from .cross_validation import (
    ProteinCrossValidator,
    StratifiedProteinSplit,
    TemporalSplit
)

__all__ = [
    'ExperimentalValidationFramework',
    'ValidationResult',
    'UncertaintyQuantifier',
    'StructuralValidationProtocol',
    'FunctionalValidationProtocol',
    'ThermodynamicValidationProtocol',
    'EnsembleUncertainty',
    'BayesianUncertainty',
    'DropoutUncertainty',
    'CalibrationAnalyzer',
    'ProteinCrossValidator',
    'StratifiedProteinSplit',
    'TemporalSplit'
]