"""
Research framework for neural operator protein design.

This module provides tools and utilities for conducting reproducible
research with neural operators for protein structure prediction and design.
"""

from .reproducibility import (
    ReproducibilityManager,
    ExperimentConfig,
    ResultsArchiver
)
from .paper_experiments import (
    PaperExperimentRunner,
    FigureGenerator,
    TableGenerator
)
from .theoretical_analysis import (
    TheoreticalAnalyzer,
    ApproximationBounds,
    ComplexityAnalysis
)

__all__ = [
    'ReproducibilityManager',
    'ExperimentConfig',
    'ResultsArchiver',
    'PaperExperimentRunner',
    'FigureGenerator',
    'TableGenerator',
    'TheoreticalAnalyzer',
    'ApproximationBounds',
    'ComplexityAnalysis'
]