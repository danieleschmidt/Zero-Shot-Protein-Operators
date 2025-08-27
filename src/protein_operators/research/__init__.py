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
try:
    from .paper_experiments import (
        PaperExperimentRunner,
        FigureGenerator,
        TableGenerator
    )
    _PAPER_EXPERIMENTS_AVAILABLE = True
except ImportError:
    _PAPER_EXPERIMENTS_AVAILABLE = False
from .theoretical_analysis import (
    TheoreticalAnalyzer,
    ApproximationBounds,
    ComplexityAnalysis
)

__all__ = [
    'ReproducibilityManager',
    'ExperimentConfig',
    'ResultsArchiver',
    'TheoreticalAnalyzer',
    'ApproximationBounds',
    'ComplexityAnalysis'
]

if _PAPER_EXPERIMENTS_AVAILABLE:
    __all__.extend(['PaperExperimentRunner', 'FigureGenerator', 'TableGenerator'])