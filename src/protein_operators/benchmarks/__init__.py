"""
Research-grade benchmarking suite for neural operator protein design.

This module provides comprehensive benchmarking capabilities for comparing
different neural operator architectures on protein folding and design tasks.
"""

from .benchmark_suite import (
    ProteinBenchmarkSuite,
    BenchmarkResult,
    StatisticalTestResult
)
from .metrics import (
    ProteinStructureMetrics,
    PhysicsMetrics,
    BiochemicalMetrics
)
from .datasets import (
    ProteinBenchmarkDatasets,
    SyntheticProteinDataset,
    CATHDataset
)
from .statistical_tests import (
    StatisticalAnalyzer,
    ComparisonResult,
    SignificanceTest
)

__all__ = [
    'ProteinBenchmarkSuite',
    'BenchmarkResult',
    'StatisticalTestResult',
    'ProteinStructureMetrics',
    'PhysicsMetrics',
    'BiochemicalMetrics',
    'ProteinBenchmarkDatasets',
    'SyntheticProteinDataset',
    'CATHDataset',
    'StatisticalAnalyzer',
    'ComparisonResult',
    'SignificanceTest'
]