"""
Database and data persistence layer for protein operators.

This module provides database connectivity, schema management, and data
access patterns for storing protein designs, experiments, and metadata.
"""

from .connection import DatabaseManager, get_database
from .models import (
    ProteinDesign,
    Experiment,
    Constraint,
    ValidationResult,
    PerformanceMetric
)
from .repositories import (
    ProteinDesignRepository,
    ExperimentRepository,
    ConstraintRepository
)
from .migrations import MigrationManager, run_migrations

__all__ = [
    # Core database
    "DatabaseManager",
    "get_database",
    
    # Models
    "ProteinDesign",
    "Experiment", 
    "Constraint",
    "ValidationResult",
    "PerformanceMetric",
    
    # Repositories
    "ProteinDesignRepository",
    "ExperimentRepository",
    "ConstraintRepository",
    
    # Migrations
    "MigrationManager",
    "run_migrations",
]