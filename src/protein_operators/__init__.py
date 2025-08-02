"""
Zero-Shot Protein Operators

A neural operator framework for PDE-constrained protein design.
"""

__version__ = "0.1.0"
__author__ = "Protein Operators Team"
__email__ = "contact@protein-operators.org"

# Import main classes for public API
from .models import ProteinDeepONet, ProteinFNO
from .constraints import (
    Constraints, 
    BindingSiteConstraint, 
    StructuralConstraint,
    CatalyticConstraint
)
from .core import ProteinDesigner
from .pde import FoldingPDE, ProteinFieldEquations

__all__ = [
    # Core functionality
    "ProteinDesigner",
    
    # Neural operator models
    "ProteinDeepONet",
    "ProteinFNO", 
    
    # Constraint system
    "Constraints",
    "BindingSiteConstraint",
    "StructuralConstraint", 
    "CatalyticConstraint",
    
    # PDE system
    "FoldingPDE",
    "ProteinFieldEquations",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
]