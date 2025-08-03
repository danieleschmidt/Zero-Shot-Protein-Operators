"""
RESTful API for protein design operations.

Provides HTTP endpoints for protein design, validation, and analysis
with FastAPI framework for high-performance async operations.
"""

from .app import create_app, get_app
from .models import (
    DesignRequest,
    DesignResponse,
    ValidationRequest,
    ValidationResponse,
    ExperimentRequest,
    ExperimentResponse
)
from .routes import router
from .dependencies import get_designer, get_database, get_current_user
from .middleware import setup_middleware

__all__ = [
    "create_app",
    "get_app",
    "DesignRequest",
    "DesignResponse", 
    "ValidationRequest",
    "ValidationResponse",
    "ExperimentRequest",
    "ExperimentResponse",
    "router",
    "get_designer",
    "get_database",
    "get_current_user",
    "setup_middleware",
]