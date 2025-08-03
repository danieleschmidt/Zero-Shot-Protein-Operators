"""
Pydantic models for API request/response schemas.

Defines data models for protein design API endpoints with validation
and serialization support.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
import torch


class OperatorType(str, Enum):
    """Supported neural operator types."""
    DEEPONET = "deeponet"
    FNO = "fno"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"


class ConstraintType(str, Enum):
    """Supported constraint types."""
    BINDING_SITE = "binding_site"
    SECONDARY_STRUCTURE = "secondary_structure"
    STABILITY = "stability"
    CATALYTIC = "catalytic"
    ALLOSTERIC = "allosteric"
    DISULFIDE_BOND = "disulfide_bond"
    METAL_SITE = "metal_site"
    FOLD = "fold"


class ValidationLevel(str, Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


# Request Models
class ConstraintSpec(BaseModel):
    """Constraint specification."""
    name: str = Field(..., description="Constraint name")
    type: ConstraintType = Field(..., description="Type of constraint")
    parameters: Dict[str, Any] = Field(..., description="Constraint parameters")
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Constraint weight")
    tolerance: float = Field(0.1, ge=0.0, le=1.0, description="Constraint tolerance")
    required: bool = Field(True, description="Whether constraint is required")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "ATP_binding_site",
                "type": "binding_site",
                "parameters": {
                    "residues": [10, 15, 20, 25],
                    "ligand": "ATP",
                    "affinity_nm": 100.0
                },
                "weight": 1.0,
                "tolerance": 0.1,
                "required": True
            }
        }


class DesignRequest(BaseModel):
    """Protein design request."""
    name: str = Field(..., description="Design name")
    description: Optional[str] = Field(None, description="Design description")
    length: int = Field(..., ge=10, le=2000, description="Target protein length")
    operator_type: OperatorType = Field(OperatorType.DEEPONET, description="Neural operator type")
    constraints: List[ConstraintSpec] = Field([], description="Design constraints")
    num_samples: int = Field(1, ge=1, le=10, description="Number of samples to generate")
    physics_guided: bool = Field(False, description="Use physics-guided refinement")
    model_checkpoint: Optional[str] = Field(None, description="Model checkpoint path")
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")
    
    # Generation parameters
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    max_iterations: int = Field(100, ge=1, le=1000, description="Maximum iterations")
    convergence_threshold: float = Field(1e-6, ge=1e-10, le=1e-3, description="Convergence threshold")
    
    @validator('constraints')
    def validate_constraints(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 constraints allowed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "ATP_binding_protein",
                "description": "Design protein that binds ATP",
                "length": 150,
                "operator_type": "deeponet",
                "constraints": [
                    {
                        "name": "ATP_binding_site",
                        "type": "binding_site",
                        "parameters": {
                            "residues": [45, 67, 89],
                            "ligand": "ATP",
                            "affinity_nm": 100.0
                        },
                        "weight": 1.0,
                        "required": True
                    }
                ],
                "num_samples": 5,
                "physics_guided": True
            }
        }


class ValidationRequest(BaseModel):
    """Structure validation request."""
    design_id: Optional[str] = Field(None, description="Design ID for validation")
    coordinates: Optional[List[List[float]]] = Field(None, description="Coordinates to validate")
    sequence: Optional[str] = Field(None, description="Amino acid sequence")
    validation_level: ValidationLevel = Field(ValidationLevel.STANDARD, description="Validation level")
    checks: List[str] = Field([], description="Specific validation checks")
    
    @root_validator
    def validate_input(cls, values):
        design_id = values.get('design_id')
        coordinates = values.get('coordinates')
        
        if not design_id and not coordinates:
            raise ValueError("Either design_id or coordinates must be provided")
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "design_id": "550e8400-e29b-41d4-a716-446655440000",
                "validation_level": "comprehensive",
                "checks": ["stereochemistry", "clashes", "ramachandran"]
            }
        }


class ExperimentRequest(BaseModel):
    """Experiment creation request."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    objective: str = Field(..., description="Experiment objective")
    parameters: Dict[str, Any] = Field({}, description="Experiment parameters")
    tags: List[str] = Field([], description="Experiment tags")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "ATP_binding_study",
                "description": "Study of ATP binding protein designs",
                "objective": "Design proteins with high ATP binding affinity",
                "parameters": {
                    "target_affinity": 100.0,
                    "operator_type": "deeponet",
                    "num_designs": 50
                },
                "tags": ["ATP", "binding", "enzymes"]
            }
        }


# Response Models
class CoordinateData(BaseModel):
    """3D coordinate data."""
    x: float
    y: float
    z: float


class DesignResponse(BaseModel):
    """Protein design response."""
    id: str = Field(..., description="Design ID")
    name: str = Field(..., description="Design name")
    sequence: Optional[str] = Field(None, description="Generated sequence")
    length: int = Field(..., description="Protein length")
    coordinates: List[List[float]] = Field(..., description="3D coordinates")
    operator_type: str = Field(..., description="Neural operator used")
    generation_time: float = Field(..., description="Generation time in seconds")
    constraint_satisfaction: Dict[str, float] = Field({}, description="Constraint satisfaction scores")
    validation_scores: Dict[str, float] = Field({}, description="Validation scores")
    status: str = Field(..., description="Design status")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "ATP_binding_protein",
                "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRIIQKLNPP",
                "length": 150,
                "coordinates": [[-5.123, 2.456, 1.789], [-1.323, 2.456, 1.789]],
                "operator_type": "deeponet",
                "generation_time": 12.5,
                "constraint_satisfaction": {"ATP_binding_site": 0.92},
                "validation_scores": {"overall_score": 0.85},
                "status": "generated",
                "created_at": "2025-01-01T12:00:00Z"
            }
        }


class ValidationResponse(BaseModel):
    """Structure validation response."""
    id: str = Field(..., description="Validation ID")
    design_id: Optional[str] = Field(None, description="Associated design ID")
    is_valid: bool = Field(..., description="Overall validity")
    overall_score: float = Field(..., description="Overall validation score")
    checks: Dict[str, Dict[str, Any]] = Field({}, description="Individual check results")
    warnings: List[str] = Field([], description="Validation warnings")
    errors: List[str] = Field([], description="Validation errors")
    validation_time: float = Field(..., description="Validation time in seconds")
    created_at: datetime = Field(..., description="Validation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "validation_123",
                "design_id": "550e8400-e29b-41d4-a716-446655440000",
                "is_valid": True,
                "overall_score": 0.85,
                "checks": {
                    "stereochemistry": {"score": 0.92, "passed": True},
                    "clashes": {"score": 0.88, "passed": True},
                    "ramachandran": {"score": 0.78, "passed": True}
                },
                "warnings": ["Minor clash detected at residue 45"],
                "errors": [],
                "validation_time": 2.1,
                "created_at": "2025-01-01T12:05:00Z"
            }
        }


class ExperimentResponse(BaseModel):
    """Experiment response."""
    id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    status: str = Field(..., description="Experiment status")
    num_designs: int = Field(..., description="Number of designs")
    success_rate: Optional[float] = Field(None, description="Success rate")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "exp_123",
                "name": "ATP_binding_study",
                "description": "Study of ATP binding protein designs",
                "status": "running",
                "num_designs": 25,
                "success_rate": 0.8,
                "started_at": "2025-01-01T10:00:00Z",
                "completed_at": None
            }
        }


class DesignSummary(BaseModel):
    """Design summary for listings."""
    id: str
    name: str
    length: int
    operator_type: str
    status: str
    validation_score: Optional[float] = None
    created_at: datetime


class ExperimentSummary(BaseModel):
    """Experiment summary for listings."""
    id: str
    name: str
    status: str
    num_designs: int
    success_rate: Optional[float] = None
    started_at: datetime


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid protein length",
                "details": {"length": "must be between 10 and 2000"},
                "timestamp": "2025-01-01T12:00:00Z"
            }
        }


class StatusResponse(BaseModel):
    """Status response model."""
    status: str = Field(..., description="Status")
    message: Optional[str] = Field(None, description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


# Utility models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    api: str
    uptime: Optional[float] = None
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Metrics response."""
    designs_total: int
    designs_validated: int
    experiments_active: int
    success_rate: float
    avg_generation_time: float
    uptime: float