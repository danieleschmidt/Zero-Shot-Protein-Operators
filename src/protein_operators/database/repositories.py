"""
Repository pattern implementations for data access.

Provides high-level data access methods with business logic
for protein designs, experiments, and related entities.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, and_, or_

from .models import ProteinDesign, Experiment, Constraint, ValidationResult, PerformanceMetric
from .connection import get_database

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_database().session()
        self._should_close = session is None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close and hasattr(self.session, 'close'):
            self.session.close()


class ProteinDesignRepository(BaseRepository):
    """Repository for protein design operations."""
    
    def create(self, design_data: Dict[str, Any]) -> ProteinDesign:
        """Create a new protein design."""
        design = ProteinDesign(**design_data)
        self.session.add(design)
        self.session.commit()
        logger.info(f"Created protein design: {design.id}")
        return design
    
    def get_by_id(self, design_id: str) -> Optional[ProteinDesign]:
        """Get design by ID."""
        return self.session.query(ProteinDesign).filter(ProteinDesign.id == design_id).first()
    
    def get_by_name(self, name: str) -> Optional[ProteinDesign]:
        """Get design by name."""
        return self.session.query(ProteinDesign).filter(ProteinDesign.name == name).first()
    
    def list_all(self, limit: int = 100, offset: int = 0) -> List[ProteinDesign]:
        """List all designs with pagination."""
        return (self.session.query(ProteinDesign)
                .order_by(desc(ProteinDesign.created_at))
                .limit(limit)
                .offset(offset)
                .all())
    
    def find_by_operator_type(self, operator_type: str) -> List[ProteinDesign]:
        """Find designs by operator type."""
        return (self.session.query(ProteinDesign)
                .filter(ProteinDesign.operator_type == operator_type)
                .order_by(desc(ProteinDesign.created_at))
                .all())
    
    def find_validated(self) -> List[ProteinDesign]:
        """Find all validated designs."""
        return (self.session.query(ProteinDesign)
                .filter(ProteinDesign.is_validated == True)
                .order_by(desc(ProteinDesign.created_at))
                .all())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get design statistics."""
        total = self.session.query(ProteinDesign).count()
        validated = self.session.query(ProteinDesign).filter(ProteinDesign.is_validated == True).count()
        
        operator_stats = (self.session.query(ProteinDesign.operator_type, func.count())
                         .group_by(ProteinDesign.operator_type)
                         .all())
        
        return {
            "total_designs": total,
            "validated_designs": validated,
            "validation_rate": validated / total if total > 0 else 0,
            "operator_distribution": dict(operator_stats)
        }


class ExperimentRepository(BaseRepository):
    """Repository for experiment operations."""
    
    def create(self, experiment_data: Dict[str, Any]) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(**experiment_data)
        self.session.add(experiment)
        self.session.commit()
        logger.info(f"Created experiment: {experiment.id}")
        return experiment
    
    def get_by_id(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.session.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def get_active_experiments(self) -> List[Experiment]:
        """Get all active (running) experiments."""
        return (self.session.query(Experiment)
                .filter(Experiment.status == "running")
                .order_by(desc(Experiment.started_at))
                .all())
    
    def update_statistics(self, experiment_id: str) -> None:
        """Update experiment statistics."""
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return
        
        # Count designs
        num_designs = (self.session.query(ProteinDesign)
                      .filter(ProteinDesign.experiment_id == experiment_id)
                      .count())
        
        # Calculate success rate
        validated_designs = (self.session.query(ProteinDesign)
                           .filter(and_(
                               ProteinDesign.experiment_id == experiment_id,
                               ProteinDesign.is_validated == True
                           ))
                           .count())
        
        experiment.num_designs = num_designs
        experiment.success_rate = validated_designs / num_designs if num_designs > 0 else 0
        
        self.session.commit()


class ConstraintRepository(BaseRepository):
    """Repository for constraint operations."""
    
    def create(self, constraint_data: Dict[str, Any]) -> Constraint:
        """Create a new constraint."""
        constraint = Constraint(**constraint_data)
        self.session.add(constraint)
        self.session.commit()
        return constraint
    
    def get_by_design(self, design_id: str) -> List[Constraint]:
        """Get constraints for a design."""
        return (self.session.query(Constraint)
                .filter(Constraint.design_id == design_id)
                .all())
    
    def get_by_type(self, constraint_type: str) -> List[Constraint]:
        """Get constraints by type."""
        return (self.session.query(Constraint)
                .filter(Constraint.constraint_type == constraint_type)
                .all())