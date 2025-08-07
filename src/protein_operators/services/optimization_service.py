"""
Optimization service for protein structure refinement and improvement.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization algorithms."""
    GRADIENT_DESCENT = "gradient_descent"
    LBFGS = "lbfgs"
    ADAM = "adam"
    CONJUGATE_GRADIENT = "conjugate_gradient"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    method: OptimizationType = OptimizationType.LBFGS
    max_iterations: int = 1000
    learning_rate: float = 0.01
    convergence_threshold: float = 1e-6
    energy_weights: Dict[str, float] = None
    constraints_weight: float = 1.0
    regularization_weight: float = 0.1
    step_size_decay: float = 0.95
    early_stopping_patience: int = 50
    
    def __post_init__(self):
        if self.energy_weights is None:
            self.energy_weights = {
                'bond': 1.0,
                'angle': 0.5,
                'torsion': 0.3,
                'vdw': 0.8,
                'electrostatic': 0.6
            }


@dataclass
class OptimizationResult:
    """Results from optimization run."""
    optimized_coordinates: torch.Tensor
    initial_energy: float
    final_energy: float
    energy_reduction: float
    iterations_completed: int
    convergence_achieved: bool
    optimization_time: float
    energy_trajectory: List[float]
    gradient_norms: List[float]
    constraint_violations: List[float]
    metadata: Dict[str, Any]


class OptimizationService:
    """
    Service for optimizing protein structures using various algorithms.
    
    This service provides robust optimization capabilities with proper
    error handling, logging, and monitoring.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize optimization service.
        
        Args:
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info(f"OptimizationService initialized on device: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computing device with fallback."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
                logger.info("Using GPU acceleration for optimization")
            else:
                device_obj = torch.device("cpu")
                logger.info("Using CPU for optimization")
        else:
            device_obj = torch.device(device)
            logger.info(f"Using specified device: {device}")
        
        return device_obj
    
    def optimize_structure(
        self,
        coordinates: torch.Tensor,
        constraints: Optional[Any] = None,
        config: Optional[OptimizationConfig] = None
    ) -> OptimizationResult:
        """
        Optimize protein structure coordinates.
        
        Args:
            coordinates: Initial protein coordinates [N, 3]
            constraints: Protein constraints object
            config: Optimization configuration
            
        Returns:
            OptimizationResult with optimized coordinates and metadata
            
        Raises:
            ValueError: If input coordinates are invalid
            RuntimeError: If optimization fails
        """
        if config is None:
            config = OptimizationConfig()
        
        logger.info(f"Starting structure optimization with {config.method.value}")
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_coordinates(coordinates)
            
            # Move to appropriate device
            coords = coordinates.clone().to(self.device)
            coords.requires_grad_(True)
            
            # Initialize optimization algorithm
            optimizer = self._create_optimizer(coords, config)
            
            # Track optimization progress
            energy_trajectory = []
            gradient_norms = []
            constraint_violations = []
            
            # Initial energy
            initial_energy = self._compute_total_energy(coords, constraints, config)
            energy_trajectory.append(float(initial_energy))
            
            logger.debug(f"Initial energy: {initial_energy:.6f}")
            
            # Optimization loop
            convergence_achieved = False
            best_energy = float('inf')
            patience_counter = 0
            
            for iteration in range(config.max_iterations):
                # Perform optimization step
                def closure():
                    optimizer.zero_grad()
                    energy = self._compute_total_energy(coords, constraints, config)
                    energy.backward()
                    return energy
                
                if config.method == OptimizationType.LBFGS:
                    energy = optimizer.step(closure)
                else:
                    energy = closure()
                    optimizer.step()
                
                current_energy = float(energy)
                energy_trajectory.append(current_energy)
                
                # Compute gradient norm
                grad_norm = torch.norm(coords.grad).item() if coords.grad is not None else 0.0
                gradient_norms.append(grad_norm)
                
                # Compute constraint violations
                if constraints is not None:
                    violations = self._compute_constraint_violations(coords, constraints)
                    constraint_violations.append(violations)
                else:
                    constraint_violations.append(0.0)
                
                # Check convergence
                if iteration > 0:
                    energy_change = abs(current_energy - energy_trajectory[-2])
                    if energy_change < config.convergence_threshold:
                        convergence_achieved = True
                        logger.info(f"Converged at iteration {iteration} (energy change: {energy_change:.8f})")
                        break
                
                # Early stopping
                if current_energy < best_energy:
                    best_energy = current_energy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping at iteration {iteration} (patience exceeded)")
                        break
                
                # Periodic logging
                if iteration % 50 == 0:
                    logger.debug(f"Iteration {iteration}: energy={current_energy:.6f}, grad_norm={grad_norm:.6f}")
                
                # Prevent runaway optimization
                if torch.any(torch.abs(coords) > 100.0):
                    logger.warning("Coordinates becoming unstable, stopping optimization")
                    break
            
            # Final results
            final_energy = energy_trajectory[-1] if energy_trajectory else initial_energy
            energy_reduction = initial_energy - final_energy
            optimization_time = time.time() - start_time
            
            result = OptimizationResult(
                optimized_coordinates=coords.detach().cpu(),
                initial_energy=float(initial_energy),
                final_energy=final_energy,
                energy_reduction=energy_reduction,
                iterations_completed=len(energy_trajectory) - 1,
                convergence_achieved=convergence_achieved,
                optimization_time=optimization_time,
                energy_trajectory=energy_trajectory,
                gradient_norms=gradient_norms,
                constraint_violations=constraint_violations,
                metadata={
                    'method': config.method.value,
                    'device': str(self.device),
                    'learning_rate': config.learning_rate,
                    'constraints_weight': config.constraints_weight
                }
            )
            
            # Store in history
            self.optimization_history.append(result)
            
            logger.info(f"Optimization completed: {energy_reduction:.6f} energy reduction "
                       f"in {optimization_time:.2f}s ({result.iterations_completed} iterations)")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Structure optimization failed: {str(e)}") from e
    
    def _validate_coordinates(self, coordinates: torch.Tensor) -> None:
        """Validate input coordinates."""
        if not isinstance(coordinates, torch.Tensor):
            raise ValueError("Coordinates must be a torch.Tensor")
        
        if coordinates.dim() != 2 or coordinates.shape[1] != 3:
            raise ValueError("Coordinates must have shape [N, 3]")
        
        if coordinates.shape[0] == 0:
            raise ValueError("Coordinates cannot be empty")
        
        if torch.any(torch.isnan(coordinates)) or torch.any(torch.isinf(coordinates)):
            raise ValueError("Coordinates contain NaN or infinite values")
        
        if torch.any(torch.abs(coordinates) > 1000.0):
            raise ValueError("Coordinates contain unreasonably large values (>1000 Å)")
    
    def _create_optimizer(self, coordinates: torch.Tensor, config: OptimizationConfig) -> torch.optim.Optimizer:
        """Create appropriate optimizer based on configuration."""
        if config.method == OptimizationType.LBFGS:
            optimizer = torch.optim.LBFGS(
                [coordinates],
                lr=config.learning_rate,
                max_iter=20,
                tolerance_grad=config.convergence_threshold,
                tolerance_change=config.convergence_threshold * 1e-2
            )
        elif config.method == OptimizationType.ADAM:
            optimizer = torch.optim.Adam(
                [coordinates],
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.method == OptimizationType.GRADIENT_DESCENT:
            optimizer = torch.optim.SGD(
                [coordinates],
                lr=config.learning_rate,
                momentum=0.9
            )
        else:
            logger.warning(f"Unknown optimization method {config.method}, defaulting to LBFGS")
            optimizer = torch.optim.LBFGS([coordinates], lr=config.learning_rate)
        
        return optimizer
    
    def _compute_total_energy(
        self,
        coordinates: torch.Tensor,
        constraints: Optional[Any],
        config: OptimizationConfig
    ) -> torch.Tensor:
        """Compute total energy including physics and constraints."""
        total_energy = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            # Physics-based energy terms
            physics_energy = self._compute_physics_energy(coordinates, config.energy_weights)
            total_energy = total_energy + physics_energy
            
            # Constraint satisfaction energy
            if constraints is not None:
                constraint_energy = self._compute_constraint_energy(coordinates, constraints)
                total_energy = total_energy + config.constraints_weight * constraint_energy
            
            # Regularization term to prevent overfitting
            regularization = config.regularization_weight * torch.sum(coordinates**2)
            total_energy = total_energy + regularization
            
            return total_energy
            
        except Exception as e:
            logger.error(f"Error computing energy: {str(e)}")
            # Return high energy to signal problem
            return torch.tensor(1e6, device=self.device, requires_grad=True)
    
    def _compute_physics_energy(self, coordinates: torch.Tensor, weights: Dict[str, float]) -> torch.Tensor:
        """Compute physics-based energy terms."""
        energy = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if coordinates.shape[0] < 2:
            return energy
        
        # Bond energy (consecutive CA atoms)
        bond_vectors = coordinates[1:] - coordinates[:-1]
        bond_lengths = torch.norm(bond_vectors, dim=1)
        ideal_bond_length = 3.8  # CA-CA distance in Angstroms
        bond_energy = weights['bond'] * torch.sum((bond_lengths - ideal_bond_length)**2)
        energy = energy + bond_energy
        
        # Angle energy (triplets of CA atoms)
        if coordinates.shape[0] >= 3:
            v1 = coordinates[1:-1] - coordinates[:-2]  # [N-2, 3]
            v2 = coordinates[2:] - coordinates[1:-1]   # [N-2, 3]
            
            # Normalize vectors
            v1_norm = torch.nn.functional.normalize(v1, dim=1)
            v2_norm = torch.nn.functional.normalize(v2, dim=1)
            
            # Compute angles
            cos_angles = torch.sum(v1_norm * v2_norm, dim=1)
            # Clamp to avoid numerical issues
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            
            # Prefer angles around 120 degrees (cos(120°) = -0.5)
            ideal_cos_angle = -0.5
            angle_energy = weights['angle'] * torch.sum((cos_angles - ideal_cos_angle)**2)
            energy = energy + angle_energy
        
        # Van der Waals energy (simplified pairwise potential)
        if coordinates.shape[0] >= 4:
            # Compute pairwise distances
            distances = torch.cdist(coordinates, coordinates)
            
            # Mask out bonded neighbors (i, i+1, i+2)
            mask = torch.ones_like(distances, dtype=torch.bool)
            n = coordinates.shape[0]
            for offset in range(3):
                if offset < n:
                    idx = torch.arange(n - offset)
                    mask[idx, idx + offset] = False
                    if offset > 0:
                        mask[idx + offset, idx] = False
            
            # Apply mask
            relevant_distances = distances[mask]
            
            # Lennard-Jones-like potential
            sigma = 4.0  # Van der Waals radius
            epsilon = weights['vdw']
            
            # Avoid division by zero
            safe_distances = torch.clamp(relevant_distances, min=0.5)
            
            # Simplified LJ potential: repulsive part only for efficiency
            vdw_energy = epsilon * torch.sum(torch.pow(sigma / safe_distances, 12))
            energy = energy + vdw_energy
        
        return energy
    
    def _compute_constraint_energy(self, coordinates: torch.Tensor, constraints: Any) -> torch.Tensor:
        """Compute constraint satisfaction energy."""
        constraint_energy = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            # Binding site constraints
            if hasattr(constraints, 'binding_sites'):
                for binding_site in constraints.binding_sites:
                    # Ensure binding site residues are clustered
                    if all(res - 1 < coordinates.shape[0] for res in binding_site.residues):
                        indices = [res - 1 for res in binding_site.residues]  # Convert to 0-based
                        site_coords = coordinates[indices]
                        
                        if len(site_coords) > 1:
                            center = torch.mean(site_coords, dim=0)
                            distances = torch.norm(site_coords - center, dim=1)
                            
                            # Penalty for dispersed binding sites
                            dispersion_penalty = torch.var(distances)
                            constraint_energy = constraint_energy + dispersion_penalty
            
            # Secondary structure constraints
            if hasattr(constraints, 'secondary_structure'):
                for ss_constraint in constraints.secondary_structure:
                    start_idx = max(0, ss_constraint.start - 1)
                    end_idx = min(coordinates.shape[0], ss_constraint.end)
                    
                    if end_idx > start_idx:
                        ss_coords = coordinates[start_idx:end_idx]
                        
                        # Different energy functions for different SS types
                        if hasattr(ss_constraint, 'ss_type'):
                            if ss_constraint.ss_type == 'helix':
                                # Helices should be relatively straight and compact
                                if len(ss_coords) >= 3:
                                    # Penalize high curvature
                                    vectors = ss_coords[1:] - ss_coords[:-1]
                                    if len(vectors) >= 2:
                                        curvature = torch.sum((vectors[1:] - vectors[:-1])**2)
                                        constraint_energy = constraint_energy + curvature * 0.1
                            
                            elif ss_constraint.ss_type == 'sheet':
                                # Sheets should be extended
                                if len(ss_coords) >= 2:
                                    distances = torch.norm(ss_coords[1:] - ss_coords[:-1], dim=1)
                                    # Prefer extended conformations
                                    extension_penalty = torch.sum((distances - 3.8)**2) * 0.05
                                    constraint_energy = constraint_energy + extension_penalty
        
        except Exception as e:
            logger.warning(f"Error computing constraint energy: {str(e)}")
            # Return small penalty to avoid breaking optimization
            constraint_energy = torch.tensor(10.0, device=self.device, requires_grad=True)
        
        return constraint_energy
    
    def _compute_constraint_violations(self, coordinates: torch.Tensor, constraints: Any) -> float:
        """Compute constraint violation score."""
        try:
            total_violation = 0.0
            
            # Check binding site clustering
            if hasattr(constraints, 'binding_sites'):
                for binding_site in constraints.binding_sites:
                    if all(res - 1 < coordinates.shape[0] for res in binding_site.residues):
                        indices = [res - 1 for res in binding_site.residues]
                        site_coords = coordinates[indices].detach()
                        
                        if len(site_coords) > 1:
                            center = torch.mean(site_coords, dim=0)
                            distances = torch.norm(site_coords - center, dim=1)
                            # Violation if spread > 8 Angstroms
                            max_distance = torch.max(distances).item()
                            if max_distance > 8.0:
                                total_violation += (max_distance - 8.0) / 8.0
            
            return total_violation
            
        except Exception as e:
            logger.warning(f"Error computing constraint violations: {str(e)}")
            return 0.0
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about completed optimizations."""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        history = self.optimization_history
        
        return {
            "total_optimizations": len(history),
            "average_energy_reduction": np.mean([r.energy_reduction for r in history]),
            "average_iterations": np.mean([r.iterations_completed for r in history]),
            "average_time": np.mean([r.optimization_time for r in history]),
            "convergence_rate": np.mean([r.convergence_achieved for r in history]),
            "best_energy_reduction": max([r.energy_reduction for r in history]),
            "total_time": sum([r.optimization_time for r in history])
        }
    
    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        logger.info("Optimization history cleared")


class MultiObjectiveOptimizer:
    """Multi-objective optimization for protein design."""
    
    def __init__(self, service: OptimizationService):
        self.service = service
    
    def optimize_pareto(
        self,
        coordinates: torch.Tensor,
        objectives: List[Callable],
        weights: Optional[List[float]] = None,
        config: Optional[OptimizationConfig] = None
    ) -> List[OptimizationResult]:
        """
        Perform multi-objective Pareto optimization.
        
        Args:
            coordinates: Initial coordinates
            objectives: List of objective functions to optimize
            weights: Weights for each objective (if None, equal weights)
            config: Optimization configuration
            
        Returns:
            List of Pareto-optimal solutions
        """
        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)
        
        # For now, implement weighted sum approach
        # TODO: Implement true Pareto optimization (NSGA-II, etc.)
        
        if config is None:
            config = OptimizationConfig()
        
        # Create composite objective
        def composite_objective(coords):
            total = torch.tensor(0.0, device=coords.device, requires_grad=True)
            for obj, weight in zip(objectives, weights):
                total = total + weight * obj(coords)
            return total
        
        # Temporarily replace energy computation
        original_compute = self.service._compute_total_energy
        
        def wrapped_compute(coords, constraints, cfg):
            return composite_objective(coords)
        
        self.service._compute_total_energy = wrapped_compute
        
        try:
            result = self.service.optimize_structure(coordinates, None, config)
            return [result]  # Single result for now
        finally:
            # Restore original function
            self.service._compute_total_energy = original_compute