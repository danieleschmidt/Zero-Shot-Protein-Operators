"""
Core protein design service with advanced algorithms.
"""

from typing import Dict, List, Optional, Tuple, Any
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging

from ..core import ProteinDesigner
from ..constraints import Constraints
from ..structure import ProteinStructure
from ..models import ProteinDeepONet, ProteinFNO


class ProteinDesignService:
    """
    Advanced protein design service with multi-objective optimization,
    iterative refinement, and constraint satisfaction algorithms.
    """
    
    def __init__(
        self,
        model_repository_path: str = "models/",
        cache_enabled: bool = True,
        max_cache_size: int = 1000
    ):
        """Initialize protein design service."""
        self.model_repo_path = Path(model_repository_path)
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        
        # Model registry
        self.loaded_models = {}
        self.model_configs = {}
        
        # Design cache for performance
        self.design_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.design_statistics = {
            'total_designs': 0,
            'successful_designs': 0,
            'average_design_time': 0.0,
            'constraint_satisfaction_rate': 0.0
        }
    
    def load_model(
        self, 
        model_name: str,
        operator_type: str = "deeponet",
        force_reload: bool = False
    ) -> ProteinDesigner:
        """Load and cache a protein design model."""
        cache_key = f"{model_name}_{operator_type}"
        
        if cache_key in self.loaded_models and not force_reload:
            return self.loaded_models[cache_key]
        
        model_path = self.model_repo_path / f"{model_name}.pt"
        
        if not model_path.exists():
            self.logger.warning(f"Model {model_name} not found, using default initialization")
            designer = ProteinDesigner(operator_type=operator_type)
        else:
            designer = ProteinDesigner(
                operator_type=operator_type,
                checkpoint=str(model_path)
            )
        
        self.loaded_models[cache_key] = designer
        self.logger.info(f"Loaded model: {model_name} ({operator_type})")
        
        return designer
    
    def design_protein_multi_objective(
        self,
        constraints: Constraints,
        objectives: Dict[str, Tuple[str, float]],
        length: int,
        model_name: str = "default",
        num_candidates: int = 10,
        optimization_rounds: int = 3
    ) -> List[ProteinStructure]:
        """
        Design proteins with multiple objectives using Pareto optimization.
        
        Args:
            constraints: Design constraints
            objectives: Dictionary of objectives with (direction, target_value)
                       e.g., {'stability': ('maximize', 80), 'binding_affinity': ('minimize', 1e-9)}
            length: Target protein length
            model_name: Model to use for design
            num_candidates: Number of candidate designs to generate
            optimization_rounds: Number of optimization iterations
            
        Returns:
            List of Pareto-optimal protein structures
        """
        self.logger.info(f"Starting multi-objective design: {len(objectives)} objectives")
        
        # Load appropriate model
        designer = self.load_model(model_name)
        
        # Generate initial candidate population
        candidates = []
        for i in range(num_candidates):
            try:
                structure = designer.generate(
                    constraints=constraints,
                    length=length,
                    num_samples=1
                )
                candidates.append(structure)
            except Exception as e:
                self.logger.warning(f"Failed to generate candidate {i}: {e}")
                continue
        
        if not candidates:
            raise RuntimeError("Failed to generate any candidate structures")
        
        # Iterative optimization
        for round_idx in range(optimization_rounds):
            self.logger.info(f"Optimization round {round_idx + 1}/{optimization_rounds}")
            
            # Evaluate objectives for all candidates
            objective_scores = self._evaluate_objectives(candidates, objectives)
            
            # Select Pareto-optimal candidates
            pareto_indices = self._find_pareto_optimal(objective_scores, objectives)
            pareto_candidates = [candidates[i] for i in pareto_indices]
            
            # Generate new candidates based on best performers
            if round_idx < optimization_rounds - 1:
                new_candidates = self._generate_variants(
                    pareto_candidates, designer, constraints, length
                )
                candidates = pareto_candidates + new_candidates
        
        # Final Pareto optimization
        final_scores = self._evaluate_objectives(candidates, objectives)
        final_pareto_indices = self._find_pareto_optimal(final_scores, objectives)
        final_structures = [candidates[i] for i in final_pareto_indices]
        
        self.design_statistics['total_designs'] += len(final_structures)
        self.logger.info(f"Generated {len(final_structures)} Pareto-optimal designs")
        
        return final_structures
    
    def design_with_active_learning(
        self,
        initial_constraints: Constraints,
        oracle_function: callable,
        length: int,
        max_iterations: int = 10,
        batch_size: int = 5,
        model_name: str = "default"
    ) -> Tuple[ProteinStructure, List[Dict]]:
        """
        Design proteins using active learning with an oracle function.
        
        Args:
            initial_constraints: Starting constraints
            oracle_function: Function that evaluates designs and returns scores
            length: Target protein length
            max_iterations: Maximum learning iterations
            batch_size: Number of designs per iteration
            model_name: Model to use
            
        Returns:
            Best structure and learning history
        """
        self.logger.info("Starting active learning design")
        
        designer = self.load_model(model_name)
        learning_history = []
        best_structure = None
        best_score = float('-inf')
        
        # Current constraints (will be updated based on feedback)
        current_constraints = initial_constraints
        
        for iteration in range(max_iterations):
            self.logger.info(f"Active learning iteration {iteration + 1}/{max_iterations}")
            
            # Generate batch of candidates
            candidates = []
            for _ in range(batch_size):
                try:
                    structure = designer.generate(
                        constraints=current_constraints,
                        length=length,
                        num_samples=1
                    )
                    candidates.append(structure)
                except Exception as e:
                    self.logger.warning(f"Failed to generate candidate: {e}")
                    continue
            
            if not candidates:
                self.logger.warning("No candidates generated in this iteration")
                continue
            
            # Evaluate candidates with oracle
            scores = []
            feedback_data = []
            
            for candidate in candidates:
                try:
                    score, feedback = oracle_function(candidate)
                    scores.append(score)
                    feedback_data.append(feedback)
                    
                    # Track best structure
                    if score > best_score:
                        best_score = score
                        best_structure = candidate
                        
                except Exception as e:
                    self.logger.warning(f"Oracle evaluation failed: {e}")
                    scores.append(float('-inf'))
                    feedback_data.append({})
            
            # Record iteration data
            iteration_data = {
                'iteration': iteration,
                'scores': scores,
                'best_score': max(scores) if scores else float('-inf'),
                'feedback': feedback_data,
                'num_candidates': len(candidates)
            }
            learning_history.append(iteration_data)
            
            # Update constraints based on feedback
            if feedback_data:
                current_constraints = self._update_constraints_from_feedback(
                    current_constraints, feedback_data, scores
                )
        
        self.logger.info(f"Active learning completed. Best score: {best_score}")
        return best_structure, learning_history
    
    def design_protein_family(
        self,
        seed_structure: ProteinStructure,
        family_size: int,
        diversity_threshold: float = 0.3,
        model_name: str = "default"
    ) -> List[ProteinStructure]:
        """
        Generate a diverse family of proteins based on a seed structure.
        
        Args:
            seed_structure: Starting protein structure
            family_size: Number of family members to generate
            diversity_threshold: Minimum diversity (RMSD) between family members
            model_name: Model to use
            
        Returns:
            List of diverse protein family members
        """
        self.logger.info(f"Generating protein family of size {family_size}")
        
        designer = self.load_model(model_name)
        family_members = [seed_structure]
        
        # Extract constraints from seed structure
        if seed_structure.constraints is None:
            # Infer basic constraints from structure
            constraints = self._infer_constraints_from_structure(seed_structure)
        else:
            constraints = seed_structure.constraints
        
        attempts = 0
        max_attempts = family_size * 10  # Prevent infinite loops
        
        while len(family_members) < family_size and attempts < max_attempts:
            attempts += 1
            
            try:
                # Add some noise to constraints for diversity
                noisy_constraints = self._add_constraint_noise(constraints, noise_level=0.1)
                
                # Generate candidate
                candidate = designer.generate(
                    constraints=noisy_constraints,
                    length=seed_structure.num_residues,
                    num_samples=1
                )
                
                # Check diversity against existing family members
                is_diverse = True
                for member in family_members:
                    rmsd = candidate.compute_rmsd(member)
                    if rmsd < diversity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    family_members.append(candidate)
                    self.logger.info(f"Added family member {len(family_members)}/{family_size}")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate family member: {e}")
                continue
        
        self.logger.info(f"Generated {len(family_members)} family members")
        return family_members
    
    def _evaluate_objectives(
        self,
        candidates: List[ProteinStructure],
        objectives: Dict[str, Tuple[str, float]]
    ) -> List[Dict[str, float]]:
        """Evaluate objective functions for candidate structures."""
        scores = []
        
        for candidate in candidates:
            candidate_scores = {}
            
            # Geometric validation scores
            validation_results = candidate.validate_geometry()
            
            for obj_name, (direction, target) in objectives.items():
                if obj_name == "stability":
                    # Stability based on compactness and bond quality
                    rg = candidate.compute_radius_of_gyration()
                    bond_quality = 1.0 / (1.0 + validation_results.get('avg_bond_deviation', 1.0))
                    stability_score = bond_quality * (1.0 / max(rg, 1.0))
                    candidate_scores[obj_name] = stability_score
                
                elif obj_name == "compactness":
                    rg = candidate.compute_radius_of_gyration()
                    candidate_scores[obj_name] = 1.0 / max(rg, 1.0)
                
                elif obj_name == "clash_avoidance":
                    num_clashes = validation_results.get('num_clashes', 0)
                    candidate_scores[obj_name] = 1.0 / (1.0 + num_clashes)
                
                elif obj_name == "binding_potential":
                    # Simplified binding potential based on surface accessibility
                    contact_map = candidate.compute_contact_map()
                    surface_residues = torch.sum(contact_map, dim=1) < (contact_map.size(0) * 0.3)
                    binding_score = torch.sum(surface_residues).float() / contact_map.size(0)
                    candidate_scores[obj_name] = binding_score.item()
                
                else:
                    # Default: random score (replace with actual evaluation)
                    candidate_scores[obj_name] = torch.rand(1).item()
            
            scores.append(candidate_scores)
        
        return scores
    
    def _find_pareto_optimal(
        self,
        objective_scores: List[Dict[str, float]],
        objectives: Dict[str, Tuple[str, float]]
    ) -> List[int]:
        """Find Pareto-optimal solutions."""
        n_candidates = len(objective_scores)
        is_pareto = [True] * n_candidates
        
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i == j:
                    continue
                
                # Check if j dominates i
                dominates = True
                for obj_name, (direction, target) in objectives.items():
                    score_i = objective_scores[i][obj_name]
                    score_j = objective_scores[j][obj_name]
                    
                    if direction == "maximize":
                        if score_j <= score_i:
                            dominates = False
                            break
                    else:  # minimize
                        if score_j >= score_i:
                            dominates = False
                            break
                
                if dominates:
                    is_pareto[i] = False
                    break
        
        return [i for i, is_opt in enumerate(is_pareto) if is_opt]
    
    def _generate_variants(
        self,
        base_structures: List[ProteinStructure],
        designer: ProteinDesigner,
        constraints: Constraints,
        length: int,
        num_variants_per_base: int = 2
    ) -> List[ProteinStructure]:
        """Generate variants of successful structures."""
        variants = []
        
        for base_structure in base_structures:
            for _ in range(num_variants_per_base):
                try:
                    # Add slight perturbation to constraints
                    perturbed_constraints = self._add_constraint_noise(
                        constraints, noise_level=0.05
                    )
                    
                    variant = designer.generate(
                        constraints=perturbed_constraints,
                        length=length,
                        num_samples=1
                    )
                    variants.append(variant)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate variant: {e}")
                    continue
        
        return variants
    
    def _update_constraints_from_feedback(
        self,
        constraints: Constraints,
        feedback_data: List[Dict],
        scores: List[float]
    ) -> Constraints:
        """Update constraints based on oracle feedback."""
        # This is a simplified implementation
        # In practice, would use more sophisticated constraint learning
        
        # For now, just return the original constraints
        # Real implementation would analyze feedback patterns and adjust
        return constraints
    
    def _infer_constraints_from_structure(
        self, 
        structure: ProteinStructure
    ) -> Constraints:
        """Infer basic constraints from a protein structure."""
        constraints = Constraints()
        
        # Infer secondary structure constraints
        ss_assignment = structure.compute_secondary_structure_simple()
        
        current_ss = None
        start_idx = 0
        
        for i, ss in enumerate(ss_assignment + ['X']):  # Add sentinel
            if ss != current_ss:
                if current_ss is not None and i - start_idx > 2:
                    # Add secondary structure constraint
                    from ..constraints.structural import StructuralConstraint
                    ss_constraint = StructuralConstraint()
                    ss_constraint.add_secondary_structure_region(
                        start_idx, i-1, current_ss
                    )
                    constraints.add_constraint(ss_constraint)
                
                current_ss = ss
                start_idx = i
        
        return constraints
    
    def _add_constraint_noise(
        self, 
        constraints: Constraints, 
        noise_level: float = 0.1
    ) -> Constraints:
        """Add noise to constraints for diversity."""
        # This is a simplified implementation
        # In practice, would add controlled noise to constraint parameters
        return constraints
    
    def get_design_statistics(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        return {
            **self.design_statistics,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'loaded_models': list(self.loaded_models.keys()),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        total_params = 0
        for model in self.loaded_models.values():
            if hasattr(model, 'model'):
                total_params += sum(p.numel() for p in model.model.parameters())
        
        # Rough estimate: 4 bytes per parameter + overhead
        memory_mb = (total_params * 4) / (1024 * 1024)
        return memory_mb
    
    def clear_cache(self):
        """Clear model and design caches."""
        self.design_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Cleared design service caches")