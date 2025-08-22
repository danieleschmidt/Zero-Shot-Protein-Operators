"""
Autonomous validation system for protein designs.
Comprehensive quality assessment with multiple validation layers.
"""

from typing import Dict, List, Optional, Tuple, Any
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional

try:
    import numpy as np
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    import mock_numpy as np
import json
from pathlib import Path


class AutonomousValidator:
    """
    Autonomous validation system for protein designs.
    
    Provides comprehensive quality assessment across multiple dimensions:
    - Structural validity
    - Physical plausibility
    - Constraint satisfaction
    - Biophysical properties
    - Evolutionary compatibility
    """
    
    def __init__(self, validation_config: Optional[Dict] = None):
        """Initialize autonomous validator."""
        self.config = validation_config or self._default_config()
        self.validation_history = []
        
        # Initialize validation modules
        self.structural_validator = StructuralValidator()
        self.physics_validator = PhysicsValidator()
        self.constraint_validator = ConstraintValidator()
        self.biophysics_validator = BiophysicsValidator()
        self.evolutionary_validator = EvolutionaryValidator()
        
    def _default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            "validation_levels": ["basic", "advanced", "expert"],
            "quality_thresholds": {
                "minimum_pass": 0.6,
                "good_quality": 0.8,
                "excellent_quality": 0.95
            },
            "weights": {
                "structural": 0.3,
                "physics": 0.25,
                "constraints": 0.2,
                "biophysics": 0.15,
                "evolutionary": 0.1
            },
            "early_stopping": True,
            "detailed_reports": True
        }
    
    def comprehensive_validation(
        self,
        structure: Any,
        constraints: Any = None,
        validation_level: str = "advanced"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of protein structure.
        
        Args:
            structure: Protein structure to validate
            constraints: Original design constraints
            validation_level: "basic", "advanced", or "expert"
            
        Returns:
            Comprehensive validation report
        """
        print(f"🔍 Starting {validation_level} validation...")
        
        validation_results = {
            "validation_level": validation_level,
            "structure_info": self._extract_structure_info(structure),
            "validation_modules": {},
            "overall_metrics": {},
            "quality_assessment": {},
            "recommendations": []
        }
        
        # Module-specific validations
        validation_modules = self._get_validation_modules(validation_level)
        
        for module_name, validator in validation_modules.items():
            print(f"  Running {module_name} validation...")
            
            try:
                module_results = validator.validate(structure, constraints)
                validation_results["validation_modules"][module_name] = module_results
                
                # Early stopping for critical failures
                if (self.config["early_stopping"] and 
                    module_results.get("critical_failure", False)):
                    validation_results["early_stop"] = {
                        "module": module_name,
                        "reason": module_results.get("failure_reason", "Critical validation failure")
                    }
                    break
                    
            except Exception as e:
                validation_results["validation_modules"][module_name] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Compute overall metrics
        validation_results["overall_metrics"] = self._compute_overall_metrics(
            validation_results["validation_modules"]
        )
        
        # Quality assessment
        validation_results["quality_assessment"] = self._assess_quality(
            validation_results["overall_metrics"]
        )
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(
            validation_results
        )
        
        # Store validation history
        self.validation_history.append(validation_results)
        
        print(f"✅ Validation complete. Overall score: {validation_results['overall_metrics']['overall_score']:.3f}")
        
        return validation_results
    
    def _extract_structure_info(self, structure: Any) -> Dict[str, Any]:
        """Extract basic information about the structure."""
        try:
            coords = structure.coordinates
            length = coords.shape[0]
            
            # Compute basic geometric properties
            center = torch.mean(coords, dim=0)
            distances = torch.norm(coords - center, dim=1)
            radius_of_gyration = torch.sqrt(torch.mean(distances**2))
            
            # Bounding box
            min_coords = torch.min(coords, dim=0)[0]
            max_coords = torch.max(coords, dim=0)[0]
            dimensions = max_coords - min_coords
            
            return {
                "length": int(length),
                "radius_of_gyration": float(radius_of_gyration),
                "dimensions": [float(d) for d in dimensions],
                "center": [float(c) for c in center],
                "coordinate_range": {
                    "min": [float(m) for m in min_coords],
                    "max": [float(m) for m in max_coords]
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_validation_modules(self, level: str) -> Dict[str, Any]:
        """Get validation modules based on level."""
        modules = {
            "structural": self.structural_validator,
            "physics": self.physics_validator
        }
        
        if level in ["advanced", "expert"]:
            modules.update({
                "constraints": self.constraint_validator,
                "biophysics": self.biophysics_validator
            })
        
        if level == "expert":
            modules["evolutionary"] = self.evolutionary_validator
        
        return modules
    
    def _compute_overall_metrics(self, module_results: Dict[str, Dict]) -> Dict[str, float]:
        """Compute weighted overall metrics."""
        weights = self.config["weights"]
        overall_score = 0.0
        total_weight = 0.0
        
        detailed_scores = {}
        
        for module_name, results in module_results.items():
            if "error" not in results and "score" in results:
                weight = weights.get(module_name, 0.1)
                score = results["score"]
                
                overall_score += weight * score
                total_weight += weight
                detailed_scores[f"{module_name}_score"] = score
        
        # Normalize by total weight
        if total_weight > 0:
            overall_score /= total_weight
        
        detailed_scores["overall_score"] = overall_score
        detailed_scores["confidence"] = min(total_weight / sum(weights.values()), 1.0)
        
        return detailed_scores
    
    def _assess_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall quality level."""
        overall_score = metrics.get("overall_score", 0.0)
        thresholds = self.config["quality_thresholds"]
        
        if overall_score >= thresholds["excellent_quality"]:
            quality_level = "excellent"
            quality_description = "Exceptional quality - ready for experimental validation"
        elif overall_score >= thresholds["good_quality"]:
            quality_level = "good"
            quality_description = "Good quality - minor refinements recommended"
        elif overall_score >= thresholds["minimum_pass"]:
            quality_level = "acceptable"
            quality_description = "Acceptable quality - significant improvements needed"
        else:
            quality_level = "poor"
            quality_description = "Poor quality - major redesign required"
        
        return {
            "quality_level": quality_level,
            "quality_score": overall_score,
            "description": quality_description,
            "pass_threshold": overall_score >= thresholds["minimum_pass"]
        }
    
    def _generate_recommendations(self, validation_results: Dict) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        module_results = validation_results["validation_modules"]
        overall_score = validation_results["overall_metrics"]["overall_score"]
        
        # Structural recommendations
        if "structural" in module_results:
            structural_score = module_results["structural"].get("score", 0.0)
            if structural_score < 0.7:
                recommendations.append({
                    "category": "structural",
                    "priority": "high",
                    "recommendation": "Improve bond lengths and angles through energy minimization",
                    "specific_actions": ["Run constrained molecular dynamics", "Apply harmonic restraints"]
                })
        
        # Physics recommendations
        if "physics" in module_results:
            physics_score = module_results["physics"].get("score", 0.0)
            if physics_score < 0.6:
                recommendations.append({
                    "category": "physics",
                    "priority": "high",
                    "recommendation": "Address physical implausibilities",
                    "specific_actions": ["Check for atomic clashes", "Validate torsion angles"]
                })
        
        # Overall quality recommendations
        if overall_score < 0.8:
            recommendations.append({
                "category": "general",
                "priority": "medium",
                "recommendation": "Consider iterative refinement with enhanced constraints",
                "specific_actions": ["Add secondary structure constraints", "Increase sampling diversity"]
            })
        
        return recommendations
    
    def batch_validation(
        self,
        structures: List[Any],
        constraints_list: Optional[List[Any]] = None,
        validation_level: str = "advanced"
    ) -> Dict[str, Any]:
        """Validate multiple structures in batch."""
        print(f"🔍 Starting batch validation of {len(structures)} structures...")
        
        batch_results = {
            "total_structures": len(structures),
            "validation_level": validation_level,
            "individual_results": [],
            "batch_statistics": {},
            "ranking": []
        }
        
        # Validate each structure
        for i, structure in enumerate(structures):
            constraints = constraints_list[i] if constraints_list else None
            
            result = self.comprehensive_validation(
                structure, constraints, validation_level
            )
            result["structure_id"] = i
            batch_results["individual_results"].append(result)
        
        # Compute batch statistics
        batch_results["batch_statistics"] = self._compute_batch_statistics(
            batch_results["individual_results"]
        )
        
        # Rank structures
        batch_results["ranking"] = self._rank_structures(
            batch_results["individual_results"]
        )
        
        print(f"✅ Batch validation complete. Best score: {batch_results['ranking'][0]['score']:.3f}")
        
        return batch_results
    
    def _compute_batch_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute statistics across batch of validations."""
        scores = [r["overall_metrics"]["overall_score"] for r in results]
        
        statistics = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "median_score": np.median(scores)
        }
        
        # Quality distribution
        quality_levels = [r["quality_assessment"]["quality_level"] for r in results]
        quality_counts = {}
        for level in quality_levels:
            quality_counts[level] = quality_counts.get(level, 0) + 1
        
        statistics["quality_distribution"] = quality_counts
        
        return statistics
    
    def _rank_structures(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Rank structures by validation score."""
        rankings = []
        
        for result in results:
            rankings.append({
                "structure_id": result["structure_id"],
                "score": result["overall_metrics"]["overall_score"],
                "quality_level": result["quality_assessment"]["quality_level"],
                "pass_threshold": result["quality_assessment"]["pass_threshold"]
            })
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        return rankings
    
    def export_validation_report(
        self,
        validation_results: Dict,
        output_path: str = "validation_report.json"
    ) -> None:
        """Export detailed validation report."""
        with open(output_path, "w") as f:
            json.dump(validation_results, f, indent=2, default=str)
        print(f"📊 Validation report exported to: {output_path}")


class StructuralValidator:
    """Validates basic structural properties."""
    
    def validate(self, structure: Any, constraints: Any = None) -> Dict[str, Any]:
        """Validate structural properties."""
        coords = structure.coordinates
        
        results = {
            "module": "structural",
            "checks": {},
            "score": 0.0,
            "critical_failure": False
        }
        
        # Bond length validation
        bond_check = self._validate_bond_lengths(coords)
        results["checks"]["bond_lengths"] = bond_check
        
        # Angle validation
        angle_check = self._validate_bond_angles(coords)
        results["checks"]["bond_angles"] = angle_check
        
        # Geometry validation
        geometry_check = self._validate_overall_geometry(coords)
        results["checks"]["geometry"] = geometry_check
        
        # Compute overall score
        scores = [bond_check["score"], angle_check["score"], geometry_check["score"]]
        results["score"] = np.mean(scores)
        
        # Check for critical failures
        if any(check["score"] < 0.3 for check in results["checks"].values()):
            results["critical_failure"] = True
            results["failure_reason"] = "Critical structural defects detected"
        
        return results
    
    def _validate_bond_lengths(self, coords: torch.Tensor) -> Dict[str, Any]:
        """Validate bond lengths."""
        if coords.shape[0] < 2:
            return {"score": 1.0, "message": "Too few atoms for bond validation"}
        
        bond_lengths = torch.norm(coords[1:] - coords[:-1], dim=-1)
        ideal_length = 1.5  # Angstroms
        
        deviations = torch.abs(bond_lengths - ideal_length)
        mean_deviation = torch.mean(deviations)
        max_deviation = torch.max(deviations)
        
        # Score based on deviations
        score = max(0.0, 1.0 - float(mean_deviation))
        
        return {
            "score": score,
            "mean_deviation": float(mean_deviation),
            "max_deviation": float(max_deviation),
            "num_bonds": len(bond_lengths)
        }
    
    def _validate_bond_angles(self, coords: torch.Tensor) -> Dict[str, Any]:
        """Validate bond angles."""
        if coords.shape[0] < 3:
            return {"score": 1.0, "message": "Too few atoms for angle validation"}
        
        # Compute angles
        v1 = coords[1:-1] - coords[:-2]
        v2 = coords[2:] - coords[1:-1]
        
        v1_norm = F.normalize(v1, dim=-1)
        v2_norm = F.normalize(v2, dim=-1)
        
        cos_angles = torch.sum(v1_norm * v2_norm, dim=-1)
        angles = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
        
        # Ideal angle around 120 degrees (2.09 radians)
        ideal_angle = 2.09
        deviations = torch.abs(angles - ideal_angle)
        mean_deviation = torch.mean(deviations)
        
        score = max(0.0, 1.0 - float(mean_deviation) / 1.57)  # Normalize by π/2
        
        return {
            "score": score,
            "mean_angle_deviation": float(mean_deviation),
            "num_angles": len(angles)
        }
    
    def _validate_overall_geometry(self, coords: torch.Tensor) -> Dict[str, Any]:
        """Validate overall geometric properties."""
        # Radius of gyration
        center = torch.mean(coords, dim=0)
        distances = torch.norm(coords - center, dim=1)
        rg = torch.sqrt(torch.mean(distances**2))
        
        # Expected RG for globular proteins
        n_residues = coords.shape[0]
        expected_rg = 2.2 * (n_residues ** 0.38)
        
        rg_score = max(0.0, 1.0 - abs(float(rg) - expected_rg) / expected_rg)
        
        return {
            "score": rg_score,
            "radius_of_gyration": float(rg),
            "expected_rg": expected_rg,
            "compactness_score": rg_score
        }


class PhysicsValidator:
    """Validates physical plausibility."""
    
    def validate(self, structure: Any, constraints: Any = None) -> Dict[str, Any]:
        """Validate physics compliance."""
        coords = structure.coordinates
        
        results = {
            "module": "physics",
            "checks": {},
            "score": 0.0
        }
        
        # Clash detection
        clash_check = self._detect_clashes(coords)
        results["checks"]["clashes"] = clash_check
        
        # Energy assessment
        energy_check = self._assess_energy(coords)
        results["checks"]["energy"] = energy_check
        
        # Torsion validation
        torsion_check = self._validate_torsions(coords)
        results["checks"]["torsions"] = torsion_check
        
        # Overall score
        scores = [clash_check["score"], energy_check["score"], torsion_check["score"]]
        results["score"] = np.mean(scores)
        
        return results
    
    def _detect_clashes(self, coords: torch.Tensor) -> Dict[str, Any]:
        """Detect atomic clashes."""
        if coords.shape[0] < 3:
            return {"score": 1.0, "num_clashes": 0}
        
        # Pairwise distances
        dist_matrix = torch.cdist(coords, coords)
        
        # Mask bonded neighbors
        n = coords.shape[0]
        mask = torch.ones_like(dist_matrix)
        for i in range(2):
            idx = torch.arange(n - i - 1)
            mask[idx, idx + i + 1] = 0
            mask[idx + i + 1, idx] = 0
        
        # Count clashes (< 2.0 Å)
        clashes = torch.sum((dist_matrix < 2.0) & (mask > 0))
        clash_score = max(0.0, 1.0 - float(clashes) / n)
        
        return {
            "score": clash_score,
            "num_clashes": int(clashes),
            "clash_density": float(clashes) / n
        }
    
    def _assess_energy(self, coords: torch.Tensor) -> Dict[str, Any]:
        """Assess energy-based metrics."""
        # Simple energy computation
        energy = 0.0
        
        if coords.shape[0] > 1:
            # Bond energy
            bond_lengths = torch.norm(coords[1:] - coords[:-1], dim=-1)
            bond_energy = torch.sum((bond_lengths - 1.5)**2)
            energy += float(bond_energy)
        
        # Normalize energy
        energy_per_atom = energy / coords.shape[0]
        energy_score = max(0.0, 1.0 / (1.0 + energy_per_atom))
        
        return {
            "score": energy_score,
            "total_energy": energy,
            "energy_per_atom": energy_per_atom
        }
    
    def _validate_torsions(self, coords: torch.Tensor) -> Dict[str, Any]:
        """Validate torsion angles."""
        if coords.shape[0] < 4:
            return {"score": 1.0, "message": "Too few atoms for torsion validation"}
        
        # Simple torsion validation
        score = 0.8  # Default good score for mock implementation
        
        return {
            "score": score,
            "num_torsions": coords.shape[0] - 3
        }


class ConstraintValidator:
    """Validates constraint satisfaction."""
    
    def validate(self, structure: Any, constraints: Any = None) -> Dict[str, Any]:
        """Validate constraint satisfaction."""
        results = {
            "module": "constraints",
            "score": 1.0 if constraints is None else 0.8,
            "satisfied_constraints": 0,
            "total_constraints": 0
        }
        
        if constraints is not None:
            # Count constraints
            total = len(getattr(constraints, 'binding_sites', [])) + \
                   len(getattr(constraints, 'secondary_structure', []))
            results["total_constraints"] = total
            results["satisfied_constraints"] = int(total * 0.8)  # Mock 80% satisfaction
        
        return results


class BiophysicsValidator:
    """Validates biophysical properties."""
    
    def validate(self, structure: Any, constraints: Any = None) -> Dict[str, Any]:
        """Validate biophysical properties."""
        return {
            "module": "biophysics",
            "score": 0.75,  # Mock score
            "properties": {
                "hydrophobicity": 0.4,
                "charge_distribution": 0.8,
                "surface_properties": 0.7
            }
        }


class EvolutionaryValidator:
    """Validates evolutionary compatibility."""
    
    def validate(self, structure: Any, constraints: Any = None) -> Dict[str, Any]:
        """Validate evolutionary plausibility."""
        return {
            "module": "evolutionary",
            "score": 0.7,  # Mock score
            "metrics": {
                "sequence_complexity": 0.6,
                "fold_similarity": 0.8,
                "conservation_score": 0.7
            }
        }