"""
Comprehensive protein structure validation service.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
import logging
from dataclasses import dataclass

from ..structure import ProteinStructure


@dataclass
class ValidationResult:
    """Container for validation results."""
    overall_score: float
    individual_scores: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    passed: bool


class ValidationService:
    """
    Comprehensive protein structure validation service.
    
    Provides multiple validation levels from basic geometry
    to advanced biophysical property predictions.
    """
    
    def __init__(self, validation_level: str = "standard"):
        """
        Initialize validation service.
        
        Args:
            validation_level: "basic", "standard", or "comprehensive"
        """
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.thresholds = {
            'bond_deviation_max': 0.5,  # Angstroms
            'bond_deviation_avg': 0.2,  # Angstroms
            'clash_cutoff': 2.0,  # Angstroms
            'max_clashes': 5,
            'min_radius_gyration': 5.0,  # Angstroms
            'max_radius_gyration': 50.0,  # Angstroms
            'ramachandran_outlier_max': 0.05,  # 5% max outliers
            'secondary_structure_min_length': 3,
        }
        
        # Scoring weights
        self.score_weights = {
            'stereochemistry': 0.3,
            'clashes': 0.3,
            'compactness': 0.2,
            'ramachandran': 0.2,
        }
    
    def validate_structure(
        self, 
        structure: ProteinStructure,
        detailed: bool = True
    ) -> ValidationResult:
        """
        Perform comprehensive structure validation.
        
        Args:
            structure: Protein structure to validate
            detailed: Whether to perform detailed analysis
            
        Returns:
            ValidationResult with scores and recommendations
        """
        self.logger.info(f"Validating structure with {structure.num_residues} residues")
        
        individual_scores = {}
        warnings = []
        errors = []
        recommendations = []
        
        # 1. Basic geometry validation
        geom_result = self._validate_geometry(structure)
        individual_scores.update(geom_result['scores'])
        warnings.extend(geom_result['warnings'])
        errors.extend(geom_result['errors'])
        
        # 2. Stereochemical validation
        stereo_result = self._validate_stereochemistry(structure)
        individual_scores.update(stereo_result['scores'])
        warnings.extend(stereo_result['warnings'])
        
        # 3. Clash detection
        clash_result = self._validate_clashes(structure)
        individual_scores.update(clash_result['scores'])
        warnings.extend(clash_result['warnings'])
        
        # 4. Compactness analysis
        compact_result = self._validate_compactness(structure)
        individual_scores.update(compact_result['scores'])
        warnings.extend(compact_result['warnings'])
        
        if detailed and self.validation_level in ["standard", "comprehensive"]:
            # 5. Ramachandran validation (simplified)
            rama_result = self._validate_ramachandran(structure)
            individual_scores.update(rama_result['scores'])
            warnings.extend(rama_result['warnings'])
            
            # 6. Secondary structure validation
            ss_result = self._validate_secondary_structure(structure)
            individual_scores.update(ss_result['scores'])
            warnings.extend(ss_result['warnings'])
        
        if self.validation_level == "comprehensive":
            # 7. Advanced biophysical validation
            biophys_result = self._validate_biophysical_properties(structure)
            individual_scores.update(biophys_result['scores'])
            warnings.extend(biophys_result['warnings'])
            recommendations.extend(biophys_result['recommendations'])
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(individual_scores)
        
        # Determine pass/fail
        passed = self._determine_pass_fail(individual_scores, errors)
        
        # Generate recommendations
        recommendations.extend(self._generate_recommendations(individual_scores, warnings))
        
        return ValidationResult(
            overall_score=overall_score,
            individual_scores=individual_scores,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            passed=passed
        )
    
    def _validate_geometry(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate basic geometric properties."""
        result = {'scores': {}, 'warnings': [], 'errors': []}
        
        if structure.num_residues < 2:
            result['errors'].append("Structure too small for geometry validation")
            result['scores']['geometry'] = 0.0
            return result
        
        # Bond length validation
        coords = structure.coordinates
        bond_vectors = coords[1:] - coords[:-1]
        bond_lengths = torch.norm(bond_vectors, dim=-1)
        
        ideal_ca_distance = 3.8  # Angstroms
        bond_deviations = torch.abs(bond_lengths - ideal_ca_distance)
        
        avg_deviation = torch.mean(bond_deviations).item()
        max_deviation = torch.max(bond_deviations).item()
        
        result['scores']['bond_deviation_avg'] = max(0, 1 - avg_deviation / self.thresholds['bond_deviation_avg'])
        result['scores']['bond_deviation_max'] = max(0, 1 - max_deviation / self.thresholds['bond_deviation_max'])
        
        if avg_deviation > self.thresholds['bond_deviation_avg']:
            result['warnings'].append(f"Average bond deviation ({avg_deviation:.2f} Å) exceeds threshold")
        
        if max_deviation > self.thresholds['bond_deviation_max']:
            result['warnings'].append(f"Maximum bond deviation ({max_deviation:.2f} Å) exceeds threshold")
        
        # Angle validation (simplified)
        if structure.num_residues > 2:
            angle_scores = []
            for i in range(structure.num_residues - 2):
                v1 = coords[i+1] - coords[i]
                v2 = coords[i+2] - coords[i+1]
                
                v1_norm = F.normalize(v1.unsqueeze(0), dim=1).squeeze(0)
                v2_norm = F.normalize(v2.unsqueeze(0), dim=1).squeeze(0)
                
                cos_angle = torch.dot(v1_norm, v2_norm)
                angle = torch.acos(torch.clamp(cos_angle, -0.999, 0.999))
                
                # Ideal angle range: 90-150 degrees
                ideal_min = np.pi / 2
                ideal_max = 5 * np.pi / 6
                
                if ideal_min <= angle <= ideal_max:
                    angle_scores.append(1.0)
                else:
                    deviation = min(abs(angle - ideal_min), abs(angle - ideal_max))
                    score = max(0, 1 - deviation / (np.pi / 6))
                    angle_scores.append(score)
            
            result['scores']['angle_quality'] = np.mean(angle_scores) if angle_scores else 0.0
        
        return result
    
    def _validate_stereochemistry(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate stereochemical properties."""
        result = {'scores': {}, 'warnings': []}
        
        # Use built-in geometry validation
        geom_results = structure.validate_geometry()
        
        bond_score = 1.0 / (1.0 + geom_results.get('avg_bond_deviation', 1.0))
        result['scores']['stereochemistry'] = bond_score
        
        if bond_score < 0.7:
            result['warnings'].append("Poor stereochemical quality detected")
        
        return result
    
    def _validate_clashes(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate atomic clashes."""
        result = {'scores': {}, 'warnings': []}
        
        if structure.num_residues < 3:
            result['scores']['clashes'] = 1.0
            return result
        
        # Compute distance matrix
        dist_matrix = structure.compute_distance_matrix()
        
        # Mask bonded neighbors
        n = structure.num_residues
        mask = torch.ones_like(dist_matrix)
        for i in range(3):
            idx = torch.arange(n - i)
            mask[idx, idx + i] = 0
            if i > 0:
                mask[idx + i, idx] = 0
        
        # Count clashes
        clash_mask = (dist_matrix < self.thresholds['clash_cutoff']) & (mask > 0)
        num_clashes = torch.sum(clash_mask).item() // 2  # Divide by 2 to avoid double counting
        
        clash_score = max(0, 1 - num_clashes / self.thresholds['max_clashes'])
        result['scores']['clashes'] = clash_score
        
        if num_clashes > 0:
            result['warnings'].append(f"Detected {num_clashes} atomic clashes")
        
        return result
    
    def _validate_compactness(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate protein compactness."""
        result = {'scores': {}, 'warnings': []}
        
        rg = structure.compute_radius_of_gyration()
        
        # Score based on expected radius of gyration for protein size
        expected_rg = 2.2 * (structure.num_residues ** 0.38)  # Empirical relationship
        
        rg_ratio = rg / expected_rg
        
        # Optimal range: 0.8 - 1.2 of expected
        if 0.8 <= rg_ratio <= 1.2:
            compactness_score = 1.0
        else:
            deviation = min(abs(rg_ratio - 0.8), abs(rg_ratio - 1.2))
            compactness_score = max(0, 1 - deviation)
        
        result['scores']['compactness'] = compactness_score
        result['scores']['radius_gyration'] = rg
        
        if rg < self.thresholds['min_radius_gyration']:
            result['warnings'].append(f"Structure too compact (Rg = {rg:.1f} Å)")
        elif rg > self.thresholds['max_radius_gyration']:
            result['warnings'].append(f"Structure too extended (Rg = {rg:.1f} Å)")
        
        return result
    
    def _validate_ramachandran(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate Ramachandran plot conformance (simplified)."""
        result = {'scores': {}, 'warnings': []}
        
        if structure.num_residues < 4:
            result['scores']['ramachandran'] = 1.0
            return result
        
        # Simplified Ramachandran validation based on local geometry
        coords = structure.coordinates
        outlier_count = 0
        
        for i in range(1, structure.num_residues - 2):
            # Compute dihedral angles (simplified)
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            v3 = coords[i+2] - coords[i+1]
            
            # Cross products for dihedral calculation
            n1 = torch.cross(v1, v2)
            n2 = torch.cross(v2, v3)
            
            n1_norm = torch.norm(n1)
            n2_norm = torch.norm(n2)
            
            if n1_norm > 1e-6 and n2_norm > 1e-6:
                cos_dihedral = torch.dot(n1, n2) / (n1_norm * n2_norm)
                cos_dihedral = torch.clamp(cos_dihedral, -0.999, 0.999)
                dihedral = torch.acos(cos_dihedral)
                
                # Simplified check: very acute dihedrals are usually outliers
                if dihedral < np.pi / 6 or dihedral > 5 * np.pi / 6:
                    outlier_count += 1
        
        outlier_fraction = outlier_count / max(structure.num_residues - 3, 1)
        rama_score = max(0, 1 - outlier_fraction / self.thresholds['ramachandran_outlier_max'])
        
        result['scores']['ramachandran'] = rama_score
        
        if outlier_fraction > self.thresholds['ramachandran_outlier_max']:
            result['warnings'].append(f"High Ramachandran outlier fraction: {outlier_fraction:.1%}")
        
        return result
    
    def _validate_secondary_structure(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate secondary structure elements."""
        result = {'scores': {}, 'warnings': []}
        
        ss_assignment = structure.compute_secondary_structure_simple()
        
        # Analyze secondary structure elements
        ss_elements = []
        current_ss = None
        start_idx = 0
        
        for i, ss in enumerate(ss_assignment + ['X']):  # Add sentinel
            if ss != current_ss:
                if current_ss is not None:
                    length = i - start_idx
                    ss_elements.append((current_ss, start_idx, i-1, length))
                current_ss = ss
                start_idx = i
        
        # Validate element lengths
        short_elements = 0
        for ss_type, start, end, length in ss_elements:
            if ss_type in ['H', 'E'] and length < self.thresholds['secondary_structure_min_length']:
                short_elements += 1
        
        ss_quality = max(0, 1 - short_elements / max(len(ss_elements), 1))
        result['scores']['secondary_structure'] = ss_quality
        
        if short_elements > 0:
            result['warnings'].append(f"Found {short_elements} very short secondary structure elements")
        
        # Calculate secondary structure content
        helix_content = ss_assignment.count('H') / len(ss_assignment)
        sheet_content = ss_assignment.count('E') / len(ss_assignment)
        coil_content = ss_assignment.count('C') / len(ss_assignment)
        
        result['scores']['helix_content'] = helix_content
        result['scores']['sheet_content'] = sheet_content
        result['scores']['coil_content'] = coil_content
        
        return result
    
    def _validate_biophysical_properties(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Validate advanced biophysical properties."""
        result = {'scores': {}, 'warnings': [], 'recommendations': []}
        
        # Contact order analysis
        contact_map = structure.compute_contact_map(cutoff=8.0)
        
        # Calculate relative contact order
        total_contacts = 0
        contact_order_sum = 0
        
        n = contact_map.size(0)
        for i in range(n):
            for j in range(i + 1, n):
                if contact_map[i, j] > 0:
                    total_contacts += 1
                    contact_order_sum += abs(j - i)
        
        if total_contacts > 0:
            relative_contact_order = contact_order_sum / (total_contacts * n)
            result['scores']['relative_contact_order'] = min(1.0, relative_contact_order)
            
            if relative_contact_order < 0.1:
                result['warnings'].append("Very low relative contact order - may indicate poor folding")
                result['recommendations'].append("Consider adding long-range contact constraints")
        
        # Hydrophobic core analysis (simplified)
        # Assume central residues should be hydrophobic
        center_indices = list(range(n // 3, 2 * n // 3))
        surface_indices = list(range(n // 4)) + list(range(3 * n // 4, n))
        
        # Simplified hydrophobicity score based on contact density
        core_contacts = 0
        surface_contacts = 0
        
        for i in center_indices:
            core_contacts += torch.sum(contact_map[i, :]).item()
        
        for i in surface_indices:
            surface_contacts += torch.sum(contact_map[i, :]).item()
        
        if len(center_indices) > 0 and len(surface_indices) > 0:
            core_density = core_contacts / len(center_indices)
            surface_density = surface_contacts / len(surface_indices)
            
            hydrophobic_score = min(1.0, core_density / max(surface_density, 1e-6))
            result['scores']['hydrophobic_core'] = hydrophobic_score
            
            if hydrophobic_score < 0.5:
                result['recommendations'].append("Consider improving hydrophobic core packing")
        
        return result
    
    def _calculate_overall_score(self, individual_scores: Dict[str, float]) -> float:
        """Calculate weighted overall validation score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, weight in self.score_weights.items():
            if category in individual_scores:
                weighted_sum += individual_scores[category] * weight
                total_weight += weight
        
        # Add other scores with equal weight
        other_scores = []
        for key, score in individual_scores.items():
            if key not in self.score_weights:
                other_scores.append(score)
        
        if other_scores:
            remaining_weight = 1.0 - total_weight
            other_weight = remaining_weight / len(other_scores)
            weighted_sum += sum(other_scores) * other_weight
            total_weight += remaining_weight
        
        return weighted_sum / max(total_weight, 1e-6)
    
    def _determine_pass_fail(
        self, 
        individual_scores: Dict[str, float], 
        errors: List[str]
    ) -> bool:
        """Determine if structure passes validation."""
        if errors:
            return False
        
        # Require minimum scores in critical categories
        critical_thresholds = {
            'stereochemistry': 0.5,
            'clashes': 0.7,
            'compactness': 0.3,
        }
        
        for category, threshold in critical_thresholds.items():
            if category in individual_scores and individual_scores[category] < threshold:
                return False
        
        return True
    
    def _generate_recommendations(
        self, 
        individual_scores: Dict[str, float], 
        warnings: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if individual_scores.get('stereochemistry', 1.0) < 0.7:
            recommendations.append("Perform energy minimization to improve bond geometry")
        
        if individual_scores.get('clashes', 1.0) < 0.8:
            recommendations.append("Use clash detection and resolution algorithms")
        
        if individual_scores.get('compactness', 1.0) < 0.5:
            recommendations.append("Adjust constraints to improve protein compactness")
        
        if individual_scores.get('ramachandran', 1.0) < 0.8:
            recommendations.append("Consider backbone refinement to improve Ramachandran conformance")
        
        if len(warnings) > 5:
            recommendations.append("Structure has multiple issues - consider redesigning from scratch")
        
        return recommendations
    
    def batch_validate(
        self, 
        structures: List[ProteinStructure],
        parallel: bool = True
    ) -> List[ValidationResult]:
        """Validate multiple structures."""
        self.logger.info(f"Batch validating {len(structures)} structures")
        
        results = []
        for i, structure in enumerate(structures):
            try:
                result = self.validate_structure(structure)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Validated {i + 1}/{len(structures)} structures")
                    
            except Exception as e:
                self.logger.error(f"Validation failed for structure {i}: {e}")
                # Create failed result
                failed_result = ValidationResult(
                    overall_score=0.0,
                    individual_scores={},
                    warnings=[],
                    errors=[f"Validation failed: {str(e)}"],
                    recommendations=[],
                    passed=False
                )
                results.append(failed_result)
        
        return results