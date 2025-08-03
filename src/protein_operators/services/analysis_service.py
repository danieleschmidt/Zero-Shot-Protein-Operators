"""
Advanced protein structure analysis and prediction service.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from dataclasses import dataclass

from ..structure import ProteinStructure


@dataclass
class AnalysisReport:
    """Container for comprehensive analysis results."""
    structure_metrics: Dict[str, float]
    biophysical_properties: Dict[str, float]
    binding_predictions: Dict[str, Any]
    stability_analysis: Dict[str, float]
    designability_score: float
    drug_likeness: Dict[str, float]
    comparisons: Dict[str, Any]
    visualizations: Dict[str, Any]


class AnalysisService:
    """
    Advanced protein structure analysis service.
    
    Provides comprehensive analysis including biophysical property
    prediction, binding site identification, stability analysis,
    and comparative structural analysis.
    """
    
    def __init__(self, reference_database_path: Optional[str] = None):
        """Initialize analysis service."""
        self.reference_db_path = reference_database_path
        self.logger = logging.getLogger(__name__)
        
        # Analysis caches
        self.analysis_cache = {}
        self.reference_structures = {}
        
        # Prediction models (would be loaded from files in production)
        self.stability_model = None
        self.binding_model = None
        self.folding_model = None
        
        # Analysis parameters
        self.params = {
            'contact_cutoff': 8.0,
            'binding_site_cutoff': 12.0,
            'cavity_probe_radius': 1.4,
            'surface_accessibility_cutoff': 0.2,
            'hydrophobicity_window': 9,
            'flexibility_window': 5,
        }
    
    def analyze_structure(
        self, 
        structure: ProteinStructure,
        analysis_level: str = "comprehensive",
        compare_to_database: bool = True
    ) -> AnalysisReport:
        """
        Perform comprehensive structure analysis.
        
        Args:
            structure: Protein structure to analyze
            analysis_level: "basic", "standard", or "comprehensive"
            compare_to_database: Whether to compare against reference structures
            
        Returns:
            AnalysisReport with detailed analysis results
        """
        self.logger.info(f"Analyzing structure with {structure.num_residues} residues")
        
        # Basic structural metrics
        structure_metrics = self._compute_structure_metrics(structure)
        
        # Biophysical property predictions
        biophysical_properties = self._predict_biophysical_properties(structure)
        
        # Binding site predictions
        binding_predictions = self._predict_binding_sites(structure)
        
        # Stability analysis
        stability_analysis = self._analyze_stability(structure)
        
        # Designability assessment
        designability_score = self._assess_designability(structure)
        
        # Drug-likeness analysis
        drug_likeness = self._analyze_drug_likeness(structure)
        
        # Comparative analysis
        comparisons = {}
        if compare_to_database and analysis_level == "comprehensive":
            comparisons = self._compare_to_database(structure)
        
        # Generate visualizations data
        visualizations = self._generate_visualization_data(structure)
        
        return AnalysisReport(
            structure_metrics=structure_metrics,
            biophysical_properties=biophysical_properties,
            binding_predictions=binding_predictions,
            stability_analysis=stability_analysis,
            designability_score=designability_score,
            drug_likeness=drug_likeness,
            comparisons=comparisons,
            visualizations=visualizations
        )
    
    def _compute_structure_metrics(self, structure: ProteinStructure) -> Dict[str, float]:
        """Compute basic structural metrics."""
        metrics = {}
        
        # Basic geometric properties
        coords = structure.coordinates
        metrics['num_residues'] = structure.num_residues
        metrics['radius_gyration'] = structure.compute_radius_of_gyration()
        
        # Distance-based metrics
        dist_matrix = structure.compute_distance_matrix()
        
        # Average pairwise distance
        upper_tri_mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1)
        pairwise_distances = dist_matrix[upper_tri_mask.bool()]
        metrics['avg_pairwise_distance'] = torch.mean(pairwise_distances).item()
        metrics['max_pairwise_distance'] = torch.max(pairwise_distances).item()
        
        # Contact density
        contact_map = structure.compute_contact_map(cutoff=self.params['contact_cutoff'])
        total_possible_contacts = structure.num_residues * (structure.num_residues - 1) / 2
        metrics['contact_density'] = torch.sum(contact_map).item() / total_possible_contacts
        
        # Asphericity (shape anisotropy)
        if structure.num_residues > 2:
            centered_coords = coords - torch.mean(coords, dim=0)
            gyration_tensor = torch.matmul(centered_coords.T, centered_coords) / structure.num_residues
            eigenvalues = torch.linalg.eigvals(gyration_tensor)
            eigenvalues = torch.sort(eigenvalues.real, descending=True)[0]
            
            lambda1, lambda2, lambda3 = eigenvalues
            asphericity = lambda1 - 0.5 * (lambda2 + lambda3)
            metrics['asphericity'] = asphericity.item()
            
            # Acylindricity
            acylindricity = lambda2 - lambda3
            metrics['acylindricity'] = acylindricity.item()
        
        # Secondary structure content
        ss_assignment = structure.compute_secondary_structure_simple()
        metrics['helix_fraction'] = ss_assignment.count('H') / len(ss_assignment)
        metrics['sheet_fraction'] = ss_assignment.count('E') / len(ss_assignment)
        metrics['coil_fraction'] = ss_assignment.count('C') / len(ss_assignment)
        
        # Compactness metrics
        convex_hull_volume = self._estimate_convex_hull_volume(coords)
        metrics['convex_hull_volume'] = convex_hull_volume
        metrics['packing_density'] = structure.num_residues / max(convex_hull_volume, 1e-6)
        
        return metrics
    
    def _predict_biophysical_properties(self, structure: ProteinStructure) -> Dict[str, float]:
        """Predict various biophysical properties."""
        properties = {}
        
        # Hydrophobicity analysis
        hydrophobicity = self._analyze_hydrophobicity(structure)
        properties.update(hydrophobicity)
        
        # Surface accessibility
        surface_analysis = self._analyze_surface_accessibility(structure)
        properties.update(surface_analysis)
        
        # Flexibility prediction
        flexibility = self._predict_flexibility(structure)
        properties.update(flexibility)
        
        # Electrostatic properties (simplified)
        electrostatic = self._analyze_electrostatic_properties(structure)
        properties.update(electrostatic)
        
        # Thermal stability prediction
        thermal_stability = self._predict_thermal_stability(structure)
        properties['predicted_tm'] = thermal_stability
        
        # Aggregation propensity
        aggregation_score = self._predict_aggregation_propensity(structure)
        properties['aggregation_propensity'] = aggregation_score
        
        return properties
    
    def _predict_binding_sites(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Predict potential binding sites."""
        binding_predictions = {}
        
        # Cavity detection (simplified)
        cavities = self._detect_cavities(structure)
        binding_predictions['cavities'] = cavities
        
        # Surface accessibility-based binding sites
        surface_sites = self._identify_surface_binding_sites(structure)
        binding_predictions['surface_sites'] = surface_sites
        
        # Geometric binding site prediction
        geometric_sites = self._predict_geometric_binding_sites(structure)
        binding_predictions['geometric_sites'] = geometric_sites
        
        # Druggability assessment
        druggability_scores = []
        for cavity in cavities:
            score = self._assess_cavity_druggability(cavity)
            druggability_scores.append(score)
        
        binding_predictions['druggability_scores'] = druggability_scores
        binding_predictions['max_druggability'] = max(druggability_scores) if druggability_scores else 0.0
        
        return binding_predictions
    
    def _analyze_stability(self, structure: ProteinStructure) -> Dict[str, float]:
        """Analyze protein stability factors."""
        stability_metrics = {}
        
        # Geometric stability indicators
        coords = structure.coordinates
        
        # Local packing quality
        packing_scores = []
        for i in range(structure.num_residues):
            # Count neighbors within shell
            distances = torch.norm(coords - coords[i], dim=1)
            neighbors = torch.sum((distances > 0) & (distances < 12.0)).item()
            packing_scores.append(neighbors)
        
        stability_metrics['avg_local_packing'] = np.mean(packing_scores)
        stability_metrics['min_local_packing'] = np.min(packing_scores)
        
        # Hydrogen bonding potential (simplified)
        hbond_potential = self._estimate_hbond_potential(structure)
        stability_metrics['hbond_potential'] = hbond_potential
        
        # Core packing analysis
        core_packing = self._analyze_core_packing(structure)
        stability_metrics.update(core_packing)
        
        # Loop stability
        loop_stability = self._analyze_loop_stability(structure)
        stability_metrics['loop_stability'] = loop_stability
        
        # Overall stability prediction
        stability_features = [
            stability_metrics['avg_local_packing'] / 20.0,  # Normalize
            stability_metrics['hbond_potential'],
            stability_metrics.get('core_packing_quality', 0.5),
            stability_metrics['loop_stability']
        ]
        
        stability_metrics['predicted_stability'] = np.mean(stability_features)
        
        return stability_metrics
    
    def _assess_designability(self, structure: ProteinStructure) -> float:
        """Assess how designable/foldable the structure is."""
        design_factors = []
        
        # Contact order (lower is more designable)
        contact_map = structure.compute_contact_map()
        contact_order = self._compute_contact_order(contact_map)
        design_factors.append(1.0 / (1.0 + contact_order))
        
        # Secondary structure regularity
        ss_assignment = structure.compute_secondary_structure_simple()
        regularity = self._compute_ss_regularity(ss_assignment)
        design_factors.append(regularity)
        
        # Compactness (optimal range)
        rg = structure.compute_radius_of_gyration()
        expected_rg = 2.2 * (structure.num_residues ** 0.38)
        compactness_score = 1.0 / (1.0 + abs(rg - expected_rg) / expected_rg)
        design_factors.append(compactness_score)
        
        # Local geometry quality
        geom_validation = structure.validate_geometry()
        geometry_score = 1.0 / (1.0 + geom_validation.get('avg_bond_deviation', 1.0))
        design_factors.append(geometry_score)
        
        return np.mean(design_factors)
    
    def _analyze_drug_likeness(self, structure: ProteinStructure) -> Dict[str, float]:
        """Analyze drug-like properties of the protein."""
        drug_props = {}
        
        # Binding site analysis
        binding_sites = self._predict_binding_sites(structure)
        num_druggable_sites = sum(1 for score in binding_sites['druggability_scores'] if score > 0.5)
        drug_props['num_druggable_sites'] = num_druggable_sites
        
        # Surface properties relevant to drug binding
        surface_area = self._estimate_surface_area(structure)
        drug_props['surface_area'] = surface_area
        
        # Cavity volume analysis
        total_cavity_volume = sum(cavity['volume'] for cavity in binding_sites['cavities'])
        drug_props['total_cavity_volume'] = total_cavity_volume
        
        # Accessibility analysis
        accessible_surface = self._compute_accessible_surface_area(structure)
        drug_props['accessible_surface_area'] = accessible_surface
        
        # Overall druggability score
        druggability_factors = [
            min(1.0, num_druggable_sites / 3.0),  # Normalize by expected number
            min(1.0, total_cavity_volume / 1000.0),  # Normalize by typical values
            binding_sites['max_druggability']
        ]
        
        drug_props['overall_druggability'] = np.mean(druggability_factors)
        
        return drug_props
    
    def _compare_to_database(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Compare structure to reference database."""
        comparisons = {}
        
        # This would typically compare against PDB structures
        # For now, we'll create mock comparison data
        
        comparisons['similar_structures'] = []
        comparisons['fold_classification'] = "Unknown"
        comparisons['structural_neighbors'] = []
        comparisons['evolutionary_distance'] = 0.0
        
        # Mock similarity search
        rg = structure.compute_radius_of_gyration()
        ss_content = structure.compute_secondary_structure_simple()
        helix_frac = ss_content.count('H') / len(ss_content)
        
        # Create mock similar structures
        for i in range(3):
            similar_struct = {
                'pdb_id': f'MOCK{i+1}',
                'rmsd': np.random.uniform(2.0, 8.0),
                'sequence_identity': np.random.uniform(0.1, 0.4),
                'structural_similarity': np.random.uniform(0.3, 0.8),
                'fold_family': f"Mock_Family_{i+1}"
            }
            comparisons['similar_structures'].append(similar_struct)
        
        return comparisons
    
    def _generate_visualization_data(self, structure: ProteinStructure) -> Dict[str, Any]:
        """Generate data for structure visualization."""
        viz_data = {}
        
        # Contact map
        contact_map = structure.compute_contact_map()
        viz_data['contact_map'] = contact_map.numpy()
        
        # Distance matrix
        dist_matrix = structure.compute_distance_matrix()
        viz_data['distance_matrix'] = dist_matrix.numpy()
        
        # Secondary structure coloring
        ss_assignment = structure.compute_secondary_structure_simple()
        ss_colors = {'H': 'red', 'E': 'blue', 'C': 'green'}
        viz_data['ss_colors'] = [ss_colors[ss] for ss in ss_assignment]
        
        # B-factor-like flexibility scores
        flexibility_scores = self._compute_residue_flexibility(structure)
        viz_data['flexibility_scores'] = flexibility_scores
        
        # Hydrophobicity coloring
        hydrophobicity_scores = self._compute_residue_hydrophobicity(structure)
        viz_data['hydrophobicity_scores'] = hydrophobicity_scores
        
        return viz_data
    
    # Helper methods for specific analyses
    
    def _analyze_hydrophobicity(self, structure: ProteinStructure) -> Dict[str, float]:
        """Analyze hydrophobicity distribution."""
        # Simplified hydrophobicity analysis
        coords = structure.coordinates
        n = structure.num_residues
        
        # Define core and surface regions
        center = torch.mean(coords, dim=0)
        distances_from_center = torch.norm(coords - center, dim=1)
        median_distance = torch.median(distances_from_center)
        
        core_residues = distances_from_center < median_distance
        surface_residues = distances_from_center >= median_distance
        
        # Mock hydrophobicity scores (would use actual amino acid properties)
        hydrophobicity_scores = torch.rand(n)  # Random scores for demonstration
        
        core_hydrophobicity = torch.mean(hydrophobicity_scores[core_residues]).item()
        surface_hydrophobicity = torch.mean(hydrophobicity_scores[surface_residues]).item()
        
        return {
            'core_hydrophobicity': core_hydrophobicity,
            'surface_hydrophobicity': surface_hydrophobicity,
            'hydrophobic_moment': abs(core_hydrophobicity - surface_hydrophobicity)
        }
    
    def _analyze_surface_accessibility(self, structure: ProteinStructure) -> Dict[str, float]:
        """Analyze surface accessibility."""
        coords = structure.coordinates
        n = structure.num_residues
        
        # Simplified surface accessibility calculation
        accessibility_scores = []
        
        for i in range(n):
            # Count neighbors within accessibility shell
            distances = torch.norm(coords - coords[i], dim=1)
            neighbors = torch.sum((distances > 0) & (distances < 10.0)).item()
            
            # Higher neighbor count = lower accessibility
            accessibility = max(0, 1.0 - neighbors / 20.0)
            accessibility_scores.append(accessibility)
        
        return {
            'avg_accessibility': np.mean(accessibility_scores),
            'max_accessibility': np.max(accessibility_scores),
            'accessible_residue_fraction': sum(1 for score in accessibility_scores if score > 0.5) / n
        }
    
    def _predict_flexibility(self, structure: ProteinStructure) -> Dict[str, float]:
        """Predict structural flexibility."""
        coords = structure.coordinates
        n = structure.num_residues
        
        flexibility_scores = []
        
        for i in range(n):
            # Local density-based flexibility prediction
            local_coords = coords[max(0, i-2):min(n, i+3)]
            if len(local_coords) > 1:
                local_variance = torch.var(local_coords, dim=0)
                flexibility = torch.mean(local_variance).item()
            else:
                flexibility = 0.5
            
            flexibility_scores.append(flexibility)
        
        return {
            'avg_flexibility': np.mean(flexibility_scores),
            'flexibility_variance': np.var(flexibility_scores),
            'rigid_fraction': sum(1 for score in flexibility_scores if score < 0.3) / n
        }
    
    def _analyze_electrostatic_properties(self, structure: ProteinStructure) -> Dict[str, float]:
        """Analyze electrostatic properties (simplified)."""
        # This would typically use actual charge calculations
        # For now, return mock values
        return {
            'net_charge': np.random.uniform(-5, 5),
            'charge_distribution': np.random.uniform(0, 1),
            'dipole_moment': np.random.uniform(0, 100)
        }
    
    def _predict_thermal_stability(self, structure: ProteinStructure) -> float:
        """Predict thermal stability (Tm)."""
        # Simplified stability prediction based on structural features
        rg = structure.compute_radius_of_gyration()
        geom_quality = structure.validate_geometry()
        
        # Mock calculation
        base_tm = 50.0  # Base temperature
        compactness_bonus = max(0, 20 - rg)  # More compact = more stable
        geometry_bonus = 30 * (1.0 - geom_quality.get('avg_bond_deviation', 1.0))
        
        predicted_tm = base_tm + compactness_bonus + geometry_bonus
        return max(25.0, min(100.0, predicted_tm))  # Clamp to reasonable range
    
    def _predict_aggregation_propensity(self, structure: ProteinStructure) -> float:
        """Predict aggregation propensity."""
        # Simplified aggregation prediction
        coords = structure.coordinates
        
        # Calculate surface hydrophobicity (mock)
        surface_hydrophobic_patches = 0
        for i in range(0, structure.num_residues, 5):  # Sample every 5th residue
            local_region = coords[i:min(i+5, structure.num_residues)]
            if len(local_region) > 2:
                local_spread = torch.std(local_region, dim=0).mean()
                if local_spread < 2.0:  # Compact hydrophobic region
                    surface_hydrophobic_patches += 1
        
        aggregation_score = surface_hydrophobic_patches / (structure.num_residues / 5)
        return min(1.0, aggregation_score)
    
    def _detect_cavities(self, structure: ProteinStructure) -> List[Dict[str, Any]]:
        """Detect structural cavities."""
        # Simplified cavity detection
        coords = structure.coordinates
        cavities = []
        
        # Grid-based cavity detection (very simplified)
        center = torch.mean(coords, dim=0)
        
        # Check potential cavity centers
        for i in range(structure.num_residues):
            pos = coords[i]
            
            # Check if this position is in a potential cavity
            distances = torch.norm(coords - pos, dim=1)
            nearby_residues = torch.sum(distances < 8.0).item()
            
            if 3 <= nearby_residues <= 8:  # Potential cavity
                cavity = {
                    'center': pos.tolist(),
                    'volume': np.random.uniform(100, 500),  # Mock volume
                    'depth': np.random.uniform(5, 15),
                    'nearby_residues': nearby_residues,
                    'accessibility': np.random.uniform(0.3, 0.9)
                }
                cavities.append(cavity)
        
        return cavities[:5]  # Return top 5 cavities
    
    def _identify_surface_binding_sites(self, structure: ProteinStructure) -> List[Dict[str, Any]]:
        """Identify surface binding sites."""
        # Simplified surface binding site identification
        coords = structure.coordinates
        sites = []
        
        # Find surface residues with good accessibility
        center = torch.mean(coords, dim=0)
        distances_from_center = torch.norm(coords - center, dim=1)
        surface_threshold = torch.quantile(distances_from_center, 0.7)
        
        surface_indices = torch.where(distances_from_center > surface_threshold)[0]
        
        for idx in surface_indices[:10]:  # Limit to 10 sites
            site = {
                'residue_index': idx.item(),
                'position': coords[idx].tolist(),
                'accessibility': np.random.uniform(0.5, 1.0),
                'binding_potential': np.random.uniform(0.3, 0.8)
            }
            sites.append(site)
        
        return sites
    
    def _predict_geometric_binding_sites(self, structure: ProteinStructure) -> List[Dict[str, Any]]:
        """Predict binding sites based on geometric features."""
        # Mock geometric binding site prediction
        sites = []
        
        for i in range(min(3, structure.num_residues // 20)):
            site = {
                'type': 'geometric',
                'confidence': np.random.uniform(0.4, 0.9),
                'size': np.random.uniform(200, 800),
                'shape_complementarity': np.random.uniform(0.3, 0.7)
            }
            sites.append(site)
        
        return sites
    
    def _assess_cavity_druggability(self, cavity: Dict[str, Any]) -> float:
        """Assess druggability of a cavity."""
        # Simplified druggability scoring
        volume_score = min(1.0, cavity['volume'] / 400.0)  # Optimal around 400 ųÄ
        depth_score = min(1.0, cavity['depth'] / 10.0)      # Deeper is better
        accessibility_score = cavity['accessibility']
        
        druggability = (volume_score + depth_score + accessibility_score) / 3.0
        return druggability
    
    def _estimate_convex_hull_volume(self, coords: torch.Tensor) -> float:
        """Estimate convex hull volume."""
        # Simplified volume estimation
        ranges = torch.max(coords, dim=0)[0] - torch.min(coords, dim=0)[0]
        volume = torch.prod(ranges).item()
        return volume * 0.5  # Approximate correction factor
    
    def _estimate_hbond_potential(self, structure: ProteinStructure) -> float:
        """Estimate hydrogen bonding potential."""
        # Mock hydrogen bond potential
        coords = structure.coordinates
        
        # Count potential hydrogen bond pairs (distance-based)
        dist_matrix = structure.compute_distance_matrix()
        hbond_distances = (dist_matrix > 2.5) & (dist_matrix < 3.5)
        
        potential_hbonds = torch.sum(hbond_distances).item() / 2  # Avoid double counting
        normalized_potential = potential_hbonds / max(structure.num_residues, 1)
        
        return min(1.0, normalized_potential / 2.0)  # Normalize to [0,1]
    
    def _analyze_core_packing(self, structure: ProteinStructure) -> Dict[str, float]:
        """Analyze hydrophobic core packing."""
        coords = structure.coordinates
        center = torch.mean(coords, dim=0)
        distances_from_center = torch.norm(coords - center, dim=1)
        
        # Define core as innermost 30% of residues
        core_threshold = torch.quantile(distances_from_center, 0.3)
        core_indices = distances_from_center <= core_threshold
        
        if torch.sum(core_indices) < 3:
            return {'core_packing_quality': 0.5}
        
        core_coords = coords[core_indices]
        
        # Analyze packing density in core
        core_distances = torch.cdist(core_coords, core_coords)
        core_contacts = torch.sum((core_distances > 0) & (core_distances < 6.0)).item()
        
        # Normalize by number of core residues
        core_size = torch.sum(core_indices).item()
        packing_density = core_contacts / max(core_size * (core_size - 1), 1)
        
        return {'core_packing_quality': min(1.0, packing_density)}
    
    def _analyze_loop_stability(self, structure: ProteinStructure) -> float:
        """Analyze loop region stability."""
        ss_assignment = structure.compute_secondary_structure_simple()
        coords = structure.coordinates
        
        # Identify loop regions
        loop_indices = [i for i, ss in enumerate(ss_assignment) if ss == 'C']
        
        if not loop_indices:
            return 1.0  # No loops = perfectly stable
        
        # Analyze loop flexibility/stability
        stability_scores = []
        
        for idx in loop_indices:
            if idx > 0 and idx < structure.num_residues - 1:
                # Local structure analysis
                local_coords = coords[max(0, idx-2):min(structure.num_residues, idx+3)]
                if len(local_coords) > 2:
                    local_var = torch.var(local_coords, dim=0).mean()
                    stability = 1.0 / (1.0 + local_var)
                    stability_scores.append(stability.item())
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _compute_contact_order(self, contact_map: torch.Tensor) -> float:
        """Compute relative contact order."""
        n = contact_map.size(0)
        total_contacts = 0
        contact_order_sum = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if contact_map[i, j] > 0:
                    total_contacts += 1
                    contact_order_sum += abs(j - i)
        
        if total_contacts == 0:
            return 0.0
        
        return contact_order_sum / (total_contacts * n)
    
    def _compute_ss_regularity(self, ss_assignment: List[str]) -> float:
        """Compute secondary structure regularity."""
        # Count transitions between different SS types
        transitions = 0
        for i in range(len(ss_assignment) - 1):
            if ss_assignment[i] != ss_assignment[i + 1]:
                transitions += 1
        
        # Lower transition rate = higher regularity
        transition_rate = transitions / max(len(ss_assignment) - 1, 1)
        regularity = 1.0 / (1.0 + transition_rate * 2)
        
        return regularity
    
    def _estimate_surface_area(self, structure: ProteinStructure) -> float:
        """Estimate surface area."""
        # Simplified surface area estimation
        rg = structure.compute_radius_of_gyration()
        # Approximate as sphere surface area with correction factor
        surface_area = 4 * np.pi * (rg ** 2) * 0.7  # Correction for protein shape
        return surface_area
    
    def _compute_accessible_surface_area(self, structure: ProteinStructure) -> float:
        """Compute accessible surface area."""
        # Simplified ASA calculation
        surface_area = self._estimate_surface_area(structure)
        # Assume 60-80% of surface is accessible
        accessible_fraction = np.random.uniform(0.6, 0.8)
        return surface_area * accessible_fraction
    
    def _compute_residue_flexibility(self, structure: ProteinStructure) -> List[float]:
        """Compute per-residue flexibility scores."""
        coords = structure.coordinates
        flexibility_scores = []
        
        for i in range(structure.num_residues):
            # Local coordinate variance as flexibility measure
            window_start = max(0, i - 2)
            window_end = min(structure.num_residues, i + 3)
            local_coords = coords[window_start:window_end]
            
            if len(local_coords) > 1:
                variance = torch.var(local_coords, dim=0).mean()
                flexibility = variance.item()
            else:
                flexibility = 0.5
            
            flexibility_scores.append(flexibility)
        
        return flexibility_scores
    
    def _compute_residue_hydrophobicity(self, structure: ProteinStructure) -> List[float]:
        """Compute per-residue hydrophobicity scores."""
        # Mock hydrophobicity scores
        # In practice, would use actual amino acid hydrophobicity scales
        return [np.random.uniform(0, 1) for _ in range(structure.num_residues)]