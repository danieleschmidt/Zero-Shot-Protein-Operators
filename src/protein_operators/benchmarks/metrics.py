"""
Comprehensive metrics for protein structure evaluation.

This module provides research-grade metrics for evaluating protein structure
prediction quality, including structural, physical, and biochemical measures.
"""

import os
import sys
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
except ImportError:
    import mock_torch as torch


class ProteinStructureMetrics:
    """
    Structural metrics for protein conformation evaluation.
    
    Implements standard metrics used in CASP, CAMEO, and other
    protein structure prediction assessments.
    """
    
    def __init__(self):
        self.ca_distance_threshold = 3.8  # Angstroms for GDT calculation
        
    def rmsd(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        align: bool = True
    ) -> float:
        """
        Compute Root Mean Square Deviation (RMSD).
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            align: Whether to perform optimal alignment
            
        Returns:
            RMSD value in Angstroms
        """
        if predicted.shape != target.shape:
            raise ValueError("Predicted and target shapes must match")
        
        if align:
            predicted_aligned = self._kabsch_align(predicted, target)
        else:
            predicted_aligned = predicted
        
        squared_diffs = torch.sum((predicted_aligned - target) ** 2, dim=1)
        rmsd_value = torch.sqrt(torch.mean(squared_diffs))
        
        return rmsd_value.item()
    
    def gdt_ts(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        thresholds: List[float] = [1.0, 2.0, 4.0, 8.0]
    ) -> float:
        """
        Compute Global Distance Test - Total Score (GDT-TS).
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            thresholds: Distance thresholds in Angstroms
            
        Returns:
            GDT-TS score (0-100)
        """
        if predicted.shape != target.shape:
            raise ValueError("Predicted and target shapes must match")
        
        # Align structures
        predicted_aligned = self._kabsch_align(predicted, target)
        
        # Compute distances
        distances = torch.norm(predicted_aligned - target, dim=1)
        
        # Calculate percentage of residues within each threshold
        scores = []
        for threshold in thresholds:
            within_threshold = (distances <= threshold).float()
            percentage = torch.mean(within_threshold) * 100
            scores.append(percentage.item())
        
        # GDT-TS is the average of the four percentages
        gdt_ts_score = np.mean(scores)
        
        return gdt_ts_score
    
    def gdt_ha(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> float:
        """
        Compute Global Distance Test - High Accuracy (GDT-HA).
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            thresholds: Distance thresholds in Angstroms
            
        Returns:
            GDT-HA score (0-100)
        """
        return self.gdt_ts(predicted, target, thresholds)
    
    def tm_score(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute Template Modeling Score (TM-score).
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            
        Returns:
            TM-score (0-1, higher is better)
        """
        if predicted.shape != target.shape:
            raise ValueError("Predicted and target shapes must match")
        
        n_residues = predicted.shape[0]
        
        # Align structures
        predicted_aligned = self._kabsch_align(predicted, target)
        
        # Compute distances
        distances = torch.norm(predicted_aligned - target, dim=1)
        
        # TM-score normalization factor
        if n_residues <= 21:
            d0 = 0.5
        else:
            d0 = 1.24 * ((n_residues - 15) ** (1/3)) - 1.8
        
        # Compute TM-score
        weights = 1.0 / (1.0 + (distances / d0) ** 2)
        tm_score = torch.sum(weights) / n_residues
        
        return tm_score.item()
    
    def ldt(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        inclusion_radius: float = 15.0
    ) -> float:
        """
        Compute Local Distance Difference Test (lDDT).
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            inclusion_radius: Radius for local environment
            
        Returns:
            lDDT score (0-100)
        """
        if predicted.shape != target.shape:
            raise ValueError("Predicted and target shapes must match")
        
        n_residues = predicted.shape[0]
        
        # Compute all pairwise distances
        target_distances = torch.cdist(target, target)
        predicted_distances = torch.cdist(predicted, predicted)
        
        # Distance difference
        distance_diff = torch.abs(predicted_distances - target_distances)
        
        # Define thresholds
        thresholds = [0.5, 1.0, 2.0, 4.0]
        
        total_score = 0
        total_pairs = 0
        
        for i in range(n_residues):
            # Find residues within inclusion radius
            local_mask = (target_distances[i] <= inclusion_radius) & (torch.arange(n_residues) != i)
            
            if torch.sum(local_mask) == 0:
                continue
            
            # Calculate score for this residue
            local_diff = distance_diff[i][local_mask]
            
            for threshold in thresholds:
                within_threshold = (local_diff <= threshold).float()
                total_score += torch.sum(within_threshold).item()
            
            total_pairs += torch.sum(local_mask).item() * len(thresholds)
        
        if total_pairs == 0:
            return 0.0
        
        ldt_score = (total_score / total_pairs) * 100
        
        return ldt_score
    
    def _kabsch_align(
        self,
        mobile: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform Kabsch alignment to minimize RMSD.
        
        Args:
            mobile: Mobile structure to align [N, 3]
            target: Target structure [N, 3]
            
        Returns:
            Aligned mobile structure
        """
        # Center structures
        mobile_centered = mobile - torch.mean(mobile, dim=0)
        target_centered = target - torch.mean(target, dim=0)
        
        # Compute covariance matrix
        H = torch.matmul(mobile_centered.T, target_centered)
        
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        
        # Compute rotation matrix
        R = torch.matmul(Vt.T, U.T)
        
        # Ensure proper rotation (det(R) = 1)
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = torch.matmul(Vt.T, U.T)
        
        # Apply alignment
        mobile_aligned = torch.matmul(mobile_centered, R.T) + torch.mean(target, dim=0)
        
        return mobile_aligned


class PhysicsMetrics:
    """
    Physics-based metrics for protein structure validation.
    
    Evaluates geometric and energetic properties of protein structures.
    """
    
    def __init__(self):
        self.ideal_bond_length = 1.53  # Angstroms (C-C bond)
        self.ideal_angle = 109.5 * math.pi / 180  # Tetrahedral angle
        self.vdw_radius = 2.0  # Van der Waals radius
        
    def bond_length_error(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        connectivity: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute bond length deviation from ideal values.
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            connectivity: Bond connectivity [M, 2] or None for sequential
            
        Returns:
            Mean absolute bond length error
        """
        if connectivity is None:
            # Assume sequential connectivity
            connectivity = torch.stack([
                torch.arange(predicted.shape[0] - 1),
                torch.arange(1, predicted.shape[0])
            ], dim=1)
        
        # Compute bond lengths for predicted structure
        bond_vectors_pred = predicted[connectivity[:, 1]] - predicted[connectivity[:, 0]]
        bond_lengths_pred = torch.norm(bond_vectors_pred, dim=1)
        
        # Compute bond lengths for target structure
        bond_vectors_target = target[connectivity[:, 1]] - target[connectivity[:, 0]]
        bond_lengths_target = torch.norm(bond_vectors_target, dim=1)
        
        # Compute error
        bond_error = torch.mean(torch.abs(bond_lengths_pred - bond_lengths_target))
        
        return bond_error.item()
    
    def angle_error(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        angle_triplets: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute bond angle deviation.
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            angle_triplets: Angle definitions [M, 3] or None for sequential
            
        Returns:
            Mean absolute angle error in degrees
        """
        if angle_triplets is None:
            # Assume sequential angle triplets
            if predicted.shape[0] < 3:
                return 0.0
            
            angle_triplets = torch.stack([
                torch.arange(predicted.shape[0] - 2),
                torch.arange(1, predicted.shape[0] - 1),
                torch.arange(2, predicted.shape[0])
            ], dim=1)
        
        def compute_angles(coords, triplets):
            """Compute angles for given coordinate set."""
            p1 = coords[triplets[:, 0]]
            p2 = coords[triplets[:, 1]]
            p3 = coords[triplets[:, 2]]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Normalize vectors
            v1_norm = F.normalize(v1, dim=1)
            v2_norm = F.normalize(v2, dim=1)
            
            # Compute angles
            cos_angles = torch.sum(v1_norm * v2_norm, dim=1)
            cos_angles = torch.clamp(cos_angles, -1, 1)  # Numerical stability
            angles = torch.acos(cos_angles)
            
            return angles
        
        # Compute angles for both structures
        angles_pred = compute_angles(predicted, angle_triplets)
        angles_target = compute_angles(target, angle_triplets)
        
        # Compute error in degrees
        angle_error = torch.mean(torch.abs(angles_pred - angles_target)) * 180 / math.pi
        
        return angle_error.item()
    
    def clash_score(
        self,
        predicted: torch.Tensor,
        min_distance: float = None
    ) -> float:
        """
        Compute atomic clash score.
        
        Args:
            predicted: Predicted coordinates [N, 3]
            min_distance: Minimum allowed distance
            
        Returns:
            Clash score (lower is better)
        """
        if min_distance is None:
            min_distance = self.vdw_radius
        
        # Compute pairwise distances
        distances = torch.cdist(predicted, predicted)
        
        # Mask diagonal (self-distances)
        mask = torch.eye(predicted.shape[0], dtype=torch.bool, device=predicted.device)
        distances = distances.masked_fill(mask, float('inf'))
        
        # Count clashes
        clashes = (distances < min_distance).float()
        clash_score = torch.sum(clashes) / 2  # Divide by 2 to avoid double counting
        
        return clash_score.item()
    
    def ramachandran_score(
        self,
        predicted: torch.Tensor,
        sequence: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute Ramachandran plot score.
        
        Args:
            predicted: Predicted coordinates [N, 3] (backbone atoms)
            sequence: Amino acid sequence [N] or None
            
        Returns:
            Ramachandran score (percentage in allowed regions)
        """
        if predicted.shape[0] < 4:
            return 100.0  # Too short to evaluate
        
        # Compute phi and psi angles
        phi_angles = []
        psi_angles = []
        
        for i in range(1, predicted.shape[0] - 1):
            # Phi angle: C(i-1) - N(i) - CA(i) - C(i)
            # Simplified: use consecutive points
            if i >= 1:
                phi = self._compute_dihedral(
                    predicted[i-1], predicted[i], predicted[i+1], 
                    predicted[min(i+2, predicted.shape[0]-1)]
                )
                phi_angles.append(phi)
            
            # Psi angle: N(i) - CA(i) - C(i) - N(i+1)
            if i < predicted.shape[0] - 2:
                psi = self._compute_dihedral(
                    predicted[i], predicted[i+1], predicted[i+2],
                    predicted[min(i+3, predicted.shape[0]-1)]
                )
                psi_angles.append(psi)
        
        if not phi_angles or not psi_angles:
            return 100.0
        
        # Check if angles are in allowed regions (simplified)
        allowed_count = 0
        total_count = min(len(phi_angles), len(psi_angles))
        
        for phi, psi in zip(phi_angles[:total_count], psi_angles[:total_count]):
            # Convert to degrees
            phi_deg = phi * 180 / math.pi
            psi_deg = psi * 180 / math.pi
            
            # Simplified allowed regions
            if self._is_in_allowed_region(phi_deg, psi_deg):
                allowed_count += 1
        
        score = (allowed_count / total_count) * 100 if total_count > 0 else 100.0
        
        return score
    
    def _compute_dihedral(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor
    ) -> float:
        """Compute dihedral angle between four points."""
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # Cross products
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        
        # Normalize
        n1_norm = F.normalize(n1, dim=0)
        n2_norm = F.normalize(n2, dim=0)
        
        # Compute angle
        cos_angle = torch.dot(n1_norm, n2_norm)
        cos_angle = torch.clamp(cos_angle, -1, 1)
        
        # Sign of angle
        sign = torch.sign(torch.dot(torch.cross(n1_norm, n2_norm), F.normalize(v2, dim=0)))
        
        angle = torch.acos(cos_angle) * sign
        
        return angle.item()
    
    def _is_in_allowed_region(self, phi: float, psi: float) -> bool:
        """Check if phi/psi angles are in allowed Ramachandran regions."""
        # Simplified allowed regions
        # Alpha-helix region
        if -80 <= phi <= -40 and -60 <= psi <= -20:
            return True
        
        # Beta-sheet region
        if -150 <= phi <= -100 and 100 <= psi <= 150:
            return True
        
        # Left-handed helix region
        if 40 <= phi <= 80 and 20 <= psi <= 60:
            return True
        
        return False


class BiochemicalMetrics:
    """
    Biochemical metrics for protein structure assessment.
    
    Evaluates biological plausibility and functional aspects.
    """
    
    def __init__(self):
        self.hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        self.hydrophilic_residues = {'SER', 'THR', 'ASN', 'GLN', 'LYS', 'ARG', 'HIS', 'ASP', 'GLU'}
        
    def secondary_structure_score(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        ss_pred: Optional[torch.Tensor] = None,
        ss_target: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute secondary structure similarity score.
        
        Args:
            predicted: Predicted coordinates [N, 3]
            target: Target coordinates [N, 3]
            ss_pred: Predicted secondary structure [N]
            ss_target: Target secondary structure [N]
            
        Returns:
            Secondary structure similarity score (0-100)
        """
        if ss_pred is None or ss_target is None:
            # Compute from coordinates (simplified)
            ss_pred = self._predict_secondary_structure(predicted)
            ss_target = self._predict_secondary_structure(target)
        
        # Compute agreement
        agreement = (ss_pred == ss_target).float()
        score = torch.mean(agreement) * 100
        
        return score.item()
    
    def hydrophobic_core_score(
        self,
        predicted: torch.Tensor,
        sequence: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate hydrophobic core formation.
        
        Args:
            predicted: Predicted coordinates [N, 3]
            sequence: Amino acid sequence
            
        Returns:
            Hydrophobic core score (0-100)
        """
        if sequence is None:
            # Cannot evaluate without sequence
            return 50.0  # Neutral score
        
        n_residues = len(sequence)
        if n_residues != predicted.shape[0]:
            return 0.0
        
        # Identify hydrophobic residues
        hydrophobic_indices = [
            i for i, aa in enumerate(sequence) 
            if aa in self.hydrophobic_residues
        ]
        
        if len(hydrophobic_indices) < 2:
            return 100.0  # Perfect score if no hydrophobic core expected
        
        # Compute center of mass
        center = torch.mean(predicted, dim=0)
        
        # Compute distances to center
        distances = torch.norm(predicted - center, dim=1)
        
        # Check if hydrophobic residues are closer to center
        hydrophobic_distances = distances[hydrophobic_indices]
        all_distances = distances
        
        # Compute percentile rank of hydrophobic residues
        scores = []
        for hyd_dist in hydrophobic_distances:
            percentile = (all_distances < hyd_dist).float().mean() * 100
            scores.append(100 - percentile.item())  # Lower distance = higher score
        
        core_score = np.mean(scores) if scores else 50.0
        
        return core_score
    
    def solvation_score(
        self,
        predicted: torch.Tensor,
        sequence: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate solvation appropriateness.
        
        Args:
            predicted: Predicted coordinates [N, 3]
            sequence: Amino acid sequence
            
        Returns:
            Solvation score (0-100)
        """
        if sequence is None:
            return 50.0  # Neutral score
        
        n_residues = len(sequence)
        if n_residues != predicted.shape[0]:
            return 0.0
        
        # Compute solvent accessible surface area (simplified)
        distances = torch.cdist(predicted, predicted)
        
        scores = []
        for i, aa in enumerate(sequence):
            # Number of neighbors within certain radius
            neighbors = (distances[i] < 8.0).sum().item() - 1  # Exclude self
            
            # Hydrophobic residues should have more neighbors (buried)
            # Hydrophilic residues should have fewer neighbors (exposed)
            if aa in self.hydrophobic_residues:
                # Higher neighbors = better for hydrophobic
                score = min(100, neighbors * 10)
            elif aa in self.hydrophilic_residues:
                # Lower neighbors = better for hydrophilic
                score = max(0, 100 - neighbors * 5)
            else:
                score = 50  # Neutral
            
            scores.append(score)
        
        solvation_score = np.mean(scores)
        
        return solvation_score
    
    def _predict_secondary_structure(
        self,
        coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple secondary structure prediction from coordinates.
        
        Args:
            coordinates: Coordinates [N, 3]
            
        Returns:
            Secondary structure assignment [N] (0=coil, 1=helix, 2=sheet)
        """
        n_residues = coordinates.shape[0]
        ss = torch.zeros(n_residues, dtype=torch.long)
        
        if n_residues < 4:
            return ss  # All coil
        
        # Compute local curvature and identify regular structures
        for i in range(2, n_residues - 2):
            # Look at local geometry
            p_prev = coordinates[i-2:i+1]
            p_next = coordinates[i:i+3]
            
            # Compute angles
            v1 = p_prev[1] - p_prev[0]
            v2 = p_prev[2] - p_prev[1]
            v3 = p_next[1] - p_next[0]
            v4 = p_next[2] - p_next[1]
            
            # Simplified helix detection (consistent angles)
            angle1 = torch.acos(torch.clamp(torch.dot(F.normalize(v1, dim=0), F.normalize(v2, dim=0)), -1, 1))
            angle2 = torch.acos(torch.clamp(torch.dot(F.normalize(v3, dim=0), F.normalize(v4, dim=0)), -1, 1))
            
            # If angles are similar and in helix range
            if abs(angle1 - angle2) < 0.5 and 1.5 < angle1 < 2.5:
                ss[i] = 1  # Helix
            
            # Simplified sheet detection (extended conformation)
            elif angle1 > 2.8 and angle2 > 2.8:
                ss[i] = 2  # Sheet
        
        return ss