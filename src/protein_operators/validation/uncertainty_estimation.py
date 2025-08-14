"""
Advanced uncertainty estimation methods for neural operator protein predictions.

This module provides multiple approaches to uncertainty quantification,
including ensemble methods, Bayesian approaches, and conformal prediction.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from abc import ABC, abstractmethod
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional
    DataLoader = object


class BaseUncertaintyEstimator(ABC):
    """Base class for uncertainty estimation methods."""
    
    def __init__(self, n_samples: int = 100, confidence_level: float = 0.95):
        self.n_samples = n_samples
        self.confidence_level = confidence_level
    
    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction uncertainty.
        
        Args:
            model: Neural network model
            inputs: Input data
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        pass


class EnsembleUncertainty(BaseUncertaintyEstimator):
    """
    Ensemble-based uncertainty estimation.
    
    Uses multiple independently trained models to estimate
    epistemic uncertainty through prediction variance.
    """
    
    def __init__(
        self,
        models: List[nn.Module] = None,
        n_samples: int = 100,
        confidence_level: float = 0.95
    ):
        super().__init__(n_samples, confidence_level)
        self.models = models or []
        
    def add_model(self, model: nn.Module):
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using ensemble variance."""
        if not self.models:
            # If no ensemble models provided, use the given model with different seeds
            return self._bootstrap_estimate(model, inputs)
        
        predictions = []
        
        # Collect predictions from all ensemble members
        with torch.no_grad():
            for ensemble_model in self.models:
                ensemble_model.eval()
                pred = ensemble_model(inputs)
                predictions.append(pred)
        
        # Stack predictions
        all_predictions = torch.stack(predictions, dim=0)  # [n_models, batch, ...]
        
        # Mean prediction
        mean_prediction = torch.mean(all_predictions, dim=0)
        
        # Predictive uncertainty (variance across ensemble)
        prediction_variance = torch.var(all_predictions, dim=0)
        uncertainty = torch.sqrt(prediction_variance)
        
        return mean_prediction, uncertainty
    
    def _bootstrap_estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bootstrap estimation when no ensemble is available."""
        # Create multiple copies with different dropout patterns
        predictions = []
        
        model.train()  # Enable dropout
        
        with torch.no_grad():
            for _ in range(min(self.n_samples, 20)):  # Limit for efficiency
                pred = model(inputs)
                predictions.append(pred)
        
        model.eval()
        
        # Stack and compute statistics
        all_predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(all_predictions, dim=0)
        uncertainty = torch.std(all_predictions, dim=0)
        
        return mean_prediction, uncertainty


class DropoutUncertainty(BaseUncertaintyEstimator):
    """
    Monte Carlo Dropout uncertainty estimation.
    
    Uses dropout at inference time to estimate model uncertainty
    through stochastic forward passes.
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        confidence_level: float = 0.95,
        dropout_rate: float = 0.1
    ):
        super().__init__(n_samples, confidence_level)
        self.dropout_rate = dropout_rate
    
    def enable_dropout(self, model: nn.Module):
        """Enable dropout layers for inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using Monte Carlo dropout."""
        # Store original training state
        original_training = model.training
        
        # Enable dropout layers
        model.eval()
        self.enable_dropout(model)
        
        predictions = []
        
        # Multiple stochastic forward passes
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = model(inputs)
                predictions.append(pred)
        
        # Restore original training state
        model.train(original_training)
        
        # Compute mean and uncertainty
        all_predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(all_predictions, dim=0)
        uncertainty = torch.std(all_predictions, dim=0)
        
        return mean_prediction, uncertainty


class BayesianUncertainty(BaseUncertaintyEstimator):
    """
    Bayesian uncertainty estimation using variational inference.
    
    Estimates both epistemic and aleatoric uncertainty through
    Bayesian neural network approximation.
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        confidence_level: float = 0.95,
        prior_std: float = 1.0
    ):
        super().__init__(n_samples, confidence_level)
        self.prior_std = prior_std
    
    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate Bayesian uncertainty."""
        # Check if model has Bayesian layers
        has_bayesian_layers = any(
            hasattr(module, 'weight_mu') for module in model.modules()
        )
        
        if not has_bayesian_layers:
            # Convert to approximate Bayesian model
            return self._approximate_bayesian_inference(model, inputs)
        
        # Use existing Bayesian layers
        return self._bayesian_inference(model, inputs)
    
    def _approximate_bayesian_inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Approximate Bayesian inference for standard neural networks."""
        # Add Gaussian noise to weights as approximation
        predictions = []
        
        original_state = copy.deepcopy(model.state_dict())
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Add noise to parameters
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * self.prior_std * 0.01
                        param.data += noise
                
                # Forward pass
                pred = model(inputs)
                predictions.append(pred)
                
                # Restore original parameters
                model.load_state_dict(original_state)
        
        # Compute statistics
        all_predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(all_predictions, dim=0)
        uncertainty = torch.std(all_predictions, dim=0)
        
        return mean_prediction, uncertainty
    
    def _bayesian_inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """True Bayesian inference for Bayesian neural networks."""
        predictions = []
        
        # Sample from posterior
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Sample weights from posterior (would need proper implementation)
                pred = model(inputs)
                predictions.append(pred)
        
        all_predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(all_predictions, dim=0)
        uncertainty = torch.std(all_predictions, dim=0)
        
        return mean_prediction, uncertainty


class ConformalUncertainty(BaseUncertaintyEstimator):
    """
    Conformal prediction for uncertainty quantification.
    
    Provides distribution-free uncertainty quantification
    with finite-sample coverage guarantees.
    """
    
    def __init__(
        self,
        calibration_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 100,
        confidence_level: float = 0.95
    ):
        super().__init__(n_samples, confidence_level)
        self.calibration_data = calibration_data
        self.quantile = None
        
        if calibration_data is not None:
            self._calibrate(*calibration_data)
    
    def _calibrate(
        self,
        calibration_inputs: torch.Tensor,
        calibration_targets: torch.Tensor,
        model: nn.Module = None
    ):
        """Calibrate conformal predictor."""
        if model is None:
            raise ValueError("Model required for calibration")
        
        model.eval()
        with torch.no_grad():
            calibration_predictions = model(calibration_inputs)
        
        # Compute nonconformity scores (residuals)
        residuals = torch.abs(calibration_predictions - calibration_targets)
        
        # Compute quantile for desired coverage
        alpha = 1 - self.confidence_level
        n_cal = len(residuals)
        q_level = (1 - alpha) * (1 + 1/n_cal)
        
        self.quantile = torch.quantile(residuals.flatten(), q_level)
    
    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate conformal prediction intervals."""
        if self.quantile is None:
            raise ValueError("Must calibrate before prediction")
        
        model.eval()
        with torch.no_grad():
            predictions = model(inputs)
        
        # Prediction intervals
        uncertainty = torch.full_like(predictions, self.quantile)
        
        return predictions, uncertainty


class CalibrationAnalyzer:
    """
    Analyzer for uncertainty calibration assessment.
    
    Provides tools to evaluate how well uncertainty estimates
    match actual prediction errors.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def compute_calibration_curve(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reliability diagram data.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            
        Returns:
            Tuple of (confidence_bins, accuracy_bins)
        """
        # Convert uncertainties to confidence scores
        max_uncertainty = torch.max(uncertainties)
        confidence_scores = 1 - (uncertainties / max_uncertainty)
        
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = (confidence_scores >= bin_lower) & (confidence_scores < bin_upper)
            
            if torch.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = torch.mean(confidence_scores[in_bin]).item()
                
                # Compute accuracy in bin
                errors = torch.abs(predictions[in_bin] - targets[in_bin])
                threshold = torch.quantile(torch.abs(predictions - targets), 0.5)
                accuracy = torch.mean((errors <= threshold).float()).item()
                
                bin_confidences.append(avg_confidence)
                bin_accuracies.append(accuracy)
        
        return np.array(bin_confidences), np.array(bin_accuracies)
    
    def compute_expected_calibration_error(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_confidences, bin_accuracies = self.compute_calibration_curve(
            predictions, uncertainties, targets
        )
        
        if len(bin_confidences) == 0:
            return 0.0
        
        # Weighted average of calibration errors
        weights = np.ones(len(bin_confidences)) / len(bin_confidences)
        ece = np.sum(weights * np.abs(bin_confidences - bin_accuracies))
        
        return ece
    
    def compute_maximum_calibration_error(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        bin_confidences, bin_accuracies = self.compute_calibration_curve(
            predictions, uncertainties, targets
        )
        
        if len(bin_confidences) == 0:
            return 0.0
        
        mce = np.max(np.abs(bin_confidences - bin_accuracies))
        
        return mce
    
    def assess_coverage(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Assess coverage of prediction intervals."""
        # Prediction intervals
        z_score = 1.96  # For 95% confidence
        lower_bound = predictions - z_score * uncertainties
        upper_bound = predictions + z_score * uncertainties
        
        # Empirical coverage
        coverage = torch.mean(
            (targets >= lower_bound) & (targets <= upper_bound)
        ).item()
        
        # Average interval width
        interval_width = torch.mean(upper_bound - lower_bound).item()
        
        # Coverage difference from expected
        coverage_error = abs(coverage - confidence_level)
        
        return {
            'coverage': coverage,
            'expected_coverage': confidence_level,
            'coverage_error': coverage_error,
            'interval_width': interval_width
        }


class UncertaintyEstimator:
    """
    Unified uncertainty estimation interface.
    
    Provides a single interface for multiple uncertainty
    estimation methods with automatic method selection.
    """
    
    def __init__(
        self,
        method: str = 'ensemble',
        n_samples: int = 100,
        confidence_level: float = 0.95,
        **kwargs
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            method: Uncertainty estimation method
            n_samples: Number of samples for stochastic methods
            confidence_level: Confidence level for intervals
            **kwargs: Additional method-specific arguments
        """
        self.method = method
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        
        # Initialize method-specific estimator
        if method == 'ensemble':
            self.estimator = EnsembleUncertainty(
                n_samples=n_samples,
                confidence_level=confidence_level,
                **kwargs
            )
        elif method == 'dropout':
            self.estimator = DropoutUncertainty(
                n_samples=n_samples,
                confidence_level=confidence_level,
                **kwargs
            )
        elif method == 'bayesian':
            self.estimator = BayesianUncertainty(
                n_samples=n_samples,
                confidence_level=confidence_level,
                **kwargs
            )
        elif method == 'conformal':
            self.estimator = ConformalUncertainty(
                n_samples=n_samples,
                confidence_level=confidence_level,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
        
        # Calibration analyzer
        self.calibration_analyzer = CalibrationAnalyzer()
    
    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate prediction uncertainty."""
        return self.estimator.estimate(model, inputs)
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Tuple[torch.Tensor, torch.Tensor]
    ):
        """Calibrate uncertainty estimator if applicable."""
        if hasattr(self.estimator, '_calibrate'):
            self.estimator._calibrate(*calibration_data, model)
    
    def analyze_calibration(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """Comprehensive calibration analysis."""
        # Calibration curve
        bin_confidences, bin_accuracies = self.calibration_analyzer.compute_calibration_curve(
            predictions, uncertainties, targets
        )
        
        # Calibration errors
        ece = self.calibration_analyzer.compute_expected_calibration_error(
            predictions, uncertainties, targets
        )
        mce = self.calibration_analyzer.compute_maximum_calibration_error(
            predictions, uncertainties, targets
        )
        
        # Coverage assessment
        coverage_metrics = self.calibration_analyzer.assess_coverage(
            predictions, uncertainties, targets, self.confidence_level
        )
        
        return {
            'calibration_curve': (bin_confidences, bin_accuracies),
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'coverage_metrics': coverage_metrics
        }