"""
Comprehensive benchmarking suite for neural operator protein design research.

This module provides a rigorous framework for evaluating and comparing
different neural operator architectures on protein folding and design tasks.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional

from .metrics import ProteinStructureMetrics, PhysicsMetrics, BiochemicalMetrics
from .datasets import ProteinBenchmarkDatasets
from .statistical_tests import StatisticalAnalyzer, SignificanceTest


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results with comprehensive metrics.
    """
    model_name: str
    dataset_name: str
    task_name: str
    metrics: Dict[str, float]
    timing_info: Dict[str, float]
    memory_usage: Dict[str, float]
    model_params: Dict[str, Any]
    runtime_info: Dict[str, Any]
    error_analysis: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StatisticalTestResult:
    """
    Container for statistical test results.
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ProteinBenchmarkSuite:
    """
    Comprehensive benchmarking suite for protein neural operators.
    
    Features:
    - Multiple evaluation metrics (structural, physical, biochemical)
    - Statistical significance testing with multiple comparison correction
    - Performance profiling (time, memory, convergence)
    - Cross-validation and bootstrap analysis
    - Visualization and reporting capabilities
    - Reproducibility guarantees
    
    Examples:
        >>> suite = ProteinBenchmarkSuite(
        ...     datasets=['cath', 'synthetic'],
        ...     metrics=['rmsd', 'gdt_ts', 'physics_score'],
        ...     statistical_tests=['wilcoxon', 'bootstrap']
        ... )
        >>> results = suite.benchmark_models([fno_model, gno_model])
        >>> suite.generate_report(results, 'benchmark_report.html')
    """
    
    def __init__(
        self,
        datasets: List[str] = None,
        metrics: List[str] = None,
        statistical_tests: List[str] = None,
        output_dir: str = "benchmark_results",
        seed: int = 42,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        multiple_testing_correction: str = "bonferroni"
    ):
        """
        Initialize benchmarking suite.
        
        Args:
            datasets: List of dataset names to use
            metrics: List of metric names to compute
            statistical_tests: List of statistical tests to perform
            output_dir: Output directory for results
            seed: Random seed for reproducibility
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level
            multiple_testing_correction: Method for multiple testing correction
        """
        self.datasets = datasets or ['cath', 'synthetic', 'casp']
        self.metrics = metrics or ['rmsd', 'gdt_ts', 'tm_score', 'physics_score', 'ramachandran']
        self.statistical_tests = statistical_tests or ['wilcoxon', 'bootstrap', 'permutation']
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.multiple_testing_correction = multiple_testing_correction
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.structure_metrics = ProteinStructureMetrics()
        self.physics_metrics = PhysicsMetrics()
        self.biochemical_metrics = BiochemicalMetrics()
        self.dataset_loader = ProteinBenchmarkDatasets()
        self.statistical_analyzer = StatisticalAnalyzer(
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            correction_method=multiple_testing_correction
        )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('ProteinBenchmark')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.output_dir / 'benchmark.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        dataset_name: str,
        task_name: str = "structure_prediction",
        n_samples: Optional[int] = None,
        device: str = "cpu"
    ) -> BenchmarkResult:
        """
        Benchmark a single model on a specific dataset.
        
        Args:
            model: Neural operator model to benchmark
            model_name: Name identifier for the model
            dataset_name: Name of the dataset
            task_name: Name of the task
            n_samples: Number of samples to evaluate (None for all)
            device: Device to run evaluation on
            
        Returns:
            Comprehensive benchmark results
        """
        self.logger.info(f"Benchmarking {model_name} on {dataset_name}")
        
        # Load dataset
        dataset = self.dataset_loader.load_dataset(dataset_name)
        if n_samples is not None:
            dataset = dataset[:n_samples]
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Initialize results containers
        predictions = []
        targets = []
        timing_info = {'forward_times': [], 'total_time': 0}
        memory_info = {'peak_memory': 0, 'memory_efficient': True}
        
        start_time = time.time()
        
        # Evaluation loop
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                batch_start = time.time()
                
                # Memory monitoring
                if torch.cuda.is_available() and device != 'cpu':
                    torch.cuda.reset_peak_memory_stats()
                
                try:
                    # Forward pass
                    if isinstance(batch, dict):
                        inputs = batch['input']
                        target = batch['target']
                        constraints = batch.get('constraints', None)
                    else:
                        inputs, target = batch
                        constraints = None
                    
                    # Move to device
                    inputs = inputs.to(device)
                    target = target.to(device)
                    if constraints is not None:
                        constraints = constraints.to(device)
                    
                    # Model prediction
                    if constraints is not None:
                        prediction = model(inputs, constraints)
                    else:
                        prediction = model(inputs)
                    
                    predictions.append(prediction.cpu())
                    targets.append(target.cpu())
                    
                    # Timing
                    batch_time = time.time() - batch_start
                    timing_info['forward_times'].append(batch_time)
                    
                    # Memory usage
                    if torch.cuda.is_available() and device != 'cpu':
                        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                        memory_info['peak_memory'] = max(memory_info['peak_memory'], peak_memory)
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {i}: {str(e)}")
                    memory_info['memory_efficient'] = False
                    continue
        
        timing_info['total_time'] = time.time() - start_time
        
        # Concatenate predictions and targets
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets, dim=0)
        
        # Compute metrics
        metrics = self._compute_all_metrics(all_predictions, all_targets)
        
        # Error analysis
        error_analysis = self._perform_error_analysis(all_predictions, all_targets)
        
        # Model parameters
        model_params = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024**2),  # Assuming float32
        }
        
        # Runtime info
        runtime_info = {
            'device': device,
            'batch_size': len(predictions),
            'n_samples': len(all_predictions),
            'avg_forward_time': np.mean(timing_info['forward_times']),
            'std_forward_time': np.std(timing_info['forward_times']),
        }
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task_name,
            metrics=metrics,
            timing_info=timing_info,
            memory_usage=memory_info,
            model_params=model_params,
            runtime_info=runtime_info,
            error_analysis=error_analysis,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Save individual result
        self._save_result(result)
        
        return result
    
    def benchmark_models(
        self,
        models: List[Tuple[nn.Module, str]],
        cross_validation: bool = True,
        n_folds: int = 5
    ) -> List[BenchmarkResult]:
        """
        Benchmark multiple models across all datasets.
        
        Args:
            models: List of (model, name) tuples
            cross_validation: Whether to perform cross-validation
            n_folds: Number of CV folds
            
        Returns:
            List of benchmark results
        """
        all_results = []
        
        for model, model_name in models:
            self.logger.info(f"Starting benchmark for {model_name}")
            
            for dataset_name in self.datasets:
                if cross_validation:
                    # Cross-validation
                    cv_results = self._cross_validate_model(
                        model, model_name, dataset_name, n_folds
                    )
                    all_results.extend(cv_results)
                else:
                    # Single evaluation
                    result = self.benchmark_model(
                        model, model_name, dataset_name
                    )
                    all_results.append(result)
        
        return all_results
    
    def _cross_validate_model(
        self,
        model: nn.Module,
        model_name: str,
        dataset_name: str,
        n_folds: int
    ) -> List[BenchmarkResult]:
        """Perform cross-validation for a model."""
        self.logger.info(f"Cross-validating {model_name} on {dataset_name}")
        
        # Load full dataset
        dataset = self.dataset_loader.load_dataset(dataset_name)
        
        # Create folds
        fold_size = len(dataset) // n_folds
        cv_results = []
        
        for fold in range(n_folds):
            self.logger.info(f"Fold {fold + 1}/{n_folds}")
            
            # Split data
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(dataset)
            
            test_data = dataset[start_idx:end_idx]
            
            # Benchmark on fold
            fold_result = self.benchmark_model(
                model.clone() if hasattr(model, 'clone') else model,
                f"{model_name}_fold_{fold}",
                dataset_name
            )
            
            cv_results.append(fold_result)
        
        return cv_results
    
    def _compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all specified metrics."""
        metrics = {}
        
        for metric_name in self.metrics:
            try:
                if metric_name in ['rmsd', 'gdt_ts', 'tm_score', 'ldt']:
                    value = getattr(self.structure_metrics, metric_name)(predictions, targets)
                elif metric_name in ['bond_length_error', 'angle_error', 'clash_score']:
                    value = getattr(self.physics_metrics, metric_name)(predictions, targets)
                elif metric_name in ['ramachandran', 'secondary_structure', 'hydrophobic_core']:
                    value = getattr(self.biochemical_metrics, metric_name)(predictions, targets)
                else:
                    self.logger.warning(f"Unknown metric: {metric_name}")
                    continue
                
                metrics[metric_name] = float(value)
                
            except Exception as e:
                self.logger.error(f"Error computing {metric_name}: {str(e)}")
                metrics[metric_name] = float('nan')
        
        return metrics
    
    def _perform_error_analysis(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        errors = predictions - targets
        
        analysis = {
            'mean_error': torch.mean(errors).item(),
            'std_error': torch.std(errors).item(),
            'max_error': torch.max(torch.abs(errors)).item(),
            'median_error': torch.median(errors).item(),
            'error_percentiles': {
                '25th': torch.quantile(errors, 0.25).item(),
                '75th': torch.quantile(errors, 0.75).item(),
                '90th': torch.quantile(errors, 0.90).item(),
                '95th': torch.quantile(errors, 0.95).item(),
                '99th': torch.quantile(errors, 0.99).item(),
            },
            'error_distribution': {
                'skewness': self._compute_skewness(errors),
                'kurtosis': self._compute_kurtosis(errors),
            }
        }
        
        return analysis
    
    def _compute_skewness(self, x: torch.Tensor) -> float:
        """Compute skewness of a tensor."""
        mean = torch.mean(x)
        std = torch.std(x)
        skewness = torch.mean(((x - mean) / std) ** 3)
        return skewness.item()
    
    def _compute_kurtosis(self, x: torch.Tensor) -> float:
        """Compute kurtosis of a tensor."""
        mean = torch.mean(x)
        std = torch.std(x)
        kurtosis = torch.mean(((x - mean) / std) ** 4) - 3
        return kurtosis.item()
    
    def compare_models(
        self,
        results: List[BenchmarkResult],
        comparison_metric: str = 'rmsd'
    ) -> Dict[str, StatisticalTestResult]:
        """
        Perform statistical comparison between models.
        
        Args:
            results: List of benchmark results
            comparison_metric: Metric to use for comparison
            
        Returns:
            Dictionary of statistical test results
        """
        self.logger.info(f"Comparing models using {comparison_metric}")
        
        # Group results by model and dataset
        grouped_results = {}
        for result in results:
            key = f"{result.model_name}_{result.dataset_name}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result.metrics[comparison_metric])
        
        # Perform pairwise comparisons
        model_names = list(grouped_results.keys())
        comparison_results = {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                data1, data2 = grouped_results[name1], grouped_results[name2]
                
                comparison_key = f"{name1}_vs_{name2}"
                
                # Perform statistical tests
                test_results = {}
                for test_name in self.statistical_tests:
                    try:
                        if test_name == 'wilcoxon':
                            test_result = self.statistical_analyzer.wilcoxon_test(data1, data2)
                        elif test_name == 'bootstrap':
                            test_result = self.statistical_analyzer.bootstrap_test(data1, data2)
                        elif test_name == 'permutation':
                            test_result = self.statistical_analyzer.permutation_test(data1, data2)
                        else:
                            continue
                        
                        test_results[test_name] = test_result
                        
                    except Exception as e:
                        self.logger.error(f"Error in {test_name} test: {str(e)}")
                
                comparison_results[comparison_key] = test_results
        
        return comparison_results
    
    def _save_result(self, result: BenchmarkResult):
        """Save individual benchmark result."""
        filename = f"{result.model_name}_{result.dataset_name}_{result.task_name}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def save_all_results(self, results: List[BenchmarkResult]):
        """Save all benchmark results."""
        # Save individual results
        for result in results:
            self._save_result(result)
        
        # Save summary
        summary = {
            'total_results': len(results),
            'models': list(set(r.model_name for r in results)),
            'datasets': list(set(r.dataset_name for r in results)),
            'metrics': self.metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.output_dir / 'benchmark_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save results as pickle for easy loading
        with open(self.output_dir / 'all_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(self, filepath: Optional[str] = None) -> List[BenchmarkResult]:
        """Load benchmark results from file."""
        if filepath is None:
            filepath = self.output_dir / 'all_results.pkl'
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_file: str = "benchmark_report.html",
        include_plots: bool = True
    ):
        """
        Generate comprehensive benchmark report.
        
        Args:
            results: List of benchmark results
            output_file: Output file path
            include_plots: Whether to include plots
        """
        self.logger.info("Generating benchmark report")
        
        # HTML report generation would be implemented here
        # For now, we'll create a detailed text report
        
        report_content = []
        report_content.append("# Protein Neural Operator Benchmark Report\n")
        report_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        report_content.append("## Summary Statistics\n")
        report_content.append(f"Total benchmarks: {len(results)}\n")
        report_content.append(f"Models evaluated: {len(set(r.model_name for r in results))}\n")
        report_content.append(f"Datasets used: {len(set(r.dataset_name for r in results))}\n\n")
        
        # Model comparison
        report_content.append("## Model Performance Comparison\n")
        for metric in self.metrics:
            report_content.append(f"### {metric.upper()}\n")
            
            # Group by model
            model_performance = {}
            for result in results:
                if result.model_name not in model_performance:
                    model_performance[result.model_name] = []
                model_performance[result.model_name].append(result.metrics.get(metric, float('nan')))
            
            # Compute statistics
            for model_name, values in model_performance.items():
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    report_content.append(f"- {model_name}: {mean_val:.4f} ± {std_val:.4f}\n")
            
            report_content.append("\n")
        
        # Statistical significance tests
        if len(set(r.model_name for r in results)) > 1:
            report_content.append("## Statistical Significance Tests\n")
            comparison_results = self.compare_models(results)
            
            for comparison, tests in comparison_results.items():
                report_content.append(f"### {comparison}\n")
                for test_name, test_result in tests.items():
                    if hasattr(test_result, 'p_value'):
                        significance = "significant" if test_result.p_value < self.alpha else "not significant"
                        report_content.append(f"- {test_name}: p={test_result.p_value:.4f} ({significance})\n")
                report_content.append("\n")
        
        # Write report
        report_path = self.output_dir / output_file
        with open(report_path, 'w') as f:
            f.writelines(report_content)
        
        self.logger.info(f"Report saved to {report_path}")
    
    def run_comprehensive_benchmark(
        self,
        models: List[Tuple[nn.Module, str]],
        generate_report: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            models: List of (model, name) tuples
            generate_report: Whether to generate report
            
        Returns:
            List of all benchmark results
        """
        self.logger.info("Starting comprehensive benchmark suite")
        
        # Run benchmarks
        results = self.benchmark_models(models)
        
        # Save results
        self.save_all_results(results)
        
        # Generate report
        if generate_report:
            self.generate_report(results)
        
        self.logger.info("Benchmark suite completed")
        
        return results