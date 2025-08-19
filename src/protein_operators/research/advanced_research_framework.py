"""
Advanced research framework for novel protein design methodologies.

Features:
- Automated hypothesis generation and testing
- Multi-objective experimental design
- Statistical significance validation
- Reproducible research protocols
- Publication-ready result generation
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    import mock_torch as torch
    nn = torch.nn
    F = torch.nn.functional

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import json
import pickle
from datetime import datetime
from collections import defaultdict
import itertools
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import ProteinDesigner
from ..models.enhanced_deeponet import EnhancedProteinDeepONet
from ..constraints import Constraints
from ..structure import ProteinStructure
from ..validation.advanced_validation import AdvancedValidationFramework
from ..utils.advanced_logger import AdvancedLogger


class ExperimentType(Enum):
    """Types of research experiments."""
    COMPARATIVE_STUDY = "comparative_study"
    ABLATION_STUDY = "ablation_study"
    PARAMETER_SWEEP = "parameter_sweep"
    NOVEL_ARCHITECTURE = "novel_architecture"
    BENCHMARK_EVALUATION = "benchmark_evaluation"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"


class StatisticalTest(Enum):
    """Statistical tests for result validation."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"


@dataclass
class Hypothesis:
    """Research hypothesis definition."""
    name: str
    description: str
    null_hypothesis: str
    alternative_hypothesis: str
    expected_effect_size: float
    significance_level: float = 0.05
    power: float = 0.8
    metrics: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalCondition:
    """Single experimental condition."""
    name: str
    parameters: Dict[str, Any]
    replications: int = 5
    randomization_seed: Optional[int] = None
    control_condition: bool = False


@dataclass
class ExperimentResult:
    """Result from a single experimental run."""
    condition_name: str
    replication_id: int
    metrics: Dict[str, float]
    structure: Optional[ProteinStructure] = None
    execution_time: float = 0.0
    validation_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StatisticalResult:
    """Statistical analysis result."""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False


class ExperimentalDesigner:
    """
    Automated experimental design for protein research studies.
    
    Generates experimental conditions, ensures proper controls,
    and optimizes sample sizes for statistical power.
    """
    
    def __init__(self):
        self.logger = AdvancedLogger(__name__)
    
    def design_comparative_study(
        self,
        baseline_method: Dict[str, Any],
        comparison_methods: List[Dict[str, Any]],
        evaluation_metrics: List[str],
        effect_size: float = 0.5,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> List[ExperimentalCondition]:
        """Design a comparative study between methods."""
        # Calculate required sample size
        sample_size = self._calculate_sample_size(effect_size, power, alpha)
        
        conditions = []
        
        # Baseline condition
        baseline_condition = ExperimentalCondition(
            name="baseline",
            parameters=baseline_method,
            replications=sample_size,
            control_condition=True
        )
        conditions.append(baseline_condition)
        
        # Comparison conditions
        for i, method in enumerate(comparison_methods):
            condition = ExperimentalCondition(
                name=f"comparison_{i+1}",
                parameters=method,
                replications=sample_size
            )
            conditions.append(condition)
        
        self.logger.info(f"Designed comparative study with {len(conditions)} conditions, {sample_size} replications each")
        return conditions
    
    def design_ablation_study(
        self,
        base_configuration: Dict[str, Any],
        ablation_components: List[str],
        evaluation_metrics: List[str]
    ) -> List[ExperimentalCondition]:
        """Design an ablation study to test component importance."""
        conditions = []
        
        # Full model condition
        full_condition = ExperimentalCondition(
            name="full_model",
            parameters=base_configuration.copy(),
            replications=10,
            control_condition=True
        )
        conditions.append(full_condition)
        
        # Ablated conditions
        for component in ablation_components:
            ablated_config = base_configuration.copy()
            
            # Disable the component (implementation depends on component type)
            if component in ablated_config:
                ablated_config[component] = False
            elif f"enable_{component}" in ablated_config:
                ablated_config[f"enable_{component}"] = False
            
            condition = ExperimentalCondition(
                name=f"ablate_{component}",
                parameters=ablated_config,
                replications=10
            )
            conditions.append(condition)
        
        self.logger.info(f"Designed ablation study with {len(conditions)} conditions")
        return conditions
    
    def design_parameter_sweep(
        self,
        base_configuration: Dict[str, Any],
        parameter_ranges: Dict[str, List[Any]],
        evaluation_metrics: List[str],
        max_conditions: int = 50
    ) -> List[ExperimentalCondition]:
        """Design parameter sweep experiment."""
        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit number of conditions if too many
        if len(all_combinations) > max_conditions:
            # Use systematic sampling
            step = len(all_combinations) // max_conditions
            selected_combinations = all_combinations[::step][:max_conditions]
            self.logger.warning(f"Too many combinations ({len(all_combinations)}), selected {len(selected_combinations)}")
        else:
            selected_combinations = all_combinations
        
        conditions = []
        for i, combination in enumerate(selected_combinations):
            config = base_configuration.copy()
            
            # Update configuration with parameter values
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            condition = ExperimentalCondition(
                name=f"param_sweep_{i+1}",
                parameters=config,
                replications=5  # Fewer replications for parameter sweeps
            )
            conditions.append(condition)
        
        self.logger.info(f"Designed parameter sweep with {len(conditions)} conditions")
        return conditions
    
    def _calculate_sample_size(
        self,
        effect_size: float,
        power: float,
        alpha: float,
        test_type: str = "two_sample_t_test"
    ) -> int:
        """Calculate required sample size for statistical power."""
        # Simplified power analysis (in practice, would use more sophisticated methods)
        if test_type == "two_sample_t_test":
            # Cohen's d effect size
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return max(5, int(np.ceil(n)))  # Minimum 5 samples
        
        return 10  # Default fallback


class StatisticalAnalyzer:
    """
    Statistical analysis framework for research results.
    
    Performs appropriate statistical tests, calculates effect sizes,
    and provides interpretation of results.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = AdvancedLogger(__name__)
    
    def analyze_results(
        self,
        results: List[ExperimentResult],
        hypotheses: List[Hypothesis]
    ) -> Dict[str, List[StatisticalResult]]:
        """Perform comprehensive statistical analysis."""
        analysis_results = {}
        
        # Group results by condition
        condition_results = defaultdict(list)
        for result in results:
            condition_results[result.condition_name].append(result)
        
        for hypothesis in hypotheses:
            hypothesis_results = []
            
            for metric in hypothesis.metrics:
                # Extract metric values for each condition
                condition_metrics = {}
                for condition, cond_results in condition_results.items():
                    values = [r.metrics.get(metric, 0.0) for r in cond_results]
                    if values:  # Only include conditions with data
                        condition_metrics[condition] = values
                
                if len(condition_metrics) < 2:
                    self.logger.warning(f"Insufficient conditions for analysis of {metric}")
                    continue
                
                # Perform appropriate statistical test
                if len(condition_metrics) == 2:
                    # Two-group comparison
                    stat_result = self._two_group_test(condition_metrics, metric)
                else:
                    # Multi-group comparison
                    stat_result = self._multi_group_test(condition_metrics, metric)
                
                hypothesis_results.append(stat_result)
            
            analysis_results[hypothesis.name] = hypothesis_results
        
        return analysis_results
    
    def _two_group_test(
        self,
        condition_metrics: Dict[str, List[float]],
        metric_name: str
    ) -> StatisticalResult:
        """Perform two-group statistical test."""
        conditions = list(condition_metrics.keys())
        group1 = condition_metrics[conditions[0]]
        group2 = condition_metrics[conditions[1]]
        
        # Check normality (simplified)
        normal1 = self._check_normality(group1)
        normal2 = self._check_normality(group2)
        
        if normal1 and normal2:
            # Use t-test
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_type = StatisticalTest.T_TEST
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) + 
                                 (len(group2) - 1) * np.var(group2)) / 
                                (len(group1) + len(group2) - 2))
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            # Use Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_type = StatisticalTest.MANN_WHITNEY
            
            # Calculate rank-biserial correlation as effect size
            r = 1 - (2 * statistic) / (len(group1) * len(group2))
            effect_size = r
        
        # Interpret results
        significant = p_value < self.significance_level
        interpretation = self._interpret_result(p_value, effect_size, significant)
        
        return StatisticalResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            significant=significant,
            interpretation=interpretation
        )
    
    def _multi_group_test(
        self,
        condition_metrics: Dict[str, List[float]],
        metric_name: str
    ) -> StatisticalResult:
        """Perform multi-group statistical test."""
        groups = list(condition_metrics.values())
        
        # Check if all groups are normal
        all_normal = all(self._check_normality(group) for group in groups)
        
        if all_normal:
            # Use ANOVA
            statistic, p_value = stats.f_oneway(*groups)
            test_type = StatisticalTest.ANOVA
            
            # Calculate eta-squared as effect size
            grand_mean = np.mean([val for group in groups for val in group])
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
            ss_total = sum((val - grand_mean)**2 for group in groups for val in group)
            effect_size = ss_between / ss_total if ss_total > 0 else 0
        else:
            # Use Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*groups)
            test_type = StatisticalTest.KRUSKAL_WALLIS
            
            # Calculate eta-squared approximation
            n_total = sum(len(group) for group in groups)
            effect_size = (statistic - len(groups) + 1) / (n_total - len(groups))
        
        significant = p_value < self.significance_level
        interpretation = self._interpret_result(p_value, effect_size, significant)
        
        return StatisticalResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            significant=significant,
            interpretation=interpretation
        )
    
    def _check_normality(self, data: List[float], alpha: float = 0.05) -> bool:
        """Check if data is normally distributed."""
        if len(data) < 3:
            return False
        
        try:
            _, p_value = stats.shapiro(data)
            return p_value > alpha
        except Exception:
            return False
    
    def _interpret_result(
        self,
        p_value: float,
        effect_size: float,
        significant: bool
    ) -> str:
        """Interpret statistical result."""
        if not significant:
            return f"No significant difference found (p = {p_value:.4f})"
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "positive" if effect_size > 0 else "negative"
        
        return f"Significant {direction} effect (p = {p_value:.4f}, effect size = {effect_size:.3f}, {magnitude} magnitude)"


class ResultVisualizer:
    """
    Visualization system for research results.
    
    Generates publication-quality plots and figures.
    """
    
    def __init__(self, style: str = "seaborn-v0_8"):
        try:
            plt.style.use(style)
        except Exception:
            pass  # Fall back to default style
        self.logger = AdvancedLogger(__name__)
    
    def plot_comparative_results(
        self,
        results: List[ExperimentResult],
        metric: str,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comparative box plot for metric across conditions."""
        # Group results by condition
        condition_data = defaultdict(list)
        for result in results:
            if metric in result.metrics:
                condition_data[result.condition_name].append(result.metrics[metric])
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions = list(condition_data.keys())
        data = [condition_data[cond] for cond in conditions]
        
        box_plot = ax.boxplot(data, labels=conditions, patch_artist=True)
        
        # Customize appearance
        colors = plt.cm.Set3(np.linspace(0, 1, len(conditions)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'Comparison of {metric} Across Conditions', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotations (simplified)
        self._add_significance_annotations(ax, data, conditions)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparative plot to {output_path}")
        
        return fig
    
    def plot_parameter_sweep(
        self,
        results: List[ExperimentResult],
        parameter: str,
        metric: str,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Create parameter sweep plot."""
        # Extract parameter values and metric values
        param_values = []
        metric_values = []
        
        for result in results:
            if parameter in result.metadata and metric in result.metrics:
                param_values.append(result.metadata[parameter])
                metric_values.append(result.metrics[metric])
        
        if not param_values:
            self.logger.warning(f"No data found for parameter {parameter} and metric {metric}")
            return plt.figure()
        
        # Create scatter plot with trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(param_values, metric_values, alpha=0.6, s=50)
        
        # Add trend line if data is numeric
        try:
            param_numeric = [float(p) for p in param_values]
            z = np.polyfit(param_numeric, metric_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(param_numeric), max(param_numeric), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        except (ValueError, TypeError):
            pass  # Skip trend line for non-numeric parameters
        
        ax.set_title(f'{metric} vs {parameter}', fontsize=14, fontweight='bold')
        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved parameter sweep plot to {output_path}")
        
        return fig
    
    def plot_correlation_matrix(
        self,
        results: List[ExperimentResult],
        metrics: List[str],
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """Create correlation matrix heatmap for metrics."""
        # Extract metric data
        metric_data = {metric: [] for metric in metrics}
        
        for result in results:
            for metric in metrics:
                if metric in result.metrics:
                    metric_data[metric].append(result.metrics[metric])
                else:
                    metric_data[metric].append(np.nan)
        
        # Create correlation matrix
        import pandas as pd
        df = pd.DataFrame(metric_data)
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            fmt='.3f'
        )
        
        ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved correlation matrix to {output_path}")
        
        return fig
    
    def _add_significance_annotations(self, ax, data, conditions):
        """Add significance annotations to box plot."""
        # Simplified significance testing and annotation
        if len(data) >= 2:
            try:
                stat, p_value = stats.ttest_ind(data[0], data[1])
                if p_value < 0.05:
                    y_max = max(max(group) for group in data)
                    y_annotation = y_max * 1.1
                    
                    ax.annotate('*', xy=(1.5, y_annotation), ha='center', va='bottom', fontsize=16)
                    ax.plot([1, 2], [y_annotation * 0.98, y_annotation * 0.98], 'k-', linewidth=1)
            except Exception:
                pass  # Skip annotation if test fails


class AdvancedResearchFramework:
    """
    Comprehensive research framework for protein design studies.
    
    Integrates experimental design, execution, analysis, and visualization
    into a unified research platform.
    """
    
    def __init__(
        self,
        output_directory: str = "research_outputs",
        enable_validation: bool = True,
        max_concurrent_experiments: int = 4
    ):
        self.logger = AdvancedLogger(__name__)
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.experimental_designer = ExperimentalDesigner()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer()
        
        if enable_validation:
            self.validator = AdvancedValidationFramework()
        else:
            self.validator = None
        
        # Execution infrastructure
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_experiments)
        
        # Results storage
        self.experiment_results = []
        self.analysis_results = {}
        
        self.logger.info(f"Advanced Research Framework initialized, output directory: {output_directory}")
    
    async def conduct_research_study(
        self,
        study_name: str,
        experiment_type: ExperimentType,
        hypotheses: List[Hypothesis],
        base_configuration: Dict[str, Any],
        **experiment_kwargs
    ) -> Dict[str, Any]:
        """Conduct a complete research study."""
        study_start_time = time.time()
        study_id = f"{study_name}_{int(study_start_time)}"
        
        self.logger.info(f"Starting research study: {study_name} ({experiment_type.value})")
        
        # 1. Design experiments
        conditions = self._design_experiments(
            experiment_type, base_configuration, **experiment_kwargs
        )
        
        # 2. Execute experiments
        results = await self._execute_experiments(study_id, conditions)
        
        # 3. Analyze results
        analysis = self._analyze_results(results, hypotheses)
        
        # 4. Generate visualizations
        visualizations = self._generate_visualizations(study_id, results, experiment_kwargs)
        
        # 5. Create research report
        report = self._generate_research_report(
            study_name, study_id, experiment_type, hypotheses,
            conditions, results, analysis, visualizations
        )
        
        # 6. Save study data
        self._save_study_data(study_id, {
            'study_name': study_name,
            'experiment_type': experiment_type.value,
            'hypotheses': hypotheses,
            'conditions': conditions,
            'results': results,
            'analysis': analysis,
            'report': report,
            'execution_time': time.time() - study_start_time
        })
        
        self.logger.info(f"Research study completed: {study_name}")
        
        return {
            'study_id': study_id,
            'results': results,
            'analysis': analysis,
            'report': report,
            'visualizations': visualizations
        }
    
    def _design_experiments(
        self,
        experiment_type: ExperimentType,
        base_configuration: Dict[str, Any],
        **kwargs
    ) -> List[ExperimentalCondition]:
        """Design experiments based on type."""
        if experiment_type == ExperimentType.COMPARATIVE_STUDY:
            return self.experimental_designer.design_comparative_study(
                baseline_method=base_configuration,
                comparison_methods=kwargs.get('comparison_methods', []),
                evaluation_metrics=kwargs.get('evaluation_metrics', []),
                effect_size=kwargs.get('effect_size', 0.5)
            )
        
        elif experiment_type == ExperimentType.ABLATION_STUDY:
            return self.experimental_designer.design_ablation_study(
                base_configuration=base_configuration,
                ablation_components=kwargs.get('ablation_components', []),
                evaluation_metrics=kwargs.get('evaluation_metrics', [])
            )
        
        elif experiment_type == ExperimentType.PARAMETER_SWEEP:
            return self.experimental_designer.design_parameter_sweep(
                base_configuration=base_configuration,
                parameter_ranges=kwargs.get('parameter_ranges', {}),
                evaluation_metrics=kwargs.get('evaluation_metrics', [])
            )
        
        else:
            # Default: single condition
            return [ExperimentalCondition(
                name="default",
                parameters=base_configuration,
                replications=10
            )]
    
    async def _execute_experiments(
        self,
        study_id: str,
        conditions: List[ExperimentalCondition]
    ) -> List[ExperimentResult]:
        """Execute all experimental conditions."""
        all_results = []
        
        # Create tasks for all experimental runs
        tasks = []
        for condition in conditions:
            for rep in range(condition.replications):
                task = asyncio.create_task(
                    self._execute_single_experiment(study_id, condition, rep)
                )
                tasks.append(task)
        
        # Execute all tasks concurrently
        self.logger.info(f"Executing {len(tasks)} experimental runs")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and collect successful results
        for result in results:
            if isinstance(result, ExperimentResult):
                all_results.append(result)
            else:
                self.logger.error(f"Experiment failed: {result}")
        
        self.logger.info(f"Completed {len(all_results)} experimental runs")
        return all_results
    
    async def _execute_single_experiment(
        self,
        study_id: str,
        condition: ExperimentalCondition,
        replication_id: int
    ) -> ExperimentResult:
        """Execute a single experimental run."""
        start_time = time.time()
        
        try:
            # Initialize designer with condition parameters
            designer = ProteinDesigner(**condition.parameters)
            
            # Generate test structure (simplified for framework demo)
            from ..constraints import Constraints
            constraints = Constraints()
            
            structure = designer.generate(
                constraints=constraints,
                length=condition.parameters.get('length', 100),
                num_samples=1
            )
            
            # Validate structure if validator available
            validation_results = None
            if self.validator:
                validation_report = self.validator.validate_structure(structure)
                validation_results = {
                    'overall_score': validation_report.overall_score,
                    'passed': validation_report.passed,
                    'metrics': {m.name: m.value for m in validation_report.metrics}
                }
            
            # Extract metrics
            metrics = designer.validate(structure)
            
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                condition_name=condition.name,
                replication_id=replication_id,
                metrics=metrics,
                structure=structure,
                execution_time=execution_time,
                validation_results=validation_results,
                metadata=condition.parameters
            )
        
        except Exception as e:
            self.logger.error(f"Experiment failed for {condition.name}, rep {replication_id}: {e}")
            # Return empty result
            return ExperimentResult(
                condition_name=condition.name,
                replication_id=replication_id,
                metrics={},
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _analyze_results(
        self,
        results: List[ExperimentResult],
        hypotheses: List[Hypothesis]
    ) -> Dict[str, Any]:
        """Analyze experimental results."""
        analysis = self.statistical_analyzer.analyze_results(results, hypotheses)
        
        # Add descriptive statistics
        condition_stats = self._compute_descriptive_statistics(results)
        analysis['descriptive_statistics'] = condition_stats
        
        return analysis
    
    def _compute_descriptive_statistics(
        self,
        results: List[ExperimentResult]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute descriptive statistics for each condition and metric."""
        condition_metrics = defaultdict(lambda: defaultdict(list))
        
        # Group metrics by condition
        for result in results:
            for metric, value in result.metrics.items():
                condition_metrics[result.condition_name][metric].append(value)
        
        # Compute statistics
        stats_results = {}
        for condition, metrics in condition_metrics.items():
            stats_results[condition] = {}
            for metric, values in metrics.items():
                if values:
                    stats_results[condition][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'count': len(values)
                    }
        
        return stats_results
    
    def _generate_visualizations(
        self,
        study_id: str,
        results: List[ExperimentResult],
        experiment_kwargs: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate visualization plots."""
        visualizations = {}
        
        # Get common metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Generate comparative plots for key metrics
        key_metrics = list(all_metrics)[:5]  # Limit to first 5 metrics
        
        for metric in key_metrics:
            plot_path = self.output_dir / f"{study_id}_comparative_{metric}.png"
            try:
                fig = self.visualizer.plot_comparative_results(
                    results, metric, str(plot_path)
                )
                visualizations[f"comparative_{metric}"] = str(plot_path)
                plt.close(fig)
            except Exception as e:
                self.logger.error(f"Failed to generate plot for {metric}: {e}")
        
        # Generate correlation matrix if enough metrics
        if len(key_metrics) >= 3:
            corr_path = self.output_dir / f"{study_id}_correlation_matrix.png"
            try:
                fig = self.visualizer.plot_correlation_matrix(
                    results, key_metrics, str(corr_path)
                )
                visualizations["correlation_matrix"] = str(corr_path)
                plt.close(fig)
            except Exception as e:
                self.logger.error(f"Failed to generate correlation matrix: {e}")
        
        return visualizations
    
    def _generate_research_report(
        self,
        study_name: str,
        study_id: str,
        experiment_type: ExperimentType,
        hypotheses: List[Hypothesis],
        conditions: List[ExperimentalCondition],
        results: List[ExperimentResult],
        analysis: Dict[str, Any],
        visualizations: Dict[str, str]
    ) -> str:
        """Generate comprehensive research report."""
        report_lines = [
            f"# Research Study Report: {study_name}",
            f"**Study ID:** {study_id}",
            f"**Experiment Type:** {experiment_type.value}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Abstract",
            f"This study investigated {study_name} using a {experiment_type.value} approach. ",
            f"A total of {len(conditions)} experimental conditions were tested with ",
            f"{sum(c.replications for c in conditions)} total experimental runs.",
            "",
            "## Hypotheses",
        ]
        
        for i, hypothesis in enumerate(hypotheses, 1):
            report_lines.extend([
                f"### Hypothesis {i}: {hypothesis.name}",
                f"**Description:** {hypothesis.description}",
                f"**Null Hypothesis:** {hypothesis.null_hypothesis}",
                f"**Alternative Hypothesis:** {hypothesis.alternative_hypothesis}",
                ""
            ])
        
        report_lines.extend([
            "## Experimental Design",
            f"**Number of Conditions:** {len(conditions)}",
            f"**Total Experimental Runs:** {len(results)}",
            "",
            "### Conditions:"
        ])
        
        for condition in conditions:
            report_lines.append(f"- **{condition.name}**: {condition.replications} replications")
        
        report_lines.extend([
            "",
            "## Results",
            "### Descriptive Statistics"
        ])
        
        # Add descriptive statistics
        if 'descriptive_statistics' in analysis:
            for condition, metrics in analysis['descriptive_statistics'].items():
                report_lines.append(f"\n#### {condition}")
                for metric, stats in metrics.items():
                    report_lines.append(
                        f"- **{metric}**: Mean = {stats['mean']:.3f} ± {stats['std']:.3f} "
                        f"(Range: {stats['min']:.3f} - {stats['max']:.3f})"
                    )
        
        report_lines.extend([
            "",
            "### Statistical Analysis"
        ])
        
        # Add statistical results
        for hypothesis_name, stat_results in analysis.items():
            if hypothesis_name != 'descriptive_statistics':
                report_lines.append(f"\n#### {hypothesis_name}")
                for stat_result in stat_results:
                    report_lines.extend([
                        f"- **Test:** {stat_result.test_type.value}",
                        f"- **p-value:** {stat_result.p_value:.4f}",
                        f"- **Effect Size:** {stat_result.effect_size:.3f}",
                        f"- **Significant:** {stat_result.significant}",
                        f"- **Interpretation:** {stat_result.interpretation}",
                        ""
                    ])
        
        if visualizations:
            report_lines.extend([
                "## Visualizations",
                ""
            ])
            
            for viz_name, viz_path in visualizations.items():
                report_lines.append(f"- {viz_name}: {viz_path}")
        
        report_lines.extend([
            "",
            "## Conclusions",
            "[To be filled based on results interpretation]",
            "",
            "## Reproducibility",
            f"This study was conducted using the Advanced Research Framework.",
            f"All experimental conditions and parameters are documented in the study data.",
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / f"{study_id}_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Generated research report: {report_path}")
        return report_text
    
    def _save_study_data(self, study_id: str, study_data: Dict[str, Any]):
        """Save complete study data for reproducibility."""
        # Save as pickle for complete data
        pickle_path = self.output_dir / f"{study_id}_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(study_data, f)
        
        # Save summary as JSON (excluding complex objects)
        json_data = {
            'study_name': study_data['study_name'],
            'experiment_type': study_data['experiment_type'],
            'num_conditions': len(study_data['conditions']),
            'num_results': len(study_data['results']),
            'execution_time': study_data['execution_time']
        }
        
        json_path = self.output_dir / f"{study_id}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Saved study data: {pickle_path}, {json_path}")
    
    async def shutdown(self):
        """Gracefully shutdown the research framework."""
        self.logger.info("Shutting down Advanced Research Framework")
        self.executor.shutdown(wait=True)
        
        if self.validator:
            await self.validator.shutdown()
        
        self.logger.info("Research framework shutdown complete")
