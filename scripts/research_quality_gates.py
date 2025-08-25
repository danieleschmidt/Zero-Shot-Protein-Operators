#!/usr/bin/env python3
"""
Research Quality Gates - Comprehensive Validation Framework.

Automated quality assurance system for research reproducibility,
statistical validation, and publication readiness.

Features:
- Reproducibility validation across multiple runs
- Statistical significance verification
- Effect size validation and power analysis  
- Publication-ready metrics generation
- Automated research integrity checks
- Experimental protocol validation
- Code quality and documentation assessment

Quality Gates:
1. Reproducibility Gate: Results consistent across random seeds
2. Statistical Gate: Proper significance testing and corrections
3. Effect Size Gate: Meaningful practical significance
4. Publication Gate: Complete documentation and metadata
5. Code Quality Gate: Research code standards compliance
6. Data Integrity Gate: Dataset validation and provenance
7. Experimental Gate: Proper controls and baselines

Usage:
    python scripts/research_quality_gates.py --mode comprehensive
    python scripts/research_quality_gates.py --gate reproducibility
    python scripts/research_quality_gates.py --validate-experiment experiment_001

Author: Protein Operators Research Team
License: MIT
"""

import os
import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Import with fallbacks for testing
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some quality gates will use mock implementations.")

try:
    from scipy import stats
    from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical tests will use simplified implementations.")

# Import research modules
try:
    from protein_operators.research.quantum_classical_hybrid import (
        QuantumEnhancedProteinOperator,
        demonstrate_quantum_advantage
    )
    from protein_operators.benchmarks.advanced_comparative_studies import (
        AdvancedComparativeStudies,
        demonstrate_advanced_comparative_studies
    )
    RESEARCH_MODULES_AVAILABLE = True
except ImportError as e:
    RESEARCH_MODULES_AVAILABLE = False
    warnings.warn(f"Research modules not available: {e}")


@dataclass
class QualityGateResult:
    """Result from a single quality gate check."""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentValidation:
    """Validation results for a complete experiment."""
    experiment_id: str
    gate_results: List[QualityGateResult]
    overall_score: float
    passed_gates: int
    total_gates: int
    research_ready: bool
    publication_ready: bool
    recommendations: List[str] = field(default_factory=list)


class ResearchQualityGates:
    """
    Comprehensive research quality validation framework.
    
    Implements systematic quality checks for computational research,
    ensuring reproducibility, statistical rigor, and publication readiness.
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        significance_level: float = 0.01,  # Stricter for research
        effect_size_threshold: float = 0.3,
        reproducibility_threshold: float = 0.95,
        publication_threshold: float = 0.85
    ):
        self.random_seed = random_seed
        self.significance_level = significance_level
        self.effect_size_threshold = effect_size_threshold
        self.reproducibility_threshold = reproducibility_threshold
        self.publication_threshold = publication_threshold
        
        # Set random seeds
        np.random.seed(random_seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(random_seed)
        
        # Quality gate registry
        self.quality_gates = {
            'reproducibility': self.reproducibility_gate,
            'statistical': self.statistical_significance_gate,
            'effect_size': self.effect_size_gate,
            'publication': self.publication_readiness_gate,
            'code_quality': self.code_quality_gate,
            'data_integrity': self.data_integrity_gate,
            'experimental': self.experimental_design_gate
        }
        
        # Results storage
        self.validation_history = []
        
    def reproducibility_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Validate reproducibility across multiple experimental runs.
        
        Args:
            experiment_data: Dictionary containing experiment results
            
        Returns:
            Quality gate result for reproducibility
        """
        print("🔄 Running Reproducibility Quality Gate...")
        
        # Extract results from multiple runs
        if 'multi_run_results' not in experiment_data:
            return QualityGateResult(
                gate_name='reproducibility',
                passed=False,
                score=0.0,
                threshold=self.reproducibility_threshold,
                details={'error': 'No multi-run data available'},
                recommendations=['Run experiments with multiple random seeds',
                               'Collect at least 5 independent runs',
                               'Document random seed usage']
            )
        
        multi_run_results = experiment_data['multi_run_results']
        
        # Compute reproducibility metrics
        reproducibility_scores = []
        metric_consistency = {}
        
        for metric_name in ['success_rate', 'accuracy', 'performance']:
            if metric_name in multi_run_results[0]:
                metric_values = [run[metric_name] for run in multi_run_results]
                
                # Coefficient of variation (lower is more reproducible)
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                cv = std_val / mean_val if mean_val > 0 else float('inf')
                
                # Convert to reproducibility score (0-1, higher is better)
                reproducibility_score = max(0, 1 - cv)
                reproducibility_scores.append(reproducibility_score)
                
                metric_consistency[metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'cv': float(cv),
                    'reproducibility_score': float(reproducibility_score)
                }
        
        # Overall reproducibility score
        overall_score = np.mean(reproducibility_scores) if reproducibility_scores else 0.0
        passed = overall_score >= self.reproducibility_threshold
        
        # Generate recommendations
        recommendations = []
        if not passed:
            recommendations.extend([
                f'Reproducibility score ({overall_score:.3f}) below threshold ({self.reproducibility_threshold})',
                'Increase number of experimental runs',
                'Check for uncontrolled randomness in code',
                'Document all sources of randomness'
            ])
        
        if any(score < 0.9 for score in reproducibility_scores):
            recommendations.append('Some metrics show high variance - investigate causes')
        
        return QualityGateResult(
            gate_name='reproducibility',
            passed=passed,
            score=overall_score,
            threshold=self.reproducibility_threshold,
            details={
                'metric_consistency': metric_consistency,
                'individual_scores': reproducibility_scores,
                'num_runs': len(multi_run_results)
            },
            recommendations=recommendations
        )
    
    def statistical_significance_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Validate proper statistical significance testing.
        
        Args:
            experiment_data: Dictionary containing statistical test results
            
        Returns:
            Quality gate result for statistical significance
        """
        print("📊 Running Statistical Significance Quality Gate...")
        
        if 'statistical_tests' not in experiment_data:
            return QualityGateResult(
                gate_name='statistical',
                passed=False,
                score=0.0,
                threshold=self.significance_level,
                details={'error': 'No statistical test results available'},
                recommendations=['Conduct proper statistical significance tests',
                               'Include appropriate baselines',
                               'Apply multiple testing corrections']
            )
        
        statistical_tests = experiment_data['statistical_tests']
        
        # Analyze statistical tests
        total_tests = 0
        significant_tests = 0
        p_values = []
        test_quality_scores = []
        
        for test_category, tests in statistical_tests.items():
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    for subtest_name, subtest_data in test_result.items():
                        if isinstance(subtest_data, dict) and 'p_value' in subtest_data:
                            total_tests += 1
                            p_val = subtest_data['p_value']
                            p_values.append(p_val)
                            
                            if p_val < self.significance_level:
                                significant_tests += 1
                            
                            # Test quality score based on p-value and effect
                            quality_score = 1.0 if p_val < self.significance_level else p_val
                            test_quality_scores.append(quality_score)
        
        # Multiple testing correction check
        if len(p_values) > 1:
            # Bonferroni correction
            bonferroni_threshold = self.significance_level / len(p_values)
            bonferroni_significant = sum(p < bonferroni_threshold for p in p_values)
            
            # False Discovery Rate (Benjamini-Hochberg)
            if SCIPY_AVAILABLE:
                from scipy.stats import false_discovery_control
                try:
                    fdr_corrected = false_discovery_control(p_values, method='bh')
                    fdr_significant = sum(p < self.significance_level for p in fdr_corrected)
                except:
                    fdr_significant = bonferroni_significant
            else:
                fdr_significant = bonferroni_significant
        else:
            bonferroni_significant = significant_tests
            fdr_significant = significant_tests
        
        # Overall statistical quality score
        if total_tests > 0:
            statistical_power = significant_tests / total_tests
            correction_adjustment = fdr_significant / max(significant_tests, 1)
            overall_score = (statistical_power + correction_adjustment) / 2
        else:
            overall_score = 0.0
        
        passed = overall_score > 0.5 and significant_tests > 0
        
        # Recommendations
        recommendations = []
        if not passed:
            recommendations.extend([
                'No statistically significant results found',
                'Check experimental design and sample size',
                'Consider power analysis for adequate sample size'
            ])
        
        if total_tests > 5 and bonferroni_significant < significant_tests:
            recommendations.append('Apply multiple testing corrections (Bonferroni, FDR)')
        
        if len(p_values) > 0 and min(p_values) > 0.001:
            recommendations.append('Consider larger sample sizes for stronger statistical power')
        
        return QualityGateResult(
            gate_name='statistical',
            passed=passed,
            score=overall_score,
            threshold=0.5,  # At least 50% tests should be meaningful
            details={
                'total_tests': total_tests,
                'significant_tests': significant_tests,
                'bonferroni_significant': bonferroni_significant,
                'fdr_significant': fdr_significant,
                'min_p_value': float(min(p_values)) if p_values else 1.0,
                'statistical_power': statistical_power if total_tests > 0 else 0.0
            },
            recommendations=recommendations
        )
    
    def effect_size_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Validate meaningful effect sizes for practical significance.
        
        Args:
            experiment_data: Dictionary containing effect size measurements
            
        Returns:
            Quality gate result for effect sizes
        """
        print("📏 Running Effect Size Quality Gate...")
        
        if 'effect_sizes' not in experiment_data:
            return QualityGateResult(
                gate_name='effect_size',
                passed=False,
                score=0.0,
                threshold=self.effect_size_threshold,
                details={'error': 'No effect size data available'},
                recommendations=['Calculate Cohen\'s d or equivalent effect sizes',
                               'Report practical significance alongside statistical significance',
                               'Include confidence intervals for effect sizes']
            )
        
        effect_sizes = experiment_data['effect_sizes']
        
        # Analyze effect sizes
        all_effect_sizes = []
        large_effects = 0
        medium_effects = 0
        small_effects = 0
        negligible_effects = 0
        
        for category, effects in effect_sizes.items():
            for comparison, effect_data in effects.items():
                if isinstance(effect_data, dict) and 'cohens_d' in effect_data:
                    effect_size = abs(effect_data['cohens_d'])
                    all_effect_sizes.append(effect_size)
                    
                    # Classify effect size
                    if effect_size >= 0.8:
                        large_effects += 1
                    elif effect_size >= 0.5:
                        medium_effects += 1
                    elif effect_size >= 0.2:
                        small_effects += 1
                    else:
                        negligible_effects += 1
        
        # Overall effect size score
        if all_effect_sizes:
            mean_effect_size = np.mean(all_effect_sizes)
            max_effect_size = max(all_effect_sizes)
            
            # Score based on proportion of meaningful effects
            total_effects = len(all_effect_sizes)
            meaningful_effects = large_effects + medium_effects
            effect_size_score = meaningful_effects / total_effects if total_effects > 0 else 0.0
            
            # Boost score for large effects
            if large_effects > 0:
                effect_size_score += 0.2 * (large_effects / total_effects)
            
            overall_score = min(1.0, effect_size_score)
        else:
            mean_effect_size = 0.0
            max_effect_size = 0.0
            overall_score = 0.0
        
        passed = overall_score >= 0.5 and mean_effect_size >= self.effect_size_threshold
        
        # Recommendations
        recommendations = []
        if not passed:
            recommendations.extend([
                f'Mean effect size ({mean_effect_size:.3f}) below threshold ({self.effect_size_threshold})',
                'Results may lack practical significance',
                'Consider experimental modifications to increase effect sizes'
            ])
        
        if negligible_effects > total_effects // 2:
            recommendations.append('Many effects are negligible - check experimental conditions')
        
        if large_effects == 0:
            recommendations.append('No large effects found - consider power analysis')
        
        return QualityGateResult(
            gate_name='effect_size',
            passed=passed,
            score=overall_score,
            threshold=self.effect_size_threshold,
            details={
                'mean_effect_size': float(mean_effect_size),
                'max_effect_size': float(max_effect_size),
                'large_effects': large_effects,
                'medium_effects': medium_effects,
                'small_effects': small_effects,
                'negligible_effects': negligible_effects,
                'total_comparisons': len(all_effect_sizes)
            },
            recommendations=recommendations
        )
    
    def publication_readiness_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Assess publication readiness based on documentation and completeness.
        
        Args:
            experiment_data: Dictionary containing experiment metadata
            
        Returns:
            Quality gate result for publication readiness
        """
        print("📄 Running Publication Readiness Quality Gate...")
        
        # Check required components for publication
        required_components = {
            'abstract': 'Abstract/summary of research',
            'methodology': 'Detailed methodology description',
            'results': 'Comprehensive results',
            'statistical_analysis': 'Statistical analysis and significance',
            'discussion': 'Discussion of results and implications',
            'limitations': 'Study limitations and future work',
            'reproducibility': 'Reproducibility information',
            'code_availability': 'Code and data availability'
        }
        
        available_components = []
        missing_components = []
        component_scores = {}
        
        for component, description in required_components.items():
            if component in experiment_data:
                available_components.append(component)
                
                # Score component completeness
                data = experiment_data[component]
                if isinstance(data, str) and len(data) > 50:
                    component_scores[component] = 1.0
                elif isinstance(data, dict) and len(data) > 3:
                    component_scores[component] = 1.0
                elif isinstance(data, list) and len(data) > 0:
                    component_scores[component] = 1.0
                else:
                    component_scores[component] = 0.5
            else:
                missing_components.append(component)
                component_scores[component] = 0.0
        
        # Overall publication score
        overall_score = np.mean(list(component_scores.values()))
        passed = overall_score >= self.publication_threshold
        
        # Additional quality checks
        quality_indicators = {}
        
        # Check for figures/visualizations
        if 'figures' in experiment_data:
            quality_indicators['has_figures'] = True
        else:
            quality_indicators['has_figures'] = False
        
        # Check for proper citations
        if 'citations' in experiment_data or 'references' in experiment_data:
            quality_indicators['has_citations'] = True
        else:
            quality_indicators['has_citations'] = False
        
        # Check for ethical considerations
        if 'ethics' in experiment_data or 'ethical_approval' in experiment_data:
            quality_indicators['addresses_ethics'] = True
        else:
            quality_indicators['addresses_ethics'] = False
        
        # Recommendations
        recommendations = []
        if not passed:
            recommendations.append(f'Publication score ({overall_score:.3f}) below threshold ({self.publication_threshold})')
        
        for component in missing_components:
            recommendations.append(f'Missing: {required_components[component]}')
        
        if not quality_indicators['has_figures']:
            recommendations.append('Add figures and visualizations')
        
        if not quality_indicators['has_citations']:
            recommendations.append('Include proper citations and references')
        
        if not quality_indicators['addresses_ethics']:
            recommendations.append('Address ethical considerations if applicable')
        
        return QualityGateResult(
            gate_name='publication',
            passed=passed,
            score=overall_score,
            threshold=self.publication_threshold,
            details={
                'available_components': available_components,
                'missing_components': missing_components,
                'component_scores': component_scores,
                'quality_indicators': quality_indicators,
                'completeness': len(available_components) / len(required_components)
            },
            recommendations=recommendations
        )
    
    def code_quality_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Validate code quality and documentation standards.
        
        Args:
            experiment_data: Dictionary containing code metadata
            
        Returns:
            Quality gate result for code quality
        """
        print("💻 Running Code Quality Gate...")
        
        # Check for code-related information
        code_quality_score = 0.0
        quality_checks = {}
        
        # Check for documentation
        if 'docstrings' in experiment_data:
            quality_checks['has_docstrings'] = True
            code_quality_score += 0.2
        else:
            quality_checks['has_docstrings'] = False
        
        # Check for type hints
        if 'type_hints' in experiment_data:
            quality_checks['has_type_hints'] = True
            code_quality_score += 0.15
        else:
            quality_checks['has_type_hints'] = False
        
        # Check for tests
        if 'tests' in experiment_data or 'test_coverage' in experiment_data:
            quality_checks['has_tests'] = True
            code_quality_score += 0.25
            
            if 'test_coverage' in experiment_data:
                coverage = experiment_data['test_coverage']
                if isinstance(coverage, (int, float)) and coverage >= 80:
                    code_quality_score += 0.1
        else:
            quality_checks['has_tests'] = False
        
        # Check for code style compliance
        if 'code_style' in experiment_data:
            quality_checks['follows_style_guide'] = True
            code_quality_score += 0.1
        else:
            quality_checks['follows_style_guide'] = False
        
        # Check for reproducibility artifacts
        if 'requirements' in experiment_data or 'environment' in experiment_data:
            quality_checks['has_dependencies'] = True
            code_quality_score += 0.1
        else:
            quality_checks['has_dependencies'] = False
        
        # Check for version control
        if 'git_info' in experiment_data or 'version_control' in experiment_data:
            quality_checks['uses_version_control'] = True
            code_quality_score += 0.1
        else:
            quality_checks['uses_version_control'] = False
        
        # Check for README/documentation
        if 'readme' in experiment_data or 'documentation' in experiment_data:
            quality_checks['has_documentation'] = True
            code_quality_score += 0.1
        else:
            quality_checks['has_documentation'] = False
        
        passed = code_quality_score >= 0.7  # Require 70% of quality checks
        
        # Recommendations
        recommendations = []
        if not passed:
            recommendations.append(f'Code quality score ({code_quality_score:.3f}) below threshold (0.70)')
        
        if not quality_checks['has_docstrings']:
            recommendations.append('Add comprehensive docstrings to all functions and classes')
        
        if not quality_checks['has_type_hints']:
            recommendations.append('Add type hints for better code clarity')
        
        if not quality_checks['has_tests']:
            recommendations.append('Implement comprehensive unit tests')
        
        if not quality_checks['follows_style_guide']:
            recommendations.append('Follow consistent code style guide (PEP 8 for Python)')
        
        if not quality_checks['has_dependencies']:
            recommendations.append('Document all dependencies and requirements')
        
        if not quality_checks['uses_version_control']:
            recommendations.append('Use version control (Git) for code management')
        
        return QualityGateResult(
            gate_name='code_quality',
            passed=passed,
            score=code_quality_score,
            threshold=0.7,
            details=quality_checks,
            recommendations=recommendations
        )
    
    def data_integrity_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Validate data integrity and provenance.
        
        Args:
            experiment_data: Dictionary containing data information
            
        Returns:
            Quality gate result for data integrity
        """
        print("🗄️ Running Data Integrity Quality Gate...")
        
        integrity_score = 0.0
        integrity_checks = {}
        
        # Check for data provenance
        if 'data_source' in experiment_data:
            integrity_checks['has_data_source'] = True
            integrity_score += 0.2
        else:
            integrity_checks['has_data_source'] = False
        
        # Check for data validation
        if 'data_validation' in experiment_data:
            integrity_checks['validates_data'] = True
            integrity_score += 0.2
        else:
            integrity_checks['validates_data'] = False
        
        # Check for data preprocessing documentation
        if 'preprocessing_steps' in experiment_data:
            integrity_checks['documents_preprocessing'] = True
            integrity_score += 0.15
        else:
            integrity_checks['documents_preprocessing'] = False
        
        # Check for missing data handling
        if 'missing_data_handling' in experiment_data:
            integrity_checks['handles_missing_data'] = True
            integrity_score += 0.15
        else:
            integrity_checks['handles_missing_data'] = False
        
        # Check for outlier analysis
        if 'outlier_analysis' in experiment_data:
            integrity_checks['analyzes_outliers'] = True
            integrity_score += 0.1
        else:
            integrity_checks['analyzes_outliers'] = False
        
        # Check for data splits documentation
        if 'data_splits' in experiment_data:
            integrity_checks['documents_data_splits'] = True
            integrity_score += 0.1
        else:
            integrity_checks['documents_data_splits'] = False
        
        # Check for bias analysis
        if 'bias_analysis' in experiment_data:
            integrity_checks['analyzes_bias'] = True
            integrity_score += 0.1
        else:
            integrity_checks['analyzes_bias'] = False
        
        passed = integrity_score >= 0.6
        
        # Recommendations
        recommendations = []
        if not passed:
            recommendations.append(f'Data integrity score ({integrity_score:.3f}) below threshold (0.60)')
        
        if not integrity_checks['has_data_source']:
            recommendations.append('Document data sources and collection methods')
        
        if not integrity_checks['validates_data']:
            recommendations.append('Implement data validation checks')
        
        if not integrity_checks['documents_preprocessing']:
            recommendations.append('Document all preprocessing steps')
        
        if not integrity_checks['handles_missing_data']:
            recommendations.append('Address missing data handling strategy')
        
        if not integrity_checks['analyzes_outliers']:
            recommendations.append('Perform outlier detection and analysis')
        
        if not integrity_checks['documents_data_splits']:
            recommendations.append('Document train/validation/test splits')
        
        return QualityGateResult(
            gate_name='data_integrity',
            passed=passed,
            score=integrity_score,
            threshold=0.6,
            details=integrity_checks,
            recommendations=recommendations
        )
    
    def experimental_design_gate(self, experiment_data: Dict[str, Any]) -> QualityGateResult:
        """
        Validate experimental design and controls.
        
        Args:
            experiment_data: Dictionary containing experimental design info
            
        Returns:
            Quality gate result for experimental design
        """
        print("🔬 Running Experimental Design Quality Gate...")
        
        design_score = 0.0
        design_checks = {}
        
        # Check for hypothesis statement
        if 'hypothesis' in experiment_data:
            design_checks['has_hypothesis'] = True
            design_score += 0.2
        else:
            design_checks['has_hypothesis'] = False
        
        # Check for control groups/baselines
        if 'control_groups' in experiment_data or 'baselines' in experiment_data:
            design_checks['has_controls'] = True
            design_score += 0.25
        else:
            design_checks['has_controls'] = False
        
        # Check for randomization
        if 'randomization' in experiment_data:
            design_checks['uses_randomization'] = True
            design_score += 0.15
        else:
            design_checks['uses_randomization'] = False
        
        # Check for sample size justification
        if 'sample_size_justification' in experiment_data or 'power_analysis' in experiment_data:
            design_checks['justifies_sample_size'] = True
            design_score += 0.15
        else:
            design_checks['justifies_sample_size'] = False
        
        # Check for blinding (if applicable)
        if 'blinding' in experiment_data:
            design_checks['uses_blinding'] = True
            design_score += 0.1
        else:
            design_checks['uses_blinding'] = False
        
        # Check for confound control
        if 'confound_control' in experiment_data:
            design_checks['controls_confounds'] = True
            design_score += 0.15
        else:
            design_checks['controls_confounds'] = False
        
        passed = design_score >= 0.6
        
        # Recommendations
        recommendations = []
        if not passed:
            recommendations.append(f'Experimental design score ({design_score:.3f}) below threshold (0.60)')
        
        if not design_checks['has_hypothesis']:
            recommendations.append('Clearly state research hypothesis')
        
        if not design_checks['has_controls']:
            recommendations.append('Include appropriate control groups or baselines')
        
        if not design_checks['uses_randomization']:
            recommendations.append('Implement proper randomization procedures')
        
        if not design_checks['justifies_sample_size']:
            recommendations.append('Provide sample size justification or power analysis')
        
        if not design_checks['controls_confounds']:
            recommendations.append('Address potential confounding variables')
        
        return QualityGateResult(
            gate_name='experimental',
            passed=passed,
            score=design_score,
            threshold=0.6,
            details=design_checks,
            recommendations=recommendations
        )
    
    def run_quality_gates(
        self,
        experiment_data: Dict[str, Any],
        gates_to_run: Optional[List[str]] = None
    ) -> List[QualityGateResult]:
        """
        Run specified quality gates on experiment data.
        
        Args:
            experiment_data: Complete experiment data and results
            gates_to_run: List of specific gates to run, or None for all
            
        Returns:
            List of quality gate results
        """
        if gates_to_run is None:
            gates_to_run = list(self.quality_gates.keys())
        
        print(f"🚀 Running {len(gates_to_run)} Quality Gates...")
        print(f"Gates: {', '.join(gates_to_run)}")
        
        results = []
        
        for gate_name in gates_to_run:
            if gate_name in self.quality_gates:
                print(f"\n{'='*50}")
                gate_function = self.quality_gates[gate_name]
                
                try:
                    result = gate_function(experiment_data)
                    results.append(result)
                    
                    # Print gate result
                    status = "✅ PASSED" if result.passed else "❌ FAILED"
                    print(f"{status} - {gate_name}: {result.score:.3f}/{result.threshold:.3f}")
                    
                    if result.recommendations:
                        print("📝 Recommendations:")
                        for rec in result.recommendations[:3]:  # Show top 3
                            print(f"   • {rec}")
                    
                except Exception as e:
                    print(f"❌ ERROR in {gate_name}: {e}")
                    results.append(QualityGateResult(
                        gate_name=gate_name,
                        passed=False,
                        score=0.0,
                        threshold=1.0,
                        details={'error': str(e)},
                        recommendations=[f'Fix error in {gate_name} gate']
                    ))
            else:
                print(f"⚠️  Unknown quality gate: {gate_name}")
        
        return results
    
    def validate_experiment(
        self,
        experiment_id: str,
        experiment_data: Dict[str, Any],
        gates_to_run: Optional[List[str]] = None
    ) -> ExperimentValidation:
        """
        Comprehensive validation of an experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_data: Complete experiment data
            gates_to_run: Specific gates to run
            
        Returns:
            Complete experiment validation results
        """
        print(f"🔬 Validating Experiment: {experiment_id}")
        print("="*60)
        
        # Run quality gates
        gate_results = self.run_quality_gates(experiment_data, gates_to_run)
        
        # Compute overall metrics
        passed_gates = sum(1 for result in gate_results if result.passed)
        total_gates = len(gate_results)
        
        # Weighted overall score (some gates are more important)
        gate_weights = {
            'reproducibility': 0.25,
            'statistical': 0.20,
            'effect_size': 0.15,
            'publication': 0.15,
            'experimental': 0.10,
            'code_quality': 0.10,
            'data_integrity': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in gate_results:
            weight = gate_weights.get(result.gate_name, 0.1)
            weighted_score += weight * result.score
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine readiness levels
        research_ready = passed_gates >= total_gates * 0.6  # 60% gates pass
        publication_ready = passed_gates >= total_gates * 0.8 and overall_score >= 0.8  # 80% gates pass + high score
        
        # Collect all recommendations
        all_recommendations = []
        for result in gate_results:
            if not result.passed:
                all_recommendations.extend(result.recommendations)
        
        # Create validation result
        validation = ExperimentValidation(
            experiment_id=experiment_id,
            gate_results=gate_results,
            overall_score=overall_score,
            passed_gates=passed_gates,
            total_gates=total_gates,
            research_ready=research_ready,
            publication_ready=publication_ready,
            recommendations=list(set(all_recommendations))  # Remove duplicates
        )
        
        # Store validation history
        self.validation_history.append(validation)
        
        # Print summary
        self._print_validation_summary(validation)
        
        return validation
    
    def _print_validation_summary(self, validation: ExperimentValidation):
        """Print comprehensive validation summary."""
        
        print(f"\n🎯 Experiment Validation Summary")
        print("="*50)
        
        print(f"Experiment ID: {validation.experiment_id}")
        print(f"Overall Score: {validation.overall_score:.3f}/1.000")
        print(f"Gates Passed: {validation.passed_gates}/{validation.total_gates}")
        print(f"Research Ready: {'✅ YES' if validation.research_ready else '❌ NO'}")
        print(f"Publication Ready: {'✅ YES' if validation.publication_ready else '❌ NO'}")
        
        # Gate results summary
        print(f"\n📋 Quality Gate Results:")
        for result in validation.gate_results:
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.gate_name:15s}: {result.score:.3f}/{result.threshold:.3f}")
        
        # Top recommendations
        if validation.recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(validation.recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        # Next steps
        print(f"\n🚀 Next Steps:")
        if validation.publication_ready:
            print("   • Experiment is ready for publication submission")
            print("   • Consider peer review and additional validation")
        elif validation.research_ready:
            print("   • Experiment meets research standards")
            print("   • Address remaining recommendations for publication")
        else:
            print("   • Address critical quality gate failures")
            print("   • Improve experimental design and analysis")
        
        print(f"\n{'='*60}")


def create_mock_experiment_data() -> Dict[str, Any]:
    """Create mock experiment data for demonstration."""
    
    # Simulate quantum vs classical comparison results
    return {
        'experiment_id': 'quantum_vs_classical_protein_design',
        'hypothesis': 'Quantum-classical hybrid algorithms provide significant advantages over classical methods for protein design optimization',
        
        # Multi-run results for reproducibility
        'multi_run_results': [
            {'success_rate': 0.85, 'accuracy': 0.82, 'performance': 0.78},
            {'success_rate': 0.87, 'accuracy': 0.84, 'performance': 0.79},
            {'success_rate': 0.86, 'accuracy': 0.83, 'performance': 0.77},
            {'success_rate': 0.84, 'accuracy': 0.81, 'performance': 0.80},
            {'success_rate': 0.88, 'accuracy': 0.85, 'performance': 0.81}
        ],
        
        # Statistical test results
        'statistical_tests': {
            'success_rate': {
                'quantum_vs_classical': {
                    'wilcoxon': {'statistic': 45.2, 'p_value': 0.001, 'significant': True},
                    'mann_whitney': {'statistic': 1250, 'p_value': 0.003, 'significant': True}
                }
            },
            'computation_time': {
                'quantum_vs_classical': {
                    'paired_ttest': {'statistic': -3.45, 'p_value': 0.008, 'significant': True}
                }
            }
        },
        
        # Effect sizes
        'effect_sizes': {
            'success_rate': {
                'quantum_vs_classical': {
                    'cohens_d': 0.78,
                    'interpretation': 'medium',
                    'magnitude': 0.78,
                    'favors': 'QuantumEnhanced'
                }
            },
            'computation_time': {
                'quantum_vs_classical': {
                    'cohens_d': -1.23,
                    'interpretation': 'large', 
                    'magnitude': 1.23,
                    'favors': 'QuantumEnhanced'
                }
            }
        },
        
        # Publication components
        'abstract': 'This study presents a comprehensive comparison of quantum-classical hybrid neural operators versus classical approaches for protein design optimization.',
        'methodology': 'We implemented quantum-enhanced algorithms using QAOA and VQE, comparing against classical baselines across multiple benchmark datasets.',
        'results': 'Quantum-classical hybrid approaches achieved 15-20% improvement in success rates with 2-3x speedup in computation time.',
        'statistical_analysis': 'Statistical significance established using Wilcoxon signed-rank tests with Bonferroni correction.',
        'discussion': 'Results demonstrate quantum advantage in protein design optimization, particularly for large-scale problems.',
        'limitations': 'Current implementation uses quantum simulation; real quantum hardware may show different performance characteristics.',
        'reproducibility': 'All experiments conducted with multiple random seeds and documented computational environments.',
        'code_availability': 'Complete source code and datasets available at https://github.com/danieleschmidt/Zero-Shot-Protein-Operators',
        
        # Code quality indicators
        'docstrings': True,
        'type_hints': True,
        'tests': True,
        'test_coverage': 85,
        'code_style': True,
        'requirements': True,
        'git_info': True,
        'readme': True,
        
        # Data integrity
        'data_source': 'PDB database and synthetic benchmark datasets',
        'data_validation': 'Comprehensive data validation including outlier detection',
        'preprocessing_steps': 'Standardization, normalization, and feature engineering documented',
        'missing_data_handling': 'Missing data imputed using domain-specific methods',
        'outlier_analysis': 'Statistical outlier detection with manual review',
        'data_splits': 'Stratified 70/15/15 train/validation/test splits',
        
        # Experimental design
        'control_groups': 'Classical baselines including Rosetta and AlphaFold-based methods',
        'randomization': 'Randomized experimental conditions with controlled seeds',
        'sample_size_justification': 'Power analysis conducted for 80% power at α=0.05',
        'confound_control': 'Controlled for computational resources and dataset complexity'
    }


def main():
    """Main function for research quality gates validation."""
    
    parser = argparse.ArgumentParser(description='Research Quality Gates Validation')
    parser.add_argument('--mode', choices=['comprehensive', 'quick', 'custom'],
                       default='comprehensive', help='Validation mode')
    parser.add_argument('--gate', action='append', 
                       help='Specific quality gate to run (can be used multiple times)')
    parser.add_argument('--experiment-id', default='demo_experiment',
                       help='Experiment identifier')
    parser.add_argument('--data-file', help='JSON file with experiment data')
    parser.add_argument('--output-file', help='Output file for validation results')
    parser.add_argument('--threshold-reproducibility', type=float, default=0.95,
                       help='Reproducibility threshold')
    parser.add_argument('--threshold-effect-size', type=float, default=0.3,
                       help='Effect size threshold')
    
    args = parser.parse_args()
    
    # Initialize quality gates
    quality_gates = ResearchQualityGates(
        reproducibility_threshold=args.threshold_reproducibility,
        effect_size_threshold=args.threshold_effect_size
    )
    
    print("🔬 Research Quality Gates - Comprehensive Validation Framework")
    print("="*70)
    
    # Load experiment data
    if args.data_file:
        with open(args.data_file, 'r') as f:
            experiment_data = json.load(f)
        print(f"📂 Loaded experiment data from: {args.data_file}")
    else:
        print("📂 Using demonstration experiment data...")
        experiment_data = create_mock_experiment_data()
    
    # Determine gates to run
    if args.mode == 'comprehensive':
        gates_to_run = None  # Run all gates
    elif args.mode == 'quick':
        gates_to_run = ['reproducibility', 'statistical', 'effect_size']
    elif args.mode == 'custom':
        gates_to_run = args.gate if args.gate else ['reproducibility']
    
    # Run validation
    validation = quality_gates.validate_experiment(
        experiment_id=args.experiment_id,
        experiment_data=experiment_data,
        gates_to_run=gates_to_run
    )
    
    # Save results if requested
    if args.output_file:
        output_data = {
            'experiment_id': validation.experiment_id,
            'overall_score': validation.overall_score,
            'passed_gates': validation.passed_gates,
            'total_gates': validation.total_gates,
            'research_ready': validation.research_ready,
            'publication_ready': validation.publication_ready,
            'gate_results': [
                {
                    'gate_name': r.gate_name,
                    'passed': r.passed,
                    'score': r.score,
                    'threshold': r.threshold,
                    'recommendations': r.recommendations
                }
                for r in validation.gate_results
            ],
            'recommendations': validation.recommendations,
            'timestamp': time.time()
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Validation results saved to: {args.output_file}")
    
    # Run actual research validation if modules available
    if RESEARCH_MODULES_AVAILABLE and args.mode == 'comprehensive':
        print(f"\n🔬 Running Research Module Validation...")
        
        try:
            # Test quantum advantage demonstration
            quantum_results = demonstrate_quantum_advantage()
            print("✅ Quantum advantage demonstration successful")
            
            # Test comparative studies
            comparative_results = demonstrate_advanced_comparative_studies()  
            print("✅ Advanced comparative studies demonstration successful")
            
        except Exception as e:
            print(f"⚠️  Research module validation failed: {e}")
    
    print(f"\n🏆 Research Quality Gates Validation Complete!")
    print(f"Experiment: {validation.experiment_id}")
    print(f"Overall Quality: {validation.overall_score:.1%}")
    print(f"Research Ready: {'YES' if validation.research_ready else 'NO'}")
    print(f"Publication Ready: {'YES' if validation.publication_ready else 'NO'}")


if __name__ == "__main__":
    main()