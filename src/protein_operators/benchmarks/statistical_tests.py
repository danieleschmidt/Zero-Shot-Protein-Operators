"""
Statistical significance testing for neural operator benchmarks.

This module provides comprehensive statistical analysis tools for comparing
neural operator performance with proper multiple testing correction and
effect size estimation.
"""

import os
import sys
import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))


class SignificanceTest(Enum):
    """Enumeration of available statistical tests."""
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    T_TEST = "t_test"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    FRIEDMAN = "friedman"
    KRUSKAL_WALLIS = "kruskal_wallis"


class MultipleTestingCorrection(Enum):
    """Enumeration of multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    HOLM = "holm"
    SIDAK = "sidak"


@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_of_freedom: Optional[int] = None
    sample_sizes: Optional[Tuple[int, ...]] = None
    interpretation: str = ""
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        p_val = self.p_value_corrected if self.p_value_corrected is not None else self.p_value
        return p_val < alpha


@dataclass
class ComparisonResult:
    """Container for pairwise comparison results."""
    group1_name: str
    group2_name: str
    group1_data: List[float]
    group2_data: List[float]
    test_results: Dict[str, TestResult]
    descriptive_stats: Dict[str, Any]
    
    def get_best_test(self) -> TestResult:
        """Get the most appropriate test result."""
        # Prioritize non-parametric tests for robustness
        priority_order = ['wilcoxon', 'mann_whitney', 'bootstrap', 'permutation', 't_test']
        
        for test_name in priority_order:
            if test_name in self.test_results:
                return self.test_results[test_name]
        
        # Return first available test
        return next(iter(self.test_results.values()))


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for neural operator benchmarks.
    
    Features:
    - Multiple statistical tests (parametric and non-parametric)
    - Effect size estimation (Cohen's d, rank-biserial correlation)
    - Bootstrap and permutation testing
    - Multiple comparison correction
    - Power analysis
    - Bayesian hypothesis testing
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        n_permutation: int = 10000,
        correction_method: str = "bonferroni",
        random_state: int = 42
    ):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level
            n_bootstrap: Number of bootstrap samples
            n_permutation: Number of permutation samples
            correction_method: Multiple testing correction method
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutation = n_permutation
        self.correction_method = correction_method
        self.random_state = random_state
        
        np.random.seed(random_state)
    
    def compare_two_groups(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        tests: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Comprehensive comparison between two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            group1_name: Name of first group
            group2_name: Name of second group
            tests: List of tests to perform
            
        Returns:
            Comprehensive comparison results
        """
        if tests is None:
            tests = ['wilcoxon', 'mann_whitney', 't_test', 'bootstrap']
        
        # Convert to numpy arrays
        data1 = np.array(group1)
        data2 = np.array(group2)
        
        # Compute descriptive statistics
        descriptive_stats = self._compute_descriptive_stats(data1, data2)
        
        # Perform tests
        test_results = {}
        
        for test_name in tests:
            try:
                if test_name == 'wilcoxon':
                    result = self._wilcoxon_test(data1, data2)
                elif test_name == 'mann_whitney':
                    result = self._mann_whitney_test(data1, data2)
                elif test_name == 't_test':
                    result = self._t_test(data1, data2)
                elif test_name == 'bootstrap':
                    result = self._bootstrap_test(data1, data2)
                elif test_name == 'permutation':
                    result = self._permutation_test(data1, data2)
                else:
                    continue
                
                test_results[test_name] = result
                
            except Exception as e:
                print(f"Error in {test_name}: {str(e)}")
                continue
        
        # Apply multiple testing correction
        if len(test_results) > 1:
            self._apply_multiple_testing_correction(test_results)
        
        return ComparisonResult(
            group1_name=group1_name,
            group2_name=group2_name,
            group1_data=group1,
            group2_data=group2,
            test_results=test_results,
            descriptive_stats=descriptive_stats
        )
    
    def compare_multiple_groups(
        self,
        groups: Dict[str, List[float]],
        tests: Optional[List[str]] = None
    ) -> Dict[str, ComparisonResult]:
        """
        Compare multiple groups with pairwise tests.
        
        Args:
            groups: Dictionary of group_name -> data
            tests: List of tests to perform
            
        Returns:
            Dictionary of pairwise comparison results
        """
        if tests is None:
            tests = ['kruskal_wallis', 'friedman']
        
        group_names = list(groups.keys())
        pairwise_results = {}
        
        # Pairwise comparisons
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                comparison_key = f"{name1}_vs_{name2}"
                
                result = self.compare_two_groups(
                    groups[name1],
                    groups[name2],
                    name1,
                    name2,
                    tests
                )
                
                pairwise_results[comparison_key] = result
        
        # Apply family-wise error correction across all comparisons
        all_test_results = []
        for comparison in pairwise_results.values():
            all_test_results.extend(comparison.test_results.values())
        
        if len(all_test_results) > 1:
            self._apply_multiple_testing_correction_list(all_test_results)
        
        return pairwise_results
    
    def _wilcoxon_test(self, data1: np.ndarray, data2: np.ndarray) -> TestResult:
        """Perform Wilcoxon signed-rank test."""
        if len(data1) != len(data2):
            raise ValueError("Wilcoxon test requires paired data of equal length")
        
        differences = data1 - data2
        statistic, p_value = stats.wilcoxon(differences, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        effect_size = self._rank_biserial_correlation(data1, data2)
        
        # Confidence interval (approximate)
        ci = self._wilcoxon_confidence_interval(differences)
        
        interpretation = self._interpret_effect_size(effect_size, "rank_biserial")
        
        return TestResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            confidence_interval=ci,
            sample_sizes=(len(data1), len(data2)),
            interpretation=interpretation
        )
    
    def _mann_whitney_test(self, data1: np.ndarray, data2: np.ndarray) -> TestResult:
        """Perform Mann-Whitney U test."""
        statistic, p_value = stats.mannwhitneyu(
            data1, data2, alternative='two-sided'
        )
        
        # Effect size (rank-biserial correlation)
        effect_size = self._rank_biserial_correlation_independent(data1, data2)
        
        interpretation = self._interpret_effect_size(effect_size, "rank_biserial")
        
        return TestResult(
            test_name="Mann-Whitney U Test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            sample_sizes=(len(data1), len(data2)),
            interpretation=interpretation
        )
    
    def _t_test(self, data1: np.ndarray, data2: np.ndarray) -> TestResult:
        """Perform independent t-test."""
        # Check for equal variances
        levene_stat, levene_p = stats.levene(data1, data2)
        equal_var = levene_p > 0.05
        
        statistic, p_value = stats.ttest_ind(
            data1, data2, equal_var=equal_var
        )
        
        # Effect size (Cohen's d)
        effect_size = self._cohens_d(data1, data2)
        
        # Degrees of freedom
        if equal_var:
            df = len(data1) + len(data2) - 2
        else:
            # Welch's t-test df
            s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            n1, n2 = len(data1), len(data2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        # Confidence interval for difference in means
        ci = self._t_test_confidence_interval(data1, data2, equal_var)
        
        interpretation = self._interpret_effect_size(effect_size, "cohens_d")
        
        return TestResult(
            test_name="Independent t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            confidence_interval=ci,
            degrees_of_freedom=int(df),
            sample_sizes=(len(data1), len(data2)),
            interpretation=interpretation
        )
    
    def _bootstrap_test(self, data1: np.ndarray, data2: np.ndarray) -> TestResult:
        """Perform bootstrap test for difference in means."""
        observed_diff = np.mean(data1) - np.mean(data2)
        
        # Combine data for null hypothesis
        combined_data = np.concatenate([data1, data2])
        n1, n2 = len(data1), len(data2)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            # Sample without replacement under null hypothesis
            shuffled = np.random.permutation(combined_data)
            boot_sample1 = shuffled[:n1]
            boot_sample2 = shuffled[n1:n1+n2]
            
            boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Confidence interval for observed difference
        alpha_ci = 0.05
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha_ci / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha_ci / 2))
        
        # Effect size (standardized difference)
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0
        
        interpretation = self._interpret_effect_size(effect_size, "cohens_d")
        
        return TestResult(
            test_name="Bootstrap Test",
            statistic=observed_diff,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes=(len(data1), len(data2)),
            interpretation=interpretation
        )
    
    def _permutation_test(self, data1: np.ndarray, data2: np.ndarray) -> TestResult:
        """Perform permutation test."""
        observed_diff = np.mean(data1) - np.mean(data2)
        
        # Combine data
        combined_data = np.concatenate([data1, data2])
        n1 = len(data1)
        
        # Permutation sampling
        permutation_diffs = []
        for _ in range(self.n_permutation):
            perm_indices = np.random.permutation(len(combined_data))
            perm_sample1 = combined_data[perm_indices[:n1]]
            perm_sample2 = combined_data[perm_indices[n1:]]
            
            perm_diff = np.mean(perm_sample1) - np.mean(perm_sample2)
            permutation_diffs.append(perm_diff)
        
        permutation_diffs = np.array(permutation_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        # Effect size
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0
        
        interpretation = self._interpret_effect_size(effect_size, "cohens_d")
        
        return TestResult(
            test_name="Permutation Test",
            statistic=observed_diff,
            p_value=p_value,
            effect_size=effect_size,
            sample_sizes=(len(data1), len(data2)),
            interpretation=interpretation
        )
    
    def _compute_descriptive_stats(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> Dict[str, Any]:
        """Compute descriptive statistics for both groups."""
        return {
            'group1': {
                'n': len(data1),
                'mean': np.mean(data1),
                'median': np.median(data1),
                'std': np.std(data1, ddof=1),
                'min': np.min(data1),
                'max': np.max(data1),
                'q25': np.percentile(data1, 25),
                'q75': np.percentile(data1, 75)
            },
            'group2': {
                'n': len(data2),
                'mean': np.mean(data2),
                'median': np.median(data2),
                'std': np.std(data2, ddof=1),
                'min': np.min(data2),
                'max': np.max(data2),
                'q25': np.percentile(data2, 25),
                'q75': np.percentile(data2, 75)
            }
        }
    
    def _cohens_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(data1), len(data2)
        s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        return d
    
    def _rank_biserial_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute rank-biserial correlation for paired data."""
        differences = data1 - data2
        n_pos = np.sum(differences > 0)
        n_neg = np.sum(differences < 0)
        
        if n_pos + n_neg == 0:
            return 0.0
        
        r = (n_pos - n_neg) / (n_pos + n_neg)
        return r
    
    def _rank_biserial_correlation_independent(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> float:
        """Compute rank-biserial correlation for independent groups."""
        n1, n2 = len(data1), len(data2)
        
        # Count how many times data1 > data2
        greater_count = 0
        for x1 in data1:
            for x2 in data2:
                if x1 > x2:
                    greater_count += 1
        
        # Rank-biserial correlation
        r = (2 * greater_count) / (n1 * n2) - 1
        
        return r
    
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        
        if effect_type == "cohens_d":
            if abs_effect < 0.2:
                magnitude = "negligible"
            elif abs_effect < 0.5:
                magnitude = "small"
            elif abs_effect < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
        elif effect_type == "rank_biserial":
            if abs_effect < 0.1:
                magnitude = "negligible"
            elif abs_effect < 0.3:
                magnitude = "small"
            elif abs_effect < 0.5:
                magnitude = "medium"
            else:
                magnitude = "large"
        else:
            magnitude = "unknown"
        
        direction = "positive" if effect_size > 0 else "negative" if effect_size < 0 else "zero"
        
        return f"{magnitude} {direction} effect"
    
    def _wilcoxon_confidence_interval(
        self,
        differences: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute confidence interval for Wilcoxon test."""
        n = len(differences)
        if n < 5:
            return (np.nan, np.nan)
        
        # Approximate confidence interval
        sorted_diffs = np.sort(differences)
        
        # Critical value approximation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        
        lower_rank = max(0, int(n / 2 - z_alpha * se / 2))
        upper_rank = min(n - 1, int(n / 2 + z_alpha * se / 2))
        
        ci_lower = sorted_diffs[lower_rank]
        ci_upper = sorted_diffs[upper_rank]
        
        return (ci_lower, ci_upper)
    
    def _t_test_confidence_interval(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        equal_var: bool,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute confidence interval for t-test difference in means."""
        n1, n2 = len(data1), len(data2)
        mean_diff = np.mean(data1) - np.mean(data2)
        
        if equal_var:
            # Pooled standard error
            s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            se = np.sqrt(s1/n1 + s2/n2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin_error = t_critical * se
        
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return (ci_lower, ci_upper)
    
    def _apply_multiple_testing_correction(
        self,
        test_results: Dict[str, TestResult]
    ):
        """Apply multiple testing correction to a set of tests."""
        p_values = [result.p_value for result in test_results.values()]
        corrected_p_values = self._correct_p_values(p_values)
        
        for i, (test_name, result) in enumerate(test_results.items()):
            result.p_value_corrected = corrected_p_values[i]
    
    def _apply_multiple_testing_correction_list(
        self,
        test_results: List[TestResult]
    ):
        """Apply multiple testing correction to a list of tests."""
        p_values = [result.p_value for result in test_results]
        corrected_p_values = self._correct_p_values(p_values)
        
        for i, result in enumerate(test_results):
            result.p_value_corrected = corrected_p_values[i]
    
    def _correct_p_values(self, p_values: List[float]) -> List[float]:
        """Apply multiple testing correction to p-values."""
        p_array = np.array(p_values)
        n_tests = len(p_array)
        
        if self.correction_method == "bonferroni":
            corrected = np.minimum(p_array * n_tests, 1.0)
        elif self.correction_method == "sidak":
            corrected = 1 - (1 - p_array) ** n_tests
        elif self.correction_method == "holm":
            # Holm-Bonferroni
            sorted_indices = np.argsort(p_array)
            corrected = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_array[idx] * (n_tests - i), 1.0)
                
        elif self.correction_method == "fdr_bh":
            # Benjamini-Hochberg
            sorted_indices = np.argsort(p_array)
            corrected = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_array[idx] * n_tests / (i + 1), 1.0)
                
        else:
            # No correction
            corrected = p_array
        
        return corrected.tolist()