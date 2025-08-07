#!/usr/bin/env python3
"""
Quality gates enforcement script for protein operators.

This script enforces quality standards and can be run in CI/CD or locally.
"""

import sys
import subprocess
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    details: Optional[str] = None
    output: Optional[str] = None
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        score_info = f" (Score: {self.score:.1f}/{self.threshold})" if self.score and self.threshold else ""
        return f"{status} {self.name}{score_info}"


class QualityGateEnforcer:
    """
    Comprehensive quality gate enforcement system.
    """
    
    def __init__(self, project_root: Path = None, config: Dict[str, Any] = None):
        """
        Initialize quality gate enforcer.
        
        Args:
            project_root: Root directory of the project
            config: Configuration for quality gates
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or self._load_default_config()
        self.results: List[QualityGateResult] = []
        
        logger.info(f"Quality gate enforcer initialized for {self.project_root}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default quality gate configuration."""
        return {
            'code_quality': {
                'test_coverage_threshold': 85.0,
                'max_complexity': 10,
                'max_line_length': 88,
                'type_coverage_threshold': 90.0
            },
            'security': {
                'max_high_vulnerabilities': 0,
                'max_medium_vulnerabilities': 5,
                'allow_known_vulnerabilities': []
            },
            'performance': {
                'max_memory_usage_mb': 2048,
                'max_cpu_usage_percent': 80,
                'max_test_duration_seconds': 300
            },
            'documentation': {
                'min_docstring_coverage': 80.0,
                'require_api_docs': True
            },
            'dependencies': {
                'max_outdated_packages': 10,
                'check_licenses': True,
                'allowed_licenses': [
                    'MIT', 'BSD', 'Apache', 'Apache 2.0', 'Apache Software License',
                    'BSD License', 'MIT License', 'Python Software Foundation License'
                ]
            }
        }
    
    def run_all_gates(self) -> bool:
        """
        Run all quality gates.
        
        Returns:
            True if all gates pass, False otherwise
        """
        logger.info("Starting quality gate enforcement")
        
        # Code quality gates
        self.check_code_formatting()
        self.check_linting()
        self.check_type_hints()
        self.check_test_coverage()
        self.check_complexity()
        
        # Security gates
        self.check_security_vulnerabilities()
        self.check_dependency_licenses()
        
        # Performance gates
        self.check_performance_tests()
        
        # Documentation gates
        self.check_documentation_coverage()
        
        # Dependency gates
        self.check_outdated_dependencies()
        
        # Print results
        self._print_results()
        
        # Determine overall success
        all_passed = all(result.passed for result in self.results)
        
        if all_passed:
            logger.info("🎉 All quality gates passed!")
        else:
            failed_count = sum(1 for r in self.results if not r.passed)
            logger.error(f"💥 {failed_count} quality gates failed")
        
        return all_passed
    
    def check_code_formatting(self) -> None:
        """Check code formatting with Black."""
        logger.info("Checking code formatting...")
        
        try:
            result = subprocess.run([
                'black', '--check', '--diff', 'src', 'tests'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            details = "Code formatting is consistent" if passed else "Code formatting issues found"
            
            self.results.append(QualityGateResult(
                name="Code Formatting (Black)",
                passed=passed,
                details=details,
                output=result.stdout + result.stderr
            ))
            
        except FileNotFoundError:
            self.results.append(QualityGateResult(
                name="Code Formatting (Black)",
                passed=False,
                details="Black not installed"
            ))
    
    def check_linting(self) -> None:
        """Check code linting with Flake8."""
        logger.info("Checking linting...")
        
        try:
            result = subprocess.run([
                'flake8', 'src', 'tests',
                '--max-line-length=88',
                '--extend-ignore=E203,W503',
                '--statistics'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            details = "No linting issues" if passed else "Linting issues found"
            
            self.results.append(QualityGateResult(
                name="Linting (Flake8)",
                passed=passed,
                details=details,
                output=result.stdout + result.stderr
            ))
            
        except FileNotFoundError:
            self.results.append(QualityGateResult(
                name="Linting (Flake8)",
                passed=False,
                details="Flake8 not installed"
            ))
    
    def check_type_hints(self) -> None:
        """Check type hints with MyPy."""
        logger.info("Checking type hints...")
        
        try:
            result = subprocess.run([
                'mypy', 'src', '--ignore-missing-imports',
                '--strict-optional', '--warn-unused-ignores'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # MyPy returns 0 for success, >0 for issues
            passed = result.returncode == 0
            details = "Type hints are correct" if passed else "Type hint issues found"
            
            self.results.append(QualityGateResult(
                name="Type Hints (MyPy)",
                passed=passed,
                details=details,
                output=result.stdout + result.stderr
            ))
            
        except FileNotFoundError:
            self.results.append(QualityGateResult(
                name="Type Hints (MyPy)",
                passed=False,
                details="MyPy not installed"
            ))
    
    def check_test_coverage(self) -> None:
        """Check test coverage."""
        logger.info("Checking test coverage...")
        
        try:
            # Run tests with coverage
            result = subprocess.run([
                'pytest', 'tests/', '--cov=src', '--cov-report=json',
                '--cov-report=term-missing', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Try to read coverage report
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                threshold = self.config['code_quality']['test_coverage_threshold']
                
                passed = total_coverage >= threshold
                details = f"Coverage: {total_coverage:.1f}% (threshold: {threshold}%)"
                
                self.results.append(QualityGateResult(
                    name="Test Coverage",
                    passed=passed,
                    score=total_coverage,
                    threshold=threshold,
                    details=details,
                    output=result.stdout
                ))
            else:
                self.results.append(QualityGateResult(
                    name="Test Coverage",
                    passed=False,
                    details="Coverage report not generated"
                ))
            
        except FileNotFoundError:
            self.results.append(QualityGateResult(
                name="Test Coverage",
                passed=False,
                details="pytest not installed"
            ))
    
    def check_complexity(self) -> None:
        """Check code complexity."""
        logger.info("Checking code complexity...")
        
        try:
            result = subprocess.run([
                'radon', 'cc', 'src', '-j'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                try:
                    complexity_data = json.loads(result.stdout)
                    max_complexity = self.config['code_quality']['max_complexity']
                    
                    high_complexity_functions = []
                    for file_path, functions in complexity_data.items():
                        for func in functions:
                            if func['complexity'] > max_complexity:
                                high_complexity_functions.append(
                                    f"{file_path}:{func['name']} ({func['complexity']})"
                                )
                    
                    passed = len(high_complexity_functions) == 0
                    if passed:
                        details = f"All functions below complexity threshold ({max_complexity})"
                    else:
                        details = f"{len(high_complexity_functions)} functions exceed complexity threshold"
                    
                    self.results.append(QualityGateResult(
                        name="Code Complexity",
                        passed=passed,
                        threshold=float(max_complexity),
                        details=details,
                        output='\n'.join(high_complexity_functions)
                    ))
                    
                except json.JSONDecodeError:
                    self.results.append(QualityGateResult(
                        name="Code Complexity",
                        passed=False,
                        details="Failed to parse complexity report"
                    ))
            else:
                self.results.append(QualityGateResult(
                    name="Code Complexity",
                    passed=False,
                    details="Complexity analysis failed"
                ))
                
        except FileNotFoundError:
            logger.warning("Radon not installed, skipping complexity check")
    
    def check_security_vulnerabilities(self) -> None:
        """Check for security vulnerabilities."""
        logger.info("Checking security vulnerabilities...")
        
        # Check with Safety
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout.strip() else []
            else:
                # Safety returns non-zero when vulnerabilities are found
                try:
                    vulnerabilities = json.loads(result.stdout) if result.stdout.strip() else []
                except json.JSONDecodeError:
                    vulnerabilities = []
            
            high_vulns = [v for v in vulnerabilities if 'high' in v.get('severity', '').lower()]
            medium_vulns = [v for v in vulnerabilities if 'medium' in v.get('severity', '').lower()]
            
            max_high = self.config['security']['max_high_vulnerabilities']
            max_medium = self.config['security']['max_medium_vulnerabilities']
            
            passed = len(high_vulns) <= max_high and len(medium_vulns) <= max_medium
            details = f"High: {len(high_vulns)}/{max_high}, Medium: {len(medium_vulns)}/{max_medium}"
            
            self.results.append(QualityGateResult(
                name="Security Vulnerabilities",
                passed=passed,
                details=details,
                output=result.stdout
            ))
            
        except FileNotFoundError:
            logger.warning("Safety not installed, skipping vulnerability check")
    
    def check_dependency_licenses(self) -> None:
        """Check dependency licenses."""
        logger.info("Checking dependency licenses...")
        
        try:
            result = subprocess.run([
                'pip-licenses', '--format=json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                licenses = json.loads(result.stdout)
                allowed_licenses = self.config['dependencies']['allowed_licenses']
                
                problematic_licenses = []
                for package in licenses:
                    license_name = package.get('License', 'Unknown')
                    if license_name not in allowed_licenses and license_name != 'Unknown':
                        problematic_licenses.append(f"{package['Name']}: {license_name}")
                
                passed = len(problematic_licenses) == 0
                details = f"All licenses approved" if passed else f"{len(problematic_licenses)} packages with unapproved licenses"
                
                self.results.append(QualityGateResult(
                    name="Dependency Licenses",
                    passed=passed,
                    details=details,
                    output='\n'.join(problematic_licenses)
                ))
            else:
                self.results.append(QualityGateResult(
                    name="Dependency Licenses",
                    passed=False,
                    details="License check failed"
                ))
                
        except FileNotFoundError:
            logger.warning("pip-licenses not installed, skipping license check")
    
    def check_performance_tests(self) -> None:
        """Check performance tests."""
        logger.info("Checking performance tests...")
        
        try:
            result = subprocess.run([
                'pytest', 'tests/performance', '--benchmark-only',
                '--benchmark-json=benchmark-results.json', '-v'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            benchmark_file = self.project_root / 'benchmark-results.json'
            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                # Analyze benchmark results
                slow_tests = []
                for benchmark in benchmark_data.get('benchmarks', []):
                    mean_time = benchmark['stats']['mean']
                    if mean_time > self.config['performance']['max_test_duration_seconds']:
                        slow_tests.append(f"{benchmark['name']}: {mean_time:.2f}s")
                
                passed = len(slow_tests) == 0
                details = "All performance tests within limits" if passed else f"{len(slow_tests)} slow tests"
                
                self.results.append(QualityGateResult(
                    name="Performance Tests",
                    passed=passed,
                    details=details,
                    output='\n'.join(slow_tests)
                ))
            else:
                self.results.append(QualityGateResult(
                    name="Performance Tests",
                    passed=result.returncode == 0,
                    details="Performance tests completed" if result.returncode == 0 else "Performance tests failed"
                ))
                
        except FileNotFoundError:
            logger.warning("pytest not installed, skipping performance tests")
    
    def check_documentation_coverage(self) -> None:
        """Check documentation coverage."""
        logger.info("Checking documentation coverage...")
        
        try:
            # Use interrogate to check docstring coverage
            result = subprocess.run([
                'interrogate', 'src', '--generate-badge', '.', '--badge-format', 'json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # Try to parse the output for coverage percentage
                output_lines = result.stdout.split('\n')
                coverage_line = [line for line in output_lines if 'TOTAL' in line and '%' in line]
                
                if coverage_line:
                    # Extract percentage from output
                    import re
                    match = re.search(r'(\d+\.?\d*)%', coverage_line[0])
                    if match:
                        coverage = float(match.group(1))
                        threshold = self.config['documentation']['min_docstring_coverage']
                        
                        passed = coverage >= threshold
                        details = f"Documentation coverage: {coverage:.1f}% (threshold: {threshold}%)"
                        
                        self.results.append(QualityGateResult(
                            name="Documentation Coverage",
                            passed=passed,
                            score=coverage,
                            threshold=threshold,
                            details=details
                        ))
                        return
            
            # Fallback if interrogate output can't be parsed
            self.results.append(QualityGateResult(
                name="Documentation Coverage",
                passed=result.returncode == 0,
                details="Documentation check completed"
            ))
            
        except FileNotFoundError:
            logger.warning("interrogate not installed, skipping documentation check")
    
    def check_outdated_dependencies(self) -> None:
        """Check for outdated dependencies."""
        logger.info("Checking outdated dependencies...")
        
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout) if result.stdout.strip() else []
                max_outdated = self.config['dependencies']['max_outdated_packages']
                
                passed = len(outdated) <= max_outdated
                details = f"{len(outdated)} outdated packages (max: {max_outdated})"
                
                outdated_list = [f"{pkg['name']}: {pkg['version']} → {pkg['latest_version']}" 
                               for pkg in outdated]
                
                self.results.append(QualityGateResult(
                    name="Outdated Dependencies",
                    passed=passed,
                    details=details,
                    output='\n'.join(outdated_list)
                ))
            else:
                self.results.append(QualityGateResult(
                    name="Outdated Dependencies",
                    passed=False,
                    details="Failed to check outdated packages"
                ))
                
        except FileNotFoundError:
            self.results.append(QualityGateResult(
                name="Outdated Dependencies",
                passed=False,
                details="pip not available"
            ))
    
    def _print_results(self) -> None:
        """Print quality gate results."""
        logger.info("\n" + "="*80)
        logger.info("QUALITY GATE RESULTS")
        logger.info("="*80)
        
        for result in self.results:
            logger.info(str(result))
            if result.output and not result.passed:
                logger.info(f"  Details: {result.output[:200]}...")
        
        logger.info("="*80)
    
    def export_results(self, filepath: Path) -> None:
        """Export results to JSON file."""
        results_data = {
            'timestamp': os.times(),
            'project_root': str(self.project_root),
            'config': self.config,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'score': r.score,
                    'threshold': r.threshold,
                    'details': r.details,
                    'output': r.output
                }
                for r in self.results
            ],
            'summary': {
                'total_gates': len(self.results),
                'passed_gates': sum(1 for r in self.results if r.passed),
                'failed_gates': sum(1 for r in self.results if not r.passed),
                'overall_success': all(r.passed for r in self.results)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")


def main():
    """Main entry point for quality gates script."""
    parser = argparse.ArgumentParser(description="Run quality gates for protein operators")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument('--config', type=Path,
                       help="Configuration file path")
    parser.add_argument('--export-results', type=Path,
                       help="Export results to JSON file")
    parser.add_argument('--gate', choices=[
        'formatting', 'linting', 'types', 'coverage', 'complexity',
        'security', 'licenses', 'performance', 'documentation', 'dependencies'
    ], help="Run specific gate only")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = None
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize enforcer
    enforcer = QualityGateEnforcer(project_root=args.project_root, config=config)
    
    # Run specific gate or all gates
    if args.gate:
        gate_methods = {
            'formatting': enforcer.check_code_formatting,
            'linting': enforcer.check_linting,
            'types': enforcer.check_type_hints,
            'coverage': enforcer.check_test_coverage,
            'complexity': enforcer.check_complexity,
            'security': enforcer.check_security_vulnerabilities,
            'licenses': enforcer.check_dependency_licenses,
            'performance': enforcer.check_performance_tests,
            'documentation': enforcer.check_documentation_coverage,
            'dependencies': enforcer.check_outdated_dependencies
        }
        
        gate_methods[args.gate]()
        enforcer._print_results()
        success = all(r.passed for r in enforcer.results)
    else:
        success = enforcer.run_all_gates()
    
    # Export results if requested
    if args.export_results:
        enforcer.export_results(args.export_results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()