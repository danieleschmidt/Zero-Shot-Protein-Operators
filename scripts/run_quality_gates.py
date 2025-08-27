#!/usr/bin/env python3
"""
Comprehensive quality gates runner for the autonomous SDLC system.

Runs all quality checks including:
- Code quality analysis
- Security scanning
- Performance benchmarks
- Integration tests
- Documentation checks
- Deployment readiness
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil


class QualityGateStatus(Enum):
    """Status of quality gate checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    status: QualityGateStatus
    score: float
    details: Dict[str, Any]
    execution_time: float
    error_message: str = ""


class QualityGatesRunner:
    """Comprehensive quality gates runner."""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.results: List[QualityGateResult] = []
        
        # Quality gate thresholds
        self.thresholds = {
            'test_coverage': 85.0,
            'code_quality': 8.0,
            'security_score': 9.0,
            'performance_score': 7.0,
            'documentation_score': 8.0
        }
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        print("🚀 Starting Autonomous SDLC Quality Gates...")
        start_time = time.time()
        
        # Run all quality gates
        self._run_code_quality_gates()
        self._run_security_gates()
        self._run_test_gates()
        self._run_performance_gates()
        self._run_documentation_gates()
        self._run_deployment_readiness_gates()
        
        execution_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_report(execution_time)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _run_code_quality_gates(self):
        """Run code quality analysis."""
        print("\n📊 Running Code Quality Analysis...")
        
        # Run flake8
        flake8_result = self._run_flake8()
        self.results.append(flake8_result)
        
        # Run black formatting check
        black_result = self._run_black_check()
        self.results.append(black_result)
        
        # Run isort import sorting check
        isort_result = self._run_isort_check()
        self.results.append(isort_result)
        
        # Run mypy type checking
        mypy_result = self._run_mypy()
        self.results.append(mypy_result)
        
        # Run complexity analysis
        complexity_result = self._run_complexity_analysis()
        self.results.append(complexity_result)
    
    def _run_security_gates(self):
        """Run security analysis."""
        print("\n🔒 Running Security Analysis...")
        
        # Run bandit security linter
        bandit_result = self._run_bandit()
        self.results.append(bandit_result)
        
        # Run safety dependency check
        safety_result = self._run_safety_check()
        self.results.append(safety_result)
        
        # Check for secrets
        secrets_result = self._check_secrets()
        self.results.append(secrets_result)
    
    def _run_test_gates(self):
        """Run testing gates."""
        print("\n🧪 Running Test Suite...")
        
        # Run unit tests
        unit_test_result = self._run_unit_tests()
        self.results.append(unit_test_result)
        
        # Run integration tests
        integration_test_result = self._run_integration_tests()
        self.results.append(integration_test_result)
        
        # Check test coverage
        coverage_result = self._check_test_coverage()
        self.results.append(coverage_result)
    
    def _run_performance_gates(self):
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks...")
        
        # Run performance tests
        performance_result = self._run_performance_tests()
        self.results.append(performance_result)
        
        # Memory usage analysis
        memory_result = self._analyze_memory_usage()
        self.results.append(memory_result)
        
        # Load testing
        load_test_result = self._run_load_tests()
        self.results.append(load_test_result)
    
    def _run_documentation_gates(self):
        """Run documentation checks."""
        print("\n📚 Running Documentation Checks...")
        
        # Check docstring coverage
        docstring_result = self._check_docstring_coverage()
        self.results.append(docstring_result)
        
        # Validate README
        readme_result = self._validate_readme()
        self.results.append(readme_result)
        
        # Check API documentation
        api_docs_result = self._check_api_documentation()
        self.results.append(api_docs_result)
    
    def _run_deployment_readiness_gates(self):
        """Run deployment readiness checks."""
        print("\n🚢 Running Deployment Readiness Checks...")
        
        # Docker build test
        docker_result = self._test_docker_build()
        self.results.append(docker_result)
        
        # Configuration validation
        config_result = self._validate_configuration()
        self.results.append(config_result)
        
        # Dependency audit
        dependency_result = self._audit_dependencies()
        self.results.append(dependency_result)
    
    def _run_flake8(self) -> QualityGateResult:
        """Run flake8 linting."""
        start_time = time.time()
        
        try:
            cmd = ["flake8", str(self.project_root / "src"), "--max-line-length=88", "--extend-ignore=E203,W503"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
                details = {"issues": 0, "output": "No issues found"}
            else:
                issues = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                status = QualityGateStatus.WARNING if issues < 10 else QualityGateStatus.FAILED
                score = max(0, 10.0 - issues * 0.5)
                details = {"issues": issues, "output": result.stdout}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "flake8 not installed"}
        
        return QualityGateResult(
            name="flake8_linting",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_black_check(self) -> QualityGateResult:
        """Run black formatting check."""
        start_time = time.time()
        
        try:
            cmd = ["black", "--check", "--diff", str(self.project_root / "src")]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
                details = {"formatted": True, "output": "Code is properly formatted"}
            else:
                status = QualityGateStatus.WARNING
                score = 7.0
                details = {"formatted": False, "output": result.stdout}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "black not installed"}
        
        return QualityGateResult(
            name="black_formatting",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_isort_check(self) -> QualityGateResult:
        """Run isort import sorting check."""
        start_time = time.time()
        
        try:
            cmd = ["isort", "--check-only", "--diff", str(self.project_root / "src")]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
                details = {"sorted": True, "output": "Imports are properly sorted"}
            else:
                status = QualityGateStatus.WARNING
                score = 7.0
                details = {"sorted": False, "output": result.stdout}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "isort not installed"}
        
        return QualityGateResult(
            name="isort_imports",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_mypy(self) -> QualityGateResult:
        """Run mypy type checking."""
        start_time = time.time()
        
        try:
            cmd = ["mypy", str(self.project_root / "src"), "--ignore-missing-imports"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse mypy output for errors
            errors = [line for line in result.stdout.split('\n') if 'error:' in line]
            error_count = len(errors)
            
            if error_count == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
            elif error_count < 5:
                status = QualityGateStatus.WARNING
                score = 8.0
            else:
                status = QualityGateStatus.FAILED
                score = max(0, 10.0 - error_count * 0.5)
            
            details = {
                "type_errors": error_count,
                "output": result.stdout[:1000]  # Truncate output
            }
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "mypy not installed"}
        
        return QualityGateResult(
            name="mypy_typing",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_complexity_analysis(self) -> QualityGateResult:
        """Run code complexity analysis."""
        start_time = time.time()
        
        try:
            # Use radon for complexity analysis
            cmd = ["radon", "cc", str(self.project_root / "src"), "-a", "-nc"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse average complexity from output
                lines = result.stdout.strip().split('\n')
                avg_complexity = 1.0  # Default
                
                for line in lines:
                    if 'Average complexity:' in line:
                        try:
                            avg_complexity = float(line.split(':')[1].strip().split()[0])
                        except (IndexError, ValueError):
                            pass
                
                if avg_complexity <= 3.0:
                    status = QualityGateStatus.PASSED
                    score = 10.0
                elif avg_complexity <= 5.0:
                    status = QualityGateStatus.WARNING
                    score = 7.0
                else:
                    status = QualityGateStatus.FAILED
                    score = max(0, 10.0 - avg_complexity)
                
                details = {
                    "average_complexity": avg_complexity,
                    "output": result.stdout[:500]
                }
            else:
                status = QualityGateStatus.FAILED
                score = 0.0
                details = {"error": result.stderr}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "radon not installed"}
        
        return QualityGateResult(
            name="complexity_analysis",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_bandit(self) -> QualityGateResult:
        """Run bandit security analysis."""
        start_time = time.time()
        
        try:
            cmd = ["bandit", "-r", str(self.project_root / "src"), "-f", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            try:
                bandit_data = json.loads(result.stdout)
                high_issues = len([issue for issue in bandit_data.get('results', []) if issue['issue_severity'] == 'HIGH'])
                medium_issues = len([issue for issue in bandit_data.get('results', []) if issue['issue_severity'] == 'MEDIUM'])
                total_issues = len(bandit_data.get('results', []))
                
                if total_issues == 0:
                    status = QualityGateStatus.PASSED
                    score = 10.0
                elif high_issues == 0 and medium_issues < 3:
                    status = QualityGateStatus.WARNING
                    score = 8.0
                else:
                    status = QualityGateStatus.FAILED
                    score = max(0, 10.0 - high_issues * 2 - medium_issues * 0.5)
                
                details = {
                    "high_severity": high_issues,
                    "medium_severity": medium_issues,
                    "total_issues": total_issues
                }
                
            except json.JSONDecodeError:
                status = QualityGateStatus.WARNING
                score = 5.0
                details = {"error": "Could not parse bandit output"}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "bandit not installed"}
        
        return QualityGateResult(
            name="bandit_security",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_safety_check(self) -> QualityGateResult:
        """Run safety dependency vulnerability check."""
        start_time = time.time()
        
        try:
            cmd = ["safety", "check", "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            try:
                safety_data = json.loads(result.stdout)
                vulnerabilities = len(safety_data)
                
                if vulnerabilities == 0:
                    status = QualityGateStatus.PASSED
                    score = 10.0
                elif vulnerabilities < 3:
                    status = QualityGateStatus.WARNING
                    score = 7.0
                else:
                    status = QualityGateStatus.FAILED
                    score = max(0, 10.0 - vulnerabilities)
                
                details = {
                    "vulnerabilities": vulnerabilities,
                    "issues": safety_data[:5]  # First 5 issues
                }
                
            except json.JSONDecodeError:
                # safety might return non-JSON output for no vulnerabilities
                if "No known security vulnerabilities" in result.stdout:
                    status = QualityGateStatus.PASSED
                    score = 10.0
                    details = {"vulnerabilities": 0}
                else:
                    status = QualityGateStatus.WARNING
                    score = 5.0
                    details = {"error": "Could not parse safety output"}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "safety not installed"}
        
        return QualityGateResult(
            name="safety_dependencies",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _check_secrets(self) -> QualityGateResult:
        """Check for hardcoded secrets."""
        start_time = time.time()
        
        # Simple secret patterns
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']*["\']',
            r'api_key\s*=\s*["\'][^"\']*["\']',
            r'secret\s*=\s*["\'][^"\']*["\']',
            r'token\s*=\s*["\'][^"\']*["\']',
        ]
        
        try:
            import re
            secrets_found = 0
            
            for python_file in (self.project_root / "src").rglob("*.py"):
                try:
                    with open(python_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            secrets_found += 1
                except Exception:
                    continue
            
            if secrets_found == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
            elif secrets_found < 3:
                status = QualityGateStatus.WARNING
                score = 6.0
            else:
                status = QualityGateStatus.FAILED
                score = 0.0
            
            details = {"potential_secrets": secrets_found}
            
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 5.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="secrets_check",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_unit_tests(self) -> QualityGateResult:
        """Run unit tests."""
        start_time = time.time()
        
        try:
            cmd = ["python", "-m", "pytest", str(self.project_root / "tests" / "unit"), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            test_summary = [line for line in output_lines if 'passed' in line and 'failed' in line]
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
            else:
                status = QualityGateStatus.FAILED
                score = 5.0
            
            details = {
                "exit_code": result.returncode,
                "summary": test_summary[-1] if test_summary else "No summary available",
                "output": result.stdout[-1000:]  # Last 1000 chars
            }
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except Exception as e:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="unit_tests",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_integration_tests(self) -> QualityGateResult:
        """Run integration tests."""
        start_time = time.time()
        
        try:
            cmd = ["python", "-m", "pytest", str(self.project_root / "tests" / "integration"), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 minutes
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            test_summary = [line for line in output_lines if 'passed' in line and ('failed' in line or 'error' in line)]
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
            else:
                # Check if there are any passed tests
                if 'passed' in result.stdout:
                    status = QualityGateStatus.WARNING
                    score = 6.0
                else:
                    status = QualityGateStatus.FAILED
                    score = 2.0
            
            details = {
                "exit_code": result.returncode,
                "summary": test_summary[-1] if test_summary else "No summary available",
                "output": result.stdout[-1000:]  # Last 1000 chars
            }
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout (20 minutes)"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 3.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="integration_tests",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _check_test_coverage(self) -> QualityGateResult:
        """Check test coverage."""
        start_time = time.time()
        
        try:
            cmd = ["python", "-m", "pytest", "--cov=src", "--cov-report=term-missing", "--cov-report=json", str(self.project_root / "tests")]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            
            # Try to read coverage report
            coverage_file = self.project_root / "coverage.json"
            coverage_percentage = 0.0
            
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    coverage_percentage = coverage_data.get('totals', {}).get('percent_covered', 0.0)
                except Exception:
                    pass
            
            # Parse coverage from stdout if json file not available
            if coverage_percentage == 0.0:
                for line in result.stdout.split('\n'):
                    if 'TOTAL' in line and '%' in line:
                        try:
                            coverage_percentage = float(line.split()[-1].replace('%', ''))
                        except (IndexError, ValueError):
                            pass
            
            if coverage_percentage >= self.thresholds['test_coverage']:
                status = QualityGateStatus.PASSED
                score = 10.0
            elif coverage_percentage >= 70.0:
                status = QualityGateStatus.WARNING
                score = 7.0
            else:
                status = QualityGateStatus.FAILED
                score = max(0, coverage_percentage / 10)
            
            details = {
                "coverage_percentage": coverage_percentage,
                "threshold": self.thresholds['test_coverage']
            }
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 5.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="test_coverage",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_performance_tests(self) -> QualityGateResult:
        """Run performance tests."""
        start_time = time.time()
        
        try:
            cmd = ["python", "-m", "pytest", str(self.project_root / "tests"), "-k", "performance", "-v"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
            elif 'passed' in result.stdout:
                status = QualityGateStatus.WARNING
                score = 7.0
            else:
                status = QualityGateStatus.FAILED
                score = 3.0
            
            details = {
                "exit_code": result.returncode,
                "output": result.stdout[-500:]  # Last 500 chars
            }
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 5.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="performance_tests",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _analyze_memory_usage(self) -> QualityGateResult:
        """Analyze memory usage patterns."""
        start_time = time.time()
        
        try:
            # Run a simple memory analysis
            cmd = ["python", "-c", """
import sys
sys.path.insert(0, 'src')
import psutil
import time

# Get initial memory
initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

# Import main modules
try:
    from protein_operators.core import ProteinDesigner
    from protein_operators.models.enhanced_deeponet import EnhancedProteinDeepONet
    import_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Create objects
    model = EnhancedProteinDeepONet()
    designer = ProteinDesigner()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f'Initial: {initial_memory:.1f}MB')
    print(f'After imports: {import_memory:.1f}MB')
    print(f'After objects: {final_memory:.1f}MB')
    print(f'Total increase: {final_memory - initial_memory:.1f}MB')
    
except Exception as e:
    print(f'Error: {e}')
"""]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and 'Total increase:' in result.stdout:
                # Parse memory usage
                lines = result.stdout.strip().split('\n')
                memory_increase = 0.0
                
                for line in lines:
                    if 'Total increase:' in line:
                        try:
                            memory_increase = float(line.split(':')[1].replace('MB', '').strip())
                        except (IndexError, ValueError):
                            pass
                
                if memory_increase < 500:  # Less than 500MB
                    status = QualityGateStatus.PASSED
                    score = 10.0
                elif memory_increase < 1000:  # Less than 1GB
                    status = QualityGateStatus.WARNING
                    score = 7.0
                else:
                    status = QualityGateStatus.FAILED
                    score = 3.0
                
                details = {
                    "memory_increase_mb": memory_increase,
                    "output": result.stdout
                }
            else:
                status = QualityGateStatus.WARNING
                score = 5.0
                details = {"error": "Could not analyze memory usage"}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 5.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="memory_analysis",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _run_load_tests(self) -> QualityGateResult:
        """Run load tests."""
        start_time = time.time()
        
        # Simplified load test
        try:
            cmd = ["python", "-c", """
import sys
sys.path.insert(0, 'src')
import time
import concurrent.futures

def load_test_function():
    try:
        from protein_operators.core import ProteinDesigner
        designer = ProteinDesigner()
        return True
    except Exception:
        return False

# Run concurrent load test
start_time = time.time()
success_count = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_test_function) for _ in range(10)]
    for future in concurrent.futures.as_completed(futures):
        if future.result():
            success_count += 1

end_time = time.time()
print(f'Success rate: {success_count}/10')
print(f'Duration: {end_time - start_time:.2f}s')
"""]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                success_rate = 0
                duration = 0.0
                
                for line in lines:
                    if 'Success rate:' in line:
                        try:
                            success_rate = int(line.split('/')[0].split(':')[1].strip())
                        except (IndexError, ValueError):
                            pass
                    elif 'Duration:' in line:
                        try:
                            duration = float(line.split(':')[1].replace('s', '').strip())
                        except (IndexError, ValueError):
                            pass
                
                if success_rate >= 9 and duration < 30:
                    status = QualityGateStatus.PASSED
                    score = 10.0
                elif success_rate >= 7:
                    status = QualityGateStatus.WARNING
                    score = 7.0
                else:
                    status = QualityGateStatus.FAILED
                    score = 3.0
                
                details = {
                    "success_rate": f"{success_rate}/10",
                    "duration_seconds": duration
                }
            else:
                status = QualityGateStatus.WARNING
                score = 5.0
                details = {"error": "Load test failed"}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 5.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="load_tests",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _check_docstring_coverage(self) -> QualityGateResult:
        """Check docstring coverage."""
        start_time = time.time()
        
        try:
            cmd = ["docstr-coverage", str(self.project_root / "src"), "--percentage-only"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                try:
                    coverage_percentage = float(result.stdout.strip().replace('%', ''))
                    
                    if coverage_percentage >= 80:
                        status = QualityGateStatus.PASSED
                        score = 10.0
                    elif coverage_percentage >= 60:
                        status = QualityGateStatus.WARNING
                        score = 7.0
                    else:
                        status = QualityGateStatus.FAILED
                        score = max(0, coverage_percentage / 10)
                    
                    details = {"docstring_coverage": coverage_percentage}
                    
                except ValueError:
                    status = QualityGateStatus.WARNING
                    score = 5.0
                    details = {"error": "Could not parse docstring coverage"}
            else:
                status = QualityGateStatus.WARNING
                score = 5.0
                details = {"error": result.stderr}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "docstr-coverage not installed"}
        
        return QualityGateResult(
            name="docstring_coverage",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _validate_readme(self) -> QualityGateResult:
        """Validate README file."""
        start_time = time.time()
        
        readme_file = self.project_root / "README.md"
        
        if not readme_file.exists():
            return QualityGateResult(
                name="readme_validation",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": "README.md not found"},
                execution_time=time.time() - start_time
            )
        
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for essential sections
            required_sections = [
                "# ",  # Title
                "## Overview",
                "## Installation",
                "## Quick Start",
                "## Examples"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            # Check length
            word_count = len(content.split())
            
            if len(missing_sections) == 0 and word_count > 500:
                status = QualityGateStatus.PASSED
                score = 10.0
            elif len(missing_sections) <= 2 and word_count > 200:
                status = QualityGateStatus.WARNING
                score = 7.0
            else:
                status = QualityGateStatus.FAILED
                score = 3.0
            
            details = {
                "word_count": word_count,
                "missing_sections": missing_sections
            }
            
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 3.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="readme_validation",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _check_api_documentation(self) -> QualityGateResult:
        """Check API documentation completeness."""
        start_time = time.time()
        
        # Simple check for documentation files
        docs_files = list((self.project_root / "docs").glob("*.md")) if (self.project_root / "docs").exists() else []
        api_docs = [f for f in docs_files if "api" in f.name.lower()]
        
        if len(api_docs) > 0:
            status = QualityGateStatus.PASSED
            score = 10.0
        elif len(docs_files) > 0:
            status = QualityGateStatus.WARNING
            score = 6.0
        else:
            status = QualityGateStatus.FAILED
            score = 2.0
        
        details = {
            "total_docs": len(docs_files),
            "api_docs": len(api_docs)
        }
        
        return QualityGateResult(
            name="api_documentation",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _test_docker_build(self) -> QualityGateResult:
        """Test Docker build."""
        start_time = time.time()
        
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            return QualityGateResult(
                name="docker_build",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details={"error": "Dockerfile not found"},
                execution_time=time.time() - start_time
            )
        
        try:
            # Test dockerfile syntax
            cmd = ["docker", "build", "--dry-run", "-f", str(dockerfile), str(self.project_root)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                score = 10.0
                details = {"build_test": "success"}
            else:
                status = QualityGateStatus.FAILED
                score = 2.0
                details = {"error": result.stderr[:500]}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except FileNotFoundError:
            status = QualityGateStatus.SKIPPED
            score = 5.0
            details = {"error": "Docker not available"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 3.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="docker_build",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _validate_configuration(self) -> QualityGateResult:
        """Validate configuration files."""
        start_time = time.time()
        
        config_files = [
            "pyproject.toml",
            "environment.yml",
            "docker-compose.yml"
        ]
        
        existing_configs = 0
        valid_configs = 0
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                existing_configs += 1
                
                try:
                    # Basic validation
                    if config_file.endswith('.toml'):
                        import toml
                        with open(config_path, 'r') as f:
                            toml.load(f)
                        valid_configs += 1
                    elif config_file.endswith('.yml') or config_file.endswith('.yaml'):
                        import yaml
                        with open(config_path, 'r') as f:
                            yaml.safe_load(f)
                        valid_configs += 1
                    else:
                        valid_configs += 1  # Assume valid for other types
                        
                except Exception:
                    pass  # Invalid config
        
        if valid_configs == existing_configs and existing_configs >= 2:
            status = QualityGateStatus.PASSED
            score = 10.0
        elif valid_configs > 0:
            status = QualityGateStatus.WARNING
            score = 6.0
        else:
            status = QualityGateStatus.FAILED
            score = 2.0
        
        details = {
            "existing_configs": existing_configs,
            "valid_configs": valid_configs,
            "total_checked": len(config_files)
        }
        
        return QualityGateResult(
            name="configuration_validation",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _audit_dependencies(self) -> QualityGateResult:
        """Audit project dependencies."""
        start_time = time.time()
        
        try:
            cmd = ["pip", "list", "--outdated", "--format=json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                try:
                    outdated_packages = json.loads(result.stdout)
                    outdated_count = len(outdated_packages)
                    
                    if outdated_count == 0:
                        status = QualityGateStatus.PASSED
                        score = 10.0
                    elif outdated_count < 5:
                        status = QualityGateStatus.WARNING
                        score = 7.0
                    else:
                        status = QualityGateStatus.FAILED
                        score = max(0, 10.0 - outdated_count * 0.5)
                    
                    details = {
                        "outdated_packages": outdated_count,
                        "packages": [pkg['name'] for pkg in outdated_packages[:5]]  # First 5
                    }
                    
                except json.JSONDecodeError:
                    status = QualityGateStatus.WARNING
                    score = 5.0
                    details = {"error": "Could not parse pip output"}
            else:
                status = QualityGateStatus.WARNING
                score = 5.0
                details = {"error": result.stderr}
            
        except subprocess.TimeoutExpired:
            status = QualityGateStatus.FAILED
            score = 0.0
            details = {"error": "Timeout"}
        except Exception as e:
            status = QualityGateStatus.WARNING
            score = 5.0
            details = {"error": str(e)}
        
        return QualityGateResult(
            name="dependency_audit",
            status=status,
            score=score,
            details=details,
            execution_time=time.time() - start_time
        )
    
    def _generate_report(self, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality gates report."""
        # Calculate overall scores
        total_score = sum(result.score for result in self.results)
        max_possible_score = len(self.results) * 10.0
        overall_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        # Count status distribution
        status_counts = {
            QualityGateStatus.PASSED: len([r for r in self.results if r.status == QualityGateStatus.PASSED]),
            QualityGateStatus.WARNING: len([r for r in self.results if r.status == QualityGateStatus.WARNING]),
            QualityGateStatus.FAILED: len([r for r in self.results if r.status == QualityGateStatus.FAILED]),
            QualityGateStatus.SKIPPED: len([r for r in self.results if r.status == QualityGateStatus.SKIPPED])
        }
        
        # Determine overall status
        if status_counts[QualityGateStatus.FAILED] > 0:
            overall_status = QualityGateStatus.FAILED
        elif status_counts[QualityGateStatus.WARNING] > 2:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.PASSED
        
        # Group results by category
        categories = {
            "Code Quality": ["flake8_linting", "black_formatting", "isort_imports", "mypy_typing", "complexity_analysis"],
            "Security": ["bandit_security", "safety_dependencies", "secrets_check"],
            "Testing": ["unit_tests", "integration_tests", "test_coverage"],
            "Performance": ["performance_tests", "memory_analysis", "load_tests"],
            "Documentation": ["docstring_coverage", "readme_validation", "api_documentation"],
            "Deployment": ["docker_build", "configuration_validation", "dependency_audit"]
        }
        
        category_scores = {}
        for category, gate_names in categories.items():
            category_results = [r for r in self.results if r.name in gate_names]
            if category_results:
                category_score = sum(r.score for r in category_results) / len(category_results)
                category_scores[category] = {
                    "score": category_score,
                    "percentage": (category_score / 10.0) * 100,
                    "gates": len(category_results)
                }
        
        return {
            "overall": {
                "status": overall_status.value,
                "score": total_score,
                "percentage": overall_percentage,
                "max_possible_score": max_possible_score,
                "execution_time_seconds": total_execution_time
            },
            "summary": {
                "total_gates": len(self.results),
                "passed": status_counts[QualityGateStatus.PASSED],
                "warnings": status_counts[QualityGateStatus.WARNING],
                "failed": status_counts[QualityGateStatus.FAILED],
                "skipped": status_counts[QualityGateStatus.SKIPPED]
            },
            "categories": category_scores,
            "detailed_results": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "score": result.score,
                    "percentage": (result.score / 10.0) * 100,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "error_message": result.error_message
                }
                for result in self.results
            ],
            "thresholds": self.thresholds,
            "timestamp": time.time()
        }
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print quality gates summary."""
        overall = report["overall"]
        summary = report["summary"]
        categories = report["categories"]
        
        print("\n" + "=" * 80)
        print("🎯 AUTONOMOUS SDLC QUALITY GATES REPORT")
        print("=" * 80)
        
        # Overall status
        status_emoji = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌"
        }
        
        print(f"\n📊 OVERALL STATUS: {status_emoji.get(overall['status'], '❓')} {overall['status'].upper()}")
        print(f"📈 OVERALL SCORE: {overall['score']:.1f}/{overall['max_possible_score']:.1f} ({overall['percentage']:.1f}%)")
        print(f"⏱️  EXECUTION TIME: {overall['execution_time_seconds']:.1f} seconds")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ⚠️  Warnings: {summary['warnings']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⏭️  Skipped: {summary['skipped']}")
        print(f"   📊 Total Gates: {summary['total_gates']}")
        
        # Category breakdown
        print(f"\n🏷️  CATEGORY BREAKDOWN:")
        for category, data in categories.items():
            status_icon = "✅" if data['percentage'] >= 80 else "⚠️" if data['percentage'] >= 60 else "❌"
            print(f"   {status_icon} {category}: {data['percentage']:.1f}% ({data['score']:.1f}/10.0)")
        
        # Failed gates
        failed_gates = [r for r in report["detailed_results"] if r["status"] == "failed"]
        if failed_gates:
            print(f"\n❌ FAILED GATES:")
            for gate in failed_gates:
                print(f"   • {gate['name']}: {gate['score']:.1f}/10.0")
                if gate.get('error_message'):
                    print(f"     Error: {gate['error_message']}")
        
        # Warning gates
        warning_gates = [r for r in report["detailed_results"] if r["status"] == "warning"]
        if warning_gates:
            print(f"\n⚠️  WARNING GATES:")
            for gate in warning_gates:
                print(f"   • {gate['name']}: {gate['score']:.1f}/10.0")
        
        print("\n" + "=" * 80)
        
        if overall['status'] == 'passed':
            print("🎉 Congratulations! All quality gates passed successfully.")
        elif overall['status'] == 'warning':
            print("⚠️  Some quality gates have warnings. Consider addressing them.")
        else:
            print("❌ Quality gates failed. Please address the issues before deployment.")
        
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run autonomous SDLC quality gates")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=Path, help="Output file for report (JSON)")
    parser.add_argument("--fail-on-warning", action="store_true", help="Fail if any warnings")
    
    args = parser.parse_args()
    
    # Initialize and run quality gates
    runner = QualityGatesRunner(args.project_root, args.verbose)
    report = runner.run_all_gates()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Determine exit code
    overall_status = report["overall"]["status"]
    
    if overall_status == "failed":
        sys.exit(1)
    elif overall_status == "warning" and args.fail_on_warning:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
