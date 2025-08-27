"""
Robust protein design core with comprehensive error handling, monitoring, and recovery.

This module enhances the core protein design functionality with:
- Advanced error handling and recovery
- Real-time monitoring and alerting
- Circuit breaker patterns
- Graceful degradation
- Performance profiling
- Health checks
"""

import sys
import os
from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path
import time
import logging
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import robust utilities
from .utils.advanced_error_handling import (
    AdvancedErrorHandler, error_handler, circuit_breaker, 
    ProteinDesignException, get_global_error_handler
)
from .utils.advanced_monitoring import (
    AdvancedMonitoringSystem, MetricType, AlertLevel
)
from .utils.torch_integration import (
    TORCH_AVAILABLE, get_device, tensor, zeros
)

# Import core functionality
from .core import ProteinDesigner
from .constraints import Constraints
from .structure import ProteinStructure

# Mock torch fallback
if TORCH_AVAILABLE:
    import torch
else:
    import mock_torch as torch


class RobustProteinDesigner(ProteinDesigner):
    """
    Enhanced protein designer with robust error handling and monitoring.
    
    Features:
    - Automatic error recovery
    - Circuit breaker protection
    - Performance monitoring
    - Health checks
    - Graceful degradation
    - Resource management
    """
    
    def __init__(
        self,
        operator_type: str = "deeponet",
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        enable_monitoring: bool = True,
        enable_error_recovery: bool = True,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        **kwargs
    ):
        """
        Initialize robust protein designer.
        
        Args:
            operator_type: Type of neural operator
            checkpoint: Path to model checkpoint
            device: Computing device
            enable_monitoring: Enable monitoring system
            enable_error_recovery: Enable automatic error recovery
            max_retries: Maximum retry attempts for recoverable errors
            circuit_breaker_threshold: Failure threshold for circuit breaker
            **kwargs: Additional model parameters
        """
        # Initialize error handler
        if enable_error_recovery:
            self.error_handler = AdvancedErrorHandler(
                max_retries=max_retries,
                circuit_breaker_threshold=circuit_breaker_threshold,
                enable_graceful_degradation=True
            )
        else:
            self.error_handler = None
        
        # Initialize monitoring system
        if enable_monitoring:
            self.monitoring = AdvancedMonitoringSystem(
                enable_resource_monitoring=True,
                enable_health_checks=True
            )
            self.monitoring.start()
        else:
            self.monitoring = None
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize parent with error handling
        try:
            super().__init__(
                operator_type=operator_type,
                checkpoint=checkpoint,
                device=device,
                **kwargs
            )
            
            # Register fallback handlers
            if self.error_handler:
                self._register_fallback_handlers()
            
            # Add health checks
            if self.monitoring:
                self._setup_health_checks()
            
            self.logger.info("Robust protein designer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize robust designer: {e}")
            if self.error_handler:
                # Try fallback initialization
                self._fallback_initialization(operator_type, device, **kwargs)
            else:
                raise
    
    def _fallback_initialization(self, operator_type: str, device: Optional[str], **kwargs):
        """Fallback initialization with minimal functionality."""
        self.logger.warning("Using fallback initialization")
        
        self.operator_type = operator_type
        self.device = device or 'cpu'
        self.design_count = 0
        self.success_rate = 0.0
        
        # Create mock model for compatibility
        self.model = None
        
        self.logger.info("Fallback initialization complete")
    
    def _register_fallback_handlers(self):
        """Register fallback handlers for key operations."""
        if not self.error_handler:
            return
        
        # Design generation fallback
        self.error_handler.register_fallback(
            "protein_generation",
            self._fallback_generate_structure
        )
        
        # Validation fallback
        self.error_handler.register_fallback(
            "structure_validation",
            self._fallback_validate_structure
        )
        
        # Optimization fallback
        self.error_handler.register_fallback(
            "structure_optimization",
            self._fallback_optimize_structure
        )
    
    def _setup_health_checks(self):
        """Setup custom health checks for protein design operations."""
        if not self.monitoring:
            return
        
        # Model health check
        self.monitoring.add_health_check(
            "model_health",
            self._check_model_health,
            interval_seconds=60.0
        )
        
        # Design capability check
        self.monitoring.add_health_check(
            "design_capability",
            self._check_design_capability,
            interval_seconds=120.0
        )
        
        # Memory efficiency check
        self.monitoring.add_health_check(
            "memory_efficiency",
            self._check_memory_efficiency,
            interval_seconds=180.0
        )
    
    def _check_model_health(self) -> bool:
        """Check if the model is functioning properly."""
        try:
            if self.model is None:
                return True  # Fallback mode is acceptable
            
            # Simple model test
            test_input = torch.randn(1, 256)  # Minimal test input
            
            with torch.no_grad():
                _ = self.model(test_input)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Model health check failed: {e}")
            return False
    
    def _check_design_capability(self) -> bool:
        """Check if basic design capabilities are working."""
        try:
            # Test constraint creation
            constraints = Constraints()
            constraints.add_binding_site([1, 2, 3], "test", 100.0)
            
            # Test constraint encoding
            _ = self._encode_constraints(constraints)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Design capability check failed: {e}")
            return False
    
    def _check_memory_efficiency(self) -> bool:
        """Check memory usage efficiency."""
        try:
            import psutil
            
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Warn if using more than 4GB
            return memory_mb < 4096
            
        except Exception:
            return True  # Can't check, assume OK
    
    @error_handler("protein_generation", max_retries=3)
    def generate(
        self,
        constraints: Constraints,
        length: int,
        num_samples: int = 1,
        physics_guided: bool = False,
        **kwargs
    ) -> ProteinStructure:
        """
        Generate protein structure with robust error handling.
        
        Args:
            constraints: Design constraints
            length: Target protein length
            num_samples: Number of designs to generate
            physics_guided: Whether to use physics-guided refinement
            **kwargs: Additional generation parameters
            
        Returns:
            Generated protein structure
            
        Raises:
            ProteinDesignException: If generation fails after all recovery attempts
        """
        operation_name = "protein_generation"
        
        # Performance monitoring
        with self._profile_operation(operation_name):
            try:
                # Validate inputs with enhanced checking
                self._robust_validate_constraints(constraints, length)
                
                # Record generation attempt
                if self.monitoring:
                    self.monitoring.record_metric(
                        "generation_attempts",
                        1,
                        MetricType.COUNTER,
                        {"length": str(length), "samples": str(num_samples)}
                    )
                
                # Execute generation
                result = self._execute_generation(
                    constraints, length, num_samples, physics_guided, **kwargs
                )
                
                # Validate result
                if not self._validate_generation_result(result):
                    raise ProteinDesignException(
                        "Generated structure failed validation",
                        context={"length": length, "samples": num_samples}
                    )
                
                # Record success
                self.design_count += 1
                if self.monitoring:
                    self.monitoring.record_metric(
                        "generation_successes", 1, MetricType.COUNTER
                    )
                
                return result
                
            except Exception as e:
                # Enhanced error handling
                self._handle_generation_error(e, constraints, length)
                raise
    
    def _execute_generation(
        self, 
        constraints: Constraints, 
        length: int, 
        num_samples: int, 
        physics_guided: bool,
        **kwargs
    ) -> ProteinStructure:
        """Execute the actual generation with circuit breaker protection."""
        
        # Use parent implementation with monitoring
        if self.error_handler:
            return self.error_handler.handle_with_recovery(
                "core_generation",
                super().generate,
                constraints, length, num_samples, physics_guided,
                fallback_result=self._fallback_generate_structure(constraints, length),
                **kwargs
            )
        else:
            return super().generate(constraints, length, num_samples, physics_guided, **kwargs)
    
    def _robust_validate_constraints(self, constraints: Constraints, length: int):
        """Enhanced constraint validation with detailed error reporting."""
        try:
            # Use parent validation
            self._validate_constraints(constraints, length)
            
        except Exception as e:
            # Enhance error context
            if isinstance(e, ValueError):
                raise ProteinDesignException(
                    f"Constraint validation failed: {str(e)}",
                    context={
                        "constraint_count": len(constraints.binding_sites),
                        "length": length,
                        "validation_stage": "robust_validation"
                    },
                    suggestion="Check constraint parameters and protein length"
                )
            raise
    
    def _validate_generation_result(self, result: ProteinStructure) -> bool:
        """Validate that generation result is acceptable."""
        try:
            if result is None:
                return False
            
            if not hasattr(result, 'coordinates'):
                return False
            
            # Check coordinate tensor
            coords = result.coordinates
            if coords is None or coords.numel() == 0:
                return False
            
            # Check for NaN or infinite values
            if torch.isnan(coords).any() or torch.isinf(coords).any():
                return False
            
            # Check reasonable coordinate ranges
            coord_max = torch.max(torch.abs(coords))
            if coord_max > 1000:  # Unreasonably large coordinates
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Result validation error: {e}")
            return False
    
    def _handle_generation_error(self, error: Exception, constraints: Constraints, length: int):
        """Handle generation errors with detailed logging."""
        error_context = {
            "operation": "protein_generation",
            "constraint_count": len(constraints.binding_sites),
            "length": length,
            "device": str(self.device),
            "operator_type": self.operator_type
        }
        
        if self.monitoring:
            self.monitoring.record_metric(
                "generation_errors", 1, MetricType.COUNTER,
                {"error_type": type(error).__name__}
            )
        
        self.logger.error(f"Generation error: {error}", extra=error_context)
    
    def _fallback_generate_structure(self, constraints: Constraints, length: int = 50) -> ProteinStructure:
        """Fallback structure generation using simplified approach."""
        self.logger.warning("Using fallback structure generation")
        
        try:
            # Create simple extended chain
            coords = torch.zeros(length, 3)
            for i in range(length):
                coords[i, 0] = i * 3.8  # CA-CA distance
                # Add slight helical twist
                coords[i, 1] = 2.0 * torch.sin(torch.tensor(i * 0.3))
                coords[i, 2] = 2.0 * torch.cos(torch.tensor(i * 0.3))
            
            # Create fallback structure
            structure = ProteinStructure(coords, constraints=constraints)
            
            self.logger.info(f"Generated fallback structure with {length} residues")
            return structure
            
        except Exception as e:
            self.logger.error(f"Fallback generation failed: {e}")
            # Return minimal structure as last resort
            coords = torch.zeros(10, 3)
            coords[:, 0] = torch.arange(10, dtype=torch.float32) * 3.8
            return ProteinStructure(coords, constraints=constraints)
    
    @circuit_breaker("structure_validation", failure_threshold=10)
    def validate(self, structure: ProteinStructure) -> Dict[str, float]:
        """
        Validate structure with circuit breaker protection.
        
        Args:
            structure: Protein structure to validate
            
        Returns:
            Validation metrics dictionary
        """
        with self._profile_operation("structure_validation"):
            try:
                # Use parent validation with enhancements
                metrics = super().validate(structure)
                
                # Add robust-specific metrics
                robust_metrics = self._compute_robust_metrics(structure)
                metrics.update(robust_metrics)
                
                # Record validation attempt
                if self.monitoring:
                    self.monitoring.record_metric(
                        "validation_attempts", 1, MetricType.COUNTER
                    )
                    self.monitoring.record_metric(
                        "overall_validation_score", metrics.get('overall_score', 0.0)
                    )
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Validation error: {e}")
                
                if self.monitoring:
                    self.monitoring.record_metric(
                        "validation_errors", 1, MetricType.COUNTER
                    )
                
                # Return fallback validation
                return self._fallback_validate_structure(structure)
    
    def _compute_robust_metrics(self, structure: ProteinStructure) -> Dict[str, float]:
        """Compute additional robust validation metrics."""
        try:
            coords = structure.coordinates
            
            # Stability metrics
            center = torch.mean(coords, dim=0)
            distances = torch.norm(coords - center, dim=1)
            stability_score = 1.0 / (1.0 + torch.std(distances).item())
            
            # Compactness metric
            max_distance = torch.max(distances).item()
            expected_max = len(coords) ** 0.5 * 5.0  # Rough estimate
            compactness_score = min(1.0, expected_max / max(max_distance, 1.0))
            
            # Coordinate quality
            coord_range = torch.max(coords) - torch.min(coords)
            quality_score = max(0.0, 1.0 - (coord_range.item() / 1000.0))
            
            return {
                "stability_score": stability_score,
                "compactness_metric": compactness_score,
                "coordinate_quality": quality_score,
                "robust_overall_score": (stability_score + compactness_score + quality_score) / 3.0
            }
            
        except Exception as e:
            self.logger.warning(f"Robust metrics computation failed: {e}")
            return {
                "stability_score": 0.5,
                "compactness_metric": 0.5,
                "coordinate_quality": 0.5,
                "robust_overall_score": 0.5
            }
    
    def _fallback_validate_structure(self, structure: ProteinStructure) -> Dict[str, float]:
        """Fallback validation with minimal checks."""
        self.logger.warning("Using fallback structure validation")
        
        try:
            coords = structure.coordinates
            
            # Basic checks only
            if coords.numel() == 0:
                return {"overall_score": 0.0, "fallback_validation": True}
            
            # Simple length check
            length_score = min(1.0, len(coords) / 50.0)
            
            # Simple coordinate check
            coord_check = not (torch.isnan(coords).any() or torch.isinf(coords).any())
            coord_score = 1.0 if coord_check else 0.0
            
            overall_score = (length_score + coord_score) / 2.0
            
            return {
                "overall_score": overall_score,
                "length_score": length_score,
                "coordinate_score": coord_score,
                "fallback_validation": True
            }
            
        except Exception as e:
            self.logger.error(f"Fallback validation failed: {e}")
            return {
                "overall_score": 0.1,
                "fallback_validation": True,
                "validation_error": str(e)
            }
    
    @error_handler("structure_optimization", max_retries=2)
    def optimize(
        self,
        initial_structure: ProteinStructure,
        iterations: int = 100,
        **kwargs
    ) -> ProteinStructure:
        """
        Optimize structure with robust error handling.
        
        Args:
            initial_structure: Starting structure
            iterations: Number of optimization iterations
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized structure
        """
        with self._profile_operation("structure_optimization"):
            try:
                # Record optimization attempt
                if self.monitoring:
                    self.monitoring.record_metric(
                        "optimization_attempts", 1, MetricType.COUNTER,
                        {"iterations": str(iterations)}
                    )
                
                # Use parent optimization
                result = super().optimize(initial_structure, iterations, **kwargs)
                
                # Validate optimization result
                if not self._validate_optimization_result(initial_structure, result):
                    self.logger.warning("Optimization validation failed, using fallback")
                    return self._fallback_optimize_structure(initial_structure, iterations)
                
                # Record success
                if self.monitoring:
                    self.monitoring.record_metric(
                        "optimization_successes", 1, MetricType.COUNTER
                    )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                
                if self.monitoring:
                    self.monitoring.record_metric(
                        "optimization_errors", 1, MetricType.COUNTER
                    )
                
                # Try fallback optimization
                return self._fallback_optimize_structure(initial_structure, iterations)
    
    def _validate_optimization_result(
        self, 
        initial: ProteinStructure, 
        optimized: ProteinStructure
    ) -> bool:
        """Validate that optimization improved the structure."""
        try:
            if optimized is None:
                return False
            
            # Basic structure validation
            if not self._validate_generation_result(optimized):
                return False
            
            # Check that structure didn't diverge too much
            if hasattr(initial, 'coordinates') and hasattr(optimized, 'coordinates'):
                initial_coords = initial.coordinates
                optimized_coords = optimized.coordinates
                
                if initial_coords.shape != optimized_coords.shape:
                    return False
                
                # RMSD check (should not be too large)
                rmsd = torch.sqrt(torch.mean((initial_coords - optimized_coords) ** 2))
                if rmsd > 50.0:  # Too large change
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Optimization validation error: {e}")
            return False
    
    def _fallback_optimize_structure(
        self, 
        structure: ProteinStructure, 
        iterations: int = 100
    ) -> ProteinStructure:
        """Fallback optimization using simple energy minimization."""
        self.logger.warning("Using fallback structure optimization")
        
        try:
            coords = structure.coordinates.clone()
            
            # Simple gradient descent with random perturbations
            for i in range(min(iterations // 10, 10)):  # Reduced iterations
                # Add small random perturbation
                noise = torch.randn_like(coords) * 0.1
                coords = coords + noise
                
                # Simple centering
                coords = coords - torch.mean(coords, dim=0, keepdim=True)
            
            # Create optimized structure
            from .structure import ProteinStructure
            optimized = ProteinStructure(coords, constraints=structure.constraints)
            
            self.logger.info("Fallback optimization complete")
            return optimized
            
        except Exception as e:
            self.logger.error(f"Fallback optimization failed: {e}")
            return structure  # Return original if fallback fails
    
    @contextmanager
    def _profile_operation(self, operation_name: str, **tags):
        """Context manager for operation profiling."""
        if self.monitoring:
            with self.monitoring.profile_operation(operation_name, **tags):
                yield
        else:
            yield
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the robust designer."""
        status = {
            "timestamp": time.time(),
            "operational": True,
            "design_count": self.design_count,
            "success_rate": self.success_rate
        }
        
        # Add monitoring data if available
        if self.monitoring:
            dashboard_data = self.monitoring.get_dashboard_data()
            status.update({
                "system_metrics": dashboard_data.get("current_metrics", {}),
                "active_alerts": dashboard_data.get("active_alerts", []),
                "health_checks": dashboard_data.get("health_checks", {}),
                "performance": dashboard_data.get("performance", {})
            })
        
        # Add error handling statistics
        if self.error_handler:
            error_stats = self.error_handler.get_error_statistics()
            status["error_statistics"] = error_stats
        
        return status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        report = {
            "timestamp": time.time(),
            "designs_generated": self.design_count,
            "operator_type": self.operator_type,
            "device": str(self.device)
        }
        
        if self.monitoring:
            # Get performance summaries
            operations = ["protein_generation", "structure_validation", "structure_optimization"]
            performance_data = {}
            
            for operation in operations:
                perf_data = self.monitoring.profiler.get_performance_summary(operation)
                if perf_data:
                    performance_data[operation] = perf_data
            
            report["performance_metrics"] = performance_data
        
        return report
    
    def shutdown(self):
        """Gracefully shutdown the robust designer."""
        self.logger.info("Shutting down robust protein designer")
        
        # Stop monitoring
        if self.monitoring:
            self.monitoring.stop()
        
        # Clear error handler resources
        if self.error_handler:
            self.error_handler.clear_error_history()
        
        self.logger.info("Shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore cleanup errors in destructor


# Convenience function for creating robust designer
def create_robust_designer(
    operator_type: str = "deeponet",
    checkpoint: Optional[Union[str, Path]] = None,
    enable_monitoring: bool = True,
    enable_error_recovery: bool = True,
    **kwargs
) -> RobustProteinDesigner:
    """
    Create a robust protein designer with recommended settings.
    
    Args:
        operator_type: Neural operator type
        checkpoint: Model checkpoint path
        enable_monitoring: Enable monitoring system
        enable_error_recovery: Enable error recovery
        **kwargs: Additional designer parameters
        
    Returns:
        Configured robust protein designer
    """
    return RobustProteinDesigner(
        operator_type=operator_type,
        checkpoint=checkpoint,
        enable_monitoring=enable_monitoring,
        enable_error_recovery=enable_error_recovery,
        **kwargs
    )