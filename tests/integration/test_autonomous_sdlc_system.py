"""
Integration tests for the complete autonomous SDLC system.

Tests the integration of all advanced components:
- Enhanced DeepONet models
- Advanced design service
- Validation framework
- Monitoring system
- Security manager
- Research framework
- Distributed coordination
- Adaptive caching
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import time
import numpy as np

# Import all the advanced components
try:
    import torch
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    import mock_torch as torch

from src.protein_operators.models.enhanced_deeponet import EnhancedProteinDeepONet
from src.protein_operators.services.advanced_design_service import (
    AdvancedDesignService, DesignObjective, OptimizationStrategy
)
from src.protein_operators.validation.advanced_validation import (
    AdvancedValidationFramework, ValidationLevel
)
from src.protein_operators.utils.advanced_monitoring import AdvancedMonitoringSystem
from src.protein_operators.utils.security_manager import SecurityManager, SecurityLevel
from src.protein_operators.research.advanced_research_framework import (
    AdvancedResearchFramework, ExperimentType, Hypothesis
)
from src.protein_operators.infrastructure.distributed_coordinator import (
    DistributedCoordinator, NodeRole
)
from src.protein_operators.utils.adaptive_caching import AdaptiveCacheSystem, CacheHitType
from src.protein_operators.constraints import Constraints
from src.protein_operators.structure import ProteinStructure


class TestAutonomousSDLCSystem:
    """Integration tests for the complete autonomous SDLC system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def enhanced_model(self):
        """Create enhanced DeepONet model."""
        model = EnhancedProteinDeepONet(
            constraint_dim=256,
            adaptive_basis=True,
            uncertainty_quantification=True,
            num_ensemble=3
        )
        return model
    
    @pytest.fixture
    def test_constraints(self):
        """Create test constraints."""
        constraints = Constraints()
        # Add some basic constraints for testing
        return constraints
    
    @pytest.fixture
    def test_coordinates(self):
        """Create test coordinates."""
        # Simple extended chain
        coords = []
        for i in range(50):
            coords.append([i * 3.8, 0.0, 0.0])
        return torch.tensor(coords, dtype=torch.float32)
    
    def test_enhanced_deeponet_integration(self, enhanced_model, test_constraints, test_coordinates):
        """Test enhanced DeepONet integration."""
        # Test basic forward pass
        batch_size = 2
        constraints_batch = torch.randn(batch_size, 5, 10)  # Mock constraints
        coords_batch = test_coordinates.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Test forward pass
        output = enhanced_model(constraints_batch, coords_batch)
        assert output.shape == coords_batch.shape
        
        # Test forward with uncertainty
        output, uncertainties = enhanced_model.forward_with_uncertainty(constraints_batch, coords_batch)
        assert output.shape == coords_batch.shape
        assert uncertainties is not None
        assert 'epistemic' in uncertainties
        assert 'aleatoric' in uncertainties
        assert 'total' in uncertainties
        
        # Test feature importance
        constraints_batch.requires_grad_(True)
        coords_batch.requires_grad_(True)
        
        importance = enhanced_model.get_feature_importance(constraints_batch, coords_batch)
        assert 'constraint_importance' in importance
        assert 'coordinate_importance' in importance
    
    @pytest.mark.asyncio
    async def test_advanced_design_service_integration(self, temp_dir):
        """Test advanced design service integration."""
        # Initialize service
        service = AdvancedDesignService(
            use_enhanced_model=True,
            max_concurrent_designs=2
        )
        
        # Create test constraints and objectives
        constraints = Constraints()
        objectives = [
            DesignObjective(
                name="overall_score",
                target_value=0.8,
                optimization_direction="maximize"
            ),
            DesignObjective(
                name="stereochemistry_score",
                target_value=0.9,
                optimization_direction="maximize"
            )
        ]
        
        # Test synchronous design
        result = service.design_protein(
            constraints=constraints,
            objectives=objectives,
            length=50,
            strategy=OptimizationStrategy.EVOLUTIONARY,
            max_iterations=5  # Short for testing
        )
        
        assert result is not None
        assert result.structure is not None
        assert isinstance(result.objectives, dict)
        assert result.optimization_time > 0
        
        # Test asynchronous design
        async_result = await service.design_protein_async(
            constraints=constraints,
            objectives=objectives,
            length=50,
            strategy=OptimizationStrategy.GRADIENT_BASED,
            max_iterations=5
        )
        
        assert async_result is not None
        assert async_result.structure is not None
        
        # Test statistics
        stats = service.get_design_statistics()
        assert stats['total_designs'] >= 2
        assert 'success_rate' in stats
        
        # Cleanup
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_validation_framework_integration(self, test_coordinates, test_constraints):
        """Test validation framework integration."""
        # Initialize framework
        framework = AdvancedValidationFramework(
            enable_ai_predictor=True,
            max_concurrent_validations=2
        )
        
        # Create test structure
        structure = ProteinStructure(test_coordinates, test_constraints)
        
        # Test synchronous validation
        report = framework.validate_structure(
            structure,
            validation_level=ValidationLevel.COMPREHENSIVE
        )
        
        assert report is not None
        assert report.overall_score >= 0.0
        assert len(report.metrics) > 0
        assert report.validation_time > 0
        
        # Test asynchronous validation
        async_report = await framework.validate_structure_async(
            structure,
            validation_level=ValidationLevel.INTERMEDIATE
        )
        
        assert async_report is not None
        assert async_report.overall_score >= 0.0
        
        # Test statistics
        stats = framework.get_validation_statistics()
        assert stats['total_validations'] >= 2
        
        # Cleanup
        await framework.shutdown()
    
    def test_monitoring_system_integration(self, temp_dir):
        """Test monitoring system integration."""
        # Initialize monitoring
        monitoring = AdvancedMonitoringSystem(
            enable_resource_monitoring=True,
            enable_health_checks=True
        )
        
        # Start monitoring
        monitoring.start()
        
        # Test profiling
        with monitoring.profile_operation("test_operation", test_tag="integration") as profile_id:
            time.sleep(0.1)  # Simulate work
            assert profile_id is not None
        
        # Test metric recording
        monitoring.record_metric("test_metric", 42.0, tags={"test": "true"})
        
        # Test dashboard data
        dashboard = monitoring.get_dashboard_data()
        assert 'current_metrics' in dashboard
        assert 'system_info' in dashboard
        assert 'timestamp' in dashboard
        
        # Export metrics
        export_path = temp_dir / "metrics.json"
        monitoring.export_metrics(str(export_path))
        assert export_path.exists()
        
        # Stop monitoring
        monitoring.stop()
    
    def test_security_manager_integration(self):
        """Test security manager integration."""
        # Initialize security manager
        security = SecurityManager(
            enable_rate_limiting=True,
            enable_audit_logging=True
        )
        
        # Create test user
        user_id = security.create_user(
            "test_user",
            "test@example.com",
            SecurityLevel.AUTHENTICATED
        )
        assert user_id is not None
        
        # Test login
        session_token = security.login_user(user_id)
        assert session_token is not None
        
        # Test secured operation decorator
        @security.secure_operation("test_operation")
        def test_operation(data: str, session_token: str = None):
            return f"Processed: {data}"
        
        # Test with valid session
        result = test_operation("test_data", session_token=session_token)
        assert result == "Processed: test_data"
        
        # Test without session (should fail)
        with pytest.raises(PermissionError):
            test_operation("test_data")
        
        # Test dashboard
        dashboard = security.get_security_dashboard()
        assert 'total_users' in dashboard
        assert 'active_sessions' in dashboard
    
    @pytest.mark.asyncio
    async def test_research_framework_integration(self, temp_dir):
        """Test research framework integration."""
        # Initialize research framework
        framework = AdvancedResearchFramework(
            output_directory=str(temp_dir / "research"),
            enable_validation=True,
            max_concurrent_experiments=2
        )
        
        # Define test hypothesis
        hypothesis = Hypothesis(
            name="test_hypothesis",
            description="Test hypothesis for integration",
            null_hypothesis="No difference between methods",
            alternative_hypothesis="Method A performs better than Method B",
            expected_effect_size=0.5,
            metrics=["overall_score", "stereochemistry_score"]
        )
        
        # Define base configuration
        base_config = {
            "operator_type": "deeponet",
            "length": 50,
            "num_samples": 1
        }
        
        # Conduct comparative study
        study_result = await framework.conduct_research_study(
            study_name="integration_test_study",
            experiment_type=ExperimentType.COMPARATIVE_STUDY,
            hypotheses=[hypothesis],
            base_configuration=base_config,
            comparison_methods=[{"operator_type": "fno", "length": 50}],
            evaluation_metrics=["overall_score", "stereochemistry_score"],
            effect_size=0.3  # Smaller effect for faster testing
        )
        
        assert study_result is not None
        assert 'study_id' in study_result
        assert 'results' in study_result
        assert 'analysis' in study_result
        assert len(study_result['results']) > 0
        
        # Check output files
        output_dir = Path(temp_dir / "research")
        assert output_dir.exists()
        
        # Cleanup
        await framework.shutdown()
    
    @pytest.mark.asyncio
    async def test_distributed_coordinator_integration(self):
        """Test distributed coordinator integration."""
        # Initialize coordinator
        coordinator = DistributedCoordinator(
            coordinator_id="test_coordinator",
            port=8081,  # Different port for testing
            max_concurrent_tasks=10
        )
        
        # Start coordinator
        await coordinator.start()
        
        # Register test node
        node_registered = coordinator.register_node(
            node_id="test_node_1",
            hostname="localhost",
            ip_address="127.0.0.1",
            port=8082,
            capabilities={
                "has_gpu": False,
                "memory_gb": 8,
                "supported_task_types": ["protein_design", "validation"]
            }
        )
        assert node_registered
        
        # Submit test tasks
        task_id_1 = coordinator.submit_task(
            task_type="protein_design",
            parameters={"length": 50, "constraints": {}},
            priority=1,
            estimated_duration=30.0
        )
        assert task_id_1 is not None
        
        task_id_2 = coordinator.submit_task(
            task_type="validation",
            parameters={"structure_id": "test_structure"},
            priority=2,
            estimated_duration=15.0
        )
        assert task_id_2 is not None
        
        # Wait for task assignment
        await asyncio.sleep(2.0)
        
        # Check task status
        status_1 = coordinator.get_task_status(task_id_1)
        assert status_1 is not None
        assert status_1['task_id'] == task_id_1
        
        # Check cluster status
        cluster_status = coordinator.get_cluster_status()
        assert cluster_status['cluster_health']['total_nodes'] >= 1
        assert cluster_status['task_statistics']['total_tasks'] >= 2
        
        # Unregister node
        node_unregistered = coordinator.unregister_node("test_node_1")
        assert node_unregistered
        
        # Stop coordinator
        await coordinator.stop()
    
    def test_adaptive_caching_integration(self, temp_dir):
        """Test adaptive caching integration."""
        # Initialize cache system
        cache = AdaptiveCacheSystem(
            l1_size_mb=16,
            l2_size_mb=64,
            l3_size_mb=128,
            cache_dir=str(temp_dir / "cache"),
            enable_compression=True,
            enable_prediction=True
        )
        
        # Start background tasks
        cache.start_background_tasks()
        
        # Test caching
        test_data = {
            "coordinates": torch.randn(100, 3),
            "metadata": {"length": 100, "type": "test"}
        }
        
        # Store in cache
        cache.put("test_key_1", test_data)
        
        # Retrieve from cache
        retrieved_data, hit_type = cache.get("test_key_1")
        assert hit_type == CacheHitType.HIT
        assert retrieved_data is not None
        assert torch.allclose(retrieved_data["coordinates"], test_data["coordinates"])
        
        # Test cache miss
        missing_data, hit_type = cache.get("nonexistent_key")
        assert hit_type == CacheHitType.MISS
        assert missing_data is None
        
        # Test cache decorator
        @cache.cached(cache, ttl=300.0)
        def expensive_computation(n: int):
            time.sleep(0.01)  # Simulate work
            return list(range(n))
        
        # First call (cache miss)
        start_time = time.time()
        result_1 = expensive_computation(100)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result_2 = expensive_computation(100)
        second_call_time = time.time() - start_time
        
        assert result_1 == result_2
        assert second_call_time < first_call_time  # Should be faster
        
        # Test performance report
        report = cache.get_performance_report()
        assert 'global_stats' in report
        assert 'level_stats' in report
        assert report['global_stats']['hit_ratio'] > 0.0
        
        # Stop background tasks
        cache.stop_background_tasks()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, temp_dir):
        """Test complete end-to-end workflow integration."""
        # Initialize all components
        cache = AdaptiveCacheSystem(
            l1_size_mb=32,
            cache_dir=str(temp_dir / "cache")
        )
        cache.start_background_tasks()
        
        monitoring = AdvancedMonitoringSystem()
        monitoring.start()
        
        security = SecurityManager()
        
        validation_framework = AdvancedValidationFramework()
        
        design_service = AdvancedDesignService(
            use_enhanced_model=True,
            max_concurrent_designs=1
        )
        
        try:
            # Create user and login
            user_id = security.create_user(
                "workflow_user",
                "workflow@example.com",
                SecurityLevel.AUTHENTICATED
            )
            session_token = security.login_user(user_id)
            
            # Define design task with caching
            @cache.cached(cache, ttl=3600.0)
            def cached_design_task(length: int, constraints_hash: str):
                constraints = Constraints()
                objectives = [
                    DesignObjective(
                        name="overall_score",
                        target_value=0.8,
                        optimization_direction="maximize"
                    )
                ]
                
                return design_service.design_protein(
                    constraints=constraints,
                    objectives=objectives,
                    length=length,
                    strategy=OptimizationStrategy.GRADIENT_BASED,
                    max_iterations=3
                )
            
            # Execute workflow with monitoring
            with monitoring.profile_operation("end_to_end_design") as profile_id:
                # First design (cache miss)
                result_1 = cached_design_task(50, "test_constraints_hash")
                
                # Validate result
                validation_report = validation_framework.validate_structure(
                    result_1.structure,
                    ValidationLevel.COMPREHENSIVE
                )
                
                # Second design (cache hit)
                result_2 = cached_design_task(50, "test_constraints_hash")
                
                # Verify cache hit
                assert result_1.structure.coordinates.shape == result_2.structure.coordinates.shape
            
            # Check monitoring data
            dashboard = monitoring.get_dashboard_data()
            assert 'performance' in dashboard
            
            # Check cache performance
            cache_report = cache.get_performance_report()
            assert cache_report['global_stats']['hit_ratio'] > 0.0
            
            # Check security audit
            security_dashboard = security.get_security_dashboard()
            assert security_dashboard['total_users'] >= 1
            
            # Verify validation results
            assert validation_report.overall_score >= 0.0
            assert len(validation_report.metrics) > 0
            
        finally:
            # Cleanup all components
            await design_service.shutdown()
            await validation_framework.shutdown()
            cache.stop_background_tasks()
            monitoring.stop()
    
    def test_system_resilience(self):
        """Test system resilience and error handling."""
        # Test with invalid inputs
        model = EnhancedProteinDeepONet()
        
        # Test with malformed inputs
        try:
            invalid_constraints = torch.tensor([])  # Empty tensor
            invalid_coords = torch.tensor([])  # Empty tensor
            
            result = model(invalid_constraints, invalid_coords)
            # Should handle gracefully
            assert result is not None
        except Exception as e:
            # Errors should be informative
            assert str(e) != ""
        
        # Test cache with invalid data
        cache = AdaptiveCacheSystem(l1_size_mb=1)
        
        # Test cache invalidation
        cache.put("test_key", "test_value")
        assert cache.invalidate("test_key")
        assert not cache.invalidate("nonexistent_key")
        
        # Test cache with very large data
        large_data = np.random.randn(1000, 1000)  # Large array
        cache.put("large_key", large_data)
        
        retrieved, hit_type = cache.get("large_key")
        if hit_type == CacheHitType.HIT:
            assert np.allclose(retrieved, large_data)
    
    def test_performance_benchmarks(self, enhanced_model, test_coordinates, test_constraints):
        """Test performance benchmarks for the system."""
        batch_size = 4
        constraints_batch = torch.randn(batch_size, 5, 10)
        coords_batch = test_coordinates.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Benchmark forward pass
        times = []
        for _ in range(10):
            start_time = time.time()
            output = enhanced_model(constraints_batch, coords_batch)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0  # Should complete in less than 1 second
        
        # Benchmark with uncertainty
        uncertainty_times = []
        for _ in range(5):
            start_time = time.time()
            output, uncertainties = enhanced_model.forward_with_uncertainty(
                constraints_batch, coords_batch
            )
            end_time = time.time()
            uncertainty_times.append(end_time - start_time)
        
        avg_uncertainty_time = sum(uncertainty_times) / len(uncertainty_times)
        assert avg_uncertainty_time < 2.0  # Uncertainty computation should be reasonable
        
        # Memory usage should be reasonable
        if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            output = enhanced_model(constraints_batch, coords_batch)
            memory_after = torch.cuda.memory_allocated()
            memory_diff = memory_after - memory_before
            
            # Memory usage should be proportional to batch size
            assert memory_diff > 0  # Should use some memory
            # Memory per item should be reasonable
            memory_per_item = memory_diff / batch_size
            assert memory_per_item < 100 * 1024 * 1024  # Less than 100MB per item
    
    def test_configuration_management(self, temp_dir):
        """Test configuration management across components."""
        config_file = temp_dir / "test_config.json"
        
        # Test configuration serialization
        config = {
            "cache": {
                "l1_size_mb": 128,
                "l2_size_mb": 512,
                "enable_compression": True
            },
            "monitoring": {
                "enable_resource_monitoring": True,
                "resource_collection_interval": 10.0
            },
            "security": {
                "enable_rate_limiting": True,
                "enable_audit_logging": True
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Test configuration loading
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == config
        
        # Test component initialization with config
        cache = AdaptiveCacheSystem(
            **loaded_config["cache"],
            cache_dir=str(temp_dir / "cache")
        )
        
        monitoring = AdvancedMonitoringSystem(
            **loaded_config["monitoring"]
        )
        
        security = SecurityManager(
            **loaded_config["security"]
        )
        
        # Verify components work with loaded config
        assert cache is not None
        assert monitoring is not None
        assert security is not None
        
        # Test configuration validation
        invalid_config = {
            "cache": {
                "l1_size_mb": -1,  # Invalid
            }
        }
        
        # Should handle invalid config gracefully
        try:
            cache_invalid = AdaptiveCacheSystem(
                l1_size_mb=max(1, invalid_config["cache"]["l1_size_mb"]),
                cache_dir=str(temp_dir / "cache")
            )
            assert cache_invalid is not None
        except Exception as e:
            assert "Invalid" in str(e) or "invalid" in str(e)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
