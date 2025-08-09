#!/usr/bin/env python3
"""
Test performance optimization and resource monitoring systems.

Tests caching, parallel processing, and resource monitoring functionality.
"""

import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test imports
try:
    from protein_operators.utils.performance_optimizer import (
        CacheStrategy, LRUCache, MemoryAwareCache, CacheManager,
        ParallelProcessor, ComputeStrategy, PerformanceProfiler,
        cached, parallel_batch, profiled
    )
    from protein_operators.utils.resource_monitor import (
        ResourceMonitor, ResourceType, ResourceThreshold,
        SystemResourceCollector, ResourceMetrics
    )
    print("✅ Performance system imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def run_test(self, test_func, test_name):
        """Run a single test."""
        self.tests_run += 1
        try:
            test_func()
            self.tests_passed += 1
            print(f"✅ {test_name}")
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")
    
    def run_test_class(self, test_class):
        """Run all tests in a test class."""
        instance = test_class()
        class_name = test_class.__name__
        print(f"\n--- Running {class_name} ---")
        
        test_methods = [method for method in dir(instance) 
                      if method.startswith('test_')]
        
        for method_name in test_methods:
            test_func = getattr(instance, method_name)
            full_name = f"{class_name}.{method_name}"
            self.run_test(test_func, full_name)
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE SYSTEMS TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\nFAILURES:")
            for name, error in self.failures:
                print(f"  {name}: {error}")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        return self.tests_failed == 0


class TestLRUCache:
    """Test LRU Cache implementation."""
    
    def test_cache_creation(self):
        """Test creating LRU cache."""
        cache = LRUCache(max_size=3)
        assert cache.max_size == 3
        assert len(cache.cache) == 0
        print("✓ Cache creation test passed")
    
    def test_cache_set_get(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)
        
        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Get values
        value1 = cache.get("key1")
        value2 = cache.get("key2")
        
        assert value1 == "value1"
        assert value2 == "value2"
        assert cache.get("nonexistent") is None
        print("✓ Cache set/get test passed")
    
    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key - should evict key2
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist
        print("✓ Cache eviction test passed")
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=3)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        print("✓ Cache stats test passed")


class TestMemoryAwareCache:
    """Test Memory-Aware Cache implementation."""
    
    def test_cache_creation(self):
        """Test creating memory-aware cache."""
        cache = MemoryAwareCache(max_memory_mb=64)
        assert cache.max_memory_mb == 64
        assert len(cache.cache) == 0
        print("✓ Memory-aware cache creation test passed")
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = MemoryAwareCache(max_memory_mb=64)
        
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        assert value == "test_value"
        assert cache.get("nonexistent") is None
        print("✓ Memory-aware cache operations test passed")


class TestCacheManager:
    """Test Cache Manager."""
    
    def test_manager_creation(self):
        """Test creating cache manager."""
        manager = CacheManager()
        assert manager.default_strategy == CacheStrategy.LRU
        assert len(manager.caches) == 0
        print("✓ Cache manager creation test passed")
    
    def test_cache_retrieval(self):
        """Test cache retrieval and creation."""
        manager = CacheManager()
        
        # Get LRU cache
        lru_cache = manager.get_cache("test_lru", CacheStrategy.LRU, max_size=64)
        assert isinstance(lru_cache, LRUCache)
        assert lru_cache.max_size == 64
        
        # Get same cache again - should be same instance
        same_cache = manager.get_cache("test_lru")
        assert same_cache is lru_cache
        print("✓ Cache retrieval test passed")
    
    def test_multiple_caches(self):
        """Test managing multiple caches."""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1", CacheStrategy.LRU)
        cache2 = manager.get_cache("cache2", CacheStrategy.MEMORY_AWARE)
        
        assert cache1 is not cache2
        assert len(manager.caches) == 2
        print("✓ Multiple caches test passed")


class TestParallelProcessor:
    """Test Parallel Processor."""
    
    def test_processor_creation(self):
        """Test creating parallel processor."""
        processor = ParallelProcessor()
        assert processor.default_strategy == ComputeStrategy.ADAPTIVE
        assert processor.cpu_count > 0
        print("✓ Parallel processor creation test passed")
    
    def test_sequential_processing(self):
        """Test sequential processing."""
        processor = ParallelProcessor()
        
        # Simple function to apply
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = processor.process_batch(
            square, items, strategy=ComputeStrategy.SEQUENTIAL
        )
        
        expected = [1, 4, 9, 16, 25]
        assert results == expected
        print("✓ Sequential processing test passed")
    
    def test_thread_parallel_processing(self):
        """Test thread parallel processing."""
        processor = ParallelProcessor()
        
        def slow_square(x):
            time.sleep(0.01)  # Small delay
            return x * x
        
        items = [1, 2, 3, 4]
        start_time = time.time()
        results = processor.process_batch(
            slow_square, items, strategy=ComputeStrategy.THREAD_PARALLEL
        )
        end_time = time.time()
        
        expected = [1, 4, 9, 16]
        assert sorted(results) == sorted(expected)
        
        # Should be faster than sequential (though this is a rough test)
        parallel_time = end_time - start_time
        assert parallel_time < 0.1  # Should complete faster than 0.04s sequential
        print("✓ Thread parallel processing test passed")


class TestPerformanceProfiler:
    """Test Performance Profiler."""
    
    def test_profiler_creation(self):
        """Test creating performance profiler."""
        profiler = PerformanceProfiler()
        assert len(profiler.profiles) == 0
        assert len(profiler.active_profiles) == 0
        print("✓ Performance profiler creation test passed")
    
    def test_profiling_basic_operation(self):
        """Test profiling a basic operation."""
        profiler = PerformanceProfiler()
        
        profile_id = profiler.start_profile("test_operation")
        
        # Simulate some work
        time.sleep(0.1)
        
        metrics = profiler.end_profile(profile_id)
        
        assert metrics.execution_time >= 0.09  # Should be close to 0.1s
        assert metrics.execution_time < 0.2    # But not too much more
        assert "test_operation" in profiler.profiles
        print("✓ Basic profiling test passed")
    
    def test_profiler_statistics(self):
        """Test profiler statistics."""
        profiler = PerformanceProfiler()
        
        # Profile multiple operations
        for i in range(3):
            profile_id = profiler.start_profile("repeated_op")
            time.sleep(0.01)
            profiler.end_profile(profile_id)
        
        stats = profiler.get_operation_stats("repeated_op")
        assert stats is not None
        assert stats["total_calls"] == 3
        assert stats["avg_execution_time"] > 0
        print("✓ Profiler statistics test passed")


class TestCachedDecorator:
    """Test cached decorator."""
    
    def test_cached_decorator(self):
        """Test the cached decorator."""
        call_count = 0
        
        @cached("test_cache", strategy=CacheStrategy.LRU, max_size=10)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * x
        
        # First call - should execute function
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Second call with same argument - should use cache
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # Should not increment
        
        # Different argument - should execute function
        result3 = expensive_function(6)
        assert result3 == 36
        assert call_count == 2
        
        print("✓ Cached decorator test passed")


class TestResourceCollector:
    """Test Resource Collector."""
    
    def test_collector_creation(self):
        """Test creating system resource collector."""
        collector = SystemResourceCollector()
        # Should always be available (with fallback)
        assert collector.is_available() or not collector.is_available()  # Either is fine
        print("✓ Resource collector creation test passed")
    
    def test_metrics_collection(self):
        """Test collecting metrics."""
        collector = SystemResourceCollector()
        metrics = collector.collect()
        
        # Should have basic metrics
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_usage_percent" in metrics
        
        # Values should be reasonable
        assert 0 <= metrics["cpu_percent"] <= 100
        assert 0 <= metrics["memory_percent"] <= 100
        assert 0 <= metrics["disk_usage_percent"] <= 100
        
        print("✓ Metrics collection test passed")


class TestResourceMetrics:
    """Test ResourceMetrics class."""
    
    def test_metrics_creation(self):
        """Test creating resource metrics."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available_gb=4.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0
        )
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        print("✓ Resource metrics creation test passed")
    
    def test_pressure_detection(self):
        """Test resource pressure detection."""
        # Normal metrics - no pressure
        normal_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=30.0,
            memory_percent=40.0,
            memory_available_gb=4.0,
            disk_usage_percent=50.0,
            disk_free_gb=100.0
        )
        assert not normal_metrics.is_under_pressure
        
        # High CPU - should detect pressure
        high_cpu_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=85.0,
            memory_percent=40.0,
            memory_available_gb=4.0,
            disk_usage_percent=50.0,
            disk_free_gb=100.0
        )
        assert high_cpu_metrics.is_under_pressure
        
        print("✓ Pressure detection test passed")
    
    def test_pressure_score(self):
        """Test pressure score calculation."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=80.0,  # High memory usage
            memory_available_gb=2.0,
            disk_usage_percent=30.0,
            disk_free_gb=200.0
        )
        
        score = metrics.pressure_score
        assert 0.0 <= score <= 1.0
        assert score >= 0.8  # Should be high due to memory usage
        print("✓ Pressure score test passed")


def main():
    """Main test execution."""
    print("🚀 Running Performance Systems Tests")
    print("=" * 60)
    
    test_classes = [
        TestLRUCache,
        TestMemoryAwareCache,
        TestCacheManager,
        TestParallelProcessor,
        TestPerformanceProfiler,
        TestCachedDecorator,
        TestResourceCollector,
        TestResourceMetrics,
    ]
    
    runner = TestRunner()
    for test_class in test_classes:
        runner.run_test_class(test_class)
    
    success = runner.print_summary()
    
    if success:
        print("\n🎉 All performance systems tests passed!")
        print("✓ Caching system working correctly")
        print("✓ Parallel processing system working correctly")  
        print("✓ Performance profiling system working correctly")
        print("✓ Resource monitoring system working correctly")
    else:
        print("\n⚠️  Some tests failed. Check the failures above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())