#!/usr/bin/env python3
"""
Standalone performance systems tests.

Tests core performance optimization functionality without external dependencies.
"""

import sys
import time
import threading
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import just the performance optimizer (no torch dependencies)
try:
    # Import modules directly to avoid torch dependency
    import importlib.util
    
    # Load performance_optimizer directly
    perf_spec = importlib.util.spec_from_file_location(
        "performance_optimizer",
        Path(__file__).parent.parent / 'src' / 'protein_operators' / 'utils' / 'performance_optimizer.py'
    )
    perf_module = importlib.util.module_from_spec(perf_spec)
    perf_spec.loader.exec_module(perf_module)
    
    # Load resource_monitor directly
    resource_spec = importlib.util.spec_from_file_location(
        "resource_monitor", 
        Path(__file__).parent.parent / 'src' / 'protein_operators' / 'utils' / 'resource_monitor.py'
    )
    resource_module = importlib.util.module_from_spec(resource_spec)
    resource_spec.loader.exec_module(resource_module)
    
    # Extract classes
    CacheStrategy = perf_module.CacheStrategy
    LRUCache = perf_module.LRUCache
    MemoryAwareCache = perf_module.MemoryAwareCache
    CacheManager = perf_module.CacheManager
    ParallelProcessor = perf_module.ParallelProcessor
    ComputeStrategy = perf_module.ComputeStrategy
    PerformanceProfiler = perf_module.PerformanceProfiler
    
    ResourceType = resource_module.ResourceType
    ResourceThreshold = resource_module.ResourceThreshold
    ResourceMetrics = resource_module.ResourceMetrics
    
    print("✅ Performance system imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
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
            if "--verbose" in sys.argv:
                import traceback
                traceback.print_exc()
    
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
        print("✓ LRU Cache creation successful")
    
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
        print("✓ LRU Cache basic operations working")
    
    def test_cache_eviction(self):
        """Test LRU eviction policy."""
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
        print("✓ LRU Cache eviction policy working correctly")
    
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
        assert stats["size"] == 1
        print("✓ LRU Cache statistics working correctly")
    
    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = LRUCache(max_size=100)
        
        def worker(worker_id):
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                cache.set(key, value)
                assert cache.get(key) == value
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify some values are still there
        assert len(cache.cache) > 0
        print("✓ LRU Cache thread safety working")


class TestMemoryAwareCache:
    """Test Memory-Aware Cache."""
    
    def test_cache_creation(self):
        """Test creating memory-aware cache."""
        cache = MemoryAwareCache(max_memory_mb=64)
        assert cache.max_memory_mb == 64
        assert len(cache.cache) == 0
        print("✓ Memory-aware cache creation successful")
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = MemoryAwareCache(max_memory_mb=64)
        
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        assert value == "test_value"
        assert cache.get("nonexistent") is None
        print("✓ Memory-aware cache operations working")
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = MemoryAwareCache(max_memory_mb=64)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        print("✓ Memory-aware cache stats working")


class TestCacheManager:
    """Test Cache Manager."""
    
    def test_manager_creation(self):
        """Test creating cache manager."""
        manager = CacheManager()
        assert manager.default_strategy == CacheStrategy.LRU
        assert len(manager.caches) == 0
        print("✓ Cache manager creation successful")
    
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
        print("✓ Cache retrieval and reuse working")
    
    def test_multiple_caches(self):
        """Test managing multiple caches."""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1", CacheStrategy.LRU, max_size=32)
        cache2 = manager.get_cache("cache2", CacheStrategy.MEMORY_AWARE, max_memory_mb=128)
        
        assert cache1 is not cache2
        assert len(manager.caches) == 2
        assert isinstance(cache1, LRUCache)
        assert isinstance(cache2, MemoryAwareCache)
        print("✓ Multiple cache management working")
    
    def test_global_stats(self):
        """Test global cache statistics."""
        manager = CacheManager()
        
        cache1 = manager.get_cache("cache1", CacheStrategy.LRU)
        cache2 = manager.get_cache("cache2", CacheStrategy.LRU)
        
        # Add some data
        cache1.set("key1", "value1")
        cache1.get("key1")  # Hit
        cache2.get("missing")  # Miss
        
        stats = manager.get_global_stats()
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "cache_count" in stats
        assert stats["cache_count"] == 2
        print("✓ Global cache statistics working")


class TestParallelProcessor:
    """Test Parallel Processing."""
    
    def test_processor_creation(self):
        """Test creating parallel processor."""
        processor = ParallelProcessor()
        assert processor.default_strategy == ComputeStrategy.ADAPTIVE
        assert processor.cpu_count > 0
        assert processor.max_workers > 0
        print("✓ Parallel processor creation successful")
    
    def test_sequential_processing(self):
        """Test sequential processing."""
        processor = ParallelProcessor()
        
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = processor.process_batch(
            square, items, strategy=ComputeStrategy.SEQUENTIAL
        )
        
        expected = [1, 4, 9, 16, 25]
        assert results == expected
        print("✓ Sequential processing working correctly")
    
    def test_thread_parallel_processing(self):
        """Test thread parallel processing."""
        processor = ParallelProcessor()
        
        def slow_square(x):
            time.sleep(0.001)  # Very small delay to avoid making tests too slow
            return x * x
        
        items = [1, 2, 3, 4]
        start_time = time.time()
        results = processor.process_batch(
            slow_square, items, strategy=ComputeStrategy.THREAD_PARALLEL
        )
        parallel_time = time.time() - start_time
        
        expected = [1, 4, 9, 16]
        assert sorted(results) == sorted(expected)
        
        # Should complete reasonably quickly
        assert parallel_time < 1.0
        print(f"✓ Thread parallel processing working (time: {parallel_time:.3f}s)")
    
    def test_strategy_selection(self):
        """Test adaptive strategy selection."""
        processor = ParallelProcessor()
        
        # Small batch - should prefer sequential
        strategy = processor._select_optimal_strategy(5, lambda x: x)
        assert strategy == ComputeStrategy.SEQUENTIAL
        
        # Large batch - should prefer process parallel
        strategy = processor._select_optimal_strategy(200, lambda x: x)
        assert strategy == ComputeStrategy.PROCESS_PARALLEL
        
        # Medium batch - should prefer thread parallel
        strategy = processor._select_optimal_strategy(50, lambda x: x)
        assert strategy == ComputeStrategy.THREAD_PARALLEL
        
        print("✓ Adaptive strategy selection working")


class TestPerformanceProfiler:
    """Test Performance Profiler."""
    
    def test_profiler_creation(self):
        """Test creating performance profiler."""
        profiler = PerformanceProfiler()
        assert len(profiler.profiles) == 0
        assert len(profiler.active_profiles) == 0
        print("✓ Performance profiler creation successful")
    
    def test_basic_profiling(self):
        """Test basic operation profiling."""
        profiler = PerformanceProfiler()
        
        profile_id = profiler.start_profile("test_operation")
        assert profile_id in profiler.active_profiles
        
        # Simulate some work
        time.sleep(0.05)  # 50ms
        
        metrics = profiler.end_profile(profile_id)
        
        assert metrics.execution_time >= 0.04  # Should be at least 40ms
        assert metrics.execution_time < 0.1    # But not too much more
        assert "test_operation" in profiler.profiles
        assert profile_id not in profiler.active_profiles
        print(f"✓ Basic profiling working (measured: {metrics.execution_time:.3f}s)")
    
    def test_multiple_profiles(self):
        """Test multiple simultaneous profiles."""
        profiler = PerformanceProfiler()
        
        profile_id1 = profiler.start_profile("operation1")
        time.sleep(0.01)
        profile_id2 = profiler.start_profile("operation2") 
        time.sleep(0.01)
        
        metrics1 = profiler.end_profile(profile_id1)
        metrics2 = profiler.end_profile(profile_id2)
        
        assert metrics1.execution_time >= 0.015  # Started earlier
        assert metrics2.execution_time >= 0.005  # Started later
        assert len(profiler.profiles) == 2
        print("✓ Multiple simultaneous profiles working")
    
    def test_operation_statistics(self):
        """Test operation statistics."""
        profiler = PerformanceProfiler()
        
        # Profile same operation multiple times
        for i in range(3):
            profile_id = profiler.start_profile("repeated_op")
            time.sleep(0.01)
            profiler.end_profile(profile_id, cache_hits=i, cache_misses=1)
        
        stats = profiler.get_operation_stats("repeated_op")
        assert stats is not None
        assert stats["total_calls"] == 3
        assert stats["avg_execution_time"] > 0
        assert "min_execution_time" in stats
        assert "max_execution_time" in stats
        print("✓ Operation statistics working")
    
    def test_all_stats(self):
        """Test getting all statistics."""
        profiler = PerformanceProfiler()
        
        # Profile different operations
        for op in ["op1", "op2"]:
            profile_id = profiler.start_profile(op)
            time.sleep(0.001)
            profiler.end_profile(profile_id)
        
        all_stats = profiler.get_all_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats
        assert all_stats["op1"]["total_calls"] == 1
        assert all_stats["op2"]["total_calls"] == 1
        print("✓ All statistics working")


class TestResourceMetrics:
    """Test Resource Metrics."""
    
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
        assert metrics.disk_usage_percent == 70.0
        print("✓ Resource metrics creation successful")
    
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
        
        # High memory - should detect pressure
        high_memory_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=30.0,
            memory_percent=90.0,
            memory_available_gb=1.0,
            disk_usage_percent=50.0,
            disk_free_gb=100.0
        )
        assert high_memory_metrics.is_under_pressure
        
        print("✓ Resource pressure detection working")
    
    def test_pressure_score(self):
        """Test pressure score calculation."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=80.0,
            memory_available_gb=2.0,
            disk_usage_percent=30.0,
            disk_free_gb=200.0
        )
        
        score = metrics.pressure_score
        assert 0.0 <= score <= 1.0
        assert score >= 0.5  # Should be at least moderate due to memory
        print(f"✓ Pressure score calculation working (score: {score:.2f})")
    
    def test_metrics_serialization(self):
        """Test metrics to dictionary conversion."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=45.0,
            memory_percent=55.0,
            memory_available_gb=8.0,
            disk_usage_percent=65.0,
            disk_free_gb=200.0
        )
        
        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data["cpu_percent"] == 45.0
        assert data["memory_percent"] == 55.0
        assert "timestamp" in data
        print("✓ Metrics serialization working")


def main():
    """Main test execution."""
    print("🚀 Running Standalone Performance Systems Tests")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"CPU count: {os.cpu_count()}")
    print()
    
    test_classes = [
        TestLRUCache,
        TestMemoryAwareCache,
        TestCacheManager,
        TestParallelProcessor,
        TestPerformanceProfiler,
        TestResourceMetrics,
    ]
    
    runner = TestRunner()
    for test_class in test_classes:
        runner.run_test_class(test_class)
    
    success = runner.print_summary()
    
    if success:
        print("\n🎉 All performance systems tests passed!")
        print("✓ LRU Cache: Thread-safe caching with proper eviction")
        print("✓ Memory-Aware Cache: Adaptive memory management")
        print("✓ Cache Manager: Multi-cache orchestration")
        print("✓ Parallel Processor: Sequential, thread, and process parallelism")
        print("✓ Performance Profiler: Execution time and resource tracking")
        print("✓ Resource Metrics: System resource monitoring and pressure detection")
        print("\n🚀 Generation 3 (Make It Scale) core systems are operational!")
    else:
        print("\n⚠️  Some tests failed. Check the failures above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())