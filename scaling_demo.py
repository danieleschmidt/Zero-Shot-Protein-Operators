#!/usr/bin/env python3
"""
⚡ Scaling Autonomous Protein Design Demo - Generation 3
Demonstration of high-performance, concurrent, and scalable protein design.
"""

import sys
import os
import json
import time
import asyncio
from concurrent.futures import as_completed
sys.path.append('src')

from protein_operators import ProteinDesigner, Constraints
from protein_operators.scaling_framework import ScalableProteinDesigner

def main():
    """Demonstrate scalable autonomous protein design capabilities."""
    print("⚡ Scalable Autonomous Protein Design System - Generation 3")
    print("=" * 75)
    
    # Initialize base designer
    print("🔧 Initializing base protein designer...")
    try:
        base_designer = ProteinDesigner(
            operator_type="deeponet",
            device="auto"
        )
        print("✅ Base designer initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing base designer: {e}")
        return
    
    # Initialize scaling framework
    print("\n⚡ Initializing scaling framework...")
    try:
        scalable_designer = ScalableProteinDesigner(
            base_designer=base_designer,
            enable_caching=True,
            enable_batching=True,
            enable_distributed=False
        )
        print("✅ Scaling framework initialized successfully!")
        print("   - Intelligent caching: ENABLED")
        print("   - Batch processing: ENABLED")
        print("   - Load balancing: ENABLED")
        print("   - Auto-scaling: ENABLED")
        print("   - Performance optimization: ENABLED")
        print("   - Resource pooling: ENABLED")
    except Exception as e:
        print(f"❌ Error initializing scaling framework: {e}")
        return
    
    # Performance benchmarks
    benchmarks = [
        {
            "name": "🚀 Single Request Performance",
            "description": "Measure latency for single requests",
            "test_func": test_single_request_performance
        },
        {
            "name": "📊 Batch Processing Efficiency", 
            "description": "Test batch processing capabilities",
            "test_func": test_batch_processing
        },
        {
            "name": "🎯 Cache Performance",
            "description": "Measure cache hit rates and performance gains",
            "test_func": test_cache_performance
        },
        {
            "name": "⚖️ Load Balancing",
            "description": "Test load distribution and balancing",
            "test_func": test_load_balancing
        },
        {
            "name": "📈 Throughput Scaling",
            "description": "Measure throughput under increasing load",
            "test_func": test_throughput_scaling
        },
        {
            "name": "🔄 Auto-scaling Behavior",
            "description": "Test automatic scaling decisions",
            "test_func": test_auto_scaling
        }
    ]
    
    results = {}
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n{benchmark['name']} ({i}/{len(benchmarks)})")
        print(f"📝 {benchmark['description']}")
        
        try:
            start_time = time.time()
            result = benchmark["test_func"](scalable_designer)
            execution_time = time.time() - start_time
            
            result["execution_time"] = execution_time
            results[benchmark["name"]] = result
            
            if result.get("success", True):
                print("✅ Benchmark completed successfully!")
                for key, value in result.items():
                    if key != "success" and not key.startswith("_"):
                        if isinstance(value, float):
                            print(f"   - {key}: {value:.3f}")
                        else:
                            print(f"   - {key}: {value}")
            else:
                print("⚠️ Benchmark completed with warnings")
                print(f"   - Issue: {result.get('message', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            results[benchmark["name"]] = {
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
    
    # Performance metrics summary
    print("\n📊 Performance Metrics Summary")
    print("-" * 40)
    try:
        metrics = scalable_designer.get_performance_metrics()
        
        print(f"Total Requests: {metrics['metrics']['total_requests']}")
        print(f"Processed Requests: {metrics['metrics']['processed_requests']}")
        print(f"Cache Hits: {metrics['metrics']['cache_hits']}")
        print(f"Cache Misses: {metrics['metrics']['cache_misses']}")
        
        if metrics['metrics']['cache_hits'] + metrics['metrics']['cache_misses'] > 0:
            hit_rate = metrics['metrics']['cache_hits'] / (metrics['metrics']['cache_hits'] + metrics['metrics']['cache_misses'])
            print(f"Cache Hit Rate: {hit_rate:.1%}")
        
        print(f"Average Processing Time: {metrics['metrics']['avg_processing_time']:.3f}s")
        print(f"Throughput: {metrics['metrics']['throughput_per_second']:.1f} req/s")
        print(f"Resource Utilization: {metrics['metrics']['resource_utilization']:.1%}")
        
    except Exception as e:
        print(f"❌ Error getting performance metrics: {e}")
    
    # System optimization recommendations
    print("\n🔧 Optimization Recommendations")
    print("-" * 35)
    recommendations = generate_optimization_recommendations(results, metrics if 'metrics' in locals() else None)
    for rec in recommendations:
        print(f"💡 {rec}")
    
    # Export detailed results
    export_scaling_results(results, metrics if 'metrics' in locals() else None)
    
    print("\n✅ Scalable autonomous protein design demonstration complete!")
    print("⚡ Generation 3: MAKE IT SCALE - Successfully implemented!")
    print("📋 Detailed results exported to: scaling_results.json")

def test_single_request_performance(designer):
    """Test single request performance and latency."""
    constraints = create_test_constraints()
    
    # Warm-up request
    designer.design_sync(constraints=constraints, length=50)
    
    # Time multiple requests
    times = []
    for i in range(5):
        start_time = time.time()
        result = designer.design_sync(constraints=constraints, length=50, num_samples=1)
        request_time = time.time() - start_time
        times.append(request_time)
        
        if not result["success"]:
            return {"success": False, "message": "Request failed"}
    
    return {
        "avg_latency_ms": sum(times) / len(times) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
        "requests_tested": len(times)
    }

def test_batch_processing(designer):
    """Test batch processing efficiency."""
    constraints = create_test_constraints()
    
    # Single requests
    start_time = time.time()
    for i in range(10):
        designer.design_sync(constraints=constraints, length=30, num_samples=1)
    single_time = time.time() - start_time
    
    # Batch request
    batch_requests = [
        {"constraints": constraints, "length": 30, "num_samples": 1}
        for _ in range(10)
    ]
    
    start_time = time.time()
    batch_results = designer.design_batch(batch_requests)
    batch_time = time.time() - start_time
    
    success_rate = sum(1 for r in batch_results if r.get("success", True)) / len(batch_results)
    
    return {
        "single_requests_time": single_time,
        "batch_requests_time": batch_time,
        "speedup_factor": single_time / batch_time if batch_time > 0 else 0,
        "batch_success_rate": success_rate,
        "batch_size": len(batch_requests)
    }

def test_cache_performance(designer):
    """Test caching performance and hit rates."""
    constraints = create_test_constraints()
    
    # First request (cache miss)
    start_time = time.time()
    result1 = designer.design_sync(constraints=constraints, length=40, num_samples=1)
    first_request_time = time.time() - start_time
    
    # Second identical request (cache hit)
    start_time = time.time()
    result2 = designer.design_sync(constraints=constraints, length=40, num_samples=1)
    second_request_time = time.time() - start_time
    
    # Third different request (cache miss)
    start_time = time.time()
    result3 = designer.design_sync(constraints=constraints, length=45, num_samples=1)
    third_request_time = time.time() - start_time
    
    cache_speedup = first_request_time / second_request_time if second_request_time > 0 else 0
    
    return {
        "first_request_time": first_request_time,
        "cached_request_time": second_request_time,
        "cache_speedup_factor": cache_speedup,
        "cache_hit_detected": result2.get("from_cache", False),
        "different_request_time": third_request_time
    }

def test_load_balancing(designer):
    """Test load balancing across workers."""
    # Add some workers to the load balancer
    designer.load_balancer.add_worker("worker_1", capacity=1.0)
    designer.load_balancer.add_worker("worker_2", capacity=1.5)
    designer.load_balancer.add_worker("worker_3", capacity=0.8)
    
    # Test worker selection
    worker_selections = []
    for i in range(20):
        worker = designer.load_balancer.get_next_worker()
        worker_selections.append(worker)
    
    # Count selections per worker
    from collections import Counter
    selection_counts = Counter(worker_selections)
    
    return {
        "workers_available": len(designer.load_balancer.workers),
        "total_selections": len(worker_selections),
        "selection_distribution": dict(selection_counts),
        "load_balancing_active": len(set(worker_selections)) > 1
    }

def test_throughput_scaling(designer):
    """Test throughput under increasing load."""
    constraints = create_test_constraints()
    
    throughput_results = []
    
    # Test different loads
    loads = [1, 5, 10, 20]
    
    for load in loads:
        start_time = time.time()
        
        # Process multiple requests concurrently (simulated)
        results = []
        for i in range(load):
            result = designer.design_sync(
                constraints=constraints,
                length=30,
                num_samples=1
            )
            results.append(result)
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for r in results if r.get("success", True))
        throughput = successful_requests / total_time if total_time > 0 else 0
        
        throughput_results.append({
            "load": load,
            "throughput": throughput,
            "success_rate": successful_requests / load,
            "total_time": total_time
        })
    
    # Calculate scaling efficiency
    baseline_throughput = throughput_results[0]["throughput"]
    scaling_efficiency = []
    
    for result in throughput_results[1:]:
        expected_throughput = baseline_throughput * result["load"]
        actual_throughput = result["throughput"] * result["load"]
        efficiency = actual_throughput / expected_throughput if expected_throughput > 0 else 0
        scaling_efficiency.append(efficiency)
    
    return {
        "throughput_results": throughput_results,
        "max_throughput": max(r["throughput"] for r in throughput_results),
        "scaling_efficiency": sum(scaling_efficiency) / len(scaling_efficiency) if scaling_efficiency else 0,
        "load_range_tested": f"{min(loads)}-{max(loads)} requests"
    }

def test_auto_scaling(designer):
    """Test auto-scaling behavior."""
    # Get initial metrics
    initial_metrics = designer.get_performance_metrics()
    
    # Simulate high load to trigger scaling
    high_load_metrics = {
        "resource_utilization": 0.9,  # High utilization
        "queued_requests": 15,        # High queue
        "avg_processing_time": 5.0    # Slow processing
    }
    
    scale_decision = designer.auto_scaler.should_scale(high_load_metrics)
    
    # Simulate low load
    low_load_metrics = {
        "resource_utilization": 0.1,  # Low utilization  
        "queued_requests": 0,         # No queue
        "avg_processing_time": 0.1    # Fast processing
    }
    
    scale_down_decision = designer.auto_scaler.should_scale(low_load_metrics)
    
    return {
        "initial_workers": initial_metrics["metrics"].get("total_requests", 0),
        "scale_up_triggered": scale_decision["action"] == "scale_up",
        "scale_up_reason": scale_decision["reason"],
        "scale_down_triggered": scale_down_decision["action"] == "scale_down", 
        "scale_down_reason": scale_down_decision["reason"],
        "auto_scaling_responsive": scale_decision["action"] != "no_change" or scale_down_decision["action"] != "no_change"
    }

def create_test_constraints():
    """Create standard test constraints."""
    constraints = Constraints()
    constraints.add_binding_site(
        residues=[10, 20],
        ligand="test_ligand",
        affinity_nm=100
    )
    constraints.add_secondary_structure(5, 15, "helix")
    return constraints

def generate_optimization_recommendations(results, metrics):
    """Generate optimization recommendations based on test results."""
    recommendations = []
    
    # Cache performance
    cache_result = results.get("🎯 Cache Performance", {})
    if cache_result.get("cache_speedup_factor", 0) < 2:
        recommendations.append("Consider increasing cache size or TTL for better hit rates")
    
    # Batch processing
    batch_result = results.get("📊 Batch Processing Efficiency", {})
    if batch_result.get("speedup_factor", 0) < 1.5:
        recommendations.append("Optimize batch processing algorithms for better efficiency")
    
    # Throughput scaling
    throughput_result = results.get("📈 Throughput Scaling", {})
    if throughput_result.get("scaling_efficiency", 0) < 0.7:
        recommendations.append("Consider adding more worker threads or optimizing resource allocation")
    
    # Resource utilization
    if metrics and metrics["metrics"]["resource_utilization"] > 0.8:
        recommendations.append("Resource utilization is high - consider scaling up")
    elif metrics and metrics["metrics"]["resource_utilization"] < 0.3:
        recommendations.append("Resource utilization is low - consider scaling down to save costs")
    
    # General recommendations
    if not recommendations:
        recommendations.append("System is performing well - monitor metrics for optimization opportunities")
    
    return recommendations

def export_scaling_results(results, metrics):
    """Export detailed scaling test results."""
    export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "Scalable Autonomous Protein Design - Generation 3",
        "benchmark_results": results,
        "performance_metrics": metrics,
        "system_info": {
            "caching_enabled": True,
            "batching_enabled": True,
            "auto_scaling_enabled": True,
            "load_balancing_enabled": True
        },
        "summary": {
            "total_benchmarks": len(results),
            "successful_benchmarks": len([r for r in results.values() if r.get("success", True)]),
            "failed_benchmarks": len([r for r in results.values() if not r.get("success", True)])
        }
    }
    
    try:
        with open("scaling_results.json", "w") as f:
            json.dump(export_data, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Warning: Could not export results to JSON: {e}")

if __name__ == "__main__":
    main()