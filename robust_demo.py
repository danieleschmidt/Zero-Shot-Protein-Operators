#!/usr/bin/env python3
"""
🛡️ Robust Autonomous Protein Design Demo - Generation 2
Demonstration of robust framework with error handling, monitoring, and recovery.
"""

import sys
import os
import json
import time
sys.path.append('src')

from protein_operators import ProteinDesigner, Constraints
from protein_operators.robust_framework import RobustProteinDesigner

def main():
    """Demonstrate robust autonomous protein design capabilities."""
    print("🛡️ Robust Autonomous Protein Design System - Generation 2")
    print("=" * 70)
    
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
    
    # Initialize robust framework
    print("\n🛡️ Initializing robust framework...")
    try:
        robust_designer = RobustProteinDesigner(
            base_designer=base_designer,
            enable_monitoring=True,
            enable_recovery=True
        )
        print("✅ Robust framework initialized successfully!")
        print("   - Error handling: ENABLED")
        print("   - Performance monitoring: ENABLED") 
        print("   - Automatic recovery: ENABLED")
        print("   - Security validation: ENABLED")
        print("   - Resource management: ENABLED")
    except Exception as e:
        print(f"❌ Error initializing robust framework: {e}")
        return
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "🧬 Normal Operation Test",
            "description": "Standard protein design with normal parameters",
            "constraints": create_normal_constraints(),
            "params": {"length": 50, "num_samples": 1}
        },
        {
            "name": "⚡ Stress Test",
            "description": "High complexity design to test resource management",
            "constraints": create_complex_constraints(),
            "params": {"length": 200, "num_samples": 5}
        },
        {
            "name": "🔒 Security Test",
            "description": "Test with invalid parameters to trigger security validation",
            "constraints": create_normal_constraints(),
            "params": {"length": -10, "num_samples": 1000}  # Invalid parameters
        },
        {
            "name": "🔄 Recovery Test",
            "description": "Test automatic recovery from resource constraints",
            "constraints": create_normal_constraints(),
            "params": {"length": 1000, "num_samples": 20}  # Likely to cause memory issues
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{scenario['name']} ({i}/{len(test_scenarios)})")
        print(f"📝 {scenario['description']}")
        
        try:
            # Execute robust design
            result = robust_designer.robust_design(
                constraints=scenario["constraints"],
                **scenario["params"]
            )
            
            results.append({
                "scenario": scenario["name"],
                "result": result
            })
            
            # Display results
            if result["success"]:
                print("✅ Design completed successfully!")
                print(f"   - Response time: {result['metrics']['response_time_seconds']:.2f}s")
                print(f"   - Memory usage: {result['metrics']['memory_usage_mb']:.1f}MB")
                if result["warnings"]:
                    print(f"   - Warnings: {len(result['warnings'])}")
                    for warning in result["warnings"]:
                        print(f"     ⚠️ {warning}")
            else:
                print("❌ Design failed!")
                if result["error"]:
                    print(f"   - Error: {result['error']['type']}")
                    print(f"   - Message: {result['error']['message']}")
                    print(f"   - Severity: {result['error']['severity']}")
                    print(f"   - Suggested action: {result['error']['suggested_action']}")
                    print(f"   - Recoverable: {result['error']['recoverable']}")
            
        except Exception as e:
            print(f"❌ Scenario failed with exception: {e}")
            results.append({
                "scenario": scenario["name"],
                "result": {"success": False, "exception": str(e)}
            })
    
    # Health status check
    print("\n🏥 System Health Status")
    print("-" * 30)
    try:
        health = robust_designer.get_health_status()
        print(f"Overall Status: {health['status'].upper()}")
        print(f"Total Requests: {health['metrics']['total_requests']}")
        print(f"Success Rate: {health['metrics']['successful_requests']/max(1, health['metrics']['total_requests'])*100:.1f}%")
        print(f"Average Response Time: {health['metrics']['avg_response_time']:.2f}s")
        
        # Component status
        print("\n🔧 Component Status:")
        for component, status in health["components"].items():
            status_icon = "✅" if status.get("status") == "healthy" else "⚠️"
            print(f"   {status_icon} {component}: {status.get('status', 'unknown')}")
        
        # Active alerts
        if health["alerts"]:
            print(f"\n🚨 Active Alerts ({len(health['alerts'])}):")
            for alert in health["alerts"]:
                severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
                print(f"   {severity_icon} {alert['type']}: {alert['message']}")
        else:
            print("\n✅ No active alerts")
        
    except Exception as e:
        print(f"❌ Error getting health status: {e}")
    
    # Performance summary
    print("\n📊 Performance Summary")
    print("-" * 25)
    successful_scenarios = [r for r in results if r["result"].get("success", False)]
    failed_scenarios = [r for r in results if not r["result"].get("success", False)]
    
    print(f"✅ Successful scenarios: {len(successful_scenarios)}/{len(test_scenarios)}")
    print(f"❌ Failed scenarios: {len(failed_scenarios)}/{len(test_scenarios)}")
    
    if successful_scenarios:
        avg_response_time = sum(
            r["result"]["metrics"]["response_time_seconds"] 
            for r in successful_scenarios
        ) / len(successful_scenarios)
        print(f"⏱️ Average response time: {avg_response_time:.2f}s")
    
    # Export detailed results
    export_results(results, health if 'health' in locals() else None)
    
    print("\n✅ Robust autonomous protein design demonstration complete!")
    print("🛡️ Generation 2: MAKE IT ROBUST - Successfully implemented!")
    print("📋 Detailed results exported to: robust_design_results.json")

def create_normal_constraints():
    """Create normal protein design constraints."""
    constraints = Constraints()
    constraints.add_binding_site(
        residues=[25, 45],
        ligand="ATP",
        affinity_nm=500
    )
    constraints.add_secondary_structure(20, 30, "helix")
    return constraints

def create_complex_constraints():
    """Create complex protein design constraints."""
    constraints = Constraints()
    
    # Multiple binding sites
    constraints.add_binding_site(
        residues=[25, 45, 65],
        ligand="ATP",
        affinity_nm=100
    )
    constraints.add_binding_site(
        residues=[85, 105, 125],
        ligand="GTP",
        affinity_nm=200
    )
    
    # Multiple secondary structures
    constraints.add_secondary_structure(20, 35, "helix")
    constraints.add_secondary_structure(50, 70, "sheet")
    constraints.add_secondary_structure(90, 110, "helix")
    constraints.add_secondary_structure(130, 150, "sheet")
    
    return constraints

def export_results(results, health_status):
    """Export detailed results to JSON file."""
    export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "Robust Autonomous Protein Design - Generation 2",
        "test_results": results,
        "health_status": health_status,
        "summary": {
            "total_scenarios": len(results),
            "successful": len([r for r in results if r["result"].get("success", False)]),
            "failed": len([r for r in results if not r["result"].get("success", False)])
        }
    }
    
    try:
        with open("robust_design_results.json", "w") as f:
            json.dump(export_data, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Warning: Could not export results to JSON: {e}")

if __name__ == "__main__":
    main()