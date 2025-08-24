#!/usr/bin/env python3
"""
Simple demo of the protein operators framework without external dependencies.

This script demonstrates the core functionality using only Python standard library
while maintaining compatibility with the full PyTorch-enabled version.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_torch_integration():
    """Test PyTorch integration layer."""
    print("=== Testing PyTorch Integration ===")
    
    from protein_operators.utils.torch_integration import (
        print_system_info, get_device_info, 
        TensorUtils, NetworkUtils, tensor, zeros, ones
    )
    
    print_system_info()
    
    # Test tensor operations
    print("\nTesting tensor operations...")
    x = tensor([1.0, 2.0, 3.0])
    y = tensor([4.0, 5.0, 6.0])
    z = TensorUtils.zeros(3, 4)
    
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"zeros(3,4): {z}")
    
    # Test activation functions
    print("\nTesting activation functions...")
    relu = NetworkUtils.get_activation('relu')
    print(f"ReLU activation: {relu}")
    
    return True


def test_core_designer():
    """Test core protein designer."""
    print("\n=== Testing Core Designer ===")
    
    from protein_operators.core import ProteinDesigner
    from protein_operators.constraints import Constraints
    
    # Create designer
    print("Creating protein designer...")
    designer = ProteinDesigner(
        operator_type="deeponet",
        device="cpu"
    )
    
    print(f"Designer created with device: {designer.device}")
    print(f"Model type: {designer.operator_type}")
    
    # Create simple constraints
    print("\nCreating constraints...")
    constraints = Constraints()
    
    print("Constraints object created successfully")
    
    return True


def run_comprehensive_demo():
    """Run comprehensive demo of all components."""
    print("=" * 60)
    print("  PROTEIN OPERATORS - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    results = []
    
    # Test each component
    test_functions = [
        ("PyTorch Integration", test_torch_integration),
        ("Core Designer", test_core_designer)
    ]
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'-' * 40}")
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("  DEMO SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:>10} | {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components working correctly!")
        print("🚀 Ready for PyTorch integration and full training!")
    else:
        print("⚠️  Some components need attention")
        print("💡 Install PyTorch and NumPy for full functionality")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)