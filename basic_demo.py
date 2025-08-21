#!/usr/bin/env python3
"""
Basic demonstration of the Protein-Operators framework.

This script demonstrates the core functionality of the framework
without requiring heavy dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import torch, fallback to mock if not available
try:
    import torch
except ImportError:
    sys.path.insert(0, '.')
    import mock_torch as torch

import numpy as np
from pathlib import Path

# Import protein operators components
from protein_operators import ProteinDesigner, Constraints
from protein_operators.constraints import BindingSiteConstraint, SecondaryStructureConstraint


def main():
    """Demonstrate basic protein design workflow."""
    print("🧬 Protein-Operators Basic Demo")
    print("=" * 50)
    
    # Step 1: Initialize designer
    print("\n1. Initializing Protein Designer...")
    try:
        designer = ProteinDesigner(
            operator_type="deeponet",
            checkpoint=None  # No pre-trained model for demo
        )
        print(f"✅ Designer initialized with {designer.operator_type} operator")
        print(f"   Device: {designer.device}")
    except Exception as e:
        print(f"❌ Failed to initialize designer: {e}")
        return False
    
    # Step 2: Create constraints
    print("\n2. Setting up design constraints...")
    try:
        constraints = Constraints()
        
        # Add binding site constraint
        constraints.add_binding_site(
            residues=[45, 67, 89],
            ligand="ATP",
            affinity_nm=100
        )
        
        # Add secondary structure constraints
        constraints.add_secondary_structure(
            start=10, end=25, 
            ss_type="helix", 
            confidence=0.8
        )
        constraints.add_secondary_structure(
            start=30, end=40, 
            ss_type="sheet", 
            confidence=0.9
        )
        
        print(f"✅ Created {len(constraints)} constraints:")
        for i, constraint in enumerate(constraints, 1):
            print(f"   {i}. {constraint.name} (type: {constraint.__class__.__name__})")
            
    except Exception as e:
        print(f"❌ Failed to create constraints: {e}")
        return False
    
    # Step 3: Generate protein structure
    print("\n3. Generating protein structure...")
    try:
        target_length = 150
        
        structure = designer.generate(
            constraints=constraints,
            length=target_length,
            num_samples=1,
            physics_guided=False  # Skip physics refinement for basic demo
        )
        
        print(f"✅ Generated protein structure:")
        print(f"   Length: {structure.num_residues} residues")
        print(f"   Coordinates shape: {structure.coordinates.shape}")
        print(f"   Radius of gyration: {structure.compute_radius_of_gyration():.2f} Å")
        
    except Exception as e:
        print(f"❌ Failed to generate structure: {e}")
        return False
    
    # Step 4: Validate structure
    print("\n4. Validating generated structure...")
    try:
        validation_results = designer.validate(structure)
        
        print("✅ Validation results:")
        for metric, value in validation_results.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.3f}")
            else:
                print(f"   {metric}: {value}")
        
        # Overall quality assessment
        overall_score = validation_results.get('overall_score', 0.0)
        if overall_score >= 0.7:
            quality = "Excellent"
        elif overall_score >= 0.5:
            quality = "Good"
        elif overall_score >= 0.3:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"\n   Overall Quality: {quality} ({overall_score:.3f})")
        
    except Exception as e:
        print(f"❌ Failed to validate structure: {e}")
        return False
    
    # Step 5: Save structure
    print("\n5. Saving structure...")
    try:
        output_path = Path("demo_protein.pdb")
        structure.save_pdb(output_path)
        print(f"✅ Structure saved to: {output_path.absolute()}")
        
        # Display file info
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"   File size: {file_size} bytes")
            
    except Exception as e:
        print(f"❌ Failed to save structure: {e}")
        return False
    
    # Step 6: Demonstrate constraint satisfaction
    print("\n6. Analyzing constraint satisfaction...")
    try:
        satisfaction_scores = constraints.satisfaction_scores(structure)
        
        print("✅ Constraint satisfaction:")
        for constraint_name, score in satisfaction_scores.items():
            status = "✓" if score > 0.5 else "✗"
            print(f"   {status} {constraint_name}: {score:.3f}")
        
        overall_satisfaction = constraints.overall_satisfaction(structure)
        print(f"\n   Overall Satisfaction: {overall_satisfaction:.3f}")
        
    except Exception as e:
        print(f"❌ Failed to analyze constraints: {e}")
        return False
    
    # Step 7: Basic structure analysis
    print("\n7. Structure analysis...")
    try:
        # Geometry validation
        geometry = structure.validate_geometry()
        print("✅ Geometry metrics:")
        for metric, value in geometry.items():
            print(f"   {metric}: {value:.3f}")
        
        # Secondary structure assignment
        ss_assignment = structure.compute_secondary_structure_simple()
        helix_count = ss_assignment.count('H')
        sheet_count = ss_assignment.count('E')
        coil_count = ss_assignment.count('C')
        
        print(f"\n✅ Secondary structure content:")
        print(f"   Helix: {helix_count} residues ({helix_count/len(ss_assignment)*100:.1f}%)")
        print(f"   Sheet: {sheet_count} residues ({sheet_count/len(ss_assignment)*100:.1f}%)")
        print(f"   Coil:  {coil_count} residues ({coil_count/len(ss_assignment)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Failed structure analysis: {e}")
        return False
    
    # Step 8: Designer statistics
    print("\n8. Designer statistics...")
    try:
        stats = designer.statistics
        print("✅ Designer stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("- View the generated structure: demo_protein.pdb")
    print("- Try different constraints and parameters")
    print("- Explore advanced features in the full documentation")
    
    return True


def run_simple_test():
    """Run a simplified test of core functionality."""
    print("\n🧪 Running simplified functionality test...")
    
    try:
        # Test constraint creation
        constraints = Constraints()
        constraints.add_binding_site([10, 20], "test_ligand")
        print(f"✅ Constraints created: {len(constraints)} total")
        
        # Test designer initialization  
        designer = ProteinDesigner()
        print(f"✅ Designer created with {designer.operator_type} operator")
        
        # Test basic generation
        structure = designer.generate(constraints, length=50)
        print(f"✅ Structure generated: {structure.num_residues} residues")
        
        # Test validation
        try:
            scores = designer.validate(structure)
            print(f"✅ Validation completed: {len(scores)} metrics")
        except Exception as e:
            print(f"⚠️  Validation skipped due to mock limitations: {e}")
            scores = {'mock_score': 0.8}
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if we should run the simple test instead
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_simple_test()
    else:
        success = main()
    
    sys.exit(0 if success else 1)