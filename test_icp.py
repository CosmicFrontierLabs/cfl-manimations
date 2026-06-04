#!/usr/bin/env python3
"""
Test script for ICP algorithm convergence
"""

import numpy as np
from icp_core import run_icp_algorithm, create_test_data


def test_icp_convergence():
    """Test ICP with various transformations"""
    
    test_cases = [
        {"rotation": 0.1, "translation": [0.1, 0.1, 0], "name": "Small transform"},
        {"rotation": 0.25, "translation": [0.3, 0.2, 0], "name": "Medium transform"},
        {"rotation": 0.5, "translation": [0.5, 0.4, 0], "name": "Large transform"},
        {"rotation": 0.8, "translation": [0.7, 0.5, 0], "name": "Very large transform"},
    ]
    
    print("=" * 60)
    print("Testing ICP Algorithm Convergence")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"  Rotation: {test['rotation']:.2f} rad ({np.degrees(test['rotation']):.1f}°)")
        print(f"  Translation: {test['translation']}")
        
        source, target = create_test_data(
            rotation=test['rotation'],
            translation=np.array(test['translation'])
        )
        
        converged, states = run_icp_algorithm(
            source, target,
            max_iterations=20,
            tolerance=0.01,
            verbose=False
        )
        
        if converged:
            final_error = states[-1]['error']
            print(f"  ✓ CONVERGED in {len(states)} iterations")
            print(f"  Final error: {final_error:.6f}")
        else:
            final_error = states[-1]['error'] if states else float('inf')
            print(f"  ✗ DID NOT CONVERGE after {len(states)} iterations")
            print(f"  Final error: {final_error:.6f}")
    
    print("\n" + "=" * 60)
    
    # Test with noise
    print("\nTesting with noise added to target points...")
    source, target = create_test_data(rotation=0.25, translation=[0.3, 0.2, 0])
    
    # Add small noise to target
    noise_level = 0.05
    target_noisy = target + np.random.normal(0, noise_level, target.shape)
    
    converged, states = run_icp_algorithm(
        source, target_noisy,
        max_iterations=20,
        tolerance=0.05,  # Higher tolerance for noisy data
        verbose=True
    )
    
    if converged:
        print(f"✓ Converged with noisy data (noise level: {noise_level})")
    else:
        print(f"✗ Failed to converge with noisy data")
    
    # Find best parameters for animation
    print("\n" + "=" * 60)
    print("Finding optimal parameters for animation...")
    
    for rot in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for trans_scale in [0.1, 0.2, 0.3, 0.4]:
            trans = np.array([trans_scale, trans_scale * 0.7, 0])
            source, target = create_test_data(rotation=rot, translation=trans)
            converged, states = run_icp_algorithm(
                source, target,
                max_iterations=10,
                tolerance=0.01,
                verbose=False
            )
            
            if converged and len(states) >= 3 and len(states) <= 6:
                print(f"\nGood animation parameters found:")
                print(f"  Rotation: {rot:.2f} rad")
                print(f"  Translation: [{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}]")
                print(f"  Converges in {len(states)} iterations")
                break
        else:
            continue
        break


if __name__ == "__main__":
    test_icp_convergence()