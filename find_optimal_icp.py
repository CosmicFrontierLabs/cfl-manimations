#!/usr/bin/env python3
"""
Find optimal ICP parameters that converge in 4-6 iterations
"""

import numpy as np
from icp_core import run_icp_algorithm, create_test_data


def find_optimal_parameters():
    """Find parameters that give good convergence in 4-6 iterations"""
    
    print("=" * 60)
    print("Finding optimal ICP parameters for animation")
    print("Target: 4-6 iterations to convergence")
    print("=" * 60)
    
    best_params = []
    
    # Try different rotation and translation combinations
    for rot in np.arange(0.3, 0.7, 0.05):
        for trans_scale in np.arange(0.3, 0.8, 0.1):
            trans = np.array([trans_scale, trans_scale * 0.8, 0])
            
            source, target = create_test_data(rotation=rot, translation=trans)
            
            # Test with tighter tolerance to force more iterations
            converged, states = run_icp_algorithm(
                source, target,
                max_iterations=20,
                tolerance=0.001,  # Tighter tolerance
                verbose=False
            )
            
            if converged and 4 <= len(states) <= 6:
                final_error = states[-1]['error']
                energy = states[-1]['energy']
                
                param_info = {
                    'rotation': rot,
                    'translation': trans.tolist(),
                    'iterations': len(states),
                    'final_error': final_error,
                    'final_energy': energy
                }
                best_params.append(param_info)
                
                print(f"\n✓ Good parameters found:")
                print(f"  Rotation: {rot:.3f} rad ({np.degrees(rot):.1f}°)")
                print(f"  Translation: [{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}]")
                print(f"  Converges in {len(states)} iterations")
                print(f"  Final error: {final_error:.6f}")
                print(f"  Final energy: {energy:.6f}")
    
    if not best_params:
        print("\nNo parameters found with 4-6 iterations. Trying wider search...")
        
        # Try with looser initial positioning but tighter convergence
        for rot in np.arange(0.4, 1.0, 0.1):
            for trans_scale in np.arange(0.5, 1.2, 0.1):
                trans = np.array([trans_scale, trans_scale * 0.6, 0])
                
                source, target = create_test_data(rotation=rot, translation=trans)
                
                converged, states = run_icp_algorithm(
                    source, target,
                    max_iterations=20,
                    tolerance=0.0001,  # Very tight tolerance
                    verbose=False
                )
                
                if converged and 4 <= len(states) <= 8:
                    final_error = states[-1]['error']
                    energy = states[-1]['energy']
                    
                    param_info = {
                        'rotation': rot,
                        'translation': trans.tolist(),
                        'iterations': len(states),
                        'final_error': final_error,
                        'final_energy': energy
                    }
                    best_params.append(param_info)
                    
                    print(f"\n✓ Parameters found:")
                    print(f"  Rotation: {rot:.3f} rad ({np.degrees(rot):.1f}°)")
                    print(f"  Translation: [{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}]")
                    print(f"  Converges in {len(states)} iterations")
                    print(f"  Final error: {final_error:.6f}")
    
    # Find the best one (prefer exactly 5 iterations)
    if best_params:
        # Sort by how close to 5 iterations
        best_params.sort(key=lambda x: abs(x['iterations'] - 5))
        best = best_params[0]
        
        print("\n" + "=" * 60)
        print("BEST PARAMETERS FOR ANIMATION:")
        print(f"  Rotation: {best['rotation']:.3f} rad")
        print(f"  Translation: {best['translation']}")
        print(f"  Iterations: {best['iterations']}")
        print(f"  Final error: {best['final_error']:.6f}")
        print("=" * 60)
        
        # Test it with verbose output
        print("\nVerifying with verbose output:")
        source, target = create_test_data(
            rotation=best['rotation'],
            translation=np.array(best['translation'])
        )
        converged, states = run_icp_algorithm(
            source, target,
            max_iterations=20,
            tolerance=0.001,
            verbose=True
        )
        
        return best
    else:
        print("\nNo suitable parameters found!")
        return None


if __name__ == "__main__":
    optimal = find_optimal_parameters()
    
    if optimal:
        print("\n" + "=" * 60)
        print("Add these parameters to icp_animation.py:")
        print(f"rotation = {optimal['rotation']:.3f}")
        print(f"translation = np.array({optimal['translation']})")
        print("=" * 60)