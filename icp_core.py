#!/usr/bin/env python3
"""
Core ICP (Iterative Closest Point) Algorithm
Pure numpy implementation without visualization dependencies
"""

import numpy as np


def find_nearest_neighbors(source, target):
    """Find nearest neighbor correspondences between source and target points"""
    correspondences = []
    for i, s_point in enumerate(source):
        distances = [np.linalg.norm(s_point - t_point) for t_point in target]
        j = np.argmin(distances)
        correspondences.append((i, j))
    return correspondences


def compute_transformation(source, target):
    """Compute optimal rotation and translation using SVD"""
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    
    centered_source = source - centroid_source
    centered_target = target - centroid_target
    
    H = centered_source.T @ centered_target
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_target - R @ centroid_source
    
    return R, t


def run_icp_algorithm(source_points, target_points, max_iterations=10, tolerance=0.01, verbose=False):
    """
    Run ICP algorithm and return convergence data
    
    Args:
        source_points: Nx3 array of source points
        target_points: Mx3 array of target points  
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance for mean error
        verbose: Print progress
        
    Returns:
        converged: Boolean indicating if algorithm converged
        states: List of dictionaries containing state at each iteration
    """
    
    # Track all states for animation
    states = []
    current_points = source_points.copy()
    
    for iteration in range(max_iterations):
        # Find nearest neighbors
        correspondences = find_nearest_neighbors(current_points, target_points)
        
        # Get matched target points
        target_matched = np.array([target_points[j] for i, j in correspondences])
        
        # Calculate current error and energy
        distances = np.linalg.norm(current_points - target_matched, axis=1)
        error = np.mean(distances)
        energy = np.sum(distances ** 2)
        
        if verbose:
            print(f"Iteration {iteration}: Error = {error:.6f}, Energy = {energy:.6f}")
        
        # Store state before transformation
        states.append({
            'iteration': iteration,
            'points': current_points.copy(),
            'correspondences': correspondences.copy(),
            'error': error,
            'energy': energy,
            'target_matched': target_matched.copy()
        })
        
        # Check convergence
        if error < tolerance:
            if verbose:
                print(f"ICP converged at iteration {iteration} with error {error:.6f}")
            return True, states
        
        # Compute and apply transformation
        R, t = compute_transformation(current_points, target_matched)
        current_points = (R @ current_points.T).T + t
    
    if verbose:
        print(f"ICP did not converge after {max_iterations} iterations. Final error: {error:.6f}")
    return False, states


def create_test_data(n_points=9, rotation=0.25, translation=None):
    """Create test point clouds for ICP"""
    if translation is None:
        translation = np.array([0.3, 0.2, 0])
    
    # Create a star-like pattern
    source = np.array([
        [-1.2, 0.8, 0],
        [-0.4, 1.0, 0],
        [0.4, 0.9, 0],
        [1.2, 0.6, 0],
        [-0.8, 0.0, 0],
        [0.0, 0.0, 0],
        [0.8, -0.2, 0],
        [-0.6, -0.8, 0],
        [0.6, -0.9, 0],
    ])
    
    # Apply rotation and translation
    cos_a = np.cos(rotation)
    sin_a = np.sin(rotation)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    target = (rotation_matrix @ source.T).T + translation
    
    return source, target