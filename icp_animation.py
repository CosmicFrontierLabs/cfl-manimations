#!/usr/bin/env python3
"""
ICP (Iterative Closest Point) Algorithm Animation using Manim
Shows how ICP aligns two point clouds through iterative refinement
"""

from manim import *
import numpy as np
from icp_core import run_icp_algorithm, create_test_data


class Spring(VMobject):
    """Custom spring visualization between two points"""
    def __init__(self, start_point, end_point, n_coils=8, width=0.15, **kwargs):
        super().__init__(**kwargs)
        self.n_coils = n_coils
        self.width = width
        self.start_point = start_point
        self.end_point = end_point
        self.generate_spring()
        
    def generate_spring(self):
        """Generate spring coil path"""
        start = np.array(self.start_point)
        end = np.array(self.end_point)
        
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 0.01:
            self.add(Line(start, end))
            return
            
        direction = direction / length
        
        # Perpendicular vector for coil width
        if abs(direction[1]) < 0.9:
            perp = np.cross(direction, np.array([0, 1, 0]))
        else:
            perp = np.cross(direction, np.array([1, 0, 0]))
        perp = perp / np.linalg.norm(perp) * self.width
        
        # Generate spring points
        points = []
        n_points = self.n_coils * 4  # 4 points per coil
        
        for i in range(n_points + 1):
            t = i / n_points
            base_pos = start + direction * length * t
            
            # Create coil effect
            angle = 2 * np.pi * self.n_coils * t
            offset = perp * np.sin(angle)
            
            points.append(base_pos + offset)
        
        # Create smooth path through points
        self.set_points_smoothly(points)
        
    def update_spring(self, start_point, end_point):
        """Update spring endpoints"""
        self.start_point = start_point
        self.end_point = end_point
        self.clear_points()
        self.generate_spring()
        return self


# ICP algorithm now imported from icp_core.py


class ICPAnimation(Scene):
    def construct(self):
        title = Text("Iterative Closest Point (ICP) Algorithm", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        
        np.random.seed(42)
        
        # Use parameters that converge in 5 iterations for better animation
        rotation = 0.300  # Converges in 5 iterations
        translation = np.array([0.4, 0.32, 0.0])  # Optimal for 5-iteration convergence
        
        # Use create_test_data from icp_core to ensure consistency
        source_points, target_points = create_test_data(
            rotation=rotation,
            translation=translation
        )
        
        # Add extra points BEFORE testing
        extra_points = np.array([
            [0.5, 0.95, 0],  # Close to a target point
            [2.0, 0.2, 0],   # Far right
            [-2.1, 0.3, 0],  # Far left
            [0.3, -1.8, 0],  # Bottom
        ])
        target_with_extras = np.vstack([target_points, extra_points])
        
        # Test ICP convergence with extra points
        print("Testing ICP convergence with extra points...")
        converged, icp_states = run_icp_algorithm(
            source_points.copy(), 
            target_with_extras,  # Include extra points
            max_iterations=10,
            tolerance=0.001  # Tighter tolerance for better animation
        )
        
        if not converged:
            print("WARNING: ICP did not converge properly!")
            # Adjust parameters for better convergence
            rotation = 0.15
            translation = np.array([0.2, 0.1, 0])
            target_points = self.transform_points(source_points, rotation, translation)
            converged, icp_states = run_icp_algorithm(
                source_points.copy(),
                target_points,
                max_iterations=10,
                tolerance=0.1
            )
            assert converged, "ICP failed to converge even with adjusted parameters"
        
        print(f"ICP converged successfully with {len(icp_states)} iterations")
        
        # Now shift everything left for animation display
        source_points = source_points * 0.8 - np.array([2.5, 0, 0])
        # Scale ALL target points including extras
        target_all = target_with_extras * 0.8 - np.array([2.5, 0, 0])
        target_points = target_all  # Use the combined set
        
        source_dots = VGroup(*[
            Dot(point, color=BLUE, radius=0.08).set_z_index(1) for point in source_points
        ])
        target_dots = VGroup(*[
            Dot(point, color=ORANGE if i >= len(source_points) else RED, radius=0.08).set_z_index(1) 
            for i, point in enumerate(target_points)
        ])
        
        source_label = Text("Source", color=BLUE, font_size=20).next_to(source_dots, DOWN)
        target_label = Text("Target (red) + Extras (orange)", color=RED, font_size=18).next_to(target_dots, DOWN)
        
        self.play(
            Create(source_dots),
            Create(target_dots),
            Write(source_label),
            Write(target_label)
        )
        self.wait(1)
        
        # Create energy diagram box
        energy_box = Rectangle(width=4, height=5, color=WHITE).shift(RIGHT * 4.5)
        energy_title = Text("Spring Energy", font_size=20).next_to(energy_box, UP)
        energy_formula = MathTex(r"E = \sum_{i} d_i^2", font_size=18).shift(RIGHT * 4.5 + UP * 2)
        
        # Get max energy from pre-computed states for proper scaling
        max_energy = max([state['energy'] for state in icp_states])
        energy_scale = max(2.0, np.ceil(max_energy))  # Round up to nearest integer
        
        # Create axes for energy plot with proper scaling
        axes = Axes(
            x_range=[0, len(icp_states)-1, 1],
            y_range=[0, energy_scale, energy_scale/5],
            x_length=3,
            y_length=3,
            axis_config={"include_numbers": True, "font_size": 16},
        ).shift(RIGHT * 4.5 + DOWN * 0.5)
        
        x_label = Text("Iteration", font_size=14).next_to(axes, DOWN)
        y_label = Text("Energy", font_size=14).rotate(PI/2).next_to(axes, LEFT)
        
        self.play(
            Create(energy_box),
            Write(energy_title),
            Write(energy_formula),
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        
        iteration_text = Text("Iteration: 0", font_size=20).to_corner(UL)
        error_text = Text("Energy: ---", font_size=20).next_to(iteration_text, DOWN)
        
        self.play(Write(iteration_text), Write(error_text))
        
        moving_dots = source_dots.copy()
        self.play(FadeOut(source_dots), FadeOut(source_label))
        self.add(moving_dots)
        
        step_text = Text("", font_size=20, color=YELLOW).to_edge(DOWN)
        
        # Track energy values for plotting
        energy_values = []
        energy_dots = VGroup()
        energy_line = VMobject()
        
        # Initialize springs (they'll be updated each iteration)
        springs = VGroup()
        
        # Use pre-computed ICP states
        for state_idx, state in enumerate(icp_states[:5]):  # Limit to 5 iterations for animation
            iteration = state['iteration']
            correspondences = state['correspondences']
            energy = state['energy']
            
            # Scale and shift the pre-computed points to match animation space
            state_points = state['points'] * 0.8 - np.array([2.5, 0, 0])
            
            # Remove old springs
            if len(springs) > 0:
                self.remove(springs)
            
            # Create new springs (z_index 2 to be on top of dots)
            springs = VGroup(*[
                Spring(
                    moving_dots[i].get_center(),
                    target_dots[j].get_center(),
                    n_coils=6,
                    width=0.08,
                    color=YELLOW,
                    stroke_width=2,
                    stroke_opacity=0.8
                ).set_z_index(2)
                for i, j in correspondences
            ])
            
            new_step_text = Text("Finding nearest neighbors...", font_size=20, color=YELLOW).to_edge(DOWN)
            self.play(
                Create(springs),
                ReplacementTransform(step_text, new_step_text)
            )
            step_text = new_step_text
            self.wait(0.5)
            
            # Update moving dots positions to match pre-computed state
            for i, dot in enumerate(moving_dots):
                dot.move_to(state_points[i])
            
            current_points = np.array([dot.get_center() for dot in moving_dots])
            target_matched = np.array([target_dots[j].get_center() for i, j in correspondences])
            
            energy_values.append(energy)
            
            # Add energy point to plot (no min() needed, axes scaled properly)
            energy_point = axes.coords_to_point(iteration, energy)
            energy_dot = Dot(energy_point, color=YELLOW, radius=0.06)
            energy_dots.add(energy_dot)
            
            # Update energy line
            if iteration > 0:
                new_line = Line(
                    axes.coords_to_point(iteration-1, energy_values[-2]),
                    energy_point,
                    color=YELLOW,
                    stroke_width=2
                )
                energy_line.add(new_line)
                self.add(new_line)
            
            self.add(energy_dot)
            
            new_step_text = Text("Computing optimal transformation...", font_size=20, color=YELLOW).to_edge(DOWN)
            self.play(ReplacementTransform(step_text, new_step_text))
            step_text = new_step_text
            self.wait(0.3)
            
            # Get next state positions if available
            if state_idx + 1 < len(icp_states):
                next_state = icp_states[state_idx + 1]
                new_positions = next_state['points'] * 0.8 - np.array([2.5, 0, 0])
            else:
                # Final iteration - use current positions
                new_positions = state_points
            
            error = state['error']
            
            new_iteration_text = Text(f"Iteration: {iteration + 1}", font_size=20).to_corner(UL)
            new_error_text = Text(f"Energy: {energy:.2f}", font_size=20).next_to(new_iteration_text, DOWN)
            
            # For now, skip spring updaters due to recursion issue
            # Springs will snap to new positions instead of smoothly transitioning
            
            animations = []
            for dot, new_pos in zip(moving_dots, new_positions):
                animations.append(dot.animate.move_to(new_pos))
            
            new_step_text = Text("Applying transformation...", font_size=20, color=YELLOW).to_edge(DOWN)
            
            # Fade out old springs during transformation
            self.play(
                FadeOut(springs),
                *animations,
                ReplacementTransform(iteration_text, new_iteration_text),
                ReplacementTransform(error_text, new_error_text),
                ReplacementTransform(step_text, new_step_text),
                run_time=1.5
            )
            step_text = new_step_text
            
            iteration_text = new_iteration_text
            error_text = new_error_text
            
            self.wait(0.5)
            
            # Convergence already tested, just continue animation
            pass
        
        success_text = Text("Convergence achieved!", color=GREEN, font_size=30).to_edge(DOWN)
        self.play(ReplacementTransform(step_text, success_text))
        self.wait(2)
        
        # Fade out springs before final fade
        self.play(FadeOut(springs))
        
        self.play(
            FadeOut(moving_dots),
            FadeOut(target_dots),
            FadeOut(target_label),
            FadeOut(iteration_text),
            FadeOut(error_text),
            FadeOut(success_text),
            FadeOut(energy_box),
            FadeOut(energy_title),
            FadeOut(energy_formula),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(energy_dots),
            FadeOut(energy_line)
        )
        
        final_text = VGroup(
            Text("ICP Algorithm Steps:", font_size=28).to_edge(UP),
            Text("1. Find nearest neighbors", font_size=22).shift(UP*0.5),
            Text("2. Compute optimal transformation", font_size=22).shift(DOWN*0.2),
            Text("3. Apply transformation", font_size=22).shift(DOWN*0.9),
            Text("4. Repeat until convergence", font_size=22).shift(DOWN*1.6),
        )
        
        self.play(Write(final_text))
        self.wait(3)
    
    def create_star_pattern(self):
        """DEPRECATED: Use create_test_data from icp_core instead"""
        # This method kept for compatibility but should use icp_core.create_test_data
        source, _ = create_test_data()
        return source
    
    def transform_points(self, points, angle, translation):
        """Apply rotation and translation to points"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        transformed = (rotation_matrix @ points.T).T + translation
        return transformed
    
    def find_nearest_neighbors(self, source, target):
        """Find nearest neighbor correspondences"""
        correspondences = []
        for i, s_point in enumerate(source):
            distances = [np.linalg.norm(s_point - t_point) for t_point in target]
            j = np.argmin(distances)
            correspondences.append((i, j))
        return correspondences
    
    def compute_transformation(self, source, target):
        """Compute optimal rotation and translation using SVD"""
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        centered_source = source - centroid_source
        centered_target = target - centroid_target
        
        H = centered_source.T @ centered_target
        
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_target - R @ centroid_source
        
        return R, t


class ICPWithNoise(Scene):
    """Extended animation showing ICP with noisy data and outliers"""
    
    def construct(self):
        title = Text("ICP with Noisy Data", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        
        np.random.seed(123)
        
        source_points = self.create_constellation()
        
        rotation = 0.4
        translation = np.array([1.2, 0.6, 0])
        noise_level = 0.15
        target_points = self.transform_and_add_noise(source_points, rotation, translation, noise_level)
        
        outliers = self.add_outliers(3)
        target_points = np.vstack([target_points, outliers])
        
        source_dots = VGroup(*[
            Dot(point, color=BLUE, radius=0.06) for point in source_points
        ])
        target_dots = VGroup(*[
            Dot(point, color=RED if i < len(source_points) else ORANGE, radius=0.06) 
            for i, point in enumerate(target_points)
        ])
        
        noise_label = Text(f"Noise: {noise_level}", font_size=20, color=YELLOW).to_corner(UR)
        outlier_label = Text("Orange = Outliers", font_size=20, color=ORANGE).next_to(noise_label, DOWN)
        
        self.play(
            Create(source_dots),
            Create(target_dots),
            Write(noise_label),
            Write(outlier_label)
        )
        self.wait(1)
        
        moving_dots = source_dots.copy()
        
        iteration_text = Text("Iteration: 0", font_size=24).to_corner(UL)
        self.play(Write(iteration_text))
        
        for iteration in range(6):
            correspondences = self.find_nearest_neighbors(
                [dot.get_center() for dot in moving_dots],
                [dot.get_center() for dot in target_dots]
            )
            
            correspondence_lines = VGroup(*[
                Line(
                    moving_dots[i].get_center(),
                    target_dots[j].get_center(),
                    stroke_width=1,
                    color=YELLOW if j < len(source_points) else DARK_GRAY,
                    stroke_opacity=0.4
                )
                for i, j in correspondences
            ])
            
            self.play(Create(correspondence_lines), run_time=0.5)
            
            current_points = np.array([dot.get_center() for dot in moving_dots])
            target_matched = np.array([target_dots[j].get_center() for i, j in correspondences])
            
            R, t = self.compute_transformation(current_points, target_matched)
            
            new_positions = (R @ current_points.T).T + t
            
            new_iteration_text = Text(f"Iteration: {iteration + 1}", font_size=24).to_corner(UL)
            
            animations = []
            for dot, new_pos in zip(moving_dots, new_positions):
                animations.append(dot.animate.move_to(new_pos))
            
            self.play(
                *animations,
                ReplacementTransform(iteration_text, new_iteration_text),
                run_time=1
            )
            
            iteration_text = new_iteration_text
            
            self.play(FadeOut(correspondence_lines), run_time=0.3)
        
        converged_text = Text("Converged despite noise!", color=GREEN, font_size=26)
        converged_text.to_edge(DOWN)
        self.play(Write(converged_text))
        self.wait(2)
    
    def create_constellation(self):
        """Create a constellation-like pattern"""
        points = [
            [-1.5, 1.0, 0],
            [-0.5, 1.2, 0],
            [0.5, 0.8, 0],
            [1.2, 1.0, 0],
            [-1.0, 0.0, 0],
            [0.0, -0.2, 0],
            [0.8, 0.0, 0],
            [-0.5, -1.0, 0],
            [0.5, -1.2, 0],
            [1.5, -0.8, 0],
        ]
        return np.array(points)
    
    def transform_and_add_noise(self, points, angle, translation, noise_level):
        """Transform points and add Gaussian noise"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        transformed = (rotation_matrix @ points.T).T + translation
        
        noise = np.random.normal(0, noise_level, transformed.shape)
        noise[:, 2] = 0
        
        return transformed + noise
    
    def add_outliers(self, n_outliers):
        """Add random outlier points"""
        outliers = []
        for _ in range(n_outliers):
            x = np.random.uniform(-2.5, 2.5)
            y = np.random.uniform(-2, 2)
            outliers.append([x, y, 0])
        return np.array(outliers)
    
    def find_nearest_neighbors(self, source, target):
        """Find nearest neighbor correspondences"""
        correspondences = []
        for i, s_point in enumerate(source):
            distances = [np.linalg.norm(s_point - t_point) for t_point in target]
            j = np.argmin(distances)
            correspondences.append((i, j))
        return correspondences
    
    def compute_transformation(self, source, target):
        """Compute optimal rotation and translation using SVD"""
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        centered_source = source - centroid_source
        centered_target = target - centroid_target
        
        H = centered_source.T @ centered_target
        
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_target - R @ centroid_source
        
        return R, t


if __name__ == "__main__":
    from manim import config
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 60
    
    scene = ICPAnimation()
    scene.render()