#!/usr/bin/env python3
"""
Pointing Error Animation using Manim

Demonstrates absolute vs relative pointing error:
1. Telescope viewport with 1D jitter around a star
2. Live timecourse trace showing pointing offset
3. Mean subtraction to extract relative pointing error
4. Gaussian distribution of the residuals
"""

from manim import *
import numpy as np


def generate_los_trajectory(duration, dt, dc_offset=3.0, noise_std=1.5,
                            smooth_sigma=3, seed=42):
    """Generate a 1D pointing trajectory: DC offset + stationary Gaussian noise.

    The noise is Gaussian-smoothed white noise, giving realistic-looking
    correlated jitter at constant amplitude (no decay).

    Returns time array and pointing offset (in arcseconds).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)

    # White noise smoothed by a Gaussian kernel
    raw_noise = rng.normal(size=len(t))
    kernel_size = int(6 * smooth_sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (k / smooth_sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(raw_noise, kernel, mode="same")

    # Scale to desired std and add DC offset
    smoothed = smoothed / np.std(smoothed) * noise_std
    signal = smoothed + dc_offset

    return t, signal


class PointingErrorIntro(Scene):
    def construct(self):
        # Title
        title = Text("Pointing Error", font_size=42)
        self.play(Write(title))
        self.wait(0.8)
        self.play(FadeOut(title))

        # --- Phase 1: Live telescope + 1D timecourse ---
        duration = 10.0
        dt = 0.05
        t, los = generate_los_trajectory(duration, dt)

        viewport_group = self.build_viewport()
        axes, axes_label, plot_group = self.build_timecourse_axes(duration, los)

        self.play(FadeIn(viewport_group), FadeIn(plot_group))
        self.wait(0.5)

        self.animate_trajectory(t, los, axes, viewport_group)
        self.show_integration_time(t, axes)
        self.wait(1)

        # Fade out phase 1
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # --- Phase 2: 100s trace, mean subtraction ---
        self.show_relative_error()

    def build_viewport(self):
        """Build the telescope viewport on the left side."""
        viewport_center = LEFT * 4

        # Circular aperture
        aperture = Circle(radius=2.5, color=WHITE, stroke_width=2)
        aperture.move_to(viewport_center)

        # Subtle grid lines
        grid = VGroup()
        for offset in [-1.5, -0.75, 0.75, 1.5]:
            h_line = Line(
                viewport_center + LEFT * 2.5 + UP * offset,
                viewport_center + RIGHT * 2.5 + UP * offset,
                stroke_width=0.5, color=GRAY, stroke_opacity=0.3,
            )
            v_line = Line(
                viewport_center + UP * 2.5 + RIGHT * offset,
                viewport_center + DOWN * 2.5 + RIGHT * offset,
                stroke_width=0.5, color=GRAY, stroke_opacity=0.3,
            )
            grid.add(h_line, v_line)

        # Target star at center
        star_glow = Circle(
            radius=0.2, color=WHITE, fill_opacity=0.15, stroke_width=0
        )
        star_glow.move_to(viewport_center)
        star_core = Dot(viewport_center, color=WHITE, radius=0.06)
        target_star = VGroup(star_glow, star_core)

        # Crosshair (moves vertically during animation)
        crosshair_size = 0.4
        crosshair_h = Line(
            LEFT * crosshair_size, RIGHT * crosshair_size, color=RED, stroke_width=2
        )
        crosshair_v = Line(
            UP * crosshair_size, DOWN * crosshair_size, color=RED, stroke_width=2
        )
        crosshair_h.move_to(viewport_center)
        crosshair_v.move_to(viewport_center)

        viewport_label = Text("Telescope FOV", font_size=16, color=GRAY)
        viewport_label.next_to(aperture, DOWN, buff=0.3)

        group = VGroup(
            aperture, grid, target_star, crosshair_h, crosshair_v, viewport_label
        )
        group.crosshair_h = crosshair_h
        group.crosshair_v = crosshair_v
        group.viewport_center = viewport_center

        return group

    def build_timecourse_axes(self, duration, los):
        """Build a single timecourse plot on the right side."""
        max_amp = np.max(np.abs(los))
        y_range_val = np.ceil(max_amp / 2) * 2 + 2

        plot_center_x = 3.0

        axes = Axes(
            x_range=[0, duration, duration / 5],
            y_range=[-y_range_val, y_range_val, y_range_val / 2],
            x_length=5,
            y_length=4.0,
            axis_config={"include_numbers": True, "font_size": 12},
        )
        axes.move_to(RIGHT * plot_center_x)

        axes_label = Text("Pointing Error (arcsec)", font_size=14, color=RED)
        axes_label.next_to(axes, UP, buff=0.15)

        time_label = Text("Time (s)", font_size=14, color=GRAY)
        time_label.next_to(axes, DOWN, buff=0.3)

        plot_group = VGroup(axes, axes_label, time_label)

        return axes, axes_label, plot_group

    def animate_trajectory(self, t, los, axes, viewport_group):
        """Animate the LOS trajectory with live crosshair and timecourse trace."""
        crosshair_h = viewport_group.crosshair_h
        crosshair_v = viewport_group.crosshair_v
        vp_center = viewport_group.viewport_center

        # Scale: arcseconds to viewport units (vertical)
        max_amp = np.max(np.abs(los))
        vp_scale = 2.0 / max_amp

        # Pre-compute screen-space points
        n_points = len(t)
        screen_pts = [axes.c2p(t[i], los[i]) for i in range(n_points)]

        # Build full curve (used as reference for partial reveal)
        full_curve = VMobject(color=RED, stroke_width=2)
        full_curve.set_points_smoothly(screen_pts)

        # Tracking dot
        trace_dot = Dot(screen_pts[0], color=RED, radius=0.06).set_z_index(1)

        # ValueTracker: 0 → 1
        progress = ValueTracker(0.0)

        # Partial curve that reveals progressively
        trace = full_curve.copy()

        def update_trace(mob):
            alpha = progress.get_value()
            mob.pointwise_become_partial(full_curve, 0, max(alpha, 1e-6))

        def update_dot(mob):
            idx = int(progress.get_value() * (n_points - 1))
            idx = min(idx, n_points - 1)
            mob.move_to(screen_pts[idx])

        def update_crosshair(_mob):
            idx = int(progress.get_value() * (n_points - 1))
            idx = min(idx, n_points - 1)
            offset_y = los[idx] * vp_scale
            new_center = vp_center + UP * offset_y
            crosshair_h.move_to(new_center)
            crosshair_v.move_to(new_center)

        trace.add_updater(update_trace)
        trace_dot.add_updater(update_dot)
        crosshair_h.add_updater(update_crosshair)

        self.add(trace, trace_dot)

        self.play(
            progress.animate.set_value(1.0),
            run_time=8.0, rate_func=linear,
        )

        trace.clear_updaters()
        trace_dot.clear_updaters()
        crosshair_h.clear_updaters()

    def show_integration_time(self, t, axes):
        """Highlight the total integration time."""
        duration = t[-1]

        bracket = BraceBetweenPoints(
            axes.c2p(0, 0), axes.c2p(duration, 0),
            direction=DOWN, color=YELLOW,
        )
        time_text = Text(f"T = {duration:.0f} s", font_size=20, color=YELLOW)
        time_text.next_to(bracket, DOWN, buff=0.1)

        self.play(Create(bracket), Write(time_text))
        self.wait(2)

    def show_relative_error(self):
        """Show 100s trace, compute mean, subtract, then show Gaussian residuals."""
        title = Text("Relative Pointing Error", font_size=36, color=GREEN)
        title.to_edge(UP)
        self.play(Write(title))

        # Generate 100s trajectory
        duration = 100.0
        dt = 0.1
        t, los = generate_los_trajectory(duration, dt)
        mean_val = np.mean(los)
        residuals = los - mean_val

        # Axes for the timecourse (full width)
        max_amp = np.max(np.abs(los))
        y_range_val = np.ceil(max_amp / 2) * 2

        tc_axes = Axes(
            x_range=[0, duration, 20],
            y_range=[-y_range_val, y_range_val, y_range_val / 2],
            x_length=10,
            y_length=3.0,
            axis_config={"include_numbers": True, "font_size": 12},
        )
        tc_axes.move_to(UP * 0.3)

        tc_label = Text("Pointing Error (arcsec)", font_size=14, color=RED)
        tc_label.next_to(tc_axes, UP, buff=0.1)
        time_label = Text("Time (s)", font_size=14, color=GRAY)
        time_label.next_to(tc_axes, DOWN, buff=0.2)

        # Draw trace
        n_pts = len(t)
        screen_pts = [tc_axes.c2p(t[i], los[i]) for i in range(n_pts)]
        curve = VMobject(color=RED, stroke_width=1.5)
        curve.set_points_smoothly(screen_pts)

        self.play(Create(tc_axes), Write(tc_label), Write(time_label))
        self.play(Create(curve), run_time=2)
        self.wait(0.5)

        # Draw mean line
        mean_line = DashedLine(
            tc_axes.c2p(0, mean_val), tc_axes.c2p(duration, mean_val),
            color=YELLOW, stroke_width=2,
        )
        mean_label = MathTex(
            rf"\bar{{x}} = {mean_val:.1f}''", font_size=22, color=YELLOW,
        )
        mean_label.next_to(mean_line, RIGHT, buff=0.2)

        avg_text = Text("Average over T = 100 s", font_size=20, color=YELLOW)
        avg_text.to_edge(DOWN, buff=0.3)

        self.play(
            Create(mean_line), Write(mean_label), Write(avg_text),
        )
        self.wait(2)

        # Subtract the mean: shift curve down
        zero_y = tc_axes.c2p(0, 0)[1]
        mean_y = tc_axes.c2p(0, mean_val)[1]
        shift_amount = zero_y - mean_y

        subtract_text = MathTex(
            r"\text{Relative} = \text{Signal} - \bar{x}", font_size=24, color=GREEN,
        )
        subtract_text.to_edge(DOWN, buff=0.3)

        self.play(
            curve.animate.shift(UP * shift_amount),
            FadeOut(mean_line), FadeOut(mean_label),
            ReplacementTransform(avg_text, subtract_text),
            run_time=2,
        )

        new_label = Text("Relative Pointing Error (arcsec)", font_size=14, color=GREEN)
        new_label.move_to(tc_label.get_center())
        self.play(ReplacementTransform(tc_label, new_label))
        self.wait(1)

        # --- Phase 3: Gaussian histogram of residuals ---
        self.show_residual_distribution(
            residuals, tc_axes, curve, new_label, time_label,
            title, subtract_text,
        )

    def show_residual_distribution(self, residuals, tc_axes, curve,
                                   tc_label, time_label, title, formula):
        """Show histogram + Gaussian fit of the mean-subtracted residuals."""
        self.play(FadeOut(formula))

        # Build histogram data
        n_bins = 25
        counts, bin_edges = np.histogram(residuals, bins=n_bins)
        bin_width = bin_edges[1] - bin_edges[0]
        max_count = np.max(counts)

        # Histogram axes below the timecourse
        hist_axes = Axes(
            x_range=[bin_edges[0] - 1, bin_edges[-1] + 1, 1],
            y_range=[0, max_count + 2, max(max_count // 4, 1)],
            x_length=10,
            y_length=2.5,
            axis_config={"include_numbers": True, "font_size": 12},
        )
        hist_axes.to_edge(DOWN, buff=0.6)

        hist_x_label = Text("Residual (arcsec)", font_size=14, color=GREEN)
        hist_x_label.next_to(hist_axes, UP, buff=0.1)
        hist_y_label = Text("Count", font_size=12, color=GRAY)
        hist_y_label.next_to(hist_axes.y_axis, UP, buff=0.1)

        # Move timecourse up to make room
        shift_up = UP * 0.8
        self.play(
            tc_axes.animate.shift(shift_up),
            curve.animate.shift(shift_up),
            tc_label.animate.shift(shift_up),
            time_label.animate.shift(shift_up),
            title.animate.shift(shift_up),
            Create(hist_axes), Write(hist_x_label), Write(hist_y_label),
        )

        # Draw histogram bars
        bars = VGroup()
        for i in range(n_bins):
            bar_left = hist_axes.c2p(bin_edges[i], 0)
            bar_right = hist_axes.c2p(bin_edges[i + 1], counts[i])

            bar = Rectangle(
                width=abs(bar_right[0] - bar_left[0]),
                height=abs(bar_right[1] - bar_left[1]),
                color=GREEN,
                fill_opacity=0.6,
                stroke_width=1,
            )
            bar.move_to(
                [(bar_left[0] + bar_right[0]) / 2,
                 (bar_left[1] + bar_right[1]) / 2, 0]
            )
            bars.add(bar)

        self.play(Create(bars), run_time=1.5)
        self.wait(0.5)

        # Overlay Gaussian PDF curve
        sigma = np.std(residuals)
        mu = np.mean(residuals)
        x_vals = np.linspace(bin_edges[0] - 1, bin_edges[-1] + 1, 200)
        # Scale PDF to match histogram counts
        pdf_scale = len(residuals) * bin_width
        y_vals = pdf_scale / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((x_vals - mu) / sigma) ** 2
        )

        gauss_pts = [hist_axes.c2p(x_vals[i], y_vals[i]) for i in range(len(x_vals))]
        gauss_curve = VMobject(color=YELLOW, stroke_width=2.5)
        gauss_curve.set_points_smoothly(gauss_pts)

        sigma_label = MathTex(
            rf"\sigma = {sigma:.2f}''", font_size=24, color=YELLOW,
        )
        sigma_label.next_to(hist_axes, RIGHT, buff=0.3)

        self.play(Create(gauss_curve), Write(sigma_label), run_time=1.5)
        self.wait(3)


if __name__ == "__main__":
    from manim import config

    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 60

    scene = PointingErrorIntro()
    scene.render()
