#!/usr/bin/env python3
"""
MONOCLE Tracking System Animation using Manim

Shows the complete tracking lifecycle:
1. Idle → Acquiring (frame accumulation)
2. Acquiring → Calibrating (star detection & filtering)
3. Calibrating → Tracking (centroid tracking with ROI)
4. Tracking → Reacquiring (loss handling)
"""

from manim import *
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class StarData:
    """Data for a star in the field"""
    star: VGroup  # The manim visual object
    x: float  # X coordinate (in field space, not screen space)
    y: float  # Y coordinate (in field space, not screen space)
    good: bool  # Whether star passes filters
    reason: Optional[str]  # Rejection reason if bad ('edge', 'saturated', 'elongated')
    brightness: float  # Star brightness 0-1


class StarField:
    """Container for a star field with extent information"""

    def __init__(self, stars: list[StarData]):
        self.stars = stars
        self._compute_extents()

    def _compute_extents(self):
        """Calculate the bounding box of all stars"""
        if not self.stars:
            self.min_x = self.max_x = self.min_y = self.max_y = 0
            return

        self.min_x = min(s.x for s in self.stars)
        self.max_x = max(s.x for s in self.stars)
        self.min_y = min(s.y for s in self.stars)
        self.max_y = max(s.y for s in self.stars)

        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

    def scale_to_box(self, target_center, target_width, target_height):
        """Scale and position all stars to fit in target box"""
        if self.width == 0 or self.height == 0:
            return

        # Calculate scale factor (use minimum to fit entirely in box)
        scale_x = target_width / self.width
        scale_y = target_height / self.height
        scale = min(scale_x, scale_y) * 0.9  # 0.9 for padding

        # Calculate field center in field space
        field_center_x = (self.min_x + self.max_x) / 2
        field_center_y = (self.min_y + self.max_y) / 2

        # Position each star
        for star_data in self.stars:
            # Translate to origin, scale, translate to target center
            new_x = (star_data.x - field_center_x) * scale
            new_y = (star_data.y - field_center_y) * scale

            star_data.star.move_to(target_center + RIGHT * new_x + UP * new_y)

    def get_vgroup(self):
        """Get VGroup of all star visuals"""
        return VGroup(*[s.star for s in self.stars])


class MonocleTracking(Scene):
    """Complete MONOCLE tracking system animation"""

    def construct(self):
        # Title
        title = Text("MONOCLE Fine Guidance System", font_size=40)
        subtitle = Text("Star Detection & Tracking", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)

        self.play(Write(title), Write(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Show state machine overview
        self.show_state_machine()
        self.wait(2)

        # Run through complete lifecycle
        self.show_acquiring_phase()
        self.show_calibration_phase()
        self.show_tracking_phase()
        self.show_summary()

    def show_state_machine(self):
        """Visualize the FGS state machine"""
        title = Text("FGS State Machine", font_size=32).to_edge(UP)
        self.play(Write(title))

        # Create state boxes
        idle_box = self.create_state_box("Idle", BLUE)
        acquiring_box = self.create_state_box("Acquiring", YELLOW)
        calibrating_box = self.create_state_box("Calibrating", ORANGE)
        tracking_box = self.create_state_box("Tracking", GREEN)
        reacquiring_box = self.create_state_box("Reacquiring", RED)

        # Position states
        idle_box.move_to(UP * 2 + LEFT * 4)
        acquiring_box.move_to(UP * 2 + RIGHT * 0)
        calibrating_box.move_to(UP * 2 + RIGHT * 4)
        tracking_box.move_to(DOWN * 0.5 + RIGHT * 4)
        reacquiring_box.move_to(DOWN * 0.5 + RIGHT * 0)

        # Create arrows with labels
        arrow1 = Arrow(idle_box.get_right(), acquiring_box.get_left(), buff=0.1)
        label1 = Text("START_FGS", font_size=14).next_to(arrow1, UP, buff=0.05)

        arrow2 = Arrow(acquiring_box.get_right(), calibrating_box.get_left(), buff=0.1)
        label2 = Text("N frames", font_size=14).next_to(arrow2, UP, buff=0.05)

        arrow3 = Arrow(calibrating_box.get_bottom(), tracking_box.get_top(), buff=0.1)
        label3 = Text("star selected", font_size=14).next_to(arrow3, RIGHT, buff=0.05)

        # Use curved arrows between Tracking and Reacquiring to avoid overlap
        # One curves down (negative angle), one curves up (positive angle)
        arrow4 = CurvedArrow(tracking_box.get_left(), reacquiring_box.get_right(),
                            angle=-TAU/6)
        label4 = Text("signal lost", font_size=14).next_to(arrow4, DOWN, buff=0.2)

        arrow5 = Arrow(reacquiring_box.get_top(), calibrating_box.get_bottom(), buff=0.1)
        label5 = Text("timeout", font_size=14).next_to(arrow5, LEFT, buff=0.05)

        arrow6 = CurvedArrow(reacquiring_box.get_right(), tracking_box.get_left(),
                            angle=TAU/6)
        arrow6.set_color(GREEN)
        label6 = Text("recovered", font_size=14, color=GREEN).next_to(arrow6, UP, buff=0.2)

        # Animate state machine
        states = VGroup(idle_box, acquiring_box, calibrating_box, tracking_box, reacquiring_box)
        arrows = VGroup(arrow1, arrow2, arrow3, arrow4, arrow5, arrow6)
        labels = VGroup(label1, label2, label3, label4, label5, label6)

        self.play(Create(states))
        self.play(Create(arrows), Write(labels))
        self.wait(2)

        # Store for later fade out
        self.state_diagram = VGroup(title, states, arrows, labels)

    def create_state_box(self, text, color):
        """Create a rounded box for a state"""
        label = Text(text, font_size=18, color=WHITE)
        box = RoundedRectangle(
            width=label.width + 0.5,
            height=label.height + 0.3,
            corner_radius=0.1,
            color=color,
            fill_opacity=0.3,
            stroke_width=2
        )
        label.move_to(box.get_center())
        return VGroup(box, label)

    def show_acquiring_phase(self):
        """Show frame acquisition and averaging"""
        self.play(FadeOut(self.state_diagram))

        title = Text("Phase 1: Acquiring", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        # Show camera frame concept
        frame_box = Rectangle(width=3, height=3, color=WHITE)
        frame_box.shift(LEFT * 4)
        frame_label = Text("Camera Frame", font_size=16).next_to(frame_box, DOWN)

        # Show accumulator
        accum_box = Rectangle(width=3, height=3, color=YELLOW)
        accum_box.shift(RIGHT * 1)
        accum_label = Text("Accumulator", font_size=16).next_to(accum_box, DOWN)

        # Counter
        counter = Variable(0, Text("Frames:", font_size=20), num_decimal_places=0)
        counter.next_to(accum_box, RIGHT, buff=1)

        self.play(
            Create(frame_box),
            Write(frame_label),
            Create(accum_box),
            Write(accum_label),
            Write(counter)
        )

        # Create consistent star field and scale to frame_box
        star_field_obj = self.create_star_field()
        star_field_obj.scale_to_box(frame_box.get_center(), 3.0, 3.0)
        star_field = star_field_obj.get_vgroup()

        # Track all accumulated stars for cleanup
        all_accum_stars = VGroup()

        # Animate frame accumulation - use SAME stars each time
        for i in range(1, 4):
            # Show stars in camera frame
            star_field_copy = star_field.copy()
            self.play(FadeIn(star_field_copy, scale=1.2), run_time=0.3)

            # Show "adding" to accumulator
            arrow = Arrow(frame_box.get_right(), accum_box.get_left(), color=YELLOW)
            plus_sign = Text("+", font_size=36, color=YELLOW).move_to(arrow.get_center())

            self.play(Create(arrow), Write(plus_sign), run_time=0.3)

            # Update counter
            counter.tracker.set_value(i)

            # Accumulate brightness (visual effect)
            accum_stars = star_field_copy.copy().move_to(accum_box.get_center())
            self.add(accum_stars)
            all_accum_stars.add(accum_stars)

            self.play(FadeOut(arrow), FadeOut(plus_sign), FadeOut(star_field_copy), run_time=0.3)
            self.wait(0.3)

        # Show averaging - position below counter to avoid overlap
        avg_text = Text("÷ 3 = Averaged Frame", font_size=20, color=GREEN)
        avg_text.next_to(counter, DOWN, buff=0.5)
        self.play(Write(avg_text))
        self.wait(1)

        # Store for transition
        self.acquiring_group = VGroup(
            title, frame_box, frame_label, accum_box, accum_label,
            counter, avg_text, all_accum_stars
        )

    def show_calibration_phase(self):
        """Show star detection and filtering"""
        self.play(FadeOut(self.acquiring_group))

        title = Text("Phase 2: Calibrating", font_size=32, color=ORANGE).to_edge(UP)
        self.play(Write(title))

        # Show averaged frame with many stars (use YELLOW for visual continuity with accum phase)
        frame = Rectangle(width=5, height=5, color=YELLOW).shift(LEFT * 2.5)

        # Create same star field as in acquiring phase and scale to frame
        star_field_obj = self.create_star_field()
        star_field_obj.scale_to_box(frame.get_center(), 5.0, 5.0)
        stars = star_field_obj.get_vgroup()
        star_data = star_field_obj.stars  # Get individual star data for filtering

        self.play(Create(frame), FadeIn(stars))
        self.wait(0.5)

        # Show filter cascade on the right
        filter_y = 2
        filter_labels = []

        filters = [
            ("Aspect Ratio Filter", f"{len(star_data)} stars", GRAY),
            ("Edge Distance Filter", None, GRAY),
            ("Saturation Filter", None, GRAY),
            ("SNR Filter", None, GRAY),
            ("Select Brightest", "1 star", GREEN),
        ]

        for filter_name, count, color in filters:
            label = Text(f"{filter_name}: {count if count else '...'}",
                        font_size=16, color=color)
            label.move_to(RIGHT * 4 + UP * filter_y)
            filter_labels.append(label)
            filter_y -= 0.6

        # Track X marks and arrows for cleanup
        all_x_marks = VGroup()
        all_arrows = VGroup()
        current_filter_arrows = VGroup()  # Track current filter's arrows to fade them out

        # Define filter colors (for arrows)
        filter_colors = {
            'elongated': BLUE,
            'edge': ORANGE,
            'saturated': PURPLE,
            'low_snr': RED,
        }

        # Apply filters sequentially
        remaining_count = len(star_data)

        for i, (filter_name, _, _) in enumerate(filters):
            # Show filter label
            if i == 0:
                self.play(Write(filter_labels[i]))

            self.wait(0.3)

            # Fade out previous filter's arrows
            if len(current_filter_arrows) > 0:
                self.play(*[FadeOut(arrow) for arrow in current_filter_arrows], run_time=0.3)
                current_filter_arrows = VGroup()

            # Highlight bad stars for this filter
            to_reject = []

            if "Edge" in filter_name:
                reason = 'edge'
            elif "Saturation" in filter_name:
                reason = 'saturated'
            elif "Aspect" in filter_name:
                reason = 'elongated'
            elif "SNR" in filter_name:
                reason = 'low_snr'
            else:
                reason = None

            if reason:
                for data in star_data:
                    if not data.good and data.reason == reason:
                        to_reject.append(data.star)

            # Reject stars
            if to_reject:
                animations = []
                arrow_color = filter_colors.get(reason, RED)

                for star in to_reject:
                    # Add arrow from filter label to rejected star
                    arrow = Arrow(
                        filter_labels[i].get_left(),
                        star.get_center(),
                        color=arrow_color,
                        stroke_width=3,
                        max_tip_length_to_length_ratio=0.15
                    )
                    all_arrows.add(arrow)
                    current_filter_arrows.add(arrow)  # Track for fade out
                    animations.append(Create(arrow))

                    # Add X mark (always red)
                    x_mark = Text("✗", font_size=20, color=RED).move_to(star.get_center())
                    all_x_marks.add(x_mark)
                    animations.append(FadeIn(x_mark, scale=1.5))
                    animations.append(star.animate.set_opacity(0.3))

                self.play(*animations, run_time=0.5)
                remaining_count -= len(to_reject)

            # Update filter label
            if i < len(filters) - 1:
                new_label = Text(f"{filter_name}: {remaining_count} stars",
                               font_size=16, color=YELLOW)
                new_label.move_to(filter_labels[i].get_center())
                self.play(ReplacementTransform(filter_labels[i], new_label))
                filter_labels[i] = new_label

        # Select brightest star
        brightest = max([d for d in star_data if d.good], key=lambda x: x.brightness)
        brightest_star = brightest.star

        self.wait(0.5)
        self.play(Write(filter_labels[-1]))

        # Highlight selected star
        highlight = Circle(radius=0.25, color=GREEN, stroke_width=4)
        highlight.move_to(brightest_star.get_center())

        roi_box = Square(side_length=1.0, color=GREEN, stroke_width=3)
        roi_box.move_to(brightest_star.get_center())

        self.play(
            Create(highlight),
            brightest_star.animate.set_color(GREEN).scale(1.3),
            run_time=0.8
        )
        self.wait(0.3)

        # Show ROI
        roi_label = Text("ROI Set", font_size=18, color=GREEN)
        roi_label.next_to(roi_box, DOWN, buff=0.2)

        self.play(
            FadeOut(highlight),
            Create(roi_box),
            Write(roi_label)
        )
        self.wait(1)

        # Store for next phase (don't include all_arrows - they were already faded out)
        self.calibration_group = VGroup(title, frame, stars, *filter_labels, roi_box, roi_label, all_x_marks)
        self.selected_star = brightest_star
        self.roi_box = roi_box

    def show_tracking_phase(self):
        """Show centroid tracking in ROI"""
        self.play(FadeOut(self.calibration_group))

        title = Text("Phase 3: Tracking", font_size=32, color=GREEN).to_edge(UP)
        self.play(Write(title))

        # Zoom into ROI
        roi_frame = Square(side_length=4, color=GREEN, stroke_width=3)
        roi_label = Text("ROI (32×32 pixels)", font_size=20).next_to(roi_frame, DOWN)

        self.play(Create(roi_frame), Write(roi_label))

        # Show star in ROI (start at center)
        star = self.create_star(0.3, 1.0)
        star.move_to(roi_frame.get_center())

        # Show centroid mask (stays at ROI center) - 50% larger than star envelope
        mask_circle = Circle(radius=0.9, color=YELLOW, stroke_width=2, fill_opacity=0.1)
        mask_circle.move_to(roi_frame.get_center())
        mask_label = Text("Centroid Mask\n(5×FWHM)", font_size=14, color=YELLOW)
        mask_label.next_to(mask_circle, RIGHT, buff=0.5)

        self.play(FadeIn(star, scale=1.2))
        self.play(Create(mask_circle), Write(mask_label))
        self.wait(0.5)

        # Show centroid calculation (smaller dot)
        centroid_dot = Dot(star.get_center(), color=RED, radius=0.05)
        centroid_label = Text("Centroid", font_size=14, color=RED)
        centroid_label.next_to(centroid_dot, LEFT, buff=0.3)

        crosshair_h = Line(
            centroid_dot.get_center() + LEFT * 0.3,
            centroid_dot.get_center() + RIGHT * 0.3,
            color=RED, stroke_width=2
        )
        crosshair_v = Line(
            centroid_dot.get_center() + UP * 0.3,
            centroid_dot.get_center() + DOWN * 0.3,
            color=RED, stroke_width=2
        )

        self.play(
            Create(centroid_dot),
            Create(crosshair_h),
            Create(crosshair_v),
            Write(centroid_label)
        )
        self.wait(0.5)

        # Show position readout
        pos_text = Text("Position: (256.34, 128.67) px\nFlux: 45,230 DN",
                       font_size=16, color=WHITE)
        pos_text.next_to(roi_frame, RIGHT, buff=1)
        self.play(Write(pos_text))
        self.wait(0.5)

        # Simulate tracking drift
        info = Text("Tracking drift...", font_size=20, color=YELLOW)
        info.to_edge(DOWN)
        self.play(Write(info))

        # Create trail
        trail_dots = []
        for i in range(8):
            # Simulate small random walk (reduced wobble)
            drift_x = np.random.uniform(-0.05, 0.05)
            drift_y = np.random.uniform(-0.05, 0.05)

            # Move star and centroid
            new_star_pos = star.get_center() + RIGHT * drift_x + UP * drift_y
            new_centroid_pos = centroid_dot.get_center() + RIGHT * drift_x + UP * drift_y

            # Add trail dot at old position (smaller to match centroid)
            trail = Dot(centroid_dot.get_center(), radius=0.02, color=BLUE, fill_opacity=0.5)
            trail_dots.append(trail)
            self.add(trail)

            # Animate movement (mask stays stationary)
            self.play(
                star.animate.move_to(new_star_pos),
                centroid_dot.animate.move_to(new_centroid_pos),
                crosshair_h.animate.move_to(new_centroid_pos),
                crosshair_v.animate.move_to(new_centroid_pos),
                run_time=0.3
            )

            # Update position
            new_x = 256.34 + (i+1) * drift_x * 10
            new_y = 128.67 + (i+1) * drift_y * 10
            new_pos_text = Text(
                f"Position: ({new_x:.2f}, {new_y:.2f}) px\nFlux: {45230 + i*100} DN",
                font_size=16, color=WHITE
            )
            new_pos_text.move_to(pos_text.get_center())
            self.remove(pos_text)
            self.add(new_pos_text)
            pos_text = new_pos_text

        self.wait(1)

        # Store for cleanup
        self.tracking_group = VGroup(
            title, roi_frame, roi_label, star, mask_circle, mask_label,
            centroid_dot, crosshair_h, crosshair_v, centroid_label,
            pos_text, info, *trail_dots
        )

    def show_summary(self):
        """Show summary of the tracking system"""
        self.play(FadeOut(self.tracking_group))

        title = Text("MONOCLE Tracking Summary", font_size=36).to_edge(UP)
        self.play(Write(title))

        summary_points = VGroup(
            Text("1. Acquire: Average N frames to reduce noise", font_size=20),
            Text("2. Calibrate: Detect & filter stars, select brightest", font_size=20),
            Text("3. Track: Compute centroid in ROI for high-rate updates", font_size=20),
            Text("4. Reacquire: Recover if signal is lost", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        summary_points.shift(UP * 0.5)

        key_features = VGroup(
            Text("✓ Multi-stage filtering for robust selection", font_size=18, color=GREEN),
            Text("✓ ROI mode for efficient tracking", font_size=18, color=GREEN),
            Text("✓ Weighted centroid for sub-pixel accuracy", font_size=18, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        key_features.next_to(summary_points, DOWN, buff=1)

        self.play(Write(summary_points), run_time=2)
        self.wait(1)
        self.play(Write(key_features), run_time=1.5)
        self.wait(3)

    def create_star_field(self, n_stars):
        """Create a random star field"""
        np.random.seed(42)
        stars = []
        for _ in range(n_stars):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            size = np.random.uniform(0.05, 0.15)
            brightness = np.random.uniform(0.3, 1.0)
            star = self.create_star(size, brightness)
            star.move_to(RIGHT * x + UP * y)
            stars.append(star)
        return VGroup(*stars)

    def create_star(self, size, brightness, saturated=False):
        """Create a single star with Gaussian-like appearance"""
        if saturated:
            # Saturated stars have blown-out centers (no colored glow)
            star = Circle(radius=size, color=WHITE, fill_opacity=1.0)
            glow = Circle(radius=size * 1.5, color=WHITE, fill_opacity=0.2)
            return VGroup(glow, star)
        else:
            # Normal stars with gradient
            color_value = interpolate_color(BLUE, WHITE, brightness)
            star = Dot(radius=size, color=color_value, fill_opacity=0.8)

            # Add glow
            glow = Circle(
                radius=size * 2,
                color=color_value,
                fill_opacity=0.2 * brightness,
                stroke_width=0
            )
            return VGroup(glow, star)

    def create_elongated_star(self):
        """Create an elongated (bad) star"""
        ellipse = Ellipse(width=0.3, height=0.1, color=WHITE, fill_opacity=0.6)
        ellipse.rotate(np.random.uniform(0, 2 * np.pi))
        return ellipse

    def create_star_field(self):
        """Create consistent star field with good and bad stars in field coordinates"""
        np.random.seed(42)
        star_data = []

        # Define stars in field coordinate space (no scaling yet)
        # Good stars (8 total)
        good_positions = [
            (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0),
            (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)
        ]
        for i, (x, y) in enumerate(good_positions):
            size = 0.08 + (i % 3) * 0.01
            brightness = 0.6 + (i % 4) * 0.1
            star = self.create_star(size, brightness)
            star_data.append(StarData(star=star, x=x, y=y, good=True, reason=None, brightness=brightness))

        # Edge stars (3 total) - close to top/bottom edge
        edge_positions = [(-1.5, 2.3), (0.5, 2.3), (1.8, -2.3)]
        for x, y in edge_positions:
            star = self.create_star(0.09, 0.7)
            star_data.append(StarData(star=star, x=x, y=y, good=False, reason='edge', brightness=0.7))

        # Saturated stars (2 total)
        saturated_positions = [(0.5, 0.2), (-1.2, -0.8)]
        for x, y in saturated_positions:
            star = self.create_star(0.15, 1.0, saturated=True)
            star_data.append(StarData(star=star, x=x, y=y, good=False, reason='saturated', brightness=1.0))

        # Elongated stars (2 total)
        elongated_positions = [(-0.5, -0.4), (1.5, 0.8)]
        for x, y in elongated_positions:
            star = self.create_elongated_star()
            star_data.append(StarData(star=star, x=x, y=y, good=False, reason='elongated', brightness=0.6))

        # Low SNR stars (2 total) - visually dim
        low_snr_positions = [(-1.2, 0.5), (1.3, -0.6)]
        for x, y in low_snr_positions:
            star = self.create_star(0.06, 0.2)  # Small and dim
            star_data.append(StarData(star=star, x=x, y=y, good=False, reason='low_snr', brightness=0.2))

        return StarField(star_data)


class MonocleStateFlow(Scene):
    """Animated flow through states with real frame examples"""

    def construct(self):
        title = Text("MONOCLE State Flow Example", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Show simplified linear flow
        states = ["Idle", "Acquiring", "Calibrating", "Tracking"]
        state_boxes = []
        state_arrows = []

        x_start = -5
        for i, state_name in enumerate(states):
            # Create state box
            color = [BLUE, YELLOW, ORANGE, GREEN][i]
            box = self.create_state_box(state_name, color)
            box.move_to(RIGHT * (x_start + i * 3.5) + UP * 2)
            state_boxes.append(box)

            # Create arrow to next state
            if i < len(states) - 1:
                arrow = Arrow(
                    box.get_right() + RIGHT * 0.1,
                    box.get_right() + RIGHT * 2.4,
                    buff=0,
                    color=WHITE
                )
                state_arrows.append(arrow)

        # Animate states appearing
        for i, box in enumerate(state_boxes):
            self.play(FadeIn(box, shift=DOWN * 0.5))
            if i < len(state_arrows):
                self.play(Create(state_arrows[i]))

        self.wait(1)

        # Show frame count indicator
        frame_indicator = VGroup(
            Text("Frame #", font_size=20),
            Integer(0, font_size=24, color=YELLOW)
        ).arrange(RIGHT, buff=0.3)
        frame_indicator.to_edge(DOWN)

        self.play(Write(frame_indicator))

        # Highlight current state with a glow
        current_state = 0
        highlight = Rectangle(
            width=state_boxes[0].width + 0.3,
            height=state_boxes[0].height + 0.3,
            color=YELLOW,
            stroke_width=4
        ).move_to(state_boxes[0].get_center())

        self.play(Create(highlight))

        # Simulate processing frames
        for frame_num in range(1, 16):
            # Update frame number
            new_counter = Integer(frame_num, font_size=24, color=YELLOW)
            new_counter.move_to(frame_indicator[1].get_center())
            self.play(
                ReplacementTransform(frame_indicator[1], new_counter),
                run_time=0.3
            )
            frame_indicator[1] = new_counter

            # State transitions
            if frame_num == 1:
                # Idle -> Acquiring
                current_state = 1
                self.play(highlight.animate.move_to(state_boxes[current_state].get_center()))
            elif frame_num == 4:
                # Acquiring -> Calibrating (after 3 frames)
                current_state = 2
                self.play(highlight.animate.move_to(state_boxes[current_state].get_center()))
            elif frame_num == 5:
                # Calibrating -> Tracking
                current_state = 3
                self.play(highlight.animate.move_to(state_boxes[current_state].get_center()))

            self.wait(0.2)

        self.wait(2)

    def create_state_box(self, text, color):
        """Create a state box"""
        label = Text(text, font_size=20, color=WHITE)
        box = RoundedRectangle(
            width=2,
            height=0.8,
            corner_radius=0.1,
            color=color,
            fill_opacity=0.3,
            stroke_width=3
        )
        label.move_to(box.get_center())
        return VGroup(box, label)


if __name__ == "__main__":
    from manim import config
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 60

    scene = MonocleTracking()
    scene.render()
