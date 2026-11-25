#!/usr/bin/env python3
"""
Window-mask heater control on the NSV-502, as run by pm_mask_loop.

The scene walks the loop in the order the data flows every 500 ms frame:

1. The frame: eight 62.5 ms windows, one mask bit per window per heater line,
   staged any time and latched at window 0.
2. RTD to PI: a proportional-integral law turns the temperature error into a
   duty percentage, with conditional integration at the rails.
3. Duty to windows with a balance: the part of a window a heater is owed or
   was granted early is carried into the next frame and the request rounds at
   half a window, so a duty between two eighths is delivered on average and
   the balance never exceeds half a window either way.
4. Scheduling under a power cap: whole windows are placed round-robin into
   the least-loaded window, never exceeding the cap of lines per window.
5. Clipping and integrator pause: when the cap grants fewer windows than
   asked, the integrator is held rather than wound up.
6. The closed loop.

The control math here is a direct port of the heater-schedule crate in
tracking-test-bench (windows_with_balance, schedule_frame, pi_duty,
pi_duty_under_budget), so the numbers on screen are the ones the hardware
loop would produce.
"""

import math

import numpy as np
from manim import (
    BLACK,
    DEFAULT_FONT_SIZE,
    DOWN,
    GREY,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    Arrow,
    Axes,
    Create,
    DashedLine,
    DecimalNumber,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    config,
    rate_functions,
    tempconfig,
)

# Palette shared with the tracking-test-bench dashboards.
BLUE = "#7aa2f7"
GREEN = "#9ece6a"
RED = "#f7768e"
ORANGE = "#e0af68"
PURPLE = "#bb9af7"
TEAL = "#7dcfff"
INK = "#c0caf5"
MUTED = "#565f89"
CELL = "#262838"

FRAME_WINDOWS = 8
WINDOW_MS = 62.5
# One bench heater on the 12 V rail, as measured: 13 Ω at 11.7 V.
SUPPLY_V = 11.7
HEATER_A = 0.9
HEATER_W = 10.6


# --------------------------------------------------------------------------
# Control law, ported from heater-schedule/src/lib.rs
# --------------------------------------------------------------------------


def pi_duty(error_c, kp, ki, integrator, dt_s):
    """One PI step. Returns (duty_pct, integrator).

    The integrator only keeps a step that does not push an already saturated
    output further into its rail (conditional integration)."""
    proportional = kp * error_c
    stepped = integrator + error_c * dt_s
    unclamped = proportional + ki * stepped
    if (unclamped > 100.0 and error_c > 0.0) or (unclamped < 0.0 and error_c < 0.0):
        integrator = integrator
    else:
        integrator = stepped
    duty = min(100.0, max(0.0, proportional + ki * integrator))
    return duty, integrator


def pi_duty_under_budget(error_c, kp, ki, integrator, dt_s, clipped_last_frame):
    """PI step that holds the integrator while the scheduler is clipping and
    the heater still wants more: the plant is not seeing the commanded duty, so
    accumulating the error would only wind up."""
    duty, new_integrator = pi_duty(error_c, kp, ki, integrator, dt_s)
    if clipped_last_frame and error_c > 0.0:
        return duty, integrator
    return duty, new_integrator


def windows_with_balance(duty_pct, balance):
    """Whole windows to request this frame, and the signed balance left over:
    positive is owed to the heater, negative was granted early. Rounds at half
    a window, so the balance stays in [-0.5, 0.5)."""
    # Snap binary-float noise so that 2.4 + 0.1 rounds like the 2.5 it is.
    owed = round(duty_pct / 100.0 * FRAME_WINDOWS + balance, 9)
    windows = min(max(int(math.floor(owed + 0.5)), 0), FRAME_WINDOWS)
    if duty_pct >= 100.0:
        remainder = 0.0
    else:
        remainder = min(max(owed - windows, -0.5), 0.5 - 1e-12)
    return windows, remainder


def schedule_frame(requests, cap, rotate):
    """Place each heater's requested windows, round-robin, into the least
    loaded window that still has room under the cap. Returns the placement
    order as (heater, window) pairs plus the granted count per heater."""
    n = len(requests)
    requested = [min(r, FRAME_WINDOWS) for r in requests]
    masks = [0] * n
    granted = [0] * n
    load = [0] * FRAME_WINDOWS
    placements = []
    for _round in range(FRAME_WINDOWS):
        order = sorted(range(n), key=lambda i: (-(requested[i] - granted[i]), (i + rotate) % max(n, 1)))
        progressed = False
        for i in order:
            if granted[i] >= requested[i]:
                continue
            free = [w for w in range(FRAME_WINDOWS) if load[w] < cap and not (masks[i] >> w) & 1]
            if not free:
                continue
            w = min(free, key=lambda w: (load[w], (w + rotate) % FRAME_WINDOWS))
            masks[i] |= 1 << w
            load[w] += 1
            granted[i] += 1
            placements.append((i, w))
            progressed = True
        if not progressed:
            break
    return masks, granted, placements


# --------------------------------------------------------------------------
# A small first-order plant so the PI section has real dynamics to show
# --------------------------------------------------------------------------


def simulate(setpoint_c, kp, ki, minutes, cap_pct=None, hold_when_clipped=True, dt_s=0.5):
    """Heat a lump from room temperature. 100 % duty heats at 2 K/min; the
    lump loses heat with a 20 min time constant. `cap_pct(t_min)` models the
    scheduler granting less than the PI asked for while other groups hold the
    budget."""
    t_amb = 23.0
    rate_k_per_s = 2.0 / 60.0
    tau_s = 20.0 * 60.0
    steps = int(minutes * 60 / dt_s)
    temp = t_amb
    integrator = 0.0
    clipped = False
    out = np.zeros((steps, 5))  # t_min, temp, duty_asked, duty_granted, integrator
    for k in range(steps):
        error = setpoint_c - temp
        if hold_when_clipped:
            duty, integrator = pi_duty_under_budget(error, kp, ki, integrator, dt_s, clipped)
        else:
            duty, integrator = pi_duty(error, kp, ki, integrator, dt_s)
        granted = min(duty, cap_pct(k * dt_s / 60.0) if cap_pct else 100.0)
        clipped = granted < duty - 1e-9
        out[k] = (k * dt_s / 60.0, temp, duty, granted, integrator)
        temp += dt_s * (rate_k_per_s * granted / 100.0 - (temp - t_amb) / tau_s)
    return out


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


FONT = "Noto Sans"
MONO = "DejaVu Sans Mono"
# Widest any single text run may be, in scene units; the frame is 14.2 wide.
MAX_TEXT_W = 13.0


# Pango rounds glyph advances to whole pixels at the size it is asked for, so
# small text comes out with uneven letter spacing. Laying it out this many
# times larger and shrinking the result keeps the spacing even.
TEXT_OVERSAMPLE = 4


def ui_text(text, **kw):
    """Text in the scene's sans font; used directly and as the constructor
    for axis numbers and DecimalNumber so every glyph comes from one face."""
    kw.setdefault("font", FONT)
    size = kw.pop("font_size", DEFAULT_FONT_SIZE)
    # Pango wraps lines at the configured pixel width, so the oversampled
    # layout needs a proportionally wider canvas or long captions fold.
    canvas = {"pixel_width": config.pixel_width * TEXT_OVERSAMPLE * 2, "pixel_height": config.pixel_height * TEXT_OVERSAMPLE}
    with tempconfig(canvas):
        mob = Text(text, font_size=size * TEXT_OVERSAMPLE, **kw)
    return mob.scale(1 / TEXT_OVERSAMPLE)


def label(text, size=24, color=INK, **kw):
    """A text run that is never wider than the frame."""
    mob = ui_text(text, font_size=size, color=color, **kw)
    if mob.width > MAX_TEXT_W:
        mob.scale_to_fit_width(MAX_TEXT_W)
    return mob


def cell_grid(rows, cols, w=0.55, h=0.42, gap=0.06):
    """A rows x cols grid of rectangles; index as grid[r][c]."""
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell = Rectangle(width=w, height=h, stroke_width=1, stroke_color=MUTED, fill_color=CELL, fill_opacity=1)
            cell.move_to(RIGHT * c * (w + gap) + DOWN * r * (h + gap))
            row.append(cell)
        grid.append(row)
    return grid


def grid_group(grid):
    return VGroup(*[cell for row in grid for cell in row])


def bar(value, max_value, length, height, color, anchor_left):
    """A horizontal bar growing to the right from anchor_left."""
    width = max(1e-3, length * value / max_value)
    rect = Rectangle(width=width, height=height, fill_color=color, fill_opacity=0.9, stroke_width=0)
    rect.move_to(anchor_left, aligned_edge=LEFT)
    return rect


# --------------------------------------------------------------------------
# The scene
# --------------------------------------------------------------------------


class ThermalControl(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        self.title_card()
        self.section_frame()
        self.section_currents()
        self.section_pi()
        self.section_balance()
        self.section_schedule()
        self.section_clip()
        self.section_loop()

    # -- helpers -----------------------------------------------------------

    def clear_all(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self._cap = None

    def heading(self, text):
        h = label(text, 34, WHITE).to_edge(UP, buff=0.4)
        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.6)
        return h

    def caption(self, text, size=24, color=INK):
        """One short line at the bottom. It replaces whatever caption is up,
        so the screen never carries more than one sentence of prose."""
        c = label(text, size, color).to_edge(DOWN, buff=0.45)
        prev = getattr(self, "_cap", None)
        anims = [FadeIn(c)]
        if prev is not None and prev in self.mobjects:
            anims.append(FadeOut(prev))
        self.play(*anims, run_time=0.5)
        self._cap = c
        return c

    # -- 0. title ----------------------------------------------------------

    def title_card(self):
        title = label("Heater control on the NSV-502", 44, WHITE)
        sub = label("one frame every 500 ms", 26, MUTED)
        sub.next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(1.4)
        self.clear_all()

    # -- 1. the frame ------------------------------------------------------

    def frame_bar(self, y, bar_w=8.0, bar_h=0.6):
        """The 500 ms frame as a bar of eight windows, centred at height y.
        Returns (frame rectangle, everything drawn)."""
        frame = Rectangle(width=bar_w, height=bar_h, stroke_color=INK, stroke_width=2).move_to(UP * y)
        ticks = VGroup()
        wlabels = VGroup()
        for w in range(FRAME_WINDOWS):
            x = frame.get_left()[0] + w * bar_w / FRAME_WINDOWS
            if w:
                ticks.add(Line(RIGHT * x + UP * frame.get_top()[1], RIGHT * x + UP * frame.get_bottom()[1], stroke_color=MUTED, stroke_width=1))
            wlabels.add(label(f"w{w}", 20, MUTED).move_to(RIGHT * (x + bar_w / FRAME_WINDOWS / 2) + UP * (frame.get_top()[1] + 0.28)))
        span = label("500 ms frame", 20, INK).next_to(frame, LEFT, buff=0.3)
        return frame, VGroup(frame, ticks, wlabels, span)

    def section_frame(self):
        head = self.heading("1. One channel: the mask is a current waveform")
        bar_w = 8.0
        frame, frame_art = self.frame_bar(2.3, bar_w)
        self.play(Create(frame), FadeIn(frame_art[1:]), run_time=0.9)
        cap = self.caption("Every 500 ms frame has eight 62.5 ms windows.")
        self.wait(1.2)

        # One heater line's mask, as eight cells aligned under the windows.
        mask = 0b00100101
        next_mask = 0b01001010
        cells = cell_grid(1, FRAME_WINDOWS, w=bar_w / FRAME_WINDOWS - 0.06, h=0.4)[0]
        strip = VGroup(*cells).move_to(frame.get_center() + DOWN * 0.75).align_to(frame, LEFT).shift(RIGHT * 0.03)
        name = label("pwr1", 20, INK).next_to(strip, LEFT, buff=0.35)
        bits = label(f"0b{mask:08b}", 18, MUTED, font=MONO).next_to(strip, RIGHT, buff=0.35)
        self.play(FadeIn(strip), FadeIn(name), FadeIn(bits), run_time=0.6)
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption("One bit per window: set means switch closed.")

        def paint(m):
            return [Transform(cells[w], cells[w].copy().set_fill(ORANGE if (m >> w) & 1 else CELL, 1)) for w in range(FRAME_WINDOWS)]

        self.play(*paint(mask), run_time=0.7)
        self.wait(1.0)

        # The current axes, windows aligned with the bar above.
        axes = Axes(x_range=[0, 500, 500], y_range=[0, 1.0, 0.5], x_length=bar_w, y_length=2.0,
                    axis_config={"color": MUTED, "include_ticks": False},
                    y_axis_config={"include_numbers": True, "font_size": 18, "label_constructor": ui_text, "decimal_number_config": {"num_decimal_places": 1}})
        axes.shift(frame.get_left() - axes.c2p(0, 0) + DOWN * (frame.get_left()[1] + 2.0))
        y_lab = label("current, A", 18, MUTED).next_to(axes.y_axis, UP, buff=0.1).align_to(axes.y_axis, LEFT)
        self.play(Create(axes), FadeIn(y_lab), run_time=0.7)

        facts = VGroup(
            label("12 V rail: 11.7 V", 18, INK),
            label("heater: 13 Ω", 18, INK),
            label("I = V ÷ R = 0.9 A", 18, ORANGE),
            label("P = V · I = 10.6 W", 18, ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(axes, RIGHT, buff=0.35).align_to(axes, UP)
        self.play(FadeIn(facts), run_time=0.5)

        # Step waveform revealed by a playhead sweeping the frame.
        def waveform(m):
            xs, ys = [], []
            for w in range(FRAME_WINDOWS):
                amps = HEATER_A if (m >> w) & 1 else 0.0
                xs += [w * WINDOW_MS, (w + 1) * WINDOW_MS]
                ys += [amps, amps]
            return np.array(xs), np.array(ys)

        t_tr = ValueTracker(0.0)

        def partial_wave(m):
            xs, ys = waveform(m)

            def build():
                t = t_tr.get_value()
                keep = xs <= t
                px, py = list(xs[keep]), list(ys[keep])
                w = min(int(t // WINDOW_MS), FRAME_WINDOWS - 1)
                px.append(t)
                py.append(HEATER_A if (m >> w) & 1 else 0.0)
                if len(px) < 2:
                    px, py = [0, 1e-3], [py[-1], py[-1]]
                return axes.plot_line_graph(px, py, add_vertex_dots=False, line_color=ORANGE, stroke_width=4)

            return always_redraw(build)

        head_line = Rectangle(width=0.06, height=frame.get_top()[1] - axes.c2p(0, 0)[1], fill_color=TEAL, fill_opacity=0.8, stroke_width=0)
        head_line.add_updater(lambda hl: hl.move_to(frame.get_left() * [1, 0, 0] + RIGHT * t_tr.get_value() / 500 * bar_w + UP * axes.c2p(0, 0)[1], aligned_edge=DOWN + LEFT))
        wave = partial_wave(mask)
        self.add(head_line, wave)
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption("Closed: 0.9 A. Open: nothing.")
        self.play(t_tr.animate.set_value(500), run_time=4.0, rate_func=rate_functions.linear)
        self.wait(0.6)

        on_count = bin(mask).count("1")
        mean_a = on_count / FRAME_WINDOWS * HEATER_A
        mean_line = DashedLine(axes.c2p(0, mean_a), axes.c2p(500, mean_a), color=GREEN)
        # Kept out of the plot so the waveform cannot strike through it.
        mean_lab = VGroup(
            label("frame average", 16, GREEN),
            label(f"{on_count}/8 × 0.9 A = {mean_a:.2f} A", 16, GREEN),
            label(f"{on_count}/8 × 10.6 W = {on_count / FRAME_WINDOWS * HEATER_W:.1f} W", 16, GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).next_to(facts, DOWN, buff=0.25).align_to(facts, LEFT)
        self.play(Create(mean_line), FadeIn(mean_lab), run_time=0.6)
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption(f"Three windows of eight: {on_count / FRAME_WINDOWS * 100:g} % duty.")
        self.wait(2.8)

        # A new mask is staged and latched at window 0.
        self.play(FadeOut(cap), FadeOut(mean_line), FadeOut(mean_lab), run_time=0.3)
        cap = self.caption("A new mask takes effect at window 0.")
        self.remove(wave)
        t_tr.set_value(0.0)
        wave = partial_wave(next_mask)
        self.add(wave)
        self.play(Transform(bits, label(f"0b{next_mask:08b}", 18, GREEN, font=MONO).move_to(bits)), *paint(next_mask), run_time=0.5)
        self.play(t_tr.animate.set_value(500), run_time=4.0, rate_func=rate_functions.linear)
        head_line.clear_updaters()
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption("One line alone can only do eighths.")
        self.wait(2.5)
        self.clear_all()

    # -- 2. many channels: currents add -----------------------------------

    def section_currents(self):
        head = self.heading("2. Many channels: currents add")
        names = ["pwr1", "pwr11", "pwr15", "pwr9", "pwr12", "pwr14"]
        requests = [3, 2, 4, 1, 2, 3]
        budget_w = 40.0
        cap_lines = int(budget_w // HEATER_W)
        colors = [BLUE, GREEN, RED, ORANGE, PURPLE, TEAL]
        # Everyone starting at window 0: the obvious mask, and the wrong one.
        naive = [(1 << r) - 1 for r in requests]

        grid = cell_grid(len(names), FRAME_WINDOWS, w=0.6, h=0.3, gap=0.06)
        g = grid_group(grid).move_to(UP * 1.55 + RIGHT * 0.9)
        rowlab = VGroup(*[label(names[r], 18, colors[r]).next_to(grid[r][0], LEFT, buff=0.3) for r in range(len(names))])
        wlab = VGroup(*[label(f"w{w}", 18, MUTED).next_to(grid[0][w], UP, buff=0.12) for w in range(FRAME_WINDOWS)])
        self.play(FadeIn(g), FadeIn(rowlab), FadeIn(wlab), run_time=0.6)

        # The stack: each closed switch is a 0.9 A block in its window's column.
        base_y = -2.6
        amp_u = 0.4  # scene units per ampere
        col_x = [grid[0][w].get_center()[0] for w in range(FRAME_WINDOWS)]
        baseline = Line(RIGHT * (col_x[0] - 0.4) + UP * base_y, RIGHT * (col_x[-1] + 0.4) + UP * base_y, color=MUTED)
        yt = VGroup()
        for amps in (0.0, 1.8, 3.6, 5.4):
            yt.add(label(f"{amps:.1f} A", 14, MUTED).move_to(RIGHT * (col_x[0] - 0.75) + UP * (base_y + amps * amp_u)))
        stack_lab = label("supply current per window", 18, MUTED).next_to(baseline, DOWN, buff=0.55).align_to(baseline, LEFT)
        self.play(Create(baseline), FadeIn(yt), run_time=0.5)
        cap = self.caption("Each closed switch adds 0.9 A to its window.")

        heights = [0] * FRAME_WINDOWS
        blocks = VGroup()
        totals = VGroup(*[label("0", 14, MUTED).move_to(RIGHT * col_x[w] + UP * (base_y - 0.22)) for w in range(FRAME_WINDOWS)])
        self.add(totals)

        def total_text(w, color=None):
            n = heights[w]
            col = color or (RED if n > cap_lines else INK)
            return label(f"{n * HEATER_A:.1f} A", 14, col).move_to(totals[w])

        def add_row(r, m):
            anims = [grid[r][w].animate.set_fill(colors[r] if (m >> w) & 1 else CELL, 1) for w in range(FRAME_WINDOWS)]
            new_blocks = VGroup()
            for w in range(FRAME_WINDOWS):
                if (m >> w) & 1:
                    b = Rectangle(width=0.54, height=HEATER_A * amp_u, fill_color=colors[r], fill_opacity=0.85, stroke_width=0)
                    b.move_to(RIGHT * col_x[w] + UP * (base_y + heights[w] * HEATER_A * amp_u), aligned_edge=DOWN)
                    heights[w] += 1
                    new_blocks.add(b)
            blocks.add(new_blocks)
            anims += [FadeIn(new_blocks, shift=UP * 0.15)]
            anims += [Transform(totals[w], total_text(w, INK)) for w in range(FRAME_WINDOWS)]
            return anims

        for r in range(len(names)):
            self.play(*add_row(r, naive[r]), run_time=0.55)
            self.wait(0.15)
        self.wait(0.8)

        # The budget, as a line the stack must stay under.
        ceiling_a = budget_w / SUPPLY_V
        ceiling = DashedLine(RIGHT * (col_x[0] - 0.4) + UP * (base_y + ceiling_a * amp_u), RIGHT * (col_x[-1] + 0.4) + UP * (base_y + ceiling_a * amp_u), color=RED)
        ceiling_lab = label(f"{budget_w:.0f} W supply budget = {ceiling_a:.1f} A", 16, RED).next_to(ceiling, RIGHT, buff=0.15)
        self.play(Create(ceiling), FadeIn(ceiling_lab), run_time=0.6)
        over = [w for w in range(FRAME_WINDOWS) if heights[w] > cap_lines]
        self.play(*[Transform(totals[w], total_text(w)) for w in over], run_time=0.4)
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption(f"All starting at window 0: {heights[0] * HEATER_A:.1f} A, {heights[0] * HEATER_W:.0f} W, in one window.")
        self.wait(2.8)
        cap = self.caption("The loop knows each channel: 13 Ω, 0.9 A, 10.6 W.")
        self.wait(2.5)
        self.play(FadeOut(cap), run_time=0.2)
        rule = label(f"{budget_w:.0f} W ÷ {HEATER_W} W = {budget_w / HEATER_W:.2f}  →  at most {cap_lines} lines per window", 22, WHITE).move_to(UP * 0.05)
        self.play(FadeIn(rule), run_time=0.5)
        self.wait(2.5)

        # Reflow under the cap: same windows per heater, spread across the frame.
        masks, _, _ = schedule_frame(requests, cap_lines, rotate=0)
        self.play(FadeOut(blocks), run_time=0.4)
        heights = [0] * FRAME_WINDOWS
        blocks = VGroup()
        anims = []
        for r in range(len(names)):
            anims += add_row(r, masks[r])
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption(f"Same windows, spread out: never above {cap_lines} lines.")
        self.play(*anims, run_time=1.2)
        self.wait(2.5)
        cap = self.caption("Module limit 125 W; the bench supply sets 40 W.")
        self.wait(2.5)
        self.clear_all()

    # -- 2. RTD -> PI --------------------------------------------------------

    def section_pi(self):
        head = self.heading("3. RTD → PI: temperature error to duty")
        setpoint, kp, ki = 45.0, 20.0, 20.0 / 600.0
        # Every 4th sample (2 s of plant time) is plenty for the plot and the bars.
        data = simulate(setpoint, kp, ki, minutes=24.0)[::4]
        t = data[:, 0]
        temp = data[:, 1]
        duty = data[:, 2]
        integ = data[:, 4]

        axes = Axes(x_range=[0, 24, 6], y_range=[20, 50, 10], x_length=6.4, y_length=3.6,
                    axis_config={"include_numbers": True, "font_size": 20, "color": MUTED, "label_constructor": ui_text, "decimal_number_config": {"num_decimal_places": 0}})
        axes.to_edge(LEFT, buff=0.7).shift(DOWN * 0.3)
        xlab = label("minutes", 18, MUTED).next_to(axes.x_axis, DOWN, buff=0.15)
        ylab = label("RTD °C", 18, MUTED).next_to(axes.y_axis, UP, buff=0.1)
        sp_line = DashedLine(axes.c2p(0, setpoint), axes.c2p(24, setpoint), color=MUTED)
        sp_lab = label("setpoint 45 °C", 18, MUTED).next_to(sp_line, UP, buff=0.05).align_to(sp_line, RIGHT)
        self.play(Create(axes), FadeIn(xlab), FadeIn(ylab), Create(sp_line), FadeIn(sp_lab), run_time=0.9)

        law = VGroup(
            label("duty = Kp·e + Ki·∫e dt", 26, WHITE),
            label("e = setpoint − RTD", 20, MUTED),
            label("Kp = 20 % per °C", 20, MUTED),
            label("Ki = Kp ÷ 600 s", 20, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        law.to_edge(RIGHT, buff=0.6).shift(UP * 1.9)
        self.play(FadeIn(law), run_time=0.7)
        cap = self.caption("P acts on the error; I removes what remains.")
        self.wait(2.5)
        self.play(FadeOut(cap), run_time=0.2)

        # Live bars: error, P term, I term, duty.
        k_tr = ValueTracker(0)
        bar_x = 2.0
        bar_len = 3.0
        rows = [("error °C", RED, lambda k: setpoint - temp[k], 25.0),
                ("P  %", BLUE, lambda k: kp * (setpoint - temp[k]), 100.0),
                ("I  %", PURPLE, lambda k: ki * integ[k], 100.0),
                ("duty %", ORANGE, lambda k: duty[k], 100.0)]
        bars = VGroup()
        for i, (name, color, fn, mx) in enumerate(rows):
            y = -0.1 - i * 0.62
            nm = label(name, 18, INK).move_to(RIGHT * (bar_x + 0.55) + UP * y)
            anchor = RIGHT * (bar_x + 1.35) + UP * y
            track = Rectangle(width=bar_len, height=0.3, stroke_color=MUTED, stroke_width=1, fill_color=CELL, fill_opacity=1).move_to(anchor, aligned_edge=LEFT)
            b = always_redraw(lambda fn=fn, mx=mx, color=color, anchor=anchor: bar(min(max(fn(int(k_tr.get_value())), 0.0), mx), mx, bar_len, 0.3, color, anchor))
            num = always_redraw(lambda fn=fn, anchor=anchor: DecimalNumber(fn(int(k_tr.get_value())), num_decimal_places=1, font_size=18, color=INK, mob_class=ui_text).next_to(anchor + RIGHT * bar_len, RIGHT, buff=0.15))
            bars.add(nm, track, b, num)
        self.play(FadeIn(bars), run_time=0.6)

        curve = always_redraw(lambda: axes.plot_line_graph(t[: max(2, int(k_tr.get_value()))], temp[: max(2, int(k_tr.get_value()))], add_vertex_dots=False, line_color=GREEN, stroke_width=3))
        dot = always_redraw(lambda: axes.plot_line_graph([t[int(k_tr.get_value())]], [temp[int(k_tr.get_value())]], vertex_dot_radius=0.07, vertex_dot_style={"fill_color": GREEN}))
        self.add(curve, dot)
        cap = self.caption("Far below setpoint: 100 %, integrator frozen.")
        n = len(t)
        self.play(k_tr.animate.set_value(int(n * 0.42)), run_time=5.0, rate_func=rate_functions.linear)
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption("Inside the band the integrator carries the load.")
        self.play(k_tr.animate.set_value(n - 1), run_time=7.0, rate_func=rate_functions.linear)
        self.wait(1.0)
        self.play(FadeOut(cap), run_time=0.2)
        cap = self.caption("One PI per RTD group.")
        self.wait(2.0)
        cap = self.caption("Duty is a percentage; frames take whole windows.")
        self.wait(2.5)
        self.clear_all()

    # -- 4. duty -> windows with a balance ------------------------------------

    def strip(self, lit, w=0.3, h=0.28, gap=0.04, balance=0.0):
        """One frame's eight windows as a small strip with the first `lit` on.
        A positive balance (owed) is drawn in purple at the start of the next
        cell; a negative one (granted ahead) in teal at the end of the last lit
        cell."""
        cells = cell_grid(1, FRAME_WINDOWS, w=w, h=h, gap=gap)[0]
        for i, cell in enumerate(cells):
            cell.set_fill(ORANGE if i < lit else CELL, 1)
        group = VGroup(*cells)
        if balance > 0 and lit < FRAME_WINDOWS:
            part = Rectangle(width=max(1e-3, balance * w), height=h * 0.7, fill_color=PURPLE, fill_opacity=0.9, stroke_width=0)
            part.move_to(cells[lit].get_left(), aligned_edge=LEFT)
            group.add(part)
        elif balance < 0 and lit > 0:
            part = Rectangle(width=max(1e-3, -balance * w), height=h * 0.7, fill_color=TEAL, fill_opacity=0.9, stroke_width=0)
            part.move_to(cells[lit - 1].get_right(), aligned_edge=RIGHT)
            group.add(part)
        return group

    def section_balance(self):
        head = self.heading("4. Duty → windows: the deficit ledger")
        duty = 30.0

        # -- A. the problem ------------------------------------------------
        intro = label("30 % of 8 windows = 2.4 windows per frame", 26, WHITE).next_to(head, DOWN, buff=0.35)
        self.play(FadeIn(intro), run_time=0.6)
        cap = self.caption("A frame holds 2 or 3, never 2.4.")
        self.wait(1.5)

        # -- B. plain rounding is short forever --------------------------------
        q = label("Round to nearest?", 24, RED).move_to(UP * 1.3)
        self.play(FadeIn(q), run_time=0.4)
        naive = VGroup()
        for f in range(5):
            s = self.strip(2).move_to(RIGHT * (-4.2 + f * 2.1) + UP * 0.5)
            fl = label(f"frame {f + 1}", 16, MUTED).next_to(s, UP, buff=0.1)
            naive.add(VGroup(s, fl))
            self.play(FadeIn(naive[-1]), run_time=0.25)
        tally = label("asked 12, delivered 10: 25 %, not 30 %", 24, RED).move_to(DOWN * 1.0)
        self.play(FadeIn(tally), run_time=0.4)
        self.wait(1.5)
        cap = self.caption("The missing 0.4 per frame is never repaid.", color=RED)
        self.wait(2.0)
        self.play(FadeOut(q), FadeOut(naive), FadeOut(tally), run_time=0.5)

        # -- C. the balance, drawn as part of a window inside the strip ----------------
        rule = label("carry the balance into the next frame", 26, GREEN).move_to(UP * 1.75)
        self.play(FadeIn(rule), FadeOut(cap), run_time=0.5)
        self.wait(0.8)
        self.play(FadeOut(intro), rule.animate.next_to(head, DOWN, buff=0.3), run_time=0.5)
        cap = self.caption("At least half a window owed: switch it on now.")

        cw, ch, gap = 0.62, 0.34, 0.05
        pitch = cw + gap
        x0 = -4.3  # left edge of every strip
        legend = VGroup(
            VGroup(Rectangle(width=0.3, height=0.2, fill_color=ORANGE, fill_opacity=1, stroke_width=0), label("on", 14, INK)).arrange(RIGHT, buff=0.1),
            VGroup(Rectangle(width=0.3, height=0.2, fill_color=PURPLE, fill_opacity=0.9, stroke_width=0), label("owed", 14, INK)).arrange(RIGHT, buff=0.1),
            VGroup(Rectangle(width=0.3, height=0.2, fill_color=TEAL, fill_opacity=0.9, stroke_width=0), label("ahead", 14, INK)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.5).move_to(UP * 1.4 + RIGHT * 0.3)
        self.play(FadeIn(legend), run_time=0.3)

        def span(y, start, end, color, opacity=0.9):
            """A bar from window position `start` to `end`, in windows, over a strip at height y."""
            width = max(1e-3, (end - start) * pitch)
            rect = Rectangle(width=width, height=ch * 0.7, fill_color=color, fill_opacity=opacity, stroke_width=0)
            rect.move_to(RIGHT * (x0 + start * pitch) + UP * y, aligned_edge=LEFT)
            return rect

        balance = 0.0
        total = 0
        tail = None
        per_frame = duty / 100 * FRAME_WINDOWS
        for f in range(6):
            windows, new_balance = windows_with_balance(duty, balance)
            owed = round(per_frame + balance, 9)
            y = 0.9 - f * 0.47
            cells = cell_grid(1, FRAME_WINDOWS, w=cw, h=ch, gap=gap)[0]
            strip = VGroup(*cells).move_to(UP * y).align_to(RIGHT * x0, LEFT)
            flab = label(f"frame {f + 1}", 16, MUTED).next_to(strip, LEFT, buff=0.25)
            self.play(FadeIn(strip), FadeIn(flab), run_time=0.25)
            # The balance slides down from the last strip, leaving a faint mark
            # behind: owed part first, then the new 2.4; a part granted ahead is
            # paid back out of the end of the new 2.4.
            carried = None
            if balance > 0 and tail is not None:
                carried = tail.copy()
                self.add(carried)
                self.play(Transform(carried, span(y, 0.0, balance, PURPLE)), tail.animate.set_opacity(0.35), run_time=0.5)
            start = max(balance, 0.0)
            fresh = span(y, start, start + 1e-3, ORANGE)
            self.add(fresh)
            self.play(Transform(fresh, span(y, start, start + per_frame, ORANGE)), run_time=0.6)
            if balance < 0 and tail is not None:
                carried = tail.copy()
                self.add(carried)
                self.play(Transform(carried, span(y, owed, per_frame, TEAL)), tail.animate.set_opacity(0.35), run_time=0.5)
            self.wait(0.2)
            # Round to the nearest window: whole windows switch on; the rest of the
            # account stays purple (owed) or teal (granted ahead).
            snap = [cells[w].animate.set_fill(ORANGE, 1) for w in range(windows)]
            if new_balance > 0:
                new_tail = span(y, windows, owed, PURPLE)
            elif new_balance < 0:
                new_tail = span(y, owed, windows, TEAL)
            else:
                new_tail = None
            fades = [FadeOut(fresh)] + ([FadeOut(carried)] if carried is not None else [])
            shows = [FadeIn(new_tail)] if new_tail is not None else []
            total += windows
            if new_balance > 0:
                note = label(f"{new_balance:.1f} owed", 16, PURPLE)
            elif new_balance < 0:
                note = label(f"{-new_balance:.1f} ahead", 16, TEAL)
            else:
                note = label("even", 16, MUTED)
            words = VGroup(label(f"{windows} on", 18, ORANGE), note).arrange(RIGHT, buff=0.25)
            words.next_to(strip, RIGHT, buff=0.3)
            self.play(*snap, *fades, *shows, FadeIn(words), run_time=0.5)
            tail = new_tail
            if f == 1:
                cap = self.caption("2.8 owed: over half, so three on; 0.2 ahead.", color=TEAL)
                self.wait(1.4)
            if f == 2:
                cap = self.caption("2.2 owed: two on, 0.2 owed.", color=PURPLE)
                self.wait(1.2)
            balance = new_balance
        self.wait(0.6)
        cap = self.caption("The balance never exceeds half a window either way.", color=GREEN)
        self.wait(2.0)
        cap = self.caption("Five frames: 12 on for 12.0 asked. Exactly 30 %.")
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head], run_time=0.5)
        self._cap = None

        # -- D. the rules that keep it bounded, one line at a time --------------------
        for line in (
            "One balance per heater line.",
            "It never exceeds half a window either way.",
            "Clipped windows are not banked; the PI handles those.",
        ):
            rule_line = label(line, 30, WHITE).move_to(UP * 0.3)
            self.play(FadeIn(rule_line, shift=UP * 0.1), run_time=0.5)
            self.wait(2.0)
            self.play(FadeOut(rule_line), run_time=0.3)

        # -- E. a low duty still gets delivered ----------------------------------------
        low = label("10 %: 0.8 windows per frame", 26, WHITE).next_to(head, DOWN, buff=0.4)
        self.play(FadeIn(low), run_time=0.5)
        balance, got = 0.0, []
        strips = VGroup()
        ledger = VGroup()
        for f in range(10):
            windows, balance = windows_with_balance(10.0, balance)
            got.append(windows)
            s = self.strip(windows, w=0.13, h=0.26, gap=0.02, balance=balance).move_to(RIGHT * (-5.4 + f * 1.2) + UP * 0.6)
            fl = label(f"{f + 1}", 14, MUTED).next_to(s, UP, buff=0.08)
            c = label(f"{balance:+.1f}", 16, PURPLE if balance > 0 else (TEAL if balance < 0 else MUTED)).next_to(s, DOWN, buff=0.1)
            strips.add(VGroup(s, fl))
            ledger.add(c)
            self.play(FadeIn(strips[-1]), FadeIn(c), run_time=0.22)
        lab = label("balance", 16, MUTED).next_to(ledger, DOWN, buff=0.15)
        result = label(f"{sum(got)} windows in 10 frames: {sum(got) / 80 * 100:g} %", 26, GREEN).move_to(DOWN * 1.4)
        self.play(FadeIn(lab), run_time=0.3)
        self.play(FadeIn(result), run_time=0.5)
        cap = self.caption("Any duty is delivered on average.")
        self.wait(2.5)
        self.clear_all()

    # -- 4. scheduling under a cap ------------------------------------------

    def section_schedule(self):
        head = self.heading("5. Placing windows under the cap")
        names = ["pwr1", "pwr11", "pwr15", "pwr9", "pwr12", "pwr14"]
        requests = [3, 2, 4, 1, 2, 3]
        cap = 3
        note = label("at most 3 lines in any window", 24, WHITE).next_to(head, DOWN, buff=0.3)
        self.play(FadeIn(note), run_time=0.6)

        grid = cell_grid(len(names), FRAME_WINDOWS, w=0.6, h=0.42, gap=0.06)
        g = grid_group(grid).move_to(DOWN * 0.55 + RIGHT * 0.9)
        rowlab = VGroup(*[label(names[r], 20, INK).next_to(grid[r][0], LEFT, buff=0.3) for r in range(len(names))])
        asked = VGroup(*[label(f"asks {requests[r]}", 18, MUTED).next_to(grid[r][-1], RIGHT, buff=0.3) for r in range(len(names))])
        wlab = VGroup(*[label(f"w{w}", 18, MUTED).next_to(grid[0][w], UP, buff=0.15) for w in range(FRAME_WINDOWS)])
        loads = [0] * FRAME_WINDOWS
        load_txt = VGroup(*[label("0", 20, INK).next_to(grid[-1][w], DOWN, buff=0.2) for w in range(FRAME_WINDOWS)])
        load_lab = label("on / cap 3", 18, MUTED).next_to(load_txt[0], LEFT, buff=0.3)
        self.play(FadeIn(g), FadeIn(rowlab), FadeIn(asked), FadeIn(wlab), FadeIn(load_txt), FadeIn(load_lab), run_time=0.7)

        masks, granted, placements = schedule_frame(requests, cap, rotate=0)
        pre = self.caption("The ledger sets the counts; the scheduler picks windows.")
        self.wait(2.5)
        pre = self.caption("Most owed goes first; ties rotate each frame.")
        self.wait(2.0)
        self.play(FadeOut(pre), run_time=0.2)
        self._cap = None
        round_lab = label("round 1", 22, TEAL).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(round_lab), run_time=0.3)
        seen = set()
        rnd = 1
        for i, w in placements:
            if i in seen:
                seen = {i}
                rnd += 1
                self.play(Transform(round_lab, label(f"round {rnd}", 22, TEAL).to_edge(DOWN, buff=0.5)), run_time=0.25)
            else:
                seen.add(i)
            loads[w] += 1
            col = ORANGE if loads[w] < cap else RED
            self.play(
                grid[i][w].animate.set_fill(ORANGE, 1),
                Transform(load_txt[w], label(str(loads[w]), 20, col).move_to(load_txt[w])),
                run_time=0.22,
            )
        self.wait(0.5)
        self.play(FadeOut(round_lab), run_time=0.2)
        cap_txt = self.caption("15 placed, none above 3.", color=GREEN)
        self.wait(1.5)
        cap_txt = self.caption("Next frame the rotation moves them.")
        masks2, _, _ = schedule_frame(requests, cap, rotate=3)
        anims = []
        for r in range(len(names)):
            for w in range(FRAME_WINDOWS):
                anims.append(grid[r][w].animate.set_fill(ORANGE if (masks2[r] >> w) & 1 else CELL, 1))
        counts = [sum((masks2[r] >> w) & 1 for r in range(len(names))) for w in range(FRAME_WINDOWS)]
        for w in range(FRAME_WINDOWS):
            anims.append(Transform(load_txt[w], label(str(counts[w]), 20, ORANGE if counts[w] < cap else RED).move_to(load_txt[w])))
        self.play(*anims, run_time=0.8)
        self.wait(2.0)
        cap_txt = self.caption("Spreads wear, flattens the load.")
        self.wait(2.0)
        cap_txt = self.caption("If it cannot fit everything: clipped.")
        self.wait(2.5)
        self.clear_all()

    # -- 5. clipping and the integrator hold --------------------------------

    def section_clip(self):
        head = self.heading("6. When the cap clips, pause the integrator")
        setpoint, kp, ki = 45.0, 20.0, 20.0 / 600.0
        squeeze = (30.0, 60.0)
        minutes = 100.0

        def cap(t_min):
            # Another group is heating from cold and takes most of the budget.
            return 25.0 if squeeze[0] <= t_min < squeeze[1] else 100.0

        hold = simulate(setpoint, kp, ki, minutes, cap_pct=cap, hold_when_clipped=True)
        wind = simulate(setpoint, kp, ki, minutes, cap_pct=cap, hold_when_clipped=False)

        intro = label("a cold group takes the budget for 30 min", 24, WHITE).next_to(head, DOWN, buff=0.25)
        self.play(FadeIn(intro), run_time=0.6)

        axis_cfg = {"include_numbers": True, "font_size": 18, "color": MUTED, "label_constructor": ui_text, "decimal_number_config": {"num_decimal_places": 0}}
        ax_t = Axes(x_range=[0, minutes, 20], y_range=[20, 50, 10], x_length=6.0, y_length=2.9, axis_config=axis_cfg)
        ax_i = Axes(x_range=[0, minutes, 20], y_range=[0, 80, 20], x_length=6.0, y_length=2.9, axis_config=axis_cfg)
        ax_t.to_edge(LEFT, buff=0.8).shift(DOWN * 0.9)
        ax_i.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.9)
        t_lab = label("RTD °C", 18, MUTED).next_to(ax_t, UP, buff=0.1).align_to(ax_t, LEFT)
        i_lab = label("integral term, % of duty", 18, MUTED).next_to(ax_i, UP, buff=0.1).align_to(ax_i, LEFT)
        m_lab = VGroup(label("minutes", 16, MUTED).next_to(ax_t.x_axis, DOWN, buff=0.1), label("minutes", 16, MUTED).next_to(ax_i.x_axis, DOWN, buff=0.1))
        sp = DashedLine(ax_t.c2p(0, setpoint), ax_t.c2p(minutes, setpoint), color=MUTED)
        bands = VGroup()
        for ax, y0, y1 in ((ax_t, 20, 50), (ax_i, 0, 80)):
            lo, hi = ax.c2p(squeeze[0], y0), ax.c2p(squeeze[1], y1)
            band = Rectangle(width=hi[0] - lo[0], height=hi[1] - lo[1], fill_color=MUTED, fill_opacity=0.25, stroke_width=0)
            band.move_to((lo + hi) / 2)
            bands.add(band)
        band_lab = label("squeezed:\ngranted ≤ 25 %", 14, INK, line_spacing=0.8).move_to(bands[0].get_bottom() + UP * 0.45)
        self.play(Create(ax_t), Create(ax_i), FadeIn(t_lab), FadeIn(i_lab), FadeIn(m_lab), Create(sp), FadeIn(bands), FadeIn(band_lab), run_time=0.9)

        k_tr = ValueTracker(2)

        def partial(ax, data, col, idx, scale=1.0):
            return always_redraw(lambda: ax.plot_line_graph(data[: int(k_tr.get_value()), 0], scale * data[: int(k_tr.get_value()), idx], add_vertex_dots=False, line_color=col, stroke_width=3))

        # Plot every 8th sample (4 s of plant time): the curves are smooth at
        # that spacing and the per-frame redraw cost drops eightfold.
        wind_plot, hold_plot = wind[::8], hold[::8]
        wind_t = partial(ax_t, wind_plot, RED, 1)
        hold_t = partial(ax_t, hold_plot, GREEN, 1)
        wind_i = partial(ax_i, wind_plot, RED, 4, ki)
        hold_i = partial(ax_i, hold_plot, GREEN, 4, ki)
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=RED, stroke_width=4), label("integrate anyway", 18, INK)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=GREEN, stroke_width=4), label("hold while clipped", 18, INK)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_edge(DOWN, buff=0.35)
        self.add(wind_t, hold_t, wind_i, hold_i)
        self.play(FadeIn(legend), run_time=0.4)
        n = len(hold_plot)
        self.play(k_tr.animate.set_value(n - 1), run_time=9.0, rate_func=rate_functions.linear)
        lift = int(squeeze[1] * 60 / 0.5)
        self.play(FadeOut(legend), run_time=0.3)
        cap = self.caption(f"At the lift: {ki * hold[lift, 4]:.0f} % held vs {ki * wind[lift, 4]:.0f} % wound.")
        self.wait(2.5)
        # Why the clipped windows are not put on the ledger: one line at a time,
        # over the plots.
        cap = self.caption("Why not bank clipped windows in the ledger?")
        self.wait(2.0)
        cap = self.caption("That would be a second, unbounded integrator.")
        self.wait(2.0)
        cap = self.caption("So the PI holds instead; nothing can run away.", color=GREEN)
        self.wait(2.5)
        self.clear_all()

    # -- 6. the loop --------------------------------------------------------

    def section_loop(self):
        head = self.heading("7. The loop, every 500 ms")
        steps = [
            ("RTD read", "sweep frame, °C per lane", GREEN),
            ("PI per group", "error → duty %", BLUE),
            ("window balance", "duty → whole windows", PURPLE),
            ("scheduler", "cap, rotation → masks", TEAL),
            ("window masks", "staged, latched at w0", ORANGE),
            ("heaters", "62.5 ms slots of 10.6 W", RED),
        ]
        boxes = VGroup()
        for name, sub, col in steps:
            box = RoundedRectangle(width=3.4, height=1.0, corner_radius=0.15, stroke_color=col, stroke_width=2, fill_color=CELL, fill_opacity=1)
            txt = VGroup(label(name, 22, WHITE), label(sub, 16, MUTED)).arrange(DOWN, buff=0.08).move_to(box)
            boxes.add(VGroup(box, txt))
        top = VGroup(boxes[0], boxes[1], boxes[2]).arrange(RIGHT, buff=0.9).shift(UP * 1.0)
        bottom = VGroup(boxes[5], boxes[4], boxes[3]).arrange(RIGHT, buff=0.9).shift(DOWN * 1.4)
        arrows = VGroup(
            Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.1, color=MUTED),
            Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.1, color=MUTED),
            Arrow(boxes[2].get_bottom(), boxes[3].get_top(), buff=0.1, color=MUTED),
            Arrow(boxes[3].get_left(), boxes[4].get_right(), buff=0.1, color=MUTED),
            Arrow(boxes[4].get_left(), boxes[5].get_right(), buff=0.1, color=MUTED),
            Arrow(boxes[5].get_top(), boxes[0].get_bottom(), buff=0.1, color=MUTED),
        )
        self.play(FadeIn(top), FadeIn(bottom), run_time=0.8)
        self.play(Create(arrows), run_time=1.0)
        fb_arrow = DashedLine(boxes[3].get_corner(UP + LEFT) + RIGHT * 0.3, boxes[1].get_corner(DOWN + RIGHT) + LEFT * 0.3, color=RED, stroke_width=2).add_tip(tip_length=0.2)
        feedback = label("clipped last frame?", 18, RED).next_to(fb_arrow.get_center(), RIGHT, buff=0.35)
        self.play(Create(fb_arrow), FadeIn(feedback), run_time=0.7)
        cap = self.caption("Two frames a second.")
        for _ in range(2):
            for i in range(6):
                self.play(Indicate(boxes[i][0], color=steps[i][2], scale_factor=1.05), run_time=0.3)
        self.wait(1.0)

        # One heater, one frame, with real numbers, one line at a time.
        cap = self.caption("pwr11: RTD 44.6 °C, error 0.4 °C, duty 18 %.", color=BLUE)
        self.wait(2.2)
        cap = self.caption("1.44 windows + 0.30 owed: 2 on, 0.26 ahead.", color=PURPLE)
        self.wait(2.2)
        cap = self.caption("Two emptiest windows → mask 0b00100001.", color=ORANGE)
        self.wait(2.5)
        self.clear_all()
