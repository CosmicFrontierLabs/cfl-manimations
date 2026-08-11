#!/usr/bin/env python3
"""
Zodiacal Plate Solver Animation using Manim

Walks through the five-stage pipeline from Lang et al. 2010, as
implemented in github.com/meawoppl/zodiacal:

  1. Source extraction    — detect bright sources in the image
  2. Quad formation       — pick a 4-star asterism, compute scale-invariant code
                            (visualized as a transformation into normalized code space)
  3. Code matching        — kd-tree lookup against prebuilt index
  4. Hypothesis fitting   — TAN-WCS from matched correspondences
  5. Bayesian verification — log-odds decision over remaining stars

Layout discipline: the image frame is fixed in the LEFT panel, all
text/legends/diagrams live strictly outside it (top header strip,
bottom caption strip, or the RIGHT info panel).

A/B/C/D quad convention (used for ALL quad visuals in this animation
and intended to be reused verbatim by sibling visualization tools so the
same star plays the same role + wears the same color everywhere):

  label  role                              manim const  hex       RGB
  -----  --------------------------------  -----------  --------  --------------
  A      backbone anchor — code (0, 0)     RED          #FC6255   (252, 98, 85)
  B      backbone tip   — code (1, 1)      GREEN        #83C167   (131, 193, 103)
  C      first interior star  (cx, cy)     BLUE         #58C4DD   (88, 196, 221)
  D      second interior star (dx, dy)     ORANGE       #FF862F   (255, 134, 47)

  AB-line       : YELLOW   #FFFF00  (drawn solid, stroke 3)
  AC, AD lines  : BLUE_C   #58C4DD  (drawn solid, stroke 2)
  AB-circle     : GREY_C dashed     (the |AB|/2 validity circle)

Relative geometry of the quad (Lang et al. 2010, mirrored in zodiacal):

  - A and B are the two outermost stars of the chosen group; |AB| sets
    the plate-scale reference. C and D are two stars that lie STRICTLY
    INSIDE the AB-diameter circle (the circle whose diameter is the AB
    segment) — that's the validity constraint, quads that fail it get
    rejected by `pick_valid_quad`.
  - The rigid transform "translate so A → (0,0), rotate so AB aligns
    with the (+x, +y) diagonal, scale so |AB| = √2" carries everything
    into "code space". In code space:
        A = (0, 0)
        B = (1, 1)
        C = (cx, cy)   with (cx − 0.5)² + (cy − 0.5)² < 0.5
        D = (dx, dy)   same constraint
    i.e. C and D lie inside the circumscribed circle of the unit square
    (radius √2/2 ≈ 0.707, centered at (0.5, 0.5)), which is the image of
    the AB-diameter circle under the same transform.
  - C/D ordering is disambiguated (astrometry.net uses cx ≤ dx) so the
    same physical quad always produces the same 4-D code.
  - The 4-D code (cx, cy, dx, dy) ∈ ℝ⁴ is what the index stores; the
    kd-tree match in stage 3 is a nearest-neighbour lookup on this code.
"""

from __future__ import annotations

import numpy as np
from manim import *

rng = np.random.default_rng(42)

# Bump this when the upstream zodiacal release we're depicting changes.
ZODIACAL_VERSION = "0.4.1"

# ---------- layout constants -------------------------------------------------
# Screen is ~14.22 wide × 8 tall in manim units (default config).
# Image frame lives on the LEFT, info panel on the RIGHT, with header/caption
# strips above/below. NO text ever lands inside the image frame.

FRAME_CENTER = LEFT * 3.5
FRAME_W = 6.0
FRAME_H = 5.0
FRAME_LEFT = -3.5 - FRAME_W / 2  # = -6.5
FRAME_RIGHT = -3.5 + FRAME_W / 2  # = -0.5
FRAME_TOP = FRAME_H / 2  # = 2.5
FRAME_BOTTOM = -FRAME_H / 2  # = -2.5

INFO_CENTER = RIGHT * 3.7
INFO_W = 6.4
INFO_H = 5.0
INFO_LEFT = 3.7 - INFO_W / 2  # = 0.5
INFO_RIGHT = 3.7 + INFO_W / 2

HEADER_Y = 3.55
CAPTION_Y = -3.55

# ---------- helpers ----------------------------------------------------------


def make_starfield(n_real: int, n_noise: int, half_w=2.6, half_h=2.0):
    """Return ((real_xy, real_mag), (noise_xy, noise_mag)).

    Returned coordinates are already in scene-space, centered on FRAME_CENTER,
    with half_w/half_h chosen so the points sit comfortably inside the frame
    (which has half-extent 3.0 × 2.5).
    """
    cx, cy = FRAME_CENTER[0], FRAME_CENTER[1]
    real_xy = np.column_stack(
        [
            rng.uniform(cx - half_w, cx + half_w, size=n_real),
            rng.uniform(cy - half_h, cy + half_h, size=n_real),
        ]
    )
    real_flux = rng.uniform(0.6, 1.0, size=n_real)
    noise_xy = np.column_stack(
        [
            rng.uniform(cx - half_w, cx + half_w, size=n_noise),
            rng.uniform(cy - half_h, cy + half_h, size=n_noise),
        ]
    )
    noise_flux = rng.uniform(0.05, 0.18, size=n_noise)
    return (real_xy, real_flux), (noise_xy, noise_flux)


def header_text(s):
    return Text(s, font_size=30, weight=BOLD).move_to([0, HEADER_Y, 0])


def caption_text(s):
    return Text(s, font_size=20, color=GREY_B).move_to([0, CAPTION_Y, 0])


# ---- Index table layout (shared by stage 2 and stage 2b) -------------------
# The table sits BELOW the code-space graph (which dominates y > -1.4 in the
# right info panel). New rows are inserted just below the header — the
# previous rows shift down by one row_h each time a new row arrives, with
# older rows eventually scrolling off the bottom of the panel.
TABLE_COL_X = [
    INFO_LEFT + 0.4,   # RA
    INFO_LEFT + 1.65,  # Dec
    INFO_LEFT + 2.85,  # |AB|
    INFO_LEFT + 3.85,  # code
]
TABLE_TOP_Y = -1.55
TABLE_ROW_H = 0.32


def table_cell(text, x, y, color=GREY_B, size=14):
    t = Text(text, font_size=size, color=color)
    t.move_to([x, y, 0], aligned_edge=LEFT)
    return t


def colored_code_cell(code, x, y, size=14):
    """Code = (cx, cy, dx, dy) with cx,cy in BLUE and dx,dy in ORANGE."""
    tokens = VGroup(
        Text("(", font_size=size, color=GREY_B),
        Text(f"{code[0]:+.2f}", font_size=size, color=BLUE),
        Text(",", font_size=size, color=GREY_B),
        Text(f"{code[1]:+.2f}", font_size=size, color=BLUE),
        Text(",", font_size=size, color=GREY_B),
        Text(f"{code[2]:+.2f}", font_size=size, color=ORANGE),
        Text(",", font_size=size, color=GREY_B),
        Text(f"{code[3]:+.2f}", font_size=size, color=ORANGE),
        Text(")", font_size=size, color=GREY_B),
    ).arrange(RIGHT, buff=0.03)
    tokens.move_to([x, y, 0], aligned_edge=LEFT)
    return tokens


def make_table_row(ra_str, dec_str, ab_arcsec, code, y):
    return VGroup(
        table_cell(ra_str, TABLE_COL_X[0], y, color=RED),
        table_cell(dec_str, TABLE_COL_X[1], y, color=RED),
        table_cell(f"{ab_arcsec:.0f}", TABLE_COL_X[2], y, color=YELLOW),
        colored_code_cell(code, TABLE_COL_X[3], y),
    )


def quad_code_2d(xy):
    """Replicate zodiacal's compute_code for a flat 4-point quad in 2D.

    Stars are ordered A, B, C, D. Returns (code_2d, normalized_corners) where
    normalized_corners are the positions of A, B, C, D in code space:
        A → (0, 0), B → (1, 1), C → (code[0], code[1]), D → (code[2], code[3]).
    """
    a, b, c, d = xy[0], xy[1], xy[2], xy[3]
    ab = b - a
    scale = ab @ ab
    inv = 1.0 / scale
    cos_t = (ab[1] + ab[0]) * inv
    sin_t = (ab[1] - ab[0]) * inv

    code = []
    for star in (c, d):
        ad = star - a
        x = ad[0] * cos_t + ad[1] * sin_t
        y = -ad[0] * sin_t + ad[1] * cos_t
        code.extend([x, y])

    normalized = [
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([code[0], code[1]]),
        np.array([code[2], code[3]]),
    ]
    return np.array(code), normalized


def pick_n_valid_quads(xy, n, exclude_anchor_pairs=None):
    """Find up to n valid quads (C, D inside the AB-diameter circle), iterating
    short pairs first. Returns a list of [A, B, C, D] index lists. Quads may
    share stars but won't share the same (A, B) anchor pair as ones in
    exclude_anchor_pairs.
    """
    exclude_anchor_pairs = set(exclude_anchor_pairs or [])
    nstars = len(xy)
    pairs_by_len = []
    for i in range(nstars):
        for j in range(i + 1, nstars):
            d2 = float(np.sum((xy[i] - xy[j]) ** 2))
            pairs_by_len.append((d2, i, j))
    pairs_by_len.sort()

    out = []
    for _, a, b in pairs_by_len:
        if (a, b) in exclude_anchor_pairs:
            continue
        mid = (xy[a] + xy[b]) / 2
        r = np.linalg.norm(xy[b] - xy[a]) / 2
        inside = [
            k
            for k in range(nstars)
            if k != a and k != b and np.linalg.norm(xy[k] - mid) < r * 0.95
        ]
        if len(inside) >= 2:
            out.append([a, b, inside[0], inside[1]])
            if len(out) >= n:
                break
    return out


def fake_radec_floats_for_anchor(xy_point):
    """Returns (ra_deg, dec_deg) as floats. The field is centered at
    (RA, Dec) = (180°, 45°) and spans exactly 2° × 2° on the sky so the
    frame's four corners land on integer degrees. RA increases to the WEST
    (= left on a standard astronomical image); Dec increases NORTH (= up).

    Frame is FRAME_W=6.0 wide × FRAME_H=5.0 tall in scene space, so the
    per-axis scales are different (RA uses 1/3 °/unit, Dec uses 0.4 °/unit).
    """
    base_ra_deg = 180.0
    base_dec_deg = 45.0
    ra_deg_per_unit = 1.0 / 3.0  # frame_w = 6.0 → 2° span
    dec_deg_per_unit = 0.4  # frame_h = 5.0 → 2° span
    ra_deg = base_ra_deg - (xy_point[0] + 3.5) * ra_deg_per_unit
    dec_deg = base_dec_deg + xy_point[1] * dec_deg_per_unit
    return ra_deg, dec_deg


def angular_distance_arcsec(scene_pt_a, scene_pt_b):
    """Approximate small-angle distance on the sky between two scene-space
    points (using the fake catalog mapping). Returns arcseconds.
    """
    ra_a, dec_a = fake_radec_floats_for_anchor(scene_pt_a)
    ra_b, dec_b = fake_radec_floats_for_anchor(scene_pt_b)
    mean_dec_rad = np.radians((dec_a + dec_b) / 2)
    delta_ra_sec = (ra_a - ra_b) * 3600 * np.cos(mean_dec_rad)
    delta_dec_sec = (dec_a - dec_b) * 3600
    return float(np.hypot(delta_ra_sec, delta_dec_sec))


def fmt_radec_int(xy_point):
    """Format RA/Dec as whole-integer degrees, e.g. ('181°', '+46°').
    Used for the four frame-corner labels which fall on exact integers."""
    ra, dec = fake_radec_floats_for_anchor(xy_point)
    return f"{int(round(ra))}°", f"{int(round(dec)):+d}°"


def fake_radec_for_anchor(xy_point):
    """Format RA/Dec as 2-decimal-place degrees. Used for star-anchor
    annotations and table rows where sub-degree precision matters."""
    ra, dec = fake_radec_floats_for_anchor(xy_point)
    return f"{ra:.2f}°", f"{dec:+.2f}°"


def pick_n_valid_quads_spread(xy, n, exclude_anchor_pairs=None):
    """Greedy spatial spread: collect ALL valid quads, then take n whose
    anchor midpoints are far apart from each other across the field."""
    exclude_anchor_pairs = set(exclude_anchor_pairs or [])
    nstars = len(xy)
    valid = []
    for i in range(nstars):
        for j in range(i + 1, nstars):
            if (i, j) in exclude_anchor_pairs:
                continue
            mid = (xy[i] + xy[j]) / 2
            r = np.linalg.norm(xy[j] - xy[i]) / 2
            inside = [
                k
                for k in range(nstars)
                if k != i and k != j and np.linalg.norm(xy[k] - mid) < r * 0.95
            ]
            if len(inside) >= 2:
                valid.append([i, j, inside[0], inside[1]])
    if not valid:
        return []

    picked = [valid.pop(0)]
    while len(picked) < n and valid:
        best_idx, best_dist = 0, -1.0
        for vi, vq in enumerate(valid):
            anchor = (xy[vq[0]] + xy[vq[1]]) / 2
            min_d = min(
                np.linalg.norm(anchor - (xy[p[0]] + xy[p[1]]) / 2)
                for p in picked
            )
            if min_d > best_dist:
                best_dist = min_d
                best_idx = vi
        picked.append(valid.pop(best_idx))
    return picked


def pick_valid_quad(xy):
    """Pick a 4-star quad obeying the astrometry.net validity constraint
    (C, D inside the AB-diameter circle), preferring quads whose code lands
    in the interior of [0, 1]² so coordinate annotations don't overlap with
    the axis tick labels. Iterates short AB pairs first to keep the quad
    visually compact in the image.
    """

    def interior_score(code):
        """Higher = both (a) farther from any 0/1 axis line and (b) C and D
        well-separated from each other in code space."""
        cx, cy, dx, dy = code
        axis_dist = min(min(abs(x), abs(x - 1)) for x in code)
        cd_dist = float(np.hypot(cx - dx, cy - dy))
        return min(axis_dist, 0.6 * cd_dist)

    n = len(xy)
    pairs_by_len = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = float(np.sum((xy[i] - xy[j]) ** 2))
            pairs_by_len.append((d2, i, j))
    pairs_by_len.sort()

    best = None
    best_score = -1.0
    for _, a, b in pairs_by_len:
        mid = (xy[a] + xy[b]) / 2
        r = np.linalg.norm(xy[b] - xy[a]) / 2
        inside = [
            k
            for k in range(n)
            if k != a and k != b and np.linalg.norm(xy[k] - mid) < r * 0.95
        ]
        if len(inside) < 2:
            continue
        # Try every (C, D) ordering from the inside set
        for ci in range(len(inside)):
            for di in range(len(inside)):
                if ci == di:
                    continue
                c, d = inside[ci], inside[di]
                code, _ = quad_code_2d(xy[[a, b, c, d]])
                s = interior_score(code)
                if s > best_score:
                    best_score = s
                    best = [a, b, c, d]
        # Good enough? stop early so we still favor a short AB.
        if best is not None and best_score > 0.18:
            return best
    if best is None:
        raise RuntimeError("no valid quad found in the supplied star list")
    return best


# ---------- scene ------------------------------------------------------------


class ZodiacalSolver(Scene):
    def construct(self):
        self._title()
        # ---- Phase 1: Index Building ----
        self._phase_card(
            "Phase 1: Index Building",
            subtitle="Catalog stars → 4-D codes → indexed table",
            color=BLUE,
        )
        self._make_panels(image_label="Catalog Patch", show_corners=True)
        real_xy, real_dots = self._stage1_extract()
        self._stage2_quad(real_xy, real_dots)
        self._stage2b_more_entries(real_xy, real_dots)
        # Stage 2c also tears down the catalog patch + info chrome (the
        # docked index table on the right is preserved through Phase 2).
        self._stage2c_pyramid(real_dots)

        # ---- Transition to Phase 2: Solving ----
        self._phase_card(
            "Phase 2: Solving an Image",
            subtitle="Unknown image → quad → index lookup → WCS",
            color=GREEN,
        )
        # Re-spawn panels and recreate the star dots at their original
        # positions (the previous copies were faded out during stage 2c).
        self._make_panels(image_label="Captured Image", show_corners=False)
        if hasattr(self, "_index_table"):
            self.bring_to_front(self._index_table)

        # Recreate the full captured-image starfield: noise + bright sources
        # together. The bright stars are present from the start; the next
        # beat SELECTS them out of the existing field rather than fading
        # them in afterward.
        fresh_noise_pts = VGroup(
            *[
                Dot(
                    point=[xy[0], xy[1], 0],
                    radius=0.03, color=GREY_C, fill_opacity=0.7,
                )
                for xy in self._noise_xy
            ]
        )
        fresh_real_dots = VGroup(
            *[
                Dot(
                    point=[xy[0], xy[1], 0],
                    radius=0.045 + 0.02 * f,
                    color=WHITE,
                )
                for xy, f in zip(self._real_xy, self._real_flux)
            ]
        )
        self.real_dots = fresh_real_dots
        self.play(
            FadeIn(fresh_noise_pts), FadeIn(fresh_real_dots), run_time=0.7,
        )

        # Select the bright sources from the extant entries — same beat as
        # stage 1 of the catalog patch (caption first, then circles draw
        # around the bright stars).
        select_circles = VGroup(
            *[
                Circle(radius=0.16, color=YELLOW, stroke_width=2).move_to(d)
                for d in fresh_real_dots
            ]
        )
        select_cap = caption_text(
            "Extract bright sources from the captured image"
        )
        self.play(FadeIn(select_cap), run_time=0.4)
        self.play(
            LaggedStart(
                *[Create(c) for c in select_circles], lag_ratio=0.04
            ),
            run_time=0.9,
        )
        self.wait(0.6)
        self.play(
            FadeOut(select_circles),
            FadeOut(select_cap),
            FadeOut(fresh_noise_pts),
            run_time=0.45,
        )

        self._stage3_match(real_xy)
        self._stage4_fit(real_xy, fresh_real_dots)
        self._stage5_verify(real_xy)
        self._outro()
        self._addendum_healpix()

    def _teardown_panels(self, real_dots):
        """Clear the left/right panels + dots BUT keep the shrunken index
        table around for Phase 2 queries."""
        fades = [
            FadeOut(self.image_frame),
            FadeOut(self.info_frame),
            FadeOut(self.image_label),
            FadeOut(real_dots),
        ]
        if len(self.corner_labels) > 0:
            fades.append(FadeOut(self.corner_labels))
        self.play(*fades, run_time=0.4)

    # 0. title -----------------------------------------------------------------

    def _title(self):
        title = Text(
            "Zodiacal: Plate Solving",
            font_size=44,
            weight=BOLD,
        )
        sub = Text(
            "Geometric hashing à la astrometry.net (Lang et al. 2010)",
            font_size=22,
            color=GREY_B,
        ).next_to(title, DOWN)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(sub))

    def _phase_card(self, label, subtitle=None, color=YELLOW):
        """Brief full-screen card announcing a top-level phase change."""
        title = Text(label, font_size=46, weight=BOLD, color=color)
        items = [title]
        if subtitle:
            sub = Text(subtitle, font_size=22, color=GREY_B).next_to(title, DOWN)
            items.append(sub)
        group = VGroup(*items).move_to(ORIGIN)
        self.play(FadeIn(group, shift=UP * 0.3), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(group), run_time=0.5)

    # panels -------------------------------------------------------------------

    def _make_panels(self, image_label="Catalog Patch", show_corners=True):
        self.image_frame = Rectangle(
            width=FRAME_W, height=FRAME_H, color=GREY_C, stroke_width=2
        ).move_to(FRAME_CENTER)
        self.info_frame = Rectangle(
            width=INFO_W, height=INFO_H, color=GREY_D, stroke_width=1, stroke_opacity=0.5
        ).move_to(INFO_CENTER)
        self.image_label = Text(image_label, font_size=18, color=GREY_B).next_to(
            self.image_frame, UP, buff=0.1
        )

        # Optional RA/Dec labels at the four corners — communicates that the
        # frame is a patch of sky drawn from a celestial catalog.
        self.corner_labels = VGroup()
        if show_corners:
            corners = [
                (np.array([FRAME_LEFT, FRAME_TOP]), UL),
                (np.array([FRAME_RIGHT, FRAME_TOP]), UR),
                (np.array([FRAME_LEFT, FRAME_BOTTOM]), DL),
                (np.array([FRAME_RIGHT, FRAME_BOTTOM]), DR),
            ]
            for corner_xy, direction in corners:
                ra_str, dec_str = fmt_radec_int(corner_xy)
                tag = (
                    VGroup(
                        Text(ra_str, font_size=14, color=RED),
                        Text(dec_str, font_size=14, color=RED),
                    )
                    .arrange(DOWN, aligned_edge=LEFT, buff=0.04)
                    .next_to([corner_xy[0], corner_xy[1], 0], direction, buff=0.08)
                )
                self.corner_labels.add(tag)

        play_args = [
            Create(self.image_frame),
            Create(self.info_frame),
            FadeIn(self.image_label),
        ]
        if len(self.corner_labels) > 0:
            play_args.append(FadeIn(self.corner_labels))
        self.play(*play_args, run_time=0.8)

    # 1. extract ---------------------------------------------------------------

    def _stage1_extract(self):
        header = header_text("1. Source selection")
        self.play(Write(header))

        (real_xy, real_flux), (noise_xy, noise_flux) = make_starfield(18, 80)

        noise_pts = VGroup(
            *[
                Dot(
                    point=[xy[0], xy[1], 0],
                    radius=0.03,
                    color=GREY_C,
                    fill_opacity=0.7,
                )
                for xy in noise_xy
            ]
        )

        # Bright sources: only ~2x larger than noise (was ~5x). Keeps a hint
        # of brightness variation without the dramatic jump.
        real_dots = VGroup(
            *[
                Dot(point=[xy[0], xy[1], 0], radius=0.045 + 0.02 * flux, color=WHITE)
                for xy, flux in zip(real_xy, real_flux)
            ]
        )

        self.real_dots = real_dots  # kept around through stages 2-5
        self._real_xy = real_xy  # stash so Phase 2 can recreate the dots
        self._real_flux = real_flux
        self._noise_xy = noise_xy
        self._noise_flux = noise_flux
        # Fade in the FULL starfield (noise + bright sources together) — the
        # bright stars are part of the scene from the start; the next beat
        # SELECTS them out of that field rather than introducing them.
        self.play(FadeIn(noise_pts), FadeIn(real_dots), run_time=0.7)
        self.wait(0.3)

        # detection circles around real stars
        circles = VGroup(
            *[
                Circle(radius=0.16, color=YELLOW, stroke_width=2).move_to(d)
                for d in real_dots
            ]
        )
        cap = caption_text("Pick the brightest catalog stars in this patch")
        # Right-panel summary
        info = VGroup(
            Text(f"sources selected: {len(real_dots)}", font_size=22, color=YELLOW),
            Text("faint stars below threshold", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(INFO_CENTER)

        # Caption appears FIRST so the viewer reads what we're about to do,
        # then the circles get drawn around the bright sources.
        self.play(FadeIn(cap), FadeIn(info), run_time=0.5)
        self.play(
            LaggedStart(*[Create(c) for c in circles], lag_ratio=0.04), run_time=1.0
        )
        self.wait(1.0)

        self.play(
            FadeOut(circles), FadeOut(cap), FadeOut(noise_pts), FadeOut(info), FadeOut(header)
        )
        return real_xy, real_dots

    # 2. quad formation --------------------------------------------------------

    def _stage2_quad(self, real_xy, real_dots):
        header = header_text("2. Quad formation")
        self.play(Write(header))

        # Pick a valid quad: C, D inside the circle with diameter AB.
        idx = pick_valid_quad(real_xy)
        self._first_entry_idx = idx
        quad_pts = real_xy[idx]
        labels = ["A", "B", "C", "D"]
        colors = [RED, GREEN, BLUE, ORANGE]

        # Draw the AB-diameter circle as a hint of the validity constraint
        a_xy, b_xy = quad_pts[0], quad_pts[1]
        ab_mid = (a_xy + b_xy) / 2
        ab_radius = float(np.linalg.norm(b_xy - a_xy) / 2)
        ab_circle = DashedVMobject(
            Circle(radius=ab_radius, color=GREY_B, stroke_width=1.5).move_to(
                [ab_mid[0], ab_mid[1], 0]
            ),
            num_dashes=40,
        )
        self.play(Create(ab_circle), run_time=0.6)

        # Step 1 — select the four stars (scale up, neutral colour) so the
        # viewer first sees "these four are chosen".
        sample_cap = caption_text("Sample 4 stars")
        self.play(
            FadeIn(sample_cap),
            *[real_dots[i].animate.scale(1.6) for i in idx],
            run_time=0.6,
        )
        self.wait(0.4)

        # Step 2 — assign role colours (A=RED, B=GREEN, C=BLUE, D=ORANGE)
        # and draw a matching coloured circle around each.
        star_circles = VGroup(
            *[
                Circle(radius=0.18, color=c, stroke_width=2.5).move_to(
                    real_dots[i]
                )
                for i, c in zip(idx, colors)
            ]
        )

        # A's RA/Dec is the entry's anchor — drop dashed RED projections
        # from A to the bottom and left edges of the patch, mirroring the
        # C/D coordinate readout in the code-space plot. Tag each line with
        # A's actual value.
        a_pos = real_dots[idx[0]].get_center()
        ra_a, dec_a = fake_radec_for_anchor(quad_pts[0])
        a_ra_line = DashedLine(
            [a_pos[0], a_pos[1], 0],
            [a_pos[0], FRAME_BOTTOM, 0],
            color=RED,
            stroke_width=1.5,
        )
        a_dec_line = DashedLine(
            [a_pos[0], a_pos[1], 0],
            [FRAME_LEFT, a_pos[1], 0],
            color=RED,
            stroke_width=1.5,
        )
        a_ra_tag = Text(ra_a, font_size=12, color=RED).next_to(
            [a_pos[0], FRAME_BOTTOM, 0], DOWN, buff=0.06
        )
        a_dec_tag = Text(dec_a, font_size=12, color=RED).next_to(
            [FRAME_LEFT, a_pos[1], 0], LEFT, buff=0.06
        )

        self.play(
            *[real_dots[i].animate.set_color(c) for i, c in zip(idx, colors)],
            *[Create(c) for c in star_circles],
            Create(a_ra_line),
            Create(a_dec_line),
            FadeIn(a_ra_tag),
            FadeIn(a_dec_tag),
            run_time=0.8,
        )

        # Right-panel legend with colored dots + letters (strictly outside frame)
        legend_rows = []
        for lab, col in zip(labels, colors):
            row = VGroup(
                Dot(radius=0.10, color=col),
                Text(lab, font_size=24, weight=BOLD, color=col),
            ).arrange(RIGHT, buff=0.25)
            legend_rows.append(row)
        legend = (
            VGroup(*legend_rows)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            .move_to([INFO_LEFT + 0.6, 1.7, 0], aligned_edge=UL)
        )
        self.play(FadeIn(legend))

        # AB backbone + AC, AD lines
        ab_line = Line(
            start=[*quad_pts[0], 0],
            end=[*quad_pts[1], 0],
            color=YELLOW,
            stroke_width=3,
        )
        ac_line = Line(
            start=[*quad_pts[0], 0],
            end=[*quad_pts[2], 0],
            color=BLUE_C,
            stroke_width=2,
        )
        ad_line = Line(
            start=[*quad_pts[0], 0],
            end=[*quad_pts[3], 0],
            color=BLUE_C,
            stroke_width=2,
        )
        cap = caption_text("A,B form the backbone — C,D positioned relative to it")
        self.play(
            Create(ab_line),
            Create(ac_line),
            Create(ad_line),
            FadeOut(sample_cap),
            FadeIn(cap),
        )
        self.wait(0.6)

        # ----- transformation into code space ---------------------------------
        # Phase 1: lift quad out of image into the right panel, preserving the
        # original geometry (translation + uniform scale to fit, no rotation).
        # Phase 2: normalize in the plotting space — rotate/scale so A→(0,0),
        # B→(1,1).

        code, normalized = quad_code_2d(quad_pts)

        # Code-space coordinate system position — sized so the unit square
        # dominates the right info panel.
        # Code-space graph: scooted up (cs_center.y was -1.0) AND grown
        # (cs_unit was 2.6) now that the right info panel no longer carries
        # the "solver state" label up top — frees vertical room for a
        # bigger unit square.
        cs_unit = 2.85
        cs_center = np.array([INFO_CENTER[0] - 0.5 * cs_unit, -0.65, 0])

        # --- Spawn ghosts at the actual image-space positions ----------------
        ghosts = VGroup()
        for orig_idx, col in zip(idx, colors):
            ghost = Dot(
                point=real_dots[orig_idx].get_center(),
                radius=0.10,
                color=col,
                stroke_width=2,
                stroke_color=WHITE,
            )
            ghosts.add(ghost)
        self.add(ghosts)

        # The journey from image space → canonical code space is split into
        # three discrete steps so each rigid-body action is its own beat:
        #   Phase 1 — lift  : translate so A lands at code (0, 0). Quad keeps
        #                     original orientation AND original |AB|.
        #   Phase 2 — rotate: spin around A until AB lies along the (1, 1)
        #                     diagonal. |AB| unchanged.
        #   Phase 3 — scale : enlarge around A so |AB| = √2·cs_unit, planting
        #                     B exactly on (1, 1).
        ab_len_img = float(np.linalg.norm(quad_pts[1] - quad_pts[0]))
        target_ab_len = np.sqrt(2) * cs_unit
        scale_factor = target_ab_len / max(ab_len_img, 1e-6)
        A_anchor = cs_center  # = code-space (0, 0)

        # Phase 1: translate only. A → cs_center, others retain image offsets.
        phase1_targets = []
        for p in quad_pts:
            rel = p - quad_pts[0]
            tgt = A_anchor + np.array([rel[0], rel[1], 0])
            phase1_targets.append(tgt)

        # Rotation that maps the (still-image-orientation) AB onto the
        # (1, 1) diagonal. Computed against post-Phase-1 positions.
        ab_vec_lifted = phase1_targets[1] - phase1_targets[0]
        ab_angle = float(np.arctan2(ab_vec_lifted[1], ab_vec_lifted[0]))
        rotation_angle = np.pi / 4 - ab_angle

        def rotate_about(p, angle, center):
            rel = np.asarray(p) - np.asarray(center)
            ca, sa = np.cos(angle), np.sin(angle)
            return np.asarray(center) + np.array(
                [rel[0] * ca - rel[1] * sa, rel[0] * sa + rel[1] * ca, 0.0]
            )

        # Phase 2 targets: rotate Phase 1 positions around A_anchor.
        phase2_targets = [
            rotate_about(t, rotation_angle, A_anchor) for t in phase1_targets
        ]

        # Phase 3 targets: scale Phase 2 positions about A_anchor.
        phase3_targets = [
            A_anchor + (t - A_anchor) * scale_factor for t in phase2_targets
        ]

        # --- Build the code-space coordinate system FIRST so the unit-square
        # circumscribed circle is visible while the quad lifts in.
        # Axes only extend slightly past the unit square so they fit in panel.
        axis_pad = 0.4
        axes = VGroup(
            Line(
                cs_center + np.array([-axis_pad, 0, 0]),
                cs_center + np.array([cs_unit + axis_pad, 0, 0]),
                color=GREY_B,
                stroke_width=1.5,
            ),
            Line(
                cs_center + np.array([0, -axis_pad, 0]),
                cs_center + np.array([0, cs_unit + axis_pad, 0]),
                color=GREY_B,
                stroke_width=1.5,
            ),
        )
        unit_box = Rectangle(
            width=cs_unit, height=cs_unit, color=GREY_C, stroke_width=1
        ).move_to(cs_center + np.array([0.5 * cs_unit, 0.5 * cs_unit, 0]))
        cs_label = Text("code space", font_size=18, color=GREY_B).move_to(
            cs_center + np.array([0.5 * cs_unit, cs_unit + axis_pad + 0.18, 0])
        )
        tick_0 = Text("0", font_size=14, color=GREY_B).next_to(
            cs_center, DL, buff=0.05
        )
        tick_1x = Text("1", font_size=14, color=GREY_B).next_to(
            cs_center + np.array([cs_unit, 0, 0]), DOWN, buff=0.05
        )
        tick_1y = Text("1", font_size=14, color=GREY_B).next_to(
            cs_center + np.array([0, cs_unit, 0]), LEFT, buff=0.05
        )

        cap2 = caption_text("Code space: A → (0,0), B → (1,1)")
        self.play(
            Transform(cap, cap2),
            Create(axes),
            FadeIn(unit_box),
            FadeIn(cs_label),
            FadeIn(tick_0),
            FadeIn(tick_1x),
            FadeIn(tick_1y),
            run_time=0.8,
        )

        # --- Translate / Rotate / Scale — one umbrella caption -------------
        # Single caption sits below the action through all three rigid-body
        # steps so the viewer reads "we're bringing the quad into a
        # non-dimensional reference frame", not three separate operation names.
        phase1_ab_mid = (phase1_targets[0] + phase1_targets[1]) / 2
        cap_normalize = caption_text("Bring into non-dimensional space")

        # Step 1: translate only — A → (0,0).
        self.play(
            Transform(cap, cap_normalize),
            FadeOut(legend),
            *[ghosts[i].animate.move_to(phase1_targets[i]) for i in range(4)],
            ab_circle.animate.move_to(phase1_ab_mid),
            run_time=1.2,
        )
        self.wait(0.3)

        # Step 2: rotate about (0,0) so AB lies along the (1,1) diagonal.
        # Explicit Rotate animations interpolate the rotation parameter, so
        # each ghost actually arcs around (0,0) rather than shortcutting.
        self.play(
            *[
                Rotate(ghosts[i], angle=rotation_angle, about_point=A_anchor)
                for i in range(4)
            ],
            Rotate(ab_circle, angle=rotation_angle, about_point=A_anchor),
            run_time=1.4,
        )
        self.wait(0.3)

        # Step 3: scale about (0,0) so B lands on (1,1).
        self.play(
            *[ghosts[i].animate.move_to(phase3_targets[i]) for i in range(4)],
            ab_circle.animate.scale(scale_factor, about_point=A_anchor),
            run_time=1.4,
        )

        # ---- Extract coordinates of C and D, plus the |AB| plate scale ------
        cap4 = caption_text("Read off C, D coords — and store |AB| as the plate scale")
        self.play(Transform(cap, cap4))

        # Helper: position of point (px, py) in code space → scene coordinates
        def cs_pos(px, py):
            return cs_center + np.array([px * cs_unit, py * cs_unit, 0])

        cx, cy = float(code[0]), float(code[1])
        dx, dy = float(code[2]), float(code[3])

        # Dashed projection lines from C to each axis
        c_vline = DashedLine(
            cs_pos(cx, cy), cs_pos(cx, 0), color=BLUE, stroke_width=1.5
        )
        c_hline = DashedLine(
            cs_pos(cx, cy), cs_pos(0, cy), color=BLUE, stroke_width=1.5
        )
        # Same for D
        d_vline = DashedLine(
            cs_pos(dx, dy), cs_pos(dx, 0), color=ORANGE, stroke_width=1.5
        )
        d_hline = DashedLine(
            cs_pos(dx, dy), cs_pos(0, dy), color=ORANGE, stroke_width=1.5
        )

        # Tuple annotations placed right next to each indexed point so they
        # sit out of the axis area. Direction picked per-point to dodge the
        # other graphics in the panel.
        c_label = Text(
            f"({cx:+.2f}, {cy:+.2f})", font_size=18, color=BLUE
        ).next_to(cs_pos(cx, cy), DR, buff=0.12)
        d_label = Text(
            f"({dx:+.2f}, {dy:+.2f})", font_size=18, color=ORANGE
        ).next_to(cs_pos(dx, dy), UL, buff=0.12)

        # Plate-scale arrow along the (0,0)→(1,1) diagonal — that segment IS
        # the original AB in image-space, so its pixel length is the scale we
        # store alongside the 4-D code.
        ab_arrow = Arrow(
            cs_pos(0, 0),
            cs_pos(1, 1),
            color=YELLOW,
            buff=0.0,
            stroke_width=4,
            tip_length=0.22,
        )
        # Compute |AB| in arcseconds from the fake catalog mapping —
        # realistic to the rendering (the patch is 2°×2° on the sky).
        ab_arcsec = angular_distance_arcsec(quad_pts[0], quad_pts[1])
        ab_arrow_label = Text(
            f"|AB| = {ab_arcsec:.0f}″ (plate scale)",
            font_size=18,
            color=YELLOW,
        ).next_to(cs_pos(0.5, 0.5), DR, buff=0.5)

        self.play(
            Create(c_vline), Create(c_hline),
            FadeIn(c_label),
            run_time=0.8,
        )
        self.play(
            Create(d_vline), Create(d_hline),
            FadeIn(d_label),
            run_time=0.8,
        )
        self.play(GrowArrow(ab_arrow), FadeIn(ab_arrow_label), run_time=0.8)

        # Anchor RA/Dec for star A — completes the (RA, Dec, plate scale, code)
        # tuple that gets stored in the index file.
        ra_str, dec_str = fake_radec_for_anchor(quad_pts[0])
        self._first_entry_radec = (ra_str, dec_str)
        self._first_entry = (ra_str, dec_str, ab_arcsec, (cx, cy, dx, dy))

        # ---- Build the index table BELOW the code-space graph, in the
        # lower portion of the right info panel. Header row + separator.
        # The first data row's cells are TARGETS for TransformFromCopy
        # animations that pull each value from its on-scene tag (left →
        # right, one column at a time).
        header_cells = VGroup(
            table_cell("RA (°)", TABLE_COL_X[0], TABLE_TOP_Y),
            table_cell("Dec (°)", TABLE_COL_X[1], TABLE_TOP_Y),
            table_cell("|AB| ″", TABLE_COL_X[2], TABLE_TOP_Y),
            table_cell("code (4D)", TABLE_COL_X[3], TABLE_TOP_Y),
        )
        sep_line = Line(
            [INFO_LEFT + 0.3, TABLE_TOP_Y - 0.16, 0],
            [INFO_RIGHT - 0.3, TABLE_TOP_Y - 0.16, 0],
            color=GREY_C,
            stroke_width=1,
        )
        self.play(
            FadeIn(header_cells),
            Create(sep_line),
            run_time=0.6,
        )

        # Row 1 cells — each will be the destination of a TransformFromCopy
        # animation pulling from the matching on-scene source.
        row1_y = TABLE_TOP_Y - TABLE_ROW_H
        ra_target = table_cell(ra_str, TABLE_COL_X[0], row1_y, color=RED)
        dec_target = table_cell(dec_str, TABLE_COL_X[1], row1_y, color=RED)
        ab_target = table_cell(
            f"{ab_arcsec:.0f}", TABLE_COL_X[2], row1_y, color=YELLOW
        )
        code_target = colored_code_cell(
            (cx, cy, dx, dy), TABLE_COL_X[3], row1_y
        )

        # Sequenced left → right so each entry has its own beat. Code cell
        # is the union of c_label and d_label morphing into the (cx,cy,dx,dy)
        # tuple in one combined animation.
        self.play(
            LaggedStart(
                TransformFromCopy(a_ra_tag, ra_target),
                TransformFromCopy(a_dec_tag, dec_target),
                TransformFromCopy(ab_arrow_label, ab_target),
                TransformFromCopy(VGroup(c_label, d_label), code_target),
                lag_ratio=0.5,
            ),
            run_time=2.4,
        )
        row1 = VGroup(ra_target, dec_target, ab_target, code_target)
        self.wait(1.5)

        # Cleanup stage 2 — keep the code-space chrome AND the freshly-built
        # table alive so stage 2b can extend the table with more rows.
        self.play(
            FadeOut(ab_line),
            FadeOut(ac_line),
            FadeOut(ad_line),
            FadeOut(ab_circle),
            FadeOut(c_vline), FadeOut(c_hline),
            FadeOut(d_vline), FadeOut(d_hline),
            FadeOut(c_label), FadeOut(d_label),
            FadeOut(ab_arrow),
            FadeOut(ab_arrow_label),
            FadeOut(cap),
            FadeOut(header),
            FadeOut(star_circles),
            FadeOut(a_ra_line),
            FadeOut(a_dec_line),
            FadeOut(a_ra_tag),
            FadeOut(a_dec_tag),
            FadeOut(ghosts),
            *[real_dots[i].animate.set_color(WHITE).scale(1 / 1.6) for i in idx],
        )
        self._last_code = code

        # Save references stage 2b needs.
        self._cs_chrome = VGroup(
            axes, unit_box, cs_label, tick_0, tick_1x, tick_1y
        )
        self._cs_center = cs_center
        self._cs_unit = cs_unit
        self._table_header_cells = header_cells
        self._table_sep_line = sep_line
        self._table_rows = [row1]

    # ---- Accumulator helpers -----------------------------------------------

    def _animate_quad_in_codespace(self, real_xy, real_dots, idx, speed=2.5):
        """Run a quad through the same translate→rotate→scale animation as
        the first quad, but at higher speed. After the animation completes,
        all the catalog-side annotations and the colored ghosts fade away —
        only the new entry (which the caller adds as a table row) persists.

        Returns (ra_str, dec_str, ab_arcsec, code_4tuple).
        """
        quad_pts = real_xy[idx]
        colors = [RED, GREEN, BLUE, ORANGE]
        cs_center = self._cs_center
        cs_unit = self._cs_unit

        # Catalog-patch annotations: color stars + circles + AB-circle + lines.
        star_circles = VGroup(
            *[
                Circle(radius=0.18, color=c, stroke_width=2.5).move_to(real_dots[i])
                for i, c in zip(idx, colors)
            ]
        )
        a_xy, b_xy = quad_pts[0], quad_pts[1]
        ab_mid = (a_xy + b_xy) / 2
        ab_radius = float(np.linalg.norm(b_xy - a_xy) / 2)
        ab_circle = DashedVMobject(
            Circle(radius=ab_radius, color=GREY_B, stroke_width=1.5).move_to(
                [ab_mid[0], ab_mid[1], 0]
            ),
            num_dashes=40,
        )
        ab_line = Line(
            start=[*quad_pts[0], 0], end=[*quad_pts[1], 0],
            color=YELLOW, stroke_width=3,
        )
        ac_line = Line(
            start=[*quad_pts[0], 0], end=[*quad_pts[2], 0],
            color=BLUE_C, stroke_width=2,
        )
        ad_line = Line(
            start=[*quad_pts[0], 0], end=[*quad_pts[3], 0],
            color=BLUE_C, stroke_width=2,
        )
        # All catalog-side annotations appear together (fast).
        self.play(
            *[real_dots[i].animate.set_color(c).scale(1.4) for i, c in zip(idx, colors)],
            *[Create(c) for c in star_circles],
            Create(ab_circle),
            Create(ab_line), Create(ac_line), Create(ad_line),
            run_time=0.4 / speed * 1.5,
        )

        # Spawn ghosts at each picked star's position.
        ghosts = VGroup(
            *[
                Dot(
                    point=real_dots[i].get_center(),
                    radius=0.10, color=c,
                    stroke_width=2, stroke_color=WHITE,
                )
                for i, c in zip(idx, colors)
            ]
        )
        self.add(ghosts)

        # Compute targets for translate / rotate / scale.
        code, _ = quad_code_2d(quad_pts)
        ab_len_img = float(np.linalg.norm(quad_pts[1] - quad_pts[0]))
        scale_factor = (np.sqrt(2) * cs_unit) / max(ab_len_img, 1e-6)

        phase1_targets = [
            cs_center + np.array([(p - quad_pts[0])[0], (p - quad_pts[0])[1], 0])
            for p in quad_pts
        ]
        ab_vec_lifted = phase1_targets[1] - phase1_targets[0]
        rotation_angle = np.pi / 4 - float(np.arctan2(ab_vec_lifted[1], ab_vec_lifted[0]))

        def rot_about(p, angle, center):
            rel = np.asarray(p) - np.asarray(center)
            ca, sa = np.cos(angle), np.sin(angle)
            return np.asarray(center) + np.array(
                [rel[0] * ca - rel[1] * sa, rel[0] * sa + rel[1] * ca, 0.0]
            )

        phase3_targets = [
            cs_center + (rot_about(t, rotation_angle, cs_center) - cs_center) * scale_factor
            for t in phase1_targets
        ]
        phase1_ab_mid = (phase1_targets[0] + phase1_targets[1]) / 2

        # Translate, rotate, scale — fast.
        self.play(
            *[ghosts[i].animate.move_to(phase1_targets[i]) for i in range(4)],
            ab_circle.animate.move_to(phase1_ab_mid),
            run_time=1.2 / speed,
        )
        self.play(
            *[
                Rotate(ghosts[i], angle=rotation_angle, about_point=cs_center)
                for i in range(4)
            ],
            Rotate(ab_circle, angle=rotation_angle, about_point=cs_center),
            run_time=1.4 / speed,
        )
        self.play(
            *[ghosts[i].animate.move_to(phase3_targets[i]) for i in range(4)],
            ab_circle.animate.scale(scale_factor, about_point=cs_center),
            run_time=1.4 / speed,
        )

        # Tear down all the catalog-side annotations and ghost dots — the
        # only thing that survives is the table row the caller will add.
        self.play(
            FadeOut(ghosts),
            FadeOut(ab_circle),
            FadeOut(ab_line),
            FadeOut(ac_line),
            FadeOut(ad_line),
            FadeOut(star_circles),
            *[real_dots[i].animate.set_color(WHITE).scale(1 / 1.4) for i in idx],
            run_time=0.5 / speed * 1.5,
        )

        ab_arcsec = angular_distance_arcsec(quad_pts[0], quad_pts[1])
        ra_str, dec_str = fake_radec_for_anchor(quad_pts[0])
        return ra_str, dec_str, ab_arcsec, (
            float(code[0]), float(code[1]), float(code[2]), float(code[3])
        )

    # 2b. populate index table -----------------------------------------------

    def _stage2b_more_entries(self, real_xy, real_dots):
        """Pick a few more random valid quads and append them to the table
        (header + first row already built in stage 2)."""
        header = header_text("…repeat for many quads → grows the index")
        self.play(Write(header))

        all_rows = list(self._table_rows)

        # Pick more valid quads (excluding the original anchor pair)
        first_anchor = (
            min(self._first_entry_idx[0], self._first_entry_idx[1]),
            max(self._first_entry_idx[0], self._first_entry_idx[1]),
        )
        more_quads = pick_n_valid_quads_spread(
            real_xy, n=4, exclude_anchor_pairs=[first_anchor]
        )

        # Each new row appears just below the header (at row1_y); the
        # previously-added rows shift DOWN by one row_h so the freshly
        # filled-in entry is always at the top of the data area. Older
        # rows fall off the bottom of the panel as the pile grows.
        row1_y = TABLE_TOP_Y - TABLE_ROW_H
        for qidx in more_quads:
            # Run the same translate→rotate→scale animation as the first
            # quad, but at higher speed. Ghosts fade out at the end.
            ra_i, dec_i, ab_sec_i, code_i = self._animate_quad_in_codespace(
                real_xy, real_dots, qidx, speed=2.5,
            )

            new_row = make_table_row(
                ra_i, dec_i, ab_sec_i, code_i, row1_y,
            )
            new_row.shift(RIGHT * 0.6).set_opacity(0.0)
            self.add(new_row)
            self.play(
                new_row.animate.shift(LEFT * 0.6).set_opacity(1.0),
                *[r.animate.shift(DOWN * TABLE_ROW_H) for r in all_rows],
                run_time=0.45,
            )
            all_rows.append(new_row)
        self.wait(0.6)

        # ---- Tear down the code-space chrome before the table shrink.
        self.play(FadeOut(self._cs_chrome), run_time=0.4)

        # ---- Shrink the table to a background asset for Phase 2 -------------
        # Group everything together and animate scale + position so the table
        # docks at the bottom of the right info panel. It survives the phase
        # transition and gets queried into during the kd-tree match step.
        table_group = VGroup(
            self._table_header_cells,
            self._table_sep_line,
            *all_rows,
        )
        target_pos = np.array([INFO_CENTER[0], -1.65, 0])
        self.play(
            table_group.animate.scale(0.55).move_to(target_pos),
            FadeOut(header),
            run_time=1.0,
        )
        self._index_table = table_group
        self._index_table_rows = all_rows  # references survive the scale

    # 2c. multi-scale pyramid ------------------------------------------------

    def _stage2c_pyramid(self, real_dots):
        """An isometric stepped trapezoidal pyramid: each level is a flat
        n×n grid of parallelogram tiles (iso projection of squares), levels
        stack vertically with a gap, each upper level is physically smaller
        in the (x, y) plane so the silhouette is a true trapezoidal pyramid.
        Most tiles (4×4 = 16) at the BOTTOM. Faint lines connect
        corresponding corners between adjacent levels to draw the pyramid
        edges in 3D."""
        new_header = header_text("Indexes are tiled at multiple scales")

        fades = [
            FadeOut(self.image_label),
            FadeOut(real_dots),
            FadeOut(self.info_frame),
        ]
        if len(self.corner_labels) > 0:
            fades.append(FadeOut(self.corner_labels))
        self.play(Write(new_header), *fades, run_time=0.6)

        # ---- Isometric projection helper.  Standard 30° iso: y goes "back".
        # Origin scoots the whole pyramid DOWN so it sits centered around the
        # vertical midline rather than crowding the upper half of the scene.
        COS30 = float(np.cos(np.radians(30)))
        SIN30 = float(np.sin(np.radians(30)))
        pyr_origin = np.array([-3.5, -1.5, 0.0])

        def iso(x, y, z):
            """Project a 3D world point to a 2D scene point relative to
            pyr_origin. Convention: +x screen-right-up, +y screen-left-up,
            +z straight-up."""
            return pyr_origin + np.array(
                [(x - y) * COS30, (x + y) * SIN30 + z, 0.0]
            )

        # Levels (bottom → top): wider levels with more tiles at the bottom.
        # tile_unit kept constant so visually the grid resolution increases
        # as you go down. Each level's xy footprint = n * tile_unit per side.
        tile_unit = 0.42
        z_gap = 0.78
        # (n_per_side, count_label) — listed BOTTOM first.
        levels_config = [
            (5, "25 tiles"),
            (4, "16 tiles"),
            (3, "9 tiles"),
            (2, "4 tiles"),
            (1, "1 tile  (whole sky)"),
        ]
        TARGET_N = 5  # the bottom (finest) level — catalog patch lives here
        target_r, target_c = 2, 2  # center of the 5×5 grid

        all_tiles = VGroup()
        all_labels = VGroup()
        level_outer_corners = []  # 4 outer corners (BL, BR, TR, TL) per level
        target_iso_center = None
        target_iso_corners = None

        for li, (n, count_label) in enumerate(levels_config):
            side = n * tile_unit
            half = side / 2
            z_level = li * z_gap

            # 4 outer corners in 3D (in the (x, y, z=z_level) plane)
            outer_3d = [
                (-half, -half, z_level),  # BL in xy
                (+half, -half, z_level),  # BR
                (+half, +half, z_level),  # TR
                (-half, +half, z_level),  # TL
            ]
            outer_iso = [iso(*p) for p in outer_3d]
            level_outer_corners.append(outer_iso)

            # Build n×n grid of parallelogram tiles (iso projection of squares)
            for r in range(n):
                for c in range(n):
                    x0 = -half + c * tile_unit
                    x1 = -half + (c + 1) * tile_unit
                    y0 = -half + r * tile_unit
                    y1 = -half + (r + 1) * tile_unit
                    is_target = (n == TARGET_N and r == target_r and c == target_c)
                    p00 = iso(x0, y0, z_level)
                    p10 = iso(x1, y0, z_level)
                    p11 = iso(x1, y1, z_level)
                    p01 = iso(x0, y1, z_level)
                    if is_target:
                        # Reserve this slot — the shrunken catalog patch
                        # docks in its center, reshaped to lie in the iso
                        # plane (a rhombus matching this tile's parallelogram).
                        target_iso_center = (p00 + p10 + p11 + p01) / 4
                        target_iso_corners = [p00, p10, p11, p01]
                        continue
                    tile = Polygon(
                        p00, p10, p11, p01,
                        color=GREY_C,
                        stroke_width=1.0,
                        fill_color=BLUE_E,
                        fill_opacity=0.06,
                    )
                    all_tiles.add(tile)

            # Level label to the right of the level
            label_anchor = iso(half + 0.15, -half, z_level)
            lbl = Text(count_label, font_size=14, color=GREY_B).move_to(
                label_anchor + RIGHT * 0.6, aligned_edge=LEFT
            )
            all_labels.add(lbl)

        # Faint pyramid edge lines: connect each pair of adjacent levels'
        # corresponding 4 corners (BL→BL, BR→BR, TR→TR, TL→TL).
        proj_lines = VGroup()
        for li in range(len(level_outer_corners) - 1):
            lower = level_outer_corners[li]
            upper = level_outer_corners[li + 1]
            for k in range(4):
                proj_lines.add(
                    Line(
                        lower[k], upper[k],
                        color=GREY_C,
                        stroke_width=1.0,
                        stroke_opacity=0.4,
                    )
                )

        # ---- Visual linkage: catalog frame shrinks AND is reshaped into
        # the iso-parallelogram of the highlighted tile slot — so it lies
        # flat in the bottom-level plane of the pyramid rather than floating
        # axis-aligned in front of it.
        margin = 0.85
        target_corners_shrunk = [
            target_iso_center + (c - target_iso_center) * margin
            for c in target_iso_corners
        ]
        target_iso_poly = Polygon(
            *target_corners_shrunk,
            color=YELLOW,
            stroke_width=2.0,
        )
        link_cap = caption_text("This patch is one tile at the finest scale")
        self.play(
            FadeIn(link_cap),
            Transform(self.image_frame, target_iso_poly),
            run_time=1.4,
        )
        self.wait(0.4)
        self.play(FadeOut(link_cap), run_time=0.2)

        # Group tiles by level (bottom → top in the order we built them).
        tiles_by_level = []
        idx_offset = 0
        for n, _ in levels_config:
            count = n * n - (1 if n == TARGET_N else 0)
            tiles_by_level.append(all_tiles[idx_offset : idx_offset + count])
            idx_offset += count

        # Reveal bottom level first (most-tiled, where the catalog patch sits)
        self.play(
            FadeIn(tiles_by_level[0]), FadeIn(all_labels[0]), run_time=0.6
        )
        # Reveal each next level UP, plus its 4 corner projection lines.
        for li in range(1, len(levels_config)):
            edge_quad = VGroup(*proj_lines[4 * (li - 1) : 4 * li])
            self.play(
                Create(edge_quad, lag_ratio=0),
                FadeIn(tiles_by_level[li]),
                FadeIn(all_labels[li]),
                run_time=0.7,
            )

        # Tag the highlighted (catalog-patch) tile
        tag = Text("← the index we just built", font_size=14, color=YELLOW).next_to(
            self.image_frame, RIGHT, buff=0.12
        )
        self.play(FadeIn(tag), run_time=0.4)

        cap = caption_text(
            "Same sky, four scales — 1, 4, 9, 16 tiles. Solver queries every level."
        )
        self.play(FadeIn(cap))
        self.wait(0.6)

        # ---- Per-level chunk → union: each scale generates its own block
        # of quad-catalog entries; we union them together into the docked
        # index table.
        cap2 = caption_text(
            "Each scale produces its own chunk of quads — union into one index"
        )
        self.play(Transform(cap, cap2), run_time=0.4)

        # Estimate quads-per-tile for the chunk-size labels (illustrative,
        # not accurate — the real numbers depend on star density).
        QUADS_PER_TILE = 60
        chunk_counts = [n * n * QUADS_PER_TILE for n, _ in levels_config]

        chunks = VGroup()
        for li, (n, _) in enumerate(levels_config):
            level_corners = level_outer_corners[li]
            level_right_x = max(c[0] for c in level_corners)
            level_y = sum(c[1] for c in level_corners) / 4
            chunk = Text(
                f"+{chunk_counts[li]:,} quads",
                font_size=15, color=YELLOW,
            ).move_to([level_right_x + 1.1, level_y, 0])
            chunks.add(chunk)

            self.play(
                Indicate(tiles_by_level[li], color=WHITE, scale_factor=1.05),
                FadeIn(chunk, shift=RIGHT * 0.25),
                run_time=0.45,
            )

        # Union — fly all chunks into the docked index table on the right,
        # collapsing them into a single highlighted update.
        cap3 = caption_text(
            "All chunks union together → one searchable index"
        )
        self.play(Transform(cap, cap3), run_time=0.4)

        if hasattr(self, "_index_table"):
            target = self._index_table.get_top() + UP * 0.05
            self.play(
                *[
                    chunk.animate.move_to(target).scale(0.3).set_opacity(0)
                    for chunk in chunks
                ],
                Indicate(self._index_table, color=YELLOW, scale_factor=1.05),
                run_time=1.4,
            )

            total = sum(chunk_counts)
            grew_label = Text(
                f"+{total:,} indexed quads",
                font_size=18, color=YELLOW,
            ).next_to(self._index_table, UP, buff=0.15)
            self.play(FadeIn(grew_label), run_time=0.4)
            self.wait(1.2)
            self.play(FadeOut(grew_label), run_time=0.3)
        else:
            self.play(
                *[FadeOut(c) for c in chunks],
                run_time=0.5,
            )

        # Cleanup — fade out the pyramid AND the shrunken catalog frame.
        # Phase 2 will recreate a fresh image frame for the unknown image.
        self.play(
            FadeOut(self.image_frame),
            FadeOut(all_tiles),
            FadeOut(all_labels),
            FadeOut(proj_lines),
            FadeOut(tag),
            FadeOut(cap),
            FadeOut(new_header),
            run_time=0.5,
        )

    # 3. code matching ---------------------------------------------------------

    def _stage3_match(self, real_xy):
        """For the captured (unknown) image: extract the same A,B,C,D quad
        we showed in stage 2, compute its 4-D code, and search the index
        table we just built. The matched row's RA/Dec gets pulled into the
        left panel as the seed for hypothesis fitting in stage 4."""
        header = header_text("3. Same recipe → search the index")
        self.play(Write(header))

        cap = caption_text("Pick a quad in the captured image")
        self.play(FadeIn(cap))

        # Reuse the same quad we built in stage 2 — same star field, same
        # quad, so its code will match the first row of the index table.
        idx = self._first_entry_idx
        quad_pts = real_xy[idx]
        colors = [RED, GREEN, BLUE, ORANGE]
        real_dots = self.real_dots

        # Show the AB-circle + colored stars + AB/AC/AD lines, matching the
        # presentation we used for the catalog quad in stage 2.
        a_xy, b_xy = quad_pts[0], quad_pts[1]
        ab_mid = (a_xy + b_xy) / 2
        ab_radius = float(np.linalg.norm(b_xy - a_xy) / 2)
        ab_circle = DashedVMobject(
            Circle(radius=ab_radius, color=GREY_B, stroke_width=1.5).move_to(
                [ab_mid[0], ab_mid[1], 0]
            ),
            num_dashes=40,
        )
        star_circles = VGroup(
            *[
                Circle(radius=0.18, color=c, stroke_width=2.5).move_to(real_dots[i])
                for i, c in zip(idx, colors)
            ]
        )
        ab_line = Line(
            start=[*quad_pts[0], 0], end=[*quad_pts[1], 0],
            color=YELLOW, stroke_width=3,
        )
        ac_line = Line(
            start=[*quad_pts[0], 0], end=[*quad_pts[2], 0],
            color=BLUE_C, stroke_width=2,
        )
        ad_line = Line(
            start=[*quad_pts[0], 0], end=[*quad_pts[3], 0],
            color=BLUE_C, stroke_width=2,
        )
        self.play(
            Create(ab_circle),
            *[
                real_dots[i].animate.set_color(c).scale(1.4)
                for i, c in zip(idx, colors)
            ],
            *[Create(c) for c in star_circles],
            Create(ab_line), Create(ac_line), Create(ad_line),
            run_time=1.0,
        )

        # Compute the code and display it as a query widget above the index
        # table on the right. This is the value we'll search for.
        code, _ = quad_code_2d(quad_pts)
        cx, cy, dx, dy = (
            float(code[0]), float(code[1]), float(code[2]), float(code[3])
        )
        code_widget = colored_code_cell((cx, cy, dx, dy), 0, 0, size=22)
        code_widget.move_to([INFO_CENTER[0], 0.6, 0])
        code_label = Text("query code:", font_size=20, color=GREY_B).next_to(
            code_widget, UP, buff=0.18
        )
        cap2 = caption_text("Compute its 4-D code → search the index")
        self.play(
            Transform(cap, cap2),
            FadeIn(code_label), FadeIn(code_widget),
            run_time=0.8,
        )
        self.wait(0.4)

        # Highlight the matched row in the docked table and draw a dashed
        # GREEN line from the query code down to the row.
        ra_str, dec_str = getattr(
            self, "_first_entry_radec", ("180.000°", "+45.000°")
        )
        row_box = None
        match_line = None
        if hasattr(self, "_index_table_rows") and self._index_table_rows:
            matched_row = self._index_table_rows[0]
            row_box = SurroundingRectangle(
                matched_row, color=GREEN, stroke_width=2, buff=0.04
            )
            match_line = DashedLine(
                code_widget.get_bottom() + DOWN * 0.05,
                row_box.get_top() + UP * 0.05,
                color=GREEN,
                stroke_width=1.5,
                dash_length=0.08,
            )
            self.play(
                Create(match_line), Create(row_box), run_time=0.8,
            )
        self.wait(0.5)

        # Pull the matched RA/Dec into the right info panel (between the
        # query-code widget and the docked table). Putting it in the right
        # panel keeps it off the captured starfield in the left frame, where
        # it would have overlapped real_dots.
        index_pull_label = (
            VGroup(
                Text("matched index entry:", font_size=14, color=GREY_B),
                Text(f"RA  = {ra_str}", font_size=16, color=RED),
                Text(f"Dec = {dec_str}", font_size=16, color=RED),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.05)
            .move_to([INFO_LEFT + 0.4, -0.2, 0], aligned_edge=UL)
        )
        self.play(FadeIn(index_pull_label), run_time=0.6)
        self.wait(1.0)

        cleanups = [
            FadeOut(cap),
            FadeOut(ab_circle),
            FadeOut(ab_line),
            FadeOut(ac_line),
            FadeOut(ad_line),
            FadeOut(star_circles),
            FadeOut(code_widget),
            FadeOut(code_label),
            FadeOut(index_pull_label),
            FadeOut(header),
            *[
                real_dots[i].animate.set_color(WHITE).scale(1 / 1.4)
                for i in idx
            ],
        ]
        if match_line is not None:
            cleanups.append(FadeOut(match_line))
        if row_box is not None:
            cleanups.append(FadeOut(row_box))
        self.play(*cleanups)

    # 4. hypothesis fitting ----------------------------------------------------

    def _stage4_fit(self, real_xy, real_dots):
        """Motivate stage 5: show that the index returns several plausible
        candidates whose quads have NEARLY THE SAME shape as the query.
        The quad alone can't tell which one is the real field — that's
        why the next stage verifies by checking the rest of the catalog
        against the rest of the scene."""
        header = header_text("4. Many candidates share similar quad shapes")
        self.play(Write(header))

        cap = caption_text(
            "Index returned several candidates — their quads all look alike"
        )
        self.play(FadeIn(cap))

        # Build a normalized version of the query quad — used as a template.
        quad_idx = self._first_entry_idx
        quad_pts = np.asarray(real_xy)[quad_idx]
        quad_centered = quad_pts - np.mean(quad_pts, axis=0)
        ab_len = float(np.linalg.norm(quad_centered[1] - quad_centered[0]))
        quad_normalized = quad_centered / ab_len

        abcd_colors = [RED, GREEN, BLUE, ORANGE]

        def mini_quad(center, perturb_seed):
            """4 colored ABCD circles + AB line + AC/AD lines + dashed
            AB-circle. Tiny per-call jitter on C and D so the two minis
            are NOT byte-for-byte identical (real candidates differ at
            sub-pixel level), but the difference is intentionally tiny —
            see the 'accentuated' note in the caption."""
            rng_local = np.random.default_rng(perturb_seed)
            scale = 1.10
            jitter = np.zeros((4, 2))
            jitter[2:] = rng_local.normal(0, 0.020, size=(2, 2))
            positions = [
                center + np.array([
                    (quad_normalized[i, 0] + jitter[i, 0]) * scale,
                    (quad_normalized[i, 1] + jitter[i, 1]) * scale,
                    0,
                ])
                for i in range(4)
            ]
            circles = VGroup(*[
                Circle(radius=0.14, color=c, stroke_width=3).move_to(positions[i])
                for i, c in enumerate(abcd_colors)
            ])
            ab_line = Line(positions[0], positions[1], color=YELLOW, stroke_width=2.5)
            ac_line = Line(positions[0], positions[2], color=BLUE_C, stroke_width=2)
            ad_line = Line(positions[0], positions[3], color=BLUE_C, stroke_width=2)
            ab_mid = (positions[0] + positions[1]) / 2
            ab_radius = float(np.linalg.norm(positions[1] - positions[0]) / 2)
            ab_circle = DashedVMobject(
                Circle(radius=ab_radius, color=GREY_C, stroke_width=1.5).move_to(ab_mid),
                num_dashes=36,
            )
            return VGroup(ab_circle, ac_line, ad_line, ab_line, circles)

        # Two mini quads side-by-side, larger and spread wider now that
        # we have the whole upper canvas to play in.
        mini_y = 0.4
        mini_xs = [-3.2, 3.2]
        cand1 = mini_quad(np.array([mini_xs[0], mini_y, 0]), perturb_seed=11)
        cand2 = mini_quad(np.array([mini_xs[1], mini_y, 0]), perturb_seed=22)

        cand1_label = VGroup(
            Text("Candidate 1", font_size=22, color=RED, weight=BOLD),
            Text("(wrong field)", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.06).next_to(cand1, DOWN, buff=0.35)
        cand2_label = VGroup(
            Text("Candidate 2", font_size=22, color=GREEN, weight=BOLD),
            Text("(correct field)", font_size=18, color=GREY_B),
        ).arrange(DOWN, buff=0.06).next_to(cand2, DOWN, buff=0.35)

        # "≈" between the two — visual shorthand for "shapes nearly equal".
        approx = Text("≈", font_size=72, color=GREY_B).move_to(
            [(mini_xs[0] + mini_xs[1]) / 2, mini_y, 0]
        )

        # Sub-caption above the captions calling out the accentuation —
        # without this the viewer might not realize how identical the
        # real candidates are at sub-pixel level.
        accent_note = Text(
            "(differences accentuated for clarity — real candidates are sub-pixel close)",
            font_size=15, color=GREY_C, slant=ITALIC,
        ).move_to([0, mini_y - 1.85, 0])

        self.play(
            FadeIn(cand1), FadeIn(cand1_label),
            FadeIn(approx),
            FadeIn(cand2), FadeIn(cand2_label),
            FadeIn(accent_note),
            run_time=1.0,
        )
        self.wait(1.6)

        cap2 = caption_text("Quad alone can't tell which is the real field")
        self.play(Transform(cap, cap2))
        self.wait(1.4)

        # Cleanup — stage 5 takes over to disambiguate.
        self.play(
            FadeOut(cand1), FadeOut(cand1_label),
            FadeOut(approx),
            FadeOut(cand2), FadeOut(cand2_label),
            FadeOut(accent_note),
            FadeOut(cap),
            FadeOut(header),
            run_time=0.5,
        )

    # 5. verification ---------------------------------------------------------

    def _stage5_verify(self, real_xy):
        """Bayesian verification (Lang et al. 2010 / zodiacal/src/verify.rs).
        Project each catalog star under a candidate WCS; per-source contribution
        to log-odds is:
          match:    log_fg - log_bg   (Gaussian likelihood vs uniform background)
          no match: log_distractor - log_bg
        with
          log_fg          = log((1-d) / (2π σ² NR)) - dist² / (2σ²)
          log_distractor  = log(d + (1-d) n_matched/NR) + log_bg
          log_bg          = log(1/area)
        The size of each per-star update depends on stellar density
        (NR catalog stars over `area`) and match noise σ — denser fields
        yield weaker per-match evidence."""
        header = header_text("5. Many candidates — verify by context")
        self.play(Write(header))

        # First: highlight several rows in the docked index table to show
        # the index search alone returned multiple plausible matches.
        cap = caption_text("Index search returns many possible matches")
        self.play(FadeIn(cap))
        candidate_boxes = VGroup()
        if hasattr(self, "_index_table_rows") and self._index_table_rows:
            for r in self._index_table_rows[: min(3, len(self._index_table_rows))]:
                candidate_boxes.add(
                    SurroundingRectangle(r, color=YELLOW, stroke_width=1.5, buff=0.03)
                )
            self.play(
                LaggedStart(*[Create(b) for b in candidate_boxes], lag_ratio=0.3),
                run_time=0.9,
            )
        self.wait(0.4)

        # Take down stage 4's catalog dots and the right-panel info card so
        # the image frame is clean for the per-candidate verification.
        teardown = []
        if hasattr(self, "cat_dots"):
            teardown.append(FadeOut(self.cat_dots))
        if hasattr(self, "fit_info"):
            teardown.append(FadeOut(self.fit_info))
        if teardown:
            self.play(*teardown, run_time=0.4)

        quad_idx = list(self._first_entry_idx)
        quad_pts = real_xy[quad_idx]
        real_xy_arr = np.asarray(real_xy, dtype=float)
        non_quad_idx = [i for i in range(len(real_xy_arr)) if i not in quad_idx]
        abcd_colors = [RED, GREEN, BLUE, ORANGE]

        # ---- Bayesian verification parameters --------------------------------
        NR = len(real_xy_arr)               # catalog star count in the patch
        image_area = FRAME_W * FRAME_H      # patch area in scene units²
        match_radius = 0.30                 # kdtree range-search radius (scene units)
        sigma = match_radius / 2.0          # mirrors verify.rs σ = match_radius / 2
        sigma_sq = sigma * sigma
        d_distractor = 0.10                 # noise/false-positive fraction
        log_bg = float(np.log(1.0 / image_area))
        log_gauss_peak = float(
            np.log((1.0 - d_distractor) / (2.0 * np.pi * sigma_sq * NR))
        )

        def per_star_delta(min_dist, n_matched_so_far):
            """Mirror of zodiacal/src/verify.rs:130-187 score-each-source loop.
            Returns (delta_log_odds, is_match)."""
            log_distractor = float(
                np.log(
                    d_distractor + (1.0 - d_distractor) * n_matched_so_far / NR
                )
            ) + log_bg
            if min_dist < match_radius:
                log_fg = log_gauss_peak - min_dist * min_dist / (2.0 * sigma_sq)
                if log_fg >= log_distractor:
                    return log_fg - log_bg, True
            return log_distractor - log_bg, False

        def quad_circles_at(positions):
            return VGroup(
                *[
                    Circle(radius=0.16, color=c, stroke_width=3).move_to(
                        [positions[i, 0], positions[i, 1], 0]
                    )
                    for i, c in zip(quad_idx, abcd_colors)
                ]
            )

        def context_circles_at(positions):
            """Each context star is rendered as a DASHED ring — visual cue
            that it's a PREDICTED catalog position (where the candidate WCS
            says we expect to see a real source) rather than an observed
            star. Per-star verification later sets the color to GREEN/RED
            to indicate match/miss against the actual scene."""
            return VGroup(
                *[
                    DashedVMobject(
                        Circle(radius=0.13, color=BLUE_C, stroke_width=2).move_to(
                            [positions[i, 0], positions[i, 1], 0]
                        ),
                        num_dashes=14,
                    )
                    for i in non_quad_idx
                ]
            )

        # ---- Right-panel widgets: log-odds counter + density callout --------
        odds_value = ValueTracker(0.0)

        def make_odds_text():
            v = odds_value.get_value()
            if v >= 20:
                color = GREEN
            elif v <= -20:
                color = RED
            else:
                color = YELLOW
            return Text(
                f"log-odds = {v:+.1f}",
                font_size=30, weight=BOLD, color=color,
            ).move_to([INFO_CENTER[0], 1.95, 0])

        odds_text = always_redraw(make_odds_text)
        threshold_text = Text(
            "accept ≥ +20      reject ≤ -20",
            font_size=14, color=GREY_C,
        ).move_to([INFO_CENTER[0], 1.45, 0])
        density_callout = VGroup(
            Text("per-star Δ scales with:", font_size=13, color=GREY_B),
            Text(f"  NR (catalog stars) = {NR}", font_size=13, color=GREY_C),
            Text(f"  σ (match radius)   = {sigma:.2f}", font_size=13, color=GREY_C),
            Text(f"  area = {image_area:.0f} units²", font_size=13, color=GREY_C),
            Text("  → denser field → smaller Δ per match", font_size=12, color=GREY_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([INFO_CENTER[0], 0.55, 0])

        self.play(
            FadeIn(odds_text), FadeIn(threshold_text), FadeIn(density_callout),
            run_time=0.6,
        )

        def run_per_star_verify(context_circles, projected_positions):
            """Animate the per-star log-odds evolution for one candidate.
            Returns (final_lo, n_matched)."""
            n_matched = 0
            running_lo = 0.0
            def schedule_linear_fade(mob, duration):
                """Fade `mob` from current opacity to 0 over `duration` seconds
                via an updater. Decoupled from any single self.play() so the
                fade keeps running across subsequent per-star plays."""
                state = {"t": 0.0, "fader": None, "start": mob.get_fill_opacity()}

                def fader(m, dt):
                    state["t"] += dt
                    if state["t"] >= duration:
                        m.set_opacity(0)
                        m.remove_updater(state["fader"])
                    else:
                        m.set_opacity(state["start"] * (1.0 - state["t"] / duration))

                state["fader"] = fader
                mob.add_updater(fader)

            for ci, bi in enumerate(non_quad_idx):
                dists = np.linalg.norm(
                    real_xy_arr - projected_positions[bi], axis=1
                )
                min_dist = float(np.min(dists))
                delta, is_match = per_star_delta(min_dist, n_matched)
                if is_match:
                    n_matched += 1
                running_lo += delta
                circle = context_circles[ci]
                color = GREEN if is_match else RED
                delta_tag = Text(
                    f"{delta:+.2f}",
                    font_size=16, weight=BOLD, color=color,
                ).next_to(circle, UP, buff=0.08)

                # If this is a hit, draw a short dashed leader from the
                # predicted-ring center to the actual matched real star —
                # makes the alignment visible without the viewer having to
                # eyeball "are these on top of each other?".
                match_line = None
                if is_match:
                    matched_real_idx = int(np.argmin(dists))
                    real_pos = real_xy_arr[matched_real_idx]
                    pred_pos = projected_positions[bi]
                    match_line = DashedLine(
                        start=[pred_pos[0], pred_pos[1], 0],
                        end=[real_pos[0], real_pos[1], 0],
                        color=GREEN,
                        stroke_width=1.5,
                        dash_length=0.04,
                    )

                # Pop in alongside the circle color change + counter step.
                play_args = [
                    circle.animate.set_color(color),
                    FadeIn(delta_tag),
                    odds_value.animate.set_value(running_lo),
                ]
                if match_line is not None:
                    play_args.append(Create(match_line))
                self.play(*play_args, run_time=0.22)
                # Detach the fade-outs from the play loop so the tag (and
                # the match leader, when present) fade over a leisurely
                # 0.75 s while later stars are processed.
                schedule_linear_fade(delta_tag, 0.75)
                if match_line is not None:
                    schedule_linear_fade(match_line, 0.75)
            return running_lo, n_matched

        # ---- Candidate 1: WRONG ------------------------------------------
        header_wrong = header_text("Candidate 1: WRONG match")
        cap_wrong = caption_text(
            "Project the catalog through this WCS — dashed rings = expected stars"
        )
        self.play(
            Transform(header, header_wrong),
            Transform(cap, cap_wrong),
            run_time=0.5,
        )

        # Wrong WCS — model as a totally different patch of sky: catalog
        # stars project to positions scattered uniformly across the frame
        # (poisson-disk-ish via rejection so they don't overlap each other
        # too much). Quad ABCD stays at its real positions because the
        # index match guarantees the quad code itself is consistent.
        rng_bad = np.random.default_rng(123)
        bad_positions = np.array(real_xy_arr, copy=True)
        placed_xy = [real_xy_arr[i] for i in quad_idx]
        min_sep = 0.30
        for i in non_quad_idx:
            for _ in range(60):
                cand = np.array([
                    rng_bad.uniform(FRAME_LEFT + 0.3, FRAME_RIGHT - 0.3),
                    rng_bad.uniform(FRAME_BOTTOM + 0.3, FRAME_TOP - 0.3),
                ])
                if all(np.linalg.norm(cand - p) > min_sep for p in placed_xy):
                    break
            bad_positions[i] = cand
            placed_xy.append(cand)

        bad_quad = quad_circles_at(bad_positions)
        bad_others = context_circles_at(bad_positions)
        odds_value.set_value(0.0)
        self.play(FadeIn(bad_quad), FadeIn(bad_others), run_time=0.5)

        # Hold on the dashed rings briefly so the viewer reads them as
        # PREDICTIONS before the per-star verification recolors them.
        expected_callout = Text(
            "each dashed ring = catalog star predicted by candidate WCS",
            font_size=15, color=BLUE_C,
        ).move_to([FRAME_CENTER[0], FRAME_BOTTOM - 0.3, 0])
        self.play(FadeIn(expected_callout), run_time=0.4)
        self.wait(0.9)
        self.play(FadeOut(expected_callout), run_time=0.3)

        # Switch the caption to the evolution narration as scoring begins.
        cap_wrong_evolve = caption_text(
            "Per-star log-odds evolves — bad WCS racks up negative evidence"
        )
        self.play(Transform(cap, cap_wrong_evolve), run_time=0.4)

        bad_lo, bad_matched = run_per_star_verify(bad_others, bad_positions)

        verdict_color = RED if bad_lo <= -20 else GREY_B
        reject = Text(
            f"final log-odds = {bad_lo:+.1f}   ⇒  REJECT",
            font_size=22, weight=BOLD, color=verdict_color,
        ).move_to([0, CAPTION_Y, 0])
        self.play(Transform(cap, reject))
        self.wait(1.4)
        self.play(FadeOut(bad_quad), FadeOut(bad_others), run_time=0.4)

        # ---- Candidate 2: CORRECT ----------------------------------------
        header_right = header_text("Candidate 2: CORRECT match")
        cap_right = caption_text(
            "Per-star log-odds evolves — good WCS racks up positive evidence"
        )
        self.play(
            Transform(header, header_right),
            Transform(cap, cap_right),
            run_time=0.5,
        )

        # Correct WCS — small residual jitter so the Gaussian likelihood is
        # high but not exactly at peak. Then perturb ONE non-quad catalog
        # star far enough that no real source matches it (simulates a star
        # the catalog predicts but that doesn't appear in the captured image
        # — cosmic ray, occultation, missed detection). The successful solve
        # still ACCEPTs because the rest of the catalog matches: real
        # verifiers don't need 100 % of catalog stars to be present.
        rng_local = np.random.default_rng(7)
        good_positions = real_xy_arr + rng_local.normal(
            0, 0.04, size=real_xy_arr.shape
        )
        missing_idx = non_quad_idx[len(non_quad_idx) // 2]  # one stable choice
        good_positions[missing_idx] = good_positions[missing_idx] + np.array(
            [0.65, 0.45]
        )
        good_positions[:, 0] = np.clip(
            good_positions[:, 0], FRAME_LEFT + 0.15, FRAME_RIGHT - 0.15
        )
        good_positions[:, 1] = np.clip(
            good_positions[:, 1], FRAME_BOTTOM + 0.15, FRAME_TOP - 0.15
        )

        good_quad = quad_circles_at(good_positions)
        good_others = context_circles_at(good_positions)
        odds_value.set_value(0.0)
        self.play(FadeIn(good_quad), FadeIn(good_others), run_time=0.5)
        self.wait(0.3)

        good_lo, good_matched = run_per_star_verify(good_others, good_positions)

        verdict_color2 = GREEN if good_lo >= 20 else GREY_B
        accept = Text(
            f"final log-odds = {good_lo:+.1f}   ⇒  ACCEPT",
            font_size=22, weight=BOLD, color=verdict_color2,
        ).move_to([0, CAPTION_Y, 0])
        self.play(Transform(cap, accept))
        self.wait(1.4)

        # ---- Follow-on: least-squares fit on the accepted matches gives
        # the image-center pointing. Show small residual vectors from each
        # matched-prediction ring to its real source, then collapse them
        # into a crosshair at FRAME_CENTER tagged with the WCS-derived
        # RA/Dec for the image center.
        ls_header = header_text("Least-squares fit → image-center pointing")
        ls_cap = caption_text(
            "Solve a TAN-WCS that minimizes residuals on the matched stars"
        )
        self.play(
            Transform(header, ls_header),
            Transform(cap, ls_cap),
            run_time=0.5,
        )

        residuals = VGroup()
        for ci, bi in enumerate(non_quad_idx):
            if bi == missing_idx:
                continue  # the missing one isn't a matched star
            pred = np.array([good_positions[bi, 0], good_positions[bi, 1], 0])
            real = np.array([real_xy_arr[bi, 0], real_xy_arr[bi, 1], 0])
            residuals.add(Line(pred, real, color=YELLOW, stroke_width=1.8))
        self.play(
            LaggedStart(*[Create(r) for r in residuals], lag_ratio=0.04),
            run_time=0.9,
        )
        self.wait(0.4)

        # Collapse residuals → image center crosshair.
        ra_str_ctr, dec_str_ctr = self._first_entry_radec
        ch_x, ch_y = FRAME_CENTER[0], FRAME_CENTER[1]
        crosshair = VGroup(
            Line(
                [ch_x - 0.30, ch_y, 0], [ch_x + 0.30, ch_y, 0],
                color=GREEN, stroke_width=3,
            ),
            Line(
                [ch_x, ch_y - 0.30, 0], [ch_x, ch_y + 0.30, 0],
                color=GREEN, stroke_width=3,
            ),
            Circle(radius=0.18, color=GREEN, stroke_width=2.5).move_to(
                [ch_x, ch_y, 0]
            ),
        )
        center_label = (
            VGroup(
                Text("image center:", font_size=14, color=GREY_B),
                Text(f"RA  = {ra_str_ctr}", font_size=18, color=GREEN, weight=BOLD),
                Text(f"Dec = {dec_str_ctr}", font_size=18, color=GREEN, weight=BOLD),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.05)
            .next_to(crosshair, DOWN, buff=0.18)
        )
        self.play(
            *[r.animate.scale(0.0).set_opacity(0) for r in residuals],
            FadeIn(crosshair),
            FadeIn(center_label),
            run_time=0.8,
        )
        self.wait(2.0)

        cleanups = [
            FadeOut(good_quad),
            FadeOut(good_others),
            FadeOut(cap),
            FadeOut(header),
            FadeOut(odds_text),
            FadeOut(threshold_text),
            FadeOut(density_callout),
            FadeOut(residuals),
            FadeOut(crosshair),
            FadeOut(center_label),
        ]
        if len(candidate_boxes) > 0:
            cleanups.append(FadeOut(candidate_boxes))
        self.play(*cleanups)

    # outro --------------------------------------------------------------------

    def _outro(self):
        # Clear panels and any leftover image-space content
        leftover = []
        if hasattr(self, "real_dots"):
            leftover.append(FadeOut(self.real_dots))
        self.play(
            FadeOut(self.image_frame),
            FadeOut(self.info_frame),
            FadeOut(self.image_label),
            *leftover,
        )
        big = Text("Pointing solved.", font_size=44, weight=BOLD, color=GREEN)
        wcs = Text(
            "RA, Dec, position angle, plate scale  →  PlateSolution",
            font_size=22,
            color=GREY_B,
        ).next_to(big, DOWN)
        self.play(Write(big), FadeIn(wcs, shift=UP * 0.2))
        self.wait(1.6)
        self.play(FadeOut(big), FadeOut(wcs))

    # ---- Addendum: healpix subdivision ---------------------------------------

    def _addendum_healpix(self):
        """Show the HEALPix sphere being progressively subdivided through
        nside ∈ {1, 2, 4, 8, 16}. A solid-fill circumsphere sits behind
        the tile polygons so the silhouette stays clean even when coarse
        tiles don't tessellate the sphere edge perfectly. At each scale
        we highlight a 3×3 cluster of patches (a center pixel + its 8
        neighbours), spawn an ABCD quad over the cluster, and fly that
        quad off into a running 'index built' counter — motivating that
        every scale contributes its own chunk of catalog quads."""
        import healpy as hp

        # Defensive cleanup — fade anything that survived earlier stages
        # (e.g. the docked index table) before we draw the sphere.
        leftovers = [FadeOut(m) for m in list(self.mobjects)]
        if leftovers:
            self.play(*leftovers, run_time=0.4)

        title = Text(
            "Addendum: HEALPix subdivision",
            font_size=34, weight=BOLD, color=YELLOW,
        ).move_to([0, 3.0, 0])
        subtitle = Text(
            "Hierarchical Equal-Area isoLatitude Pixelization",
            font_size=18, color=GREY_B,
        ).next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.15))
        self.wait(0.3)

        # Fixed orthographic projection — pitch about X, yaw about Y. Sphere
        # shifted left so the right side of the scene is free for the
        # running "index built" counter and the per-scale ABCD quads.
        sphere_center = np.array([-2.3, -0.4, 0])
        sphere_R = 2.2
        pitch = np.radians(28)
        yaw = np.radians(-22)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cyy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Ry = np.array([[cyy, 0, sy], [0, 1, 0], [-sy, 0, cyy]])
        R = Rx @ Ry

        def project_one(xyz):
            rotated = R @ xyz
            return sphere_center + np.array([rotated[0], rotated[1], 0]) * sphere_R

        # Circumsphere — solid-fill silhouette drawn FIRST so the tile
        # polygons render on top. Fills any silhouette gap where culled
        # back-faces would have been visible.
        circumsphere = Circle(
            radius=sphere_R,
            color=GREY_D, stroke_width=1.5,
            fill_color=BLUE_E, fill_opacity=0.45,
        ).move_to(sphere_center)
        self.play(Create(circumsphere), run_time=0.5)

        # Right-column index-build readout
        index_label_top = Text(
            "Index growing:", font_size=18, color=GREY_B,
        ).move_to([3.5, 1.95, 0])
        index_count_value = ValueTracker(0)
        index_count_text = always_redraw(
            lambda: Text(
                f"{int(index_count_value.get_value()):,} quads",
                font_size=26, weight=BOLD, color=YELLOW,
            ).move_to([3.5, 1.45, 0])
        )
        self.play(FadeIn(index_label_top), FadeIn(index_count_text), run_time=0.4)

        def silhouette_arc_pts(angle_start, angle_end, num=10):
            """Sub-divided arc on the silhouette circle from angle_start to
            angle_end via the SHORTER arc. Returns intermediate screen points
            (excluding endpoints — the caller already supplied those as the
            exact boundary-crossing intersections)."""
            delta = angle_end - angle_start
            while delta > np.pi:
                delta -= 2 * np.pi
            while delta < -np.pi:
                delta += 2 * np.pi
            pts = []
            for k in range(1, num):
                t = k / num
                a = angle_start + delta * t
                pts.append(
                    sphere_center
                    + np.array([np.cos(a), np.sin(a), 0]) * sphere_R
                )
            return pts

        def visible_polygon_pts(boundary_xyz, arc_segments=10):
            """Walk a tile boundary, replacing each back-facing run with the
            silhouette arc from the front→back crossing point around to the
            back→front crossing point. Front portions render as projected
            boundary; back portions render as a true silhouette arc — so
            tiles that straddle the terminator are still drawn correctly,
            with the visible front edge bounded by the sphere's silhouette."""
            n = boundary_xyz.shape[1]
            rotated_z = R[2] @ boundary_xyz

            out = []
            pending_exit_angle = None

            for i in range(n):
                j = (i + 1) % n
                zi = float(rotated_z[i])
                zj = float(rotated_z[j])
                pi = boundary_xyz[:, i]
                pj = boundary_xyz[:, j]

                if zi >= 0:
                    out.append(project_one(pi))

                if zi >= 0 and zj < 0:
                    # Front → back crossing — find intersection on z=0
                    # silhouette in rotated frame, renormalize to sphere.
                    t = zi / (zi - zj)
                    cross = pi * (1 - t) + pj * t
                    cross /= np.linalg.norm(cross)
                    cross_rot = R @ cross
                    pending_exit_angle = float(
                        np.arctan2(cross_rot[1], cross_rot[0])
                    )
                    out.append(project_one(cross))

                if zi < 0 and zj >= 0:
                    # Back → front crossing — close the silhouette arc.
                    t = -zi / (zj - zi)
                    cross = pi * (1 - t) + pj * t
                    cross /= np.linalg.norm(cross)
                    cross_rot = R @ cross
                    angle_enter = float(
                        np.arctan2(cross_rot[1], cross_rot[0])
                    )
                    if pending_exit_angle is not None:
                        out.extend(silhouette_arc_pts(
                            pending_exit_angle, angle_enter, arc_segments
                        ))
                        pending_exit_angle = None
                    out.append(project_one(cross))

            return out

        def tiles_for_nside(nside):
            """Build front-visible HEALPix tiles, including those that
            straddle the silhouette (clipped properly, not dropped)."""
            npix = hp.nside2npix(nside)

            # More boundary subdivisions for coarser nside (smoother arcs).
            step = 8 if nside <= 1 else (4 if nside <= 4 else 2)
            arc_segments = 14 if nside <= 1 else (10 if nside <= 4 else 6)

            tiles = VGroup()
            polys_by_pix = {}
            for ipix in range(npix):
                xyz = hp.boundaries(nside, int(ipix), step=step)
                rotated_z_all = R[2] @ xyz
                # Drop only pixels that are entirely on the back hemisphere.
                if np.max(rotated_z_all) < 0.0:
                    continue
                screen_pts = visible_polygon_pts(xyz, arc_segments=arc_segments)
                if len(screen_pts) < 3:
                    continue  # degenerate after clipping
                poly = Polygon(
                    *screen_pts,
                    color=GREY_B, stroke_width=1.0,
                    fill_color=BLUE_E, fill_opacity=0.22,
                )
                tiles.add(poly)
                polys_by_pix[int(ipix)] = poly
            return tiles, npix, polys_by_pix

        def pick_cluster_center(nside, polys_by_pix):
            """Pick a front-facing pixel (in polys_by_pix) whose 8 neighbours
            are also front-facing — guarantees a clean 3×3 cluster."""
            for ipix in polys_by_pix:
                neighbours = hp.get_all_neighbours(nside, ipix)
                if all(n >= 0 and int(n) in polys_by_pix for n in neighbours):
                    return ipix
            # Fallback: pick the pixel with the most visible neighbours.
            best, best_n = None, -1
            for ipix in polys_by_pix:
                neighbours = hp.get_all_neighbours(nside, ipix)
                count = sum(1 for n in neighbours if n >= 0 and int(n) in polys_by_pix)
                if count > best_n:
                    best, best_n = ipix, count
            return best

        nside_levels = [1, 2, 4, 8, 16]
        # Quads-per-tile: rough multiplier so the running counter grows
        # nicely; the actual ratio depends on star density per tile.
        quads_per_tile = 12

        cur_tiles = None
        cur_label = None

        for li, nside in enumerate(nside_levels):
            new_tiles, npix, polys_by_pix = tiles_for_nside(nside)
            new_label = Text(
                f"nside = {nside}    ({npix:,} pixels)",
                font_size=22, color=YELLOW,
            ).move_to([sphere_center[0], -3.1, 0])

            if cur_tiles is None:
                self.play(FadeIn(new_tiles), FadeIn(new_label), run_time=1.0)
                cur_label = new_label
            else:
                self.play(
                    FadeOut(cur_tiles),
                    FadeIn(new_tiles),
                    Transform(cur_label, new_label),
                    run_time=0.8,
                )
            cur_tiles = new_tiles

            # ---- Highlight a 3×3 cluster (a pixel + its 8 neighbours) ----
            center_ipix = pick_cluster_center(nside, polys_by_pix)
            neighbours = hp.get_all_neighbours(nside, center_ipix)
            cluster_ipixs = [center_ipix] + [
                int(n) for n in neighbours if n >= 0 and int(n) in polys_by_pix
            ]
            cluster_polys = [polys_by_pix[i] for i in cluster_ipixs]
            self.play(
                *[
                    p.animate.set_fill(YELLOW, opacity=0.55).set_stroke(
                        YELLOW, width=1.5
                    )
                    for p in cluster_polys
                ],
                run_time=0.5,
            )

            # ---- Spawn an ABCD quad over the cluster center, then fly it
            # off into the running "index built" counter.
            theta_c, phi_c = hp.pix2ang(nside, center_ipix)
            center_xyz = np.array([
                np.sin(theta_c) * np.cos(phi_c),
                np.sin(theta_c) * np.sin(phi_c),
                np.cos(theta_c),
            ])
            cs_screen = project_one(center_xyz)
            # Spread A,B,C,D in a diamond so the AB-circle constraint reads.
            r_quad = max(0.12, 0.55 / max(nside, 1))
            offsets = [
                np.array([-r_quad * 0.9, -r_quad * 0.7, 0]),  # A
                np.array([+r_quad * 0.9, +r_quad * 0.7, 0]),  # B
                np.array([+r_quad * 0.2, -r_quad * 0.4, 0]),  # C
                np.array([-r_quad * 0.3, +r_quad * 0.5, 0]),  # D
            ]
            abcd_colors = [RED, GREEN, BLUE, ORANGE]
            abcd_dots = VGroup(*[
                Dot(cs_screen + off, radius=0.06, color=c, stroke_width=1.5,
                    stroke_color=WHITE)
                for off, c in zip(offsets, abcd_colors)
            ])
            ab_line_q = Line(
                cs_screen + offsets[0], cs_screen + offsets[1],
                color=YELLOW, stroke_width=1.8,
            )
            ac_line_q = Line(
                cs_screen + offsets[0], cs_screen + offsets[2],
                color=BLUE_C, stroke_width=1.2,
            )
            ad_line_q = Line(
                cs_screen + offsets[0], cs_screen + offsets[3],
                color=BLUE_C, stroke_width=1.2,
            )
            quad_group = VGroup(ab_line_q, ac_line_q, ad_line_q, abcd_dots)
            self.play(FadeIn(quad_group), run_time=0.4)
            self.wait(0.2)

            # Fly the highlighted cluster's quad into the index counter,
            # ticking up by just the cluster's contribution (one quad
            # icon = one demo of the per-tile process).
            target_pos = np.array([3.5, 1.45, 0])
            cluster_added = len(cluster_ipixs) * quads_per_tile
            self.play(
                quad_group.animate.move_to(target_pos).scale(0.25).set_opacity(0),
                index_count_value.animate.set_value(
                    int(index_count_value.get_value()) + cluster_added
                ),
                run_time=0.7,
            )

            # ---- Motivate "every visible pixel contributes" with a
            # sequenced sweep across the non-cluster tiles. Order by screen
            # Y descending then X ascending so the sweep flows top-to-bottom,
            # left-to-right — the eye reads it as 'we're walking through all
            # the patches in order'. Each tile pops yellow then settles back
            # to blue. The index counter ticks up across the same play so
            # the count growth is visibly correlated with the sweep.
            other_ipixs = [ip for ip in polys_by_pix if ip not in cluster_ipixs]
            level_added = npix * quads_per_tile
            remaining_added = level_added - cluster_added

            all_cap = Text(
                "every pixel contributes a batch of quads",
                font_size=16, color=YELLOW, slant=ITALIC,
            ).move_to([sphere_center[0], -3.55, 0])

            if other_ipixs:
                # Pre-compute the rotated (screen-space) center for every
                # visible non-cluster pixel via healpy, ordered top→bottom
                # then left→right so the cascade reads as a coherent sweep.
                ipix_arr = np.asarray(other_ipixs, dtype=int)
                thetas, phis = hp.pix2ang(nside, ipix_arr)
                centers = np.stack([
                    np.sin(thetas) * np.cos(phis),
                    np.sin(thetas) * np.sin(phis),
                    np.cos(thetas),
                ])  # (3, N)
                rot_centers = R @ centers  # (3, N)
                # Sort by (-screen_y, screen_x). lexsort uses the LAST key
                # as primary, so put screen_x first and -screen_y second.
                order = np.lexsort((rot_centers[0], -rot_centers[1]))
                ordered = [int(ipix_arr[i]) for i in order]

                # Sweep EVERY visible non-cluster pixel — no cap. Use a
                # very tight lag and short per-tile pop so total time stays
                # bounded even when nside=16 has 1k+ tiles.
                flashed_polys = [polys_by_pix[ip] for ip in ordered]
                n_flash = len(flashed_polys)
                pop_dur = 0.05
                settle_dur = 0.10
                per_anim = pop_dur + settle_dur

                # Total play time: ~1.5s for small N, growing to ~3.5s for
                # nside=16. Solve for lag_ratio that fits.
                target_total = float(np.clip(n_flash * 0.0035, 1.5, 3.5))
                if n_flash > 1:
                    lag_ratio = (target_total - per_anim) / (
                        (n_flash - 1) * per_anim
                    )
                    lag_ratio = max(0.001, lag_ratio)
                else:
                    lag_ratio = 0.0

                flash_anims = [
                    Succession(
                        p.animate(run_time=pop_dur).set_fill(
                            YELLOW, opacity=0.55
                        ).set_stroke(YELLOW, width=1.5),
                        p.animate(run_time=settle_dur).set_fill(
                            BLUE_E, opacity=0.22
                        ).set_stroke(GREY_B, width=1.0),
                    )
                    for p in flashed_polys
                ]

                self.play(
                    FadeIn(all_cap),
                    LaggedStart(*flash_anims, lag_ratio=lag_ratio),
                    index_count_value.animate.set_value(
                        int(index_count_value.get_value()) + remaining_added
                    ),
                    run_time=target_total,
                )
                self.play(FadeOut(all_cap), run_time=0.3)
            elif remaining_added > 0:
                # No other visible tiles (coarse nside, cluster covers all);
                # still tick the counter for the back-facing tiles' quads.
                self.play(
                    FadeIn(all_cap),
                    index_count_value.animate.set_value(
                        int(index_count_value.get_value()) + remaining_added
                    ),
                    run_time=0.6,
                )
                self.play(FadeOut(all_cap), run_time=0.3)

            # Unhighlight the cluster so the next scale starts fresh.
            self.play(
                *[
                    p.animate.set_fill(BLUE_E, opacity=0.22).set_stroke(
                        GREY_B, width=1.0
                    )
                    for p in cluster_polys
                ],
                run_time=0.3,
            )

        self.wait(1.5)
        self.play(
            FadeOut(cur_tiles),
            FadeOut(cur_label),
            FadeOut(circumsphere),
            FadeOut(index_label_top),
            FadeOut(index_count_text),
            FadeOut(title),
            FadeOut(subtitle),
            run_time=0.6,
        )
