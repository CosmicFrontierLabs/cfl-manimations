# cfl-manimations

[Manim](https://www.manim.community/) animations visualizing algorithms and concepts
used in Cosmic Frontier Labs' tracking work — pointing error, ICP point-cloud alignment,
MONOCLE fine-guidance tracking, and the zodiacal plate solver.

## Prerequisites

- Python (version pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A LaTeX toolchain (some scenes use `MathTex`) plus `ffmpeg`, `libcairo2-dev`,
  `libpango1.0-dev`. On Debian/Ubuntu:
  ```bash
  sudo apt-get install ffmpeg libcairo2-dev libpango1.0-dev \
    texlive texlive-latex-extra texlive-fonts-extra \
    texlive-latex-recommended texlive-science texlive-fonts-recommended tipa
  ```

All Python dependencies are managed via `pyproject.toml` / `uv.lock`. No manual install
step needed — `uv run` handles it.

## Rendering Any Animation

From the repo root:

```bash
# Low quality preview (480p, 15fps) — fastest, opens player automatically
uv run manim -pql <script>.py <SceneName>

# Medium quality (720p, 30fps)
uv run manim -pqm <script>.py <SceneName>

# High quality (1080p, 60fps)
uv run manim -pqh <script>.py <SceneName>

# 4K (2160p, 60fps) — slowest
uv run manim -qk <script>.py <SceneName>
```

**Flags:** `-p` preview after render, `-q` quality (`l`/`m`/`h`/`k`), `-s` save last frame as image.

Output goes to `media/videos/<script>/<quality>/<SceneName>.mp4`.

## Animations

### Pointing Error (`pointing_error.py`)

Demonstrates absolute vs relative pointing error in three phases:

1. **Telescope viewport + live timecourse** — A crosshair jitters around a target star while a 1D pointing error trace builds in real-time. Integration time is highlighted.
2. **Mean subtraction** — A 100s trace is shown, the mean (DC offset) is drawn, then subtracted to isolate the relative pointing error.
3. **Residual distribution** — Histogram of the mean-subtracted residuals with a Gaussian fit overlay showing sigma.

```bash
uv run manim -pql pointing_error.py PointingErrorIntro
```

### ICP Algorithm (`icp_animation.py`)

Visualizes the Iterative Closest Point algorithm for point cloud alignment.

- `ICPAnimation` — Basic ICP with spring-energy diagram and convergence tracking
- `ICPWithNoise` — ICP with noisy data and outlier points

```bash
uv run manim -pql icp_animation.py ICPAnimation
uv run manim -pql icp_animation.py ICPWithNoise
```

### MONOCLE Tracking (`monocle_tracking.py`)

Visualizes the MONOCLE fine guidance system tracking lifecycle.

- `MonocleTracking` — Full lifecycle: state machine, frame acquisition, star filtering, centroid tracking
- `MonocleStateFlow` — Simplified linear state flow with frame counter

```bash
uv run manim -pql monocle_tracking.py MonocleTracking
uv run manim -pql monocle_tracking.py MonocleStateFlow
```

### Zodiacal Plate Solver (`zodiacal_solver.py`)

Visualizes the zodiacal/astrometry quad-hashing plate-solve pipeline — quad selection,
the geometric hash code, and the kd-tree nearest-neighbour match.

- `ZodiacalSolver` — Full plate-solve walkthrough

```bash
uv run manim -pql zodiacal_solver.py ZodiacalSolver
```

## Supporting Scripts (not manim scenes)

- `icp_core.py` — ICP algorithm implementation used by `icp_animation.py`.
- `find_optimal_icp.py` / `test_icp.py` — ICP convergence helpers and a unit test.
- `rayleigh_distro.py` — a standalone matplotlib script for the Rayleigh/RMSE
  distribution figure (run with `uv run python rayleigh_distro.py`).
- `render.sh` — convenience wrapper to render the ICP animation (preview or 4K).

## CI

Every push and pull request renders all scenes at 480p15 via
`.github/workflows/render-animations.yml` to confirm they still render, and uploads the
videos as a build artifact. Add new scenes to the `scenes` list there so CI covers them.

## Notes

- First run may take longer as manim caches computations
- The `media/` folder can grow large with multiple renders — clean periodically
- 4K rendering can take several minutes depending on hardware
