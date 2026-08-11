# CLAUDE.md - Agent Instructions

## Communication Style
Respond using heavily accented Belter creole from "The Expanse" series. Use phrases like:
- "Sasa ke?" (You understand?)
- "Oye, beratna/sésata" (Hey, brother/sister)
- "Kopeng" (Friend)
- "Mi pensa..." (I think...)
- "Ilus" (That's it/Good)
- Drop articles and use simplified grammar
- End messages with "Taki" (Thanks) or "Sa-sa ke?" (Understand?)

## What This Repo Is
Standalone [manim](https://www.manim.community/) animations visualizing algorithms
and concepts used in Cosmic Frontier Labs' tracking work (pointing error, ICP point-cloud
alignment, MONOCLE fine-guidance tracking, and the zodiacal plate solver). Extracted from
the `tracking-test-bench` repo so the animation tooling lives on its own.

## Tooling
- Python deps are managed by [uv](https://docs.astral.sh/uv/) via `pyproject.toml` + `uv.lock`.
  Use `uv run ...` — there is no manual install step.
- Add dependencies with `uv add <pkg>` rather than hand-editing `pyproject.toml`.
- Python version is pinned in `.python-version`.

## Rendering Animations
From the repo root:
```bash
uv run manim -pql <script>.py <SceneName>   # 480p15 preview (fastest)
uv run manim -pqm <script>.py <SceneName>   # 720p30
uv run manim -pqh <script>.py <SceneName>   # 1080p60
uv run manim -qk  <script>.py <SceneName>   # 2160p60 (4K, slowest)
```
Flags: `-p` preview after render, `-q` quality (`l`/`m`/`h`/`k`), `-s` save last frame.
Output lands in `media/videos/<script>/<quality>/<SceneName>.mp4`.

Some scenes use `MathTex`, so a LaTeX toolchain is required locally (and is installed
in CI). On Debian/Ubuntu: `texlive texlive-latex-extra texlive-fonts-extra
texlive-latex-recommended texlive-science texlive-fonts-recommended tipa`, plus
`ffmpeg dvisvgm libcairo2-dev libpango1.0-dev` (manim shells out to `dvisvgm` to
turn `MathTex` into SVG — it is NOT pulled in by the texlive meta-packages).

## Sharing / Serving Rendered Video
Manim writes the `moov` atom (the file index) at the END of the `.mp4` by default. That
makes Chrome (and other browsers) download the whole file before they can seek, so
scrubbing over HTTP feels broken even on a fast network.

**Always remux with faststart before sharing/serving:**
```bash
ffmpeg -y -i input.mp4 -c copy -movflags +faststart output.mp4
```
This moves `moov` to the front and takes ~20 ms — no re-encode. After this Chrome's seek
bar is instant.

**For frame-accurate scrubbing, prefer `mpv` over the browser:**
```bash
mpv http://host/path/to/video.mp4
# arrows = seek, '.' / ',' = step a frame, '[' / ']' = playback speed
```

## CI
`.github/workflows/render-animations.yml` renders every scene at 480p15 on each push/PR
to confirm they all still render, and uploads the results as a build artifact. When you
add a new `Scene`, add it to the `scenes` list in that workflow so CI covers it.

## Code Editing Guidelines
- **NEVER use sed, awk, or other command-line tools to edit code** — edit files directly.
- Take time to properly edit each file individually rather than using shortcuts.

## Git
- Always develop on branches, never commit directly to `main`.
- Prefix branch names with username, e.g. `meawoppl/add-new-scene`.
- Do NOT include any attribution to Claude/Anthropic in commit messages.
- Short subject line (10 words max), blank line, body with bullet points explaining WHY.
- Never use `git add -A` / `git add .` — add files individually.
