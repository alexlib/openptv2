# trackcorr Candidate Viewer (Interactive)

## Overview

An interactive marimo notebook for tuning trackcorr's tracking parameters by
eye: pick a particle in a frame, see every candidate trackcorr's real search
considered in the next frame (ranked and labeled by cost), which one it
picked and why the others lost -- overlaid on the camera images.

This replaces the older ["Tracking -> Debugging with
display"](tracking_debug_visualization.md) GUI panel, which does not work
(see that page's warning). The viewer here reconstructs the *real* trackcorr
search by calling the same compiled kernel the tracker itself uses --
nothing here is approximated or reimplemented.

Scope: **trackcorr only** (`track_mode=0`). Other trackers (track3d /
priority_segment_3d) are not covered.

## Running it

```bash
uv run --extra viz marimo run src/openptv2/gui/trackcorr_debug_nb.py -- \
    --dataset test_data/test_cavity --first 10001 --last 10002 --particle 0
```

This opens a read-only app view (sliders/buttons live, no visible code). To
edit the notebook itself:

```bash
uv run --extra viz marimo edit src/openptv2/gui/trackcorr_debug_nb.py
```

`--dataset` is a directory containing `parameters.yaml`, `cal/`, `img_orig/`
(or `img/`), and `res_orig/` (or `res/`) -- point it at your own experiment
once you have run sequence + tracking there. `--first`/`--last` bound the
frame range trackcorr steps over (`--last` is exclusive); `--particle` is
the row index of the particle to inspect in the first stepped frame. All
four are also editable fields in the notebook itself, so the CLI flags are
just convenient defaults.

## What you see

- **One matplotlib panel per camera**, each showing that camera's image for
  the frame the candidates live in. Small gray dots are every detection in
  that frame; colored circles are the candidates trackcorr's real search
  actually evaluated (passed the acceleration/angle gate), labeled with
  their rank and tracer ID; the winning link is outlined in red.
- **A candidate table** below the plot: rank, cost (lower is better), the
  candidate's row, which cameras it was seen in, and whether it's the
  winner.
- **A status line** telling you whether you're looking at the *real* link
  trackcorr made during the actual run, or an *isolated probe* recomputed
  with your tuned parameters.

## Tuning parameters

The `dvxmin/max`, `dvymin/max`, `dvzmin/max`, `dacc`, `dangle` sliders
default to the run's real tracking parameters. Nothing recomputes as you
drag them -- a single-particle recompute is roughly 200x faster than a real
full-frame step, but re-running on every drag event is still needless
recompute for what is a "look, then decide" workflow. Move the sliders, then
press **Run with these parameters** to see the candidate set and winner the
tuned parameters would produce for that one particle.

Because this probes *one particle in isolation*, its winner is the
best-by-cost candidate for that particle alone -- it does not account for
cross-particle competition for the same next-frame target, which the real
full-frame trackcorr run resolves globally. The status line makes this
distinction explicit. Use the isolated probe to see how a parameter change
affects one particle's candidate set and costs; re-run the real sequence in
the GUI/CLI to confirm the effect on the actual tracked result.
