# Auto-Calibration with `openptv warmup`

`openptv warmup` measures a small window of your sequence, tunes the tracking
search parameters (`dvxmin`/`dvxmax`/.../`dacc`) from what it actually finds,
picks between the two directly-supported tracking engines, and (optionally)
writes the result back into your `parameters.yaml` — so you don't have to
guess `dvxmax`/`dacc` by hand or run a manual parameter sweep.

It is a **standalone step you run before tracking**, as many times as you
like. It never runs automatically as part of `openptv track`.

## Prerequisites

- A `parameters.yaml` experiment directory with calibration (`cal/`) already
  done.
- The **sequence step already run** (`res/rt_is.#` correspondences and
  `img/camN.#_targets` present, or written into the run's zarr store) — warmup
  measures existing correspondences, it does not detect particles itself.
- A zarr store for the run (`res/run.zarr`), the way a normal sequence run
  produces one. See [Batch Processing](batch_processing.md) and
  `docs/zarr-hdf5-storage.md` for how the store gets created.

## Basic usage

```bash
uv run openptv warmup parameters.yaml
```

This is a **dry run** — it prints a report but does not touch your YAML.
Add `--write` once you're happy with the result:

```bash
uv run openptv warmup parameters.yaml --write
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `yaml_file` | (required) | Path to the experiment's `parameters.yaml` |
| `--frames N` | 25 | Size of the measurement window, starting at the sequence's first frame |
| `--max-cycles N` | 3 | Max tuning cycles (each cycle re-measures and re-tunes) |
| `--write` | off | Persist the chosen tracker + tuned params into the YAML |

## What it actually does

For each cycle, on the frame window:

1. Runs forward **and** backward tracking (`Tracker.full_forward()` +
   `full_backward()`) on a scratch linkage group — your real tracking output
   is untouched.
2. Runs `enforce_reciprocity` (does every forward link have a matching
   backward link?) as a **ground-truth-free quality signal** — this works
   the same way on real data as on synthetic data, since it never needs to
   know the true particle identities.
3. Reads the displacement distribution of the *confirmed* (reciprocal) links
   and uses its 99th percentile to set the velocity search box
   (`dvxmin`/`dvxmax`/.../`dvzmax`/`dacc`) for the next cycle.
4. After the cycles settle, it runs the two directly-supported engines
   (`priority_segment_3d` = track3d, `full_multipass` = trackcorr
   forward+backward) with the tuned parameters and picks whichever produces
   longer average trajectories.

## Example output

```text
$ uv run openptv warmup parameters.yaml --frames 8 --max-cycles 2

Warmup window: frames 10001-10008 (2 cycle(s))
Chosen tracker: priority_segment_3d  (engine scores: {'priority_segment_3d': 6.51, 'full_multipass': 3.29})
Forward/backward agreement: 100.0%
Empirical noise estimate: 2.920 mm
Tuned track params:
  dvxmin: -13.284303841747132
  dvxmax: 13.284303841747132
  dvymin: -13.284303841747132
  dvymax: 13.284303841747132
  dvzmin: -13.284303841747132
  dvzmax: 13.284303841747132
  dangle: 120.0
  dacc: 13.284303841747132
  add: 1

Dry run (pass --write to persist into the YAML)
```

With `--write`, the same run updates `parameters.yaml`'s `track:` block
(`dvxmin`/`dvxmax`/.../`dacc`/`angle`) and `plugins.selected_tracking`, so a
plain `openptv track` afterward picks up the tuned config automatically —
`openptv track` has no warmup-awareness of its own, it just reads whatever
config is in the YAML.

## Reading the report

- **Chosen tracker** — whichever of the two engines produced longer average
  trajectories on the window with the tuned parameters. `engine scores` are
  the mean trajectory length for each.
- **Forward/backward agreement** — the fraction of forward links confirmed
  by a matching backward link. This is a *self-consistency* measure, not a
  correctness measure against ground truth — see the caveat below.
- **Empirical noise estimate** — the standard deviation of confirmed-link
  displacements, in mm. A rough proxy for how much position noise the setup
  has, useful for sanity-checking whether your calibration/detection is
  behaving as expected for this rig.
- **Tuned track params** — what gets written with `--write`.

## Known limitation: check the result before trusting it

Benchmarking against real ground truth (`scripts/benchmark_stage_improvements.py`,
see `docs/plans/2026-08-15-tracking-quality-overhaul.md`'s Stage 1 write-up)
found that warmup's current tuning margin can **overshoot** on data that was
already reasonably configured: it widens the search box using
`p99(confirmed displacements) × 3.0`, a margin calibrated against a
deliberately under-tuned case. On several measured scenes this widened an
already-good default and **lowered precision** while raising the
acceleration-kurtosis physics metric (a sign of more spurious link swaps),
even though "Forward/backward agreement" still read 100%.

**In practice:** if your `parameters.yaml` already has a reasonable
`dvxmax`/`dacc` (e.g. carried over from a prior calibration run, or set by
hand from the expected flow speed), run warmup as a **dry run first**, and
compare its suggested `dvxmax`/`dacc` against your current values before
`--write`ing. Warmup is most useful when your current search box is
*clearly* too loose or too tight for the flow (the scenario it was
originally built to fix — see the plan doc's commit `07f1fc1` reference for
the motivating real-world case), not as a blind "always run this" step yet.
A fix for the margin is tracked as Stage 5 part 2, item 1 in the plan doc.

## See also

- [Batch Processing](batch_processing.md) — running the sequence step that
  warmup measures
- [Sequence & Tracking Plugins](plugins.md) — `track.corrective_passes` and
  other `track:` YAML options warmup can tune
- `docs/plans/2026-08-15-tracking-quality-overhaul.md` — the full design
  rationale, what warmup deliberately does *not* do yet (GMM prediction,
  the other tracker engines), and measured benchmark results
