# Matcher trace (2026-11-23)

One-off analysis scripts behind section 14 of `docs/plans/2026-11-21-tracker-setup.md`. Each takes a benchmark case
name (`uv run python bench/matcher_trace/<script>.py <case>`); paths to the benchmark folder are hard-coded to the
2026-09-29 benchmark. They need the case's `res/run.zarr` (targets and the matcher's own points) and `truth.npz`
(`blob_tid`, `ndet`).

- `gap_by_blob_class.py <case> <tracker...>`: truth points by blob class, found by which tracker (section 13 table).
- `pair_tests.py <case>`: the matcher's epipolar and ratio tests for every camera pair of every true particle,
  split by matched or missing, clean or shared; best consistent clique per particle.
- `pair_flag.py <case>`: the real matcher with `pair_flag` 0 and 1, scored against truth.
- `shared_blob_second_pass.py <case> <mode>`: a second matcher pass over the leftover targets plus blobs treated as
  shared (`oracle` = truth, a number = brightness against the camera median, `xcam:<r>` = brightness against the same
  matcher row's other cameras).
