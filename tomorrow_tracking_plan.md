# Tomorrow Plan: Tracking Slowdown Investigation

## Goal
Continue the Python tracking performance work by validating the latest `searchquader` allocation-reduction patch, measuring whether slowdown follows local target density, and then narrowing the next optimization target if needed.

## Phase 1: Validate the Current Patch
- [ ] Run the single-step tracking stage test with debug timing enabled.
- [ ] Confirm the refactored `searchquader` still produces links and does not regress tracking output.
- [ ] Record whether the patch reduced `searchquader` time or total step time.

**Verification**
- `OPENPTV_TRACK_DEBUG=1 uv run pytest algorithms/tests/integration/test_batch_stages_legacy.py -vv -s -k "test_single_step_produces_links" --tb=short`
- Confirm the test passes and the debug output still shows non-empty links.

## Phase 2: Correlate Slowdowns With Density
- [ ] Compare `camera_candidates` counts against `searchquader` and total timings.
- [ ] Identify whether slow spikes are tied to local particle/target density, not the multimedia lookup itself.
- [ ] Note any frames or particles where timings jump sharply.

**Verification**
- Inspect debug lines containing `searchquader`, `sort`, `camera_times`, and `camera_candidates`.
- Capture the highest-latency examples and their candidate counts.

## Phase 3: Isolate the Next Hotspot
- [ ] If `searchquader` is still expensive, inspect remaining allocation or projection overhead in `algorithms/track.py`.
- [ ] If sorting dominates, target `sort_candidates_by_freq` next.
- [ ] If candidate search dominates, measure `candsearch_in_pix` / `register_closest_neighbs` more directly.

**Verification**
- Use a targeted test or microbenchmark to compare the slowest phase before and after any change.
- Keep link-preservation tests green after every change.

## Phase 4: Preserve Correctness
- [ ] Re-run the link-preservation batch tests after any performance change.
- [ ] Make sure no optimization silently drops candidates or changes linkage behavior.

**Verification**
- `uv run pytest algorithms/tests/test_batch.py::TestPythonBatch::test_sequence_produces_rt_is -v --tb=short -s`
- `uv run pytest algorithms/tests/test_batch_stages.py -v --tb=short -k "Stage7" -s`

## Risks and Notes
- The slowdown appears data-dependent, so a single fast run may hide the problem.
- The current evidence suggests repeated search-volume construction and sorting are more important than the multimedia projection itself.
- Freshly generated outputs should be trusted more than stale checked-in `_targets` data when checking link correctness.
