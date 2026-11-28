# Changelog

All notable user-facing changes to **OpenPTV2**, in plain language.

## Unreleased

### Camera-subset alignment (opt-in)

* New option `track.align_4cam` (`--align-4cam`, GUI: Tracking Parameters → Pipeline), default **off**, for every tracker. It removes the sideways jump between the 3-camera and the 4-camera position of a particle that a small calibration error causes. 4-camera positions are not changed. See [Camera-subset alignment](docs/camera-subset-alignment.md).
* The aligned positions are stored in a separate array (`correspondences_aligned/`); the measured positions are never overwritten. `RunStore.read_correspondences(frame, raw=True)` returns the measured ones.
* Two options that the same work added, both off by default: `refine_merged(align_4cam=...)` and `log_brightness_from_arrays(mask_shared=...)`.
