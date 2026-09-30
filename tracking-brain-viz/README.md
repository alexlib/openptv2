# Tracking Brain Viz — see how the tracker thinks, in image space

Rarely-used debug tool. Rebuilds the old GUI flow
(`Tracking → Debugging with display`, see
`docs/tutorials/tracking_debug_visualization.md`):
click a tracer → zoom window around it over 3–4 frames → candidates →
the tracker's choice.

Standalone marimo project (no Qt/chaco/TraitsUI):

```bash
uv run marimo run tracking-brain-viz/brain_viz.py
```

## Data contract

* **Overlays** come from zarr: `targets/cam_{c}/frame_{f:06d}}`
  `(N,8)` = `[pnr, x, y, n, nx, ny, sumg, tnr]`,
  `correspondences/frame_*` = `(N, 3+C)` 3D pos + per-cam target ids,
  `linkage/ptv_is/frame_*` = `prev/next/pos` (row ids are per-frame array rows).
* **Pixels** come from disk (raw images are NOT in zarr):
  `img_dir/cam{cam}.{frame}` (extensionless TIFF in `test_data/*`).
* Defaults point at `test_data/test_cavity/run.zarr` + `test_data/test_cavity/img`.

## Layers (all toggleable)

`image | target ids | linkage arrows | gate + candidates | chosen link`

View 1 = full frame, one camera, selected particle circled.
View 2 = zoom strip `t … t+3`: crop around the tracer (followed through
linkage `next` pointers), pixel gate circle, candidates numbered,
winner green / losers red.
View 3 = decision table: per candidate `dx, dy, dist_px, chosen`.

Note: the image-space gate here is a pixel-radius *approximation* of the
tracker's 3D gate — it shows *what the tracker had to choose between*,
not the tracker's internal search volume. Pair with
`notebooks/two_phase_link_lab.py` (synthetic, exact 3D gate + cost matrix)
when a verdict needs the algorithm-level "why".
