# Next Steps - Debug Visualization Feature

**Last updated**: 2026-04-08

---

## Immediate Priority

### 1. Debug Why Visualization Doesn't Render

**Problem**: Click handler works (prints to console) but no visualization appears on camera views.

**Investigation needed**:
- [ ] Check if `_tracking_debug_click` correctly calls visualization drawing methods
- [ ] Verify canvas update is triggered after drawing
- [ ] Check if `self._tracking_viz_panel` exists and is properly initialized
- [ ] Verify the drawing methods in tracking_viz_panel.py are being called
- [ ] Check for any exceptions being silently caught

**Location**: `gui/pyptv/pyptv_gui.py` - `_tracking_debug_click` method

### 2. Test Left-Click vs Right-Click Behavior

**Problem**: May only work with right-click in some cases

**Investigation needed**:
- [ ] Verify which mouse button event triggers `_tracking_debug_click`
- [ ] Check if left-click is bound to same handler
- [ ] Test both buttons to see which works

---

## Secondary Tasks

### 3. Add More Visualization Elements

Once basic rendering works:
- [ ] Draw search volume rectangles for frames t+1, t+2, t+3 (green/yellow/orange)
- [ ] Draw epipolar lines (cyan) connecting to other cameras
- [ ] Show candidate particles in next frame color-coded by distance

### 4. Console Statistics

- [ ] Print detailed statistics to console about:
  - Number of candidates found
  - Distance to each candidate
  - Which candidate was selected

---

## Files to Investigate

| File | Purpose |
|------|---------|
| `gui/pyptv/pyptv_gui.py` | Main GUI, `_tracking_debug_click` method |
| `gui/pyptv/tracking_viz_panel.py` | Visualization panel, drawing methods |
| `gui/pyptv/tracking_debug_utils.py` | Helper functions for search volume computation |
| `gui/pyptv/camwidget.py` | Camera view widget, mouse event handling |

---

## Notes

- User must run "Start → Init / Reload" first to load calibrations into memory
- Parameters already in memory (from experiment YAML) should be used instead of defaults
- VolumePar uses `x_lay`, `z_min_lay`, `z_max_lay` attributes
