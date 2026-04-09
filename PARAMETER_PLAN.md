# Parameter Management Plan for OpenPTV-Python

## Overview

This document describes the plan to consolidate parameter handling in OpenPTV-Python, creating a single pipeline from YAML files to both Cython bindings and Python algorithm dataclasses.

**Problem**: Multiple, inconsistent ways to handle parameters across the codebase cause bugs and confusion.

**Goal**: Single I/O (YAML ↔ ParameterManager), then convert to algorithm types in ONE place.

---

## Current State Analysis

### Files Involved

| File | Purpose | Issues |
|------|---------|--------|
| `gui/pyptv/parameter_manager.py` | Read/write YAML | Canonical I/O (keep) |
| `gui/pyptv/experiment.py` | Experiment management | Uses ParameterManager (keep) |
| `gui/pyptv/legacy_parameters.py` | Old .par file handling | Legacy only (keep) |
| `algorithms/parameters.py` | Python dataclasses | Missing key normalizers |
| `gui/pyptv/parameter_defaults.py` | **DOES NOT EXIST** | Need to create |
| `algorithms/parameter_converters.py` | **DOES NOT EXIST** | Need to create |
| `bindings/optv/parameters.pyx` | Cython bindings | Keep as-is |
| `algorithms/parameters_adapter.py` | Legacy adapter | Can potentially simplify |

### YAML Key Variations (The Core Problem)

| Concept | YAML Key #1 | YAML Key #2 | Python Class Field |
|---------|-------------|-------------|----------------|
| Search volume | `criteria` | `volume` | `VolumePar.x_lay`, `z_min_lay` |
| X bounds | `X_lay` | `x_lay` | `x_lay` |
| Z min | `Zmin_lay` | `z_min_lay` | `z_min_lay` |
| Z max | `Zmax_lay` | `z_max_lay` | `z_max_lay` |
| Tracking | `track` | `tracking` | `TrackParTuple.dvxmin`, etc. |
| Angle | `dangle` | `angle` | `dangle` |

### Current Broken Code Example

From `gui/pyptv/pyptv_gui.py` (lines 1469-1501):

```python
# Manual key mapping - WRONG approach
vol_params = params.get("criteria") or params.get("volume")
xmin = vol_params.get("xmin", vol_params.get("X_lay", [0, 100])[0]...)  # 5 variations!

# Hardcoded defaults scattered everywhere - WRONG
vpar.z_max_lay = [zmax, zmax]  # Not clear where 50 comes from
```

### Issues Summary

1. **Hardcoded defaults** sprinkled in multiple files - no single source of truth
2. **Key variations** handled differently in each file - `criteria` vs `volume`, `X_lay` vs `x_lay`
3. **Missing ValueError** - code silently uses wrong defaults, breaking tracking without warning
4. **Duplicate logic** in pyptv_gui.py, tracking_viz_panel.py, tracking_preview.py
5. **No backward compatibility** - changing YAML keys would break existing files

---

## Proposed Solution

### Architecture

```
                    ┌─────────────────────────┐
                    │  YAML File              │
                    │  (parameters_Run1.yaml)│
                    └───────────┬─────────────┘
                                │
                                ▼
                ┌─────────────────────────────────┐
                │  ParameterManager            │
                │  gui/pyptv/parameter_manager │
                │  from_yaml() / to_yaml()    │
                └───────────┬─────────────────┘
                            │
                            ▼ YAML dict (pm.parameters)
                ┌─────────────────────────────────┐
                │  parameter_converters.py      │ ◄── NEW - ONE PLACE
                │  algorithms/parameter_converters│
                └───────────┬─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ ControlPar   │  │ VolumePar   │  │TrackParTuple│
│(algorithms) │  │(algorithms) │  │(algorithms)│
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌──────────────┐
                    │  Tracker    │
                    │  Cython    │
                    └───────────┘
```

### File Structure

#### 1. `gui/pyptv/parameter_defaults.py` (NEW)

SINGLE PLACE for default values. All converters import from here.

**Key principle**: `None` = required (no default), value = optional with default.

#### 2. `algorithms/parameter_converters.py` (NEW)

All converters in one place. Each converter:
1. Checks required params → raises `ValueError` if missing
2. Handles key variations (criteria/volume, X_lay/x_lay)
3. Merges with defaults for optional params
4. Returns clean Python dataclass

---

## Implementation Steps

### Phase 1: Create New Files

1. Create `gui/pyptv/parameter_defaults.py` with all defaults
2. Create `algorithms/parameter_converters.py` with all converters

### Phase 2: Update Code Using Converters

3. Update `gui/pyptv/pyptv_gui.py` - `_tracking_debug_click`
4. Update `gui/pyptv/tracking_viz_panel.py` - TrackingDebugPanel
5. Update any other files with manual parameter mapping

---

## Backward Compatibility

The converters handle these YAML key variations:

| YAML Key | Alternative | Notes |
|---------|-------------|--------|
| `criteria` | `volume` | Search volume section |
| `X_lay` | `x_lay` | X bounds |
| `Zmin_lay` | `z_min_lay` | Z min |
| `Zmax_lay` | `z_max_lay` | Z max |
| `track` | `tracking` | Tracking parameters |
| `dangle` | `angle` | Angle limit |

---

## Error Handling

### Current (BAD):

```python
# Silently uses wrong value - tracked for 2 hours!
vpar.z_max_lay = [vol_params.get("zmax", 50)]  # What is 50??
```

### New (GOOD):

```python
# Fails fast with clear message
if missing:
    raise ValueError(f"Missing required criteria parameters: {missing}")

# Result: 
# ValueError: Missing required criteria parameters: ['X_lay']
```

---

## Summary Table

| Aspect | Before | After |
|--------|--------|-------|
| Default values | Scattered in 5+ files | ONE file (`parameter_defaults.py`) |
| Key variations | Manual in each file | ONE file (`parameter_converters.py`) |
| Missing required | Silent failure | ValueError with message |
| Backward compat | N/A | Automatic |
| Lines to get VolumePar | ~25 | 1 |
| Where to edit defaults | Search everywhere | ONE place |

---

## Related Files

- `gui/pyptv/parameter_manager.py` - YAML I/O (unchanged)
- `gui/pyptv/experiment.py` - Uses ParameterManager (unchanged)
- `gui/pyptv/parameter_defaults.py` - **NEW** - All defaults
- `algorithms/parameter_converters.py` - **NEW** - All converters
- `algorithms/parameters.py` - Dataclass definitions (unchanged)
- `gui/pyptv/pyptv_gui.py` - Update to use converters
- `gui/pyptv/tracking_viz_panel.py` - Update to use converters
- `algorithms/tests/test_*.py` - Add tests

---

*Document created: 2026-04-08*
*Last updated: 2026-04-08*
