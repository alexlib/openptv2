# openptv-params

## Overview
Inspect and mutate OpenPTV YAML parameter files without opening the GUI.
Catches common configuration mistakes (swapped refractive indices, bad image
size, bad sequence range) before they cause silent failures downstream.

## Dependencies
- openptv2 checkout with `uv` venv (the script itself uses only stdlib + PyYAML)
- Run with `uv run python` from the openptv2 checkout

## Quick Reference
```
PP=skills/openptv-params/scripts/params.py
uv run python $PP show     <dataset> [--section ptv]
uv run python $PP set      <dataset> section.field value
uv run python $PP validate <dataset>
uv run python $PP diff     <yaml1> <yaml2>
```

## When to Use Each Subcommand

### show — read parameters
```
uv run python $PP show <dataset>               # full YAML
uv run python $PP show <dataset> --section ptv # just the ptv section
uv run python $PP show <dataset> --section track
```

### set — mutate a field
```
uv run python $PP set <dataset> ptv.mmp_n2 1.46
uv run python $PP set <dataset> ptv.mmp_n3 1.33
uv run python $PP set <dataset> sequence.first 1
uv run python $PP set <dataset> sequence.last 100
```
`set` backs up the YAML to `<file>.parbak` before writing.
Values are auto-coerced: `1` → int, `1.46` → float, `true`/`false` → bool, else str.

### validate — sanity-check before running
```
uv run python $PP validate <dataset>
```
Checks:
- `ptv.mmp_n1` ≈ 1.0 (air)
- `ptv.mmp_n2` > `ptv.mmp_n3` (glass slot > water slot); if swapped → ERROR
- `ptv.imx`, `ptv.imy`, `ptv.pix_x`, `ptv.pix_y` all > 0
- `sequence.first` ≤ `sequence.last`
- `cal_ori.fixp_name` file exists on disk
- `track.dvxmin/dvymin/dvzmin` are negative, `dvxmax/dvymax/dvzmax` are positive

### diff — compare two datasets
```
uv run python $PP diff <dataset1>/parameters_*.yaml <dataset2>/parameters_*.yaml
```
Prints only changed keys in unified diff style.

## Most Common Fix: Refractive Index Swap
The most frequently seen bug in OpenPTV datasets:
```yaml
# WRONG — swapped
ptv:
  mmp_n2: 1.33   # water in the glass slot
  mmp_n3: 1.46   # glass in the water slot

# CORRECT
ptv:
  mmp_n2: 1.46   # glass (~1.46)
  mmp_n3: 1.33   # water (~1.33)
```
Fix:
```
uv run python $PP set <dataset> ptv.mmp_n2 1.46
uv run python $PP set <dataset> ptv.mmp_n3 1.33
uv run python $PP validate <dataset>   # should show OK
```

## Workflow: Checking a New Dataset
1. `uv run python $PP validate <dataset>` — fix any ERRORs first
2. `uv run python $PP show <dataset> --section sequence` — confirm frame range
3. `uv run python $PP show <dataset> --section ptv` — confirm image size and pixel size

## Common Mistakes
- **Running on a directory that has no YAML** — the script globs `parameters_*.yaml` then `*.yaml`; make sure you're pointing at the dataset root (the directory containing the parameters file)
- **Key format** — always `section.field` (e.g. `ptv.mmp_n2`), not just `mmp_n2`
- **Values don't need quotes** — `uv run python $PP set <dataset> ptv.mmp_n2 1.46` not `"1.46"`
