# Parameter Management

This tutorial covers reading, editing, and validating OpenPTV YAML parameter
files using the `openptv-params` skill — no GUI required.

## Setup

```bash
cd /path/to/openptv2
PP=skills/openptv-params/scripts/params.py
```

## Dataset Layout

All parameters live in a single YAML file in the dataset root:

```
<dataset>/
  parameters_<name>.yaml    ← all parameters (ptv, sequence, track, …)
```

The script finds the YAML automatically when you pass the dataset directory.

## Workflow

### 1. Validate first

Before running any calibration or tracking, check for known configuration
mistakes:

```bash
uv run python $PP validate <dataset>
```

Typical output for a correctly configured dataset:
```
OK
```

Output when the most common bug is present:
```
ERROR: ptv.mmp_n2=1.33 < mmp_n3=1.46 — looks swapped (n2=glass~1.46, n3=water~1.33)
```

### 2. Inspect the parameters

```bash
# Full YAML
uv run python $PP show <dataset>

# Just one section
uv run python $PP show <dataset> --section ptv
uv run python $PP show <dataset> --section sequence
uv run python $PP show <dataset> --section track
```

### 3. Fix a parameter

```bash
uv run python $PP set <dataset> ptv.mmp_n2 1.46
uv run python $PP set <dataset> ptv.mmp_n3 1.33
```

The original file is backed up to `<file>.parbak` before every write.

Values are auto-coerced: `1` → int, `1.46` → float, `true`/`false` → bool.

### 4. Compare two datasets

```bash
uv run python $PP diff run1/parameters_run1.yaml run2/parameters_run2.yaml
```

Output:
```
- sequence.last: 50
+ sequence.last: 200
- ptv.mmp_n2: 1.33
+ ptv.mmp_n2: 1.46
```

## Common Parameters and Their Meaning

| Key | Description |
|-----|-------------|
| `ptv.mmp_n1` | refractive index of medium 1 (air, should be ~1.0) |
| `ptv.mmp_n2` | refractive index of medium 2 (**glass**, should be ~1.46) |
| `ptv.mmp_n3` | refractive index of medium 3 (water, should be ~1.33) |
| `ptv.imx` / `ptv.imy` | image width / height in pixels |
| `ptv.pix_x` / `ptv.pix_y` | pixel size in mm |
| `sequence.first` / `sequence.last` | frame range to process |
| `cal_ori.fixp_name` | path to 3D calibration body file |

## The Refractive Index Bug

The most common mistake in OpenPTV datasets: `n2` (the **glass** slot) is set
to the water value and `n3` (the **water** slot) is set to the glass value.
This looks correct because both values are present, but the internal ray-trace
puts them in the wrong layer order.

**Wrong:**
```yaml
ptv:
  mmp_n2: 1.33   # ← water index in the glass slot
  mmp_n3: 1.46   # ← glass index in the water slot
```

**Correct:**
```yaml
ptv:
  mmp_n2: 1.46   # glass
  mmp_n3: 1.33   # water
```

Fix in one command:
```bash
uv run python $PP set <dataset> ptv.mmp_n2 1.46
uv run python $PP set <dataset> ptv.mmp_n3 1.33
uv run python $PP validate <dataset>
```

## Restoring a Backup

If you set a wrong value:

```bash
# The backup is <yaml_file>.parbak — copy it back
cp <dataset>/parameters_run1.yaml.parbak <dataset>/parameters_run1.yaml
```
