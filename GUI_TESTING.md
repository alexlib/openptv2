# GUI Testing Checklist

Since I cannot see your screen, please follow this checklist to test the GUI with both engines.

## Pre-Test Setup

```bash
cd /home/user/Documents/GitHub/openptv2
git status  # Should be on python_integration_into_pyptv branch
```

---

## Test 1: Launch GUI with Python Engine

```bash
export OPENPTV_ENGINE=python
uv run python -m gui.pyptv.pyptv_gui
```

**Expected:** GUI window opens successfully

**If it crashes:** Copy the full error message and share it with me.

---

## Test 2: Load Existing Project (Python Engine)

With GUI open (Python engine):

1. **File → Open Project**
2. Navigate to: `test_data/synthetic/` or `test_data/burgers/`
3. Select `parameters/` directory
4. Click OK

**Expected:** 
- Project loads without errors
- Status bar shows loaded parameters
- Camera count shows correct number

**If errors occur:** Note the error message and which step failed.

---

## Test 3: Run Detection (Python Engine)

1. **Segmentation → Detection**
2. Set frame number (e.g., 10001)
3. Click "Detect"

**Expected:**
- Progress bar appears
- Targets detected and displayed
- Target count shows in status bar

**Check:**
- [ ] Detection runs without errors
- [ ] Targets are visible in image
- [ ] Target count is reasonable (>0)

---

## Test 4: Run Correspondences (Python Engine)

1. **Tracking → Correspondences**
2. Click "Find Correspondences"

**Expected:**
- Correspondences found
- 3D positions calculated
- Points displayed

**Check:**
- [ ] Correspondences run without errors
- [ ] Some matches found (count > 0)

---

## Test 5: Run Tracking (Python Engine)

1. **Tracking → Track Forward**
2. Set frame range (e.g., 10001-10003)
3. Click "Start Tracking"

**Expected:**
- Tracking progress bar
- Trajectories generated
- Output files created in `res/`

**Check:**
- [ ] Tracking runs to completion
- [ ] Files created: `res/ptv_is.*`, `res/rt_is.*`
- [ ] Trajectory count > 0

---

## Test 6: Restart with optv Engine

Close the GUI, then:

```bash
export OPENPTV_ENGINE=optv
uv run python -m gui.pyptv.pyptv_gui
```

**Expected:** GUI launches (optv engine)

**Check:**
- [ ] GUI opens successfully
- [ ] Status bar or title shows current engine (if displayed)

---

## Test 7: Repeat Workflow with optv Engine

Repeat Tests 2-5 with optv engine.

**Note:** Due to a pre-existing optv C extension bug, some operations may crash. If they do:
1. Note which operation crashed
2. We know Python engine works as fallback
3. This is a known optv issue, not our code

---

## Test 8: Compare Results

After running both engines:

```bash
# Check output files
ls -lh test_data/synthetic/res/

# Compare particle counts (should be similar)
wc -l test_data/synthetic/res/rt_is.10001
```

**Expected:**
- Both engines produce output files
- Particle/trajectory counts are similar (within 5%)
- Files have reasonable size (not empty, not huge)

---

## Test 9: Engine Switching

Test that environment variable controls engine:

```bash
# Python engine
OPENPTV_ENGINE=python uv run python -c "import openptv2; print(openptv2.get_engine())"
# Output: python

# optv engine
OPENPTV_ENGINE=optv uv run python -c "import openptv2; print(openptv2.get_engine())"
# Output: optv

# Auto-detect (defaults to optv if available)
uv run python -c "import openptv2; print(openptv2.get_engine())"
# Output: optv (or python if optv unavailable)
```

---

## Test 10: Notebooks (Optional)

If you use Jupyter notebooks:

```bash
export OPENPTV_ENGINE=python
uv run jupyter notebook gui/notebooks/
```

Open any notebook and try importing:
```python
from openptv2 import Calibration, Tracker
import openptv2
print(openptv2.get_engine())  # Should show 'python'
```

---

## Reporting Issues

If anything fails, please provide:

1. **Which test failed** (Test 1, 2, 3, etc.)
2. **Engine used** (python or optv)
3. **Full error message** (copy entire traceback)
4. **What you were doing** (clicked button X, loaded file Y)
5. **Screenshot** if helpful (describe what you see)

---

## Success Criteria

- [x] Python engine: GUI launches
- [ ] Python engine: Can load project
- [ ] Python engine: Detection works
- [ ] Python engine: Correspondences work
- [ ] Python engine: Tracking works
- [ ] optv engine: GUI launches
- [ ] optv engine: Basic operations work (or known crash documented)
- [ ] Output files created and non-empty
- [ ] No import errors or missing modules

---

## Quick Smoke Test (5 min)

Minimum viable test:

```bash
# Test Python engine
export OPENPTV_ENGINE=python
uv run python -m gui.pyptv.pyptv_gui
# → GUI opens, try loading a project and running detection
# → Note any errors

# Test optv engine
export OPENPTV_ENGINE=optv
uv run python -m gui.pyptv.pyptv_gui  
# → GUI opens, try same project
# → Note any errors
```

**Report back:** "Python engine works" / "optv engine works" / "Both crash with error: ..."

---

## Notes

- **Cannot see screen:** I can't see what's happening, so detailed error messages help
- **Screenshots:** If you paste error text, that's better than screenshots (I can search/copy)
- **Pre-existing bugs:** optv C extension has known issues, Python engine is our fallback
- **Test data:** Use `test_data/synthetic/` or `test_data/burgers/` for testing

Ready to test? Start with the Quick Smoke Test and let me know how it goes!
