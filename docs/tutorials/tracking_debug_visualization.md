# Tracking Debug Visualization Tutorial

> **This panel is currently broken** -- clicking a particle raises an
> `AttributeError` (`_compute_search_volumes` is not defined), and the
> triangulated 3D position it would show is a hardcoded stub even where it
> doesn't crash. Use the
> [trackcorr Candidate Viewer](trackcorr_candidate_viewer.md) marimo
> notebook instead -- it reconstructs the same real candidate search from
> the actual tracking kernel, verified against real tracking runs. This page
> is kept for reference until the GUI panel is fixed or removed.

## Overview

The tracking debug visualization allows you to click on any detected particle and see:
- Search volume boundaries for future frames (green/yellow/orange rectangles)
- Candidate particles in the next frame (color-coded by distance)
- Epipolar lines connecting to other cameras (cyan)

## Required Prerequisites

Before using the debug visualization, you need to have:
1. **Detected particles** - Run detection to have particles in the frames
2. **Calibrations loaded** - Run Init to load calibration files into memory

## Step-by-Step Usage

### Step 1: Open PyPTV and Load Experiment

1. Launch PyPTV (`openptv2-gui` or `pyptv`)
2. Open your experiment YAML file

### Step 2: Initialize System (Required)

Go to menu: **Start → Init / Reload**

This step is critical because it:
- Loads calibration files into `mainGui.cals`
- Creates parameter objects (cpar, vpar, tpar, etc.)
- Without this, the debug visualization cannot work

Expected console output:
```
Read all the parameters and calibrations successfully
```

### Step 3: Detect Particles (Optional but recommended)

Go to menu: **Preprocess → Image coord**

This finds particles in your images. You should see blue crosses on the camera views.

### Step 4: Enable Debug Mode

Go to menu: **Tracking → Debugging with display**

You will see in the console:
```
Starting tracking debug mode
Debug mode ON - click on particles in camera views to visualize search volumes
Use 'Debugging with display' again to turn off
```

### Step 5: Click on a Particle

**Left-click** on any particle (blue cross) in any camera view.

The debug visualization only responds to **left-clicks**, not right-clicks. 

When you click successfully, you should see:

1. **Console output** with detailed information:
```
[DEBUG] _tracking_debug_click called
[DEBUG] cals available: 4
[DEBUG] Click at camera 0: (440.0, 588.0)
[DEBUG] ptv_params: True
[DEBUG] track_params: True
[DEBUG] Creating tracker...
[DEBUG] Running tracker steps...
[DEBUG] Step 0: True
...
=== Selected particle 5 at (440.0, 588.0) in camera 0 ===
3D position: (12.34, 45.67, 100.00)
Search volumes drawn for frames t+1 (green), t+2 (yellow), t+3 (orange)
Epipolar lines (cyan) from particle to other cameras
```

2. **On camera views** - You should see:
   - **Green rectangle** on all cameras = search volume for frame t+1
   - **Yellow rectangle** on all cameras = search volume for frame t+2  
   - **Orange rectangle** on all cameras = search volume for frame t+3
   - **Cyan epipolar lines** from clicked particle to other cameras
   - **Colored crosses** in frame t+1 showing candidates:
     - Green = close to predicted position
     - Yellow = within acceptable range
     - Red = outside range

### Step 6: Disable Debug Mode

Go to menu: **Tracking → Debugging with display** again

Console output:
```
Debug mode OFF
```

## What Can Go Wrong

### "No calibrations available. Run Init first."

**Cause**: You haven't run Start → Init / Reload

**Solution**: Go to **Start → Init / Reload** and try again

### "No particle found near (x, y)"

**Cause**: Clicked too far from any detected particle

**Solution**: Click directly on a blue cross (detected particle)

### "Not enough frames in buffer"

**Cause**: Not enough frames in the sequence

**Solution**: Ensure your sequence has at least 2 frames

### Nothing happens when clicking

**Cause**: Debug mode not properly enabled

**Solution**: 
1. Check console for "Debug mode ON" message
2. Make sure you clicked on a camera view (not the parameter panel)

## Understanding the Visualization

### Search Volume Rectangles

The tracker predicts where a particle will be in future frames based on:
- Current position
- Velocity (change from previous frame)
- Parameter limits (dvxmin, dvxmax, etc.)

The rectangles show the search area:
- **Green (t+1)**: Where to look in the next frame
- **Yellow (t+2)**: Where to look 2 frames ahead
- **Orange (t+3)**: Where to look 3 frames ahead

The rectangle grows with frame offset because uncertainty increases.

### Candidate Particles

In frame t+1, particles in the search area are shown as crosses:
- **Green**: Very close to predicted position (high probability of being the same particle)
- **Yellow**: Within acceptable range (could be the same particle)
- **Red**: Far from predicted position (low probability, but still in search area)

### Epipolar Lines

Cyan lines show where the clicked particle should appear in other cameras based on the calibration geometry. This helps verify correspondences.

## Summary

| Action | Menu Path | Purpose |
|--------|-----------|---------|
| Init | Start → Init / Reload | Load calibrations (REQUIRED) |
| Detect | Preprocess → Image coord | Find particles (optional but recommended) |
| Enable Debug | Tracking → Debugging with display | Turn on debug mode |
| Visualize | Left-click on particle | See search volumes and candidates |
| Disable Debug | Tracking → Debugging with display | Turn off debug mode |