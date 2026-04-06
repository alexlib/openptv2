# Tracking Visualization

The tracking visualization panel provides a way to preview tracking results before running the full tracking process. It displays particle positions, statistics, and allows frame-by-frame navigation.

## Usage

There are two ways to use the tracking visualization:

### 1. Programmatic Usage

```python
from gui.pyptv.tracking_viz_panel import show_tracking_preview, create_tracking_viz_panel

# Option 1: Create and display panel directly
panel = show_tracking_preview(main_gui, num_frames=10)

# Option 2: Create panel first, then display
panel = create_tracking_viz_panel(main_gui, num_frames=5)
panel.configure_traits()
```

### 2. GUI Integration

The tracking visualization panel is integrated into the main GUI. Look for the "Tracking Visualization" panel in the interface.

## Features

- **Run Preview**: Click to run tracking on a small number of frames
- **Frame Navigation**: Navigate through frames using Previous/Next buttons or direct input
- **Statistics Display**:
  - Number of particles detected
  - Number of linked particles
  - Linking ratio (%)
- **2D Scatter Plot**: Visualizes particle positions in the current frame

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| main_gui | MainGUI | required | The main GUI instance |
| num_frames | int | 5 | Number of frames to preview |

## Returns

`TrackingVizPanel`: A TraitsUI panel with:
- `current_frame`: Current frame number (0-indexed)
- `total_frames`: Total number of frames in preview
- `num_particles`: Particles in current frame
- `num_linked`: Linked particles in current frame
- `linking_ratio`: Ratio of linked to total particles

## Example

```python
from gui.pyptv.tracking_viz_panel import show_tracking_preview

# Show tracking preview with 10 frames
panel = show_tracking_preview(main_gui, num_frames=10)

# Access results
print(f"Average particles: {panel.avg_particles}")
print(f"Average linking ratio: {panel.avg_linking_ratio}")
```
