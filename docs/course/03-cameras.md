# 3. Cameras and calibration: how a spot in an image becomes a line in space

**Time:** 20 minutes. **You need:** the repository folder; pages 1 and 2.

## What this step does

An image gives you a spot at pixel (x, y). To turn it into a place in the laboratory, you must know four things about the
camera: where it is, which way it looks, how its lens maps directions to pixels, and how the tank window bends the light.
This knowledge is the **calibration**. It is two small text files per camera.

Everything later depends on it: matching, 3D positions and tracks. A calibration that is a little wrong makes everything a
little wrong, and nothing warns you.

## The idea in four points

1. A camera is a **pinhole**. A spot in the image is the picture of a point somewhere on one straight line through the pinhole.
2. Where the pinhole is (x, y, z in mm) and which way the camera looks (three angles) are the **exterior** calibration.
3. The **interior** calibration has three parts: the distance from pinhole to sensor (**cc**, here 70 mm), the **principal
   point** (where the optical axis meets the sensor), and the **lens distortion** (the lens bends straight lines a little).
4. If the camera looks through a **window** into water, the light bends at the glass (Snell's law). The line of sight is not
   straight any more. The program corrects this with the refractive indices, the glass thickness and the position of the window
   (the **glass vector**). The indices and thickness are in the YAML (`mmp_n1`, `mmp_n2`, `mmp_n3`, `mmp_d`).

## Where the cameras are

![Top view of the four cameras](figures/camera_layout.png)

*The example rig: two cameras on each side of a water cell with flat windows (blue). The red box is the measurement volume.
The origin of the laboratory coordinates is in the middle of the cell.*

## The files

Open `my_run/cal/cam1.tif.ori`:

```text
80.40705424 17.25785483 -570.09601905      position of the pinhole: x, y, z (mm)
    0.01531878  2.97855729  -0.01723759      the three angles (radians) of the look direction
    -0.9865926 -0.0170082  0.1623141       the same orientation as a 3 x 3 rotation matrix
    -0.0147487  0.9997770  0.0151150
    -0.1625349  0.0125185 -0.9866234
      0.0002  -0.0000                      principal point offset xh, yh (mm)
     70.0000                               cc: pinhole to sensor distance (mm)
       0.0  0.0  -125.0                    glass vector: where the window is (mm)
```

and `cam1.tif.addpar`. It holds the lens terms: `k1 k2 k3` (radial distortion) and `p1 p2` (decentering). Then come a scale
and a shear, which are `1` and `0` for a digital camera. So `0 0 0 0 0 1 0` means "no distortion".

The **glass vector** is the arrow from the origin to the window, at right angles to the window. Its length (125 mm here) is
the distance from the middle of the cell to the water side of the window.

**Pixel positions.** The origin (0, 0) is the top-left corner of the image. x grows to the right and y grows **downward**, as in
every image viewer. A position of 246.6 in y, in an image that is 256 pixels high, is therefore close to the bottom edge.

## Do it

=== "Command line"

    ```bash
    python -c "
    import numpy as np
    from openptv2.recorrespond import load_parameters
    from openptv2.loco import project_pixels

    cpar, vpar, cals, params = load_parameters('my_run/parameters_Run1.yaml')
    for i, c in enumerate(cals, 1):
        x, y, z = c.get_pos()
        print(f'camera {i}: position ({x:8.1f}, {y:7.1f}, {z:8.1f}) mm,  cc = {c.int_par.cc:.1f} mm,  glass vector {np.round(c.get_glass_vec(), 1)}')
    point = np.array([[0.0, 0.0, 0.0]])          # the middle of the measurement volume
    print('pixel of the point (0, 0, 0) in each camera:')
    for i, c in enumerate(cals, 1):
        u, v = project_pixels(point, c, cpar)[0]
        print(f'  camera {i}: x = {u:6.1f}, y = {v:6.1f}')
    "
    ```

=== "Graphical interface"

    The window shows the images, not the calibration numbers. To *see* the cameras in 3D, open the camera notebook, which reads
    the same `.ori` files from the YAML:

    ```bash
    marimo run src/openptv2/gui/visualize_cameras_nb.py
    ```

    (Calibrating, rather than reading, is **Calibration → Create calibration** in the window; see the
    [recipes](../recipes/index.md).)

## What you should see

```text
camera 1: position (    80.4,    17.3,   -570.1) mm,  cc =  70.0 mm,  glass vector [   0.    0. -125.]
camera 2: position (  -116.2,    29.5,   -578.9) mm,  cc =  70.0 mm,  glass vector [   0.    0. -125.]
camera 3: position (  -111.7,    70.6,    585.0) mm,  cc =  70.0 mm,  glass vector [   0.   0. 125.]
camera 4: position (   124.6,    67.0,    573.7) mm,  cc =  70.0 mm,  glass vector [   0.   0. 125.]
pixel of the point (0, 0, 0) in each camera:
  camera 1: x =   45.8, y =  226.5
  camera 2: x =   93.7, y =  224.7
  camera 3: x =  123.1, y =  246.6
  camera 4: x =  191.7, y =  165.2
```

## Check that it worked

Two quick checks catch most wrong calibrations:

1. **The point in the middle of the volume must land inside every image.** Here that means 0 to 256 in x and in y. If it lands
   outside or on an edge, a camera file does not belong to this experiment, or its units are wrong.
2. **Cameras on opposite sides need glass vectors that point in opposite directions.** Here z = -125 and z = +125. Each one
   points from the middle of the cell to its own window.

A stronger check asks whether the cameras agree with each other on real particles. That comes in
[page 8](08-calibration.md). A calibration can have a small reprojection error and still be wrong between cameras.

## See what the window does to a ray

A notebook lets you move a particle and change the glass thickness, the water index and the window tilt. It draws the bent
light ray. It is the quickest way to see why the window matters:
[Ray tracing through the window](../ray-tracing-vs-radial-shift.md)
(`marimo edit docs/notebooks/ray_tracing_multimedia.py`).

## If it did not work

| You see | Cause | Do |
|---|---|---|
| the point lands outside the image | wrong camera file, wrong units (mm or m), or a calibration of another setup | measure the camera position with a tape and compare |
| the `k` terms are large (more than 0.01) | a bad fit of the lens terms | set `.addpar` to zeros and calibrate again with more plate positions |
| the glass vector has length 0 | there is no window (the cameras look straight into the fluid) | then `mmp_n2`, `mmp_n3`, `mmp_d` should be 1, 1, 0 |

[Next: 4. Find the particles →](04-detect.md){ .md-button .md-button--primary }
