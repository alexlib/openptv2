# Glossary

Words are in the order a beginner meets them.

| Word | Plain meaning |
|---|---|
| **Frame** | One moment in time: one image from each camera. Numbered (here 10000, 10001, ...). |
| **Pixel** | One dot of an image. Positions are given to a fraction of a pixel. |
| **Target** | A bright spot found in one image: a candidate particle seen by one camera. Has a centre (x, y) in pixels and a size. |
| **Detection** | Finding the targets in an image: threshold, then group neighbouring bright pixels. |
| **Calibration** | The files that say where each camera is, which way it looks, and how its lens and window bend light. Two files per camera (`.ori`, `.addpar`). |
| **Exterior / interior calibration** | Exterior: the camera's position and angles. Interior: pinhole-to-sensor distance `cc`, principal point, lens distortion. |
| **Principal distance, `cc`** | The distance from the pinhole to the sensor, mm. Close to the focal length when the lens is focused at infinity; larger when focused near. |
| **Lens distortion** | Straight lines in the world appear slightly curved in the image. Described by a few small numbers (`k1`, `k2`, `p1`, `p2`). |
| **Multimedia / refraction** | Light bends where it passes from air into glass and from glass into water. Needs the indices (1.0, 1.46, 1.33), the glass thickness and the window position. |
| **Glass vector** | The arrow from the origin of the lab coordinates to the window, perpendicular to it. Its sign tells which side the window is on. |
| **Blob** | The group of bright pixels of one detected spot. Two particles that touch in the image make one blob. |
| **High-pass filter** | Removes the slowly changing background of an image, so that only small bright spots stay. The program applies it before the threshold. |
| **`run.zarr`** | The file (a folder) in `res/` where the program stores targets, matches and trajectories. |
| **Sealed** | The trajectories were written to `run.zarr` and closed. |
| **`--mode sequence / tracking / both`** | `sequence`: detect and match, frame by frame. `tracking`: link the matched points into trajectories. `both`: the two, one after the other. |
| **`v_max`, `confirm_tol`** | The largest step of a particle between two frames that the tracker accepts, and a second check of each link. See the warning on page 6. |
| **Held-out** | Frames that the fit did not use. A fit is judged on them, as an exam is judged on questions the student has not seen. |
| **Yield** | How many 4-camera points the program finds. A better calibration finds more. |
| **Footprint** | One pixel, expressed as a length in the volume (here 95 µm). A jump of 0.3 px footprint is 0.3 pixels. |
| **Seeding** | The particles you put into the fluid: how many and how big. |
| **Epipolar line** | The image in camera 2 of the line of sight of one spot in camera 1. The same particle must lie on it. |
| **`eps0`** | How far from its epipolar line a spot may be and still count as a match (mm on the sensor). |
| **Correspondence / match** | Targets in different cameras that belong to the same particle. |
| **Triangulation** | Computing the 3D point from the lines of sight of its matching targets. |
| **Quadruplet, triplet, pair** | A match seen by 4, 3, or 2 cameras. More cameras means a more accurate and more reliable point. |
| **Ghost** | A false match: spots from different particles that happen to line up. |
| **Trajectory (track)** | The 3D path of one particle over frames. |
| **Link** | The connection of a particle in one frame with the same particle in the next. |
| **Reprojection error** | Project a known 3D point back into the image and measure the miss, in pixels. Small does not prove that the calibration is right. |
| **Ray convergence (RCM)** | How closely the lines of sight of the cameras meet at one point, in mm. Sees errors between cameras that the reprojection error does not. |
| **Camera-switching jump** | A step in a track's position when the set of cameras that saw the particle changes. A sign that the cameras disagree. |
| **Self-calibration** | Improving the calibration using the tracer particles themselves, with no plate. |
| **Measurement volume** | The box in lab coordinates where particles are accepted (`X_lay`, `Zmin_lay`, `Zmax_lay`). |
