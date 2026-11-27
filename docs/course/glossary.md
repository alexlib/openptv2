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
| **Glass vector** | The arrow from the origin of the lab coordinates to the window, perpendicular to it. |
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
