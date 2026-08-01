These four distinct, implementation-ready plans cover the integration of Lie groups and Lie algebra into the tracking and trajectory prediction phases of `openptv2`.

---

# Plan 1: Accelerating "Shake-the-Box" (STB) Shaking Step via Analytical Manifold Jacobians

### 1. Objective
Replace numerical finite-difference calculations of projection gradients during the image-matching (shaking) phase with analytical, closed-form Jacobians evaluated directly on the Lie-group camera model.

### 2. Mathematical formulation
In STB, we adjust a particle's 3D position $\mathbf{X} \in \mathbb{R}^3$ to match the observed image intensity. The optimization gradient is:
$$\mathbf{g} = \frac{\partial E}{\partial \mathbf{X}} = -2 \sum_{p \in \text{pixels}} \left( I_{\text{actual}}(p) - I_{\text{synth}}(p) \right) \frac{\partial I_{\text{synth}}(p)}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \mathbf{X}}$$

Where:
* $\mathbf{x} = [u, v]^T$ is the projected 2D location of the particle on the sensor.
* $\frac{\partial I_{\text{synth}}(p)}{\partial \mathbf{x}}$ is the analytical gradient of the particle's intensity profile (typically modeled as a 2D Gaussian).
* $\frac{\partial \mathbf{x}}{\partial \mathbf{X}}$ is the Jacobian of the 3D-to-2D projection function with respect to the 3D coordinates.

### 3. Implementation steps
1. **Symbolic Jacobian Generation:**
   * Extend the SymForce codegen script to output `project_jacobian_wrt_point(pose, point_world, ...)`.
   * This yields a highly optimized $2 \times 3$ matrix representing $\frac{\partial \mathbf{x}}{\partial \mathbf{X}}$.
2. **Integration into the Shaking Loop:**
   * In the tracker correction loop (which iterates over all active particles), retrieve the pre-computed $SE(3)$ camera pose.
   * Call the generated analytical Jacobian function directly using the particle's predicted 3D position.
3. **Multi-Camera Gradient Accumulation:**
   * For a particle visible in $N$ cameras, evaluate the $2 \times 3$ Jacobian for each camera.
   * Map the individual 2D image-space intensity gradients back to the 3D space to compute the joint update step:
     $$\mathbf{X}_{k+1} = \mathbf{X}_k - \alpha \sum_{c=1}^N \mathbf{J}_c^T \mathbf{g}_{2D, c}$$

### 4. Testing & verification
* **Jacobian Accuracy Test:** Compare the analytical Jacobian matrix $\frac{\partial \mathbf{x}}{\partial \mathbf{X}}$ against a central finite-difference approximation. Ensure relative error is $< 10^{-6}$.
* **Profiling:** Measure the execution time of 10,000 "shaking" updates using numerical gradients versus the analytical Lie-manifold gradients.

---

# Plan 2: Fluid-Centric Trajectory Prediction (Vorticity as $\mathfrak{so}(3)$)

### 1. Objective
Incorporate physical rotational fluid dynamics into the temporal predictor step of the particle tracker by representing local fluid vorticity as an element of the Lie algebra $\mathfrak{so}(3)$.

### 2. Mathematical formulation
The velocity gradient tensor $\mathbf{J}$ at a particle's position is decomposed into symmetric strain $\mathbf{S}$ and anti-symmetric vorticity $\mathbf{\Omega}$:
$$\mathbf{J} = \mathbf{S} + \mathbf{\Omega}$$
The vorticity tensor $\mathbf{\Omega}$ is skew-symmetric and is a member of the Lie algebra $\mathfrak{so}(3)$:
$$\mathbf{\Omega} = [\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix} \in \mathfrak{so}(3)$$
where $\boldsymbol{\omega}$ is the physical vorticity vector. The rotation of the local fluid parcel over a time step $\Delta t$ is computed via the exponential map:
$$R_{\text{fluid}} = \exp([\boldsymbol{\omega}]_\times \Delta t) \in SO(3)$$

### 3. Implementation steps
1. **State Vector Expansion:**
   * Expand the tracker's particle state vector from $[\mathbf{X}, \mathbf{U}, \mathbf{A}]$ (position, velocity, acceleration) to include the local vorticity vector $\boldsymbol{\omega} \in \mathbb{R}^3$.
2. **Manifold Predictor Integration:**
   * Write a Lie Group path-integration step (e.g., Runge-Kutta 2nd or 4th order on manifolds).
   * Update the particle's velocity direction using the exponential map of the estimated local vorticity over $\Delta t$:
     $$\mathbf{U}(t + \Delta t) = \exp([\boldsymbol{\omega}(t)]_\times \Delta t) \cdot \mathbf{U}(t) + \mathbf{A}(t)\Delta t$$
3. **Temporal Vorticity Estimation:**
   * Estimate $\boldsymbol{\omega}$ for each tracked particle by analyzing the relative rotation of its nearest neighbors over consecutive frames.

### 4. Testing & verification
* **Simulation Test:** Propagate particles through a analytical 3D flow field with high shear and rotation (e.g., a Taylor-Green Vortex or Arnold-Beltrami-Childress flow).
* **Comparison:** Quantify the reduction in prediction error when using the Lie-algebraic rotation predictor versus a standard linear Taylor-series predictor.

---

# Plan 3: Track Segment Association and Cluster Tracking via $SE(3)$

### 1. Objective
Resolve tracking ambiguities, path crossings, and temporary occlusions in high-density particle regions by tracking local groups of particles as coherent clusters undergoing $SE(3)$ transformations.

### 2. Mathematical formulation
For a cluster of $N$ neighboring particles, their motion from frame $k$ to frame $k+1$ is modeled as a joint rigid-body transformation $T = (R, \mathbf{t}) \in SE(3)$.
We minimize the alignment error between the particle coordinates in both frames:
$$E(T) = \sum_{i=1}^N w_i \| \mathbf{X}_{i, k+1} - (R \mathbf{X}_{i, k} + \mathbf{t}) \|^2$$
where $w_i$ are weights based on tracking confidence.

### 3. Implementation steps
1. **Dynamic Cluster Identification:**
   * Implement a spatial K-Nearest Neighbors (KNN) or radius search using SciPy's `KDTree` to group nearby tracked particles.
2. **SE(3) Alignment Solver:**
   * Write a closed-form Kabsch-like solver on the $SE(3)$ manifold (using SVD or quaternion representation) to find the optimal relative transformation $T_{rel} \in SE(3)$ between frame $k$ and $k+1$ for each cluster.
   * Implement a RANSAC step to ignore particles that undergo strong non-rigid deformation (outliers).
3. **Track Association Predictor:**
   * For particles whose individual tracks are lost or ambiguous in frame $k+1$, apply the cluster's collective $T_{rel}$ to project their expected positions.
   * Use this $SE(3)$-corrected search window to assign the correct candidate particles.

### 4. Testing & verification
* **High-Density Challenge:** Create a simulation with a high particle-per-pixel (PPP) density ($>0.08$) undergoing a shearing shear-layer flow.
* **Tracking Yield:** Measure the track length distribution and the percentage of tracking "swaps" (where the tracker accidentally jumps to a nearby particle) compared to a standard nearest-neighbor tracker.

---

# Plan 4: 6-DoF Tracking of Anisotropic Particles via Lie Group Extended Kalman Filtering (LG-EKF)

### 1. Objective
Track both the 3D position and 3D orientation of non-spherical tracer particles (rods, ellipsoids, or fibers) using an Extended Kalman Filter designed natively on the $SE(3)$ manifold.

### 2. Mathematical formulation
The state of an anisotropic particle is represented as an element of the Lie group $X \in SE(3)$ along with its linear and angular velocities:
$$\mathbf{x} = \left( X, \mathbf{v}, \boldsymbol{\omega} \right) \in SE(3) \times \mathbb{R}^3 \times \mathbb{R}^3$$
The covariance matrix $\mathbf{P} \in \mathbb{R}^{12 \times 12}$ is maintained in the local flat tangent space (Lie algebra $\mathfrak{se}(3) \times \mathbb{R}^6$). The error state $\tilde{\mathbf{x}}$ is defined as:
$$\tilde{\mathbf{x}} = \begin{bmatrix} \boldsymbol{\delta}\boldsymbol{\theta} \\ \boldsymbol{\delta}\mathbf{x} \\ \boldsymbol{\delta}\mathbf{v} \\ \boldsymbol{\delta}\boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{12}$$

### 3. Implementation steps
1. **Kinematic Propagation on Manifold:**
   * Implement the state propagation step. The orientation component $R \in SO(3)$ is updated using the angular velocity $\boldsymbol{\omega}$ via the exponential map:
     $$R_{k+1} = R_k \cdot \exp(\boldsymbol{\omega}_k \Delta t)$$
2. **Error Covariance Propagation:**
   * Implement the discrete-time error-state transition matrix $\mathbf{F}_k$ using Lie-algebraic derivatives (adjoint representation of $SE(3)$):
     $$\mathbf{P}_{k+1} = \mathbf{F}_k \mathbf{P}_k \mathbf{F}_k^T + \mathbf{Q}_k$$
3. **Measurement Model (Sensor Projection):**
   * Project the major axis of the anisotropic particle (e.g., the endpoints of a rod) onto the camera sensors.
   * Compute the measurement Jacobian with respect to the Lie-algebraic error state $\tilde{\mathbf{x}}$ to perform the EKF update step.

### 4. Testing & verification
* **Consistency Check:** Verify that the estimated covariance $\mathbf{P}$ correctly represents the true state error over a Monte Carlo simulation run (Normalized Innovation Squared - NIS test).
* **Tracking Robustness:** Evaluate orientation tracking accuracy in simulated turbulent flows under varying levels of image noise and partial occlusions.


The articles you provided on **Plücker coordinates** (Karthikmr), **Non-Single Viewpoint (NSVP) Caustics** (Swaminathan et al.), and **Flat Refractive Geometry** (Treibitz et al.) [Treibitz, Swaminathan] provide a cohesive framework for restructuring the mathematics of `openptv2`.

By combining these concepts, we can define a mathematically rigorous, computationally efficient, and physically sound representation for refractive 3D-PTV.

---

### 1. The Core Concept: The Fluid-Centric Plücker Ray Field

As Treibitz et al. and Swaminathan et al. establish, a camera looking through a flat interface into a refractive medium is a **Non-Single Viewpoint (NSVP) system** [Treibitz, Swaminathan]. The chief rays do not intersect at a single point; instead, their envelope forms a **caustic surface** [Treibitz, Swaminathan].

However, inside the fluid (water) domain, these refracted rays are **straight lines**. 

Instead of parameterizing these lines using a traditional point-and-direction form ($P(\lambda) = \mathbf{p}_1 + \lambda(\mathbf{p}_2 - \mathbf{p}_1)$), which Karthikmr notes is computationally cumbersome when transitioning between coordinate frames, we can represent each fluid-domain ray as a homogeneous **Plücker line** [Karthikmr]:
$$\mathbf{L}(u, v) = \begin{bmatrix} \mathbf{v}(u, v) \\ \mathbf{m}(u, v) \end{bmatrix} \in \mathbb{R}^6$$

Where:
* $\mathbf{v} \in \mathbb{R}^3$ is the unit direction of the ray inside the water [Karthikmr].
* $\mathbf{m} = \mathbf{p} \times \mathbf{v} \in \mathbb{R}^3$ is the **moment of the line** (where $\mathbf{p}$ is any point on the ray, such as its point of refraction on the glass interface) [Karthikmr].
* These coordinates naturally satisfy the orthogonality constraint: $\mathbf{v}^T \mathbf{m} = 0$ [Karthikmr].

---

### 2. How Plücker Geometry Simplifies the Tracking Pipeline

Using Plücker coordinates offers three major advantages for the performance of `openptv2` (specifically in `epi.py` and triangulation):

#### A. Near-Instantaneous Epipolar Constraint Checks
In standard PTV, matching particles across cameras requires checking if a candidate point on Camera B lies on the epipolar curve generated by Camera A. 
With Plücker coordinates, the distance $d$ between the rays of Camera A ($\mathbf{L}_1 = \{\mathbf{v}_1 \mid \mathbf{m}_1\}$) and Camera B ($\mathbf{L}_2 = \{\mathbf{v}_2 \mid \mathbf{m}_2\}$) is given by the **reciprocal product** formula from Karthikmr's Section II.1 [Karthikmr]:
$$d = \frac{|\mathbf{v}_1^T \mathbf{m}_2 + \mathbf{v}_2^T \mathbf{m}_1|}{\|\mathbf{v}_1 \times \mathbf{v}_2\|}$$

The numerator, $\mathbf{v}_1^T \mathbf{m}_2 + \mathbf{v}_2^T \mathbf{m}_1$, is a simple dot product. 
* If this value is close to zero, the two rays are coplanar and intersect in 3D space [Karthikmr].
* This allows `epi.py` to evaluate candidate matches across multiple cameras using only **6 multiplications and 5 additions** per ray pair. It completely avoids expensive iterative ray-tracing, standard projections, or SVDs during the candidate-matching phase.

#### B. Unified $SE(3)$ Adjoint Transformations
If the camera moves, or if we perform a volume self-calibration update, we must transform the ray field. As detailed in Section II.2 of Karthikmr's article, transforming a Plücker line by a rigid-body transformation $T = (R, \mathbf{t}) \in SE(3)$ is a linear operation [Karthikmr]:
$$\mathbf{v}' = R \mathbf{v}$$
$$\mathbf{m}' = R \mathbf{m} + \mathbf{t} \times (R \mathbf{v})$$

This is represented as a single $6 \times 6$ block matrix multiplication (the Adjoint representation of $SE(3)$):
$$\begin{bmatrix} \mathbf{v}' \\ \mathbf{m}' \end{bmatrix} = \begin{bmatrix} R & \mathbf{0}_{3\times3} \\ [\mathbf{t}]_\times R & R \end{bmatrix} \begin{bmatrix} \mathbf{v} \\ \mathbf{m} \end{bmatrix}$$
Where $[\mathbf{t}]_\times$ is the skew-symmetric cross-product matrix. This linear transformation is highly parallelizable and can be evaluated on a GPU, as suggested in the paper [Karthikmr].

---

### 3. Python/NumPy Boilerplate: Plücker Ray Field & Fast Epipolar Check

The following code illustrates how to represent the refractive ray field using Plücker coordinates and perform the fast epipolar distance check using NumPy:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R


class PluckerRayField:
    def __init__(self, omega: float, phi: float, kappa: float, translation: np.ndarray):
        """
        Base camera representation. The Plücker rays are defined inside the
        refractive medium (water) where the paths are straight.
        """
        self.R_cam = R.from_euler("xyz", [omega, phi, kappa]).as_matrix()
        self.t_cam = np.array(translation, dtype=float)

        # 6x6 Adjoint Transformation Matrix for Plücker lines (SE3 transformation)
        # Transforming from Local Camera coordinates to World coordinates
        t_cross = np.array(
            [
                [0, -self.t_cam[2], self.t_cam[1]],
                [self.t_cam[2], 0, -self.t_cam[0]],
                [-self.t_cam[1], self.t_cam[0], 0],
            ]
        )

        self.Adjoint_T = np.zeros((6, 6))
        self.Adjoint_T[0:3, 0:3] = self.R_cam
        self.Adjoint_T[3:6, 0:3] = t_cross @ self.R_cam
        self.Adjoint_T[3:6, 3:6] = self.R_cam

    def get_plucker_ray(
        self, u: float, v: float, d: float, n_water: float = 1.333
    ) -> np.ndarray:
        """
        Computes the Plücker coordinates of the ray inside the water
        for a given pixel (u, v), accounting for flat refractive geometry.
        """
        # 1. Local ray direction in air (before interface)
        v_air = np.array([u, v, 1.0])
        v_air /= np.linalg.norm(v_air)

        # 2. Compute the refraction on the flat interface (Snell's Law)
        # For simplicity, assuming interface normal along Z axis in local frame
        sin_theta_air = np.sqrt(v_air[0] ** 2 + v_air[1] ** 2)
        sin_theta_water = sin_theta_air / n_water

        # Local ray direction inside the water
        v_water_local = np.array(
            [
                v_air[0] * (sin_theta_water / (sin_theta_air + 1e-9)),
                v_air[1] * (sin_theta_water / (sin_theta_air + 1e-9)),
                np.sqrt(1.0 - sin_theta_water**2),
            ]
        )

        # 3. Compute point of refraction on the interface (at distance d from camera center)
        # local z = d (location of the flat glass interface)
        scale = d / (v_air[2] + 1e-9)
        p_refraction_local = v_air * scale

        # 4. Form local Plücker coordinates: {v_local | m_local}
        m_water_local = np.cross(p_refraction_local, v_water_local)
        L_local = np.concatenate([v_water_local, m_water_local])

        # 5. Transform Plücker coordinate to World frame using the 6x6 Adjoint
        L_world = self.Adjoint_T @ L_local
        return L_world


def compute_ray_distance_plucker(L1: np.ndarray, L2: np.ndarray) -> float:
    """
    Computes the shortest Euclidean distance between two 3D rays
    using their Plücker representations. Very fast and non-iterative.
    """
    v1, m1 = L1[0:3], L1[3:6]
    v2, m2 = L2[0:3], L2[3:6]

    # Reciprocal product (numerator of distance formula)
    reciprocal_product = np.dot(v1, m2) + np.dot(v2, m1)

    # Cross product of directions (denominator)
    v1_xv2 = np.cross(v1, v2)
    denom = np.linalg.norm(v1_xv2)

    if denom < 1e-9:
        # Parallel rays fallback
        return np.linalg.norm(np.cross(v1, m2 - m1))

    return np.abs(reciprocal_product) / denom


# =====================================================================
# Verification Demonstration
# =====================================================================
if __name__ == "__main__":
    # Define two cameras looking at a common particle inside a water volume
    cam1 = PluckerRayField(
        omega=0.0, phi=0.1, kappa=0.0, translation=[-100.0, 0.0, 500.0]
    )
    cam2 = PluckerRayField(
        omega=0.0, phi=-0.1, kappa=0.0, translation=[100.0, 0.0, 500.0]
    )

    # Compute Plücker rays in water for a matching particle projection
    L1 = cam1.get_plucker_ray(u=0.02, v=0.01, d=40.0)
    L2 = cam2.get_plucker_ray(u=-0.02, v=0.01, d=40.0)

    # Verify the internal orthogonality constraint (v^T * m == 0)
    print(f"Orthogonality check Cam 1: {np.dot(L1[0:3], L1[3:6]):.3e}")

    # Epipolar match check (shortest 3D distance between rays)
    dist = compute_ray_distance_plucker(L1, L2)
    print(f"3D Distance between rays: {dist:.6f} mm")
```

This geometric approach aligns the physical reality of the non-SVP caustic surface [Treibitz, Swaminathan] with a clean, vectorizable algebraic model [Karthikmr]. It provides the exact mathematical framework needed to accelerate the epipolar searches and triangulation steps within `openptv2`.
