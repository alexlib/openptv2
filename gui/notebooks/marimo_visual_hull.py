import marimo as mo
import numpy as np
import plotly.graph_objects as go
from openptv_python.calibration import Calibration
from openptv_python.trafo import img_coord
from openptv_python.parameters import ControlParams

def find_feasible_point(calibrations):
    """
    Finds the point that is closest to the optical axes of all cameras.
    """
    origins = []
    dirs = []
    
    for cal in calibrations:
        # Camera position (C)
        ext = cal.get_external()
        R = np.array(ext.get_rotation_matrix())
        C = -R.T @ np.array(ext.get_translation())
        
        # Optical axis direction (pointing from camera through center of sensor)
        # In OpenPTV, the z-axis of the camera coordinate system is the optical axis
        D = R.T @ np.array([0, 0, 1]) 
        
        origins.append(C)
        dirs.append(D / np.linalg.norm(D))
        
    # Solve least squares for intersection of lines
    # Formula: minimize sum||(P - Oi) x Di||^2
    SXX, SXY, SXZ = 0, 0, 0
    SYY, SYZ, SZZ = 0, 0, 0
    RX, RY, RZ = 0, 0, 0
    
    for O, D in zip(origins, dirs):
        M = np.eye(3) - np.outer(D, D)
        SXX += M[0,0]; SXY += M[0,1]; SXZ += M[0,2]
        SYY += M[1,1]; SYZ += M[1,2]; SZZ += M[2,2]
        RX += M[0,0]*O[0] + M[0,1]*O[1] + M[0,2]*O[2]
        RY += M[1,0]*O[0] + M[1,1]*O[1] + M[1,2]*O[2]
        RZ += M[2,0]*O[0] + M[2,1]*O[1] + M[2,2]*O[2]
        
    A = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])
    B = np.array([RX, RY, RZ])
    return np.linalg.solve(A, B)

def get_overlapping_voxels(calibrations, cparams, center, side_length=100, res=20):
    """
    Samples a grid around the center and checks visibility in all cameras.
    """
    x = np.linspace(center[0]-side_length, center[0]+side_length, res)
    y = np.linspace(center[1]-side_length, center[1]+side_length, res)
    z = np.linspace(center[2]-side_length, center[2]+side_length, res)
    
    X, Y, Z = np.meshgrid(x, y, z)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    w, h = cparams.get_image_size()
    mask = np.ones(len(pts), dtype=bool)
    
    for cal in calibrations:
        # Project 3D points to 2D pixels for this camera
        for i, pt in enumerate(pts):
            if not mask[i]: continue
            
            # OpenPTV img_coord returns the sensor position in mm or pixels
            # Depending on your version, ensure this matches your control_par
            pos_2d = img_coord(pt, cal, cparams)
            
            # Check if within sensor bounds (assuming pos_2d is in pixels)
            if not (0 <= pos_2d[0] <= w and 0 <= pos_2d[1] <= h):
                mask[i] = False
                
    return pts[mask]

# --- Marimo App ---

@mo.md
def title():
    return "# OpenPTV Volume Viewer"

@mo.run
def app():
    # 1. Setup Mock/Real Data
    # cparams = ControlParams().from_file("control.par")
    # calibrations = [Calibration().from_file(...) for i in range(4)]
    
    # 2. Widgets
    side = mo.ui.slider(10, 500, value=100, label="Box Size (mm)")
    res = mo.ui.slider(10, 50, value=25, label="Voxel Resolution")
    
    # 3. Calculation
    # center = find_feasible_point(calibrations)
    # valid_pts = get_overlapping_voxels(calibrations, cparams, center, side.value, res.value)
    
    # 4. Visualization
    fig = go.Figure()
    
    # Add the voxels as a scatter plot
    # fig.add_trace(go.Scatter3d(
    #    x=valid_pts[:,0], y=valid_pts[:,1], z=valid_pts[:,2],
    #    mode='markers',
    #    marker=dict(size=4, color=valid_pts[:,2], colorscale='Viridis', opacity=0.6)
    # ))
    
    return mo.vstack([side, res, mo.ui.plotly(fig)])

title()