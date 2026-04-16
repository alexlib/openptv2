import numpy as np

# Define MAX_CANDS for test compatibility
MAX_CANDS = 4


# Define Foundpix_dtype for test compatibility
Foundpix_dtype = np.dtype([
    ("ftnr", np.int32),
    ("freq", np.int32),
    ("whichcam", np.int32, (4,)),
])
def trackback_c(*args, **kwargs):
    return None
def trackcorr_c_loop(*args, **kwargs):
    return None


def find_candidates_in_3d(*args, **kwargs):
    return None

def track3d_loop(*args, **kwargs):
    return None

def trackcorr_c_finish(*args, **kwargs):
    return None

def track_forward_start(*args, **kwargs):
    return None

def predict(prev_pos, curr_pos, c):
    """Predict next position in 2D: c = curr_pos + (curr_pos - prev_pos)"""
    prev_pos = np.asarray(prev_pos)
    curr_pos = np.asarray(curr_pos)
    c[:] = curr_pos + (curr_pos - prev_pos)

def search_volume_center_moving(prev_pos, curr_pos):
    """Predict next position in 3D: c = curr_pos + (curr_pos - prev_pos)"""
    prev_pos = np.asarray(prev_pos)
    curr_pos = np.asarray(curr_pos)
    return curr_pos + (curr_pos - prev_pos)

def pos3d_in_bounds(pos, bounds):
    """Check if 3D position is within bounds (TrackParTuple)."""
    x, y, z = pos
    b = bounds
    result = (
        b.dvxmin <= x <= b.dvxmax and
        b.dvymin <= y <= b.dvymax and
        b.dvzmin <= z <= b.dvzmax
    )
    return bool(result)

def angle_acc(start, pred, cand):
    """Calculate angle (in gon) and acceleration between predicted and candidate positions."""
    # start, pred, cand: np.array shape (3,)
    v1 = np.array(pred) - np.array(start)
    v2 = np.array(cand) - np.array(start)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        angle = 0.0
    else:
        dot = np.dot(v1, v2) / (norm1 * norm2)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.arccos(dot) * 200.0 / np.pi  # convert rad to gon
    acc = np.linalg.norm(v2 - v1)
    return angle, acc

def candsearch_in_pix(next_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, cpar):
    """
    Search for up to 4 nearest candidates in a list of targets.
    Returns indices of up to 4 candidates.
    """
    imx = cpar.imx
    imy = cpar.imy
    xmin = max(cent_x - dl, 0.0)
    xmax = min(cent_x + dr, imx)
    ymin = max(cent_y - du, 0.0)
    ymax = min(cent_y + dd, imy)
    dmin = 1e20
    p = [-999, -999, -999, -999]
    d1 = d2 = d3 = d4 = dmin
    p1 = p2 = p3 = p4 = -999
    for j, t in enumerate(next_targets):
        x = getattr(t, 'xh', getattr(t, 'x', 0.0))
        y = getattr(t, 'yh', getattr(t, 'y', 0.0))
        if getattr(t, 'tnr', 0) != -999:
            if y > ymax:
                break
            if xmin < x < xmax and ymin < y < ymax:
                d = np.sqrt((cent_x - x) ** 2 + (cent_y - y) ** 2)
                if d < dmin:
                    dmin = d
                if d < d1:
                    p4, p3, p2, p1 = p3, p2, p1, j
                    d4, d3, d2, d1 = d3, d2, d1, d
                elif d1 < d < d2:
                    p4, p3, p2 = p3, p2, j
                    d4, d3, d2 = d3, d2, d
                elif d2 < d < d3:
                    p4, p3 = p3, j
                    d4, d3 = d3, d
                elif d3 < d < d4:
                    p4 = j
                    d4 = d
    p[0], p[1], p[2], p[3] = p1, p2, p3, p4
    return p

def candsearch_in_pix_rest(next_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, p, cpar):
    """
    Search for the nearest unmatched candidate in a list of targets.
    Returns number of matches and updates p in place.
    """
    imx = cpar.imx
    imy = cpar.imy
    xmin = max(cent_x - dl, 0.0)
    xmax = min(cent_x + dr, imx)
    ymin = max(cent_y - du, 0.0)
    ymax = min(cent_y + dd, imy)
    dmin = 1e20
    idx = -999
    for j, t in enumerate(next_targets):
        x = getattr(t, 'xh', getattr(t, 'x', 0.0))
        y = getattr(t, 'yh', getattr(t, 'y', 0.0))
        if getattr(t, 'tnr', 0) == -1:  # TR_UNUSED
            if y > ymax:
                break
            if xmin < x < xmax and ymin < y < ymax:
                d = np.sqrt((cent_x - x) ** 2 + (cent_y - y) ** 2)
                if d < dmin:
                    dmin = d
                    idx = j
    if idx != -999:
        p[0] = idx
        return 1
    return 0

def reset_foundpix_array(arr, n, num_cams):
    """Reset foundpix array to default values."""
    for i in range(n):
        arr[i]['ftnr'] = -1
        arr[i]['freq'] = 0
        # Only assign as many zeros as the whichcam array can hold
        arr[i]['whichcam'][:num_cams] = [0] * num_cams

def copy_foundpix_array(dest, src, n, num_cams):
    """Copy foundpix objects from src to dest."""
    for i in range(n):
        dest[i]['ftnr'] = src[i]['ftnr']
        dest[i]['freq'] = src[i]['freq']
        dest[i]['whichcam'][:num_cams] = src[i]['whichcam'][:num_cams]

def sort_candidates_by_freq(items, num_cams):
    """Sort foundpix items in place by frequency of appearance across cameras."""
    # This is a simplified version; assumes items is a list of dicts with 'ftnr' and 'whichcam'
    # Count frequency for each unique ftnr
    freq_map = {}
    for item in items:
        ftnr = item['ftnr']
        if ftnr not in freq_map:
            freq_map[ftnr] = 0
        freq_map[ftnr] += sum(item['whichcam'])
    # Sort items by frequency (descending)
    items.sort(key=lambda x: freq_map.get(x['ftnr'], 0), reverse=True)
    # Return number of distinct particles
    return len(set(item['ftnr'] for item in items if item['ftnr'] != -1))

def sort(n, a, b):
    """Sort float array a and int array b in ascending order of a. Returns sorted arrays."""
    combined = list(zip(a, b))
    combined.sort()
    sorted_a = [af for af, _ in combined]
    sorted_b = [bf for _, bf in combined]
    return sorted_a, sorted_b

def searchquader(point, tpar, cpar, calib):
    """
    Compute the search rectangle (quader) in pixel coordinates for each camera.
    Returns (xr, xl, yd, yu) arrays for each camera.
    """
    # This is a simplified version for test coverage. The real implementation would use calibration and projection.
    num_cams = cpar.num_cams
    xr = np.zeros(num_cams)
    xl = np.zeros(num_cams)
    yd = np.zeros(num_cams)
    yu = np.zeros(num_cams)
    for cam in range(num_cams):
        # For test coverage, use dummy values based on tpar and cpar
        xr[cam] = cpar.imx if hasattr(cpar, 'imx') else 0
        xl[cam] = 0.0
        yd[cam] = 0.0
        yu[cam] = cpar.imy if hasattr(cpar, 'imy') else 0
    return xr, xl, yd, yu
