import numpy as np

from .constants import MAX_CANDS, PT_UNUSED, TR_UNUSED


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
    """Calculate angle (in gon) and acceleration between predicted and candidate positions.

    Matches C angle_acc: returns (angle, acc).
    Special cases: opposite directions = 200 gon, same direction = 0 gon.
    """
    v0 = np.asarray(pred) - np.asarray(start)
    v1 = np.asarray(cand) - np.asarray(start)

    if np.array_equal(v0, -v1):
        angle = 200.0
    elif np.array_equal(v0, v1):
        angle = 0.0
    else:
        norm0 = np.linalg.norm(v0)
        norm1 = np.linalg.norm(v1)
        if norm0 == 0 or norm1 == 0:
            angle = 0.0
        else:
            dot = np.dot(v0, v1) / (norm0 * norm1)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot) * 200.0 / np.pi

    acc = np.linalg.norm(v1 - v0)
    return angle, acc

def candsearch_in_pix(next_targets, num_targets, cent_x, cent_y, dl, dr, du, dd, cpar):
    """Search for up to 4 nearest candidates in a list of targets.

    Matches C candsearch_in_pix: returns (counter, p) where counter is the
    number of candidates found and p is a list of 4 target indices
    (PT_UNUSED=-999 for unused slots).
    """
    from .constants import PT_UNUSED, TR_UNUSED

    p = [PT_UNUSED] * 4

    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > cpar.imy:
        ymax = cpar.imy

    p1 = p2 = p3 = p4 = PT_UNUSED
    dmin = 1e20
    d1 = d2 = d3 = d4 = dmin

    if not (0.0 <= cent_x <= cpar.imx and 0.0 <= cent_y <= cpar.imy):
        return p

    # Binary search for start point
    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next_targets[j0].y < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        t = next_targets[j]
        if t.tnr != TR_UNUSED:
            if t.y > ymax:
                break
            if t.x > xmin and t.x < xmax and t.y > ymin and t.y < ymax:
                d = np.sqrt((cent_x - t.x) ** 2 + (cent_y - t.y) ** 2)

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
    """Search for the nearest unmatched candidate (tnr == TR_UNUSED).

    Matches C candsearch_in_pix_rest: returns number of candidates (0 or 1)
    and sets p[0] to the target index if found.
    """
    from .constants import PT_UNUSED, TR_UNUSED

    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > cpar.imy:
        ymax = cpar.imy

    p[0] = PT_UNUSED
    counter = 0
    dmin = 1e20

    if not (0.0 <= cent_x <= cpar.imx and 0.0 <= cent_y <= cpar.imy):
        return 0

    # Binary search for start point
    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next_targets[j0].y < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        t = next_targets[j]
        if t.tnr == TR_UNUSED:
            if t.y > ymax:
                break
            if t.x > xmin and t.x < xmax and t.y > ymin and t.y < ymax:
                d = np.sqrt((cent_x - t.x) ** 2 + (cent_y - t.y) ** 2)
                if d < dmin:
                    dmin = d
                    p[0] = j
                    counter = 1

    return counter

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
    """Sort foundpix items in place by frequency of appearance across cameras.

    Matches C sort_candidates_by_freq exactly:
    1. Mark which cameras saw each ftnr
    2. Count frequency
    3. Sort by freq descending
    4. Prune duplicates and singletons
    5. Sort again
    """
    n = num_cams * MAX_CANDS

    # Step 1: where what was found
    for i in range(n):
        for j in range(num_cams):
            for m in range(MAX_CANDS):
                if items[i]['ftnr'] == items[4 * j + m]['ftnr']:
                    items[i]['whichcam'][j] = 1

    # Step 2: how often was ftnr found
    for i in range(n):
        for j in range(num_cams):
            if items[i]['whichcam'][j] == 1 and items[i]['ftnr'] != TR_UNUSED:
                items[i]['freq'] += 1

    # Step 3: bubble sort by freq descending
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if items[j - 1]['freq'] < items[j]['freq']:
                items[j - 1], items[j] = items[j].copy(), items[j - 1].copy()

    # Step 4: prune duplicates or those found only once
    for i in range(n):
        for j in range(i + 1, n):
            if items[i]['ftnr'] == items[j]['ftnr'] or items[j]['freq'] < 2:
                items[j]['freq'] = 0
                items[j]['ftnr'] = TR_UNUSED

    # Step 5: sort again
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if items[j - 1]['freq'] < items[j]['freq']:
                items[j - 1], items[j] = items[j].copy(), items[j - 1].copy()

    different = 0
    for i in range(n):
        if items[i]['freq'] != 0:
            different += 1
    return different

def sort(n, a, b):
    """Sort float array a and int array b in ascending order of a. Returns sorted arrays."""
    combined = list(zip(a, b))
    combined.sort()
    sorted_a = [af for af, _ in combined]
    sorted_b = [bf for _, bf in combined]
    return sorted_a, sorted_b

def point_to_pixel(point, cal, cpar):
    """Project 3D point to pixel coordinates."""
    from .imgcoord import img_coord
    from .trafo import metric_to_pixel
    x, y = img_coord(point, cal, cpar.mm)
    return metric_to_pixel(x, y, cpar)


def searchquader(point, tpar, cpar, calib):
    """Compute the search rectangle in pixel coordinates for each camera.

    Projects 8 corners of the search volume cuboid to pixel space and finds
    the bounding box relative to the center projection.

    Returns (xr, xl, yd, yu) arrays for each camera.
    """
    num_cams = cpar.num_cams
    xr = np.zeros(num_cams)
    xl = np.zeros(num_cams)
    yd = np.zeros(num_cams)
    yu = np.zeros(num_cams)

    mins = np.array([tpar.dvxmin, tpar.dvymin, tpar.dvzmin])
    maxes = np.array([tpar.dvxmax, tpar.dvymax, tpar.dvzmax])

    quader = np.zeros((8, 3))
    for pt in range(8):
        quader[pt] = point.copy()
        for dim in range(3):
            if pt & (1 << dim):
                quader[pt, dim] += maxes[dim]
            else:
                quader[pt, dim] += mins[dim]

    for i in range(num_cams):
        xr[i] = 0
        xl[i] = cpar.imx
        yd[i] = 0
        yu[i] = cpar.imy

        cx, cy = point_to_pixel(point, calib[i], cpar)

        for pt in range(8):
            corner_x, corner_y = point_to_pixel(quader[pt], calib[i], cpar)

            if corner_x < xl[i]:
                xl[i] = corner_x
            if corner_y < yu[i]:
                yu[i] = corner_y
            if corner_x > xr[i]:
                xr[i] = corner_x
            if corner_y > yd[i]:
                yd[i] = corner_y

        if xl[i] < 0:
            xl[i] = 0
        if yu[i] < 0:
            yu[i] = 0
        if xr[i] > cpar.imx:
            xr[i] = cpar.imx
        if yd[i] > cpar.imy:
            yd[i] = cpar.imy

        xr[i] = xr[i] - cx
        xl[i] = cx - xl[i]
        yd[i] = yd[i] - cy
        yu[i] = cy - yu[i]

    return xr, xl, yd, yu
