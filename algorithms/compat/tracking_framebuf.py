"""
Tracking frame buffer compatibility wrappers providing optv-like API.
"""

import numpy as np
from algorithms.tracking_frame_buf import (
    Target as AlgoTarget,
    Frame as AlgoFrame,
    read_targets,
    write_targets,
    read_path_frame,
    write_path_frame,
)

# Constants matching optv
CORRES_NONE = -1
PT_UNUSED = -999


class Target:
    """Wrapper for algorithms.tracking_frame_buf.Target with optv-compatible API."""

    def __init__(self, target=None, **kwargs):
        """
        Initialize Target wrapper.

        Args:
            target: AlgoTarget instance, or None to create from kwargs
            **kwargs: pnr, x, y, n, nx, ny, sumg, tnr
        """
        if target is None:
            self._target = AlgoTarget(**kwargs)
        elif isinstance(target, AlgoTarget):
            self._target = target
        elif hasattr(target, 'pnr') and hasattr(target, 'x'):
            self._target = AlgoTarget(
                pnr=target.pnr, x=target.x, y=target.y,
                n=target.n, nx=target.nx, ny=target.ny,
                sumg=target.sumg, tnr=target.tnr
            )
        else:
            raise TypeError(f"Expected Target, got {type(target)}")

    def pnr(self):
        """Get particle number."""
        return self._target.pnr

    def set_pnr(self, pnr):
        """Set particle number."""
        self._target.pnr = int(pnr)

    def pos(self):
        """Get position as ndarray[2]."""
        return np.array([self._target.x, self._target.y])

    def set_pos(self, pos):
        """Set position from ndarray[2] or tuple."""
        self._target.x = float(pos[0])
        self._target.y = float(pos[1])

    def x(self):
        """Get x coordinate."""
        return self._target.x

    def y(self):
        """Get y coordinate."""
        return self._target.y

    def tnr(self):
        """Get track number."""
        return self._target.tnr

    def set_tnr(self, tnr):
        """Set track number."""
        self._target.tnr = int(tnr)

    def count_pixels(self):
        """Get pixel counts as tuple (n, nx, ny)."""
        return (self._target.n, self._target.nx, self._target.ny)

    def set_pixel_counts(self, n, nx, ny):
        """Set pixel counts."""
        self._target.n = int(n)
        self._target.nx = int(nx)
        self._target.ny = int(ny)

    def sum_grey_value(self):
        """Get sum grey value."""
        return self._target.sumg

    def set_sum_grey_value(self, sumg):
        """Set sum grey value."""
        self._target.sumg = int(sumg)


class TargetArray:
    """Wrapper providing optv-compatible TargetArray API."""

    def __init__(self, size_or_list=0):
        """
        Initialize TargetArray.

        Args:
            size_or_list: int (size) or list of AlgoTarget instances
        """
        if isinstance(size_or_list, int):
            # Create empty array of given size
            self._targets = [
                AlgoTarget(pnr=-1, x=0.0, y=0.0, n=0, nx=0, ny=0, sumg=0, tnr=CORRES_NONE)
                for _ in range(size_or_list)
            ]
        elif isinstance(size_or_list, list):
            # Wrap existing list
            self._targets = size_or_list
        else:
            raise TypeError(f"Expected int or list, got {type(size_or_list)}")

    def __getitem__(self, idx):
        """Get Target at index."""
        return Target(self._targets[idx])

    def __setitem__(self, idx, val):
        """Set Target at index."""
        if isinstance(val, Target):
            self._targets[idx] = val._target
        elif isinstance(val, AlgoTarget):
            self._targets[idx] = val
        elif hasattr(val, 'pnr') and hasattr(val, 'x'):
            self._targets[idx] = AlgoTarget(
                pnr=val.pnr, x=val.x, y=val.y, n=val.n,
                nx=val.nx, ny=val.ny, sumg=val.sumg, tnr=val.tnr
            )
        else:
            raise TypeError(f"Expected Target or AlgoTarget, got {type(val)}")

    def __len__(self):
        """Get number of targets."""
        return len(self._targets)

    def sort_y(self):
        """Sort targets by Y coordinate in-place."""
        self._targets.sort(key=lambda t: t.y)

    def write(self, file_base, frame_num):
        """Write targets to file."""
        filename = f"{file_base}{frame_num:04d}_targets"
        return write_targets(self._targets, len(self._targets), filename, frame_num)

    @staticmethod
    def read_targets(base_name, frame_num, cpar=None):
        """
        Read targets from file.

        Args:
            base_name: Base filename (without frame number)
            frame_num: Frame number
            cpar: ControlParams (unused, for API compatibility)

        Returns:
            TargetArray instance
        """
        filename = f"{base_name}{frame_num:04d}_targets"
        targets = read_targets(filename, frame_num)
        return TargetArray(targets)


class Frame:
    """Wrapper for algorithms.tracking_frame_buf.Frame with optv-compatible API."""

    def __init__(self, num_cams=4, **kwargs):
        """
        Initialize Frame wrapper.

        Args:
            num_cams: Number of cameras
            **kwargs: corres_file_base, linkage_file_base, prio_file_base,
                     target_file_base, frame_num
        """
        self._num_cams = num_cams
        self._frame = AlgoFrame(num_cams=num_cams, max_targets=1000)

        # Read from files if provided
        if 'frame_num' in kwargs and 'target_file_base' in kwargs:
            self.read(
                kwargs.get('corres_file_base'),
                kwargs.get('linkage_file_base'),
                kwargs.get('target_file_base'),
                kwargs['frame_num'],
                kwargs.get('prio_file_base')
            )

    def read(self, corres_file_base, linkage_file_base, target_file_base,
             frame_num, prio_file_base=None):
        """
        Read frame data from files.

        Args:
            corres_file_base: Correspondence file base name
            linkage_file_base: Linkage file base name
            target_file_base: Target file base name
            frame_num: Frame number
            prio_file_base: Priority file base name (optional)

        Returns:
            bool: True if successful
        """
        return self._frame.read(
            corres_file_base, linkage_file_base, prio_file_base,
            target_file_base, frame_num
        )

    def positions(self):
        """Get 3D positions as ndarray[n, 3]."""
        # Extract positions from path_info
        num_parts = self._frame.num_parts
        positions = np.zeros((num_parts, 3), dtype=np.float64)
        for i in range(num_parts):
            positions[i] = self._frame.path_info[i].x
        return positions

    def target_positions_for_camera(self, cam):
        """
        Get 2D target positions for specific camera.

        Args:
            cam: Camera index

        Returns:
            ndarray[n, 2]: Target positions
        """
        num_targs = self._frame.num_targets[cam]
        positions = np.zeros((num_targs, 2), dtype=np.float64)
        for i in range(num_targs):
            positions[i, 0] = self._frame.targets[cam][i].x
            positions[i, 1] = self._frame.targets[cam][i].y
        return positions


def read_targets_compat(file_base, frame_num):
    """
    Read targets from file (optv-compatible function).

    Args:
        file_base: Base filename
        frame_num: Frame number

    Returns:
        TargetArray instance
    """
    return TargetArray.read_targets(file_base, frame_num)
