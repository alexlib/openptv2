"""Cython3DTracker — Modular tracking plugin wrapping OpenPTV2's compiled Cython track3d kernel.

Uses OpenPTV2's native ``_track3d_loop_fast`` (compiled C/Cython) 4-level global priority
segment linking cascade for ultra-fast 3D position-space trajectory reconstruction.
"""

from __future__ import annotations

import logging
import numpy as np

from openptv2.algorithms.track_kernels_track3d import track3d_loop_fast
from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE

log = logging.getLogger("openptv2.cython_3d_tracking")

MAX_CANDS = 32

class Cython3DTracker:
    """Modular 3D tracker using OpenPTV2's compiled Cython segment-priority engine."""

    def __init__(
        self,
        v_max: float = 0.015,
        a_max: float = 0.010,
        max_cands: int = MAX_CANDS,
        dt: float = 1.0,
        ptv=None,
        exp=None,
    ):
        """
        Parameters
        ----------
        v_max : float
            Maximum expected velocity (search box half-width dx=dy=dz).
        a_max : float
            Maximum expected acceleration bound (dacc in Cython kernel).
        max_cands : int
            Maximum number of candidate neighbors evaluated per particle (default 32).
        dt : float
            Frame time step interval.
        ptv, exp : optional
            OpenPTV2 module and experiment objects for pipeline plugin runner.
        """
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.max_cands = int(max_cands)
        self.dt = float(dt)
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        """Execute experiment-level tracking via OpenPTV2 full_forward_3d()."""
        if self.exp is None:
            raise ValueError("No experiment object provided")

        if self.ptv is not None:
            tracker = self.ptv.py_trackcorr_init(self.exp)
            self.exp.tracker = tracker
            tracker.full_forward_3d()

    def track_frames(self, frame_particles: list[np.ndarray]) -> list[dict]:
        """Track 3D particle coordinate arrays across time frames.

        Parameters
        ----------
        frame_particles : list of ndarray
            List of length N_frames where each entry is an (N_i, 3) float array of 3D positions.

        Returns
        -------
        tracks : list of dict
            Reconstructed 3D trajectories with keys: 'id', 'pos', 'time', 'vel'.
        """
        num_frames = len(frame_particles)
        if num_frames < 2:
            return []

        # Ensure C-contiguous float64 coordinate arrays
        positions = [
            np.ascontiguousarray(pts, dtype=np.float64) for pts in frame_particles
        ]
        counts = [len(pts) for pts in positions]

        # Allocate link buffers (next and prev pointers per frame)
        prev_links = [np.full(n, PREV_NONE, dtype=np.int32) for n in counts]
        next_links = [np.full(n, NEXT_NONE, dtype=np.int32) for n in counts]

        # Sequential Cython 3-frame sliding window tracking loop
        for t in range(num_frames - 1):
            n1 = counts[t]
            n2 = counts[t + 1]
            if n1 == 0 or n2 == 0:
                continue

            if t == 0:
                # Cold start: Frame 0 has no velocity history
                n0 = 0
                pos_0 = np.empty((0, 3), dtype=np.float64)
                prev_0 = np.empty(0, dtype=np.int32)
            else:
                n0 = counts[t - 1]
                pos_0 = positions[t - 1]
                prev_0 = prev_links[t - 1]

            pos_1 = positions[t]
            prev_1 = prev_links[t]
            next_1 = next_links[t]

            pos_2 = positions[t + 1]
            prev_2 = prev_links[t + 1]
            next_2 = next_links[t + 1]

            # Execute compiled Cython track3d_loop_fast kernel!
            track3d_loop_fast(
                n1,
                pos_0, prev_0, n0,
                pos_1, prev_1, next_1, n1,
                pos_2, prev_2, next_2, n2,
                self.v_max, self.v_max, self.v_max,
                self.max_cands,
                self.a_max,
            )

        # Assemble individual tracks from next_links graph
        tracks = []
        visited = [np.zeros(n, dtype=bool) for n in counts]
        next_track_id = 0

        for t in range(num_frames - 1):
            for i in range(counts[t]):
                if visited[t][i]:
                    continue

                # Start a new trajectory if this particle is unvisited
                # Trace chain forward through next_links
                curr_t = t
                curr_i = i
                
                tr_pos = []
                tr_time = []

                while curr_t < num_frames and curr_i != NEXT_NONE:
                    if visited[curr_t][curr_i]:
                        break
                    visited[curr_t][curr_i] = True
                    tr_pos.append(positions[curr_t][curr_i])
                    tr_time.append(curr_t)

                    # Get next linked particle index
                    nxt = next_links[curr_t][curr_i]
                    if nxt >= 0 and curr_t + 1 < num_frames:
                        curr_t += 1
                        curr_i = nxt
                    else:
                        break

                if len(tr_pos) >= 2:
                    pos_arr = np.array(tr_pos, dtype=np.float64)
                    time_arr = np.array(tr_time, dtype=np.float64) * self.dt
                    
                    # Estimate velocities along the track
                    vel_arr = np.zeros_like(pos_arr)
                    if len(pos_arr) > 1:
                        vel_arr[1:] = np.diff(pos_arr, axis=0) / self.dt
                        vel_arr[0] = vel_arr[1]

                    tracks.append({
                        "id": next_track_id,
                        "pos": pos_arr,
                        "time": time_arr,
                        "vel": vel_arr,
                    })
                    next_track_id += 1

        return tracks


# Plugin contract alias
Tracking = Cython3DTracker

