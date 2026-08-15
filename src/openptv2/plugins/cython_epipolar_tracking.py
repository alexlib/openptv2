"""CythonEpipolarTracker — Modular tracking plugin wrapping OpenPTV2's compiled Cython trackcorr epipolar engine.

Uses OpenPTV2's native ``trackcorr_c_loop`` (compiled Cython / C) multi-camera epipolar search loop
for high-performance 2D+3D trajectory reconstruction across multi-camera setups.
"""

from __future__ import annotations

import logging

import numpy as np

from openptv2.tracker import Tracker
from openptv2.algorithms.parameters import ControlPar, VolumePar, TrackPar, SequencePar

log = logging.getLogger("openptv2.cython_epipolar_tracking")


class CythonEpipolarTracker:
    """Modular tracking plugin wrapping OpenPTV2's compiled Cython multi-camera epipolar engine."""

    def __init__(
        self,
        cpar: ControlPar | None = None,
        vpar: VolumePar | None = None,
        tpar: TrackPar | None = None,
        spar: SequencePar | None = None,
        cals: list | None = None,
        dvxmin: float = -0.015,
        dvxmax: float = 0.015,
        dvymin: float = -0.015,
        dvymax: float = 0.015,
        dvzmin: float = -0.015,
        dvzmax: float = 0.015,
        dacc: float = 0.010,
        angle: float = 60.0,
        dt: float = 1.0,
        ptv=None,
        exp=None,
        **kwargs,
    ):
        self.cpar = cpar
        self.vpar = vpar
        self.tpar = tpar
        self.spar = spar
        self.cals = cals

        self.dvxmin = dvxmin
        self.dvxmax = dvxmax
        self.dvymin = dvymin
        self.dvymax = dvymax
        self.dvzmin = dvzmin
        self.dvzmax = dvzmax
        self.dacc = dacc
        self.angle = angle
        self.dt = dt
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        """Execute experiment-level tracking, honoring the resolved
        direction (forward-only vs forward+backward) and postprocess/
        corrective-pass config -- delegates to default_tracking.Tracking,
        which every preset mapped to this plugin module (trackcorr,
        full_multipass, standard_forward, two_directional, plus this
        plugin's own cython_epipolar_tracking/openptv_epipolar keys) shares.

        Previously this called tracker.full_forward() unconditionally,
        regardless of preset -- so "full_multipass"/"two_directional"
        (meant to run forward+backward, see tracking_presets.
        _DIRECTION_BACKWARD_PRESETS) silently ran forward-only too, same as
        every other preset here. Confirmed directly: benchmarking full_
        multipass against trackcorr produced byte-identical metrics and
        full_multipass was not even slower, both symptoms of running the
        exact same code path.
        """
        if self.exp is None:
            raise ValueError("No experiment object provided")

        from openptv2.plugins.default_tracking import Tracking as _DefaultTracking

        _DefaultTracking(ptv=self.ptv, exp=self.exp).do_tracking()

    def track_sequence(
        self,
        cpar: ControlPar,
        vpar: VolumePar,
        tpar: TrackPar,
        spar: SequencePar,
        cals: list,
        naming: dict | None = None,
    ) -> Tracker:
        """Execute full multi-camera forward epipolar tracking loop via compiled Cython trackcorr.

        Parameters
        ----------
        cpar, vpar, tpar, spar, cals : OpenPTV2 Parameters and Calibrations
        naming : dict, optional
            File naming dict specifying 'corres', 'linkage', 'prio' bases.

        Returns
        -------
        tracker : Tracker
            Executed Tracker instance.
        """
        tracker = Tracker(
            cpar=cpar,
            vpar=vpar,
            tpar=tpar,
            spar=spar,
            cals=cals,
            naming=naming,
        )
        tracker.full_forward()
        return tracker

    def track_frames(self, frame_particles: list[np.ndarray]) -> list[dict]:
        """Convenience fallback to Cython 3D segment-priority engine if raw 3D coordinate
        arrays are supplied without multi-camera 2D image targets.
        """
        from openptv2.plugins.cython_3d_tracking import Cython3DTracker

        v_max = max(abs(self.dvxmax), abs(self.dvxmin))
        alt_tracker = Cython3DTracker(v_max=v_max, a_max=self.dacc, dt=self.dt)
        return alt_tracker.track_frames(frame_particles)


# Plugin contract alias
Tracking = CythonEpipolarTracker

