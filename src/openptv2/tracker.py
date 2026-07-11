"""Streamlined particle tracking control class."""

from openptv2.algorithms.tracking_run import TrackingRun
from openptv2.algorithms.track import (
    track_forward_start,
    trackcorr_c_loop,
    trackcorr_c_finish,
    trackback_c,
)
from openptv2.algorithms.track3d import track3d_loop
from openptv2.algorithms.parameters import convert_track_par_to_tuple

# Default file naming (matches optv)
default_naming = {
    'corres': 'res/rt_is',
    'linkage': 'res/ptv_is',
    'prio': 'res/added',
}


class Tracker:
    """
    Control class for particle tracking.

    Uses functional tracking API (track_forward_start, trackcorr_c_loop, etc.)
    wrapped in a class-based interface matching optv.Tracker.
    """

    def __init__(self, cpar, vpar, tpar, spar, cals, naming=None, flatten_tol=0.0001):
        """
        Initialize Tracker.

        Args:
            cpar: ControlParams instance
            vpar: VolumeParams instance
            tpar: TrackingParams instance
            spar: SequenceParams instance
            cals: List of Calibration instances
            naming: Dict with 'corres', 'linkage', 'prio' file base names
            flatten_tol: Flatness tolerance for epipolar matching
        """
        self._cpar = cpar
        self._vpar = vpar
        self._tpar = tpar
        self._spar = spar
        self._cals = cals
        self._flatten_tol = flatten_tol

        # File naming
        if naming is None:
            naming = default_naming.copy()
        self._naming = naming

        # Params are the algorithms *Par classes directly; tpar still needs the
        # tuple conversion the tracking loop expects.
        self._cpar_algo = cpar
        self._vpar_algo = vpar
        self._tpar_algo = convert_track_par_to_tuple(tpar)
        self._spar_algo = spar
        self._cals_algo = list(cals)

        # Tracking run object
        self._run = None
        self._current_step = None
        self._is_initialized = False

    def restart(self):
        """
        Initialize tracking run (prepare for forward tracking).
        """
        # Create TrackingRun
        self._run = TrackingRun(
            seq_par=self._spar_algo,
            tpar=self._tpar_algo,
            vpar=self._vpar_algo,
            cpar=self._cpar_algo,
            buf_len=4,  # Standard buffer length
            max_targets=10000,  # Generous max
            corres_file_base=self._naming['corres'],
            linkage_file_base=self._naming['linkage'],
            prio_file_base=self._naming['prio'],
            cal=self._cals_algo,
            flatten_tol=self._flatten_tol,
        )

        # Initialize forward tracking
        track_forward_start(self._run)

        # Set current step to first frame
        self._current_step = self._spar.get_first()
        self._is_initialized = True

    def step_forward(self):
        """
        Process one frame of forward tracking.

        Returns:
            bool: True if more frames remain, False if done
        """
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Call restart() first.")

        # Check if we've reached the end
        if self._current_step > self._spar.get_last():
            return False

        # Process current frame
        trackcorr_c_loop(self._run, self._current_step)

        # Advance to next frame
        self._current_step += 1

        return self._current_step <= self._spar.get_last()

    def finalize(self):
        """
        Finalize forward tracking (finish last frame).
        """
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Call restart() first.")

        # Finish tracking at last processed frame
        trackcorr_c_finish(self._run, self._current_step - 1)

    def full_forward(self):
        """
        Run complete forward tracking (restart + loop + finalize).
        """
        self.restart()

        # Process all frames
        while self.step_forward():
            pass

        # Finalize
        self.finalize()

    def full_backward(self):
        """
        Run backward tracking.

        Must be called after full_forward().
        """
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Run full_forward() first.")

        trackback_c(self._run)

    def step_forward_3d(self):
        """
        Process one frame of 3D tracking.

        Returns:
            bool: True if more frames remain, False if done
        """
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Call restart() first.")

        # Check if we've reached the end (mirrors range(first, last))
        if self._current_step >= self._spar.get_last():
            return False

        # Process current frame
        track3d_loop(self._run, self._current_step)

        # Advance to next frame
        self._current_step += 1

        return self._current_step < self._spar.get_last()

    def full_forward_3d(self):
        """
        Run complete 3D forward tracking.
        """
        self.restart()

        # Process all frames
        while self.step_forward_3d():
            pass

        # Write final frame (mirrors trackcorr_c_finish at seq_par.last)
        trackcorr_c_finish(self._run, self._spar.get_last())

    def current_step(self):
        """
        Get current frame number.

        Returns:
            int: Current frame number
        """
        if not self._is_initialized:
            return -1
        return self._current_step


__all__ = ["Tracker", "default_naming"]
