"""Streamlined particle tracking control class."""

from pathlib import Path

from openptv2.algorithms.parameters import convert_track_par_to_tuple
from openptv2.algorithms.track import (
    track_forward_start,
    trackback_c,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.track3d import track3d_loop
from openptv2.algorithms.track4be import track4be_loop
from openptv2.algorithms.tracking_run import TrackingRun

# Default file naming (matches optv)
default_naming = {
    "corres": "res/rt_is",
    "linkage": "res/ptv_is",
    "prio": "res/added",
}

DEFAULT_MAX_TARGETS = 10000
_MAX_TARGETS_MARGIN = 1.5  # headroom over the largest frame actually seen


def _estimate_max_targets(spar, naming, store, floor=DEFAULT_MAX_TARGETS):
    """Per-frame target/particle buffers (Frame(num_cams, max_targets), one
    per slot in the 4-frame ring buffer) are preallocated ONCE for the whole
    run and written into with boundscheck disabled in the compiled kernels --
    a frame with more particles than max_targets silently corrupts memory
    rather than raising. The old code hardcoded max_targets=10000 regardless
    of the actual data; this peeks at the real correspondence counts across
    the run's frames (via the store if attached, else the rt_is ASCII line
    counts) and sizes from the observed maximum instead, so a 20k-particle
    run doesn't run into the same fixed cap a small run happened to fit
    under.

    Falls back to `floor` if no counts could be read (e.g. sequence hasn't
    been processed yet) -- unchanged old behaviour, not a regression.
    """
    counts = []
    first, last = spar.get_first(), spar.get_last()
    for f in range(first, last + 1):
        n = None
        if store is not None:
            try:
                pos, _ids = store.read_correspondences(f)
                n = len(pos)
            except Exception:
                n = None
        if n is None:
            path = f"{naming['corres']}.{f}"
            try:
                with open(path, encoding="utf-8") as fh:
                    n = int(fh.readline().strip())
            except (OSError, ValueError):
                n = None
        if n is not None:
            counts.append(n)

    if not counts:
        return floor
    return max(floor, int(max(counts) * _MAX_TARGETS_MARGIN))


class Tracker:
    """
    Control class for particle tracking.

    Uses functional tracking API (track_forward_start, trackcorr_c_loop, etc.)
    wrapped in a class-based interface matching optv.Tracker.
    """

    def __init__(
        self, cpar, vpar, tpar, spar, cals, naming=None, flatten_tol=0.0001, store=None
    ):
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
            store: An openptv2.storage.RunStore, or None. When given, every
                frame written/read during tracking also goes through the
                unified store (unconditional dual-write with ASCII).
        """
        self._cpar = cpar
        self._vpar = vpar
        self._tpar = tpar
        self._spar = spar
        self._cals = cals
        self._flatten_tol = flatten_tol
        self._store = store

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
        if self._store is not None and self._naming.get("linkage"):
            self._store.clear_linkage(Path(self._naming["linkage"]).name)

        max_targets = _estimate_max_targets(self._spar, self._naming, self._store)

        # Create TrackingRun
        self._run = TrackingRun(
            seq_par=self._spar_algo,
            tpar=self._tpar_algo,
            vpar=self._vpar_algo,
            cpar=self._cpar_algo,
            buf_len=4,  # Standard buffer length
            max_targets=max_targets,
            corres_file_base=self._naming["corres"],
            linkage_file_base=self._naming["linkage"],
            prio_file_base=self._naming["prio"],
            cal=self._cals_algo,
            flatten_tol=self._flatten_tol,
            store=self._store,
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

        # Stop before the last frame: step k links frame k -> k+1, so the last
        # valid step is (last - 1). Mirrors step_forward_3d / range(first, last).
        if self._current_step >= self._spar.get_last():
            return False

        # Process current frame
        trackcorr_c_loop(self._run, self._current_step)

        # Advance to next frame
        self._current_step += 1

        return self._current_step < self._spar.get_last()

    def finalize(self):
        """
        Finalize forward tracking (write the last frame).
        """
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Call restart() first.")

        # Finish at seq_par.last (mirrors full_forward_3d); independent of how
        # many steps ran.
        trackcorr_c_finish(self._run, self._spar.get_last())

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

    def postprocess(
        self,
        cold_start: bool = True,
        reciprocity: bool = True,
        gap_relinking: bool = True,
        max_gap: int = 2,
    ):
        """Disk-level trajectory-quality post-passes over the linkage files.

        Run after full_forward (+ full_backward). ``cold_start`` recovers the
        under-linked first transition using the velocity field the later frames
        established; ``reciprocity`` severs any non-bidirectional links; ``gap_relinking``
        bridges occluded particle trajectory gaps. Returns a stats dict.
        """
        from openptv2.tracking_postprocess import (
            count_links,
            enforce_reciprocity,
            relink_trajectory_gaps,
            seed_cold_start,
        )

        base = self._naming["linkage"]
        first, last = self._spar.get_first(), self._spar.get_last()
        store = self._store
        stats = {"links_before": count_links(base, first, last, store=store)}
        if cold_start:
            stats["cold_start"] = seed_cold_start(
                base, first, last, float(self._tpar_algo.dvxmax), store=store
            )
        if gap_relinking:
            stats["gap_relinking"] = relink_trajectory_gaps(
                base,
                first,
                last,
                max_gap=max_gap,
                max_accel_err=float(self._tpar_algo.dacc),
                store=store,
            )
        if reciprocity:
            stats["reciprocity"] = enforce_reciprocity(base, first, last, store=store)
        stats["links_after"] = count_links(base, first, last, store=store)
        return stats

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

    def step_forward_4be(self):
        """Process one frame of 4BE tracking. Returns True while frames remain."""
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized. Call restart() first.")

        if self._current_step >= self._spar.get_last():
            return False

        track4be_loop(self._run, self._current_step)
        self._current_step += 1
        return self._current_step < self._spar.get_last()

    def full_forward_4be(self):
        """Run complete 4BE forward tracking (Ouellette et al. four-frame
        best estimate) -- same stereo-3D input as full_forward_3d, different
        candidate cost. See openptv2.algorithms.track4be."""
        self.restart()
        while self.step_forward_4be():
            pass
        trackcorr_c_finish(self._run, self._spar.get_last())

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
        if not self._is_initialized:
            return -1
        return self._current_step

    @property
    def npart(self):
        return self._run.npart if self._run else 0

    @property
    def nlinks(self):
        return self._run.nlinks if self._run else 0


__all__ = ["Tracker", "default_naming"]
