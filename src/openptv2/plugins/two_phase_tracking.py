"""Two-phase tracking: 3D search + 2D leaf ranking.

Phase 1: 3D KD-tree finds candidate matches within a search radius.
Phase 2: 2D pixel distances become the cost matrix; Hungarian assignment
         within connected components picks the best match globally.

This exploits the tree-forest architecture: 3D positions are the "trunk"
(structural search), per-camera 2D leaf positions are the "signature"
(disambiguation when 3D is noisy or ambiguous).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class TwoPhaseTrackerConfig:
    """Configuration for the two-phase tracker.

    Attributes
    ----------
    v_max : float
        Maximum velocity in mm/frame. Search radius for frame-to-frame matching.
    max_gap : int
        Maximum number of frames a track can survive without a match.
    dt : float
        Time step between frames (for velocity computation).
    leaf_weight : float
        Weight for 2D leaf distances in the cost matrix. If 0, falls back
        to pure 3D matching.
    """

    v_max: float = 5.0
    max_gap: int = 2
    dt: float = 1.0
    leaf_weight: float = 1.0


def _match_two_phase_frame(
    pts0: np.ndarray,
    pts1: np.ndarray,
    xy0: np.ndarray,
    xy1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    leaf_weight: float = 1.0,
) -> set[tuple[int, int]]:
    """Two-phase frame-to-frame matching: 3D search + 2D ranking.

    Parameters
    ----------
    pts0, pts1 : (N, 3) and (M, 3) — 3D positions in mm
    xy0, xy1 : (N, D) and (M, D) — flattened 2D leaf features
    p0, p1 : particle IDs for frame 0 and 1
    radius : float — 3D search radius in mm
    leaf_weight : float — weight for 2D distances in cost matrix

    Returns
    -------
    links : set of (pid0, pid1) pairs
    """
    n_pred = len(pts0)
    n_cand = len(pts1)
    if n_pred == 0 or n_cand == 0:
        return set()

    # Phase 1: 3D KD-tree candidate search
    tree3d = cKDTree(pts1)
    neighbours = tree3d.query_ball_point(pts0, r=radius)

    # Build edge list with 2D costs
    rows, cols, costs = [], [], []
    for pi in range(n_pred):
        cands = neighbours[pi]
        if len(cands) == 0:
            continue
        if leaf_weight > 0 and xy0.shape[1] > 0:
            # 2D cost: mean Euclidean distance per camera, weighted by overlap count
            C = xy0.shape[1] // 2
            xy0_cam = xy0[pi].reshape(C, 2)
            d2d = np.zeros(len(cands))
            for ci_idx, ci in enumerate(cands):
                xy1_cam = xy1[ci].reshape(C, 2)
                valid = ~np.isnan(xy0_cam[:, 0]) & ~np.isnan(xy1_cam[:, 0])
                n_valid = valid.sum()
                if n_valid > 0:
                    cam_dists = np.linalg.norm(xy0_cam[valid] - xy1_cam[valid], axis=1)
                    # Weight: more shared cameras = more reliable distance
                    d2d[ci_idx] = cam_dists.mean() * (C / n_valid)
                else:
                    d2d[ci_idx] = 1e6
            for ci_idx, ci in enumerate(cands):
                rows.append(pi)
                cols.append(ci)
                costs.append(d2d[ci_idx] * leaf_weight)
        else:
            # Fallback: 3D distance
            for ci in cands:
                rows.append(pi)
                cols.append(ci)
                costs.append(np.linalg.norm(pts0[pi] - pts1[ci]))

    if len(rows) == 0:
        return set()

    rows = np.array(rows)
    cols = np.array(cols)
    costs = np.array(costs)

    # Phase 2: Hungarian via connected components
    n_nodes = n_pred + n_cand
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols + n_pred)),
        shape=(n_nodes, n_nodes),
    )
    n_comp, labels = connected_components(graph, directed=False)

    links = set()
    edge_comp = labels[rows]
    comp_edges = np.bincount(edge_comp, minlength=n_comp)

    # Trivial components: accept directly
    trivial = comp_edges[edge_comp] == 1
    for r, c in zip(rows[trivial], cols[trivial]):
        links.add((int(p0[r]), int(p1[c])))

    # Non-trivial: small dense Hungarian per component
    rest = np.flatnonzero(~trivial)
    if len(rest):
        rest = rest[np.argsort(edge_comp[rest], kind="stable")]
        splits = np.flatnonzero(np.diff(edge_comp[rest])) + 1
        for group in np.split(rest, splits):
            c_rows = rows[group].tolist()
            c_cols = cols[group].tolist()
            c_costs = costs[group].tolist()
            uniq_r = sorted(set(c_rows))
            uniq_c = sorted(set(c_cols))
            r_local = {v: i for i, v in enumerate(uniq_r)}
            c_local = {v: i for i, v in enumerate(uniq_c)}
            max_cost = max(c_costs) if c_costs else 1.0
            sentinel = max_cost * len(c_costs) + 1.0
            sub = np.full((len(uniq_r), len(uniq_c)), sentinel, dtype=np.float64)
            for rr, cc, dd in zip(c_rows, c_cols, c_costs):
                sub[r_local[rr], c_local[cc]] = dd
            r_ind, c_ind = linear_sum_assignment(sub)
            real = sub[r_ind, c_ind] < sentinel
            for r_i, c_i in zip(r_ind[real], c_ind[real]):
                links.add((int(p0[uniq_r[r_i]]), int(p1[uniq_c[c_i]])))

    return links


class TwoPhaseTracker:
    """Frame-to-frame tracker using 3D search + 2D leaf ranking.

    Parameters
    ----------
    config : TwoPhaseTrackerConfig
        Tracker configuration.
    """

    def __init__(self, config: TwoPhaseTrackerConfig | None = None):
        self.cfg = config or TwoPhaseTrackerConfig()

    def track_frames(
        self,
        frame_particles: list[np.ndarray],
        frame_leaves: list[np.ndarray] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Track particles across frames using two-phase matching.

        Parameters
        ----------
        frame_particles : list of (N_i, 3) arrays
            3D positions per frame.
        frame_leaves : list of (N_i, D) arrays, optional
            Flattened 2D leaf features per frame. If None, falls back to
            pure 3D matching.

        Returns
        -------
        links : list of (t0, pid0, t1, pid1) tuples
            Frame-to-frame particle links (0-based time indices).
        """
        num_frames = len(frame_particles)
        if num_frames < 2:
            return []

        if frame_leaves is None:
            frame_leaves = [np.zeros((len(p), 0)) for p in frame_particles]

        all_links = []
        for i in range(num_frames - 1):
            t0, t1 = i, i + 1
            pts0, pts1 = frame_particles[t0], frame_particles[t1]
            lf0, lf1 = frame_leaves[t0], frame_leaves[t1]
            p0 = np.arange(len(pts0), dtype=np.int32)
            p1 = np.arange(len(pts1), dtype=np.int32)

            links = _match_two_phase_frame(
                pts0, pts1, lf0, lf1, p0, p1,
                self.cfg.v_max, self.cfg.leaf_weight,
            )
            for pid0, pid1 in links:
                all_links.append((t0, pid0, t1, pid1))

        return all_links


class Tracking:
    """Plugin interface for the GUI/batch pipeline.

    Reads ``leaf_weight`` and ``dvxmax`` (as v_max) from the experiment's
    ``track`` parameter section, loads 3D positions + 2D leaves from the
    RunStore, runs two-phase matching, and writes links back.
    """

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)

        track_cfg = pm.parameters.get("track", {}) if pm else {}
        leaf_weight = float(track_cfg.get("leaf_weight", 0.0))
        v_max = float(track_cfg.get("dvxmax", 15.5))

        store = getattr(self.exp, "_store", None)
        if store is None:
            print(
                "TwoPhaseTracker requires a RunStore (zarr) with "
                "correspondences and targets. Falling back to default tracker."
            )
            tracker = self.ptv.py_trackcorr_init(self.exp)
            self.exp.tracker = tracker
            tracker.full_forward()
            return

        frames = sorted(store.frames())
        num_cams = len(store.cam_ids)

        frame_particles = []
        frame_leaves = []
        for f in frames:
            pos_3d, _ = store.read_correspondences(f)
            pos_3d = np.asarray(pos_3d)
            frame_particles.append(pos_3d)

            n = len(pos_3d)
            cam_ids = np.asarray(
                store.root[f"correspondences/frame_{f:06d}"]
            )[:, 3:].astype(int)
            xy = np.full((n, num_cams, 2), np.nan)
            for c in range(num_cams):
                key = f"targets/cam_{c}/frame_{f:06d}"
                if key in store.root:
                    t = np.asarray(store.root[key])
                    valid = cam_ids[:, c] >= 0
                    xy[valid, c] = t[cam_ids[valid, c], 1:3]
            frame_leaves.append(np.nan_to_num(xy.reshape(n, -1)))

        cfg = TwoPhaseTrackerConfig(v_max=v_max, leaf_weight=leaf_weight)
        tracker = TwoPhaseTracker(cfg)
        links = tracker.track_frames(frame_particles, frame_leaves)

        from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE

        for t0, p0, t1, p1 in links:
            f0 = frames[t0]
            f1 = frames[t1]
            linkage_key = f"linkage/{f1:06d}"
            if linkage_key not in store.root:
                continue
            prev_arr = np.asarray(store.root[linkage_key][:, 0], dtype=np.int32)
            prev_arr[p1] = p0
            store.root[linkage_key][:, 0] = prev_arr

            linkage_key0 = f"linkage/{f0:06d}"
            if linkage_key0 not in store.root:
                continue
            nxt_arr = np.asarray(store.root[linkage_key0][:, 1], dtype=np.int32)
            nxt_arr[p0] = p1
            store.root[linkage_key0][:, 1] = nxt_arr

        print(
            f"TwoPhaseTracker: {len(links)} links across {len(frames)} frames "
            f"(leaf_weight={leaf_weight}, v_max={v_max})"
        )
