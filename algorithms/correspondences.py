"""Multi-camera correspondence matching.

Translation of lib/src/correspondences.c and lib/include/correspondences.h.

Establishes correspondences between detected targets across 2-4 cameras
using epipolar geometry and clique finding.

Design:
- SoA layout for n-tupels (p0, p1, p2, p3, corr arrays)
- No quicksort implementations - use numpy.argsort
- Clear separation of matching stages
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence

# Maximum correspondences per frame
NMAX = 20240


@dataclass
class NTupel:
    """A correspondence match across multiple cameras.

    Attributes:
        p: target indices per camera (-1 if none), length 4.
        corr: correspondence quality score.
    """
    p: list[int] = field(default_factory=lambda: [-1, -1, -1, -1])
    corr: float = 0.0


@dataclass
class Correspond:
    """Adjacency list entry for candidate matching.

    Attributes:
        p1: master point number.
        p2: candidate point numbers.
        corr: feature-based correlation coefficients.
        dist: distances perpendicular to epipolar line.
    """
    p1: int = 0
    p2: list[int] = field(default_factory=list)
    corr: list[float] = field(default_factory=list)
    dist: list[float] = field(default_factory=list)


def match_pairs(
    targets: list[list[dict]],
    corrected: list[list[tuple[float, float]]],
    vpar: dict,
    cpar: dict,
    calibrations: list[dict],
    mm_params: dict,
) -> list[list[Correspond]]:
    """Build pairwise adjacency lists between all camera pairs.

    For each target in each camera, projects epipolar lines into other
    cameras and finds candidate matches.

    Args:
        targets: per-camera target lists (each target is a dict with n, nx, ny, sumg, x, y).
        corrected: per-camera corrected (flat-image) coordinates [(x, y), ...].
        vpar: volume parameters (eps0, cn, cnx, cny, csumg, X_lay, Zmin_lay, Zmax_lay).
        cpar: control parameters (imx, imy, pix_x, pix_y, chfield).
        calibrations: per-camera calibration dicts.
        mm_params: multimedia parameters.

    Returns:
        adj_lists[i][j]: list of Correspond from camera i to camera j.
    """
    from .epi import epi_mm, find_candidate, Coord2d

    num_cams = len(targets)
    adj_lists = [[[] for _ in range(num_cams)] for _ in range(num_cams)]

    for cam1 in range(num_cams):
        for cam2 in range(cam1 + 1, num_cams):
            # Build sorted coordinate list for cam2
            crd2 = [
                Coord2d(pnr=j, x=corrected[cam2][j][0], y=corrected[cam2][j][1])
                for j in range(len(corrected[cam2]))
            ]
            # Sort by x for binary search
            sort_idx = np.argsort([c.x for c in crd2])
            crd2_sorted = [crd2[i] for i in sort_idx]

            for t1_idx, t1 in enumerate(targets[cam1]):
                # Get epipolar line in cam2
                xmin, ymin, xmax, ymax = epi_mm(
                    corrected[cam1][t1_idx][0],
                    corrected[cam1][t1_idx][1],
                    calibrations[cam1],
                    calibrations[cam2],
                    mm_params["n1"],
                    mm_params["n2_0"],
                    mm_params["n3"],
                    mm_params["d0"],
                    tuple(vpar["X_lay"]),
                    tuple(vpar["Zmin_lay"]),
                    tuple(vpar["Zmax_lay"]),
                )

                # Find candidates
                cands = find_candidate(
                    crd2_sorted,
                    targets[cam2],
                    xmin, ymin, xmax, ymax,
                    t1["n"], t1["nx"], t1["ny"], t1["sumg"],
                    vpar["eps0"],
                    vpar["cn"], vpar["cnx"], vpar["cny"], vpar["csumg"],
                    cpar["imx"], cpar["imy"],
                    cpar["pix_x"], cpar["pix_y"],
                    calibrations[cam2]["int_xh"],
                    calibrations[cam2]["int_yh"],
                    calibrations[cam2]["k1"], calibrations[cam2]["k2"],
                    calibrations[cam2]["k3"], calibrations[cam2]["p1"],
                    calibrations[cam2]["p2"], calibrations[cam2]["scx"],
                    calibrations[cam2]["she"],
                )

                if cands:
                    corr_entry = Correspond(
                        p1=t1_idx,
                        p2=[c.pnr for c in cands],
                        corr=[c.corr for c in cands],
                        dist=[c.tol for c in cands],
                    )
                    adj_lists[cam1][cam2].append(corr_entry)

            # Repeat for cam2 -> cam1
            crd1 = [
                Coord2d(pnr=j, x=corrected[cam1][j][0], y=corrected[cam1][j][1])
                for j in range(len(corrected[cam1]))
            ]
            sort_idx = np.argsort([c.x for c in crd1])
            crd1_sorted = [crd1[i] for i in sort_idx]

            for t2_idx, t2 in enumerate(targets[cam2]):
                xmin, ymin, xmax, ymax = epi_mm(
                    corrected[cam2][t2_idx][0],
                    corrected[cam2][t2_idx][1],
                    calibrations[cam2],
                    calibrations[cam1],
                    mm_params["n1"],
                    mm_params["n2_0"],
                    mm_params["n3"],
                    mm_params["d0"],
                    tuple(vpar["X_lay"]),
                    tuple(vpar["Zmin_lay"]),
                    tuple(vpar["Zmax_lay"]),
                )

                cands = find_candidate(
                    crd1_sorted,
                    targets[cam1],
                    xmin, ymin, xmax, ymax,
                    t2["n"], t2["nx"], t2["ny"], t2["sumg"],
                    vpar["eps0"],
                    vpar["cn"], vpar["cnx"], vpar["cny"], vpar["csumg"],
                    cpar["imx"], cpar["imy"],
                    cpar["pix_x"], cpar["pix_y"],
                    calibrations[cam1]["int_xh"],
                    calibrations[cam1]["int_yh"],
                    calibrations[cam1]["k1"], calibrations[cam1]["k2"],
                    calibrations[cam1]["k3"], calibrations[cam1]["p1"],
                    calibrations[cam1]["p2"], calibrations[cam1]["scx"],
                    calibrations[cam1]["she"],
                )

                if cands:
                    corr_entry = Correspond(
                        p1=t2_idx,
                        p2=[c.pnr for c in cands],
                        corr=[c.corr for c in cands],
                        dist=[c.tol for c in cands],
                    )
                    adj_lists[cam2][cam1].append(corr_entry)

    return adj_lists


def four_camera_matching(
    adj_lists: list[list[list[Correspond]]],
    base_target_count: int,
    accept_corr: float,
) -> list[NTupel]:
    """Find consistent 4-camera correspondences (quadruplets).

    Cross-references all 6 pairwise adjacency lists to find
    targets that are mutually consistent across all 4 cameras.

    Args:
        adj_lists: pairwise adjacency lists [cam_i][cam_j].
        base_target_count: number of targets in base camera.
        accept_corr: minimum correlation threshold.

    Returns:
        List of 4-tupel matches.
    """
    num_cams = 4
    quadruplets = []

    # Build lookup: (cam1, p1) -> set of cam2->p2 mappings
    pair_lookup = {}
    for cam1 in range(num_cams):
        for cam2 in range(num_cams):
            if cam1 == cam2:
                continue
            key = (cam1, cam2)
            pair_lookup[key] = {}
            for corr_entry in adj_lists[cam1][cam2]:
                p1 = corr_entry.p1
                for idx, p2 in enumerate(corr_entry.p2):
                    if p1 not in pair_lookup[key]:
                        pair_lookup[key][p1] = {}
                    pair_lookup[key][p1][p2] = corr_entry.corr[idx]

    # For each target in camera 0, try to find consistent matches
    for p0 in range(base_target_count):
        # Get candidates in other cameras
        cands_per_cam = {}
        for cam in range(1, num_cams):
            if (0, cam) in pair_lookup and p0 in pair_lookup[(0, cam)]:
                cands_per_cam[cam] = list(pair_lookup[(0, cam)][p0].keys())
            else:
                cands_per_cam[cam] = []

        if any(len(c) == 0 for c in cands_per_cam.values()):
            continue

        # Check consistency across all pairs
        for p1 in cands_per_cam[1]:
            for p2 in cands_per_cam[2]:
                for p3 in cands_per_cam[3]:
                    # Verify all cross-pairs exist
                    if (
                        p2 not in pair_lookup.get((1, 2), {}).get(p1, {})
                        or p3 not in pair_lookup.get((1, 3), {}).get(p1, {})
                        or p3 not in pair_lookup.get((2, 3), {}).get(p2, {})
                    ):
                        continue

                    # Compute average correlation
                    corrs = [
                        pair_lookup[(0, 1)][p0][p1],
                        pair_lookup[(0, 2)][p0][p2],
                        pair_lookup[(0, 3)][p0][p3],
                        pair_lookup[(1, 2)][p1][p2],
                        pair_lookup[(1, 3)][p1][p3],
                        pair_lookup[(2, 3)][p2][p3],
                    ]
                    avg_corr = sum(corrs) / len(corrs)

                    if avg_corr >= accept_corr:
                        quadruplets.append(
                            NTupel(p=[p0, p1, p2, p3], corr=avg_corr)
                        )

    return quadruplets


def three_camera_matching(
    adj_lists: list[list[list[Correspond]]],
    target_counts: list[int],
    accept_corr: float,
    used_targets: list[set[int]],
) -> list[NTupel]:
    """Find 3-camera correspondences, skipping targets used by quadruplets.

    Args:
        adj_lists: pairwise adjacency lists.
        target_counts: number of targets per camera.
        accept_corr: minimum correlation threshold.
        used_targets: sets of target indices already used (per camera).

    Returns:
        List of 3-tupel matches.
    """
    num_cams = 4
    triplets = []

    # Build lookup
    pair_lookup = {}
    for cam1 in range(num_cams):
        for cam2 in range(num_cams):
            if cam1 == cam2:
                continue
            key = (cam1, cam2)
            pair_lookup[key] = {}
            for corr_entry in adj_lists[cam1][cam2]:
                p1 = corr_entry.p1
                for idx, p2 in enumerate(corr_entry.p2):
                    if p1 not in pair_lookup[key]:
                        pair_lookup[key][p1] = {}
                    pair_lookup[key][p1][p2] = corr_entry.corr[idx]

    # Try all 3-camera combinations
    cam_combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    for c1, c2, c3 in cam_combos:
        for p1 in range(target_counts[c1]):
            if p1 in used_targets[c1]:
                continue

            if (c1, c2) not in pair_lookup or p1 not in pair_lookup[(c1, c2)]:
                continue

            for p2 in pair_lookup[(c1, c2)][p1]:
                if p2 in used_targets[c2]:
                    continue

                if (c1, c3) not in pair_lookup or p1 not in pair_lookup[(c1, c3)]:
                    continue

                for p3 in pair_lookup[(c1, c3)][p1]:
                    if p3 in used_targets[c3]:
                        continue

                    # Check c2-c3 consistency
                    if p3 not in pair_lookup.get((c2, c3), {}).get(p2, {}):
                        continue

                    avg_corr = (
                        pair_lookup[(c1, c2)][p1][p2]
                        + pair_lookup[(c1, c3)][p1][p3]
                        + pair_lookup[(c2, c3)][p2][p3]
                    ) / 3.0

                    if avg_corr >= accept_corr:
                        p = [-1, -1, -1, -1]
                        p[c1] = p1
                        p[c2] = p2
                        p[c3] = p3
                        triplets.append(NTupel(p=p, corr=avg_corr))

    return triplets


def consistent_pair_matching(
    adj_lists: list[list[list[Correspond]]],
    target_counts: list[int],
    accept_corr: float,
    used_targets: list[set[int]],
) -> list[NTupel]:
    """Find unambiguous 2-camera pairs (only one candidate).

    Args:
        adj_lists: pairwise adjacency lists.
        target_counts: number of targets per camera.
        accept_corr: minimum correlation threshold.
        used_targets: sets of already-used target indices.

    Returns:
        List of 2-tupel matches.
    """
    num_cams = 4
    pairs = []

    for cam1 in range(num_cams):
        for cam2 in range(cam1 + 1, num_cams):
            for corr_entry in adj_lists[cam1][cam2]:
                p1 = corr_entry.p1
                if p1 in used_targets[cam1]:
                    continue

                if len(corr_entry.p2) != 1:
                    continue

                p2 = corr_entry.p2[0]
                if p2 in used_targets[cam2]:
                    continue

                # Check reverse: cam2->cam1 should also have only 1 candidate
                reverse_match = None
                for rev_entry in adj_lists[cam2][cam1]:
                    if rev_entry.p1 == p2:
                        reverse_match = rev_entry
                        break

                if reverse_match is None or len(reverse_match.p2) != 1:
                    continue
                if reverse_match.p2[0] != p1:
                    continue

                corr = corr_entry.corr[0]
                if corr >= accept_corr:
                    p = [-1, -1, -1, -1]
                    p[cam1] = p1
                    p[cam2] = p2
                    pairs.append(NTupel(p=p, corr=corr))

    return pairs


def take_best_candidates(
    src: list[NTupel],
    num_cams: int,
) -> list[NTupel]:
    """Sort by correlation and greedily select non-overlapping matches.

    Args:
        src: unsorted list of n-tupel candidates.
        num_cams: number of cameras.

    Returns:
        Sorted, non-overlapping matches.
    """
    # Sort by correlation descending
    sorted_matches = sorted(src, key=lambda m: -m.corr)

    used = [set() for _ in range(num_cams)]
    result = []

    for match in sorted_matches:
        # Check if any target is already used
        conflict = False
        for cam in range(num_cams):
            if match.p[cam] >= 0 and match.p[cam] in used[cam]:
                conflict = True
                break

        if conflict:
            continue

        # Accept this match
        for cam in range(num_cams):
            if match.p[cam] >= 0:
                used[cam].add(match.p[cam])
        result.append(match)

    return result


def correspondences(
    targets: list[list[dict]],
    corrected: list[list[tuple[float, float]]],
    vpar: dict,
    cpar: dict,
    calibrations: list[dict],
    mm_params: dict,
    accept_corr: float = 0.0,
) -> list[NTupel]:
    """Full correspondence matching pipeline.

    Priority order: quadruplets > triplets > pairs.

    Args:
        targets: per-camera target lists.
        corrected: per-camera corrected coordinates.
        vpar: volume parameters.
        cpar: control parameters.
        calibrations: per-camera calibrations.
        mm_params: multimedia parameters.
        accept_corr: minimum correlation threshold.

    Returns:
        List of accepted n-tupel correspondences.
    """
    num_cams = len(targets)
    target_counts = [len(t) for t in targets]

    # Stage 1: Build pairwise adjacencies
    adj_lists = match_pairs(targets, corrected, vpar, cpar, calibrations, mm_params)

    # Stage 2: Find quadruplets (4-camera matches)
    quadruplets = four_camera_matching(adj_lists, target_counts[0], accept_corr)

    # Track used targets
    used_targets = [set() for _ in range(num_cams)]
    for match in quadruplets:
        for cam in range(num_cams):
            if match.p[cam] >= 0:
                used_targets[cam].add(match.p[cam])

    # Stage 3: Find triplets
    triplets = three_camera_matching(adj_lists, target_counts, accept_corr, used_targets)

    for match in triplets:
        for cam in range(num_cams):
            if match.p[cam] >= 0:
                used_targets[cam].add(match.p[cam])

    # Stage 4: Find consistent pairs
    pairs = consistent_pair_matching(adj_lists, target_counts, accept_corr, used_targets)

    # Combine and deduplicate
    all_matches = quadruplets + triplets + pairs

    # Select best non-overlapping
    return take_best_candidates(all_matches, num_cams)
