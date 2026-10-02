"""Bubble detection sequence plugin for faint glare-points and bubbles (e.g. HFSB).

Replaces standard global-threshold connected-components target recognition with
locally-adaptive z-score contrast scoring (against sliding-window local mean/std)
and radius-hop glare-fragment merging via compiled `detect_bubbles_fast`.

Designed for large-distance measurements (e.g. Ilmenau Rayleigh-Bénard convection
with helium-filled soap bubbles) where bubbles appear as faint pairs of specular
reflections rather than solid Gaussian blobs.

Configured via the optional `bubble_detection:` block in parameters_*.yaml:
    plugins:
      selected_sequence: bubble_sequence
    bubble_detection:
      win: 25           # local background window in pixels (default 25)
      z_thresh: 7.0     # local-sigma cutoff above background (default 7.0)
      merge_radius: 4   # max pixel gap bridged between glare spots (default 4)
      nnmin: 1          # min pixel count per blob (default 1)
      nnmax: 5000       # max pixel count per blob (default 5000)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openptv2.algorithms.bubble_detection import detect_bubbles_fast
from openptv2.algorithms.tracking_frame_buf import Target, TargetArray
from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.image_processing import preprocess_image
from openptv2.orientation import point_positions


class Sequence:
    """Sequence plugin that uses local z-score bubble detection and glare-merging.

    Connection to the ptv module is given via ``self.ptv`` and connection to
    the active experiment via ``self.exp``, both injected by the loader.
    """

    def __init__(self, ptv: Any = None, exp: Any = None) -> None:
        self.ptv = ptv
        self.exp = exp

    def do_sequence(self) -> None:
        if self.exp is None or self.ptv is None:
            raise ValueError("No experiment or ptv module provided")

        pm, num_cams, cpar, spar, vpar, tpar, cals = (
            self.ptv._resolve_experiment_params(self.exp)
        )

        bubble_cfg = (
            pm.parameters.get("bubble_detection", {})
            if hasattr(pm, "parameters") and isinstance(pm.parameters, dict)
            else {}
        )
        if not isinstance(bubble_cfg, dict):
            try:
                bubble_cfg = pm.get_parameter("bubble_detection")
            except ValueError:
                bubble_cfg = {}

        win = int(bubble_cfg.get("win", 25))
        z_thresh = float(bubble_cfg.get("z_thresh", 7.0))
        merge_radius = int(bubble_cfg.get("merge_radius", 4))
        nnmin = int(bubble_cfg.get("nnmin", 1))
        nnmax = int(bubble_cfg.get("nnmax", 5000))
        filter_hp = int(bubble_cfg.get("filter_hp", 0))
        lowpass_dim = int(bubble_cfg.get("lowpass_dim", 1))

        first_frame = spar.get_first()
        last_frame = spar.get_last()
        img_base_names = [spar.get_img_base_name(i) for i in range(num_cams)]
        short_file_bases = self.exp.target_filenames
        self.ptv._ensure_target_output_writable(short_file_bases)
        store = self.ptv._open_run_store(self.exp)

        print(
            f"BubbleSequence: frames {first_frame}..{last_frame} across {num_cams} camera(s) "
            f"(win={win}, z_thresh={z_thresh:g}, merge_radius={merge_radius})"
        )

        for frame in range(first_frame, last_frame + 1):
            frame_images = self.ptv.read_frame_images(
                pm, img_base_names, num_cams, frame
            )
            detections = []
            corrected = []
            for i_cam in range(num_cams):
                img = frame_images[i_cam]
                hp = np.asarray(
                    preprocess_image(
                        img, filter_hp=filter_hp, cpar=cpar, lowpass_dim=lowpass_dim
                    )
                ).astype(np.float64)

                xs, ys, npix = detect_bubbles_fast(
                    hp,
                    win=win,
                    z_thresh=z_thresh,
                    merge_radius=merge_radius,
                    nnmin=nnmin,
                    nnmax=nnmax,
                )
                n_det = len(xs)
                targs = TargetArray(n_det)
                for i in range(n_det):
                    targs[i] = Target(
                        pnr=i,
                        x=float(xs[i]),
                        y=float(ys[i]),
                        n=int(npix[i]),
                        nx=1,
                        ny=1,
                        sumg=int(npix[i]),
                        tnr=-1,
                    )

                if n_det > 0:
                    targs.sort_y()

                detections.append(targs)
                matched_coords = MatchedCoords(targs, cpar, cals[i_cam])
                corrected.append(matched_coords)

            sorted_pos, sorted_corresp, _ = correspondences(
                detections, corrected, cals, vpar, cpar
            )

            for i_cam in range(num_cams):
                self.ptv.write_targets(
                    detections[i_cam],
                    short_file_bases[i_cam],
                    frame,
                    store=store,
                    cam_idx=i_cam,
                )

            print(
                f"Frame {frame} had {[s.shape[1] for s in sorted_pos]} correspondences."
            )

            sorted_pos = np.concatenate(sorted_pos, axis=1)
            sorted_corresp = np.concatenate(sorted_corresp, axis=1)

            flat = np.array(
                [
                    corr.get_by_pnrs(corresp)
                    for corr, corresp in zip(corrected, sorted_corresp)
                ]
            )
            pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

            if len(cals) < 4:
                print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]))
                print_corresp[: len(cals), :] = sorted_corresp
            else:
                print_corresp = sorted_corresp

            if store is not None:
                store.write_correspondences(
                    frame=frame, pos_3d=pos, cam_target_ids=print_corresp.T
                )
