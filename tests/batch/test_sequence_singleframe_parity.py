"""Parity regression test: the sequence/batch pipeline and the single-frame
(GUI) detection+correspondence pipeline must produce the SAME correspondences
for the same frame.

This guards the whole class of "single-frame works, sequence is off" bugs.
The concrete case that motivated it: the sequence loop high-passed with
``tpar.get_cross_size()`` (=2, the target cross-marking size) while the
single-frame path used the correct default kernel (25), so the same frame gave
many correspondences interactively but almost none in the sequence.

Plain ``count > 0`` smoke tests cannot catch this — a degraded pipeline still
returns *some* correspondences on a clean dataset. Only comparing the two
pipelines head-to-head exposes the divergence.
"""

import os

import numpy as np
import pytest
from imageio.v3 import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from openptv2.batch import pyptv_batch

FRAME = 10001  # a frame with real 3D structure in test_cavity_small


def _rt_is_count(path) -> int:
    return int(path.read_text().strip().splitlines()[0])


def _single_frame_reference(small_dir, small_yaml, frame):
    """Correspondence count via the single-frame (GUI) pipeline helpers.

    Mirrors exactly what the GUI does for one frame: load images, high-pass
    with py_pre_processing_c, detect with py_detection_proc_c, then match with
    correspondences(). Returns the total number of 3D correspondences.
    """
    from openptv2.gui.experiment import Experiment
    from openptv2.gui.ptv import (
        py_detection_proc_c,
        py_pre_processing_c,
        py_start_proc_c,
    )
    from openptv2.correspondences import correspondences

    cwd0 = os.getcwd()
    os.chdir(small_dir)
    try:
        exp = Experiment()
        exp.pm.from_yaml(small_yaml)
        num_cams = exp.pm.num_cams
        cpar, spar, vpar, _track, _tpar, cals, _epar = py_start_proc_c(exp.pm)
        ptv_params = exp.pm.get_parameter("ptv")
        target_params = {"targ_rec": exp.pm.get_parameter("targ_rec")}

        imgs = []
        for i in range(num_cams):
            base = spar.get_img_base_name(i)
            if isinstance(base, bytes):
                base = base.decode()
            im = imread(base % frame)
            if im.ndim > 2:
                im = rgb2gray(im)
            if im.dtype != np.uint8:
                im = img_as_ubyte(im)
            imgs.append(im)

        hp = py_pre_processing_c(num_cams, imgs, ptv_params)
        dets, corr = py_detection_proc_c(num_cams, hp, ptv_params, target_params)
        sorted_pos, _, _ = correspondences(dets, corr, cals, vpar, cpar)
        return int(sum(s.shape[1] for s in sorted_pos))
    finally:
        os.chdir(cwd0)


def test_sequence_matches_single_frame(small_dir, small_yaml):
    ref_total = _single_frame_reference(small_dir, small_yaml, FRAME)

    res = small_dir / "res"
    for f in res.glob(f"rt_is.{FRAME}"):
        f.unlink()
    pyptv_batch.main(small_yaml, FRAME, FRAME, mode="sequence")
    seq_total = _rt_is_count(res / f"rt_is.{FRAME}")

    assert ref_total > 0, "single-frame reference produced no correspondences"
    assert seq_total == ref_total, (
        f"sequence produced {seq_total} 3D points but the single-frame pipeline "
        f"produced {ref_total} for frame {FRAME}: the two pipelines diverged "
        f"(e.g. a high-pass kernel mismatch between them)."
    )
