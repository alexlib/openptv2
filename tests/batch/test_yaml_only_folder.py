"""YAML-only operation tests.

openptv2 is YAML-only at runtime: an experiment folder needs only its
``parameters_*.yaml`` — no legacy ``parameters/`` .par directory. These tests
lock that in and also verify the new ``highpass_size`` YAML parameter is
actually honored by the preprocessing.
"""

import os
import shutil

import numpy as np
from imageio.v3 import imread

from openptv2.batch import pyptv_batch
from openptv2.storage import RunStore, resolve_store_path


def test_sequence_runs_without_parameters_dir(small_dir, small_yaml, tmp_path):
    """Sequence must run on a folder that has ONLY the YAML (no parameters/)."""
    work = tmp_path / "yaml_only"
    shutil.copytree(small_dir, work)
    # Remove every legacy .par artifact — YAML must be sufficient.
    for d in work.glob("parameters*"):
        if d.is_dir():
            shutil.rmtree(d)
    assert not any(work.glob("parameters*/")), "no legacy .par dir should remain"
    assert (work / "parameters_Run1.yaml").exists()

    frame = 10001
    pyptv_batch.main(work / "parameters_Run1.yaml", frame, frame, mode="sequence")

    store = RunStore(resolve_store_path(work / "res"), mode="r")
    assert store.has_correspondences(frame), (
        "sequence must produce correspondences from YAML alone"
    )
    pos, _ = store.read_correspondences(frame)
    assert len(pos) > 0, "YAML-only sequence found no correspondences"


def test_highpass_size_is_honored(small_dir, small_yaml):
    """A different highpass_size must change the high-pass output.

    Regression guard for the bug class where the wrong value (cr_sz) was fed
    as the high-pass kernel: proves the YAML parameter actually drives it.
    """
    from openptv2.gui.experiment import Experiment
    from openptv2.gui.ptv import py_pre_processing_c

    cwd0 = os.getcwd()
    os.chdir(small_dir)
    try:
        exp = Experiment()
        exp.pm.from_yaml(small_yaml)
        num_cams = exp.pm.num_cams
        ptv = dict(exp.pm.get_parameter("ptv"))

        base = exp.pm.get_parameter("sequence")["base_name"][0]
        if isinstance(base, bytes):
            base = base.decode()
        img = imread(base % 10001)
        if img.ndim > 2:
            from skimage.color import rgb2gray
            from skimage.util import img_as_ubyte

            img = img_as_ubyte(rgb2gray(img))
        imgs = [img] * num_cams

        hp_small = py_pre_processing_c(num_cams, imgs, {**ptv, "highpass_size": 3})
        hp_large = py_pre_processing_c(num_cams, imgs, {**ptv, "highpass_size": 25})
    finally:
        os.chdir(cwd0)

    assert not np.array_equal(np.asarray(hp_small[0]), np.asarray(hp_large[0])), (
        "highpass_size had no effect — the YAML parameter is not wired through"
    )
