#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Tests for the Tracker class

Created on Mon Apr 24 10:57:01 2017

@author: yosef
"""

import unittest
import yaml
import shutil
import os
from optv.tracker import Tracker
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, SequenceParams

framebuf_naming = {
    "corres": b"test_data/track/res/particles",
    "linkage": b"test_data/track/res/linkage",
    "prio": b"test_data/track/res/whatever",
}


class TestTracker(unittest.TestCase):
    def setUp(self):
        with open(b"test_data/track/conf.yaml") as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        seq_cfg = yaml_conf["sequence"]

        self.cals = []
        img_base = []
        print((yaml_conf["cameras"]))
        for cix, cam_spec in enumerate(yaml_conf["cameras"]):
            cam_spec.setdefault(b"addpar_file", None)
            cal = Calibration()
            cal.from_file(
                cam_spec["ori_file"].encode(), cam_spec["addpar_file"].encode()
            )
            self.cals.append(cal)
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))

        self.cpar = ControlParams(len(yaml_conf["cameras"]), **yaml_conf["scene"])
        self.vpar = VolumeParams(**yaml_conf["correspondences"])
        self.tpar = TrackingParams(**yaml_conf["tracking"])
        self.spar = SequenceParams(
            image_base=img_base, frame_range=(seq_cfg["first"], seq_cfg["last"])
        )

        self.tracker = Tracker(
            self.cpar, self.vpar, self.tpar, self.spar, self.cals, framebuf_naming
        )

    def test_forward(self):
        """Manually running a full forward tracking run."""
        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward():
            # print(f"step is {self.tracker.current_step()}\n")
            # print(self.tracker.current_step() > last_step)
            self.assertTrue(self.tracker.current_step() > last_step)
            with open("test_data/track/res/linkage.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                if last_step == 10003:
                    self.assertTrue(lines[0] == "0\n")
                else:
                    self.assertTrue(lines[0] == "1\n")
            last_step += 1
        self.tracker.finalize()

    def test_forward_3d(self):
        """Manually running a full forward tracking run."""
        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward_3d():
            # print(f"step is {self.tracker.current_step()}\n")
            # print(self.tracker.current_step() > last_step)
            self.assertGreater(self.tracker.current_step(), last_step)
            with open("test_data/track/res/linkage.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                if last_step == 10003:
                    self.assertTrue(lines[0] == "0\n")
                else:
                    self.assertTrue(lines[0] == "1\n")
            last_step += 1
        self.tracker.finalize()

    def test_full_forward(self):
        """Automatic full forward tracking run."""
        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_forward_3d(self):
        """Automatic full forward tracking run."""
        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")
        self.tracker.full_forward_3d()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_forward_3d_output_matches_reference(self):
        """Verify track3d output matches reference res_orig/ files."""
        import numpy as np

        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")
        self.tracker.full_forward_3d()

        for step in range(10001, 10005):
            out_file = f"test_data/track/res/particles.{step}"
            ref_file = f"test_data/track/res_orig/particles.{step}"

            self.assertTrue(os.path.exists(out_file), f"Missing output: {out_file}")

            with open(out_file) as f_out, open(ref_file) as f_ref:
                out_lines = f_out.readlines()
                ref_lines = f_ref.readlines()

                # First line is particle count (-1 treated as 0)
                out_count = int(out_lines[0].strip())
                ref_count = int(ref_lines[0].strip())
                if out_count < 0:
                    out_count = 0
                if ref_count < 0:
                    ref_count = 0
                self.assertEqual(
                    out_count,
                    ref_count,
                    f"Particle count mismatch at step {step}",
                )

                # Compare positions
                out_parts = [list(map(float, l.split())) for l in out_lines[1:]]
                ref_parts = [list(map(float, l.split())) for l in ref_lines[1:]]

                for i, (o, r) in enumerate(zip(out_parts, ref_parts)):
                    np.testing.assert_allclose(
                        o,
                        r,
                        atol=1e-5,
                        err_msg=f"Position mismatch at step {step}, particle {i}",
                    )

    def test_forward_3d_step_by_step_output(self):
        """Verify step_forward_3d produces correct per-step output."""
        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward_3d():
            self.assertGreater(self.tracker.current_step(), last_step)

            # Verify output file exists for completed step
            out_file = f"test_data/track/res/particles.{last_step}"
            self.assertTrue(
                os.path.exists(out_file),
                f"Missing output after step {last_step}",
            )

            last_step += 1
        self.tracker.finalize()

    def test_full_backward(self):
        """Automatic full backward correction phase."""
        shutil.copytree("test_data/track/res_orig/", "test_data/track/res/")
        self.tracker.full_forward()
        self.tracker.full_backward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_tracker_string_handling(self):
        """Test that Tracker handles both strings and bytes correctly"""
        # Using regular strings - will be encoded automatically
        naming_strings = {
            "corres": "res/rt_is",
            "linkage": "res/ptv_is",
            "prio": "res/added",
        }
        tracker1 = Tracker(
            self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_strings
        )

        # Using bytes directly - will be passed through
        naming_bytes = {
            "corres": b"res/rt_is",
            "linkage": b"res/ptv_is",
            "prio": b"res/added",
        }
        tracker2 = Tracker(
            self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_bytes
        )

        # Using mixed - both will work
        naming_mixed = {
            "corres": "res/rt_is",  # string
            "linkage": b"res/ptv_is",  # bytes
            "prio": "res/added",  # string
        }
        tracker3 = Tracker(
            self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_mixed
        )

        # Using partial dict - missing keys will use defaults
        naming_partial = {
            "corres": "res/rt_is"  # only specify what you need to change
        }
        tracker4 = Tracker(
            self.cpar, self.vpar, self.tpar, self.spar, self.cals, naming_partial
        )

    def tearDown(self):
        if os.path.exists("test_data/track/res/"):
            shutil.rmtree("test_data/track/res/")


if __name__ == "__main__":
    unittest.main()
