# -*- coding: utf-8 -*-
"""
Tests for the Tracker with add_particles using Burgers vortex data
with ground truth

Created on Mon Apr 24 10:57:01 2017

@author: alexlib
"""

import unittest
import yaml
import os
import shutil
from optv.tracker import Tracker
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, SequenceParams

framebuf_naming = {
    "corres": b"test_data/burgers/res/rt_is",
    "linkage": b"test_data/burgers/res/ptv_is",
    "prio": b"test_data/burgers/res/whatever",
}


class TestTracker(unittest.TestCase):
    def setUp(self):
        with open("test_data/burgers/conf.yaml") as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        seq_cfg = yaml_conf["sequence"]

        cals = []
        img_base = []
        print(yaml_conf["cameras"])
        for cix, cam_spec in enumerate(yaml_conf["cameras"]):
            cam_spec.setdefault("addpar_file", None)
            cal = Calibration()
            cal.from_file(
                cam_spec["ori_file"].encode(),
                cam_spec["addpar_file"].encode() if cam_spec["addpar_file"] else None,
            )
            cals.append(cal)
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))

        cpar = ControlParams(len(yaml_conf["cameras"]), **yaml_conf["scene"])
        vpar = VolumeParams(**yaml_conf["correspondences"])
        tpar = TrackingParams(**yaml_conf["tracking"])
        spar = SequenceParams(
            image_base=img_base, frame_range=(seq_cfg["first"], seq_cfg["last"])
        )

        self.tracker = Tracker(cpar, vpar, tpar, spar, cals, framebuf_naming)

    def copy_data_dirs(self):
        for dst, src in [
            ("test_data/burgers/res/", "test_data/burgers/res_orig/"),
            ("test_data/burgers/img/", "test_data/burgers/img_orig/"),
        ]:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    def test_forward(self):
        """Manually running a full forward tracking run."""
        # path = 'test_data/burgers/res'
        # try:
        #     os.mkdir(path)
        # except OSError:
        #     print("Creation of the directory %s failed" % path)
        # else:
        #     print("Successfully created the directory %s " % path)

        self.copy_data_dirs()

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward():
            self.assertTrue(self.tracker.current_step() > last_step)
            with open("test_data/burgers/res/rt_is.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                # print(lines)
                if last_step == 10003:
                    self.assertTrue(lines[0] == "4\n")
                else:
                    self.assertTrue(lines[0] == "5\n")
            last_step += 1
        self.tracker.finalize()

    def test_forward_3d(self):
        """Manually running a full forward tracking run."""
        # path = 'test_data/burgers/res'
        # try:
        #     os.mkdir(path)
        # except OSError:
        #     print("Creation of the directory %s failed" % path)
        # else:
        #     print("Successfully created the directory %s " % path)

        self.copy_data_dirs()

        self.tracker.restart()
        last_step = 10001
        while self.tracker.step_forward_3d():
            self.assertTrue(self.tracker.current_step() > last_step)
            with open("test_data/burgers/res/rt_is.%d" % last_step) as f:
                lines = f.readlines()
                # print(last_step,lines[0])
                # print(lines)
                if last_step == 10003:
                    self.assertTrue(lines[0] == "4\n")
                else:
                    self.assertTrue(lines[0] == "5\n")
            last_step += 1
        self.tracker.finalize()

    def test_full_forward(self):
        """Automatic full forward tracking run."""
        # os.mkdir('test_data/burgers/res')
        self.copy_data_dirs()
        self.tracker.full_forward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_forward_3d(self):
        """Automatic full forward tracking run."""
        # os.mkdir('test_data/burgers/res')
        self.copy_data_dirs()
        self.tracker.full_forward_3d()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def test_full_backward(self):
        """Automatic full backward correction phase."""
        self.copy_data_dirs()
        self.tracker.full_forward()
        self.tracker.full_backward()
        # if it passes without error, we assume it's ok. The actual test is in
        # the C code.

    def tearDown(self):
        if os.path.exists("test_data/burgers/res/"):
            shutil.rmtree("test_data/burgers/res/")
        if os.path.exists("test_data/burgers/img/"):
            shutil.rmtree("test_data/burgers/img/")
            # print("there is a /res folder\n")
            # pass


if __name__ == "__main__":
    unittest.main()
