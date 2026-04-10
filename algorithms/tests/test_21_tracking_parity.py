"""
Step-by-step parity tests: Python algorithms vs Cython/C bindings.

Each test isolates ONE assumption and verifies that the Python
algorithms module produces the same output as the Cython optv bindings
for the SAME input.  Uses the small 'track' test dataset (1 particle,
4 cameras, 5 frames, deterministic straight-line motion).

Test order matches the pipeline:
  1. Parameter reading
  2. Calibration reading
  3. Target / frame reading
  4. Coordinate transforms (img_coord, metric ↔ pixel, dist_to_flat)
  5. searchquader (search region)
  6. candsearch_in_pix (candidate search)
  7. Full tracker: trackcorr_c_loop output files
  8. Full tracker: track3d_loop output files
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
TRACK_DATA = HERE.parent.parent / "test_data" / "track"
CONF_YAML = TRACK_DATA / "conf.yaml"


def _have_optv():
    """Return True if the Cython optv package is importable."""
    try:
        from optv.tracker import Tracker  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_optv = pytest.mark.skipif(not _have_optv(), reason="optv (Cython) not installed")


@pytest.fixture(scope="module")
def yaml_conf():
    with open(CONF_YAML) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PARAMETER READING PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterParity:
    """Verify that Python and Cython build identical parameter objects."""

    @skip_no_optv
    def test_control_par(self, yaml_conf):
        """ControlPar: image size, pixel size, multimedia."""
        from optv.parameters import ControlParams as CyControlParams
        from algorithms.parameters import ControlPar as PyControlPar, MultimediaPar

        scene = yaml_conf["scene"]
        num_cams = len(yaml_conf["cameras"])

        # --- Cython ---
        cy = CyControlParams(num_cams, **scene)

        # --- Python ---
        mm = MultimediaPar(
            nlay=len(scene.get("wall_ns", [1])),
            n1=scene.get("cam_side_n", 1.0),
            n2=scene.get("wall_ns", [1.0]),
            d=scene.get("wall_thicks", [0.0]),
            n3=scene.get("object_side_n", 1.0),
        )
        py = PyControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0],
            imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0],
            pix_y=scene["pixel_size"][1],
            mm=mm,
        )

        # --- Compare ---
        assert py.num_cams == num_cams
        cy_imsize = cy.get_image_size()
        assert (py.imx, py.imy) == (cy_imsize[0], cy_imsize[1]), \
            f"image_size: py=({py.imx},{py.imy}) cy={cy_imsize}"
        cy_pix = cy.get_pixel_size()
        assert pytest.approx(py.pix_x, abs=1e-8) == cy_pix[0]
        assert pytest.approx(py.pix_y, abs=1e-8) == cy_pix[1]

        # Multimedia
        cy_mm = cy.get_multimedia_params()
        nlay = py.mm.nlay
        assert nlay == cy_mm.get_nlay()
        assert pytest.approx(py.mm.n1) == cy_mm.get_n1()
        assert pytest.approx(py.mm.n3) == cy_mm.get_n3()
        # Cython returns fixed-size (3,) arrays; compare only first nlay elements
        np.testing.assert_allclose(py.mm.n2[:nlay], cy_mm.get_n2()[:nlay], atol=1e-8)
        np.testing.assert_allclose(py.mm.d[:nlay], cy_mm.get_d()[:nlay], atol=1e-8)

    @skip_no_optv
    def test_volume_par(self, yaml_conf):
        """VolumePar: correspondence criteria."""
        from optv.parameters import VolumeParams as CyVolumeParams
        from algorithms.parameters import VolumePar as PyVolumePar

        c = yaml_conf["correspondences"]

        cy = CyVolumeParams(**c)
        py = PyVolumePar(
            x_lay=c["x_span"],
            z_min_lay=[c["z_spans"][i][0] for i in range(len(c["z_spans"]))],
            z_max_lay=[c["z_spans"][i][1] for i in range(len(c["z_spans"]))],
            cn=c.get("pixels_tot", 0),
            cnx=c.get("pixels_x", 0),
            cny=c.get("pixels_y", 0),
            csumg=c.get("ref_gray", 0),
            eps0=c.get("epipolar_band", 0),
            corrmin=c.get("min_correlation", 0),
        )

        np.testing.assert_allclose(py.x_lay, cy.get_X_lay(), atol=1e-8)
        np.testing.assert_allclose(py.z_min_lay, cy.get_Zmin_lay(), atol=1e-8)
        np.testing.assert_allclose(py.z_max_lay, cy.get_Zmax_lay(), atol=1e-8)
        assert pytest.approx(py.cn) == cy.get_cn()
        assert pytest.approx(py.cnx) == cy.get_cnx()
        assert pytest.approx(py.cny) == cy.get_cny()
        assert pytest.approx(py.csumg) == cy.get_csumg()
        assert pytest.approx(py.eps0) == cy.get_eps0()
        assert pytest.approx(py.corrmin) == cy.get_corrmin()

    @skip_no_optv
    def test_tracking_par(self, yaml_conf):
        """TrackPar: velocity, angle, acceleration limits."""
        from optv.parameters import TrackingParams as CyTrackingParams
        from algorithms.parameters import TrackParTuple

        t = yaml_conf["tracking"]

        cy = CyTrackingParams(**t)
        vel = t["velocity_lims"]
        py = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=t["angle_lim"],
            dacc=t["accel_lim"],
            add=t.get("add_particle", 0),
            dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
        )

        assert pytest.approx(py.dvxmin) == cy.get_dvxmin()
        assert pytest.approx(py.dvxmax) == cy.get_dvxmax()
        assert pytest.approx(py.dvymin) == cy.get_dvymin()
        assert pytest.approx(py.dvymax) == cy.get_dvymax()
        assert pytest.approx(py.dvzmin) == cy.get_dvzmin()
        assert pytest.approx(py.dvzmax) == cy.get_dvzmax()
        assert pytest.approx(py.dangle) == cy.get_dangle()
        assert pytest.approx(py.dacc) == cy.get_dacc()

    @skip_no_optv
    def test_sequence_par(self, yaml_conf):
        """SequencePar: frame range and image base names."""
        from optv.parameters import SequenceParams as CySequenceParams
        from algorithms.parameters import SequencePar as PySequencePar

        seq = yaml_conf["sequence"]
        num_cams = len(yaml_conf["cameras"])

        img_base = [seq["targets_template"].format(cam=i + 1) for i in range(num_cams)]

        cy = CySequenceParams(
            image_base=img_base,
            frame_range=(seq["first"], seq["last"]),
        )
        py = PySequencePar(
            img_base_name=img_base,
            first=seq["first"],
            last=seq["last"],
        )

        assert py.first == cy.get_first()
        assert py.last == cy.get_last()
        for cam in range(num_cams):
            cy_base = cy.get_img_base_name(cam)
            if isinstance(cy_base, bytes):
                cy_base = cy_base.decode()
            assert py.img_base_name[cam] == cy_base, \
                f"cam{cam}: py={py.img_base_name[cam]} cy={cy_base}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. CALIBRATION READING PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibrationParity:
    """Verify that Python and Cython read identical calibration data."""

    @skip_no_optv
    def test_calibration_values(self, yaml_conf):
        """All calibration fields must match between engines."""
        from optv.calibration import Calibration as CyCalibration
        from algorithms.calibration import Calibration as PyCalibration, read_calibration

        for cam_spec in yaml_conf["cameras"]:
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")

            cy_cal = CyCalibration()
            cy_cal.from_file(ori.encode(), addpar.encode() if addpar else None)

            py_cal = read_calibration(ori, addpar)

            # Exterior: position and angles
            np.testing.assert_allclose(
                [py_cal.ext_par.x0, py_cal.ext_par.y0, py_cal.ext_par.z0],
                cy_cal.get_pos(),
                atol=1e-6,
                err_msg=f"Position mismatch for {ori}",
            )
            np.testing.assert_allclose(
                [py_cal.ext_par.omega, py_cal.ext_par.phi, py_cal.ext_par.kappa],
                cy_cal.get_angles(),
                atol=1e-6,
                err_msg=f"Angles mismatch for {ori}",
            )

            # Rotation matrix
            np.testing.assert_allclose(
                py_cal.ext_par.dm,
                cy_cal.get_rotation_matrix(),
                atol=1e-6,
                err_msg=f"Rotation matrix mismatch for {ori}",
            )

            # Interior: principal point
            np.testing.assert_allclose(
                [py_cal.int_par.xh, py_cal.int_par.yh, py_cal.int_par.cc],
                cy_cal.get_primary_point(),
                atol=1e-6,
                err_msg=f"Principal point mismatch for {ori}",
            )

            # Glass vector (plain ndarray [vec_x, vec_y, vec_z])
            np.testing.assert_allclose(
                py_cal.glass_par,
                cy_cal.get_glass_vec(),
                atol=1e-6,
                err_msg=f"Glass vector mismatch for {ori}",
            )

            # Distortion (added_par is a plain ndarray [k1, k2, k3, p1, p2, scx, she])
            ap = py_cal.added_par
            np.testing.assert_allclose(
                ap[:3],
                cy_cal.get_radial_distortion(),
                atol=1e-10,
                err_msg=f"Radial distortion mismatch for {ori}",
            )
            np.testing.assert_allclose(
                ap[3:5],
                cy_cal.get_decentering(),
                atol=1e-10,
                err_msg=f"Decentering mismatch for {ori}",
            )
            np.testing.assert_allclose(
                ap[5:7],
                cy_cal.get_affine(),
                atol=1e-10,
                err_msg=f"Affine mismatch for {ori}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TARGET / FRAME READING PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetReadingParity:
    """Verify that targets are read identically."""

    @skip_no_optv
    def test_read_targets_per_camera(self, yaml_conf):
        """read_targets for each camera/frame matches Cython TargetArray."""
        from optv.tracking_framebuf import read_targets as cy_read_targets
        from algorithms.tracking_frame_buf import read_targets as py_read_targets

        seq = yaml_conf["sequence"]
        num_cams = len(yaml_conf["cameras"])
        img_bases = [seq["targets_template"].format(cam=i + 1) for i in range(num_cams)]

        for frame in range(seq["first"], seq["last"] + 1):
            for cam in range(num_cams):
                cy_targs = cy_read_targets(img_bases[cam], frame)
                py_targs = py_read_targets(img_bases[cam], frame)

                assert len(py_targs) == len(cy_targs), \
                    f"cam{cam} frame{frame}: py={len(py_targs)} cy={len(cy_targs)}"

                for i in range(len(py_targs)):
                    cy_t = cy_targs[i]
                    py_t = py_targs[i]

                    assert py_t.pnr == cy_t.pnr(), \
                        f"cam{cam} frame{frame} targ{i}: pnr py={py_t.pnr} cy={cy_t.pnr()}"
                    assert pytest.approx(py_t.x, abs=1e-4) == cy_t.pos()[0], \
                        f"cam{cam} frame{frame} targ{i}: x py={py_t.x} cy={cy_t.pos()[0]}"
                    assert pytest.approx(py_t.y, abs=1e-4) == cy_t.pos()[1], \
                        f"cam{cam} frame{frame} targ{i}: y py={py_t.y} cy={cy_t.pos()[1]}"
                    assert py_t.tnr == cy_t.tnr(), \
                        f"cam{cam} frame{frame} targ{i}: tnr py={py_t.tnr} cy={cy_t.tnr()}"
                    pxcnt = cy_t.count_pixels()
                    assert py_t.n == pxcnt[0], \
                        f"cam{cam} frame{frame} targ{i}: n py={py_t.n} cy={pxcnt[0]}"
                    assert py_t.sumg == cy_t.sum_grey_value(), \
                        f"cam{cam} frame{frame} targ{i}: sumg py={py_t.sumg} cy={cy_t.sum_grey_value()}"

    @skip_no_optv
    def test_read_frame_positions(self, yaml_conf):
        """Verify read_path_frame reads 3D positions that match the particles file."""
        from algorithms.tracking_frame_buf import read_path_frame

        seq = yaml_conf["sequence"]
        corres_base = str(TRACK_DATA / "res_orig" / "particles")

        for frame in range(seq["first"], seq["last"] + 1):
            py_cor, py_path = read_path_frame(corres_base, "", "", frame)

            # Read reference particles file directly
            # Format: "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n"
            # Fields: id x y z cam0 cam1 cam2 cam3
            pfile = TRACK_DATA / "res_orig" / f"particles.{frame}"
            lines = pfile.read_text().strip().splitlines()
            nparts = int(lines[0])

            assert len(py_cor) == nparts, \
                f"Frame {frame}: expected {nparts} particles, got {len(py_cor)}"
            assert len(py_path) == nparts

            for i in range(nparts):
                vals = lines[1 + i].split()
                # id, x, y, z, cam0, cam1, cam2, cam3
                ref_x = [float(vals[1]), float(vals[2]), float(vals[3])]

                np.testing.assert_allclose(
                    py_path[i].x, ref_x, atol=1e-5,
                    err_msg=f"Frame {frame} particle {i}: position differs",
                )


# ═══════════════════════════════════════════════════════════════════════════
# 4. COORDINATE TRANSFORMS PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestCoordinateTransformsParity:
    """Verify that img_coord, metric_to_pixel, pixel_to_metric, dist_to_flat match."""

    @skip_no_optv
    def test_img_coord(self, yaml_conf):
        """img_coord: 3D → 2D metric projection."""
        from optv.calibration import Calibration as CyCalibration
        from optv.parameters import ControlParams as CyControlParams
        from optv.imgcoord import image_coordinates as cy_image_coordinates

        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar
        from algorithms.imgcoord import img_coord as py_img_coord

        scene = yaml_conf["scene"]
        num_cams = len(yaml_conf["cameras"])

        cy_cpar = CyControlParams(num_cams, **scene)
        mm = MultimediaPar(
            nlay=len(scene.get("wall_ns", [1])),
            n1=scene.get("cam_side_n", 1.0),
            n2=scene.get("wall_ns", [1.0]),
            d=scene.get("wall_thicks", [0.0]),
            n3=scene.get("object_side_n", 1.0),
        )

        # Test points: origin and a few others
        test_points = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [5.0, 3.0, -2.0],
            [-10.0, 7.0, 15.0],
        ])

        for cam_spec in yaml_conf["cameras"]:
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")

            cy_cal = CyCalibration()
            cy_cal.from_file(ori.encode(), addpar.encode() if addpar else None)
            cy_mm = cy_cpar.get_multimedia_params()

            py_cal = read_calibration(ori, addpar)

            # Cython (array API)
            cy_coords = cy_image_coordinates(test_points, cy_cal, cy_mm)

            # Python (single-point API)
            for i, pt in enumerate(test_points):
                py_x, py_y = py_img_coord(pt, py_cal, mm)
                np.testing.assert_allclose(
                    [py_x, py_y], cy_coords[i], atol=1e-6,
                    err_msg=f"img_coord mismatch for {ori}, point {pt}",
                )

    @skip_no_optv
    def test_metric_to_pixel(self, yaml_conf):
        """metric_to_pixel and pixel_to_metric round-trip."""
        from optv.parameters import ControlParams as CyControlParams
        from optv.transforms import convert_arr_metric_to_pixel as cy_m2p
        from optv.transforms import convert_arr_pixel_to_metric as cy_p2m

        from algorithms.parameters import ControlPar, MultimediaPar
        from algorithms.trafo import metric_to_pixel, pixel_to_metric

        scene = yaml_conf["scene"]
        num_cams = len(yaml_conf["cameras"])

        cy_cpar = CyControlParams(num_cams, **scene)
        py_cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0],
            imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0],
            pix_y=scene["pixel_size"][1],
            mm=MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1),
        )

        # Test metric coordinates
        metric_pts = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [-3.5, 4.2],
            [5.0, -1.0],
        ])

        cy_pixels = cy_m2p(metric_pts, cy_cpar)

        for i, (mx, my) in enumerate(metric_pts):
            py_px, py_py = metric_to_pixel(mx, my, py_cpar)
            np.testing.assert_allclose(
                [py_px, py_py], cy_pixels[i], atol=1e-4,
                err_msg=f"metric_to_pixel mismatch: ({mx},{my})",
            )

        # Reverse
        cy_metric = cy_p2m(cy_pixels, cy_cpar)
        for i in range(len(cy_pixels)):
            py_mx, py_my = pixel_to_metric(cy_pixels[i][0], cy_pixels[i][1], py_cpar)
            np.testing.assert_allclose(
                [py_mx, py_my], cy_metric[i], atol=1e-6,
                err_msg=f"pixel_to_metric mismatch: pixel {cy_pixels[i]}",
            )

    @skip_no_optv
    def test_dist_to_flat(self, yaml_conf):
        """dist_to_flat (iterative distortion correction)."""
        from optv.calibration import Calibration as CyCalibration
        from optv.transforms import distorted_to_flat as cy_dist_to_flat

        from algorithms.calibration import read_calibration
        from algorithms.trafo import dist_to_flat as py_dist_to_flat

        # Test with metric coords that include some distortion
        metric_pts = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [-3.5, 4.2],
        ])

        for cam_spec in yaml_conf["cameras"]:
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")

            cy_cal = CyCalibration()
            cy_cal.from_file(ori.encode(), addpar.encode() if addpar else None)
            py_cal = read_calibration(ori, addpar)

            cy_flat = cy_dist_to_flat(metric_pts, cy_cal, tol=0.0001)

            for i, (mx, my) in enumerate(metric_pts):
                py_fx, py_fy = py_dist_to_flat(mx, my, py_cal, 0.0001)
                np.testing.assert_allclose(
                    [py_fx, py_fy], cy_flat[i], atol=1e-5,
                    err_msg=f"dist_to_flat mismatch for {ori}, point ({mx},{my})",
                )


# ═══════════════════════════════════════════════════════════════════════════
# 5. SEARCHQUADER PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchquaderParity:
    """searchquader not exposed by Cython, so test Python vs known values.

    We validate by projecting the 8 corners of the 3D search box and checking
    the pixel distances are geometrically consistent.
    """

    def test_searchquader_symmetry(self, yaml_conf):
        """For a point at origin with symmetric cameras, search quader
        should produce reasonable, non-zero, non-negative values."""
        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar, TrackParTuple
        from algorithms.track import searchquader

        scene = yaml_conf["scene"]
        tracking = yaml_conf["tracking"]
        vel = tracking["velocity_lims"]
        num_cams = len(yaml_conf["cameras"])

        mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
        cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0], imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0], pix_y=scene["pixel_size"][1],
            mm=mm,
        )
        tpar = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=0, dsumg=0, dn=0, dnx=0, dny=0,
        )
        cals = []
        for cam_spec in yaml_conf["cameras"]:
            cals.append(read_calibration(cam_spec["ori_file"], cam_spec.get("addpar_file")))

        point = np.array([0.0, 0.0, 0.0])
        right, left, down, up = searchquader(point, tpar, cpar, cals)

        for cam in range(num_cams):
            assert right[cam] > 0, f"cam{cam}: right={right[cam]} should be > 0"
            assert left[cam] > 0, f"cam{cam}: left={left[cam]} should be > 0"
            assert down[cam] > 0, f"cam{cam}: down={down[cam]} should be > 0"
            assert up[cam] > 0, f"cam{cam}: up={up[cam]} should be > 0"


# ═══════════════════════════════════════════════════════════════════════════
# 6. POINT_TO_PIXEL (full projection 3D → pixel) PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestPointToPixelParity:
    """Verify that point_to_pixel produces correct pixel coordinates
    by comparing with Cython image_coordinates + metric_to_pixel."""

    @skip_no_optv
    def test_point_to_pixel(self, yaml_conf):
        """3D point → pixel coord must match Cython pipeline."""
        from optv.calibration import Calibration as CyCalibration
        from optv.parameters import ControlParams as CyControlParams
        from optv.imgcoord import image_coordinates as cy_img_coord
        from optv.transforms import convert_arr_metric_to_pixel as cy_m2p

        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar
        from algorithms.track import point_to_pixel

        scene = yaml_conf["scene"]
        num_cams = len(yaml_conf["cameras"])
        cy_cpar = CyControlParams(num_cams, **scene)
        mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
        py_cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0], imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0], pix_y=scene["pixel_size"][1],
            mm=mm,
        )

        test_pts = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ])

        for cam_spec in yaml_conf["cameras"]:
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")

            cy_cal = CyCalibration()
            cy_cal.from_file(ori.encode(), addpar.encode() if addpar else None)
            cy_mm = cy_cpar.get_multimedia_params()
            py_cal = read_calibration(ori, addpar)

            # Cython: img_coord → metric_to_pixel
            cy_metric = cy_img_coord(test_pts, cy_cal, cy_mm)
            cy_pixel = cy_m2p(cy_metric, cy_cpar)

            for i, pt in enumerate(test_pts):
                py_pix = point_to_pixel(pt, py_cal, py_cpar)
                np.testing.assert_allclose(
                    py_pix, cy_pixel[i], atol=0.1,
                    err_msg=f"point_to_pixel mismatch: {ori}, point {pt}",
                )


# ═══════════════════════════════════════════════════════════════════════════
# 7. FULL TRACKER: trackcorr_c_loop OUTPUT PARITY
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def track_test_env(tmp_path):
    """Set up a temporary copy of track test data for both engines."""
    src = TRACK_DATA

    cy_dir = tmp_path / "cython"
    py_dir = tmp_path / "python"

    for dest in (cy_dir, py_dir):
        dest.mkdir()
        # Symlink immutable data
        (dest / "cal").symlink_to((src / "cal").resolve())
        (dest / "newpart").symlink_to((src / "newpart").resolve())

        # Copy res_orig → res
        res = dest / "res"
        shutil.copytree(src / "res_orig", res)

    return cy_dir, py_dir


class TestTrackcorrOutputParity:
    """Compare trackcorr_c_loop (2D-correlation tracker) output files."""

    @skip_no_optv
    def test_linkage_files_match(self, yaml_conf, track_test_env):
        """linkage.* files must be identical between engines."""
        cy_dir, py_dir = track_test_env
        seq = yaml_conf["sequence"]
        num_cams = len(yaml_conf["cameras"])

        # --- Cython Tracker ---
        from optv.tracker import Tracker as CyTracker
        from optv.calibration import Calibration as CyCalibration
        from optv.parameters import (
            ControlParams as CyControlParams,
            VolumeParams as CyVolumeParams,
            TrackingParams as CyTrackingParams,
            SequenceParams as CySequenceParams,
        )

        cy_cals = []
        cy_img_base = []
        for cix, cam_spec in enumerate(yaml_conf["cameras"]):
            cal = CyCalibration()
            cal.from_file(
                cam_spec["ori_file"].encode(),
                cam_spec.get("addpar_file", "").encode(),
            )
            cy_cals.append(cal)
            cy_img_base.append(seq["targets_template"].format(cam=cix + 1))

        cy_cpar = CyControlParams(num_cams, **yaml_conf["scene"])
        cy_vpar = CyVolumeParams(**yaml_conf["correspondences"])
        cy_tpar = CyTrackingParams(**yaml_conf["tracking"])
        cy_spar = CySequenceParams(
            image_base=cy_img_base,
            frame_range=(seq["first"], seq["last"]),
        )

        cy_naming = {
            "corres": str(cy_dir / "res" / "particles").encode(),
            "linkage": str(cy_dir / "res" / "linkage").encode(),
            "prio": str(cy_dir / "res" / "whatever").encode(),
        }

        cy_tracker = CyTracker(cy_cpar, cy_vpar, cy_tpar, cy_spar, cy_cals, cy_naming)
        cy_tracker.full_forward()

        # --- Python Tracker ---
        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar, SequencePar, VolumePar, TrackParTuple
        from algorithms.track import Tracker as PyTracker

        scene = yaml_conf["scene"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]
        vel = tracking["velocity_lims"]

        py_cals = [read_calibration(c["ori_file"], c.get("addpar_file"))
                   for c in yaml_conf["cameras"]]

        mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
        py_cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0], imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0], pix_y=scene["pixel_size"][1],
            mm=mm,
        )
        py_vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))],
            z_max_lay=[corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))],
            cn=corresp.get("pixels_tot", 0),
            cnx=corresp.get("pixels_x", 0),
            cny=corresp.get("pixels_y", 0),
            csumg=corresp.get("ref_gray", 0),
            eps0=corresp.get("epipolar_band", 0),
            corrmin=corresp.get("min_correlation", 0),
        )
        py_tpar = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking.get("add_particle", 0),
            dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
        )
        py_img_base = [seq["targets_template"].format(cam=i + 1) for i in range(num_cams)]
        py_spar = SequencePar(
            img_base_name=py_img_base,
            first=seq["first"], last=seq["last"],
        )
        py_naming = {
            "corres": str(py_dir / "res" / "particles"),
            "linkage": str(py_dir / "res" / "linkage"),
            "prio": str(py_dir / "res" / "whatever"),
        }

        py_tracker = PyTracker(py_cpar, py_vpar, py_tpar, py_spar, py_cals, py_naming)
        py_tracker.full_forward()

        # --- Compare linkage files ---
        for step in range(seq["first"], seq["last"]):
            cy_file = cy_dir / "res" / f"linkage.{step}"
            py_file = py_dir / "res" / f"linkage.{step}"

            assert cy_file.exists(), f"Cython missing linkage.{step}"
            assert py_file.exists(), f"Python missing linkage.{step}"

            cy_text = cy_file.read_text().strip()
            py_text = py_file.read_text().strip()

            assert cy_text == py_text, \
                f"linkage.{step} differs:\n  Cython: {cy_text!r}\n  Python: {py_text!r}"

    @skip_no_optv
    def test_particles_files_match(self, yaml_conf, track_test_env):
        """particles.* (rt_is) files must match between engines."""
        cy_dir, py_dir = track_test_env
        seq = yaml_conf["sequence"]

        # Reuse logic from test_linkage_files_match (run both trackers)
        # For simplicity, just run the Python tracker and compare to res_orig
        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar, SequencePar, VolumePar, TrackParTuple
        from algorithms.track import Tracker as PyTracker

        scene = yaml_conf["scene"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]
        vel = tracking["velocity_lims"]
        num_cams = len(yaml_conf["cameras"])

        py_cals = [read_calibration(c["ori_file"], c.get("addpar_file"))
                   for c in yaml_conf["cameras"]]

        mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
        py_cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0], imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0], pix_y=scene["pixel_size"][1],
            mm=mm,
        )
        py_vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))],
            z_max_lay=[corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))],
        )
        py_tpar = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking.get("add_particle", 0),
            dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
        )
        py_img_base = [seq["targets_template"].format(cam=i + 1) for i in range(num_cams)]
        py_spar = SequencePar(img_base_name=py_img_base, first=seq["first"], last=seq["last"])
        py_naming = {
            "corres": str(py_dir / "res" / "particles"),
            "linkage": str(py_dir / "res" / "linkage"),
            "prio": str(py_dir / "res" / "whatever"),
        }

        py_tracker = PyTracker(py_cpar, py_vpar, py_tpar, py_spar, py_cals, py_naming)
        py_tracker.full_forward()

        # Compare against res_orig reference
        for step in range(seq["first"], seq["last"] + 1):
            ref_file = TRACK_DATA / "res_orig" / f"particles.{step}"
            py_file = py_dir / "res" / f"particles.{step}"

            if not ref_file.exists():
                continue
            assert py_file.exists(), f"Python missing particles.{step}"

            ref_lines = ref_file.read_text().strip().splitlines()
            py_lines = py_file.read_text().strip().splitlines()

            ref_count = int(ref_lines[0])
            py_count = int(py_lines[0])
            if ref_count < 0:
                ref_count = 0
            if py_count < 0:
                py_count = 0

            assert ref_count == py_count, \
                f"particles.{step}: ref count={ref_count} py count={py_count}"

            for i in range(1, min(len(ref_lines), len(py_lines))):
                ref_vals = [float(x) for x in ref_lines[i].split()]
                py_vals = [float(x) for x in py_lines[i].split()]
                np.testing.assert_allclose(
                    py_vals, ref_vals, atol=1e-3,
                    err_msg=f"particles.{step} line {i}",
                )


# ═══════════════════════════════════════════════════════════════════════════
# 8. FULL TRACKER: track3d_loop OUTPUT PARITY
# ═══════════════════════════════════════════════════════════════════════════

class TestTrack3DOutputParity:
    """Compare track3d_loop (3D-direct tracker) output files."""

    @skip_no_optv
    def test_track3d_linkage_files(self, yaml_conf, track_test_env):
        """track3d linkage files must match between engines."""
        cy_dir, py_dir = track_test_env
        seq = yaml_conf["sequence"]
        num_cams = len(yaml_conf["cameras"])

        # --- Cython ---
        from optv.tracker import Tracker as CyTracker
        from optv.calibration import Calibration as CyCalibration
        from optv.parameters import (
            ControlParams, VolumeParams, TrackingParams, SequenceParams,
        )

        cy_cals = []
        cy_img_base = []
        for cix, cam_spec in enumerate(yaml_conf["cameras"]):
            cal = CyCalibration()
            cal.from_file(
                cam_spec["ori_file"].encode(),
                cam_spec.get("addpar_file", "").encode(),
            )
            cy_cals.append(cal)
            cy_img_base.append(seq["targets_template"].format(cam=cix + 1))

        cy_cpar = ControlParams(num_cams, **yaml_conf["scene"])
        cy_vpar = VolumeParams(**yaml_conf["correspondences"])
        cy_tpar = TrackingParams(**yaml_conf["tracking"])
        cy_spar = SequenceParams(
            image_base=cy_img_base,
            frame_range=(seq["first"], seq["last"]),
        )
        cy_naming = {
            "corres": str(cy_dir / "res" / "particles").encode(),
            "linkage": str(cy_dir / "res" / "linkage").encode(),
            "prio": str(cy_dir / "res" / "whatever").encode(),
        }
        cy_tracker = CyTracker(cy_cpar, cy_vpar, cy_tpar, cy_spar, cy_cals, cy_naming)
        cy_tracker.full_forward_3d()

        # --- Python ---
        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar, SequencePar, VolumePar, TrackParTuple
        from algorithms.track import Tracker as PyTracker

        scene = yaml_conf["scene"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]
        vel = tracking["velocity_lims"]

        py_cals = [read_calibration(c["ori_file"], c.get("addpar_file"))
                   for c in yaml_conf["cameras"]]
        mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
        py_cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0], imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0], pix_y=scene["pixel_size"][1],
            mm=mm,
        )
        py_vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))],
            z_max_lay=[corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))],
        )
        py_tpar = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking.get("add_particle", 0),
            dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
        )
        py_img_base = [seq["targets_template"].format(cam=i + 1) for i in range(num_cams)]
        py_spar = SequencePar(img_base_name=py_img_base, first=seq["first"], last=seq["last"])
        py_naming = {
            "corres": str(py_dir / "res" / "particles"),
            "linkage": str(py_dir / "res" / "linkage"),
            "prio": str(py_dir / "res" / "whatever"),
        }
        py_tracker = PyTracker(py_cpar, py_vpar, py_tpar, py_spar, py_cals, py_naming)
        py_tracker.full_forward_3d()

        # --- Compare ---
        for step in range(seq["first"], seq["last"]):
            cy_file = cy_dir / "res" / f"linkage.{step}"
            py_file = py_dir / "res" / f"linkage.{step}"

            assert cy_file.exists(), f"Cython missing linkage.{step}"
            assert py_file.exists(), f"Python missing linkage.{step}"

            cy_text = cy_file.read_text().strip()
            py_text = py_file.read_text().strip()

            assert cy_text == py_text, \
                f"track3d linkage.{step} differs:\n  Cython: {cy_text!r}\n  Python: {py_text!r}"

    @skip_no_optv
    def test_track3d_particles_vs_reference(self, yaml_conf, track_test_env):
        """track3d particles files must match res_orig reference."""
        _, py_dir = track_test_env
        seq = yaml_conf["sequence"]
        num_cams = len(yaml_conf["cameras"])

        from algorithms.calibration import read_calibration
        from algorithms.parameters import ControlPar, MultimediaPar, SequencePar, VolumePar, TrackParTuple
        from algorithms.track import Tracker as PyTracker

        scene = yaml_conf["scene"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]
        vel = tracking["velocity_lims"]

        py_cals = [read_calibration(c["ori_file"], c.get("addpar_file"))
                   for c in yaml_conf["cameras"]]
        mm = MultimediaPar(nlay=1, n1=1, n2=[1], d=[0], n3=1)
        py_cpar = ControlPar(
            num_cams=num_cams,
            imx=scene["image_size"][0], imy=scene["image_size"][1],
            pix_x=scene["pixel_size"][0], pix_y=scene["pixel_size"][1],
            mm=mm,
        )
        py_vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))],
            z_max_lay=[corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))],
        )
        py_tpar = TrackParTuple(
            dvxmin=vel[0][0], dvxmax=vel[0][1],
            dvymin=vel[1][0], dvymax=vel[1][1],
            dvzmin=vel[2][0], dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking.get("add_particle", 0),
            dsumg=0.0, dn=0.0, dnx=0.0, dny=0.0,
        )
        py_img_base = [seq["targets_template"].format(cam=i + 1) for i in range(num_cams)]
        py_spar = SequencePar(img_base_name=py_img_base, first=seq["first"], last=seq["last"])
        py_naming = {
            "corres": str(py_dir / "res" / "particles"),
            "linkage": str(py_dir / "res" / "linkage"),
            "prio": str(py_dir / "res" / "whatever"),
        }
        py_tracker = PyTracker(py_cpar, py_vpar, py_tpar, py_spar, py_cals, py_naming)
        py_tracker.full_forward_3d()

        for step in range(seq["first"], seq["last"] + 1):
            ref_file = TRACK_DATA / "res_orig" / f"particles.{step}"
            py_file = py_dir / "res" / f"particles.{step}"

            if not ref_file.exists():
                continue
            assert py_file.exists(), f"Python missing particles.{step}"

            ref_lines = ref_file.read_text().strip().splitlines()
            py_lines = py_file.read_text().strip().splitlines()

            ref_count = max(0, int(ref_lines[0]))
            py_count = max(0, int(py_lines[0]))

            assert ref_count == py_count, \
                f"track3d particles.{step}: ref={ref_count} py={py_count}"

            for i in range(1, min(len(ref_lines), len(py_lines))):
                ref_vals = [float(x) for x in ref_lines[i].split()]
                py_vals = [float(x) for x in py_lines[i].split()]
                np.testing.assert_allclose(
                    py_vals, ref_vals, atol=1e-3,
                    err_msg=f"track3d particles.{step} line {i}",
                )
