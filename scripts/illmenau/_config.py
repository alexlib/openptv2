"""Shared configuration for the Illmenau calibration drivers.

Everything that used to be duplicated (and hard-coded to cameras 1-4) across the
scripts in this folder lives here, so a second camera group is a matter of
environment rather than editing eight files:

    ILLMENAU_RAW    dataset root holding Kalibrierung_<cam>/
    ILLMENAU_DIR    openptv2 working folder (cal/, parameters_Run1.yaml, ...)
    ILLMENAU_CAMS   comma-separated PHYSICAL camera numbers, e.g. "5,6,7,8"
    ILLMENAU_NPZ    name of the cached detections inside cal/

Camera numbers are kept physical (cam5..cam8 really are named cam5..cam8 on
disk) so nothing downstream has to remember an offset, while the arrays stay
indexed 0..n-1 within the group.  `cam_number(ci)` is the only bridge between
the two, and every script goes through it.

The two groups are SEPARATE WORLDS on purpose.  Cameras 1-4 see the front face
of the plate and 5-8 the back face, whose dot pattern is a different pattern
with its own ids -- there is no dot-to-dot correspondence to exploit.  Each
group is therefore calibrated standalone, anchored to its own reference frame's
coded L-corner dot.  What ties them together later is the plate itself: the two
dot planes are parallel and 6 mm apart, so the per-frame plate poses already
solved for cams 1-4 describe the same physical positions.  See
`plate.yaml:relation_to_front_rig` in the 5-8 working folder.
"""
from __future__ import annotations

import os
from pathlib import Path

ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))
RAW = Path(ILLMENAU_RAW)
DIR = Path(ILLMENAU_DIR)
CAL = DIR / "cal"

CAMS = [int(c) for c in os.environ.get("ILLMENAU_CAMS", "1,2,3,4").split(",") if c.strip()]
NCAM = len(CAMS)
NPZ = os.environ.get("ILLMENAU_NPZ", "labelled_all_frames.npz")

# Plate geometry.  nx/ny/pitch describe the printed lattice; the datum is the
# grid index of the coded L corner, which is what pins the world to a physical
# dot (see plate_labeler.label_coded_6x7).  Read from plate.yaml when present so
# the back-face group can carry its own datum without editing code.
PITCH, NX, NY = 120.0, 6, 7
DATUM_IX, DATUM_IY = 2, 3
REF = "00000000"
IMX, IMY, PIX = 2560, 2048, 0.005

_pl = DIR / "plate.yaml"
if _pl.exists():
    import yaml
    _p = (yaml.safe_load(_pl.read_text(encoding="utf-8")) or {}).get("plate", {})
    PITCH = float(_p.get("pitch_x", PITCH))
    NX, NY = int(_p.get("nx", NX)), int(_p.get("ny", NY))
    REF = str(_p.get("origin_frame", REF))
    _d = _p.get("datum") or {}
    if _d.get("ix") is not None:
        DATUM_IX, DATUM_IY = int(_d["ix"]), int(_d["iy"])


def cam_number(ci: int) -> int:
    """Physical camera number for group index `ci`."""
    return CAMS[ci]


def cam_ori(ci: int) -> tuple[str, str]:
    """(.ori, .addpar) paths for group index `ci`."""
    n = cam_number(ci)
    return str(CAL / f"cam{n}.tif.ori"), str(CAL / f"cam{n}.tif.addpar")


def image_dir(ci: int) -> Path:
    return RAW / f"Kalibrierung_{cam_number(ci)}"


def control_par():
    from openptv2.algorithms.parameters import ControlPar, MmNp
    return ControlPar(num_cams=NCAM, imx=IMX, imy=IMY, pix_x=PIX, pix_y=PIX,
                      mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0,
                      tiff_flag=1, hp_flag=1, allCam_flag=0,
                      img_base_name=[""] * NCAM, cal_img_base_name=[""] * NCAM)


def load_calibrations():
    from openptv2.algorithms.calibration import Calibration
    cals = []
    for ci in range(NCAM):
        c = Calibration()
        c.from_file(*cam_ori(ci))
        cals.append(c)
    return cals


def load_views(npz: str | None = None) -> dict:
    """{(group_index, frame): (ids, pixels)} from the cached detections."""
    import numpy as np
    d = np.load(CAL / (npz or NPZ))
    views = {}
    for k in d.files:
        if k.endswith("_ids"):
            c, fr, _ = k.split("_")
            views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])
    return views


def obj_of(ids):
    """Plate coordinates of point ids, datum dot at the origin, plate in z=0."""
    import numpy as np
    ids = np.asarray(ids)
    ix, iy = (ids - 1) % NX, (ids - 1) // NX
    return np.stack([(ix - DATUM_IX) * PITCH, (iy - DATUM_IY) * PITCH,
                     np.zeros(len(ix))], 1).astype(float)


def banner() -> str:
    return (f"cams {CAMS}  dir {DIR}  npz {NPZ}\n"
            f"plate {NX}x{NY} pitch {PITCH} mm, datum grid ({DATUM_IX},{DATUM_IY}), "
            f"reference frame {REF}")
