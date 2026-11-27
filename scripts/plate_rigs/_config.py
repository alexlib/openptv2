"""Shared configuration for the plate-rig calibration drivers.

Everything that used to be duplicated (and hard-coded to cameras 1-4) across the
scripts in this folder lives here, so a second camera group is a matter of
environment rather than editing eight files:

    PLATE_RIG_RAW    dataset root holding plate_cam<cam>/
    PLATE_RIG_DIR    openptv2 working folder (cal/, parameters_Run1.yaml, ...)
    PLATE_RIG_CAMS   comma-separated PHYSICAL camera numbers, e.g. "5,6,7,8"
    PLATE_RIG_NPZ    name of the cached detections inside cal/

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

from openptv2.plate_multiplane import PlateGrid, load_plate_views

PLATE_RIG_RAW = os.environ.get("PLATE_RIG_RAW", ".")
PLATE_RIG_DIR = os.environ.get(
    "PLATE_RIG_DIR", os.path.join(PLATE_RIG_RAW, "openptv_rig_a_4cam")
)
RAW = Path(PLATE_RIG_RAW)
DIR = Path(PLATE_RIG_DIR)
CAL = DIR / "cal"

CAMS = [
    int(c) for c in os.environ.get("PLATE_RIG_CAMS", "1,2,3,4").split(",") if c.strip()
]
NCAM = len(CAMS)
NPZ = os.environ.get("PLATE_RIG_NPZ", "labelled_all_frames.npz")

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

GRID = PlateGrid(
    nx=NX, ny=NY, pitch_x=PITCH, pitch_y=PITCH, datum_ix=DATUM_IX, datum_iy=DATUM_IY
)


def _check_group_matches_folder() -> None:
    """Refuse to mix a camera group with another group's working folder.

    Each group is a separate world with its own calibration, and the folders are
    the only thing keeping them apart.  PLATE_RIG_DIR defaults to the cameras 1-4
    folder, so forgetting to set it while working on 5-8 -- or running a
    write-script "just to check" -- silently overwrites a good calibration with
    one fitted from the wrong cameras.  That happened once here: a regression
    check against cams 1-4 clobbered their bundled .ori.

    parameters_Run1.yaml names its cameras (cal/cam5.tif.ori ...), so the folder
    states which group it belongs to.  Disagreement is a mistake, not a use case.
    """
    par = DIR / "parameters_Run1.yaml"
    if not par.exists():
        return
    import re

    names = set(re.findall(r"cam(\d+)\.tif\.ori", par.read_text(encoding="utf-8")))
    if not names:
        return
    want = {str(c) for c in CAMS}
    if names != want:
        raise SystemExit(
            f"camera group / working folder mismatch.\n"
            f"  PLATE_RIG_CAMS = {sorted(int(c) for c in want)}\n"
            f"  {par} describes cameras {sorted(int(n) for n in names)}\n"
            f"These are separate calibrations of separate worlds; writing one into "
            f"the other's folder destroys it.\nSet PLATE_RIG_DIR and PLATE_RIG_CAMS "
            f"together, or fix parameters_Run1.yaml if the folder is genuinely new."
        )


_check_group_matches_folder()


def cam_number(ci: int) -> int:
    """Physical camera number for group index `ci`."""
    return CAMS[ci]


def cam_ori(ci: int) -> tuple[str, str]:
    """(.ori, .addpar) paths for group index `ci`."""
    n = cam_number(ci)
    return str(CAL / f"cam{n}.tif.ori"), str(CAL / f"cam{n}.tif.addpar")


def image_dir(ci: int) -> Path:
    return RAW / f"plate_cam{cam_number(ci)}"


def control_par():
    from openptv2.algorithms.parameters import ControlPar, MmNp

    return ControlPar(
        num_cams=NCAM,
        imx=IMX,
        imy=IMY,
        pix_x=PIX,
        pix_y=PIX,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""] * NCAM,
        cal_img_base_name=[""] * NCAM,
    )


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
    return load_plate_views(CAL / (npz or NPZ))


def obj_of(ids):
    """Plate coordinates of point ids, datum dot at the origin, plate in z=0."""
    return GRID.object_points(ids)


def banner() -> str:
    return (
        f"cams {CAMS}  dir {DIR}  npz {NPZ}\n"
        f"plate {NX}x{NY} pitch {PITCH} mm, datum grid ({DATUM_IX},{DATUM_IY}), "
        f"reference frame {REF}"
    )
