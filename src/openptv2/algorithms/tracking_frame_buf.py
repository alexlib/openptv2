"""Tracking frame buffer data structures and I/O.

Translation of lib/src/tracking_frame_buf.c and lib/include/tracking_frame_buf.h.
"""

from pathlib import Path

import cython
import numpy as np

from .constants import COORD_UNUSED, CORRES_NONE, NEXT_NONE, POSI, PREV_NONE, PT_UNUSED


class CallableInt(int):
    def __call__(self):
        return int(self)


class CallableFloat(float):
    def __call__(self):
        return float(self)


@cython.cclass
class Target:
    """Particle target in a camera frame."""

    c_pnr: cython.int = cython.declare(cython.int, visibility="public")
    c_x: cython.double = cython.declare(cython.double, visibility="public")
    c_y: cython.double = cython.declare(cython.double, visibility="public")
    n: cython.int = cython.declare(cython.int, visibility="public")
    nx: cython.int = cython.declare(cython.int, visibility="public")
    ny: cython.int = cython.declare(cython.int, visibility="public")
    sumg: cython.int = cython.declare(cython.int, visibility="public")
    c_tnr: cython.int = cython.declare(cython.int, visibility="public")

    def __init__(self, pnr=0, x=0.0, y=0.0, n=0, nx=0, ny=0, sumg=0, tnr=0):
        self.c_pnr = int(pnr)
        self.c_x = float(x)
        self.c_y = float(y)
        self.n = int(n)
        self.nx = int(nx)
        self.ny = int(ny)
        self.sumg = int(sumg)
        self.c_tnr = int(tnr)

    def __repr__(self):
        return (
            f"Target(pnr={self.c_pnr}, x={self.c_x}, y={self.c_y}, "
            f"n={self.n}, nx={self.nx}, ny={self.ny}, "
            f"sumg={self.sumg}, tnr={self.c_tnr})"
        )

    # --- Direct field access preferred in compiled code ---
    # Use c_x, c_y, c_tnr, c_pnr directly for C-speed access

    @property
    def pnr(self):
        return CallableInt(self.c_pnr)

    @pnr.setter
    def pnr(self, val):
        self.c_pnr = int(val)

    @property
    def x(self):
        return CallableFloat(self.c_x)

    @x.setter
    def x(self, val):
        self.c_x = float(val)

    @property
    def y(self):
        return CallableFloat(self.c_y)

    @y.setter
    def y(self, val):
        self.c_y = float(val)

    @property
    def tnr(self):
        return CallableInt(self.c_tnr)

    @tnr.setter
    def tnr(self, val):
        self.c_tnr = int(val)

    # --- Backward Compatibility OOP Methods ---
    def set_pnr(self, pnr: int) -> None:
        self.c_pnr = int(pnr)

    def pos(self) -> np.ndarray:
        return np.array([self.c_x, self.c_y], dtype=np.float64)

    def set_pos(self, pos) -> None:
        self.c_x = float(pos[0])
        self.c_y = float(pos[1])

    def set_tnr(self, tnr: int) -> None:
        self.c_tnr = int(tnr)

    def count_pixels(self) -> tuple[int, int, int]:
        return (self.n, self.nx, self.ny)

    def set_pixel_counts(self, n: int, nx: int, ny: int) -> None:
        self.n = int(n)
        self.nx = int(nx)
        self.ny = int(ny)

    def sum_grey_value(self) -> int:
        return self.sumg

    def set_sum_grey_value(self, sumg: int) -> None:
        self.sumg = int(sumg)


class TargetArray(list):
    """A list of Targets that behaves like the legacy TargetArray."""

    def __init__(self, size_or_list=0):
        if isinstance(size_or_list, int):
            super().__init__([Target(pnr=-1) for _ in range(size_or_list)])
        elif isinstance(size_or_list, list):
            super().__init__(size_or_list)
        else:
            raise TypeError(f"Expected int or list, got {type(size_or_list)}")

    def sort_y(self) -> None:
        self.sort(key=lambda t: t.y)

    def write(self, file_base: str, frame_num: int) -> bool:
        return write_targets(self, len(self), file_base, frame_num)

    @staticmethod
    def read_targets(base_name: str, frame_num: int, cpar=None) -> "TargetArray":
        targets = read_targets(base_name, frame_num)
        return TargetArray(targets)


@cython.ccall
def compare_targets(t1, t2):
    return (
        t1.pnr == t2.pnr
        and t1.x == t2.x
        and t1.y == t2.y
        and t1.n == t2.n
        and t1.nx == t2.nx
        and t1.ny == t2.ny
        and t1.sumg == t2.sumg
        and t1.tnr == t2.tnr
    )


def _resolve_file_base(file_base, frame_num):
    file_base_str = str(file_base)
    if "%d" in file_base_str:
        return (file_base_str % frame_num) + "_targets"
    if frame_num > 0:
        return f"{file_base_str}{frame_num:04d}_targets"
    return f"{file_base_str}_targets"


@cython.ccall
def read_targets(file_base, frame_num):
    fname = _resolve_file_base(file_base, frame_num)

    try:
        with open(fname, "r") as f:
            num_targets = int(f.readline().strip())
            targets = []
            for _ in range(num_targets):
                parts = f.readline().split()
                if len(parts) < 8:
                    return []
                targets.append(
                    Target(
                        pnr=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                        n=int(parts[3]),
                        nx=int(parts[4]),
                        ny=int(parts[5]),
                        sumg=int(parts[6]),
                        tnr=int(parts[7]),
                    )
                )
            return targets
    except FileNotFoundError:
        import re
        p = Path(fname)
        cam_match = re.search(r"cam(\d+)", p.name)
        cam_idx = int(cam_match.group(1)) - 1 if cam_match else 0
        zarr_candidates = [
            p.parent / "run.zarr",
            p.parent / "targets.zarr",
            p.parent.parent / "res" / "run.zarr",
        ]
        for zpath in zarr_candidates:
            if zpath.exists():
                from openptv2.storage import ZarrFrameStore
                try:
                    store = ZarrFrameStore(zpath, mode="r")
                    if store.has_targets(cam_idx, frame_num):
                        return list(store.read_targets(cam_idx, frame_num))
                except Exception:
                    pass
        return []


@cython.ccall
def write_targets(tbuf, num_targets, file_base, frame_num):
    fname = _resolve_file_base(file_base, frame_num)

    try:
        with open(fname, "w") as f:
            f.write(f"{num_targets}\n")
            for i in range(num_targets):
                t = tbuf[i]
                f.write(
                    f"{t.pnr:4d} {t.x:9.4f} {t.y:9.4f} "
                    f"{t.n:5d} {t.nx:5d} {t.ny:5d} {t.sumg:5d} {t.tnr:5d}\n"
                )
        return True
    except IOError:
        return False


@cython.cclass
class Corres:
    """Correspondence between cameras for a 3D particle."""

    nr: cython.int = cython.declare(cython.int, visibility="public")
    p: np.ndarray = cython.declare(object, visibility="public")

    def __init__(self, nr=0, p=None):
        self.nr = int(nr)
        if p is None:
            self.p = np.array([CORRES_NONE] * 4, dtype=np.int32)
        else:
            self.p = np.asarray(p, dtype=np.int32)


@cython.ccall
def compare_corres(c1, c2):
    return (
        c1.nr == c2.nr
        and c1.p[0] == c2.p[0]
        and c1.p[1] == c2.p[1]
        and c1.p[2] == c2.p[2]
        and c1.p[3] == c2.p[3]
    )


@cython.cclass
class Pathinfo:
    """Particle path information across frames."""

    x: np.ndarray = cython.declare(object, visibility="public")
    prev: cython.int = cython.declare(cython.int, visibility="public")
    next_idx: cython.int = cython.declare(cython.int, visibility="public")
    prio: cython.int = cython.declare(cython.int, visibility="public")
    finaldecis: cython.double = cython.declare(cython.double, visibility="public")
    inlist: cython.int = cython.declare(cython.int, visibility="public")
    # Backing storage for decis/linkdecis is lazily materialized (see
    # properties below): Frame() mass-preallocates max_targets Pathinfo
    # objects but only ~num_parts are ever touched per frame, so eagerly
    # building two POSI-length lists per object wastes the vast majority
    # of the allocations.
    _decis: list = cython.declare(object)
    _linkdecis: list = cython.declare(object)

    def __init__(
        self,
        x=None,
        prev=PREV_NONE,
        next_idx=NEXT_NONE,
        prio=4,
        finaldecis=1000000.0,
        inlist=0,
        decis=None,
        linkdecis=None,
    ):
        if x is None:
            self.x = np.zeros(3, dtype=np.float64)
        else:
            self.x = np.asarray(x, dtype=np.float64)
        self.prev = int(prev)
        self.next_idx = int(next_idx)
        self.prio = int(prio)
        self.finaldecis = float(finaldecis)
        self.inlist = int(inlist)
        self._decis = list(decis) if decis is not None else None
        self._linkdecis = list(linkdecis) if linkdecis is not None else None

    @property
    def decis(self):
        if self._decis is None:
            self._decis = [0.0] * POSI
        return self._decis

    @decis.setter
    def decis(self, value):
        self._decis = list(value)

    @property
    def linkdecis(self):
        if self._linkdecis is None:
            self._linkdecis = [PT_UNUSED] * POSI
        return self._linkdecis

    @linkdecis.setter
    def linkdecis(self, value):
        self._linkdecis = list(value)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def compare_path_info(p1, p2):
    if not (
        p1.prev == p2.prev
        and p1.next_idx == p2.next_idx
        and p1.prio == p2.prio
        and p1.finaldecis == p2.finaldecis
        and p1.inlist == p2.inlist
        and np.allclose(p1.x, p2.x)
    ):
        return False
    for i in range(POSI):
        if p1.decis[i] != p2.decis[i]:
            return False
        if p1.linkdecis[i] != p2.linkdecis[i]:
            return False
    return True


@cython.ccall
def register_link_candidate(path, fitness, cand):
    path.decis[path.inlist] = fitness
    path.linkdecis[path.inlist] = cand
    path.inlist += 1


@cython.ccall
def reset_links(path):
    path.prev = PREV_NONE
    path.next_idx = NEXT_NONE
    path.prio = 2  # PRIO_DEFAULT


@cython.ccall
def read_path_frame(corres_file_base, linkage_file_base, prio_file_base, frame_num):
    fname = f"{corres_file_base}.{frame_num}"
    try:
        corres_file = open(fname, "r")
    except FileNotFoundError:
        return [], []

    corres_file.readline()  # number of points (read but use EOF)

    linkage_file = None
    if linkage_file_base:
        lfname = f"{linkage_file_base}.{frame_num}"
        try:
            linkage_file = open(lfname, "r")
            linkage_file.readline()  # skip header
        except FileNotFoundError:
            linkage_file = None

    prio_file = None
    if prio_file_base:
        pfname = f"{prio_file_base}.{frame_num}"
        try:
            prio_file = open(pfname, "r")
            prio_file.readline()  # skip header
        except FileNotFoundError:
            prio_file = None

    cor_buf = []
    path_buf = []
    targets = 0

    for line in corres_file:
        line = line.strip()
        if not line:
            continue

        # Read linkage data if available
        if linkage_file is not None:
            link_line = linkage_file.readline().strip()
            if link_line:
                link_parts = link_line.split()
                prev_val = int(link_parts[0])
                next_val = int(link_parts[1])
            else:
                prev_val = PREV_NONE
                next_val = NEXT_NONE
        else:
            prev_val = PREV_NONE
            next_val = NEXT_NONE

        # Read prio data if available
        if prio_file is not None:
            prio_line = prio_file.readline().strip()
            if prio_line:
                prio_parts = prio_line.split()
                prio_val = int(prio_parts[5])
            else:
                prio_val = 4
        else:
            prio_val = 4

        # Parse corres line: nr x y z p[0] p[1] p[2] p[3]
        parts = line.split()
        if len(parts) < 8:
            break

        path = Pathinfo(
            x=np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
            prev=prev_val,
            next_idx=next_val,
            prio=prio_val,
            finaldecis=1000000.0,
            inlist=0,
        )

        targets += 1
        cor = Corres(
            nr=targets,
            p=np.array(
                [int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])],
                dtype=np.int32,
            ),
        )

        cor_buf.append(cor)
        path_buf.append(path)

    corres_file.close()
    if linkage_file is not None:
        linkage_file.close()
    if prio_file is not None:
        prio_file.close()

    return cor_buf, path_buf


@cython.ccall
def write_path_frame(
    cor_buf,
    path_buf,
    num_parts,
    corres_file_base,
    linkage_file_base,
    prio_file_base,
    frame_num,
):
    corres_fname = f"{corres_file_base}.{frame_num}"
    linkage_fname = f"{linkage_file_base}.{frame_num}" if linkage_file_base else None

    try:
        corres_file = open(corres_fname, "w")
    except IOError:
        return False

    linkage_file = None
    if linkage_fname:
        try:
            linkage_file = open(linkage_fname, "w")
        except IOError:
            corres_file.close()
            return False

    prio_file = None
    if prio_file_base:
        prio_fname = f"{prio_file_base}.{frame_num}"
        try:
            prio_file = open(prio_fname, "w")
        except IOError:
            corres_file.close()
            if linkage_file:
                linkage_file.close()
            return False

    corres_file.write(f"{num_parts}\n")
    if linkage_file:
        linkage_file.write(f"{num_parts}\n")
    if prio_file:
        prio_file.write(f"{num_parts}\n")

    for pix in range(num_parts):
        p = path_buf[pix]

        # Handle cor_buf: can be list of Corres objects, or (nr_array, p_array) tuple
        if (
            isinstance(cor_buf, (list, tuple))
            and len(cor_buf) == 2
            and isinstance(cor_buf[0], np.ndarray)
        ):
            int(cor_buf[0][pix])
            c_p = cor_buf[1][pix]
        elif isinstance(cor_buf, list) and isinstance(cor_buf[0], Corres):
            c_p = cor_buf[pix].p
        else:
            c_p = (
                cor_buf[pix].p
                if hasattr(cor_buf[pix], "p")
                else np.zeros(4, dtype=np.int32)
            )

        if linkage_file:
            linkage_file.write(
                f"{p.prev:4d} {p.next_idx:4d} {p.x[0]:10.3f} {p.x[1]:10.3f} {p.x[2]:10.3f}\n"
            )

        corres_file.write(
            f"{pix + 1:4d} {p.x[0]:9.3f} {p.x[1]:9.3f} {p.x[2]:9.3f} "
            f"{c_p[0]:4d} {c_p[1]:4d} {c_p[2]:4d} {c_p[3]:4d}\n"
        )

        if prio_file:
            prio_file.write(
                f"{p.prev:4d} {p.next_idx:4d} {p.x[0]:10.3f} {p.x[1]:10.3f} {p.x[2]:10.3f} {p.prio:d}\n"
            )

    corres_file.close()
    if linkage_file:
        linkage_file.close()
    if prio_file:
        prio_file.close()

    return True


@cython.cclass
class Frame:
    num_cams: cython.int = cython.declare(cython.int, visibility="public")
    _num_cams: cython.int = cython.declare(cython.int, visibility="public")
    max_targets: cython.int = cython.declare(cython.int, visibility="public")
    targets: object = cython.declare(object, visibility="public")
    correspond: object = cython.declare(object, visibility="public")
    path_info: object = cython.declare(object, visibility="public")
    num_targets: object = cython.declare(object, visibility="public")
    num_parts: cython.int = cython.declare(cython.int, visibility="public")
    targ_x: object = cython.declare(object, visibility="public")
    targ_y: object = cython.declare(object, visibility="public")
    targ_tnr: object = cython.declare(object, visibility="public")
    path_x: object = cython.declare(object, visibility="public")
    path_prev: object = cython.declare(object, visibility="public")
    path_next: object = cython.declare(object, visibility="public")
    path_prio: object = cython.declare(object, visibility="public")
    path_inlist: object = cython.declare(object, visibility="public")
    path_finaldecis: object = cython.declare(object, visibility="public")
    path_decis: object = cython.declare(object, visibility="public")
    path_linkdecis: object = cython.declare(object, visibility="public")
    corres_nr: object = cython.declare(object, visibility="public")
    corres_p: object = cython.declare(object, visibility="public")

    def __init__(self, num_cams=4, max_targets=1000, **kwargs):
        self.num_cams = num_cams
        self._num_cams = num_cams
        self.max_targets = max_targets
        self.targets = [[Target() for _ in range(max_targets)] for _ in range(num_cams)]
        self.correspond = [Corres() for _ in range(max_targets)]
        self.path_info = [Pathinfo() for _ in range(max_targets)]
        self.num_targets = [0] * num_cams
        self.num_parts = 0

        # SoA for targets — native 2D arrays (num_cams, max_targets)
        self.targ_x = np.full((num_cams, max_targets), COORD_UNUSED, dtype=np.float64)
        self.targ_y = np.full((num_cams, max_targets), COORD_UNUSED, dtype=np.float64)
        self.targ_tnr = np.full((num_cams, max_targets), PT_UNUSED, dtype=np.int32)

        # SoA for Pathinfo
        self.path_x = np.zeros((max_targets, 3), dtype=np.float64)
        self.path_prev = np.full(max_targets, PREV_NONE, dtype=np.int32)
        self.path_next = np.full(max_targets, NEXT_NONE, dtype=np.int32)
        self.path_prio = np.full(max_targets, 4, dtype=np.int32)
        self.path_inlist = np.zeros(max_targets, dtype=np.int32)
        self.path_finaldecis = np.full(max_targets, 1000000.0, dtype=np.float64)
        self.path_decis = np.zeros((max_targets, POSI), dtype=np.float64)
        self.path_linkdecis = np.full((max_targets, POSI), PT_UNUSED, dtype=np.int32)

        # SoA for Corres
        self.corres_nr = np.zeros(max_targets, dtype=np.int32)
        self.corres_p = np.full((max_targets, 4), CORRES_NONE, dtype=np.int32)

        # Legacy convenience: read if file info is provided in kwargs
        if "frame_num" in kwargs and "target_file_base" in kwargs:
            self.read(
                kwargs.get("corres_file_base"),
                kwargs.get("linkage_file_base"),
                kwargs.get("prio_file_base"),
                kwargs.get("target_file_base"),
                kwargs["frame_num"],
            )

    def positions(self) -> np.ndarray:
        """Get 3D positions as ndarray[n, 3]."""
        num_parts = self.num_parts
        positions = np.zeros((num_parts, 3), dtype=np.float64)
        for i in range(num_parts):
            positions[i] = self.path_info[i].x
        return positions

    def target_positions_for_camera(self, cam: int) -> np.ndarray:
        """Get 2D target positions for specific camera as ndarray[n, 2]."""
        num_targs = self.num_targets[cam]
        positions = np.zeros((num_targs, 2), dtype=np.float64)
        for i in range(num_targs):
            positions[i, 0] = self.targets[cam][i].x
            positions[i, 1] = self.targets[cam][i].y
        return positions

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _sync_path_to_soa(self):
        """Copy AoS path_info/correspond into SoA arrays (memoryview-optimized)."""
        i: cython.int
        k: cython.int
        num_parts: cython.int = self.num_parts
        p: Pathinfo
        c: Corres

        # Local typed memoryview references — avoids repeated self.path_* lookups
        path_x: cython.double[:, ::1] = cython.declare(
            cython.double[:, ::1], self.path_x
        )
        path_prev: cython.int[::1] = cython.declare(cython.int[::1], self.path_prev)
        path_next: cython.int[::1] = cython.declare(cython.int[::1], self.path_next)
        path_prio: cython.int[::1] = cython.declare(cython.int[::1], self.path_prio)
        path_inlist: cython.int[::1] = cython.declare(cython.int[::1], self.path_inlist)
        path_finaldecis: cython.double[::1] = cython.declare(
            cython.double[::1], self.path_finaldecis
        )
        path_decis: cython.double[:, ::1] = cython.declare(
            cython.double[:, ::1], self.path_decis
        )
        path_linkdecis: cython.int[:, ::1] = cython.declare(
            cython.int[:, ::1], self.path_linkdecis
        )
        corres_nr: cython.int[::1] = cython.declare(cython.int[::1], self.corres_nr)
        corres_p: cython.int[:, ::1] = cython.declare(cython.int[:, ::1], self.corres_p)

        for i in range(num_parts):
            p = self.path_info[i]
            path_x[i, 0] = p.x[0]
            path_x[i, 1] = p.x[1]
            path_x[i, 2] = p.x[2]
            path_prev[i] = p.prev
            path_next[i] = p.next_idx
            path_prio[i] = p.prio
            path_inlist[i] = p.inlist
            path_finaldecis[i] = p.finaldecis
            for k in range(POSI):
                path_decis[i, k] = p.decis[k]
                path_linkdecis[i, k] = p.linkdecis[k]

            c = self.correspond[i]
            corres_nr[i] = c.nr
            corres_p[i, 0] = c.p[0]
            corres_p[i, 1] = c.p[1]
            corres_p[i, 2] = c.p[2]
            corres_p[i, 3] = c.p[3]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _sync_soa_to_path(self):
        """Copy SoA arrays back into AoS path_info/correspond (memoryview-optimized)."""
        i: cython.int
        k: cython.int
        num_parts: cython.int = self.num_parts
        p: Pathinfo
        c: Corres

        # Local typed memoryview references — avoids repeated self.path_* lookups
        path_x: cython.double[:, ::1] = cython.declare(
            cython.double[:, ::1], self.path_x
        )
        path_prev: cython.int[::1] = cython.declare(cython.int[::1], self.path_prev)
        path_next: cython.int[::1] = cython.declare(cython.int[::1], self.path_next)
        path_prio: cython.int[::1] = cython.declare(cython.int[::1], self.path_prio)
        path_inlist: cython.int[::1] = cython.declare(cython.int[::1], self.path_inlist)
        path_finaldecis: cython.double[::1] = cython.declare(
            cython.double[::1], self.path_finaldecis
        )
        path_decis: cython.double[:, ::1] = cython.declare(
            cython.double[:, ::1], self.path_decis
        )
        path_linkdecis: cython.int[:, ::1] = cython.declare(
            cython.int[:, ::1], self.path_linkdecis
        )
        corres_nr: cython.int[::1] = cython.declare(cython.int[::1], self.corres_nr)
        corres_p: cython.int[:, ::1] = cython.declare(cython.int[:, ::1], self.corres_p)

        for i in range(num_parts):
            p = self.path_info[i]
            p.x[0] = path_x[i, 0]
            p.x[1] = path_x[i, 1]
            p.x[2] = path_x[i, 2]
            p.prev = int(path_prev[i])
            p.next_idx = int(path_next[i])
            p.prio = int(path_prio[i])
            p.inlist = int(path_inlist[i])
            p.finaldecis = float(path_finaldecis[i])
            for k in range(POSI):
                p.decis[k] = float(path_decis[i, k])
                p.linkdecis[k] = int(path_linkdecis[i, k])

            c = self.correspond[i]
            c.nr = int(corres_nr[i])
            c.p[0] = int(corres_p[i, 0])
            c.p[1] = int(corres_p[i, 1])
            c.p[2] = int(corres_p[i, 2])
            c.p[3] = int(corres_p[i, 3])

    def read(self, corres_file_base, linkage_file_base, *args, **kwargs):
        prio_file_base = None
        target_file_base = None
        frame_num = None

        if "prio_file_base" in kwargs:
            prio_file_base = kwargs["prio_file_base"]
        if "target_file_base" in kwargs:
            target_file_base = kwargs["target_file_base"]
        if "frame_num" in kwargs:
            frame_num = kwargs["frame_num"]

        remaining_args = list(args)

        is_legacy = False
        if len(remaining_args) >= 2:
            if isinstance(remaining_args[1], int) or isinstance(
                remaining_args[0], list
            ):
                is_legacy = True

        if is_legacy:
            if target_file_base is None and len(remaining_args) > 0:
                target_file_base = remaining_args.pop(0)
            if frame_num is None and len(remaining_args) > 0:
                frame_num = remaining_args.pop(0)
            if prio_file_base is None and len(remaining_args) > 0:
                prio_file_base = remaining_args.pop(0)
        else:
            if prio_file_base is None and len(remaining_args) > 0:
                prio_file_base = remaining_args.pop(0)
            if target_file_base is None and len(remaining_args) > 0:
                target_file_base = remaining_args.pop(0)
            if frame_num is None and len(remaining_args) > 0:
                frame_num = remaining_args.pop(0)

        # Execute read using resolved parameters
        fname = f"{corres_file_base}.{frame_num}"
        if not Path(fname).exists():
            return False

        cor_list, path_list = read_path_frame(
            corres_file_base,
            linkage_file_base if linkage_file_base else "",
            prio_file_base if prio_file_base else "",
            frame_num,
        )

        self.num_parts = len(cor_list)
        for i in range(self.num_parts):
            self.correspond[i] = cor_list[i]
            self.path_info[i] = path_list[i]

        self._sync_path_to_soa()

        for cam in range(self.num_cams):
            targets = read_targets(
                target_file_base[cam]
                if isinstance(target_file_base, list)
                else target_file_base,
                frame_num,
            )
            self.num_targets[cam] = len(targets)
            tx = self.targ_x[cam]
            ty = self.targ_y[cam]
            ttnr = self.targ_tnr[cam]
            for j, t in enumerate(targets):
                self.targets[cam][j] = t
                tx[j] = t.x
                ty[j] = t.y
                ttnr[j] = t.tnr

        return True

    def write(
        self,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        target_file_base,
        frame_num,
    ):
        ok = write_path_frame(
            self.correspond,
            self.path_info,
            self.num_parts,
            corres_file_base,
            linkage_file_base,
            prio_file_base if prio_file_base else None,
            frame_num,
        )
        if not ok:
            return False

        for cam in range(self.num_cams):
            if self.num_targets[cam] > 0:
                ok = write_targets(
                    self.targets[cam],
                    self.num_targets[cam],
                    target_file_base[cam],
                    frame_num,
                )
                if not ok:
                    return False

        return True


@cython.cclass
class FrameBuf:
    buf_len: cython.int = cython.declare(cython.int, visibility="public")
    num_cams: cython.int = cython.declare(cython.int, visibility="public")
    _frames: object = cython.declare(object, visibility="public")
    buf: object = cython.declare(object, visibility="public")
    corres_file_base: object = cython.declare(object, visibility="public")
    linkage_file_base: object = cython.declare(object, visibility="public")
    prio_file_base: object = cython.declare(object, visibility="public")
    target_file_base: object = cython.declare(object, visibility="public")

    def __init__(
        self,
        buf_len,
        num_cams,
        max_targets,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        target_file_base,
    ):
        self.buf_len = buf_len
        self.num_cams = num_cams
        self._frames = [Frame(num_cams, max_targets) for _ in range(buf_len)]
        self.buf = list(self._frames)

        self.corres_file_base = corres_file_base
        self.linkage_file_base = linkage_file_base
        self.prio_file_base = prio_file_base
        self.target_file_base = target_file_base

    @property
    def _buf_start(self):
        return 0

    def fb_next(self):
        self.buf.append(self.buf.pop(0))

    def fb_prev(self):
        self.buf.insert(0, self.buf.pop())

    def read_frame_at_end(self, frame_num, read_links=True):
        frame = self.buf[self.buf_len - 1]
        linkage = self.linkage_file_base if read_links else ""
        prio = self.prio_file_base if read_links else ""
        return frame.read(
            self.corres_file_base, linkage, prio, self.target_file_base, frame_num
        )

    def write_frame_from_start(self, frame_num):
        frame = self.buf[0]
        return frame.write(
            self.corres_file_base,
            self.linkage_file_base,
            self.prio_file_base,
            self.target_file_base,
            frame_num,
        )


Corres_dtype = np.dtype([("nr", np.int32), ("p", np.int32, (4,))])


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
