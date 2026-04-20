"""Tracking frame buffer data structures and I/O.

Translation of lib/src/tracking_frame_buf.c and lib/include/tracking_frame_buf.h.
"""

import numpy as np
from pathlib import Path

from .constants import POSI, PT_UNUSED, CORRES_NONE, PREV_NONE, NEXT_NONE


class Target:
    __slots__ = ('pnr', 'x', 'y', 'n', 'nx', 'ny', 'sumg', 'tnr')

    def __init__(self, pnr=0, x=0.0, y=0.0, n=0, nx=0, ny=0, sumg=0, tnr=0):
        self.pnr = pnr
        self.x = x
        self.y = y
        self.n = n
        self.nx = nx
        self.ny = ny
        self.sumg = sumg
        self.tnr = tnr


def compare_targets(t1, t2):
    return (t1.pnr == t2.pnr and t1.x == t2.x and t1.y == t2.y and
            t1.n == t2.n and t1.nx == t2.nx and t1.ny == t2.ny and
            t1.sumg == t2.sumg and t1.tnr == t2.tnr)


def _resolve_file_base(file_base, frame_num):
    if '%d' in file_base:
        return (file_base % frame_num) + "_targets"
    if frame_num > 0:
        return f"{file_base}{frame_num:04d}_targets"
    return f"{file_base}_targets"


def read_targets(file_base, frame_num):
    fname = _resolve_file_base(file_base, frame_num)

    try:
        with open(fname, 'r') as f:
            num_targets = int(f.readline().strip())
            targets = []
            for _ in range(num_targets):
                parts = f.readline().split()
                if len(parts) < 8:
                    return []
                targets.append(Target(
                    pnr=int(parts[0]),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    n=int(parts[3]),
                    nx=int(parts[4]),
                    ny=int(parts[5]),
                    sumg=int(parts[6]),
                    tnr=int(parts[7]),
                ))
            return targets
    except FileNotFoundError:
        return []


def write_targets(tbuf, num_targets, file_base, frame_num):
    fname = _resolve_file_base(file_base, frame_num)

    try:
        with open(fname, 'w') as f:
            f.write(f"{num_targets}\n")
            for i in range(num_targets):
                t = tbuf[i]
                f.write(f"{t.pnr:4d} {t.x:9.4f} {t.y:9.4f} "
                        f"{t.n:5d} {t.nx:5d} {t.ny:5d} {t.sumg:5d} {t.tnr:5d}\n")
        return True
    except IOError:
        return False


class Corres:
    __slots__ = ('nr', 'p')

    def __init__(self, nr=0, p=None):
        self.nr = nr
        self.p = np.array([CORRES_NONE] * 4, dtype=np.int32) if p is None else np.asarray(p, dtype=np.int32)


def compare_corres(c1, c2):
    return (c1.nr == c2.nr and
            c1.p[0] == c2.p[0] and c1.p[1] == c2.p[1] and
            c1.p[2] == c2.p[2] and c1.p[3] == c2.p[3])


class Pathinfo:
    __slots__ = ('x', 'prev', 'next', 'prio', 'decis', 'finaldecis',
                 'linkdecis', 'inlist')

    def __init__(self, x=None, prev=PREV_NONE, next=NEXT_NONE,
                 prio=4, finaldecis=1000000.0, inlist=0,
                 decis=None, linkdecis=None):
        self.x = np.zeros(3, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
        self.prev = prev
        self.next = next
        self.prio = prio
        self.finaldecis = finaldecis
        self.inlist = inlist
        self.decis = [0.0] * POSI if decis is None else list(decis)
        self.linkdecis = [PT_UNUSED] * POSI if linkdecis is None else list(linkdecis)


def compare_path_info(p1, p2):
    if not (p1.prev == p2.prev and p1.next == p2.next and
            p1.prio == p2.prio and p1.finaldecis == p2.finaldecis and
            p1.inlist == p2.inlist and np.allclose(p1.x, p2.x)):
        return False
    for i in range(POSI):
        if p1.decis[i] != p2.decis[i]:
            return False
        if p1.linkdecis[i] != p2.linkdecis[i]:
            return False
    return True


def register_link_candidate(path, fitness, cand):
    path.decis[path.inlist] = fitness
    path.linkdecis[path.inlist] = cand
    path.inlist += 1


def reset_links(path):
    path.prev = PREV_NONE
    path.next = NEXT_NONE
    path.prio = 2  # PRIO_DEFAULT


def read_path_frame(corres_file_base, linkage_file_base, prio_file_base,
                    frame_num):
    fname = f"{corres_file_base}.{frame_num}"
    try:
        corres_file = open(fname, 'r')
    except FileNotFoundError:
        return [], []

    header = corres_file.readline()  # number of points (read but use EOF)

    linkage_file = None
    if linkage_file_base:
        lfname = f"{linkage_file_base}.{frame_num}"
        try:
            linkage_file = open(lfname, 'r')
            linkage_file.readline()  # skip header
        except FileNotFoundError:
            linkage_file = None

    prio_file = None
    if prio_file_base:
        pfname = f"{prio_file_base}.{frame_num}"
        try:
            prio_file = open(pfname, 'r')
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
            next=next_val,
            prio=prio_val,
            finaldecis=1000000.0,
            inlist=0,
        )

        targets += 1
        cor = Corres(
            nr=targets,
            p=np.array([int(parts[4]), int(parts[5]),
                         int(parts[6]), int(parts[7])], dtype=np.int32)
        )

        cor_buf.append(cor)
        path_buf.append(path)

    corres_file.close()
    if linkage_file is not None:
        linkage_file.close()
    if prio_file is not None:
        prio_file.close()

    return cor_buf, path_buf


def write_path_frame(cor_buf, path_buf, num_parts, corres_file_base,
                     linkage_file_base, prio_file_base, frame_num):
    corres_fname = f"{corres_file_base}.{frame_num}"
    linkage_fname = f"{linkage_file_base}.{frame_num}" if linkage_file_base else None

    try:
        corres_file = open(corres_fname, 'w')
    except IOError:
        return False

    linkage_file = None
    if linkage_fname:
        try:
            linkage_file = open(linkage_fname, 'w')
        except IOError:
            corres_file.close()
            return False

    prio_file = None
    if prio_file_base:
        prio_fname = f"{prio_file_base}.{frame_num}"
        try:
            prio_file = open(prio_fname, 'w')
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
        if isinstance(cor_buf, (list, tuple)) and len(cor_buf) == 2 and isinstance(cor_buf[0], np.ndarray):
            c_nr = int(cor_buf[0][pix])
            c_p = cor_buf[1][pix]
        elif isinstance(cor_buf, list) and isinstance(cor_buf[0], Corres):
            c_p = cor_buf[pix].p
        else:
            c_p = cor_buf[pix].p if hasattr(cor_buf[pix], 'p') else np.zeros(4, dtype=np.int32)

        if linkage_file:
            linkage_file.write(
                f"{p.prev:4d} {p.next:4d} {p.x[0]:10.3f} {p.x[1]:10.3f} {p.x[2]:10.3f}\n"
            )

        corres_file.write(
            f"{pix + 1:4d} {p.x[0]:9.3f} {p.x[1]:9.3f} {p.x[2]:9.3f} "
            f"{c_p[0]:4d} {c_p[1]:4d} {c_p[2]:4d} {c_p[3]:4d}\n"
        )

        if prio_file:
            prio_file.write(
                f"{p.prev:4d} {p.next:4d} {p.x[0]:10.3f} {p.x[1]:10.3f} {p.x[2]:10.3f} {p.prio:d}\n"
            )

    corres_file.close()
    if linkage_file:
        linkage_file.close()
    if prio_file:
        prio_file.close()

    return True


class Frame:
    def __init__(self, num_cams=4, max_targets=100):
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.targets = [[Target() for _ in range(max_targets)] for _ in range(num_cams)]
        self.correspond = [Corres() for _ in range(max_targets)]
        self.path_info = [Pathinfo() for _ in range(max_targets)]
        self.num_targets = [0] * num_cams
        self.num_parts = 0

    def read(self, corres_file_base, linkage_file_base, prio_file_base,
             target_file_base, frame_num):
        fname = f"{corres_file_base}.{frame_num}"
        if not Path(fname).exists():
            return False

        cor_list, path_list = read_path_frame(
            corres_file_base,
            linkage_file_base if linkage_file_base else "",
            prio_file_base if prio_file_base else "",
            frame_num
        )

        self.num_parts = len(cor_list)
        for i in range(self.num_parts):
            self.correspond[i] = cor_list[i]
            self.path_info[i] = path_list[i]

        for cam in range(self.num_cams):
            targets = read_targets(target_file_base[cam], frame_num)
            self.num_targets[cam] = len(targets)
            for j, t in enumerate(targets):
                self.targets[cam][j] = t

        return True

    def write(self, corres_file_base, linkage_file_base, prio_file_base,
              target_file_base, frame_num):
        ok = write_path_frame(
            self.correspond, self.path_info, self.num_parts,
            corres_file_base, linkage_file_base,
            prio_file_base if prio_file_base else None,
            frame_num
        )
        if not ok:
            return False

        for cam in range(self.num_cams):
            if self.num_targets[cam] > 0:
                ok = write_targets(
                    self.targets[cam], self.num_targets[cam],
                    target_file_base[cam], frame_num
                )
                if not ok:
                    return False

        return True


class FrameBuf:
    def __init__(self, buf_len, num_cams, max_targets,
                 corres_file_base, linkage_file_base, prio_file_base,
                 target_file_base):
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
        return frame.read(self.corres_file_base, linkage, prio,
                          self.target_file_base, frame_num)

    def write_frame_from_start(self, frame_num):
        frame = self.buf[0]
        return frame.write(self.corres_file_base, self.linkage_file_base,
                           self.prio_file_base, self.target_file_base,
                           frame_num)


Corres_dtype = np.dtype([('nr', np.int32), ('p', np.int32, (4,))])
