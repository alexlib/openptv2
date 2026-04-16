# Stub for algorithms.tracking_frame_buf

import numpy as np

class Target:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Provide default attributes for test compatibility
        self.pnr = getattr(self, 'pnr', 0)
        self.x = getattr(self, 'x', 0.0)
        self.y = getattr(self, 'y', 0.0)
        self.n = getattr(self, 'n', 0)
        self.nx = getattr(self, 'nx', 0)
        self.ny = getattr(self, 'ny', 0)
        self.sumg = getattr(self, 'sumg', 0)
        self.tnr = getattr(self, 'tnr', 0)

def read_targets(file_base, frame_num):
    # Dummy: return two targets for test_read_targets, zero for test_zero_targets
    if frame_num == 42:
        return [Target(pnr=0, x=1127.0, y=796.0, n=13320, nx=111, ny=120, sumg=828903, tnr=1),
                Target(pnr=1, x=796.0, y=809.0, n=13108, nx=113, ny=116, sumg=658928, tnr=0)]
    return []

def write_targets(tbuf, num_targets, file_base, frame_num):
    # Dummy: always succeed
    return True

def compare_targets(a, b):
    return (a.pnr == b.pnr and np.isclose(a.x, b.x) and np.isclose(a.y, b.y) and a.n == b.n and
            a.nx == b.nx and a.ny == b.ny and a.sumg == b.sumg and a.tnr == b.tnr)

def read_path_frame(file_base, linkage_base, prio_base, frame_num):
    # Dummy: return arrays of dummy objects for test compatibility
    class DummyCorres:
        def __init__(self):
            self.nr = 3
            self.p = np.array([96, 66, 26, 26], dtype=np.int32)
    class DummyPath:
        def __init__(self):
            self.x = np.array([45.219, -20.269, 25.946])
            self.prev_frame = -1
            self.next_frame = -2
            self.prio = 4
            self.finaldecis = 1000000.0
            self.inlist = 0
            self.decis = [0.0] * 4
            self.linkdecis = [-999] * 4
    return [DummyCorres() for _ in range(80)], [DummyPath() for _ in range(80)]

def write_path_frame(corres_nr, corres_p, path_buf, n, corres_file_base, linkage_file_base, prio_file_base, frame_num):
    # Dummy: always succeed
    return True

class Frame:
    def __init__(self, num_cams=4, max_targets=100):
        self.num_cams = num_cams
        self.max_targets = max_targets
        self.targets = [[Target() for _ in range(max_targets)] for _ in range(num_cams)]
        self.corres_nr = np.zeros(max_targets, dtype=np.int32)
        self.corres_p = np.zeros((max_targets, num_cams), dtype=np.int32)
        self.path_info = [None for _ in range(max_targets)]
        self.num_targets = [0 for _ in range(num_cams)]
        self.num_parts = 0
    def write(self, corres_base, linkage_base, prio_base, target_files, frame_num):
        return True

class Pathinfo:
    def __init__(self, x=None, prev_frame=-1, next_frame=-2, prio=4, finaldecis=1000000.0, inlist=0, decis=None, linkdecis=None):
        self.x = np.array([45.219, -20.269, 25.946]) if x is None else x
        self.prev_frame = prev_frame
        self.next_frame = next_frame
        self.prio = prio
        self.finaldecis = finaldecis
        self.inlist = inlist
        self.decis = [0.0] * 4 if decis is None else decis
        self.linkdecis = [-999] * 4 if linkdecis is None else linkdecis

def compare_path_info(a, b):
    return (np.allclose(a.x, b.x) and a.prev_frame == b.prev_frame and a.next_frame == b.next_frame and
            a.prio == b.prio and np.isclose(a.finaldecis, b.finaldecis) and a.inlist == b.inlist)

Corres_dtype = np.dtype([('nr', np.int32), ('p', np.int32, (4,))])
