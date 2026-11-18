"""Pure-Python line-coverage tests for tracking_frame_buf.py.

Run with:
    cd /home/user/Documents/GitHub/openptv2
    COVERAGE_FILE=/tmp/.cov_tfb uv run pytest tests/unit/test_tracking_frame_buf_coverage.py \
      -o pythonpath=/tmp/ppsrc \
      -p no:cacheprovider \
      --cov=/tmp/ppsrc/openptv2 \
      --cov-config=/tmp/covrc \
      --cov-report=term-missing \
      -q

Compiled-mode safety: skip tests that depend on pure-Python internals.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.constants import (
    CORRES_NONE,
    NEXT_NONE,
    POSI,
    PREV_NONE,
    PT_UNUSED,
)
from openptv2.algorithms.tracking_frame_buf import (
    CallableFloat,
    CallableInt,
    Corres,
    Corres_dtype,
    Frame,
    FrameBuf,
    Pathinfo,
    Target,
    TargetArray,
    compare_corres,
    compare_path_info,
    compare_targets,
    is_compiled,
    read_path_frame,
    read_targets,
    register_link_candidate,
    reset_links,
    write_path_frame,
    write_targets,
)

_needs_pure_python = pytest.mark.skipif(
    is_compiled(), reason="asserts is_compiled() is False by design"
)
_needs_pure_python_lazy_attrs = pytest.mark.skipif(
    is_compiled(),
    reason="tests Pathinfo._decis/_linkdecis, private lazy-init attributes "
    "that exist only on the pure-Python dataclass; the compiled "
    "@cython.cclass has fixed declared fields (decis/linkdecis directly, no "
    "leading-underscore backing attribute) -- AttributeError is expected",
)


# ===========================================================================
# CallableInt / CallableFloat
# ===========================================================================


class TestCallableInt:
    def test_is_int(self):
        ci = CallableInt(5)
        assert int(ci) == 5

    def test_call_returns_int(self):
        ci = CallableInt(7)
        assert ci() == 7
        assert isinstance(ci(), int)

    def test_zero(self):
        ci = CallableInt(0)
        assert ci() == 0


class TestCallableFloat:
    def test_is_float(self):
        cf = CallableFloat(3.14)
        assert float(cf) == pytest.approx(3.14)

    def test_call_returns_float(self):
        cf = CallableFloat(2.5)
        assert cf() == pytest.approx(2.5)
        assert isinstance(cf(), float)


# ===========================================================================
# Target
# ===========================================================================


class TestTarget:
    def test_default_init(self):
        t = Target()
        assert t.c_pnr == 0
        assert t.c_x == 0.0
        assert t.c_y == 0.0
        assert t.n == 0
        assert t.nx == 0
        assert t.ny == 0
        assert t.sumg == 0
        assert t.c_tnr == 0

    def test_custom_init(self):
        t = Target(pnr=1, x=2.0, y=3.0, n=4, nx=5, ny=6, sumg=7, tnr=8)
        assert t.c_pnr == 1
        assert t.c_x == pytest.approx(2.0)
        assert t.c_y == pytest.approx(3.0)
        assert t.n == 4
        assert t.nx == 5
        assert t.ny == 6
        assert t.sumg == 7
        assert t.c_tnr == 8

    def test_repr(self):
        t = Target(pnr=1, x=1.5, y=2.5)
        r = repr(t)
        assert "Target" in r
        assert "pnr=1" in r

    def test_pnr_property_get_set(self):
        t = Target()
        t.pnr = 42
        assert t.pnr == 42
        assert isinstance(t.pnr, CallableInt)

    def test_x_property_get_set(self):
        t = Target()
        t.x = 1.23
        assert t.x == pytest.approx(1.23)
        assert isinstance(t.x, CallableFloat)

    def test_y_property_get_set(self):
        t = Target()
        t.y = 4.56
        assert t.y == pytest.approx(4.56)
        assert isinstance(t.y, CallableFloat)

    def test_tnr_property_get_set(self):
        t = Target()
        t.tnr = 99
        assert t.tnr == 99
        assert isinstance(t.tnr, CallableInt)

    def test_set_pnr(self):
        t = Target()
        t.set_pnr(10)
        assert t.c_pnr == 10

    def test_pos(self):
        t = Target(x=1.0, y=2.0)
        pos = t.pos()
        assert list(pos) == [pytest.approx(1.0), pytest.approx(2.0)]

    def test_set_pos(self):
        t = Target()
        t.set_pos([3.0, 4.0])
        assert t.c_x == pytest.approx(3.0)
        assert t.c_y == pytest.approx(4.0)

    def test_set_tnr(self):
        t = Target()
        t.set_tnr(55)
        assert t.c_tnr == 55

    def test_count_pixels(self):
        t = Target(n=10, nx=5, ny=2)
        assert t.count_pixels() == (10, 5, 2)

    def test_set_pixel_counts(self):
        t = Target()
        t.set_pixel_counts(11, 6, 3)
        assert t.n == 11
        assert t.nx == 6
        assert t.ny == 3

    def test_sum_grey_value(self):
        t = Target(sumg=200)
        assert t.sum_grey_value() == 200

    def test_set_sum_grey_value(self):
        t = Target()
        t.set_sum_grey_value(128)
        assert t.sumg == 128


# ===========================================================================
# TargetArray
# ===========================================================================


class TestTargetArray:
    def test_init_int(self):
        ta = TargetArray(3)
        assert len(ta) == 3
        assert all(isinstance(t, Target) for t in ta)

    def test_init_list(self):
        targets = [Target(pnr=i) for i in range(2)]
        ta = TargetArray(targets)
        assert len(ta) == 2

    def test_init_invalid_type(self):
        with pytest.raises(TypeError):
            TargetArray("bad")

    def test_sort_y(self):
        t1 = Target(y=3.0)
        t2 = Target(y=1.0)
        t3 = Target(y=2.0)
        ta = TargetArray([t1, t2, t3])
        ta.sort_y()
        assert [t.c_y for t in ta] == [
            pytest.approx(1.0),
            pytest.approx(2.0),
            pytest.approx(3.0),
        ]

    def test_write_and_read_targets(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        base = str(tmp_path / "cam1")
        t1 = Target(pnr=0, x=1.0, y=2.0, n=10, nx=5, ny=3, sumg=100, tnr=7)
        ta = TargetArray([t1])
        result = ta.write(base, 1, store=store, cam_idx=0)
        assert result is True
        # read back using static method
        ta2 = TargetArray.read_targets(base, 1, store=store, cam_idx=0)
        assert len(ta2) == 1

    def test_write_zero_frame(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        base = str(tmp_path / "cam1")
        ta = TargetArray([Target(pnr=0, x=1.0, y=2.0, n=1, nx=1, ny=1, sumg=50, tnr=0)])
        result = ta.write(base, 0, store=store, cam_idx=0)
        assert result is True


# ===========================================================================
# compare_targets
# ===========================================================================


class TestCompareTargets:
    def test_equal(self):
        t1 = Target(pnr=1, x=1.0, y=2.0, n=3, nx=4, ny=5, sumg=6, tnr=7)
        t2 = Target(pnr=1, x=1.0, y=2.0, n=3, nx=4, ny=5, sumg=6, tnr=7)
        assert compare_targets(t1, t2) is True

    def test_not_equal_pnr(self):
        t1 = Target(pnr=1)
        t2 = Target(pnr=2)
        assert compare_targets(t1, t2) is False

    def test_not_equal_x(self):
        t1 = Target(x=1.0)
        t2 = Target(x=2.0)
        assert compare_targets(t1, t2) is False


# ===========================================================================
# read_targets / write_targets
# ===========================================================================


class TestReadWriteTargets:
    def test_write_then_read(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        base = str(tmp_path / "cam1")
        targets = [
            Target(pnr=i, x=float(i), y=float(i * 2), n=10, nx=5, ny=5, sumg=100, tnr=i)
            for i in range(3)
        ]
        ok = write_targets(targets, 3, base, 1, store=store, cam_idx=0)
        assert ok is True
        read = read_targets(base, 1, cam_idx=0, store=store)
        assert len(read) == 3
        assert read[0].c_pnr == 0

    def test_read_nonexistent(self, tmp_path):
        result = read_targets(str(tmp_path / "nofile"), 1)
        assert result == []

    def test_write_without_store_raises(self):
        with pytest.raises(ValueError, match="RunStore"):
            write_targets([], 0, "/nonexistent_dir_xyz/cam", 1)

    def test_write_zero_targets(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        base = str(tmp_path / "cam1")
        ok = write_targets([], 0, base, 1, store=store, cam_idx=0)
        assert ok is True
        read = read_targets(base, 1, cam_idx=0, store=store)
        assert read == []


# ===========================================================================
# Corres
# ===========================================================================


class TestCorres:
    def test_default_init(self):
        c = Corres()
        assert c.nr == 0
        assert list(c.p) == [CORRES_NONE] * 4

    def test_custom_init(self):
        c = Corres(nr=5, p=[0, 1, 2, 3])
        assert c.nr == 5
        assert list(c.p) == [0, 1, 2, 3]


class TestCompareCorres:
    def test_equal(self):
        c1 = Corres(nr=1, p=[0, 1, 2, 3])
        c2 = Corres(nr=1, p=[0, 1, 2, 3])
        assert compare_corres(c1, c2)

    def test_not_equal_nr(self):
        c1 = Corres(nr=1)
        c2 = Corres(nr=2)
        assert not compare_corres(c1, c2)

    def test_not_equal_p(self):
        c1 = Corres(nr=1, p=[0, 1, 2, 3])
        c2 = Corres(nr=1, p=[0, 1, 2, 9])
        assert not compare_corres(c1, c2)


# ===========================================================================
# Pathinfo
# ===========================================================================


class TestPathinfo:
    def test_default_init(self):
        p = Pathinfo()
        assert list(p.x) == [0.0, 0.0, 0.0]
        assert p.prev == PREV_NONE
        assert p.next_idx == NEXT_NONE
        assert p.prio == 4
        assert p.finaldecis == pytest.approx(1000000.0)
        assert p.inlist == 0

    def test_custom_init(self):
        p = Pathinfo(
            x=[1.0, 2.0, 3.0], prev=5, next_idx=6, prio=2, finaldecis=0.5, inlist=1
        )
        assert list(p.x) == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]
        assert p.prev == 5
        assert p.next_idx == 6
        assert p.prio == 2

    @_needs_pure_python_lazy_attrs
    def test_decis_lazy_init(self):
        p = Pathinfo()
        # _decis is None initially, accessing .decis materializes it
        assert p._decis is None
        d = p.decis
        assert len(d) == POSI
        assert all(v == 0.0 for v in d)

    @_needs_pure_python_lazy_attrs
    def test_linkdecis_lazy_init(self):
        p = Pathinfo()
        assert p._linkdecis is None
        ld = p.linkdecis
        assert len(ld) == POSI
        assert all(v == PT_UNUSED for v in ld)

    @_needs_pure_python_lazy_attrs
    def test_decis_setter(self):
        p = Pathinfo()
        p.decis = [1.0] * POSI
        assert p._decis == [1.0] * POSI

    @_needs_pure_python_lazy_attrs
    def test_linkdecis_setter(self):
        p = Pathinfo()
        p.linkdecis = [0] * POSI
        assert p._linkdecis == [0] * POSI

    @_needs_pure_python_lazy_attrs
    def test_init_with_decis_linkdecis(self):
        d = [0.5] * POSI
        ld = [1] * POSI
        p = Pathinfo(decis=d, linkdecis=ld)
        assert p._decis == d
        assert p._linkdecis == ld

    @_needs_pure_python_lazy_attrs
    def test_decis_already_materialized(self):
        p = Pathinfo()
        _ = p.decis  # materialize
        p._decis[0] = 99.0
        assert p.decis[0] == pytest.approx(99.0)

    @_needs_pure_python_lazy_attrs
    def test_linkdecis_already_materialized(self):
        p = Pathinfo()
        _ = p.linkdecis  # materialize
        p._linkdecis[0] = 42
        assert p.linkdecis[0] == 42


# ===========================================================================
# compare_path_info
# ===========================================================================


class TestComparePathInfo:
    def _make(self, **kw):
        p = Pathinfo(**kw)
        # materialize decis/linkdecis so comparison is deterministic
        _ = p.decis
        _ = p.linkdecis
        return p

    def test_equal(self):
        p1 = self._make()
        p2 = self._make()
        assert compare_path_info(p1, p2) is True

    def test_not_equal_prev(self):
        p1 = self._make(prev=0)
        p2 = self._make(prev=1)
        assert compare_path_info(p1, p2) is False

    def test_not_equal_next_idx(self):
        p1 = self._make(next_idx=0)
        p2 = self._make(next_idx=1)
        assert compare_path_info(p1, p2) is False

    def test_not_equal_prio(self):
        p1 = self._make(prio=1)
        p2 = self._make(prio=2)
        assert compare_path_info(p1, p2) is False

    def test_not_equal_finaldecis(self):
        p1 = self._make(finaldecis=1.0)
        p2 = self._make(finaldecis=2.0)
        assert compare_path_info(p1, p2) is False

    def test_not_equal_inlist(self):
        p1 = self._make(inlist=0)
        p2 = self._make(inlist=1)
        assert compare_path_info(p1, p2) is False

    def test_not_equal_x(self):
        p1 = self._make(x=[1.0, 0.0, 0.0])
        p2 = self._make(x=[2.0, 0.0, 0.0])
        assert compare_path_info(p1, p2) is False

    def test_not_equal_decis(self):
        p1 = self._make()
        p2 = self._make()
        p1.decis[0] = 1.0
        p2.decis[0] = 2.0
        assert compare_path_info(p1, p2) is False

    def test_not_equal_linkdecis(self):
        p1 = self._make()
        p2 = self._make()
        p1.linkdecis[0] = 1
        p2.linkdecis[0] = 2
        assert compare_path_info(p1, p2) is False


# ===========================================================================
# register_link_candidate / reset_links
# ===========================================================================


class TestRegisterLinkCandidate:
    def test_basic(self):
        p = Pathinfo()
        register_link_candidate(p, 0.5, 3)
        assert p.linkdecis[0] == 3
        assert p.decis[0] == pytest.approx(0.5)
        assert p.inlist == 1

    def test_multiple(self):
        p = Pathinfo()
        register_link_candidate(p, 0.1, 10)
        register_link_candidate(p, 0.2, 20)
        assert p.inlist == 2
        assert p.linkdecis[1] == 20


class TestResetLinks:
    def test_reset(self):
        p = Pathinfo(prev=5, next_idx=6, prio=3)
        reset_links(p)
        assert p.prev == PREV_NONE
        assert p.next_idx == NEXT_NONE
        assert p.prio == 2


# ===========================================================================
# read_path_frame / write_path_frame
# ===========================================================================


def _make_corres_file(path, entries):
    """Write a corres file: header line then one entry per (x, y, z, p0..p3)."""
    with open(path, "w") as f:
        f.write(f"{len(entries)}\n")
        for i, (x, y, z, p0, p1, p2, p3) in enumerate(entries):
            f.write(
                f"{i + 1:4d} {x:9.3f} {y:9.3f} {z:9.3f} {p0:4d} {p1:4d} {p2:4d} {p3:4d}\n"
            )


def _make_linkage_file(path, entries):
    """Write a linkage file: header then (prev, next, x, y, z) per entry."""
    with open(path, "w") as f:
        f.write(f"{len(entries)}\n")
        for prev_v, next_v, x, y, z in entries:
            f.write(f"{prev_v:4d} {next_v:4d} {x:10.3f} {y:10.3f} {z:10.3f}\n")


def _make_prio_file(path, entries):
    """Write a prio file: header then (prev, next, x, y, z, prio) per entry."""
    with open(path, "w") as f:
        f.write(f"{len(entries)}\n")
        for prev_v, next_v, x, y, z, prio in entries:
            f.write(f"{prev_v:4d} {next_v:4d} {x:10.3f} {y:10.3f} {z:10.3f} {prio:d}\n")


class TestReadPathFrame:
    def test_nonexistent_corres(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cor, path = read_path_frame("rt_is", "", "", 1, store=store)
        assert cor == []
        assert path == []

    def test_none_store_returns_empty(self):
        cor, path = read_path_frame("rt_is", "", "", 1, store=None)
        assert cor == []
        assert path == []

    def test_simple_read(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        store.write_correspondences(
            frame=1,
            pos_3d=np.array([[1.0, 2.0, 3.0]]),
            cam_target_ids=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        cor, path = read_path_frame("rt_is", "", "", 1, store=store)
        assert len(cor) == 1
        assert cor[0].nr == 1
        assert list(cor[0].p) == [0, 1, 2, 3]
        assert list(path[0].x) == [
            pytest.approx(1.0),
            pytest.approx(2.0),
            pytest.approx(3.0),
        ]

    def test_read_with_linkage(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        pos = np.array([[1.0, 2.0, 3.0]])
        store.write_correspondences(
            frame=1, pos_3d=pos, cam_target_ids=np.array([[0, 1, 2, 3]], dtype=np.int32)
        )
        store.write_linkage(
            frame=1,
            prev_ids=np.array([-1], dtype=np.int32),
            next_ids=np.array([-2], dtype=np.int32),
            pos_3d=pos,
            name="ptv_is",
        )
        cor, path = read_path_frame("rt_is", "ptv_is", "", 1, store=store)
        assert len(path) == 1
        assert path[0].prev == -1
        assert path[0].next_idx == -2

    def test_read_with_prio(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        pos = np.array([[1.0, 2.0, 3.0]])
        store.write_correspondences(
            frame=2, pos_3d=pos, cam_target_ids=np.array([[0, 1, 2, 3]], dtype=np.int32)
        )
        store.write_linkage(
            frame=2,
            prev_ids=np.array([-1], dtype=np.int32),
            next_ids=np.array([-2], dtype=np.int32),
            pos_3d=pos,
            name="ptv_is",
            prio=np.array([3], dtype=np.int32),
        )
        cor, path = read_path_frame("rt_is", "ptv_is", "added", 2, store=store)
        assert path[0].prio == 3

    def test_read_nonexistent_linkage(self, tmp_path):
        """Linkage group does not exist → defaults to PREV_NONE / NEXT_NONE / prio=4."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        store.write_correspondences(
            frame=1,
            pos_3d=np.array([[1.0, 2.0, 3.0]]),
            cam_target_ids=np.array([[0, 1, 2, 3]], dtype=np.int32),
        )
        cor, path = read_path_frame("rt_is", "ptv_is", "", 1, store=store)
        assert len(path) == 1
        assert path[0].prev == PREV_NONE
        assert path[0].next_idx == NEXT_NONE
        assert path[0].prio == 4

    def test_read_multiple_entries(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        pos = np.array([[float(i), 0.0, 0.0] for i in range(5)])
        cam_ids = np.array([[i, i, i, i] for i in range(5)], dtype=np.int32)
        store.write_correspondences(frame=3, pos_3d=pos, cam_target_ids=cam_ids)
        cor, path = read_path_frame("rt_is", "", "", 3, store=store)
        assert len(cor) == 5


class TestWritePathFrame:
    def test_write_requires_store(self):
        with pytest.raises(ValueError, match="RunStore"):
            write_path_frame([], [], 0, "rt_is", "", None, 1, store=None)

    def test_write_then_read_roundtrip(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cor_buf = [Corres(nr=1, p=[0, 1, 2, 3])]
        p = Pathinfo(x=[1.0, 2.0, 3.0], prev=-1, next_idx=-2, prio=3)

        ok = write_path_frame(
            cor_buf, [p], 1, "rt_is", "ptv_is", "added", 5, store=store
        )
        assert ok is True
        assert store.has_correspondences(5)
        assert store.has_linkage(5, "ptv_is")

        cor2, path2 = read_path_frame("rt_is", "ptv_is", "added", 5, store=store)
        assert len(cor2) == 1
        assert path2[0].prio == 3

    def test_write_no_linkage_no_prio(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cor_buf = [Corres(nr=1, p=[0, 1, 2, 3])]
        p = Pathinfo(x=[1.0, 2.0, 3.0])
        ok = write_path_frame(cor_buf, [p], 1, "rt_is", "", None, 1, store=store)
        assert ok is True
        assert store.has_correspondences(1)
        assert not store.has_linkage(1, "ptv_is")

    def test_write_with_numpy_array_cor_buf(self, tmp_path):
        """Branch: cor_buf is (nr_array, p_array) tuple of ndarrays."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        nr_arr = np.array([1], dtype=np.int32)
        p_arr = np.array([[0, 1, 2, 3]], dtype=np.int32)
        p = Pathinfo(x=[1.0, 2.0, 3.0])
        ok = write_path_frame((nr_arr, p_arr), [p], 1, "rt_is", "", None, 1, store=store)
        assert ok is True

    def test_write_with_generic_object_cor_buf(self, tmp_path):
        """Branch: cor_buf[pix] has .p attribute (else branch)."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")

        class FakeCor:
            def __init__(self):
                self.p = np.zeros(4, dtype=np.int32)

        p = Pathinfo(x=[1.0, 2.0, 3.0])
        cor_buf = (FakeCor(),)
        ok = write_path_frame(cor_buf, [p], 1, "rt_is", "", None, 2, store=store)
        assert ok is True

    def test_write_with_no_p_attr(self, tmp_path):
        """Branch: cor_buf[pix] has no .p attribute → zeros."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")

        class NoPAttr:
            pass

        p = Pathinfo(x=[0.0, 0.0, 0.0])
        cor_buf = (NoPAttr(),)
        ok = write_path_frame(cor_buf, [p], 1, "rt_is", "", None, 3, store=store)
        assert ok is True

    def test_write_zero_parts(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        ok = write_path_frame([], [], 0, "rt_is", "ptv_is", None, 1, store=store)
        assert ok is True


# ===========================================================================
# Frame
# ===========================================================================


class TestFrame:
    def test_default_init(self):
        frm = Frame()
        assert frm.num_cams == 4
        assert frm.max_targets == 1000
        assert frm.num_parts == 0
        assert len(frm.targets) == 4
        assert len(frm.correspond) == 1000
        assert len(frm.path_info) == 1000

    def test_custom_init(self):
        frm = Frame(num_cams=2, max_targets=50)
        assert frm.num_cams == 2
        assert frm.max_targets == 50
        assert frm.targ_x.shape == (2, 50)
        assert frm.path_x.shape == (50, 3)

    def test_positions_empty(self):
        frm = Frame(num_cams=1, max_targets=10)
        pos = frm.positions()
        assert pos.shape == (0, 3)

    def test_positions_with_parts(self):
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 2
        frm.path_info[0].x[:] = [1.0, 2.0, 3.0]
        frm.path_info[1].x[:] = [4.0, 5.0, 6.0]
        pos = frm.positions()
        assert pos.shape == (2, 3)
        assert list(pos[0]) == [
            pytest.approx(1.0),
            pytest.approx(2.0),
            pytest.approx(3.0),
        ]

    def test_target_positions_for_camera(self):
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_targets[0] = 2
        frm.targets[0][0].c_x = 1.0
        frm.targets[0][0].c_y = 2.0
        frm.targets[0][1].c_x = 3.0
        frm.targets[0][1].c_y = 4.0
        pos = frm.target_positions_for_camera(0)
        assert pos.shape == (2, 2)
        assert pos[0, 0] == pytest.approx(1.0)
        assert pos[1, 1] == pytest.approx(4.0)

    def test_sync_path_to_soa(self):
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 2
        frm.path_info[0].x[:] = [1.0, 2.0, 3.0]
        frm.path_info[0].prev = 5
        frm.path_info[0].next_idx = 6
        frm.path_info[0].prio = 3
        frm.path_info[0].inlist = 1
        frm.path_info[0].finaldecis = 0.1
        frm.path_info[0].decis[0] = 0.5
        frm.path_info[0].linkdecis[0] = 7

        frm.correspond[0].nr = 42
        frm.correspond[0].p[:] = [1, 2, 3, 4]

        frm._sync_path_to_soa()

        assert list(frm.path_x[0]) == [
            pytest.approx(1.0),
            pytest.approx(2.0),
            pytest.approx(3.0),
        ]
        assert frm.path_prev[0] == 5
        assert frm.path_next[0] == 6
        assert frm.path_prio[0] == 3
        assert frm.path_inlist[0] == 1
        assert frm.path_finaldecis[0] == pytest.approx(0.1)
        assert frm.path_decis[0, 0] == pytest.approx(0.5)
        assert frm.path_linkdecis[0, 0] == 7
        assert frm.corres_nr[0] == 42
        assert list(frm.corres_p[0]) == [1, 2, 3, 4]

    def test_sync_soa_to_path(self):
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 2
        frm.path_x[0] = [7.0, 8.0, 9.0]
        frm.path_prev[0] = 11
        frm.path_next[0] = 12
        frm.path_prio[0] = 1
        frm.path_inlist[0] = 2
        frm.path_finaldecis[0] = 99.0
        frm.path_decis[0, 0] = 3.14
        frm.path_linkdecis[0, 0] = 55

        frm.corres_nr[0] = 77
        frm.corres_p[0] = [10, 11, 12, 13]

        frm._sync_soa_to_path()

        p = frm.path_info[0]
        assert list(p.x) == [pytest.approx(7.0), pytest.approx(8.0), pytest.approx(9.0)]
        assert p.prev == 11
        assert p.next_idx == 12
        assert p.prio == 1
        assert p.inlist == 2
        assert p.finaldecis == pytest.approx(99.0)
        assert p.decis[0] == pytest.approx(3.14)
        assert p.linkdecis[0] == 55

        c = frm.correspond[0]
        assert c.nr == 77
        assert list(c.p) == [10, 11, 12, 13]

    def test_read_nonexistent_returns_false(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        frm = Frame(num_cams=1, max_targets=10)
        result = frm.read("rt_is", "", frame_num=1, store=store)
        assert result is False

    def test_read_and_write_roundtrip(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        pbase = "added"
        tbase = "cam1."

        t = Target(pnr=0, x=1.0, y=2.0, n=10, nx=5, ny=5, sumg=100, tnr=0)
        write_targets([t], 1, tbase, 7, store=store, cam_idx=0)

        cor_buf = [Corres(nr=1, p=[0, -1, -1, -1])]
        path_buf = [Pathinfo(x=[1.0, 2.0, 3.0])]
        write_path_frame(cor_buf, path_buf, 1, cbase, lbase, pbase, 7, store=store)

        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read(cbase, lbase, pbase, [tbase], 7, store=store)
        assert ok is True
        assert frm.num_parts == 1
        assert frm.num_targets[0] == 1

    def test_read_legacy_args(self, tmp_path):
        """Exercise the is_legacy branch: target_file_base is list, frame_num is int."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        tbase = ["cam1."]

        write_targets(
            [Target(pnr=0, x=1.0, y=2.0, n=1, nx=1, ny=1, sumg=50, tnr=0)],
            1,
            tbase[0],
            2,
            store=store,
            cam_idx=0,
        )
        write_path_frame(
            [Corres(nr=1, p=[0, -1, -1, -1])],
            [Pathinfo(x=[1.0, 2.0, 3.0])],
            1,
            cbase,
            lbase,
            None,
            2,
            store=store,
        )

        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read(cbase, lbase, tbase, 2, store=store)
        assert ok is True

    def test_read_with_string_target_file_base(self, tmp_path):
        """Branch: target_file_base is a string (not list)."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        tbase = "cam1."

        write_targets(
            [Target(pnr=0, x=1.0, y=2.0, n=1, nx=1, ny=1, sumg=50, tnr=0)],
            1,
            tbase,
            3,
            store=store,
            cam_idx=0,
        )
        write_path_frame(
            [Corres(nr=1, p=[0, -1, -1, -1])],
            [Pathinfo(x=[1.0, 2.0, 3.0])],
            1,
            cbase,
            lbase,
            None,
            3,
            store=store,
        )

        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read(cbase, lbase, None, tbase, 3, store=store)
        assert ok is True

    def test_read_kwargs_prio_and_target(self, tmp_path):
        """Cover prio_file_base and target_file_base passed as kwargs."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        pbase = "added"
        tbase = ["cam1."]

        write_targets(
            [Target(pnr=0, x=1.0, y=2.0, n=1, nx=1, ny=1, sumg=50, tnr=0)],
            1,
            tbase[0],
            4,
            store=store,
            cam_idx=0,
        )
        write_path_frame(
            [Corres(nr=1, p=[0, -1, -1, -1])],
            [Pathinfo(x=[1.0, 2.0, 3.0])],
            1,
            cbase,
            lbase,
            pbase,
            4,
            store=store,
        )

        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read(
            cbase,
            lbase,
            prio_file_base=pbase,
            target_file_base=tbase,
            frame_num=4,
            store=store,
        )
        assert ok is True

    def test_read_legacy_target_file_base_preset_via_kwargs(self, tmp_path):
        """Legacy branch: target_file_base already set via kwarg."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read("rt_is", "ptv_is", 99, 99, target_file_base=["cam1."], store=store)
        assert ok is False

    def test_read_legacy_target_and_frame_num_preset(self, tmp_path):
        """Legacy branch: both target_file_base and frame_num preset."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read(
            "rt_is", "ptv_is", 99, 99, target_file_base=["cam1."], frame_num=99, store=store
        )
        assert ok is False

    def test_read_legacy_with_prio_in_args(self, tmp_path):
        """Legacy branch with 3 positional args: tbase, frame_num, prio_base."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        pbase = "added"
        tbase = ["cam1."]

        write_targets(
            [Target(pnr=0, x=1.0, y=2.0, n=1, nx=1, ny=1, sumg=50, tnr=0)],
            1,
            tbase[0],
            9,
            store=store,
            cam_idx=0,
        )
        write_path_frame(
            [Corres(nr=1, p=[0, -1, -1, -1])],
            [Pathinfo(x=[1.0, 2.0, 3.0])],
            1,
            cbase,
            lbase,
            pbase,
            9,
            store=store,
        )

        frm = Frame(num_cams=1, max_targets=10)
        ok = frm.read(cbase, lbase, tbase, 9, pbase, store=store)
        assert ok is True

    def test_write(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        pbase = "added"
        tbase = ["cam1."]

        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.correspond[0] = Corres(nr=1, p=[0, 1, 2, 3])
        frm.path_info[0] = Pathinfo(x=[1.0, 2.0, 3.0])
        frm.num_targets[0] = 1
        frm.targets[0][0] = Target(pnr=0, x=1.0, y=2.0, n=5, nx=3, ny=3, sumg=80, tnr=0)

        ok = frm.write(cbase, lbase, pbase, tbase, 1, store=store)
        assert ok is True
        assert store.has_correspondences(1)
        assert store.has_linkage(1, "ptv_is")
        assert store.has_targets(0, 1)

    def test_write_no_targets(self, tmp_path):
        """Write branch: num_targets[cam] == 0 skips write_targets."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 0
        frm.num_targets[0] = 0

        ok = frm.write("rt_is", "ptv_is", None, ["cam1."], 1, store=store)
        assert ok is True
        assert store.has_correspondences(1)

    def test_write_requires_store(self):
        """Frame.write raises ValueError when store is None."""
        frm = Frame(num_cams=1, max_targets=10)
        frm.num_parts = 1
        frm.correspond[0] = Corres(nr=1, p=[0, 1, 2, 3])
        frm.path_info[0] = Pathinfo(x=[1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="RunStore"):
            frm.write("rt_is", "ptv_is", None, ["cam1."], 1, store=None)

    def test_init_with_frame_num(self, tmp_path):
        """Frame.__init__ kwargs branch: read called when frame_num provided."""
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        frm = Frame(
            num_cams=1,
            max_targets=10,
            frame_num=1,
            corres_file_base="rt_is",
            linkage_file_base="ptv_is",
            prio_file_base="added",
            target_file_base=["cam1."],
            store=store,
        )
        assert frm.num_parts == 0  # nothing read (frame missing)


# ===========================================================================
# FrameBuf
# ===========================================================================


class TestFrameBuf:
    def _make_fb(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        return FrameBuf(
            buf_len=3,
            num_cams=1,
            max_targets=10,
            corres_file_base="rt_is",
            linkage_file_base="ptv_is",
            prio_file_base="added",
            target_file_base=["cam1."],
            store=store,
        )

    def test_init(self, tmp_path):
        fb = self._make_fb(tmp_path)
        assert fb.buf_len == 3
        assert fb.num_cams == 1
        assert len(fb.buf) == 3
        assert len(fb._frames) == 3

    def test_buf_start(self, tmp_path):
        fb = self._make_fb(tmp_path)
        assert fb._buf_start == 0

    def test_fb_next(self, tmp_path):
        fb = self._make_fb(tmp_path)
        first = fb.buf[0]
        fb.fb_next()
        assert fb.buf[-1] is first

    def test_fb_prev(self, tmp_path):
        fb = self._make_fb(tmp_path)
        last = fb.buf[-1]
        fb.fb_prev()
        assert fb.buf[0] is last

    def test_read_frame_at_end_no_file(self, tmp_path):
        fb = self._make_fb(tmp_path)
        result = fb.read_frame_at_end(99, read_links=True)
        assert result is False

    def test_read_frame_at_end_no_links(self, tmp_path):
        fb = self._make_fb(tmp_path)
        result = fb.read_frame_at_end(99, read_links=False)
        assert result is False

    def test_write_frame_from_start(self, tmp_path):
        fb = self._make_fb(tmp_path)
        fb.buf[0].num_parts = 0
        ok = fb.write_frame_from_start(1)
        assert ok is True
        assert fb.store.has_correspondences(1)

    def test_read_then_write_roundtrip(self, tmp_path):
        from openptv2.storage import RunStore

        store = RunStore(tmp_path / "run.zarr", mode="w")
        cbase = "rt_is"
        lbase = "ptv_is"
        pbase = "added"
        tbase = ["cam1."]

        write_targets(
            [Target(pnr=0, x=1.0, y=2.0, n=1, nx=1, ny=1, sumg=50, tnr=0)],
            1,
            tbase[0],
            5,
            store=store,
            cam_idx=0,
        )
        write_path_frame(
            [Corres(nr=1, p=[0, -1, -1, -1])],
            [Pathinfo(x=[1.0, 2.0, 3.0])],
            1,
            cbase,
            lbase,
            pbase,
            5,
            store=store,
        )

        fb = FrameBuf(
            buf_len=3,
            num_cams=1,
            max_targets=10,
            corres_file_base=cbase,
            linkage_file_base=lbase,
            prio_file_base=pbase,
            target_file_base=tbase,
            store=store,
        )
        ok = fb.read_frame_at_end(5, read_links=True)
        assert ok is True
        assert fb.buf[fb.buf_len - 1].num_parts == 1

        ok2 = fb.write_frame_from_start(6)
        assert ok2 is True
        assert store.has_correspondences(6)


# ===========================================================================
# Corres_dtype / is_compiled
# ===========================================================================


def test_corres_dtype():
    assert Corres_dtype is not None
    assert Corres_dtype.names == ("nr", "p")


@_needs_pure_python
def test_is_compiled_returns_bool():
    result = is_compiled()
    assert isinstance(result, bool)
    # Since we skipped if compiled, must be False here
    assert result is False
