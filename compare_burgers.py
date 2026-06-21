"""Compare tracking cases on burgers data:
1. trackcorr Python
2. trackcorr Cython
3. track3d Python
4. track3d Cython

Compares per-step linkage files (ptv_is.*) field by field.
"""
import os
import sys
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def read_all_calibration(num_cams, base_path="."):
    from algorithms.calibration import Calibration
    cals = []
    for cam in range(num_cams):
        ori = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        add = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cals.append(Calibration.from_file(ori, add))
    return cals


def parse_linkage_file(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    particles = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        particles.append({
            "prev": int(parts[0]),
            "next": int(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
        })
    return particles


def parse_corres_file(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    particles = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        particles.append({
            "nr": int(parts[0]),
            "x": float(parts[1]),
            "y": float(parts[2]),
            "z": float(parts[3]),
            "p": [int(parts[4 + j]) for j in range(4)],
        })
    return particles


def reset_test_data():
    if os.path.exists("res"):
        shutil.rmtree("res")
    if os.path.exists("img"):
        shutil.rmtree("img")
    shutil.copytree("res_orig", "res")
    shutil.copytree("img_orig", "img")


def collect_files(first, last, linkage_base="res/ptv_is", corres_base="res/rt_is"):
    data = {}
    for s in range(first, last + 1):
        lf = f"{linkage_base}.{s}"
        cf = f"{corres_base}.{s}"
        entry = {}
        if os.path.exists(lf):
            entry["linkage"] = parse_linkage_file(lf)
        if os.path.exists(cf):
            entry["corres"] = parse_corres_file(cf)
        data[s] = entry
    return data


def run_trackcorr_python(first, last):
    """Run trackcorr with Python algorithms."""
    from algorithms.track import track_forward_start, trackcorr_c_loop, trackcorr_c_finish
    from algorithms.parameters import read_control_par
    from algorithms.tracking_run import tr_new

    reset_test_data()
    cpar = read_control_par("parameters/ptv.par")
    calib = read_all_calibration(cpar.num_cams)
    run = tr_new(
        "parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
        "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added",
        calib, 0.0001
    )
    track_forward_start(run)
    step_links = []
    for step in range(run.seq_par.first, run.seq_par.last):
        old_nlinks = run.nlinks
        trackcorr_c_loop(run, step)
        step_links.append(int(run.nlinks - old_nlinks))
    trackcorr_c_finish(run, run.seq_par.last)

    result = collect_files(first, last)
    return result, run.npart, run.nlinks, step_links


def _make_cython_tracker(first, last):
    """Create a Cython tracker for the burgers test case."""
    from optv.tracker import Tracker
    from optv.calibration import Calibration as CCalib
    from optv.parameters import ControlParams, VolumeParams, TrackingParams, SequenceParams

    cpar = ControlParams(4)
    cpar.read_control_par("parameters/ptv.par")
    vpar = VolumeParams()
    vpar.read_volume_par("parameters/criteria.par")
    tpar = TrackingParams()
    tpar.read_track_par("parameters/track.par")
    img_base = [f"img/cam{i + 1}." for i in range(4)]
    spar = SequenceParams(image_base=img_base, frame_range=(first, last))
    cal = []
    for i in range(4):
        c = CCalib()
        c.from_file(f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar")
        cal.append(c)

    naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}
    return Tracker(cpar, vpar, tpar, spar, cal, naming)


def run_trackcorr_cython(first, last):
    """Run trackcorr with Cython (optv) — uses full_forward() = trackcorr_c_loop."""
    reset_test_data()
    tracker = _make_cython_tracker(first, last)
    tracker.full_forward()
    return collect_files(first, last)


def run_track3d_cython_native(first, last):
    """Run track3d with Cython (optv) — uses full_forward_3d() = track3d_loop."""
    reset_test_data()
    tracker = _make_cython_tracker(first, last)
    tracker.full_forward_3d()
    return collect_files(first, last)


def run_track3d_python(first, last):
    """Run track3d with Python algorithms."""
    from algorithms.track3d import track3d_loop
    from algorithms.track import track_forward_start, trackcorr_c_finish
    from algorithms.parameters import read_control_par
    from algorithms.tracking_run import tr_new

    reset_test_data()
    cpar = read_control_par("parameters/ptv.par")
    calib = read_all_calibration(cpar.num_cams)
    run = tr_new(
        "parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
        "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added",
        calib, 0.0001
    )
    run.tpar = run.tpar._replace(add=0)
    track_forward_start(run)
    step_links = []
    for step in range(run.seq_par.first, run.seq_par.last):
        old_nlinks = run.nlinks
        track3d_loop(run, step)
        step_links.append(int(run.nlinks - old_nlinks))
    trackcorr_c_finish(run, run.seq_par.last)

    result = collect_files(first, last)
    return result, run.npart, run.nlinks, step_links


def compare_linkage(name_a, data_a, name_b, data_b, steps):
    """Compare linkage files between two runs."""
    all_match = True
    for s in steps:
        la = data_a.get(s, {}).get("linkage", [])
        lb = data_b.get(s, {}).get("linkage", [])
        if len(la) != len(lb):
            print(f"  Step {s}: particle count {name_a}={len(la)} vs {name_b}={len(lb)}")
            all_match = False
            continue
        for i, (pa, pb) in enumerate(zip(la, lb)):
            diffs = []
            if pa["prev"] != pb["prev"]:
                diffs.append(f"prev={pa['prev']}vs{pb['prev']}")
            if pa["next"] != pb["next"]:
                diffs.append(f"next={pa['next']}vs{pb['next']}")
            dx = abs(pa["x"] - pb["x"])
            dy = abs(pa["y"] - pb["y"])
            dz = abs(pa["z"] - pb["z"])
            if dx > 1e-4 or dy > 1e-4 or dz > 1e-4:
                diffs.append(f"pos=({dx:.6f},{dy:.6f},{dz:.6f})")
            if diffs:
                all_match = False
                print(f"  Step {s} particle {i}: {', '.join(diffs)}")
    return all_match


def count_links_per_step(data, steps):
    """Count links (next >= 0, i.e. actual valid link index) per step."""
    result = {}
    for s in steps:
        linkage = data.get(s, {}).get("linkage", [])
        result[s] = sum(1 for p in linkage if p["next"] >= 0)
    return result


def main():
    original = os.getcwd()
    try:
        os.chdir(os.path.join(os.path.dirname(__file__), "test_data/burgers"))
        first, last = 10001, 10005
        steps = list(range(first, last + 1))

        print("=" * 80)
        print("RUNNING ALL 4 CASES ON BURGERS DATA")
        print("=" * 80)

        # Case 1: trackcorr Cython (C trackcorr_c_loop via full_forward)
        print("\n--- Case 1: trackcorr Cython (optv full_forward) ---")
        tc_cy = run_trackcorr_cython(first, last)

        # Case 2: trackcorr Python
        print("\n--- Case 2: trackcorr Python ---")
        tc_py_data, tc_py_npart, tc_py_nlinks, tc_py_step_links = run_trackcorr_python(first, last)

        # Case 3: track3d Cython (C track3d_loop via full_forward_3d)
        print("\n--- Case 3: track3d Cython (optv full_forward_3d) ---")
        t3_cy = run_track3d_cython_native(first, last)

        # Case 4: track3d Python
        print("\n--- Case 4: track3d Python ---")
        t3_py_data, t3_py_npart, t3_py_nlinks, t3_py_step_links = run_track3d_python(first, last)

        all_cases = [
            ("TC-Cy", tc_cy),
            ("TC-Py", tc_py_data),
            ("T3-Cy", t3_cy),
            ("T3-Py", t3_py_data),
        ]

        print("\n" + "=" * 80)
        print("LINK COUNTS PER STEP (next >= 0 in output file)")
        print("=" * 80)

        all_links = {name: count_links_per_step(data, steps) for name, data in all_cases}

        header = f"{'Step':>6}" + "".join(f" | {name:>6}" for name, _ in all_cases)
        print(header)
        print("-" * len(header))
        for s in steps:
            row = f"{s:>6}"
            for name, _ in all_cases:
                row += f" | {all_links[name][s]:>6}"
            print(row)
        totals = f"{'Total':>6}"
        for name, _ in all_cases:
            totals += f" | {sum(all_links[name].values()):>6}"
        print(totals)

        print(f"\ntrackcorr Python: npart={tc_py_npart}, nlinks={tc_py_nlinks}, step_links={tc_py_step_links}")
        print(f"track3d Python: npart={t3_py_npart}, nlinks={t3_py_nlinks}, step_links={t3_py_step_links}")

        print("\n" + "=" * 80)
        print("PARTICLE COUNTS PER STEP")
        print("=" * 80)
        header = f"{'Step':>6}" + "".join(f" | {name:>6}" for name, _ in all_cases)
        print(header)
        print("-" * len(header))
        for s in steps:
            row = f"{s:>6}"
            for _, data in all_cases:
                linkage = data.get(s, {}).get("linkage", [])
                row += f" | {len(linkage):>6}"
            print(row)

        print("\n" + "=" * 80)
        print("DETAILED PAIRWISE COMPARISONS")
        print("=" * 80)

        pairs = [
            ("TC-Cy", tc_cy, "TC-Py", tc_py_data, "trackcorr: Cython vs Python"),
            ("T3-Cy", t3_cy, "T3-Py", t3_py_data, "track3d: Cython vs Python"),
            ("TC-Cy", tc_cy, "T3-Cy", t3_cy, "Cython: trackcorr vs track3d"),
            ("TC-Py", tc_py_data, "T3-Py", t3_py_data, "Python: trackcorr vs track3d"),
        ]
        for na, da, nb, db, desc in pairs:
            print(f"\n--- {desc} ---")
            match = compare_linkage(na, da, nb, db, steps)
            if match:
                print("  PERFECT MATCH")

        print("\n" + "=" * 80)
        print("FULL PER-PARTICLE DUMP (all cases, all steps)")
        print("=" * 80)
        for s in steps:
            case_data = [(name, data.get(s, {}).get("linkage", [])) for name, data in all_cases]
            max_n = max(len(ld) for _, ld in case_data)
            if max_n == 0:
                continue
            print(f"\n  Step {s} ({max_n} particles):")
            hdr = f"  {'i':>3}"
            for name, _ in case_data:
                hdr += f" | {name+' prev':>10} {'next':>5} {'x':>10} {'y':>10} {'z':>10}"
            print(hdr)
            for i in range(max_n):
                def fmt(lst, idx):
                    if idx < len(lst):
                        p = lst[idx]
                        return f"{p['prev']:>10} {p['next']:>5} {p['x']:>10.4f} {p['y']:>10.4f} {p['z']:>10.4f}"
                    return f"{'---':>10} {'---':>5} {'---':>10} {'---':>10} {'---':>10}"
                row = f"  {i:>3}"
                for _, ld in case_data:
                    row += f" | {fmt(ld, i)}"
                print(row)

    finally:
        os.chdir(original)


if __name__ == "__main__":
    main()
