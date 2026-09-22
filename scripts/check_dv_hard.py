"""Per-axis legality of the 34 hard links vs dv=1.9."""

from pathlib import Path

import numpy as np

DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"


def read_pn(d, f):
    lines = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


def read_xyz(f):
    lines = (REF / f"rt_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()[1:4]] for l in lines[1: n + 1]])


def main():
    import csv
    rows = list(csv.DictReader(open("scratch/hard_links/hard_links.csv")))
    print("    frm  row     dx     dy     dz  maxax  dv")
    for r in rows:
        f, i = int(r["frame"]), int(r["row"])
        X1, X3 = read_xyz(f - 1), read_xyz(f)
        p = int(r["ref_prev"])
        d = np.array(X3[i]) - np.array(X1[p])
        m = float(np.abs(d).max())
        flag = "VIOL" if m >= 1.9 else "ok"
        print(f"{f:>7}{i:>5}{d[0]:>7.2f}{d[1]:>7.2f}{d[2]:>7.2f}"
              f"{m:>7.2f}  {flag}")


if __name__ == "__main__":
    main()
