"""Inspect Class-A refusals: is the candidate even visible to the search?"""

from pathlib import Path

import numpy as np

DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"


def rp_of(d, f):
    l = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(l[0])
    return [int(x.split()[0]) for x in l[1: n + 1]]


def rx(f):
    l = (REF / f"rt_is.{f}").read_text().splitlines()
    n = int(l[0])
    pos = [[float(v) for v in x.split()[1:4]] for x in l[1: n + 1]]
    pcols = [x.split()[4:8] for x in l[1: n + 1]]
    return pos, pcols


def main():
    import math
    for (f, i) in [(100006, 535), (100007, 421), (100006, 334),
                   (100006, 875)]:
        p = rp_of(REF, f)[i]
        X1, P1 = rx(f - 1)
        X3, P3 = rx(f)
        print(f"--- frame {f} row {i} ref_prev={p} ---")
        print(f"  rt frame{f - 1}[{p}] pcols={P1[p]}")
        print(f"  rt frame{f}[{i}] pcols={P3[i]}")
        d = np.array(X3[i]) - np.array(X1[p])
        print(f"  per-axis step={np.round(d, 3)} max={np.abs(d).max():.3f} "
              f"3D={np.linalg.norm(d):.3f}")
        pp = rp_of(REF, f - 1)[p]
        if pp >= 0:
            X0, _ = rx(f - 2)
            A, B, G = np.array(X1[p]), np.array(X3[i]), np.array(X0[pp])
            X2 = 2 * A - G
            v0, v1 = X2 - A, B - A
            n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
            if n0 == 0 or n1 == 0:
                ang = 0.0
            else:
                ang = math.acos(np.clip(np.dot(v0, v1) / (n0 * n1), -1, 1))
                ang *= 200 / math.pi
            print(f"  fallback: acc={np.linalg.norm(B - X2):.3f} "
                  f"angle={ang:.1f}gon grandparent={pp}")
        for cam in range(1, 5):
            j = P3[i][cam - 1]
            msg = f"  cam{cam}: rt->target {j}"
            if j != "-1":
                tl = (DS / "img_3dptv" / f"Cam{cam}.{f}_targets"
                      ).read_text().splitlines()
                rec = [li for li in tl[1:] if li.split()[0] == j]
                msg += (" file_tnr=" + rec[0].split()[7]) if rec else \
                    " TARGET-ROW-MISSING"
            print(msg)


if __name__ == "__main__":
    main()
