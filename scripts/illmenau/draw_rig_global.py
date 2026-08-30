"""Global lab frame + camera rig, 3D (style of Multiview-Calibration multiview_calibration.py:157).

Global frame == the .ori frame:
  +X left->right,  +Y bottom->top (up, gravity is -Y),  +Z object->camera.
  Origin = the coded L-corner dot of frame 00000000 (plate.yaml datum), 615 mm
  above the heating plate
  in the barrel frame that point is (0, -1175, 0).
Barrel axis is along +Y.  Cameras 1-4 all sit on the +Z side
5-8 opposite at -Z.
"""
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

base = Path(__file__).parent
plate = yaml.safe_load((base / "plate.yaml").read_text())["plate"]
R = plate["test_section"]["radius"]
H = plate["test_section"]["height"]
Y0 = plate["test_section"]["height"] / 2 + plate["datum"]["barrel_frame"][1]   # datum height above the floor [mm] = 3580/2 - 1175 = 615
y_floor, y_ceil = -Y0, H - Y0
hl, hh = (plate["cameras"]["heights"][k] - Y0 for k in ("low", "high"))

nominal = {}
for cid, (sx, y) in {1: (+1, hl), 2: (+1, hh), 3: (-1, hl), 4: (-1, hh)}.items():
    nominal[cid] = np.array([sx * R / np.sqrt(2), y, +R / np.sqrt(2)])
for cid in (5, 6, 7, 8):
    p = nominal[cid - 4].copy()
    p[[0, 2]] *= -1
    nominal[cid] = p

rig = yaml.safe_load((base / "rig.yaml").read_text())
calib = {c["id"]: np.asarray(c["position"], float) for c in rig["cameras"]}

px, py = plate["pitch_x"], plate["pitch_y"]
gx, gy = np.meshgrid((np.arange(plate["nx"]) - (plate["nx"] - 1) / 2) * px,
                     (np.arange(plate["ny"]) - (plate["ny"] - 1) / 2) * py)
th = np.linspace(0, 2 * np.pi, 200)

fig = plt.figure(figsize=(16, 11))
# 3D panel: matplotlib axes carry (X, Z, Y) so that the screen-vertical axis is
# our +Y (up), like multiview_calibration.py:157.  view_init then behaves
# naturally: azim picks the horizontal viewing direction, elev the height.
ax = fig.add_subplot(121, projection="3d")
def M(P):
    return (P[0], P[2], P[1])          # world (X,Y,Z) -> mpl (x,y,z)

for y in (y_floor, y_ceil):
    ax.plot(R * np.cos(th), R * np.sin(th), np.full_like(th, y), color="k", lw=1.2)
for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    ax.plot([R * np.cos(a)] * 2, [R * np.sin(a)] * 2, [y_floor, y_ceil], color="k", lw=.3, alpha=.35)
ax.scatter(gx, np.zeros_like(gx), gy, c="green", s=16, depthshade=False, label="cal plate 6x7")
ax.scatter(0, 0, 0, c="gold", edgecolors="k", s=160, marker="*", depthshade=False,
           label="datum = L corner (0,0,0)")

L = 900.0
for cid, P in nominal.items():
    col = "tab:blue" if cid <= 4 else "tab:gray"
    ax.scatter(*M(P), c=col, s=60, depthshade=False,
               label="nominal cam 1-4" if cid == 1 else ("nominal cam 5-8" if cid == 5 else None))
    ax.quiver(*M(P), *M(-P / np.linalg.norm(P) * L), color=col, lw=1.2, arrow_length_ratio=.25)
    ax.text(*M(P + [0, 170, 0]), str(cid), fontsize=9)
for cid, P in calib.items():
    ax.scatter(*M(P), c="crimson", marker="^", s=70, depthshade=False,
               label="calibrated .ori" if cid == 1 else None)
    ax.quiver(*M(P), *M(-P / np.linalg.norm(P) * L), color="crimson", lw=1.2, arrow_length_ratio=.25)
    ax.text(*M(P - [0, 250, 0]), f"c{cid}'", fontsize=9, color="crimson")
for v, c, n in ((np.eye(3)[0], "r", "+X  left->right"), (np.eye(3)[1], "g", "+Y  up"),
                (np.eye(3)[2], "b", "+Z  object->camera")):
    ax.quiver(0, 0, 0, *M(v * 1400), color=c, lw=2.5, arrow_length_ratio=.12)
    ax.text(*M(v * 1700), n, color=c, fontsize=10, weight="bold")
ax.set(xlabel="X [mm]  left->right", ylabel="Z [mm]  object->camera", zlabel="Y [mm]  up",
       xlim=(R, -R), ylim=(-R, R), zlim=(y_floor, y_ceil))
ax.set_box_aspect((2 * R, 2 * R, H))
# looking from above cams 3,4 (X<0, Z>0) down onto the calibration target
ax.view_init(elev=26, azim=45)
ax.set_title("3D from above cams 3/4, looking at the plate\norigin = coded L-corner dot of frame 00000000")
ax.legend(loc="upper left", fontsize=8)

# --- top view (X-Z), looking down -Y
a1 = fig.add_subplot(222)
a1.plot(R * np.cos(th), R * np.sin(th), "k", lw=1.5)
for cid, P in nominal.items():
    a1.plot(P[0], P[2], "s", c="tab:blue" if cid <= 4 else "tab:gray")
    a1.annotate(str(cid), (P[0], P[2]), textcoords="offset points", xytext=(6, 6), fontsize=8)
for cid, P in calib.items():
    a1.plot(P[0], P[2], "^", c="crimson")
    a1.plot([P[0], 0], [P[2], 0], "--", c="crimson", lw=.7)
a1.plot(0, 0, "k+", ms=12)
a1.set(xlabel="X [mm]  left->right", ylabel="Z [mm]  object->camera",
       title="top view (looking down -Y)", aspect="equal")
a1.grid(alpha=.3)

# --- back view (X-Y), looking along +Z from behind the plate
a2 = fig.add_subplot(224)
a2.axhline(y_floor, color="saddlebrown", lw=3, label="heating plate")
a2.axhline(y_ceil, color="k", ls="--", lw=2, label="ceiling")
a2.add_patch(plt.Rectangle((-plate["nx"] * px / 2, -plate["ny"] * py / 2),
                           plate["nx"] * px, plate["ny"] * py, fc="lightgreen", ec="g", alpha=.6))
for cid, P in nominal.items():
    if cid <= 4:
        a2.plot(P[0], P[1], "s", c="tab:blue")
        a2.annotate(f"cam{cid}", (P[0], P[1]), textcoords="offset points", xytext=(8, 6), fontsize=8)
for cid, P in calib.items():
    a2.plot(P[0], P[1], "^", c="crimson")
a2.plot(0, 0, "k+", ms=12)
a2.set(xlabel="X [mm]  left->right", ylabel="Y [mm]  bottom->top", xlim=(-R, R),
       title="back view (looking along +Z)", aspect="equal")
a2.legend(fontsize=8)
a2.grid(alpha=.3)

fig.suptitle("Illmenau global frame:  +X left->right,  +Y bottom->top,  +Z object->camera\n"
             "blue squares = nominal wall mounts   |   red triangles = calibrated .ori (cal/cam*.tif.ori)")
fig.tight_layout()
fig.savefig(base / "rig_3d_global.png", dpi=120)
print("wrote", base / "rig_3d_global.png")

print("\ncam   nominal (X,Y,Z)          calibrated (X,Y,Z)        azimuth nom/cal [deg]  radius nom/cal")
for cid in (1, 2, 3, 4):
    n, c = nominal[cid], calib[cid]
    def az(p):
        return np.degrees(np.arctan2(p[0], p[2]))
    def rr(p):
        return np.hypot(p[0], p[2])
    print(f" {cid}  {np.round(n,0)}  {np.round(c,0)}   {az(n):6.1f} / {az(c):6.1f}"
          f"     {rr(n):6.0f} / {rr(c):6.0f}")
print("\nbaseline  nominal / calibrated [mm]")
for a, b in ((1, 3), (2, 4), (1, 2), (3, 4)):
    print(f" cam{a}-cam{b}: {np.linalg.norm(nominal[a]-nominal[b]):7.0f} / {np.linalg.norm(calib[a]-calib[b]):7.0f}")
