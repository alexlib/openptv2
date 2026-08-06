from pathlib import Path

import numpy as np

wp1_res = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\res")


def parse_ptv_is(filepath):
    lines = filepath.read_text().strip().splitlines()
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5:
            prev_id = int(parts[0])
            next_id = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            data.append((prev_id, next_id, x, y, z))
    return data


f1_data = parse_ptv_is(wp1_res / "ptv_is.1")
f2_data = parse_ptv_is(wp1_res / "ptv_is.2")

f1_pos = np.array([[d[2], d[3], d[4]] for d in f1_data])
f1_next = np.array([d[1] for d in f1_data])

f2_pos = np.array([[d[2], d[3], d[4]] for d in f2_data])
f2_prev = np.array([d[0] for d in f2_data])

linked = []
displacements = []

for i, (prev_id, next_id, x, y, z) in enumerate(f1_data):
    if next_id >= 0 and next_id < len(f2_data):
        p1 = np.array([x, y, z])
        p2 = np.array([f2_data[next_id][2], f2_data[next_id][3], f2_data[next_id][4]])
        disp = p2 - p1
        displacements.append(disp)
        linked.append(i)

displacements = np.array(displacements)
print(f"Total frame 1 particles: {len(f1_data)}")
print(
    f"Linked particles: {len(displacements)} ({len(displacements) / len(f1_data) * 100:.1f}%)"
)
if len(displacements) > 0:
    print("Displacement stats (mm):")
    print(
        f"  dx: min={displacements[:, 0].min():.3f}, max={displacements[:, 0].max():.3f}, mean={displacements[:, 0].mean():.3f}, std={displacements[:, 0].std():.3f}"
    )
    print(
        f"  dy: min={displacements[:, 1].min():.3f}, max={displacements[:, 1].max():.3f}, mean={displacements[:, 1].mean():.3f}, std={displacements[:, 1].std():.3f}"
    )
    print(
        f"  dz: min={displacements[:, 2].min():.3f}, max={displacements[:, 2].max():.3f}, mean={displacements[:, 2].mean():.3f}, std={displacements[:, 2].std():.3f}"
    )
    mags = np.linalg.norm(displacements, axis=1)
    print(
        f"  magnitude: min={mags.min():.3f}, max={mags.max():.3f}, mean={mags.mean():.3f}, p95={np.percentile(mags, 95):.3f}"
    )

# For unlinked particles in frame 1, find nearest neighbor in frame 2
unlinked_idx = [i for i in range(len(f1_data)) if f1_next[i] == -1]
print(f"\nUnlinked particles in frame 1: {len(unlinked_idx)}")

if unlinked_idx:
    unlinked_pos = f1_pos[unlinked_idx]
    nn_dists = []
    nn_disps = []
    for p in unlinked_pos:
        diffs = f2_pos - p
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        nn_dists.append(dists[min_idx])
        nn_disps.append(diffs[min_idx])
    nn_dists = np.array(nn_dists)
    nn_disps = np.array(nn_disps)
    print("Nearest neighbor distance in frame 2 for unlinked particles:")
    print(
        f"  min={nn_dists.min():.3f}, max={nn_dists.max():.3f}, median={np.median(nn_dists):.3f}, p90={np.percentile(nn_dists, 90):.3f}"
    )
    print("  Nearest neighbor disps for unlinked:")
    print(f"    dx: min={nn_disps[:, 0].min():.3f}, max={nn_disps[:, 0].max():.3f}")
    print(f"    dy: min={nn_disps[:, 1].min():.3f}, max={nn_disps[:, 1].max():.3f}")
    print(f"    dz: min={nn_disps[:, 2].min():.3f}, max={nn_disps[:, 2].max():.3f}")
