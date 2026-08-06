import glob
import numpy as np
from pathlib import Path
from openptv2.plugins._assignment import match_within_radius

wp1_res = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\res")

def load_rt_is(filepath):
    lines = filepath.read_text().strip().splitlines()
    if not lines:
        return np.zeros((0, 3))
    pts = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts)

frame_files = sorted(glob.glob(str(wp1_res / "rt_is.*")), key=lambda x: int(Path(x).suffix[1:]))[:10]
frames = [load_rt_is(Path(f)) for f in frame_files]

def simulate_fast3d(frames, max_cands=4, dx=20.0, dy=18.0, dz=23.0, dacc=30.0):
    num_frames = len(frames)
    links_per_step = []
    
    # Simulate frame-by-frame 3-level tracking exactly as track3d_loop_fast does
    path_prev = [np.full(len(f), -1, dtype=int) for f in frames]
    path_next = [np.full(len(f), -1, dtype=int) for f in frames]
    
    for f in range(num_frames - 1):
        f0 = frames[f-1] if f > 0 else np.zeros((0, 3))
        f1 = frames[f]
        f2 = frames[f+1]
        
        p0_prev = path_prev[f-1] if f > 0 else np.zeros(0, dtype=int)
        p1_prev = path_prev[f]
        p1_next = path_next[f]
        p2_prev = path_prev[f+1]
        
        count1 = 0
        orig_parts = len(f1)
        np2 = len(f2)
        
        # Helper: candidate search in 3d box
        def find_cands(pred_pt, box_x, box_y, box_z):
            cand_inds = []
            cand_dists = []
            for k in range(np2):
                ddx = abs(f2[k, 0] - pred_pt[0])
                ddy = abs(f2[k, 1] - pred_pt[1])
                ddz = abs(f2[k, 2] - pred_pt[2])
                if ddx < box_x and ddy < box_y and ddz < box_z:
                    d = np.sqrt(ddx**2 + ddy**2 + ddz**2)
                    cand_inds.append(k)
                    cand_dists.append(d)
            if not cand_inds:
                return [], []
            # Sort top max_cands
            order = np.argsort(cand_dists)[:max_cands]
            return [cand_inds[o] for o in order], [cand_dists[o] for o in order]

        # Level 1: Particles with previous links
        for i in range(orig_parts):
            if p1_prev[i] < 0:
                continue
            prev_idx = p1_prev[i]
            if prev_idx < 0 or prev_idx >= len(f0):
                continue
            pred_pt = 2.0 * f1[i] - f0[prev_idx]
            cand_inds, _ = find_cands(pred_pt, dacc, dacc, dacc)
            
            # Find closest unclaimed candidate
            cand_assigned = False
            for k in cand_inds:
                if p2_prev[k] < 0:
                    p1_next[i] = k
                    p2_prev[k] = i
                    count1 += 1
                    cand_assigned = True
                    break
            if not cand_assigned:
                p1_next[i] = -1

        # Level 2: Neighbor velocity
        for i in range(orig_parts):
            if p1_prev[i] >= 0 or p1_next[i] >= 0:
                continue
            cx, cy, cz = f1[i]
            vels = []
            for j in range(orig_parts):
                if j == i: continue
                if abs(f1[j,0]-cx)<dx and abs(f1[j,1]-cy)<dy and abs(f1[j,2]-cz)<dz and p1_prev[j]>=0:
                    pj = p1_prev[j]
                    vels.append(f1[j] - f0[pj])
            if not vels:
                continue
            avg_vel = np.mean(vels, axis=0)
            pred_pt = f1[i] + avg_vel
            cand_inds, _ = find_cands(pred_pt, dacc, dacc, dacc)
            
            cand_assigned = False
            for k in cand_inds:
                if p2_prev[k] < 0:
                    p1_next[i] = k
                    p2_prev[k] = i
                    count1 += 1
                    cand_assigned = True
                    break
            if not cand_assigned:
                p1_next[i] = -1

        # Level 3: Static prediction
        for i in range(orig_parts):
            if p1_prev[i] >= 0 or p1_next[i] >= 0:
                continue
            pred_pt = f1[i]
            cand_inds, _ = find_cands(pred_pt, dx, dy, dz)
            
            cand_assigned = False
            for k in cand_inds:
                if p2_prev[k] < 0:
                    p1_next[i] = k
                    p2_prev[k] = i
                    count1 += 1
                    cand_assigned = True
                    break
            if not cand_assigned:
                p1_next[i] = -1
                
        links_per_step.append(count1)
        
    return np.mean(links_per_step)

print("=== Simulating MAX_CANDS and Bipartite Matching in fast_3d ===")

for max_c in [4, 8, 16, 32, 64, 128]:
    avg_l = simulate_fast3d(frames, max_cands=max_c)
    print(f"fast_3d (max_cands={max_c:3d}): avg links/step = {avg_l:.1f} / 385.4 ({avg_l/385.4*100:.1f}%)")
