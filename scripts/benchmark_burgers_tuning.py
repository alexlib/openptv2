"""Benchmark all trackers on Burgers (5 frames) and synthetic_turbulent_1k (10 frames) – 10-frame window tuning.

Each tracker is swept over dvxmax/dacc/angle and ranked by long+ smooth:
score = mean_len - 0.02*kurtosis .  Prints the best per tracker/dataset.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from openptv2.benchmarking.metrics import compute_physics_metrics
from openptv2.benchmarking.runner import run_tracker
from openptv2.tracking_registry import TRACKER_REGISTRY

ALL_TRACKERS = [t for t in ["priority_segment_3d","4be","full_multipass","standard_forward","two_directional","nearest_hungarian_3d","predictive_gmm_3d","hybrid_deltat_3d","two_phase"] if t in TRACKER_REGISTRY]

def phys(yaml_path, tracker, ov):
    t0=time.perf_counter()
    try:
        pred=run_tracker(yaml_path, tracker, track_overrides=ov)
        dt=time.perf_counter()-t0
        pm=compute_physics_metrics(pred, dt=1.0)
        return {"error":None,"time_s":round(dt,3),"n_tracks":pm.n_tracks,"mean_len":round(pm.mean_track_length,2),"frac10":round(pm.frac_tracks_over_10,3),"kurt": round(float(pm.acceleration_kurtosis),1) if np.isfinite(pm.acceleration_kurtosis) else None, "pred":pred}
    except Exception as e:
        return {"error":str(e)[:500],"time_s":None,"n_tracks":0,"mean_len":0,"frac10":0,"kurt":None,"pred":{}}

def sweep(yaml_path, tracker, base_dvx, base_dacc, base_ang):
    dv_vals=sorted(set([max(0.5,base_dvx*0.4), base_dvx, min(15,base_dvx*2)]))
    dacc_vals=sorted(set([max(0.1,base_dacc*0.5), base_dacc, min(10,base_dacc*3)]))
    ang_vals=sorted(set([max(20,base_ang-40), base_ang]))
    best=None
    rows=[]
    for dv in dv_vals:
        for da in dacc_vals:
            for ang in ang_vals:
                ov={"dvxmax":dv,"dvxmin":-dv,"dvymax":dv,"dvymin":-dv,"dvzmax":dv,"dvzmin":-dv,"dacc":da,"angle":ang}
                r=phys(yaml_path, tracker, ov)
                r.update({"dvxmax":dv,"dacc":da,"angle":ang})
                rows.append(r)
                if r["error"] is None:
                    score=r["mean_len"] - 0.02*(r["kurt"] if r["kurt"] is not None else 20)
                    if best is None or score>best["_score"]:
                        r["_score"]=score
                        best=r
    return best, rows

def run_dataset(label, yaml_path, base_dvx, base_dacc, base_ang):
    print(f"\n=== {label} {yaml_path} ===")
    all_rows=[]
    for tr in ALL_TRACKERS:
        best,_=sweep(yaml_path, tr, base_dvx, base_dacc, base_ang)
        if best is None:
            print(f"{tr:25} FAILED")
            continue
        print(f"{tr:25} dv={best['dvxmax']:4.1f} dacc={best['dacc']:4.1f} ang={best['angle']:3.0f}  mean_len={best['mean_len']:4.1f} frac10={best['frac10']:.2f} kurt={best['kurt']} time={best['time_s']}s  {best['error'] or ''}")
        all_rows.append({"tracker":tr,"dataset":label, **{k:best[k] for k in ("dvxmax","dacc","angle","time_s","n_tracks","mean_len","frac10","kurt")}} )
    return pd.DataFrame(all_rows)

if __name__=="__main__":
    burgers=Path("test_data/burgers/parameters_Run1.yaml").resolve()
    turb=Path("test_data/synthetic_turbulent_1k/parameters_Run1.yaml").resolve()
    print("burgers exists",burgers.exists(), "turb exists", turb.exists())
    df_b=run_dataset("burgers (5 fr, vortex)", burgers, base_dvx=2.0, base_dacc=0.5, base_ang=60)
    print(df_b.to_string(index=False))
    df_t=run_dataset("synthetic_turbulent_1k (10 fr, 1k)", turb, base_dvx=8.0, base_dacc=5.0, base_ang=120)
    print(df_t.to_string(index=False))
    # also save
    out=Path("notebooks/benchmark_results.csv")
    pd.concat([df_b,df_t]).to_csv(out,index=False)
    print(f"saved {out}")
