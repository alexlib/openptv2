# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "matplotlib>=3.7.0",
#     "scipy>=1.11.0",
#     "pandas>=2.0.0",
# ]
# ///

"""Two-phase link lab: why did a good link lose, or a ghost win?

Synthetic 3-frame scenes with known ground truth. The critical decision
is the t=1 -> t=2 step. For each track we replay Phase 1 (3D gate) and
Phase 2 (2D cost + Hungarian) and classify:

  OK            winner == truth
  OUT_OF_RADIUS truth outside v_max ball around prediction (bad vel / v_max)
  WRONG_WINNER  truth in gate but higher 2D cost than winner (tie / noise /
                leaf_weight / stale appearance)
  STOLEN        truth claimed by another track in same component
  GHOST_WIN     winner is an injected triangulation ghost

Run:  uv run marimo run notebooks/two_phase_link_lab.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    return cKDTree, linear_sum_assignment, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Two-phase link lab

    **Question per link:** was the truth outside the gate, outscored inside
    the gate, stolen by a neighbour, or did a ghost steal it?

    Move the sliders: `separation` controls how close the crossing is,
    `v_max` the Phase-1 gate, `leaf_weight` the Phase-2 trust in pixels,
    `noise` the 2D pixel noise, `ghost` injects one phantom detection
    midway between the two particles at `t=2`.
    """)
    return


@app.cell
def _(mo):
    scenario = mo.ui.dropdown(
        ["x_crossing", "parallel", "head_on_ghost"],
        value="x_crossing",
        label="scenario",
    )
    separation = mo.ui.slider(0.0, 2.0, step=0.1, value=0.2, label="separation d")
    v_max = mo.ui.slider(0.5, 8.0, step=0.1, value=3.0, label="v_max (3D gate)")
    leaf_weight = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="leaf_weight")
    noise = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="2D noise (px)")
    use_velocity = mo.ui.checkbox(value=True, label="use_velocity")
    ghost = mo.ui.checkbox(value=False, label="inject ghost at t=2")
    mo.vstack([
    mo.hstack([scenario, separation, v_max, leaf_weight]),
    mo.hstack([noise, use_velocity, ghost])
    ])
    return ghost, leaf_weight, noise, scenario, separation, use_velocity, v_max


@app.cell
def _(np):
    def build_scene(kind, d, ghost_on, rng):
        """Return (frames, truth_links t1->t2, ghost_row or None).

        frames: 3 x (N,3). Two true particles + optional ghost row at t=2.
        truth: {pred_idx -> true cand row} for the decision step.
        """
        if kind == "parallel":
            f0 = np.array([[0.0, 0, 0], [0.0, d, 0]])
            f1 = np.array([[1.0, 0, 0], [1.0, d, 0]])
            f2 = np.array([[2.0, 0, 0], [2.0, d, 0]])
        elif kind == "head_on_ghost":
            f0 = np.array([[0.0, 0, 0], [4.0, 0, 0]])
            f1 = np.array([[1.0, 0, 0], [3.0, 0, 0]])
            f2 = np.array([[2.0, 0, 0], [2.0, d, 0]])
        else:  # x_crossing
            f0 = np.array([[0.0, 0, 0], [2.0, 0, 0]])
            f1 = np.array([[1.0 - d / 2, 0, 0], [1.0 + d / 2, 0, 0]])
            f2 = np.array([[2.0, 0, 0], [0.0, 0, 0]])
        frames = [f0, f1, f2]
        truth = {0: 0, 1: 1}
        if kind == "x_crossing":
            truth = {0: 1, 1: 0}  # straight-through: row0->row1? see below
            # row order at t=2 is [x=2 (from left), x=0 (from right)]:
            # left particle (f0 row0) -> f2 row0; right -> f2 row1
            truth = {0: 0, 1: 1}
        ghost_row = None
        if ghost_on:
            g = (f2[0] + f2[1]) / 2  # midpoint phantom
            frames[2] = np.vstack([f2, g + np.array([0.0, 0.05, 0.0])])
            ghost_row = 2
        return frames, truth, ghost_row

    def leaves_of(frames, noise_amp, rng):
        out = []
        for P in frames:
            xy = P[:, :2].copy()
            if noise_amp > 0:
                xy += rng.normal(0, noise_amp, xy.shape)
            out.append(xy)
        return out

    return build_scene, leaves_of


@app.cell
def _(
    build_scene,
    cKDTree,
    ghost,
    leaf_weight,
    leaves_of,
    linear_sum_assignment,
    noise,
    np,
    scenario,
    separation,
    use_velocity,
    v_max,
):
    rng = np.random.default_rng(0)
    frames, truth, ghost_row = build_scene(
        scenario.value, separation.value, ghost.value, rng
    )
    leaves = leaves_of(frames, noise.value, rng)

    # --- prediction for the decision step t=1 -> t=2 ---
    pts1, pts2 = frames[1], frames[2]
    lf2 = leaves[2]
    if use_velocity.value:
        vel = frames[1][:2] - frames[0][:2]  # const-vel from history
        pred = pts1[:2] + vel
    else:
        pred = pts1[:2].copy()
    pred_xy = pred[:, :2]  # ident projection: leaves == XY

    # --- Phase 1: gate ---
    tree = cKDTree(pts2)
    neighbours = tree.query_ball_point(pred, r=v_max.value)

    # --- Phase 2: cost matrix over gated edges ---
    n_pred, n_cand = len(pred), len(pts2)
    import numpy as _np

    INF = 1e9
    cost = _np.full((n_pred, n_cand), INF)
    for pi in range(n_pred):
        for ci in neighbours[pi]:
            if leaf_weight.value > 0:
                cost[pi, ci] = _np.linalg.norm(pred_xy[pi] - lf2[ci]) * leaf_weight.value
            else:
                cost[pi, ci] = _np.linalg.norm(pred[pi] - pts2[ci])
    # Hungarian on finite submatrix
    rows = [pi for pi in range(n_pred) if (cost[pi] < INF).any()]
    cols = sorted({ci for pi in rows for ci in neighbours[pi]})
    winner = {}
    if rows and cols:
        sub = cost[_np.ix_(rows, cols)]
        sentinel = _np.nanmax(sub[sub < INF]) * sub.size + 1.0 if (sub < INF).any() else 1.0
        sub_f = _np.where(sub >= INF, sentinel, sub)
        ri, ci_ = linear_sum_assignment(sub_f)
        for a, b in zip(ri, ci_):
            if sub[a, b] < INF:
                winner[rows[a]] = cols[b]

    # --- verdict per track ---
    verdicts = []
    for pi in range(n_pred):
        trow = truth[pi]
        w = winner.get(pi, None)
        is_ghost = w is not None and ghost_row is not None and w == ghost_row
        if w == trow:
            v = "OK"
        elif trow not in neighbours[pi]:
            v = "OUT_OF_RADIUS"
        elif is_ghost:
            v = "GHOST_WIN"
        elif w is None:
            v = "OUT_OF_RADIUS"
        elif list(winner.values()).count(trow) > 0 and w != trow:
            v = "STOLEN"
        else:
            v = "WRONG_WINNER"
        verdicts.append(
            {
                "track": pi,
                "pred": pred[pi].tolist(),
                "truth_row": trow,
                "truth_3d": float(_np.linalg.norm(pred[pi] - pts2[trow])),
                "truth_2d": float(_np.linalg.norm(pred_xy[pi] - lf2[trow])),
                "winner": w,
                "win_cost": float(cost[pi, w]) if w is not None else None,
                "truth_cost": float(cost[pi, trow]),
                "n_cand": len(neighbours[pi]),
                "verdict": v,
            }
        )
    result = {
        "frames": frames,
        "leaves": leaves,
        "pred": pred,
        "neighbours": neighbours,
        "cost": cost,
        "winner": winner,
        "verdicts": verdicts,
        "ghost_row": ghost_row,
        "truth": truth,
    }
    result
    return (result,)


@app.cell
def _(mo, result):
    import pandas as pd

    _df = pd.DataFrame(result["verdicts"])[
        ["track", "n_cand", "truth_3d", "truth_2d", "truth_cost", "winner", "win_cost", "verdict"]
    ]
    mo.hstack(
        [
            mo.md(f"### Verdicts `{', '.join(v for v in _df['verdict'])}`"),
            mo.ui.table(_df, selection=None),
        ]
    )
    return


@app.cell
def _(plt, result, scenario, separation, v_max):
    _frames = result["frames"]
    _pred = result["pred"]
    _f1, _f2 = _frames[1], _frames[2]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(_f1[:, 0], _f1[:, 1], s=80, marker="o", label="t=1")
    ax.scatter(_f2[:, 0], _f2[:, 1], s=80, marker="s", label="t=2 cand")
    for _i, _p in enumerate(_pred):
        ax.scatter([_p[0]], [_p[1]], s=120, marker="x")
        ax.text(_p[0], _p[1] + 0.08, f"pred{_i}")
        _c = plt.Circle((_p[0], _p[1]), v_max.value, fill=False, ls="--")
        ax.add_patch(_c)
    for _pi, _w in result["winner"].items():
        ax.plot([_pred[_pi, 0], _f2[_w, 0]], [_pred[_pi, 1], _f2[_w, 1]], "-", lw=2)
    for _pi, _trow in result["truth"].items():
        ax.plot([_pred[_pi, 0], _f2[_trow, 0]], [_pred[_pi, 1], _f2[_trow, 1]], ":", lw=1)
    if result["ghost_row"] is not None:
        _g = result["ghost_row"]
        ax.text(_f2[_g, 0], _f2[_g, 1] + 0.08, "GHOST")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{scenario.value} d={separation.value} v_max={v_max.value}  (- winner, : truth, x pred, -- gate)")
    fig.tight_layout()
    fig
    return


@app.cell
def _(np, plt, result):
    _C = result["cost"]
    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    im = ax2.imshow(np.where(_C > 1e8, np.nan, _C), aspect="auto")
    ax2.set_xlabel("candidate row at t=2")
    ax2.set_ylabel("track (pred idx)")
    ax2.set_title("Phase-2 cost matrix (gated edges only, nan = outside gate)")
    for (_i, _j), _z in np.ndenumerate(_C):
        if _z < 1e8:
            ax2.text(_j, _i, f"{_z:.2f}", ha="center", va="center", fontsize=9)
    fig2.colorbar(im)
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    **How to read a failure:**

    * `OUT_OF_RADIUS` — prediction missed the gate. Fix in Phase 1:
      larger `v_max`, better velocity (poisoned history?), or `max_gap`
      bridging. Turn `use_velocity` off to see the bounce return.
    * `WRONG_WINNER` — truth gated but outscored. Fix in Phase 2:
      shrink noise, raise `leaf_weight`, re-project at prediction
      (stale appearance nulls prediction). Cost matrix shows the margin.
    * `STOLEN` — two tracks want one candidate; Hungarian gave it away.
      Needs joint reasoning (shared-observation, confirm pass).
    * `GHOST_WIN` — phantom between the truths wins on cost. Needs
      epipolar / multi-camera veto, not just mean-2D distance.

    Next code step once a mode dominates: add the matching veto/score
    inside `_match_two_phase_frame`, then re-run this lab as regression.
    """)
    return


if __name__ == "__main__":
    app.run()
