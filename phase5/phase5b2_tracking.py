"""
Phase 5b-2 - Individual nuclear tracking via btrack.

For each pescoid with viable H2A signal (>=30 nuclei detected at any frame in
the 12-16 hpf window per Phase 5a quality scan), run Bayesian tracking on the
per-frame Cellpose nuclei masks. Output:
  - per-pescoid tracks.csv: ID, t, y, x, parent, root, generation
  - cell-level metrics: track duration, displacement, residence time in
    mezzo+ pole region, recruited-vs-local classification at peak.

Cross-condition outputs (mezzo_H2A_phase5b2/plots/):
  01_track_count_by_phenotype.png
  02_displacement_by_phenotype.png
  03_residence_time_in_pole_by_phenotype.png
  04_recruitment_fraction_by_phenotype.png
  05_trajectory_examples.png      sanity: a few pescoids' tracks overlaid

Usage:
    python phase5b2_tracking.py --pids S21          # one pescoid
    python phase5b2_tracking.py --all              # all viable pescoids
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import btrack
from btrack.utils import segmentation_to_objects

# Set GLPK MIP time limit at the cvxopt module level (btrack's own
# `optimizer_options` field is silently ignored in this version - the
# TrackOptimiser source has a TODO to parse it but doesn't yet).
# Without this, GLPK can run for hours per pescoid trying to close the
# proof-of-optimality gap below 0.1%.
# btrack's `tracker.optimize(**kwargs)` is a proxy that forwards kwargs to
# the TrackOptimiser, which passes them as `options=` to cvxopt.glpk.ilp().
# That's the path that actually controls GLPK; module-level glpk.options is
# ignored by btrack. See `_TRACKER_OPTIMIZE_KW` below for the actual values.
_TRACKER_OPTIMIZE_KW = {
    "mip_gap": 0.01,         # 1% MIP gap tolerance - stop once gap closes
    "tm_lim":  300_000,      # 5 min hard cap (belt-and-suspenders)
    "msg_lev": "GLP_MSG_OFF" # silence the MIP iteration log
}

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
PHASE5A = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5a")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b2")
PER = OUT / "per_pescoid"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PER, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 7.0
HPF_INTERVAL = 698.8316040039062 / 3600.0
PEAK_HPF = (12.0, 16.0)
RNG = np.random.default_rng(42)

PHENO_ORDER = ["coordinated_monopolar", "mezzo_bipolar_only", "multipolar",
               "diffuse_mezzo", "no_induction"]
PHENO_COLOR = {
    "coordinated_monopolar": "#386cb0",
    "mezzo_bipolar_only":    "#7570b3",
    "multipolar":            "#e7298a",
    "diffuse_mezzo":         "#66c2a5",
    "no_induction":          "#999999",
}
MIN_NUCLEI_FOR_TRACKING = 30   # per Phase 5a quality scan, need this many in peak

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Standard btrack 2D-cell-motion config (state = (x,y,z,vx,vy,vz), z held at 0)
# Matrices passed as numpy arrays per the current btrack (>=0.5) Pydantic API.
# ---------------------------------------------------------------------------
def _build_motion_model():
    from btrack.models import MotionModel
    sigma_P = 150.0
    sigma_Q = 50.0
    sigma_R = 5.0
    P_diag = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0]) * sigma_P
    Q_diag = np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]) * sigma_Q
    R_diag = np.eye(3) * sigma_R
    A = np.array([[1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]], dtype=np.float64)
    return MotionModel(
        name="cell_motion",
        measurements=3, states=6,
        A=A, H=H, P=P_diag, Q=Q_diag, R=R_diag,
        dt=1.0, accuracy=7.5, prob_not_assign=0.001, max_lost=5,
    )


def _build_hypothesis_model():
    from btrack.models import HypothesisModel
    return HypothesisModel(
        name="cell_hypothesis",
        hypotheses=["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"],
        lambda_time=5.0, lambda_dist=3.0, lambda_link=10.0, lambda_branch=50.0,
        eta=1e-10, theta_dist=20.0, theta_time=5.0,
        dist_thresh=40, time_thresh=2, apop_thresh=5,
        segmentation_miss_rate=0.1, apoptosis_rate=0.001, relax=True,
    )


def load_pole_stack(cond, pid):
    csv_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
    tif_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.tif"
    if not (csv_p.exists() and tif_p.exists()):
        return None
    stack = tifffile.imread(str(tif_p))
    try:
        df = pd.read_csv(str(csv_p))
    except pd.errors.EmptyDataError:
        return np.zeros_like(stack, dtype=bool)
    if df.empty or "cluster_type" not in df.columns:
        return np.zeros_like(stack, dtype=bool)
    pole_ids = df[df["cluster_type"] == "pole"]["track_id"].tolist()
    if not pole_ids:
        return np.zeros_like(stack, dtype=bool)
    return np.isin(stack, pole_ids)


def viable_pescoids(summary, quality_csv):
    q = pd.read_csv(str(quality_csv))
    keep = q[q["best_nuclei"] >= MIN_NUCLEI_FOR_TRACKING]
    via = summary.merge(keep[["condition", "pescoid", "best_nuclei"]],
                        on=["condition", "pescoid"], how="inner")
    return via


# ---------------------------------------------------------------------------
def track_pescoid(cond, pid):
    """Run btrack on the per-frame Cellpose nuclei masks."""
    nuclei_path = PHASE5A / "per_pescoid" / cond / pid / "nuclei_masks.tif"
    if not nuclei_path.exists():
        return None
    masks = tifffile.imread(str(nuclei_path))  # (T, Y, X)
    if masks.ndim != 3 or masks.max() == 0:
        return None

    objects = segmentation_to_objects(masks, properties=("area",))
    if not objects:
        return None

    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure({
            "motion_model":     _build_motion_model(),
            "hypothesis_model": _build_hypothesis_model(),
            # GLPK MIP time limit per pescoid: 5 min (300_000 ms).
            # The solver hits <0.1% gap fast; the extra time is proof-of-optimality
            # search that does NOT change the biological track assignment.
            "optimizer_options": {"tm_lim": 300_000},
        })
        tracker.append(objects)
        tracker.volume = ((0, masks.shape[2]), (0, masks.shape[1]), (-1e5, 1e5))
        tracker.track()
        tracker.optimize(**_TRACKER_OPTIMIZE_KW)
        tracks = tracker.tracks
    # Flatten to a DataFrame
    rows = []
    for tr in tracks:
        for i, t in enumerate(tr.t):
            rows.append({
                "track_id": int(tr.ID),
                "frame": int(t),
                "y": float(tr.y[i]),
                "x": float(tr.x[i]),
                "parent": int(tr.parent) if tr.parent else 0,
                "root": int(tr.root) if tr.root else int(tr.ID),
                "generation": int(tr.generation) if hasattr(tr, "generation") else 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def per_track_metrics(tdf, mask_stack, pole_stack):
    """Compute per-track metrics: duration, displacement, peak residence in pole,
    pole-presence-at-peak, and whether the track started far from the pole."""
    if tdf.empty:
        return pd.DataFrame()
    out = []
    for tid, g in tdf.groupby("track_id"):
        g = g.sort_values("frame")
        first = g.iloc[0]; last = g.iloc[-1]
        duration = int(last["frame"] - first["frame"] + 1)
        # displacement
        disp = float(np.hypot(last["y"] - first["y"], last["x"] - first["x"]))
        # path length
        dy = np.diff(g["y"].values); dx = np.diff(g["x"].values)
        path_len = float(np.sum(np.hypot(dy, dx)))
        # residence: fraction of frames where pole_stack says the (y,x) is inside a pole
        in_pole_frames = 0
        peak_in_pole_frames = 0
        peak_frames_present = 0
        for _, row in g.iterrows():
            t = int(row["frame"])
            if t >= mask_stack.shape[0]:
                continue
            yi = int(np.clip(round(row["y"]), 0, mask_stack.shape[1] - 1))
            xi = int(np.clip(round(row["x"]), 0, mask_stack.shape[2] - 1))
            if pole_stack is not None and t < pole_stack.shape[0]:
                if pole_stack[t, yi, xi]:
                    in_pole_frames += 1
            hpf = HPF_START + t * HPF_INTERVAL
            if PEAK_HPF[0] <= hpf <= PEAK_HPF[1]:
                peak_frames_present += 1
                if pole_stack is not None and t < pole_stack.shape[0]:
                    if pole_stack[t, yi, xi]:
                        peak_in_pole_frames += 1
        out.append({
            "track_id": int(tid),
            "duration_frames": duration,
            "displacement_px": disp,
            "path_length_px": path_len,
            "tortuosity": (disp / path_len) if path_len > 0 else np.nan,
            "in_pole_frames": in_pole_frames,
            "in_pole_fraction": in_pole_frames / max(1, len(g)),
            "peak_frames_present": peak_frames_present,
            "peak_in_pole_fraction": peak_in_pole_frames / max(1, peak_frames_present),
            "ends_in_pole": (peak_in_pole_frames > 0
                              and peak_frames_present > 0
                              and peak_in_pole_frames / peak_frames_present >= 0.5),
            "first_y": float(first["y"]), "first_x": float(first["x"]),
            "last_y": float(last["y"]), "last_x": float(last["x"]),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def recruitment_classification(tdf, mask_stack, pole_stack):
    """For tracks that end inside a mezzo+ pole at peak: classify as 'recruited'
    if their initial position was far from any pole centroid, else 'born_local'.
    """
    if tdf.empty or pole_stack is None:
        return pd.DataFrame()
    # Find pole centroid at peak frame (max pole area frame in 12-16 hpf)
    T = pole_stack.shape[0]
    peak_window = [t for t in range(T)
                   if PEAK_HPF[0] <= HPF_START + t * HPF_INTERVAL <= PEAK_HPF[1]]
    if not peak_window:
        return pd.DataFrame()
    pole_areas = [int(pole_stack[t].sum()) for t in peak_window]
    if max(pole_areas) == 0:
        return pd.DataFrame()
    t_peak = peak_window[int(np.argmax(pole_areas))]
    # All pole centroids at t_peak
    lab = measure.label(pole_stack[t_peak])
    pole_cents = [(p.centroid, p.area)
                   for p in measure.regionprops(lab)]
    if not pole_cents:
        return pd.DataFrame()
    # Equivalent pole radius = sqrt(area/pi)
    rows = []
    for tid, g in tdf.groupby("track_id"):
        g = g.sort_values("frame")
        # Determine whether the track is at a pole at peak frames
        peak_present = 0
        in_pole_at_peak = 0
        for _, r in g.iterrows():
            t = int(r["frame"])
            hpf = HPF_START + t * HPF_INTERVAL
            if not (PEAK_HPF[0] <= hpf <= PEAK_HPF[1]):
                continue
            peak_present += 1
            yi = int(np.clip(round(r["y"]), 0, pole_stack.shape[1] - 1))
            xi = int(np.clip(round(r["x"]), 0, pole_stack.shape[2] - 1))
            if pole_stack[t, yi, xi]:
                in_pole_at_peak += 1
        if peak_present == 0 or in_pole_at_peak / peak_present < 0.5:
            continue   # not a track that "ends" at a pole
        # Initial position
        first = g.iloc[0]
        # Nearest pole centroid at peak
        d_init = min(np.hypot(first["y"] - pc[0][0], first["x"] - pc[0][1])
                     for pc in pole_cents)
        # Use the area of the nearest pole as scale
        nearest_idx = int(np.argmin([np.hypot(first["y"] - pc[0][0], first["x"] - pc[0][1])
                                      for pc in pole_cents]))
        pole_eq_r = float(np.sqrt(pole_cents[nearest_idx][1] / np.pi))
        recruited = d_init > 2.0 * pole_eq_r
        rows.append({
            "track_id": int(tid),
            "init_dist_to_nearest_pole_px": d_init,
            "pole_eq_r_px": pole_eq_r,
            "init_dist_norm": d_init / pole_eq_r if pole_eq_r > 0 else np.nan,
            "recruited": bool(recruited),
            "born_local": not recruited,
            "first_y": float(first["y"]), "first_x": float(first["x"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def process_pescoid(cond, pid, save_outputs=True):
    print(f"  [{cond}/{pid}]  tracking...", end=" ")
    tdf = track_pescoid(cond, pid)
    if tdf is None or tdf.empty:
        print("SKIP (no tracks)")
        return None, None, None
    print(f"{tdf['track_id'].nunique()} tracks", end=", ")
    # Load mask + pole stack for metrics
    mask = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "mask_aligned.tif")) > 0
    pole = load_pole_stack(cond, pid)
    if pole is None:
        pole = np.zeros_like(mask, dtype=bool)
    tm = per_track_metrics(tdf, mask, pole)
    rc = recruitment_classification(tdf, mask, pole)
    print(f"metrics OK, recruitment-classified {len(rc)} pole-ending tracks")

    if save_outputs:
        pdir = PER / cond / pid
        pdir.mkdir(parents=True, exist_ok=True)
        tdf.to_csv(str(pdir / "tracks.csv"), index=False)
        tm.to_csv(str(pdir / "track_metrics.csv"), index=False)
        rc.to_csv(str(pdir / "recruitment.csv"), index=False)
        h2a = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "h2a_aligned.tif"))
        save_trajectory_plot(cond, pid, tdf, mask, pole, h2a,
                              pdir / f"{pid}_trajectories.png")
    return tdf, tm, rc


def save_trajectory_plot(cond, pid, tdf, mask, pole, h2a, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    # use a representative late frame
    T = h2a.shape[0]
    t_show = min(T - 1, int(round((14.0 - HPF_START) / HPF_INTERVAL)))
    h2a_n = (h2a[t_show] - h2a[t_show].min()) / (np.percentile(h2a[t_show], 99.5) - h2a[t_show].min() + 1e-9)
    ax.imshow(np.clip(h2a_n, 0, 1), cmap="gray")
    if mask[t_show].any():
        for c in measure.find_contours(mask[t_show].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "w-", lw=0.6, alpha=0.5)
    if pole[t_show].any():
        for c in measure.find_contours(pole[t_show].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "lime", lw=2)
    # plot trajectories with frame coloring
    for tid, g in tdf.groupby("track_id"):
        g = g.sort_values("frame")
        if len(g) < 3:
            continue
        ax.plot(g["x"], g["y"], color="cyan", lw=0.5, alpha=0.5)
    ax.set_title(f"{cond} / {pid}  -  trajectories on H2A @ t={t_show}",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(pescoid_filter=None):
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    via = viable_pescoids(summary, PHASE5A / "tables" / "h2a_quality_scan.csv")
    if pescoid_filter:
        via = via[via["pescoid"].isin(pescoid_filter)]
    print(f"Viable pescoids for tracking: {len(via)}")

    all_metrics = []
    all_rc = []
    for _, row in via.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        try:
            tdf, tm, rc = process_pescoid(cond, pid)
            if tm is None:
                continue
            tm["condition"] = cond; tm["pescoid"] = pid
            rc["condition"] = cond; rc["pescoid"] = pid
            all_metrics.append(tm); all_rc.append(rc)
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")
            import traceback; traceback.print_exc()

    if not all_metrics:
        print("No tracking results.")
        return

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    rc_all = pd.concat(all_rc, ignore_index=True) if all_rc else pd.DataFrame()
    metrics_all = metrics_all.merge(summary[["condition", "pescoid", "phenotype_peak"]],
                                     on=["condition", "pescoid"], how="left")
    rc_all = rc_all.merge(summary[["condition", "pescoid", "phenotype_peak"]],
                            on=["condition", "pescoid"], how="left") if not rc_all.empty else rc_all
    metrics_all.to_csv(str(TABLES / "track_metrics_all.csv"), index=False)
    if not rc_all.empty:
        rc_all.to_csv(str(TABLES / "recruitment_all.csv"), index=False)

    # Per-pescoid headline stats
    pp = metrics_all.groupby(["condition", "pescoid", "phenotype_peak"]).agg(
        n_tracks=("track_id", "nunique"),
        median_displacement=("displacement_px", "median"),
        median_path_length=("path_length_px", "median"),
        mean_tortuosity=("tortuosity", "mean"),
        mean_pole_residence=("in_pole_fraction", "mean"),
        n_tracks_ending_in_pole=("ends_in_pole", "sum"),
    ).reset_index()
    if not rc_all.empty:
        rc_pp = rc_all.groupby(["condition", "pescoid"]).agg(
            n_pole_ending=("track_id", "count"),
            n_recruited=("recruited", "sum"),
            n_born_local=("born_local", "sum"),
        ).reset_index()
        rc_pp["recruited_fraction"] = rc_pp["n_recruited"] / rc_pp["n_pole_ending"].clip(lower=1)
        pp = pp.merge(rc_pp, on=["condition", "pescoid"], how="left")
    pp.to_csv(str(TABLES / "phase5b2_per_pescoid_summary.csv"), index=False)

    print(f"\nSaved {len(metrics_all)} tracks across {pp['pescoid'].nunique()} pescoids")
    print("\n=== Phase 5b-2 condition summary ===")
    s = pp.groupby("condition").agg(
        n_pescoids=("pescoid", "count"),
        median_n_tracks=("n_tracks", "median"),
        median_displacement_px=("median_displacement", "median"),
        median_pole_residence=("mean_pole_residence", "median"),
        median_recruited_fraction=("recruited_fraction", "median") if "recruited_fraction" in pp.columns else ("n_tracks", "count"),
    ).round(3)
    print(s.to_string())

    # Plots
    cross_condition_plots(metrics_all, pp, rc_all)
    print(f"\nPlots in: {PLOTS}/   Tables in: {TABLES}/")


def cross_condition_plots(metrics_all, pp, rc_all):
    # 01 track count by phenotype
    stratify_by_phenotype(pp, "n_tracks",
                           "# tracks per pescoid",
                           "Number of nuclear tracks per pescoid (12-16 hpf range)",
                           "01_track_count_by_phenotype.png")
    print("  01_track_count_by_phenotype.png")
    # 02 displacement by phenotype
    stratify_by_phenotype(pp, "median_displacement",
                           "Median displacement [px]",
                           "Median nuclear displacement per pescoid",
                           "02_displacement_by_phenotype.png")
    print("  02_displacement_by_phenotype.png")
    # 03 pole residence by phenotype
    stratify_by_phenotype(pp, "mean_pole_residence",
                           "Mean per-track fraction of time in pole",
                           "Mean fraction of track time spent INSIDE mezzo+ pole",
                           "03_residence_time_in_pole_by_phenotype.png",
                           ylim=(0, 1))
    print("  03_residence_time_in_pole_by_phenotype.png")
    # 04 recruitment fraction by phenotype
    if "recruited_fraction" in pp.columns:
        stratify_by_phenotype(pp, "recruited_fraction",
                               "Recruited / (recruited + born-local)",
                               "Fraction of pole-resident tracks recruited from far away\n"
                               "(initial distance > 2x pole radius from any pole)",
                               "04_recruitment_fraction_by_phenotype.png",
                               ylim=(0, 1), hline=0.5)
        print("  04_recruitment_fraction_by_phenotype.png")


def stratify_by_phenotype(df, ycol, ylabel, title, fname, ylim=None, hline=None):
    fig, ax = plt.subplots(figsize=(13, 6))
    positions = []; labels = []; colors = []; groups = []
    pos = 0
    for cond in CONDITIONS:
        for pheno in PHENO_ORDER:
            sub = df[(df["condition"] == cond) & (df["phenotype_peak"] == pheno)]
            v = sub[ycol].dropna().values
            if v.size == 0:
                continue
            groups.append(v)
            positions.append(pos)
            labels.append(f"{COND_LABEL[cond]}\n{pheno}\nn={len(v)}")
            colors.append(PHENO_COLOR[pheno])
            pos += 1
        pos += 0.8
    if groups:
        bp = ax.boxplot(groups, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(alpha=0.40),
                        medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        for i, (vals, c) in enumerate(zip(groups, colors)):
            x = positions[i] + RNG.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(x, vals, s=45, color=c, edgecolors="black", lw=0.4,
                       alpha=0.85, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if hline is not None:
        ax.axhline(hline, ls="--", color="k", lw=1, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pids", nargs="*", help="pescoid IDs")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.pids:
        run(pescoid_filter=args.pids)
    elif args.all:
        run()
    else:
        p.print_help()
