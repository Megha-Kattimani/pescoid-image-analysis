"""
Phase 5c - WHY and HOW multipolar pescoids form.

Four predictor analyses, asking whether multipolar phenotype is determined
BEFORE the mezzo+ poles appear:

  1. Early convergence-sink count
     - Number of distinct local minima of the divergence field (Phase 5b-1)
       in the 8-10 hpf formation window (pre-pole).
     - If multipolar pescoids show >= 2 sinks early, multipolarity is
       prefigured by tissue flow geometry.

  2. Cell-mass / size predisposition
     - Pescoid area at 8 hpf, total H2A intensity at 8 hpf (cell-mass proxy).
     - Are multipolar pescoids bigger / have more cells from the start?

  3. Per-pole catchment basins (after Phase 5b-2 tracks are available)
     - For each pole in a multipolar pescoid, find tracks that ended there.
     - Map their starting positions: do different poles draw from
       non-overlapping starting territories? (independent recruitment vs
       primary-pole splitting)

  4. Initial nuclear heterogeneity
     - Variance / coefficient of variation of smoothed H2A intensity inside
       the mask at 7 hpf.  Multipolar pescoids may start with stronger
       spatial clustering of nuclei.

Outputs:
  mezzo_H2A_phase5c/tables/
    early_convergence_sinks.csv
    cell_mass_size.csv
    initial_heterogeneity.csv
    catchment_basins.csv     (when 5b-2 done)
    phase5c_predictors_combined.csv
  mezzo_H2A_phase5c/plots/
    01_early_sinks_vs_pole_count.png
    02_cell_mass_by_phenotype.png
    03_initial_heterogeneity_by_phenotype.png
    04_catchment_basins.png   (when 5b-2 done)
    05_predictor_summary.png  (composite)

Usage:
    python phase5c_multipolar_mechanism.py             # runs sections 1,2,4
    python phase5c_multipolar_mechanism.py --basins    # runs section 3 (needs 5b-2)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import morphology, measure, feature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
PHASE5A = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5a")
PHASE5B1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b1")
PHASE5B2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b2_v2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5c_v2")
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 7.0
HPF_INTERVAL = 698.8316040039062 / 3600.0
FORM_HPF = (8.0, 10.0)         # pre-pole formation window for early sinks
INITIAL_HPF = (7.0, 8.0)       # initial pre-mezzo window for heterogeneity

PHENO_ORDER = ["coordinated_monopolar", "mezzo_bipolar_only", "multipolar",
               "diffuse_mezzo", "no_induction"]
PHENO_COLOR = {
    "coordinated_monopolar": "#386cb0",
    "mezzo_bipolar_only":    "#7570b3",
    "multipolar":            "#e7298a",
    "diffuse_mezzo":         "#66c2a5",
    "no_induction":          "#999999",
}

RNG = np.random.default_rng(42)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# 1. Early convergence sinks
# Stricter detector: heavily smoothed divergence + strong threshold +
# connected-component grouping so each "sink" is a spatially extended region.
# ---------------------------------------------------------------------------
DIV_SINK_THRESHOLD = -0.02   # convergent grid points
DIV_SMOOTH_SIGMA_GRID = 1.0  # light smoothing on the grid (~12 px)
MIN_SINK_AREA_GRID = 3       # connected component must span >= this many grid points
MIN_SINK_DISTANCE_PX = 30    # (no longer used; CC handles separation)


def load_div_field(cond, pid):
    f = PHASE5B1 / "per_pescoid" / cond / pid / "div_field.npy"
    if not f.exists():
        return None
    return np.load(str(f))


def count_early_sinks(cond, pid):
    """Count local minima of divergence field in 8-10 hpf, within the mask."""
    div = load_div_field(cond, pid)
    if div is None:
        return None
    mask = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "mask_aligned.tif")) > 0
    # PIV grid coordinates from 5a
    gx = np.load(str(PHASE5A / "per_pescoid" / cond / pid / "piv_grid_x.npy"))
    gy = np.load(str(PHASE5A / "per_pescoid" / cond / pid / "piv_grid_y.npy"))

    # frames in formation window
    T_pairs = div.shape[0]
    frames = [t for t in range(T_pairs)
              if FORM_HPF[0] <= HPF_START + t * HPF_INTERVAL <= FORM_HPF[1]]
    if not frames:
        return None

    # Per-frame: average div over the formation window first (more robust than
    # a single frame), masked by the union of mask points
    avg_div = np.nanmean(div[frames], axis=0)

    # Filter to grid points inside mask in MOST of the window
    in_mask_count = np.zeros_like(avg_div, dtype=int)
    for t in frames:
        if not mask[t].any():
            continue
        yi = np.clip(np.round(gy).astype(int), 0, mask.shape[1] - 1)
        xi = np.clip(np.round(gx).astype(int), 0, mask.shape[2] - 1)
        in_mask_count += mask[t][yi, xi]
    inside = in_mask_count >= max(1, len(frames) // 2)

    # Smooth the time-averaged div field on the grid; outside-mask -> 0 (neutral)
    smoothed = avg_div.copy()
    smoothed[~inside] = 0.0
    smoothed = ndi.gaussian_filter(smoothed, sigma=DIV_SMOOTH_SIGMA_GRID)
    # Connected-component "sink" regions: smoothed div < threshold AND inside mask
    sink_mask = (smoothed < DIV_SINK_THRESHOLD) & inside
    if not sink_mask.any():
        return {"n_sinks": 0, "sink_positions": []}
    lab = measure.label(sink_mask, connectivity=2)
    sinks = []
    for p in measure.regionprops(lab):
        if p.area < MIN_SINK_AREA_GRID:
            continue
        cy_g, cx_g = p.centroid
        cyi = int(round(cy_g)); cxi = int(round(cx_g))
        sinks.append({"grid_y": cyi, "grid_x": cxi,
                      "px_y": float(gy[cyi, cxi]), "px_x": float(gx[cyi, cxi]),
                      "n_grid_points": int(p.area),
                      "min_div_value": float(smoothed[lab == p.label].min())})
    return {"n_sinks": len(sinks), "sink_positions": sinks}


def section_1_early_sinks(summary):
    print("\n=== Section 1: early convergence sinks (8-10 hpf) ===")
    rows = []
    for _, r in summary.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        out = count_early_sinks(cond, pid)
        if out is None:
            continue
        rows.append({
            "condition": cond, "pescoid": pid,
            "phenotype_peak": r["phenotype_peak"],
            "n_mezzo_poles_peak": int(r["n_mezzo_poles_peak"]),
            "n_early_sinks": out["n_sinks"],
            "sink_positions": out["sink_positions"],
        })
    df = pd.DataFrame(rows)
    df.drop(columns=["sink_positions"]).to_csv(
        str(TABLES / "early_convergence_sinks.csv"), index=False)
    print(df[["condition", "pescoid", "phenotype_peak",
              "n_mezzo_poles_peak", "n_early_sinks"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 2. Cell mass / size predisposition
# ---------------------------------------------------------------------------
def section_2_cell_mass(summary):
    print("\n=== Section 2: cell-mass / size predisposition ===")
    rows = []
    early_lo = int(round((INITIAL_HPF[0] - HPF_START) / HPF_INTERVAL))
    early_hi = int(round((FORM_HPF[0] - HPF_START) / HPF_INTERVAL))  # 8 hpf
    for _, r in summary.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        p1 = PHASE1 / "per_pescoid" / cond / pid
        mask = tifffile.imread(str(p1 / "mask_aligned.tif")) > 0
        h2a = tifffile.imread(str(p1 / "h2a_aligned.tif"))
        # area at 7 and 8 hpf
        def area_at(t):
            if t < mask.shape[0] and mask[t].any():
                return int(mask[t].sum())
            return np.nan
        def total_h2a_at(t):
            if t < mask.shape[0] and mask[t].any():
                return float(h2a[t][mask[t]].sum())
            return np.nan
        rows.append({
            "condition": cond, "pescoid": pid,
            "phenotype_peak": r["phenotype_peak"],
            "n_mezzo_poles_peak": int(r["n_mezzo_poles_peak"]),
            "area_at_7hpf":  area_at(early_lo),
            "area_at_8hpf":  area_at(early_hi),
            "total_h2a_at_7hpf": total_h2a_at(early_lo),
            "total_h2a_at_8hpf": total_h2a_at(early_hi),
        })
    df = pd.DataFrame(rows)
    df.to_csv(str(TABLES / "cell_mass_size.csv"), index=False)
    print(df[["condition", "pescoid", "phenotype_peak", "area_at_8hpf",
              "total_h2a_at_8hpf"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 4. Initial nuclear heterogeneity
# ---------------------------------------------------------------------------
HET_SMOOTH_SIGMA = 8.0


def heterogeneity_metric(h2a_frame, mask_frame):
    """CV of smoothed H2A intensity within the mask. Higher CV = more spatial
    clustering of nuclear density."""
    if not mask_frame.any():
        return np.nan, np.nan
    sm = ndi.gaussian_filter(h2a_frame.astype(np.float32), sigma=HET_SMOOTH_SIGMA)
    inside = sm[mask_frame]
    if inside.size == 0 or inside.mean() == 0:
        return np.nan, np.nan
    cv = float(inside.std() / inside.mean())
    var_norm = float(inside.var() / (inside.mean() ** 2 + 1e-9))
    return cv, var_norm


def section_4_heterogeneity(summary):
    print("\n=== Section 4: initial nuclear heterogeneity (7-8 hpf) ===")
    rows = []
    lo = int(round((INITIAL_HPF[0] - HPF_START) / HPF_INTERVAL))
    hi = int(round((INITIAL_HPF[1] - HPF_START) / HPF_INTERVAL))
    for _, r in summary.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        p1 = PHASE1 / "per_pescoid" / cond / pid
        mask = tifffile.imread(str(p1 / "mask_aligned.tif")) > 0
        h2a = tifffile.imread(str(p1 / "h2a_aligned.tif"))
        cvs = []
        for t in range(lo, min(hi + 1, mask.shape[0])):
            cv, _ = heterogeneity_metric(h2a[t], mask[t])
            if not np.isnan(cv):
                cvs.append(cv)
        rows.append({
            "condition": cond, "pescoid": pid,
            "phenotype_peak": r["phenotype_peak"],
            "n_mezzo_poles_peak": int(r["n_mezzo_poles_peak"]),
            "heterogeneity_cv_initial": float(np.mean(cvs)) if cvs else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(str(TABLES / "initial_heterogeneity.csv"), index=False)
    print(df[["condition", "pescoid", "phenotype_peak",
              "heterogeneity_cv_initial"]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 3. Per-pole catchment basins (when 5b-2 tracks available)
# ---------------------------------------------------------------------------
def section_3_basins(summary):
    """For each multipolar/bipolar pescoid with mezzo poles, assign each track
    to the nearest pole AT PEAK FRAME and report the starting-position
    centroid of each pole's catchment basin."""
    print("\n=== Section 3: per-pole catchment basins ===")
    targets = summary[summary["n_mezzo_poles_peak"] >= 2]
    print(f"  pescoids with >=2 mezzo poles to analyse: {len(targets)}")

    rows = []
    for _, r in targets.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        tcsv = PHASE5B2 / "per_pescoid" / cond / pid / "tracks.csv"
        mz_csv = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
        if not (tcsv.exists() and mz_csv.exists()):
            continue
        tdf = pd.read_csv(str(tcsv))
        if tdf.empty:
            continue
        try:
            mz = pd.read_csv(str(mz_csv))
        except pd.errors.EmptyDataError:
            continue
        # Mezzo POLE tracks only - one row per pole, with peak centroid
        poles = mz[mz["cluster_type"] == "pole"][
            ["track_id", "peak_centroid_y", "peak_centroid_x",
             "peak_frame", "peak_area_px"]
        ].reset_index(drop=True)
        if len(poles) < 2:
            continue
        # Use the median peak_frame across the poles as the assignment frame
        t_peak = int(round(poles["peak_frame"].median()))
        # Bound to track frames present
        t_present = sorted(tdf["frame"].unique())
        if not t_present:
            continue
        # Snap to nearest available frame within +/-4 of t_peak
        t_peak = min(t_present, key=lambda x: abs(x - t_peak)
                                              if abs(x - t_peak) <= 4 else 999)
        if abs(t_peak - int(round(poles["peak_frame"].median()))) > 4:
            continue

        # Assign each track present at t_peak to the NEAREST mezzo pole
        pole_pos = poles[["peak_centroid_y", "peak_centroid_x"]].values
        tracks_at_peak = tdf[tdf["frame"] == t_peak][["track_id", "y", "x"]]
        if tracks_at_peak.empty:
            continue
        track_pole = {}
        track_dist_to_assigned_pole = {}
        for _, trow in tracks_at_peak.iterrows():
            dists = np.hypot(pole_pos[:, 0] - trow["y"],
                             pole_pos[:, 1] - trow["x"])
            j = int(np.argmin(dists))
            track_pole[int(trow["track_id"])] = j
            track_dist_to_assigned_pole[int(trow["track_id"])] = float(dists[j])

        poles_with_tracks = set(track_pole.values())
        if len(poles_with_tracks) < 2:
            continue

        # First-frame position per assigned track
        first_pos_by_pole = {}
        for tid, j in track_pole.items():
            g0 = tdf[tdf["track_id"] == tid].sort_values("frame").iloc[0]
            first_pos_by_pole.setdefault(j, []).append((g0["y"], g0["x"]))

        basin_rows = []
        for j in sorted(poles_with_tracks):
            positions = first_pos_by_pole.get(j, [])
            if len(positions) < 2:
                continue
            arr = np.array(positions)
            basin_rows.append({
                "condition": cond, "pescoid": pid,
                "phenotype_peak": r["phenotype_peak"],
                "pole_idx": int(j),
                "pole_area_px": int(poles.loc[j, "peak_area_px"]),
                "pole_centroid_y": float(poles.loc[j, "peak_centroid_y"]),
                "pole_centroid_x": float(poles.loc[j, "peak_centroid_x"]),
                "n_tracks_to_pole": int(len(positions)),
                "basin_centroid_y": float(arr[:, 0].mean()),
                "basin_centroid_x": float(arr[:, 1].mean()),
                "basin_spread_px": float(np.hypot(arr[:, 0].std(), arr[:, 1].std())),
                "basin_to_pole_dist_px": float(np.hypot(
                    arr[:, 0].mean() - poles.loc[j, "peak_centroid_y"],
                    arr[:, 1].mean() - poles.loc[j, "peak_centroid_x"])),
            })

        if len(basin_rows) < 2:
            continue
        # pairwise basin-centroid distances between poles
        cents = np.array([[b["basin_centroid_y"], b["basin_centroid_x"]]
                          for b in basin_rows])
        pole_cents = np.array([[b["pole_centroid_y"], b["pole_centroid_x"]]
                               for b in basin_rows])
        bd, pd_ = [], []
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                bd.append(float(np.hypot(*(cents[i] - cents[j]))))
                pd_.append(float(np.hypot(*(pole_cents[i] - pole_cents[j]))))
        basin_pair_dist = float(np.mean(bd))
        pole_pair_dist = float(np.mean(pd_))
        # higher ratio (basin_pair_dist / pole_pair_dist) means basins are
        # spread out as far apart as the poles -> spatially distinct
        # territories ("independent recruitment").
        # ~0 means basins coincide -> shared origin ("splitting").
        for br in basin_rows:
            br["basin_pair_dist_mean"] = basin_pair_dist
            br["pole_pair_dist_mean"] = pole_pair_dist
            br["basin_to_pole_separation_ratio"] = (basin_pair_dist / pole_pair_dist
                                                    if pole_pair_dist > 0 else np.nan)

        rows.extend(basin_rows)
        print(f"  {cond}/{pid} ({r['phenotype_peak']}): "
              f"{len(basin_rows)} poles, basin pair dist={basin_pair_dist:.0f} px, "
              f"pole pair dist={pole_pair_dist:.0f} px, "
              f"ratio={basin_pair_dist/pole_pair_dist:.2f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(str(TABLES / "catchment_basins.csv"), index=False)
        # Aggregate per-pescoid
        per_p = df.groupby(["condition", "pescoid", "phenotype_peak"]).agg(
            n_poles_basin=("pole_id", "nunique"),
            basin_pair_dist=("basin_pair_dist_mean", "first"),
            pole_pair_dist=("pole_pair_dist_mean", "first"),
            separation_ratio=("basin_to_pole_separation_ratio", "first"),
        ).reset_index()
        per_p.to_csv(str(TABLES / "catchment_basins_per_pescoid.csv"), index=False)
        # Plot 04
        fig, ax = plt.subplots(figsize=(10, 6))
        for pheno in PHENO_ORDER:
            sub = per_p[per_p["phenotype_peak"] == pheno]
            if sub.empty:
                continue
            x = [PHENO_ORDER.index(pheno)] * len(sub)
            ax.scatter([xi + RNG.uniform(-0.1, 0.1) for xi in x],
                       sub["separation_ratio"],
                       s=80, color=PHENO_COLOR[pheno], edgecolors="black", lw=0.6,
                       label=f"{pheno} (n={len(sub)})")
        ax.axhline(1.0, ls="--", color="k", lw=1, alpha=0.5,
                    label="ratio=1 (basins as separated as poles)")
        ax.axhline(0.2, ls=":", color="gray", lw=1, alpha=0.5,
                    label="ratio<0.2 (shared origin, splitting)")
        ax.set_xticks(range(len(PHENO_ORDER)))
        ax.set_xticklabels(PHENO_ORDER, rotation=20)
        ax.set_ylabel("Basin separation / pole separation")
        ax.set_title("Per-pole catchment basins: independent recruitment "
                     "(ratio~1) vs shared origin (ratio~0)",
                     fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(str(PLOTS / "04_catchment_basins.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print("  04_catchment_basins.png")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_01_early_sinks(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        x = sub["n_early_sinks"] + RNG.uniform(-0.15, 0.15, len(sub))
        y = sub["n_mezzo_poles_peak"] + RNG.uniform(-0.15, 0.15, len(sub))
        colors = [PHENO_COLOR.get(p, "#888") for p in sub["phenotype_peak"]]
        ax.scatter(x, y, s=70, c=colors, edgecolors="black", lw=0.7,
                   alpha=0.85, marker="o" if cond == "P_ctrl" else "s",
                   label=COND_LABEL[cond])
    ax.plot([0, 6], [0, 6], "k--", lw=1, alpha=0.4, label="y = x")
    ax.set_xlabel("# early convergence sinks (8-10 hpf, div < -0.02)")
    ax.set_ylabel("# mezzo+ poles at peak (12-16 hpf)")
    ax.set_title("Do early flow-convergence sinks predict pole count?",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_early_sinks_vs_pole_count.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_02_cell_mass(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for axi, ycol, ylab in [(0, "area_at_8hpf", "Pescoid mask area at 8 hpf [px]"),
                             (1, "total_h2a_at_8hpf", "Total H2A intensity at 8 hpf")]:
        _stratify(ax[axi], df, ycol, ylab)
    plt.suptitle("Cell-mass / pescoid size at 8 hpf, by phenotype",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(PLOTS / "02_cell_mass_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_03_heterogeneity(df):
    fig, ax = plt.subplots(figsize=(13, 6))
    _stratify(ax, df, "heterogeneity_cv_initial",
              "CV of smoothed H2A intensity at 7-8 hpf")
    ax.set_title("Initial nuclear heterogeneity by phenotype\n"
                 "(higher CV = more spatial clustering of nuclei early)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "03_initial_heterogeneity_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _stratify(ax, df, ycol, ylabel, ylim=None, hline=None):
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
    if hline is not None:
        ax.axhline(hline, ls="--", color="k", lw=1, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_05_summary(df1, df2, df4):
    """Composite scatter: each predictor vs n_mezzo_poles_peak with phenotype colors."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (df, xcol, xlabel) in zip(axes, [
            (df1, "n_early_sinks", "# early convergence sinks (8-10 hpf)"),
            (df2, "area_at_8hpf",  "Pescoid area at 8 hpf [px]"),
            (df4, "heterogeneity_cv_initial", "Initial nuclear CV (7-8 hpf)"),
    ]):
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            x = sub[xcol]
            y = sub["n_mezzo_poles_peak"]
            colors = [PHENO_COLOR.get(p, "#888") for p in sub["phenotype_peak"]]
            marker = "o" if cond == "P_ctrl" else "s"
            ax.scatter(x, y, s=70, c=colors, edgecolors="black", lw=0.6,
                       alpha=0.85, marker=marker, label=COND_LABEL[cond])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("# mezzo+ poles at peak")
        ax.legend(fontsize=9)
    plt.suptitle("Multipolar predictors vs final pole count\n"
                 "(colors = phenotype, circles = P_ctrl, squares = P_Activin)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(PLOTS / "05_predictor_summary.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(do_basins=False):
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary = summary[summary["condition"].isin(CONDITIONS)].copy()
    print(f"Pescoids: {len(summary)}")

    df1 = section_1_early_sinks(summary)
    df2 = section_2_cell_mass(summary)
    df4 = section_4_heterogeneity(summary)

    print("\nGenerating plots...")
    plot_01_early_sinks(df1)
    print("  01_early_sinks_vs_pole_count.png")
    plot_02_cell_mass(df2)
    print("  02_cell_mass_by_phenotype.png")
    plot_03_heterogeneity(df4)
    print("  03_initial_heterogeneity_by_phenotype.png")
    plot_05_summary(df1, df2, df4)
    print("  05_predictor_summary.png")

    if do_basins:
        df3 = section_3_basins(summary)
        print("  (catchment basins computed; plot deferred)")

    # combined table
    combined = df1[["condition", "pescoid", "phenotype_peak",
                     "n_mezzo_poles_peak", "n_early_sinks"]].merge(
        df2[["condition", "pescoid", "area_at_8hpf", "total_h2a_at_8hpf"]],
        on=["condition", "pescoid"]).merge(
        df4[["condition", "pescoid", "heterogeneity_cv_initial"]],
        on=["condition", "pescoid"])
    combined.to_csv(str(TABLES / "phase5c_predictors_combined.csv"), index=False)

    print("\n=== Headline correlations (Spearman) ===")
    for col in ["n_early_sinks", "area_at_8hpf", "total_h2a_at_8hpf",
                "heterogeneity_cv_initial"]:
        rho = combined[[col, "n_mezzo_poles_peak"]].dropna().corr(method="spearman").iloc[0, 1]
        print(f"  {col:35s}  rho_to_poles = {rho:+.2f}")
    print(f"\nOutputs in: {OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--basins", action="store_true",
                   help="Also run section 3 (per-pole catchment basins; needs 5b-2)")
    args = p.parse_args()
    main(do_basins=args.basins)
