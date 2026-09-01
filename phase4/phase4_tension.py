"""
Phase 4 — LynTom tissue-tension analysis with mezzo+ pole co-localisation.

For each pescoid (P_ctrl + P_Activin only, 52 in total):
  1. Compute structure tensor per frame on LynTom signal
        - Q nematic (global tissue alignment, 0..1)
        - Mean local coherence (anisotropy field)
        - Cortex (boundary band 15 px) vs interior (eroded mask) coherence
  2. Identify mezzo+ pole regions from Phase 2 tracks at peak frame
  3. Compute coherence INSIDE pole regions vs OUTSIDE (rest of mask)
     -> the "fate vs mechanics" coupling
  4. Save per-frame timeseries CSV + peak-frame heatmap PNG per pescoid
  5. Cross-condition + cross-phenotype comparison plots

Usage:
    python phase4_tension.py --sanity            # G046 + G021 only
    python phase4_tension.py --all               # all 52 P pescoids
"""

import argparse
import re
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
PHASE4 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase4")
PER = PHASE4 / "per_pescoid"
PLOTS = PHASE4 / "plots"
GALLERY = PHASE4 / "heatmap_gallery"
TABLES = PHASE4 / "tables"
for d in (PHASE4, PER, PLOTS, GALLERY, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 6.0
HPF_INTERVAL = 0.4
PEAK_HPF = (11.0, 15.0)

# Structure tensor params
SIGMA_GRAD = 5
SIGMA_TENSOR = 10
CORTEX_THICKNESS_PX = 15
N_RADIAL_BINS = 10
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# ===========================================================================
# Structure-tensor helpers
# ===========================================================================
def norm_pct(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def structure_tensor_fields(img, sigma_grad=SIGMA_GRAD, sigma_tensor=SIGMA_TENSOR):
    ax_y = ndi.gaussian_filter(img, sigma=sigma_grad, order=(1, 0))
    ax_x = ndi.gaussian_filter(img, sigma=sigma_grad, order=(0, 1))
    Jxx = ndi.gaussian_filter(ax_x * ax_x, sigma=sigma_tensor)
    Jyy = ndi.gaussian_filter(ax_y * ax_y, sigma=sigma_tensor)
    Jxy = ndi.gaussian_filter(ax_x * ax_y, sigma=sigma_tensor)
    trace = Jxx + Jyy
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2
    coherence = np.where(lam1 + lam2 > 0, (lam1 - lam2) / (lam1 + lam2 + 1e-10), 0.0)
    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    return orientation, coherence


def nematic_Q(orient, coh, mask):
    """Coherence-weighted nematic order parameter Q in [0, 1]."""
    if not mask.any():
        return 0.0
    a = orient[mask]
    w = coh[mask]
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    w = w / w_sum
    return float(np.hypot(np.sum(w * np.cos(2 * a)), np.sum(w * np.sin(2 * a))))


def cortex_interior_coherence(coh, mask, thickness=CORTEX_THICKNESS_PX):
    interior = morphology.binary_erosion(mask, morphology.disk(thickness))
    cortex = mask & ~interior
    cort = float(coh[cortex].mean()) if cortex.any() else np.nan
    inter = float(coh[interior].mean()) if interior.any() else np.nan
    return cort, inter


def radial_coherence_profile(coh, mask, n_bins=N_RADIAL_BINS):
    if not mask.any():
        return np.full(n_bins, np.nan)
    props = measure.regionprops(mask.astype(int))
    if not props:
        return np.full(n_bins, np.nan)
    p0 = max(props, key=lambda r: r.area)
    cy, cx = p0.centroid
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    dist = np.hypot(yy - cy, xx - cx)
    max_d = dist[mask].max() if mask.any() else 1
    dist_n = dist / max_d
    edges = np.linspace(0, 1, n_bins + 1)
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        ring = mask & (dist_n >= edges[i]) & (dist_n < edges[i + 1])
        if ring.sum() > 5:
            out[i] = float(coh[ring].mean())
    return out


# ===========================================================================
# Loading
# ===========================================================================
def load_aligned(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif")) if (pdir / "bf_aligned.tif").exists() else None
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif")) if (pdir / "gfp_aligned.tif").exists() else None
    lyn = tifffile.imread(str(pdir / "lyntom_aligned.tif")) if (pdir / "lyntom_aligned.tif").exists() else None
    masks = (tifffile.imread(str(pdir / "mask_aligned.tif")) > 0) if (pdir / "mask_aligned.tif").exists() else None
    return bf, gfp, lyn, masks


def load_mezzo_pole_regions(cond, pid):
    """Return (T, Y, X) bool stack: True where mezzo POLE-classified tracks exist."""
    tracks_path = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.tif"
    csv_path = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
    if not (tracks_path.exists() and csv_path.exists()):
        return None
    tracks_stack = tifffile.imread(str(tracks_path))   # uint16, track ids
    # Handle empty CSV (no_induction pescoids may have header-only file)
    try:
        tracks_df = pd.read_csv(str(csv_path))
    except pd.errors.EmptyDataError:
        return np.zeros_like(tracks_stack, dtype=bool)
    if tracks_df.empty or "cluster_type" not in tracks_df.columns:
        return np.zeros_like(tracks_stack, dtype=bool)
    pole_ids = tracks_df[tracks_df["cluster_type"] == "pole"]["track_id"].tolist()
    if not pole_ids:
        return np.zeros_like(tracks_stack, dtype=bool)
    return np.isin(tracks_stack, pole_ids)


def load_phase2_summary():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    return df[df["condition"].isin(CONDITIONS)].copy()


# ===========================================================================
# Per-pescoid tension processing
# ===========================================================================
def process_pescoid(cond, pid, summary_row=None, save_outputs=True):
    bf, gfp, lyn, masks = load_aligned(cond, pid)
    if any(x is None for x in (lyn, masks)):
        return None
    pole_stack = load_mezzo_pole_regions(cond, pid)
    if pole_stack is None:
        pole_stack = np.zeros_like(masks, dtype=bool)

    T = masks.shape[0]
    rows = []
    radial_per_frame = []

    # Per-frame fields cached for the peak-frame heatmap
    peak_t_in_window = None
    if summary_row is not None and not np.isnan(summary_row.get("first_mezzo_pole_emergence_hpf", np.nan)):
        # pick the frame within peak window with the largest mezzo pole area
        # We just walk peak window and pick the frame whose total pole-pixel area is max.
        peak_frames = [t for t in range(T)
                       if PEAK_HPF[0] <= (HPF_START + t * HPF_INTERVAL) <= PEAK_HPF[1]]
        if peak_frames:
            pole_areas = [int(pole_stack[t].sum()) for t in peak_frames]
            if max(pole_areas) > 0:
                peak_t_in_window = peak_frames[int(np.argmax(pole_areas))]
    if peak_t_in_window is None:
        # fall back to mid of peak window
        mid_hpf = (PEAK_HPF[0] + PEAK_HPF[1]) / 2
        peak_t_in_window = int(round((mid_hpf - HPF_START) / HPF_INTERVAL))
        peak_t_in_window = max(0, min(T - 1, peak_t_in_window))

    peak_fields = {}

    for t in range(T):
        mask = masks[t]
        if not mask.any():
            continue
        lyn_n = norm_pct(lyn[t].astype(float))
        orient, coh = structure_tensor_fields(lyn_n)
        Q = nematic_Q(orient, coh, mask)
        cort_c, int_c = cortex_interior_coherence(coh, mask)
        radial = radial_coherence_profile(coh, mask)
        radial_per_frame.append(radial)

        # Inside vs outside mezzo+ pole regions
        pole_mask = pole_stack[t] if t < len(pole_stack) else np.zeros_like(mask, dtype=bool)
        pole_in_mask = pole_mask & mask
        non_pole_in_mask = mask & ~pole_in_mask
        coh_in = float(coh[pole_in_mask].mean()) if pole_in_mask.any() else np.nan
        coh_out = float(coh[non_pole_in_mask].mean()) if non_pole_in_mask.any() else np.nan
        Q_in = nematic_Q(orient, coh, pole_in_mask) if pole_in_mask.any() else np.nan
        Q_out = nematic_Q(orient, coh, non_pole_in_mask) if non_pole_in_mask.any() else np.nan

        rows.append({
            "time": t,
            "hpf": HPF_START + t * HPF_INTERVAL,
            "Q_global": Q,
            "mean_coherence": float(coh[mask].mean()),
            "cortex_coherence": cort_c,
            "interior_coherence": int_c,
            "cortex_minus_interior": (cort_c - int_c) if not (np.isnan(cort_c) or np.isnan(int_c)) else np.nan,
            "coh_inside_mezzo_pole": coh_in,
            "coh_outside_mezzo_pole": coh_out,
            "Q_inside_mezzo_pole": Q_in,
            "Q_outside_mezzo_pole": Q_out,
            "pole_area_px": int(pole_in_mask.sum()),
            "mask_area_px": int(mask.sum()),
        })

        if t == peak_t_in_window:
            peak_fields = {
                "lyn_n": lyn_n, "coh": coh, "orient": orient,
                "mask": mask, "pole_mask": pole_in_mask,
                "Q": Q, "cort": cort_c, "int": int_c,
                "coh_in": coh_in, "coh_out": coh_out,
            }

    df = pd.DataFrame(rows)
    radial = np.array(radial_per_frame) if radial_per_frame else np.zeros((0, N_RADIAL_BINS))

    # Peak-window stats per pescoid
    in_peak = df[(df["hpf"] >= PEAK_HPF[0]) & (df["hpf"] <= PEAK_HPF[1])]
    pescoid_peak = {
        "condition": cond,
        "pescoid": pid,
        "peak_frame": peak_t_in_window,
        "peak_hpf": HPF_START + peak_t_in_window * HPF_INTERVAL,
        "Q_peak_window_mean": float(in_peak["Q_global"].mean()) if not in_peak.empty else np.nan,
        "Q_peak_frame": peak_fields.get("Q", np.nan),
        "mean_coherence_peak_mean": float(in_peak["mean_coherence"].mean()) if not in_peak.empty else np.nan,
        "cortex_coherence_peak_mean": float(in_peak["cortex_coherence"].mean()) if not in_peak.empty else np.nan,
        "interior_coherence_peak_mean": float(in_peak["interior_coherence"].mean()) if not in_peak.empty else np.nan,
        "cortex_minus_interior_peak_mean": float(in_peak["cortex_minus_interior"].mean()) if not in_peak.empty else np.nan,
        "coh_inside_mezzo_pole_peak_mean": float(in_peak["coh_inside_mezzo_pole"].dropna().mean()) if not in_peak["coh_inside_mezzo_pole"].dropna().empty else np.nan,
        "coh_outside_mezzo_pole_peak_mean": float(in_peak["coh_outside_mezzo_pole"].dropna().mean()) if not in_peak["coh_outside_mezzo_pole"].dropna().empty else np.nan,
        "n_frames_with_mezzo_pole_in_peak": int((in_peak["pole_area_px"] > 0).sum()),
    }

    if save_outputs:
        pdir = PER / cond / pid
        pdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(pdir / "tension_timeseries.csv"), index=False)
        np.save(str(pdir / "radial_coherence_per_frame.npy"), radial)
        with open(str(pdir / "tension_peak_summary.json"), "w") as f:
            json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v)
                       for k, v in pescoid_peak.items()}, f, indent=2, default=float)
        if peak_fields:
            save_peak_heatmap(peak_fields, pid, cond, peak_t_in_window,
                              pdir / f"{pid}_tension_heatmap_peak.png")

    return pescoid_peak, df, radial


# ===========================================================================
# Peak-frame 4-panel heatmap
# ===========================================================================
def save_peak_heatmap(pf, pid, cond, t, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5))

    # 1) LynTom + mask + pole contour
    axes[0].imshow(pf["lyn_n"], cmap="hot")
    if pf["mask"].any():
        for c in measure.find_contours(pf["mask"].astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "w-", lw=1.2)
    if pf["pole_mask"].any():
        for c in measure.find_contours(pf["pole_mask"].astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[0].set_title(f"LynTom + mask (white) + mezzo+ pole (lime)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2) Coherence map (masked to pescoid)
    coh_disp = np.where(pf["mask"], pf["coh"], np.nan)
    im = axes[1].imshow(coh_disp, cmap="inferno", vmin=0, vmax=1)
    if pf["pole_mask"].any():
        for c in measure.find_contours(pf["pole_mask"].astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[1].set_title(f"Coherence (mean in mask = {pf['coh'][pf['mask']].mean():.2f})",
                       fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.75, label="coherence")

    # 3) Orientation quiver
    axes[2].imshow(pf["lyn_n"], cmap="gray")
    ys, xs = np.mgrid[0:pf["lyn_n"].shape[0]:18, 0:pf["lyn_n"].shape[1]:18]
    in_mask = pf["mask"][ys, xs]
    ys_in, xs_in = ys[in_mask], xs[in_mask]
    o = pf["orient"][ys_in, xs_in]
    c = pf["coh"][ys_in, xs_in]
    dy = np.sin(o + np.pi / 2) * c * 8
    dx = np.cos(o + np.pi / 2) * c * 8
    axes[2].quiver(xs_in, ys_in, dx, -dy, c, cmap="inferno", clim=(0, 1),
                    scale=180, headwidth=2, headlength=3, pivot="middle", alpha=0.95)
    if pf["mask"].any():
        for cc in measure.find_contours(pf["mask"].astype(float), 0.5):
            axes[2].plot(cc[:, 1], cc[:, 0], "w-", lw=1.2)
    if pf["pole_mask"].any():
        for cc in measure.find_contours(pf["pole_mask"].astype(float), 0.5):
            axes[2].plot(cc[:, 1], cc[:, 0], "lime", lw=2)
    axes[2].set_title(f"Orientation quiver (Q = {pf['Q']:.2f})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4) Inside-vs-outside summary bar
    coh_in_val = pf["coh_in"] if not np.isnan(pf["coh_in"]) else 0
    coh_out_val = pf["coh_out"] if not np.isnan(pf["coh_out"]) else 0
    if pf["pole_mask"].any():
        axes[3].bar([0, 1], [coh_in_val, coh_out_val],
                     color=["lime", "#888"], edgecolor="black", lw=1)
        axes[3].set_xticks([0, 1])
        axes[3].set_xticklabels(["Inside mezzo+ pole", "Outside (rest of mask)"])
        axes[3].set_ylim(0, 1)
        axes[3].set_ylabel("Mean coherence")
        axes[3].set_title(
            f"Coherence inside vs outside mezzo+ pole\n"
            f"in={coh_in_val:.2f}   out={coh_out_val:.2f}",
            fontsize=11, fontweight="bold")
    else:
        axes[3].text(0.5, 0.5, "No mezzo+ pole detected", ha="center", va="center",
                      fontsize=12)
        axes[3].axis("off")
    axes[3].grid(axis="y", alpha=0.3)

    plt.suptitle(f"{cond} / {pid}   —   peak frame t={t}  ({HPF_START + t*HPF_INTERVAL:.1f} hpf)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Cross-pescoid plotting
# ===========================================================================
def jitter(n, w=0.15):
    return RNG.uniform(-w, w, size=n)


def swarm_box(ax, df, ycol, xcol="condition", colors=COND_COLOR, labels=COND_LABEL,
              hline=None, ylim=None):
    cond_order = [c for c in CONDITIONS if c in df[xcol].unique()]
    groups = [df[df[xcol] == c][ycol].dropna().values for c in cond_order]
    bp = ax.boxplot(groups, positions=range(len(cond_order)), widths=0.5,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(alpha=0.35), medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], cond_order):
        patch.set_facecolor(colors[c])
    for i, (c, vals) in enumerate(zip(cond_order, groups)):
        x = i + jitter(len(vals))
        ax.scatter(x, vals, s=55, color=colors[c], edgecolors="black", lw=0.6,
                   alpha=0.85, zorder=3)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([f"{labels[c]}\nn={len(g)}" for c, g in zip(cond_order, groups)])
    if hline is not None:
        ax.axhline(hline, ls="--", color="k", lw=1, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)


def paired_inside_outside(df, ycol_in, ycol_out, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    cond_pos = {"P_ctrl": 0, "P_Activin_3-5hpf": 1}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].dropna(subset=[ycol_in, ycol_out])
        if sub.empty:
            continue
        x_base = cond_pos[cond]
        x_in = x_base - 0.2 + jitter(len(sub), 0.06)
        x_out = x_base + 0.2 + jitter(len(sub), 0.06)
        # paired lines
        for vi, vo, xi, xo in zip(sub[ycol_in].values, sub[ycol_out].values, x_in, x_out):
            ax.plot([xi, xo], [vi, vo], "-", color=COND_COLOR[cond], alpha=0.35, lw=0.8)
        ax.scatter(x_in, sub[ycol_in], s=60, color="lime", edgecolors="black", lw=0.6,
                    label="inside mezzo+ pole" if cond == CONDITIONS[0] else None, zorder=3)
        ax.scatter(x_out, sub[ycol_out], s=60, color="#888", edgecolors="black", lw=0.6,
                    label="outside (rest of mask)" if cond == CONDITIONS[0] else None, zorder=3)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values, n=500):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([np.mean(RNG.choice(values, size=len(values), replace=True)) for _ in range(n)])
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def trajectory_plot(traj_df, ycol, ylabel, title, fname, clip_zero=True, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    for cond in CONDITIONS:
        sub = traj_df[traj_df["condition"] == cond]
        for pid, g in sub.groupby("pescoid"):
            ax.plot(g["hpf"], g[ycol], color=COND_COLOR[cond], lw=0.6, alpha=0.25)
    for cond in CONDITIONS:
        sub = traj_df[traj_df["condition"] == cond]
        if sub.empty:
            continue
        xs, ms, los, his = [], [], [], []
        for x, g in sub.groupby("hpf"):
            m, lo, hi = bootstrap_ci(g[ycol].dropna().values)
            xs.append(x); ms.append(m); los.append(lo); his.append(hi)
        xs = np.array(xs); ms = np.array(ms); los = np.array(los); his = np.array(his)
        if clip_zero:
            ms = np.clip(ms, 0, None); los = np.clip(los, 0, None); his = np.clip(his, 0, None)
        ax.fill_between(xs, los, his, color=COND_COLOR[cond], alpha=0.2)
        ax.plot(xs, ms, color=COND_COLOR[cond], lw=3, label=f"{COND_LABEL[cond]} (mean)")
    ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.10, label="Peak window")
    ax.set_xlabel("Time [hpf]"); ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main run
# ===========================================================================
def run(pescoid_filter=None):
    print("=" * 60)
    print("  Phase 4 — LynTom tissue-tension + mezzo+ co-localisation")
    print("=" * 60)

    summary_p2 = load_phase2_summary()
    if pescoid_filter:
        summary_p2 = summary_p2[summary_p2["pescoid"].isin(pescoid_filter)]
    print(f"\n  Pescoids to process: {len(summary_p2)}")

    pescoid_rows = []
    traj_rows = []

    for _, row in summary_p2.iterrows():
        cond = row["condition"]; pid = row["pescoid"]
        print(f"  {cond}/{pid}: ", end="")
        try:
            result = process_pescoid(cond, pid, summary_row=row.to_dict())
            if result is None:
                print("SKIP")
                continue
            peak_row, traj_df, _ = result
            traj_df["condition"] = cond
            traj_df["pescoid"] = pid
            pescoid_rows.append(peak_row)
            traj_rows.append(traj_df)
            print(
                f"Q_peak={peak_row['Q_peak_window_mean']:.2f}, "
                f"coh_in={peak_row['coh_inside_mezzo_pole_peak_mean']:.2f}, "
                f"coh_out={peak_row['coh_outside_mezzo_pole_peak_mean']:.2f}"
            )
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback; traceback.print_exc()

    if not pescoid_rows:
        print("No pescoids processed.")
        return

    pdf = pd.DataFrame(pescoid_rows)
    pdf.to_csv(str(TABLES / "tension_per_pescoid.csv"), index=False)
    traj_all = pd.concat(traj_rows, ignore_index=True)
    traj_all.to_csv(str(TABLES / "tension_per_frame.csv"), index=False)

    # Merge in phenotype labels
    pdf = pdf.merge(summary_p2[["condition", "pescoid", "phenotype_peak", "phenotype_endstate",
                                  "n_mezzo_poles_peak"]],
                     on=["condition", "pescoid"], how="left")

    # ----- Plots -----
    print("\nGenerating cross-condition plots...")
    fig, ax = plt.subplots(figsize=(7, 6))
    swarm_box(ax, pdf, "Q_peak_window_mean", ylim=(0, 1))
    ax.set_ylabel("Q nematic (peak-window mean)")
    ax.set_title("Global tissue alignment at peak window", fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_global_Q_at_peak_swarm.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  01_global_Q_at_peak_swarm.png")

    # Q by phenotype
    fig, ax = plt.subplots(figsize=(12, 6))
    phenos_present = [p for p in pdf["phenotype_peak"].unique() if pd.notna(p)]
    groups = []
    labels = []
    for cond in CONDITIONS:
        for p in phenos_present:
            sub = pdf[(pdf["condition"] == cond) & (pdf["phenotype_peak"] == p)]
            if sub.empty:
                continue
            groups.append(sub["Q_peak_window_mean"].dropna().values)
            labels.append(f"{COND_LABEL[cond]}\n{p}\nn={len(sub)}")
    if groups:
        positions = range(len(groups))
        bp = ax.boxplot(groups, positions=positions, widths=0.5, patch_artist=True,
                          showfliers=False, boxprops=dict(alpha=0.35),
                          medianprops=dict(color="black", lw=2))
        for i, vals in enumerate(groups):
            x = i + jitter(len(vals))
            ax.scatter(x, vals, s=55, edgecolors="black", lw=0.6, alpha=0.85, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Q nematic (peak-window mean)")
    ax.set_title("Tissue alignment Q stratified by phenotype", fontweight="bold")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_Q_by_phenotype.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  02_Q_by_phenotype.png")

    # Trajectories
    trajectory_plot(traj_all, "Q_global", "Q nematic",
                     "Tissue alignment Q over time", "03_Q_over_time.png", ylim=(0, 1))
    print("  03_Q_over_time.png")
    trajectory_plot(traj_all, "mean_coherence", "Mean coherence",
                     "Mean local coherence over time", "04_coherence_over_time.png", ylim=(0, 1))
    print("  04_coherence_over_time.png")
    trajectory_plot(traj_all, "cortex_minus_interior", "Cortex − Interior coherence",
                     "Cortical anisotropy excess over time", "05_cortex_minus_interior.png",
                     clip_zero=False)
    print("  05_cortex_minus_interior.png")

    # Inside vs outside paired
    paired_inside_outside(pdf, "coh_inside_mezzo_pole_peak_mean",
                            "coh_outside_mezzo_pole_peak_mean",
                            "Mean coherence",
                            "Coherence INSIDE vs OUTSIDE mezzo+ pole (peak window)\n"
                            "Lines connect paired values per pescoid",
                            "06_coherence_inside_vs_outside_paired.png")
    print("  06_coherence_inside_vs_outside_paired.png")

    # Q vs n_mezzo_poles scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond in CONDITIONS:
        sub = pdf[pdf["condition"] == cond]
        ax.scatter(sub["n_mezzo_poles_peak"] + jitter(len(sub), 0.08),
                    sub["Q_peak_window_mean"],
                    s=70, color=COND_COLOR[cond], edgecolors="black", lw=0.6,
                    alpha=0.85, label=f"{COND_LABEL[cond]} (n={len(sub)})")
    ax.set_xlabel("Number of mezzo poles at peak")
    ax.set_ylabel("Q nematic (peak)")
    ax.set_title("Tissue alignment vs pole multiplicity", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "07_Q_vs_n_mezzo_poles.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  07_Q_vs_n_mezzo_poles.png")

    # Summary
    out = pdf.groupby("condition").agg(
        n=("pescoid", "count"),
        mean_Q_peak=("Q_peak_window_mean", "mean"),
        mean_coh_in_pole=("coh_inside_mezzo_pole_peak_mean", lambda x: x.dropna().mean()),
        mean_coh_out_pole=("coh_outside_mezzo_pole_peak_mean", lambda x: x.dropna().mean()),
        mean_cortex_minus_interior_peak=("cortex_minus_interior_peak_mean", "mean"),
        n_with_mezzo_pole_in_peak=("n_frames_with_mezzo_pole_in_peak", lambda x: (x > 0).sum()),
    ).round(3)
    out.to_csv(str(TABLES / "phase4_condition_summary.csv"))
    print("\n=== Phase 4 condition summary ===")
    print(out.to_string())
    print(f"\nAll plots in: {PLOTS}/")
    print(f"Tables in: {TABLES}/")
    print(f"Per-pescoid heatmaps in: {PER}/<cond>/<pid>/<pid>_tension_heatmap_peak.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true", help="G046 + G021 only")
    p.add_argument("--all", action="store_true", help="all 52 P pescoids")
    p.add_argument("--pids", nargs="*", help="manual pid list")
    args = p.parse_args()
    if args.sanity:
        run(pescoid_filter=["G046", "G021"])
    elif args.all:
        run()
    elif args.pids:
        run(pescoid_filter=args.pids)
    else:
        p.print_help()
