"""
Cleaned-up mezzo analysis: time-trajectory plots only.

HEADLINE
  - S/B ratio over time (4 conditions, dashed S/B=1 line, E-overlap annotation).

MAIN TRAJECTORIES (one per figure)
  - Total integrated GFP over time = sum(bg-subtracted intensity in mask).
  - GFP+ fraction over time using a SINGLE fixed threshold:
        thr = 99th percentile of (bg-subtracted GFP) in P_ctrl frame 0.
        Same threshold applied to all conditions / all frames.
  - Aspect ratio over time.
  - Major axis over time.

SUPPLEMENTARY
  - 4-panel raw GFP (mean / max / S/B / background).
  - GFP+ region intensity over time (bg-subtracted mean inside GFP+ region).
  - Area over time.

All curves: solid line = mean per condition; shaded band = bootstrap 95% CI.
PchipInterpolator for smoothing aggregates only (NOT raw values).
Non-negative quantities clipped >= 0.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import filters as skf, measure, morphology
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF"
)
ANALYSIS_DIR = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_yen_fixed")
OUT_DIR = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_refactor")
FIGS = OUT_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embryos are NOT analyzed — they're QC only. Only pescoids (P_*) are quantified.
CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COLORS = {
    "P_ctrl": "#377eb8",
    "P_Activin_3-5hpf": "#ff7f00",
}
LABELS = {
    "P_ctrl": "P ctrl",
    "P_Activin_3-5hpf": "P Activin 3-5h",
}

HPF_START = 6.0
HPF_INTERVAL = 0.4
PX_UM = 1.29
GFP_BLUR_SIGMA = 3
RNG = np.random.default_rng(42)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F7F7",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "font.size": 11,
    }
)


# ---------------------------------------------------------------------------
# Reset figs/ directory (delete all old plots)
# ---------------------------------------------------------------------------
def reset_figs():
    if FIGS.exists():
        for f in FIGS.iterdir():
            if f.is_file():
                f.unlink()
    FIGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-frame data collection (raw + bg-subtracted + integrated + GFP+ region)
# ---------------------------------------------------------------------------
def collect_per_frame_data(threshold_bgsub=None):
    """
    For every pescoid x frame: load mask + raw GFP, compute:
      - raw mean / max / p99 / std in mask
      - bg mean
      - bg-subtracted mean
      - S/B ratio
      - integrated GFP = sum(bg-subtracted intensity in mask)
      - GFP+ fraction (using fixed threshold_bgsub if provided)
      - Mean GFP intensity inside GFP+ region (bg-subtracted)
      - Aspect ratio + major axis from BF mask
    """
    rows = []

    for cond in CONDITIONS:
        data_dir = DATA_ROOT / cond
        out_dir = ANALYSIS_DIR / cond
        if not data_dir.exists() or not out_dir.exists():
            continue

        for sample_path in sorted(data_dir.glob("*.tif")):
            pid_match = re.search(r"(G\d+)", sample_path.stem)
            pid = pid_match.group(1) if pid_match else sample_path.stem

            mask_path = out_dir / sample_path.stem / "masks" / f"{sample_path.stem}_masks.tif"
            if not mask_path.exists():
                continue

            stack = tifffile.imread(str(sample_path))
            masks = tifffile.imread(str(mask_path)) > 0
            T = min(stack.shape[0], len(masks))

            for t in range(T):
                bf_mask = masks[t]
                if not bf_mask.any():
                    continue

                gfp_raw = np.max(stack[t, :, 0], axis=0).astype(float)
                bg_mean = float(gfp_raw[~bf_mask].mean())

                gfp_in = gfp_raw[bf_mask]
                gfp_bg = np.clip(gfp_raw - bg_mean, 0, None)

                # Integrated GFP = sum of bg-subtracted intensity inside mask
                integrated = float(gfp_bg[bf_mask].sum())

                # GFP+ fraction with fixed threshold (in bg-subtracted units)
                gfp_pos_area = 0
                gfp_pos_mean_bgsub = 0.0
                if threshold_bgsub is not None and threshold_bgsub > 0:
                    gfp_blur = skf.gaussian(gfp_bg, sigma=GFP_BLUR_SIGMA, preserve_range=True)
                    gfp_pos = (gfp_blur > threshold_bgsub) & bf_mask
                    gfp_pos = morphology.binary_opening(gfp_pos, morphology.disk(2))
                    gfp_pos = morphology.remove_small_objects(gfp_pos, min_size=50)
                    gfp_pos_area = int(gfp_pos.sum())
                    if gfp_pos.any():
                        gfp_pos_mean_bgsub = float(gfp_bg[gfp_pos].mean())

                # Aspect ratio + major axis from BF mask
                bf_props = measure.regionprops(bf_mask.astype(int))
                if bf_props:
                    p = bf_props[0]
                    ar = float(p.major_axis_length / p.minor_axis_length) if p.minor_axis_length > 0 else 1.0
                    major_um = float(p.major_axis_length / PX_UM)
                else:
                    ar, major_um = 1.0, 0.0

                bf_area = int(bf_mask.sum())

                rows.append(
                    {
                        "condition": cond,
                        "pescoid": pid,
                        "time": t,
                        "hpf": HPF_START + t * HPF_INTERVAL,
                        "bf_area": bf_area,
                        "bg_mean": bg_mean,
                        "gfp_raw_mean": float(gfp_in.mean()),
                        "gfp_raw_max": float(gfp_in.max()),
                        "gfp_raw_p99": float(np.percentile(gfp_in, 99)),
                        "gfp_mean_bgsub": float(gfp_in.mean() - bg_mean),
                        "gfp_p99_bgsub": float(np.percentile(gfp_in, 99) - bg_mean),
                        "sb_ratio": float(gfp_in.mean() / bg_mean) if bg_mean > 0 else 0.0,
                        "integrated_gfp": integrated,
                        "gfp_pos_area": gfp_pos_area,
                        "gfp_fraction": gfp_pos_area / bf_area,
                        "gfp_pos_mean_bgsub": gfp_pos_mean_bgsub,
                        "aspect_ratio": ar,
                        "major_axis_um": major_um,
                    }
                )

            print(f"  {cond}/{pid}: {T} frames")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoothing / bootstrap helpers
# ---------------------------------------------------------------------------
def pchip_smooth(x, y):
    """Monotonic interpolation, no overshoot."""
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, ys = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(xs) < 4:
        return xs, ys
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) < 4:
        return xs, ys
    interp = PchipInterpolator(xs, ys)
    xs_dense = np.linspace(xs.min(), xs.max(), 100)
    return xs_dense, interp(xs_dense)


def bootstrap_ci(values, n_boot=1000, ci=95):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.array(
        [np.mean(RNG.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(values)), float(lo), float(hi)


def per_timepoint_mean_ci(df, xcol, ycol, clip_zero=False):
    if df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    xs, means, los, his = [], [], [], []
    for x, group in df.groupby(xcol):
        m, lo, hi = bootstrap_ci(group[ycol].values)
        if clip_zero:
            m, lo, hi = max(0, m), max(0, lo), max(0, hi)
        xs.append(x)
        means.append(m)
        los.append(lo)
        his.append(hi)
    return np.array(xs), np.array(means), np.array(los), np.array(his)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_trajectory(
    df,
    ycol,
    ylabel,
    title,
    fname,
    description,
    clip_zero=False,
    hline=None,
    annotate_overlap=False,
    smooth=True,
):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        xs, means, los, his = per_timepoint_mean_ci(sub, "hpf", ycol, clip_zero=clip_zero)
        if len(xs) == 0:
            continue

        # Shaded CI band on raw aggregates
        ax.fill_between(xs, los, his, color=COLORS[cond], alpha=0.15)

        if smooth:
            xs_s, means_s = pchip_smooth(xs, means)
            if clip_zero:
                means_s = np.clip(means_s, 0, None)
            ax.plot(xs_s, means_s, color=COLORS[cond], lw=2.5, label=LABELS[cond])
        else:
            ax.plot(xs, means, color=COLORS[cond], lw=2.5, label=LABELS[cond], marker="o", ms=4)

    if hline is not None:
        ax.axhline(hline, ls="--", color="k", alpha=0.5, lw=1.2)
        xlim = ax.get_xlim()
        ax.text(
            xlim[1] - 0.05 * (xlim[1] - xlim[0]),
            hline,
            f" S/B = {hline}",
            va="bottom",
            ha="right",
            fontsize=9,
            color="k",
        )

    if annotate_overlap:
        ax.annotate(
            "E ctrl and E Activin overlap\n(= embryos insensitive to\nexogenous Activin at this stage)",
            xy=(0.55, 0.5),
            xycoords="axes fraction",
            fontsize=10,
            color="#444",
            bbox=dict(facecolor="white", edgecolor="#aaa", alpha=0.9),
        )

    if clip_zero:
        ax.set_ylim(bottom=0)

    ax.set_xlabel("Time [hpf]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Condition", fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(str(FIGS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}  ::  {description}")


def plot_raw_4panel(df, fname):
    """Supplementary 4-panel raw figure: mean / max / S/B / background."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("gfp_raw_mean", "Mean raw GFP intensity", False),
        ("gfp_raw_max", "Max raw GFP intensity", False),
        ("sb_ratio", "Signal / Background ratio", False),
        ("bg_mean", "Mean background intensity", False),
    ]
    for ax, (col, title, clip) in zip(axes.flat, metrics):
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            xs, means, los, his = per_timepoint_mean_ci(sub, "hpf", col, clip_zero=clip)
            if len(xs) == 0:
                continue
            ax.fill_between(xs, los, his, color=COLORS[cond], alpha=0.15)
            xs_s, means_s = pchip_smooth(xs, means)
            ax.plot(xs_s, means_s, color=COLORS[cond], lw=2, label=LABELS[cond])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time [hpf]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Raw GFP / S/B / background (supplementary)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(FIGS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}  ::  4-panel: raw mean / max / S/B / background (supplementary)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Cleaned-up trajectory analysis")
    print("=" * 60)

    reset_figs()

    # ---- Step 1: collect per-frame data ----
    csv_no_thresh = OUT_DIR / "per_frame_pre_threshold.csv"
    if csv_no_thresh.exists():
        print(f"\nLoading cached: {csv_no_thresh}")
        df_pre = pd.read_csv(str(csv_no_thresh))
    else:
        print("\nCollecting per-frame data (no GFP+ threshold yet)...")
        df_pre = collect_per_frame_data(threshold_bgsub=None)
        df_pre.to_csv(str(csv_no_thresh), index=False)

    # ---- Step 2: derive fixed threshold from P_ctrl frame 0 ----
    # Pescoid medium (L15 + Phenol Red) creates background, so the threshold
    # must clear the phenol-red variance, not just the bg mean.
    # Use the highest of:
    #   - 99th percentile of bg-subtracted P_ctrl frame 0 (capture peak noise)
    #   - 3 * std of bg-subtracted intensity in P_ctrl frame 0 (3-sigma noise floor)
    pctrl0 = df_pre[(df_pre["condition"] == "P_ctrl") & (df_pre["time"] == 0)]
    p99_thresh = float(np.percentile(pctrl0["gfp_p99_bgsub"], 99))
    # 3-sigma over background variance: use std of bg-subtracted means at t=0
    sigma3_thresh = float(3.0 * pctrl0["gfp_mean_bgsub"].std() + pctrl0["gfp_mean_bgsub"].mean())
    fixed_thresh_bgsub = max(p99_thresh, sigma3_thresh)
    if fixed_thresh_bgsub <= 0:
        fixed_thresh_bgsub = float(pctrl0["gfp_p99_bgsub"].max() * 1.5)
    print(f"\nThreshold derivation (P_ctrl frame 0):")
    print(f"  99th pct of bg-subtracted intensity: {p99_thresh:.2f}")
    print(f"  3-sigma noise floor:                 {sigma3_thresh:.2f}")
    print(f"  Final fixed threshold (max of both): {fixed_thresh_bgsub:.2f}")

    # ---- Step 3: re-collect with the fixed threshold to fill GFP+ metrics ----
    csv_full = OUT_DIR / "per_frame_with_threshold.csv"
    if csv_full.exists():
        print(f"\nLoading cached: {csv_full}")
        df = pd.read_csv(str(csv_full))
    else:
        print(f"\nRe-collecting per-frame data with fixed threshold = {fixed_thresh_bgsub:.2f}...")
        df = collect_per_frame_data(threshold_bgsub=fixed_thresh_bgsub)
        df.to_csv(str(csv_full), index=False)

    # ---- Step 4: plots ----
    print("\nGenerating figures:\n")

    # HEADLINE
    plot_trajectory(
        df, "sb_ratio",
        "Signal / Background ratio",
        "GFP Signal-to-Background ratio (HEADLINE)",
        "01_HEADLINE_sb_ratio.png",
        "S/B ratio over time, dashed S/B=1, E-overlap annotated",
        hline=1.0,
        annotate_overlap=True,
    )

    # MAIN TRAJECTORIES
    plot_trajectory(
        df, "integrated_gfp",
        "Total integrated GFP (a.u. x px)",
        "Total integrated GFP over time",
        "02_integrated_gfp.png",
        "Sum of bg-subtracted intensity in mask (NOT area-normalised)",
        clip_zero=True,
    )

    plot_trajectory(
        df, "gfp_fraction",
        f"GFP+ fraction (fixed thr = {fixed_thresh_bgsub:.1f} bg-sub)",
        "GFP+ fraction over time (fixed threshold)",
        "03_gfp_positive_fraction.png",
        f"GFP+ fraction, single fixed threshold ({fixed_thresh_bgsub:.1f}) from P_ctrl frame 0",
        clip_zero=True,
    )

    plot_trajectory(
        df, "aspect_ratio",
        "Aspect ratio",
        "Aspect ratio over time",
        "04_aspect_ratio.png",
        "Aspect ratio of BF mask (major / minor axis)",
        clip_zero=True,
    )

    plot_trajectory(
        df, "major_axis_um",
        "Major axis [um]",
        "Major axis length over time",
        "05_major_axis.png",
        "Major axis length [um] of BF mask",
        clip_zero=True,
    )

    # SUPPLEMENTARY
    plot_raw_4panel(df, "S1_raw_4panel.png")

    plot_trajectory(
        df, "gfp_pos_mean_bgsub",
        "Mean GFP intensity in GFP+ region (bg-sub)",
        "GFP+ region intensity over time",
        "S2_gfp_positive_intensity.png",
        "Mean bg-subtracted intensity inside GFP+ regions only",
        clip_zero=True,
    )

    plot_trajectory(
        df, "bf_area",
        "Pescoid / embryo area [px]",
        "Sample area over time",
        "S3_area.png",
        "BF mask area in pixels (growth)",
        clip_zero=True,
    )

    print(f"\nAll figures saved to: {FIGS}/")
    print(f"All CSVs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
