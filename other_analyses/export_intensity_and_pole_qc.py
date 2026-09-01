"""
Two things:
  1. Build a single Excel with EVERY intensity column for every pescoid x frame,
     side-by-side (raw + bg-subtracted + threshold used), so the user can spot
     why E < P contradicts biological expectation.
  2. Generate pole-QC images: for each pescoid that has detected poles, draw
     a 3-panel figure showing:
        - kymograph with pole positions overlaid (red dashed lines)
        - activity profile (time-averaged |delta-r|) with pole peaks marked
        - BF mask at peak-AR timepoint with the pole wedge highlighted
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage import measure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

DATA_ROOT = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF"
)
EXISTING = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_yen_fixed")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_final")
QC = OUT / "pole_QC"
QC.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["E_ctrl", "E_Activin_3-5hpf", "P_ctrl", "P_Activin_3-5hpf"]
HPF_START = 6.0
HPF_INTERVAL = 0.4

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.size": 11,
})


# ============================================================================
# 1. INTENSITY EXCEL EXPORT
# ============================================================================
def collect_raw_intensities():
    """Re-read each TIF and mask to get raw uint16 intensities (no normalization)."""
    rows = []
    for cond in CONDITIONS:
        data_dir = DATA_ROOT / cond
        if not data_dir.exists():
            continue
        for sample_path in sorted(data_dir.glob("*.tif")):
            pid_match = re.search(r"(G\d+)", sample_path.stem)
            pid = pid_match.group(1) if pid_match else sample_path.stem
            mask_path = EXISTING / cond / sample_path.stem / "masks" / f"{sample_path.stem}_masks.tif"
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
                bg_pixels = gfp_raw[~bf_mask]
                in_pixels = gfp_raw[bf_mask]
                bg_mean = float(bg_pixels.mean())
                bg_std = float(bg_pixels.std())

                rows.append({
                    "condition": cond,
                    "pescoid": pid,
                    "time": t,
                    "hpf": HPF_START + t * HPF_INTERVAL,
                    "bf_area_px": int(bf_mask.sum()),

                    # Raw (uint16, no normalization)
                    "gfp_raw_mean_in_mask": float(in_pixels.mean()),
                    "gfp_raw_median_in_mask": float(np.median(in_pixels)),
                    "gfp_raw_max_in_mask": float(in_pixels.max()),
                    "gfp_raw_p99_in_mask": float(np.percentile(in_pixels, 99)),
                    "gfp_raw_std_in_mask": float(in_pixels.std()),

                    # Background
                    "bg_mean_outside_mask": bg_mean,
                    "bg_std_outside_mask": bg_std,

                    # Background-subtracted
                    "gfp_mean_bgsub": float(in_pixels.mean() - bg_mean),
                    "gfp_p99_bgsub": float(np.percentile(in_pixels, 99) - bg_mean),

                    # S/B ratio
                    "sb_ratio_mean": float(in_pixels.mean() / bg_mean) if bg_mean > 0 else 0,
                    "sb_ratio_p99": float(np.percentile(in_pixels, 99) / bg_mean) if bg_mean > 0 else 0,

                    # Integrated GFP (sum of bg-sub intensity in mask)
                    "integrated_gfp_bgsub": float((in_pixels - bg_mean).clip(min=0).sum()),
                })
            print(f"  {cond}/{pid}: {T} frames")
    return pd.DataFrame(rows)


def load_pipeline_csvs():
    """Merge in the pipeline's mezzo_expression_over_time.csv (which has normalized values)."""
    rows = []
    for cond in CONDITIONS:
        cond_dir = EXISTING / cond
        if not cond_dir.exists():
            continue
        for sample_dir in sorted(cond_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pid_match = re.search(r"(G\d+)", sample_dir.name)
            pid = pid_match.group(1) if pid_match else sample_dir.name
            mp = sample_dir / "mezzo_expression_over_time.csv"
            if not mp.exists():
                continue
            df = pd.read_csv(str(mp))
            df["condition"] = cond
            df["pescoid"] = pid
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_intensity_excel():
    print("\n[1/2] Building intensity Excel...")
    raw = collect_raw_intensities()
    pipeline = load_pipeline_csvs()

    # Rename pipeline columns to mark them as 'normalized' (post-percentile)
    pipeline_renamed = pipeline.rename(columns={
        "gfp_fraction": "pipeline_gfp_fraction",
        "gfp_mean_intensity": "pipeline_gfp_mean_normed",
        "gfp_pos_mean_intensity": "pipeline_gfp_pos_mean_normed",
        "gfp_threshold": "pipeline_gfp_threshold_normed",
        "gfp_positive_area": "pipeline_gfp_positive_area",
    })
    keep = [c for c in [
        "condition", "pescoid", "time",
        "pipeline_gfp_fraction",
        "pipeline_gfp_positive_area",
        "pipeline_gfp_mean_normed",
        "pipeline_gfp_pos_mean_normed",
        "pipeline_gfp_threshold_normed",
    ] if c in pipeline_renamed.columns]
    pipeline_subset = pipeline_renamed[keep]

    merged = raw.merge(pipeline_subset, on=["condition", "pescoid", "time"], how="left")

    excel_out = OUT / "intensity_values_all_pescoids.xlsx"
    csv_out = OUT / "intensity_values_all_pescoids.csv"
    summary_out = OUT / "intensity_summary_by_condition.csv"

    merged.to_csv(str(csv_out), index=False)

    # Build per-condition summary
    summary = merged.groupby("condition").agg(
        n_samples=("pescoid", "nunique"),
        n_frames=("time", "count"),
        mean_bg=("bg_mean_outside_mask", "mean"),
        mean_raw=("gfp_raw_mean_in_mask", "mean"),
        mean_bgsub=("gfp_mean_bgsub", "mean"),
        mean_sb=("sb_ratio_mean", "mean"),
        mean_integrated_gfp=("integrated_gfp_bgsub", "mean"),
        mean_pipeline_fraction=("pipeline_gfp_fraction", "mean"),
    ).round(2)
    summary.to_csv(str(summary_out))

    try:
        with pd.ExcelWriter(str(excel_out), engine="xlsxwriter") as writer:
            merged.to_excel(writer, sheet_name="all_per_frame", index=False)
            summary.to_excel(writer, sheet_name="summary_by_condition")
            # Per-condition sheets too (capped at 1M rows ofc)
            for cond in CONDITIONS:
                sub = merged[merged["condition"] == cond]
                sub.to_excel(writer, sheet_name=cond[:31], index=False)
        print(f"  Saved Excel: {excel_out}")
    except Exception as e:
        print(f"  Excel save failed ({e}); CSV still saved.")

    print(f"  Saved CSV:   {csv_out}")
    print(f"  Saved summary: {summary_out}")

    print("\nSummary by condition:")
    print(summary.to_string())
    return merged


# ============================================================================
# 2. POLE QC IMAGES — show exactly which peaks are counted
# ============================================================================
def make_pole_qc(cond, sample_dir, sample_path):
    """For one pescoid: render kymograph + activity profile + BF mask with pole wedges."""
    pid_match = re.search(r"(G\d+)", sample_dir.name)
    pid = pid_match.group(1) if pid_match else sample_dir.name

    kymo_npy = sample_dir / "kymograph_matrix.npy"
    mask_tif = sample_dir / "masks" / f"{sample_dir.name}_masks.tif"
    if not kymo_npy.exists() or not mask_tif.exists():
        return None

    kymo_raw = np.load(str(kymo_npy))
    masks = tifffile.imread(str(mask_tif)) > 0

    # Compute activity profile + peaks (mirroring main_analysis settings)
    activity = np.mean(np.abs(kymo_raw), axis=1)
    activity_smooth = gaussian_filter1d(activity, sigma=3, mode="wrap")
    P, T = kymo_raw.shape
    peaks, _ = find_peaks(activity_smooth, prominence=0.12, distance=25)

    if len(peaks) == 0:
        return None  # nothing to show

    # Classify outward vs inward
    mean_signed = np.mean(kymo_raw, axis=1)
    outward = peaks[mean_signed[peaks] > 0]
    inward = peaks[mean_signed[peaks] < 0]

    # Pick peak-AR frame
    ars = []
    for t in range(T):
        if t >= len(masks):
            ars.append(1.0)
            continue
        props = measure.regionprops(masks[t].astype(int))
        if props:
            p = max(props, key=lambda r: r.area)
            mn = p.minor_axis_length or 1e-6
            ars.append(p.major_axis_length / mn)
        else:
            ars.append(1.0)
    peak_t = int(np.argmax(ars))
    mask = masks[peak_t]
    props = measure.regionprops(mask.astype(int))
    if not props:
        return None
    p0 = max(props, key=lambda r: r.area)
    cy, cx = p0.centroid

    # Map kymograph perimeter bin -> angle in degrees (image coords, atan2 convention)
    def bin_to_deg(i):
        return np.degrees(-np.pi + 2 * np.pi * (i / P))

    # FWHM around each peak -> wedge angles
    wedge_info = []
    for pi in peaks:
        half = activity_smooth[pi] / 2
        l = pi
        for _ in range(P):
            lp = (l - 1) % P
            if activity_smooth[lp] < half:
                break
            l = lp
        r = pi
        for _ in range(P):
            rn = (r + 1) % P
            if activity_smooth[rn] < half:
                break
            r = rn
        wedge_info.append((pi, bin_to_deg(l), bin_to_deg(r),
                           mean_signed[pi] > 0))

    # === Plot 3 panels ===
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A: kymograph with pole columns marked
    vmax = max(0.001, np.abs(kymo_raw).max())
    axes[0].imshow(kymo_raw, aspect="auto", cmap="seismic",
                   vmin=-vmax, vmax=vmax, origin="lower")
    for pi in peaks:
        is_out = mean_signed[pi] > 0
        color = "lime" if is_out else "cyan"
        axes[0].axhline(pi, color=color, lw=1.2, alpha=0.9, ls="--")
        axes[0].text(0.5, pi + 1.5, f"P@{pi}", color=color, fontsize=9,
                     fontweight="bold")
    axes[0].set_title(f"Kymograph (perimeter bins x time)\n"
                      f"Detected poles marked (green=outward, cyan=inward)",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time [frame]")
    axes[0].set_ylabel("Perimeter bin")

    # Panel B: activity profile with peaks
    axes[1].plot(activity_smooth, "k-", lw=1.8, label="Smoothed activity ⟨|Δr|⟩ₜ")
    axes[1].plot(activity, "0.6", lw=0.8, label="Raw activity")
    if len(outward) > 0:
        axes[1].scatter(outward, activity_smooth[outward], s=140, marker="^",
                        c="red", edgecolors="black", zorder=4,
                        label=f"Outward poles (n={len(outward)})")
        for pi in outward:
            axes[1].annotate(f"P@{pi}", (pi, activity_smooth[pi]),
                             xytext=(5, 8), textcoords="offset points",
                             fontsize=9, color="red", fontweight="bold")
    if len(inward) > 0:
        axes[1].scatter(inward, activity_smooth[inward], s=140, marker="v",
                        c="blue", edgecolors="black", zorder=4,
                        label=f"Inward poles (n={len(inward)})")
        for pi in inward:
            axes[1].annotate(f"P@{pi}", (pi, activity_smooth[pi]),
                             xytext=(5, -14), textcoords="offset points",
                             fontsize=9, color="blue", fontweight="bold")
    axes[1].set_xlabel("Perimeter bin")
    axes[1].set_ylabel("Time-averaged |Δr|")
    axes[1].set_title(f"Activity profile — total {len(peaks)} peaks\n"
                      f"(prominence ≥ 0.12, min distance 25 bins)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel C: BF mask at peak-AR with pole wedges
    axes[2].imshow(mask, cmap="Greys_r")
    # Mark centroid
    axes[2].plot(cx, cy, "y+", markersize=14, mew=2.5)
    # Draw wedges
    R = max(mask.shape) * 0.6
    for (pi, a_lo, a_hi, is_out) in wedge_info:
        color = "lime" if is_out else "cyan"
        if a_hi < a_lo:
            a_hi += 360
        wedge = Wedge((cx, cy), R, a_lo, a_hi, alpha=0.25,
                      color=color, edgecolor=color, linewidth=2)
        axes[2].add_patch(wedge)
        # Label position at midpoint of wedge
        mid = np.radians((a_lo + a_hi) / 2)
        lx = cx + R * 0.7 * np.cos(mid)
        ly = cy + R * 0.7 * np.sin(mid)
        axes[2].text(lx, ly, f"P@{pi}", color=color, fontsize=11,
                     fontweight="bold", ha="center",
                     bbox=dict(facecolor="black", alpha=0.5, pad=2))
    axes[2].set_xlim(0, mask.shape[1])
    axes[2].set_ylim(mask.shape[0], 0)
    axes[2].set_title(f"BF mask at peak-AR (t={peak_t})\n"
                      f"Pole wedges = FWHM around each peak",
                      fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(f"{cond} / {pid} — pole-detection QC", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = QC / f"{cond}_{pid}_poleQC.png"
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_pole_qc():
    print("\n[2/2] Generating pole-QC images for every pescoid with detected poles...")
    n = 0
    for cond in CONDITIONS:
        cond_dir = EXISTING / cond
        if not cond_dir.exists():
            continue
        for sample_dir in sorted(cond_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            out = make_pole_qc(cond, sample_dir, sample_dir)
            if out is not None:
                n += 1
                if n % 10 == 0:
                    print(f"  {n} QC images saved...")
    print(f"  Total: {n} pole-QC images in {QC}/")
    return n


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("  Intensity Excel + Pole QC")
    print("=" * 60)
    build_intensity_excel()
    generate_all_pole_qc()
    print(f"\nDone.")
    print(f"  Excel: {OUT}/intensity_values_all_pescoids.xlsx")
    print(f"  Pole QC: {QC}/")


if __name__ == "__main__":
    main()
