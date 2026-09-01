"""
Phase 1 QC — segmentation + registration + centering pipeline.

Usage:
    # Single pescoid (sanity-check first)
    python phase1_qc.py --sample G046

    # Full dataset run after sanity-check approval
    python phase1_qc.py --all

Outputs:
    Lyn_mezzo_phase1/
        per_pescoid/<cond>/<pid>/
            mask_aligned.tif           # T-stack of aligned BF masks (uint8 255)
            bf_aligned.tif             # aligned BF (float32)
            gfp_aligned.tif            # aligned GFP max-Z (float32)
            lyntom_aligned.tif         # aligned LynTom max-Z (float32)
            metrics.json
        qc_images/<cond>_<pid>_qc.png   # 3-panel verification
        phase1_QC.xlsx                  # one row per pescoid
        phase1_QC_summary.csv           # per-condition counts of pass/fail/excluded
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from skimage import measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarking_segment_tools"))

from main_analysis import rotate_centered, compute_orientation
from run_3dunet import UNet

# ===========================================================================
# Paths / constants
# ===========================================================================
DATA_ROOT = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF"
)
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PER = OUT / "per_pescoid"
QC = OUT / "qc_images"
for d in (OUT, PER, QC):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["E_ctrl", "E_Activin_3-5hpf", "P_ctrl", "P_Activin_3-5hpf"]

# QC thresholds (finalised with user)
QC_SEG_FAILED_MAX = 3
QC_SEG_MEDIAN_AREA_MIN = 8000
QC_SEG_AREA_CV_MAX = 0.40
QC_ALIGN_MAX_ROT_DEG = 30.0
QC_ALIGN_ORIENT_STD_DEG = 20.0
QC_ALIGN_CENTROID_DRIFT_PX = 5.0
QC_DISINTEGRATION_AREA_DROP = 0.50  # mask area < 50% of median area => disintegrated

UNET_PATH = Path(
    r"C:\Users\kattimani\Project\pescoid-image-analysis"
    r"\annotation_workspace\model\unet_pescoid_v2.pth"
)
if not UNET_PATH.exists():
    UNET_PATH = Path(
        r"C:\Users\kattimani\Project\pescoid-image-analysis"
        r"\benchmarking_segment_tools\output\3dunet\model\unet_pescoid.pth"
    )


# ===========================================================================
# Helpers
# ===========================================================================
def norm_pct(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def z_project_best_focus(z_stack):
    """Brenner gradient: pick the sharpest single slice."""
    if z_stack.ndim == 2:
        return z_stack.astype(float)
    Z = z_stack.shape[0]
    scores = []
    for z in range(Z):
        img = z_stack[z].astype(float)
        dx = img[:, 2:] - img[:, :-2]
        dy = img[2:, :] - img[:-2, :]
        scores.append(np.mean(dx**2) + np.mean(dy**2))
    return z_stack[int(np.argmax(scores))].astype(float)


def z_project_max(z_stack):
    if z_stack.ndim == 2:
        return z_stack.astype(float)
    return np.max(z_stack, axis=0).astype(float)


def load_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    unet.load_state_dict(torch.load(str(UNET_PATH), map_location=device, weights_only=True))
    unet.eval()
    return unet, device


def segment_unet(unet, device, bf_norm):
    with torch.no_grad():
        t_img = torch.from_numpy(bf_norm.astype(np.float32)[np.newaxis, np.newaxis]).to(device)
        pred = torch.sigmoid(unet(t_img)).cpu().numpy()[0, 0]
    mask = pred > 0.5
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1))
    return mask.astype(bool)


# ===========================================================================
# Core: process one pescoid (load → Z-project → segment → align → QC)
# ===========================================================================
def process_pescoid(sample_path, cond, unet, device):
    pid_match = re.search(r"(G\d+)", sample_path.stem)
    pid = pid_match.group(1) if pid_match else sample_path.stem

    print(f"  [{cond} / {pid}] loading {sample_path.name}")
    stack = tifffile.imread(str(sample_path))
    T, Z, C, Y, X = stack.shape

    # Z-projection per channel per timepoint
    bf_raw = np.zeros((T, Y, X), dtype=np.float64)
    gfp_raw = np.zeros((T, Y, X), dtype=np.float64)
    lyn_raw = np.zeros((T, Y, X), dtype=np.float64)
    for t in range(T):
        bf_raw[t] = z_project_best_focus(stack[t, :, 2])    # BF
        gfp_raw[t] = z_project_max(stack[t, :, 0])          # GFP max-Z
        lyn_raw[t] = z_project_max(stack[t, :, 1])          # LynTom max-Z

    # Segment
    bf_norm_stack = np.array([norm_pct(bf_raw[t]) for t in range(T)])
    masks = np.zeros((T, Y, X), dtype=bool)
    failed_frames = []
    for t in range(T):
        m = segment_unet(unet, device, bf_norm_stack[t])
        if m.sum() == 0:
            failed_frames.append(t)
        masks[t] = m

    # Per-frame centroids + aspect ratios (UNALIGNED — used to pick reference frame)
    centroid_unaligned = []
    aspect_ratios_unaligned = []
    orient_unaligned = []
    for t in range(T):
        if masks[t].any():
            props = measure.regionprops(masks[t].astype(int))
            p = max(props, key=lambda r: r.area)
            centroid_unaligned.append(p.centroid)
            mn = p.minor_axis_length or 1e-6
            aspect_ratios_unaligned.append(p.major_axis_length / mn)
            orient_unaligned.append(np.degrees(compute_orientation(masks[t])))
        else:
            centroid_unaligned.append((np.nan, np.nan))
            aspect_ratios_unaligned.append(np.nan)
            orient_unaligned.append(np.nan)

    # === OPTION B: single stable reference orientation from peak-AR frame ===
    # 1) pick the frame with the highest aspect ratio (the most clearly elongated)
    valid_ar = [(t, ar) for t, ar in enumerate(aspect_ratios_unaligned)
                if not np.isnan(ar) and masks[t].any()]
    if not valid_ar:
        ref_t = 0
        ref_angle_rad = 0.0
    else:
        ref_t = max(valid_ar, key=lambda x: x[1])[0]
        ref_angle_rad = compute_orientation(masks[ref_t])
    ref_angle_deg = -np.degrees(ref_angle_rad)
    print(f"  ref frame for orientation: t={ref_t} "
          f"(AR={aspect_ratios_unaligned[ref_t]:.2f}, "
          f"ref_angle={np.degrees(ref_angle_rad):+.1f} deg)")

    # 2) apply the SAME rotation to all frames, plus per-frame translation
    #    that puts each frame's own centroid at the image centre.
    aligned_masks = np.zeros_like(masks)
    aligned_bf = np.zeros_like(bf_norm_stack)
    aligned_gfp = np.zeros_like(gfp_raw)
    aligned_lyn = np.zeros_like(lyn_raw)
    orient_aligned = []
    centroid_aligned = []
    centroid_drift_px = []
    for t in range(T):
        if not masks[t].any():
            orient_aligned.append(np.nan)
            centroid_aligned.append((np.nan, np.nan))
            centroid_drift_px.append(np.nan)
            continue
        cy, cx = centroid_unaligned[t]
        # rotate_centered: rotation around (cy, cx) AND translate centroid to image centre
        # The angle is FIXED — the reference angle, not per-frame
        aligned_masks[t] = rotate_centered(masks[t].astype(np.float64), ref_angle_deg, (cy, cx), order=0) > 0.5
        aligned_bf[t] = rotate_centered(bf_norm_stack[t], ref_angle_deg, (cy, cx), order=1)
        aligned_gfp[t] = rotate_centered(gfp_raw[t], ref_angle_deg, (cy, cx), order=1)
        aligned_lyn[t] = rotate_centered(lyn_raw[t], ref_angle_deg, (cy, cx), order=1)

        # Centroid + orientation AFTER alignment (for QC)
        if aligned_masks[t].any():
            props = measure.regionprops(aligned_masks[t].astype(int))
            p = max(props, key=lambda r: r.area)
            cy_a, cx_a = p.centroid
            centroid_aligned.append((cy_a, cx_a))
            centroid_drift_px.append(float(np.hypot(cy_a - Y / 2, cx_a - X / 2)))
            orient_aligned.append(np.degrees(compute_orientation(aligned_masks[t])))
        else:
            orient_aligned.append(np.nan)
            centroid_aligned.append((np.nan, np.nan))
            centroid_drift_px.append(np.nan)

    # ===== QC METRICS =====
    areas = np.array([m.sum() for m in aligned_masks])
    valid_areas = areas[areas > 0]
    median_area = float(np.median(valid_areas)) if valid_areas.size else 0.0
    area_cv = float(valid_areas.std() / valid_areas.mean()) if valid_areas.mean() > 0 else 0.0

    # disintegration: first frame after the peak where area falls below 50% of median
    disintegration_frame = -1
    if median_area > 0:
        for t in range(T):
            if areas[t] > 0 and areas[t] < QC_DISINTEGRATION_AREA_DROP * median_area:
                # but ignore early growth frames (must be after frame 5)
                if t >= 5:
                    disintegration_frame = t
                    break

    # Aspect ratio (recorded only, not pass/fail)
    aspect_ratios = []
    for t in range(T):
        if not aligned_masks[t].any():
            aspect_ratios.append(np.nan)
            continue
        props = measure.regionprops(aligned_masks[t].astype(int))
        p = max(props, key=lambda r: r.area)
        mn = p.minor_axis_length or 1e-6
        aspect_ratios.append(p.major_axis_length / mn)
    aspect_ratio_max = float(np.nanmax(aspect_ratios)) if any(~np.isnan(aspect_ratios)) else 0.0

    # Alignment QC — Option B: only score frames where major axis is well-defined.
    # Skip round frames (AR < 1.3) because their measured orientation is noise.
    AR_MIN_FOR_ORIENT = 1.3
    aligned_ars = []
    for t in range(T):
        if not aligned_masks[t].any():
            aligned_ars.append(np.nan)
            continue
        props = measure.regionprops(aligned_masks[t].astype(int))
        p = max(props, key=lambda r: r.area)
        mn = p.minor_axis_length or 1e-6
        aligned_ars.append(p.major_axis_length / mn)

    well_def = [
        ((orient_aligned[t] + 90) % 180) - 90
        for t in range(T)
        if not np.isnan(orient_aligned[t]) and not np.isnan(aligned_ars[t])
        and aligned_ars[t] >= AR_MIN_FOR_ORIENT
    ]
    well_def_arr = np.array(well_def)
    n_orient_frames = int(len(well_def_arr))
    if n_orient_frames > 0:
        align_max_rot = float(np.max(np.abs(well_def_arr)))
        align_orient_std = float(np.std(well_def_arr))
    else:
        align_max_rot = 0.0
        align_orient_std = 0.0

    align_centroid_drift = (
        float(np.nanmean(centroid_drift_px)) if centroid_drift_px else 0.0
    )

    # Pass/fail
    seg_ok = len(failed_frames) <= QC_SEG_FAILED_MAX
    area_ok = median_area >= QC_SEG_MEDIAN_AREA_MIN
    cv_ok = area_cv <= QC_SEG_AREA_CV_MAX
    # PASS/FAIL gate (Option B finalised):
    #   - max_rot and orient_std are RECORDED ONLY, not pass/fail criteria,
    #     because with Option B applied they measure shape dynamics (multipolar
    #     pescoids changing axis as poles grow), not registration quality.
    #   - Real alignment quality is captured by centroid_drift.
    drift_ok = align_centroid_drift <= QC_ALIGN_CENTROID_DRIFT_PX

    qc_pass = all([seg_ok, area_ok, cv_ok, drift_ok])

    exclude_reason = []
    if not seg_ok: exclude_reason.append(f"seg_failed_frames={len(failed_frames)}>{QC_SEG_FAILED_MAX}")
    if not area_ok: exclude_reason.append(f"median_area={median_area:.0f}<{QC_SEG_MEDIAN_AREA_MIN}")
    if not cv_ok: exclude_reason.append(f"area_cv={area_cv:.2f}>{QC_SEG_AREA_CV_MAX}")
    if not drift_ok: exclude_reason.append(f"centroid_drift={align_centroid_drift:.1f}>{QC_ALIGN_CENTROID_DRIFT_PX}")

    metrics = {
        "condition": cond,
        "pescoid": pid,
        "sample_file": sample_path.name,
        "n_frames": T,
        "seg_failed_frames": len(failed_frames),
        "seg_median_area_px": median_area,
        "seg_area_cv": area_cv,
        "align_max_rotation_deg": align_max_rot,
        "align_orientation_std_deg": align_orient_std,
        "align_n_orient_frames": n_orient_frames,
        "align_centroid_drift_px": align_centroid_drift,
        "aspect_ratio_max": aspect_ratio_max,         # recorded only
        "disintegration_frame": disintegration_frame, # recorded only
        "qc_pass": qc_pass,
        "exclude_reason": "; ".join(exclude_reason) if exclude_reason else "",
        "notes": "",
    }

    # Save per-pescoid outputs
    pdir = PER / cond / pid
    pdir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(pdir / "mask_aligned.tif"), (aligned_masks.astype(np.uint8) * 255))
    tifffile.imwrite(str(pdir / "bf_aligned.tif"), aligned_bf.astype(np.float32))
    tifffile.imwrite(str(pdir / "gfp_aligned.tif"), aligned_gfp.astype(np.float32))
    tifffile.imwrite(str(pdir / "lyntom_aligned.tif"), aligned_lyn.astype(np.float32))
    with open(str(pdir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # All-timepoints montage in the same folder
    save_aligned_montage(aligned_bf, aligned_masks, cond, pid,
                         pdir / f"{pid}_all_timepoints_aligned.png")

    return metrics, aligned_bf, aligned_masks


# ===========================================================================
# QC visualisation — 3-panel verification image
# ===========================================================================
def save_qc_image(cond, pid, bf_raw_t0, mask_raw_t0, aligned_bf, aligned_masks, metrics):
    T = aligned_bf.shape[0]
    t_mid = T // 2
    t_end = T - 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    # Panel 1: raw t=0 with mask contour + centroid
    axes[0].imshow(bf_raw_t0, cmap="gray")
    if mask_raw_t0.any():
        for c in measure.find_contours(mask_raw_t0.astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "r-", lw=1.5)
        props = measure.regionprops(mask_raw_t0.astype(int))
        if props:
            p = max(props, key=lambda r: r.area)
            cy0, cx0 = p.centroid
            axes[0].plot(cx0, cy0, "y+", markersize=18, mew=2.5)
    axes[0].axhline(bf_raw_t0.shape[0] / 2, color="cyan", ls="--", lw=0.8, alpha=0.5)
    axes[0].axvline(bf_raw_t0.shape[1] / 2, color="cyan", ls="--", lw=0.8, alpha=0.5)
    axes[0].set_title(f"Raw t=0 (unaligned)\ncentroid = yellow +", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: aligned t=mid with mask + centroid + major axis
    _draw_aligned_panel(axes[1], aligned_bf[t_mid], aligned_masks[t_mid], f"Aligned t={t_mid}")

    # Panel 3: aligned t=end with mask + centroid + major axis
    _draw_aligned_panel(axes[2], aligned_bf[t_end], aligned_masks[t_end], f"Aligned t={t_end}")

    pass_str = "PASS" if metrics["qc_pass"] else "FAIL"
    color = "green" if metrics["qc_pass"] else "red"
    plt.suptitle(
        f"{cond} / {pid}   —   QC {pass_str}   "
        f"(seg_failed={metrics['seg_failed_frames']}, "
        f"median_area={metrics['seg_median_area_px']:.0f}px, "
        f"area_cv={metrics['seg_area_cv']:.2f}, "
        f"orient_std={metrics['align_orientation_std_deg']:.1f}°, "
        f"centroid_drift={metrics['align_centroid_drift_px']:.1f}px)",
        fontsize=13, fontweight="bold", color=color,
    )
    if metrics["exclude_reason"]:
        plt.figtext(0.5, 0.02, f"Exclude reason: {metrics['exclude_reason']}",
                    ha="center", fontsize=10, color="red")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = QC / f"{cond}_{pid}_qc.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_aligned_montage(aligned_bf, aligned_masks, cond, pid, out_path,
                          n_cols=6):
    """Single image with ALL aligned timepoints (BF + mask contour + centroid)."""
    T = aligned_bf.shape[0]
    n_rows = int(np.ceil(T / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    H, W = aligned_bf[0].shape
    for t in range(T):
        ax = axes[t]
        ax.imshow(aligned_bf[t], cmap="gray")
        if aligned_masks[t].any():
            for c in measure.find_contours(aligned_masks[t].astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], "r-", lw=1.0)
            props = measure.regionprops(aligned_masks[t].astype(int))
            p = max(props, key=lambda r: r.area)
            cy, cx = p.centroid
            ax.plot(cx, cy, "y+", markersize=10, mew=1.8)
        # Image-centre crosshair (faint)
        ax.axhline(H / 2, color="cyan", ls="--", lw=0.4, alpha=0.4)
        ax.axvline(W / 2, color="cyan", ls="--", lw=0.4, alpha=0.4)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
    for t in range(T, len(axes)):
        axes[t].axis("off")
    plt.suptitle(f"{cond} / {pid} — all {T} timepoints (aligned, centred)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(out_path), dpi=90, bbox_inches="tight")
    plt.close(fig)


def _draw_aligned_panel(ax, bf, mask, title):
    ax.imshow(bf, cmap="gray")
    H, W = bf.shape
    # Image-center crosshair
    ax.axhline(H / 2, color="cyan", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(W / 2, color="cyan", ls="--", lw=0.8, alpha=0.6)
    if mask.any():
        for c in measure.find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "r-", lw=1.5)
        props = measure.regionprops(mask.astype(int))
        p = max(props, key=lambda r: r.area)
        cy, cx = p.centroid
        # Centroid
        ax.plot(cx, cy, "y+", markersize=18, mew=2.5)
        # Major axis line
        a = p.orientation
        L = p.major_axis_length / 2
        dx = L * np.cos(a)
        dy = -L * np.sin(a)  # image y goes down
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "lime", lw=2)
        ax.set_title(
            f"{title}  centroid=({cx:.0f}, {cy:.0f})  axis_angle={np.degrees(a):+.1f}°",
            fontsize=10, fontweight="bold",
        )
    else:
        ax.set_title(f"{title} (no mask)", fontsize=10)
    ax.axis("off")


# ===========================================================================
# Main entry points
# ===========================================================================
def find_sample(condition, pid):
    """Return the TIF Path for a (condition, pid) pair."""
    data_dir = DATA_ROOT / condition
    for tif in data_dir.glob("*.tif"):
        if pid in tif.stem:
            return tif
    return None


def find_sample_anywhere(pid):
    """Search all conditions for a pescoid id."""
    for cond in CONDITIONS:
        s = find_sample(cond, pid)
        if s is not None:
            return cond, s
    return None, None


def run_one(pid):
    print(f"=== Phase 1 sanity-check on a single pescoid: {pid} ===\n")
    cond, sample_path = find_sample_anywhere(pid)
    if sample_path is None:
        print(f"ERROR: could not find a sample matching {pid}")
        sys.exit(1)
    print(f"Found: {cond}/{sample_path.name}\n")

    unet, device = load_unet()
    print(f"U-Net loaded on {device}\n")

    # Load original first frame for QC panel 1
    stack = tifffile.imread(str(sample_path))
    bf_t0_raw = norm_pct(z_project_best_focus(stack[0, :, 2]))
    mask_t0 = segment_unet(unet, device, bf_t0_raw)

    metrics, aligned_bf, aligned_masks = process_pescoid(sample_path, cond, unet, device)
    qc_path = save_qc_image(cond, metrics["pescoid"], bf_t0_raw, mask_t0, aligned_bf, aligned_masks, metrics)

    print(f"\nQC image: {qc_path}")
    print(f"Per-pescoid outputs: {PER / cond / metrics['pescoid']}\n")
    print(f"QC pass: {metrics['qc_pass']}")
    if metrics["exclude_reason"]:
        print(f"Exclude reason: {metrics['exclude_reason']}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def run_all():
    print("=== Phase 1 full dataset run ===\n")
    unet, device = load_unet()
    print(f"U-Net loaded on {device}\n")

    all_metrics = []
    for cond in CONDITIONS:
        data_dir = DATA_ROOT / cond
        if not data_dir.exists():
            continue
        for sample_path in sorted(data_dir.glob("*.tif")):
            try:
                stack = tifffile.imread(str(sample_path))
                bf_t0_raw = norm_pct(z_project_best_focus(stack[0, :, 2]))
                mask_t0 = segment_unet(unet, device, bf_t0_raw)
                metrics, aligned_bf, aligned_masks = process_pescoid(sample_path, cond, unet, device)
                save_qc_image(cond, metrics["pescoid"], bf_t0_raw, mask_t0,
                              aligned_bf, aligned_masks, metrics)
                all_metrics.append(metrics)
            except Exception as e:
                warnings.warn(f"Failed on {cond}/{sample_path.name}: {e}")
                all_metrics.append({
                    "condition": cond,
                    "pescoid": re.search(r"(G\d+)", sample_path.stem).group(1)
                              if re.search(r"(G\d+)", sample_path.stem) else sample_path.stem,
                    "sample_file": sample_path.name,
                    "n_frames": 0,
                    "qc_pass": False,
                    "exclude_reason": f"exception: {e}",
                    "notes": "",
                })

    df = pd.DataFrame(all_metrics)
    df.to_csv(str(OUT / "phase1_QC.csv"), index=False)
    try:
        with pd.ExcelWriter(str(OUT / "phase1_QC.xlsx"), engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="all_pescoids", index=False)
            for cond in CONDITIONS:
                sub = df[df["condition"] == cond]
                if not sub.empty:
                    sub.to_excel(writer, sheet_name=cond[:31], index=False)
    except Exception as e:
        print(f"Excel write failed: {e}")

    # Summary
    summary = df.groupby("condition").agg(
        n_total=("pescoid", "count"),
        n_pass=("qc_pass", "sum"),
    ).reset_index()
    summary["n_fail"] = summary["n_total"] - summary["n_pass"]
    summary.to_csv(str(OUT / "phase1_QC_summary.csv"), index=False)

    print("\n=== Phase 1 summary ===")
    print(summary.to_string(index=False))
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", help="Pescoid ID for sanity check (e.g. G046)")
    p.add_argument("--all", action="store_true", help="Run on all pescoids")
    args = p.parse_args()
    if args.all:
        run_all()
    elif args.sample:
        run_one(args.sample)
    else:
        p.print_help()
