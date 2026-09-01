"""
Phase 1 QC - segmentation + registration + centering pipeline.
mezzo + H2A dataset (260512_mezzo_H2A_3-5hpfActivin, Zeiss 10x).

Adapted from phase1_qc.py:
  - Single input folder of per-scene TIFFs (T,Z,C,Y,X), C = [mezzo-GFP, H2A, BF]
  - Scene number -> condition mapping (contiguous blocks)
  - LynTom channel replaced by H2A nuclear channel
  - Timing: start 7 hpf, interval 698.83 s (~11.65 min)

Usage:
    python phase1_h2a_qc.py --sample S05            # one pescoid sanity-check
    python phase1_h2a_qc.py --scenes 5 15 35        # several scenes
    python phase1_h2a_qc.py --all                   # full dataset

Outputs:
    mezzo_H2A_phase1/
        per_pescoid/<cond>/<pid>/
            mask_aligned.tif      # aligned BF masks (uint8 255)
            bf_aligned.tif        # aligned BF (float32, normalised)
            gfp_aligned.tif       # aligned mezzo-GFP max-Z (float32)
            h2a_aligned.tif       # aligned H2A max-Z (float32)
            metrics.json
            <pid>_all_timepoints_aligned.png
        qc_images/<cond>_<pid>_qc.png
        phase1_QC.xlsx / phase1_QC.csv / phase1_QC_summary.csv
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
from skimage import measure, morphology, exposure, filters
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
    r"Z:\Megha_Kattimani\Imaging\Zeiss"
    r"\260409_mezzo_titration_3-5hpf_obj10x_8hpf\tiff"
)
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_titration_phase1")
PER = OUT / "per_pescoid"
QC = OUT / "qc_images"
for d in (OUT, PER, QC):
    d.mkdir(parents=True, exist_ok=True)

# Activin dose-titration groups (from AVI folder assortment)
CONDITIONS = ["E", "Pctrl", "P10ngml", "P30ngml", "P50ngml"]

# Channel indices in the raw (T,Z,C,Y,X) stack - only 2 channels here
CH_GFP = 0   # mezzo-GFP
CH_BF = 1    # brightfield

# Timing
HPF_START = 8.0
HPF_INTERVAL = 1796.2020263671875 / 3600.0   # ~0.499 hpf (~30 min)

# QC thresholds (carried over from phase1_qc.py)
QC_SEG_FAILED_MAX = 3
QC_SEG_MEDIAN_AREA_MIN = 8000
QC_SEG_AREA_CV_MAX = 0.40
QC_ALIGN_MAX_ROT_DEG = 30.0
QC_ALIGN_ORIENT_STD_DEG = 20.0
QC_ALIGN_CENTROID_DRIFT_PX = 5.0
QC_DISINTEGRATION_AREA_DROP = 0.50   # mask area < 50% of preceding peak => disintegrated
QC_DISINT_MIN_HPF = 10.0             # ignore area drops before this hpf (early growth)
QC_BLANK_PTP_MIN = 50.0              # projected-BF peak-to-peak below this => blank/unacquired frame
QC_LEAK_AREA_FRAC = 0.55             # mask covering > this fraction of the frame => background leak

# Adaptive two-pass segmentation
QC_TREND_WINDOW = 21                 # median-filter window (frames) for the smooth area trend
QC_OUTLIER_LO = 0.60                 # area below this x trend => flag for CLAHE retry
QC_OUTLIER_HI = 1.60                 # area above this x trend => flag for CLAHE retry
QC_CLAHE_CLIP = 0.03                 # CLAHE clip_limit for the contrast-enhanced retry pass
QC_RESCUE_LO = 0.50                  # after retry, area still below this x trend => seg failure
QC_RESCUE_HI = 1.80                  # after retry, area still above this x trend => seg failure

# Fine-tuned on the titration BF (saturated-background); falls back to v2.
UNET_PATH = Path(
    r"C:\Users\kattimani\Project\pescoid-image-analysis"
    r"\annotation_workspace\model\unet_pescoid_titration.pth"
)
if not UNET_PATH.exists():
    UNET_PATH = Path(
        r"C:\Users\kattimani\Project\pescoid-image-analysis"
        r"\annotation_workspace\model\unet_pescoid_v2.pth"
    )


# ===========================================================================
# Scene <-> condition mapping
# ===========================================================================
# Scene -> group mapping from the AVI folder assortment (user-sorted by eye).
# Scenes #2 and #4 are excluded (failed). Two acquisition batches: #1-50 then
# #51-60 (extra P10/P50 replicates).
SCENE_GROUPS = {
    "E":       [5, 6, 7, 8, 9, 10, 11, 12, 13],
    "Pctrl":   [1, 3, 14, 15, 16, 17, 18, 19, 20],
    "P10ngml": [21, 22, 23, 24, 25, 26, 51, 52, 53, 54, 55, 56],
    "P30ngml": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
    "P50ngml": [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 57, 58, 59, 60],
}
_SCENE_TO_GROUP = {n: g for g, scenes in SCENE_GROUPS.items() for n in scenes}

# Scenes excluded by the user: pescoids that fall apart early / not useful.
EXCLUDE_SCENES = {3, 14, 15, 30, 32, 33, 34, 35, 41, 46, 48, 49, 50, 52, 56, 57}


def scene_condition(scene_num):
    """Return the dose group for a scene number, or None if unassigned/excluded."""
    if scene_num in EXCLUDE_SCENES:
        return None
    return _SCENE_TO_GROUP.get(scene_num)


def scene_to_pid(scene_num):
    return f"S{scene_num:02d}"


def list_scenes():
    """Return sorted list of (scene_num, condition, path) for every TIFF found."""
    out = []
    for tif in DATA_ROOT.glob("*.tif"):
        m = re.search(r"#(\d+)", tif.stem)
        if not m:
            continue
        n = int(m.group(1))
        cond = scene_condition(n)
        if cond is None:
            continue
        out.append((n, cond, tif))
    return sorted(out, key=lambda x: x[0])


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


# More inclusive segmentation so bud tips and small peripheral (secondary)
# poles fall INSIDE the mask - Phase 2 detects morph poles from contour
# curvature and mezzo poles from in-mask GFP, so a mask that clips tips would
# miss them. Lower prob threshold + small dilation widens the boundary.
SEG_PROB_THRESH = 0.40
SEG_DILATE_PX = 3


def segment_unet(unet, device, bf_norm):
    with torch.no_grad():
        t_img = torch.from_numpy(bf_norm.astype(np.float32)[np.newaxis, np.newaxis]).to(device)
        pred = torch.sigmoid(unet(t_img)).cpu().numpy()[0, 0]
    mask = pred > SEG_PROB_THRESH
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1))
    if SEG_DILATE_PX > 0:
        mask = morphology.binary_dilation(mask, morphology.disk(SEG_DILATE_PX))
    return mask.astype(bool)


# Classical dark-blob segmentation. The titration BF shows the pescoid as a
# DARK compact object on a bright background; the U-Net (trained on a
# saturated-background BF) inverts and segments the background instead. Otsu on
# the inverted, smoothed BF reliably isolates the dark pescoid.
SEG_SMOOTH_SIGMA = 3
SEG_MIN_OBJECT_PX = 1000
SEG_PERCENTILE_FALLBACK = 82   # keep darkest ~18% of pixels (bright after invert)


def segment_classical(bf_norm, method="otsu"):
    inv = 1.0 - bf_norm
    sm = ndi.gaussian_filter(inv, SEG_SMOOTH_SIGMA)
    if method == "otsu":
        try:
            thr = filters.threshold_otsu(sm)
        except Exception:
            thr = np.percentile(sm, SEG_PERCENTILE_FALLBACK)
    else:
        thr = np.percentile(sm, SEG_PERCENTILE_FALLBACK)
    mask = sm > thr
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_objects(mask, SEG_MIN_OBJECT_PX)
    mask = ndi.binary_fill_holes(mask)
    labeled, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1))
    return mask.astype(bool)


# ===========================================================================
# Core: process one pescoid (load -> Z-project -> segment -> align -> QC)
# ===========================================================================
def process_pescoid(scene_num, cond, sample_path, unet, device):
    pid = scene_to_pid(scene_num)

    print(f"  [{cond} / {pid}] loading {sample_path.name}")
    stack = tifffile.imread(str(sample_path))
    T, Z, C, Y, X = stack.shape

    # Z-projection per channel per timepoint (2 channels: GFP + BF, no H2A)
    bf_raw = np.zeros((T, Y, X), dtype=np.float64)
    gfp_raw = np.zeros((T, Y, X), dtype=np.float64)
    for t in range(T):
        bf_raw[t] = z_project_best_focus(stack[t, :, CH_BF])
        gfp_raw[t] = z_project_max(stack[t, :, CH_GFP])

    # Detect blank / unacquired frames (e.g. trailing empty frame in the CZI export)
    img_area = Y * X
    blank_frames = [t for t in range(T)
                    if (bf_raw[t].max() - bf_raw[t].min()) < QC_BLANK_PTP_MIN]

    def _seg_with_guard(bf_img, method=None):
        """Segment one preprocessed BF image with the fine-tuned U-Net;
        discard background leaks. `method` kept for signature compatibility."""
        m = segment_unet(unet, device, bf_img)
        leaked = m.sum() > QC_LEAK_AREA_FRAC * img_area
        if leaked:
            m = np.zeros_like(m)
        return m, leaked

    # --- Pass 1: norm_pct on every non-blank frame ---
    bf_norm_stack = np.array([norm_pct(bf_raw[t]) for t in range(T)])
    masks = np.zeros((T, Y, X), dtype=bool)
    for t in range(T):
        if t in blank_frames:
            continue
        masks[t], _ = _seg_with_guard(bf_norm_stack[t])
    areas1 = np.array([float(masks[t].sum()) for t in range(T)])

    # --- Smooth temporal area trend (median filter, robust to runs of bad frames) ---
    valid = np.array([t not in blank_frames for t in range(T)])
    pos_valid = areas1[valid & (areas1 > 0)]
    med_valid = float(np.median(pos_valid)) if pos_valid.size else 0.0
    a_for_trend = areas1.copy()
    a_for_trend[~valid] = med_valid           # neutral fill so blanks don't drag the trend
    trend = ndi.median_filter(a_for_trend, size=QC_TREND_WINDOW, mode="nearest")

    # --- Flag frames whose area departs from the trend ---
    flagged = []
    for t in range(T):
        if t in blank_frames:
            continue
        tr = trend[t]
        if tr <= 0:
            continue
        if areas1[t] < QC_OUTLIER_LO * tr or areas1[t] > QC_OUTLIER_HI * tr:
            flagged.append(t)

    # --- Pass 2: re-segment flagged frames with CLAHE; keep best-fitting mask ---
    rescued_frames = []
    failed_frames = []
    for t in flagged:
        # retry flagged frames on a CLAHE-enhanced image (different contrast
        # may let the U-Net recover an outlier frame)
        clahe = exposure.equalize_adapthist(bf_norm_stack[t], clip_limit=QC_CLAHE_CLIP)
        m_c, _ = _seg_with_guard(clahe.astype(np.float64))
        tr = trend[t]
        a_norm = float(masks[t].sum())
        a_clahe = float(m_c.sum())
        d_norm = abs(a_norm - tr) if a_norm > 0 else np.inf
        d_clahe = abs(a_clahe - tr) if a_clahe > 0 else np.inf
        if d_clahe < d_norm:
            masks[t] = m_c
            a_chosen, chose_clahe = a_clahe, True
        else:
            a_chosen, chose_clahe = a_norm, False
        if a_chosen <= 0 or a_chosen < QC_RESCUE_LO * tr or a_chosen > QC_RESCUE_HI * tr:
            failed_frames.append(t)          # still implausible after retry
        elif chose_clahe:
            rescued_frames.append(t)

    # Non-flagged frames that simply produced an empty mask are also failures
    for t in range(T):
        if t in blank_frames or t in flagged:
            continue
        if masks[t].sum() == 0:
            failed_frames.append(t)
    failed_frames = sorted(set(failed_frames))
    n_seg_failed = len(failed_frames)
    print(f"  seg: pass1 done, flagged={len(flagged)}, "
          f"CLAHE-rescued={len(rescued_frames)}, failed={n_seg_failed}, "
          f"blank={len(blank_frames)}")

    # Per-frame centroids + aspect ratios (UNALIGNED - used to pick reference frame)
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

    # Apply the SAME rotation to all frames + per-frame centring translation
    aligned_masks = np.zeros_like(masks)
    aligned_bf = np.zeros_like(bf_norm_stack)
    aligned_gfp = np.zeros_like(gfp_raw)
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
        aligned_masks[t] = rotate_centered(masks[t].astype(np.float64), ref_angle_deg, (cy, cx), order=0) > 0.5
        aligned_bf[t] = rotate_centered(bf_norm_stack[t], ref_angle_deg, (cy, cx), order=1)
        aligned_gfp[t] = rotate_centered(gfp_raw[t], ref_angle_deg, (cy, cx), order=1)

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

    # disintegration: first frame past QC_DISINT_MIN_HPF where the mask area falls
    # below 50% of the PRECEDING peak (sustained shrink, not early growth).
    disintegration_frame = -1
    run_max = 0.0
    for t in range(T):
        if areas[t] <= 0:
            continue
        hpf = HPF_START + t * HPF_INTERVAL
        if (hpf >= QC_DISINT_MIN_HPF and run_max > 0
                and areas[t] < QC_DISINTEGRATION_AREA_DROP * run_max):
            disintegration_frame = t
            break
        run_max = max(run_max, areas[t])

    # Aspect ratio (recorded only)
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

    # Alignment QC - only score frames with a well-defined major axis
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

    # Pass/fail (drift-based gate, Option B)
    seg_ok = n_seg_failed <= QC_SEG_FAILED_MAX
    area_ok = median_area >= QC_SEG_MEDIAN_AREA_MIN
    cv_ok = area_cv <= QC_SEG_AREA_CV_MAX
    drift_ok = align_centroid_drift <= QC_ALIGN_CENTROID_DRIFT_PX
    qc_pass = all([seg_ok, area_ok, cv_ok, drift_ok])

    exclude_reason = []
    if not seg_ok: exclude_reason.append(f"seg_failed_frames={n_seg_failed}>{QC_SEG_FAILED_MAX}")
    if not area_ok: exclude_reason.append(f"median_area={median_area:.0f}<{QC_SEG_MEDIAN_AREA_MIN}")
    if not cv_ok: exclude_reason.append(f"area_cv={area_cv:.2f}>{QC_SEG_AREA_CV_MAX}")
    if not drift_ok: exclude_reason.append(f"centroid_drift={align_centroid_drift:.1f}>{QC_ALIGN_CENTROID_DRIFT_PX}")

    metrics = {
        "condition": cond,
        "pescoid": pid,
        "scene_num": scene_num,
        "sample_file": sample_path.name,
        "n_frames": T,
        "hpf_start": HPF_START,
        "hpf_interval": round(HPF_INTERVAL, 4),
        "n_blank_frames": len(blank_frames),
        "blank_frames": blank_frames,
        "seg_failed_frames": n_seg_failed,
        "seg_flagged_frames": len(flagged),
        "seg_clahe_rescued_frames": len(rescued_frames),
        "seg_median_area_px": median_area,
        "seg_area_cv": area_cv,
        "align_max_rotation_deg": align_max_rot,
        "align_orientation_std_deg": align_orient_std,
        "align_n_orient_frames": n_orient_frames,
        "align_centroid_drift_px": align_centroid_drift,
        "aspect_ratio_max": aspect_ratio_max,
        "disintegration_frame": disintegration_frame,
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
    with open(str(pdir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_aligned_montage(aligned_bf, aligned_masks, cond, pid,
                         pdir / f"{pid}_all_timepoints_aligned.png")

    return metrics, aligned_bf, aligned_masks


# ===========================================================================
# QC visualisation
# ===========================================================================
def save_qc_image(cond, pid, bf_raw_t0, mask_raw_t0, aligned_bf, aligned_masks, metrics):
    T = aligned_bf.shape[0]
    frames_with_mask = [t for t in range(T) if aligned_masks[t].any()]
    t_end = frames_with_mask[-1] if frames_with_mask else T - 1
    t_mid = frames_with_mask[len(frames_with_mask) // 2] if frames_with_mask else T // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

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
    axes[0].set_title("Raw t=0 (unaligned)\ncentroid = yellow +", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    _draw_aligned_panel(axes[1], aligned_bf[t_mid], aligned_masks[t_mid], f"Aligned t={t_mid}")
    _draw_aligned_panel(axes[2], aligned_bf[t_end], aligned_masks[t_end], f"Aligned t={t_end}")

    pass_str = "PASS" if metrics["qc_pass"] else "FAIL"
    color = "green" if metrics["qc_pass"] else "red"
    plt.suptitle(
        f"{cond} / {pid}   -   QC {pass_str}   "
        f"(seg_failed={metrics['seg_failed_frames']}, "
        f"median_area={metrics['seg_median_area_px']:.0f}px, "
        f"area_cv={metrics['seg_area_cv']:.2f}, "
        f"orient_std={metrics['align_orientation_std_deg']:.1f}deg, "
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


def save_aligned_montage(aligned_bf, aligned_masks, cond, pid, out_path, n_cols=12):
    """Single image with ALL aligned timepoints (BF + mask contour + centroid)."""
    T = aligned_bf.shape[0]
    n_rows = int(np.ceil(T / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
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
        ax.axhline(H / 2, color="cyan", ls="--", lw=0.4, alpha=0.4)
        ax.axvline(W / 2, color="cyan", ls="--", lw=0.4, alpha=0.4)
        hpf = HPF_START + t * HPF_INTERVAL
        ax.set_title(f"t={t} ({hpf:.1f}h)", fontsize=7)
        ax.axis("off")
    for t in range(T, len(axes)):
        axes[t].axis("off")
    plt.suptitle(f"{cond} / {pid} - all {T} timepoints (aligned, centred)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(str(out_path), dpi=90, bbox_inches="tight")
    plt.close(fig)


def _draw_aligned_panel(ax, bf, mask, title):
    ax.imshow(bf, cmap="gray")
    H, W = bf.shape
    ax.axhline(H / 2, color="cyan", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(W / 2, color="cyan", ls="--", lw=0.8, alpha=0.6)
    if mask.any():
        for c in measure.find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "r-", lw=1.5)
        props = measure.regionprops(mask.astype(int))
        p = max(props, key=lambda r: r.area)
        cy, cx = p.centroid
        ax.plot(cx, cy, "y+", markersize=18, mew=2.5)
        a = p.orientation
        L = p.major_axis_length / 2
        dx = L * np.cos(a)
        dy = -L * np.sin(a)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "lime", lw=2)
        ax.set_title(
            f"{title}  centroid=({cx:.0f}, {cy:.0f})  axis_angle={np.degrees(a):+.1f}deg",
            fontsize=10, fontweight="bold",
        )
    else:
        ax.set_title(f"{title} (no mask)", fontsize=10)
    ax.axis("off")


# ===========================================================================
# Main entry points
# ===========================================================================
def _process_and_qc(scene_num, cond, sample_path, unet, device):
    stack = tifffile.imread(str(sample_path))
    bf_t0_raw = norm_pct(z_project_best_focus(stack[0, :, CH_BF]))
    mask_t0 = segment_unet(unet, device, bf_t0_raw)
    metrics, aligned_bf, aligned_masks = process_pescoid(
        scene_num, cond, sample_path, unet, device)
    save_qc_image(cond, metrics["pescoid"], bf_t0_raw, mask_t0,
                  aligned_bf, aligned_masks, metrics)
    return metrics


def run_scenes(scene_nums):
    scenes = list_scenes()
    by_num = {n: (cond, path) for n, cond, path in scenes}
    print(f"=== Phase 1 (mezzo+H2A) - sanity check on scenes {scene_nums} ===\n")
    unet, device = load_unet()
    print(f"U-Net loaded on {device}\n")
    for n in scene_nums:
        if n not in by_num:
            print(f"ERROR: scene #{n} not found / not in a mapped block")
            continue
        cond, path = by_num[n]
        m = _process_and_qc(n, cond, path, unet, device)
        print(f"\n  {cond}/{m['pescoid']}: QC {'PASS' if m['qc_pass'] else 'FAIL'}")
        for k, v in m.items():
            print(f"    {k}: {v}")
        print()


def run_all():
    print("=== Phase 1 (mezzo+H2A) full dataset run ===\n")
    scenes = list_scenes()
    print(f"Found {len(scenes)} scenes")
    unet, device = load_unet()
    print(f"U-Net loaded on {device}\n")

    all_metrics = []
    for n, cond, path in scenes:
        try:
            all_metrics.append(_process_and_qc(n, cond, path, unet, device))
        except Exception as e:
            warnings.warn(f"Failed on {cond}/#{n} ({path.name}): {e}")
            all_metrics.append({
                "condition": cond, "pescoid": scene_to_pid(n), "scene_num": n,
                "sample_file": path.name, "n_frames": 0, "qc_pass": False,
                "exclude_reason": f"exception: {e}", "notes": "",
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
    p.add_argument("--sample", help="Pescoid ID for sanity check, e.g. S05")
    p.add_argument("--scenes", nargs="*", type=int, help="Scene numbers, e.g. 5 15 35")
    p.add_argument("--all", action="store_true", help="Run on all scenes")
    args = p.parse_args()
    if args.all:
        run_all()
    elif args.scenes:
        run_scenes(args.scenes)
    elif args.sample:
        m = re.search(r"(\d+)", args.sample)
        if not m:
            print("Could not parse a scene number from --sample")
            sys.exit(1)
        run_scenes([int(m.group(1))])
    else:
        p.print_help()
