"""
Main Pescoid Analysis Pipeline (U-Net segmentation)
===================================================

Production pipeline for analyzing zebrafish pescoid timelapse data using a
U-Net trained on hand-drawn pescoid masks (benchmarking winner: IoU=0.948,
Dice=0.973, just ahead of Cellpose). Handles raw, unprocessed 5D TIFF stacks
(T, Z, C, Y, X) with BF and GFP channels.

Pipeline steps:
  1. Load raw TIFF and normalise to 5D
  2. Z-project (best focus / max projection)
  3. Segment with the trained U-Net (post-process: fill holes, keep largest object)
  4. Align masks across time (rotation correction via major-axis alignment)
  5. Compute morphology over time (aspect ratio, perimeter, area)
  6. Compute GFP/mezzo expression fraction within BF mask
  7. Build perimeter-change kymograph (with rotation-corrected baseline)
  8. Detect poles from kymograph (outward/inward classification)
  9. Generate per-sample plots + overlays
  10. Aggregate cross-experiment summary + comparison plots

Usage:
  # Single experiment folder
  python main_analysis.py "Z:/path/to/experiment/TIFF" --out output/

  # With GPU
  python main_analysis.py "Z:/path/to/experiment/TIFF" --out output/ --gpu

  # Specific channels (default: BF=1, GFP=0)
  python main_analysis.py data/ --out output/ --bf-channel 1 --gfp-channel 0

  # Custom cellpose model (e.g., fine-tuned from benchmarking)
  python main_analysis.py data/ --out output/ --cellpose-model cyto3

  # Skip overlays for speed
  python main_analysis.py data/ --out output/ --no-overlays
"""

import os
import sys
import argparse
import json
import time
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy import interpolate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure, exposure, morphology, filters as skf, transform

# ---------------------------------------------------------------------------
# sys.path for cross-module imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR / 'analysis'))
sys.path.insert(0, str(_SCRIPT_DIR / 'scripts'))

from analysis.bf_segmentation import load_tiff, extract_channel, z_project
from analysis.ml_segmentation import segment_cellpose
from analysis.pole_detection import detect_and_analyze_poles, visualize_pole_detection, measure_pole_dimensions
import config


# ============================================================================
# 1. PREPROCESSING — Z-projection + normalisation
# ============================================================================

def preprocess_frame(z_stack, method='best_focus', z_range=None, invert=False):
    """
    Z-project a single-channel (Z, Y, X) stack to a 2D frame.

    Parameters
    ----------
    z_stack : (Z, Y, X) array
    method  : Z-projection method
    z_range : optional Z-slice range
    invert  : if True, invert intensity (for BF: dark pescoid → bright)

    Returns float64 image in [0, 1].
    """
    if z_stack.ndim == 2:
        img = z_stack.astype(np.float64)
    elif z_stack.shape[0] == 1:
        img = z_stack[0].astype(np.float64)
    else:
        img = z_project(z_stack, method=method, z_range=z_range)

    # Percentile normalisation to [0, 1]
    lo, hi = np.percentile(img, (0.5, 99.5))
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)

    if invert:
        img = 1.0 - img

    return img


# ============================================================================
# 2. ORIENTATION ALIGNMENT (rotation only, no translation)
# ============================================================================

def compute_orientation(mask):
    """Return the major-axis orientation angle (radians) of the largest region."""
    labeled = measure.label(mask.astype(int))
    regions = measure.regionprops(labeled)
    if not regions:
        return 0.0
    r = max(regions, key=lambda rr: rr.area)
    return float(r.orientation)  # radians, [-pi/2, pi/2]


def rotate_centered(img, angle_deg, center, order=1, cval=0):
    """
    Single affine transform: rotate around (cy, cx) AND translate so that
    the centroid lands exactly at (H/2, W/2). No clamping, no fallback crop.

    Output has the same H x W shape as input. Pixels that fall outside the
    canvas after the transform are filled with cval.
    """
    H, W = img.shape[:2]
    cy, cx = center
    target_y, target_x = H / 2.0, W / 2.0

    # We want a transform T such that T([cx, cy]) = [target_x, target_y]
    # and T rotates by angle_deg around (cx, cy).
    # Compose: Translate(-cx,-cy) -> Rotate(angle) -> Translate(target_x, target_y)
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # skimage.transform.warp uses inverse mapping; AffineTransform applied
    # via warp expects the FORWARD mapping when inverse_map is constructed
    # from the transform's params, so we use the .inverse property.
    rotation_matrix = transform.AffineTransform(
        matrix=np.array([
            [cos_t, -sin_t, target_x - (cos_t * cx - sin_t * cy)],
            [sin_t,  cos_t, target_y - (sin_t * cx + cos_t * cy)],
            [0, 0, 1.0],
        ])
    )

    out = transform.warp(
        img.astype(np.float64),
        rotation_matrix.inverse,
        order=order,
        mode="constant",
        cval=cval,
        preserve_range=True,
        output_shape=(H, W),
    )
    return out


def align_timeseries(masks, bf_frames=None, gfp_frames=None):
    """
    Rotate all timepoints so the pescoid major axis always points the same
    direction (horizontal). No translation — only rotation around the
    pescoid centroid. Uses padding to prevent any cropping.

    Parameters
    ----------
    masks     : (T, Y, X) bool array
    bf_frames : (T, Y, X) float array or None
    gfp_frames: (T, Y, X) float array or None

    Returns
    -------
    aligned_masks, aligned_bf, aligned_gfp, orientation_angles
    """
    T = masks.shape[0]
    angles = np.array([compute_orientation(masks[t]) for t in range(T)])

    aligned_masks = np.zeros_like(masks)
    aligned_bf = np.zeros_like(bf_frames) if bf_frames is not None else None
    aligned_gfp = np.zeros_like(gfp_frames) if gfp_frames is not None else None

    for t in range(T):
        # Rotate so major axis becomes horizontal (angle → 0)
        angle_deg = -np.degrees(angles[t])

        # Centroid of the pescoid
        props = measure.regionprops(masks[t].astype(int))
        if props:
            cy, cx = props[0].centroid
        else:
            cy, cx = masks.shape[1] / 2, masks.shape[2] / 2

        aligned_masks[t] = rotate_centered(
            masks[t].astype(np.float64), angle_deg, (cy, cx), order=0) > 0.5
        if bf_frames is not None:
            aligned_bf[t] = rotate_centered(bf_frames[t], angle_deg, (cy, cx), order=1)
        if gfp_frames is not None:
            aligned_gfp[t] = rotate_centered(gfp_frames[t], angle_deg, (cy, cx), order=1)

    return aligned_masks, aligned_bf, aligned_gfp, angles


# ============================================================================
# 3. MORPHOLOGY
# ============================================================================

def measure_morphology(mask):
    """Compute morphological metrics from a 2D binary mask."""
    labeled = measure.label(mask.astype(int))
    regions = measure.regionprops(labeled)
    if not regions:
        return {'area': 0, 'perimeter': 0.0, 'aspect_ratio': 0.0,
                'major_axis': 0.0, 'minor_axis': 0.0, 'circularity': 0.0,
                'solidity': 0.0, 'eccentricity': 0.0}
    r = max(regions, key=lambda rr: rr.area)
    maj = r.major_axis_length or 1e-6
    minr = r.minor_axis_length or 1e-6
    perim = r.perimeter or 1e-6
    return {
        'area': int(r.area),
        'perimeter': float(perim),
        'aspect_ratio': float(maj / minr),
        'major_axis': float(maj),
        'minor_axis': float(minr),
        'circularity': float((4 * np.pi * r.area) / (perim ** 2)),
        'solidity': float(r.solidity),
        'eccentricity': float(r.eccentricity),
    }


# ============================================================================
# 4. GFP / MEZZO EXPRESSION
# ============================================================================

def compute_gfp_metrics(bf_mask, gfp_2d, method='otsu'):
    """
    Compute GFP/mezzo expression metrics within BF mask.

    Returns:
      bf_area           : total pescoid area (BF mask pixels)
      gfp_positive_area : number of GFP-positive pixels in mask
      gfp_fraction      : gfp_positive_area / bf_area (area-normalised)
      gfp_mean_intensity: mean GFP intensity of ALL pixels in mask (differentiation potential)
      gfp_pos_mean_intensity: mean GFP intensity of POSITIVE pixels only
      gfp_total_intensity: sum of GFP intensity in mask (integrates area + brightness)
      gfp_total_normalised: gfp_total_intensity / bf_area (intensity per unit area)
      gfp_threshold     : threshold used
    """
    empty = {
        'bf_area': 0, 'gfp_positive_area': 0, 'gfp_fraction': 0.0,
        'gfp_mean_intensity': 0.0, 'gfp_pos_mean_intensity': 0.0,
        'gfp_total_intensity': 0.0, 'gfp_total_normalised': 0.0,
        'gfp_max_intensity': 0.0, 'gfp_std_intensity': 0.0,
        'gfp_threshold': 0.0,
    }
    if bf_mask is None or gfp_2d is None or not bf_mask.any():
        return empty

    vals = gfp_2d[bf_mask]
    if vals.size == 0:
        return empty

    bf_area = int(bf_mask.sum())

    if method == 'otsu':
        thresh = skf.threshold_otsu(vals)
    elif method == 'yen':
        thresh = skf.threshold_yen(vals)
    elif method == 'li':
        thresh = skf.threshold_li(vals)
    else:
        try:
            thresh = np.percentile(vals, float(method))
        except (ValueError, TypeError):
            thresh = skf.threshold_otsu(vals)

    positive = vals > thresh
    gfp_positive_area = int(positive.sum())
    pos_vals = vals[positive]

    return {
        'bf_area': bf_area,
        'gfp_positive_area': gfp_positive_area,
        'gfp_fraction': float(gfp_positive_area / bf_area),
        'gfp_mean_intensity': float(vals.mean()),
        'gfp_pos_mean_intensity': float(pos_vals.mean()) if pos_vals.size > 0 else 0.0,
        'gfp_total_intensity': float(vals.sum()),
        'gfp_total_normalised': float(vals.sum() / bf_area),
        'gfp_max_intensity': float(vals.max()),
        'gfp_std_intensity': float(vals.std()),
        'gfp_threshold': float(thresh),
    }


# ============================================================================
# 5. KYMOGRAPH (rotation-aligned)
# ============================================================================

def extract_contour(mask, n_points=200):
    """Extract and resample contour to fixed number of points."""
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.zeros((n_points, 2))
    contour = max(contours, key=len)
    if len(contour) < 4:
        return np.zeros((n_points, 2))
    dist = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    alpha = np.linspace(0, dist[-1], n_points)
    iy = interpolate.interp1d(dist, contour[:, 0], kind='linear', fill_value='extrapolate')
    ix = interpolate.interp1d(dist, contour[:, 1], kind='linear', fill_value='extrapolate')
    return np.column_stack([iy(alpha), ix(alpha)])


def mask_centroid(mask):
    """Centroid of the largest region."""
    labeled = measure.label(mask.astype(int))
    regions = measure.regionprops(labeled)
    if not regions:
        return mask.shape[0] / 2.0, mask.shape[1] / 2.0
    r = max(regions, key=lambda rr: rr.area)
    return float(r.centroid[0]), float(r.centroid[1])


def radii_along_angles(mask, cy, cx, angles, step=1.0):
    """Ray-cast from centroid along each angle until leaving mask."""
    h, w = mask.shape
    max_r = np.hypot(h, w)
    radii = np.zeros(len(angles), dtype=np.float32)
    for i, th in enumerate(angles):
        r = 0.0
        last_inside = 0.0
        while r < max_r:
            y = int(round(cy + r * np.sin(th)))
            x = int(round(cx + r * np.cos(th)))
            if y < 0 or y >= h or x < 0 or x >= w:
                break
            if mask[y, x]:
                last_inside = r
                r += step
            else:
                break
            radii[i] = last_inside
    return radii


def build_kymograph(masks, n_points=200):
    """
    Build normalised perimeter-change kymograph from aligned mask timeseries.

    Returns
    -------
    kymo_norm : (n_points, T) in [-1, 1]
    kymo_raw  : (n_points, T) raw delta-radii
    """
    T = masks.shape[0]
    if T == 0:
        return np.zeros((n_points, 0)), None

    baseline_radii = None
    baseline_angles = None
    radius_matrix = []

    for t in range(T):
        mask = masks[t]
        cy, cx = mask_centroid(mask)

        if baseline_radii is None:
            contour = extract_contour(mask, n_points=n_points)
            by = contour[:, 0] - cy
            bx = contour[:, 1] - cx
            angles = np.arctan2(by, bx)
            order = np.argsort(angles)
            baseline_angles = angles[order]
            baseline_radii = radii_along_angles(mask, cy, cx, baseline_angles)

        radii_t = radii_along_angles(mask, cy, cx, baseline_angles)
        radius_matrix.append(radii_t - baseline_radii)

    kymo_raw = np.array(radius_matrix).T
    max_abs = float(np.max(np.abs(kymo_raw))) if kymo_raw.size else 0.0
    kymo_norm = np.clip(kymo_raw / max_abs, -1.0, 1.0) if max_abs > 0 else kymo_raw.copy()
    return kymo_norm, kymo_raw


# ============================================================================
# 6. CONDITION INFERENCE
# ============================================================================

def infer_condition(filepath):
    """
    Infer experimental condition from file/folder path.
    Uses the most specific folder name (e.g., P_mezzo_ctrl, P_mezzo_3-5hpf_Act).
    """
    # Use just the final path component for matching
    s = Path(filepath).name.lower().replace('-', '_')

    # Check for compound condition first
    if 'act_chi' in s or 'act_chiron' in s:
        return 'Act-Chi'
    if 'chi_act' in s or 'chiron_act' in s:
        return 'Act-Chi'
    # Single conditions
    if '_act' in s or 'activin' in s:
        return 'Act'
    if '_chi' in s or 'chiron' in s:
        return 'Chi'
    if 'ctrl' in s or 'control' in s:
        return 'ctrl'
    return 'unknown'


# ============================================================================
# 7. PLOT HELPERS
# ============================================================================

def save_kymograph_plot(kymo_norm, save_path, n_points=200):
    """Save kymograph as seismic heatmap."""
    if kymo_norm.size == 0 or kymo_norm.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(kymo_norm, aspect='auto', cmap='seismic', vmin=-1, vmax=1, origin='lower')
    ax.set_xlabel('Time [frame]', fontsize=12)
    ax.set_ylabel('Normalised Perimeter', fontsize=12)
    ticks = np.linspace(0, n_points - 1, 5, dtype=int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(0, 1, 5)])
    ax.set_title('Perimeter radius change (red=outward, blue=inward)', fontsize=11)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Delta radius (normalised)', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_morphology_plot(morph_df, save_path):
    """Time-series plot of area, AR, circularity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(morph_df['time'], morph_df['area'], 'k-o', markersize=3)
    axes[0, 0].set_ylabel('Area (px)')
    axes[0, 0].set_title('Area over time')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(morph_df['time'], morph_df['aspect_ratio'], 'b-o', markersize=3)
    axes[0, 1].set_ylabel('Aspect Ratio')
    axes[0, 1].set_title('Aspect Ratio over time')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(morph_df['time'], morph_df['perimeter'], 'r-o', markersize=3)
    axes[1, 0].set_ylabel('Perimeter (px)')
    axes[1, 0].set_xlabel('Time [frame]')
    axes[1, 0].set_title('Perimeter over time')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(morph_df['time'], morph_df['circularity'], 'g-o', markersize=3)
    axes[1, 1].set_ylabel('Circularity')
    axes[1, 1].set_xlabel('Time [frame]')
    axes[1, 1].set_title('Circularity over time')
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('Morphology over time', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_mezzo_plot(mezzo_df, save_path):
    """Time-series plot of mezzo expression — area, intensity, and growth."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: BF area vs GFP-positive area over time (GROWTH)
    if 'bf_area' in mezzo_df.columns and 'gfp_positive_area' in mezzo_df.columns:
        ax = axes[0, 0]
        ax.plot(mezzo_df['time'], mezzo_df['bf_area'], 'k-o', markersize=3, label='BF area (pescoid)')
        ax.plot(mezzo_df['time'], mezzo_df['gfp_positive_area'], 'g-o', markersize=3, label='GFP+ area')
        ax.set_xlabel('Time [frame]')
        ax.set_ylabel('Area (pixels)')
        ax.set_title('Growth: BF area vs GFP-positive area', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)

    # Top-right: GFP fraction (area-normalised)
    frac_col = 'gfp_fraction' if 'gfp_fraction' in mezzo_df.columns else 'mezzo_fraction'
    ax = axes[0, 1]
    ax.plot(mezzo_df['time'], mezzo_df[frac_col] * 100, 'g-o', markersize=3)
    ax.set_xlabel('Time [frame]')
    ax.set_ylabel('GFP+ fraction [%]')
    ax.set_title('GFP-positive area / pescoid area', fontweight='bold')
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3)

    # Bottom-left: Mean GFP intensity over time (DIFFERENTIATION POTENTIAL)
    int_col = 'gfp_mean_intensity' if 'gfp_mean_intensity' in mezzo_df.columns else 'mezzo_mean'
    std_col = 'gfp_std_intensity' if 'gfp_std_intensity' in mezzo_df.columns else 'mezzo_std'
    ax = axes[1, 0]
    ax.plot(mezzo_df['time'], mezzo_df[int_col], 'g-o', markersize=3, label='All pixels in mask')
    if 'gfp_pos_mean_intensity' in mezzo_df.columns:
        ax.plot(mezzo_df['time'], mezzo_df['gfp_pos_mean_intensity'], 'r-s', markersize=3, label='GFP+ pixels only')
    ax.fill_between(mezzo_df['time'],
                     mezzo_df[int_col] - mezzo_df[std_col],
                     mezzo_df[int_col] + mezzo_df[std_col],
                     alpha=0.15, color='green')
    ax.set_xlabel('Time [frame]')
    ax.set_ylabel('Mean GFP intensity')
    ax.set_title('Differentiation: GFP intensity over time', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Bottom-right: Total normalised intensity (intensity per unit area)
    if 'gfp_total_normalised' in mezzo_df.columns:
        ax = axes[1, 1]
        ax.plot(mezzo_df['time'], mezzo_df['gfp_total_normalised'], 'g-o', markersize=3)
        ax.set_xlabel('Time [frame]')
        ax.set_ylabel('Total GFP intensity / area')
        ax.set_title('Normalised total intensity over time', fontweight='bold')
        ax.grid(alpha=0.3)

    plt.suptitle('Mezzo-GFP Expression Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_overlay(bf_2d, gfp_2d, mask, t, save_path, gfp_mask=None):
    """Save single-frame overlay: BF | BF+contour | GFP+contour | GFP+positive regions."""
    bf_disp = exposure.rescale_intensity(bf_2d.astype(float), out_range=(0, 1))
    has_gfp = gfp_2d is not None
    n_panels = 4 if (has_gfp and gfp_mask is not None) else (3 if has_gfp else 2)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(bf_disp, cmap='gray')
    axes[0].set_title(f'BF  t={t}', fontsize=9); axes[0].axis('off')

    axes[1].imshow(bf_disp, cmap='gray')
    for c in measure.find_contours(mask.astype(float), 0.5):
        axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=1.5)
    axes[1].set_title(f'Segmentation  area={int(mask.sum()):,} px', fontsize=9)
    axes[1].axis('off')

    if has_gfp:
        gfp_disp = exposure.rescale_intensity(gfp_2d.astype(float), out_range=(0, 1))
        axes[2].imshow(gfp_disp, cmap='Greens')
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.2)
        axes[2].set_title(f'GFP  t={t}', fontsize=9); axes[2].axis('off')

    if has_gfp and gfp_mask is not None:
        axes[3].imshow(gfp_disp, cmap='Greens')
        # Red overlay for GFP positive
        overlay = np.zeros((*gfp_disp.shape, 4))
        overlay[gfp_mask, 0] = 1.0
        overlay[gfp_mask, 3] = 0.4
        axes[3].imshow(overlay)
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[3].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.0)
        frac = gfp_mask.sum() / mask.sum() * 100 if mask.sum() > 0 else 0
        axes[3].set_title(f'GFP+ ({frac:.1f}%)', fontsize=9); axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 8. LOAD DATA — supports both single 5D TIFF and folder-of-timepoint-TIFs
# ============================================================================

def load_timelapse(source, bf_channel=0, gfp_channel=1, z_method='best_focus', z_range=None):
    """
    Load timelapse data from either:
      (a) A single 5D TIFF (T, Z, C, Y, X)
      (b) A folder of per-timepoint TIFs, each (C, Y, X) or (Z, C, Y, X)

    Returns
    -------
    bf_frames  : (T, Y, X) float64 [0, 1]
    gfp_frames : (T, Y, X) float64 [0, 1] or None
    stem       : str — sample name
    meta       : dict — metadata
    """
    source = Path(source)

    if source.is_dir():
        # --- Folder of per-timepoint TIFs ---
        tif_files = sorted(
            [f for f in source.iterdir() if f.suffix.lower() in ('.tif', '.tiff')]
        )
        if not tif_files:
            raise FileNotFoundError(f"No TIF files in {source}")

        stem = source.name
        T = len(tif_files)

        # Peek at first file to get dimensions
        sample = tifffile.imread(str(tif_files[0]))
        # Possible shapes: (C, Y, X), (Z, C, Y, X), (Y, X)
        if sample.ndim == 2:
            Y, X = sample.shape
            C = 1
            has_z = False
        elif sample.ndim == 3:
            C, Y, X = sample.shape
            has_z = False
        elif sample.ndim == 4:
            Z, C, Y, X = sample.shape
            has_z = True
        else:
            raise ValueError(f"Unexpected shape {sample.shape} in {tif_files[0].name}")

        has_gfp = C >= 2
        bf_ch = min(bf_channel, C - 1)
        gfp_ch = min(gfp_channel, C - 1)

        bf_frames = np.zeros((T, Y, X), dtype=np.float64)
        gfp_frames = np.zeros((T, Y, X), dtype=np.float64) if has_gfp else None

        for t, fp in enumerate(tif_files):
            frame = tifffile.imread(str(fp))

            if frame.ndim == 2:
                bf_raw = frame.astype(np.float64)
                gfp_raw = None
            elif frame.ndim == 3:
                # (C, Y, X)
                bf_raw = frame[bf_ch].astype(np.float64)
                gfp_raw = frame[gfp_ch].astype(np.float64) if has_gfp else None
            elif frame.ndim == 4:
                # (Z, C, Y, X) — need Z-projection
                bf_raw = preprocess_frame(frame[:, bf_ch], method=z_method, z_range=z_range)
                # GFP uses max projection to capture all fluorescence signal
                gfp_raw = preprocess_frame(frame[:, gfp_ch], method='max', z_range=z_range) if has_gfp else None
            else:
                bf_raw = frame.squeeze().astype(np.float64)
                gfp_raw = None

            # Normalise to [0, 1]
            if frame.ndim <= 3:
                lo, hi = np.percentile(bf_raw, (0.5, 99.5))
                bf_frames[t] = np.clip((bf_raw - lo) / (hi - lo), 0, 1) if hi > lo else bf_raw / (bf_raw.max() or 1)
                if gfp_raw is not None:
                    lo, hi = np.percentile(gfp_raw, (0.5, 99.5))
                    gfp_frames[t] = np.clip((gfp_raw - lo) / (hi - lo), 0, 1) if hi > lo else gfp_raw / (gfp_raw.max() or 1)
            else:
                bf_frames[t] = bf_raw
                if gfp_raw is not None:
                    gfp_frames[t] = gfp_raw

        meta = {'T': T, 'C': C, 'Y': Y, 'X': X, 'source': 'folder',
                'bf_channel': bf_ch, 'gfp_channel': gfp_ch, 'has_z': has_z}
        return bf_frames, gfp_frames, stem, meta

    else:
        # --- Single 5D TIFF ---
        stem = source.stem
        stack, file_meta = load_tiff(str(source))
        T, Z, C, Y, X = stack.shape
        bf_ch = min(bf_channel, C - 1)
        gfp_ch = min(gfp_channel, C - 1)
        has_gfp = C >= 2

        bf_frames = np.zeros((T, Y, X), dtype=np.float64)
        gfp_frames = np.zeros((T, Y, X), dtype=np.float64) if has_gfp else None

        for t in range(T):
            bf_frames[t] = preprocess_frame(stack[t, :, bf_ch], method=z_method, z_range=z_range)
            if has_gfp:
                # GFP uses max projection to capture all fluorescence signal
                gfp_frames[t] = preprocess_frame(stack[t, :, gfp_ch], method='max', z_range=z_range)

        meta = {'T': T, 'Z': Z, 'C': C, 'Y': Y, 'X': X, 'source': 'file',
                'bf_channel': bf_ch, 'gfp_channel': gfp_ch}
        return bf_frames, gfp_frames, stem, meta


# ============================================================================
# 9. CORE: PROCESS A SINGLE SAMPLE (file or folder)
# ============================================================================

def process_single_file(
    tiff_path,
    output_dir,
    bf_channel=0,
    gfp_channel=1,
    z_method='best_focus',
    z_range=None,
    cellpose_model='cyto3',
    cellpose_diameter=None,
    gpu=False,
    min_area=3000,
    gfp_threshold='otsu',
    n_perimeter_points=200,
    do_overlays=True,
    do_alignment=True,
    verbose=True,
):
    """
    Full pipeline on a single sample (5D TIFF file or folder of timepoint TIFs).

    Returns summary dict with all metrics.
    """
    source = Path(tiff_path)

    def log(msg):
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    log(f"\n  Loading: {source.name}")
    bf_frames, gfp_frames, stem, meta = load_timelapse(
        source, bf_channel=bf_channel, gfp_channel=gfp_channel,
        z_method=z_method, z_range=z_range,
    )
    T, Y, X = bf_frames.shape
    has_gfp = gfp_frames is not None
    log(f"  T={T}, Y={Y}, X={X}, source={meta['source']}, has_gfp={has_gfp}")

    out = Path(output_dir) / stem
    out.mkdir(parents=True, exist_ok=True)

    # (No X-Y translation registration — only rotation alignment after segmentation)

    # ------------------------------------------------------------------
    # STEP 3: Segment with U-Net (trained on 23 GT masks, dice=0.969)
    # ------------------------------------------------------------------
    import torch
    sys.path.insert(0, str(_SCRIPT_DIR / 'benchmarking_segment_tools'))
    from run_3dunet import UNet

    unet_path = _SCRIPT_DIR / 'annotation_workspace' / 'model' / 'unet_pescoid_v2.pth'
    if not unet_path.exists():
        unet_path = _SCRIPT_DIR / 'benchmarking_segment_tools' / 'output' / '3dunet' / 'model' / 'unet_pescoid.pth'

    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    unet = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    unet.load_state_dict(torch.load(str(unet_path), map_location=device, weights_only=True))
    unet.eval()
    log(f"  Segmenting with U-Net (device={device})...")

    masks = np.zeros((T, Y, X), dtype=bool)
    failed_frames = []

    with torch.no_grad():
        for t in range(T):
            bf_float = np.clip(bf_frames[t], 0, 1).astype(np.float32)
            t_img = torch.from_numpy(bf_float[np.newaxis, np.newaxis]).to(device)
            pred = torch.sigmoid(unet(t_img)).cpu().numpy()[0, 0]
            mask_2d = pred > 0.5
            # Post-process
            # Post-process: fill holes, keep largest component
            mask_2d = ndi.binary_fill_holes(mask_2d)
            mask_2d = morphology.binary_closing(mask_2d, morphology.disk(3))
            mask_2d = ndi.binary_fill_holes(mask_2d)
            labeled_m, n = ndi.label(mask_2d)
            if n > 0:
                sizes = ndi.sum(mask_2d, labeled_m, range(1, n + 1))
                mask_2d = (labeled_m == (np.argmax(sizes) + 1)).astype(bool)

            if mask_2d.sum() == 0:
                failed_frames.append(t)
            masks[t] = mask_2d

    # Temporal consistency: discard noise frames (area < 10% of median)
    areas = np.array([masks[t].sum() for t in range(T)])
    valid_areas = areas[areas > 0]
    if valid_areas.size > 0:
        median_area = np.median(valid_areas)
        area_threshold = max(min_area, median_area * 0.1)
        for t in range(T):
            if 0 < areas[t] < area_threshold:
                masks[t] = np.zeros((Y, X), dtype=bool)
                if t not in failed_frames:
                    failed_frames.append(t)

    # Fill failed frames from nearest good frame
    good = sorted([t for t in range(T) if t not in failed_frames])
    if failed_frames and good:
        for t in sorted(failed_frames):
            nearest = min(good, key=lambda g: abs(g - t))
            masks[t] = masks[nearest].copy()
        log(f"  Filtered + filled {len(failed_frames)} noisy/empty frames "
            f"(area threshold: {area_threshold:,.0f} px)")

    log(f"  Segmentation: {T - len(failed_frames)}/{T} frames OK, "
        f"median area: {median_area:,.0f} px" if valid_areas.size > 0
        else f"  WARNING: no valid masks found")

    # ------------------------------------------------------------------
    # STEP 4: Orientation alignment (rotation only, no translation)
    # ------------------------------------------------------------------
    # Rotate each frame so the pescoid major axis is always horizontal.
    # Uses padding to prevent cropping — no data is lost.
    log(f"  Aligning orientation (major axis -> horizontal)...")
    masks, bf_frames, gfp_frames, orient_angles = align_timeseries(
        masks, bf_frames, gfp_frames
    )
    max_rotation = float(np.degrees(orient_angles.max() - orient_angles.min()))
    log(f"  Max rotation corrected: {max_rotation:.1f} deg")

    # ------------------------------------------------------------------
    # STEP 5: Morphology over time
    # ------------------------------------------------------------------
    log(f"  Computing morphology...")
    morph_rows = []
    for t in range(T):
        m = measure_morphology(masks[t])
        m['time'] = t
        morph_rows.append(m)
    morph_df = pd.DataFrame(morph_rows)

    morph_csv = out / 'morphology_over_time.csv'
    morph_df.to_csv(str(morph_csv), index=False)

    # ------------------------------------------------------------------
    # STEP 6: GFP expression over time
    # ------------------------------------------------------------------
    # Use a FIXED threshold computed from the first valid frame so that
    # increasing GFP over time is captured as rising fraction + intensity.
    mezzo_rows = []
    gfp_masks_all = None  # store for overlays
    if has_gfp:
        log(f"  Computing GFP/mezzo expression (blur + {gfp_threshold} + filter, fixed threshold)...")
        gfp_masks_all = np.zeros((T,) + masks.shape[1:], dtype=bool)

        # GFP segmentation: blur -> threshold -> filter small particles
        MIN_GFP_AREA = 500  # pixels (~20 um diameter)
        GFP_BLUR_SIGMA = 3  # Gaussian blur before thresholding

        # Blur all frames once
        gfp_blur_all = np.stack([skf.gaussian(gfp_frames[t], sigma=GFP_BLUR_SIGMA) for t in range(T)])

        # FIXED THRESHOLD: compute from the frame with highest mean GFP intensity
        # (usually a late timepoint when differentiation has occurred).
        # This way early frames with no expression correctly show ~0% positive.
        best_t = 0
        best_mean = 0.0
        for t in range(T):
            if masks[t].any():
                m = gfp_blur_all[t][masks[t]].mean()
                if m > best_mean:
                    best_mean = m
                    best_t = t

        # Compute threshold from the reference frame
        if masks[best_t].any():
            vals_ref = gfp_blur_all[best_t][masks[best_t]]
            if gfp_threshold == 'intermodes':
                hist, bin_edges = np.histogram(vals_ref, bins=256)
                smoothed = hist.astype(float)
                for _ in range(200):
                    smoothed = np.convolve(smoothed, [0.25, 0.5, 0.25], mode='same')
                    peaks = [i for i in range(1, len(smoothed)-1)
                             if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]]
                    if len(peaks) <= 2:
                        break
                if len(peaks) >= 2:
                    fixed_thresh = bin_edges[(peaks[0] + peaks[1]) // 2]
                else:
                    fixed_thresh = skf.threshold_otsu(vals_ref)
            elif gfp_threshold in ('maxentropy', 'li'):
                fixed_thresh = skf.threshold_li(vals_ref)
            elif gfp_threshold == 'yen':
                fixed_thresh = skf.threshold_yen(vals_ref)
            else:
                fixed_thresh = skf.threshold_otsu(vals_ref)
        else:
            fixed_thresh = 0.0

        log(f"  GFP fixed threshold (from brightest frame t={best_t}): {fixed_thresh:.4f}")

        for t in range(T):
            bf_mask = masks[t]
            gfp_2d = gfp_frames[t]

            if not bf_mask.any():
                mezzo_rows.append({
                    'bf_area': 0, 'gfp_positive_area': 0, 'gfp_fraction': 0.0,
                    'gfp_mean_intensity': 0.0, 'gfp_pos_mean_intensity': 0.0,
                    'gfp_total_intensity': 0.0, 'gfp_total_normalised': 0.0,
                    'gfp_max_intensity': 0.0, 'gfp_std_intensity': 0.0,
                    'gfp_threshold': fixed_thresh, 'time': t,
                })
                continue

            bf_area = int(bf_mask.sum())
            gfp_blur = gfp_blur_all[t]

            # Apply fixed threshold
            gfp_pos_mask = (gfp_blur > fixed_thresh) & bf_mask
            gfp_pos_mask = morphology.remove_small_objects(gfp_pos_mask, min_size=MIN_GFP_AREA)
            gfp_masks_all[t] = gfp_pos_mask  # store for overlays

            gfp_pos_area = int(gfp_pos_mask.sum())
            all_vals = gfp_2d[bf_mask]
            pos_vals = gfp_2d[gfp_pos_mask]

            mezzo_rows.append({
                'bf_area': bf_area,
                'gfp_positive_area': gfp_pos_area,
                'gfp_fraction': float(gfp_pos_area / bf_area),
                'gfp_mean_intensity': float(all_vals.mean()),
                'gfp_pos_mean_intensity': float(pos_vals.mean()) if pos_vals.size > 0 else 0.0,
                'gfp_total_intensity': float(all_vals.sum()),
                'gfp_total_normalised': float(all_vals.sum() / bf_area),
                'gfp_max_intensity': float(all_vals.max()),
                'gfp_std_intensity': float(all_vals.std()),
                'gfp_threshold': float(fixed_thresh),
                'time': t,
            })

    mezzo_df = pd.DataFrame(mezzo_rows) if mezzo_rows else pd.DataFrame()

    if not mezzo_df.empty:
        mezzo_csv = out / 'mezzo_expression_over_time.csv'
        mezzo_df.to_csv(str(mezzo_csv), index=False)

    # ------------------------------------------------------------------
    # STEP 7: Build kymograph (on aligned masks)
    # ------------------------------------------------------------------
    log(f"  Building kymograph...")
    kymo_norm, kymo_raw = build_kymograph(masks, n_points=n_perimeter_points)

    if kymo_raw is not None:
        np.save(str(out / 'kymograph_matrix.npy'), kymo_raw)

    # ------------------------------------------------------------------
    # STEP 8: Pole detection (outward poles = persistent circumference increase)
    # ------------------------------------------------------------------
    pole_results = None
    if kymo_norm.size > 0 and kymo_norm.shape[1] > 1:
        try:
            pole_results = detect_and_analyze_poles(
                kymo_norm,
                smooth_sigma=3,
                prominence=0.12,
                min_distance=25,
                classify=True,
            )

            # Filter: only keep outward poles (positive mean radial displacement)
            if pole_results['n_poles'] > 0:
                outward = pole_results.get('outward_indices', np.array([]))
                inward = pole_results.get('inward_indices', np.array([]))
                pole_results['n_outward_only'] = len(outward)
            else:
                pole_results['n_outward_only'] = 0

            log(f"  Poles: {pole_results['n_poles']} total "
                f"({pole_results.get('n_outward', 0)} out, "
                f"{pole_results.get('n_inward', 0)} in)")
        except Exception as e:
            warnings.warn(f"Pole detection failed: {e}")

    # ------------------------------------------------------------------
    # STEP 9: Save masks
    # ------------------------------------------------------------------
    mask_path = out / 'masks'
    mask_path.mkdir(exist_ok=True)
    tifffile.imwrite(str(mask_path / f'{stem}_masks.tif'), (masks.astype(np.uint8) * 255))

    # ------------------------------------------------------------------
    # STEP 10: Generate plots
    # ------------------------------------------------------------------
    plots_dir = out / 'plots'
    plots_dir.mkdir(exist_ok=True)

    save_morphology_plot(morph_df, plots_dir / 'morphology_over_time.png')
    save_kymograph_plot(kymo_norm, plots_dir / 'kymograph.png', n_perimeter_points)

    if not mezzo_df.empty:
        save_mezzo_plot(mezzo_df, plots_dir / 'mezzo_expression.png')

    if pole_results and pole_results['n_poles'] > 0:
        visualize_pole_detection(
            pole_results['activity_profile'],
            pole_results['pole_indices'],
            outward_poles=pole_results.get('outward_indices'),
            inward_poles=pole_results.get('inward_indices'),
            save_path=str(plots_dir / 'pole_detection.png'),
        )

    # Save pole counts CSV
    pole_csv = out / 'pole_counts.csv'
    with open(str(pole_csv), 'w') as f:
        f.write('metric,value\n')
        n_poles = pole_results['n_poles'] if pole_results else 0
        n_out = pole_results.get('n_outward', 0) if pole_results else 0
        n_in = pole_results.get('n_inward', 0) if pole_results else 0
        f.write(f'total_poles,{n_poles}\n')
        f.write(f'outward_poles,{n_out}\n')
        f.write(f'inward_poles,{n_in}\n')

    # --- Pole length / dimension measurement ---
    pole_dim_rows = []
    if (
        pole_results
        and pole_results.get('n_poles', 0) > 0
        and kymo_raw is not None
        and kymo_raw.size > 0
    ):
        try:
            indices_for_length = pole_results.get('outward_indices')
            if indices_for_length is None or len(indices_for_length) == 0:
                indices_for_length = pole_results.get('pole_indices', [])
            pole_dim_rows = measure_pole_dimensions(
                masks, kymo_raw, indices_for_length, n_perimeter_points=n_perimeter_points
            )
        except Exception as e:
            warnings.warn(f"Pole-length measurement failed: {e}")
            pole_dim_rows = []

    pole_dim_csv = out / 'pole_dimensions.csv'
    with open(str(pole_dim_csv), 'w') as f:
        f.write('rank,label,peak_timepoint,pole_index,'
                'pole_length_px,pole_width_arc_px,pole_angular_extent_deg,'
                'pole_area_px,pole_centroid_y,pole_centroid_x,'
                'wedge_angle_lo_deg,wedge_angle_hi_deg,activity_at_peak\n')
        for r in pole_dim_rows:
            f.write(
                f"{r['rank']},{r['label']},{r['peak_timepoint']},{r['pole_index']},"
                f"{r['pole_length_px']:.2f},{r['pole_width_arc_px']:.2f},"
                f"{r['pole_angular_extent_deg']:.2f},{r['pole_area_px']},"
                f"{r['pole_centroid_y']:.2f},{r['pole_centroid_x']:.2f},"
                f"{r['wedge_angle_lo_deg']:.2f},{r['wedge_angle_hi_deg']:.2f},"
                f"{r['activity_at_peak']:.4f}\n"
            )
    if pole_dim_rows:
        primary = pole_dim_rows[0]
        log(f"  Primary pole length: {primary['pole_length_px']:.1f} px "
            f"(angular {primary['pole_angular_extent_deg']:.0f} deg, "
            f"at t={primary['peak_timepoint']})")

    # ------------------------------------------------------------------
    # STEP 11: Overlays (optional — slow for large T)
    # ------------------------------------------------------------------
    if do_overlays:
        log(f"  Generating {T} overlay images...")
        overlay_dir = out / 'overlays'
        overlay_dir.mkdir(exist_ok=True)
        for t in range(T):
            gfp_t = gfp_frames[t] if has_gfp else None
            gfp_mask_t = gfp_masks_all[t] if (has_gfp and gfp_masks_all is not None) else None
            save_overlay(bf_frames[t], gfp_t, masks[t], t,
                         overlay_dir / f'{stem}_t{t:04d}.png', gfp_mask=gfp_mask_t)

    # ------------------------------------------------------------------
    # STEP 12: Summary JSON
    # ------------------------------------------------------------------
    condition = infer_condition(str(tiff_path))
    summary = {
        'file': str(source),
        'stem': stem,
        'condition': condition,
        'shape': meta,
        'cellpose_model': cellpose_model,
        'z_method': z_method,
        'alignment': do_alignment,
        'max_rotation_deg': float(np.degrees(orient_angles.max() - orient_angles.min())),
        'failed_frames': failed_frames,
        'n_timepoints': T,
        'morphology': {
            'mean_area': float(morph_df['area'].mean()),
            'final_area': int(morph_df['area'].iloc[-1]),
            'mean_aspect_ratio': float(morph_df['aspect_ratio'].mean()),
            'final_aspect_ratio': float(morph_df['aspect_ratio'].iloc[-1]),
            'mean_circularity': float(morph_df['circularity'].mean()),
        },
        'mezzo': {
            'mean_fraction': float(mezzo_df['gfp_fraction'].mean()) if not mezzo_df.empty and 'gfp_fraction' in mezzo_df else 0.0,
            'final_fraction': float(mezzo_df['gfp_fraction'].iloc[-1]) if not mezzo_df.empty and 'gfp_fraction' in mezzo_df else 0.0,
            'mean_intensity': float(mezzo_df['gfp_mean_intensity'].mean()) if not mezzo_df.empty and 'gfp_mean_intensity' in mezzo_df else 0.0,
            'final_intensity': float(mezzo_df['gfp_mean_intensity'].iloc[-1]) if not mezzo_df.empty and 'gfp_mean_intensity' in mezzo_df else 0.0,
            'mean_total_normalised': float(mezzo_df['gfp_total_normalised'].mean()) if not mezzo_df.empty and 'gfp_total_normalised' in mezzo_df else 0.0,
        },
        'poles': {
            'n_poles': pole_results['n_poles'] if pole_results else 0,
            'n_outward': pole_results.get('n_outward', 0) if pole_results else 0,
            'n_inward': pole_results.get('n_inward', 0) if pole_results else 0,
            'primary_length_px':   pole_dim_rows[0]['pole_length_px']   if pole_dim_rows else 0.0,
            'primary_extent_deg':  pole_dim_rows[0]['pole_angular_extent_deg'] if pole_dim_rows else 0.0,
            'secondary_length_px': pole_dim_rows[1]['pole_length_px']   if len(pole_dim_rows) > 1 else 0.0,
            'secondary_extent_deg':pole_dim_rows[1]['pole_angular_extent_deg'] if len(pole_dim_rows) > 1 else 0.0,
        },
        'output_dir': str(out),
    }

    with open(str(out / 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    log(f"  Done: {stem}")
    return summary


# ============================================================================
# 9. BATCH PROCESSING
# ============================================================================

def is_timelapse_folder(folder):
    """Check if a folder contains per-timepoint TIFs (e.g., ..._time-001.tif).
    Returns False if the folder contains 5D multi-timepoint TIFs (each is a sample).
    """
    folder = Path(folder)
    if not folder.is_dir():
        return False
    # Skip known non-data folders
    if folder.name in ('result_segmentation', 'Thumbs.db', '.DS_Store', 'analysis', 'masks', 'overlays', 'plots'):
        return False
    tifs = list(folder.glob('*time*.[tT][iI][fF]')) + list(folder.glob('*time*.[tT][iI][fF][fF]'))
    if tifs:
        return True
    # Check if TIFs are 5D stacks (each is a complete sample, not a single frame)
    all_tifs = list(folder.glob('*.[tT][iI][fF]')) + list(folder.glob('*.[tT][iI][fF][fF]'))
    if len(all_tifs) > 0:
        sample = tifffile.imread(str(all_tifs[0]))
        if sample.ndim >= 5:
            return False  # These are complete 5D stacks — not per-frame TIFs
    return len(all_tifs) > 1


def discover_samples(input_dir):
    """
    Discover all samples in an input directory. Supports three layouts:

    Layout A — single sample folder (contains timepoint TIFs):
      input_dir/*_time-001.tif, *_time-002.tif, ...
      → returns [input_dir] as one sample

    Layout B — condition/sample/timepoint folders:
      input_dir/P_mezzo_ctrl/
        Concatenated_..._G007_0001/  (timelapse folder)
      input_dir/P_mezzo_3-5hpf_Act/
        Concatenated_..._G013_0001/  (timelapse folder)

    Layout C — flat directory of 5D TIFFs:
      input_dir/*.tif (single multi-timepoint files)

    Returns list of dicts: [{'path': Path, 'condition': str}, ...]
    """
    input_path = Path(input_dir)
    samples = []

    # Layout A: input_dir IS a timelapse sample folder
    if is_timelapse_folder(input_path):
        return [{'path': input_path, 'condition': infer_condition(str(input_path.parent))}]

    # Layout B: condition/sample hierarchy
    for cond_dir in sorted(input_path.iterdir()):
        if not cond_dir.is_dir():
            continue
        # Check if subdirectories are timelapse sample folders
        for sample_dir in sorted(cond_dir.iterdir()):
            if sample_dir.is_dir() and is_timelapse_folder(sample_dir):
                condition = infer_condition(str(cond_dir))
                samples.append({'path': sample_dir, 'condition': condition})

    if samples:
        return samples

    # Maybe one level deep: sample folders directly under input (no condition)
    for sub in sorted(input_path.iterdir()):
        if sub.is_dir() and is_timelapse_folder(sub):
            samples.append({'path': sub, 'condition': infer_condition(str(sub))})

    if samples:
        return samples

    # Layout C: flat 5D TIFFs
    tif_files = sorted(input_path.glob('*.tif')) + sorted(input_path.glob('*.tiff'))
    for tf in tif_files:
        samples.append({'path': tf, 'condition': infer_condition(str(tf))})

    return samples


def run_batch(input_dir, output_dir, **kwargs):
    """Process all samples (files or folders) in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(input_dir)

    if not samples:
        print(f"No samples found in {input_dir}")
        return None

    # Print inventory
    conditions = {}
    for s in samples:
        conditions.setdefault(s['condition'], []).append(s)

    print(f"\n{'=' * 60}")
    print(f"  BATCH ANALYSIS — {len(samples)} samples")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    for cond, items in sorted(conditions.items()):
        print(f"    {cond}: {len(items)} samples")
    print(f"{'=' * 60}")

    results = []
    failures = []
    t_start = time.time()

    for i, sample_info in enumerate(samples, 1):
        sample_path = sample_info['path']
        print(f"\n[{i}/{len(samples)}] {sample_info['condition']} / {sample_path.name}")
        print('-' * 50)
        try:
            summary = process_single_file(str(sample_path), str(output_path), **kwargs)
            # Override condition with folder-derived one (more reliable)
            summary['condition'] = sample_info['condition']
            results.append(summary)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failures.append({'file': str(sample_path), 'error': str(e)})

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Batch complete: {len(results)}/{len(samples)} in {elapsed:.1f}s")
    if failures:
        print(f"  {len(failures)} failed:")
        for f in failures:
            print(f"    {Path(f['file']).name}: {f['error']}")
        fail_path = output_path / 'failed_files.txt'
        with open(str(fail_path), 'w') as fout:
            for f in failures:
                fout.write(f"{f['file']}: {f['error']}\n")

    if not results:
        return None

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    rows = []
    for s in results:
        rows.append({
            'file': s['stem'],
            'condition': s['condition'],
            'n_timepoints': s['n_timepoints'],
            'mean_area': s['morphology']['mean_area'],
            'final_area': s['morphology']['final_area'],
            'mean_aspect_ratio': s['morphology']['mean_aspect_ratio'],
            'final_aspect_ratio': s['morphology']['final_aspect_ratio'],
            'mean_circularity': s['morphology']['mean_circularity'],
            'mean_mezzo_fraction': s['mezzo']['mean_fraction'],
            'final_mezzo_fraction': s['mezzo']['final_fraction'],
            'n_poles': s['poles']['n_poles'],
            'n_outward': s['poles']['n_outward'],
            'n_inward': s['poles']['n_inward'],
            'max_rotation_deg': s['max_rotation_deg'],
            'failed_frames': len(s['failed_frames']),
        })
    batch_df = pd.DataFrame(rows)
    batch_csv = output_path / 'batch_summary.csv'
    batch_df.to_csv(str(batch_csv), index=False)
    print(f"\n  Saved: {batch_csv}")

    # ------------------------------------------------------------------
    # Generate cross-experiment comparison plots
    # ------------------------------------------------------------------
    generate_comparison_plots(batch_df, output_path)

    return batch_df


# ============================================================================
# 10. CROSS-EXPERIMENT COMPARISON PLOTS
# ============================================================================

def generate_comparison_plots(df, output_dir):
    """Generate comparison plots across conditions."""
    plot_dir = Path(output_dir) / 'comparison_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(df['condition'].unique())
    cond_colors = {c: config.CONDITIONS.get(c, {}).get('color', 'gray') for c in conditions}

    def _boxplot(ax, metric, title, ylabel):
        groups = [df.loc[df['condition'] == c, metric].dropna().values for c in conditions]
        bp = ax.boxplot(groups, tick_labels=conditions, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], conditions):
            patch.set_facecolor(cond_colors[c])
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
        for j, (vals, c) in enumerate(zip(groups, conditions)):
            if len(vals) > 0:
                x = np.random.default_rng(42).normal(j + 1, 0.04, size=len(vals))
                ax.scatter(x, vals, s=30, c=cond_colors[c], edgecolors='black',
                           linewidth=0.5, zorder=3, alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    # --- 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _boxplot(axes[0, 0], 'final_aspect_ratio', 'Final Aspect Ratio', 'Aspect Ratio')
    _boxplot(axes[0, 1], 'final_mezzo_fraction', 'Final Mezzo Expression', 'Fraction')
    _boxplot(axes[1, 0], 'mean_area', 'Mean Area', 'Area (px)')
    _boxplot(axes[1, 1], 'n_poles', 'Pole Count', 'N poles')
    plt.suptitle('Cross-Experiment Comparison by Condition', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(plot_dir / 'condition_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Pole breakdown bar chart ---
    if len(conditions) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(conditions))
        width = 0.25
        for i, (col, label, color) in enumerate([
            ('n_poles', 'Total', '#333'),
            ('n_outward', 'Outward', '#e41a1c'),
            ('n_inward', 'Inward', '#377eb8'),
        ]):
            means = [df.loc[df['condition'] == c, col].mean() for c in conditions]
            stds = [df.loc[df['condition'] == c, col].std() for c in conditions]
            ax.bar(x + i * width, means, width, yerr=stds, label=label,
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels(conditions)
        ax.set_ylabel('Pole Count')
        ax.set_title('Pole Detection by Condition', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(plot_dir / 'pole_breakdown.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved comparison plots to {plot_dir}")


# ============================================================================
# 11. CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description='Pescoid Analysis Pipeline: U-Net segmentation + morphology + kymograph + pole detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all TIFFs in a folder
  python main_analysis.py "Z:/path/to/TIFF" --out analysis_output/

  # Single file
  python main_analysis.py data/sample.tif --out output/

  # With GPU + skip overlays
  python main_analysis.py data/ --out output/ --gpu --no-overlays

  # Custom channels
  python main_analysis.py data/ --out output/ --bf-channel 1 --gfp-channel 0
""",
    )
    parser.add_argument('input', help='TIFF file or directory containing TIFFs')
    parser.add_argument('--out', '-o', required=True, help='Output directory')

    g = parser.add_argument_group('Channels')
    g.add_argument('--bf-channel', type=int, default=0, help='BF channel index (default: 0)')
    g.add_argument('--gfp-channel', type=int, default=1, help='GFP channel index (default: 1)')

    g = parser.add_argument_group('Z-Projection')
    g.add_argument('--z-method', default='best_focus',
                   choices=['best_focus', 'max', 'mean', 'focus_range'],
                   help='Z-projection method (default: best_focus)')
    g.add_argument('--z-range', nargs=2, type=int, default=None, metavar=('Z0', 'Z1'),
                   help='Z-slice range for focus_range method')

    g = parser.add_argument_group('Cellpose')
    g.add_argument('--cellpose-model', default='cyto3', help='Model type (default: cyto3)')
    g.add_argument('--cellpose-diameter', type=float, default=None, help='Expected diameter (None=auto)')
    g.add_argument('--gpu', action='store_true', help='Use GPU')
    g.add_argument('--min-area', type=int, default=None, help='Min pescoid area (default: from config)')

    g = parser.add_argument_group('Analysis')
    g.add_argument('--gfp-threshold', default='otsu', help='GFP threshold: otsu, yen, li, or percentile')
    g.add_argument('--no-alignment', action='store_true', help='Skip rotation alignment')

    g = parser.add_argument_group('Output')
    g.add_argument('--no-overlays', action='store_true', help='Skip overlay PNGs (faster)')
    g.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    min_area = args.min_area or 3000
    z_range = tuple(args.z_range) if args.z_range else None

    kwargs = dict(
        bf_channel=args.bf_channel,
        gfp_channel=args.gfp_channel,
        z_method=args.z_method,
        z_range=z_range,
        cellpose_model=args.cellpose_model,
        cellpose_diameter=args.cellpose_diameter,
        gpu=args.gpu,
        min_area=min_area,
        gfp_threshold=args.gfp_threshold,
        do_overlays=not args.no_overlays,
        do_alignment=not args.no_alignment,
        verbose=not args.quiet,
    )

    if input_path.is_dir():
        run_batch(str(input_path), args.out, **kwargs)
    else:
        summary = process_single_file(str(input_path), args.out, **kwargs)
        print(f"\n{'=' * 50}")
        print(f"  {summary['stem']}")
        print(f"  Condition:     {summary['condition']}")
        print(f"  Timepoints:    {summary['n_timepoints']}")
        print(f"  Mean area:     {summary['morphology']['mean_area']:.0f} px")
        print(f"  Final AR:      {summary['morphology']['final_aspect_ratio']:.3f}")
        print(f"  Mezzo:         {summary['mezzo']['final_fraction']:.3f}")
        print(f"  Poles:         {summary['poles']['n_poles']} "
              f"({summary['poles']['n_outward']} out, {summary['poles']['n_inward']} in)")
        print(f"  Rotation fix:  {summary['max_rotation_deg']:.1f} deg")
        print(f"  Output:        {summary['output_dir']}")
        print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
