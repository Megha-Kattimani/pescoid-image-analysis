"""
Unified Cellpose Segmentation + Analysis Pipeline for Pescoid Embryos
=====================================================================

End-to-end pipeline that segments zebrafish pescoid embryos using Cellpose
on brightfield images, then computes morphological and fluorescence analysis:

  - Aspect ratio over time (major/minor axis from regionprops)
  - Mesendodermal marker (mezzo-GFP) expression fraction within BF mask
  - Perimeter-change kymograph (radial displacement over time)
  - Pole detection (outward/inward poles from kymograph activity)

Input:
  5D TIFF stacks (T, Z, C, Y, X) with BF (ch1) and GFP (ch0).
  Z-projected to 2D before segmentation.

Usage:
  # Single file
  python scripts/cellpose_pipeline.py data/sample.tif --out output/

  # Batch (all TIFFs in a directory)
  python scripts/cellpose_pipeline.py data/ --out output/ --batch

  # With GPU and custom parameters
  python scripts/cellpose_pipeline.py data/ --out output/ --batch \\
      --gpu --cellpose-model cyto3 --gfp-threshold otsu

Prerequisites:
  pip install cellpose
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
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure, exposure, filters as skf

# ---------------------------------------------------------------------------
# sys.path setup for cross-module imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'analysis'))
sys.path.insert(0, str(_SCRIPT_DIR))

from bf_segmentation import load_tiff, extract_channel, z_project
from ml_segmentation import segment_cellpose
from compute_aspect_ratio_and_kymograph import (
    measure_props, extract_contour, mask_centroid, radii_along_angles
)
from pole_detection import detect_and_analyze_poles, visualize_pole_detection
import config


# ============================================================================
# HELPER: 2D Fluorescence Fraction
# ============================================================================

def compute_fluorescence_fraction_2d(
    bf_mask_2d: np.ndarray,
    gfp_2d: np.ndarray,
    threshold_method: str = 'otsu',
) -> dict:
    """
    Compute fraction of GFP/mezzo signal within the BF mask for a single 2D frame.

    Parameters
    ----------
    bf_mask_2d : (Y, X) bool -- BF segmentation mask
    gfp_2d     : (Y, X) float -- Z-projected GFP image
    threshold_method : 'otsu', 'yen', 'li', or a numeric percentile (e.g. '75')

    Returns
    -------
    dict with mezzo_fraction, mezzo_intensity_mean/max/std, mezzo_threshold
    """
    empty = {
        'mezzo_fraction': 0.0,
        'mezzo_intensity_mean': 0.0,
        'mezzo_intensity_max': 0.0,
        'mezzo_intensity_std': 0.0,
        'mezzo_threshold': 0.0,
    }

    if bf_mask_2d is None or gfp_2d is None:
        return empty

    gfp_in_mask = gfp_2d[bf_mask_2d]
    if gfp_in_mask.size == 0:
        return empty

    if threshold_method == 'otsu':
        thresh = skf.threshold_otsu(gfp_in_mask)
    elif threshold_method == 'yen':
        thresh = skf.threshold_yen(gfp_in_mask)
    elif threshold_method == 'li':
        thresh = skf.threshold_li(gfp_in_mask)
    else:
        try:
            thresh = np.percentile(gfp_in_mask, float(threshold_method))
        except (ValueError, TypeError):
            thresh = skf.threshold_otsu(gfp_in_mask)

    positive_count = int((gfp_in_mask > thresh).sum())
    fraction = positive_count / gfp_in_mask.size

    return {
        'mezzo_fraction': float(fraction),
        'mezzo_intensity_mean': float(gfp_in_mask.mean()),
        'mezzo_intensity_max': float(gfp_in_mask.max()),
        'mezzo_intensity_std': float(gfp_in_mask.std()),
        'mezzo_threshold': float(thresh),
    }


# ============================================================================
# HELPER: Infer experimental condition from file path
# ============================================================================

def infer_condition(filepath: str) -> str:
    """Infer experimental condition (ctrl/Act/Chi/Act-Chi) from file path."""
    s = str(filepath).lower().replace('\\', '/').replace('-', '_')
    # Check compound condition first
    if 'act_chi' in s or 'chir_act' in s or 'activin_chiron' in s or 'chiron_activin' in s:
        return 'Act-Chi'
    if 'act' in s or 'activin' in s:
        return 'Act'
    if 'chi' in s or 'chiron' in s or 'chir' in s:
        return 'Chi'
    if 'ctrl' in s or 'control' in s:
        return 'ctrl'
    return 'unknown'


# ============================================================================
# HELPER: Build kymograph from time-series of masks
# ============================================================================

def build_kymograph(
    masks: np.ndarray,
    n_perimeter_points: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build a normalised perimeter-change kymograph from a time-series of masks.

    Parameters
    ----------
    masks : (T, Y, X) bool array
    n_perimeter_points : number of perimeter samples

    Returns
    -------
    kymo_2d : (n_perimeter_points, T) normalised kymograph in [-1, 1]
    kymo_raw : (n_perimeter_points, T) raw delta-radii (unnormalised)
    """
    T = masks.shape[0]
    if T == 0:
        return np.zeros((n_perimeter_points, 0)), None

    # Establish baseline from first valid mask
    baseline_radii = None
    baseline_angles = None
    radius_matrix = []

    for t in range(T):
        mask = masks[t]
        cy, cx = mask_centroid(mask)

        if baseline_radii is None:
            contour = extract_contour(mask, n_points=n_perimeter_points)
            by = contour[:, 0] - cy
            bx = contour[:, 1] - cx
            angles = np.arctan2(by, bx)
            order = np.argsort(angles)
            baseline_angles = angles[order]
            baseline_radii = radii_along_angles(mask, cy, cx, baseline_angles)

        radii_t = radii_along_angles(mask, cy, cx, baseline_angles)
        delta_r = radii_t - baseline_radii
        radius_matrix.append(delta_r)

    kymo_raw = np.array(radius_matrix).T  # (n_perimeter_points, T)
    max_abs = float(np.max(np.abs(kymo_raw))) if kymo_raw.size else 0.0
    if max_abs > 0:
        kymo_2d = np.clip(kymo_raw / max_abs, -1.0, 1.0)
    else:
        kymo_2d = kymo_raw.copy()

    return kymo_2d, kymo_raw


# ============================================================================
# HELPER: Generate overlay images
# ============================================================================

def save_overlays(
    stack: np.ndarray,
    masks: np.ndarray,
    bf_channel: int,
    gfp_channel: int,
    z_projection: str,
    z_range: Optional[Tuple[int, int]],
    output_dir: Path,
    stem: str,
):
    """Save per-frame BF + contour + GFP overlay PNGs."""
    overlay_dir = output_dir / 'overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    T, Z, C, Y, X = stack.shape

    for t in range(T):
        bf_z = stack[t, :, bf_channel, :, :]
        bf_2d = z_project(bf_z, method=z_projection, z_range=z_range) if Z > 1 else bf_z[0].astype(float)
        bf_disp = exposure.rescale_intensity(bf_2d.astype(float), out_range=(0, 1))

        has_gfp = gfp_channel < C
        if has_gfp:
            gfp_z = stack[t, :, gfp_channel, :, :]
            gfp_2d = z_project(gfp_z, method=z_projection, z_range=z_range) if Z > 1 else gfp_z[0].astype(float)
            gfp_disp = exposure.rescale_intensity(gfp_2d.astype(float), out_range=(0, 1))

        n_panels = 3 if has_gfp else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

        # Panel 1: BF
        axes[0].imshow(bf_disp, cmap='gray')
        axes[0].set_title(f'BF  t={t}', fontsize=9)
        axes[0].axis('off')

        # Panel 2: BF + mask contour
        axes[1].imshow(bf_disp, cmap='gray')
        for c in measure.find_contours(masks[t].astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=1.5)
        area = int(masks[t].sum())
        axes[1].set_title(f'Segmentation  area={area:,} px', fontsize=9)
        axes[1].axis('off')

        # Panel 3: GFP + contour
        if has_gfp:
            axes[2].imshow(gfp_disp, cmap='Greens')
            for c in measure.find_contours(masks[t].astype(float), 0.5):
                axes[2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.2)
            axes[2].set_title(f'GFP (mezzo)  t={t}', fontsize=9)
            axes[2].axis('off')

        plt.suptitle(f'{stem}  |  Cellpose', fontsize=10, y=1.01)
        plt.tight_layout()
        plt.savefig(str(overlay_dir / f'{stem}_t{t:04d}_overlay.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# CORE: Process a single TIFF file
# ============================================================================

def process_single_file(
    tiff_path: str,
    output_dir: str,
    bf_channel: int = 1,
    gfp_channel: int = 0,
    z_projection: str = 'best_focus',
    z_range: Optional[Tuple[int, int]] = None,
    cellpose_model: str = 'cyto3',
    cellpose_diameter: Optional[float] = None,
    gpu: bool = False,
    min_area: int = 50000,
    gfp_threshold_method: str = 'otsu',
    n_perimeter_points: int = 200,
    pole_params: Optional[dict] = None,
    do_overlays: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the full Cellpose segmentation + analysis pipeline on a single TIFF.

    Returns a summary dict with all computed metrics.
    """
    tiff_path = Path(tiff_path)
    stem = tiff_path.stem
    out = Path(output_dir) / stem
    mask_dir = out / 'masks'
    analysis_dir = out / 'analysis'
    mask_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # 1. Load TIFF
    # ------------------------------------------------------------------
    log(f"Loading: {tiff_path.name}")
    stack, file_meta = load_tiff(str(tiff_path))
    T, Z, C, Y, X = stack.shape
    log(f"  Shape: T={T}, Z={Z}, C={C}, Y={Y}, X={X}")

    bf_ch = bf_channel if bf_channel < C else 0
    gfp_ch = gfp_channel if gfp_channel < C else 0
    log(f"  BF channel: {bf_ch}, GFP channel: {gfp_ch}")

    # ------------------------------------------------------------------
    # 2. Per-timepoint segmentation + measurements
    # ------------------------------------------------------------------
    masks = np.zeros((T, Y, X), dtype=bool)
    aspect_data = []
    mezzo_data = []
    failed_frames = []

    for t in range(T):
        # BF segmentation
        bf_z = stack[t, :, bf_ch, :, :]
        bf_2d = z_project(bf_z, method=z_projection, z_range=z_range) if Z > 1 else bf_z[0].astype(float)

        try:
            mask_2d = segment_cellpose(
                bf_2d,
                model_type=cellpose_model,
                diameter=cellpose_diameter,
                gpu=gpu,
                min_area=min_area,
            )
        except Exception as e:
            warnings.warn(f"Frame {t}: Cellpose failed ({e})")
            mask_2d = np.zeros((Y, X), dtype=bool)

        if mask_2d.sum() == 0:
            failed_frames.append(t)
            log(f"  [t={t:03d}] WARNING: empty mask")
        else:
            log(f"  [t={t:03d}] area={mask_2d.sum():,} px")

        masks[t] = mask_2d

        # Aspect ratio
        perimeter, aspect, major, minor = measure_props(mask_2d)
        aspect_data.append({
            'time_index': t,
            'perimeter': perimeter,
            'aspect_ratio': aspect,
            'major_axis_length': major,
            'minor_axis_length': minor,
        })

        # GFP / mezzo expression
        gfp_z = stack[t, :, gfp_ch, :, :]
        gfp_2d = z_project(gfp_z, method=z_projection, z_range=z_range) if Z > 1 else gfp_z[0].astype(float)
        fluo = compute_fluorescence_fraction_2d(mask_2d, gfp_2d, gfp_threshold_method)
        fluo['time_index'] = t
        mezzo_data.append(fluo)

    # ------------------------------------------------------------------
    # 3. Fill failed frames from nearest good frame
    # ------------------------------------------------------------------
    if failed_frames:
        good = [t for t in range(T) if t not in failed_frames]
        if good:
            for t in failed_frames:
                nearest = min(good, key=lambda g: abs(g - t))
                masks[t] = masks[nearest].copy()
                log(f"  [t={t:03d}] filled from t={nearest}")

    # ------------------------------------------------------------------
    # 4. Save mask TIFF
    # ------------------------------------------------------------------
    mask_path = mask_dir / f'{stem}_finalMask.tif'
    tifffile.imwrite(str(mask_path), (masks.astype(np.uint8) * 255))
    log(f"  Saved masks: {mask_path.name}")

    # ------------------------------------------------------------------
    # 5. Save aspect ratio CSV
    # ------------------------------------------------------------------
    csv_path = analysis_dir / 'aspect_ratio_over_time.csv'
    with open(str(csv_path), 'w') as f:
        f.write('time_index,perimeter,aspect_ratio,major_axis_length,minor_axis_length\n')
        for row in aspect_data:
            f.write(f"{row['time_index']},{row['perimeter']},{row['aspect_ratio']},"
                    f"{row['major_axis_length']},{row['minor_axis_length']}\n")

    # ------------------------------------------------------------------
    # 6. Save mezzo expression CSV
    # ------------------------------------------------------------------
    mezzo_csv = analysis_dir / 'mezzo_expression_over_time.csv'
    with open(str(mezzo_csv), 'w') as f:
        f.write('time_index,mezzo_fraction,mezzo_intensity_mean,mezzo_intensity_max,'
                'mezzo_intensity_std,mezzo_threshold\n')
        for row in mezzo_data:
            f.write(f"{row['time_index']},{row['mezzo_fraction']},"
                    f"{row['mezzo_intensity_mean']},{row['mezzo_intensity_max']},"
                    f"{row['mezzo_intensity_std']},{row['mezzo_threshold']}\n")

    # ------------------------------------------------------------------
    # 7. Build kymograph
    # ------------------------------------------------------------------
    kymo_2d, kymo_raw = build_kymograph(masks, n_perimeter_points=n_perimeter_points)

    # Save kymograph matrix
    np.save(str(analysis_dir / 'kymograph_matrix.npy'), kymo_raw if kymo_raw is not None else kymo_2d)

    # Plot kymograph
    if kymo_2d.size > 0 and kymo_2d.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(kymo_2d, aspect='auto', cmap='seismic', vmin=-1, vmax=1, origin='lower')
        ax.set_xlabel('Time [index]', fontsize=12)
        ax.set_ylabel('Normalised Perimeter', fontsize=12)
        n_ticks = 5
        tick_idx = np.linspace(0, n_perimeter_points - 1, n_ticks, dtype=int)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(0, 1, n_ticks)])
        ax.set_title('Radius change (red=outward, blue=inward)', fontsize=11)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Delta radius (normalised)', fontsize=10)
        plt.tight_layout()
        plt.savefig(str(analysis_dir / 'kymograph.png'), dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Pole detection
    # ------------------------------------------------------------------
    pole_results = None
    pp = pole_params or {}
    if kymo_2d.size > 0 and kymo_2d.shape[1] > 1:
        try:
            pole_results = detect_and_analyze_poles(
                kymo_2d,
                smooth_sigma=pp.get('smooth_sigma', 2),
                prominence=pp.get('prominence', 0.15),
                min_distance=pp.get('min_distance', 20),
                classify=True,
            )

            if pole_results['n_poles'] > 0:
                visualize_pole_detection(
                    pole_results['activity_profile'],
                    pole_results['pole_indices'],
                    outward_poles=pole_results.get('outward_indices'),
                    inward_poles=pole_results.get('inward_indices'),
                    save_path=str(analysis_dir / 'pole_detection.png'),
                )

            # Pole counts CSV
            with open(str(analysis_dir / 'pole_counts.csv'), 'w') as f:
                f.write('metric,value\n')
                f.write(f"total_poles,{pole_results['n_poles']}\n")
                f.write(f"outward_poles,{pole_results.get('n_outward', 0)}\n")
                f.write(f"inward_poles,{pole_results.get('n_inward', 0)}\n")

            log(f"  Poles: {pole_results['n_poles']} total "
                f"({pole_results.get('n_outward', 0)} out, "
                f"{pole_results.get('n_inward', 0)} in)")
        except Exception as e:
            warnings.warn(f"Pole detection failed: {e}")

    # ------------------------------------------------------------------
    # 9. Overlays
    # ------------------------------------------------------------------
    if do_overlays:
        log(f"  Generating {T} overlay images...")
        save_overlays(stack, masks, bf_ch, gfp_ch, z_projection, z_range, out, stem)

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    condition = infer_condition(str(tiff_path))
    areas = [int(masks[t].sum()) for t in range(T)]
    aspects = [d['aspect_ratio'] for d in aspect_data]
    mezzo_fracs = [d['mezzo_fraction'] for d in mezzo_data]

    summary = {
        'file': str(tiff_path),
        'stem': stem,
        'condition': condition,
        'dimensions': {'T': T, 'Z': Z, 'C': C, 'Y': Y, 'X': X},
        'bf_channel': bf_ch,
        'gfp_channel': gfp_ch,
        'cellpose_model': cellpose_model,
        'z_projection': z_projection,
        'failed_frames': failed_frames,
        'mask_areas': areas,
        'mean_area': float(np.mean(areas)) if areas else 0.0,
        'aspect_ratio': {
            'values': aspects,
            'mean': float(np.mean(aspects)) if aspects else 0.0,
            'final': aspects[-1] if aspects else 0.0,
        },
        'mezzo_expression': {
            'fractions': mezzo_fracs,
            'mean_fraction': float(np.mean(mezzo_fracs)) if mezzo_fracs else 0.0,
            'final_fraction': mezzo_fracs[-1] if mezzo_fracs else 0.0,
        },
        'pole_detection': {
            'n_poles': pole_results['n_poles'] if pole_results else 0,
            'n_outward': pole_results.get('n_outward', 0) if pole_results else 0,
            'n_inward': pole_results.get('n_inward', 0) if pole_results else 0,
            'pole_indices': (pole_results['pole_indices'].tolist()
                            if pole_results and pole_results['n_poles'] > 0 else []),
        },
        'kymograph_shape': list(kymo_2d.shape) if kymo_2d.size > 0 else [0, 0],
        'output_dir': str(out),
    }

    with open(str(out / 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    log(f"  Done: {stem} -> {out}")
    return summary


# ============================================================================
# BATCH: Process all TIFFs in a directory
# ============================================================================

def run_batch(
    input_dir: str,
    output_dir: str,
    **pipeline_kwargs,
) -> Optional['pd.DataFrame']:
    """
    Process all TIFF files in input_dir. Returns a pandas DataFrame summary.
    """
    import pandas as pd

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Gather TIFF files (non-recursive + one level of subdirectories)
    tif_files = sorted(input_path.glob('*.tif')) + sorted(input_path.glob('*.tiff'))
    for sub in sorted(input_path.iterdir()):
        if sub.is_dir():
            tif_files += sorted(sub.glob('*.tif')) + sorted(sub.glob('*.tiff'))
    # Deduplicate and sort
    tif_files = sorted(set(tif_files))

    if not tif_files:
        print(f"No TIFF files found in {input_dir}")
        return None

    print(f"Batch: {len(tif_files)} TIFF files found in {input_dir}")
    print('=' * 60)

    results = []
    failures = []
    t_start = time.time()

    for i, tif_path in enumerate(tif_files, 1):
        print(f"\n[{i}/{len(tif_files)}] {tif_path.name}")
        print('-' * 40)
        try:
            summary = process_single_file(
                str(tif_path),
                str(output_path),
                **pipeline_kwargs,
            )
            results.append(summary)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failures.append({'file': str(tif_path), 'error': str(e)})

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Batch complete: {len(results)}/{len(tif_files)} succeeded in {elapsed:.1f}s")

    if failures:
        print(f"  {len(failures)} failed:")
        fail_path = output_path / 'failed_files.txt'
        with open(str(fail_path), 'w') as f:
            for fail in failures:
                line = f"{fail['file']}: {fail['error']}"
                print(f"    {line}")
                f.write(line + '\n')

    if not results:
        return None

    # Build summary DataFrame
    rows = []
    for s in results:
        rows.append({
            'file': s['stem'],
            'condition': s['condition'],
            'n_timepoints': s['dimensions']['T'],
            'mean_area': s['mean_area'],
            'mean_aspect_ratio': s['aspect_ratio']['mean'],
            'final_aspect_ratio': s['aspect_ratio']['final'],
            'mean_mezzo_fraction': s['mezzo_expression']['mean_fraction'],
            'final_mezzo_fraction': s['mezzo_expression']['final_fraction'],
            'n_poles': s['pole_detection']['n_poles'],
            'n_outward_poles': s['pole_detection']['n_outward'],
            'n_inward_poles': s['pole_detection']['n_inward'],
            'failed_frames': len(s['failed_frames']),
        })
    df = pd.DataFrame(rows)

    csv_path = output_path / 'batch_summary.csv'
    df.to_csv(str(csv_path), index=False)
    print(f"\nSaved: {csv_path}")

    # Generate comparison plots
    generate_comparison_plots(df, str(output_path))

    return df


# ============================================================================
# COMPARISON PLOTS across conditions
# ============================================================================

def generate_comparison_plots(df: 'pd.DataFrame', output_dir: str):
    """Generate box plots comparing metrics across experimental conditions."""
    plot_dir = Path(output_dir) / 'comparison_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(df['condition'].unique())
    if len(conditions) < 2:
        print("  Skipping comparison plots (need >= 2 conditions)")
        return

    # Color map from config
    cond_colors = {}
    for cond in conditions:
        cfg = config.CONDITIONS.get(cond)
        cond_colors[cond] = cfg['color'] if cfg else 'gray'

    # -- Plot 1: Aspect Ratio --
    fig, ax = plt.subplots(figsize=(8, 5))
    data_groups = [df.loc[df['condition'] == c, 'mean_aspect_ratio'].values for c in conditions]
    bp = ax.boxplot(data_groups, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(cond_colors[cond])
        patch.set_alpha(0.6)
    ax.set_ylabel('Mean Aspect Ratio')
    ax.set_title('Aspect Ratio by Condition')
    plt.tight_layout()
    plt.savefig(str(plot_dir / 'aspect_ratio_by_condition.png'), dpi=150)
    plt.close(fig)

    # -- Plot 2: Mezzo Fraction --
    fig, ax = plt.subplots(figsize=(8, 5))
    data_groups = [df.loc[df['condition'] == c, 'final_mezzo_fraction'].values for c in conditions]
    bp = ax.boxplot(data_groups, labels=conditions, patch_artist=True)
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(cond_colors[cond])
        patch.set_alpha(0.6)
    ax.set_ylabel('Final Mezzo Fraction')
    ax.set_title('Mesendodermal Marker Expression by Condition')
    plt.tight_layout()
    plt.savefig(str(plot_dir / 'mezzo_fraction_by_condition.png'), dpi=150)
    plt.close(fig)

    # -- Plot 3: Pole Counts --
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    width = 0.25
    for i, (col, label) in enumerate([
        ('n_poles', 'Total'),
        ('n_outward_poles', 'Outward'),
        ('n_inward_poles', 'Inward'),
    ]):
        means = [df.loc[df['condition'] == c, col].mean() for c in conditions]
        stds = [df.loc[df['condition'] == c, col].std() for c in conditions]
        ax.bar(x + i * width, means, width, yerr=stds, label=label, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Pole Count')
    ax.set_title('Pole Detection by Condition')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(plot_dir / 'pole_count_by_condition.png'), dpi=150)
    plt.close(fig)

    print(f"  Saved comparison plots to {plot_dir}")


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Cellpose segmentation + analysis pipeline for pescoid embryos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python scripts/cellpose_pipeline.py data/sample.tif --out output/

  # Batch (all TIFFs in directory)
  python scripts/cellpose_pipeline.py data/ --out output/ --batch

  # GPU + custom parameters
  python scripts/cellpose_pipeline.py data/ --out output/ --batch \\
      --gpu --cellpose-model cyto3 --gfp-threshold otsu --no-overlays
""",
    )

    parser.add_argument('input', help='TIFF file or directory (batch mode)')
    parser.add_argument('--out', '-o', required=True, help='Output directory')

    # Channels
    parser.add_argument('--bf-channel', type=int, default=1,
                        help='0-indexed BF channel (default: 1)')
    parser.add_argument('--gfp-channel', type=int, default=0,
                        help='0-indexed GFP/mezzo channel (default: 0)')

    # Z-projection
    parser.add_argument('--z-projection', default='best_focus',
                        choices=['best_focus', 'max', 'mean', 'focus_range'],
                        help='Z-projection method (default: best_focus)')
    parser.add_argument('--z-range', nargs=2, type=int, default=None,
                        metavar=('Z0', 'Z1'),
                        help='Z-slice range for focus_range method')

    # Cellpose
    parser.add_argument('--cellpose-model', default='cyto3',
                        help='Cellpose model type (default: cyto3)')
    parser.add_argument('--cellpose-diameter', type=float, default=None,
                        help='Expected organoid diameter in px (None=auto)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for Cellpose')

    # Segmentation
    parser.add_argument('--min-area', type=int, default=None,
                        help='Min pescoid area in pixels (default: from config.py)')

    # GFP threshold
    parser.add_argument('--gfp-threshold', default='otsu',
                        help='GFP threshold: otsu, yen, li, or percentile value')

    # Batch
    parser.add_argument('--batch', action='store_true',
                        help='Process all TIFs in input directory')

    # Output control
    parser.add_argument('--no-overlays', action='store_true',
                        help='Skip overlay PNGs (faster)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    min_area = args.min_area or config.BF_SEGMENTATION['min_area']
    z_range = tuple(args.z_range) if args.z_range else None

    kwargs = dict(
        bf_channel=args.bf_channel,
        gfp_channel=args.gfp_channel,
        z_projection=args.z_projection,
        z_range=z_range,
        cellpose_model=args.cellpose_model,
        cellpose_diameter=args.cellpose_diameter,
        gpu=args.gpu,
        min_area=min_area,
        gfp_threshold_method=args.gfp_threshold,
        do_overlays=not args.no_overlays,
        verbose=not args.quiet,
    )

    if args.batch or input_path.is_dir():
        run_batch(str(input_path), args.out, **kwargs)
    else:
        result = process_single_file(str(input_path), args.out, **kwargs)
        print(f"\nResults:")
        print(f"  Timepoints:          {result['dimensions']['T']}")
        print(f"  Mean area:           {result['mean_area']:.0f} px")
        print(f"  Mean aspect ratio:   {result['aspect_ratio']['mean']:.3f}")
        print(f"  Final aspect ratio:  {result['aspect_ratio']['final']:.3f}")
        print(f"  Mean mezzo fraction: {result['mezzo_expression']['mean_fraction']:.3f}")
        print(f"  Final mezzo fraction:{result['mezzo_expression']['final_fraction']:.3f}")
        print(f"  Poles detected:      {result['pole_detection']['n_poles']}")
        print(f"  Output:              {result['output_dir']}")


if __name__ == '__main__':
    main()
