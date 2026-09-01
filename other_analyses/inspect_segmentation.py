"""
Inspect Segmentation Quality — Per-Pescoid Overlays + Annotation Export
========================================================================

For each experiment and each pescoid:
  1. Generates a 3-panel overlay montage (BF | BF+contour | GFP+contour)
     at multiple timepoints so you can visually judge segmentation quality.
  2. Exports BF images as 8-bit TIFs for manual annotation in Fiji.

Output:
  annotation_workspace/
    overlays/<experiment>/
      <pescoid>_overview.png       <- 3-row montage (early, mid, late)
    images/
      <NNN>_<experiment>_<pescoid>_<timepoint>.tif  <- BF for annotation
    masks/
      (empty — put your Fiji annotations here, same name + _mask.tif)

Usage:
  python inspect_segmentation.py
  python inspect_segmentation.py --experiment P_mezzo_ctrl
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from skimage import measure, exposure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"Z:\Megha_Kattimani\segmentation pipeline data\250212_mezzo_3-5hpf_activin_chiron")
ANALYSIS_DIR = Path("analysis_output")
WORKSPACE = Path("annotation_workspace")


def load_frame(tif_path, channel):
    """Load a single channel from a per-timepoint TIF, return float [0,1]."""
    data = tifffile.imread(str(tif_path))
    if data.ndim == 3 and data.shape[0] <= 4:
        raw = data[min(channel, data.shape[0] - 1)].astype(np.float64)
    elif data.ndim == 2:
        raw = data.astype(np.float64)
    else:
        raw = data.squeeze().astype(np.float64)
    lo, hi = np.percentile(raw, (0.5, 99.5))
    return np.clip((raw - lo) / (hi - lo), 0, 1) if hi > lo else raw / max(raw.max(), 1)


def generate_pescoid_overview(exp_name, sample_dir, mask_tif_path, overlay_dir,
                              bf_ch=0, gfp_ch=1):
    """
    Generate a multi-timepoint overview for one pescoid:
    5 rows (t=0, T/4, T/2, 3T/4, T-1) x 3 columns (BF | BF+mask | GFP+mask)
    """
    tifs = sorted(sample_dir.glob('*time*.[tT][iI][fF]'))
    if not tifs:
        tifs = sorted(sample_dir.glob('*.[tT][iI][fF]'))
    if not tifs:
        return
    T = len(tifs)

    # Load masks
    if mask_tif_path.exists():
        masks_all = tifffile.imread(str(mask_tif_path))
        if masks_all.ndim == 2:
            masks_all = masks_all[np.newaxis]
        masks_all = masks_all > 0
    else:
        masks_all = None

    # Select timepoints to show
    indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    indices = sorted(set(indices))  # deduplicate

    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, t_idx in enumerate(indices):
        bf = load_frame(tifs[t_idx], bf_ch)
        has_gfp = tifffile.imread(str(tifs[t_idx])).ndim == 3 and tifffile.imread(str(tifs[t_idx])).shape[0] >= 2
        gfp = load_frame(tifs[t_idx], gfp_ch) if has_gfp else np.zeros_like(bf)

        mask = masks_all[t_idx] if masks_all is not None and t_idx < len(masks_all) else np.zeros_like(bf, dtype=bool)
        area = int(mask.sum())

        # Panel 1: BF
        axes[row, 0].imshow(bf, cmap='gray')
        axes[row, 0].set_title(f't={t_idx}  BF', fontsize=10)
        axes[row, 0].set_ylabel(f't={t_idx}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # Panel 2: BF + mask contour
        axes[row, 1].imshow(bf, cmap='gray')
        if mask.any():
            for c in measure.find_contours(mask.astype(float), 0.5):
                axes[row, 1].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
        axes[row, 1].set_title(f'Mask  area={area:,} px', fontsize=10)
        axes[row, 1].axis('off')

        # Panel 3: GFP + mask contour
        axes[row, 2].imshow(gfp, cmap='Greens', vmin=0, vmax=max(gfp.max(), 0.01))
        if mask.any():
            for c in measure.find_contours(mask.astype(float), 0.5):
                axes[row, 2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.5)
        axes[row, 2].set_title('GFP (mezzo)', fontsize=10)
        axes[row, 2].axis('off')

    pescoid_id = re.search(r'(G\d+)', sample_dir.name)
    pid = pescoid_id.group(1) if pescoid_id else sample_dir.name
    plt.suptitle(f'{exp_name} / {pid} — Segmentation Overview',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    exp_overlay_dir = overlay_dir / exp_name
    exp_overlay_dir.mkdir(parents=True, exist_ok=True)
    save_path = exp_overlay_dir / f'{sample_dir.name}_overview.png'
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


def export_annotation_frames(exp_name, sample_dir, annotation_dir, bf_ch=0,
                             n_frames=3):
    """Export early, mid, late BF frames as 8-bit TIFs for Fiji annotation."""
    tifs = sorted(sample_dir.glob('*time*.[tT][iI][fF]'))
    if not tifs:
        tifs = sorted(sample_dir.glob('*.[tT][iI][fF]'))
    if not tifs:
        return []

    T = len(tifs)
    indices = [0, T // 2, T - 1]  # early, mid, late
    exported = []

    img_dir = annotation_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    for t_idx in indices:
        data = tifffile.imread(str(tifs[t_idx]))
        if data.ndim == 3 and data.shape[0] >= 2:
            bf = data[bf_ch].astype(np.float64)
        else:
            bf = data.astype(np.float64)

        lo, hi = np.percentile(bf, (0.5, 99.5))
        bf_norm = np.clip((bf - lo) / (hi - lo), 0, 1) if hi > lo else bf / max(bf.max(), 1)
        bf_uint8 = (bf_norm * 255).astype(np.uint8)

        pescoid_id = re.search(r'(G\d+)', sample_dir.name)
        pid = pescoid_id.group(1) if pescoid_id else sample_dir.name[-8:]
        name = f'{exp_name}_{pid}_t{t_idx:03d}'
        tifffile.imwrite(str(img_dir / f'{name}.tif'), bf_uint8)
        exported.append(name)

    return exported


def main():
    parser = argparse.ArgumentParser(description='Inspect segmentation + export for annotation')
    parser.add_argument('--experiment', default=None,
                        help='Process only this experiment folder (e.g. P_mezzo_ctrl)')
    parser.add_argument('--data-root', default=str(DATA_ROOT))
    parser.add_argument('--analysis-dir', default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    analysis_dir = Path(args.analysis_dir)
    workspace = WORKSPACE
    overlay_dir = workspace / 'overlays'
    workspace.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    (workspace / 'masks').mkdir(exist_ok=True)

    print("=" * 60)
    print("  SEGMENTATION INSPECTION + ANNOTATION EXPORT")
    print("=" * 60)

    all_exported = []
    total_pescoids = 0

    for exp_dir in sorted(data_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'result_segmentation':
            continue
        if args.experiment and exp_dir.name != args.experiment:
            continue

        exp_name = exp_dir.name
        samples = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        print(f"\n{'=' * 60}")
        print(f"  {exp_name} — {len(samples)} pescoids")
        print(f"{'=' * 60}")

        for sample_dir in samples:
            pescoid_id = re.search(r'(G\d+)', sample_dir.name)
            pid = pescoid_id.group(1) if pescoid_id else sample_dir.name
            print(f"  {pid}", end='')

            # Find mask file
            mask_path = analysis_dir / sample_dir.name / 'masks' / f'{sample_dir.name}_masks.tif'

            # Generate overlay
            try:
                generate_pescoid_overview(exp_name, sample_dir, mask_path, overlay_dir)
                print(f" -> overlay", end='')
            except Exception as e:
                print(f" -> overlay FAILED ({e})", end='')

            # Export annotation frames (first sample per experiment only, to keep manageable)
            if sample_dir == samples[0]:
                exported = export_annotation_frames(exp_name, sample_dir, workspace)
                all_exported.extend(exported)
                print(f" + {len(exported)} annotation frames", end='')

            print()
            total_pescoids += 1

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"{'=' * 60}")
    print(f"\n  Pescoids processed: {total_pescoids}")
    print(f"  Overlays: {overlay_dir}/")
    print(f"  Annotation frames: {workspace / 'images'}/  ({len(all_exported)} TIFs)")
    print(f"  Put your Fiji masks in: {workspace / 'masks'}/")
    print(f"\n  --- ANNOTATION INSTRUCTIONS ---")
    print(f"  1. Open each TIF in {workspace / 'images'}/ with Fiji")
    print(f"  2. Draw the pescoid outline (Freehand Selection tool)")
    print(f"  3. Edit > Selection > Create Mask  (or threshold + fill)")
    print(f"  4. Save as: {workspace / 'masks'}/<same_name>_mask.tif")
    print(f"     (binary: 0=background, 255=foreground)")
    print(f"  5. Aim for 10-15 annotated frames across conditions")


if __name__ == '__main__':
    main()
