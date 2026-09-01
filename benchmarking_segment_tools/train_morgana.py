"""
Train Morgana on Pescoid JPG Images and Generate Masks
======================================================

Complete pipeline:
  1. Extract BF channel from merged BF+GFP JPGs
  2. Create approximate GT masks via thresholding (bootstrap)
  3. Train Morgana's pixel classifier (LogisticRegression on multi-scale features)
  4. Predict on ALL images
  5. Post-process with Morgana's smooth_mask + watershed
  6. Save final binary masks

Input:
  Z:/Megha_Kattimani/Imaging/Zeiss/Benchmarking_Segmentation tools/JPG/
    control/{initial,elongated}/*.jpg
    treated/{initial,elongation}/*.jpg

Output:
  benchmarking_segment_tools/output/morgana/
    model/              <- trained classifier, scaler, params
    masks/              <- final binary masks (TIF, 0/255)
    overlays/           <- visual overlays (BF + contour + GFP)
    training_overview/  <- shows GT vs prediction for training images

Usage:
  python benchmarking_segment_tools/train_morgana.py
  python benchmarking_segment_tools/train_morgana.py --refine-gt   # launch GUI to fix GT masks
"""

import os
import sys
import json
import argparse
import warnings
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage import io as skio, measure, morphology, filters, exposure
from skimage.util import img_as_float
from scipy import ndimage as ndi
import tifffile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'MOrgAna'))

from morgana.MLModel import train as morgana_train
from morgana.MLModel import predict as morgana_predict
from morgana.MLModel import io as morgana_io
from morgana.ImageTools.segmentation.segment import smooth_mask
from morgana.DatasetTools import io as morgana_dt_io

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INPUT = r"Z:\Megha_Kattimani\Imaging\Zeiss\Benchmarking_Segmentation tools\JPG"
DEFAULT_OUTPUT = str(_SCRIPT_DIR / 'output' / 'morgana')

# Training hyperparameters
SIGMAS = [1.0, 5.0, 15.0]       # multi-scale Gaussian sigmas
DOWN_SHAPE = 0.5                  # downscale factor for training (speed)
EDGE_SIZE = 5                     # edge dilation for weighting
FRACTION = 0.25                   # fraction of pixels per image for training
BIAS = 0.4                        # bias toward foreground pixels
FEATURE_MODE = 'ilastik'          # feature type


# ============================================================================
# STEP 0: Discover and load images
# ============================================================================

def discover_images(input_dir: str) -> List[dict]:
    """Find all JPGs and return metadata."""
    base = Path(input_dir)
    images = []
    folder_map = {
        ('control', 'initial'): ('control', 'initial'),
        ('control', 'elongated'): ('control', 'elongated'),
        ('treated', 'initial'): ('treated', 'initial'),
        ('treated', 'elongation'): ('treated', 'elongated'),
    }
    for (cond_folder, tp_folder), (condition, timepoint) in folder_map.items():
        folder = base / cond_folder / tp_folder
        if not folder.exists():
            continue
        for f in sorted(folder.glob('*.jpg')):
            name = f.stem
            sample_id = name
            for part in name.split('#'):
                if part and part[0].isdigit():
                    sample_id = part.split('.')[0].split('_')[0]
                    break
            images.append({
                'path': f,
                'condition': condition,
                'timepoint': timepoint,
                'filename': f.name,
                'sample_id': sample_id,
            })
    return images


def split_channels(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract BF (red channel) and GFP (green - red) from merged composite."""
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    bf = r / 255.0
    gfp_raw = np.clip(g - r, 0, 255)
    gfp_max = gfp_raw.max()
    gfp = gfp_raw / gfp_max if gfp_max > 0 else gfp_raw
    return bf, gfp


# ============================================================================
# STEP 1: Create approximate ground truth masks
# ============================================================================

def create_gt_mask(bf_gray: np.ndarray) -> np.ndarray:
    """
    Create an approximate GT mask using multi-step thresholding.
    This serves as a bootstrap for Morgana training.
    The user can refine these later if needed.
    """
    # Invert: pescoid is darker than background
    inverted = 1.0 - bf_gray

    # Multi-scale approach: combine coarse and fine thresholds
    smooth_coarse = filters.gaussian(inverted, sigma=10)
    smooth_fine = filters.gaussian(inverted, sigma=3)

    thresh_coarse = filters.threshold_otsu(smooth_coarse)
    thresh_fine = filters.threshold_otsu(smooth_fine)

    mask_coarse = smooth_coarse > thresh_coarse
    mask_fine = smooth_fine > thresh_fine

    # Combine: coarse gives the overall shape, fine gives edges
    mask = np.logical_or(mask_coarse, mask_fine)
    mask = ndi.binary_fill_holes(mask)

    # Morphological cleanup
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.binary_opening(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)

    # Keep only the largest component
    labeled, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(bf_gray, dtype=np.uint8)
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    mask = (labeled == largest)

    # Smooth boundary
    mask = morphology.binary_erosion(mask, morphology.disk(2))
    mask = morphology.binary_dilation(mask, morphology.disk(2))

    return mask.astype(np.uint8)


# ============================================================================
# STEP 2: Set up training data in Morgana format
# ============================================================================

def setup_training_data(images: List[dict], model_dir: Path,
                        n_train: int = 6) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Select representative images for training, create GT masks,
    save in Morgana's expected format.

    Selects images from different conditions/timepoints for diversity.
    """
    training_dir = model_dir / 'trainingset'
    training_dir.mkdir(parents=True, exist_ok=True)

    # Select diverse training images: pick from each condition/timepoint combo
    groups = {}
    for img_info in images:
        key = (img_info['condition'], img_info['timepoint'])
        groups.setdefault(key, []).append(img_info)

    selected = []
    # Pick ~equal from each group
    per_group = max(1, n_train // len(groups))
    for key, group_imgs in groups.items():
        selected.extend(group_imgs[:per_group])
    # Fill remainder
    remaining = [im for im in images if im not in selected]
    while len(selected) < n_train and remaining:
        selected.append(remaining.pop(0))

    print(f"\nSelected {len(selected)} training images:")
    train_images = []
    train_gts = []

    for img_info in selected:
        img_rgb = skio.imread(str(img_info['path']))
        bf, _ = split_channels(img_rgb)

        # Convert to uint8 for Morgana (expects integer images)
        bf_uint8 = (bf * 255).astype(np.uint8)

        # Create GT mask
        gt = create_gt_mask(bf)

        # Save in Morgana format
        name = f"{img_info['sample_id']}_{img_info['condition']}_{img_info['timepoint']}"
        img_path = training_dir / f"{name}.tif"
        gt_path = training_dir / f"{name}_GT.tif"

        tifffile.imwrite(str(img_path), bf_uint8)
        tifffile.imwrite(str(gt_path), (gt * 255).astype(np.uint8))

        train_images.append(bf_uint8)
        train_gts.append(gt)

        print(f"  {name}: {img_info['condition']}/{img_info['timepoint']} "
              f"(GT area={gt.sum():,} px)")

    return train_images, train_gts


# ============================================================================
# STEP 3: Train Morgana classifier
# ============================================================================

def train_model(train_images: List[np.ndarray], train_gts: List[np.ndarray],
                model_dir: Path):
    """Train Morgana's LogisticRegression pixel classifier."""
    print(f"\n{'='*60}")
    print("TRAINING MORGANA CLASSIFIER")
    print(f"{'='*60}")
    print(f"  Images: {len(train_images)}")
    print(f"  Sigmas: {SIGMAS}")
    print(f"  Downscale: {DOWN_SHAPE}")
    print(f"  Feature mode: {FEATURE_MODE}")
    print(f"  Pixel fraction: {FRACTION}")
    print(f"  Foreground bias: {BIAS}")
    print()

    t0 = time.time()

    # Generate training set (features + labels)
    X_train, Y_train, weight_train, scaler = morgana_train.generate_training_set(
        train_images,
        [gt.astype(np.uint8) for gt in train_gts],
        sigmas=SIGMAS,
        down_shape=DOWN_SHAPE,
        edge_size=EDGE_SIZE,
        fraction=FRACTION,
        bias=BIAS,
        feature_mode=FEATURE_MODE,
    )

    print(f"\n  Training set shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Class distribution: {np.bincount(Y_train.astype(int))}")

    # Train classifier
    classifier = morgana_train.train_classifier(X_train, Y_train, weight_train, deep=False)

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")

    # Save model
    morgana_io.save_model(
        str(model_dir),
        classifier,
        scaler,
        sigmas=SIGMAS,
        down_shape=DOWN_SHAPE,
        edge_size=EDGE_SIZE,
        fraction=FRACTION,
        bias=BIAS,
        feature_mode=FEATURE_MODE,
        deep=False,
    )
    print(f"  Model saved to: {model_dir}")

    return classifier, scaler


# ============================================================================
# STEP 4: Predict on all images
# ============================================================================

def predict_and_save(images: List[dict], classifier, scaler, output_dir: Path):
    """Run Morgana prediction on all images and save masks + overlays."""
    print(f"\n{'='*60}")
    print("PREDICTING ON ALL IMAGES")
    print(f"{'='*60}")

    mask_dir = output_dir / 'masks'
    overlay_dir = output_dir / 'overlays'
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for img_info in images:
        label = f"{img_info['condition']}/{img_info['timepoint']}/{img_info['sample_id']}"
        print(f"  {label}", end='')

        img_rgb = skio.imread(str(img_info['path']))
        bf, gfp = split_channels(img_rgb)
        bf_uint8 = (bf * 255).astype(np.uint8)

        t0 = time.time()

        # Predict with Morgana
        pred, prob = morgana_predict.predict_image(
            bf_uint8,
            classifier,
            scaler,
            sigmas=SIGMAS,
            new_shape_scale=DOWN_SHAPE,
            feature_mode=FEATURE_MODE,
            deep=False,
        )

        # Post-process: extract foreground, remove border objects
        negative = ndi.binary_fill_holes(pred == 0)
        mask_pred = ((pred == 1) * negative).astype(np.uint8)

        # Edge probability for watershed
        edge_prob = ((2**16 - 1) * prob[2]).astype(np.uint16)

        # Watershed to get clean boundary
        mask_watershed = morgana_predict.make_watershed(
            mask_pred, edge_prob,
            new_shape_scale=DOWN_SHAPE,
        )

        # Apply Morgana's smooth_mask for clean boundaries
        mask_final = smooth_mask(
            mask_watershed, mode='classifier',
            thin_order=5, smooth_order=15,
        ).astype(bool)

        # Keep largest component
        labeled, n = ndi.label(mask_final)
        if n > 0:
            sizes = ndi.sum(mask_final, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            mask_final = (labeled == largest)

        elapsed = time.time() - t0
        area = int(mask_final.sum())
        print(f"  -> area={area:,} px  ({elapsed:.1f}s)")

        # Compute metrics
        metrics = compute_metrics(mask_final, gfp)
        metrics.update({
            'condition': img_info['condition'],
            'timepoint': img_info['timepoint'],
            'sample_id': img_info['sample_id'],
            'filename': img_info['filename'],
            'area': area,
            'segmentation_time_s': round(elapsed, 2),
        })
        results.append(metrics)

        # Save mask
        name = f"{img_info['sample_id']}_{img_info['condition']}_{img_info['timepoint']}"
        tifffile.imwrite(str(mask_dir / f"{name}_mask.tif"),
                         (mask_final.astype(np.uint8) * 255))

        # Save overlay
        save_overlay(bf, gfp, mask_final, overlay_dir / f"{name}_overlay.png", img_info)

    return results


def compute_metrics(mask: np.ndarray, gfp: np.ndarray) -> dict:
    """Compute morphology + mezzo metrics."""
    result = {}
    if mask.sum() == 0:
        return {'aspect_ratio': 1.0, 'circularity': 0, 'solidity': 0,
                'eccentricity': 0, 'perimeter': 0,
                'mezzo_fraction': 0, 'mezzo_mean': 0}

    props = measure.regionprops(mask.astype(int))
    if not props:
        return {'aspect_ratio': 1.0, 'circularity': 0, 'solidity': 0,
                'eccentricity': 0, 'perimeter': 0,
                'mezzo_fraction': 0, 'mezzo_mean': 0}
    p = props[0]
    perimeter = p.perimeter if p.perimeter > 0 else 1e-6
    result['aspect_ratio'] = float(p.major_axis_length / p.minor_axis_length) if p.minor_axis_length > 0 else 1.0
    result['circularity'] = float((4 * np.pi * p.area) / (perimeter ** 2))
    result['solidity'] = float(p.solidity)
    result['eccentricity'] = float(p.eccentricity)
    result['perimeter'] = float(perimeter)

    # Mezzo
    gfp_in_mask = gfp[mask]
    if gfp_in_mask.size > 0 and gfp.max() > 0:
        try:
            thresh = filters.threshold_otsu(gfp_in_mask)
        except ValueError:
            thresh = 0.1
        result['mezzo_fraction'] = float((gfp_in_mask > thresh).sum() / gfp_in_mask.size)
        result['mezzo_mean'] = float(gfp_in_mask.mean())
    else:
        result['mezzo_fraction'] = 0.0
        result['mezzo_mean'] = 0.0

    return result


def save_overlay(bf, gfp, mask, save_path, img_info):
    """Save 3-panel overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bf, cmap='gray')
    axes[0].set_title(f"BF — {img_info['condition']} {img_info['timepoint']}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(bf, cmap='gray')
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
    axes[1].set_title(f"Morgana — area={int(mask.sum()):,} px", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(gfp, cmap='Greens', vmin=0, vmax=max(gfp.max(), 0.01))
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.5)
    axes[2].set_title('GFP (mezzo)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f"{img_info['sample_id']} | Morgana (trained)", fontsize=9, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# STEP 5: Save training overview (GT vs prediction)
# ============================================================================

def save_training_overview(train_images, train_gts, classifier, scaler,
                           output_dir: Path):
    """Show GT mask vs Morgana prediction for each training image."""
    overview_dir = output_dir / 'training_overview'
    overview_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, gt) in enumerate(zip(train_images, train_gts)):
        pred, prob = morgana_predict.predict_image(
            img, classifier, scaler,
            sigmas=SIGMAS, new_shape_scale=DOWN_SHAPE,
            feature_mode=FEATURE_MODE, deep=False,
        )
        negative = ndi.binary_fill_holes(pred == 0)
        mask_pred = ((pred == 1) * negative).astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input BF', fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title(f'Ground Truth (area={gt.sum():,})', fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(img, cmap='gray')
        for c in measure.find_contours(gt.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'g-', linewidth=2, label='GT')
        for c in measure.find_contours(mask_pred.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'r--', linewidth=2, label='Predicted')
        axes[2].set_title('GT (green) vs Predicted (red)', fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(str(overview_dir / f'train_{i:02d}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"  Training overview saved to: {overview_dir}")


# ============================================================================
# STEP 6: Generate box plots
# ============================================================================

def generate_boxplots(results: List[dict], output_dir: Path):
    """Generate box plots: control vs treated at initial vs elongated."""
    import pandas as pd

    df = pd.DataFrame(results)
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['aspect_ratio', 'circularity', 'solidity', 'area', 'mezzo_fraction', 'mezzo_mean']
    metric_labels = {
        'area': 'Area (pixels)',
        'aspect_ratio': 'Aspect Ratio',
        'circularity': 'Circularity',
        'solidity': 'Solidity',
        'mezzo_fraction': 'Mezzo Fraction',
        'mezzo_mean': 'Mezzo Mean Intensity',
    }
    cond_colors = {'control': '#888888', 'treated': '#cc3333'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        groups, labels, colors = [], [], []
        for tp in ['initial', 'elongated']:
            for cond in ['control', 'treated']:
                subset = df[(df['condition'] == cond) & (df['timepoint'] == tp)]
                vals = subset[metric].values
                groups.append(vals if len(vals) > 0 else np.array([np.nan]))
                labels.append(f"{cond}\n({tp})")
                colors.append(cond_colors[cond])

        bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        for j, vals in enumerate(groups):
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                x = np.random.normal(j + 1, 0.05, size=len(valid))
                ax.scatter(x, valid, alpha=0.8, s=40, c=colors[j],
                           edgecolors='black', linewidth=0.5, zorder=3)
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Morgana (Trained) — Control vs Treated', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(plot_dir / 'boxplot_morgana_trained.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Box plots saved to: {plot_dir}")

    df.to_csv(str(output_dir / 'morgana_results.csv'), index=False)
    print(f"  Results CSV saved")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Morgana and generate masks')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Input JPG directory')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output directory')
    parser.add_argument('--n-train', type=int, default=6,
                        help='Number of training images (default: 6)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    model_dir = output_dir / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  MORGANA TRAINING & PREDICTION PIPELINE")
    print("="*60)

    # Step 0: Discover images
    images = discover_images(args.input)
    print(f"\nFound {len(images)} images in {args.input}")
    for cond in ['control', 'treated']:
        for tp in ['initial', 'elongated']:
            n = sum(1 for im in images if im['condition'] == cond and im['timepoint'] == tp)
            if n > 0:
                print(f"  {cond}/{tp}: {n}")

    # Step 1-2: Setup training data with GT masks
    print(f"\n{'='*60}")
    print("STEP 1: Creating training data with GT masks")
    print(f"{'='*60}")
    train_images, train_gts = setup_training_data(images, model_dir, n_train=args.n_train)

    # Step 3: Train
    classifier, scaler = train_model(train_images, train_gts, model_dir)

    # Step 4: Training overview
    print(f"\n{'='*60}")
    print("STEP 2: Saving training overview")
    print(f"{'='*60}")
    save_training_overview(train_images, train_gts, classifier, scaler, output_dir)

    # Step 5: Predict on all images
    results = predict_and_save(images, classifier, scaler, output_dir)

    # Step 6: Box plots
    print(f"\n{'='*60}")
    print("STEP 3: Generating box plots")
    print(f"{'='*60}")
    generate_boxplots(results, output_dir)

    print(f"\n{'='*60}")
    print("  DONE!")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Model:     {model_dir}")
    print(f"  Masks:     {output_dir / 'masks'}")
    print(f"  Overlays:  {output_dir / 'overlays'}")
    print(f"  Plots:     {output_dir / 'plots'}")
    print(f"  Overview:  {output_dir / 'training_overview'}")


if __name__ == '__main__':
    main()
