"""
Cellpose Segmentation Pipeline for Pescoid Benchmarking
========================================================

Fine-tunes Cellpose on manually annotated GT masks (from Morgana training set),
then predicts on all images. Computes morphology metrics and generates box plots.

Training data:
  Z:/.../Benchmarking_Segmentation tools/JPG/model/trainingset/
    {name}.tif      <- BF images (512x512, uint8)
    {name}_GT.tif   <- manual GT masks (512x512, binary)

Input for prediction:
  Z:/.../Benchmarking_Segmentation tools/JPG/MORGANA/*.tif

Output:
  benchmarking_segment_tools/output/cellpose/
    model/          <- fine-tuned Cellpose model
    masks/          <- binary masks (TIF, 0/255)
    overlays/       <- 3-panel overlays (BF + contour + GFP)
    plots/          <- box plots comparing control vs treated
    training_overview/  <- GT vs prediction comparison
    cellpose_results.csv

Usage:
  python benchmarking_segment_tools/run_cellpose.py
"""

import sys
import glob as glob_mod
from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage import measure, filters
import tifffile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent

JPG_BASE = Path(r"Z:\Megha_Kattimani\Imaging\Zeiss\Benchmarking_Segmentation tools\JPG")
TRAINING_DIR = JPG_BASE / "model" / "trainingset"
BF_DIR = JPG_BASE / "MORGANA"
OUTPUT_DIR = _SCRIPT_DIR / "output" / "cellpose"

# Sample -> condition mapping
SAMPLE_MAP = {
    '033': 'control', '043': 'control',
    '049': 'treated', '055': 'treated', '061': 'treated',
    '073': 'treated', '075': 'treated', '095': 'treated', '129': 'treated',
}

# Training config
N_EPOCHS = 200
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
USE_GPU = True


# ============================================================================
# STEP 0: Discover images
# ============================================================================

def discover_images() -> List[dict]:
    """Find all TIF images in the MORGANA BF folder and map metadata."""
    images = []
    for tif_path in sorted(BF_DIR.glob("*.tif")):
        name = tif_path.stem
        name_lower = name.lower()
        if name_lower.endswith('e'):
            sample_id = name_lower[:-1]
            timepoint = 'elongated'
        else:
            sample_id = name_lower
            timepoint = 'initial'

        condition = SAMPLE_MAP.get(sample_id)
        if condition is None:
            print(f"  Warning: unknown sample '{sample_id}' in {name}, skipping")
            continue

        # Find original JPG to extract GFP channel
        if condition == 'control':
            tp_folder = 'elongated' if timepoint == 'elongated' else 'initial'
        else:
            tp_folder = 'elongation' if timepoint == 'elongated' else 'initial'
        jpg_pattern = f"*#{sample_id}*"
        jpg_folder = JPG_BASE / condition / tp_folder
        jpg_files = list(jpg_folder.glob(jpg_pattern + ".jpg"))

        images.append({
            'bf_path': tif_path,
            'jpg_path': jpg_files[0] if jpg_files else None,
            'name': name,
            'sample_id': sample_id,
            'condition': condition,
            'timepoint': timepoint,
        })

    return images


def split_channels_from_jpg(jpg_path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract BF and GFP from merged JPG."""
    from skimage import io as skio
    img_rgb = skio.imread(str(jpg_path))
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    bf = r / 255.0
    gfp_raw = np.clip(g - r, 0, 255)
    gfp_max = gfp_raw.max()
    gfp = gfp_raw / gfp_max if gfp_max > 0 else gfp_raw
    return bf, gfp

#BUT WHY ARE WE SPLITTING CHANNELS FROM JPG? BECAUSE THE ORIGINAL TIF FILES ONLY HAVE THE BRIGHTFIELD, AND THE GFP SIGNAL IS ENCODED IN THE MERGED JPGS. WE NEED THE GFP TO COMPUTE THE MEZZO METRICS, SO WE EXTRACT IT FROM THE JPG BY SUBTRACTING THE RED CHANNEL (BF) FROM THE GREEN CHANNEL (GFP).


# ============================================================================
# STEP 1: Load GT training data
# ============================================================================

def load_training_data():
    """Load BF images and GT masks from the Morgana training set."""
    gt_files = sorted(TRAINING_DIR.glob("*_GT.tif"))
    print(f"\nFound {len(gt_files)} GT masks in {TRAINING_DIR}")

    train_images = []
    train_labels = []
    train_names = []

    for gt_path in gt_files:
        name = gt_path.stem.replace('_GT', '')
        img_path = gt_path.parent / f"{name}.tif"

        if not img_path.exists():
            print(f"  Warning: no image for {name}, skipping")
            continue

        img = tifffile.imread(str(img_path))   # uint8, 512x512
        gt = tifffile.imread(str(gt_path))     # uint16, binary (0/1)

        # Cellpose expects labels as integer instance masks (1 = first object)
        label = (gt > 0).astype(np.int32)

        train_images.append(img)
        train_labels.append(label)
        train_names.append(name)

        area = label.sum()
        print(f"  {name}: image {img.shape}/{img.dtype}, GT area={area:,} px")

    return train_images, train_labels, train_names


# ============================================================================
# STEP 2: Train (fine-tune) Cellpose
# ============================================================================

def train_cellpose(train_images, train_labels, model_dir):
    """Fine-tune Cellpose on the GT masks."""
    from cellpose import models, train

    model_dir.mkdir(parents=True, exist_ok=True)
    # Cellpose expects this subdirectory to exist for saving
    (model_dir / 'models').mkdir(parents=True, exist_ok=True)

    # Use ALL images for training (this is benchmarking, not generalization)
    train_imgs = train_images
    train_lbls = train_labels

    print(f"\n  Training set: {len(train_imgs)} images (all GT annotations)")

    # Initialize from pretrained cpsam, then fine-tune
    print(f"\n  Loading pretrained Cellpose model...")
    model = models.CellposeModel(pretrained_model='cpsam', gpu=USE_GPU)

    print(f"  Fine-tuning for {N_EPOCHS} epochs (lr={LEARNING_RATE})...")
    print(f"  This may take a while on CPU...\n")

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_lbls,
        test_data=train_imgs,
        test_labels=train_lbls,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        save_path=str(model_dir),
        save_every=N_EPOCHS,  # save at end
        min_train_masks=1,    # we have 1 mask per image (whole pescoid)
        model_name='cellpose_pescoid',
    )

    print(f"\n  Model saved to: {model_path}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    if len(test_losses) > 0:
        print(f"  Final test loss:  {test_losses[-1]:.4f}")

    # Save training loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train loss', color='#2196F3')
    if len(test_losses) > 0:
        # Test losses are evaluated less frequently
        test_x = np.linspace(0, len(train_losses) - 1, len(test_losses))
        ax.plot(test_x, test_losses, label='Test loss', color='#F44336')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cellpose Fine-tuning Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(model_dir / 'training_loss.png'), dpi=150)
    plt.close(fig)

    return model_path, model


# ============================================================================
# STEP 3: Predict with trained model
# ============================================================================

def predict_with_trained_model(images: List[dict], model_path: str) -> List[dict]:
    """Run prediction using the fine-tuned model."""
    from cellpose import models

    print(f"\n  Loading fine-tuned model from: {model_path}")
    model = models.CellposeModel(pretrained_model=model_path, gpu=USE_GPU)

    results = []
    for img_info in images:
        bf_img = tifffile.imread(str(img_info['bf_path']))
        print(f"  Predicting {img_info['name']}...", end=' ')

        masks, flows, styles = model.eval(bf_img, flow_threshold=0.4, cellprob_threshold=0.0)

        if masks.max() == 0:
            print("no mask found!")
            binary_mask = np.zeros_like(bf_img, dtype=bool)
        else:
            labels = np.unique(masks)
            labels = labels[labels > 0]
            largest_label = max(labels, key=lambda l: (masks == l).sum())
            binary_mask = masks == largest_label
            area = binary_mask.sum()
            print(f"found {len(labels)} region(s), largest area={area:,} px")

        img_info['mask'] = binary_mask
        results.append(img_info)

    return results


# ============================================================================
# STEP 4: Training overview (GT vs prediction)
# ============================================================================

def save_training_overview(train_images, train_labels, train_names, model_path, output_dir):
    """Compare GT masks vs Cellpose predictions on the training images."""
    from cellpose import models

    overview_dir = output_dir / 'training_overview'
    overview_dir.mkdir(parents=True, exist_ok=True)

    model = models.CellposeModel(pretrained_model=model_path, gpu=USE_GPU)

    for i, (img, gt, name) in enumerate(zip(train_images, train_labels, train_names)):
        masks, _, _ = model.eval(img, flow_threshold=0.4, cellprob_threshold=0.0)

        if masks.max() > 0:
            labels = np.unique(masks)
            labels = labels[labels > 0]
            largest = max(labels, key=lambda l: (masks == l).sum())
            pred = (masks == largest)
        else:
            pred = np.zeros_like(img, dtype=bool)

        gt_bool = gt > 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'{name} — BF', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img, cmap='gray')
        for c in measure.find_contours(gt_bool.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'g-', linewidth=2, label='GT')
        axes[1].set_title(f'GT mask (area={gt_bool.sum():,})', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(img, cmap='gray')
        for c in measure.find_contours(pred.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'r-', linewidth=2, label='Cellpose')
        if gt_bool.sum() > 0:
            for c in measure.find_contours(gt_bool.astype(float), 0.5):
                axes[2].plot(c[:, 1], c[:, 0], 'g--', linewidth=1, alpha=0.7)
        axes[2].set_title(f'Cellpose prediction (area={pred.sum():,})', fontsize=10)
        axes[2].axis('off')

        # Compute IoU
        intersection = (gt_bool & pred).sum()
        union = (gt_bool | pred).sum()
        iou = intersection / union if union > 0 else 0
        plt.suptitle(f'{name} — GT vs Cellpose (IoU={iou:.3f})', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(str(overview_dir / f'overview_{name}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"  Training overview saved to: {overview_dir}")


# ============================================================================
# STEP 5: Compute metrics
# ============================================================================

def compute_metrics(mask: np.ndarray, gfp: np.ndarray) -> dict:
    """Compute morphology + mezzo metrics from mask and GFP."""
    if mask.sum() == 0:
        return {'aspect_ratio': np.nan, 'circularity': np.nan, 'solidity': np.nan,
                'area': 0, 'eccentricity': np.nan, 'perimeter': 0,
                'mezzo_fraction': np.nan, 'mezzo_mean': np.nan}

    props = measure.regionprops(mask.astype(int))
    if not props:
        return {'aspect_ratio': np.nan, 'circularity': np.nan, 'solidity': np.nan,
                'area': 0, 'eccentricity': np.nan, 'perimeter': 0,
                'mezzo_fraction': np.nan, 'mezzo_mean': np.nan}

    p = props[0]
    perimeter = p.perimeter if p.perimeter > 0 else 1e-6

    result = {
        'aspect_ratio': float(p.major_axis_length / p.minor_axis_length) if p.minor_axis_length > 0 else 1.0,
        'circularity': float((4 * np.pi * p.area) / (perimeter ** 2)),
        'solidity': float(p.solidity),
        'area': int(p.area),
        'eccentricity': float(p.eccentricity),
        'perimeter': float(perimeter),
    }

    # Mezzo (GFP) analysis
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


# ============================================================================
# STEP 6: Save overlays
# ============================================================================

def save_overlay(bf, gfp, mask, save_path, img_info):
    """Save 3-panel overlay: BF | BF+contour | GFP+contour."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bf, cmap='gray')
    axes[0].set_title(f"BF — {img_info['condition']} {img_info['timepoint']}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(bf, cmap='gray')
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
    axes[1].set_title(f"Cellpose — area={int(mask.sum()):,} px", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(gfp, cmap='Greens', vmin=0, vmax=max(gfp.max(), 0.01))
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.5)
    axes[2].set_title('GFP (mezzo)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f"{img_info['sample_id']} — Cellpose segmentation", fontsize=9, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# STEP 7: Box plots
# ============================================================================

def generate_boxplots(rows: List[dict], output_dir: Path):
    """Generate box plots: control vs treated at initial vs elongated."""
    import pandas as pd

    df = pd.DataFrame(rows)
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
        groups, tick_labels, colors = [], [], []
        for tp in ['initial', 'elongated']:
            for cond in ['control', 'treated']:
                subset = df[(df['condition'] == cond) & (df['timepoint'] == tp)]
                vals = subset[metric].dropna().values
                groups.append(vals if len(vals) > 0 else np.array([np.nan]))
                tick_labels.append(f"{cond}\n({tp})")
                colors.append(cond_colors[cond])

        bp = ax.boxplot(groups, tick_labels=tick_labels, patch_artist=True, widths=0.6)
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

    plt.suptitle('Cellpose (fine-tuned) — Control vs Treated', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(plot_dir / 'boxplot_cellpose.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Box plots saved to: {plot_dir}")

    csv_path = output_dir / 'cellpose_results.csv'
    try:
        df.to_csv(str(csv_path), index=False)
        print(f"  Results CSV saved to: {csv_path}")
    except PermissionError:
        alt_path = output_dir / 'cellpose_results_new.csv'
        df.to_csv(str(alt_path), index=False)
        print(f"  WARNING: {csv_path} is locked. Saved to: {alt_path}")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  CELLPOSE FINE-TUNED SEGMENTATION PIPELINE")
    print("=" * 60)

    # Create output directories
    model_dir = OUTPUT_DIR / 'model'
    mask_dir = OUTPUT_DIR / 'masks'
    overlay_dir = OUTPUT_DIR / 'overlays'
    for d in [model_dir, mask_dir, overlay_dir, OUTPUT_DIR / 'plots']:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load GT training data
    print(f"\n{'=' * 60}")
    print("STEP 1: Loading GT training data")
    print(f"{'=' * 60}")
    train_images, train_labels, train_names = load_training_data()

    # Step 2: Train Cellpose
    print(f"\n{'=' * 60}")
    print("STEP 2: Fine-tuning Cellpose on GT masks")
    print(f"{'=' * 60}")
    model_path, _ = train_cellpose(train_images, train_labels, model_dir)

    # Step 3: Training overview
    print(f"\n{'=' * 60}")
    print("STEP 3: Saving training overview (GT vs prediction)")
    print(f"{'=' * 60}")
    save_training_overview(train_images, train_labels, train_names, model_path, OUTPUT_DIR)

    # Step 4: Discover all images and predict
    print(f"\n{'=' * 60}")
    print("STEP 4: Predicting on all images")
    print(f"{'=' * 60}")
    images = discover_images()
    print(f"\nFound {len(images)} images")
    for cond in ['control', 'treated']:
        for tp in ['initial', 'elongated']:
            n = sum(1 for im in images if im['condition'] == cond and im['timepoint'] == tp)
            if n > 0:
                print(f"  {cond}/{tp}: {n}")

    results = predict_with_trained_model(images, model_path)

    # Step 5: Compute metrics and save masks/overlays
    print(f"\n{'=' * 60}")
    print("STEP 5: Computing metrics and saving outputs")
    print(f"{'=' * 60}")
    rows = []
    for img_info in results:
        mask = img_info['mask']

        # Get GFP from original JPG
        if img_info['jpg_path'] is not None:
            _, gfp = split_channels_from_jpg(img_info['jpg_path'])
        else:
            gfp = np.zeros_like(mask, dtype=float)
            print(f"  Warning: no JPG found for {img_info['name']}, GFP will be zeros")

        metrics = compute_metrics(mask, gfp)
        metrics['sample_id'] = img_info['sample_id']
        metrics['condition'] = img_info['condition']
        metrics['timepoint'] = img_info['timepoint']
        metrics['name'] = img_info['name']
        rows.append(metrics)

        out_name = f"{img_info['sample_id']}_{img_info['condition']}_{img_info['timepoint']}"
        tifffile.imwrite(str(mask_dir / f"{out_name}_mask.tif"),
                         (mask.astype(np.uint8) * 255))

        bf_img = tifffile.imread(str(img_info['bf_path']))
        bf_norm = bf_img.astype(float) / 255.0
        save_overlay(bf_norm, gfp, mask, overlay_dir / f"{out_name}_overlay.png", img_info)

        print(f"  {img_info['name']}: AR={metrics['aspect_ratio']:.2f}, "
              f"circ={metrics['circularity']:.2f}, sol={metrics['solidity']:.2f}, "
              f"mezzo={metrics['mezzo_fraction']:.2f}")

    # Step 6: Box plots
    print(f"\n{'=' * 60}")
    print("STEP 6: Generating box plots")
    print(f"{'=' * 60}")
    generate_boxplots(rows, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("  DONE!")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    print(f"  Model:    {model_dir}")
    print(f"  Masks:    {mask_dir}")
    print(f"  Overlays: {overlay_dir}")
    print(f"  Overview: {OUTPUT_DIR / 'training_overview'}")
    print(f"  Plots:    {OUTPUT_DIR / 'plots'}")
    print(f"  CSV:      {OUTPUT_DIR / 'cellpose_results.csv'}")


if __name__ == '__main__':
    main()
