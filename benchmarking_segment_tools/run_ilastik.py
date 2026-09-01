"""
Ilastik Pixel Classification Pipeline for Pescoid Benchmarking
===============================================================

Uses easilastik to run the actual Ilastik software (headless mode) for
pixel classification. Creates an Ilastik project (.ilp) from GT masks,
trains the classifier, then predicts on all images.

Training data:
  Z:/.../Benchmarking_Segmentation tools/JPG/model/trainingset/
    {name}.tif      <- BF images (512x512, uint8)
    {name}_GT.tif   <- manual GT masks (512x512, binary)

Input for prediction:
  Z:/.../Benchmarking_Segmentation tools/JPG/MORGANA/*.tif

Output:
  benchmarking_segment_tools/output/ilastik/
    project/        <- Ilastik project (.ilp) + training images
    masks/          <- binary masks (TIF, 0/255)
    overlays/       <- 3-panel overlays (BF + contour + GFP)
    plots/          <- box plots comparing control vs treated
    training_overview/  <- GT vs prediction comparison
    ilastik_results.csv

Usage:
  python benchmarking_segment_tools/run_ilastik.py
"""

import sys
import json
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage import measure, filters
from scipy import ndimage as ndi
import tifffile
import h5py

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
OUTPUT_DIR = _SCRIPT_DIR / "output" / "ilastik"

ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.0.post1\ilastik.exe"

# Sample -> condition mapping
SAMPLE_MAP = {
    '033': 'control', '043': 'control',
    '049': 'treated', '055': 'treated', '061': 'treated',
    '073': 'treated', '075': 'treated', '095': 'treated', '129': 'treated',
}

# How many training images to embed labels for (use a subset for speed)
N_TRAIN_LABELS = 6

# Ilastik axis tags JSON template
AXISTAGS_YXC = json.dumps({
    "axes": [
        {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
        {"key": "c", "typeFlags": 1, "resolution": 0, "description": ""},
    ]
})


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

        img = tifffile.imread(str(img_path))
        gt = tifffile.imread(str(gt_path))

        label = (gt > 0).astype(np.uint8)

        train_images.append(img)
        train_labels.append(label)
        train_names.append(name)

        area = int(label.sum())
        print(f"  {name}: image {img.shape}/{img.dtype}, GT area={area:,} px")

    return train_images, train_labels, train_names


# ============================================================================
# STEP 2: Create Ilastik project (.ilp) with GT labels
# ============================================================================

def _gt_to_label_blocks(gt_mask: np.ndarray, n_fg_strips=5, n_bg_strips=3, strip_width=8):
    """
    Convert a GT binary mask to sparse Ilastik label blocks.

    Creates horizontal strips through foreground (class 1) and
    background (class 2) regions to mimic user brush annotations.

    Returns list of (block_data, block_slice_str) tuples.
    """
    H, W = gt_mask.shape
    blocks = []

    # Foreground strips: horizontal lines through the mask interior
    fg_rows = np.where(gt_mask.any(axis=1))[0]
    if len(fg_rows) > 0:
        # Pick evenly spaced rows within the foreground
        step = max(1, len(fg_rows) // (n_fg_strips + 1))
        for i in range(1, n_fg_strips + 1):
            row_idx = fg_rows[min(i * step, len(fg_rows) - 1)]
            # Find the column extent of the mask on this row
            cols = np.where(gt_mask[row_idx, :])[0]
            if len(cols) < 10:
                continue
            # Take a strip through the middle of the mask
            c_start = cols[len(cols) // 4]
            c_end = cols[3 * len(cols) // 4]
            if c_end - c_start < 5:
                continue

            y0 = max(0, row_idx - strip_width // 2)
            y1 = min(H, row_idx + strip_width // 2)

            block = np.zeros((y1 - y0, c_end - c_start, 1), dtype=np.uint8)
            # Label 1 = foreground (class 1 in Ilastik)
            sub_mask = gt_mask[y0:y1, c_start:c_end]
            block[sub_mask, 0] = 1

            slice_str = f"[{y0}:{y1},{c_start}:{c_end},0:1]"
            blocks.append((block, slice_str))

    # Background strips: above and below the object
    bg_top_row = max(0, fg_rows[0] - 30) if len(fg_rows) > 0 else 10
    bg_bot_row = min(H, fg_rows[-1] + 30) if len(fg_rows) > 0 else H - 10

    # Top background
    y0 = max(0, bg_top_row - strip_width)
    y1 = bg_top_row
    if y1 > y0:
        block = np.ones((y1 - y0, W // 2, 1), dtype=np.uint8) * 2  # class 2 = background
        x0 = W // 4
        slice_str = f"[{y0}:{y1},{x0}:{x0 + W // 2},0:1]"
        blocks.append((block, slice_str))

    # Bottom background
    y0 = bg_bot_row
    y1 = min(H, bg_bot_row + strip_width)
    if y1 > y0:
        block = np.ones((y1 - y0, W // 2, 1), dtype=np.uint8) * 2
        x0 = W // 4
        slice_str = f"[{y0}:{y1},{x0}:{x0 + W // 2},0:1]"
        blocks.append((block, slice_str))

    # Side background
    fg_cols = np.where(gt_mask.any(axis=0))[0]
    if len(fg_cols) > 0:
        x_left = max(0, fg_cols[0] - 30)
        if x_left > strip_width:
            block = np.ones((H // 3, strip_width, 1), dtype=np.uint8) * 2
            y0 = H // 3
            slice_str = f"[{y0}:{y0 + H // 3},{x_left - strip_width}:{x_left},0:1]"
            blocks.append((block, slice_str))

    return blocks


def create_ilastik_project(train_images, train_labels, train_names, project_dir):
    """
    Create an Ilastik Pixel Classification project (.ilp) with training
    images and GT-derived label annotations.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    training_img_dir = project_dir / 'training_images'
    training_img_dir.mkdir(parents=True, exist_ok=True)

    # Select subset for labelling
    n_label = min(N_TRAIN_LABELS, len(train_images))
    # Pick diverse set: evenly spaced
    indices = np.linspace(0, len(train_images) - 1, n_label, dtype=int)

    # Save training images as TIF files for Ilastik
    label_img_paths = []
    for idx in indices:
        img = train_images[idx]
        name = train_names[idx]
        save_path = training_img_dir / f"{name}.tif"
        tifffile.imwrite(str(save_path), img)
        label_img_paths.append(save_path)

    ilp_path = project_dir / 'pescoid_pixclass.ilp'
    print(f"\n  Creating Ilastik project: {ilp_path}")
    print(f"  Training images with labels: {n_label}")

    with h5py.File(str(ilp_path), 'w') as f:
        # --- Project metadata ---
        f.create_dataset('workflowName', data='Pixel Classification')
        f.create_dataset('ilastikVersion', data='1.4.0.post1')
        f.create_dataset('currentApplet', data=0)
        f.create_dataset('time', data='2026-03-18 12:00:00')

        # --- Input Data ---
        input_grp = f.create_group('Input Data')
        input_grp.create_dataset('Role Names', data=[b'Raw Data', b'Prediction Mask'])
        input_grp.create_dataset('StorageVersion', data='0.2')
        infos = input_grp.create_group('infos')
        input_grp.create_group('local_data')

        for i, (idx, img_path) in enumerate(zip(indices, label_img_paths)):
            lane = infos.create_group(f'lane{i:04d}')
            lane.create_group('Prediction Mask')
            raw = lane.create_group('Raw Data')

            raw.create_dataset('__class__', data='FilesystemDatasetInfo')
            raw.create_dataset('allowLabels', data=True)
            raw.create_dataset('axistags', data=json.dumps({
                "axes": [
                    {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
                    {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
                ]
            }))
            raw.create_dataset('datasetId', data=f'dataset-{i:04d}')
            raw.create_dataset('display_mode', data='default')
            # Use absolute path with forward slashes
            fpath = str(img_path).replace('\\', '/')
            raw.create_dataset('filePath', data=fpath)
            raw.create_dataset('location', data='FileSystem')
            raw.create_dataset('nickname', data=img_path.stem)
            raw.create_dataset('normalizeDisplay', data=False)
            raw.create_dataset('scale_locked', data=False)
            raw.create_dataset('working_scale', data='')

            H, W = train_images[idx].shape[:2]
            raw.create_dataset('shape', data=np.array([H, W], dtype=np.int32))

        # --- Feature Selections (matching the existing test project) ---
        fs = f.create_group('FeatureSelections')
        scales = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
        feature_ids = [
            'GaussianSmoothing', 'LaplacianOfGaussian',
            'GaussianGradientMagnitude', 'DifferenceOfGaussians',
            'StructureTensorEigenvalues', 'HessianOfGaussianEigenvalues',
        ]
        fs.create_dataset('Scales', data=np.array(scales, dtype=np.float64))
        fs.create_dataset('FeatureIds', data=[s.encode() for s in feature_ids])

        # Enable all features at all scales
        sel_matrix = np.ones((len(feature_ids), len(scales)), dtype=bool)
        fs.create_dataset('SelectionMatrix', data=sel_matrix)
        fs.create_dataset('ComputeIn2d', data=np.ones(len(scales), dtype=bool))
        fs.create_dataset('StorageVersion', data='0.1')

        # --- Pixel Classification ---
        pc = f.create_group('PixelClassification')
        pc.create_dataset('StorageVersion', data='0.1')
        pc.create_dataset('LabelColors', data=np.array([[0, 130, 200], [230, 25, 75]], dtype=np.int32))
        pc.create_dataset('LabelNames', data=[b'Foreground', b'Background'])
        pc.create_dataset('PmapColors', data=np.array([[0, 0, 255], [255, 0, 0]], dtype=np.int32))
        pc.create_dataset('ClassifierFactory',
                          data='ParallelVigraRfLazyflowClassifierFactory(100)')
        pc.create_group('ClassifierForests')
        bk = pc.create_group('Bookmarks')

        # Create label sets from GT masks
        label_sets = pc.create_group('LabelSets')
        for i, idx in enumerate(indices):
            gt = train_labels[idx]
            name = train_names[idx]

            lane_labels = label_sets.create_group(f'labels{i:03d}')
            bk.create_dataset(f'{i:04d}', data=np.zeros(6, dtype=np.uint8))

            # Convert GT mask to sparse label blocks
            blocks = _gt_to_label_blocks(gt)
            for j, (block_data, slice_str) in enumerate(blocks):
                ds = lane_labels.create_dataset(f'block{j:04d}', data=block_data)
                ds.attrs['blockSlice'] = slice_str
                ds.attrs['axistags'] = AXISTAGS_YXC

            print(f"    {name}: {len(blocks)} label blocks")

        # --- Prediction Export ---
        pe = f.create_group('Prediction Export')
        pe.create_dataset('OutputFilenameFormat', data='{dataset_dir}/{nickname}_Segmentation')
        pe.create_dataset('OutputFormat', data='tiff')
        pe.create_dataset('OutputInternalPath', data='exported_data')
        pe.create_dataset('StorageVersion', data='0.1')

    print(f"  Project saved: {ilp_path}")
    return str(ilp_path)


# ============================================================================
# STEP 3: Run Ilastik headless prediction via easilastik
# ============================================================================

def prepare_prediction_images(images: List[dict], pred_input_dir: Path):
    """Copy BF images to a local folder for Ilastik batch processing."""
    pred_input_dir.mkdir(parents=True, exist_ok=True)

    for img_info in images:
        src = img_info['bf_path']
        dst = pred_input_dir / f"{img_info['name']}.tif"
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
        img_info['pred_input_path'] = dst

    return images


def run_ilastik_prediction(ilp_path: str, images: List[dict], output_dir: Path):
    """
    Run Ilastik headless prediction using easilastik.
    Returns list of image dicts with 'mask' added.
    """
    from easilastik import run_ilastik

    pred_output_dir = output_dir / 'ilastik_raw_output'
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    pred_input_dir = output_dir / 'prediction_input'
    images = prepare_prediction_images(images, pred_input_dir)

    print(f"\n  Running Ilastik headless on {len(images)} images...")
    print(f"  Project: {ilp_path}")
    print(f"  Ilastik: {ILASTIK_EXE}")
    print(f"  Input:   {pred_input_dir}")
    print(f"  Output:  {pred_output_dir}")

    t0 = time.time()

    # Run easilastik on the folder of input images
    run_ilastik(
        input_path=str(pred_input_dir),
        model_path=ilp_path,
        result_base_path=str(pred_output_dir),
        ilastik_script_path=ILASTIK_EXE,
        export_source='Simple Segmentation',
        output_format='tiff',
    )

    elapsed = time.time() - t0
    print(f"\n  Ilastik prediction complete in {elapsed:.1f}s")

    # Parse output segmentation maps
    results = []
    for img_info in images:
        name = img_info['name']

        # Find the output file (Ilastik may use various naming patterns)
        possible_names = [
            f"{name}_Simple Segmentation.tiff",
            f"{name}_Simple Segmentation.tif",
            f"{name}_Segmentation.tiff",
            f"{name}_Segmentation.tif",
            f"{name}.tiff",
            f"{name}.tif",
        ]

        seg_path = None
        for pn in possible_names:
            candidate = pred_output_dir / pn
            if candidate.exists():
                seg_path = candidate
                break

        # Also try glob for partial matches (use underscore to avoid 033 matching 033e)
        if seg_path is None:
            matches = list(pred_output_dir.glob(f"{name}_*"))
            if matches:
                seg_path = matches[0]

        if seg_path is None:
            print(f"  Warning: no output found for {name}")
            img_info['mask'] = np.zeros((512, 512), dtype=bool)
            results.append(img_info)
            continue

        seg = tifffile.imread(str(seg_path))
        print(f"  {name}: seg shape={seg.shape}, values={np.unique(seg)}", end=' ')

        # Ilastik Simple Segmentation: class 1 = foreground, class 2 = background
        # (1-indexed). Extract foreground.
        binary_mask = (seg == 1)
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[:, :, 0] if binary_mask.shape[2] == 1 else binary_mask.any(axis=2)

        # Keep largest component
        labeled, n = ndi.label(binary_mask)
        if n > 0:
            sizes = ndi.sum(binary_mask, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            binary_mask = (labeled == largest)

        area = binary_mask.sum()
        print(f"-> area={area:,} px")

        img_info['mask'] = binary_mask
        results.append(img_info)

    return results


# ============================================================================
# STEP 4: Training overview (GT vs Ilastik prediction)
# ============================================================================

def save_training_overview(train_images, train_labels, train_names,
                           ilp_path, output_dir):
    """Compare GT masks vs Ilastik predictions on the training images."""
    from easilastik import run_ilastik

    overview_dir = output_dir / 'training_overview'
    overview_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training images for Ilastik
    train_input_dir = output_dir / 'training_pred_input'
    train_output_dir = output_dir / 'training_pred_output'
    train_input_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)

    for img, name in zip(train_images, train_names):
        save_path = train_input_dir / f"{name}.tif"
        if not save_path.exists():
            tifffile.imwrite(str(save_path), img)

    print(f"  Running Ilastik on {len(train_images)} training images...")
    run_ilastik(
        input_path=str(train_input_dir),
        model_path=ilp_path,
        result_base_path=str(train_output_dir),
        ilastik_script_path=ILASTIK_EXE,
        export_source='Simple Segmentation',
        output_format='tiff',
    )

    for i, (img, gt, name) in enumerate(zip(train_images, train_labels, train_names)):
        # Find prediction (use underscore to avoid 033 matching 033e)
        matches = list(train_output_dir.glob(f"{name}_*"))
        if not matches:
            print(f"  Warning: no prediction for training image {name}")
            continue

        seg = tifffile.imread(str(matches[0]))
        pred = (seg == 1)
        if pred.ndim == 3:
            pred = pred[:, :, 0] if pred.shape[2] == 1 else pred.any(axis=2)

        gt_bool = gt > 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'{name} — BF', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img, cmap='gray')
        for c in measure.find_contours(gt_bool.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'g-', linewidth=2)
        axes[1].set_title(f'GT mask (area={gt_bool.sum():,})', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(img, cmap='gray')
        for c in measure.find_contours(pred.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
        if gt_bool.sum() > 0:
            for c in measure.find_contours(gt_bool.astype(float), 0.5):
                axes[2].plot(c[:, 1], c[:, 0], 'g--', linewidth=1, alpha=0.7)
        axes[2].set_title(f'Ilastik prediction (area={pred.sum():,})', fontsize=10)
        axes[2].axis('off')

        intersection = (gt_bool & pred).sum()
        union = (gt_bool | pred).sum()
        iou = intersection / union if union > 0 else 0
        plt.suptitle(f'{name} — GT vs Ilastik (IoU={iou:.3f})', fontsize=12, fontweight='bold')

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
    axes[1].set_title(f"Ilastik — area={int(mask.sum()):,} px", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(gfp, cmap='Greens', vmin=0, vmax=max(gfp.max(), 0.01))
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.5)
    axes[2].set_title('GFP (mezzo)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f"{img_info['sample_id']} — Ilastik segmentation", fontsize=9, y=0.98)
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

    plt.suptitle('Ilastik — Control vs Treated', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(plot_dir / 'boxplot_ilastik.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Box plots saved to: {plot_dir}")

    csv_path = output_dir / 'ilastik_results.csv'
    try:
        df.to_csv(str(csv_path), index=False)
        print(f"  Results CSV saved to: {csv_path}")
    except PermissionError:
        alt_path = output_dir / 'ilastik_results_new.csv'
        df.to_csv(str(alt_path), index=False)
        print(f"  WARNING: {csv_path} is locked. Saved to: {alt_path}")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  ILASTIK PIXEL CLASSIFICATION PIPELINE (easilastik)")
    print("=" * 60)

    # Create output directories
    project_dir = OUTPUT_DIR / 'project'
    mask_dir = OUTPUT_DIR / 'masks'
    overlay_dir = OUTPUT_DIR / 'overlays'
    for d in [project_dir, mask_dir, overlay_dir, OUTPUT_DIR / 'plots']:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load GT training data
    print(f"\n{'=' * 60}")
    print("STEP 1: Loading GT training data")
    print(f"{'=' * 60}")
    train_images, train_labels, train_names = load_training_data()

    # Step 2: Create Ilastik project with GT labels
    print(f"\n{'=' * 60}")
    print("STEP 2: Creating Ilastik project with GT labels")
    print(f"{'=' * 60}")
    ilp_path = create_ilastik_project(train_images, train_labels, train_names, project_dir)

    # Step 3: Training overview
    print(f"\n{'=' * 60}")
    print("STEP 3: Saving training overview (GT vs prediction)")
    print(f"{'=' * 60}")
    save_training_overview(train_images, train_labels, train_names, ilp_path, OUTPUT_DIR)

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

    results = run_ilastik_prediction(ilp_path, images, OUTPUT_DIR)

    # Step 5: Compute metrics and save masks/overlays
    print(f"\n{'=' * 60}")
    print("STEP 5: Computing metrics and saving outputs")
    print(f"{'=' * 60}")
    rows = []
    for img_info in results:
        mask = img_info['mask']

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
    print(f"  Project: {project_dir}")
    print(f"  Masks:   {mask_dir}")
    print(f"  Overlays:{overlay_dir}")
    print(f"  Overview:{OUTPUT_DIR / 'training_overview'}")
    print(f"  Plots:   {OUTPUT_DIR / 'plots'}")
    print(f"  CSV:     {OUTPUT_DIR / 'ilastik_results.csv'}")


if __name__ == '__main__':
    main()
