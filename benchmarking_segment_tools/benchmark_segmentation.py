"""
Benchmarking Segmentation Tools for Pescoid Images
====================================================
Compare segmentation quality across tools on merged BF+GFP JPG images
of control vs treated pescoids at two timepoints (initial ~7.75hpf,
elongated ~12hpf).

Image format: 512x512 RGB JPGs where BF is grayscale across all channels
and GFP (mezzo) is overlaid in the green channel. So:
  - BF  = red channel (clean, no GFP contamination)
  - GFP = green_channel - red_channel (isolates mezzo signal)

Tools benchmarked (one per day):
  1. Morgana   -- ML pixel classifier (MOrgAna package)
  2. Cellpose  -- Deep learning (cyto3 model)
  3. StarDist  -- U-Net based (pretrained 2D versatile)
  4. Ilastik   -- Pixel Random Forest classifier

Metrics computed per image:
  - Area, perimeter, aspect ratio, circularity, solidity
  - Mezzo (GFP) fraction within the segmented mask
  - Mezzo mean/max intensity within mask

Input folder layout:
  Z:/Megha_Kattimani/Imaging/Zeiss/Benchmarking_Segmentation tools/JPG/
    control/
      initial/      *.jpg  (7.75 hpf)
      elongated/    *.jpg  (12 hpf)
    treated/
      initial/      *.jpg  (7.75 hpf)
      elongation/   *.jpg  (12 hpf)

Usage:
  python benchmarking_segment_tools/benchmark_segmentation.py --tool morgana
  python benchmarking_segment_tools/benchmark_segmentation.py --tool cellpose --gpu
  python benchmarking_segment_tools/benchmark_segmentation.py --tool stardist
  python benchmarking_segment_tools/benchmark_segmentation.py --tool ilastik
  python benchmarking_segment_tools/benchmark_segmentation.py  # all tools
"""

import os
import sys
import json
import argparse
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from skimage import io as skio, measure, morphology, filters, exposure
from skimage.util import img_as_float
from scipy import ndimage as ndi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'analysis'))
sys.path.insert(0, str(_PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(_PROJECT_ROOT / 'MOrgAna'))

# Default input path
DEFAULT_INPUT = r"Z:\Megha_Kattimani\Imaging\Zeiss\Benchmarking_Segmentation tools\JPG"


# ============================================================================
# IMAGE LOADING & CHANNEL SEPARATION
# ============================================================================

def load_images(input_dir: str) -> List[dict]:
    """
    Scan the JPG folder structure and return image metadata.
    Handles the actual folder naming: control/{initial,elongated}, treated/{initial,elongation}
    """
    images = []
    base = Path(input_dir)

    # Map actual folder names to standardised timepoint labels
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
        for f in sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.jpeg')):
            # Extract sample ID from filename (e.g., "#033" -> "033")
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
    """
    Split merged BF+GFP composite into separate channels.

    Returns:
        bf_gray: float64 [0,1] BF image (from red channel)
        gfp:     float64 [0,1] GFP signal (green - red, clipped)
    """
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)

    # BF = red channel (no GFP contamination)
    bf_gray = r / 255.0

    # GFP = excess green over the BF baseline
    gfp_raw = np.clip(g - r, 0, 255)
    gfp_max = gfp_raw.max()
    gfp = gfp_raw / gfp_max if gfp_max > 0 else gfp_raw

    return bf_gray, gfp


# ============================================================================
# SEGMENTATION BACKENDS
# ============================================================================

def segment_morgana(bf_gray: np.ndarray, min_area: int = 3000) -> np.ndarray:
    """
    Morgana segmentation using MOrgAna's smooth_mask post-processing.
    If a trained ML model exists, uses it; otherwise threshold + smooth_mask.
    """
    try:
        from morgana.ImageTools.segmentation.segment import smooth_mask
        has_smooth = True
    except ImportError:
        has_smooth = False
        print("    [Morgana] Could not import smooth_mask, using basic morphology")

    # Try ML model first
    try:
        import pickle
        from morgana.MLModel.predict import predict_image

        model_paths = list(_SCRIPT_DIR.glob('morgana_model*.pkl')) + \
                      list(_PROJECT_ROOT.glob('morgana_model*.pkl'))
        if model_paths:
            print(f"    [Morgana] Using trained model: {model_paths[0].name}")
            with open(model_paths[0], 'rb') as f:
                data = pickle.load(f)
            pred, prob = predict_image(
                (bf_gray * 255).astype(np.uint8),
                data['classifier'], data['scaler'],
                sigmas=data.get('params', {}).get('sigmas', [0.1, 0.5, 1, 2.5, 5, 7.5, 10]),
                new_shape_scale=data.get('params', {}).get('down_shape', 0.5),
                feature_mode=data.get('params', {}).get('feature_mode', 'ilastik'),
            )
            binary = (pred == 1).astype(np.uint8)
            if has_smooth:
                binary = smooth_mask(binary, mode='classifier', thin_order=5, smooth_order=15)
            return _largest_component(binary.astype(bool), min_area)
    except (ImportError, Exception) as e:
        if 'model_paths' in dir() and model_paths:
            print(f"    [Morgana] ML model failed: {e}")

    # Threshold-based Morgana approach
    # Invert BF (pescoid is darker than background)
    smoothed = filters.gaussian(bf_gray, sigma=3)
    thresh = filters.threshold_otsu(smoothed)
    binary = (smoothed < thresh).astype(np.uint8)
    binary = ndi.binary_fill_holes(binary).astype(np.uint8)

    if has_smooth:
        binary = smooth_mask(binary, mode='classifier', thin_order=5, smooth_order=15)

    return _largest_component(binary.astype(bool), min_area)


def segment_cellpose(bf_gray: np.ndarray, gpu: bool = False,
                     min_area: int = 3000) -> np.ndarray:
    """Cellpose cyto3 segmentation."""
    from cellpose import models

    model = models.Cellpose(gpu=gpu, model_type='cyto3')
    # Cellpose expects uint8 or float with reasonable range
    img_input = (bf_gray * 255).astype(np.uint8)
    masks_cp, flows, styles, diams = model.eval(
        [img_input], diameter=None, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    mask = masks_cp[0]
    if mask.max() == 0:
        return np.zeros_like(bf_gray, dtype=bool)
    props = measure.regionprops(mask)
    if not props:
        return np.zeros_like(bf_gray, dtype=bool)
    largest = max(props, key=lambda p: p.area)
    return (mask == largest.label).astype(bool)


def segment_stardist(bf_gray: np.ndarray, min_area: int = 3000) -> np.ndarray:
    """StarDist 2D (U-Net based) segmentation."""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # Invert BF so pescoid is bright (StarDist expects bright objects)
    inverted = 1.0 - bf_gray
    labels, details = model.predict_instances(normalize(inverted))
    if labels.max() == 0:
        return np.zeros_like(bf_gray, dtype=bool)
    props = measure.regionprops(labels)
    if not props:
        return np.zeros_like(bf_gray, dtype=bool)
    largest = max(props, key=lambda p: p.area)
    return (labels == largest.label).astype(bool)


def segment_ilastik(bf_gray: np.ndarray, min_area: int = 3000) -> np.ndarray:
    """Ilastik-style pixel Random Forest classifier with multi-scale features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Build Ilastik-style multi-scale features
    sigmas = [1, 2, 5, 10]
    features = []
    for sigma in sigmas:
        smoothed = filters.gaussian(bf_gray, sigma=sigma)
        features.append(smoothed)
        features.append(filters.sobel(smoothed))
        features.append(filters.laplace(smoothed))
    # Add raw + edges
    features.append(bf_gray)
    features.append(filters.sobel(bf_gray))

    feature_stack = np.stack(features, axis=-1)
    H, W, n_feat = feature_stack.shape
    X = feature_stack.reshape(-1, n_feat)

    # Generate training labels from Otsu on this image
    smoothed = filters.gaussian(bf_gray, sigma=3)
    thresh = filters.threshold_otsu(smoothed)
    labels_otsu = (smoothed < thresh).astype(int)
    labels_otsu = ndi.binary_fill_holes(labels_otsu).astype(int)
    y = labels_otsu.ravel()

    # Subsample for speed
    n_samples = min(50000, len(y))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y), n_samples, replace=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx])

    clf = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    clf.fit(X_train, y[idx])

    pred = clf.predict(scaler.transform(X)).reshape(H, W)
    mask = ndi.binary_fill_holes(pred.astype(bool))
    return _largest_component(mask, min_area)


def _largest_component(mask: np.ndarray, min_area: int = 3000) -> np.ndarray:
    """Keep only the largest connected component above min_area."""
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    labeled, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(bool)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOLS = {
    'morgana':  {'fn': segment_morgana,   'label': 'Morgana',        'color': '#e41a1c'},
    'cellpose': {'fn': segment_cellpose,  'label': 'Cellpose',       'color': '#377eb8'},
    'stardist': {'fn': segment_stardist,  'label': 'StarDist/U-Net', 'color': '#4daf4a'},
    'ilastik':  {'fn': segment_ilastik,   'label': 'Ilastik (RF)',   'color': '#984ea3'},
}


# ============================================================================
# METRICS
# ============================================================================

def compute_morphology_metrics(mask: np.ndarray) -> dict:
    """Compute shape metrics from a binary mask."""
    if mask.sum() == 0:
        return {'area': 0, 'perimeter': 0, 'aspect_ratio': 1.0,
                'circularity': 0, 'solidity': 0,
                'major_axis': 0, 'minor_axis': 0, 'eccentricity': 0}
    props = measure.regionprops(mask.astype(int))
    if not props:
        return {'area': 0, 'perimeter': 0, 'aspect_ratio': 1.0,
                'circularity': 0, 'solidity': 0,
                'major_axis': 0, 'minor_axis': 0, 'eccentricity': 0}
    p = props[0]
    perimeter = p.perimeter if p.perimeter > 0 else 1e-6
    circularity = (4 * np.pi * p.area) / (perimeter ** 2)
    aspect = p.major_axis_length / p.minor_axis_length if p.minor_axis_length > 0 else 1.0
    return {
        'area': int(p.area),
        'perimeter': float(perimeter),
        'aspect_ratio': float(aspect),
        'circularity': float(circularity),
        'solidity': float(p.solidity),
        'major_axis': float(p.major_axis_length),
        'minor_axis': float(p.minor_axis_length),
        'eccentricity': float(p.eccentricity),
    }


def compute_mezzo_metrics(mask: np.ndarray, gfp: np.ndarray) -> dict:
    """Compute mezzo/GFP expression metrics within the segmentation mask."""
    if mask.sum() == 0 or gfp.max() == 0:
        return {'mezzo_fraction': 0.0, 'mezzo_mean': 0.0,
                'mezzo_max': 0.0, 'mezzo_std': 0.0}
    gfp_in_mask = gfp[mask]
    if gfp_in_mask.size == 0:
        return {'mezzo_fraction': 0.0, 'mezzo_mean': 0.0,
                'mezzo_max': 0.0, 'mezzo_std': 0.0}

    # Threshold: Otsu on the GFP signal within mask
    try:
        thresh = filters.threshold_otsu(gfp_in_mask)
    except ValueError:
        thresh = 0.1
    fraction = float((gfp_in_mask > thresh).sum() / gfp_in_mask.size)

    return {
        'mezzo_fraction': fraction,
        'mezzo_mean': float(gfp_in_mask.mean()),
        'mezzo_max': float(gfp_in_mask.max()),
        'mezzo_std': float(gfp_in_mask.std()),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_overlay(bf_gray, gfp, mask, save_path, tool_name, img_info):
    """Save a 3-panel overlay: BF | BF+contour | GFP+contour."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: BF
    axes[0].imshow(bf_gray, cmap='gray')
    axes[0].set_title(f"BF — {img_info['condition']} {img_info['timepoint']}", fontsize=10)
    axes[0].axis('off')

    # Panel 2: BF + segmentation contour
    axes[1].imshow(bf_gray, cmap='gray')
    if mask.sum() > 0:
        for contour in measure.find_contours(mask.astype(float), 0.5):
            axes[1].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    area = int(mask.sum())
    axes[1].set_title(f"{tool_name} — area={area:,} px", fontsize=10)
    axes[1].axis('off')

    # Panel 3: GFP + contour
    axes[2].imshow(gfp, cmap='Greens', vmin=0, vmax=max(gfp.max(), 0.01))
    if mask.sum() > 0:
        for contour in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1.5)
    axes[2].set_title('GFP (mezzo)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f"{img_info['sample_id']}  |  {tool_name}", fontsize=9, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


def generate_boxplots(results: List[dict], output_dir: Path):
    """Generate box plots comparing control vs treated for each tool and timepoint."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        print("No results to plot.")
        return df

    tools_present = df['tool'].unique()
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Metrics to plot
    morph_metrics = ['aspect_ratio', 'circularity', 'solidity', 'area']
    mezzo_metrics = ['mezzo_fraction', 'mezzo_mean']
    all_metrics = morph_metrics + mezzo_metrics

    metric_labels = {
        'area': 'Area (pixels)',
        'aspect_ratio': 'Aspect Ratio',
        'circularity': 'Circularity (4piA/P²)',
        'solidity': 'Solidity',
        'mezzo_fraction': 'Mezzo Fraction',
        'mezzo_mean': 'Mezzo Mean Intensity',
    }

    cond_colors = {'control': '#888888', 'treated': '#cc3333'}

    # --- Per-tool plots ---
    for tool_name in tools_present:
        tool_df = df[df['tool'] == tool_name]
        tool_label = TOOLS.get(tool_name, {}).get('label', tool_name)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for i, metric in enumerate(all_metrics):
            ax = axes[i]
            groups = []
            labels = []
            colors = []
            for tp in ['initial', 'elongated']:
                for cond in ['control', 'treated']:
                    subset = tool_df[(tool_df['condition'] == cond) & (tool_df['timepoint'] == tp)]
                    vals = subset[metric].values
                    groups.append(vals if len(vals) > 0 else np.array([np.nan]))
                    labels.append(f"{cond}\n({tp})")
                    colors.append(cond_colors[cond])

            bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.6)
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            # Overlay data points
            for j, vals in enumerate(groups):
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0:
                    x = np.random.normal(j + 1, 0.05, size=len(valid))
                    ax.scatter(x, valid, alpha=0.8, s=40, c=colors[j],
                               edgecolors='black', linewidth=0.5, zorder=3)

            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f'{tool_label} — Control vs Treated (7.75 hpf → 12 hpf)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig_path = plot_dir / f'boxplot_{tool_name}.png'
        plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path.name}")

    # --- Cross-tool comparison (if multiple tools) ---
    if len(tools_present) > 1:
        for metric in all_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for tp_idx, tp in enumerate(['initial', 'elongated']):
                ax = axes[tp_idx]
                groups, labels, colors = [], [], []
                for tool_name in tools_present:
                    tool_info = TOOLS.get(tool_name, {'label': tool_name, 'color': 'gray'})
                    for cond in ['control', 'treated']:
                        subset = df[(df['tool'] == tool_name) &
                                    (df['condition'] == cond) &
                                    (df['timepoint'] == tp)]
                        vals = subset[metric].values
                        groups.append(vals if len(vals) > 0 else np.array([np.nan]))
                        labels.append(f"{tool_info['label']}\n{cond}")
                        colors.append('#aaaaaa' if cond == 'control' else tool_info['color'])

                bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.6)
                for patch, c in zip(bp['boxes'], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.5)
                for j, vals in enumerate(groups):
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0:
                        x = np.random.normal(j + 1, 0.05, size=len(valid))
                        ax.scatter(x, valid, alpha=0.7, s=25, c=colors[j],
                                   edgecolors='black', linewidth=0.5, zorder=3)

                ax.set_title(f'{tp.capitalize()} ({("7.75" if tp == "initial" else "12")} hpf)',
                            fontsize=12)
                ax.set_ylabel(metric_labels[metric], fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(axis='y', alpha=0.3)

            plt.suptitle(f'Tool Comparison — {metric_labels[metric]}',
                        fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            plt.savefig(str(plot_dir / f'comparison_{metric}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
        print("  Saved cross-tool comparison plots")

    # --- Summary table ---
    summary_metrics = ['area', 'aspect_ratio', 'circularity', 'solidity',
                       'mezzo_fraction', 'mezzo_mean']
    summary = df.groupby(['tool', 'condition', 'timepoint'])[summary_metrics].agg(
        ['mean', 'std', 'count'])
    csv_path = output_dir / 'benchmark_summary.csv'
    summary.to_csv(str(csv_path))
    print(f"  Saved: {csv_path.name}")

    # Also save flat results
    df.to_csv(str(output_dir / 'benchmark_results.csv'), index=False)
    print(f"  Saved: benchmark_results.csv")

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_benchmark(input_dir: str, output_dir: str,
                  tools_to_run: List[str], gpu: bool = False):
    """Run the full benchmarking pipeline."""
    images = load_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}!")
        print("Expected structure: control/{{initial,elongated}}/ and treated/{{initial,elongation}}/")
        return

    print(f"Found {len(images)} images:")
    for cond in ['control', 'treated']:
        for tp in ['initial', 'elongated']:
            n = sum(1 for im in images if im['condition'] == cond and im['timepoint'] == tp)
            if n > 0:
                print(f"  {cond}/{tp}: {n} images")
    print()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []

    for tool_name in tools_to_run:
        if tool_name not in TOOLS:
            print(f"Unknown tool: {tool_name}, skipping")
            continue

        tool_info = TOOLS[tool_name]
        print(f"\n{'='*60}")
        print(f"  {tool_info['label']}")
        print(f"{'='*60}")

        tool_out = out / tool_name
        overlay_dir = tool_out / 'overlays'
        mask_dir = tool_out / 'masks'
        overlay_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_info in images:
            print(f"  {img_info['condition']}/{img_info['timepoint']}/"
                  f"{img_info['sample_id']}", end='')

            # Load image and split channels
            img_rgb = skio.imread(str(img_info['path']))
            bf_gray, gfp = split_channels(img_rgb)

            # Segment
            t0 = time.time()
            try:
                if tool_name == 'cellpose':
                    mask = tool_info['fn'](bf_gray, gpu=gpu)
                else:
                    mask = tool_info['fn'](bf_gray)
                elapsed = time.time() - t0
                print(f"  -> area={mask.sum():,} px  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  -> FAILED: {e}")
                mask = np.zeros(bf_gray.shape, dtype=bool)
                elapsed = time.time() - t0

            # Compute metrics
            morph = compute_morphology_metrics(mask)
            mezzo = compute_mezzo_metrics(mask, gfp)

            row = {
                'tool': tool_name,
                'condition': img_info['condition'],
                'timepoint': img_info['timepoint'],
                'sample_id': img_info['sample_id'],
                'filename': img_info['filename'],
                'segmentation_time_s': round(elapsed, 2),
                **morph,
                **mezzo,
            }
            all_results.append(row)

            # Save binary mask as TIFF (0/255)
            mask_name = f"{img_info['sample_id']}_{img_info['condition']}_{img_info['timepoint']}_mask.tif"
            tifffile.imwrite(str(mask_dir / mask_name),
                             (mask.astype(np.uint8) * 255))

            # Save overlay
            overlay_name = f"{img_info['sample_id']}_{img_info['condition']}_{img_info['timepoint']}.png"
            save_overlay(bf_gray, gfp, mask, overlay_dir / overlay_name,
                        tool_info['label'], img_info)

    # Save raw results as JSON
    with open(str(out / 'benchmark_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate plots
    print(f"\n{'='*60}")
    print("Generating box plots...")
    print(f"{'='*60}")
    df = generate_boxplots(all_results, out)

    print(f"\nDone! All results in: {out}")
    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark segmentation tools on pescoid BF+GFP JPGs',
    )
    parser.add_argument('--tool', type=str, default=None,
                        choices=list(TOOLS.keys()),
                        help='Run a specific tool (default: all available)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for Cellpose')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f'Input directory (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: benchmarking_segment_tools/output)')
    args = parser.parse_args()

    output_dir = args.output or str(_SCRIPT_DIR / 'output')

    if args.tool:
        tools_to_run = [args.tool]
    else:
        tools_to_run = list(TOOLS.keys())
        print(f"Will attempt all tools: {', '.join(tools_to_run)}")
        print("(Unavailable ones will fail gracefully)\n")

    run_benchmark(args.input, output_dir, tools_to_run, gpu=args.gpu)


if __name__ == '__main__':
    main()
