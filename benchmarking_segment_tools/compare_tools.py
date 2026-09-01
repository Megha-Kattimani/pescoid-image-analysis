"""
Cross-Tool Segmentation Benchmarking Comparison
=================================================

Compares Morgana, Cellpose, 3D U-Net, and Ilastik segmentation results
against ground truth (GT) masks. Generates publication-ready comparison
plots and a summary table.

Metrics computed per tool per image:
  - IoU (Intersection over Union / Jaccard Index)
  - Dice coefficient (F1 score)
  - Precision and Recall
  - Boundary distance (Hausdorff distance)
  - Morphology deviations from GT (area, aspect ratio, circularity, etc.)

Output:
  benchmarking_segment_tools/output/comparison/
    tool_accuracy_boxplots.png      <- IoU, Dice, Precision, Recall per tool
    morphology_comparison.png       <- Side-by-side metric box plots
    radar_chart.png                 <- Spider chart of overall tool performance
    per_sample_iou_heatmap.png      <- Heatmap: tool x sample IoU
    scatter_area_vs_gt.png          <- Predicted vs GT area scatter
    visual_comparison_grid.png      <- Overlay grid: all tools on sample images
    comparison_summary.csv          <- Full numerical results
    tool_ranking.csv                <- Ranked summary table

Usage:
  python benchmarking_segment_tools/compare_tools.py
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from skimage import measure
from scipy import ndimage as ndi
from scipy.spatial.distance import directed_hausdorff
import tifffile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = _SCRIPT_DIR / "output"
COMPARISON_DIR = OUTPUT_BASE / "comparison"

JPG_BASE = Path(r"Z:\Megha_Kattimani\Imaging\Zeiss\Benchmarking_Segmentation tools\JPG")
GT_DIR = JPG_BASE / "model" / "trainingset"
BF_DIR = JPG_BASE / "MORGANA"

# Tool configurations: name -> (mask_dir, color, marker)
TOOLS = {
    'Morgana':   {'dir': OUTPUT_BASE / 'morgana_trained' / 'masks', 'color': '#e41a1c', 'marker': 'o'},
    'Cellpose':  {'dir': OUTPUT_BASE / 'cellpose' / 'masks',        'color': '#377eb8', 'marker': 's'},
    '3D U-Net':  {'dir': OUTPUT_BASE / '3dunet' / 'masks',          'color': '#4daf4a', 'marker': '^'},
    'Ilastik':   {'dir': OUTPUT_BASE / 'ilastik' / 'masks',         'color': '#984ea3', 'marker': 'D'},
}

# Sample mapping
SAMPLE_MAP = {
    '033': 'control', '043': 'control',
    '049': 'treated', '055': 'treated', '061': 'treated',
    '073': 'treated', '075': 'treated', '095': 'treated', '129': 'treated',
}


# ============================================================================
# Load masks
# ============================================================================

def load_gt_masks() -> Dict[str, np.ndarray]:
    """Load all GT masks, keyed by sample name (e.g., '033', '033e')."""
    masks = {}
    for gt_path in sorted(GT_DIR.glob("*_GT.tif")):
        name = gt_path.stem.replace('_GT', '')
        gt = tifffile.imread(str(gt_path))
        masks[name] = (gt > 0).astype(bool)
    return masks


def load_tool_masks(tool_name: str) -> Dict[str, np.ndarray]:
    """Load all predicted masks for a tool, keyed by sample name."""
    tool_info = TOOLS[tool_name]
    mask_dir = tool_info['dir']
    masks = {}

    if not mask_dir.exists():
        print(f"  Warning: {mask_dir} does not exist for {tool_name}")
        return masks

    for mask_path in sorted(mask_dir.glob("*_mask.tif")):
        # Parse filename: either {sample_id}_{condition}_{timepoint}_mask.tif
        # or {name}_mask.tif (morgana_trained)
        stem = mask_path.stem.replace('_mask', '')
        parts = stem.split('_')

        # Try to reconstruct sample name from parts
        if len(parts) >= 3:
            sample_id = parts[0]
            condition = parts[1]
            timepoint = parts[2]
            # Reconstruct name: initial -> just ID, elongated -> ID + 'e'
            if timepoint == 'elongated':
                name = sample_id + 'e'
                # Handle uppercase E variants
                name_upper = sample_id + 'E'
            else:
                name = sample_id
                name_upper = sample_id
        else:
            name = stem
            name_upper = stem

        mask = tifffile.imread(str(mask_path))
        masks[name] = (mask > 0).astype(bool)
        if name_upper != name:
            masks[name_upper] = masks[name]

    return masks


# ============================================================================
# Compute segmentation quality metrics
# ============================================================================

def compute_segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute IoU, Dice, Precision, Recall, Hausdorff between pred and gt masks."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()

    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0

    # Hausdorff distance (boundary-based)
    hausdorff = np.nan
    if pred_sum > 0 and gt_sum > 0:
        pred_boundary = np.argwhere(pred_bool & ~ndi.binary_erosion(pred_bool))
        gt_boundary = np.argwhere(gt_bool & ~ndi.binary_erosion(gt_bool))
        if len(pred_boundary) > 0 and len(gt_boundary) > 0:
            # Subsample for speed if boundaries are large
            if len(pred_boundary) > 500:
                idx = np.random.RandomState(42).choice(len(pred_boundary), 500, replace=False)
                pred_boundary = pred_boundary[idx]
            if len(gt_boundary) > 500:
                idx = np.random.RandomState(42).choice(len(gt_boundary), 500, replace=False)
                gt_boundary = gt_boundary[idx]
            h1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
            h2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
            hausdorff = max(h1, h2)

    # Morphology from pred
    morph = {}
    if pred_sum > 0:
        props = measure.regionprops(pred_bool.astype(int))
        if props:
            p = props[0]
            perimeter = p.perimeter if p.perimeter > 0 else 1e-6
            morph['area'] = int(p.area)
            morph['aspect_ratio'] = float(p.major_axis_length / p.minor_axis_length) if p.minor_axis_length > 0 else 1.0
            morph['circularity'] = float((4 * np.pi * p.area) / (perimeter ** 2))
            morph['solidity'] = float(p.solidity)

    # Morphology from GT
    gt_morph = {}
    if gt_sum > 0:
        props = measure.regionprops(gt_bool.astype(int))
        if props:
            p = props[0]
            perimeter = p.perimeter if p.perimeter > 0 else 1e-6
            gt_morph['gt_area'] = int(p.area)
            gt_morph['gt_aspect_ratio'] = float(p.major_axis_length / p.minor_axis_length) if p.minor_axis_length > 0 else 1.0
            gt_morph['gt_circularity'] = float((4 * np.pi * p.area) / (perimeter ** 2))
            gt_morph['gt_solidity'] = float(p.solidity)

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'hausdorff': hausdorff,
        **morph,
        **gt_morph,
    }


# ============================================================================
# PLOT 1: Accuracy box plots (IoU, Dice, Precision, Recall)
# ============================================================================

def plot_accuracy_boxplots(df: pd.DataFrame):
    """Box plots of IoU, Dice, Precision, Recall per tool."""
    metrics = ['iou', 'dice', 'precision', 'recall']
    titles = ['IoU (Jaccard)', 'Dice (F1)', 'Precision', 'Recall']
    tool_names = list(TOOLS.keys())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        groups = []
        colors = []

        for tool in tool_names:
            vals = df[df['tool'] == tool][metric].dropna().values
            groups.append(vals)
            colors.append(TOOLS[tool]['color'])

        bp = ax.boxplot(groups, tick_labels=tool_names, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')

        # Overlay data points
        for j, (vals, c) in enumerate(zip(groups, colors)):
            if len(vals) > 0:
                x = np.random.normal(j + 1, 0.06, size=len(vals))
                ax.scatter(x, vals, alpha=0.7, s=30, c=c,
                           edgecolors='black', linewidth=0.5, zorder=3)

        # Add mean line
        for j, vals in enumerate(groups):
            if len(vals) > 0:
                mean_val = np.mean(vals)
                ax.hlines(mean_val, j + 0.7, j + 1.3, colors='black',
                          linestyles='--', linewidth=1.5, alpha=0.7)
                ax.text(j + 1.35, mean_val, f'{mean_val:.3f}', fontsize=8,
                        va='center', fontweight='bold')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel('Score' if i == 0 else '')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3)

    plt.suptitle('Segmentation Accuracy — Tool Comparison vs Ground Truth',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(str(COMPARISON_DIR / 'tool_accuracy_boxplots.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: tool_accuracy_boxplots.png")


# ============================================================================
# PLOT 2: Morphology comparison (predicted metrics side-by-side)
# ============================================================================

def plot_morphology_comparison(df: pd.DataFrame):
    """Side-by-side box plots of morphology metrics per tool, with GT reference."""
    metrics = ['area', 'aspect_ratio', 'circularity', 'solidity']
    gt_metrics = ['gt_area', 'gt_aspect_ratio', 'gt_circularity', 'gt_solidity']
    titles = ['Area (pixels)', 'Aspect Ratio', 'Circularity', 'Solidity']
    tool_names = list(TOOLS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (metric, gt_metric, title) in enumerate(zip(metrics, gt_metrics, titles)):
        ax = axes[i]

        # Collect data: GT + each tool
        all_labels = ['GT'] + tool_names
        all_groups = []
        all_colors = ['#333333'] + [TOOLS[t]['color'] for t in tool_names]

        # GT values (unique per sample, not repeated per tool)
        gt_vals = df.drop_duplicates(subset='sample_name')[gt_metric].dropna().values
        all_groups.append(gt_vals)

        for tool in tool_names:
            vals = df[df['tool'] == tool][metric].dropna().values
            all_groups.append(vals)

        bp = ax.boxplot(all_groups, tick_labels=all_labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], all_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')

        for j, (vals, c) in enumerate(zip(all_groups, all_colors)):
            if len(vals) > 0:
                x = np.random.normal(j + 1, 0.06, size=len(vals))
                ax.scatter(x, vals, alpha=0.7, s=25, c=c,
                           edgecolors='black', linewidth=0.5, zorder=3)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Morphology Metrics — GT vs Tool Predictions',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(COMPARISON_DIR / 'morphology_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: morphology_comparison.png")


# ============================================================================
# PLOT 3: Radar / spider chart
# ============================================================================

def plot_radar_chart(df: pd.DataFrame):
    """Spider chart showing mean scores across multiple dimensions per tool."""
    tool_names = list(TOOLS.keys())
    categories = ['IoU', 'Dice', 'Precision', 'Recall',
                  'Area\nAccuracy', 'Shape\nAccuracy']
    N = len(categories)

    # Compute mean scores per tool
    tool_scores = {}
    for tool in tool_names:
        tdf = df[df['tool'] == tool]
        iou = tdf['iou'].mean()
        dice = tdf['dice'].mean()
        precision = tdf['precision'].mean()
        recall = tdf['recall'].mean()

        # Area accuracy: 1 - mean(|pred_area - gt_area| / gt_area)
        area_err = (np.abs(tdf['area'] - tdf['gt_area']) / tdf['gt_area']).mean()
        area_acc = max(0, 1 - area_err)

        # Shape accuracy: average of AR accuracy + circularity accuracy + solidity accuracy
        ar_err = (np.abs(tdf['aspect_ratio'] - tdf['gt_aspect_ratio']) / tdf['gt_aspect_ratio']).mean()
        circ_err = (np.abs(tdf['circularity'] - tdf['gt_circularity']) / tdf['gt_circularity'].clip(0.01)).mean()
        sol_err = (np.abs(tdf['solidity'] - tdf['gt_solidity']) / tdf['gt_solidity'].clip(0.01)).mean()
        shape_acc = max(0, 1 - (ar_err + circ_err + sol_err) / 3)

        tool_scores[tool] = [iou, dice, precision, recall, area_acc, shape_acc]

    # Plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for tool in tool_names:
        values = tool_scores[tool] + tool_scores[tool][:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                color=TOOLS[tool]['color'], label=tool, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=TOOLS[tool]['color'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Overall Tool Performance Radar', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(str(COMPARISON_DIR / 'radar_chart.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: radar_chart.png")


# ============================================================================
# PLOT 4: Per-sample IoU heatmap
# ============================================================================

def plot_iou_heatmap(df: pd.DataFrame):
    """Heatmap showing IoU per tool per sample."""
    tool_names = list(TOOLS.keys())
    sample_names = sorted(df['sample_name'].unique())

    iou_matrix = np.full((len(tool_names), len(sample_names)), np.nan)
    for i, tool in enumerate(tool_names):
        for j, sample in enumerate(sample_names):
            vals = df[(df['tool'] == tool) & (df['sample_name'] == sample)]['iou']
            if len(vals) > 0:
                iou_matrix[i, j] = vals.values[0]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(iou_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Annotations
    for i in range(len(tool_names)):
        for j in range(len(sample_names)):
            val = iou_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(tool_names)))
    ax.set_yticklabels(tool_names, fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('IoU', fontsize=11)

    plt.title('IoU per Sample per Tool', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(COMPARISON_DIR / 'per_sample_iou_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: per_sample_iou_heatmap.png")


# ============================================================================
# PLOT 5: Area scatter (predicted vs GT)
# ============================================================================

def plot_area_scatter(df: pd.DataFrame):
    """Scatter plot of predicted area vs GT area for each tool."""
    tool_names = list(TOOLS.keys())

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect line
    all_areas = pd.concat([df['gt_area'], df['area']]).dropna()
    lim = [0, all_areas.max() * 1.1]
    ax.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.5, label='Perfect (y=x)')
    ax.fill_between(lim, [x * 0.9 for x in lim], [x * 1.1 for x in lim],
                    alpha=0.08, color='green', label='±10% band')

    for tool in tool_names:
        tdf = df[df['tool'] == tool]
        ax.scatter(tdf['gt_area'], tdf['area'],
                   c=TOOLS[tool]['color'], marker=TOOLS[tool]['marker'],
                   s=60, edgecolors='black', linewidth=0.5,
                   alpha=0.8, label=tool, zorder=3)

    ax.set_xlabel('GT Area (pixels)', fontsize=12)
    ax.set_ylabel('Predicted Area (pixels)', fontsize=12)
    ax.set_title('Predicted vs GT Area', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(str(COMPARISON_DIR / 'scatter_area_vs_gt.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: scatter_area_vs_gt.png")


# ============================================================================
# PLOT 6: Visual comparison grid
# ============================================================================

def plot_visual_comparison(df: pd.DataFrame, gt_masks: dict, tool_masks: dict):
    """Grid showing BF + GT + each tool's contour for selected samples."""
    tool_names = list(TOOLS.keys())

    # Pick 4 representative samples (2 initial, 2 elongated)
    samples = ['033', '043E', '055', '075e']
    # Filter to available samples
    available_gt = set(gt_masks.keys())
    samples = [s for s in samples if s in available_gt]
    if len(samples) < 4:
        # Fill with whatever's available
        for s in sorted(available_gt):
            if s not in samples:
                samples.append(s)
            if len(samples) >= 4:
                break

    n_cols = 2 + len(tool_names)  # BF + GT + each tool
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        # Load BF image
        # Try to find the BF image
        bf_path = None
        for tif in BF_DIR.glob("*.tif"):
            if tif.stem.lower() == sample.lower():
                bf_path = tif
                break
        if bf_path is None:
            # Try matching in GT dir
            bf_path_gt = GT_DIR / f"{sample}.tif"
            if bf_path_gt.exists():
                bf_path = bf_path_gt

        if bf_path is not None:
            bf = tifffile.imread(str(bf_path))
        else:
            bf = np.zeros((512, 512), dtype=np.uint8)

        gt = gt_masks.get(sample, np.zeros((512, 512), dtype=bool))

        # Column 0: BF image
        axes[row, 0].imshow(bf, cmap='gray')
        axes[row, 0].set_title('BF' if row == 0 else '', fontsize=10)
        axes[row, 0].set_ylabel(sample, fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # Column 1: GT contour
        axes[row, 1].imshow(bf, cmap='gray')
        if gt.sum() > 0:
            for c in measure.find_contours(gt.astype(float), 0.5):
                axes[row, 1].plot(c[:, 1], c[:, 0], '-', color='lime', linewidth=2)
        axes[row, 1].set_title('Ground Truth' if row == 0 else '', fontsize=10)
        axes[row, 1].axis('off')

        # Columns 2+: each tool
        for col, tool in enumerate(tool_names):
            ax = axes[row, 2 + col]
            ax.imshow(bf, cmap='gray')

            # GT contour (dashed)
            if gt.sum() > 0:
                for c in measure.find_contours(gt.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], '--', color='lime', linewidth=1, alpha=0.6)

            # Tool contour (solid)
            pred = tool_masks[tool].get(sample, np.zeros((512, 512), dtype=bool))
            if pred.sum() > 0:
                for c in measure.find_contours(pred.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], '-', color=TOOLS[tool]['color'], linewidth=2)

            # Compute IoU for label
            if gt.sum() > 0 and pred.sum() > 0:
                inter = (gt & pred).sum()
                union = (gt | pred).sum()
                iou = inter / union if union > 0 else 0
                ax.text(5, 505, f'IoU={iou:.3f}', fontsize=8, color='white',
                        fontweight='bold', va='bottom',
                        bbox=dict(boxstyle='round', facecolor=TOOLS[tool]['color'], alpha=0.7))

            ax.set_title(tool if row == 0 else '', fontsize=10,
                        color=TOOLS[tool]['color'], fontweight='bold')
            ax.axis('off')

    plt.suptitle('Visual Comparison — GT (green dashed) vs Predictions (colored solid)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(str(COMPARISON_DIR / 'visual_comparison_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: visual_comparison_grid.png")


# ============================================================================
# PLOT 7: Ranked summary bar chart
# ============================================================================

def plot_ranking_bars(summary: pd.DataFrame):
    """Horizontal bar chart ranking tools by mean IoU, Dice, etc."""
    summary_sorted = summary.sort_values('mean_iou', ascending=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    metrics = [('mean_iou', 'Mean IoU'), ('mean_dice', 'Mean Dice'),
               ('mean_precision', 'Mean Precision'), ('mean_recall', 'Mean Recall')]

    for ax, (col, title) in zip(axes, metrics):
        sorted_df = summary.sort_values(col, ascending=True)
        colors = [TOOLS[t]['color'] for t in sorted_df['tool']]
        bars = ax.barh(sorted_df['tool'], sorted_df[col], color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, sorted_df[col]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(0, 1.15)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=1.0, color='green', linestyle=':', alpha=0.3)

    plt.suptitle('Tool Ranking by Segmentation Accuracy',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(COMPARISON_DIR / 'tool_ranking_bars.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: tool_ranking_bars.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  CROSS-TOOL SEGMENTATION COMPARISON")
    print("=" * 60)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load GT masks
    print(f"\n{'=' * 60}")
    print("STEP 1: Loading ground truth masks")
    print(f"{'=' * 60}")
    gt_masks = load_gt_masks()
    print(f"  Loaded {len(gt_masks)} GT masks: {sorted(gt_masks.keys())}")

    # Step 2: Load tool masks
    print(f"\n{'=' * 60}")
    print("STEP 2: Loading tool predictions")
    print(f"{'=' * 60}")
    tool_masks = {}
    for tool_name in TOOLS:
        masks = load_tool_masks(tool_name)
        tool_masks[tool_name] = masks
        print(f"  {tool_name}: {len(masks)} masks loaded")

    # Step 3: Compute metrics
    print(f"\n{'=' * 60}")
    print("STEP 3: Computing segmentation metrics vs GT")
    print(f"{'=' * 60}")
    rows = []
    for tool_name in TOOLS:
        for gt_name, gt_mask in gt_masks.items():
            # Find matching prediction
            pred_mask = tool_masks[tool_name].get(gt_name)
            if pred_mask is None:
                continue

            metrics = compute_segmentation_metrics(pred_mask, gt_mask)

            # Determine condition and timepoint
            name_lower = gt_name.lower()
            if name_lower.endswith('e'):
                sample_id = name_lower[:-1]
                timepoint = 'elongated'
            else:
                sample_id = name_lower
                timepoint = 'initial'
            condition = SAMPLE_MAP.get(sample_id, 'unknown')

            metrics['tool'] = tool_name
            metrics['sample_name'] = gt_name
            metrics['sample_id'] = sample_id
            metrics['condition'] = condition
            metrics['timepoint'] = timepoint
            rows.append(metrics)

    df = pd.DataFrame(rows)
    print(f"\n  Total comparisons: {len(df)}")
    for tool in TOOLS:
        n = len(df[df['tool'] == tool])
        mean_iou = df[df['tool'] == tool]['iou'].mean()
        print(f"  {tool}: {n} images, mean IoU={mean_iou:.3f}")

    # Step 4: Generate plots
    print(f"\n{'=' * 60}")
    print("STEP 4: Generating comparison plots")
    print(f"{'=' * 60}")

    plot_accuracy_boxplots(df)
    plot_morphology_comparison(df)
    plot_radar_chart(df)
    plot_iou_heatmap(df)
    plot_area_scatter(df)
    plot_visual_comparison(df, gt_masks, tool_masks)

    # Step 5: Summary table
    print(f"\n{'=' * 60}")
    print("STEP 5: Creating summary tables")
    print(f"{'=' * 60}")

    summary_rows = []
    for tool in TOOLS:
        tdf = df[df['tool'] == tool]
        summary_rows.append({
            'tool': tool,
            'n_images': len(tdf),
            'mean_iou': tdf['iou'].mean(),
            'std_iou': tdf['iou'].std(),
            'mean_dice': tdf['dice'].mean(),
            'std_dice': tdf['dice'].std(),
            'mean_precision': tdf['precision'].mean(),
            'std_precision': tdf['precision'].std(),
            'mean_recall': tdf['recall'].mean(),
            'std_recall': tdf['recall'].std(),
            'mean_hausdorff': tdf['hausdorff'].mean(),
            'std_hausdorff': tdf['hausdorff'].std(),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values('mean_iou', ascending=False)

    plot_ranking_bars(summary)

    # Save CSVs
    df.to_csv(str(COMPARISON_DIR / 'comparison_full.csv'), index=False)
    summary.to_csv(str(COMPARISON_DIR / 'tool_ranking.csv'), index=False)
    print(f"  Saved: comparison_full.csv")
    print(f"  Saved: tool_ranking.csv")

    # Print ranking table
    print(f"\n{'=' * 60}")
    print("  TOOL RANKING (by mean IoU)")
    print(f"{'=' * 60}")
    print(f"\n  {'Tool':<12} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8} {'Hausdorff':>10}")
    print(f"  {'-'*56}")
    for _, row in summary.iterrows():
        print(f"  {row['tool']:<12} {row['mean_iou']:>7.3f}  {row['mean_dice']:>7.3f}  "
              f"{row['mean_precision']:>7.3f}  {row['mean_recall']:>7.3f}  "
              f"{row['mean_hausdorff']:>9.1f}")

    print(f"\n{'=' * 60}")
    print("  DONE!")
    print(f"{'=' * 60}")
    print(f"\nAll outputs in: {COMPARISON_DIR}")


if __name__ == '__main__':
    main()
