"""
Publication-Quality Experiment Comparison Plots
================================================

Generates smooth time-course plots matching the R-style plots:
  1. Aspect Ratio per Condition over Time [hpf]
  2. Major Axis Length per Condition over Time [hpf]
  3. Normalised Fluorescence per Condition over Time [hpf]
  4. Aspect Ratio over Major Axis Length per Condition
  5. Fluorescence over Aspect Ratio per Condition
  6. Fluorescence over Major Axis Length per Condition

Also organizes per-experiment output folders.

Usage:
  python plot_experiment_comparison.py
  python plot_experiment_comparison.py --analysis-dir analysis_output_v3
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import UnivariateSpline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"Z:\Megha_Kattimani\segmentation pipeline data\250212_mezzo_3-5hpf_activin_chiron")
ANALYSIS_DIR = Path("analysis_output_v3")

# Timing: 6 hpf start, 31 timepoints, ~24 min intervals
HPF_START = 6.0
HPF_INTERVAL = 0.4  # hours per frame (24 min)

# Pixel to micron conversion (10x objective)
PIXEL_TO_UM = 1.29  # approx for 10x Zeiss

# Experiment display config
EXP_CONFIG = {
    'P_mezzo_ctrl':           {'label': 'Control',           'color': '#999999', 'lw': 2.5},
    'P_mezzo_3-5hpf_Act':     {'label': '3-5 hpf, Act',     'color': '#FF00FF', 'lw': 2.5},
    'P_mezzo_3-5hpf_Chi':     {'label': '3-5 hpf, Chi',     'color': '#FFD700', 'lw': 2.5},
    'P_mezzo_3-5hpf_Act-Chi': {'label': '3-5 hpf, Act, Chi','color': '#00DDDD', 'lw': 2.5},
    'P_mezzo_5-7hpf_Act':     {'label': '5-7 hpf, Act',     'color': '#660066', 'lw': 2.5},
    'P_mezzo_5-7hpf_Chi':     {'label': '5-7 hpf, Chi',     'color': '#FF8C00', 'lw': 2.5},
    'P_mezzo_5-7hpf_Act-Chi': {'label': '5-7 hpf, Act, Chi','color': '#006666', 'lw': 2.5},
}

EXP_ORDER = list(EXP_CONFIG.keys())


def load_all_timeseries(data_root, analysis_dir, hpf_start=6.0, hpf_interval=0.4,
                        pixel_to_um=1.29):
    """Load morphology + mezzo timeseries for all pescoids, grouped by experiment."""
    data_root = Path(data_root)
    analysis_dir = Path(analysis_dir)
    experiments = {}

    for exp_dir in sorted(data_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'result_segmentation':
            continue
        exp_name = exp_dir.name
        experiments[exp_name] = []

        for sample_dir in sorted(exp_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pescoid_name = sample_dir.name
            pid = re.search(r'(G\d+)', pescoid_name)
            pid = pid.group(1) if pid else pescoid_name

            # Support both flat (analysis_dir/<sample>/) and nested (analysis_dir/<experiment>/<sample>/)
            morph_path = analysis_dir / pescoid_name / 'morphology_over_time.csv'
            mezzo_path = analysis_dir / pescoid_name / 'mezzo_expression_over_time.csv'
            if not morph_path.exists():
                # Try nested: analysis_dir/<experiment>/<sample>/
                morph_path = analysis_dir / exp_name / pescoid_name / 'morphology_over_time.csv'
                mezzo_path = analysis_dir / exp_name / pescoid_name / 'mezzo_expression_over_time.csv'
            if not morph_path.exists():
                continue

            morph = pd.read_csv(str(morph_path))
            mezzo = pd.read_csv(str(mezzo_path)) if mezzo_path.exists() else None

            # Convert time to hpf
            morph['hpf'] = hpf_start + morph['time'] * hpf_interval
            # Convert major_axis to microns
            morph['major_axis_um'] = morph['major_axis'] * pixel_to_um

            if mezzo is not None:
                mezzo['hpf'] = hpf_start + mezzo['time'] * hpf_interval
                # Normalise fluorescence to percentage
                mezzo['fluorescence_pct'] = mezzo['mezzo_fraction'] * 100

            experiments[exp_name].append({
                'pid': pid,
                'name': pescoid_name,
                'morph': morph,
                'mezzo': mezzo,
            })

    return experiments


def smooth_curve(x, y, s_factor=0.5):
    """LOESS-like smoothing using spline. Returns smooth x, y."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 4:
        return x[valid], y[valid]
    xs, ys = x[valid], y[valid]
    # Sort by x
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    try:
        spl = UnivariateSpline(xs, ys, s=len(xs) * s_factor)
        x_smooth = np.linspace(xs.min(), xs.max(), 200)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth
    except Exception:
        return xs, ys


def compute_experiment_mean(pescoids, x_col, y_col, source='morph', smooth=True):
    """Compute mean + SEM across pescoids for a given x-y pair."""
    all_x = []
    all_y = []
    for p in pescoids:
        df = p[source]
        if df is None or x_col not in df.columns or y_col not in df.columns:
            continue
        all_x.append(df[x_col].values)
        all_y.append(df[y_col].values)

    if not all_y:
        return None, None, None, None

    # Align to common x grid
    max_len = max(len(y) for y in all_y)
    y_padded = np.full((len(all_y), max_len), np.nan)
    x_ref = all_x[0][:max_len] if len(all_x[0]) >= max_len else all_x[np.argmax([len(a) for a in all_x])][:max_len]

    for i, y in enumerate(all_y):
        y_padded[i, :len(y)] = y

    mean_y = np.nanmean(y_padded, axis=0)
    sem_y = np.nanstd(y_padded, axis=0) / np.sqrt(np.sum(~np.isnan(y_padded), axis=0).clip(1))

    if smooth and len(x_ref) > 4:
        x_sm, mean_sm = smooth_curve(x_ref, mean_y, s_factor=0.3)
        _, upper_sm = smooth_curve(x_ref, mean_y + sem_y, s_factor=0.5)
        _, lower_sm = smooth_curve(x_ref, mean_y - sem_y, s_factor=0.5)
        return x_sm, mean_sm, lower_sm, upper_sm
    return x_ref, mean_y, mean_y - sem_y, mean_y + sem_y


def setup_plot_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#F5F5F5',
        'axes.grid': True,
        'grid.color': 'white',
        'grid.linewidth': 1.2,
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })


def make_legend(ax):
    """Create experiment legend."""
    handles = []
    for exp in EXP_ORDER:
        cfg = EXP_CONFIG[exp]
        handles.append(Line2D([0], [0], color=cfg['color'], lw=cfg['lw'], label=cfg['label']))
    ax.legend(handles=handles, title='Pulse', title_fontsize=12,
              loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_metric_over_time(experiments, y_col, ylabel, title, save_path,
                          source='morph', smooth=True, ylim=None):
    """Generic: metric vs time [hpf]."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for exp in EXP_ORDER:
        if exp not in experiments or not experiments[exp]:
            continue
        cfg = EXP_CONFIG[exp]
        x, mean, lo, hi = compute_experiment_mean(
            experiments[exp], 'hpf', y_col, source=source, smooth=smooth)
        if x is None:
            continue
        ax.plot(x, mean, color=cfg['color'], lw=cfg['lw'], label=cfg['label'])

    ax.set_xlabel('Time [hpf]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    make_legend(ax)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()


def plot_y_over_x(experiments, x_col, y_col, xlabel, ylabel, title, save_path,
                  source_x='morph', source_y='morph', smooth=True):
    """Generic: y_col vs x_col across conditions (scatter + smooth)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for exp in EXP_ORDER:
        if exp not in experiments or not experiments[exp]:
            continue
        cfg = EXP_CONFIG[exp]

        # Collect all (x, y) pairs across pescoids
        all_x, all_y = [], []
        for p in experiments[exp]:
            df_x = p[source_x]
            df_y = p[source_y] if source_y != source_x else df_x
            if df_x is None or df_y is None:
                continue
            if x_col not in df_x.columns or y_col not in df_y.columns:
                continue
            xv = df_x[x_col].values
            yv = df_y[y_col].values
            n = min(len(xv), len(yv))
            all_x.append(xv[:n])
            all_y.append(yv[:n])

        if not all_x:
            continue

        # Concatenate and sort by x for smoothing
        cat_x = np.concatenate(all_x)
        cat_y = np.concatenate(all_y)
        valid = ~(np.isnan(cat_x) | np.isnan(cat_y))
        cat_x, cat_y = cat_x[valid], cat_y[valid]

        if len(cat_x) < 4:
            continue

        x_sm, y_sm = smooth_curve(cat_x, cat_y, s_factor=0.3)
        ax.plot(x_sm, y_sm, color=cfg['color'], lw=cfg['lw'], label=cfg['label'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    make_legend(ax)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# ORGANIZE PER-EXPERIMENT FOLDERS
# ============================================================================

def organize_per_experiment(data_root, analysis_dir):
    """Copy/link results into per-experiment folders for easy browsing."""
    data_root = Path(data_root)
    analysis_dir = Path(analysis_dir)

    for exp_dir in sorted(data_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'result_segmentation':
            continue

        exp_out = analysis_dir / 'by_experiment' / exp_dir.name
        exp_out.mkdir(parents=True, exist_ok=True)

        for sample_dir in sorted(exp_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            src = analysis_dir / sample_dir.name
            if not src.exists():
                continue

            pid = re.search(r'(G\d+)', sample_dir.name)
            pid = pid.group(1) if pid else sample_dir.name
            dst = exp_out / pid
            dst.mkdir(exist_ok=True)

            # Copy key files
            import shutil
            for fname in ['morphology_over_time.csv', 'mezzo_expression_over_time.csv',
                          'summary.json', 'pole_counts.csv']:
                s = src / fname
                if s.exists():
                    shutil.copy2(str(s), str(dst / fname))

            # Copy plots
            plots_src = src / 'plots'
            if plots_src.exists():
                plots_dst = dst / 'plots'
                plots_dst.mkdir(exist_ok=True)
                for f in plots_src.glob('*.png'):
                    shutil.copy2(str(f), str(plots_dst / f.name))

    print("  Organized per-experiment folders")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', default=str(ANALYSIS_DIR))
    parser.add_argument('--data-root', default=str(DATA_ROOT))
    parser.add_argument('--hpf-start', type=float, default=HPF_START)
    parser.add_argument('--hpf-interval', type=float, default=HPF_INTERVAL)
    parser.add_argument('--pixel-to-um', type=float, default=PIXEL_TO_UM)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out = analysis_dir / 'comparison'
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PUBLICATION-QUALITY EXPERIMENT COMPARISON")
    print("=" * 60)

    hpf_start = args.hpf_start
    hpf_interval = args.hpf_interval
    pixel_to_um = args.pixel_to_um

    print("\nLoading timeseries data...")
    experiments = load_all_timeseries(args.data_root, args.analysis_dir,
                                     hpf_start=hpf_start, hpf_interval=hpf_interval,
                                     pixel_to_um=pixel_to_um)
    for exp, pescoids in experiments.items():
        label = EXP_CONFIG.get(exp, {}).get('label', exp)
        print(f"  {label:25s}: {len(pescoids)} pescoids")

    print("\nGenerating plots...")

    # 1. Aspect Ratio over Time
    plot_metric_over_time(
        experiments, 'aspect_ratio', 'Aspect Ratio',
        'Aspect Ratio per Condition',
        out / 'aspect_ratio_over_time.png')
    print("  Saved: aspect_ratio_over_time.png")

    # 2. Major Axis Length over Time
    plot_metric_over_time(
        experiments, 'major_axis_um', 'Major Axis Length [um]',
        'Major Axis Length per Condition',
        out / 'major_axis_over_time.png')
    print("  Saved: major_axis_over_time.png")

    # 3. Normalised Fluorescence over Time
    plot_metric_over_time(
        experiments, 'fluorescence_pct', 'Fluorescence [%]',
        'Normalised Fluorescence (Area) per Condition',
        out / 'fluorescence_over_time.png',
        source='mezzo', ylim=(-5, 65))
    print("  Saved: fluorescence_over_time.png")

    # 4. Aspect Ratio over Major Axis Length
    plot_y_over_x(
        experiments, 'major_axis_um', 'aspect_ratio',
        'Major Axis Length [um]', 'Aspect Ratio',
        'Aspect Ratio over Major Axis Length per Condition',
        out / 'ar_over_major_axis.png')
    print("  Saved: ar_over_major_axis.png")

    # 5. Fluorescence over Aspect Ratio
    plot_y_over_x(
        experiments, 'aspect_ratio', 'fluorescence_pct',
        'Aspect Ratio', 'Fluorescence [%]',
        'Fluorescence over Aspect Ratio per Condition',
        out / 'fluorescence_over_ar.png',
        source_x='morph', source_y='mezzo')
    print("  Saved: fluorescence_over_ar.png")

    # 6. Fluorescence over Major Axis Length
    plot_y_over_x(
        experiments, 'major_axis_um', 'fluorescence_pct',
        'Major Axis Length [um]', 'Fluorescence [%]',
        'Fluorescence over Major Axis Length per Condition',
        out / 'fluorescence_over_major_axis.png',
        source_x='morph', source_y='mezzo')
    print("  Saved: fluorescence_over_major_axis.png")

    # Organize per-experiment folders
    print("\nOrganizing per-experiment folders...")
    organize_per_experiment(args.data_root, args.analysis_dir)

    print(f"\nAll plots in: {out}/")
    print(f"Per-experiment data in: {analysis_dir / 'by_experiment'}/")


if __name__ == '__main__':
    main()
