"""
Cross-Experiment Comparison — Grouped by Experiment Folder
==========================================================

Reads per-pescoid analysis results from analysis_output/ and groups them
by their parent experiment folder (e.g., P_mezzo_3-5hpf_Act). Each
Concatenated_..._G0XX folder is one pescoid replicate within that experiment.

Experiments:
  P_mezzo_ctrl           — Control (no treatment)
  P_mezzo_3-5hpf_Act     — Activin 3-5 hpf
  P_mezzo_3-5hpf_Chi     — Chiron 3-5 hpf
  P_mezzo_3-5hpf_Act-Chi — Activin+Chiron 3-5 hpf
  P_mezzo_5-7hpf_Act     — Activin 5-7 hpf
  P_mezzo_5-7hpf_Chi     — Chiron 5-7 hpf
  P_mezzo_5-7hpf_Act-Chi — Activin+Chiron 5-7 hpf

Output:
  analysis_output/experiment_comparison/
    experiment_summary.csv
    morphology_by_experiment.png
    mezzo_by_experiment.png
    poles_by_experiment.png
    ar_timecourse_by_experiment.png
    mezzo_timecourse_by_experiment.png
    early_vs_late_treatment.png

Usage:
  python compare_experiments.py
  python compare_experiments.py --data-root "Z:/..." --analysis-dir analysis_output
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"Z:\Megha_Kattimani\segmentation pipeline data\250212_mezzo_3-5hpf_activin_chiron")
ANALYSIS_DIR = Path("analysis_output")
OUT_DIR = ANALYSIS_DIR / "experiment_comparison"

# Experiment display config: short label, color, order
EXP_CONFIG = {
    'P_mezzo_ctrl':           {'label': 'Control',         'color': '#888888', 'order': 0},
    'P_mezzo_3-5hpf_Act':     {'label': 'Act 3-5h',       'color': '#e41a1c', 'order': 1},
    'P_mezzo_3-5hpf_Chi':     {'label': 'Chi 3-5h',       'color': '#377eb8', 'order': 2},
    'P_mezzo_3-5hpf_Act-Chi': {'label': 'Act+Chi 3-5h',   'color': '#984ea3', 'order': 3},
    'P_mezzo_5-7hpf_Act':     {'label': 'Act 5-7h',       'color': '#ff7f00', 'order': 4},
    'P_mezzo_5-7hpf_Chi':     {'label': 'Chi 5-7h',       'color': '#4daf4a', 'order': 5},
    'P_mezzo_5-7hpf_Act-Chi': {'label': 'Act+Chi 5-7h',   'color': '#a65628', 'order': 6},
}


# ============================================================================
# 1. Load all per-pescoid results and map to experiment
# ============================================================================

def load_all_results(data_root, analysis_dir):
    """
    Walk the data_root to find experiment -> pescoid mapping,
    then load results from analysis_dir.
    """
    data_root = Path(data_root)
    analysis_dir = Path(analysis_dir)
    rows = []

    for exp_dir in sorted(data_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'result_segmentation':
            continue
        experiment = exp_dir.name

        for sample_dir in sorted(exp_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pescoid_name = sample_dir.name

            # Extract pescoid ID (e.g., G014 from Concatenated_..._G014_0001)
            match = re.search(r'(G\d+)', pescoid_name)
            pescoid_id = match.group(1) if match else pescoid_name

            # Load summary.json — support both flat and nested layouts
            summary_path = analysis_dir / pescoid_name / 'summary.json'
            if not summary_path.exists():
                summary_path = analysis_dir / experiment / pescoid_name / 'summary.json'
            if not summary_path.exists():
                print(f"  Warning: no results for {pescoid_name}")
                continue

            with open(str(summary_path)) as f:
                summary = json.load(f)

            # Load morphology timeseries — support nested layout
            morph_path = analysis_dir / pescoid_name / 'morphology_over_time.csv'
            if not morph_path.exists():
                morph_path = analysis_dir / experiment / pescoid_name / 'morphology_over_time.csv'
            morph_df = pd.read_csv(str(morph_path)) if morph_path.exists() else None

            # Load mezzo timeseries
            mezzo_path = analysis_dir / pescoid_name / 'mezzo_expression_over_time.csv'
            if not mezzo_path.exists():
                mezzo_path = analysis_dir / experiment / pescoid_name / 'mezzo_expression_over_time.csv'
            mezzo_df = pd.read_csv(str(mezzo_path)) if mezzo_path.exists() else None

            row = {
                'experiment': experiment,
                'pescoid_id': pescoid_id,
                'pescoid_name': pescoid_name,
                'n_timepoints': summary['n_timepoints'],
                'mean_area': summary['morphology']['mean_area'],
                'final_area': summary['morphology']['final_area'],
                'mean_ar': summary['morphology']['mean_aspect_ratio'],
                'final_ar': summary['morphology']['final_aspect_ratio'],
                'mean_circ': summary['morphology']['mean_circularity'],
                'mean_mezzo': summary['mezzo']['mean_fraction'],
                'final_mezzo': summary['mezzo']['final_fraction'],
                'n_poles': summary['poles']['n_poles'],
                'n_outward': summary['poles']['n_outward'],
                'n_inward': summary['poles']['n_inward'],
                'max_rotation_deg': summary['max_rotation_deg'],
                'failed_frames': len(summary['failed_frames']),
                '_morph_df': morph_df,
                '_mezzo_df': mezzo_df,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by experiment order
    df['_order'] = df['experiment'].map(lambda e: EXP_CONFIG.get(e, {}).get('order', 99))
    df = df.sort_values(['_order', 'pescoid_id']).reset_index(drop=True)
    return df


def exp_label(exp_name):
    return EXP_CONFIG.get(exp_name, {}).get('label', exp_name)


def exp_color(exp_name):
    return EXP_CONFIG.get(exp_name, {}).get('color', 'gray')


def sorted_experiments(df):
    """Return experiment names sorted by display order."""
    exps = df['experiment'].unique()
    return sorted(exps, key=lambda e: EXP_CONFIG.get(e, {}).get('order', 99))


# ============================================================================
# 2. Box plots — morphology by experiment
# ============================================================================

def plot_morphology_boxplots(df, out_dir):
    metrics = [
        ('final_ar', 'Final Aspect Ratio'),
        ('mean_area', 'Mean Area (px)'),
        ('mean_circ', 'Mean Circularity'),
        ('final_area', 'Final Area (px)'),
    ]
    experiments = sorted_experiments(df)
    labels = [exp_label(e) for e in experiments]
    colors = [exp_color(e) for e in experiments]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        groups = [df[df['experiment'] == e][metric].dropna().values for e in experiments]

        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')

        for j, (vals, c) in enumerate(zip(groups, colors)):
            if len(vals) > 0:
                x = np.random.default_rng(42).normal(j + 1, 0.06, size=len(vals))
                ax.scatter(x, vals, s=40, c=c, edgecolors='black',
                           linewidth=0.5, zorder=3, alpha=0.8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Morphology — Experiment Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(out_dir / 'morphology_by_experiment.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: morphology_by_experiment.png")


# ============================================================================
# 3. Box plots — mezzo expression by experiment
# ============================================================================

def plot_mezzo_boxplots(df, out_dir):
    experiments = sorted_experiments(df)
    labels = [exp_label(e) for e in experiments]
    colors = [exp_color(e) for e in experiments]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, (metric, title) in enumerate([
        ('mean_mezzo', 'Mean Mezzo Fraction'),
        ('final_mezzo', 'Final Mezzo Fraction'),
    ]):
        ax = axes[i]
        groups = [df[df['experiment'] == e][metric].dropna().values for e in experiments]

        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')

        for j, (vals, c) in enumerate(zip(groups, colors)):
            if len(vals) > 0:
                x = np.random.default_rng(42).normal(j + 1, 0.06, size=len(vals))
                ax.scatter(x, vals, s=40, c=c, edgecolors='black',
                           linewidth=0.5, zorder=3, alpha=0.8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 0.7)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Mezzo-GFP Expression — Experiment Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(out_dir / 'mezzo_by_experiment.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: mezzo_by_experiment.png")


# ============================================================================
# 4. Pole counts by experiment
# ============================================================================

def plot_pole_counts(df, out_dir):
    experiments = sorted_experiments(df)
    labels = [exp_label(e) for e in experiments]
    colors = [exp_color(e) for e in experiments]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: total poles box plot
    ax = axes[0]
    groups = [df[df['experiment'] == e]['n_poles'].values for e in experiments]
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
        patch.set_edgecolor('black')
    for j, (vals, c) in enumerate(zip(groups, colors)):
        if len(vals) > 0:
            x = np.random.default_rng(42).normal(j + 1, 0.06, size=len(vals))
            ax.scatter(x, vals, s=40, c=c, edgecolors='black',
                       linewidth=0.5, zorder=3, alpha=0.8)
    ax.set_title('Total Poles per Pescoid', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Right: stacked bar (outward vs inward)
    ax = axes[1]
    x = np.arange(len(experiments))
    width = 0.6
    outward = [df[df['experiment'] == e]['n_outward'].mean() for e in experiments]
    inward = [df[df['experiment'] == e]['n_inward'].mean() for e in experiments]
    ax.bar(x, outward, width, label='Outward', color='#e41a1c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x, inward, width, bottom=outward, label='Inward', color='#377eb8', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Pole Count')
    ax.set_title('Outward vs Inward Poles', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Pole Detection — Experiment Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(out_dir / 'poles_by_experiment.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: poles_by_experiment.png")


# ============================================================================
# 5. Timecourse plots — AR and mezzo over time (mean ± SEM per experiment)
# ============================================================================

def plot_timecourse(df, out_dir):
    experiments = sorted_experiments(df)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Aspect Ratio timecourse ---
    ax = axes[0]
    for exp in experiments:
        exp_df = df[df['experiment'] == exp]
        # Collect all morph timeseries
        all_ar = []
        for _, row in exp_df.iterrows():
            mdf = row['_morph_df']
            if mdf is not None and 'aspect_ratio' in mdf.columns:
                all_ar.append(mdf['aspect_ratio'].values)
        if not all_ar:
            continue
        # Pad to same length
        max_t = max(len(a) for a in all_ar)
        padded = np.full((len(all_ar), max_t), np.nan)
        for i, a in enumerate(all_ar):
            padded[i, :len(a)] = a

        mean_ar = np.nanmean(padded, axis=0)
        sem_ar = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0).clip(1))
        t = np.arange(max_t)
        c = exp_color(exp)
        ax.plot(t, mean_ar, '-', color=c, linewidth=2, label=exp_label(exp))
        ax.fill_between(t, mean_ar - sem_ar, mean_ar + sem_ar, color=c, alpha=0.15)

    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Aspect Ratio', fontsize=12)
    ax.set_title('Aspect Ratio over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # --- Mezzo timecourse ---
    ax = axes[1]
    for exp in experiments:
        exp_df = df[df['experiment'] == exp]
        all_mezzo = []
        for _, row in exp_df.iterrows():
            mdf = row['_mezzo_df']
            if mdf is not None and 'mezzo_fraction' in mdf.columns:
                all_mezzo.append(mdf['mezzo_fraction'].values)
        if not all_mezzo:
            continue
        max_t = max(len(a) for a in all_mezzo)
        padded = np.full((len(all_mezzo), max_t), np.nan)
        for i, a in enumerate(all_mezzo):
            padded[i, :len(a)] = a

        mean_m = np.nanmean(padded, axis=0)
        sem_m = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0).clip(1))
        t = np.arange(max_t)
        c = exp_color(exp)
        ax.plot(t, mean_m, '-', color=c, linewidth=2, label=exp_label(exp))
        ax.fill_between(t, mean_m - sem_m, mean_m + sem_m, color=c, alpha=0.15)

    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Mezzo Fraction', fontsize=12)
    ax.set_title('Mezzo-GFP Expression over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(-0.05, 0.7)
    ax.grid(alpha=0.3)

    plt.suptitle('Time-Course — Mean ± SEM across Pescoids', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(out_dir / 'timecourse_by_experiment.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: timecourse_by_experiment.png")


# ============================================================================
# 6. Early vs Late treatment comparison
# ============================================================================

def plot_early_vs_late(df, out_dir):
    """Compare 3-5hpf vs 5-7hpf treatment for each drug."""
    drugs = ['Act', 'Chi', 'Act-Chi']
    metrics = [
        ('final_ar', 'Final Aspect Ratio'),
        ('final_mezzo', 'Final Mezzo Fraction'),
        ('n_poles', 'Pole Count'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(drugs))
        width = 0.3

        # Control reference line
        ctrl_mean = df[df['experiment'] == 'P_mezzo_ctrl'][metric].mean()
        ax.axhline(y=ctrl_mean, color='gray', linestyle='--', alpha=0.6, label='Control')

        early_means, early_sems, late_means, late_sems = [], [], [], []
        for drug in drugs:
            early_exp = f'P_mezzo_3-5hpf_{drug}'
            late_exp = f'P_mezzo_5-7hpf_{drug}'

            e_vals = df[df['experiment'] == early_exp][metric].dropna()
            l_vals = df[df['experiment'] == late_exp][metric].dropna()

            early_means.append(e_vals.mean() if len(e_vals) > 0 else 0)
            early_sems.append(e_vals.sem() if len(e_vals) > 1 else 0)
            late_means.append(l_vals.mean() if len(l_vals) > 0 else 0)
            late_sems.append(l_vals.sem() if len(l_vals) > 1 else 0)

        bars1 = ax.bar(x - width/2, early_means, width, yerr=early_sems,
                        label='3-5 hpf (early)', color='#e41a1c', alpha=0.7,
                        edgecolor='black', linewidth=0.5, capsize=4)
        bars2 = ax.bar(x + width/2, late_means, width, yerr=late_sems,
                        label='5-7 hpf (late)', color='#377eb8', alpha=0.7,
                        edgecolor='black', linewidth=0.5, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(drugs, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Early vs Late Treatment Window — 3-5 hpf vs 5-7 hpf',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(out_dir / 'early_vs_late_treatment.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: early_vs_late_treatment.png")


# ============================================================================
# 7. Per-experiment pescoid summary table
# ============================================================================

def save_summary_table(df, out_dir):
    """Save a clean summary CSV grouped by experiment."""
    # Drop internal columns
    export_cols = [c for c in df.columns if not c.startswith('_')]
    clean = df[export_cols].copy()
    clean.insert(0, 'experiment_label', clean['experiment'].map(exp_label))
    clean.to_csv(str(out_dir / 'experiment_summary.csv'), index=False)

    # Aggregated table
    agg = df.groupby('experiment').agg(
        n_pescoids=('pescoid_id', 'count'),
        mean_ar=('mean_ar', 'mean'),
        sem_ar=('mean_ar', 'sem'),
        final_ar=('final_ar', 'mean'),
        mean_mezzo=('mean_mezzo', 'mean'),
        sem_mezzo=('mean_mezzo', 'sem'),
        final_mezzo=('final_mezzo', 'mean'),
        mean_area=('mean_area', 'mean'),
        mean_poles=('n_poles', 'mean'),
        sem_poles=('n_poles', 'sem'),
    ).round(4)
    agg.insert(0, 'label', agg.index.map(exp_label))
    agg['_order'] = agg.index.map(lambda e: EXP_CONFIG.get(e, {}).get('order', 99))
    agg = agg.sort_values('_order').drop(columns='_order')
    agg.to_csv(str(out_dir / 'experiment_aggregated.csv'))
    print("  Saved: experiment_summary.csv")
    print("  Saved: experiment_aggregated.csv")
    return agg


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-experiment comparison')
    parser.add_argument('--data-root', default=str(DATA_ROOT), help='Root data directory')
    parser.add_argument('--analysis-dir', default=str(ANALYSIS_DIR), help='Analysis output directory')
    args = parser.parse_args()

    out_dir = Path(args.analysis_dir) / 'experiment_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CROSS-EXPERIMENT COMPARISON")
    print("=" * 60)

    # Load all results
    print(f"\nLoading results from {args.analysis_dir}...")
    df = load_all_results(args.data_root, args.analysis_dir)
    print(f"  Loaded {len(df)} pescoids across {df['experiment'].nunique()} experiments:")
    for exp in sorted_experiments(df):
        n = len(df[df['experiment'] == exp])
        print(f"    {exp_label(exp):20s} ({exp}): {n} pescoids")

    # Generate plots
    print(f"\nGenerating comparison plots...")
    plot_morphology_boxplots(df, out_dir)
    plot_mezzo_boxplots(df, out_dir)
    plot_pole_counts(df, out_dir)
    plot_timecourse(df, out_dir)
    plot_early_vs_late(df, out_dir)

    # Summary tables
    print(f"\nSaving summary tables...")
    agg = save_summary_table(df, out_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Experiment':<22s} {'N':>3s} {'AR':>7s} {'Mezzo':>7s} {'Poles':>7s} {'Area':>9s}")
    print(f"{'-'*58}")
    for exp in sorted_experiments(df):
        edf = df[df['experiment'] == exp]
        print(f"  {exp_label(exp):<20s} {len(edf):>3d} "
              f"{edf['final_ar'].mean():>6.2f}  "
              f"{edf['final_mezzo'].mean():>6.3f}  "
              f"{edf['n_poles'].mean():>5.1f}  "
              f"{edf['mean_area'].mean():>8.0f}")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == '__main__':
    main()
