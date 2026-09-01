"""
Aggregation Module

Consolidates per-pescoid analyses into condition-level summaries.
Generates comparative statistics, plots, and export formats.

Workflow:
1. Collect all per-pescoid result CSVs
2. Group by experimental condition
3. Compute descriptive statistics (mean, std, median, quartiles)
4. Export summary tables and figures
5. Generate comparative plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def load_pescoid_results(result_csv_path):
    """
    Load single pescoid result CSV.
    
    Expected format: rows = different measurements/timepoints,
                     columns = metric names
    
    Parameters
    ----------
    result_csv_path : str or Path
        Path to pescoid result CSV
    
    Returns
    -------
    pd.DataFrame
        Loaded results
    """
    
    return pd.read_csv(result_csv_path, index_col=0) if Path(result_csv_path).exists() else pd.DataFrame()


def aggregate_condition_results(condition_folder, condition_name=None):
    """
    Aggregate all pescoids from one experimental condition.
    
    Parameters
    ----------
    condition_folder : str or Path
        Path to condition result folder containing per-pescoid CSVs
    condition_name : str, optional
        Label for this condition (defaults to folder name)
    
    Returns
    -------
    dict
        - 'condition_name': condition label
        - 'n_pescoids': number analyzed
        - 'metrics_per_pescoid': list of (pescoid_id, metric_dict) tuples
        - 'summary_stats': per-metric statistics
    """
    
    condition_folder = Path(condition_folder)
    if not condition_folder.exists():
        return {
            'condition_name': condition_name or 'unknown',
            'n_pescoids': 0,
            'metrics_per_pescoid': [],
            'summary_stats': {},
        }
    
    # Find all result CSVs
    result_files = sorted(condition_folder.glob('*_results.csv'))
    
    metrics_per_pescoid = []
    all_metrics = {}
    
    for csv_path in result_files:
        pescoid_id = csv_path.stem.replace('_results', '')
        df = load_pescoid_results(csv_path)
        
        if not df.empty:
            # Flatten if multiple rows
            metric_dict = df.iloc[0].to_dict() if len(df) > 0 else {}
            metrics_per_pescoid.append((pescoid_id, metric_dict))
            
            # Collect values by metric
            for col in df.columns:
                if col not in all_metrics:
                    all_metrics[col] = []
                all_metrics[col].extend(df[col].values)
    
    # Compute statistics
    summary_stats = {}
    for metric, values in all_metrics.items():
        valid = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
        if len(valid) > 0:
            summary_stats[metric] = {
                'count': len(valid),
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid)),
                'median': float(np.median(valid)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'q25': float(np.percentile(valid, 25)),
                'q75': float(np.percentile(valid, 75)),
            }
    
    return {
        'condition_name': condition_name or condition_folder.name,
        'n_pescoids': len(metrics_per_pescoid),
        'metrics_per_pescoid': metrics_per_pescoid,
        'summary_stats': summary_stats,
    }


def create_summary_table(condition_aggregates, metrics_to_include=None):
    """
    Create summary DataFrame across multiple conditions.
    
    Parameters
    ----------
    condition_aggregates : list of dict
        Output from aggregate_condition_results() for multiple conditions
    metrics_to_include : list, optional
        Subset of metrics to include (default: all)
    
    Returns
    -------
    pd.DataFrame
        Rows = metrics, columns = condition-level statistics
    """
    
    rows = []
    
    for cond in condition_aggregates:
        for metric, stats in cond.get('summary_stats', {}).items():
            if metrics_to_include and metric not in metrics_to_include:
                continue
            
            row = {
                'condition': cond['condition_name'],
                'metric': metric,
                'n_pescoids': cond['n_pescoids'],
                **stats,
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def export_condition_summary_json(condition_data, output_path):
    """
    Export condition summary to JSON format.
    
    Parameters
    ----------
    condition_data : dict
        Output from aggregate_condition_results()
    output_path : str or Path
        Path to save JSON
    """
    
    # Convert to JSON-serializable format
    output = {
        'condition_name': condition_data['condition_name'],
        'n_pescoids': condition_data['n_pescoids'],
        'summary_stats': condition_data['summary_stats'],
        'pescoid_ids': [pid for pid, _ in condition_data['metrics_per_pescoid']],
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def compute_condition_comparisons(condition_aggregates, metrics_list=None):
    """
    Compute pairwise comparisons between conditions (e.g., mean differences).
    
    Parameters
    ----------
    condition_aggregates : list of dict
        Aggregated data for each condition
    metrics_list : list, optional
        Metrics to compare (default: all)
    
    Returns
    -------
    dict
        - 'pairwise_differences': (n_conditions, n_conditions, n_metrics) array
        - 'metric_names': metric labels
    """
    
    if len(condition_aggregates) < 2:
        return {
            'pairwise_differences': np.array([]),
            'metric_names': [],
        }
    
    # Get all metric names
    all_metrics = set()
    for cond in condition_aggregates:
        all_metrics.update(cond['summary_stats'].keys())
    
    if metrics_list:
        all_metrics = set(metrics_list) & all_metrics
    
    metric_names = sorted(all_metrics)
    n_conds = len(condition_aggregates)
    
    # Extract means
    means = []
    for cond in condition_aggregates:
        cond_means = []
        for metric in metric_names:
            stats = cond['summary_stats'].get(metric, {})
            cond_means.append(stats.get('mean', 0.0))
        means.append(cond_means)
    
    means = np.array(means)
    
    # Pairwise differences
    diffs = np.zeros((n_conds, n_conds, len(metric_names)))
    for i in range(n_conds):
        for j in range(n_conds):
            diffs[i, j, :] = means[j, :] - means[i, :]
    
    return {
        'pairwise_differences': diffs,
        'metric_names': metric_names,
        'condition_names': [c['condition_name'] for c in condition_aggregates],
    }


def export_summary_csv(condition_aggregates, output_path):
    """
    Export summary table to CSV.
    
    Parameters
    ----------
    condition_aggregates : list of dict
        Aggregated results per condition
    output_path : str or Path
        Output CSV path
    """
    
    summary_df = create_summary_table(condition_aggregates)
    summary_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    # Synthetic test: mock aggregation
    
    # Create fake condition aggregates
    mock_conditions = [
        {
            'condition_name': '3-5hpf_Activin',
            'n_pescoids': 10,
            'metrics_per_pescoid': [],
            'summary_stats': {
                'volume': {
                    'count': 10,
                    'mean': 5e6,
                    'std': 5e5,
                    'median': 5.1e6,
                    'min': 4.2e6,
                    'max': 5.9e6,
                    'q25': 4.8e6,
                    'q75': 5.3e6,
                },
                'sphericity': {
                    'count': 10,
                    'mean': 0.15,
                    'std': 0.05,
                    'median': 0.14,
                    'min': 0.08,
                    'max': 0.25,
                    'q25': 0.11,
                    'q75': 0.18,
                },
            },
        },
        {
            'condition_name': '3-5hpf_Chiron',
            'n_pescoids': 12,
            'metrics_per_pescoid': [],
            'summary_stats': {
                'volume': {
                    'count': 12,
                    'mean': 4.8e6,
                    'std': 6e5,
                    'median': 4.9e6,
                    'min': 3.9e6,
                    'max': 6.0e6,
                    'q25': 4.5e6,
                    'q75': 5.2e6,
                },
                'sphericity': {
                    'count': 12,
                    'mean': 0.18,
                    'std': 0.06,
                    'median': 0.17,
                    'min': 0.09,
                    'max': 0.28,
                    'q25': 0.13,
                    'q75': 0.22,
                },
            },
        },
    ]
    
    # Test comparison
    comp = compute_condition_comparisons(mock_conditions, ['volume', 'sphericity'])
    print("Pairwise comparisons:")
    for i, c1 in enumerate(comp['condition_names']):
        for j, c2 in enumerate(comp['condition_names']):
            print(f"  {c1} → {c2}: volume diff = {comp['pairwise_differences'][i, j, 0]:.0f}")
    
    # Test summary table
    summary = create_summary_table(mock_conditions)
    print("\nSummary table shape:", summary.shape)
    print(summary.head())
