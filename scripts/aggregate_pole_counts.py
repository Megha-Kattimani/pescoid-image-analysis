"""
Aggregate pole counts from multiple pescoid analyses.
"""

import os
import json
import pandas as pd
from pathlib import Path


def aggregate_pole_counts(analysis_root):
    """
    Aggregate pole detection results from all analyzed folders.
    
    Parameters
    ----------
    analysis_root : str
        Root directory containing analysis subfolders.
    
    Returns
    -------
    pd.DataFrame
        Summary dataframe with pole counts per pescoid.
    """
    results = []
    
    analysis_path = Path(analysis_root)
    
    for folder in analysis_path.iterdir():
        if not folder.is_dir():
            continue
        
        summary_file = folder / 'summary.json'
        pole_csv = folder / 'pole_counts.csv'
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            pescoid_name = summary.get('folder', folder.name)
            # Extract meaningful name from path
            if 'Concatenated' in pescoid_name:
                parts = pescoid_name.split('\\')
                for part in parts:
                    if 'Concatenated' in part:
                        pescoid_name = part.replace('result_segmentation', '').strip('\\')
                        break
            
            pole_data = summary.get('pole_detection', {})
            
            result = {
                'pescoid': pescoid_name,
                'folder': folder.name,
                'n_timepoints': len(summary.get('perimeters', [])),
                'baseline_perimeter': summary.get('baseline_perimeter', 0),
                'final_perimeter': summary.get('perimeters', [0])[-1] if summary.get('perimeters') else 0,
                'total_poles': pole_data.get('n_poles', 0),
                'outward_poles': pole_data.get('n_outward', 0),
                'inward_poles': pole_data.get('n_inward', 0),
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('pescoid')
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate pole counts from analyses')
    parser.add_argument('analysis_root', help='Root directory with analysis subfolders')
    parser.add_argument('--output', default='pole_summary.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    df = aggregate_pole_counts(args.analysis_root)
    
    print("\n=== Pole Detection Summary ===")
    print(df.to_string(index=False))
    print(f"\nTotal pescoids analyzed: {len(df)}")
    print(f"Average poles per pescoid: {df['total_poles'].mean():.2f}")
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
