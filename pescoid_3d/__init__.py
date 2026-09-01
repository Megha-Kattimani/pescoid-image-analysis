"""
3D Pescoid Analysis Pipeline

Hierarchical structure:
- pescoid_3d/
  - segmentation/ - 3D body segmentation, volume, surface area
  - fluorescence/ - mezzo, nuclei, membrane signal analysis
  - poles/ - 3D pole detection, characterization
  - tissue_flow/ - 2D/3D tissue flow, nematic order, connectivity
  - regionalization/ - anterior/posterior segmentation
  - data_loader/ - experiment folder parsing, multi-channel handling
  - aggregation/ - summaries, comparative analyses

Main analysis flow:
1. Load 3D image stacks (BF, mezzo, nuclei, membrane channels)
2. Segment pescoid body (3D)
3. Compute 3D properties (volume, surface area, roundness)
4. Extract fluorescence metrics (fraction, intensity distributions)
5. Detect 3D poles and characterize
6. Compute tissue flow fields and nematic order
7. Regionalize and analyze by position and condition
8. Aggregate and generate comparative summaries
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Import all analysis modules
try:
    from . import data_loader
    from . import segmentation
    from . import fluorescence
    from . import poles
    from . import tissue_flow
    from . import regionalization
    from . import aggregation
except ImportError:
    # Fallback for direct imports
    pass


class PescoidExperiment:
    """Container for a single pescoid experiment across time and channels."""
    
    def __init__(self, exp_id, condition, timepoint, channels_dict):
        """
        Parameters
        ----------
        exp_id : str
            Experiment identifier (e.g., "G017_ctrl")
        condition : str
            Condition label (e.g., "Activin", "Chiron", "ctrl")
        timepoint : float
            Time in hpf or minutes
        channels_dict : dict
            Dict mapping channel names to 3D arrays (z, y, x)
            e.g., {'brightfield': array, 'mezzo': array, 'nuclei': array, 'membrane': array}
        """
        self.exp_id = exp_id
        self.condition = condition
        self.timepoint = timepoint
        self.channels = channels_dict
        
        # Placeholder results
        self.results = {
            'segmentation': {},
            'fluorescence': {},
            'poles': {},
            'flow': {},
            'regionalization': {},
        }
    
    def add_segmentation_result(self, key, value):
        self.results['segmentation'][key] = value
    
    def add_fluorescence_result(self, key, value):
        self.results['fluorescence'][key] = value
    
    def add_poles_result(self, key, value):
        self.results['poles'][key] = value
    
    def add_flow_result(self, key, value):
        self.results['flow'][key] = value
    
    def add_regionalization_result(self, key, value):
        self.results['regionalization'][key] = value


class PescoidExperimentalSeries:
    """Container for a time series of pescoid measurements."""
    
    def __init__(self, exp_id, condition, experiments_list):
        """
        Parameters
        ----------
        exp_id : str
            Experiment identifier
        condition : str
            Condition label
        experiments_list : list of PescoidExperiment
            Time-ordered list of experiments
        """
        self.exp_id = exp_id
        self.condition = condition
        self.experiments = sorted(experiments_list, key=lambda x: x.timepoint)
        
        # Summary dataframe (will be populated after analysis)
        self.summary_df = None
    
    def to_dataframe(self):
        """Convert experiments to a pandas DataFrame."""
        rows = []
        for exp in self.experiments:
            row = {
                'exp_id': exp.exp_id,
                'condition': exp.condition,
                'timepoint': exp.timepoint,
            }
            # Flatten results
            for category, metrics in exp.results.items():
                for key, val in metrics.items():
                    col_name = f"{category}_{key}"
                    row[col_name] = val
            rows.append(row)
        
        self.summary_df = pd.DataFrame(rows)
        return self.summary_df


class AnalysisPipeline:
    """Main orchestrator for 3D pescoid analysis."""
    
    def __init__(self, data_root, output_root):
        """
        Parameters
        ----------
        data_root : str
            Root folder with experiment subfolders
        output_root : str
            Where to save results
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Experimental series containers
        self.series_dict = {}  # exp_id -> PescoidExperimentalSeries
    
    def register_experiment(self, exp_id, condition, timepoint, channels_dict):
        """Add an experiment to the pipeline."""
        exp = PescoidExperiment(exp_id, condition, timepoint, channels_dict)
        
        if exp_id not in self.series_dict:
            self.series_dict[exp_id] = []
        self.series_dict[exp_id].append(exp)
        
        return exp
    
    def finalize_series(self):
        """Organize registered experiments into PescoidExperimentalSeries."""
        finalized = {}
        for exp_id, exps in self.series_dict.items():
            # Group by condition
            by_condition = {}
            for exp in exps:
                if exp.condition not in by_condition:
                    by_condition[exp.condition] = []
                by_condition[exp.condition].append(exp)
            
            # Create series per condition
            for condition, cond_exps in by_condition.items():
                series = PescoidExperimentalSeries(exp_id, condition, cond_exps)
                finalized[(exp_id, condition)] = series
        
        self.series_dict = finalized
        return finalized
    
    def save_summary_csvs(self):
        """Save per-condition summary tables."""
        for (exp_id, condition), series in self.series_dict.items():
            df = series.to_dataframe()
            csv_path = self.output_root / f"{exp_id}_{condition}_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")


if __name__ == '__main__':
    # Example usage
    pipeline = AnalysisPipeline(
        r"Z:\Megha_Kattimani\segmentation pipeline data",
        r"C:\Users\kattimani\Project\pescoid-image-analysis\3d_analysis_output"
    )
    
    print("3D Pescoid Analysis Pipeline initialized.")
    print(f"Data root: {pipeline.data_root}")
    print(f"Output root: {pipeline.output_root}")
