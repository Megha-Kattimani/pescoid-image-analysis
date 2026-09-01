"""
Data loader for the 230212 mezzo/nuclei experiment folder structure.

Folder layout:
Z:\Megha_Kattimani\segmentation pipeline data\230212_mezzo_H2A_Chiron_Acitivin_Methyl-Cellolulose_med-L15_time-15.00_dur-18h_int-30min_cyc-36_stage-50epiboly\
  Activin\
    G001\
      *.tif (multi-channel stacks)
    G002\
      *.tif
  Chiron\
    G001\
      *.tif
  ...

Each TIF is expected to be a 3D+channel stack (z, y, x, channels or similar).
"""

import os
from pathlib import Path
import re
import numpy as np
import tifffile as tiff
from typing import Dict, List, Tuple


def parse_experiment_folder_structure(root_path: str) -> Dict:
    """
    Parse the 230212 experiment folder structure.
    
    Folder layout can be:
    - Structure A: root/Condition/Pescoid/files.tif
    - Structure B: root/Condition/files.tif (flat)
    
    Returns
    -------
    dict
        {
            'experiment_name': str,
            'conditions': {
                'Activin': [
                    {'exp_id': 'G001', 'path': '/...', 'files': [...]},
                    ...
                ],
                ...
            }
        }
    """
    root = Path(root_path)
    
    # Extract experiment name from folder
    exp_name = root.name
    
    result = {
        'experiment_name': exp_name,
        'conditions': {}
    }
    
    # Iterate condition folders
    for condition_dir in sorted(root.iterdir()):
        if not condition_dir.is_dir():
            continue
        
        condition_name = condition_dir.name
        result['conditions'][condition_name] = []
        
        # Check if TIF files are directly in condition folder (Structure B)
        tif_files_direct = list(condition_dir.glob('*.tif'))
        if tif_files_direct:
            # Flat structure: files directly in condition folder
            # Use filename stems as exp_ids
            tif_files_direct = sorted(tif_files_direct)
            for tif_file in tif_files_direct:
                exp_id = tif_file.stem
                result['conditions'][condition_name].append({
                    'exp_id': exp_id,
                    'condition': condition_name,
                    'path': str(condition_dir),
                    'files': [str(tif_file)],
                    'n_files': 1,
                })
        else:
            # Hierarchical structure: pescoid subfolders
            # Iterate pescoid folders (e.g., G001, G002, ...)
            for pescoid_dir in sorted(condition_dir.iterdir()):
                if not pescoid_dir.is_dir():
                    continue
                
                exp_id = pescoid_dir.name
                
                # Find all TIF files
                tif_files = list(pescoid_dir.glob('*.tif'))
                tif_files = sorted(tif_files)
                
                if tif_files:
                    result['conditions'][condition_name].append({
                        'exp_id': exp_id,
                        'condition': condition_name,
                        'path': str(pescoid_dir),
                        'files': [str(f) for f in tif_files],
                        'n_files': len(tif_files),
                    })
    
    return result


def load_3d_stack_from_file(filepath: str) -> np.ndarray:
    """
    Load a 3D image stack from a TIF file.
    
    Parameters
    ----------
    filepath : str
        Path to TIF file
    
    Returns
    -------
    np.ndarray
        Stack with shape (z, y, x) or (z, y, x, c) depending on file
    """
    try:
        stack = tiff.imread(filepath)
        return stack.astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def separate_channels(stack: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Separate multi-channel 3D stack into individual channel arrays.
    
    Handles various stack shapes:
    - (z, y, x): single channel (returns as 'brightfield')
    - (z, y, x, c): standard 4D (separates channels)
    - (z, c, ?, y, x): unusual 5D (mezzo/nuclei experiments)
    
    Expected channel order (4D case):
    - c=0: brightfield
    - c=1: mezzo (fluorescence)
    - c=2: nuclei (H2A or other nuclear marker)
    - c=3: membrane (if available)
    
    Parameters
    ----------
    stack : np.ndarray
        Shape (z, y, x), (z, y, x, c), or (z, c, ?, y, x)
    
    Returns
    -------
    dict
        {'brightfield': array, 'mezzo': array, 'nuclei': array, ...}
    """
    if stack is None:
        return None
    
    if stack.ndim == 3:
        # Single channel (z, y, x)
        return {'brightfield': stack}
    
    if stack.ndim == 4:
        # Multi-channel (z, y, x, c)
        channels = {}
        channel_names = ['brightfield', 'mezzo', 'nuclei', 'membrane']
        
        for i in range(min(stack.shape[3], len(channel_names))):
            channels[channel_names[i]] = stack[:, :, :, i]
        
        return channels
    
    if stack.ndim == 5:
        # Unusual 5D: (z, c, ?, y, x)
        # Interpret as: channel in dim 1, timepoint in dim 2
        channels = {}
        channel_names = ['brightfield', 'mezzo', 'nuclei', 'membrane']
        
        # Extract each channel by taking all z-planes, and averaging over the third dimension
        for i in range(min(stack.shape[1], len(channel_names))):
            # stack[:, i, :, :, :] -> (z, ?, y, x), take max projection over dim 1
            ch_data = stack[:, i, :, :, :].max(axis=1)  # (z, y, x)
            channels[channel_names[i]] = ch_data
        
        return channels
    
    return None


def load_condition_experiments(condition_path: str) -> List[Dict]:
    """
    Load all pescoid experiments for a given condition.
    
    Returns
    -------
    list of dict
        Each dict has:
        - 'exp_id': pescoid identifier
        - 'condition': condition name
        - 'channels': dict of channel arrays
        - 'metadata': dict with file info
    """
    condition_path = Path(condition_path)
    condition_name = condition_path.name
    
    experiments = []
    
    for pescoid_dir in sorted(condition_path.iterdir()):
        if not pescoid_dir.is_dir():
            continue
        
        exp_id = pescoid_dir.name
        tif_files = sorted(pescoid_dir.glob('*.tif'))
        
        if not tif_files:
            continue
        
        # Load and merge all TIFs for this pescoid (assuming single timepoint per folder)
        # If multiple TIFs, stack them along z
        stacks = []
        for tif_file in tif_files:
            stack = load_3d_stack_from_file(str(tif_file))
            if stack is not None:
                stacks.append(stack)
        
        if stacks:
            # Concatenate along z (depth)
            merged_stack = np.concatenate(stacks, axis=0)
            channels = separate_channels(merged_stack)
            
            experiments.append({
                'exp_id': exp_id,
                'condition': condition_name,
                'channels': channels,
                'metadata': {
                    'path': str(pescoid_dir),
                    'n_files': len(tif_files),
                    'stack_shape': merged_stack.shape,
                }
            })
    
    return experiments


if __name__ == '__main__':
    # Example usage
    root = r"Z:\Megha_Kattimani\segmentation pipeline data\230212_mezzo_H2A_Chiron_Acitivin_Methyl-Cellolulose_med-L15_time-15.00_dur-18h_int-30min_cyc-36_stage-50epiboly"
    
    structure = parse_experiment_folder_structure(root)
    print(f"Experiment: {structure['experiment_name']}")
    print(f"Conditions: {list(structure['conditions'].keys())}")
    
    for condition, exps in structure['conditions'].items():
        print(f"\n{condition}: {len(exps)} pescoids")
        for exp in exps[:2]:  # Show first 2
            print(f"  {exp['exp_id']}: {exp['n_files']} files")
