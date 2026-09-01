"""
Configuration file for pescoid pipeline
Adjust these parameters based on your diagnostic results
"""

# ============================================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================================

# Z-projection ranges
BF_Z_RANGE = (4, 7)  # Z-slices to use for BF max projection
GFP_Z_RANGE = None   # None = use all slices for mean projection

# ============================================================================
# SEGMENTATION PARAMETERS (adjust based on diagnostic_segmentation.py results)
# ============================================================================

# Brightfield segmentation
BF_SEGMENTATION = {
    'min_area': 50000,           # Minimum pescoid area in pixels
    'smoothing_sigma': 2.0,      # Gaussian smoothing (higher = smoother, less noise)
    'closing_radius': 3,         # Morphological closing disk radius
    'invert_mask': True,         # True if pescoid is darker than background
    'preprocessing': 'clahe',    # 'clahe' or 'rescale'
}

# GFP segmentation
GFP_SEGMENTATION = {
    'min_area': 1000,            # Minimum GFP region area
    'threshold_method': 'maxentropy',  # 'maxentropy', 'otsu', or 'li'
    'smoothing_sigma': 1.5,
}

# ============================================================================
# POLE DETECTION PARAMETERS
# ============================================================================

POLE_DETECTION = {
    'n_angles': 360,              # Angular resolution for radial profiling
    'std_threshold': 1.5,         # Standard deviation multiplier for pole detection
    'min_angle_separation_deg': 20,  # Minimum angle between poles
    'dilate_before_poles': False, # Apply dilation before pole detection
    'dilate_radius': 3,           # Dilation radius if enabled
}

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

OUTPUT = {
    'video_fps': 4,              # Frames per second for output videos
    'save_overlays': True,       # Save individual frame overlays
    'save_masks': True,          # Save mask composites
    'save_kymographs': True,     # Save radial kymographs
    'save_plots': True,          # Save summary plots
}

# ============================================================================
# PERFORMANCE PARAMETERS
# ============================================================================

PERFORMANCE = {
    'workers': 1,                # Number of parallel workers (1 = sequential)
    'per_file_groups': True,     # Treat each TIFF as separate experiment
    'limit': None,               # Limit number of experiments (None = all)
}

# ============================================================================
# PATHS (can be overridden via CLI)
# ============================================================================

PATHS = {
    'base_path': r"G:\.shortcut-targets-by-id\1Uqt7OMVK1xAAdSEgKGyzLLdGHFXGltYG\NM_MK\Chemical Boundaries (Activin, Chiron)",
    'output_path': r"D:\IISc docs\EMBL Barcelona\Project\Chemical Boundaries",
}

# ============================================================================
# EXPERIMENTAL CONDITIONS (for organizing results)
# ============================================================================

CONDITIONS = {
    'ctrl': {
        'name': 'Control',
        'color': 'gray',
        'marker': 'o',
    },
    'Act': {
        'name': 'Activin',
        'color': 'red',
        'marker': 's',
    },
    'Chi': {
        'name': 'Chiron',
        'color': 'blue',
        'marker': '^',
    },
    'Act-Chi': {
        'name': 'Activin+Chiron',
        'color': 'purple',
        'marker': 'D',
    },
}
