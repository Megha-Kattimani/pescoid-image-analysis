"""
Regionalization Module

Segments pescoid into spatial regions (anterior/posterior, dorsal/ventral).
Analyzes region-specific properties and tissue dynamics.

Key features:
- PCA-based principal axes definition
- Anterior/posterior separation along major axis
- Dorsal/ventral separation along minor axis
- Per-region nuclei, volume, fluorescence, flow statistics
"""

import numpy as np
from scipy import ndimage as ndi, linalg
from scipy.ndimage import label
from skimage import measure


def compute_principal_axes(bf_mask):
    """
    Compute major/minor/minor axes from 3D mask inertia tensor.
    
    Returns
    -------
    dict
        - 'principal_axes': (3, 3) eigenvectors (columns)
        - 'eigenvalues': (3,) inertia moments
        - 'major_axis_direction': primary direction
        - 'minor_axis_direction': secondary direction
    """
    
    coords = np.where(bf_mask)
    if len(coords[0]) == 0:
        return {
            'principal_axes': np.eye(3),
            'eigenvalues': np.ones(3),
            'major_axis_direction': np.array([0, 0, 1]),
            'minor_axis_direction': np.array([1, 0, 0]),
        }
    
    # Centroid
    centroid = np.array([
        np.mean(coords[0]),
        np.mean(coords[1]),
        np.mean(coords[2]),
    ])
    
    # Center coordinates (more efficient computation)
    coords_centered = np.array([
        coords[0] - centroid[0],
        coords[1] - centroid[1],
        coords[2] - centroid[2],
    ]).astype(float)
    
    # Inertia tensor using matrix multiplication (vectorized)
    # I = Σ(r·r^T) / N
    I = coords_centered @ coords_centered.T / coords_centered.shape[1]
    
    # Eigendecomposition
    evals, evecs = linalg.eigh(I)
    
    # Sort by magnitude (largest first = major axis)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    return {
        'principal_axes': evecs,
        'eigenvalues': evals,
        'centroid': centroid,
        'major_axis_direction': evecs[:, 0],
        'minor_axis_direction': evecs[:, 1],
    }


def segment_anterior_posterior(bf_mask, centroid, major_axis_direction):
    """
    Separate anterior and posterior hemispheres along major axis.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask
    centroid : np.ndarray
        (3,) centroid position
    major_axis_direction : np.ndarray
        (3,) unit vector of major axis
    
    Returns
    -------
    tuple
        (anterior_mask, posterior_mask)
    """
    
    # Vectorized computation
    zz, yy, xx = np.mgrid[0:bf_mask.shape[0], 0:bf_mask.shape[1], 0:bf_mask.shape[2]]
    pos = np.array([
        zz - centroid[0],
        yy - centroid[1],
        xx - centroid[2]
    ])
    
    proj = (pos[0] * major_axis_direction[0] +
            pos[1] * major_axis_direction[1] +
            pos[2] * major_axis_direction[2])
    
    anterior = bf_mask & (proj > 0)
    posterior = bf_mask & (proj <= 0)
    
    return anterior, posterior


def segment_dorsal_ventral(bf_mask, centroid, minor_axis_direction):
    """
    Separate dorsal and ventral regions along minor axis.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask
    centroid : np.ndarray
        (3,) centroid position
    minor_axis_direction : np.ndarray
        (3,) unit vector of minor axis
    
    Returns
    -------
    tuple
        (dorsal_mask, ventral_mask)
    """
    
    # Vectorized computation
    zz, yy, xx = np.mgrid[0:bf_mask.shape[0], 0:bf_mask.shape[1], 0:bf_mask.shape[2]]
    pos = np.array([
        zz - centroid[0],
        yy - centroid[1],
        xx - centroid[2]
    ])
    
    proj = (pos[0] * minor_axis_direction[0] +
            pos[1] * minor_axis_direction[1] +
            pos[2] * minor_axis_direction[2])
    
    dorsal = bf_mask & (proj > 0)
    ventral = bf_mask & (proj <= 0)
    
    return dorsal, ventral


def compute_region_properties(region_mask, fluorescence_mask=None, nuclei_mask=None):
    """
    Compute morphology and fluorescence in a region.
    
    Parameters
    ----------
    region_mask : np.ndarray
        Binary mask of region
    fluorescence_mask : np.ndarray, optional
        Mezzo intensity (not thresholded)
    nuclei_mask : np.ndarray, optional
        Nuclei binary mask
    
    Returns
    -------
    dict
        - 'volume': voxel count
        - 'mean_intensity': mean fluorescence (if provided)
        - 'nuclei_count': number of nuclei (if provided)
        - 'nuclei_density': nuclei per unit volume
    """
    
    volume = region_mask.sum()
    
    results = {
        'volume': int(volume),
        'mean_intensity': 0.0,
        'nuclei_count': 0,
        'nuclei_density': 0.0,
    }
    
    if fluorescence_mask is not None:
        inside = region_mask & (fluorescence_mask > 0)
        if inside.sum() > 0:
            results['mean_intensity'] = float(fluorescence_mask[inside].mean())
    
    if nuclei_mask is not None:
        nuclei_in_region = region_mask & nuclei_mask
        labeled, n_nuclei = label(nuclei_in_region)
        results['nuclei_count'] = int(n_nuclei)
        if volume > 0:
            results['nuclei_density'] = n_nuclei / (volume / 1e6)  # nuclei per million voxels
    
    return results


def four_quadrant_segmentation(bf_mask, centroid, major_axis, minor_axis):
    """
    Segment pescoid into four quadrants: anterior/dorsal, anterior/ventral, etc.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask
    centroid : np.ndarray
        (3,) centroid
    major_axis : np.ndarray
        (3,) major axis direction
    minor_axis : np.ndarray
        (3,) minor axis direction
    
    Returns
    -------
    dict
        Maps region name to binary mask:
        - 'anterior_dorsal'
        - 'anterior_ventral'
        - 'posterior_dorsal'
        - 'posterior_ventral'
    """
    
    # Split along both axes
    ant, post = segment_anterior_posterior(bf_mask, centroid, major_axis)
    dor, ven = segment_dorsal_ventral(bf_mask, centroid, minor_axis)
    
    return {
        'anterior_dorsal': ant & dor,
        'anterior_ventral': ant & ven,
        'posterior_dorsal': post & dor,
        'posterior_ventral': post & ven,
    }


def analyze_regionalization(bf_mask, fluorescence=None, nuclei=None):
    """
    Full regionalization analysis: split into regions and analyze each.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D pescoid mask
    fluorescence : np.ndarray, optional
        Mezzo intensity image
    nuclei : np.ndarray, optional
        Nuclei binary mask
    
    Returns
    -------
    dict
        - 'axes': principal axes information
        - 'quadrants': per-quadrant properties
        - 'hemicircles': anterior/posterior statistics
    """
    
    # Get axes
    axes_info = compute_principal_axes(bf_mask)
    centroid = axes_info['centroid']
    major = axes_info['major_axis_direction']
    minor = axes_info['minor_axis_direction']
    
    # Four quadrants
    quads = four_quadrant_segmentation(bf_mask, centroid, major, minor)
    quad_props = {}
    for name, mask in quads.items():
        quad_props[name] = compute_region_properties(mask, fluorescence, nuclei)
    
    # Hemispheres (anterior/posterior)
    ant, post = segment_anterior_posterior(bf_mask, centroid, major)
    ant_props = compute_region_properties(ant, fluorescence, nuclei)
    post_props = compute_region_properties(post, fluorescence, nuclei)
    
    return {
        'axes': axes_info,
        'quadrants': quad_props,
        'hemispheres': {
            'anterior': ant_props,
            'posterior': post_props,
        },
    }


if __name__ == '__main__':
    # Synthetic test: elongated ellipsoid
    z, y, x = 30, 100, 100
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    
    # Ellipsoid: major axis along z, minor along x
    dist_ellipsoid = np.sqrt(
        ((zz - z//2) / 12)**2 +
        ((yy - y//2) / 20)**2 +
        ((xx - x//2) / 20)**2
    )
    bf_mask = (dist_ellipsoid < 1).astype(np.uint8)
    
    # Regionalization
    result = analyze_regionalization(bf_mask)
    
    print("Principal axes eigenvalues:", result['axes']['eigenvalues'])
    print("Quadrant volumes:")
    for name, props in result['quadrants'].items():
        print(f"  {name}: {props['volume']} voxels")
    
    print("Hemisphere volumes:")
    print(f"  Anterior: {result['hemispheres']['anterior']['volume']} voxels")
    print(f"  Posterior: {result['hemispheres']['posterior']['volume']} voxels")
