"""
3D Pescoid Segmentation Module

Segments the pescoid body from brightfield, computes:
- Volume and surface area
- 3D shape descriptors (sphericity, solidity)
- Centroid and principal axes
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, find_objects
from skimage import measure, filters, morphology


def segment_brightfield_3d(bf_stack, threshold_method='otsu', min_volume=1000):
    """
    Segment pescoid body from 3D brightfield stack.
    
    Parameters
    ----------
    bf_stack : np.ndarray
        3D brightfield image (z, y, x)
    threshold_method : str
        'otsu', 'yen', 'mean', or fixed int value
    min_volume : int
        Minimum voxel count to keep
    
    Returns
    -------
    mask : np.ndarray
        Binary 3D mask of pescoid body
    """
    
    if bf_stack is None or bf_stack.size == 0:
        return None
    
    # Normalize to uint8
    bf = np.asarray(bf_stack)
    bf_min, bf_max = bf.min(), bf.max()
    if bf_max > bf_min:
        bf = ((bf - bf_min) / (bf_max - bf_min) * 255).astype(np.uint8)
    else:
        bf = np.zeros_like(bf, dtype=np.uint8)
    
    # Apply Gaussian smoothing to reduce noise
    bf_smooth = filters.gaussian(bf, sigma=1.0)
    
    # Threshold
    if threshold_method == 'otsu':
        thresh = filters.threshold_otsu(bf_smooth)
    elif threshold_method == 'yen':
        thresh = filters.threshold_yen(bf_smooth)
    elif threshold_method == 'mean':
        thresh = bf_smooth.mean()
    else:
        thresh = float(threshold_method)
    
    # Pescoids are dark (low intensity), so invert
    mask = bf_smooth < thresh
    
    # Remove small objects
    mask = morphology.remove_small_objects(mask, min_size=min_volume)
    
    # Fill holes
    mask = ndimage.binary_fill_holes(mask)
    
    # Keep largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())
        largest_label = np.argmax(sizes[1:]) + 1
        mask = labeled == largest_label
    
    return mask.astype(np.uint8)


def compute_3d_properties(mask):
    """
    Compute 3D morphological properties of the pescoid.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary 3D mask
    
    Returns
    -------
    dict
        Properties including:
        - volume (voxels)
        - surface_area (voxels²)
        - centroid
        - principal_axes_length (largest to smallest)
        - sphericity
        - solidity
    """
    
    if mask is None or mask.sum() == 0:
        return {
            'volume': 0,
            'surface_area': 0,
            'centroid': None,
            'principal_axes_length': None,
            'sphericity': 0,
            'solidity': 0,
        }
    
    # Volume
    volume = mask.sum()
    
    # Surface area (approximate using boundary voxels)
    boundary = ndimage.binary_erosion(mask) ^ mask
    surface_area = boundary.sum()
    
    # Centroid
    coords = np.where(mask)
    centroid = (
        np.mean(coords[0]),  # z
        np.mean(coords[1]),  # y
        np.mean(coords[2]),  # x
    )
    
    # Inertia tensor for principal axes
    # Center coordinates
    z_c, y_c, x_c = coords[0] - centroid[0], coords[1] - centroid[1], coords[2] - centroid[2]
    
    # Compute moments
    Ixx = (y_c**2 + z_c**2).sum()
    Iyy = (x_c**2 + z_c**2).sum()
    Izz = (x_c**2 + y_c**2).sum()
    Ixy = -(x_c * y_c).sum()
    Ixz = -(x_c * z_c).sum()
    Iyz = -(y_c * z_c).sum()
    
    inertia_tensor = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    
    eigenvalues = np.linalg.eigvals(inertia_tensor)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Largest to smallest
    principal_axes = np.sqrt(eigenvalues / volume)
    
    # Sphericity = (6*V*sqrt(π)) / SA³
    # (1 = perfect sphere, <1 = elongated)
    pi = np.pi
    if surface_area > 0:
        sphericity = (6 * volume * np.sqrt(pi)) / (surface_area ** (3/2)) if surface_area > 0 else 0
        sphericity = min(sphericity, 1.0)
    else:
        sphericity = 0
    
    # Solidity = volume / convex_hull_volume
    # (1 = convex, <1 = concave)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.column_stack(coords))
        solidity = volume / hull.volume
    except:
        solidity = 1.0
    
    return {
        'volume': int(volume),
        'surface_area': int(surface_area),
        'centroid': centroid,
        'principal_axes_length': principal_axes,
        'sphericity': float(sphericity),
        'solidity': float(min(solidity, 1.0)),
    }


def get_3d_bounding_box(mask):
    """
    Get 3D bounding box of the masked region.
    
    Returns
    -------
    tuple
        (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    
    return (
        coords[0].min(), coords[0].max(),
        coords[1].min(), coords[1].max(),
        coords[2].min(), coords[2].max(),
    )


if __name__ == '__main__':
    # Example: create synthetic 3D pescoid
    z, y, x = 30, 100, 100
    bf = np.random.randint(100, 200, (z, y, x)).astype(np.float32)
    
    # Add a sphere-like dark region
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    dist = np.sqrt((zz - z//2)**2 + (yy - y//2)**2 + (xx - x//2)**2)
    bf[dist < 30] = 50
    
    # Segment
    mask = segment_brightfield_3d(bf)
    props = compute_3d_properties(mask)
    
    print("3D Segmentation Properties:")
    for key, val in props.items():
        print(f"  {key}: {val}")
