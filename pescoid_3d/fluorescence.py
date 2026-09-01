"""
Fluorescence Analysis Module

Analyzes mezzo (generalized epithelial marker), nuclei (H2A), and membrane signals.
Computes:
- Fluorescence fraction (mezzo intensity / total volume)
- Nuclei segmentation and cell counting
- Cell size statistics
- Membrane signal intensity and localization
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import label, find_objects
from skimage import measure, filters, morphology
from skimage.segmentation import watershed


def compute_fluorescence_fraction(bf_mask, mezzo_stack, threshold_method='otsu'):
    """
    Compute fraction of mezzo signal within pescoid body.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask of pescoid body (z, y, x)
    mezzo_stack : np.ndarray
        3D mezzo fluorescence (z, y, x)
    threshold_method : str
        'otsu', 'yen', or percentile value (0-100)
    
    Returns
    -------
    dict
        - 'mezzo_fraction': fraction of voxels above threshold
        - 'mezzo_intensity_mean': mean intensity within mask
        - 'mezzo_intensity_max': max intensity within mask
        - 'mezzo_intensity_std': std of intensity within mask
    """
    
    if bf_mask is None or mezzo_stack is None:
        return {
            'mezzo_fraction': 0,
            'mezzo_intensity_mean': 0,
            'mezzo_intensity_max': 0,
            'mezzo_intensity_std': 0,
        }
    
    # Crop to bounding box for efficiency
    bbox = np.where(bf_mask)
    if len(bbox[0]) == 0:
        return {
            'mezzo_fraction': 0,
            'mezzo_intensity_mean': 0,
            'mezzo_intensity_max': 0,
            'mezzo_intensity_std': 0,
        }
    
    zmin, zmax = bbox[0].min(), bbox[0].max()
    ymin, ymax = bbox[1].min(), bbox[1].max()
    xmin, xmax = bbox[2].min(), bbox[2].max()
    
    mask_crop = bf_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    mezzo_crop = mezzo_stack[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    
    # Intensity within mask
    mezzo_in_mask = mezzo_crop[mask_crop > 0]
    
    if mezzo_in_mask.size == 0:
        return {
            'mezzo_fraction': 0,
            'mezzo_intensity_mean': 0,
            'mezzo_intensity_max': 0,
            'mezzo_intensity_std': 0,
        }
    
    # Threshold mezzo
    if threshold_method == 'otsu':
        thresh = filters.threshold_otsu(mezzo_in_mask.astype(np.uint8))
    elif threshold_method == 'yen':
        thresh = filters.threshold_yen(mezzo_in_mask.astype(np.uint8))
    else:
        # Percentile
        thresh = np.percentile(mezzo_in_mask, float(threshold_method))
    
    mezzo_positive = (mezzo_in_mask > thresh).sum()
    mezzo_fraction = mezzo_positive / mezzo_in_mask.size
    
    return {
        'mezzo_fraction': float(mezzo_fraction),
        'mezzo_intensity_mean': float(mezzo_in_mask.mean()),
        'mezzo_intensity_max': float(mezzo_in_mask.max()),
        'mezzo_intensity_std': float(mezzo_in_mask.std()),
    }


def segment_nuclei_3d(nuclei_stack, min_size=100, max_size=1500, sigma=0.8, 
                       median_size=3, threshold_method='dynamic', min_distance=8):
    """Segment nuclei in 3D using local maxima detection with median filtering.
    
    Implements MATLAB-inspired approach: median filter → dynamic thresholding → 
    Gaussian smoothing → local maxima detection with minimum distance constraint.
    Based on MATLAB Cell-Flow analysis code from Tissue Flow project.
    
    Parameters
    ----------
    nuclei_stack : np.ndarray
        3D nuclear marker image (z, y, x)
    min_size : int
        Minimum voxel count per nucleus (default: 15)
    max_size : int
        Maximum voxel count per nucleus (default: 1500)
    sigma : float
        Gaussian smoothing sigma (default: 0.8, matched to ~6-8 pixel nucleus)
    median_size : int
        Median filter size to remove salt-pepper noise (default: 3, yields 3x3 kernel)
    threshold_method : str
        'dynamic' = mean + 0.5*std (robust), 'percentile' = 75th percentile, 
        'otsu' = Otsu method (default: 'dynamic')
    min_distance : int
        Minimum voxel distance between nuclei centers (default: 8)
    
    Returns
    -------
    labeled_mask : np.ndarray
        Integer label image where each nucleus has unique ID
    """
    
    if nuclei_stack is None or nuclei_stack.size == 0:
        return None
    
    # Normalize to 0-1 range
    nuc = np.asarray(nuclei_stack, dtype=np.float32)
    nuc_min, nuc_max = nuc.min(), nuc.max()
    if nuc_max <= nuc_min:
        return None
    
    nuc = (nuc - nuc_min) / (nuc_max - nuc_min)
    
    # Step 1: Median filter to remove salt-pepper noise (3D median per slice)
    nuc_filtered = np.zeros_like(nuc)
    for z in range(nuc.shape[0]):
        nuc_filtered[z] = filters.median(nuc[z], footprint=morphology.disk(median_size//2))
    
    # Step 2: Dynamic thresholding - more robust than Otsu for variable backgrounds
    if threshold_method == 'dynamic':
        # For normalized 0-1 data: use mean + 0.5*std to capture bright regions
        # This is more conservative than percentile-based
        mean_val = nuc_filtered.mean()
        std_val = nuc_filtered.std()
        thresh = mean_val + 0.5 * std_val
        # For variable backgrounds, use lower percentile
        thresh_alt = np.percentile(nuc_filtered, 70)
        # Take the lower threshold to be more permissive
        thresh = min(thresh, thresh_alt)
    elif threshold_method == 'percentile':
        # Use 75th percentile to capture moderate-to-high intensity voxels
        thresh = np.percentile(nuc_filtered, 75)
    else:  # otsu
        thresh = filters.threshold_otsu(nuc_filtered)
    
    # Step 3: Apply threshold to get initial mask
    mask_threshed = nuc_filtered > thresh
    
    # If mask is too small, relax threshold
    if mask_threshed.sum() < 100:  # Less than 100 voxels
        thresh = np.percentile(nuc_filtered, 50)  # Use median as fallback
        mask_threshed = nuc_filtered > thresh
    
    # Step 4: Gaussian smoothing on thresholded region
    nuc_smooth = filters.gaussian(nuc_filtered * mask_threshed, sigma=sigma)
    
    # Step 5: Find local maxima with minimum distance constraint
    # This gives us approximate nuclei centers
    from scipy.ndimage import maximum_filter
    
    local_max = np.zeros_like(nuc_smooth, dtype=bool)
    
    # Process each slice independently for 2D local maxima
    footprint_size = max(3, min_distance // 2)
    for z in range(nuc_smooth.shape[0]):
        slice_2d = nuc_smooth[z]
        if slice_2d.max() > 0:
            # Find 2D local maxima in this slice
            local_max_2d = slice_2d == maximum_filter(slice_2d, size=footprint_size)
            # Only keep maxima above half threshold (less strict)
            local_max_2d = local_max_2d & (slice_2d > thresh * 0.5)
            local_max[z] = local_max_2d
    
    # Step 6: Label connected components of maxima
    labeled_maxima, num_maxima = ndi.label(local_max)
    
    if num_maxima == 0:
        # Fallback: if no local maxima found, use connected components directly
        labeled_initial, num_labels = ndi.label(mask_threshed)
        if num_labels > 0:
            sizes = np.bincount(labeled_initial.ravel())
            valid_labels = np.where((sizes >= min_size) & (sizes <= max_size))[0]
            if len(valid_labels) > 0:
                new_labeled = np.zeros_like(labeled_initial)
                for new_id, old_id in enumerate(valid_labels, 1):
                    new_labeled[labeled_initial == old_id] = new_id
                return new_labeled.astype(np.uint16)
        return None
    
    # Step 7: Grow regions from maxima using watershed
    # This expands maxima into full nuclei regions
    distance_map = ndi.distance_transform_edt(mask_threshed)
    inverted = -distance_map
    
    # Watershed from markers (in skimage.segmentation)
    from skimage.segmentation import watershed
    markers = labeled_maxima.copy()
    labeled = watershed(inverted, markers=markers, mask=mask_threshed)
    
    # Step 8: Filter by size
    if labeled.max() > 0:
        sizes = np.bincount(labeled.ravel())
        valid_labels = np.where((sizes >= min_size) & (sizes <= max_size))[0]
        
        if len(valid_labels) > 0:
            new_labeled = np.zeros_like(labeled)
            for new_id, old_id in enumerate(valid_labels, 1):
                new_labeled[labeled == old_id] = new_id
            labeled = new_labeled
        else:
            # If strict filtering removed all, relax it
            valid_labels = np.where((sizes >= max(min_size//2, 5)) & (sizes <= max_size*3))[0]
            if len(valid_labels) > 0:
                new_labeled = np.zeros_like(labeled)
                for new_id, old_id in enumerate(valid_labels, 1):
                    new_labeled[labeled == old_id] = new_id
                labeled = new_labeled
            else:
                # Last resort: keep labeled maxima
                labeled = labeled_maxima
    
    return labeled.astype(np.uint16)


def analyze_nuclei(nuclei_labeled):
    """
    Compute statistics from labeled nuclei volume.
    
    Parameters
    ----------
    nuclei_labeled : np.ndarray
        Integer label image from segment_nuclei_3d
    
    Returns
    -------
    dict
        - 'n_nuclei': number of nuclei detected
        - 'mean_nucleus_volume': mean voxel count per nucleus
        - 'std_nucleus_volume': std of nucleus volumes
        - 'nucleus_volumes': list of individual nucleus volumes
    """
    
    if nuclei_labeled is None:
        return {
            'n_nuclei': 0,
            'mean_nucleus_volume': 0,
            'std_nucleus_volume': 0,
            'nucleus_volumes': [],
        }
    
    n_nuclei = nuclei_labeled.max()
    
    if n_nuclei == 0:
        return {
            'n_nuclei': 0,
            'mean_nucleus_volume': 0,
            'std_nucleus_volume': 0,
            'nucleus_volumes': [],
        }
    
    sizes = np.bincount(nuclei_labeled.ravel())[1:]  # Skip background
    
    return {
        'n_nuclei': int(n_nuclei),
        'mean_nucleus_volume': float(sizes.mean()),
        'std_nucleus_volume': float(sizes.std()),
        'nucleus_volumes': sizes.tolist(),
    }


def analyze_membrane_signal(bf_mask, membrane_stack, morphology_radius=2):
    """
    Analyze membrane signal intensity and localization.
    
    Computes intensity in a shell near the surface of the pescoid.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask of pescoid body
    membrane_stack : np.ndarray
        3D membrane marker image
    morphology_radius : int
        Radius for erosion to define interior shell
    
    Returns
    -------
    dict
        - 'membrane_intensity_mean': mean intensity at boundary
        - 'membrane_intensity_max': max intensity at boundary
        - 'membrane_fraction': fraction of boundary voxels above threshold
    """
    
    if bf_mask is None or membrane_stack is None:
        return {
            'membrane_intensity_mean': 0,
            'membrane_intensity_max': 0,
            'membrane_fraction': 0,
        }
    
    # Define boundary as exterior erosion
    interior = morphology.binary_erosion(bf_mask, morphology.ball(morphology_radius))
    boundary = bf_mask.astype(np.uint8) - interior.astype(np.uint8)
    
    mem_at_boundary = membrane_stack[boundary > 0]
    
    if mem_at_boundary.size == 0:
        return {
            'membrane_intensity_mean': 0,
            'membrane_intensity_max': 0,
            'membrane_fraction': 0,
        }
    
    # Threshold
    thresh = np.percentile(mem_at_boundary, 75)
    mem_positive = (mem_at_boundary > thresh).sum()
    mem_fraction = mem_positive / mem_at_boundary.size
    
    return {
        'membrane_intensity_mean': float(mem_at_boundary.mean()),
        'membrane_intensity_max': float(mem_at_boundary.max()),
        'membrane_fraction': float(mem_fraction),
    }


def diagnose_nuclei_detection(nuclei_stack, bf_mask=None):
    """
    Debug nuclei detection with multiple threshold strategies.
    
    Helps diagnose why nuclei count is low/high.
    
    Parameters
    ----------
    nuclei_stack : np.ndarray
        3D nuclear marker image
    bf_mask : np.ndarray, optional
        Brightfield mask to constrain analysis region
    
    Returns
    -------
    dict
        Detection statistics for each threshold strategy
    """
    
    nuc = np.asarray(nuclei_stack, dtype=np.float32)
    nuc_min, nuc_max = nuc.min(), nuc.max()
    if nuc_max <= nuc_min:
        return None
    
    nuc_norm = (nuc - nuc_min) / (nuc_max - nuc_min)
    
    if bf_mask is not None:
        nuc_masked = nuc_norm[bf_mask > 0]
    else:
        nuc_masked = nuc_norm.ravel()
    
    results = {
        'stack_shape': nuc.shape,
        'intensity_min': float(nuc_min),
        'intensity_max': float(nuc_max),
        'intensity_mean': float(nuc_masked.mean()),
        'intensity_std': float(nuc_masked.std()),
        'intensity_median': float(np.median(nuc_masked)),
        'thresholds': {},
        'nuclei_counts': {},
    }
    
    # Smooth
    nuc_smooth = filters.gaussian(nuc_norm, sigma=0.8)
    
    # Try different thresholds
    thresholds = {
        'otsu': filters.threshold_otsu(nuc_smooth),
        'li': None,
        'mean': nuc_smooth.mean(),
        'mean+std': nuc_smooth.mean() + nuc_smooth.std(),
        'percentile_75': np.percentile(nuc_smooth, 75),
        'percentile_80': np.percentile(nuc_smooth, 80),
        'percentile_85': np.percentile(nuc_smooth, 85),
    }
    
    try:
        thresholds['li'] = filters.threshold_li(nuc_smooth)
    except:
        pass
    
    for name, thresh in thresholds.items():
        if thresh is None:
            continue
        
        mask = nuc_smooth > thresh
        labeled, n_nuclei = ndi.label(mask)
        
        # Filter by size
        if n_nuclei > 0:
            sizes = np.bincount(labeled.ravel())
            valid = np.sum((sizes >= 30) & (sizes <= 10000))
        else:
            valid = 0
        
        results['thresholds'][name] = float(thresh) if thresh is not None else None
        results['nuclei_counts'][name] = {'total': int(n_nuclei), 'size_filtered': int(valid)}
    
    return results


def segment_nuclei_3d_with_bf_constraint(nuclei_stack, bf_mask, min_size=100, max_size=1500, 
                                         sigma=0.8, median_size=3, threshold_method='dynamic'):
    """Segment nuclei constrained to brightfield mask (inside pescoid only).
    
    Implements MATLAB-inspired nuclei detection with BF boundary constraint:
    - Only detects nuclei within BF mask
    - Median filter + dynamic thresholding + watershed
    - Much more robust than global thresholding
    
    Parameters
    ----------
    nuclei_stack : np.ndarray
        3D nuclear marker image (z, y, x)
    bf_mask : np.ndarray
        Binary 3D mask of pescoid body (z, y, x)
    min_size : int
        Minimum voxel count per nucleus (default: 15)
    max_size : int
        Maximum voxel count per nucleus (default: 1500)
    sigma : float
        Gaussian smoothing sigma (default: 0.8)
    median_size : int
        Median filter size (default: 3)
    threshold_method : str
        'dynamic' = mean + 0.5·std, 'percentile' = 85th percentile (default: 'dynamic')
    
    Returns
    -------
    labeled_mask : np.ndarray
        Integer label image where each nucleus has unique ID
    """
    
    if nuclei_stack is None or nuclei_stack.size == 0 or bf_mask is None:
        return None
    
    # Normalize nuclei to 0-1 range
    nuc = np.asarray(nuclei_stack, dtype=np.float32)
    nuc_min, nuc_max = nuc.min(), nuc.max()
    if nuc_max <= nuc_min:
        return None
    
    nuc = (nuc - nuc_min) / (nuc_max - nuc_min)
    
    # Apply BF mask constraint (only look inside pescoid)
    bf_mask_bool = bf_mask > 0
    nuc = nuc * bf_mask_bool
    
    # Step 1: Median filter per slice to remove salt-pepper noise
    nuc_filtered = np.zeros_like(nuc)
    for z in range(nuc.shape[0]):
        nuc_filtered[z] = filters.median(nuc[z], footprint=morphology.disk(median_size//2))
    
    # Step 2: Dynamic thresholding (robust to variable backgrounds)
    if threshold_method == 'dynamic':
        # Only compute stats inside BF mask
        nuc_inside = nuc_filtered[bf_mask_bool]
        if len(nuc_inside) > 0:
            thresh = nuc_inside.mean() + 0.5 * nuc_inside.std()
            thresh = np.clip(thresh, 0.15, 0.90)
        else:
            thresh = 0.5
    else:  # percentile
        nuc_inside = nuc_filtered[bf_mask_bool]
        if len(nuc_inside) > 0:
            thresh = np.percentile(nuc_inside, 85)
        else:
            thresh = 0.5
    
    # Step 3: Apply threshold within BF mask
    mask_threshed = (nuc_filtered > thresh) & bf_mask_bool
    
    # Step 4: Gaussian smoothing to enhance peaks
    nuc_smooth = filters.gaussian(nuc_filtered * mask_threshed, sigma=sigma)
    
    # Step 5: Find local 2D maxima per slice with minimum distance
    from scipy.ndimage import maximum_filter
    
    local_max = np.zeros_like(nuc_smooth, dtype=bool)
    footprint_size = max(3, 4)  # ~8 pixel minimum distance
    
    for z in range(nuc_smooth.shape[0]):
        slice_2d = nuc_smooth[z]
        if slice_2d.max() > 0:
            local_max_2d = slice_2d == maximum_filter(slice_2d, size=footprint_size)
            local_max_2d = local_max_2d & (slice_2d > thresh) & bf_mask_bool[z]
            local_max[z] = local_max_2d
    
    # Step 6: Label maxima as seeds
    labeled_maxima, num_maxima = ndi.label(local_max)
    
    if num_maxima == 0:
        return None
    
    # Step 7: Watershed from seeds within BF mask
    distance_map = ndi.distance_transform_edt(mask_threshed)
    inverted = -distance_map
    markers = labeled_maxima.copy()
    
    labeled = watershed(inverted, markers=markers, mask=mask_threshed)
    
    # Step 8: Filter by size
    if labeled.max() > 0:
        sizes = np.bincount(labeled.ravel())
        valid_labels = np.where((sizes >= min_size) & (sizes <= max_size))[0]
        
        if len(valid_labels) > 0:
            new_labeled = np.zeros_like(labeled)
            for new_id, old_id in enumerate(valid_labels, 1):
                new_labeled[labeled == old_id] = new_id
            labeled = new_labeled
        else:
            # Relax if too strict
            valid_labels = np.where((sizes >= max(min_size//2, 5)) & (sizes <= max_size*2))[0]
            if len(valid_labels) > 0:
                new_labeled = np.zeros_like(labeled)
                for new_id, old_id in enumerate(valid_labels, 1):
                    new_labeled[labeled == old_id] = new_id
                labeled = new_labeled
            else:
                labeled = labeled_maxima  # Fall back to maxima
    
    return labeled.astype(np.uint16)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Synthetic test
    z, y, x = 30, 100, 100
    
    # Create synthetic BF mask (sphere)
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    dist = np.sqrt((zz - z//2)**2 + (yy - y//2)**2 + (xx - x//2)**2)
    bf_mask = (dist < 30).astype(np.uint8)
    
    # Synthetic mezzo (higher inside)
    mezzo = np.random.randint(100, 200, (z, y, x)).astype(np.float32)
    mezzo[dist < 30] += 100
    
    # Synthetic nuclei (random blobs)
    nuclei = np.random.randint(50, 150, (z, y, x)).astype(np.float32)
    nuclei[dist < 30] += 150  # Brighter inside
    
    # Analyze
    fluo = compute_fluorescence_fraction(bf_mask, mezzo)
    print("Fluorescence:", fluo)
    
    nuclei_labeled = segment_nuclei_3d(nuclei)
    nuc_stats = analyze_nuclei(nuclei_labeled)
    print("Nuclei:", nuc_stats)
    
    mem = analyze_membrane_signal(bf_mask, mezzo)
    print("Membrane:", mem)
