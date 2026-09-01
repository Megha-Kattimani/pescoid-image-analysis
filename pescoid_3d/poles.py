"""
3D Pole Detection Module

Extends 2D pole detection to 3D using spherical coordinates.
Detects persistent angular/spatial locations with large boundary deformation.

Key concepts:
- Measure radius in spherical coordinates (θ, φ) from centroid
- Time-average |Δr| over depth (z) and time (if available)
- Peak detection yields pole locations on sphere
- Characterize pole size (major/minor axis on sphere)
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage import measure, filters


def get_spherical_surface_sampling(bf_mask, n_theta=32, n_phi=16):
    """
    Sample the pescoid surface in spherical coordinates.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask (z, y, x)
    n_theta : int
        Number of azimuth samples (around z-axis)
    n_phi : int
        Number of elevation samples (pole to equator)
    
    Returns
    -------
    dict
        - 'centroid': (z, y, x) centroid of mask
        - 'radii': (n_phi, n_theta) array of radii to surface
        - 'theta': azimuth angles (radians)
        - 'phi': elevation angles (radians)
    """
    
    # Centroid
    coords = np.where(bf_mask)
    if len(coords[0]) == 0:
        return None
    
    cz, cy, cx = (np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2]))
    
    # Spherical grid
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)
    
    radii = np.zeros((n_phi, n_theta))
    
    # For each (phi, theta), find distance from centroid to boundary
    for i, ph in enumerate(phi):
        for j, th in enumerate(theta):
            # Direction in cartesian
            # θ = azimuth (around z), φ = elevation from north pole
            dx = np.sin(ph) * np.cos(th)  # x direction
            dy = np.sin(ph) * np.sin(th)  # y direction
            dz = np.cos(ph)               # z direction
            
            # Ray-cast along this direction from centroid
            max_dist = np.linalg.norm(np.array(bf_mask.shape))
            for d in np.linspace(0, max_dist, 100):
                px = int(np.clip(cy + dy * d, 0, bf_mask.shape[1] - 1))
                py = int(np.clip(cx + dx * d, 0, bf_mask.shape[2] - 1))
                pz = int(np.clip(cz + dz * d, 0, bf_mask.shape[0] - 1))
                
                if not bf_mask[pz, py, px]:
                    # Hit boundary
                    radii[i, j] = d
                    break
    
    return {
        'centroid': (cz, cy, cx),
        'radii': radii,
        'theta': theta,
        'phi': phi,
    }


def detect_poles_3d(bf_mask, smooth_sigma=1.0, prominence_threshold=0.1):
    """
    Detect poles on 3D pescoid surface.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask
    smooth_sigma : float
        Gaussian smoothing on sphere surface
    prominence_threshold : float
        Peak prominence threshold (relative)
    
    Returns
    -------
    dict
        - 'n_poles': number detected
        - 'pole_locations': list of (phi_idx, theta_idx, deviation) tuples
        - 'activity_map': (n_phi, n_theta) array of deviations
    """
    
    surf = get_spherical_surface_sampling(bf_mask)
    if surf is None:
        return {
            'n_poles': 0,
            'pole_locations': [],
            'activity_map': None,
        }
    
    radii = surf['radii']
    baseline_r = radii[0, :]  # Take north pole as baseline
    
    # Compute deviation
    delta_r = radii - baseline_r[np.newaxis, :]
    
    # Smooth on surface
    activity = np.abs(delta_r)
    for _ in range(2):
        # Circular smooth on theta
        activity_smooth = np.zeros_like(activity)
        for i in range(activity.shape[0]):
            activity_smooth[i, :] = filters.gaussian(activity[i, :], sigma=smooth_sigma)
        activity = activity_smooth
    
    # Time-averaged activity (if multiple timepoints, average here)
    activity_avg = activity.mean(axis=0) if activity.ndim > 2 else activity
    
    # Peak detection on equator (middle latitude)
    n_phi, n_theta = activity.shape
    equator_activity = activity[n_phi // 2, :]
    
    prominence = prominence_threshold * equator_activity.max()
    peaks, _ = find_peaks(equator_activity, prominence=prominence, distance=max(1, n_theta//8))
    
    pole_locations = []
    for pk in peaks:
        pole_locations.append((n_phi // 2, pk, equator_activity[pk]))
    
    return {
        'n_poles': len(peaks),
        'pole_locations': pole_locations,
        'activity_map': equator_activity,
        'radii': radii,
    }


def measure_pole_dimensions(bf_mask, pole_phi_idx, pole_theta_idx, window_size=3):
    """
    Measure major/minor axis lengths of a pole region.
    
    Parameters
    ----------
    bf_mask : np.ndarray
        Binary 3D mask
    pole_phi_idx : int
        Pole latitude index
    pole_theta_idx : int
        Pole longitude index
    window_size : int
        Size of region around pole to analyze
    
    Returns
    -------
    dict
        - 'major_axis': major axis length in pixels
        - 'minor_axis': minor axis length in pixels
        - 'elongation': major / minor ratio
    """
    
    surf = get_spherical_surface_sampling(bf_mask)
    if surf is None:
        return {
            'major_axis': 0,
            'minor_axis': 0,
            'elongation': 1.0,
        }
    
    radii = surf['radii']
    n_phi, n_theta = radii.shape
    
    # Extract region around pole
    phi_start = max(0, pole_phi_idx - window_size)
    phi_end = min(n_phi, pole_phi_idx + window_size + 1)
    theta_start = (pole_theta_idx - window_size) % n_theta
    theta_end = (pole_theta_idx + window_size + 1) % n_theta
    
    if theta_end > theta_start:
        region = radii[phi_start:phi_end, theta_start:theta_end]
    else:
        # Wrap around
        region = np.concatenate([
            radii[phi_start:phi_end, theta_start:],
            radii[phi_start:phi_end, :theta_end]
        ], axis=1)
    
    # Fit ellipse to peak region
    region_binary = (region > region.mean() + region.std()).astype(np.uint8)
    if region_binary.sum() == 0:
        return {
            'major_axis': 0,
            'minor_axis': 0,
            'elongation': 1.0,
        }
    
    labeled, _ = ndi.label(region_binary)
    if labeled.max() == 0:
        return {
            'major_axis': 0,
            'minor_axis': 0,
            'elongation': 1.0,
        }
    
    props = measure.regionprops(labeled)
    if len(props) == 0:
        return {
            'major_axis': 0,
            'minor_axis': 0,
            'elongation': 1.0,
        }
    
    p = sorted(props, key=lambda x: x.area, reverse=True)[0]
    
    return {
        'major_axis': float(p.major_axis_length),
        'minor_axis': float(p.minor_axis_length),
        'elongation': float(p.major_axis_length / (p.minor_axis_length + 1e-6)),
    }


if __name__ == '__main__':
    # Synthetic test
    z, y, x = 30, 100, 100
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    dist = np.sqrt((zz - z//2)**2 + (yy - y//2)**2 + (xx - x//2)**2)
    
    # Pescoid with two poles (elongated)
    bf_mask = (dist < 25).astype(np.uint8)
    # Add protrusions at poles
    bf_mask[(zz < 10) & ((yy - y//2)**2 + (xx - x//2)**2 < 400)] = 1
    bf_mask[(zz > 20) & ((yy - y//2)**2 + (xx - x//2)**2 < 400)] = 1
    
    poles = detect_poles_3d(bf_mask)
    print(f"3D Poles detected: {poles['n_poles']}")
    print(f"Pole locations: {poles['pole_locations']}")
