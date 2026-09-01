"""
3D Particle Image Velocimetry (PIV) Analysis Module

Computes tissue flow velocity fields from 3D+time image sequences.
Optimized for H2B-labeled nuclei (TZCYX format).

Key features:
- Frame-by-frame PIV computation
- 3D divergence/vorticity analysis (identifies organizing centers)
- Temporal tracking of flow patterns
- Multi-scale analysis (whole embryo vs regional)
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage import filters
import warnings

try:
    from openpiv import tools, pyprocess, validation, filters as piv_filters
    HAS_OPENPIV = True
except ImportError:
    HAS_OPENPIV = False
    warnings.warn("openpiv not installed. Install with: pip install openpiv")


def compute_piv_2d_frame(frame1, frame2, window_size=32, overlap=16, 
                         dt=1.0, scale=1.0, sig2noise_threshold=1.5):
    """Compute 2D PIV between two consecutive frames.
    
    Uses cross-correlation to find displacement vectors.
    
    Parameters
    ----------
    frame1, frame2 : np.ndarray
        2D frames (y, x) at times t and t+dt
    window_size : int
        PIV interrogation window size (default: 32)
    overlap : int
        Overlap between windows (default: 16)
    dt : float
        Time difference in minutes (default: 1.0)
    scale : float
        Microns per pixel (default: 1.0)
    sig2noise_threshold : float
        Signal-to-noise ratio threshold for outlier removal (default: 1.5)
    
    Returns
    -------
    x, y : np.ndarray
        Grid coordinates of velocity vectors
    u, v : np.ndarray
        Velocity components (microns/minute)
    sig2noise : np.ndarray
        Signal-to-noise ratio per vector
    """
    
    if not HAS_OPENPIV:
        raise RuntimeError("openpiv required for PIV computation. Install with: pip install openpiv")
    
    # Normalize to uint8 for PIV (cross-correlation works better with contrast)
    f1 = (255 * (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-6)).astype(np.uint8)
    f2 = (255 * (frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-6)).astype(np.uint8)
    
    # Compute cross-correlation
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        f1, f2,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=window_size,
        sig2noise_method='peak2mean'
    )
    
    # Get grid
    x, y = pyprocess.get_coordinates(
        image_size=frame1.shape,
        window_size=window_size,
        overlap=overlap
    )
    
    # Validation: remove unreliable vectors
    u_valid, v_valid = validation.global_std_test(u, v, threshold=3.0)
    
    # Scale velocities (from pixels/frame to microns/minute)
    u = u_valid * scale / dt
    v = v_valid * scale / dt
    
    return x, y, u, v, sig2noise


def compute_piv_3d_frame(stack1, stack2, z_slice=None, **kwargs):
    """Compute PIV on 3D stack by taking maximum/mean projection.
    
    Parameters
    ----------
    stack1, stack2 : np.ndarray
        3D stacks (z, y, x) at times t and t+dt
    z_slice : int, optional
        If provided, use single z-slice instead of projection
    **kwargs : dict
        Arguments passed to compute_piv_2d_frame
    
    Returns
    -------
    x, y, u, v, sig2noise : arrays
        PIV results from projection or single slice
    """
    
    if z_slice is not None:
        # Use single z-slice
        frame1 = stack1[z_slice].astype(np.float32)
        frame2 = stack2[z_slice].astype(np.float32)
    else:
        # Maximum projection (nuclei are bright, this finds peak intensity)
        frame1 = stack1.max(axis=0).astype(np.float32)
        frame2 = stack2.max(axis=0).astype(np.float32)
    
    return compute_piv_2d_frame(frame1, frame2, **kwargs)


def compute_divergence_3d(u, v, w=None, dx=1.0):
    """Compute divergence of velocity field.
    
    Identifies flow convergence (negative divergence = compression).
    In development, high negative divergence marks potential organizing centers.
    
    Parameters
    ----------
    u, v : np.ndarray
        2D velocity components (y, x)
    w : np.ndarray, optional
        3D z-component (not used in 2D PIV)
    dx : float
        Grid spacing (default: 1.0)
    
    Returns
    -------
    divergence : np.ndarray
        Divergence field (same shape as u, v)
        Negative = convergence, Positive = divergence
    """
    
    # Central differences for spatial derivatives
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dx, axis=0)
    
    divergence = du_dx + dv_dy
    
    return divergence


def compute_vorticity_2d(u, v, dx=1.0):
    """Compute 2D vorticity (curl) of velocity field.
    
    Identifies rotational/swirling motion.
    
    Parameters
    ----------
    u, v : np.ndarray
        2D velocity components (y, x)
    dx : float
        Grid spacing (default: 1.0)
    
    Returns
    -------
    vorticity : np.ndarray
        Vorticity field (positive = counterclockwise)
    """
    
    # ∂v/∂x - ∂u/∂y
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dx, axis=0)
    
    vorticity = dv_dx - du_dy
    
    return vorticity


def track_piv_sequence(image_sequence, window_size=32, overlap=16, 
                      dt=1.0, scale=1.0, z_slice=None,
                      channel_idx=0, smooth_sigma=1.0):
    """Compute PIV for entire time sequence.
    
    Parameters
    ----------
    image_sequence : np.ndarray
        Time-series 3D stack (T, Z, Y, X) or (T, Y, X)
    window_size : int
        PIV window size (default: 32)
    overlap : int
        PIV overlap (default: 16)
    dt : float
        Time between frames in minutes (default: 1.0)
    scale : float
        Microns per pixel (default: 1.0)
    z_slice : int, optional
        Use single z-slice (default: max projection)
    channel_idx : int
        Channel to use if 5D (default: 0)
    smooth_sigma : float
        Gaussian smoothing sigma for fields (default: 1.0)
    
    Returns
    -------
    dict with keys:
        'u_fields' : list of u velocity arrays
        'v_fields' : list of v velocity arrays
        'divergence_maps' : list of divergence fields
        'vorticity_maps' : list of vorticity fields
        'x_grids' : list of x-coordinates
        'y_grids' : list of y-coordinates
        'sig2noise_ratios' : list of signal-to-noise arrays
    """
    
    results = {
        'u_fields': [],
        'v_fields': [],
        'divergence_maps': [],
        'vorticity_maps': [],
        'x_grids': [],
        'y_grids': [],
        'sig2noise_ratios': []
    }
    
    # Handle different input shapes
    if image_sequence.ndim == 4:
        # (T, Z, Y, X) - 3D+time
        sequences = [image_sequence[:, z_slice if z_slice else 0, :, :] 
                    for _ in range(1)]  # Single sequence
    elif image_sequence.ndim == 5:
        # (T, Z, C, Y, X) - 3D+time+channel
        sequences = [image_sequence[:, :, channel_idx, :, :]]  # Use specified channel
    elif image_sequence.ndim == 3:
        # (T, Y, X) - 2D+time
        sequences = [image_sequence]
    else:
        raise ValueError(f"Unsupported sequence shape: {image_sequence.ndim}D")
    
    # Process each timepoint
    seq = sequences[0]
    for t in range(seq.shape[0] - 1):
        print(f"  Processing timepoint {t+1}/{seq.shape[0]-1}...", end=' ')
        
        try:
            # Get 2D frames (either single z-slice or max projection)
            if seq.ndim == 4:
                if z_slice is not None:
                    frame1 = seq[t, z_slice].astype(np.float32)
                    frame2 = seq[t+1, z_slice].astype(np.float32)
                else:
                    frame1 = seq[t].max(axis=0).astype(np.float32)
                    frame2 = seq[t+1].max(axis=0).astype(np.float32)
            else:  # 3D
                frame1 = seq[t].astype(np.float32)
                frame2 = seq[t+1].astype(np.float32)
            
            # Compute PIV
            x, y, u, v, sig2noise = compute_piv_2d_frame(
                frame1, frame2,
                window_size=window_size,
                overlap=overlap,
                dt=dt,
                scale=scale
            )
            
            # Smooth velocity fields
            u_smooth = gaussian_filter(u, sigma=smooth_sigma)
            v_smooth = gaussian_filter(v, sigma=smooth_sigma)
            
            # Compute flow properties
            divergence = compute_divergence_3d(u_smooth, v_smooth, dx=scale)
            vorticity = compute_vorticity_2d(u_smooth, v_smooth, dx=scale)
            
            # Store results
            results['x_grids'].append(x)
            results['y_grids'].append(y)
            results['u_fields'].append(u_smooth)
            results['v_fields'].append(v_smooth)
            results['divergence_maps'].append(divergence)
            results['vorticity_maps'].append(vorticity)
            results['sig2noise_ratios'].append(sig2noise)
            
            print("[OK]")
        
        except Exception as e:
            print(f"[ERROR] {e}")
            # Continue with next frame
            continue
    
    return results


def find_convergence_zones(divergence_maps, threshold=-0.5):
    """Identify zones of flow convergence across timepoints.
    
    Convergence (negative divergence) indicates where flows meet,
    often marking organizing centers.
    
    Parameters
    ----------
    divergence_maps : list of np.ndarray
        Divergence field for each timepoint
    threshold : float
        Divergence threshold for convergence (default: -0.5)
    
    Returns
    -------
    dict with:
        'persistent_convergence' : zones appearing in >80% of frames
        'peak_convergence_sites' : strongest convergence regions
        'temporal_persistence' : how many frames each region appears in
    """
    
    # Stack divergence maps
    div_stack = np.array(divergence_maps)  # (T, Y, X)
    
    # Find convergence zones (negative divergence)
    convergence = div_stack < threshold
    
    # Temporal persistence: count frames where each pixel is convergent
    persistence = convergence.sum(axis=0)  # (Y, X)
    
    # Persistent zones: appear in >80% of frames
    threshold_persistence = int(0.8 * div_stack.shape[0])
    persistent = persistence >= threshold_persistence
    
    return {
        'persistent_convergence': persistent,
        'peak_convergence_sites': persistence.max(),
        'temporal_persistence': persistence,
        'mean_divergence': div_stack.mean(axis=0)
    }


def compute_flow_statistics(u_fields, v_fields):
    """Compute statistics of velocity fields.
    
    Parameters
    ----------
    u_fields : list of np.ndarray
        U-component velocity at each timepoint
    v_fields : list of np.ndarray
        V-component velocity at each timepoint
    
    Returns
    -------
    dict with velocity statistics
    """
    
    u_stack = np.array(u_fields)
    v_stack = np.array(v_fields)
    
    speed = np.sqrt(u_stack**2 + v_stack**2)
    
    return {
        'mean_speed': speed.mean(),
        'max_speed': speed.max(),
        'std_speed': speed.std(),
        'spatial_mean_speed': speed.mean(axis=(1, 2)),  # Mean per timepoint
        'temporal_mean_speed': speed.mean(axis=0)  # Mean per location
    }
