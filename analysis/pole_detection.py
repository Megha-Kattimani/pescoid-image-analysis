"""
Pole detection from perimeter-time kymographs.

A pole corresponds to a persistent angular location along the perimeter where 
|Δradius| is large and coherent over time. This module provides robust detection
based on time-averaged activity profiles.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def count_poles_from_kymograph(
    kymo,
    smooth_sigma=2,
    prominence=0.15,
    min_distance=20
):
    """
    Count polarity poles from a perimeter-time kymograph.

    Parameters
    ----------
    kymo : ndarray (P, T)
        Normalized radius change kymograph.
        P = perimeter bins, T = timepoints.
    smooth_sigma : float
        Gaussian smoothing along perimeter (default: 2).
    prominence : float
        Peak prominence threshold (default: 0.15).
    min_distance : int
        Minimum separation (in perimeter bins) between poles (default: 20).

    Returns
    -------
    pole_indices : ndarray
        Indices of detected poles along the perimeter.
    activity_profile : ndarray
        Time-averaged absolute activity along perimeter.
    """

    # 1. Time-averaged absolute activity
    activity_profile = np.mean(np.abs(kymo), axis=1)

    # 2. Smooth along perimeter (periodic boundary)
    activity_smooth = gaussian_filter1d(
        activity_profile, sigma=smooth_sigma, mode="wrap"
    )

    # 3. Peak detection
    peaks, properties = find_peaks(
        activity_smooth,
        prominence=prominence,
        distance=min_distance
    )

    return peaks, activity_smooth


def classify_poles(kymo, pole_indices):
    """
    Classify poles as outward (protrusions) or inward (indentations).

    Parameters
    ----------
    kymo : ndarray (P, T)
        Normalized radius change kymograph.
    pole_indices : ndarray
        Indices of detected poles.

    Returns
    -------
    outward_poles : ndarray
        Indices of outward-bulging poles.
    inward_poles : ndarray
        Indices of inward-indenting poles.
    """
    # Time-averaged signed displacement
    mean_signed = np.mean(kymo, axis=1)

    outward = pole_indices[mean_signed[pole_indices] > 0]
    inward = pole_indices[mean_signed[pole_indices] < 0]

    return outward, inward


def visualize_pole_detection(
    activity_profile,
    pole_indices,
    outward_poles=None,
    inward_poles=None,
    save_path=None
):
    """
    Visualize detected poles on the activity profile.

    Parameters
    ----------
    activity_profile : ndarray
        Time-averaged activity along perimeter.
    pole_indices : ndarray
        All detected pole indices.
    outward_poles : ndarray, optional
        Indices of outward poles (will be marked in red).
    inward_poles : ndarray, optional
        Indices of inward poles (will be marked in blue).
    save_path : str, optional
        Path to save the figure.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(activity_profile, 'k-', label="Perimeter activity", linewidth=1.5)
    
    if outward_poles is not None and len(outward_poles) > 0:
        plt.scatter(
            outward_poles, 
            activity_profile[outward_poles], 
            color="red", 
            s=100, 
            zorder=3, 
            label=f"Outward poles (n={len(outward_poles)})",
            marker='^'
        )
    
    if inward_poles is not None and len(inward_poles) > 0:
        plt.scatter(
            inward_poles, 
            activity_profile[inward_poles], 
            color="blue", 
            s=100, 
            zorder=3, 
            label=f"Inward poles (n={len(inward_poles)})",
            marker='v'
        )
    
    if outward_poles is None and inward_poles is None:
        plt.scatter(
            pole_indices, 
            activity_profile[pole_indices], 
            color="red", 
            s=100, 
            zorder=3, 
            label=f"Poles (n={len(pole_indices)})"
        )
    
    plt.xlabel("Perimeter index (normalized position)")
    plt.ylabel("⟨|Δr|⟩ₜ (time-averaged activity)")
    plt.title("Pole Detection from Kymograph")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def measure_pole_dimensions(masks, kymo_raw, pole_indices, n_perimeter_points=None):
    """
    Measure physical dimensions of each detected pole on the BF mask timeseries.

    For each pole index in the kymograph (= a position on the resampled perimeter):
      1. Find the angular range it covers (full width at half-maximum of activity peak)
      2. At the timepoint of peak elongation, identify the pole region of the mask
         within that angular wedge from the centroid
      3. Measure:
         - pole_length_px         : radial distance from centroid to the farthest mask
                                    pixel inside the wedge (i.e. tip distance)
         - pole_width_arc_px      : arc length spanned by the angular range at the
                                    measured tip radius
         - pole_angular_extent_deg: angular width of the pole in degrees
         - pole_area_px           : pescoid-mask area inside the wedge
         - pole_centroid_y/x      : centroid of the pole region within the mask

    Parameters
    ----------
    masks            : (T, Y, X) bool array — BF masks across time
    kymo_raw         : (P, T) array — unnormalised perimeter-time kymograph
                       (delta-radius values, P = perimeter bins)
    pole_indices     : 1-D array of pole positions along the perimeter (in [0, P))
    n_perimeter_points : int or None — total number of perimeter bins. Inferred
                       from kymo_raw.shape[0] if None.

    Returns
    -------
    list of dicts, one per pole, sorted by pole_length_px descending.
    Empty list if kymo_raw is empty or no poles given.
    """
    from skimage import measure

    if kymo_raw is None or kymo_raw.size == 0 or len(pole_indices) == 0:
        return []

    P, T = kymo_raw.shape
    if n_perimeter_points is None:
        n_perimeter_points = P

    # Time-averaged absolute activity (smoothed periodically)
    activity = np.mean(np.abs(kymo_raw), axis=1)
    activity_smooth = gaussian_filter1d(activity, sigma=2, mode="wrap")

    # Find the peak-elongation timepoint (max aspect ratio)
    aspect_ratios = []
    for t in range(T):
        if t >= len(masks):
            break
        props = measure.regionprops(masks[t].astype(int))
        if props:
            p = max(props, key=lambda r: r.area)
            mn = p.minor_axis_length or 1e-6
            aspect_ratios.append(p.major_axis_length / mn)
        else:
            aspect_ratios.append(1.0)
    if not aspect_ratios:
        return []
    peak_t = int(np.argmax(aspect_ratios))
    mask = masks[peak_t]

    # Centroid + image extent
    props = measure.regionprops(mask.astype(int))
    if not props:
        return []
    p0 = max(props, key=lambda r: r.area)
    cy, cx = p0.centroid

    # Per-pixel angles around the centroid
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    dy = yy - cy
    dx = xx - cx
    angles_img = np.arctan2(dy, dx)  # range [-pi, pi]
    radii_img = np.hypot(dy, dx)

    # Map pole index (perimeter bin) -> angle in [-pi, pi]
    # The kymograph perimeter axis is indexed 0..P-1 corresponding to baseline angles.
    # We approximate: bin_to_angle = -pi + 2*pi * (bin / P)
    bin_to_angle = lambda i: -np.pi + 2 * np.pi * (i / P)

    results = []
    for pi in pole_indices:
        # FWHM around the peak
        peak_val = activity_smooth[pi]
        half = peak_val / 2.0
        # Walk left
        l = pi
        for _ in range(P):
            l_prev = (l - 1) % P
            if activity_smooth[l_prev] < half:
                break
            l = l_prev
        # Walk right
        r = pi
        for _ in range(P):
            r_next = (r + 1) % P
            if activity_smooth[r_next] < half:
                break
            r = r_next

        a_lo = bin_to_angle(l)
        a_hi = bin_to_angle(r)

        # Compute angular extent handling wrap-around
        if a_hi >= a_lo:
            extent_rad = a_hi - a_lo
            in_wedge = (angles_img >= a_lo) & (angles_img <= a_hi)
        else:
            extent_rad = (2 * np.pi) - (a_lo - a_hi)
            in_wedge = (angles_img >= a_lo) | (angles_img <= a_hi)

        wedge_mask = mask & in_wedge
        if not wedge_mask.any():
            continue

        # Tip distance = max radial distance inside the wedge
        radii_in = radii_img[wedge_mask]
        pole_length_px = float(radii_in.max())
        pole_area_px = int(wedge_mask.sum())

        # Arc length at tip = radius * angular extent
        pole_width_arc_px = float(pole_length_px * extent_rad)

        # Centroid of pole region
        ys, xs = np.where(wedge_mask)
        pole_cy = float(ys.mean())
        pole_cx = float(xs.mean())

        results.append({
            "pole_index": int(pi),
            "peak_timepoint": int(peak_t),
            "pole_length_px": pole_length_px,
            "pole_width_arc_px": pole_width_arc_px,
            "pole_angular_extent_deg": float(np.degrees(extent_rad)),
            "pole_area_px": pole_area_px,
            "pole_centroid_y": pole_cy,
            "pole_centroid_x": pole_cx,
            "wedge_angle_lo_deg": float(np.degrees(a_lo)),
            "wedge_angle_hi_deg": float(np.degrees(a_hi)),
            "activity_at_peak": float(peak_val),
        })

    # Sort by length descending (primary pole first)
    results.sort(key=lambda d: d["pole_length_px"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["label"] = "primary" if i == 0 else ("secondary" if i == 1 else f"pole_{i+1}")
    return results


def detect_and_analyze_poles(
    kymo,
    smooth_sigma=2,
    prominence=0.15,
    min_distance=20,
    classify=True
):
    """
    Complete pole detection and analysis pipeline.

    Parameters
    ----------
    kymo : ndarray (P, T)
        Normalized radius change kymograph.
    smooth_sigma : float
        Gaussian smoothing parameter.
    prominence : float
        Peak prominence threshold.
    min_distance : int
        Minimum separation between poles.
    classify : bool
        Whether to classify poles as outward/inward.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'n_poles': total number of poles
        - 'pole_indices': array of pole positions
        - 'activity_profile': time-averaged activity
        - 'n_outward': number of outward poles (if classify=True)
        - 'n_inward': number of inward poles (if classify=True)
        - 'outward_indices': outward pole positions (if classify=True)
        - 'inward_indices': inward pole positions (if classify=True)
    """
    # Detect poles
    pole_indices, activity_profile = count_poles_from_kymograph(
        kymo, smooth_sigma, prominence, min_distance
    )

    results = {
        'n_poles': len(pole_indices),
        'pole_indices': pole_indices,
        'activity_profile': activity_profile
    }

    # Classify if requested
    if classify and len(pole_indices) > 0:
        outward, inward = classify_poles(kymo, pole_indices)
        results['n_outward'] = len(outward)
        results['n_inward'] = len(inward)
        results['outward_indices'] = outward
        results['inward_indices'] = inward

    return results
