"""
Tissue Tension Analysis from LynTom Membrane Signal
=====================================================

Computes tension proxies from membrane marker (LynTom-RFP) at 10x:
  - Angular sector analysis: mean intensity + gradient magnitude per sector
  - Radial profile: membrane signal from center to edge
  - Tension anisotropy: ratio of max/min sector tension (polarity measure)
  - Pole tension: compare tension at pole tip vs equator

Also computes pole dimensions from BF masks:
  - Primary pole: largest protrusion from kymograph
  - Secondary pole: second largest
  - Major/minor axis of each pole region
"""

import numpy as np
from skimage import measure, filters, morphology
from scipy import ndimage as ndi
from scipy.signal import find_peaks


def compute_sector_tension(lyn_norm, mask, n_sectors=12):
    """
    Divide the pescoid mask into angular sectors from centroid,
    compute LynTom intensity and gradient per sector.

    Returns list of dicts (one per sector) + metadata.
    """
    Y, X = mask.shape
    props = measure.regionprops(mask.astype(int))
    if not props:
        return None

    p = props[0]
    cy, cx = p.centroid
    orientation = p.orientation

    yy, xx = np.mgrid[:Y, :X]
    dy = yy - cy
    dx = xx - cx
    angles = np.arctan2(dy, dx)
    angles_aligned = (angles - orientation + np.pi) % (2 * np.pi) - np.pi

    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    sector_map = np.clip(np.digitize(angles_aligned, sector_edges) - 1, 0, n_sectors - 1)

    grad = filters.sobel(lyn_norm)

    sectors = []
    for s in range(n_sectors):
        smask = (sector_map == s) & mask
        if smask.sum() < 10:
            sectors.append({
                'sector': s,
                'angle_deg': float(np.degrees(sector_edges[s] + np.diff(sector_edges[:2])[0] / 2)),
                'mean_lyn': 0.0, 'std_lyn': 0.0, 'gradient_mag': 0.0, 'area': 0,
            })
            continue

        vals = lyn_norm[smask]
        sectors.append({
            'sector': s,
            'angle_deg': float(np.degrees(sector_edges[s] + np.diff(sector_edges[:2])[0] / 2)),
            'mean_lyn': float(vals.mean()),
            'std_lyn': float(vals.std()),
            'gradient_mag': float(grad[smask].mean()),
            'area': int(smask.sum()),
        })

    # Tension anisotropy: max/min gradient ratio
    grad_vals = [s['gradient_mag'] for s in sectors if s['area'] > 0]
    anisotropy = max(grad_vals) / min(grad_vals) if grad_vals and min(grad_vals) > 0 else 1.0

    return {
        'sectors': sectors,
        'centroid': (float(cy), float(cx)),
        'orientation': float(orientation),
        'anisotropy': float(anisotropy),
    }


def compute_radial_profile(lyn_norm, mask, n_bins=20):
    """
    Radial LynTom intensity profile from centroid to edge.
    """
    props = measure.regionprops(mask.astype(int))
    if not props:
        return None

    cy, cx = props[0].centroid
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = dist[mask].max() if mask.any() else 1
    dist_norm = dist / max_dist

    bins = np.linspace(0, 1, n_bins + 1)
    profile = []
    for i in range(n_bins):
        ring = mask & (dist_norm >= bins[i]) & (dist_norm < bins[i + 1])
        if ring.sum() > 5:
            profile.append(float(lyn_norm[ring].mean()))
        else:
            profile.append(np.nan)

    return {
        'radial_bins': np.linspace(0, 1, n_bins).tolist(),
        'radial_profile': profile,
    }


def compute_pole_dimensions(masks, kymo_raw=None, n_perimeter_points=200):
    """
    Measure primary and secondary pole dimensions from the mask timeseries.

    Uses the final timepoint mask. A "pole" is a protruding region
    identified by high curvature along the contour.

    Returns dict with pole1 and pole2 major/minor axis + area.
    """
    T = masks.shape[0]
    # Use the timepoint with highest aspect ratio (most elongated)
    ars = []
    for t in range(T):
        props = measure.regionprops(masks[t].astype(int))
        if props:
            p = props[0]
            ar = p.major_axis_length / p.minor_axis_length if p.minor_axis_length > 0 else 1
            ars.append(ar)
        else:
            ars.append(1.0)

    best_t = int(np.argmax(ars))
    mask = masks[best_t]

    props = measure.regionprops(mask.astype(int))
    if not props:
        return {'n_poles': 0}

    p = props[0]
    cy, cx = p.centroid
    orientation = p.orientation
    major = p.major_axis_length
    minor = p.minor_axis_length

    # Find pole tips: endpoints of the major axis
    # The major axis runs at angle = orientation from centroid
    half_major = major / 2
    pole1_y = cy + half_major * np.sin(orientation)
    pole1_x = cx + half_major * np.cos(orientation)
    pole2_y = cy - half_major * np.sin(orientation)
    pole2_x = cx - half_major * np.cos(orientation)

    # Extract regions around each pole tip
    r = int(minor * 0.4)  # pole radius ~ 40% of minor axis

    def measure_pole_region(py, px, mask, label):
        Y, X = mask.shape
        y0 = max(0, int(py) - r)
        y1 = min(Y, int(py) + r)
        x0 = max(0, int(px) - r)
        x1 = min(X, int(px) + r)
        pole_mask = mask[y0:y1, x0:x1].copy()
        if pole_mask.sum() == 0:
            return {'label': label, 'area': 0, 'major_axis': 0, 'minor_axis': 0, 'aspect_ratio': 0}
        pole_props = measure.regionprops(pole_mask.astype(int))
        if not pole_props:
            return {'label': label, 'area': 0, 'major_axis': 0, 'minor_axis': 0, 'aspect_ratio': 0}
        pp = max(pole_props, key=lambda r: r.area)
        maj = pp.major_axis_length or 1e-6
        minr = pp.minor_axis_length or 1e-6
        return {
            'label': label,
            'area': int(pp.area),
            'major_axis': float(maj),
            'minor_axis': float(minr),
            'aspect_ratio': float(maj / minr),
            'center_y': float(py),
            'center_x': float(px),
        }

    pole1 = measure_pole_region(pole1_y, pole1_x, mask, 'pole1')
    pole2 = measure_pole_region(pole2_y, pole2_x, mask, 'pole2')

    # Primary = larger area
    if pole1['area'] >= pole2['area']:
        primary, secondary = pole1, pole2
    else:
        primary, secondary = pole2, pole1
    primary['label'] = 'primary'
    secondary['label'] = 'secondary'

    return {
        'best_timepoint': best_t,
        'best_aspect_ratio': float(ars[best_t]),
        'whole_major': float(major),
        'whole_minor': float(minor),
        'primary_pole': primary,
        'secondary_pole': secondary,
        'pole_ratio': float(secondary['area'] / primary['area']) if primary['area'] > 0 else 0,
    }


def tension_over_time(stack_5d, masks, lyn_channel=1, n_sectors=12, n_radial=20):
    """
    Compute tissue tension metrics across all timepoints.

    Parameters
    ----------
    stack_5d : (T, Z, C, Y, X) or None — raw stack for Z-projection.
               If None, lyn_frames must be provided separately.
    masks    : (T, Y, X) bool — pescoid masks
    lyn_channel : channel index for LynTom

    Returns list of per-timepoint dicts.
    """
    T = masks.shape[0]
    results = []

    for t in range(T):
        mask = masks[t]
        if not mask.any():
            results.append({'time': t, 'anisotropy': np.nan, 'mean_tension': np.nan})
            continue

        # Z-project LynTom (max)
        if stack_5d is not None:
            lyn_raw = np.max(stack_5d[t, :, lyn_channel], axis=0).astype(float)
        else:
            continue

        lo, hi = np.percentile(lyn_raw, (1, 99.5))
        lyn_n = np.clip((lyn_raw - lo) / (hi - lo), 0, 1) if hi > lo else lyn_raw / max(lyn_raw.max(), 1)

        sector_result = compute_sector_tension(lyn_n, mask, n_sectors=n_sectors)
        radial_result = compute_radial_profile(lyn_n, mask, n_bins=n_radial)

        grad = filters.sobel(lyn_n)
        mean_tension = float(grad[mask].mean())

        row = {
            'time': t,
            'anisotropy': sector_result['anisotropy'] if sector_result else np.nan,
            'mean_tension': mean_tension,
            'mean_lyn': float(lyn_n[mask].mean()),
            'std_lyn': float(lyn_n[mask].std()),
        }

        # Add per-sector tension values
        if sector_result:
            for s in sector_result['sectors']:
                row[f'sector_{s["sector"]:02d}_tension'] = s['gradient_mag']
                row[f'sector_{s["sector"]:02d}_lyn'] = s['mean_lyn']

        results.append(row)

    return results
