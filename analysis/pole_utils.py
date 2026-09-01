import math
import numpy as np
from skimage import measure, segmentation
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


RADIAL_SAMPLES = 360
POLE_STD_THRESHOLD = 1.5


def radial_deviation_poles(mask, n_angles=RADIAL_SAMPLES, reference_r=None):
    """Compute radial distances from centroid to mask boundary for n_angles.

    Returns (centroid_x, centroid_y), r (array), dev (r - reference_r or zeros)
    """
    props = measure.regionprops(measure.label(mask))
    if not props:
        return None
    p = sorted(props, key=lambda x: x.area, reverse=True)[0]
    cy, cx = p.centroid
    angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    r = np.zeros(n_angles, dtype=float)
    H, W = mask.shape
    for i, a in enumerate(angles):
        dx, dy = math.cos(a), math.sin(a)
        rr, found = 0.0, False
        # step outwards from centroid until we leave the mask after having been inside
        for step in range(1, max(H, W) * 2):
            yy = int(round(cy + dy * step))
            xx = int(round(cx + dx * step))
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                break
            if mask[yy, xx]:
                rr = math.hypot(yy - cy, xx - cx)
                found = True
            elif found:
                break
        r[i] = rr if found else 0.0
    dev = np.zeros_like(r) if reference_r is None else r - reference_r
    return (cx, cy), r, dev


def detect_poles_from_dev(dev, angles, threshold_factor=POLE_STD_THRESHOLD, min_angle_separation_deg=20):
    """Detect pole angles from radial deviation array.

    This improved method smooths the deviation signal, finds local peaks using
    scipy.signal.find_peaks and applies combined thresholds: an adaptive
    mean+std threshold and a fraction-of-max threshold, plus a minimal
    prominence requirement. This helps detect small bud-like poles while
    rejecting noisy spurious peaks.

    Returns list of (angle_deg, deviation_value) sorted by angle.
    """
    dev = np.asarray(dev, dtype=float)
    if dev.size == 0 or np.all(dev == 0):
        return []

    # Smooth dev to suppress noise but keep real protrusions
    dev_smooth = gaussian_filter1d(dev, sigma=2)

    mu = np.mean(dev_smooth)
    sigma = np.std(dev_smooth)
    max_dev = np.max(dev_smooth)

    # Adaptive height threshold: choose the larger of mean+factor*std and a fraction of max_dev
    height_thresh = max(mu + sigma * threshold_factor, max_dev * 0.12)

    # Minimum prominence (relative to global std and max_dev)
    prom_thresh = max(sigma * 0.25, max_dev * 0.04)

    # find peaks with minimum distance equal to min_angle_separation_deg (converted to bins)
    min_dist_bins = max(1, int(round(len(dev) * (min_angle_separation_deg / 360.0))))

    peaks, props = find_peaks(dev_smooth, height=height_thresh, distance=min_dist_bins, prominence=prom_thresh)
    if peaks.size == 0:
        # As a fallback, try lower height but still require prominence
        peaks, props = find_peaks(dev_smooth, height=max_dev * 0.08, distance=min_dist_bins, prominence=prom_thresh * 0.5)
        if peaks.size == 0:
            return []

    prominences = peak_prominences(dev_smooth, peaks)[0]

    # collect peaks with their actual dev value (unsmoothed) and prominence
    peak_list = []
    for i, p in enumerate(peaks):
        val = float(dev[p])
        pr = float(prominences[i]) if i < len(prominences) else float(props['prominences'][i])
        peak_list.append((p, val, pr))

    # sort by prominence then by value
    peak_list = sorted(peak_list, key=lambda x: (x[2], x[1]), reverse=True)

    # Now enforce angular separation: keep strongest peaks at least min_angle_separation_deg apart
    kept = []
    for p, val, pr in peak_list:
        ang_deg = math.degrees(angles[p]) % 360
        if all(abs(((ang_deg - aa + 180) % 360) - 180) >= min_angle_separation_deg for aa, _, _ in kept):
            kept.append((ang_deg, val, pr))

    # Return sorted by angle with (angle_deg, deviation)
    return sorted([(a, v) for a, v, _ in kept], key=lambda x: x[0])
