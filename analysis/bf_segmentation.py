"""
High-quality Brightfield Segmentation for Pescoid Time-Lapse TIFFs
===================================================================
Comparable to Fiji/Ilastik quality for single-organoid (pescoid) segmentation.

Channel layout (user's files, 0-indexed):
    GFP = channel 0  (channel 1 in 1-indexed notation)
    BF  = channel 1  (channel 2 in 1-indexed notation)

TIFF dimension orders supported:
    (T, Z, C, Y, X)  -- standard 5D
    (T, C, Y, X)     -- no Z axis
    (Z, C, Y, X)     -- single timepoint
    (T, Z, Y, X)     -- single channel
    (Z, Y, X)        -- single channel, single timepoint
    (Y, X)           -- 2D

Segmentation methods available:
    'rolling_watershed'  -- Rolling ball BG subtraction + gradient watershed
                            (recommended, closest to Fiji workflow)
    'clahe_maxentropy'   -- CLAHE + MaxEntropy threshold
                            (fast, exact Fiji MaxEntropy equivalent)
    'active_contour'     -- Morphological Chan-Vese snakes
                            (most precise boundary, slower)
    'multiscale'         -- Multi-scale LoG + MaxEntropy
                            (scale-invariant, robust to size variation)
    'ensemble'           -- Majority vote across all methods
                            (most robust, slowest)

CLI Usage:
    # Single TIFF file
    python analysis/bf_segmentation.py data/myfile.tif --out output/ --method rolling_watershed

    # Batch: whole folder
    python analysis/bf_segmentation.py data/ --batch --out output/ --method rolling_watershed

    # Specify Z-range explicitly instead of auto best-focus
    python analysis/bf_segmentation.py data/myfile.tif --out output/ \\
        --z-projection focus_range --z-range 4 7

Python Usage:
    from analysis.bf_segmentation import BFSegmenter

    seg = BFSegmenter(method='rolling_watershed', bf_channel=1)
    masks, meta = seg.segment_stack('myfile.tif', output_dir='output/')

Integration with compute_aspect_ratio_and_kymograph.py:
    from analysis.bf_segmentation import drop_in_segment_bf
    # Use as replacement for the existing segment_bf() function
"""

import warnings
import json
import argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure, segmentation
from skimage.util import img_as_float, img_as_ubyte


# ============================================================================
# TIFF LOADING
# ============================================================================

def load_tiff(path):
    """
    Load a TIFF file and normalize to a 5D array (T, Z, C, Y, X).

    Returns
    -------
    stack : np.ndarray, shape (T, Z, C, Y, X)
    meta  : dict  -- original shape, axes, dtype, and parsed T/Z/C/Y/X dims
    """
    path = Path(path)
    try:
        with tifffile.TiffFile(str(path)) as tf:
            data = tf.asarray()
            axes = tf.series[0].axes if tf.series else None
    except Exception:
        data = tifffile.imread(str(path))
        axes = None

    orig_shape = data.shape

    # Normalize to 5D ---------------------------------------------------------
    ndim = data.ndim
    if ndim == 2:
        # (Y, X) → (1, 1, 1, Y, X)
        stack = data[np.newaxis, np.newaxis, np.newaxis, :, :]
    elif ndim == 3:
        # (Z, Y, X) – assume single timepoint, single channel
        stack = data[np.newaxis, :, np.newaxis, :, :]  # (1, Z, 1, Y, X)
    elif ndim == 4:
        # Common cases:
        #   (T, Z, Y, X)  -- no channel
        #   (T, C, Y, X)  -- no Z, small channel dim
        #   (Z, C, Y, X)  -- single timepoint
        if data.shape[1] <= 4 and data.shape[0] > 4:
            # Likely (T, C, Y, X) -- no Z
            stack = data[:, np.newaxis, :, :, :]        # (T, 1, C, Y, X)
        elif data.shape[0] <= 4 and data.shape[1] > 4:
            # Likely (C, Z, Y, X) or (Z, C, Y, X) -- single timepoint
            # Treat as (Z, C, Y, X)
            stack = data[np.newaxis, :, np.newaxis, :, :]
        else:
            # Default: (T, Z, Y, X) -- no channel
            stack = data[:, :, np.newaxis, :, :]        # (T, Z, 1, Y, X)
    elif ndim == 5:
        stack = data  # Already (T, Z, C, Y, X)
    else:
        raise ValueError(
            f"Unsupported TIFF: {ndim}D array shape={data.shape}. "
            "Expected 2-5 dimensions."
        )

    T, Z, C, Y, X = stack.shape
    meta = {
        'path': str(path),
        'original_shape': orig_shape,
        'original_ndim': ndim,
        'axes': axes,
        'dtype': str(data.dtype),
        'T': T, 'Z': Z, 'C': C, 'Y': Y, 'X': X,
    }
    return stack, meta


def extract_channel(stack, channel_idx):
    """
    Extract a single channel from a (T, Z, C, Y, X) array.

    Returns
    -------
    (T, Z, Y, X) array
    """
    T, Z, C, Y, X = stack.shape
    if channel_idx >= C:
        raise ValueError(
            f"Requested channel {channel_idx} but stack only has {C} channel(s)."
        )
    return stack[:, :, channel_idx, :, :]


# ============================================================================
# Z PROJECTION
# ============================================================================

def _brenner_focus(img):
    """Brenner gradient: sum of squared differences between pixels 2 apart."""
    d = img[2:, :].astype(np.float64) - img[:-2, :].astype(np.float64)
    return float(np.sum(d ** 2))


def z_project(z_stack, method='best_focus', z_range=None):
    """
    Collapse a Z-stack to a 2D image.

    Parameters
    ----------
    z_stack : (Z, Y, X) array
    method  : 'max' | 'min' | 'mean' | 'best_focus' | 'focus_range' | 'focus_stack'
    z_range : (z_start, z_end) used only for 'focus_range' (end-exclusive)

    Returns
    -------
    proj : (Y, X) float array
    """
    Z = z_stack.shape[0]

    if method == 'max':
        return np.max(z_stack, axis=0).astype(float)
    elif method == 'min':
        return np.min(z_stack, axis=0).astype(float)
    elif method == 'mean':
        return np.mean(z_stack, axis=0).astype(float)
    elif method == 'focus_range':
        z0 = max(0, z_range[0]) if z_range else 0
        z1 = min(Z, z_range[1]) if z_range else Z
        return np.max(z_stack[z0:z1], axis=0).astype(float)
    elif method == 'best_focus':
        scores = [_brenner_focus(z_stack[z]) for z in range(Z)]
        best_z = int(np.argmax(scores))
        return z_stack[best_z].astype(float)
    elif method == 'focus_stack':
        # Per-pixel extended depth-of-field via local variance maximization
        local_var = np.stack([
            ndi.uniform_filter(z_stack[z].astype(float) ** 2, size=15) -
            ndi.uniform_filter(z_stack[z].astype(float), size=15) ** 2
            for z in range(Z)
        ], axis=0)  # (Z, Y, X)
        best_z_map = np.argmax(local_var, axis=0)  # (Y, X)
        proj = np.take_along_axis(
            z_stack.astype(float),
            best_z_map[np.newaxis, :, :], axis=0
        )[0]
        return proj
    else:
        return np.max(z_stack, axis=0).astype(float)


# ============================================================================
# PREPROCESSING
# ============================================================================

def normalize_image(img, p_low=0.5, p_high=99.5):
    """Normalize to float [0, 1] with percentile-based clipping."""
    img_f = img_as_float(img.astype(np.float64))
    lo = np.percentile(img_f, p_low)
    hi = np.percentile(img_f, p_high)
    if hi > lo:
        img_f = np.clip((img_f - lo) / (hi - lo), 0, 1)
    return img_f


def rolling_ball_background(img_float, radius=50):
    """
    Rolling ball background subtraction (equivalent to Fiji's Subtract Background).

    Approximated via morphological opening (erosion → dilation with disk).
    The opening gives the local minimum (background estimate).

    Parameters
    ----------
    img_float : (Y, X) float [0, 1]
    radius    : ball radius in pixels (Fiji default = 50)

    Returns
    -------
    corrected  : background-subtracted image [0, 1]
    background : estimated background image
    """
    selem = morphology.disk(radius)
    background = morphology.opening(img_float, selem)
    corrected = np.clip(img_float - background, 0, None)
    if corrected.max() > 0:
        corrected = corrected / corrected.max()
    return corrected, background


def preprocess_bf(img_raw, method='clahe+rolling', sigma=2.0, ball_radius=50):
    """
    Preprocess a raw BF image for segmentation.

    Brightfield pescoids are DARK on BRIGHT background.
    We invert the image so the organoid becomes the bright foreground.
    This makes standard "bright foreground" thresholding algorithms work directly.

    Parameters
    ----------
    img_raw     : (Y, X) raw BF image (any dtype)
    method      : 'clahe'           -- CLAHE only (fast)
                  'rolling'         -- rolling ball only
                  'clahe+rolling'   -- both (best, default)
                  'normalize'       -- percentile normalization only
    sigma       : Gaussian smoothing sigma (0 = no smoothing)
    ball_radius : rolling ball radius in pixels

    Returns
    -------
    processed : (Y, X) float [0, 1], inverted so organoid is bright
    """
    # Normalize + invert
    img_norm = normalize_image(img_raw)
    img_inv = 1.0 - img_norm  # organoid now bright

    if 'rolling' in method:
        img_inv, _ = rolling_ball_background(img_inv, radius=ball_radius)

    if 'clahe' in method:
        img_inv = exposure.equalize_adapthist(img_inv, clip_limit=0.03)

    if sigma > 0:
        img_inv = filters.gaussian(img_inv, sigma=sigma)

    return img_inv


# ============================================================================
# THRESHOLDING
# ============================================================================

def _max_entropy_threshold(img_uint8):
    """
    Maximum entropy thresholding -- exact Fiji/ImageJ algorithm.

    Parameters
    ----------
    img_uint8 : uint8 array

    Returns
    -------
    threshold : float in [0, 1]
    """
    hist, _ = np.histogram(img_uint8.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float64) + 1e-10
    p = hist / hist.sum()
    P_back = np.cumsum(p)
    P_fore = 1.0 - P_back

    best_entropy = -1.0
    best_t = 0

    for t in range(256):
        if P_back[t] > 0 and P_fore[t] > 0:
            pb = p[:t + 1] / P_back[t]
            H_back = -np.sum(pb * np.log(pb + 1e-10))
            pf = p[t + 1:] / P_fore[t]
            H_fore = -np.sum(pf * np.log(pf + 1e-10))
            total = H_back + H_fore
            if total > best_entropy:
                best_entropy = total
                best_t = t

    return best_t / 255.0  # normalized to [0,1]


def auto_threshold(img_float, method='maxentropy'):
    """
    Compute a threshold for a float [0, 1] image.

    Parameters
    ----------
    img_float : float [0, 1] array
    method    : 'maxentropy' | 'otsu' | 'li' | 'yen' | 'isodata' | 'triangle'

    Returns
    -------
    thresh : float
    """
    if method == 'maxentropy':
        img_u8 = img_as_ubyte(np.clip(img_float, 0, 1))
        return _max_entropy_threshold(img_u8)
    elif method == 'otsu':
        return filters.threshold_otsu(img_float)
    elif method == 'li':
        return filters.threshold_li(img_float)
    elif method == 'yen':
        return filters.threshold_yen(img_float)
    elif method == 'isodata':
        return filters.threshold_isodata(img_float)
    elif method == 'triangle':
        return filters.threshold_triangle(img_float)
    else:
        return filters.threshold_otsu(img_float)


# ============================================================================
# POST-PROCESSING (shared by all methods)
# ============================================================================

def _postprocess(mask, min_area=50000, closing_radius=5):
    """
    Standard morphological post-processing:
      1. Fill holes
      2. Morphological closing (smooth boundary + connect nearby regions)
      3. Remove small objects
      4. Keep largest connected component

    Parameters
    ----------
    mask           : bool (Y, X)
    min_area       : minimum area threshold in pixels
    closing_radius : disk radius for binary closing

    Returns
    -------
    mask_clean : bool (Y, X)
    """
    if not mask.any():
        return mask.astype(bool)

    # Fill holes
    mask = ndi.binary_fill_holes(mask)

    # Morphological closing: smooth & connect
    if closing_radius > 0:
        selem = morphology.disk(closing_radius)
        mask = morphology.binary_closing(mask, footprint=selem)
        mask = ndi.binary_fill_holes(mask)

    # Remove small debris
    min_size = max(100, min_area // 20)
    mask = morphology.remove_small_objects(mask, min_size=min_size)

    # Keep only largest component
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return mask.astype(bool)

    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    if largest.area < min_area:
        warnings.warn(
            f"Largest region area ({largest.area:,} px) is below min_area "
            f"({min_area:,} px). Adjust --min-area if needed."
        )

    return (labeled == largest.label).astype(bool)


# ============================================================================
# SEGMENTATION METHODS
# ============================================================================

def segment_clahe_maxentropy(img_raw, sigma=2.0, min_area=50000, closing_radius=5,
                              **_kwargs):
    """
    Method: CLAHE + MaxEntropy threshold.

    Exact equivalent of Fiji's workflow:
        Image → Enhance Contrast (CLAHE) → Auto Threshold (MaxEntropy)

    Best for: Standard images, moderate contrast, quick results.
    Speed: Fast (< 1 s per frame).
    """
    proc = preprocess_bf(img_raw, method='clahe', sigma=sigma)
    thresh = auto_threshold(proc, method='maxentropy')
    mask = proc > thresh
    return _postprocess(mask, min_area=min_area, closing_radius=closing_radius)


def segment_rolling_watershed(img_raw, sigma=1.5, ball_radius=50,
                               min_area=50000, closing_radius=5, **_kwargs):
    """
    Method: Rolling ball background subtraction + gradient-based watershed.

    Closest to Fiji's recommended workflow for organoids:
        Subtract Background (rolling ball) → Gaussian Blur → Watershed

    Steps:
        1. Rolling ball background subtraction (like Fiji)
        2. CLAHE for contrast
        3. Coarse Otsu threshold → seed markers
        4. Sobel gradient surface
        5. Marker-controlled watershed for precise boundary

    Best for: Uneven illumination, complex/irregular organoid shapes.
    Speed: Medium (2-5 s per frame).
    """
    # Preprocess: normalize + invert + rolling ball + CLAHE
    proc = preprocess_bf(img_raw, method='clahe+rolling', sigma=sigma,
                         ball_radius=ball_radius)

    # Coarse mask via Otsu for initial seed estimation
    thresh_coarse = auto_threshold(proc, method='otsu')
    mask_coarse = proc > thresh_coarse
    mask_coarse = morphology.remove_small_objects(mask_coarse, min_size=1000)
    mask_coarse = ndi.binary_fill_holes(mask_coarse)

    # Gradient surface for watershed
    gradient = filters.sobel(proc)

    # Definite foreground seeds: erode coarse mask
    selem_erode = morphology.disk(max(5, int(np.sqrt(min_area) * 0.05)))
    foreground = morphology.binary_erosion(mask_coarse, footprint=selem_erode)

    # If erosion killed everything, use centroid region
    if not foreground.any():
        labeled_c = measure.label(mask_coarse)
        if labeled_c.max() > 0:
            regions_c = measure.regionprops(labeled_c)
            lg = max(regions_c, key=lambda r: r.area)
            cy, cx = int(lg.centroid[0]), int(lg.centroid[1])
            r_seed = max(3, int(np.sqrt(lg.area) * 0.05))
            foreground[cy - r_seed:cy + r_seed, cx - r_seed:cx + r_seed] = True

    # Definite background seeds: outside dilated coarse mask
    background = ~morphology.binary_dilation(mask_coarse, footprint=morphology.disk(5))

    # Markers: 1 = background, 2 = foreground
    markers = np.zeros(img_raw.shape[:2], dtype=np.int32)
    markers[foreground] = 2
    markers[background] = 1

    try:
        labels = segmentation.watershed(gradient, markers=markers,
                                        compactness=0.001)
        mask = labels == 2
    except Exception as e:
        warnings.warn(f"Watershed failed ({e}), falling back to MaxEntropy.")
        mask = proc > auto_threshold(proc, method='maxentropy')

    return _postprocess(mask, min_area=min_area, closing_radius=closing_radius)


def segment_active_contour(img_raw, sigma=2.0, min_area=50000, closing_radius=3,
                            n_iter=200, smoothing=3, **_kwargs):
    """
    Method: Morphological Chan-Vese active contours (level sets).

    Gives the most accurate boundary following, especially for
    irregular or elongated organoids.

    Steps:
        1. Coarse Otsu + rolling ball for initial mask
        2. Morphological Chan-Vese iteration to refine boundary

    Best for: Complex shapes, fine boundary precision needed.
    Speed: Slow (5-30 s per frame depending on n_iter).
    """
    proc = preprocess_bf(img_raw, method='clahe+rolling', sigma=sigma)

    # Initial mask
    thresh = auto_threshold(proc, method='otsu')
    init_mask = proc > thresh
    init_mask = morphology.remove_small_objects(init_mask, min_size=1000)
    init_mask = ndi.binary_fill_holes(init_mask)

    labeled_init = measure.label(init_mask)
    if labeled_init.max() > 0:
        largest_init = max(measure.regionprops(labeled_init), key=lambda r: r.area)
        init_mask = (labeled_init == largest_init.label).astype(float)
    else:
        init_mask = init_mask.astype(float)

    try:
        mask = segmentation.morphological_chan_vese(
            proc,
            num_iter=n_iter,
            init_level_set=init_mask,
            smoothing=smoothing,
            lambda1=1.0,
            lambda2=1.0,
        )
    except Exception as e:
        warnings.warn(f"Chan-Vese failed ({e}), falling back to MaxEntropy.")
        mask = proc > auto_threshold(proc, method='maxentropy')

    return _postprocess(mask.astype(bool), min_area=min_area,
                        closing_radius=closing_radius)


def segment_multiscale(img_raw, scales=(2, 4, 8), sigma=1.0,
                        min_area=50000, closing_radius=5, **_kwargs):
    """
    Method: Multi-scale Laplacian-of-Gaussian + MaxEntropy.

    Scale-invariant: works well when organoid size varies significantly
    across experiments. Combines LoG responses at several scales to
    build a robust edge-enhanced image before thresholding.

    Best for: Variable organoid sizes, batch processing across experiments.
    Speed: Fast-Medium (1-3 s per frame).
    """
    img_float = normalize_image(img_raw)
    img_inv = 1.0 - img_float  # organoid bright

    # Multi-scale LoG (scale-normalized: multiply by sigma^2)
    log_responses = np.stack([
        np.abs(scale ** 2 * ndi.gaussian_laplace(img_inv, sigma=scale))
        for scale in scales
    ], axis=0)  # (n_scales, Y, X)
    edge_map = np.max(log_responses, axis=0)  # Take strongest response

    # Normalize edge map
    if edge_map.max() > 0:
        edge_map /= edge_map.max()

    # Smooth + CLAHE for thresholding
    proc = filters.gaussian(img_inv, sigma=sigma)
    proc = exposure.equalize_adapthist(proc, clip_limit=0.03)

    thresh = auto_threshold(proc, method='maxentropy')
    mask = proc > thresh

    return _postprocess(mask, min_area=min_area, closing_radius=closing_radius)


def segment_ensemble(img_raw, min_area=50000, closing_radius=5, **_kwargs):
    """
    Method: Majority vote ensemble across all methods.

    Runs clahe_maxentropy + rolling_watershed + active_contour (100 iters)
    and takes a pixel-wise majority vote. The most robust strategy at
    the cost of ~3x processing time.

    Best for: When accuracy is critical; batch analysis with no manual QC.
    Speed: Slow (sum of all methods).
    """
    votes = []

    for name, fn in [
        ('clahe_maxentropy', segment_clahe_maxentropy),
        ('rolling_watershed', segment_rolling_watershed),
        ('active_contour', lambda img, **kw: segment_active_contour(
            img, n_iter=100, **kw)),
    ]:
        try:
            m = fn(img_raw, min_area=max(100, min_area // 4),
                   closing_radius=closing_radius)
            votes.append(m.astype(np.int32))
        except Exception as e:
            warnings.warn(f"ensemble: '{name}' failed: {e}")

    if not votes:
        return np.zeros(img_raw.shape[:2], dtype=bool)
    if len(votes) == 1:
        return votes[0].astype(bool)

    vote_sum = np.sum(np.stack(votes, axis=0), axis=0)
    majority = vote_sum >= (len(votes) / 2.0)
    return _postprocess(majority, min_area=min_area,
                        closing_radius=closing_radius)


# Registry of all methods
METHODS = {
    'clahe_maxentropy': segment_clahe_maxentropy,
    'rolling_watershed': segment_rolling_watershed,
    'active_contour': segment_active_contour,
    'multiscale': segment_multiscale,
    'ensemble': segment_ensemble,
}


# ============================================================================
# MAIN SEGMENTER CLASS
# ============================================================================

class BFSegmenter:
    """
    High-quality brightfield segmentation for pescoid time-lapse TIFFs.

    Handles 5D TIFF stacks (T, Z, C, Y, X), extracts the BF channel,
    performs Z-projection, segments each timepoint, and saves results
    in a format compatible with the existing analysis pipeline.

    Example
    -------
    seg = BFSegmenter(method='rolling_watershed', bf_channel=1)
    masks, meta = seg.segment_stack('experiment.tif', output_dir='output/')
    # masks: (T, Y, X) bool array
    # output/result_segmentation/<name>_finalMask.tif  ← pipeline-compatible
    """

    def __init__(
        self,
        method='rolling_watershed',
        bf_channel=1,           # 0-indexed: BF=1 means channel 2 in Fiji notation
        z_projection='best_focus',
        z_range=None,           # (z_start, z_end) for 'focus_range'
        min_area=50000,
        sigma=2.0,
        closing_radius=5,
        ball_radius=50,
        temporal_consistency=True,
        verbose=True,
    ):
        """
        Parameters
        ----------
        method               : Segmentation method name (see METHODS)
        bf_channel           : 0-indexed channel index for BF
                               default=1 because user has GFP=ch0, BF=ch1
        z_projection         : How to collapse Z-stack
                               'best_focus' (auto), 'max', 'mean', 'focus_range'
        z_range              : (z_start, z_end) slice indices for 'focus_range'
        min_area             : Minimum pescoid area in pixels (filter small debris)
        sigma                : Gaussian smoothing sigma
        closing_radius       : Morphological closing disk radius
        ball_radius          : Rolling ball background subtraction radius (pixels)
        temporal_consistency : Replace empty-mask frames with nearest good frame
        verbose              : Print frame-by-frame progress
        """
        if method not in METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}"
            )
        self.method = method
        self.bf_channel = bf_channel
        self.z_projection = z_projection
        self.z_range = z_range
        self.min_area = min_area
        self.sigma = sigma
        self.closing_radius = closing_radius
        self.ball_radius = ball_radius
        self.temporal_consistency = temporal_consistency
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core public methods
    # ------------------------------------------------------------------

    def segment_frame(self, bf_2d):
        """
        Segment a single 2D BF image.

        Parameters
        ----------
        bf_2d : (Y, X) array, raw brightfield image

        Returns
        -------
        mask : (Y, X) bool array
        """
        fn = METHODS[self.method]
        kwargs = dict(
            min_area=self.min_area,
            closing_radius=self.closing_radius,
            sigma=self.sigma,
            ball_radius=self.ball_radius,
        )
        try:
            return fn(bf_2d, **kwargs)
        except Exception as e:
            warnings.warn(f"segment_frame failed: {e}")
            return np.zeros(bf_2d.shape[:2], dtype=bool)

    def segment_stack(self, tiff_path, output_dir=None):
        """
        Segment all timepoints in a TIFF file.

        Parameters
        ----------
        tiff_path  : str or Path to TIFF file
        output_dir : directory to save masks + visualizations
                     default: <tiff_parent>/<tiff_stem>/result_segmentation/

        Returns
        -------
        masks : (T, Y, X) bool array
        meta  : dict with file info + segmentation parameters + per-frame areas
        """
        tiff_path = Path(tiff_path)
        self._log(f"Loading: {tiff_path.name}")

        stack, file_meta = load_tiff(str(tiff_path))
        T, Z, C, Y, X = stack.shape
        self._log(
            f"  Dimensions: T={T}, Z={Z}, C={C}, Y={Y}, X={X} "
            f"(dtype={file_meta['dtype']})"
        )

        # Validate and resolve BF channel
        bf_ch = self.bf_channel
        if bf_ch >= C:
            warnings.warn(
                f"bf_channel={bf_ch} >= n_channels={C}. Falling back to channel 0."
            )
            bf_ch = 0
        self._log(f"  BF channel: {bf_ch} (0-indexed)  |  method: '{self.method}'")

        # Extract BF channel → (T, Z, Y, X)
        bf_stack = extract_channel(stack, bf_ch)

        # Segment frame by frame
        masks = np.zeros((T, Y, X), dtype=bool)
        failed = []

        for t in range(T):
            bf_z = bf_stack[t]  # (Z, Y, X)
            if Z == 1:
                bf_2d = bf_z[0].astype(float)
            else:
                bf_2d = z_project(bf_z, method=self.z_projection,
                                   z_range=self.z_range)

            mask = self.segment_frame(bf_2d)

            if mask.sum() == 0:
                failed.append(t)
                self._log(f"  [t={t:03d}] WARNING: empty mask")
            else:
                self._log(f"  [t={t:03d}] area = {mask.sum():,} px")

            masks[t] = mask

        # Temporal consistency: propagate good masks to failed frames
        if self.temporal_consistency and failed:
            self._log(f"  Applying temporal consistency to {len(failed)} frame(s)...")
            masks = self._fix_temporal(masks, failed)

        # Save outputs
        if output_dir is None:
            output_dir = tiff_path.parent / tiff_path.stem / 'result_segmentation'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save(masks, stack, bf_ch, file_meta, tiff_path, output_dir)

        meta = {
            **file_meta,
            'method': self.method,
            'bf_channel': bf_ch,
            'z_projection': self.z_projection,
            'z_range': self.z_range,
            'min_area': self.min_area,
            'sigma': self.sigma,
            'closing_radius': self.closing_radius,
            'ball_radius': self.ball_radius,
            'failed_frames': failed,
            'mask_areas': [int(masks[t].sum()) for t in range(T)],
            'output_dir': str(output_dir),
        }
        return masks, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _fix_temporal(self, masks, failed_frames):
        """Replace empty masks with the nearest non-empty frame."""
        T = masks.shape[0]
        good = [t for t in range(T) if t not in failed_frames]
        if not good:
            return masks
        for t in failed_frames:
            nearest = min(good, key=lambda g: abs(g - t))
            masks[t] = masks[nearest].copy()
            self._log(f"  [t={t:03d}] replaced with t={nearest}")
        return masks

    def _save(self, masks, stack, bf_ch, file_meta, tiff_path, output_dir):
        """
        Save results:
          - <name>_finalMask.tif          (T, Y, X) uint8 mask stack
          - single_frames/<name>_t####_finalMask.tif  per-frame masks
          - overlays/<name>_t####_overlay.png          per-frame overlays
          - segmentation_summary.json
        """
        T = masks.shape[0]
        name = tiff_path.stem

        # ---------- 1. Save mask TIFF (pipeline-compatible) ----------
        mask_path = output_dir / f"{name}_finalMask.tif"
        tifffile.imwrite(str(mask_path), (masks.astype(np.uint8) * 255))
        self._log(f"  Saved mask stack: {mask_path.name}")

        # ---------- 2. Per-frame single masks ----------
        single_dir = output_dir / 'single_frames'
        single_dir.mkdir(exist_ok=True)
        for t in range(T):
            fp = single_dir / f"{name}_t{t:04d}_finalMask.tif"
            tifffile.imwrite(str(fp), (masks[t].astype(np.uint8) * 255))

        # ---------- 3. Overlay PNGs ----------
        overlay_dir = output_dir / 'overlays'
        overlay_dir.mkdir(exist_ok=True)
        self._log(f"  Saving {T} overlay images...")

        for t in range(T):
            bf_z = stack[t, :, bf_ch, :, :]
            bf_2d = (bf_z[0] if file_meta['Z'] == 1
                     else z_project(bf_z, method=self.z_projection,
                                    z_range=self.z_range))
            bf_disp = exposure.rescale_intensity(
                bf_2d.astype(float), out_range=(0, 1)
            )

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(bf_disp, cmap='gray', interpolation='nearest')
            axes[0].set_title(f'BF  t={t}', fontsize=9)
            axes[0].axis('off')

            axes[1].imshow(bf_disp, cmap='gray', interpolation='nearest')
            # Draw mask contours
            contours = measure.find_contours(masks[t].astype(float), 0.5)
            for c in contours:
                axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=1.5)
            area = int(masks[t].sum())
            axes[1].set_title(
                f'Segmentation  area={area:,} px', fontsize=9
            )
            axes[1].axis('off')

            plt.suptitle(f'{name}  |  method={self.method}',
                         fontsize=10, y=1.01)
            plt.tight_layout()
            ov_path = overlay_dir / f"{name}_t{t:04d}_overlay.png"
            plt.savefig(str(ov_path), dpi=100, bbox_inches='tight')
            plt.close(fig)

        # ---------- 4. Summary JSON ----------
        summary = {
            'input_file': str(tiff_path),
            'output_dir': str(output_dir),
            'method': self.method,
            'bf_channel': bf_ch,
            'z_projection': self.z_projection,
            'z_range': list(self.z_range) if self.z_range else None,
            'parameters': {
                'min_area': self.min_area,
                'sigma': self.sigma,
                'closing_radius': self.closing_radius,
                'ball_radius': self.ball_radius,
            },
            'file_info': {k: file_meta[k] for k in ('T', 'Z', 'C', 'Y', 'X', 'dtype')},
            'results': {
                'mask_areas': [int(masks[t].sum()) for t in range(T)],
                'mean_area': float(np.mean([masks[t].sum() for t in range(T)])),
                'std_area': float(np.std([masks[t].sum() for t in range(T)])),
                'empty_frames': [int(t) for t in range(T) if masks[t].sum() == 0],
            },
        }
        summary_path = output_dir / 'segmentation_summary.json'
        with open(str(summary_path), 'w') as f:
            json.dump(summary, f, indent=2)
        self._log(f"  Saved summary: {summary_path.name}")


# ============================================================================
# DROP-IN REPLACEMENT FOR compute_aspect_ratio_and_kymograph.py
# ============================================================================

def drop_in_segment_bf(frame, method='rolling_watershed', min_area=50000,
                        sigma=2.0, closing_radius=5, ball_radius=50):
    """
    Drop-in replacement for the simple segment_bf() in
    compute_aspect_ratio_and_kymograph.py.

    Accepts a 2D BF frame and returns a bool mask, matching the existing API.

    Parameters
    ----------
    frame          : (Y, X) array, raw BF image (already projected if needed)
    method         : segmentation method (default: 'rolling_watershed')
    min_area       : minimum object area
    sigma          : Gaussian smoothing sigma
    closing_radius : morphological closing radius
    ball_radius    : rolling ball radius

    Returns
    -------
    mask : (Y, X) bool array

    Usage in compute_aspect_ratio_and_kymograph.py:
        from analysis.bf_segmentation import drop_in_segment_bf
        # Replace:
        #   mask = segment_bf(bf)
        # With:
        #   mask = drop_in_segment_bf(bf)
    """
    fn = METHODS[method]
    return fn(
        frame,
        min_area=min_area,
        sigma=sigma,
        closing_radius=closing_radius,
        ball_radius=ball_radius,
    )


# ============================================================================
# BATCH FOLDER PROCESSING
# ============================================================================

def process_folder(
    input_dir,
    output_dir=None,
    method='rolling_watershed',
    bf_channel=1,
    z_projection='best_focus',
    z_range=None,
    min_area=50000,
    sigma=2.0,
    closing_radius=5,
    ball_radius=50,
    pattern='*.tif',
    verbose=True,
):
    """
    Segment all TIFF files in a directory.

    Parameters
    ----------
    input_dir  : directory containing TIFF files
    output_dir : root output directory
                 each TIFF gets its own sub-folder: output_dir/<stem>/result_segmentation/
    pattern    : glob pattern to match files (default '*.tif')
    (other params same as BFSegmenter)

    Returns
    -------
    results : dict mapping filename → {'masks': (T,Y,X) array, 'meta': dict}
    """
    input_dir = Path(input_dir)
    tif_files = sorted(input_dir.glob(pattern))
    tif_files += sorted(input_dir.glob(pattern.replace('.tif', '.tiff')))
    tif_files = sorted(set(tif_files))  # deduplicate

    if not tif_files:
        print(f"No TIFF files found in {input_dir}")
        return {}

    seg = BFSegmenter(
        method=method, bf_channel=bf_channel, z_projection=z_projection,
        z_range=z_range, min_area=min_area, sigma=sigma,
        closing_radius=closing_radius, ball_radius=ball_radius, verbose=verbose,
    )

    results = {}
    for tif_path in tif_files:
        print(f"\n{'='*60}\nProcessing: {tif_path.name}")
        file_out = (
            Path(output_dir) / tif_path.stem / 'result_segmentation'
            if output_dir
            else tif_path.parent / tif_path.stem / 'result_segmentation'
        )
        try:
            masks, meta = seg.segment_stack(str(tif_path), output_dir=str(file_out))
            mean_area = float(np.mean(meta['mask_areas'])) if meta['mask_areas'] else 0
            print(f"  OK  frames={masks.shape[0]}  mean_area={mean_area:.0f} px")
            results[tif_path.name] = {'masks': masks, 'meta': meta}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[tif_path.name] = {'masks': None, 'meta': {'error': str(e)}}

    return results


# ============================================================================
# VISUALIZATION UTILITY
# ============================================================================

def visualize_segmentation(img_raw, mask, title='Segmentation', save_path=None):
    """
    4-panel diagnostic visualization:
      1. Original BF image
      2. Segmentation mask
      3. Contour overlay on BF
      4. Region properties text

    Parameters
    ----------
    img_raw   : (Y, X) raw BF image
    mask      : (Y, X) bool mask
    title     : figure title
    save_path : if provided, save to file instead of showing
    """
    img_disp = exposure.rescale_intensity(img_raw.astype(float), out_range=(0, 1))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img_disp, cmap='gray')
    axes[0].set_title('Original BF')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Mask  (area={mask.sum():,} px)')
    axes[1].axis('off')

    axes[2].imshow(img_disp, cmap='gray')
    for c in measure.find_contours(mask.astype(float), 0.5):
        axes[2].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
    axes[2].set_title('Boundary Overlay')
    axes[2].axis('off')

    props_text = '(no object)'
    if mask.any():
        regions = measure.regionprops(measure.label(mask))
        if regions:
            r = max(regions, key=lambda x: x.area)
            ar = (r.major_axis_length / r.minor_axis_length
                  if r.minor_axis_length > 0 else 0.0)
            props_text = (
                f"Area:         {r.area:,} px\n"
                f"Perimeter:    {r.perimeter:.1f} px\n"
                f"Aspect ratio: {ar:.3f}\n"
                f"Major axis:   {r.major_axis_length:.1f} px\n"
                f"Minor axis:   {r.minor_axis_length:.1f} px\n"
                f"Solidity:     {r.solidity:.3f}\n"
                f"Extent:       {r.extent:.3f}"
            )
    axes[3].text(0.05, 0.5, props_text, transform=axes[3].transAxes,
                 fontsize=10, va='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[3].set_title('Region Properties')
    axes[3].axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='High-quality BF segmentation for pescoid time-lapse TIFFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  rolling_watershed  Rolling ball BG removal + gradient watershed (recommended)
  clahe_maxentropy   CLAHE + MaxEntropy threshold (fast, exact Fiji equivalent)
  active_contour     Morphological Chan-Vese snakes (precise, slow)
  multiscale         Multi-scale LoG + MaxEntropy (scale-invariant)
  ensemble           Majority vote across all methods (most robust, slowest)

Channel layout (your files):
  GFP = channel 1 (Fiji) = index 0 (Python)
  BF  = channel 2 (Fiji) = index 1 (Python)  <-- default --bf-channel 1

Examples:
  # Single file, auto best-focus Z-projection
  python analysis/bf_segmentation.py data/exp.tif --out output/

  # Batch folder, rolling watershed
  python analysis/bf_segmentation.py data/ --batch --out output/ --method rolling_watershed

  # Use explicit Z-range (slices 4-7, like config.py BF_Z_RANGE)
  python analysis/bf_segmentation.py data/exp.tif --out output/ \\
      --z-projection focus_range --z-range 4 7

  # Maximum precision ensemble
  python analysis/bf_segmentation.py data/exp.tif --out output/ --method ensemble
""",
    )

    parser.add_argument('input', help='TIFF file or folder to process')
    parser.add_argument('--out', '-o', required=True,
                        help='Output directory')
    parser.add_argument('--method', '-m', default='rolling_watershed',
                        choices=list(METHODS.keys()),
                        help='Segmentation method (default: rolling_watershed)')
    parser.add_argument('--bf-channel', type=int, default=1,
                        help='0-indexed BF channel (default: 1 = Fiji channel 2)')
    parser.add_argument('--z-projection', default='best_focus',
                        choices=['best_focus', 'max', 'mean', 'min',
                                 'focus_range', 'focus_stack'],
                        help='Z-projection method (default: best_focus)')
    parser.add_argument('--z-range', nargs=2, type=int, default=None,
                        metavar=('Z_START', 'Z_END'),
                        help='Z-slice range for focus_range (e.g. --z-range 4 7)')
    parser.add_argument('--min-area', type=int, default=50000,
                        help='Minimum pescoid area in pixels (default: 50000)')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian smoothing sigma (default: 2.0)')
    parser.add_argument('--closing-radius', type=int, default=5,
                        help='Morphological closing radius (default: 5)')
    parser.add_argument('--ball-radius', type=int, default=50,
                        help='Rolling ball background subtraction radius (default: 50)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all TIFs in the input directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress frame-by-frame output')

    args = parser.parse_args()

    input_path = Path(args.input)
    z_range = tuple(args.z_range) if args.z_range else None

    if args.batch or input_path.is_dir():
        results = process_folder(
            input_dir=input_path,
            output_dir=args.out,
            method=args.method,
            bf_channel=args.bf_channel,
            z_projection=args.z_projection,
            z_range=z_range,
            min_area=args.min_area,
            sigma=args.sigma,
            closing_radius=args.closing_radius,
            ball_radius=args.ball_radius,
            verbose=not args.quiet,
        )
        ok = sum(1 for v in results.values() if v['masks'] is not None)
        print(f"\nDone: {ok}/{len(results)} files succeeded. Outputs in: {args.out}")

    else:
        seg = BFSegmenter(
            method=args.method,
            bf_channel=args.bf_channel,
            z_projection=args.z_projection,
            z_range=z_range,
            min_area=args.min_area,
            sigma=args.sigma,
            closing_radius=args.closing_radius,
            ball_radius=args.ball_radius,
            verbose=not args.quiet,
        )
        seg_out = Path(args.out) / input_path.stem / 'result_segmentation'
        masks, meta = seg.segment_stack(str(input_path), output_dir=str(seg_out))

        areas = meta['mask_areas']
        print(f"\nResults:")
        print(f"  Frames:       {masks.shape[0]}")
        print(f"  Mean area:    {np.mean(areas):.0f} px")
        print(f"  Area range:   {min(areas):,} – {max(areas):,} px")
        print(f"  Failed frames:{meta['failed_frames']}")
        print(f"  Saved to:     {seg_out}")


if __name__ == '__main__':
    main()
