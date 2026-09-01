import os
import sys
import argparse
import json
from typing import List, Tuple

import numpy as np
from skimage import measure, morphology, filters
from scipy import interpolate
import tifffile as tiff
import matplotlib.pyplot as plt

# Add analysis directory to path for pole detection module and new segmenter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis'))
from pole_detection import detect_and_analyze_poles, visualize_pole_detection

# Optional: high-quality BF segmentation (drop-in replacement for segment_bf)
try:
    from bf_segmentation import drop_in_segment_bf as _improved_segment_bf
    _HAS_IMPROVED_SEG = True
except ImportError:
    _HAS_IMPROVED_SEG = False

"""
Compute aspect ratio over time and perimeter-change kymograph (positive/negative changes).

Assumptions:
- If a mask directory is provided (e.g., result_segmentation), we will ONLY analyze mask files
    whose filename ends with "finalMask" (case-insensitive) and typical extensions (.tif, .tiff).
- Otherwise, we fall back to reading TIF images and segmenting BF with Otsu.

Outputs per folder:
- analysis/aspect_ratio_over_time.csv (includes major/minor axis lengths)
- analysis/perimeter_change_kymograph.png
- analysis/perimeter_change_matrix.npy

Perimeter change:
- Baseline perimeter P0 from the first timepoint.
- For each timepoint t: delta = P(t) - P0
- Sign convention: + if perimeter increases (outwards growth), - if decreases (inwards)
- We render a heatmap over time where intensity corresponds to delta (you can rescale if needed).
"""


def find_mask_for_frame(folder: str, frame_index: int, image_name: str) -> np.ndarray:
    """Try to load a mask from MOrgAna outputs; else return None.
    Expects a subfolder named result_segmentation with mask files.
    """
    seg_dir = os.path.join(folder, "result_segmentation")
    if not os.path.isdir(seg_dir):
        return None
    # Heuristics: mask file could be named like <image>_mask_frameXXXX.tif or similar;
    # fallback: take any mask in result_segmentation for this image and frame.
    # For simplicity, if there is a single mask for the whole image, use it.
    candidates = []
    for fn in os.listdir(seg_dir):
        if not fn.lower().endswith('.tif'):
            continue
        if image_name.split('.')[0] in fn:
            candidates.append(os.path.join(seg_dir, fn))
    if not candidates:
        return None
    try:
        mask = tiff.imread(candidates[0])
        # Ensure binary
        if mask.dtype != np.bool_:
            mask = mask > 0
        return mask.astype(bool)
    except Exception:
        return None


def segment_bf(frame: np.ndarray, method: str = 'rolling_watershed') -> np.ndarray:
    """
    BF segmentation with optional high-quality method.

    Parameters
    ----------
    frame  : (Y, X) 2D brightfield image (already Z-projected)
    method : segmentation method to use when bf_segmentation module is available:
             'rolling_watershed' (default), 'clahe_maxentropy', 'active_contour',
             'multiscale', 'ensemble', or 'otsu' (original simple Otsu fallback)

    When bf_segmentation.py is present:
        Uses the selected high-quality method (comparable to Fiji/Ilastik).
    When bf_segmentation.py is absent or method='otsu':
        Falls back to the original simple Otsu threshold.
    """
    # High-quality path
    if _HAS_IMPROVED_SEG and method != 'otsu':
        try:
            return _improved_segment_bf(frame, method=method)
        except Exception as e:
            print(f"  [segment_bf] improved method failed ({e}), using Otsu fallback")

    # Original simple Otsu fallback
    img = frame.astype(np.float32)
    thresh = filters.threshold_otsu(img)
    mask = img > thresh
    mask = morphology.remove_small_objects(mask, 100)
    mask = morphology.remove_small_holes(mask, 100)
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return mask
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    return labeled == largest.label


def measure_props(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Return perimeter, aspect ratio (major/minor), major and minor axis lengths."""
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return 0.0, 0.0, 0.0, 0.0
    r = max(regions, key=lambda rr: rr.area)
    perimeter = r.perimeter
    maj = getattr(r, 'major_axis_length', 0.0) or 0.0
    minr = getattr(r, 'minor_axis_length', 0.0) or 0.0
    aspect = (maj / minr) if minr > 0 else 0.0
    return perimeter, aspect, maj, minr


def extract_contour(mask: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Extract and resample contour to fixed number of points."""
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.zeros((n_points, 2))
    # Use longest contour
    contour = max(contours, key=len)
    # Resample to n_points
    if len(contour) < 4:
        return np.zeros((n_points, 2))
    # Parametric interpolation
    distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    alpha = np.linspace(0, distance[-1], n_points)
    interp_y = interpolate.interp1d(distance, contour[:, 0], kind='linear', fill_value='extrapolate')
    interp_x = interpolate.interp1d(distance, contour[:, 1], kind='linear', fill_value='extrapolate')
    resampled = np.column_stack([interp_y(alpha), interp_x(alpha)])
    return resampled


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return (mask.shape[0] / 2.0, mask.shape[1] / 2.0)
    r = max(regions, key=lambda rr: rr.area)
    return (float(r.centroid[0]), float(r.centroid[1]))


def radii_along_angles(mask: np.ndarray, cy: float, cx: float, angles: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Compute radius for each angle by ray casting from centroid until leaving mask.
    Returns an array of radii with same length as angles.
    """
    h, w = mask.shape
    # maximum possible radius to avoid infinite loop
    max_r = np.hypot(h, w)
    radii = np.zeros_like(angles, dtype=np.float32)
    for i, th in enumerate(angles):
        r = 0.0
        last_inside = 0.0
        # march along the ray
        while r < max_r:
            y = int(round(cy + r * np.sin(th)))
            x = int(round(cx + r * np.cos(th)))
            if y < 0 or y >= h or x < 0 or x >= w:
                break
            if mask[y, x]:
                last_inside = r
                r += step
            else:
                break
        radii[i] = last_inside
    return radii


def load_bf_frame(stack: np.ndarray) -> np.ndarray:
    """Extract BF frame: if multi-channel, use the first channel; else use frame as-is."""
    # Support shapes: (T, H, W), (T, C, H, W), (H, W), (C, H, W)
    arr = stack
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Could be (T, H, W) or (C, H, W). Assume if first dim small (<=5), treat as channels.
        if arr.shape[0] <= 5:
            return arr[0]
        else:
            return arr[0]
    if arr.ndim == 4:
        # (T, C, H, W) → use C=0
        return arr[0, 0]
    return arr.squeeze()


def list_final_masks(mask_root: str, image_prefix: str = None) -> List[str]:
    """List mask files ending with finalMask in names (case-insensitive).
    Optionally filter by image_prefix contained in filename.
    """
    if not os.path.isdir(mask_root):
        return []
    exts = ('.tif', '.tiff')
    out = []
    for fn in sorted(os.listdir(mask_root)):
        lfn = fn.lower()
        if not lfn.endswith(exts):
            continue
        if 'finalmask' not in lfn:
            continue
        if image_prefix and image_prefix not in fn:
            continue
        out.append(os.path.join(mask_root, fn))
    return out


def process_folder(folder: str, save_dir: str, mask_only: bool = False,
                   image_prefix: str = None, segmentation_method: str = 'rolling_watershed'):
    os.makedirs(save_dir, exist_ok=True)
    # Fallback to local workspace if save_dir not writable
    test_file = os.path.join(save_dir, '__write_test.tmp')
    try:
        with open(test_file, 'w') as tf:
            tf.write('ok')
        os.remove(test_file)
    except Exception:
        base_local = r"C:\Users\kattimani\Project\pescoid-image-analysis\quick_test_output\NM_MK_analysis"
        safe_name = folder.replace(':', '').replace('\\', '_').replace('/', '_')
        save_dir = os.path.join(base_local, safe_name)
        os.makedirs(save_dir, exist_ok=True)
    # If mask_only: list masks in folder (e.g., result_segmentation)
    if mask_only:
        mask_files = list_final_masks(folder, image_prefix=image_prefix)
        if not mask_files:
            print(f"No finalMask files found in {folder}")
            return
    else:
        # Find tif files in folder
        tifs = [fn for fn in os.listdir(folder) if fn.lower().endswith('.tif')]
        if not tifs:
            print(f"No TIF files found in {folder}")
            return
    perimeters = []
    aspects = []
    majors = []
    minors = []
    baseline_set = False
    P0 = 0.0
    
    # For kymograph: store radius changes over time along normalized perimeter
    n_perimeter_points = 200
    radius_matrix = []  # rows: time, cols: perimeter samples
    baseline_radii = None

    if mask_only:
        iterable = mask_files
    else:
        iterable = [os.path.join(folder, name) for name in sorted(tifs)]

    for idx, path in enumerate(iterable):
        if mask_only:
            try:
                mask = tiff.imread(path)
                if mask.dtype != np.bool_:
                    mask = mask > 0
                mask = mask.astype(bool)
            except Exception as e:
                print(f"Failed to read mask {path}: {e}")
                continue
        else:
            try:
                stack = tiff.imread(path)
            except Exception as e:
                print(f"Failed to read {path}: {e}")
                continue
            bf = load_bf_frame(stack)
            mask = find_mask_for_frame(folder, idx, os.path.basename(path))
            if mask is None:
                mask = segment_bf(bf, method=segmentation_method)
        # measure properties for both branches
        perimeter, aspect, maj, minr = measure_props(mask)
        if not baseline_set:
            P0 = perimeter
            baseline_set = True
        perimeters.append(perimeter)
        aspects.append(aspect)
        majors.append(maj)
        minors.append(minr)

        # Build radius-change kymograph relative to t0 (initial = 0)
        # 1) establish angles from baseline mask only once using baseline contour points
        if baseline_radii is None:
            # baseline: use centroid and angles from baseline contour points
            cy0, cx0 = mask_centroid(mask)
            baseline_contour = extract_contour(mask, n_points=n_perimeter_points)
            by = baseline_contour[:, 0] - cy0
            bx = baseline_contour[:, 1] - cx0
            base_angles = np.arctan2(by, bx)
            # unwrap and sort by angle to produce a consistent mapping 0..2pi
            order = np.argsort(base_angles)
            base_angles = base_angles[order]
            # compute baseline radii along these angles
            baseline_radii = radii_along_angles(mask, cy0, cx0, base_angles)
            baseline_angles = base_angles

        # 2) for current mask, compute radii along baseline angles
        cy, cx = mask_centroid(mask)
        radii_t = radii_along_angles(mask, cy, cx, baseline_angles)
        delta_r = radii_t - baseline_radii
        radius_matrix.append(delta_r)

    if not perimeters:
        print(f"No measurements computed for {folder}")
        return

    # Compute delta perimeter vs baseline
    deltas = [p - P0 for p in perimeters]

    # Save aspect ratio CSV
    csv_path = os.path.join(save_dir, 'aspect_ratio_over_time.csv')
    try:
        with open(csv_path, 'w') as f:
            f.write('time_index,major_axis_length,minor_axis_length,aspect_ratio,perimeter\n')
            for i, (maj, minr, ar, per) in enumerate(zip(majors, minors, aspects, perimeters)):
                f.write(f"{i},{maj},{minr},{ar},{per}\n")
    except PermissionError:
        # Fallback to local path
        base_local = r"C:\Users\kattimani\Project\pescoid-image-analysis\quick_test_output\NM_MK_analysis"
        safe_name = folder.replace(':', '').replace('\\', '_').replace('/', '_')
        alt_dir = os.path.join(base_local, safe_name)
        os.makedirs(alt_dir, exist_ok=True)
        csv_path = os.path.join(alt_dir, 'aspect_ratio_over_time.csv')
        with open(csv_path, 'w') as f:
            f.write('time_index,major_axis_length,minor_axis_length,aspect_ratio,perimeter\n')
            for i, (maj, minr, ar, per) in enumerate(zip(majors, minors, aspects, perimeters)):
                f.write(f"{i},{maj},{minr},{ar},{per}\n")
        save_dir = alt_dir

    # Save deltas matrix (1D vector for now)
    mat_path = os.path.join(save_dir, 'perimeter_change_matrix.npy')
    try:
        np.save(mat_path, np.array(deltas, dtype=np.float32))
    except PermissionError:
        mat_path = os.path.join(save_dir, 'perimeter_change_matrix_local.npy')
        np.save(mat_path, np.array(deltas, dtype=np.float32))

    # Build 2D kymograph: normalized perimeter position (Y) vs time (X) with Δ radius (normalized) as color
    if radius_matrix:
        kymo_2d = np.array(radius_matrix).T  # shape: (n_perimeter_points, n_timepoints)
        max_abs_dev = float(np.max(np.abs(kymo_2d))) if kymo_2d.size else 0.0
        if max_abs_dev > 0:
            kymo_2d = kymo_2d / max_abs_dev
        kymo_2d = np.clip(kymo_2d, -1.0, 1.0)
        vmin, vmax = -1.0, 1.0
    else:
        kymo_2d = np.zeros((n_perimeter_points, len(deltas)), dtype=np.float32)
        vmin, vmax = -1.0, 1.0

    # Plot 2D kymograph
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(kymo_2d, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xlabel('Time [index]', fontsize=12)
    ax.set_ylabel('Normalized Perimeter', fontsize=12)
    # Set Y-axis ticks to show normalized perimeter (0 to 1)
    n_ticks = 5
    tick_indices = np.linspace(0, n_perimeter_points - 1, n_ticks, dtype=int)
    tick_labels = np.linspace(0, 1, n_ticks)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels([f'{val:.1f}' for val in tick_labels])
    ax.set_title('Normalized radius change (initial=0, red=outward, blue=inward)', fontsize=11)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Δ radius (normalized)', fontsize=10)
    out_png = os.path.join(save_dir, 'perimeter_change_kymograph.png')
    plt.tight_layout()
    try:
        plt.savefig(out_png, dpi=200)
    except PermissionError:
        out_png = os.path.join(save_dir, 'perimeter_change_kymograph_local.png')
        plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # =====================================================================
    # POLE DETECTION from kymograph
    # =====================================================================
    pole_results = None
    if kymo_2d.size > 0:
        try:
            pole_results = detect_and_analyze_poles(
                kymo_2d,
                smooth_sigma=2,
                prominence=0.15,
                min_distance=20,
                classify=True
            )
            
            # Save pole detection visualization
            pole_vis_path = os.path.join(save_dir, 'pole_detection.png')
            try:
                if pole_results['n_poles'] > 0:
                    visualize_pole_detection(
                        pole_results['activity_profile'],
                        pole_results['pole_indices'],
                        outward_poles=pole_results.get('outward_indices'),
                        inward_poles=pole_results.get('inward_indices'),
                        save_path=pole_vis_path
                    )
            except PermissionError:
                pole_vis_path = os.path.join(save_dir, 'pole_detection_local.png')
                if pole_results['n_poles'] > 0:
                    visualize_pole_detection(
                        pole_results['activity_profile'],
                        pole_results['pole_indices'],
                        outward_poles=pole_results.get('outward_indices'),
                        inward_poles=pole_results.get('inward_indices'),
                        save_path=pole_vis_path
                    )
            
            # Save pole count CSV
            pole_csv_path = os.path.join(save_dir, 'pole_counts.csv')
            try:
                with open(pole_csv_path, 'w') as f:
                    f.write('metric,value\n')
                    f.write(f"total_poles,{pole_results['n_poles']}\n")
                    f.write(f"outward_poles,{pole_results.get('n_outward', 0)}\n")
                    f.write(f"inward_poles,{pole_results.get('n_inward', 0)}\n")
            except PermissionError:
                pole_csv_path = os.path.join(save_dir, 'pole_counts_local.csv')
                with open(pole_csv_path, 'w') as f:
                    f.write('metric,value\n')
                    f.write(f"total_poles,{pole_results['n_poles']}\n")
                    f.write(f"outward_poles,{pole_results.get('n_outward', 0)}\n")
                    f.write(f"inward_poles,{pole_results.get('n_inward', 0)}\n")
            
            print(f"  Detected {pole_results['n_poles']} poles: {pole_results.get('n_outward', 0)} outward, {pole_results.get('n_inward', 0)} inward")
            
        except Exception as e:
            print(f"  Warning: Pole detection failed: {e}")
            pole_results = None

    # Save summary JSON
    summary = {
        'folder': folder,
        'baseline_perimeter': P0,
        'perimeters': perimeters,
        'deltas': deltas,
        'aspects': aspects,
        'major_axis_lengths': majors,
        'minor_axis_lengths': minors,
        'kymograph_shape': kymo_2d.shape if kymo_2d.size else (0, 0),
        'pole_detection': {
            'n_poles': pole_results['n_poles'] if pole_results else 0,
            'n_outward': pole_results.get('n_outward', 0) if pole_results else 0,
            'n_inward': pole_results.get('n_inward', 0) if pole_results else 0,
            'pole_indices': pole_results['pole_indices'].tolist() if pole_results and pole_results['n_poles'] > 0 else [],
        } if pole_results else {},
        'outputs': {
            'aspect_csv': csv_path,
            'delta_matrix_npy': mat_path,
            'kymograph_png': out_png,
            'pole_detection_png': pole_vis_path if pole_results and pole_results['n_poles'] > 0 else None,
            'pole_counts_csv': pole_csv_path if pole_results else None,
        }
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as jf:
        json.dump(summary, jf, indent=2)
    # Use ASCII arrow for Windows console compatibility
    print(f"Done: {folder} -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compute aspect ratio and perimeter-change kymograph')
    parser.add_argument('root', help='Folder containing time-lapse TIFs or masks (e.g., result_segmentation)')
    parser.add_argument('--out', default=None, help='Output analysis folder (default: <root>/analysis)')
    parser.add_argument('--mask-only', action='store_true', help='Analyze only mask files ending with finalMask')
    parser.add_argument('--image-prefix', default=None, help='Optional filter: only masks containing this substring in filename')
    parser.add_argument(
        '--segmentation-method', default='rolling_watershed',
        choices=['rolling_watershed', 'clahe_maxentropy', 'active_contour',
                 'multiscale', 'ensemble', 'otsu'],
        help=(
            'BF segmentation method used when no pre-computed masks exist '
            '(default: rolling_watershed). '
            'rolling_watershed is comparable to Fiji; otsu = original simple fallback. '
            'Requires analysis/bf_segmentation.py.'
        ),
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)
    out = args.out or os.path.join(root, 'analysis')
    process_folder(root, out, mask_only=args.mask_only, image_prefix=args.image_prefix,
                   segmentation_method=args.segmentation_method)


if __name__ == '__main__':
    main()
