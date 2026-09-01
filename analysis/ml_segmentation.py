"""
ML-Based Segmentation for Pescoid Brightfield Images
=====================================================
Three backends, from zero-annotation to fully supervised:

  Backend 1: 'cellpose'   -- Deep learning (bio-specific, built-in brightfield model)
                             No training needed. Best first thing to try.
                             Install: pip install cellpose

  Backend 2: 'sam'        -- Segment Anything Model (Meta, zero-shot)
                             No training needed. Excellent precise boundaries.
                             Install: pip install segment-anything
                             Weights: download ViT-H SAM checkpoint (~375 MB)
                             OR:  pip install mobile-sam  (lightweight, ~40 MB)

  Backend 3: 'pixel_rf'   -- Ilastik-style Random Forest on pixel features
                             TRAINABLE: annotate 2-3 tricky frames in Fiji,
                             train once, apply to the whole batch.
                             No extra packages needed (uses scikit-learn).

Annotation workflow for pixel_rf
---------------------------------
  1. Open raw TIFF in Fiji (one frame at a time).
  2. Threshold roughly → binary mask → manually fix boundary.
  3. Save as TIFF (0 = background, 255 = foreground).
  4. Name each file:  <anything>_annotation.tif
  5. Put them all in one folder  (e.g.  annotations/ ).
  6. Train:
       python scripts/train_pixel_classifier.py annotations/ --model model.pkl
  7. Segment batch using trained model:
       python analysis/ml_segmentation.py Z:/path/to/TIFF --out output/ \\
           --backend pixel_rf --model model.pkl --batch

CLI examples
------------
  # Cellpose (zero-shot, try first)
  python analysis/ml_segmentation.py data/file.tif --out output/ --backend cellpose

  # SAM with MobileSAM weights
  python analysis/ml_segmentation.py data/file.tif --out output/ \\
      --backend sam --sam-checkpoint mobile_sam.pt

  # Pixel RF with trained model
  python analysis/ml_segmentation.py data/file.tif --out output/ \\
      --backend pixel_rf --model annotations/model.pkl

  # Batch
  python analysis/ml_segmentation.py data/ --batch --out output/ --backend cellpose
"""

import warnings
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure
from skimage.util import img_as_float, img_as_ubyte


# ============================================================================
# SHARED UTILITIES  (reuse from bf_segmentation if available, else inline)
# ============================================================================

def _normalize(img, p_low=0.5, p_high=99.5):
    f = img_as_float(img.astype(np.float64))
    lo, hi = np.percentile(f, p_low), np.percentile(f, p_high)
    if hi > lo:
        f = np.clip((f - lo) / (hi - lo), 0, 1)
    return f


def _load_tiff(path):
    """Load + normalize to (T, Z, C, Y, X)."""
    try:
        from bf_segmentation import load_tiff
        return load_tiff(path)
    except ImportError:
        data = tifffile.imread(str(path))
        if data.ndim == 5:
            stack = data
        elif data.ndim == 4:
            stack = data[:, :, np.newaxis, :, :]
        elif data.ndim == 3:
            stack = data[np.newaxis, :, np.newaxis, :, :]
        else:
            stack = data[np.newaxis, np.newaxis, np.newaxis, :, :]
        meta = {k: v for k, v in zip(
            ['T', 'Z', 'C', 'Y', 'X'], stack.shape)}
        meta.update({'path': str(path), 'dtype': str(data.dtype),
                     'original_shape': data.shape, 'original_ndim': data.ndim})
        return stack, meta


def _z_project(z_stack, method='best_focus', z_range=None):
    try:
        from bf_segmentation import z_project
        return z_project(z_stack, method=method, z_range=z_range)
    except ImportError:
        return np.max(z_stack, axis=0).astype(float)


def _postprocess(mask, min_area=50000, closing_radius=5):
    if not mask.any():
        return mask.astype(bool)
    mask = ndi.binary_fill_holes(mask)
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, footprint=morphology.disk(closing_radius))
        mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=max(100, min_area // 20))
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return mask.astype(bool)
    largest = max(measure.regionprops(labeled), key=lambda r: r.area)
    if largest.area < min_area:
        warnings.warn(f"Largest region ({largest.area:,} px) < min_area ({min_area:,} px).")
    return (labeled == largest.label).astype(bool)


def _detect_organoid_center(bf_2d):
    """
    Coarsely detect the organoid center for use as a SAM prompt.

    Pescoid = dark on bright background.
    Strategy: find the largest dark region centroid.
    """
    img = _normalize(bf_2d)
    inv = 1.0 - img
    blurred = filters.gaussian(inv, sigma=10)
    thresh = filters.threshold_otsu(blurred)
    coarse = blurred > thresh
    coarse = ndi.binary_fill_holes(coarse)
    coarse = morphology.remove_small_objects(coarse, min_size=500)
    labeled = measure.label(coarse)
    if labeled.max() == 0:
        H, W = bf_2d.shape
        return (H // 2, W // 2)
    largest = max(measure.regionprops(labeled), key=lambda r: r.area)
    cy, cx = int(largest.centroid[0]), int(largest.centroid[1])
    return (cy, cx)


# ============================================================================
# BACKEND 1: CELLPOSE
# ============================================================================

def segment_cellpose(bf_2d, model_type='cyto3', diameter=None, channels=None,
                     min_area=50000, closing_radius=5, gpu=False):
    """
    Segment using Cellpose deep learning model.

    Cellpose is designed for biological images. The 'cyto3' model works well
    for brightfield organoids.

    Parameters
    ----------
    bf_2d      : (Y, X) raw BF image
    model_type : Cellpose model type
                 'cyto3'       -- best general-purpose (recommended)
                 'brightfield' -- explicit brightfield model (if available)
                 'cyto2'       -- older cytoplasm model
    diameter   : expected organoid diameter in pixels (None = auto-estimate)
    channels   : [cytoplasm_ch, nucleus_ch] -- use [0,0] for grayscale
    gpu        : use GPU if available

    Returns
    -------
    mask : (Y, X) bool array
    """
    try:
        from cellpose import models
    except ImportError:
        raise ImportError(
            "Cellpose is not installed.\n"
            "Install with:  pip install cellpose\n"
            "Or with GPU:   pip install cellpose[gui]"
        )

    if channels is None:
        channels = [0, 0]  # grayscale

    # Normalize to uint8
    img_8bit = img_as_ubyte(_normalize(bf_2d))

    # Cellpose v4 changed the API: model_type / channels are deprecated
    import cellpose
    cp_version = getattr(cellpose, '__version__', '0')
    major = int(cp_version.split('.')[0]) if cp_version[0].isdigit() else 0

    if major >= 4:
        # v4+: use model_name kwarg (or default to built-in model)
        try:
            model = models.CellposeModel(gpu=gpu, model_name=model_type)
        except TypeError:
            model = models.CellposeModel(gpu=gpu)
        masks_cp, _, _ = model.eval([img_8bit], diameter=diameter, do_3D=False)
    else:
        model = models.CellposeModel(gpu=gpu, model_type=model_type)
        masks_cp, _, _ = model.eval(
            [img_8bit], diameter=diameter, channels=channels,
            do_3D=False, batch_size=1,
        )
    raw_mask = masks_cp[0]

    if raw_mask.max() == 0:
        warnings.warn("Cellpose returned empty mask.")
        return np.zeros(bf_2d.shape[:2], dtype=bool)

    # For single-organoid images, keep the largest detected object
    regions = measure.regionprops(raw_mask)
    largest = max(regions, key=lambda r: r.area)
    mask = (raw_mask == largest.label).astype(bool)

    return _postprocess(mask, min_area=min_area, closing_radius=closing_radius)


# ============================================================================
# BACKEND 2: SAM (Segment Anything Model)
# ============================================================================

def segment_sam(bf_2d, checkpoint, model_type='auto',
                center_point=None, extra_neg_points=True,
                min_area=50000, closing_radius=3):
    """
    Segment using Meta's Segment Anything Model (SAM).

    Extremely accurate boundaries, zero-shot (no training needed).
    Provide a positive point near the organoid center; SAM finds the boundary.

    Parameters
    ----------
    bf_2d          : (Y, X) raw BF image
    checkpoint     : path to SAM or MobileSAM weights file
                     SAM ViT-H  : download from Meta (~375 MB)
                     MobileSAM  : pip install mobile-sam + download weights (~40 MB)
    model_type     : 'auto' (detect from checkpoint name), 'vit_h', 'vit_l',
                     'vit_b', or 'mobile_sam'
    center_point   : (row, col) of organoid center; None = auto-detect
    extra_neg_points : add corner pixels as negative (background) prompts
    min_area       : minimum area
    closing_radius : closing after SAM

    Returns
    -------
    mask : (Y, X) bool array

    Install
    -------
    # Full SAM (best quality):
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Download weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    # MobileSAM (lightweight, ~40MB, fast):
    pip install mobile-sam
    # Download: https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    H, W = bf_2d.shape

    # Auto-detect model type from filename
    if model_type == 'auto':
        name = checkpoint.name.lower()
        if 'mobile' in name:
            model_type = 'mobile_sam'
        elif 'vit_h' in name or '_h_' in name:
            model_type = 'vit_h'
        elif 'vit_l' in name or '_l_' in name:
            model_type = 'vit_l'
        elif 'vit_b' in name or '_b_' in name:
            model_type = 'vit_b'
        else:
            model_type = 'vit_h'

    # Load model
    if model_type == 'mobile_sam':
        try:
            from mobile_sam import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "MobileSAM not installed.\n"
                "Install with:  pip install mobile-sam"
            )
        sam = sam_model_registry['vit_t'](checkpoint=str(checkpoint))
    else:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed.\n"
                "Install with:  "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))

    sam.eval()

    # Auto-detect organoid center if not provided
    if center_point is None:
        cy, cx = _detect_organoid_center(bf_2d)
    else:
        cy, cx = int(center_point[0]), int(center_point[1])

    # Build point prompts
    point_coords = [[cx, cy]]  # SAM uses (x, y) order
    point_labels = [1]         # 1 = foreground

    if extra_neg_points:
        # Add corners as definite background
        margin = 10
        for ry, rx in [(margin, margin), (margin, W-margin),
                        (H-margin, margin), (H-margin, W-margin)]:
            point_coords.append([rx, ry])
            point_labels.append(0)  # 0 = background

    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)

    # Convert BF to uint8 RGB (SAM expects 3-channel)
    img_norm = _normalize(bf_2d)
    img_8bit = img_as_ubyte(img_norm)
    img_rgb = np.stack([img_8bit, img_8bit, img_8bit], axis=-1)

    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    masks_sam, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Pick the mask with highest score that is also the most reasonable size
    # (filter out masks that are too small or fill the whole frame)
    total_px = H * W
    valid = []
    for mask_s, score in zip(masks_sam, scores):
        area = mask_s.sum()
        if min_area // 4 < area < total_px * 0.8:
            valid.append((score, mask_s))

    if not valid:
        # Fallback: just take highest-scoring mask
        best_idx = int(np.argmax(scores))
        mask = masks_sam[best_idx]
    else:
        _, mask = max(valid, key=lambda x: x[0])

    return _postprocess(mask.astype(bool), min_area=min_area,
                        closing_radius=closing_radius)


# ============================================================================
# BACKEND 3: PIXEL RANDOM FOREST (Ilastik-style, trainable)
# ============================================================================

def _extract_features(img_float, sigmas=(1, 2, 4, 8)):
    """
    Extract multi-scale pixel features (same as Ilastik's feature set).

    Features per pixel:
      - Gaussian smoothed intensity         (n_scales)
      - Gradient magnitude (Sobel of Gauss) (n_scales)
      - Laplacian of Gaussian               (n_scales)
      - Hessian eigenvalue 1 (min curv)     (n_scales)
      - Hessian eigenvalue 2 (max curv)     (n_scales)
      - Local mean                          (n_scales)
      - Local standard deviation            (n_scales)

    Returns
    -------
    features : (Y, X, n_features) float array
    """
    feats = [img_float]  # raw intensity

    for sigma in sigmas:
        # Gaussian
        gauss = filters.gaussian(img_float, sigma=sigma)
        feats.append(gauss)

        # Gradient magnitude
        grad = filters.sobel(gauss)
        feats.append(grad)

        # Laplacian of Gaussian
        log = ndi.gaussian_laplace(img_float, sigma=sigma)
        feats.append(log)

        # Hessian eigenvalues (measure of curvature — great for dark ring)
        Hxx = ndi.gaussian_filter(img_float, sigma=sigma, order=(2, 0))
        Hyy = ndi.gaussian_filter(img_float, sigma=sigma, order=(0, 2))
        Hxy = ndi.gaussian_filter(img_float, sigma=sigma, order=(1, 1))
        # Eigenvalues of 2x2 symmetric Hessian per pixel
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        disc = np.sqrt(np.maximum(0, (trace / 2) ** 2 - det))
        lam1 = trace / 2 - disc
        lam2 = trace / 2 + disc
        feats.extend([lam1, lam2])

        # Local stats (patch context)
        size = max(3, int(sigma * 2 + 1))
        local_mean = ndi.uniform_filter(img_float, size=size)
        local_sq   = ndi.uniform_filter(img_float ** 2, size=size)
        local_std  = np.sqrt(np.maximum(0, local_sq - local_mean ** 2))
        feats.extend([local_mean, local_std])

    return np.stack(feats, axis=-1)  # (Y, X, n_features)


class PixelClassifier:
    """
    Ilastik-style Random Forest pixel classifier.

    Train on a handful of annotated frames, then apply to the whole batch.

    Annotation format (Fiji workflow):
      1. Open a frame in Fiji.
      2. Use paintbrush / polygon / freehand to outline the organoid.
      3. Fill the interior → binary mask (255 = organoid, 0 = background).
      4. Save as TIFF (File → Save As → Tiff).
      5. Name it  <anything>_annotation.tif  and put in one folder.

    Example
    -------
    clf = PixelClassifier()
    # images : list of 2D float arrays
    # masks  : list of 2D bool arrays (True = organoid)
    clf.train(images, masks)
    clf.save('model.pkl')

    # Later:
    clf2 = PixelClassifier.load('model.pkl')
    mask = clf2.predict(bf_2d)
    """

    def __init__(self, n_estimators=200, sigmas=(1, 2, 4, 8),
                 max_samples_per_image=50000):
        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,          # use all CPU cores
            random_state=42,
            class_weight='balanced',  # handles foreground/background imbalance
        )
        self.sigmas = sigmas
        self.max_samples_per_image = max_samples_per_image
        self._trained = False

    def train(self, images, masks, verbose=True):
        """
        Train the classifier on a list of (image, mask) pairs.

        Parameters
        ----------
        images : list of (Y, X) float or raw arrays  (BF frames)
        masks  : list of (Y, X) bool/uint8 arrays    (True/255 = organoid)
        verbose: print progress
        """
        if verbose:
            print(f"Extracting features from {len(images)} annotated frame(s)...")

        X_all, y_all = [], []
        for i, (img, mask) in enumerate(zip(images, masks)):
            img_f = _normalize(img)
            mask_b = (mask > 0).astype(np.uint8)

            features = _extract_features(img_f, sigmas=self.sigmas)
            H, W, nf = features.shape

            # Flatten
            X = features.reshape(-1, nf)   # (H*W, nf)
            y = mask_b.ravel()             # (H*W,)

            # Subsample for speed (keep class balance)
            fg_idx = np.where(y == 1)[0]
            bg_idx = np.where(y == 0)[0]

            n_each = min(self.max_samples_per_image // 2,
                         len(fg_idx), len(bg_idx))
            if n_each == 0:
                warnings.warn(f"Frame {i}: mask has no annotated pixels, skipping.")
                continue

            rng = np.random.default_rng(42)
            fg_sel = rng.choice(fg_idx, n_each, replace=False)
            bg_sel = rng.choice(bg_idx, n_each, replace=False)
            sel = np.concatenate([fg_sel, bg_sel])

            X_all.append(X[sel])
            y_all.append(y[sel])

            if verbose:
                print(f"  Frame {i+1}: fg={len(fg_idx):,} bg={len(bg_idx):,} "
                      f"sampled={2*n_each:,} pixels")

        if not X_all:
            raise ValueError("No valid annotated frames found.")

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)

        if verbose:
            print(f"Training Random Forest on {len(X_train):,} pixels "
                  f"({(y_train==1).sum():,} fg, {(y_train==0).sum():,} bg)...")

        self.rf.fit(X_train, y_train)
        self._trained = True

        if verbose:
            from sklearn.metrics import accuracy_score
            y_pred = self.rf.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            print(f"Training accuracy: {acc:.4f}  "
                  f"(OOB if enough data: use oob_score=True)")

    def predict(self, bf_2d, min_area=50000, closing_radius=5):
        """
        Predict organoid mask for a single BF frame.

        Parameters
        ----------
        bf_2d : (Y, X) raw BF image

        Returns
        -------
        mask : (Y, X) bool
        """
        if not self._trained:
            raise RuntimeError("PixelClassifier: call .train() before .predict()")

        img_f = _normalize(bf_2d)
        features = _extract_features(img_f, sigmas=self.sigmas)
        H, W, nf = features.shape

        proba = self.rf.predict_proba(features.reshape(-1, nf))
        # fg probability map
        fg_idx = list(self.rf.classes_).index(1)
        prob_map = proba[:, fg_idx].reshape(H, W)

        mask = prob_map > 0.5
        return _postprocess(mask, min_area=min_area, closing_radius=closing_radius)

    def predict_proba_map(self, bf_2d):
        """Return the raw [0,1] probability map (useful for threshold tuning)."""
        if not self._trained:
            raise RuntimeError("PixelClassifier: call .train() first.")
        img_f = _normalize(bf_2d)
        features = _extract_features(img_f, sigmas=self.sigmas)
        H, W, nf = features.shape
        proba = self.rf.predict_proba(features.reshape(-1, nf))
        fg_idx = list(self.rf.classes_).index(1)
        return proba[:, fg_idx].reshape(H, W)

    def save(self, path):
        """Save trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), 'wb') as f:
            pickle.dump({'rf': self.rf, 'sigmas': self.sigmas,
                         'trained': self._trained}, f)
        print(f"Saved model: {path}")

    @classmethod
    def load(cls, path):
        """Load a previously saved model."""
        with open(str(path), 'rb') as f:
            data = pickle.load(f)
        obj = cls(sigmas=data['sigmas'])
        obj.rf = data['rf']
        obj._trained = data['trained']
        print(f"Loaded model: {path}  (trained={obj._trained})")
        return obj


def segment_pixel_rf(bf_2d, classifier, min_area=50000, closing_radius=5):
    """
    Segment using a trained PixelClassifier.

    Parameters
    ----------
    bf_2d      : (Y, X) raw BF frame
    classifier : trained PixelClassifier instance or path to saved .pkl

    Returns
    -------
    mask : (Y, X) bool
    """
    if isinstance(classifier, (str, Path)):
        classifier = PixelClassifier.load(str(classifier))
    return classifier.predict(bf_2d, min_area=min_area,
                               closing_radius=closing_radius)


# ============================================================================
# TRAINING FROM ANNOTATION FOLDER
# ============================================================================

def train_from_annotations(annotation_dir, tiff_dir=None, bf_channel=1,
                             z_projection='best_focus', z_range=None,
                             n_estimators=200, output_model='model.pkl',
                             verbose=True):
    """
    Train a PixelClassifier from a folder of Fiji-annotated mask TIFFs.

    Annotation naming convention (flexible):
      Case A: paired raw+mask in same folder
         raw:        myfile_t0005.tif   (any TIF without 'annotation' in name)
         annotation: myfile_t0005_annotation.tif

      Case B: annotation-only folder, with matching raw TIFFs elsewhere
         annotations/  → <stem>_annotation.tif files
         tiff_dir/     → <stem>.tif  (raw BF files)

      Case C: single-frame TIFs where the annotation mask covers the whole frame
         annotations/  → myfile_annotation.tif  (one annotation per file)
         Raw image: loaded from tiff_dir

    Parameters
    ----------
    annotation_dir  : folder containing *_annotation.tif files
    tiff_dir        : folder with raw TIF files (if different from annotation_dir)
    bf_channel      : 0-indexed BF channel (default 1)
    z_projection    : Z-projection method
    z_range         : z slice range
    n_estimators    : number of RF trees
    output_model    : where to save the trained model (.pkl)
    verbose         : print progress

    Returns
    -------
    clf : trained PixelClassifier
    """
    annotation_dir = Path(annotation_dir)
    ann_files = sorted(annotation_dir.glob('*_annotation.tif'))
    ann_files += sorted(annotation_dir.glob('*_annotation.tiff'))
    ann_files = sorted(set(ann_files))

    if not ann_files:
        raise FileNotFoundError(
            f"No *_annotation.tif files found in {annotation_dir}\n"
            "Annotation files must end with '_annotation.tif'"
        )

    if verbose:
        print(f"Found {len(ann_files)} annotation file(s):")
        for f in ann_files:
            print(f"  {f.name}")

    images, masks = [], []

    for ann_path in ann_files:
        # Load annotation mask
        ann_mask = tifffile.imread(str(ann_path))
        if ann_mask.ndim > 2:
            ann_mask = ann_mask[0] if ann_mask.ndim == 3 else ann_mask[0, 0]
        ann_mask = (ann_mask > 0).astype(np.uint8)

        # Find matching raw TIFF
        # Stem without _annotation suffix
        stem = ann_path.stem.replace('_annotation', '')
        raw_path = None

        # Search order: annotation_dir → tiff_dir
        search_dirs = [annotation_dir]
        if tiff_dir:
            search_dirs.append(Path(tiff_dir))

        for d in search_dirs:
            for ext in ('.tif', '.tiff'):
                candidate = d / (stem + ext)
                if candidate.exists() and 'annotation' not in candidate.stem.lower():
                    raw_path = candidate
                    break
            if raw_path:
                break

        if raw_path is None:
            warnings.warn(
                f"No matching raw TIF found for {ann_path.name} "
                f"(looked for '{stem}.tif'). "
                "Treating the annotation mask's own image as the BF image. "
                "For best results, provide matching raw TIF files."
            )
            # Use the annotation as both image and mask (fallback)
            bf_2d = ann_mask.astype(float)
        else:
            if verbose:
                print(f"  {ann_path.name}  ←→  {raw_path.name}")
            stack, meta = _load_tiff(str(raw_path))
            T, Z, C = stack.shape[:3]
            ch = bf_channel if bf_channel < C else 0
            # Use first timepoint (or all if single frame)
            bf_z = stack[0, :, ch, :, :]
            bf_2d = _z_project(bf_z, method=z_projection, z_range=z_range)

        # Check shape match
        if bf_2d.shape != ann_mask.shape:
            from skimage.transform import resize
            ann_mask = resize(ann_mask, bf_2d.shape, order=0,
                              anti_aliasing=False).astype(np.uint8)
            warnings.warn(
                f"Resized annotation {ann_path.name} from "
                f"{ann_mask.shape} to {bf_2d.shape}"
            )

        images.append(bf_2d)
        masks.append(ann_mask)

    if not images:
        raise ValueError("No valid image/annotation pairs found.")

    # Train
    clf = PixelClassifier(n_estimators=n_estimators)
    clf.train(images, masks, verbose=verbose)
    clf.save(output_model)

    return clf


# ============================================================================
# UNIFIED ML SEGMENTER
# ============================================================================

class MLSegmenter:
    """
    Unified ML-based BF segmenter.

    Usage
    -----
    # Cellpose (no training needed):
    seg = MLSegmenter(backend='cellpose')

    # SAM (no training needed):
    seg = MLSegmenter(backend='sam', sam_checkpoint='mobile_sam.pt')

    # Pixel RF (trained model):
    seg = MLSegmenter(backend='pixel_rf', model_path='model.pkl')

    masks, meta = seg.segment_stack('file.tif', output_dir='output/')
    """

    def __init__(
        self,
        backend='cellpose',
        bf_channel=1,
        z_projection='best_focus',
        z_range=None,
        min_area=50000,
        closing_radius=5,
        # Cellpose options
        cellpose_model='cyto3',
        cellpose_diameter=None,
        cellpose_gpu=False,
        # SAM options
        sam_checkpoint=None,
        sam_model_type='auto',
        # Pixel RF options
        model_path=None,
        verbose=True,
    ):
        self.backend = backend
        self.bf_channel = bf_channel
        self.z_projection = z_projection
        self.z_range = z_range
        self.min_area = min_area
        self.closing_radius = closing_radius
        self.cellpose_model = cellpose_model
        self.cellpose_diameter = cellpose_diameter
        self.cellpose_gpu = cellpose_gpu
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.model_path = model_path
        self.verbose = verbose

        # Pre-load pixel RF model if provided
        self._pixel_clf = None
        if backend == 'pixel_rf' and model_path:
            self._pixel_clf = PixelClassifier.load(model_path)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def segment_frame(self, bf_2d):
        """Segment a single 2D BF frame."""
        if self.backend == 'cellpose':
            return segment_cellpose(
                bf_2d,
                model_type=self.cellpose_model,
                diameter=self.cellpose_diameter,
                gpu=self.cellpose_gpu,
                min_area=self.min_area,
                closing_radius=self.closing_radius,
            )
        elif self.backend == 'sam':
            if not self.sam_checkpoint:
                raise ValueError("SAM requires --sam-checkpoint path to weights file.")
            return segment_sam(
                bf_2d,
                checkpoint=self.sam_checkpoint,
                model_type=self.sam_model_type,
                min_area=self.min_area,
                closing_radius=self.closing_radius,
            )
        elif self.backend == 'pixel_rf':
            if self._pixel_clf is None:
                raise ValueError(
                    "pixel_rf backend requires a trained model. "
                    "Run:  python scripts/train_pixel_classifier.py  first, "
                    "then pass --model model.pkl"
                )
            return self._pixel_clf.predict(
                bf_2d,
                min_area=self.min_area,
                closing_radius=self.closing_radius,
            )
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. "
                             "Choose: cellpose, sam, pixel_rf")

    def segment_stack(self, tiff_path, output_dir=None):
        """
        Segment all timepoints. Saves outputs. Returns (masks, meta).
        API matches BFSegmenter.segment_stack() for drop-in compatibility.
        """
        tiff_path = Path(tiff_path)
        self._log(f"Loading: {tiff_path.name}")

        stack, file_meta = _load_tiff(str(tiff_path))
        T, Z, C, Y, X = stack.shape
        self._log(f"  T={T}, Z={Z}, C={C}, Y={Y}, X={X}  (backend='{self.backend}')")

        bf_ch = self.bf_channel if self.bf_channel < C else 0
        self._log(f"  BF channel: {bf_ch}")

        masks = np.zeros((T, Y, X), dtype=bool)
        failed = []

        for t in range(T):
            bf_z = stack[t, :, bf_ch, :, :]
            bf_2d = (bf_z[0].astype(float) if Z == 1
                     else _z_project(bf_z, method=self.z_projection,
                                     z_range=self.z_range))
            try:
                mask = self.segment_frame(bf_2d)
            except Exception as e:
                warnings.warn(f"Frame {t}: segmentation failed ({e})")
                mask = np.zeros((Y, X), dtype=bool)

            if mask.sum() == 0:
                failed.append(t)
                self._log(f"  [t={t:03d}] WARNING: empty mask")
            else:
                self._log(f"  [t={t:03d}] area = {mask.sum():,} px")
            masks[t] = mask

        # Temporal consistency: fill failed frames from nearest good frame
        if failed:
            good = [t for t in range(T) if t not in failed]
            if good:
                for t in failed:
                    nearest = min(good, key=lambda g: abs(g - t))
                    masks[t] = masks[nearest].copy()
                    self._log(f"  [t={t:03d}] replaced with t={nearest}")

        # Save
        if output_dir is None:
            output_dir = tiff_path.parent / tiff_path.stem / 'result_segmentation'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save(masks, stack, bf_ch, file_meta, tiff_path, output_dir)

        meta = {
            **file_meta,
            'backend': self.backend,
            'bf_channel': bf_ch,
            'z_projection': self.z_projection,
            'failed_frames': failed,
            'mask_areas': [int(masks[t].sum()) for t in range(T)],
            'output_dir': str(output_dir),
        }
        return masks, meta

    def _save(self, masks, stack, bf_ch, file_meta, tiff_path, output_dir):
        T = masks.shape[0]
        name = tiff_path.stem

        # Mask stack (pipeline-compatible)
        tifffile.imwrite(
            str(output_dir / f"{name}_finalMask.tif"),
            (masks.astype(np.uint8) * 255)
        )
        self._log(f"  Saved: {name}_finalMask.tif")

        # Per-frame single masks
        single_dir = output_dir / 'single_frames'
        single_dir.mkdir(exist_ok=True)
        for t in range(T):
            tifffile.imwrite(
                str(single_dir / f"{name}_t{t:04d}_finalMask.tif"),
                (masks[t].astype(np.uint8) * 255)
            )

        # Overlays
        overlay_dir = output_dir / 'overlays'
        overlay_dir.mkdir(exist_ok=True)
        self._log(f"  Saving {T} overlay images...")
        for t in range(T):
            bf_z = stack[t, :, bf_ch, :, :]
            bf_2d = (bf_z[0].astype(float) if file_meta['Z'] == 1
                     else _z_project(bf_z, method=self.z_projection,
                                     z_range=self.z_range))
            bf_disp = exposure.rescale_intensity(bf_2d, out_range=(0, 1))

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(bf_disp, cmap='gray')
            axes[0].set_title(f'BF  t={t}', fontsize=9)
            axes[0].axis('off')
            axes[1].imshow(bf_disp, cmap='gray')
            for c in measure.find_contours(masks[t].astype(float), 0.5):
                axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=1.5)
            axes[1].set_title(
                f'Segmentation  area={masks[t].sum():,} px', fontsize=9)
            axes[1].axis('off')
            plt.suptitle(f'{name}  |  backend={self.backend}',
                         fontsize=10, y=1.01)
            plt.tight_layout()
            plt.savefig(str(overlay_dir / f"{name}_t{t:04d}_overlay.png"),
                        dpi=100, bbox_inches='tight')
            plt.close(fig)

        # Summary JSON
        summary = {
            'input_file': str(tiff_path),
            'backend': self.backend,
            'bf_channel': bf_ch,
            'file_info': {k: file_meta[k]
                          for k in ('T', 'Z', 'C', 'Y', 'X', 'dtype')},
            'results': {
                'mask_areas': [int(masks[t].sum()) for t in range(T)],
                'mean_area': float(np.mean([masks[t].sum() for t in range(T)])),
                'empty_frames': [int(t) for t in range(T) if masks[t].sum() == 0],
            },
        }
        with open(str(output_dir / 'segmentation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ML-based BF segmentation for pescoid TIFFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends:
  cellpose   Deep learning (bio-specific). No training needed.
             Install: pip install cellpose
  sam        Segment Anything Model (Meta). No training needed.
             Install: pip install mobile-sam
             Weights: download mobile_sam.pt (~40MB)
  pixel_rf   Ilastik-style Random Forest. Trainable from your annotations.
             No extra packages needed.

Examples:
  # Cellpose (try this first)
  python analysis/ml_segmentation.py data/file.tif --out output/ --backend cellpose

  # SAM with MobileSAM weights
  python analysis/ml_segmentation.py data/file.tif --out output/ \\
      --backend sam --sam-checkpoint mobile_sam.pt

  # Train pixel classifier from annotations, then segment
  python scripts/train_pixel_classifier.py annotations/ --tiff-dir data/ --model model.pkl
  python analysis/ml_segmentation.py data/ --batch --out output/ \\
      --backend pixel_rf --model model.pkl

  # Batch with Cellpose
  python analysis/ml_segmentation.py "Z:/path/to/TIFF" --batch --out output/ --backend cellpose
""",
    )

    parser.add_argument('input', help='TIFF file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output directory')
    parser.add_argument('--backend', default='cellpose',
                        choices=['cellpose', 'sam', 'pixel_rf'],
                        help='ML backend (default: cellpose)')
    parser.add_argument('--bf-channel', type=int, default=1,
                        help='0-indexed BF channel (default: 1)')
    parser.add_argument('--z-projection', default='best_focus',
                        choices=['best_focus', 'max', 'mean', 'focus_range'],
                        help='Z-projection (default: best_focus)')
    parser.add_argument('--z-range', nargs=2, type=int, default=None,
                        metavar=('Z0', 'Z1'))
    parser.add_argument('--min-area', type=int, default=50000)
    parser.add_argument('--closing-radius', type=int, default=5)

    # Cellpose
    parser.add_argument('--cellpose-model', default='cyto3',
                        help='Cellpose model type (default: cyto3)')
    parser.add_argument('--cellpose-diameter', type=float, default=None,
                        help='Expected organoid diameter in pixels (None=auto)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for Cellpose/SAM')

    # SAM
    parser.add_argument('--sam-checkpoint', default=None,
                        help='Path to SAM or MobileSAM weights (.pt file)')
    parser.add_argument('--sam-model-type', default='auto',
                        help='SAM model type: auto, vit_h, vit_l, vit_b, mobile_sam')

    # Pixel RF
    parser.add_argument('--model', default=None,
                        help='Path to trained pixel RF model (.pkl)')

    parser.add_argument('--batch', action='store_true',
                        help='Process all TIFs in input directory')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    z_range = tuple(args.z_range) if args.z_range else None

    seg = MLSegmenter(
        backend=args.backend,
        bf_channel=args.bf_channel,
        z_projection=args.z_projection,
        z_range=z_range,
        min_area=args.min_area,
        closing_radius=args.closing_radius,
        cellpose_model=args.cellpose_model,
        cellpose_diameter=args.cellpose_diameter,
        cellpose_gpu=args.gpu,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        model_path=args.model,
        verbose=not args.quiet,
    )

    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        tif_files = sorted(input_path.glob('*.tif'))
        tif_files += sorted(input_path.glob('*.tiff'))
        tif_files = sorted(set(tif_files))
        print(f"Batch: {len(tif_files)} files in {input_path}")
        ok = 0
        for tif_path in tif_files:
            print(f"\n{'='*60}\n{tif_path.name}")
            out = Path(args.out) / tif_path.stem / 'result_segmentation'
            try:
                masks, meta = seg.segment_stack(str(tif_path), output_dir=str(out))
                print(f"  OK  mean_area={np.mean(meta['mask_areas']):.0f} px")
                ok += 1
            except Exception as e:
                print(f"  FAILED: {e}")
        print(f"\nDone: {ok}/{len(tif_files)} succeeded.")
    else:
        out = Path(args.out) / input_path.stem / 'result_segmentation'
        masks, meta = seg.segment_stack(str(input_path), output_dir=str(out))
        areas = meta['mask_areas']
        print(f"\nResults:")
        print(f"  Frames:      {masks.shape[0]}")
        print(f"  Mean area:   {np.mean(areas):.0f} px")
        print(f"  Area range:  {min(areas):,} – {max(areas):,} px")
        print(f"  Saved to:    {out}")


if __name__ == '__main__':
    main()
