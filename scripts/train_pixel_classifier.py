"""
Train Pixel Classifier from Fiji Annotations
=============================================
Ilastik-style Random Forest training for pescoid BF segmentation.
Train on 2-5 manually annotated frames, apply to the whole batch.

HOW TO ANNOTATE IN FIJI (step-by-step)
---------------------------------------
1. Open a raw TIFF file in Fiji.
2. Navigate to the frame (timepoint) that failed the automated segmentation.
3. Use one of these tools to draw the organoid boundary:
      Polygon Selections  (press P)  →  trace around organoid
      Freehand Selection  (press F)  →  draw freehand around organoid
4. Once selected:  Edit → Fill  (sets interior to white = 255)
5. If needed, invert (Edit → Invert) so organoid = white, background = black.
6. Flatten to 8-bit if needed: Image → Type → 8-bit
7. Save as TIFF:  File → Save As → Tiff
8. Name the file:  <anything>_annotation.tif
   Example:  Experiment-898_t0025_annotation.tif

QUICK FIJI MACRO (paste into Macro editor, run on each frame)
--------------------------------------------------------------
  run("8-bit");
  setAutoThreshold("Default dark");
  run("Convert to Mask");
  saveAs("Tiff", getDirectory("image") + "frame_annotation.tif");

TRAINING
--------
  python scripts/train_pixel_classifier.py annotations/ \\
      --tiff-dir "Z:/Megha_Kattimani/.../TIFF" \\
      --bf-channel 1 \\
      --model annotations/model.pkl

APPLYING TO BATCH
-----------------
  python analysis/ml_segmentation.py "Z:/Megha_Kattimani/.../TIFF" \\
      --batch --out output/ --backend pixel_rf --model annotations/model.pkl
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import exposure, measure

# Add analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis'))
from ml_segmentation import PixelClassifier, train_from_annotations, _normalize, _z_project
from bf_segmentation import load_tiff, extract_channel


def visualize_training_data(images, masks, output_dir):
    """Save a diagnostic grid showing each training frame + its annotation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, mask) in enumerate(zip(images, masks)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        img_disp = exposure.rescale_intensity(img.astype(float), out_range=(0, 1))
        axes[0].imshow(img_disp, cmap='gray')
        axes[0].set_title('BF (raw)')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Annotation  (fg={mask.sum():,} px)')
        axes[1].axis('off')

        axes[2].imshow(img_disp, cmap='gray')
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.suptitle(f'Training frame {i+1}', fontsize=12)
        plt.tight_layout()
        plt.savefig(str(output_dir / f'training_frame_{i+1:03d}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved training QC images to: {output_dir}")


def validate_model(clf, images, masks, output_dir):
    """Run the trained model on training data and report accuracy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ious = []
    for i, (img, gt_mask) in enumerate(zip(images, masks)):
        pred = clf.predict(img)
        gt = (gt_mask > 0)

        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        iou = intersection / (union + 1e-6)
        ious.append(iou)

        # Save comparison
        img_disp = exposure.rescale_intensity(img.astype(float), out_range=(0, 1))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_disp, cmap='gray')
        for c in measure.find_contours(gt.astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], 'g-', linewidth=2, label='Annotation')
        axes[0].set_title('Ground Truth (green)')
        axes[0].axis('off')

        axes[1].imshow(img_disp, cmap='gray')
        for c in measure.find_contours(pred.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
        axes[1].set_title('Prediction (red)')
        axes[1].axis('off')

        # Difference map
        diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
        diff[(pred & gt)]  = [0, 200, 0]    # True positive (green)
        diff[(pred & ~gt)] = [200, 0, 0]    # False positive (red)
        diff[(~pred & gt)] = [0, 0, 200]    # False negative (blue)
        axes[2].imshow(diff)
        axes[2].set_title(
            f'Diff  IoU={iou:.3f}\nGreen=OK  Red=FP  Blue=FN', fontsize=9)
        axes[2].axis('off')

        plt.suptitle(f'Validation frame {i+1}  (IoU={iou:.3f})',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_dir / f'validation_frame_{i+1:03d}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    mean_iou = np.mean(ious)
    print(f"\n  Validation (on training data):")
    for i, iou in enumerate(ious):
        quality = 'GOOD' if iou > 0.85 else ('OK' if iou > 0.70 else 'POOR')
        print(f"    Frame {i+1}: IoU={iou:.3f}  [{quality}]")
    print(f"  Mean IoU: {mean_iou:.3f}")

    if mean_iou < 0.70:
        print("\n  WARNING: Low IoU. Suggestions:")
        print("    - Add more annotated frames (especially from different conditions)")
        print("    - Check that annotations are accurate (use Fiji to verify)")
        print("    - Try annotating frames with different lighting / organoid sizes")
    elif mean_iou > 0.90:
        print("\n  Excellent model quality. Ready for batch segmentation.")

    return ious


def main():
    parser = argparse.ArgumentParser(
        description='Train pixel RF classifier from Fiji annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How to annotate in Fiji:
  1. Open a raw TIFF → navigate to difficult frame
  2. Use Polygon/Freehand selection to outline organoid
  3. Edit → Fill  (interior becomes white=255)
  4. Save as: <name>_annotation.tif

Examples:
  # Basic: annotations and raw TIFs in same folder
  python scripts/train_pixel_classifier.py annotations/ --model model.pkl

  # Raw TIFs in separate location
  python scripts/train_pixel_classifier.py annotations/ \\
      --tiff-dir "Z:/Megha_Kattimani/.../TIFF" \\
      --model annotations/model.pkl

  # After training, apply to batch:
  python analysis/ml_segmentation.py "Z:/path/TIFF" --batch \\
      --out output/ --backend pixel_rf --model annotations/model.pkl
""",
    )

    parser.add_argument('annotation_dir',
                        help='Folder containing *_annotation.tif files')
    parser.add_argument('--tiff-dir', default=None,
                        help='Folder with raw TIFF files (if different from annotation_dir)')
    parser.add_argument('--bf-channel', type=int, default=1,
                        help='0-indexed BF channel (default: 1)')
    parser.add_argument('--z-projection', default='best_focus',
                        choices=['best_focus', 'max', 'mean', 'focus_range'])
    parser.add_argument('--z-range', nargs=2, type=int, default=None,
                        metavar=('Z0', 'Z1'))
    parser.add_argument('--model', default='model.pkl',
                        help='Output model file path (default: model.pkl)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of RF trees (default: 200, more=better but slower)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Run validation on training data after training')
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    parser.add_argument('--qc-dir', default=None,
                        help='Directory for QC visualizations (default: <annotation_dir>/training_qc)')

    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    z_range = tuple(args.z_range) if args.z_range else None
    qc_dir = Path(args.qc_dir) if args.qc_dir else annotation_dir / 'training_qc'
    model_path = Path(args.model)

    print("=" * 60)
    print("PIXEL CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Annotation dir: {annotation_dir}")
    print(f"TIFF dir:       {args.tiff_dir or '(same as annotation dir)'}")
    print(f"BF channel:     {args.bf_channel} (0-indexed)")
    print(f"Model output:   {model_path}")

    # Run training
    clf = train_from_annotations(
        annotation_dir=annotation_dir,
        tiff_dir=args.tiff_dir,
        bf_channel=args.bf_channel,
        z_projection=args.z_projection,
        z_range=z_range,
        n_estimators=args.n_estimators,
        output_model=str(model_path),
        verbose=True,
    )

    # Collect training images for QC
    from ml_segmentation import _load_tiff
    ann_files = sorted(annotation_dir.glob('*_annotation.tif'))
    ann_files += sorted(annotation_dir.glob('*_annotation.tiff'))
    ann_files = sorted(set(ann_files))

    images, masks = [], []
    for ann_path in ann_files:
        ann_mask = tifffile.imread(str(ann_path))
        if ann_mask.ndim > 2:
            ann_mask = ann_mask[0]
        ann_mask = (ann_mask > 0).astype(np.uint8)

        stem = ann_path.stem.replace('_annotation', '')
        raw_path = None
        search = [annotation_dir] + ([Path(args.tiff_dir)] if args.tiff_dir else [])
        for d in search:
            for ext in ('.tif', '.tiff'):
                c = d / (stem + ext)
                if c.exists() and 'annotation' not in c.stem.lower():
                    raw_path = c
                    break
            if raw_path:
                break

        if raw_path:
            stack, meta = _load_tiff(str(raw_path))
            T, Z, C = stack.shape[:3]
            ch = args.bf_channel if args.bf_channel < C else 0
            bf_z = stack[0, :, ch, :, :]
            bf_2d = _z_project(bf_z, method=args.z_projection, z_range=z_range)
            images.append(bf_2d)
            masks.append(ann_mask)

    if images:
        print(f"\nSaving training QC visualizations...")
        visualize_training_data(images, masks, qc_dir / 'training_data')

        if args.validate:
            print(f"\nRunning validation...")
            validate_model(clf, images, masks, qc_dir / 'validation')

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"Model saved: {model_path}")
    print(f"QC images:   {qc_dir}")
    print("=" * 60)
    print("\nNext step — apply to batch:")
    print(f'  python analysis/ml_segmentation.py "Z:/path/to/TIFF" \\')
    print(f'      --batch --out output/ --backend pixel_rf --model "{model_path}"')


if __name__ == '__main__':
    main()
