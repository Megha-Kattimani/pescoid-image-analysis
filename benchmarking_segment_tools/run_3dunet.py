"""
3D U-Net Segmentation Pipeline for Pescoid Benchmarking
========================================================

Trains a U-Net (PyTorch) on manually annotated GT masks (from Morgana training set),
then predicts on all images. Computes morphology metrics and generates box plots.

Training data:
  Z:/.../Benchmarking_Segmentation tools/JPG/model/trainingset/
    {name}.tif      <- BF images (512x512, uint8)
    {name}_GT.tif   <- manual GT masks (512x512, binary)

Input for prediction:
  Z:/.../Benchmarking_Segmentation tools/JPG/MORGANA/*.tif

Output:
  benchmarking_segment_tools/output/3dunet/
    model/          <- trained U-Net model (.pth)
    masks/          <- binary masks (TIF, 0/255)
    overlays/       <- 3-panel overlays (BF + contour + GFP)
    plots/          <- box plots comparing control vs treated
    training_overview/  <- GT vs prediction comparison
    3dunet_results.csv

Usage:
  python benchmarking_segment_tools/run_3dunet.py
"""

import sys
import glob as glob_mod
from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage import measure, filters
import tifffile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent

JPG_BASE = Path(r"Z:\Megha_Kattimani\Imaging\Zeiss\Benchmarking_Segmentation tools\JPG")
TRAINING_DIR = JPG_BASE / "model" / "trainingset"
BF_DIR = JPG_BASE / "MORGANA"
OUTPUT_DIR = _SCRIPT_DIR / "output" / "3dunet"

# Sample -> condition mapping
SAMPLE_MAP = {
    '033': 'control', '043': 'control',
    '049': 'treated', '055': 'treated', '061': 'treated',
    '073': 'treated', '075': 'treated', '095': 'treated', '129': 'treated',
}

# Training config
N_EPOCHS = 150
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
USE_GPU = True
IMG_SIZE = 512  # images are already 512x512


# ============================================================================
# U-Net Architecture (PyTorch)
# ============================================================================

class DoubleConv(nn.Module):
    """Two consecutive conv-BN-ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net with 4 encoder/decoder levels."""
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder path
        prev_ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(prev_ch, f))
            prev_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoders)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            # Handle size mismatches
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)

        return self.final_conv(x)


# ============================================================================
# Dataset
# ============================================================================

class PescoidDataset(Dataset):
    """Dataset for training U-Net on pescoid BF images + GT masks."""
    def __init__(self, images, labels, augment=True):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        lbl = self.labels[idx].astype(np.float32)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                lbl = np.flip(lbl, axis=0).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            lbl = np.rot90(lbl, k).copy()

        # Add channel dimension: (H, W) -> (1, H, W)
        img = img[np.newaxis, ...]
        lbl = lbl[np.newaxis, ...]

        return torch.from_numpy(img), torch.from_numpy(lbl)


# ============================================================================
# Dice Loss + BCE combo
# ============================================================================

class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sig * target).sum()
        dice = (2.0 * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
        dice_loss = 1.0 - dice

        return bce_loss + dice_loss


# ============================================================================
# STEP 0: Discover images
# ============================================================================

def discover_images() -> List[dict]:
    """Find all TIF images in the MORGANA BF folder and map metadata."""
    images = []
    for tif_path in sorted(BF_DIR.glob("*.tif")):
        name = tif_path.stem
        name_lower = name.lower()
        if name_lower.endswith('e'):
            sample_id = name_lower[:-1]
            timepoint = 'elongated'
        else:
            sample_id = name_lower
            timepoint = 'initial'

        condition = SAMPLE_MAP.get(sample_id)
        if condition is None:
            print(f"  Warning: unknown sample '{sample_id}' in {name}, skipping")
            continue

        # Find original JPG to extract GFP channel
        if condition == 'control':
            tp_folder = 'elongated' if timepoint == 'elongated' else 'initial'
        else:
            tp_folder = 'elongation' if timepoint == 'elongated' else 'initial'
        jpg_pattern = f"*#{sample_id}*"
        jpg_folder = JPG_BASE / condition / tp_folder
        jpg_files = list(jpg_folder.glob(jpg_pattern + ".jpg"))

        images.append({
            'bf_path': tif_path,
            'jpg_path': jpg_files[0] if jpg_files else None,
            'name': name,
            'sample_id': sample_id,
            'condition': condition,
            'timepoint': timepoint,
        })

    return images


def split_channels_from_jpg(jpg_path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract BF and GFP from merged JPG."""
    from skimage import io as skio
    img_rgb = skio.imread(str(jpg_path))
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    bf = r / 255.0
    gfp_raw = np.clip(g - r, 0, 255)
    gfp_max = gfp_raw.max()
    gfp = gfp_raw / gfp_max if gfp_max > 0 else gfp_raw
    return bf, gfp


# ============================================================================
# STEP 1: Load GT training data
# ============================================================================

def load_training_data():
    """Load BF images and GT masks from the Morgana training set."""
    gt_files = sorted(TRAINING_DIR.glob("*_GT.tif"))
    print(f"\nFound {len(gt_files)} GT masks in {TRAINING_DIR}")

    train_images = []
    train_labels = []
    train_names = []

    for gt_path in gt_files:
        name = gt_path.stem.replace('_GT', '')
        img_path = gt_path.parent / f"{name}.tif"

        if not img_path.exists():
            print(f"  Warning: no image for {name}, skipping")
            continue

        img = tifffile.imread(str(img_path))   # uint8, 512x512
        gt = tifffile.imread(str(gt_path))     # uint16, binary (0/1)

        # U-Net expects binary labels
        label = (gt > 0).astype(np.float32)

        train_images.append(img)
        train_labels.append(label)
        train_names.append(name)

        area = int(label.sum())
        print(f"  {name}: image {img.shape}/{img.dtype}, GT area={area:,} px")

    return train_images, train_labels, train_names


# ============================================================================
# STEP 2: Train U-Net
# ============================================================================

def train_unet(train_images, train_labels, model_dir):
    """Train a U-Net on the GT masks."""
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset and dataloader
    dataset = PescoidDataset(train_images, train_labels, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True)

    # Initialize model
    model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Training set: {len(dataset)} images (all GT annotations)")
    print(f"  Epochs: {N_EPOCHS}, LR: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")
    print()

    train_losses = []
    dice_scores = []

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0

        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()

            # Compute dice score for monitoring
            with torch.no_grad():
                pred_bin = (torch.sigmoid(preds) > 0.5).float()
                intersection = (pred_bin * lbls).sum()
                dice = (2.0 * intersection + 1e-6) / (pred_bin.sum() + lbls.sum() + 1e-6)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        train_losses.append(avg_loss)
        dice_scores.append(avg_dice)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
                  f"loss={avg_loss:.4f}, dice={avg_dice:.4f}, lr={lr:.2e}")

    # Save model
    model_path = model_dir / 'unet_pescoid.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': N_EPOCHS,
        'train_losses': train_losses,
        'dice_scores': dice_scores,
    }, str(model_path))
    print(f"\n  Model saved to: {model_path}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final dice score: {dice_scores[-1]:.4f}")

    # Save training loss + dice plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, color='#2196F3', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Dice + BCE)')
    ax1.set_title('U-Net Training Loss')
    ax1.grid(alpha=0.3)

    ax2.plot(dice_scores, color='#4CAF50', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('U-Net Training Dice')
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(model_dir / 'training_curves.png'), dpi=150)
    plt.close(fig)

    return str(model_path), model, device


# ============================================================================
# STEP 3: Predict with trained model
# ============================================================================

def predict_with_trained_model(images: List[dict], model_path: str,
                                model=None, device=None) -> List[dict]:
    """Run prediction using the trained U-Net."""
    if model is None:
        device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
        model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    results = []

    for img_info in images:
        bf_img = tifffile.imread(str(img_info['bf_path']))
        print(f"  Predicting {img_info['name']}...", end=' ')

        # Prepare input: normalize, add batch + channel dims
        img_tensor = torch.from_numpy(
            bf_img.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]

        # Threshold at 0.5
        binary_mask = pred_prob > 0.5

        if binary_mask.sum() == 0:
            print("no mask found!")
        else:
            # Keep largest component
            from scipy import ndimage as ndi
            labeled, n = ndi.label(binary_mask)
            if n > 0:
                sizes = ndi.sum(binary_mask, labeled, range(1, n + 1))
                largest = np.argmax(sizes) + 1
                binary_mask = (labeled == largest)
            area = binary_mask.sum()
            print(f"area={area:,} px")

        img_info['mask'] = binary_mask
        img_info['pred_prob'] = pred_prob
        results.append(img_info)

    return results


# ============================================================================
# STEP 4: Training overview (GT vs prediction)
# ============================================================================

def save_training_overview(train_images, train_labels, train_names, model, device, output_dir):
    """Compare GT masks vs U-Net predictions on the training images."""
    overview_dir = output_dir / 'training_overview'
    overview_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for i, (img, gt, name) in enumerate(zip(train_images, train_labels, train_names)):
        # Predict
        img_tensor = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_logits = model(img_tensor)
            pred_prob = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]

        pred = pred_prob > 0.5
        gt_bool = gt > 0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'{name} — BF', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img, cmap='gray')
        for c in measure.find_contours(gt_bool.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'g-', linewidth=2)
        axes[1].set_title(f'GT mask (area={gt_bool.sum():,})', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(pred_prob, cmap='magma', vmin=0, vmax=1)
        axes[2].set_title('U-Net probability map', fontsize=10)
        axes[2].axis('off')

        axes[3].imshow(img, cmap='gray')
        for c in measure.find_contours(pred.astype(float), 0.5):
            axes[3].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
        if gt_bool.sum() > 0:
            for c in measure.find_contours(gt_bool.astype(float), 0.5):
                axes[3].plot(c[:, 1], c[:, 0], 'g--', linewidth=1, alpha=0.7)
        axes[3].set_title(f'U-Net prediction (area={pred.sum():,})', fontsize=10)
        axes[3].axis('off')

        # Compute IoU
        intersection = (gt_bool & pred).sum()
        union = (gt_bool | pred).sum()
        iou = intersection / union if union > 0 else 0
        plt.suptitle(f'{name} — GT vs 3D U-Net (IoU={iou:.3f})', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(str(overview_dir / f'overview_{name}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"  Training overview saved to: {overview_dir}")


# ============================================================================
# STEP 5: Compute metrics
# ============================================================================

def compute_metrics(mask: np.ndarray, gfp: np.ndarray) -> dict:
    """Compute morphology + mezzo metrics from mask and GFP."""
    if mask.sum() == 0:
        return {'aspect_ratio': np.nan, 'circularity': np.nan, 'solidity': np.nan,
                'area': 0, 'eccentricity': np.nan, 'perimeter': 0,
                'mezzo_fraction': np.nan, 'mezzo_mean': np.nan}

    props = measure.regionprops(mask.astype(int))
    if not props:
        return {'aspect_ratio': np.nan, 'circularity': np.nan, 'solidity': np.nan,
                'area': 0, 'eccentricity': np.nan, 'perimeter': 0,
                'mezzo_fraction': np.nan, 'mezzo_mean': np.nan}

    p = props[0]
    perimeter = p.perimeter if p.perimeter > 0 else 1e-6

    result = {
        'aspect_ratio': float(p.major_axis_length / p.minor_axis_length) if p.minor_axis_length > 0 else 1.0,
        'circularity': float((4 * np.pi * p.area) / (perimeter ** 2)),
        'solidity': float(p.solidity),
        'area': int(p.area),
        'eccentricity': float(p.eccentricity),
        'perimeter': float(perimeter),
    }

    # Mezzo (GFP) analysis
    gfp_in_mask = gfp[mask]
    if gfp_in_mask.size > 0 and gfp.max() > 0:
        try:
            thresh = filters.threshold_otsu(gfp_in_mask)
        except ValueError:
            thresh = 0.1
        result['mezzo_fraction'] = float((gfp_in_mask > thresh).sum() / gfp_in_mask.size)
        result['mezzo_mean'] = float(gfp_in_mask.mean())
    else:
        result['mezzo_fraction'] = 0.0
        result['mezzo_mean'] = 0.0

    return result


# ============================================================================
# STEP 6: Save overlays
# ============================================================================

def save_overlay(bf, gfp, mask, save_path, img_info):
    """Save 3-panel overlay: BF | BF+contour | GFP+contour."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bf, cmap='gray')
    axes[0].set_title(f"BF — {img_info['condition']} {img_info['timepoint']}", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(bf, cmap='gray')
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], 'r-', linewidth=2)
    axes[1].set_title(f"3D U-Net — area={int(mask.sum()):,} px", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(gfp, cmap='Greens', vmin=0, vmax=max(gfp.max(), 0.01))
    if mask.sum() > 0:
        for c in measure.find_contours(mask.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], 'w-', linewidth=1.5)
    axes[2].set_title('GFP (mezzo)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f"{img_info['sample_id']} — 3D U-Net segmentation", fontsize=9, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# STEP 7: Box plots
# ============================================================================

def generate_boxplots(rows: List[dict], output_dir: Path):
    """Generate box plots: control vs treated at initial vs elongated."""
    import pandas as pd

    df = pd.DataFrame(rows)
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['aspect_ratio', 'circularity', 'solidity', 'area', 'mezzo_fraction', 'mezzo_mean']
    metric_labels = {
        'area': 'Area (pixels)',
        'aspect_ratio': 'Aspect Ratio',
        'circularity': 'Circularity',
        'solidity': 'Solidity',
        'mezzo_fraction': 'Mezzo Fraction',
        'mezzo_mean': 'Mezzo Mean Intensity',
    }
    cond_colors = {'control': '#888888', 'treated': '#cc3333'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        groups, tick_labels, colors = [], [], []
        for tp in ['initial', 'elongated']:
            for cond in ['control', 'treated']:
                subset = df[(df['condition'] == cond) & (df['timepoint'] == tp)]
                vals = subset[metric].dropna().values
                groups.append(vals if len(vals) > 0 else np.array([np.nan]))
                tick_labels.append(f"{cond}\n({tp})")
                colors.append(cond_colors[cond])

        bp = ax.boxplot(groups, tick_labels=tick_labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        for j, vals in enumerate(groups):
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                x = np.random.normal(j + 1, 0.05, size=len(valid))
                ax.scatter(x, valid, alpha=0.8, s=40, c=colors[j],
                           edgecolors='black', linewidth=0.5, zorder=3)
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('3D U-Net — Control vs Treated', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(plot_dir / 'boxplot_3dunet.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Box plots saved to: {plot_dir}")

    csv_path = output_dir / '3dunet_results.csv'
    try:
        df.to_csv(str(csv_path), index=False)
        print(f"  Results CSV saved to: {csv_path}")
    except PermissionError:
        alt_path = output_dir / '3dunet_results_new.csv'
        df.to_csv(str(alt_path), index=False)
        print(f"  WARNING: {csv_path} is locked. Saved to: {alt_path}")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  3D U-NET SEGMENTATION PIPELINE")
    print("=" * 60)

    # Create output directories
    model_dir = OUTPUT_DIR / 'model'
    mask_dir = OUTPUT_DIR / 'masks'
    overlay_dir = OUTPUT_DIR / 'overlays'
    for d in [model_dir, mask_dir, overlay_dir, OUTPUT_DIR / 'plots']:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load GT training data
    print(f"\n{'=' * 60}")
    print("STEP 1: Loading GT training data")
    print(f"{'=' * 60}")
    train_images, train_labels, train_names = load_training_data()

    # Step 2: Train U-Net
    print(f"\n{'=' * 60}")
    print("STEP 2: Training U-Net on GT masks")
    print(f"{'=' * 60}")
    model_path, model, device = train_unet(train_images, train_labels, model_dir)

    # Step 3: Training overview
    print(f"\n{'=' * 60}")
    print("STEP 3: Saving training overview (GT vs prediction)")
    print(f"{'=' * 60}")
    save_training_overview(train_images, train_labels, train_names, model, device, OUTPUT_DIR)

    # Step 4: Discover all images and predict
    print(f"\n{'=' * 60}")
    print("STEP 4: Predicting on all images")
    print(f"{'=' * 60}")
    images = discover_images()
    print(f"\nFound {len(images)} images")
    for cond in ['control', 'treated']:
        for tp in ['initial', 'elongated']:
            n = sum(1 for im in images if im['condition'] == cond and im['timepoint'] == tp)
            if n > 0:
                print(f"  {cond}/{tp}: {n}")

    results = predict_with_trained_model(images, model_path, model, device)

    # Step 5: Compute metrics and save masks/overlays
    print(f"\n{'=' * 60}")
    print("STEP 5: Computing metrics and saving outputs")
    print(f"{'=' * 60}")
    rows = []
    for img_info in results:
        mask = img_info['mask']

        # Get GFP from original JPG
        if img_info['jpg_path'] is not None:
            _, gfp = split_channels_from_jpg(img_info['jpg_path'])
        else:
            gfp = np.zeros_like(mask, dtype=float)
            print(f"  Warning: no JPG found for {img_info['name']}, GFP will be zeros")

        metrics = compute_metrics(mask, gfp)
        metrics['sample_id'] = img_info['sample_id']
        metrics['condition'] = img_info['condition']
        metrics['timepoint'] = img_info['timepoint']
        metrics['name'] = img_info['name']
        rows.append(metrics)

        out_name = f"{img_info['sample_id']}_{img_info['condition']}_{img_info['timepoint']}"
        tifffile.imwrite(str(mask_dir / f"{out_name}_mask.tif"),
                         (mask.astype(np.uint8) * 255))

        bf_img = tifffile.imread(str(img_info['bf_path']))
        bf_norm = bf_img.astype(float) / 255.0
        save_overlay(bf_norm, gfp, mask, overlay_dir / f"{out_name}_overlay.png", img_info)

        print(f"  {img_info['name']}: AR={metrics['aspect_ratio']:.2f}, "
              f"circ={metrics['circularity']:.2f}, sol={metrics['solidity']:.2f}, "
              f"mezzo={metrics['mezzo_fraction']:.2f}")

    # Step 6: Box plots
    print(f"\n{'=' * 60}")
    print("STEP 6: Generating box plots")
    print(f"{'=' * 60}")
    generate_boxplots(rows, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("  DONE!")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    print(f"  Model:    {model_dir}")
    print(f"  Masks:    {mask_dir}")
    print(f"  Overlays: {overlay_dir}")
    print(f"  Overview: {OUTPUT_DIR / 'training_overview'}")
    print(f"  Plots:    {OUTPUT_DIR / 'plots'}")
    print(f"  CSV:      {OUTPUT_DIR / '3dunet_results.csv'}")


if __name__ == '__main__':
    main()
