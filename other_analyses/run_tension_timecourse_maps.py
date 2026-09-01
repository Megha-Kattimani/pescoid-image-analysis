"""Generate radial tension kymographs: how tension changes over ALL timepoints."""

import tifffile
import numpy as np
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the `analysis` package
from analysis.bf_segmentation import z_project
from skimage import measure, filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = Path(r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF")
ANALYSIS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo")


def norm(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def radial_angular_profile(stack_t, mask, n_sectors=24, n_radial=10):
    """Compute radial and angular profiles for one timepoint."""
    Z, C, Y, X = stack_t.shape
    gfp = np.max(stack_t[:, 0], axis=0).astype(float)
    lyn = np.max(stack_t[:, 1], axis=0).astype(float)

    gfp_n, lyn_n = norm(gfp), norm(lyn)
    grad = filters.sobel(lyn_n)

    props = measure.regionprops(mask.astype(int))
    if not props:
        return None
    p = props[0]
    cy, cx = p.centroid
    orientation = p.orientation

    yy, xx = np.mgrid[:Y, :X]
    dy, dx = yy - cy, xx - cx
    angles = (np.arctan2(dy, dx) - orientation + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt(dy**2 + dx**2)
    max_dist = dist[mask].max() if mask.any() else 1
    dist_n = dist / max_dist

    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    radial_edges = np.linspace(0, 1, n_radial + 1)

    # Radial profiles (averaged over angle)
    lyn_radial = np.full(n_radial, np.nan)
    gfp_radial = np.full(n_radial, np.nan)
    tension_radial = np.full(n_radial, np.nan)

    for r in range(n_radial):
        ring = mask & (dist_n >= radial_edges[r]) & (dist_n < radial_edges[r + 1])
        if ring.sum() < 5:
            continue
        lyn_radial[r] = lyn_n[ring].mean()
        gfp_radial[r] = gfp_n[ring].mean()
        tension_radial[r] = grad[ring].mean()

    # Angular profiles (averaged over radius)
    lyn_angular = np.full(n_sectors, np.nan)
    gfp_angular = np.full(n_sectors, np.nan)
    tension_angular = np.full(n_sectors, np.nan)

    for s in range(n_sectors):
        sector = mask & (angles >= sector_edges[s]) & (angles < sector_edges[s + 1])
        if sector.sum() < 5:
            continue
        lyn_angular[s] = lyn_n[sector].mean()
        gfp_angular[s] = gfp_n[sector].mean()
        tension_angular[s] = grad[sector].mean()

    return {
        'lyn_radial': lyn_radial, 'gfp_radial': gfp_radial,
        'tension_radial': tension_radial,
        'lyn_angular': lyn_angular, 'gfp_angular': gfp_angular,
        'tension_angular': tension_angular,
    }


count = 0
for exp_name in ['P_ctrl', 'P_Activin_3-5hpf']:
    data_dir = DATA_ROOT / exp_name
    tifs = sorted(data_dir.glob('*.tif'))

    for sample_path in tifs:
        pid = re.search(r'(G\d+)', sample_path.stem).group(1)
        mask_path = (ANALYSIS / exp_name / sample_path.stem /
                     'masks' / f'{sample_path.stem}_masks.tif')
        if not mask_path.exists():
            continue

        masks = tifffile.imread(str(mask_path)) > 0
        stack = tifffile.imread(str(sample_path))
        T = stack.shape[0]

        n_radial = 10
        n_sectors = 24

        # Build kymographs: (metric, radius/angle) x time
        lyn_rad_kymo = np.full((n_radial, T), np.nan)
        gfp_rad_kymo = np.full((n_radial, T), np.nan)
        tension_rad_kymo = np.full((n_radial, T), np.nan)
        lyn_ang_kymo = np.full((n_sectors, T), np.nan)
        gfp_ang_kymo = np.full((n_sectors, T), np.nan)
        tension_ang_kymo = np.full((n_sectors, T), np.nan)

        for t in range(T):
            if not masks[t].any():
                continue
            result = radial_angular_profile(stack[t], masks[t],
                                            n_sectors=n_sectors, n_radial=n_radial)
            if result is None:
                continue
            lyn_rad_kymo[:, t] = result['lyn_radial']
            gfp_rad_kymo[:, t] = result['gfp_radial']
            tension_rad_kymo[:, t] = result['tension_radial']
            lyn_ang_kymo[:, t] = result['lyn_angular']
            gfp_ang_kymo[:, t] = result['gfp_angular']
            tension_ang_kymo[:, t] = result['tension_angular']

        # Plot: 2 rows x 3 cols
        # Row 1: Radial kymographs (radius x time)
        # Row 2: Angular kymographs (angle x time)
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))

        for col, (data, cmap, title) in enumerate([
            (tension_rad_kymo, 'inferno', 'Tension'),
            (lyn_rad_kymo, 'hot', 'LynTom'),
            (gfp_rad_kymo, 'Greens', 'mezzo-GFP'),
        ]):
            ax = axes[0, col]
            im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                           extent=[0, T - 1, 0, 1], interpolation='bilinear')
            ax.set_xlabel('Time [frame]', fontsize=12)
            ax.set_ylabel('Radius (0=center, 1=edge)', fontsize=12)
            ax.set_title(f'{title} — Radial over Time', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

        for col, (data, cmap, title) in enumerate([
            (tension_ang_kymo, 'inferno', 'Tension'),
            (lyn_ang_kymo, 'hot', 'LynTom'),
            (gfp_ang_kymo, 'Greens', 'mezzo-GFP'),
        ]):
            ax = axes[1, col]
            im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                           extent=[0, T - 1, -180, 180], interpolation='bilinear')
            ax.set_xlabel('Time [frame]', fontsize=12)
            ax.set_ylabel('Angle (deg, 0=major axis tip)', fontsize=12)
            ax.set_title(f'{title} — Angular over Time', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle(f'{exp_name} / {pid} — Radial + Angular Kymographs',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        map_dir = ANALYSIS / exp_name / sample_path.stem / 'tension_maps'
        map_dir.mkdir(exist_ok=True)
        plt.savefig(str(map_dir / f'{pid}_tension_kymograph.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        count += 1
        print(f'  [{count}] {exp_name}/{pid} done')

print(f'\nDone: {count} pescoids')
