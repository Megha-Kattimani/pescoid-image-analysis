"""Generate radial+angular tension/expression maps for pescoids."""

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
from matplotlib.gridspec import GridSpec

DATA_ROOT = Path(r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF")
ANALYSIS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo")
OUT = ANALYSIS / "tissue_tension_analysis"
OUT.mkdir(exist_ok=True)


def norm(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def angular_radial_analysis(stack_t, mask, n_sectors=24, n_radial=10):
    Z, C, Y, X = stack_t.shape
    bf = z_project(stack_t[:, 2], method='best_focus').astype(float)
    gfp = np.max(stack_t[:, 0], axis=0).astype(float)
    lyn = np.max(stack_t[:, 1], axis=0).astype(float)

    gfp_n, lyn_n, bf_n = norm(gfp), norm(lyn), norm(bf)
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

    lyn_map = np.full((n_sectors, n_radial), np.nan)
    gfp_map = np.full((n_sectors, n_radial), np.nan)
    tension_map = np.full((n_sectors, n_radial), np.nan)

    for s in range(n_sectors):
        for r in range(n_radial):
            region = (mask &
                      (angles >= sector_edges[s]) & (angles < sector_edges[s + 1]) &
                      (dist_n >= radial_edges[r]) & (dist_n < radial_edges[r + 1]))
            if region.sum() < 5:
                continue
            lyn_map[s, r] = lyn_n[region].mean()
            gfp_map[s, r] = gfp_n[region].mean()
            tension_map[s, r] = grad[region].mean()

    lyn_angular = np.nanmean(lyn_map, axis=1)
    gfp_angular = np.nanmean(gfp_map, axis=1)
    tension_angular = np.nanmean(tension_map, axis=1)
    angle_centers = [(sector_edges[i] + sector_edges[i + 1]) / 2 for i in range(n_sectors)]

    return {
        'lyn_map': lyn_map, 'gfp_map': gfp_map, 'tension_map': tension_map,
        'lyn_angular': lyn_angular, 'gfp_angular': gfp_angular,
        'tension_angular': tension_angular,
        'angle_centers': angle_centers,
        'bf_n': bf_n, 'gfp_n': gfp_n, 'lyn_n': lyn_n, 'grad': grad,
        'centroid': (cy, cx), 'orientation': orientation,
    }


def plot_tension_map(result, mask, exp_name, pid, t, save_path):
    fig = plt.figure(figsize=(28, 16))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1.2])

    # Row 1: raw images
    for col, (img, cmap, title) in enumerate([
        (result['bf_n'], 'gray', 'BF'),
        (result['lyn_n'], 'hot', 'LynTom'),
        (result['gfp_n'], 'Greens', 'mezzo-GFP'),
        (result['grad'], 'inferno', 'Tension (gradient)'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        masked_img = np.where(mask, img, np.nan)
        ax.imshow(masked_img, cmap=cmap)
        for c in measure.find_contours(mask.astype(float), 0.5):
            lc = 'w-' if cmap != 'gray' else 'r-'
            ax.plot(c[:, 1], c[:, 0], lc, lw=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Row 1 col 5: Angular polar plot
    ax_polar = fig.add_subplot(gs[0, 4], polar=True)
    angles = result['angle_centers'] + [result['angle_centers'][0]]
    for vals, color, label in [
        (result['tension_angular'], 'red', 'Tension'),
        (result['lyn_angular'], 'orange', 'LynTom'),
        (result['gfp_angular'], 'green', 'mezzo'),
    ]:
        v = list(vals) + [vals[0]]
        mx = max(np.nanmax(v), 1e-6)
        v_norm = [x / mx for x in v]
        ax_polar.plot(angles, v_norm, '-o', color=color, lw=2, markersize=3, label=label)
        ax_polar.fill(angles, v_norm, color=color, alpha=0.1)
    ax_polar.set_title('Angular distribution', fontsize=11, pad=15)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # Row 2: Radial-angular heatmaps
    for col, (data, cmap, title) in enumerate([
        (result['lyn_map'], 'hot', 'LynTom (angle x radius)'),
        (result['gfp_map'], 'Greens', 'mezzo-GFP (angle x radius)'),
        (result['tension_map'], 'inferno', 'Tension (angle x radius)'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                       extent=[0, 1, -180, 180])
        ax.set_xlabel('Radius (0=center, 1=edge)', fontsize=10)
        ax.set_ylabel('Angle (deg)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2 col 4: Radial profiles overlaid
    ax = fig.add_subplot(gs[1, 3])
    r_bins = np.linspace(0, 1, result['lyn_map'].shape[1])
    for vals_2d, color, label in [
        (result['tension_map'], 'red', 'Tension'),
        (result['lyn_map'], 'orange', 'LynTom'),
        (result['gfp_map'], 'green', 'mezzo'),
    ]:
        vals = np.nanmean(vals_2d, axis=0)
        mx = np.nanmax(vals) if np.nanmax(vals) > 0 else 1
        ax.plot(r_bins, vals / mx, '-o', color=color, lw=2, markersize=4, label=label)
    ax.set_xlabel('Radius (center to edge)', fontsize=11)
    ax.set_ylabel('Normalized intensity', fontsize=11)
    ax.set_title('Radial profiles', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Row 2 col 5: GFP vs LynTom angular correlation
    ax = fig.add_subplot(gs[1, 4])
    valid = ~(np.isnan(result['gfp_angular']) | np.isnan(result['lyn_angular']))
    if valid.sum() > 3:
        ax.scatter(result['lyn_angular'][valid], result['gfp_angular'][valid],
                   c='purple', s=50, edgecolors='black', lw=0.5)
        z = np.polyfit(result['lyn_angular'][valid], result['gfp_angular'][valid], 1)
        x_fit = np.linspace(result['lyn_angular'][valid].min(),
                            result['lyn_angular'][valid].max(), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), 'k--', lw=1.5)
        corr = np.corrcoef(result['lyn_angular'][valid],
                           result['gfp_angular'][valid])[0, 1]
        ax.set_title(f'mezzo vs LynTom (r={corr:.2f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('LynTom (angular mean)', fontsize=10)
    ax.set_ylabel('mezzo-GFP (angular mean)', fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(f'{exp_name} / {pid} / t={t} -- Tissue Tension + Expression Maps',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()


# Run on ALL pescoids (P only), 3 timepoints each
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

        # Save maps for early, mid, late
        map_dir = ANALYSIS / exp_name / sample_path.stem / 'tension_maps'
        map_dir.mkdir(exist_ok=True)

        for t in [5, T // 2, T - 2]:
            result = angular_radial_analysis(stack[t], masks[t])
            if result is None:
                continue
            save_path = map_dir / f'{pid}_t{t:02d}_tension_map.png'
            plot_tension_map(result, masks[t], exp_name, pid, t, save_path)

        count += 1
        print(f'  [{count}] {exp_name}/{pid} done')

print(f'\nDone: {count} pescoids')
print(f'Maps saved in each pescoid folder under tension_maps/')
