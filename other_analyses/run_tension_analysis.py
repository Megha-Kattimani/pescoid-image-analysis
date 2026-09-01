"""
Tissue tension analysis from LynTom membrane channel using structure tensor.

For every pescoid x frame:
  - Compute structure tensor → eigenvalues lam1 >= lam2
  - Coherence (anisotropy) = (lam1 - lam2) / (lam1 + lam2)
  - Orientation = 0.5 * arctan2(2*Jxy, Jxx - Jyy)
  - Nematic order parameter Q = sqrt(<cos(2θ)>² + <sin(2θ)>²) over mask
  - Mean coherence in mask, in interior (eroded), in cortex (boundary band)
  - Radial profile of coherence: center to edge

Outputs:
  CSV per pescoid + aggregated trajectory plots
  Visual previews (orientation field maps) for 4 samples per condition
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from skimage import filters as skf, measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF"
)
ANALYSIS_DIR = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_yen_fixed")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\lyntom_tension")
OUT.mkdir(parents=True, exist_ok=True)
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
PREVIEWS = OUT / "previews"
PREVIEWS.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COLORS = {"P_ctrl": "#377eb8", "P_Activin_3-5hpf": "#ff7f00"}
LABELS = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}

HPF_START = 6.0
HPF_INTERVAL = 0.4
SIGMA_GRAD = 5
SIGMA_TENSOR = 10
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


def z_project_bf(z_stack):
    if z_stack.ndim == 2:
        return z_stack.astype(float)
    Z = z_stack.shape[0]
    scores = []
    for z in range(Z):
        img = z_stack[z].astype(float)
        dx = img[:, 2:] - img[:, :-2]
        dy = img[2:, :] - img[:-2, :]
        scores.append(np.mean(dx**2) + np.mean(dy**2))
    return z_stack[int(np.argmax(scores))].astype(float)


def norm_pct(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def structure_tensor_fields(img, sigma_grad=SIGMA_GRAD, sigma_tensor=SIGMA_TENSOR):
    ax_y = ndi.gaussian_filter(img, sigma=sigma_grad, order=(1, 0))
    ax_x = ndi.gaussian_filter(img, sigma=sigma_grad, order=(0, 1))
    Jxx = ndi.gaussian_filter(ax_x * ax_x, sigma=sigma_tensor)
    Jyy = ndi.gaussian_filter(ax_y * ax_y, sigma=sigma_tensor)
    Jxy = ndi.gaussian_filter(ax_x * ax_y, sigma=sigma_tensor)
    trace = Jxx + Jyy
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2
    coherence = np.where(lam1 + lam2 > 0,
                         (lam1 - lam2) / (lam1 + lam2 + 1e-10), 0.0)
    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    energy = np.sqrt(lam1 ** 2 + lam2 ** 2)
    return orientation, coherence, energy


def nematic_order(orientation, coherence, mask):
    if not mask.any():
        return 0.0, 0.0
    angles = orientation[mask]
    weights = coherence[mask]
    weights = weights / max(weights.sum(), 1e-10)
    c2 = np.sum(weights * np.cos(2 * angles))
    s2 = np.sum(weights * np.sin(2 * angles))
    Q = np.hypot(c2, s2)
    return float(Q), float(np.degrees(0.5 * np.arctan2(s2, c2)))


def radial_coherence_profile(coherence, mask, n_bins=10):
    if not mask.any():
        return np.full(n_bins, np.nan)
    props = measure.regionprops(mask.astype(int))
    cy, cx = props[0].centroid
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    dist = np.hypot(yy - cy, xx - cx)
    max_d = dist[mask].max() if mask.any() else 1
    dist_n = dist / max_d
    edges = np.linspace(0, 1, n_bins + 1)
    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        ring = mask & (dist_n >= edges[i]) & (dist_n < edges[i + 1])
        if ring.sum() > 5:
            profile[i] = coherence[ring].mean()
    return profile


def cortex_vs_interior(coherence, mask, cortex_thickness=15):
    interior = morphology.binary_erosion(mask, morphology.disk(cortex_thickness))
    cortex = mask & ~interior
    cort = float(coherence[cortex].mean()) if cortex.any() else 0.0
    inter = float(coherence[interior].mean()) if interior.any() else 0.0
    return cort, inter


def process_sample(stack, masks_aligned):
    T = min(stack.shape[0], len(masks_aligned))
    rows = []
    radial_profiles = []

    for t in range(T):
        bf_mask = masks_aligned[t]
        if not bf_mask.any():
            continue
        lyn_raw = np.max(stack[t, :, 1], axis=0).astype(float)
        lyn_n = norm_pct(lyn_raw)
        orient, coh, energy = structure_tensor_fields(lyn_n)
        Q, dom_angle = nematic_order(orient, coh, bf_mask)
        cort_coh, int_coh = cortex_vs_interior(coh, bf_mask)
        prof = radial_coherence_profile(coh, bf_mask, n_bins=10)
        radial_profiles.append(prof)

        props = measure.regionprops(bf_mask.astype(int))
        if props:
            p = props[0]
            major = float(p.major_axis_length)
            minor = float(p.minor_axis_length)
            ar = major / minor if minor > 0 else 1.0
        else:
            major, minor, ar = 0.0, 0.0, 1.0

        rows.append({
            "time": t,
            "hpf": HPF_START + t * HPF_INTERVAL,
            "bf_area": int(bf_mask.sum()),
            "aspect_ratio": ar,
            "major_axis": major,
            "Q_nematic": Q,
            "dominant_angle_deg": dom_angle,
            "mean_coherence": float(coh[bf_mask].mean()),
            "cortex_coherence": cort_coh,
            "interior_coherence": int_coh,
            "cortex_minus_interior": cort_coh - int_coh,
            "mean_energy": float(energy[bf_mask].mean()),
        })
    return pd.DataFrame(rows), np.array(radial_profiles)


def save_orientation_preview(stack, masks_aligned, t, out_path, title):
    bf_raw = z_project_bf(stack[t, :, 2]).astype(float)
    lyn_raw = np.max(stack[t, :, 1], axis=0).astype(float)
    bf_n = norm_pct(bf_raw); lyn_n = norm_pct(lyn_raw)
    bf_mask = masks_aligned[t] if t < len(masks_aligned) else np.zeros_like(bf_n, bool)
    orient, coh, _ = structure_tensor_fields(lyn_n)
    Q, dom = nematic_order(orient, coh, bf_mask)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(bf_n, cmap="gray")
    if bf_mask.any():
        for c in measure.find_contours(bf_mask.astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "r-", lw=1.5)
    axes[0].set_title("BF + mask"); axes[0].axis("off")

    axes[1].imshow(lyn_n, cmap="hot")
    if bf_mask.any():
        for c in measure.find_contours(bf_mask.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], "w-", lw=1)
    axes[1].set_title("LynTom (max-Z)"); axes[1].axis("off")

    coh_disp = np.where(bf_mask, coh, np.nan)
    im2 = axes[2].imshow(coh_disp, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title(f"Coherence (mean={coh[bf_mask].mean() if bf_mask.any() else 0:.2f})")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    axes[3].imshow(lyn_n, cmap="gray")
    ys, xs = np.mgrid[0:lyn_n.shape[0]:18, 0:lyn_n.shape[1]:18]
    in_mask = bf_mask[ys, xs]
    ys_in, xs_in = ys[in_mask], xs[in_mask]
    o = orient[ys_in, xs_in]; c = coh[ys_in, xs_in]
    dy = np.sin(o + np.pi / 2) * c * 8
    dx = np.cos(o + np.pi / 2) * c * 8
    axes[3].quiver(xs_in, ys_in, dx, -dy, c, cmap="inferno", clim=(0, 1),
                   scale=180, headwidth=2, headlength=3, pivot="middle", alpha=0.95)
    if bf_mask.any():
        for c2 in measure.find_contours(bf_mask.astype(float), 0.5):
            axes[3].plot(c2[:, 1], c2[:, 0], "w-", lw=1.5)
    axes[3].set_title(f"Orientation (Q={Q:.2f}, axis={dom:+.0f}deg)"); axes[3].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def pchip_smooth(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, ys = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(xs) < 4:
        return xs, ys
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) < 4:
        return xs, ys
    interp = PchipInterpolator(xs, ys)
    xs_dense = np.linspace(xs.min(), xs.max(), 100)
    return xs_dense, interp(xs_dense)


def bootstrap_ci(values, n_boot=500):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                     for _ in range(n_boot)])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_timepoint_ci(df, ycol):
    xs, means, los, his = [], [], [], []
    for x, group in df.groupby("hpf"):
        m, lo, hi = bootstrap_ci(group[ycol].values)
        xs.append(x); means.append(m); los.append(lo); his.append(hi)
    return np.array(xs), np.array(means), np.array(los), np.array(his)


def plot_trajectory(all_df, ycol, ylabel, title, fname, clip_zero=False):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for cond in CONDITIONS:
        sub = all_df[all_df["condition"] == cond]
        if sub.empty:
            continue
        xs, means, los, his = per_timepoint_ci(sub, ycol)
        if clip_zero:
            means = np.clip(means, 0, None)
            los = np.clip(los, 0, None)
            his = np.clip(his, 0, None)
        ax.fill_between(xs, los, his, color=COLORS[cond], alpha=0.15)
        xs_s, means_s = pchip_smooth(xs, means)
        if clip_zero:
            means_s = np.clip(means_s, 0, None)
        ax.plot(xs_s, means_s, color=COLORS[cond], lw=2.5, label=LABELS[cond])
    ax.set_xlabel("Time [hpf]"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(str(FIGS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def plot_radial_profile(all_radial, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    radii = np.linspace(0, 1, 10)
    for cond in CONDITIONS:
        if all_radial.get(cond) is None or all_radial[cond].size == 0:
            continue
        late = all_radial[cond][:, -5:, :]
        m = np.nanmean(late, axis=(0, 1))
        s = np.nanstd(late, axis=(0, 1))
        ax.plot(radii, m, color=COLORS[cond], lw=2.5, marker="o", label=LABELS[cond])
        ax.fill_between(radii, m - s, m + s, color=COLORS[cond], alpha=0.2)
    ax.set_xlabel("Normalised radius (0=center, 1=edge)")
    ax.set_ylabel("Mean coherence (anisotropy)")
    ax.set_title("Radial profile of cell anisotropy (last 5 frames)", fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(str(FIGS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def main():
    print("=" * 60)
    print("  LynTom Tissue Tension (structure tensor)")
    print("=" * 60)

    all_rows = []
    all_radial = {cond: [] for cond in CONDITIONS}

    for cond in CONDITIONS:
        data_dir = DATA_ROOT / cond
        if not data_dir.exists():
            continue
        tifs = sorted(data_dir.glob("*.tif"))
        preview_indices = list(range(min(4, len(tifs))))

        for i, sample_path in enumerate(tifs):
            pid_match = re.search(r"(G\d+)", sample_path.stem)
            pid = pid_match.group(1) if pid_match else sample_path.stem
            mask_path = ANALYSIS_DIR / cond / sample_path.stem / "masks" / f"{sample_path.stem}_masks.tif"
            if not mask_path.exists():
                print(f"  Missing mask for {cond}/{pid}, skipping")
                continue

            stack = tifffile.imread(str(sample_path))
            masks = tifffile.imread(str(mask_path)) > 0
            df, radial = process_sample(stack, masks)
            df["condition"] = cond
            df["pescoid"] = pid
            df.to_csv(str(OUT / f"{cond}_{pid}_tension.csv"), index=False)
            all_rows.append(df)
            if radial.size > 0:
                all_radial[cond].append(radial)

            if i in preview_indices:
                t_mid = min(15, stack.shape[0] - 1)
                save_orientation_preview(
                    stack, masks, t_mid,
                    PREVIEWS / f"{cond}_{pid}_t{t_mid}.png",
                    f"{LABELS[cond]} / {pid} / t={t_mid}",
                )

            print(f"  {cond}/{pid}: T={len(df)}, meanQ={df['Q_nematic'].mean():.2f}, final_coh={df['mean_coherence'].iloc[-1]:.2f}")

    if not all_rows:
        print("No data!")
        return

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(str(OUT / "all_tension_data.csv"), index=False)

    for cond in CONDITIONS:
        if all_radial[cond]:
            T_max = max(r.shape[0] for r in all_radial[cond])
            padded = []
            for r in all_radial[cond]:
                if r.shape[0] < T_max:
                    pad = np.full((T_max - r.shape[0], r.shape[1]), np.nan)
                    r = np.vstack([r, pad])
                padded.append(r)
            all_radial[cond] = np.array(padded)
        else:
            all_radial[cond] = np.zeros((0, 0, 0))

    print("\nGenerating trajectory plots...")
    plot_trajectory(big, "Q_nematic", "Nematic order Q (0=isotropic, 1=aligned)",
                    "Tissue alignment over time", "01_Q_nematic.png", clip_zero=True)
    plot_trajectory(big, "mean_coherence", "Mean coherence",
                    "Mean local coherence over time", "02_coherence.png", clip_zero=True)
    plot_trajectory(big, "cortex_coherence", "Cortical coherence",
                    "Cortical anisotropy over time", "03_cortex_coherence.png", clip_zero=True)
    plot_trajectory(big, "interior_coherence", "Interior coherence",
                    "Interior anisotropy over time", "04_interior_coherence.png", clip_zero=True)
    plot_trajectory(big, "cortex_minus_interior", "Cortex - Interior",
                    "Cortical excess anisotropy", "05_cortex_minus_interior.png")
    plot_trajectory(big, "aspect_ratio", "Aspect ratio",
                    "Pescoid aspect ratio over time", "06_aspect_ratio.png", clip_zero=True)
    plot_trajectory(big, "mean_energy", "Membrane signal strength (energy)",
                    "Membrane tensor energy over time", "07_energy.png", clip_zero=True)
    plot_radial_profile(all_radial, "08_radial_profile_late.png")

    print(f"\nDone. Outputs in: {OUT}/")
    print(f"  Plots: {FIGS}/")
    print(f"  Previews: {PREVIEWS}/")


if __name__ == "__main__":
    main()
