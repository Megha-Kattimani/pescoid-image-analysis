"""
Phase 5a - H2A nuclear dynamics (Tier 1).
mezzo + H2A dataset (260512_mezzo_H2A_3-5hpfActivin, Zeiss 10x).

For each pescoid:
  1. Nuclear segmentation per frame (Cellpose 'cpsam' on DoG-enhanced H2A).
     Reliable in the post-12 hpf window; early frames may show 0 (signal too
     weak); reported regardless.
  2. Density field per frame (Gaussian-smoothed H2A intensity inside the mask)
     - works at every frame regardless of nucleus visibility.
  3. PIV flow field between consecutive H2A frames (openpiv,
     extended search area cross-correlation).
  4. Inside-vs-outside mezzo+ pole region: density and flow magnitude.

Uses Phase 1 aligned tifs + Phase 2 mezzo_tracks.tif (pole-classified IDs).

Sanity:
    python phase5a_h2a_dynamics.py --pids S36 S21

Full:
    python phase5a_h2a_dynamics.py --all
"""

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage as ndi
from skimage import measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cellpose import models as cp_models
from openpiv import pyprocess as piv

# ---------------------------------------------------------------------------
PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5a")
PER = OUT / "per_pescoid"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PER, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 7.0
HPF_INTERVAL = 698.8316040039062 / 3600.0   # ~0.1941 hpf
ANALYSIS_MAX_HPF = 16.0
PEAK_HPF = (7.0, 16.0)
NUCLEI_CONFIDENT_HPF_MIN = 12.0   # report-but-confidence after this hpf

# Cellpose / DoG preprocessing
DOG_SIGMA_SMALL = 1.0
DOG_SIGMA_LARGE = 6.0
CELLPOSE_DIAMETER = 12
CELLPOSE_FLOW_THR = 0.5
CELLPOSE_CELLPROB_THR = -2.0

# Density: Gaussian-smoothed H2A intensity (within mask) as nuclear density proxy
DENSITY_SIGMA_PX = 15.0

# PIV (openpiv)
PIV_WINDOW = 24
PIV_OVERLAP = 12
PIV_SEARCH_AREA = 28

plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})


# ===========================================================================
# Helpers
# ===========================================================================
def norm01(x):
    lo, hi = np.percentile(x, (1, 99.5))
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def dog_h2a(img, s_lo=DOG_SIGMA_SMALL, s_hi=DOG_SIGMA_LARGE):
    """Difference of Gaussians: enhances punctate nuclear signal."""
    f = img.astype(np.float32)
    d = ndi.gaussian_filter(f, sigma=s_lo) - ndi.gaussian_filter(f, sigma=s_hi)
    return norm01(d.clip(min=0))


def load_aligned(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    h2a = tifffile.imread(str(pdir / "h2a_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    return h2a, gfp, mask


def load_pole_stack(cond, pid):
    """Bool (T, Y, X): True where mezzo+ pole-classified tracks are."""
    csv_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
    tif_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.tif"
    if not (csv_p.exists() and tif_p.exists()):
        return None
    stack = tifffile.imread(str(tif_p))
    try:
        df = pd.read_csv(str(csv_p))
    except pd.errors.EmptyDataError:
        return np.zeros_like(stack, dtype=bool)
    if df.empty or "cluster_type" not in df.columns:
        return np.zeros_like(stack, dtype=bool)
    pole_ids = df[df["cluster_type"] == "pole"]["track_id"].tolist()
    if not pole_ids:
        return np.zeros_like(stack, dtype=bool)
    return np.isin(stack, pole_ids)


def load_summary():
    return pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))


# ===========================================================================
# Per-frame operations
# ===========================================================================
def segment_nuclei(dog_img, model):
    """Run Cellpose 'cpsam' on a DoG-enhanced H2A frame. Returns (labeled, n)."""
    m, _, _ = model.eval(
        dog_img,
        diameter=CELLPOSE_DIAMETER,
        flow_threshold=CELLPOSE_FLOW_THR,
        cellprob_threshold=CELLPOSE_CELLPROB_THR,
        normalize=False,   # we already DoG+normed
    )
    n = int(m.max())
    return m, n


def density_field(h2a_frame, mask, sigma=DENSITY_SIGMA_PX):
    """Gaussian-smoothed H2A intensity within the mask. Returns float density."""
    img = h2a_frame.astype(np.float32) * mask
    return ndi.gaussian_filter(img, sigma=sigma) * mask


def piv_flow(img1, img2):
    """Cross-correlation flow on consecutive H2A frames. Returns (u, v, x, y)."""
    a = norm01(img1).astype(np.float32)
    b = norm01(img2).astype(np.float32)
    u, v, _ = piv.extended_search_area_piv(
        a, b,
        window_size=PIV_WINDOW,
        overlap=PIV_OVERLAP,
        search_area_size=PIV_SEARCH_AREA,
        sig2noise_method="peak2peak",
    )
    x, y = piv.get_coordinates(
        image_size=a.shape,
        search_area_size=PIV_SEARCH_AREA,
        overlap=PIV_OVERLAP,
    )
    # openpiv returns v positive downward; here y increases downward so leave as is
    return u, v, x, y


def grid_from_mask(x, y, mask):
    """Bool array over the PIV grid indicating which grid points fall inside the mask."""
    yi = np.clip(np.round(y).astype(int), 0, mask.shape[0] - 1)
    xi = np.clip(np.round(x).astype(int), 0, mask.shape[1] - 1)
    return mask[yi, xi]


# ===========================================================================
# Per-pescoid processing
# ===========================================================================
def process_pescoid(cond, pid, cp_model, save_outputs=True, max_frames=None):
    print(f"  [{cond}/{pid}]")
    h2a, gfp, mask = load_aligned(cond, pid)
    pole_stack = load_pole_stack(cond, pid)
    if pole_stack is None:
        pole_stack = np.zeros_like(mask, dtype=bool)

    T = h2a.shape[0]
    cap = int(np.floor((ANALYSIS_MAX_HPF - HPF_START) / HPF_INTERVAL)) + 1
    T_eff = min(T, cap)
    if max_frames:
        T_eff = min(T_eff, max_frames)

    rows = []
    nuclei_masks = np.zeros((T_eff, mask.shape[1], mask.shape[2]), dtype=np.uint16)
    density_stack = np.zeros((T_eff, mask.shape[1], mask.shape[2]), dtype=np.float32)
    # PIV vector field: store per-pair on a fixed grid
    piv_u_list, piv_v_list = [], []
    piv_x = piv_y = None

    for t in range(T_eff):
        if not mask[t].any():
            rows.append({"time": t, "hpf": HPF_START + t * HPF_INTERVAL,
                         "n_nuclei": 0, "density_inside_pole": np.nan,
                         "density_outside_pole": np.nan,
                         "flow_mag_inside_pole": np.nan, "flow_mag_outside_pole": np.nan,
                         "total_h2a_intensity": 0.0, "pole_pixels": 0,
                         "non_pole_pixels": 0})
            continue

        # 1. Nuclei
        dog = dog_h2a(h2a[t])
        lab, n_nuc = segment_nuclei(dog, cp_model)
        # restrict to inside mask
        lab = lab * mask[t]
        nuclei_masks[t] = lab.astype(np.uint16)
        # 2. Density
        dens = density_field(h2a[t], mask[t])
        density_stack[t] = dens
        # 3. Pole region
        pole = pole_stack[t] if t < len(pole_stack) else np.zeros_like(mask[t])
        pole_in = pole & mask[t]
        non_pole_in = mask[t] & ~pole_in
        # density inside/outside
        d_in = float(dens[pole_in].mean()) if pole_in.any() else np.nan
        d_out = float(dens[non_pole_in].mean()) if non_pole_in.any() else np.nan
        # 4. Flow magnitude (between t and t+1)
        if t < T_eff - 1 and mask[t + 1].any():
            u, v, gx, gy = piv_flow(h2a[t], h2a[t + 1])
            piv_u_list.append(u); piv_v_list.append(v)
            if piv_x is None:
                piv_x, piv_y = gx, gy
            mag = np.hypot(u, v)
            # which grid points are inside pole vs outside
            in_pole_g = grid_from_mask(gx, gy, pole_in)
            in_non_g = grid_from_mask(gx, gy, non_pole_in)
            fm_in = float(mag[in_pole_g].mean()) if in_pole_g.any() else np.nan
            fm_out = float(mag[in_non_g].mean()) if in_non_g.any() else np.nan
        else:
            fm_in = np.nan
            fm_out = np.nan

        total_h2a = float(h2a[t][mask[t]].sum())
        rows.append({
            "time": t, "hpf": HPF_START + t * HPF_INTERVAL,
            "n_nuclei": n_nuc,
            "density_inside_pole": d_in,
            "density_outside_pole": d_out,
            "flow_mag_inside_pole": fm_in,
            "flow_mag_outside_pole": fm_out,
            "total_h2a_intensity": total_h2a,
            "pole_pixels": int(pole_in.sum()),
            "non_pole_pixels": int(non_pole_in.sum()),
        })

    df = pd.DataFrame(rows)

    # Peak frame: pick the frame in the peak window with max pole area
    peak_t = None
    pole_areas = df["pole_pixels"].values if "pole_pixels" in df else np.zeros(T_eff)
    in_peak = [t for t in range(T_eff)
               if PEAK_HPF[0] <= HPF_START + t * HPF_INTERVAL <= PEAK_HPF[1] and pole_areas[t] > 0]
    if in_peak:
        peak_t = max(in_peak, key=lambda t: pole_areas[t])
    elif T_eff > 0:
        peak_t = T_eff - 1

    if save_outputs:
        pdir = PER / cond / pid
        pdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(pdir / "h2a_per_frame.csv"), index=False)
        tifffile.imwrite(str(pdir / "nuclei_masks.tif"), nuclei_masks)
        np.save(str(pdir / "density_stack.npy"), density_stack)
        if piv_u_list:
            np.save(str(pdir / "piv_u.npy"), np.array(piv_u_list))
            np.save(str(pdir / "piv_v.npy"), np.array(piv_v_list))
            np.save(str(pdir / "piv_grid_x.npy"), piv_x)
            np.save(str(pdir / "piv_grid_y.npy"), piv_y)
        # save 4-panel heatmap at peak
        if peak_t is not None:
            save_peak_heatmap(cond, pid, peak_t, h2a, gfp, mask, pole_stack,
                              nuclei_masks, density_stack,
                              piv_u_list, piv_v_list, piv_x, piv_y,
                              pdir / f"{pid}_5a_peak.png")
    return df, peak_t


# ===========================================================================
# Peak-frame 4-panel heatmap
# ===========================================================================
def save_peak_heatmap(cond, pid, t, h2a, gfp, mask, pole_stack,
                      nuclei_masks, density_stack,
                      piv_u_list, piv_v_list, piv_x, piv_y, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5))
    pole = pole_stack[t] & mask[t] if t < len(pole_stack) else np.zeros_like(mask[t])

    # Panel 1: H2A with nucleus centroids + mezzo pole outline
    axes[0].imshow(norm01(h2a[t]), cmap="magma")
    if mask[t].any():
        for c in measure.find_contours(mask[t].astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "w-", lw=0.8, alpha=0.6)
    if pole.any():
        for c in measure.find_contours(pole.astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "lime", lw=2)
    # nucleus centroids
    nm = nuclei_masks[t]
    n_nuc = int(nm.max())
    if n_nuc > 0:
        props = measure.regionprops(nm)
        for p in props:
            cy, cx = p.centroid
            axes[0].plot(cx, cy, "co", markersize=3, alpha=0.85)
    axes[0].set_title(f"H2A + {n_nuc} nuclei (cyan) + pole (lime)",
                      fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: density (Gaussian-smoothed H2A intensity)
    dens = density_stack[t]
    dens_disp = np.where(mask[t], dens, np.nan)
    im = axes[1].imshow(dens_disp, cmap="viridis")
    if pole.any():
        for c in measure.find_contours(pole.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[1].set_title("Density (smoothed H2A intensity)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.75, label="density")

    # Panel 3: flow vectors at this frame (between t and t+1)
    axes[2].imshow(norm01(h2a[t]), cmap="gray")
    if piv_u_list and t < len(piv_u_list) and piv_x is not None:
        u = piv_u_list[t]
        v = piv_v_list[t]
        in_mask = grid_from_mask(piv_x, piv_y, mask[t])
        if in_mask.any():
            axes[2].quiver(piv_x[in_mask], piv_y[in_mask],
                            u[in_mask], -v[in_mask],   # flip y for image coords
                            color="yellow", scale=80, headwidth=3,
                            headlength=4, pivot="middle", alpha=0.9)
        mag = np.hypot(u, v)
        mag_mean = float(mag[in_mask].mean()) if in_mask.any() else 0.0
    else:
        mag_mean = 0.0
    if mask[t].any():
        for c in measure.find_contours(mask[t].astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], "w-", lw=0.8, alpha=0.6)
    if pole.any():
        for c in measure.find_contours(pole.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[2].set_title(f"PIV flow (mean |v|={mag_mean:.1f} px/frame)",
                      fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: bar chart inside vs outside pole
    pole_in = pole & mask[t]
    non_pole_in = mask[t] & ~pole_in
    d_in = float(dens[pole_in].mean()) if pole_in.any() else 0
    d_out = float(dens[non_pole_in].mean()) if non_pole_in.any() else 0
    f_in = f_out = 0.0
    if piv_u_list and t < len(piv_u_list) and piv_x is not None:
        u = piv_u_list[t]; v = piv_v_list[t]
        mg = np.hypot(u, v)
        in_g = grid_from_mask(piv_x, piv_y, pole_in)
        out_g = grid_from_mask(piv_x, piv_y, non_pole_in)
        if in_g.any(): f_in = float(mg[in_g].mean())
        if out_g.any(): f_out = float(mg[out_g].mean())
    if pole_in.any():
        ax = axes[3]
        bar_w = 0.35
        ax.bar([0, 1], [d_in, d_out], width=bar_w, color=["lime", "#888"],
               edgecolor="black", lw=1, label="density")
        ax2 = ax.twinx()
        ax2.bar([0 + bar_w + 0.05, 1 + bar_w + 0.05], [f_in, f_out], width=bar_w,
                color=["#1f77b4", "#bbbbbb"], edgecolor="black", lw=1, label="flow |v|")
        ax.set_xticks([0 + bar_w / 2, 1 + bar_w / 2])
        ax.set_xticklabels(["Inside pole", "Outside"])
        ax.set_ylabel("Density (smoothed H2A)")
        ax2.set_ylabel("Mean |flow vector| px/frame")
        ax.set_title(f"Inside vs outside mezzo+ pole\n"
                      f"density: in={d_in:.0f} out={d_out:.0f}  |  "
                      f"flow: in={f_in:.1f} out={f_out:.1f}",
                      fontsize=10, fontweight="bold")
    else:
        axes[3].text(0.5, 0.5, "No mezzo+ pole at peak", ha="center", va="center",
                      fontsize=12)
        axes[3].axis("off")

    plt.suptitle(f"{cond} / {pid}   -   peak t={t}  ({HPF_START + t * HPF_INTERVAL:.1f} hpf)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def run(pescoid_filter=None):
    print("=" * 60)
    print("  Phase 5a - H2A nuclear dynamics")
    print("=" * 60)
    summary = load_summary()
    summary = summary[summary["condition"].isin(CONDITIONS)].copy()
    if pescoid_filter:
        summary = summary[summary["pescoid"].isin(pescoid_filter)]
    print(f"\nPescoids to process: {len(summary)}\n")

    print("Loading Cellpose 'cpsam'...")
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available())
    print(f"  GPU: {torch.cuda.is_available()}\n")

    all_rows = []
    for _, row in summary.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        try:
            df, peak_t = process_pescoid(cond, pid, cp_model)
            df["condition"] = cond
            df["pescoid"] = pid
            all_rows.append(df)
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")
            import traceback; traceback.print_exc()

    if not all_rows:
        print("No pescoids processed.")
        return
    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(str(TABLES / "h2a_per_frame_all.csv"), index=False)
    print(f"\nSaved combined per-frame CSV ({len(big)} rows)")
    print(f"All outputs in: {OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pids", nargs="*", help="Pescoid IDs (e.g. S36 S21)")
    p.add_argument("--all", action="store_true", help="Run on all P pescoids")
    args = p.parse_args()
    if args.pids:
        run(pescoid_filter=args.pids)
    elif args.all:
        run()
    else:
        p.print_help()
