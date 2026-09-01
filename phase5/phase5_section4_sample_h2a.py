"""
Section 4 SAMPLE - H2A dataset variant.

Sample: S27 (P_Activin, 260512 H2A dataset, mezzo:GFP x H2A:mCherry)
Channels: BF (gray) + mezzo:GFP (green) + H2A:mCherry (red, distinct from LynTom)
Timestamps: 7, 10, 12, 15 hpf

H2A dataset metadata (from per-pescoid metrics.json):
  hpf_start = 7.0, hpf_interval = 0.1941 (~11.6 min/frame), n_frames = 93

Channels per pescoid: bf_aligned.tif, gfp_aligned.tif, h2a_aligned.tif, mask_aligned.tif
"""
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\section4")
OUT.mkdir(parents=True, exist_ok=True)

HPF_START = 7.0
HPF_INTERVAL = 0.1941
TARGET_HPF = [7.0, 10.0, 12.0, 15.0]

# Same tuning as LynTom sample
BF_LO, BF_HI = 1.0, 99.7
BF_WEIGHT = 0.55
GFP_LO, GFP_HI = 55, 99.5
GFP_GAIN = 1.40
RED_LO, RED_HI = 12, 99.0
RED_GAIN = 1.15

GFP_TINT = (0.00, 1.00, 0.00)   # green
H2A_TINT = (1.00, 0.00, 0.00)   # PURE RED (H2A:mCherry)

# Try S27 first (per user message earlier - one of the good treated pescoids)
SAMPLE = ("P_Activin_3-5hpf", "S27", "P_Activin (H2A)")
RED_LABEL = "H2A:mCherry (nuclei)"

UM_PER_PX = 1.5
SCALE_BAR_UM = 100


def norm01_mask(img, lo, hi, mask):
    if mask is None or not mask.any():
        vals = img.ravel()
    else:
        vals = img[mask]
    p_lo, p_hi = np.percentile(vals, lo), np.percentile(vals, hi)
    return np.clip((img.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1.0),
                    0, 1)


def merged_rgb(bf, gfp, red, mask, red_tint):
    bf_n = norm01_mask(bf, BF_LO, BF_HI, None)
    if (~mask).any():
        gfp_bg = float(gfp[~mask].mean())
        red_bg = float(red[~mask].mean())
    else:
        gfp_bg = red_bg = 0.0
    gfp_sub = np.clip(gfp.astype(np.float32) - gfp_bg, 0, None)
    red_sub = np.clip(red.astype(np.float32) - red_bg, 0, None)
    gfp_n = np.clip(norm01_mask(gfp_sub, GFP_LO, GFP_HI, mask) * GFP_GAIN, 0, 1)
    red_n = np.clip(norm01_mask(red_sub, RED_LO, RED_HI, mask) * RED_GAIN, 0, 1)
    rgb = np.zeros(bf.shape + (3,), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = bf_n * BF_WEIGHT
    for c, t in enumerate(GFP_TINT):
        if t > 0:
            rgb[..., c] = np.where(mask, rgb[..., c] + gfp_n * t, rgb[..., c])
    for c, t in enumerate(red_tint):
        if t > 0:
            rgb[..., c] = np.where(mask, rgb[..., c] + red_n * t, rgb[..., c])
    return np.clip(rgb, 0, 1)


def crop_union(rgb_stack, mask_union, pad=18):
    ys, xs = np.where(mask_union)
    if ys.size == 0:
        return rgb_stack
    y0 = max(0, ys.min() - pad)
    y1 = min(rgb_stack.shape[-3], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(rgb_stack.shape[-2], xs.max() + pad + 1)
    return rgb_stack[..., y0:y1, x0:x1, :]


def main():
    cond, pid, label = SAMPLE
    pdir = PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    red = tifffile.imread(str(pdir / "h2a_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    union = mask.any(axis=0)
    print(f"{pid} ({label}): T={T}, mask union shape={mask.shape[-2:]}")
    panels = []
    for h in TARGET_HPF:
        t = int(round((h - HPF_START) / HPF_INTERVAL))
        t = max(0, min(T - 1, t))
        actual_hpf = HPF_START + t * HPF_INTERVAL
        rgb = merged_rgb(bf[t], gfp[t], red[t], mask[t], H2A_TINT)
        panels.append(rgb)
        print(f"  target {h:.1f} hpf -> frame {t} (actual {actual_hpf:.1f} hpf)")
    panels = np.stack(panels)
    panels = crop_union(panels, union, pad=18)
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.6), facecolor="black")
    for i, ax in enumerate(axes):
        ax.imshow(panels[i])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.04, 0.96, f"{TARGET_HPF[i]:.0f} hpf",
                transform=ax.transAxes, fontsize=22, fontweight="bold",
                color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.25",
                           fc="#111111", ec="none", alpha=0.7))
    last_panel = panels[-1]
    H, W = last_panel.shape[:2]
    bar_len_px = SCALE_BAR_UM / UM_PER_PX
    bar_h = max(4, int(H * 0.012))
    rect_y = H - int(H * 0.06) - bar_h
    rect_x = W - int(W * 0.07) - int(bar_len_px)
    axes[-1].add_patch(Rectangle((rect_x, rect_y), bar_len_px, bar_h,
                                   color="white", lw=0))
    axes[-1].text(rect_x + bar_len_px / 2, rect_y - 6, f"{SCALE_BAR_UM} um",
                  ha="center", va="bottom", color="white", fontsize=14,
                  fontweight="bold")
    h2a_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in H2A_TINT)
    handles = [
        plt.Line2D([0], [0], color="white", lw=8, label="brightfield"),
        plt.Line2D([0], [0], color="#00FF00", lw=8, label="mezzo:GFP"),
        plt.Line2D([0], [0], color=h2a_hex, lw=8, label=RED_LABEL),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 1.005), frameon=False,
                fontsize=16, labelcolor="white")
    fig.suptitle(f"Section 4 SAMPLE  -  {pid}  ({label})",
                  fontsize=18, fontweight="bold", color="white", y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / f"section4_sample_{pid}_H2A.png"
    plt.savefig(str(out), dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
