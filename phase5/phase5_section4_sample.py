"""
Section 4 SAMPLE - one pescoid, 4 timestamps, merged channels at conference quality.

Sample: G047 (P_Activin coord_bipolar, LynTom dataset)
Channels: BF (gray, high-contrast) + mezzo:GFP (green) + LynTom:mCherry (red)
Timestamps: 7, 10, 12, 15 hpf  -> frames {2 or 3, 10, 15, 23}

Goal: validate the look before generating the full 4-row gallery.

Output:
  Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase5_morphometrics\\poster\\section4\\section4_sample_G047.png

Quality upgrades over the previous montage:
  - BF gets a tighter percentile stretch and higher weight (clearer texture)
  - GFP / LynTom mask-restricted bg-subtract with adjustable per-channel gain
  - Each panel has a hpf label
  - Optional 100-um scale bar (1 pixel ~= 1.5 um for 10x objective on this rig)
"""
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\section4")
OUT.mkdir(parents=True, exist_ok=True)

HPF_START = 6.0
HPF_INTERVAL = 0.4
TARGET_HPF = [7.0, 10.0, 12.0, 15.0]

# Per-channel tuning knobs
BF_LO, BF_HI = 1.0, 99.7     # percentile stretch for BF (wider -> brighter highlights)
BF_WEIGHT = 0.55             # how strong BF is in the composite base
GFP_LO, GFP_HI = 55, 99.5    # mezzo:GFP percentile stretch inside mask
GFP_GAIN = 1.40
LYN_LO, LYN_HI = 12, 99.0    # LynTom percentile stretch inside mask
LYN_GAIN = 1.15

# Channel TINTS (R, G, B in 0-1). Pure mezzo:GFP green; LynTom in ORANGE-RED
# to distinguish it from H2A:mCherry (which gets pure RED in the other rows).
GFP_TINT = (0.00, 1.00, 0.00)   # green
LYN_TINT = (1.00, 0.40, 0.00)   # orange-red (mCherry, LynTom)
H2A_TINT = (1.00, 0.00, 0.00)   # pure red (mCherry, H2A)

SAMPLE = ("P_Activin_3-5hpf", "G047", "P_Activin coord_bipolar")
SAMPLE_RED_CHANNEL = "LynTom"

UM_PER_PX = 1.5              # 10x objective approx scale (adjust if needed)
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
    """Build a brightfield + GFP-green + (red-channel)-tinted composite.
    `red_tint` is an (R, G, B) triple in [0,1] used to colourise the
    second fluorescent channel (LynTom or H2A or myosin)."""
    bf_n = norm01_mask(bf, BF_LO, BF_HI, None)
    if (~mask).any():
        gfp_bg = float(gfp[~mask].mean())
        red_bg = float(red[~mask].mean())
    else:
        gfp_bg = red_bg = 0.0
    gfp_sub = np.clip(gfp.astype(np.float32) - gfp_bg, 0, None)
    red_sub = np.clip(red.astype(np.float32) - red_bg, 0, None)
    gfp_n = np.clip(norm01_mask(gfp_sub, GFP_LO, GFP_HI, mask) * GFP_GAIN, 0, 1)
    red_n = np.clip(norm01_mask(red_sub, LYN_LO, LYN_HI, mask) * LYN_GAIN, 0, 1)

    rgb = np.zeros(bf.shape + (3,), dtype=np.float32)
    # BF in gray everywhere as the base layer
    rgb[..., 0] = bf_n * BF_WEIGHT
    rgb[..., 1] = bf_n * BF_WEIGHT
    rgb[..., 2] = bf_n * BF_WEIGHT
    # Additive overlay of GFP (green) only inside mask
    for c, t in enumerate(GFP_TINT):
        if t > 0:
            rgb[..., c] = np.where(mask, rgb[..., c] + gfp_n * t,
                                     rgb[..., c])
    # Additive overlay of mCherry-channel only inside mask
    for c, t in enumerate(red_tint):
        if t > 0:
            rgb[..., c] = np.where(mask, rgb[..., c] + red_n * t,
                                     rgb[..., c])
    return np.clip(rgb, 0, 1)


def crop_union(rgb_stack, mask_union, pad=12):
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
    lyn = tifffile.imread(str(pdir / "lyntom_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    union = mask.any(axis=0)
    print(f"{pid} ({label}): T={T}, mask union shape={mask.shape[-2:]}")

    frames = []
    panels = []
    red_tint = LYN_TINT
    for h in TARGET_HPF:
        t = int(round((h - HPF_START) / HPF_INTERVAL))
        t = max(0, min(T - 1, t))
        actual_hpf = HPF_START + t * HPF_INTERVAL
        rgb = merged_rgb(bf[t], gfp[t], lyn[t], mask[t], red_tint)
        frames.append((t, actual_hpf))
        panels.append(rgb)
        print(f"  target {h:.1f} hpf -> frame {t} (actual {actual_hpf:.1f} hpf)")
    panels = np.stack(panels)
    panels = crop_union(panels, union, pad=18)
    n = len(panels)

    # Tall figure: 1 row x 4 cols. Wide panels.
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.6),
                              facecolor="black")
    for i, ax in enumerate(axes):
        ax.imshow(panels[i])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")
        # spine off
        for sp in ax.spines.values():
            sp.set_visible(False)
        # hpf label top-left of each panel
        ax.text(0.04, 0.96, f"{TARGET_HPF[i]:.0f} hpf",
                transform=ax.transAxes, fontsize=22, fontweight="bold",
                color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.25",
                           fc="#111111", ec="none", alpha=0.7))
    # scale bar on the LAST panel (bottom-right)
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
    # legend across the top
    lyn_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in LYN_TINT)
    handles = [
        plt.Line2D([0], [0], color="white", lw=8, label="brightfield"),
        plt.Line2D([0], [0], color="#00FF00", lw=8, label="mezzo:GFP"),
        plt.Line2D([0], [0], color=lyn_hex, lw=8,
                   label="LynTom:mCherry (membrane)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 1.005), frameon=False,
                fontsize=16, labelcolor="white")
    fig.suptitle(f"Section 4 SAMPLE  -  {pid}  ({label})",
                  fontsize=18, fontweight="bold", color="white", y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / f"section4_sample_{pid}.png"
    plt.savefig(str(out), dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
