"""
Section 4 GALLERY - one figure with two transgenic lines (LynTom + H2A),
control row + Activin row each, four timestamps per row.

Layout:
  rows = 4    (LynTom ctrl, LynTom Activin, H2A ctrl, H2A Activin)
  cols = 4    (7 hpf, 10 hpf, 12 hpf, 15 hpf)

Channels:
  BF (gray) + mezzo:GFP (green) + mCherry-channel
    LynTom -> orange-red (1.0, 0.4, 0.0)
    H2A    -> pure red   (1.0, 0.0, 0.0)

Output:
  Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase5_morphometrics\\poster\\section4\\section4_gallery_4rows.png

Run after this to add the myosin (260514) row once Phase 1 is done.
"""
from pathlib import Path
import json
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LYN_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
H2A_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\section4")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_HPF = [7.0, 10.0, 12.0, 15.0]

# Display tuning
BF_LO, BF_HI = 1.0, 99.7
BF_WEIGHT = 0.55
GFP_LO, GFP_HI = 55, 99.5
GFP_GAIN = 1.40
RED_LO, RED_HI = 12, 99.0
RED_GAIN = 1.15

GFP_TINT = (0.00, 1.00, 0.00)
LYN_TINT = (1.00, 0.40, 0.00)   # orange-red mCherry (LynTom)
H2A_TINT = (1.00, 0.00, 0.00)   # pure red mCherry  (H2A)

UM_PER_PX = 1.5
SCALE_BAR_UM = 200

# Per-row configuration: (dataset_root, cond, pid, row_label, red_channel_filename, red_tint)
LYNTOM_TIMING = {"hpf_start": 6.0, "hpf_interval": 0.4}
H2A_TIMING = {"hpf_start": 7.0, "hpf_interval": 0.1941}

ROWS = [
    {"root": LYN_PHASE1, "cond": "P_ctrl",           "pid": "G030",
     "label": "LynTom\nP ctrl",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYNTOM_TIMING},
    {"root": LYN_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "G047",
     "label": "LynTom\nP Activin",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYNTOM_TIMING},
    {"root": H2A_PHASE1, "cond": "P_ctrl",           "pid": "S11",
     "label": "H2A\nP ctrl",
     "red_file": "h2a_aligned.tif", "red_tint": H2A_TINT,
     "timing": H2A_TIMING},
    {"root": H2A_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "S27",
     "label": "H2A\nP Activin",
     "red_file": "h2a_aligned.tif", "red_tint": H2A_TINT,
     "timing": H2A_TIMING},
]


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


def render_row(row):
    pdir = row["root"] / "per_pescoid" / row["cond"] / row["pid"]
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    red = tifffile.imread(str(pdir / row["red_file"]))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    union = mask.any(axis=0)
    panels = []
    for h in TARGET_HPF:
        t = int(round((h - row["timing"]["hpf_start"]) /
                        row["timing"]["hpf_interval"]))
        t = max(0, min(T - 1, t))
        rgb = merged_rgb(bf[t], gfp[t], red[t], mask[t], row["red_tint"])
        panels.append(rgb)
    panels = np.stack(panels)
    panels = crop_union(panels, union, pad=18)
    return panels


def main():
    print("Rendering 4 rows...")
    rows_rendered = []
    for row in ROWS:
        try:
            panels = render_row(row)
            rows_rendered.append((row, panels))
            print(f"  {row['label'].replace(chr(10), ' / ')}  ({row['pid']})  "
                  f"panels {panels.shape[1]}x{panels.shape[2]}")
        except Exception as e:
            print(f"  FAIL {row['pid']}: {e}")

    n_rows = len(rows_rendered)
    n_cols = len(TARGET_HPF)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows),
                              facecolor="black", squeeze=False)
    for ri, (row, panels) in enumerate(rows_rendered):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.imshow(panels[ci])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(f"{TARGET_HPF[ci]:.0f} hpf", fontsize=22,
                             fontweight="bold", color="white")
            if ci == 0:
                ax.set_ylabel(row["label"] + f"\n({row['pid']})",
                              fontsize=16, fontweight="bold",
                              color="white", rotation=0, ha="right",
                              va="center", labelpad=80)
        # scale bar on the rightmost panel of each row
        last_panel = panels[-1]
        H, W = last_panel.shape[:2]
        bar_len_px = SCALE_BAR_UM / UM_PER_PX
        bar_h = max(3, int(H * 0.012))
        rect_y = H - int(H * 0.06) - bar_h
        rect_x = W - int(W * 0.07) - int(bar_len_px)
        axes[ri, -1].add_patch(Rectangle((rect_x, rect_y), bar_len_px, bar_h,
                                          color="white", lw=0))
        if ri == n_rows - 1:
            axes[ri, -1].text(rect_x + bar_len_px / 2, rect_y - 6,
                               f"{SCALE_BAR_UM} um",
                               ha="center", va="bottom", color="white",
                               fontsize=14, fontweight="bold")

    # legend at the top
    lyn_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in LYN_TINT)
    h2a_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in H2A_TINT)
    handles = [
        plt.Line2D([0], [0], color="white", lw=10, label="brightfield"),
        plt.Line2D([0], [0], color="#00FF00", lw=10, label="mezzo:GFP"),
        plt.Line2D([0], [0], color=lyn_hex, lw=10,
                   label="LynTom:mCherry (membrane)"),
        plt.Line2D([0], [0], color=h2a_hex, lw=10,
                   label="H2A:mCherry (nuclei)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
                bbox_to_anchor=(0.5, 1.00), frameon=False,
                fontsize=16, labelcolor="white")
    fig.suptitle("Section 4 - Transgenic phenotypes: control vs Activin 3-5h\n"
                 "Same chemistry, discrete outcomes - one organiser (mono) "
                 "or two (bipolar) along AP axis",
                 fontsize=20, fontweight="bold", color="white", y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(left=0.08, wspace=0.04, hspace=0.04)
    out = OUT / "section4_gallery_4rows.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
