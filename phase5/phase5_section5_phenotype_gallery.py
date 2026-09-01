"""
Section 5 phenotype gallery v3 - mixed datasets (LynTom + H2A),
3 on top / 3 on bottom, no annotations.

Slots (top-left -> bottom-right):
  1. G046  LynTom   monopolar              (12 hpf)
  2. G039  LynTom   bipolar / multi best   (12 hpf)
  3. G040  LynTom   bipolar / multi        (12 hpf)
  4. S24   H2A      bipolar / multi        (12 hpf)
  5. S31   H2A      multipolar             (best frame 10-12 hpf)
  6. G043  LynTom   disorganised multipolar (12 hpf)
"""
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LYN_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
H2A_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\section5")
OUT.mkdir(parents=True, exist_ok=True)

BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"

PANEL_PX = 380
UM_PER_PX = 1.5
SCALE_BAR_UM = 200

BF_LO, BF_HI = 1.0, 99.7
BF_WEIGHT = 0.55
GFP_LO, GFP_HI = 55, 99.5
GFP_GAIN = 1.40
RED_LO, RED_HI = 12, 99.0
RED_GAIN = 1.15
GFP_TINT = (0.00, 1.00, 0.00)
LYN_TINT = (1.00, 0.40, 0.00)
H2A_TINT = (1.00, 0.00, 0.00)

LYN_TIMING = {"hpf_start": 6.0, "hpf_interval": 0.4}
H2A_TIMING = {"hpf_start": 7.0, "hpf_interval": 0.1941}

# Each example: dataset, condition, pescoid id, red-channel filename, red tint,
# timing, hpf target (single value OR (lo, hi) window -> pick best frame)
EXAMPLES = [
    # top row
    {"root": LYN_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "G046",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYN_TIMING, "hpf": 12.0},
    {"root": LYN_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "G039",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYN_TIMING, "hpf": 12.0},
    {"root": LYN_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "G040",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYN_TIMING, "hpf": 12.0},
    # bottom row
    {"root": H2A_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "S24",
     "red_file": "h2a_aligned.tif", "red_tint": H2A_TINT,
     "timing": H2A_TIMING, "hpf": 12.0},
    {"root": H2A_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "S31",
     "red_file": "h2a_aligned.tif", "red_tint": H2A_TINT,
     "timing": H2A_TIMING, "hpf": 10.5},   # show third tiny pole emerging
    {"root": LYN_PHASE1, "cond": "P_Activin_3-5hpf", "pid": "G043",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYN_TIMING, "hpf": 12.0},
]


def norm01_mask(img, lo, hi, mask):
    vals = img[mask] if mask is not None and mask.any() else img.ravel()
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


def crop_centered(rgb, mask, panel_px):
    ys, xs = np.where(mask)
    if ys.size == 0:
        cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
    else:
        cy = int(ys.mean()); cx = int(xs.mean())
    half = panel_px // 2
    y0, x0 = cy - half, cx - half
    out = np.zeros((panel_px, panel_px, 3), dtype=np.float32)
    sy0, sx0 = max(0, -y0), max(0, -x0)
    iy0, ix0 = max(0, y0), max(0, x0)
    iy1 = min(rgb.shape[0], y0 + panel_px)
    ix1 = min(rgb.shape[1], x0 + panel_px)
    h, w = iy1 - iy0, ix1 - ix0
    if h > 0 and w > 0:
        out[sy0:sy0 + h, sx0:sx0 + w] = rgb[iy0:iy1, ix0:ix1]
    return out


def best_frame_in_window(gfp, mask, t_lo, t_hi):
    """Pick the frame in [t_lo, t_hi] (inclusive) with the highest p99 mezzo
    signal inside the mask - proxy for the clearest multipolar moment."""
    best_t, best_v = t_lo, -np.inf
    for t in range(t_lo, t_hi + 1):
        if t < 0 or t >= gfp.shape[0] or not mask[t].any():
            continue
        v = float(np.percentile(gfp[t][mask[t]].astype(np.float32), 99))
        if v > best_v:
            best_v = v; best_t = t
    return best_t


def render_example(spec):
    pdir = spec["root"] / "per_pescoid" / spec["cond"] / spec["pid"]
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    red = tifffile.imread(str(pdir / spec["red_file"]))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    timing = spec["timing"]
    hpf = spec["hpf"]
    if isinstance(hpf, tuple):
        t_lo = int(round((hpf[0] - timing["hpf_start"]) / timing["hpf_interval"]))
        t_hi = int(round((hpf[1] - timing["hpf_start"]) / timing["hpf_interval"]))
        t_lo = max(0, min(T - 1, t_lo))
        t_hi = max(0, min(T - 1, t_hi))
        t = best_frame_in_window(gfp, mask, t_lo, t_hi)
        actual = timing["hpf_start"] + t * timing["hpf_interval"]
        print(f"  {spec['pid']}: best frame in [{hpf[0]:.1f}, {hpf[1]:.1f}] "
              f"hpf = frame {t} ({actual:.2f} hpf)")
    else:
        t = int(round((hpf - timing["hpf_start"]) / timing["hpf_interval"]))
        t = max(0, min(T - 1, t))
        print(f"  {spec['pid']}: frame {t} ({timing['hpf_start']+t*timing['hpf_interval']:.2f} hpf)")
    rgb = merged_rgb(bf[t], gfp[t], red[t], mask[t], spec["red_tint"])
    return crop_centered(rgb, mask[t], PANEL_PX)


def main():
    panels = []
    for spec in EXAMPLES:
        try:
            panels.append(render_example(spec))
        except Exception as e:
            print(f"  FAIL {spec['pid']}: {e}")
            panels.append(None)

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.6 * n_cols, 3.4 * n_rows),
                              facecolor=BG_CREAM)
    flat = axes.flatten()
    for i, ax in enumerate(flat):
        if i < len(panels) and panels[i] is not None:
            ax.imshow(panels[i])
        else:
            ax.imshow(np.zeros((PANEL_PX, PANEL_PX, 3)))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(EMBL_NAVY)
            sp.set_linewidth(0.8)
    # 200um scale bar on the very last panel
    bar_len_px = SCALE_BAR_UM / UM_PER_PX
    bar_h = max(3, int(PANEL_PX * 0.013))
    rect_y = PANEL_PX - int(PANEL_PX * 0.06) - bar_h
    rect_x = PANEL_PX - int(PANEL_PX * 0.06) - int(bar_len_px)
    flat[-1].add_patch(Rectangle((rect_x, rect_y), bar_len_px, bar_h,
                                   color="white", lw=0))
    flat[-1].text(rect_x + bar_len_px / 2, rect_y - 6, f"{SCALE_BAR_UM} um",
                   ha="center", va="bottom", color="white", fontsize=11,
                   fontweight="bold")

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    out = OUT / "section5_phenotype_gallery.png"
    plt.savefig(str(out), dpi=170, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
