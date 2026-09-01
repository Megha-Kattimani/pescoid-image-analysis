"""
Section 4 GALLERY v2 - uniform panel sizes + myosin row.

Rows:
  1. LynTom    P ctrl       G030 (from Lyn_mezzo_phase1)
  2. LynTom    P Activin    G047 (bipolar)
  3. H2A       P ctrl       S11  (from mezzo_H2A_phase1)
  4. H2A       P Activin    S27  (multipolar)
  5. myosin    P ctrl       scene 13 (from raw 260514 TIFF)
  6. myosin    P Activin    scene 27 (from raw 260514 TIFF)

Channel tints:
  mezzo:GFP        green             (0.00, 1.00, 0.00)
  LynTom:mCherry   orange-red        (1.00, 0.40, 0.00)
  H2A:mCherry      pure red          (1.00, 0.00, 0.00)
  myosin:mCherry   pink-magenta      (1.00, 0.30, 0.60)

Layout: all panels padded to a fixed PANEL_PX square so the grid is a clean matrix.
Scale bar: 200 um (UM_PER_PX = 1.5).
"""
from pathlib import Path
import numpy as np
import tifffile
from skimage import filters, morphology, measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LYN_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
H2A_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
MYO_RAW = Path(r"Z:\Megha_Kattimani\Imaging\Zeiss\260514_mezzo_myo_3-5hpfChiron0.5-1-5nm_obj10x\TIFF")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\section4")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_HPF = [7.0, 10.0, 12.0, 15.0]
PANEL_PX = 380              # uniform panel size in pixels (square)

BF_LO, BF_HI = 1.0, 99.7
BF_WEIGHT = 0.55
GFP_LO, GFP_HI = 55, 99.5
GFP_GAIN = 1.40
RED_LO, RED_HI = 12, 99.0
RED_GAIN = 1.15

GFP_TINT = (0.00, 1.00, 0.00)
LYN_TINT = (1.00, 0.40, 0.00)
H2A_TINT = (1.00, 0.00, 0.00)
MYO_TINT = (1.00, 0.30, 0.60)

UM_PER_PX = 1.5
SCALE_BAR_UM = 200

# EMBL template palette (matches the poster v10)
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
EMBL_GREEN_HEX = "#00A689"
TEXT_DARK_HEX = "#1F1F1F"

LYNTOM_TIMING = {"hpf_start": 6.0, "hpf_interval": 0.4}
H2A_TIMING = {"hpf_start": 7.0, "hpf_interval": 0.1941}
MYO_TIMING = {"hpf_start": 7.0, "hpf_interval": 0.1941}    # assumed - similar 10x rig

ROWS = [
    {"source": "phase1", "root": LYN_PHASE1, "cond": "P_ctrl",
     "pid": "G030", "label": "LynTom\nP ctrl",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYNTOM_TIMING},
    {"source": "phase1", "root": LYN_PHASE1, "cond": "P_Activin_3-5hpf",
     "pid": "G047", "label": "LynTom\nP Activin",
     "red_file": "lyntom_aligned.tif", "red_tint": LYN_TINT,
     "timing": LYNTOM_TIMING},
    {"source": "phase1", "root": H2A_PHASE1, "cond": "P_ctrl",
     "pid": "S11", "label": "H2A\nP ctrl",
     "red_file": "h2a_aligned.tif", "red_tint": H2A_TINT,
     "timing": H2A_TIMING},
    {"source": "phase1", "root": H2A_PHASE1, "cond": "P_Activin_3-5hpf",
     "pid": "S27", "label": "H2A\nP Activin",
     "red_file": "h2a_aligned.tif", "red_tint": H2A_TINT,
     "timing": H2A_TIMING},
    {"source": "raw_260514", "scene": 13, "pid": "scene13",
     "label": "myosin\nP ctrl",
     "red_tint": MYO_TINT, "timing": MYO_TIMING},
    {"source": "raw_260514", "scene": 27, "pid": "scene27",
     "label": "myosin\nP Activin",
     "red_tint": MYO_TINT, "timing": MYO_TIMING},
]

# 260514 raw channel order  -- C0=GFP, C1=mCherry-myosin, C2=BF (verified by intensity profile)
RAW_260514_CH = {"gfp": 0, "red": 1, "bf": 2}


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


def quick_mask(bf):
    """Simple Otsu-based mask of the pescoid in BF (for raw data without a saved mask)."""
    # invert if needed (pescoid darker than background)
    bg = np.median(bf)
    is_dark = bf.mean() > np.percentile(bf, 30)
    work = bf.astype(np.float32)
    thr = filters.threshold_otsu(work)
    if is_dark:
        m = work < thr
    else:
        m = work > thr
    m = morphology.binary_closing(m, morphology.disk(5))
    m = morphology.remove_small_holes(m, area_threshold=500)
    labels = measure.label(m)
    if labels.max() == 0:
        return m
    props = measure.regionprops(labels)
    biggest = max(props, key=lambda r: r.area)
    keep = labels == biggest.label
    return keep


def crop_centered(rgb, mask, panel_px):
    """Centre a (panel_px x panel_px) crop on the mask centroid, padding with black."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
    else:
        cy = int(ys.mean()); cx = int(xs.mean())
    half = panel_px // 2
    y0 = cy - half; x0 = cx - half
    out = np.zeros((panel_px, panel_px, 3), dtype=np.float32)
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    img_y0 = max(0, y0)
    img_x0 = max(0, x0)
    img_y1 = min(rgb.shape[0], y0 + panel_px)
    img_x1 = min(rgb.shape[1], x0 + panel_px)
    h_take = img_y1 - img_y0
    w_take = img_x1 - img_x0
    if h_take > 0 and w_take > 0:
        out[src_y0:src_y0 + h_take, src_x0:src_x0 + w_take] = \
            rgb[img_y0:img_y1, img_x0:img_x1]
    return out


def render_row_phase1(row):
    pdir = row["root"] / "per_pescoid" / row["cond"] / row["pid"]
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    red = tifffile.imread(str(pdir / row["red_file"]))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    panels = []
    for h in TARGET_HPF:
        t = int(round((h - row["timing"]["hpf_start"]) /
                        row["timing"]["hpf_interval"]))
        t = max(0, min(T - 1, t))
        rgb = merged_rgb(bf[t], gfp[t], red[t], mask[t], row["red_tint"])
        panels.append(crop_centered(rgb, mask[t], PANEL_PX))
    return np.stack(panels)


def render_row_raw_260514(row):
    """Load the raw CZI/TIFF for one scene, max-project z, render BF+GFP+myosin."""
    scene_num = row["scene"]
    fname = f"New-01.czi - New-01.czi #{scene_num:02d}.tif"
    arr = tifffile.imread(str(MYO_RAW / fname))      # T, Z, C, Y, X
    T, Z, C, H, W = arr.shape
    # max-project z
    bf_stack = arr[:, Z // 2, RAW_260514_CH["bf"]]              # mid-z BF
    gfp_stack = arr[:, :, RAW_260514_CH["gfp"]].max(axis=1)      # z-max GFP
    red_stack = arr[:, :, RAW_260514_CH["red"]].max(axis=1)      # z-max myosin
    panels = []
    for h in TARGET_HPF:
        t = int(round((h - row["timing"]["hpf_start"]) /
                        row["timing"]["hpf_interval"]))
        t = max(0, min(T - 1, t))
        bf = bf_stack[t]; gfp = gfp_stack[t]; red = red_stack[t]
        mask = quick_mask(bf)
        rgb = merged_rgb(bf, gfp, red, mask, row["red_tint"])
        panels.append(crop_centered(rgb, mask, PANEL_PX))
    return np.stack(panels)


def main():
    print("Rendering rows...")
    rows_rendered = []
    for row in ROWS:
        try:
            if row["source"] == "phase1":
                panels = render_row_phase1(row)
            elif row["source"] == "raw_260514":
                panels = render_row_raw_260514(row)
            else:
                raise ValueError(row["source"])
            rows_rendered.append((row, panels))
            print(f"  {row['label'].replace(chr(10), ' / ')} ({row['pid']}) ok")
        except Exception as e:
            print(f"  FAIL {row['pid']}: {e}")

    n_rows = len(rows_rendered)
    n_cols = len(TARGET_HPF)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.4 * n_cols, 3.1 * n_rows),
                              facecolor=BG_CREAM, squeeze=False)
    for ri, (row, panels) in enumerate(rows_rendered):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.imshow(panels[ci])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            # subtle navy border around each panel
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(EMBL_NAVY)
                sp.set_linewidth(0.8)
            if ri == 0:
                ax.set_title(f"{TARGET_HPF[ci]:.0f} hpf", fontsize=20,
                             fontweight="bold", color=EMBL_NAVY)
            if ci == 0:
                ax.set_ylabel(row["label"] + f"\n({row['pid']})",
                              fontsize=13, fontweight="bold",
                              color=EMBL_NAVY, rotation=0, ha="right",
                              va="center", labelpad=80)
        # scale bar on the rightmost panel of EVERY row (white on the
        # black panel background - readable, no clash with cream poster bg)
        bar_len_px = SCALE_BAR_UM / UM_PER_PX
        bar_h = max(3, int(PANEL_PX * 0.012))
        rect_y = PANEL_PX - int(PANEL_PX * 0.06) - bar_h
        rect_x = PANEL_PX - int(PANEL_PX * 0.06) - int(bar_len_px)
        axes[ri, -1].add_patch(Rectangle((rect_x, rect_y), bar_len_px, bar_h,
                                          color="white", lw=0))
        if ri == n_rows - 1:
            axes[ri, -1].text(rect_x + bar_len_px / 2, rect_y - 6,
                               f"{SCALE_BAR_UM} um",
                               ha="center", va="bottom", color="white",
                               fontsize=13, fontweight="bold")

    lyn_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in LYN_TINT)
    h2a_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in H2A_TINT)
    myo_hex = "#%02X%02X%02X" % tuple(int(c * 255) for c in MYO_TINT)
    handles = [
        plt.Line2D([0], [0], color=TEXT_DARK_HEX, lw=10, label="brightfield"),
        plt.Line2D([0], [0], color=EMBL_GREEN_HEX, lw=10, label="mezzo:GFP"),
        plt.Line2D([0], [0], color=lyn_hex, lw=10,
                   label="LynTom:mCherry (membrane)"),
        plt.Line2D([0], [0], color=h2a_hex, lw=10,
                   label="H2A:mCherry (nuclei)"),
        plt.Line2D([0], [0], color=myo_hex, lw=10,
                   label="myosin:mCherry (cortex)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5,
                bbox_to_anchor=(0.5, 1.00), frameon=False,
                fontsize=13, labelcolor=EMBL_NAVY)
    fig.suptitle("Transgenic phenotypes - control vs Activin 3-5h\n"
                 "Same chemistry, discrete outcomes: one, two, or many "
                 "organisers along the AP axis",
                 fontsize=18, fontweight="bold", color=EMBL_NAVY, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(left=0.08, wspace=0.04, hspace=0.06)
    out = OUT / "section4_gallery_uniform.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
