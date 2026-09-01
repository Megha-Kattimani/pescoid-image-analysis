"""
Joint plot: pescoid AR vs mezzo+ fraction at peak elongation, with
representative pescoid thumbnails pinned to the scatter.

Left panel: scatter, one point per pescoid.
  x = aspect ratio at peak (BF mask ellipse fit)
  y = mezzo+ pixel fraction at peak (gfp_bgsub > global threshold inside mask)
  colour = phenotype, marker size = number of mezzo poles
  4 letter markers (A B C D) at the locations of 4 representative pescoids

Right panel: 2x2 grid of pescoid thumbnails labelled A B C D, matching the
scatter markers. Each thumbnail is BF + mezzo:GFP + LynTom (orange-red),
showing which kind of mezzo pattern each AR/fraction region corresponds to.

Output:
  Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase5_morphometrics\\
      plots\\ar_vs_mezzo_joint.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec

LYN_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
LYN_PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\plots")
OUT.mkdir(parents=True, exist_ok=True)

# Palette
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
EMBL_GREEN = "#00A689"
TEXT_DARK = "#1F1F1F"

HPF_START = 6.0
HPF_INTERVAL = 0.4
GLOBAL_THR_BGSUB = 188.17

PHENO_ORDER = ["no_induction", "diffuse_mezzo", "coordinated_monopolar",
               "mezzo_bipolar_only", "coordinated_bipolar",
               "disorganised_multipolar", "multipolar", "oc_only"]
PHENO_COLOR = {
    "no_induction": "#999999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77", "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a", "oc_only": "#e6ab02",
}
PHENO_DISPLAY = {
    "no_induction": "no induction", "diffuse_mezzo": "diffuse mezzo",
    "coordinated_monopolar": "monopolar", "mezzo_bipolar_only": "mezzo bipolar",
    "coordinated_bipolar": "coord bipolar",
    "disorganised_multipolar": "disorg multipolar",
    "multipolar": "multipolar", "oc_only": "OC only",
}

# 4 representative thumbnails (LynTom dataset), one per scatter quadrant
THUMBS = [
    ("A", "G024", "no induction\nlow AR, no mezzo"),
    ("B", "G036", "diffuse mezzo\nlow AR, high mezzo"),
    ("C", "G046", "monopolar\nhigh AR, focused pole"),
    ("D", "G050", "bipolar\nhigh AR, two poles"),
]

LYN_TINT = (1.00, 0.40, 0.00)
GFP_TINT = (0.00, 1.00, 0.00)
PANEL_PX = 360
BF_LO, BF_HI = 1.0, 99.7
BF_WEIGHT = 0.55
GFP_LO, GFP_HI = 55, 99.5
GFP_GAIN = 1.40
RED_LO, RED_HI = 12, 99.0
RED_GAIN = 1.15


# ------- per-pescoid AR + mezzo fraction at peak ---------
def per_pescoid_peak(cond, pid):
    pdir = LYN_PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = mask.shape[0]
    ar = np.full(T, np.nan); frac = np.full(T, np.nan)
    for t in range(T):
        if not mask[t].any():
            continue
        props = measure.regionprops(mask[t].astype(int))
        if not props:
            continue
        p = max(props, key=lambda r: r.area)
        minor = float(p.minor_axis_length) or 1e-6
        ar[t] = p.major_axis_length / minor
        bg = float(gfp[t][~mask[t]].mean()) if (~mask[t]).any() else 0.0
        pos = (gfp[t] - bg) > GLOBAL_THR_BGSUB
        frac[t] = (pos & mask[t]).sum() / max(mask[t].sum(), 1)
    return float(np.nanmax(ar)), float(np.nanmax(frac))


# ------- render thumbnail merged channels ---------
def norm01_mask(img, lo, hi, mask):
    vals = img[mask] if mask is not None and mask.any() else img.ravel()
    p_lo, p_hi = np.percentile(vals, lo), np.percentile(vals, hi)
    return np.clip((img.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1.0),
                    0, 1)


def merged_rgb(bf, gfp, red, mask):
    bf_n = norm01_mask(bf, BF_LO, BF_HI, None)
    if (~mask).any():
        gfp_bg = float(gfp[~mask].mean()); red_bg = float(red[~mask].mean())
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
    for c, t in enumerate(LYN_TINT):
        if t > 0:
            rgb[..., c] = np.where(mask, rgb[..., c] + red_n * t, rgb[..., c])
    return np.clip(rgb, 0, 1)


def crop_centered(rgb, mask, panel_px):
    ys, xs = np.where(mask)
    if ys.size == 0:
        cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
    else:
        cy, cx = int(ys.mean()), int(xs.mean())
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


def render_thumb(pid):
    # Determine condition from phenotype summary
    df = pd.read_csv(str(LYN_PHASE2 / "phenotype_summary.csv"))
    row = df[df.pescoid == pid].iloc[0]
    cond = row.condition
    pdir = LYN_PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    red = tifffile.imread(str(pdir / "lyntom_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    t = int(round((12.0 - HPF_START) / HPF_INTERVAL))
    t = max(0, min(T - 1, t))
    rgb = merged_rgb(bf[t], gfp[t], red[t], mask[t])
    return crop_centered(rgb, mask[t], PANEL_PX)


def main():
    df = pd.read_csv(str(LYN_PHASE2 / "phenotype_summary.csv"))
    df = df[df.condition.isin(["P_ctrl", "P_Activin_3-5hpf"])].copy()

    rows = []
    for _, r in df.iterrows():
        try:
            ar_max, frac_max = per_pescoid_peak(r.condition, r.pescoid)
        except Exception as e:
            print(f"  FAIL {r.pescoid}: {e}")
            continue
        rows.append({
            "condition": r.condition, "pescoid": r.pescoid,
            "phenotype_peak": r.phenotype_peak,
            "n_poles": int(r.n_mezzo_poles_peak),
            "ar_max": ar_max, "frac_max": frac_max,
        })
    res = pd.DataFrame(rows)
    res.to_csv(str(OUT / "ar_vs_mezzo_per_pescoid.csv"), index=False)
    print(f"\nN = {len(res)}")
    print(res[["condition", "pescoid", "phenotype_peak", "n_poles",
                "ar_max", "frac_max"]].head())

    # ---- Figure ----
    plt.rcParams.update({"font.family": "Arial"})
    fig = plt.figure(figsize=(16, 9), facecolor=BG_CREAM)
    gs = GridSpec(2, 4, width_ratios=[2.4, 1, 1, 0.05],
                   height_ratios=[1, 1], hspace=0.15, wspace=0.20,
                   left=0.05, right=0.97, top=0.92, bottom=0.08)

    # ---- left: scatter ----
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(BG_CREAM)
    for pheno in PHENO_ORDER:
        sub = res[res.phenotype_peak == pheno]
        if sub.empty:
            continue
        sizes = 60 + sub.n_poles * 80
        ax.scatter(sub.ar_max, sub.frac_max, s=sizes,
                    color=PHENO_COLOR[pheno], edgecolors=EMBL_NAVY, lw=0.7,
                    alpha=0.85, label=PHENO_DISPLAY[pheno])
    # mark the thumbnails A/B/C/D with letter labels on top of the scatter
    thumb_coords = {}
    for letter, pid, _ in THUMBS:
        row = res[res.pescoid == pid]
        if row.empty:
            print(f"WARNING: thumbnail {pid} not found in results")
            continue
        x, y = float(row.ar_max.iloc[0]), float(row.frac_max.iloc[0])
        thumb_coords[letter] = (x, y, pid)
        # circular outline + letter
        ax.scatter(x, y, s=400, facecolor="white", edgecolor=EMBL_NAVY,
                    lw=2.2, zorder=5)
        ax.text(x, y, letter, ha="center", va="center", fontsize=18,
                fontweight="bold", color=EMBL_NAVY, zorder=6)

    ax.set_xlabel("Pescoid aspect ratio at peak (BF mask)",
                  fontsize=14, color=EMBL_NAVY, fontweight="bold")
    ax.set_ylabel("Mezzo+ pixel fraction at peak\n"
                  "(gfp_bgsub above global threshold, inside mask)",
                  fontsize=14, color=EMBL_NAVY, fontweight="bold")
    ax.tick_params(colors=EMBL_NAVY, labelsize=12)
    for sp in ax.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color="white", lw=1.2)
    ax.set_title("Pescoid shape vs mezzo+ induction at peak elongation "
                  "(12 hpf)\nMarker size = number of mezzo+ poles",
                  fontsize=14, fontweight="bold", color=EMBL_NAVY, pad=10)
    ax.legend(fontsize=10, frameon=False, labelcolor=EMBL_NAVY,
              loc="upper left", ncol=2)

    # ---- right: 2x2 thumbnails ----
    grid_positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for (letter, pid, label), (gr, gc) in zip(THUMBS, grid_positions):
        ax_t = fig.add_subplot(gs[gr, gc])
        try:
            rgb = render_thumb(pid)
        except Exception as e:
            print(f"  FAIL thumb {pid}: {e}")
            continue
        ax_t.imshow(rgb)
        ax_t.set_xticks([]); ax_t.set_yticks([])
        ax_t.set_facecolor("black")
        for sp in ax_t.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(EMBL_NAVY); sp.set_linewidth(1.2)
        # letter badge in top-left corner of thumbnail
        ax_t.add_patch(Rectangle((10, 10), 55, 55, facecolor="white",
                                    edgecolor=EMBL_NAVY, lw=1.5, zorder=4))
        ax_t.text(37, 36, letter, ha="center", va="center", fontsize=22,
                  fontweight="bold", color=EMBL_NAVY, zorder=5)
        # caption below
        ax_t.set_title(f"{letter} - {pid}\n{label}", fontsize=10,
                        color=EMBL_NAVY, fontweight="bold", pad=4)

    fig.suptitle("Aspect ratio vs mezzo+ fraction - "
                  "which shape corresponds to which mezzo pattern",
                  fontsize=17, fontweight="bold", color=EMBL_NAVY, y=0.985)

    out = OUT / "ar_vs_mezzo_joint.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
