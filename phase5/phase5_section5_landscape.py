"""
Section 5 - The multipolar landscape.

Left (large): phenotype distribution stacked bar (P_ctrl vs P_Activin 3-5 hpf).
Right: 4 representative pescoid images (LynTom merged channels at 12 hpf),
       one per phenotype, with QR-code placeholders under each (to link to
       AVI movies once uploaded).

Headline: "Activin 3-5 hpf raises multi-organiser phenotypes from 0% (N=15 ctrl)
           to ~32% (N=37 treated)."
Annotation: "Identical chemistry -> broad outcome distribution. The system is
             operating at a critical point where mechanics, not chemistry alone,
             sets the outcome."

Palette: EMBL cream + navy (matches the poster template).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

LYN_PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
LYN_PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\section5")
OUT.mkdir(parents=True, exist_ok=True)

# Palette
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
EMBL_GREEN = "#00A689"
EMBL_RED = "#C04848"
TEXT_DARK = "#1F1F1F"
LIGHT_GREY = "#D8D2C4"

PHENO_ORDER = ["no_induction", "diffuse_mezzo", "coordinated_monopolar",
               "mezzo_bipolar_only", "coordinated_bipolar",
               "disorganised_multipolar", "multipolar", "oc_only"]
PHENO_COLOR = {
    "no_induction": "#999999",
    "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0",
    "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77",
    "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a",
    "oc_only": "#e6ab02",
}
PHENO_DISPLAY = {
    "no_induction": "no induction",
    "diffuse_mezzo": "diffuse mezzo",
    "coordinated_monopolar": "monopolar",
    "mezzo_bipolar_only": "mezzo bipolar",
    "coordinated_bipolar": "coord bipolar",
    "disorganised_multipolar": "disorg multipolar",
    "multipolar": "multipolar",
    "oc_only": "OC only",
}

# 4 example pescoids (one per phenotype family) -- all LynTom dataset
EXAMPLES = [
    ("G046", "Monopolar",            "coordinated_monopolar"),
    ("G047", "Bipolar",              "coordinated_bipolar"),
    ("G060", "Multipolar",           "multipolar"),
    ("G061", "Disorganised multipolar", "disorganised_multipolar"),
]
EXAMPLE_HPF = 12.0
LYN_TIMING = {"hpf_start": 6.0, "hpf_interval": 0.4}

BF_LO, BF_HI = 1.0, 99.7
BF_WEIGHT = 0.55
GFP_LO, GFP_HI = 55, 99.5
GFP_GAIN = 1.40
RED_LO, RED_HI = 12, 99.0
RED_GAIN = 1.15
LYN_TINT = (1.00, 0.40, 0.00)
GFP_TINT = (0.00, 1.00, 0.00)

PANEL_PX = 360
UM_PER_PX = 1.5
SCALE_BAR_UM = 200


# ----- rendering helpers (reused) ----------
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
    sy0 = max(0, -y0); sx0 = max(0, -x0)
    iy0 = max(0, y0); ix0 = max(0, x0)
    iy1 = min(rgb.shape[0], y0 + panel_px)
    ix1 = min(rgb.shape[1], x0 + panel_px)
    h_take = iy1 - iy0; w_take = ix1 - ix0
    if h_take > 0 and w_take > 0:
        out[sy0:sy0 + h_take, sx0:sx0 + w_take] = rgb[iy0:iy1, ix0:ix1]
    return out


def render_example(pid):
    cond = "P_Activin_3-5hpf"
    pdir = LYN_PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    red = tifffile.imread(str(pdir / "lyntom_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    t = int(round((EXAMPLE_HPF - LYN_TIMING["hpf_start"]) /
                    LYN_TIMING["hpf_interval"]))
    t = max(0, min(T - 1, t))
    rgb = merged_rgb(bf[t], gfp[t], red[t], mask[t], LYN_TINT)
    return crop_centered(rgb, mask[t], PANEL_PX)


def qr_placeholder(size_px, seed):
    """Deterministic faux-QR pattern (replaced with the real QR before printing).
    Black/white modules in a grid - looks QR-ish at a glance."""
    rng = np.random.default_rng(seed)
    grid_n = 21          # 21x21 module grid (QR v1 default)
    pad = 2              # quiet-zone modules
    total_n = grid_n + 2 * pad
    grid = rng.integers(0, 2, size=(grid_n, grid_n))
    # finder patterns (top-left, top-right, bottom-left)
    fp = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])
    for (gy, gx) in [(0, 0), (0, grid_n - 7), (grid_n - 7, 0)]:
        grid[gy:gy + 7, gx:gx + 7] = fp
    full = np.ones((total_n, total_n), dtype=np.uint8)
    full[pad:pad + grid_n, pad:pad + grid_n] = grid
    # upsample to size_px (nearest-neighbour for crisp modules)
    rep = size_px // total_n
    image = np.kron(full, np.ones((rep, rep), dtype=np.uint8))
    # pad to size_px
    out = np.ones((size_px, size_px), dtype=np.uint8)
    out[:image.shape[0], :image.shape[1]] = image
    return out  # 0 = black, 1 = white


def main():
    df = pd.read_csv(str(LYN_PHASE2 / "phenotype_summary.csv"))
    df = df[df.condition.isin(["P_ctrl", "P_Activin_3-5hpf"])].copy()
    n_ctrl = (df.condition == "P_ctrl").sum()
    n_act = (df.condition == "P_Activin_3-5hpf").sum()
    print(f"N ctrl = {n_ctrl}, N Activin = {n_act}")

    # ----- phenotype stacked bar -----
    conds = ["P_ctrl", "P_Activin_3-5hpf"]
    cond_labels = [f"P ctrl\n(N={n_ctrl})", f"P Activin 3-5h\n(N={n_act})"]
    fracs = {}
    for p in PHENO_ORDER:
        fracs[p] = []
        for c in conds:
            sub = df[df.condition == c]
            fracs[p].append((sub.phenotype_peak == p).mean() * 100 if len(sub) else 0)

    # ----- rendering -----
    plt.rcParams.update({"font.family": "Arial"})
    fig = plt.figure(figsize=(20, 9), facecolor=BG_CREAM)
    gs = GridSpec(2, 6, height_ratios=[1, 0.18], width_ratios=[2.4, 1, 1, 1, 1, 0.05],
                   hspace=0.20, wspace=0.18, left=0.04, right=0.98,
                   top=0.86, bottom=0.04)

    # ---- LEFT: stacked bar ----
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.set_facecolor(BG_CREAM)
    bottoms = np.zeros(len(conds))
    for p in PHENO_ORDER:
        vals = np.array(fracs[p])
        if vals.sum() == 0:
            continue
        ax_bar.bar(range(len(conds)), vals, bottom=bottoms,
                   color=PHENO_COLOR[p], label=PHENO_DISPLAY[p],
                   edgecolor="white", lw=1.3, width=0.65)
        # annotate non-trivial slices
        for i, v in enumerate(vals):
            if v >= 6:
                ax_bar.text(i, bottoms[i] + v / 2, f"{v:.0f}%",
                            ha="center", va="center", color="white",
                            fontsize=12, fontweight="bold")
        bottoms += vals
    ax_bar.set_xticks(range(len(conds)))
    ax_bar.set_xticklabels(cond_labels, fontsize=14, color=EMBL_NAVY,
                            fontweight="bold")
    ax_bar.set_ylabel("% of pescoids", fontsize=14, color=EMBL_NAVY,
                       fontweight="bold")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_yticks([0, 25, 50, 75, 100])
    ax_bar.tick_params(colors=EMBL_NAVY)
    for sp in ax_bar.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_title("Phenotype distribution at peak elongation (12 hpf)",
                      fontsize=15, fontweight="bold", color=EMBL_NAVY, pad=10)
    ax_bar.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11,
                   frameon=False, labelcolor=EMBL_NAVY)

    # ---- RIGHT: 4 phenotype example pescoids + QR placeholders ----
    print("\nRendering 4 phenotype examples...")
    for i, (pid, label, pheno) in enumerate(EXAMPLES):
        try:
            rgb = render_example(pid)
        except Exception as e:
            print(f"  FAIL {pid}: {e}")
            continue
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(PHENO_COLOR[pheno])
            sp.set_linewidth(2.0)
        ax.set_title(f"{label}\n({pid})", fontsize=12, fontweight="bold",
                      color=PHENO_COLOR[pheno], pad=6)
        # scale bar on the last image
        if i == len(EXAMPLES) - 1:
            bar_len_px = SCALE_BAR_UM / UM_PER_PX
            bar_h = max(3, int(PANEL_PX * 0.014))
            ry = PANEL_PX - int(PANEL_PX * 0.06) - bar_h
            rx = PANEL_PX - int(PANEL_PX * 0.06) - int(bar_len_px)
            ax.add_patch(Rectangle((rx, ry), bar_len_px, bar_h,
                                     color="white", lw=0))
            ax.text(rx + bar_len_px / 2, ry - 6, f"{SCALE_BAR_UM} um",
                    ha="center", va="bottom", color="white", fontsize=10,
                    fontweight="bold")
        # QR code below
        ax_qr = fig.add_subplot(gs[1, i + 1])
        qr = qr_placeholder(180, seed=hash(pid) & 0xFFFF)
        ax_qr.imshow(qr, cmap="binary", aspect="equal")
        ax_qr.set_xticks([]); ax_qr.set_yticks([])
        ax_qr.set_xlabel("scan -> AVI movie", fontsize=9, color=EMBL_NAVY,
                          fontweight="bold")
        for sp in ax_qr.spines.values():
            sp.set_color(EMBL_NAVY); sp.set_linewidth(0.6)

    # ---- Headline + annotation ----
    fig.text(0.50, 0.96,
             "Activin 3-5 hpf raises multi-organiser phenotypes from "
             f"0% (N={n_ctrl} ctrl) to ~32% (N={n_act} treated)",
             ha="center", va="center", fontsize=18, fontweight="bold",
             color=EMBL_NAVY)
    fig.text(0.50, 0.92,
             "Section 5 -- Classification: the multipolar landscape",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color=EMBL_NAVY, style="italic")
    fig.text(0.50, 0.005,
             "Identical chemistry -> broad outcome distribution.  "
             "The system operates at a critical point where mechanics, "
             "not chemistry alone, sets the outcome.",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=EMBL_RED, style="italic")

    out = OUT / "section5_multipolar_landscape.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight",
                facecolor=BG_CREAM)
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
