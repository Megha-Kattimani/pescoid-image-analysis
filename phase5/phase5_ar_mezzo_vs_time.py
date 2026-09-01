"""
AR and mezzo+ fraction as functions of TIME on one graph (dual y-axis).

Left y-axis  : pescoid aspect ratio (BF mask ellipse fit) - solid lines
Right y-axis : mezzo+ pixel fraction inside the mask        - dashed lines

Lines coloured by condition (P ctrl vs P Activin 3-5 hpf), shown with
95% CI ribbons. Two solid + two dashed lines total.

Bonus second panel: faceted by phenotype, same dual-axis design.
Helps read which phenotype contributes which timing.

Output:
  Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase5_morphometrics\\plots\\ar_mezzo_vs_time.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COND_ORDER = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#386cb0", "P_Activin_3-5hpf": "#C04848"}

PHENO_ORDER = ["no_induction", "diffuse_mezzo", "coordinated_monopolar",
               "mezzo_bipolar_only", "coordinated_bipolar",
               "disorganised_multipolar", "multipolar"]
PHENO_COLOR = {
    "no_induction": "#999999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77",
    "disorganised_multipolar": "#d95f02", "multipolar": "#e7298a",
}
PHENO_DISPLAY = {
    "no_induction": "no induction", "diffuse_mezzo": "diffuse mezzo",
    "coordinated_monopolar": "monopolar", "mezzo_bipolar_only": "mezzo bipolar",
    "coordinated_bipolar": "coord bipolar",
    "disorganised_multipolar": "disorg multipolar", "multipolar": "multipolar",
}

HPF_GRID = np.arange(7.0, 16.1, 0.4)        # common time grid for averaging


def per_pescoid_traj(cond, pid):
    """Return arrays (hpf, AR, mezzo+_frac) per frame."""
    pdir = LYN_PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = mask.shape[0]
    hpf = HPF_START + np.arange(T) * HPF_INTERVAL
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
    return hpf, ar, frac


def mean_ci(per_pescoid_y, n_boot=300, seed=0):
    """Per (T, N) -> mean and 95% CI by bootstrap across pescoids.
    Returns mean, lo, hi each of length T."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(per_pescoid_y)             # (T, N)
    mean = np.nanmean(Y, axis=1)
    N = Y.shape[1]
    if N < 2:
        return mean, mean, mean
    idx = rng.integers(0, N, size=(n_boot, N))
    bs = np.nanmean(Y[:, idx], axis=2).T        # (n_boot, T)? Actually careful
    # rewrite cleanly:
    bs = np.empty((n_boot, Y.shape[0]))
    for b in range(n_boot):
        sel = rng.integers(0, N, size=N)
        bs[b] = np.nanmean(Y[:, sel], axis=1)
    lo = np.nanpercentile(bs, 2.5, axis=0)
    hi = np.nanpercentile(bs, 97.5, axis=0)
    return mean, lo, hi


def interp_onto_grid(hpf, y, grid):
    return np.interp(grid, hpf, y, left=np.nan, right=np.nan)


def main():
    df = pd.read_csv(str(LYN_PHASE2 / "phenotype_summary.csv"))
    df = df[df.condition.isin(COND_ORDER)].copy()

    print("Loading per-pescoid trajectories...")
    cond_arr = {c: {"ar": [], "frac": []} for c in COND_ORDER}
    pheno_arr = {p: {"ar": [], "frac": []} for p in PHENO_ORDER}
    for _, r in df.iterrows():
        try:
            hpf, ar, frac = per_pescoid_traj(r.condition, r.pescoid)
        except Exception as e:
            print(f"  FAIL {r.pescoid}: {e}")
            continue
        ar_g = interp_onto_grid(hpf, ar, HPF_GRID)
        frac_g = interp_onto_grid(hpf, frac, HPF_GRID)
        cond_arr[r.condition]["ar"].append(ar_g)
        cond_arr[r.condition]["frac"].append(frac_g)
        if r.phenotype_peak in pheno_arr:
            pheno_arr[r.phenotype_peak]["ar"].append(ar_g)
            pheno_arr[r.phenotype_peak]["frac"].append(frac_g)

    # ----- Figure ----------------------------------------------------------
    plt.rcParams.update({"font.family": "Arial"})
    fig = plt.figure(figsize=(16, 11), facecolor=BG_CREAM)
    gs = GridSpec(2, 1, height_ratios=[1.1, 1.0], hspace=0.32,
                   left=0.06, right=0.94, top=0.94, bottom=0.06)

    # ======== Top panel: condition mean (ctrl vs Activin) with dual axis ===
    ax_ar = fig.add_subplot(gs[0])
    ax_ar.set_facecolor(BG_CREAM)
    ax_frac = ax_ar.twinx()
    ax_frac.set_facecolor("none")
    for c in COND_ORDER:
        ar_mat = np.column_stack(cond_arr[c]["ar"])
        frac_mat = np.column_stack(cond_arr[c]["frac"])
        m_ar, lo_ar, hi_ar = mean_ci(ar_mat)
        m_fr, lo_fr, hi_fr = mean_ci(frac_mat)
        col = COND_COLOR[c]
        ax_ar.fill_between(HPF_GRID, lo_ar, hi_ar, color=col, alpha=0.15)
        ax_ar.plot(HPF_GRID, m_ar, color=col, lw=2.6, ls="-",
                    label=f"{COND_LABEL[c]} - AR (n={ar_mat.shape[1]})")
        ax_frac.fill_between(HPF_GRID, lo_fr, hi_fr, color=col, alpha=0.10)
        ax_frac.plot(HPF_GRID, m_fr, color=col, lw=2.6, ls="--",
                      label=f"{COND_LABEL[c]} - mezzo+ frac")
    ax_ar.axvline(12.0, color="black", ls=":", lw=1.2, alpha=0.5)
    ax_ar.text(12.05, ax_ar.get_ylim()[1] * 0.97, "12 hpf\n(peak elong.)",
                fontsize=10, color=EMBL_NAVY, va="top", ha="left")
    ax_ar.set_xlabel("hpf", fontsize=13, color=EMBL_NAVY, fontweight="bold")
    ax_ar.set_ylabel("Aspect ratio (BF mask)  -  solid lines",
                      fontsize=13, color=EMBL_NAVY, fontweight="bold")
    ax_frac.set_ylabel("Mezzo+ pixel fraction  -  dashed lines",
                       fontsize=13, color=EMBL_NAVY, fontweight="bold")
    ax_ar.tick_params(colors=EMBL_NAVY, labelsize=11)
    ax_frac.tick_params(colors=EMBL_NAVY, labelsize=11)
    for sp in ax_ar.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    for sp in ax_frac.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax_ar.spines["top"].set_visible(False)
    ax_frac.spines["top"].set_visible(False)
    ax_ar.set_title("Aspect ratio and mezzo+ fraction over time - "
                      "population means (95% CI)",
                      fontsize=15, fontweight="bold", color=EMBL_NAVY, pad=8)
    h1, l1 = ax_ar.get_legend_handles_labels()
    h2, l2 = ax_frac.get_legend_handles_labels()
    ax_ar.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10,
                  frameon=False, labelcolor=EMBL_NAVY, ncol=2)

    # ======== Bottom panel: faceted by phenotype, same dual-axis ==========
    keep = [p for p in PHENO_ORDER if len(pheno_arr[p]["ar"]) >= 3]
    n_p = len(keep)
    gs2 = GridSpec(1, n_p, left=0.06, right=0.94, bottom=0.06, top=0.46,
                    wspace=0.35)
    for i, p in enumerate(keep):
        ax = fig.add_subplot(gs2[i])
        ax.set_facecolor(BG_CREAM)
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        ar_mat = np.column_stack(pheno_arr[p]["ar"])
        frac_mat = np.column_stack(pheno_arr[p]["frac"])
        m_ar, lo_ar, hi_ar = mean_ci(ar_mat)
        m_fr, lo_fr, hi_fr = mean_ci(frac_mat)
        col = PHENO_COLOR[p]
        ax.fill_between(HPF_GRID, lo_ar, hi_ar, color=col, alpha=0.15)
        ax.plot(HPF_GRID, m_ar, color=col, lw=2.0, ls="-")
        ax2.fill_between(HPF_GRID, lo_fr, hi_fr, color=col, alpha=0.10)
        ax2.plot(HPF_GRID, m_fr, color=col, lw=2.0, ls="--")
        ax.axvline(12.0, color="black", ls=":", lw=0.9, alpha=0.5)
        ax.set_title(f"{PHENO_DISPLAY[p]}\n(n={ar_mat.shape[1]})",
                     fontsize=11, fontweight="bold", color=col, pad=4)
        ax.set_xlabel("hpf", fontsize=10, color=EMBL_NAVY)
        ax.set_ylabel("AR (solid)", fontsize=10, color=EMBL_NAVY)
        ax2.set_ylabel("mezzo+ frac (dashed)", fontsize=10, color=EMBL_NAVY)
        ax.tick_params(colors=EMBL_NAVY, labelsize=9)
        ax2.tick_params(colors=EMBL_NAVY, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(EMBL_NAVY); sp.set_linewidth(0.6)
        for sp in ax2.spines.values():
            sp.set_color(EMBL_NAVY); sp.set_linewidth(0.6)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    fig.text(0.5, 0.985,
              "AR and mezzo+ fraction as functions of time (peak elongation 12 hpf)",
              ha="center", va="top", fontsize=17, fontweight="bold",
              color=EMBL_NAVY)
    fig.text(0.5, 0.485,
              "Faceted by phenotype - which group drives which timing",
              ha="center", va="top", fontsize=12, fontweight="bold",
              color=EMBL_NAVY, style="italic")

    out = OUT / "ar_mezzo_vs_time.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
