"""
Clean poster versions of two figures:

  1. Phenotype distribution stacked bar (P ctrl vs P Activin)
     - legend ordered TOP-to-BOTTOM to match the visual stack order
     - readable display names, cream/navy theme, % + count labels

  2. Secondary vs primary mezzo pole area scatter
     - neater: cream/navy theme, clean y=x and y=x/2 guides,
       per-point phenotype colour, shaded "secondary < half primary" region

Reads:
  Lyn_mezzo_phase2/phenotype_summary.csv

Writes:
  Lyn_mezzo_phase5_morphometrics/plots/phenotype_distribution_clean.png
  Lyn_mezzo_phase5_morphometrics/plots/secondary_vs_primary_clean.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\plots")
OUT.mkdir(parents=True, exist_ok=True)

# Palette
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
EMBL_GREEN = "#00A689"
EMBL_RED = "#C04848"

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#386cb0", "P_Activin_3-5hpf": EMBL_RED}

# Visual stack order: BOTTOM of the bar first -> TOP of the bar last.
# We want the most "ordered/organised" phenotypes at the bottom (blue/green),
# building up to the multipolar / no-induction at the top.
STACK_ORDER = [
    "no_induction",
    "diffuse_mezzo",
    "coordinated_monopolar",
    "mezzo_bipolar_only",
    "coordinated_bipolar",
    "disorganised_multipolar",
    "multipolar",
    "oc_only",
]
PHENO_COLOR = {
    "no_induction": "#B0B0B0",
    "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0",
    "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77",
    "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a",
    "oc_only": "#e6ab02",
}
PHENO_DISPLAY = {
    "no_induction": "No induction",
    "diffuse_mezzo": "Diffuse mezzo",
    "coordinated_monopolar": "Monopolar",
    "mezzo_bipolar_only": "Mezzo bipolar",
    "coordinated_bipolar": "Coordinated bipolar",
    "disorganised_multipolar": "Disorganised multipolar",
    "multipolar": "Multipolar",
    "oc_only": "OC only",
}

plt.rcParams.update({"font.family": "Arial"})


def phenotype_distribution(df):
    counts = df.groupby(["condition", "phenotype_peak"]).size().unstack(fill_value=0)
    cats = [p for p in STACK_ORDER if p in counts.columns]
    counts = counts.reindex(columns=cats, fill_value=0)
    counts = counts.reindex([c for c in CONDITIONS if c in counts.index])
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(8.5, 8), facecolor=BG_CREAM)
    ax.set_facecolor(BG_CREAM)
    x = np.arange(len(counts.index))
    bottom = np.zeros(len(counts.index))
    seg_handles = []
    for cat in cats:                       # bottom -> top
        h = pct[cat].values
        ax.bar(x, h, bottom=bottom, color=PHENO_COLOR[cat],
                edgecolor="white", lw=1.2, width=0.62)
        for i, n_pesc in enumerate(counts[cat].values):
            if n_pesc >= 1 and h[i] >= 5:
                ax.text(x[i], bottom[i] + h[i] / 2,
                        f"{h[i]:.0f}%  (n={n_pesc})",
                        ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold")
        seg_handles.append(Patch(facecolor=PHENO_COLOR[cat],
                                  edgecolor="white",
                                  label=PHENO_DISPLAY[cat]))
        bottom = bottom + h

    ax.set_xticks(x)
    ax.set_xticklabels([f"{COND_LABEL[c]}\n(N = {int(totals.iloc[i])})"
                         for i, c in enumerate(counts.index)],
                        fontsize=14, color=EMBL_NAVY, fontweight="bold")
    ax.set_ylabel("% of pescoids", fontsize=14, color=EMBL_NAVY,
                   fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=EMBL_NAVY, labelsize=12)
    for sp in ax.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Phenotype distribution at peak elongation (12 hpf)",
                  fontsize=15, fontweight="bold", color=EMBL_NAVY, pad=12)

    # LEGEND ordered TOP-to-BOTTOM to match the visual stack: reverse the
    # bottom->top handle list so the top segment is the first legend entry.
    ax.legend(handles=list(reversed(seg_handles)),
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=12, frameon=False, labelcolor=EMBL_NAVY,
              title="Phenotype (top -> bottom of bar)",
              title_fontproperties={"weight": "bold", "size": 12},
              labelspacing=0.9, handlelength=1.6, handleheight=1.4)

    plt.tight_layout()
    out = OUT / "phenotype_distribution_clean.png"
    plt.savefig(str(out), dpi=180, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"  {out.name}")


def secondary_vs_primary(df):
    sub = df[(df.primary_mezzo_area_peak > 0)
              & (df.secondary_mezzo_area_peak > 0)].copy()
    n = len(sub)
    mx = float(sub.primary_mezzo_area_peak.max()) * 1.08

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_CREAM)
    ax.set_facecolor(BG_CREAM)

    # shaded "secondary < half primary" region (below the y = x/2 line)
    ax.fill_between([0, mx], [0, 0], [0, mx / 2],
                     color=EMBL_GREEN, alpha=0.08, zorder=0)
    # guide lines
    ax.plot([0, mx], [0, mx], ls="--", color=EMBL_NAVY, lw=1.3, alpha=0.5,
             label="y = x  (equal poles)", zorder=1)
    ax.plot([0, mx], [0, mx / 2], ls=":", color=EMBL_NAVY, lw=1.6, alpha=0.6,
             label="y = x / 2  (secondary = half primary)", zorder=1)

    # points coloured by phenotype
    seen = set()
    for _, row in sub.iterrows():
        pheno = row.get("phenotype_peak", "")
        c = PHENO_COLOR.get(pheno, "#888")
        lbl = PHENO_DISPLAY.get(pheno, pheno) if pheno not in seen else None
        seen.add(pheno)
        ax.scatter(row.primary_mezzo_area_peak, row.secondary_mezzo_area_peak,
                    s=130, color=c, edgecolors=EMBL_NAVY, lw=0.8, alpha=0.9,
                    zorder=3, label=lbl)

    # fraction below half-line
    below = (sub.secondary_mezzo_area_peak
             < 0.5 * sub.primary_mezzo_area_peak).sum()
    ax.text(0.97, 0.04,
            f"{below}/{n} secondary poles are < half the primary area",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, fontweight="bold", color=EMBL_GREEN,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec=EMBL_GREEN, lw=1.0))

    ax.set_xlabel("Primary pole area at peak [px]", fontsize=13,
                   color=EMBL_NAVY, fontweight="bold")
    ax.set_ylabel("Secondary pole area at peak [px]", fontsize=13,
                   color=EMBL_NAVY, fontweight="bold")
    ax.set_title("Secondary pole is consistently smaller than the primary",
                  fontsize=14, fontweight="bold", color=EMBL_NAVY, pad=10)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=EMBL_NAVY, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="white", lw=1.0, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False, labelcolor=EMBL_NAVY,
              loc="upper left")

    plt.tight_layout()
    out = OUT / "secondary_vs_primary_clean.png"
    plt.savefig(str(out), dpi=180, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"  {out.name}")


def main():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    df = df[df.condition.isin(CONDITIONS)].copy()
    phenotype_distribution(df)
    secondary_vs_primary(df)
    print(f"\nAll in: {OUT}")


if __name__ == "__main__":
    main()
