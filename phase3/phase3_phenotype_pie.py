"""
Polarity phenotype distribution, all experiments compiled into ONE donut.

Every pescoid is classed by how many mezzo+ poles it forms at peak:
  1 pole   -> Monopolar
  2 poles  -> Bipolar
  >=3 poles -> Multipolar
Pescoids that never form a pole are reported in the centre, not as a slice.

Pooled across all analysed experiments:
  Lyn_mezzo, mezzo_H2A, mezzo_titration  (all conditions)

Simple white background, big labels, no clutter.

Reads:
  <Lyn_mezzo|mezzo_H2A|mezzo_titration>_phase2/phenotype_summary.csv  (on Z:)

Writes:
  <PHASE5_PLOTS>/phenotype_polarity_pie.png
  ./phenotype_polarity_pie.png   (local copy for quick viewing)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"Z:\Megha_Kattimani\Full_pipeline test")
PHASE5_PLOTS = BASE / "Lyn_mezzo_phase5_morphometrics" / "plots"
PHASE5_PLOTS.mkdir(parents=True, exist_ok=True)
LOCAL_OUT = Path("phenotype_polarity_pie.png")

FILES = [
    "Lyn_mezzo_phase2/phenotype_summary.csv",
    "mezzo_H2A_phase2/phenotype_summary.csv",
    "mezzo_titration_phase2/phenotype_summary.csv",
]
NPOLES = "n_mezzo_poles_peak"

# Simple, distinct colours (increasing polarity: green -> blue -> orange)
INK = "#222222"
COLORS = {"Monopolar": "#4C9F70", "Bipolar": "#2C5BAA", "Multipolar": "#E8820C"}
ORDER = ["Monopolar", "Bipolar", "Multipolar"]


def classify(n):
    if n >= 3:
        return "Multipolar"
    if n == 2:
        return "Bipolar"
    if n == 1:
        return "Monopolar"
    return "No pole"


def main():
    df = pd.concat([pd.read_csv(str(BASE / f)) for f in FILES], ignore_index=True)
    df["class"] = df[NPOLES].apply(classify)

    n_total = len(df)
    n_none = int((df["class"] == "No pole").sum())
    counts = [int((df["class"] == c).sum()) for c in ORDER]
    n_pol = sum(counts)

    plt.rcParams.update({"font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(9, 8.4), facecolor="white")
    ax.set_facecolor("white")

    wedges, _texts, autotexts = ax.pie(
        counts,
        colors=[COLORS[c] for c in ORDER],
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=3),
        autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct/100*n_pol))})",
        pctdistance=0.79,
        textprops=dict(fontsize=15, fontweight="bold", color="white"),
    )

    # category labels just outside each slice, anchored by side so they never
    # overlap the ring
    for w, c in zip(wedges, ORDER):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        cos, sin = np.cos(ang), np.sin(ang)
        x, y = 1.06 * cos, 1.08 * sin
        if abs(cos) < 0.25:
            ha = "center"
        else:
            ha = "left" if cos > 0 else "right"
        ax.text(x, y, c, ha=ha, va="center", fontsize=19,
                fontweight="bold", color=COLORS[c])

    # centre summary
    ax.text(0, 0.10, f"{n_pol}", ha="center", va="center", fontsize=40,
            fontweight="bold", color=INK)
    ax.text(0, -0.13, "pescoids\nformed a pole", ha="center", va="center",
            fontsize=14, color="#555555")

    ax.set_title("Polarity phenotype across all experiments",
                 fontsize=22, fontweight="bold", color=INK, pad=18)
    fig.text(0.5, 0.04,
             f"pooled across 3 experiments  -  {n_pol} of {n_total} pescoids "
             f"formed >=1 pole ({n_none} formed none)",
             ha="center", va="bottom", fontsize=13, color="#555555")

    ax.set(aspect="equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.45, 1.5)
    for out in (PHASE5_PLOTS / "phenotype_polarity_pie.png", LOCAL_OUT):
        plt.savefig(str(out), dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {out}")
    plt.close(fig)

    print(f"\nPolarity classes (pooled, n_polarized={n_pol}; no-pole={n_none}; total={n_total}):")
    for c, v in zip(ORDER, counts):
        print(f"  {c:12s} {v:3d}  ({100*v/n_pol:.0f}% of polarized)")


if __name__ == "__main__":
    main()
