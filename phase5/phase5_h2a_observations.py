"""
H2A nuclear analysis - OBSERVATIONS ONLY panel (no interpretation/extrapolation).

Three sub-panels, all purely descriptive:
  A. Nuclear flow divergence inside vs outside the mezzo+ pole, at pole-formation
     frames. Negative = inward flow (convergence). Paired per pescoid.
  B. Of the cells that end inside a pole, fraction recruited from outside the
     pole region vs born locally. Control vs Activin.
  C. Cell-division (mitosis) counts inside vs outside the pole region. Paired.

Dataset: 260512 mezzo:GFP x H2A:mCherry (nuclei).
Labels: Control (blue) vs Activin (orange). Fonts >= 18 pt.

Reads:
  mezzo_H2A_phase5b1/tables/phase5b1_per_pescoid_summary.csv
  mezzo_H2A_phase5b2/tables/phase5b2_per_pescoid_summary.csv
  mezzo_H2A_phase5b3_v3/tables/phase5b3_v3_per_pescoid_summary.csv

Writes:
  Lyn_mezzo_phase5_morphometrics/plots/h2a_observations.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"Z:\Megha_Kattimani\Full_pipeline test")
B1 = BASE / "mezzo_H2A_phase5b1" / "tables" / "phase5b1_per_pescoid_summary.csv"
B2 = BASE / "mezzo_H2A_phase5b2" / "tables" / "phase5b2_per_pescoid_summary.csv"
B3 = BASE / "mezzo_H2A_phase5b3_v3" / "tables" / "phase5b3_v3_per_pescoid_summary.csv"
OUT = BASE / "Lyn_mezzo_phase5_morphometrics" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Palette + fonts
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
CONTROL_BLUE = "#2C6FB0"
ACTIVIN_ORANGE = "#E67E22"
COND_ORDER = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "Control", "P_Activin_3-5hpf": "Activin"}
COND_COLOR = {"P_ctrl": CONTROL_BLUE, "P_Activin_3-5hpf": ACTIVIN_ORANGE}

FS_TITLE = 19
FS_AXIS = 18
FS_TICK = 16
FS_ANNOT = 16

plt.rcParams.update({"font.family": "Arial"})


def _style(ax):
    ax.set_facecolor(BG_CREAM)
    ax.tick_params(colors=EMBL_NAVY, labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="white", lw=1.0, alpha=0.7)
    ax.set_axisbelow(True)


def panel_divergence(ax, rng):
    df = pd.read_csv(str(B1))
    # paired inside vs outside at formation frames
    for ci, cond in enumerate(COND_ORDER):
        sub = df[df.condition == cond]
        x_in = ci * 2 + 0
        x_out = ci * 2 + 0.8
        col = COND_COLOR[cond]
        for _, r in sub.iterrows():
            ax.plot([x_in, x_out], [r.div_in_form, r.div_out_form],
                    "-", color=col, alpha=0.30, lw=0.9, zorder=2)
        ax.scatter(np.full(len(sub), x_in) + rng.uniform(-0.08, 0.08, len(sub)),
                   sub.div_in_form, s=60, color=col, edgecolors=EMBL_NAVY,
                   lw=0.5, zorder=3)
        ax.scatter(np.full(len(sub), x_out) + rng.uniform(-0.08, 0.08, len(sub)),
                   sub.div_out_form, s=60, color=col, edgecolors=EMBL_NAVY,
                   lw=0.5, marker="s", zorder=3)
        # mean bars
        ax.plot([x_in - 0.18, x_in + 0.18],
                [sub.div_in_form.mean()] * 2, color=EMBL_NAVY, lw=3, zorder=4)
        ax.plot([x_out - 0.18, x_out + 0.18],
                [sub.div_out_form.mean()] * 2, color=EMBL_NAVY, lw=3, zorder=4)
    ax.axhline(0, color=EMBL_NAVY, ls="--", lw=1.2, alpha=0.6)
    ax.text(0.5, ax.get_ylim()[0] * 0.92, "negative = inward flow",
            fontsize=FS_ANNOT - 1, color=EMBL_NAVY, style="italic", ha="center")
    ax.set_xticks([0.4, 2.4])
    ax.set_xticklabels([f"Control\n(n={int((df.condition=='P_ctrl').sum())})",
                         f"Activin\n(n={int((df.condition=='P_Activin_3-5hpf').sum())})"],
                        fontsize=FS_AXIS, fontweight="bold", color=EMBL_NAVY)
    ax.set_ylabel("Nuclear flow divergence", fontsize=FS_AXIS,
                  color=EMBL_NAVY, fontweight="bold")
    ax.set_title("A.  Nuclear flow inside vs outside the pole\n"
                 "(circle = inside, square = outside)",
                 fontsize=FS_TITLE, fontweight="bold", color=EMBL_NAVY, pad=8)
    _style(ax)


def panel_recruitment(ax, rng):
    df = pd.read_csv(str(B2))
    for ci, cond in enumerate(COND_ORDER):
        sub = df[df.condition == cond]
        col = COND_COLOR[cond]
        x = np.full(len(sub), ci) + rng.uniform(-0.14, 0.14, len(sub))
        ax.scatter(x, sub.recruited_fraction, s=80, color=col,
                   edgecolors=EMBL_NAVY, lw=0.6, alpha=0.9, zorder=3)
        m = sub.recruited_fraction.mean()
        ax.plot([ci - 0.22, ci + 0.22], [m, m], color=EMBL_NAVY, lw=3.2,
                zorder=4)
        ax.text(ci, m + 0.02, f"{m:.2f}", ha="center", va="bottom",
                fontsize=FS_ANNOT, fontweight="bold", color=EMBL_NAVY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Control\n(n={int((df.condition=='P_ctrl').sum())})",
                         f"Activin\n(n={int((df.condition=='P_Activin_3-5hpf').sum())})"],
                        fontsize=FS_AXIS, fontweight="bold", color=EMBL_NAVY)
    ax.set_ylabel("Fraction of pole cells\nrecruited from outside",
                  fontsize=FS_AXIS, color=EMBL_NAVY, fontweight="bold")
    ax.set_title("B.  Origin of cells ending inside the pole",
                 fontsize=FS_TITLE, fontweight="bold", color=EMBL_NAVY, pad=8)
    ax.set_xlim(-0.5, 1.5)
    _style(ax)


def panel_mitosis(ax, rng):
    df = pd.read_csv(str(B3))
    act = df[df.condition == "P_Activin_3-5hpf"]
    # paired inside vs outside (Activin only - poles defined there)
    x_in, x_out = 0, 1
    col = ACTIVIN_ORANGE
    for _, r in act.iterrows():
        ax.plot([x_in, x_out], [r.n_inside_pole, r.n_outside_pole],
                "-", color=col, alpha=0.30, lw=0.9, zorder=2)
    ax.scatter(np.full(len(act), x_in) + rng.uniform(-0.1, 0.1, len(act)),
               act.n_inside_pole, s=70, color=col, edgecolors=EMBL_NAVY,
               lw=0.5, zorder=3)
    ax.scatter(np.full(len(act), x_out) + rng.uniform(-0.1, 0.1, len(act)),
               act.n_outside_pole, s=70, color=col, edgecolors=EMBL_NAVY,
               lw=0.5, zorder=3)
    ax.plot([x_in - 0.2, x_in + 0.2],
            [act.n_inside_pole.median()] * 2, color=EMBL_NAVY, lw=3.2, zorder=4)
    ax.plot([x_out - 0.2, x_out + 0.2],
            [act.n_outside_pole.median()] * 2, color=EMBL_NAVY, lw=3.2, zorder=4)
    ax.text(x_in, act.n_inside_pole.median() + 1.5,
            f"median {act.n_inside_pole.median():.0f}", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=EMBL_NAVY)
    ax.text(x_out, act.n_outside_pole.median() + 1.5,
            f"median {act.n_outside_pole.median():.0f}", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=EMBL_NAVY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inside\npole", "Outside\npole"],
                        fontsize=FS_AXIS, fontweight="bold", color=EMBL_NAVY)
    ax.set_ylabel("Cell divisions per pescoid",
                  fontsize=FS_AXIS, color=EMBL_NAVY, fontweight="bold")
    ax.set_title(f"C.  Where cell divisions occur (Activin, n={len(act)})",
                 fontsize=FS_TITLE, fontweight="bold", color=EMBL_NAVY, pad=8)
    ax.set_xlim(-0.5, 1.5)
    _style(ax)


def main():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG_CREAM)
    panel_divergence(axes[0], rng)
    panel_recruitment(axes[1], rng)
    panel_mitosis(axes[2], rng)
    fig.suptitle("Nuclear (H2A) dynamics during pole formation - "
                 "observations (mezzo:GFP x H2A:mCherry)",
                 fontsize=22, fontweight="bold", color=EMBL_NAVY, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "h2a_observations.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
