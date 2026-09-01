"""
Two clean poster figures for the LynTom cortical-mechanics story.

FIG 1 - pre-emergence:
  At FUTURE pole sites, cortical alignment (structure-tensor coherence) is
  elevated vs random sites in the same pescoid, BEFORE mezzo signal emerges.
  delta = coherence(future pole site) - coherence(random sites), per pole.
  Activin: +0.06 units mean (n=23 pescoids / 47 poles).

FIG 2 - peak:
  At PEAK elongation, cortical alignment is LOWER inside mezzo+ poles than
  in the surrounding tissue. Paired per pescoid (n=25: 23 Activin + 2 ctrl);
  inside < outside in 18/25, Wilcoxon p = 0.0025.

Reads:
  Lyn_mezzo_phase4/tables/phase4b_b_per_pole.csv     (pre-emergence)
  Lyn_mezzo_phase4/tables/phase4b_a_per_pescoid.csv  (peak inside vs outside)

Writes (cream/navy poster theme):
  Lyn_mezzo_phase5_morphometrics/plots/cortical_preemergence.png
  Lyn_mezzo_phase5_morphometrics/plots/cortical_peak_inside_outside.png
  Lyn_mezzo_phase5_morphometrics/plots/cortical_combined.png  (both side by side)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P4 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase4\tables")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\plots")
OUT.mkdir(parents=True, exist_ok=True)

# Palette
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
EMBL_GREEN = "#00A689"
EMBL_RED = "#C04848"
# Nick: blue = control, orange = Activin treatment. Drop "P" / "early pulse".
CONTROL_BLUE = "#2C6FB0"
ACTIVIN_ORANGE = "#E67E22"
COND_COLOR = {"P_ctrl": CONTROL_BLUE, "P_Activin_3-5hpf": ACTIVIN_ORANGE}
COND_LABEL = {"P_ctrl": "Control", "P_Activin_3-5hpf": "Activin"}

# Poster font floor (Nick: minimum 18 pt on a poster)
FS_TITLE = 20
FS_AXIS = 18
FS_TICK = 16
FS_ANNOT = 17

PHENO_COLOR = {
    "no_induction": "#999999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77", "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a", "oc_only": "#e6ab02",
}

plt.rcParams.update({"font.family": "Arial"})


# ---------------------------------------------------------------------------
def panel_preemergence(ax):
    b = pd.read_csv(str(P4 / "phase4b_b_per_pole.csv"))
    b = b.dropna(subset=["delta_site_minus_random"])
    act = b[b.condition == "P_Activin_3-5hpf"]["delta_site_minus_random"].values
    n_poles = len(act)
    n_pescoids = b[b.condition == "P_Activin_3-5hpf"]["pescoid"].nunique()
    mean_delta = float(np.mean(act))
    frac_pos = float((act > 0).mean())
    w, p = stats.wilcoxon(act)

    rng = np.random.default_rng(7)
    x = rng.uniform(-0.16, 0.16, size=n_poles)
    # violin
    parts = ax.violinplot([act], positions=[0], widths=0.55,
                           showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(EMBL_RED); body.set_alpha(0.22)
        body.set_edgecolor(EMBL_NAVY)
    ax.scatter(x, act, s=70, color=EMBL_RED, edgecolors=EMBL_NAVY, lw=0.6,
               alpha=0.85, zorder=3)
    # zero reference + mean
    ax.axhline(0, color=EMBL_NAVY, ls="--", lw=1.3, alpha=0.6)
    ax.plot([-0.3, 0.3], [mean_delta, mean_delta], color=EMBL_GREEN, lw=3.5,
            zorder=4)
    ax.annotate(f"mean = +{mean_delta:.2f}",
                xy=(0.3, mean_delta), xytext=(0.42, mean_delta + 0.04),
                fontsize=13, fontweight="bold", color=EMBL_GREEN,
                arrowprops=dict(arrowstyle="-", color=EMBL_GREEN, lw=1.5))
    ax.set_xticks([0])
    ax.set_xticklabels([f"P Activin 3-5h\n({n_pescoids} pescoids, "
                         f"{n_poles} poles)"],
                        fontsize=12, color=EMBL_NAVY, fontweight="bold")
    ax.set_ylabel("delta coherence\n(future pole site  -  random sites)",
                  fontsize=13, color=EMBL_NAVY, fontweight="bold")
    ax.set_title("BEFORE mezzo emerges, future pole sites are\n"
                 "more cortically aligned than random tissue",
                 fontsize=13, fontweight="bold", color=EMBL_NAVY, pad=10)
    # annotation box
    ax.text(0.02, 0.98,
            f"{frac_pos:.0%} of poles above 0\n"
            f"mean +{mean_delta:.2f} units\n"
            f"(per-pole Wilcoxon p = {p:.3f})",
            transform=ax.transAxes, fontsize=11, va="top", ha="left",
            color=EMBL_NAVY,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec=EMBL_NAVY, lw=1.0))
    ax.set_xlim(-0.7, 0.9)
    _style(ax)


def panel_peak(ax):
    a = pd.read_csv(str(P4 / "phase4b_a_per_pescoid.csv"))
    sub = a.dropna(subset=["coh_inside_pole_peak", "coh_outside_pole_peak"]).copy()
    n = len(sub)
    inside_lt = int((sub.coh_inside_pole_peak < sub.coh_outside_pole_peak).sum())
    w, p = stats.wilcoxon(sub.coh_inside_pole_peak, sub.coh_outside_pole_peak)

    for _, row in sub.iterrows():
        c = PHENO_COLOR.get(row.get("phenotype_peak", ""), "#888")
        ax.plot([0, 1],
                [row.coh_inside_pole_peak, row.coh_outside_pole_peak],
                "-", color=c, alpha=0.5, lw=1.3, zorder=2)
        ax.scatter([0, 1],
                   [row.coh_inside_pole_peak, row.coh_outside_pole_peak],
                   s=70, color=c, edgecolors=EMBL_NAVY, lw=0.5, zorder=3)
    # mean markers
    mi = sub.coh_inside_pole_peak.mean()
    mo = sub.coh_outside_pole_peak.mean()
    ax.plot([0, 1], [mi, mo], "-", color=EMBL_NAVY, lw=3.4, zorder=5)
    ax.scatter([0, 1], [mi, mo], s=200, color=EMBL_NAVY, marker="D",
               zorder=6, label="mean")
    ax.text(0, mi - 0.045, f"{mi:.2f}", ha="center", va="top",
            fontsize=FS_ANNOT, fontweight="bold", color=EMBL_NAVY)
    ax.text(1, mo + 0.035, f"{mo:.2f}", ha="center", va="bottom",
            fontsize=FS_ANNOT, fontweight="bold", color=EMBL_NAVY)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inside\nmezzo+ pole", "Outside\n(surrounding tissue)"],
                        fontsize=FS_AXIS, color=EMBL_NAVY, fontweight="bold")
    ax.set_ylabel("Cortical coherence\n(membrane structure tensor)",
                  fontsize=FS_AXIS, color=EMBL_NAVY, fontweight="bold")
    ax.set_title("At peak elongation, the cortex inside mezzo+ poles\n"
                 "is LESS aligned than the surrounding tissue",
                 fontsize=FS_TITLE, fontweight="bold", color=EMBL_NAVY, pad=10)
    ax.text(0.02, 0.02,
            f"inside < outside in {inside_lt}/{n} pescoids\n"
            f"(Wilcoxon p = {p:.4f})",
            transform=ax.transAxes, fontsize=FS_ANNOT, va="bottom", ha="left",
            color=EMBL_NAVY,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec=EMBL_NAVY, lw=1.0))
    ax.set_xlim(-0.4, 1.4)
    _style(ax)


def _style(ax):
    ax.set_facecolor(BG_CREAM)
    ax.tick_params(colors=EMBL_NAVY, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="white", lw=1.0, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    # Fig 1 standalone
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG_CREAM)
    panel_preemergence(ax)
    plt.tight_layout()
    plt.savefig(str(OUT / "cortical_preemergence.png"), dpi=180,
                bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print("  cortical_preemergence.png")

    # Fig 2 standalone
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG_CREAM)
    panel_peak(ax)
    plt.tight_layout()
    plt.savefig(str(OUT / "cortical_peak_inside_outside.png"), dpi=180,
                bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print("  cortical_peak_inside_outside.png")

    # Combined side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor=BG_CREAM)
    panel_preemergence(axes[0])
    panel_peak(axes[1])
    fig.suptitle("Cortical mechanics around mezzo+ organisers (LynTom dataset)",
                 fontsize=16, fontweight="bold", color=EMBL_NAVY, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(str(OUT / "cortical_combined.png"), dpi=180,
                bbox_inches="tight", facecolor=BG_CREAM)
    plt.close(fig)
    print("  cortical_combined.png")
    print(f"\nAll in: {OUT}")


if __name__ == "__main__":
    main()
