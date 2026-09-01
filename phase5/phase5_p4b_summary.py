"""
Phase 5 - clean summary of the Phase 4 / 4b mechanics findings for the poster.

Three panels, one figure each + a unified mechanics figure for the poster:
  10_coh_inside_vs_outside.png    consequence: at peak time, inside the pole
                                  is less coherent than outside.
  11_pre_emergence_coh.png        cause-vs-consequence: BEFORE the pole emerges,
                                  the future site already has elevated coherence
                                  compared to random sites in the same pescoid.
  12_angular_separation.png       angular separation between pole pairs by
                                  phenotype - tests the "no global axis"
                                  prediction (uniform distribution if no axis).
  13_mechanics_combined_for_poster.png    3-in-1 panel for the poster.

Reads from:
  Lyn_mezzo_phase4/tables/phase4b_a_per_pescoid.csv  (inside vs outside)
  Lyn_mezzo_phase4/tables/phase4b_b_per_pole.csv     (pre-emergence)
  Lyn_mezzo_phase4/tables/phase4b_c_pole_pairs.csv   (angular separation)
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

COND_ORDER = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#386cb0", "P_Activin_3-5hpf": "#C04848"}
PHENO_COLOR = {
    "no_induction": "#999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77", "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a", "oc_only": "#e6ab02",
}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
                     "axes.grid": True, "grid.color": "white", "font.size": 11})


# ---------------------------------------------------------------------------
def panel_inside_vs_outside(ax, df, rng):
    """Paired plot per pescoid: coh inside pole vs coh outside pole at peak.
    Only pescoids with a detected pole (inside value exists) are shown."""
    sub = df.dropna(subset=["coh_inside_pole_peak", "coh_outside_pole_peak"]).copy()
    n_total = len(sub)
    if n_total == 0:
        ax.text(0.5, 0.5, "no inside-pole data", transform=ax.transAxes,
                 ha="center")
        return
    inside_lt_outside = (sub["coh_inside_pole_peak"]
                         < sub["coh_outside_pole_peak"]).sum()
    # paired lines
    for _, row in sub.iterrows():
        c = PHENO_COLOR.get(row.get("phenotype_peak", ""), "#888")
        ax.plot([0, 1],
                [row["coh_inside_pole_peak"], row["coh_outside_pole_peak"]],
                "-", color=c, alpha=0.55, lw=1.0)
        ax.scatter([0, 1],
                   [row["coh_inside_pole_peak"], row["coh_outside_pole_peak"]],
                   s=55, color=c, edgecolors="black", lw=0.5, zorder=3)
    # paired Wilcoxon
    w, p = stats.wilcoxon(sub["coh_inside_pole_peak"],
                           sub["coh_outside_pole_peak"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["inside pole", "outside pole"])
    ax.set_ylabel("cortical coherence (LynTom structure tensor)")
    ax.set_title(f"Inside < outside in {inside_lt_outside} / {n_total} pescoids\n"
                 f"Wilcoxon p = {p:.3g}", fontweight="bold", fontsize=11)


def panel_pre_emergence(ax, df_pole, rng):
    """Per-pole delta_site_minus_random, by condition. Positive = future pole
    site has higher cortical coherence than random sites BEFORE emergence."""
    sub = df_pole.dropna(subset=["delta_site_minus_random"]).copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no pre-emergence data", transform=ax.transAxes,
                 ha="center")
        return
    vals = {c: sub.loc[sub["condition"] == c,
                         "delta_site_minus_random"].values
              for c in COND_ORDER}
    positions = list(range(len(COND_ORDER)))
    parts = ax.violinplot([vals[c] for c in COND_ORDER], positions=positions,
                           widths=0.7, showmeans=False, showmedians=False,
                           showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(COND_COLOR[COND_ORDER[i]]); b.set_alpha(0.35)
        b.set_edgecolor("black")
    for i, c in enumerate(COND_ORDER):
        v = vals[c]
        if len(v) == 0:
            continue
        x = positions[i] + rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(x, v, s=55, color=COND_COLOR[c], edgecolors="black", lw=0.6,
                   zorder=3, alpha=0.9)
        med = np.median(v)
        ax.plot([positions[i] - 0.25, positions[i] + 0.25], [med, med],
                color="black", lw=2.5, zorder=4)
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    ax.set_ylabel("delta = site - random  (pre-emergence)")
    pos_frac = (sub["delta_site_minus_random"] > 0).mean()
    n_acti = sub[sub["condition"] == "P_Activin_3-5hpf"]
    if len(n_acti) >= 3:
        w, p = stats.wilcoxon(n_acti["delta_site_minus_random"])
        title = (f"Pre-emergence coherence at future pole site\n"
                 f"vs random sites. Activin: {pos_frac:.0%} above 0 "
                 f"(Wilcoxon p = {p:.3g})")
    else:
        title = (f"Pre-emergence coherence at future pole site\n"
                 f"vs random sites. {pos_frac:.0%} above 0")
    ax.set_title(title, fontweight="bold", fontsize=11)


def panel_angular_separation(ax, df_pairs):
    """Histogram of angular separations between pole pairs, by # poles."""
    if df_pairs.empty:
        ax.text(0.5, 0.5, "no pole pairs", transform=ax.transAxes, ha="center")
        return
    pairs = df_pairs.dropna(subset=["abs_angular_separation_deg"]).copy()
    bins = np.arange(0, 181, 15)
    by_n = {}
    for n_poles, g in pairs.groupby("n_poles_in_pescoid"):
        by_n[int(n_poles)] = g["abs_angular_separation_deg"].values
    # plot 2-pole and 3+ poles separately
    if 2 in by_n:
        ax.hist(by_n[2], bins=bins, alpha=0.6, color="#7570b3",
                 edgecolor="black", label=f"2 poles (n={len(by_n[2])} pairs)")
    multi_pairs = np.concatenate([v for k, v in by_n.items() if k >= 3]) \
        if any(k >= 3 for k in by_n) else np.array([])
    if multi_pairs.size:
        ax.hist(multi_pairs, bins=bins, alpha=0.6, color="#e7298a",
                 edgecolor="black",
                 label=f">=3 poles (n={multi_pairs.size} pairs)")
    ax.axvline(180, color="k", ls="--", lw=1, alpha=0.6,
               label="antipodal (180 deg)")
    ax.set_xlabel("angular separation between pole pair (deg)")
    ax.set_ylabel("# pole pairs")
    ax.set_title("Pole-pair angular separation\n"
                 "broad distribution = no global axis constraint",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 200)


def main():
    p4b_a = pd.read_csv(str(P4 / "phase4b_a_per_pescoid.csv"))
    p4b_b = pd.read_csv(str(P4 / "phase4b_b_per_pole.csv"))
    p4b_c = pd.read_csv(str(P4 / "phase4b_c_pole_pairs.csv"))
    rng = np.random.default_rng(42)

    # Individual panels
    for name, fn, arg in [
        ("10_coh_inside_vs_outside.png", panel_inside_vs_outside, (p4b_a, rng)),
        ("11_pre_emergence_coh.png", panel_pre_emergence, (p4b_b, rng)),
        ("12_angular_separation.png", panel_angular_separation, (p4b_c,)),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6.5))
        fn(ax, *arg)
        plt.tight_layout()
        plt.savefig(str(OUT / name), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}")

    # 3-in-1 combined panel for the poster
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    panel_inside_vs_outside(axes[0], p4b_a, rng)
    panel_pre_emergence(axes[1], p4b_b, rng)
    panel_angular_separation(axes[2], p4b_c)
    plt.suptitle("Cortical mechanics around mezzo+ poles - consequence, "
                 "cause-vs-consequence, and pole-pair geometry",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(OUT / "13_mechanics_combined_for_poster.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  13_mechanics_combined_for_poster.png")


if __name__ == "__main__":
    main()
