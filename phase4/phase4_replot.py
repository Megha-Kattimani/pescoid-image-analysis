"""
Phase 4 — fast re-plot of cross-condition figures using already-saved
tension data, after phenotype labels were updated (e.g., new diffuse_mezzo cat).

Reads from existing:
  Lyn_mezzo_phase4/tables/tension_per_pescoid.csv
  Lyn_mezzo_phase4/tables/tension_per_frame.csv
  Lyn_mezzo_phase2/phenotype_summary.csv (updated)

Regenerates only the cross-condition plots in Lyn_mezzo_phase4/plots/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
PHASE4 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase4")
PLOTS = PHASE4 / "plots"
TABLES = PHASE4 / "tables"

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 6.0
HPF_INTERVAL = 0.4
PEAK_HPF = (11.0, 15.0)
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


def jitter(n, w=0.15):
    return RNG.uniform(-w, w, size=n)


def swarm_box(ax, df, ycol, xcol="condition", colors=COND_COLOR, labels=COND_LABEL,
              hline=None, ylim=None):
    cond_order = [c for c in CONDITIONS if c in df[xcol].unique()]
    groups = [df[df[xcol] == c][ycol].dropna().values for c in cond_order]
    bp = ax.boxplot(groups, positions=range(len(cond_order)), widths=0.5,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(alpha=0.35), medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], cond_order):
        patch.set_facecolor(colors[c])
    for i, (c, vals) in enumerate(zip(cond_order, groups)):
        x = i + jitter(len(vals))
        ax.scatter(x, vals, s=55, color=colors[c], edgecolors="black", lw=0.6,
                   alpha=0.85, zorder=3)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([f"{labels[c]}\nn={len(g)}" for c, g in zip(cond_order, groups)])
    if hline is not None:
        ax.axhline(hline, ls="--", color="k", lw=1, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)


def paired_inside_outside(df, ycol_in, ycol_out, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    cond_pos = {"P_ctrl": 0, "P_Activin_3-5hpf": 1}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].dropna(subset=[ycol_in, ycol_out])
        if sub.empty:
            continue
        x_base = cond_pos[cond]
        x_in = x_base - 0.2 + jitter(len(sub), 0.06)
        x_out = x_base + 0.2 + jitter(len(sub), 0.06)
        for vi, vo, xi, xo in zip(sub[ycol_in].values, sub[ycol_out].values, x_in, x_out):
            ax.plot([xi, xo], [vi, vo], "-", color=COND_COLOR[cond], alpha=0.35, lw=0.8)
        ax.scatter(x_in, sub[ycol_in], s=60, color="lime", edgecolors="black", lw=0.6,
                    label="inside mezzo+ pole" if cond == CONDITIONS[0] else None, zorder=3)
        ax.scatter(x_out, sub[ycol_out], s=60, color="#888", edgecolors="black", lw=0.6,
                    label="outside (rest of mask)" if cond == CONDITIONS[0] else None, zorder=3)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values, n=500):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([np.mean(RNG.choice(values, size=len(values), replace=True)) for _ in range(n)])
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def trajectory_plot(traj_df, ycol, ylabel, title, fname, clip_zero=True, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    for cond in CONDITIONS:
        sub = traj_df[traj_df["condition"] == cond]
        for pid, g in sub.groupby("pescoid"):
            ax.plot(g["hpf"], g[ycol], color=COND_COLOR[cond], lw=0.6, alpha=0.25)
    for cond in CONDITIONS:
        sub = traj_df[traj_df["condition"] == cond]
        if sub.empty:
            continue
        xs, ms, los, his = [], [], [], []
        for x, g in sub.groupby("hpf"):
            m, lo, hi = bootstrap_ci(g[ycol].dropna().values)
            xs.append(x); ms.append(m); los.append(lo); his.append(hi)
        xs = np.array(xs); ms = np.array(ms); los = np.array(los); his = np.array(his)
        if clip_zero:
            ms = np.clip(ms, 0, None); los = np.clip(los, 0, None); his = np.clip(his, 0, None)
        ax.fill_between(xs, los, his, color=COND_COLOR[cond], alpha=0.2)
        ax.plot(xs, ms, color=COND_COLOR[cond], lw=3, label=f"{COND_LABEL[cond]} (mean)")
    ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.10, label="Peak window")
    ax.set_xlabel("Time [hpf]"); ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  Phase 4 — re-plot from cached tension data")
    print("=" * 60)

    pdf = pd.read_csv(str(TABLES / "tension_per_pescoid.csv"))
    traj = pd.read_csv(str(TABLES / "tension_per_frame.csv"))
    summary_p2 = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))

    # Merge updated phenotype labels into the tension per-pescoid dataframe
    cols = ["condition", "pescoid", "phenotype_peak", "phenotype_endstate", "n_mezzo_poles_peak"]
    pdf = pdf.drop(columns=[c for c in ["phenotype_peak", "phenotype_endstate", "n_mezzo_poles_peak"] if c in pdf.columns],
                    errors="ignore").merge(
        summary_p2[cols], on=["condition", "pescoid"], how="left"
    )

    # Plot 01: Global Q at peak
    fig, ax = plt.subplots(figsize=(7, 6))
    swarm_box(ax, pdf, "Q_peak_window_mean", ylim=(0, 1))
    ax.set_ylabel("Q nematic (peak-window mean)")
    ax.set_title("Global tissue alignment at peak window", fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_global_Q_at_peak_swarm.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  01_global_Q_at_peak_swarm.png")

    # Plot 02: Q stratified by phenotype (now includes diffuse_mezzo)
    fig, ax = plt.subplots(figsize=(14, 6))
    phenos_present = [p for p in pdf["phenotype_peak"].dropna().unique()]
    groups = []
    labels = []
    cond_colours = []
    for cond in CONDITIONS:
        for p in phenos_present:
            sub = pdf[(pdf["condition"] == cond) & (pdf["phenotype_peak"] == p)]
            if sub.empty:
                continue
            groups.append(sub["Q_peak_window_mean"].dropna().values)
            labels.append(f"{COND_LABEL[cond]}\n{p}\nn={len(sub)}")
            cond_colours.append(COND_COLOR[cond])
    if groups:
        positions = list(range(len(groups)))
        bp = ax.boxplot(groups, positions=positions, widths=0.5, patch_artist=True,
                         showfliers=False, boxprops=dict(alpha=0.35),
                         medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], cond_colours):
            patch.set_facecolor(c)
        for i, (vals, c) in enumerate(zip(groups, cond_colours)):
            x = i + jitter(len(vals))
            ax.scatter(x, vals, s=55, color=c, edgecolors="black", lw=0.6, alpha=0.85, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Q nematic (peak-window mean)")
    ax.set_title("Tissue alignment Q stratified by phenotype", fontweight="bold")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_Q_by_phenotype.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  02_Q_by_phenotype.png")

    # Plot 03-05: trajectories from cached tension_per_frame
    trajectory_plot(traj, "Q_global", "Q nematic", "Tissue alignment Q over time",
                     "03_Q_over_time.png", ylim=(0, 1))
    print("  03_Q_over_time.png")
    trajectory_plot(traj, "mean_coherence", "Mean coherence",
                     "Mean local coherence over time", "04_coherence_over_time.png", ylim=(0, 1))
    print("  04_coherence_over_time.png")
    trajectory_plot(traj, "cortex_minus_interior", "Cortex − Interior coherence",
                     "Cortical anisotropy excess over time", "05_cortex_minus_interior.png",
                     clip_zero=False)
    print("  05_cortex_minus_interior.png")

    # Plot 06: paired inside vs outside
    paired_inside_outside(pdf, "coh_inside_mezzo_pole_peak_mean",
                            "coh_outside_mezzo_pole_peak_mean",
                            "Mean coherence",
                            "Coherence INSIDE vs OUTSIDE mezzo+ pole (peak window)\n"
                            "Lines connect paired values per pescoid",
                            "06_coherence_inside_vs_outside_paired.png")
    print("  06_coherence_inside_vs_outside_paired.png")

    # Plot 07: Q vs n_mezzo_poles
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond in CONDITIONS:
        sub = pdf[pdf["condition"] == cond]
        ax.scatter(sub["n_mezzo_poles_peak"] + jitter(len(sub), 0.08),
                    sub["Q_peak_window_mean"],
                    s=70, color=COND_COLOR[cond], edgecolors="black", lw=0.6,
                    alpha=0.85, label=f"{COND_LABEL[cond]} (n={len(sub)})")
    ax.set_xlabel("Number of mezzo poles at peak")
    ax.set_ylabel("Q nematic (peak)")
    ax.set_title("Tissue alignment vs pole multiplicity", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "07_Q_vs_n_mezzo_poles.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  07_Q_vs_n_mezzo_poles.png")

    print(f"\nDone. Plots in {PLOTS}/")


if __name__ == "__main__":
    main()
