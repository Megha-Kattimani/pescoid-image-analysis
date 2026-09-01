"""
Phase 5a cross-condition plots.

Inputs:
  - mezzo_H2A_phase5a/tables/h2a_per_frame_all.csv (from phase5a_h2a_dynamics)
  - mezzo_H2A_phase2/phenotype_summary.csv

Outputs (mezzo_H2A_phase5a/plots/):
  01_density_inside_vs_outside_paired.png    headline: do cells aggregate at poles?
  02_flow_inside_vs_outside_paired.png       headline: are poles flow hotspots?
  03_density_by_phenotype.png
  04_flow_by_phenotype.png
  05_density_trajectory_inside_vs_outside.png
  06_flow_trajectory_inside_vs_outside.png
  07_total_h2a_intensity_over_time.png
  08_nuclei_count_over_time.png
  09_nuclei_count_by_phenotype.png
  tables/phase5a_per_pescoid_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
PHASE5A = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5a")
PLOTS = PHASE5A / "plots"
TABLES = PHASE5A / "tables"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

PEAK_HPF = (12.0, 16.0)   # post-12 hpf high-confidence window (per user)
RNG = np.random.default_rng(42)

PHENO_ORDER = [
    "coordinated_monopolar", "mezzo_bipolar_only", "multipolar",
    "diffuse_mezzo", "no_induction",
]
PHENO_COLOR = {
    "coordinated_monopolar": "#386cb0",
    "mezzo_bipolar_only":    "#7570b3",
    "multipolar":            "#e7298a",
    "diffuse_mezzo":         "#66c2a5",
    "no_induction":          "#999999",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


def jitter(n, w=0.15):
    return RNG.uniform(-w, w, size=n)


def bootstrap_ci(values, n=300):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                      for _ in range(n)])
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
def load_data():
    big = pd.read_csv(str(TABLES / "h2a_per_frame_all.csv"))
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    big = big.merge(
        summary[["condition", "pescoid", "phenotype_peak", "phenotype_endstate"]],
        on=["condition", "pescoid"], how="left")
    return big, summary


def per_pescoid_peak(big):
    """Mean inside/outside metrics in the post-12-hpf high-confidence window,
    plus peak nuclei count."""
    peak = big[(big["hpf"] >= PEAK_HPF[0]) & (big["hpf"] <= PEAK_HPF[1])].copy()
    out = peak.groupby(["condition", "pescoid", "phenotype_peak"]).agg(
        density_inside=("density_inside_pole", lambda x: x.dropna().mean()),
        density_outside=("density_outside_pole", lambda x: x.dropna().mean()),
        flow_inside=("flow_mag_inside_pole", lambda x: x.dropna().mean()),
        flow_outside=("flow_mag_outside_pole", lambda x: x.dropna().mean()),
        n_nuclei_max=("n_nuclei", "max"),
        n_nuclei_mean=("n_nuclei", "mean"),
        total_h2a_peak=("total_h2a_intensity", "mean"),
    ).reset_index()
    out["density_ratio"] = out["density_inside"] / out["density_outside"]
    out["flow_ratio"] = out["flow_inside"] / out["flow_outside"]
    return out


# ---------------------------------------------------------------------------
def paired_in_vs_out(df, ycol_in, ycol_out, ylabel, title, fname,
                      use_phenotype_color=False):
    fig, ax = plt.subplots(figsize=(9, 6))
    cond_pos = {"P_ctrl": 0, "P_Activin_3-5hpf": 1}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].dropna(subset=[ycol_in, ycol_out])
        if sub.empty:
            continue
        x_base = cond_pos[cond]
        x_in = x_base - 0.2 + jitter(len(sub), 0.06)
        x_out = x_base + 0.2 + jitter(len(sub), 0.06)
        for (i, row), xi, xo in zip(sub.iterrows(), x_in, x_out):
            vi, vo = row[ycol_in], row[ycol_out]
            color = PHENO_COLOR.get(row["phenotype_peak"], "#888") if use_phenotype_color else COND_COLOR[cond]
            ax.plot([xi, xo], [vi, vo], "-", color=color, alpha=0.40, lw=0.9)
            ax.scatter([xi], [vi], s=55, color="lime", edgecolors="black", lw=0.6,
                       zorder=3)
            ax.scatter([xo], [vo], s=55, color="#888888", edgecolors="black", lw=0.6,
                       zorder=3)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="lime", lw=0, markeredgecolor="black",
               markersize=9, label="inside mezzo+ pole"),
        Line2D([0], [0], marker="o", color="#888", lw=0, markeredgecolor="black",
               markersize=9, label="outside (rest of mask)"),
    ]
    ax.legend(handles=handles, loc="best")
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def stratified_by_phenotype(df, ycol, ylabel, title, fname, ylim=None, hline=None):
    """Box+swarm of a metric across phenotypes (per condition)."""
    fig, ax = plt.subplots(figsize=(13, 6))
    positions = []
    labels = []
    colors = []
    groups = []
    pos = 0
    for cond in CONDITIONS:
        for pheno in PHENO_ORDER:
            sub = df[(df["condition"] == cond) & (df["phenotype_peak"] == pheno)]
            v = sub[ycol].dropna().values
            if v.size == 0:
                continue
            groups.append(v)
            positions.append(pos)
            labels.append(f"{COND_LABEL[cond]}\n{pheno}\nn={len(v)}")
            colors.append(PHENO_COLOR[pheno])
            pos += 1
        pos += 0.8
    if groups:
        bp = ax.boxplot(groups, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(alpha=0.40),
                        medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        for i, (vals, c) in enumerate(zip(groups, colors)):
            x = positions[i] + jitter(len(vals), 0.1)
            ax.scatter(x, vals, s=45, color=c, edgecolors="black", lw=0.4,
                       alpha=0.85, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if hline is not None:
        ax.axhline(hline, ls="--", color="k", lw=1, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def trajectory_plot(big, ycol, ylabel, title, fname, ylim=None, clip_zero=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    for cond in CONDITIONS:
        sub = big[big["condition"] == cond]
        for pid, g in sub.groupby("pescoid"):
            ax.plot(g["hpf"], g[ycol], color=COND_COLOR[cond], lw=0.6, alpha=0.25)
    for cond in CONDITIONS:
        sub = big[big["condition"] == cond]
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
        ax.plot(xs, ms, color=COND_COLOR[cond], lw=2.5, label=f"{COND_LABEL[cond]} (mean)")
    ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.10,
                label=f"Peak window {PEAK_HPF[0]}-{PEAK_HPF[1]} hpf")
    ax.set_xlabel("Time [hpf]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def density_traj_in_vs_out(big, fname, ycol_in="density_inside_pole",
                             ycol_out="density_outside_pole",
                             ylabel="Density (smoothed H2A intensity)",
                             title="Density inside mezzo+ pole vs outside, over time"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axi, cond in enumerate(CONDITIONS):
        ax = axes[axi]
        sub = big[big["condition"] == cond]
        for series, color, label in [(ycol_in, "lime", "inside"),
                                      (ycol_out, "#888", "outside")]:
            xs, ms, los, his = [], [], [], []
            for x, g in sub.groupby("hpf"):
                m, lo, hi = bootstrap_ci(g[series].dropna().values)
                xs.append(x); ms.append(m); los.append(lo); his.append(hi)
            xs = np.array(xs); ms = np.array(ms); los = np.array(los); his = np.array(his)
            ms = np.clip(ms, 0, None); los = np.clip(los, 0, None); his = np.clip(his, 0, None)
            ax.fill_between(xs, los, his, color=color, alpha=0.25)
            ax.plot(xs, ms, color=color, lw=2.5, label=label)
        ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.10)
        ax.set_xlabel("Time [hpf]")
        ax.set_title(f"{COND_LABEL[cond]}", fontweight="bold")
        ax.legend()
    axes[0].set_ylabel(ylabel)
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Phase 5a cross-condition plots")
    print("=" * 60)
    big, summary = load_data()
    print(f"  trajectory rows: {len(big)}")
    print(f"  pescoids in summary: {len(summary)}")

    per_p = per_pescoid_peak(big)
    per_p.to_csv(str(TABLES / "phase5a_per_pescoid_summary.csv"), index=False)
    print(f"  per-pescoid summary: {len(per_p)} pescoids")

    # 01: paired density inside vs outside
    paired_in_vs_out(per_p, "density_inside", "density_outside",
                      "Density (smoothed H2A intensity)",
                      "Density INSIDE vs OUTSIDE mezzo+ pole\n"
                      "(mean over 12-16 hpf; lines connect paired values per pescoid)",
                      "01_density_inside_vs_outside_paired.png")
    print("  01_density_inside_vs_outside_paired.png")

    # 02: paired flow inside vs outside
    paired_in_vs_out(per_p, "flow_inside", "flow_outside",
                      "PIV flow magnitude [px/frame]",
                      "PIV flow magnitude INSIDE vs OUTSIDE mezzo+ pole\n"
                      "(mean over 12-16 hpf; lines connect paired values per pescoid)",
                      "02_flow_inside_vs_outside_paired.png")
    print("  02_flow_inside_vs_outside_paired.png")

    # 03: density ratio (in/out) by phenotype
    stratified_by_phenotype(per_p, "density_ratio",
                             "Density inside / outside",
                             "Density ratio (inside mezzo+ pole / outside) by phenotype",
                             "03_density_ratio_by_phenotype.png", hline=1.0)
    print("  03_density_ratio_by_phenotype.png")

    # 04: flow ratio (in/out) by phenotype
    stratified_by_phenotype(per_p, "flow_ratio",
                             "Flow |v| inside / outside",
                             "Flow magnitude ratio (inside / outside) by phenotype",
                             "04_flow_ratio_by_phenotype.png", hline=1.0)
    print("  04_flow_ratio_by_phenotype.png")

    # 05: density trajectory inside vs outside per condition
    density_traj_in_vs_out(big, "05_density_trajectory_inside_vs_outside.png")
    print("  05_density_trajectory_inside_vs_outside.png")

    # 06: flow trajectory inside vs outside per condition
    density_traj_in_vs_out(big, "06_flow_trajectory_inside_vs_outside.png",
                            ycol_in="flow_mag_inside_pole",
                            ycol_out="flow_mag_outside_pole",
                            ylabel="PIV flow magnitude [px/frame]",
                            title="PIV flow magnitude inside vs outside, over time")
    print("  06_flow_trajectory_inside_vs_outside.png")

    # 07: total H2A intensity over time
    trajectory_plot(big, "total_h2a_intensity",
                     "Total H2A intensity in mask",
                     "Total H2A intensity over time (cell-mass proxy)",
                     "07_total_h2a_intensity_over_time.png")
    print("  07_total_h2a_intensity_over_time.png")

    # 08: nuclei count over time
    trajectory_plot(big, "n_nuclei", "Detected nuclei count",
                     "Detected nuclei per frame over time\n"
                     "(reliable post-12 hpf; pescoids with diffuse H2A may report 0)",
                     "08_nuclei_count_over_time.png")
    print("  08_nuclei_count_over_time.png")

    # 09: max nuclei by phenotype
    stratified_by_phenotype(per_p, "n_nuclei_max",
                             "Max nuclei detected in 12-16 hpf",
                             "Peak-window nuclei count by phenotype",
                             "09_nuclei_count_by_phenotype.png")
    print("  09_nuclei_count_by_phenotype.png")

    # Headline summary table
    print("\n=== Phase 5a per-condition summary (12-16 hpf) ===")
    s = per_p.groupby("condition").agg(
        n=("pescoid", "count"),
        density_in_mean=("density_inside", "mean"),
        density_out_mean=("density_outside", "mean"),
        density_ratio_median=("density_ratio", "median"),
        flow_in_mean=("flow_inside", "mean"),
        flow_out_mean=("flow_outside", "mean"),
        flow_ratio_median=("flow_ratio", "median"),
        nuclei_max_mean=("n_nuclei_max", "mean"),
    ).round(2)
    s.to_csv(str(TABLES / "phase5a_condition_summary.csv"))
    print(s.to_string())

    print(f"\nAll plots in: {PLOTS}/")


if __name__ == "__main__":
    main()
