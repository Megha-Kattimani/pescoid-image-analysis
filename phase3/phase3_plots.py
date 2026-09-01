"""
Phase 3 — Plots stratified by phenotype + condition.

Inputs:
  - Phase 1 per-pescoid trajectories (morphology, mezzo expression)
  - Phase 2 phenotype_summary.csv (one row per pescoid with all pole counts +
    primary/secondary pole metadata + phenotype categories)
  - Phase 2 per-pescoid mezzo_tracks.csv (for per-pole metrics)

Outputs (in Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase3\\):
  plots/01_phenotype_stack_peak.png        ← HEADLINE multipolarity figure
  plots/02_phenotype_stack_endstate.png
  plots/03_n_mezzo_poles.png               swarm + box per condition
  plots/04_n_morph_poles.png
  plots/05_n_coordinated_poles.png
  plots/06_primary_pole_area_peak.png
  plots/07_primary_pole_intensity_peak.png
  plots/08_secondary_pole_area_peak.png    NEW — bipolar pescoids only
  plots/09_secondary_pole_intensity_peak.png
  plots/10_secondary_vs_primary_scatter.png
  plots/11_secondary_primary_ratio.png
  plots/12_pole_emergence_timing.png
  plots/13_coordination_score.png
  plots/14_aspect_ratio_over_time.png      per-sample lines + mean
  plots/15_gfp_fraction_over_time.png
  plots/16_integrated_mezzo_over_time.png
  plots/17_mean_gfp_bgsub_over_time.png
  tables/phase3_summary.csv                per-condition stats
"""

import re
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase3")
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

PHENO_CATS = [
    "coordinated_bipolar", "coordinated_monopolar", "multipolar",
    "disorganised_multipolar", "mezzo_bipolar_only", "morph_bipolar_only",
    "oc_only", "diffuse_mezzo", "no_induction", "disintegrated_early",
]
PHENO_COLORS = {
    "coordinated_bipolar":     "#1b9e77",
    "coordinated_monopolar":   "#386cb0",
    "multipolar":              "#e7298a",
    "disorganised_multipolar": "#d95f02",
    "mezzo_bipolar_only":      "#7570b3",
    "morph_bipolar_only":      "#a6761d",
    "oc_only":                 "#e6ab02",
    "diffuse_mezzo":           "#66c2a5",
    "no_induction":            "#999999",
    "disintegrated_early":     "#000000",
}

HPF_START = 6.0
HPF_INTERVAL = 0.4
PEAK_HPF = (11.0, 15.0)
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# ===========================================================================
# Data loading
# ===========================================================================
def load_summary():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    df = df[df["condition"].isin(CONDITIONS)].copy()
    # Parse list-like string columns
    for c in ("mezzo_pole_areas_all", "mezzo_pole_intensities_all", "morph_pole_protrusions_all"):
        if c in df.columns:
            df[c] = df[c].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [])
    return df


def load_per_pescoid_trajectories():
    """Return long-format DataFrame: condition, pescoid, time, hpf, area, AR, gfp_fraction,
    gfp_mean_bgsub, gfp_integrated_bgsub."""
    rows = []
    for cond in CONDITIONS:
        pdir = PHASE1 / "per_pescoid" / cond
        if not pdir.exists():
            continue
        for sample_dir in sorted(pdir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pid = sample_dir.name
            # Phase 1 aligned tifs are not what we want — we need the per-frame
            # morphology + mezzo CSVs that the original pipeline wrote. Those live
            # in main_analysis output directories. Fall back: compute on the fly.
            pass
    # Actually, easier: read Phase 2 mezzo_tracks for emergence-time stuff;
    # for AR / GFP fractions over time, read from main_analysis CSVs if present,
    # otherwise compute from the aligned phase1 tifs.
    return rows


def load_perframe_metrics_from_phase1():
    """Compute per-frame area, AR, GFP fraction, mean bgsub GFP, integrated GFP
    directly from the Phase 1 aligned TIFFs."""
    import tifffile
    from skimage import measure

    GLOBAL_THR_BGSUB = 188.17  # as established in Phase 2
    rows = []
    for cond in CONDITIONS:
        pdir = PHASE1 / "per_pescoid" / cond
        if not pdir.exists():
            continue
        for sample_dir in sorted(pdir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pid = sample_dir.name
            bf_p = sample_dir / "bf_aligned.tif"
            gfp_p = sample_dir / "gfp_aligned.tif"
            mask_p = sample_dir / "mask_aligned.tif"
            if not (bf_p.exists() and gfp_p.exists() and mask_p.exists()):
                continue
            gfp = tifffile.imread(str(gfp_p))
            masks = tifffile.imread(str(mask_p)) > 0
            T = min(gfp.shape[0], len(masks))
            for t in range(T):
                if not masks[t].any():
                    continue
                area_px = int(masks[t].sum())
                # AR from regionprops
                props = measure.regionprops(masks[t].astype(int))
                if not props:
                    continue
                p = max(props, key=lambda r: r.area)
                mn = p.minor_axis_length or 1e-6
                ar = float(p.major_axis_length / mn)
                bg = float(gfp[t][~masks[t]].mean()) if (~masks[t]).any() else 0.0
                bgsub = (gfp[t] - bg).clip(min=0)
                gfp_in_mask = bgsub[masks[t]]
                mean_bgsub = float(gfp_in_mask.mean())
                integrated = float(gfp_in_mask.sum())
                positive_mask = (gfp[t] - bg) > GLOBAL_THR_BGSUB
                positive_in_mask = (positive_mask & masks[t]).sum()
                gfp_fraction = positive_in_mask / area_px if area_px > 0 else 0.0
                rows.append({
                    "condition": cond,
                    "pescoid": pid,
                    "time": t,
                    "hpf": HPF_START + t * HPF_INTERVAL,
                    "area_px": area_px,
                    "aspect_ratio": ar,
                    "gfp_mean_bgsub": mean_bgsub,
                    "gfp_integrated_bgsub": integrated,
                    "gfp_fraction": gfp_fraction,
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Plotting helpers
# ===========================================================================
def jitter(n, w=0.18):
    return RNG.uniform(-w, w, size=n)


def swarm_box(ax, df, ycol, xcol="condition", colors=COND_COLOR, labels=COND_LABEL,
               only_positive=False, log=False, ylim=None, hline=None):
    cond_order = [c for c in CONDITIONS if c in df[xcol].unique()]
    groups = []
    for c in cond_order:
        v = df[df[xcol] == c][ycol].dropna().values
        if only_positive:
            v = v[v > 0]
        groups.append(v)
    bp = ax.boxplot(groups, positions=range(len(cond_order)), widths=0.5,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(alpha=0.35),
                     medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], cond_order):
        patch.set_facecolor(colors[c])
    for i, (c, vals) in enumerate(zip(cond_order, groups)):
        if len(vals) == 0:
            continue
        x = i + jitter(len(vals))
        ax.scatter(x, vals, s=55, color=colors[c], edgecolors="black", lw=0.6,
                   alpha=0.85, zorder=3)
        # Add N annotation
        ax.text(i, ax.get_ylim()[1] if ylim is None else ylim[1],
                 f"n={len(vals)}", ha="center", va="bottom", fontsize=9, color="#444")
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([labels[c] for c in cond_order])
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1, alpha=0.5)


def stacked_bar_phenotype(df, pheno_col, title, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df.groupby(["condition", pheno_col]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[p for p in PHENO_CATS if p in counts.columns], fill_value=0)
    counts = counts.reindex([c for c in CONDITIONS if c in counts.index])
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100

    x = np.arange(len(counts.index))
    bottom = np.zeros(len(counts.index))
    for cat in counts.columns:
        h = pct[cat].values
        ax.bar(x, h, bottom=bottom, color=PHENO_COLORS.get(cat, "#777"),
                label=f"{cat}", edgecolor="white", lw=0.5)
        # Annotate cell counts if non-zero
        for i, (n_pesc, total) in enumerate(zip(counts[cat].values, totals.values)):
            if n_pesc >= 1:
                ax.text(x[i], bottom[i] + h[i] / 2, f"{n_pesc}",
                         ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom = bottom + h
    ax.set_xticks(x)
    ax.set_xticklabels([f"{COND_LABEL[c]}\nN={totals.iloc[i]}"
                        for i, c in enumerate(counts.index)])
    ax.set_ylabel("% of pescoids")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_swarm_only(df, ycol, ylabel, title, fname, ylim=None, only_positive=False, hline=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    swarm_box(ax, df, ycol, only_positive=only_positive, ylim=ylim, hline=hline)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_primary_secondary(df, fname):
    """Primary vs secondary pole area scatter per pescoid (those with both)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for c in CONDITIONS:
        sub = df[(df["condition"] == c)
                 & (df["primary_mezzo_area_peak"] > 0)
                 & (df["secondary_mezzo_area_peak"] > 0)]
        ax.scatter(sub["primary_mezzo_area_peak"], sub["secondary_mezzo_area_peak"],
                    s=70, color=COND_COLOR[c], edgecolors="black", lw=0.6,
                    alpha=0.85, label=f"{COND_LABEL[c]} (n={len(sub)})")
    # y=x reference
    ax.plot([0, df["primary_mezzo_area_peak"].max() * 1.05],
            [0, df["primary_mezzo_area_peak"].max() * 1.05],
            ls="--", color="k", alpha=0.4, label="y = x (equal poles)")
    # y=x/2 reference
    ax.plot([0, df["primary_mezzo_area_peak"].max() * 1.05],
            [0, df["primary_mezzo_area_peak"].max() * 1.05 / 2],
            ls=":", color="k", alpha=0.35, label="y = x/2 (secondary = half primary)")
    ax.set_xlabel("Primary pole area at peak [px]")
    ax.set_ylabel("Secondary pole area at peak [px]")
    ax.set_title("Secondary vs primary mezzo pole area", fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean_ci(values, n=500, ci=95):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array(
        [np.mean(RNG.choice(values, size=len(values), replace=True)) for _ in range(n)]
    )
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(values)), float(lo), float(hi)


def per_timepoint_ci(df, ycol):
    xs, ms, los, his = [], [], [], []
    for x, g in df.groupby("hpf"):
        m, lo, hi = bootstrap_mean_ci(g[ycol].dropna().values)
        xs.append(x); ms.append(m); los.append(lo); his.append(hi)
    return np.array(xs), np.array(ms), np.array(los), np.array(his)


def trajectory_plot(traj_df, ycol, ylabel, title, fname, clip_zero=True):
    fig, ax = plt.subplots(figsize=(11, 7))
    # Per-sample lines (thin, semi-transparent)
    for cond in CONDITIONS:
        sub = traj_df[traj_df["condition"] == cond]
        for pid, g in sub.groupby("pescoid"):
            ax.plot(g["hpf"], g[ycol], color=COND_COLOR[cond],
                     lw=0.6, alpha=0.25)
    # Mean +/- bootstrap CI per condition
    for cond in CONDITIONS:
        sub = traj_df[traj_df["condition"] == cond]
        if sub.empty:
            continue
        xs, ms, los, his = per_timepoint_ci(sub, ycol)
        if clip_zero:
            ms = np.clip(ms, 0, None)
            los = np.clip(los, 0, None)
            his = np.clip(his, 0, None)
        ax.fill_between(xs, los, his, color=COND_COLOR[cond], alpha=0.2)
        ax.plot(xs, ms, color=COND_COLOR[cond], lw=3, label=f"{COND_LABEL[cond]} (mean)")
    # Peak window shading
    ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.10,
                label=f"Peak window {PEAK_HPF[0]}-{PEAK_HPF[1]} hpf")
    ax.set_xlabel("Time [hpf]"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("  Phase 3 — plots stratified by phenotype + condition")
    print("=" * 60)

    print("\nLoading summary + trajectories...")
    summary = load_summary()
    print(f"  Pescoids in summary: {len(summary)}")
    print(f"    {summary['condition'].value_counts().to_dict()}")

    traj = load_perframe_metrics_from_phase1()
    print(f"  Per-frame trajectory rows: {len(traj)}")

    # ----- Plot 1+2: HEADLINE phenotype distribution stacked bars -----
    print("\nGenerating plots...")
    stacked_bar_phenotype(summary, "phenotype_peak",
                           f"Phenotype distribution at peak ({PEAK_HPF[0]}-{PEAK_HPF[1]} hpf)",
                           "01_phenotype_stack_peak.png")
    print("  01_phenotype_stack_peak.png")
    stacked_bar_phenotype(summary, "phenotype_endstate",
                           "Phenotype distribution at endstate (last 3 frames)",
                           "02_phenotype_stack_endstate.png")
    print("  02_phenotype_stack_endstate.png")

    # ----- 3-5: pole counts swarm/box -----
    plot_swarm_only(summary, "n_mezzo_poles_peak", "# mezzo poles (peak)",
                     "Number of mezzo poles per pescoid", "03_n_mezzo_poles.png",
                     ylim=(-0.3, summary["n_mezzo_poles_peak"].max() + 1))
    print("  03_n_mezzo_poles.png")
    plot_swarm_only(summary, "n_morph_poles_peak", "# morph poles (peak)",
                     "Number of morph poles per pescoid", "04_n_morph_poles.png",
                     ylim=(-0.3, max(summary["n_morph_poles_peak"].max() + 1, 1)))
    print("  04_n_morph_poles.png")
    plot_swarm_only(summary, "n_coordinated_poles_peak", "# coordinated poles (peak)",
                     "Number of coordinated mezzo + morph poles per pescoid",
                     "05_n_coordinated_poles.png",
                     ylim=(-0.3, max(summary["n_coordinated_poles_peak"].max() + 1, 1)))
    print("  05_n_coordinated_poles.png")

    # ----- 6-7: primary pole area + intensity -----
    plot_swarm_only(summary[summary["n_mezzo_poles_peak"] >= 1],
                     "primary_mezzo_area_peak", "Primary pole area [px]",
                     "Primary mezzo pole area at peak", "06_primary_pole_area_peak.png")
    print("  06_primary_pole_area_peak.png")
    plot_swarm_only(summary[summary["n_mezzo_poles_peak"] >= 1],
                     "primary_mezzo_intensity_peak", "Primary pole mean intensity (bg-sub)",
                     "Primary mezzo pole intensity at peak",
                     "07_primary_pole_intensity_peak.png")
    print("  07_primary_pole_intensity_peak.png")

    # ----- 8-9: secondary pole area + intensity -----
    plot_swarm_only(summary[summary["n_mezzo_poles_peak"] >= 2],
                     "secondary_mezzo_area_peak", "Secondary pole area [px]",
                     "Secondary mezzo pole area at peak\n(pescoids with >=2 mezzo poles)",
                     "08_secondary_pole_area_peak.png", only_positive=True)
    print("  08_secondary_pole_area_peak.png")
    plot_swarm_only(summary[summary["n_mezzo_poles_peak"] >= 2],
                     "secondary_mezzo_intensity_peak", "Secondary pole intensity (bg-sub)",
                     "Secondary mezzo pole intensity at peak\n(pescoids with >=2 mezzo poles)",
                     "09_secondary_pole_intensity_peak.png", only_positive=True)
    print("  09_secondary_pole_intensity_peak.png")

    # ----- 10: secondary vs primary scatter -----
    plot_scatter_primary_secondary(summary, "10_secondary_vs_primary_scatter.png")
    print("  10_secondary_vs_primary_scatter.png")

    # ----- 11: secondary/primary ratio -----
    summary["secondary_primary_ratio"] = np.where(
        summary["primary_mezzo_area_peak"] > 0,
        summary["secondary_mezzo_area_peak"] / summary["primary_mezzo_area_peak"],
        np.nan,
    )
    plot_swarm_only(summary[summary["n_mezzo_poles_peak"] >= 2],
                     "secondary_primary_ratio",
                     "Secondary / primary pole area ratio",
                     "Secondary pole size relative to primary\n(pescoids with >=2 mezzo poles)",
                     "11_secondary_primary_ratio.png", ylim=(0, 1.05), hline=1.0)
    print("  11_secondary_primary_ratio.png")

    # ----- 12: pole emergence timing -----
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, cond in enumerate(CONDITIONS):
        first = summary[summary["condition"] == cond]["first_mezzo_pole_emergence_hpf"].dropna()
        second = summary[summary["condition"] == cond]["second_mezzo_pole_emergence_hpf"].dropna()
        x_first = i + jitter(len(first), 0.10) - 0.18
        x_second = i + jitter(len(second), 0.10) + 0.18
        ax.scatter(x_first, first, s=60, color=COND_COLOR[cond],
                    edgecolors="black", lw=0.6, alpha=0.85, marker="o",
                    label=f"1st pole ({COND_LABEL[cond]}, n={len(first)})" if i == 0 else f"1st pole (n={len(first)})")
        if len(second) > 0:
            ax.scatter(x_second, second, s=60, color=COND_COLOR[cond],
                        edgecolors="black", lw=0.6, alpha=0.85, marker="^",
                        label=f"2nd pole ({COND_LABEL[cond]}, n={len(second)})" if i == 0 else f"2nd pole (n={len(second)})")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel("Time [hpf]")
    ax.set_title("Mezzo pole emergence timing", fontweight="bold")
    ax.axhspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.10)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "12_pole_emergence_timing.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  12_pole_emergence_timing.png")

    # ----- 13: coordination score -----
    plot_swarm_only(summary, "coordination_score_peak",
                     "Coordination score (peak)",
                     "Mezzo+morph pole coordination at peak\n(0 = uncoordinated, 1 = all coordinated)",
                     "13_coordination_score.png", ylim=(-0.05, 1.05))
    print("  13_coordination_score.png")

    # ----- 14-17: trajectory plots -----
    trajectory_plot(traj, "aspect_ratio", "Aspect ratio",
                     "Aspect ratio over time (per-pescoid + mean ± 95% CI)",
                     "14_aspect_ratio_over_time.png")
    print("  14_aspect_ratio_over_time.png")
    trajectory_plot(traj, "gfp_fraction", "GFP+ fraction",
                     "GFP+ fraction over time (per-pescoid + mean ± 95% CI)",
                     "15_gfp_fraction_over_time.png")
    print("  15_gfp_fraction_over_time.png")
    trajectory_plot(traj, "gfp_integrated_bgsub", "Integrated mezzo (bg-sub)",
                     "Integrated mezzo over time (per-pescoid + mean ± 95% CI)",
                     "16_integrated_mezzo_over_time.png")
    print("  16_integrated_mezzo_over_time.png")
    trajectory_plot(traj, "gfp_mean_bgsub", "Mean GFP (bg-sub)",
                     "Mean mezzo intensity (bg-sub) over time",
                     "17_mean_gfp_bgsub_over_time.png")
    print("  17_mean_gfp_bgsub_over_time.png")

    # ----- Summary table -----
    summary_table = summary.groupby("condition").agg(
        n=("pescoid", "count"),
        mean_n_mezzo_poles_peak=("n_mezzo_poles_peak", "mean"),
        mean_n_morph_poles_peak=("n_morph_poles_peak", "mean"),
        mean_n_coordinated_peak=("n_coordinated_poles_peak", "mean"),
        pct_coordinated_bipolar_peak=("phenotype_peak",
                                       lambda x: 100 * (x == "coordinated_bipolar").mean()),
        pct_multipolar_peak=("phenotype_peak",
                              lambda x: 100 * (x == "multipolar").mean()),
        pct_no_induction_peak=("phenotype_peak",
                                lambda x: 100 * (x == "no_induction").mean()),
        mean_primary_area_peak=("primary_mezzo_area_peak", "mean"),
        mean_secondary_area_peak=("secondary_mezzo_area_peak",
                                   lambda x: x[x > 0].mean() if any(x > 0) else 0),
        mean_first_pole_hpf=("first_mezzo_pole_emergence_hpf", "mean"),
        mean_second_pole_hpf=("second_mezzo_pole_emergence_hpf",
                               lambda x: x.dropna().mean() if x.dropna().size > 0 else np.nan),
    ).round(2)
    summary_table.to_csv(str(TABLES / "phase3_summary.csv"))
    print("\n=== Phase 3 summary ===")
    print(summary_table.to_string())

    print(f"\nAll plots in: {PLOTS}/")
    print(f"Summary table: {TABLES}/phase3_summary.csv")


if __name__ == "__main__":
    main()
