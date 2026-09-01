"""Trajectory plots for the 250212 mezzo+E-cad dataset (14 conditions)."""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\250212_mezzo_E-Cad")
FIGS = ANALYSIS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

HPF_START = 6.0
HPF_INTERVAL = 0.4
PX_UM = 1.29
RNG = np.random.default_rng(42)

# Mezzo (gold/orange/blue tones) and E-cad (purple/teal tones)
EXPERIMENTS = [
    "P_mezzo_ctrl",
    "P_mezzo_3-5hpf_Act",
    "P_mezzo_3-5hpf_Chi",
    "P_mezzo_3-5hpf_Act-Chi",
    "P_mezzo_5-7hpf_Act",
    "P_mezzo_5-7hpf_Chi",
    "P_mezzo_5-7hpf_Act-Chi",
    "P_E-cad_ctrl",
    "P_E-cad_3-5hpf-pulse_Activin",
    "P_E-cad_3-5hpf-pulse_Chiron",
    "P_E-cad_3-5hpf-pulse_Activin-Chiron",
    "P_E-cad_5-7hpf-pulse_Activin",
    "P_E-cad_5-7hpf-pulse_Chiron",
    "P_E-cad_5-7hpf-pulse_Activin-Chiron",
]

LABELS = {
    "P_mezzo_ctrl":                          "Mezzo ctrl",
    "P_mezzo_3-5hpf_Act":                    "Mezzo Act 3-5h",
    "P_mezzo_3-5hpf_Chi":                    "Mezzo Chi 3-5h",
    "P_mezzo_3-5hpf_Act-Chi":                "Mezzo Act+Chi 3-5h",
    "P_mezzo_5-7hpf_Act":                    "Mezzo Act 5-7h",
    "P_mezzo_5-7hpf_Chi":                    "Mezzo Chi 5-7h",
    "P_mezzo_5-7hpf_Act-Chi":                "Mezzo Act+Chi 5-7h",
    "P_E-cad_ctrl":                          "E-cad ctrl",
    "P_E-cad_3-5hpf-pulse_Activin":          "E-cad Act 3-5h",
    "P_E-cad_3-5hpf-pulse_Chiron":           "E-cad Chi 3-5h",
    "P_E-cad_3-5hpf-pulse_Activin-Chiron":   "E-cad Act+Chi 3-5h",
    "P_E-cad_5-7hpf-pulse_Activin":          "E-cad Act 5-7h",
    "P_E-cad_5-7hpf-pulse_Chiron":           "E-cad Chi 5-7h",
    "P_E-cad_5-7hpf-pulse_Activin-Chiron":   "E-cad Act+Chi 5-7h",
}

# Mezzo cool palette + E-cad warm palette to visually separate the two markers
COLORS = {
    "P_mezzo_ctrl":                          "#888888",
    "P_mezzo_3-5hpf_Act":                    "#1f78b4",
    "P_mezzo_3-5hpf_Chi":                    "#a6cee3",
    "P_mezzo_3-5hpf_Act-Chi":                "#33a02c",
    "P_mezzo_5-7hpf_Act":                    "#0072b2",
    "P_mezzo_5-7hpf_Chi":                    "#56b4e9",
    "P_mezzo_5-7hpf_Act-Chi":                "#117733",
    "P_E-cad_ctrl":                          "#444444",
    "P_E-cad_3-5hpf-pulse_Activin":          "#e41a1c",
    "P_E-cad_3-5hpf-pulse_Chiron":           "#fb9a99",
    "P_E-cad_3-5hpf-pulse_Activin-Chiron":   "#ff7f00",
    "P_E-cad_5-7hpf-pulse_Activin":          "#b2182b",
    "P_E-cad_5-7hpf-pulse_Chiron":           "#fdbf6f",
    "P_E-cad_5-7hpf-pulse_Activin-Chiron":   "#cc6633",
}


plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.2,
    "font.size": 10,
})


def load_all():
    """Load morphology + mezzo CSVs for every pescoid in every experiment."""
    experiments = {e: [] for e in EXPERIMENTS}
    for exp in EXPERIMENTS:
        exp_dir = ANALYSIS / exp
        if not exp_dir.exists():
            continue
        for sample_dir in sorted(exp_dir.iterdir()):
            if not sample_dir.is_dir() or sample_dir.name in ("comparison_plots",):
                continue
            morph_path = sample_dir / "morphology_over_time.csv"
            mezzo_path = sample_dir / "mezzo_expression_over_time.csv"
            if not morph_path.exists():
                continue
            morph = pd.read_csv(str(morph_path))
            mezzo = pd.read_csv(str(mezzo_path)) if mezzo_path.exists() else None
            morph["hpf"] = HPF_START + morph["time"] * HPF_INTERVAL
            morph["major_axis_um"] = morph["major_axis"] * PX_UM
            if mezzo is not None:
                mezzo["hpf"] = HPF_START + mezzo["time"] * HPF_INTERVAL
            pid_match = re.search(r"(G\d+)", sample_dir.name)
            pid = pid_match.group(1) if pid_match else sample_dir.name
            experiments[exp].append({"pid": pid, "morph": morph, "mezzo": mezzo})
    return experiments


def pchip_smooth(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, ys = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(xs) < 4:
        return xs, ys
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) < 4:
        return xs, ys
    interp = PchipInterpolator(xs, ys)
    xs_dense = np.linspace(xs.min(), xs.max(), 100)
    return xs_dense, interp(xs_dense)


def bootstrap_ci(values, n_boot=500):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                     for _ in range(n_boot)])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_timepoint_ci(pescoids, xcol, ycol, src):
    arrs = []
    times = None
    for p in pescoids:
        df = p[src]
        if df is None or xcol not in df.columns or ycol not in df.columns:
            continue
        arrs.append(df[ycol].values)
        if times is None:
            times = df[xcol].values
    if not arrs or times is None:
        return None, None, None, None
    ml = max(len(a) for a in arrs)
    pad = np.full((len(arrs), ml), np.nan)
    for i, a in enumerate(arrs):
        pad[i, :len(a)] = a
    means, los, his, xs = [], [], [], []
    for t in range(ml):
        col = pad[:, t]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        m, lo, hi = bootstrap_ci(col)
        means.append(m); los.append(lo); his.append(hi)
        xs.append(times[t] if t < len(times) else times[-1])
    return np.array(xs), np.array(means), np.array(los), np.array(his)


def split_by_marker(pescoids_dict):
    """Return separate dicts for mezzo and E-cad to plot separately."""
    mezzo = {k: v for k, v in pescoids_dict.items() if "mezzo" in k}
    ecad = {k: v for k, v in pescoids_dict.items() if "E-cad" in k}
    return mezzo, ecad


def plot_trajectory(pescoids_dict, ycol, ylabel, title, fname, src="morph", clip_zero=False):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.5), sharey=True)
    mezzo, ecad = split_by_marker(pescoids_dict)

    for ax, (group, group_name) in zip(axes, [(mezzo, "Mezzo (GFP)"), (ecad, "E-cad")]):
        for exp in [e for e in EXPERIMENTS if e in group]:
            pesc = group[exp]
            if not pesc:
                continue
            xs, means, los, his = per_timepoint_ci(pesc, "hpf", ycol, src)
            if xs is None or len(xs) == 0:
                continue
            if clip_zero:
                means = np.clip(means, 0, None)
                los = np.clip(los, 0, None)
                his = np.clip(his, 0, None)
            ax.fill_between(xs, los, his, color=COLORS[exp], alpha=0.12)
            xs_s, means_s = pchip_smooth(xs, means)
            if clip_zero:
                means_s = np.clip(means_s, 0, None)
            ax.plot(xs_s, means_s, color=COLORS[exp], lw=2.2, label=LABELS[exp])
        ax.set_xlabel("Time [hpf]")
        ax.set_title(group_name, fontweight="bold")
        ax.legend(fontsize=8, loc="best", ncol=1)
        if clip_zero:
            ax.set_ylim(bottom=0)

    axes[0].set_ylabel(ylabel)
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(FIGS / fname), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def main():
    print(f"Loading data from {ANALYSIS}/...")
    experiments = load_all()
    for e, p in experiments.items():
        print(f"  {LABELS[e]:35s}: {len(p)} pescoids")

    print(f"\nGenerating trajectory plots in {FIGS}/...")

    plot_trajectory(experiments, "aspect_ratio", "Aspect Ratio",
                    "Aspect Ratio over Time", "01_aspect_ratio.png", src="morph", clip_zero=True)
    plot_trajectory(experiments, "major_axis_um", "Major Axis [um]",
                    "Major Axis over Time", "02_major_axis.png", src="morph", clip_zero=True)
    plot_trajectory(experiments, "bf_area", "Pescoid Area [px]",
                    "Pescoid Area over Time", "03_area.png", src="mezzo", clip_zero=True)
    plot_trajectory(experiments, "gfp_fraction", "GFP+ Fraction",
                    "GFP+ Fraction over Time (fixed threshold per sample)",
                    "04_gfp_fraction.png", src="mezzo", clip_zero=True)
    plot_trajectory(experiments, "gfp_mean_intensity", "Mean GFP intensity",
                    "Mean GFP intensity over Time",
                    "05_gfp_mean_intensity.png", src="mezzo", clip_zero=True)
    plot_trajectory(experiments, "gfp_pos_mean_intensity", "GFP+ region intensity",
                    "Mean intensity within GFP+ region",
                    "06_gfp_pos_intensity.png", src="mezzo", clip_zero=True)
    plot_trajectory(experiments, "gfp_total_normalised", "Total GFP / Area",
                    "Normalised total GFP",
                    "07_gfp_total_normalised.png", src="mezzo", clip_zero=True)

    print(f"\nDone. Figures in: {FIGS}/")


if __name__ == "__main__":
    main()
