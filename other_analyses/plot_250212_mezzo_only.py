"""Trajectory plots for the 250212 mezzo-only dataset (7 conditions)."""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\250212_mezzo_only")
FIGS = ANALYSIS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

HPF_START = 6.0
HPF_INTERVAL = 0.4
PX_UM = 1.29
RNG = np.random.default_rng(42)

EXPERIMENTS = [
    "P_mezzo_ctrl",
    "P_mezzo_3-5hpf_Act",
    "P_mezzo_3-5hpf_Chi",
    "P_mezzo_3-5hpf_Act-Chi",
    "P_mezzo_5-7hpf_Act",
    "P_mezzo_5-7hpf_Chi",
    "P_mezzo_5-7hpf_Act-Chi",
]

LABELS = {
    "P_mezzo_ctrl": "ctrl",
    "P_mezzo_3-5hpf_Act": "Act 3-5h",
    "P_mezzo_3-5hpf_Chi": "Chi 3-5h",
    "P_mezzo_3-5hpf_Act-Chi": "Act+Chi 3-5h",
    "P_mezzo_5-7hpf_Act": "Act 5-7h",
    "P_mezzo_5-7hpf_Chi": "Chi 5-7h",
    "P_mezzo_5-7hpf_Act-Chi": "Act+Chi 5-7h",
}

COLORS = {
    "P_mezzo_ctrl":         "#888888",
    "P_mezzo_3-5hpf_Act":   "#e41a1c",
    "P_mezzo_3-5hpf_Chi":   "#377eb8",
    "P_mezzo_3-5hpf_Act-Chi": "#4daf4a",
    "P_mezzo_5-7hpf_Act":   "#984ea3",
    "P_mezzo_5-7hpf_Chi":   "#ff7f00",
    "P_mezzo_5-7hpf_Act-Chi": "#a65628",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


def load_all():
    experiments = {e: [] for e in EXPERIMENTS}
    for exp in EXPERIMENTS:
        exp_dir = ANALYSIS / exp
        if not exp_dir.exists():
            continue
        for sample_dir in sorted(exp_dir.iterdir()):
            if not sample_dir.is_dir() or sample_dir.name in ("comparison_plots",):
                continue
            morph_p = sample_dir / "morphology_over_time.csv"
            mezzo_p = sample_dir / "mezzo_expression_over_time.csv"
            if not morph_p.exists():
                continue
            morph = pd.read_csv(str(morph_p))
            mezzo = pd.read_csv(str(mezzo_p)) if mezzo_p.exists() else None
            morph["hpf"] = HPF_START + morph["time"] * HPF_INTERVAL
            morph["major_axis_um"] = morph["major_axis"] * PX_UM
            if mezzo is not None:
                mezzo["hpf"] = HPF_START + mezzo["time"] * HPF_INTERVAL
            pid = re.search(r"(G\d+)", sample_dir.name)
            pid = pid.group(1) if pid else sample_dir.name
            experiments[exp].append({"pid": pid, "morph": morph, "mezzo": mezzo})
    return experiments


def pchip_smooth(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, ys = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(xs) < 4: return xs, ys
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) < 4: return xs, ys
    interp = PchipInterpolator(xs, ys)
    xs_d = np.linspace(xs.min(), xs.max(), 100)
    return xs_d, interp(xs_d)


def bootstrap_ci(values, n_boot=500):
    if len(values) == 0: return np.nan, np.nan, np.nan
    boot = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                     for _ in range(n_boot)])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_timepoint_ci(pescoids, ycol, src):
    arrs = []; times = None
    for p in pescoids:
        df = p[src]
        if df is None or "hpf" not in df.columns or ycol not in df.columns:
            continue
        arrs.append(df[ycol].values)
        if times is None:
            times = df["hpf"].values
    if not arrs:
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


def plot_trajectory(experiments, ycol, ylabel, title, fname, src="morph", clip_zero=False):
    fig, ax = plt.subplots(figsize=(11, 7))
    for exp in EXPERIMENTS:
        pesc = experiments[exp]
        if not pesc:
            continue
        xs, means, los, his = per_timepoint_ci(pesc, ycol, src)
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
        ax.plot(xs_s, means_s, color=COLORS[exp], lw=2.3, label=LABELS[exp])
    ax.set_xlabel("Time [hpf]"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if clip_zero:
        ax.set_ylim(bottom=0)
    ax.legend(title="Condition", fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(str(FIGS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


def main():
    print(f"Loading data from {ANALYSIS}/...")
    experiments = load_all()
    for e, p in experiments.items():
        print(f"  {LABELS[e]:18s}: {len(p)} pescoids")

    print(f"\nGenerating trajectory plots in {FIGS}/...")
    plot_trajectory(experiments, "aspect_ratio", "Aspect Ratio",
                    "Aspect Ratio over Time", "01_aspect_ratio.png", clip_zero=True)
    plot_trajectory(experiments, "major_axis_um", "Major Axis [um]",
                    "Major Axis over Time", "02_major_axis.png", clip_zero=True)
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

    # Summary table
    print("\nFinal-3-frame summary:")
    print(f'{"Condition":<18s} {"N":>3s} {"AR":>6s} {"GFP frac":>9s} {"GFP int":>8s} {"Area":>8s}')
    summary_rows = []
    for exp in EXPERIMENTS:
        if not experiments[exp]:
            continue
        ars, fracs, ints, areas = [], [], [], []
        for p in experiments[exp]:
            m, z = p["morph"], p["mezzo"]
            if m is not None:
                ars.append(m["aspect_ratio"].iloc[-3:].mean())
                areas.append(m["area"].iloc[-3:].mean())
            if z is not None:
                if "gfp_fraction" in z.columns:
                    fracs.append(z["gfp_fraction"].iloc[-3:].mean())
                if "gfp_mean_intensity" in z.columns:
                    ints.append(z["gfp_mean_intensity"].iloc[-3:].mean())
        n = len(experiments[exp])
        ar_m = np.mean(ars) if ars else 0
        frac_m = np.mean(fracs) * 100 if fracs else 0
        int_m = np.mean(ints) if ints else 0
        area_m = np.mean(areas) if areas else 0
        print(f'{LABELS[exp]:<18s} {n:>3d} {ar_m:>5.2f} {frac_m:>7.1f}% {int_m:>7.3f} {area_m:>7.0f}')
        summary_rows.append({"condition": LABELS[exp], "n": n, "ar": ar_m,
                              "gfp_frac": frac_m, "gfp_int": int_m, "area": area_m})

    pd.DataFrame(summary_rows).to_csv(str(ANALYSIS / "summary_table.csv"), index=False)
    print(f"\nSummary saved: {ANALYSIS}/summary_table.csv")
    print(f"Plots in: {FIGS}/")


if __name__ == "__main__":
    main()
