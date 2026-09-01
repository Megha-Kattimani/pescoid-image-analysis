"""
Vikas slide updates v2 - cleaner, anchored to 12 hpf peak elongation,
faceted where useful.

  10_v2 - pole AREA vs emergence TIME (one panel, two markers per pescoid).
          Like the AR scatter aesthetic - clean, clearly readable, with the
          7-16 hpf window shaded and 12 hpf as the peak-elongation reference.
  11_v2 - 1st vs 2nd pole emergence (the diagonal plot only - cleanest version)
          with 12 hpf reference and a per-phenotype delay histogram below.
  13_v2 - AR vs GFP fraction trajectories FACETED by phenotype.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT_DIR = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_kymograph\plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HPF_START = 6.0
HPF_INTERVAL = 0.4
GLOBAL_THR_BGSUB = 188.17
PEAK_ELONGATION_HPF = 12.0
WINDOW = (7.0, 16.0)

PHENO_COLOR = {
    "no_induction": "#999999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77", "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a", "oc_only": "#e6ab02",
}
PHENO_ORDER = ["no_induction", "diffuse_mezzo", "coordinated_monopolar",
               "mezzo_bipolar_only", "coordinated_bipolar",
               "disorganised_multipolar", "multipolar"]
COND_MARKER = {"P_ctrl": "o", "P_Activin_3-5hpf": "s"}
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
                     "axes.grid": True, "grid.color": "white", "font.size": 11})


# ---------------------------------------------------------------------------
# 10_v2 - pole AREA vs emergence TIME (clean, AR-style)
# ---------------------------------------------------------------------------
def plot_10_v2(df):
    """One scatter, each pole = one point. 1st-pole = circle, 2nd-pole = triangle.
    1st and 2nd pole of the same pescoid connected by a thin line."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axvspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.08,
               label=f"window of interest {WINDOW[0]:.0f}-{WINDOW[1]:.0f} hpf")
    ax.axvline(PEAK_ELONGATION_HPF, color="cyan", lw=1.6,
               label=f"peak elongation = {PEAK_ELONGATION_HPF:.0f} hpf")

    plotted_pheno = set()
    for _, row in df.iterrows():
        t1 = row.get("first_mezzo_pole_emergence_hpf")
        a1 = row.get("primary_mezzo_area_peak")
        t2 = row.get("second_mezzo_pole_emergence_hpf")
        a2 = row.get("secondary_mezzo_area_peak")
        pheno = row["phenotype_peak"]
        c = PHENO_COLOR.get(pheno, "#888")
        if pd.notna(t1) and a1 and a1 > 0:
            ax.scatter(t1, a1, s=110, color=c, edgecolors="black", lw=0.7,
                       marker="o", zorder=3,
                       label=pheno if pheno not in plotted_pheno else None)
            plotted_pheno.add(pheno)
            if pd.notna(t2) and a2 and a2 > 0:
                ax.plot([t1, t2], [a1, a2], "-", color=c, alpha=0.4, lw=0.8,
                        zorder=2)
                ax.scatter(t2, a2, s=110, color=c, edgecolors="black", lw=0.7,
                           marker="^", zorder=3)

    ax.set_xlabel("Pole emergence [hpf]")
    ax.set_ylabel("Pole area at peak [px]")
    ax.set_title("Mezzo+ pole area vs emergence time\n"
                 "(circle = 1st pole, triangle = 2nd pole; line connects same pescoid; colour = phenotype)",
                 fontweight="bold")
    # legend
    handles = [Line2D([0], [0], marker="o", color=PHENO_COLOR[p], lw=0,
                       markeredgecolor="black", markersize=9, label=p)
               for p in PHENO_ORDER if p in plotted_pheno]
    handles += [Line2D([0], [0], marker="o", color="gray", lw=0,
                        markeredgecolor="black", markersize=9, label="1st pole"),
                Line2D([0], [0], marker="^", color="gray", lw=0,
                        markeredgecolor="black", markersize=9, label="2nd pole"),
                Line2D([0], [0], color="cyan", lw=1.6, label="12 hpf (peak elongation)")]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "10_v2_pole_area_vs_time.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  10_v2_pole_area_vs_time.png")


# ---------------------------------------------------------------------------
# 11_v2 - 1st vs 2nd pole emergence, anchored at 12 hpf
# ---------------------------------------------------------------------------
def plot_11_v2(df):
    sub = df[df["first_mezzo_pole_emergence_hpf"].notna()].copy()
    sub_both = sub[sub["second_mezzo_pole_emergence_hpf"].notna()].copy()
    sub_both["delay"] = (sub_both["second_mezzo_pole_emergence_hpf"]
                          - sub_both["first_mezzo_pole_emergence_hpf"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                              gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: 1st vs 2nd emergence scatter (cleaner)
    ax = axes[0]
    ax.axvspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.06)
    ax.axhspan(WINDOW[0], WINDOW[1], color="yellow", alpha=0.06)
    ax.axvline(PEAK_ELONGATION_HPF, color="cyan", lw=1.5, alpha=0.7)
    ax.axhline(PEAK_ELONGATION_HPF, color="cyan", lw=1.5, alpha=0.7)
    mn = WINDOW[0]; mx = max(sub_both[["first_mezzo_pole_emergence_hpf",
                                         "second_mezzo_pole_emergence_hpf"]].max().max(), WINDOW[1])
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, alpha=0.5, label="y = x (simultaneous)")
    plotted = set()
    for _, row in sub_both.iterrows():
        c = PHENO_COLOR.get(row["phenotype_peak"], "#888")
        pheno = row["phenotype_peak"]
        ax.scatter(row["first_mezzo_pole_emergence_hpf"],
                   row["second_mezzo_pole_emergence_hpf"],
                   s=140, color=c, edgecolors="black", lw=0.7,
                   marker=COND_MARKER.get(row["condition"], "o"),
                   label=pheno if pheno not in plotted else None)
        plotted.add(pheno)
    ax.set_xlabel("1st pole emergence [hpf]")
    ax.set_ylabel("2nd pole emergence [hpf]")
    ax.set_title("1st vs 2nd mezzo+ pole emergence\n"
                 "yellow = 7-16 hpf window; cyan = 12 hpf",
                 fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color=PHENO_COLOR[p], lw=0,
                       markeredgecolor="black", markersize=9, label=p)
               for p in PHENO_ORDER if p in plotted]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    # Panel B: delay between poles, by phenotype
    ax = axes[1]
    bins = np.arange(0, sub_both["delay"].max() + 1, 0.8)
    for pheno in PHENO_ORDER:
        s = sub_both[sub_both["phenotype_peak"] == pheno]
        if s.empty:
            continue
        ax.hist(s["delay"], bins=bins, color=PHENO_COLOR[pheno],
                alpha=0.6, edgecolor="black",
                label=f"{pheno} (n={len(s)})")
    ax.set_xlabel("Delay between 1st and 2nd pole [hpf]")
    ax.set_ylabel("# pescoids")
    ax.set_title("Distribution of 1st→2nd pole delay, by phenotype", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "11_v2_pole_emergence_clean.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  11_v2_pole_emergence_clean.png")

    # Diagnostic: which pescoids have 1st pole AFTER 12 hpf?
    late = sub[sub["first_mezzo_pole_emergence_hpf"] > 12.0].copy()
    if not late.empty:
        late = late[["condition", "pescoid", "phenotype_peak",
                      "first_mezzo_pole_emergence_hpf",
                      "primary_mezzo_area_peak"]].sort_values(
            "first_mezzo_pole_emergence_hpf")
        late.to_csv(str(OUT_DIR.parent / "tables" / "late_first_pole_pescoids.csv"),
                    index=False)
        print(f"  late-1st-pole pescoids (>12 hpf): {len(late)} - see tables/late_first_pole_pescoids.csv")
        print(late.to_string(index=False))


# ---------------------------------------------------------------------------
# 13_v2 - AR vs GFP-fraction trajectories, FACETED by phenotype
# ---------------------------------------------------------------------------
def compute_ar_gfp(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    rows = []
    for t in range(mask.shape[0]):
        if not mask[t].any():
            continue
        props = measure.regionprops(mask[t].astype(int))
        if not props:
            continue
        p = max(props, key=lambda r: r.area)
        minor = float(p.minor_axis_length) or 1e-6
        ar = float(p.major_axis_length / minor)
        bg = float(gfp[t][~mask[t]].mean()) if (~mask[t]).any() else 0.0
        positive = (gfp[t] - bg) > GLOBAL_THR_BGSUB
        frac = float((positive & mask[t]).sum() / max(mask[t].sum(), 1))
        rows.append({"hpf": HPF_START + t * HPF_INTERVAL,
                     "ar": ar, "gfp_frac": frac})
    return pd.DataFrame(rows)


def plot_13_v2(df):
    trajs = {}
    for _, row in df.iterrows():
        try:
            t = compute_ar_gfp(row["condition"], row["pescoid"])
            if not t.empty:
                trajs[(row["condition"], row["pescoid"], row["phenotype_peak"])] = t
        except Exception:
            pass

    phenos = [p for p in PHENO_ORDER
              if any(k[2] == p for k in trajs)]
    n = len(phenos)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows),
                              sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, pheno in zip(axes, phenos):
        color = PHENO_COLOR[pheno]
        keys = [k for k in trajs if k[2] == pheno]
        for k in keys:
            tr = trajs[k]
            # gradient: alpha increases with time
            ax.plot(tr["ar"], tr["gfp_frac"], "-", color=color, lw=0.9, alpha=0.5)
            ax.scatter(tr["ar"].iloc[0], tr["gfp_frac"].iloc[0], s=22,
                       color=color, marker="o", edgecolors="white", lw=0.4,
                       zorder=4)   # start
            ax.scatter(tr["ar"].iloc[-1], tr["gfp_frac"].iloc[-1], s=55,
                       color=color, marker="s", edgecolors="black", lw=0.6,
                       zorder=5)   # end
        ax.set_title(f"{pheno}  (n={len(keys)})", fontweight="bold", fontsize=11,
                     color=color)
        ax.set_xlabel("aspect ratio")
        ax.set_ylabel("GFP+ fraction")
        ax.grid(True, alpha=0.4)
    for ax in axes[len(phenos):]:
        ax.axis("off")
    plt.suptitle("AR vs GFP+ fraction trajectories, faceted by phenotype\n"
                 "small circle = t=6 hpf (start); square = t=end. Line = pescoid trajectory.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(str(OUT_DIR / "13_v2_ar_gfp_trajectories_faceted.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  13_v2_ar_gfp_trajectories_faceted.png")


def main():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    print(f"Loaded {len(df)} pescoids.")
    plot_10_v2(df)
    plot_11_v2(df)
    plot_13_v2(df)
    print(f"\nAll plots in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
