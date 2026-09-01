"""
Slide updates for the next Vikas presentation:
  S7  - primary vs secondary pole AREA, colored by emergence-time DELAY (one
        panel = three variables in a single plot).
  S8  - per-pescoid PAIRED pole emergence: 1st and 2nd pole connected by a
        line + the diagonal scatter version (1st vs 2nd emergence time).
  AR  - aspect-ratio vs gfp_fraction joint plot in two flavours:
          (a) peak-value scatter, one point per pescoid
          (b) trajectory plot, each pescoid is a trace through (AR, gfp_frac)
              space with time as a colour gradient.

Inputs:
  Lyn_mezzo_phase2/phenotype_summary.csv
  Lyn_mezzo_phase1/per_pescoid/<cond>/<pid>/{bf,gfp,mask}_aligned.tif (for AR/gfp_frac per frame)

Outputs (Lyn_mezzo_phase5_kymograph/plots/):
  10_pole_area_with_timing.png
  11_pole_emergence_paired.png
  12_ar_vs_gfp_fraction_peak.png
  13_ar_vs_gfp_fraction_trajectory.png
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

PHENO_COLOR = {
    "no_induction": "#999999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77", "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a", "oc_only": "#e6ab02",
}
COND_MARKER = {"P_ctrl": "o", "P_Activin_3-5hpf": "s"}
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
                     "axes.grid": True, "grid.color": "white", "font.size": 11})


# ---------------------------------------------------------------------------
# Slide 7 - primary vs secondary pole area, colored by emergence delay
# ---------------------------------------------------------------------------
def plot_s7(df):
    """One point per pescoid with 2+ poles; colour = (t_pole2 - t_pole1)."""
    sub = df[(df["primary_mezzo_area_peak"] > 0)
              & (df["secondary_mezzo_area_peak"] > 0)
              & df["first_mezzo_pole_emergence_hpf"].notna()
              & df["second_mezzo_pole_emergence_hpf"].notna()].copy()
    sub["delay_hr"] = sub["second_mezzo_pole_emergence_hpf"] - sub["first_mezzo_pole_emergence_hpf"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(sub["primary_mezzo_area_peak"], sub["secondary_mezzo_area_peak"],
                    c=sub["delay_hr"], cmap="viridis",
                    s=130, edgecolors="black", lw=0.7,
                    vmin=0, vmax=max(sub["delay_hr"].max(), 1.0))
    # diagonal y=x reference
    mn = 0
    mx = max(sub["primary_mezzo_area_peak"].max(), sub["secondary_mezzo_area_peak"].max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.4, label="y = x (equal areas)")
    # annotate phenotype with small text
    for _, row in sub.iterrows():
        ax.text(row["primary_mezzo_area_peak"], row["secondary_mezzo_area_peak"],
                f" {row['pescoid']}", fontsize=7, alpha=0.6, va="center")
    cbar = plt.colorbar(sc, ax=ax, label="Δt (2nd − 1st pole emergence, hr)")
    ax.set_xlabel("Primary pole area at peak [px]")
    ax.set_ylabel("Secondary pole area at peak [px]")
    ax.set_title("Primary vs Secondary pole AREA, with emergence DELAY (colour)\n"
                 "darker = simultaneous; brighter = later 2nd pole",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "10_pole_area_with_timing.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  10_pole_area_with_timing.png")


# ---------------------------------------------------------------------------
# Slide 8 - paired pole emergence per pescoid
# ---------------------------------------------------------------------------
def plot_s8(df):
    sub = df[df["first_mezzo_pole_emergence_hpf"].notna()].copy()
    sub_both = sub[sub["second_mezzo_pole_emergence_hpf"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: paired plot, sorted by 1st pole time
    ax = axes[0]
    s = sub.sort_values("first_mezzo_pole_emergence_hpf").reset_index(drop=True)
    for i, row in s.iterrows():
        c = PHENO_COLOR.get(row["phenotype_peak"], "#888")
        t1 = row["first_mezzo_pole_emergence_hpf"]
        t2 = row.get("second_mezzo_pole_emergence_hpf", np.nan)
        ax.plot(i, t1, "o", color=c, markersize=9, markeredgecolor="black", lw=0)
        if pd.notna(t2):
            ax.plot([i, i], [t1, t2], "-", color=c, lw=1.2, alpha=0.5)
            ax.plot(i, t2, "^", color=c, markersize=9, markeredgecolor="black", lw=0)
    ax.set_xlabel("pescoid (sorted by 1st pole emergence)")
    ax.set_ylabel("emergence time [hpf]")
    ax.set_title("Per-pescoid pole emergence: 1st (o) and 2nd (▲) connected",
                 fontweight="bold")
    # phenotype legend
    phenos_in = [p for p in PHENO_COLOR if (s["phenotype_peak"] == p).any()]
    handles = [Line2D([0], [0], marker="o", color=PHENO_COLOR[p], lw=0,
                       markeredgecolor="black", markersize=8, label=p)
               for p in phenos_in]
    ax.legend(handles=handles, loc="best", fontsize=8)

    # Panel 2: scatter of (t1, t2) with diagonal
    ax = axes[1]
    for _, row in sub_both.iterrows():
        c = PHENO_COLOR.get(row["phenotype_peak"], "#888")
        m = COND_MARKER.get(row["condition"], "o")
        ax.scatter(row["first_mezzo_pole_emergence_hpf"],
                   row["second_mezzo_pole_emergence_hpf"],
                   s=130, color=c, edgecolors="black", lw=0.7, marker=m)
    mn = sub_both[["first_mezzo_pole_emergence_hpf",
                    "second_mezzo_pole_emergence_hpf"]].min().min()
    mx = sub_both[["first_mezzo_pole_emergence_hpf",
                    "second_mezzo_pole_emergence_hpf"]].max().max()
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("1st pole emergence [hpf]")
    ax.set_ylabel("2nd pole emergence [hpf]")
    ax.set_title("1st vs 2nd pole emergence time\n"
                 "distance above diagonal = delay between poles", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "11_pole_emergence_paired.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  11_pole_emergence_paired.png")


# ---------------------------------------------------------------------------
# AR vs GFP fraction - peak scatter + trajectory
# ---------------------------------------------------------------------------
def compute_ar_gfp_per_frame(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    rows = []
    T = mask.shape[0]
    for t in range(T):
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
        gfp_frac = float((positive & mask[t]).sum() / max(mask[t].sum(), 1))
        rows.append({"time": t, "hpf": HPF_START + t * HPF_INTERVAL,
                     "aspect_ratio": ar, "gfp_fraction": gfp_frac})
    return pd.DataFrame(rows)


def ar_gfp_panels(df_summary):
    """Per-pescoid traces in (AR, gfp_fraction) space; peak scatter and
    full-trajectory views."""
    per_p_trajectories = {}
    for _, row in df_summary.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        try:
            traj = compute_ar_gfp_per_frame(cond, pid)
        except Exception:
            continue
        if traj.empty:
            continue
        per_p_trajectories[(cond, pid)] = traj

    # Peak scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    peak_rows = []
    for (cond, pid), traj in per_p_trajectories.items():
        peak_ar = traj["aspect_ratio"].max()
        peak_gfp = traj["gfp_fraction"].max()
        row_meta = df_summary[(df_summary["condition"] == cond)
                              & (df_summary["pescoid"] == pid)].iloc[0]
        pheno = row_meta["phenotype_peak"]
        peak_rows.append({"condition": cond, "pescoid": pid,
                          "phenotype_peak": pheno,
                          "peak_aspect_ratio": peak_ar,
                          "peak_gfp_fraction": peak_gfp})
    peak_df = pd.DataFrame(peak_rows)
    for cond in ["P_ctrl", "P_Activin_3-5hpf"]:
        sub = peak_df[peak_df["condition"] == cond]
        if sub.empty:
            continue
        m = COND_MARKER[cond]
        for _, r in sub.iterrows():
            ax.scatter(r["peak_aspect_ratio"], r["peak_gfp_fraction"],
                        s=110, color=PHENO_COLOR.get(r["phenotype_peak"], "#888"),
                        edgecolors="black", lw=0.6, marker=m,
                        label=COND_LABEL[cond] if _ == sub.index[0] else None,
                        alpha=0.85)
    ax.set_xlabel("Peak aspect ratio")
    ax.set_ylabel("Peak GFP+ fraction (mezzo+ pixels / mask)")
    ax.set_title("Peak aspect ratio vs peak GFP+ fraction\n"
                 "(circle = P_ctrl, square = P_Activin; colour = phenotype)",
                 fontweight="bold")
    # phenotype legend
    phenos_in = [p for p in PHENO_COLOR if (peak_df["phenotype_peak"] == p).any()]
    handles = [Line2D([0], [0], marker="o", color=PHENO_COLOR[p], lw=0,
                       markeredgecolor="black", markersize=8, label=p)
               for p in phenos_in]
    ax.legend(handles=handles, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "12_ar_vs_gfp_fraction_peak.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  12_ar_vs_gfp_fraction_peak.png")

    # Trajectories (faceted by condition)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, cond in zip(axes, ["P_ctrl", "P_Activin_3-5hpf"]):
        pescs = [(c, p, t) for (c, p), t in per_p_trajectories.items() if c == cond]
        for c, p, traj in pescs:
            pheno = df_summary[(df_summary["condition"] == c)
                                & (df_summary["pescoid"] == p)].iloc[0]["phenotype_peak"]
            color = PHENO_COLOR.get(pheno, "#888")
            ax.plot(traj["aspect_ratio"], traj["gfp_fraction"], "-",
                    color=color, lw=0.8, alpha=0.5)
            ax.scatter(traj["aspect_ratio"].iloc[-1], traj["gfp_fraction"].iloc[-1],
                        s=40, color=color, edgecolors="black", lw=0.4, zorder=3)
        ax.set_xlabel("aspect ratio")
        ax.set_title(COND_LABEL[cond], fontweight="bold")
    axes[0].set_ylabel("GFP+ fraction")
    plt.suptitle("Aspect ratio vs GFP+ fraction trajectories per pescoid\n"
                 "(end-of-trace marker, faceted by condition, colour by phenotype)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(str(OUT_DIR / "13_ar_vs_gfp_fraction_trajectory.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  13_ar_vs_gfp_fraction_trajectory.png")


def main():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    print(f"Loaded {len(df)} pescoids from Phase 2 summary.")
    plot_s7(df)
    plot_s8(df)
    ar_gfp_panels(df)
    print(f"\nAll plots in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
