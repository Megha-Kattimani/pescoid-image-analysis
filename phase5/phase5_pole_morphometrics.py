"""
Phase 5 - Pole morphometrics for Vikas's poster ask.

Question: do mezzo+ poles in Activin pescoids differ in elongation,
size, and shape from poles in P_ctrl pescoids?

Defensible quantities, one value per pescoid (primary pole only):
  pole_major_px, pole_minor_px         ellipse fit of the pole REGION
  pole_ar = major / minor              pole elongation
  pole_area                            pole size at peak
  pole_length_over_pescoid_length      relative pole size along AP
  pole_area_over_pescoid_area          fraction of pescoid that is pole
  pole_ar_trajectory                   AR over the pole track's lifetime

Inputs:
  Lyn_mezzo_phase1/per_pescoid/<cond>/<pid>/{mask, gfp, bf}_aligned.tif
  Lyn_mezzo_phase2/per_pescoid/<cond>/<pid>/mezzo_tracks.{tif, csv}
  Lyn_mezzo_phase2/phenotype_summary.csv

Outputs (Lyn_mezzo_phase5_morphometrics/):
  tables/primary_pole_morphometrics.csv
  tables/pole_trajectories.csv
  plots/01_primary_pole_ar.png             ctrl vs Activin distributions
  plots/02_pole_relative_size.png          pole vs pescoid sizes
  plots/03_pole_ar_trajectories.png        pole AR over the track lifetime
  plots/04_pescoid_vs_pole_elongation.png  joint scatter (ar_pescoid, ar_pole)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics")
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

HPF_START = 6.0
HPF_INTERVAL = 0.4
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


def fit_ellipse(binary):
    if not binary.any():
        return None
    props = measure.regionprops(binary.astype(int))
    if not props:
        return None
    p = max(props, key=lambda r: r.area)
    minor = float(p.minor_axis_length) or 1e-6
    return {
        "area_px": float(p.area),
        "major_px": float(p.major_axis_length),
        "minor_px": float(minor),
        "ar": float(p.major_axis_length / minor),
        "orientation_rad": float(p.orientation),
    }


def primary_pole_track(tracks_df):
    """Return the row of the highest-area pole-class track (the 'primary pole')."""
    if tracks_df.empty or "cluster_type" not in tracks_df.columns:
        return None
    poles = tracks_df[tracks_df["cluster_type"] == "pole"].copy()
    if poles.empty:
        return None
    return poles.loc[poles["peak_area_px"].idxmax()].to_dict()


def process_pescoid(cond, pid):
    p1 = PHASE1 / "per_pescoid" / cond / pid
    p2 = PHASE2 / "per_pescoid" / cond / pid
    mask = tifffile.imread(str(p1 / "mask_aligned.tif")) > 0
    T = mask.shape[0]

    tcsv = p2 / "mezzo_tracks.csv"
    if not tcsv.exists():
        return None, []
    try:
        tdf = pd.read_csv(str(tcsv))
    except pd.errors.EmptyDataError:
        return None, []

    primary = primary_pole_track(tdf)
    if primary is None:
        # no primary pole; still record pescoid-level shape
        peak_t = T // 2
        pescoid_g = fit_ellipse(mask[peak_t])
        return {
            "condition": cond, "pescoid": pid,
            "n_poles_total": 0,
            "primary_track_id": None,
            "peak_frame": peak_t,
            "peak_hpf": HPF_START + peak_t * HPF_INTERVAL,
            "pole_area_px": np.nan, "pole_major_px": np.nan,
            "pole_minor_px": np.nan, "pole_ar": np.nan,
            "pescoid_area_px": pescoid_g["area_px"] if pescoid_g else np.nan,
            "pescoid_major_px": pescoid_g["major_px"] if pescoid_g else np.nan,
            "pescoid_minor_px": pescoid_g["minor_px"] if pescoid_g else np.nan,
            "pescoid_ar": pescoid_g["ar"] if pescoid_g else np.nan,
            "pole_length_over_pescoid_length": np.nan,
            "pole_area_over_pescoid_area": np.nan,
        }, []

    # primary pole's peak frame
    pf = int(primary["peak_frame"])
    pf = max(0, min(T - 1, pf))
    tid = int(primary["track_id"])

    # Pole pixels at peak frame from the 3D label volume
    label_tif = p2 / "mezzo_tracks.tif"
    if not label_tif.exists():
        return None, []
    labels = tifffile.imread(str(label_tif))
    pole_mask_peak = labels[pf] == tid
    pole_g = fit_ellipse(pole_mask_peak)
    pescoid_g = fit_ellipse(mask[pf])

    if pole_g is None or pescoid_g is None:
        return None, []

    record = {
        "condition": cond, "pescoid": pid,
        "n_poles_total": int((tdf["cluster_type"] == "pole").sum()),
        "primary_track_id": tid,
        "peak_frame": pf,
        "peak_hpf": HPF_START + pf * HPF_INTERVAL,
        "pole_area_px": pole_g["area_px"],
        "pole_major_px": pole_g["major_px"],
        "pole_minor_px": pole_g["minor_px"],
        "pole_ar": pole_g["ar"],
        "pescoid_area_px": pescoid_g["area_px"],
        "pescoid_major_px": pescoid_g["major_px"],
        "pescoid_minor_px": pescoid_g["minor_px"],
        "pescoid_ar": pescoid_g["ar"],
        "pole_length_over_pescoid_length":
            pole_g["major_px"] / max(pescoid_g["major_px"], 1.0),
        "pole_area_over_pescoid_area":
            pole_g["area_px"] / max(pescoid_g["area_px"], 1.0),
    }

    # trajectory: pole AR over the track's lifetime
    traj = []
    for t in range(T):
        pm = labels[t] == tid
        if not pm.any():
            continue
        g = fit_ellipse(pm)
        if g is None:
            continue
        pesc = fit_ellipse(mask[t])
        traj.append({
            "condition": cond, "pescoid": pid, "track_id": tid,
            "frame": t, "hpf": HPF_START + t * HPF_INTERVAL,
            "pole_ar": g["ar"], "pole_major_px": g["major_px"],
            "pole_area_px": g["area_px"],
            "pole_length_over_pescoid_length":
                g["major_px"] / max(pesc["major_px"], 1.0) if pesc else np.nan,
        })
    return record, traj


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def violin_with_jitter(ax, data_by_cond, ylabel, title, rng):
    """data_by_cond: {cond_name: 1-D array}."""
    positions = list(range(len(COND_ORDER)))
    vals = [np.array(data_by_cond.get(c, []), dtype=float) for c in COND_ORDER]
    vals = [v[~np.isnan(v)] for v in vals]
    if all(len(v) == 0 for v in vals):
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    parts = ax.violinplot(vals, positions=positions, widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COND_COLOR[COND_ORDER[i]])
        body.set_alpha(0.4)
        body.set_edgecolor("black")
    for i, v in enumerate(vals):
        if len(v) == 0:
            continue
        x = positions[i] + rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(x, v, s=55, color=COND_COLOR[COND_ORDER[i]],
                   edgecolors="black", lw=0.6, zorder=3, alpha=0.9)
        med = np.median(v)
        ax.plot([positions[i] - 0.25, positions[i] + 0.25], [med, med],
                color="black", lw=2.5, zorder=4)
    # stats: Mann-Whitney if both groups have data
    if all(len(v) >= 3 for v in vals):
        u, p = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided")
        ax.set_title(f"{title}\nctrl: n={len(vals[0])}  Activin: n={len(vals[1])}  "
                     f"Mann-Whitney p = {p:.3g}", fontweight="bold", fontsize=10)
    else:
        ax.set_title(f"{title}\nctrl: n={len(vals[0])}  Activin: n={len(vals[1])}",
                     fontweight="bold", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
    ax.set_ylabel(ylabel)


def plot_pole_ar(df, rng):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pole_ar = {c: df.loc[df["condition"] == c, "pole_ar"].dropna().values
                for c in COND_ORDER}
    pescoid_ar = {c: df.loc[df["condition"] == c, "pescoid_ar"].dropna().values
                   for c in COND_ORDER}
    violin_with_jitter(axes[0], pole_ar, "Primary pole aspect ratio",
                       "Primary mezzo+ pole AR (peak frame)", rng)
    violin_with_jitter(axes[1], pescoid_ar, "Pescoid aspect ratio (BF mask)",
                       "Pescoid AR at primary-pole peak frame", rng)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_primary_pole_ar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  01_primary_pole_ar.png")


def plot_relative_size(df, rng):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pl_len = {c: df.loc[df["condition"] == c,
                          "pole_length_over_pescoid_length"].dropna().values
                for c in COND_ORDER}
    pl_area = {c: df.loc[df["condition"] == c,
                          "pole_area_over_pescoid_area"].dropna().values
                 for c in COND_ORDER}
    violin_with_jitter(axes[0], pl_len, "pole major / pescoid major",
                       "Relative pole LENGTH along AP", rng)
    violin_with_jitter(axes[1], pl_area, "pole area / pescoid area",
                       "Pole AREA as fraction of pescoid", rng)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_pole_relative_size.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  02_pole_relative_size.png")


def plot_trajectories(traj_df):
    """Pole AR over time (peak-AR aligned). Mean +/- 95% CI per condition."""
    if traj_df.empty:
        return
    # Align each trajectory to its OWN peak-AR frame within the track lifetime
    aligned = []
    for (cond, pid, tid), g in traj_df.groupby(["condition", "pescoid", "track_id"]):
        if g.empty or g["pole_ar"].isna().all():
            continue
        peak_idx = g["pole_ar"].idxmax()
        peak_hpf = g.loc[peak_idx, "hpf"]
        gg = g.copy()
        gg["dt"] = gg["hpf"] - peak_hpf
        aligned.append(gg)
    if not aligned:
        return
    A = pd.concat(aligned)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Panel A: individual traces
    ax = axes[0]
    for (cond, pid, tid), g in A.groupby(["condition", "pescoid", "track_id"]):
        ax.plot(g["dt"], g["pole_ar"], color=COND_COLOR.get(cond, "#888"),
                lw=0.8, alpha=0.4)
    for c in COND_ORDER:
        ax.plot([], [], color=COND_COLOR[c], lw=1.5, label=COND_LABEL[c])
    ax.axvline(0, ls="--", color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("time relative to pole peak-AR (hpf)")
    ax.set_ylabel("primary pole aspect ratio")
    ax.set_title("Per-pescoid pole AR trajectories, aligned at each pole's peak",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)

    # Panel B: condition mean +/- 95% CI on a common dt grid
    ax = axes[1]
    grid = np.arange(np.floor(A["dt"].min()), np.ceil(A["dt"].max()) + 0.1, 0.4)
    for c in COND_ORDER:
        sub = A[A["condition"] == c]
        if sub.empty:
            continue
        per_pesc = []
        for (pid, tid), g in sub.groupby(["pescoid", "track_id"]):
            y = np.interp(grid, g["dt"], g["pole_ar"], left=np.nan, right=np.nan)
            per_pesc.append(y)
        Y = np.array(per_pesc)
        valid = np.sum(~np.isnan(Y), axis=0)
        keep = valid >= max(3, Y.shape[0] // 3)
        mean = np.nanmean(Y, axis=0)
        sem = np.nanstd(Y, axis=0, ddof=1) / np.sqrt(np.maximum(valid, 1))
        ax.plot(grid[keep], mean[keep], color=COND_COLOR[c], lw=2.2,
                label=f"{COND_LABEL[c]} (n={Y.shape[0]} poles)")
        ax.fill_between(grid[keep], (mean - 1.96 * sem)[keep],
                         (mean + 1.96 * sem)[keep],
                         color=COND_COLOR[c], alpha=0.2)
    ax.axvline(0, ls="--", color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("time relative to pole peak-AR (hpf)")
    ax.set_ylabel("primary pole aspect ratio")
    ax.set_title("Mean +/- 95% CI primary pole AR by condition",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "03_pole_ar_trajectories.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  03_pole_ar_trajectories.png")


def plot_joint(df):
    """(pescoid AR, pole AR) scatter per pescoid, colored by phenotype, marker by cond."""
    sub = df.dropna(subset=["pole_ar", "pescoid_ar"]).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"P_ctrl": "o", "P_Activin_3-5hpf": "s"}
    seen_pheno = set()
    for _, row in sub.iterrows():
        c = PHENO_COLOR.get(row.get("phenotype_peak", ""), "#888")
        m = markers.get(row["condition"], "o")
        label = (row.get("phenotype_peak", "")
                 if row.get("phenotype_peak", "") not in seen_pheno else None)
        ax.scatter(row["pescoid_ar"], row["pole_ar"], s=130, color=c,
                   marker=m, edgecolors="black", lw=0.7, label=label)
        seen_pheno.add(row.get("phenotype_peak", ""))
    # diagonal: pole_ar = pescoid_ar
    lo = min(sub["pescoid_ar"].min(), sub["pole_ar"].min())
    hi = max(sub["pescoid_ar"].max(), sub["pole_ar"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4,
            label="y = x (pole as elongated as pescoid)")
    ax.set_xlabel("Pescoid aspect ratio (BF mask)")
    ax.set_ylabel("Primary pole aspect ratio")
    ax.set_title("Pescoid elongation vs pole elongation\n"
                 "Below y = x: pole is rounder than pescoid; "
                 "above: pole is more elongated",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "04_pescoid_vs_pole_elongation.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  04_pescoid_vs_pole_elongation.png")


def main():
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary = summary[summary["condition"].isin(COND_ORDER)].copy()
    print(f"Processing {len(summary)} pescoids...")

    records = []
    traj_rows = []
    for _, row in summary.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        try:
            rec, traj = process_pescoid(cond, pid)
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")
            continue
        if rec is None:
            continue
        rec["phenotype_peak"] = row.get("phenotype_peak", "")
        records.append(rec)
        traj_rows.extend(traj)
        if not np.isnan(rec.get("pole_ar", np.nan)):
            print(f"  {cond}/{pid}: pole AR = {rec['pole_ar']:.2f}, "
                  f"pole len/pescoid len = "
                  f"{rec['pole_length_over_pescoid_length']:.2f}")

    df = pd.DataFrame(records)
    traj_df = pd.DataFrame(traj_rows)
    df.to_csv(str(TABLES / "primary_pole_morphometrics.csv"), index=False)
    traj_df.to_csv(str(TABLES / "pole_trajectories.csv"), index=False)
    print(f"\nSaved tables: {len(df)} pescoids, {len(traj_df)} trajectory rows")

    rng = np.random.default_rng(42)
    plot_pole_ar(df, rng)
    plot_relative_size(df, rng)
    plot_trajectories(traj_df)
    plot_joint(df)

    # condition-level summary stats
    sm = (df.groupby("condition")
            .agg(n=("pescoid", "size"),
                 n_with_pole=("pole_ar", lambda s: int(s.notna().sum())),
                 pole_ar_mean=("pole_ar", "mean"),
                 pole_ar_median=("pole_ar", "median"),
                 pole_ar_std=("pole_ar", "std"),
                 pole_len_over_pescoid_mean=(
                     "pole_length_over_pescoid_length", "mean"),
                 pescoid_ar_mean=("pescoid_ar", "mean"))
            .reset_index())
    sm.to_csv(str(TABLES / "condition_summary.csv"), index=False)
    print("\n", sm.to_string(index=False))
    print(f"\nOutputs in: {OUT}")


if __name__ == "__main__":
    main()
