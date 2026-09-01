"""
Phase 5b-3 v3 - Shape/intensity-based mitosis detection.

Detects mitotic nuclei from per-frame StarDist labels by looking at the
characteristic morphology of dividing cells:
  - Condensed chromatin -> high per-pixel H2A intensity in the nucleus
  - Compact rounded shape -> small area + high solidity
  - During telophase/anaphase: dumbbell or pair-of-blobs shape

Strategy: per frame, label each StarDist nucleus with
  area, mean_intensity_bgsub, solidity, eccentricity.
Then flag a nucleus as a mitotic candidate if BOTH:
  (a) mean intensity > 80th-percentile of the frame's nuclei
  (b) area < 30th-percentile of the frame's nuclei (small compact)

Persistence filter: candidate must be a candidate for >=2 of the next 3 frames
(so single-frame outliers don't count).

This is independent of btrack's branching and works on the existing
StarDist masks from Phase 5a v2.

Outputs (mezzo_H2A_phase5b3_v3/):
  per_pescoid/<cond>/<pid>/mitosis_candidates.csv
  per_pescoid/<cond>/<pid>/<pid>_mitosis_map_v3.png
  plots/01_mitosis_count_by_phenotype.png
  plots/02_mitosis_inside_vs_outside_pole.png
  plots/03_mitosis_rate_over_time.png
  plots/04_mitoses_formation_by_phenotype.png
  plots/05_mitoses_vs_pole_count.png
  tables/phase5b3_v3_per_event.csv
  tables/phase5b3_v3_per_pescoid_summary.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
PHASE5A = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5a")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b3_v3")
PER = OUT / "per_pescoid"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PER, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 7.0
HPF_INTERVAL = 698.8316040039062 / 3600.0
FORM_HPF = (8.0, 12.0)
PEAK_HPF = (12.0, 16.0)
RNG = np.random.default_rng(42)

# Mitotic candidate thresholds (relative within each frame)
INT_PERCENTILE = 80   # mean intensity above this percentile -> bright
AREA_PERCENTILE = 30  # area below this percentile -> compact
PERSISTENCE_FRAMES = 2   # candidate must persist for >=2 consecutive frames

PHENO_ORDER = ["coordinated_monopolar", "mezzo_bipolar_only", "multipolar",
               "diffuse_mezzo", "no_induction"]
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


def load_pole_stack(cond, pid):
    csv_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
    tif_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.tif"
    if not (csv_p.exists() and tif_p.exists()):
        return None
    stack = tifffile.imread(str(tif_p))
    try:
        df = pd.read_csv(str(csv_p))
    except pd.errors.EmptyDataError:
        return np.zeros_like(stack, dtype=bool)
    if df.empty or "cluster_type" not in df.columns:
        return np.zeros_like(stack, dtype=bool)
    pole_ids = df[df["cluster_type"] == "pole"]["track_id"].tolist()
    if not pole_ids:
        return np.zeros_like(stack, dtype=bool)
    return np.isin(stack, pole_ids)


def per_frame_candidates(h2a_frame, nuclei_label, frame, mask_frame):
    """Return DataFrame of mitotic candidates in this frame."""
    if nuclei_label.max() == 0:
        return pd.DataFrame()
    # Per-nucleus props
    bg = float(h2a_frame[~mask_frame].mean()) if (~mask_frame).any() else 0.0
    rows = []
    for p in measure.regionprops(nuclei_label, intensity_image=h2a_frame.astype(np.float32)):
        mean_int = float(p.mean_intensity - bg)
        if mean_int < 0:
            mean_int = 0.0
        rows.append({
            "frame": frame,
            "nucleus_id": int(p.label),
            "y": float(p.centroid[0]),
            "x": float(p.centroid[1]),
            "area": int(p.area),
            "mean_intensity_bgsub": mean_int,
            "solidity": float(p.solidity),
            "eccentricity": float(p.eccentricity),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if len(df) < 5:
        return pd.DataFrame()
    int_thr = np.percentile(df["mean_intensity_bgsub"], INT_PERCENTILE)
    area_thr = np.percentile(df["area"], AREA_PERCENTILE)
    candidates = df[(df["mean_intensity_bgsub"] >= int_thr)
                     & (df["area"] <= area_thr)
                     & (df["solidity"] >= 0.85)]
    return candidates


def persist_filter(events, max_dist_px=15):
    """Keep candidates that have at least one neighbour candidate in the
    next frame within max_dist_px (i.e. mitotic-looking object persists)."""
    if events.empty:
        return events
    kept = []
    by_frame = {t: g for t, g in events.groupby("frame")}
    for t, g in by_frame.items():
        nextg = by_frame.get(t + 1, pd.DataFrame())
        for _, row in g.iterrows():
            if nextg.empty:
                continue
            d = np.hypot(nextg["y"] - row["y"], nextg["x"] - row["x"])
            if (d <= max_dist_px).any():
                kept.append(row.to_dict())
    return pd.DataFrame(kept)


def process_pescoid(cond, pid):
    nuclei_path = PHASE5A / "per_pescoid" / cond / pid / "nuclei_masks_stardist.tif"
    h2a_path = PHASE1 / "per_pescoid" / cond / pid / "h2a_aligned.tif"
    mask_path = PHASE1 / "per_pescoid" / cond / pid / "mask_aligned.tif"
    if not (nuclei_path.exists() and h2a_path.exists() and mask_path.exists()):
        return None
    nuclei = tifffile.imread(str(nuclei_path))
    h2a = tifffile.imread(str(h2a_path))
    mask = tifffile.imread(str(mask_path)) > 0
    T = nuclei.shape[0]
    pole = load_pole_stack(cond, pid)
    if pole is None:
        pole = np.zeros_like(mask, dtype=bool)

    candidates_all = []
    for t in range(T):
        if not mask[t].any() or nuclei[t].max() == 0:
            continue
        c = per_frame_candidates(h2a[t], nuclei[t], t, mask[t])
        if not c.empty:
            candidates_all.append(c)
    if not candidates_all:
        return pd.DataFrame()
    raw_events = pd.concat(candidates_all, ignore_index=True)
    events = persist_filter(raw_events)
    if events.empty:
        return events
    # Annotate hpf + inside pole
    events["hpf"] = events["frame"].apply(lambda f: HPF_START + f * HPF_INTERVAL)
    inside = []
    for _, row in events.iterrows():
        t = int(row["frame"])
        if t >= pole.shape[0]:
            inside.append(False); continue
        yi = int(np.clip(round(row["y"]), 0, pole.shape[1] - 1))
        xi = int(np.clip(round(row["x"]), 0, pole.shape[2] - 1))
        inside.append(bool(pole[t, yi, xi]))
    events["inside_pole"] = inside

    # Save per-pescoid
    pdir = PER / cond / pid
    pdir.mkdir(parents=True, exist_ok=True)
    events.to_csv(str(pdir / "mitosis_candidates.csv"), index=False)
    save_mitosis_map(cond, pid, events, mask, pole, h2a, pdir / f"{pid}_mitosis_map_v3.png")
    return events


def save_mitosis_map(cond, pid, events, mask, pole, h2a, out_path):
    T = h2a.shape[0]
    peak_window = [t for t in range(T)
                    if PEAK_HPF[0] <= HPF_START + t * HPF_INTERVAL <= PEAK_HPF[1]]
    if peak_window:
        areas = [int(pole[t].sum()) for t in peak_window]
        t_show = peak_window[int(np.argmax(areas))] if max(areas) > 0 else peak_window[-1]
    else:
        t_show = min(T - 1, int(round((14.0 - HPF_START) / HPF_INTERVAL)))
    img = h2a[t_show].astype(float)
    img = (img - img.min()) / (np.percentile(img, 99.5) - img.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(np.clip(img, 0, 1), cmap="gray")
    if mask[t_show].any():
        for c in measure.find_contours(mask[t_show].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "w-", lw=0.6, alpha=0.5)
    if pole[t_show].any():
        for c in measure.find_contours(pole[t_show].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "lime", lw=2)
    for _, ev in events.iterrows():
        h = float(ev["hpf"])
        if h < FORM_HPF[1]:
            c = "red"
        elif h < PEAK_HPF[1]:
            c = "orange"
        else:
            c = "gray"
        ax.scatter([ev["x"]], [ev["y"]], s=120, color=c, edgecolors="black",
                   lw=1.0, alpha=0.75, marker="*")
    ax.set_title(f"{cond} / {pid}  -  {len(events)} mitotic candidates\n"
                 f"red=formation 8-12h, orange=peak 12-16h, gray=late",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run():
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary = summary[summary["condition"].isin(CONDITIONS)].copy()
    # only process those that had StarDist nuclei
    sd = pd.read_csv(str(PHASE5A / "tables" / "stardist_nuclei_summary.csv"))
    summary = summary.merge(sd[["condition", "pescoid", "peak_nuclei_count"]],
                             on=["condition", "pescoid"])
    summary = summary[summary["peak_nuclei_count"] >= 15]
    print(f"Pescoids to process: {len(summary)}")

    all_events = []
    for _, r in summary.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        try:
            events = process_pescoid(cond, pid)
            if events is None or events.empty:
                continue
            events["condition"] = cond
            events["pescoid"] = pid
            all_events.append(events)
            print(f"  {cond}/{pid}: {len(events)} mitotic candidates")
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")
            import traceback; traceback.print_exc()

    if not all_events:
        print("No candidates.")
        return
    big = pd.concat(all_events, ignore_index=True)
    big = big.merge(summary[["condition", "pescoid", "phenotype_peak",
                             "n_mezzo_poles_peak"]],
                     on=["condition", "pescoid"])
    big.to_csv(str(TABLES / "phase5b3_v3_per_event.csv"), index=False)
    print(f"\nTotal mitotic candidates: {len(big)} across {big['pescoid'].nunique()} pescoids")

    per_p = big.groupby(["condition", "pescoid", "phenotype_peak",
                          "n_mezzo_poles_peak"]).agg(
        n_total=("hpf", "count"),
        n_formation=("hpf", lambda x: ((x >= FORM_HPF[0]) & (x <= FORM_HPF[1])).sum()),
        n_peak=("hpf", lambda x: ((x >= PEAK_HPF[0]) & (x <= PEAK_HPF[1])).sum()),
        n_inside_pole=("inside_pole", "sum"),
        n_outside_pole=("inside_pole", lambda x: (~x).sum()),
    ).reset_index()
    # also fill in zeros for pescoids that had no candidates
    full = summary[["condition", "pescoid", "phenotype_peak", "n_mezzo_poles_peak"]].merge(
        per_p, on=["condition", "pescoid", "phenotype_peak", "n_mezzo_poles_peak"],
        how="left").fillna(0)
    for col in ["n_total", "n_formation", "n_peak",
                "n_inside_pole", "n_outside_pole"]:
        full[col] = full[col].astype(int)
    full.to_csv(str(TABLES / "phase5b3_v3_per_pescoid_summary.csv"), index=False)

    # Plots
    _stratify_plot(full, "n_total",
                    "# mitotic candidates per pescoid",
                    "Total mitotic candidates (7-16 hpf) by phenotype",
                    "01_mitosis_count_by_phenotype.png")
    _stratify_plot(full, "n_formation",
                    "# mitotic candidates during 8-12 hpf",
                    "Mitotic candidates during pole formation (8-12 hpf)",
                    "04_mitoses_formation_by_phenotype.png")

    # Inside vs outside pole (paired)
    fig, ax = plt.subplots(figsize=(9, 6))
    cond_pos = {"P_ctrl": 0, "P_Activin_3-5hpf": 1}
    for cond in CONDITIONS:
        sub = full[full["condition"] == cond]
        if sub.empty:
            continue
        x_base = cond_pos[cond]
        n = len(sub)
        x_in = x_base - 0.2 + RNG.uniform(-0.06, 0.06, n)
        x_out = x_base + 0.2 + RNG.uniform(-0.06, 0.06, n)
        for (_, row), xi, xo in zip(sub.iterrows(), x_in, x_out):
            ax.plot([xi, xo], [row["n_inside_pole"], row["n_outside_pole"]],
                    "-", color=COND_COLOR[cond], alpha=0.35, lw=0.8)
            ax.scatter([xi], [row["n_inside_pole"]], s=55, color="lime",
                       edgecolors="black", lw=0.6, zorder=3)
            ax.scatter([xo], [row["n_outside_pole"]], s=55, color="#888",
                       edgecolors="black", lw=0.6, zorder=3)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel("# mitotic candidates")
    ax.set_title("Mitotic candidates INSIDE (lime) vs OUTSIDE (grey) mezzo+ pole",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_mitosis_inside_vs_outside_pole.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Rate over time
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.arange(7, 17, 0.5)
    for cond in CONDITIONS:
        sub = big[big["condition"] == cond]
        ax.hist(sub["hpf"], bins=bins, alpha=0.5,
                color=COND_COLOR[cond],
                label=f"{COND_LABEL[cond]} ({len(sub)} events, "
                      f"{sub['pescoid'].nunique()} pescoids)",
                edgecolor="black", linewidth=0.5)
    ax.axvspan(FORM_HPF[0], FORM_HPF[1], color="orange", alpha=0.07,
                label="formation 8-12 hpf")
    ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.07,
                label="peak 12-16 hpf")
    ax.set_xlabel("hpf")
    ax.set_ylabel("# mitotic candidates")
    ax.set_title("Mitotic candidates over time by condition", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "03_mitosis_rate_over_time.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Mitoses (formation window) vs final pole count
    fig, ax = plt.subplots(figsize=(9, 6))
    for cond in CONDITIONS:
        sub = full[full["condition"] == cond]
        x = sub["n_formation"] + RNG.uniform(-0.2, 0.2, len(sub))
        y = sub["n_mezzo_poles_peak"] + RNG.uniform(-0.15, 0.15, len(sub))
        colors = [PHENO_COLOR.get(p, "#888") for p in sub["phenotype_peak"]]
        ax.scatter(x, y, s=70, c=colors, edgecolors="black", lw=0.6,
                   alpha=0.85, marker="o" if cond == "P_ctrl" else "s",
                   label=COND_LABEL[cond])
    rho = full[["n_formation", "n_mezzo_poles_peak"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("# mitotic candidates during 8-12 hpf")
    ax.set_ylabel("# mezzo+ poles at peak")
    ax.set_title(f"Pole-formation mitoses vs final pole count "
                 f"(Spearman rho = {rho:+.2f})",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "05_mitoses_vs_pole_count.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Condition summary
    s = full.groupby("condition").agg(
        n_pescoids=("pescoid", "count"),
        median_total=("n_total", "median"),
        median_formation=("n_formation", "median"),
        median_inside_pole=("n_inside_pole", "median"),
        median_outside_pole=("n_outside_pole", "median"),
    ).round(2)
    s.to_csv(str(TABLES / "phase5b3_v3_condition_summary.csv"))
    print("\n=== Phase 5b-3 v3 condition summary ===")
    print(s.to_string())

    # Inside-vs-outside ratios per pescoid
    full["pole_enrichment"] = (full["n_inside_pole"] /
                                (full["n_inside_pole"] + full["n_outside_pole"]).clip(lower=1))
    print()
    print(f"Spearman(n_formation, n_mezzo_poles_peak) = "
          f"{full[['n_formation', 'n_mezzo_poles_peak']].corr(method='spearman').iloc[0,1]:+.2f}")
    print(f"Spearman(n_total,     n_mezzo_poles_peak) = "
          f"{full[['n_total', 'n_mezzo_poles_peak']].corr(method='spearman').iloc[0,1]:+.2f}")
    print(f"\nOutputs in: {OUT}/")


def _stratify_plot(df, ycol, ylabel, title, fname, ylim=None, hline=None):
    fig, ax = plt.subplots(figsize=(13, 6))
    positions = []; labels = []; colors = []; groups = []
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
            x = positions[i] + RNG.uniform(-0.1, 0.1, size=len(vals))
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    args = p.parse_args()
    run()
