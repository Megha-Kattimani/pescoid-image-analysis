"""
Phase 5b-3 - Mitosis detection from btrack output.

For each pescoid that has tracks (Phase 5b-2):
  - Detect mitosis events from track branching (a parent track ending and
    two child tracks starting at the same frame, near the parent's last
    position).
  - Tag each event with: hpf, position, inside_pole (yes/no),
    parent_track_id, daughter_ids.
  - Cross-condition / phenotype questions:
      1. Is mitosis rate higher in P_Activin vs P_ctrl?
      2. Are mitoses concentrated INSIDE mezzo+ poles vs outside?
      3. Do multipolar pescoids divide more during pole formation (8-12 hpf)?
      4. Spatial map: are mitoses spatially clustered (seeding poles) or
         distributed?

Outputs (mezzo_H2A_phase5b3/):
  per_pescoid/<cond>/<pid>/mitosis_events.csv
  per_pescoid/<cond>/<pid>/<pid>_mitosis_map.png
  plots/01_mitosis_rate_by_phenotype.png
  plots/02_mitosis_inside_vs_outside_pole.png
  plots/03_mitosis_rate_over_time.png
  plots/04_mitosis_spatial_clustering.png
  tables/phase5b3_per_pescoid_summary.csv
  tables/phase5b3_per_event.csv
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
PHASE5B2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b3")
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


def detect_mitoses(tdf):
    """A mitosis is a track whose parent != 0; the parent track ended and
    two daughter tracks started at the same frame, all near each other.
    Group rows by parent."""
    if "parent" not in tdf.columns:
        return pd.DataFrame()
    # Each daughter track has parent>0. Group daughters by parent.
    daughter_tids = tdf[tdf["parent"] > 0]["track_id"].unique()
    if len(daughter_tids) == 0:
        return pd.DataFrame()
    events = []
    for parent in tdf[tdf["parent"] > 0]["parent"].unique():
        daughters = tdf[tdf["parent"] == parent]["track_id"].unique()
        if len(daughters) < 2:
            continue
        # earliest frame of first daughter
        first_frames = [int(tdf[tdf["track_id"] == d]["frame"].min()) for d in daughters]
        t_mitosis = int(min(first_frames))
        # daughter positions at first frame
        positions = []
        for d, ff in zip(daughters, first_frames):
            row = tdf[(tdf["track_id"] == d) & (tdf["frame"] == ff)]
            if not row.empty:
                positions.append((float(row["y"].iloc[0]), float(row["x"].iloc[0])))
        if len(positions) < 2:
            continue
        arr = np.array(positions)
        events.append({
            "parent_track_id": int(parent),
            "daughter_track_ids": ",".join(str(int(d)) for d in daughters),
            "n_daughters": int(len(daughters)),
            "frame": t_mitosis,
            "hpf": HPF_START + t_mitosis * HPF_INTERVAL,
            "y": float(arr[:, 0].mean()),
            "x": float(arr[:, 1].mean()),
            "daughter_spread_px": float(np.hypot(arr[:, 0].std(), arr[:, 1].std())),
        })
    return pd.DataFrame(events)


def annotate_inside_pole(events, pole_stack):
    if events.empty:
        events["inside_pole"] = False
        return events
    insides = []
    for _, row in events.iterrows():
        t = int(row["frame"])
        yi = int(np.clip(round(row["y"]), 0, pole_stack.shape[1] - 1))
        xi = int(np.clip(round(row["x"]), 0, pole_stack.shape[2] - 1))
        if t < pole_stack.shape[0]:
            insides.append(bool(pole_stack[t, yi, xi]))
        else:
            insides.append(False)
    events["inside_pole"] = insides
    return events


def process_pescoid(cond, pid):
    tcsv = PHASE5B2 / "per_pescoid" / cond / pid / "tracks.csv"
    if not tcsv.exists():
        return None
    tdf = pd.read_csv(str(tcsv))
    pole = load_pole_stack(cond, pid)
    if pole is None:
        pole = np.zeros((tdf["frame"].max() + 1 if not tdf.empty else 1, 512, 512), bool)
    events = detect_mitoses(tdf)
    events = annotate_inside_pole(events, pole)

    pdir = PER / cond / pid
    pdir.mkdir(parents=True, exist_ok=True)
    events.to_csv(str(pdir / "mitosis_events.csv"), index=False)

    # Per-pescoid spatial map at one of the form-window frames
    if not events.empty:
        save_mitosis_map(cond, pid, events, pole, pdir / f"{pid}_mitosis_map.png")

    return events


def save_mitosis_map(cond, pid, events, pole_stack, out_path):
    """Map of mitosis events with mezzo+ pole contour at peak frame."""
    h2a = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "h2a_aligned.tif"))
    mask = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "mask_aligned.tif")) > 0
    # pick the peak frame (max pole area in 12-16 hpf)
    T = pole_stack.shape[0]
    peak_window = [t for t in range(T)
                    if PEAK_HPF[0] <= HPF_START + t * HPF_INTERVAL <= PEAK_HPF[1]]
    if peak_window:
        areas = [int(pole_stack[t].sum()) for t in peak_window]
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
    if pole_stack[t_show].any():
        for c in measure.find_contours(pole_stack[t_show].astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], "lime", lw=2)
    # plot events: red = pre-peak (formation), orange = at peak, gray = post
    for _, ev in events.iterrows():
        h = float(ev["hpf"])
        if h < FORM_HPF[1]:
            c = "red"; label = "formation (8-12h)"
        elif h < PEAK_HPF[1]:
            c = "orange"; label = "peak (12-16h)"
        else:
            c = "gray"; label = "late"
        ax.scatter([ev["x"]], [ev["y"]], s=140, color=c, edgecolors="black",
                   lw=1.2, alpha=0.9, marker="*")
    ax.set_title(f"{cond} / {pid}  -  {len(events)} mitoses\n"
                 f"red=formation 8-12h, orange=peak 12-16h, gray=late",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(pescoid_filter=None):
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary = summary[summary["condition"].isin(CONDITIONS)].copy()
    if pescoid_filter:
        summary = summary[summary["pescoid"].isin(pescoid_filter)]
    # only those with btrack tracks
    has_tracks = []
    for _, r in summary.iterrows():
        if (PHASE5B2 / "per_pescoid" / r["condition"] / r["pescoid"] / "tracks.csv").exists():
            has_tracks.append((r["condition"], r["pescoid"]))
    print(f"Pescoids with tracks: {len(has_tracks)}")

    all_events = []
    for cond, pid in has_tracks:
        events = process_pescoid(cond, pid)
        if events is None or events.empty:
            continue
        events["condition"] = cond
        events["pescoid"] = pid
        all_events.append(events)

    if not all_events:
        print("No mitosis events.")
        return
    big = pd.concat(all_events, ignore_index=True)
    big = big.merge(summary[["condition", "pescoid", "phenotype_peak", "n_mezzo_poles_peak"]],
                     on=["condition", "pescoid"])
    big.to_csv(str(TABLES / "phase5b3_per_event.csv"), index=False)
    print(f"Total mitosis events: {len(big)} across {big['pescoid'].nunique()} pescoids")

    # Per-pescoid metrics
    per_p = big.groupby(["condition", "pescoid", "phenotype_peak",
                         "n_mezzo_poles_peak"]).agg(
        n_mitoses_total=("hpf", "count"),
        n_mitoses_formation=("hpf", lambda x: ((x >= FORM_HPF[0]) & (x <= FORM_HPF[1])).sum()),
        n_mitoses_peak=("hpf", lambda x: ((x >= PEAK_HPF[0]) & (x <= PEAK_HPF[1])).sum()),
        n_mitoses_inside_pole=("inside_pole", "sum"),
        n_mitoses_outside_pole=("inside_pole", lambda x: (~x).sum()),
    ).reset_index()
    # Append zeros for pescoids with tracks but no mitoses found
    summary_with_tracks = pd.DataFrame(has_tracks, columns=["condition", "pescoid"]).merge(
        summary[["condition", "pescoid", "phenotype_peak", "n_mezzo_poles_peak"]],
        on=["condition", "pescoid"])
    per_p = summary_with_tracks.merge(per_p, on=["condition", "pescoid",
                                                  "phenotype_peak", "n_mezzo_poles_peak"],
                                       how="left").fillna(0)
    for col in ["n_mitoses_total", "n_mitoses_formation", "n_mitoses_peak",
                "n_mitoses_inside_pole", "n_mitoses_outside_pole"]:
        per_p[col] = per_p[col].astype(int)
    per_p.to_csv(str(TABLES / "phase5b3_per_pescoid_summary.csv"), index=False)

    # ----- Plots -----
    # 01 mitosis count by phenotype
    fig, ax = plt.subplots(figsize=(13, 6))
    _stratify(ax, per_p, "n_mitoses_total",
              "# mitoses per pescoid (7-16 hpf)",
              "Total mitosis events per pescoid, by phenotype")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_mitosis_rate_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  01_mitosis_rate_by_phenotype.png")

    # 02 mitosis inside vs outside pole (paired)
    fig, ax = plt.subplots(figsize=(9, 6))
    cond_pos = {"P_ctrl": 0, "P_Activin_3-5hpf": 1}
    for cond in CONDITIONS:
        sub = per_p[per_p["condition"] == cond]
        if sub.empty:
            continue
        x_base = cond_pos[cond]
        n = len(sub)
        x_in = x_base - 0.2 + RNG.uniform(-0.06, 0.06, n)
        x_out = x_base + 0.2 + RNG.uniform(-0.06, 0.06, n)
        for (_, row), xi, xo in zip(sub.iterrows(), x_in, x_out):
            ax.plot([xi, xo], [row["n_mitoses_inside_pole"], row["n_mitoses_outside_pole"]],
                    "-", color=COND_COLOR[cond], alpha=0.35, lw=0.8)
            ax.scatter([xi], [row["n_mitoses_inside_pole"]], s=60, color="lime",
                       edgecolors="black", lw=0.6, zorder=3)
            ax.scatter([xo], [row["n_mitoses_outside_pole"]], s=60, color="#888",
                       edgecolors="black", lw=0.6, zorder=3)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel("# mitoses (counts per pescoid)")
    ax.set_title("Mitosis events INSIDE mezzo+ pole (lime) vs OUTSIDE (grey)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_mitosis_inside_vs_outside_pole.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  02_mitosis_inside_vs_outside_pole.png")

    # 03 mitosis rate over time (hpf histogram per condition)
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.arange(7, 17, 0.5)
    for cond in CONDITIONS:
        sub = big[big["condition"] == cond]
        ax.hist(sub["hpf"], bins=bins, alpha=0.5,
                color=COND_COLOR[cond], label=f"{COND_LABEL[cond]} (n_events={len(sub)})",
                edgecolor="black", linewidth=0.5)
    ax.axvspan(FORM_HPF[0], FORM_HPF[1], color="orange", alpha=0.07, label="formation 8-12 hpf")
    ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.07, label="peak 12-16 hpf")
    ax.set_xlabel("hpf")
    ax.set_ylabel("# mitosis events (across all pescoids of condition)")
    ax.set_title("Mitosis events over time by condition", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "03_mitosis_rate_over_time.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  03_mitosis_rate_over_time.png")

    # 04 mitosis rate by phenotype during formation window
    fig, ax = plt.subplots(figsize=(13, 6))
    _stratify(ax, per_p, "n_mitoses_formation",
              "# mitoses during 8-12 hpf",
              "Mitoses during pole-formation window (8-12 hpf), by phenotype")
    plt.tight_layout()
    plt.savefig(str(PLOTS / "04_mitoses_formation_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  04_mitoses_formation_by_phenotype.png")

    # Condition summary
    s = per_p.groupby("condition").agg(
        n_pescoids=("pescoid", "count"),
        median_mitoses_total=("n_mitoses_total", "median"),
        median_mitoses_formation=("n_mitoses_formation", "median"),
        median_mitoses_inside_pole=("n_mitoses_inside_pole", "median"),
        median_mitoses_outside_pole=("n_mitoses_outside_pole", "median"),
    ).round(2)
    s.to_csv(str(TABLES / "phase5b3_condition_summary.csv"))
    print("\n=== Phase 5b-3 condition summary ===")
    print(s.to_string())

    # Spearman: mitosis count during formation vs final pole count
    rho = per_p[["n_mitoses_formation", "n_mezzo_poles_peak"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman(mitoses 8-12 hpf, n_mezzo_poles_peak) = {rho:+.2f}")

    print(f"\nPlots in: {PLOTS}/   Tables in: {TABLES}/")


def _stratify(ax, df, ycol, ylabel, title, ylim=None, hline=None):
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pids", nargs="*")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.pids:
        run(pescoid_filter=args.pids)
    elif args.all:
        run()
    else:
        run()
