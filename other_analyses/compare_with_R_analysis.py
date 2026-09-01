"""
Compare our pipeline results vs Nick's existing R/Fiji Analysis_V2.

For each mezzo condition (7 total in their analysis):
  - Load Results.csv (Fiji/ImageJ measurements)
  - Aggregate per timepoint: mean AR, mean major, GFP+ area, GFP+ fraction
  - Load our morphology_over_time.csv + mezzo_expression_over_time.csv
  - Plot side-by-side comparison
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_ANALYSIS = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250212_mezzo_E-Cad_Chiron_Activin\Analysis_V2"
)
OUR_ANALYSIS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\250212_mezzo_E-Cad")
OUT = OUR_ANALYSIS / "comparison_with_R"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "P_mezzo_ctrl",
    "P_mezzo_3-5hpf_Act",
    "P_mezzo_3-5hpf_Chi",
    "P_mezzo_3-5hpf_Act-Chi",
    "P_mezzo_5-7hpf_Act",
    "P_mezzo_5-7hpf_Chi",
    "P_mezzo_5-7hpf_Act-Chi",
]
LABELS = {
    "P_mezzo_ctrl": "Mezzo ctrl",
    "P_mezzo_3-5hpf_Act": "Mezzo Act 3-5h",
    "P_mezzo_3-5hpf_Chi": "Mezzo Chi 3-5h",
    "P_mezzo_3-5hpf_Act-Chi": "Mezzo Act+Chi 3-5h",
    "P_mezzo_5-7hpf_Act": "Mezzo Act 5-7h",
    "P_mezzo_5-7hpf_Chi": "Mezzo Chi 5-7h",
    "P_mezzo_5-7hpf_Act-Chi": "Mezzo Act+Chi 5-7h",
}

HPF_START = 6.0
HPF_INTERVAL = 0.4
PX_UM = 0.775  # ImageJ pixel size — adjust if needed


def load_R_analysis(condition):
    """Aggregate Fiji/ImageJ Results.csv to per-timepoint values per pescoid."""
    csv_path = R_ANALYSIS / condition / "Results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(str(csv_path))
    # Extract pescoid ID and channel from label
    df["channel"] = df["Label"].str.contains("Mask_GFP").map({True: "GFP", False: "BF"})
    df["pescoid"] = df["Label"].str.extract(r"(G\d+)")
    df["time"] = df["Slice"]
    df["hpf"] = HPF_START + (df["time"] - 1) * HPF_INTERVAL  # Slice is 1-indexed

    # BF: one row per pescoid per timepoint (whole pescoid)
    bf = df[df["channel"] == "BF"].copy()
    bf_agg = bf.groupby(["pescoid", "time", "hpf"]).agg(
        bf_area=("Area", "first"),
        major=("Major", "first"),
        minor=("Minor", "first"),
        ar=("AR", "first"),
    ).reset_index()

    # GFP: multiple rows per pescoid per timepoint (one per GFP+ region)
    gfp = df[df["channel"] == "GFP"].copy()
    gfp_agg = gfp.groupby(["pescoid", "time", "hpf"]).agg(
        gfp_total_area=("Area", "sum"),
        n_gfp_regions=("Area", "count"),
    ).reset_index()

    merged = bf_agg.merge(gfp_agg, on=["pescoid", "time", "hpf"], how="left")
    merged["gfp_fraction"] = merged["gfp_total_area"] / merged["bf_area"]
    return merged


def load_our_analysis(condition):
    """Load per-pescoid timeseries from our pipeline."""
    cond_dir = OUR_ANALYSIS / condition
    if not cond_dir.exists():
        return None
    rows = []
    for sample_dir in sorted(cond_dir.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name == "comparison_plots":
            continue
        morph_p = sample_dir / "morphology_over_time.csv"
        mezzo_p = sample_dir / "mezzo_expression_over_time.csv"
        if not morph_p.exists():
            continue
        m = pd.read_csv(str(morph_p))
        z = pd.read_csv(str(mezzo_p)) if mezzo_p.exists() else None
        pid = re.search(r"(G\d+)", sample_dir.name)
        pid = pid.group(1) if pid else sample_dir.name
        m["pescoid"] = pid
        m["hpf"] = HPF_START + m["time"] * HPF_INTERVAL
        if z is not None:
            z["pescoid"] = pid
            # Use whichever GFP+ area column exists in the mezzo CSV
            if "gfp_pos_area" not in z.columns and "gfp_positive_area" in z.columns:
                z["gfp_pos_area"] = z["gfp_positive_area"]
            cols = ["time"] + [c for c in ["gfp_fraction", "gfp_mean_intensity", "gfp_pos_area"]
                               if c in z.columns]
            m = m.merge(z[cols], on="time", how="left")
        rows.append(m)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def plot_comparison(metric, our_col, R_col, ylabel, fname):
    """6 conditions x 2 panels (R vs ours) — one figure per metric."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)
    axes = axes.flatten()

    for i, cond in enumerate(CONDITIONS):
        ax = axes[i]
        R_df = load_R_analysis(cond)
        ours = load_our_analysis(cond)

        if R_df is not None and R_col in R_df.columns:
            R_mean = R_df.groupby("hpf")[R_col].mean()
            R_std = R_df.groupby("hpf")[R_col].std()
            ax.plot(R_mean.index, R_mean.values, color="#e41a1c", lw=2, label="Fiji/R analysis", marker="o", ms=3)
            ax.fill_between(R_mean.index, R_mean - R_std, R_mean + R_std, color="#e41a1c", alpha=0.15)

        if ours is not None and our_col in ours.columns:
            our_mean = ours.groupby("hpf")[our_col].mean()
            our_std = ours.groupby("hpf")[our_col].std()
            ax.plot(our_mean.index, our_mean.values, color="#377eb8", lw=2, label="Our pipeline", marker="s", ms=3)
            ax.fill_between(our_mean.index, our_mean - our_std, our_mean + our_std, color="#377eb8", alpha=0.15)

        ax.set_title(LABELS[cond], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [hpf]")
        if i % 4 == 0:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(len(CONDITIONS), len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Comparison: Fiji/R vs Our pipeline — {metric}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(OUT / fname), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")


plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 10,
})


def main():
    print(f"Comparing {OUR_ANALYSIS} vs {R_ANALYSIS}")
    print(f"Output: {OUT}/\n")

    plot_comparison("Aspect Ratio", "aspect_ratio", "ar",
                    "Aspect Ratio", "01_AR_comparison.png")
    plot_comparison("Major Axis", "major_axis", "major",
                    "Major Axis [px]", "02_MajorAxis_comparison.png")
    plot_comparison("Pescoid Area", "area", "bf_area",
                    "Pescoid Area [px]", "03_Area_comparison.png")
    plot_comparison("GFP+ fraction", "gfp_fraction", "gfp_fraction",
                    "GFP+ fraction", "04_GFPfraction_comparison.png")
    plot_comparison("GFP+ area", "gfp_pos_area", "gfp_total_area",
                    "GFP+ area [px]", "05_GFParea_comparison.png")

    # Summary stats: final timepoint comparison
    print("\nFinal-timepoint summary:")
    print(f'{"Condition":<22s} {"Metric":<20s} {"Fiji/R":>12s} {"Ours":>12s} {"diff%":>8s}')
    print("-" * 80)
    rows = []
    for cond in CONDITIONS:
        R_df = load_R_analysis(cond)
        ours = load_our_analysis(cond)
        if R_df is None or ours is None:
            continue
        # Last 3 frames mean
        R_late = R_df[R_df["time"] >= R_df["time"].max() - 2]
        our_late = ours[ours["time"] >= ours["time"].max() - 2]
        for r_col, our_col, name in [
            ("ar", "aspect_ratio", "AR"),
            ("major", "major_axis", "Major"),
            ("bf_area", "area", "Area"),
            ("gfp_fraction", "gfp_fraction", "GFP frac"),
        ]:
            if r_col not in R_late.columns or our_col not in our_late.columns:
                continue
            r_val = R_late[r_col].mean()
            o_val = our_late[our_col].mean()
            if pd.isna(r_val) or pd.isna(o_val) or r_val == 0:
                continue
            delta = (o_val - r_val) / r_val * 100
            print(f'{LABELS[cond]:<22s} {name:<20s} {r_val:>12.2f} {o_val:>12.2f} {delta:>7.1f}%')
            rows.append({"condition": LABELS[cond], "metric": name,
                         "fiji_r": r_val, "ours": o_val, "delta_pct": delta})

    pd.DataFrame(rows).to_csv(str(OUT / "summary_table.csv"), index=False)
    print(f"\nSummary saved: {OUT}/summary_table.csv")


if __name__ == "__main__":
    main()
