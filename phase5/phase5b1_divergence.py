"""
Phase 5b-1 - Local convergence/divergence maps from PIV.

For each pescoid the saved PIV grid (Phase 5a) gives a vector field (u, v) over
a regular grid every frame pair. Here we compute:
  - div = du/dx + dv/dy    (negative = convergence, positive = divergence)
  - curl = dv/dx - du/dy   (rotation)

For each pescoid:
  - per-frame mean div inside mezzo+ pole vs outside
  - per-frame mean |curl| inside vs outside
  - peak-frame divergence heatmap (overlay on H2A) with pole contour
For the dataset:
  - paired in-vs-out divergence per pescoid
  - phenotype-stratified divergence
  - convergence trajectory over time

Inputs:
  mezzo_H2A_phase5a/per_pescoid/<cond>/<pid>/piv_{u,v,grid_x,grid_y}.npy
  mezzo_H2A_phase1/per_pescoid/<cond>/<pid>/{h2a,mask}_aligned.tif
  mezzo_H2A_phase2/per_pescoid/<cond>/<pid>/mezzo_tracks.{csv,tif}
  mezzo_H2A_phase2/phenotype_summary.csv

Outputs (mezzo_H2A_phase5b1/):
  per_pescoid/<cond>/<pid>/{div_field,curl_field}.npy
  per_pescoid/<cond>/<pid>/<pid>_5b1_peak.png
  plots/01_div_inside_vs_outside_paired.png
  plots/02_curl_inside_vs_outside_paired.png
  plots/03_div_by_phenotype.png
  plots/04_div_trajectory_inside_vs_outside.png
  plots/05_convergence_fraction_by_phenotype.png
  tables/phase5b1_per_pescoid_summary.csv
  tables/phase5b1_per_frame.csv
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
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5b1")
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
PEAK_HPF = (12.0, 16.0)
FORM_HPF = (8.0, 12.0)   # pole-formation window
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


# ---------------------------------------------------------------------------
def jitter(n, w=0.15):
    return RNG.uniform(-w, w, size=n)


def bootstrap_ci(values, n=300):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                      for _ in range(n)])
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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


# ---------------------------------------------------------------------------
def compute_div_curl(u, v, x_step, y_step):
    """Central-difference divergence and curl on a regular grid (T-1, ny, nx)."""
    # du/dx along axis 2 (x), dv/dy along axis 1 (y)
    du_dy, du_dx = np.gradient(u, y_step, x_step, axis=(1, 2))
    dv_dy, dv_dx = np.gradient(v, y_step, x_step, axis=(1, 2))
    div = du_dx + dv_dy
    curl = dv_dx - du_dy
    return div, curl


def grid_inside_mask(grid_x, grid_y, mask):
    yi = np.clip(np.round(grid_y).astype(int), 0, mask.shape[0] - 1)
    xi = np.clip(np.round(grid_x).astype(int), 0, mask.shape[1] - 1)
    return mask[yi, xi]


# ---------------------------------------------------------------------------
def process_pescoid(cond, pid):
    src5a = PHASE5A / "per_pescoid" / cond / pid
    if not (src5a / "piv_u.npy").exists():
        return None
    u = np.load(str(src5a / "piv_u.npy"))    # (T-1, ny, nx)
    v = np.load(str(src5a / "piv_v.npy"))
    gx = np.load(str(src5a / "piv_grid_x.npy"))  # (ny, nx)
    gy = np.load(str(src5a / "piv_grid_y.npy"))

    # grid spacing in pixels
    x_step = float(gx[0, 1] - gx[0, 0]) if gx.shape[1] > 1 else 1.0
    y_step = float(gy[1, 0] - gy[0, 0]) if gy.shape[0] > 1 else 1.0

    div, curl = compute_div_curl(u, v, x_step, y_step)
    # NaN out points where openpiv flagged with sentinel large values
    sentinel = np.abs(u) + np.abs(v) > 1e3
    div[sentinel] = np.nan
    curl[sentinel] = np.nan

    # Load mask + pole stack
    h2a = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "h2a_aligned.tif"))
    mask = tifffile.imread(str(PHASE1 / "per_pescoid" / cond / pid / "mask_aligned.tif")) > 0
    pole_stack = load_pole_stack(cond, pid)
    if pole_stack is None:
        pole_stack = np.zeros_like(mask, dtype=bool)

    T_pairs = u.shape[0]
    rows = []
    for t in range(T_pairs):
        if not mask[t].any():
            rows.append({"time": t, "hpf": HPF_START + t * HPF_INTERVAL,
                         "div_inside_pole": np.nan, "div_outside_pole": np.nan,
                         "abs_curl_inside_pole": np.nan, "abs_curl_outside_pole": np.nan,
                         "pole_pixels": 0, "non_pole_pixels": 0})
            continue
        pole_in = pole_stack[t] & mask[t]
        non_pole_in = mask[t] & ~pole_in
        in_g = grid_inside_mask(gx, gy, pole_in)
        out_g = grid_inside_mask(gx, gy, non_pole_in)
        d = div[t]
        c = np.abs(curl[t])
        rows.append({
            "time": t, "hpf": HPF_START + t * HPF_INTERVAL,
            "div_inside_pole":  float(np.nanmean(d[in_g])) if in_g.any() else np.nan,
            "div_outside_pole": float(np.nanmean(d[out_g])) if out_g.any() else np.nan,
            "abs_curl_inside_pole":  float(np.nanmean(c[in_g])) if in_g.any() else np.nan,
            "abs_curl_outside_pole": float(np.nanmean(c[out_g])) if out_g.any() else np.nan,
            "pole_pixels": int(pole_in.sum()),
            "non_pole_pixels": int(non_pole_in.sum()),
        })

    df = pd.DataFrame(rows)

    # Peak frame: pole-bearing frame in 12-16 hpf with max pole area
    peak_t = None
    in_peak = [t for t in range(T_pairs)
               if PEAK_HPF[0] <= HPF_START + t * HPF_INTERVAL <= PEAK_HPF[1]
               and df.loc[t, "pole_pixels"] > 0]
    if in_peak:
        peak_t = max(in_peak, key=lambda t: df.loc[t, "pole_pixels"])

    # save
    pdir = PER / cond / pid
    pdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(pdir / "div_curl_per_frame.csv"), index=False)
    np.save(str(pdir / "div_field.npy"), div)
    np.save(str(pdir / "curl_field.npy"), curl)
    if peak_t is not None:
        save_peak_div_heatmap(cond, pid, peak_t, h2a, mask, pole_stack,
                              u, v, gx, gy, div, curl,
                              pdir / f"{pid}_5b1_peak.png")
    return df


def save_peak_div_heatmap(cond, pid, t, h2a, mask, pole_stack, u, v, gx, gy,
                          div, curl, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    pole = pole_stack[t] & mask[t]
    # H2A with quiver overlay
    h2a_n = (h2a[t] - h2a[t].min()) / (np.percentile(h2a[t], 99.5) - h2a[t].min() + 1e-9)
    axes[0].imshow(np.clip(h2a_n, 0, 1), cmap="gray")
    in_g = grid_inside_mask(gx, gy, mask[t])
    if in_g.any():
        axes[0].quiver(gx[in_g], gy[in_g], u[t][in_g], -v[t][in_g],
                       color="yellow", scale=80, headwidth=3, headlength=4,
                       pivot="middle", alpha=0.9)
    if mask[t].any():
        for c in measure.find_contours(mask[t].astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "w-", lw=0.8, alpha=0.6)
    if pole.any():
        for c in measure.find_contours(pole.astype(float), 0.5):
            axes[0].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[0].set_title("H2A + PIV vectors + pole (lime)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Divergence heatmap (interpolated to image grid)
    from scipy.interpolate import griddata
    yy, xx = np.mgrid[: mask.shape[1], : mask.shape[2]]
    points = np.column_stack([gy.ravel(), gx.ravel()])
    div_img = griddata(points, div[t].ravel(), (yy, xx), method="linear")
    div_img = np.where(mask[t], div_img, np.nan)
    vmax = np.nanpercentile(np.abs(div_img), 95) if np.isfinite(div_img).any() else 0.1
    im = axes[1].imshow(div_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if pole.any():
        for c in measure.find_contours(pole.astype(float), 0.5):
            axes[1].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[1].set_title(f"Divergence  (blue = convergence, red = divergence)\n"
                      f"vmax = {vmax:.3f} 1/frame",
                      fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.75, label="div [1/frame]")

    # |Curl| heatmap
    curl_img = griddata(points, np.abs(curl[t]).ravel(), (yy, xx), method="linear")
    curl_img = np.where(mask[t], curl_img, np.nan)
    im2 = axes[2].imshow(curl_img, cmap="viridis")
    if pole.any():
        for c in measure.find_contours(pole.astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], "lime", lw=2)
    axes[2].set_title("|Curl| (rotation magnitude)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.75, label="|curl| [1/frame]")

    plt.suptitle(f"{cond} / {pid}   -   peak t={t}  ({HPF_START + t * HPF_INTERVAL:.1f} hpf)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(pescoid_filter=None):
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    if pescoid_filter:
        summary = summary[summary["pescoid"].isin(pescoid_filter)]
    print(f"Processing {len(summary)} pescoids...")
    all_frames = []
    for _, row in summary.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        try:
            df = process_pescoid(cond, pid)
            if df is None:
                print(f"  SKIP {cond}/{pid}: no PIV data")
                continue
            df["condition"] = cond
            df["pescoid"] = pid
            all_frames.append(df)
            print(f"  {cond}/{pid}: T_pairs={len(df)}")
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")
            import traceback; traceback.print_exc()

    if not all_frames:
        print("No pescoids processed.")
        return
    big = pd.concat(all_frames, ignore_index=True)
    big = big.merge(summary[["condition", "pescoid", "phenotype_peak"]],
                     on=["condition", "pescoid"], how="left")
    big.to_csv(str(TABLES / "phase5b1_per_frame.csv"), index=False)

    # Per-pescoid summary in peak window (12-16 hpf) and formation window (8-12 hpf)
    def mean_window(big, lo, hi, suffix):
        sub = big[(big["hpf"] >= lo) & (big["hpf"] <= hi)]
        return sub.groupby(["condition", "pescoid", "phenotype_peak"]).agg(
            **{
                f"div_in_{suffix}":  ("div_inside_pole",  lambda x: x.dropna().mean()),
                f"div_out_{suffix}": ("div_outside_pole", lambda x: x.dropna().mean()),
                f"curl_in_{suffix}":  ("abs_curl_inside_pole",  lambda x: x.dropna().mean()),
                f"curl_out_{suffix}": ("abs_curl_outside_pole", lambda x: x.dropna().mean()),
            }
        ).reset_index()
    pk = mean_window(big, PEAK_HPF[0], PEAK_HPF[1], "peak")
    fo = mean_window(big, FORM_HPF[0], FORM_HPF[1], "form")
    per_p = pk.merge(fo, on=["condition", "pescoid", "phenotype_peak"], how="outer")
    per_p["div_diff_peak"] = per_p["div_in_peak"] - per_p["div_out_peak"]
    per_p["div_diff_form"] = per_p["div_in_form"] - per_p["div_out_form"]
    per_p.to_csv(str(TABLES / "phase5b1_per_pescoid_summary.csv"), index=False)

    # ---- Plots ----
    # 01 div inside vs outside paired (formation window 8-12 hpf, when convergence should happen)
    paired_plot(per_p, "div_in_form", "div_out_form",
                 "Divergence [1/frame]",
                 "Divergence INSIDE vs OUTSIDE mezzo+ pole (formation window 8-12 hpf)\n"
                 "Negative = convergence (cells flowing IN)",
                 "01_div_inside_vs_outside_formation.png")
    print("  01_div_inside_vs_outside_formation.png")

    # 02 div inside vs outside paired (peak window 12-16 hpf)
    paired_plot(per_p, "div_in_peak", "div_out_peak",
                 "Divergence [1/frame]",
                 "Divergence INSIDE vs OUTSIDE mezzo+ pole (peak window 12-16 hpf)",
                 "02_div_inside_vs_outside_peak.png")
    print("  02_div_inside_vs_outside_peak.png")

    # 03 div by phenotype (formation window)
    stratify_by_phenotype(per_p, "div_in_form",
                           "Mean divergence inside pole [1/frame]",
                           "Mean div INSIDE pole during pole formation (8-12 hpf)",
                           "03_div_inside_by_phenotype_formation.png", hline=0.0)
    print("  03_div_inside_by_phenotype_formation.png")

    # 04 div trajectory inside vs outside per condition
    div_trajectory(big, "04_div_trajectory_inside_vs_outside.png")
    print("  04_div_trajectory_inside_vs_outside.png")

    # 05 fraction-of-time convergent inside pole, by phenotype
    convergence_fraction = (big[big["div_inside_pole"].notna()]
                            .assign(is_conv=lambda d: d["div_inside_pole"] < 0)
                            .groupby(["condition", "pescoid"])
                            .agg(frac_conv=("is_conv", "mean")).reset_index())
    convergence_fraction = convergence_fraction.merge(
        summary[["condition", "pescoid", "phenotype_peak"]], on=["condition", "pescoid"])
    stratify_by_phenotype(convergence_fraction, "frac_conv",
                           "Fraction of frames with div<0 inside pole",
                           "Fraction of time the pole region is CONVERGENT (div<0)",
                           "05_convergence_fraction_by_phenotype.png",
                           ylim=(0, 1), hline=0.5)
    print("  05_convergence_fraction_by_phenotype.png")

    # Condition summary
    s = per_p.groupby("condition").agg(
        n=("pescoid", "count"),
        div_in_form_mean=("div_in_form", "mean"),
        div_out_form_mean=("div_out_form", "mean"),
        div_in_peak_mean=("div_in_peak", "mean"),
        div_out_peak_mean=("div_out_peak", "mean"),
        pct_convergent_in_pole_form=("div_in_form",
                                     lambda x: (x.dropna() < 0).mean() * 100),
    ).round(4)
    s.to_csv(str(TABLES / "phase5b1_condition_summary.csv"))
    print("\n=== Phase 5b-1 condition summary ===")
    print(s.to_string())
    print(f"\nPlots in: {PLOTS}/   Tables in: {TABLES}/")


def paired_plot(df, ycol_in, ycol_out, ylabel, title, fname):
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
            ax.plot([xi, xo], [row[ycol_in], row[ycol_out]], "-",
                    color=COND_COLOR[cond], alpha=0.35, lw=0.8)
            ax.scatter([xi], [row[ycol_in]], s=55, color="lime",
                       edgecolors="black", lw=0.6, zorder=3)
            ax.scatter([xo], [row[ycol_out]], s=55, color="#888",
                       edgecolors="black", lw=0.6, zorder=3)
    ax.axhline(0, ls="--", color="k", lw=1, alpha=0.5)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="lime", lw=0, markeredgecolor="black",
               markersize=9, label="inside pole"),
        Line2D([0], [0], marker="o", color="#888", lw=0, markeredgecolor="black",
               markersize=9, label="outside"),
    ]
    ax.legend(handles=handles, loc="best")
    plt.tight_layout()
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def stratify_by_phenotype(df, ycol, ylabel, title, fname, ylim=None, hline=None):
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


def div_trajectory(big, fname):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axi, cond in enumerate(CONDITIONS):
        ax = axes[axi]
        sub = big[big["condition"] == cond]
        for series, color, label in [("div_inside_pole", "lime", "inside"),
                                      ("div_outside_pole", "#888", "outside")]:
            xs, ms, los, his = [], [], [], []
            for x, g in sub.groupby("hpf"):
                m, lo, hi = bootstrap_ci(g[series].dropna().values)
                xs.append(x); ms.append(m); los.append(lo); his.append(hi)
            xs = np.array(xs); ms = np.array(ms); los = np.array(los); his = np.array(his)
            ax.fill_between(xs, los, his, color=color, alpha=0.25)
            ax.plot(xs, ms, color=color, lw=2.5, label=label)
        ax.axhline(0, ls="--", color="k", lw=1, alpha=0.5)
        ax.axvspan(FORM_HPF[0], FORM_HPF[1], color="orange", alpha=0.07,
                    label="formation 8-12 hpf")
        ax.axvspan(PEAK_HPF[0], PEAK_HPF[1], color="yellow", alpha=0.07,
                    label="peak 12-16 hpf")
        ax.set_xlabel("Time [hpf]")
        ax.set_title(f"{COND_LABEL[cond]}", fontweight="bold")
        ax.legend(loc="best", fontsize=9)
    axes[0].set_ylabel("Mean divergence [1/frame]   (negative=convergence)")
    plt.suptitle("Divergence inside vs outside mezzo+ pole, over time",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(PLOTS / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pids", nargs="*", help="pescoid IDs")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.pids:
        run(pescoid_filter=args.pids)
    elif args.all:
        run()
    else:
        p.print_help()
