"""
Comprehensive final analysis for the 250402 mezzo-LynTom Activin dataset.

Pipeline:
  1. Reuse existing masks (Lyn_mezzo_yen_fixed/) and morphology/mezzo CSVs
  2. Run tissue tension (structure tensor) on ALL 4 conditions (E_ctrl, E_Activin, P_ctrl, P_Activin)
  3. Generate per-pescoid registration montages (all timepoints in one image)
  4. Identify double-pole pescoids and save their kymograph + montage previews
  5. Generate comparison plots:
        - All 4 conditions: aspect ratio, GFP fraction, # poles, tissue tension
        - P only (P_ctrl vs P_Activin): same metrics, separately

Output: Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_final\\
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from skimage import filters as skf, measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_ROOT = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF"
)
EXISTING = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_yen_fixed")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_final")
OUT.mkdir(parents=True, exist_ok=True)
DOUBLE = OUT / "double_pole_pescoids"
DOUBLE.mkdir(parents=True, exist_ok=True)
COMP_ALL = OUT / "comparison" / "all_4_conditions"
COMP_P = OUT / "comparison" / "P_only"
COMP_ALL.mkdir(parents=True, exist_ok=True)
COMP_P.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["E_ctrl", "E_Activin_3-5hpf", "P_ctrl", "P_Activin_3-5hpf"]
P_CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COLORS = {
    "E_ctrl": "#888888",
    "E_Activin_3-5hpf": "#e41a1c",
    "P_ctrl": "#377eb8",
    "P_Activin_3-5hpf": "#ff7f00",
}
LABELS = {
    "E_ctrl": "E ctrl",
    "E_Activin_3-5hpf": "E Activin 3-5h",
    "P_ctrl": "P ctrl",
    "P_Activin_3-5hpf": "P Activin 3-5h",
}

HPF_START = 6.0
HPF_INTERVAL = 0.4
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def z_project_bf(z_stack):
    if z_stack.ndim == 2:
        return z_stack.astype(float)
    Z = z_stack.shape[0]
    scores = []
    for z in range(Z):
        img = z_stack[z].astype(float)
        dx = img[:, 2:] - img[:, :-2]
        dy = img[2:, :] - img[:-2, :]
        scores.append(np.mean(dx**2) + np.mean(dy**2))
    return z_stack[int(np.argmax(scores))].astype(float)


def norm_pct(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def structure_tensor_fields(img, sigma_grad=5, sigma_tensor=10):
    ax_y = ndi.gaussian_filter(img, sigma=sigma_grad, order=(1, 0))
    ax_x = ndi.gaussian_filter(img, sigma=sigma_grad, order=(0, 1))
    Jxx = ndi.gaussian_filter(ax_x * ax_x, sigma=sigma_tensor)
    Jyy = ndi.gaussian_filter(ax_y * ax_y, sigma=sigma_tensor)
    Jxy = ndi.gaussian_filter(ax_x * ax_y, sigma=sigma_tensor)
    trace = Jxx + Jyy
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2
    coherence = np.where(lam1 + lam2 > 0,
                         (lam1 - lam2) / (lam1 + lam2 + 1e-10), 0.0)
    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    return orientation, coherence


def nematic_order(orient, coh, mask):
    if not mask.any():
        return 0.0
    a = orient[mask]
    w = coh[mask]
    w = w / max(w.sum(), 1e-10)
    return float(np.hypot(np.sum(w * np.cos(2 * a)), np.sum(w * np.sin(2 * a))))


def cortex_minus_interior(coh, mask, thickness=15):
    interior = morphology.binary_erosion(mask, morphology.disk(thickness))
    cortex = mask & ~interior
    c = float(coh[cortex].mean()) if cortex.any() else 0.0
    i = float(coh[interior].mean()) if interior.any() else 0.0
    return c - i


# ---------------------------------------------------------------------------
# STEP 1: Run tissue tension on all 4 conditions
# ---------------------------------------------------------------------------
def run_tension_all(force=False):
    print("\n[1/4] Tissue tension analysis (all 4 conditions)...")
    tension_csv = OUT / "tension_all_conditions.csv"
    if tension_csv.exists() and not force:
        print(f"  Cached: {tension_csv}")
        return pd.read_csv(str(tension_csv))

    rows = []
    for cond in CONDITIONS:
        data_dir = DATA_ROOT / cond
        if not data_dir.exists():
            continue
        for sample_path in sorted(data_dir.glob("*.tif")):
            pid = re.search(r"(G\d+)", sample_path.stem)
            pid = pid.group(1) if pid else sample_path.stem
            mask_path = EXISTING / cond / sample_path.stem / "masks" / f"{sample_path.stem}_masks.tif"
            if not mask_path.exists():
                print(f"    Missing mask: {cond}/{pid}, skipping")
                continue
            stack = tifffile.imread(str(sample_path))
            masks = tifffile.imread(str(mask_path)) > 0
            T = min(stack.shape[0], len(masks))
            for t in range(T):
                bf_mask = masks[t]
                if not bf_mask.any():
                    continue
                lyn = norm_pct(np.max(stack[t, :, 1], axis=0).astype(float))
                orient, coh = structure_tensor_fields(lyn)
                Q = nematic_order(orient, coh, bf_mask)
                cmi = cortex_minus_interior(coh, bf_mask)
                rows.append({
                    "condition": cond, "pescoid": pid, "time": t,
                    "hpf": HPF_START + t * HPF_INTERVAL,
                    "Q_nematic": Q,
                    "mean_coherence": float(coh[bf_mask].mean()),
                    "cortex_minus_interior": cmi,
                })
            print(f"    {cond}/{pid}: T={T}")
    df = pd.DataFrame(rows)
    df.to_csv(str(tension_csv), index=False)
    print(f"  Saved: {tension_csv}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Per-pescoid registration montage (all timepoints in one image)
# ---------------------------------------------------------------------------
def make_pescoid_folder(cond, pid, sample_stem):
    """Copy/symlink existing per-pescoid outputs into final folder."""
    src = EXISTING / cond / sample_stem
    dst = OUT / cond / pid
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for sub in ["masks", "plots", "overlays"]:
            sub_src = src / sub
            sub_dst = dst / sub
            if sub_src.exists() and not sub_dst.exists():
                try:
                    shutil.copytree(str(sub_src), str(sub_dst))
                except Exception:
                    pass
        for f in ["morphology_over_time.csv", "mezzo_expression_over_time.csv",
                  "pole_counts.csv", "pole_dimensions.csv", "summary.json",
                  "kymograph_matrix.npy"]:
            sf = src / f
            df = dst / f
            if sf.exists() and not df.exists():
                try:
                    shutil.copy2(str(sf), str(df))
                except Exception:
                    pass
    return dst


def make_registration_montage(cond, pid, sample_path, dst_folder):
    """Save montage of all timepoints' overlays (BF | mask | GFP)."""
    overlays_dir = dst_folder / "overlays"
    if not overlays_dir.exists():
        return None
    overlays = sorted(overlays_dir.glob("*.png"))
    if not overlays:
        return None

    n = len(overlays)
    n_cols = 6
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 1.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i, f in enumerate(overlays):
        try:
            img = mpimg.imread(str(f))
            axes[i].imshow(img)
        except Exception:
            pass
        axes[i].set_title(f"t={i}", fontsize=8)
        axes[i].axis("off")
    for i in range(n, n_rows * n_cols):
        axes[i].axis("off")

    plt.suptitle(f"{LABELS[cond]} / {pid} — all {n} timepoints (registered)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = dst_folder / f"{pid}_registration_montage.png"
    plt.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# STEP 3: Identify double-pole pescoids
# ---------------------------------------------------------------------------
def identify_double_pole():
    print("\n[3/4] Identifying double-pole pescoids...")
    double_rows = []
    for cond in CONDITIONS:
        cond_dir = EXISTING / cond
        if not cond_dir.exists():
            continue
        for sample_dir in sorted(cond_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pc_csv = sample_dir / "pole_counts.csv"
            if not pc_csv.exists():
                continue
            df = pd.read_csv(str(pc_csv))
            counts = dict(zip(df["metric"], df["value"]))
            n_out = int(counts.get("outward_poles", 0))
            n_total = int(counts.get("total_poles", 0))

            pid = re.search(r"(G\d+)", sample_dir.name)
            pid = pid.group(1) if pid else sample_dir.name

            double_rows.append({
                "condition": cond, "pescoid": pid, "sample_stem": sample_dir.name,
                "total_poles": n_total, "outward_poles": n_out,
            })

            # Save preview if 2+ outward poles
            if n_out >= 2:
                dest_dir = OUT / cond / pid
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Copy kymograph + pole detection plots
                kymo_src = sample_dir / "plots" / "kymograph.png"
                pole_src = sample_dir / "plots" / "pole_detection.png"
                montage_src = OUT / cond / pid / f"{pid}_registration_montage.png"

                # Two-panel preview: kymograph + pole detection
                fig, axes = plt.subplots(1, 2, figsize=(20, 6))
                if kymo_src.exists():
                    axes[0].imshow(mpimg.imread(str(kymo_src)))
                axes[0].set_title("Kymograph", fontsize=12, fontweight="bold")
                axes[0].axis("off")
                if pole_src.exists():
                    axes[1].imshow(mpimg.imread(str(pole_src)))
                axes[1].set_title(f"Poles detected: {n_total} ({n_out} outward)",
                                  fontsize=12, fontweight="bold", color="red")
                axes[1].axis("off")
                plt.suptitle(f"DOUBLE POLE: {LABELS[cond]} / {pid}", fontsize=14, fontweight="bold")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                out_path = DOUBLE / f"{cond}_{pid}_double_pole.png"
                plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
                plt.close(fig)
                print(f"  Double pole: {cond}/{pid} ({n_out} outward) -> {out_path.name}")

    pc_df = pd.DataFrame(double_rows)
    pc_df.to_csv(str(OUT / "pole_counts_all.csv"), index=False)
    return pc_df


# ---------------------------------------------------------------------------
# STEP 4: Comparison plots
# ---------------------------------------------------------------------------
def pchip_smooth(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, ys = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(xs) < 4:
        return xs, ys
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) < 4:
        return xs, ys
    interp = PchipInterpolator(xs, ys)
    xs_d = np.linspace(xs.min(), xs.max(), 100)
    return xs_d, interp(xs_d)


def bootstrap_ci(values, n_boot=500):
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([np.mean(RNG.choice(values, size=len(values), replace=True))
                     for _ in range(n_boot)])
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_time_ci(df, ycol):
    xs, ms, los, his = [], [], [], []
    for x, group in df.groupby("hpf"):
        m, lo, hi = bootstrap_ci(group[ycol].dropna().values)
        xs.append(x); ms.append(m); los.append(lo); his.append(hi)
    return np.array(xs), np.array(ms), np.array(los), np.array(his)


def load_per_pescoid_csvs():
    rows = []
    for cond in CONDITIONS:
        cond_dir = EXISTING / cond
        if not cond_dir.exists():
            continue
        for sample_dir in sorted(cond_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            pid = re.search(r"(G\d+)", sample_dir.name)
            pid = pid.group(1) if pid else sample_dir.name
            morph_p = sample_dir / "morphology_over_time.csv"
            mezzo_p = sample_dir / "mezzo_expression_over_time.csv"
            if not morph_p.exists():
                continue
            m = pd.read_csv(str(morph_p))
            z = pd.read_csv(str(mezzo_p)) if mezzo_p.exists() else None
            m["hpf"] = HPF_START + m["time"] * HPF_INTERVAL
            m["condition"] = cond
            m["pescoid"] = pid
            if z is not None:
                m = m.merge(z, on="time", how="left", suffixes=("", "_z"))
            rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_trajectory(df, ycol, ylabel, title, fname, conditions, out_dir, clip_zero=True):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for cond in conditions:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        xs, ms, los, his = per_time_ci(sub, ycol)
        if clip_zero:
            ms = np.clip(ms, 0, None)
            los = np.clip(los, 0, None)
            his = np.clip(his, 0, None)
        ax.fill_between(xs, los, his, color=COLORS[cond], alpha=0.15)
        xs_s, ms_s = pchip_smooth(xs, ms)
        if clip_zero:
            ms_s = np.clip(ms_s, 0, None)
        ax.plot(xs_s, ms_s, color=COLORS[cond], lw=2.5, label=LABELS[cond])
    ax.set_xlabel("Time [hpf]"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Condition", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out_dir / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir.name}/{fname}")


def plot_pole_counts(pc_df, conditions, out_dir, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    positions = list(range(len(conditions)))
    groups = []
    for i, cond in enumerate(conditions):
        vals = pc_df[pc_df["condition"] == cond]["outward_poles"].values
        groups.append(vals)
        x = i + RNG.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=60, color=COLORS[cond], edgecolors="black",
                   lw=0.6, alpha=0.85, zorder=3)
    bp = ax.boxplot(groups, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], conditions):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Number of outward poles per pescoid")
    ax.set_title("Pole count per pescoid", fontweight="bold")
    ax.set_ylim(bottom=-0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir.name}/{fname}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  FINAL Lyn_mezzo Comprehensive Analysis")
    print("=" * 60)

    # --- Step 1: tissue tension on all 4 conditions ---
    tension_df = run_tension_all()

    # --- Step 2: organize per-pescoid folders + registration montages ---
    print("\n[2/4] Organising per-pescoid folders + registration montages...")
    for cond in CONDITIONS:
        data_dir = DATA_ROOT / cond
        if not data_dir.exists():
            continue
        for sample_path in sorted(data_dir.glob("*.tif")):
            pid = re.search(r"(G\d+)", sample_path.stem)
            pid = pid.group(1) if pid else sample_path.stem
            dst = make_pescoid_folder(cond, pid, sample_path.stem)
            mont = make_registration_montage(cond, pid, sample_path, dst)
            if mont:
                pass  # print(f"  {cond}/{pid}: montage saved")

    # --- Step 3: double-pole detection ---
    pc_df = identify_double_pole()

    # --- Step 4: comparison plots ---
    print("\n[4/4] Generating comparison plots...")
    big = load_per_pescoid_csvs()
    # merge tension (per-pescoid average over time):
    tension_per_pesc = tension_df.copy()

    # ---- Trajectory plots: ALL 4 CONDITIONS ----
    print("\n  -- ALL 4 conditions --")
    plot_trajectory(big, "aspect_ratio", "Aspect Ratio",
                    "Aspect Ratio over Time", "01_aspect_ratio.png",
                    CONDITIONS, COMP_ALL)
    plot_trajectory(big, "gfp_fraction", "GFP+ fraction",
                    "Mezzo+ fraction over Time", "02_gfp_fraction.png",
                    CONDITIONS, COMP_ALL)
    plot_trajectory(big, "gfp_mean_intensity", "Mean GFP intensity",
                    "Mean GFP intensity over Time", "03_gfp_mean_intensity.png",
                    CONDITIONS, COMP_ALL)
    plot_trajectory(tension_df, "Q_nematic", "Q (nematic order)",
                    "Tissue alignment (LynTom) over Time", "04_tension_Q_nematic.png",
                    CONDITIONS, COMP_ALL)
    plot_trajectory(tension_df, "cortex_minus_interior",
                    "Cortex − Interior coherence",
                    "Cortical tension excess over Time", "05_tension_cortex_excess.png",
                    CONDITIONS, COMP_ALL, clip_zero=False)
    plot_pole_counts(pc_df, CONDITIONS, COMP_ALL, "06_pole_counts.png")

    # ---- Trajectory plots: P only ----
    print("\n  -- P_ctrl vs P_Activin --")
    plot_trajectory(big, "aspect_ratio", "Aspect Ratio",
                    "Aspect Ratio (P_ctrl vs P_Activin)", "01_aspect_ratio.png",
                    P_CONDITIONS, COMP_P)
    plot_trajectory(big, "gfp_fraction", "GFP+ fraction",
                    "Mezzo+ fraction (P_ctrl vs P_Activin)", "02_gfp_fraction.png",
                    P_CONDITIONS, COMP_P)
    plot_trajectory(big, "gfp_mean_intensity", "Mean GFP intensity",
                    "Mean GFP intensity (P_ctrl vs P_Activin)", "03_gfp_mean_intensity.png",
                    P_CONDITIONS, COMP_P)
    plot_trajectory(tension_df, "Q_nematic", "Q (nematic order)",
                    "Tissue alignment (P_ctrl vs P_Activin)", "04_tension_Q_nematic.png",
                    P_CONDITIONS, COMP_P)
    plot_trajectory(tension_df, "cortex_minus_interior",
                    "Cortex − Interior coherence",
                    "Cortical tension excess (P_ctrl vs P_Activin)", "05_tension_cortex_excess.png",
                    P_CONDITIONS, COMP_P, clip_zero=False)
    plot_pole_counts(pc_df, P_CONDITIONS, COMP_P, "06_pole_counts.png")

    print(f"\nAll outputs in: {OUT}/")
    print(f"  Per-pescoid folders (with overlays + montages): {OUT}/<condition>/<pid>/")
    print(f"  Double-pole previews: {DOUBLE}/")
    print(f"  All 4 conditions comparison plots: {COMP_ALL}/")
    print(f"  P-only comparison plots: {COMP_P}/")


if __name__ == "__main__":
    main()
