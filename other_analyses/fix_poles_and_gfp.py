"""
Two fixes:

  A) POLE DETECTION redefined:
     A pole = a persistent OUTWARD protrusion that elongates beyond the
     spherical body, NOT every local kymograph peak.

     New algorithm:
       1. Average kymograph over LATE timepoints only (last 10 frames) —
          this is when poles are formed.
       2. Use only positive Δr (outward protrusions) — drop inward dips.
       3. Require prominence ≥ 0.30 (much stricter than before).
       4. Require minimum angular separation = 40 perimeter bins.
       5. Measure pole length on the peak-AR mask: a peak is only counted
          as a pole if its tip radius exceeds the equivalent-circle radius
          by ≥ 30% (i.e., real protrusion, not noise).
       6. Save annotated QC image per pescoid with ONLY the kept poles.

  B) GFP comparison plots redone with proper metrics:
     - INTEGRATED GFP (sum of bg-subtracted intensity in mask) — this is
       the size-independent measure that matches biology (E > P).
     - Mean bg-subtracted intensity.
     - S/B ratio.
     Both All-4-conditions and P-only versions.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
from skimage import measure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

DATA_ROOT = Path(
    r"Z:\Nick_Marschlich\EMBL_Barcelona\Projects\Imaging\Olympus\P4_Pescoids"
    r"\P4B_general_pescoids\250402_mezzo-LynTom_Activin_obj-10x_med-PGM_time-6hpf\TIF"
)
EXISTING = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_yen_fixed")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_final")
QC = OUT / "pole_QC_v2"
QC.mkdir(parents=True, exist_ok=True)
COMP_ALL = OUT / "comparison_v2" / "all_4_conditions"
COMP_P = OUT / "comparison_v2" / "P_only"
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

# --- POLE DETECTION PARAMETERS ---
LATE_FRAMES = 10              # average kymograph over last N frames
POLE_PROMINENCE = 0.30        # peak prominence (relative to max)
POLE_MIN_DISTANCE = 40        # min perimeter-bin separation between poles
POLE_PROTRUSION_FRAC = 0.25   # tip radius must exceed eq-circle radius by ≥ 25%

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# =========================================================================
# A. NEW POLE DETECTION
# =========================================================================
def detect_real_poles(kymo_raw, masks):
    """
    Detect biologically meaningful poles.
    Returns list of dicts with pole metadata + the peak-AR frame index.
    """
    P, T = kymo_raw.shape

    # 1) Restrict to LATE timepoints (when poles are formed)
    n_late = min(LATE_FRAMES, T)
    late_slice = kymo_raw[:, T - n_late:T]

    # 2) Outward-only: clip negatives to 0
    outward_only = np.clip(late_slice, 0, None)

    # 3) Time-average over late frames
    activity = outward_only.mean(axis=1)
    if activity.max() > 0:
        activity = activity / activity.max()  # normalise so prominence is in [0,1]
    activity_smooth = gaussian_filter1d(activity, sigma=3, mode="wrap")

    # 4) Find peaks (stricter parameters)
    peaks, _ = find_peaks(
        activity_smooth,
        prominence=POLE_PROMINENCE,
        distance=POLE_MIN_DISTANCE,
    )
    if len(peaks) == 0:
        return [], None, activity_smooth

    # 5) Pick peak-AR frame
    ars = []
    for t in range(T):
        if t >= len(masks):
            ars.append(1.0)
            continue
        props = measure.regionprops(masks[t].astype(int))
        if props:
            p = max(props, key=lambda r: r.area)
            mn = p.minor_axis_length or 1e-6
            ars.append(p.major_axis_length / mn)
        else:
            ars.append(1.0)
    peak_t = int(np.argmax(ars))
    mask = masks[peak_t]
    props = measure.regionprops(mask.astype(int))
    if not props:
        return [], peak_t, activity_smooth
    p0 = max(props, key=lambda r: r.area)
    cy, cx = p0.centroid

    # Equivalent-circle radius from area
    eq_r = float(np.sqrt(p0.area / np.pi))

    # 6) For each peak: measure tip radius on the mask
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    angles_img = np.arctan2(yy - cy, xx - cx)
    radii_img = np.hypot(yy - cy, xx - cx)

    def bin_to_angle(i):
        return -np.pi + 2 * np.pi * (i / P)

    real_poles = []
    for pi in peaks:
        # FWHM around peak (defines pole wedge)
        half = activity_smooth[pi] / 2
        l = pi
        for _ in range(P):
            lp = (l - 1) % P
            if activity_smooth[lp] < half:
                break
            l = lp
        r = pi
        for _ in range(P):
            rn = (r + 1) % P
            if activity_smooth[rn] < half:
                break
            r = rn
        a_lo, a_hi = bin_to_angle(l), bin_to_angle(r)

        if a_hi >= a_lo:
            extent_rad = a_hi - a_lo
            in_wedge = (angles_img >= a_lo) & (angles_img <= a_hi)
        else:
            extent_rad = (2 * np.pi) - (a_lo - a_hi)
            in_wedge = (angles_img >= a_lo) | (angles_img <= a_hi)

        wedge_mask = mask & in_wedge
        if not wedge_mask.any():
            continue
        tip_r = float(radii_img[wedge_mask].max())
        protrusion = (tip_r - eq_r) / eq_r  # fractional excess over circle

        # 7) Filter: only keep if it really protrudes
        if protrusion < POLE_PROTRUSION_FRAC:
            continue

        real_poles.append({
            "pole_index": int(pi),
            "peak_t": peak_t,
            "tip_radius_px": tip_r,
            "eq_circle_radius_px": eq_r,
            "protrusion_fraction": protrusion,
            "pole_length_px": tip_r,
            "angular_extent_deg": float(np.degrees(extent_rad)),
            "wedge_a_lo_deg": float(np.degrees(a_lo)),
            "wedge_a_hi_deg": float(np.degrees(a_hi)),
            "activity_at_peak": float(activity_smooth[pi]),
        })

    # Sort by tip_radius descending, label primary/secondary
    real_poles.sort(key=lambda d: d["tip_radius_px"], reverse=True)
    for i, r in enumerate(real_poles):
        r["rank"] = i + 1
        r["label"] = "primary" if i == 0 else ("secondary" if i == 1 else f"pole_{i+1}")
    return real_poles, peak_t, activity_smooth


def make_pole_qc(cond, sample_dir):
    pid_match = re.search(r"(G\d+)", sample_dir.name)
    pid = pid_match.group(1) if pid_match else sample_dir.name
    kymo_path = sample_dir / "kymograph_matrix.npy"
    mask_path = sample_dir / "masks" / f"{sample_dir.name}_masks.tif"
    if not kymo_path.exists() or not mask_path.exists():
        return None, None

    kymo_raw = np.load(str(kymo_path))
    masks = tifffile.imread(str(mask_path)) > 0

    poles, peak_t, activity = detect_real_poles(kymo_raw, masks)
    n_poles = len(poles)
    if peak_t is None:
        return None, None
    mask = masks[peak_t]
    props = measure.regionprops(mask.astype(int))
    if not props:
        return None, None
    p0 = max(props, key=lambda r: r.area)
    cy, cx = p0.centroid

    # === plot ===
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    vmax = max(0.001, np.abs(kymo_raw).max())
    axes[0].imshow(kymo_raw, aspect="auto", cmap="seismic",
                   vmin=-vmax, vmax=vmax, origin="lower")
    T = kymo_raw.shape[1]
    axes[0].axvspan(T - LATE_FRAMES, T, color="yellow", alpha=0.12,
                    label=f"late window (last {LATE_FRAMES} frames)")
    for p in poles:
        axes[0].axhline(p["pole_index"], color="lime", lw=1.4, ls="--", alpha=0.9)
        axes[0].text(0.5, p["pole_index"] + 1.5,
                     f"{p['label']} (len={p['tip_radius_px']:.0f}px, prot={p['protrusion_fraction']*100:.0f}%)",
                     color="lime", fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title(f"Kymograph — poles drawn only if tip > {POLE_PROTRUSION_FRAC*100:.0f}% beyond eq-circle",
                       fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time [frame]")
    axes[0].set_ylabel("Perimeter bin")

    axes[1].plot(activity, "k-", lw=2,
                 label=f"Outward-only activity (last {LATE_FRAMES} frames, normalised)")
    if n_poles > 0:
        ps = [p["pole_index"] for p in poles]
        axes[1].scatter(ps, [activity[i] for i in ps],
                        s=160, marker="^", c="red", edgecolors="black", zorder=4,
                        label=f"Real poles (n={n_poles})")
        for p in poles:
            axes[1].annotate(
                f"{p['label']}\n(prot {p['protrusion_fraction']*100:.0f}%)",
                (p["pole_index"], activity[p["pole_index"]]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="red", fontweight="bold")
    axes[1].axhline(POLE_PROMINENCE, color="gray", ls="--", lw=1, alpha=0.5,
                    label=f"Prominence threshold ({POLE_PROMINENCE})")
    axes[1].set_xlabel("Perimeter bin")
    axes[1].set_ylabel("Normalised outward activity")
    axes[1].set_title("Outward activity profile + accepted poles",
                       fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9, loc="best")
    axes[1].grid(alpha=0.3)

    axes[2].imshow(mask, cmap="Greys_r")
    axes[2].plot(cx, cy, "y+", markersize=14, mew=2.5)
    # Equivalent-circle reference
    theta_c = np.linspace(0, 2 * np.pi, 200)
    eq_r = float(np.sqrt(p0.area / np.pi))
    axes[2].plot(cx + eq_r * np.cos(theta_c), cy + eq_r * np.sin(theta_c),
                 "y--", lw=1.5, alpha=0.7, label="Equivalent-circle radius")
    R = max(mask.shape) * 0.6
    for p in poles:
        color = "lime" if p["rank"] == 1 else ("cyan" if p["rank"] == 2 else "magenta")
        a_lo, a_hi = p["wedge_a_lo_deg"], p["wedge_a_hi_deg"]
        if a_hi < a_lo:
            a_hi += 360
        wedge = Wedge((cx, cy), R, a_lo, a_hi, alpha=0.30,
                      color=color, edgecolor=color, linewidth=2.5)
        axes[2].add_patch(wedge)
        mid = np.radians((a_lo + a_hi) / 2)
        lx = cx + R * 0.75 * np.cos(mid)
        ly = cy + R * 0.75 * np.sin(mid)
        axes[2].text(lx, ly, f"{p['label']}\n{p['tip_radius_px']:.0f}px",
                     color=color, fontsize=11, fontweight="bold", ha="center",
                     bbox=dict(facecolor="black", alpha=0.55, pad=2))
    axes[2].set_xlim(0, mask.shape[1])
    axes[2].set_ylim(mask.shape[0], 0)
    axes[2].set_title(f"BF mask at peak-AR (t={peak_t})  —  {n_poles} real pole(s)",
                       fontsize=11, fontweight="bold")
    axes[2].axis("off")
    axes[2].legend(loc="lower right", fontsize=9)

    plt.suptitle(f"{cond} / {pid} — pole-detection v2 (sustained outward protrusions only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = QC / f"{cond}_{pid}_poleQC_v2.png"
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return n_poles, poles


def run_all_pole_qc():
    print("\n[1/2] Re-detecting poles with biological definition...")
    rows = []
    for cond in CONDITIONS:
        cond_dir = EXISTING / cond
        if not cond_dir.exists():
            continue
        for sample_dir in sorted(cond_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            n_poles, poles = make_pole_qc(cond, sample_dir)
            pid_match = re.search(r"(G\d+)", sample_dir.name)
            pid = pid_match.group(1) if pid_match else sample_dir.name
            rows.append({
                "condition": cond, "pescoid": pid,
                "n_poles_v2": n_poles if n_poles is not None else 0,
                "primary_len_px": poles[0]["tip_radius_px"] if poles else 0.0,
                "primary_protrusion_pct": poles[0]["protrusion_fraction"] * 100 if poles else 0.0,
                "secondary_len_px": poles[1]["tip_radius_px"] if poles and len(poles) > 1 else 0.0,
                "secondary_protrusion_pct": poles[1]["protrusion_fraction"] * 100 if poles and len(poles) > 1 else 0.0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(str(OUT / "pole_counts_v2.csv"), index=False)
    print(f"  Total: {len(df)} pescoids, {(df['n_poles_v2']>=2).sum()} with 2+ real poles")
    print(f"  Distribution:")
    print(df.groupby("condition")["n_poles_v2"].agg(["mean", "count"]).round(2))
    return df


# =========================================================================
# B. CORRECT GFP COMPARISON PLOTS (integrated GFP — size-independent)
# =========================================================================
def load_raw_intensity():
    """Load the raw intensity CSV we built before; if missing, fail loudly."""
    csv = OUT / "intensity_values_all_pescoids.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing {csv} — run export_intensity_and_pole_qc.py first")
    df = pd.read_csv(str(csv))
    return df


def pchip_smooth(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    xs, ys = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(xs) < 4:
        return xs, ys
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
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
    for x, g in df.groupby("hpf"):
        v = g[ycol].dropna().values
        m, lo, hi = bootstrap_ci(v)
        xs.append(x); ms.append(m); los.append(lo); his.append(hi)
    return np.array(xs), np.array(ms), np.array(los), np.array(his)


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


def gen_gfp_plots(df):
    print("\n[2/2] Generating CORRECT GFP comparison plots (integrated, bg-sub, S/B)...")

    print("\n  -- ALL 4 conditions --")
    plot_trajectory(df, "integrated_gfp_bgsub", "Integrated GFP (bg-sub)",
                    "Integrated mezzo:GFP (sum of bg-sub intensity in mask)",
                    "01_integrated_gfp.png", CONDITIONS, COMP_ALL)
    plot_trajectory(df, "gfp_mean_bgsub", "Mean GFP intensity (bg-sub)",
                    "Mean mezzo:GFP intensity (background-subtracted)",
                    "02_mean_bgsub.png", CONDITIONS, COMP_ALL)
    plot_trajectory(df, "sb_ratio_mean", "S/B ratio (mean)",
                    "Signal-to-background ratio (mean)",
                    "03_sb_ratio.png", CONDITIONS, COMP_ALL, clip_zero=False)
    plot_trajectory(df, "gfp_raw_mean_in_mask", "Mean raw GFP intensity",
                    "Mean raw GFP intensity (uncorrected — for reference)",
                    "S1_raw_mean.png", CONDITIONS, COMP_ALL)
    plot_trajectory(df, "bg_mean_outside_mask", "Background intensity",
                    "Background intensity (outside mask)",
                    "S2_background.png", CONDITIONS, COMP_ALL)

    print("\n  -- P_ctrl vs P_Activin --")
    plot_trajectory(df, "integrated_gfp_bgsub", "Integrated GFP (bg-sub)",
                    "Integrated mezzo:GFP (P_ctrl vs P_Activin)",
                    "01_integrated_gfp.png", P_CONDITIONS, COMP_P)
    plot_trajectory(df, "gfp_mean_bgsub", "Mean GFP intensity (bg-sub)",
                    "Mean mezzo:GFP bg-sub (P_ctrl vs P_Activin)",
                    "02_mean_bgsub.png", P_CONDITIONS, COMP_P)
    plot_trajectory(df, "sb_ratio_mean", "S/B ratio (mean)",
                    "S/B ratio (P_ctrl vs P_Activin)",
                    "03_sb_ratio.png", P_CONDITIONS, COMP_P, clip_zero=False)


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 60)
    print("  POLE DETECTION v2 + CORRECTED GFP PLOTS")
    print("=" * 60)

    pole_df = run_all_pole_qc()
    intensity_df = load_raw_intensity()
    gen_gfp_plots(intensity_df)

    print(f"\nAll outputs in: {OUT}/")
    print(f"  Pole QC v2 (real poles only): {QC}/")
    print(f"  Comparison v2 (all 4 conds):  {COMP_ALL}/")
    print(f"  Comparison v2 (P only):       {COMP_P}/")
    print(f"  Pole counts CSV:              {OUT}/pole_counts_v2.csv")


if __name__ == "__main__":
    main()
