"""
Phase 5 (Lyn_mezzo) - AP-axis mezzo kymographs + population AP profile.

What Vikas asked for:
  - Per pescoid: kymograph of mezzo intensity projected onto the AP axis
    (defined by the ellipse major axis), with time on y, normalised AP on x.
  - Population-level: mean mezzo profile along normalised AP, stratified by
    phenotype, at peak elongation (peak-AR aligned).
  - Three normalisations baked in:
      spatial  = AP position / major axis length            (per frame)
      temporal = each pescoid's time axis aligned to its
                 peak-aspect-ratio frame (t_norm = 0 at peak)
      intensity = bg-subtracted GFP, then per-pescoid max-normalised

Outputs (Lyn_mezzo_phase5_kymograph/):
  per_pescoid/<cond>/<pid>/<pid>_ap_kymograph.png
                          <pid>_ap_profiles.npy           (T, n_bins)
                          <pid>_meta.json
  plots/01_kymographs_examples.png    5-pescoid panel for the poster
  plots/02_ap_profile_by_phenotype.png  population mean +/- 95% CI at peak
  plots/03_ap_pole_position_histogram.png pole AP positions by phenotype
  tables/ap_profiles_summary.csv

Usage:
    python phase5_kymograph_ap.py --pids G046              # sanity on one
    python phase5_kymograph_ap.py --examples               # the 5 poster set
    python phase5_kymograph_ap.py --all                    # every phenotyped pescoid
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_kymograph")
PER = OUT / "per_pescoid"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PER, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}

HPF_START = 6.0
HPF_INTERVAL = 0.4    # original Lyn_mezzo timing

AP_BINS = 100         # spatial resolution of the AP profile (bins along [0,1])
GLOBAL_THR_BGSUB = 188.17   # established Phase 2 threshold for "mezzo+" pixels
PEAK_ELONGATION_HPF = 12.0  # universal reference (highest pescoid elongation)
SIGNAL_MODE = "fraction"    # "fraction" = mezzo+ pixel fraction per AP slice (sharper peaks),
                            # "intensity" = mean bg-sub GFP per AP slice (smoother)

# 5 representative pescoids for the poster kymograph panel
EXAMPLES = [
    ("P_ctrl",           "G022", "coord_mono"),
    ("P_Activin_3-5hpf", "G046", "coord_mono"),
    ("P_Activin_3-5hpf", "G047", "coord_bipolar"),
    ("P_Activin_3-5hpf", "G050", "mezzo_bipolar"),
    ("P_Activin_3-5hpf", "G060", "multipolar"),
]

PHENO_ORDER = ["no_induction", "diffuse_mezzo", "coordinated_monopolar",
               "mezzo_bipolar_only", "coordinated_bipolar",
               "disorganised_multipolar", "multipolar", "oc_only"]
PHENO_COLOR = {
    "no_induction": "#999999", "diffuse_mezzo": "#66c2a5",
    "coordinated_monopolar": "#386cb0", "mezzo_bipolar_only": "#7570b3",
    "coordinated_bipolar": "#1b9e77", "disorganised_multipolar": "#d95f02",
    "multipolar": "#e7298a", "oc_only": "#e6ab02",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "font.size": 11,
})


# ---------------------------------------------------------------------------
# Per-frame ellipse fit  ->  AP axis  ->  projection
# ---------------------------------------------------------------------------
def per_frame_geometry(mask_frame):
    """Ellipse fit (skimage regionprops) + actual mask extent along the major
    axis. orientation is the angle of the major axis to image y."""
    if not mask_frame.any():
        return None
    props = measure.regionprops(mask_frame.astype(int))
    if not props:
        return None
    p = max(props, key=lambda r: r.area)
    minor = float(p.minor_axis_length) or 1e-6
    # mask extent along the major axis (anterior/posterior tips)
    yy, xx = np.where(mask_frame)
    cos_o, sin_o = float(np.cos(p.orientation)), float(np.sin(p.orientation))
    s_pix = (yy - float(p.centroid[0])) * cos_o - (xx - float(p.centroid[1])) * sin_o
    return {
        "cy": float(p.centroid[0]), "cx": float(p.centroid[1]),
        "orientation": float(p.orientation),
        "major": float(p.major_axis_length),
        "minor": float(minor),
        "ar": float(p.major_axis_length / minor),
        "s_min": float(s_pix.min()), "s_max": float(s_pix.max()),
    }


def project_to_ap(gfp_frame, mask_frame, geom, n_bins=AP_BINS):
    """Project the bg-subtracted GFP signal onto the major-axis (AP) coordinate.
    Returns a 1-D vector of length n_bins with mean signal per AP bin."""
    if geom is None or not mask_frame.any():
        return np.full(n_bins, np.nan)
    yy, xx = np.mgrid[: mask_frame.shape[0], : mask_frame.shape[1]]
    # AP coordinate along the major axis (sign convention from regionprops):
    # s = (y - cy) * cos(theta) - (x - cx) * sin(theta)
    cos_o, sin_o = np.cos(geom["orientation"]), np.sin(geom["orientation"])
    s = (yy - geom["cy"]) * cos_o - (xx - geom["cx"]) * sin_o
    # Normalise to [0, 1] using the ACTUAL extent of the mask along the AP axis
    # (most-anterior pixel = 0, most-posterior = 1). This guarantees the mask
    # spans the full [0, 1] range and avoids white bins for asymmetric pescoids.
    s_in_mask = s[mask_frame]
    if s_in_mask.size == 0:
        return np.full(n_bins, np.nan)
    s_min, s_max = float(s_in_mask.min()), float(s_in_mask.max())
    span = max(s_max - s_min, 1.0)
    s_norm = (s - s_min) / span
    # background-subtract within this frame (bg = mean GFP OUTSIDE mask)
    bg = float(gfp_frame[~mask_frame].mean()) if (~mask_frame).any() else 0.0
    sig = (gfp_frame - bg).clip(min=0)
    # accumulate per AP bin (only count mask pixels)
    inside = mask_frame & (s_norm >= 0) & (s_norm <= 1)
    if not inside.any():
        return np.full(n_bins, np.nan)
    bin_idx = np.clip((s_norm[inside] * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    if SIGNAL_MODE == "fraction":
        # mezzo+ pixel fraction per AP bin = fraction of mask pixels in that
        # AP slice that exceed the global threshold. Gives sharp peaks where
        # real poles concentrate, instead of a smeared average.
        pos = (sig > GLOBAL_THR_BGSUB) & inside
        pos_idx = np.clip((s_norm[pos] * n_bins).astype(int), 0, n_bins - 1)
        pos_counts = np.bincount(pos_idx, minlength=n_bins)
        profile = np.full(n_bins, np.nan)
        nonzero = counts > 0
        profile[nonzero] = pos_counts[nonzero] / counts[nonzero]
    else:
        sig_vals = sig[inside]
        sums = np.bincount(bin_idx, weights=sig_vals, minlength=n_bins)
        profile = np.full(n_bins, np.nan)
        nonzero = counts > 0
        profile[nonzero] = sums[nonzero] / counts[nonzero]
    return profile


# ---------------------------------------------------------------------------
# AP coordinate of a pole centroid (using the same per-frame geometry)
# ---------------------------------------------------------------------------
def pole_ap_position(geom, py, px):
    cos_o, sin_o = np.cos(geom["orientation"]), np.sin(geom["orientation"])
    s = (py - geom["cy"]) * cos_o - (px - geom["cx"]) * sin_o
    span = max(geom["s_max"] - geom["s_min"], 1.0)
    return float((s - geom["s_min"]) / span)


# ---------------------------------------------------------------------------
# Per pescoid: build kymograph + meta
# ---------------------------------------------------------------------------
def process_pescoid(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = gfp.shape[0]

    geoms = [per_frame_geometry(mask[t]) for t in range(T)]
    # Per-frame AR series and peak-AR time (temporal anchor)
    ar = np.array([g["ar"] if g else np.nan for g in geoms])
    peak_ar_t = int(np.nanargmax(ar)) if np.isfinite(np.nanmax(ar)) else T // 2
    # Per-frame AP profile
    profiles = np.array([project_to_ap(gfp[t], mask[t], geoms[t]) for t in range(T)])

    # AP-orientation convention: flip the AP axis so the END WITH THE HIGHEST
    # MEZZO SIGNAL at 12 hpf (peak elongation) sits at AP = 1. Makes monopolar
    # profiles align (peak always on the right), so the population mean is a
    # clean single peak instead of a centre-blurred curve.
    t12 = int(round((PEAK_ELONGATION_HPF - HPF_START) / HPF_INTERVAL))
    t12 = max(0, min(T - 1, t12))
    profile_12 = profiles[t12]
    flipped = False
    if np.isfinite(profile_12).any():
        # half with the larger total signal -> posterior (AP=1)
        half = AP_BINS // 2
        left_sum = np.nansum(profile_12[:half])
        right_sum = np.nansum(profile_12[half:])
        if left_sum > right_sum:
            profiles = profiles[:, ::-1]
            flipped = True

    # For fraction mode the values are already in [0,1] (no per-pescoid max-norm).
    # For intensity mode keep max-norm so pescoids are comparable.
    if SIGNAL_MODE == "fraction":
        profiles_norm = profiles.copy()
    else:
        pmax = np.nanmax(profiles) if np.isfinite(np.nanmax(profiles)) else 1.0
        profiles_norm = profiles / max(pmax, 1.0)

    # Pole AP positions (mezzo_tracks from Phase 2 — pole-type only)
    pole_aps = []
    p2dir = PHASE2 / "per_pescoid" / cond / pid
    tcsv = p2dir / "mezzo_tracks.csv"
    if tcsv.exists():
        try:
            tdf = pd.read_csv(str(tcsv))
        except pd.errors.EmptyDataError:
            tdf = pd.DataFrame()
        if not tdf.empty and "cluster_type" in tdf.columns:
            poles = tdf[tdf["cluster_type"] == "pole"]
            for _, row in poles.iterrows():
                pf = int(row["peak_frame"])
                if 0 <= pf < T and geoms[pf] is not None:
                    ap = pole_ap_position(geoms[pf], row["peak_centroid_y"],
                                          row["peak_centroid_x"])
                    if flipped:
                        ap = 1.0 - ap
                    pole_aps.append({"track_id": int(row["track_id"]),
                                     "peak_frame": pf,
                                     "peak_hpf": HPF_START + pf * HPF_INTERVAL,
                                     "ap_position": ap,
                                     "area_px": float(row["peak_area_px"])})
    meta = {
        "condition": cond, "pescoid": pid, "T": T,
        "peak_ar_t": peak_ar_t,
        "peak_ar_hpf": HPF_START + peak_ar_t * HPF_INTERVAL,
        "peak_elongation_hpf": PEAK_ELONGATION_HPF,
        "ap_flipped": bool(flipped),
        "signal_mode": SIGNAL_MODE,
        "pole_ap_positions": pole_aps,
    }
    # Save per-pescoid
    pdir_out = PER / cond / pid
    pdir_out.mkdir(parents=True, exist_ok=True)
    np.save(str(pdir_out / f"{pid}_ap_profiles.npy"), profiles_norm)
    with open(str(pdir_out / f"{pid}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    save_single_kymograph(cond, pid, profiles_norm, meta,
                           pdir_out / f"{pid}_ap_kymograph.png")
    return profiles_norm, meta


def save_single_kymograph(cond, pid, profiles_norm, meta, out_path):
    """Two-panel: kymograph on the left, 1D AP profile at peak elongation
    (12 hpf) on the right. Profile shape: monopolar -> one peak; bipolar ->
    two peaks; multipolar -> several peaks."""
    T = profiles_norm.shape[0]
    hpf = HPF_START + np.arange(T) * HPF_INTERVAL
    cbar_label = ("mezzo+ pixel fraction"
                  if meta["signal_mode"] == "fraction"
                  else "bg-sub GFP / pescoid-max")
    vmax = max(np.nanpercentile(profiles_norm, 98), 0.05)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 0.05, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    axp = fig.add_subplot(gs[2])

    im = ax.imshow(profiles_norm, aspect="auto", origin="lower",
                   extent=[0, 1, hpf[0], hpf[-1] + HPF_INTERVAL],
                   cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax.axhline(PEAK_ELONGATION_HPF, color="cyan", lw=1.6,
               label=f"peak elongation = {PEAK_ELONGATION_HPF:.0f} hpf")
    ax.axhline(meta["peak_ar_hpf"], ls="--", color="white", lw=1.0, alpha=0.6,
               label=f"this pescoid's peak-AR @ {meta['peak_ar_hpf']:.1f} hpf")
    for p in meta["pole_ap_positions"]:
        ax.plot(p["ap_position"], p["peak_hpf"], "o", color="lime",
                markersize=6 + 6 * min(p["area_px"] / 6000, 1.0),
                markeredgecolor="black", markeredgewidth=0.7)
    ax.set_xlabel("AP position (normalised; AP=1 = peak-mezzo end)")
    ax.set_ylabel("hpf")
    flip_tag = "  [AP flipped]" if meta["ap_flipped"] else ""
    ax.set_title(f"{cond} / {pid}{flip_tag}\nmezzo:GFP along AP axis ({meta['signal_mode']})",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    plt.colorbar(im, cax=cax, label=cbar_label)

    # 1-D profile at 12 hpf (the user's "where is mezzo density highest along AP")
    t12 = int(round((PEAK_ELONGATION_HPF - HPF_START) / HPF_INTERVAL))
    t12 = max(0, min(T - 1, t12))
    x = np.linspace(0, 1, profiles_norm.shape[1])
    axp.fill_between(x, 0, np.nan_to_num(profiles_norm[t12]),
                     color="#386cb0", alpha=0.6)
    axp.plot(x, profiles_norm[t12], color="#1f1f1f", lw=1.5)
    for p in meta["pole_ap_positions"]:
        if abs(p["peak_hpf"] - PEAK_ELONGATION_HPF) <= 2.0:
            axp.axvline(p["ap_position"], ls=":", color="lime", lw=1.2)
    axp.set_xlim(0, 1)
    axp.set_xlabel("AP position")
    axp.set_ylabel(cbar_label)
    axp.set_title(f"AP profile @ {PEAK_ELONGATION_HPF:.0f} hpf",
                  fontsize=11, fontweight="bold")
    axp.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5-pescoid poster panel
# ---------------------------------------------------------------------------
def example_panel(results):
    """5-pescoid kymograph row + 1-D profile-at-12-hpf row below."""
    n = len(EXAMPLES)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 11),
                              gridspec_kw={"height_ratios": [3, 1.2]})
    # Common colour scale (98th percentile across the 5 pescoids)
    vmax = max(np.nanpercentile(np.stack([r[0] for r in
                                            [results[(c, p)] for c, p, _ in EXAMPLES]]),
                                  98), 0.05)
    cbar_label = ("mezzo+ pixel fraction" if SIGNAL_MODE == "fraction"
                  else "bg-sub GFP / pescoid-max")
    for ci, (cond, pid, label) in enumerate(EXAMPLES):
        profiles, meta = results[(cond, pid)]
        T = profiles.shape[0]
        hpf = HPF_START + np.arange(T) * HPF_INTERVAL
        ax = axes[0, ci]
        im = ax.imshow(profiles, aspect="auto", origin="lower",
                       extent=[0, 1, hpf[0], hpf[-1] + HPF_INTERVAL],
                       cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.axhline(PEAK_ELONGATION_HPF, color="cyan", lw=1.5)
        for p in meta["pole_ap_positions"]:
            ax.plot(p["ap_position"], p["peak_hpf"], "o", color="lime",
                    markersize=5 + 5 * min(p["area_px"] / 6000, 1.0),
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.set_xlabel("AP position")
        flip = " [flipped]" if meta["ap_flipped"] else ""
        ax.set_title(f"{COND_LABEL[cond]}\n{pid} ({label}){flip}",
                     fontsize=10, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("hpf")
        # 1-D profile at 12 hpf
        axp = axes[1, ci]
        t12 = int(round((PEAK_ELONGATION_HPF - HPF_START) / HPF_INTERVAL))
        t12 = max(0, min(T - 1, t12))
        x = np.linspace(0, 1, profiles.shape[1])
        prof = np.nan_to_num(profiles[t12])
        axp.fill_between(x, 0, prof, color="#386cb0", alpha=0.7)
        axp.plot(x, prof, color="#1f1f1f", lw=1.4)
        for p in meta["pole_ap_positions"]:
            if abs(p["peak_hpf"] - PEAK_ELONGATION_HPF) <= 2.0:
                axp.axvline(p["ap_position"], ls=":", color="lime", lw=1.2)
        axp.set_xlim(0, 1)
        axp.set_ylim(0, vmax)
        axp.set_xlabel("AP position")
        if ci == 0:
            axp.set_ylabel(f"{cbar_label}\n@ 12 hpf")
        axp.grid(True, alpha=0.3)
    fig.colorbar(im, ax=axes[0, -1], label=cbar_label, shrink=0.85)
    plt.suptitle("AP-mezzo kymographs (top) + 1-D AP profiles at peak elongation, "
                 "12 hpf (bottom)\nAP=1 = end with peak mezzo signal; "
                 "cyan line = 12 hpf; lime markers = mezzo+ poles",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(PLOTS / "01_kymographs_examples.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  01_kymographs_examples.png")


# ---------------------------------------------------------------------------
# Population AP profile at peak-AR, by phenotype
# ---------------------------------------------------------------------------
def population_ap_profile(all_results, summary):
    """Mean AP profile (each pescoid's value AT ITS peak-AR frame) by phenotype.
    Returns one curve per phenotype with bootstrap 95% CI."""
    by_pheno = {}
    for (cond, pid), (profiles, meta) in all_results.items():
        row = summary[(summary["condition"] == cond) & (summary["pescoid"] == pid)]
        if row.empty:
            continue
        pheno = row.iloc[0]["phenotype_peak"]
        pa_t = meta["peak_ar_t"]
        if pa_t < 0 or pa_t >= profiles.shape[0]:
            continue
        prof = profiles[pa_t]
        if np.all(np.isnan(prof)):
            continue
        by_pheno.setdefault(pheno, []).append(prof)

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, AP_BINS)
    summary_rows = []
    for pheno in PHENO_ORDER:
        arr = by_pheno.get(pheno, [])
        if len(arr) < 2:
            continue
        arr = np.array(arr)
        mean = np.nanmean(arr, axis=0)
        # bootstrap 95% CI per bin
        n_boot = 500
        boot = np.array([np.nanmean(arr[rng.choice(len(arr), size=len(arr),
                                                     replace=True)], axis=0)
                          for _ in range(n_boot)])
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        color = PHENO_COLOR.get(pheno, "#888")
        ax.fill_between(x, lo, hi, color=color, alpha=0.2)
        ax.plot(x, mean, color=color, lw=2.2,
                label=f"{pheno} (n={len(arr)})")
        summary_rows.append({"phenotype": pheno, "n": len(arr),
                              "peak_ap": float(x[np.nanargmax(mean)]),
                              "peak_value": float(np.nanmax(mean))})
    ax.set_xlabel("AP position (normalised)")
    ax.set_ylabel("mezzo:GFP (bg-sub, per-pescoid max-norm)")
    ax.set_title("Population mezzo profile along AP axis at peak elongation\n"
                 "(mean +/- 95% CI bootstrap)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_ap_profile_by_phenotype.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(str(TABLES / "ap_profiles_summary.csv"), index=False)
    print("  02_ap_profile_by_phenotype.png")


def pole_ap_histogram(all_results, summary):
    """Histogram of pole AP positions, stratified by phenotype."""
    by_pheno = {}
    for (cond, pid), (_, meta) in all_results.items():
        row = summary[(summary["condition"] == cond) & (summary["pescoid"] == pid)]
        if row.empty:
            continue
        pheno = row.iloc[0]["phenotype_peak"]
        for p in meta["pole_ap_positions"]:
            by_pheno.setdefault(pheno, []).append(p["ap_position"])
    phenos = [p for p in PHENO_ORDER if p in by_pheno and len(by_pheno[p]) >= 2]
    if not phenos:
        return
    fig, axes = plt.subplots(1, len(phenos), figsize=(3.5 * len(phenos), 4), sharey=True)
    if len(phenos) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 11)
    for ax, pheno in zip(axes, phenos):
        ax.hist(by_pheno[pheno], bins=bins, color=PHENO_COLOR.get(pheno, "#888"),
                edgecolor="black")
        ax.set_title(f"{pheno}\n({len(by_pheno[pheno])} poles)", fontsize=10,
                      fontweight="bold")
        ax.set_xlabel("AP position")
    axes[0].set_ylabel("# poles")
    plt.suptitle("Distribution of mezzo+ pole AP positions by phenotype",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(PLOTS / "03_ap_pole_position_histogram.png"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  03_ap_pole_position_histogram.png")


# ---------------------------------------------------------------------------
def main(pids=None, examples=False, all_=False):
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary = summary[summary["condition"].isin(CONDITIONS)].copy()

    if pids:
        targets = summary[summary["pescoid"].isin(pids)].copy()
    elif examples:
        ex_set = {(c, p) for c, p, _ in EXAMPLES}
        targets = summary[summary[["condition", "pescoid"]].apply(
            lambda r: (r["condition"], r["pescoid"]) in ex_set, axis=1)].copy()
    elif all_:
        targets = summary
    else:
        print("Pass --pids, --examples, or --all"); return

    print(f"Processing {len(targets)} pescoids...")
    results = {}
    for _, row in targets.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        try:
            profiles, meta = process_pescoid(cond, pid)
            results[(cond, pid)] = (profiles, meta)
            print(f"  {cond}/{pid}: peak-AR @ {meta['peak_ar_hpf']:.1f} hpf, "
                  f"{len(meta['pole_ap_positions'])} poles")
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")

    if not results:
        return

    if examples or all_:
        # ensure the 5 example pescoids are processed for the panel
        for cond, pid, _ in EXAMPLES:
            if (cond, pid) not in results:
                try:
                    results[(cond, pid)] = process_pescoid(cond, pid)
                except Exception as e:
                    print(f"  example FAIL {cond}/{pid}: {e}")
        example_panel(results)

    if all_:
        population_ap_profile(results, summary)
        pole_ap_histogram(results, summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pids", nargs="*")
    p.add_argument("--examples", action="store_true",
                   help="5 representative pescoids for the poster panel")
    p.add_argument("--all", action="store_true",
                   help="all phenotyped pescoids + population plots")
    args = p.parse_args()
    main(pids=args.pids, examples=args.examples, all_=args.all)
