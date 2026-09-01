"""
Phase 4b - three follow-up analyses on the de-coherence finding:
  (a) Inside-vs-outside coherence stratified by phenotype, plus the number of
      distinct low-coherence connected components inside the mask at the peak
      frame.
  (b) Cause-vs-consequence: coherence at future pole sites BEFORE mezzo
      emergence vs random non-pole locations in the same pescoid at the same
      pre-emergence frames. Does coherence already differ at sites where poles
      will later form?
  (c) Pole-pole angular separation in bipolar (and multipolar) pescoids.
      Histogram + test for a peak near 180 deg (axial symmetry).

Inputs (no new acquisitions):
  Lyn_mezzo_phase1/per_pescoid/<cond>/<pid>/{lyntom,mask}_aligned.tif
  Lyn_mezzo_phase2/per_pescoid/<cond>/<pid>/mezzo_tracks.{csv,tif}
  Lyn_mezzo_phase2/phenotype_summary.csv
  Lyn_mezzo_phase4/tables/tension_per_pescoid.csv  (for peak_frame)

Coherence field is recomputed on the fly from lyntom_aligned (Gaussian-only,
fast; ~5 min for 52 pescoids).

Outputs:
  Lyn_mezzo_phase4/plots_b/   (PNGs)
  Lyn_mezzo_phase4/tables/phase4b_*.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
PHASE4 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase4")
PLOTS_B = PHASE4 / "plots_b"
TABLES = PHASE4 / "tables"
PLOTS_B.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#1f77b4", "P_Activin_3-5hpf": "#ff7f0e"}

HPF_START = 6.0
HPF_INTERVAL = 0.4
PEAK_HPF = (11.0, 15.0)

# Structure-tensor params (match phase4_tension.py)
SIGMA_GRAD = 5
SIGMA_TENSOR = 10

# Phase 4b thresholds
LOW_COH_THRESH = 0.20       # pixels below this are "low coherence"
MIN_ZONE_AREA = 200         # min connected-component area in px
POLE_SAMPLE_RADIUS = 12     # px disk radius around pole centroid for sampling
N_RANDOM_SITES = 8          # random non-pole sites per pescoid
RANDOM_MIN_DIST_FROM_POLE = 40  # px exclusion radius around pole centroid
RNG = np.random.default_rng(42)

# Phenotype groupings for (a)
PHENO_GROUPS = {
    "monopolar":      ["coordinated_monopolar"],
    "bipolar":        ["coordinated_bipolar", "mezzo_bipolar_only"],
    "multipolar":     ["multipolar", "disorganised_multipolar"],
    "diffuse_mezzo":  ["diffuse_mezzo"],
    "no_induction":   ["no_induction"],
}
PHENO_GROUP_ORDER = list(PHENO_GROUPS.keys())
PHENO_GROUP_COLOR = {
    "monopolar":     "#2ca02c",
    "bipolar":       "#9467bd",
    "multipolar":    "#d62728",
    "diffuse_mezzo": "#66c2a5",
    "no_induction":  "#888888",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0,
    "font.size": 11,
})


# ===========================================================================
# Helpers
# ===========================================================================
def norm_pct(img):
    lo, hi = np.percentile(img, (1, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else img / max(img.max(), 1)


def structure_tensor_coh(img, sigma_grad=SIGMA_GRAD, sigma_tensor=SIGMA_TENSOR):
    """Return (coherence, orientation) field."""
    ay = ndi.gaussian_filter(img, sigma=sigma_grad, order=(1, 0))
    ax = ndi.gaussian_filter(img, sigma=sigma_grad, order=(0, 1))
    Jxx = ndi.gaussian_filter(ax * ax, sigma=sigma_tensor)
    Jyy = ndi.gaussian_filter(ay * ay, sigma=sigma_tensor)
    Jxy = ndi.gaussian_filter(ax * ay, sigma=sigma_tensor)
    trace = Jxx + Jyy
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2
    coh = np.where(lam1 + lam2 > 0, (lam1 - lam2) / (lam1 + lam2 + 1e-10), 0.0)
    return coh.astype(np.float32)


def assign_phenotype_group(phenotype_peak):
    for grp, names in PHENO_GROUPS.items():
        if phenotype_peak in names:
            return grp
    return "other"


def load_aligned(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    lyn_p = pdir / "lyntom_aligned.tif"
    msk_p = pdir / "mask_aligned.tif"
    if not (lyn_p.exists() and msk_p.exists()):
        return None, None
    lyn = tifffile.imread(str(lyn_p))
    masks = tifffile.imread(str(msk_p)) > 0
    return lyn, masks


def load_pole_tracks(cond, pid):
    """Return DataFrame of cluster_type=='pole' tracks (may be empty)."""
    csv_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
    if not csv_p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(str(csv_p))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty or "cluster_type" not in df.columns:
        return pd.DataFrame()
    return df[df["cluster_type"] == "pole"].copy()


def load_pole_stack(cond, pid):
    """Return (T, Y, X) bool mask: True where mezzo pole tracks exist per frame."""
    tif_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.tif"
    csv_p = PHASE2 / "per_pescoid" / cond / pid / "mezzo_tracks.csv"
    if not (tif_p.exists() and csv_p.exists()):
        return None
    tracks_stack = tifffile.imread(str(tif_p))
    try:
        df = pd.read_csv(str(csv_p))
    except pd.errors.EmptyDataError:
        return np.zeros_like(tracks_stack, dtype=bool)
    if df.empty or "cluster_type" not in df.columns:
        return np.zeros_like(tracks_stack, dtype=bool)
    pole_ids = df[df["cluster_type"] == "pole"]["track_id"].tolist()
    if not pole_ids:
        return np.zeros_like(tracks_stack, dtype=bool)
    return np.isin(tracks_stack, pole_ids)


def disk_mask(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def mask_centroid(mask):
    props = measure.regionprops(mask.astype(int))
    if not props:
        return None
    p0 = max(props, key=lambda r: r.area)
    return p0.centroid  # (cy, cx)


def angle_diff_deg(a, b):
    """Smallest absolute angular distance in [0, 180]."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


# ===========================================================================
# (a) Per-pescoid: low-coherence zone count + inside/outside stratified
# ===========================================================================
def analysis_a_per_pescoid(cond, pid, peak_frame, lyn, masks, pole_stack):
    """Returns dict with n_low_coh_zones, total_low_coh_area, coh_in, coh_out at peak frame."""
    if peak_frame < 0 or peak_frame >= masks.shape[0]:
        return None
    mask = masks[peak_frame]
    if not mask.any():
        return None
    lyn_n = norm_pct(lyn[peak_frame].astype(np.float32))
    coh = structure_tensor_coh(lyn_n)

    # Inside vs outside mezzo+ pole at peak
    pole_mask = pole_stack[peak_frame] if pole_stack is not None and peak_frame < len(pole_stack) else np.zeros_like(mask, bool)
    pole_in_mask = pole_mask & mask
    non_pole_in_mask = mask & ~pole_in_mask
    coh_in = float(coh[pole_in_mask].mean()) if pole_in_mask.any() else np.nan
    coh_out = float(coh[non_pole_in_mask].mean()) if non_pole_in_mask.any() else np.nan

    # Low-coherence connected components inside the mask
    low = (coh < LOW_COH_THRESH) & mask
    low = morphology.binary_opening(low, morphology.disk(2))  # de-noise
    lab = measure.label(low, connectivity=2)
    props = measure.regionprops(lab)
    sizes = [p.area for p in props if p.area >= MIN_ZONE_AREA]
    n_zones = len(sizes)
    total_low_area = float(np.sum(sizes))
    mask_area = float(mask.sum())

    return {
        "n_low_coh_zones": n_zones,
        "low_coh_area_frac": total_low_area / mask_area if mask_area else 0.0,
        "coh_inside_pole_peak": coh_in,
        "coh_outside_pole_peak": coh_out,
        "mean_coh_peak": float(coh[mask].mean()),
    }


# ===========================================================================
# (b) Cause-vs-consequence: coherence at future pole site BEFORE emergence
# ===========================================================================
def analysis_b_per_pescoid(cond, pid, lyn, masks, pole_tracks_df, pole_stack):
    """Per pole track: mean coherence at the (future) pole site over frames
    [0, first_frame-1], plus matched random non-pole sites in the same frames.
    """
    if pole_tracks_df.empty:
        return []

    rows = []
    T = masks.shape[0]
    # Precompute coherence per frame on demand (cache)
    coh_cache = {}

    def get_coh(t):
        if t not in coh_cache:
            if not masks[t].any():
                coh_cache[t] = None
            else:
                coh_cache[t] = structure_tensor_coh(norm_pct(lyn[t].astype(np.float32)))
        return coh_cache[t]

    for _, tr in pole_tracks_df.iterrows():
        t_emerge = int(tr["first_frame"])
        if t_emerge <= 2:
            continue  # no meaningful "before" window
        cy, cx = float(tr["peak_centroid_y"]), float(tr["peak_centroid_x"])
        # Frames strictly before emergence
        pre_frames = list(range(0, t_emerge))

        # Future-pole-site disk at the centroid
        site_disk = disk_mask(masks.shape[1:], cy, cx, POLE_SAMPLE_RADIUS)

        # Future-pole-site coherence: average per pre-frame, then mean
        site_vals = []
        random_vals_per_frame = []  # list of lists per random site (size N_RANDOM_SITES)

        # Build exclusion zone using all pole centroids in this pescoid
        all_pole_centroids = [(float(r["peak_centroid_y"]), float(r["peak_centroid_x"]))
                              for _, r in pole_tracks_df.iterrows()]

        # Sample random non-pole sites using the mask at the *median* pre frame
        ref_t = pre_frames[len(pre_frames) // 2]
        ref_mask = masks[ref_t] if masks[ref_t].any() else (
            masks[max(0, t_emerge - 1)] if masks[max(0, t_emerge - 1)].any() else None
        )
        if ref_mask is None or not ref_mask.any():
            continue
        # Erode mask to avoid cortex pixels; exclude near-pole regions
        eroded = morphology.binary_erosion(ref_mask, morphology.disk(15))
        excl = np.zeros_like(eroded, bool)
        for (pcy, pcx) in all_pole_centroids:
            excl |= disk_mask(eroded.shape, pcy, pcx, RANDOM_MIN_DIST_FROM_POLE)
        candidate = eroded & ~excl
        if not candidate.any():
            continue
        ys, xs = np.where(candidate)
        if len(ys) < N_RANDOM_SITES:
            continue
        idx = RNG.choice(len(ys), size=N_RANDOM_SITES, replace=False)
        random_centroids = [(ys[i], xs[i]) for i in idx]
        random_disks = [disk_mask(masks.shape[1:], rcy, rcx, POLE_SAMPLE_RADIUS)
                        for (rcy, rcx) in random_centroids]

        # Loop pre-frames
        for t in pre_frames:
            mask_t = masks[t]
            if not mask_t.any():
                continue
            coh_t = get_coh(t)
            if coh_t is None:
                continue
            site_pix = site_disk & mask_t
            if site_pix.any():
                site_vals.append(float(coh_t[site_pix].mean()))
            for rd in random_disks:
                rd_pix = rd & mask_t
                if rd_pix.any():
                    random_vals_per_frame.append(float(coh_t[rd_pix].mean()))

        if not site_vals or not random_vals_per_frame:
            continue
        rows.append({
            "condition": cond,
            "pescoid": pid,
            "track_id": int(tr["track_id"]),
            "first_frame": t_emerge,
            "first_hpf": HPF_START + t_emerge * HPF_INTERVAL,
            "n_pre_frames": len(site_vals),
            "site_coh_mean": float(np.mean(site_vals)),
            "site_coh_n": len(site_vals),
            "random_coh_mean": float(np.mean(random_vals_per_frame)),
            "random_coh_n": len(random_vals_per_frame),
            "delta_site_minus_random": float(np.mean(site_vals) - np.mean(random_vals_per_frame)),
        })
    return rows


# ===========================================================================
# (c) Pole-pole angular separation
# ===========================================================================
def analysis_c_per_pescoid(cond, pid, pole_tracks_df, phenotype_peak, n_mezzo):
    """Return list of pairwise (angle_diff) records for all pole pairs."""
    if len(pole_tracks_df) < 2:
        return []
    angles = pole_tracks_df["peak_angular_pos_deg"].values
    rows = []
    n = len(angles)
    for i in range(n):
        for j in range(i + 1, n):
            d = angle_diff_deg(float(angles[i]), float(angles[j]))
            rows.append({
                "condition": cond,
                "pescoid": pid,
                "phenotype_peak": phenotype_peak,
                "n_mezzo_poles": int(n_mezzo),
                "angle_i_deg": float(angles[i]),
                "angle_j_deg": float(angles[j]),
                "abs_angular_separation_deg": d,
                "n_poles_in_pescoid": n,
            })
    return rows


# ===========================================================================
# Plot helpers
# ===========================================================================
def jitter(n, w=0.15):
    return RNG.uniform(-w, w, size=n)


def boxplot_by_group(ax, groups, labels, colors, ylabel, title, ylim=None,
                     hline=None, scatter=True):
    positions = list(range(len(groups)))
    bp = ax.boxplot(groups, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, boxprops=dict(alpha=0.35),
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    if scatter:
        for i, (vals, c) in enumerate(zip(groups, colors)):
            if len(vals) == 0:
                continue
            x = i + jitter(len(vals))
            ax.scatter(x, vals, s=55, color=c, edgecolors="black", lw=0.6,
                       alpha=0.85, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if hline is not None:
        ax.axhline(hline, ls="--", color="k", lw=1, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)


# ===========================================================================
# Main
# ===========================================================================
def run(pescoid_filter=None):
    print("=" * 60)
    print("  Phase 4b - decoherence follow-up analyses")
    print("=" * 60)

    # Load metadata
    summary_p2 = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary_p2 = summary_p2[summary_p2["condition"].isin(CONDITIONS)].copy()
    tension = pd.read_csv(str(TABLES / "tension_per_pescoid.csv"))

    df = summary_p2.merge(
        tension[["condition", "pescoid", "peak_frame", "peak_hpf"]],
        on=["condition", "pescoid"], how="left",
    )
    df["pheno_group"] = df["phenotype_peak"].apply(assign_phenotype_group)

    if pescoid_filter:
        df = df[df["pescoid"].isin(pescoid_filter)]
    print(f"  Pescoids to process: {len(df)}")

    rows_a = []
    rows_b = []
    rows_c = []

    for _, r in df.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        peak_frame = int(r["peak_frame"]) if pd.notna(r.get("peak_frame")) else None
        print(f"  {cond}/{pid} ({r['phenotype_peak']})... ", end="")

        lyn, masks = load_aligned(cond, pid)
        if lyn is None:
            print("SKIP (no aligned data)")
            continue
        pole_stack = load_pole_stack(cond, pid)
        pole_tracks = load_pole_tracks(cond, pid)

        # (a)
        if peak_frame is not None:
            a = analysis_a_per_pescoid(cond, pid, peak_frame, lyn, masks, pole_stack)
            if a is not None:
                a.update({
                    "condition": cond, "pescoid": pid,
                    "phenotype_peak": r["phenotype_peak"],
                    "pheno_group": r["pheno_group"],
                    "peak_frame": peak_frame,
                })
                rows_a.append(a)

        # (b)
        b_rows = analysis_b_per_pescoid(cond, pid, lyn, masks, pole_tracks, pole_stack)
        rows_b.extend(b_rows)

        # (c)
        c_rows = analysis_c_per_pescoid(cond, pid, pole_tracks,
                                        r["phenotype_peak"],
                                        r.get("n_mezzo_poles_peak", np.nan))
        rows_c.extend(c_rows)
        print(f"a:zones={rows_a[-1]['n_low_coh_zones'] if rows_a and rows_a[-1]['pescoid']==pid else '-'}, "
              f"b:poles_analysed={len(b_rows)}, c:pairs={len(c_rows)}")

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_c = pd.DataFrame(rows_c)
    df_a.to_csv(str(TABLES / "phase4b_a_per_pescoid.csv"), index=False)
    df_b.to_csv(str(TABLES / "phase4b_b_per_pole.csv"), index=False)
    df_c.to_csv(str(TABLES / "phase4b_c_pole_pairs.csv"), index=False)

    # -----------------------------------------------------------------------
    # Plot (a1): inside-vs-outside coherence stratified by phenotype group
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    cond_pos = {"P_ctrl": 0, "P_Activin_3-5hpf": 1}
    group_xticks = []
    group_xlabels = []
    pos = 0
    for grp in PHENO_GROUP_ORDER:
        sub = df_a[df_a["pheno_group"] == grp]
        if sub.empty:
            continue
        # Combine across conditions for this phenotype group
        in_vals = sub["coh_inside_pole_peak"].dropna().values
        out_vals = sub["coh_outside_pole_peak"].dropna().values
        x_in = pos + 0
        x_out = pos + 0.6
        bp_in = ax.boxplot([in_vals], positions=[x_in], widths=0.45, patch_artist=True,
                           showfliers=False, boxprops=dict(alpha=0.35),
                           medianprops=dict(color="black", lw=2))
        bp_out = ax.boxplot([out_vals], positions=[x_out], widths=0.45, patch_artist=True,
                            showfliers=False, boxprops=dict(alpha=0.35),
                            medianprops=dict(color="black", lw=2))
        for p in bp_in["boxes"]:
            p.set_facecolor("lime")
        for p in bp_out["boxes"]:
            p.set_facecolor("#888888")
        if len(in_vals):
            ax.scatter(x_in + jitter(len(in_vals), 0.08), in_vals, s=45,
                       color="lime", edgecolors="black", lw=0.5, alpha=0.85, zorder=3)
        if len(out_vals):
            ax.scatter(x_out + jitter(len(out_vals), 0.08), out_vals, s=45,
                       color="#888888", edgecolors="black", lw=0.5, alpha=0.85, zorder=3)
        group_xticks.append(pos + 0.3)
        group_xlabels.append(f"{grp}\n(n_in={len(in_vals)}, n_out={len(out_vals)})")
        pos += 1.6
    ax.set_xticks(group_xticks)
    ax.set_xticklabels(group_xlabels, fontsize=9)
    ax.set_ylabel("Mean coherence at peak frame")
    ax.set_title("Coherence INSIDE mezzo+ pole (lime) vs OUTSIDE (grey)\n"
                 "Stratified by phenotype group",
                 fontweight="bold")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(str(PLOTS_B / "a1_inside_vs_outside_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  a1_inside_vs_outside_by_phenotype.png")

    # -----------------------------------------------------------------------
    # Plot (a2): number of low-coherence zones per phenotype group
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = []
    labels = []
    colors = []
    for grp in PHENO_GROUP_ORDER:
        sub = df_a[df_a["pheno_group"] == grp]
        if sub.empty:
            continue
        groups.append(sub["n_low_coh_zones"].dropna().values)
        labels.append(f"{grp}\nn={len(sub)}")
        colors.append(PHENO_GROUP_COLOR[grp])
    boxplot_by_group(ax, groups, labels, colors,
                     ylabel=f"# low-coherence zones (coh<{LOW_COH_THRESH}, >={MIN_ZONE_AREA}px)",
                     title="Number of distinct low-coherence zones at peak frame",
                     hline=None)
    plt.tight_layout()
    plt.savefig(str(PLOTS_B / "a2_low_coh_zones_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  a2_low_coh_zones_by_phenotype.png")

    # -----------------------------------------------------------------------
    # Plot (a3): low-coherence AREA FRACTION per phenotype group (companion)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = []
    labels = []
    colors = []
    for grp in PHENO_GROUP_ORDER:
        sub = df_a[df_a["pheno_group"] == grp]
        if sub.empty:
            continue
        groups.append(sub["low_coh_area_frac"].dropna().values)
        labels.append(f"{grp}\nn={len(sub)}")
        colors.append(PHENO_GROUP_COLOR[grp])
    boxplot_by_group(ax, groups, labels, colors,
                     ylabel="Low-coherence area / mask area",
                     title="Fraction of mask occupied by low-coherence zones at peak",
                     ylim=(0, 1))
    plt.tight_layout()
    plt.savefig(str(PLOTS_B / "a3_low_coh_area_fraction_by_phenotype.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  a3_low_coh_area_fraction_by_phenotype.png")

    # -----------------------------------------------------------------------
    # Plot (b): paired site vs random coherence, per pescoid + condition
    # -----------------------------------------------------------------------
    if not df_b.empty:
        # Aggregate per pescoid (mean across multiple pole tracks if any)
        agg_b = df_b.groupby(["condition", "pescoid"]).agg(
            site=("site_coh_mean", "mean"),
            random=("random_coh_mean", "mean"),
            delta=("delta_site_minus_random", "mean"),
            first_hpf=("first_hpf", "mean"),
            n_tracks=("track_id", "count"),
        ).reset_index()
        agg_b.to_csv(str(TABLES / "phase4b_b_per_pescoid.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: paired strip
        ax = axes[0]
        for cond in CONDITIONS:
            sub = agg_b[agg_b["condition"] == cond]
            if sub.empty:
                continue
            x_base = list(CONDITIONS).index(cond)
            x_site = x_base - 0.2 + jitter(len(sub), 0.06)
            x_rand = x_base + 0.2 + jitter(len(sub), 0.06)
            for vs, vr, xs_, xr_ in zip(sub["site"], sub["random"], x_site, x_rand):
                ax.plot([xs_, xr_], [vs, vr], "-", color=COND_COLOR[cond],
                        alpha=0.35, lw=0.8)
            ax.scatter(x_site, sub["site"], s=60, color="lime", edgecolors="black",
                       lw=0.6, label="future pole site" if cond == CONDITIONS[0] else None,
                       zorder=3)
            ax.scatter(x_rand, sub["random"], s=60, color="#888", edgecolors="black",
                       lw=0.6, label="random non-pole site" if cond == CONDITIONS[0] else None,
                       zorder=3)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
        ax.set_ylabel("Mean coherence in pre-emergence frames")
        ax.set_title("Coherence at FUTURE pole site vs random non-pole site\n"
                     "(frames before mezzo emergence at that location)",
                     fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.set_ylim(0, 1)

        # Panel 2: delta swarm
        ax = axes[1]
        groups = [agg_b[agg_b["condition"] == c]["delta"].dropna().values for c in CONDITIONS]
        labels = [f"{COND_LABEL[c]}\nn={len(g)}" for c, g in zip(CONDITIONS, groups)]
        colors = [COND_COLOR[c] for c in CONDITIONS]
        boxplot_by_group(ax, groups, labels, colors,
                         ylabel="Delta coherence (site - random)",
                         title="Pre-emergence: site minus random\n"
                               "(positive = pole site MORE coherent before forming)",
                         hline=0.0)
        plt.tight_layout()
        plt.savefig(str(PLOTS_B / "b_cause_vs_consequence.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  b_cause_vs_consequence.png")

        # Print condition-level summary
        cond_summary = agg_b.groupby("condition").agg(
            n=("pescoid", "count"),
            mean_site=("site", "mean"),
            mean_random=("random", "mean"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
        ).round(4)
        cond_summary.to_csv(str(TABLES / "phase4b_b_condition_summary.csv"))
        print("\n=== (b) Pre-emergence coherence: site vs random ===")
        print(cond_summary.to_string())
    else:
        print("  (b) skipped: no pole tracks with sufficient pre-emergence frames")

    # -----------------------------------------------------------------------
    # Plot (c): pole-pole angular separation histogram
    # -----------------------------------------------------------------------
    if not df_c.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: histogram for bipolar (exactly 2 poles) pescoids
        ax = axes[0]
        bipolar = df_c[df_c["n_poles_in_pescoid"] == 2]
        bins = np.arange(0, 181, 15)
        if not bipolar.empty:
            ax.hist(bipolar["abs_angular_separation_deg"], bins=bins,
                    color="#9467bd", edgecolor="black", alpha=0.85)
            mean_a = bipolar["abs_angular_separation_deg"].mean()
            ax.axvline(180, ls="--", color="red", lw=2, label="180 deg")
            ax.axvline(mean_a, ls="-", color="black", lw=1.5,
                       label=f"mean = {mean_a:.0f} deg")
            ax.legend()
        ax.set_xlabel("|angular separation| [deg]")
        ax.set_ylabel("# pole pairs")
        ax.set_title(f"Bipolar pescoids (exactly 2 mezzo poles)\n"
                     f"n_pescoids = {bipolar['pescoid'].nunique()}",
                     fontweight="bold")
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))

        # Panel 2: histogram for multipolar (>=3 poles) - all pairwise
        ax = axes[1]
        multi = df_c[df_c["n_poles_in_pescoid"] >= 3]
        if not multi.empty:
            ax.hist(multi["abs_angular_separation_deg"], bins=bins,
                    color="#d62728", edgecolor="black", alpha=0.85)
            ax.axvline(180, ls="--", color="red", lw=2, label="180 deg")
            ax.axvline(multi["abs_angular_separation_deg"].mean(), ls="-",
                       color="black", lw=1.5,
                       label=f"mean = {multi['abs_angular_separation_deg'].mean():.0f} deg")
            ax.legend()
        ax.set_xlabel("|angular separation| [deg]")
        ax.set_ylabel("# pole pairs")
        ax.set_title(f"Multipolar pescoids (>=3 mezzo poles) - all pairs\n"
                     f"n_pescoids = {multi['pescoid'].nunique()}, "
                     f"n_pairs = {len(multi)}",
                     fontweight="bold")
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))

        plt.tight_layout()
        plt.savefig(str(PLOTS_B / "c_pole_angular_separation_hist.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  c_pole_angular_separation_hist.png")

        # Print bipolar summary
        if not bipolar.empty:
            print("\n=== (c) Bipolar pescoids - pole-pole angular separation ===")
            print(bipolar[["condition", "pescoid", "phenotype_peak",
                           "abs_angular_separation_deg"]].to_string(index=False))
    else:
        print("  (c) skipped: no pescoids with >=2 mezzo poles")

    print(f"\nDone. Plots in: {PLOTS_B}/")
    print(f"Tables in: {TABLES}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true",
                   help="run on G046 + G021 only")
    p.add_argument("--all", action="store_true", help="run on all P pescoids")
    p.add_argument("--pids", nargs="*", help="manual pid list")
    args = p.parse_args()
    if args.sanity:
        run(pescoid_filter=["G046", "G021"])
    elif args.all:
        run()
    elif args.pids:
        run(pescoid_filter=args.pids)
    else:
        p.print_help()
