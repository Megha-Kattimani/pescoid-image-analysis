"""
Phase 2 — Pole / Organising-centre classification per pescoid.
=================================================================

Two pole axes are tracked IN PARALLEL:
  1) MEZZO poles  : GFP+ clusters segmented from the fluorescence channel.
  2) MORPH  poles : BF-outline curvature peaks (protrusions of the shape).

Co-located morph + mezzo poles (centroids within COORD_DIST_PX) are
flagged as 'coordinated'. The phenotype categorisation combines both
axes plus organising centres.

Mezzo cluster classification
----------------------------
  Pole               : persistent (>= POLE_PERSISTENCE_MIN frames) AND
                       peak area >= AREA_POLE_PX AND
                       peripheral at peak (offset > PERIPHERAL_OFFSET_FRAC)
                       AND independent (see below).
  Organising centre  : persistent (>= OC_PERSISTENCE_MIN frames) AND
                       peak area >= AREA_OC_PX AND
                       (NOT peripheral OR AREA_OC_PX <= area < AREA_POLE_PX).
  Transient cluster  : everything else.

Independence rule (replaces simple inter-pole distance)
-------------------------------------------------------
A track counts as a *separate* pole if ALL hold:
  (a) duration  >= POLE_PERSISTENCE_MIN frames
  (b) peak area >= AREA_POLE_PX
  (c) peak offset > PERIPHERAL_OFFSET_FRAC of equivalent-circle radius
  AND ONE of:
  (i)   it did NOT emerge as a split-daughter of any previously-existing
        cluster (= independent emergence), OR
  (ii)  it emerged as a split-daughter BUT persists >=
        SPLIT_DAUGHTER_PERSISTENCE frames AND shows stable / growing
        area over its lifetime (mean area in second half >= mean area
        in first half).

GFP intensity threshold X
-------------------------
GLOBAL: 99th percentile of bg-subtracted GFP in P_ctrl + E_ctrl at
frames 0-5. Single number applied to every pescoid.

Morphological pole detection
----------------------------
For each frame, find local maxima of the radial distance from centroid
along the BF outline (curvature peaks). A peak counts as a morph pole
if its radial distance > (1 + MORPH_PROTRUSION_FRAC) * eq_circle_r,
and the peak persists (with nearest-neighbour linking) for at least
POLE_PERSISTENCE_MIN frames.

Phenotype categorisation
------------------------
Computed both at PEAK window (11-15 hpf) and at ENDSTATE (last 3 frames):
  coordinated_bipolar    : 2 morph + 2 mezzo, >=2 coordinated
  coordinated_monopolar  : 1 morph + 1 mezzo, coordinated
  morph_bipolar_only     : 2 morph + <2 mezzo
  mezzo_bipolar_only     : <2 morph + 2 mezzo
  disorganised_multipolar: >=2 morph + >=2 mezzo but <2 coordinated, OR
                           >=2 OCs with <2 poles
  multipolar             : >=3 morph poles OR >=3 mezzo poles
  oc_only                : 0 morph + 0 mezzo, >=2 OCs
  no_induction           : 0 morph + 0 mezzo, <=1 OC

Usage
-----
    python phase2_poles.py --sanity            # G046 + G021 sanity check
    python phase2_poles.py --all               # full run on QC-pass pescoids
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage import measure, morphology
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE1_QC = PHASE1 / "phase1_QC.csv"
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
OUT.mkdir(parents=True, exist_ok=True)
PER = OUT / "per_pescoid"
PER.mkdir(parents=True, exist_ok=True)
PHENO = OUT / "phenotype_categorised"
PHENO.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["E_ctrl", "E_Activin_3-5hpf", "P_ctrl", "P_Activin_3-5hpf"]
HPF_START = 6.0
HPF_INTERVAL = 0.4

# --- thresholds (locked in with user) ---
GLOBAL_THR_PERCENTILE = 99
GLOBAL_THR_FRAMES = (0, 5)
GLOBAL_THR_CONDITIONS = ["P_ctrl", "E_ctrl"]

AREA_POLE_PX = 500
AREA_OC_PX = 100                 # was 200 -> 100
POLE_PERSISTENCE_MIN = 5         # was 3 -> 5 (more conservative for poles)
OC_PERSISTENCE_MIN = 3
SPLIT_DAUGHTER_PERSISTENCE = 10  # split-daughter must last >= 10 frames to be a pole
PERIPHERAL_OFFSET_FRAC = 0.30
LINK_IOU = 0.30
PEAK_WINDOW_HPF = (11.0, 15.0)   # was (11.0, 13.0) -> (11.0, 15.0)

# Morphological pole detection
MORPH_PROTRUSION_FRAC = 0.30     # contour radius must exceed (1 + frac) * eq_r
MORPH_SMOOTH_SIGMA = 5           # gaussian smoothing on radial profile (perimeter bins)
MORPH_PEAK_PROMINENCE_FRAC = 0.10  # peak prominence as fraction of eq_r
MORPH_PEAK_MIN_DIST_DEG = 30       # min angular separation of curvature peaks
MORPH_LINK_DIST_PX = 50            # nearest-neighbour linking distance across frames
MORPH_LINK_MAX_GAP = 2             # max # frames a track can be missing and still be linked

# Coordination matching: morph + mezzo poles at same ANGULAR position
COORD_ANGLE_DEG = 45  # +/- 45° around pescoid centroid counts as 'same place'

# Mezzo confirmation of morph poles
#   A BF curvature peak is only promoted to a "morph pole" if the mezzo signal
#   inside the bud WEDGE (from pescoid centroid out to the bud tip, ±half-angle)
#   is locally elevated vs the rest of the mask. The wedge captures the whole
#   bud — both the tip and the base where it joins the body — so mezzo that is
#   concentrated near the bud's base (not its tip) is still detected.
LOCAL_MEZZO_FACTOR = 1.00      # local_mean must be strictly > surrounding_mean
                                # (user's criterion: "higher than the surrounding")
LOCAL_MEZZO_WEDGE_HALF_DEG = 25  # half-angle of the wedge in degrees (-> 50 deg total)

# Conditions to phenotype (embryos are QC-only and NOT classified in Phase 2)
PHENOTYPE_CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]

# Phenotype categories
PHENO_CATS = [
    "coordinated_bipolar", "coordinated_monopolar",
    "morph_bipolar_only", "mezzo_bipolar_only",
    "multipolar", "disorganised_multipolar",
    "oc_only", "no_induction", "disintegrated_early",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.size": 10,
})


# ===========================================================================
# Helpers
# ===========================================================================
def load_phase1_metrics():
    return pd.read_csv(str(PHASE1_QC))


def load_aligned(cond, pid):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    bf_p = pdir / "bf_aligned.tif"
    gfp_p = pdir / "gfp_aligned.tif"
    mask_p = pdir / "mask_aligned.tif"
    if not (bf_p.exists() and gfp_p.exists() and mask_p.exists()):
        return None
    bf = tifffile.imread(str(bf_p))
    gfp = tifffile.imread(str(gfp_p))
    masks = tifffile.imread(str(mask_p)) > 0
    return bf, gfp, masks


def compute_global_threshold(qc_df):
    print(f"\nComputing global GFP threshold from {GLOBAL_THR_CONDITIONS} at frames {GLOBAL_THR_FRAMES}...")
    vals = []
    for _, row in qc_df.iterrows():
        if row["condition"] not in GLOBAL_THR_CONDITIONS or not row["qc_pass"]:
            continue
        loaded = load_aligned(row["condition"], row["pescoid"])
        if loaded is None:
            continue
        bf, gfp, masks = loaded
        T = gfp.shape[0]
        for t in range(GLOBAL_THR_FRAMES[0], min(GLOBAL_THR_FRAMES[1] + 1, T)):
            mask = masks[t]
            if not mask.any():
                continue
            in_vals = gfp[t][mask]
            bg_mean = float(gfp[t][~mask].mean())
            vals.append((in_vals - bg_mean).clip(min=0))
    all_vals = np.concatenate(vals)
    thr = float(np.percentile(all_vals, GLOBAL_THR_PERCENTILE))
    print(f"  N pixels used: {len(all_vals):,}")
    print(f"  Global threshold (bg-subtracted GFP, p{GLOBAL_THR_PERCENTILE}): {thr:.2f}")
    return thr


# ===========================================================================
# 1.  MEZZO clusters per frame
# ===========================================================================
def label_clusters(gfp_frame, bf_mask, bg_mean, thr_bgsub, min_area=AREA_OC_PX):
    bg_sub = (gfp_frame - bg_mean).clip(min=0)
    positive = (bg_sub > thr_bgsub) & bf_mask
    positive = morphology.binary_opening(positive, morphology.disk(2))
    positive = ndi.binary_fill_holes(positive)
    labeled, n = ndi.label(positive)
    if n > 0:
        sizes = ndi.sum(positive, labeled, range(1, n + 1))
        keep = np.where(sizes >= min_area)[0] + 1
        new_labeled = np.zeros_like(labeled)
        for i, k in enumerate(keep, start=1):
            new_labeled[labeled == k] = i
        labeled = new_labeled
    return labeled


def iou(mask_a, mask_b):
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return inter / union if union > 0 else 0.0


def link_tracks(labeled_stack):
    """IoU-based mezzo-cluster tracker. Returns track-id stack + fusion/split events."""
    T = labeled_stack.shape[0]
    track_id_stack = np.zeros_like(labeled_stack)
    next_id = 1
    frame_clusters = [[] for _ in range(T)]

    for t in range(T):
        lab = labeled_stack[t]
        n = int(lab.max())
        clusters = []
        for k in range(1, n + 1):
            cluster_mask = (lab == k)
            if t == 0:
                tid = next_id; next_id += 1
                track_id_stack[t][cluster_mask] = tid
            else:
                candidates = []
                for prev_tid, _, _, prev_mask in frame_clusters[t - 1]:
                    if iou(cluster_mask, prev_mask) > LINK_IOU:
                        candidates.append((prev_tid, iou(cluster_mask, prev_mask)))
                if candidates:
                    candidates.sort(key=lambda x: -x[1])
                    tid = candidates[0][0]
                    track_id_stack[t][cluster_mask] = tid
                else:
                    tid = next_id; next_id += 1
                    track_id_stack[t][cluster_mask] = tid
            clusters.append((tid, k, int(cluster_mask.sum()), cluster_mask))
        frame_clusters[t] = clusters

    fusion_events = []
    split_events = []
    for t in range(1, T):
        for tid_curr, _, _, curr_mask in frame_clusters[t]:
            distinct_prev = set()
            for prev_tid, _, _, prev_mask in frame_clusters[t - 1]:
                if iou(prev_mask, curr_mask) > LINK_IOU:
                    distinct_prev.add(prev_tid)
            if len(distinct_prev) >= 2:
                ys, xs = np.where(curr_mask)
                fusion_events.append({
                    "frame": t,
                    "parent_tracks": sorted(distinct_prev),
                    "child_track": tid_curr,
                    "position_y": float(ys.mean()),
                    "position_x": float(xs.mean()),
                })
    for t in range(T - 1):
        for prev_tid, _, _, prev_mask in frame_clusters[t]:
            daughters = set()
            for tid_next, _, _, next_mask in frame_clusters[t + 1]:
                if iou(prev_mask, next_mask) > LINK_IOU:
                    daughters.add(tid_next)
            if len(daughters) >= 2:
                ys, xs = np.where(prev_mask)
                split_events.append({
                    "frame": t + 1,
                    "parent_track": prev_tid,
                    "daughter_tracks": sorted(daughters),
                    "position_y": float(ys.mean()),
                    "position_x": float(xs.mean()),
                })
    return track_id_stack, fusion_events, split_events


def classify_mezzo_tracks(track_id_stack, masks_aligned, gfp_aligned, bg_means,
                          split_events):
    """
    Classify each track using track-history independence rule.

    A track counts as a separate POLE iff:
      (a) duration  >= POLE_PERSISTENCE_MIN
      (b) peak area >= AREA_POLE_PX
      (c) peripheral at peak (> PERIPHERAL_OFFSET_FRAC)
      AND ONE of:
        (i)  did NOT emerge as a split-daughter, OR
        (ii) emerged as split-daughter AND persists >= SPLIT_DAUGHTER_PERSISTENCE
             AND mean area in 2nd half >= mean area in 1st half.

    OC = persistent (>= OC_PERSISTENCE_MIN) + area >= AREA_OC_PX +
         (NOT peripheral OR small area).
    """
    T = track_id_stack.shape[0]
    all_ids = np.unique(track_id_stack); all_ids = all_ids[all_ids > 0]

    # Build a set of (frame, track_id) that are split-daughters
    split_daughter_set = set()
    for ev in split_events:
        for d in ev["daughter_tracks"]:
            split_daughter_set.add((ev["frame"], d))

    tracks = {}
    for tid in all_ids:
        present = []
        for t in range(T):
            cluster_mask = (track_id_stack[t] == tid)
            if not cluster_mask.any():
                continue
            area = int(cluster_mask.sum())
            vals = gfp_aligned[t][cluster_mask]
            mean_int = float(vals.mean() - bg_means[t])
            ys, xs = np.where(cluster_mask)
            cy, cx = float(ys.mean()), float(xs.mean())
            bf_mask = masks_aligned[t]
            if not bf_mask.any():
                continue
            bf_props = measure.regionprops(bf_mask.astype(int))
            if not bf_props:
                continue
            p0 = max(bf_props, key=lambda r: r.area)
            bf_cy, bf_cx = p0.centroid
            eq_r = float(np.sqrt(p0.area / np.pi))
            dist = float(np.hypot(cy - bf_cy, cx - bf_cx))
            offset_norm = dist / eq_r if eq_r > 0 else 0.0
            ang = float(np.degrees(np.arctan2(cy - bf_cy, cx - bf_cx)))
            present.append({
                "frame": t, "area": area, "mean_int_bgsub": mean_int,
                "centroid_y": cy, "centroid_x": cx,
                "offset_norm": offset_norm, "angular_pos_deg": ang,
            })
        if not present:
            continue
        df = pd.DataFrame(present)
        peak_row = df.loc[df["area"].idxmax()]
        duration = len(df)
        peak_area = float(peak_row["area"])
        peak_offset = float(peak_row["offset_norm"])

        # Track-history independence
        # A track emerged as a split-daughter if (first_frame, tid) is in split_daughter_set
        emerged_as_split = (int(df["frame"].min()), int(tid)) in split_daughter_set
        if emerged_as_split:
            # check growth: 2nd half mean area >= 1st half
            mid = len(df) // 2 if len(df) >= 2 else 1
            first_half_area = df["area"].iloc[:mid].mean()
            second_half_area = df["area"].iloc[mid:].mean()
            growth_ok = (second_half_area >= first_half_area)
            independent = (duration >= SPLIT_DAUGHTER_PERSISTENCE and growth_ok)
        else:
            independent = True  # emerged on its own

        # Classification
        if (duration >= POLE_PERSISTENCE_MIN
                and peak_area >= AREA_POLE_PX
                and peak_offset > PERIPHERAL_OFFSET_FRAC
                and independent):
            ctype = "pole"
        elif (duration >= OC_PERSISTENCE_MIN and peak_area >= AREA_OC_PX
              and (peak_offset <= PERIPHERAL_OFFSET_FRAC or peak_area < AREA_POLE_PX)):
            ctype = "organising_centre"
        else:
            ctype = "transient"

        tracks[int(tid)] = {
            "track_id": int(tid),
            "duration_frames": duration,
            "first_frame": int(df["frame"].min()),
            "last_frame": int(df["frame"].max()),
            "peak_frame": int(peak_row["frame"]),
            "peak_area_px": peak_area,
            "peak_intensity_bgsub": float(peak_row["mean_int_bgsub"]),
            "peak_offset_norm": peak_offset,
            "peak_angular_pos_deg": float(peak_row["angular_pos_deg"]),
            "peak_centroid_y": float(peak_row["centroid_y"]),
            "peak_centroid_x": float(peak_row["centroid_x"]),
            "emerged_as_split_daughter": bool(emerged_as_split),
            "cluster_type": ctype,
            "presence_frames": df["frame"].tolist(),
        }
    return tracks


# ===========================================================================
# 2.  MORPHOLOGICAL poles from BF outline curvature
# ===========================================================================
def detect_morph_peaks_in_frame(mask):
    """
    Detect candidate morphological protrusion tips in one frame.

    Returns list of {y, x, radial_dist, eq_r, protrusion_frac, angle_deg}.
    """
    if not mask.any():
        return []
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    if len(contour) < 20:
        return []

    bf_props = measure.regionprops(mask.astype(int))
    p0 = max(bf_props, key=lambda r: r.area)
    cy, cx = p0.centroid
    eq_r = float(np.sqrt(p0.area / np.pi))

    radii = np.hypot(contour[:, 0] - cy, contour[:, 1] - cx)
    radii_smooth = gaussian_filter1d(radii, sigma=MORPH_SMOOTH_SIGMA, mode="wrap")

    # min distance between peaks: convert MORPH_PEAK_MIN_DIST_DEG to perimeter bins
    n_bins = len(radii_smooth)
    min_dist = int(MORPH_PEAK_MIN_DIST_DEG / 360.0 * n_bins)
    peaks, _ = find_peaks(
        radii_smooth,
        prominence=MORPH_PEAK_PROMINENCE_FRAC * eq_r,
        distance=max(5, min_dist),
    )

    results = []
    for pi in peaks:
        r_val = float(radii_smooth[pi])
        protrusion_frac = (r_val - eq_r) / eq_r if eq_r > 0 else 0
        if protrusion_frac < MORPH_PROTRUSION_FRAC:
            continue
        py, px = float(contour[pi, 0]), float(contour[pi, 1])
        ang = float(np.degrees(np.arctan2(py - cy, px - cx)))
        results.append({
            "y": py, "x": px,
            "radial_dist": r_val,
            "eq_r": eq_r,
            "protrusion_frac": protrusion_frac,
            "angle_deg": ang,
        })
    return results


def mezzo_local_vs_surrounding(track_summary, masks, gfp, bg_means):
    """
    Check whether the BF protrusion region of a morph track shows locally
    elevated bg-subtracted mezzo intensity vs the rest of the mask.

    Returns
    -------
    local_mean      : mean bg-subtracted GFP in the disk around the tip
    surrounding_mean: mean bg-subtracted GFP in (mask \ disk)
    ratio           : local_mean / surrounding_mean   (np.nan if surrounding == 0)
    confirmed       : bool, ratio >= LOCAL_MEZZO_FACTOR
    """
    t = track_summary["peak_frame"]
    if t >= masks.shape[0] or not masks[t].any():
        return 0.0, 0.0, np.nan, False
    yy, xx = np.mgrid[: masks[t].shape[0], : masks[t].shape[1]]
    bf_props = measure.regionprops(masks[t].astype(int))
    if not bf_props:
        return 0.0, 0.0, np.nan, False
    p0 = max(bf_props, key=lambda r: r.area)
    cy0, cx0 = p0.centroid  # pescoid centroid

    # Bud wedge = angular sector from centroid out to the bud tip, ±half-angle.
    # Captures both tip and base of the protrusion.
    tip_angle_deg = track_summary["peak_angle_deg"]
    pixel_angles = np.degrees(np.arctan2(yy - cy0, xx - cx0))
    angle_diff = (pixel_angles - tip_angle_deg + 180) % 360 - 180
    in_wedge = (np.abs(angle_diff) <= LOCAL_MEZZO_WEDGE_HALF_DEG)

    wedge_in_mask = masks[t] & in_wedge
    surrounding = masks[t] & ~in_wedge
    if not wedge_in_mask.any() or not surrounding.any():
        return 0.0, 0.0, np.nan, False

    bg = bg_means[t]
    bg_sub = (gfp[t] - bg).clip(min=0)
    local_mean = float(bg_sub[wedge_in_mask].mean())
    surrounding_mean = float(bg_sub[surrounding].mean())
    if surrounding_mean <= 0:
        return local_mean, surrounding_mean, np.nan, False
    ratio = local_mean / surrounding_mean
    return local_mean, surrounding_mean, ratio, ratio >= LOCAL_MEZZO_FACTOR


def track_morph_peaks(masks_aligned, gfp_aligned=None, bg_means=None):
    """
    Track curvature peaks across frames using nearest-neighbour linking.

    If gfp_aligned + bg_means are provided, also computes mezzo-confirmation:
    a morph peak is promoted to a "morph pole" ONLY IF its protrusion-tip region
    shows locally elevated mezzo intensity vs the rest of the mask (>=
    LOCAL_MEZZO_FACTOR x surrounding mean). This catches BF-visible buds whose
    mezzo signal is below the strict global GFP threshold but locally enriched.

    Returns dict track_id -> summary with `is_pole` flag set.
    """
    T = len(masks_aligned)
    per_frame = [detect_morph_peaks_in_frame(masks_aligned[t]) for t in range(T)]

    tracks = {}        # track_id -> list of (frame, peak)
    next_id = 1
    active = {}        # track_id -> (last_frame, last_pos)

    for t, frame_peaks in enumerate(per_frame):
        used = set()
        for pk in frame_peaks:
            # Find best matching active track within MORPH_LINK_DIST_PX
            best, best_d = None, float("inf")
            for tid, (last_t, last_pos) in active.items():
                if t - last_t > MORPH_LINK_MAX_GAP:
                    continue
                if tid in used:
                    continue
                d = float(np.hypot(pk["y"] - last_pos["y"], pk["x"] - last_pos["x"]))
                if d < MORPH_LINK_DIST_PX and d < best_d:
                    best_d, best = d, tid
            if best is not None:
                used.add(best)
                tracks[best].append((t, pk))
                active[best] = (t, pk)
            else:
                tracks[next_id] = [(t, pk)]
                active[next_id] = (t, pk)
                next_id += 1

    # Summarise each track + classify
    summaries = {}
    for tid, items in tracks.items():
        frames = [it[0] for it in items]
        peaks = [it[1] for it in items]
        duration = len(items)
        peak_idx = int(np.argmax([p["radial_dist"] for p in peaks]))
        peak_frame = frames[peak_idx]
        peak_pk = peaks[peak_idx]

        shape_ok = (duration >= POLE_PERSISTENCE_MIN
                    and peak_pk["protrusion_frac"] > PERIPHERAL_OFFSET_FRAC)
        ts = {
            "track_id": int(tid),
            "duration_frames": duration,
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "peak_frame": int(peak_frame),
            "peak_y": float(peak_pk["y"]),
            "peak_x": float(peak_pk["x"]),
            "peak_radial_dist": float(peak_pk["radial_dist"]),
            "peak_protrusion_frac": float(peak_pk["protrusion_frac"]),
            "peak_angle_deg": float(peak_pk["angle_deg"]),
            "presence_frames": list(frames),
            "shape_ok": bool(shape_ok),
        }

        # Mezzo confirmation: only count as a pole if local mezzo > surrounding
        if gfp_aligned is not None and bg_means is not None and shape_ok:
            local, surround, ratio, confirmed = mezzo_local_vs_surrounding(
                ts, masks_aligned, gfp_aligned, bg_means
            )
            ts["local_mezzo_bgsub_mean"] = local
            ts["surrounding_mezzo_bgsub_mean"] = surround
            ts["local_to_surrounding_ratio"] = ratio
            ts["mezzo_confirmed"] = bool(confirmed)
            ts["is_pole"] = bool(shape_ok and confirmed)
        else:
            ts["local_mezzo_bgsub_mean"] = 0.0
            ts["surrounding_mezzo_bgsub_mean"] = 0.0
            ts["local_to_surrounding_ratio"] = np.nan
            ts["mezzo_confirmed"] = False
            ts["is_pole"] = False  # if we can't confirm, don't count

        summaries[int(tid)] = ts
    return summaries


# ===========================================================================
# 3.  Coordination matching
# ===========================================================================
def _ang_diff_deg(a, b):
    """Signed shortest angular difference in degrees, range [-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return d


def match_coordination(mezzo_tracks, morph_tracks):
    """Pair morph + mezzo poles whose ANGULAR position around the pescoid centroid
    differs by less than COORD_ANGLE_DEG. This is more robust than Euclidean
    distance for elongated pescoids where the morph tip and the mezzo cluster
    centroid sit at the same end of the pescoid but ~50-100 px apart.
    """
    mz_poles = {tid: t for tid, t in mezzo_tracks.items() if t["cluster_type"] == "pole"}
    mo_poles = {tid: t for tid, t in morph_tracks.items() if t["is_pole"]}

    coords = []
    used_mz = set()
    for mo_tid, mo in mo_poles.items():
        best, best_da = None, 999.0
        for mz_tid, mz in mz_poles.items():
            if mz_tid in used_mz:
                continue
            da = abs(_ang_diff_deg(mo["peak_angle_deg"], mz["peak_angular_pos_deg"]))
            if da < best_da:
                best_da, best = da, mz_tid
        if best is not None and best_da < COORD_ANGLE_DEG:
            mz = mz_poles[best]
            d_px = float(np.hypot(mo["peak_y"] - mz["peak_centroid_y"],
                                    mo["peak_x"] - mz["peak_centroid_x"]))
            coords.append({
                "morph_track": mo_tid,
                "mezzo_track": best,
                "angle_diff_deg": best_da,
                "dist_px": d_px,
                "morph_peak_frame": mo["peak_frame"],
                "mezzo_peak_frame": mz["peak_frame"],
            })
            used_mz.add(best)
    return coords


# ===========================================================================
# 4.  Phenotype categorisation
# ===========================================================================
def count_per_window(items, frame_range, key_present="presence_frames"):
    out = []
    for it in items:
        if set(it[key_present]) & set(frame_range):
            out.append(it)
    return out


def phenotype_category(n_morph, n_mezzo, n_coord, n_ocs):
    """
    Map (n_morph_poles, n_mezzo_poles, n_coordinated, n_ocs) -> phenotype label.

    Priority order:
      1. multipolar (>= 3 poles on either axis)
      2. bipolar cases (>=2 of either axis)
      3. monopolar cases (1+1)
      4. mezzo-only or morph-only single
      5. oc_only
      6. no_induction
    """
    # Multipolar wins everything
    if n_morph >= 3 or n_mezzo >= 3:
        return "multipolar"

    # Bipolar regime: at least 2 of either axis
    if n_morph >= 2 and n_mezzo >= 2:
        return "coordinated_bipolar" if n_coord >= 2 else "disorganised_multipolar"
    if n_morph >= 2 and n_mezzo < 2:
        return "morph_bipolar_only"
    if n_mezzo >= 2 and n_morph < 2:
        return "mezzo_bipolar_only"

    # Monopolar regime: 1+1 paired or single-axis 1
    if n_morph == 1 and n_mezzo == 1:
        # Single morph + single mezzo: coordinated if they're co-located,
        # disorganised otherwise (chemistry and mechanics on different axes).
        return "coordinated_monopolar" if n_coord >= 1 else "disorganised_multipolar"

    # Single-axis monopolar variants
    # 1 mezzo + 0 morph: real mezzo induction without sustained morph protrusion.
    # We still call this 'coordinated_monopolar' as a monopolar fate-only state
    # — distinct from no_induction.
    if n_mezzo == 1 and n_morph == 0:
        return "coordinated_monopolar"

    # 1 morph + 0 mezzo: a single morph peak on a non-induced pescoid is noise
    # (a transient bump on a round shape). Treat as no_induction unless OCs present.
    if n_morph == 1 and n_mezzo == 0:
        if n_ocs >= 2:
            return "oc_only"
        return "no_induction"

    # 0 morph + 0 mezzo: OC-only vs no_induction
    if n_ocs >= 2:
        return "oc_only"
    return "no_induction"


# ===========================================================================
# Core: process one pescoid
# ===========================================================================
def process_pescoid(cond, pid, global_thr, save_outputs=True,
                    disintegration_frame=None):
    loaded = load_aligned(cond, pid)
    if loaded is None:
        return None
    bf, gfp, masks = loaded
    T = bf.shape[0]

    # Cap analysis at disintegration_frame if any (recorded in Phase 1)
    T_eff = T if disintegration_frame is None or disintegration_frame < 0 else min(T, int(disintegration_frame))
    if T_eff < T:
        bf = bf[:T_eff]; gfp = gfp[:T_eff]; masks = masks[:T_eff]; T = T_eff

    bg_means = np.zeros(T)
    for t in range(T):
        bg_pixels = gfp[t][~masks[t]] if (~masks[t]).any() else gfp[t]
        bg_means[t] = float(bg_pixels.mean())

    # ----- MEZZO -----
    labeled_stack = np.zeros_like(masks, dtype=np.int32)
    for t in range(T):
        labeled_stack[t] = label_clusters(gfp[t], masks[t], bg_means[t], global_thr,
                                            min_area=AREA_OC_PX)
    track_id_stack, fusion_events, split_events = link_tracks(labeled_stack)
    mezzo_tracks = classify_mezzo_tracks(track_id_stack, masks, gfp, bg_means, split_events)

    # ----- MORPH (with mezzo confirmation) -----
    morph_tracks = track_morph_peaks(masks, gfp_aligned=gfp, bg_means=bg_means)

    # ----- COORDINATION -----
    coord_pairs = match_coordination(mezzo_tracks, morph_tracks)

    # ----- Phenotype categories at peak window and endstate -----
    peak_frames = [t for t in range(T)
                   if PEAK_WINDOW_HPF[0] <= (HPF_START + t * HPF_INTERVAL) <= PEAK_WINDOW_HPF[1]]
    endstate_frames = list(range(max(0, T - 3), T))

    def items_in(window, source, predicate=lambda x: True):
        out = []
        for tid, item in source.items():
            if not predicate(item):
                continue
            if set(item["presence_frames"]) & set(window):
                out.append(item)
        return out

    mz_poles_peak  = items_in(peak_frames, mezzo_tracks, lambda x: x["cluster_type"] == "pole")
    mz_poles_end   = items_in(endstate_frames, mezzo_tracks, lambda x: x["cluster_type"] == "pole")
    mz_ocs_peak    = items_in(peak_frames, mezzo_tracks, lambda x: x["cluster_type"] == "organising_centre")
    mz_ocs_end     = items_in(endstate_frames, mezzo_tracks, lambda x: x["cluster_type"] == "organising_centre")
    mo_poles_peak  = items_in(peak_frames, morph_tracks, lambda x: x["is_pole"])
    mo_poles_end   = items_in(endstate_frames, morph_tracks, lambda x: x["is_pole"])

    # coordinated counts at each window
    def n_coord_in(window):
        n = 0
        for c in coord_pairs:
            mo = morph_tracks[c["morph_track"]]
            mz = mezzo_tracks[c["mezzo_track"]]
            if (set(mo["presence_frames"]) & set(window)) and (set(mz["presence_frames"]) & set(window)):
                n += 1
        return n
    n_coord_peak = n_coord_in(peak_frames)
    n_coord_end = n_coord_in(endstate_frames)

    pheno_peak = phenotype_category(len(mo_poles_peak), len(mz_poles_peak), n_coord_peak, len(mz_ocs_peak))
    pheno_end  = phenotype_category(len(mo_poles_end),  len(mz_poles_end),  n_coord_end,  len(mz_ocs_end))

    # Disintegration override
    if disintegration_frame is not None and disintegration_frame >= 0 and disintegration_frame < int(PEAK_WINDOW_HPF[1] - HPF_START) / HPF_INTERVAL:
        pheno_peak = "disintegrated_early"
        pheno_end = "disintegrated_early"

    pole_tracks_sorted = sorted([t for t in mezzo_tracks.values() if t["cluster_type"] == "pole"],
                                 key=lambda d: -d["peak_area_px"])
    primary = pole_tracks_sorted[0] if pole_tracks_sorted else None
    secondary = pole_tracks_sorted[1] if len(pole_tracks_sorted) > 1 else None

    morph_pole_sorted = sorted([t for t in morph_tracks.values() if t["is_pole"]],
                                key=lambda d: -d["peak_radial_dist"])

    summary = {
        "condition": cond,
        "pescoid": pid,
        "T_analysed": T,
        "global_gfp_threshold_bgsub": global_thr,
        # Mezzo counts
        "n_mezzo_poles_peak": len(mz_poles_peak),
        "n_mezzo_poles_endstate": len(mz_poles_end),
        "n_organising_centres_peak": len(mz_ocs_peak),
        "n_organising_centres_endstate": len(mz_ocs_end),
        "n_transient_clusters": sum(1 for t in mezzo_tracks.values() if t["cluster_type"] == "transient"),
        "n_fusion_events": len(fusion_events),
        "n_split_events": len(split_events),
        # Morph counts
        "n_morph_poles_peak": len(mo_poles_peak),
        "n_morph_poles_endstate": len(mo_poles_end),
        # Coordination
        "n_coordinated_poles_peak": n_coord_peak,
        "n_coordinated_poles_endstate": n_coord_end,
        "coordination_score_peak":
            n_coord_peak / max(len(mo_poles_peak), len(mz_poles_peak)) if max(len(mo_poles_peak), len(mz_poles_peak)) > 0 else 0.0,
        "coordination_score_endstate":
            n_coord_end / max(len(mo_poles_end), len(mz_poles_end)) if max(len(mo_poles_end), len(mz_poles_end)) > 0 else 0.0,
        "phenotype_peak": pheno_peak,
        "phenotype_endstate": pheno_end,
        # Emergence timing
        "first_mezzo_pole_emergence_hpf":
            HPF_START + min((t["first_frame"] for t in pole_tracks_sorted), default=np.nan) * HPF_INTERVAL
            if pole_tracks_sorted else np.nan,
        "second_mezzo_pole_emergence_hpf":
            HPF_START + sorted(t["first_frame"] for t in pole_tracks_sorted)[1] * HPF_INTERVAL
            if len(pole_tracks_sorted) >= 2 else np.nan,
        "first_morph_pole_emergence_hpf":
            HPF_START + min((t["first_frame"] for t in morph_pole_sorted), default=np.nan) * HPF_INTERVAL
            if morph_pole_sorted else np.nan,
        # Primary / secondary mezzo pole metrics
        "primary_mezzo_area_peak":      primary["peak_area_px"] if primary else 0,
        "primary_mezzo_intensity_peak": primary["peak_intensity_bgsub"] if primary else 0,
        "primary_mezzo_offset_norm":    primary["peak_offset_norm"] if primary else 0,
        "primary_mezzo_angle_deg":      primary["peak_angular_pos_deg"] if primary else 0,
        "secondary_mezzo_area_peak":      secondary["peak_area_px"] if secondary else 0,
        "secondary_mezzo_intensity_peak": secondary["peak_intensity_bgsub"] if secondary else 0,
        "secondary_mezzo_offset_norm":    secondary["peak_offset_norm"] if secondary else 0,
        "secondary_mezzo_angle_deg":      secondary["peak_angular_pos_deg"] if secondary else 0,
        "mezzo_pole_areas_all":           [t["peak_area_px"] for t in pole_tracks_sorted],
        "mezzo_pole_intensities_all":     [t["peak_intensity_bgsub"] for t in pole_tracks_sorted],
        "morph_pole_protrusions_all":     [t["peak_protrusion_frac"] for t in morph_pole_sorted],
    }

    if save_outputs:
        pdir = PER / cond / pid
        pdir.mkdir(parents=True, exist_ok=True)

        # Per-track CSVs
        pd.DataFrame([{k: v for k, v in t.items() if k != "presence_frames"}
                      for t in mezzo_tracks.values()]).to_csv(
            str(pdir / "mezzo_tracks.csv"), index=False
        )
        pd.DataFrame([{k: v for k, v in t.items() if k != "presence_frames"}
                      for t in morph_tracks.values()]).to_csv(
            str(pdir / "morph_tracks.csv"), index=False
        )
        pd.DataFrame(coord_pairs).to_csv(str(pdir / "coordination_pairs.csv"), index=False)
        pd.DataFrame(fusion_events).to_csv(str(pdir / "fusion_events.csv"), index=False)
        pd.DataFrame(split_events).to_csv(str(pdir / "split_events.csv"), index=False)

        with open(str(pdir / "summary.json"), "w") as f:
            json.dump({k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in summary.items()},
                      f, indent=2, default=float)
        tifffile.imwrite(str(pdir / "mezzo_tracks.tif"), track_id_stack.astype(np.uint16))

        save_tracking_montage(bf, gfp, bg_means, masks, track_id_stack,
                               mezzo_tracks, morph_tracks,
                               fusion_events, split_events, coord_pairs, summary,
                               pdir / f"{pid}_tracking_montage.png", cond, pid,
                               global_thr=global_thr)

    return {
        "summary": summary, "mezzo_tracks": mezzo_tracks, "morph_tracks": morph_tracks,
        "fusion_events": fusion_events, "split_events": split_events,
        "coord_pairs": coord_pairs,
        "track_id_stack": track_id_stack, "labeled_stack": labeled_stack,
        "bf": bf, "gfp": gfp, "masks": masks,
    }


# ===========================================================================
# Montage with cluster overlay + morph peaks + fusion/split arrows
# ===========================================================================
def _composite_bf_gfp(bf_frame, gfp_frame, bg_mean, global_thr):
    """
    Return an RGB image where:
      - BF is shown in grayscale (slightly attenuated to give green room)
      - GFP-positive pixels (bg-subtracted) are added as a green overlay
        scaled by intensity relative to the global threshold.
    """
    bf_n = bf_frame.astype(np.float32)
    bf_n = (bf_n - bf_n.min()) / (bf_n.max() - bf_n.min() + 1e-9)
    rgb = np.stack([bf_n, bf_n, bf_n], axis=-1) * 0.85   # gray base

    gfp_bg = (gfp_frame.astype(np.float32) - bg_mean).clip(min=0)
    # Normalise GFP to [0,1] relative to ~2x global threshold (so things ~thr are mid-bright)
    scale = max(2.0 * global_thr, 1.0)
    g_norm = np.clip(gfp_bg / scale, 0, 1)

    # Add green where there's signal
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.85 * g_norm, 0, 1)
    return rgb


def save_tracking_montage(bf, gfp, bg_means, masks, track_id_stack,
                          mezzo_tracks, morph_tracks,
                          fusion_events, split_events, coord_pairs,
                          summary, out_path, cond, pid, global_thr=1.0):
    T = bf.shape[0]
    n_cols = 6
    n_rows = int(np.ceil(T / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    fusion_by_frame = defaultdict(list)
    for ev in fusion_events:
        fusion_by_frame[ev["frame"]].append(ev)
    split_by_frame = defaultdict(list)
    for ev in split_events:
        split_by_frame[ev["frame"]].append(ev)

    # Pre-compute morph peak positions per frame
    morph_per_frame = defaultdict(list)
    for tid, t in morph_tracks.items():
        for f, pk in zip(t["presence_frames"], [None] * len(t["presence_frames"])):
            # presence_frames stored; we need original peak data — re-detect at this frame
            pass
    # Easier: collect per-frame morph_track_id at each peak position
    # For drawing, just mark the per-frame morph track summary's peak position
    # That's slightly off; instead, re-detect peaks per frame for drawing
    for t in range(T):
        peaks = detect_morph_peaks_in_frame(masks[t])
        for pk in peaks:
            # Find which morph_track contains this peak (nearest)
            best, best_d = None, float("inf")
            for tid, tr in morph_tracks.items():
                if t not in tr["presence_frames"]:
                    continue
                d = float(np.hypot(pk["y"] - tr["peak_y"], pk["x"] - tr["peak_x"]))
                if d < best_d:
                    best_d, best = d, tid
            morph_per_frame[t].append({**pk, "track_id": best})

    for t in range(T):
        ax = axes[t]
        # Composite: BF gray + GFP green overlay
        rgb = _composite_bf_gfp(bf[t], gfp[t], bg_means[t], global_thr)
        ax.imshow(rgb)
        if masks[t].any():
            for c in measure.find_contours(masks[t].astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], "w-", lw=0.6, alpha=0.6)
        track_frame = track_id_stack[t]
        if track_frame.max() > 0:
            for tid in np.unique(track_frame):
                if tid == 0:
                    continue
                cluster_mask = (track_frame == tid)
                if not cluster_mask.any():
                    continue
                ctype = mezzo_tracks.get(int(tid), {}).get("cluster_type", "transient")
                edge = "lime" if ctype == "pole" else ("yellow" if ctype == "organising_centre" else "magenta")
                for c in measure.find_contours(cluster_mask.astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=edge, lw=1.2)
                ys, xs = np.where(cluster_mask)
                ax.text(xs.mean(), ys.mean(), str(int(tid)), color=edge,
                        fontsize=7, fontweight="bold", ha="center",
                        bbox=dict(facecolor="black", alpha=0.5, pad=0.5))
        # Morph poles as triangle markers (orange = morph pole, gray = transient)
        for pk in morph_per_frame.get(t, []):
            tid = pk.get("track_id")
            is_pole = (tid is not None) and morph_tracks.get(tid, {}).get("is_pole", False)
            color = "orange" if is_pole else "lightgray"
            ax.scatter([pk["x"]], [pk["y"]], marker="^", s=120, c=color,
                       edgecolors="black", lw=1.2, zorder=4)
        # Fusion / split arrows
        for ev in fusion_by_frame.get(t, []):
            ax.annotate(f"F→{ev['child_track']}", xy=(ev["position_x"], ev["position_y"]),
                         xytext=(ev["position_x"] + 25, ev["position_y"] - 25),
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                         color="red", fontsize=7, fontweight="bold")
        for ev in split_by_frame.get(t, []):
            ax.annotate(f"S←{ev['parent_track']}", xy=(ev["position_x"], ev["position_y"]),
                         xytext=(ev["position_x"] + 25, ev["position_y"] - 25),
                         arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2),
                         color="cyan", fontsize=7, fontweight="bold")
        ax.set_title(f"t={t}  hpf={HPF_START+t*HPF_INTERVAL:.1f}", fontsize=9)
        ax.axis("off")
    for t in range(T, len(axes)):
        axes[t].axis("off")

    handles = [
        Line2D([0], [0], color=(0, 0.85, 0), lw=4,
               label="GFP signal (green overlay on BF)"),
        Line2D([0], [0], color="lime", lw=2, label="Mezzo pole contour"),
        Line2D([0], [0], color="yellow", lw=2, label="Organising centre"),
        Line2D([0], [0], color="magenta", lw=2, label="Transient cluster"),
        Line2D([0], [0], marker="^", color="orange", lw=0, markersize=10,
               markeredgecolor="black", label="Morph pole (mezzo-confirmed)"),
        Line2D([0], [0], marker="^", color="lightgray", lw=0, markersize=10,
               markeredgecolor="black", label="BF curvature peak (passive)"),
        Line2D([0], [0], color="red", lw=2, label="Fusion"),
        Line2D([0], [0], color="cyan", lw=2, label="Split"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9, frameon=True)

    plt.suptitle(
        f"{cond} / {pid}   —   peak: {summary['phenotype_peak']}   "
        f"|   endstate: {summary['phenotype_endstate']}   "
        f"|   mezzo poles peak/end: {summary['n_mezzo_poles_peak']}/{summary['n_mezzo_poles_endstate']}   "
        f"morph poles peak/end: {summary['n_morph_poles_peak']}/{summary['n_morph_poles_endstate']}   "
        f"coord peak/end: {summary['n_coordinated_poles_peak']}/{summary['n_coordinated_poles_endstate']}   "
        f"OCs: {summary['n_organising_centres_peak']}   "
        f"transient: {summary['n_transient_clusters']}   "
        f"F: {summary['n_fusion_events']}   S: {summary['n_split_events']}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Phenotype-sorted folder building
# ===========================================================================
def build_phenotype_folders(summary_rows):
    for cat in PHENO_CATS:
        catdir = PHENO / cat
        if catdir.exists():
            shutil.rmtree(str(catdir))
        catdir.mkdir(parents=True, exist_ok=True)
    for row in summary_rows:
        cond = row["condition"]; pid = row["pescoid"]
        src = PER / cond / pid / f"{pid}_tracking_montage.png"
        if not src.exists():
            continue
        for label, cat in [("peak", row["phenotype_peak"]), ("end", row["phenotype_endstate"])]:
            catdir = PHENO / cat
            catdir.mkdir(parents=True, exist_ok=True)
            dst = catdir / f"{cond}_{pid}__{label}.png"
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                print(f"  copy failed: {src} -> {dst}: {e}")


# ===========================================================================
# Main entry
# ===========================================================================
def run(pescoid_filter=None, include_embryos=False):
    qc = load_phase1_metrics()
    qc_pass = qc[qc["qc_pass"]].copy()
    print(f"\nPhase 1 QC-pass pescoids: {len(qc_pass)}")
    global_thr = compute_global_threshold(qc)
    if not include_embryos:
        before = len(qc_pass)
        qc_pass = qc_pass[qc_pass["condition"].isin(PHENOTYPE_CONDITIONS)]
        print(f"  Filtered to pescoids only (P_ctrl + P_Activin): {len(qc_pass)} of {before}")
    if pescoid_filter:
        qc_pass = qc_pass[qc_pass["pescoid"].isin(pescoid_filter)]
        print(f"  Filtered to: {pescoid_filter}")
    print(f"\nProcessing {len(qc_pass)} pescoids...\n")
    summary_rows = []
    for _, row in qc_pass.iterrows():
        cond = row["condition"]; pid = row["pescoid"]
        dis = row.get("disintegration_frame", -1) if "disintegration_frame" in row else -1
        print(f"  {cond}/{pid}: ", end="")
        try:
            out = process_pescoid(cond, pid, global_thr, save_outputs=True,
                                   disintegration_frame=dis)
            if out is None:
                print("SKIP (no aligned data)")
                continue
            s = out["summary"]
            print(
                f"mz_poles peak/end={s['n_mezzo_poles_peak']}/{s['n_mezzo_poles_endstate']}, "
                f"mo_poles peak/end={s['n_morph_poles_peak']}/{s['n_morph_poles_endstate']}, "
                f"coord peak={s['n_coordinated_poles_peak']}, OCs={s['n_organising_centres_peak']}, "
                f"F={s['n_fusion_events']}, S={s['n_split_events']} -> "
                f"peak={s['phenotype_peak']}, end={s['phenotype_endstate']}"
            )
            summary_rows.append(s)
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback; traceback.print_exc()

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(str(OUT / "phenotype_summary.csv"), index=False)
        try:
            with pd.ExcelWriter(str(OUT / "phenotype_summary.xlsx"), engine="xlsxwriter") as w:
                df.to_excel(w, sheet_name="all", index=False)
                for c in CONDITIONS:
                    sub = df[df["condition"] == c]
                    if not sub.empty:
                        sub.to_excel(w, sheet_name=c[:31], index=False)
        except Exception as e:
            print(f"Excel write failed: {e}")
        build_phenotype_folders(summary_rows)
        cross_peak = df.groupby(["condition", "phenotype_peak"]).size().unstack(fill_value=0)
        cross_end = df.groupby(["condition", "phenotype_endstate"]).size().unstack(fill_value=0)
        print("\n=== Phenotype PEAK counts per condition ===")
        print(cross_peak.to_string())
        print("\n=== Phenotype ENDSTATE counts per condition ===")
        print(cross_end.to_string())
        cross_peak.to_csv(str(OUT / "phenotype_peak_counts.csv"))
        cross_end.to_csv(str(OUT / "phenotype_endstate_counts.csv"))
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true",
                   help="Run on G046 + G021 only for sanity check")
    p.add_argument("--all", action="store_true",
                   help="Run on all QC-pass pescoids")
    p.add_argument("--pids", nargs="*",
                   help="Manual list of pescoid IDs to run on")
    args = p.parse_args()
    if args.sanity:
        run(pescoid_filter=["G046", "G021"])
    elif args.all:
        run()
    elif args.pids:
        run(pescoid_filter=args.pids)
    else:
        p.print_help()
