"""Per-region cortical coherence over time for bi-/multipolar pescoids.

Reproduces the G047 worked-example breakdown (anterior vs primary pole vs
secondary pole vs body core) and applies it to any pescoid, so we can test
whether the pattern (persistent primary pole = low order; transient secondary
and bare anterior = higher) is consistent across pescoids.

Coherence is the exact structure-tensor measure from phase4_tension.py
(SIGMA_GRAD=5, SIGMA_TENSOR=10). Regions:
  PRIMARY pole   = largest-peak-area mezzo 'pole' track
  SECONDARY pole = next mezzo 'pole' track
  BODY CORE      = eroded interior of the mask, minus pole pixels
  ANTERIOR       = peripheral cortex wedge opposite the poles, mezzo-negative

Usage:  python other_analyses/per_pole_coherence_timecourse.py G047 G050 G054
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import morphology, measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
COND = "P_Activin_3-5hpf"
OUTDIR = Path(r"c:\Users\kattimani\Project\pescoid-image-analysis\docs\images")

HPF_START, HPF_INTERVAL = 6.0, 0.4
SIGMA_GRAD, SIGMA_TENSOR = 5, 10
GFP_NEG_FRAC = 0.20      # below this normalised GFP -> mezzo-negative
MIN_PX = 40              # minimum pixels for a region's coherence to count

COL = {"anterior": "#1f77b4", "primary": "#d62728",
       "secondary": "#ff7f0e", "body": "#7f7f7f"}


def norm_pct(img, lo=1, hi=99.5):
    a, b = np.percentile(img, (lo, hi))
    return np.clip((img - a) / (b - a), 0, 1) if b > a else np.zeros_like(img, float)


def structure_tensor_coherence(img):
    gy = ndi.gaussian_filter(img, SIGMA_GRAD, order=(1, 0))
    gx = ndi.gaussian_filter(img, SIGMA_GRAD, order=(0, 1))
    Jxx = ndi.gaussian_filter(gx * gx, SIGMA_TENSOR)
    Jyy = ndi.gaussian_filter(gy * gy, SIGMA_TENSOR)
    Jxy = ndi.gaussian_filter(gx * gy, SIGMA_TENSOR)
    tr = Jxx + Jyy
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    return np.where(tr > 0, disc / (tr + 1e-10), 0.0)   # (l1-l2)/(l1+l2)


def pole_ids_ordered(pid):
    df = pd.read_csv(str(PHASE2 / "per_pescoid" / COND / pid / "mezzo_tracks.csv"))
    poles = df[df["cluster_type"] == "pole"].sort_values("peak_area_px", ascending=False)
    ids = poles["track_id"].tolist()
    cents = {int(r.track_id): (r.peak_centroid_y, r.peak_centroid_x)
             for r in poles.itertuples()}
    primary = ids[0] if len(ids) >= 1 else None
    secondary = ids[1] if len(ids) >= 2 else None
    return primary, secondary, cents


def mean_in(coh, region):
    return float(coh[region].mean()) if region.sum() >= MIN_PX else np.nan


def analyse(pid):
    pdir = PHASE1 / "per_pescoid" / COND / pid
    lyn = tifffile.imread(str(pdir / "lyntom_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    tracks = tifffile.imread(str(PHASE2 / "per_pescoid" / COND / pid / "mezzo_tracks.tif"))
    primary, secondary, cents = pole_ids_ordered(pid)

    # pole direction from peak centroids (mean of pole centroids vs global centroid)
    allm = mask.any(0)
    gcy, gcx = ndi.center_of_mass(allm)
    if cents:
        pcy = np.mean([c[0] for c in cents.values()])
        pcx = np.mean([c[1] for c in cents.values()])
        pdir = np.array([pcy - gcy, pcx - gcx])
        pdir = pdir / (np.linalg.norm(pdir) + 1e-9)
    else:
        pdir = None

    T = mask.shape[0]
    rows = []
    for t in range(T):
        m = mask[t]
        if m.sum() < 50:
            continue
        coh = structure_tensor_coherence(norm_pct(lyn[t].astype(float)))
        gfp_n = norm_pct(gfp[t].astype(float))
        interior = morphology.binary_erosion(m, morphology.disk(20))
        cortex = m & ~morphology.binary_erosion(m, morphology.disk(15))

        prim = morphology.binary_dilation((tracks[t] == primary), morphology.disk(2)) & m if primary else np.zeros_like(m)
        sec = morphology.binary_dilation((tracks[t] == secondary), morphology.disk(2)) & m if secondary else np.zeros_like(m)
        poles = prim | sec
        body = interior & ~poles

        # anterior: peripheral band opposite the poles (far hemisphere), pole pixels removed
        ant = cortex & ~poles
        if pdir is not None and ant.any():
            cy, cx = ndi.center_of_mass(m)
            ys, xs = np.where(ant)
            d = np.stack([ys - cy, xs - cx], 1).astype(float)
            d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
            far = d @ pdir < -0.15
            keep = np.zeros_like(ant)
            keep[ys[far], xs[far]] = True
            ant = keep
        else:
            ant = np.zeros_like(m)

        rows.append({
            "hpf": HPF_START + t * HPF_INTERVAL,
            "anterior": mean_in(coh, ant),
            "primary": mean_in(coh, prim),
            "secondary": mean_in(coh, sec),
            "body": mean_in(coh, body),
        })
    return pd.DataFrame(rows), (primary, secondary)


def main(pids):
    results = {}
    for pid in pids:
        df, (p, s) = analyse(pid)
        results[pid] = df
        df.to_csv(str(OUTDIR / f"per_pole_coherence_{pid}.csv"), index=False)
        means = {k: round(df[k].mean(), 3) for k in ["anterior", "primary", "secondary", "body"]}
        print(f"{pid}: primary_id={p} secondary_id={s}  means={means}")

    n = len(pids)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), squeeze=False)
    for i, pid in enumerate(pids):
        ax = axes[0, i]
        df = results[pid]
        for key, lab in [("anterior", "ANTERIOR (bare cortex)"),
                         ("primary", "PRIMARY pole"),
                         ("secondary", "SECONDARY pole"),
                         ("body", "BODY CORE")]:
            if df[key].notna().any():
                ax.plot(df["hpf"], df[key], "o-", color=COL[key], lw=2, ms=4, label=lab)
        ax.set_title(f"{pid}: coherence by region over time", fontweight="bold", fontsize=12)
        ax.set_xlabel("hpf")
        ax.set_ylabel("cortical coherence")
        ax.set_ylim(0.4, 0.95)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = OUTDIR / ("per_pole_coherence_" + "_".join(pids) + ".png")
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    print("SAVED", out)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["G047", "G050", "G054"])
