"""
Phase 5a v2 - Re-segment H2A nuclei with StarDist + DoG preprocessing.

This recovers nuclei in pescoids that Cellpose missed (faint H2A) by:
  1. DoG enhancement (sigma_small=1, sigma_large=6) on the aligned H2A image
  2. StarDist 2D 'versatile_fluo' pretrained model on the DoG image
  3. Restrict labels to inside the BF mask
  4. Save nuclei_masks_stardist.tif (uint16 instance labels per frame)

After running, phase5b2_tracking_v2.py reads these instead of the original
Cellpose masks.

Inputs:
  Lyn_mezzo->mezzo_H2A_phase1/per_pescoid/<cond>/<pid>/{h2a,mask}_aligned.tif

Outputs:
  mezzo_H2A_phase5a/per_pescoid/<cond>/<pid>/nuclei_masks_stardist.tif

Usage:
    python phase5a_v2_segment.py --pids S21 S26 S36   # test on a few first
    python phase5a_v2_segment.py --all                # full 39 pescoids
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Silence noisy library imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from stardist.models import StarDist2D
from csbdeep.utils import normalize as cs_normalize
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
PHASE5A = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase5a")

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
HPF_START = 7.0
HPF_INTERVAL = 698.8316040039062 / 3600.0
ANALYSIS_MAX_HPF = 16.0

DOG_SIGMA_SMALL = 1.0
DOG_SIGMA_LARGE = 6.0
# Lower prob_thresh than StarDist's default (0.479) - empirically recovers
# 2x-5x more nuclei in faint-signal pescoids (S31: 20->131, S36: 11->99)
# without picking up background noise (detections stay within pescoid mask).
STARDIST_PROB_THRESH = 0.20


def norm01(x):
    lo, hi = np.percentile(x, (1, 99.5))
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def dog_h2a(img):
    f = img.astype(np.float32)
    d = ndi.gaussian_filter(f, sigma=DOG_SIGMA_SMALL) - ndi.gaussian_filter(f, sigma=DOG_SIGMA_LARGE)
    return norm01(d.clip(min=0))


def segment_one_pescoid(cond, pid, model):
    pdir = PHASE1 / "per_pescoid" / cond / pid
    if not (pdir / "h2a_aligned.tif").exists():
        return None
    h2a = tifffile.imread(str(pdir / "h2a_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0

    T = h2a.shape[0]
    cap = int(np.floor((ANALYSIS_MAX_HPF - HPF_START) / HPF_INTERVAL)) + 1
    T_eff = min(T, cap)

    nuclei_stack = np.zeros((T_eff, mask.shape[1], mask.shape[2]), dtype=np.uint16)
    per_frame_counts = []
    for t in range(T_eff):
        if not mask[t].any():
            per_frame_counts.append(0)
            continue
        dog = dog_h2a(h2a[t])
        labs, _ = model.predict_instances(dog, verbose=False,
                                           prob_thresh=STARDIST_PROB_THRESH)
        labs = labs * mask[t]  # restrict to BF mask
        # relabel sequentially so IDs are 1..N (max label may not equal count)
        unique = np.unique(labs)
        unique = unique[unique > 0]
        remap = np.zeros(int(labs.max()) + 1, dtype=np.int32) if labs.max() > 0 else None
        if remap is not None:
            for new_id, old_id in enumerate(unique, start=1):
                remap[int(old_id)] = new_id
            labs = remap[labs]
        nuclei_stack[t] = labs.astype(np.uint16)
        per_frame_counts.append(int(labs.max()))

    out_dir = PHASE5A / "per_pescoid" / cond / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_dir / "nuclei_masks_stardist.tif"), nuclei_stack)
    pd.DataFrame({
        "frame": list(range(T_eff)),
        "hpf": [HPF_START + t * HPF_INTERVAL for t in range(T_eff)],
        "n_nuclei_stardist": per_frame_counts,
    }).to_csv(str(out_dir / "stardist_nuclei_per_frame.csv"), index=False)
    return per_frame_counts


def main(pescoid_filter=None):
    summary = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    summary = summary[summary["condition"].isin(CONDITIONS)].copy()
    if pescoid_filter:
        summary = summary[summary["pescoid"].isin(pescoid_filter)]
    print(f"Pescoids to re-segment: {len(summary)}")
    print("Loading StarDist 2D_versatile_fluo...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    print("Model loaded.\n")

    all_counts = []
    for _, r in summary.iterrows():
        cond, pid = r["condition"], r["pescoid"]
        print(f"  {cond}/{pid}...", end=" ")
        try:
            counts = segment_one_pescoid(cond, pid, model)
            if counts is None:
                print("SKIP (no input)")
                continue
            peak_count = max(counts) if counts else 0
            n_frames_with_nuclei = sum(1 for c in counts if c > 0)
            all_counts.append({
                "condition": cond, "pescoid": pid,
                "phenotype_peak": r["phenotype_peak"],
                "peak_nuclei_count": peak_count,
                "n_frames_with_nuclei": n_frames_with_nuclei,
                "mean_nuclei_count": float(np.mean(counts)) if counts else 0.0,
            })
            print(f"peak={peak_count}, mean={np.mean(counts):.1f}, "
                  f"frames_with_nuclei={n_frames_with_nuclei}/{len(counts)}")
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_counts)
    out_csv = PHASE5A / "tables" / "stardist_nuclei_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)
    print(f"\nSummary saved to: {out_csv}")
    print("\n=== Per-condition summary ===")
    if not df.empty:
        print(df.groupby("condition")[["peak_nuclei_count", "mean_nuclei_count",
                                        "n_frames_with_nuclei"]].describe().round(1).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pids", nargs="*", help="pescoid IDs")
    p.add_argument("--all", action="store_true", help="all 39 P pescoids")
    args = p.parse_args()
    if args.pids:
        main(pescoid_filter=args.pids)
    elif args.all:
        main()
    else:
        p.print_help()
