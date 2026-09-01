"""
Re-classify Phase 2 phenotypes adding a 'diffuse_mezzo' category.

Definition:
  diffuse_mezzo = no poles + no OCs + no transient clusters,
                  BUT mean bg-subtracted GFP in mask (peak window) >= DIFFUSE_FLOOR
                  -> uniform basal mezzo activity without spatially clustered induction.

DIFFUSE_FLOOR is set at 2.5x typical background noise ≈ 50 (bg-sub intensity).

Outputs (in place, overwriting Phase 2 summary):
  Lyn_mezzo_phase2/phenotype_summary.csv  (updated)
  Lyn_mezzo_phase2/phenotype_summary.xlsx (updated)
  Lyn_mezzo_phase2/phenotype_peak_counts.csv
  Lyn_mezzo_phase2/phenotype_endstate_counts.csv

  Also re-sorts phenotype_categorised/<cat>/ folders.
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import tifffile

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase1")
PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_H2A_phase2")
PHENO = PHASE2 / "phenotype_categorised"

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
HPF_START = 7.0
HPF_INTERVAL = 698.8316040039062 / 3600.0   # ~0.1941 hpf
PEAK_WINDOW_HPF = (7.0, 16.0)

# Scaled from old DIFFUSE_FLOOR=50 (with pole threshold 188) to the new pole
# threshold 789: 50 * (789/188) = 210 -> rounded to 200.
DIFFUSE_FLOOR = 200.0  # mean bg-sub GFP threshold for diffuse_mezzo

PHENO_CATS = [
    "coordinated_bipolar", "coordinated_monopolar",
    "morph_bipolar_only", "mezzo_bipolar_only",
    "multipolar", "disorganised_multipolar",
    "oc_only", "diffuse_mezzo", "no_induction", "disintegrated_early",
]


def compute_mean_bgsub_peak(cond, pid):
    """Mean bg-subtracted GFP intensity in BF mask, averaged over peak-window frames."""
    pdir = PHASE1 / "per_pescoid" / cond / pid
    gfp_p = pdir / "gfp_aligned.tif"
    mask_p = pdir / "mask_aligned.tif"
    if not (gfp_p.exists() and mask_p.exists()):
        return np.nan, np.nan
    gfp = tifffile.imread(str(gfp_p))
    masks = tifffile.imread(str(mask_p)) > 0
    T = gfp.shape[0]
    peak_frames = [
        t for t in range(T)
        if PEAK_WINDOW_HPF[0] <= (HPF_START + t * HPF_INTERVAL) <= PEAK_WINDOW_HPF[1]
    ]
    if not peak_frames:
        return np.nan, np.nan
    peak_vals_mean = []
    peak_vals_p99 = []
    for t in peak_frames:
        if not masks[t].any():
            continue
        bg = float(gfp[t][~masks[t]].mean()) if (~masks[t]).any() else 0.0
        in_vals = (gfp[t][masks[t]] - bg).clip(min=0)
        peak_vals_mean.append(float(in_vals.mean()))
        peak_vals_p99.append(float(np.percentile(in_vals, 99)))
    if not peak_vals_mean:
        return np.nan, np.nan
    return float(np.mean(peak_vals_mean)), float(np.mean(peak_vals_p99))


def reclassify(row):
    """Return updated (phenotype_peak, phenotype_endstate)."""
    new_peak = row["phenotype_peak"]
    new_end = row["phenotype_endstate"]

    # Only upgrade no_induction -> diffuse_mezzo when mean bg-sub >= floor
    if (row.get("phenotype_peak") == "no_induction"
            and row.get("mean_gfp_bgsub_peak", 0) >= DIFFUSE_FLOOR):
        new_peak = "diffuse_mezzo"
    if (row.get("phenotype_endstate") == "no_induction"
            and row.get("mean_gfp_bgsub_peak", 0) >= DIFFUSE_FLOOR):
        # Use peak-window mean as proxy for endstate too — if peak shows diffuse
        # signal and endstate had no clusters, it's diffuse.
        new_end = "diffuse_mezzo"
    return pd.Series({"phenotype_peak": new_peak, "phenotype_endstate": new_end})


def rebuild_phenotype_folders(df):
    # Wipe old categories
    for cat in PHENO_CATS:
        catdir = PHENO / cat
        if catdir.exists():
            shutil.rmtree(str(catdir))
        catdir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        cond, pid = row["condition"], row["pescoid"]
        src = PHASE2 / "per_pescoid" / cond / pid / f"{pid}_tracking_montage.png"
        if not src.exists():
            continue
        for tag, cat in [("peak", row["phenotype_peak"]), ("end", row["phenotype_endstate"])]:
            catdir = PHENO / cat
            catdir.mkdir(parents=True, exist_ok=True)
            dst = catdir / f"{cond}_{pid}__{tag}.png"
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                print(f"  copy failed: {src} -> {dst}: {e}")


def main():
    print("=" * 60)
    print("  Re-classifying Phase 2 phenotypes with 'diffuse_mezzo' category")
    print("=" * 60)

    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    df = df[df["condition"].isin(CONDITIONS)].copy()

    print(f"\nComputing mean bg-subtracted GFP per pescoid (peak window {PEAK_WINDOW_HPF[0]}-{PEAK_WINDOW_HPF[1]} hpf)...")
    means = []
    p99s = []
    for _, row in df.iterrows():
        m, p99 = compute_mean_bgsub_peak(row["condition"], row["pescoid"])
        means.append(m)
        p99s.append(p99)
    df["mean_gfp_bgsub_peak"] = means
    df["p99_gfp_bgsub_peak"] = p99s

    # Apply re-classification
    df_new = df.copy()
    new_cats = df.apply(reclassify, axis=1)
    df_new["phenotype_peak"] = new_cats["phenotype_peak"]
    df_new["phenotype_endstate"] = new_cats["phenotype_endstate"]

    # Save updated summary
    df_new.to_csv(str(PHASE2 / "phenotype_summary.csv"), index=False)
    try:
        with pd.ExcelWriter(str(PHASE2 / "phenotype_summary.xlsx"), engine="xlsxwriter") as w:
            df_new.to_excel(w, sheet_name="all", index=False)
            for c in CONDITIONS:
                sub = df_new[df_new["condition"] == c]
                if not sub.empty:
                    sub.to_excel(w, sheet_name=c[:31], index=False)
    except Exception as e:
        print(f"Excel write failed: {e}")

    # Counts
    cross_peak = pd.crosstab(df_new["condition"], df_new["phenotype_peak"])
    cross_end = pd.crosstab(df_new["condition"], df_new["phenotype_endstate"])
    cross_peak.to_csv(str(PHASE2 / "phenotype_peak_counts.csv"))
    cross_end.to_csv(str(PHASE2 / "phenotype_endstate_counts.csv"))

    print(f"\nDIFFUSE_FLOOR (mean bg-sub) = {DIFFUSE_FLOOR}")
    n_reclassified_peak = (df_new["phenotype_peak"] == "diffuse_mezzo").sum()
    n_reclassified_end = (df_new["phenotype_endstate"] == "diffuse_mezzo").sum()
    print(f"Pescoids reclassified as 'diffuse_mezzo':")
    print(f"  at peak:     {n_reclassified_peak}")
    print(f"  at endstate: {n_reclassified_end}")

    print("\n=== PEAK counts ===")
    print(cross_peak.to_string())
    print("\n=== ENDSTATE counts ===")
    print(cross_end.to_string())

    print("\n=== Reclassified pescoids (peak) ===")
    reclass = df_new[df_new["phenotype_peak"] == "diffuse_mezzo"]
    print(reclass[["condition", "pescoid", "mean_gfp_bgsub_peak", "p99_gfp_bgsub_peak"]].to_string(index=False))

    # Re-sort phenotype folders
    print("\nRebuilding phenotype_categorised/ folders...")
    rebuild_phenotype_folders(df_new)
    print(f"Done.")


if __name__ == "__main__":
    main()
