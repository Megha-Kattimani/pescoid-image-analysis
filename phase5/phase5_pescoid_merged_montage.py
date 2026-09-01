"""
Phase 5 - merged-channel pescoid montages for the top of the poster.

Goal: show what is actually happening - the pescoid forming 0, 1, 2, or many
mezzo+ organisers - by merging BF (gray) + mezzo:GFP (green) + LynTom (magenta)
across a time-course.

Layout (one figure):
  4 rows (phenotype)  x  6 columns (timepoints ~ 6, 8, 10, 12, 14, 16 hpf)
  + phenotype label column on the left, hpf row label on top.

Selected representatives:
  P_ctrl   G030   coordinated_monopolar (weak, the ctrl baseline)
  P_Activin G046  coordinated_monopolar (one strong pole)
  P_Activin G047  coordinated_bipolar   (two clear poles)
  P_Activin G060  multipolar            (3+ poles)

Output: Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase5_kymograph\\plots\\30_pescoid_merged_montage.png

Channel display:
  BF      -> gray, brightness-stretched
  GFP     -> green   (mezzo:GFP, per-pescoid bg-subtract + max-norm)
  LynTom  -> magenta (per-pescoid bg-subtract + max-norm)
"""
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_kymograph\plots")
OUT.mkdir(parents=True, exist_ok=True)

HPF_START = 6.0
HPF_INTERVAL = 0.4

# Phenotype representatives in display order (top -> bottom)
EXAMPLES = [
    ("P_ctrl",           "G024", "no induction (ctrl)"),
    ("P_Activin_3-5hpf", "G046", "monopolar (Activin)"),
    ("P_Activin_3-5hpf", "G047", "bipolar (Activin)"),
    ("P_Activin_3-5hpf", "G060", "multipolar (Activin)"),
]

# Target hpf timepoints
TARGET_HPF = [6.4, 8.0, 10.0, 12.0, 14.0, 16.0]


def norm01(img, p_lo=2, p_hi=99, within_mask=None):
    if within_mask is not None and within_mask.any():
        vals = img[within_mask]
    else:
        vals = img
    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)


def merge_rgb(bf, gfp, lyntom, mask):
    """Build an RGB composite with BF gray, GFP green, LynTom magenta."""
    bf_n = norm01(bf, p_lo=1, p_hi=99.5, within_mask=mask)
    gfp_bg = float(gfp[~mask].mean()) if (~mask).any() else 0.0
    gfp_bgsub = np.clip(gfp.astype(np.float32) - gfp_bg, 0, None)
    gfp_n = norm01(gfp_bgsub, p_lo=50, p_hi=99.5, within_mask=mask)
    lyn_bg = float(lyntom[~mask].mean()) if (~mask).any() else 0.0
    lyn_bgsub = np.clip(lyntom.astype(np.float32) - lyn_bg, 0, None)
    lyn_n = norm01(lyn_bgsub, p_lo=10, p_hi=99.0, within_mask=mask)

    rgb = np.zeros(bf.shape + (3,), dtype=np.float32)
    # BF in gray everywhere (background)
    bf_weight = 0.55
    rgb[..., 0] = bf_n * bf_weight
    rgb[..., 1] = bf_n * bf_weight
    rgb[..., 2] = bf_n * bf_weight
    # GFP in green (only inside the mask)
    rgb[..., 1] = np.where(mask, np.maximum(rgb[..., 1], gfp_n), rgb[..., 1])
    # LynTom magenta = R + B
    rgb[..., 0] = np.where(mask, np.maximum(rgb[..., 0], lyn_n * 0.85),
                            rgb[..., 0])
    rgb[..., 2] = np.where(mask, np.maximum(rgb[..., 2], lyn_n * 0.85),
                            rgb[..., 2])
    return np.clip(rgb, 0, 1)


def crop_to_pescoid(img, mask_any, pad=20):
    """Crop to the union mask bounding box across the whole timecourse, +pad."""
    if not mask_any.any():
        return img
    ys, xs = np.where(mask_any)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[-3], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[-2], xs.max() + pad + 1)
    return img[..., y0:y1, x0:x1, :]


def closest_frame(target_hpf, T):
    idx = int(round((target_hpf - HPF_START) / HPF_INTERVAL))
    return max(0, min(T - 1, idx))


def build_pescoid_row(cond, pid):
    p = PHASE1 / "per_pescoid" / cond / pid
    bf = tifffile.imread(str(p / "bf_aligned.tif"))
    gfp = tifffile.imread(str(p / "gfp_aligned.tif"))
    lyn = tifffile.imread(str(p / "lyntom_aligned.tif"))
    mask = tifffile.imread(str(p / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    union = mask.any(axis=0)
    frames = [closest_frame(h, T) for h in TARGET_HPF]
    rgb_stack = []
    for t in frames:
        rgb = merge_rgb(bf[t], gfp[t], lyn[t], mask[t])
        rgb_stack.append(rgb)
    rgb_stack = np.stack(rgb_stack)        # (n_t, H, W, 3)
    # crop to pescoid union bounding box
    rgb_stack_c = crop_to_pescoid(rgb_stack, union, pad=15)
    return rgb_stack_c, frames


def main():
    print("Building merged-channel montage for 4 phenotype representatives...")
    rows = []
    for cond, pid, label in EXAMPLES:
        try:
            rgb_stack, frames = build_pescoid_row(cond, pid)
            rows.append((label, pid, rgb_stack, frames))
            print(f"  {pid} ({label}) ok, {rgb_stack.shape[1]}x"
                  f"{rgb_stack.shape[2]} px")
        except Exception as e:
            print(f"  FAIL {pid}: {e}")

    if not rows:
        return

    n_rows = len(rows)
    n_cols = len(TARGET_HPF)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows),
                              squeeze=False)
    for ri, (label, pid, rgb_stack, _) in enumerate(rows):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.imshow(rgb_stack[ci])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            if ri == 0:
                ax.set_title(f"{TARGET_HPF[ci]:.1f} hpf", fontsize=14,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{label}\n{pid}", fontsize=12,
                              fontweight="bold", rotation=0, ha="right",
                              va="center", labelpad=70)

    # legend at top
    handles = [
        plt.Line2D([0], [0], color="white", lw=8, label="brightfield (gray)"),
        plt.Line2D([0], [0], color="#00FF00", lw=8, label="mezzo:GFP (green)"),
        plt.Line2D([0], [0], color="#FF00FF", lw=8, label="LynTom membrane (magenta)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=14)
    plt.suptitle("How pescoids decide between one, two, or many organisers - "
                 "merged channels across imaging time",
                 fontsize=15, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(left=0.10)
    out = OUT / "30_pescoid_merged_montage.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
