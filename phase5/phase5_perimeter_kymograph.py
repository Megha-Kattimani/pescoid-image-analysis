"""
Phase 5 - perimeter kymograph for the 5 example pescoids, paired with the
AP-mezzo kymograph for the poster.

Perimeter kymograph for each frame:
  1. Compute distance from centroid to boundary as a function of angle theta
     (angle measured from the major-axis = AP direction)
  2. Sample 360 equally spaced theta values
  3. Stack frames vertically -> kymograph with theta on x, time on y, signal =
     delta-radius from per-pescoid mean radius (so bulges are red, contractions blue)

Then a 5-column paired figure:
  top row  = perimeter kymograph  (mechanics: where the boundary bulges/contracts)
  bottom row = AP-mezzo kymograph  (fate:   where mezzo+ pixels concentrate along AP)

Same 5 pescoids as Phase 5 kymograph:
  G022 P_ctrl coord_mono | G046 P_Activin coord_mono | G047 coord_bipolar |
  G050 mezzo_bipolar     | G060 multipolar
"""
from pathlib import Path
import json
import numpy as np
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
KYM5 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_kymograph")
OUT_PLOTS = KYM5 / "plots"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
PER_PESCOID = KYM5 / "per_pescoid"

HPF_START = 6.0
HPF_INTERVAL = 0.4
PEAK_ELONGATION_HPF = 12.0
N_THETA = 360

EXAMPLES = [
    ("P_ctrl",           "G022", "coord_mono"),
    ("P_Activin_3-5hpf", "G046", "coord_mono"),
    ("P_Activin_3-5hpf", "G047", "coord_bipolar"),
    ("P_Activin_3-5hpf", "G050", "mezzo_bipolar"),
    ("P_Activin_3-5hpf", "G060", "multipolar"),
]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 11})


def perimeter_radii(mask_frame, n_theta=N_THETA):
    """For one mask frame, sample radii r(theta) at n_theta equispaced angles
    measured from the major axis. Returns (r_array, ap_flipped_theta_at_zero).
    """
    if not mask_frame.any():
        return np.full(n_theta, np.nan), None
    props = measure.regionprops(mask_frame.astype(int))
    if not props:
        return np.full(n_theta, np.nan), None
    p = max(props, key=lambda r: r.area)
    cy, cx = float(p.centroid[0]), float(p.centroid[1])
    orientation = float(p.orientation)
    contours = measure.find_contours(mask_frame.astype(float), 0.5)
    if not contours:
        return np.full(n_theta, np.nan), None
    # use the longest contour as the boundary
    contour = max(contours, key=len)
    dy = contour[:, 0] - cy
    dx = contour[:, 1] - cx
    # angle measured from major axis (rotate by -orientation)
    # major axis direction: (cos orientation, -sin orientation)
    theta_raw = np.arctan2(dx * np.cos(orientation) + dy * np.sin(orientation),
                            dy * np.cos(orientation) - dx * np.sin(orientation))
    r = np.sqrt(dy ** 2 + dx ** 2)
    # wrap to [0, 2pi)
    theta = (theta_raw + 2 * np.pi) % (2 * np.pi)
    order = np.argsort(theta)
    theta_sorted = theta[order]
    r_sorted = r[order]
    # sample to n_theta equispaced
    theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r_grid = np.interp(theta_grid, theta_sorted, r_sorted,
                        period=2 * np.pi)
    return r_grid, theta_grid


def build_perimeter_kymograph(cond, pid):
    """Build T x N_THETA delta-radius kymograph for this pescoid.
    Apply AP-flip convention (from meta JSON) so theta=0 is anterior."""
    p1 = PHASE1 / "per_pescoid" / cond / pid
    mask = tifffile.imread(str(p1 / "mask_aligned.tif")) > 0
    T = mask.shape[0]
    R = np.array([perimeter_radii(mask[t])[0] for t in range(T)])
    # delta-radius from per-pescoid mean radius (subtract per-pescoid mean of r)
    mean_r = np.nanmean(R)
    dR = R - mean_r
    # apply AP-flip if needed (from kymograph meta)
    meta_path = PER_PESCOID / cond / pid / f"{pid}_meta.json"
    flipped = False
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        flipped = bool(meta.get("ap_flipped", False))
    if flipped:
        # theta = 0 was at the AP=0 (anterior) end of original orientation.
        # Flipping AP means theta should reverse direction: theta -> -theta mod 2pi.
        # In array terms, reverse the theta axis around index 0:
        # f(theta) -> f(-theta) = f(2pi - theta).
        # For an N-array this is: out[i] = arr[(-i) mod N] = arr[N-i if i>0 else 0]
        dR = np.column_stack([dR[:, :1], dR[:, :0:-1]])
    return dR


def save_paired_panel(results_perim, results_apmezzo_paths):
    """5-column paired figure: top=perimeter kymograph, bottom=AP-mezzo kymograph."""
    n = len(EXAMPLES)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 10),
                              gridspec_kw={"height_ratios": [1.1, 1.1]})
    vlim = max(np.nanpercentile(np.abs(np.stack(list(results_perim.values()))),
                                  99), 1.0)
    apmezzo_vmax = None
    apmezzo_data = []
    for cond, pid, _ in EXAMPLES:
        npy = PER_PESCOID / cond / pid / f"{pid}_ap_profiles.npy"
        apmezzo_data.append(np.load(str(npy)))
    apmezzo_vmax = max(np.nanpercentile(np.stack([a for a in apmezzo_data]),
                                          98), 0.05)
    for ci, (cond, pid, label) in enumerate(EXAMPLES):
        # Top: perimeter
        dR = results_perim[(cond, pid)]
        T = dR.shape[0]
        hpf = HPF_START + np.arange(T) * HPF_INTERVAL
        ax = axes[0, ci]
        im = ax.imshow(dR, aspect="auto", origin="lower",
                       extent=[0, 360, hpf[0], hpf[-1] + HPF_INTERVAL],
                       cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       interpolation="nearest")
        ax.axhline(PEAK_ELONGATION_HPF, color="lime", lw=1.5)
        ax.axvline(180, color="k", lw=0.6, ls=":", alpha=0.4)
        ax.set_xlabel("perimeter angle from AP axis (deg)")
        flip_tag = ""  # already in AP frame
        ax.set_title(f"{COND_LABEL[cond]}\n{pid} ({label})",
                     fontsize=10, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("hpf")
        # Bottom: AP-mezzo
        ax = axes[1, ci]
        prof = apmezzo_data[ci]
        T = prof.shape[0]
        hpf = HPF_START + np.arange(T) * HPF_INTERVAL
        im_ap = ax.imshow(prof, aspect="auto", origin="lower",
                          extent=[0, 1, hpf[0], hpf[-1] + HPF_INTERVAL],
                          cmap="magma", vmin=0, vmax=apmezzo_vmax,
                          interpolation="nearest")
        ax.axhline(PEAK_ELONGATION_HPF, color="cyan", lw=1.5)
        ax.set_xlabel("AP position (normalised)")
        if ci == 0:
            ax.set_ylabel("hpf")
    fig.colorbar(im, ax=axes[0, -1], label="Δ radius (px)", shrink=0.85)
    fig.colorbar(im_ap, ax=axes[1, -1], label="mezzo+ pixel fraction",
                  shrink=0.85)
    plt.suptitle("Paired kymographs: boundary mechanics (top) vs AP-mezzo "
                 "fate signal (bottom)\nFor each pescoid: outward bulges (red) "
                 "and inward contractions (blue) of the boundary AND where "
                 "the mezzo+ poles concentrate along AP. "
                 "Lime/cyan line = 12 hpf peak elongation.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(OUT_PLOTS / "20_paired_perimeter_apmezzo.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  20_paired_perimeter_apmezzo.png")


def main():
    print("Building perimeter kymographs for 5 example pescoids...")
    results_perim = {}
    for cond, pid, label in EXAMPLES:
        try:
            dR = build_perimeter_kymograph(cond, pid)
            results_perim[(cond, pid)] = dR
            print(f"  {cond}/{pid}: ok, T={dR.shape[0]}")
        except Exception as e:
            print(f"  FAIL {cond}/{pid}: {e}")
    save_paired_panel(results_perim, None)


if __name__ == "__main__":
    main()
