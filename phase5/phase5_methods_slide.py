"""
Methods + framing figure for the kymograph slide deck.

One PNG with:
  (top) annotated registration / major-axis fit visual for ONE pescoid
        showing BF + mask contour + ellipse major axis + AP=0 / AP=1 labels
  (bottom) a text block describing the 3 normalisations + AP convention +
        pole-source explanation (pre-empts "why does the kymograph agree/
        disagree with the Phase 2 pole count").
"""
from pathlib import Path
import numpy as np
import tifffile
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_kymograph_ap import (per_frame_geometry, HPF_START, HPF_INTERVAL,
                                  PEAK_ELONGATION_HPF, EXAMPLES)

PHASE1 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase1")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_kymograph\plots"
           r"\00_methods_and_framing.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Use G047 (coord_bipolar) as the demo - has a clear elongated shape with two poles
DEMO_COND, DEMO_PID = "P_Activin_3-5hpf", "G047"
DEMO_T = None   # filled below with that pescoid's peak-AR frame


def main():
    pdir = PHASE1 / "per_pescoid" / DEMO_COND / DEMO_PID
    bf = tifffile.imread(str(pdir / "bf_aligned.tif"))
    gfp = tifffile.imread(str(pdir / "gfp_aligned.tif"))
    mask = tifffile.imread(str(pdir / "mask_aligned.tif")) > 0
    T = bf.shape[0]
    # pick that pescoid's peak-AR frame
    ar = []
    for t in range(T):
        g = per_frame_geometry(mask[t])
        ar.append(g["ar"] if g else np.nan)
    t_demo = int(np.nanargmax(ar))
    g = per_frame_geometry(mask[t_demo])

    # Build figure: top-left = annotated BF+mask+axis; top-right = AP projection schematic
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], width_ratios=[1, 1.05],
                          hspace=0.25, wspace=0.18)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_proj = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    # Annotated registration + ellipse fit
    ax_img.imshow(bf[t_demo], cmap="gray")
    for c in measure.find_contours(mask[t_demo].astype(float), 0.5):
        ax_img.plot(c[:, 1], c[:, 0], "r-", lw=1.2)
    # major axis endpoints from s_min/s_max along orientation
    cos_o, sin_o = np.cos(g["orientation"]), np.sin(g["orientation"])
    y_post = g["cy"] + g["s_max"] * cos_o
    x_post = g["cx"] - g["s_max"] * sin_o
    y_ant = g["cy"] + g["s_min"] * cos_o
    x_ant = g["cx"] - g["s_min"] * sin_o
    ax_img.plot([x_ant, x_post], [y_ant, y_post], "lime", lw=2.4)
    ax_img.plot(g["cx"], g["cy"], "y+", markersize=18, mew=3)
    # AP=0 / AP=1 endpoints
    ax_img.annotate("AP = 0\n(anterior)", xy=(x_ant, y_ant),
                    xytext=(x_ant - 60, y_ant - 30),
                    fontsize=11, color="lime", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
    ax_img.annotate("AP = 1\n(posterior\n= peak-mezzo end\nby convention)",
                    xy=(x_post, y_post), xytext=(x_post + 30, y_post + 20),
                    fontsize=11, color="lime", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
    ax_img.text(g["cx"] + 15, g["cy"] + 15, "centroid",
                fontsize=10, color="yellow")
    ax_img.set_title(f"Per-frame ellipse fit (skimage.measure.regionprops)\n"
                     f"{DEMO_COND} / {DEMO_PID} at peak-AR (t={t_demo}, "
                     f"{HPF_START + t_demo * HPF_INTERVAL:.1f} hpf)",
                     fontsize=11, fontweight="bold")
    ax_img.set_xlabel(f"AR = major / minor = {g['major']:.0f} / {g['minor']:.0f} "
                      f"= {g['ar']:.2f}")
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # AP projection schematic - show GFP + projection arrows perpendicular to axis
    ax_proj.imshow(gfp[t_demo], cmap="magma")
    ax_proj.plot([x_ant, x_post], [y_ant, y_post], "lime", lw=2.0)
    # draw a few perpendicular projection lines
    n_demo = 6
    for f in np.linspace(0.15, 0.85, n_demo):
        # point along the axis
        y_p = y_ant + f * (y_post - y_ant)
        x_p = x_ant + f * (x_post - x_ant)
        # perpendicular direction (rotate axis dir by 90 deg)
        perp_y, perp_x = -sin_o, -cos_o
        L = max(g["minor"], 60) / 2
        ax_proj.plot([x_p - L * perp_x, x_p + L * perp_x],
                     [y_p - L * perp_y, y_p + L * perp_y],
                     "cyan", lw=0.7, alpha=0.7)
    ax_proj.set_title("AP projection: at each position along the AP axis, the\n"
                      "mezzo+ pixel fraction across the perpendicular slice is binned",
                      fontsize=11, fontweight="bold")
    ax_proj.set_xticks([]); ax_proj.set_yticks([])

    # Text block: methods + framing
    text = (
        "**Methods & framing**  (mezzo+LynTom dataset, n=52 pescoids, "
        "P_ctrl + P_Activin 3-5 hpf)\n\n"
        "**1. Major axis (AP axis)** is the longer principal axis of an ellipse fit "
        "to each frame's BF mask (skimage.measure.regionprops; computed from the "
        "second central moments of the binary region). Aspect ratio = major / minor. "
        "Phase 1 stable-orientation alignment (Option B, peak-AR frame anchor) "
        "prevents axis flipping between frames; per-frame regionprops gives the "
        "instantaneous ellipse.\n\n"
        "**2. Three normalisations baked in (so pescoids are comparable):**\n"
        "   * Spatial: AP position = (s - s_min) / (s_max - s_min) per frame, where "
        "s is the projection of each mask pixel onto the major axis. AP = 0 is the "
        "most anterior, AP = 1 the most posterior tip of THAT frame's mask.\n"
        "   * Temporal: 12 hpf (peak elongation across the dataset) is the universal "
        "reference, marked on every kymograph as a cyan line. Each pescoid's "
        "per-frame peak-AR is also recorded (white dashed) for its individual clock.\n"
        "   * Intensity: signal = fraction of mask pixels in each AP slice that "
        "exceed the global mezzo+ threshold (Phase 2's p99 of bg-sub GFP in "
        "P_ctrl + E_ctrl early frames, = 188.17). Inherently in [0, 1] so absolute "
        "values are comparable across all pescoids without per-sample rescaling.\n\n"
        "**3. AP orientation convention:** per pescoid, the AP axis is flipped (if "
        "needed) so the end with the larger total mezzo signal at 12 hpf sits at "
        "AP = 1. Without this, monopolar pescoids' lone pole would scatter randomly "
        "between AP=0 and AP=1, averaging to a center peak. With it, all monopolar "
        "profiles align (peak at AP=1), and the population profile is interpretable.\n\n"
        "**4. Pole markers on kymographs:** lime dots are the SAME mezzo+ poles "
        "from Phase 2 (cluster_type == 'pole' in mezzo_tracks.csv), projected onto "
        "this AP coordinate. The kymograph does NOT do its own pole detection - it "
        "shows the mezzo+ pixel-fraction profile + the Phase 2 pole centroids "
        "overlaid. By construction, the kymograph and Phase 2 phenotype agree."
    )
    ax_text.text(0.0, 1.0, text, fontsize=11, va="top", ha="left",
                 family="DejaVu Sans", linespacing=1.5, wrap=True)

    plt.suptitle("AP-mezzo kymograph: methods, normalisations, and "
                 "consistency with Phase 2",
                 fontsize=14, fontweight="bold")
    plt.savefig(str(OUT), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
