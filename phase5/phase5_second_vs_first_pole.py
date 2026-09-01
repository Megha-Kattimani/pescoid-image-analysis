"""
First vs second pole in TIME (x) and peak SIZE (y) space, with the two poles of
the same pescoid joined by a faint line.

All TREATED pescoids from every analysed experiment are pooled (controls
excluded - they essentially never form a second pole):
  - Lyn_mezzo        : P_Activin_3-5hpf
  - mezzo_H2A        : P_Activin_3-5hpf
  - mezzo_titration  : P10ngml / P30ngml / P50ngml (Activin dose titration)

  x = emergence time (hours post fertilisation)
  y = peak pole size (highest mezzo+ area the pole reaches)

A blue shaded region marks where first poles live (early + large); an orange
shaded region marks where second poles live (later + smaller). A faint grey line
joins each pescoid's first pole to its second pole. Poles after 13 hpf excluded.

Reads:
  <Lyn_mezzo|mezzo_H2A|mezzo_titration>_phase2/phenotype_summary.csv  (on Z:)

Writes:
  <PHASE5_PLOTS>/pole_timing_vs_size_regions_treated.png
  ./pole_timing_vs_size_regions_treated.png   (local copy for quick viewing)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.spatial import ConvexHull
    HAVE_HULL = True
except Exception:
    HAVE_HULL = False

BASE = Path(r"Z:\Megha_Kattimani\Full_pipeline test")
PHASE5_PLOTS = BASE / "Lyn_mezzo_phase5_morphometrics" / "plots"
PHASE5_PLOTS.mkdir(parents=True, exist_ok=True)
LOCAL_OUT = Path("pole_timing_vs_size_regions_treated.png")

T1C, T2C = "first_mezzo_pole_emergence_hpf", "second_mezzo_pole_emergence_hpf"
A1C, A2C = "primary_mezzo_area_peak", "secondary_mezzo_area_peak"   # peak (highest) area

EMERGENCE_MAX_HPF = 13.0   # poles after this are not physiological
EMERGENCE_MIN_HPF = 6.0    # imaging starts at 6 hpf

EXPERIMENTS = [
    {"file": "Lyn_mezzo_phase2/phenotype_summary.csv", "treated": {"P_Activin_3-5hpf"}},
    {"file": "mezzo_H2A_phase2/phenotype_summary.csv", "treated": {"P_Activin_3-5hpf"}},
    {"file": "mezzo_titration_phase2/phenotype_summary.csv",
     "treated": {"P10ngml", "P30ngml", "P50ngml"}},
]

# Palette
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
C_FIRST = "#2C5BAA"    # first pole  - blue
C_SECOND = "#E8820C"   # second pole - orange


def load_treated():
    frames = []
    for exp in EXPERIMENTS:
        df = pd.read_csv(str(BASE / exp["file"]))
        frames.append(df[df["condition"].isin(exp["treated"])].copy())
    allt = pd.concat(frames, ignore_index=True)
    p = allt[allt[T1C].notna() & allt[T2C].notna() &
             allt[A1C].notna() & allt[A2C].notna()]
    p = p[(p[T1C] <= EMERGENCE_MAX_HPF) & (p[T2C] <= EMERGENCE_MAX_HPF)]
    return p.reset_index(drop=True)


def region(ax, x, y, color):
    """Tight outline (convex hull) hugging the real (x, y) points - no smoothing."""
    if not HAVE_HULL or len(x) < 3:
        return
    pts = np.column_stack([x, y])
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    poly = np.vstack([poly, poly[0]])           # close the ring
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.16, lw=0, zorder=1)
    ax.plot(poly[:, 0], poly[:, 1], color=color, lw=2.2, alpha=0.7, zorder=2)


def main():
    p = load_treated()
    n = len(p)
    t1, a1 = p[T1C].to_numpy(), p[A1C].to_numpy() / 1000.0   # -> x10^3 px^2
    t2, a2 = p[T2C].to_numpy(), p[A2C].to_numpy() / 1000.0

    simul = t1 == t2                       # both poles emerged in the same frame
    x_lo, x_hi = EMERGENCE_MIN_HPF - 0.4, EMERGENCE_MAX_HPF + 0.4
    y_lo, y_hi = 0.0, max(a1.max(), a2.max()) * 1.12

    plt.rcParams.update({"font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(11.5, 9), facecolor=BG_CREAM)
    ax.set_facecolor(BG_CREAM)

    # tight outline regions first (behind everything)
    region(ax, t1, a1, C_FIRST)
    region(ax, t2, a2, C_SECOND)

    # faint lines linking the two poles of the SAME pescoid
    for i in range(n):
        ax.plot([t1[i], t2[i]], [a1[i], a2[i]], color="#6B6B6B", lw=0.9,
                alpha=0.22, zorder=3, solid_capstyle="round")

    # the poles themselves
    ax.scatter(t1, a1, s=95, color=C_FIRST, edgecolors="white", lw=0.8,
               alpha=0.95, zorder=5)
    ax.scatter(t2, a2, s=95, color=C_SECOND, edgecolors="white", lw=0.8,
               alpha=0.95, zorder=5)

    # flag pescoids whose two poles emerged simultaneously (same frame)
    if simul.any():
        ax.scatter(np.r_[t1[simul], t2[simul]], np.r_[a1[simul], a2[simul]],
                   s=260, facecolors="none", edgecolors="black", lw=1.8,
                   zorder=6, label="_simul")

    # plain on-plot labels (no legend)
    ax.text(t1.mean() - 0.3, a1.mean() + 2.0, "First pole", color=C_FIRST,
            fontsize=20, fontweight="bold", ha="center", va="bottom")
    ax.text(t2.mean() + 1.2, a2.mean() - 0.4, "Second pole", color=C_SECOND,
            fontsize=20, fontweight="bold", ha="center", va="top")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Emergence time  (hours post fertilisation)", fontsize=16,
                  color=EMBL_NAVY, fontweight="bold")
    ax.set_ylabel(r"Peak pole size  (highest area, $\times10^{3}$ px$^2$)",
                  fontsize=16, color=EMBL_NAVY, fontweight="bold")
    ax.set_title("First pole: early & large   -->   Second pole: later & smaller",
                 fontsize=21, color=EMBL_NAVY, fontweight="bold", pad=14)
    ax.tick_params(colors=EMBL_NAVY, labelsize=13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(EMBL_NAVY)
        ax.spines[sp].set_linewidth(0.9)
    ax.grid(True, color="white", lw=1.3, alpha=0.8)
    ax.set_axisbelow(True)

    fig.text(0.5, 0.005,
             f"n = {n} treated pescoids, pooled across experiments  "
             f"(faint line = the two poles of one pescoid;  black ring = both "
             f"poles emerged in the same frame, n={int(simul.sum())};  "
             f"poles after {EMERGENCE_MAX_HPF:.0f} hpf excluded)",
             ha="center", va="bottom", fontsize=11.5, color=EMBL_NAVY, alpha=0.8)

    plt.tight_layout()
    for out in (PHASE5_PLOTS / "pole_timing_vs_size_regions_treated.png", LOCAL_OUT):
        plt.savefig(str(out), dpi=180, bbox_inches="tight", facecolor=BG_CREAM)
        print(f"Saved -> {out}")
    plt.close(fig)

    print(f"\nPooled treated pescoids (both poles <= {EMERGENCE_MAX_HPF:.0f} hpf): n={n}")
    print(f"first  pole: {t1.mean():.1f} hpf, {a1.mean():.1f}e3 px^2")
    print(f"second pole: {t2.mean():.1f} hpf, {a2.mean():.1f}e3 px^2")


if __name__ == "__main__":
    main()
