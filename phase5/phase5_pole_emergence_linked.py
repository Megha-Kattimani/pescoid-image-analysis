"""
Updated pole emergence timing plot with:
  - 1st and 2nd pole of the SAME pescoid linked with a thin vertical line
  - Peak window 8-15 hpf highlighted in yellow (was 11-15)
  - Cream / navy poster palette to match the rest of the deck

Reads:
  Lyn_mezzo_phase2/phenotype_summary.csv

Writes:
  Lyn_mezzo_phase3/plots/12_pole_emergence_timing.png        (overwrites the old one)
  Lyn_mezzo_phase5_morphometrics/plots/pole_emergence_linked.png  (poster copy)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase2")
PHASE3_PLOTS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase3\plots")
POSTER_PLOTS = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\plots")
PHASE3_PLOTS.mkdir(parents=True, exist_ok=True)
POSTER_PLOTS.mkdir(parents=True, exist_ok=True)

# Palette
BG_CREAM = "#F7F2E7"
EMBL_NAVY = "#21295C"
EMBL_GREEN = "#00A689"
EMBL_RED = "#C04848"

CONDITIONS = ["P_ctrl", "P_Activin_3-5hpf"]
COND_LABEL = {"P_ctrl": "P ctrl", "P_Activin_3-5hpf": "P Activin 3-5h"}
COND_COLOR = {"P_ctrl": "#386cb0", "P_Activin_3-5hpf": EMBL_RED}
WINDOW = (8.0, 15.0)
PEAK_HPF = 12.0


def main():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    df = df[df.condition.isin(CONDITIONS)].copy()

    plt.rcParams.update({"font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_CREAM)
    ax.set_facecolor(BG_CREAM)

    # Yellow window of interest (8 to 15 hpf)
    ax.axhspan(WINDOW[0], WINDOW[1], color="#F5D71A", alpha=0.18, zorder=0)
    ax.axhline(PEAK_HPF, color=EMBL_NAVY, ls=":", lw=1.4, alpha=0.7,
                zorder=1, label=f"peak elongation = {PEAK_HPF:.0f} hpf")

    rng = np.random.default_rng(42)
    legend_added = {"ctrl_1": False, "ctrl_2": False,
                     "act_1": False, "act_2": False, "link": False}

    for i, cond in enumerate(CONDITIONS):
        sub = df[df.condition == cond].copy()
        sub = sub.sort_values("first_mezzo_pole_emergence_hpf",
                                kind="mergesort")
        # per-pescoid x position: spread across +/- 0.3 of the column centre
        n = len(sub)
        if n == 0:
            continue
        col_center = i
        x_pos = col_center + np.linspace(-0.32, 0.32, n)
        # tiny per-pescoid jitter so ties don't perfectly overlap
        x_pos = x_pos + rng.uniform(-0.015, 0.015, size=n)

        c = COND_COLOR[cond]
        for x, (_, row) in zip(x_pos, sub.iterrows()):
            t1 = row.get("first_mezzo_pole_emergence_hpf", np.nan)
            t2 = row.get("second_mezzo_pole_emergence_hpf", np.nan)
            if pd.notna(t1) and pd.notna(t2):
                ax.plot([x, x], [t1, t2], "-", color=c, lw=1.1, alpha=0.55,
                         zorder=2,
                         label="1st -> 2nd pole (same pescoid)"
                         if not legend_added["link"] else None)
                legend_added["link"] = True
            if pd.notna(t1):
                k = f"{'ctrl' if i == 0 else 'act'}_1"
                ax.scatter(x, t1, s=72, color=c, edgecolors="black", lw=0.7,
                            alpha=0.95, marker="o", zorder=3,
                            label=f"1st pole - {COND_LABEL[cond]}"
                            if not legend_added[k] else None)
                legend_added[k] = True
            if pd.notna(t2):
                k = f"{'ctrl' if i == 0 else 'act'}_2"
                ax.scatter(x, t2, s=78, color=c, edgecolors="black", lw=0.7,
                            alpha=0.95, marker="^", zorder=3,
                            label=f"2nd pole - {COND_LABEL[cond]}"
                            if not legend_added[k] else None)
                legend_added[k] = True

    # axes / cosmetics - label each column "N_with_pole / N_total formed mezzo+ poles"
    xtick_labels = []
    for c in CONDITIONS:
        n_total = int((df.condition == c).sum())
        n_with = int(df[(df.condition == c) &
                          df["first_mezzo_pole_emergence_hpf"].notna()].shape[0])
        n_no_pole = n_total - n_with
        xtick_labels.append(
            f"{COND_LABEL[c]}\n"
            f"{n_with}/{n_total} formed a mezzo+ pole\n"
            f"({n_no_pole} with no detected pole)")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(xtick_labels, fontsize=11, color=EMBL_NAVY,
                        fontweight="bold")
    ax.set_ylabel("Mezzo+ pole emergence time [hpf]",
                   fontsize=13, color=EMBL_NAVY, fontweight="bold")
    ax.set_title("Pole emergence timing - 1st and 2nd pole of the same pescoid linked\n"
                  f"Yellow band = window of interest ({WINDOW[0]:.0f}-{WINDOW[1]:.0f} hpf), "
                  f"peak elongation at {PEAK_HPF:.0f} hpf",
                  fontsize=14, fontweight="bold", color=EMBL_NAVY, pad=10)
    ax.tick_params(colors=EMBL_NAVY, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(EMBL_NAVY); sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color="white", lw=1.2, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 1.6)

    # Compose a clean ordered legend
    handles, labels = ax.get_legend_handles_labels()
    # de-duplicate while preserving order
    seen = set(); ord_handles = []; ord_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); ord_handles.append(h); ord_labels.append(l)
    ax.legend(ord_handles, ord_labels, loc="upper left", fontsize=10,
              frameon=False, labelcolor=EMBL_NAVY)

    plt.tight_layout()
    out_phase3 = PHASE3_PLOTS / "12_pole_emergence_timing.png"
    out_poster = POSTER_PLOTS / "pole_emergence_linked.png"
    plt.savefig(str(out_phase3), dpi=180, bbox_inches="tight",
                facecolor=BG_CREAM)
    plt.savefig(str(out_poster), dpi=180, bbox_inches="tight",
                facecolor=BG_CREAM)
    plt.close(fig)
    print(f"Saved -> {out_phase3}")
    print(f"Saved -> {out_poster}")


if __name__ == "__main__":
    main()
