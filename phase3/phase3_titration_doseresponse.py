"""
Phase 3 - Titration dose-response: does Activin concentration affect
multipolarity?

Reads mezzo_titration_phase2/phenotype_summary.csv and produces dose-response
plots ordered Pctrl -> 10 -> 30 -> 50 ng/ml.

Plots (mezzo_titration_phase3/plots/):
  01_mean_mezzo_poles_vs_dose.png    swarm + mean +/- SEM
  02_multipolarity_rate_vs_dose.png  %>=2 poles, %multipolar, %no_induction
  03_phenotype_stack_vs_dose.png     stacked phenotype fractions
  04_pole_count_distribution.png     per-pescoid pole-count histogram by dose
  05_morph_vs_mezzo_poles.png        mezzo vs morph poles by dose
  tables/dose_response_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE2 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_titration_phase2")
OUT = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_titration_phase3")
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
for d in (OUT, PLOTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

DOSE_ORDER = ["Pctrl", "P10ngml", "P30ngml", "P50ngml"]
DOSE_LABEL = {"Pctrl": "0 (ctrl)", "P10ngml": "10 ng/ml",
              "P30ngml": "30 ng/ml", "P50ngml": "50 ng/ml"}
DOSE_X = {"Pctrl": 0, "P10ngml": 10, "P30ngml": 30, "P50ngml": 50}
DOSE_COLOR = {"Pctrl": "#1f77b4", "P10ngml": "#9ecae1",
              "P30ngml": "#fd8d3c", "P50ngml": "#d62728"}

PHENO_ORDER = ["no_induction", "coordinated_monopolar", "mezzo_bipolar_only",
               "morph_bipolar_only", "disorganised_multipolar", "multipolar",
               "coordinated_bipolar", "oc_only", "diffuse_mezzo"]
PHENO_COLOR = {
    "no_induction": "#cccccc", "coordinated_monopolar": "#386cb0",
    "mezzo_bipolar_only": "#7570b3", "morph_bipolar_only": "#a6761d",
    "disorganised_multipolar": "#d95f02", "multipolar": "#e7298a",
    "coordinated_bipolar": "#1b9e77", "oc_only": "#e6ab02",
    "diffuse_mezzo": "#66c2a5",
}
RNG = np.random.default_rng(42)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F7F7F7",
                     "axes.grid": True, "grid.color": "white", "font.size": 11})


def jitter(n, w=0.6):
    return RNG.uniform(-w, w, size=n)


def main():
    df = pd.read_csv(str(PHASE2 / "phenotype_summary.csv"))
    df = df[df["condition"].isin(DOSE_ORDER)].copy()
    df["dose"] = df["condition"].map(DOSE_X)

    # ---- summary table ----
    rows = []
    for g in DOSE_ORDER:
        s = df[df["condition"] == g]
        n = len(s)
        rows.append({
            "dose": DOSE_LABEL[g], "dose_ngml": DOSE_X[g], "n": n,
            "mean_mezzo_poles": s["n_mezzo_poles_peak"].mean(),
            "sem_mezzo_poles": s["n_mezzo_poles_peak"].std(ddof=1) / np.sqrt(n) if n > 1 else 0,
            "mean_morph_poles": s["n_morph_poles_peak"].mean(),
            "pct_ge2_poles": (s["n_mezzo_poles_peak"] >= 2).mean() * 100,
            "pct_multipolar": (s["phenotype_peak"] == "multipolar").mean() * 100,
            "pct_no_induction": (s["phenotype_peak"] == "no_induction").mean() * 100,
            "max_poles": s["n_mezzo_poles_peak"].max(),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(str(TABLES / "dose_response_summary.csv"), index=False)
    print(summ.to_string(index=False))

    # ---- 01: mean mezzo poles vs dose (swarm + mean+SEM) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in DOSE_ORDER:
        s = df[df["condition"] == g]
        x = DOSE_X[g] + jitter(len(s), 1.2)
        ax.scatter(x, s["n_mezzo_poles_peak"], s=70, color=DOSE_COLOR[g],
                   edgecolors="black", lw=0.6, alpha=0.8, zorder=3)
    m = summ["mean_mezzo_poles"].values
    xs = summ["dose_ngml"].values
    sem = summ["sem_mezzo_poles"].values
    ax.errorbar(xs, m, yerr=sem, color="black", lw=2, marker="o", markersize=9,
                capsize=5, zorder=4, label="mean ± SEM")
    ax.set_xlabel("Activin [ng/ml]")
    ax.set_ylabel("# mezzo+ poles per pescoid (peak)")
    ax.set_title("Multipolarity vs Activin dose\n"
                 "(peaks at 30 ng/ml, declines at 50)", fontweight="bold")
    ax.set_xticks([0, 10, 30, 50])
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "01_mean_mezzo_poles_vs_dose.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  01_mean_mezzo_poles_vs_dose.png")

    # ---- 02: multipolarity rates vs dose ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(summ["dose_ngml"], summ["pct_ge2_poles"], "o-", lw=2, markersize=9,
            color="#e7298a", label="% with >=2 mezzo poles")
    ax.plot(summ["dose_ngml"], summ["pct_multipolar"], "s-", lw=2, markersize=9,
            color="#d62728", label="% multipolar (>=3 poles)")
    ax.plot(summ["dose_ngml"], summ["pct_no_induction"], "^--", lw=2, markersize=9,
            color="#999999", label="% no induction")
    ax.set_xlabel("Activin [ng/ml]")
    ax.set_ylabel("% of pescoids")
    ax.set_title("Pole-multiplicity rates vs Activin dose", fontweight="bold")
    ax.set_xticks([0, 10, 30, 50])
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "02_multipolarity_rate_vs_dose.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  02_multipolarity_rate_vs_dose.png")

    # ---- 03: phenotype stacked fractions ----
    fig, ax = plt.subplots(figsize=(9, 6))
    phenos_present = [p for p in PHENO_ORDER if p in df["phenotype_peak"].unique()]
    bottoms = np.zeros(len(DOSE_ORDER))
    for p in phenos_present:
        fracs = []
        for g in DOSE_ORDER:
            s = df[df["condition"] == g]
            fracs.append((s["phenotype_peak"] == p).mean() * 100 if len(s) else 0)
        ax.bar(range(len(DOSE_ORDER)), fracs, bottom=bottoms,
               color=PHENO_COLOR.get(p, "#000"), label=p, edgecolor="white", lw=0.5)
        bottoms += np.array(fracs)
    ax.set_xticks(range(len(DOSE_ORDER)))
    ax.set_xticklabels([f"{DOSE_LABEL[g]}\nn={len(df[df.condition==g])}" for g in DOSE_ORDER])
    ax.set_ylabel("% of pescoids")
    ax.set_title("Phenotype distribution vs Activin dose (peak)", fontweight="bold")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(str(PLOTS / "03_phenotype_stack_vs_dose.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  03_phenotype_stack_vs_dose.png")

    # ---- 04: pole-count distribution by dose ----
    fig, axes = plt.subplots(1, len(DOSE_ORDER), figsize=(16, 4), sharey=True)
    maxp = int(df["n_mezzo_poles_peak"].max())
    for ax, g in zip(axes, DOSE_ORDER):
        s = df[df["condition"] == g]
        ax.hist(s["n_mezzo_poles_peak"], bins=np.arange(-0.5, maxp + 1.5, 1),
                color=DOSE_COLOR[g], edgecolor="black")
        ax.set_title(f"{DOSE_LABEL[g]} (n={len(s)})", fontweight="bold")
        ax.set_xlabel("# mezzo poles")
        ax.set_xticks(range(0, maxp + 1))
    axes[0].set_ylabel("# pescoids")
    plt.suptitle("Mezzo pole-count distribution by Activin dose", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(PLOTS / "04_pole_count_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  04_pole_count_distribution.png")

    # ---- 05: mezzo vs morph poles by dose ----
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 3.5
    ax.bar(summ["dose_ngml"] - width / 2, summ["mean_mezzo_poles"], width,
           color="#2ca02c", edgecolor="black", label="mezzo poles")
    ax.bar(summ["dose_ngml"] + width / 2, summ["mean_morph_poles"], width,
           color="#ff7f0e", edgecolor="black", label="morph poles")
    ax.set_xlabel("Activin [ng/ml]")
    ax.set_ylabel("mean # poles per pescoid")
    ax.set_title("Mezzo vs morphological poles by dose\n"
                 "(P50: morph rises, mezzo falls -> diffuse not clustered)",
                 fontweight="bold")
    ax.set_xticks([0, 10, 30, 50])
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(PLOTS / "05_morph_vs_mezzo_poles.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  05_morph_vs_mezzo_poles.png")

    print(f"\nAll plots in: {PLOTS}/")


if __name__ == "__main__":
    main()
