# Pescoid Image Analysis Pipeline: How to Run

This document is a **lookup table** for every script in the project. Each entry
explains: **what the command does**, **its arguments**, and **what it outputs**.

> Most scripts also support `python <script>.py --help` for argparse-based help.

---

## 0. Setup (do this first, once per terminal session)

```powershell
cd C:\Users\kattimani\Project\pescoid-image-analysis
.\venv_morgana\Scripts\Activate
```
You should see `(venv_morgana)` at the prompt.

---

## 1. Main pipeline (`main_analysis.py`)

End-to-end per-pescoid analysis: load → rotation alignment → segment with U-Net
→ morphology → mezzo fraction → kymograph → pole detection → **pole-length
measurement** → overlays.

### Single sample or whole condition folder
```powershell
python main_analysis.py <INPUT> --out <OUTPUT> [options]
```

### All flags
| Flag | Default | Meaning |
|---|---|---|
| `<INPUT>` | required | A 5D TIFF file, OR a folder containing 5D TIFFs / timepoint subfolders |
| `--out PATH` | required | Output directory |
| `--bf-channel N` | `0` | BF channel index (set per dataset: `1` for 250212, `2` for Lyn_mezzo) |
| `--gfp-channel N` | `1` | GFP/fluorescence channel index |
| `--z-method NAME` | `best_focus` | Z-projection method (`best_focus`, `max`, `mean`, `focus_range`) |
| `--z-range Z0 Z1` | none | Z-slice range used for `focus_range` |
| `--cellpose-model NAME` | `cyto3` | (unused with U-Net, kept for back-compat) |
| `--cellpose-diameter N` | auto | (unused with U-Net) |
| `--gpu` | off | Use CUDA (highly recommended) |
| `--min-area N` | `config.py` | Minimum pescoid area in pixels |
| `--gfp-threshold NAME` | `otsu` | GFP threshold: `yen`, `li`, `intermodes`, `otsu`, or a numeric percentile |
| `--no-alignment` | off | Skip rotation alignment |
| `--no-overlays` | off | Skip per-frame overlay PNGs (much faster) |
| `--quiet` | off | Suppress verbose log lines |

### Outputs per pescoid
```
<output>/<sample_name>/
  morphology_over_time.csv          AR, area, perimeter, circularity, eccentricity, solidity
  mezzo_expression_over_time.csv    bf_area, gfp_positive_area, gfp_fraction, gfp_mean_intensity,
                                    gfp_pos_mean_intensity, gfp_total_intensity, gfp_total_normalised,
                                    gfp_max_intensity, gfp_std_intensity, gfp_threshold, time
  kymograph_matrix.npy              raw delta-radius matrix (n_perimeter_points x T)
  pole_counts.csv                   total_poles, outward_poles, inward_poles
  pole_dimensions.csv               NEW: rank, label, peak_timepoint, pole_index,
                                    pole_length_px, pole_width_arc_px,
                                    pole_angular_extent_deg, pole_area_px,
                                    pole_centroid_y/x, wedge_angle_lo/hi_deg, activity_at_peak
  summary.json                      one-shot record for the sample (now includes
                                    primary_length_px, primary_extent_deg,
                                    secondary_length_px, secondary_extent_deg)
  masks/<sample>_masks.tif          T-stack of segmentation masks (uint8 0/255)
  plots/morphology_over_time.png
  plots/mezzo_expression.png        4-panel: area / fraction / intensity / normalised total
  plots/kymograph.png               seismic perimeter-time heatmap
  plots/pole_detection.png          1D activity profile with outward/inward markers
  overlays/<sample>_tNNNN.png       4-panel per-frame overlay (BF | mask | GFP | GFP+ region)
```

### Outputs per condition folder
```
<output>/batch_summary.csv          one row per sample with mean metrics + pole counts
<output>/comparison_plots/          condition-level box plots (auto-generated)
<output>/failed_files.txt           paths that failed (if any)
```

### Examples
```powershell
# Single 5D TIFF, GPU, overlays on
python main_analysis.py "Z:\path\sample.tif" --out output\ --gpu `
  --bf-channel 1 --gfp-channel 0 --min-area 3000 --gfp-threshold yen

# Whole condition folder, no overlays (faster), Li threshold
python main_analysis.py "Z:\path\P_mezzo_ctrl" --out output\P_mezzo_ctrl `
  --gpu --bf-channel 1 --gfp-channel 0 --min-area 3000 --no-overlays --gfp-threshold li

# Loop over all conditions of a dataset (PowerShell):
$BASE = "Z:\Nick_Marschlich\...\250212_mezzo_E-Cad_Chiron_Activin\TIF"
$OUT  = "Z:\Megha_Kattimani\Full_pipeline test\250212_mezzo_only"
foreach ($exp in @(
    "P_mezzo_ctrl","P_mezzo_3-5hpf_Act","P_mezzo_3-5hpf_Chi","P_mezzo_3-5hpf_Act-Chi",
    "P_mezzo_5-7hpf_Act","P_mezzo_5-7hpf_Chi","P_mezzo_5-7hpf_Act-Chi")) {
  python main_analysis.py "$BASE\$exp" --out "$OUT\$exp" `
    --gpu --bf-channel 1 --gfp-channel 0 --min-area 3000 --gfp-threshold yen
}
```

---

## 2. Cross-condition trajectory plots

After `main_analysis.py` finishes for every condition, generate the aggregate
trajectory plots (mean ± bootstrap 95% CI per condition).

### `plot_250212_mezzo_only.py`: mezzo conditions (7) of the 250212 dataset
```powershell
python other_analyses/plot_250212_mezzo_only.py
```
- **Input:** `Z:\Megha_Kattimani\Full_pipeline test\250212_mezzo_only\`
- **Output:** `figs/01_aspect_ratio.png` … `07_gfp_total_normalised.png`
              `summary_table.csv` (mean per condition over last 3 frames)

### `plot_250212_trajectories.py`: mezzo + E-cad (14 conditions, side by side)
```powershell
python other_analyses/plot_250212_trajectories.py
```
- **Input:** `Z:\Megha_Kattimani\Full_pipeline test\250212_mezzo_E-Cad\`
- **Output:** `figs/01_aspect_ratio.png` … `07_gfp_total_normalised.png`
  (each plot has two panels: Mezzo channel left, E-cad right)

### `refactor_analysis.py`: Lyn_mezzo refactored analysis (S/B ratio, etc.)
```powershell
python other_analyses/refactor_analysis.py
```
- **Input:** `Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_yen_fixed\`
- **Output:** `Z:\...\Lyn_mezzo_refactor\figs\01_HEADLINE_sb_ratio.png` etc.
- Includes: S/B ratio (HEADLINE), integrated GFP, GFP+ fraction with fixed threshold,
  aspect ratio, major axis. Supplementary: raw 4-panel, GFP+ region intensity, area.

---

## 3. Comparison with Fiji/R Analysis_V2

### `compare_with_R_analysis.py`
Compares our pipeline output to Nick's Fiji+R `Analysis_V2` for the mezzo conditions
of the 250212 dataset.
```powershell
python other_analyses/compare_with_R_analysis.py
```
- **Output:** `Z:\...\250212_mezzo_E-Cad\comparison_with_R\`
  - `01_AR_comparison.png`, `02_MajorAxis_comparison.png`, etc.
  - `summary_table.csv` showing absolute values + % difference per metric

---

## 4. Tissue tension (LynTom membrane), set aside

### `run_tension_analysis.py`
Structure-tensor based tissue alignment / cortical-tension proxy on the Lyn_mezzo
dataset (pescoids only, no embryos).
```powershell
python other_analyses/run_tension_analysis.py
```
- **Output:** `Z:\Megha_Kattimani\Full_pipeline test\lyntom_tension\`
  - `figs/01_Q_nematic.png` …  `08_radial_profile_late.png`
  - `previews/<cond>_<G>_t15.png`: 4-panel orientation field maps
  - per-pescoid CSVs + `all_tension_data.csv`

---

## 5. Argparse-based help

For any script with argparse (currently `main_analysis.py`):
```powershell
python main_analysis.py --help
```
prints all available flags + a short description per flag.

The other scripts (`plot_*.py`, `compare_*.py`, `run_tension_analysis.py`,
`refactor_analysis.py`) have hard-coded paths at the top, so open the file and
edit `DATA_ROOT`, `ANALYSIS_DIR`, `OUT` if running on a different dataset.

---

## 6. Quick reference: when to use which script

| Task | Script |
|---|---|
| Per-sample segmentation + morphometrics + mezzo + kymograph + pole detection + **pole length** | `main_analysis.py` |
| Cross-condition trajectory plots (mezzo only) | `plot_250212_mezzo_only.py` |
| Cross-condition trajectory plots (mezzo + E-cad side by side) | `plot_250212_trajectories.py` |
| Pescoid-only refactor with S/B + bg-subtracted + integrated GFP | `refactor_analysis.py` |
| Compare with Fiji/R Analysis_V2 | `compare_with_R_analysis.py` |
| Tissue tension / membrane orientation field | `run_tension_analysis.py` |

---

## 7. Output column dictionary

### `morphology_over_time.csv`
| Column | Units | Meaning |
|---|---|---|
| `time` | frame index | 0-based |
| `area` | px² | BF mask area |
| `perimeter` | px | Mask perimeter |
| `aspect_ratio` | unitless | major_axis / minor_axis |
| `major_axis` | px | Fitted ellipse major axis length |
| `minor_axis` | px | Fitted ellipse minor axis length |
| `circularity` | 0–1 | 4π·area / perimeter² |
| `solidity` | 0–1 | area / convex_hull_area |
| `eccentricity` | 0–1 | Ellipse eccentricity |

### `mezzo_expression_over_time.csv`
| Column | Units | Meaning |
|---|---|---|
| `time` | frame | |
| `bf_area` | px² | BF mask area |
| `gfp_positive_area` | px² | GFP+ pixels in mask after blur + threshold + 20 µm filter |
| `gfp_fraction` | 0–1 | gfp_positive_area / bf_area |
| `gfp_mean_intensity` | a.u. | Mean GFP across mask (raw, normalised to [0,1] per frame) |
| `gfp_pos_mean_intensity` | a.u. | Mean GFP **inside GFP+ region** only |
| `gfp_total_intensity` | a.u. | Sum of GFP intensity in mask |
| `gfp_total_normalised` | a.u./px² | gfp_total_intensity / bf_area |
| `gfp_max_intensity` | a.u. | Max GFP in mask |
| `gfp_std_intensity` | a.u. | SD of GFP in mask |
| `gfp_threshold` | a.u. | Fixed threshold (from brightest frame, blurred) |

### `pole_dimensions.csv` (NEW)
| Column | Units | Meaning |
|---|---|---|
| `rank` | 1, 2, … | 1 = primary (longest) pole |
| `label` | text | `primary`, `secondary`, or `pole_N` |
| `peak_timepoint` | frame | Frame of maximum aspect ratio used for measurement |
| `pole_index` | perimeter bin | Pole position along the kymograph perimeter axis |
| `pole_length_px` | px | Radial distance from centroid to farthest mask pixel in pole wedge (= **pole length**) |
| `pole_width_arc_px` | px | Arc length at the tip radius |
| `pole_angular_extent_deg` | degrees | FWHM angular width of the pole's activity peak |
| `pole_area_px` | px² | Mask area inside the pole wedge |
| `pole_centroid_y/x` | px | Pole region centroid (image coordinates) |
| `wedge_angle_lo/hi_deg` | degrees | Bounds of the pole wedge from centroid |
| `activity_at_peak` | a.u. | Smoothed kymograph activity at the pole index |

### `pole_counts.csv`
| Column | Meaning |
|---|---|
| `total_poles` | Number of pole peaks detected on the kymograph |
| `outward_poles` | Subset with positive time-averaged radial displacement (protrusions) |
| `inward_poles` | Subset with negative time-averaged radial displacement (indentations) |

### `summary.json`
Top-level dict with `file`, `condition`, `shape`, `cellpose_model`, `z_method`,
`failed_frames`, `morphology`, `mezzo`, **`poles` (n_poles, n_outward, n_inward,
primary_length_px, primary_extent_deg, secondary_length_px,
secondary_extent_deg)**, `kymograph_shape`, `output_dir`.

### `batch_summary.csv` (per condition)
Aggregated metrics per pescoid: mean area, AR, GFP fraction/intensity, pole counts, etc.

---

## 8. Notes / Gotchas

- **Pixel size**: our output is in pixels; Fiji/R Analysis_V2 is in microns at
  about **1.78 µm/px**. Multiply our values by 1.78 to compare with Nick's R output.
- **Background**: pescoid medium (L15 + Phenol Red) creates a brighter background
  than embryo medium (E3). The `refactor_analysis.py` script handles this by
  computing background per-frame from outside the mask and subtracting it.
- **Embryos**: only used for staging / QC. Don't include `E_*` conditions in
  pescoid analyses.
- **Rotation alignment**: rotation-only (no translation) with padded canvas so
  no edges are cropped. Disable with `--no-alignment` if you want raw frames.
