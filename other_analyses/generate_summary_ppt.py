"""
Generate a rich summary PPT with embedded plots from the actual analysis outputs.
Covers: benchmarking, segmentation iterations, registration fix, GFP threshold work,
trajectory plots, Fiji/R comparison, tissue tension, pole length.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Output PPT
OUT_PATH = Path(
    r"C:\Users\kattimani\Project\pescoid-image-analysis\Pescoid_Pipeline_Summary_v2.pptx"
)

# Image roots
PROJECT = Path(r"C:\Users\kattimani\Project\pescoid-image-analysis")
SERVER = Path(r"Z:\Megha_Kattimani\Full_pipeline test")

# Colours
ACCENT = RGBColor(0x1F, 0x77, 0xB4)
TEXT = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x66, 0x66, 0x66)
HIGHLIGHT = RGBColor(0xE4, 0x1A, 0x1C)

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _title(slide, text, color=ACCENT, size=28):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.7))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    r = p.runs[0]
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = color


def _subtitle(slide, text, top=0.95):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    r = p.runs[0]
    r.font.size = Pt(14)
    r.font.italic = True
    r.font.color.rgb = MUTED


def _footer(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.RIGHT
    r = p.runs[0]
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED


def title_slide(title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.6), Inches(2.4), Inches(12), Inches(1.5))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    r = p.runs[0]
    r.font.size = Pt(44); r.font.bold = True; r.font.color.rgb = TEXT

    tb2 = s.shapes.add_textbox(Inches(0.6), Inches(3.6), Inches(12), Inches(1.5))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = subtitle
    r2 = p2.runs[0]
    r2.font.size = Pt(20); r2.font.color.rgb = MUTED
    return s


def bullets_slide(title, bullets, footer=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.2), Inches(12.2), Inches(5.6))
    tf = body.text_frame; tf.word_wrap = True
    for i, b in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        indent, text = (b if isinstance(b, tuple) else (0, b))
        para.text = text
        para.level = indent
        for r in para.runs:
            r.font.size = Pt(16 if indent == 0 else 14)
            r.font.color.rgb = TEXT if indent == 0 else MUTED
        para.space_after = Pt(8)
    if footer:
        _footer(s, footer)
    return s


def table_slide(title, headers, rows, footer=None, col_widths=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)

    n_cols = len(headers); n_rows = len(rows) + 1
    table_w = Inches(12.3); table_h = Inches(0.42 * n_rows + 0.4)
    tbl_shape = s.shapes.add_table(n_rows, n_cols, Inches(0.5), Inches(1.2),
                                    table_w, table_h)
    tbl = tbl_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j); cell.text = h
        for p_ in cell.text_frame.paragraphs:
            for r_ in p_.runs:
                r_.font.bold = True; r_.font.size = Pt(13)
                r_.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid(); cell.fill.fore_color.rgb = ACCENT
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j); cell.text = str(val)
            for p_ in cell.text_frame.paragraphs:
                for r_ in p_.runs:
                    r_.font.size = Pt(11); r_.font.color.rgb = TEXT
            if i % 2 == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF2, 0xF2)
    if footer:
        _footer(s, footer)
    return s


def image_slide(title, image_path, caption=None, footer=None,
                left=0.8, top=1.1, width=11.7, height=None):
    """One big image."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    p = Path(image_path)
    if p.exists():
        if height is None:
            s.shapes.add_picture(str(p), Inches(left), Inches(top), width=Inches(width))
        else:
            s.shapes.add_picture(str(p), Inches(left), Inches(top),
                                  width=Inches(width), height=Inches(height))
    else:
        warn = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.5))
        warn.text_frame.paragraphs[0].text = f"[Image missing: {p}]"
    if caption:
        _subtitle(s, caption, top=6.7)
    if footer:
        _footer(s, footer)
    return s


def two_image_slide(title, img1, img2, cap1="", cap2="", footer=None):
    """Two images side by side."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)

    for col, (img, cap) in enumerate(zip([img1, img2], [cap1, cap2])):
        left = Inches(0.4 + col * 6.45)
        p = Path(img)
        if p.exists():
            s.shapes.add_picture(str(p), left, Inches(1.2), width=Inches(6.3))
        else:
            warn = s.shapes.add_textbox(left, Inches(1.2), Inches(6.3), Inches(0.4))
            warn.text_frame.paragraphs[0].text = f"[missing: {p.name}]"
        if cap:
            cb = s.shapes.add_textbox(left, Inches(6.4), Inches(6.3), Inches(0.5))
            cp = cb.text_frame.paragraphs[0]
            cp.text = cap
            cp.alignment = PP_ALIGN.CENTER
            cr = cp.runs[0]; cr.font.size = Pt(12); cr.font.color.rgb = MUTED
            cr.font.italic = True
    if footer:
        _footer(s, footer)
    return s


def four_image_slide(title, images, captions, footer=None):
    """2x2 grid of images."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title)
    positions = [(0.4, 1.1), (6.85, 1.1), (0.4, 4.15), (6.85, 4.15)]
    for (img, cap), (l, t) in zip(zip(images, captions), positions):
        p = Path(img)
        if p.exists():
            s.shapes.add_picture(str(p), Inches(l), Inches(t), width=Inches(6.1))
        else:
            warn = s.shapes.add_textbox(Inches(l), Inches(t), Inches(6.1), Inches(0.4))
            warn.text_frame.paragraphs[0].text = f"[missing: {p.name}]"
        if cap:
            cb = s.shapes.add_textbox(Inches(l), Inches(t + 2.85), Inches(6.1), Inches(0.4))
            cp = cb.text_frame.paragraphs[0]
            cp.text = cap; cp.alignment = PP_ALIGN.CENTER
            cr = cp.runs[0]; cr.font.size = Pt(10); cr.font.italic = True
            cr.font.color.rgb = MUTED
    if footer:
        _footer(s, footer)
    return s


# =========================================================================
# SLIDE 1: title
# =========================================================================
title_slide(
    "Pescoid Image Analysis Pipeline",
    "Development Summary — Mid-February to Late April 2026  ·  Megha Kattimani"
)

# =========================================================================
# SLIDE 2: goal
# =========================================================================
bullets_slide(
    "Project Goal",
    [
        "Build an end-to-end Python pipeline to analyze zebrafish pescoid imaging.",
        (1, "Replace manual Fiji + R workflow with a reproducible, GPU-accelerated pipeline"),
        (1, "Quantify morphology (aspect ratio, area, major axis) over developmental time"),
        (1, "Quantify mezzo:GFP expression dynamics (differentiation)"),
        (1, "Detect outward poles forming during pescoid elongation"),
        (1, "Compare conditions (Activin, Chiron, Activin+Chiron, time windows)"),
        "Bonus: tissue-level tension proxies from LynTom membrane label",
    ],
)

# =========================================================================
# SLIDE 3: datasets
# =========================================================================
table_slide(
    "Datasets analyzed",
    ["Dataset", "Conditions", "Samples", "Modality"],
    [
        ["250212 mezzo Chi/Act", "7 (mezzo only)", "48", "BF + mezzo-GFP, 31 timepoints"],
        ["250212 mezzo + E-cad", "14 (7 mezzo + 7 E-cad)", "96", "BF + reporter, 31 timepoints"],
        ["250402 mezzo-LynTom Activin", "4 (E_ctrl/Act, P_ctrl/Act)", "72", "BF + mezzo + LynTom, 30 tp"],
        ["Benchmarking GT set", "18 single frames", "18", "BF + GT masks for training"],
    ],
    footer="Imaging: Olympus 10x, Zeiss, 13-slice Z-stacks",
    col_widths=[3.3, 2.7, 1.5, 4.8],
)

# =========================================================================
# SECTION: SEGMENTATION TOOL BENCHMARKING
# =========================================================================
bullets_slide(
    "Section 1 — Segmentation Tool Benchmarking",
    [
        "Compared 4 segmentation methods on 18 hand-annotated GT masks:",
        (1, "Morgana (ML pixel classifier)"),
        (1, "Cellpose cyto3 (deep learning, fine-tuned)"),
        (1, "3D U-Net (trained from scratch on GT)"),
        (1, "Ilastik (random forest via easilastik)"),
        "Metrics: IoU, Dice, Precision, Recall, Hausdorff distance.",
        "Visualisations: radar chart, accuracy box plots, per-sample IoU heatmap, area scatter.",
    ],
    footer="Output: benchmarking_segment_tools/output/comparison/",
)

# Benchmarking results table
table_slide(
    "Benchmarking — quantitative results",
    ["Tool", "IoU", "Dice", "Precision", "Recall", "Hausdorff (px)"],
    [
        ["3D U-Net", "0.948", "0.973", "0.973", "0.974", "10.6"],
        ["Cellpose (fine-tuned)", "0.945", "0.972", "0.975", "0.969", "8.5"],
        ["Morgana", "0.882", "0.937", "0.890", "0.990", "14.2"],
        ["Ilastik (easilastik)", "0.803", "0.842", "0.850", "0.839", "57.8"],
    ],
    footer="3D U-Net selected as primary segmenter (highest IoU + Dice).",
    col_widths=[3.5, 1.5, 1.5, 1.7, 1.7, 2.4],
)

# Embedded benchmark plots
image_slide(
    "Benchmarking — Accuracy comparison (IoU / Dice / Precision / Recall)",
    PROJECT / "benchmarking_segment_tools/output/comparison/tool_accuracy_boxplots.png",
    caption="Box plots: each dot = one of 18 GT images",
)

image_slide(
    "Benchmarking — Radar chart of overall performance",
    PROJECT / "benchmarking_segment_tools/output/comparison/radar_chart.png",
    caption="3D U-Net and Cellpose dominate; Ilastik weakest",
    width=8.5, left=2.4,
)

image_slide(
    "Benchmarking — Per-sample IoU heatmap",
    PROJECT / "benchmarking_segment_tools/output/comparison/per_sample_iou_heatmap.png",
    caption="Green = high IoU; red = poor. Pescoid sample on x-axis, tool on y-axis.",
)

image_slide(
    "Benchmarking — Visual comparison on GT samples",
    PROJECT / "benchmarking_segment_tools/output/comparison/visual_comparison_grid.png",
    caption="GT (green dashed) vs each tool's prediction (coloured solid). U-Net & Cellpose fit best.",
)

# =========================================================================
# SECTION: MAIN PIPELINE
# =========================================================================
bullets_slide(
    "Section 2 — Main pipeline (main_analysis.py)",
    [
        "End-to-end per-pescoid analysis (CLI, GPU):",
        (1, "Load 5D TIFF (T, Z, C, Y, X) or folder of timepoint TIFFs"),
        (1, "Z-project (best-focus BF, max-Z for fluorescence)"),
        (1, "Rotation alignment (major-axis → horizontal, padded → no cropping)"),
        (1, "Segment with U-Net (fine-tuned on 18 GT masks)"),
        (1, "Per-frame morphology: area, AR, major/minor axis, perimeter, circularity, solidity"),
        (1, "Mezzo expression: blur + threshold (Yen) + 20 µm filter, fixed threshold from brightest frame"),
        (1, "Perimeter-time kymograph from contour radii"),
        (1, "Pole detection + NEW pole-length measurement"),
        (1, "Per-frame 4-panel overlays (BF | mask | GFP | GFP+ region)"),
    ],
    footer="See HOW_TO_RUN.md for all flags and column dictionary",
)

table_slide(
    "Pipeline iterations & fixes",
    ["Version", "Change", "Outcome"],
    [
        ["v1", "Cellpose zero-shot (cyto3)", "288 failed frames / 1488 — many empty masks"],
        ["v2", "Fine-tuned Cellpose on 18 GT", "133 failed frames — elongated shapes still fail"],
        ["v3", "+ 5 manual annotations + classical fallback", "0 failed frames"],
        ["v4 (final)", "3D U-Net + rotation align + Yen + fixed thr + 20 µm filter", "Production-ready"],
    ],
    footer="Other fixes: GFP max-Z, valid-mask after registration, no-translation alignment",
    col_widths=[1.6, 6.5, 4.5],
)

# =========================================================================
# SECTION: REGISTRATION FIX
# =========================================================================
bullets_slide(
    "Section 3 — Registration / Alignment fix",
    [
        "Problem: pescoids drift up to ~90 px and rotate during 30-frame imaging.",
        "Attempt 1: phase cross-correlation (Fiji-style translation).",
        (1, "Worked but introduced black borders that thresholding picked up."),
        (1, "Caused mask artifacts in many samples."),
        "Attempt 2 (current): rotation-only alignment with padded canvas.",
        (1, "Compute major-axis orientation per frame from U-Net mask."),
        (1, "Rotate around centroid, padded enough so nothing is cropped."),
        (1, "All channels (BF / GFP / LynTom) rotate together with the same transform."),
        "Result: pescoid stays horizontal across all frames; no data loss.",
    ],
)

# G046 all timepoints registration preview
image_slide(
    "Registration fix — G046 (P_Activin) all timepoints",
    SERVER / "Lyn_mezzo_v5/test/Activin_mezzo_LynTom_A01_G046_0001/G046_all_timepoints.png",
    caption="All 30 frames after rotation alignment: pescoid stays in frame, major axis horizontal",
    width=12.5, left=0.4, top=1.1, height=5.7,
)

# =========================================================================
# SECTION: GFP / MEZZO THRESHOLD
# =========================================================================
bullets_slide(
    "Section 4 — GFP / Mezzo quantification (threshold journey)",
    [
        "Problem: 'GFP+ fraction' must reflect real differentiation, not noise.",
        "Iteration 1: Otsu per-frame — adapts to each frame, masks the rising signal.",
        "Iteration 2: Fixed threshold from brightest frame.",
        (1, "Early frames now correctly show 0% positive (no expression)."),
        "Iteration 3: blur (σ=3) + Intermodes / Li / Yen + ≥20-µm particle filter.",
        (1, "Yen + fixed threshold matched expected biology best for pescoids."),
        "Iteration 4: background subtraction (P-medium ≈ 180, E ≈ 161).",
        (1, "Threshold = 99th pct of bg-subtracted P_ctrl frame 0 (≈ 109.7)."),
        "Final metrics: GFP+ fraction, Integrated GFP, S/B ratio, mean GFP in mask, mean in GFP+ region.",
    ],
)

# Example GFP threshold comparison overlays
two_image_slide(
    "GFP threshold tuning — comparison on individual samples",
    SERVER / "gfp_method_comparison/per_sample_fixed_threshold/P_ctrl_G021_fixed_threshold.png",
    SERVER / "gfp_method_comparison/per_sample_fixed_threshold/P_Activin_3-5hpf_G036_fixed_threshold.png",
    cap1="P ctrl G021 — early (t=0) shows ~0% positive, late frames show real expression",
    cap2="P Activin G036 — clear expression at mid/late timepoints with fixed threshold",
)

# =========================================================================
# SECTION: TRAJECTORY PLOTS — Lyn_mezzo refactor
# =========================================================================
image_slide(
    "Lyn_mezzo refactor — HEADLINE S/B ratio",
    SERVER / "Lyn_mezzo_refactor/figs/01_HEADLINE_sb_ratio.png",
    caption="Background-corrected signal/background ratio. Dashed line at S/B=1.",
)

four_image_slide(
    "Lyn_mezzo refactor — main trajectory panels",
    images=[
        SERVER / "Lyn_mezzo_refactor/figs/02_integrated_gfp.png",
        SERVER / "Lyn_mezzo_refactor/figs/03_gfp_positive_fraction.png",
        SERVER / "Lyn_mezzo_refactor/figs/04_aspect_ratio.png",
        SERVER / "Lyn_mezzo_refactor/figs/05_major_axis.png",
    ],
    captions=[
        "Integrated GFP (sum of bg-sub intensity)",
        "GFP+ fraction (fixed threshold)",
        "Aspect ratio over time",
        "Major axis over time",
    ],
)

# =========================================================================
# SECTION: 250212 mezzo-only trajectories
# =========================================================================
four_image_slide(
    "250212 mezzo dataset — 7 conditions over time",
    images=[
        SERVER / "250212_mezzo_only/figs/01_aspect_ratio.png",
        SERVER / "250212_mezzo_only/figs/02_major_axis.png",
        SERVER / "250212_mezzo_only/figs/04_gfp_fraction.png",
        SERVER / "250212_mezzo_only/figs/05_gfp_mean_intensity.png",
    ],
    captions=[
        "Aspect ratio",
        "Major axis (µm)",
        "GFP+ fraction",
        "Mean GFP intensity in mask",
    ],
    footer="48 pescoids · ctrl + 6 chemical conditions (Act / Chi / Act+Chi × 3-5h / 5-7h)",
)

# =========================================================================
# SECTION: Mezzo + E-cad side by side
# =========================================================================
two_image_slide(
    "Mezzo + E-cad (14 conditions side by side)",
    SERVER / "250212_mezzo_E-Cad/figs/01_aspect_ratio.png",
    SERVER / "250212_mezzo_E-Cad/figs/04_gfp_fraction.png",
    cap1="Aspect ratio — left: 7 mezzo conditions, right: 7 E-cad conditions",
    cap2="Reporter+ fraction — comparable trajectories between markers",
)

# =========================================================================
# SECTION: Comparison with Fiji + R Analysis_V2
# =========================================================================
bullets_slide(
    "Section 5 — Comparison with Nick's Fiji + R workflow",
    [
        "Compared vs Analysis_V2 (250212 dataset, 7 mezzo conditions).",
        "Systematic differences:",
        (1, "Units: their Fiji output in microns (~1.78 µm/px), ours in pixels"),
        (1, "Their BF masks ~1.5× larger in area (Fiji thresholding more permissive)"),
        "Biology-relevant differences:",
        (1, "Their aspect ratio ≈ 1.0 across all conditions (masks too circular)"),
        (1, "Our aspect ratio = 1.27–1.71 (captures actual elongation)"),
        (1, "GFP+ fraction differs: theirs 2-8%, ours 19-47% (different thresholding strategy)"),
        "Output: comparison_with_R/ with 5 side-by-side plots.",
    ],
)

two_image_slide(
    "Fiji/R vs our pipeline — Aspect Ratio + Major Axis",
    SERVER / "250212_mezzo_E-Cad/comparison_with_R/01_AR_comparison.png",
    SERVER / "250212_mezzo_E-Cad/comparison_with_R/02_MajorAxis_comparison.png",
    cap1="Aspect ratio — Fiji (red) stays near 1.0; ours (blue) captures elongation",
    cap2="Major axis — Fiji values ~1.78x larger (unit conversion factor)",
)

# =========================================================================
# SECTION: Tissue tension (LynTom structure tensor)
# =========================================================================
bullets_slide(
    "Section 6 — Tissue tension proxy from LynTom",
    [
        "Per-cell segmentation at 10x is unreliable (~1000 cells, ~10 px each).",
        "Switched to tissue-level structure-tensor metrics:",
        (1, "Q nematic (0 = isotropic, 1 = aligned) — global tissue alignment"),
        (1, "Coherence — local anisotropy"),
        (1, "Cortex − Interior coherence — cortical tension proxy"),
        (1, "Radial coherence profile (center → edge)"),
        "Outputs: 8 trajectory plots + per-pescoid CSVs + orientation-field previews",
        "Caveat: this is a PROXY, not absolute force measurement.",
    ],
)

four_image_slide(
    "Tissue tension — trajectory metrics (P_ctrl vs P_Activin)",
    images=[
        SERVER / "lyntom_tension/figs/01_Q_nematic.png",
        SERVER / "lyntom_tension/figs/02_coherence.png",
        SERVER / "lyntom_tension/figs/05_cortex_minus_interior.png",
        SERVER / "lyntom_tension/figs/08_radial_profile_late.png",
    ],
    captions=[
        "Q nematic (global alignment)",
        "Mean coherence (local anisotropy)",
        "Cortex − Interior (cortical excess)",
        "Radial coherence profile (last 5 frames)",
    ],
)

two_image_slide(
    "Tissue tension — orientation field previews",
    SERVER / "lyntom_tension/previews/P_ctrl_G021_t15.png",
    SERVER / "lyntom_tension/previews/P_Activin_3-5hpf_G036_t15.png",
    cap1="P_ctrl G021 at t=15: BF | LynTom | coherence | orientation quiver",
    cap2="P_Activin G036 at t=15 — quiver shows dominant tissue alignment",
)

# =========================================================================
# SECTION: Pole length
# =========================================================================
bullets_slide(
    "Section 7 — NEW: Pole-length measurement",
    [
        "Pole detection already counted outward/inward poles via kymograph activity.",
        "Now also measures physical pole dimensions on the BF mask:",
        (1, "Identifies angular wedge of each pole (FWHM around activity peak)"),
        (1, "Picks the timepoint of maximum aspect ratio (peak elongation)"),
        (1, "Measures radial length from centroid to farthest mask pixel in wedge"),
        (1, "Plus: arc length, angular extent, pole area, centroid position"),
        "Output: pole_dimensions.csv per pescoid; ranked primary / secondary / etc.",
        "Example: P_Activin G013 → primary pole = 200.5 px at frame 11.",
    ],
    footer="pole_dimensions.csv columns: rank, label, peak_timepoint, pole_length_px, ...",
)

# =========================================================================
# SECTION: Plotting / stats improvements
# =========================================================================
bullets_slide(
    "Section 8 — Plotting / statistics improvements",
    [
        "Replaced LOESS / polynomial smoothing with PchipInterpolator (no overshoot).",
        "Replaced ±SD bands with bootstrap 95% confidence intervals.",
        "Clipped non-negative quantities (fractions, areas, intensities) to ≥ 0.",
        "Time axis in hpf (hours post fertilization) instead of frame index.",
        "Standard plot suite per dataset:",
        (1, "Aspect ratio, major axis, area"),
        (1, "GFP+ fraction, mean intensity, GFP+ region intensity, normalised total"),
        (1, "S/B ratio (HEADLINE for Lyn_mezzo), integrated GFP"),
    ],
)

# =========================================================================
# SECTION: cleanup + docs
# =========================================================================
bullets_slide(
    "Section 9 — Repository cleanup + docs",
    [
        "Moved 60+ test/tmp/debug scripts to junk/ folder.",
        "Final project root (clean):",
        (1, "main_analysis.py — main pipeline"),
        (1, "plot_*.py — cross-condition trajectory plots"),
        (1, "compare_with_R_analysis.py — Fiji/R comparison"),
        (1, "refactor_analysis.py — Lyn_mezzo S/B + bg-subtracted analysis"),
        (1, "run_tension_analysis.py — LynTom structure tensor"),
        (1, "analysis/ — bf_segmentation, ml_segmentation, pole_detection, tissue_tension"),
        (1, "annotation_workspace/, benchmarking_segment_tools/, MOrgAna/"),
        "HOW_TO_RUN.md — lookup table + column dictionary for all scripts.",
        "Code pushed to GitHub: github.com/Megha-Kattimani/pescoid-image-analysis",
    ],
)

# =========================================================================
# SECTION: Key findings + what's next
# =========================================================================
bullets_slide(
    "Key biological findings so far",
    [
        "Pescoids elongate over time (AR 1.0 → ~1.5–1.7); our pipeline captures this faithfully.",
        "5-7 h treatments give the highest aspect ratios (Chi 5-7h ≈ 1.71).",
        "Activin + Chiron synergistically boosts mezzo (47% fraction, 0.44 mean intensity).",
        "Activin-treated pescoids show clear mezzo induction by 11-13 hpf.",
        "Pescoid medium (L15 + phenol red) has higher background — handled by bg subtraction.",
        "Embryos serve as developmental-stage QC only; not quantified.",
        "Primary pole forms around peak-AR time; pole length now tracked.",
    ],
)

bullets_slide(
    "What's next",
    [
        "Per-cell analysis once nuclear label (H2B-mCherry) data is available.",
        "Higher-magnification imaging (40x +) for individual cell segmentation.",
        "Tissue flow / PIV analysis between consecutive registered frames.",
        "3D readout of tissue flow + nematic order (volumetric U-Net).",
        "Combine pole-length + S/B + tension metrics into a per-sample fingerprint.",
        "Publication-ready figure set with consistent styling.",
    ],
    footer="Project repo: github.com/Megha-Kattimani/pescoid-image-analysis",
)

prs.save(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")
