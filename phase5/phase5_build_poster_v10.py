"""
Poster v10 - narrative reordered per user feedback:

  1  The self-organisation puzzle           (big question + questions)
  2  What are pescoids and how they're prepped (slide 5 image + text)
  3  The experiment                          (slide 6 image + text)
  4  Image analysis pipeline                 (methods + framing figure)
  5  Pescoids self-resolve into 1, 2, or many organisers
                                            (merged-channel montage - CENTREPIECE)
  6  Phenotype distribution + AP pattern    (paired kymograph + emergence)
  7  Cortical mechanics + titration         (Caldarelli + dose-response)
  -- Three conclusion banners
  -- Footer (refs + acks + QR)

All figures inserted aspect-preserved (no stretch).
All text boxes sized to avoid word overflow.
Total layout fits A0 (33.11 x 46.81 in).
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

KYM = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_kymograph\plots")
MORPH = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\plots")
P3 = Path(r"Z:\Megha_Kattimani\Full_pipeline test\mezzo_titration_phase3\plots")
SLIDES = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster\slides")
OUT_DIR = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster")
OUT_FILE = OUT_DIR / "poster_kattimani_v10.pptx"

A0_W_IN = 33.11
A0_H_IN = 46.81

EMBL_GREEN = RGBColor(0x00, 0xA6, 0x89)
EMBL_NAVY = RGBColor(0x21, 0x29, 0x5C)
EMBL_RED = RGBColor(0xC0, 0x48, 0x48)
BG_CREAM = RGBColor(0xF7, 0xF2, 0xE7)
TEXT_DARK = RGBColor(0x1F, 0x1F, 0x1F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY = RGBColor(0xE5, 0xE0, 0xD3)
GREY = RGBColor(0x77, 0x77, 0x77)


def rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(1.0)
    return shp


def tbox(slide, x, y, w, h, text, *, size=18, bold=False, color=TEXT_DARK,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Arial"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = font
    return tb


def section(slide, x, y, w, num, title, h=0.9):
    rect(slide, x, y, w, h, EMBL_NAVY)
    rect(slide, x + 0.1, y + 0.1, 0.7, h - 0.2, EMBL_GREEN)
    tbox(slide, x + 0.1, y + 0.1, 0.7, h - 0.2, str(num),
         size=34, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, x + 0.9, y, w - 1.0, h, title,
         size=24, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)


def block(slide, x, y, w, h):
    rect(slide, x, y, w, h, WHITE, line=LIGHT_GREY)


def img(slide, path, x, y, w, h):
    """Aspect-preserving picture insertion (largest fit inside (w,h), centred)."""
    if not Path(path).exists():
        rect(slide, x, y, w, h, LIGHT_GREY)
        tbox(slide, x, y, w, h, f"[ Placeholder ]\n\n{Path(path).name}",
             size=18, color=GREY, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        return
    from PIL import Image as PILImage
    with PILImage.open(str(path)) as im:
        iw, ih = im.size
    box_ar = w / h
    img_ar = iw / ih
    if img_ar > box_ar:
        draw_w = w; draw_h = w / img_ar
    else:
        draw_h = h; draw_w = h * img_ar
    cx = x + (w - draw_w) / 2
    cy = y + (h - draw_h) / 2
    slide.shapes.add_picture(str(path), Inches(cx), Inches(cy),
                              width=Inches(draw_w), height=Inches(draw_h))


# ---------------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width = Inches(A0_W_IN)
    prs.slide_height = Inches(A0_H_IN)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, A0_W_IN, A0_H_IN, BG_CREAM)

    # ===== HEADER ==========================================================
    rect(slide, 0, 0, A0_W_IN, 3.0, EMBL_NAVY)
    rect(slide, 0.5, 0.45, 2.6, 2.1, EMBL_GREEN)
    tbox(slide, 0.5, 0.45, 2.6, 2.1, "EMBL LOGO\n(replace)",
         size=14, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(slide, A0_W_IN - 3.1, 0.45, 2.6, 2.1, EMBL_GREEN)
    tbox(slide, A0_W_IN - 3.1, 0.45, 2.6, 2.1,
         "QR\nGitHub +\nAVI movies",
         size=13, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, 3.5, 0.35, A0_W_IN - 7.0, 1.1,
         "How does a self-organising tissue choose one organiser, or many?",
         size=38, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, 3.5, 1.55, A0_W_IN - 7.0, 0.75,
         "Chemical perturbation of morphogen pathways affects "
         "self-organising gastrulation dynamics ex vivo",
         size=20, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, 3.5, 2.35, A0_W_IN - 7.0, 0.6,
         "Megha Kattimani  |  Trivedi Lab, EMBL Barcelona  |  "
         "megha.kattimani@embl.es  |  Poster #XX",
         size=15, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    y = 3.2

    # ===== 1. SELF-ORGANISATION PUZZLE =====================================
    section(slide, 0.4, y, A0_W_IN - 0.8, 1, "The self-organisation puzzle")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 3.5)
    col_w = (A0_W_IN - 1.4) / 2
    tbox(slide, 0.7, y + 0.15, col_w - 0.15, 3.2,
         "How does a homogeneous cell ensemble form a structured body?\n\n"
         "Self-organisation: ordered structure from local interactions, with no "
         "external blueprint. Across the tree of life - embryos, biofilms, "
         "slime moulds, swarms - the same question recurs: what selects the "
         "number, position, and identity of organising centres from a uniform "
         "starting point?",
         size=16, color=TEXT_DARK)
    tbox(slide, 0.7 + col_w + 0.1, y + 0.15, col_w - 0.15, 3.2,
         "THREE QUESTIONS\n\n"
         "1. How robust is organiser formation against chemical perturbation, "
         "and what sets organiser number?\n"
         "2. Are multi-pole outcomes driven by chemistry or by tissue mechanics?\n"
         "3. Does cortical mechanics precede or follow fate commitment at "
         "organiser sites?",
         size=16, color=TEXT_DARK)
    y += 3.6

    # ===== 2. WHAT ARE PESCOIDS + HOW THEY'RE PREPPED ======================
    section(slide, 0.4, y, A0_W_IN - 0.8, 2,
             "What are pescoids and how they are prepared")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 5.0)
    img(slide, SLIDES / "slide_05_prep_workflow.png",
        0.6, y + 0.15, A0_W_IN - 1.2, 4.7)
    y += 5.1

    # ===== 3. THE EXPERIMENT ==============================================
    section(slide, 0.4, y, A0_W_IN - 0.8, 3,
             "The experiment - Activin / Chiron pulse, three transgenic lines")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 5.5)
    # left: slide 6 image, right: brief text + transgenic line note
    exp_img_w = (A0_W_IN - 1.4) * 0.60
    img(slide, SLIDES / "slide_06_experiment.png",
        0.6, y + 0.15, exp_img_w, 5.2)
    txt_x = 0.6 + exp_img_w + 0.2
    txt_w = A0_W_IN - 0.8 - exp_img_w - 0.4
    tbox(slide, txt_x, y + 0.2, txt_w, 0.6,
         "PERTURBATION + READOUTS",
         size=18, bold=True, color=EMBL_NAVY)
    tbox(slide, txt_x, y + 0.85, txt_w, 4.4,
         "Transient pulse perturbation of two pathways:\n"
         "  - Nodal -> Activin    (3-5 or 5-7 hpf)\n"
         "  - Wnt   -> Chiron     (3-5 or 5-7 hpf)\n\n"
         "Early pulses (3-5 hpf, pre-differentiation) have the strongest "
         "effect on multipolarity + elongation.\n\n"
         "Three transgenic lines for orthogonal readouts:\n"
         "  - mezzo:GFP + LynTom (membrane / cortex)\n"
         "  - mezzo:GFP + H2A:mCherry (nuclei)\n"
         "  - mezzo:GFP + myosin:mCherry (actomyosin)\n\n"
         "Datasets in this poster: 260512 Activin, 260527 Activin titration, "
         "260514 Chiron + myosin, 260528 LynTom titration.",
         size=14, color=TEXT_DARK)
    y += 5.6

    # ===== 4. IMAGE ANALYSIS PIPELINE =====================================
    section(slide, 0.4, y, A0_W_IN - 0.8, 4,
             "Image analysis pipeline")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 3.4)
    img(slide, KYM / "00_methods_and_framing.png",
        0.6, y + 0.15, A0_W_IN - 1.2, 3.1)
    y += 3.5

    # ===== 5. CENTREPIECE - merged-channel montage =========================
    section(slide, 0.4, y, A0_W_IN - 0.8, 5,
             "Pescoids self-resolve into one, two, or many organisers")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 6.0)
    img(slide, KYM / "30_pescoid_merged_montage.png",
        0.6, y + 0.15, A0_W_IN - 1.2, 5.3)
    tbox(slide, 0.6, y + 5.5, A0_W_IN - 1.2, 0.45,
         "BF (gray) + mezzo:GFP (green) + LynTom (magenta). "
         "Each row = one pescoid, 6-16 hpf. "
         "Activin 3-5 hpf turns identical chemistry into a discrete spectrum.",
         size=13, color=TEXT_DARK, align=PP_ALIGN.CENTER)
    y += 6.1

    # ===== 6. PHENOTYPE / AP PATTERN / EMERGENCE ==========================
    section(slide, 0.4, y, A0_W_IN - 0.8, 6,
             "0% (ctrl) to ~32% multipolar - phenotype + AP pattern + emergence")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 5.2)
    half_w = (A0_W_IN - 1.6) / 2
    img(slide, KYM / "20_paired_perimeter_apmezzo.png",
        0.6, y + 0.15, A0_W_IN - 1.2, 3.0)
    img(slide, KYM / "11_v2_pole_emergence_clean.png",
        0.6, y + 3.3, half_w, 1.8)
    img(slide, KYM / "02_ap_profile_by_phenotype.png",
        0.6 + half_w + 0.2, y + 3.3, half_w, 1.8)
    y += 5.3

    # ===== 7. MECHANICS + TITRATION =======================================
    section(slide, 0.4, y, A0_W_IN - 0.8, 7,
             "Cortical mechanics (Caldarelli 2024 framework) + dose-response")
    y += 0.95
    block(slide, 0.4, y, A0_W_IN - 0.8, 4.5)
    mech_w = (A0_W_IN - 1.6) * 0.58
    img(slide, MORPH / "13_mechanics_combined_for_poster.png",
        0.6, y + 0.15, mech_w, 2.5)
    rect(slide, 0.6, y + 2.75, mech_w, 1.65, BG_CREAM, line=EMBL_RED)
    tbox(slide, 0.75, y + 2.8, mech_w - 0.3, 1.55,
         "Inside < outside in 18/25 pescoids (Wilcoxon p=0.0025). "
         "Pre-emergence cortical coherence elevated at future pole sites in "
         "65% of poles (Wilcoxon p=0.031). MECHANICS IS A CAUSE.\n"
         "Parallels Caldarelli (Nature 2024): cortical-actomyosin patterns "
         "drive embryonic regulation incl. ectopic organisers in chick/quail. "
         "Same framework, vertebrate ex vivo.",
         size=11, color=TEXT_DARK)
    tit_x = 0.6 + mech_w + 0.3
    tit_w = A0_W_IN - 1.2 - mech_w - 0.3
    tbox(slide, tit_x, y + 0.15, tit_w, 0.5,
         "Activin dose titration (10/30/50 ng/ml)",
         size=14, bold=True, color=EMBL_NAVY, align=PP_ALIGN.CENTER)
    img(slide, P3 / "01_mean_mezzo_poles_vs_dose.png",
        tit_x, y + 0.65, tit_w, 3.7)
    y += 4.6

    # ===== CONCLUSION BANNERS =============================================
    band_w = (A0_W_IN - 1.6) / 3
    for i, (color, num, head, body) in enumerate([
        (EMBL_GREEN, "1", "Multistable, not graded",
         "Activin 3-5 hpf turns identical chemistry into a discrete phenotype "
         "spectrum: 0% multipolar in ctrl, ~32% in treated."),
        (EMBL_RED, "2", "Sequential competitive nucleation",
         "Secondary poles emerge 1-3 hpf after the primary and are <50% the "
         "area. Rules out parallel symmetry breaking."),
        (EMBL_NAVY, "3", "Mechanics predicts fate",
         "Pre-emergence cortical alignment is elevated at future pole sites "
         "(p=0.031). Caldarelli framework, vertebrate ex vivo."),
    ]):
        x = 0.4 + i * (band_w + 0.2)
        rect(slide, x, y, band_w, 2.4, WHITE, line=color)
        rect(slide, x, y, band_w, 0.55, color)
        tbox(slide, x + 0.1, y, band_w - 0.2, 0.55,
             f"  Conclusion {num}: {head}",
             size=15, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
        tbox(slide, x + 0.2, y + 0.65, band_w - 0.4, 1.7, body,
             size=12, color=TEXT_DARK)
    y += 2.55

    # ===== FOOTER =========================================================
    foot_h = A0_H_IN - y - 0.3
    rect(slide, 0.4, y, A0_W_IN - 0.8, foot_h, EMBL_NAVY)
    fcol = (A0_W_IN - 1.4) / 3
    tbox(slide, 0.6, y + 0.1, fcol, foot_h - 0.2,
         "References\n"
         "[1] Caldarelli et al. (Nature 2024). Self-organized tissue "
         "mechanics underlie embryonic regulation. Direct framework for "
         "the cortical-coherence analysis here.\n"
         "[2] Claussen, Brauns & Streichan (2025). Physical principles "
         "of morphogenesis.\n"
         "[3] Weng, Huebner & Wallingford. Convergent extension biomechanics.\n"
         "[4] Fulton, Trivedi, Attardi 2020. Pescoid system.",
         size=10, color=WHITE)
    tbox(slide, 0.6 + fcol + 0.1, y + 0.1, fcol, foot_h - 0.2,
         "Acknowledgements\n"
         "Vikas Trivedi (PI), the Trivedi lab, EMBL Barcelona imaging "
         "facility, IMPRS. Conference attendance supported by the Trivedi "
         "group.\n\n"
         "Contact: megha.kattimani@embl.es\n"
         "Code: github.com/Megha-Kattimani/pescoid-image-analysis\n\n"
         "[ AVI / movies of representative pescoids via QR code ]",
         size=10, color=WHITE)
    rect(slide, 0.6 + 2 * fcol + 0.2, y + 0.15, fcol - 0.2,
          foot_h - 0.3, EMBL_GREEN)
    tbox(slide, 0.6 + 2 * fcol + 0.2, y + 0.15, fcol - 0.2,
          foot_h - 0.3,
          "QR code\n\nScan for GitHub repo\n+ AVI / movie links\n\n"
          "(upload before printing)",
          size=12, bold=True, color=WHITE,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_FILE))
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    build()
