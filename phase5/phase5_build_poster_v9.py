"""
Poster v9 - narrative rebuild following the EMBL unit-meeting deck slides 5-11.

Story flow (top -> bottom):
  Header     : tagline + official title + author + EMBL logo + QR + poster #
  Intro band : Big question + What are pescoids + Development timeline
  Approach   : The experiment (Nodal->Activin, Wnt->Chiron, pulses) + the
               three transgenic lines (E / P_ctrl / P_treated, LynTom / H2A /
               myosin)
  Methods    : image-analysis pipeline workflow
  CENTRE     : 4-phenotype merged-channel montage (the visual hook)
  Phenotype  : population-level phenotype distribution + AP profile
  Mechanics  : Phase 4b combined panel + Caldarelli framing callout
  Titration  : dose-response (260527) condition montage + mean poles vs dose
  Conclusions: three banners + GitHub QR + AVI / movie slot + references

Output:
  Z:\\Megha_Kattimani\\Full_pipeline test\\Lyn_mezzo_phase5_morphometrics\\poster\\poster_kattimani_v9.pptx
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
OUT_DIR = Path(r"Z:\Megha_Kattimani\Full_pipeline test\Lyn_mezzo_phase5_morphometrics\poster")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "poster_kattimani_v9.pptx"

A0_W_IN = 33.11
A0_H_IN = 46.81

EMBL_GREEN = RGBColor(0x00, 0xA6, 0x89)
EMBL_NAVY = RGBColor(0x21, 0x29, 0x5C)
EMBL_RED = RGBColor(0xC0, 0x48, 0x48)
EMBL_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
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
        shp.line.color.rgb = line
        shp.line.width = Pt(1.0)
    return shp


def tbox(slide, x, y, w, h, text, *, size=18, bold=False, color=TEXT_DARK,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Arial"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run(); run.text = line
        run.font.size = Pt(size); run.font.bold = bold
        run.font.color.rgb = color; run.font.name = font
    return tb


def section(slide, x, y, w, num, title, *, h=0.9):
    rect(slide, x, y, w, h, EMBL_NAVY)
    rect(slide, x + 0.1, y + 0.1, 0.7, h - 0.2, EMBL_GREEN)
    tbox(slide, x + 0.1, y + 0.1, 0.7, h - 0.2, str(num),
         size=34, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, x + 0.9, y, w - 1.0, h, title,
         size=26, bold=True, color=WHITE,
         anchor=MSO_ANCHOR.MIDDLE)


def block(slide, x, y, w, h):
    rect(slide, x, y, w, h, WHITE, line=LIGHT_GREY)


def img(slide, path, x, y, w, h):
    """Insert image preserving its native aspect ratio; fit largest possible
    size within the (w, h) bounding box; centred horizontally and vertically."""
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
        # image is wider relative to box -> constrain width
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
    rect(slide, 0, 0, A0_W_IN, 3.4, EMBL_NAVY)
    # EMBL logo placeholder (top-left)
    rect(slide, 0.6, 0.55, 3.0, 2.3, EMBL_GREEN)
    tbox(slide, 0.6, 0.55, 3.0, 2.3, "EMBL LOGO\n(replace)",
         size=16, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # QR (top-right)
    rect(slide, A0_W_IN - 3.6, 0.55, 3.0, 2.3, EMBL_GREEN)
    tbox(slide, A0_W_IN - 3.6, 0.55, 3.0, 2.3,
         "QR\nGitHub:\nscan for code\n+ AVI movies",
         size=14, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # Title block
    tbox(slide, 4.0, 0.5, A0_W_IN - 8.0, 1.1,
         "How does a self-organising tissue choose one organiser, or many?",
         size=42, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, 4.0, 1.75, A0_W_IN - 8.0, 0.85,
         "Chemical perturbation of morphogen pathways affects "
         "self-organising gastrulation dynamics ex vivo",
         size=22, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tbox(slide, 4.0, 2.7, A0_W_IN - 8.0, 0.6,
         "Megha Kattimani  |  Trivedi Lab, EMBL Barcelona  |  "
         "megha.kattimani@embl.es  |  Poster #XX",
         size=17, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # ===== ROW 1 - Big question + What are pescoids + Timeline =============
    y = 3.7
    section(slide, 0.4, y, A0_W_IN - 0.8, 1, "The self-organisation puzzle")
    y += 1.0
    block(slide, 0.4, y, A0_W_IN - 0.8, 5.0)

    # 3 sub-columns
    col_w = (A0_W_IN - 1.4) / 3
    # 1a: Big question
    tbox(slide, 0.7, y + 0.2, col_w - 0.2, 4.6,
         "How does a homogeneous cell ensemble form a structured body?\n\n"
         "Self-organisation: ordered structure from local interactions, with "
         "no external blueprint. Across the tree of life - "
         "embryos, biofilms, slime moulds, swarms - the same question "
         "recurs: what selects the number, position, and identity of "
         "organising centres from a uniform starting point?\n\n"
         "Connects directly to cortical-mechanics + cell-flow as a "
         "self-organising layer driving embryonic regulation in mouse "
         "and chick (Caldarelli et al., Nature 2024).",
         size=18, color=TEXT_DARK)
    # 1b: What are pescoids (image slot)
    rect(slide, 0.7 + col_w + 0.1, y + 0.2, col_w - 0.2, 4.6, BG_CREAM,
          line=LIGHT_GREY)
    tbox(slide, 0.8 + col_w + 0.1, y + 0.3, col_w - 0.4, 4.4,
         "WHAT IS A PESCOID?\n\n"
         "Zebrafish blastula cap explant (Fulton, Trivedi, Attardi 2020). "
         "Deep cells + enveloping layer with the yolk removed.\n\n"
         "Maternal pre-patterning factors are removed by explantation, "
         "stripping the system to its core.\n\n"
         "Despite this, pescoids retain the capacity for:\n"
         "  - axis formation\n"
         "  - germ layer emergence\n"
         "  - mesendoderm pole formation\n\n"
         "[ replace with embryo + pescoid side-by-side image from "
         "the unit-meeting deck slide 7 ]",
         size=15, color=TEXT_DARK)
    # 1c: Three questions
    rect(slide, 0.7 + 2 * col_w + 0.2, y + 0.2, col_w - 0.2, 4.6, BG_CREAM,
          line=LIGHT_GREY)
    tbox(slide, 0.8 + 2 * col_w + 0.2, y + 0.3, col_w - 0.4, 4.4,
         "THREE QUESTIONS\n\n"
         "1. How robust is organiser formation against chemical "
         "perturbation, and what sets organiser number?\n\n"
         "2. Are multi-pole outcomes driven by chemistry or by tissue "
         "mechanics?\n\n"
         "3. Does cortical mechanics precede or follow fate commitment "
         "at organiser sites?\n\n"
         "Imaging window: 6-18 hpf.  Window of interest: 7-16 hpf, "
         "with peak elongation at 12 hpf.",
         size=15, color=TEXT_DARK)

    # ===== ROW 2 - The experiment (compact, no transgenic-lines grid) ======
    y += 5.1
    section(slide, 0.4, y, A0_W_IN - 0.8, 2,
             "Chemical perturbation of morphogen pathways - "
             "three transgenic lines")
    y += 1.0
    block(slide, 0.4, y, A0_W_IN - 0.8, 4.5)
    # Left: experiment description
    left_w = (A0_W_IN - 1.4) * 0.50
    tbox(slide, 0.7, y + 0.2, left_w, 4.1,
         "We restore the chemistry by transient PULSE perturbation of two "
         "morphogen pathways during early development:\n\n"
         "  - Nodal -> Activin  (pulse 3-5 hpf or 5-7 hpf)\n"
         "  - Wnt   -> Chiron   (pulse 3-5 hpf or 5-7 hpf)\n\n"
         "Early pulses (3-5 hpf, prior to differentiation) have the strongest "
         "effect on multipolarity and elongation.\n\n"
         "Datasets in this poster (3 transgenic lines):\n"
         "  - mezzo:GFP + LynTom    (membrane / cortex)\n"
         "  - mezzo:GFP + H2A:mCherry  (nuclei)\n"
         "  - mezzo:GFP + myosin:mCherry  (actomyosin)\n\n"
         "Experiments: 260512 Activin, 260527 Activin titration, "
         "260514 Chiron + myosin, 260528 LynTom titration.",
         size=15, color=TEXT_DARK)
    # Right: experiment schematic placeholder
    sch_x = 0.7 + left_w + 0.2
    sch_w = A0_W_IN - 1.4 - left_w - 0.2
    rect(slide, sch_x, y + 0.2, sch_w, 4.1, BG_CREAM, line=LIGHT_GREY)
    tbox(slide, sch_x, y + 0.2, sch_w, 4.1,
         "[ Replace with EXPERIMENT SCHEMATIC from EMBL Unit Meeting slide 9 ]\n\n"
         "Explantation 2.5 hpf -> Pulse 3-5 hpf -> Elongation 12 hpf\n\n"
         "Plus a small column showing the three transgenic lines\n"
         "(E / P ctrl / P treated thumbnails per line) -\n"
         "to be added once the 260514 + 260528 montages are rendered.",
         size=14, color=GREY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # ===== ROW 3 - METHODS: prep workflow + analysis pipeline ==============
    y += 4.6
    section(slide, 0.4, y, A0_W_IN - 0.8, 3,
             "Methods - pescoid preparation and image-analysis pipeline")
    y += 1.0
    block(slide, 0.4, y, A0_W_IN - 0.8, 4.5)
    # Left: pescoid prep workflow (slide 8 bottom row) - placeholder
    prep_w = (A0_W_IN - 1.7) * 0.45
    rect(slide, 0.7, y + 0.2, prep_w, 4.1, BG_CREAM, line=LIGHT_GREY)
    tbox(slide, 0.7, y + 0.25, prep_w, 0.55,
         "PESCOID PREPARATION",
         size=18, bold=True, color=EMBL_NAVY, align=PP_ALIGN.CENTER)
    tbox(slide, 0.85, y + 0.9, prep_w - 0.3, 3.2,
         "[ Replace this box with the BOTTOM ROW of slide 8\n"
         "  from the EMBL Unit Meeting deck:\n\n"
         "  2.5 hpf -> 6 hpf -> 12 hpf -> >17 hpf\n"
         "  Explantation / Mesoendoderm differentiation /\n"
         "  Elongation / Disintegration\n\n"
         "  Do NOT include the top-right Cheng et al. 2023\n"
         "  reference image. Only the 4-circle pescoid\n"
         "  workflow at the bottom of slide 8. ]",
         size=12, color=GREY, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    # Right: image analysis pipeline (methods + framing figure)
    pipe_x = 0.7 + prep_w + 0.3
    pipe_w = A0_W_IN - 1.4 - prep_w - 0.3
    tbox(slide, pipe_x, y + 0.25, pipe_w, 0.55,
         "IMAGE ANALYSIS PIPELINE",
         size=18, bold=True, color=EMBL_NAVY, align=PP_ALIGN.CENTER)
    img(slide, KYM / "00_methods_and_framing.png",
        pipe_x, y + 0.9, pipe_w, 3.2)

    # ===== ROW 4 - CENTREPIECE: merged-channel montage =====================
    y += 4.6
    section(slide, 0.4, y, A0_W_IN - 0.8, 4,
             "Pescoids self-resolve into one, two, or many organisers")
    y += 1.0
    block(slide, 0.4, y, A0_W_IN - 0.8, 6.5)
    img(slide, KYM / "30_pescoid_merged_montage.png",
        0.7, y + 0.2, A0_W_IN - 1.4, 5.6)
    tbox(slide, 0.7, y + 5.9, A0_W_IN - 1.4, 0.5,
         "BF (gray) + mezzo:GFP (green) + LynTom membrane (magenta). "
         "Each row is one pescoid imaged 6-16 hpf. "
         "Activin 3-5 hpf turns the same chemistry into a "
         "discrete spectrum of organiser counts.",
         size=14, color=TEXT_DARK, align=PP_ALIGN.CENTER)

    # ===== ROW 5 - phenotype distribution + AP profile + emergence =========
    y += 6.6
    section(slide, 0.4, y, A0_W_IN - 0.8, 5,
             "From 0% to ~32% multipolar - phenotype distribution + AP pattern")
    y += 1.0
    block(slide, 0.4, y, A0_W_IN - 0.8, 5.5)
    half_w = (A0_W_IN - 1.7) / 2
    img(slide, KYM / "20_paired_perimeter_apmezzo.png",
        0.7, y + 0.2, A0_W_IN - 1.4, 3.3)
    img(slide, KYM / "11_v2_pole_emergence_clean.png",
        0.7, y + 3.6, half_w, 1.8)
    img(slide, KYM / "02_ap_profile_by_phenotype.png",
        0.7 + half_w + 0.3, y + 3.6, half_w, 1.8)

    # ===== ROW 6 - Mechanics (left) + Titration (right) combined ==========
    y += 5.6
    section(slide, 0.4, y, A0_W_IN - 0.8, 6,
             "Cortical mechanics (Caldarelli 2024 framework)  +  "
             "dose modulates pole count")
    y += 1.0
    block(slide, 0.4, y, A0_W_IN - 0.8, 4.5)
    # Left half = mechanics
    mech_w = (A0_W_IN - 1.7) * 0.55
    img(slide, MORPH / "13_mechanics_combined_for_poster.png",
        0.7, y + 0.2, mech_w, 2.4)
    rect(slide, 0.7, y + 2.7, mech_w, 1.7, BG_CREAM, line=EMBL_RED)
    tbox(slide, 0.85, y + 2.75, mech_w - 0.3, 1.6,
         "Inside the pole < outside: 18 / 25 pescoids, Wilcoxon p = 0.0025.\n"
         "Pre-emergence at the future pole site, coherence is elevated vs "
         "random sites in 65% of poles (Wilcoxon p = 0.031).\n"
         "Mechanics is a CAUSE, not only a consequence.\n\n"
         "Parallels Caldarelli et al. (Nature 2024): cortical-actomyosin "
         "patterns drive embryonic regulation in chick / quail including "
         "ectopic organiser formation. Same framework, vertebrate ex vivo.",
         size=12, color=TEXT_DARK)
    # Right half = titration
    tit_x = 0.7 + mech_w + 0.3
    tit_w = A0_W_IN - 1.4 - mech_w - 0.3
    tbox(slide, tit_x, y + 0.2, tit_w, 0.5,
         "Activin dose titration (260527, 10 / 30 / 50 ng/ml)",
         size=14, bold=True, color=EMBL_NAVY, align=PP_ALIGN.CENTER)
    img(slide, P3 / "01_mean_mezzo_poles_vs_dose.png",
        tit_x, y + 0.7, tit_w, 3.6)

    # ===== CONCLUSIONS =====================================================
    y += 4.6
    band_w = (A0_W_IN - 1.6) / 3
    for i, (color, num, head, body) in enumerate([
        (EMBL_GREEN, "1", "Multistable, not graded",
         "Activin 3-5 hpf turns identical chemistry into a discrete "
         "phenotype spectrum: 0% multipolar in ctrl, ~32% in treated."),
        (EMBL_RED, "2", "Sequential competitive nucleation",
         "Secondary poles emerge 1-3 hpf after the primary and are <50% "
         "the area. Rules out parallel symmetry breaking."),
        (EMBL_NAVY, "3", "Mechanics predicts fate",
         "Pre-emergence cortical alignment is elevated at future pole "
         "sites (p = 0.031). Caldarelli framework, vertebrate ex vivo."),
    ]):
        x = 0.4 + i * (band_w + 0.2)
        rect(slide, x, y, band_w, 2.5, WHITE, line=color)
        rect(slide, x, y, band_w, 0.6, color)
        tbox(slide, x + 0.1, y, band_w - 0.2, 0.6,
             f"  Conclusion {num}: {head}",
             size=17, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
        tbox(slide, x + 0.2, y + 0.7, band_w - 0.4, 1.7, body,
             size=13, color=TEXT_DARK)

    # ===== FOOTER ==========================================================
    y += 2.7
    rect(slide, 0.4, y, A0_W_IN - 0.8, A0_H_IN - y - 0.4, EMBL_NAVY)
    foot_col = (A0_W_IN - 1.4) / 3
    # Refs
    tbox(slide, 0.7, y + 0.15, foot_col, A0_H_IN - y - 0.6,
         "References\n"
         "[1] Caldarelli et al. (Nature 2024). Self-organized tissue mechanics "
         "underlie embryonic regulation. Direct framework for our cortical-"
         "coherence analysis.\n"
         "[2] Claussen, Brauns & Streichan (2025). Physical principles of "
         "morphogenesis.\n"
         "[3] Weng, Huebner & Wallingford. Convergent extension biomechanics.\n"
         "[4] Fulton, Trivedi, Attardi 2020. Pescoid system reference.\n"
         "[5] Cheng et al. 2023. Zebrafish explant developmental timeline.",
         size=12, color=WHITE)
    # Acknowledgements
    tbox(slide, 0.7 + foot_col + 0.1, y + 0.15, foot_col, A0_H_IN - y - 0.6,
         "Acknowledgements\n"
         "Vikas Trivedi (PI, EMBL Barcelona). The Trivedi lab. EMBL Barcelona "
         "imaging facility. IMPRS for prior training. Conference attendance "
         "supported by the Trivedi group.\n\n"
         "Contact: megha.kattimani@embl.es\n"
         "Code: github.com/Megha-Kattimani/pescoid-image-analysis\n\n"
         "[ AVI files / movies of representative pescoids available via QR ]",
         size=12, color=WHITE)
    # QR placeholder
    rect(slide, 0.7 + 2 * foot_col + 0.2, y + 0.2,
          foot_col - 0.2, A0_H_IN - y - 0.7, EMBL_GREEN)
    tbox(slide, 0.7 + 2 * foot_col + 0.2, y + 0.2,
          foot_col - 0.2, A0_H_IN - y - 0.7,
          "QR code\n\nScan for GitHub repo:\npescoid-image-analysis\n\n"
          "and AVI / movie\nlinks (upload\nbefore printing)",
          size=14, bold=True, color=WHITE,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    prs.save(str(OUT_FILE))
    print(f"Saved -> {OUT_FILE}")


if __name__ == "__main__":
    build()
