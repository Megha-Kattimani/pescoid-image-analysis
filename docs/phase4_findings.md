# Phase 4 findings: membrane texture order and the mezzo regions

This note records what the Phase 4 structure-tensor analysis actually shows, the
corrections it went through, and the caveats. Read it before quoting any "tissue tension"
result, because the wording matters.

## What is being measured

Phase 4 (`phase4/phase4_tension.py`) computes a **structure-tensor coherence** on the
LynTom membrane channel: in each single frame, within a roughly 10 pixel neighbourhood, how
consistently do the membrane-intensity gradients point one way. Coherence runs from 0
(disordered) to 1 (aligned).

Two things to keep straight:

- It is **spatial and single-frame**, not a frame-to-frame motion measurement. The motion
  measure is the Phase 5 PIV flow.
- The LynTom image is a **max-Z projection over the full 12 to 13 slice stack** (EVL down to
  deep cells). So this is best called **membrane texture order**, not a direct tension
  readout. Per-slice checks show the coherence is similar across depth (it is not one bright
  EVL slice driving it), and the rim-versus-core gradient is an in-plane geometry effect, not
  a depth artifact.

## The main result

Defining the mezzo region from the **actual GFP signal** (not Phase 2 pole labels), across
all 51 pescoids (14 control, 37 Activin):

- The **mezzo-GFP positive tissue is less coherent than the mezzo-negative tissue**, in 50 of
  51 pescoids (control 0.535 vs 0.791; Activin 0.569 vs 0.711; both p < 0.001).
- mezzo+ tissue is also somewhat more central (mean radius 0.48 vs 0.58), so part of the raw
  gap is the rim/core geometry. After matching radial position (comparing mezzo+ vs mezzo-
  only within the same distance-from-edge shells) the effect **survives**: about -0.12, in 49
  of 51 pescoids, Wilcoxon p approximately 3e-9. Roughly 30 percent of the raw difference was
  geometry; the rest is real.
- It is **not a brightness or signal-to-noise artifact**: the mezzo regions are brighter than
  the body, not dimmer, and coherence does not track brightness (r = 0.12, not significant).
- It holds in **both** conditions, so it is about mezzo identity, not about Activin or about
  being a pole.

Read plainly: wherever the mezzo (mesendoderm) marker is on, the membrane texture is more
disordered than mezzo-negative tissue at the same radial position. That is consistent with
mezzo+ cells being less epithelially ordered (more mesenchymal-like), but confirming that
interpretation needs an orthogonal readout.

Figures: `images/phase4_gfp_based_ctrl_vs_activin.png` (main result),
`images/phase4_radius_matched_test.png` (the geometry control),
`images/phase4_regions_G047.png` (how the regions are defined),
`images/phase4_kymograph_coherence_mezzo_shape.png` (coherence vs mezzo vs shape over time).

## Controls do express mezzo (an important correction)

An earlier version of this analysis said controls rarely form poles and dropped them. That
was wrong. Controls are full of mezzo-GFP (see `images/ctrl_gfp_check.png`); the difference
is that control mezzo is **diffuse**, filling much of the pescoid, while Activin **localizes**
it into discrete poles. Phase 2 only labels a localized, persistent, peripheral cluster a
"pole", so it tags diffuse control mezzo as "transient" or "organising_centre" or nothing,
which made the controls look empty. Measured by GFP fraction, control mezzo covers about 46
percent of the pescoid versus 32 percent for Activin (p = 0.012). Overall tissue coherence is
the same between conditions (0.66 vs 0.66).

So any pole-based comparison is Activin-specific by construction. To include controls you must
define the mezzo region from the GFP signal, as done above.

## The kymograph

`images/phase4_kymograph_coherence_mezzo_shape.png` stacks, for one monopolar pescoid (G046)
on a shared time axis: LynTom coherence along the AP axis, mezzo-GFP along the AP axis, and
the perimeter shape-change kymograph. The mezzo builds at the posterior over time, and the
coherence dips where and when the mezzo is high (pooled r = -0.25). Note the AP axis is
oriented by the mezzo direction, so "mezzo is posterior" is partly by construction; the
content is the coherence dip co-located with the mezzo.

## G047 worked example (single bipolar pescoid)

For one bipolar Activin pescoid (G047) the per-region time courses are worth recording, with
the standing caveat that single pescoids are geometry-confounded and the population numbers
are the real evidence.

Anatomy: G047 has three extensions, the anterior end (a bare top bulge with no mezzo) and two
posterior mezzo-GFP poles.

- `images/phase4_kymograph_G047.png`: coherence, mezzo, and shape-change kymographs on a
  shared time axis. Along the anterior-to-posterior axis, coherence and mezzo are
  anti-correlated, pooled r = -0.50 (stronger than the G046 monopolar case).
- `images/phase4_G047_two_pole_axes.png`: coherence and mezzo along the axis to each of the
  two poles. Both axes show the same coherence-down, mezzo-up gradient toward the pole.
- `images/phase4_G047_with_bodycore.png`: coherence over time for the anterior, the two
  poles, and the body core.

The body core is one of the least coherent regions (about 0.58). The two poles are not
equivalent. The **primary** pole (persistent, larger, lasts the whole movie) is low coherence
(about 0.59) and tracks the body core; the **secondary** pole (transient, gone by about
11 hpf) is high (about 0.73), like the anterior bulge (about 0.77). So the persistent,
dominant mezzo pole is the disordered one, continuous with the disordered core, which lines up
with the population result, while the bare anterior and the short-lived secondary pole stay
ordered. This is one pescoid, so a hypothesis, not a result.

## Caveats to carry forward

- Membrane **texture order**, from a max-Z projection. Not a direct tension or EMT measure.
- The honest effect size is the radius-matched one (about 0.12), not the raw 0.17.
- The kymograph and the per-pescoid time courses are single pescoids; the population numbers
  are the evidence.
- Phase 2 under-labels diffuse mezzo as non-pole. That is a real limitation of the pole
  classifier, separate from the coherence question, and worth fixing if bare or diffuse mezzo
  matters biologically.

## Open next steps

- An orthogonal check of the mesenchymal interpretation (junction or polarity markers, or a
  single-plane acquisition instead of max-Z).
- Per-slice cortex-vs-interior to fully separate EVL from deep.
- Whether the coherence dip precedes or follows mezzo onset at a given location
  (cause versus consequence), using the kymograph time axis.
