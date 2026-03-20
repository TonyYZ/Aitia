"""
demos.py — Concept Representation Demos
========================================

Illustrates how to build ConceptGraphs for spatial image schemas and verify
that evaluations behave as intended.

Run with:  python demos.py

Key conventions
---------------
Band reading direction:
    VERTICAL   pattern[0]=bottom  pattern[n-1]=top
    HORIZONTAL pattern[0]=left    pattern[n-1]=right

Background: any all-zero pattern (坤, 阴, 太阴, …) returns 0 — yin is
passive; it does not scan.  Dimness at a location is captured implicitly
when a yang-scanner there returns a low value.

There is no `layer` attribute on ObjectNode.  Conditional relationships
between nodes are expressed exclusively via dashed edges.
"""

import numpy as np
import textwrap
from embodied_lot import (
    ConceptGraph, ObjectNode, SerialNode, ParallelNode, BodyNode,
    ReceptiveFieldNode, RetinaNode, TactileNode,
    ReceptiveField, Orientation,
    TRIGRAM_PATTERNS, MONOGRAM_PATTERNS,
)


# ── Display helpers ───────────────────────────────────────────────────

def header(title):
    print(); print("=" * 62); print(f"  {title}"); print("=" * 62)

def result(label, value, note=""):
    bar  = "█" * round(value * 20)
    pad  = "░" * (20 - round(value * 20))
    note = f"   ← {note}" if note else ""
    print(f"  {label:<38s}  {value:.3f}  |{bar}{pad}|{note}")

def section(text):
    print(f"\n  ── {text}")

def field(recipe):
    a = np.zeros((9, 9))
    if   recipe == "top_bright":    a[:3,  :] = 0.9; a[3:,  :] = 0.1
    elif recipe == "bottom_bright": a[6:,  :] = 0.9; a[:6,  :] = 0.1
    elif recipe == "left_bright":   a[:,  :3] = 0.9; a[:, 3:]  = 0.1
    elif recipe == "right_bright":  a[:, 6:]  = 0.9; a[:, :6]  = 0.1
    elif recipe == "center_bright": a[:,   :] = 0.1; a[3:6, 3:6] = 0.9
    elif recipe == "uniform_high":  a[:,   :] = 0.9
    elif recipe == "uniform_mid":   a[:,   :] = 0.5
    elif recipe == "uniform_low":   a[:,   :] = 0.1
    elif recipe == "outer_bright":
        a[:, :] = 0.1; a[0,:]=a[-1,:]=a[:,0]=a[:,-1]=0.9
    return ReceptiveField(a)

def single(pattern, ori, name=""):
    """One-node concept: ParallelNode [retina] + ObjectNode scanner."""
    g = ConceptGraph(name or str(pattern))
    par = ParallelNode("par")
    sc  = ObjectNode("sc", pattern, ori, name=name or str(pattern))
    g.add_nodes([par, sc]); g.add_solid_edge(par, sc); g.root = par
    return g


# =============================================================
# Demo 1 — VERTICAL POSITION with trigrams used directly
# =============================================================
header("Demo 1 — Vertical Position: 'above' and 'below'")
print(textwrap.dedent("""
  Convention reminder:
    VERTICAL  pattern[0]=bottom  pattern[2]=top

  艮 (0,0,1) — bottom yin, middle yin, top yang  → "figure at the top"
  震 (1,0,0) — bottom yang, middle yin, top yin  → "figure at the bottom"

  Both scan the entire field at once.  No need for a SerialNode here:
  the three-band gradient structure is precisely what the trigram encodes.

  Comparison field breakdown (9 rows, 3 equal bands of 3):
    top_bright:    top-band (rows 0-2)=0.9  mid(rows 3-5)=0.1  bot(rows 6-8)=0.1
    bottom_bright: top-band=0.1  mid=0.1  bot=0.9
"""))

above = single(TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, "top")
below = single(TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL, "bottom")

section("top (0,0,1) — above schema  [艮 Gèn]")
for recipe, note in [
    ("top_bright",    "top-band=0.9 matches yang→HIGH"),
    ("bottom_bright", "top-band=0.1 mismatches yang→LOW"),
    ("uniform_mid",   "all bands 0.5→MODERATE"),
    ("uniform_high",  "all bands 0.9: top-band✓ mid/bot also match yin? no→HIGH"),
]:
    result(recipe, above.evaluate_against(field(recipe)), note)

section("bottom (1,0,0) — below schema  [震 Zhèn]")
for recipe, note in [
    ("bottom_bright", "bot-band=0.9 matches yang→HIGH"),
    ("top_bright",    "bot-band=0.1 mismatches yang→LOW"),
    ("uniform_high",  "all 0.9: bot matches✓ mid/top also high"),
]:
    result(recipe, below.evaluate_against(field(recipe)), note)

# Show exact band means for verification
section("Exact band means (bottom→top) for top_bright and bottom_bright:")
for recipe in ("top_bright", "bottom_bright"):
    f9 = field(recipe).activation
    bot = f9[6:9, :].mean(); mid = f9[3:6, :].mean(); top = f9[0:3, :].mean()
    print(f"  {recipe}: band[0]=bottom={bot:.1f}  band[1]=mid={mid:.1f}  band[2]=top={top:.1f}")
print("  艮 pattern: (0,0,1) → sims: (1-|bot-0|, 1-|mid-0|, 1-|top-1|)")


# =============================================================
# Demo 2 — HORIZONTAL POSITION
# =============================================================
header("Demo 2 — Horizontal Position: 'left-of' and 'right-of'")
print(textwrap.dedent("""
  HORIZONTAL  pattern[0]=left  pattern[2]=right

  震 (1,0,0) HORIZONTAL — left yang, center yin, right yin → "figure on left"
  艮 (0,0,1) HORIZONTAL — left yin,  center yin, right yang → "figure on right"
"""))

left_of  = single(TRIGRAM_PATTERNS["bottom"], Orientation.HORIZONTAL, "left")
right_of = single(TRIGRAM_PATTERNS["top"], Orientation.HORIZONTAL, "right")

section("left/bottom (1,0,0) HORIZONTAL — left-of  [震H]")
for recipe, note in [
    ("left_bright",  "left-band=0.9 matches yang→HIGH"),
    ("right_bright", "left-band=0.1 mismatches→LOW"),
    ("top_bright",   "all column-bands ≈ equal mean→MODERATE"),
]:
    result(recipe, left_of.evaluate_against(field(recipe)), note)

section("right/top (0,0,1) HORIZONTAL — right-of  [艮H]")
for recipe, note in [
    ("right_bright", "right-band=0.9→HIGH"),
    ("left_bright",  "right-band=0.1→LOW"),
]:
    result(recipe, right_of.evaluate_against(field(recipe)), note)


# =============================================================
# Demo 3 — CENTER PRESENCE (坎)
# =============================================================
header("Demo 3 — Center Presence: 坎 (0,1,0)")
print(textwrap.dedent("""
  坎 (0,1,0) — bottom yin, middle yang, top yin.
  Applied both VERTICAL and HORIZONTAL and averaged (ParallelNode):
  high when center row AND center column are bright.
"""))

def build_center():
    g = ConceptGraph("center")
    root = ParallelNode("root"); blend = ParallelNode("blend")
    cv = ObjectNode("cv", TRIGRAM_PATTERNS["center"], Orientation.VERTICAL,   name="center-V")
    ch = ObjectNode("ch", TRIGRAM_PATTERNS["center"], Orientation.HORIZONTAL, name="center-H")
    g.add_nodes([root, blend, cv, ch])
    g.add_solid_edge(root, blend)
    g.add_solid_edge(blend, cv); g.add_solid_edge(blend, ch)
    g.root = root; return g

center = build_center()
for recipe, note in [
    ("center_bright", "center occupied→HIGH"),
    ("outer_bright",  "periphery only→LOW"),
    ("top_bright",    "top row bright but center-col not isolated→partial"),
    ("uniform_high",  "middle band matches but so do edges→LOW"),
]:
    result(recipe, center.evaluate_against(field(recipe)), note)


# =============================================================
# Demo 4 — UNIVERSALITY (乾)
# =============================================================
header("Demo 4 — Universality: 乾 (1,1,1)")
print(textwrap.dedent("""
  乾 (1,1,1) — all bands yang.  Scores HIGH when the field is uniformly bright.
"""))

all_g = single(TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL, "universal")
for recipe, note in [
    ("uniform_high",  "fully saturated→HIGH"),
    ("uniform_mid",   "half saturated→MODERATE"),
    ("top_bright",    "only top-band bright, mid+bot yin mismatch→LOW"),
    ("uniform_low",   "nothing→LOW"),
]:
    result(recipe, all_g.evaluate_against(field(recipe)), note)


# =============================================================
# Demo 5 — SERIAL NODE: "A above B" (two distinct scanners)
# =============================================================
header("Demo 5 — SerialNode: 'bright top AND bright bottom' AND gate")
print(textwrap.dedent("""
  SerialNode VERTICAL splits into two strips:
    children[0] = bottom strip   children[1] = top strip

  Both children use 乾 (all-yang) so we can see the strip assignment clearly.
  Activation = min(乾_bottom, 乾_top) — both strips must be bright.

  Expected:
    uniform_high   → both strips bright → min(0.9, 0.9) = 0.9
    top_bright     → bottom strip dim   → min(low,  0.9) = low
    bottom_bright  → top strip dim      → min(high, low) = low
"""))

def build_two_region_and():
    g = ConceptGraph("two_region_AND")
    par = ParallelNode("par")
    ser = SerialNode("ser", split_orientation=Orientation.VERTICAL)
    bot = ObjectNode("bot", TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL, name="universal-bot")
    top = ObjectNode("top", TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL, name="universal-top")
    g.add_nodes([par, ser, bot, top])
    g.add_solid_edge(par, ser)
    g.add_solid_edge(ser, bot)   # children[0] = bottom strip
    g.add_solid_edge(ser, top)   # children[1] = top strip
    g.root = par; return g

and_g = build_two_region_and()
for recipe, note in [
    ("uniform_high",  "both strips bright→HIGH"),
    ("top_bright",    "bottom strip dim→LOW (min bottleneck)"),
    ("bottom_bright", "top strip dim→LOW (min bottleneck)"),
    ("uniform_low",   "both dim→LOW"),
]:
    result(recipe, and_g.evaluate_against(field(recipe)), note)

section("Using different scanners: top(艮) on bottom strip, bottom(震) on top strip")
print(textwrap.dedent("""    (This encodes: bottom strip should look top-heavy (艮) AND
     top strip should look bottom-heavy (震) — a deliberately odd concept
     to show that each child scans its own strip independently.)"""))

def build_mixed_serial():
    g = ConceptGraph("mixed_serial")
    par = ParallelNode("par")
    ser = SerialNode("ser", split_orientation=Orientation.VERTICAL)
    bot = ObjectNode("bot", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, name="top on bot-strip")
    top = ObjectNode("top", TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL, name="bottom on top-strip")
    g.add_nodes([par, ser, bot, top])
    g.add_solid_edge(par, ser); g.add_solid_edge(ser, bot); g.add_solid_edge(ser, top)
    g.root = par; return g

mix = build_mixed_serial()
for recipe, note in [
    ("top_bright",    "bot-strip(rows 4-8)=dim; 艮 on dim strip; 震 on top-strip(dim+bright)"),
    ("uniform_mid",   "all 0.5→moderate for both scanners"),
]:
    result(recipe, mix.evaluate_against(field(recipe)), note)


# =============================================================
# Demo 6 — BODY NODE: spatial template unfolding
# =============================================================
header("Demo 6 — BodyNode: template directs where content is evaluated")
print(textwrap.dedent("""
  BodyNode children:
    child[0] = spatial template (determines WHERE child[1] scans)
    child[1] = content scanner (evaluated in the yang-regions of the template)

  Schema: 艮 (0,0,1) as template → yang region = top third of the field.
          乾 (1,1,1) as content → evaluated only within the top-third sub-field.

  Result: high when the TOP THIRD of the field is bright (regardless of the rest).
"""))

def build_body():
    g = ConceptGraph("body_above")
    par  = ParallelNode("par")
    body = BodyNode("body", subfield_shape=(9, 9))
    tmpl = ObjectNode("tmpl", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, name="top-template")
    cont = ObjectNode("cont", TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL, name="universal-content")
    g.add_nodes([par, body, tmpl, cont])
    g.add_solid_edge(par, body); g.add_solid_edge(body, tmpl); g.add_solid_edge(body, cont)
    g.root = par; return g

body_g = build_body()
for recipe, note in [
    ("top_bright",    "top third bright → 乾 on bright sub-field → HIGH"),
    ("bottom_bright", "top third dim    → 乾 on dim sub-field   → LOW"),
    ("center_bright", "top third dim (center is rows 3-6) → LOW"),
    ("uniform_high",  "top third bright → HIGH"),
    ("uniform_low",   "top third dim    → LOW"),
]:
    result(recipe, body_g.evaluate_against(field(recipe)), note)

section("Using bottom(震) as template → content placed in BOTTOM third")
def build_body_below():
    g = ConceptGraph("body_below")
    par  = ParallelNode("par")
    body = BodyNode("body2", subfield_shape=(9, 9))
    tmpl = ObjectNode("tmpl", TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL, name="bottom-template")
    cont = ObjectNode("cont", TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL, name="universal-content")
    g.add_nodes([par, body, tmpl, cont])
    g.add_solid_edge(par, body); g.add_solid_edge(body, tmpl); g.add_solid_edge(body, cont)
    g.root = par; return g

body_below = build_body_below()
for recipe, note in [
    ("bottom_bright", "bottom third bright → HIGH"),
    ("top_bright",    "bottom third dim    → LOW"),
]:
    result(recipe, body_below.evaluate_against(field(recipe)), note)


# =============================================================
# Demo 7 — CONDITIONAL GATING via dashed edge → field node
# =============================================================
header("Demo 7 — Dashed Edge: conditional modulation of field presence")
print(textwrap.dedent("""
  Dashed edges modulate node.activation AFTER the forward pass.
  For field nodes, .activation IS the persistent presence weight.
  So dashing a condition to a field node gates which modality is active
  on subsequent evaluation calls.

  Setup:
    ParallelNode
        ├── RetinaNode  (top_bright, w=1.0)
        ├── TactileNode (bottom_bright, w=0.5)
        ├── 艮 scanner  (detects bright top)
        └── 乾 condition (fires strongly on uniform_high, weakly on dim fields)
    DashedEdge: 乾 ⤳ TactileNode (gain=0.0) — suppresses tactile

  Cycle 1 forward: condition fires; tactile still w=0.5; result = blend
  Cycle 1 dashed:  tactile.activation ×= (乾_act × 0.0) = 0
  Cycle 2 forward: tactile.activation=0 → ParallelNode ignores tactile
                   → only retina contributes → result ≈ retina-only score
"""))

def build_gated():
    g    = ConceptGraph("gated")
    par  = ParallelNode("par")
    ser  = SerialNode("ser", split_orientation=Orientation.VERTICAL)
    # scanner: 艮 on top strip, 震 on bottom strip — the "above" concept
    sc_top = ObjectNode("top", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, name="top")
    sc_bot = ObjectNode("bot", TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL, name="bottom")
    cond   = ObjectNode("cond", TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL, name="universal-cond")
    ret    = RetinaNode("retina",  field=field("top_bright"),    initial_weight=1.0)
    tac    = TactileNode("tactile", field=field("bottom_bright"), initial_weight=0.5)
    g.add_nodes([par, ser, sc_top, sc_bot, cond, ret, tac])
    g.add_solid_edge(par, ret); g.add_solid_edge(par, tac)
    g.add_solid_edge(par, cond)
    g.add_solid_edge(par, ser)
    g.add_solid_edge(ser, sc_bot)   # bottom strip
    g.add_solid_edge(ser, sc_top)   # top strip
    g.add_dashed_edge(cond, tac, gain=0.0)
    g.root = par; return g

gated = build_gated()
act1 = gated.evaluate()
result("Cycle 1 (retina w=1, tactile w=0.5)", act1, "weighted blend of two opposing fields")
print(f"    tactile.activation after dashed pass: {gated._nodes['tactile'].activation:.3f}  (→ 0)")
act2 = gated.evaluate()
result("Cycle 2 (tactile now w=0)", act2, "only retina contributes")

section("Verify: single-field 'above' score for each modality separately:")
ab = ConceptGraph("above_ref")
par_r = ParallelNode("par")
ser_r = SerialNode("ser", split_orientation=Orientation.VERTICAL)
a_bot = ObjectNode("b", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, name="top")
a_top = ObjectNode("t", TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL, name="bottom")
ab.add_nodes([par_r, ser_r, a_bot, a_top])
ab.add_solid_edge(par_r, ser_r); ab.add_solid_edge(ser_r, a_bot); ab.add_solid_edge(ser_r, a_top)
ab.root = par_r
result("above | retina=top_bright  (cycle 2 target)", ab.evaluate_against(field("top_bright")))
result("above | tactile=bot_bright (suppressed)",     ab.evaluate_against(field("bottom_bright")))


# =============================================================
# Demo 8 — MULTI-MODAL field pooling
# =============================================================
header("Demo 8 — Multi-modal Pooling: retina + tactile")
print(textwrap.dedent("""
  The 'above' concept (艮 scan of whole field) sees both:
    Retina:  top_bright field, weight=1.0  → 艮 scores ~0.9
    Tactile: bottom_bright field, weight=0.5 → 艮 scores ~0.367

  Pooled result = (1.0 × 0.9 + 0.5 × 0.367) / 1.5 ≈ 0.722
"""))

def build_multimodal_above():
    g   = ConceptGraph("above_mm")
    par = ParallelNode("par")
    sc  = ObjectNode("sc", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, name="top")
    ret = RetinaNode("ret",  field=field("top_bright"),    initial_weight=1.0)
    tac = TactileNode("tac", field=field("bottom_bright"), initial_weight=0.5)
    g.add_nodes([par, sc, ret, tac])
    g.add_solid_edge(par, ret); g.add_solid_edge(par, tac); g.add_solid_edge(par, sc)
    g.root = par; return g

mm = build_multimodal_above()
act_mm = mm.evaluate()
result("retina(top_bright,w=1) + tactile(bot_bright,w=0.5)", act_mm, "expected ≈0.722")
section("Verify individual modality scores:")
ab2 = single(TRIGRAM_PATTERNS["top"], Orientation.VERTICAL, "top")
result("  top(艮) | top_bright (retina alone)", ab2.evaluate_against(field("top_bright")))
result("  top(艮) | bot_bright (tactile alone)", ab2.evaluate_against(field("bottom_bright")))
print(f"  Weighted avg = (1.0×{ab2.evaluate_against(field('top_bright')):.3f} + "
      f"0.5×{ab2.evaluate_against(field('bottom_bright')):.3f}) / 1.5 = {act_mm:.3f}")


# =============================================================
# Demo 9 — CONCEPT CLASSIFIER
# =============================================================
header("Demo 9 — Concept Classifier: which schema best matches?")
print(textwrap.dedent("""
  Five concepts scored against five fields.
  A good lexicon produces unambiguous winners.
"""))

concepts = {
    "above  (top/艮 V)" : single(TRIGRAM_PATTERNS["top"], Orientation.VERTICAL,   "top-V"),
    "below  (bottom/震 V)" : single(TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL,   "bottom-V"),
    "left   (bottom/震 H)" : single(TRIGRAM_PATTERNS["bottom"], Orientation.HORIZONTAL, "left"),
    "right  (top/艮 H)" : single(TRIGRAM_PATTERNS["top"], Orientation.HORIZONTAL, "right"),
    "center (坎)"  : build_center(),
    "all    (universal/乾)"  : single(TRIGRAM_PATTERNS["universal"], Orientation.VERTICAL,   "universal-V"),
}

test_cases = {
    "top_bright   ": "top_bright",
    "bottom_bright": "bottom_bright",
    "left_bright  ": "left_bright",
    "right_bright ": "right_bright",
    "center_bright": "center_bright",
    "uniform_high ": "uniform_high",
}

for flabel, recipe in test_cases.items():
    section(f"Field: {flabel.strip()}")
    scores = {n: g.evaluate_against(field(recipe)) for n, g in concepts.items()}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (n, score) in enumerate(ranked):
        result(f"  {n}", score, "← BEST" if rank == 0 else "")

print()
print("=" * 62)
print("  All demos complete.")
print("=" * 62)
print()