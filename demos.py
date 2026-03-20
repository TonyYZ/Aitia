"""
demos.py — Concept Representation Demos
========================================

Illustrates how to build ConceptGraphs for spatial image schemas and verify
that evaluations behave as intended under the current model.

Run with:  python demos.py

Current model summary
---------------------
ObjectNode: only yang (>0) bands contribute to scanning.
    activation = value × mean(1 - |field_band_i - 1| for yang bands only)
    Yin bands are passive — they do not scan at all.
    VOID (all-zero) returns 0 for the same reason: no yang bands.

Node.value (default 1.0): set by incoming dashed edges.
    value = mean(source.activation for all incoming dashed edges)
    Governs Plan/Body replacement probability and ObjectNode output scaling.

DashedEdge: no gain parameter. Simply sets target.value = source.activation
    (or mean of multiple sources).

Plan/Body replacement rule: for template T (value p) and content C:
    activation = p × yang_scan(T, C, field)
               + (1-p) × C.evaluate(full_field)
    yang_scan = mean of C evaluated on each yang sub-region of T.
    All-yin template → yang_scan = 0 → result = (1-p) × full_scan.

SerialNode: averages child activations (no longer min).

ReturnNode: output sink, placed as sibling of scanners inside a ParallelNode.
    The ParallelNode writes its scanner average into the ReturnNode.
    ConceptGraph.evaluate() returns the last ReturnNode's activation.

Evaluation: three passes — (1) scan with values=1, (2) update values from
    dashed edges, (3) re-scan with updated values.
"""

import numpy as np
import textwrap
from embodied_lot import (
    ConceptGraph, ObjectNode, SerialNode, ParallelNode, BodyNode, PlanNode,
    ReceptiveFieldNode, RetinaNode, TactileNode,
    ReturnNode, ReceptiveField, Orientation,
    TRIGRAM_PATTERNS,
)
from trigrams import (
    TOP, BOTTOM, LEFT, RIGHT, CENTER_V, CENTER_H,
    UNIVERSAL, VOID, PERIPHERY_V, MAJORITY_UPPER, MAJORITY_LOWER,
    NEUTRAL, V, H,
)


# ── Display helpers ───────────────────────────────────────────────────

def header(title):
    print(); print("=" * 64); print(f"  {title}"); print("=" * 64)

def result(label, value, note=""):
    bar  = "█" * round(value * 20)
    pad  = "░" * (20 - round(value * 20))
    note = f"   ← {note}" if note else ""
    print(f"  {label:<40s}  {value:.3f}  |{bar}{pad}|{note}")

def section(text):
    print(f"\n  ── {text}")

def field(recipe):
    a = np.zeros((9, 9))
    if   recipe == "top":    a[:3,  :] = 0.9; a[3:,  :] = 0.1
    elif recipe == "bot":    a[6:,  :] = 0.9; a[:6,  :] = 0.1
    elif recipe == "left":   a[:,  :3] = 0.9; a[:, 3:]  = 0.1
    elif recipe == "right":  a[:, 6:]  = 0.9; a[:, :6]  = 0.1
    elif recipe == "center": a[:,   :] = 0.1; a[3:6, 3:6] = 0.9
    elif recipe == "uni":    a[:,   :] = 0.9
    elif recipe == "mid":    a[:,   :] = 0.5
    elif recipe == "low":    a[:,   :] = 0.1
    elif recipe == "outer":
        a[:,:]=0.1; a[0,:]=a[-1,:]=a[:,0]=a[:,-1]=0.9
    return ReceptiveField(a)

def single(pattern, ori, name=""):
    """Minimal concept: ParallelNode [retina] + one ObjectNode scanner."""
    g   = ConceptGraph(name or str(pattern))
    par = ParallelNode("par")
    sc  = ObjectNode("sc", pattern, ori, name=name or str(pattern))
    g.add_nodes([par, sc])
    g.add_solid_edge(par, sc)
    g.root = par
    return g


# =================================================================
# Demo 1 — Yang-only scanning: what the new ObjectNode does
# =================================================================
header("Demo 1 — Yang-only scanning")
print(textwrap.dedent("""
  Only yang (>0) positions in the pattern contribute to the activation.
  Yin positions are skipped entirely — they are passive.

  Convention (VERTICAL):
    pattern[0] = bottom band   pattern[2] = top band

  TOP  (0,0,1): only band[2] scans → detects brightness at top
  BOTTOM (1,0,0): only band[0] scans → detects brightness at bottom
  VOID (0,0,0): no yang bands → always returns 0

  Scoring for TOP on top-bright field (top-band mean ≈ 0.9):
    sim = 1 - |0.9 - 1| = 0.9
    activation = value(1.0) × 0.9 = 0.9
"""))

section("TOP (0,0,1) VERTICAL — detects figure at top")
top_g = single(TOP, V, "top")
for recipe, note in [
    ("top",  "top-band bright → HIGH"),
    ("bot",  "top-band dim    → LOW"),
    ("uni",  "top-band=0.9 but mid,bot yang? no — only band[2] counts → 0.9"),
    ("low",  "everything dim  → LOW"),
]:
    result(recipe, top_g.evaluate_against(field(recipe)), note)

section("BOTTOM (1,0,0) VERTICAL — detects figure at bottom")
bot_g = single(BOTTOM, V, "bottom")
for recipe, note in [
    ("bot",  "bottom-band bright → HIGH"),
    ("top",  "bottom-band dim    → LOW"),
]:
    result(recipe, bot_g.evaluate_against(field(recipe)), note)

section("VOID (0,0,0) — all yin, no yang bands → always 0")
void_g = single(VOID, V, "void")
for recipe in ["top", "uni"]:
    result(recipe, void_g.evaluate_against(field(recipe)), "always 0")

section("Contrast: UNIVERSAL (1,1,1) — all three bands scan")
uni_g = single(UNIVERSAL, V, "universal")
for recipe, note in [
    ("uni",  "all bands 0.9 → all sims 0.9 → mean 0.9"),
    ("top",  "band0=0.1(sim0.1) band1=0.1(sim0.1) band2=0.9(sim0.9) → mean 0.367"),
    ("mid",  "all bands 0.5 → mean sim 0.5"),
]:
    result(recipe, uni_g.evaluate_against(field(recipe)), note)


# =================================================================
# Demo 2 — Node.value scales activation
# =================================================================
header("Demo 2 — Node.value scales activation (set by dashed edges)")
print(textwrap.dedent("""
  When a dashed edge points to a node, its value becomes the source's
  activation.  The node's output is then scaled by its value:
      activation = value × yang_scan

  Example: UNIVERSAL scanner, gated by a BOTTOM condition.
    On top-bright field:  BOTTOM scanner fires low (~0.1)
                          → UNIVERSAL.value = 0.1
                          → UNIVERSAL.activation ≈ 0.1 × 0.367 ≈ 0.037
    On bot-bright field:  BOTTOM scanner fires high (~0.9)
                          → UNIVERSAL.value ≈ 0.9
                          → UNIVERSAL.activation ≈ 0.9 × 0.9 ≈ 0.810
"""))

def build_gated():
    g    = ConceptGraph("gated")
    par  = ParallelNode("par")
    cond = ObjectNode("cond", BOTTOM, V, name="bottom-cond")
    tgt  = ObjectNode("tgt",  UNIVERSAL, V, name="universal-tgt")
    g.add_nodes([par, cond, tgt])
    g.add_solid_edge(par, cond)
    g.add_solid_edge(par, tgt)
    g.add_dashed_edge(cond, tgt)   # tgt.value = cond.activation
    g.root = par
    return g

gated = build_gated()
for recipe, note in [
    ("top",  "BOTTOM cond low  → tgt suppressed"),
    ("bot",  "BOTTOM cond high → tgt active"),
    ("uni",  "BOTTOM cond=0.9  → tgt.value=0.9 → output 0.810"),
    ("low",  "both dim         → near zero"),
]:
    act  = gated.evaluate_against(field(recipe))
    cond_act = gated._nodes["cond"].activation
    tgt_val  = gated._nodes["tgt"].value
    result(recipe, act, f"cond={cond_act:.2f} → tgt.value={tgt_val:.2f} → act={act:.3f} {note}")


# =================================================================
# Demo 3 — PlanNode replacement rule
# =================================================================
header("Demo 3 — PlanNode: probabilistic spatial replacement rule")
print(textwrap.dedent("""
  PlanNode([template, content], field):
    activation = p × yang_scan + (1-p) × full_scan
    p = template.value (default 1.0)

  With p=1 (no dashed edge on template):
    activation = 1 × yang_scan + 0 × full_scan = yang_scan
    = content evaluated only in template's yang sub-regions.

  With p=0.8 (some dashed edge set template.value=0.8):
    activation = 0.8 × yang_scan + 0.2 × full_scan
    20% of the full-field scan "leaks through" even where template is yin.
"""))

section("TOP template (p=1) + UNIVERSAL content")
print("  Content constrained to the top-third sub-region of the field.")
print("  High when that region is bright; bottom brightness irrelevant.")

g3 = ConceptGraph("plan_top_uni")
par3  = ParallelNode("par")
plan3 = PlanNode("plan")
tmpl3 = ObjectNode("t", TOP, V, name="top-tmpl")
cont3 = ObjectNode("c", UNIVERSAL, V, name="uni-cont")
g3.add_nodes([par3, plan3, tmpl3, cont3])
g3.add_solid_edge(par3, plan3)
g3.add_solid_edge(plan3, tmpl3)
g3.add_solid_edge(plan3, cont3)
g3.root = par3

for recipe, note in [
    ("top",    "top zone bright → content sees bright region → HIGH"),
    ("bot",    "top zone dim    → content sees dim region   → LOW"),
    ("uni",    "top zone bright → HIGH (bottom irrelevant)"),
    ("center", "top zone dim    → LOW (center is mid-field)"),
]:
    result(recipe, g3.evaluate_against(field(recipe)), note)

section("TOP template (p=0.8) + UNIVERSAL content")
print("  80% constrained to top zone, 20% full-field scan leaks through.")
print("  On bot-bright field: top zone dim (0.1) but 20% of full scan adds back.")

g3b = ConceptGraph("plan_top_uni_partial")
par3b=ParallelNode("par"); plan3b=PlanNode("plan")
tmpl3b=ObjectNode("t",TOP,V,name="top-tmpl"); cont3b=ObjectNode("c",UNIVERSAL,V,name="uni-cont")
tmpl3b.value = 0.8   # manually set for illustration (normally via dashed edge)
g3b.add_nodes([par3b,plan3b,tmpl3b,cont3b])
g3b.add_solid_edge(par3b,plan3b); g3b.add_solid_edge(plan3b,tmpl3b); g3b.add_solid_edge(plan3b,cont3b)
g3b.root=par3b

# Note: evaluate() resets all values to 1.0 before pass 1,
# so we use _plan_activate directly to show p=0.8 arithmetic.
for recipe, note in [
    ("top", "0.8×0.9 + 0.2×UNIV(full)"),
    ("bot", "0.8×0.1 + 0.2×UNIV(full)"),
]:
    tmpl3b.value = 0.8
    result(recipe, plan3b._plan_activate([tmpl3b, cont3b], field(recipe)), note)


# =================================================================
# Demo 4 — PlanNode chaining: three children
# =================================================================
header("Demo 4 — PlanNode chaining: PERIPHERY → BOTTOM → UNIVERSAL")
print(textwrap.dedent("""
  Three-child plan: [PERIPHERY, BOTTOM, UNIVERSAL]

  Step 1: realize plan([BOTTOM, UNIVERSAL], field) for each yang
          sub-region of PERIPHERY.
  Step 2: PERIPHERY templates the result of step 1.

  PERIPHERY (1,0,1): yang at bottom and top thirds.
  BOTTOM (1,0,0):    yang at bottom third (of its sub-region).
  UNIVERSAL (1,1,1): scans all bands of its sub-region.

  With all values=1, realized pattern:
    PERIPHERY yang → bottom-third: BOTTOM scans it → its bottom-third
    PERIPHERY yang → top-third:    BOTTOM scans it → its bottom-third

  So the realized concept detects: bottom of the bottom-third
  AND bottom of the top-third (i.e. rows at 6-9 and 4-6 of a 9-row field).
"""))

g4 = ConceptGraph("chain3")
par4   = ParallelNode("par")
plan4  = PlanNode("plan")
peri   = ObjectNode("peri", PERIPHERY_V, V, name="periphery")
bot_n  = ObjectNode("bot",  BOTTOM,      V, name="bottom")
uni4   = ObjectNode("uni",  UNIVERSAL,   V, name="universal")
g4.add_nodes([par4, plan4, peri, bot_n, uni4])
g4.add_solid_edge(par4, plan4)
g4.add_solid_edge(plan4, peri)
g4.add_solid_edge(plan4, bot_n)
g4.add_solid_edge(plan4, uni4)
g4.root = par4

for recipe, note in [
    ("bot",    "bottom is bright → realized zone bright → HIGH"),
    ("top",    "bottom zones dim → LOW"),
    ("uni",    "all zones bright → HIGH"),
    ("low",    "all dim          → LOW"),
]:
    result(recipe, g4.evaluate_against(field(recipe)), note)


# =================================================================
# Demo 5 — ReturnNode
# =================================================================
header("Demo 5 — ReturnNode: named output portal")
print(textwrap.dedent("""
  A ReturnNode is placed as a child of a ParallelNode alongside scanners.
  The ParallelNode writes its scanner average into the ReturnNode.
  ConceptGraph.evaluate() returns the last ReturnNode's activation.

  This lets a concept have multiple named outputs, and separates the
  "answer" from intermediate computation.  The ReturnNode does NOT alter
  any computation — it is a passive marker.
"""))

g5 = ConceptGraph("return_demo")
par5 = ParallelNode("par")
sc5  = ObjectNode("sc", TOP, V, name="top-scanner")
ret5 = ReturnNode("ret")
g5.add_nodes([par5, sc5, ret5])
g5.add_solid_edge(par5, sc5)
g5.add_solid_edge(par5, ret5)
g5.root = par5

for recipe, note in [
    ("top", "top bright → scanner fires → ReturnNode receives result"),
    ("bot", "top dim    → scanner low   → ReturnNode receives low"),
]:
    act = g5.evaluate_against(field(recipe))
    result(recipe, act, f"ReturnNode.activation = {ret5.activation:.3f}  {note}")


# =================================================================
# Demo 6 — "above": the full concept using scan + plan + dashed edges
# =================================================================
header('Demo 6 — "above": full concept graph')
print(textwrap.dedent("""
  Cognitive structure of "above":
    Figure is present in the top region AND the bottom region is empty ground.

  Graph structure:
    par_root [retina]
      ├── ser (VERTICAL split: children[0]=bottom, children[1]=top)
      │     ├── yang_0  UNIVERSAL — scans bottom strip
      │     └── yang_1  UNIVERSAL — scans top strip
      └── par_out
            ├── plan
            │     ├── yin_0  VOID  — template (all-yin, blocks by default)
            │     └── yang_2 UNIVERSAL — content
            └── ReturnNode

  Dashed edges:
    yang_0 ⤳ yin_0   (bottom detection → yin template value)
    yang_1 ⤳ yang_2  (top detection → content value)

  Replacement rule for plan(yin_0, yang_2):
    yin_0 is all-yin → yang_scan = 0 always
    result = yin_0.value × 0 + (1 - yin_0.value) × yang_2.evaluate(field)
           = (1 - bottom_substance) × (top_substance × field_scan)

  → HIGH when top is bright (substance present) AND bottom is dim (empty ground).
  → LOW  when bottom is bright (ground occupied — so top is not "above" it).
  → LOW  when both empty (nothing to be above anything).
"""))

def build_above():
    g = ConceptGraph("above")
    par_root = ParallelNode("par_root")
    ser      = SerialNode("ser", split_orientation=Orientation.VERTICAL)
    yang_0   = ObjectNode("y0",  UNIVERSAL, V, name="bot-scanner")
    yang_1   = ObjectNode("y1",  UNIVERSAL, V, name="top-scanner")
    par_out  = ParallelNode("par_out")
    plan     = PlanNode("plan")
    yin_0    = ObjectNode("yn0", VOID, V, name="yin-template")
    yang_2   = ObjectNode("y2",  UNIVERSAL, V, name="content")
    ret      = ReturnNode("ret")

    g.add_nodes([par_root, ser, yang_0, yang_1,
                 par_out, plan, yin_0, yang_2, ret])

    g.add_solid_edge(par_root, ser)
    g.add_solid_edge(ser,      yang_0)    # children[0] = bottom strip
    g.add_solid_edge(ser,      yang_1)    # children[1] = top strip
    g.add_solid_edge(par_root, par_out)
    g.add_solid_edge(par_out,  plan)
    g.add_solid_edge(par_out,  ret)
    g.add_solid_edge(plan,     yin_0)     # template
    g.add_solid_edge(plan,     yang_2)    # content

    g.add_dashed_edge(yang_0, yin_0)      # bottom scan sets template gate
    g.add_dashed_edge(yang_1, yang_2)     # top scan sets content gate

    g.root = par_root
    return g

above = build_above()

section("Results with trace")
for recipe, note in [
    ("top",  "top bright, bottom dim   — above TRUE"),
    ("bot",  "bottom bright, top dim   — above FALSE"),
    ("uni",  "both bright              — ambiguous/low (ground occupied)"),
    ("low",  "both empty               — nothing to be above anything"),
    ("mid",  "uniform 0.5             — partial everywhere"),
]:
    act = above.evaluate_against(field(recipe))
    y0  = above._nodes["y0"];  y1  = above._nodes["y1"]
    yn  = above._nodes["yn0"]; y2  = above._nodes["y2"]
    pl  = above._nodes["plan"]
    print(f"  {recipe}:  bot={y0.activation:.2f}  top={y1.activation:.2f}  "
          f"yin.v={yn.value:.2f}  cont.v={y2.value:.2f}  "
          f"plan={pl.activation:.3f}  final={act:.3f}   {note}")


# =================================================================
# Demo 7 — Multi-modal pooling
# =================================================================
header("Demo 7 — Multi-modal pooling: retina + tactile")
print(textwrap.dedent("""
  The same TOP scanner evaluates both modalities simultaneously.
    Retina  (visual): top-bright field, weight=1.0 → TOP activation ≈ 0.9
    Tactile (haptic): bot-bright field, weight=0.5 → TOP activation ≈ 0.1

  Pooled result = (1.0 × 0.9 + 0.5 × 0.1) / 1.5 ≈ 0.633
"""))

g7   = ConceptGraph("above_mm")
par7 = ParallelNode("par")
sc7  = ObjectNode("sc", TOP, V, name="top-sc")
ret7 = RetinaNode("ret",  field=field("top"),  initial_weight=1.0)
tac7 = TactileNode("tac", field=field("bot"),  initial_weight=0.5)
g7.add_nodes([par7, sc7, ret7, tac7])
g7.add_solid_edge(par7, ret7)
g7.add_solid_edge(par7, tac7)
g7.add_solid_edge(par7, sc7)
g7.root = par7

act7 = g7.evaluate()
result("retina(top,w=1) + tactile(bot,w=0.5)", act7, "expected ≈ 0.633")

section("Verify individual scores:")
g7v = single(TOP, V, "top")
result("  TOP | top-bright alone", g7v.evaluate_against(field("top")))
result("  TOP | bot-bright alone", g7v.evaluate_against(field("bot")))
print(f"  Weighted avg = (1.0×0.900 + 0.5×0.100) / 1.5 = {(0.9+0.05)/1.5:.3f}")


# =================================================================
# Demo 8 — Concept classifier
# =================================================================
header("Demo 8 — Concept classifier: which schema best matches?")
print(textwrap.dedent("""
  Each concept is a single ObjectNode scanning the full field.
  The trigram whose yang-band pattern best matches the field's
  spatial distribution scores highest.
"""))

concepts = {
    "top      (0,0,1)": single(TOP,            V, "top"),
    "bottom   (1,0,0)": single(BOTTOM,         V, "bottom"),
    "left     (1,0,0)": single(LEFT,           H, "left"),
    "right    (0,0,1)": single(RIGHT,          H, "right"),
    "center-V (0,1,0)": single(CENTER_V,       V, "center-V"),
    "universal(1,1,1)": single(UNIVERSAL,      V, "universal"),
    "upper    (0,1,1)": single(MAJORITY_UPPER, V, "upper"),
    "lower    (1,1,0)": single(MAJORITY_LOWER, V, "lower"),
}

test_fields = {
    "top-bright   ": "top",
    "bot-bright   ": "bot",
    "left-bright  ": "left",
    "right-bright ": "right",
    "center-bright": "center",
    "uniform-high ": "uni",
}

for fname, recipe in test_fields.items():
    section(f"Field: {fname.strip()}")
    scores = {n: g.evaluate_against(field(recipe)) for n, g in concepts.items()}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    for rank, (n, score) in enumerate(ranked[:4]):   # show top 4
        result(f"  {n}", score, "← BEST" if rank == 0 else "")

print()
print("=" * 64)
print("  All demos complete.")
print("=" * 64)
print()