"""
trigrams.py — Semantic aliases for the trigram vocabulary
==========================================================

Re-exports the trigram patterns from embodied_lot under names that
reflect their spatial meaning rather than their Yijing character names.
This is the recommended import for experiment scripts.

The Chinese character names (艮, 震, …) remain available in embodied_lot
as comments cross-referencing the Yijing literature, but researchers
do not need to type or search for them in normal use.

Usage
-----
    from trigrams import TOP, BOTTOM, LEFT, RIGHT, CENTER_V, UNIVERSAL, V, H
    from embodied_lot import ConceptGraph, ObjectNode, ParallelNode

    g   = ConceptGraph("above")
    par = ParallelNode("par")
    sc  = ObjectNode("sc", TOP, V)
    g.add_nodes([par, sc])
    g.add_solid_edge(par, sc)
    g.root = par
"""

from embodied_lot import TRIGRAM_PATTERNS, Orientation

V = Orientation.VERTICAL    # bands: bottom → top
H = Orientation.HORIZONTAL  # bands: left   → right

# ── Vertical (and vertical-axis horizontal) trigrams ─────────────────

VOID        = TRIGRAM_PATTERNS["void"]       # (0,0,0)  background; does not scan
BOTTOM      = TRIGRAM_PATTERNS["bottom"]     # (1,0,0)  figure at bottom only
CENTER_V    = TRIGRAM_PATTERNS["center"]     # (0,1,0)  figure at center row
MAJORITY_LOWER = TRIGRAM_PATTERNS["lower"]  # (1,1,0)  figure in lower two-thirds
UNIVERSAL   = TRIGRAM_PATTERNS["universal"]  # (1,1,1)  figure everywhere
PERIPHERY_V = TRIGRAM_PATTERNS["periphery"]  # (1,0,1)  figure at outer rows
MAJORITY_UPPER = TRIGRAM_PATTERNS["upper"]  # (0,1,1)  figure in upper two-thirds
TOP         = TRIGRAM_PATTERNS["top"]        # (0,0,1)  figure at top only
NEUTRAL     = TRIGRAM_PATTERNS["neutral"]    # (0.5,…)  uniform prior

# ── Horizontal aliases (same patterns, intent made explicit) ──────────

LEFT           = TRIGRAM_PATTERNS["bottom"]     # (1,0,0)  figure at left only
CENTER_H       = TRIGRAM_PATTERNS["center"]     # (0,1,0)  figure at center column
MAJORITY_LEFT  = TRIGRAM_PATTERNS["lower"]      # (1,1,0)  figure in left two-thirds
RIGHT          = TRIGRAM_PATTERNS["top"]        # (0,0,1)  figure at right only
PERIPHERY_H    = TRIGRAM_PATTERNS["periphery"]  # (1,0,1)  figure at outer columns
MAJORITY_RIGHT = TRIGRAM_PATTERNS["upper"]      # (0,1,1)  figure in right two-thirds

# ── Sanity checks ─────────────────────────────────────────────────────

assert TOP            == (0, 0, 1)
assert BOTTOM         == (1, 0, 0)
assert MAJORITY_UPPER == (0, 1, 1), "巽: yang in mid+top"
assert MAJORITY_LOWER == (1, 1, 0), "兑: yang in bot+mid"
assert LEFT  == BOTTOM
assert RIGHT == TOP
