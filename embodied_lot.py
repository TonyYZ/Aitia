"""
embodied_lot.py — Embodied Language of Thought: Core Model
===========================================================

A symbolic-spatial cognitive model for concept learning and composition,
grounded in embodied cognition theory and Yijing trigrams as image-schema
primitives.

Theoretical background
----------------------
Concepts are *lazy scanner functions*: structured programs that scan a
spatial receptive field only when given perceptual input.  Primitive
features are drawn from the Yijing trigram system, whose binary line
structure naturally encodes spatial distributions.

The model distinguishes three kinds of things:

    Object nodes      Primitive scanners; each encodes one trigram schema.
    Structure nodes   Composite scanners built from primitives:
                        SerialNode   — spatial adjacency / AND
                        ParallelNode — alpha-blend / OR  [also the context hub]
                        BodyNode     — template unfolding with spatial memory
                        PlanNode     — static template (BodyNode, no self-motion)
    Field nodes       Perceptual input providers (RetinaNode, TactileNode, …).
                      A field node lives *inside* the graph as a child of a
                      ParallelNode; it supplies the spatial substrate that
                      activates its scanner siblings.

Lazy evaluation semantics
--------------------------
Every node is a *lambda function waiting for context*.  A scanner node only
fires — i.e. actually reads a receptive field — when it is a sibling of a
ReceptiveFieldNode inside a ParallelNode.  Without that co-presence, it
remains inert (activation 0).

When a ParallelNode holds both field nodes and scanner nodes as children:
  1. Each scanner is evaluated against each field node's spatial data.
  2. Results are weighted by the field node's current activation weight
     (its "degree of presence"), yielding a weighted average.
  3. Field node activations can themselves be modulated by dashed edges,
     so conditional top-down attention changes *which* field dominates.

This design means the same concept graph can be evaluated against vision,
touch, or any other modality simply by attaching the appropriate field node.

Activation as probability
--------------------------
Every node's `activation` is a probability p ∈ [0, 1] that its yang lines
are currently scanning.  We *never sample* — the system always stores and
manipulates the probability distribution.  Dashed-edge modulation is
deterministic and multiplicative:

    target.activation  ←  target.activation × (source.activation × gain)

The Bayesian learning component (see concept_learning.py) is the only
process that structurally modifies the graph.

Edge types
----------
SolidEdge   Compositional hierarchy: parent StructureNode → child Node.
DashedEdge  Conditional modulation: condition Node → target Node.
            Evaluated in strict topological order along the dashed-edge chain
            so that A → B → C chains propagate correctly.

Graph structure
---------------
Local field nodes in a nested ParallelNode *override* the field context
inherited from a parent SerialNode.  This gives each ParallelNode a clean,
independent scanning context, and allows a concept to simultaneously contain
sub-concepts that scan different modalities.

Quick-start example
-------------------
    from embodied_lot import (
        ConceptGraph, ObjectNode, SerialNode, ParallelNode,
        RetinaNode, Orientation, TRIGRAM_PATTERNS,
        ReceptiveField,
    )
    import numpy as np

    # ── "A is above B": two trigrams arranged vertically ──────────────
    graph = ConceptGraph(name="above")

    top    = ObjectNode("top", TRIGRAM_PATTERNS["bottom"],
                        orientation=Orientation.VERTICAL)
    bot    = ObjectNode("bot", TRIGRAM_PATTERNS["top"],
                        orientation=Orientation.VERTICAL)
    serial = SerialNode("serial", split_orientation=Orientation.VERTICAL)
    par    = ParallelNode("par")

    # Build the field-context hub: retina node + serial scanner as siblings
    retina = RetinaNode("retina", field=ReceptiveField.random(9, 9, seed=0))

    graph.add_nodes([top, bot, serial, par, retina])
    graph.add_solid_edge(par, retina)   # field node — provides context
    graph.add_solid_edge(par, serial)   # scanner node — fires when retina present
    graph.add_solid_edge(serial, top)
    graph.add_solid_edge(serial, bot)
    graph.root = par

    activation = graph.evaluate()   # float in [0, 1]
    print(f"above activation: {activation:.3f}")
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================
# Section 1 — Trigram System
# ============================================================

class Orientation(Enum):
    """Axis along which a receptive field or scanner is oriented."""
    HORIZONTAL = 0   # bands / strips run left → right
    VERTICAL   = 1   # bands / strips run bottom → top


# Binary pattern for each Yijing trigram ─────────────────────────────
# Index 0 = top band, 1 = middle band, 2 = bottom band.
# 0 = yin line (background / absent), 1 = yang line (figure / present).
TRIGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "void":    (0,   0,   0  ),   # 坤 Kūn  — all yin; pure ground
    "top":     (0,   0,   1  ),   # 艮 Gèn  — mountain; figure at top
    "center":  (0,   1,   0  ),   # 坎 Kǎn  — water; figure at center
    "upper":   (0,   1,   1  ),   # 巽 Xùn  — wind; majority upper
    "bottom":  (1,   0,   0  ),   # 震 Zhèn — thunder; figure at bottom
    "periphery":(1,  0,   1  ),   # 离 Lí   — fire; figure at periphery
    "lower":   (1,   1,   0  ),   # 兑 Duì  — lake; majority lower
    "universal":(1,  1,   1  ),   # 乾 Qián — heaven; all yang
    "neutral": (0.5, 0.5, 0.5),   # 中 Zhōng — neutral / unknown
}

BIGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "tai_yin":  (0, 0),   # 太阴
    "shao_yin": (0, 1),   # 少阴
    "shao_yang":(1, 0),   # 少阳
    "tai_yang": (1, 1),   # 太阳
}

MONOGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "yin":  (0,),   # 阴
    "yang": (1,),   # 阳
}

ALL_PATTERNS: Dict[str, Tuple[float, ...]] = {
    **TRIGRAM_PATTERNS, **BIGRAM_PATTERNS, **MONOGRAM_PATTERNS,
}
PATTERN_TO_NAME: Dict[Tuple, str] = {v: k for k, v in ALL_PATTERNS.items()}

# Expected motion vectors for dynamic (勢 / 動) trigrams ──────────────
# Format: (direction_vec, magnitude_vec)
#   direction: (upper_half_dir, lower_half_dir) ∈ {-1, 0, 1}
#   magnitude: (upper_half_mag, lower_half_mag) ∈ [0, 1]
TRIGRAM_MOTION: Dict[str, Tuple[Tuple, Tuple]] = {
    "void":     ((0,  0),  (0,   0  )),
    "top":      ((-1, -1), (1,   1  )),
    "center":   ((-1,  1), (1,   1  )),
    "upper":    ((-1,  0), (1,   0  )),
    "bottom":   (( 1,  1), (1,   1  )),
    "periphery":(( 1, -1), (1,   1  )),
    "lower":    (( 0,  1), (0,   1  )),
    "universal":(( 0,  0), (1,   1  )),
    "neutral":  (( 0,  0), (0.5, 0.5)),
}

# Unicode display symbols ─────────────────────────────────────────────
TRIGRAM_SYMBOLS: Dict[Tuple, str] = {
    (0, 0, 0)      : "☷", (0, 0, 1): "☶", (0, 1, 0): "☵",
    (0, 1, 1)      : "☴", (1, 0, 0): "☳", (1, 0, 1): "☲",
    (1, 1, 0)      : "☱", (1, 1, 1): "☰", (0.5, 0.5, 0.5): "𝌀",
}


def trigram_symbol(pattern: tuple) -> str:
    return TRIGRAM_SYMBOLS.get(pattern, "?")


# ============================================================
# Section 2 — Receptive Field (spatial data container)
# ============================================================

class ReceptiveField:
    """
    A 2D spatial field with an optional motion channel.

    This is the *data payload* carried by a ReceptiveFieldNode.  Values
    represent feature salience at each position.  In future work this can
    be derived from image features, neural activations, or semantic
    embeddings.

    Parameters
    ----------
    activation : np.ndarray, shape (H, W)
        Static activation / salience in [0, 1].
    motion : np.ndarray, shape (H, W, 2), optional
        Per-pixel velocity (dy, dx).  Used by dynamic ObjectNodes.
    """

    def __init__(
        self,
        activation: np.ndarray,
        motion: Optional[np.ndarray] = None,
    ) -> None:
        if activation.ndim != 2:
            raise ValueError("activation must be 2-D (H, W).")
        if not (0 <= activation.min() and activation.max() <= 1 + 1e-6):
            warnings.warn("Clipping activation values to [0, 1].")
        self.activation = np.clip(activation, 0.0, 1.0).astype(float)
        self.motion     = motion

    @classmethod
    def from_array(cls, arr: np.ndarray, motion=None) -> "ReceptiveField":
        lo, hi = arr.min(), arr.max()
        return cls(np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1), motion)

    @classmethod
    def uniform(cls, height: int, width: int, value: float = 0.5) -> "ReceptiveField":
        return cls(np.full((height, width), value))

    @classmethod
    def random(cls, height: int, width: int,
               seed: Optional[int] = None) -> "ReceptiveField":
        return cls(np.random.default_rng(seed).random((height, width)))

    @property
    def height(self) -> int:
        return self.activation.shape[0]

    @property
    def width(self) -> int:
        return self.activation.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.activation.shape

    def subfield(
        self,
        row_start: int, row_end: int,
        col_start: int, col_end: int,
    ) -> "ReceptiveField":
        """Crop a rectangular sub-field."""
        sub_act = self.activation[row_start:row_end, col_start:col_end]
        sub_mot = (
            self.motion[row_start:row_end, col_start:col_end]
            if self.motion is not None else None
        )
        return ReceptiveField(sub_act, sub_mot)

    def __repr__(self) -> str:
        return f"ReceptiveField(shape={self.shape}, mean={self.activation.mean():.3f})"


# ============================================================
# Section 3 — Edge Types
# ============================================================

class EdgeType(Enum):
    SOLID  = "solid"
    DASHED = "dashed"


class SolidEdge:
    """
    Compositional (solid) edge: parent StructureNode → child Node.

    Encodes the spatial / logical hierarchy.  Children receive sub-regions
    of their parent's active field (SerialNode), or the same full field
    (ParallelNode), or a body-internal subfield (BodyNode / PlanNode).
    """

    def __init__(self, parent: "Node", child: "Node") -> None:
        self.source    = parent
        self.target    = child
        self.edge_type = EdgeType.SOLID

    def __repr__(self) -> str:
        return f"SolidEdge({self.source.node_id!r} → {self.target.node_id!r})"


class DashedEdge:
    """
    Conditional (dashed) edge: condition Node ⤳  target Node.

    After a node is evaluated, any dashed edges *from* that node scale
    the target's activation:

        target.activation  ←  target.activation  ×  clamp(source.activation × gain)

    Crucially, dashed edges are applied in topological order along the
    dashed-edge chain, so A ⤳ B ⤳ C propagates correctly.

    Dashed edges can point to any node — including ReceptiveFieldNodes —
    allowing top-down attention to reweight which sensory modality dominates
    a given scanning context.

    Parameters
    ----------
    condition : Node
        The conditioning node.
    target : Node
        The node whose activation is multiplicatively gated.
    gain : float
        Weight in the gating formula (default 1.0).
    """

    def __init__(
        self, condition: "Node", target: "Node", gain: float = 1.0
    ) -> None:
        self.source    = condition
        self.target    = target
        self.gain      = gain
        self.edge_type = EdgeType.DASHED

    def __repr__(self) -> str:
        return (
            f"DashedEdge({self.source.node_id!r} ⤳  {self.target.node_id!r}, "
            f"gain={self.gain:.2f})"
        )


# ============================================================
# Section 4 — Node Base Class
# ============================================================

class Node(ABC):
    """
    Abstract base for all nodes in a ConceptGraph.

    Every node is a *lazy scanner function*.  It holds its most recently
    computed activation and a list of solid-edge children.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id   = node_id
        self.activation: float = 0.0
        self.children: List["Node"] = []

    @abstractmethod
    def evaluate(
        self,
        inherited_field: Optional[ReceptiveField] = None,
    ) -> float:
        """
        Compute activation, given an optional inherited field context.

        Parameters
        ----------
        inherited_field : ReceptiveField or None
            The field passed down from a parent SerialNode or BodyNode.
            If None, no field context exists from above.

        Returns
        -------
        float
            Activation in [0, 1].  Returns 0 if no field context is
            available (lazy semantics).
        """
        ...

    @property
    @abstractmethod
    def node_type(self) -> str:
        """Short string identifier for display and serialisation."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node_id!r}, act={self.activation:.3f})"


# ============================================================
# Section 5 — Receptive Field Nodes (perceptual input providers)
# ============================================================

class ReceptiveFieldNode(Node):
    """
    A field node: provides spatial input data as a child of a ParallelNode.

    Field nodes are *not* scanners; they are the *context* that activates
    scanners.  When a ParallelNode finds a ReceptiveFieldNode among its
    children, it passes that node's spatial field to all sibling scanners.

    A field node's `activation` (default 1.0) represents the degree to
    which this modality is currently present.  It is used as a weight in
    the cross-field average inside its parent ParallelNode, and can be
    modulated by dashed edges to implement top-down attentional gating.

    Subclasses
    ----------
    RetinaNode   — visual / image features
    TactileNode  — haptic / touch features
    ProprioNode  — proprioceptive / body-position features

    Custom modalities can be added by subclassing ReceptiveFieldNode and
    setting a `modality` string.

    Parameters
    ----------
    node_id : str
        Unique identifier.
    field : ReceptiveField or None
        The spatial data this node provides.  Can be updated between
        evaluation calls to model dynamic sensory input.
    modality : str
        Human-readable modality label.
    initial_weight : float
        Initial activation / presence weight in [0, 1].
    """

    modality: str = "generic"

    def __init__(
        self,
        node_id          : str,
        field            : Optional[ReceptiveField] = None,
        modality         : Optional[str] = None,
        initial_weight   : float = 1.0,
    ) -> None:
        super().__init__(node_id)
        self.field      = field
        self.activation = float(np.clip(initial_weight, 0.0, 1.0))
        if modality is not None:
            self.modality = modality

    @property
    def node_type(self) -> str:
        return f"field:{self.modality}"

    def evaluate(
        self, inherited_field: Optional[ReceptiveField] = None
    ) -> float:
        """
        Field nodes return their weight unchanged; they do not scan.
        Their role is as context providers, not as scanners.
        """
        return self.activation

    def __repr__(self) -> str:
        field_info = f"shape={self.field.shape}" if self.field else "no field"
        return (
            f"ReceptiveFieldNode(id={self.node_id!r}, "
            f"modality={self.modality!r}, weight={self.activation:.3f}, "
            f"{field_info})"
        )


class RetinaNode(ReceptiveFieldNode):
    """Visual receptive field node (image / pixel features)."""
    modality = "retina"


class TactileNode(ReceptiveFieldNode):
    """Haptic / touch receptive field node."""
    modality = "tactile"


class ProprioNode(ReceptiveFieldNode):
    """Proprioceptive (body-position) receptive field node."""
    modality = "proprio"


# ============================================================
# Section 6 — Object Nodes (primitive trigram scanners)
# ============================================================

class ObjectNode(Node):
    """
    Primitive scanner: evaluates a receptive field using a trigram pattern.

    Cognitive role
    --------------
    Each ObjectNode encodes one image schema at the level of a single
    trigram.  When placed as a sibling of a ReceptiveFieldNode inside a
    ParallelNode, it fires and returns the probability that the field matches
    the expected spatial (or motion) distribution.

    Band reading conventions
    ------------------------
    Trigrams follow the Yijing convention — **bottom to top** vertically,
    **left to right** horizontally:

        VERTICAL   pattern[0] = bottom band   pattern[n-1] = top band
        HORIZONTAL pattern[0] = left band     pattern[n-1] = right band

    Examples:
        震 (1,0,0) VERTICAL   → bottom yang, middle yin, top yin  ("below")
        艮 (0,0,1) VERTICAL   → bottom yin,  middle yin, top yang ("above")
        震 (1,0,0) HORIZONTAL → left yang, center yin, right yin  ("left-of")

    Background and void patterns
    ----------------------------
    Any all-zero pattern — 坤 (0,0,0), bigram (0,0), or monogram 阴 (0,) —
    is **background / void**: it does not scan and always returns 0.
    In Yijing, yin is passive and receptive; it is the *absence* of scanning.
    Dimness at a location is captured implicitly when a yang-scanner there
    returns a low value, not by an active yin-detector.

    中 (0.5,…,0.5) — neutral prior, returns 0.5.

    Scanning (static)
    -----------------
    The field is divided into `n` equal bands.  For each pattern position i:

        sim_i = 1 − |mean_field_band_i − pattern[i]|

    activation = mean(sim_0, …, sim_{n-1})

    Scanning (dynamic, is_dynamic=True)
    ------------------------------------
    Scans the *motion* channel instead of activation, using TRIGRAM_MOTION.

    Lazy semantics
    --------------
    Returns 0 without scanning if `inherited_field` is None.

    Parameters
    ----------
    node_id : str
    pattern : tuple of float
        Trigram / bigram / monogram pattern.
    orientation : Orientation
        VERTICAL (default) = bands bottom→top.
        HORIZONTAL         = bands left→right.
    is_dynamic : bool
        If True, scans the motion channel.
    name : str, optional
        Human-readable pattern name; inferred from pattern if omitted.
    """

    def __init__(
        self,
        node_id     : str,
        pattern     : Tuple[float, ...],
        orientation : Orientation  = Orientation.VERTICAL,
        is_dynamic  : bool         = False,
        name        : Optional[str] = None,
    ) -> None:
        super().__init__(node_id)
        self.pattern     = tuple(pattern)
        self.orientation = orientation
        self.is_dynamic  = is_dynamic
        self.name        = name or PATTERN_TO_NAME.get(self.pattern, str(pattern))

    @property
    def node_type(self) -> str:
        return "object"

    @property
    def n_bands(self) -> int:
        return len(self.pattern)

    # ── Internal ─────────────────────────────────────────────────────

    def _band_means(self, field: ReceptiveField) -> List[float]:
        """
        Return the mean activation of each band, respecting reading direction:

        VERTICAL   pattern[0] ↔ bottom rows  (reversed from numpy row order)
        HORIZONTAL pattern[0] ↔ left columns (same as numpy column order)
        """
        n     = self.n_bands
        means = []

        if self.orientation == Orientation.VERTICAL:
            total = field.height
            for i in range(n):
                # pattern[0] = bottom → maps to the last row-block in numpy.
                field_band_from_top = n - 1 - i
                r0 = round(total * field_band_from_top / n)
                r1 = round(total * (field_band_from_top + 1) / n)
                if r1 <= r0:
                    means.append(0.0); continue
                band = field.activation[r0:r1, :]
                if self.is_dynamic and field.motion is not None:
                    band = np.abs(field.motion[r0:r1, :, 0])  # dy
                means.append(float(band.mean()) if band.size > 0 else 0.0)

        else:  # HORIZONTAL: left-to-right matches numpy column order
            total = field.width
            for i in range(n):
                c0 = round(total * i / n)
                c1 = round(total * (i + 1) / n)
                if c1 <= c0:
                    means.append(0.0); continue
                band = field.activation[:, c0:c1]
                if self.is_dynamic and field.motion is not None:
                    band = np.abs(field.motion[:, c0:c1, 1])  # dx
                means.append(float(band.mean()) if band.size > 0 else 0.0)

        return means

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(
        self, inherited_field: Optional[ReceptiveField] = None
    ) -> float:
        if inherited_field is None:
            self.activation = 0.0
            return 0.0

        # All-zero patterns (坤, 太阴, 阴, …) — background / void, do not scan.
        if all(v == 0 for v in self.pattern):
            self.activation = 0.0
            return 0.0

        # Neutral 中 — uniform prior.
        if all(v == 0.5 for v in self.pattern):
            self.activation = 0.5
            return 0.5

        band_means   = self._band_means(inherited_field)
        similarities = [
            1.0 - abs(obs - exp)
            for obs, exp in zip(band_means, self.pattern)
        ]
        print(similarities)
        self.activation = float(np.mean(similarities))
        return self.activation

    def __repr__(self) -> str:
        sym = trigram_symbol(self.pattern)
        dyn = "動" if self.is_dynamic else "靜"
        ori = "V"  if self.orientation == Orientation.VERTICAL else "H"
        return f"ObjectNode(id={self.node_id!r}, {sym}{self.name}{dyn} {ori})"


# ============================================================
# Section 7 — Structure Nodes
# ============================================================

class StructureNode(Node, ABC):
    """Abstract base for nodes that compose child activations."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)


class ParallelNode(StructureNode):
    """
    Parallel (并) node — the field-context hub and alpha-blend compositor.

    This node plays two intertwined roles:

    1. **Context hub**: any ReceptiveFieldNode children supply spatial data
       that activates sibling scanner nodes.  Field nodes from different
       modalities (retina, tactile, …) each carry an activation weight that
       represents their current degree of presence.

    2. **Alpha-blend compositor**: scanner children are each evaluated
       against the pool of available fields (both local and inherited),
       weighted by each field's presence weight, and the results are averaged
       across scanners.

    Field context pooling
    ---------------------
    The node collects *all* fields that are currently active:

      a) **Local field nodes**: ReceptiveFieldNode children attached directly
         to this ParallelNode, weighted by their `activation` attribute.

      b) **Inherited field**: a field passed down from a parent SerialNode
         or BodyNode, weighted by `inherited_weight` (default 1.0).

    When both exist, they are pooled together in a single weighted average.
    This means a nested ParallelNode with its own retina sibling will blend
    its local visual signal with whatever spatial context was passed from
    above — rather than ignoring one or the other.

    If no field is available from either source, the node is inert
    (activation 0), preserving lazy semantics.

    Evaluation formula
    ------------------
    For each scanner s, across all active fields f_i with weight w_i:

        scanner_activation[s] = Σ_i  w_i × s.evaluate(f_i.field)
                                 ──────────────────────────────────
                                         Σ_i  w_i

    self.activation = mean(scanner_activation[s] for all scanners s)

    Parameters
    ----------
    node_id : str
    inherited_weight : float
        Weight assigned to any inherited field (default 1.0).  Can be set
        per-instance if you want inherited context to dominate less strongly
        than local field nodes.
    """

    def __init__(self, node_id: str, inherited_weight: float = 1.0) -> None:
        super().__init__(node_id)
        self.inherited_weight = float(inherited_weight)

    @property
    def node_type(self) -> str:
        return "parallel"

    def _partition_children(self):
        """Separate children into (field_nodes, scanner_nodes)."""
        field_nodes   = [c for c in self.children
                         if isinstance(c, ReceptiveFieldNode)]
        scanner_nodes = [c for c in self.children
                         if not isinstance(c, ReceptiveFieldNode)]
        return field_nodes, scanner_nodes

    def evaluate(
        self, inherited_field: Optional[ReceptiveField] = None
    ) -> float:
        if not self.children:
            self.activation = 0.0
            return 0.0

        field_nodes, scanner_nodes = self._partition_children()

        # ── Build the pooled field list ────────────────────────────────
        # Each entry is (ReceptiveField, weight).
        # Local field nodes contribute with their activation weight.
        # The inherited field (from a parent SerialNode) contributes with
        # inherited_weight.  Both pools are merged into a single average.
        pooled: List[Tuple[ReceptiveField, float]] = []

        for fn in field_nodes:
            if fn.field is not None and fn.activation > 1e-8:
                pooled.append((fn.field, fn.activation))

        if inherited_field is not None and self.inherited_weight > 1e-8:
            pooled.append((inherited_field, self.inherited_weight))

        if not pooled:
            # No direct field available.  Propagate to children anyway:
            # StructureNode children (e.g. nested ParallelNodes with their own
            # local field nodes) can activate themselves without an inherited
            # field.  ObjectNode children correctly return 0 (lazy semantics).
            child_acts = []
            for child in self.children:
                if not isinstance(child, ReceptiveFieldNode):
                    act = child.evaluate(inherited_field=None)
                    if act > 0:
                        child_acts.append(act)
            self.activation = float(np.mean(child_acts)) if child_acts else 0.0
            return self.activation

        if not scanner_nodes:
            # Only field nodes, no scanners — activation is the mean
            # presence weight of the fields (normalised to [0, 1])
            total_w = sum(w for _, w in pooled)
            self.activation = float(
                np.clip(total_w / len(pooled), 0.0, 1.0)
            )
            return self.activation

        # ── Evaluate each scanner against the pooled fields ────────────
        total_w = sum(w for _, w in pooled)
        scanner_activations: List[float] = []

        for scanner in scanner_nodes:
            weighted_sum = 0.0
            for field, weight in pooled:
                scan_result   = scanner.evaluate(field)
                weighted_sum += weight * scan_result
            scanner_activations.append(weighted_sum / total_w)

        self.activation = float(np.mean(scanner_activations))
        return self.activation


class SerialNode(StructureNode):
    """
    Serial (串) node — spatial adjacency / AND composition.

    Divides its received field into `n` equal strips and evaluates each
    child on its allocated strip, returning the *minimum* activation (AND).

    Strip ordering convention
    -------------------------
    Follows the same reading direction as trigram patterns:

        VERTICAL   (default): children[0] = bottom strip, children[-1] = top
        HORIZONTAL           : children[0] = left strip,  children[-1] = right

    So for "A above B": children = [B_scanner, A_scanner] with VERTICAL split.
    (B is lower in the field → bottom strip → children[0].)

    Lazy semantics
    --------------
    Returns 0 if no field context is available.

    Parameters
    ----------
    node_id : str
    split_orientation : Orientation
    """

    def __init__(
        self,
        node_id           : str,
        split_orientation : Orientation = Orientation.VERTICAL,
    ) -> None:
        super().__init__(node_id)
        self.split_orientation = split_orientation

    @property
    def node_type(self) -> str:
        return "serial"

    def _make_strip(
        self, field: ReceptiveField, i: int, n: int
    ) -> ReceptiveField:
        """
        Return the strip for child i, respecting reading-direction convention.

        VERTICAL   children[0] = bottom strip → highest row indices in numpy.
        HORIZONTAL children[0] = left strip   → lowest column indices (unchanged).
        """
        if self.split_orientation == Orientation.VERTICAL:
            # child[0] = bottom = last row-block in numpy (row 0 = top of image)
            field_band_from_top = n - 1 - i
            r0 = round(field.height * field_band_from_top / n)
            r1 = round(field.height * (field_band_from_top + 1) / n)
            return field.subfield(r0, r1, 0, field.width)
        else:
            c0 = round(field.width * i / n)
            c1 = round(field.width * (i + 1) / n)
            return field.subfield(0, field.height, c0, c1)

    def evaluate(
        self, inherited_field: Optional[ReceptiveField] = None
    ) -> float:
        if not self.children or inherited_field is None:
            self.activation = 0.0
            return 0.0

        n           = len(self.children)
        activations = [
            child.evaluate(self._make_strip(inherited_field, i, n))
            for i, child in enumerate(self.children)
        ]
        self.activation = float(min(activations))
        return self.activation


class BodyNode(StructureNode):
    """
    Body (身) node — spatial unfolding template with persistent spatial state.

    Children are indexed 0 … n−1.  Each child i except the last serves as
    a *spatial template* that determines *where* child i+1 is evaluated
    within the body's subfield.  Only the last child contributes activation.

    Template rules
    --------------
    ObjectNode (static):  child i+1 is evaluated in the yang-band sub-regions
                          of child i's trigram pattern.
    ObjectNode (dynamic): child i+1 is evaluated in the half-field indicated
                          by child i's motion direction.
    StructureNode:        child i+1 receives the full current subfield
                          (fallback for non-leaf templates).

    Persistent state
    ----------------
    After evaluation, the final child's expected pattern is written into the
    body's internal grid `_grid`.  This accumulates spatial history across
    successive evaluation calls, modelling a motor routine that evolves over
    time.

    Field context
    -------------
    The body evaluates against the `inherited_field` it receives from its
    parent context.  The internal `_grid` records history but does not
    replace the incoming field.  If no field is available, the body is inert.

    Parameters
    ----------
    node_id : str
    subfield_shape : (H, W), optional
        Internal grid dimensions.  Defaults to the incoming field's shape.
    """

    def __init__(
        self,
        node_id        : str,
        subfield_shape : Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(node_id)
        self.subfield_shape = subfield_shape
        self._grid: Optional[np.ndarray] = None

    @property
    def node_type(self) -> str:
        return "body"

    def _ensure_grid(self, shape: Tuple[int, int]) -> None:
        H, W = shape
        if self._grid is None or self._grid.shape != (H, W):
            self._grid = np.zeros((H, W))

    def _yang_regions(
        self, template: ObjectNode, field: ReceptiveField
    ) -> List[ReceptiveField]:
        """
        Sub-regions corresponding to yang (active) bands of a static template,
        using the same bottom-to-top / left-to-right band convention as ObjectNode.
        """
        H, W    = field.shape
        n       = template.n_bands
        is_vert = template.orientation == Orientation.VERTICAL
        regions = []
        for i, bit in enumerate(template.pattern):
            if bit < 0.5:
                continue
            if is_vert:
                # pattern[0] = bottom → highest row indices in numpy
                field_band_from_top = n - 1 - i
                rs = round(H * field_band_from_top / n)
                re = round(H * (field_band_from_top + 1) / n)
                regions.append(field.subfield(rs, re, 0, W))
            else:
                cs = round(W * i / n); ce = round(W * (i + 1) / n)
                regions.append(field.subfield(0, H, cs, ce))
        return regions or [field]

    def _motion_region(
        self, template: ObjectNode, field: ReceptiveField
    ) -> ReceptiveField:
        """Sub-region in the motion direction of a dynamic template."""
        name = PATTERN_TO_NAME.get(template.pattern)
        dy, dx = TRIGRAM_MOTION.get(name, ((0, 0), (0, 0)))[0]
        H, W   = field.shape
        if dy > 0:   return field.subfield(H // 2, H, 0, W)
        elif dy < 0: return field.subfield(0, H // 2, 0, W)
        elif dx > 0: return field.subfield(0, H, W // 2, W)
        elif dx < 0: return field.subfield(0, H, 0, W // 2)
        return field

    def evaluate(
        self, inherited_field: Optional[ReceptiveField] = None
    ) -> float:
        if not self.children or inherited_field is None:
            self.activation = 0.0
            return 0.0

        target_shape = self.subfield_shape or inherited_field.shape
        self._ensure_grid(target_shape)

        # Adapt incoming field to declared body shape
        H, W = target_shape
        rh   = min(H, inherited_field.height)
        rw   = min(W, inherited_field.width)
        working_field = inherited_field.subfield(0, rh, 0, rw)

        if len(self.children) == 1:
            self.activation = self.children[0].evaluate(working_field)
            self._update_grid(self.children[0], working_field)
            return self.activation

        current_regions = [working_field]

        for i in range(len(self.children) - 1):
            template = self.children[i]
            next_regions: List[ReceptiveField] = []
            for region in current_regions:
                template.evaluate(region)
                if isinstance(template, ObjectNode):
                    if not template.is_dynamic:
                        next_regions.extend(self._yang_regions(template, region))
                    else:
                        next_regions.append(self._motion_region(template, region))
                else:
                    next_regions.append(region)
            current_regions = next_regions or [working_field]

        final_child = self.children[-1]
        acts        = [final_child.evaluate(r) for r in current_regions]
        self.activation = float(np.mean(acts))
        self._update_grid(final_child, working_field)
        return self.activation

    def _update_grid(self, final_child: Node, field: ReceptiveField) -> None:
        if not isinstance(final_child, ObjectNode) or self._grid is None:
            return
        H, W    = self._grid.shape
        n       = final_child.n_bands
        is_vert = final_child.orientation == Orientation.VERTICAL
        for i, bit in enumerate(final_child.pattern):
            if is_vert:
                field_band_from_top = n - 1 - i
                rs = round(H * field_band_from_top / n)
                re = round(H * (field_band_from_top + 1) / n)
                self._grid[rs:re, :] = bit
            else:
                cs = round(W * i / n); ce = round(W * (i + 1) / n)
                self._grid[:, cs:ce] = bit


class PlanNode(BodyNode):
    """
    Plan (阵) node — static spatial template.

    Identical to BodyNode in composition semantics.  The conceptual
    distinction: a PlanNode describes a *planned / anticipated* static
    arrangement; a BodyNode describes an active motor routine whose spatial
    state persists and moves across evaluation calls.
    """

    @property
    def node_type(self) -> str:
        return "plan"


# ============================================================
# Section 8 — Concept Graph
# ============================================================

class ConceptGraph:
    """
    A directed graph representing a concept as a structured composition of
    scanner nodes, field nodes, and conditional edges.

    Evaluation protocol
    -------------------
    1. **Dashed-edge pass** (topological order):
       Apply conditional modulation in dependency order.  Each dashed edge
       A ⤳ B multiplicatively scales B.activation by A.activation × gain.
       Field node activations are also modulated here, adjusting which
       sensory modalities dominate.

    2. **Solid-edge pass** (recursive, root-down):
       The root node triggers recursive evaluation.  ParallelNodes extract
       their ReceptiveFieldNode children and pass fields to scanner siblings.
       SerialNodes divide the inherited field.  BodyNodes apply template
       unfolding.

    3. **Re-aggregation pass** (root-up):
       Propagate post-modulation leaf changes back to the root using each
       StructureNode's own composition rule.

    Parameters
    ----------
    name : str
        Human-readable concept name.
    """

    def __init__(self, name: str = "") -> None:
        self.name                             = name
        self._nodes  : Dict[str, Node]        = {}
        self._solid  : List[SolidEdge]        = []
        self._dashed : List[DashedEdge]       = []
        self.root    : Optional[Node]         = None

    # ── Construction ──────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.node_id!r}.")
        self._nodes[node.node_id] = node

    def add_nodes(self, nodes: List[Node]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_solid_edge(self, parent: Node, child: Node) -> None:
        """Add a compositional edge and register child under parent."""
        self._solid.append(SolidEdge(parent, child))
        if child not in parent.children:
            parent.children.append(child)

    def add_dashed_edge(
        self, condition: Node, target: Node, gain: float = 1.0
    ) -> None:
        """Add a conditional modulation edge."""
        self._dashed.append(DashedEdge(condition, target, gain))

    # ── Topological order for dashed edges ────────────────────────────

    def _dashed_topological_order(self) -> List[Node]:
        """
        Return nodes in topological order considering only dashed edges.
        Nodes with no dashed-edge predecessors come first.
        Used to ensure A ⤳ B ⤳ C chains propagate correctly.
        """
        # Build in-degree count and adjacency via dashed edges
        in_degree: Dict[str, int]       = defaultdict(int)
        successors: Dict[str, List[Node]] = defaultdict(list)

        all_ids = set(self._nodes.keys())
        for edge in self._dashed:
            in_degree[edge.target.node_id] += 1
            successors[edge.source.node_id].append(edge.target)

        queue = deque(
            nid for nid in all_ids
            if in_degree[nid] == 0
        )
        order: List[Node] = []

        while queue:
            nid  = queue.popleft()
            node = self._nodes.get(nid)
            if node:
                order.append(node)
            for succ in successors[nid]:
                in_degree[succ.node_id] -= 1
                if in_degree[succ.node_id] == 0:
                    queue.append(succ.node_id)

        # Append any nodes unreachable via dashed edges (no dashed deps)
        seen = {n.node_id for n in order}
        for nid, node in self._nodes.items():
            if nid not in seen:
                order.append(node)

        return order

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self) -> float:
        """
        Compute the concept's activation.

        Field nodes must already be attached as children of the appropriate
        ParallelNodes before calling this method.  Use `attach_field()` or
        `evaluate_against()` for convenience.

        Evaluation order
        ----------------
        1. **Solid-edge pass** (root → leaves, recursive):
           The root node recursively evaluates its subtree.  ParallelNodes
           extract their ReceptiveFieldNode children, pass each field to
           scanner siblings, and compute the weighted average.  SerialNodes
           divide the inherited field among their children.

        2. **Dashed-edge pass** (topological order along dashed chains):
           After all scanner activations are computed, conditional edges
           scale target activations:
               target.activation ←  target.activation × clamp(source.activation × gain)
           Applied in strict topological order so A ⤳ B ⤳ C chains propagate
           correctly.  This includes field nodes — a dashed edge to a
           ReceptiveFieldNode reduces its presence weight for future cycles.

        Cycle semantics
        ---------------
        Dashed-edge modulations to leaf nodes update stored probabilities for
        the *current* result and carry forward into subsequent calls.  They do
        not retroactively alter a ParallelNode's weighted-average computation
        within the same evaluation step — that average was already computed
        correctly in pass 1.  This mirrors the cognitive intuition that
        top-down attention shapes what you perceive on the next look, not what
        you already saw.

        Returns
        -------
        float
            Root node activation in [0, 1].
        """
        if self.root is None:
            raise RuntimeError("ConceptGraph has no root node.")

        # Pass 1: solid-edge recursive evaluation (root → leaves)
        self.root.evaluate(inherited_field=None)

        # Pass 2: dashed-edge conditional modulation (topological order)
        if self._dashed:
            # Build lookup: target_id → incoming dashed edges
            dashed_to: Dict[str, List[DashedEdge]] = defaultdict(list)
            for edge in self._dashed:
                dashed_to[edge.target.node_id].append(edge)

            for node in self._dashed_topological_order():
                for edge in dashed_to.get(node.node_id, []):
                    scale = float(
                        np.clip(edge.source.activation * edge.gain, 0.0, 1.0)
                    )
                    node.activation = float(
                        np.clip(node.activation * scale, 0.0, 1.0)
                    )

        return float(np.clip(self.root.activation, 0.0, 1.0))

    # ── Field attachment helpers ───────────────────────────────────────

    def attach_field(
        self,
        field       : ReceptiveField,
        at_node_id  : Optional[str] = None,
        modality    : str = "retina",
        weight      : float = 1.0,
        field_node_id: Optional[str] = None,
    ) -> ReceptiveFieldNode:
        """
        Convenience method: create a field node and attach it as a child
        of the specified ParallelNode (or the root if it is a ParallelNode).

        Parameters
        ----------
        field : ReceptiveField
            The spatial data to attach.
        at_node_id : str, optional
            Node id of the target ParallelNode.  Defaults to the root.
        modality : str
            Modality label ('retina', 'tactile', etc.).
        weight : float
            Initial presence weight for this field node.
        field_node_id : str, optional
            Id for the new field node; auto-generated if omitted.

        Returns
        -------
        ReceptiveFieldNode
            The newly created and attached field node.
        """
        cls_map = {
            "retina" : RetinaNode,
            "tactile": TactileNode,
            "proprio": ProprioNode,
        }
        FieldCls = cls_map.get(modality, ReceptiveFieldNode)

        fid   = field_node_id or f"_field_{modality}_{len(self._nodes)}"
        fnode = FieldCls(fid, field=field, initial_weight=weight)

        target_id = at_node_id or (self.root.node_id if self.root else None)
        if target_id is None or target_id not in self._nodes:
            raise ValueError(
                f"No target node found for field attachment "
                f"(at_node_id={at_node_id!r})."
            )
        target = self._nodes[target_id]
        if not isinstance(target, ParallelNode):
            raise TypeError(
                f"Field nodes can only be attached to ParallelNodes; "
                f"'{target_id}' is a {type(target).__name__}."
            )

        self.add_node(fnode)
        self.add_solid_edge(target, fnode)
        return fnode

    def detach_field_nodes(self) -> None:
        """
        Remove all ReceptiveFieldNode children from the graph.
        Useful for resetting between evaluation calls.
        """
        field_ids = {
            n.node_id for n in self._nodes.values()
            if isinstance(n, ReceptiveFieldNode)
        }
        for fid in field_ids:
            node = self._nodes.pop(fid)
            for parent in self._nodes.values():
                if isinstance(parent, StructureNode) and node in parent.children:
                    parent.children.remove(node)
        self._solid = [
            e for e in self._solid
            if e.source.node_id not in field_ids
               and e.target.node_id not in field_ids
        ]

    def evaluate_against(
        self,
        field          : ReceptiveField,
        at_node_id     : Optional[str] = None,
        modality       : str = "retina",
        detach_after   : bool = True,
    ) -> float:
        """
        Convenience: temporarily attach a field node, evaluate, and
        optionally detach.  Useful for one-shot evaluation without
        permanently modifying the graph.

        Parameters
        ----------
        field : ReceptiveField
        at_node_id : str, optional
            Target ParallelNode id.  Defaults to root.
        modality : str
        detach_after : bool
            If True (default), removes the field node after evaluation.

        Returns
        -------
        float
        """
        fnode = self.attach_field(field, at_node_id=at_node_id,
                                  modality=modality)
        result = self.evaluate()
        if detach_after:
            self.detach_field_nodes()
        return result

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def solid_edges(self) -> List[SolidEdge]:
        return list(self._solid)

    @property
    def dashed_edges(self) -> List[DashedEdge]:
        return list(self._dashed)

    def depth(self) -> int:
        def _d(n: Node) -> int:
            return 0 if not n.children else 1 + max(_d(c) for c in n.children)
        return _d(self.root) if self.root else 0

    def size(self) -> int:
        return len(self._nodes)

    def leaf_nodes(self) -> List[ObjectNode]:
        return [n for n in self._nodes.values() if isinstance(n, ObjectNode)]

    def activation_summary(self) -> Dict[str, float]:
        return {nid: n.activation for nid, n in self._nodes.items()}

    def __repr__(self) -> str:
        return (
            f"ConceptGraph(name={self.name!r}, "
            f"size={self.size()}, depth={self.depth()})"
        )

    # ── Serialisation (backward compatibility) ────────────────────────

    def to_nested_list(self) -> list:
        """
        Serialize to the nested-list format of the original code.
        Field nodes are excluded (they are evaluation-time additions).
        """
        if self.root is None:
            return []
        return self._node_to_list(self.root)

    def _node_to_list(self, node: Node) -> list:
        if isinstance(node, ReceptiveFieldNode):
            return []   # field nodes are not part of the concept definition
        if isinstance(node, ObjectNode):
            return [node.name, node.orientation.value, node.layer]
        result = [node.node_type]
        for child in node.children:
            if not isinstance(child, ReceptiveFieldNode):
                result.append(self._node_to_list(child))
        return result


# ============================================================
# Section 9 — Graph Construction from Nested Lists
# ============================================================

_STRUCTURE_MAP = {
    "serial"  : SerialNode,
    "parallel": ParallelNode,
    "body"    : BodyNode,
    "plan"    : PlanNode,
}


def graph_from_nested_list(tree: list, name: str = "") -> ConceptGraph:
    """
    Build a ConceptGraph from the nested-list format of the original code.

    Field nodes are *not* included — the returned graph is a pure scanner
    definition.  Attach field nodes with `graph.attach_field()` or use
    `graph.evaluate_against(field)` to evaluate.

    Leaf format:
        [trigram_name, orientation_int, layer_int]
        e.g.  ["bottom", 1, 0]

    Structure format:
        [node_type_str, child_1, child_2, …]
        e.g.  ["serial", ["bottom", 1, 0], ["top", 1, 0]]

    A ParallelNode is automatically inserted as the root if the top-level
    node is not already a ParallelNode — this ensures every concept has a
    field-context hub at its root.
    """
    graph   = ConceptGraph(name=name)
    counter = [0]

    def _build(subtree: list) -> Node:
        counter[0] += 1
        nid = f"n{counter[0]}"

        if not isinstance(subtree, list) or not subtree:
            node = ObjectNode(nid, TRIGRAM_PATTERNS["neutral"])
            graph.add_node(node)
            return node

        head = subtree[0]

        if head in _STRUCTURE_MAP:
            node = _STRUCTURE_MAP[head](nid)
            graph.add_node(node)
            for child_spec in subtree[1:]:
                if isinstance(child_spec, int):
                    continue   # skip integer body-code annotations
                child = _build(child_spec)
                graph.add_solid_edge(node, child)
        else:
            trig_name   = head
            orient_int  = int(subtree[1]) if len(subtree) > 1 else 1
            pattern     = ALL_PATTERNS.get(trig_name, TRIGRAM_PATTERNS["neutral"])
            orientation = Orientation(orient_int)
            node = ObjectNode(
                nid, pattern, orientation, name=trig_name,
            )
            graph.add_node(node)

        return node

    inner_root = _build(tree)

    # Ensure the root is a ParallelNode (field-context hub)
    if isinstance(inner_root, ParallelNode):
        graph.root = inner_root
    else:
        hub = ParallelNode(f"hub_{name}")
        graph.add_node(hub)
        graph.add_solid_edge(hub, inner_root)
        graph.root = hub

    return graph
