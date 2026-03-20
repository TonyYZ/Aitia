# EmbodiedLoT — A Spatial Language of Thought with Bayesian Concept Learning

EmbodiedLoT is a Python implementation of a symbolic-spatial cognitive model for concept representation and learning, grounded in embodied cognition theory. Concepts are modelled as structured scanner programs that evaluate spatial receptive fields. Primitive image schemas are drawn from the Yijing (I Ching) trigram system, whose binary line structure naturally encodes spatial distributions. A Metropolis-Hastings sampler searches the space of concept graphs to find the hypothesis that best explains observed perceptual responses.

---

## Motivation

Embodied cognition research holds that conceptual structure is grounded in sensorimotor experience, with image schemas (Johnson 1987; Lakoff 1987) serving as the primitive building blocks of meaning. Existing computational LoT models (Piantadosi 2021) treat programs as purely symbolic trees; this project explores a hybrid where programs scan spatial fields and are composed via a graph structure that includes conditional (dashed) edges for top-down attentional gating. The Yijing trigram system provides a cross-cultural vocabulary of spatial primitives with a natural binary encoding, connecting the model to work on language iconicity and spatial semantics. The model is also designed to interface with the [Nguasach](https://github.com/TonyYZ/Nguasach) phonetic-semantic corpus, where trigram encodings of words provide a bridge between phonetic form and spatial meaning.

---

## Model Architecture

### Primitives: trigram image schemas

Each `ObjectNode` encodes one spatial schema as a trigram pattern over a receptive field. Bands are read **bottom to top** (vertical) or **left to right** (horizontal), following the Yijing convention:

| Pattern | Name        | Spatial meaning (vertical) |
| ------- | ----------- | -------------------------- |
| (0,0,0) | `void`      | background; does not scan  |
| (1,0,0) | `bottom`    | figure at bottom only      |
| (0,1,0) | `center`    | figure at center row       |
| (1,1,0) | `lower`     | figure in lower two-thirds |
| (1,1,1) | `universal` | figure everywhere          |
| (1,0,1) | `periphery` | figure at outer rows       |
| (0,1,1) | `upper`     | figure in upper two-thirds |
| (0,0,1) | `top`       | figure at top only         |

Only **yang (non-zero) bands** contribute to scanning. A node's `value` (default 1.0, set by incoming dashed edges) gates its activation: `activation = value × mean_yang_similarity`.

### Composition: the concept graph

Concepts are directed graphs with two edge types:

- **Solid edges** — compositional hierarchy: `ParallelNode` (alpha-blend / field hub), `SerialNode` (spatial strip division / average), `BodyNode` / `PlanNode` (probabilistic replacement rule), `ReturnNode` (output sink).
- **Dashed edges** — conditional modulation: a dashed edge from node A to node B sets `B.value = mean(activation of all incoming sources)`. This implements top-down attentional gating without changing the scanning field.

### Replacement rule (Body / Plan nodes)

For template T (value _p_) and content C:

```
activation = p × mean(C.evaluate(yang_sub_region_i))
           + (1−p) × C.evaluate(full_field)
```

With _p_ = 1 (default), T is a strict spatial filter: only its yang sub-regions let C through. A dashed edge from a low-activation condition lowers _p_, leaking C's full-field scan through the yin positions.

### Lazy evaluation and field nodes

Every node is a **lazy scanner function** — it only fires when a `ReceptiveFieldNode` sibling exists inside the same `ParallelNode`. Field nodes (`RetinaNode`, `TactileNode`, `ProprioNode`) are input portals that supply spatial data. `ReturnNode` is the output portal that names the concept's answer area. Neither type alters computation; they are external observers and displayers.

Multi-modal evaluation is supported: when a `ParallelNode` has multiple field node siblings with different presence weights, scanner activations are computed as a weighted average across modalities.

### Evaluation protocol

`ConceptGraph.evaluate()` runs three passes:

1. **Forward scan** — all node values at default 1.0; produces initial activations.
2. **Value update** — dashed edges set `target.value = mean(source activations)` in topological order.
3. **Re-scan** — with updated values; produces the final result.

---

## Files

| File                  | Description                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `embodied_lot.py`     | Core model: trigram system, `ReceptiveField`, all node types, `ConceptGraph`                                                  |
| `trigrams.py`         | ASCII aliases for trigram patterns (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`, etc.)                                                   |
| `concept_learning.py` | Metropolis-Hastings concept learner: `Observation`, `ConceptPrior`, `ConceptLikelihood`, `MutationProposal`, `ConceptLearner` |
| `demos.py`            | Worked examples of spatial schemas and concept evaluation                                                                     |

---

## Usage

### Building a concept

```python
from trigrams import TOP, BOTTOM, UNIVERSAL, VOID, V
from embodied_lot import (
    ConceptGraph, ObjectNode, SerialNode, ParallelNode,
    PlanNode, ReturnNode, ReceptiveField,
)
import numpy as np

# "above": figure is at the top AND the bottom is empty ground
g = ConceptGraph("above")

# Scan subgraph: split field into bottom and top strips
par_root = ParallelNode("par_root")
ser      = SerialNode("ser", split_orientation=V)
yang_0   = ObjectNode("y0", UNIVERSAL, V)   # scans bottom strip
yang_1   = ObjectNode("y1", UNIVERSAL, V)   # scans top strip

# Output subgraph: plan node with yin template + content
par_out  = ParallelNode("par_out")
plan     = PlanNode("plan")
yin_0    = ObjectNode("yn0", VOID, V)       # all-yin: blocks content when bottom is occupied
yang_2   = ObjectNode("y2", UNIVERSAL, V)   # content: active when top is occupied
ret      = ReturnNode("ret")

g.add_nodes([par_root, ser, yang_0, yang_1,
             par_out, plan, yin_0, yang_2, ret])

g.add_solid_edge(par_root, ser)
g.add_solid_edge(ser, yang_0)              # children[0] = bottom strip
g.add_solid_edge(ser, yang_1)              # children[1] = top strip
g.add_solid_edge(par_root, par_out)
g.add_solid_edge(par_out, plan)
g.add_solid_edge(par_out, ret)
g.add_solid_edge(plan, yin_0)              # template
g.add_solid_edge(plan, yang_2)             # content

g.add_dashed_edge(yang_0, yin_0)           # bottom substance → yin template value
g.add_dashed_edge(yang_1, yang_2)          # top substance → content value

g.root = par_root
```

### Evaluating against a receptive field

```python
# Field: figure at the top, empty at the bottom
field_arr = np.zeros((9, 9))
field_arr[:3, :] = 0.9    # bright top
field_arr[3:, :] = 0.1    # dim bottom
field = ReceptiveField(field_arr)

activation = g.evaluate_against(field)    # → ~0.21 (above TRUE)

# Reversed field: figure at the bottom
field_arr2 = 1.0 - field_arr
activation2 = g.evaluate_against(field2)  # → ~0.01 (above FALSE)
```

### Learning a concept from observations

```python
from concept_learning import Observation, ConceptLearner

# Build observations: (ReceptiveField, observed_response) pairs
observations = [
    Observation(ReceptiveField.random(9, 9), response=0.8),
    # ...
]

learner = ConceptLearner(observations, n_iterations=500, random_seed=42)
best_graph, trace = learner.run(verbose=True)

print("Best concept:", best_graph.to_nested_list())
print("Acceptance rate:", trace.acceptance_rate)
```

---

## Key Dependencies

- `numpy` — array computation for receptive fields and activation values
- `scipy` (optional) — for future extensions to continuous field processing

---

## Related Projects

- [Nguasach](https://github.com/TonyYZ/Nguasach) — cross-linguistic phonetic-semantic corpus; `sortHexagram.py` converts phonetic embeddings into trigram-style binary encodings that can be used as receptive fields for this model
- [Ítí (Ete)](https://github.com/TonyYZ/Ete) — an artistic constructed language and web-based learnability experiment

---

## References

- Johnson, M. (1987). _The Body in the Mind_. University of Chicago Press.
- Lakoff, G. (1987). _Women, Fire, and Dangerous Things_. University of Chicago Press.
- Piantadosi, S. T. (2021). The computational origin of representation. _Minds and Machines_, 31, 1–58.
- Fodor, J. A. (1975). _The Language of Thought_. Harvard University Press.

---

## Author

Yutong (Tony) Zhou — M1 Cognitive Science, ENS-PSL
Background in cognitive science, computational linguistics, and embodied semantics.
