"""
concept_learning.py — Metropolis-Hastings Concept Learner
==========================================================

Implements a Bayesian concept learner that searches the space of
ConceptGraphs using a Metropolis-Hastings (MH) MCMC algorithm.

The model treats concept learning as Bayesian inference:

    P(concept | observations) ∝ P(observations | concept) × P(concept)

where:
  - P(concept) is a prior that penalises complexity (number of nodes).
  - P(observations | concept) is a likelihood that measures how well the
    concept's predicted field activations match observed responses.

Observations
------------
An `Observation` is a (ReceptiveField, target_activation) pair:
  - `field`      : what the learner perceives
  - `response`   : the observed graded response (float in [0, 1])
                   e.g. from an infant looking-time paradigm, or a
                   human similarity rating

MCMC procedure
--------------
1. Start with a random ConceptGraph.
2. Propose a local mutation (add/remove node, change trigram, etc.).
3. Accept or reject using the MH acceptance ratio.
4. Repeat for `n_iterations` steps, recording the best graph found.

This mirrors the cognitive hypothesis that learners maintain a current
"best guess" concept and update it in response to new evidence.

Usage example
-------------
    from embodied_lot import ReceptiveField, TRIGRAM_PATTERNS, Orientation
    from concept_learning import Observation, ConceptLearner

    import numpy as np

    # ── Simulate observations from a "top-heavy" concept ──────────────
    true_pattern = TRIGRAM_PATTERNS["震"]   # (1, 0, 0)
    rng          = np.random.default_rng(42)

    observations = []
    for _ in range(20):
        field = ReceptiveField.random(9, 9)
        # Target: high activation when the field has a bright top region
        top_mean   = field.activation[:3, :].mean()
        response   = float(np.clip(top_mean + rng.normal(0, 0.05), 0, 1))
        observations.append(Observation(field, response))

    # ── Run the learner ───────────────────────────────────────────────
    learner = ConceptLearner(
        observations = observations,
        n_iterations = 500,
        random_seed  = 0,
    )
    best_graph, trace = learner.run(verbose=True)

    print("Best concept:", best_graph)
    print("Posterior score:", trace["best_score"])
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from embodied_lot import (
    ALL_PATTERNS,
    TRIGRAM_PATTERNS,
    ConceptGraph,
    Node,
    ObjectNode,
    Orientation,
    ParallelNode,
    PlanNode,
    ReceptiveField,
    SerialNode,
    BodyNode,
    StructureNode,
    DashedEdge,
    ReceptiveFieldNode,
    RetinaNode,
    graph_from_nested_list,
)


# ============================================================
# Section 1 — Observations
# ============================================================

@dataclass
class Observation:
    """
    A single learning trial: one perceptual scene paired with one
    behavioural response.

    Parameters
    ----------
    field : ReceptiveField
        The spatial scene that the learner perceives.
    response : float
        Observed graded response in [0, 1].  Conceptually, this could
        be an infant's looking time (normalised), a similarity rating,
        or a neural activation magnitude.
    weight : float
        Optional importance weight for this trial (default 1.0).
    label : str
        Optional human-readable description.
    """
    field    : ReceptiveField
    response : float
    weight   : float = 1.0
    label    : str   = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.response <= 1.0:
            raise ValueError("response must be in [0, 1].")


# ============================================================
# Section 2 — Prior
# ============================================================

class ConceptPrior:
    """
    Prior probability over ConceptGraphs.

    Implements an Occam's-razor-style prior: smaller, shallower graphs
    are preferred.  The log-prior is:

        log P(G) = − size_penalty × size(G) − depth_penalty × depth(G)

    Both terms are negative, so larger/deeper graphs receive lower
    log-prior.  The unnormalised prior probability is then:

        P(G) ∝ exp(log P(G))

    Parameters
    ----------
    size_penalty : float
        Penalty per node (default 0.1).
    depth_penalty : float
        Penalty per level of depth (default 0.05).
    """

    def __init__(
        self,
        size_penalty : float = 0.1,
        depth_penalty: float = 0.05,
    ) -> None:
        self.size_penalty  = size_penalty
        self.depth_penalty = depth_penalty

    def log_prior(self, graph: ConceptGraph) -> float:
        """Return the (unnormalised) log-prior for a ConceptGraph."""
        return (
            - self.size_penalty  * graph.size()
            - self.depth_penalty * graph.depth()
        )

    def __repr__(self) -> str:
        return (
            f"ConceptPrior(size_penalty={self.size_penalty}, "
            f"depth_penalty={self.depth_penalty})"
        )


# ============================================================
# Section 3 — Likelihood
# ============================================================

class ConceptLikelihood:
    """
    Likelihood of observations given a ConceptGraph.

    Models each observation as a noisy measurement of the concept's
    true activation.  Given the concept predicts activation `a` and
    the observed response is `r`, the log-likelihood contribution is:

        log P(r | a) = − (r − a)² / (2 × noise²)

    This corresponds to a Gaussian noise model.  The total log-likelihood
    is the weighted sum over all observations:

        log P(observations | G) = Σ_i weight_i × log P(r_i | G(field_i))

    Parameters
    ----------
    noise : float
        Standard deviation of the Gaussian noise model (default 0.15).
    """

    def __init__(self, noise: float = 0.15) -> None:
        if noise <= 0:
            raise ValueError("noise must be positive.")
        self.noise = noise

    def log_likelihood(
        self,
        graph       : ConceptGraph,
        observations: List[Observation],
    ) -> float:
        """
        Compute the total weighted log-likelihood.

        Each observation's field is evaluated via `graph.evaluate_against()`,
        which attaches a RetinaNode temporarily and detaches it after use,
        leaving the concept graph unmodified between observations.
        """
        total = 0.0
        for obs in observations:
            predicted = graph.evaluate_against(obs.field)
            error     = obs.response - predicted
            total    += obs.weight * (- error ** 2 / (2 * self.noise ** 2))
        return total

    def __repr__(self) -> str:
        return f"ConceptLikelihood(noise={self.noise})"


# ============================================================
# Section 4 — Proposal Distribution (Mutations)
# ============================================================

# All available trigram patterns as a list for sampling
_ALL_PATTERN_LIST = list(TRIGRAM_PATTERNS.values())
_STRUCTURE_TYPES  = [SerialNode, ParallelNode, BodyNode, PlanNode]


def _random_object_node(
    node_id: str,
    rng    : random.Random,
) -> ObjectNode:
    """Sample a random ObjectNode from the trigram vocabulary."""
    pattern     = rng.choice(_ALL_PATTERN_LIST)
    orientation = rng.choice(list(Orientation))
    is_dynamic  = rng.random() < 0.3
    layer       = rng.choice([0, -1])
    name        = next(
        (k for k, v in TRIGRAM_PATTERNS.items() if v == tuple(pattern)),
        "neutral",
    )
    return ObjectNode(
        node_id, pattern, orientation,
        is_dynamic=is_dynamic, layer=layer, name=name,
    )


def _random_structure_node(node_id: str, rng: random.Random) -> Node:
    """Sample a random StructureNode type."""
    cls = rng.choice(_STRUCTURE_TYPES)
    return cls(node_id)


class MutationProposal:
    """
    Proposes a new ConceptGraph by applying a random local mutation to
    the current graph.

    Available mutation operations
    -----------------------------
    change_trigram    : Replace a leaf node's trigram pattern with a new one.
    flip_orientation  : Toggle a leaf node's orientation (H ↔ V).
    flip_dynamic      : Toggle a leaf node's is_dynamic flag.
    add_leaf          : Add a new ObjectNode as a child of a StructureNode.
    remove_leaf       : Remove a leaf node from its parent.
    add_structure     : Insert a new StructureNode between a node and its parent.
    change_structure  : Change a StructureNode's type (e.g. serial → parallel).

    Each proposal is symmetric (same probability forward and backward),
    so the MH acceptance ratio reduces to the posterior ratio, simplifying
    the sampler.

    Parameters
    ----------
    operation_weights : dict, optional
        Relative probabilities of each mutation type.
    seed : int, optional
        Random seed for reproducibility.
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "change_trigram"   : 0.30,
        "flip_orientation" : 0.10,
        "flip_dynamic"     : 0.10,
        "add_leaf"         : 0.15,
        "remove_leaf"      : 0.15,
        "change_structure" : 0.20,
    }

    def __init__(
        self,
        operation_weights: Optional[Dict[str, float]] = None,
        seed             : Optional[int] = None,
    ) -> None:
        weights    = operation_weights or self._DEFAULT_WEIGHTS
        ops, probs = zip(*weights.items())
        total      = sum(probs)
        self._ops   = list(ops)
        self._probs = [p / total for p in probs]
        self._rng   = random.Random(seed)
        self._np_rng= np.random.default_rng(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"proposed_{self._counter}"

    def propose(self, graph: ConceptGraph) -> ConceptGraph:
        """
        Return a mutated deep copy of `graph`.

        If no valid mutation is found after several tries (e.g. the graph
        is too small for removal), a `change_trigram` is applied instead.
        """
        candidate = copy.deepcopy(graph)
        operation = self._rng.choices(self._ops, weights=self._probs, k=1)[0]

        success = getattr(self, f"_op_{operation}")(candidate)
        if not success:
            # Fallback: always-valid trigram change
            self._op_change_trigram(candidate)

        return candidate

    # ── Mutation implementations ──────────────────────────────────────

    def _op_change_trigram(self, graph: ConceptGraph) -> bool:
        """Replace a random leaf node's trigram pattern."""
        leaves = graph.leaf_nodes()
        if not leaves:
            return False
        node         = self._rng.choice(leaves)
        new_pattern  = self._rng.choice(_ALL_PATTERN_LIST)
        node.pattern = tuple(new_pattern)
        node.name    = next(
            (k for k, v in TRIGRAM_PATTERNS.items()
             if v == node.pattern), "?"
        )
        return True

    def _op_flip_orientation(self, graph: ConceptGraph) -> bool:
        """Toggle a random leaf node's orientation."""
        leaves = graph.leaf_nodes()
        if not leaves:
            return False
        node = self._rng.choice(leaves)
        node.orientation = (
            Orientation.HORIZONTAL
            if node.orientation == Orientation.VERTICAL
            else Orientation.VERTICAL
        )
        return True

    def _op_flip_dynamic(self, graph: ConceptGraph) -> bool:
        """Toggle a random leaf node's is_dynamic flag."""
        leaves = graph.leaf_nodes()
        if not leaves:
            return False
        node           = self._rng.choice(leaves)
        node.is_dynamic = not node.is_dynamic
        return True

    def _op_add_leaf(self, graph: ConceptGraph) -> bool:
        """
        Add a new ObjectNode as a child of a randomly chosen
        StructureNode, if one exists.
        """
        structure_nodes = [
            n for n in graph.nodes
            if isinstance(n, (SerialNode, ParallelNode, BodyNode, PlanNode))
        ]
        if not structure_nodes:
            return False

        parent   = self._rng.choice(structure_nodes)
        new_node = _random_object_node(self._next_id(), self._rng)
        graph.add_node(new_node)
        graph.add_solid_edge(parent, new_node)
        return True

    def _op_remove_leaf(self, graph: ConceptGraph) -> bool:
        """
        Remove a random leaf node from its parent, if the parent will
        still have at least one child after removal.
        """
        # Find leaves whose parent has ≥ 2 children
        removable = [
            leaf for leaf in graph.leaf_nodes()
            if any(
                leaf in parent.children and len(parent.children) >= 2
                for parent in graph.nodes
                if isinstance(parent, StructureNode)
            )
        ]
        if not removable:
            return False

        target = self._rng.choice(removable)

        # Remove from parent's child list
        for parent in graph.nodes:
            if isinstance(parent, (SerialNode, ParallelNode,
                                   BodyNode, PlanNode)):
                if target in parent.children:
                    parent.children.remove(target)
                    break

        # Remove solid edges involving target
        graph._solid = [
            e for e in graph._solid
            if e.source is not target and e.target is not target
        ]
        # Remove from node registry
        del graph._nodes[target.node_id]
        return True

    def _op_change_structure(self, graph: ConceptGraph) -> bool:
        """
        Change the type of a random StructureNode (e.g. serial → parallel).
        The root ParallelNode is never mutated — it must remain a ParallelNode
        to serve as the field-context hub for evaluate_against().
        """
        structure_nodes = [
            n for n in graph.nodes
            if isinstance(n, (SerialNode, ParallelNode, BodyNode, PlanNode))
            and not (n is graph.root and isinstance(n, ParallelNode))
        ]
        if not structure_nodes:
            return False

        old_node  = self._rng.choice(structure_nodes)
        other_cls = [
            cls for cls in _STRUCTURE_TYPES
            if not isinstance(old_node, cls)
        ]
        if not other_cls:
            return False

        new_cls  = self._rng.choice(other_cls)
        new_node = new_cls(old_node.node_id)
        new_node.children = old_node.children

        graph._nodes[old_node.node_id] = new_node

        for edge in graph._solid:
            if edge.source is old_node:
                edge.source = new_node
            if edge.target is old_node:
                edge.target = new_node

        for node in graph._nodes.values():
            if old_node in node.children:
                node.children[node.children.index(old_node)] = new_node

        if graph.root is old_node:
            graph.root = new_node

        return True


# ============================================================
# Section 5 — Metropolis-Hastings Learner
# ============================================================

@dataclass
class MCMCTrace:
    """
    Container for the MCMC run history.

    Attributes
    ----------
    scores : list of float
        Log-posterior at each accepted iteration.
    best_score : float
        Maximum log-posterior observed.
    best_iteration : int
        Iteration at which the best graph was found.
    acceptance_rate : float
        Fraction of proposals accepted.
    """
    scores        : List[float] = field(default_factory=list)
    best_score    : float       = -np.inf
    best_iteration: int         = 0
    n_accepted    : int         = 0
    n_total       : int         = 0

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(self.n_total, 1)


class ConceptLearner:
    """
    A Metropolis-Hastings concept learner.

    Searches the space of ConceptGraphs to find the graph that maximises

        log P(concept | observations)
            = log P(observations | concept) + log P(concept)

    The sampler maintains a *current* hypothesis and proposes local
    mutations at each step.  Proposals are accepted with probability

        min(1, exp(Δ log-posterior))

    where Δ is the change in log-posterior.

    Parameters
    ----------
    observations : list of Observation
        The learning data.
    n_iterations : int
        Total number of MH steps.
    prior : ConceptPrior, optional
        Prior over concept complexity.  Defaults to ConceptPrior().
    likelihood : ConceptLikelihood, optional
        Noise model.  Defaults to ConceptLikelihood().
    proposal : MutationProposal, optional
        Mutation operator.  Defaults to MutationProposal().
    initial_graph : ConceptGraph, optional
        Starting hypothesis.  If None, a minimal random graph is used.
    random_seed : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        observations  : List[Observation],
        n_iterations  : int               = 1000,
        prior         : Optional[ConceptPrior]      = None,
        likelihood    : Optional[ConceptLikelihood] = None,
        proposal      : Optional[MutationProposal]  = None,
        initial_graph : Optional[ConceptGraph]      = None,
        random_seed   : Optional[int]               = None,
    ) -> None:
        self.observations  = observations
        self.n_iterations  = n_iterations
        self.prior         = prior       or ConceptPrior()
        self.likelihood    = likelihood  or ConceptLikelihood()
        self.proposal      = proposal    or MutationProposal(seed=random_seed)
        self._initial_graph = initial_graph
        self._rng          = random.Random(random_seed)
        self._np_rng       = np.random.default_rng(random_seed)

    # ── Scoring ───────────────────────────────────────────────────────

    def log_posterior(self, graph: ConceptGraph) -> float:
        """Unnormalised log-posterior = log-likelihood + log-prior."""
        return (
            self.likelihood.log_likelihood(graph, self.observations)
            + self.prior.log_prior(graph)
        )

    # ── Initialisation ────────────────────────────────────────────────

    def _make_initial_graph(self) -> ConceptGraph:
        """
        Build a minimal starting concept: a neutral ObjectNode under a
        ParallelNode hub (required to receive field context during evaluation).
        """
        graph = ConceptGraph(name="initial")
        hub   = ParallelNode("hub")
        root  = ObjectNode(
            "root",
            TRIGRAM_PATTERNS["neutral"],
            Orientation.VERTICAL,
            name="neutral",
        )
        graph.add_node(hub)
        graph.add_node(root)
        graph.add_solid_edge(hub, root)
        graph.root = hub
        return graph

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, verbose: bool = False) -> Tuple[ConceptGraph, MCMCTrace]:
        """
        Run the MH sampler for `n_iterations` steps.

        Parameters
        ----------
        verbose : bool
            If True, print progress every 100 iterations.

        Returns
        -------
        best_graph : ConceptGraph
            The graph with the highest log-posterior seen during the run.
        trace : MCMCTrace
            Run statistics and score history.
        """
        current = (
            self._initial_graph
            if self._initial_graph is not None
            else self._make_initial_graph()
        )
        current_score = self.log_posterior(current)

        best_graph = copy.deepcopy(current)
        best_score = current_score

        trace = MCMCTrace()

        for iteration in range(self.n_iterations):
            # ── Propose ───────────────────────────────────────────────
            candidate       = self.proposal.propose(current)
            candidate_score = self.log_posterior(candidate)

            # ── Accept / reject ───────────────────────────────────────
            log_accept = min(0.0, candidate_score - current_score)
            accept     = np.log(self._np_rng.random() + 1e-300) < log_accept

            trace.n_total += 1

            if accept:
                current       = candidate
                current_score = candidate_score
                trace.n_accepted += 1
                trace.scores.append(current_score)

                # Track best hypothesis
                if current_score > best_score:
                    best_score         = current_score
                    best_graph         = copy.deepcopy(current)
                    trace.best_score     = best_score
                    trace.best_iteration = iteration

            # ── Logging ───────────────────────────────────────────────
            if verbose and iteration % 100 == 0:
                print(
                    f"  Iteration {iteration:>5d} / {self.n_iterations}  "
                    f"score={current_score:+.3f}  "
                    f"best={best_score:+.3f}  "
                    f"accept_rate={trace.acceptance_rate:.2f}  "
                    f"graph_size={current.size()}"
                )

        if verbose:
            print(
                f"\n  Run complete.  Best score: {trace.best_score:+.3f}  "
                f"(iteration {trace.best_iteration})  "
                f"Acceptance rate: {trace.acceptance_rate:.2f}"
            )

        return best_graph, trace


# ============================================================
# Section 6 — Structural Similarity (for analysis & comparison)
# ============================================================

def tree_edit_similarity(g1: ConceptGraph, g2: ConceptGraph) -> float:
    """
    Compute a normalised structural similarity between two ConceptGraphs
    based on matching their node sequences (via nested-list serialisation).

    Returns a value in [0, 1] where 1 means structurally identical.

    This is a lightweight approximation using Levenshtein distance on
    flattened node-type sequences.  For a full tree-edit distance,
    consider the APTED or ZhangShasha algorithm.

    Notes
    -----
    The similarity is symmetric and does not account for node ordering
    within ParallelNodes (which is semantically irrelevant).
    """
    seq1 = _flatten_structure(g1.root) if g1.root else []
    seq2 = _flatten_structure(g2.root) if g2.root else []

    # Levenshtein ratio (normalised edit distance)
    distance = _levenshtein(seq1, seq2)
    max_len  = max(len(seq1), len(seq2), 1)
    return 1.0 - distance / max_len


def _flatten_structure(node: Optional[Node]) -> List[str]:
    """Pre-order traversal of node types."""
    if node is None:
        return []
    result = [node.node_type]
    for child in node.children:
        result.extend(_flatten_structure(child))
    return result


def _levenshtein(s1: List[str], s2: List[str]) -> int:
    """
    Standard dynamic-programming Levenshtein distance between two
    lists of strings.
    """
    m, n = len(s1), len(s2)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


# ============================================================
# Section 7 — Convenience: batch evaluation
# ============================================================

def evaluate_on_dataset(
    graph       : ConceptGraph,
    observations: List[Observation],
) -> Dict[str, float]:
    """
    Evaluate a ConceptGraph against every observation and return summary
    statistics useful for reporting results to colleagues.

    Returns
    -------
    dict with keys:
        "mean_error"  : mean absolute error |predicted − response|
        "rmse"        : root mean squared error
        "correlation" : Pearson r between predictions and responses
        "predictions" : list of predicted activations
        "targets"     : list of observed responses
    """
    predictions = [graph.evaluate_against(obs.field) for obs in observations]
    targets     = [obs.response             for obs in observations]

    errors = [abs(p - t) for p, t in zip(predictions, targets)]
    sq_err = [(p - t) ** 2 for p, t in zip(predictions, targets)]

    preds_arr   = np.array(predictions)
    targets_arr = np.array(targets)

    corr = (
        float(np.corrcoef(preds_arr, targets_arr)[0, 1])
        if len(predictions) > 1 and preds_arr.std() > 0 and targets_arr.std() > 0
        else float("nan")
    )

    return {
        "mean_error" : float(np.mean(errors)),
        "rmse"       : float(np.sqrt(np.mean(sq_err))),
        "correlation": corr,
        "predictions": predictions,
        "targets"    : targets,
    }
