from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class PoleType(str, Enum):
    """Represents which pole of a neuron's activation distribution to target."""
    POSITIVE = "positive"  # Activations > 0 (excitatory motif)
    NEGATIVE = "negative"  # Activations < 0 (inhibitory / opposing gate motif)
    BOTH = "both"  # Entire neuron basis across all values


class InterventionMode(str, Enum):
    """Supported surgical neuron manipulation strategies."""
    ZERO_ABLATE = "zero_ablate"  # Set target neuron(s) directly to 0.0
    HARD_CLAMP = "hard_clamp"  # Force neuron(s) to a fixed constant c
    SCALE = "scale"  # Multiply neuron activation by alpha
    ADDITIVE_STEER = "additive_steer"  # Shift activation by +delta (concept injection)
    WEAK_PRUNE = "weak_prune"  # Zero out weak activations where |a| < threshold ("Do Weak Activations Matter?")
    POLE_RECTIFY = "pole_rectify"  # Zero out one pole while preserving the other
    CONDITIONAL_GATE = "conditional_gate"  # Closed-loop dynamic gating triggered by an upstream/detector neuron


class SparsityTailPrior(str, Enum):
    """
    Statistical activation-shape prior computed in O(1) memory from streaming moments
    and lexical token concentration (distinct from semantic `NeuronTaxonomy` verification):
      - SPARSE_DETECTOR: Heavy excess tail above the SwiGLU/Gaussian null & sparse firing.
      - BIPOLAR_ROUTER_CANDIDATE: Opposing poles with high token-conditional variance (R^2).
      - GRADED_DIAL: Continuous firing across dynamic range with structured lexical variance.
      - DENSE_UNSELECTIVE: Indistinguishable from random Gaussian / SwiGLU white-noise firing.
      - DEAD_OR_FLAT: Near-zero variance, zero outgoing weight norm, or near-zero firing rate.
    """
    SPARSE_DETECTOR = "sparse_detector"
    BIPOLAR_ROUTER_CANDIDATE = "bipolar_router_candidate"
    GRADED_DIAL = "graded_dial"
    DENSE_UNSELECTIVE = "dense_unselective"
    DEAD_OR_FLAT = "dead_or_flat"


class NeuronTaxonomy(str, Enum):
    """
    Classification of a neuron's (or single pole's) semantic organization following
    Cammarata ('Neural Networks Are Already Clean', Goodfire 2026):
      - CLEAN_MONOSEMANTIC (~90% of verified poles): Performs a single clear core job / differential split.
      - STRUCTURED_RECURSIVE_MIXTURE (~7%): Core trunk motif with related semantic offshoots.
      - TRUE_POLYSEMANTIC (~2-4%): Unrelated concepts packed on the same pole.
      - DEAD_OR_NOISY: Near-zero variance or flat unselective noise matching a random-init baseline.
      - UNVERIFIED: Not yet semantically verified and lacks a decisive statistical/lexical signature.
    """
    CLEAN_MONOSEMANTIC = "clean_monosemantic"
    STRUCTURED_RECURSIVE_MIXTURE = "structured_recursive_mixture"
    TRUE_POLYSEMANTIC = "true_polysemantic"
    DEAD_OR_NOISY = "dead_or_noisy"
    UNVERIFIED = "unverified"


@dataclass
class ActivationSample:
    """
    Stores a single top-activating or quantile-stratified context example for a neuron pole.
    Includes `next_token_str` and `window_str` to surface forward next-token prediction motifs
    (e.g., Cammarata's 'Titanic -> boat' finding) and `is_double_negative_route` to flag
    SwiGLU activations where `gate < 0` and `up < 0` multiply to a positive value
    ('Separating the Sign Regimes').
    """
    activation_value: float
    sample_id: int
    position_idx: Optional[int] = None  # Token index (for sequences) or spatial index (for CNNs)
    token_str: Optional[str] = None  # Decoded focal token or patch label
    context_str: Optional[str] = None  # Surrounding text window or sample metadata
    quantile_band: Optional[str] = None  # "peak_trunk", "secondary_branch", or "tertiary_offshoot"
    next_token_str: Optional[str] = None  # Token at position_idx + 1 (forward prediction clue)
    window_str: Optional[str] = None  # Local snippet highlighting "... prev [FOCAL] -> next ..."
    is_double_negative_route: bool = False  # True when SwiGLU gate < 0 and up < 0 yield a > 0


@dataclass
class NeuronPoleProfile:
    """
    Detailed statistical and semantic profile for a single neuron across both poles,
    including conditional pole-specific selectivity/taxonomy, stratified quantile bands,
    SwiGLU four-quadrant sign regime diagnostics, and firing motif metrics
    (current-token detector vs. next-token preparer vs. punctuation summary motif).
    """
    layer_name: str
    neuron_idx: int
    mean: float
    std: float
    max_val: float
    min_val: float
    positive_firing_rate: float
    negative_firing_rate: float
    pole_selectivity_score: float  # Null-calibrated tail & lexical selectivity score
    top_positive_samples: List[ActivationSample] = field(default_factory=list)
    top_negative_samples: List[ActivationSample] = field(default_factory=list)
    top_promoted_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_suppressed_tokens: List[Tuple[str, float]] = field(default_factory=list)

    # Pole-specific conditional statistics ("How I Think About Neurons": Evaluating Poles Independently)
    pos_pole_mean: float = 0.0
    pos_pole_std: float = 0.0
    pos_pole_selectivity_score: float = 0.0
    pos_taxonomy_class: NeuronTaxonomy = NeuronTaxonomy.DEAD_OR_NOISY
    neg_pole_mean: float = 0.0
    neg_pole_std: float = 0.0
    neg_pole_selectivity_score: float = 0.0
    neg_taxonomy_class: NeuronTaxonomy = NeuronTaxonomy.DEAD_OR_NOISY

    # Stratified quantile activation buckets per pole ("Recursive Generalization" / "Do Weak Activations Matter?")
    # Keys: "peak_trunk" ([80%..100%]), "secondary_branch" ([40%..80%)), "tertiary_offshoot" ([15%..40%))
    positive_quantile_buckets: Dict[str, List[ActivationSample]] = field(default_factory=dict)
    negative_quantile_buckets: Dict[str, List[ActivationSample]] = field(default_factory=dict)

    # SwiGLU Double-Negative & 4-Quadrant Sign Regime diagnostics ("Separating the Sign Regimes")
    double_negative_positive_rate: float = 0.0
    double_negative_samples: List[ActivationSample] = field(default_factory=list)
    swiglu_sign_regime_rates: Dict[str, float] = field(default_factory=dict)
    double_negative_band_rates: Dict[str, float] = field(default_factory=dict)

    # Statistical priors & language motif diagnostics ("Summary Motif", "Titanic -> boat", "Don't Say Dot")
    pos_sparsity_prior: SparsityTailPrior = SparsityTailPrior.DEAD_OR_FLAT
    neg_sparsity_prior: SparsityTailPrior = SparsityTailPrior.DEAD_OR_FLAT
    lexical_selectivity_r2: float = 0.0
    current_token_concentration: float = 0.0
    next_token_concentration: float = 0.0
    punctuation_firing_rate: float = 0.0
    inferred_firing_motif: str = "unclassified"

    @property
    def tail_prior(self) -> SparsityTailPrior:
        """Dominant statistical activation-shape prior across both poles."""
        return (
            self.pos_sparsity_prior
            if self.pos_pole_selectivity_score >= self.neg_pole_selectivity_score
            else self.neg_sparsity_prior
        )

    @property
    def is_clean_candidate(self) -> bool:
        """Heuristic indicating strong null-calibrated selectivity on either pole or overall."""
        return (
            self.pole_selectivity_score >= 2.0
            or self.pos_pole_selectivity_score >= 2.0
            or self.neg_pole_selectivity_score >= 2.0
            or self.lexical_selectivity_r2 >= 0.35
        )


@dataclass
class GateConditionSpec:
    """
    Defines a closed-loop conditional trigger for `InterventionMode.CONDITIONAL_GATE`:
    Only applies `action_mode` to the target neuron(s) when `detector_neuron` in `detector_layer`
    fires on `detector_pole` with magnitude exceeding `threshold`.

    When `sequence_scope == "any_token"` and `sticky_on_decode == True`, a trigger during
    prompt prefill (`seq_len > 1`) or an earlier decoding step latches the gate open across
    subsequent single-token KV-cached decoding steps (`seq_len == 1`).
    """
    detector_layer: str
    detector_neuron: int
    detector_pole: PoleType = PoleType.POSITIVE
    threshold: float = 0.5
    action_mode: InterventionMode = InterventionMode.ADDITIVE_STEER
    sequence_scope: str = "token_wise"  # "token_wise" (per token position) or "any_token" (whole sequence)
    sticky_on_decode: bool = True  # Latches "any_token" trigger across seq_len=1 KV-cached generation steps


@dataclass
class InterventionSpec:
    """
    Specification for a surgical intervention on one or more neurons in a layer.
    Supports scalar `value`, native per-neuron adaptive values via `per_neuron_values`,
    and closed-loop dynamic concept gating via `gate_condition`.
    """
    layer_name: str
    neuron_indices: List[int]
    mode: InterventionMode = InterventionMode.ZERO_ABLATE
    pole: PoleType = PoleType.BOTH
    value: float = 0.0  # Default scalar clamp constant, scale factor, shift delta, or weak threshold
    per_neuron_values: Optional[List[float]] = None  # Optional per-neuron values aligned with `neuron_indices`
    token_positions: Optional[List[int]] = None  # If None, applies across all sequence/spatial positions
    gate_condition: Optional[GateConditionSpec] = None  # Optional upstream/same-layer detector trigger
    custom_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    enabled: bool = True


class InteractionRole(str, Enum):
    """Classifies how an upstream neuron pole influences a downstream neuron."""
    EXCITATORY = "excitatory"  # Drives downstream neuron strongly positive (+Delta)
    INHIBITORY = "inhibitory"  # Drives downstream neuron negative or suppresses it (-Delta)
    GATING_MODULATOR = "gating"  # Primarily modulates the SwiGLU gate rather than the linear up-projection


@dataclass
class CircuitEdge:
    """
    Represents a directed neuron-to-neuron connection between Layer L_src and Layer L_dst.
    Combines static weight geometry with dynamic causal ablation & Jacobian sensitivity.
    """
    src_layer: str
    src_neuron: int
    src_pole: PoleType
    dst_layer: str
    dst_neuron: int
    dst_pole_tendency: PoleType

    # Causal & Dynamic metrics (measured via forward-hook surgical intervention)
    mean_causal_delta: float  # E[A_clean(dst) - A_ablated(dst)] when src_pole is ablated
    max_causal_delta: float  # Peak signed shift across tokens/samples
    clean_dst_mean: float  # Baseline activation of dst_neuron before src ablation
    ablated_dst_mean: float  # Activation of dst_neuron after src_pole ablation
    jacobian_sensitivity: float = 0.0  # Mean d(A_dst) / d(A_src) across active tokens

    # Static Structural metrics (Virtual Weights through residual stream)
    virtual_weight_up: float = 0.0  # <W_up[dst], W_down[:, src]>
    virtual_weight_gate: float = 0.0  # <W_gate[dst], W_down[:, src]>
    static_alignment_cos: float = 0.0  # Cosine similarity between src output vector and dst input vector

    @property
    def role(self) -> InteractionRole:
        if abs(self.virtual_weight_gate) > 2.0 * abs(self.virtual_weight_up) and abs(self.virtual_weight_gate) > 1e-6:
            return InteractionRole.GATING_MODULATOR
        return InteractionRole.EXCITATORY if self.mean_causal_delta >= 0 else InteractionRole.INHIBITORY

    @property
    def composite_importance(self) -> float:
        """Combined ranking score prioritizing actual causal impact reinforced by structural weight."""
        return abs(self.mean_causal_delta) * (1.0 + 0.25 * abs(self.static_alignment_cos))


@dataclass
class CircuitGraph:
    """
    Stores a traced multi-neuron or multi-layer circuit subgraph.
    Supports both downstream traces (1 source neuron -> many destination neurons)
    and upstream incoming traces (many source neurons -> 1 destination neuron).
    """
    src_layer: str
    dst_layer: str
    src_neuron: int
    src_pole: PoleType
    edges: List[CircuitEdge] = field(default_factory=list)
    dst_neuron: Optional[int] = None

    def top_excitatory(self, k: int = 5) -> List[CircuitEdge]:
        """Returns edges with the strongest positive causal contribution (+Delta)."""
        pos = [e for e in self.edges if e.mean_causal_delta > 0]
        return sorted(pos, key=lambda e: e.mean_causal_delta, reverse=True)[:k]

    def top_inhibitory(self, k: int = 5) -> List[CircuitEdge]:
        """Returns edges with the strongest negative/inhibitory causal contribution (-Delta)."""
        neg = [e for e in self.edges if e.mean_causal_delta < 0]
        return sorted(neg, key=lambda e: e.mean_causal_delta, reverse=False)[:k]

    def to_agent_payload(self) -> Dict[str, Any]:
        """Serializes a downstream circuit into a structured dictionary for the LLM Explainer Agent."""
        return {
            "source": {
                "layer": self.src_layer,
                "neuron_idx": self.src_neuron,
                "pole": self.src_pole.value,
            },
            "target_layer": self.dst_layer,
            "top_excited_downstream": [
                {
                    "dst_neuron": e.dst_neuron,
                    "dst_pole_tendency": e.dst_pole_tendency.value,
                    "role": e.role.value,
                    "mean_causal_delta": round(e.mean_causal_delta, 4),
                    "max_causal_delta": round(e.max_causal_delta, 4),
                    "jacobian_sensitivity": round(e.jacobian_sensitivity, 4),
                    "virtual_weight_up": round(e.virtual_weight_up, 4),
                    "virtual_weight_gate": round(e.virtual_weight_gate, 4),
                }
                for e in self.top_excitatory(5)
            ],
            "top_inhibited_downstream": [
                {
                    "dst_neuron": e.dst_neuron,
                    "dst_pole_tendency": e.dst_pole_tendency.value,
                    "role": e.role.value,
                    "mean_causal_delta": round(e.mean_causal_delta, 4),
                    "max_causal_delta": round(e.max_causal_delta, 4),
                    "jacobian_sensitivity": round(e.jacobian_sensitivity, 4),
                    "virtual_weight_up": round(e.virtual_weight_up, 4),
                    "virtual_weight_gate": round(e.virtual_weight_gate, 4),
                }
                for e in self.top_inhibitory(5)
            ],
        }

    def to_incoming_agent_payload(self) -> Dict[str, Any]:
        """Serializes an upstream incoming circuit (L_src -> dst_neuron) for the LLM Explainer Agent."""
        target_n = self.dst_neuron if self.dst_neuron is not None else (
            self.edges[0].dst_neuron if self.edges else -1
        )
        return {
            "source_layer": self.src_layer,
            "target": {
                "layer": self.dst_layer,
                "neuron_idx": target_n,
                "pole": self.src_pole.value,
            },
            "top_excitatory_upstream": [
                {
                    "src_neuron": e.src_neuron,
                    "src_pole": e.src_pole.value,
                    "role": e.role.value,
                    "mean_causal_delta": round(e.mean_causal_delta, 4),
                    "max_causal_delta": round(e.max_causal_delta, 4),
                    "jacobian_sensitivity": round(e.jacobian_sensitivity, 4),
                    "virtual_weight_up": round(e.virtual_weight_up, 4),
                    "virtual_weight_gate": round(e.virtual_weight_gate, 4),
                }
                for e in self.top_excitatory(5)
            ],
            "top_inhibitory_upstream": [
                {
                    "src_neuron": e.src_neuron,
                    "src_pole": e.src_pole.value,
                    "role": e.role.value,
                    "mean_causal_delta": round(e.mean_causal_delta, 4),
                    "max_causal_delta": round(e.max_causal_delta, 4),
                    "jacobian_sensitivity": round(e.jacobian_sensitivity, 4),
                    "virtual_weight_up": round(e.virtual_weight_up, 4),
                    "virtual_weight_gate": round(e.virtual_weight_gate, 4),
                }
                for e in self.top_inhibitory(5)
            ],
        }


@dataclass
class UpstreamAttributionBundle:
    """
    Represents a clustered bundle of upstream neurons feeding a target downstream neuron
    (Cammarata 'Explanations That Can Lose': clustering incoming attribution into semantic
    routes and measuring each bundle's total causal contribution and shared activating tokens).
    """
    dst_layer: str
    dst_neuron: int
    dst_pole: PoleType
    src_layer: str
    bundle_id: int
    upstream_neurons: List[int] = field(default_factory=list)
    dominant_pole: PoleType = PoleType.POSITIVE
    total_causal_attribution: float = 0.0
    mean_pairwise_cosine: float = 0.0
    shared_activating_tokens: List[str] = field(default_factory=list)


@dataclass
class ResidualSinkEntry:
    """
    Per-neuron residual-stream sink diagnostic entry ('Looking at the Residual Stream' /
    'Finding Clean Circuits'): measures outgoing write norm alongside maximum cosine alignment
    with all downstream MLP readers and vocabulary unembedding rows.
    """
    neuron_idx: int
    outgoing_weight_norm: float
    max_reader_cosine: float
    unembedding_alignment_cosine: float
    sink_null_fraction: float
    is_residual_sink_candidate: bool = False


@dataclass
class ResidualSinkReport:
    """
    Empirical audit of Residual Stream Sink neurons in `src_layer` (Cammarata: 'Looking at the
    Residual Stream' — identifying saliency-knob neurons that write significant norm into the
    residual stream to regulate LayerNorm/RMSNorm scale without aligning with downstream feature readers).
    """
    src_layer: str
    num_neurons_audited: int
    sink_candidate_indices: List[int] = field(default_factory=list)
    entries: List[ResidualSinkEntry] = field(default_factory=list)


@dataclass
class DownstreamReadoutReport:
    """
    Empirical audit report for Cammarata's 'Nothing Reads That Combination' / 'Do Weak Activations Matter?'
    hypothesis. Tests whether weak/secondary cross-activations across a group of neurons in `src_layer`
    cross the gating/activation threshold of any downstream neuron in `dst_layer` or shift output logits
    relative to a matched random-neuron null baseline.
    """
    src_layer: str
    dst_layer: str
    src_neurons: List[int]
    applied_thresholds: List[float]
    max_downstream_activation_shift: float
    mean_downstream_activation_shift: float
    max_downstream_gate_shift: float
    mean_downstream_gate_shift: float
    readout_tolerance: float
    affected_downstream_neurons: List[int] = field(default_factory=list)
    is_benign_spillover: bool = True
    output_kl_divergence: float = 0.0
    null_baseline_max_shift: float = 0.0
    null_baseline_kl: float = 0.0
    effect_vs_null_ratio: float = 0.0
    downstream_gate_flip_rate: float = 0.0
    relative_max_activation_shift: float = 0.0


@dataclass
class VerificationExperimentSpec:
    """
    Falsifiable contrastive, blind-maximum, near-miss, distant-superposition, and surgical
    ablation experiment proposed by the Explainer Agent ("Explanations That Can Lose" /
    "Explanations That Could Lose").
    """
    positive_pole_test_tokens: List[str] = field(default_factory=list)
    negative_pole_test_tokens: List[str] = field(default_factory=list)
    expected_promoted_tokens: List[str] = field(default_factory=list)
    expected_suppressed_tokens: List[str] = field(default_factory=list)
    expected_excited_dst_neurons: List[int] = field(default_factory=list)
    expected_inhibited_dst_neurons: List[int] = field(default_factory=list)
    blind_maximum_prompt: Optional[str] = None
    near_miss_test_tokens: List[str] = field(default_factory=list)
    distant_control_tokens: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """
    Empirical results from executing the agent's falsification & surgical ablation tests:
      - Standardized pole separation (Cohen's d) & share of corpus maximum (>= 30% strong response)
      - Blind maximum challenge ("Explanations That Can Lose")
      - Positive controls vs. Near misses vs. Distant superposition controls ("Explanations That Could Lose")
      - Surgical zero-ablation & scale-relative pole-flip vs. random-neuron control baseline
      - Held-out binned activation simulation correlation ("On Simulation Scores")
    """
    positive_test_mean_activation: float
    negative_test_mean_activation: float
    pole_separation_passed: bool  # True if standardized separation & sign split pass
    zero_ablation_target_logit_drop: float  # Drop in expected promoted logits when neuron is set to 0
    pole_flip_target_logit_shift: float  # Shift toward opposite pole logits when clamped to native negative scale
    downstream_circuit_match_ratio: float  # Fraction of predicted downstream neurons that shifted as predicted
    empirical_verification_score: float  # Overall falsifiable causal verification score in [0.0, 1.0]
    summary_notes: str = ""
    standardized_pole_separation_d: float = 0.0
    positive_share_of_corpus_max: float = 0.0
    blind_max_ratio_to_corpus_max: float = 0.0
    near_miss_mean_activation: float = 0.0
    distant_control_mean_activation: float = 0.0
    falsification_outcome: str = "untested"  # "won", "tied", "lost", or "untested"
    random_neuron_control_kl: float = 0.0
    target_neuron_ablation_kl: float = 0.0
    causal_kl_vs_random_ratio: float = 0.0
    held_out_simulation_correlation: float = 0.0
    branch_relatedness_score: float = 0.0


@dataclass
class BipolarVerbHypothesis:
    """
    Complete structured output produced by the Agentic Neuron & Circuit Explainer.
    Frames the neuron as a bipolar 'verb' (differential control-flow gate/router) rather than a static noun.
    """
    layer_name: str
    neuron_idx: int
    positive_pole_motif: str
    negative_pole_motif: str
    unified_verb_hypothesis: str
    recursive_offshoots: List[str]
    taxonomy_class: NeuronTaxonomy
    confidence_score: float
    proposed_experiment: VerificationExperimentSpec
    verification_report: Optional[VerificationReport] = None


@dataclass
class EcologyNeighborContext:
    """
    Represents an already-explained upstream, downstream, or same-layer sibling anchor clue
    injected into the Crossword / CircuitEcologyMap solver loop ('Understanding Compounds').
    """
    layer_name: str
    neuron_idx: int
    direction: str  # "upstream", "downstream", or "same_layer_sibling"
    role: str
    mean_causal_delta: float
    positive_pole_motif: str
    negative_pole_motif: str
    unified_verb_hypothesis: str
    confidence_score: float


@dataclass
class LayerEcologySolveReport:
    """
    Summary of a multi-pass Crossword / Circuit Ecology solve across a layer ('Understanding Compounds').
    Pass 1 solves high-selectivity monosemantic 'anchor' neurons and registers their circuit/sibling
    links; Pass 2 uses those verified anchor motifs to constrain and explain contextual neurons.
    """
    layer_name: str
    anchor_neurons_explained: List[int]
    contextual_neurons_explained: List[int]
    hypotheses: Dict[int, BipolarVerbHypothesis] = field(default_factory=dict)
    mean_verification_score: float = 0.0
    contextual_clues_injected_count: int = 0


@dataclass
class ContrastiveNeuronMatch:
    """
    Represents a single native-basis neuron ranked by its scale-invariant Welch's Bipolar
    Separation t-statistic across two contrastive prompt sets D+ and D-.
    """
    layer_name: str
    neuron_idx: int
    t_statistic: float  # Signed Welch's t-statistic: (mu_pos - mu_neg) / (sqrt(var_pos/N+ + var_neg/N-) + s0)
    abs_t_statistic: float
    pos_mean: float
    neg_mean: float
    pos_std: float
    neg_std: float
    bipolar_sign_flip: bool  # True if mu_pos and mu_neg have opposite signs (true bipolar verb split)
    recommended_pole: PoleType


@dataclass
class ContrastiveDiscoveryReport:
    """
    Report returned by `discover_contrastive_neurons` ranking native neurons across layers
    that best separate positive vs. negative behavioral datasets without an SAE.
    """
    layers_scanned: List[str]
    pos_tokens_count: int
    neg_tokens_count: int
    top_matches: List[ContrastiveNeuronMatch] = field(default_factory=list)


@dataclass
class NeuronCheckpointSnapshot:
    """
    Captures a single neuron's statistical, structural, and quantile-bucket state at one
    training checkpoint or fine-tuning stage ('Watching Neurons Develop' / 'Recursive Generalization').
    """
    checkpoint_label: str
    mean: float
    std: float
    pos_pole_selectivity: float
    neg_pole_selectivity: float
    pos_taxonomy: NeuronTaxonomy
    neg_taxonomy: NeuronTaxonomy
    outgoing_weight_norm: float
    weight_cosine_to_final: float
    peak_trunk_tokens: List[str] = field(default_factory=list)
    secondary_branch_tokens: List[str] = field(default_factory=list)
    tertiary_offshoot_tokens: List[str] = field(default_factory=list)
    double_negative_positive_rate: float = 0.0


@dataclass
class NeuronEvolutionReport:
    """
    Tracks how a neuron's selectivity, weight direction, and recursive offshoots evolve
    across a sequence of checkpoints or LoRA adapters ('Watching Neurons Develop').
    """
    layer_name: str
    neuron_idx: int
    snapshots: List[NeuronCheckpointSnapshot] = field(default_factory=list)
    emerged_offshoot_tokens: List[str] = field(default_factory=list)
    selectivity_delta_pos: float = 0.0
    selectivity_delta_neg: float = 0.0


@dataclass
class SynergisticAblationResult:
    """
    Result of `AblationMaskOptimizer` identifying a minimal coalition of k neurons
    whose joint surgical ablation overcomes backup circuits ('Circuit Ecology') to maximize
    causal impact or flip a target behavior.
    """
    layer_name: str
    method: str  # "greedy" or "l1_optim"
    selected_neurons: List[int]
    stepwise_scores: List[float]
    joint_ablation_kl: float
    sum_of_individual_kls: float
    synergy_ratio: float  # joint_ablation_kl / max(sum_of_individual_kls, 1e-9)
    target_logit_shift: float = 0.0


@dataclass
class NeuronScanSummary:
    """
    Compact layer-wide audit metrics for a single neuron used in rankings and heatmaps.
    Exposes both the statistical activation-shape prior (`sparsity_tail_prior`) and
    `taxonomy_class` (with `semantic_verified=True` when empirically verified by `NeuronExplainerAgent`).
    """
    neuron_idx: int
    taxonomy_class: NeuronTaxonomy
    pole_selectivity_score: float
    bipolar_balance: float
    positive_firing_rate: float
    negative_firing_rate: float
    mean: float
    std: float
    max_val: float
    min_val: float
    outgoing_weight_norm: float
    estimated_causal_impact: float
    ablation_kl_divergence: Optional[float] = None
    top_pos_token: Optional[str] = None
    top_neg_token: Optional[str] = None
    pos_pole_selectivity_score: float = 0.0
    neg_pole_selectivity_score: float = 0.0
    pos_taxonomy_class: NeuronTaxonomy = NeuronTaxonomy.DEAD_OR_NOISY
    neg_taxonomy_class: NeuronTaxonomy = NeuronTaxonomy.DEAD_OR_NOISY
    double_negative_positive_rate: float = 0.0
    semantic_verified: bool = False
    sparsity_tail_prior: SparsityTailPrior = SparsityTailPrior.DEAD_OR_FLAT
    lexical_selectivity_r2: float = 0.0


@dataclass
class LayerCleanlinessReport:
    """
    Aggregated layer-wide cleanliness audit report mirroring Cammarata (2026).
    Provides statistical prior counts alongside whole-neuron and independent pole-level
    taxonomy percentages.
    """
    layer_name: str
    num_neurons: int
    tokens_scanned: int
    taxonomy_counts: Dict[str, int]
    taxonomy_percentages: Dict[str, float]
    clean_monosemantic_pct: float
    structured_recursive_pct: float
    combined_clean_and_structured_pct: float
    mean_selectivity_score: float
    neurons: List[NeuronScanSummary] = field(default_factory=list)
    pole_taxonomy_counts: Dict[str, int] = field(default_factory=dict)
    pole_taxonomy_percentages: Dict[str, float] = field(default_factory=dict)
    clean_poles_pct: float = 0.0
    structured_poles_pct: float = 0.0
    combined_clean_and_structured_poles_pct: float = 0.0
    prior_counts: Dict[str, int] = field(default_factory=dict)
    semantically_verified_count: int = 0


@dataclass
class CompiledSteeringVector:
    """
    Represents a pre-compiled residual-stream (or layer-output) steering vector v_steer
    synthesized from one or more unconditional neuron-basis interventions in `layer_name`.
    `clamped_neurons` tracks indices whose dynamic activation must be zeroed so that adding
    `v_steer` reproduces exact `HARD_CLAMP` semantics.
    """
    layer_name: str
    d_model: int
    source_neurons: List[int]
    source_deltas: List[float]
    vector_norm: float
    vector_values: List[float]  # Serialized 1D vector of length d_model
    clamped_neurons: List[int] = field(default_factory=list)


@dataclass
class ProductionPresetBundle:
    """
    Portable, JSON-serializable production configuration containing both
    surgical neuron-basis hook specifications and pre-compiled residual steering vectors.
    """
    preset_name: str
    model_name: str
    created_at_utc: str
    description: str
    interventions: List[Dict[str, Any]]
    compiled_vectors: List[CompiledSteeringVector] = field(default_factory=list)
