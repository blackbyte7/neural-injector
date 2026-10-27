from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class PoleType(str, Enum):
    """Represents which pole of a neuron's activation distribution to target."""
    POSITIVE = "positive"  # Activations > 0 (excitatory motif)
    NEGATIVE = "negative"  # Activations < 0 (inhibitory / opposing motif)
    BOTH = "both"  # Entire neuron basis across all values


class InterventionMode(str, Enum):
    """Supported surgical neuron manipulation strategies."""
    ZERO_ABLATE = "zero_ablate"  # Set target neuron(s) directly to 0.0
    HARD_CLAMP = "hard_clamp"  # Force neuron(s) to a fixed constant c
    SCALE = "scale"  # Multiply neuron activation by alpha
    ADDITIVE_STEER = "additive_steer"  # Shift activation by +delta (concept injection)
    WEAK_PRUNE = "weak_prune"  # Zero out weak activations where |a| < threshold (tests "Weak Activations Don't Matter")
    POLE_RECTIFY = "pole_rectify"  # Zero out one pole while preserving the other
    CONDITIONAL_GATE = "conditional_gate"  # Closed-loop dynamic gating triggered by an upstream/detector neuron


class NeuronTaxonomy(str, Enum):
    """
    Classification of a neuron's (or single pole's) cleanliness following Cammarata (2026):
      - CLEAN_MONOSEMANTIC (~90% of neuron poles): Performs a single clear core job / differential split.
      - STRUCTURED_RECURSIVE_MIXTURE (~7%): Core trunk motif with related semantic offshoots.
      - TRUE_POLYSEMANTIC (~3%): Unrelated concepts superposed on the same pole.
      - DEAD_OR_NOISY: Near-zero variance or flat unselective firing.
    """
    CLEAN_MONOSEMANTIC = "clean_monosemantic"
    STRUCTURED_RECURSIVE_MIXTURE = "structured_recursive_mixture"
    TRUE_POLYSEMANTIC = "true_polysemantic"
    DEAD_OR_NOISY = "dead_or_noisy"


@dataclass
class ActivationSample:
    """
    Stores a single top-activating or quantile-stratified context example for a neuron pole.
    Includes `next_token_str` and `window_str` to surface forward next-token prediction motifs
    (e.g., Cammarata's 'Titanic -> boat' finding) and `is_double_negative_route` to flag
    SwiGLU activations where `gate < 0` and `up < 0` multiply to a positive value.
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
    and SwiGLU double-negative route diagnostics.
    """
    layer_name: str
    neuron_idx: int
    mean: float
    std: float
    max_val: float
    min_val: float
    positive_firing_rate: float
    negative_firing_rate: float
    pole_selectivity_score: float  # Combined full-distribution 4th-moment tail selectivity
    top_positive_samples: List[ActivationSample] = field(default_factory=list)
    top_negative_samples: List[ActivationSample] = field(default_factory=list)
    top_promoted_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_suppressed_tokens: List[Tuple[str, float]] = field(default_factory=list)

    # Pole-specific conditional statistics (Cammarata Section 3.1: Evaluating Poles Independently)
    pos_pole_mean: float = 0.0
    pos_pole_std: float = 0.0
    pos_pole_selectivity_score: float = 0.0
    pos_taxonomy_class: NeuronTaxonomy = NeuronTaxonomy.DEAD_OR_NOISY
    neg_pole_mean: float = 0.0
    neg_pole_std: float = 0.0
    neg_pole_selectivity_score: float = 0.0
    neg_taxonomy_class: NeuronTaxonomy = NeuronTaxonomy.DEAD_OR_NOISY

    # Stratified quantile activation buckets per pole (Cammarata Section 3.3: Recursive Generalization Offshoots)
    # Keys: "peak_trunk" ([80%..100%]), "secondary_branch" ([40%..80%)), "tertiary_offshoot" ([15%..40%))
    positive_quantile_buckets: Dict[str, List[ActivationSample]] = field(default_factory=dict)
    negative_quantile_buckets: Dict[str, List[ActivationSample]] = field(default_factory=dict)

    # SwiGLU Double-Negative Route diagnostics (gate < 0 and up < 0 -> silu(gate)*up > 0)
    double_negative_positive_rate: float = 0.0
    double_negative_samples: List[ActivationSample] = field(default_factory=list)

    @property
    def is_clean_candidate(self) -> bool:
        """Heuristic following Cammarata (2026): high tail selectivity on either pole or overall."""
        return (
            self.pole_selectivity_score >= 2.5
            or self.pos_pole_selectivity_score >= 2.1
            or self.neg_pole_selectivity_score >= 2.1
        )


@dataclass
class GateConditionSpec:
    """
    Defines a closed-loop conditional trigger for `InterventionMode.CONDITIONAL_GATE` (Section 5.2):
    Only applies `action_mode` to the target neuron(s) when `detector_neuron` in `detector_layer`
    fires on `detector_pole` with magnitude exceeding `threshold`.
    """
    detector_layer: str
    detector_neuron: int
    detector_pole: PoleType = PoleType.POSITIVE
    threshold: float = 0.5
    action_mode: InterventionMode = InterventionMode.ADDITIVE_STEER
    sequence_scope: str = "token_wise"  # "token_wise" (per token position) or "any_token" (whole sequence)


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
        if abs(self.virtual_weight_gate) > 2.0 * abs(self.virtual_weight_up) and abs(self.virtual_weight_gate) > 0.1:
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
class DownstreamReadoutReport:
    """
    Empirical audit report for Cammarata's 'Nothing Reads That Combination' hypothesis (Section 3.5).
    Tests whether weak/secondary cross-activations across a group of neurons in `src_layer`
    actually cross the gating/activation threshold of any downstream neuron in `dst_layer`.
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


@dataclass
class VerificationExperimentSpec:
    """
    Contrastive test and surgical ablation experiment proposed by the Explainer Agent
    to empirically test its own hypothesis.
    """
    positive_pole_test_tokens: List[str] = field(default_factory=list)
    negative_pole_test_tokens: List[str] = field(default_factory=list)
    expected_promoted_tokens: List[str] = field(default_factory=list)
    expected_suppressed_tokens: List[str] = field(default_factory=list)
    expected_excited_dst_neurons: List[int] = field(default_factory=list)
    expected_inhibited_dst_neurons: List[int] = field(default_factory=list)


@dataclass
class VerificationReport:
    """
    Empirical results from executing the agent's proposed contrastive & surgical ablation tests
    through NeuralInjector.
    """
    positive_test_mean_activation: float
    negative_test_mean_activation: float
    pole_separation_passed: bool  # True if Pos inputs -> >0 and Neg inputs -> <0
    zero_ablation_target_logit_drop: float  # Drop in expected promoted logits when neuron is set to 0
    pole_flip_target_logit_shift: float  # Shift toward opposite pole logits when clamped to negative pole
    downstream_circuit_match_ratio: float  # Fraction of predicted downstream neurons that shifted as predicted
    empirical_verification_score: float  # Overall causal verification score in [0.0, 1.0]
    summary_notes: str = ""


@dataclass
class BipolarVerbHypothesis:
    """
    Complete structured output produced by the Agentic Neuron & Circuit Explainer.
    Frames the neuron as a bipolar 'verb' (differential router) rather than a static noun.
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
    Represents an already-explained upstream or downstream neighbor clue injected into
    the Crossword / CircuitEcologyMap solver loop (Cammarata Section 3.2: 'Understanding Compounds').
    """
    layer_name: str
    neuron_idx: int
    direction: str  # "upstream" or "downstream"
    role: str
    mean_causal_delta: float
    positive_pole_motif: str
    negative_pole_motif: str
    unified_verb_hypothesis: str
    confidence_score: float


@dataclass
class LayerEcologySolveReport:
    """
    Summary of a multi-pass Crossword / Circuit Ecology solve across a layer.
    Pass 1 solves high-selectivity monosemantic 'anchor' neurons; Pass 2 uses those
    verified anchor motifs to constrain and explain harder recursive-mixture neurons.
    """
    layer_name: str
    anchor_neurons_explained: List[int]
    contextual_neurons_explained: List[int]
    hypotheses: Dict[int, BipolarVerbHypothesis] = field(default_factory=dict)
    mean_verification_score: float = 0.0


@dataclass
class ContrastiveNeuronMatch:
    """
    Represents a single native-basis neuron ranked by its Bipolar Separation t-statistic
    across two contrastive prompt sets D+ and D- (Section 5.3).
    """
    layer_name: str
    neuron_idx: int
    t_statistic: float  # Signed Welch's t-statistic: (mu_pos - mu_neg) / sqrt(var_pos/N+ + var_neg/N-)
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
    training checkpoint or fine-tuning stage (Section 5.4).
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


@dataclass
class NeuronEvolutionReport:
    """
    Tracks how a neuron's selectivity, weight direction, and recursive offshoots evolve
    across a sequence of checkpoints or LoRA adapters (Section 5.4).
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
    Result of `AblationMaskOptimizer` (Section 5.5) identifying a minimal coalition of k neurons
    whose joint surgical ablation overcomes backup circuits to maximize causal impact or flip behavior.
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
    Note: Initial taxonomy classification is a statistical tail-selectivity prior unless
    `semantic_verified=True` (updated after closed-loop agentic verification).
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


@dataclass
class LayerCleanlinessReport:
    """
    Aggregated layer-wide cleanliness audit report mirroring Cammarata (2026).
    Provides both whole-neuron and independent pole-level taxonomy percentages.
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


@dataclass
class CompiledSteeringVector:
    """
    Represents a pre-compiled residual-stream steering vector v_steer in R^{d_model}
    synthesized from one or more unconditional neuron-basis interventions in `layer_name`.
    `clamped_neurons` tracks indices whose incoming dynamic activation must be zeroed
    prior to `down_proj` so that adding `v_steer` reproduces exact `HARD_CLAMP` semantics.
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
