from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class PoleType(str, Enum):
    """Which pole of a neuron's activation distribution to target."""
    POSITIVE = "positive"  # a > 0
    NEGATIVE = "negative"  # a < 0
    BOTH = "both"          # every value


class SignRegime(str, Enum):
    """
    Sign regime of a single activation ('Separating the Sign Regimes').

    For gated MLPs (SwiGLU / GeGLU) a = act(g) * u, and because act(g) < 0 for g < 0 the
    product's sign is decided by the signs of the gate pre-activation g and value u:
      ++ : g > 0, u > 0 -> a > 0   (open gate, positive value)
      -- : g < 0, u < 0 -> a > 0   (double-negative regime)
      +- : g > 0, u < 0 -> a < 0
      -+ : g < 0, u > 0 -> a < 0
    For non-gated MLPs only the pole is known ("+" / "-").
    """
    PP = "++"
    NN = "--"
    PN = "+-"
    NP = "-+"
    POS = "+"
    NEG = "-"

    @property
    def pole(self) -> PoleType:
        return PoleType.POSITIVE if self in (SignRegime.PP, SignRegime.NN, SignRegime.POS) else PoleType.NEGATIVE


GATED_REGIMES: Tuple[SignRegime, ...] = (SignRegime.PP, SignRegime.NN, SignRegime.PN, SignRegime.NP)
UNGATED_REGIMES: Tuple[SignRegime, ...] = (SignRegime.POS, SignRegime.NEG)


class InterventionMode(str, Enum):
    """Supported surgical neuron manipulation strategies."""
    ZERO_ABLATE = "zero_ablate"            # a <- 0
    HARD_CLAMP = "hard_clamp"              # a <- c
    SCALE = "scale"                        # a <- alpha * a
    ADDITIVE_STEER = "additive_steer"      # a <- a + delta
    WEAK_PRUNE = "weak_prune"              # a <- 0 where |a| < threshold ('Do Weak Activations Matter?')
    POLE_RECTIFY = "pole_rectify"          # keep only the selected pole, zero the other
    CONDITIONAL_GATE = "conditional_gate"  # apply `gate_condition.action_mode` only when a detector neuron fires
    EXTREME_CULL = "extreme_cull"          # a <- 0 where the signal (gate or activation) exceeds a per-neuron threshold
                                           # ('Memorization and Composition': zero output when gate > ordinary-text p99.9)
    DOUBLE_NEGATIVE_ABLATE = "double_negative_ablate"  # a <- 0 where g < 0 and u < 0 (the '--' regime)


class NeuronTaxonomy(str, Enum):
    """
    Semantic judgement of a single neuron *pole*, as made by an explainer agent and then put
    at risk by the blind test protocol ('Explanations That Could Lose'). The categories follow
    the paper's audit language ("roughly 90% of neuron poles have one main job, often with small
    related offshoots; another ~7% have two related jobs; and only ~2-4% are not yet understood").

    These labels are never assigned from activation statistics. A pole is `UNEXPLAINED` until an
    explainer agent has written an explanation for it.
    """
    ONE_MAIN_JOB = "one_main_job"              # one job, possibly with related offshoots
    FEW_RELATED_JOBS = "few_related_jobs"      # two or three related jobs
    UNRELATED_JOBS = "unrelated_jobs"          # the explanation needs unrelated jobs (what packing would look like)
    NOT_YET_UNDERSTOOD = "not_yet_understood"  # the agent could not form a specific explanation
    UNEXPLAINED = "unexplained"                # no explanation has been written yet


class FalsificationOutcome(str, Enum):
    """Outcome of the positive-control vs near-miss comparison ('Explanations That Could Lose')."""
    WON = "won"
    TIED = "tied"
    LOST = "lost"
    UNTESTED = "untested"


# =============================================================================
# Audited activation examples and statistics
# =============================================================================
@dataclass
class ActivationSample:
    """
    One audited activation of a neuron at one token (or spatial) position, with the context
    needed to read it both backward (what came before) and forward (what comes next; the
    paper's 'Titanic -> boat' point: many apparent extra jobs are next-token preparation).
    """
    activation_value: float
    sample_id: int
    position_idx: Optional[int] = None
    token_str: Optional[str] = None
    context_str: Optional[str] = None          # full text of the sample the token came from
    next_token_str: Optional[str] = None
    window_str: Optional[str] = None           # "... prev tokens [FOCAL] next tokens ..."
    sign_regime: Optional[str] = None          # SignRegime value, None if unknown
    fraction_of_pole_extreme: Optional[float] = None  # |a| / |corpus extreme of this pole|

    @property
    def is_double_negative_route(self) -> bool:
        return self.sign_regime == SignRegime.NN.value


@dataclass
class DistributionMoments:
    """Exact streaming moments of a set of values (central moments, population normalisation)."""
    count: int = 0
    rate: float = 0.0                 # count / tokens observed
    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    excess_kurtosis: float = 0.0
    extreme: float = 0.0              # max for positive sets, min for negative sets


@dataclass
class ActivationBand:
    """
    One ring of the 'every strong firing' picture: activations of one pole whose magnitude lies
    in [lower_fraction, upper_fraction) of that pole's corpus extreme. Counts are exact (from the
    streaming log-magnitude histogram, so band edges are snapped to histogram bin edges and the
    snapped edges are reported); `samples` are uniform random examples from inside the band.
    """
    pole: PoleType
    lower_fraction: float
    upper_fraction: float
    lower_value: float
    upper_value: float
    count: int
    regime_counts: Dict[str, int] = field(default_factory=dict)
    samples: List[ActivationSample] = field(default_factory=list)

    @property
    def regime_shares(self) -> Dict[str, float]:
        tot = sum(self.regime_counts.values())
        return {k: (v / tot if tot else 0.0) for k, v in self.regime_counts.items()}


@dataclass
class NeuronPoleProfile:
    """
    Descriptive profile of one neuron, built so a reader (human or agent) can look at each pole
    and each sign regime separately without the neuron being replaced by anything.
    Nothing in this profile is a judgement of whether the neuron is 'clean'.
    """
    layer_name: str
    neuron_idx: int
    tokens_observed: int
    mean: float
    std: float
    max_val: float
    min_val: float
    positive_firing_rate: float
    negative_firing_rate: float
    gated: bool                                     # True if gate/up sign regimes were recorded
    positive_pole: DistributionMoments = field(default_factory=DistributionMoments)
    negative_pole: DistributionMoments = field(default_factory=DistributionMoments)  # moments of |a| for a < 0
    regime_moments: Dict[str, DistributionMoments] = field(default_factory=dict)
    swiglu_sign_regime_rates: Dict[str, float] = field(default_factory=dict)   # share of all tokens
    double_negative_positive_rate: float = 0.0      # share of a > 0 tokens that are in the -- regime

    # Raw top lists (a > 0 and a < 0, every regime mixed, each sample tagged with its regime)
    top_positive_samples: List[ActivationSample] = field(default_factory=list)
    top_negative_samples: List[ActivationSample] = field(default_factory=list)
    # The same lists separated by sign regime
    regime_top_samples: Dict[str, List[ActivationSample]] = field(default_factory=dict)
    # Fraction-of-extreme rings per pole
    positive_bands: List[ActivationBand] = field(default_factory=list)
    negative_bands: List[ActivationBand] = field(default_factory=list)

    # Direct logit attribution of the write vector (filled by NeuralInjector.get_neuron_profile)
    top_promoted_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_suppressed_tokens: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def double_negative_samples(self) -> List[ActivationSample]:
        return self.regime_top_samples.get(SignRegime.NN.value, [])

    def pole_extreme(self, pole: PoleType) -> float:
        return self.max_val if pole == PoleType.POSITIVE else self.min_val


@dataclass
class NeuronScanSummary:
    """Per-neuron descriptive row of a layer scan (no statistical 'cleanliness' label)."""
    neuron_idx: int
    mean: float
    std: float
    max_val: float
    min_val: float
    positive_firing_rate: float
    negative_firing_rate: float
    positive_excess_kurtosis: float
    negative_excess_kurtosis: float
    double_negative_positive_rate: float
    outgoing_weight_norm: float
    rms_write_norm: float                  # sqrt(E[a^2]) * ||W_down[:, i]||: RMS norm of what the neuron writes
    top_pos_token: Optional[str] = None
    top_neg_token: Optional[str] = None
    ablation_kl_divergence: Optional[float] = None
    explained: bool = False
    pos_taxonomy: NeuronTaxonomy = NeuronTaxonomy.UNEXPLAINED
    neg_taxonomy: NeuronTaxonomy = NeuronTaxonomy.UNEXPLAINED
    pos_outcome: FalsificationOutcome = FalsificationOutcome.UNTESTED
    neg_outcome: FalsificationOutcome = FalsificationOutcome.UNTESTED


@dataclass
class ExplanationAuditSummary:
    """
    Aggregate of explained-and-tested poles. Only this is a cleanliness estimate in the paper's
    sense, and only when the explained neurons were sampled at random (`sampling`).
    """
    poles_explained: int = 0
    taxonomy_counts: Dict[str, int] = field(default_factory=dict)
    outcome_counts: Dict[str, int] = field(default_factory=dict)
    positive_control_strong_rate: float = 0.0
    near_miss_strong_rate: float = 0.0
    distant_strong_rate: float = 0.0
    poles_never_fired_on_distant: int = 0
    sampling: str = "unspecified"   # "uniform_random(seed=...)" or "user_selected"


@dataclass
class LayerScanReport:
    """Layer-wide descriptive scan plus the aggregate of any explanation audits on the layer."""
    layer_name: str
    num_neurons: int
    tokens_scanned: int
    gated: bool
    neurons: List[NeuronScanSummary] = field(default_factory=list)
    layer_double_negative_positive_rate: float = 0.0   # share of all a > 0 activations in the layer that are --
    explanation_audit: ExplanationAuditSummary = field(default_factory=ExplanationAuditSummary)


# =============================================================================
# Interventions
# =============================================================================
@dataclass
class GateConditionSpec:
    """
    Closed-loop trigger for `InterventionMode.CONDITIONAL_GATE`: `action_mode` is applied to the
    target neuron(s) only where `detector_neuron` in `detector_layer` fires on `detector_pole`
    beyond `threshold`. With `sequence_scope == "any_token"` and `sticky_on_decode`, a trigger
    during prefill latches across later single-token KV-cached decoding steps.
    """
    detector_layer: str
    detector_neuron: int
    threshold: float
    detector_pole: PoleType = PoleType.POSITIVE
    action_mode: InterventionMode = InterventionMode.ADDITIVE_STEER
    sequence_scope: str = "token_wise"
    sticky_on_decode: bool = True


@dataclass
class InterventionSpec:
    """
    One surgical intervention on neurons of a layer.
      - `value` / `per_neuron_values`: clamp constant, scale factor, shift, prune threshold or cull threshold.
      - `condition_signal`: for EXTREME_CULL, which quantity is compared with the threshold:
        "gate" (the gate pre-activation, as in the paper's p99.9 experiment) or "activation".
      - `token_positions`: restrict to positions (None = all).
    """
    layer_name: str
    neuron_indices: List[int]
    mode: InterventionMode = InterventionMode.ZERO_ABLATE
    pole: PoleType = PoleType.BOTH
    value: float = 0.0
    per_neuron_values: Optional[List[float]] = None
    token_positions: Optional[List[int]] = None
    gate_condition: Optional[GateConditionSpec] = None
    condition_signal: str = "activation"
    custom_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    enabled: bool = True

    @property
    def needs_gate_signal(self) -> bool:
        return self.mode == InterventionMode.DOUBLE_NEGATIVE_ABLATE or (
            self.mode == InterventionMode.EXTREME_CULL and self.condition_signal == "gate"
        )


# =============================================================================
# Circuits
# =============================================================================
class InteractionRole(str, Enum):
    """How an upstream neuron pole influences a downstream neuron (from measured path terms)."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    GATING_MODULATOR = "gating"


@dataclass
class CircuitEdge:
    """
    Directed connection from (src_layer, src_neuron, src_pole) to (dst_layer, dst_neuron).
    `mean_causal_delta` = E[A_dst(clean) - A_dst(src pole ablated)] over the source's active positions.
    For gated destination neurons, `gate_path_delta` / `up_path_delta` split the measured change into
    the first-order gate term act'(g) * u * dg and value term act(g) * du.
    """
    src_layer: str
    src_neuron: int
    src_pole: PoleType
    dst_layer: str
    dst_neuron: int
    dst_pole_tendency: PoleType
    mean_causal_delta: float
    max_causal_delta: float
    clean_dst_mean: float
    ablated_dst_mean: float
    jacobian_sensitivity: float = 0.0
    virtual_weight_up: float = 0.0
    virtual_weight_gate: float = 0.0
    static_alignment_cos: float = 0.0
    gate_path_delta: Optional[float] = None
    up_path_delta: Optional[float] = None

    @property
    def role(self) -> InteractionRole:
        if (
            self.gate_path_delta is not None
            and self.up_path_delta is not None
            and abs(self.gate_path_delta) > abs(self.up_path_delta)
        ):
            return InteractionRole.GATING_MODULATOR
        return InteractionRole.EXCITATORY if self.mean_causal_delta >= 0 else InteractionRole.INHIBITORY


@dataclass
class CircuitGraph:
    """A traced subgraph: one source -> many destinations, or many sources -> one destination."""
    src_layer: str
    dst_layer: str
    src_neuron: int
    src_pole: PoleType
    edges: List[CircuitEdge] = field(default_factory=list)
    dst_neuron: Optional[int] = None

    def top_excitatory(self, k: int = 5) -> List[CircuitEdge]:
        pos = [e for e in self.edges if e.mean_causal_delta > 0]
        return sorted(pos, key=lambda e: e.mean_causal_delta, reverse=True)[:k]

    def top_inhibitory(self, k: int = 5) -> List[CircuitEdge]:
        neg = [e for e in self.edges if e.mean_causal_delta < 0]
        return sorted(neg, key=lambda e: e.mean_causal_delta)[:k]

    @staticmethod
    def _edge_dict(e: CircuitEdge, key: str) -> Dict[str, Any]:
        out = {
            key: e.dst_neuron if key == "dst_neuron" else e.src_neuron,
            "role": e.role.value,
            "mean_causal_delta": round(e.mean_causal_delta, 5),
            "jacobian_sensitivity": round(e.jacobian_sensitivity, 5),
            "virtual_weight_up": round(e.virtual_weight_up, 5),
            "virtual_weight_gate": round(e.virtual_weight_gate, 5),
        }
        if key == "src_neuron":
            out["src_pole"] = e.src_pole.value
        return out

    def to_agent_payload(self) -> Dict[str, Any]:
        return {
            "source": {"layer": self.src_layer, "neuron_idx": self.src_neuron, "pole": self.src_pole.value},
            "target_layer": self.dst_layer,
            "top_excited_downstream": [self._edge_dict(e, "dst_neuron") for e in self.top_excitatory(5)],
            "top_inhibited_downstream": [self._edge_dict(e, "dst_neuron") for e in self.top_inhibitory(5)],
        }

    def to_incoming_agent_payload(self) -> Dict[str, Any]:
        return {
            "source_layer": self.src_layer,
            "target": {"layer": self.dst_layer, "neuron_idx": self.dst_neuron, "pole": self.src_pole.value},
            "top_excitatory_upstream": [self._edge_dict(e, "src_neuron") for e in self.top_excitatory(5)],
            "top_inhibitory_upstream": [self._edge_dict(e, "src_neuron") for e in self.top_inhibitory(5)],
        }


@dataclass
class UpstreamAttributionBundle:
    """
    A cluster of upstream neurons whose attribution into the target co-varies across the target's
    activating contexts ('Explanations That Can Lose': cluster the attribution flowing into a neuron).
    """
    dst_layer: str
    dst_neuron: int
    dst_pole: PoleType
    src_layer: str
    bundle_id: int
    upstream_neurons: List[int] = field(default_factory=list)
    attribution_share: float = 0.0           # share of total |attribution| (over selected upstream neurons)
    mean_within_cosine: float = 0.0          # mean cosine between member attribution profiles
    silhouette: float = 0.0                  # silhouette of the best clustering found
    silhouette_null_p: float = float("nan")  # share of permutation-null silhouettes >= observed (bundles split only if < alpha)
    driving_context_indices: List[int] = field(default_factory=list)  # contexts this bundle drives most
    driving_contexts: List[str] = field(default_factory=list)


@dataclass
class ResidualCoordinateEntry:
    """One residual-stream coordinate: share of activation mass vs share of what downstream readers read."""
    coordinate: int
    mass_share: float       # E[x_d^2] / sum_d E[x_d^2]
    read_share: float       # sum over readers of ||W_read[:, d] * gamma_d||^2 / total
    is_sink_candidate: bool


@dataclass
class ResidualSinkReport:
    """
    'Looking at the Residual Stream': a coordinate that carries a large share of the activation
    mass while downstream readers barely read it acts as a saliency knob through LayerNorm/RMSNorm.
    A coordinate is a candidate when its mass share exceeds the uniform share 1/d_model and exceeds its
    read share (both measured, nothing tuned); candidates are ordered by mass share.
    """
    residual_site: str
    d_model: int
    tokens_observed: int
    coordinates: List[ResidualCoordinateEntry] = field(default_factory=list)
    sink_candidate_coordinates: List[int] = field(default_factory=list)
    neuron_write_share_on_sinks: Dict[str, float] = field(default_factory=dict)  # 'layer#neuron' -> share of ||w||^2 on sink coordinates


@dataclass
class CombinationReadoutReport:
    """
    'Nothing reads that combination': does any downstream reader align with the combination
    sum_i c_i W_down[:, i] of a set of source neurons more than with random combinations of the
    same size and coefficient magnitudes, and more than with the best single constituent?
    """
    src_layer: str
    dst_layers: List[str]
    src_neurons: List[int]
    coefficients: List[float]
    combination_max_reader_cosine: float
    best_reader: Tuple[str, int, str]                 # (layer, reader index, "up"|"gate")
    best_single_constituent_cosine: float
    null_samples: int
    null_mean: float
    null_p95: float
    combination_percentile_in_null: float            # fraction of null samples below the observed value
    reads_combination_beyond_constituents: bool
    causal_interaction: Optional[Dict[str, float]] = None


@dataclass
class InterventionEffectReport:
    """Effect of a set of interventions on next-token prediction over a set of batches."""
    label: str
    tokens_evaluated: int
    clean_nll: float
    intervened_nll: float
    mean_kl: float
    top1_agreement: float
    flipped_positions: int = 0
    flipped_clean_margin_mean: float = 0.0     # mean clean p(top1) - p(top2) at flipped positions
    all_clean_margin_mean: float = 0.0
    entries_modified: int = 0                  # number of (token, neuron) activations changed


@dataclass
class ExtremeCullReport:
    """'Remove extreme responses': gate-percentile culling vs random interventions matched by count / magnitude."""
    layers: List[str]
    percentile: float
    signal: str
    culled: InterventionEffectReport
    count_matched_random: InterventionEffectReport
    magnitude_matched_random: InterventionEffectReport
    generations: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Explanations and the blind test protocol
# =============================================================================
@dataclass
class BlindTestSuite:
    """
    Written by the test-writer agent, which sees only the explanation of one pole
    ('blind maximum', 'Explanations That Could Lose').
    """
    pole: PoleType
    explanation: str
    blind_maximum: str = ""
    positive_examples: List[str] = field(default_factory=list)
    near_misses: List[str] = field(default_factory=list)
    distant_concept: str = ""
    distant_examples: List[str] = field(default_factory=list)


@dataclass
class TextActivation:
    """Peak response of the neuron's pole to one text, as a share of the pole's corpus extreme."""
    text: str
    peak_activation: float
    share_of_corpus_extreme: float
    strong: bool
    peak_token: Optional[str] = None
    peak_position: Optional[int] = None
    peak_sign_regime: Optional[str] = None


@dataclass
class PoleVerificationReport:
    """Result of running a BlindTestSuite on the live model."""
    pole: PoleType
    corpus_extreme: float
    strong_fraction: float
    blind_maximum: Optional[TextActivation] = None
    positive_controls: List[TextActivation] = field(default_factory=list)
    near_misses: List[TextActivation] = field(default_factory=list)
    distant: List[TextActivation] = field(default_factory=list)
    positive_strong_count: int = 0
    near_miss_strong_count: int = 0
    distant_strong_count: int = 0
    outcome: FalsificationOutcome = FalsificationOutcome.UNTESTED

    @property
    def blind_max_ratio(self) -> float:
        return self.blind_maximum.share_of_corpus_extreme if self.blind_maximum else 0.0


@dataclass
class CausalCheckReport:
    """
    Supplementary causal checks on the positive-control texts: ablation KL of the neuron compared
    with ablation KL of randomly chosen neurons of the same layer on the same texts, and the
    model's continuations with the neuron turned up / down to its corpus extremes.
    """
    target_ablation_kl: float
    control_ablation_kls: List[float] = field(default_factory=list)
    target_kl_percentile_among_controls: float = 0.0
    steering_generations: Dict[str, str] = field(default_factory=dict)


@dataclass
class BundleBlindMaximumResult:
    """Blind maximum for one upstream attribution bundle while every other bundle is ablated."""
    bundle_id: int
    upstream_neurons: List[int]
    bundle_explanation: str
    blind_maximum: str
    blind_max_activation_under_ablation: float
    best_corpus_activation_under_ablation: float
    ratio_to_corpus_under_ablation: float
    related_to_main_explanation: Optional[bool] = None


@dataclass
class BipolarVerbHypothesis:
    """
    An explanation of one neuron as a distinction between its two poles ('verb', not 'noun'),
    together with the evidence that put it at risk.
    """
    layer_name: str
    neuron_idx: int
    positive_pole_motif: str
    negative_pole_motif: str
    unified_verb_hypothesis: str
    recursive_offshoots: List[str] = field(default_factory=list)
    pos_taxonomy: NeuronTaxonomy = NeuronTaxonomy.UNEXPLAINED
    neg_taxonomy: NeuronTaxonomy = NeuronTaxonomy.UNEXPLAINED
    confidence_score: float = 0.0
    test_suites: Dict[str, BlindTestSuite] = field(default_factory=dict)
    verification: Dict[str, PoleVerificationReport] = field(default_factory=dict)
    causal_checks: Optional[CausalCheckReport] = None
    bundle_results: List[BundleBlindMaximumResult] = field(default_factory=list)
    neighbor_clues_used: int = 0

    def outcome(self, pole: PoleType) -> FalsificationOutcome:
        rep = self.verification.get(pole.value)
        return rep.outcome if rep is not None else FalsificationOutcome.UNTESTED

    def taxonomy(self, pole: PoleType) -> NeuronTaxonomy:
        return self.pos_taxonomy if pole == PoleType.POSITIVE else self.neg_taxonomy


@dataclass
class EcologyNeighborContext:
    """An already-explained neuron connected to the target by a traced circuit edge or shared reader."""
    layer_name: str
    neuron_idx: int
    direction: str          # "upstream" | "downstream" | "shares_downstream_reader"
    role: str
    mean_causal_delta: float
    positive_pole_motif: str
    negative_pole_motif: str
    unified_verb_hypothesis: str
    positive_outcome: str
    negative_outcome: str


@dataclass
class LayerEcologySolveReport:
    """
    Crossword solve ('Understanding Compounds'): pass 1 explains the seed neurons and traces their
    real circuit edges; pass 2 explains neurons connected to solved ones, with the solved neighbors'
    explanations as evidence.
    """
    layer_names: List[str]
    pass1_neurons: List[Tuple[str, int]]
    pass2_neurons: List[Tuple[str, int]]
    hypotheses: Dict[Tuple[str, int], BipolarVerbHypothesis] = field(default_factory=dict)
    clues_injected: Dict[Tuple[str, int], int] = field(default_factory=dict)


# =============================================================================
# Contrastive discovery, development tracking, coalitions, export
# =============================================================================
@dataclass
class ContrastiveNeuronMatch:
    """
    A neuron ranked by Welch's t between per-prompt mean activations on two prompt sets
    (prompts, not tokens, are the independent units).
    """
    layer_name: str
    neuron_idx: int
    t_statistic: float
    abs_t_statistic: float
    degrees_of_freedom: float
    p_value: float
    bonferroni_p_value: float
    pos_mean: float
    neg_mean: float
    pos_std: float
    neg_std: float
    n_pos_prompts: int
    n_neg_prompts: int
    bipolar_sign_flip: bool        # per-prompt means have opposite signs
    recommended_pole: PoleType
    cohens_d: float = 0.0          # (mean_pos - mean_neg) / pooled per-prompt std


@dataclass
class ContrastiveDiscoveryReport:
    layers_scanned: List[str]
    pos_prompts_count: int
    neg_prompts_count: int
    pos_tokens_count: int
    neg_tokens_count: int
    tests_performed: int
    top_matches: List[ContrastiveNeuronMatch] = field(default_factory=list)


@dataclass
class NeuronCheckpointSnapshot:
    """One neuron at one training checkpoint ('Watching Neurons Develop')."""
    checkpoint_label: str
    mean: float
    std: float
    max_val: float
    min_val: float
    positive_firing_rate: float
    negative_firing_rate: float
    outgoing_weight_norm: float
    weight_cosine_to_final: float
    band_tokens: Dict[str, List[str]] = field(default_factory=dict)   # "positive:0.75-1.00" -> tokens
    double_negative_positive_rate: float = 0.0


@dataclass
class NeuronEvolutionReport:
    layer_name: str
    neuron_idx: int
    snapshots: List[NeuronCheckpointSnapshot] = field(default_factory=list)
    emerged_tokens_by_band: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SynergisticAblationResult:
    """Minimal coalition whose joint ablation overcomes backup circuits."""
    layer_name: str
    method: str
    objective: str
    selected_neurons: List[int]
    stepwise_scores: List[float]
    joint_score: float
    sum_of_individual_scores: float
    synergy_ratio: float


@dataclass
class CompiledSteeringVector:
    """Residual-stream vector v = sum_i delta_i W_down[:, i] for purely additive, unconditional steering."""
    layer_name: str
    d_model: int
    source_neurons: List[int]
    source_deltas: List[float]
    vector_norm: float
    vector_values: List[float]


@dataclass
class ProductionPresetBundle:
    preset_name: str
    model_name: str
    created_at_utc: str
    description: str
    interventions: List[Dict[str, Any]]
    compiled_vectors: List[CompiledSteeringVector] = field(default_factory=list)
