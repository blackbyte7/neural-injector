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


@dataclass
class ActivationSample:
    """Stores a single top-activating or top-inhibiting context example for a neuron pole."""
    activation_value: float
    sample_id: int
    position_idx: Optional[int] = None  # Token index (for sequences) or spatial index (for CNNs)
    token_str: Optional[str] = None  # Decoded focal token or patch label
    context_str: Optional[str] = None  # Surrounding text window or sample metadata


@dataclass
class NeuronPoleProfile:
    """Detailed statistical and semantic profile for a single neuron across both poles."""
    layer_name: str
    neuron_idx: int
    mean: float
    std: float
    max_val: float
    min_val: float
    positive_firing_rate: float
    negative_firing_rate: float
    pole_selectivity_score: float  # Measures how sharply peaked/clean the neuron's top activations are (kurtosis/tail)
    top_positive_samples: List[ActivationSample] = field(default_factory=list)
    top_negative_samples: List[ActivationSample] = field(default_factory=list)
    top_promoted_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_suppressed_tokens: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def is_clean_candidate(self) -> bool:
        """Heuristic following Cammarata (2026): high tail selectivity indicates a focused, clean pole."""
        return self.pole_selectivity_score >= 2.5


@dataclass
class InterventionSpec:
    """Specification for a surgical intervention on one or more neurons in a layer."""
    layer_name: str
    neuron_indices: List[int]
    mode: InterventionMode = InterventionMode.ZERO_ABLATE
    pole: PoleType = PoleType.BOTH
    value: float = 0.0  # Used as clamp constant, scale factor, shift delta, or weak threshold
    token_positions: Optional[List[int]] = None  # If None, applies across all sequence/spatial positions
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
    Provides filtering, pruning, and export utilities for downstream LLM agent explanation.
    """
    src_layer: str
    dst_layer: str
    src_neuron: int
    src_pole: PoleType
    edges: List[CircuitEdge] = field(default_factory=list)

    def top_excitatory(self, k: int = 5) -> List[CircuitEdge]:
        """Returns downstream neurons most strongly excited (pushed positive) by the source pole."""
        pos = [e for e in self.edges if e.mean_causal_delta > 0]
        return sorted(pos, key=lambda e: e.mean_causal_delta, reverse=True)[:k]

    def top_inhibitory(self, k: int = 5) -> List[CircuitEdge]:
        """Returns downstream neurons most strongly inhibited or driven negative by the source pole."""
        neg = [e for e in self.edges if e.mean_causal_delta < 0]
        return sorted(neg, key=lambda e: e.mean_causal_delta, reverse=False)[:k]

    def to_agent_payload(self) -> Dict[str, Any]:
        """Serializes the circuit into a structured dictionary ready for an LLM Explainer Agent."""
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


class NeuronTaxonomy(str, Enum):
    """
    Classification of a neuron's cleanliness following Cammarata (2026):
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
class NeuronScanSummary:
    """Compact layer-wide audit metrics for a single neuron used in rankings and heatmaps."""
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


@dataclass
class LayerCleanlinessReport:
    """
    Aggregated layer-wide cleanliness audit report mirroring Cammarata (2026).
    Provides exact taxonomy percentages and ranked neuron summaries.
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


@dataclass
class CompiledSteeringVector:
    """
    Represents a pre-compiled residual-stream steering vector v_steer in R^{d_model}
    synthesized from one or more neuron-basis interventions in `layer_name`.
    """
    layer_name: str
    d_model: int
    source_neurons: List[int]
    source_deltas: List[float]
    vector_norm: float
    vector_values: List[float]  # Serialized 1D vector of length d_model


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
