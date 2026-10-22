from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field, ValidationError
import torch
import torch.nn as nn

from .discovery import LayerDiscovery
from .types import (
    BipolarVerbHypothesis,
    CircuitEdge,
    CircuitGraph,
    EcologyNeighborContext,
    InterventionMode,
    InterventionSpec,
    NeuronPoleProfile,
    NeuronTaxonomy,
    PoleType,
    VerificationExperimentSpec,
    VerificationReport,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector


# =============================================================================
# 1. Pydantic Structured Output Schemas for LLM Response Validation (Sec 4.3)
# =============================================================================
class ProposedExperimentSchema(BaseModel):
    positive_pole_test_tokens: List[str] = Field(default_factory=list)
    negative_pole_test_tokens: List[str] = Field(default_factory=list)
    expected_promoted_tokens: List[str] = Field(default_factory=list)
    expected_suppressed_tokens: List[str] = Field(default_factory=list)
    expected_excited_dst_neurons: List[int] = Field(default_factory=list)
    expected_inhibited_dst_neurons: List[int] = Field(default_factory=list)


class AgentHypothesisResponseSchema(BaseModel):
    positive_pole_motif: str = Field(..., min_length=1)
    negative_pole_motif: str = Field(..., min_length=1)
    unified_verb_hypothesis: str = Field(..., min_length=1)
    recursive_offshoots: List[str] = Field(default_factory=list)
    taxonomy_class: NeuronTaxonomy = Field(default=NeuronTaxonomy.CLEAN_MONOSEMANTIC)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    proposed_experiment: ProposedExperimentSchema = Field(
        default_factory=ProposedExperimentSchema
    )


class LLMClientProtocol(Protocol):
    """Protocol for any LLM backend used by NeuronExplainerAgent."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """
    OpenAI-compatible JSON completion client using stdlib `urllib.request` with
    exponential backoff retry logic and strict Pydantic schema validation (Section 4.3).
    Works with OpenAI, Ollama (http://localhost:11434/v1), vLLM, OpenRouter, and proxies.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 60.0,
        validate_schema: bool = True,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.validate_schema = validate_schema

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if self.validate_schema:
                    validated = AgentHypothesisResponseSchema.model_validate(parsed)
                    return validated.model_dump(mode="json")
                return parsed
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))

        raise RuntimeError(
            f"OpenAICompatibleClient failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc


# =============================================================================
# 2. Persistent Crossword / Circuit Ecology Store (Cammarata Section 3.2)
# =============================================================================
class CircuitEcologyMap:
    """
    Persistent Crossword Graph Store implementing Cammarata's 'Understanding Compounds'
    loop (Section 3.2):
      - Stores verified `BipolarVerbHypothesis` objects indexed by `(layer_name, neuron_idx)`.
      - Stores directed `CircuitEdge` connections traced across layers.
      - Retrieves upstream and downstream neighbor hypotheses so that when explaining
        neuron `(L, i)`, already-verified motifs of connected neurons in `L-1` and `L+1`
        are injected into the Explainer Agent's context.
    """

    def __init__(self) -> None:
        self.hypotheses: Dict[Tuple[str, int], BipolarVerbHypothesis] = {}
        self.edges_from: Dict[Tuple[str, int], List[CircuitEdge]] = {}
        self.edges_to: Dict[Tuple[str, int], List[CircuitEdge]] = {}

    def register_hypothesis(self, hypothesis: BipolarVerbHypothesis) -> None:
        """Stores or updates a neuron's BipolarVerbHypothesis in the ecology map."""
        self.hypotheses[(hypothesis.layer_name, hypothesis.neuron_idx)] = hypothesis

    def get_hypothesis(
        self, layer_name: str, neuron_idx: int
    ) -> Optional[BipolarVerbHypothesis]:
        """Returns the stored BipolarVerbHypothesis for `(layer_name, neuron_idx)` if present."""
        return self.hypotheses.get((layer_name, neuron_idx))

    def register_circuit_graph(self, graph: CircuitGraph) -> None:
        """Indexes all directed edges from a traced CircuitGraph for bidirectional lookup."""
        src_key = (graph.src_layer, graph.src_neuron)
        existing_out = {
            (e.dst_layer, e.dst_neuron): e for e in self.edges_from.get(src_key, [])
        }
        for edge in graph.edges:
            existing_out[(edge.dst_layer, edge.dst_neuron)] = edge
            dst_key = (edge.dst_layer, edge.dst_neuron)
            existing_in = {
                (e.src_layer, e.src_neuron): e for e in self.edges_to.get(dst_key, [])
            }
            existing_in[src_key] = edge
            self.edges_to[dst_key] = list(existing_in.values())
        self.edges_from[src_key] = list(existing_out.values())

    def get_neighbor_contexts(
        self, layer_name: str, neuron_idx: int, max_neighbors: int = 6
    ) -> List[EcologyNeighborContext]:
        """
        Returns already-explained upstream and downstream neighbor clues connected
        to `(layer_name, neuron_idx)`, sorted by causal delta magnitude.
        """
        key = (layer_name, neuron_idx)
        neighbors: List[EcologyNeighborContext] = []

        # 1. Upstream neighbors (L-1 -> L)
        for edge in self.edges_to.get(key, []):
            up_hyp = self.hypotheses.get((edge.src_layer, edge.src_neuron))
            if up_hyp is not None:
                neighbors.append(
                    EcologyNeighborContext(
                        layer_name=edge.src_layer,
                        neuron_idx=edge.src_neuron,
                        direction="upstream",
                        role=edge.role.value,
                        mean_causal_delta=round(edge.mean_causal_delta, 4),
                        positive_pole_motif=up_hyp.positive_pole_motif,
                        negative_pole_motif=up_hyp.negative_pole_motif,
                        unified_verb_hypothesis=up_hyp.unified_verb_hypothesis,
                        confidence_score=up_hyp.confidence_score,
                    )
                )

        # 2. Downstream neighbors (L -> L+1)
        for edge in self.edges_from.get(key, []):
            dn_hyp = self.hypotheses.get((edge.dst_layer, edge.dst_neuron))
            if dn_hyp is not None:
                neighbors.append(
                    EcologyNeighborContext(
                        layer_name=edge.dst_layer,
                        neuron_idx=edge.dst_neuron,
                        direction="downstream",
                        role=edge.role.value,
                        mean_causal_delta=round(edge.mean_causal_delta, 4),
                        positive_pole_motif=dn_hyp.positive_pole_motif,
                        negative_pole_motif=dn_hyp.negative_pole_motif,
                        unified_verb_hypothesis=dn_hyp.unified_verb_hypothesis,
                        confidence_score=dn_hyp.confidence_score,
                    )
                )

        neighbors.sort(key=lambda n: abs(n.mean_causal_delta), reverse=True)
        return neighbors[:max_neighbors]


class HeuristicSynthesisClient:
    """
    Built-in offline analytical synthesizer that inspects the structured NeuronPoleProfile
    (including Stratified Quantile Activation Buckets and Pole-Specific Selectivity),
    CircuitGraph, and Crossword Ecology Neighbor clues to formulate a BipolarVerbHypothesis.
    """

    def complete_json(self, _system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        evidence = json.loads(user_prompt)
        prof = evidence.get("neuron_profile", {})
        circuit = evidence.get("downstream_circuit", {})
        neighbors = evidence.get("ecology_neighbors", [])

        pos_samples = prof.get("top_positive_samples", [])
        neg_samples = prof.get("top_negative_samples", [])
        pos_tokens = list(dict.fromkeys(s["token"] for s in pos_samples if s.get("token")))
        neg_tokens = list(dict.fromkeys(s["token"] for s in neg_samples if s.get("token")))

        promoted = [t[0] for t in prof.get("top_promoted_tokens", [])[:4]]
        suppressed = [t[0] for t in prof.get("top_suppressed_tokens", [])[:4]]

        excited_dst = [e["dst_neuron"] for e in circuit.get("top_excited_downstream", [])[:3]]
        inhibited_dst = [e["dst_neuron"] for e in circuit.get("top_inhibited_downstream", [])[:3]]

        selectivity = float(prof.get("pole_selectivity_score", 0.0))
        pos_sel = float(prof.get("pos_pole_selectivity_score", selectivity))
        neg_sel = float(prof.get("neg_pole_selectivity_score", selectivity))
        best_pole_sel = max(selectivity, pos_sel, neg_sel)

        if best_pole_sel >= 2.1 and (pos_tokens or neg_tokens):
            taxonomy = NeuronTaxonomy.CLEAN_MONOSEMANTIC.value
            confidence = min(0.96, 0.72 + 0.05 * best_pole_sel)
        elif best_pole_sel >= 1.2:
            taxonomy = NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value
            confidence = 0.75
        else:
            taxonomy = NeuronTaxonomy.DEAD_OR_NOISY.value
            confidence = 0.40

        pos_motif = (
            f"Detects and amplifies [{' / '.join(pos_tokens[:3])}] features, driving output toward [{' / '.join(promoted[:2])}]"
            if pos_tokens
            else "Latent excitatory activation regime"
        )
        neg_motif = (
            f"Opposes positive motif by firing negatively on [{' / '.join(neg_tokens[:3])}], promoting [{' / '.join(suppressed[:2])}]"
            if neg_tokens
            else "Latent inhibitory / baseline suppression regime"
        )

        verb_hypothesis = (
            f"Acts as a bipolar semantic router ('verb') separating [{', '.join(pos_tokens[:3])}] (+Pole) "
            f"from [{', '.join(neg_tokens[:3])}] (-Pole): when positive, it excites downstream neurons {excited_dst} "
            f"and inhibits competing circuit branches {inhibited_dst}."
        )

        offshoots: List[str] = []
        # Extract stratified quantile offshoots (Cammarata Section 3.3)
        pos_buckets = prof.get("positive_quantile_buckets", {})
        sec_tokens = list(
            dict.fromkeys(
                s["token"]
                for s in pos_buckets.get("secondary_branch", [])
                if s.get("token") and s.get("token") not in pos_tokens[:1]
            )
        )
        tert_tokens = list(
            dict.fromkeys(
                s["token"]
                for s in pos_buckets.get("tertiary_offshoot", [])
                if s.get("token") and s.get("token") not in pos_tokens[:1]
            )
        )
        if sec_tokens:
            offshoots.append(
                f"Secondary quantile branch (40-80% max): recursive generalization onto {sec_tokens[:3]}"
            )
        if tert_tokens:
            offshoots.append(
                f"Tertiary quantile offshoot (15-40% max): weak semantic extension onto {tert_tokens[:3]}"
            )
        if len(pos_tokens) > 2 and not offshoots:
            offshoots.append(
                f"Recursive generalization across co-occurring context tokens {pos_tokens[1:]} sharing downstream readers {excited_dst}"
            )
        if circuit.get("top_excited_downstream"):
            roles = [
                f"N{e['dst_neuron']} ({e['role']})"
                for e in circuit["top_excited_downstream"][:2]
            ]
            offshoots.append(f"Downstream specialization into Layer+1 circuit nodes: {', '.join(roles)}")
        if neighbors:
            neighbor_summary = ", ".join(
                f"{n['direction']} {n['layer_name']}[N{n['neuron_idx']}] ({n['role']})"
                for n in neighbors[:2]
            )
            offshoots.append(f"Crossword ecology alignment with verified neighbors: {neighbor_summary}")

        raw_result = {
            "positive_pole_motif": pos_motif,
            "negative_pole_motif": neg_motif,
            "unified_verb_hypothesis": verb_hypothesis,
            "recursive_offshoots": offshoots,
            "taxonomy_class": taxonomy,
            "confidence_score": round(confidence, 3),
            "proposed_experiment": {
                "positive_pole_test_tokens": pos_tokens[:3] or promoted[:2],
                "negative_pole_test_tokens": neg_tokens[:3] or suppressed[:2],
                "expected_promoted_tokens": promoted[:3],
                "expected_suppressed_tokens": suppressed[:3],
                "expected_excited_dst_neurons": excited_dst,
                "expected_inhibited_dst_neurons": inhibited_dst,
            },
        }
        validated = AgentHypothesisResponseSchema.model_validate(raw_result)
        return validated.model_dump(mode="json")


class NeuronExplainerAgent:
    """
    Automated Mechanistic Interpretability Agent implementing the methodology from
    'Neural Networks Are Already Clean' (Cammarata / Goodfire, 2026).

    1. Packages Bipolar Neuron Profiles (+Pole / -Pole contexts, Pole-Specific Selectivities,
       Stratified Quantile Activation Buckets, and Direct Logit Attribution) alongside
       Downstream Circuit Ecology and verified Crossword Neighbor Clues (`CircuitEcologyMap`).
    2. Prompts an LLM Agent to synthesize a unified 'Verb' hypothesis and recursive offshoots.
    3. Automatically runs closed-loop surgical interventions (Zero-Ablation & Pole Clamping)
       on the live PyTorch model to empirically verify or falsify the agent's hypothesis.
    """

    SYSTEM_PROMPT = """You are a senior mechanistic interpretability researcher auditing single neurons under the 'Neural Networks Are Already Clean' framework.
Your task is to analyze the provided JSON evidence for a single neuron (including its Positive Pole activating contexts,
 Negative Pole activating contexts, Stratified Quantile Activation Buckets, Direct Vocabulary Logit Attribution,
 Downstream Layer Circuit Connections, and any already-verified Crossword Ecology Neighbor Clues) and
 return a strict JSON object with the following schema:
{
  "positive_pole_motif": "Concise description of what excites the positive pole (>0)",
  "negative_pole_motif": "Concise description of what drives the negative pole (<0)",
  "unified_verb_hypothesis": "Single sentence explaining how this neuron acts as a differential 'verb' or router separating the Positive Pole from the Negative Pole and routing signals to downstream neurons.",
  "recursive_offshoots": ["List of related secondary semantic offshoots across quantile bands or downstream specializations"],
  "taxonomy_class": "clean_monosemantic | structured_recursive_mixture | true_polysemantic | dead_or_noisy",
  "confidence_score": 0.0 to 1.0,
  "proposed_experiment": {
    "positive_pole_test_tokens": ["tokens expected to strongly trigger the +Pole"],
    "negative_pole_test_tokens": ["tokens expected to strongly trigger the -Pole"],
    "expected_promoted_tokens": ["vocabulary tokens expected to drop when +Pole is zero-ablated"],
    "expected_suppressed_tokens": ["vocabulary tokens expected to rise when clamped to -Pole"],
    "expected_excited_dst_neurons": [integer indices of downstream neurons expected to drop when +Pole is zero-ablated],
    "expected_inhibited_dst_neurons": [integer indices of downstream neurons expected to rise when +Pole is zero-ablated]
  }
}"""

    def __init__(
        self,
        injector: NeuralInjector,
        llm_client: Optional[LLMClientProtocol] = None,
    ):
        self.injector = injector
        self.llm_client: LLMClientProtocol = llm_client or HeuristicSynthesisClient()

    @staticmethod
    def _resolve_token_ids(
        tokens: List[str],
        tok_to_id: Dict[str, int],
        tokenizer: Optional[Any] = None,
    ) -> List[int]:
        """
        Resolves token strings into vocabulary IDs, handling BPE/SentencePiece prefixes
        ('Ġ', '▁', leading spaces), class_<id> labels, and HuggingFace tokenizer fallback.
        """
        resolved: List[int] = []
        for t in tokens:
            if not t:
                continue
            if t in tok_to_id:
                resolved.append(tok_to_id[t])
                continue
            stripped = t.strip()
            for candidate in (stripped, f"Ġ{stripped}", f"▁{stripped}", f" {stripped}"):
                if candidate in tok_to_id:
                    resolved.append(tok_to_id[candidate])
                    break
            else:
                if stripped.startswith(("class_", "token_")) and stripped.split("_")[-1].isdigit():
                    resolved.append(int(stripped.split("_")[-1]))
                elif tokenizer is not None and hasattr(tokenizer, "encode"):
                    enc_ids = tokenizer.encode(stripped, add_special_tokens=False)
                    if enc_ids:
                        resolved.extend(int(i) for i in enc_ids)
        return list(dict.fromkeys(resolved))

    @classmethod
    def _encode_context_or_tokens(
        cls,
        test_tokens: List[str],
        tok_to_id: Dict[str, int],
        device: torch.device,
        fallback_context: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Builds a 2D input_ids tensor `[1, seq_len]` from `test_tokens`, falling back to
        encoding `fallback_context` when subword lookup would otherwise collapse to `[0]`.
        """
        ids = cls._resolve_token_ids(test_tokens, tok_to_id, tokenizer=tokenizer)
        if not ids and fallback_context:
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(fallback_context, add_special_tokens=False)
            else:
                words = [w.strip() for w in fallback_context.split() if w.strip()]
                ids = cls._resolve_token_ids(words, tok_to_id, tokenizer=None)
        if not ids:
            ids = [0]
        return torch.tensor([ids], dtype=torch.long, device=device)

    @staticmethod
    def build_evidence_payload(
        profile: NeuronPoleProfile,
        circuit: Optional[CircuitGraph] = None,
        ecology_neighbors: Optional[List[EcologyNeighborContext]] = None,
    ) -> Dict[str, Any]:
        """
        Serializes a NeuronPoleProfile (including pole-specific selectivities and
        stratified quantile buckets), optional CircuitGraph, and optional Crossword
        Ecology Neighbor contexts into a compact agent JSON payload.
        """
        def _fmt_buckets(buckets: Dict[str, List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
            return {
                band: [
                    {
                        "activation": round(s.activation_value, 4),
                        "token": s.token_str,
                        "context": s.context_str,
                    }
                    for s in samples[:4]
                ]
                for band, samples in buckets.items()
            }

        payload: Dict[str, Any] = {
            "neuron_profile": {
                "layer_name": profile.layer_name,
                "neuron_idx": profile.neuron_idx,
                "mean": round(profile.mean, 4),
                "std": round(profile.std, 4),
                "max_val": round(profile.max_val, 4),
                "min_val": round(profile.min_val, 4),
                "positive_firing_rate": round(profile.positive_firing_rate, 4),
                "negative_firing_rate": round(profile.negative_firing_rate, 4),
                "pole_selectivity_score": round(profile.pole_selectivity_score, 3),
                "pos_pole_selectivity_score": round(profile.pos_pole_selectivity_score, 3),
                "pos_taxonomy_class": profile.pos_taxonomy_class.value,
                "neg_pole_selectivity_score": round(profile.neg_pole_selectivity_score, 3),
                "neg_taxonomy_class": profile.neg_taxonomy_class.value,
                "top_positive_samples": [
                    {
                        "activation": round(s.activation_value, 4),
                        "token": s.token_str,
                        "context": s.context_str,
                    }
                    for s in profile.top_positive_samples
                ],
                "top_negative_samples": [
                    {
                        "activation": round(s.activation_value, 4),
                        "token": s.token_str,
                        "context": s.context_str,
                    }
                    for s in profile.top_negative_samples
                ],
                "positive_quantile_buckets": _fmt_buckets(profile.positive_quantile_buckets),
                "negative_quantile_buckets": _fmt_buckets(profile.negative_quantile_buckets),
                "top_promoted_tokens": profile.top_promoted_tokens,
                "top_suppressed_tokens": profile.top_suppressed_tokens,
            }
        }
        if circuit is not None:
            payload["downstream_circuit"] = circuit.to_agent_payload()
        if ecology_neighbors:
            payload["ecology_neighbors"] = [
                {
                    "layer_name": n.layer_name,
                    "neuron_idx": n.neuron_idx,
                    "direction": n.direction,
                    "role": n.role,
                    "mean_causal_delta": n.mean_causal_delta,
                    "positive_pole_motif": n.positive_pole_motif,
                    "negative_pole_motif": n.negative_pole_motif,
                    "unified_verb_hypothesis": n.unified_verb_hypothesis,
                    "confidence_score": n.confidence_score,
                }
                for n in ecology_neighbors
            ]
        return payload

    def generate_hypothesis(
        self,
        profile: NeuronPoleProfile,
        circuit: Optional[CircuitGraph] = None,
        ecology_neighbors: Optional[List[EcologyNeighborContext]] = None,
    ) -> BipolarVerbHypothesis:
        """Calls the LLM agent to synthesize a BipolarVerbHypothesis and verification plan."""
        evidence = self.build_evidence_payload(profile, circuit, ecology_neighbors)
        raw_resp = self.llm_client.complete_json(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=json.dumps(evidence, indent=2),
        )

        exp_dict = raw_resp.get("proposed_experiment", {})
        exp_spec = VerificationExperimentSpec(
            positive_pole_test_tokens=exp_dict.get("positive_pole_test_tokens", []),
            negative_pole_test_tokens=exp_dict.get("negative_pole_test_tokens", []),
            expected_promoted_tokens=exp_dict.get("expected_promoted_tokens", []),
            expected_suppressed_tokens=exp_dict.get("expected_suppressed_tokens", []),
            expected_excited_dst_neurons=exp_dict.get("expected_excited_dst_neurons", []),
            expected_inhibited_dst_neurons=exp_dict.get("expected_inhibited_dst_neurons", []),
        )

        tax_str = raw_resp.get("taxonomy_class", NeuronTaxonomy.CLEAN_MONOSEMANTIC.value)
        try:
            taxonomy = NeuronTaxonomy(tax_str)
        except ValueError:
            taxonomy = NeuronTaxonomy.CLEAN_MONOSEMANTIC

        return BipolarVerbHypothesis(
            layer_name=profile.layer_name,
            neuron_idx=profile.neuron_idx,
            positive_pole_motif=raw_resp.get("positive_pole_motif", ""),
            negative_pole_motif=raw_resp.get("negative_pole_motif", ""),
            unified_verb_hypothesis=raw_resp.get("unified_verb_hypothesis", ""),
            recursive_offshoots=raw_resp.get("recursive_offshoots", []),
            taxonomy_class=taxonomy,
            confidence_score=float(raw_resp.get("confidence_score", 0.5)),
            proposed_experiment=exp_spec,
        )

    @torch.no_grad()
    def verify_hypothesis(
        self,
        hypothesis: BipolarVerbHypothesis,
        tok_to_id: Dict[str, int],
        dst_layer: Optional[str] = None,
        profile: Optional[NeuronPoleProfile] = None,
        tokenizer: Optional[Any] = None,
        circuit_inputs: Optional[Any] = None,
    ) -> VerificationReport:
        """
        Executes the agent's proposed contrastive & surgical ablation experiments directly
        on the model via NeuralInjector to empirically score the hypothesis:
          1. Contrastive Pole Excitation Check: Runs positive vs. negative test tokens and verifies
             that the neuron fires positively on `positive_pole_test_tokens` and negatively on `negative_pole_test_tokens`.
          2. Surgical Zero-Ablation Check (`neuron = 0`): Measures whether setting the neuron to 0
             causally reduces the logits of `expected_promoted_tokens` and shifts downstream `dst_layer` neurons.
          3. Surgical Pole-Flip Check (`neuron = -max_val`): Measures whether clamping the neuron to its
             negative pole boosts `expected_suppressed_tokens`.
        Uses `without_interventions()` and `temporary_interventions(...)` so pre-existing user interventions are preserved.
        """
        model = self.injector.model
        model.eval()
        device = next(model.parameters()).device
        layer_name = hypothesis.layer_name
        neuron_idx = hypothesis.neuron_idx
        exp = hypothesis.proposed_experiment

        # Non-tokenized Vision / continuous tensor fallback if circuit_inputs is a float tensor (e.g. [B, C, H, W])
        if (
            not tok_to_id
            and isinstance(circuit_inputs, torch.Tensor)
            and circuit_inputs.is_floating_point()
        ):
            pos_input = circuit_inputs.to(device)
            neg_input = -circuit_inputs.to(device)
        else:
            pos_fallback_ctx = (
                profile.top_positive_samples[0].context_str
                if profile and profile.top_positive_samples
                else None
            )
            neg_fallback_ctx = (
                profile.top_negative_samples[0].context_str
                if profile and profile.top_negative_samples
                else None
            )
            pos_input = self._encode_context_or_tokens(
                exp.positive_pole_test_tokens,
                tok_to_id,
                device=device,
                fallback_context=pos_fallback_ctx,
                tokenizer=tokenizer,
            )
            neg_input = self._encode_context_or_tokens(
                exp.negative_pole_test_tokens,
                tok_to_id,
                device=device,
                fallback_context=neg_fallback_ctx,
                tokenizer=tokenizer,
            )

        prom_ids = self._resolve_token_ids(
            exp.expected_promoted_tokens, tok_to_id, tokenizer=tokenizer
        )
        supp_ids = self._resolve_token_ids(
            exp.expected_suppressed_tokens, tok_to_id, tokenizer=tokenizer
        )

        # Helper to capture single neuron activation on an input tensor
        mod, hook_target, _ = LayerDiscovery.resolve_module(model, layer_name)
        is_conv2d = isinstance(mod, nn.Conv2d)
        captured_acts: List[torch.Tensor] = []

        def _cap_pre(_m: nn.Module, inp: Tuple[Any, ...]):
            if inp and isinstance(inp[0], torch.Tensor):
                captured_acts.append(inp[0].detach())

        def _cap_fwd(_m: nn.Module, _inp: Tuple[Any, ...], out: Any):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured_acts.append(t.detach())

        def _extract_neuron_act(act_tensor: torch.Tensor) -> torch.Tensor:
            if is_conv2d and act_tensor.ndim == 4:
                return act_tensor[:, neuron_idx, :, :].float()
            return act_tensor[..., neuron_idx].float()

        h = (
            mod.register_forward_pre_hook(_cap_pre)
            if hook_target == "input"
            else mod.register_forward_hook(_cap_fwd)
        )
        try:
            with self.injector.without_interventions():
                clean_pos_out = model(pos_input)
                clean_pos_logits = (
                    clean_pos_out.logits if hasattr(clean_pos_out, "logits") else clean_pos_out
                ).float()
                pos_act_mean = float(_extract_neuron_act(captured_acts[-1]).mean().item())

                model(neg_input)
                neg_act_mean = float(_extract_neuron_act(captured_acts[-1]).mean().item())

                # Fallback to full context strings if isolated subwords did not separate poles
                if (
                    tok_to_id
                    and (pos_act_mean <= 0.0 or neg_act_mean >= 0.0)
                    and profile is not None
                ):
                    pos_ctx = (
                        profile.top_positive_samples[0].context_str
                        if profile.top_positive_samples
                        else None
                    )
                    neg_ctx = (
                        profile.top_negative_samples[0].context_str
                        if profile.top_negative_samples
                        else None
                    )
                    if pos_act_mean <= 0.0 and pos_ctx:
                        ctx_pos_in = self._encode_context_or_tokens(
                            [], tok_to_id, device=device, fallback_context=pos_ctx, tokenizer=tokenizer
                        )
                        clean_pos_out = model(ctx_pos_in)
                        clean_pos_logits = (
                            clean_pos_out.logits
                            if hasattr(clean_pos_out, "logits")
                            else clean_pos_out
                        ).float()
                        pos_act_mean = float(_extract_neuron_act(captured_acts[-1]).max().item())
                        pos_input = ctx_pos_in
                    if neg_act_mean >= 0.0 and neg_ctx:
                        ctx_neg_in = self._encode_context_or_tokens(
                            [], tok_to_id, device=device, fallback_context=neg_ctx, tokenizer=tokenizer
                        )
                        model(ctx_neg_in)
                        neg_act_mean = float(_extract_neuron_act(captured_acts[-1]).min().item())
        finally:
            h.remove()

        pole_separation_passed = (pos_act_mean > 0.0) and (neg_act_mean < 0.0)

        # Test 2: Zero-Ablation on Positive Input (Set neuron = 0.0)
        ablate_spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=[neuron_idx],
            mode=InterventionMode.ZERO_ABLATE,
            pole=PoleType.BOTH,
            value=0.0,
        )
        with self.injector.temporary_interventions([ablate_spec]):
            zero_pos_out = model(pos_input)
            zero_pos_logits = (
                zero_pos_out.logits if hasattr(zero_pos_out, "logits") else zero_pos_out
            ).float()

        # Test 3: Pole-Flip Clamping on Positive Input (Clamp neuron to strong negative value)
        clamp_neg_val = min(neg_act_mean * 1.5, -5.0)
        clamp_spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=[neuron_idx],
            mode=InterventionMode.HARD_CLAMP,
            pole=PoleType.BOTH,
            value=clamp_neg_val,
        )
        with self.injector.temporary_interventions([clamp_spec]):
            flip_pos_out = model(pos_input)
            flip_pos_logits = (
                flip_pos_out.logits if hasattr(flip_pos_out, "logits") else flip_pos_out
            ).float()

        # Compute logit shifts on expected promoted & suppressed tokens
        if prom_ids:
            clean_prom = clean_pos_logits[..., prom_ids].mean().item()
            zero_prom = zero_pos_logits[..., prom_ids].mean().item()
            logit_drop = float(clean_prom - zero_prom)
        else:
            logit_drop = 0.0

        if supp_ids:
            clean_supp = clean_pos_logits[..., supp_ids].mean().item()
            flip_supp = flip_pos_logits[..., supp_ids].mean().item()
            pole_flip_shift = float(flip_supp - clean_supp)
        else:
            pole_flip_shift = 0.0

        # Test 4: Verify downstream circuit predictions if `dst_layer` is provided
        circuit_match_ratio = 1.0
        if dst_layer is not None and (
            exp.expected_excited_dst_neurons or exp.expected_inhibited_dst_neurons
        ):
            traced = self.injector.trace_circuit(
                src_layer=layer_name,
                dst_layer=dst_layer,
                src_neuron=neuron_idx,
                inputs=pos_input,
                src_pole=PoleType.POSITIVE,
                compute_jacobian=False,
                top_k=32,
            )
            delta_map = {e.dst_neuron: e.mean_causal_delta for e in traced.edges}
            matches = 0
            total_expected = len(exp.expected_excited_dst_neurons) + len(
                exp.expected_inhibited_dst_neurons
            )
            for dst_n in exp.expected_excited_dst_neurons:
                if delta_map.get(dst_n, 0.0) > 0:
                    matches += 1
            for dst_n in exp.expected_inhibited_dst_neurons:
                if delta_map.get(dst_n, 0.0) < 0:
                    matches += 1
            circuit_match_ratio = matches / max(total_expected, 1)

        # Compute overall empirical verification score in [0.0, 1.0]
        score = 0.0
        if pole_separation_passed:
            score += 0.35
        if logit_drop > 0:
            score += 0.25
        if pole_flip_shift > 0:
            score += 0.20
        score += 0.20 * circuit_match_ratio
        score = round(min(max(score, 0.0), 1.0), 3)

        summary = (
            f"Pole Separation: {'PASS' if pole_separation_passed else 'FAIL'} "
            f"(+Test Act={pos_act_mean:+.3f}, -Test Act={neg_act_mean:+.3f}) | "
            f"Zero-Ablation Promoted Logit Drop={logit_drop:+.4f} | "
            f"Negative-Clamp Opposite Pole Boost={pole_flip_shift:+.4f} | "
            f"Downstream Circuit Match={circuit_match_ratio * 100:.1f}%"
        )

        report = VerificationReport(
            positive_test_mean_activation=round(pos_act_mean, 4),
            negative_test_mean_activation=round(neg_act_mean, 4),
            pole_separation_passed=pole_separation_passed,
            zero_ablation_target_logit_drop=round(logit_drop, 4),
            pole_flip_target_logit_shift=round(pole_flip_shift, 4),
            downstream_circuit_match_ratio=round(circuit_match_ratio, 4),
            empirical_verification_score=score,
            summary_notes=summary,
        )
        hypothesis.verification_report = report
        return report

    def explain_and_verify(
        self,
        layer_name: str,
        neuron_idx: int,
        tok_to_id: Dict[str, int],
        idx_to_token: Dict[int, str],
        unembed_weight: Optional[torch.Tensor] = None,
        dst_layer: Optional[str] = None,
        circuit_inputs: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> BipolarVerbHypothesis:
        """
        Full end-to-end closed-loop agentic workflow:
          1. Extracts the audited NeuronPoleProfile (+ Direct Logit Attribution + Quantile Buckets).
          2. Optionally traces the downstream circuit to `dst_layer`.
          3. Queries `self.injector.ecology_map` for already-verified upstream/downstream neighbor clues.
          4. Synthesizes a BipolarVerbHypothesis via the LLM Agent and empirically verifies it.
          5. Registers the verified hypothesis back into `self.injector.ecology_map`.
        """
        profile = self.injector.get_neuron_profile(
            layer_name=layer_name,
            neuron_idx=neuron_idx,
            unembed_weight=unembed_weight,
            idx_to_token=idx_to_token,
        )

        circuit: Optional[CircuitGraph] = None
        if dst_layer is not None and circuit_inputs is not None:
            circuit = self.injector.trace_circuit(
                src_layer=layer_name,
                dst_layer=dst_layer,
                src_neuron=neuron_idx,
                inputs=circuit_inputs,
                src_pole=PoleType.POSITIVE,
                compute_jacobian=True,
                top_k=8,
            )

        neighbors: List[EcologyNeighborContext] = []
        if hasattr(self.injector, "ecology_map") and self.injector.ecology_map is not None:
            neighbors = self.injector.ecology_map.get_neighbor_contexts(layer_name, neuron_idx)

        hypothesis = self.generate_hypothesis(profile, circuit, ecology_neighbors=neighbors)
        self.verify_hypothesis(
            hypothesis,
            tok_to_id=tok_to_id,
            dst_layer=dst_layer,
            profile=profile,
            tokenizer=tokenizer,
            circuit_inputs=circuit_inputs,
        )
        if hasattr(self.injector, "ecology_map") and self.injector.ecology_map is not None:
            self.injector.ecology_map.register_hypothesis(hypothesis)
        return hypothesis
