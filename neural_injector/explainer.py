from __future__ import annotations
import json
import os
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple
import torch
import torch.nn as nn

from .discovery import LayerDiscovery
from .types import (
    BipolarVerbHypothesis,
    CircuitGraph,
    NeuronPoleProfile,
    NeuronTaxonomy,
    PoleType,
    VerificationExperimentSpec,
    VerificationReport,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector


class LLMClientProtocol(Protocol):
    """Protocol for any LLM backend used by NeuronExplainerAgent."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """
    Zero-dependency OpenAI-compatible JSON completion client using stdlib `urllib.request`.
    Works with OpenAI, Ollama (http://localhost:11434/v1), vLLM, OpenRouter, and Gemini/Anthropic proxies.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

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
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)


class HeuristicSynthesisClient:
    """
    Built-in offline analytical synthesizer that inspects the structured NeuronPoleProfile
    and CircuitGraph JSON payload and formulates a BipolarVerbHypothesis + verification plan
    without requiring an external API key.
    """

    def complete_json(self, _system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        evidence = json.loads(user_prompt)
        prof = evidence.get("neuron_profile", {})
        circuit = evidence.get("downstream_circuit", {})

        pos_samples = prof.get("top_positive_samples", [])
        neg_samples = prof.get("top_negative_samples", [])
        pos_tokens = list(dict.fromkeys(s["token"] for s in pos_samples if s.get("token")))
        neg_tokens = list(dict.fromkeys(s["token"] for s in neg_samples if s.get("token")))

        promoted = [t[0] for t in prof.get("top_promoted_tokens", [])[:4]]
        suppressed = [t[0] for t in prof.get("top_suppressed_tokens", [])[:4]]

        excited_dst = [e["dst_neuron"] for e in circuit.get("top_excited_downstream", [])[:3]]
        inhibited_dst = [e["dst_neuron"] for e in circuit.get("top_inhibited_downstream", [])[:3]]

        selectivity = float(prof.get("pole_selectivity_score", 0.0))
        if selectivity >= 2.2 and (pos_tokens or neg_tokens):
            taxonomy = NeuronTaxonomy.CLEAN_MONOSEMANTIC.value
            confidence = min(0.95, 0.70 + 0.05 * selectivity)
        elif selectivity >= 1.2:
            taxonomy = NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value
            confidence = 0.72
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

        offshoots = []
        if len(pos_tokens) > 2:
            offshoots.append(
                f"Recursive generalization across co-occurring context tokens {pos_tokens[1:]} sharing downstream readers {excited_dst}"
            )
        if circuit.get("top_excited_downstream"):
            roles = [
                f"N{e['dst_neuron']} ({e['role']})"
                for e in circuit["top_excited_downstream"][:2]
            ]
            offshoots.append(f"Downstream specialization into Layer+1 circuit nodes: {', '.join(roles)}")

        return {
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


class NeuronExplainerAgent:
    """
    Automated Mechanistic Interpretability Agent implementing the methodology from
    'Neural Networks Are Already Clean' (Cammarata / Goodfire, 2026).

    1. Packages Bipolar Neuron Profiles (+Pole / -Pole contexts + Direct Logit Attribution)
       alongside Downstream Circuit Ecology (Excited, Inhibited, and Gating edges).
    2. Prompts an LLM Agent to synthesize a unified 'Verb' hypothesis and recursive offshoots.
    3. Automatically runs closed-loop surgical interventions (Zero-Ablation & Pole Clamping)
       on the live PyTorch model to empirically verify or falsify the agent's hypothesis.
    """

    SYSTEM_PROMPT = """You are a senior mechanistic interpretability researcher auditing single neurons under the 'Neural Networks Are Already Clean' framework.
Your task is to analyze the provided JSON evidence for a single neuron (including its Positive Pole activating contexts,
 Negative Pole activating contexts, Direct Vocabulary Logit Attribution, and Downstream Layer Circuit Connections) and 
 return a strict JSON object with the following schema:
{
  "positive_pole_motif": "Concise description of what excites the positive pole (>0)",
  "negative_pole_motif": "Concise description of what drives the negative pole (<0)",
  "unified_verb_hypothesis": "Single sentence explaining how this neuron acts as a differential 'verb' or router separating the Positive Pole from the Negative Pole and routing signals to downstream neurons.",
  "recursive_offshoots": ["List of related secondary semantic offshoots or downstream specializations"],
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
    def build_evidence_payload(
        profile: NeuronPoleProfile,
        circuit: Optional[CircuitGraph] = None,
    ) -> Dict[str, Any]:
        """Serializes a NeuronPoleProfile and optional CircuitGraph into a compact agent JSON payload."""
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
                "top_promoted_tokens": profile.top_promoted_tokens,
                "top_suppressed_tokens": profile.top_suppressed_tokens,
            }
        }
        if circuit is not None:
            payload["downstream_circuit"] = circuit.to_agent_payload()
        return payload

    def generate_hypothesis(
        self,
        profile: NeuronPoleProfile,
        circuit: Optional[CircuitGraph] = None,
    ) -> BipolarVerbHypothesis:
        """Calls the LLM agent to synthesize a BipolarVerbHypothesis and verification plan."""
        evidence = self.build_evidence_payload(profile, circuit)
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
        """
        model = self.injector.model
        model.eval()
        device = next(model.parameters()).device
        layer_name = hypothesis.layer_name
        neuron_idx = hypothesis.neuron_idx
        exp = hypothesis.proposed_experiment

        pos_ids = [tok_to_id[t] for t in exp.positive_pole_test_tokens if t in tok_to_id] or [0]
        neg_ids = [tok_to_id[t] for t in exp.negative_pole_test_tokens if t in tok_to_id] or [0]
        prom_ids = [tok_to_id[t] for t in exp.expected_promoted_tokens if t in tok_to_id]
        supp_ids = [tok_to_id[t] for t in exp.expected_suppressed_tokens if t in tok_to_id]

        pos_input = torch.tensor([pos_ids], dtype=torch.long, device=device)
        neg_input = torch.tensor([neg_ids], dtype=torch.long, device=device)

        # Helper to capture single neuron activation on an input tensor
        mod, hook_target, _ = LayerDiscovery.resolve_module(model, layer_name)
        captured_acts: List[torch.Tensor] = []

        def _cap_pre(_m: nn.Module, inp: Tuple[Any, ...]):
            if inp and isinstance(inp[0], torch.Tensor):
                captured_acts.append(inp[0].detach())

        def _cap_fwd(_m: nn.Module, _inp: Tuple[Any, ...], out: Any):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured_acts.append(t.detach())

        h = (
            mod.register_forward_pre_hook(_cap_pre)
            if hook_target == "input"
            else mod.register_forward_hook(_cap_fwd)
        )
        try:
            self.injector.clear_interventions()
            clean_pos_out = model(pos_input)
            clean_pos_logits = (
                clean_pos_out.logits if hasattr(clean_pos_out, "logits") else clean_pos_out
            )
            pos_act_mean = float(captured_acts[-1][..., neuron_idx].mean().item())

            model(neg_input)
            neg_act_mean = float(captured_acts[-1][..., neuron_idx].mean().item())
        finally:
            h.remove()

        pole_separation_passed = (pos_act_mean > 0.0) and (neg_act_mean < 0.0)

        # Test 2: Zero-Ablation on Positive Input (Set neuron = 0.0)
        self.injector.clear_interventions()
        self.injector.zero_ablate(layer_name, neuron_indices=neuron_idx, pole=PoleType.BOTH)
        with self.injector.active_injection():
            zero_pos_out = model(pos_input)
            zero_pos_logits = (
                zero_pos_out.logits if hasattr(zero_pos_out, "logits") else zero_pos_out
            )

        # Test 3: Pole-Flip Clamping on Positive Input (Clamp neuron to strong negative value)
        clamp_neg_val = min(neg_act_mean * 1.5, -5.0)
        self.injector.clear_interventions()
        self.injector.clamp_neurons(layer_name, neuron_indices=neuron_idx, value=clamp_neg_val)
        with self.injector.active_injection():
            flip_pos_out = model(pos_input)
            flip_pos_logits = (
                flip_pos_out.logits if hasattr(flip_pos_out, "logits") else flip_pos_out
            )
        self.injector.clear_interventions()

        # Compute logit shifts on expected promoted & suppressed tokens
        if prom_ids:
            clean_prom = clean_pos_logits[:, :, prom_ids].mean().item()
            zero_prom = zero_pos_logits[:, :, prom_ids].mean().item()
            logit_drop = float(clean_prom - zero_prom)
        else:
            logit_drop = 0.0

        if supp_ids:
            clean_supp = clean_pos_logits[:, :, supp_ids].mean().item()
            flip_supp = flip_pos_logits[:, :, supp_ids].mean().item()
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
    ) -> BipolarVerbHypothesis:
        """
        Full end-to-end closed-loop agentic workflow:
          1. Extracts the audited NeuronPoleProfile (+ Direct Logit Attribution).
          2. Optionally traces the downstream circuit to `dst_layer`.
          3. Synthesizes a BipolarVerbHypothesis via the LLM Agent.
          4. Runs surgical Zero-Ablation and Pole-Clamping verification experiments to validate the hypothesis.
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

        hypothesis = self.generate_hypothesis(profile, circuit)
        self.verify_hypothesis(hypothesis, tok_to_id=tok_to_id, dst_layer=dst_layer)
        return hypothesis
