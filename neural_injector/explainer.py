from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Type

from pydantic import BaseModel, Field, ValidationError
import torch
import torch.nn as nn
import torch.nn.functional as F

from .analyzer import is_punctuation_token
from .hooks import HookSet, default_forward, extract_logits, flat_neuron_view
from .types import (
    ActivationBand,
    ActivationSample,
    BipolarVerbHypothesis,
    BlindTestSuite,
    BundleBlindMaximumResult,
    CausalCheckReport,
    CircuitEdge,
    CircuitGraph,
    EcologyNeighborContext,
    ExplanationAuditSummary,
    FalsificationOutcome,
    InterventionMode,
    InterventionSpec,
    LayerEcologySolveReport,
    NeuronPoleProfile,
    NeuronTaxonomy,
    PoleType,
    PoleVerificationReport,
    TextActivation,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector

# 'Responses are a share of the neuron's strongest example in the corpus; a strong response is 30% or more.'
PAPER_STRONG_FRACTION = 0.30


# =============================================================================
# Structured outputs
# =============================================================================
class ExplanationResponse(BaseModel):
    positive_pole_explanation: str = Field(..., description="What drives the positive pole, or '' if unknown")
    negative_pole_explanation: str = Field(..., description="What drives the negative pole, or '' if unknown")
    unified_explanation: str = Field(..., description="The distinction the neuron draws between its poles")
    recursive_offshoots: List[str] = Field(default_factory=list)
    positive_pole_taxonomy: NeuronTaxonomy
    negative_pole_taxonomy: NeuronTaxonomy
    confidence: float = Field(..., ge=0.0, le=1.0)


class TestSuiteResponse(BaseModel):
    blind_maximum: str = Field(..., min_length=1)
    positive_examples: List[str] = Field(..., min_length=5, max_length=5)
    near_misses: List[str] = Field(..., min_length=5, max_length=5)
    distant_concept: str = Field(..., min_length=1)
    distant_examples: List[str] = Field(..., min_length=5, max_length=5)


class BundleExplanationResponse(BaseModel):
    explanation: str = Field(..., min_length=1)


class BlindMaximumResponse(BaseModel):
    blind_maximum: str = Field(..., min_length=1)


class RelatednessResponse(BaseModel):
    related: bool
    reason: str = ""


# =============================================================================
# LLM clients
# =============================================================================
class LLMClientProtocol(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """
    JSON chat-completion client for any OpenAI-compatible endpoint (OpenAI, Ollama at
    http://localhost:11434/v1, vLLM, OpenRouter, LM Studio, ...). Retries transport and JSON errors
    with exponential backoff; schema validation is done by the caller.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1",
                 temperature: float = 0.0, max_retries: int = 3, backoff_base: float = 1.0, timeout: float = 120.0,
                 json_mode: bool = True):
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.json_mode = json_mode

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return _parse_json_content(content)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** attempt))
        raise RuntimeError(f"OpenAICompatibleClient failed after {self.max_retries} attempts: {last}") from last


def _parse_json_content(content: str) -> Dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"):]
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < 0:
        raise json.JSONDecodeError("no JSON object in response", content, 0)
    return json.loads(content[start:end + 1])


class CallableLLMClient:
    """Wraps any `fn(system_prompt, user_prompt) -> dict | str` (e.g. a human in the loop or another SDK)."""

    def __init__(self, fn: Callable[[str, str], Any]):
        self.fn = fn

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        out = self.fn(system_prompt, user_prompt)
        return _parse_json_content(out) if isinstance(out, str) else dict(out)


# =============================================================================
# Crossword store
# =============================================================================
class CircuitEcologyMap:
    """
    Stores explanations and traced circuit edges so explained neurons become clues for their real
    circuit neighbours ('Understanding Compounds'). Neighbours are only ever neurons connected by a
    traced edge, or same-layer neurons whose traced top edges reach the same downstream neuron.
    """

    def __init__(self) -> None:
        self.hypotheses: Dict[Tuple[str, int], BipolarVerbHypothesis] = {}
        self.edges_from: Dict[Tuple[str, int], Dict[Tuple[str, int], CircuitEdge]] = {}
        self.edges_to: Dict[Tuple[str, int], Dict[Tuple[str, int], CircuitEdge]] = {}

    def register_hypothesis(self, h: BipolarVerbHypothesis) -> None:
        self.hypotheses[(h.layer_name, h.neuron_idx)] = h

    def get_hypothesis(self, layer_name: str, neuron_idx: int) -> Optional[BipolarVerbHypothesis]:
        return self.hypotheses.get((layer_name, neuron_idx))

    def register_circuit_graph(self, graph: CircuitGraph) -> None:
        for e in graph.edges:
            s, d = (e.src_layer, e.src_neuron), (e.dst_layer, e.dst_neuron)
            self.edges_from.setdefault(s, {})[d] = e
            self.edges_to.setdefault(d, {})[s] = e

    def _ctx(self, layer: str, idx: int, direction: str, role: str, delta: float) -> Optional[EcologyNeighborContext]:
        h = self.hypotheses.get((layer, idx))
        if h is None:
            return None
        return EcologyNeighborContext(
            layer_name=layer, neuron_idx=idx, direction=direction, role=role, mean_causal_delta=delta,
            positive_pole_motif=h.positive_pole_motif, negative_pole_motif=h.negative_pole_motif,
            unified_verb_hypothesis=h.unified_verb_hypothesis,
            positive_outcome=h.outcome(PoleType.POSITIVE).value, negative_outcome=h.outcome(PoleType.NEGATIVE).value,
        )

    def get_neighbor_contexts(self, layer_name: str, neuron_idx: int, max_neighbors: int = 8) -> List[EcologyNeighborContext]:
        key = (layer_name, neuron_idx)
        out: List[EcologyNeighborContext] = []
        for (sl, si), e in self.edges_to.get(key, {}).items():
            c = self._ctx(sl, si, "upstream", e.role.value, e.mean_causal_delta)
            if c:
                out.append(c)
        for (dl, di), e in self.edges_from.get(key, {}).items():
            c = self._ctx(dl, di, "downstream", e.role.value, e.mean_causal_delta)
            if c:
                out.append(c)
        my_readers = set(self.edges_from.get(key, {}).keys())
        if my_readers:
            for (ol, oi), outs in self.edges_from.items():
                if ol != layer_name or oi == neuron_idx:
                    continue
                shared = my_readers & set(outs.keys())
                if shared:
                    r = sorted(shared)[0]
                    c = self._ctx(ol, oi, "shares_downstream_reader", f"both feed {r[0]}[{r[1]}]",
                                  outs[r].mean_causal_delta)
                    if c:
                        out.append(c)
        out.sort(key=lambda n: abs(n.mean_causal_delta), reverse=True)
        return out[:max_neighbors]

    def connected_neurons(self, layer_name: str, neuron_idx: int, limit: int) -> List[Tuple[str, int, float]]:
        key = (layer_name, neuron_idx)
        cands = [(k[0], k[1], e.mean_causal_delta) for k, e in self.edges_from.get(key, {}).items()]
        cands += [(k[0], k[1], e.mean_causal_delta) for k, e in self.edges_to.get(key, {}).items()]
        cands.sort(key=lambda x: abs(x[2]), reverse=True)
        return cands[:limit]

    def summarize(self, layer_name: Optional[str] = None, neuron_indices: Optional[List[int]] = None,
                  sampling: str = "all explained neurons") -> ExplanationAuditSummary:
        s = ExplanationAuditSummary(sampling=sampling)
        pos_strong = pos_tot = nm_strong = nm_tot = d_strong = d_tot = 0
        sel = set(neuron_indices) if neuron_indices is not None else None
        for (lname, idx), h in self.hypotheses.items():
            if layer_name is not None and lname != layer_name:
                continue
            if sel is not None and idx not in sel:
                continue
            for pole in (PoleType.POSITIVE, PoleType.NEGATIVE):
                tax = h.taxonomy(pole)
                if tax == NeuronTaxonomy.UNEXPLAINED:
                    continue
                s.poles_explained += 1
                s.taxonomy_counts[tax.value] = s.taxonomy_counts.get(tax.value, 0) + 1
                rep = h.verification.get(pole.value)
                if rep is None or rep.outcome == FalsificationOutcome.UNTESTED:
                    continue
                s.outcome_counts[rep.outcome.value] = s.outcome_counts.get(rep.outcome.value, 0) + 1
                pos_strong += rep.positive_strong_count
                pos_tot += len(rep.positive_controls)
                nm_strong += rep.near_miss_strong_count
                nm_tot += len(rep.near_misses)
                d_strong += rep.distant_strong_count
                d_tot += len(rep.distant)
                if rep.distant_strong_count == 0:
                    s.poles_never_fired_on_distant += 1
        s.positive_control_strong_rate = pos_strong / pos_tot if pos_tot else 0.0
        s.near_miss_strong_rate = nm_strong / nm_tot if nm_tot else 0.0
        s.distant_strong_rate = d_strong / d_tot if d_tot else 0.0
        return s

    def clear(self) -> None:
        self.hypotheses.clear()
        self.edges_from.clear()
        self.edges_to.clear()


# =============================================================================
# The explainer and the blind test protocol
# =============================================================================
EXPLAINER_SYSTEM_PROMPT = """You are a mechanistic interpretability researcher explaining one MLP neuron of a language model from evidence about it.
Working assumptions (from 'Neural Networks Are Already Clean'):
- A neuron is better described as a verb than a noun: it draws a distinction that downstream neurons use. Its positive and negative poles both matter; describe each.
- Ask what single shared pattern explains the examples. Avoid explanations that are lists of nouns; if the examples look unrelated, look for the shared pattern before concluding they are.
- Read forward as well as backward: many activations are about what comes next (the next token), not the current token.
- Attention often stores a running summary of the context on punctuation (commas, periods). If a neuron fires on punctuation, read the surrounding sentences to see what the summary is about. Some neurons push against a specific next token (for example 'don't say a dot'); check the suppressed logits.
- In gated MLPs the positive pole mixes two sign regimes: '++' (gate open, value positive) and '--' (gate and value both negative, a small echo that usually resembles the negative pole). Explain the positive pole from the '++' examples and treat '--' examples as a separate regime.
- Activation bands are fractions of the pole's strongest activation. Weaker bands usually radiate outward from the strongest meaning (related offshoots). Report offshoots separately from the main job.
- Steering continuations show what the model writes when the neuron is forced to its corpus extremes.
- Neighbour explanations come from neurons connected to this one by traced circuit edges; use them as clues, not as answers.
- Say so when you cannot form a specific explanation. An explanation must be specific enough that it could be wrong.
Pole taxonomy, one per pole:
  one_main_job        - one job, possibly with small related offshoots
  few_related_jobs    - two or three related jobs
  unrelated_jobs      - the examples need jobs that are unrelated to each other
  not_yet_understood  - you cannot form a specific explanation
Return only JSON: {"positive_pole_explanation": str, "negative_pole_explanation": str, "unified_explanation": str,
 "recursive_offshoots": [str], "positive_pole_taxonomy": str, "negative_pole_taxonomy": str, "confidence": float in [0,1]}"""

TEST_WRITER_SYSTEM_PROMPT = """You write test texts for an explanation of one pole of a neuron in a language model.
You never see the neuron or any of its data, only the explanation. A text counts as driving the neuron if the neuron responds strongly somewhere inside it.
Write:
- blind_maximum: one short, readable text you predict drives this pole as strongly as possible (aim at the center of what the explanation describes).
- positive_examples: exactly 5 new, varied texts that should drive it strongly.
- near_misses: exactly 5 texts that share surface features or topic with the explanation but should NOT drive it, because they miss the specific thing it describes.
- distant_concept: the concept most distant from this explanation that you can think of.
- distant_examples: exactly 5 texts about that distant concept.
Return only JSON: {"blind_maximum": str, "positive_examples": [5 str], "near_misses": [5 str], "distant_concept": str, "distant_examples": [5 str]}"""

BUNDLE_EXPLAINER_SYSTEM_PROMPT = """You explain one bundle of upstream neurons that jointly drive a target neuron in a language model.
You get the target's explanation, the contexts where this bundle drives the target most, and what the bundle's upstream neurons respond to.
Describe in one or two sentences what this bundle contributes. Return only JSON: {"explanation": str}"""

BUNDLE_BLIND_MAX_SYSTEM_PROMPT = """You see only a short description of what drives a neuron. Write one short, readable text you predict drives it as strongly as possible.
Return only JSON: {"blind_maximum": str}"""

RELATEDNESS_SYSTEM_PROMPT = """Decide whether two descriptions of what drives the same neuron are related (one grew out of, or shares a pattern with, the other) or unrelated.
Return only JSON: {"related": bool, "reason": str}"""

EncodeFn = Callable[[str], Tuple[torch.Tensor, List[str]]]


class NeuronExplainerAgent:
    """
    Two-agent explanation protocol:
      1. the explainer agent reads the neuron's evidence (both poles, sign regimes, activation bands,
         forward context, logit attribution, steering continuations, traced circuits and explained
         neighbours) and writes an explanation of each pole;
      2. the test-writer agent sees only one pole's explanation and writes a blind maximum, five
         positive examples, five near misses and five texts about the most distant concept;
      3. every text is run through the model; a response is strong when the pole's peak activation in
         the text reaches `strong_fraction` (paper: 30%) of the pole's corpus extreme; the explanation
         wins if more positive examples than near misses are strong, ties if equal, loses otherwise.
    No offline 'explainer' is provided: explanations require an LLM (or human) client.
    """

    def __init__(self, injector: "NeuralInjector", llm_client: Optional[LLMClientProtocol] = None,
                 test_writer_client: Optional[LLMClientProtocol] = None, max_attempts: int = 3,
                 strong_fraction: float = PAPER_STRONG_FRACTION):
        self.injector = injector
        self.llm_client = llm_client
        self.test_writer_client = test_writer_client or llm_client
        self.max_attempts = max_attempts
        self.strong_fraction = strong_fraction
        self.prompt_log: List[Dict[str, str]] = []  # every (role, system, user) sent, for auditability

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------
    def _call(self, client: Optional[LLMClientProtocol], role: str, system: str, user: str, schema: Type[BaseModel]) -> BaseModel:
        if client is None:
            raise RuntimeError(
                "No LLM client configured. Explanations need an explainer model: pass "
                "OpenAICompatibleClient(...) (works with OpenAI, Ollama, vLLM, ...) or CallableLLMClient(fn)."
            )
        last_err = ""
        for _ in range(self.max_attempts):
            prompt = user if not last_err else f"{user}\n\nYour previous answer was invalid: {last_err}\nReturn corrected JSON only."
            self.prompt_log.append({"role": role, "system": system, "user": prompt})
            raw = client.complete_json(system, prompt)
            try:
                return schema.model_validate(raw)
            except ValidationError as exc:
                last_err = str(exc)[:2000]
        raise RuntimeError(f"{role} did not return valid {schema.__name__} after {self.max_attempts} attempts: {last_err}")

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_sample(s: ActivationSample, max_context_chars: Optional[int]) -> Dict[str, Any]:
        ctx = s.context_str
        if ctx is not None and max_context_chars is not None and len(ctx) > max_context_chars:
            ctx = "..." + ctx[-max_context_chars:]
        return {
            "activation": round(s.activation_value, 5),
            "share_of_pole_extreme": s.fraction_of_pole_extreme,
            "sign_regime": s.sign_regime,
            "token": s.token_str,
            "next_token": s.next_token_str,
            "window": s.window_str,
            "context": ctx,
        }

    @staticmethod
    def _describe_top(samples: List[ActivationSample]) -> Dict[str, Any]:
        toks = [s.token_str for s in samples if s.token_str is not None]
        nxt = [s.next_token_str for s in samples if s.next_token_str is not None]
        return {
            "examples": len(samples),
            "distinct_focal_tokens": len(set(toks)),
            "distinct_next_tokens": len(set(nxt)),
            "punctuation_focal_share": (sum(is_punctuation_token(t) for t in toks) / len(toks)) if toks else 0.0,
        }

    def build_evidence_payload(self, profile: NeuronPoleProfile, circuits: Optional[Dict[str, Any]] = None,
                               neighbors: Optional[List[EcologyNeighborContext]] = None,
                               steering: Optional[Dict[str, str]] = None,
                               max_context_chars: Optional[int] = None) -> Dict[str, Any]:
        def _bands(bands: List[ActivationBand]) -> List[Dict[str, Any]]:
            return [
                {
                    "share_of_extreme": [b.lower_fraction, b.upper_fraction],
                    "count": b.count,
                    "regime_counts": b.regime_counts,
                    "examples": [self._fmt_sample(s, max_context_chars) for s in b.samples],
                }
                for b in bands
            ]

        def _moments(m) -> Dict[str, Any]:
            return {"count": m.count, "rate": m.rate, "mean": m.mean, "std": m.std,
                    "skewness": m.skewness, "excess_kurtosis": m.excess_kurtosis, "extreme": m.extreme}

        payload: Dict[str, Any] = {
            "neuron": {"layer": profile.layer_name, "index": profile.neuron_idx,
                       "tokens_observed": profile.tokens_observed, "gated_mlp": profile.gated},
            "positive_pole": {
                "moments": _moments(profile.positive_pole),
                "top_examples_all_regimes": [self._fmt_sample(s, max_context_chars) for s in profile.top_positive_samples],
                "top_summary": self._describe_top(profile.top_positive_samples),
                "bands": _bands(profile.positive_bands),
            },
            "negative_pole": {
                "moments": _moments(profile.negative_pole),
                "top_examples_all_regimes": [self._fmt_sample(s, max_context_chars) for s in profile.top_negative_samples],
                "top_summary": self._describe_top(profile.top_negative_samples),
                "bands": _bands(profile.negative_bands),
            },
            "sign_regimes": {
                r: {"moments": _moments(profile.regime_moments[r]),
                    "top_examples": [self._fmt_sample(s, max_context_chars) for s in profile.regime_top_samples.get(r, [])]}
                for r in profile.regime_moments
            },
            "double_negative_share_of_positive_activations": profile.double_negative_positive_rate,
            "direct_logit_attribution": {
                "promoted": [[t, round(v, 5)] for t, v in profile.top_promoted_tokens],
                "suppressed": [[t, round(v, 5)] for t, v in profile.top_suppressed_tokens],
            },
        }
        if circuits:
            payload["circuits"] = circuits
        if neighbors:
            payload["explained_neighbours"] = [n.__dict__ for n in neighbors]
        if steering:
            payload["steering_continuations"] = steering
        return payload

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def _default_encode(self, tokenizer: Any) -> EncodeFn:
        device = next(self.injector.model.parameters()).device

        def enc(text: str) -> Tuple[torch.Tensor, List[str]]:
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            toks = [tokenizer.decode([i]) for i in ids[0].tolist()]
            return ids, toks

        return enc

    @torch.no_grad()
    def measure_texts(self, layer_name: str, neuron_idx: int, texts: Sequence[str], pole: PoleType,
                      corpus_extreme: float, encode_fn: EncodeFn, forward_fn: Callable = default_forward,
                      specs: Optional[List[InterventionSpec]] = None) -> List[TextActivation]:
        """Peak response of one pole of the neuron inside each text, as a share of the pole's corpus extreme."""
        hs = HookSet(self.injector.model)
        site = hs.site(layer_name)
        got: Dict[str, torch.Tensor] = {}
        hs.add_capture(layer_name, lambda a, g, r: got.update(a=a.detach().float(), g=(g.detach().float() if g is not None else None)),
                       need_gate=site.gated)
        sign = 1.0 if pole == PoleType.POSITIVE else -1.0
        ext = abs(corpus_extreme)
        out: List[TextActivation] = []
        for text in texts:
            ids, toks = encode_fn(text)
            got.clear()
            with self.injector.temporary_interventions(specs or []), hs.active():
                forward_fn(self.injector.model, ids)
            col = flat_neuron_view(got["a"], site.is_conv2d)[:, neuron_idx]
            signed = col * sign
            p = int(signed.argmax())
            peak = float(col[p])
            share = max(0.0, sign * peak) / ext if ext > 0 else 0.0
            regime = None
            if got.get("g") is not None and peak != 0:
                gneg = float(flat_neuron_view(got["g"], site.is_conv2d)[p, neuron_idx]) < 0
                regime = ("--" if gneg else "++") if peak > 0 else ("-+" if gneg else "+-")
            out.append(TextActivation(text=text, peak_activation=peak, share_of_corpus_extreme=share,
                                      strong=share >= self.strong_fraction,
                                      peak_token=toks[p] if p < len(toks) else None, peak_position=p,
                                      peak_sign_regime=regime))
        return out

    def verify_pole(self, layer_name: str, neuron_idx: int, suite: BlindTestSuite, corpus_extreme: float,
                    encode_fn: EncodeFn, forward_fn: Callable = default_forward) -> PoleVerificationReport:
        m = lambda texts: self.measure_texts(layer_name, neuron_idx, texts, suite.pole, corpus_extreme, encode_fn, forward_fn)
        bm = m([suite.blind_maximum])[0]
        pos, nm, dist = m(suite.positive_examples), m(suite.near_misses), m(suite.distant_examples)
        ps, ns, ds = sum(t.strong for t in pos), sum(t.strong for t in nm), sum(t.strong for t in dist)
        outcome = FalsificationOutcome.WON if ps > ns else (FalsificationOutcome.TIED if ps == ns else FalsificationOutcome.LOST)
        return PoleVerificationReport(pole=suite.pole, corpus_extreme=corpus_extreme, strong_fraction=self.strong_fraction,
                                      blind_maximum=bm, positive_controls=pos, near_misses=nm, distant=dist,
                                      positive_strong_count=ps, near_miss_strong_count=ns, distant_strong_count=ds,
                                      outcome=outcome)

    def write_test_suite(self, pole: PoleType, explanation: str) -> BlindTestSuite:
        """Agent 2: receives nothing but the explanation text of one pole."""
        user = json.dumps({"explanation_of_what_drives_the_neuron": explanation})
        r: TestSuiteResponse = self._call(self.test_writer_client, "test_writer", TEST_WRITER_SYSTEM_PROMPT, user, TestSuiteResponse)  # type: ignore[assignment]
        return BlindTestSuite(pole=pole, explanation=explanation, blind_maximum=r.blind_maximum,
                              positive_examples=list(r.positive_examples), near_misses=list(r.near_misses),
                              distant_concept=r.distant_concept, distant_examples=list(r.distant_examples))

    # ------------------------------------------------------------------
    # Causal checks and steering
    # ------------------------------------------------------------------
    def steering_continuations(self, layer_name: str, neuron_idx: int, profile: NeuronPoleProfile,
                               prompts: Sequence[str], n_tokens: int, encode_fn: EncodeFn,
                               decode_fn: Callable[[List[int]], str]) -> Dict[str, str]:
        """Greedy continuations with the neuron clamped to its corpus max and to its corpus min at every position."""
        out: Dict[str, str] = {}
        for pi, prompt in enumerate(prompts):
            ids, _ = encode_fn(prompt)
            for label, value in (("clean", None), ("neuron_at_corpus_max", profile.max_val), ("neuron_at_corpus_min", profile.min_val)):
                if value is None:
                    fac = None
                else:
                    spec = InterventionSpec(layer_name, [neuron_idx], InterventionMode.HARD_CLAMP, PoleType.BOTH, float(value))

                    def fac(_c, spec=spec):
                        from .interventions import InterventionKernel
                        return {layer_name: lambda a, g: InterventionKernel.apply(a, [spec], gate=g)}
                gen = self.injector._greedy_generate(ids, n_tokens, fac)
                out[f"prompt{pi}:{label}"] = prompt + decode_fn(gen)
        return out

    @torch.no_grad()
    def causal_checks(self, layer_name: str, neuron_idx: int, texts: Sequence[str], encode_fn: EncodeFn,
                      n_controls: int, seed: int, forward_fn: Callable = default_forward) -> CausalCheckReport:
        """Zero-ablation KL of the neuron on the texts vs zero-ablation KL of random neurons of the same layer."""
        n = HookSet(self.injector.model).site(layer_name).num_neurons
        rng = random.Random(seed * 7919 + neuron_idx)
        controls = rng.sample([i for i in range(n) if i != neuron_idx], k=min(n_controls, n - 1))

        def kl_of(idx: int) -> float:
            vals = []
            for t in texts:
                ids, _ = encode_fn(t)
                vals.append(self.injector.measure_ablation_kl_divergence(layer_name, idx, ids, forward_fn=forward_fn))
            return sum(vals) / len(vals)

        target = kl_of(neuron_idx)
        ctrl = [kl_of(c) for c in controls]
        pct = sum(c < target for c in ctrl) / len(ctrl) if ctrl else float("nan")
        return CausalCheckReport(target_ablation_kl=target, control_ablation_kls=ctrl, target_kl_percentile_among_controls=pct)

    # ------------------------------------------------------------------
    # Bundles
    # ------------------------------------------------------------------
    def run_bundle_blind_maximum(self, layer_name: str, neuron_idx: int, src_layer: str, profile: NeuronPoleProfile,
                                 main_explanation: str, pole: PoleType, encode_fn: EncodeFn,
                                 forward_fn: Callable = default_forward, max_bundles: int = 6,
                                 top_k_upstream: int = 24) -> List[BundleBlindMaximumResult]:
        """
        'We then run a blind maximum for each bundle, ablating all the others, and compare it against
        the strongest corpus example under the same ablation.'
        """
        samples = profile.top_positive_samples if pole == PoleType.POSITIVE else profile.top_negative_samples
        texts = list(dict.fromkeys(s.context_str for s in samples if s.context_str))
        if len(texts) < 2:
            raise ValueError("Bundle analysis needs at least two distinct activating contexts with text.")
        contexts = [encode_fn(t)[0] for t in texts]
        bundles = self.injector.cluster_incoming_attribution_bundles(src_layer, layer_name, neuron_idx, contexts, pole,
                                                                     forward_fn, top_k_upstream=top_k_upstream,
                                                                     max_bundles=max_bundles, context_labels=texts)
        src_tracker = self.injector.get_stats_tracker(src_layer)
        ext = profile.pole_extreme(pole)
        results = []
        for b in bundles:
            others = sorted({n for o in bundles if o.bundle_id != b.bundle_id for n in o.upstream_neurons})
            ablate = [InterventionSpec(src_layer, others, InterventionMode.ZERO_ABLATE)] if others else []
            up_info = []
            if src_tracker is not None:
                for u in b.upstream_neurons[:8]:
                    p = src_tracker.export_profile(u, samples_per_band=0)
                    up_info.append({"neuron": u, "top_positive": [s.window_str for s in p.top_positive_samples[:5]],
                                    "top_negative": [s.window_str for s in p.top_negative_samples[:5]]})
            user = json.dumps({"target_explanation": main_explanation, "contexts_this_bundle_drives_most": b.driving_contexts,
                               "upstream_neurons": up_info})
            be: BundleExplanationResponse = self._call(self.llm_client, "bundle_explainer", BUNDLE_EXPLAINER_SYSTEM_PROMPT, user, BundleExplanationResponse)  # type: ignore[assignment]
            bm: BlindMaximumResponse = self._call(self.test_writer_client, "bundle_test_writer", BUNDLE_BLIND_MAX_SYSTEM_PROMPT,
                                                  json.dumps({"description": be.explanation}), BlindMaximumResponse)  # type: ignore[assignment]
            blind = self.measure_texts(layer_name, neuron_idx, [bm.blind_maximum], pole, ext, encode_fn, forward_fn, ablate)[0]
            corpus = self.measure_texts(layer_name, neuron_idx, texts, pole, ext, encode_fn, forward_fn, ablate)
            best = max(corpus, key=lambda t: t.share_of_corpus_extreme)
            rel: RelatednessResponse = self._call(self.llm_client, "relatedness", RELATEDNESS_SYSTEM_PROMPT,
                                                  json.dumps({"description_a": main_explanation, "description_b": be.explanation}),
                                                  RelatednessResponse)  # type: ignore[assignment]
            ratio = (blind.share_of_corpus_extreme / best.share_of_corpus_extreme) if best.share_of_corpus_extreme > 0 else float("nan")
            results.append(BundleBlindMaximumResult(
                bundle_id=b.bundle_id, upstream_neurons=b.upstream_neurons, bundle_explanation=be.explanation,
                blind_maximum=bm.blind_maximum, blind_max_activation_under_ablation=blind.peak_activation,
                best_corpus_activation_under_ablation=best.peak_activation, ratio_to_corpus_under_ablation=ratio,
                related_to_main_explanation=rel.related,
            ))
        return results

    # ------------------------------------------------------------------
    # End to end
    # ------------------------------------------------------------------
    def explain_and_verify(
        self,
        layer_name: str,
        neuron_idx: int,
        tokenizer: Optional[Any] = None,
        encode_fn: Optional[EncodeFn] = None,
        decode_fn: Optional[Callable[[List[int]], str]] = None,
        unembed_weight: Optional[torch.Tensor] = None,
        idx_to_token: Optional[Dict[int, str]] = None,
        dst_layer: Optional[str] = None,
        src_layer: Optional[str] = None,
        circuit_inputs: Optional[Any] = None,
        steering_prompts: Optional[Sequence[str]] = None,
        steering_tokens: int = 24,
        n_control_neurons: int = 8,
        run_bundles: bool = False,
        bundle_src_layer: Optional[str] = None,
        max_context_chars: Optional[int] = None,
        forward_fn: Callable = default_forward,
        seed: int = 0,
    ) -> BipolarVerbHypothesis:
        inj = self.injector
        if encode_fn is None:
            if tokenizer is None:
                raise ValueError("Testing explanations needs `tokenizer` or `encode_fn` to turn texts into inputs.")
            encode_fn = self._default_encode(tokenizer)
        if decode_fn is None and tokenizer is not None:
            decode_fn = lambda ids: tokenizer.decode(ids)
        if unembed_weight is None and hasattr(inj.model, "get_output_embeddings"):
            emb = inj.model.get_output_embeddings()
            unembed_weight = emb.weight if emb is not None else None
        if idx_to_token is None and tokenizer is not None and unembed_weight is not None:
            idx_to_token = _LazyVocab(tokenizer)
        profile = inj.get_neuron_profile(layer_name, neuron_idx, unembed_weight=unembed_weight, idx_to_token=idx_to_token)

        circuits: Dict[str, Any] = {}
        if circuit_inputs is not None:
            if dst_layer is not None:
                for pole in (PoleType.POSITIVE, PoleType.NEGATIVE):
                    try:
                        g = inj.trace_circuit(layer_name, dst_layer, neuron_idx, circuit_inputs, src_pole=pole, compute_jacobian=False)
                        circuits[f"downstream_from_{pole.value}_pole"] = g.to_agent_payload()
                    except ValueError:
                        pass
            if src_layer is not None:
                for pole in (PoleType.POSITIVE, PoleType.NEGATIVE):
                    try:
                        g = inj.trace_incoming_circuit(src_layer, layer_name, neuron_idx, circuit_inputs, dst_pole=pole)
                        circuits[f"upstream_into_{pole.value}_pole"] = g.to_incoming_agent_payload()
                    except (ValueError, NotImplementedError):
                        pass

        steering: Dict[str, str] = {}
        if steering_prompts and decode_fn is not None:
            steering = self.steering_continuations(layer_name, neuron_idx, profile, steering_prompts, steering_tokens, encode_fn, decode_fn)

        neighbors = inj.ecology_map.get_neighbor_contexts(layer_name, neuron_idx)
        evidence = self.build_evidence_payload(profile, circuits, neighbors, steering, max_context_chars)
        ex: ExplanationResponse = self._call(self.llm_client, "explainer", EXPLAINER_SYSTEM_PROMPT, json.dumps(evidence), ExplanationResponse)  # type: ignore[assignment]

        hyp = BipolarVerbHypothesis(
            layer_name=layer_name, neuron_idx=neuron_idx,
            positive_pole_motif=ex.positive_pole_explanation, negative_pole_motif=ex.negative_pole_explanation,
            unified_verb_hypothesis=ex.unified_explanation, recursive_offshoots=list(ex.recursive_offshoots),
            pos_taxonomy=ex.positive_pole_taxonomy, neg_taxonomy=ex.negative_pole_taxonomy,
            confidence_score=ex.confidence, neighbor_clues_used=len(neighbors),
        )

        for pole, text, tax, ext in (
            (PoleType.POSITIVE, ex.positive_pole_explanation, ex.positive_pole_taxonomy, profile.max_val),
            (PoleType.NEGATIVE, ex.negative_pole_explanation, ex.negative_pole_taxonomy, profile.min_val),
        ):
            if not text.strip() or tax in (NeuronTaxonomy.NOT_YET_UNDERSTOOD, NeuronTaxonomy.UNEXPLAINED) or ext == 0.0:
                continue
            suite = self.write_test_suite(pole, text)
            hyp.test_suites[pole.value] = suite
            hyp.verification[pole.value] = self.verify_pole(layer_name, neuron_idx, suite, ext, encode_fn, forward_fn)

        pos_suite = hyp.test_suites.get(PoleType.POSITIVE.value) or hyp.test_suites.get(PoleType.NEGATIVE.value)
        if pos_suite is not None and n_control_neurons > 0:
            hyp.causal_checks = self.causal_checks(layer_name, neuron_idx, pos_suite.positive_examples, encode_fn,
                                                   n_control_neurons, seed, forward_fn)
            hyp.causal_checks.steering_generations = steering

        if run_bundles:
            src = bundle_src_layer or src_layer
            if src is None:
                raise ValueError("run_bundles needs `bundle_src_layer` (or `src_layer`).")
            for pole, text in ((PoleType.POSITIVE, ex.positive_pole_explanation), (PoleType.NEGATIVE, ex.negative_pole_explanation)):
                if pole.value in hyp.verification:
                    hyp.bundle_results.extend(
                        self.run_bundle_blind_maximum(layer_name, neuron_idx, src, profile, text, pole, encode_fn, forward_fn)
                    )
        inj.ecology_map.register_hypothesis(hyp)
        return hyp

    # ------------------------------------------------------------------
    # Crossword
    # ------------------------------------------------------------------
    def solve_circuit_ecology(self, seed_neurons: List[Tuple[str, int]], neighbor_layers: List[str],
                              circuit_inputs_fn: Callable[[str, int], Any], max_pass2_neurons: int = 8,
                              edges_per_neuron: int = 4, **explain_kwargs: Any) -> LayerEcologySolveReport:
        """
        Pass 1 explains the seed neurons and traces their real circuit edges into `neighbor_layers`
        (downstream for later layers, upstream for earlier ones) on `circuit_inputs_fn(layer, neuron)`.
        Pass 2 explains the most strongly connected unexplained neurons, whose evidence then contains
        the pass-1 explanations of their traced neighbours.
        """
        inj = self.injector
        order = list(inj.list_targetable_layers().keys())
        report = LayerEcologySolveReport(layer_names=sorted({l for l, _ in seed_neurons} | set(neighbor_layers)),
                                         pass1_neurons=list(seed_neurons), pass2_neurons=[])
        for layer, idx in seed_neurons:
            h = inj.explain_neuron(layer, idx, **explain_kwargs)
            report.hypotheses[(layer, idx)] = h
            report.clues_injected[(layer, idx)] = h.neighbor_clues_used
            inputs = circuit_inputs_fn(layer, idx)
            for nl in neighbor_layers:
                if nl == layer or nl not in order or layer not in order:
                    continue
                try:
                    if order.index(nl) > order.index(layer):
                        inj.trace_circuit(layer, nl, idx, inputs, compute_jacobian=False, top_k=edges_per_neuron)
                    else:
                        inj.trace_incoming_circuit(nl, layer, idx, inputs, top_k=edges_per_neuron)
                except (ValueError, NotImplementedError):
                    continue
        cands: Dict[Tuple[str, int], float] = {}
        for layer, idx in seed_neurons:
            for l2, i2, d in inj.ecology_map.connected_neurons(layer, idx, edges_per_neuron):
                if (l2, i2) not in report.hypotheses and inj.is_layer_audited(l2):
                    cands[(l2, i2)] = max(cands.get((l2, i2), 0.0), abs(d))
        for key in sorted(cands, key=lambda k: cands[k], reverse=True)[:max_pass2_neurons]:
            h = inj.explain_neuron(key[0], key[1], **explain_kwargs)
            report.pass2_neurons.append(key)
            report.hypotheses[key] = h
            report.clues_injected[key] = h.neighbor_clues_used
        return report


class _LazyVocab(dict):
    """idx -> decoded token string, decoded on demand."""

    def __init__(self, tokenizer: Any):
        super().__init__()
        self._tok = tokenizer

    def __bool__(self) -> bool:
        return True

    def get(self, key: int, default: Any = None) -> Any:  # type: ignore[override]
        if key not in self:
            try:
                self[key] = self._tok.decode([key])
            except Exception:
                return default
        return super().get(key, default)
