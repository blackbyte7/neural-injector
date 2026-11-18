from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from .analyzer import (
    DEFAULT_BAND_EDGES,
    CheckpointEvolutionTracker,
    ContrastivePoleFinder,
    LogitLensAttributor,
    StreamingLayerStats,
)
from .circuit_tracer import CircuitTracer
from .discovery import LayerDescriptor, LayerDiscovery
from .explainer import CircuitEcologyMap, LLMClientProtocol, NeuronExplainerAgent
from .exporter import SteeringExporter
from .hooks import HookSet, default_forward, extract_logits, flat_neuron_view, resolve_site, valid_position_mask
from .interventions import AblationMaskOptimizer, InterventionKernel, build_intervention_hookset
from .types import (
    BipolarVerbHypothesis,
    CircuitGraph,
    CombinationReadoutReport,
    ContrastiveDiscoveryReport,
    ContrastiveNeuronMatch,
    ExplanationAuditSummary,
    ExtremeCullReport,
    GateConditionSpec,
    InterventionEffectReport,
    InterventionMode,
    InterventionSpec,
    LayerEcologySolveReport,
    LayerScanReport,
    NeuronEvolutionReport,
    NeuronPoleProfile,
    PoleType,
    ProductionPresetBundle,
    ResidualSinkReport,
    SynergisticAblationResult,
    UpstreamAttributionBundle,
)

# A modifier factory receives a counter dict and returns {layer_name: modifier(act, gate) -> act}
ModifierFactory = Callable[[Dict[str, int]], Dict[str, Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]]]


class NeuralInjector:
    """
    Native-basis neuron auditing, surgical intervention, circuit tracing and explanation testing
    following 'Neural Networks Are Already Clean' (Cammarata, Goodfire 2026), plus weight folding
    and export of verified steering interventions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        tokenizer: Optional[Any] = None,
        allowed_workspace_dir: Optional[Union[str, Path]] = None,
        test_writer_client: Optional[LLMClientProtocol] = None,
    ):
        self.model = model
        try:
            default_dev = str(next(model.parameters()).device)
        except StopIteration:
            default_dev = "cpu"
        self.device = str(device) if device is not None else default_dev
        self.tokenizer = tokenizer
        self.allowed_workspace_dir = allowed_workspace_dir
        self._lock = threading.RLock()
        self.discovered_layers: Dict[str, LayerDescriptor] = LayerDiscovery.discover_neuron_layers(model)

        self._interventions: Dict[str, List[InterventionSpec]] = {}
        self._hookset: Optional[HookSet] = None
        self._extra_handles: List[RemovableHandle] = []
        self._latched_masks: Dict[int, torch.Tensor] = {}
        self._modification_counter: Optional[Dict[str, int]] = None
        self._stats_trackers: Dict[str, StreamingLayerStats] = {}

        self.ecology_map = CircuitEcologyMap()
        self.tracer = CircuitTracer(self)
        self.explainer = NeuronExplainerAgent(self, llm_client=llm_client, test_writer_client=test_writer_client)
        self.ablation_optimizer = AblationMaskOptimizer(self)

    # -------------------------------------------------------------------------
    # 1. Discovery and state
    # -------------------------------------------------------------------------
    def list_targetable_layers(self, include_all_linear: bool = False) -> Dict[str, LayerDescriptor]:
        if include_all_linear:
            return LayerDiscovery.discover_neuron_layers(self.model, include_all_linear=True)
        return self.discovered_layers

    def is_layer_audited(self, layer_name: str) -> bool:
        return layer_name in self._stats_trackers

    def get_stats_tracker(self, layer_name: str) -> Optional[StreamingLayerStats]:
        return self._stats_trackers.get(layer_name)

    def neuron_write_vectors(self, layer_name: str) -> Optional[torch.Tensor]:
        return LayerDiscovery.neuron_write_vectors(self.model, layer_name)

    def _require_tracker(self, layer_name: str) -> StreamingLayerStats:
        if layer_name not in self._stats_trackers:
            raise KeyError(f"Layer '{layer_name}' has not been audited yet. Call `audit_layers` first.")
        return self._stats_trackers[layer_name]

    @property
    def active_intervention_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._interventions.values())

    @property
    def active_hook_count(self) -> int:
        with self._lock:
            n = len(self._hookset._handles) if self._hookset is not None else 0
            return n + len(self._extra_handles)

    def get_active_interventions(self) -> List[InterventionSpec]:
        with self._lock:
            return [s for specs in self._interventions.values() for s in specs]

    def reset_latched_gates(self) -> None:
        with self._lock:
            self._latched_masks.clear()

    def _module_order(self) -> List[str]:
        return [n for n, _ in self.model.named_modules() if n]

    def _validate_detector_layer_order(self, detector_layer: str, target_layer: str) -> None:
        if detector_layer == target_layer:
            return
        order = self._module_order()
        d_base, _ = LayerDiscovery.split_expert_layer_name(detector_layer)
        t_base, _ = LayerDiscovery.split_expert_layer_name(target_layer)
        if d_base in order and t_base in order and order.index(d_base) > order.index(t_base):
            raise ValueError(
                f"Invalid CONDITIONAL_GATE: detector '{detector_layer}' executes after target '{target_layer}'."
            )

    # -------------------------------------------------------------------------
    # 2. Interventions
    # -------------------------------------------------------------------------
    def add_intervention(self, spec: InterventionSpec) -> None:
        with self._lock:
            site = resolve_site(self.model, spec.layer_name)
            for i in spec.neuron_indices:
                if not 0 <= int(i) < site.num_neurons:
                    raise ValueError(f"Neuron {i} out of range for '{spec.layer_name}' ({site.num_neurons} neurons).")
            if spec.needs_gate_signal and not site.gated:
                raise ValueError(f"{spec.mode.value} needs a gated MLP, but '{spec.layer_name}' has no gate projection.")
            if spec.mode == InterventionMode.EXTREME_CULL and spec.condition_signal not in ("gate", "activation"):
                raise ValueError("condition_signal must be 'gate' or 'activation'")
            if site.target == "moe_expert" and (spec.token_positions is not None or spec.gate_condition is not None):
                raise ValueError("token_positions and conditional gates are not defined for routed MoE expert rows.")
            if spec.gate_condition is not None:
                self._validate_detector_layer_order(spec.gate_condition.detector_layer, spec.layer_name)
            self._interventions.setdefault(spec.layer_name, []).append(spec)

    def zero_ablate(self, layer_name: str, neuron_indices: Union[int, List[int]], pole: PoleType = PoleType.BOTH,
                    token_positions: Optional[List[int]] = None) -> InterventionSpec:
        idx = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        spec = InterventionSpec(layer_name, idx, InterventionMode.ZERO_ABLATE, pole, 0.0, token_positions=token_positions)
        self.add_intervention(spec)
        return spec

    def clamp_neurons(self, layer_name: str, neuron_indices: Union[int, List[int]], value: float,
                      pole: PoleType = PoleType.BOTH, token_positions: Optional[List[int]] = None,
                      per_neuron_values: Optional[List[float]] = None) -> InterventionSpec:
        idx = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        spec = InterventionSpec(layer_name, idx, InterventionMode.HARD_CLAMP, pole, value,
                                per_neuron_values=per_neuron_values, token_positions=token_positions)
        self.add_intervention(spec)
        return spec

    def steer_neurons(self, layer_name: str, neuron_indices: Union[int, List[int]], delta: Optional[float] = None,
                      scale: Optional[float] = None, pole: PoleType = PoleType.BOTH,
                      per_neuron_values: Optional[List[float]] = None,
                      token_positions: Optional[List[int]] = None) -> InterventionSpec:
        """ADDITIVE_STEER by `delta` (or per-neuron deltas) or SCALE by `scale`; exactly one is required."""
        if (delta is None) == (scale is None) and per_neuron_values is None:
            raise ValueError("Specify exactly one of `delta` or `scale` (or `per_neuron_values` with a mode).")
        idx = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        mode = InterventionMode.SCALE if scale is not None else InterventionMode.ADDITIVE_STEER
        val = float(scale if scale is not None else (delta if delta is not None else 0.0))
        spec = InterventionSpec(layer_name, idx, mode, pole, val, per_neuron_values=per_neuron_values,
                                token_positions=token_positions)
        self.add_intervention(spec)
        return spec

    def prune_weak_activations(
        self,
        layer_name: str,
        neuron_indices: Optional[Union[int, List[int]]] = None,
        threshold: Optional[float] = None,
        fraction_of_extreme: Optional[float] = None,
        percentile: Optional[float] = None,
        std_multiplier: Optional[float] = None,
        pole: PoleType = PoleType.BOTH,
    ) -> List[InterventionSpec]:
        """
        'Do Weak Activations Matter?': zero activations whose magnitude is below a threshold.
        Exactly one rule: an absolute `threshold`, or per-neuron rules from the audit
        (`fraction_of_extreme` of each pole's own extreme, within-pole `percentile`, or `std_multiplier`).
        Per-neuron rules create one spec per pole, since the two poles have different scales.
        """
        n = resolve_site(self.model, layer_name).num_neurons
        idx = list(range(n)) if neuron_indices is None else ([neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices))
        rules = [x is not None for x in (threshold, fraction_of_extreme, percentile, std_multiplier)]
        if sum(rules) != 1:
            raise ValueError("Specify exactly one of threshold, fraction_of_extreme, percentile, std_multiplier.")
        poles = [PoleType.POSITIVE, PoleType.NEGATIVE] if pole == PoleType.BOTH else [pole]
        specs: List[InterventionSpec] = []
        if threshold is not None:
            for p in poles:
                specs.append(InterventionSpec(layer_name, idx, InterventionMode.WEAK_PRUNE, p, float(threshold)))
        else:
            tr = self._require_tracker(layer_name)
            for p in poles:
                vals = tr.compute_thresholds(idx, p, fraction_of_extreme=fraction_of_extreme,
                                             percentile=percentile, std_multiplier=std_multiplier)
                specs.append(InterventionSpec(layer_name, idx, InterventionMode.WEAK_PRUNE, p, 0.0, per_neuron_values=vals))
        for s in specs:
            self.add_intervention(s)
        return specs

    def ablate_double_negative_regime(self, layer_names: Optional[List[str]] = None) -> List[InterventionSpec]:
        """
        Zero every activation in the -- regime (g < 0, u < 0) of the given gated layers, or of every
        gated layer in the model ('Ablating them across the whole network ups most benchmarks,
        but lowers some reasoning ones').
        """
        names = layer_names or [n for n in self.discovered_layers if resolve_site(self.model, n).gated]
        specs = []
        for name in names:
            n = resolve_site(self.model, name).num_neurons
            spec = InterventionSpec(name, list(range(n)), InterventionMode.DOUBLE_NEGATIVE_ABLATE, PoleType.POSITIVE)
            self.add_intervention(spec)
            specs.append(spec)
        return specs

    def add_conditional_gate(
        self,
        target_layer: str,
        target_neurons: Union[int, List[int]],
        detector_layer: str,
        detector_neuron: int,
        detector_pole: PoleType = PoleType.POSITIVE,
        threshold: Optional[float] = None,
        detector_fraction_of_extreme: Optional[float] = None,
        detector_percentile: Optional[float] = None,
        action_mode: InterventionMode = InterventionMode.ADDITIVE_STEER,
        value: float = 0.0,
        target_pole: PoleType = PoleType.BOTH,
        sequence_scope: str = "token_wise",
        sticky_on_decode: bool = True,
    ) -> InterventionSpec:
        """Closed-loop gate: `action_mode` on the targets only where the detector neuron fires beyond a threshold."""
        rules = [x is not None for x in (threshold, detector_fraction_of_extreme, detector_percentile)]
        if sum(rules) != 1:
            raise ValueError("Specify exactly one of threshold, detector_fraction_of_extreme, detector_percentile.")
        if detector_pole == PoleType.BOTH and threshold is None:
            raise ValueError("A data-derived detector threshold needs a single detector pole.")
        if threshold is not None:
            thr = float(threshold)
        else:
            thr = self._require_tracker(detector_layer).compute_thresholds(
                [detector_neuron], detector_pole, fraction_of_extreme=detector_fraction_of_extreme,
                percentile=detector_percentile,
            )[0]
        idx = [target_neurons] if isinstance(target_neurons, int) else list(target_neurons)
        cond = GateConditionSpec(detector_layer, detector_neuron, thr, detector_pole, action_mode, sequence_scope, sticky_on_decode)
        spec = InterventionSpec(target_layer, idx, InterventionMode.CONDITIONAL_GATE, target_pole, value, gate_condition=cond)
        self.add_intervention(spec)
        return spec

    def clear_interventions(self) -> None:
        with self._lock:
            self.detach_hooks()
            self._interventions.clear()
            self._latched_masks.clear()

    def attach_hooks(self) -> None:
        with self._lock:
            self.detach_hooks()
            if not self._interventions:
                return
            hs, cond_masks = build_intervention_hookset(self.model, self._interventions, self._latched_masks,
                                                        self._modification_counter)
            self._extra_handles.append(self.model.register_forward_pre_hook(lambda _m, _i: cond_masks.clear()))
            hs.attach()
            self._hookset = hs

    def detach_hooks(self) -> None:
        with self._lock:
            if self._hookset is not None:
                self._hookset.detach()
                self._hookset = None
            for h in self._extra_handles:
                h.remove()
            self._extra_handles.clear()

    @contextmanager
    def active_injection(self):
        with self._lock:
            self.reset_latched_gates()
            self.attach_hooks()
            try:
                yield self
            finally:
                self.detach_hooks()
                self.reset_latched_gates()

    @contextmanager
    def without_interventions(self):
        with self._lock:
            was = self._hookset is not None
            if was:
                self.detach_hooks()
            try:
                yield self
            finally:
                if was:
                    self.attach_hooks()

    @contextmanager
    def temporary_interventions(self, temp_specs: Optional[List[InterventionSpec]] = None):
        with self._lock:
            saved = {k: list(v) for k, v in self._interventions.items()}
            saved_latch = dict(self._latched_masks)
            was = self._hookset is not None
            self.detach_hooks()
            self._interventions = {}
            self._latched_masks.clear()
            try:
                for s in temp_specs or []:
                    self.add_intervention(s)
                self.attach_hooks()
                yield self
            finally:
                self.detach_hooks()
                self._interventions = saved
                self._latched_masks.clear()
                self._latched_masks.update(saved_latch)
                if was:
                    self.attach_hooks()

    @contextmanager
    def clean_forward(self):
        with self.temporary_interventions([]):
            yield self

    # -------------------------------------------------------------------------
    # 3. Auditing
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def audit_layers(
        self,
        layer_names: List[str],
        dataloader_or_batches: Iterable[Dict[str, Any]],
        forward_fn: Callable[[nn.Module, Dict[str, Any]], Any] = default_forward,
        top_k: int = 16,
        **tracker_kwargs: Any,
    ) -> Dict[str, StreamingLayerStats]:
        """
        Streams batches through the unmodified model and records, for each layer, the exact streaming
        statistics of `StreamingLayerStats`. Batch dicts may carry `attention_mask`, `sample_ids`,
        `tokens_per_sample` (decoded token strings) and `contexts_per_sample` (sample texts).
        """
        with self._lock:
            hs = HookSet(self.model)
            meta: Dict[str, Any] = {}
            for name in layer_names:
                site = hs.site(name)
                tracker = StreamingLayerStats(name, site.num_neurons, gated=site.gated, top_k=top_k,
                                              device=self.device, **tracker_kwargs)
                self._stats_trackers[name] = tracker

                def _cap(act: torch.Tensor, gate: Optional[torch.Tensor], routed: Optional[torch.Tensor], t=tracker, s=site) -> None:
                    grid = meta.get("grid")
                    if routed is not None and grid is None:
                        raise ValueError("MoE expert audits need batches with a 2D `input_ids` tensor to map routed rows to positions.")
                    t.update(
                        act.detach(),
                        sample_ids=meta.get("sample_ids"),
                        tokens_per_sample=meta.get("tokens_per_sample"),
                        contexts_per_sample=meta.get("contexts_per_sample"),
                        attention_mask=meta.get("attention_mask") if routed is None else None,
                        gate_activations=gate.detach() if gate is not None else None,
                        routed_token_indices=routed,
                        token_grid=grid if routed is not None else None,
                    )

                hs.add_capture(name, _cap, need_gate=site.gated, dense_moe=False)

            was_training = self.model.training
            self.model.eval()
            try:
                with self.clean_forward(), hs.active():
                    for batch in dataloader_or_batches:
                        meta.clear()
                        if isinstance(batch, dict):
                            for k in ("sample_ids", "tokens_per_sample", "contexts_per_sample", "attention_mask"):
                                meta[k] = batch.get(k)
                            ids = batch.get("input_ids")
                            if isinstance(ids, torch.Tensor) and ids.ndim == 2:
                                meta["grid"] = (int(ids.shape[0]), int(ids.shape[1]))
                        elif isinstance(batch, torch.Tensor) and batch.ndim == 2 and not batch.is_floating_point():
                            meta["grid"] = (int(batch.shape[0]), int(batch.shape[1]))
                        forward_fn(self.model, batch)
            finally:
                if was_training:
                    self.model.train()
            return {n: self._stats_trackers[n] for n in layer_names}

    def get_neuron_profile(
        self,
        layer_name: str,
        neuron_idx: int,
        unembed_weight: Optional[torch.Tensor] = None,
        idx_to_token: Optional[Dict[int, str]] = None,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
        samples_per_band: int = 4,
        logit_top_k: int = 10,
    ) -> NeuronPoleProfile:
        with self._lock:
            profile = self._require_tracker(layer_name).export_profile(neuron_idx, band_edges, samples_per_band)
            if unembed_weight is not None:
                wv = self.neuron_write_vectors(layer_name)
                if wv is not None:
                    promoted, suppressed = LogitLensAttributor.attribute_write_vector(
                        wv[neuron_idx], unembed_weight, idx_to_token=idx_to_token, top_k=logit_top_k,
                        final_norm_module=LayerDiscovery.resolve_final_norm(self.model),
                    )
                    profile.top_promoted_tokens = promoted
                    profile.top_suppressed_tokens = suppressed
            return profile

    # -------------------------------------------------------------------------
    # 4. Causal measurements
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def measure_ablation_kl_divergence(
        self,
        layer_name: str,
        neuron_idx: Union[int, List[int]],
        inputs: Any,
        pole: PoleType = PoleType.BOTH,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Mean over valid positions of KL(P_clean || P_ablated) when the neuron(s) are zero-ablated."""
        run = forward_fn or default_forward
        idx = [neuron_idx] if isinstance(neuron_idx, int) else list(neuron_idx)
        if attention_mask is None and isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
        with self._lock:
            was_training = self.model.training
            self.model.eval()
            try:
                with self.clean_forward():
                    clean = F.log_softmax(extract_logits(run(self.model, inputs)).float(), dim=-1)
                spec = InterventionSpec(layer_name, idx, InterventionMode.ZERO_ABLATE, pole)
                with self.temporary_interventions([spec]):
                    abl = F.log_softmax(extract_logits(run(self.model, inputs)).float(), dim=-1)
                kl = (clean.exp() * (clean - abl)).sum(dim=-1).reshape(-1)
                valid = valid_position_mask(attention_mask, kl.numel(), kl.device)
                return float(kl[valid].mean().item())
            finally:
                if was_training:
                    self.model.train()

    @torch.no_grad()
    def measure_token_set_effect(
        self,
        specs: List[InterventionSpec],
        inputs: Any,
        token_ids: Sequence[int],
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Change in next-token log-probability mass of a token set (e.g. every vocabulary token that
        starts with '.') under the given interventions — the 'Don't Say Dot' check: remove the
        neuron and see whether the next period gets likelier.
        """
        run = forward_fn or default_forward
        if attention_mask is None and isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
        t_idx = torch.as_tensor(list(token_ids), dtype=torch.long)
        with self._lock:
            with self.clean_forward():
                c = F.log_softmax(extract_logits(run(self.model, inputs)).float(), dim=-1)
            with self.temporary_interventions(specs):
                m = F.log_softmax(extract_logits(run(self.model, inputs)).float(), dim=-1)
        t_idx = t_idx.to(c.device)
        c_mass = torch.logsumexp(c[..., t_idx], dim=-1).reshape(-1)
        m_mass = torch.logsumexp(m[..., t_idx], dim=-1).reshape(-1)
        valid = valid_position_mask(attention_mask, c_mass.numel(), c_mass.device)
        delta = (m_mass - c_mass)[valid]
        return {
            "positions": int(valid.sum().item()),
            "mean_clean_logprob_mass": float(c_mass[valid].mean().item()),
            "mean_intervened_logprob_mass": float(m_mass[valid].mean().item()),
            "mean_delta_logprob_mass": float(delta.mean().item()),
            "per_position_delta": delta.cpu().tolist(),
        }

    def _batch_labels(self, batch: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            return batch.get("input_ids"), batch.get("attention_mask")
        if isinstance(batch, torch.Tensor) and not batch.is_floating_point():
            return batch, None
        return None, None

    @torch.no_grad()
    def _evaluate_modifiers(
        self,
        batches: List[Any],
        forward_fn: Callable[[nn.Module, Any], Any],
        label: str,
        factory: ModifierFactory,
        need_gate: Optional[Dict[str, bool]] = None,
    ) -> InterventionEffectReport:
        tot_tokens = 0
        clean_nll = mod_nll = kl_sum = 0.0
        agree = flips = 0
        flip_margin = all_margin = 0.0
        counter: Dict[str, int] = {}
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in batches:
                ids, mask = self._batch_labels(batch)
                with self.clean_forward():
                    cl = extract_logits(forward_fn(self.model, batch)).float()
                hs = HookSet(self.model)
                for name, fn in factory(counter).items():
                    hs.add_modifier(name, fn, need_gate=bool(need_gate and need_gate.get(name)))
                with self.clean_forward(), hs.active():
                    ml = extract_logits(forward_fn(self.model, batch)).float()
                c_lp = F.log_softmax(cl, dim=-1)
                m_lp = F.log_softmax(ml, dim=-1)
                if ids is not None and cl.ndim == 3:
                    tgt = ids[:, 1:].to(cl.device)
                    c_lp_s, m_lp_s = c_lp[:, :-1], m_lp[:, :-1]
                    valid = torch.ones_like(tgt, dtype=torch.bool)
                    if mask is not None:
                        mk = mask.to(cl.device).bool()
                        valid = mk[:, 1:] & mk[:, :-1]
                    clean_nll += float(-c_lp_s.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[valid].sum())
                    mod_nll += float(-m_lp_s.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[valid].sum())
                    pos_valid = valid
                    c_use, m_use = c_lp_s, m_lp_s
                else:
                    c_use, m_use = c_lp, m_lp
                    pos_valid = valid_position_mask(mask, c_lp.shape[:-1].numel(), cl.device).reshape(c_lp.shape[:-1])
                kl = (c_use.exp() * (c_use - m_use)).sum(-1)
                kl_sum += float(kl[pos_valid].sum())
                c_top = c_use.topk(2, dim=-1)
                m_top1 = m_use.argmax(-1)
                same = (c_top.indices[..., 0] == m_top1) & pos_valid
                agree += int(same.sum())
                flipped = (~same) & pos_valid
                flips += int(flipped.sum())
                probs = c_top.values.exp()
                margin = probs[..., 0] - probs[..., 1]
                flip_margin += float(margin[flipped].sum())
                all_margin += float(margin[pos_valid].sum())
                tot_tokens += int(pos_valid.sum())
        finally:
            if was_training:
                self.model.train()
        n = max(tot_tokens, 1)
        return InterventionEffectReport(
            label=label,
            tokens_evaluated=tot_tokens,
            clean_nll=clean_nll / n,
            intervened_nll=mod_nll / n,
            mean_kl=kl_sum / n,
            top1_agreement=agree / n,
            flipped_positions=flips,
            flipped_clean_margin_mean=flip_margin / flips if flips else float("nan"),
            all_clean_margin_mean=all_margin / n,
            entries_modified=sum(counter.values()),
        )

    def evaluate_intervention_effect(
        self,
        batches: Iterable[Any],
        specs: Optional[List[InterventionSpec]] = None,
        forward_fn: Callable[[nn.Module, Any], Any] = default_forward,
        label: str = "interventions",
    ) -> InterventionEffectReport:
        """
        Next-token NLL, KL to the clean model and top-1 agreement under `specs` (or the currently
        registered interventions). Flipped predictions are reported with their clean top-1/top-2
        probability margin, to see whether changes only tip near-ties.
        """
        specs = specs if specs is not None else self.get_active_interventions()
        by_layer: Dict[str, List[InterventionSpec]] = {}
        for s in specs:
            by_layer.setdefault(s.layer_name, []).append(s)
        if any(s.gate_condition is not None for s in specs):
            raise ValueError("evaluate_intervention_effect does not support conditional gates; use active_injection().")

        def factory(counter: Dict[str, int]):
            out = {}
            for name, ls in by_layer.items():
                conv = resolve_site(self.model, name).is_conv2d

                def fn(act, gate, ls=ls, nm=name, conv=conv):
                    new = InterventionKernel.apply(act, ls, is_conv2d=conv, gate=gate)
                    counter[nm] = counter.get(nm, 0) + int((new != act).sum().item())
                    return new
                out[name] = fn
            return out

        need = {n: any(s.needs_gate_signal for s in ls) for n, ls in by_layer.items()}
        with self._lock:
            return self._evaluate_modifiers(list(batches), forward_fn, label, factory, need)

    def measure_weak_activation_ablation(
        self,
        layer_names: List[str],
        batches: Iterable[Any],
        fraction_of_extreme: Optional[float] = None,
        percentile: Optional[float] = None,
        forward_fn: Callable[[nn.Module, Any], Any] = default_forward,
        seed: int = 0,
    ) -> Dict[str, InterventionEffectReport]:
        """
        Removes the weak activations of every neuron in `layer_names` (per pole, below a fraction of
        the pole's own extreme or below a within-pole percentile) and compares with a control that
        zeros the same number of activations per neuron chosen uniformly at random.
        """
        batches = list(batches)
        th: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in layer_names:
            tr = self._require_tracker(name)
            idx = list(range(tr.num_neurons))
            tp = tr.compute_thresholds(idx, PoleType.POSITIVE, fraction_of_extreme=fraction_of_extreme, percentile=percentile)
            tn = tr.compute_thresholds(idx, PoleType.NEGATIVE, fraction_of_extreme=fraction_of_extreme, percentile=percentile)
            th[name] = (torch.tensor(tp), torch.tensor(tn))
        gen = torch.Generator().manual_seed(seed)

        def _weak_mask(act: torch.Tensor, name: str) -> torch.Tensor:
            tp, tn = th[name]
            shape = [1] * (act.ndim - 1) + [-1]
            tp_ = tp.to(act.device, act.dtype).view(*shape)
            tn_ = tn.to(act.device, act.dtype).view(*shape)
            return ((act > 0) & (act < tp_)) | ((act < 0) & (-act < tn_))

        def weak_factory(counter):
            def make(name):
                def fn(act, _g):
                    m = _weak_mask(act, name)
                    counter[name] = counter.get(name, 0) + int(m.sum())
                    return torch.where(m, torch.zeros_like(act), act)
                return fn
            return {n: make(n) for n in layer_names}

        def random_factory(counter):
            def make(name):
                def fn(act, _g):
                    m = _weak_mask(act, name)
                    flat = flat_neuron_view(act, False)
                    mflat = flat_neuron_view(m, False)
                    cnt = mflat.sum(dim=0)
                    keys = torch.rand(flat.shape, generator=gen).to(flat.device)
                    ranks = keys.argsort(dim=0).argsort(dim=0)
                    rm = ranks < cnt.unsqueeze(0)
                    counter[name] = counter.get(name, 0) + int(rm.sum())
                    out = torch.where(rm, torch.zeros_like(flat), flat)
                    return out.reshape(act.shape)
                return fn
            return {n: make(n) for n in layer_names}

        rule = f"fraction_of_extreme={fraction_of_extreme}" if fraction_of_extreme is not None else f"percentile={percentile}"
        with self._lock:
            return {
                "weak_removed": self._evaluate_modifiers(batches, forward_fn, f"weak activations removed ({rule})", weak_factory),
                "count_matched_random": self._evaluate_modifiers(batches, forward_fn, "count-matched random removal", random_factory),
            }

    @torch.no_grad()
    def calibrate_tail_thresholds(
        self,
        layer_names: List[str],
        batches: Iterable[Any],
        percentile: float = 99.9,
        signal: str = "gate",
        forward_fn: Callable[[nn.Module, Any], Any] = default_forward,
    ) -> Dict[str, List[float]]:
        """
        Exact per-neuron upper-tail order statistic of the gate pre-activation (or the activation)
        over all valid tokens of `batches`: the value exceeded by a (100 - percentile)% share of tokens.
        Two passes: one to count tokens, one keeping the exact top-(K+1) values per neuron.
        """
        if signal not in ("gate", "activation"):
            raise ValueError("signal must be 'gate' or 'activation'")
        batches = list(batches)
        q = percentile / 100.0

        def _collect(fn_per_layer):
            hs = HookSet(self.model)
            for name in layer_names:
                site = hs.site(name)
                if signal == "gate" and not site.gated:
                    raise ValueError(f"'{name}' has no gate projection.")
                hs.add_capture(name, fn_per_layer(name, site), need_gate=(signal == "gate"), dense_moe=False)
            meta: Dict[str, Any] = {}
            with self.clean_forward(), hs.active():
                for b in batches:
                    meta["mask"] = b.get("attention_mask") if isinstance(b, dict) else None
                    _collect.meta = meta
                    forward_fn(self.model, b)

        counts: Dict[str, int] = {n: 0 for n in layer_names}

        def count_fn(name, site):
            def fn(act, gate, routed):
                x = flat_neuron_view(act, site.is_conv2d)
                m = _collect.meta.get("mask") if routed is None else None
                counts[name] += int(valid_position_mask(m, x.shape[0], x.device).sum()) if m is not None else x.shape[0]
            return fn

        _collect.meta = {}
        _collect(count_fn)
        k_keep = {n: int(math.floor((1.0 - q) * counts[n])) + 1 for n in layer_names}
        tops: Dict[str, Optional[torch.Tensor]] = {n: None for n in layer_names}

        def top_fn(name, site):
            def fn(act, gate, routed):
                src = gate if signal == "gate" else act
                x = flat_neuron_view(src.float(), site.is_conv2d)
                m = _collect.meta.get("mask") if routed is None else None
                if m is not None:
                    x = x[valid_position_mask(m, x.shape[0], x.device)]
                cur = tops[name]
                cat = x if cur is None else torch.cat([cur, x], dim=0)
                k = min(k_keep[name], cat.shape[0])
                tops[name] = torch.topk(cat, k=k, dim=0).values
            return fn

        _collect(top_fn)
        out: Dict[str, List[float]] = {}
        for n in layer_names:
            t = tops[n]
            if t is None or t.shape[0] < k_keep[n]:
                raise ValueError(f"Not enough tokens in '{n}' for the {percentile} percentile.")
            out[n] = t[k_keep[n] - 1].cpu().tolist()
        return out

    def run_extreme_culling_experiment(
        self,
        thresholds: Dict[str, List[float]],
        batches: Iterable[Any],
        percentile: float,
        signal: str = "gate",
        forward_fn: Callable[[nn.Module, Any], Any] = default_forward,
        seed: int = 0,
        generation_prompt_ids: Optional[torch.Tensor] = None,
        generation_tokens: int = 0,
        decode_fn: Optional[Callable[[List[int]], str]] = None,
    ) -> ExtremeCullReport:
        """
        'Remove extreme responses': zero each neuron's output wherever its gate (or activation) exceeds
        its calibrated tail threshold, against two random controls applied to the same forward pass:
        the same number of zeroed activations per neuron chosen at random (count-matched), and random
        activations whose summed |a| matches the culled |a| per neuron (magnitude-matched).
        """
        batches = list(batches)
        layers = list(thresholds.keys())
        th = {n: torch.tensor(v) for n, v in thresholds.items()}
        gen = torch.Generator().manual_seed(seed)

        def _cull_mask(act, gate, name):
            src = gate if signal == "gate" else act
            shape = [1] * (act.ndim - 1) + [-1]
            return src > th[name].to(src.device, src.dtype).view(*shape)

        def cull_factory(counter):
            def make(name):
                def fn(act, gate):
                    m = _cull_mask(act, gate, name)
                    counter[name] = counter.get(name, 0) + int(m.sum())
                    return torch.where(m, torch.zeros_like(act), act)
                return fn
            return {n: make(n) for n in layers}

        def count_factory(counter):
            def make(name):
                def fn(act, gate):
                    m = flat_neuron_view(_cull_mask(act, gate, name), False)
                    flat = flat_neuron_view(act, False)
                    cnt = m.sum(0)
                    keys = torch.rand(flat.shape, generator=gen).to(flat.device)
                    keys = torch.where(m, torch.full_like(keys, 2.0), keys)  # never pick culled entries
                    ranks = keys.argsort(0).argsort(0)
                    rm = ranks < cnt.unsqueeze(0)
                    counter[name] = counter.get(name, 0) + int(rm.sum())
                    return torch.where(rm, torch.zeros_like(flat), flat).reshape(act.shape)
                return fn
            return {n: make(n) for n in layers}

        def mag_factory(counter):
            def make(name):
                def fn(act, gate):
                    m = flat_neuron_view(_cull_mask(act, gate, name), False)
                    flat = flat_neuron_view(act, False)
                    target = (flat.abs() * m).sum(0)
                    keys = torch.rand(flat.shape, generator=gen).to(flat.device)
                    keys = torch.where(m, torch.full_like(keys, 2.0), keys)
                    order = keys.argsort(0)
                    mags = torch.gather(flat.abs() * (~m), 0, order)
                    cum = mags.cumsum(0)
                    # smallest random prefix whose summed |a| reaches the culled mass
                    take_sorted = (cum - mags) < target.unsqueeze(0)
                    take_sorted &= target.unsqueeze(0) > 0
                    rm = torch.zeros_like(m)
                    rm.scatter_(0, order, take_sorted)
                    counter[name] = counter.get(name, 0) + int(rm.sum())
                    return torch.where(rm, torch.zeros_like(flat), flat).reshape(act.shape)
                return fn
            return {n: make(n) for n in layers}

        need = {n: signal == "gate" for n in layers}
        with self._lock:
            culled = self._evaluate_modifiers(batches, forward_fn, f"cull > p{percentile} ({signal})", cull_factory, need)
            cm = self._evaluate_modifiers(batches, forward_fn, "count-matched random", count_factory, need)
            mm = self._evaluate_modifiers(batches, forward_fn, "magnitude-matched random", mag_factory, need)
            gens: Dict[str, str] = {}
            if generation_prompt_ids is not None and generation_tokens > 0:
                for lbl, fac in (("clean", None), ("culled", cull_factory), ("count_matched_random", count_factory),
                                 ("magnitude_matched_random", mag_factory)):
                    ids = self._greedy_generate(generation_prompt_ids, generation_tokens, fac, need)
                    gens[lbl] = decode_fn(ids) if decode_fn else " ".join(map(str, ids))
        return ExtremeCullReport(layers, percentile, signal, culled, cm, mm, gens)

    @torch.no_grad()
    def _greedy_generate(self, prompt_ids: torch.Tensor, n_tokens: int, factory: Optional[ModifierFactory],
                         need_gate: Optional[Dict[str, bool]] = None) -> List[int]:
        ids = prompt_ids.clone()
        out: List[int] = []
        for _ in range(n_tokens):
            hs = HookSet(self.model)
            if factory is not None:
                for name, fn in factory({}).items():
                    hs.add_modifier(name, fn, need_gate=bool(need_gate and need_gate.get(name)))
            with self.clean_forward(), hs.active():
                logits = extract_logits(self.model(ids)).float()
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        return out

    # -------------------------------------------------------------------------
    # 5. Circuits
    # -------------------------------------------------------------------------
    def trace_circuit(self, src_layer: str, dst_layer: str, src_neuron: int, inputs: Any,
                      src_pole: PoleType = PoleType.POSITIVE, forward_fn: Optional[Callable] = None,
                      compute_jacobian: bool = True, top_k: int = 10, attention_mask: Optional[torch.Tensor] = None,
                      position_mode: str = "all_active") -> CircuitGraph:
        with self._lock:
            return self.tracer.trace_circuit(src_layer=src_layer, dst_layer=dst_layer, src_neuron=src_neuron,
                                             inputs=inputs, forward_fn=forward_fn, src_pole=src_pole,
                                             compute_jacobian=compute_jacobian, top_k=top_k,
                                             attention_mask=attention_mask, position_mode=position_mode)

    def trace_incoming_circuit(self, src_layer: str, dst_layer: str, dst_neuron: int, inputs: Any,
                               dst_pole: PoleType = PoleType.POSITIVE, forward_fn: Optional[Callable] = None,
                               top_k: int = 8, attention_mask: Optional[torch.Tensor] = None) -> CircuitGraph:
        with self._lock:
            return self.tracer.trace_incoming_circuit(src_layer=src_layer, dst_layer=dst_layer, dst_neuron=dst_neuron,
                                                      inputs=inputs, forward_fn=forward_fn, dst_pole=dst_pole,
                                                      top_k=top_k, attention_mask=attention_mask)

    def cluster_incoming_attribution_bundles(self, src_layer: str, dst_layer: str, dst_neuron: int,
                                             contexts: List[Any], dst_pole: PoleType = PoleType.POSITIVE,
                                             forward_fn: Optional[Callable] = None, top_k_upstream: int = 24,
                                             max_bundles: int = 6, context_labels: Optional[List[str]] = None,
                                             n_permutations: int = 100, alpha: float = 0.05,
                                             ) -> List[UpstreamAttributionBundle]:
        with self._lock:
            return self.tracer.cluster_incoming_attribution_bundles(
                src_layer=src_layer, dst_layer=dst_layer, dst_neuron=dst_neuron, contexts=contexts,
                dst_pole=dst_pole, forward_fn=forward_fn, top_k_upstream=top_k_upstream,
                max_bundles=max_bundles, context_labels=context_labels, n_permutations=n_permutations, alpha=alpha)

    def audit_residual_stream_sinks(self, residual_site: str, batches: Iterable[Any],
                                    forward_fn: Callable = default_forward,
                                    neuron_layers: Optional[List[str]] = None) -> ResidualSinkReport:
        with self._lock:
            return self.tracer.audit_residual_stream_sinks(residual_site=residual_site, batches=batches,
                                                           forward_fn=forward_fn, neuron_layers=neuron_layers)

    def trace_forward_ad_all_layers(self, src_layer: str, src_neuron: int, inputs: Any,
                                    dst_layers: Optional[List[str]] = None, src_pole: PoleType = PoleType.POSITIVE,
                                    forward_fn: Optional[Callable] = None, position_mode: str = "all_active") -> Dict[str, torch.Tensor]:
        with self._lock:
            return self.tracer.trace_forward_ad_all_layers(src_layer=src_layer, src_neuron=src_neuron, inputs=inputs,
                                                           dst_layers=dst_layers, src_pole=src_pole,
                                                           forward_fn=forward_fn, position_mode=position_mode)

    def verify_combination_readout(self, src_layer: str, src_neurons: List[int], coefficients: Optional[List[float]] = None,
                                   dst_layers: Optional[List[str]] = None, null_samples: int = 1000, seed: int = 0,
                                   causal_inputs: Any = None, forward_fn: Optional[Callable] = None) -> CombinationReadoutReport:
        with self._lock:
            return self.tracer.verify_combination_readout(src_layer=src_layer, src_neurons=src_neurons,
                                                          coefficients=coefficients, dst_layers=dst_layers,
                                                          null_samples=null_samples, seed=seed,
                                                          causal_inputs=causal_inputs, forward_fn=forward_fn)

    def decompose_direct_vs_indirect_effect(self, layer_name: str, neuron_idx: int, inputs: Any, target_token_id: int,
                                            downstream_layer: Optional[str] = None,
                                            rescue_downstream_neurons: Optional[List[int]] = None,
                                            pole: PoleType = PoleType.BOTH, forward_fn: Optional[Callable] = None,
                                            position: int = -1) -> Dict[str, float]:
        with self._lock:
            return self.tracer.decompose_direct_vs_indirect_effect(
                layer_name=layer_name, neuron_idx=neuron_idx, inputs=inputs, target_token_id=target_token_id,
                downstream_layer=downstream_layer, rescue_downstream_neurons=rescue_downstream_neurons,
                pole=pole, forward_fn=forward_fn, position=position)

    # -------------------------------------------------------------------------
    # 6. Explanations
    # -------------------------------------------------------------------------
    def set_explainer_llm_client(self, llm_client: LLMClientProtocol,
                                 test_writer_client: Optional[LLMClientProtocol] = None) -> None:
        with self._lock:
            self.explainer.llm_client = llm_client
            self.explainer.test_writer_client = test_writer_client or llm_client

    def explain_neuron(self, layer_name: str, neuron_idx: int, **kwargs: Any) -> BipolarVerbHypothesis:
        """Explain one neuron and put the explanation at risk; see `NeuronExplainerAgent.explain_and_verify`."""
        kwargs.setdefault("tokenizer", self.tokenizer)
        with self._lock:
            return self.explainer.explain_and_verify(layer_name=layer_name, neuron_idx=neuron_idx, **kwargs)

    def audit_layer_explanations(self, layer_name: str, n_neurons: int, seed: int = 0,
                                 neuron_indices: Optional[List[int]] = None, **kwargs: Any) -> ExplanationAuditSummary:
        """
        Explain-and-test a uniform random sample of neurons from `layer_name` (or the given indices) and
        aggregate the outcomes. Only a random sample gives an unbiased estimate for the layer.
        """
        tr = self._require_tracker(layer_name)
        if neuron_indices is None:
            rng = random.Random(seed)
            neuron_indices = rng.sample(range(tr.num_neurons), k=min(n_neurons, tr.num_neurons))
            sampling = f"uniform_random(seed={seed}, n={len(neuron_indices)})"
        else:
            sampling = "user_selected"
        for i in neuron_indices:
            self.explain_neuron(layer_name, int(i), **kwargs)
        return self.ecology_map.summarize(layer_name, neuron_indices=list(neuron_indices), sampling=sampling)

    def solve_circuit_ecology(self, seed_neurons: List[Tuple[str, int]], neighbor_layers: List[str],
                              circuit_inputs_fn: Callable[[str, int], Any], max_pass2_neurons: int = 8,
                              edges_per_neuron: int = 4, **explain_kwargs: Any) -> LayerEcologySolveReport:
        with self._lock:
            return self.explainer.solve_circuit_ecology(seed_neurons=seed_neurons, neighbor_layers=neighbor_layers,
                                                        circuit_inputs_fn=circuit_inputs_fn,
                                                        max_pass2_neurons=max_pass2_neurons,
                                                        edges_per_neuron=edges_per_neuron, **explain_kwargs)

    # -------------------------------------------------------------------------
    # 7. Layer scan
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def scan_layer(self, layer_name: str, sort_by: str = "rms_write_norm", compute_kl_for_top_n: int = 0,
                   kl_eval_inputs: Optional[Any] = None, limit: Optional[int] = None) -> LayerScanReport:
        """Descriptive per-neuron table of an audited layer plus the aggregate of its explanation audits."""
        with self._lock:
            tr = self._require_tracker(layer_name)
            wv = self.neuron_write_vectors(layer_name)
            rows = tr.scan(wv.float().norm(dim=1) if wv is not None else None)
            for r in rows:
                h = self.ecology_map.get_hypothesis(layer_name, r.neuron_idx)
                if h is not None:
                    r.explained = True
                    r.pos_taxonomy, r.neg_taxonomy = h.pos_taxonomy, h.neg_taxonomy
                    r.pos_outcome, r.neg_outcome = h.outcome(PoleType.POSITIVE), h.outcome(PoleType.NEGATIVE)
            keys = {
                "rms_write_norm": lambda r: (r.rms_write_norm if math.isfinite(r.rms_write_norm) else -1.0),
                "std": lambda r: r.std,
                "max_val": lambda r: r.max_val,
                "min_val": lambda r: -r.min_val,
                "positive_excess_kurtosis": lambda r: r.positive_excess_kurtosis,
                "negative_excess_kurtosis": lambda r: r.negative_excess_kurtosis,
                "double_negative_positive_rate": lambda r: r.double_negative_positive_rate,
            }
            if sort_by == "neuron_idx":
                rows.sort(key=lambda r: r.neuron_idx)
            elif sort_by in keys:
                rows.sort(key=keys[sort_by], reverse=True)
            else:
                raise ValueError(f"Unknown sort_by '{sort_by}'. Options: neuron_idx, {', '.join(keys)}")
            if compute_kl_for_top_n > 0 and kl_eval_inputs is not None:
                for r in rows[:compute_kl_for_top_n]:
                    r.ablation_kl_divergence = self.measure_ablation_kl_divergence(layer_name, r.neuron_idx, kl_eval_inputs)
            if limit is not None:
                rows = rows[:limit]
            return LayerScanReport(
                layer_name=layer_name,
                num_neurons=tr.num_neurons,
                tokens_scanned=tr.count,
                gated=tr.gated,
                neurons=rows,
                layer_double_negative_positive_rate=tr.layer_double_negative_positive_rate(),
                explanation_audit=self.ecology_map.summarize(layer_name),
            )

    # -------------------------------------------------------------------------
    # 8. Contrastive discovery, development, coalitions
    # -------------------------------------------------------------------------
    def discover_contrastive_neurons(self, pos_inputs: Union[torch.Tensor, Iterable[Any]],
                                     neg_inputs: Union[torch.Tensor, Iterable[Any]],
                                     layer_names: Optional[List[str]] = None,
                                     forward_fn: Optional[Callable] = None, top_k: int = 16) -> ContrastiveDiscoveryReport:
        """Welch's t on per-prompt mean activations between two prompt sets (prompts are the units)."""
        layers = layer_names or list(self.list_targetable_layers().keys())
        with self._lock, self.clean_forward():
            return ContrastivePoleFinder.discover(self.model, pos_inputs, neg_inputs, layers,
                                                  forward_fn=forward_fn or default_forward, top_k=top_k,
                                                  device=self.device)

    def inject_contrastive_behavior(self, contrastive_report_or_matches: Union[ContrastiveDiscoveryReport, List[ContrastiveNeuronMatch]],
                                    strength: float, top_k: Optional[int] = None, direction: float = 1.0,
                                    mode: InterventionMode = InterventionMode.ADDITIVE_STEER) -> List[InterventionSpec]:
        """
        Steers the discovered neurons on their own native scale: ADDITIVE_STEER by
        direction * strength * (mean_pos - mean_neg), or HARD_CLAMP to strength * (mean of the chosen set).
        """
        matches = (contrastive_report_or_matches.top_matches
                   if isinstance(contrastive_report_or_matches, ContrastiveDiscoveryReport)
                   else list(contrastive_report_or_matches))
        if top_k is not None:
            matches = matches[:top_k]
        by_layer: Dict[str, List[Tuple[int, float]]] = {}
        for m in matches:
            if mode == InterventionMode.HARD_CLAMP:
                val = (m.pos_mean if direction >= 0 else m.neg_mean) * abs(strength)
            elif mode == InterventionMode.ADDITIVE_STEER:
                val = float(direction) * float(strength) * (m.pos_mean - m.neg_mean)
            else:
                raise ValueError("mode must be ADDITIVE_STEER or HARD_CLAMP")
            by_layer.setdefault(m.layer_name, []).append((m.neuron_idx, float(val)))
        specs = []
        for name, pairs in by_layer.items():
            spec = InterventionSpec(name, [p[0] for p in pairs], mode, PoleType.BOTH, 0.0,
                                    per_neuron_values=[p[1] for p in pairs])
            self.add_intervention(spec)
            specs.append(spec)
        return specs

    def track_checkpoint_evolution(self, checkpoints, layer_name: str, neuron_idx: int, batches, forward_fn=default_forward,
                                   top_k: int = 16) -> NeuronEvolutionReport:
        return CheckpointEvolutionTracker.track_neuron_evolution(checkpoints, layer_name, neuron_idx, batches,
                                                                 forward_fn, top_k=top_k)

    def find_synergistic_ablation_set(self, layer_name: str, inputs: Any, **kwargs: Any) -> SynergisticAblationResult:
        with self._lock:
            return self.ablation_optimizer.find_synergistic_ablation_set(layer_name=layer_name, inputs=inputs, **kwargs)

    # -------------------------------------------------------------------------
    # 9. Export and weight folding
    # -------------------------------------------------------------------------
    def export_production_preset(self, preset_name: str = "steering_preset", model_name: str = "custom-model",
                                 description: str = "", save_json_path: Optional[str] = None,
                                 save_pt_vectors_path: Optional[str] = None,
                                 allowed_workspace_dir: Optional[Union[str, Path]] = None) -> ProductionPresetBundle:
        with self._lock:
            bundle = SteeringExporter.build_preset_bundle(self.model, self._interventions, preset_name, model_name, description)
            if save_json_path is not None:
                ws = allowed_workspace_dir if allowed_workspace_dir is not None else self.allowed_workspace_dir
                SteeringExporter.save_preset_json(bundle, save_json_path, save_pt_vectors_path, ws)
            return bundle

    def load_production_preset(self, preset_data_or_path: Union[str, Dict[str, Any]], replace_existing: bool = True,
                               allowed_workspace_dir: Optional[Union[str, Path]] = None) -> ProductionPresetBundle:
        with self._lock:
            if isinstance(preset_data_or_path, str):
                ws = allowed_workspace_dir if allowed_workspace_dir is not None else self.allowed_workspace_dir
                safe = SteeringExporter.validate_safe_export_path(preset_data_or_path, allowed_workspace_dir=ws)
                raw = json.loads(safe.read_text(encoding="utf-8"))
            else:
                raw = preset_data_or_path
            bundle, specs = SteeringExporter.parse_preset_dict(raw)
            if replace_existing:
                self.clear_interventions()
            for s in specs:
                self.add_intervention(s)
            return bundle

    def fold_interventions_into_weights(self) -> Dict[str, Any]:
        """Folds unconditional affine interventions into weights; unfoldable specs stay as hooks."""
        with self._lock:
            res = SteeringExporter.fold_interventions_into_weights(self.model, self._interventions)
            folded = set(res.get("folded_spec_ids", []))
            if folded:
                was = self._hookset is not None
                self.detach_hooks()
                for name in list(self._interventions):
                    rest = [s for s in self._interventions[name] if id(s) not in folded]
                    if rest:
                        self._interventions[name] = rest
                    else:
                        del self._interventions[name]
                self._latched_masks.clear()
                if was and self._interventions:
                    self.attach_hooks()
            return res
