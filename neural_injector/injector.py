from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .analyzer import (
    CheckpointEvolutionTracker,
    ContrastivePoleFinder,
    LogitLensAttributor,
    StreamingLayerStats,
)
from .circuit_tracer import CircuitTracer
from .discovery import LayerDescriptor, LayerDiscovery
from .explainer import CircuitEcologyMap, LLMClientProtocol, NeuronExplainerAgent
from .exporter import SteeringExporter
from .interventions import AblationMaskOptimizer, InterventionKernel
from .types import (
    BipolarVerbHypothesis,
    CircuitGraph,
    ContrastiveDiscoveryReport,
    DownstreamReadoutReport,
    GateConditionSpec,
    InterventionMode,
    InterventionSpec,
    LayerCleanlinessReport,
    LayerEcologySolveReport,
    NeuronEvolutionReport,
    NeuronPoleProfile,
    NeuronTaxonomy,
    PoleType,
    ProductionPresetBundle,
    SynergisticAblationResult,
)


class NeuralInjector:
    """
    Universal Neural Network Analyzer, Surgical Activation Injector, Circuit Tracer,
    and Automated Agentic Neuron Explainer.

    Implements the full suite of algorithms from 'Neural Networks Are Already Clean'
    (Cammarata / Goodfire, Oct 2026) plus production hardening and frontier capabilities:
      - Direct native-basis neuron inspection (Pre-DownProj SwiGLU/MLP hooks, Conv2d, Linear, 3D MoE)
      - Pole-specific conditional moments (M1+, M2+, M4+ and M1-, M2-, M4-) & Stratified Quantile Buckets
      - Crossword / Circuit Ecology propagation store (`CircuitEcologyMap` & `solve_layer_ecology`)
      - Native per-neuron adaptive thresholds (`std_multiplier` / `percentile`)
      - Superposition readout auditing (`verify_downstream_readout`)
      - Single-pass Forward-Mode AD (`torch.autograd.forward_ad`) multi-layer Jacobian tracing
      - Closed-loop dynamic concept gating (`CONDITIONAL_GATE`)
      - Contrastive Pole Discovery (`discover_contrastive_neurons`)
      - Cross-checkpoint Recursive Growth tracking (`track_checkpoint_evolution`)
      - Multi-neuron synergistic group ablation (`find_synergistic_ablation_set`)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        tokenizer: Optional[Any] = None,
        allowed_workspace_dir: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device.type
        self.tokenizer = tokenizer
        self.allowed_workspace_dir = allowed_workspace_dir
        self._lock = threading.RLock()

        self.discovered_layers: Dict[str, LayerDescriptor] = LayerDiscovery.discover_neuron_layers(
            model
        )

        # Active surgical interventions keyed by layer_name
        self._interventions: Dict[str, List[InterventionSpec]] = {}
        self._active_handles: List[RemovableHandle] = []

        # Streaming bipolar neuron statistics keyed by layer_name
        self._stats_trackers: Dict[str, StreamingLayerStats] = {}

        # Persistent Crossword / Circuit Ecology store & sub-engines
        self.ecology_map = CircuitEcologyMap()
        self.tracer = CircuitTracer(self)
        self.explainer = NeuronExplainerAgent(self, llm_client=llm_client)
        self.ablation_optimizer = AblationMaskOptimizer(self)

    # -------------------------------------------------------------------------
    # 1. Layer Discovery & Public State Inspection
    # -------------------------------------------------------------------------
    def list_targetable_layers(self, include_all_linear: bool = False) -> Dict[str, LayerDescriptor]:
        """Returns all targetable neuron layers discovered in the model."""
        if include_all_linear:
            return LayerDiscovery.discover_neuron_layers(self.model, include_all_linear=True)
        return self.discovered_layers

    def is_layer_audited(self, layer_name: str) -> bool:
        """Returns True if streaming statistics have been collected for `layer_name`."""
        return layer_name in self._stats_trackers

    @property
    def active_intervention_count(self) -> int:
        """Returns the total number of registered InterventionSpecs across all layers."""
        with self._lock:
            return sum(len(specs) for specs in self._interventions.values())

    @property
    def active_hook_count(self) -> int:
        """Returns the number of currently attached PyTorch forward hooks."""
        with self._lock:
            return len(self._active_handles)

    def get_active_interventions(self) -> List[InterventionSpec]:
        """Returns a flat list of all registered InterventionSpecs."""
        with self._lock:
            return [spec for specs in self._interventions.values() for spec in specs]

    # -------------------------------------------------------------------------
    # 2. Surgical Neuron Manipulation (Zero-Ablate, Clamp, Steer, Prune, Gate)
    # -------------------------------------------------------------------------
    def add_intervention(self, spec: InterventionSpec) -> None:
        """Registers a surgical intervention specification on a target layer."""
        with self._lock:
            if spec.layer_name not in self._interventions:
                self._interventions[spec.layer_name] = []
            self._interventions[spec.layer_name].append(spec)

    def zero_ablate(
        self,
        layer_name: str,
        neuron_indices: Union[int, List[int]],
        pole: PoleType = PoleType.BOTH,
        token_positions: Optional[List[int]] = None,
    ) -> InterventionSpec:
        """
        Directly sets the specified neuron(s) to 0.0 during forward passes.
        Can target the entire neuron (PoleType.BOTH) or only its Positive/Negative pole.
        """
        indices = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=indices,
            mode=InterventionMode.ZERO_ABLATE,
            pole=pole,
            value=0.0,
            token_positions=token_positions,
        )
        self.add_intervention(spec)
        return spec

    def clamp_neurons(
        self,
        layer_name: str,
        neuron_indices: Union[int, List[int]],
        value: float,
        pole: PoleType = PoleType.BOTH,
        token_positions: Optional[List[int]] = None,
        per_neuron_values: Optional[List[float]] = None,
    ) -> InterventionSpec:
        """Forces target neuron(s) to fire at a constant scalar `value` or `per_neuron_values`."""
        indices = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=indices,
            mode=InterventionMode.HARD_CLAMP,
            pole=pole,
            value=value,
            per_neuron_values=per_neuron_values,
            token_positions=token_positions,
        )
        self.add_intervention(spec)
        return spec

    def steer_neurons(
        self,
        layer_name: str,
        neuron_indices: Union[int, List[int]],
        delta: Optional[float] = None,
        scale: Optional[float] = None,
        pole: PoleType = PoleType.BOTH,
        per_neuron_values: Optional[List[float]] = None,
    ) -> InterventionSpec:
        """Amplifies, dampens, or shifts target neuron activations toward a desired pole."""
        indices = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        if scale is not None:
            mode = InterventionMode.SCALE
            val = scale
        else:
            mode = InterventionMode.ADDITIVE_STEER
            val = delta if delta is not None else 1.0

        spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=indices,
            mode=mode,
            pole=pole,
            value=val,
            per_neuron_values=per_neuron_values,
        )
        self.add_intervention(spec)
        return spec

    def prune_weak_activations(
        self,
        layer_name: str,
        neuron_indices: Union[int, List[int]],
        threshold: Optional[Union[float, List[float]]] = None,
        std_multiplier: Optional[float] = None,
        percentile: Optional[float] = None,
    ) -> InterventionSpec:
        """
        Tests Cammarata's 'Do Weak Activations Matter?' hypothesis by zeroing out
        activations below a scalar `threshold` OR native per-neuron adaptive thresholds
        computed from each neuron's audited distribution via `std_multiplier` or `percentile`
        (Section 3.4: Respecting Native Activation Ranges & Percentiles).
        """
        indices = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        per_neuron_vals: Optional[List[float]] = None

        if isinstance(threshold, list):
            per_neuron_vals = [float(x) for x in threshold]
            scalar_val = per_neuron_vals[0] if per_neuron_vals else 0.0
        elif threshold is not None and std_multiplier is None and percentile is None:
            scalar_val = float(threshold)
        elif self.is_layer_audited(layer_name):
            tracker = self._stats_trackers[layer_name]
            per_neuron_vals = tracker.compute_adaptive_thresholds(
                indices, std_multiplier=std_multiplier, percentile=percentile
            )
            scalar_val = per_neuron_vals[0] if per_neuron_vals else 0.0
        else:
            scalar_val = float(threshold) if threshold is not None else 1.0

        spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=indices,
            mode=InterventionMode.WEAK_PRUNE,
            pole=PoleType.BOTH,
            value=scalar_val,
            per_neuron_values=per_neuron_vals,
        )
        self.add_intervention(spec)
        return spec

    def add_conditional_gate(
        self,
        target_layer: str,
        target_neurons: Union[int, List[int]],
        detector_layer: str,
        detector_neuron: int,
        detector_pole: PoleType = PoleType.POSITIVE,
        threshold: Optional[float] = None,
        detector_percentile: Optional[float] = None,
        action_mode: InterventionMode = InterventionMode.ADDITIVE_STEER,
        value: float = 0.0,
        target_pole: PoleType = PoleType.BOTH,
        sequence_scope: str = "token_wise",
    ) -> InterventionSpec:
        """
        Registers a Closed-Loop Dynamic Concept Gate (`CONDITIONAL_GATE`, Section 5.2):
        Only applies `action_mode` (with `value`) to `target_neurons` in `target_layer`
        when `detector_neuron` in `detector_layer` fires on `detector_pole` above `threshold`
        (or above its audited `detector_percentile`).
        """
        indices = [target_neurons] if isinstance(target_neurons, int) else list(target_neurons)
        if threshold is not None:
            resolved_thresh = float(threshold)
        elif detector_percentile is not None and self.is_layer_audited(detector_layer):
            resolved_thresh = self._stats_trackers[detector_layer].compute_adaptive_thresholds(
                [detector_neuron], percentile=detector_percentile
            )[0]
        elif self.is_layer_audited(detector_layer):
            resolved_thresh = self._stats_trackers[detector_layer].compute_adaptive_thresholds(
                [detector_neuron], std_multiplier=1.0
            )[0]
        else:
            resolved_thresh = 0.5

        cond = GateConditionSpec(
            detector_layer=detector_layer,
            detector_neuron=detector_neuron,
            detector_pole=detector_pole,
            threshold=resolved_thresh,
            action_mode=action_mode,
            sequence_scope=sequence_scope,
        )
        spec = InterventionSpec(
            layer_name=target_layer,
            neuron_indices=indices,
            mode=InterventionMode.CONDITIONAL_GATE,
            pole=target_pole,
            value=value,
            gate_condition=cond,
        )
        self.add_intervention(spec)
        return spec

    def clear_interventions(self) -> None:
        """Removes all registered interventions and detaches hooks."""
        with self._lock:
            self.detach_hooks()
            self._interventions.clear()

    def attach_hooks(self) -> None:
        """
        Attaches PyTorch pre-hooks or forward-hooks for all registered interventions,
        including upstream detector hooks for any cross-layer `CONDITIONAL_GATE` specs.
        """
        with self._lock:
            self.detach_hooks()
            conditional_masks: Dict[int, torch.Tensor] = {}

            # 1. Register detector hooks first so upstream detector masks are populated before target hooks fire
            for layer_name, specs in self._interventions.items():
                for spec in specs:
                    if not spec.enabled or spec.gate_condition is None:
                        continue
                    det_layer = spec.gate_condition.detector_layer
                    if det_layer == layer_name:
                        # Handled inline inside InterventionKernel.apply
                        continue
                    det_mod, det_target, _ = LayerDiscovery.resolve_module(self.model, det_layer)
                    is_det_conv = isinstance(det_mod, nn.Conv2d)

                    def _make_det_pre(s_obj: InterventionSpec, conv_f: bool):
                        def _det_pre(_mod: nn.Module, inputs: Tuple[Any, ...]) -> None:
                            if inputs and isinstance(inputs[0], torch.Tensor) and s_obj.gate_condition:
                                conditional_masks[id(s_obj)] = (
                                    InterventionKernel.compute_detector_mask(
                                        inputs[0].detach(), s_obj.gate_condition, is_conv2d=conv_f
                                    )
                                )

                        return _det_pre

                    def _make_det_fwd(s_obj: InterventionSpec, conv_f: bool):
                        def _det_fwd(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                            act = output[0] if isinstance(output, tuple) else output
                            if isinstance(act, torch.Tensor) and s_obj.gate_condition:
                                conditional_masks[id(s_obj)] = (
                                    InterventionKernel.compute_detector_mask(
                                        act.detach(), s_obj.gate_condition, is_conv2d=conv_f
                                    )
                                )

                        return _det_fwd

                    if det_target == "input":
                        self._active_handles.append(
                            det_mod.register_forward_pre_hook(_make_det_pre(spec, is_det_conv))
                        )
                    else:
                        self._active_handles.append(
                            det_mod.register_forward_hook(_make_det_fwd(spec, is_det_conv))
                        )

            # 2. Register target intervention hooks
            for layer_name, specs in self._interventions.items():
                if not specs:
                    continue
                module, hook_target, _ = LayerDiscovery.resolve_module(self.model, layer_name)
                is_conv2d = isinstance(module, nn.Conv2d)

                if hook_target == "input":

                    def _make_pre_hook(layer_specs: List[InterventionSpec], conv_flag: bool):
                        def _pre_hook(_mod: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                            if not inputs or not isinstance(inputs[0], torch.Tensor):
                                return inputs
                            modified = InterventionKernel.apply(
                                inputs[0],
                                layer_specs,
                                is_conv2d=conv_flag,
                                conditional_masks=conditional_masks,
                            )
                            return (modified,) + inputs[1:]

                        return _pre_hook

                    handle = module.register_forward_pre_hook(_make_pre_hook(specs, is_conv2d))
                else:

                    def _make_fwd_hook(layer_specs: List[InterventionSpec], conv_flag: bool):
                        def _fwd_hook(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
                            if isinstance(output, torch.Tensor):
                                return InterventionKernel.apply(
                                    output,
                                    layer_specs,
                                    is_conv2d=conv_flag,
                                    conditional_masks=conditional_masks,
                                )
                            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                                modified = InterventionKernel.apply(
                                    output[0],
                                    layer_specs,
                                    is_conv2d=conv_flag,
                                    conditional_masks=conditional_masks,
                                )
                                return (modified,) + output[1:]
                            return output

                        return _fwd_hook

                    handle = module.register_forward_hook(_make_fwd_hook(specs, is_conv2d))

                self._active_handles.append(handle)

    def detach_hooks(self) -> None:
        """Safely removes all active PyTorch hooks."""
        with self._lock:
            for handle in self._active_handles:
                handle.remove()
            self._active_handles.clear()

    @contextmanager
    def active_injection(self):
        """
        Thread-safe context manager that activates all registered neuron interventions
        for the duration of a block and guarantees cleanup afterward.
        """
        with self._lock:
            self.attach_hooks()
            try:
                yield self
            finally:
                self.detach_hooks()

    @contextmanager
    def without_interventions(self):
        """
        Non-destructive, thread-safe context manager that temporarily detaches any active hooks
        for a clean forward pass, then restores the previous hook state upon exit
        without clearing `self._interventions`.
        """
        with self._lock:
            was_attached = len(self._active_handles) > 0
            if was_attached:
                self.detach_hooks()
            try:
                yield self
            finally:
                if was_attached:
                    self.attach_hooks()

    @contextmanager
    def temporary_interventions(self, temp_specs: Optional[List[InterventionSpec]] = None):
        """
        Non-destructive, thread-safe context manager that temporarily replaces active interventions
        with `temp_specs`, attaches their hooks for the duration of the block, and
        restores the exact prior `self._interventions` and hook state upon exit.
        """
        with self._lock:
            saved_interventions = {k: list(v) for k, v in self._interventions.items()}
            was_attached = len(self._active_handles) > 0
            self.detach_hooks()
            self._interventions = {}
            for spec in temp_specs or []:
                self.add_intervention(spec)
            if temp_specs:
                self.attach_hooks()
            try:
                yield self
            finally:
                self.detach_hooks()
                self._interventions = saved_interventions
                if was_attached:
                    self.attach_hooks()

    @contextmanager
    def clean_forward(self):
        """
        Non-destructive context manager that guarantees zero active intervention hooks during
        a baseline forward pass while preserving any user-registered interventions afterward.
        """
        with self.temporary_interventions([]):
            yield self

    # -------------------------------------------------------------------------
    # 3. Bipolar Neuron Auditing & Concept Discovery
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def audit_layers(
        self,
        layer_names: List[str],
        dataloader_or_batches: Iterable[Dict[str, Any]],
        forward_fn: Callable[[nn.Module, Dict[str, Any]], Any],
        top_k: int = 8,
        tracked_neurons: Optional[List[int]] = None,
    ) -> Dict[str, StreamingLayerStats]:
        """
        Streams batches through the model and records bipolar activation statistics,
        pole-specific conditional moments, and quantile-stratified activating contexts.
        """
        with self._lock:
            handles: List[RemovableHandle] = []
            current_batch_meta: Dict[str, Any] = {}

            for layer_name in layer_names:
                module, hook_target, num_neurons = LayerDiscovery.resolve_module(
                    self.model, layer_name
                )
                tracker = StreamingLayerStats(
                    layer_name=layer_name,
                    num_neurons=num_neurons,
                    top_k=top_k,
                    device=self.device,
                )
                self._stats_trackers[layer_name] = tracker

                if hook_target == "input":

                    def _make_audit_pre_hook(t: StreamingLayerStats):
                        def _hook(_mod: nn.Module, inputs: Tuple[Any, ...]) -> None:
                            if inputs and isinstance(inputs[0], torch.Tensor):
                                t.update(
                                    inputs[0].detach(),
                                    sample_ids=current_batch_meta.get("sample_ids"),
                                    tokens_per_sample=current_batch_meta.get("tokens_per_sample"),
                                    contexts_per_sample=current_batch_meta.get(
                                        "contexts_per_sample"
                                    ),
                                    tracked_neurons=tracked_neurons,
                                    attention_mask=current_batch_meta.get("attention_mask"),
                                    routed_token_indices=current_batch_meta.get(
                                        "routed_token_indices"
                                    ),
                                )

                        return _hook

                    handles.append(module.register_forward_pre_hook(_make_audit_pre_hook(tracker)))
                else:

                    def _make_audit_fwd_hook(t: StreamingLayerStats):
                        def _hook(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                            act = output[0] if isinstance(output, tuple) else output
                            if isinstance(act, torch.Tensor):
                                t.update(
                                    act.detach(),
                                    sample_ids=current_batch_meta.get("sample_ids"),
                                    tokens_per_sample=current_batch_meta.get("tokens_per_sample"),
                                    contexts_per_sample=current_batch_meta.get(
                                        "contexts_per_sample"
                                    ),
                                    tracked_neurons=tracked_neurons,
                                    attention_mask=current_batch_meta.get("attention_mask"),
                                    routed_token_indices=current_batch_meta.get(
                                        "routed_token_indices"
                                    ),
                                )

                        return _hook

                    handles.append(module.register_forward_hook(_make_audit_fwd_hook(tracker)))

            was_training = self.model.training
            self.model.eval()
            try:
                with self.without_interventions():
                    for batch in dataloader_or_batches:
                        current_batch_meta = {
                            "sample_ids": batch.get("sample_ids"),
                            "tokens_per_sample": batch.get("tokens_per_sample"),
                            "contexts_per_sample": batch.get("contexts_per_sample"),
                            "attention_mask": batch.get("attention_mask"),
                            "routed_token_indices": batch.get("routed_token_indices"),
                        }
                        forward_fn(self.model, batch)
            finally:
                for h in handles:
                    h.remove()
                if was_training:
                    self.model.train()

            return self._stats_trackers

    def get_neuron_profile(
        self,
        layer_name: str,
        neuron_idx: int,
        unembed_weight: Optional[torch.Tensor] = None,
        idx_to_token: Optional[Dict[int, str]] = None,
    ) -> NeuronPoleProfile:
        """
        Exports a complete NeuronPoleProfile for `neuron_idx` in `layer_name`,
        optionally enriched with Direct Weight-to-Logit vocabulary attributions.
        """
        if layer_name not in self._stats_trackers:
            raise KeyError(
                f"Layer '{layer_name}' has not been audited yet. Call `audit_layers` first."
            )

        profile = self._stats_trackers[layer_name].export_profile(neuron_idx)

        if unembed_weight is not None:
            module, _, _ = LayerDiscovery.resolve_module(self.model, layer_name)
            if hasattr(module, "weight"):
                final_norm = LayerDiscovery.resolve_final_norm(self.model)
                promoted, suppressed = LogitLensAttributor.attribute_neuron_to_vocab(
                    down_proj_module=module,
                    unembed_weight=unembed_weight,
                    neuron_idx=neuron_idx,
                    idx_to_token=idx_to_token,
                    final_norm_module=final_norm,
                )
                profile.top_promoted_tokens = promoted
                profile.top_suppressed_tokens = suppressed

        return profile

    @torch.no_grad()
    def measure_ablation_kl_divergence(
        self,
        layer_name: str,
        neuron_idx: int,
        inputs: torch.Tensor,
        pole: PoleType = PoleType.BOTH,
    ) -> float:
        """
        Measures the exact causal impact of zero-ablating a single neuron (or one of its poles)
        by computing the KL-Divergence D_KL(P_clean || P_ablated) on the model's output logits.
        """
        with self._lock:
            self.model.eval()
            with self.without_interventions():
                clean_out = self.model(inputs)
                clean_logits = clean_out.logits if hasattr(clean_out, "logits") else clean_out
                clean_log_probs = torch.log_softmax(clean_logits.float(), dim=-1)
                clean_probs = clean_log_probs.exp()

            ablate_spec = InterventionSpec(
                layer_name=layer_name,
                neuron_indices=[neuron_idx],
                mode=InterventionMode.ZERO_ABLATE,
                pole=pole,
                value=0.0,
            )
            with self.temporary_interventions([ablate_spec]):
                ablated_out = self.model(inputs)
                ablated_logits = (
                    ablated_out.logits if hasattr(ablated_out, "logits") else ablated_out
                )
                ablated_log_probs = torch.log_softmax(ablated_logits.float(), dim=-1)

            kl = (clean_probs * (clean_log_probs - ablated_log_probs)).sum(dim=-1).mean().item()
            return float(kl)

    # -------------------------------------------------------------------------
    # 4. Layer-to-Layer Circuit Tracing & Superposition Readout Verification
    # -------------------------------------------------------------------------
    def trace_circuit(
        self,
        src_layer: str,
        dst_layer: str,
        src_neuron: int,
        inputs: Any,
        src_pole: PoleType = PoleType.POSITIVE,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        compute_jacobian: bool = True,
        top_k: int = 10,
    ) -> CircuitGraph:
        """
        Traces how `(src_layer, src_neuron, src_pole)` excites, inhibits, or gates
        downstream neurons in `dst_layer`.
        """
        with self._lock:
            return self.tracer.trace_circuit(
                src_layer=src_layer,
                dst_layer=dst_layer,
                src_neuron=src_neuron,
                inputs=inputs,
                forward_fn=forward_fn,
                src_pole=src_pole,
                compute_jacobian=compute_jacobian,
                top_k=top_k,
            )

    def trace_forward_ad_all_layers(
        self,
        src_layer: str,
        src_neuron: int,
        inputs: Any,
        dst_layers: Optional[List[str]] = None,
        src_pole: PoleType = PoleType.POSITIVE,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-pass Forward-Mode AD (`torch.autograd.forward_ad`) circuit tracer (Section 5.1):
        Returns a dictionary mapping every downstream layer name to its exact 1D Jacobian
        sensitivity tensor `d(A_dst)/d(a_{src_layer, src_neuron})` in a single forward pass.
        """
        with self._lock:
            return self.tracer.trace_forward_ad_all_layers(
                src_layer=src_layer,
                src_neuron=src_neuron,
                inputs=inputs,
                dst_layers=dst_layers,
                src_pole=src_pole,
                forward_fn=forward_fn,
            )

    def verify_downstream_readout(
        self,
        src_layer: str,
        dst_layer: str,
        src_neurons: List[int],
        inputs: Any,
        weak_threshold: Optional[Union[float, List[float]]] = None,
        std_multiplier: Optional[float] = 0.75,
        percentile: Optional[float] = None,
        readout_tolerance: float = 0.05,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    ) -> DownstreamReadoutReport:
        """
        Tests Cammarata's 'Nothing Reads That Combination' hypothesis (Section 3.5)
        across `src_neurons` in `src_layer` with respect to `dst_layer`.
        """
        with self._lock:
            return self.tracer.verify_downstream_readout(
                src_layer=src_layer,
                dst_layer=dst_layer,
                src_neurons=src_neurons,
                inputs=inputs,
                weak_threshold=weak_threshold,
                std_multiplier=std_multiplier,
                percentile=percentile,
                readout_tolerance=readout_tolerance,
                forward_fn=forward_fn,
            )

    # -------------------------------------------------------------------------
    # 5. Automated LLM-Agent Explanation & Crossword Ecology Solver
    # -------------------------------------------------------------------------
    def explain_neuron(
        self,
        layer_name: str,
        neuron_idx: int,
        tok_to_id: Optional[Dict[str, int]] = None,
        idx_to_token: Optional[Dict[int, str]] = None,
        unembed_weight: Optional[torch.Tensor] = None,
        dst_layer: Optional[str] = None,
        circuit_inputs: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> BipolarVerbHypothesis:
        """
        Synthesizes a bipolar 'verb' explanation for `(layer_name, neuron_idx)` and runs
        automated surgical Zero-Ablation and Pole-Clamping experiments to empirically verify it.
        """
        with self._lock:
            return self.explainer.explain_and_verify(
                layer_name=layer_name,
                neuron_idx=neuron_idx,
                tok_to_id=tok_to_id or {},
                idx_to_token=idx_to_token or {},
                unembed_weight=unembed_weight,
                dst_layer=dst_layer,
                circuit_inputs=circuit_inputs,
                tokenizer=tokenizer or self.tokenizer,
            )

    def solve_layer_ecology(
        self,
        layer_name: str,
        tok_to_id: Optional[Dict[str, int]] = None,
        idx_to_token: Optional[Dict[int, str]] = None,
        unembed_weight: Optional[torch.Tensor] = None,
        dst_layer: Optional[str] = None,
        circuit_inputs: Optional[Any] = None,
        top_anchors: int = 3,
        top_contextual: int = 3,
        tokenizer: Optional[Any] = None,
    ) -> LayerEcologySolveReport:
        """
        Implements Cammarata's Crossword / Circuit Ecology propagation loop (Section 3.2:
        'Understanding Compounds'):
          - Pass 1 (Easy Crossword Clues): Ranks audited neurons by selectivity/impact and
            explains + verifies the top `top_anchors` monosemantic anchor neurons, storing
            their verified hypotheses and circuit edges in `self.ecology_map`.
          - Pass 2 (Compound Understanding): Explains the next `top_contextual` structured
            recursive mixture / harder neurons while injecting the verified neighbor motifs
            from `self.ecology_map` into the LLM prompt.
        """
        with self._lock:
            scan = self.scan_layer_cleanliness(
                layer_name=layer_name, sort_by="estimated_causal_impact"
            )
            anchors = [
                n.neuron_idx
                for n in scan.neurons
                if n.taxonomy_class == NeuronTaxonomy.CLEAN_MONOSEMANTIC
            ][:top_anchors]
            if len(anchors) < top_anchors:
                for n in scan.neurons:
                    if n.neuron_idx not in anchors:
                        anchors.append(n.neuron_idx)
                    if len(anchors) >= top_anchors:
                        break

            contextual = [
                n.neuron_idx
                for n in scan.neurons
                if n.neuron_idx not in anchors
                and n.taxonomy_class != NeuronTaxonomy.DEAD_OR_NOISY
            ][:top_contextual]

            solved: Dict[int, BipolarVerbHypothesis] = {}
            scores: List[float] = []

            # Pass 1: Solve anchor neurons
            for n_idx in anchors:
                hyp = self.explain_neuron(
                    layer_name=layer_name,
                    neuron_idx=n_idx,
                    tok_to_id=tok_to_id,
                    idx_to_token=idx_to_token,
                    unembed_weight=unembed_weight,
                    dst_layer=dst_layer,
                    circuit_inputs=circuit_inputs,
                    tokenizer=tokenizer,
                )
                solved[n_idx] = hyp
                if hyp.verification_report is not None:
                    scores.append(hyp.verification_report.empirical_verification_score)

            # Pass 2: Solve contextual neurons with ecology neighbor clues populated
            for n_idx in contextual:
                hyp = self.explain_neuron(
                    layer_name=layer_name,
                    neuron_idx=n_idx,
                    tok_to_id=tok_to_id,
                    idx_to_token=idx_to_token,
                    unembed_weight=unembed_weight,
                    dst_layer=dst_layer,
                    circuit_inputs=circuit_inputs,
                    tokenizer=tokenizer,
                )
                solved[n_idx] = hyp
                if hyp.verification_report is not None:
                    scores.append(hyp.verification_report.empirical_verification_score)

            mean_score = sum(scores) / max(len(scores), 1) if scores else 0.0
            return LayerEcologySolveReport(
                layer_name=layer_name,
                anchor_neurons_explained=anchors,
                contextual_neurons_explained=contextual,
                hypotheses=solved,
                mean_verification_score=round(mean_score, 3),
            )

    # -------------------------------------------------------------------------
    # 6. Batch Layer-Wide & Pole-Specific Cleanliness Scanning
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def scan_layer_cleanliness(
            self,
            layer_name: str,
            sort_by: str = "pole_selectivity_score",
            compute_kl_for_top_n: int = 0,
            kl_eval_inputs: Optional[torch.Tensor] = None,
            limit: Optional[int] = None,
    ) -> LayerCleanlinessReport:
        """
        Scans all neurons in `layer_name` in parallel using GPU Welford moments and
        outgoing weight norms, returning the exact cleanliness breakdown (Clean Monosemantic %,
        Structured Recursive Mixture %, Polysemantic %, Dead/Noisy %) and ranked neuron list.

        Optionally computes exact causal ablation KL-divergence for the top `compute_kl_for_top_n`
        neurons when `kl_eval_inputs` is provided.
        """
        if layer_name not in self._stats_trackers:
            raise KeyError(
                f"Layer '{layer_name}' has not been audited yet. Call `audit_layers` first."
            )

        tracker = self._stats_trackers[layer_name]
        module, _, _ = LayerDiscovery.resolve_module(self.model, layer_name)

        counts, pcts, summaries = tracker.scan_layer_cleanliness(down_proj_module=module)

        # Aggregate independent pole-level taxonomy counts across all 2 * num_neurons poles (Section 3.1)
        pole_counts: Dict[str, int] = {
            NeuronTaxonomy.CLEAN_MONOSEMANTIC.value: 0,
            NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value: 0,
            NeuronTaxonomy.TRUE_POLYSEMANTIC.value: 0,
            NeuronTaxonomy.DEAD_OR_NOISY.value: 0,
        }
        for summary_item in summaries:
            pole_counts[summary_item.pos_taxonomy_class.value] += 1
            pole_counts[summary_item.neg_taxonomy_class.value] += 1

        total_poles = max(2 * tracker.num_neurons, 1)
        pole_pcts = {k: round((v / total_poles) * 100.0, 2) for k, v in pole_counts.items()}
        clean_poles_pct = pole_pcts.get(NeuronTaxonomy.CLEAN_MONOSEMANTIC.value, 0.0)
        struct_poles_pct = pole_pcts.get(NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value, 0.0)

        if sort_by == "estimated_causal_impact":
            summaries.sort(key=lambda item: item.estimated_causal_impact, reverse=True)
        elif sort_by == "bipolar_balance":
            summaries.sort(key=lambda item: item.bipolar_balance, reverse=True)
        elif sort_by == "neuron_idx":
            summaries.sort(key=lambda item: item.neuron_idx)
        else:
            summaries.sort(key=lambda item: item.pole_selectivity_score, reverse=True)

        if compute_kl_for_top_n > 0 and kl_eval_inputs is not None:
            for summary in summaries[:compute_kl_for_top_n]:
                summary.ablation_kl_divergence = round(
                    self.measure_ablation_kl_divergence(
                        layer_name=layer_name,
                        neuron_idx=summary.neuron_idx,
                        inputs=kl_eval_inputs,
                    ),
                    5,
                )

        clean_pct = pcts.get(NeuronTaxonomy.CLEAN_MONOSEMANTIC.value, 0.0)
        struct_pct = pcts.get(NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value, 0.0)
        mean_sel = (
            sum(item.pole_selectivity_score for item in summaries) / max(len(summaries), 1)
            if summaries
            else 0.0
        )

        if limit is not None:
            summaries = summaries[:limit]

        return LayerCleanlinessReport(
            layer_name=layer_name,
            num_neurons=tracker.num_neurons,
            tokens_scanned=tracker.count,
            taxonomy_counts=counts,
            taxonomy_percentages=pcts,
            clean_monosemantic_pct=clean_pct,
            structured_recursive_pct=struct_pct,
            combined_clean_and_structured_pct=round(clean_pct + struct_pct, 2),
            mean_selectivity_score=round(mean_sel, 3),
            neurons=summaries,
            pole_taxonomy_counts=pole_counts,
            pole_taxonomy_percentages=pole_pcts,
            clean_poles_pct=clean_poles_pct,
            structured_poles_pct=struct_poles_pct,
            combined_clean_and_structured_poles_pct=round(clean_poles_pct + struct_poles_pct, 2),
        )

    # -------------------------------------------------------------------------
    # 7. Contrastive Discovery, Checkpoint Evolution & Synergistic Ablation
    # -------------------------------------------------------------------------
    def discover_contrastive_neurons(
        self,
        pos_inputs: Union[torch.Tensor, Iterable[Any]],
        neg_inputs: Union[torch.Tensor, Iterable[Any]],
        layer_names: Optional[List[str]] = None,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        top_k: int = 16,
    ) -> ContrastiveDiscoveryReport:
        """
        Contrastive Pole Discovery (Section 5.3):
        Ranks all native neurons across `layer_names` by their Welch's Bipolar
        Separation t-statistic between `pos_inputs` (D+) and `neg_inputs` (D-).
        """
        target_layers = layer_names or list(self.list_targetable_layers().keys())
        with self._lock, self.without_interventions():
            return ContrastivePoleFinder.discover(
                model=self.model,
                pos_inputs=pos_inputs,
                neg_inputs=neg_inputs,
                layer_names=target_layers,
                forward_fn=forward_fn,
                top_k=top_k,
                device=self.device,
            )

    def track_checkpoint_evolution(
        self,
        checkpoints: List[Tuple[str, nn.Module]],
        layer_name: str,
        neuron_idx: int,
        batches: Iterable[Dict[str, Any]],
        forward_fn: Callable[[nn.Module, Dict[str, Any]], Any],
        top_k: int = 8,
    ) -> NeuronEvolutionReport:
        """
        Cross-Checkpoint 'Recursive Growth' Tracker (Section 5.4):
        Measures how `(layer_name, neuron_idx)` evolves its pole selectivity, outgoing
        weight vector, and stratified quantile offshoots across `checkpoints`.
        """
        return CheckpointEvolutionTracker.track_neuron_evolution(
            checkpoints=checkpoints,
            layer_name=layer_name,
            neuron_idx=neuron_idx,
            batches=batches,
            forward_fn=forward_fn,
            top_k=top_k,
        )

    def find_synergistic_ablation_set(
        self,
        layer_name: str,
        inputs: Any,
        max_subset_size: int = 3,
        candidate_pool_size: int = 16,
        method: str = "greedy",
        target_token_ids: Optional[List[int]] = None,
        pole: PoleType = PoleType.BOTH,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    ) -> SynergisticAblationResult:
        """
        Subspace / Multi-Neuron Synergistic Ablation (Section 5.5):
        Finds a minimal coalition of `max_subset_size` neurons in `layer_name` whose
        joint ablation overcomes backup circuits to maximize output shift.
        """
        with self._lock:
            return self.ablation_optimizer.find_synergistic_ablation_set(
                layer_name=layer_name,
                inputs=inputs,
                max_subset_size=max_subset_size,
                candidate_pool_size=candidate_pool_size,
                method=method,
                target_token_ids=target_token_ids,
                pole=pole,
                forward_fn=forward_fn,
            )

    # -------------------------------------------------------------------------
    # 8. Production Steering Vector & Preset Export / Weight Folding
    # -------------------------------------------------------------------------
    def export_production_preset(
        self,
        preset_name: str = "production_steering_preset",
        model_name: str = "custom-model",
        description: str = "Verified surgical neuron interventions & compiled residual vectors",
        save_json_path: Optional[str] = None,
        save_pt_vectors_path: Optional[str] = None,
        allowed_workspace_dir: Optional[Union[str, Path]] = None,
    ) -> ProductionPresetBundle:
        """
        Packages all currently active interventions and compiles their linear residual-stream
        steering vectors v_steer = sum_i delta_i * W_down[:, i] into a ProductionPresetBundle.
        """
        with self._lock:
            bundle = SteeringExporter.build_preset_bundle(
                model=self.model,
                interventions_by_layer=self._interventions,
                preset_name=preset_name,
                model_name=model_name,
                description=description,
            )
            if save_json_path is not None:
                workspace = (
                    allowed_workspace_dir
                    if allowed_workspace_dir is not None
                    else self.allowed_workspace_dir
                )
                SteeringExporter.save_preset_json(
                    bundle=bundle,
                    filepath=save_json_path,
                    export_pt_vectors_path=save_pt_vectors_path,
                    allowed_workspace_dir=workspace,
                )
            return bundle

    def load_production_preset(
        self,
        preset_data_or_path: Union[str, Dict[str, Any]],
        replace_existing: bool = True,
        allowed_workspace_dir: Optional[Union[str, Path]] = None,
    ) -> ProductionPresetBundle:
        """Loads a saved ProductionPresetBundle (from JSON path or dict) into active interventions."""
        with self._lock:
            if isinstance(preset_data_or_path, str):
                workspace = (
                    allowed_workspace_dir
                    if allowed_workspace_dir is not None
                    else self.allowed_workspace_dir
                )
                safe_path = SteeringExporter.validate_safe_export_path(
                    preset_data_or_path, allowed_workspace_dir=workspace
                )
                raw = json.loads(safe_path.read_text(encoding="utf-8"))
            else:
                raw = preset_data_or_path

            bundle, specs = SteeringExporter.parse_preset_dict(raw)
            if replace_existing:
                self.clear_interventions()
            for spec in specs:
                self.add_intervention(spec)
            return bundle

    def fold_interventions_into_weights(self) -> Dict[str, Any]:
        """
        Permanently folds unconditional ZERO_ABLATE, SCALE, ADDITIVE_STEER, and HARD_CLAMP
        interventions directly into `down_proj` weights and biases for 0% hook latency in production.
        """
        with self._lock:
            res = SteeringExporter.fold_interventions_into_weights(
                self.model, self._interventions
            )
            self.clear_interventions()
            return res
