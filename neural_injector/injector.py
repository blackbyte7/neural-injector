from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .analyzer import LogitLensAttributor, StreamingLayerStats
from .circuit_tracer import CircuitTracer
from .discovery import LayerDescriptor, LayerDiscovery
from .explainer import LLMClientProtocol, NeuronExplainerAgent
from .exporter import SteeringExporter
from .interventions import InterventionKernel
from .types import (
    BipolarVerbHypothesis,
    CircuitGraph,
    InterventionMode,
    InterventionSpec,
    LayerCleanlinessReport,
    NeuronPoleProfile,
    NeuronTaxonomy,
    PoleType,
    ProductionPresetBundle,
)


class NeuralInjector:
    """
    Universal Neural Network Analyzer, Surgical Activation Injector, Circuit Tracer,
    and Automated Agentic Neuron Explainer.

    Implements the algorithms and principles from 'Neural Networks Are Already Clean'
    (Cammarata / Goodfire, Oct 2026):
      - Direct native-basis neuron inspection (Pre-DownProj SwiGLU/MLP hooks, Conv2d, Linear)
      - Bipolar neuron profiling (Positive Pole vs. Negative Pole contexts & selectivity kurtosis)
      - Direct weight-to-vocabulary logit attribution
      - Vectorized surgical interventions (Zero-Ablation, Constant Clamping, Pole Scaling, Weak-Activation Pruning)
      - Layer-to-layer causal & structural circuit tracing (Virtual Weights + Pole Patching + Local Jacobians)
      - Closed-loop LLM-agent bipolar 'verb' explanation and empirical ablation verification
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        llm_client: Optional[LLMClientProtocol] = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device.type
        self.discovered_layers: Dict[str, LayerDescriptor] = LayerDiscovery.discover_neuron_layers(
            model
        )

        # Active surgical interventions keyed by layer_name
        self._interventions: Dict[str, List[InterventionSpec]] = {}
        self._active_handles: List[RemovableHandle] = []

        # Streaming bipolar neuron statistics keyed by layer_name
        self._stats_trackers: Dict[str, StreamingLayerStats] = {}

        # Sub-engines for circuit tracing and automated LLM agent explanation
        self.tracer = CircuitTracer(self)
        self.explainer = NeuronExplainerAgent(self, llm_client=llm_client)

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
        return sum(len(specs) for specs in self._interventions.values())

    @property
    def active_hook_count(self) -> int:
        """Returns the number of currently attached PyTorch forward hooks."""
        return len(self._active_handles)

    def get_active_interventions(self) -> List[InterventionSpec]:
        """Returns a flat list of all registered InterventionSpecs."""
        return [spec for specs in self._interventions.values() for spec in specs]

    # -------------------------------------------------------------------------
    # 2. Surgical Neuron Manipulation (Zero-Ablate, Clamp, Scale, Steer, Prune)
    # -------------------------------------------------------------------------
    def add_intervention(self, spec: InterventionSpec) -> None:
        """Registers a surgical intervention specification on a target layer."""
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
    ) -> InterventionSpec:
        """Forces target neuron(s) to fire at a constant scalar value `value`."""
        indices = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=indices,
            mode=InterventionMode.HARD_CLAMP,
            pole=pole,
            value=value,
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
        )
        self.add_intervention(spec)
        return spec

    def prune_weak_activations(
        self,
        layer_name: str,
        neuron_indices: Union[int, List[int]],
        threshold: float,
    ) -> InterventionSpec:
        """
        Tests Cammarata's 'Do Weak Activations Matter?' hypothesis by zeroing out
        any activation whose magnitude |a| falls below `threshold`, while keeping strong spikes intact.
        """
        indices = [neuron_indices] if isinstance(neuron_indices, int) else list(neuron_indices)
        spec = InterventionSpec(
            layer_name=layer_name,
            neuron_indices=indices,
            mode=InterventionMode.WEAK_PRUNE,
            pole=PoleType.BOTH,
            value=threshold,
        )
        self.add_intervention(spec)
        return spec

    def clear_interventions(self) -> None:
        """Removes all registered interventions and detaches hooks."""
        self.detach_hooks()
        self._interventions.clear()

    def attach_hooks(self) -> None:
        """Attaches PyTorch pre-hooks or forward-hooks for all registered interventions."""
        self.detach_hooks()

        for layer_name, specs in self._interventions.items():
            if not specs:
                continue
            module, hook_target, _ = LayerDiscovery.resolve_module(self.model, layer_name)
            is_conv2d = isinstance(module, nn.Conv2d)

            if hook_target == "input":
                # Pre-forward hook modifies the input tuple to `down_proj` (the post-SwiGLU/GELU neuron basis)
                def _make_pre_hook(layer_specs: List[InterventionSpec], conv_flag: bool):
                    def _pre_hook(_mod: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                        if not inputs or not isinstance(inputs[0], torch.Tensor):
                            return inputs
                        modified = InterventionKernel.apply(
                            inputs[0], layer_specs, is_conv2d=conv_flag
                        )
                        return (modified,) + inputs[1:]

                    return _pre_hook

                handle = module.register_forward_pre_hook(_make_pre_hook(specs, is_conv2d))
            else:
                # Standard forward hook modifies the output of the layer
                def _make_fwd_hook(layer_specs: List[InterventionSpec], conv_flag: bool):
                    def _fwd_hook(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
                        if isinstance(output, torch.Tensor):
                            return InterventionKernel.apply(
                                output, layer_specs, is_conv2d=conv_flag
                            )
                        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                            modified = InterventionKernel.apply(
                                output[0], layer_specs, is_conv2d=conv_flag
                            )
                            return (modified,) + output[1:]
                        return output

                    return _fwd_hook

                handle = module.register_forward_hook(_make_fwd_hook(specs, is_conv2d))

            self._active_handles.append(handle)

    def detach_hooks(self) -> None:
        """Safely removes all active PyTorch hooks."""
        for handle in self._active_handles:
            handle.remove()
        self._active_handles.clear()

    @contextmanager
    def active_injection(self):
        """
        Context manager that activates all registered neuron interventions for the duration
        of a block and guarantees cleanup afterward.
        """
        self.attach_hooks()
        try:
            yield self
        finally:
            self.detach_hooks()

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
        Streams batches through the model and records bipolar activation statistics
        and top-activating/inhibiting contexts for the specified layers.
        """
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
                                contexts_per_sample=current_batch_meta.get("contexts_per_sample"),
                                tracked_neurons=tracked_neurons,
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
                                contexts_per_sample=current_batch_meta.get("contexts_per_sample"),
                                tracked_neurons=tracked_neurons,
                            )

                    return _hook

                handles.append(module.register_forward_hook(_make_audit_fwd_hook(tracker)))

        was_training = self.model.training
        self.model.eval()
        try:
            for batch in dataloader_or_batches:
                current_batch_meta = {
                    "sample_ids": batch.get("sample_ids"),
                    "tokens_per_sample": batch.get("tokens_per_sample"),
                    "contexts_per_sample": batch.get("contexts_per_sample"),
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
                promoted, suppressed = LogitLensAttributor.attribute_neuron_to_vocab(
                    down_proj_module=module,
                    unembed_weight=unembed_weight,
                    neuron_idx=neuron_idx,
                    idx_to_token=idx_to_token,
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
        self.model.eval()
        clean_out = self.model(inputs)
        clean_logits = clean_out.logits if hasattr(clean_out, "logits") else clean_out
        clean_log_probs = torch.log_softmax(clean_logits, dim=-1)
        clean_probs = clean_log_probs.exp()

        self.clear_interventions()
        self.zero_ablate(layer_name=layer_name, neuron_indices=[neuron_idx], pole=pole)

        with self.active_injection():
            ablated_out = self.model(inputs)
            ablated_logits = ablated_out.logits if hasattr(ablated_out, "logits") else ablated_out
            ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)

        self.clear_interventions()

        kl = (clean_probs * (clean_log_probs - ablated_log_probs)).sum(dim=-1).mean().item()
        return float(kl)

    # -------------------------------------------------------------------------
    # 4. Layer-to-Layer Circuit Tracing
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

    # -------------------------------------------------------------------------
    # 5. Automated LLM-Agent Explanation & Closed-Loop Verification
    # -------------------------------------------------------------------------
    def explain_neuron(
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
        Synthesizes a bipolar 'verb' explanation for `(layer_name, neuron_idx)` and runs
        automated surgical Zero-Ablation and Pole-Clamping experiments to empirically verify it.
        """
        return self.explainer.explain_and_verify(
            layer_name=layer_name,
            neuron_idx=neuron_idx,
            tok_to_id=tok_to_id,
            idx_to_token=idx_to_token,
            unembed_weight=unembed_weight,
            dst_layer=dst_layer,
            circuit_inputs=circuit_inputs,
        )

    # -------------------------------------------------------------------------
    # 6. Batch Layer-Wide Cleanliness Scanning
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

        # Sort neurons by requested criterion
        if sort_by == "estimated_causal_impact":
            summaries.sort(key=lambda s: s.estimated_causal_impact, reverse=True)
        elif sort_by == "bipolar_balance":
            summaries.sort(key=lambda s: s.bipolar_balance, reverse=True)
        elif sort_by == "neuron_idx":
            summaries.sort(key=lambda s: s.neuron_idx)
        else:
            summaries.sort(key=lambda s: s.pole_selectivity_score, reverse=True)

        # Optionally compute exact causal zero-ablation KL divergence on top candidates
        if compute_kl_for_top_n > 0 and kl_eval_inputs is not None:
            saved_interventions = {k: list(v) for k, v in self._interventions.items()}
            for summary in summaries[:compute_kl_for_top_n]:
                summary.ablation_kl_divergence = round(
                    self.measure_ablation_kl_divergence(
                        layer_name=layer_name,
                        neuron_idx=summary.neuron_idx,
                        inputs=kl_eval_inputs,
                    ),
                    5,
                )
            self._interventions = saved_interventions

        clean_pct = pcts.get(NeuronTaxonomy.CLEAN_MONOSEMANTIC.value, 0.0)
        struct_pct = pcts.get(NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value, 0.0)
        mean_sel = (
            sum(s.pole_selectivity_score for s in summaries) / max(len(summaries), 1)
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
        )

    # -------------------------------------------------------------------------
    # 7. Production Steering Vector & Preset Export / Weight Folding
    # -------------------------------------------------------------------------
    def export_production_preset(
        self,
        preset_name: str = "production_steering_preset",
        model_name: str = "custom-model",
        description: str = "Verified surgical neuron interventions & compiled residual vectors",
        save_json_path: Optional[str] = None,
        save_pt_vectors_path: Optional[str] = None,
    ) -> ProductionPresetBundle:
        """
        Packages all currently active interventions and compiles their linear residual-stream
        steering vectors v_steer = sum_i delta_i * W_down[:, i] into a ProductionPresetBundle.
        """
        from .exporter import SteeringExporter

        bundle = SteeringExporter.build_preset_bundle(
            model=self.model,
            interventions_by_layer=self._interventions,
            preset_name=preset_name,
            model_name=model_name,
            description=description,
        )
        if save_json_path is not None:
            SteeringExporter.save_preset_json(
                bundle=bundle,
                filepath=save_json_path,
                export_pt_vectors_path=save_pt_vectors_path,
            )
        return bundle

    def load_production_preset(
        self,
        preset_data_or_path: Union[str, Dict[str, Any]],
        replace_existing: bool = True,
    ) -> ProductionPresetBundle:
        """Loads a saved ProductionPresetBundle (from JSON path or dict) into active interventions."""
        from .exporter import SteeringExporter

        if isinstance(preset_data_or_path, str):
            raw = json.loads(Path(preset_data_or_path).read_text(encoding="utf-8"))
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

        res = SteeringExporter.fold_interventions_into_weights(self.model, self._interventions)
        self.clear_interventions()
        return res
