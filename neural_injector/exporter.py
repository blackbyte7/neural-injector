from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .discovery import LayerDiscovery
from .interventions import InterventionKernel
from .types import (
    CompiledSteeringVector,
    GateConditionSpec,
    InterventionMode,
    InterventionSpec,
    PoleType,
    ProductionPresetBundle,
)


class SteeringExporter:
    """
    Compiles surgical neuron manipulations into residual-stream steering vectors,
    exports/imports portable JSON production presets (with path-traversal security),
    and folds unconditional interventions directly into model weights for zero-overhead inference.
    """

    @staticmethod
    def validate_safe_export_path(
        filepath: Union[str, Path],
        allowed_workspace_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Validates a file path against path-traversal attacks (`..` segments or escaping
        `allowed_workspace_dir` when configured). Raises `PermissionError` if unsafe.
        """
        raw_path = Path(filepath)
        if ".." in raw_path.parts:
            raise PermissionError(
                f"Path traversal ('..') is not permitted in export path: '{filepath}'."
            )
        resolved = raw_path.expanduser().resolve()
        if allowed_workspace_dir is not None:
            workspace_root = Path(allowed_workspace_dir).expanduser().resolve()
            if not resolved.is_relative_to(workspace_root):
                raise PermissionError(
                    f"Export path '{resolved}' escapes allowed workspace '{workspace_root}'."
                )
        return resolved

    @staticmethod
    def is_spec_compilable_to_residual_vector(spec: InterventionSpec) -> bool:
        """
        Returns True if and only if `spec` is an unconditional, position-independent
        ADDITIVE_STEER, HARD_CLAMP, or ZERO_ABLATE operation without dynamic gating.
        """
        return (
            spec.enabled
            and spec.mode
            in (
                InterventionMode.ADDITIVE_STEER,
                InterventionMode.HARD_CLAMP,
                InterventionMode.ZERO_ABLATE,
            )
            and spec.pole == PoleType.BOTH
            and spec.token_positions is None
            and spec.gate_condition is None
            and spec.custom_fn is None
        )

    @staticmethod
    def is_spec_weight_foldable(spec: InterventionSpec) -> bool:
        """
        Returns True if and only if `spec` is an unconditional (`PoleType.BOTH`),
        position-independent affine neuron operation (`ZERO_ABLATE`, `SCALE`,
        `ADDITIVE_STEER`, or `HARD_CLAMP`) without dynamic gating or custom callables.
        """
        return (
            spec.enabled
            and spec.pole == PoleType.BOTH
            and spec.token_positions is None
            and spec.gate_condition is None
            and spec.custom_fn is None
            and spec.mode
            in (
                InterventionMode.ZERO_ABLATE,
                InterventionMode.SCALE,
                InterventionMode.ADDITIVE_STEER,
                InterventionMode.HARD_CLAMP,
            )
        )

    @staticmethod
    def _resolve_composed_neuron_affine_ops(
        specs: List[InterventionSpec],
        d_mlp: int,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Composes a sequence of unconditional specs on the same layer in exact execution order.
        For each targeted neuron `i`, tracks `(scale_i, shift_i)` such that:
            a_final[i] = scale_i * a_clean[i] + shift_i
        This guarantees that sequential interventions on the same neuron (e.g., HARD_CLAMP(-10)
        followed by HARD_CLAMP(+5) or ADDITIVE_STEER(+2)) produce identical results in
        compiled vectors and weight folding as in `InterventionKernel.apply`.
        """
        composed: Dict[int, Tuple[float, float]] = {}

        for spec in specs:
            if not SteeringExporter.is_spec_weight_foldable(spec):
                continue

            per_vals = (
                spec.per_neuron_values
                if spec.per_neuron_values is not None
                and len(spec.per_neuron_values) == len(spec.neuron_indices)
                else [float(spec.value)] * len(spec.neuron_indices)
            )

            for n_idx, val_i in zip(spec.neuron_indices, per_vals):
                idx = int(n_idx)
                if idx < 0 or idx >= d_mlp:
                    continue
                v = float(val_i)
                scale_curr, shift_curr = composed.get(idx, (1.0, 0.0))

                if spec.mode == InterventionMode.ZERO_ABLATE:
                    composed[idx] = (0.0, 0.0)
                elif spec.mode == InterventionMode.HARD_CLAMP:
                    composed[idx] = (0.0, v)
                elif spec.mode == InterventionMode.SCALE:
                    composed[idx] = (scale_curr * v, shift_curr * v)
                elif spec.mode == InterventionMode.ADDITIVE_STEER:
                    composed[idx] = (scale_curr, shift_curr + v)

        return composed

    @classmethod
    @torch.no_grad()
    def compile_residual_steering_vectors(
        cls,
        model: nn.Module,
        interventions_by_layer: Dict[str, List[InterventionSpec]],
    ) -> List[CompiledSteeringVector]:
        """
        For any non-MoE, non-quantized layer where all enabled interventions are unconditional
        ADDITIVE_STEER, HARD_CLAMP, or ZERO_ABLATE operations, resolves the exact composed
        per-neuron affine state `(scale_i, shift_i)` and compiles the linear shift:
          - When `hook_target == "input"` (`down_proj`):
                v_steer = sum_{i} shift_i * W_down[:, i]  (shape: [d_model])
          - When `hook_target == "output"` (output-hooked Linear/Conv1D):
                v_steer[i] = shift_i  (shape: [out_features])
        For neurons whose composed `scale_i == 0.0` (i.e. HARD_CLAMPed or ZERO_ABLATEd),
        records their indices in `clamped_neurons` so dynamic activations are zeroed prior to
        adding `v_steer`.
        """
        compiled_list: List[CompiledSteeringVector] = []

        for layer_name, specs in interventions_by_layer.items():
            enabled_specs = [s for s in specs if s.enabled]
            if not enabled_specs:
                continue
            if not all(cls.is_spec_compilable_to_residual_vector(s) for s in enabled_specs):
                continue

            module, hook_target, num_neurons = LayerDiscovery.resolve_module(model, layer_name)
            if not hasattr(module, "weight") or LayerDiscovery.is_quantized_or_offloaded(module):
                continue

            w = module.weight.detach()
            # 3D MoE experts only apply to routed token subsets, so static all-token residual
            # vectors cannot be compiled without a router mask; use standard hooks instead.
            if w.ndim == 3 or isinstance(module, nn.Conv2d):
                continue

            is_conv1d = module.__class__.__name__ == "Conv1D"
            if hook_target == "input":
                d_model = w.shape[1] if is_conv1d else w.shape[0]
                d_mlp = w.shape[0] if is_conv1d else w.shape[1]
            else:
                d_model = num_neurons
                d_mlp = num_neurons

            composed = cls._resolve_composed_neuron_affine_ops(enabled_specs, d_mlp)
            if not composed:
                continue

            v_steer = torch.zeros(d_model, dtype=torch.float32, device=w.device)
            src_neurons: List[int] = []
            src_deltas: List[float] = []
            clamped_neurons: List[int] = []

            for n_idx, (scale_i, shift_i) in composed.items():
                if abs(scale_i) < 1e-12:
                    clamped_neurons.append(int(n_idx))

                if hook_target == "input":
                    if is_conv1d:
                        col_vec = w[n_idx, :].float()
                    else:
                        col_vec = w[:, n_idx].float()
                    v_steer = v_steer + float(shift_i) * col_vec
                else:
                    v_steer[n_idx] += float(shift_i)

                src_neurons.append(int(n_idx))
                src_deltas.append(float(shift_i))

            if src_neurons:
                vec_norm = float(torch.norm(v_steer, p=2).item())
                compiled_list.append(
                    CompiledSteeringVector(
                        layer_name=layer_name,
                        d_model=d_model,
                        source_neurons=src_neurons,
                        source_deltas=src_deltas,
                        vector_norm=round(vec_norm, 5),
                        vector_values=[round(float(x), 6) for x in v_steer.cpu().tolist()],
                        clamped_neurons=clamped_neurons,
                    )
                )

        return compiled_list

    @classmethod
    def build_preset_bundle(
        cls,
        model: nn.Module,
        interventions_by_layer: Dict[str, List[InterventionSpec]],
        preset_name: str = "default_steering_preset",
        model_name: str = "custom-model",
        description: str = "Exported surgical neuron interventions and compiled residual steering vectors.",
    ) -> ProductionPresetBundle:
        """Creates a complete ProductionPresetBundle from currently registered interventions."""
        serialized_specs: List[Dict[str, Any]] = []
        for _layer_name, specs in interventions_by_layer.items():
            for s in specs:
                if not s.enabled:
                    continue
                item_dict: Dict[str, Any] = {
                    "layer_name": s.layer_name,
                    "neuron_indices": list(s.neuron_indices),
                    "mode": s.mode.value,
                    "pole": s.pole.value,
                    "value": float(s.value),
                    "token_positions": s.token_positions,
                }
                if s.per_neuron_values is not None:
                    item_dict["per_neuron_values"] = [float(v) for v in s.per_neuron_values]
                if s.gate_condition is not None:
                    item_dict["gate_condition"] = {
                        "detector_layer": s.gate_condition.detector_layer,
                        "detector_neuron": s.gate_condition.detector_neuron,
                        "detector_pole": s.gate_condition.detector_pole.value,
                        "threshold": float(s.gate_condition.threshold),
                        "action_mode": s.gate_condition.action_mode.value,
                        "sequence_scope": s.gate_condition.sequence_scope,
                        "sticky_on_decode": bool(s.gate_condition.sticky_on_decode),
                    }
                serialized_specs.append(item_dict)

        compiled_vectors = cls.compile_residual_steering_vectors(model, interventions_by_layer)

        return ProductionPresetBundle(
            preset_name=preset_name,
            model_name=model_name,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            description=description,
            interventions=serialized_specs,
            compiled_vectors=compiled_vectors,
        )

    @classmethod
    def save_preset_json(
        cls,
        bundle: ProductionPresetBundle,
        filepath: Union[str, Path],
        export_pt_vectors_path: Optional[Union[str, Path]] = None,
        allowed_workspace_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """
        Writes the ProductionPresetBundle to disk as a portable JSON file.
        Optionally also saves a `.pt` dictionary mapping `{layer_name: tensor(d_model)}`
        for direct loading into vLLM / PyTorch serving pipelines.
        """
        path = cls.validate_safe_export_path(filepath, allowed_workspace_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(bundle), indent=2), encoding="utf-8")

        result = {"json_preset_path": str(path)}

        if export_pt_vectors_path is not None and bundle.compiled_vectors:
            pt_path = cls.validate_safe_export_path(export_pt_vectors_path, allowed_workspace_dir)
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_dict = {
                cv.layer_name: torch.tensor(cv.vector_values, dtype=torch.float32)
                for cv in bundle.compiled_vectors
            }
            torch.save(tensor_dict, pt_path)
            result["pt_vectors_path"] = str(pt_path)

        return result

    @staticmethod
    def parse_preset_dict(
        data: Dict[str, Any],
    ) -> Tuple[ProductionPresetBundle, List[InterventionSpec]]:
        """Parses a raw dictionary (from JSON or API payload) back into a bundle and InterventionSpecs."""
        specs: List[InterventionSpec] = []
        for item in data.get("interventions", []):
            gc_raw = item.get("gate_condition")
            gate_cond = (
                GateConditionSpec(
                    detector_layer=gc_raw["detector_layer"],
                    detector_neuron=int(gc_raw["detector_neuron"]),
                    detector_pole=PoleType(gc_raw.get("detector_pole", "positive")),
                    threshold=float(gc_raw.get("threshold", 0.5)),
                    action_mode=InterventionMode(gc_raw.get("action_mode", "additive_steer")),
                    sequence_scope=gc_raw.get("sequence_scope", "token_wise"),
                    sticky_on_decode=bool(gc_raw.get("sticky_on_decode", True)),
                )
                if isinstance(gc_raw, dict)
                else None
            )
            specs.append(
                InterventionSpec(
                    layer_name=item["layer_name"],
                    neuron_indices=[int(i) for i in item["neuron_indices"]],
                    mode=InterventionMode(item.get("mode", "zero_ablate")),
                    pole=PoleType(item.get("pole", "both")),
                    value=float(item.get("value", 0.0)),
                    per_neuron_values=item.get("per_neuron_values"),
                    token_positions=item.get("token_positions"),
                    gate_condition=gate_cond,
                )
            )

        compiled = [
            CompiledSteeringVector(
                layer_name=cv["layer_name"],
                d_model=int(cv["d_model"]),
                source_neurons=cv["source_neurons"],
                source_deltas=cv["source_deltas"],
                vector_norm=float(cv["vector_norm"]),
                vector_values=cv["vector_values"],
                clamped_neurons=[int(i) for i in cv.get("clamped_neurons", [])],
            )
            for cv in data.get("compiled_vectors", [])
        ]

        bundle = ProductionPresetBundle(
            preset_name=data.get("preset_name", "imported_preset"),
            model_name=data.get("model_name", "unknown"),
            created_at_utc=data.get("created_at_utc", ""),
            description=data.get("description", ""),
            interventions=data.get("interventions", []),
            compiled_vectors=compiled,
        )
        return bundle, specs

    @classmethod
    @torch.no_grad()
    def fold_interventions_into_weights(
        cls,
        model: nn.Module,
        interventions_by_layer: Dict[str, List[InterventionSpec]],
    ) -> Dict[str, Any]:
        """
        Folds unconditional (`PoleType.BOTH`, all token positions) surgical interventions
        directly into module weights and biases for ZERO runtime hook overhead.

        Supports BOTH:
          - `hook_target == "input"` (e.g. `down_proj` where neuron `i` is an input column):
              1. Adds `sum_i shift_i * W_down_original[:, i]` to `module.bias`
              2. Multiplies `W_down[:, i] *= scale_i`
          - `hook_target == "output"` (e.g. `fc_in`, `Conv2d`, or fallback `Linear` where neuron `i`
            is an output row/channel `y[i] = W[i, :] @ x + b[i]`):
              1. Multiplies `W[i, :] *= scale_i`
              2. Updates `b[i] = b[i] * scale_i + shift_i`

        Never silently drops unfoldable specs (`CONDITIONAL_GATE`, `pole != BOTH`, `WEAK_PRUNE`,
        `POLE_RECTIFY`, or `token_positions != None`): returns `"folded_spec_ids"` so callers
        retain all unfoldable specs in active hooks.
        Raises `ValueError` if called on quantized/offloaded or 3D parameter-container MoE layers.
        """
        folded_summary: List[Dict[str, Any]] = []
        folded_spec_ids: Set[int] = set()
        retained_unfoldable_count = 0

        for layer_name, specs in interventions_by_layer.items():
            enabled_specs = [s for s in specs if s.enabled]
            if not enabled_specs:
                continue

            module, hook_target, num_neurons = LayerDiscovery.resolve_module(model, layer_name)
            if LayerDiscovery.is_quantized_or_offloaded(module):
                raise ValueError(
                    f"Cannot fold interventions into quantized or offloaded layer '{layer_name}'. "
                    "Use active_injection() or ProductionSteeringRuntime instead."
                )
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 3:
                raise ValueError(
                    f"Cannot fold interventions into 3D MoE parameter container '{layer_name}'. "
                    "Use active_injection() or ProductionSteeringRuntime instead."
                )

            is_conv1d = module.__class__.__name__ == "Conv1D" and hasattr(module, "nf")
            is_conv2d = isinstance(module, nn.Conv2d)
            if not isinstance(module, nn.Linear) and not is_conv1d and not is_conv2d:
                retained_unfoldable_count += len(enabled_specs)
                continue

            # Identify neurons touched by any unfoldable spec so we do not partially fold
            # a neuron that also has a conditional or pole-specific hook on the same layer.
            unfoldable_neurons: Set[int] = set()
            for s in enabled_specs:
                if not cls.is_spec_weight_foldable(s):
                    retained_unfoldable_count += 1
                    unfoldable_neurons.update(int(i) for i in s.neuron_indices)

            foldable_specs = [
                s
                for s in enabled_specs
                if cls.is_spec_weight_foldable(s)
                and not any(int(i) in unfoldable_neurons for i in s.neuron_indices)
            ]
            for s in enabled_specs:
                if cls.is_spec_weight_foldable(s) and s not in foldable_specs:
                    retained_unfoldable_count += 1

            if not foldable_specs:
                continue

            composed = cls._resolve_composed_neuron_affine_ops(foldable_specs, num_neurons)
            if not composed:
                continue

            device = module.weight.device
            dtype = module.weight.dtype

            if hook_target == "input":
                out_features = (
                    module.nf
                    if is_conv1d
                    else (module.out_channels if is_conv2d else module.out_features)
                )
                v_shift_total = torch.zeros(out_features, dtype=dtype, device=device)
                has_bias_shift = False

                for n_idx, (scale_i, shift_i) in composed.items():
                    if abs(shift_i) > 0.0:
                        has_bias_shift = True
                        if is_conv1d:
                            v_shift_total.add_(module.weight[n_idx, :] * float(shift_i))
                        elif is_conv2d:
                            v_shift_total.add_(
                                module.weight[:, n_idx, :, :].sum(dim=(1, 2)) * float(shift_i)
                            )
                        else:
                            v_shift_total.add_(module.weight[:, n_idx] * float(shift_i))

                    if is_conv1d:
                        module.weight[n_idx, :] *= float(scale_i)
                    elif is_conv2d:
                        module.weight[:, n_idx, :, :] *= float(scale_i)
                    else:
                        module.weight[:, n_idx] *= float(scale_i)

                if has_bias_shift:
                    if module.bias is None:
                        module.bias = nn.Parameter(
                            torch.zeros(out_features, dtype=dtype, device=device)
                        )
                    module.bias.add_(v_shift_total)
            else:
                # hook_target == "output": y'[n_idx] = scale_i * (W[n_idx] @ x + b[n_idx]) + shift_i
                out_features = num_neurons
                need_bias = any(abs(shift_i) > 0.0 for _, shift_i in composed.values())
                if need_bias and module.bias is None:
                    module.bias = nn.Parameter(
                        torch.zeros(out_features, dtype=dtype, device=device)
                    )

                for n_idx, (scale_i, shift_i) in composed.items():
                    if is_conv1d:
                        module.weight[:, n_idx] *= float(scale_i)
                    elif is_conv2d:
                        module.weight[n_idx, :, :, :] *= float(scale_i)
                    else:
                        module.weight[n_idx, :] *= float(scale_i)

                    if module.bias is not None:
                        module.bias[n_idx] = (
                            module.bias[n_idx] * float(scale_i) + float(shift_i)
                        )

            axis_word = "columns" if hook_target == "input" else "rows"
            for s in foldable_specs:
                folded_spec_ids.add(id(s))
                if s.mode == InterventionMode.ZERO_ABLATE:
                    op_label = f"zero_weight_{axis_word}"
                elif s.mode == InterventionMode.SCALE:
                    op_label = f"scale_weight_{axis_word}({s.value})"
                else:
                    op_label = f"folded_{s.mode.value}_into_{axis_word}_and_bias"
                folded_summary.append({
                    "layer": layer_name,
                    "neurons": list(s.neuron_indices),
                    "hook_target": hook_target,
                    "folded_op": op_label,
                })

        return {
            "status": "folded_into_weights",
            "operations_applied": len(folded_summary),
            "unfolded_specs_retained": retained_unfoldable_count,
            "folded_spec_ids": list(folded_spec_ids),
            "details": folded_summary,
        }


class ProductionSteeringRuntime:
    """
    Lightweight, standalone production inference applier.
    Can load a saved JSON preset and apply either:
      - Mode A (`use_compiled_residual_vectors=True`): Fast residual-stream vector addition `output + v_steer`
        for layers with compiled vectors (plus dynamic pre-zeroing for any `clamped_neurons`),
        and exact neuron hooks for any remaining layers.
      - Mode B (`use_compiled_residual_vectors=False`): Exact pole-conditional and
        cross-layer `CONDITIONAL_GATE` neuron hooks with KV-cache latching (`sticky_on_decode`).
    """

    def __init__(self, model: nn.Module, preset_json_or_dict: Union[str, Path, Dict[str, Any]]):
        self.model = model
        if isinstance(preset_json_or_dict, (str, Path)):
            raw = json.loads(Path(preset_json_or_dict).read_text(encoding="utf-8"))
        else:
            raw = preset_json_or_dict

        self.bundle, self.specs = SteeringExporter.parse_preset_dict(raw)
        self._handles: List[RemovableHandle] = []
        self._latched_masks: Dict[int, torch.Tensor] = {}

    def reset_latched_gates(self) -> None:
        """Clears any latched KV-cache masks accumulated by `sticky_on_decode` conditional gates."""
        self._latched_masks.clear()

    def _validate_against_model(self, use_compiled_residual_vectors: bool = False) -> None:
        """
        Validates that all layer names, neuron indices, and compiled vector dimensions
        in the loaded preset match `self.model` before attaching hooks.
        """
        for s in self.specs:
            _, _, num_neurons = LayerDiscovery.resolve_module(self.model, s.layer_name)
            for idx in s.neuron_indices:
                if idx < 0 or idx >= num_neurons:
                    raise ValueError(
                        f"Neuron index {idx} in preset spec for layer '{s.layer_name}' "
                        f"is out of range for model layer with {num_neurons} neurons."
                    )
            if s.gate_condition is not None:
                _, _, det_num = LayerDiscovery.resolve_module(
                    self.model, s.gate_condition.detector_layer
                )
                d_idx = s.gate_condition.detector_neuron
                if d_idx < 0 or d_idx >= det_num:
                    raise ValueError(
                        f"Detector neuron index {d_idx} for layer "
                        f"'{s.gate_condition.detector_layer}' is out of range ({det_num})."
                    )

        if use_compiled_residual_vectors and self.bundle.compiled_vectors:
            for cv in self.bundle.compiled_vectors:
                module, hook_target, num_neurons = LayerDiscovery.resolve_module(
                    self.model, cv.layer_name
                )
                for c_idx in cv.clamped_neurons or []:
                    if c_idx < 0 or c_idx >= num_neurons:
                        raise ValueError(
                            f"Clamped neuron index {c_idx} in compiled vector for layer "
                            f"'{cv.layer_name}' is out of range for {num_neurons} neurons."
                        )
                expected_dim: Optional[int] = None
                if isinstance(module, nn.Linear):
                    expected_dim = module.out_features
                elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                    if module.__class__.__name__ == "Conv1D":
                        expected_dim = int(module.weight.shape[1])
                    elif module.weight.ndim == 2:
                        expected_dim = int(module.weight.shape[0])
                if expected_dim is not None and len(cv.vector_values) != expected_dim:
                    raise ValueError(
                        f"Compiled steering vector dimension mismatch for layer '{cv.layer_name}': "
                        f"preset vector has length {len(cv.vector_values)}, but model module has "
                        f"output dimension {expected_dim}."
                    )

    def attach(self, use_compiled_residual_vectors: bool = False) -> None:
        self.detach()
        self._validate_against_model(
            use_compiled_residual_vectors=use_compiled_residual_vectors
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        conditional_masks: Dict[int, torch.Tensor] = {}
        latched_masks = self._latched_masks

        # Clear conditional_masks at the start of every forward pass on the root model
        self._handles.append(
            self.model.register_forward_pre_hook(lambda _m, _inp: conditional_masks.clear())
        )

        compiled_layers = set()
        if use_compiled_residual_vectors and self.bundle.compiled_vectors:
            for cv in self.bundle.compiled_vectors:
                module, hook_target, _ = LayerDiscovery.resolve_module(self.model, cv.layer_name)
                steer_tensor = torch.tensor(cv.vector_values, dtype=dtype, device=device)
                compiled_layers.add(cv.layer_name)
                clamp_idx = list(cv.clamped_neurons) if cv.clamped_neurons else []

                # If any neurons in this compiled vector were HARD_CLAMPed, zero their dynamic
                # activations before `down_proj` (when hook_target == "input") so adding `v_steer`
                # gives exact clamp output.
                if clamp_idx and hook_target == "input":

                    def _make_clamp_zero_pre(c_indices: List[int]):
                        def _pre_zero(_mod: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                            if inputs and isinstance(inputs[0], torch.Tensor):
                                cloned = inputs[0].clone()
                                idx_t = torch.as_tensor(
                                    c_indices, dtype=torch.long, device=cloned.device
                                )
                                cloned[..., idx_t] = 0.0
                                return (cloned,) + inputs[1:]
                            return inputs

                        return _pre_zero

                    self._handles.append(
                        module.register_forward_pre_hook(_make_clamp_zero_pre(clamp_idx))
                    )

                def _make_res_hook(
                    vec: torch.Tensor, c_indices: List[int], h_target: str
                ):
                    def _fwd_hook(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any) -> Any:
                        t0 = out[0] if isinstance(out, tuple) else out
                        if not isinstance(t0, torch.Tensor):
                            return out
                        modified = t0.clone() if (c_indices and h_target == "output") else t0
                        if c_indices and h_target == "output":
                            idx_t = torch.as_tensor(
                                c_indices, dtype=torch.long, device=modified.device
                            )
                            modified[..., idx_t] = 0.0
                        modified = modified + vec.to(device=modified.device, dtype=modified.dtype)
                        return (modified,) + out[1:] if isinstance(out, tuple) else modified

                    return _fwd_hook

                self._handles.append(
                    module.register_forward_hook(
                        _make_res_hook(steer_tensor, clamp_idx, hook_target)
                    )
                )

        # Standard surgical neuron-basis hooks grouped by layer (for any layer not already handled by a compiled vector)
        by_layer: Dict[str, List[InterventionSpec]] = {}
        for s in self.specs:
            if use_compiled_residual_vectors and s.layer_name in compiled_layers:
                continue
            by_layer.setdefault(s.layer_name, []).append(s)

        # Register cross-layer detector hooks first so conditional_masks are populated
        for layer_name, layer_specs in by_layer.items():
            for spec in layer_specs:
                if not spec.enabled or spec.gate_condition is None:
                    continue
                det_layer = spec.gate_condition.detector_layer
                if det_layer == layer_name:
                    continue
                det_mod, det_target, _ = LayerDiscovery.resolve_module(self.model, det_layer)
                is_det_conv = isinstance(det_mod, nn.Conv2d)

                def _make_det_pre(s_obj: InterventionSpec, conv_f: bool):
                    def _det_pre(_mod: nn.Module, inputs: Tuple[Any, ...]) -> None:
                        if inputs and isinstance(inputs[0], torch.Tensor) and s_obj.gate_condition:
                            conditional_masks[id(s_obj)] = (
                                InterventionKernel.compute_detector_mask(
                                    inputs[0].detach(),
                                    s_obj.gate_condition,
                                    is_conv2d=conv_f,
                                    latched_masks=latched_masks,
                                    spec_id=id(s_obj),
                                )
                            )

                    return _det_pre

                def _make_det_fwd(s_obj: InterventionSpec, conv_f: bool):
                    def _det_fwd(_mod: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                        act = output[0] if isinstance(output, tuple) else output
                        if isinstance(act, torch.Tensor) and s_obj.gate_condition:
                            conditional_masks[id(s_obj)] = (
                                InterventionKernel.compute_detector_mask(
                                    act.detach(),
                                    s_obj.gate_condition,
                                    is_conv2d=conv_f,
                                    latched_masks=latched_masks,
                                    spec_id=id(s_obj),
                                )
                            )

                    return _det_fwd

                if det_target == "input":
                    self._handles.append(
                        det_mod.register_forward_pre_hook(_make_det_pre(spec, is_det_conv))
                    )
                else:
                    self._handles.append(
                        det_mod.register_forward_hook(_make_det_fwd(spec, is_det_conv))
                    )

        for layer_name, layer_specs in by_layer.items():
            module, hook_target, _ = LayerDiscovery.resolve_module(self.model, layer_name)
            is_conv2d = isinstance(module, nn.Conv2d)
            if hook_target == "input":

                def _make_pre(ls: List[InterventionSpec], cf: bool):
                    def _pre(_mod: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            return (
                                InterventionKernel.apply(
                                    inputs[0],
                                    ls,
                                    is_conv2d=cf,
                                    conditional_masks=conditional_masks,
                                    latched_masks=latched_masks,
                                ),
                            ) + inputs[1:]
                        return inputs

                    return _pre

                self._handles.append(
                    module.register_forward_pre_hook(_make_pre(layer_specs, is_conv2d))
                )
            else:

                def _make_fwd(ls: List[InterventionSpec], cf: bool):
                    def _fwd(_mod: nn.Module, _inputs: Tuple[Any, ...], out: Any) -> Any:
                        if isinstance(out, torch.Tensor):
                            return InterventionKernel.apply(
                                out,
                                ls,
                                is_conv2d=cf,
                                conditional_masks=conditional_masks,
                                latched_masks=latched_masks,
                            )
                        elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                            return (
                                InterventionKernel.apply(
                                    out[0],
                                    ls,
                                    is_conv2d=cf,
                                    conditional_masks=conditional_masks,
                                    latched_masks=latched_masks,
                                ),
                            ) + out[1:]
                        return out

                    return _fwd

                self._handles.append(
                    module.register_forward_hook(_make_fwd(layer_specs, is_conv2d))
                )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextmanager
    def active(self, use_compiled_residual_vectors: bool = False):
        self.reset_latched_gates()
        self.attach(use_compiled_residual_vectors=use_compiled_residual_vectors)
        try:
            yield self
        except Exception:
            raise
        finally:
            self.detach()
            self.reset_latched_gates()
