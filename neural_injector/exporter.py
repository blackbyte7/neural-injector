from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        ADDITIVE_STEER or HARD_CLAMP operation without dynamic gating.
        """
        return (
            spec.enabled
            and spec.mode in (InterventionMode.ADDITIVE_STEER, InterventionMode.HARD_CLAMP)
            and spec.pole == PoleType.BOTH
            and spec.token_positions is None
            and spec.gate_condition is None
            and spec.custom_fn is None
        )

    @classmethod
    @torch.no_grad()
    def compile_residual_steering_vectors(
        cls,
        model: nn.Module,
        interventions_by_layer: Dict[str, List[InterventionSpec]],
    ) -> List[CompiledSteeringVector]:
        """
        For any layer where all enabled interventions are unconditional ADDITIVE_STEER
        or HARD_CLAMP operations, compiles the combined linear shift in the residual stream:
            v_steer = sum_{i} delta_i * W_down[:, i]  (shape: [d_model])
        For HARD_CLAMP neurons, also records their indices in `clamped_neurons` so that
        incoming dynamic activations are zeroed prior to `down_proj` (preventing a_i(x)
        from adding onto the constant clamp value c).
        """
        compiled_list: List[CompiledSteeringVector] = []

        for layer_name, specs in interventions_by_layer.items():
            enabled_specs = [s for s in specs if s.enabled]
            if not enabled_specs:
                continue
            if not all(cls.is_spec_compilable_to_residual_vector(s) for s in enabled_specs):
                continue

            module, _, _ = LayerDiscovery.resolve_module(model, layer_name)
            if not hasattr(module, "weight"):
                continue

            w = module.weight.detach()
            is_conv1d = module.__class__.__name__ == "Conv1D"
            if w.ndim == 3:
                # 3D MoE down_proj: [num_experts, d_model, d_mlp]
                d_model, d_mlp = w.shape[1], w.shape[2]
            else:
                d_model = w.shape[1] if is_conv1d else w.shape[0]
                d_mlp = w.shape[0] if is_conv1d else w.shape[1]

            v_steer = torch.zeros(d_model, dtype=torch.float32, device=w.device)
            src_neurons: List[int] = []
            src_deltas: List[float] = []
            clamped_neurons: List[int] = []

            for spec in enabled_specs:
                per_vals = (
                    spec.per_neuron_values
                    if spec.per_neuron_values is not None
                    and len(spec.per_neuron_values) == len(spec.neuron_indices)
                    else [float(spec.value)] * len(spec.neuron_indices)
                )
                for n_idx, val_i in zip(spec.neuron_indices, per_vals):
                    if n_idx < 0 or n_idx >= d_mlp:
                        continue
                    if w.ndim == 3:
                        col_vec = w[:, :, n_idx].float().mean(dim=0)
                    elif is_conv1d:
                        col_vec = w[n_idx, :].float()
                    else:
                        col_vec = w[:, n_idx].float()

                    # If a neuron is HARD_CLAMPed after an earlier spec in the same layer,
                    # its previous contribution is overwritten by the clamp constant.
                    if spec.mode == InterventionMode.HARD_CLAMP and int(n_idx) not in clamped_neurons:
                        clamped_neurons.append(int(n_idx))

                    v_steer = v_steer + float(val_i) * col_vec
                    src_neurons.append(int(n_idx))
                    src_deltas.append(float(val_i))

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

    @staticmethod
    @torch.no_grad()
    def fold_interventions_into_weights(
        model: nn.Module,
        interventions_by_layer: Dict[str, List[InterventionSpec]],
    ) -> Dict[str, Any]:
        """
        Folds unconditional (`PoleType.BOTH`, all token positions) surgical interventions
        directly into `down_proj` weights and biases for ZERO runtime hook overhead:
          - ZERO_ABLATE : sets `down_proj.weight[:, i] = 0.0` (or `weight[i, :] = 0.0` for Conv1D)
          - SCALE       : multiplies `down_proj.weight[:, i] *= value`
          - ADDITIVE_STEER / HARD_CLAMP: folds `v_steer = sum(val * W_down[:, i])` into `down_proj.bias`
            (and zeroes `W_down[:, i]` first if mode is HARD_CLAMP).
        """
        folded_summary: List[Dict[str, Any]] = []

        for layer_name, specs in interventions_by_layer.items():
            module, _, _ = LayerDiscovery.resolve_module(model, layer_name)
            is_conv1d = module.__class__.__name__ == "Conv1D" and hasattr(module, "nf")
            if not isinstance(module, nn.Linear) and not is_conv1d:
                continue

            for spec in specs:
                if (
                    not spec.enabled
                    or spec.pole != PoleType.BOTH
                    or spec.token_positions is not None
                    or spec.gate_condition is not None
                ):
                    continue

                idx = torch.as_tensor(
                    spec.neuron_indices, dtype=torch.long, device=module.weight.device
                )
                if spec.per_neuron_values is not None and len(spec.per_neuron_values) == len(spec.neuron_indices):
                    val_vec = torch.as_tensor(
                        spec.per_neuron_values,
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )
                else:
                    val_vec = torch.full(
                        (len(spec.neuron_indices),),
                        float(spec.value),
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )

                if spec.mode == InterventionMode.ZERO_ABLATE:
                    if is_conv1d:
                        module.weight[idx, :] = 0.0
                    else:
                        module.weight[:, idx] = 0.0
                    folded_summary.append({
                        "layer": layer_name,
                        "neurons": spec.neuron_indices,
                        "folded_op": "zero_weight_columns",
                    })

                elif spec.mode == InterventionMode.SCALE:
                    if is_conv1d:
                        module.weight[idx, :] *= val_vec.unsqueeze(1)
                    else:
                        module.weight[:, idx] *= val_vec.unsqueeze(0)
                    folded_summary.append({
                        "layer": layer_name,
                        "neurons": spec.neuron_indices,
                        "folded_op": f"scale_weight_columns({spec.value})",
                    })

                elif spec.mode in (InterventionMode.ADDITIVE_STEER, InterventionMode.HARD_CLAMP):
                    # Compute residual steering vector contribution: sum_{i in idx} value_i * W_down[:, i]
                    if is_conv1d:
                        v_shift = (module.weight[idx, :] * val_vec.unsqueeze(1)).sum(dim=0)
                        out_features = module.nf
                    else:
                        v_shift = (module.weight[:, idx] * val_vec.unsqueeze(0)).sum(dim=1)
                        out_features = module.out_features

                    if module.bias is None:
                        module.bias = nn.Parameter(
                            torch.zeros(
                                out_features,
                                dtype=module.weight.dtype,
                                device=module.weight.device,
                            )
                        )
                    module.bias.add_(v_shift)

                    if spec.mode == InterventionMode.HARD_CLAMP:
                        # Remove input-dependent variation on clamped neurons so output is strictly constant
                        if is_conv1d:
                            module.weight[idx, :] = 0.0
                        else:
                            module.weight[:, idx] = 0.0

                    folded_summary.append({
                        "layer": layer_name,
                        "neurons": spec.neuron_indices,
                        "folded_op": f"folded_{spec.mode.value}_into_bias",
                    })

        return {
            "status": "folded_into_weights",
            "operations_applied": len(folded_summary),
            "details": folded_summary,
        }


class ProductionSteeringRuntime:
    """
    Lightweight, standalone production inference applier.
    Can load a saved JSON preset and apply either:
      - Mode A (`use_compiled_residual_vectors=True`): Fast residual-stream vector addition `output + v_steer`
        for layers with compiled vectors (plus dynamic pre-zeroing for any `clamped_neurons`),
        and exact neuron hooks for any remaining layers.
      - Mode B (`use_compiled_residual_vectors=False`): Exact pole-conditional neuron hooks.
    """

    def __init__(self, model: nn.Module, preset_json_or_dict: Union[str, Path, Dict[str, Any]]):
        self.model = model
        if isinstance(preset_json_or_dict, (str, Path)):
            raw = json.loads(Path(preset_json_or_dict).read_text(encoding="utf-8"))
        else:
            raw = preset_json_or_dict

        self.bundle, self.specs = SteeringExporter.parse_preset_dict(raw)
        self._handles: List[RemovableHandle] = []

    def attach(self, use_compiled_residual_vectors: bool = False) -> None:
        self.detach()
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        compiled_layers = set()
        if use_compiled_residual_vectors and self.bundle.compiled_vectors:
            for cv in self.bundle.compiled_vectors:
                module, hook_target, _ = LayerDiscovery.resolve_module(self.model, cv.layer_name)
                steer_tensor = torch.tensor(cv.vector_values, dtype=dtype, device=device)
                compiled_layers.add(cv.layer_name)

                # If any neurons in this compiled vector were HARD_CLAMPed, zero their dynamic
                # activations before `down_proj` so adding `v_steer` gives exact clamp output.
                if cv.clamped_neurons and hook_target == "input":
                    clamp_idx = list(cv.clamped_neurons)

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

                def _make_res_hook(vec: torch.Tensor):
                    def _fwd_hook(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any) -> Any:
                        if isinstance(out, torch.Tensor):
                            return out + vec.to(device=out.device, dtype=out.dtype)
                        elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                            t0 = out[0]
                            return (t0 + vec.to(device=t0.device, dtype=t0.dtype),) + out[1:]
                        return out

                    return _fwd_hook

                self._handles.append(module.register_forward_hook(_make_res_hook(steer_tensor)))

        # Standard surgical neuron-basis hooks grouped by layer (for any layer not already handled by a compiled vector)
        by_layer: Dict[str, List[InterventionSpec]] = {}
        for s in self.specs:
            if use_compiled_residual_vectors and s.layer_name in compiled_layers:
                continue
            by_layer.setdefault(s.layer_name, []).append(s)

        for layer_name, layer_specs in by_layer.items():
            module, hook_target, _ = LayerDiscovery.resolve_module(self.model, layer_name)
            is_conv2d = isinstance(module, nn.Conv2d)
            if hook_target == "input":

                def _make_pre(ls: List[InterventionSpec], cf: bool):
                    def _pre(_mod: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            return (
                                InterventionKernel.apply(inputs[0], ls, is_conv2d=cf),
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
                            return InterventionKernel.apply(out, ls, is_conv2d=cf)
                        elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                            return (
                                InterventionKernel.apply(out[0], ls, is_conv2d=cf),
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
        self.attach(use_compiled_residual_vectors=use_compiled_residual_vectors)
        try:
            yield self
        finally:
            self.detach()
