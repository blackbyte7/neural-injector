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
from .interventions import build_intervention_hookset
from .types import (
    CompiledSteeringVector,
    GateConditionSpec,
    InterventionMode,
    InterventionSpec,
    PoleType,
    ProductionPresetBundle,
)

_AFFINE_MODES = (InterventionMode.ZERO_ABLATE, InterventionMode.SCALE, InterventionMode.ADDITIVE_STEER, InterventionMode.HARD_CLAMP)


class SteeringExporter:
    """
    Turns registered interventions into deployable artefacts:
      * JSON presets (all modes, including conditional gates and gate-conditioned modes),
      * residual-stream steering vectors v = sum_i delta_i W_down[:, i] for layers whose interventions
        are purely additive and unconditional (the only case in which a vector reproduces them exactly
        with a single output hook),
      * exact weight folding of unconditional affine interventions a' = s a + t into W_down and its bias.
    """

    @staticmethod
    def validate_safe_export_path(filepath: Union[str, Path], allowed_workspace_dir: Optional[Union[str, Path]] = None) -> Path:
        raw = Path(filepath)
        if ".." in raw.parts:
            raise PermissionError(f"Path traversal ('..') is not permitted: '{filepath}'.")
        resolved = raw.expanduser().resolve()
        if allowed_workspace_dir is not None:
            root = Path(allowed_workspace_dir).expanduser().resolve()
            if not resolved.is_relative_to(root):
                raise PermissionError(f"Path '{resolved}' escapes allowed workspace '{root}'.")
        return resolved

    @staticmethod
    def _unconditional(spec: InterventionSpec) -> bool:
        return (spec.enabled and spec.pole == PoleType.BOTH and spec.token_positions is None
                and spec.gate_condition is None and spec.custom_fn is None)

    @classmethod
    def is_spec_compilable_to_residual_vector(cls, spec: InterventionSpec) -> bool:
        return cls._unconditional(spec) and spec.mode == InterventionMode.ADDITIVE_STEER

    @classmethod
    def is_spec_weight_foldable(cls, spec: InterventionSpec) -> bool:
        return cls._unconditional(spec) and spec.mode in _AFFINE_MODES

    @classmethod
    def _compose_affine(cls, specs: List[InterventionSpec], d_mlp: int) -> Dict[int, Tuple[float, float]]:
        """Per neuron (scale, shift) with a_final = scale * a + shift, composed in execution order."""
        composed: Dict[int, Tuple[float, float]] = {}
        for spec in specs:
            if not cls.is_spec_weight_foldable(spec):
                continue
            vals = spec.per_neuron_values if spec.per_neuron_values is not None else [float(spec.value)] * len(spec.neuron_indices)
            for n_idx, v in zip(spec.neuron_indices, vals):
                i = int(n_idx)
                if not 0 <= i < d_mlp:
                    continue
                sc, sh = composed.get(i, (1.0, 0.0))
                v = float(v)
                if spec.mode == InterventionMode.ZERO_ABLATE:
                    composed[i] = (0.0, 0.0)
                elif spec.mode == InterventionMode.HARD_CLAMP:
                    composed[i] = (0.0, v)
                elif spec.mode == InterventionMode.SCALE:
                    composed[i] = (sc * v, sh * v)
                else:
                    composed[i] = (sc, sh + v)
        return composed

    @classmethod
    @torch.no_grad()
    def compile_residual_steering_vectors(cls, model: nn.Module, interventions_by_layer: Dict[str, List[InterventionSpec]]) -> List[CompiledSteeringVector]:
        out: List[CompiledSteeringVector] = []
        for layer_name, specs in interventions_by_layer.items():
            active = [s for s in specs if s.enabled]
            if not active or not all(cls.is_spec_compilable_to_residual_vector(s) for s in active):
                continue
            module, target, n = LayerDiscovery.resolve_module(model, layer_name)
            if target != "input" or LayerDiscovery.is_quantized_or_offloaded(module):
                continue
            W = LayerDiscovery.neuron_write_vectors(model, layer_name)
            if W is None:
                continue
            composed = cls._compose_affine(active, n)
            v = torch.zeros(W.shape[1], dtype=torch.float64, device=W.device)
            for i, (_sc, sh) in composed.items():
                v += float(sh) * W[i].double()
            out.append(CompiledSteeringVector(
                layer_name=layer_name, d_model=int(W.shape[1]), source_neurons=list(composed.keys()),
                source_deltas=[float(sh) for _sc, sh in composed.values()], vector_norm=float(v.norm()),
                vector_values=[float(x) for x in v.cpu().tolist()],
            ))
        return out

    @staticmethod
    def serialize_spec(s: InterventionSpec) -> Dict[str, Any]:
        if s.custom_fn is not None:
            raise ValueError("Specs with custom_fn cannot be serialised.")
        d: Dict[str, Any] = {
            "layer_name": s.layer_name, "neuron_indices": [int(i) for i in s.neuron_indices], "mode": s.mode.value,
            "pole": s.pole.value, "value": float(s.value), "token_positions": s.token_positions,
            "condition_signal": s.condition_signal,
        }
        if s.per_neuron_values is not None:
            d["per_neuron_values"] = [float(v) for v in s.per_neuron_values]
        if s.gate_condition is not None:
            g = s.gate_condition
            d["gate_condition"] = {"detector_layer": g.detector_layer, "detector_neuron": g.detector_neuron,
                                   "detector_pole": g.detector_pole.value, "threshold": float(g.threshold),
                                   "action_mode": g.action_mode.value, "sequence_scope": g.sequence_scope,
                                   "sticky_on_decode": bool(g.sticky_on_decode)}
        return d

    @classmethod
    def build_preset_bundle(cls, model: nn.Module, interventions_by_layer: Dict[str, List[InterventionSpec]],
                            preset_name: str = "steering_preset", model_name: str = "custom-model",
                            description: str = "") -> ProductionPresetBundle:
        specs = [cls.serialize_spec(s) for ss in interventions_by_layer.values() for s in ss if s.enabled]
        return ProductionPresetBundle(preset_name=preset_name, model_name=model_name,
                                      created_at_utc=datetime.now(timezone.utc).isoformat(), description=description,
                                      interventions=specs,
                                      compiled_vectors=cls.compile_residual_steering_vectors(model, interventions_by_layer))

    @classmethod
    def save_preset_json(cls, bundle: ProductionPresetBundle, filepath: Union[str, Path],
                         export_pt_vectors_path: Optional[Union[str, Path]] = None,
                         allowed_workspace_dir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
        path = cls.validate_safe_export_path(filepath, allowed_workspace_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(bundle), indent=2), encoding="utf-8")
        res = {"json_preset_path": str(path)}
        if export_pt_vectors_path is not None and bundle.compiled_vectors:
            pt = cls.validate_safe_export_path(export_pt_vectors_path, allowed_workspace_dir)
            pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({cv.layer_name: torch.tensor(cv.vector_values, dtype=torch.float32) for cv in bundle.compiled_vectors}, pt)
            res["pt_vectors_path"] = str(pt)
        return res

    @staticmethod
    def parse_preset_dict(data: Dict[str, Any]) -> Tuple[ProductionPresetBundle, List[InterventionSpec]]:
        specs: List[InterventionSpec] = []
        for item in data.get("interventions", []):
            g = item.get("gate_condition")
            gate = GateConditionSpec(
                detector_layer=g["detector_layer"], detector_neuron=int(g["detector_neuron"]),
                threshold=float(g["threshold"]), detector_pole=PoleType(g.get("detector_pole", "positive")),
                action_mode=InterventionMode(g.get("action_mode", "additive_steer")),
                sequence_scope=g.get("sequence_scope", "token_wise"), sticky_on_decode=bool(g.get("sticky_on_decode", True)),
            ) if isinstance(g, dict) else None
            specs.append(InterventionSpec(
                layer_name=item["layer_name"], neuron_indices=[int(i) for i in item["neuron_indices"]],
                mode=InterventionMode(item.get("mode", "zero_ablate")), pole=PoleType(item.get("pole", "both")),
                value=float(item.get("value", 0.0)), per_neuron_values=item.get("per_neuron_values"),
                token_positions=item.get("token_positions"), gate_condition=gate,
                condition_signal=item.get("condition_signal", "activation"),
            ))
        compiled = [CompiledSteeringVector(
            layer_name=cv["layer_name"], d_model=int(cv["d_model"]), source_neurons=list(cv["source_neurons"]),
            source_deltas=list(cv["source_deltas"]), vector_norm=float(cv["vector_norm"]), vector_values=list(cv["vector_values"]),
        ) for cv in data.get("compiled_vectors", [])]
        bundle = ProductionPresetBundle(preset_name=data.get("preset_name", "imported_preset"),
                                        model_name=data.get("model_name", "unknown"),
                                        created_at_utc=data.get("created_at_utc", ""), description=data.get("description", ""),
                                        interventions=data.get("interventions", []), compiled_vectors=compiled)
        return bundle, specs

    @classmethod
    @torch.no_grad()
    def fold_interventions_into_weights(cls, model: nn.Module, interventions_by_layer: Dict[str, List[InterventionSpec]]) -> Dict[str, Any]:
        """
        Folds unconditional affine interventions into weights:
          input-hooked layers (neuron i = column i of W):  b += sum_i t_i W[:, i];  W[:, i] *= s_i
          output-hooked layers (unit i = row i):            W[i, :] *= s_i;  b[i] = s_i b[i] + t_i
        A neuron also touched by any unfoldable spec is not folded, so execution order is preserved.
        """
        summary: List[Dict[str, Any]] = []
        folded: Set[int] = set()
        retained = 0
        for layer_name, specs in interventions_by_layer.items():
            active = [s for s in specs if s.enabled]
            if not active:
                continue
            module, target, n = LayerDiscovery.resolve_module(model, layer_name)
            if target == "moe_expert" or (hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 3):
                retained += len(active)
                continue
            if LayerDiscovery.is_quantized_or_offloaded(module):
                raise ValueError(f"Cannot fold into quantized/offloaded layer '{layer_name}'.")
            is_c1 = module.__class__.__name__ == "Conv1D" and hasattr(module, "nf")
            is_c2 = isinstance(module, nn.Conv2d)
            if not (isinstance(module, nn.Linear) or is_c1 or is_c2):
                retained += len(active)
                continue
            blocked: Set[int] = set()
            for s in active:
                if not cls.is_spec_weight_foldable(s):
                    blocked.update(int(i) for i in s.neuron_indices)
            foldable = [s for s in active if cls.is_spec_weight_foldable(s) and not blocked.intersection(int(i) for i in s.neuron_indices)]
            retained += len(active) - len(foldable)
            if not foldable:
                continue
            composed = cls._compose_affine(foldable, n)
            W = module.weight
            if target == "input":
                out_f = module.nf if is_c1 else (module.out_channels if is_c2 else module.out_features)
                shift = torch.zeros(out_f, dtype=torch.float64, device=W.device)
                for i, (sc, sh) in composed.items():
                    if sh != 0.0:
                        col = W[i, :] if is_c1 else (W[:, i].sum(dim=(1, 2)) if is_c2 else W[:, i])
                        shift += float(sh) * col.double()
                    if is_c1:
                        W[i, :] *= sc
                    elif is_c2:
                        W[:, i] *= sc
                    else:
                        W[:, i] *= sc
                if bool((shift != 0).any()):
                    if module.bias is None:
                        module.bias = nn.Parameter(torch.zeros(out_f, dtype=W.dtype, device=W.device))
                    module.bias.add_(shift.to(W.dtype))
            else:
                if any(sh != 0.0 for _s, sh in composed.values()) and module.bias is None:
                    module.bias = nn.Parameter(torch.zeros(n, dtype=W.dtype, device=W.device))
                for i, (sc, sh) in composed.items():
                    if is_c1:
                        W[:, i] *= sc
                    else:
                        W[i] *= sc
                    if module.bias is not None:
                        module.bias[i] = module.bias[i] * sc + sh
            for s in foldable:
                folded.add(id(s))
                summary.append({"layer": layer_name, "neurons": list(s.neuron_indices), "mode": s.mode.value, "hook_target": target})
        return {"status": "folded_into_weights", "operations_applied": len(summary), "unfolded_specs_retained": retained,
                "folded_spec_ids": list(folded), "details": summary}


class ProductionSteeringRuntime:
    """
    Standalone applier for a saved preset, without NeuralInjector:
      * compiled mode: layers with compiled vectors get one output hook adding v; all other layers use
        exact neuron hooks;
      * hook mode: exact neuron hooks for every spec (conditional gates with KV-cache latching,
        gate-conditioned modes, pole-specific operations).
    """

    def __init__(self, model: nn.Module, preset_json_or_dict: Union[str, Path, Dict[str, Any]]):
        self.model = model
        raw = json.loads(Path(preset_json_or_dict).read_text(encoding="utf-8")) if isinstance(preset_json_or_dict, (str, Path)) else preset_json_or_dict
        self.bundle, self.specs = SteeringExporter.parse_preset_dict(raw)
        self._hookset = None
        self._handles: List[RemovableHandle] = []
        self._latched_masks: Dict[int, torch.Tensor] = {}

    def reset_latched_gates(self) -> None:
        self._latched_masks.clear()

    def _validate(self, compiled: bool) -> None:
        for s in self.specs:
            _, _, n = LayerDiscovery.resolve_module(self.model, s.layer_name)
            bad = [i for i in s.neuron_indices if not 0 <= i < n]
            if bad:
                raise ValueError(f"Neuron indices {bad} out of range for '{s.layer_name}' ({n} neurons).")
            if s.gate_condition is not None:
                _, _, dn = LayerDiscovery.resolve_module(self.model, s.gate_condition.detector_layer)
                if not 0 <= s.gate_condition.detector_neuron < dn:
                    raise ValueError(f"Detector neuron out of range for '{s.gate_condition.detector_layer}'.")
        if compiled:
            for cv in self.bundle.compiled_vectors:
                W = LayerDiscovery.neuron_write_vectors(self.model, cv.layer_name)
                if W is None or W.shape[1] != len(cv.vector_values):
                    raise ValueError(f"Compiled vector for '{cv.layer_name}' does not match the model.")

    def attach(self, use_compiled_residual_vectors: bool = False) -> None:
        self.detach()
        self._validate(use_compiled_residual_vectors)
        compiled_layers: Set[str] = set()
        if use_compiled_residual_vectors:
            for cv in self.bundle.compiled_vectors:
                module, _, _ = LayerDiscovery.resolve_module(self.model, cv.layer_name)
                vec = torch.tensor(cv.vector_values, dtype=torch.float64)
                compiled_layers.add(cv.layer_name)

                def _add(_m, _i, out, v=vec):
                    t = out[0] if isinstance(out, tuple) else out
                    t2 = t + v.to(device=t.device, dtype=t.dtype)
                    return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2

                self._handles.append(module.register_forward_hook(_add))
        by_layer: Dict[str, List[InterventionSpec]] = {}
        for s in self.specs:
            if s.layer_name not in compiled_layers:
                by_layer.setdefault(s.layer_name, []).append(s)
        if by_layer:
            hs, cond = build_intervention_hookset(self.model, by_layer, self._latched_masks)
            self._handles.append(self.model.register_forward_pre_hook(lambda _m, _i: cond.clear()))
            hs.attach()
            self._hookset = hs

    def detach(self) -> None:
        if self._hookset is not None:
            self._hookset.detach()
            self._hookset = None
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextmanager
    def active(self, use_compiled_residual_vectors: bool = False):
        self.reset_latched_gates()
        self.attach(use_compiled_residual_vectors)
        try:
            yield self
        finally:
            self.detach()
            self.reset_latched_gates()
