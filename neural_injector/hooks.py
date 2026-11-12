from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .discovery import LayerDiscovery

# (activation, gate pre-activation or None) -> modified activation
ModifierFn = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
# (activation, gate pre-activation or None, routed token index or None) -> None
CaptureFn = Callable[[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], None]


@dataclass
class ActivationSite:
    """Where a layer's neuron activations live in the forward pass and how to read its gate."""
    name: str
    module: nn.Module
    target: str                       # "input" (pre-hook on down_proj), "output", "moe_expert"
    num_neurons: int
    expert_idx: Optional[int]
    is_conv2d: bool
    gate_module: Optional[nn.Module]
    gate_kind: str                    # "full", "first_half", "moe", "none"

    @property
    def gated(self) -> bool:
        return self.gate_kind != "none"


def resolve_site(model: nn.Module, layer_name: str) -> ActivationSite:
    module, target, n = LayerDiscovery.resolve_module(model, layer_name)
    gate_mod, gate_kind = LayerDiscovery.gate_source(model, layer_name)
    return ActivationSite(
        name=layer_name,
        module=module,
        target=target,
        num_neurons=n,
        expert_idx=LayerDiscovery.expert_index(model, layer_name),
        is_conv2d=isinstance(module, nn.Conv2d),
        gate_module=gate_mod,
        gate_kind=gate_kind,
    )


class HookSet:
    """
    Builds and attaches forward hooks for activation captures and activation modifiers.
    At each site, modifiers run first (in registration order), then captures see the final tensor.
    Gate pre-activations are read from the gate projection in the same forward pass. Experts of one
    MoE parameter container share a single container hook so their modifications compose.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._sites: Dict[str, ActivationSite] = {}
        self._modifiers: Dict[str, List[ModifierFn]] = {}
        self._captures: Dict[str, List[Tuple[CaptureFn, bool, bool]]] = {}  # (fn, need_gate, dense_moe)
        self._need_gate: Dict[str, bool] = {}
        self._handles: List[RemovableHandle] = []
        self._gate_store: Dict[str, torch.Tensor] = {}

    def site(self, layer_name: str) -> ActivationSite:
        if layer_name not in self._sites:
            self._sites[layer_name] = resolve_site(self.model, layer_name)
        return self._sites[layer_name]

    def add_modifier(self, layer_name: str, fn: ModifierFn, need_gate: bool = False) -> None:
        s = self.site(layer_name)
        if need_gate and not s.gated:
            raise ValueError(f"Layer '{layer_name}' has no gate projection; gate-conditioned interventions are undefined.")
        self._modifiers.setdefault(layer_name, []).append(fn)
        self._need_gate[layer_name] = self._need_gate.get(layer_name, False) or need_gate

    def add_capture(self, layer_name: str, fn: CaptureFn, need_gate: bool = False, dense_moe: bool = True) -> None:
        s = self.site(layer_name)
        if need_gate and not s.gated:
            raise ValueError(f"Layer '{layer_name}' has no gate projection.")
        self._captures.setdefault(layer_name, []).append((fn, need_gate, dense_moe))
        self._need_gate[layer_name] = self._need_gate.get(layer_name, False) or need_gate

    # ------------------------------------------------------------------
    def _run_site(self, name: str, act: torch.Tensor, gate: Optional[torch.Tensor], routed: Optional[torch.Tensor]) -> torch.Tensor:
        for fn in self._modifiers.get(name, []):
            act = fn(act, gate)
        return act

    def _run_captures(self, name: str, act: torch.Tensor, gate: Optional[torch.Tensor], routed: Optional[torch.Tensor], total_tokens: Optional[int]) -> None:
        for fn, _ng, dense in self._captures.get(name, []):
            if routed is not None and dense:
                d_act = torch.zeros((total_tokens, act.shape[-1]), dtype=act.dtype, device=act.device)
                d_act[routed] = act
                d_gate = None
                if gate is not None:
                    d_gate = torch.zeros_like(d_act)
                    d_gate[routed] = gate.to(d_gate.dtype)
                fn(d_act, d_gate, None)
            else:
                fn(act, gate, routed)

    def attach(self) -> None:
        self.detach()
        names = set(self._modifiers) | set(self._captures)
        moe_groups: Dict[int, List[str]] = {}
        for name in names:
            s = self.site(name)
            if s.target == "moe_expert":
                moe_groups.setdefault(id(s.module), []).append(name)
                continue
            if self._need_gate.get(name):
                self._attach_gate_reader(s)
            if s.target == "input":
                self._handles.append(s.module.register_forward_pre_hook(self._make_pre(name)))
            else:
                self._handles.append(s.module.register_forward_hook(self._make_post(name)))
        for _mid, group in moe_groups.items():
            self._handles.append(self.site(group[0]).module.register_forward_hook(self._make_moe(group)))

    def _attach_gate_reader(self, s: ActivationSite) -> None:
        name = s.name

        def _g(_m: nn.Module, _inp: Tuple[Any, ...], out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            if s.gate_kind == "first_half":
                t = t[..., : t.shape[-1] // 2]
            self._gate_store[name] = t.detach()

        self._handles.append(s.gate_module.register_forward_hook(_g))

    def _make_pre(self, name: str):
        def _pre(_m: nn.Module, inputs: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return None
            gate = self._gate_store.pop(name, None) if self._need_gate.get(name) else None
            if gate is not None and gate.shape != inputs[0].shape:
                gate = None
            if self._need_gate.get(name) and gate is None:
                raise RuntimeError(f"Gate activations for '{name}' were not observed in this forward pass.")
            act = self._run_site(name, inputs[0], gate, None)
            self._run_captures(name, act, gate, None, None)
            if self._modifiers.get(name):
                return (act,) + tuple(inputs[1:])
            return None
        return _pre

    def _make_post(self, name: str):
        def _post(_m: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            t = output[0] if isinstance(output, tuple) else output
            if not isinstance(t, torch.Tensor):
                return None
            gate = self._gate_store.pop(name, None) if self._need_gate.get(name) else None
            act = self._run_site(name, t, gate, None)
            self._run_captures(name, act, gate, None, None)
            if self._modifiers.get(name):
                return (act,) + tuple(output[1:]) if isinstance(output, tuple) else act
            return None
        return _post

    def _make_moe(self, group: List[str]):
        by_expert: Dict[int, List[str]] = {}
        for name in group:
            by_expert.setdefault(self.site(name).expert_idx, []).append(name)
        any_mod = any(self._modifiers.get(n) for n in group)

        def _post(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            total = inputs[0].shape[0] if inputs and isinstance(inputs[0], torch.Tensor) else None

            def _expert_fn(act: torch.Tensor, gate: torch.Tensor, token_idx: torch.Tensor, names: List[str]) -> torch.Tensor:
                for nm in names:
                    act = self._run_site(nm, act, gate, token_idx)
                for nm in names:
                    self._run_captures(nm, act, gate, token_idx, total)
                return act

            fns = {e: (lambda a, g, t, nm=names: _expert_fn(a, g, t, nm)) for e, names in by_expert.items()}
            if any_mod:
                new = LayerDiscovery.recompute_moe_container(module, inputs, fns)
                if new is None:
                    return None
                if isinstance(output, tuple):
                    return (new.to(output[0].dtype),) + tuple(output[1:])
                return new.to(output.dtype)
            for e, token_idx, _pos, gate, _up, act in LayerDiscovery.iter_moe_experts(module, inputs):
                if e in fns:
                    fns[e](act, gate, token_idx)
            return None
        return _post

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._gate_store.clear()

    @contextmanager
    def active(self) -> Iterator["HookSet"]:
        self.attach()
        try:
            yield self
        finally:
            self.detach()


def extract_logits(out: Any) -> torch.Tensor:
    """Logits tensor from a model output (HF ModelOutput, tuple or tensor)."""
    t = out.logits if hasattr(out, "logits") else out
    if isinstance(t, (tuple, list)):
        t = t[0]
    if not isinstance(t, torch.Tensor):
        raise TypeError("Model output does not contain a logits tensor.")
    return t


def default_forward(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        kw = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "pixel_values", "inputs_embeds")}
        return model(**kw) if kw else model(**inputs)
    return model(inputs)


def flat_neuron_view(act: torch.Tensor, is_conv2d: bool) -> torch.Tensor:
    """[B, S, D] / [B, D] / [B, C, H, W] -> [positions, D]."""
    if is_conv2d and act.ndim == 4:
        return act.permute(0, 2, 3, 1).reshape(-1, act.shape[1])
    return act.reshape(-1, act.shape[-1])


def valid_position_mask(attention_mask: Optional[torch.Tensor], n_positions: int, device: torch.device) -> torch.Tensor:
    if attention_mask is not None and attention_mask.numel() == n_positions:
        return attention_mask.reshape(-1).to(device=device, dtype=torch.bool)
    return torch.ones(n_positions, dtype=torch.bool, device=device)
