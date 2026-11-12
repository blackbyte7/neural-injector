from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hooks import HookSet, default_forward, extract_logits, flat_neuron_view, valid_position_mask
from .types import (
    GateConditionSpec,
    InterventionMode,
    InterventionSpec,
    PoleType,
    SynergisticAblationResult,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector


class InterventionKernel:
    """
    Vectorised neuron manipulations applied inside forward hooks.
    Supported activation shapes: [B, S, D], [B, D] (also routed MoE rows), [B, C, H, W] (conv).
    Gate-conditioned modes (DOUBLE_NEGATIVE_ABLATE, EXTREME_CULL with condition_signal="gate")
    receive the gate pre-activation tensor of the same shape.
    """

    @staticmethod
    def compute_detector_mask(
        detector_acts: torch.Tensor,
        condition: GateConditionSpec,
        is_conv2d: bool = False,
        latched_masks: Optional[Dict[int, torch.Tensor]] = None,
        spec_id: Optional[int] = None,
    ) -> torch.Tensor:
        n_idx = condition.detector_neuron
        if is_conv2d and detector_acts.ndim == 4:
            det = detector_acts[:, n_idx: n_idx + 1, :, :]
        else:
            det = detector_acts[..., n_idx: n_idx + 1]
        thresh = abs(float(condition.threshold))
        if condition.detector_pole == PoleType.POSITIVE:
            mask = det > thresh
        elif condition.detector_pole == PoleType.NEGATIVE:
            mask = det < -thresh
        else:
            mask = det.abs() > thresh

        if condition.sequence_scope == "any_token" and mask.ndim == 3:
            step = mask.any(dim=1, keepdim=True)
            if condition.sticky_on_decode and latched_masks is not None and spec_id is not None:
                if mask.shape[1] > 1:
                    latched_masks[spec_id] = step.detach().clone()
                else:
                    prev = latched_masks.get(spec_id)
                    if prev is not None and prev.shape[0] == step.shape[0]:
                        step = step | prev.to(device=step.device, dtype=torch.bool)
                    latched_masks[spec_id] = step.detach().clone()
            mask = step.expand_as(mask)
        return mask

    @staticmethod
    def apply(
        tensor: torch.Tensor,
        specs: List[InterventionSpec],
        is_conv2d: bool = False,
        conditional_masks: Optional[Dict[int, torch.Tensor]] = None,
        latched_masks: Optional[Dict[int, torch.Tensor]] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not specs:
            return tensor
        out = tensor.clone()
        device, dtype = out.device, out.dtype

        for spec in specs:
            if not spec.enabled or not spec.neuron_indices:
                continue
            if spec.needs_gate_signal and gate is None:
                raise RuntimeError(f"{spec.mode.value} on '{spec.layer_name}' requires the gate pre-activation.")

            cond_mask: Optional[torch.Tensor] = None
            if spec.mode == InterventionMode.CONDITIONAL_GATE or spec.gate_condition is not None:
                if conditional_masks is not None and id(spec) in conditional_masks:
                    cond_mask = conditional_masks[id(spec)]
                elif spec.gate_condition is not None and spec.gate_condition.detector_layer == spec.layer_name:
                    cond_mask = InterventionKernel.compute_detector_mask(
                        tensor, spec.gate_condition, is_conv2d=is_conv2d,
                        latched_masks=latched_masks, spec_id=id(spec),
                    )
                if cond_mask is None or not bool(cond_mask.any()):
                    continue

            idx = torch.as_tensor(spec.neuron_indices, dtype=torch.long, device=device)
            if is_conv2d and out.ndim == 4:
                sub = out[:, idx, :, :]
                g_sub = gate[:, idx, :, :] if gate is not None else None
                out[:, idx, :, :] = InterventionKernel._transform(sub, spec, dtype, True, cond_mask, g_sub)
            elif out.ndim == 3 and spec.token_positions is not None:
                seq_len = out.shape[1]
                valid = [(seq_len + p if p < 0 else p) for p in spec.token_positions]
                valid = [p for p in valid if 0 <= p < seq_len]
                if not valid:
                    continue
                pos = torch.as_tensor(valid, dtype=torch.long, device=device)
                sub = out[:, pos[:, None], idx]
                g_sub = gate[:, pos[:, None], idx] if gate is not None else None
                c_sub = cond_mask[:, pos, :] if (cond_mask is not None and cond_mask.ndim == 3) else cond_mask
                out[:, pos[:, None], idx] = InterventionKernel._transform(sub, spec, dtype, False, c_sub, g_sub)
            else:
                if spec.token_positions is not None and out.ndim != 3:
                    raise ValueError("token_positions requires [B, S, D] activations.")
                sub = out[..., idx]
                g_sub = gate[..., idx] if gate is not None else None
                out[..., idx] = InterventionKernel._transform(sub, spec, dtype, False, cond_mask, g_sub)
        return out

    @staticmethod
    def _values(sub: torch.Tensor, spec: InterventionSpec, dtype: torch.dtype, is_conv2d: bool) -> torch.Tensor:
        if spec.per_neuron_values is not None:
            if len(spec.per_neuron_values) != len(spec.neuron_indices):
                raise ValueError("per_neuron_values must align with neuron_indices")
            v = torch.as_tensor(spec.per_neuron_values, dtype=dtype, device=sub.device)
            shape = [1, -1, 1, 1] if (is_conv2d and sub.ndim == 4) else [1] * (sub.ndim - 1) + [-1]
            return v.view(*shape).expand_as(sub)
        return torch.full_like(sub, float(spec.value))

    @staticmethod
    def _transform(
        sub: torch.Tensor,
        spec: InterventionSpec,
        dtype: torch.dtype,
        is_conv2d: bool,
        cond_mask: Optional[torch.Tensor],
        gate_sub: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if spec.custom_fn is not None:
            new = spec.custom_fn(sub).to(dtype)
            return torch.where(cond_mask.to(sub.device), new, sub) if cond_mask is not None else new

        if spec.pole == PoleType.POSITIVE:
            mask = sub > 0
        elif spec.pole == PoleType.NEGATIVE:
            mask = sub < 0
        else:
            mask = torch.ones_like(sub, dtype=torch.bool)
        if cond_mask is not None:
            mask = mask & cond_mask.to(device=sub.device, dtype=torch.bool).expand_as(mask)

        mode = spec.mode
        if mode == InterventionMode.CONDITIONAL_GATE:
            if spec.gate_condition is None:
                raise ValueError("CONDITIONAL_GATE requires gate_condition")
            mode = spec.gate_condition.action_mode

        zeros = torch.zeros_like(sub)
        if mode == InterventionMode.ZERO_ABLATE:
            return torch.where(mask, zeros, sub)
        if mode == InterventionMode.POLE_RECTIFY:
            return torch.where(mask, sub, zeros)
        if mode == InterventionMode.DOUBLE_NEGATIVE_ABLATE:
            # a > 0 with g < 0 implies u < 0 because act(g) < 0 for g < 0
            return torch.where(mask & (sub > 0) & (gate_sub.to(sub.device) < 0), zeros, sub)

        val = InterventionKernel._values(sub, spec, dtype, is_conv2d)
        if mode == InterventionMode.HARD_CLAMP:
            return torch.where(mask, val, sub)
        if mode == InterventionMode.SCALE:
            return torch.where(mask, sub * val, sub)
        if mode == InterventionMode.ADDITIVE_STEER:
            return torch.where(mask, sub + val, sub)
        if mode == InterventionMode.WEAK_PRUNE:
            return torch.where(mask & (sub.abs() < val.abs()), zeros, sub)
        if mode == InterventionMode.EXTREME_CULL:
            if spec.condition_signal == "gate":
                hit = gate_sub.to(device=sub.device, dtype=val.dtype) > val
            elif spec.pole == PoleType.NEGATIVE:
                hit = sub < -val.abs()
            elif spec.pole == PoleType.POSITIVE:
                hit = sub > val.abs()
            else:
                hit = sub.abs() > val.abs()
            return torch.where(mask & hit, zeros, sub)
        raise ValueError(f"Unsupported intervention mode {mode}")


def build_intervention_hookset(
    model: nn.Module,
    interventions: Dict[str, List[InterventionSpec]],
    latched_masks: Dict[int, torch.Tensor],
    counter: Optional[Dict[str, int]] = None,
):
    """
    HookSet applying `interventions` (with cross-layer conditional-gate detectors and gate signals)
    plus the dict of per-forward detector masks, which the caller must clear at the start of every
    forward pass of the root model.
    """
    hs = HookSet(model)
    cond_masks: Dict[int, torch.Tensor] = {}
    for layer_name, specs in interventions.items():
        for spec in specs:
            if not spec.enabled or spec.gate_condition is None:
                continue
            det = spec.gate_condition.detector_layer
            if det == layer_name:
                continue
            det_site = hs.site(det)

            def _det(act, _g, _r, s=spec, conv=det_site.is_conv2d) -> None:
                cond_masks[id(s)] = InterventionKernel.compute_detector_mask(
                    act.detach(), s.gate_condition, is_conv2d=conv, latched_masks=latched_masks, spec_id=id(s))

            hs.add_capture(det, _det)
    for layer_name, specs in interventions.items():
        active = [s for s in specs if s.enabled]
        if not active:
            continue
        site = hs.site(layer_name)

        def _mod(act, gate, ls=active, conv=site.is_conv2d, nm=layer_name):
            new = InterventionKernel.apply(act, ls, is_conv2d=conv, conditional_masks=cond_masks,
                                           latched_masks=latched_masks, gate=gate)
            if counter is not None:
                counter[nm] = counter.get(nm, 0) + int((new != act).sum().item())
            return new

        hs.add_modifier(layer_name, _mod, need_gate=any(s.needs_gate_signal for s in active))
    return hs, cond_masks


class AblationMaskOptimizer:
    """
    Finds a small coalition of neurons in one layer whose joint ablation has the largest effect,
    which single-neuron ablation misses when backup circuits compensate.

    Objective: mean KL(P_clean || P_ablated) over valid positions, or, when `target_token_ids` are
    given, the drop in log-probability mass of those tokens (one quantity, never a sum of both).
      * method="greedy": exact forward-pass greedy selection over a candidate pool.
      * method="l1_optim": continuous mask m = sigmoid(alpha) on the pool, alpha initialised at 0
        (m = 0.5, the neutral midpoint, so the objective has gradient from the first step),
        maximising objective - l1_penalty * sum(1 - m); the top `max_subset_size` are then evaluated exactly.
    """

    def __init__(self, injector: "NeuralInjector"):
        self.injector = injector
        self.model = injector.model

    def find_synergistic_ablation_set(
        self,
        layer_name: str,
        inputs: Any,
        max_subset_size: int = 3,
        candidate_pool_size: int = 16,
        method: str = "greedy",
        target_token_ids: Optional[List[int]] = None,
        pole: PoleType = PoleType.BOTH,
        l1_steps: int = 30,
        l1_lr: float = 0.1,
        l1_penalty: float = 0.0,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        candidate_neurons: Optional[List[int]] = None,
    ) -> SynergisticAblationResult:
        run_fwd = forward_fn or default_forward
        if attention_mask is None and isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
        inj = self.injector
        site = HookSet(self.model).site(layer_name)
        n_neurons = site.num_neurons
        k_max = min(max_subset_size, n_neurons)

        was_training = self.model.training
        self.model.eval()
        try:
            captured: List[torch.Tensor] = []
            hs = HookSet(self.model)
            hs.add_capture(layer_name, lambda a, g, r: captured.append(a.detach()))
            with torch.no_grad(), inj.clean_forward(), hs.active():
                clean_logits = extract_logits(run_fwd(self.model, inputs)).float()
            clean_lp = F.log_softmax(clean_logits, dim=-1)
            clean_p = clean_lp.exp()
            n_pos = clean_logits.shape[:-1].numel()
            valid = valid_position_mask(attention_mask, n_pos, clean_logits.device)

            def _objective(logits: torch.Tensor) -> torch.Tensor:
                lp = F.log_softmax(logits.float(), dim=-1)
                if target_token_ids:
                    t_idx = torch.as_tensor(target_token_ids, device=lp.device)
                    clean_mass = torch.logsumexp(clean_lp[..., t_idx], dim=-1).reshape(-1)
                    new_mass = torch.logsumexp(lp[..., t_idx], dim=-1).reshape(-1)
                    return (clean_mass - new_mass)[valid].mean()
                kl = (clean_p * (clean_lp - lp)).sum(dim=-1).reshape(-1)
                return kl[valid].mean()

            if candidate_neurons is not None:
                cand = [int(i) for i in candidate_neurons if 0 <= int(i) < n_neurons]
            else:
                acts = flat_neuron_view(captured[-1].float(), site.is_conv2d)
                act_valid = valid_position_mask(attention_mask, acts.shape[0], acts.device)
                mean_abs = acts[act_valid].abs().mean(dim=0)
                wv = inj.neuron_write_vectors(layer_name)
                wn = wv.float().norm(dim=1).to(mean_abs.device) if wv is not None else torch.ones_like(mean_abs)
                pool = min(max(candidate_pool_size, k_max), n_neurons)
                cand = torch.topk(mean_abs * wn, k=pool).indices.tolist()

            def _score_set(neurons: List[int]) -> float:
                spec = InterventionSpec(layer_name=layer_name, neuron_indices=list(neurons), mode=InterventionMode.ZERO_ABLATE, pole=pole)
                with torch.no_grad(), inj.temporary_interventions([spec]):
                    return float(_objective(extract_logits(run_fwd(self.model, inputs))).item())

            selected: List[int] = []
            stepwise: List[float] = []
            if method == "l1_optim":
                device = clean_logits.device
                alpha = nn.Parameter(torch.zeros(len(cand), device=device))
                opt = torch.optim.Adam([alpha], lr=l1_lr)
                cand_t = torch.as_tensor(cand, dtype=torch.long)

                def _soft(act: torch.Tensor, _g: Optional[torch.Tensor]) -> torch.Tensor:
                    x = act.clone()
                    ci = cand_t.to(x.device)
                    sub = x[..., ci]
                    m = torch.sigmoid(alpha).to(dtype=x.dtype, device=x.device)
                    shape = [1] * (sub.ndim - 1) + [-1]
                    m = m.view(*shape)
                    if pole == PoleType.POSITIVE:
                        sub = torch.where(sub > 0, sub * m, sub)
                    elif pole == PoleType.NEGATIVE:
                        sub = torch.where(sub < 0, sub * m, sub)
                    else:
                        sub = sub * m
                    x[..., ci] = sub
                    return x

                hs_opt = HookSet(self.model)
                hs_opt.add_modifier(layer_name, _soft)
                with inj.clean_forward(), hs_opt.active(), torch.enable_grad():
                    for _ in range(l1_steps):
                        opt.zero_grad()
                        obj = _objective(extract_logits(run_fwd(self.model, inputs)))
                        loss = -obj + l1_penalty * (1.0 - torch.sigmoid(alpha)).sum()
                        loss.backward()
                        opt.step()
                strength = (1.0 - torch.sigmoid(alpha)).detach()
                order = torch.argsort(strength, descending=True)[:k_max].tolist()
                selected = [cand[i] for i in order]
                for s in range(1, len(selected) + 1):
                    stepwise.append(_score_set(selected[:s]))
            elif method == "greedy":
                remaining = list(cand)
                for _ in range(k_max):
                    best_n, best_s = None, float("-inf")
                    for c in remaining:
                        sc = _score_set(selected + [c])
                        if sc > best_s:
                            best_n, best_s = c, sc
                    if best_n is None:
                        break
                    selected.append(best_n)
                    remaining.remove(best_n)
                    stepwise.append(best_s)
            else:
                raise ValueError("method must be 'greedy' or 'l1_optim'")

            joint = _score_set(selected) if selected else 0.0
            indiv = sum(_score_set([n]) for n in selected)
            return SynergisticAblationResult(
                layer_name=layer_name,
                method=method,
                objective="target_logprob_mass_drop" if target_token_ids else "mean_kl",
                selected_neurons=selected,
                stepwise_scores=stepwise,
                joint_score=joint,
                sum_of_individual_scores=indiv,
                synergy_ratio=joint / indiv if indiv > 0 else float("nan"),
            )
        finally:
            if was_training:
                self.model.train()
