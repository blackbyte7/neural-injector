from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discovery import LayerDiscovery
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
    High-efficiency vectorized tensor operations that manipulate neuron values
    during forward passes with minimal memory or latency overhead.
    Supports scalar values, native per-neuron adaptive thresholds/values, and
    closed-loop dynamic concept gating (`InterventionMode.CONDITIONAL_GATE`),
    including stateful KV-cache latching (`sticky_on_decode`) across single-token
    autoregressive generation steps.
    """

    @staticmethod
    def compute_detector_mask(
        detector_acts: torch.Tensor,
        condition: GateConditionSpec,
        is_conv2d: bool = False,
        latched_masks: Optional[Dict[int, torch.Tensor]] = None,
        spec_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Evaluates whether `condition.detector_neuron` in `detector_acts` satisfies
        the pole and threshold trigger criteria.
        Returns a boolean mask of shape `[B, S, 1]` (3D), `[B, 1]` (2D), or `[B, 1, H, W]` (4D).

        When `condition.sequence_scope == "any_token"` and `condition.sticky_on_decode == True`:
          - On a multi-token prompt prefill (`S > 1`), stores the sequence-level trigger
            `[B, 1, 1]` into `latched_masks[spec_id]`.
          - On subsequent single-token KV-cached decoding steps (`S == 1`), OR-combines
            the current step's trigger with `latched_masks[spec_id]`, keeping the gate
            latched open for the remainder of the generated response.
        """
        n_idx = condition.detector_neuron
        if is_conv2d and detector_acts.ndim == 4:
            det_slice = detector_acts[:, n_idx: n_idx + 1, :, :]
        else:
            det_slice = detector_acts[..., n_idx: n_idx + 1]

        thresh = abs(float(condition.threshold))
        if condition.detector_pole == PoleType.POSITIVE:
            mask = det_slice > thresh
        elif condition.detector_pole == PoleType.NEGATIVE:
            mask = det_slice < -thresh
        else:
            mask = det_slice.abs() > thresh

        if condition.sequence_scope == "any_token" and mask.ndim == 3:
            step_trigger = mask.any(dim=1, keepdim=True)  # [B, 1, 1]
            if (
                condition.sticky_on_decode
                and latched_masks is not None
                and spec_id is not None
            ):
                seq_len = mask.shape[1]
                if seq_len > 1:
                    latched_masks[spec_id] = step_trigger.detach().clone()
                else:
                    prev_latch = latched_masks.get(spec_id)
                    if prev_latch is not None and prev_latch.shape[0] == step_trigger.shape[0]:
                        step_trigger = step_trigger | prev_latch.to(
                            device=step_trigger.device, dtype=torch.bool
                        )
                    latched_masks[spec_id] = step_trigger.detach().clone()

            mask = step_trigger.expand_as(mask)

        return mask

    @staticmethod
    def apply(
        tensor: torch.Tensor,
        specs: List[InterventionSpec],
        is_conv2d: bool = False,
        conditional_masks: Optional[Dict[int, torch.Tensor]] = None,
        latched_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Applies a list of InterventionSpecs to an activation tensor.
        Supports shapes:
          - [batch, seq_len, hidden_dim] (Transformers / Sequence models)
          - [batch, hidden_dim]          (Standard MLPs / Routed MoE)
          - [batch, channels, H, W]      (CNNs when is_conv2d=True)
        """
        if not specs:
            return tensor

        # Clone once to avoid corrupting autograd graphs or shared views when modifying in-place
        out = tensor.clone()
        device = out.device
        dtype = out.dtype

        for spec in specs:
            if not spec.enabled or not spec.neuron_indices:
                continue

            # Resolve optional closed-loop conditional trigger mask
            cond_mask: Optional[torch.Tensor] = None
            if spec.gate_condition is not None or spec.mode == InterventionMode.CONDITIONAL_GATE:
                if conditional_masks is not None and id(spec) in conditional_masks:
                    cond_mask = conditional_masks[id(spec)]
                elif (
                    spec.gate_condition is not None
                    and spec.gate_condition.detector_layer == spec.layer_name
                ):
                    # Same-layer detector: evaluate directly on the incoming activation tensor
                    cond_mask = InterventionKernel.compute_detector_mask(
                        tensor,
                        spec.gate_condition,
                        is_conv2d=is_conv2d,
                        latched_masks=latched_masks,
                        spec_id=id(spec),
                    )
                if cond_mask is None or not cond_mask.any():
                    continue

            idx = torch.as_tensor(spec.neuron_indices, dtype=torch.long, device=device)

            # Extract the slice corresponding to target neurons
            if is_conv2d and out.ndim == 4:
                # Shape: [B, C, H, W] -> index along channel dimension (dim=1)
                sub = out[:, idx, :, :]
                modified = InterventionKernel._transform_slice(
                    sub, spec, dtype, is_conv2d=True, cond_mask=cond_mask
                )
                out[:, idx, :, :] = modified

            elif out.ndim == 3:
                # Shape: [B, S, D] -> Transformer sequence activation
                if spec.token_positions is not None:
                    seq_len = out.shape[1]
                    # Normalize negative indices (e.g. -1 -> seq_len - 1) and filter out-of-bounds
                    # positions so single-token KV-cached decode steps (seq_len == 1) never crash.
                    valid_positions = [
                        (seq_len + p if p < 0 else p)
                        for p in spec.token_positions
                        if 0 <= (seq_len + p if p < 0 else p) < seq_len
                    ]
                    if not valid_positions:
                        continue
                    pos_idx = torch.as_tensor(valid_positions, dtype=torch.long, device=device)
                    sub = out[:, pos_idx[:, None], idx]
                    pos_cond = (
                        cond_mask[:, pos_idx, :]
                        if (cond_mask is not None and cond_mask.ndim == 3)
                        else cond_mask
                    )
                    modified = InterventionKernel._transform_slice(
                        sub, spec, dtype, cond_mask=pos_cond
                    )
                    out[:, pos_idx[:, None], idx] = modified
                else:
                    sub = out[:, :, idx]
                    modified = InterventionKernel._transform_slice(
                        sub, spec, dtype, cond_mask=cond_mask
                    )
                    out[:, :, idx] = modified

            elif out.ndim == 2:
                # Shape: [B, D] -> Standard 2D batch activation
                sub = out[:, idx]
                modified = InterventionKernel._transform_slice(
                    sub, spec, dtype, cond_mask=cond_mask
                )
                out[:, idx] = modified

            else:
                # Generic fallback: assume last dimension is the neuron basis
                sub = out[..., idx]
                modified = InterventionKernel._transform_slice(
                    sub, spec, dtype, cond_mask=cond_mask
                )
                out[..., idx] = modified

        return out

    @staticmethod
    def _resolve_value_tensor(
        sub: torch.Tensor,
        spec: InterventionSpec,
        dtype: torch.dtype,
        is_conv2d: bool = False,
    ) -> torch.Tensor:
        """
        Resolves either a broadcastable per-neuron value tensor (from `spec.per_neuron_values`)
        or a uniform scalar tensor (from `spec.value`) matching `sub`.
        """
        if spec.per_neuron_values is not None and len(spec.per_neuron_values) == len(spec.neuron_indices):
            val_1d = torch.as_tensor(spec.per_neuron_values, dtype=dtype, device=sub.device)
            if is_conv2d and sub.ndim == 4:
                return val_1d.view(1, -1, 1, 1).expand_as(sub)
            else:
                view_shape = [1] * (sub.ndim - 1) + [-1]
                return val_1d.view(*view_shape).expand_as(sub)
        return torch.full_like(sub, float(spec.value))

    @staticmethod
    def _transform_slice(
        sub: torch.Tensor,
        spec: InterventionSpec,
        dtype: torch.dtype,
        is_conv2d: bool = False,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes the specific pole-aware mathematical operation on the selected neuron slice."""
        if spec.custom_fn is not None:
            transformed = spec.custom_fn(sub).to(dtype)
            if cond_mask is not None:
                return torch.where(cond_mask.to(device=sub.device), transformed, sub)
            return transformed

        # Build pole mask if targeting only POSITIVE (> 0) or NEGATIVE (< 0) neuron pole
        if spec.pole == PoleType.POSITIVE:
            pole_mask = sub > 0
        elif spec.pole == PoleType.NEGATIVE:
            pole_mask = sub < 0
        else:
            pole_mask = torch.ones_like(sub, dtype=torch.bool)

        if cond_mask is not None:
            aligned_cond = cond_mask.to(device=sub.device, dtype=torch.bool)
            if aligned_cond.shape == pole_mask.shape or aligned_cond.ndim == pole_mask.ndim:
                pole_mask = pole_mask & aligned_cond.expand_as(pole_mask)
            else:
                pole_mask = pole_mask & aligned_cond.any()

        effective_mode = (
            spec.gate_condition.action_mode
            if (spec.mode == InterventionMode.CONDITIONAL_GATE and spec.gate_condition is not None)
            else (
                InterventionMode.ADDITIVE_STEER
                if spec.mode == InterventionMode.CONDITIONAL_GATE
                else spec.mode
            )
        )

        if effective_mode == InterventionMode.ZERO_ABLATE:
            return torch.where(pole_mask, torch.zeros_like(sub), sub)

        elif effective_mode == InterventionMode.POLE_RECTIFY:
            return torch.where(pole_mask, sub, torch.zeros_like(sub))

        val_tensor = InterventionKernel._resolve_value_tensor(
            sub, spec, dtype, is_conv2d=is_conv2d
        )

        if effective_mode == InterventionMode.HARD_CLAMP:
            return torch.where(pole_mask, val_tensor, sub)

        elif effective_mode == InterventionMode.SCALE:
            return torch.where(pole_mask, sub * val_tensor, sub)

        elif effective_mode == InterventionMode.ADDITIVE_STEER:
            return torch.where(pole_mask, sub + val_tensor, sub)

        elif effective_mode == InterventionMode.WEAK_PRUNE:
            weak_mask = pole_mask & (sub.abs() < val_tensor.abs())
            return torch.where(weak_mask, torch.zeros_like(sub), sub)

        return sub


class AblationMaskOptimizer:
    """
    Subspace / Multi-Neuron Synergistic Ablation Optimizer (Cammarata: 'Understanding Compounds').
    Because backup circuits ('circuit ecology') often compensate when a single neuron is ablated,
    `AblationMaskOptimizer` finds the minimal coalition of `k` neurons in `layer_name` whose
    joint ablation maximizes causal divergence or flips a target token behavior.

    Supports two search strategies:
      1. `method="greedy"`: Fast candidate pre-filtering (by activation magnitude * weight norm)
         followed by exact forward-pass greedy subset selection.
      2. `method="l1_optim"`: Differentiable continuous gate mask `m in [0, 1]^D` optimized via
         Adam with an L1 sparsity penalty, discretized to the top `max_subset_size` neurons.
    """

    def __init__(self, injector: NeuralInjector):
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
        l1_steps: int = 15,
        l1_lr: float = 0.25,
        l1_penalty: float = 0.01,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> SynergisticAblationResult:
        def _default_fwd(m: nn.Module, inp: Any) -> Any:
            if isinstance(inp, dict):
                return m(inp["input_ids"]) if "input_ids" in inp else m(**inp)
            return m(inp)

        run_fwd = forward_fn or _default_fwd
        if attention_mask is None and isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")

        mod, hook_target, num_neurons = LayerDiscovery.resolve_module(self.model, layer_name)
        k_max = min(max_subset_size, num_neurons)
        pool_k = min(max(candidate_pool_size, k_max * 2), num_neurons)

        was_training = self.model.training
        self.model.eval()
        try:
            # Step 1: Clean baseline forward pass to capture clean logits & active neuron magnitudes
            captured_acts: List[torch.Tensor] = []
            h_cap = (
                mod.register_forward_pre_hook(
                    lambda _m, inp: (
                        captured_acts.append(inp[0].detach())
                        if inp and isinstance(inp[0], torch.Tensor)
                        else None
                    )
                )
                if hook_target == "input"
                else mod.register_forward_hook(
                    lambda _m, _inp, out: captured_acts.append(
                        (out[0] if isinstance(out, tuple) else out).detach()
                    )
                )
            )

            try:
                with torch.no_grad(), self.injector.without_interventions():
                    clean_out = run_fwd(self.model, inputs)
                    clean_logits = (
                        clean_out.logits if hasattr(clean_out, "logits") else clean_out
                    ).detach().float()
                    clean_log_probs = F.log_softmax(clean_logits, dim=-1)
                    clean_probs = clean_log_probs.exp()
            finally:
                h_cap.remove()

            act_flat = captured_acts[-1].reshape(-1, num_neurons).float()
            if attention_mask is not None:
                mask_flat = attention_mask.reshape(-1).to(
                    device=act_flat.device, dtype=torch.bool
                )
                if mask_flat.numel() == act_flat.shape[0] and mask_flat.any():
                    act_flat = act_flat[mask_flat]
            mean_abs_act = act_flat.abs().mean(dim=0)

            # Weight by outgoing L2 norm if available
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                w = mod.weight.detach().float()
                if w.ndim == 3 and w.shape[-1] == num_neurons:
                    wnorm = torch.norm(w, p=2, dim=1).mean(dim=0)
                elif mod.__class__.__name__ == "Conv1D" and w.shape[0] == num_neurons:
                    wnorm = torch.norm(w, p=2, dim=1)
                elif w.ndim == 2 and w.shape[1] == num_neurons:
                    wnorm = torch.norm(w, p=2, dim=0)
                elif w.ndim == 2 and w.shape[0] == num_neurons:
                    wnorm = torch.norm(w, p=2, dim=1)
                else:
                    wnorm = torch.ones_like(mean_abs_act)
            else:
                wnorm = torch.ones_like(mean_abs_act)

            importance = mean_abs_act * wnorm.to(mean_abs_act.device)
            _, top_cand_tensor = torch.topk(importance, k=pool_k)
            candidate_indices = [int(i) for i in top_cand_tensor.tolist()]

            def _masked_kl(mod_log_probs: torch.Tensor) -> torch.Tensor:
                token_kl = (clean_probs * (clean_log_probs - mod_log_probs)).sum(dim=-1)
                if attention_mask is not None and token_kl.shape == attention_mask.shape:
                    m_bool = attention_mask.to(device=token_kl.device, dtype=torch.bool)
                    if m_bool.any():
                        return token_kl[m_bool].mean()
                return token_kl.mean()

            def _score_logits(mod_logits: torch.Tensor) -> float:
                mod_log_probs = F.log_softmax(mod_logits.float(), dim=-1)
                kl = float(_masked_kl(mod_log_probs).item())
                if target_token_ids:
                    drop = float(
                        (
                            clean_logits[..., target_token_ids].mean()
                            - mod_logits.float()[..., target_token_ids].mean()
                        ).item()
                    )
                    return kl + max(drop, 0.0)
                return kl

            selected: List[int] = []
            stepwise_scores: List[float] = []

            if method == "l1_optim":
                # Differentiable continuous mask optimization over `candidate_indices`
                device = clean_logits.device
                logits_alpha = nn.Parameter(
                    torch.full((len(candidate_indices),), 2.0, device=device)
                )
                optimizer = torch.optim.Adam([logits_alpha], lr=l1_lr)
                cand_idx_t = torch.as_tensor(candidate_indices, dtype=torch.long, device=device)

                def _apply_soft_gate(x_in: torch.Tensor) -> torch.Tensor:
                    x = x_in.clone()
                    sub = x[..., cand_idx_t]
                    gate_vals = torch.sigmoid(logits_alpha).to(dtype=x.dtype, device=x.device)
                    view_shape = [1] * (sub.ndim - 1) + [-1]
                    g_broad = gate_vals.view(*view_shape).expand_as(sub)
                    if pole == PoleType.POSITIVE:
                        gated = torch.where(sub > 0, sub * g_broad, sub)
                    elif pole == PoleType.NEGATIVE:
                        gated = torch.where(sub < 0, sub * g_broad, sub)
                    else:
                        gated = sub * g_broad
                    x[..., cand_idx_t] = gated
                    return x

                def _mask_pre(_m: nn.Module, inp: Any):
                    if not inp or not isinstance(inp[0], torch.Tensor):
                        return inp
                    return (_apply_soft_gate(inp[0]),) + inp[1:]

                def _mask_fwd(_m: nn.Module, _inp: Any, out: Any):
                    t0 = out[0] if isinstance(out, tuple) else out
                    if not isinstance(t0, torch.Tensor):
                        return out
                    x = _apply_soft_gate(t0)
                    return (x,) + out[1:] if isinstance(out, tuple) else x

                h_opt = (
                    mod.register_forward_pre_hook(_mask_pre)
                    if hook_target == "input"
                    else mod.register_forward_hook(_mask_fwd)
                )
                try:
                    with self.injector.without_interventions(), torch.enable_grad():
                        for _ in range(l1_steps):
                            optimizer.zero_grad()
                            out_step = run_fwd(self.model, inputs)
                            step_logits = (
                                out_step.logits if hasattr(out_step, "logits") else out_step
                            ).float()
                            step_log_probs = F.log_softmax(step_logits, dim=-1)
                            kl_term = _masked_kl(step_log_probs)
                            if target_token_ids:
                                t_drop = (
                                    clean_logits[..., target_token_ids].mean()
                                    - step_logits[..., target_token_ids].mean()
                                )
                                objective = kl_term + t_drop
                            else:
                                objective = kl_term
                            # Maximize objective while minimizing number of ablated neurons (1 - gate)
                            gates = torch.sigmoid(logits_alpha)
                            ablation_mass = (1.0 - gates).sum()
                            loss = -objective + l1_penalty * ablation_mass
                            loss.backward()
                            optimizer.step()
                finally:
                    h_opt.remove()

                with torch.no_grad():
                    ablation_strength = (1.0 - torch.sigmoid(logits_alpha)).detach()
                    _, best_local = torch.topk(
                        ablation_strength, k=min(k_max, len(candidate_indices))
                    )
                    selected = [candidate_indices[int(i)] for i in best_local.tolist()]
                    for step_i in range(1, len(selected) + 1):
                        spec = InterventionSpec(
                            layer_name=layer_name,
                            neuron_indices=selected[:step_i],
                            mode=InterventionMode.ZERO_ABLATE,
                            pole=pole,
                        )
                        with self.injector.temporary_interventions([spec]):
                            m_out = run_fwd(self.model, inputs)
                            m_logits = m_out.logits if hasattr(m_out, "logits") else m_out
                            stepwise_scores.append(round(_score_logits(m_logits), 5))
            else:
                # Greedy forward subset selection
                remaining = list(candidate_indices)
                with torch.no_grad():
                    for _ in range(k_max):
                        best_n: Optional[int] = None
                        best_score = -1.0
                        for cand in remaining:
                            trial_set = selected + [cand]
                            spec = InterventionSpec(
                                layer_name=layer_name,
                                neuron_indices=trial_set,
                                mode=InterventionMode.ZERO_ABLATE,
                                pole=pole,
                            )
                            with self.injector.temporary_interventions([spec]):
                                m_out = run_fwd(self.model, inputs)
                                m_logits = m_out.logits if hasattr(m_out, "logits") else m_out
                                sc = _score_logits(m_logits)
                            if sc > best_score:
                                best_score = sc
                                best_n = cand
                        if best_n is not None:
                            selected.append(best_n)
                            remaining.remove(best_n)
                            stepwise_scores.append(round(best_score, 5))

            # Measure joint KL vs sum of individual KLs to quantify backup-circuit synergy
            with torch.no_grad():
                joint_spec = InterventionSpec(
                    layer_name=layer_name,
                    neuron_indices=selected,
                    mode=InterventionMode.ZERO_ABLATE,
                    pole=pole,
                )
                with self.injector.temporary_interventions([joint_spec]):
                    j_out = run_fwd(self.model, inputs)
                    j_logits = (j_out.logits if hasattr(j_out, "logits") else j_out).float()
                    j_log_probs = F.log_softmax(j_logits, dim=-1)
                    joint_kl = float(_masked_kl(j_log_probs).item())
                    target_shift = (
                        float(
                            (
                                clean_logits[..., target_token_ids].mean()
                                - j_logits[..., target_token_ids].mean()
                            ).item()
                        )
                        if target_token_ids
                        else 0.0
                    )

                indiv_kl_sum = 0.0
                for n_idx in selected:
                    indiv_kl_sum += self.injector.measure_ablation_kl_divergence(
                        layer_name, n_idx, inputs=inputs, pole=pole, forward_fn=run_fwd
                    )

            synergy_ratio = joint_kl / max(indiv_kl_sum, 1e-9) if selected else 1.0
            return SynergisticAblationResult(
                layer_name=layer_name,
                method=method,
                selected_neurons=selected,
                stepwise_scores=stepwise_scores,
                joint_ablation_kl=round(joint_kl, 5),
                sum_of_individual_kls=round(indiv_kl_sum, 5),
                synergy_ratio=round(synergy_ratio, 4),
                target_logit_shift=round(target_shift, 5),
            )
        finally:
            if was_training:
                self.model.train()
