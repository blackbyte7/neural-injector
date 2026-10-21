from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discovery import LayerDiscovery
from .types import (
    CircuitEdge,
    CircuitGraph,
    InterventionMode,
    InterventionSpec,
    PoleType,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector


class CircuitTracer:
    """
    Traces structural and causal neuron-to-neuron circuits across layers.
    Designed around the 'Neural Networks Are Already Clean' paradigm:
    treats individual neuron poles as interpretable routing units ('verbs')
    and measures how an upstream pole excites, inhibits, or gates downstream neurons.
    """

    def __init__(self, injector: NeuralInjector):
        self.injector = injector
        self.model = injector.model

    # -------------------------------------------------------------------------
    # 1. Static Structural Virtual Weights (Weight-Space Circuitry)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def compute_virtual_weights(
        self,
        src_layer: str,
        dst_layer: str,
        src_neuron: int,
        apply_pre_norm_scaling: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the direct residual-stream virtual weights from `src_neuron` in `src_layer`
        to all neurons in `dst_layer`.

        Returns three 1D tensors of shape [num_dst_neurons]:
          - virtual_up:   <W_up^{(dst)}[j, :], W_down^{(src)}[:, src_neuron]>
          - virtual_gate: <W_gate^{(dst)}[j, :], W_down^{(src)}[:, src_neuron]> (zeros if non-gated MLP)
          - cos_sim:      Cosine similarity between src write vector and dst read vector
        """
        src_write_mod, _, _ = LayerDiscovery.resolve_mlp_read_write_modules(self.model, src_layer)
        dst_write_mod, dst_up_mod, dst_gate_mod = LayerDiscovery.resolve_mlp_read_write_modules(
            self.model, dst_layer
        )
        _, _, num_dst = LayerDiscovery.resolve_module(self.model, dst_layer)

        device = next(self.model.parameters()).device
        zeros = torch.zeros(num_dst, dtype=torch.float32, device=device)

        if not hasattr(src_write_mod, "weight"):
            return zeros, zeros, zeros

        # Extract outgoing vector written into residual stream by src_neuron: shape [d_model]
        w_src = src_write_mod.weight.detach()
        if w_src.ndim == 3:
            # 3D MoE down_proj: [num_experts, d_model, d_mlp]
            src_vec = w_src[:, :, src_neuron].mean(dim=0).to(dtype=torch.float32)
        elif src_write_mod.__class__.__name__ == "Conv1D":
            src_vec = w_src[src_neuron, :].to(dtype=torch.float32)
        else:
            # Standard Linear down_proj: shape [d_model, d_mlp]
            src_vec = w_src[:, src_neuron].to(dtype=torch.float32)

        target_device = src_vec.device
        zeros = zeros.to(target_device)

        # Optionally apply the destination block's pre-MLP RMSNorm / LayerNorm weight scaling gamma
        if apply_pre_norm_scaling:
            pre_norm = LayerDiscovery.resolve_pre_mlp_norm(self.model, dst_layer)
            if pre_norm is not None and hasattr(pre_norm, "weight"):
                norm_w = pre_norm.weight.detach().to(device=target_device, dtype=torch.float32)
                if norm_w.shape == src_vec.shape:
                    src_vec = src_vec * norm_w

        read_mod = dst_up_mod if dst_up_mod is not None else dst_write_mod
        if read_mod is None or not hasattr(read_mod, "weight"):
            return zeros, zeros, zeros

        w_up = read_mod.weight.detach().to(device=target_device, dtype=torch.float32)
        if w_up.ndim == 3:
            # 3D MoE up_proj: [num_experts, d_mlp, d_model]
            w_up = w_up.mean(dim=0)
        elif read_mod.__class__.__name__ == "Conv1D":
            # [d_model, d_mlp] -> transpose to [d_mlp, d_model]
            w_up = w_up.T

        if w_up.shape[1] != src_vec.shape[0]:
            return zeros, zeros, zeros

        # Compute dot products and cosine similarities with all downstream neurons
        virtual_up = torch.mv(w_up, src_vec)
        src_norm = F.normalize(src_vec.unsqueeze(0), dim=-1).squeeze(0)
        up_norm = F.normalize(w_up, dim=-1)
        cos_sim = torch.mv(up_norm, src_norm)

        if dst_gate_mod is not None and hasattr(dst_gate_mod, "weight"):
            w_gate = dst_gate_mod.weight.detach().to(device=target_device, dtype=torch.float32)
            if w_gate.ndim == 3:
                w_gate = w_gate.mean(dim=0)
            elif dst_gate_mod.__class__.__name__ == "Conv1D":
                w_gate = w_gate.T
            virtual_gate = (
                torch.mv(w_gate, src_vec) if w_gate.shape[1] == src_vec.shape[0] else zeros
            )
        else:
            virtual_gate = zeros

        return virtual_up, virtual_gate, cos_sim

    # -------------------------------------------------------------------------
    # 2. Causal Pole-to-Pole Circuit Tracing (Activation Patching + Jacobian)
    # -------------------------------------------------------------------------
    def trace_circuit(
        self,
        src_layer: str,
        dst_layer: str,
        src_neuron: int,
        inputs: Any,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        src_pole: PoleType = PoleType.POSITIVE,
        intervention_mode: InterventionMode = InterventionMode.ZERO_ABLATE,
        intervention_value: float = 0.0,
        token_positions: Optional[List[int]] = None,
        compute_jacobian: bool = True,
        active_threshold: float = 1e-5,
        top_k: int = 16,
    ) -> CircuitGraph:
        """
        Traces the causal circuit from `(src_layer, src_neuron, src_pole)` to all neurons in `dst_layer`.

        Algorithm:
          1. Run a forward pass capturing clean activations at `src_layer` and `dst_layer`.
             If `compute_jacobian=True`, also compute the exact local gradient d(A_dst)/d(A_src).
          2. Identify token/sample positions where `src_neuron` is actively firing on `src_pole`.
          3. Apply a surgical intervention (`ZERO_ABLATE` or `HARD_CLAMP`) restricted to `src_pole`
             of `src_neuron` and re-run the forward pass to capture perturbed `dst_layer` activations.
          4. Combine causal deltas, local Jacobians, and static virtual weights into `CircuitEdge` objects.
        """
        run_fwd = forward_fn or (lambda m, inp: m(inp))
        src_mod, src_target, _ = LayerDiscovery.resolve_module(self.model, src_layer)
        dst_mod, dst_target, num_dst = LayerDiscovery.resolve_module(self.model, dst_layer)

        # Step 1: Clean pass (with optional autograd hook for Jacobian sensitivity)
        captured_src: List[torch.Tensor] = []
        captured_dst: List[torch.Tensor] = []

        def _capture_src_pre(_mod: nn.Module, inp: Tuple[Any, ...]):
            if inp and isinstance(inp[0], torch.Tensor):
                t = inp[0].detach().clone().requires_grad_(compute_jacobian)
                captured_src.append(t)
                return (t,) + inp[1:]
            return inp

        def _capture_src_fwd(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any):
            t_orig = out[0] if isinstance(out, tuple) else out
            if isinstance(t_orig, torch.Tensor):
                t = t_orig.detach().clone().requires_grad_(compute_jacobian)
                captured_src.append(t)
                return (t,) + out[1:] if isinstance(out, tuple) else t
            return out

        def _capture_dst_pre(_mod: nn.Module, inp: Tuple[Any, ...]):
            if inp and isinstance(inp[0], torch.Tensor):
                captured_dst.append(inp[0])

        def _capture_dst_fwd(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any):
            t_orig = out[0] if isinstance(out, tuple) else out
            if isinstance(t_orig, torch.Tensor):
                captured_dst.append(t_orig)

        h_src = (
            src_mod.register_forward_pre_hook(_capture_src_pre)
            if src_target == "input"
            else src_mod.register_forward_hook(_capture_src_fwd)
        )
        h_dst = (
            dst_mod.register_forward_pre_hook(_capture_dst_pre)
            if dst_target == "input"
            else dst_mod.register_forward_hook(_capture_dst_fwd)
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with self.injector.without_interventions(), torch.set_grad_enabled(compute_jacobian):
                run_fwd(self.model, inputs)

                if not captured_src or not captured_dst:
                    raise RuntimeError(
                        f"Failed to capture activations for '{src_layer}' or '{dst_layer}'."
                    )

                src_act = captured_src[-1]
                dst_act = captured_dst[-1]

                # Flatten batch & sequence/spatial dimensions to [N_tokens, D]
                if isinstance(src_mod, nn.Conv2d) and src_act.ndim == 4:
                    src_flat = src_act.permute(0, 2, 3, 1).reshape(-1, src_act.shape[1])
                else:
                    src_flat = src_act.reshape(-1, src_act.shape[-1])

                if isinstance(dst_mod, nn.Conv2d) and dst_act.ndim == 4:
                    dst_flat = dst_act.permute(0, 2, 3, 1).reshape(-1, dst_act.shape[1])
                else:
                    dst_flat = dst_act.reshape(-1, dst_act.shape[-1])

                # Determine active mask for the requested pole of src_neuron
                src_col = src_flat[:, src_neuron].detach()
                if src_pole == PoleType.POSITIVE:
                    active_mask = src_col > active_threshold
                elif src_pole == PoleType.NEGATIVE:
                    active_mask = src_col < -active_threshold
                else:
                    active_mask = src_col.abs() > active_threshold

                # Fallback to all positions if the pole didn't cross threshold on this sample
                if not active_mask.any():
                    active_mask = torch.ones_like(src_col, dtype=torch.bool)

                # If src and dst have different token/spatial counts (e.g. routed MoE or pooled CNN),
                # align active_mask to dst_flat length
                if active_mask.shape[0] != dst_flat.shape[0]:
                    dst_active_mask = torch.ones(
                        dst_flat.shape[0], dtype=torch.bool, device=dst_flat.device
                    )
                else:
                    dst_active_mask = active_mask.to(dst_flat.device)

                # Compute exact Jacobian sensitivity d(mean_active_dst[j]) / d(src[src_neuron])
                jacobian_vec = torch.zeros(num_dst, dtype=torch.float32, device=dst_flat.device)
                if compute_jacobian and dst_act.requires_grad:
                    # Sum dst activations over active token positions so we can compute vector-Jacobian product
                    active_dst_sum = dst_flat[dst_active_mask].float().sum(dim=0)
                    candidate_variance = dst_flat[dst_active_mask].detach().float().abs().mean(dim=0)
                    jac_k = min(num_dst, max(top_k * 2, 64))
                    _, top_jac_indices = torch.topk(candidate_variance, k=jac_k)

                    for j_idx in top_jac_indices.tolist():
                        grad_src = torch.autograd.grad(
                            outputs=active_dst_sum[j_idx],
                            inputs=src_act,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad_src is not None:
                            if isinstance(src_mod, nn.Conv2d) and grad_src.ndim == 4:
                                grad_flat = grad_src.permute(0, 2, 3, 1).reshape(
                                    -1, grad_src.shape[1]
                                )
                            else:
                                grad_flat = grad_src.reshape(-1, grad_src.shape[-1])
                            jacobian_vec[j_idx] = (
                                grad_flat[active_mask, src_neuron].float().mean().detach()
                            )
        finally:
            h_src.remove()
            h_dst.remove()

        clean_dst_flat = dst_flat.detach().float().clone()

        # Step 2: Intervened pass (surgically ablate or clamp src_neuron's pole without clearing user specs)
        captured_dst_ablated: List[torch.Tensor] = []
        h_dst_ab = (
            dst_mod.register_forward_pre_hook(
                lambda _m, inp: (
                    captured_dst_ablated.append(inp[0].detach())
                    if inp and isinstance(inp[0], torch.Tensor)
                    else None
                )
            )
            if dst_target == "input"
            else dst_mod.register_forward_hook(
                lambda _m, _inp, out: captured_dst_ablated.append(
                    (out[0] if isinstance(out, tuple) else out).detach()
                )
            )
        )

        trace_spec = InterventionSpec(
            layer_name=src_layer,
            neuron_indices=[src_neuron],
            mode=intervention_mode,
            pole=src_pole,
            value=intervention_value,
            token_positions=token_positions,
        )

        try:
            with torch.no_grad(), self.injector.temporary_interventions([trace_spec]):
                run_fwd(self.model, inputs)
        finally:
            h_dst_ab.remove()
            if was_training:
                self.model.train()

        raw_ab = captured_dst_ablated[-1]
        if isinstance(dst_mod, nn.Conv2d) and raw_ab.ndim == 4:
            ablated_dst_flat = raw_ab.permute(0, 2, 3, 1).reshape(-1, num_dst).float()
        else:
            ablated_dst_flat = raw_ab.reshape(-1, num_dst).float()

        # Step 3: Compute causal shift Delta = A_clean - A_ablated over active positions
        delta_active = clean_dst_flat[dst_active_mask] - ablated_dst_flat[dst_active_mask]
        mean_delta = delta_active.mean(dim=0)

        # Signed max delta (retaining sign of the largest absolute shift per downstream neuron)
        max_abs_idx = delta_active.abs().argmax(dim=0)
        max_delta = delta_active[max_abs_idx, torch.arange(num_dst, device=delta_active.device)]

        clean_mean = clean_dst_flat[dst_active_mask].mean(dim=0)
        ablated_mean = ablated_dst_flat[dst_active_mask].mean(dim=0)

        # Step 4: Combine with static virtual weights
        v_up, v_gate, cos_sim = self.compute_virtual_weights(src_layer, dst_layer, src_neuron)
        v_up = v_up.to(device=mean_delta.device, dtype=mean_delta.dtype)
        v_gate = v_gate.to(device=mean_delta.device, dtype=mean_delta.dtype)
        cos_sim = cos_sim.to(device=mean_delta.device, dtype=mean_delta.dtype)

        # Select top_k most affected downstream neurons (by absolute mean causal delta + static alignment)
        ranking_metric = mean_delta.abs() + 1e-4 * v_up.abs()
        k_select = min(top_k, num_dst)
        _, top_dst_indices = torch.topk(ranking_metric, k=k_select)

        edges: List[CircuitEdge] = []
        for dst_idx in top_dst_indices.tolist():
            c_mean = float(clean_mean[dst_idx].item())
            m_delta = float(mean_delta[dst_idx].item())
            dst_pole = PoleType.POSITIVE if c_mean >= 0 else PoleType.NEGATIVE

            edges.append(
                CircuitEdge(
                    src_layer=src_layer,
                    src_neuron=src_neuron,
                    src_pole=src_pole,
                    dst_layer=dst_layer,
                    dst_neuron=dst_idx,
                    dst_pole_tendency=dst_pole,
                    mean_causal_delta=m_delta,
                    max_causal_delta=float(max_delta[dst_idx].item()),
                    clean_dst_mean=c_mean,
                    ablated_dst_mean=float(ablated_mean[dst_idx].item()),
                    jacobian_sensitivity=float(jacobian_vec[dst_idx].item()),
                    virtual_weight_up=float(v_up[dst_idx].item()),
                    virtual_weight_gate=float(v_gate[dst_idx].item()),
                    static_alignment_cos=float(cos_sim[dst_idx].item()),
                )
            )

        edges.sort(key=lambda e: abs(e.mean_causal_delta), reverse=True)
        return CircuitGraph(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neuron=src_neuron,
            src_pole=src_pole,
            edges=edges,
        )

    # -------------------------------------------------------------------------
    # 3. Multi-Hop Circuit Subgraph Tracing (L_0 -> L_1 -> ... -> L_n)
    # -------------------------------------------------------------------------
    def trace_multi_hop(
        self,
        layer_sequence: List[str],
        root_neuron: int,
        root_pole: PoleType,
        inputs: Any,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        beam_width: int = 3,
    ) -> List[CircuitGraph]:
        """
        Traces a multi-layer cascade starting from `(layer_sequence[0], root_neuron, root_pole)`
        and expanding the top `beam_width` downstream neurons at each subsequent layer.
        """
        if len(layer_sequence) < 2:
            raise ValueError("`layer_sequence` must contain at least 2 layers.")

        graphs: List[CircuitGraph] = []
        current_front: List[Tuple[int, PoleType]] = [(root_neuron, root_pole)]

        for hop in range(len(layer_sequence) - 1):
            src_layer = layer_sequence[hop]
            dst_layer = layer_sequence[hop + 1]
            next_front: List[Tuple[int, PoleType]] = []

            for neuron_idx, pole in current_front:
                graph = self.trace_circuit(
                    src_layer=src_layer,
                    dst_layer=dst_layer,
                    src_neuron=neuron_idx,
                    inputs=inputs,
                    forward_fn=forward_fn,
                    src_pole=pole,
                    top_k=beam_width * 2,
                )
                graphs.append(graph)
                for edge in graph.edges[:beam_width]:
                    next_front.append((edge.dst_neuron, edge.dst_pole_tendency))

            # Deduplicate next frontier while preserving order
            seen = set()
            dedup_front = []
            for item in next_front:
                if item not in seen:
                    seen.add(item)
                    dedup_front.append(item)
            current_front = dedup_front[:beam_width]

        return graphs
