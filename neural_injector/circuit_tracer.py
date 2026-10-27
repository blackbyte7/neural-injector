from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from .discovery import LayerDiscovery, _SlicedLinearView
from .types import (
    CircuitEdge,
    CircuitGraph,
    DownstreamReadoutReport,
    InterventionMode,
    InterventionSpec,
    PoleType,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector


class CircuitTracer:
    """
    Traces structural and causal neuron-to-neuron circuits across layers (both
    downstream L_src -> L_dst and incoming upstream L_src -> dst_neuron).
    Designed around the 'Neural Networks Are Already Clean' paradigm:
    treats individual neuron poles as interpretable routing units ('verbs')
    and measures how an upstream pole excites, inhibits, or gates downstream neurons.

    Uses Single-Pass Forward-Mode Automatic Differentiation (`torch.autograd.forward_ad`)
    to compute exact Jacobians d(A_dst)/d(a_src) across all downstream neurons and layers
    simultaneously in a single forward pass (Sections 4.4 & 5.1), and reverse-mode
    vector-Jacobian products for incoming circuit decomposition.
    """

    def __init__(self, injector: NeuralInjector):
        self.injector = injector
        self.model = injector.model

    # -------------------------------------------------------------------------
    # 1. Static Structural Virtual Weights (Downstream & Incoming)
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

    @torch.no_grad()
    def compute_incoming_virtual_weights(
        self,
        src_layer: str,
        dst_layer: str,
        dst_neuron: int,
        apply_pre_norm_scaling: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes incoming virtual weights from ALL neurons in `src_layer` into a single
        target `dst_neuron` in `dst_layer`.

        Returns three 1D tensors of shape [num_src_neurons]:
          - virtual_up:   <W_up^{(dst)}[dst_neuron, :], W_down^{(src)}[:, i]>
          - virtual_gate: <W_gate^{(dst)}[dst_neuron, :], W_down^{(src)}[:, i]>
          - cos_sim:      Cosine similarity between each src write vector and dst_neuron's read vector
        """
        src_write_mod, _, _ = LayerDiscovery.resolve_mlp_read_write_modules(self.model, src_layer)
        dst_write_mod, dst_up_mod, dst_gate_mod = LayerDiscovery.resolve_mlp_read_write_modules(
            self.model, dst_layer
        )
        _, _, num_src = LayerDiscovery.resolve_module(self.model, src_layer)

        device = next(self.model.parameters()).device
        zeros = torch.zeros(num_src, dtype=torch.float32, device=device)

        if not hasattr(src_write_mod, "weight"):
            return zeros, zeros, zeros

        # Extract all src write columns: shape [d_model, num_src]
        w_src = src_write_mod.weight.detach()
        if w_src.ndim == 3:
            w_down = w_src.mean(dim=0).to(dtype=torch.float32)
        elif src_write_mod.__class__.__name__ == "Conv1D":
            w_down = w_src.T.to(dtype=torch.float32)
        else:
            w_down = w_src.to(dtype=torch.float32)

        target_device = w_down.device
        zeros = zeros.to(target_device)

        if apply_pre_norm_scaling:
            pre_norm = LayerDiscovery.resolve_pre_mlp_norm(self.model, dst_layer)
            if pre_norm is not None and hasattr(pre_norm, "weight"):
                norm_w = pre_norm.weight.detach().to(device=target_device, dtype=torch.float32)
                if norm_w.shape[0] == w_down.shape[0]:
                    w_down = w_down * norm_w.unsqueeze(1)

        read_mod = dst_up_mod if dst_up_mod is not None else dst_write_mod
        if read_mod is None or not hasattr(read_mod, "weight"):
            return zeros, zeros, zeros

        w_up = read_mod.weight.detach().to(device=target_device, dtype=torch.float32)
        if w_up.ndim == 3:
            w_up = w_up.mean(dim=0)
        elif read_mod.__class__.__name__ == "Conv1D":
            w_up = w_up.T

        if w_up.shape[1] != w_down.shape[0] or dst_neuron >= w_up.shape[0]:
            return zeros, zeros, zeros

        dst_up_vec = w_up[dst_neuron, :]  # [d_model]
        virtual_up = torch.mv(w_down.T, dst_up_vec)

        dst_up_norm = F.normalize(dst_up_vec.unsqueeze(0), dim=-1).squeeze(0)
        w_down_norm = F.normalize(w_down.T, dim=-1)
        cos_sim = torch.mv(w_down_norm, dst_up_norm)

        if dst_gate_mod is not None and hasattr(dst_gate_mod, "weight"):
            w_gate = dst_gate_mod.weight.detach().to(device=target_device, dtype=torch.float32)
            if w_gate.ndim == 3:
                w_gate = w_gate.mean(dim=0)
            elif dst_gate_mod.__class__.__name__ == "Conv1D":
                w_gate = w_gate.T
            if w_gate.shape[1] == w_down.shape[0] and dst_neuron < w_gate.shape[0]:
                virtual_gate = torch.mv(w_down.T, w_gate[dst_neuron, :])
            else:
                virtual_gate = zeros
        else:
            virtual_gate = zeros

        return virtual_up, virtual_gate, cos_sim

    # -------------------------------------------------------------------------
    # 2. Single-Pass Forward-Mode AD Multi-Layer Jacobian Tracer (Section 5.1)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def trace_forward_ad_all_layers(
        self,
        src_layer: str,
        src_neuron: int,
        inputs: Any,
        dst_layers: Optional[List[str]] = None,
        src_pole: PoleType = PoleType.POSITIVE,
        active_threshold: float = 1e-5,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Attaches a forward-mode autograd dual tangent `tangent = one_hot(src_neuron)`
        at `src_layer` inside `torch.autograd.forward_ad.dual_level()`.
        A single forward pass computes the exact Jacobian sensitivity vector
        `d(A_dst) / d(a_{src_layer, src_neuron})` across EVERY downstream layer and
        every downstream neuron simultaneously!
        """
        run_fwd = forward_fn or (lambda m, inp: m(inp))
        all_discovered = self.injector.list_targetable_layers()
        if dst_layers is None:
            layer_keys = list(all_discovered.keys())
            if src_layer in layer_keys:
                src_pos = layer_keys.index(src_layer)
                target_dst_layers = layer_keys[src_pos + 1:]
            else:
                target_dst_layers = [k for k in layer_keys if k != src_layer]
        else:
            target_dst_layers = list(dst_layers)

        src_mod, src_target, _ = LayerDiscovery.resolve_module(self.model, src_layer)
        is_src_conv = isinstance(src_mod, nn.Conv2d)

        active_count_holder: List[int] = [1]
        active_mask_holder: List[torch.Tensor] = []
        captured_tangents: Dict[str, torch.Tensor] = {}
        handles: List[RemovableHandle] = []

        def _make_dual_on_src(t: torch.Tensor) -> torch.Tensor:
            if not t.is_floating_point():
                return t
            if is_src_conv and t.ndim == 4:
                src_flat = t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])
            else:
                src_flat = t.reshape(-1, t.shape[-1])
            src_col = src_flat[:, src_neuron].detach()
            if src_pole == PoleType.POSITIVE:
                amask = src_col > active_threshold
            elif src_pole == PoleType.NEGATIVE:
                amask = src_col < -active_threshold
            else:
                amask = src_col.abs() > active_threshold
            if not amask.any():
                amask = torch.ones_like(src_col, dtype=torch.bool)

            active_mask_holder.append(amask)
            active_count_holder[0] = max(int(amask.sum().item()), 1)

            tangent_flat = torch.zeros_like(src_flat)
            tangent_flat[amask, src_neuron] = 1.0
            if is_src_conv and t.ndim == 4:
                b_dim, c_dim, h_dim, w_dim = t.shape
                tangent = (
                    tangent_flat.reshape(b_dim, h_dim, w_dim, c_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            else:
                tangent = tangent_flat.reshape_as(t)
            return fwAD.make_dual(t, tangent)

        if src_target == "input":
            handles.append(
                src_mod.register_forward_pre_hook(
                    lambda _m, inp: (
                        (_make_dual_on_src(inp[0]),) + inp[1:]
                        if inp and isinstance(inp[0], torch.Tensor)
                        else inp
                    )
                )
            )
        else:
            handles.append(
                src_mod.register_forward_hook(
                    lambda _m, _inp, out: (
                        (_make_dual_on_src(out[0]),) + out[1:]
                        if isinstance(out, tuple) and isinstance(out[0], torch.Tensor)
                        else (_make_dual_on_src(out) if isinstance(out, torch.Tensor) else out)
                    )
                )
            )

        for d_layer in target_dst_layers:
            d_mod, d_target, num_d = LayerDiscovery.resolve_module(self.model, d_layer)
            is_d_conv = isinstance(d_mod, nn.Conv2d)

            def _make_dst_unpack(lname: str, n_dst: int, conv_flag: bool):
                def _unpack_tensor(t_out: torch.Tensor) -> None:
                    _, tang = fwAD.unpack_dual(t_out)
                    if tang is None:
                        captured_tangents[lname] = torch.zeros(
                            n_dst, dtype=torch.float32, device=t_out.device
                        )
                        return
                    if conv_flag and tang.ndim == 4:
                        dst_tang_flat = tang.permute(0, 2, 3, 1).reshape(-1, n_dst)
                    else:
                        dst_tang_flat = tang.reshape(-1, n_dst)
                    if (
                        active_mask_holder
                        and active_mask_holder[-1].shape[0] == dst_tang_flat.shape[0]
                    ):
                        d_mask = active_mask_holder[-1].to(dst_tang_flat.device)
                        jac_vec = (
                            dst_tang_flat[d_mask].float().sum(dim=0) / active_count_holder[0]
                        )
                    else:
                        jac_vec = dst_tang_flat.float().mean(dim=0)
                    captured_tangents[lname] = jac_vec.detach()

                return _unpack_tensor

            unpack_fn = _make_dst_unpack(d_layer, num_d, is_d_conv)
            if d_target == "input":
                handles.append(
                    d_mod.register_forward_pre_hook(
                        lambda _m, inp, u=unpack_fn: (
                            u(inp[0]) if inp and isinstance(inp[0], torch.Tensor) else None
                        )
                    )
                )
            else:
                handles.append(
                    d_mod.register_forward_hook(
                        lambda _m, _inp, out, u=unpack_fn: u(
                            out[0] if isinstance(out, tuple) else out
                        )
                    )
                )

        was_training = self.model.training
        self.model.eval()
        try:
            with self.injector.without_interventions(), fwAD.dual_level():
                run_fwd(self.model, inputs)
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self.model.train()

        return captured_tangents

    # -------------------------------------------------------------------------
    # 3. Causal Pole-to-Pole Circuit Tracing (Downstream & Incoming Upstream)
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
        Uses single-pass Forward-Mode AD (`torch.autograd.forward_ad`) to compute exact
        Jacobian sensitivities across all downstream neurons in one forward pass, with
        automatic fallback to reverse-mode autograd if a custom op does not support dual tensors.
        """
        run_fwd = forward_fn or (lambda m, inp: m(inp))
        src_mod, src_target, _ = LayerDiscovery.resolve_module(self.model, src_layer)
        dst_mod, dst_target, num_dst = LayerDiscovery.resolve_module(self.model, dst_layer)
        is_src_conv = isinstance(src_mod, nn.Conv2d)
        is_dst_conv = isinstance(dst_mod, nn.Conv2d)

        captured_src_primal: List[torch.Tensor] = []
        captured_dst_primal: List[torch.Tensor] = []
        captured_dst_tangent: List[Optional[torch.Tensor]] = []
        active_mask_holder: List[torch.Tensor] = []

        def _attach_dual_src(t: torch.Tensor) -> torch.Tensor:
            primal = t.detach()
            captured_src_primal.append(primal)
            if not compute_jacobian or not primal.is_floating_point():
                return t
            if is_src_conv and primal.ndim == 4:
                s_flat = primal.permute(0, 2, 3, 1).reshape(-1, primal.shape[1])
            else:
                s_flat = primal.reshape(-1, primal.shape[-1])
            s_col = s_flat[:, src_neuron]
            if src_pole == PoleType.POSITIVE:
                amask = s_col > active_threshold
            elif src_pole == PoleType.NEGATIVE:
                amask = s_col < -active_threshold
            else:
                amask = s_col.abs() > active_threshold
            if not amask.any():
                amask = torch.ones_like(s_col, dtype=torch.bool)
            active_mask_holder.append(amask)

            src_tangent_flat = torch.zeros_like(s_flat)
            src_tangent_flat[amask, src_neuron] = 1.0
            if is_src_conv and primal.ndim == 4:
                b_dim, c_dim, h_dim, w_dim = primal.shape
                tangent = (
                    src_tangent_flat.reshape(b_dim, h_dim, w_dim, c_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            else:
                tangent = src_tangent_flat.reshape_as(primal)
            return fwAD.make_dual(primal, tangent)

        def _capture_dst_dual(t: torch.Tensor) -> None:
            primal, tangent = fwAD.unpack_dual(t)
            captured_dst_primal.append(primal.detach())
            captured_dst_tangent.append(tangent.detach() if tangent is not None else None)

        h_src = (
            src_mod.register_forward_pre_hook(
                lambda _m, inp: (
                    (_attach_dual_src(inp[0]),) + inp[1:]
                    if inp and isinstance(inp[0], torch.Tensor)
                    else inp
                )
            )
            if src_target == "input"
            else src_mod.register_forward_hook(
                lambda _m, _inp, out: (
                    (_attach_dual_src(out[0]),) + out[1:]
                    if isinstance(out, tuple) and isinstance(out[0], torch.Tensor)
                    else (_attach_dual_src(out) if isinstance(out, torch.Tensor) else out)
                )
            )
        )
        h_dst = (
            dst_mod.register_forward_pre_hook(
                lambda _m, inp: (
                    _capture_dst_dual(inp[0])
                    if inp and isinstance(inp[0], torch.Tensor)
                    else None
                )
            )
            if dst_target == "input"
            else dst_mod.register_forward_hook(
                lambda _m, _inp, out: _capture_dst_dual(
                    out[0] if isinstance(out, tuple) else out
                )
            )
        )

        was_training = self.model.training
        self.model.eval()
        fw_ad_succeeded = False
        try:
            with self.injector.without_interventions(), torch.no_grad(), fwAD.dual_level():
                try:
                    run_fwd(self.model, inputs)
                    fw_ad_succeeded = True
                except RuntimeError:
                    # Fallback if a custom op does not support forward-mode AD dual tensors
                    captured_src_primal.clear()
                    captured_dst_primal.clear()
                    captured_dst_tangent.clear()
                    active_mask_holder.clear()
        finally:
            h_src.remove()
            h_dst.remove()

        if fw_ad_succeeded and captured_src_primal and captured_dst_primal:
            src_act = captured_src_primal[-1]
            dst_act = captured_dst_primal[-1]
            dst_tang = captured_dst_tangent[-1] if captured_dst_tangent else None

            if is_src_conv and src_act.ndim == 4:
                src_flat = src_act.permute(0, 2, 3, 1).reshape(-1, src_act.shape[1])
            else:
                src_flat = src_act.reshape(-1, src_act.shape[-1])

            if is_dst_conv and dst_act.ndim == 4:
                dst_flat = dst_act.permute(0, 2, 3, 1).reshape(-1, dst_act.shape[1])
            else:
                dst_flat = dst_act.reshape(-1, dst_act.shape[-1])

            if active_mask_holder:
                active_mask = active_mask_holder[-1]
            else:
                src_col = src_flat[:, src_neuron]
                active_mask = (
                    (src_col > active_threshold)
                    if src_pole == PoleType.POSITIVE
                    else (
                        (src_col < -active_threshold)
                        if src_pole == PoleType.NEGATIVE
                        else (src_col.abs() > active_threshold)
                    )
                )
                if not active_mask.any():
                    active_mask = torch.ones_like(src_col, dtype=torch.bool)

            if active_mask.shape[0] != dst_flat.shape[0]:
                dst_active_mask = torch.ones(
                    dst_flat.shape[0], dtype=torch.bool, device=dst_flat.device
                )
            else:
                dst_active_mask = active_mask.to(dst_flat.device)

            jacobian_vec = torch.zeros(num_dst, dtype=torch.float32, device=dst_flat.device)
            if compute_jacobian and dst_tang is not None:
                if is_dst_conv and dst_tang.ndim == 4:
                    dst_tang_flat = dst_tang.permute(0, 2, 3, 1).reshape(-1, num_dst)
                else:
                    dst_tang_flat = dst_tang.reshape(-1, num_dst)
                n_active = max(int(active_mask.sum().item()), 1)
                jacobian_vec = dst_tang_flat[dst_active_mask].float().sum(dim=0) / n_active
        else:
            # Reverse-mode autograd fallback
            rev_src: List[torch.Tensor] = []
            rev_dst: List[torch.Tensor] = []

            def _rev_src_pre(_m: nn.Module, inp: Tuple[Any, ...]):
                if inp and isinstance(inp[0], torch.Tensor):
                    t = inp[0].detach().clone().requires_grad_(compute_jacobian)
                    rev_src.append(t)
                    return (t,) + inp[1:]
                return inp

            def _rev_src_fwd(_m: nn.Module, _inp: Tuple[Any, ...], out: Any):
                t_orig = out[0] if isinstance(out, tuple) else out
                if isinstance(t_orig, torch.Tensor):
                    t = t_orig.detach().clone().requires_grad_(compute_jacobian)
                    rev_src.append(t)
                    return (t,) + out[1:] if isinstance(out, tuple) else t
                return out

            hr_src = (
                src_mod.register_forward_pre_hook(_rev_src_pre)
                if src_target == "input"
                else src_mod.register_forward_hook(_rev_src_fwd)
            )
            hr_dst = (
                dst_mod.register_forward_pre_hook(
                    lambda _m, inp: (
                        rev_dst.append(inp[0])
                        if inp and isinstance(inp[0], torch.Tensor)
                        else None
                    )
                )
                if dst_target == "input"
                else dst_mod.register_forward_hook(
                    lambda _m, _inp, out: rev_dst.append(
                        out[0] if isinstance(out, tuple) else out
                    )
                )
            )
            try:
                with self.injector.without_interventions(), torch.set_grad_enabled(compute_jacobian):
                    run_fwd(self.model, inputs)
            finally:
                hr_src.remove()
                hr_dst.remove()

            if not rev_src or not rev_dst:
                raise RuntimeError(
                    f"Failed to capture activations for '{src_layer}' or '{dst_layer}'."
                )

            src_act = rev_src[-1]
            dst_act = rev_dst[-1]
            src_flat = (
                src_act.permute(0, 2, 3, 1).reshape(-1, src_act.shape[1])
                if (is_src_conv and src_act.ndim == 4)
                else src_act.reshape(-1, src_act.shape[-1])
            )
            dst_flat = (
                dst_act.permute(0, 2, 3, 1).reshape(-1, dst_act.shape[1])
                if (is_dst_conv and dst_act.ndim == 4)
                else dst_act.reshape(-1, dst_act.shape[-1])
            )
            src_col = src_flat[:, src_neuron].detach()
            active_mask = (
                (src_col > active_threshold)
                if src_pole == PoleType.POSITIVE
                else (
                    (src_col < -active_threshold)
                    if src_pole == PoleType.NEGATIVE
                    else (src_col.abs() > active_threshold)
                )
            )
            if not active_mask.any():
                active_mask = torch.ones_like(src_col, dtype=torch.bool)
            dst_active_mask = (
                torch.ones(dst_flat.shape[0], dtype=torch.bool, device=dst_flat.device)
                if active_mask.shape[0] != dst_flat.shape[0]
                else active_mask.to(dst_flat.device)
            )
            jacobian_vec = torch.zeros(num_dst, dtype=torch.float32, device=dst_flat.device)
            if compute_jacobian and dst_act.requires_grad:
                with torch.enable_grad():
                    active_dst_sum = dst_flat[dst_active_mask].float().sum(dim=0)
                    cand_var = dst_flat[dst_active_mask].detach().float().abs().mean(dim=0)
                    jac_k = min(num_dst, max(top_k * 2, 64))
                    _, top_jac_indices = torch.topk(cand_var, k=jac_k)
                    for j_idx in top_jac_indices.tolist():
                        grad_src = torch.autograd.grad(
                            outputs=active_dst_sum[j_idx],
                            inputs=src_act,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad_src is not None:
                            g_flat = (
                                grad_src.permute(0, 2, 3, 1).reshape(-1, grad_src.shape[1])
                                if (is_src_conv and grad_src.ndim == 4)
                                else grad_src.reshape(-1, grad_src.shape[-1])
                            )
                            jacobian_vec[j_idx] = (
                                g_flat[active_mask, src_neuron].float().mean().detach()
                            )

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
        if is_dst_conv and raw_ab.ndim == 4:
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
        graph = CircuitGraph(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neuron=src_neuron,
            src_pole=src_pole,
            edges=edges,
        )
        if hasattr(self.injector, "ecology_map") and self.injector.ecology_map is not None:
            self.injector.ecology_map.register_circuit_graph(graph)
        return graph

    def trace_incoming_circuit(
        self,
        src_layer: str,
        dst_layer: str,
        dst_neuron: int,
        inputs: Any,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        dst_pole: PoleType = PoleType.POSITIVE,
        active_threshold: float = 1e-5,
        top_k: int = 8,
    ) -> CircuitGraph:
        """
        Traces the incoming upstream circuit from ALL neurons in `src_layer` into a single
        target `(dst_layer, dst_neuron)` (Cammarata's upstream decomposition of recursive offshoots):
          1. Computes exact reverse-mode Jacobian sensitivities `d(A_dst[dst_neuron]) / d(a_src)`
             across all upstream neurons in `src_layer` in a single backward pass.
          2. Combines first-order attribution `a_src * d(A_dst)/d(a_src)` with incoming virtual
             weights (`W_up` and `W_gate`) to select the top candidate upstream contributors.
          3. Runs exact surgical pole-ablation verification on the top candidates to populate
             `mean_causal_delta` and registers all discovered edges into `self.injector.ecology_map`.
        """
        run_fwd = forward_fn or (lambda m, inp: m(inp))
        src_mod, src_target, num_src = LayerDiscovery.resolve_module(self.model, src_layer)
        dst_mod, dst_target, num_dst = LayerDiscovery.resolve_module(self.model, dst_layer)
        is_src_conv = isinstance(src_mod, nn.Conv2d)
        is_dst_conv = isinstance(dst_mod, nn.Conv2d)

        captured_src: List[torch.Tensor] = []
        captured_dst: List[torch.Tensor] = []

        def _cap_src_pre(_m: nn.Module, inp: Tuple[Any, ...]):
            if inp and isinstance(inp[0], torch.Tensor):
                t = inp[0].detach().clone().requires_grad_(True)
                captured_src.append(t)
                return (t,) + inp[1:]
            return inp

        def _cap_src_fwd(_m: nn.Module, _inp: Tuple[Any, ...], out: Any):
            t0 = out[0] if isinstance(out, tuple) else out
            if isinstance(t0, torch.Tensor):
                t = t0.detach().clone().requires_grad_(True)
                captured_src.append(t)
                return (t,) + out[1:] if isinstance(out, tuple) else t
            return out

        h_src = (
            src_mod.register_forward_pre_hook(_cap_src_pre)
            if src_target == "input"
            else src_mod.register_forward_hook(_cap_src_fwd)
        )
        h_dst = (
            dst_mod.register_forward_pre_hook(
                lambda _m, inp: (
                    captured_dst.append(inp[0])
                    if inp and isinstance(inp[0], torch.Tensor)
                    else None
                )
            )
            if dst_target == "input"
            else dst_mod.register_forward_hook(
                lambda _m, _inp, out: captured_dst.append(
                    out[0] if isinstance(out, tuple) else out
                )
            )
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with self.injector.without_interventions(), torch.enable_grad():
                run_fwd(self.model, inputs)
                src_act = captured_src[-1]
                dst_act = captured_dst[-1]

                dst_flat = (
                    dst_act.permute(0, 2, 3, 1).reshape(-1, num_dst)
                    if (is_dst_conv and dst_act.ndim == 4)
                    else dst_act.reshape(-1, num_dst)
                )
                dst_col = dst_flat[:, dst_neuron]
                if dst_pole == PoleType.POSITIVE:
                    amask = dst_col.detach() > active_threshold
                elif dst_pole == PoleType.NEGATIVE:
                    amask = dst_col.detach() < -active_threshold
                else:
                    amask = dst_col.detach().abs() > active_threshold
                if not amask.any():
                    amask = torch.ones_like(dst_col, dtype=torch.bool)

                target_scalar = dst_col[amask].float().mean()
                grad_src = torch.autograd.grad(
                    outputs=target_scalar,
                    inputs=src_act,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
        finally:
            h_src.remove()
            h_dst.remove()

        src_flat = (
            src_act.detach().permute(0, 2, 3, 1).reshape(-1, num_src).float()
            if (is_src_conv and src_act.ndim == 4)
            else src_act.detach().reshape(-1, num_src).float()
        )
        clean_dst_col = dst_col.detach().float()
        clean_dst_mean = float(clean_dst_col[amask].mean().item())

        if grad_src is not None:
            g_flat = (
                grad_src.detach().permute(0, 2, 3, 1).reshape(-1, num_src).float()
                if (is_src_conv and grad_src.ndim == 4)
                else grad_src.detach().reshape(-1, num_src).float()
            )
            # Note: target_scalar was mean over `amask`, so sum across positions gives mean sensitivity
            jac_vec = g_flat.sum(dim=0)
            first_order_attrib = (src_flat * g_flat).sum(dim=0)
        else:
            jac_vec = torch.zeros(num_src, dtype=torch.float32, device=src_flat.device)
            first_order_attrib = torch.zeros(num_src, dtype=torch.float32, device=src_flat.device)

        v_up, v_gate, cos_sim = self.compute_incoming_virtual_weights(
            src_layer, dst_layer, dst_neuron
        )
        v_up = v_up.to(device=src_flat.device)
        v_gate = v_gate.to(device=src_flat.device)
        cos_sim = cos_sim.to(device=src_flat.device)

        src_mean_act = src_flat.mean(dim=0)
        ranking_score = (
            first_order_attrib.abs()
            + 0.1 * (src_mean_act.abs() * (v_up.abs() + v_gate.abs()))
        )
        k_select = min(top_k, num_src)
        _, top_src_indices = torch.topk(ranking_score, k=k_select)

        edges: List[CircuitEdge] = []
        dst_ab_list: List[torch.Tensor] = []
        h_dst_ab = (
            dst_mod.register_forward_pre_hook(
                lambda _m, inp: (
                    dst_ab_list.append(inp[0].detach())
                    if inp and isinstance(inp[0], torch.Tensor)
                    else None
                )
            )
            if dst_target == "input"
            else dst_mod.register_forward_hook(
                lambda _m, _inp, out: dst_ab_list.append(
                    (out[0] if isinstance(out, tuple) else out).detach()
                )
            )
        )

        try:
            with torch.no_grad():
                for s_idx in top_src_indices.tolist():
                    s_mean = float(src_mean_act[s_idx].item())
                    s_pole = PoleType.POSITIVE if s_mean >= 0.0 else PoleType.NEGATIVE
                    spec = InterventionSpec(
                        layer_name=src_layer,
                        neuron_indices=[s_idx],
                        mode=InterventionMode.ZERO_ABLATE,
                        pole=s_pole,
                        value=0.0,
                    )
                    dst_ab_list.clear()
                    with self.injector.temporary_interventions([spec]):
                        run_fwd(self.model, inputs)
                    ab_raw = dst_ab_list[-1]
                    ab_flat = (
                        ab_raw.permute(0, 2, 3, 1).reshape(-1, num_dst).float()
                        if (is_dst_conv and ab_raw.ndim == 4)
                        else ab_raw.reshape(-1, num_dst).float()
                    )
                    ab_col = ab_flat[:, dst_neuron]
                    delta_vec = clean_dst_col[amask] - ab_col[amask]
                    m_delta = float(delta_vec.mean().item())
                    max_idx = int(delta_vec.abs().argmax().item())
                    max_delta = float(delta_vec[max_idx].item())
                    ab_mean = float(ab_col[amask].mean().item())

                    edges.append(
                        CircuitEdge(
                            src_layer=src_layer,
                            src_neuron=s_idx,
                            src_pole=s_pole,
                            dst_layer=dst_layer,
                            dst_neuron=dst_neuron,
                            dst_pole_tendency=dst_pole,
                            mean_causal_delta=m_delta,
                            max_causal_delta=max_delta,
                            clean_dst_mean=clean_dst_mean,
                            ablated_dst_mean=ab_mean,
                            jacobian_sensitivity=float(jac_vec[s_idx].item()),
                            virtual_weight_up=float(v_up[s_idx].item()),
                            virtual_weight_gate=float(v_gate[s_idx].item()),
                            static_alignment_cos=float(cos_sim[s_idx].item()),
                        )
                    )
        finally:
            h_dst_ab.remove()
            if was_training:
                self.model.train()

        edges.sort(key=lambda e: abs(e.mean_causal_delta), reverse=True)
        graph = CircuitGraph(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neuron=edges[0].src_neuron if edges else 0,
            src_pole=dst_pole,
            edges=edges,
            dst_neuron=dst_neuron,
        )
        if hasattr(self.injector, "ecology_map") and self.injector.ecology_map is not None:
            self.injector.ecology_map.register_circuit_graph(graph)
        return graph

    # -------------------------------------------------------------------------
    # 4. Multi-Hop Circuit Subgraph Tracing (L_0 -> L_1 -> ... -> L_n)
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

    # -------------------------------------------------------------------------
    # 5. Auditing Superposition Hiding Places ("Nothing Reads That Combination")
    # -------------------------------------------------------------------------
    @torch.no_grad()
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
        Tests Cammarata's 'Nothing Reads That Combination' hypothesis (Section 3.5):
        Evaluates whether weak/secondary cross-activations across a group of neurons
        `src_neurons` in `src_layer` actually cross the gating/activation threshold of
        any downstream neuron in `dst_layer`.
        """
        run_fwd = forward_fn or (lambda m, inp: m(inp))
        dst_mod, dst_target, num_dst = LayerDiscovery.resolve_module(self.model, dst_layer)
        _, _, dst_gate_mod = LayerDiscovery.resolve_mlp_read_write_modules(self.model, dst_layer)

        tracker = self.injector.get_stats_tracker(src_layer)
        if isinstance(weak_threshold, list):
            applied_thresholds = [float(x) for x in weak_threshold]
            scalar_val = applied_thresholds[0] if applied_thresholds else 0.0
            per_neuron_vals: Optional[List[float]] = applied_thresholds
        elif isinstance(weak_threshold, (int, float)):
            scalar_val = float(weak_threshold)
            applied_thresholds = [scalar_val] * len(src_neurons)
            per_neuron_vals = None
        elif tracker is not None:
            applied_thresholds = tracker.compute_adaptive_thresholds(
                src_neurons, std_multiplier=std_multiplier, percentile=percentile
            )
            scalar_val = applied_thresholds[0] if applied_thresholds else 0.5
            per_neuron_vals = applied_thresholds
        else:
            scalar_val = 0.5
            applied_thresholds = [0.5] * len(src_neurons)
            per_neuron_vals = None

        captured_dst: List[torch.Tensor] = []
        captured_gate: List[torch.Tensor] = []

        h_dst = (
            dst_mod.register_forward_pre_hook(
                lambda _m, inp: (
                    captured_dst.append(inp[0].detach().float())
                    if inp and isinstance(inp[0], torch.Tensor)
                    else None
                )
            )
            if dst_target == "input"
            else dst_mod.register_forward_hook(
                lambda _m, _inp, out: captured_dst.append(
                    (out[0] if isinstance(out, tuple) else out).detach().float()
                )
            )
        )

        h_gate = None
        if dst_gate_mod is not None and isinstance(dst_gate_mod, nn.Module) and not isinstance(
            dst_gate_mod, _SlicedLinearView
        ):
            h_gate = dst_gate_mod.register_forward_hook(
                lambda _m, _inp, out: captured_gate.append(
                    (out[0] if isinstance(out, tuple) else out).detach().float()
                )
            )

        was_training = self.model.training
        self.model.eval()
        try:
            with self.injector.without_interventions():
                run_fwd(self.model, inputs)
            clean_dst = captured_dst[-1].reshape(-1, num_dst)
            clean_gate = captured_gate[-1].reshape(-1, num_dst) if captured_gate else None

            prune_spec = InterventionSpec(
                layer_name=src_layer,
                neuron_indices=list(src_neurons),
                mode=InterventionMode.WEAK_PRUNE,
                pole=PoleType.BOTH,
                value=scalar_val,
                per_neuron_values=per_neuron_vals,
            )
            with self.injector.temporary_interventions([prune_spec]):
                run_fwd(self.model, inputs)
            pruned_dst = captured_dst[-1].reshape(-1, num_dst)
            pruned_gate = (
                captured_gate[-1].reshape(-1, num_dst)
                if (captured_gate and len(captured_gate) >= 2)
                else None
            )
        finally:
            h_dst.remove()
            if h_gate is not None:
                h_gate.remove()
            if was_training:
                self.model.train()

        dst_diff = (clean_dst - pruned_dst).abs()
        per_dst_max_shift = dst_diff.max(dim=0).values
        max_act_shift = float(per_dst_max_shift.max().item()) if per_dst_max_shift.numel() > 0 else 0.0
        mean_act_shift = float(dst_diff.mean().item()) if dst_diff.numel() > 0 else 0.0

        if clean_gate is not None and pruned_gate is not None:
            gate_diff = (clean_gate - pruned_gate).abs()
            max_gate_shift = float(gate_diff.max().item())
            mean_gate_shift = float(gate_diff.mean().item())
        else:
            max_gate_shift = 0.0
            mean_gate_shift = 0.0

        affected_dst = torch.nonzero(per_dst_max_shift > readout_tolerance, as_tuple=False).squeeze(-1)
        affected_list = (
            [int(affected_dst.item())]
            if affected_dst.ndim == 0 and affected_dst.numel() == 1
            else [int(i) for i in affected_dst.tolist()]
        )

        return DownstreamReadoutReport(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neurons=list(src_neurons),
            applied_thresholds=[round(float(t), 5) for t in applied_thresholds],
            max_downstream_activation_shift=round(max_act_shift, 5),
            mean_downstream_activation_shift=round(mean_act_shift, 5),
            max_downstream_gate_shift=round(max_gate_shift, 5),
            mean_downstream_gate_shift=round(mean_gate_shift, 5),
            readout_tolerance=readout_tolerance,
            affected_downstream_neurons=affected_list,
            is_benign_spillover=(len(affected_list) == 0),
        )
