from __future__ import annotations

from contextlib import nullcontext
import math
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from .discovery import LayerDiscovery
from .hooks import HookSet, default_forward, extract_logits, flat_neuron_view, valid_position_mask
from .types import (
    CircuitEdge,
    CircuitGraph,
    CombinationReadoutReport,
    InterventionMode,
    InterventionSpec,
    PoleType,
    ResidualCoordinateEntry,
    ResidualSinkReport,
    UpstreamAttributionBundle,
)

if TYPE_CHECKING:
    from .injector import NeuralInjector

_ATTN_READER_LEAVES = ("q_proj", "k_proj", "v_proj", "qkv_proj", "c_attn", "query", "key", "value", "query_key_value", "Wqkv")
_ATTN_NORM_NAMES = ("input_layernorm", "ln_1", "pre_attention_layernorm", "attention_norm", "layernorm_before", "norm1")


def _sdpa_math_context():
    """Forward-mode AD through scaled_dot_product_attention needs the math kernel."""
    try:
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    except Exception:
        pass
    return nullcontext()


def _pole_mask(col: torch.Tensor, pole: PoleType) -> torch.Tensor:
    if pole == PoleType.POSITIVE:
        return col > 0
    if pole == PoleType.NEGATIVE:
        return col < 0
    return col != 0


class CircuitTracer:
    """Static (virtual weight) and causal (ablation, Jacobian, attribution) neuron-to-neuron tracing."""

    def __init__(self, injector: "NeuralInjector"):
        self.injector = injector
        self.model = injector.model

    # ------------------------------------------------------------------
    # 1. Virtual weights
    # ------------------------------------------------------------------
    def _dst_norm_scale(self, dst_layer: str, d_model: int, device: torch.device) -> Optional[torch.Tensor]:
        norm = LayerDiscovery.resolve_pre_mlp_norm(self.model, LayerDiscovery.split_expert_layer_name(dst_layer)[0])
        return LayerDiscovery.extract_norm_scale(norm, d_model=d_model, device=device)

    @torch.no_grad()
    def compute_virtual_weights(self, src_layer: str, dst_layer: str, src_neuron: int,
                                apply_pre_norm_scaling: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Direct-path coupling from src neuron i to every dst neuron j through the residual stream:
          up:   <W_up_dst[j] ⊙ gamma, W_down_src[:, i]>,  gate: <W_gate_dst[j] ⊙ gamma, W_down_src[:, i]>,
          cos:  cosine between the write vector and the value read vector.
        gamma is the destination's pre-MLP norm scale (the norm's data-dependent denominator is ignored).
        """
        w = LayerDiscovery.neuron_write_vectors(self.model, src_layer)
        up, gate = LayerDiscovery.neuron_read_vectors(self.model, dst_layer)
        if w is None or up is None:
            raise ValueError(f"Virtual weights need a write matrix for '{src_layer}' and read matrix for '{dst_layer}'.")
        v = w[src_neuron].float()
        up = up.float().to(v.device)
        if apply_pre_norm_scaling:
            g = self._dst_norm_scale(dst_layer, v.shape[0], v.device)
            if g is not None:
                v = v * g
        vu = up @ v
        cos = F.normalize(up, dim=-1) @ F.normalize(v, dim=0)
        vg = gate.float().to(v.device) @ v if gate is not None else torch.zeros_like(vu)
        return vu, vg, cos

    @torch.no_grad()
    def compute_incoming_virtual_weights(self, src_layer: str, dst_layer: str, dst_neuron: int,
                                         apply_pre_norm_scaling: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = LayerDiscovery.neuron_write_vectors(self.model, src_layer)
        up, gate = LayerDiscovery.neuron_read_vectors(self.model, dst_layer)
        if w is None or up is None:
            raise ValueError("Incoming virtual weights need write/read matrices.")
        W = w.float()
        if apply_pre_norm_scaling:
            g = self._dst_norm_scale(dst_layer, W.shape[1], W.device)
            if g is not None:
                W = W * g.unsqueeze(0)
        r = up[dst_neuron].float().to(W.device)
        vu = W @ r
        cos = F.normalize(W, dim=-1) @ F.normalize(r, dim=0)
        vg = W @ gate[dst_neuron].float().to(W.device) if gate is not None else torch.zeros_like(vu)
        return vu, vg, cos

    # ------------------------------------------------------------------
    # 2. Forward-mode AD
    # ------------------------------------------------------------------
    @torch.no_grad()
    def trace_forward_ad_all_layers(self, src_layer: str, src_neuron: int, inputs: Any, dst_layers: Optional[List[str]] = None,
                                    src_pole: PoleType = PoleType.POSITIVE, forward_fn: Optional[Callable] = None,
                                    position_mode: str = "all_active") -> Dict[str, torch.Tensor]:
        """
        One forward pass with a tangent on the source neuron gives d(A_dst)/d(a_src) for every neuron
        of every listed downstream layer: mean over the seeded positions of the summed tangent.
        `position_mode="focal_token"` seeds only the position of the source's strongest pole activation.
        """
        run = forward_fn or default_forward
        all_layers = list(self.injector.list_targetable_layers().keys())
        if dst_layers is None:
            dst_layers = all_layers[all_layers.index(src_layer) + 1:] if src_layer in all_layers else []
        hs = HookSet(self.model)
        src_site = hs.site(src_layer)
        if src_site.target == "moe_expert":
            raise NotImplementedError("Forward-mode AD from routed MoE expert neurons is not supported.")
        mask_holder: List[torch.Tensor] = []
        out: Dict[str, torch.Tensor] = {}

        def _seed(act: torch.Tensor, _g: Optional[torch.Tensor]) -> torch.Tensor:
            flat = flat_neuron_view(act.detach(), src_site.is_conv2d)
            col = flat[:, src_neuron]
            if position_mode == "focal_token":
                score = col if src_pole == PoleType.POSITIVE else (-col if src_pole == PoleType.NEGATIVE else col.abs())
                m = torch.zeros_like(col, dtype=torch.bool)
                m[int(score.argmax())] = True
            else:
                m = _pole_mask(col, src_pole)
            mask_holder.append(m)
            tan = torch.zeros_like(flat)
            tan[m, src_neuron] = 1.0
            if src_site.is_conv2d and act.ndim == 4:
                b, c, h, w = act.shape
                tan = tan.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            else:
                tan = tan.reshape_as(act)
            return fwAD.make_dual(act, tan)

        hs.add_modifier(src_layer, _seed)
        for d in dst_layers:
            d_site = hs.site(d)

            def _read(act: torch.Tensor, _g, _r, name=d, conv=d_site.is_conv2d, n=d_site.num_neurons) -> None:
                _, tan = fwAD.unpack_dual(act)
                if tan is None:
                    out[name] = torch.zeros(n)
                    return
                t = flat_neuron_view(tan, conv).float()
                m = mask_holder[-1] if mask_holder and mask_holder[-1].shape[0] == t.shape[0] else None
                out[name] = (t[m].sum(0) / max(int(m.sum()), 1)) if m is not None else t.mean(0)

            hs.add_capture(d, _read)
        was = self.model.training
        self.model.eval()
        try:
            with self.injector.clean_forward(), _sdpa_math_context(), fwAD.dual_level(), hs.active():
                run(self.model, inputs)
        finally:
            if was:
                self.model.train()
        return {k: v.detach().cpu() for k, v in out.items()}

    # ------------------------------------------------------------------
    # 3. Downstream causal tracing
    # ------------------------------------------------------------------
    def _gate_up_capture(self, hs: HookSet, layer: str, store: Dict[str, List[torch.Tensor]]) -> bool:
        """Capture dst gate g and value u (for path decomposition). Returns False if not available."""
        site = hs.site(layer)
        if not site.gated or site.target == "moe_expert":
            return False
        up_mod, up_kind = LayerDiscovery.up_source(self.model, layer)
        if up_mod is None:
            return False

        def _u(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if up_kind == "second_half":
                t = t[..., t.shape[-1] // 2:]
            store.setdefault("u", []).append(t.detach().float())

        h = up_mod.register_forward_hook(_u)
        store.setdefault("_handles", []).append(h)  # type: ignore[arg-type]
        hs.add_capture(layer, lambda a, g, r: store.setdefault("g", []).append(g.detach().float()), need_gate=True)
        return True

    def trace_circuit(self, src_layer: str, dst_layer: str, src_neuron: int, inputs: Any,
                      forward_fn: Optional[Callable] = None, src_pole: PoleType = PoleType.POSITIVE,
                      compute_jacobian: bool = True, top_k: int = 16, attention_mask: Optional[torch.Tensor] = None,
                      position_mode: str = "all_active") -> CircuitGraph:
        """
        Effect of ablating one pole of (src_layer, src_neuron) on every neuron of dst_layer, averaged
        over the positions where the source pole is active. For gated destinations the measured change
        is split into the first-order gate path act'(g)u·Δg and value path act(g)·Δu, which decides
        whether the edge is gating.
        """
        run = forward_fn or default_forward
        if attention_mask is None and isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
        hs = HookSet(self.model)
        src_site, dst_site = hs.site(src_layer), hs.site(dst_layer)
        src_store: List[torch.Tensor] = []
        dst_clean: List[torch.Tensor] = []
        hs.add_capture(src_layer, lambda a, g, r: src_store.append(a.detach().float()))
        hs.add_capture(dst_layer, lambda a, g, r: dst_clean.append(a.detach().float()))
        gu_clean: Dict[str, List[Any]] = {}
        has_paths = self._gate_up_capture(hs, dst_layer, gu_clean)
        was = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self.injector.clean_forward(), hs.active():
                run(self.model, inputs)
        finally:
            for h in gu_clean.pop("_handles", []):
                h.remove()
        src_flat = flat_neuron_view(src_store[-1], src_site.is_conv2d)
        dst_flat = flat_neuron_view(dst_clean[-1], dst_site.is_conv2d)
        if src_flat.shape[0] != dst_flat.shape[0]:
            raise ValueError("Source and destination activations do not share positions.")
        valid = valid_position_mask(attention_mask, src_flat.shape[0], src_flat.device)
        col = src_flat[:, src_neuron]
        if position_mode == "focal_token":
            score = torch.where(valid, col if src_pole != PoleType.NEGATIVE else -col, torch.full_like(col, float("-inf")))
            active = torch.zeros_like(valid)
            active[int(score.argmax())] = True
        else:
            active = _pole_mask(col, src_pole) & valid
        if not bool(active.any()):
            raise ValueError(f"Source neuron {src_neuron} has no {src_pole.value} activations on these inputs.")

        jac = torch.zeros(dst_site.num_neurons)
        if compute_jacobian and src_site.target != "moe_expert":
            jd = self.trace_forward_ad_all_layers(src_layer, src_neuron, inputs, [dst_layer], src_pole, run, position_mode)
            jac = jd.get(dst_layer, jac)

        spec = InterventionSpec(src_layer, [src_neuron], InterventionMode.ZERO_ABLATE, src_pole)
        hs2 = HookSet(self.model)
        dst_abl: List[torch.Tensor] = []
        hs2.add_capture(dst_layer, lambda a, g, r: dst_abl.append(a.detach().float()))
        gu_abl: Dict[str, List[Any]] = {}
        if has_paths:
            self._gate_up_capture(hs2, dst_layer, gu_abl)
        try:
            with torch.no_grad(), self.injector.temporary_interventions([spec]), hs2.active():
                run(self.model, inputs)
        finally:
            for h in gu_abl.pop("_handles", []):
                h.remove()
            if was:
                self.model.train()
        abl_flat = flat_neuron_view(dst_abl[-1], dst_site.is_conv2d)
        delta = dst_flat[active] - abl_flat[active]
        mean_delta = delta.mean(0)
        max_idx = delta.abs().argmax(0)
        max_delta = delta[max_idx, torch.arange(delta.shape[1])]
        clean_mean = dst_flat[active].mean(0)
        abl_mean = abl_flat[active].mean(0)

        gate_term = up_term = None
        if has_paths and gu_clean.get("g") and gu_clean.get("u") and gu_abl.get("g") and gu_abl.get("u"):
            act_fn = LayerDiscovery.resolve_mlp_activation(self.model, dst_layer)
            if act_fn is not None:
                g0 = flat_neuron_view(gu_clean["g"][-1], False)[active]
                u0 = flat_neuron_view(gu_clean["u"][-1], False)[active]
                g1 = flat_neuron_view(gu_abl["g"][-1], False)[active]
                u1 = flat_neuron_view(gu_abl["u"][-1], False)[active]
                with torch.enable_grad():
                    gg = g0.clone().requires_grad_(True)
                    y = act_fn(gg)
                    dact = torch.autograd.grad(y.sum(), gg)[0]
                gate_term = (dact * u0 * (g0 - g1)).mean(0)
                up_term = (y.detach() * (u0 - u1)).mean(0)

        vu, vg, cos = self._safe_virtual(src_layer, dst_layer, src_neuron, dst_site.num_neurons)
        k = min(top_k, dst_site.num_neurons)
        top = torch.topk(mean_delta.abs(), k=k).indices.tolist()
        edges = []
        for j in top:
            edges.append(CircuitEdge(
                src_layer=src_layer, src_neuron=src_neuron, src_pole=src_pole, dst_layer=dst_layer, dst_neuron=j,
                dst_pole_tendency=PoleType.POSITIVE if float(clean_mean[j]) >= 0 else PoleType.NEGATIVE,
                mean_causal_delta=float(mean_delta[j]), max_causal_delta=float(max_delta[j]),
                clean_dst_mean=float(clean_mean[j]), ablated_dst_mean=float(abl_mean[j]),
                jacobian_sensitivity=float(jac[j]), virtual_weight_up=float(vu[j]), virtual_weight_gate=float(vg[j]),
                static_alignment_cos=float(cos[j]),
                gate_path_delta=float(gate_term[j]) if gate_term is not None else None,
                up_path_delta=float(up_term[j]) if up_term is not None else None,
            ))
        graph = CircuitGraph(src_layer, dst_layer, src_neuron, src_pole, edges)
        self.injector.ecology_map.register_circuit_graph(graph)
        return graph

    def _safe_virtual(self, src: str, dst: str, i: int, n_dst: int):
        try:
            vu, vg, cos = self.compute_virtual_weights(src, dst, i)
            return vu.cpu(), vg.cpu(), cos.cpu()
        except ValueError:
            z = torch.zeros(n_dst)
            return z, z, z

    # ------------------------------------------------------------------
    # 4. Upstream attribution and incoming circuits
    # ------------------------------------------------------------------
    def _attribution_into(self, src_layer: str, dst_layer: str, dst_neuron: int, inputs: Any, dst_pole: PoleType,
                          run: Callable, attention_mask: Optional[torch.Tensor], focal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gradient-times-activation attribution of every src neuron into the dst neuron:
        target = mean of the dst neuron over its active pole positions (or its single strongest
        position when `focal`), attribution_i = sum_positions a_i * d target / d a_i.
        Returns (attribution [n_src], jacobian [n_src], src mean over target positions [n_src], target mask).
        """
        hs = HookSet(self.model)
        src_site, dst_site = hs.site(src_layer), hs.site(dst_layer)
        if src_site.target == "moe_expert" or dst_site.target == "moe_expert":
            raise NotImplementedError("Gradient attribution through routed MoE expert rows is not supported.")
        holder: Dict[str, torch.Tensor] = {}

        def _leaf(act: torch.Tensor, _g) -> torch.Tensor:
            t = act.detach().clone().requires_grad_(True)
            holder["src"] = t
            return t

        hs.add_modifier(src_layer, _leaf)
        hs.add_capture(dst_layer, lambda a, g, r: holder.__setitem__("dst", a))
        with torch.enable_grad(), self.injector.clean_forward(), hs.active():
            run(self.model, inputs)
            dst = flat_neuron_view(holder["dst"], dst_site.is_conv2d)
            col = dst[:, dst_neuron]
            valid = valid_position_mask(attention_mask, col.shape[0], col.device)
            if focal:
                score = torch.where(valid, col.detach() if dst_pole != PoleType.NEGATIVE else -col.detach(),
                                    torch.full_like(col, float("-inf")))
                m = torch.zeros_like(valid)
                m[int(score.argmax())] = True
            else:
                m = _pole_mask(col.detach(), dst_pole) & valid
                if not bool(m.any()):
                    raise ValueError(f"Target neuron {dst_neuron} has no {dst_pole.value} activations on these inputs.")
            target = col[m].float().mean()
            grad = torch.autograd.grad(target, holder["src"], allow_unused=True)[0]
        src = flat_neuron_view(holder["src"].detach().float(), src_site.is_conv2d)
        g = flat_neuron_view(grad.float(), src_site.is_conv2d) if grad is not None else torch.zeros_like(src)
        attribution = (src * g).sum(0)
        jac = g.sum(0)
        src_mean = src[m].mean(0) if src.shape[0] == m.shape[0] else src.mean(0)
        return attribution, jac, src_mean, m

    def trace_incoming_circuit(self, src_layer: str, dst_layer: str, dst_neuron: int, inputs: Any,
                               forward_fn: Optional[Callable] = None, dst_pole: PoleType = PoleType.POSITIVE,
                               top_k: int = 8, attention_mask: Optional[torch.Tensor] = None) -> CircuitGraph:
        """
        Ranks every src neuron by |gradient x activation| attribution into the target pole, then
        verifies the top `top_k` by ablating each one's active pole and measuring the target's change.
        """
        run = forward_fn or default_forward
        if attention_mask is None and isinstance(inputs, dict):
            attention_mask = inputs.get("attention_mask")
        attribution, jac, src_mean, m = self._attribution_into(src_layer, dst_layer, dst_neuron, inputs, dst_pole,
                                                               run, attention_mask, focal=False)
        vu, vg, cos = (t.cpu() for t in self._safe_incoming(src_layer, dst_layer, dst_neuron, attribution.numel()))
        top = torch.topk(attribution.abs(), k=min(top_k, attribution.numel())).indices.tolist()
        hs = HookSet(self.model)
        dst_site = hs.site(dst_layer)
        clean: List[torch.Tensor] = []
        hs.add_capture(dst_layer, lambda a, g, r: clean.append(a.detach().float()))
        with torch.no_grad(), self.injector.clean_forward(), hs.active():
            run(self.model, inputs)
        clean_col = flat_neuron_view(clean[-1], dst_site.is_conv2d)[:, dst_neuron]
        edges = []
        for s in top:
            s_pole = PoleType.POSITIVE if float(src_mean[s]) >= 0 else PoleType.NEGATIVE
            spec = InterventionSpec(src_layer, [s], InterventionMode.ZERO_ABLATE, s_pole)
            abl: List[torch.Tensor] = []
            hs2 = HookSet(self.model)
            hs2.add_capture(dst_layer, lambda a, g, r: abl.append(a.detach().float()))
            with torch.no_grad(), self.injector.temporary_interventions([spec]), hs2.active():
                run(self.model, inputs)
            ab_col = flat_neuron_view(abl[-1], dst_site.is_conv2d)[:, dst_neuron]
            d = clean_col[m] - ab_col[m]
            edges.append(CircuitEdge(
                src_layer=src_layer, src_neuron=s, src_pole=s_pole, dst_layer=dst_layer, dst_neuron=dst_neuron,
                dst_pole_tendency=dst_pole, mean_causal_delta=float(d.mean()), max_causal_delta=float(d[d.abs().argmax()]),
                clean_dst_mean=float(clean_col[m].mean()), ablated_dst_mean=float(ab_col[m].mean()),
                jacobian_sensitivity=float(jac[s]), virtual_weight_up=float(vu[s]), virtual_weight_gate=float(vg[s]),
                static_alignment_cos=float(cos[s]),
            ))
        edges.sort(key=lambda e: abs(e.mean_causal_delta), reverse=True)
        graph = CircuitGraph(src_layer, dst_layer, edges[0].src_neuron if edges else -1, dst_pole, edges, dst_neuron=dst_neuron)
        self.injector.ecology_map.register_circuit_graph(graph)
        return graph

    def _safe_incoming(self, src: str, dst: str, j: int, n_src: int):
        try:
            return self.compute_incoming_virtual_weights(src, dst, j)
        except ValueError:
            z = torch.zeros(n_src)
            return z, z, z

    @staticmethod
    def _spherical_kmeans(X: torch.Tensor, k: int, gen: torch.Generator, n_init: int = 10, iters: int = 100) -> Tuple[torch.Tensor, float]:
        """Cosine k-means (rows of X are unit vectors). Returns (labels, inertia as 1 - mean cos to centroid)."""
        n = X.shape[0]
        best_labels, best_score = None, float("inf")
        for _ in range(n_init):
            first = int(torch.randint(n, (1,), generator=gen))
            cents = [X[first]]
            for _c in range(1, k):
                sims = torch.stack([X @ c for c in cents], 1).max(1).values
                d = torch.clamp(1.0 - sims, min=0.0)
                probs = d / d.sum() if d.sum() > 0 else torch.full((n,), 1.0 / n)
                cents.append(X[int(torch.multinomial(probs, 1, generator=gen))])
            C = torch.stack(cents)
            labels = torch.zeros(n, dtype=torch.long)
            for _it in range(iters):
                new = (X @ C.T).argmax(1)
                if torch.equal(new, labels) and _it > 0:
                    break
                labels = new
                for c in range(k):
                    mem = X[labels == c]
                    if mem.shape[0]:
                        C[c] = F.normalize(mem.sum(0), dim=0)
            score = float((1.0 - (X * C[labels]).sum(1)).mean())
            if score < best_score:
                best_score, best_labels = score, labels.clone()
        return best_labels, best_score

    @staticmethod
    def _silhouette_cosine(X: torch.Tensor, labels: torch.Tensor) -> float:
        D = torch.clamp(1.0 - X @ X.T, min=0.0)
        n = X.shape[0]
        ks = labels.unique()
        if len(ks) < 2:
            return float("nan")
        s = torch.zeros(n)
        for i in range(n):
            same = labels == labels[i]
            n_same = int(same.sum()) - 1
            if n_same == 0:
                s[i] = 0.0
                continue
            a = D[i, same].sum() / n_same
            b = min(float(D[i, labels == c].mean()) for c in ks if c != labels[i])
            s[i] = (b - a) / max(float(a), b) if max(float(a), b) > 0 else 0.0
        return float(s.mean())

    def cluster_incoming_attribution_bundles(self, src_layer: str, dst_layer: str, dst_neuron: int, contexts: List[Any],
                                             dst_pole: PoleType = PoleType.POSITIVE, forward_fn: Optional[Callable] = None,
                                             top_k_upstream: int = 24, max_bundles: int = 6, seed: int = 0,
                                             context_labels: Optional[List[str]] = None, n_permutations: int = 100,
                                             alpha: float = 0.05) -> List[UpstreamAttributionBundle]:
        """
        'We also cluster the attribution flowing into the neuron': for each of the target's activating
        contexts, attribute the target's strongest pole activation to every upstream neuron; the
        `top_k_upstream` neurons by total |attribution| are clustered by their attribution profiles
        across contexts (cosine k-means, k in [2, max_bundles] chosen by silhouette).
        The clustering is kept only if its silhouette exceeds the (1 - alpha) quantile of silhouettes
        obtained after independently permuting every profile across contexts (which destroys shared
        structure); otherwise all upstream neurons form a single bundle.
        """
        run = forward_fn or default_forward
        if len(contexts) < 3:
            raise ValueError("Bundle clustering needs at least three activating contexts.")
        rows = []
        for ctx in contexts:
            am = ctx.get("attention_mask") if isinstance(ctx, dict) else None
            a, _, _, _ = self._attribution_into(src_layer, dst_layer, dst_neuron, ctx, dst_pole, run, am, focal=True)
            rows.append(a.cpu())
        A = torch.stack(rows)  # [n_ctx, n_src]
        total = A.abs().sum(0)
        top = torch.topk(total, k=min(top_k_upstream, total.numel())).indices
        P = A[:, top].T  # [M, n_ctx]
        keep = P.norm(dim=1) > 0
        top, P = top[keep], P[keep]
        X = F.normalize(P, dim=1)
        M = X.shape[0]
        gen = torch.Generator().manual_seed(seed)

        def _best(Xm: torch.Tensor) -> Tuple[torch.Tensor, float]:
            bl, bs = torch.zeros(Xm.shape[0], dtype=torch.long), float("nan")
            for k in range(2, min(max_bundles, Xm.shape[0] - 1) + 1):
                labels, _ = self._spherical_kmeans(Xm, k, gen)
                sil = self._silhouette_cosine(Xm, labels)
                if math.isfinite(sil) and (not math.isfinite(bs) or sil > bs):
                    bl, bs = labels, sil
            return bl, bs

        best_labels, best_sil = _best(X)
        null_p = float("nan")
        if math.isfinite(best_sil) and n_permutations > 0:
            null = []
            for _ in range(n_permutations):
                Pp = torch.stack([row[torch.randperm(row.numel(), generator=gen)] for row in P])
                null.append(_best(F.normalize(Pp, dim=1))[1])
            null_t = torch.tensor([v for v in null if math.isfinite(v)])
            null_p = float((null_t >= best_sil).float().mean()) if null_t.numel() else float("nan")
        if not (math.isfinite(null_p) and null_p < alpha):
            best_labels = torch.zeros(M, dtype=torch.long)
        grand = float(total[top].sum()) or 1.0
        bundles = []
        for b in best_labels.unique().tolist():
            idx = (best_labels == b).nonzero().reshape(-1)
            members = top[idx]
            Xm = X[idx]
            mean_cos = float(((Xm @ Xm.T).sum() - len(idx)) / max(len(idx) * (len(idx) - 1), 1)) if len(idx) > 1 else 1.0
            drive = A[:, members].sum(1) * (1.0 if dst_pole != PoleType.NEGATIVE else -1.0)
            order = torch.argsort(drive, descending=True).tolist()
            bundles.append(UpstreamAttributionBundle(
                dst_layer=dst_layer, dst_neuron=dst_neuron, dst_pole=dst_pole, src_layer=src_layer, bundle_id=b,
                upstream_neurons=[int(x) for x in members.tolist()],
                attribution_share=float(total[members].sum()) / grand,
                mean_within_cosine=mean_cos, silhouette=best_sil if math.isfinite(best_sil) else float('nan'),
                silhouette_null_p=null_p,
                driving_context_indices=order[: min(5, len(order))],
                driving_contexts=[context_labels[i] for i in order[:5]] if context_labels else [],
            ))
        bundles.sort(key=lambda x: x.attribution_share, reverse=True)
        for i, b in enumerate(bundles):
            b.bundle_id = i
        return bundles

    # ------------------------------------------------------------------
    # 5. Residual stream sinks
    # ------------------------------------------------------------------
    def _norm_for_attention(self, block: nn.Module) -> Optional[nn.Module]:
        for nm in _ATTN_NORM_NAMES:
            if hasattr(block, nm):
                return getattr(block, nm)
        return None

    @torch.no_grad()
    def audit_residual_stream_sinks(self, residual_site: str, batches: Iterable[Any], forward_fn: Callable = default_forward,
                                    neuron_layers: Optional[List[str]] = None) -> ResidualSinkReport:
        """
        'Looking at the Residual Stream': for the residual stream entering module `residual_site`
        (e.g. a decoder block, or the final norm), compares each coordinate's share of activation mass
        E[x_d^2] with its share of what downstream readers read, sum over readers of ||W[:, d] * gamma_d||^2
        (attention q/k/v and MLP gate/up of later blocks, and the unembedding, each with its own norm scale).
        A coordinate is a candidate when it holds more than the uniform share 1/d_model of the mass and a
        larger share of the mass than of the reading; candidates are listed by mass share.
        """
        mod = self.model.get_submodule(residual_site)
        sq: Dict[str, Any] = {"sum": None, "n": 0}
        cur: Dict[str, Any] = {}

        def _pre(_m, inp):
            x = inp[0] if inp and isinstance(inp[0], torch.Tensor) else None
            if x is None:
                return
            flat = x.detach().float().reshape(-1, x.shape[-1])
            mk = cur.get("mask")
            if mk is not None and mk.numel() == flat.shape[0]:
                flat = flat[mk.reshape(-1).bool().to(flat.device)]
            s = (flat ** 2).sum(0)
            sq["sum"] = s if sq["sum"] is None else sq["sum"] + s
            sq["n"] += flat.shape[0]

        h = mod.register_forward_pre_hook(_pre)
        try:
            with self.injector.clean_forward():
                for b in batches:
                    cur["mask"] = b.get("attention_mask") if isinstance(b, dict) else None
                    forward_fn(self.model, b)
        finally:
            h.remove()
        if sq["sum"] is None:
            raise ValueError(f"No inputs were observed at '{residual_site}'.")
        mass = sq["sum"].cpu()
        d_model = mass.numel()
        mass_share = mass / mass.sum()

        order = [n for n, _ in self.model.named_modules() if n]
        site_pos = order.index(residual_site)
        read = torch.zeros(d_model)
        # MLP readers of later layers
        for name in self.injector.list_targetable_layers():
            base = LayerDiscovery.split_expert_layer_name(name)[0]
            if base not in order or order.index(base) <= site_pos:
                continue
            up, gate = LayerDiscovery.neuron_read_vectors(self.model, name)
            g = self._dst_norm_scale(name, d_model, torch.device("cpu"))
            for Wr in (up, gate):
                if Wr is not None and Wr.shape[1] == d_model:
                    Wf = Wr.float().cpu() * (g.cpu() if g is not None else 1.0)
                    read += (Wf ** 2).sum(0)
        # attention readers of later blocks
        for name, m in self.model.named_modules():
            if not name or order.index(name) <= site_pos or name.split(".")[-1] not in _ATTN_READER_LEAVES:
                continue
            w = getattr(m, "weight", None)
            if not isinstance(w, torch.Tensor) or w.ndim != 2:
                continue
            Wf = w.detach().float().cpu()
            if m.__class__.__name__ == "Conv1D":
                Wf = Wf.T
            if Wf.shape[1] != d_model:
                continue
            parts = name.split(".")
            gamma = None
            for depth in range(len(parts) - 1, 0, -1):
                blk = self.model.get_submodule(".".join(parts[:depth]))
                nm = self._norm_for_attention(blk)
                if nm is not None:
                    gamma = LayerDiscovery.extract_norm_scale(nm, d_model=d_model, device=torch.device("cpu"))
                    break
            read += ((Wf * (gamma if gamma is not None else 1.0)) ** 2).sum(0)
        # unembedding
        emb = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
        if emb is not None and isinstance(getattr(emb, "weight", None), torch.Tensor) and emb.weight.shape[1] == d_model:
            fn = LayerDiscovery.resolve_final_norm(self.model)
            gamma = LayerDiscovery.extract_norm_scale(fn, d_model=d_model, device=torch.device("cpu"))
            read += ((emb.weight.detach().float().cpu() * (gamma if gamma is not None else 1.0)) ** 2).sum(0)
        read_share = read / read.sum() if read.sum() > 0 else read
        uniform = 1.0 / d_model
        entries = []
        sinks = []
        for d in torch.argsort(mass_share, descending=True).tolist():
            # more than its uniform share of the mass, and more of the mass than of what readers read
            cand = float(mass_share[d]) > uniform and float(mass_share[d]) > float(read_share[d])
            if cand:
                sinks.append(d)
            entries.append(ResidualCoordinateEntry(d, float(mass_share[d]), float(read_share[d]), cand))
        write_share: Dict[str, float] = {}
        if neuron_layers and sinks:
            for nl in neuron_layers:
                W = LayerDiscovery.neuron_write_vectors(self.model, nl)
                if W is None:
                    continue
                Wf = W.float().cpu()
                share = (Wf[:, sinks] ** 2).sum(1) / torch.clamp((Wf ** 2).sum(1), min=1e-30)
                for i, v in enumerate(share.tolist()):
                    write_share[f"{nl}#{i}"] = v
        return ResidualSinkReport(residual_site, d_model, int(sq["n"]), entries, sinks, write_share)

    # ------------------------------------------------------------------
    # 6. 'Nothing reads that combination'
    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify_combination_readout(self, src_layer: str, src_neurons: List[int], coefficients: Optional[List[float]] = None,
                                   dst_layers: Optional[List[str]] = None, null_samples: int = 1000, seed: int = 0,
                                   alpha: float = 0.05, causal_inputs: Any = None,
                                   forward_fn: Optional[Callable] = None) -> CombinationReadoutReport:
        """
        For a hypothesised hidden combination c of src neurons, measures the largest |cosine| between
        sum_i c_i W_down[:, i] and any downstream reader row (value and gate rows of `dst_layers`,
        norm-scaled), and compares it with (a) the best single constituent and (b) the same statistic
        for `null_samples` random neuron sets of the same size and coefficients. The combination is
        'read' only if it beats both its constituents and the (1 - alpha) quantile of the null.
        """
        coeffs = coefficients or [1.0] * len(src_neurons)
        if len(coeffs) != len(src_neurons):
            raise ValueError("coefficients must align with src_neurons")
        W = LayerDiscovery.neuron_write_vectors(self.model, src_layer)
        if W is None:
            raise ValueError(f"'{src_layer}' has no residual write matrix.")
        W = W.float().cpu()
        layers = list(self.injector.list_targetable_layers().keys())
        if dst_layers is None:
            dst_layers = layers[layers.index(src_layer) + 1:] if src_layer in layers else []
        readers: List[Tuple[str, str, torch.Tensor]] = []
        for d in dst_layers:
            up, gate = LayerDiscovery.neuron_read_vectors(self.model, d)
            g = self._dst_norm_scale(d, W.shape[1], torch.device("cpu"))
            for kind, R in (("up", up), ("gate", gate)):
                if R is not None:
                    Rf = R.float().cpu() * (g.cpu() if g is not None else 1.0)
                    readers.append((d, kind, F.normalize(Rf, dim=1)))
        if not readers:
            raise ValueError("No downstream readers found.")

        def _max_cos(vec: torch.Tensor) -> Tuple[float, Tuple[str, int, str]]:
            v = F.normalize(vec, dim=0)
            best, where = -1.0, ("", -1, "")
            for d, kind, R in readers:
                c = (R @ v).abs()
                j = int(c.argmax())
                if float(c[j]) > best:
                    best, where = float(c[j]), (d, j, kind)
            return best, where

        ct = torch.tensor(coeffs, dtype=torch.float32)
        comb = (W[src_neurons] * ct.unsqueeze(1)).sum(0)
        obs, where = _max_cos(comb)
        single = max(_max_cos(W[i])[0] for i in src_neurons)
        rng = random.Random(seed)
        pool = list(range(W.shape[0]))
        null = []
        for _ in range(null_samples):
            idx = rng.sample(pool, k=len(src_neurons))
            null.append(_max_cos((W[idx] * ct.unsqueeze(1)).sum(0))[0])
        null_t = torch.tensor(null)
        q = float(torch.quantile(null_t, 1.0 - alpha)) if null else float("nan")
        pct = float((null_t < obs).float().mean()) if null else float("nan")

        causal = None
        if causal_inputs is not None and where[1] >= 0:
            run = forward_fn or default_forward
            dl, j, _ = where
            hs_site = HookSet(self.model).site(dl)

            def _read_dst(specs):
                hs = HookSet(self.model)
                got: List[torch.Tensor] = []
                hs.add_capture(dl, lambda a, g, r: got.append(a.detach().float()))
                with self.injector.temporary_interventions(specs), hs.active():
                    run(self.model, causal_inputs)
                return flat_neuron_view(got[-1], hs_site.is_conv2d)[:, j]

            base = _read_dst([])
            joint = _read_dst([InterventionSpec(src_layer, list(src_neurons), InterventionMode.ZERO_ABLATE)])
            indiv = sum((base - _read_dst([InterventionSpec(src_layer, [i], InterventionMode.ZERO_ABLATE)])) for i in src_neurons)
            causal = {
                "reader_mean_clean": float(base.mean()),
                "joint_ablation_delta": float((base - joint).mean()),
                "sum_individual_ablation_deltas": float(indiv.mean()),
                "interaction": float(((base - joint) - indiv).mean()),
            }
        return CombinationReadoutReport(
            src_layer=src_layer, dst_layers=dst_layers, src_neurons=list(src_neurons), coefficients=list(coeffs),
            combination_max_reader_cosine=obs, best_reader=where, best_single_constituent_cosine=single,
            null_samples=null_samples, null_mean=float(null_t.mean()) if null else float("nan"), null_p95=q,
            combination_percentile_in_null=pct, reads_combination_beyond_constituents=bool(obs > single and obs > q),
            causal_interaction=causal,
        )

    # ------------------------------------------------------------------
    # 7. Direct vs indirect effect ('Memorization and Composition')
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decompose_direct_vs_indirect_effect(self, layer_name: str, neuron_idx: int, inputs: Any, target_token_id: int,
                                            downstream_layer: Optional[str] = None,
                                            rescue_downstream_neurons: Optional[List[int]] = None,
                                            pole: PoleType = PoleType.BOTH, forward_fn: Optional[Callable] = None,
                                            position: int = -1) -> Dict[str, float]:
        """
        Probability of `target_token_id` at `position` when the neuron is
          * untouched (clean),
          * fully ablated (total effect),
          * removed only from the direct path to the unembedding: its write a_i * W_down[:, i] is
            subtracted from the final residual stream while every downstream layer sees the clean
            value (direct effect),
          * ablated while the listed downstream neurons are restored to their clean values (rescue).
        """
        run = forward_fn or default_forward
        hs = HookSet(self.model)
        site = hs.site(layer_name)
        if site.target != "input":
            raise NotImplementedError("Direct-path removal needs a down-projection (input-hooked) neuron layer.")
        W = LayerDiscovery.neuron_write_vectors(self.model, layer_name)
        cap: List[torch.Tensor] = []
        hs.add_capture(layer_name, lambda a, g, r: cap.append(a.detach()))
        rescue_clean: List[torch.Tensor] = []
        if downstream_layer and rescue_downstream_neurons:
            hs.add_capture(downstream_layer, lambda a, g, r: rescue_clean.append(a.detach().clone()))

        def _prob(logits: torch.Tensor) -> float:
            return float(F.softmax(logits.float(), dim=-1)[0, position, target_token_id])

        with self.injector.clean_forward(), hs.active():
            p_clean = _prob(extract_logits(run(self.model, inputs)))
        a = cap[-1][..., neuron_idx].float()
        if pole == PoleType.POSITIVE:
            a = torch.clamp(a, min=0.0)
        elif pole == PoleType.NEGATIVE:
            a = torch.clamp(a, max=0.0)
        abl = InterventionSpec(layer_name, [neuron_idx], InterventionMode.ZERO_ABLATE, pole)
        with self.injector.temporary_interventions([abl]):
            p_total = _prob(extract_logits(run(self.model, inputs)))

        final = LayerDiscovery.resolve_final_norm(self.model)
        target_mod = final if final is not None else (self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None)
        if target_mod is None:
            raise ValueError("Cannot locate the final norm or unembedding to remove the direct path.")
        w = W[neuron_idx].to(cap[-1].device)

        def _remove_direct(_m, inp):
            x = inp[0]
            contrib = a.to(x.dtype).unsqueeze(-1) * w.to(x.dtype)
            return (x - contrib,) + tuple(inp[1:])

        h = target_mod.register_forward_pre_hook(_remove_direct)
        try:
            with self.injector.clean_forward():
                p_direct = _prob(extract_logits(run(self.model, inputs)))
        finally:
            h.remove()

        out = {
            "clean_target_prob": p_clean,
            "total_ablated_target_prob": p_total,
            "direct_path_removed_target_prob": p_direct,
            "total_effect": p_clean - p_total,
            "direct_effect": p_clean - p_direct,
        }
        if rescue_clean:
            idx = torch.as_tensor(rescue_downstream_neurons, dtype=torch.long)
            ref = rescue_clean[-1]
            hs2 = HookSet(self.model)

            def _patch(act, _g):
                act = act.clone()
                ii = idx.to(act.device)
                act[..., ii] = ref[..., ii].to(act.dtype)
                return act

            hs2.add_modifier(downstream_layer, _patch)
            with self.injector.temporary_interventions([abl]), hs2.active():
                p_rescued = _prob(extract_logits(run(self.model, inputs)))
            out["rescued_target_prob"] = p_rescued
            denom = p_clean - p_total
            out["rescued_fraction_of_total_effect"] = (p_rescued - p_total) / denom if denom != 0 else float("nan")
        return out

    # ------------------------------------------------------------------
    # 8. Multi-hop
    # ------------------------------------------------------------------
    def trace_multi_hop(self, layer_sequence: List[str], root_neuron: int, root_pole: PoleType, inputs: Any,
                        forward_fn: Optional[Callable] = None, beam_width: int = 3) -> List[CircuitGraph]:
        if len(layer_sequence) < 2:
            raise ValueError("`layer_sequence` must contain at least 2 layers.")
        graphs: List[CircuitGraph] = []
        front: List[Tuple[int, PoleType]] = [(root_neuron, root_pole)]
        for hop in range(len(layer_sequence) - 1):
            nxt: List[Tuple[int, PoleType]] = []
            for n, p in front:
                try:
                    g = self.trace_circuit(layer_sequence[hop], layer_sequence[hop + 1], n, inputs, forward_fn, p,
                                           compute_jacobian=False, top_k=beam_width)
                except ValueError:
                    continue
                graphs.append(g)
                nxt.extend((e.dst_neuron, e.dst_pole_tendency) for e in g.edges[:beam_width])
            front = list(dict.fromkeys(nxt))[:beam_width]
        return graphs
