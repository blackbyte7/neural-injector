from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .types import (
    ActivationSample,
    NeuronPoleProfile,
    NeuronScanSummary,
    NeuronTaxonomy,
)


class StreamingLayerStats:
    """
    GPU-accelerated streaming statistics tracker for an entire layer of neurons.
    Tracks mean, variance (Welford), min, max, pole firing rates, tail fourth-moment
    (kurtosis/selectivity), and top-k activating contexts for both positive and negative poles.
    """

    def __init__(self, layer_name: str, num_neurons: int, top_k: int = 10, device: str = "cpu"):
        self.layer_name = layer_name
        self.num_neurons = num_neurons
        self.top_k = top_k
        self.device = device

        self.count = 0
        self.mean = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.m3 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.m4 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.max_val = torch.full((num_neurons,), float("-inf"), dtype=torch.float64, device=device)
        self.min_val = torch.full((num_neurons,), float("inf"), dtype=torch.float64, device=device)
        self.pos_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.neg_count = torch.zeros(num_neurons, dtype=torch.long, device=device)

        # Per-neuron bounded min-heaps for top positive and negative pole activations
        # Stores tuples of (score, tie_breaker, ActivationSample)
        self._tie_breaker = 0
        self.pos_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = {
            i: [] for i in range(num_neurons)
        }
        self.neg_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = {
            i: [] for i in range(num_neurons)
        }

    @torch.no_grad()
    def update(
        self,
        activations: torch.Tensor,
        sample_ids: Optional[List[int]] = None,
        tokens_per_sample: Optional[List[List[str]]] = None,
        contexts_per_sample: Optional[List[str]] = None,
        tracked_neurons: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        routed_token_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Updates streaming neuron statistics from a batch of activations.
        activations shape: [B, S, D] or [B, D] or [B, C, H, W] or [N_routed, D] (MoE)
        """
        orig_shape = activations.shape
        spatial_hw: Optional[Tuple[int, int]] = None
        if activations.ndim == 4:
            _, c_dim, h_dim, w_dim = orig_shape
            flat_acts = activations.permute(0, 2, 3, 1).reshape(-1, c_dim).to(torch.float64)
            seq_len = h_dim * w_dim
            spatial_hw = (h_dim, w_dim)
        elif activations.ndim == 3:
            _, seq_len, d_dim = orig_shape
            flat_acts = activations.reshape(-1, d_dim).to(torch.float64)
        else:
            # 2D [B, D] or MoE routed [N_routed, D]: infer seq_len from tokens_per_sample if available
            if tokens_per_sample and len(tokens_per_sample) > 0 and len(tokens_per_sample[0]) > 1:
                seq_len = len(tokens_per_sample[0])
            else:
                seq_len = 1
            flat_acts = activations.to(torch.float64)

        flat_acts = flat_acts.to(self.device)

        # Map each row back to its original flattened (batch * seq_len + pos) index
        valid_flat_indices: Optional[torch.Tensor] = None
        if routed_token_indices is not None:
            valid_flat_indices = routed_token_indices.reshape(-1).to(
                device=self.device, dtype=torch.long
            )
        elif attention_mask is not None:
            mask_flat = attention_mask.reshape(-1).to(device=self.device, dtype=torch.bool)
            if mask_flat.numel() == flat_acts.shape[0]:
                valid_flat_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)
                if valid_flat_indices.ndim == 0:
                    valid_flat_indices = valid_flat_indices.unsqueeze(0)
                flat_acts = flat_acts[valid_flat_indices]

        batch_n = flat_acts.shape[0]
        if batch_n == 0:
            return

        # Batch Welford / Pébay parallel update for exact M1, M2, M3, M4 moments
        n_a = self.count
        n_b = batch_n
        new_count = n_a + n_b
        batch_mean = flat_acts.mean(dim=0)
        centered_b = flat_acts - batch_mean
        batch_m2 = (centered_b ** 2).sum(dim=0)
        batch_m3 = (centered_b ** 3).sum(dim=0)
        batch_m4 = (centered_b ** 4).sum(dim=0)

        if n_a == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.m3 = batch_m3
            self.m4 = batch_m4
        else:
            delta = batch_mean - self.mean
            delta_n = delta / new_count
            delta_n2 = delta_n * delta_n
            prod_n = n_a * n_b

            self.m4 += (
                batch_m4
                + prod_n * (n_a * n_a - prod_n + n_b * n_b) * delta * delta_n * delta_n2
                + 6.0 * (n_a * n_a * batch_m2 + n_b * n_b * self.m2) * delta_n2
                + 4.0 * (n_a * batch_m3 - n_b * self.m3) * delta_n
            )
            self.m3 += (
                batch_m3
                + prod_n * (n_a - n_b) * delta * delta_n2
                + 3.0 * (n_a * batch_m2 - n_b * self.m2) * delta_n
            )
            self.m2 += batch_m2 + (delta ** 2) * (n_a * n_b / new_count)
            self.mean += delta * (n_b / new_count)

        self.max_val = torch.maximum(self.max_val, flat_acts.max(dim=0).values)
        self.min_val = torch.minimum(self.min_val, flat_acts.min(dim=0).values)
        self.pos_count += (flat_acts > 0).sum(dim=0)
        self.neg_count += (flat_acts < 0).sum(dim=0)
        self.count = new_count

        # Update top-k positive & negative pole heaps for tracked neurons (all neurons by default)
        target_indices = (
            tracked_neurons if tracked_neurons is not None else list(range(self.num_neurons))
        )
        if not target_indices:
            return

        sub_acts = flat_acts[:, target_indices]
        k_curr = min(self.top_k, batch_n)

        # Vectorized GPU top-k across all tracked neurons simultaneously
        top_pos_vals, top_pos_flat_idx = torch.topk(sub_acts, k=k_curr, dim=0, largest=True)
        top_neg_vals, top_neg_flat_idx = torch.topk(sub_acts, k=k_curr, dim=0, largest=False)

        top_pos_vals_cpu = top_pos_vals.T.cpu().tolist()
        top_pos_idx_cpu = top_pos_flat_idx.T.cpu().tolist()
        top_neg_vals_cpu = top_neg_vals.T.cpu().tolist()
        top_neg_idx_cpu = top_neg_flat_idx.T.cpu().tolist()
        valid_idx_cpu = valid_flat_indices.cpu().tolist() if valid_flat_indices is not None else None

        for local_i, neuron_idx in enumerate(target_indices):
            pos_heap = self.pos_heaps[neuron_idx]
            neg_heap = self.neg_heaps[neuron_idx]

            # Top positive activations in this batch
            for val, r_idx in zip(top_pos_vals_cpu[local_i], top_pos_idx_cpu[local_i]):
                if val <= 0:
                    continue
                if len(pos_heap) >= self.top_k and val <= pos_heap[0][0]:
                    break
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                )
                self._push_heap(pos_heap, val, sample)

            for val, r_idx in zip(top_neg_vals_cpu[local_i], top_neg_idx_cpu[local_i]):
                if val >= 0:
                    continue
                if len(neg_heap) >= self.top_k and (-val) <= neg_heap[0][0]:
                    break
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                )
                self._push_heap(neg_heap, -val, sample)

    @staticmethod
    def _build_sample(
        val: float,
        flat_idx: int,
        seq_len: int,
        sample_ids: Optional[List[int]],
        tokens_per_sample: Optional[List[List[str]]],
        contexts_per_sample: Optional[List[str]],
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> ActivationSample:
        b_idx = flat_idx // seq_len
        pos_idx = flat_idx % seq_len
        s_id = sample_ids[b_idx] if sample_ids and b_idx < len(sample_ids) else b_idx
        tok_str = None
        if tokens_per_sample and b_idx < len(tokens_per_sample):
            tok_list = tokens_per_sample[b_idx]
            if pos_idx < len(tok_list):
                tok_str = tok_list[pos_idx]
        elif spatial_hw is not None:
            # Automatic Vision / Conv2d patch coordinate label when no text tokens are supplied
            h_dim, w_dim = spatial_hw
            row, col = pos_idx // w_dim, pos_idx % w_dim
            tok_str = f"patch_({row},{col})"

        ctx_str = (
            contexts_per_sample[b_idx]
            if contexts_per_sample and b_idx < len(contexts_per_sample)
            else (f"image_sample_{s_id}" if spatial_hw is not None else None)
        )
        return ActivationSample(
            activation_value=float(val),
            sample_id=s_id,
            position_idx=pos_idx,
            token_str=tok_str,
            context_str=ctx_str,
        )

    def _push_heap(
        self,
        heap: List[Tuple[float, int, ActivationSample]],
        priority: float,
        sample: ActivationSample,
    ) -> None:
        self._tie_breaker += 1
        item = (priority, self._tie_breaker, sample)
        if len(heap) < self.top_k:
            heapq.heappush(heap, item)
        elif priority > heap[0][0]:
            heapq.heapreplace(heap, item)

    def export_profile(self, neuron_idx: int) -> NeuronPoleProfile:
        """Compiles the final NeuronPoleProfile for a given neuron index."""
        n = max(self.count, 1)
        mean_val = float(self.mean[neuron_idx].item())
        var_val = float((self.m2[neuron_idx] / n).item())
        std_val = max(var_val ** 0.5, 1e-9)

        kurtosis = float((self.m4[neuron_idx] / n).item()) / ((var_val ** 2) + 1e-12)
        selectivity = float(min(max(kurtosis ** 0.25, 0.0), 100.0))

        pos_samples = [
            item[2]
            for item in sorted(self.pos_heaps.get(neuron_idx, []), key=lambda x: x[0], reverse=True)
        ]
        neg_samples = [
            item[2]
            for item in sorted(self.neg_heaps.get(neuron_idx, []), key=lambda x: x[0], reverse=True)
        ]

        max_v = float(self.max_val[neuron_idx].item())
        min_v = float(self.min_val[neuron_idx].item())

        return NeuronPoleProfile(
            layer_name=self.layer_name,
            neuron_idx=neuron_idx,
            mean=mean_val,
            std=std_val,
            max_val=max_v if max_v != float("-inf") else 0.0,
            min_val=min_v if min_v != float("inf") else 0.0,
            positive_firing_rate=float(self.pos_count[neuron_idx].item() / n),
            negative_firing_rate=float(self.neg_count[neuron_idx].item() / n),
            pole_selectivity_score=selectivity,
            top_positive_samples=pos_samples,
            top_negative_samples=neg_samples,
        )

    @torch.no_grad()
    def scan_layer_cleanliness(
        self,
        down_proj_module: Optional[nn.Module] = None,
    ) -> Tuple[Dict[str, int], Dict[str, float], List[NeuronScanSummary]]:
        """
        Vectorized GPU pass over all tracked moments in the layer to classify every neuron
        into the Cammarata (2026) cleanliness taxonomy and rank them by selectivity & impact.
        """
        n = max(self.count, 1)
        var_vec = self.m2 / n
        std_vec = torch.sqrt(torch.clamp(var_vec, min=1e-12))

        kurt_vec = (self.m4 / n) / (var_vec.pow(2) + 1e-12)
        selectivity_vec = torch.clamp(kurt_vec.pow(0.25), min=0.0, max=100.0)

        pos_rate_vec = self.pos_count.to(torch.float64) / n
        neg_rate_vec = self.neg_count.to(torch.float64) / n
        total_rate_vec = pos_rate_vec + neg_rate_vec

        bipolar_vec = (4.0 * pos_rate_vec * neg_rate_vec) / (total_rate_vec.pow(2) + 1e-9)

        weight_norms = torch.ones(self.num_neurons, dtype=torch.float64, device=self.device)
        if down_proj_module is not None and hasattr(down_proj_module, "weight"):
            w = down_proj_module.weight.detach().to(self.device, dtype=torch.float64)
            if w.ndim == 3 and w.shape[-1] == self.num_neurons:
                # 3D MoE down_proj: [num_experts, d_model, d_mlp] -> mean L2 norm across experts
                weight_norms = torch.norm(w, p=2, dim=1).mean(dim=0)
            elif down_proj_module.__class__.__name__ == "Conv1D":
                # [d_mlp, d_model]
                if w.shape[0] == self.num_neurons:
                    weight_norms = torch.norm(w, p=2, dim=1)
            else:
                # Standard Linear down_proj: [d_model, d_mlp]
                if w.shape[1] == self.num_neurons:
                    weight_norms = torch.norm(w, p=2, dim=0)

        impact_vec = std_vec * weight_norms

        sel_cpu = selectivity_vec.cpu().tolist()
        bip_cpu = bipolar_vec.cpu().tolist()
        pos_cpu = pos_rate_vec.cpu().tolist()
        neg_cpu = neg_rate_vec.cpu().tolist()
        mean_cpu = self.mean.cpu().tolist()
        std_cpu = std_vec.cpu().tolist()
        max_cpu = self.max_val.cpu().tolist()
        min_cpu = self.min_val.cpu().tolist()
        wnorm_cpu = weight_norms.cpu().tolist()
        imp_cpu = impact_vec.cpu().tolist()

        counts: Dict[str, int] = {
            NeuronTaxonomy.CLEAN_MONOSEMANTIC.value: 0,
            NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE.value: 0,
            NeuronTaxonomy.TRUE_POLYSEMANTIC.value: 0,
            NeuronTaxonomy.DEAD_OR_NOISY.value: 0,
        }

        summaries: List[NeuronScanSummary] = []

        for idx in range(self.num_neurons):
            sel = float(sel_cpu[idx])
            bip = float(bip_cpu[idx])
            std_v = float(std_cpu[idx])
            wnorm = float(wnorm_cpu[idx])
            tot_rate = float(pos_cpu[idx] + neg_cpu[idx])

            # Taxonomy classification rules aligned with Cammarata (2026)
            if std_v < 1e-5 or wnorm < 1e-5 or tot_rate < 1e-4:
                tax = NeuronTaxonomy.DEAD_OR_NOISY
            elif sel >= 2.1:
                tax = NeuronTaxonomy.CLEAN_MONOSEMANTIC
            elif sel >= 1.35 or (sel >= 1.15 and bip >= 0.45):
                tax = NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE
            else:
                tax = NeuronTaxonomy.TRUE_POLYSEMANTIC

            counts[tax.value] += 1

            pos_heap = self.pos_heaps.get(idx, [])
            neg_heap = self.neg_heaps.get(idx, [])
            top_pos_tok = max(pos_heap, key=lambda x: x[0])[2].token_str if pos_heap else None
            top_neg_tok = max(neg_heap, key=lambda x: x[0])[2].token_str if neg_heap else None

            summaries.append(
                NeuronScanSummary(
                    neuron_idx=idx,
                    taxonomy_class=tax,
                    pole_selectivity_score=round(sel, 3),
                    bipolar_balance=round(bip, 3),
                    positive_firing_rate=round(float(pos_cpu[idx]), 4),
                    negative_firing_rate=round(float(neg_cpu[idx]), 4),
                    mean=round(float(mean_cpu[idx]), 4),
                    std=round(std_v, 4),
                    max_val=round(float(max_cpu[idx]) if max_cpu[idx] != float("-inf") else 0.0, 4),
                    min_val=round(float(min_cpu[idx]) if min_cpu[idx] != float("inf") else 0.0, 4),
                    outgoing_weight_norm=round(wnorm, 4),
                    estimated_causal_impact=round(float(imp_cpu[idx]), 4),
                    top_pos_token=top_pos_tok,
                    top_neg_token=top_neg_tok,
                )
            )

        total_n = max(self.num_neurons, 1)
        percentages = {k: round((v / total_n) * 100.0, 2) for k, v in counts.items()}
        return counts, percentages, summaries


class LogitLensAttributor:
    """
    Computes Direct Weight-to-Vocabulary Attribution for any MLP neuron.
    By multiplying a neuron's outgoing column vector in `down_proj` with the model's
    unembedding matrix `lm_head`, we see which tokens the Positive and Negative poles
    directly push into the output distribution.
    """

    @staticmethod
    @torch.no_grad()
    def attribute_neuron_to_vocab(
        down_proj_module: nn.Module,
        unembed_weight: torch.Tensor,
        neuron_idx: int,
        idx_to_token: Optional[Dict[int, str]] = None,
        top_k: int = 8,
        final_norm_module: Optional[nn.Module] = None,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Returns (top_promoted_tokens, top_suppressed_tokens) for `neuron_idx`.
        - `down_proj_module`: nn.Linear (shape [d_model, d_mlp]), GPT2 Conv1D (shape [d_mlp, d_model]),
          or 3D MoE expert module (shape [num_experts, d_model, d_mlp]).
        - `unembed_weight`: shape [vocab_size, d_model]
        - `final_norm_module`: optional pre-unembedding LayerNorm / RMSNorm module
        """
        weight = down_proj_module.weight.detach()
        if weight.ndim == 3:
            # 3D MoE down_proj: [num_experts, d_model, d_mlp] -> average write direction across experts
            neuron_out_vec = weight[:, :, neuron_idx].mean(dim=0).to(dtype=torch.float32)
        elif down_proj_module.__class__.__name__ == "Conv1D":
            # [d_mlp, d_model]
            neuron_out_vec = weight[neuron_idx, :].to(dtype=torch.float32)
        else:
            # Standard nn.Linear: [d_model, d_mlp]
            neuron_out_vec = weight[:, neuron_idx].to(dtype=torch.float32)

        # Optionally apply final RMSNorm / LayerNorm scaling vector gamma before unembedding
        if final_norm_module is not None and hasattr(final_norm_module, "weight"):
            norm_w = final_norm_module.weight.detach().to(
                device=neuron_out_vec.device, dtype=torch.float32
            )
            if norm_w.shape == neuron_out_vec.shape:
                neuron_out_vec = neuron_out_vec * norm_w

        # Project onto vocabulary space: [vocab_size]
        unembed = unembed_weight.detach().to(neuron_out_vec.device, dtype=torch.float32)
        if unembed.shape[1] != neuron_out_vec.shape[0] and unembed.shape[0] == neuron_out_vec.shape[0]:
            unembed = unembed.T

        logit_contributions = torch.mv(unembed, neuron_out_vec)
        k = min(top_k, logit_contributions.numel())

        top_pos_vals, top_pos_ids = torch.topk(logit_contributions, k=k, largest=True)
        top_neg_vals, top_neg_ids = torch.topk(logit_contributions, k=k, largest=False)

        def _fmt(ids: torch.Tensor, vals: torch.Tensor) -> List[Tuple[str, float]]:
            results = []
            for tok_id, score in zip(ids.tolist(), vals.tolist()):
                label = (
                    idx_to_token.get(tok_id, f"token_{tok_id}")
                    if idx_to_token
                    else f"class_{tok_id}"
                )
                results.append((label, round(float(score), 4)))
            return results

        return _fmt(top_pos_ids, top_pos_vals), _fmt(top_neg_ids, top_neg_vals)
