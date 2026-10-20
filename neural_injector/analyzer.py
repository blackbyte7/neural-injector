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
    ) -> None:
        """
        Updates streaming neuron statistics from a batch of activations.
        activations shape: [B, S, D] or [B, D] or [B, C, H, W]
        """
        orig_shape = activations.shape
        if activations.ndim == 4:
            _, c_dim, h_dim, w_dim = orig_shape
            flat_acts = activations.permute(0, 2, 3, 1).reshape(-1, c_dim).to(torch.float64)
            seq_len = h_dim * w_dim
        elif activations.ndim == 3:
            _, seq_len, d_dim = orig_shape
            flat_acts = activations.reshape(-1, d_dim).to(torch.float64)
        else:
            seq_len = 1
            flat_acts = activations.to(torch.float64)

        flat_acts = flat_acts.to(self.device)
        batch_n = flat_acts.shape[0]
        if batch_n == 0:
            return

        # Batch Welford update
        new_count = self.count + batch_n
        batch_mean = flat_acts.mean(dim=0)
        delta = batch_mean - self.mean

        self.mean += delta * (batch_n / new_count)
        batch_m2 = ((flat_acts - batch_mean) ** 2).sum(dim=0)
        self.m2 += batch_m2 + (delta ** 2) * (self.count * batch_n / new_count)

        batch_m4 = ((flat_acts - self.mean) ** 4).sum(dim=0)
        self.m4 += batch_m4

        self.max_val = torch.maximum(self.max_val, flat_acts.max(dim=0).values)
        self.min_val = torch.minimum(self.min_val, flat_acts.min(dim=0).values)
        self.pos_count += (flat_acts > 0).sum(dim=0)
        self.neg_count += (flat_acts < 0).sum(dim=0)
        self.count = new_count

        # Update top-k positive & negative pole heaps for tracked neurons
        target_indices = (
            tracked_neurons if tracked_neurons is not None else list(range(min(self.num_neurons, 64)))
        )
        acts_cpu = flat_acts[:, target_indices].cpu()

        for local_i, neuron_idx in enumerate(target_indices):
            col = acts_cpu[:, local_i]

            # Top positive activations in this batch
            k_curr = min(self.top_k, batch_n)
            top_pos_vals, top_pos_flat_idx = torch.topk(col, k=k_curr, largest=True)
            top_neg_vals, top_neg_flat_idx = torch.topk(col, k=k_curr, largest=False)

            for val, f_idx in zip(top_pos_vals.tolist(), top_pos_flat_idx.tolist()):
                if val <= 0:
                    continue
                sample = self._build_sample(
                    val, f_idx, seq_len, sample_ids, tokens_per_sample, contexts_per_sample
                )
                self._push_heap(self.pos_heaps[neuron_idx], val, sample)

            for val, f_idx in zip(top_neg_vals.tolist(), top_neg_flat_idx.tolist()):
                if val >= 0:
                    continue
                sample = self._build_sample(
                    val, f_idx, seq_len, sample_ids, tokens_per_sample, contexts_per_sample
                )
                self._push_heap(self.neg_heaps[neuron_idx], -val, sample)

    @staticmethod
    def _build_sample(
        val: float,
        flat_idx: int,
        seq_len: int,
        sample_ids: Optional[List[int]],
        tokens_per_sample: Optional[List[List[str]]],
        contexts_per_sample: Optional[List[str]],
    ) -> ActivationSample:
        b_idx = flat_idx // seq_len
        pos_idx = flat_idx % seq_len
        s_id = sample_ids[b_idx] if sample_ids and b_idx < len(sample_ids) else b_idx
        tok_str = None
        if tokens_per_sample and b_idx < len(tokens_per_sample):
            tok_list = tokens_per_sample[b_idx]
            if pos_idx < len(tok_list):
                tok_str = tok_list[pos_idx]
        ctx_str = (
            contexts_per_sample[b_idx]
            if contexts_per_sample and b_idx < len(contexts_per_sample)
            else None
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

        return NeuronPoleProfile(
            layer_name=self.layer_name,
            neuron_idx=neuron_idx,
            mean=mean_val,
            std=std_val,
            max_val=float(self.max_val[neuron_idx].item()),
            min_val=float(self.min_val[neuron_idx].item()),
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
            if down_proj_module.__class__.__name__ == "Conv1D":
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
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Returns (top_promoted_tokens, top_suppressed_tokens) for `neuron_idx`.
        - `down_proj_module`: nn.Linear (shape [d_model, d_mlp]) or GPT2 Conv1D (shape [d_mlp, d_model])
        - `unembed_weight`: shape [vocab_size, d_model]
        """
        weight = down_proj_module.weight.detach()
        if down_proj_module.__class__.__name__ == "Conv1D":
            # [d_mlp, d_model]
            neuron_out_vec = weight[neuron_idx, :]
        else:
            # Standard nn.Linear: [d_model, d_mlp]
            neuron_out_vec = weight[:, neuron_idx]

        # Project onto vocabulary space: [vocab_size]
        unembed = unembed_weight.detach().to(neuron_out_vec.device, dtype=neuron_out_vec.dtype)
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
