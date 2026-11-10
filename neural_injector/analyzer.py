from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import math
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from .discovery import LayerDiscovery
from .types import (
    ActivationSample,
    BipolarVerbHypothesis,
    ContrastiveDiscoveryReport,
    ContrastiveNeuronMatch,
    NeuronCheckpointSnapshot,
    NeuronEvolutionReport,
    NeuronPoleProfile,
    NeuronScanSummary,
    NeuronTaxonomy,
    PoleType,
    SparsityTailPrior,
)


class StreamingLayerStats:
    """
    GPU-accelerated streaming statistics tracker for an entire layer of neurons.
    Tracks:
      - Full-distribution Welford/Pébay moments (M1, M2, M3, M4), min, max, and firing rates
      - Pole-specific conditional moments (M1+, M2+, M3+, M4+ for a > 0 and M1-, M2-, M3-, M4- for a < 0)
        so Positive and Negative poles are evaluated independently (Cammarata: 'Thinking in Neurons')
      - Full 4-quadrant SwiGLU sign regime counters (`pos_gate_pos_up`, `neg_gate_neg_up`,
        `pos_gate_neg_up`, `neg_gate_pos_up`) plus SwiGLU Double-Negative Route activations
        (`gate < 0` and `up < 0` yielding `silu(gate) * up > 0`) and per-quantile-band double-negative shares
      - Top-k activating contexts AND Stratified Quantile Activation Buckets per pole
        ("peak_trunk" [80%..100%], "secondary_branch" [40%..80%), "tertiary_offshoot" [15%..40%))
        sampled across the full dynamic range of each batch
      - Unbiased Algorithm R active-magnitude reservoirs per neuron for accurate empirical percentiles.
    """

    QUANTILE_BAND_SPECS: Tuple[Tuple[str, float, float], ...] = (
        ("peak_trunk", 0.80, 1.01),
        ("secondary_branch", 0.40, 0.80),
        ("tertiary_offshoot", 0.15, 0.40),
    )

    _PUNCTUATION_CHARS = frozenset(".,;:!?()[]{}-/\"'`~@#$%^&*_+=<>|\\")

    def __init__(self, layer_name: str, num_neurons: int, top_k: int = 10, device: str = "cpu"):
        self.layer_name = layer_name
        self.num_neurons = num_neurons
        self.top_k = top_k
        self.device = device

        # Full-distribution moments
        self.count = 0
        self.mean = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.m3 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.m4 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.max_val = torch.full((num_neurons,), float("-inf"), dtype=torch.float64, device=device)
        self.min_val = torch.full((num_neurons,), float("inf"), dtype=torch.float64, device=device)

        # Conditional Positive Pole (a > 0) moments
        self.pos_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.pos_mean = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.pos_m2 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.pos_m3 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.pos_m4 = torch.zeros(num_neurons, dtype=torch.float64, device=device)

        # Conditional Negative Pole (|a| where a < 0) moments
        self.neg_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.neg_mean = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.neg_m2 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.neg_m3 = torch.zeros(num_neurons, dtype=torch.float64, device=device)
        self.neg_m4 = torch.zeros(num_neurons, dtype=torch.float64, device=device)

        # SwiGLU Double-Negative route counter (a > 0 where gate < 0 and up < 0)
        # and full 4-quadrant sign regime counters
        self.double_neg_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.q_pp_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.q_nn_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.q_pn_count = torch.zeros(num_neurons, dtype=torch.long, device=device)
        self.q_np_count = torch.zeros(num_neurons, dtype=torch.long, device=device)

        # Per-neuron bounded min-heaps for top positive, negative, and SwiGLU double-negative activations
        self._tie_breaker = 0
        self.pos_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = defaultdict(list)
        self.neg_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = defaultdict(list)
        self.double_neg_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = defaultdict(list)

        # Stratified active sample reservoirs per neuron for quantile-band bucketing
        self._reservoir_cap = max(top_k * 6, 36)
        self.pos_band_reservoirs: Dict[int, List[Tuple[float, int, ActivationSample]]] = defaultdict(list)
        self.neg_band_reservoirs: Dict[int, List[Tuple[float, int, ActivationSample]]] = defaultdict(list)

        # Unbiased Algorithm R active magnitude reservoirs for accurate empirical percentiles
        self._mag_reservoir_cap = 512
        self._rng = random.Random(42)
        self.pos_mag_reservoirs: Dict[int, List[float]] = defaultdict(list)
        self.neg_mag_reservoirs: Dict[int, List[float]] = defaultdict(list)
        self._pos_mag_seen: Dict[int, int] = defaultdict(int)
        self._neg_mag_seen: Dict[int, int] = defaultdict(int)

    @staticmethod
    def _update_conditional_moments(
        mask: torch.Tensor,
        magnitudes: torch.Tensor,
        count_vec: torch.Tensor,
        mean_vec: torch.Tensor,
        m2_vec: torch.Tensor,
        m3_vec: torch.Tensor,
        m4_vec: torch.Tensor,
    ) -> None:
        """
        Vectorized GPU Pébay parallel moment update across all neurons where `mask` is True.
        Keeps 2D `[batch_n, D]` intermediates in `magnitudes.dtype` (`float32`) and accumulates
        reduced 1D `[D]` moment sums in `float64` for both high speed and numerical precision.
        Updates `count_vec`, `mean_vec`, `m2_vec`, `m3_vec`, `m4_vec` in-place.
        """
        n_b_long = mask.sum(dim=0)
        active_neurons = n_b_long > 0
        if not active_neurons.any():
            return

        n_a = count_vec.to(torch.float64)
        n_b = n_b_long.to(torch.float64)
        new_count = n_a + n_b
        safe_nb = torch.clamp(n_b, min=1.0)
        safe_new = torch.clamp(new_count, min=1.0)

        zero_2d = torch.zeros(1, dtype=magnitudes.dtype, device=magnitudes.device)
        masked_vals = torch.where(mask, magnitudes, zero_2d)
        batch_mean = masked_vals.sum(dim=0, dtype=torch.float64) / safe_nb

        centered_b = torch.where(
            mask,
            magnitudes - batch_mean.to(magnitudes.dtype).unsqueeze(0),
            zero_2d,
        )
        c2 = centered_b * centered_b
        batch_m2 = c2.sum(dim=0, dtype=torch.float64)
        batch_m3 = (c2 * centered_b).sum(dim=0, dtype=torch.float64)
        batch_m4 = (c2 * c2).sum(dim=0, dtype=torch.float64)

        delta = batch_mean - mean_vec
        delta_n = delta / safe_new
        delta_n2 = delta_n * delta_n
        prod_n = n_a * n_b

        new_m4 = (
            m4_vec
            + batch_m4
            + prod_n * (n_a * n_a - prod_n + n_b * n_b) * delta * delta_n * delta_n2
            + 6.0 * (n_a * n_a * batch_m2 + n_b * n_b * m2_vec) * delta_n2
            + 4.0 * (n_a * batch_m3 - n_b * m3_vec) * delta_n
        )
        new_m3 = (
            m3_vec
            + batch_m3
            + prod_n * (n_a - n_b) * delta * delta_n2
            + 3.0 * (n_a * batch_m2 - n_b * m2_vec) * delta_n
        )
        new_m2 = m2_vec + batch_m2 + (delta ** 2) * (prod_n / safe_new)
        new_mean = mean_vec + delta * (n_b / safe_new)

        first_seen = active_neurons & (count_vec == 0)
        continuing = active_neurons & (count_vec > 0)

        mean_vec[:] = torch.where(first_seen, batch_mean, torch.where(continuing, new_mean, mean_vec))
        m2_vec[:] = torch.where(first_seen, batch_m2, torch.where(continuing, new_m2, m2_vec))
        m3_vec[:] = torch.where(first_seen, batch_m3, torch.where(continuing, new_m3, m3_vec))
        m4_vec[:] = torch.where(first_seen, batch_m4, torch.where(continuing, new_m4, m4_vec))
        count_vec.add_(n_b_long)

    def _record_unbiased_magnitude(
        self, mag: float, reservoir: List[float], seen_dict: Dict[int, int], neuron_idx: int
    ) -> None:
        """Algorithm R reservoir sampling over active firing magnitudes for unbiased quantiles."""
        seen = seen_dict[neuron_idx] + 1
        seen_dict[neuron_idx] = seen
        if len(reservoir) < self._mag_reservoir_cap:
            reservoir.append(mag)
        else:
            j = self._rng.randint(0, seen - 1)
            if j < self._mag_reservoir_cap:
                reservoir[j] = mag

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
        gate_activations: Optional[torch.Tensor] = None,
        up_activations: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Updates streaming neuron statistics from a batch of activations.
        activations shape: [B, S, D] or [B, D] or [B, C, H, W] or [N_routed, D] (MoE).
        When `gate_activations` and `up_activations` are provided (for SwiGLU blocks),
        automatically detects and separates the 4 sign quadrants including the
        'double-negative' route (`gate < 0` and `up < 0`).
        """
        orig_shape = activations.shape
        spatial_hw: Optional[Tuple[int, int]] = None
        work_dtype = torch.float64 if activations.dtype == torch.float64 else torch.float32
        if activations.ndim == 4:
            _, c_dim, h_dim, w_dim = orig_shape
            flat_acts = activations.permute(0, 2, 3, 1).reshape(-1, c_dim).to(
                device=self.device, dtype=work_dtype
            )
            seq_len = h_dim * w_dim
            spatial_hw = (h_dim, w_dim)
        elif activations.ndim == 3:
            _, seq_len, d_dim = orig_shape
            flat_acts = activations.reshape(-1, d_dim).to(device=self.device, dtype=work_dtype)
        else:
            # 2D [B, D] or MoE routed [N_routed, D]: infer seq_len from tokens_per_sample if available
            if tokens_per_sample and len(tokens_per_sample) > 0 and len(tokens_per_sample[0]) > 1:
                seq_len = len(tokens_per_sample[0])
            else:
                seq_len = 1
            flat_acts = activations.to(device=self.device, dtype=work_dtype)

        # Align optional SwiGLU gate and up projections to the same flattened shape
        flat_gate: Optional[torch.Tensor] = None
        flat_up: Optional[torch.Tensor] = None
        if gate_activations is not None and up_activations is not None:
            try:
                g_reshaped = gate_activations.reshape(-1, flat_acts.shape[-1]).to(self.device)
                u_reshaped = up_activations.reshape(-1, flat_acts.shape[-1]).to(self.device)
                if g_reshaped.shape == flat_acts.shape and u_reshaped.shape == flat_acts.shape:
                    flat_gate = g_reshaped
                    flat_up = u_reshaped
            except RuntimeError:
                flat_gate = None
                flat_up = None

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
                if flat_gate is not None and flat_up is not None:
                    flat_gate = flat_gate[valid_flat_indices]
                    flat_up = flat_up[valid_flat_indices]

        batch_n = flat_acts.shape[0]
        if batch_n == 0:
            return

        # 1. Batch Welford / Pébay parallel update for exact full-distribution M1, M2, M3, M4 moments
        n_a = self.count
        n_b = batch_n
        new_count = n_a + n_b
        batch_mean = flat_acts.mean(dim=0, dtype=torch.float64)
        centered_b = flat_acts - batch_mean.to(flat_acts.dtype).unsqueeze(0)
        c2 = centered_b * centered_b
        batch_m2 = c2.sum(dim=0, dtype=torch.float64)
        batch_m3 = (c2 * centered_b).sum(dim=0, dtype=torch.float64)
        batch_m4 = (c2 * c2).sum(dim=0, dtype=torch.float64)

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

        self.max_val = torch.maximum(
            self.max_val, flat_acts.max(dim=0).values.to(torch.float64)
        )
        self.min_val = torch.minimum(
            self.min_val, flat_acts.min(dim=0).values.to(torch.float64)
        )
        self.count = new_count

        # 2. Update Pole-Specific Conditional Moments and SwiGLU 4-Quadrant Sign Regime Counts
        pos_mask = flat_acts > 0
        neg_mask = flat_acts < 0
        if flat_gate is not None and flat_up is not None:
            g_pos = flat_gate > 0
            g_neg = flat_gate < 0
            u_pos = flat_up > 0
            u_neg = flat_up < 0
            dbl_neg_mask = pos_mask & g_neg & u_neg
            self.double_neg_count.add_(dbl_neg_mask.sum(dim=0))
            self.q_pp_count.add_((pos_mask & g_pos & u_pos).sum(dim=0))
            self.q_nn_count.add_(dbl_neg_mask.sum(dim=0))
            self.q_pn_count.add_((neg_mask & g_pos & u_neg).sum(dim=0))
            self.q_np_count.add_((neg_mask & g_neg & u_pos).sum(dim=0))
        else:
            dbl_neg_mask = None

        self._update_conditional_moments(
            pos_mask,
            flat_acts,
            self.pos_count,
            self.pos_mean,
            self.pos_m2,
            self.pos_m3,
            self.pos_m4,
        )
        self._update_conditional_moments(
            neg_mask,
            -flat_acts,
            self.neg_count,
            self.neg_mean,
            self.neg_m2,
            self.neg_m3,
            self.neg_m4,
        )

        # 3. Update top-k heaps, stratified quantile reservoirs, and unbiased magnitude reservoirs
        target_indices = (
            tracked_neurons if tracked_neurons is not None else list(range(self.num_neurons))
        )
        if not target_indices:
            return

        num_targets = len(target_indices)
        if num_targets == self.num_neurons and target_indices == list(range(self.num_neurons)):
            sub_acts = flat_acts
            sub_dbl_neg = dbl_neg_mask
        else:
            sub_acts = flat_acts[:, target_indices]
            sub_dbl_neg = dbl_neg_mask[:, target_indices] if dbl_neg_mask is not None else None

        valid_idx_cpu = valid_flat_indices.cpu().tolist() if valid_flat_indices is not None else None
        curr_max_cpu = self.max_val[target_indices].cpu().tolist()
        curr_min_cpu = self.min_val[target_indices].cpu().tolist()

        # Fast-path for wide layers / large token batches: instead of sorting all [batch_n, D]
        # elements and materializing 4M+ Python floats, extract only top-k positive peaks,
        # top-k negative peaks, and a stratified slice of intermediate rows on GPU/C++.
        if num_targets * batch_n > 65536 and batch_n > self.top_k * 2:
            self._update_heaps_and_reservoirs_fast(
                sub_acts=sub_acts,
                sub_dbl_neg=sub_dbl_neg,
                target_indices=target_indices,
                valid_idx_cpu=valid_idx_cpu,
                curr_max_cpu=curr_max_cpu,
                curr_min_cpu=curr_min_cpu,
                seq_len=seq_len,
                sample_ids=sample_ids,
                tokens_per_sample=tokens_per_sample,
                contexts_per_sample=contexts_per_sample,
                spatial_hw=spatial_hw,
            )
            return

        # Full-resolution path for compact layers / batches
        sorted_vals, sorted_flat_idx = torch.sort(sub_acts, dim=0, descending=True)
        sorted_vals_cpu = sorted_vals.T.cpu().tolist()
        sorted_idx_cpu = sorted_flat_idx.T.cpu().tolist()
        dbl_neg_cpu = sub_dbl_neg.T.cpu().tolist() if sub_dbl_neg is not None else None

        for local_i, neuron_idx in enumerate(target_indices):
            pos_heap = self.pos_heaps[neuron_idx]
            neg_heap = self.neg_heaps[neuron_idx]
            dbl_heap = self.double_neg_heaps[neuron_idx]
            pos_res = self.pos_band_reservoirs[neuron_idx]
            neg_res = self.neg_band_reservoirs[neuron_idx]
            pos_mag_res = self.pos_mag_reservoirs[neuron_idx]
            neg_mag_res = self.neg_mag_reservoirs[neuron_idx]

            vals_row = sorted_vals_cpu[local_i]
            idx_row = sorted_idx_cpu[local_i]
            dbl_row = dbl_neg_cpu[local_i] if dbl_neg_cpu is not None else None
            running_pos_max = max(float(curr_max_cpu[local_i]), 1e-9)
            running_neg_max = max(abs(min(float(curr_min_cpu[local_i]), 0.0)), 1e-9)

            # Collect positive entries (vals_row is descending, so positive entries are at the front)
            pos_positions: List[int] = []
            for rank_i, val in enumerate(vals_row):
                if val <= 0.0:
                    break
                pos_positions.append(rank_i)
                self._record_unbiased_magnitude(
                    float(val), pos_mag_res, self._pos_mag_seen, neuron_idx
                )

            # Select candidate ranks covering top-k AND evenly spaced quantiles across all positive firings
            if len(pos_positions) <= self._reservoir_cap:
                selected_pos_ranks = pos_positions
            else:
                stride = max(
                    1,
                    (len(pos_positions) - self.top_k) // max(1, self._reservoir_cap - self.top_k),
                )
                selected_pos_ranks = list(range(self.top_k)) + pos_positions[self.top_k:: stride]

            for rank_i in selected_pos_ranks:
                val = float(vals_row[rank_i])
                r_idx = idx_row[rank_i]
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                is_dbl_neg = bool(dbl_row[r_idx]) if dbl_row is not None else False

                need_top = rank_i < self.top_k and (
                    len(pos_heap) < self.top_k or val > pos_heap[0][0]
                )
                need_dbl = is_dbl_neg and (
                    len(dbl_heap) < self.top_k or val > dbl_heap[0][0]
                )
                # Keep reservoir samples (including double-negative tagged samples so band rates can be measured)
                need_res = True

                if not (need_top or need_dbl or need_res):
                    continue

                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                    is_double_negative_route=is_dbl_neg,
                )
                if need_top:
                    self._push_heap(pos_heap, val, sample, self.top_k)
                if need_dbl:
                    self._push_heap(dbl_heap, val, sample, self.top_k)
                if need_res:
                    self._push_stratified_reservoir(pos_res, val, sample, running_pos_max)

            # Collect negative entries from the tail (ascending magnitude from the end of vals_row)
            neg_positions: List[int] = []
            for rev_i in range(len(vals_row) - 1, -1, -1):
                val = vals_row[rev_i]
                if val >= 0.0:
                    break
                neg_positions.append(rev_i)
                self._record_unbiased_magnitude(
                    float(-val), neg_mag_res, self._neg_mag_seen, neuron_idx
                )

            if len(neg_positions) <= self._reservoir_cap:
                selected_neg_ranks = neg_positions
            else:
                stride = max(
                    1,
                    (len(neg_positions) - self.top_k) // max(1, self._reservoir_cap - self.top_k),
                )
                selected_neg_ranks = (
                    neg_positions[: self.top_k] + neg_positions[self.top_k:: stride]
                )

            for rank_pos, rev_i in enumerate(selected_neg_ranks):
                val = float(vals_row[rev_i])
                mag = -val
                r_idx = idx_row[rev_i]
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx

                need_top = rank_pos < self.top_k and (
                    len(neg_heap) < self.top_k or mag > neg_heap[0][0]
                )
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                    is_double_negative_route=False,
                )
                if need_top:
                    self._push_heap(neg_heap, mag, sample, self.top_k)
                self._push_stratified_reservoir(neg_res, mag, sample, running_neg_max)

    def _update_heaps_and_reservoirs_fast(
        self,
        sub_acts: torch.Tensor,
        sub_dbl_neg: Optional[torch.Tensor],
        target_indices: List[int],
        valid_idx_cpu: Optional[List[int]],
        curr_max_cpu: List[float],
        curr_min_cpu: List[float],
        seq_len: int,
        sample_ids: Optional[List[int]],
        tokens_per_sample: Optional[List[List[str]]],
        contexts_per_sample: Optional[List[str]],
        spatial_hw: Optional[Tuple[int, int]],
    ) -> None:
        """
        High-throughput O(k * D) top-k and reservoir extraction for wide layers (e.g. d_mlp >= 4096).
        Uses `torch.topk` on device and a small strided batch subset instead of full-matrix sort.
        """
        batch_n = sub_acts.shape[0]
        k = min(self.top_k, batch_n)

        pos_top_vals, pos_top_idx = torch.topk(sub_acts, k=k, dim=0, largest=True)
        neg_top_vals, neg_top_idx = torch.topk(sub_acts, k=k, dim=0, largest=False)

        # Sample up to 6 evenly spaced rows across the batch for unbiased magnitude & band reservoirs
        n_stride = min(6, batch_n)
        stride_step = max(1, batch_n // n_stride)
        sample_row_ids = list(range(0, batch_n, stride_step))[:n_stride]
        stride_vals = sub_acts[sample_row_ids, :]

        pos_vals_cpu = pos_top_vals.T.cpu().tolist()
        pos_idx_cpu = pos_top_idx.T.cpu().tolist()
        neg_vals_cpu = neg_top_vals.T.cpu().tolist()
        neg_idx_cpu = neg_top_idx.T.cpu().tolist()
        stride_vals_cpu = stride_vals.T.cpu().tolist()

        if sub_dbl_neg is not None:
            pos_dbl_cpu = torch.gather(sub_dbl_neg, 0, pos_top_idx).T.cpu().tolist()
            stride_dbl_cpu = sub_dbl_neg[sample_row_ids, :].T.cpu().tolist()
        else:
            pos_dbl_cpu = None
            stride_dbl_cpu = None

        for local_i, neuron_idx in enumerate(target_indices):
            pos_heap = self.pos_heaps[neuron_idx]
            neg_heap = self.neg_heaps[neuron_idx]
            dbl_heap = self.double_neg_heaps[neuron_idx]
            pos_res = self.pos_band_reservoirs[neuron_idx]
            neg_res = self.neg_band_reservoirs[neuron_idx]
            pos_mag_res = self.pos_mag_reservoirs[neuron_idx]
            neg_mag_res = self.neg_mag_reservoirs[neuron_idx]

            running_pos_max = max(float(curr_max_cpu[local_i]), 1e-9)
            running_neg_max = max(abs(min(float(curr_min_cpu[local_i]), 0.0)), 1e-9)

            p_vals = pos_vals_cpu[local_i]
            p_idxs = pos_idx_cpu[local_i]
            p_dbls = pos_dbl_cpu[local_i] if pos_dbl_cpu is not None else None
            p_min_heap = pos_heap[0][0] if len(pos_heap) >= self.top_k else float("-inf")

            for j in range(k):
                val = float(p_vals[j])
                if val <= 0.0:
                    break
                r_idx = p_idxs[j]
                is_dbl = bool(p_dbls[j]) if p_dbls is not None else False
                need_top = len(pos_heap) < self.top_k or val > p_min_heap
                need_dbl = is_dbl and (len(dbl_heap) < self.top_k or val > dbl_heap[0][0])
                need_res = len(pos_res) < self._reservoir_cap
                if not (need_top or need_dbl or need_res):
                    continue
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                    is_double_negative_route=is_dbl,
                )
                if need_top:
                    self._push_heap(pos_heap, val, sample, self.top_k)
                    p_min_heap = pos_heap[0][0] if len(pos_heap) >= self.top_k else float("-inf")
                if need_dbl:
                    self._push_heap(dbl_heap, val, sample, self.top_k)
                if need_res:
                    self._push_stratified_reservoir(pos_res, val, sample, running_pos_max)

            n_vals = neg_vals_cpu[local_i]
            n_idxs = neg_idx_cpu[local_i]
            n_min_heap = neg_heap[0][0] if len(neg_heap) >= self.top_k else float("-inf")

            for j in range(k):
                val = float(n_vals[j])
                if val >= 0.0:
                    break
                mag = -val
                r_idx = n_idxs[j]
                need_top = len(neg_heap) < self.top_k or mag > n_min_heap
                need_res = len(neg_res) < self._reservoir_cap
                if not (need_top or need_res):
                    continue
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                    is_double_negative_route=False,
                )
                if need_top:
                    self._push_heap(neg_heap, mag, sample, self.top_k)
                    n_min_heap = neg_heap[0][0] if len(neg_heap) >= self.top_k else float("-inf")
                if need_res:
                    self._push_stratified_reservoir(neg_res, mag, sample, running_neg_max)

            # Update unbiased magnitude reservoirs & intermediate bands from strided batch rows
            s_vals = stride_vals_cpu[local_i]
            s_dbls = stride_dbl_cpu[local_i] if stride_dbl_cpu is not None else None
            for s_pos, r_idx in enumerate(sample_row_ids):
                val = float(s_vals[s_pos])
                if val > 0.0:
                    self._record_unbiased_magnitude(
                        val, pos_mag_res, self._pos_mag_seen, neuron_idx
                    )
                    if 0.15 * running_pos_max <= val < 0.80 * running_pos_max and (
                        len(pos_res) < self._reservoir_cap or s_pos == 0
                    ):
                        f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                        is_dbl = bool(s_dbls[s_pos]) if s_dbls is not None else False
                        sample = self._build_sample(
                            val,
                            f_idx,
                            seq_len,
                            sample_ids,
                            tokens_per_sample,
                            contexts_per_sample,
                            spatial_hw=spatial_hw,
                            is_double_negative_route=is_dbl,
                        )
                        self._push_stratified_reservoir(pos_res, val, sample, running_pos_max)
                elif val < 0.0:
                    mag = -val
                    self._record_unbiased_magnitude(
                        mag, neg_mag_res, self._neg_mag_seen, neuron_idx
                    )
                    if 0.15 * running_neg_max <= mag < 0.80 * running_neg_max and (
                        len(neg_res) < self._reservoir_cap or s_pos == 0
                    ):
                        f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                        sample = self._build_sample(
                            val,
                            f_idx,
                            seq_len,
                            sample_ids,
                            tokens_per_sample,
                            contexts_per_sample,
                            spatial_hw=spatial_hw,
                            is_double_negative_route=False,
                        )
                        self._push_stratified_reservoir(neg_res, mag, sample, running_neg_max)

    @staticmethod
    def _build_sample(
        val: float,
        flat_idx: int,
        seq_len: int,
        sample_ids: Optional[List[int]],
        tokens_per_sample: Optional[List[List[str]]],
        contexts_per_sample: Optional[List[str]],
        spatial_hw: Optional[Tuple[int, int]] = None,
        is_double_negative_route: bool = False,
    ) -> ActivationSample:
        b_idx = flat_idx // seq_len
        pos_idx = flat_idx % seq_len
        s_id = sample_ids[b_idx] if sample_ids and b_idx < len(sample_ids) else b_idx
        tok_str = None
        next_tok_str = None
        window_str = None

        if tokens_per_sample and b_idx < len(tokens_per_sample):
            tok_list = tokens_per_sample[b_idx]
            if pos_idx < len(tok_list):
                tok_str = tok_list[pos_idx]
                next_tok_str = tok_list[pos_idx + 1] if (pos_idx + 1) < len(tok_list) else "<eos>"
                start_w = max(0, pos_idx - 2)
                end_w = min(len(tok_list), pos_idx + 3)
                prefix = " ".join(tok_list[start_w:pos_idx])
                suffix = " ".join(tok_list[pos_idx + 1: end_w])
                focal = f"[{tok_str}] -> {next_tok_str}"
                parts = [p for p in (prefix, focal, suffix) if p]
                window_str = " ".join(parts)
        elif spatial_hw is not None:
            # Automatic Vision / Conv2d patch coordinate label when no text tokens are supplied
            h_dim, w_dim = spatial_hw
            row, col = pos_idx // w_dim, pos_idx % w_dim
            tok_str = f"patch_({row},{col})"
            window_str = tok_str

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
            next_token_str=next_tok_str,
            window_str=window_str,
            is_double_negative_route=is_double_negative_route,
        )

    def _push_heap(
        self,
        heap: List[Tuple[float, int, ActivationSample]],
        priority: float,
        sample: ActivationSample,
        max_len: int,
    ) -> None:
        self._tie_breaker += 1
        item = (priority, self._tie_breaker, sample)
        if len(heap) < max_len:
            heapq.heappush(heap, item)
        elif priority > heap[0][0]:
            heapq.heapreplace(heap, item)

    def _push_stratified_reservoir(
        self,
        reservoir: List[Tuple[float, int, ActivationSample]],
        magnitude: float,
        sample: ActivationSample,
        running_pole_max: float,
    ) -> None:
        """
        Maintains a stratified pool of active samples across all quantile bands
        (`peak_trunk` >= 0.80, `secondary_branch` [0.40..0.80), `tertiary_offshoot` [0.15..0.40)).
        Unlike naive nearest-neighbor replacement, each band has a protected quota of
        `self._reservoir_cap // 3` slots, preventing new peak activations from collapsing
        onto a single slot or starving intermediate bands.
        """
        self._tie_breaker += 1
        item = (magnitude, self._tie_breaker, sample)
        if len(reservoir) < self._reservoir_cap:
            reservoir.append(item)
            return

        pole_max = max(running_pole_max, magnitude, max((x[0] for x in reservoir), default=1e-9))

        def _band_of(mag_val: float) -> int:
            r = mag_val / pole_max
            if r >= 0.80:
                return 0  # peak_trunk
            elif r >= 0.40:
                return 1  # secondary_branch
            elif r >= 0.15:
                return 2  # tertiary_offshoot
            return 3      # sub-tertiary (< 15% of max)

        target_band = _band_of(magnitude)
        if target_band == 3:
            return

        band_indices: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
        for idx, (m_val, _, _) in enumerate(reservoir):
            band_indices[_band_of(m_val)].append(idx)

        # 1. Always evict sub-tertiary (< 15% of current max) entries first
        if band_indices[3]:
            evict_idx = band_indices[3][0]
            reservoir[evict_idx] = item
            return

        # 2. If another band is over-represented (> cap // 3), evict from the largest band
        quota = max(1, self._reservoir_cap // 3)
        if len(band_indices[target_band]) < quota:
            overfull_band = max((0, 1, 2), key=lambda b: len(band_indices[b]))
            candidates = band_indices[overfull_band]
        else:
            candidates = band_indices[target_band]

        if candidates:
            evict_idx = self._rng.choice(candidates)
            reservoir[evict_idx] = item

    def _build_quantile_buckets(
        self,
        neuron_idx: int,
        is_positive: bool,
        pole_max_abs: float,
    ) -> Tuple[Dict[str, List[ActivationSample]], Dict[str, float]]:
        """
        Partitions collected samples for a pole into Cammarata's Stratified Quantile Bands:
          - `peak_trunk`:        [0.80 * max_abs, 1.00 * max_abs]
          - `secondary_branch`:  [0.40 * max_abs, 0.80 * max_abs)
          - `tertiary_offshoot`: [0.15 * max_abs, 0.40 * max_abs)
        Also computes per-band SwiGLU double-negative rates (`double_negative_band_rates`)
        before filtering double-negative (`gate < 0, up < 0`) samples out of true positive
        quantile buckets (unless no true positive `gate > 0, up > 0` samples exist in a band).
        """
        buckets: Dict[str, List[ActivationSample]] = {
            name: [] for name, _, _ in self.QUANTILE_BAND_SPECS
        }
        band_dbl_rates: Dict[str, float] = {
            name: 0.0 for name, _, _ in self.QUANTILE_BAND_SPECS
        }
        if pole_max_abs <= 1e-9:
            return buckets, band_dbl_rates

        heap_items = (
            self.pos_heaps.get(neuron_idx, []) + self.double_neg_heaps.get(neuron_idx, [])
            if is_positive
            else self.neg_heaps.get(neuron_idx, [])
        )
        res_items = (
            self.pos_band_reservoirs.get(neuron_idx, [])
            if is_positive
            else self.neg_band_reservoirs.get(neuron_idx, [])
        )

        # Deduplicate all candidate samples by (sample_id, position_idx, round(activation_value, 6))
        seen = set()
        all_candidates: List[ActivationSample] = []
        for mag, _, s in sorted(heap_items + res_items, key=lambda x: x[0], reverse=True):
            key = (s.sample_id, s.position_idx, round(s.activation_value, 6))
            if key not in seen and mag > 0:
                seen.add(key)
                all_candidates.append(s)

        has_true_positive = any(not s.is_double_negative_route for s in all_candidates)
        band_totals: Dict[str, int] = {name: 0 for name, _, _ in self.QUANTILE_BAND_SPECS}
        band_dbls: Dict[str, int] = {name: 0 for name, _, _ in self.QUANTILE_BAND_SPECS}

        for s in all_candidates:
            ratio = abs(s.activation_value) / pole_max_abs
            for band_name, low_r, high_r in self.QUANTILE_BAND_SPECS:
                if low_r <= ratio <= high_r:
                    band_totals[band_name] += 1
                    if s.is_double_negative_route:
                        band_dbls[band_name] += 1
                    if is_positive and s.is_double_negative_route and has_true_positive:
                        break
                    tagged = ActivationSample(
                        activation_value=s.activation_value,
                        sample_id=s.sample_id,
                        position_idx=s.position_idx,
                        token_str=s.token_str,
                        context_str=s.context_str,
                        quantile_band=band_name,
                        next_token_str=s.next_token_str,
                        window_str=s.window_str,
                        is_double_negative_route=s.is_double_negative_route,
                    )
                    if len(buckets[band_name]) < self.top_k:
                        buckets[band_name].append(tagged)
                    break

        for band_name, _, _ in self.QUANTILE_BAND_SPECS:
            tot = band_totals[band_name]
            band_dbl_rates[band_name] = round(band_dbls[band_name] / tot, 4) if tot > 0 else 0.0

        return buckets, band_dbl_rates

    @staticmethod
    def _classify_pole_prior(
        pole_selectivity: float,
        pole_firing_rate: float,
        pole_std: float,
        other_pole_rate: float = 0.0,
        outgoing_weight_norm: float = 1.0,
    ) -> SparsityTailPrior:
        """Assigns the statistical SparsityTailPrior for a single pole."""
        if pole_std < 1e-6 or outgoing_weight_norm < 1e-5 or pole_firing_rate < 1e-4:
            return SparsityTailPrior.DEAD_OR_FLAT
        if pole_firing_rate <= 0.20 and pole_selectivity >= 1.85:
            return SparsityTailPrior.SPARSE_DETECTOR
        if pole_firing_rate >= 0.05 and other_pole_rate >= 0.05 and pole_selectivity >= 1.35:
            return SparsityTailPrior.BIPOLAR_ROUTER_CANDIDATE
        if pole_selectivity >= 1.25:
            return SparsityTailPrior.GRADED_DIAL
        return SparsityTailPrior.DENSE_UNSELECTIVE

    @staticmethod
    def _classify_pole_taxonomy(
        pole_selectivity: float,
        pole_firing_rate: float,
        pole_std: float,
        outgoing_weight_norm: float = 1.0,
    ) -> NeuronTaxonomy:
        """Classifies an individual neuron pole (Positive or Negative) independently."""
        if pole_std < 1e-6 or outgoing_weight_norm < 1e-5 or pole_firing_rate < 1e-4:
            return NeuronTaxonomy.DEAD_OR_NOISY
        elif pole_selectivity >= 2.05:
            return NeuronTaxonomy.CLEAN_MONOSEMANTIC
        elif pole_selectivity >= 1.48:
            return NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE
        else:
            return NeuronTaxonomy.TRUE_POLYSEMANTIC

    def _compute_pole_selectivities_and_stds(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes conditional standard deviations and null-calibrated tail-selectivity scores
        for both the Positive Pole and Negative Pole across all neurons in the layer.

        Statistical Calibration against the Gaussian / SwiGLU Null:
          - For two independent standard normal projections G, U ~ N(0, 1), the SwiGLU
            bilinear product `a = SiLU(G) * U` fires positively ~50% of the time (`p_pole ~ 0.50`)
            with theoretical conditional positive kurtosis `kappa_+ ~ 9.0` (`9.0^0.25 = 1.732`).
          - Uncalibrated `(kurt^0.25) * (1 / p_pole)^0.25` gave `1.732 * 1.189 = 2.06 >= 1.95`
            on 100% random white-noise SwiGLU neurons!
          - Here:
            1. Sparsity boost is referenced to a genuine sparse firing rate `p_ref = 0.20`:
               `sparsity_factor = (0.20 / max(p_pole, 1e-4))^0.25`, which boosts sparse
               detectors (`p_pole < 0.20`) and penalizes dense coin-flip firing (`p_pole ~ 0.50`).
            2. For dense poles (`p_pole > 0.25`), the SwiGLU/Gaussian baseline kurtosis (`13.5`)
               is subtracted (`eff_kurt = clamp(kurt - 13.5, min=1.0)`), ensuring random-init
               SwiGLU layers score near `0.8 - 1.2` instead of masquerading as clean detectors.
        """
        n_total = float(max(self.count, 1))

        overall_std = torch.sqrt(torch.clamp(self.m2 / n_total, min=1e-12))

        pos_n = torch.clamp(self.pos_count.to(torch.float64), min=1.0)
        pos_rate = pos_n / n_total
        pos_var = self.pos_m2 / pos_n
        pos_std = torch.sqrt(torch.clamp(pos_var, min=1e-12))
        pos_kurt = (self.pos_m4 / pos_n) / (pos_var.pow(2) + 1e-12)
        pos_null_kurt = torch.where(pos_rate > 0.25, 22.0, 12.5)
        pos_eff_kurt = torch.clamp(pos_kurt - pos_null_kurt, min=1.0)
        pos_snr = torch.where(
            self.pos_count == 1,
            torch.full_like(self.pos_mean, 6.0),
            self.pos_mean.abs() / torch.clamp(pos_std, min=1e-6),
        )
        pos_focal_kurt = torch.where(
            (pos_rate <= 0.15) & (self.pos_mean.abs() >= overall_std),
            torch.clamp((pos_snr - 2.0) * 6.0, min=1.0, max=64.0),
            torch.ones_like(pos_snr),
        )
        pos_eff_kurt = torch.maximum(pos_eff_kurt, pos_focal_kurt)
        pos_raw_sparsity = (0.20 / torch.clamp(pos_rate, min=1e-4)).pow(0.25)
        pos_sparsity_factor = torch.where(
            pos_eff_kurt > 1.05,
            pos_raw_sparsity,
            torch.clamp(pos_raw_sparsity, max=1.0),
        )
        pos_sel = torch.where(
            self.pos_count > 0,
            torch.clamp(pos_eff_kurt.pow(0.25) * pos_sparsity_factor, min=0.0, max=100.0),
            torch.zeros_like(pos_var),
        )

        neg_n = torch.clamp(self.neg_count.to(torch.float64), min=1.0)
        neg_rate = neg_n / n_total
        neg_var = self.neg_m2 / neg_n
        neg_std = torch.sqrt(torch.clamp(neg_var, min=1e-12))
        neg_kurt = (self.neg_m4 / neg_n) / (neg_var.pow(2) + 1e-12)
        neg_null_kurt = torch.where(neg_rate > 0.25, 22.0, 12.5)
        neg_eff_kurt = torch.clamp(neg_kurt - neg_null_kurt, min=1.0)
        neg_snr = torch.where(
            self.neg_count == 1,
            torch.full_like(self.neg_mean, 6.0),
            self.neg_mean.abs() / torch.clamp(neg_std, min=1e-6),
        )
        neg_focal_kurt = torch.where(
            (neg_rate <= 0.15) & (self.neg_mean.abs() >= overall_std),
            torch.clamp((neg_snr - 2.0) * 6.0, min=1.0, max=64.0),
            torch.ones_like(neg_snr),
        )
        neg_eff_kurt = torch.maximum(neg_eff_kurt, neg_focal_kurt)
        neg_raw_sparsity = (0.20 / torch.clamp(neg_rate, min=1e-4)).pow(0.25)
        neg_sparsity_factor = torch.where(
            neg_eff_kurt > 1.05,
            neg_raw_sparsity,
            torch.clamp(neg_raw_sparsity, max=1.0),
        )
        neg_sel = torch.where(
            self.neg_count > 0,
            torch.clamp(neg_eff_kurt.pow(0.25) * neg_sparsity_factor, min=0.0, max=100.0),
            torch.zeros_like(neg_var),
        )

        return pos_std, pos_sel, neg_std, neg_sel

    def _compute_token_selectivity_and_motifs(
        self, neuron_idx: int, pos_rate: float, neg_rate: float, pos_std: float, neg_std: float
    ) -> Tuple[float, float, float, float, str]:
        """
        Computes token-conditional lexical consistency and firing motif diagnostics
        (Cammarata: 'What Do Neurons Actually Compute?') from collected samples:
          - `lexical_selectivity_r2`: token-group ANOVA variance ratio R^2 = SS_between / SS_total
            (or top-token concentration when sample count is small)
          - `current_token_concentration`: share of active samples on the top-2 focal tokens
          - `next_token_concentration`: share of active samples on the top-2 next tokens
          - `punctuation_firing_rate`: share of active samples firing on punctuation / clause boundaries
          - `inferred_firing_motif`: 'current_token_detector' | 'next_token_preparer' |
            'punctuation_consolidation' | 'bipolar_router' | 'distributed_contextual'
        """
        seen = set()
        samples: List[ActivationSample] = []
        for _, _, s in (
            self.pos_heaps.get(neuron_idx, [])
            + self.neg_heaps.get(neuron_idx, [])
            + self.pos_band_reservoirs.get(neuron_idx, [])
            + self.neg_band_reservoirs.get(neuron_idx, [])
        ):
            key = (s.sample_id, s.position_idx, round(s.activation_value, 6))
            if key not in seen:
                seen.add(key)
                samples.append(s)

        tok_samples = [s for s in samples if s.token_str is not None]
        if not tok_samples:
            motif = (
                "bipolar_router"
                if (pos_rate >= 0.05 and neg_rate >= 0.05 and pos_std > 0.1 and neg_std > 0.1)
                else "distributed_contextual"
            )
            return 0.0, 0.0, 0.0, 0.0, motif

        n_tok = len(tok_samples)
        tok_counts = Counter(s.token_str for s in tok_samples if s.token_str)
        top2_curr = sum(c for _, c in tok_counts.most_common(2))
        curr_conc = float(top2_curr / max(n_tok, 1))

        next_valid = [
            s.next_token_str
            for s in tok_samples
            if s.next_token_str is not None and s.next_token_str != "<eos>"
        ]
        if next_valid:
            next_counts = Counter(next_valid)
            top2_next = sum(c for _, c in next_counts.most_common(2))
            next_conc = float(top2_next / len(next_valid))
        else:
            next_conc = 0.0

        punct_hits = sum(
            1
            for s in tok_samples
            if s.token_str
            and (
                all(ch in self._PUNCTUATION_CHARS for ch in s.token_str.strip())
                or s.token_str.strip() in {".", ",", ";", ":", "!", "?", "(", ")", "-", "/"}
            )
        )
        punct_rate = float(punct_hits / max(n_tok, 1))

        # Between-token vs total activation variance ratio R^2 = SS_between / SS_total
        # restricted to repeated tokens (len(g) >= 2) to prevent singleton degrees-of-freedom inflation
        vals = [float(s.activation_value) for s in tok_samples]
        grand_mean = sum(vals) / n_tok
        ss_total = sum((v - grand_mean) ** 2 for v in vals)
        if ss_total > 1e-9 and len(tok_counts) < n_tok:
            by_tok: Dict[str, List[float]] = defaultdict(list)
            for s in tok_samples:
                if s.token_str is not None:
                    by_tok[s.token_str].append(float(s.activation_value))
            repeated_groups = [g for g in by_tok.values() if len(g) >= 2]
            if repeated_groups:
                ss_between = sum(
                    len(g) * ((sum(g) / len(g)) - grand_mean) ** 2 for g in repeated_groups
                )
                raw_r2 = float(min(max(ss_between / ss_total, 0.0), 1.0))
                k_rep = len(repeated_groups)
                if n_tok > k_rep + 2:
                    adj_r2 = 1.0 - (1.0 - raw_r2) * (n_tok - 1.0) / float(n_tok - k_rep - 1)
                    r2 = float(min(max(adj_r2, 0.0), 1.0))
                else:
                    r2 = raw_r2 * float(sum(len(g) for g in repeated_groups) / max(n_tok, 1))
            else:
                r2 = 0.0
        else:
            r2 = 0.0

        if punct_rate >= 0.40:
            motif = "punctuation_consolidation"
        elif next_conc > curr_conc + 0.15 and next_conc >= 0.35:
            motif = "next_token_preparer"
        elif pos_rate >= 0.05 and neg_rate >= 0.05 and pos_std > 0.1 and neg_std > 0.1 and r2 >= 0.25:
            motif = "bipolar_router"
        elif curr_conc >= 0.25:
            motif = "current_token_detector"
        else:
            motif = "distributed_contextual"

        return (
            round(r2, 4),
            round(curr_conc, 4),
            round(next_conc, 4),
            round(punct_rate, 4),
            motif,
        )

    def compute_adaptive_thresholds(
        self,
        neuron_indices: List[int],
        std_multiplier: Optional[float] = None,
        percentile: Optional[float] = None,
        pole: PoleType = PoleType.BOTH,
    ) -> List[float]:
        """
        Computes per-neuron native activation thresholds from audited statistics
        (Cammarata: 'Finding Clean Circuits' — Respecting Native Activation Ranges & Percentiles).
          - If `percentile` (in (0, 100)) is provided, computes the empirical quantile over
            each neuron's unbiased Algorithm R active-magnitude reservoir (`pos_mag_reservoirs`,
            `neg_mag_reservoirs`, or both depending on `pole`), avoiding top-k outlier bias.
          - Otherwise uses `|mean_i| + k * std_i` where `k = std_multiplier or 1.0`.
        """
        n = max(self.count, 1)
        var_vec = self.m2 / n
        std_vec = torch.sqrt(torch.clamp(var_vec, min=1e-12))
        thresholds: List[float] = []

        for idx in neuron_indices:
            if percentile is not None:
                q = min(max(float(percentile) / 100.0, 0.01), 0.99)
                if pole == PoleType.POSITIVE:
                    mags = list(self.pos_mag_reservoirs.get(idx, []))
                elif pole == PoleType.NEGATIVE:
                    mags = list(self.neg_mag_reservoirs.get(idx, []))
                else:
                    mags = list(self.pos_mag_reservoirs.get(idx, [])) + list(
                        self.neg_mag_reservoirs.get(idx, [])
                    )
                if len(mags) >= 4:
                    mags_sorted = sorted(mags)
                    pos_f = q * (len(mags_sorted) - 1)
                    lo = int(math.floor(pos_f))
                    hi = int(math.ceil(pos_f))
                    frac = pos_f - lo
                    interp = mags_sorted[lo] * (1.0 - frac) + mags_sorted[hi] * frac
                    thresholds.append(float(interp))
                else:
                    # Fallback to Gaussian quantile on neuron's native std
                    z_approx = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0)).item()
                    base_std = float(std_vec[idx].item())
                    thresholds.append(
                        max(abs(float(self.mean[idx].item())) + max(z_approx, 0.1) * base_std, 1e-6)
                    )
            else:
                k = float(std_multiplier if std_multiplier is not None else 1.0)
                base_mean = abs(float(self.mean[idx].item()))
                base_std = float(std_vec[idx].item())
                thresholds.append(base_mean + k * base_std)

        return thresholds

    def export_profile(self, neuron_idx: int) -> NeuronPoleProfile:
        """Compiles the final NeuronPoleProfile for a given neuron index."""
        n = max(self.count, 1)
        mean_val = float(self.mean[neuron_idx].item())
        var_val = float((self.m2[neuron_idx] / n).item())
        std_val = max(var_val ** 0.5, 1e-9)

        pos_rate = float(self.pos_count[neuron_idx].item() / n)
        neg_rate = float(self.neg_count[neuron_idx].item() / n)
        tot_rate = pos_rate + neg_rate

        raw_kurtosis = float((self.m4[neuron_idx] / n).item()) / ((var_val ** 2) + 1e-12)
        if pos_rate > 0.25 and neg_rate > 0.25:
            eff_kurtosis = max(raw_kurtosis - 20.0, 1.0)
        else:
            active_kurt = raw_kurtosis * max(tot_rate, 1e-4)
            eff_kurtosis = max(active_kurt - 12.5, 1.0)
        raw_sparsity = (0.20 / max(tot_rate, 1e-4)) ** 0.25 if tot_rate > 0 else 0.0
        sparsity_factor = raw_sparsity if eff_kurtosis > 1.05 else min(raw_sparsity, 1.0)
        selectivity = float(min(max((eff_kurtosis ** 0.25) * sparsity_factor, 0.0), 100.0))

        pos_std_vec, pos_sel_vec, neg_std_vec, neg_sel_vec = (
            self._compute_pole_selectivities_and_stds()
        )
        pos_sel = float(pos_sel_vec[neuron_idx].item())
        neg_sel = float(neg_sel_vec[neuron_idx].item())
        pos_std = float(pos_std_vec[neuron_idx].item())
        neg_std = float(neg_std_vec[neuron_idx].item())

        all_stds = torch.sqrt(torch.clamp(self.m2 / n, min=1e-12))
        active_stds = all_stds[all_stds > 1e-6]
        median_std = float(torch.median(active_stds).item()) if active_stds.numel() >= 4 else std_val
        scale_salience = std_val / max(median_std, 1e-6) if self.num_neurons >= 8 else 1.0

        r2, curr_conc, next_conc, punct_rate, motif = self._compute_token_selectivity_and_motifs(
            neuron_idx, pos_rate, neg_rate, pos_std, neg_std
        )
        salient_for_lex = (
            scale_salience >= 4.0
            if self.num_neurons >= 8
            else max(selectivity, pos_sel, neg_sel) >= 1.50
        )
        if r2 >= 0.35 and salient_for_lex:
            scale_boost = (
                min(max(0.45 * math.log1p(max(scale_salience - 3.5, 0.0)), 0.0), 2.5)
                if scale_salience >= 4.0
                else 0.0
            )
            lex_bonus = 0.85 * r2 + scale_boost
            selectivity = round(selectivity + lex_bonus, 3)
            if pos_rate > 0:
                pos_sel = round(pos_sel + lex_bonus, 3)
            if neg_rate > 0:
                neg_sel = round(neg_sel + lex_bonus, 3)

        pos_cnt = int(self.pos_count[neuron_idx].item())
        dbl_neg_cnt = int(self.double_neg_count[neuron_idx].item())
        dbl_neg_rate = float(dbl_neg_cnt / max(pos_cnt, 1)) if pos_cnt > 0 else 0.0

        # 4-quadrant SwiGLU sign regime rates
        q_pp = int(self.q_pp_count[neuron_idx].item())
        q_nn = int(self.q_nn_count[neuron_idx].item())
        q_pn = int(self.q_pn_count[neuron_idx].item())
        q_np = int(self.q_np_count[neuron_idx].item())
        q_tot = q_pp + q_nn + q_pn + q_np
        if q_tot > 0:
            swiglu_regimes = {
                "pos_gate_pos_up": round(q_pp / q_tot, 4),
                "neg_gate_neg_up": round(q_nn / q_tot, 4),
                "pos_gate_neg_up": round(q_pn / q_tot, 4),
                "neg_gate_pos_up": round(q_np / q_tot, 4),
            }
        else:
            swiglu_regimes = {}

        pos_tax = self._classify_pole_taxonomy(pos_sel, pos_rate, pos_std)
        neg_tax = self._classify_pole_taxonomy(neg_sel, neg_rate, neg_std)
        pos_prior = self._classify_pole_prior(pos_sel, pos_rate, pos_std, other_pole_rate=neg_rate)
        neg_prior = self._classify_pole_prior(neg_sel, neg_rate, neg_std, other_pole_rate=pos_rate)

        pos_samples = [
            item[2]
            for item in sorted(self.pos_heaps.get(neuron_idx, []), key=lambda x: x[0], reverse=True)
        ]
        neg_samples = [
            item[2]
            for item in sorted(self.neg_heaps.get(neuron_idx, []), key=lambda x: x[0], reverse=True)
        ]
        dbl_neg_samples = [
            item[2]
            for item in sorted(
                self.double_neg_heaps.get(neuron_idx, []), key=lambda x: x[0], reverse=True
            )
        ]

        max_v = float(self.max_val[neuron_idx].item())
        min_v = float(self.min_val[neuron_idx].item())
        clean_max_v = max_v if max_v != float("-inf") else 0.0
        clean_min_v = min_v if min_v != float("inf") else 0.0

        pos_buckets, dbl_band_rates = self._build_quantile_buckets(
            neuron_idx, is_positive=True, pole_max_abs=max(clean_max_v, 0.0)
        )
        neg_buckets, _ = self._build_quantile_buckets(
            neuron_idx, is_positive=False, pole_max_abs=abs(min(clean_min_v, 0.0))
        )

        return NeuronPoleProfile(
            layer_name=self.layer_name,
            neuron_idx=neuron_idx,
            mean=mean_val,
            std=std_val,
            max_val=clean_max_v,
            min_val=clean_min_v,
            positive_firing_rate=pos_rate,
            negative_firing_rate=neg_rate,
            pole_selectivity_score=selectivity,
            top_positive_samples=pos_samples,
            top_negative_samples=neg_samples,
            pos_pole_mean=float(self.pos_mean[neuron_idx].item()),
            pos_pole_std=pos_std,
            pos_pole_selectivity_score=round(pos_sel, 3),
            pos_taxonomy_class=pos_tax,
            pos_sparsity_prior=pos_prior,
            neg_pole_mean=-float(self.neg_mean[neuron_idx].item()),
            neg_pole_std=neg_std,
            neg_pole_selectivity_score=round(neg_sel, 3),
            neg_taxonomy_class=neg_tax,
            neg_sparsity_prior=neg_prior,
            positive_quantile_buckets=pos_buckets,
            negative_quantile_buckets=neg_buckets,
            double_negative_positive_rate=round(dbl_neg_rate, 4),
            double_negative_samples=dbl_neg_samples,
            swiglu_sign_regime_rates=swiglu_regimes,
            double_negative_band_rates=dbl_band_rates,
            lexical_selectivity_r2=r2,
            current_token_concentration=curr_conc,
            next_token_concentration=next_conc,
            punctuation_firing_rate=punct_rate,
            inferred_firing_motif=motif,
        )

    @torch.no_grad()
    def scan_layer_cleanliness(
        self,
        down_proj_module: Optional[nn.Module] = None,
        verified_hypotheses: Optional[Dict[Tuple[str, int], BipolarVerbHypothesis]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, float], List[NeuronScanSummary]]:
        """
        Vectorized GPU pass over all tracked moments in the layer to classify every neuron
        AND every individual neuron pole into:
          1. Statistical `SparsityTailPrior` (null-calibrated excess kurtosis + sparsity prior)
          2. `NeuronTaxonomy` (promoted when semantically verified in `CircuitEcologyMap` or when
             excess tail-selectivity clearly exceeds the Gaussian/SwiGLU random-init null).
        """
        n = max(self.count, 1)
        var_vec = self.m2 / n
        std_vec = torch.sqrt(torch.clamp(var_vec, min=1e-12))

        pos_rate_vec = self.pos_count.to(torch.float64) / n
        neg_rate_vec = self.neg_count.to(torch.float64) / n
        total_rate_vec = pos_rate_vec + neg_rate_vec

        kurt_vec = (self.m4 / n) / (var_vec.pow(2) + 1e-12)
        active_kurt_vec = kurt_vec * torch.clamp(total_rate_vec, min=1e-4)
        eff_kurt_vec = torch.where(
            (pos_rate_vec > 0.25) & (neg_rate_vec > 0.25),
            torch.clamp(kurt_vec - 20.0, min=1.0),
            torch.clamp(active_kurt_vec - 12.5, min=1.0),
        )
        raw_sparsity_vec = torch.where(
            total_rate_vec > 0,
            (0.20 / torch.clamp(total_rate_vec, min=1e-4)).pow(0.25),
            torch.zeros_like(total_rate_vec),
        )
        sparsity_factor_vec = torch.where(
            eff_kurt_vec > 1.05,
            raw_sparsity_vec,
            torch.clamp(raw_sparsity_vec, max=1.0),
        )
        selectivity_vec = torch.clamp(
            eff_kurt_vec.pow(0.25) * sparsity_factor_vec, min=0.0, max=100.0
        )

        dbl_neg_rate_vec = self.double_neg_count.to(torch.float64) / torch.clamp(
            self.pos_count.to(torch.float64), min=1.0
        )

        bipolar_vec = (4.0 * pos_rate_vec * neg_rate_vec) / (total_rate_vec.pow(2) + 1e-9)
        pos_std_vec, pos_sel_vec, neg_std_vec, neg_sel_vec = (
            self._compute_pole_selectivities_and_stds()
        )
        selectivity_vec = torch.maximum(selectivity_vec, torch.maximum(pos_sel_vec, neg_sel_vec))

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
                # Standard Linear down_proj: [d_model, d_mlp] (or output-hooked [d_mlp, d_in])
                if w.shape[1] == self.num_neurons:
                    weight_norms = torch.norm(w, p=2, dim=0)
                elif w.shape[0] == self.num_neurons:
                    weight_norms = torch.norm(w, p=2, dim=1)
        elif (
            down_proj_module is not None
            and hasattr(down_proj_module, "down_proj")
            and isinstance(getattr(down_proj_module, "down_proj"), torch.Tensor)
        ):
            w = getattr(down_proj_module, "down_proj").detach().to(self.device, dtype=torch.float64)
            if w.ndim == 3 and w.shape[-1] == self.num_neurons:
                weight_norms = torch.norm(w, p=2, dim=1).mean(dim=0)

        impact_vec = std_vec * weight_norms
        active_stds = std_vec[std_vec > 1e-6]
        median_std = (
            float(torch.median(active_stds).item()) if active_stds.numel() >= 4 else 1.0
        )
        active_impacts = impact_vec[impact_vec > 1e-6]
        median_impact = (
            float(torch.median(active_impacts).item()) if active_impacts.numel() >= 4 else 1.0
        )

        sel_cpu = selectivity_vec.cpu().tolist()
        bip_cpu = bipolar_vec.cpu().tolist()
        pos_cpu = pos_rate_vec.cpu().tolist()
        neg_cpu = neg_rate_vec.cpu().tolist()
        dbl_cpu = dbl_neg_rate_vec.cpu().tolist()
        mean_cpu = self.mean.cpu().tolist()
        std_cpu = std_vec.cpu().tolist()
        max_cpu = self.max_val.cpu().tolist()
        min_cpu = self.min_val.cpu().tolist()
        wnorm_cpu = weight_norms.cpu().tolist()
        imp_cpu = impact_vec.cpu().tolist()
        pos_sel_cpu = pos_sel_vec.cpu().tolist()
        neg_sel_cpu = neg_sel_vec.cpu().tolist()
        pos_std_cpu = pos_std_vec.cpu().tolist()
        neg_std_cpu = neg_std_vec.cpu().tolist()

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
            pos_r = float(pos_cpu[idx])
            neg_r = float(neg_cpu[idx])
            dbl_r = float(dbl_cpu[idx]) if pos_r > 0 else 0.0
            tot_rate = pos_r + neg_r

            pos_sel_i = float(pos_sel_cpu[idx])
            neg_sel_i = float(neg_sel_cpu[idx])
            pos_std_i = float(pos_std_cpu[idx])
            neg_std_i = float(neg_std_cpu[idx])

            pos_heap = self.pos_heaps.get(idx, [])
            neg_heap = self.neg_heaps.get(idx, [])
            r2_i = 0.0
            scale_salience = (
                max(std_v / max(median_std, 1e-6), float(imp_cpu[idx]) / max(median_impact, 1e-6))
                if self.num_neurons >= 8
                else 1.0
            )
            if (pos_heap and pos_heap[0][2].token_str is not None) or (
                neg_heap and neg_heap[0][2].token_str is not None
            ):
                r2_i, _, _, _, _ = self._compute_token_selectivity_and_motifs(
                    idx, pos_r, neg_r, pos_std_i, neg_std_i
                )
                salient_for_lex = (
                    scale_salience >= 4.0
                    if self.num_neurons >= 8
                    else max(sel, pos_sel_i, neg_sel_i) >= 1.50
                )
                if r2_i >= 0.35 and salient_for_lex:
                    scale_boost = (
                        min(max(0.45 * math.log1p(max(scale_salience - 3.5, 0.0)), 0.0), 2.5)
                        if scale_salience >= 4.0
                        else 0.0
                    )
                    lex_bonus = 0.85 * r2_i + scale_boost
                    sel += lex_bonus
                    if pos_r > 0:
                        pos_sel_i += lex_bonus
                    if neg_r > 0:
                        neg_sel_i += lex_bonus

            pos_tax = self._classify_pole_taxonomy(pos_sel_i, pos_r, pos_std_i, wnorm)
            neg_tax = self._classify_pole_taxonomy(neg_sel_i, neg_r, neg_std_i, wnorm)
            prior = self._classify_pole_prior(
                max(sel, pos_sel_i, neg_sel_i),
                min(pos_r if pos_r > 0 else 1.0, neg_r if neg_r > 0 else 1.0),
                std_v,
                other_pole_rate=max(pos_r, neg_r),
                outgoing_weight_norm=wnorm,
            )

            # Statistical null-calibrated tail-selectivity classification
            if std_v < 1e-5 or wnorm < 1e-5 or tot_rate < 1e-4:
                tax = NeuronTaxonomy.DEAD_OR_NOISY
            elif sel >= 2.05 or max(pos_sel_i, neg_sel_i) >= 2.1:
                tax = NeuronTaxonomy.CLEAN_MONOSEMANTIC
            elif sel >= 1.48 or max(pos_sel_i, neg_sel_i) >= 1.52:
                tax = NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE
            else:
                tax = NeuronTaxonomy.TRUE_POLYSEMANTIC

            # Override with semantically verified taxonomy if present in CircuitEcologyMap
            is_verified = False
            if verified_hypotheses and (self.layer_name, idx) in verified_hypotheses:
                v_hyp = verified_hypotheses[(self.layer_name, idx)]
                if (
                    v_hyp.verification_report is not None
                    and v_hyp.verification_report.empirical_verification_score >= 0.6
                ):
                    tax = v_hyp.taxonomy_class
                    is_verified = True

            if tax.value not in counts:
                counts[tax.value] = 0
            counts[tax.value] += 1

            top_pos_tok = max(pos_heap, key=lambda x: x[0])[2].token_str if pos_heap else None
            top_neg_tok = max(neg_heap, key=lambda x: x[0])[2].token_str if neg_heap else None

            summaries.append(
                NeuronScanSummary(
                    neuron_idx=idx,
                    taxonomy_class=tax,
                    pole_selectivity_score=round(sel, 3),
                    bipolar_balance=round(bip, 3),
                    positive_firing_rate=round(pos_r, 4),
                    negative_firing_rate=round(neg_r, 4),
                    mean=round(float(mean_cpu[idx]), 4),
                    std=round(std_v, 4),
                    max_val=round(float(max_cpu[idx]) if max_cpu[idx] != float("-inf") else 0.0, 4),
                    min_val=round(float(min_cpu[idx]) if min_cpu[idx] != float("inf") else 0.0, 4),
                    outgoing_weight_norm=round(wnorm, 4),
                    estimated_causal_impact=round(float(imp_cpu[idx]), 4),
                    top_pos_token=top_pos_tok,
                    top_neg_token=top_neg_tok,
                    pos_pole_selectivity_score=round(pos_sel_i, 3),
                    neg_pole_selectivity_score=round(neg_sel_i, 3),
                    pos_taxonomy_class=pos_tax,
                    neg_taxonomy_class=neg_tax,
                    double_negative_positive_rate=round(dbl_r, 4),
                    semantic_verified=is_verified,
                    sparsity_tail_prior=prior,
                    lexical_selectivity_r2=round(r2_i, 4),
                )
            )

        total_n = max(self.num_neurons, 1)
        percentages = {k: round((v / total_n) * 100.0, 2) for k, v in counts.items()}
        return counts, percentages, summaries

    def save_state(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Serializes all streaming moment tensors, heaps, and reservoirs into a portable
        state dictionary (optionally saved via `torch.save` to `path`) so multi-hour
        corpus audits can be checkpointed and resumed across sessions.
        """
        state: Dict[str, Any] = {
            "layer_name": self.layer_name,
            "num_neurons": self.num_neurons,
            "top_k": self.top_k,
            "count": self.count,
            "mean": self.mean.cpu(),
            "m2": self.m2.cpu(),
            "m3": self.m3.cpu(),
            "m4": self.m4.cpu(),
            "max_val": self.max_val.cpu(),
            "min_val": self.min_val.cpu(),
            "pos_count": self.pos_count.cpu(),
            "pos_mean": self.pos_mean.cpu(),
            "pos_m2": self.pos_m2.cpu(),
            "pos_m3": self.pos_m3.cpu(),
            "pos_m4": self.pos_m4.cpu(),
            "neg_count": self.neg_count.cpu(),
            "neg_mean": self.neg_mean.cpu(),
            "neg_m2": self.neg_m2.cpu(),
            "neg_m3": self.neg_m3.cpu(),
            "neg_m4": self.neg_m4.cpu(),
            "double_neg_count": self.double_neg_count.cpu(),
            "q_pp_count": self.q_pp_count.cpu(),
            "q_nn_count": self.q_nn_count.cpu(),
            "q_pn_count": self.q_pn_count.cpu(),
            "q_np_count": self.q_np_count.cpu(),
            "pos_mag_reservoirs": dict(self.pos_mag_reservoirs),
            "neg_mag_reservoirs": dict(self.neg_mag_reservoirs),
            "pos_mag_seen": dict(self._pos_mag_seen),
            "neg_mag_seen": dict(self._neg_mag_seen),
        }
        if path is not None:
            torch.save(state, str(path))
        return state

    @classmethod
    def load_state(
        cls,
        state_or_path: Union[Dict[str, Any], str, Path],
        device: str = "cpu",
    ) -> StreamingLayerStats:
        """Restores a `StreamingLayerStats` instance from `save_state()` output or file path."""
        state = (
            torch.load(str(state_or_path), map_location="cpu", weights_only=False)
            if isinstance(state_or_path, (str, Path))
            else state_or_path
        )
        inst = cls(
            layer_name=state["layer_name"],
            num_neurons=int(state["num_neurons"]),
            top_k=int(state.get("top_k", 10)),
            device=device,
        )
        inst.count = int(state["count"])
        for attr in (
            "mean",
            "m2",
            "m3",
            "m4",
            "max_val",
            "min_val",
            "pos_count",
            "pos_mean",
            "pos_m2",
            "pos_m3",
            "pos_m4",
            "neg_count",
            "neg_mean",
            "neg_m2",
            "neg_m3",
            "neg_m4",
            "double_neg_count",
            "q_pp_count",
            "q_nn_count",
            "q_pn_count",
            "q_np_count",
        ):
            if attr in state:
                getattr(inst, attr).copy_(state[attr].to(device=device))
        for k, v in state.get("pos_mag_reservoirs", {}).items():
            inst.pos_mag_reservoirs[int(k)] = list(v)
        for k, v in state.get("neg_mag_reservoirs", {}).items():
            inst.neg_mag_reservoirs[int(k)] = list(v)
        for k, v in state.get("pos_mag_seen", {}).items():
            inst._pos_mag_seen[int(k)] = int(v)
        for k, v in state.get("neg_mag_seen", {}).items():
            inst._neg_mag_seen[int(k)] = int(v)
        return inst


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
          (automatically handles Gemma's `1 + weight` RMSNorm parameterization).
        """
        if hasattr(down_proj_module, "weight") and isinstance(down_proj_module.weight, torch.Tensor):
            weight = down_proj_module.weight.detach()
        elif hasattr(down_proj_module, "down_proj") and isinstance(getattr(down_proj_module, "down_proj"), torch.Tensor):
            weight = getattr(down_proj_module, "down_proj").detach()
        else:
            return [], []
        if weight.ndim == 3:
            # 3D MoE down_proj: [num_experts, d_model, d_mlp] -> average write direction across experts
            neuron_out_vec = weight[:, :, neuron_idx].mean(dim=0).to(dtype=torch.float32)
        elif down_proj_module.__class__.__name__ == "Conv1D":
            # [d_mlp, d_model]
            neuron_out_vec = weight[neuron_idx, :].to(dtype=torch.float32)
        else:
            # Standard nn.Linear: [d_model, d_mlp]
            if neuron_idx < weight.shape[1]:
                neuron_out_vec = weight[:, neuron_idx].to(dtype=torch.float32)
            else:
                neuron_out_vec = weight[neuron_idx, :].to(dtype=torch.float32)

        # Apply final RMSNorm / LayerNorm scaling vector gamma before unembedding
        # (including Gemma's 1.0 + weight offset via LayerDiscovery.extract_norm_scale)
        if final_norm_module is not None:
            norm_scale = LayerDiscovery.extract_norm_scale(
                final_norm_module,
                d_model=neuron_out_vec.shape[0],
                device=neuron_out_vec.device,
                dtype=torch.float32,
            )
            if norm_scale is not None and norm_scale.shape == neuron_out_vec.shape:
                neuron_out_vec = neuron_out_vec * norm_scale

        # Project onto vocabulary space: [vocab_size]
        unembed = unembed_weight.detach().to(neuron_out_vec.device, dtype=torch.float32)
        if unembed.shape[1] != neuron_out_vec.shape[0] and unembed.shape[0] == neuron_out_vec.shape[0]:
            unembed = unembed.T

        if unembed.shape[1] != neuron_out_vec.shape[0]:
            return [], []

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


class ContrastivePoleFinder:
    """
    Contrastive Pole Discovery Engine (Cammarata: 'Thinking in Neurons' & 'Finding Clean Circuits'):
    Ranks all native-basis neurons across one or more layers by regularized Welch's
    Bipolar Separation t-statistic between two contrastive input sets (D+ and D-):
        t_i = (mu+(a_i) - mu-(a_i)) / (sqrt(var+(a_i)/N+ + var-(a_i)/N-) + s_0)
    Directly surfaces native 'verb' neurons that route between positive and negative
    behaviors without requiring an external Sparse Autoencoder (SAE).
    """

    @staticmethod
    @torch.no_grad()
    def discover(
        model: nn.Module,
        pos_inputs: Union[torch.Tensor, Iterable[Any]],
        neg_inputs: Union[torch.Tensor, Iterable[Any]],
        layer_names: List[str],
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        top_k: int = 16,
        device: str = "cpu",
    ) -> ContrastiveDiscoveryReport:
        run_fwd = forward_fn or (lambda m, inp: m(inp))

        def _normalize_batches(data: Union[torch.Tensor, Iterable[Any]]) -> List[Any]:
            if isinstance(data, torch.Tensor):
                return [data]
            return list(data)

        pos_batches = _normalize_batches(pos_inputs)
        neg_batches = _normalize_batches(neg_inputs)

        def _collect_stats(batches: List[Any]) -> Dict[str, StreamingLayerStats]:
            trackers: Dict[str, StreamingLayerStats] = {}
            handles: List[RemovableHandle] = []
            current_mask: Dict[str, Optional[torch.Tensor]] = {"mask": None}

            for target_layer in layer_names:
                mod, hook_target, num_n = LayerDiscovery.resolve_module(model, target_layer)
                st = StreamingLayerStats(target_layer, num_n, top_k=1, device=device)
                trackers[target_layer] = st
                if hook_target == "input":
                    handles.append(
                        mod.register_forward_pre_hook(
                            lambda _m, inp, s=st: (
                                s.update(
                                    inp[0].detach(),
                                    tracked_neurons=[],
                                    attention_mask=current_mask["mask"],
                                )
                                if inp and isinstance(inp[0], torch.Tensor)
                                else None
                            )
                        )
                    )
                else:
                    handles.append(
                        mod.register_forward_hook(
                            lambda _m, _inp, out, s=st: s.update(
                                (out[0] if isinstance(out, tuple) else out).detach(),
                                tracked_neurons=[],
                                attention_mask=current_mask["mask"],
                            )
                        )
                    )
            try:
                for b in batches:
                    current_mask["mask"] = (
                        b.get("attention_mask") if isinstance(b, dict) else None
                    )
                    run_fwd(model, b)
            finally:
                for handle in handles:
                    handle.remove()
            return trackers

        was_training = model.training
        model.eval()
        try:
            pos_stats = _collect_stats(pos_batches)
            neg_stats = _collect_stats(neg_batches)
        finally:
            if was_training:
                model.train()

        matches: List[ContrastiveNeuronMatch] = []
        pos_tokens_count = 0
        neg_tokens_count = 0

        for layer_name in layer_names:
            ps = pos_stats[layer_name]
            ns = neg_stats[layer_name]
            n_pos = max(ps.count, 1)
            n_neg = max(ns.count, 1)
            pos_tokens_count = max(pos_tokens_count, ps.count)
            neg_tokens_count = max(neg_tokens_count, ns.count)

            var_pos = ps.m2 / n_pos
            var_neg = ns.m2 / n_neg
            raw_se_vec = torch.sqrt((var_pos / n_pos) + (var_neg / n_neg) + 1e-9)
            pooled_std = torch.sqrt(0.5 * (var_pos + var_neg) + 1e-12)
            # Scale-relative standard-error floor s0 so near-dead background neurons with
            # tiny variance do not outrank large-effect routers, while preserving sensitivity
            # across layers of any activation magnitude.
            scale_floor = max(float(pooled_std.median().item()) * 0.25, 1e-4)
            s0 = float(raw_se_vec.median().item()) * 0.25 + scale_floor
            t_vec = (ps.mean - ns.mean) / (raw_se_vec + s0)

            std_pos = torch.sqrt(torch.clamp(var_pos, min=0.0))
            std_neg = torch.sqrt(torch.clamp(var_neg, min=0.0))
            sign_floor = scale_floor * 0.05

            t_cpu = t_vec.cpu().tolist()
            mp_cpu = ps.mean.cpu().tolist()
            mn_cpu = ns.mean.cpu().tolist()
            sp_cpu = std_pos.cpu().tolist()
            sn_cpu = std_neg.cpu().tolist()

            for idx in range(ps.num_neurons):
                t_val = float(t_cpu[idx])
                mp = float(mp_cpu[idx])
                mn = float(mn_cpu[idx])
                sign_flip = (mp > sign_floor and mn < -sign_floor) or (
                    mp < -sign_floor and mn > sign_floor
                )
                rec_pole = PoleType.POSITIVE if t_val >= 0.0 else PoleType.NEGATIVE
                matches.append(
                    ContrastiveNeuronMatch(
                        layer_name=layer_name,
                        neuron_idx=idx,
                        t_statistic=round(t_val, 4),
                        abs_t_statistic=round(abs(t_val), 4),
                        pos_mean=round(mp, 4),
                        neg_mean=round(mn, 4),
                        pos_std=round(float(sp_cpu[idx]), 4),
                        neg_std=round(float(sn_cpu[idx]), 4),
                        bipolar_sign_flip=sign_flip,
                        recommended_pole=rec_pole,
                    )
                )

        matches.sort(
            key=lambda m: m.abs_t_statistic * (1.25 if m.bipolar_sign_flip else 1.0),
            reverse=True,
        )
        return ContrastiveDiscoveryReport(
            layers_scanned=list(layer_names),
            pos_tokens_count=pos_tokens_count,
            neg_tokens_count=neg_tokens_count,
            top_matches=matches[:top_k],
        )


class CheckpointEvolutionTracker:
    """
    Cross-Checkpoint 'Recursive Growth' Tracker (Cammarata: 'On Generalization — Recursive Growth'):
    Inspired by Cammarata's observation of neurons sprouting recursive offshoots across
    training epochs, `CheckpointEvolutionTracker` audits a specific `(layer_name, neuron_idx)`
    across a sequence of `(checkpoint_label, model_or_loader_fn)` pairs on a shared evaluation dataset,
    tracking how pole selectivity, outgoing residual write vectors, SwiGLU double-negative rate,
    and stratified quantile buckets (`peak_trunk`, `secondary_branch`, `tertiary_offshoot`) evolve.
    """

    @staticmethod
    @torch.no_grad()
    def track_neuron_evolution(
        checkpoints: List[Tuple[str, Union[nn.Module, Callable[[], nn.Module]]]],
        layer_name: str,
        neuron_idx: int,
        batches: Iterable[Dict[str, Any]],
        forward_fn: Callable[[nn.Module, Dict[str, Any]], Any],
        top_k: int = 8,
    ) -> NeuronEvolutionReport:
        from .injector import NeuralInjector

        batch_list = list(batches)
        snapshots: List[NeuronCheckpointSnapshot] = []
        write_vecs: List[torch.Tensor] = []
        raw_profiles: List[Tuple[str, NeuronPoleProfile]] = []

        # Stream one checkpoint at a time so lazy callables `(label, lambda: load_ckpt(...))`
        # never require holding multiple multi-GB models in memory simultaneously.
        for label, ckpt_or_fn in checkpoints:
            ckpt_model = ckpt_or_fn() if callable(ckpt_or_fn) and not isinstance(ckpt_or_fn, nn.Module) else ckpt_or_fn
            mod, _, _ = LayerDiscovery.resolve_module(ckpt_model, layer_name)
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                w = mod.weight.detach().float()
                if w.ndim == 3:
                    v = w[:, :, neuron_idx].mean(dim=0)
                elif mod.__class__.__name__ == "Conv1D":
                    v = w[neuron_idx, :]
                else:
                    v = w[:, neuron_idx] if neuron_idx < w.shape[1] else w[neuron_idx, :]
                write_vecs.append(v.cpu())
            else:
                write_vecs.append(torch.ones(1))

            inj = NeuralInjector(ckpt_model)
            inj.audit_layers(
                layer_names=[layer_name],
                dataloader_or_batches=batch_list,
                forward_fn=forward_fn,
                top_k=top_k,
                tracked_neurons=[neuron_idx],
            )
            prof = inj.get_neuron_profile(layer_name, neuron_idx)
            raw_profiles.append((label, prof))

        final_vec = (
            F.normalize(write_vecs[-1].unsqueeze(0), dim=-1).squeeze(0)
            if write_vecs
            else torch.ones(1)
        )

        for (label, prof), v_curr in zip(raw_profiles, write_vecs):
            wnorm = float(torch.norm(v_curr, p=2).item())
            if v_curr.shape == final_vec.shape and wnorm > 1e-9:
                cos_final = float(
                    torch.dot(
                        F.normalize(v_curr.unsqueeze(0), dim=-1).squeeze(0), final_vec
                    ).item()
                )
            else:
                cos_final = 1.0

            pos_b = prof.positive_quantile_buckets
            peak_toks = list(
                dict.fromkeys(
                    s.token_str for s in pos_b.get("peak_trunk", []) if s.token_str is not None
                )
            )
            sec_toks = list(
                dict.fromkeys(
                    s.token_str
                    for s in pos_b.get("secondary_branch", [])
                    if s.token_str is not None
                )
            )
            tert_toks = list(
                dict.fromkeys(
                    s.token_str
                    for s in pos_b.get("tertiary_offshoot", [])
                    if s.token_str is not None
                )
            )

            snapshots.append(
                NeuronCheckpointSnapshot(
                    checkpoint_label=label,
                    mean=round(prof.mean, 4),
                    std=round(prof.std, 4),
                    pos_pole_selectivity=round(prof.pos_pole_selectivity_score, 4),
                    neg_pole_selectivity=round(prof.neg_pole_selectivity_score, 4),
                    pos_taxonomy=prof.pos_taxonomy_class,
                    neg_taxonomy=prof.neg_taxonomy_class,
                    outgoing_weight_norm=round(wnorm, 4),
                    weight_cosine_to_final=round(cos_final, 4),
                    peak_trunk_tokens=peak_toks,
                    secondary_branch_tokens=sec_toks,
                    tertiary_offshoot_tokens=tert_toks,
                    double_negative_positive_rate=round(prof.double_negative_positive_rate, 4),
                )
            )

        if snapshots:
            initial_tokens = set(
                snapshots[0].peak_trunk_tokens
                + snapshots[0].secondary_branch_tokens
                + snapshots[0].tertiary_offshoot_tokens
            )
            final_offshoots = (
                snapshots[-1].secondary_branch_tokens + snapshots[-1].tertiary_offshoot_tokens
            )
            emerged = list(dict.fromkeys(t for t in final_offshoots if t not in initial_tokens))
            sel_delta_pos = round(
                snapshots[-1].pos_pole_selectivity - snapshots[0].pos_pole_selectivity, 4
            )
            sel_delta_neg = round(
                snapshots[-1].neg_pole_selectivity - snapshots[0].neg_pole_selectivity, 4
            )
        else:
            emerged = []
            sel_delta_pos = 0.0
            sel_delta_neg = 0.0

        return NeuronEvolutionReport(
            layer_name=layer_name,
            neuron_idx=neuron_idx,
            snapshots=snapshots,
            emerged_offshoot_tokens=emerged,
            selectivity_delta_pos=sel_delta_pos,
            selectivity_delta_neg=sel_delta_neg,
        )
