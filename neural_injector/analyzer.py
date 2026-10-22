from __future__ import annotations

import heapq
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from .discovery import LayerDiscovery
from .types import (
    ActivationSample,
    ContrastiveDiscoveryReport,
    ContrastiveNeuronMatch,
    NeuronCheckpointSnapshot,
    NeuronEvolutionReport,
    NeuronPoleProfile,
    NeuronScanSummary,
    NeuronTaxonomy,
    PoleType,
)


class StreamingLayerStats:
    """
    GPU-accelerated streaming statistics tracker for an entire layer of neurons.
    Tracks:
      - Full-distribution Welford/Pébay moments (M1, M2, M3, M4), min, max, and firing rates
      - Pole-specific conditional moments (M1+, M2+, M3+, M4+ for a > 0 and M1-, M2-, M3-, M4- for a < 0)
        so Positive and Negative poles are evaluated independently (Cammarata Section 3.1)
      - Top-k activating contexts AND Stratified Quantile Activation Buckets per pole
        ("peak_trunk" [80%..100%], "secondary_branch" [40%..80%), "tertiary_offshoot" [15%..40%))
        to capture recursive generalization offshoots (Cammarata Section 3.3).
    """

    QUANTILE_BAND_SPECS: Tuple[Tuple[str, float, float], ...] = (
        ("peak_trunk", 0.80, 1.01),
        ("secondary_branch", 0.40, 0.80),
        ("tertiary_offshoot", 0.15, 0.40),
    )

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

        # Per-neuron bounded min-heaps for top positive and negative pole activations
        # Stores tuples of (score, tie_breaker, ActivationSample)
        self._tie_breaker = 0
        self.pos_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = {
            i: [] for i in range(num_neurons)
        }
        self.neg_heaps: Dict[int, List[Tuple[float, int, ActivationSample]]] = {
            i: [] for i in range(num_neurons)
        }

        # Stratified active sample reservoirs per neuron for quantile-band bucketing
        self._reservoir_cap = max(top_k * 4, 24)
        self.pos_band_reservoirs: Dict[int, List[Tuple[float, int, ActivationSample]]] = {
            i: [] for i in range(num_neurons)
        }
        self.neg_band_reservoirs: Dict[int, List[Tuple[float, int, ActivationSample]]] = {
            i: [] for i in range(num_neurons)
        }

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

        masked_vals = torch.where(mask, magnitudes, torch.zeros_like(magnitudes))
        batch_mean = masked_vals.sum(dim=0) / safe_nb

        centered_b = torch.where(
            mask, magnitudes - batch_mean.unsqueeze(0), torch.zeros_like(magnitudes)
        )
        batch_m2 = (centered_b ** 2).sum(dim=0)
        batch_m3 = (centered_b ** 3).sum(dim=0)
        batch_m4 = (centered_b ** 4).sum(dim=0)

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

        # 1. Batch Welford / Pébay parallel update for exact full-distribution M1, M2, M3, M4 moments
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
        self.count = new_count

        # 2. Update Pole-Specific Conditional Moments (Positive Pole a > 0, Negative Pole a < 0)
        pos_mask = flat_acts > 0
        neg_mask = flat_acts < 0
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

        # 3. Update top-k positive & negative pole heaps AND stratified reservoirs
        target_indices = (
            tracked_neurons if tracked_neurons is not None else list(range(self.num_neurons))
        )
        if not target_indices:
            return

        sub_acts = flat_acts[:, target_indices]
        k_curr = min(self._reservoir_cap, batch_n)

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
            pos_res = self.pos_band_reservoirs[neuron_idx]
            neg_res = self.neg_band_reservoirs[neuron_idx]

            # Process positive activations in this batch
            for rank_i, (val, r_idx) in enumerate(
                zip(top_pos_vals_cpu[local_i], top_pos_idx_cpu[local_i])
            ):
                if val <= 0:
                    break
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                need_top = rank_i < self.top_k and (
                    len(pos_heap) < self.top_k or val > pos_heap[0][0]
                )
                need_res = len(pos_res) < self._reservoir_cap or (rank_i % 2 == 1)
                if not need_top and not need_res:
                    continue
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                )
                if need_top:
                    self._push_heap(pos_heap, val, sample, self.top_k)
                if need_res:
                    self._push_diversified_reservoir(pos_res, val, sample)

            # Process negative activations in this batch
            for rank_i, (val, r_idx) in enumerate(
                zip(top_neg_vals_cpu[local_i], top_neg_idx_cpu[local_i])
            ):
                if val >= 0:
                    break
                f_idx = valid_idx_cpu[r_idx] if valid_idx_cpu is not None else r_idx
                mag = -val
                need_top = rank_i < self.top_k and (
                    len(neg_heap) < self.top_k or mag > neg_heap[0][0]
                )
                need_res = len(neg_res) < self._reservoir_cap or (rank_i % 2 == 1)
                if not need_top and not need_res:
                    continue
                sample = self._build_sample(
                    val,
                    f_idx,
                    seq_len,
                    sample_ids,
                    tokens_per_sample,
                    contexts_per_sample,
                    spatial_hw=spatial_hw,
                )
                if need_top:
                    self._push_heap(neg_heap, mag, sample, self.top_k)
                if need_res:
                    self._push_diversified_reservoir(neg_res, mag, sample)

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
        max_len: int,
    ) -> None:
        self._tie_breaker += 1
        item = (priority, self._tie_breaker, sample)
        if len(heap) < max_len:
            heapq.heappush(heap, item)
        elif priority > heap[0][0]:
            heapq.heapreplace(heap, item)

    def _push_diversified_reservoir(
        self,
        reservoir: List[Tuple[float, int, ActivationSample]],
        magnitude: float,
        sample: ActivationSample,
    ) -> None:
        """
        Maintains a magnitude-diversified pool of active samples across the neuron's
        dynamic range so that secondary (40..80%) and tertiary (15..40%) recursive
        offshoots are retained alongside peak (80..100%) trunk activations.
        """
        self._tie_breaker += 1
        item = (magnitude, self._tie_breaker, sample)
        if len(reservoir) < self._reservoir_cap:
            reservoir.append(item)
            return
        # Replace an entry with the closest magnitude to preserve coverage across all quantile bands
        closest_idx = min(
            range(len(reservoir)), key=lambda i: abs(reservoir[i][0] - magnitude)
        )
        reservoir[closest_idx] = item

    def _build_quantile_buckets(
        self,
        neuron_idx: int,
        is_positive: bool,
        pole_max_abs: float,
    ) -> Dict[str, List[ActivationSample]]:
        """
        Partitions collected samples for a pole into Cammarata's Stratified Quantile Bands:
          - `peak_trunk`:        [0.80 * max_abs, 1.00 * max_abs]
          - `secondary_branch`:  [0.40 * max_abs, 0.80 * max_abs)
          - `tertiary_offshoot`: [0.15 * max_abs, 0.40 * max_abs)
        """
        buckets: Dict[str, List[ActivationSample]] = {
            name: [] for name, _, _ in self.QUANTILE_BAND_SPECS
        }
        if pole_max_abs <= 1e-9:
            return buckets

        heap_items = self.pos_heaps.get(neuron_idx, []) if is_positive else self.neg_heaps.get(neuron_idx, [])
        res_items = (
            self.pos_band_reservoirs.get(neuron_idx, [])
            if is_positive
            else self.neg_band_reservoirs.get(neuron_idx, [])
        )

        # Deduplicate by (sample_id, position_idx, round(activation_value, 6))
        seen = set()
        candidates: List[ActivationSample] = []
        for mag, _, s in sorted(heap_items + res_items, key=lambda x: x[0], reverse=True):
            key = (s.sample_id, s.position_idx, round(s.activation_value, 6))
            if key not in seen and mag > 0:
                seen.add(key)
                candidates.append(s)

        for s in candidates:
            ratio = abs(s.activation_value) / pole_max_abs
            for band_name, low_r, high_r in self.QUANTILE_BAND_SPECS:
                if low_r <= ratio <= high_r:
                    tagged = ActivationSample(
                        activation_value=s.activation_value,
                        sample_id=s.sample_id,
                        position_idx=s.position_idx,
                        token_str=s.token_str,
                        context_str=s.context_str,
                        quantile_band=band_name,
                    )
                    if len(buckets[band_name]) < self.top_k:
                        buckets[band_name].append(tagged)
                    break

        return buckets

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
        elif pole_selectivity >= 1.95:
            return NeuronTaxonomy.CLEAN_MONOSEMANTIC
        elif pole_selectivity >= 1.25:
            return NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE
        else:
            return NeuronTaxonomy.TRUE_POLYSEMANTIC

    def _compute_pole_selectivities_and_stds(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes conditional standard deviations and tail-selectivity scores for both
        the Positive Pole and Negative Pole across all neurons in the layer.
        To reflect both in-pole kurtosis and activation sparsity across the corpus,
        conditional 4th-moment kurtosis is boosted by inverse firing rate `(count / pole_count)^0.25`.
        """
        n_total = max(self.count, 1)

        pos_n = torch.clamp(self.pos_count.to(torch.float64), min=1.0)
        pos_var = self.pos_m2 / pos_n
        pos_std = torch.sqrt(torch.clamp(pos_var, min=1e-12))
        pos_kurt = (self.pos_m4 / pos_n) / (pos_var.pow(2) + 1e-12)
        pos_sparsity_boost = (n_total / pos_n).pow(0.25)
        pos_sel = torch.where(
            self.pos_count > 0,
            torch.clamp((pos_kurt.pow(0.25)) * pos_sparsity_boost, min=0.0, max=100.0),
            torch.zeros_like(pos_var),
        )

        neg_n = torch.clamp(self.neg_count.to(torch.float64), min=1.0)
        neg_var = self.neg_m2 / neg_n
        neg_std = torch.sqrt(torch.clamp(neg_var, min=1e-12))
        neg_kurt = (self.neg_m4 / neg_n) / (neg_var.pow(2) + 1e-12)
        neg_sparsity_boost = (n_total / neg_n).pow(0.25)
        neg_sel = torch.where(
            self.neg_count > 0,
            torch.clamp((neg_kurt.pow(0.25)) * neg_sparsity_boost, min=0.0, max=100.0),
            torch.zeros_like(neg_var),
        )

        return pos_std, pos_sel, neg_std, neg_sel

    def compute_adaptive_thresholds(
            self,
            neuron_indices: List[int],
            std_multiplier: Optional[float] = None,
            percentile: Optional[float] = None,
    ) -> List[float]:
        """
        Computes per-neuron native activation thresholds from audited statistics
        (Cammarata Section 3.4: Respecting Native Activation Ranges & Percentiles).
          - If `percentile` (in (0, 100)) is provided, estimates each neuron's native
            active magnitude percentile from its audited conditional distribution and reservoir.
          - Otherwise uses `|mean_i| + k * std_i` where `k = std_multiplier or 1.0`.
        """
        n = max(self.count, 1)
        var_vec = self.m2 / n
        std_vec = torch.sqrt(torch.clamp(var_vec, min=1e-12))
        thresholds: List[float] = []

        for idx in neuron_indices:
            if percentile is not None:
                q = min(max(float(percentile) / 100.0, 0.01), 0.99)
                # Gather observed active magnitudes from positive & negative reservoirs and heaps
                mags: List[float] = []
                for collection in (
                        self.pos_band_reservoirs.get(idx, []),
                        self.neg_band_reservoirs.get(idx, []),
                        self.pos_heaps.get(idx, []),
                        self.neg_heaps.get(idx, []),
                ):
                    for _priority, _tb, sample in collection:
                        mags.append(abs(sample.activation_value))

                if len(mags) >= 4:
                    mags_t = torch.tensor(sorted(mags), dtype=torch.float32)
                    q_idx = min(int(math.floor(q * (len(mags_t) - 1))), len(mags_t) - 1)
                    thresholds.append(float(mags_t[q_idx].item()))
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

        kurtosis = float((self.m4[neuron_idx] / n).item()) / ((var_val ** 2) + 1e-12)
        selectivity = float(min(max(kurtosis ** 0.25, 0.0), 100.0))

        pos_std_vec, pos_sel_vec, neg_std_vec, neg_sel_vec = (
            self._compute_pole_selectivities_and_stds()
        )
        pos_sel = float(pos_sel_vec[neuron_idx].item())
        neg_sel = float(neg_sel_vec[neuron_idx].item())
        pos_std = float(pos_std_vec[neuron_idx].item())
        neg_std = float(neg_std_vec[neuron_idx].item())
        pos_rate = float(self.pos_count[neuron_idx].item() / n)
        neg_rate = float(self.neg_count[neuron_idx].item() / n)

        pos_tax = self._classify_pole_taxonomy(pos_sel, pos_rate, pos_std)
        neg_tax = self._classify_pole_taxonomy(neg_sel, neg_rate, neg_std)

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
        clean_max_v = max_v if max_v != float("-inf") else 0.0
        clean_min_v = min_v if min_v != float("inf") else 0.0

        pos_buckets = self._build_quantile_buckets(
            neuron_idx, is_positive=True, pole_max_abs=max(clean_max_v, 0.0)
        )
        neg_buckets = self._build_quantile_buckets(
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
            neg_pole_mean=-float(self.neg_mean[neuron_idx].item()),
            neg_pole_std=neg_std,
            neg_pole_selectivity_score=round(neg_sel, 3),
            neg_taxonomy_class=neg_tax,
            positive_quantile_buckets=pos_buckets,
            negative_quantile_buckets=neg_buckets,
        )

    @torch.no_grad()
    def scan_layer_cleanliness(
        self,
        down_proj_module: Optional[nn.Module] = None,
    ) -> Tuple[Dict[str, int], Dict[str, float], List[NeuronScanSummary]]:
        """
        Vectorized GPU pass over all tracked moments in the layer to classify every neuron
        AND every individual neuron pole into the Cammarata (2026) cleanliness taxonomy.
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
        pos_std_vec, pos_sel_vec, neg_std_vec, neg_sel_vec = (
            self._compute_pole_selectivities_and_stds()
        )

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
            tot_rate = pos_r + neg_r

            pos_sel_i = float(pos_sel_cpu[idx])
            neg_sel_i = float(neg_sel_cpu[idx])
            pos_tax = self._classify_pole_taxonomy(pos_sel_i, pos_r, float(pos_std_cpu[idx]), wnorm)
            neg_tax = self._classify_pole_taxonomy(neg_sel_i, neg_r, float(neg_std_cpu[idx]), wnorm)

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


class ContrastivePoleFinder:
    """
    Contrastive Pole Discovery Engine (Section 5.3):
    Ranks all native-basis neurons across one or more layers by Welch's Bipolar
    Separation t-statistic between two contrastive input sets (D+ and D-):
        t_i = (mu+(a_i) - mu-(a_i)) / sqrt(var+(a_i)/N+ + var-(a_i)/N- + eps)
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
            for lname in layer_names:
                mod, hook_target, num_n = LayerDiscovery.resolve_module(model, lname)
                st = StreamingLayerStats(lname, num_n, top_k=1, device=device)
                trackers[lname] = st
                if hook_target == "input":
                    handles.append(
                        mod.register_forward_pre_hook(
                            lambda _m, inp, s=st: (
                                s.update(inp[0].detach())
                                if inp and isinstance(inp[0], torch.Tensor)
                                else None
                            )
                        )
                    )
                else:
                    handles.append(
                        mod.register_forward_hook(
                            lambda _m, _inp, out, s=st: s.update(
                                (out[0] if isinstance(out, tuple) else out).detach()
                            )
                        )
                    )
            try:
                for b in batches:
                    run_fwd(model, b)
            finally:
                for h in handles:
                    h.remove()
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

        for lname in layer_names:
            ps = pos_stats[lname]
            ns = neg_stats[lname]
            n_pos = max(ps.count, 1)
            n_neg = max(ns.count, 1)
            pos_tokens_count = max(pos_tokens_count, ps.count)
            neg_tokens_count = max(neg_tokens_count, ns.count)

            var_pos = ps.m2 / n_pos
            var_neg = ns.m2 / n_neg
            se_vec = torch.sqrt((var_pos / n_pos) + (var_neg / n_neg) + 1e-9)
            t_vec = (ps.mean - ns.mean) / se_vec

            std_pos = torch.sqrt(torch.clamp(var_pos, min=0.0))
            std_neg = torch.sqrt(torch.clamp(var_neg, min=0.0))

            t_cpu = t_vec.cpu().tolist()
            mp_cpu = ps.mean.cpu().tolist()
            mn_cpu = ns.mean.cpu().tolist()
            sp_cpu = std_pos.cpu().tolist()
            sn_cpu = std_neg.cpu().tolist()

            for idx in range(ps.num_neurons):
                t_val = float(t_cpu[idx])
                mp = float(mp_cpu[idx])
                mn = float(mn_cpu[idx])
                sign_flip = (mp > 0.0 and mn < 0.0) or (mp < 0.0 and mn > 0.0)
                rec_pole = PoleType.POSITIVE if t_val >= 0.0 else PoleType.NEGATIVE
                matches.append(
                    ContrastiveNeuronMatch(
                        layer_name=lname,
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

        matches.sort(key=lambda m: (m.bipolar_sign_flip, m.abs_t_statistic), reverse=True)
        return ContrastiveDiscoveryReport(
            layers_scanned=list(layer_names),
            pos_tokens_count=pos_tokens_count,
            neg_tokens_count=neg_tokens_count,
            top_matches=matches[:top_k],
        )


class CheckpointEvolutionTracker:
    """
    Cross-Checkpoint 'Recursive Growth' Tracker (Section 5.4):
    Inspired by Cammarata's observation of neurons sprouting recursive offshoots across
    training epochs, `CheckpointEvolutionTracker` audits a specific `(layer_name, neuron_idx)`
    across a sequence of `(checkpoint_label, model)` pairs on a shared evaluation dataset,
    tracking how pole selectivity, outgoing residual write vectors, and stratified quantile
    buckets (`peak_trunk`, `secondary_branch`, `tertiary_offshoot`) evolve.
    """

    @staticmethod
    @torch.no_grad()
    def track_neuron_evolution(
            checkpoints: List[Tuple[str, nn.Module]],
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

        # First pass: extract outgoing write vectors for cosine similarity to the final checkpoint
        for _label, ckpt_model in checkpoints:
            mod, _, _ = LayerDiscovery.resolve_module(ckpt_model, layer_name)
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                w = mod.weight.detach().float()
                if w.ndim == 3:
                    v = w[:, :, neuron_idx].mean(dim=0)
                elif mod.__class__.__name__ == "Conv1D":
                    v = w[neuron_idx, :]
                else:
                    v = w[:, neuron_idx]
                write_vecs.append(v.cpu())
            else:
                write_vecs.append(torch.ones(1))

        final_vec = (
            F.normalize(write_vecs[-1].unsqueeze(0), dim=-1).squeeze(0)
            if write_vecs
            else torch.ones(1)
        )

        for (label, ckpt_model), v_curr in zip(checkpoints, write_vecs):
            inj = NeuralInjector(ckpt_model)
            inj.audit_layers(
                layer_names=[layer_name],
                dataloader_or_batches=batch_list,
                forward_fn=forward_fn,
                top_k=top_k,
                tracked_neurons=[neuron_idx],
            )
            prof = inj.get_neuron_profile(layer_name, neuron_idx)
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