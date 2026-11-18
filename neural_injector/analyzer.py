from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discovery import LayerDiscovery
from .types import (
    ActivationBand,
    ActivationSample,
    ContrastiveDiscoveryReport,
    ContrastiveNeuronMatch,
    DistributionMoments,
    GATED_REGIMES,
    NeuronCheckpointSnapshot,
    NeuronEvolutionReport,
    NeuronPoleProfile,
    NeuronScanSummary,
    PoleType,
    SignRegime,
    UNGATED_REGIMES,
)

# Global token codes: code = internal_sample_index * POS_BASE + position
POS_BASE = 1 << 24
# Upper bound on the number of [tokens x neurons] elements materialised at once in one update chunk
_CHUNK_ELEMS = 1 << 23

# Default ring edges as fractions of the pole extreme: successive halvings ("octave rings"),
# i.e. the strong half, the next quarter, ... down to everything below 1/32 of the extreme.
DEFAULT_BAND_EDGES: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0)


def _excess_kurtosis(n: torch.Tensor, m2: torch.Tensor, m4: torch.Tensor) -> torch.Tensor:
    safe = m2 > 0
    out = torch.zeros_like(m2)
    out[safe] = n[safe] * m4[safe] / (m2[safe] ** 2) - 3.0
    return out


def _skewness(n: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
    safe = m2 > 0
    out = torch.zeros_like(m2)
    out[safe] = torch.sqrt(n[safe]) * m3[safe] / (m2[safe] ** 1.5)
    return out


def is_punctuation_token(token: Optional[str]) -> bool:
    """True if the token, with BPE/SentencePiece whitespace markers removed, consists only of
    Unicode punctuation (P*) or symbol (S*) characters."""
    if not token:
        return False
    stripped = token.replace("Ġ", "").replace("▁", "").replace("Ċ", "").strip()
    if not stripped:
        return False
    return all(unicodedata.category(ch)[0] in ("P", "S") for ch in stripped)


class StreamingLayerStats:
    """
    Exact, vectorised streaming record of every neuron in one layer.

    Per neuron it keeps
      * exact Pébay moments (M1..M4) of all activations, of each pole, and of each sign regime
        (gated MLPs: ++, --, +-, -+; otherwise + and -),
      * exact top-k activations per sign regime (so '--' echoes never enter the '++' list, and the
        raw mixed list is their exact merge),
      * an exact log2-magnitude histogram per sign regime (resolution `bins_per_octave`), from which
        fraction-of-extreme band counts and within-pole percentiles are read,
      * a uniform random sample of activations inside every magnitude octave of each pole
        (bottom-k priority sampling, mergeable across batches), used to show examples from every
        activation band rather than only the top of the list.
    Token strings and contexts are stored once per sample in a registry that is compacted
    to the samples still referenced.
    """

    def __init__(
        self,
        layer_name: str,
        num_neurons: int,
        gated: bool,
        top_k: int = 16,
        device: str = "cpu",
        bins_per_octave: int = 4,
        min_exponent: int = -20,
        max_exponent: int = 20,
        samples_per_octave: int = 3,
        seed: int = 0,
        window_tokens_before: int = 12,
        window_tokens_after: int = 4,
    ):
        self.layer_name = layer_name
        self.num_neurons = int(num_neurons)
        self.gated = bool(gated)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self.bins_per_octave = int(bins_per_octave)
        self.min_exponent = int(min_exponent)
        self.max_exponent = int(max_exponent)
        self.samples_per_octave = int(samples_per_octave)
        self.seed = int(seed)
        self.window_tokens_before = int(window_tokens_before)
        self.window_tokens_after = int(window_tokens_after)

        self.regimes: Tuple[SignRegime, ...] = GATED_REGIMES if self.gated else UNGATED_REGIMES
        R = len(self.regimes)
        D = self.num_neurons
        dev = self.device
        f64 = torch.float64

        self.count = 0
        self.mean = torch.zeros(D, dtype=f64, device=dev)
        self.m2 = torch.zeros(D, dtype=f64, device=dev)
        self.m3 = torch.zeros(D, dtype=f64, device=dev)
        self.m4 = torch.zeros(D, dtype=f64, device=dev)
        self.max_val = torch.full((D,), float("-inf"), dtype=f64, device=dev)
        self.min_val = torch.full((D,), float("inf"), dtype=f64, device=dev)

        # pole moments: index 0 = positive (values a), 1 = negative (magnitudes |a|)
        self.pole_count = torch.zeros(2, D, dtype=torch.long, device=dev)
        self.pole_mean = torch.zeros(2, D, dtype=f64, device=dev)
        self.pole_m2 = torch.zeros(2, D, dtype=f64, device=dev)
        self.pole_m3 = torch.zeros(2, D, dtype=f64, device=dev)
        self.pole_m4 = torch.zeros(2, D, dtype=f64, device=dev)
        # regime moments on magnitudes
        self.reg_count = torch.zeros(R, D, dtype=torch.long, device=dev)
        self.reg_mean = torch.zeros(R, D, dtype=f64, device=dev)
        self.reg_m2 = torch.zeros(R, D, dtype=f64, device=dev)
        self.reg_m3 = torch.zeros(R, D, dtype=f64, device=dev)
        self.reg_m4 = torch.zeros(R, D, dtype=f64, device=dev)

        K = self.top_k
        self.top_vals = torch.full((R, K, D), float("-inf"), dtype=torch.float32, device=dev)
        self.top_codes = torch.full((R, K, D), -1, dtype=torch.long, device=dev)

        self.n_bins = (self.max_exponent - self.min_exponent) * self.bins_per_octave + 2
        self.hist = torch.zeros(R, D, self.n_bins, dtype=torch.int32, device=dev)

        self.n_octaves = (self.max_exponent - self.min_exponent) + 2
        C = self.samples_per_octave
        self.res_keys = torch.full((2, D, self.n_octaves, C), float("inf"), dtype=torch.float32, device=dev)
        self.res_vals = torch.zeros(2, D, self.n_octaves, C, dtype=torch.float32, device=dev)
        self.res_codes = torch.full((2, D, self.n_octaves, C), -1, dtype=torch.long, device=dev)
        self.res_regime = torch.full((2, D, self.n_octaves, C), -1, dtype=torch.int8, device=dev)

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._next_internal = 0
        # internal sample index -> (user sample id, tokens or None, context or None, spatial (h, w) or None)
        self.registry: Dict[int, Tuple[Any, Optional[List[str]], Optional[str], Optional[Tuple[int, int]]]] = {}

    # ------------------------------------------------------------------
    # Moment updates
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_moments(
        mask: torch.Tensor,
        values: torch.Tensor,
        count_vec: torch.Tensor,
        mean_vec: torch.Tensor,
        m2_vec: torch.Tensor,
        m3_vec: torch.Tensor,
        m4_vec: torch.Tensor,
    ) -> None:
        """Pébay pairwise merge of the masked batch moments into the running moments (in place)."""
        n_b_long = mask.sum(dim=0)
        if not bool((n_b_long > 0).any()):
            return
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        n_a = count_vec.to(torch.float64)
        n_b = n_b_long.to(torch.float64)
        n = n_a + n_b
        safe_nb = torch.clamp(n_b, min=1.0)
        safe_n = torch.clamp(n, min=1.0)

        mv = torch.where(mask, values, zero)
        b_mean = mv.sum(dim=0, dtype=torch.float64) / safe_nb
        c = torch.where(mask, values - b_mean.to(values.dtype).unsqueeze(0), zero)
        c2 = c * c
        b_m2 = c2.sum(dim=0, dtype=torch.float64)
        b_m3 = (c2 * c).sum(dim=0, dtype=torch.float64)
        b_m4 = (c2 * c2).sum(dim=0, dtype=torch.float64)

        delta = b_mean - mean_vec
        d_n = delta / safe_n
        d_n2 = d_n * d_n
        prod = n_a * n_b
        new_m4 = (
            m4_vec + b_m4
            + prod * (n_a * n_a - prod + n_b * n_b) * delta * d_n * d_n2
            + 6.0 * (n_a * n_a * b_m2 + n_b * n_b * m2_vec) * d_n2
            + 4.0 * (n_a * b_m3 - n_b * m3_vec) * d_n
        )
        new_m3 = m3_vec + b_m3 + prod * (n_a - n_b) * delta * d_n2 + 3.0 * (n_a * b_m2 - n_b * m2_vec) * d_n
        new_m2 = m2_vec + b_m2 + delta * delta * prod / safe_n
        new_mean = mean_vec + delta * n_b / safe_n

        active = n_b_long > 0
        mean_vec.copy_(torch.where(active, new_mean, mean_vec))
        m2_vec.copy_(torch.where(active, new_m2, m2_vec))
        m3_vec.copy_(torch.where(active, new_m3, m3_vec))
        m4_vec.copy_(torch.where(active, new_m4, m4_vec))
        count_vec.add_(n_b_long)

    # ------------------------------------------------------------------
    # Binning helpers
    # ------------------------------------------------------------------
    def _bin_index(self, mags: torch.Tensor) -> torch.Tensor:
        lg = torch.log2(torch.clamp(mags, min=1e-38))
        b = torch.floor((lg - self.min_exponent) * self.bins_per_octave).to(torch.long) + 1
        return torch.clamp(b, 0, self.n_bins - 1)

    def _octave_index(self, mags: torch.Tensor) -> torch.Tensor:
        lg = torch.log2(torch.clamp(mags, min=1e-38))
        o = torch.floor(lg).to(torch.long) - self.min_exponent + 1
        return torch.clamp(o, 0, self.n_octaves - 1)

    def _bin_lower_value(self, b: int) -> float:
        if b <= 0:
            return 0.0
        return 2.0 ** (self.min_exponent + (b - 1) / self.bins_per_octave)

    def _value_to_edge(self, v: float) -> int:
        """Nearest histogram bin edge index for a magnitude (edge e = lower edge of bin e)."""
        if v <= 0.0:
            return 0
        e = int(round((math.log2(v) - self.min_exponent) * self.bins_per_octave)) + 1
        return max(1, min(self.n_bins - 1, e))

    def _regime_codes_for_pole(self, pole: PoleType) -> List[int]:
        return [i for i, r in enumerate(self.regimes) if r.pole == pole]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        activations: torch.Tensor,
        sample_ids: Optional[Sequence[Any]] = None,
        tokens_per_sample: Optional[Sequence[Optional[List[str]]]] = None,
        contexts_per_sample: Optional[Sequence[Optional[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        gate_activations: Optional[torch.Tensor] = None,
        routed_token_indices: Optional[torch.Tensor] = None,
        token_grid: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Adds one batch.
          activations: [B, S, D], [B, D], [B, D, H, W] (conv) or [N_routed, D] (MoE expert, with
                       `routed_token_indices` giving each row's flat index into the [B*S] grid
                       described by `token_grid=(B, S)`).
          gate_activations: gate pre-activations g of the same shape (required when `gated`).
        """
        if self.gated and gate_activations is None:
            raise ValueError(
                f"Layer '{self.layer_name}' is tracked as gated but no gate activations were supplied."
            )

        spatial_hw: Optional[Tuple[int, int]] = None
        if activations.ndim == 4:
            b_dim, c_dim, h_dim, w_dim = activations.shape
            flat = activations.permute(0, 2, 3, 1).reshape(-1, c_dim)
            gate_flat = (
                gate_activations.permute(0, 2, 3, 1).reshape(-1, c_dim) if gate_activations is not None else None
            )
            n_batch, seq_len = b_dim, h_dim * w_dim
            spatial_hw = (h_dim, w_dim)
        elif activations.ndim == 3:
            n_batch, seq_len, d_dim = activations.shape
            flat = activations.reshape(-1, d_dim)
            gate_flat = gate_activations.reshape(-1, d_dim) if gate_activations is not None else None
        elif activations.ndim == 2:
            flat = activations
            gate_flat = gate_activations
            if routed_token_indices is not None:
                if token_grid is None:
                    raise ValueError("`token_grid=(B, S)` is required with `routed_token_indices`.")
                n_batch, seq_len = token_grid
            else:
                n_batch, seq_len = activations.shape[0], 1
        else:
            raise ValueError(f"Unsupported activation shape {tuple(activations.shape)}")

        if flat.shape[-1] != self.num_neurons:
            raise ValueError(
                f"Layer '{self.layer_name}': expected {self.num_neurons} neurons, got {flat.shape[-1]}"
            )
        work_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32
        flat = flat.to(device=self.device, dtype=work_dtype)
        if gate_flat is not None:
            gate_flat = gate_flat.to(device=self.device, dtype=work_dtype)
            if gate_flat.shape != flat.shape:
                raise ValueError("gate activations do not match activation shape")

        # Flat [B*S] grid index of every row
        if routed_token_indices is not None:
            grid_idx = routed_token_indices.reshape(-1).to(device=self.device, dtype=torch.long)
        else:
            grid_idx = torch.arange(flat.shape[0], device=self.device, dtype=torch.long)
            if attention_mask is not None and spatial_hw is None:
                m = attention_mask.reshape(-1).to(device=self.device, dtype=torch.bool)
                if m.numel() != flat.shape[0]:
                    raise ValueError("attention_mask does not match the activation grid")
                keep = torch.nonzero(m, as_tuple=False).reshape(-1)
                flat = flat[keep]
                if gate_flat is not None:
                    gate_flat = gate_flat[keep]
                grid_idx = keep

        # Register samples of this batch
        base = self._next_internal
        self._next_internal += n_batch
        for b in range(n_batch):
            sid = sample_ids[b] if sample_ids is not None and b < len(sample_ids) else base + b
            toks = list(tokens_per_sample[b]) if tokens_per_sample is not None and b < len(tokens_per_sample) and tokens_per_sample[b] is not None else None
            ctx = contexts_per_sample[b] if contexts_per_sample is not None and b < len(contexts_per_sample) else None
            self.registry[base + b] = (sid, toks, ctx, spatial_hw)

        N = flat.shape[0]
        if N == 0:
            return
        codes_rows = (base + grid_idx // seq_len) * POS_BASE + (grid_idx % seq_len)

        # 1. full-distribution moments
        n_a = self.count
        n_b = N
        n = n_a + n_b
        b_mean = flat.mean(dim=0, dtype=torch.float64)
        c = flat - b_mean.to(flat.dtype).unsqueeze(0)
        c2 = c * c
        b_m2 = c2.sum(dim=0, dtype=torch.float64)
        b_m3 = (c2 * c).sum(dim=0, dtype=torch.float64)
        b_m4 = (c2 * c2).sum(dim=0, dtype=torch.float64)
        if n_a == 0:
            self.mean, self.m2, self.m3, self.m4 = b_mean, b_m2, b_m3, b_m4
        else:
            delta = b_mean - self.mean
            d_n = delta / n
            d_n2 = d_n * d_n
            prod = float(n_a) * float(n_b)
            self.m4 = (
                self.m4 + b_m4
                + prod * (n_a * n_a - prod + n_b * n_b) * delta * d_n * d_n2
                + 6.0 * (n_a * n_a * b_m2 + n_b * n_b * self.m2) * d_n2
                + 4.0 * (n_a * b_m3 - n_b * self.m3) * d_n
            )
            self.m3 = self.m3 + b_m3 + prod * (n_a - n_b) * delta * d_n2 + 3.0 * (n_a * b_m2 - n_b * self.m2) * d_n
            self.m2 = self.m2 + b_m2 + delta * delta * prod / n
            self.mean = self.mean + delta * (n_b / n)
        self.count = n
        self.max_val = torch.maximum(self.max_val, flat.max(dim=0).values.to(torch.float64))
        self.min_val = torch.minimum(self.min_val, flat.min(dim=0).values.to(torch.float64))

        # 2. sign regime of every entry
        pos = flat > 0
        neg = flat < 0
        if self.gated:
            g_neg = gate_flat < 0
            regime = torch.where(
                pos,
                torch.where(g_neg, 1, 0),
                torch.where(neg, torch.where(g_neg, 3, 2), -1),
            ).to(torch.int8)
        else:
            regime = torch.where(pos, 0, torch.where(neg, 1, -1)).to(torch.int8)
        mags = flat.abs()

        # 3. pole and regime moments
        self._merge_moments(pos, flat, self.pole_count[0], self.pole_mean[0], self.pole_m2[0], self.pole_m3[0], self.pole_m4[0])
        self._merge_moments(neg, mags, self.pole_count[1], self.pole_mean[1], self.pole_m2[1], self.pole_m3[1], self.pole_m4[1])
        for r in range(len(self.regimes)):
            self._merge_moments(regime == r, mags, self.reg_count[r], self.reg_mean[r], self.reg_m2[r], self.reg_m3[r], self.reg_m4[r])

        # 4-6. histogram, per-regime top-k and octave reservoirs, chunked over neurons
        mags32 = mags.to(torch.float32)
        chunk = max(1, _CHUNK_ELEMS // max(N, 1))
        for c0 in range(0, self.num_neurons, chunk):
            c1 = min(self.num_neurons, c0 + chunk)
            self._update_chunk(mags32[:, c0:c1], regime[:, c0:c1], codes_rows, c0, c1)

        self._maybe_compact_registry()

    def _update_chunk(
        self,
        mags: torch.Tensor,
        regime: torch.Tensor,
        codes_rows: torch.Tensor,
        c0: int,
        c1: int,
    ) -> None:
        N, Dc = mags.shape
        R = len(self.regimes)
        dev = self.device
        valid = regime >= 0

        # histogram (exact)
        b_idx = self._bin_index(mags)
        col = torch.arange(Dc, device=dev).unsqueeze(0).expand(N, Dc)
        flat_bin = (regime.to(torch.long) * Dc + col) * self.n_bins + b_idx
        counts = torch.bincount(flat_bin[valid], minlength=R * Dc * self.n_bins)
        self.hist[:, c0:c1, :] += counts.view(R, Dc, self.n_bins).to(torch.int32)

        # per-regime exact top-k (magnitudes)
        K = self.top_k
        k_new = min(K, N)
        neg_inf = torch.tensor(float("-inf"), dtype=mags.dtype, device=dev)
        for r in range(R):
            masked = torch.where(regime == r, mags, neg_inf)
            v_new, i_new = torch.topk(masked, k=k_new, dim=0)
            code_new = codes_rows[i_new]
            code_new = torch.where(torch.isfinite(v_new), code_new, torch.full_like(code_new, -1))
            v_cat = torch.cat([self.top_vals[r, :, c0:c1], v_new], dim=0)
            c_cat = torch.cat([self.top_codes[r, :, c0:c1], code_new], dim=0)
            v_top, i_top = torch.topk(v_cat, k=K, dim=0)
            self.top_vals[r, :, c0:c1] = v_top
            self.top_codes[r, :, c0:c1] = torch.gather(c_cat, 0, i_top)

        # uniform bottom-k reservoir per (pole, octave)
        C = self.samples_per_octave
        if C <= 0:
            return
        octv = self._octave_index(mags)
        keys = torch.rand((N, Dc), generator=self._generator).to(dev)
        bounds_lo = torch.arange(self.n_octaves, device=dev, dtype=torch.float32)
        bounds_hi = bounds_lo + 1.0
        for p, pole in enumerate((PoleType.POSITIVE, PoleType.NEGATIVE)):
            pole_codes = self._regime_codes_for_pole(pole)
            pole_mask = torch.zeros_like(valid)
            for rc in pole_codes:
                pole_mask |= regime == rc
            if not bool(pole_mask.any()):
                continue
            composite = torch.where(pole_mask, octv.to(torch.float32) + keys, torch.full_like(keys, float("inf")))
            sorted_c, perm = torch.sort(composite, dim=0)
            sc_t = sorted_c.T.contiguous()                       # [Dc, N]
            perm_t = perm.T.contiguous()
            lo = torch.searchsorted(sc_t, bounds_lo.unsqueeze(0).expand(Dc, -1).contiguous())  # [Dc, n_oct]
            hi = torch.searchsorted(sc_t, bounds_hi.unsqueeze(0).expand(Dc, -1).contiguous())
            cand_keys = torch.full((Dc, self.n_octaves, C), float("inf"), dtype=torch.float32, device=dev)
            cand_vals = torch.zeros((Dc, self.n_octaves, C), dtype=torch.float32, device=dev)
            cand_codes = torch.full((Dc, self.n_octaves, C), -1, dtype=torch.long, device=dev)
            cand_reg = torch.full((Dc, self.n_octaves, C), -1, dtype=torch.int8, device=dev)
            mags_t = mags.T
            reg_t = regime.T
            for j in range(C):
                pos_j = lo + j
                ok = pos_j < hi
                pos_c = torch.clamp(pos_j, max=N - 1)
                rows = torch.gather(perm_t, 1, pos_c)                        # [Dc, n_oct]
                kv = torch.gather(sc_t, 1, pos_c) - bounds_lo.unsqueeze(0)
                cand_keys[:, :, j] = torch.where(ok, kv, torch.full_like(kv, float("inf")))
                cand_vals[:, :, j] = torch.gather(mags_t, 1, rows)
                cand_codes[:, :, j] = torch.where(ok, codes_rows[rows], torch.full_like(rows, -1))
                cand_reg[:, :, j] = torch.where(ok, torch.gather(reg_t, 1, rows), torch.full_like(rows, -1, dtype=torch.int8))
            all_keys = torch.cat([self.res_keys[p, c0:c1], cand_keys], dim=-1)
            all_vals = torch.cat([self.res_vals[p, c0:c1], cand_vals], dim=-1)
            all_codes = torch.cat([self.res_codes[p, c0:c1], cand_codes], dim=-1)
            all_reg = torch.cat([self.res_regime[p, c0:c1], cand_reg], dim=-1)
            kk, ki = torch.topk(all_keys, k=C, dim=-1, largest=False)
            self.res_keys[p, c0:c1] = kk
            self.res_vals[p, c0:c1] = torch.gather(all_vals, -1, ki)
            self.res_codes[p, c0:c1] = torch.gather(all_codes, -1, ki)
            self.res_regime[p, c0:c1] = torch.gather(all_reg, -1, ki)

    def _referenced_internal_indices(self) -> torch.Tensor:
        codes = torch.cat([self.top_codes.reshape(-1), self.res_codes.reshape(-1)])
        codes = codes[codes >= 0]
        return torch.unique(codes // POS_BASE)

    def _maybe_compact_registry(self) -> None:
        # Memory management only: drop samples no stored example points to any more.
        if len(self.registry) <= 4 * max(self.top_k, 1) * max(len(self.regimes), 1) + 1024:
            return
        ref = set(int(i) for i in self._referenced_internal_indices().tolist())
        if len(self.registry) > 2 * len(ref):
            self.registry = {k: v for k, v in self.registry.items() if k in ref}

    # ------------------------------------------------------------------
    # Sample decoding
    # ------------------------------------------------------------------
    def _decode_sample(self, code: int, value: float, regime_code: int, pole_extreme: float) -> ActivationSample:
        internal, pos = divmod(int(code), POS_BASE)
        sid, toks, ctx, spatial = self.registry.get(internal, (internal, None, None, None))
        token = nxt = window = None
        if toks is not None and pos < len(toks):
            token = toks[pos]
            nxt = toks[pos + 1] if pos + 1 < len(toks) else None
            pre = "".join(toks[max(0, pos - self.window_tokens_before): pos])
            post = "".join(toks[pos + 1: pos + 1 + self.window_tokens_after])
            window = f"{pre}[[{token}]]{post}"
        elif spatial is not None:
            h, w = spatial
            token = f"patch_({pos // w},{pos % w})"
            window = token
        regime = self.regimes[regime_code].value if 0 <= regime_code < len(self.regimes) else None
        frac = abs(value) / abs(pole_extreme) if pole_extreme not in (0.0, float("inf"), float("-inf")) else None
        return ActivationSample(
            activation_value=float(value),
            sample_id=sid,
            position_idx=pos,
            token_str=token,
            context_str=ctx,
            next_token_str=nxt,
            window_str=window,
            sign_regime=regime,
            fraction_of_pole_extreme=round(frac, 6) if frac is not None else None,
        )

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    def pole_histogram(self, neuron_idx: int, pole: PoleType) -> torch.Tensor:
        codes = self._regime_codes_for_pole(pole)
        return self.hist[codes, neuron_idx, :].to(torch.long).sum(dim=0)

    def pole_extreme_magnitude(self, neuron_idx: int, pole: PoleType) -> float:
        if pole == PoleType.POSITIVE:
            v = float(self.max_val[neuron_idx].item())
            return v if v > 0 else 0.0
        v = float(self.min_val[neuron_idx].item())
        return -v if v < 0 else 0.0

    def percentile_within_pole(self, neuron_idx: int, pole: PoleType, percentile: float) -> float:
        """
        Magnitude below which `percentile`% of this pole's activations fall, read from the exact
        log-magnitude histogram with log-linear interpolation inside the bin (relative resolution
        2^(1/bins_per_octave)).
        """
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be in (0, 100)")
        h = self.pole_histogram(neuron_idx, pole)
        total = int(h.sum().item())
        if total == 0:
            return 0.0
        target = percentile / 100.0 * total
        cum = torch.cumsum(h, dim=0)
        b = int(torch.searchsorted(cum, torch.tensor(target, dtype=cum.dtype)).item())
        b = min(b, self.n_bins - 1)
        prev = int(cum[b - 1].item()) if b > 0 else 0
        in_bin = int(h[b].item())
        frac = (target - prev) / in_bin if in_bin > 0 else 0.0
        extreme = self.pole_extreme_magnitude(neuron_idx, pole)
        if b == 0:
            return frac * min(2.0 ** self.min_exponent, extreme)
        if b == self.n_bins - 1:
            lo = self._bin_lower_value(b)
            return lo + frac * max(extreme - lo, 0.0)
        lo_exp = self.min_exponent + (b - 1) / self.bins_per_octave
        return min(2.0 ** (lo_exp + frac / self.bins_per_octave), extreme)

    def compute_thresholds(
        self,
        neuron_indices: Sequence[int],
        pole: PoleType,
        fraction_of_extreme: Optional[float] = None,
        percentile: Optional[float] = None,
        std_multiplier: Optional[float] = None,
    ) -> List[float]:
        """
        Per-neuron magnitude thresholds for one pole, from exactly one rule:
          * `fraction_of_extreme`: f * |corpus extreme of the pole| (the paper's preferred view:
            the neuron's own raw shape, 'fraction of its max'),
          * `percentile`: within-pole percentile of |a|,
          * `std_multiplier`: k * std of the pole's magnitudes above zero.
        """
        rules = [x is not None for x in (fraction_of_extreme, percentile, std_multiplier)]
        if sum(rules) != 1:
            raise ValueError("Specify exactly one of fraction_of_extreme, percentile, std_multiplier.")
        if pole == PoleType.BOTH:
            raise ValueError("Thresholds are per pole; call once for POSITIVE and once for NEGATIVE.")
        p = 0 if pole == PoleType.POSITIVE else 1
        out: List[float] = []
        for idx in neuron_indices:
            if fraction_of_extreme is not None:
                out.append(float(fraction_of_extreme) * self.pole_extreme_magnitude(idx, pole))
            elif percentile is not None:
                out.append(self.percentile_within_pole(idx, pole, float(percentile)))
            else:
                n = max(int(self.pole_count[p, idx].item()), 1)
                std = math.sqrt(max(float(self.pole_m2[p, idx].item()) / n, 0.0))
                out.append(float(std_multiplier) * std)
        return out

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    def _moments(self, count: torch.Tensor, mean: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor, m4: torch.Tensor, idx: int, extreme: float) -> DistributionMoments:
        n = int(count[idx].item())
        if n == 0:
            return DistributionMoments(count=0, rate=0.0, extreme=extreme)
        nf = torch.tensor([float(n)], dtype=torch.float64)
        m2i = m2[idx:idx + 1].cpu()
        return DistributionMoments(
            count=n,
            rate=n / max(self.count, 1),
            mean=float(mean[idx].item()),
            std=math.sqrt(max(float(m2[idx].item()) / n, 0.0)),
            skewness=float(_skewness(nf, m2i, m3[idx:idx + 1].cpu())[0].item()),
            excess_kurtosis=float(_excess_kurtosis(nf, m2i, m4[idx:idx + 1].cpu())[0].item()),
            extreme=extreme,
        )

    def _regime_top_samples(self, neuron_idx: int, r: int, pole_extreme: float) -> List[ActivationSample]:
        sign = 1.0 if self.regimes[r].pole == PoleType.POSITIVE else -1.0
        vals = self.top_vals[r, :, neuron_idx].cpu().tolist()
        codes = self.top_codes[r, :, neuron_idx].cpu().tolist()
        return [
            self._decode_sample(c, sign * v, r, pole_extreme)
            for v, c in zip(vals, codes)
            if c >= 0 and math.isfinite(v)
        ]

    def activation_bands(
        self,
        neuron_idx: int,
        pole: PoleType,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
        samples_per_band: int = 4,
    ) -> List[ActivationBand]:
        """
        Rings of one pole as fractions of its corpus extreme. `band_edges` must be descending from
        1.0 to 0.0. Counts are exact over the snapped edges; samples are drawn uniformly from inside
        each band (octave reservoirs reweighted by octave counts).
        """
        edges = list(band_edges)
        if edges[0] != 1.0 or edges[-1] != 0.0 or any(a <= b for a, b in zip(edges, edges[1:])):
            raise ValueError("band_edges must descend strictly from 1.0 to 0.0")
        extreme = self.pole_extreme_magnitude(neuron_idx, pole)
        if extreme <= 0.0:
            return []
        p = 0 if pole == PoleType.POSITIVE else 1
        sign = 1.0 if pole == PoleType.POSITIVE else -1.0
        reg_codes = self._regime_codes_for_pole(pole)
        hist_r = {rc: self.hist[rc, neuron_idx, :].to(torch.long) for rc in reg_codes}
        pole_hist = sum(hist_r.values())

        keys = self.res_keys[p, neuron_idx].reshape(-1)
        vals = self.res_vals[p, neuron_idx].reshape(-1)
        codes = self.res_codes[p, neuron_idx].reshape(-1)
        regs = self.res_regime[p, neuron_idx].reshape(-1)
        finite = torch.isfinite(keys) & (codes >= 0)
        vals, codes, regs = vals[finite], codes[finite], regs[finite]
        octs = self._octave_index(vals)
        # octave population (exact) and candidates per octave -> inverse-inclusion weights
        oct_pop = torch.zeros(self.n_octaves, dtype=torch.float64)
        for b in range(self.n_bins):
            c = int(pole_hist[b].item())
            if c:
                lo_v = self._bin_lower_value(b) if b > 0 else 0.0
                o = int(self._octave_index(torch.tensor([max(lo_v, 1e-38)]))[0].item()) if b > 0 else 0
                oct_pop[o] += c
        cand_per_oct = torch.bincount(octs, minlength=self.n_octaves).to(torch.float64)

        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + neuron_idx * 2 + p)
        bands: List[ActivationBand] = []
        for hi_f, lo_f in zip(edges, edges[1:]):
            e_hi = self.n_bins if hi_f == 1.0 else self._value_to_edge(hi_f * extreme)
            e_lo = 0 if lo_f == 0.0 else self._value_to_edge(lo_f * extreme)
            v_hi = extreme if hi_f == 1.0 else self._bin_lower_value(e_hi)
            v_lo = 0.0 if lo_f == 0.0 else self._bin_lower_value(e_lo)
            reg_counts = {self.regimes[rc].value: int(hist_r[rc][e_lo:e_hi].sum().item()) for rc in reg_codes}
            count = sum(reg_counts.values())
            in_band = (vals >= v_lo) & ((vals < v_hi) if hi_f != 1.0 else torch.ones_like(vals, dtype=torch.bool))
            idxs = torch.nonzero(in_band, as_tuple=False).reshape(-1)
            samples: List[ActivationSample] = []
            if idxs.numel() > 0 and samples_per_band > 0:
                w = oct_pop[octs[idxs]] / torch.clamp(cand_per_oct[octs[idxs]], min=1.0)
                w = torch.where(w > 0, w, torch.ones_like(w))
                take = min(samples_per_band, idxs.numel())
                pick = torch.multinomial(w, take, replacement=False, generator=gen)
                chosen = idxs[pick]
                chosen = chosen[torch.argsort(vals[chosen], descending=True)]
                for ci in chosen.tolist():
                    samples.append(
                        self._decode_sample(int(codes[ci]), sign * float(vals[ci]), int(regs[ci]), sign * extreme)
                    )
            bands.append(
                ActivationBand(
                    pole=pole,
                    lower_fraction=round(v_lo / extreme, 6),
                    upper_fraction=round(min(v_hi / extreme, 1.0), 6),
                    lower_value=sign * v_lo,
                    upper_value=sign * v_hi,
                    count=count,
                    regime_counts=reg_counts,
                    samples=samples,
                )
            )
        return bands

    def export_profile(
        self,
        neuron_idx: int,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
        samples_per_band: int = 4,
    ) -> NeuronPoleProfile:
        n = max(self.count, 1)
        max_v = float(self.max_val[neuron_idx].item())
        min_v = float(self.min_val[neuron_idx].item())
        max_v = max_v if math.isfinite(max_v) else 0.0
        min_v = min_v if math.isfinite(min_v) else 0.0
        pos_ext = max_v if max_v > 0 else 0.0
        neg_ext = min_v if min_v < 0 else 0.0

        pos_m = self._moments(self.pole_count[0], self.pole_mean[0], self.pole_m2[0], self.pole_m3[0], self.pole_m4[0], neuron_idx, pos_ext)
        neg_m = self._moments(self.pole_count[1], self.pole_mean[1], self.pole_m2[1], self.pole_m3[1], self.pole_m4[1], neuron_idx, neg_ext)
        regime_m: Dict[str, DistributionMoments] = {}
        regime_top: Dict[str, List[ActivationSample]] = {}
        for r, reg in enumerate(self.regimes):
            ext = pos_ext if reg.pole == PoleType.POSITIVE else neg_ext
            regime_m[reg.value] = self._moments(self.reg_count[r], self.reg_mean[r], self.reg_m2[r], self.reg_m3[r], self.reg_m4[r], neuron_idx, ext)
            regime_top[reg.value] = self._regime_top_samples(neuron_idx, r, ext)

        def _merge(pole: PoleType) -> List[ActivationSample]:
            merged: List[ActivationSample] = []
            for r, reg in enumerate(self.regimes):
                if reg.pole == pole:
                    merged.extend(regime_top[reg.value])
            merged.sort(key=lambda s: abs(s.activation_value), reverse=True)
            return merged[: self.top_k]

        pos_count = int(self.pole_count[0, neuron_idx].item())
        nn_rate = 0.0
        if self.gated and pos_count > 0:
            nn_rate = int(self.reg_count[1, neuron_idx].item()) / pos_count

        return NeuronPoleProfile(
            layer_name=self.layer_name,
            neuron_idx=neuron_idx,
            tokens_observed=self.count,
            mean=float(self.mean[neuron_idx].item()),
            std=math.sqrt(max(float(self.m2[neuron_idx].item()) / n, 0.0)),
            max_val=max_v,
            min_val=min_v,
            positive_firing_rate=pos_m.rate,
            negative_firing_rate=neg_m.rate,
            gated=self.gated,
            positive_pole=pos_m,
            negative_pole=neg_m,
            regime_moments=regime_m,
            swiglu_sign_regime_rates={k: v.rate for k, v in regime_m.items()} if self.gated else {},
            double_negative_positive_rate=nn_rate,
            top_positive_samples=_merge(PoleType.POSITIVE),
            top_negative_samples=_merge(PoleType.NEGATIVE),
            regime_top_samples=regime_top,
            positive_bands=self.activation_bands(neuron_idx, PoleType.POSITIVE, band_edges, samples_per_band),
            negative_bands=self.activation_bands(neuron_idx, PoleType.NEGATIVE, band_edges, samples_per_band),
        )

    @torch.no_grad()
    def scan(self, write_norms: Optional[torch.Tensor] = None) -> List[NeuronScanSummary]:
        """Vectorised descriptive summary of every neuron (no cleanliness classification)."""
        n = max(self.count, 1)
        std = torch.sqrt(torch.clamp(self.m2 / n, min=0.0))
        rms = torch.sqrt(torch.clamp(self.m2 / n + self.mean ** 2, min=0.0))
        pc = self.pole_count.to(torch.float64)
        pos_k = _excess_kurtosis(pc[0], self.pole_m2[0], self.pole_m4[0])
        neg_k = _excess_kurtosis(pc[1], self.pole_m2[1], self.pole_m4[1])
        if self.gated:
            dbl = self.reg_count[1].to(torch.float64) / torch.clamp(pc[0], min=1.0)
        else:
            dbl = torch.zeros_like(std)
        wn = write_norms.to(torch.float64).cpu() if write_norms is not None else torch.full_like(std.cpu(), float("nan"))
        rows: List[NeuronScanSummary] = []
        mean_l = self.mean.cpu().tolist()
        std_l = std.cpu().tolist()
        rms_l = rms.cpu().tolist()
        max_l = self.max_val.cpu().tolist()
        min_l = self.min_val.cpu().tolist()
        pr = (pc[0] / n).cpu().tolist()
        nr = (pc[1] / n).cpu().tolist()
        pk = pos_k.cpu().tolist()
        nk = neg_k.cpu().tolist()
        dl = dbl.cpu().tolist()
        wl = wn.tolist()
        pos_codes = self._regime_codes_for_pole(PoleType.POSITIVE)
        neg_codes = self._regime_codes_for_pole(PoleType.NEGATIVE)
        top_pos_val, top_pos_reg = self.top_vals[pos_codes, 0, :].max(dim=0)
        top_neg_val, top_neg_reg = self.top_vals[neg_codes, 0, :].max(dim=0)
        top_pos_code = self.top_codes[pos_codes, 0, :].gather(0, top_pos_reg.unsqueeze(0))[0]
        top_neg_code = self.top_codes[neg_codes, 0, :].gather(0, top_neg_reg.unsqueeze(0))[0]
        tpc = top_pos_code.cpu().tolist()
        tnc = top_neg_code.cpu().tolist()

        def _tok(code: int) -> Optional[str]:
            if code < 0:
                return None
            internal, pos = divmod(code, POS_BASE)
            entry = self.registry.get(internal)
            if entry is None or entry[1] is None or pos >= len(entry[1]):
                return None
            return entry[1][pos]

        for i in range(self.num_neurons):
            mx = max_l[i] if math.isfinite(max_l[i]) else 0.0
            mn = min_l[i] if math.isfinite(min_l[i]) else 0.0
            w = wl[i]
            rows.append(
                NeuronScanSummary(
                    neuron_idx=i,
                    mean=mean_l[i],
                    std=std_l[i],
                    max_val=mx,
                    min_val=mn,
                    positive_firing_rate=pr[i],
                    negative_firing_rate=nr[i],
                    positive_excess_kurtosis=pk[i],
                    negative_excess_kurtosis=nk[i],
                    double_negative_positive_rate=dl[i],
                    outgoing_weight_norm=w,
                    rms_write_norm=rms_l[i] * w if math.isfinite(w) else float("nan"),
                    top_pos_token=_tok(int(tpc[i])),
                    top_neg_token=_tok(int(tnc[i])),
                )
            )
        return rows

    def layer_regime_band_counts(
        self,
        pole: PoleType = PoleType.POSITIVE,
        fraction_edges: Optional[Sequence[float]] = None,
        percentile_edges: Optional[Sequence[float]] = None,
        neuron_indices: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        'The double-negative regime fills the noisy middle': sign-regime counts inside activation bands,
        summed over neurons, where each neuron's bands are defined on its own scale, either as fractions
        of its pole extreme (`fraction_edges`, descending 1.0 -> 0.0) or as within-pole percentiles
        (`percentile_edges`, ascending 0 -> 100). Bins are assigned to a band by the value (fraction mode)
        or cumulative share (percentile mode) at the bin's centre, so counts are exact up to one bin.
        """
        if (fraction_edges is None) == (percentile_edges is None):
            raise ValueError("Give exactly one of fraction_edges or percentile_edges.")
        idxs = list(range(self.num_neurons)) if neuron_indices is None else list(neuron_indices)
        codes = self._regime_codes_for_pole(pole)
        if fraction_edges is not None:
            edges = list(fraction_edges)
            labels = [(lo, hi) for hi, lo in zip(edges, edges[1:])]
        else:
            edges = list(percentile_edges)
            labels = [(lo, hi) for lo, hi in zip(edges, edges[1:])]
        totals = [{self.regimes[c].value: 0 for c in codes} for _ in labels]
        centres = torch.tensor([0.0] + [2.0 ** (self.min_exponent + (b - 0.5) / self.bins_per_octave) for b in range(1, self.n_bins)],
                               dtype=torch.float64)
        for i in idxs:
            h = self.hist[codes, i, :].to(torch.float64)  # [R_pole, n_bins]
            ph = h.sum(0)
            tot = float(ph.sum())
            if tot == 0:
                continue
            if fraction_edges is not None:
                ext = self.pole_extreme_magnitude(i, pole)
                key = centres / ext
                key[-1] = min(float(key[-1]), 1.0)
                for k, (lo, hi) in enumerate(labels):
                    sel = (key >= lo) & ((key < hi) if hi < 1.0 else torch.ones_like(key, dtype=torch.bool))
                    for r, c in enumerate(codes):
                        totals[k][self.regimes[c].value] += int(h[r][sel].sum())
            else:
                cum = torch.cumsum(ph, 0)
                mid = (cum - ph / 2.0) / tot * 100.0
                for k, (lo, hi) in enumerate(labels):
                    sel = (mid >= lo) & ((mid < hi) if hi < 100.0 else torch.ones_like(mid, dtype=torch.bool))
                    for r, c in enumerate(codes):
                        totals[k][self.regimes[c].value] += int(h[r][sel].sum())
        out = []
        for (lo, hi), cnt in zip(labels, totals):
            s = sum(cnt.values())
            out.append({"band": [lo, hi], "counts": cnt, "shares": {k: (v / s if s else 0.0) for k, v in cnt.items()}})
        return out

    def layer_double_negative_positive_rate(self) -> float:
        if not self.gated:
            return 0.0
        pos_total = int(self.pole_count[0].sum().item())
        return int(self.reg_count[1].sum().item()) / pos_total if pos_total else 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    _TENSOR_FIELDS = (
        "mean", "m2", "m3", "m4", "max_val", "min_val",
        "pole_count", "pole_mean", "pole_m2", "pole_m3", "pole_m4",
        "reg_count", "reg_mean", "reg_m2", "reg_m3", "reg_m4",
        "top_vals", "top_codes", "hist",
        "res_keys", "res_vals", "res_codes", "res_regime",
    )

    def save_state(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Serialises the complete tracker (moments, top lists, histograms, reservoirs, registry)."""
        ref = set(int(i) for i in self._referenced_internal_indices().tolist())
        state: Dict[str, Any] = {
            "config": {
                "layer_name": self.layer_name,
                "num_neurons": self.num_neurons,
                "gated": self.gated,
                "top_k": self.top_k,
                "bins_per_octave": self.bins_per_octave,
                "min_exponent": self.min_exponent,
                "max_exponent": self.max_exponent,
                "samples_per_octave": self.samples_per_octave,
                "seed": self.seed,
                "window_tokens_before": self.window_tokens_before,
                "window_tokens_after": self.window_tokens_after,
            },
            "count": self.count,
            "next_internal": self._next_internal,
            "generator_state": self._generator.get_state(),
            "registry": {k: v for k, v in self.registry.items() if k in ref},
        }
        for f in self._TENSOR_FIELDS:
            state[f] = getattr(self, f).cpu()
        if path is not None:
            torch.save(state, str(path))
        return state

    @classmethod
    def load_state(cls, state_or_path: Union[Dict[str, Any], str, Path], device: str = "cpu") -> "StreamingLayerStats":
        state = (
            torch.load(str(state_or_path), map_location="cpu", weights_only=False)
            if isinstance(state_or_path, (str, Path))
            else state_or_path
        )
        inst = cls(device=device, **state["config"])
        inst.count = int(state["count"])
        inst._next_internal = int(state["next_internal"])
        inst._generator.set_state(state["generator_state"])
        inst.registry = dict(state["registry"])
        for f in cls._TENSOR_FIELDS:
            setattr(inst, f, state[f].to(device=inst.device))
        return inst


class LogitLensAttributor:
    """
    Direct logit attribution of a neuron's write vector: logits contribution of a unit activation,
    W_U (gamma ⊙ W_down[:, i]), where gamma is the final norm's scale. This is the direct path only
    (it ignores everything downstream and the norm's data-dependent denominator).
    """

    @staticmethod
    @torch.no_grad()
    def attribute_write_vector(
        write_vec: torch.Tensor,
        unembed_weight: torch.Tensor,
        idx_to_token: Optional[Dict[int, str]] = None,
        top_k: int = 10,
        final_norm_module: Optional[nn.Module] = None,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        vec = write_vec.detach().to(torch.float32)
        if final_norm_module is not None:
            scale = LayerDiscovery.extract_norm_scale(final_norm_module, d_model=vec.shape[0], device=vec.device)
            if scale is not None:
                vec = vec * scale
        unembed = unembed_weight.detach().to(vec.device, dtype=torch.float32)
        if unembed.shape[1] != vec.shape[0]:
            if unembed.shape[0] == vec.shape[0]:
                unembed = unembed.T
            else:
                return [], []
        logits = torch.mv(unembed, vec)
        k = min(top_k, logits.numel())
        pv, pi = torch.topk(logits, k=k, largest=True)
        nv, ni = torch.topk(logits, k=k, largest=False)

        def _fmt(ids: torch.Tensor, vals: torch.Tensor) -> List[Tuple[str, float]]:
            out = []
            for t, s in zip(ids.tolist(), vals.tolist()):
                label = idx_to_token.get(t, f"token_{t}") if idx_to_token else f"token_{t}"
                out.append((label, float(s)))
            return out

        return _fmt(pi, pv), _fmt(ni, nv)


# =============================================================================
# Contrastive discovery with prompts as the independent units
# =============================================================================
def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-16) -> float:
    """Continued fraction for the regularised incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else 1.0 + aa / tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else 1.0 + aa / tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    if not math.isfinite(t):
        return 0.0
    if df <= 0:
        return 1.0
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


class ContrastivePoleFinder:
    """
    Ranks neurons by Welch's t between per-prompt mean activations on a positive prompt set and a
    negative prompt set. Tokens inside one prompt are not independent, so each prompt contributes one
    observation (its mean over valid tokens). Reports Welch–Satterthwaite degrees of freedom,
    two-sided p-values and a Bonferroni correction for the number of neurons tested.
    """

    @staticmethod
    @torch.no_grad()
    def _per_prompt_means(
        model: nn.Module,
        batches: List[Any],
        layer_names: List[str],
        forward_fn: Callable[[nn.Module, Any], Any],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        counts = {"tokens": 0}
        current: Dict[str, Any] = {}
        rows: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}

        def _make_capture(name: str):
            def _cap(act: torch.Tensor) -> None:
                a = act.detach().float()
                mask = current.get("mask")
                if a.ndim == 2:  # [B, D] (one position per prompt)
                    rows[name].append(a.to(device))
                    return
                if a.ndim == 4:
                    a = a.flatten(2).transpose(1, 2)  # [B, HW, C]
                if mask is not None and mask.shape == a.shape[:2]:
                    m = mask.to(device=a.device, dtype=a.dtype).unsqueeze(-1)
                    per = (a * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1.0)
                else:
                    per = a.mean(dim=1)
                rows[name].append(per.to(device))
            return _cap

        from .hooks import HookSet

        hs = HookSet(model)
        for name in layer_names:
            cap = _make_capture(name)

            def _c(act, _g, _r, c=cap, site=hs.site(name)):
                if site.target == "moe_expert":
                    grid = current.get("grid")
                    if grid is None:
                        raise ValueError("MoE expert layers need 2D input_ids batches.")
                    act = act.reshape(grid[0], grid[1], -1)  # unrouted tokens are 0: the neuron does not fire
                c(act)

            hs.add_capture(name, _c)
        hs.attach()
        try:
            for b in batches:
                current["mask"] = b.get("attention_mask") if isinstance(b, dict) else None
                ids = b.get("input_ids") if isinstance(b, dict) else b
                current["grid"] = tuple(ids.shape) if isinstance(ids, torch.Tensor) and ids.ndim == 2 else None
                if current["mask"] is not None:
                    counts["tokens"] += int(current["mask"].sum().item())
                forward_fn(model, b)
        finally:
            hs.detach()
        out: Dict[str, torch.Tensor] = {}
        n_prompts = 0
        for name in layer_names:
            if rows[name]:
                out[name] = torch.cat(rows[name], dim=0)
                n_prompts = out[name].shape[0]
        return out, counts["tokens"] if counts["tokens"] else n_prompts

    @classmethod
    def discover(
        cls,
        model: nn.Module,
        pos_inputs: Union[torch.Tensor, Iterable[Any]],
        neg_inputs: Union[torch.Tensor, Iterable[Any]],
        layer_names: List[str],
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        top_k: int = 16,
        device: str = "cpu",
    ) -> ContrastiveDiscoveryReport:
        run_fwd = forward_fn or (lambda m, inp: m(inp))
        pos_batches = [pos_inputs] if isinstance(pos_inputs, torch.Tensor) else list(pos_inputs)
        neg_batches = [neg_inputs] if isinstance(neg_inputs, torch.Tensor) else list(neg_inputs)
        dev = torch.device(device)
        was_training = model.training
        model.eval()
        try:
            pos_means, pos_tok = cls._per_prompt_means(model, pos_batches, layer_names, run_fwd, dev)
            neg_means, neg_tok = cls._per_prompt_means(model, neg_batches, layer_names, run_fwd, dev)
        finally:
            if was_training:
                model.train()

        candidates: List[Tuple[float, str, int, Dict[str, float]]] = []
        tests = 0
        n_pos_prompts = n_neg_prompts = 0
        for name in layer_names:
            P = pos_means.get(name)
            Q = neg_means.get(name)
            if P is None or Q is None:
                continue
            n1, n2 = P.shape[0], Q.shape[0]
            n_pos_prompts, n_neg_prompts = n1, n2
            if n1 < 2 or n2 < 2:
                raise ValueError("Each prompt set needs at least two prompts for a Welch t-test.")
            m1, m2 = P.mean(0), Q.mean(0)
            v1, v2 = P.var(0, unbiased=True), Q.var(0, unbiased=True)
            se2 = v1 / n1 + v2 / n2
            t = (m1 - m2) / torch.sqrt(se2)
            t = torch.where(se2 > 0, t, torch.zeros_like(t))
            df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
            df = torch.where(torch.isfinite(df), df, torch.full_like(df, float(n1 + n2 - 2)))
            pooled = torch.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            d = torch.where(pooled > 0, (m1 - m2) / pooled, torch.zeros_like(pooled))
            tests += t.numel()
            k = min(top_k, t.numel())
            _, idx = torch.topk(t.abs(), k=k)
            for i in idx.tolist():
                candidates.append((
                    abs(float(t[i])), name, i,
                    {
                        "t": float(t[i]), "df": float(df[i]), "m1": float(m1[i]), "m2": float(m2[i]),
                        "s1": math.sqrt(max(float(v1[i]), 0.0)), "s2": math.sqrt(max(float(v2[i]), 0.0)),
                        "d": float(d[i]),
                    },
                ))
        candidates.sort(key=lambda x: x[0], reverse=True)
        matches: List[ContrastiveNeuronMatch] = []
        for _, name, i, s in candidates[:top_k]:
            p = student_t_two_sided_p(s["t"], s["df"])
            matches.append(
                ContrastiveNeuronMatch(
                    layer_name=name,
                    neuron_idx=i,
                    t_statistic=s["t"],
                    abs_t_statistic=abs(s["t"]),
                    degrees_of_freedom=s["df"],
                    p_value=p,
                    bonferroni_p_value=min(1.0, p * tests),
                    pos_mean=s["m1"],
                    neg_mean=s["m2"],
                    pos_std=s["s1"],
                    neg_std=s["s2"],
                    n_pos_prompts=n_pos_prompts,
                    n_neg_prompts=n_neg_prompts,
                    bipolar_sign_flip=(s["m1"] > 0) != (s["m2"] > 0) and s["m1"] != 0 and s["m2"] != 0,
                    recommended_pole=PoleType.POSITIVE if s["t"] >= 0 else PoleType.NEGATIVE,
                    cohens_d=s["d"],
                )
            )
        return ContrastiveDiscoveryReport(
            layers_scanned=list(layer_names),
            pos_prompts_count=n_pos_prompts,
            neg_prompts_count=n_neg_prompts,
            pos_tokens_count=pos_tok,
            neg_tokens_count=neg_tok,
            tests_performed=tests,
            top_matches=matches,
        )


class CheckpointEvolutionTracker:
    """
    'Watching Neurons Develop': audits one neuron across checkpoints on a shared corpus and reports,
    per checkpoint, its moments, write-vector drift and which tokens populate each activation band,
    so the order in which branches appear can be read off.
    """

    @staticmethod
    @torch.no_grad()
    def track_neuron_evolution(
        checkpoints: List[Tuple[str, Union[nn.Module, Callable[[], nn.Module]]]],
        layer_name: str,
        neuron_idx: int,
        batches: Iterable[Dict[str, Any]],
        forward_fn: Callable[[nn.Module, Dict[str, Any]], Any],
        top_k: int = 16,
        band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
    ) -> NeuronEvolutionReport:
        from .injector import NeuralInjector

        batch_list = list(batches)
        snapshots: List[NeuronCheckpointSnapshot] = []
        write_vecs: List[Optional[torch.Tensor]] = []
        profiles: List[Tuple[str, NeuronPoleProfile]] = []
        for label, ckpt in checkpoints:
            model = ckpt() if callable(ckpt) and not isinstance(ckpt, nn.Module) else ckpt
            try:
                wv = LayerDiscovery.neuron_write_vectors(model, layer_name)
                write_vecs.append(wv[neuron_idx].float().cpu() if wv is not None else None)
            except (ValueError, IndexError):
                write_vecs.append(None)
            inj = NeuralInjector(model)
            inj.audit_layers([layer_name], batch_list, forward_fn=forward_fn, top_k=top_k)
            profiles.append((label, inj.get_neuron_profile(layer_name, neuron_idx, band_edges=band_edges)))
            del inj

        final = write_vecs[-1] if write_vecs else None
        for (label, prof), wv in zip(profiles, write_vecs):
            if wv is not None and final is not None and wv.shape == final.shape and wv.norm() > 0 and final.norm() > 0:
                cos = float(F.cosine_similarity(wv.unsqueeze(0), final.unsqueeze(0)).item())
                wnorm = float(wv.norm().item())
            else:
                cos, wnorm = float("nan"), float("nan")
            band_tokens: Dict[str, List[str]] = {}
            for pole_name, bands in (("positive", prof.positive_bands), ("negative", prof.negative_bands)):
                for b in bands:
                    key = f"{pole_name}:{b.lower_fraction:.4f}-{b.upper_fraction:.4f}"
                    band_tokens[key] = list(dict.fromkeys(s.token_str for s in b.samples if s.token_str))
            snapshots.append(
                NeuronCheckpointSnapshot(
                    checkpoint_label=label,
                    mean=prof.mean,
                    std=prof.std,
                    max_val=prof.max_val,
                    min_val=prof.min_val,
                    positive_firing_rate=prof.positive_firing_rate,
                    negative_firing_rate=prof.negative_firing_rate,
                    outgoing_weight_norm=wnorm,
                    weight_cosine_to_final=cos,
                    band_tokens=band_tokens,
                    double_negative_positive_rate=prof.double_negative_positive_rate,
                )
            )

        emerged: Dict[str, List[str]] = {}
        if len(snapshots) >= 2:
            first, last = snapshots[0].band_tokens, snapshots[-1].band_tokens
            seen_first = {t for toks in first.values() for t in toks}
            for key, toks in last.items():
                new = [t for t in toks if t not in seen_first]
                if new:
                    emerged[key] = new
        return NeuronEvolutionReport(
            layer_name=layer_name,
            neuron_idx=neuron_idx,
            snapshots=snapshots,
            emerged_tokens_by_band=emerged,
        )
