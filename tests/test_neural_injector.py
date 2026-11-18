"""
Tests check every statistic and intervention against an independent brute-force reference,
and check the explanation protocol's information flow (what each agent is shown).
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import threading
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_injector as ni
from neural_injector import (
    CallableLLMClient,
    FalsificationOutcome,
    InterventionKernel,
    InterventionMode,
    InterventionSpec,
    LayerDiscovery,
    NeuralInjector,
    NeuronTaxonomy,
    PoleType,
    ProductionSteeringRuntime,
    StreamingLayerStats,
    WorkbenchSession,
)
from neural_injector.analyzer import student_t_two_sided_p
from neural_injector.hooks import HookSet

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def llama() -> nn.Module:
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def llama_batches() -> List[Dict[str, Any]]:
    g = torch.Generator().manual_seed(1)
    out = []
    for _ in range(5):
        ids = torch.randint(0, 97, (2, 15), generator=g)
        mask = torch.ones_like(ids)
        mask[1, 10:] = 0
        out.append({"input_ids": ids, "attention_mask": mask,
                    "tokens_per_sample": [[f"<{i}>" for i in r.tolist()] for r in ids],
                    "contexts_per_sample": [f"ctx-{i}" for i in range(2)]})
    return out


@pytest.fixture(scope="module")
def llama_layers(llama) -> List[str]:
    return list(NeuralInjector(llama).list_targetable_layers().keys())


def _collect(model, layer, batches, need_gate=True):
    """Brute-force reference: every valid activation and gate of `layer`."""
    acts, gates = [], []
    hs = HookSet(model)
    cur: Dict[str, Any] = {}

    def cap(a, g, r):
        m = cur["mask"].reshape(-1).bool()
        acts.append(a.reshape(-1, a.shape[-1])[m].float())
        if g is not None:
            gates.append(g.reshape(-1, g.shape[-1])[m].float())

    hs.add_capture(layer, cap, need_gate=need_gate)
    with torch.no_grad(), hs.active():
        for b in batches:
            cur["mask"] = b["attention_mask"]
            model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    return torch.cat(acts), (torch.cat(gates) if gates else None)


# =============================================================================
# 1. Discovery
# =============================================================================
class TestDiscovery:
    def test_version(self):
        assert ni.__version__ == "0.7.0"

    def test_llama_swiglu_layers_and_gate_source(self, llama, llama_layers):
        assert llama_layers == [f"model.layers.{i}.mlp.down_proj" for i in range(3)]
        mod, kind = LayerDiscovery.gate_source(llama, llama_layers[0])
        assert kind == "full" and mod is llama.model.layers[0].mlp.gate_proj
        W = LayerDiscovery.neuron_write_vectors(llama, llama_layers[0])
        assert torch.equal(W, llama.model.layers[0].mlp.down_proj.weight.T)
        up, gate = LayerDiscovery.neuron_read_vectors(llama, llama_layers[0])
        assert torch.equal(up, llama.model.layers[0].mlp.up_proj.weight)
        assert torch.equal(gate, llama.model.layers[0].mlp.gate_proj.weight)

    def test_gpt2_conv1d_not_gated(self):
        m = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_embd=32, n_layer=2, n_head=4)).eval()
        layers = NeuralInjector(m).list_targetable_layers()
        assert list(layers) == ["transformer.h.0.mlp.c_proj", "transformer.h.1.mlp.c_proj"]
        assert LayerDiscovery.gate_source(m, "transformer.h.0.mlp.c_proj")[1] == "none"
        W = LayerDiscovery.neuron_write_vectors(m, "transformer.h.0.mlp.c_proj")
        assert W.shape == (128, 32)

    def test_moe_param_container_exposes_each_expert_as_its_own_layer(self):
        m = MixtralForCausalLM(MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                                             num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                                             num_experts_per_tok=2)).eval()
        layers = list(NeuralInjector(m).list_targetable_layers())
        assert layers == [f"model.layers.0.mlp.experts@expert{e}" for e in range(4)]
        ex = m.model.layers[0].mlp.experts
        W = LayerDiscovery.neuron_write_vectors(m, layers[2])
        assert torch.equal(W, ex.down_proj[2].T)
        up, gate = LayerDiscovery.neuron_read_vectors(m, layers[2])
        g_ref, u_ref = ex.gate_up_proj[2].chunk(2, dim=0)
        assert torch.equal(up, u_ref) and torch.equal(gate, g_ref)

    def test_fused_gate_up_is_split_gate_first(self):
        class Fused(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Linear(8, 12, bias=False)
                self.down_proj = nn.Linear(6, 8, bias=False)

            def forward(self, x):
                g, u = self.gate_up_proj(x).chunk(2, dim=-1)
                return self.down_proj(F.silu(g) * u)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = Fused()

            def forward(self, x):
                return self.mlp(x)

        m = M()
        assert LayerDiscovery.gate_source(m, "mlp.down_proj")[1] == "first_half"
        got = {}
        hs = HookSet(m)
        hs.add_capture("mlp.down_proj", lambda a, g, r: got.update(a=a, g=g), need_gate=True)
        x = torch.randn(2, 3, 8)
        with hs.active():
            m(x)
        g_ref, u_ref = m.mlp.gate_up_proj(x).chunk(2, dim=-1)
        assert torch.allclose(got["g"], g_ref) and torch.allclose(got["a"], F.silu(g_ref) * u_ref)


# =============================================================================
# 2. Streaming statistics are exact
# =============================================================================
class TestStreamingStatistics:
    @pytest.fixture(scope="class")
    def audited(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        inj.audit_layers(llama_layers, llama_batches, top_k=6)
        return inj

    def test_moments_match_full_batch(self, audited, llama, llama_batches, llama_layers):
        tr = audited.get_stats_tracker(llama_layers[1])
        A, _ = _collect(llama, llama_layers[1], llama_batches)
        A = A.double()
        assert tr.count == A.shape[0] == sum(int(b["attention_mask"].sum()) for b in llama_batches)
        assert torch.allclose(tr.mean, A.mean(0), atol=1e-9)
        c = A - A.mean(0)
        for k, m in ((2, tr.m2), (3, tr.m3), (4, tr.m4)):
            assert torch.allclose(m, (c ** k).sum(0), rtol=1e-5, atol=1e-12)
        pos = A > 0
        for j in (0, 7, 33):
            v = A[pos[:, j], j]
            n = v.numel()
            mean = v.mean()
            m2 = ((v - mean) ** 2).sum()
            m4 = ((v - mean) ** 4).sum()
            assert int(tr.pole_count[0, j]) == n
            assert math.isclose(float(tr.pole_mean[0, j]), float(mean), rel_tol=1e-6)
            assert math.isclose(float(tr.pole_m2[0, j]), float(m2), rel_tol=1e-5)
            prof = tr.export_profile(j)
            assert math.isclose(prof.positive_pole.excess_kurtosis, float(n * m4 / m2 ** 2 - 3), rel_tol=1e-4)

    def test_sign_regimes_and_top_k_are_exact(self, audited, llama, llama_batches, llama_layers):
        tr = audited.get_stats_tracker(llama_layers[1])
        A, G = _collect(llama, llama_layers[1], llama_batches)
        for j in (2, 11, 40):
            a, g = A[:, j], G[:, j]
            ref = {"++": (a > 0) & (g > 0), "--": (a > 0) & (g < 0), "+-": (a < 0) & (g > 0), "-+": (a < 0) & (g < 0)}
            prof = tr.export_profile(j)
            for r, m in ref.items():
                assert prof.regime_moments[r].count == int(m.sum())
                exp = torch.sort(a[m].abs(), descending=True).values[: tr.top_k].tolist()
                got = [abs(s.activation_value) for s in prof.regime_top_samples[r]]
                assert got == pytest.approx(exp, rel=1e-6)
                assert all(s.sign_regime == r for s in prof.regime_top_samples[r])
            # the ++ list never contains -- echoes; the raw list is their exact merge
            raw = torch.sort(a[a > 0], descending=True).values[: tr.top_k].tolist()
            assert [s.activation_value for s in prof.top_positive_samples] == pytest.approx(raw, rel=1e-6)
            assert prof.double_negative_positive_rate == pytest.approx(int(ref["--"].sum()) / int((a > 0).sum()))

    def test_band_counts_are_exact_over_snapped_edges(self, audited, llama, llama_batches, llama_layers):
        tr = audited.get_stats_tracker(llama_layers[2])
        A, G = _collect(llama, llama_layers[2], llama_batches)
        j = 5
        a, g = A[:, j], G[:, j]
        bands = tr.activation_bands(j, PoleType.POSITIVE)
        assert sum(b.count for b in bands) == int((a > 0).sum())
        for b in bands:
            inside = (a > 0) & (a >= b.lower_value) & ((a < b.upper_value) if b.upper_fraction < 1.0 else torch.ones_like(a, dtype=torch.bool))
            assert b.count == int(inside.sum())
            assert b.regime_counts["--"] == int((inside & (g < 0)).sum())
            for s in b.samples:
                assert b.lower_value <= s.activation_value and (b.upper_fraction == 1.0 or s.activation_value < b.upper_value)

    def test_percentile_within_one_histogram_bin(self, audited, llama, llama_batches, llama_layers):
        tr = audited.get_stats_tracker(llama_layers[0])
        A, _ = _collect(llama, llama_layers[0], llama_batches)
        for j in (1, 9):
            v = torch.sort(A[A[:, j] < 0, j].abs()).values
            est = tr.percentile_within_pole(j, PoleType.NEGATIVE, 50.0)
            exact = float(torch.quantile(v, 0.5))
            assert abs(math.log2(est / exact)) <= 1.0 / tr.bins_per_octave + 1e-9

    def test_save_load_round_trip_preserves_examples_and_bands(self, audited, llama_layers, tmp_path):
        tr = audited.get_stats_tracker(llama_layers[1])
        path = tmp_path / "state.pt"
        tr.save_state(path)
        tr2 = StreamingLayerStats.load_state(path)
        for j in (0, 3, 79):
            p1, p2 = tr.export_profile(j), tr2.export_profile(j)
            assert [asdict(s) for s in p1.top_positive_samples] == [asdict(s) for s in p2.top_positive_samples]
            assert [asdict(s) for s in p1.top_negative_samples] == [asdict(s) for s in p2.top_negative_samples]
            assert [asdict(b) for b in p1.positive_bands] == [asdict(b) for b in p2.positive_bands]
            assert p1.top_positive_samples[0].token_str is not None

    def test_padding_is_excluded(self, audited, llama_layers):
        tr = audited.get_stats_tracker(llama_layers[0])
        for j in range(5):
            for s in tr.export_profile(j).top_positive_samples:
                assert not (s.sample_id % 2 == 1 and s.position_idx >= 10)

    def test_wide_batches_fill_every_band(self):
        torch.manual_seed(3)
        tr = StreamingLayerStats("x", 1536, gated=False, top_k=8, samples_per_octave=3)
        acts = torch.randn(4, 64, 1536) * torch.rand(1536)
        for b in range(4):
            tr.update(acts[b:b + 1])
        bands = tr.activation_bands(10, PoleType.POSITIVE)
        assert all(b.count == 0 or len(b.samples) > 0 for b in bands)
        assert sum(b.count for b in bands) == int((acts[..., 10] > 0).sum())

    def test_reservoir_samples_are_uniform_within_octave(self):
        torch.manual_seed(0)
        hits = torch.zeros(64)
        for seed in range(400):
            tr = StreamingLayerStats("x", 1, gated=False, top_k=1, samples_per_octave=1, seed=seed, min_exponent=-4, max_exponent=4)
            vals = 1.0 + torch.arange(64, dtype=torch.float32) / 64.0  # all in octave [1, 2)
            tr.update(vals.view(4, 16, 1)[:2])
            tr.update(vals.view(4, 16, 1)[2:])
            code = int(tr.res_codes[0, 0, tr._octave_index(torch.tensor([1.5]))[0], 0])
            internal, pos = divmod(code, ni.analyzer.POS_BASE)
            hits[internal * 16 + pos] += 1
        chi2 = float(((hits - hits.mean()) ** 2 / hits.mean()).sum())
        assert chi2 < 120  # df = 63; p ~ 1e-5 bound


# =============================================================================
# 3. Intervention kernel and gate-conditioned modes
# =============================================================================
class TestInterventions:
    def test_kernel_modes(self):
        x = torch.tensor([[[-2.0, -0.5, 0.25, 3.0]]])
        g = torch.tensor([[[-1.0, 1.0, -0.375, 2.0]]])
        idx = [0, 1, 2, 3]
        k = lambda spec: InterventionKernel.apply(x, [spec], gate=g)
        assert k(InterventionSpec("l", idx, InterventionMode.ZERO_ABLATE, PoleType.NEGATIVE)).tolist() == [[[0.0, 0.0, 0.25, 3.0]]]
        assert k(InterventionSpec("l", idx, InterventionMode.WEAK_PRUNE, PoleType.BOTH, 1.0)).tolist() == [[[-2.0, 0.0, 0.0, 3.0]]]
        assert k(InterventionSpec("l", idx, InterventionMode.DOUBLE_NEGATIVE_ABLATE, PoleType.POSITIVE)).tolist() == [[[-2.0, -0.5, 0.0, 3.0]]]
        cull = InterventionSpec("l", idx, InterventionMode.EXTREME_CULL, value=0.0, per_neuron_values=[0.0, 0.5, 0.0, 1.5], condition_signal="gate")
        assert k(cull).tolist() == [[[-2.0, 0.0, 0.25, 0.0]]]
        with pytest.raises(RuntimeError):
            InterventionKernel.apply(x, [cull])

    def test_double_negative_ablation_live(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        inj.ablate_double_negative_regime([llama_layers[1]])
        got = {}
        hs = HookSet(llama)
        hs.add_capture(llama_layers[1], lambda a, g, r: got.update(a=a, g=g), need_gate=True)
        with inj.active_injection(), hs.active():
            llama(llama_batches[0]["input_ids"])
        assert int(((got["a"] > 0) & (got["g"] < 0)).sum()) == 0
        assert int((got["a"] > 0).sum()) > 0

    def test_gate_modes_rejected_on_ungated_layers(self):
        m = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_embd=32, n_layer=1, n_head=4)).eval()
        inj = NeuralInjector(m)
        with pytest.raises(ValueError):
            inj.ablate_double_negative_regime(["transformer.h.0.mlp.c_proj"])

    def test_tail_thresholds_are_exact_order_statistics(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        th = inj.calibrate_tail_thresholds([llama_layers[1]], llama_batches, percentile=99.0, signal="gate")
        _, G = _collect(llama, llama_layers[1], llama_batches)
        K = int(math.floor(0.01 * G.shape[0])) + 1
        ref = torch.topk(G, K, dim=0).values[K - 1]
        assert torch.allclose(torch.tensor(th[llama_layers[1]]), ref)

    def test_extreme_cull_controls_match_count_and_magnitude(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        th = inj.calibrate_tail_thresholds([llama_layers[1]], llama_batches, percentile=95.0)
        rep = inj.run_extreme_culling_experiment(th, llama_batches, 95.0)
        assert rep.culled.entries_modified > 0
        assert rep.count_matched_random.entries_modified == rep.culled.entries_modified
        assert rep.magnitude_matched_random.entries_modified >= rep.culled.entries_modified

    def test_weak_prune_rules_are_explicit(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        with pytest.raises(KeyError):
            inj.prune_weak_activations(llama_layers[0], fraction_of_extreme=0.25)
        inj.audit_layers([llama_layers[0]], llama_batches)
        with pytest.raises(ValueError):
            inj.prune_weak_activations(llama_layers[0])
        specs = inj.prune_weak_activations(llama_layers[0], fraction_of_extreme=0.25)
        assert [s.pole for s in specs] == [PoleType.POSITIVE, PoleType.NEGATIVE]
        tr = inj.get_stats_tracker(llama_layers[0])
        assert specs[0].per_neuron_values[3] == pytest.approx(0.25 * float(tr.max_val[3]))
        res = inj.measure_weak_activation_ablation([llama_layers[0]], llama_batches, fraction_of_extreme=0.25)
        assert res["weak_removed"].entries_modified == res["count_matched_random"].entries_modified > 0

    def test_conditional_gate_requires_explicit_threshold(self, llama, llama_layers):
        inj = NeuralInjector(llama)
        with pytest.raises(ValueError):
            inj.add_conditional_gate(llama_layers[1], [0], llama_layers[0], 3)
        with pytest.raises(ValueError):
            inj.add_conditional_gate(llama_layers[0], [0], llama_layers[1], 3, threshold=0.1)

    def test_synergy_l1_starts_at_neutral_mask(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        r = inj.find_synergistic_ablation_set(llama_layers[1], llama_batches[0]["input_ids"][:1], max_subset_size=2,
                                              method="l1_optim", candidate_pool_size=6, l1_steps=3)
        assert len(r.selected_neurons) == 2 and r.objective == "mean_kl" and r.joint_score > 0
        g = inj.find_synergistic_ablation_set(llama_layers[1], llama_batches[0]["input_ids"][:1], max_subset_size=2,
                                              candidate_pool_size=4, target_token_ids=[3, 4])
        assert g.objective == "target_logprob_mass_drop"


# =============================================================================
# 4. Circuits
# =============================================================================
class TestCircuits:
    def test_virtual_weights_match_manual(self, llama, llama_layers):
        inj = NeuralInjector(llama)
        vu, vg, _ = inj.tracer.compute_virtual_weights(llama_layers[0], llama_layers[1], 4)
        gamma = llama.model.layers[1].post_attention_layernorm.weight
        w = llama.model.layers[0].mlp.down_proj.weight[:, 4] * gamma
        assert torch.allclose(vu, llama.model.layers[1].mlp.up_proj.weight @ w, atol=1e-6)
        assert torch.allclose(vg, llama.model.layers[1].mlp.gate_proj.weight @ w, atol=1e-6)

    def test_trace_delta_equals_manual_ablation(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        ids = llama_batches[0]["input_ids"][:1]
        g = inj.trace_circuit(llama_layers[0], llama_layers[1], 4, ids, src_pole=PoleType.POSITIVE, top_k=3, compute_jacobian=False)
        A0, _ = _collect(llama, llama_layers[0], [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
        active = A0[:, 4] > 0
        clean, _ = _collect(llama, llama_layers[1], [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
        inj.zero_ablate(llama_layers[0], 4, pole=PoleType.POSITIVE)
        with inj.active_injection():
            abl, _ = _collect(llama, llama_layers[1], [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
        ref = (clean[active] - abl[active]).mean(0)
        for e in g.edges:
            assert e.mean_causal_delta == pytest.approx(float(ref[e.dst_neuron]), rel=1e-4, abs=1e-9)
            assert e.gate_path_delta is not None and e.up_path_delta is not None

    def test_forward_ad_matches_finite_difference(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        ids = llama_batches[1]["input_ids"][:1]
        jac = inj.trace_forward_ad_all_layers(llama_layers[0], 2, ids, [llama_layers[1]], position_mode="focal_token")[llama_layers[1]]
        A0, _ = _collect(llama, llama_layers[0], [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
        t = int(A0[:, 2].argmax())
        eps = 1e-3
        outs = []
        for sgn in (1, -1):
            spec = InterventionSpec(llama_layers[0], [2], InterventionMode.ADDITIVE_STEER, value=sgn * eps, token_positions=[t])
            with inj.temporary_interventions([spec]):
                d, _ = _collect(llama, llama_layers[1], [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
            outs.append(d[t])
        fd = (outs[0] - outs[1]) / (2 * eps)
        # tangent was seeded at t only; compare its effect on position t
        assert torch.allclose(jac, fd, atol=5e-4, rtol=1e-2)

    def test_incoming_attribution_is_grad_times_activation(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        ids = llama_batches[2]["input_ids"][:1]
        attr, jac, _, m = inj.tracer._attribution_into(llama_layers[0], llama_layers[1], 6, ids, PoleType.POSITIVE,
                                                       lambda mm, x: mm(x), None, focal=False)
        holder = {}
        hs = HookSet(llama)

        def leaf(a, g):
            t = a.detach().clone().requires_grad_(True)
            holder["s"] = t
            return t

        hs.add_modifier(llama_layers[0], leaf)
        hs.add_capture(llama_layers[1], lambda a, g, r: holder.__setitem__("d", a))
        with torch.enable_grad(), hs.active():
            llama(ids)
            col = holder["d"].reshape(-1, 80)[:, 6]
            tgt = col[col.detach() > 0].mean()
            gr = torch.autograd.grad(tgt, holder["s"])[0]
        ref = (holder["s"] * gr).reshape(-1, 80).sum(0)
        assert torch.allclose(attr, ref.detach(), atol=1e-7)

    def test_bundles_recover_planted_structure_and_reject_random(self):
        class Toy(nn.Module):
            """3 upstream 'neurons' groups drive the target on disjoint context types."""
            def __init__(self):
                super().__init__()
                self.l0 = nn.Linear(6, 6, bias=False)
                self.mlp = nn.Module()
                self.mlp.down_proj = nn.Linear(6, 4, bias=False)
                self.mlp2 = nn.Module()
                self.mlp2.down_proj = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                h = self.mlp.down_proj(torch.relu(self.l0(x)))
                return self.mlp2.down_proj(torch.relu(h))

        torch.manual_seed(0)
        m = Toy()
        with torch.no_grad():
            m.l0.weight.copy_(torch.eye(6))
            m.mlp.down_proj.weight.copy_(torch.tensor([[1., 1., 1., 1., 1., 1.]] + [[0.0] * 6] * 3))
        inj = NeuralInjector(m)
        ctxs = []
        for k in range(12):
            x = torch.zeros(1, 1, 6)
            grp = k % 2
            x[0, 0, 3 * grp: 3 * grp + 3] = 1.0 + 0.1 * torch.rand(3)
            ctxs.append(x)
        b = inj.cluster_incoming_attribution_bundles("mlp.down_proj", "mlp2.down_proj", 0, ctxs, top_k_upstream=6, max_bundles=3)
        groups = sorted(sorted(x.upstream_neurons) for x in b)
        assert groups == [[0, 1, 2], [3, 4, 5]] and b[0].silhouette_null_p < 0.05

    def test_residual_sink_detects_planted_unread_high_mass_coordinate(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 8)
                self.block = nn.Identity()
                self.norm = nn.LayerNorm(8)
                self.mlp = nn.Module()
                self.mlp.up_proj = nn.Linear(8, 6, bias=False)
                self.mlp.down_proj = nn.Linear(6, 8, bias=False)
                self.lm_head = nn.Linear(8, 10, bias=False)

            def get_output_embeddings(self):
                return self.lm_head

            def forward(self, ids):
                x = self.block(self.embed(ids))
                x = x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(self.norm(x))))
                return self.lm_head(x)

        torch.manual_seed(0)
        m = Toy()
        with torch.no_grad():
            m.embed.weight[:, 5] = 40.0
            m.mlp.up_proj.weight[:, 5] = 0.0
            m.lm_head.weight[:, 5] = 0.0
        inj = NeuralInjector(m)
        rep = inj.audit_residual_stream_sinks("block", [torch.randint(0, 10, (2, 6)) for _ in range(3)])
        assert rep.sink_candidate_coordinates == [5]
        assert rep.coordinates[0].coordinate == 5 and rep.coordinates[0].read_share == 0.0

    def test_combination_readout_null(self, llama, llama_layers):
        inj = NeuralInjector(llama)
        W = LayerDiscovery.neuron_write_vectors(llama, llama_layers[0]).clone()
        r = inj.verify_combination_readout(llama_layers[0], [1, 2, 3], null_samples=100)
        assert 0.0 <= r.combination_percentile_in_null <= 1.0 and r.best_single_constituent_cosine > 0
        # plant a downstream reader of exactly the combination
        gamma = llama.model.layers[1].post_attention_layernorm.weight
        comb = (W[1] + W[2] + W[3])
        with torch.no_grad():
            saved = llama.model.layers[1].mlp.up_proj.weight[0].clone()
            llama.model.layers[1].mlp.up_proj.weight[0] = comb / gamma
            try:
                r2 = inj.verify_combination_readout(llama_layers[0], [1, 2, 3], null_samples=100)
            finally:
                llama.model.layers[1].mlp.up_proj.weight[0] = saved
        assert r2.reads_combination_beyond_constituents and r2.combination_max_reader_cosine == pytest.approx(1.0, abs=1e-5)

    def test_direct_effect_equals_total_for_last_layer(self, llama, llama_batches, llama_layers):
        inj = NeuralInjector(llama)
        ids = llama_batches[0]["input_ids"][:1]
        d = inj.decompose_direct_vs_indirect_effect(llama_layers[2], 7, ids, target_token_id=5)
        # nothing reads the last MLP except the unembedding, so removing the direct path == ablating
        assert d["direct_path_removed_target_prob"] == pytest.approx(d["total_ablated_target_prob"], rel=1e-5)
        d0 = inj.decompose_direct_vs_indirect_effect(llama_layers[0], 7, ids, target_token_id=5,
                                                     downstream_layer=llama_layers[1], rescue_downstream_neurons=list(range(80)))
        assert "rescued_target_prob" in d0


# =============================================================================
# 5. Contrastive discovery
# =============================================================================
class TestContrastive:
    def test_t_distribution_p_value(self):
        assert student_t_two_sided_p(2.228138852, 10) == pytest.approx(0.05, abs=1e-6)
        assert student_t_two_sided_p(0.0, 5) == pytest.approx(1.0)

    def test_welch_on_per_prompt_means(self, llama, llama_layers):
        inj = NeuralInjector(llama)
        g = torch.Generator().manual_seed(5)
        pos = [torch.randint(0, 40, (1, 9), generator=g) for _ in range(4)]
        neg = [torch.randint(50, 97, (1, 11), generator=g) for _ in range(5)]
        rep = inj.discover_contrastive_neurons(pos, neg, layer_names=[llama_layers[1]], top_k=3)
        m = rep.top_matches[0]
        P = torch.stack([_collect(llama, llama_layers[1], [{"input_ids": x, "attention_mask": torch.ones_like(x)}])[0].mean(0) for x in pos])
        Q = torch.stack([_collect(llama, llama_layers[1], [{"input_ids": x, "attention_mask": torch.ones_like(x)}])[0].mean(0) for x in neg])
        j = m.neuron_idx
        t = (P[:, j].mean() - Q[:, j].mean()) / torch.sqrt(P[:, j].var() / 4 + Q[:, j].var() / 5)
        assert m.t_statistic == pytest.approx(float(t), rel=1e-4)
        assert rep.tests_performed == 80 and m.n_pos_prompts == 4 and m.n_neg_prompts == 5
        specs = inj.inject_contrastive_behavior(rep, strength=1.0, top_k=1)
        assert specs[0].per_neuron_values[0] == pytest.approx(m.pos_mean - m.neg_mean)


# =============================================================================
# 6. Export and folding
# =============================================================================
class TestExport:
    def test_vectors_only_for_additive_and_exact_equivalence(self, llama, llama_batches, llama_layers):
        torch.manual_seed(0)
        model = LlamaForCausalLM(llama.config).eval()
        model.load_state_dict(llama.state_dict())
        inj = NeuralInjector(model)
        ids = llama_batches[0]["input_ids"]
        inj.steer_neurons(llama_layers[1], [1, 2], delta=0.5)
        inj.clamp_neurons(llama_layers[2], [4], 1.25)
        inj.add_intervention(InterventionSpec(llama_layers[0], [3], InterventionMode.SCALE, value=1.7))
        with inj.active_injection():
            ref = model(ids).logits
        bundle = inj.export_production_preset()
        assert [cv.layer_name for cv in bundle.compiled_vectors] == [llama_layers[1]]
        rt = ProductionSteeringRuntime(model, json.loads(json.dumps(asdict(bundle))))
        with rt.active(use_compiled_residual_vectors=True):
            vec = model(ids).logits
        with rt.active(use_compiled_residual_vectors=False):
            hk = model(ids).logits
        res = inj.fold_interventions_into_weights()
        assert res["operations_applied"] == 3 and inj.active_intervention_count == 0
        fold = model(ids).logits
        for other in (vec, hk, fold):
            assert torch.allclose(ref, other, atol=1e-5)

    def test_unfoldable_specs_are_retained(self, llama, llama_layers):
        model = LlamaForCausalLM(llama.config).eval()
        model.load_state_dict(llama.state_dict())
        inj = NeuralInjector(model)
        inj.ablate_double_negative_regime([llama_layers[0]])
        inj.zero_ablate(llama_layers[1], 2, pole=PoleType.POSITIVE)
        inj.zero_ablate(llama_layers[1], 5)
        r = inj.fold_interventions_into_weights()
        assert r["operations_applied"] == 1 and r["unfolded_specs_retained"] == 2
        assert {s.mode for s in inj.get_active_interventions()} == {InterventionMode.DOUBLE_NEGATIVE_ABLATE, InterventionMode.ZERO_ABLATE}

    def test_preset_round_trip_with_gate_and_culling(self, llama, llama_layers, tmp_path):
        inj = NeuralInjector(llama, allowed_workspace_dir=tmp_path)
        inj.add_conditional_gate(llama_layers[2], [1], llama_layers[0], 3, threshold=0.01, value=0.4, sequence_scope="any_token")
        inj.add_intervention(InterventionSpec(llama_layers[1], [0, 1], InterventionMode.EXTREME_CULL, per_neuron_values=[0.1, 0.2], condition_signal="gate"))
        inj.export_production_preset(save_json_path=str(tmp_path / "p.json"))
        inj2 = NeuralInjector(llama, allowed_workspace_dir=tmp_path)
        inj2.load_production_preset(str(tmp_path / "p.json"))
        a = {(s.layer_name, s.mode, s.condition_signal) for s in inj.get_active_interventions()}
        b = {(s.layer_name, s.mode, s.condition_signal) for s in inj2.get_active_interventions()}
        assert a == b
        with pytest.raises(PermissionError):
            inj2.load_production_preset(str(tmp_path / ".." / "x.json"))

    def test_sticky_gate_latches_across_kv_cache_decode(self, llama, llama_layers):
        model = LlamaForCausalLM(llama.config).eval()
        model.load_state_dict(llama.state_dict())
        inj = NeuralInjector(model)
        ids = torch.randint(0, 97, (1, 8))
        A0, _ = _collect(model, llama_layers[0], [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
        thr = float(A0[:, 3].max()) * 0.99  # fires only at its strongest prompt token
        spec = inj.add_conditional_gate(llama_layers[2], [1], llama_layers[0], 3, threshold=thr, value=5.0,
                                        sequence_scope="any_token", sticky_on_decode=True)
        got = []
        hs = HookSet(model)
        hs.add_capture(llama_layers[2], lambda a, g, r: got.append(a[..., 1].clone()))
        with inj.active_injection(), hs.active():
            out = model(ids, use_cache=True)
            model(torch.tensor([[5]]), past_key_values=out.past_key_values, use_cache=True)
        clean = []
        hs2 = HookSet(model)
        hs2.add_capture(llama_layers[2], lambda a, g, r: clean.append(a[..., 1].clone()))
        with hs2.active():
            out = model(ids, use_cache=True)
            model(torch.tensor([[5]]), past_key_values=out.past_key_values, use_cache=True)
        assert torch.allclose(got[1] - clean[1], torch.full_like(clean[1], 5.0), atol=1e-5)


# =============================================================================
# 7. MoE experts
# =============================================================================
class TestMoE:
    def test_expert_audit_and_intervention(self):
        torch.manual_seed(0)
        m = MixtralForCausalLM(MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                                             num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                                             num_experts_per_tok=2)).eval()
        inj = NeuralInjector(m)
        layers = list(inj.list_targetable_layers())
        batches = [{"input_ids": torch.randint(0, 64, (2, 7)), "attention_mask": torch.ones(2, 7, dtype=torch.long)} for _ in range(3)]
        inj.audit_layers(layers, batches)
        routed = sum(inj.get_stats_tracker(l).count for l in layers)
        assert routed == 3 * 14 * 2  # every token visits top-2 experts
        x = batches[0]["input_ids"]
        base = m(x).logits
        with inj.active_injection():
            assert torch.equal(m(x).logits, base)
        inj.zero_ablate(layers[1], list(range(48)))
        with inj.active_injection():
            assert not torch.allclose(m(x).logits, base)
        inj.clear_interventions()
        pos = [torch.randint(0, 32, (1, 6)) for _ in range(3)]
        neg = [torch.randint(32, 64, (1, 6)) for _ in range(3)]
        rep = inj.discover_contrastive_neurons(pos, neg, layer_names=layers[:2], top_k=2)
        assert rep.tests_performed == 2 * 48 and all(mm.layer_name in layers[:2] for mm in rep.top_matches)


# =============================================================================
# 8. Explanation protocol
# =============================================================================
class ScriptedLLM:
    """Records every prompt; answers the explainer with a fixed explanation and the test writer with fixed texts."""

    def __init__(self, positives: List[str], near: List[str], distant: List[str], blind: str):
        self.calls: List[Dict[str, str]] = []
        self.positives, self.near, self.distant, self.blind = positives, near, distant, blind

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if "You write test texts" in system_prompt:
            return {"blind_maximum": self.blind, "positive_examples": self.positives, "near_misses": self.near,
                    "distant_concept": "volcanoes", "distant_examples": self.distant}
        return {"positive_pole_explanation": "cryptographic protocol terms", "negative_pole_explanation": "SQL query keywords",
                "unified_explanation": "routes crypto vs SQL", "recursive_offshoots": [],
                "positive_pole_taxonomy": "one_main_job", "negative_pole_taxonomy": "one_main_job", "confidence": 0.8}


class TestExplainer:
    @pytest.fixture()
    def demo(self) -> WorkbenchSession:
        return WorkbenchSession.create_demo_session()

    def test_requires_llm_client(self, demo):
        with pytest.raises(RuntimeError, match="No LLM client"):
            demo.explain_neuron("mlp_block_0.down_proj", 7, n_control_neurons=0)

    def test_two_agent_information_flow_and_outcome(self, demo):
        llm = ScriptedLLM(
            positives=["AES256 cipher", "TLS handshake", "the cipher", "AES256 TLS", "TLS cipher"],
            near=["the packet", "the server", "buffer latencies", "packet request", "the system"],
            distant=["the server", "packet", "request", "buffer", "latencies"],
            blind="AES256 TLS cipher handshake",
        )
        demo.injector.set_explainer_llm_client(llm)
        h = demo.explain_neuron("mlp_block_0.down_proj", 7, n_control_neurons=4)
        writer_calls = [c for c in llm.calls if "You write test texts" in c["system"]]
        assert len(writer_calls) == 2
        for c in writer_calls:
            payload = json.loads(c["user"])
            assert set(payload) == {"explanation_of_what_drives_the_neuron"}
            # the test writer never sees corpus data
            for leak in ("AES256", "SELECT", "postgres", "activation", "context"):
                assert leak not in c["user"]
        explainer_payload = json.loads(llm.calls[0]["user"])
        assert "positive_pole" in explainer_payload and "sign_regimes" in explainer_payload
        rep = h.verification["positive"]
        assert rep.strong_fraction == ni.PAPER_STRONG_FRACTION
        assert rep.positive_strong_count == 5 and rep.near_miss_strong_count == 0
        assert rep.outcome == FalsificationOutcome.WON
        for t in rep.positive_controls + rep.near_misses:
            assert t.share_of_corpus_extreme == pytest.approx(max(0.0, t.peak_activation) / demo.injector.get_stats_tracker("mlp_block_0.down_proj").max_val[7].item())
        assert h.causal_checks is not None and len(h.causal_checks.control_ablation_kls) == 4
        summ = demo.injector.ecology_map.summarize("mlp_block_0.down_proj")
        assert summ.poles_explained == 2 and summ.taxonomy_counts == {"one_main_job": 2}

    def test_wrong_explanation_loses(self, demo):
        llm = ScriptedLLM(positives=["the packet", "the server", "buffer", "request", "latencies"],
                          near=["AES256", "TLS", "cipher", "handshake", "AES256 cipher"],
                          distant=["packet"] * 5, blind="the server")
        demo.injector.set_explainer_llm_client(llm)
        h = demo.explain_neuron("mlp_block_0.down_proj", 7, n_control_neurons=0)
        assert h.verification["positive"].outcome == FalsificationOutcome.LOST
        assert h.verification["positive"].blind_max_ratio < ni.PAPER_STRONG_FRACTION

    def test_invalid_json_is_retried_then_rejected(self, demo):
        bad = CallableLLMClient(lambda s, u: {"positive_pole_explanation": "x"})
        demo.injector.set_explainer_llm_client(bad)
        with pytest.raises(RuntimeError, match="did not return valid"):
            demo.explain_neuron("mlp_block_0.down_proj", 7, n_control_neurons=0)

    def test_neighbours_only_from_traced_edges(self, demo):
        llm = ScriptedLLM(["AES256"] * 5, ["packet"] * 5, ["packet"] * 5, "AES256")
        demo.injector.set_explainer_llm_client(llm)
        assert demo.injector.ecology_map.get_neighbor_contexts("mlp_block_1.down_proj", 12) == []
        demo.explain_neuron("mlp_block_0.down_proj", 7, n_control_neurons=0)
        assert demo.injector.ecology_map.get_neighbor_contexts("mlp_block_1.down_proj", 12) == []  # no edge traced yet
        ids, _ = demo.encode_prompt("the system negotiated AES256 TLS cipher")
        demo.injector.trace_circuit("mlp_block_0.down_proj", "mlp_block_1.down_proj", 7, ids, compute_jacobian=False, top_k=4)
        nb = demo.injector.ecology_map.get_neighbor_contexts("mlp_block_1.down_proj", 12)
        assert len(nb) == 1 and nb[0].direction == "upstream" and nb[0].neuron_idx == 7
        h = demo.explain_neuron("mlp_block_1.down_proj", 12, n_control_neurons=0)
        assert h.neighbor_clues_used == 1
        assert "explained_neighbours" in json.loads(llm.calls[-3]["user"])


# =============================================================================
# 9. Workbench surfaces
# =============================================================================
class TestWorkbench:
    def test_scan_has_no_statistical_cleanliness_label(self):
        s = WorkbenchSession.create_demo_session()
        rep = s.injector.scan_layer("mlp_block_0.down_proj", limit=5)
        assert rep.explanation_audit.poles_explained == 0
        assert all(n.pos_taxonomy == NeuronTaxonomy.UNEXPLAINED for n in rep.neurons)
        assert rep.neurons[0].neuron_idx == 7  # the planted router writes the most

    def test_fastapi_endpoints(self):
        from fastapi.testclient import TestClient
        from neural_injector.server import create_workbench_app

        c = TestClient(create_workbench_app(WorkbenchSession.create_demo_session()))
        assert c.get("/api/layers").status_code == 200
        r = c.get("/api/layers/mlp_block_0.down_proj/scan?compute_kl_top_n=0&limit=3")
        assert r.status_code == 200 and "explanation_audit" in r.json()
        p = c.get("/api/layers/mlp_block_0.down_proj/neurons/7").json()
        assert p["positive_bands"] and p["regime_top_samples"]["++"]
        assert c.post("/api/interventions", json={"layer_name": "mlp_block_0.down_proj", "neuron_indices": [7], "mode": "zero_ablate"}).status_code == 200
        ev = c.post("/api/evaluate", json={"prompt": "the system negotiated AES256 TLS cipher"}).json()
        assert ev["kl_divergence"] > 0
        assert c.post("/api/explain", json={"layer_name": "mlp_block_0.down_proj", "neuron_idx": 7}).status_code == 400
        assert c.get("/").status_code == 200

    def test_cli_requires_explicit_audit(self, capsys, tmp_path):
        from neural_injector.cli import NeuralInjectorREPL
        s = WorkbenchSession.create_demo_session()
        s.injector._stats_trackers.clear()
        repl = NeuralInjectorREPL(s)
        repl.onecmd("scan mlp_block_0.down_proj")
        assert "has not been audited" in capsys.readouterr().out
        corpus = tmp_path / "c.txt"
        corpus.write_text("the system negotiated AES256 TLS cipher\n\nthe system executed SELECT WHERE postgres\n", encoding="utf-8")
        repl.onecmd(f"audit mlp_block_0.down_proj @{corpus}")
        repl.onecmd("scan mlp_block_0.down_proj")
        repl.onecmd("profile mlp_block_0.down_proj 7")
        out = capsys.readouterr().out
        assert "Audited 1 layer(s) on 2 prompt(s)" in out and "TOP POSITIVE" in out

    def test_thread_safety(self):
        s = WorkbenchSession.create_demo_session()
        s.injector.zero_ablate("mlp_block_0.down_proj", 7)
        errs = []

        def work():
            try:
                for _ in range(5):
                    s.evaluate_prompt_comparison("the system negotiated AES256 TLS cipher")
            except Exception as e:  # pragma: no cover
                errs.append(e)

        ts = [threading.Thread(target=work) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errs

    def test_checkpoint_evolution(self):
        s = WorkbenchSession.create_demo_session()
        ids, toks = s.encode_prompt("the system negotiated AES256 TLS cipher")
        batches = [{"input_ids": ids, "tokens_per_sample": [toks]}]
        rep = s.injector.track_checkpoint_evolution([("a", s.model), ("b", s.model)], "mlp_block_0.down_proj", 7, batches,
                                                    forward_fn=lambda m, b: m(b["input_ids"]))
        assert len(rep.snapshots) == 2 and rep.snapshots[0].weight_cosine_to_final == pytest.approx(1.0)
