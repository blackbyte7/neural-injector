from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from neural_injector import (
    BipolarVerbHypothesis,
    InteractionRole,
    InterventionKernel,
    InterventionMode,
    InterventionSpec,
    LayerCleanlinessReport,
    LayerDiscovery,
    NeuralInjector,
    NeuronTaxonomy,
    PoleType,
    ProductionSteeringRuntime,
    StreamingLayerStats,
    WorkbenchSession,
)
from neural_injector.cli import NeuralInjectorREPL, main as cli_main


# =============================================================================
# Shared Pytest Fixtures
# =============================================================================
@pytest.fixture
def demo_session() -> WorkbenchSession:
    """Returns a fresh, deterministically seeded 2-layer SwiGLU WorkbenchSession."""
    return WorkbenchSession.create_demo_session()


@pytest.fixture
def sample_prompt_ids(demo_session: WorkbenchSession) -> torch.Tensor:
    """Returns tokenized input_ids for a cryptography prompt that excites Layer 0 Neuron #7 (+Pole)."""
    ids, _ = demo_session.encode_prompt("the system negotiated AES256 TLS cipher")
    return ids


@pytest.fixture
def sql_prompt_ids(demo_session: WorkbenchSession) -> torch.Tensor:
    """Returns tokenized input_ids for an SQL prompt that drives Layer 0 Neuron #7 (-Pole)."""
    ids, _ = demo_session.encode_prompt("the system executed SELECT WHERE postgres")
    return ids


# =============================================================================
# 1. Layer Discovery & Architecture Resolution Tests
# =============================================================================
class TestLayerDiscovery:
    def test_discover_swiglu_down_proj_layers(self, demo_session: WorkbenchSession):
        layers = demo_session.injector.list_targetable_layers(include_all_linear=False)
        assert "mlp_block_0.down_proj" in layers
        assert "mlp_block_1.down_proj" in layers

        desc0 = layers["mlp_block_0.down_proj"]
        assert desc0.num_neurons == 128
        assert desc0.hook_target == "input"
        assert "Pre-DownProj" in desc0.module_type

    def test_discover_all_linear_and_conv2d_fallback(self):
        class VisionCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.head = nn.Linear(16, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.conv1(x)).mean(dim=(-2, -1))
                return self.head(x)

        cnn = VisionCNN()
        discovered = LayerDiscovery.discover_neuron_layers(cnn)
        assert "conv1" in discovered
        assert discovered["conv1"].num_neurons == 16
        assert discovered["conv1"].hook_target == "output"
        assert "head" in discovered
        assert discovered["head"].num_neurons == 4

    def test_resolve_mlp_read_write_siblings(self, demo_session: WorkbenchSession):
        write_mod, up_mod, gate_mod = LayerDiscovery.resolve_mlp_read_write_modules(
            demo_session.model, "mlp_block_0.down_proj"
        )
        assert isinstance(write_mod, nn.Linear)
        assert isinstance(up_mod, nn.Linear)
        assert isinstance(gate_mod, nn.Linear)
        assert write_mod.in_features == up_mod.out_features == gate_mod.out_features == 128


# =============================================================================
# 2. Vectorized Intervention Kernel & Surgical Hook Tests
# =============================================================================
class TestSurgicalInterventions:
    def test_kernel_pole_specific_zero_ablation(self):
        x = torch.tensor([[[2.5, -3.0, 1.0], [-1.5, 4.0, -0.5]]])  # shape [1, 2, 3]

        # Ablate ONLY the positive pole of neuron 0
        spec_pos = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[0],
            mode=InterventionMode.ZERO_ABLATE,
            pole=PoleType.POSITIVE,
        )
        out_pos = InterventionKernel.apply(x, [spec_pos])
        assert out_pos[0, 0, 0].item() == 0.0
        assert out_pos[0, 1, 0].item() == -1.5  # Negative pole preserved

        # Ablate ONLY the negative pole of neuron 1
        spec_neg = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[1],
            mode=InterventionMode.ZERO_ABLATE,
            pole=PoleType.NEGATIVE,
        )
        out_neg = InterventionKernel.apply(x, [spec_neg])
        assert out_neg[0, 0, 1].item() == 0.0  # Negative pole zeroed
        assert out_neg[0, 1, 1].item() == 4.0  # Positive pole preserved

    def test_kernel_weak_activation_pruning(self):
        # Tests Cammarata (2026) "Do Weak Activations Matter?" filter
        x = torch.tensor([[0.2, -0.4, 3.5, -4.2]])
        spec = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[0, 1, 2, 3],
            mode=InterventionMode.WEAK_PRUNE,
            pole=PoleType.BOTH,
            value=1.0,  # Threshold |a| < 1.0 -> 0.0
        )
        out = InterventionKernel.apply(x, [spec])
        assert torch.allclose(out, torch.tensor([[0.0, 0.0, 3.5, -4.2]]))

    def test_kernel_token_position_selective_clamping(self):
        x = torch.ones((1, 4, 6))  # [batch=1, seq=4, dim=6]
        spec = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[2, 5],
            mode=InterventionMode.HARD_CLAMP,
            pole=PoleType.BOTH,
            value=-9.0,
            token_positions=[1, 3],
        )
        out = InterventionKernel.apply(x, [spec])
        # Positions 1 and 3 at neurons 2 and 5 should be -9.0; positions 0 and 2 remain 1.0
        assert out[0, 1, 2].item() == -9.0
        assert out[0, 3, 5].item() == -9.0
        assert out[0, 0, 2].item() == 1.0
        assert out[0, 2, 5].item() == 1.0

    def test_kernel_conv2d_channel_intervention(self):
        x = torch.ones((2, 8, 4, 4))  # [B, C, H, W]
        spec = InterventionSpec(
            layer_name="conv",
            neuron_indices=[3],
            mode=InterventionMode.SCALE,
            pole=PoleType.BOTH,
            value=0.0,
        )
        out = InterventionKernel.apply(x, [spec], is_conv2d=True)
        assert torch.all(out[:, 3, :, :] == 0.0)
        assert torch.all(out[:, 0, :, :] == 1.0)

    def test_live_model_zero_ablation_and_kl_divergence(
            self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        injector = demo_session.injector
        layer = "mlp_block_0.down_proj"

        kl_active_neuron = injector.measure_ablation_kl_divergence(layer, neuron_idx=7, inputs=sample_prompt_ids)
        assert kl_active_neuron > 0.05  # Neuron #7 carries strong causal signal on Crypto tokens

        # Verify context manager cleanly detaches hooks after exit
        assert len(injector._active_handles) == 0


# =============================================================================
# 3. Bipolar Neuron Auditing & Direct Logit Attribution Tests
# =============================================================================
class TestBipolarNeuronAuditing:
    def test_bipolar_profile_captures_opposing_poles(self, demo_session: WorkbenchSession):
        profile = demo_session.injector.get_neuron_profile(
            layer_name="mlp_block_0.down_proj",
            neuron_idx=7,
            unembed_weight=demo_session.unembed_weight,
            idx_to_token=demo_session.id_to_tok,
        )
        assert profile.neuron_idx == 7
        assert profile.max_val > 0.0
        assert profile.min_val < 0.0
        assert profile.positive_firing_rate > 0.0
        assert profile.negative_firing_rate > 0.0

        # Top positive pole samples should contain Cryptography tokens
        pos_tokens = {s.token_str for s in profile.top_positive_samples}
        assert pos_tokens.intersection({"AES256", "TLS", "cipher", "handshake"})

        # Top negative pole samples should contain SQL tokens
        neg_tokens = {s.token_str for s in profile.top_negative_samples}
        assert neg_tokens.intersection({"SELECT", "WHERE", "postgres", "index_scan"})

        # Direct vocabulary attribution should promote Crypto tokens on +Pole and SQL on -Pole
        promoted_words = {tok for tok, _ in profile.top_promoted_tokens[:4]}
        suppressed_words = {tok for tok, _ in profile.top_suppressed_tokens[:4]}
        assert promoted_words.intersection({"AES256", "TLS", "cipher", "handshake"})
        assert suppressed_words.intersection({"SELECT", "WHERE", "postgres", "index_scan"})


# =============================================================================
# 4. Batch Layer-Wide Cleanliness Scanning Tests
# =============================================================================
class TestLayerCleanlinessScanner:
    def test_scan_layer_cleanliness_metrics_and_ranking(
            self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        report: LayerCleanlinessReport = demo_session.injector.scan_layer_cleanliness(
            layer_name="mlp_block_0.down_proj",
            sort_by="estimated_causal_impact",
            compute_kl_for_top_n=2,
            kl_eval_inputs=sample_prompt_ids,
            limit=10,
        )
        assert report.num_neurons == 128
        assert report.tokens_scanned > 0
        assert len(report.neurons) == 10

        # Taxonomy percentages must sum to ~100%
        total_pct = sum(report.taxonomy_percentages.values())
        assert abs(total_pct - 100.0) < 0.1
        assert abs(
            report.combined_clean_and_structured_pct
            - (report.clean_monosemantic_pct + report.structured_recursive_pct)
        ) < 0.05

        # Neuron #7 was calibrated with strong outgoing norm & selectivity, so it should rank #1 in impact
        top_neuron = report.neurons[0]
        assert top_neuron.neuron_idx == 7
        assert top_neuron.ablation_kl_divergence is not None
        assert top_neuron.ablation_kl_divergence > 0.05

    def test_dead_neuron_detection(self):
        tracker = StreamingLayerStats("test_layer", num_neurons=4, top_k=3)
        # Feed batch where neuron 0 is strictly 0.0 (dead) and neuron 1 spikes sharply
        batch = torch.zeros((20, 4))
        batch[0, 1] = 50.0
        tracker.update(batch)

        counts, _, summaries = tracker.scan_layer_cleanliness()
        by_idx = {s.neuron_idx: s for s in summaries}
        assert by_idx[0].taxonomy_class == NeuronTaxonomy.DEAD_OR_NOISY
        assert by_idx[1].taxonomy_class == NeuronTaxonomy.CLEAN_MONOSEMANTIC
        assert counts[NeuronTaxonomy.DEAD_OR_NOISY.value] >= 1


# =============================================================================
# 5. Layer-to-Layer Circuit Tracer Tests
# =============================================================================
class TestCircuitTracer:
    def test_virtual_weights_and_causal_pole_tracing(
            self,
            demo_session: WorkbenchSession,
            sample_prompt_ids: torch.Tensor,
            sql_prompt_ids: torch.Tensor,
    ):
        injector = demo_session.injector
        src_layer = "mlp_block_0.down_proj"
        dst_layer = "mlp_block_1.down_proj"

        # 1. Trace Positive Pole on Crypto input: should excite L1 Neuron #12 and inhibit L1 Neuron #19
        pos_graph = injector.trace_circuit(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neuron=7,
            inputs=sample_prompt_ids,
            src_pole=PoleType.POSITIVE,
            compute_jacobian=True,
            top_k=10,
        )
        excited_ids = [e.dst_neuron for e in pos_graph.top_excitatory(3)]
        inhibited_ids = [e.dst_neuron for e in pos_graph.top_inhibitory(3)]
        assert 12 in excited_ids
        assert 19 in inhibited_ids

        edge_12 = next(e for e in pos_graph.edges if e.dst_neuron == 12)
        assert edge_12.mean_causal_delta > 0.5
        assert edge_12.virtual_weight_up > 0.0
        assert edge_12.role == InteractionRole.EXCITATORY

        # 2. Trace Negative Pole on SQL input: should excite L1 Neuron #19
        neg_graph = injector.trace_circuit(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neuron=7,
            inputs=sql_prompt_ids,
            src_pole=PoleType.NEGATIVE,
            compute_jacobian=True,
            top_k=10,
        )
        neg_excited_ids = [e.dst_neuron for e in neg_graph.top_excitatory(3)]
        assert 19 in neg_excited_ids

        # 3. Verify serialization for LLM agent
        payload = pos_graph.to_agent_payload()
        assert payload["source"]["neuron_idx"] == 7
        assert len(payload["top_excited_downstream"]) > 0

    def test_multi_hop_circuit_tracing(
            self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        graphs = demo_session.injector.tracer.trace_multi_hop(
            layer_sequence=["mlp_block_0.down_proj", "mlp_block_1.down_proj"],
            root_neuron=7,
            root_pole=PoleType.POSITIVE,
            inputs=sample_prompt_ids,
            beam_width=2,
        )
        assert len(graphs) == 1
        assert graphs[0].src_neuron == 7


# =============================================================================
# 6. Automated LLM-Agent Explainer & Closed-Loop Surgical Verification Tests
# =============================================================================
class TestAgenticNeuronExplainer:
    def test_closed_loop_explain_and_verify_with_heuristic_client(
            self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        hypothesis: BipolarVerbHypothesis = demo_session.injector.explain_neuron(
            layer_name="mlp_block_0.down_proj",
            neuron_idx=7,
            tok_to_id=demo_session.tok_to_id,
            idx_to_token=demo_session.id_to_tok,
            unembed_weight=demo_session.unembed_weight,
            dst_layer="mlp_block_1.down_proj",
            circuit_inputs=sample_prompt_ids,
        )
        assert hypothesis.neuron_idx == 7
        assert hypothesis.taxonomy_class in (
            NeuronTaxonomy.CLEAN_MONOSEMANTIC,
            NeuronTaxonomy.STRUCTURED_RECURSIVE_MIXTURE,
        )
        assert "verb" in hypothesis.unified_verb_hypothesis.lower()
        assert hypothesis.verification_report is not None

        rep = hypothesis.verification_report
        assert rep.pole_separation_passed is True
        assert rep.positive_test_mean_activation > 0.0
        assert rep.negative_test_mean_activation < 0.0
        assert rep.zero_ablation_target_logit_drop > 0.0
        assert rep.pole_flip_target_logit_shift > 0.0
        assert rep.downstream_circuit_match_ratio >= 0.5
        assert rep.empirical_verification_score >= 0.8

    def test_custom_mock_llm_client_protocol(
            self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        class MockLLMClient:
            def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
                data = json.loads(user_prompt)
                assert "neuron_profile" in data
                assert "downstream_circuit" in data
                return {
                    "positive_pole_motif": "Cryptography & TLS cipher negotiation",
                    "negative_pole_motif": "Relational SQL query execution",
                    "unified_verb_hypothesis": "Routes between cryptographic security (+Pole) and SQL queries (-Pole).",
                    "recursive_offshoots": ["TLS handshake specialization in Layer 1 Neuron #12"],
                    "taxonomy_class": "clean_monosemantic",
                    "confidence_score": 0.96,
                    "proposed_experiment": {
                        "positive_pole_test_tokens": ["AES256", "TLS", "cipher"],
                        "negative_pole_test_tokens": ["SELECT", "WHERE", "postgres"],
                        "expected_promoted_tokens": ["AES256", "TLS"],
                        "expected_suppressed_tokens": ["SELECT", "postgres"],
                        "expected_excited_dst_neurons": [12],
                        "expected_inhibited_dst_neurons": [19],
                    },
                }

        demo_session.injector.explainer.llm_client = MockLLMClient()
        hyp = demo_session.injector.explain_neuron(
            layer_name="mlp_block_0.down_proj",
            neuron_idx=7,
            tok_to_id=demo_session.tok_to_id,
            idx_to_token=demo_session.id_to_tok,
            unembed_weight=demo_session.unembed_weight,
            dst_layer="mlp_block_1.down_proj",
            circuit_inputs=sample_prompt_ids,
        )
        assert hyp.confidence_score == 0.96
        assert hyp.verification_report is not None
        assert hyp.verification_report.empirical_verification_score == 1.0


# =============================================================================
# 7. Production Steering Export, Runtime & Weight Folding Equivalence Tests
# =============================================================================
class TestProductionSteeringExportAndWeightFolding:
    def test_export_import_and_compiled_vector_equivalence(
            self,
            demo_session: WorkbenchSession,
            sample_prompt_ids: torch.Tensor,
            tmp_path: Path,
    ):
        injector = demo_session.injector
        layer = "mlp_block_0.down_proj"
        injector.clear_interventions()
        injector.steer_neurons(layer, neuron_indices=[7], delta=-12.5)

        # 1. Capture logits using standard hook-based NeuralInjector
        with torch.no_grad(), injector.active_injection():
            hook_logits = demo_session.model(sample_prompt_ids)

        # 2. Export preset to JSON and .pt files
        json_file = tmp_path / "preset.json"
        pt_file = tmp_path / "vectors.pt"
        bundle = injector.export_production_preset(
            preset_name="test_preset",
            model_name=demo_session.model_name,
            save_json_path=str(json_file),
            save_pt_vectors_path=str(pt_file),
        )
        assert json_file.exists()
        assert pt_file.exists()
        assert len(bundle.compiled_vectors) == 1
        assert bundle.compiled_vectors[0].d_model == 64

        # 3. Detach injector and apply standalone ProductionSteeringRuntime (compiled vector mode)
        injector.clear_interventions()
        runtime = ProductionSteeringRuntime(demo_session.model, json_file)
        with torch.no_grad(), runtime.active(use_compiled_residual_vectors=True):
            compiled_vec_logits = demo_session.model(sample_prompt_ids)

        # Mathematical check: adding delta * W_down[:, 7] in residual space must match
        # adding delta to neuron 7 prior to linear down_proj!
        assert torch.allclose(hook_logits, compiled_vec_logits, atol=1e-4)

    @pytest.mark.parametrize(
        "mode,val",
        [
            (InterventionMode.ZERO_ABLATE, 0.0),
            (InterventionMode.SCALE, 2.5),
            (InterventionMode.ADDITIVE_STEER, -10.0),
            (InterventionMode.HARD_CLAMP, 8.0),
        ],
    )
    def test_weight_folding_exact_numerical_equivalence(
            self,
            demo_session: WorkbenchSession,
            sample_prompt_ids: torch.Tensor,
            mode: InterventionMode,
            val: float,
    ):
        """
        Verifies that folding ZERO_ABLATE, SCALE, ADDITIVE_STEER, or HARD_CLAMP
        into `down_proj` weights/biases produces identical model outputs to active hooks
        while running with zero forward hooks attached.
        """
        model_copy = copy.deepcopy(demo_session.model)
        inj = NeuralInjector(model_copy)
        layer = "mlp_block_0.down_proj"

        inj.add_intervention(
            InterventionSpec(
                layer_name=layer,
                neuron_indices=[7, 12],
                mode=mode,
                pole=PoleType.BOTH,
                value=val,
            )
        )

        with torch.no_grad(), inj.active_injection():
            expected_hook_logits = model_copy(sample_prompt_ids)

        # Now fold permanently into weights/biases (which also clears all hooks)
        fold_res = inj.fold_interventions_into_weights()
        assert fold_res["operations_applied"] == 1
        assert len(inj._active_handles) == 0

        with torch.no_grad():
            folded_logits = model_copy(sample_prompt_ids)

        assert torch.allclose(expected_hook_logits, folded_logits, atol=1e-4)


# =============================================================================
# 8. FastAPI Endpoints & CLI / REPL Workbench Tests
# =============================================================================
class TestFastAPIAndCLIWorkbench:
    def test_fastapi_endpoints_end_to_end(self, demo_session: WorkbenchSession):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from neural_injector.server import create_workbench_app

        app = create_workbench_app(demo_session)
        client = TestClient(app)

        # 1. GET / (HTML Workbench UI)
        ui_resp = client.get("/")
        assert ui_resp.status_code == 200
        assert "Neural Injector Workbench" in ui_resp.text

        # 2. GET /api/layers
        layers_resp = client.get("/api/layers")
        assert layers_resp.status_code == 200
        layers_data = layers_resp.json()
        assert len(layers_data["layers"]) >= 2

        # 3. GET /api/layers/{layer_name}/scan
        scan_resp = client.get("/api/layers/mlp_block_0.down_proj/scan?sort_by=estimated_causal_impact&limit=8")
        assert scan_resp.status_code == 200
        scan_data = scan_resp.json()
        assert scan_data["num_neurons"] == 128
        assert len(scan_data["neurons"]) == 8

        # 4. GET /api/layers/{layer_name}/neurons/{neuron_idx}
        prof_resp = client.get("/api/layers/mlp_block_0.down_proj/neurons/7")
        assert prof_resp.status_code == 200
        assert prof_resp.json()["neuron_idx"] == 7

        # 5. POST /api/interventions & POST /api/evaluate
        int_resp = client.post(
            "/api/interventions",
            json={
                "layer_name": "mlp_block_0.down_proj",
                "neuron_indices": [7],
                "mode": "hard_clamp",
                "pole": "both",
                "value": -15.0,
            },
        )
        assert int_resp.status_code == 200
        assert int_resp.json()["active_count"] == 1

        eval_resp = client.post(
            "/api/evaluate",
            json={"prompt": "the system negotiated AES256", "top_k_logits": 4},
        )
        assert eval_resp.status_code == 200
        assert eval_resp.json()["kl_divergence"] > 0.01

        # 6. POST /api/export/preset & POST /api/import/preset
        exp_resp = client.post(
            "/api/export/preset",
            json={"preset_name": "api_test_preset"},
        )
        assert exp_resp.status_code == 200
        preset_bundle = exp_resp.json()
        assert len(preset_bundle["compiled_vectors"]) == 1

        del_resp = client.delete("/api/interventions")
        assert del_resp.status_code == 200
        assert del_resp.json()["active_count"] == 0

        imp_resp = client.post(
            "/api/import/preset",
            json={"preset_payload": preset_bundle, "replace_existing": True},
        )
        assert imp_resp.status_code == 200
        assert imp_resp.json()["active_interventions"] == 1

        # 7. POST /api/circuits/trace
        trace_resp = client.post(
            "/api/circuits/trace",
            json={
                "src_layer": "mlp_block_0.down_proj",
                "dst_layer": "mlp_block_1.down_proj",
                "src_neuron": 7,
                "src_pole": "positive",
                "prompt": "the system negotiated AES256 TLS",
                "compute_jacobian": True,
                "top_k": 5,
            },
        )
        assert trace_resp.status_code == 200
        assert trace_resp.json()["source"]["neuron_idx"] == 7

        # 8. POST /api/explain
        explain_resp = client.post(
            "/api/explain",
            json={
                "layer_name": "mlp_block_0.down_proj",
                "neuron_idx": 7,
                "dst_layer": "mlp_block_1.down_proj",
                "circuit_prompt": "the system negotiated AES256 TLS cipher",
            },
        )
        assert explain_resp.status_code == 200
        assert explain_resp.json()["verification_report"]["pole_separation_passed"] is True

    def test_cli_repl_commands_and_entrypoints(
            self, demo_session: WorkbenchSession, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        repl = NeuralInjectorREPL(demo_session)

        repl.do_layers("")
        repl.do_scan("mlp_block_0.down_proj estimated_causal_impact 5")
        repl.do_profile("mlp_block_0.down_proj 7")
        repl.do_clamp("mlp_block_0.down_proj 7 -12.0")
        repl.do_eval("the system negotiated AES256")

        preset_file = tmp_path / "cli_preset.json"
        repl.do_export(str(preset_file))
        assert preset_file.exists()

        repl.do_clear("")
        repl.do_import(str(preset_file))
        repl.do_trace("mlp_block_0.down_proj mlp_block_1.down_proj 7 positive")
        repl.do_explain("mlp_block_0.down_proj 7 mlp_block_1.down_proj")

        captured = capsys.readouterr().out
        assert "LAYER-WIDE CLEANLINESS SCAN" in captured
        assert "NEURON PROFILE: mlp_block_0.down_proj [Neuron #7]" in captured
        assert "CIRCUIT TRACE" in captured
        assert "AGENTIC EXPLANATION" in captured

        # One-shot CLI entrypoint test
        cli_main(["profile", "mlp_block_0.down_proj", "7"])
        out_oneshot = capsys.readouterr().out
        assert "Neuron #7" in out_oneshot
