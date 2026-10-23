from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from neural_injector import (
    AgentHypothesisResponseSchema,
    BipolarVerbHypothesis,
    InteractionRole,
    InterventionKernel,
    InterventionMode,
    InterventionSpec,
    LayerCleanlinessReport,
    LayerDiscovery,
    NeuralInjector,
    NeuronTaxonomy,
    OpenAICompatibleClient,
    PoleType,
    ProductionSteeringRuntime,
    SteeringExporter,
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
        x = torch.tensor([[[2.5, -3.0, 1.0], [-1.5, 4.0, -0.5]]])

        # Ablate ONLY the positive pole of neuron 0
        spec_pos = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[0],
            mode=InterventionMode.ZERO_ABLATE,
            pole=PoleType.POSITIVE,
        )
        out_pos = InterventionKernel.apply(x, [spec_pos])
        assert out_pos[0, 0, 0].item() == 0.0
        assert out_pos[0, 1, 0].item() == -1.5

        # Ablate ONLY the negative pole of neuron 1
        spec_neg = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[1],
            mode=InterventionMode.ZERO_ABLATE,
            pole=PoleType.NEGATIVE,
        )
        out_neg = InterventionKernel.apply(x, [spec_neg])
        assert out_neg[0, 0, 1].item() == 0.0
        assert out_neg[0, 1, 1].item() == 4.0

    def test_kernel_weak_activation_pruning(self):
        # Tests Cammarata (2026) "Do Weak Activations Matter?" filter
        x = torch.tensor([[0.2, -0.4, 3.5, -4.2]])
        spec = InterventionSpec(
            layer_name="dummy",
            neuron_indices=[0, 1, 2, 3],
            mode=InterventionMode.WEAK_PRUNE,
            pole=PoleType.BOTH,
            value=1.0,
        )
        out = InterventionKernel.apply(x, [spec])
        assert torch.allclose(out, torch.tensor([[0.0, 0.0, 3.5, -4.2]]))

    def test_kernel_token_position_selective_clamping(self):
        x = torch.ones((1, 4, 6))
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
        x = torch.ones((2, 8, 4, 4))
        spec = InterventionSpec(
            layer_name="conv",
            neuron_indices=[3],
            mode=InterventionMode.SCALE,
            pole=PoleType.BOTH,
            value=0.0,
        )
        out = InterventionKernel.apply(x, [spec], is_conv2d=True)
        assert torch.allclose(out[:, 3, :, :], torch.zeros_like(out[:, 3, :, :]))
        assert torch.allclose(out[:, 0, :, :], torch.ones_like(out[:, 0, :, :]))

    def test_live_model_zero_ablation_and_kl_divergence(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        injector = demo_session.injector
        layer = "mlp_block_0.down_proj"

        kl_active_neuron = injector.measure_ablation_kl_divergence(
            layer, neuron_idx=7, inputs=sample_prompt_ids
        )
        assert kl_active_neuron > 0.05
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
        assert sum(report.pole_taxonomy_counts.values()) == 256

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
        assert edge_12.jacobian_sensitivity != 0.0
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
        # adding delta to neuron 7 prior to linear down_proj
        assert torch.allclose(hook_logits, compiled_vec_logits, atol=1e-4)

    def test_hard_clamp_compiled_vector_exact_equivalence(
        self,
        demo_session: WorkbenchSession,
        sample_prompt_ids: torch.Tensor,
        tmp_path: Path,
    ):
        """
        Verifies Bug Fix 1B: HARD_CLAMP exported to a compiled residual vector records
        `clamped_neurons` and zeroes incoming dynamic activations before adding `v_steer`,
        matching hook-based HARD_CLAMP instead of turning into ADDITIVE_STEER.
        """
        injector = demo_session.injector
        layer = "mlp_block_0.down_proj"
        injector.clear_interventions()
        injector.clamp_neurons(layer, neuron_indices=[7], value=-15.0)

        with torch.no_grad(), injector.active_injection():
            expected_clamp_logits = demo_session.model(sample_prompt_ids)

        json_file = tmp_path / "clamp_preset.json"
        bundle = injector.export_production_preset(
            preset_name="clamp_preset",
            model_name=demo_session.model_name,
            save_json_path=str(json_file),
        )
        assert len(bundle.compiled_vectors) == 1
        assert bundle.compiled_vectors[0].clamped_neurons == [7]

        injector.clear_interventions()
        runtime = ProductionSteeringRuntime(demo_session.model, json_file)
        with torch.no_grad(), runtime.active(use_compiled_residual_vectors=True):
            compiled_clamp_logits = demo_session.model(sample_prompt_ids)

        assert torch.allclose(expected_clamp_logits, compiled_clamp_logits, atol=1e-4)

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
    def test_fastapi_endpoints_end_to_end(
        self, demo_session: WorkbenchSession, tmp_path: Path
    ):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from neural_injector.server import create_workbench_app

        app = create_workbench_app(demo_session, allowed_workspace_dir=tmp_path)
        client = TestClient(app)

        # 1. GET / (HTML Workbench UI)
        ui_resp = client.get("/")
        assert ui_resp.status_code == 200
        assert "Neural Injector Workbench" in ui_resp.text

        # 2. GET /api/layers
        layers_resp = client.get("/api/layers")
        assert layers_resp.status_code == 200
        assert len(layers_resp.json()["layers"]) >= 2

        # 3. GET /api/layers/{layer_name}/scan
        scan_resp = client.get(
            "/api/layers/mlp_block_0.down_proj/scan?sort_by=estimated_causal_impact&limit=8"
        )
        assert scan_resp.status_code == 200
        assert scan_resp.json()["num_neurons"] == 128

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
        assert preset_bundle["compiled_vectors"][0]["clamped_neurons"] == [7]

        # Verify path traversal protection returns HTTP 403
        bad_exp_resp = client.post(
            "/api/export/preset",
            json={
                "preset_name": "malicious_preset",
                "save_json_path": "../escaped_preset.json",
            },
        )
        assert bad_exp_resp.status_code == 403

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


# =============================================================================
# 9. Regression Tests for Audit Fixes & Cammarata (2026) Section 3 Mechanisms
# =============================================================================
class TestAuditFixesAndCammarataExtensions:
    def test_online_m4_kurtosis_matches_exact_full_batch(self):
        """
        Verifies that streaming multiple batches with shifting means produces the
        exact same 2nd, 3rd, and 4th central moments (M2, M3, M4) as a single full pass,
        as well as exact conditional positive and negative pole moments.
        """
        torch.manual_seed(99)
        b1 = torch.randn(32, 8) * 1.5 - 2.0
        b2 = torch.randn(48, 8) * 3.0 + 4.5
        b3 = torch.randn(20, 8) * 0.5 + 1.0
        full_data = torch.cat([b1, b2, b3], dim=0)

        streaming_stats = StreamingLayerStats("test_layer", num_neurons=8, top_k=5)
        for batch in (b1, b2, b3):
            streaming_stats.update(batch)

        exact_stats = StreamingLayerStats("test_layer", num_neurons=8, top_k=5)
        exact_stats.update(full_data)

        assert torch.allclose(streaming_stats.mean, exact_stats.mean, atol=1e-6)
        assert torch.allclose(streaming_stats.m2, exact_stats.m2, atol=1e-5)
        assert torch.allclose(streaming_stats.m4, exact_stats.m4, atol=1e-4)
        assert torch.allclose(streaming_stats.pos_mean, exact_stats.pos_mean, atol=1e-6)
        assert torch.allclose(streaming_stats.neg_mean, exact_stats.neg_mean, atol=1e-6)
        assert torch.allclose(streaming_stats.pos_m4, exact_stats.pos_m4, atol=1e-4)
        assert torch.allclose(streaming_stats.neg_m4, exact_stats.neg_m4, atol=1e-4)

    def test_non_destructive_tracing_and_kl_preserve_active_interventions(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        """
        Verifies that running `measure_ablation_kl_divergence`, `trace_circuit`,
        or `clean_forward` does NOT wipe out previously registered user interventions.
        """
        injector = demo_session.injector
        layer = "mlp_block_0.down_proj"
        injector.clear_interventions()
        injector.steer_neurons(layer, neuron_indices=[3], delta=5.0)
        assert injector.active_intervention_count == 1

        with injector.clean_forward():
            _ = demo_session.model(sample_prompt_ids)

        _ = injector.measure_ablation_kl_divergence(layer, neuron_idx=7, inputs=sample_prompt_ids)
        _ = injector.trace_circuit(
            src_layer="mlp_block_0.down_proj",
            dst_layer="mlp_block_1.down_proj",
            src_neuron=7,
            inputs=sample_prompt_ids,
        )

        active = injector.get_active_interventions()
        assert len(active) == 1
        assert active[0].neuron_indices == [3]
        assert active[0].value == 5.0

    def test_track_all_neurons_beyond_index_64(self, demo_session: WorkbenchSession):
        """
        Verifies that neurons at index >= 64 (e.g. Neuron #110) still collect
        top-k positive and negative pole samples even when tracked_neurons=None.
        """
        profile_110 = demo_session.injector.get_neuron_profile(
            layer_name="mlp_block_0.down_proj",
            neuron_idx=110,
        )
        assert len(profile_110.top_positive_samples) > 0 or len(profile_110.top_negative_samples) > 0

    def test_section_3_1_pole_specific_cleanliness_auditing(self):
        """
        Verifies Cammarata Section 3.1: Evaluating Positive and Negative poles independently.
        Constructs a neuron whose Positive Pole is a sharp monosemantic spike while its
        Negative Pole is uniform dense noise, verifying independent pole taxonomies.
        """
        torch.manual_seed(123)
        tracker = StreamingLayerStats("asymmetric_layer", num_neurons=2, top_k=5)
        acts = torch.zeros((200, 2))
        # Neuron 0: Positive pole fires sharply on only 2 tokens; Negative pole is flat noise on 150 tokens
        acts[0, 0] = 45.0
        acts[1, 0] = 42.0
        acts[2:152, 0] = -torch.rand(150) * 0.5 - 0.5

        tracker.update(acts)
        prof = tracker.export_profile(0)

        assert prof.pos_pole_selectivity_score > 2.2
        assert prof.pos_taxonomy_class == NeuronTaxonomy.CLEAN_MONOSEMANTIC
        assert prof.neg_pole_selectivity_score < prof.pos_pole_selectivity_score

    def test_section_3_2_crossword_circuit_ecology_solver(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        """
        Verifies Cammarata Section 3.2 ('Understanding Compounds'):
        `solve_layer_ecology` explains high-selectivity anchor neurons first, stores
        them in `CircuitEcologyMap`, and propagates neighbor clues into connected neurons.
        """
        injector = demo_session.injector
        ecology_report = injector.solve_layer_ecology(
            layer_name="mlp_block_0.down_proj",
            tok_to_id=demo_session.tok_to_id,
            idx_to_token=demo_session.id_to_tok,
            unembed_weight=demo_session.unembed_weight,
            dst_layer="mlp_block_1.down_proj",
            circuit_inputs=sample_prompt_ids,
            top_anchors=2,
            top_contextual=2,
        )
        assert len(ecology_report.anchor_neurons_explained) == 2
        assert len(ecology_report.hypotheses) >= 2
        assert ecology_report.mean_verification_score > 0.0

        # Now explain downstream Layer 1 Neuron #12 and verify it receives Layer 0 upstream ecology clues
        neighbors = injector.ecology_map.get_neighbor_contexts("mlp_block_1.down_proj", 12)
        assert any(n.direction == "upstream" for n in neighbors)

        hyp_l1 = injector.explain_neuron(
            layer_name="mlp_block_1.down_proj",
            neuron_idx=12,
            tok_to_id=demo_session.tok_to_id,
            idx_to_token=demo_session.id_to_tok,
            unembed_weight=demo_session.unembed_weight,
        )
        assert hyp_l1.neuron_idx == 12

    def test_section_3_3_stratified_quantile_activation_buckets(self):
        """
        Verifies Cammarata Section 3.3: Stratified Quantile Activation Buckets capture
        `peak_trunk` ([80%..100%]), `secondary_branch` ([40%..80%)), and
        `tertiary_offshoot` ([15%..40%)) samples per pole.
        """
        tracker = StreamingLayerStats("quantile_layer", num_neurons=1, top_k=4)
        # Create activations spanning all three quantile bands of max=10.0:
        # 9.5 -> peak_trunk (95%), 6.0 -> secondary_branch (60%), 2.5 -> tertiary_offshoot (25%)
        acts = torch.tensor([[9.5], [6.0], [2.5], [-8.0], [-5.0], [-2.0]], dtype=torch.float32)
        tokens = [["iy"], ["diy"], ["do_it_yourself"], ["SELECT"], ["FROM"], ["db_table"]]
        contexts = ["ctx_peak", "ctx_sec", "ctx_tert", "neg_peak", "neg_sec", "neg_tert"]

        tracker.update(
            acts,
            sample_ids=[0, 1, 2, 3, 4, 5],
            tokens_per_sample=tokens,
            contexts_per_sample=contexts,
        )
        prof = tracker.export_profile(0)

        pos_b = prof.positive_quantile_buckets
        neg_b = prof.negative_quantile_buckets

        assert any(s.token_str == "iy" for s in pos_b["peak_trunk"])
        assert any(s.token_str == "diy" for s in pos_b["secondary_branch"])
        assert any(s.token_str == "do_it_yourself" for s in pos_b["tertiary_offshoot"])

        assert any(s.token_str == "SELECT" for s in neg_b["peak_trunk"])
        assert any(s.token_str == "FROM" for s in neg_b["secondary_branch"])
        assert any(s.token_str == "db_table" for s in neg_b["tertiary_offshoot"])

    def test_section_3_4_native_per_neuron_adaptive_thresholds(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        """
        Verifies Cammarata Section 3.4: Respecting Native Activation Ranges & Percentiles
        via per-neuron adaptive thresholds in `prune_weak_activations` and `InterventionKernel`.
        """
        injector = demo_session.injector
        layer = "mlp_block_0.down_proj"
        injector.clear_interventions()

        spec_std = injector.prune_weak_activations(
            layer, neuron_indices=[0, 7, 12], std_multiplier=0.8
        )
        assert spec_std.per_neuron_values is not None
        assert len(spec_std.per_neuron_values) == 3
        # Neuron #7 has much larger activation scale than background Neuron #0
        assert spec_std.per_neuron_values[1] > spec_std.per_neuron_values[0]

        injector.clear_interventions()
        spec_pct = injector.prune_weak_activations(
            layer, neuron_indices=[0, 7, 12], percentile=50.0
        )
        assert spec_pct.per_neuron_values is not None
        assert len(spec_pct.per_neuron_values) == 3

        with torch.no_grad(), injector.active_injection():
            out = demo_session.model(sample_prompt_ids)
        assert out.shape[-1] == len(demo_session.tok_to_id)
        injector.clear_interventions()

    def test_section_3_5_verify_downstream_readout_superposition_audit(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        """
        Verifies Cammarata Section 3.5 ('Nothing Reads That Combination'):
        `verify_downstream_readout` distinguishes benign weak spillover across background
        neurons from causally read primary circuit neurons.
        """
        injector = demo_session.injector
        src_layer = "mlp_block_0.down_proj"
        dst_layer = "mlp_block_1.down_proj"

        # 1. Weak activations on unaligned background neurons [0, 1, 2] with small threshold
        benign_report = injector.verify_downstream_readout(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neurons=[0, 1, 2],
            inputs=sample_prompt_ids,
            weak_threshold=0.05,
            readout_tolerance=0.25,
        )
        assert benign_report.is_benign_spillover is True
        assert len(benign_report.affected_downstream_neurons) == 0

        # 2. Pruning strong primary router Neuron #7 with high threshold triggers downstream shift
        active_report = injector.verify_downstream_readout(
            src_layer=src_layer,
            dst_layer=dst_layer,
            src_neurons=[7],
            inputs=sample_prompt_ids,
            weak_threshold=50.0,
            readout_tolerance=0.1,
        )
        assert active_report.is_benign_spillover is False
        assert 12 in active_report.affected_downstream_neurons


# =============================================================================
# 10. Universal & Model-Agnostic Architecture Tests (Section 2)
# =============================================================================
class TestUniversalModelAgnosticSupport:
    def test_gpt2_conv1d_discovery_and_attn_c_proj_disambiguation(self):
        class Conv1D(nn.Module):
            def __init__(self, nf: int, nx: int):
                super().__init__()
                self.nf = nf
                self.weight = nn.Parameter(torch.randn(nx, nf) * 0.02)
                self.bias = nn.Parameter(torch.zeros(nf))

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                size_out = inp.size()[:-1] + (self.nf,)
                out_t = torch.addmm(self.bias, inp.view(-1, inp.size(-1)), self.weight)
                return out_t.view(size_out)

        class MiniGPT2Block(nn.Module):
            def __init__(self, d_model: int = 32, d_mlp: int = 64):
                super().__init__()
                self.attn = nn.ModuleDict({"c_proj": Conv1D(d_model, d_model)})
                self.mlp = nn.ModuleDict({
                    "c_fc": Conv1D(d_mlp, d_model),
                    "c_proj": Conv1D(d_model, d_mlp),
                })

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                hidden_states = hidden_states + self.attn["c_proj"](hidden_states)
                h = torch.relu(self.mlp["c_fc"](hidden_states))
                return hidden_states + self.mlp["c_proj"](h)

        model = MiniGPT2Block()
        inj = NeuralInjector(model)
        layers = inj.list_targetable_layers(include_all_linear=False)

        assert "mlp.c_proj" in layers
        assert "attn.c_proj" not in layers
        assert layers["mlp.c_proj"].num_neurons == 64

        sample_x = torch.randn(2, 4, 32)
        inj.steer_neurons("mlp.c_proj", neuron_indices=[5, 10], delta=4.0)
        with torch.no_grad(), inj.active_injection():
            hook_out = model(sample_x)

        fold_res = inj.fold_interventions_into_weights()
        assert fold_res["operations_applied"] == 1
        with torch.no_grad():
            folded_out = model(sample_x)
        assert torch.allclose(hook_out, folded_out, atol=1e-5)

    def test_fused_gate_up_proj_and_3d_moe_virtual_weights(self):
        """
        Verifies that models with a single fused `gate_up_proj` (`[2 * d_mlp, d_model]`)
        or 3D MoE fused `gate_up_proj` (`[num_experts, 2 * d_mlp, d_model]`) properly
        resolve virtual gate and up weights without shape mismatch errors.
        """
        class FusedSwiGLUMLP(nn.Module):
            def __init__(self, d_model: int = 32, d_mlp: int = 64):
                super().__init__()
                self.gate_up_proj = nn.Linear(d_model, 2 * d_mlp, bias=False)
                self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
                return self.down_proj(torch.nn.functional.silu(gate) * up)

        class FusedTwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm_0 = nn.LayerNorm(32)
                self.mlp_block_0 = FusedSwiGLUMLP(32, 64)
                self.norm_1 = nn.LayerNorm(32)
                self.mlp_block_1 = FusedSwiGLUMLP(32, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.mlp_block_0(self.norm_0(x))
                return x + self.mlp_block_1(self.norm_1(x))

        model = FusedTwoLayerModel()
        # Verify resolve_pre_mlp_norm maps mlp_block_0 -> norm_0 and mlp_block_1 -> norm_1
        assert LayerDiscovery.resolve_pre_mlp_norm(model, "mlp_block_0.down_proj") is model.norm_0
        assert LayerDiscovery.resolve_pre_mlp_norm(model, "mlp_block_1.down_proj") is model.norm_1

        inj = NeuralInjector(model)
        v_up, v_gate, _ = inj.tracer.compute_virtual_weights(
            "mlp_block_0.down_proj", "mlp_block_1.down_proj", src_neuron=3
        )
        assert v_up.shape == (64,)
        assert v_gate.shape == (64,)
        assert v_up.abs().sum().item() > 0.0
        assert v_gate.abs().sum().item() > 0.0

    def test_half_precision_bfloat16_audit_and_circuit_trace(self, demo_session: WorkbenchSession):
        """
        Verifies that `bfloat16` models run auditing, logit attribution, and circuit
        tracing without dtype mismatch errors.
        """
        bf16_model = copy.deepcopy(demo_session.model).to(dtype=torch.bfloat16)
        inj = NeuralInjector(bf16_model)
        ids = torch.tensor([[1, 2, 3, 5, 6]], dtype=torch.long)

        inj.audit_layers(
            layer_names=["mlp_block_0.down_proj"],
            dataloader_or_batches=[{"input_ids": ids}],
            forward_fn=lambda m, b: m(b["input_ids"]),
        )
        prof = inj.get_neuron_profile(
            "mlp_block_0.down_proj",
            neuron_idx=7,
            unembed_weight=bf16_model.lm_head.weight,
            idx_to_token=demo_session.id_to_tok,
        )
        assert len(prof.top_promoted_tokens) > 0

        graph = inj.trace_circuit(
            src_layer="mlp_block_0.down_proj",
            dst_layer="mlp_block_1.down_proj",
            src_neuron=7,
            inputs=ids,
            compute_jacobian=True,
        )
        assert len(graph.edges) > 0


# =============================================================================
# 11. Production Hardening & Section 5 Frontier Capabilities Tests
# =============================================================================
class TestProductionHardeningAndSection5Capabilities:
    def test_section_4_1_thread_safety_under_concurrent_evaluations(
        self, demo_session: WorkbenchSession
    ):
        demo_session.injector.clear_interventions()
        demo_session.injector.steer_neurons("mlp_block_0.down_proj", neuron_indices=[7], delta=-8.0)

        def _worker(_idx: int) -> float:
            res = demo_session.evaluate_prompt_comparison("the system negotiated AES256")
            return float(res["kl_divergence"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            kls = list(pool.map(_worker, range(8)))

        assert all(abs(k - kls[0]) < 1e-5 for k in kls)
        assert demo_session.injector.active_hook_count == 0
        demo_session.injector.clear_interventions()

    def test_section_4_2_path_traversal_security(self, tmp_path: Path):
        with pytest.raises(PermissionError):
            SteeringExporter.validate_safe_export_path(
                "../outside.json", allowed_workspace_dir=tmp_path
            )
        outside = tmp_path.parent / "escaped_file.json"
        with pytest.raises(PermissionError):
            SteeringExporter.validate_safe_export_path(
                outside, allowed_workspace_dir=tmp_path
            )
        safe = SteeringExporter.validate_safe_export_path(
            tmp_path / "subdir" / "ok.json", allowed_workspace_dir=tmp_path
        )
        assert safe.is_relative_to(tmp_path.resolve())

    def test_section_4_3_pydantic_schema_validation_and_retry(self, monkeypatch: pytest.MonkeyPatch):
        attempts = {"count": 0}

        class _FakeHTTPResponse:
            def __init__(self, body_str: str):
                self._bytes = body_str.encode("utf-8")

            def read(self) -> bytes:
                return self._bytes

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _fake_urlopen(_req: Any, **_kwargs: Any):
            attempts["count"] += 1
            if attempts["count"] == 1:
                # Malformed JSON on attempt 1 -> triggers retry
                return _FakeHTTPResponse('{"choices": [{"message": {"content": "NOT_JSON"}}]}')
            valid_payload = {
                "positive_pole_motif": "Crypto TLS",
                "negative_pole_motif": "SQL SELECT",
                "unified_verb_hypothesis": "Routes crypto vs SQL.",
                "recursive_offshoots": ["TLS branch"],
                "taxonomy_class": "clean_monosemantic",
                "confidence_score": 0.91,
                "proposed_experiment": {"positive_pole_test_tokens": ["AES256"]},
            }
            return _FakeHTTPResponse(
                json.dumps({"choices": [{"message": {"content": json.dumps(valid_payload)}}]})
            )

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        client = OpenAICompatibleClient(max_retries=3, backoff_base=0.01)
        out = client.complete_json("sys", "user")
        assert attempts["count"] == 2
        assert out["confidence_score"] == 0.91
        assert (
                AgentHypothesisResponseSchema.model_validate(out).taxonomy_class
                == NeuronTaxonomy.CLEAN_MONOSEMANTIC
        )

    def test_section_5_1_forward_mode_ad_single_pass_circuit_tracer(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        jac_map = demo_session.injector.trace_forward_ad_all_layers(
            src_layer="mlp_block_0.down_proj",
            src_neuron=7,
            inputs=sample_prompt_ids,
            src_pole=PoleType.POSITIVE,
        )
        assert "mlp_block_1.down_proj" in jac_map
        jac_l1 = jac_map["mlp_block_1.down_proj"]
        assert jac_l1.shape == (128,)
        # L0 Neuron #7 (+Pole) strongly excites L1 Neuron #12 and inhibits L1 Neuron #19
        assert jac_l1[12].item() > 0.1
        assert jac_l1[19].item() < -0.1

    def test_section_5_2_closed_loop_conditional_concept_gating(
        self,
        demo_session: WorkbenchSession,
        sample_prompt_ids: torch.Tensor,
        sql_prompt_ids: torch.Tensor,
    ):
        injector = demo_session.injector
        injector.clear_interventions()

        # Only steer Layer 1 Neuron #12 by +20.0 when Detector Neuron #7 in Layer 0 fires positively (> 1.0)
        injector.add_conditional_gate(
            target_layer="mlp_block_1.down_proj",
            target_neurons=[12],
            detector_layer="mlp_block_0.down_proj",
            detector_neuron=7,
            detector_pole=PoleType.POSITIVE,
            threshold=1.0,
            action_mode=InterventionMode.ADDITIVE_STEER,
            value=20.0,
            sequence_scope="any_token",
        )

        with torch.no_grad():
            with injector.without_interventions():
                clean_crypto = demo_session.model(sample_prompt_ids)
                clean_sql = demo_session.model(sql_prompt_ids)
            with injector.active_injection():
                gated_crypto = demo_session.model(sample_prompt_ids)
                gated_sql = demo_session.model(sql_prompt_ids)

        # Crypto prompt triggers L0 N7 (+Pole) -> gate opens -> output shifts
        assert not torch.allclose(clean_crypto, gated_crypto, atol=1e-3)
        # SQL prompt drives L0 N7 negatively -> gate stays closed -> zero collateral damage!
        assert torch.allclose(clean_sql, gated_sql, atol=1e-6)
        injector.clear_interventions()

    def test_section_5_3_contrastive_pole_discovery(
        self,
        demo_session: WorkbenchSession,
        sample_prompt_ids: torch.Tensor,
        sql_prompt_ids: torch.Tensor,
    ):
        report = demo_session.injector.discover_contrastive_neurons(
            pos_inputs=sample_prompt_ids,
            neg_inputs=sql_prompt_ids,
            layer_names=["mlp_block_0.down_proj"],
            top_k=5,
        )
        assert len(report.top_matches) == 5
        top_match = report.top_matches[0]
        assert top_match.neuron_idx == 7
        assert top_match.bipolar_sign_flip is True
        assert top_match.t_statistic > 1.5

    def test_section_5_4_cross_checkpoint_recursive_growth_tracker(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        ckpt_early = copy.deepcopy(demo_session.model)
        ckpt_final = copy.deepcopy(demo_session.model)
        with torch.no_grad():
            # Simulate early checkpoint having weaker gate/up weights on Neuron #7
            ckpt_early.mlp_block_0.up_proj.weight[7] *= 0.25

        batches = [
            {
                "input_ids": sample_prompt_ids,
                "sample_ids": [0],
                "tokens_per_sample": [["the", "system", "negotiated", "AES256", "TLS", "cipher"]],
                "contexts_per_sample": ["the system negotiated AES256 TLS cipher"],
            }
        ]
        evo_report = demo_session.injector.track_checkpoint_evolution(
            checkpoints=[("epoch_1", ckpt_early), ("epoch_5", ckpt_final)],
            layer_name="mlp_block_0.down_proj",
            neuron_idx=7,
            batches=batches,
            forward_fn=lambda m, b: m(b["input_ids"]),
        )
        assert len(evo_report.snapshots) == 2
        assert evo_report.snapshots[0].checkpoint_label == "epoch_1"
        assert evo_report.snapshots[1].checkpoint_label == "epoch_5"
        assert evo_report.snapshots[1].std > evo_report.snapshots[0].std

    def test_section_5_5_synergistic_multi_neuron_ablation(
        self, demo_session: WorkbenchSession, sample_prompt_ids: torch.Tensor
    ):
        injector = demo_session.injector
        res_greedy = injector.find_synergistic_ablation_set(
            layer_name="mlp_block_0.down_proj",
            inputs=sample_prompt_ids,
            max_subset_size=2,
            candidate_pool_size=8,
            method="greedy",
        )
        assert len(res_greedy.selected_neurons) == 2
        assert 7 in res_greedy.selected_neurons
        assert res_greedy.joint_ablation_kl > 0.05

        res_l1 = injector.find_synergistic_ablation_set(
            layer_name="mlp_block_0.down_proj",
            inputs=sample_prompt_ids,
            max_subset_size=2,
            candidate_pool_size=8,
            method="l1_optim",
        )
        assert len(res_l1.selected_neurons) == 2
        assert res_l1.joint_ablation_kl > 0.0
