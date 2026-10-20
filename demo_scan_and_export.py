import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_injector.exporter import ProductionSteeringRuntime
from neural_injector.session import WorkbenchSession


def run_scan_and_export_demo():
    # 1. Initialize pre-calibrated 2-layer SwiGLU session (audits all 128 neurons)
    session = WorkbenchSession.create_demo_session()
    target_layer = "mlp_block_0.down_proj"
    default_prompts = [
        "the system negotiated AES256 TLS cipher",
        "the system executed SELECT WHERE postgres",
        "packet buffer TLS handshake AES256 cipher",
        "packet buffer postgres index_scan SELECT WHERE",
        "the server executed packet buffer latencies",
    ]
    session.run_audit(
        layer_names=[target_layer],
        prompts=default_prompts,
        tracked_neurons=list(range(128)),
    )

    # =========================================================================
    # PART 1: Batch Layer-Wide Cleanliness Scan
    # =========================================================================
    eval_ids, _ = session.encode_prompt("the system negotiated AES256 TLS cipher")
    scan_report = session.injector.scan_layer_cleanliness(
        layer_name=target_layer,
        sort_by="estimated_causal_impact",
        compute_kl_for_top_n=3,
        kl_eval_inputs=eval_ids,
        limit=6,
    )

    print("=== 1. BATCH LAYER-WIDE CLEANLINESS SCAN REPORT ===")
    print(
        f"Layer: {scan_report.layer_name} | Total Neurons: {scan_report.num_neurons} | Tokens Scanned: {scan_report.tokens_scanned}")
    print(
        f"Clean Monosemantic: {scan_report.clean_monosemantic_pct:.1f}% | "
        f"Structured Recursive: {scan_report.structured_recursive_pct:.1f}% | "
        f"Combined Clean+Structured: {scan_report.combined_clean_and_structured_pct:.1f}%"
    )
    print("\nTop 6 Neurons Ranked by Estimated Causal Impact:")
    for n in scan_report.neurons:
        kl_s = f"{n.ablation_kl_divergence:.4f}" if n.ablation_kl_divergence is not None else "N/A"
        print(
            f"  Neuron #{n.neuron_idx:<3} | Class: {n.taxonomy_class.value:<28} | "
            f"Selectivity: {n.pole_selectivity_score:5.2f} | Bipolar: {n.bipolar_balance:4.2f} | "
            f"Impact: {n.estimated_causal_impact:6.3f} | Exact Ablation KL: {kl_s}"
        )

    # =========================================================================
    # PART 2: Export Production Steering Preset & Compiled Residual Vector
    # =========================================================================
    print("\n=== 2. EXPORTING PRODUCTION STEERING PRESET & COMPILED VECTOR ===")
    session.injector.clear_interventions()
    # Steer Neuron #7 strongly toward its Negative Pole (-15.0) to flip Crypto -> SQL
    session.injector.steer_neurons(target_layer, neuron_indices=[7], delta=-15.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "crypto_to_sql_preset.json"
        pt_path = Path(tmpdir) / "crypto_to_sql_vectors.pt"

        bundle = session.injector.export_production_preset(
            preset_name="crypto_to_sql_steering",
            model_name=session.model_name,
            save_json_path=str(json_path),
            save_pt_vectors_path=str(pt_path),
        )

        cv = bundle.compiled_vectors[0]
        print(f"Saved JSON Preset : {json_path.name} ({json_path.stat().st_size} bytes)")
        print(
            f"Saved .pt Vectors : {pt_path.name} (Layer: {cv.layer_name}, d_model={cv.d_model}, ||v_steer||_2={cv.vector_norm:.4f})")

        # Detach workbench hooks and test standalone ProductionSteeringRuntime
        session.injector.clear_interventions()
        runtime = ProductionSteeringRuntime(session.model, json_path)

        sql_token_ids = [session.tok_to_id["SELECT"], session.tok_to_id["WHERE"], session.tok_to_id["postgres"]]

        with torch.no_grad():
            clean_probs = F.softmax(session.model(eval_ids)[0, -1, :], dim=-1)
            clean_sql_prob = clean_probs[sql_token_ids].sum().item()

            # Mode A: Fast compiled residual-stream vector addition (output + v_steer)
            with runtime.active(use_compiled_residual_vectors=True):
                vec_probs = F.softmax(session.model(eval_ids)[0, -1, :], dim=-1)
                vec_sql_prob = vec_probs[sql_token_ids].sum().item()

            # Mode B: Zero-overhead permanent weight folding into down_proj.bias
            session.injector.load_production_preset(str(json_path))
            fold_report = session.injector.fold_interventions_into_weights()
            folded_probs = F.softmax(session.model(eval_ids)[0, -1, :], dim=-1)
            folded_sql_prob = folded_probs[sql_token_ids].sum().item()

        print("\n=== 3. PRODUCTION RUNTIME & WEIGHT-FOLDING VERIFICATION ===")
        print(f"  Clean Unmodified Model SQL Probability            : {clean_sql_prob:.4f}")
        print(f"  ProductionSteeringRuntime (Compiled Vector Mode)  : {vec_sql_prob:.4f}")
        print(
            f"  Zero-Overhead Weight-Folded Model (0 Hooks Active): {folded_sql_prob:.4f} ({fold_report['details'][0]['folded_op']})")


if __name__ == "__main__":
    run_scan_and_export_demo()
