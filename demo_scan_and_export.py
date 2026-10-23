from pathlib import Path
import tempfile

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
    # PART 1: Batch Layer-Wide & Pole-Specific Cleanliness Scan (Section 3.1)
    # =========================================================================
    eval_ids, _ = session.encode_prompt("the system negotiated AES256 TLS cipher")
    scan_report = session.injector.scan_layer_cleanliness(
        layer_name=target_layer,
        sort_by="estimated_causal_impact",
        compute_kl_for_top_n=3,
        kl_eval_inputs=eval_ids,
        limit=6,
    )

    print("=== 1. BATCH LAYER-WIDE & POLE-SPECIFIC CLEANLINESS SCAN REPORT ===")
    print(
        f"Layer: {scan_report.layer_name} | Total Neurons: {scan_report.num_neurons} | Tokens Scanned: {scan_report.tokens_scanned}"
    )
    print(
        f"Whole Neurons -> Clean: {scan_report.clean_monosemantic_pct:.1f}% | "
        f"Structured: {scan_report.structured_recursive_pct:.1f}% | "
        f"Combined: {scan_report.combined_clean_and_structured_pct:.1f}%"
    )
    print(
        f"Neuron Poles  -> Clean Poles: {scan_report.clean_poles_pct:.1f}% | "
        f"Structured Poles: {scan_report.structured_poles_pct:.1f}% | "
        f"Combined Poles: {scan_report.combined_clean_and_structured_poles_pct:.1f}%"
    )
    print("\nTop 6 Neurons Ranked by Estimated Causal Impact:")
    for n in scan_report.neurons:
        kl_s = f"{n.ablation_kl_divergence:.4f}" if n.ablation_kl_divergence is not None else "N/A"
        print(
            f"  Neuron #{n.neuron_idx:<3} | Class: {n.taxonomy_class.value:<28} | "
            f"+Sel/-Sel: {n.pos_pole_selectivity_score:4.1f}/{n.neg_pole_selectivity_score:4.1f} | "
            f"Impact: {n.estimated_causal_impact:6.3f} | Exact Ablation KL: {kl_s}"
        )

    # =========================================================================
    # PART 2: Synergistic Multi-Neuron Group Ablation (Section 5.5)
    # =========================================================================
    synergy = session.injector.find_synergistic_ablation_set(
        layer_name=target_layer,
        inputs=eval_ids,
        max_subset_size=3,
        method="greedy",
    )
    print("\n=== 2. SYNERGISTIC MULTI-NEURON GROUP ABLATION (Section 5.5) ===")
    print(
        f"  Selected Coalition: {synergy.selected_neurons} | "
        f"Joint KL: {synergy.joint_ablation_kl:.4f} | "
        f"Sum Individual KLs: {synergy.sum_of_individual_kls:.4f} | "
        f"Synergy Ratio: {synergy.synergy_ratio:.3f}"
    )

    # =========================================================================
    # PART 3: Export Production Steering Preset & Compiled Residual Vector
    # =========================================================================
    print("\n=== 3. EXPORTING PRODUCTION STEERING PRESET & COMPILED VECTOR ===")
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
            allowed_workspace_dir=tmpdir,
        )

        cv = bundle.compiled_vectors[0]
        print(f"Saved JSON Preset : {json_path.name} ({json_path.stat().st_size} bytes)")
        print(
            f"Saved .pt Vectors : {pt_path.name} (Layer: {cv.layer_name}, d_model={cv.d_model}, ||v_steer||_2={cv.vector_norm:.4f})"
        )

        # Detach workbench hooks and test standalone ProductionSteeringRuntime
        session.injector.clear_interventions()
        runtime = ProductionSteeringRuntime(session.model, json_path)

        sql_token_ids = [
            session.tok_to_id["SELECT"],
            session.tok_to_id["WHERE"],
            session.tok_to_id["postgres"],
        ]

        with torch.no_grad():
            clean_probs = F.softmax(session.model(eval_ids)[0, -1, :], dim=-1)
            clean_sql_prob = clean_probs[sql_token_ids].sum().item()

            # Mode A: Fast compiled residual-stream vector addition (output + v_steer)
            with runtime.active(use_compiled_residual_vectors=True):
                vec_probs = F.softmax(session.model(eval_ids)[0, -1, :], dim=-1)
                vec_sql_prob = vec_probs[sql_token_ids].sum().item()

            session.injector.load_production_preset(str(json_path), allowed_workspace_dir=tmpdir)
            fold_report = session.injector.fold_interventions_into_weights()
            folded_probs = F.softmax(session.model(eval_ids)[0, -1, :], dim=-1)
            folded_sql_prob = folded_probs[sql_token_ids].sum().item()

        print("\n=== 4. PRODUCTION RUNTIME & WEIGHT-FOLDING VERIFICATION ===")
        print(f"  Clean Unmodified Model SQL Probability            : {clean_sql_prob:.4f}")
        print(f"  ProductionSteeringRuntime (Compiled Vector Mode)  : {vec_sql_prob:.4f}")
        print(
            f"  Zero-Overhead Weight-Folded Model (0 Hooks Active): {folded_sql_prob:.4f} ({fold_report['details'][0]['folded_op']})"
        )


if __name__ == "__main__":
    run_scan_and_export_demo()
