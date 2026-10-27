from __future__ import annotations

import argparse
import cmd
import shlex
from typing import List, Optional

from .session import WorkbenchSession
from .types import InterventionMode, PoleType


def _split_cli_args(arg: str) -> List[str]:
    """
    Splits CLI arguments while preserving Windows backslash path separators (`\\`).
    Using `posix=False` prevents `shlex.split` from stripping `\\` out of Windows paths.
    """
    return [tok.strip("\"'") for tok in shlex.split(arg, posix=False) if tok.strip("\"'")]


class NeuralInjectorREPL(cmd.Cmd):
    """
    Stateful interactive terminal workbench for auditing neurons, manipulating poles,
    tracing downstream & incoming circuits, solving crossword circuit ecology, discovering
    contrastive neurons, and running closed-loop agent explanations.
    """

    intro = (
        "\n======================================================================\n"
        "  NEURAL INJECTOR INTERACTIVE WORKBENCH (REPL)\n"
        "  Type 'help' or '?' to list commands. Type 'layers' to get started.\n"
        "======================================================================\n"
    )
    prompt = "neural-injector> "

    def __init__(self, session: WorkbenchSession):
        super().__init__()
        self.session = session

    def do_layers(self, _arg: str) -> None:
        """layers : List all discovered targetable neuron layers in the model."""
        layers = self.session.injector.list_targetable_layers()
        print(f"\nModel: {self.session.model_name} | Vocab Size: {len(self.session.tok_to_id)}")
        print(f"{'LAYER NAME':<28} | {'TYPE':<32} | {'NEURONS':<8} | {'HOOK':<6}")
        print("-" * 82)
        for name, desc in layers.items():
            print(
                f"{name:<28} | {desc.module_type:<32} | {desc.num_neurons:<8} | {desc.hook_target:<6}"
            )
        print()

    def do_scan(self, arg: str) -> None:
        """scan <layer_name> [sort_by] [top_n] : Run a batch layer-wide & pole-specific cleanliness audit."""
        parts = _split_cli_args(arg)
        if not parts:
            print(
                "Usage: scan <layer_name> [pole_selectivity_score|estimated_causal_impact|bipolar_balance] [top_n]"
            )
            return
        layer_name = parts[0]
        sort_by = parts[1] if len(parts) >= 2 else "pole_selectivity_score"
        limit = int(parts[2]) if len(parts) >= 3 else 12

        kl_inputs = self.session.get_default_eval_inputs()
        report = self.session.injector.scan_layer_cleanliness(
            layer_name=layer_name,
            sort_by=sort_by,
            compute_kl_for_top_n=3,
            kl_eval_inputs=kl_inputs,
            limit=limit,
        )

        print(
            f"\n=== LAYER-WIDE CLEANLINESS SCAN: {report.layer_name} ({report.num_neurons} neurons / {2 * report.num_neurons} poles) ==="
        )
        print(
            f"Whole-Neuron Clean : {report.clean_monosemantic_pct:.1f}% | "
            f"Structured Recursive: {report.structured_recursive_pct:.1f}% | "
            f"Combined: {report.combined_clean_and_structured_pct:.1f}%"
        )
        print(
            f"Independent Poles  : {report.clean_poles_pct:.1f}% Clean | "
            f"Structured Poles: {report.structured_poles_pct:.1f}% | "
            f"Combined Poles: {report.combined_clean_and_structured_poles_pct:.1f}%"
        )
        print(
            f"\n{'IDX':<5} | {'TAXONOMY':<28} | {'SEL (+ / -)':<19} | {'BIPOLAR':<7} | {'IMPACT':<8} | {'KL_ABL':<8} | {'+TOK / -TOK'}"
        )
        print("-" * 106)
        for n in report.neurons:
            kl_str = (
                f"{n.ablation_kl_divergence:.4f}"
                if n.ablation_kl_divergence is not None
                else "  -   "
            )
            sel_pair = f"{n.pole_selectivity_score:4.1f} ({n.pos_pole_selectivity_score:4.1f}/{n.neg_pole_selectivity_score:4.1f})"
            tok_pair = f"{n.top_pos_token or '-'} / {n.top_neg_token or '-'}"
            tax_label = f"{n.taxonomy_class.value}{'*' if n.semantic_verified else ''}"
            print(
                f"N{n.neuron_idx:<4} | {tax_label:<28} | {sel_pair:<19} | "
                f"{n.bipolar_balance:<7.3f} | {n.estimated_causal_impact:<8.3f} | {kl_str:<8} | {tok_pair}"
            )
        print()

    def do_profile(self, arg: str) -> None:
        """profile <layer_name> <neuron_idx> : Inspect Positive & Negative poles, Next-Token clues, SwiGLU Double-Negative rate, Quantile Bands, and Logit Attribution."""
        parts = _split_cli_args(arg)
        if len(parts) < 2:
            print("Usage: profile <layer_name> <neuron_idx>  (e.g., profile mlp_block_0.down_proj 7)")
            return
        layer_name, neuron_idx = parts[0], int(parts[1])
        try:
            p = self.session.injector.get_neuron_profile(
                layer_name=layer_name,
                neuron_idx=neuron_idx,
                unembed_weight=self.session.unembed_weight,
                idx_to_token=self.session.id_to_tok,
            )
            print(f"\n--- NEURON PROFILE: {p.layer_name} [Neuron #{p.neuron_idx}] ---")
            print(
                f"Mean: {p.mean:+.4f} | Std: {p.std:.4f} | Overall Selectivity: {p.pole_selectivity_score:.2f} | Clean Candidate: {p.is_clean_candidate}"
            )
            print(
                f"+Pole Selectivity: {p.pos_pole_selectivity_score:.2f} ({p.pos_taxonomy_class.value}) | "
                f"-Pole Selectivity: {p.neg_pole_selectivity_score:.2f} ({p.neg_taxonomy_class.value}) | "
                f"SwiGLU Double-Neg (+Pole): {p.double_negative_positive_rate * 100:.1f}%"
            )
            print(
                f"\n[+] POSITIVE POLE (Firing Rate: {p.positive_firing_rate * 100:.1f}%, Max: {p.max_val:+.3f}):"
            )
            for s in p.top_positive_samples[:4]:
                nxt = f" -> '{s.next_token_str}'" if s.next_token_str else ""
                dbl = " [SwiGLU g<0,u<0]" if s.is_double_negative_route else ""
                print(
                    f"    Act: {s.activation_value:+.3f}{dbl} | Token: '{s.token_str}'{nxt} | Context: \"{s.context_str}\""
                )
            for band, b_samples in p.positive_quantile_buckets.items():
                toks = [s.token_str for s in b_samples[:4] if s.token_str]
                if toks:
                    print(f"    [Band: {band:<17}] Tokens: {toks}")

            print(
                f"\n[-] NEGATIVE POLE (Firing Rate: {p.negative_firing_rate * 100:.1f}%, Min: {p.min_val:+.3f}):"
            )
            for s in p.top_negative_samples[:4]:
                nxt = f" -> '{s.next_token_str}'" if s.next_token_str else ""
                print(
                    f"    Act: {s.activation_value:+.3f} | Token: '{s.token_str}'{nxt} | Context: \"{s.context_str}\""
                )
            print(f"\nDirect Vocab Attribution (+Pole Promotes) : {p.top_promoted_tokens[:5]}")
            print(f"Direct Vocab Attribution (-Pole Promotes) : {p.top_suppressed_tokens[:5]}\n")
        except Exception as exc:
            print(f"Error: {exc}")

    def do_ablate(self, arg: str) -> None:
        """ablate <layer_name> <neuron_idx> [both|positive|negative] : Surgically set a neuron or pole to 0.0."""
        parts = _split_cli_args(arg)
        if len(parts) < 2:
            print("Usage: ablate <layer_name> <neuron_idx> [both|positive|negative]")
            return
        layer_name, idx = parts[0], int(parts[1])
        pole = PoleType(parts[2]) if len(parts) >= 3 else PoleType.BOTH
        self.session.injector.zero_ablate(layer_name, neuron_indices=idx, pole=pole)
        print(
            f"Registered ZERO_ABLATE on {layer_name}[{idx}] (pole={pole.value}). Run 'eval <prompt>' to test."
        )

    def do_clamp(self, arg: str) -> None:
        """clamp <layer_name> <neuron_idx> <value> [both|positive|negative] : Force a neuron to a constant value."""
        parts = _split_cli_args(arg)
        if len(parts) < 3:
            print("Usage: clamp <layer_name> <neuron_idx> <value> [both|positive|negative]")
            return
        layer_name, idx, val = parts[0], int(parts[1]), float(parts[2])
        pole = PoleType(parts[3]) if len(parts) >= 4 else PoleType.BOTH
        self.session.injector.clamp_neurons(layer_name, neuron_indices=idx, value=val, pole=pole)
        print(f"Registered HARD_CLAMP={val:+.2f} on {layer_name}[{idx}] (pole={pole.value}).")

    def do_prune(self, arg: str) -> None:
        """prune <layer_name> <threshold_or_std_mult> [adaptive] : Zero out weak activations across the layer."""
        parts = _split_cli_args(arg)
        if len(parts) < 2:
            print("Usage: prune <layer_name> <threshold> [adaptive]")
            return
        layer_name, val = parts[0], float(parts[1])
        use_adaptive = len(parts) >= 3 and parts[2].lower() == "adaptive"
        layers = self.session.injector.list_targetable_layers()
        num_n = layers[layer_name].num_neurons if layer_name in layers else 128
        if use_adaptive:
            self.session.injector.prune_weak_activations(
                layer_name, list(range(num_n)), std_multiplier=val
            )
            print(
                f"Registered adaptive WEAK_PRUNE (std_multiplier={val}) across {num_n} neurons in {layer_name}."
            )
        else:
            self.session.injector.prune_weak_activations(
                layer_name, list(range(num_n)), threshold=val
            )
            print(
                f"Registered WEAK_PRUNE (|a| below {val}) across {num_n} neurons in {layer_name}."
            )

    def do_gate(self, arg: str) -> None:
        """gate <target_layer> <target_neuron> <det_layer> <det_neuron> <delta> [threshold] : Register a closed-loop conditional gate."""
        parts = _split_cli_args(arg)
        if len(parts) < 5:
            print(
                "Usage: gate <target_layer> <target_neuron> <det_layer> <det_neuron> <delta> [threshold]"
            )
            return
        t_layer, t_idx = parts[0], int(parts[1])
        d_layer, d_idx = parts[2], int(parts[3])
        delta = float(parts[4])
        thresh = float(parts[5]) if len(parts) >= 6 else 1.0
        self.session.injector.add_conditional_gate(
            target_layer=t_layer,
            target_neurons=[t_idx],
            detector_layer=d_layer,
            detector_neuron=d_idx,
            detector_pole=PoleType.POSITIVE,
            threshold=thresh,
            action_mode=InterventionMode.ADDITIVE_STEER,
            value=delta,
            sequence_scope="any_token",
        )
        print(
            f"Registered CONDITIONAL_GATE: steer {t_layer}[N{t_idx}] by {delta:+.2f} when {d_layer}[N{d_idx}] > {thresh}."
        )

    def do_clear(self, _arg: str) -> None:
        """clear : Remove all active surgical neuron interventions."""
        self.session.injector.clear_interventions()
        print("Cleared all active interventions.")

    def do_eval(self, arg: str) -> None:
        """eval <prompt> : Compare Clean vs. Surgically Intervened next-token probabilities and KL divergence."""
        prompt = arg.strip() or self.session.get_default_eval_prompt()
        res = self.session.evaluate_prompt_comparison(prompt)
        print(f"\nPrompt: \"{res['prompt']}\" | Sequence KL-Divergence: {res['kl_divergence']:.5f}")
        print(f"Active Interventions: {len(res['active_interventions'])}")
        print(f"\n{'CLEAN BASELINE TOP TOKENS':<36} | {'INTERVENED TOP TOKENS':<36}")
        print("-" * 75)
        for c, m in zip(res["clean_top_predictions"], res["intervened_top_predictions"]):
            c_str = f"{c['token']:<18} ({c['prob'] * 100:5.1f}%)"
            m_str = f"{m['token']:<18} ({m['prob'] * 100:5.1f}%)"
            print(f"{c_str:<36} | {m_str:<36}")
        print()

    def do_trace(self, arg: str) -> None:
        """trace <src_layer> <dst_layer> <src_neuron> [positive|negative] [prompt] : Trace downstream layer-to-layer circuit."""
        parts = _split_cli_args(arg)
        if len(parts) < 3:
            print("Usage: trace <src_layer> <dst_layer> <src_neuron> [positive|negative] [prompt]")
            return
        src_l, dst_l, n_idx = parts[0], parts[1], int(parts[2])
        pole = PoleType(parts[3]) if len(parts) >= 4 else PoleType.POSITIVE
        prompt = " ".join(parts[4:]) if len(parts) >= 5 else self.session.get_default_eval_prompt()
        input_ids, _ = self.session.encode_prompt(prompt)
        graph = self.session.injector.trace_circuit(
            src_layer=src_l,
            dst_layer=dst_l,
            src_neuron=n_idx,
            inputs=input_ids,
            src_pole=pole,
            compute_jacobian=True,
            top_k=6,
        )
        print(f"\n--- CIRCUIT TRACE: ({src_l}, N{n_idx}, {pole.value}) --> {dst_l} ---")
        print("[Top Excited Downstream Neurons]:")
        for e in graph.top_excitatory(3):
            print(
                f"  -> N{e.dst_neuron:02d} | Role: {e.role.value:<10} | Causal Delta: {e.mean_causal_delta:+.4f} "
                f"| Jacobian: {e.jacobian_sensitivity:+.4f} | W_up: {e.virtual_weight_up:+.3f}, W_gate: {e.virtual_weight_gate:+.3f}"
            )
        print("[Top Inhibited Downstream Neurons]:")
        for e in graph.top_inhibitory(3):
            print(
                f"  -> N{e.dst_neuron:02d} | Role: {e.role.value:<10} | Causal Delta: {e.mean_causal_delta:+.4f} "
                f"| Jacobian: {e.jacobian_sensitivity:+.4f} | W_up: {e.virtual_weight_up:+.3f}, W_gate: {e.virtual_weight_gate:+.3f}"
            )
        print()

    def do_trace_incoming(self, arg: str) -> None:
        """trace_incoming <src_layer> <dst_layer> <dst_neuron> [positive|negative] [prompt] : Trace upstream neurons driving dst_neuron."""
        parts = _split_cli_args(arg)
        if len(parts) < 3:
            print("Usage: trace_incoming <src_layer> <dst_layer> <dst_neuron> [positive|negative] [prompt]")
            return
        src_l, dst_l, n_idx = parts[0], parts[1], int(parts[2])
        pole = PoleType(parts[3]) if len(parts) >= 4 else PoleType.POSITIVE
        prompt = " ".join(parts[4:]) if len(parts) >= 5 else self.session.get_default_eval_prompt()
        input_ids, _ = self.session.encode_prompt(prompt)
        graph = self.session.injector.trace_incoming_circuit(
            src_layer=src_l,
            dst_layer=dst_l,
            dst_neuron=n_idx,
            inputs=input_ids,
            dst_pole=pole,
            top_k=6,
        )
        print(f"\n--- INCOMING UPSTREAM TRACE: {src_l} --> ({dst_l}, N{n_idx}, {pole.value}) ---")
        print("[Top Excitatory Upstream Neurons]:")
        for e in graph.top_excitatory(3):
            print(
                f"  N{e.src_neuron:02d} ({e.src_pole.value}) -> | Role: {e.role.value:<10} | Causal Delta: {e.mean_causal_delta:+.4f} "
                f"| Jacobian: {e.jacobian_sensitivity:+.4f} | W_up: {e.virtual_weight_up:+.3f}, W_gate: {e.virtual_weight_gate:+.3f}"
            )
        print("[Top Inhibitory Upstream Neurons]:")
        for e in graph.top_inhibitory(3):
            print(
                f"  N{e.src_neuron:02d} ({e.src_pole.value}) -> | Role: {e.role.value:<10} | Causal Delta: {e.mean_causal_delta:+.4f} "
                f"| Jacobian: {e.jacobian_sensitivity:+.4f} | W_up: {e.virtual_weight_up:+.3f}, W_gate: {e.virtual_weight_gate:+.3f}"
            )
        print()

    def do_readout(self, arg: str) -> None:
        """readout <src_layer> <dst_layer> <neuron_indices_csv> [threshold] : Audit whether weak cross-activations are read downstream."""
        parts = _split_cli_args(arg)
        if len(parts) < 3:
            print("Usage: readout <src_layer> <dst_layer> <neuron_indices_csv> [threshold]")
            return
        src_l, dst_l = parts[0], parts[1]
        neurons = [int(x.strip()) for x in parts[2].split(",") if x.strip()]
        thresh = float(parts[3]) if len(parts) >= 4 else None
        input_ids = self.session.get_default_eval_inputs()
        rep = self.session.injector.verify_downstream_readout(
            src_layer=src_l,
            dst_layer=dst_l,
            src_neurons=neurons,
            inputs=input_ids,
            weak_threshold=thresh,
        )
        print(f"\n--- DOWNSTREAM READOUT AUDIT ({src_l} {neurons} -> {dst_l}) ---")
        print(
            f"Max Downstream Act Shift : {rep.max_downstream_activation_shift:.5f} | "
            f"Max Gate Shift: {rep.max_downstream_gate_shift:.5f}"
        )
        print(
            f"Benign Spillover ('Nothing Reads That Combination'): {rep.is_benign_spillover} | "
            f"Affected Downstream Neurons: {rep.affected_downstream_neurons}\n"
        )

    def do_explain(self, arg: str) -> None:
        """explain <layer_name> <neuron_idx> [dst_layer] [src_layer] : Run the Automated LLM Explainer & Surgical Verifier."""
        parts = _split_cli_args(arg)
        if len(parts) < 2:
            print("Usage: explain <layer_name> <neuron_idx> [dst_layer] [src_layer]")
            return
        layer_name, idx = parts[0], int(parts[1])
        dst_layer = parts[2] if len(parts) >= 3 and parts[2] != "-" else None
        src_layer = parts[3] if len(parts) >= 4 and parts[3] != "-" else None
        input_ids = self.session.get_default_eval_inputs()
        hyp = self.session.injector.explain_neuron(
            layer_name=layer_name,
            neuron_idx=idx,
            tok_to_id=self.session.tok_to_id,
            idx_to_token=self.session.id_to_tok,
            unembed_weight=self.session.unembed_weight,
            dst_layer=dst_layer,
            circuit_inputs=input_ids if (dst_layer or src_layer) else None,
            src_layer=src_layer,
        )
        print(f"\n=== AGENTIC EXPLANATION: {hyp.layer_name} [Neuron #{hyp.neuron_idx}] ===")
        print(f"Taxonomy       : {hyp.taxonomy_class.value} (Confidence: {hyp.confidence_score:.2f})")
        print(f"+Pole Motif    : {hyp.positive_pole_motif}")
        print(f"-Pole Motif    : {hyp.negative_pole_motif}")
        print(f"Unified 'Verb' : {hyp.unified_verb_hypothesis}")
        if hyp.recursive_offshoots:
            print("Offshoots      :")
            for branch in hyp.recursive_offshoots:
                print(f"  - {branch}")
        if hyp.verification_report:
            print(
                f"Verification   : Score={hyp.verification_report.empirical_verification_score:.3f} | {hyp.verification_report.summary_notes}\n"
            )

    def do_ecology(self, arg: str) -> None:
        """ecology <layer_name> [dst_layer] [top_anchors] [top_contextual] : Solve layer ecology via Crossword propagation."""
        parts = _split_cli_args(arg)
        if not parts:
            print("Usage: ecology <layer_name> [dst_layer] [top_anchors] [top_contextual]")
            return
        layer_name = parts[0]
        dst_layer = parts[1] if len(parts) >= 2 else None
        top_a = int(parts[2]) if len(parts) >= 3 else 2
        top_c = int(parts[3]) if len(parts) >= 4 else 2
        input_ids = self.session.get_default_eval_inputs()
        rep = self.session.injector.solve_layer_ecology(
            layer_name=layer_name,
            tok_to_id=self.session.tok_to_id,
            idx_to_token=self.session.id_to_tok,
            unembed_weight=self.session.unembed_weight,
            dst_layer=dst_layer,
            circuit_inputs=input_ids if dst_layer else None,
            top_anchors=top_a,
            top_contextual=top_c,
        )
        print(f"\n=== CROSSWORD CIRCUIT ECOLOGY SOLVE: {rep.layer_name} ===")
        print(
            f"Anchor Neurons Solved: {rep.anchor_neurons_explained} | "
            f"Contextual Neurons Solved: {rep.contextual_neurons_explained} | "
            f"Mean Verification Score: {rep.mean_verification_score:.3f}"
        )
        for n_idx, hyp in rep.hypotheses.items():
            print(f"  * N{n_idx:<3} [{hyp.taxonomy_class.value}]: {hyp.unified_verb_hypothesis}")
        print()

    def do_contrastive(self, arg: str) -> None:
        """contrastive <pos_prompt> | <neg_prompt> : Discover native neurons separating two contrastive prompts."""
        if "|" not in arg:
            print(
                "Usage: contrastive <pos_prompt> | <neg_prompt>  (e.g., contrastive the system negotiated AES256 TLS | the system executed SELECT postgres)"
            )
            return
        pos_str, neg_str = [s.strip() for s in arg.split("|", 1)]
        pos_ids, _ = self.session.encode_prompt(pos_str)
        neg_ids, _ = self.session.encode_prompt(neg_str)
        rep = self.session.injector.discover_contrastive_neurons(
            pos_inputs=pos_ids, neg_inputs=neg_ids, top_k=6
        )
        print("\n=== CONTRASTIVE BIPOLAR NEURON DISCOVERY (Welch's t-statistic) ===")
        for m in rep.top_matches:
            print(
                f"  {m.layer_name}[N{m.neuron_idx:<3}] | t={m.t_statistic:+.3f} | "
                f"mu+={m.pos_mean:+.3f}, mu-={m.neg_mean:+.3f} | "
                f"SignFlip={m.bipolar_sign_flip} | Pole={m.recommended_pole.value}"
            )
        print()

    def do_synergy(self, arg: str) -> None:
        """synergy <layer_name> [k] [greedy|l1_optim] : Find a minimal synergistic coalition of k neurons to ablate."""
        parts = _split_cli_args(arg)
        if not parts:
            print("Usage: synergy <layer_name> [k] [greedy|l1_optim]")
            return
        layer_name = parts[0]
        k = int(parts[1]) if len(parts) >= 2 else 3
        method = parts[2] if len(parts) >= 3 else "greedy"
        input_ids = self.session.get_default_eval_inputs()
        res = self.session.injector.find_synergistic_ablation_set(
            layer_name=layer_name, inputs=input_ids, max_subset_size=k, method=method
        )
        print(f"\n=== SYNERGISTIC MULTI-NEURON ABLATION ({res.layer_name}, method={res.method}) ===")
        print(f"Selected Coalition : {res.selected_neurons}")
        print(f"Stepwise Scores    : {res.stepwise_scores}")
        print(
            f"Joint Ablation KL  : {res.joint_ablation_kl:.5f} | "
            f"Sum Individual KLs: {res.sum_of_individual_kls:.5f} | "
            f"Synergy Ratio: {res.synergy_ratio:.3f}\n"
        )

    def do_export(self, arg: str) -> None:
        """export <json_filepath> [pt_vectors_filepath] : Export active interventions & compiled steering vectors."""
        parts = _split_cli_args(arg)
        if not parts:
            print(
                "Usage: export <json_filepath> [pt_vectors_filepath]  (e.g., export steering_preset.json vectors.pt)"
            )
            return
        json_path = parts[0]
        pt_path = parts[1] if len(parts) >= 2 else None

        bundle = self.session.injector.export_production_preset(
            preset_name="cli_exported_preset",
            model_name=self.session.model_name,
            save_json_path=json_path,
            save_pt_vectors_path=pt_path,
        )
        print(f"\nExported '{bundle.preset_name}' to {json_path}:")
        print(f"  - Surgical Interventions : {len(bundle.interventions)}")
        print(f"  - Compiled d_model Vectors: {len(bundle.compiled_vectors)}")
        for cv in bundle.compiled_vectors:
            print(
                f"    * Layer '{cv.layer_name}' (d_model={cv.d_model}) | Neurons={cv.source_neurons} | ||v_steer||_2 = {cv.vector_norm:.4f}"
            )
        print()

    def do_import(self, arg: str) -> None:
        """import <json_filepath> : Load a saved production steering preset into the active session."""
        parts = _split_cli_args(arg)
        if not parts:
            print("Usage: import <json_filepath>")
            return
        bundle = self.session.injector.load_production_preset(parts[0])
        print(f"Loaded preset '{bundle.preset_name}' with {len(bundle.interventions)} interventions.")

    def do_fold(self, _arg: str) -> None:
        """fold : Permanently fold active unconditional interventions into down_proj weights & biases."""
        res = self.session.injector.fold_interventions_into_weights()
        print(f"Folded {res['operations_applied']} interventions directly into model weights:")
        for d in res["details"]:
            print(f"  - {d['layer']} (neurons {d['neurons']}): {d['folded_op']}")

    def do_exit(self, _arg: str) -> bool:
        """exit : Exit the Neural Injector REPL."""
        return True

    def do_EOF(self, _arg: str) -> bool:
        print()
        return True


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="neural-injector",
        description="Neural Injector CLI & FastAPI Workbench ('Neural Networks Are Already Clean')",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Optional HuggingFace model ID to load (e.g. 'gpt2' or 'Qwen/Qwen2.5-1.5B'). Defaults to built-in 2-layer SwiGLU model.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu', 'cuda', 'cuda:0', etc.)")

    sub = parser.add_subparsers(dest="command")

    # 1. Interactive REPL
    sub.add_parser("repl", help="Start the stateful interactive CLI workbench shell")

    # 2. FastAPI Server
    serve_p = sub.add_parser("serve", help="Launch the FastAPI REST server & browser UI workbench")
    serve_p.add_argument("--host", type=str, default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    # 3. Direct one-shot profile command
    prof_p = sub.add_parser("profile", help="Inspect a neuron's bipolar profile")
    prof_p.add_argument("layer_name", type=str)
    prof_p.add_argument("neuron_idx", type=int)

    # 4. Direct one-shot explain command
    exp_p = sub.add_parser(
        "explain", help="Run the agentic explainer + surgical verification on a neuron"
    )
    exp_p.add_argument("layer_name", type=str)
    exp_p.add_argument("neuron_idx", type=int)
    exp_p.add_argument("--dst-layer", type=str, default=None)
    exp_p.add_argument("--src-layer", type=str, default=None)

    args = parser.parse_args(argv)

    session = (
        WorkbenchSession.from_huggingface(args.hf_model, device=args.device)
        if args.hf_model
        else WorkbenchSession.create_demo_session()
    )

    if args.command == "serve":
        import uvicorn
        from .server import create_workbench_app

        app = create_workbench_app(session)
        print(f"Launching Neural Injector Workbench at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "profile":
        repl = NeuralInjectorREPL(session)
        repl.do_profile(f"{args.layer_name} {args.neuron_idx}")

    elif args.command == "explain":
        repl = NeuralInjectorREPL(session)
        dst_arg = args.dst_layer or "-"
        src_arg = args.src_layer or "-"
        repl.do_explain(f"{args.layer_name} {args.neuron_idx} {dst_arg} {src_arg}")

    else:
        # Default to interactive REPL
        NeuralInjectorREPL(session).cmdloop()


if __name__ == "__main__":
    main()
