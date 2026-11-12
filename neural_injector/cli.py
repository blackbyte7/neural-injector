from __future__ import annotations

import argparse
import cmd
import os
from pathlib import Path
import shlex
from typing import List, Optional

from .explainer import OpenAICompatibleClient
from .session import WorkbenchSession
from .types import InterventionMode, PoleType


def _split_cli_args(arg: str) -> List[str]:
    """Splits arguments while keeping Windows backslashes."""
    return [tok.strip("\"'") for tok in shlex.split(arg, posix=False) if tok.strip("\"'")]


def load_corpus(path: str) -> List[str]:
    """Loads prompts from a text file: blocks separated by blank lines (or one prompt per line if no blank lines)."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return blocks


def _prompts_from(arg: str) -> List[str]:
    arg = arg.strip()
    if arg.startswith("@"):
        return load_corpus(arg[1:].strip())
    return [p.strip() for p in arg.split("|") if p.strip()]


class NeuralInjectorREPL(cmd.Cmd):
    """Interactive workbench: audit, inspect, intervene, trace, explain and export."""

    intro = (
        "\n======================================================================\n"
        "  NEURAL INJECTOR WORKBENCH\n"
        "  'help' lists commands. Start with 'layers', then 'audit'.\n"
        "======================================================================\n"
    )
    prompt = "neural-injector> "

    def __init__(self, session: WorkbenchSession):
        super().__init__()
        self.session = session

    def _audited(self, layer: str) -> bool:
        if not self.session.injector.is_layer_audited(layer):
            print(f"Layer '{layer}' has not been audited. Run: audit {layer} @corpus.txt  (or prompts separated by |)")
            return False
        return True

    def onecmd(self, line: str) -> bool:
        try:
            return super().onecmd(line)
        except Exception as exc:  # keep the REPL alive, but show the real error
            print(f"Error: {type(exc).__name__}: {exc}")
            return False

    def do_layers(self, _arg: str) -> None:
        """layers : list targetable neuron layers."""
        print(f"\nModel: {self.session.model_name}")
        print(f"{'LAYER NAME':<44} | {'TYPE':<44} | {'NEURONS':<8} | HOOK")
        print("-" * 112)
        for name, d in self.session.injector.list_targetable_layers().items():
            print(f"{name:<44} | {d.module_type:<44} | {d.num_neurons:<8} | {d.hook_target}")
        print()

    def do_audit(self, arg: str) -> None:
        """audit <layer[,layer...]|all> <@corpus.txt | prompt1 | prompt2 | ...> : record streaming statistics."""
        parts = arg.strip().split(None, 1)
        if len(parts) < 2:
            print("Usage: audit <layer[,layer]|all> <@corpus.txt | prompt1 | prompt2 ...>")
            return
        layers = list(self.session.injector.list_targetable_layers()) if parts[0] == "all" else [x for x in parts[0].split(",") if x]
        prompts = _prompts_from(parts[1])
        res = self.session.run_audit(layer_names=layers, prompts=prompts)
        print(f"Audited {len(res['layers'])} layer(s) on {res['num_prompts']} prompt(s).")

    def do_llm(self, arg: str) -> None:
        """llm <model_name> [base_url] [api_key] : configure the explainer and test-writer LLM (OpenAI-compatible)."""
        parts = _split_cli_args(arg)
        if not parts:
            print("Usage: llm <model_name> [base_url] [api_key]   e.g. llm llama3.1 http://localhost:11434/v1")
            return
        base = parts[1] if len(parts) >= 2 else os.environ.get("NEURAL_INJECTOR_LLM_BASE_URL", "https://api.openai.com/v1")
        key = parts[2] if len(parts) >= 3 else os.environ.get("OPENAI_API_KEY")
        self.session.injector.set_explainer_llm_client(OpenAICompatibleClient(model=parts[0], api_key=key, base_url=base))
        print(f"Explainer and test writer: {parts[0]} @ {base}")

    def do_scan(self, arg: str) -> None:
        """scan <layer> [sort_by] [top_n] : descriptive table (sort_by: rms_write_norm, std, max_val, min_val, positive_excess_kurtosis, negative_excess_kurtosis, double_negative_positive_rate, neuron_idx)."""
        parts = _split_cli_args(arg)
        if not parts or not self._audited(parts[0]):
            return
        rep = self.session.injector.scan_layer(parts[0], sort_by=parts[1] if len(parts) > 1 else "rms_write_norm",
                                               limit=int(parts[2]) if len(parts) > 2 else 15)
        print(f"\n=== {rep.layer_name}: {rep.num_neurons} neurons, {rep.tokens_scanned} tokens, gated={rep.gated} ===")
        if rep.gated:
            print(f"Share of a>0 activations in the -- regime (layer): {rep.layer_double_negative_positive_rate:.4f}")
        ea = rep.explanation_audit
        if ea.poles_explained:
            print(f"Explained poles: {ea.poles_explained} | taxonomy {ea.taxonomy_counts} | outcomes {ea.outcome_counts} "
                  f"| strong rates: positives {ea.positive_control_strong_rate:.2f}, near misses {ea.near_miss_strong_rate:.2f}, "
                  f"distant {ea.distant_strong_rate:.2f}")
        print(f"\n{'IDX':<6} {'MAX':>9} {'MIN':>9} {'P+':>6} {'P-':>6} {'K+':>8} {'K-':>8} {'--%':>6} {'RMSWRITE':>9}  +TOK / -TOK  [explained]")
        for n in rep.neurons:
            ex = f"[{n.pos_taxonomy.value}/{n.pos_outcome.value}]" if n.explained else ""
            print(f"N{n.neuron_idx:<5} {n.max_val:>9.3f} {n.min_val:>9.3f} {n.positive_firing_rate:>6.3f} {n.negative_firing_rate:>6.3f} "
                  f"{n.positive_excess_kurtosis:>8.2f} {n.negative_excess_kurtosis:>8.2f} {100 * n.double_negative_positive_rate:>5.1f} "
                  f"{n.rms_write_norm:>9.3f}  {n.top_pos_token!r} / {n.top_neg_token!r} {ex}")
        print()

    def do_profile(self, arg: str) -> None:
        """profile <layer> <neuron> : both poles, sign regimes, activation bands, logit attribution."""
        parts = _split_cli_args(arg)
        if len(parts) < 2 or not self._audited(parts[0]):
            return
        p = self.session.injector.get_neuron_profile(parts[0], int(parts[1]), unembed_weight=self.session.unembed_weight,
                                                     idx_to_token=self.session.id_to_tok)
        print(f"\n--- {p.layer_name} [N{p.neuron_idx}] | tokens {p.tokens_observed} | mean {p.mean:+.4f} std {p.std:.4f} ---")
        for name, m in (("+", p.positive_pole), ("-", p.negative_pole)):
            print(f"pole {name}: rate {m.rate:.4f} extreme {m.extreme:+.4f} mean|a| {m.mean:.4f} std {m.std:.4f} excess kurtosis {m.excess_kurtosis:.2f}")
        if p.gated:
            print(f"sign regime rates: {p.swiglu_sign_regime_rates} | -- share of a>0: {p.double_negative_positive_rate:.4f}")
        for title, samples in (("TOP POSITIVE (all regimes)", p.top_positive_samples), ("TOP NEGATIVE", p.top_negative_samples)):
            print(f"\n[{title}]")
            for s in samples[:8]:
                print(f"  {s.activation_value:+9.4f} {s.sign_regime or ''} {s.window_str!r} -> next {s.next_token_str!r}")
        for title, bands in (("POSITIVE BANDS", p.positive_bands), ("NEGATIVE BANDS", p.negative_bands)):
            print(f"\n[{title}] (share of pole extreme: count, regime counts, examples)")
            for b in bands:
                print(f"  [{b.lower_fraction:.3f}, {b.upper_fraction:.3f}) n={b.count} {b.regime_counts} {[s.token_str for s in b.samples]}")
        print(f"\npromoted logits:   {[(t, round(v, 3)) for t, v in p.top_promoted_tokens[:8]]}")
        print(f"suppressed logits: {[(t, round(v, 3)) for t, v in p.top_suppressed_tokens[:8]]}\n")

    def do_ablate(self, arg: str) -> None:
        """ablate <layer> <neuron> [both|positive|negative]"""
        p = _split_cli_args(arg)
        self.session.injector.zero_ablate(p[0], int(p[1]), PoleType(p[2]) if len(p) > 2 else PoleType.BOTH)
        print("Registered ZERO_ABLATE.")

    def do_clamp(self, arg: str) -> None:
        """clamp <layer> <neuron> <value> [both|positive|negative]"""
        p = _split_cli_args(arg)
        self.session.injector.clamp_neurons(p[0], int(p[1]), float(p[2]), PoleType(p[3]) if len(p) > 3 else PoleType.BOTH)
        print("Registered HARD_CLAMP.")

    def do_steer(self, arg: str) -> None:
        """steer <layer> <neuron> <delta> : additive steering by delta."""
        p = _split_cli_args(arg)
        self.session.injector.steer_neurons(p[0], int(p[1]), delta=float(p[2]))
        print("Registered ADDITIVE_STEER.")

    def do_prune(self, arg: str) -> None:
        """prune <layer> <fraction|percentile|absolute> <value> : zero weak activations of every neuron of the layer."""
        p = _split_cli_args(arg)
        if len(p) < 3:
            print("Usage: prune <layer> <fraction|percentile|absolute> <value>")
            return
        kw = {"fraction": "fraction_of_extreme", "percentile": "percentile", "absolute": "threshold"}[p[1]]
        specs = self.session.injector.prune_weak_activations(p[0], **{kw: float(p[2])})
        print(f"Registered {len(specs)} WEAK_PRUNE spec(s) on {p[0]}.")

    def do_dblneg(self, arg: str) -> None:
        """dblneg [layer,layer...|all] : zero the -- (double-negative) regime in gated layers."""
        names = None if not arg.strip() or arg.strip() == "all" else arg.strip().split(",")
        specs = self.session.injector.ablate_double_negative_regime(names)
        print(f"Registered DOUBLE_NEGATIVE_ABLATE on {len(specs)} layer(s).")

    def do_gate(self, arg: str) -> None:
        """gate <target_layer> <target_neuron> <det_layer> <det_neuron> <delta> <threshold> : closed-loop conditional steering."""
        p = _split_cli_args(arg)
        if len(p) < 6:
            print("Usage: gate <target_layer> <target_neuron> <det_layer> <det_neuron> <delta> <threshold>")
            return
        self.session.injector.add_conditional_gate(p[0], [int(p[1])], p[2], int(p[3]), threshold=float(p[5]),
                                                   action_mode=InterventionMode.ADDITIVE_STEER, value=float(p[4]),
                                                   sequence_scope="any_token", sticky_on_decode=True)
        print("Registered CONDITIONAL_GATE.")

    def do_clear(self, _arg: str) -> None:
        """clear : remove all interventions."""
        self.session.injector.clear_interventions()
        print("Cleared all interventions.")

    def do_eval(self, arg: str) -> None:
        """eval <prompt> : clean vs intervened next-token predictions."""
        res = self.session.evaluate_prompt_comparison(arg.strip() or self.session.get_default_eval_prompt())
        print(f"\nKL (mean over positions): {res['kl_divergence']:.5f} | interventions: {len(res['active_interventions'])}")
        for c, m in zip(res["clean_top_predictions"], res["intervened_top_predictions"]):
            print(f"  {c['token']!r:<22} {100 * c['prob']:5.1f}%   |   {m['token']!r:<22} {100 * m['prob']:5.1f}%")
        print()

    def do_generate(self, arg: str) -> None:
        """generate [n_tokens] <prompt> : greedy clean vs intervened continuation."""
        p = arg.strip().split(None, 1)
        n = int(p[0]) if p and p[0].isdigit() else 24
        prompt = (p[1] if p and p[0].isdigit() and len(p) > 1 else arg.strip()) or self.session.get_default_eval_prompt()
        res = self.session.generate_comparison(prompt=prompt, max_new_tokens=n)
        print(f"\nclean:      {res['clean_completion_text']!r}\nintervened: {res['intervened_completion_text']!r}\n"
              f"tokens changed: {res['tokens_changed_count']}\n")

    def do_trace(self, arg: str) -> None:
        """trace <src_layer> <dst_layer> <src_neuron> <positive|negative> <prompt> : downstream causal edges."""
        p = arg.strip().split(None, 4)
        if len(p) < 5:
            print("Usage: trace <src_layer> <dst_layer> <src_neuron> <positive|negative> <prompt>")
            return
        ids, _ = self.session.encode_prompt(p[4])
        g = self.session.injector.trace_circuit(p[0], p[1], int(p[2]), ids, src_pole=PoleType(p[3]), top_k=8)
        for e in g.edges:
            gp = f" gate-path {e.gate_path_delta:+.4f} value-path {e.up_path_delta:+.4f}" if e.gate_path_delta is not None else ""
            print(f"  -> N{e.dst_neuron:<5} delta {e.mean_causal_delta:+.5f} J {e.jacobian_sensitivity:+.5f} role {e.role.value}{gp}")

    def do_trace_incoming(self, arg: str) -> None:
        """trace_incoming <src_layer> <dst_layer> <dst_neuron> <positive|negative> <prompt> : upstream attribution + ablation."""
        p = arg.strip().split(None, 4)
        if len(p) < 5:
            print("Usage: trace_incoming <src_layer> <dst_layer> <dst_neuron> <positive|negative> <prompt>")
            return
        ids, _ = self.session.encode_prompt(p[4])
        g = self.session.injector.trace_incoming_circuit(p[0], p[1], int(p[2]), ids, dst_pole=PoleType(p[3]), top_k=8)
        for e in g.edges:
            print(f"  N{e.src_neuron:<5} ({e.src_pole.value}) delta {e.mean_causal_delta:+.5f} J {e.jacobian_sensitivity:+.5f}")

    def do_readout(self, arg: str) -> None:
        """readout <src_layer> <neuron_csv> : does any downstream reader read this combination beyond chance and beyond its parts?"""
        p = _split_cli_args(arg)
        if len(p) < 2:
            print("Usage: readout <src_layer> <neuron_csv>")
            return
        r = self.session.injector.verify_combination_readout(p[0], [int(x) for x in p[1].split(",")])
        print(f"combination max |cos| {r.combination_max_reader_cosine:.4f} at {r.best_reader} | best constituent {r.best_single_constituent_cosine:.4f} "
              f"| null mean {r.null_mean:.4f}, p95 {r.null_p95:.4f}, percentile {r.combination_percentile_in_null:.3f} "
              f"| read beyond constituents: {r.reads_combination_beyond_constituents}")

    def do_explain(self, arg: str) -> None:
        """explain <layer> <neuron> [steer prompt ...] : two-agent explanation + blind tests (needs 'llm')."""
        p = arg.strip().split(None, 2)
        if len(p) < 2 or not self._audited(p[0]):
            return
        steer = [p[2]] if len(p) > 2 else None
        h = self.session.explain_neuron(p[0], int(p[1]), steering_prompts=steer)
        print(f"\n=== {h.layer_name} [N{h.neuron_idx}] (confidence {h.confidence_score:.2f}, neighbour clues {h.neighbor_clues_used}) ===")
        print(f"+ ({h.pos_taxonomy.value}): {h.positive_pole_motif}")
        print(f"- ({h.neg_taxonomy.value}): {h.negative_pole_motif}")
        print(f"distinction: {h.unified_verb_hypothesis}")
        for o in h.recursive_offshoots:
            print(f"  offshoot: {o}")
        for pole, rep in h.verification.items():
            print(f"[{pole}] outcome {rep.outcome.value} | positives strong {rep.positive_strong_count}/5, near misses strong "
                  f"{rep.near_miss_strong_count}/5, distant strong {rep.distant_strong_count}/5 | blind max {rep.blind_max_ratio:.2f} of corpus extreme")
        if h.causal_checks:
            print(f"ablation KL {h.causal_checks.target_ablation_kl:.5f}, percentile among random neurons "
                  f"{h.causal_checks.target_kl_percentile_among_controls:.2f}")
        print()

    def do_contrastive(self, arg: str) -> None:
        """contrastive <@pos.txt | p1 ;; p2 ...> || <@neg.txt | n1 ;; n2 ...> : Welch t over per-prompt means (>= 2 prompts per side)."""
        if "||" not in arg:
            print("Usage: contrastive <pos prompts> || <neg prompts>   (prompts separated by ';;' or @file)")
            return
        a, b = [x.strip() for x in arg.split("||", 1)]
        side = lambda s: load_corpus(s[1:]) if s.startswith("@") else [x.strip() for x in s.split(";;") if x.strip()]
        pos = [self.session.encode_prompt(t)[0] for t in side(a)]
        neg = [self.session.encode_prompt(t)[0] for t in side(b)]
        rep = self.session.injector.discover_contrastive_neurons(pos, neg, top_k=10)
        print(f"\n{rep.tests_performed} neurons tested on {rep.pos_prompts_count} vs {rep.neg_prompts_count} prompts")
        for m in rep.top_matches:
            print(f"  {m.layer_name}[N{m.neuron_idx}] t={m.t_statistic:+.3f} df={m.degrees_of_freedom:.1f} p={m.p_value:.2e} "
                  f"(Bonferroni {m.bonferroni_p_value:.2e}) d={m.cohens_d:+.2f} mean+ {m.pos_mean:+.4f} mean- {m.neg_mean:+.4f}")

    def do_synergy(self, arg: str) -> None:
        """synergy <layer> [k] [greedy|l1_optim] <prompt> : coalition whose joint ablation matters most."""
        p = arg.strip().split(None, 3)
        if len(p) < 4:
            print("Usage: synergy <layer> <k> <greedy|l1_optim> <prompt>")
            return
        ids, _ = self.session.encode_prompt(p[3])
        r = self.session.injector.find_synergistic_ablation_set(p[0], ids, max_subset_size=int(p[1]), method=p[2])
        print(f"coalition {r.selected_neurons} | joint {r.joint_score:.5f} vs sum of individual {r.sum_of_individual_scores:.5f} "
              f"| ratio {r.synergy_ratio:.3f} ({r.objective})")

    def do_export(self, arg: str) -> None:
        """export <json_path> [pt_vectors_path]"""
        p = _split_cli_args(arg)
        b = self.session.injector.export_production_preset(model_name=self.session.model_name, save_json_path=p[0],
                                                           save_pt_vectors_path=p[1] if len(p) > 1 else None)
        print(f"Exported {len(b.interventions)} interventions, {len(b.compiled_vectors)} compiled vectors.")

    def do_import(self, arg: str) -> None:
        """import <json_path>"""
        b = self.session.injector.load_production_preset(_split_cli_args(arg)[0])
        print(f"Loaded preset '{b.preset_name}' ({len(b.interventions)} interventions).")

    def do_fold(self, _arg: str) -> None:
        """fold : fold unconditional affine interventions into weights."""
        r = self.session.injector.fold_interventions_into_weights()
        print(f"Folded {r['operations_applied']} spec(s); {r['unfolded_specs_retained']} kept as hooks.")

    def do_exit(self, _arg: str) -> bool:
        """exit"""
        return True

    def do_EOF(self, _arg: str) -> bool:
        print()
        return True


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="neural-injector", description="Neural Injector workbench")
    parser.add_argument("--hf-model", type=str, default=None, help="HuggingFace model id (default: built-in toy model)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=os.environ.get("NEURAL_INJECTOR_LLM_MODEL"))
    parser.add_argument("--llm-base-url", type=str, default=os.environ.get("NEURAL_INJECTOR_LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-api-key", type=str, default=os.environ.get("OPENAI_API_KEY"))
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("repl")
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    ap = sub.add_parser("audit")
    ap.add_argument("layer_name")
    ap.add_argument("corpus", help="@file or prompts separated by |")
    args = parser.parse_args(argv)

    session = (WorkbenchSession.from_huggingface(args.hf_model, device=args.device, device_map=args.device_map)
               if args.hf_model else WorkbenchSession.create_demo_session())
    if args.llm_model:
        session.injector.set_explainer_llm_client(OpenAICompatibleClient(model=args.llm_model, api_key=args.llm_api_key,
                                                                         base_url=args.llm_base_url))
    if args.command == "serve":
        import uvicorn
        from .server import create_workbench_app
        uvicorn.run(create_workbench_app(session), host=args.host, port=args.port)
    elif args.command == "audit":
        NeuralInjectorREPL(session).do_audit(f"{args.layer_name} {args.corpus}")
    else:
        NeuralInjectorREPL(session).cmdloop()


if __name__ == "__main__":
    main()
