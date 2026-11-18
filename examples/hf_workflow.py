"""
End-to-end workflow on any HuggingFace causal LM.

Every number reported is a measurement; neurons are described only through explanations that were tested.

  python examples/hf_workflow.py --model HuggingFaceTB/SmolLM2-135M --layers 1,7,14,21,28 \
      --hf-dataset Salesforce/wikitext --hf-config wikitext-103-raw-v1 --hf-split validation --n-texts 1000 \
      --out results/smollm2

  # optional: two-agent explanation audit of a random neuron sample (needs an OpenAI-compatible LLM)
      --llm-model gpt-4o --llm-base-url https://api.openai.com/v1 --explain-n 20
  # optional: contrastive steering between two prompt files (one prompt per blank-line block)
      --pos-file code.txt --neg-file prose.txt --steer-prompt "def mean(xs):" --steer-strength 2

Steps
  1. Audit the chosen layers on the calibration half of the corpus (exact streaming statistics).
  2. Sign regimes: share of '--' vs '++' inside fraction-of-extreme bands and percentile bands.
  3. Weak activations: remove each neuron's weakest activations (below its own within-pole percentile)
     on the held-out half, against count-matched random removal; flipped predictions with margins.
  4. Double-negative regime ablated in every gated layer: held-out NLL / KL / top-1 agreement.
  5. Extreme culling: zero each neuron where its gate exceeds its calibration-text p99.9, against
     count- and magnitude-matched random controls; optional recitation prompt continuation.
  6. Optional explanation audit and contrastive steering with exact weight folding.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List

import torch

from neural_injector import (
    DEFAULT_BAND_EDGES,
    InterventionMode,
    InterventionSpec,
    OpenAICompatibleClient,
    PoleType,
    WorkbenchSession,
    resolve_site,
)
from neural_injector.cli import load_corpus


def load_texts(args) -> List[str]:
    if args.corpus_file:
        texts = load_corpus(args.corpus_file)
    elif args.hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        texts = [t.strip() for t in ds[args.text_field] if isinstance(t, str) and len(t.strip()) >= args.min_chars]
    else:
        raise SystemExit("Give --corpus-file or --hf-dataset.")
    rng = random.Random(args.seed)
    rng.shuffle(texts)
    return texts[: args.n_texts]


def batches_for(session: WorkbenchSession, texts: List[str], max_tokens: int) -> List[Dict[str, Any]]:
    out = []
    for i, t in enumerate(texts):
        ids, toks, mask = session.encode_prompt_with_mask(t)
        ids, toks, mask = ids[:, :max_tokens], toks[:max_tokens], mask[:, :max_tokens]
        out.append({"input_ids": ids, "attention_mask": mask.long(), "sample_ids": [i], "tokens_per_sample": [toks],
                    "contexts_per_sample": [session.tokenizer.decode(ids[0])]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layers", required=True, help="comma-separated layer indices into the discovered layer list, or 'all'")
    ap.add_argument("--corpus-file")
    ap.add_argument("--hf-dataset")
    ap.add_argument("--hf-config")
    ap.add_argument("--hf-split", default="validation")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--n-texts", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weak-percentile", type=float, default=75.0, help="paper: keep roughly the strongest quarter")
    ap.add_argument("--cull-percentile", type=float, default=99.9)
    ap.add_argument("--recitation-prompt", default="The Lord is my shepherd;")
    ap.add_argument("--recitation-tokens", type=int, default=40)
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    ap.add_argument("--llm-api-key")
    ap.add_argument("--explain-n", type=int, default=0)
    ap.add_argument("--pos-file")
    ap.add_argument("--neg-file")
    ap.add_argument("--steer-prompt")
    ap.add_argument("--steer-strength", type=float, default=1.0)
    ap.add_argument("--steer-top-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"args": vars(args)}
    t0 = time.time()
    session = WorkbenchSession.from_huggingface(args.model, device=args.device, torch_dtype=getattr(torch, args.dtype))
    inj = session.injector
    all_layers = list(inj.list_targetable_layers())
    layers = all_layers if args.layers == "all" else [all_layers[int(i)] for i in args.layers.split(",")]
    texts = load_texts(args)
    half = len(texts) // 2
    calib, held = batches_for(session, texts[:half], args.max_tokens), batches_for(session, texts[half:], args.max_tokens)
    report["corpus"] = {"calibration_texts": len(calib), "held_out_texts": len(held),
                        "calibration_tokens": sum(int(b["attention_mask"].sum()) for b in calib),
                        "held_out_tokens": sum(int(b["attention_mask"].sum()) for b in held)}
    print(f"[load] {args.model}: {len(all_layers)} layers; corpus {report['corpus']}")

    # 1-2. audit and sign regimes
    inj.audit_layers(layers, calib)
    report["layers"] = {}
    for name in layers:
        tr = inj.get_stats_tracker(name)
        entry: Dict[str, Any] = {"tokens": tr.count, "neurons": tr.num_neurons, "gated": tr.gated}
        if tr.gated:
            entry["double_negative_share_of_positive_activations"] = tr.layer_double_negative_positive_rate()
            entry["regimes_by_fraction_of_extreme"] = tr.layer_regime_band_counts(PoleType.POSITIVE, fraction_edges=DEFAULT_BAND_EDGES)
            entry["regimes_by_percentile"] = tr.layer_regime_band_counts(PoleType.POSITIVE, percentile_edges=[0, 75, 99, 99.9, 99.99, 100])
        report["layers"][name] = entry
        print(f"[audit] {name}: {entry.get('double_negative_share_of_positive_activations')}")

    # 3. weak activations
    weak = inj.measure_weak_activation_ablation(layers, held, percentile=args.weak_percentile)
    report["weak_activation_ablation"] = {k: asdict(v) for k, v in weak.items()}
    print(f"[weak] removed: KL {weak['weak_removed'].mean_kl:.5f} top1 {weak['weak_removed'].top1_agreement:.4f} | "
          f"random: KL {weak['count_matched_random'].mean_kl:.5f} top1 {weak['count_matched_random'].top1_agreement:.4f}")

    # 4. double-negative ablation in every gated layer
    gated = [n for n in all_layers if resolve_site(inj.model, n).gated]
    if gated:
        specs = [InterventionSpec(n, list(range(resolve_site(inj.model, n).num_neurons)), InterventionMode.DOUBLE_NEGATIVE_ABLATE)
                 for n in gated]
        dn = inj.evaluate_intervention_effect(held, specs, label="double-negative regime ablated (all gated layers)")
        report["double_negative_ablation"] = asdict(dn)
        print(f"[--] NLL {dn.clean_nll:.4f} -> {dn.intervened_nll:.4f} | KL {dn.mean_kl:.5f} | top1 {dn.top1_agreement:.4f}")

        # 5. extreme culling on the audited layers
        th = inj.calibrate_tail_thresholds(layers, calib, percentile=args.cull_percentile, signal="gate")
        prompt_ids, _ = session.encode_prompt(args.recitation_prompt)
        cull = inj.run_extreme_culling_experiment(th, held, args.cull_percentile, generation_prompt_ids=prompt_ids,
                                                  generation_tokens=args.recitation_tokens, decode_fn=session.decode_ids)
        report["extreme_culling"] = asdict(cull)
        for r in (cull.culled, cull.count_matched_random, cull.magnitude_matched_random):
            print(f"[cull] {r.label}: NLL {r.clean_nll:.4f} -> {r.intervened_nll:.4f}, KL {r.mean_kl:.5f}, entries {r.entries_modified}")
        for k, v in cull.generations.items():
            print(f"   {k}: {v!r}")

    # 6a. explanation audit
    if args.llm_model and args.explain_n > 0:
        client = OpenAICompatibleClient(model=args.llm_model, base_url=args.llm_base_url, api_key=args.llm_api_key)
        inj.set_explainer_llm_client(client)
        report["explanation_audits"] = {}
        for name in layers:
            summ = inj.audit_layer_explanations(name, n_neurons=args.explain_n, seed=args.seed,
                                                encode_fn=session.encode_text, decode_fn=session.decode_ids,
                                                unembed_weight=session.unembed_weight, idx_to_token=session.id_to_tok)
            report["explanation_audits"][name] = asdict(summ)
            print(f"[explain] {name}: {summ}")
        report["explanations"] = {f"{k[0]}#{k[1]}": asdict(h) for k, h in inj.ecology_map.hypotheses.items()}

    # 6b. contrastive steering with folding
    if args.pos_file and args.neg_file and args.steer_prompt:
        pos = [session.encode_prompt(t)[0][:, : args.max_tokens] for t in load_corpus(args.pos_file)]
        neg = [session.encode_prompt(t)[0][:, : args.max_tokens] for t in load_corpus(args.neg_file)]
        disc = inj.discover_contrastive_neurons(pos, neg, layer_names=layers, top_k=args.steer_top_k)
        report["contrastive"] = asdict(disc)
        inj.inject_contrastive_behavior(disc, strength=args.steer_strength)
        gen = session.generate_comparison(args.steer_prompt, max_new_tokens=32)
        report["steering_generation"] = gen
        ids, _ = session.encode_prompt(args.steer_prompt)
        with inj.active_injection():
            hooked = inj.model(ids).logits
        inj.fold_interventions_into_weights()
        folded = inj.model(ids).logits
        report["fold_max_abs_logit_error"] = float((hooked - folded).abs().max())
        print(f"[steer] clean: {gen['clean_completion_text']!r}\n        steered: {gen['intervened_completion_text']!r}\n"
              f"        fold error {report['fold_max_abs_logit_error']:.3e}")

    report["seconds"] = time.time() - t0
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[done] {report['seconds']:.0f}s -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
