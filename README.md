# neural-injector

Native-basis neuron auditing, circuit tracing, explanation testing and surgical steering for PyTorch / HuggingFace models, following Nick Cammarata's *Neural Networks Are Already Clean* (Goodfire, 2026).

The tool studies a model's own neurons directly. It never trains a dictionary, and it never labels a neuron "clean" from activation statistics. Cleanliness, in the paper's sense, is a property of explanations that could have lost and didn't. The tool measures, runs the paper's tests, and reports what came out.

## What the paper proposes and where it lives

| Paper | What is measured | API |
|---|---|---|
| Neurons as verbs: both poles matter | Exact moments, top-k and activation bands for the positive and negative pole separately | `audit_layers`, `get_neuron_profile` |
| Separating the sign regimes (SwiGLU / GeGLU) | Every activation tagged `++`, `--`, `+-`, `-+` from the gate sign; exact per-regime top-k, moments and histograms; `--` never enters the `++` list | `NeuronPoleProfile.regime_top_samples`, `layer_regime_band_counts` |
| "Every strong firing … rings are fractions of its max" | Exact counts per fraction-of-extreme band (octave rings by default), uniform examples inside each band | `activation_bands`, `DEFAULT_BAND_EDGES` |
| Percentiles vs raw activations | Within-pole percentiles from an exact log-magnitude histogram (bin resolution 2^(1/4)) | `percentile_within_pole`, `compute_thresholds` |
| Read forward (Titanic → boat), summary motif, long context | Every example carries its next token, a token window and the full sample text | `ActivationSample` |
| Don't say dot | Direct logit attribution; change in next-token log-prob mass of a token set under ablation | `top_suppressed_tokens`, `measure_token_set_effect` |
| Blind maximum; "Explanations That Could Lose" | Agent 1 explains each pole from the evidence. Agent 2 sees only the explanation and writes a blind maximum, 5 positives, 5 near misses and 5 distant-concept texts. A response is **strong** at ≥ 30% of the pole's corpus extreme. **won / tied / lost** = strong positives vs strong near misses | `explain_neuron`, `audit_layer_explanations` |
| Cluster the attribution flowing in; blind max per bundle while ablating the others | Gradient × activation attribution per activating context, cosine k-means on attribution profiles, silhouette vs permutation null; per-bundle blind max under ablation vs best corpus example under the same ablation | `cluster_incoming_attribution_bundles`, `run_bundles=True` |
| Steering: "what the model writes when you turn the neuron up" | Greedy continuations with the neuron clamped to its corpus max / min, given to the explainer as evidence | `steering_prompts=` |
| Understanding compounds (crossword) | Neighbour explanations come only from traced circuit edges or shared traced readers | `solve_circuit_ecology`, `CircuitEcologyMap` |
| Do weak activations matter? | Remove each neuron's weak activations (fraction of extreme or within-pole percentile) vs count-matched random removal; NLL, KL, top-1 agreement, margins of flipped predictions | `measure_weak_activation_ablation` |
| Double-negative ablation across the network | Zero the `--` regime in every gated layer, evaluate held-out next-token prediction | `ablate_double_negative_regime`, `evaluate_intervention_effect` |
| Remove extreme responses (gate > ordinary-text p99.9) | Exact per-neuron gate order statistic, culling vs count-matched and magnitude-matched random controls, optional recitation continuation | `calibrate_tail_thresholds`, `run_extreme_culling_experiment` |
| Poe "dreary": total vs direct effect, downstream rescue | Full ablation; removal of only the direct write a·W_down[:, i] from the final residual stream; rescue by restoring downstream neurons | `decompose_direct_vs_indirect_effect` |
| Looking at the residual stream (unread saliency coordinate) | Per-coordinate activation-mass share vs norm-scaled read share of all downstream readers | `audit_residual_stream_sinks` |
| Nothing reads that combination | Max reader alignment of a neuron combination vs its best constituent and vs random same-size combinations; optional causal interaction | `verify_combination_readout` |
| Watching neurons develop | Per-checkpoint moments, write-vector drift, tokens per activation band | `track_checkpoint_evolution` |
| Experts are far more specialised | Each expert of a MoE parameter container is its own neuron layer (`…experts@expert7`) | discovery |

Additions beyond the paper (the steering side):
- Contrastive neuron discovery uses Welch's t on per-prompt means, with prompts as the independent units, plus p-values and a Bonferroni correction.
- Native-scale steering.
- Conditional gates that latch across KV-cached decoding.
- Synergistic ablation coalitions.
- Exact weight folding.
- Residual steering vectors for purely additive interventions.
- JSON presets and a standalone runtime.

## Constants and parameters

The paper fixes these numbers, and they are used as given:
- 30% strong-response threshold.
- 5 positive, 5 near-miss and 5 distant texts.
- p99.9 gate culling.
- "Keep roughly the strongest quarter" (weak-activation percentile 75 in the example workflow).

Everything else is either a measurement resolution, a display choice or a statistical convention. Each one is an explicit argument:
- Histogram bins per octave.
- Band edges (octave rings by default).
- Examples per band.
- Number of random control neurons.
- Null samples.
- `alpha = 0.05`.

No score combines metrics with hand-chosen weights, and no taxonomy is assigned from statistics.

## Install

```bash
pip install -e ".[huggingface,dev]"
```

## Quickstart

```python
from neural_injector import WorkbenchSession, OpenAICompatibleClient, PoleType

s = WorkbenchSession.from_huggingface("HuggingFaceTB/SmolLM2-135M")
layer = list(s.injector.list_targetable_layers())[14]
s.run_audit([layer], prompts=my_corpus)          # use a broad corpus; narrow corpora make neurons look polysemantic

p = s.injector.get_neuron_profile(layer, 123, unembed_weight=s.unembed_weight, idx_to_token=s.id_to_tok)
p.regime_top_samples["++"], p.regime_top_samples["--"], p.positive_bands, p.top_suppressed_tokens

s.injector.set_explainer_llm_client(OpenAICompatibleClient(model="gpt-4o"))   # or Ollama/vLLM base_url
h = s.explain_neuron(layer, 123, steering_prompts=["The"])
h.verification["positive"].outcome, h.verification["positive"].positive_strong_count

summary = s.injector.audit_layer_explanations(layer, n_neurons=50, seed=0,
                                              encode_fn=s.encode_text, decode_fn=s.decode_ids)
```

To run the full experimental workflow on any model and corpus, see `examples/hf_workflow.py`.

## CLI

```bash
neural-injector --hf-model HuggingFaceTB/SmolLM2-135M repl
```

Commands: `layers`, `audit <layers> @corpus.txt`, `scan`, `profile`, `llm`, `explain`, `trace`, `trace_incoming`, `readout`, `contrastive`, `synergy`, `ablate`, `clamp`, `steer`, `prune`, `dblneg`, `gate`, `eval`, `generate`, `export`, `import`, `fold`. Layers are never audited implicitly.

## Limitations

- Explanations need an LLM (or a human through `CallableLLMClient`). There is no offline explainer, because a template built from the top tokens cannot lose.
- Forward-mode Jacobians and gradient attribution are not implemented for routed MoE expert rows. Ablation-based tracing works there.
- Removing the direct path requires an input-hooked (down-projection) neuron layer.
- Attention heads are only traced through the neurons they affect.
- Random controls in the weak-activation and culling experiments are drawn over all positions of a batch. Use unpadded batches (batch size 1) so padding positions are never chosen.
