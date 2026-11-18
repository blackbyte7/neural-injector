# neural-injector

A toolkit for studying and steering the individual neurons of PyTorch and HuggingFace models. It takes a model's MLP neurons as the unit of analysis: it audits what each neuron responds to, tests explanations of those neurons so that they can fail, traces the circuits between neurons, and lets you steer a model by intervening on chosen neurons ("neural injection"). Interventions run as hooks or are folded into the weights.

The analysis side implements the methods and experiments of Nick Cammarata, *Neural Networks Are Already Clean: A working hypothesis about neurons, generalization, and the ecology of circuits* (Goodfire, 2026). The steering side builds on those measurements.

## Features

- **Neuron audit.** Streams a corpus through any set of layers. For every neuron it keeps exact statistics per pole (positive and negative) and per SwiGLU/GeGLU sign regime: top examples, moments, log-magnitude histograms and activation bands.
- **Explanations that can lose.** Implements the paper's two-agent protocol. One LLM explains a neuron from its evidence. A second LLM sees only that explanation and writes test texts. The neuron's real activations on those texts decide whether the explanation won, tied or lost.
- **Circuits.** Causal edges between neurons, split into gate path and value path. Also forward-mode Jacobians, attribution bundles, direct-versus-indirect effect decomposition, residual-stream sink audits and readout checks.
- **Paper experiments.** Weak-activation removal, removal of the double-negative regime, extreme-response culling and token-suppression effects. Each is measured against matched random controls.
- **Neural injection.** Ablate, clamp, scale or shift neurons on their own native scale. Gate interventions on a detector neuron, and keep gates latched across KV-cached decoding. Find neurons that separate two prompt sets and steer along that contrast.
- **Deployment.** Save interventions as JSON presets, apply them with a standalone runtime, compile additive interventions to residual vectors, or fold them into the weights so that no hooks remain.
- **Interfaces.** Python API, interactive CLI and a local web workbench.

## Supported models

Layers are discovered automatically for common transformer MLPs:
- gated MLPs (Llama, Qwen, Mistral, Gemma, Phi-3, GLM), with fused or split gate/up projections;
- GPT-2-style `Conv1D` MLPs;
- GPT-J / GPT-NeoX / OPT-style MLPs;
- T5 feed-forward blocks;
- ConvNeXt pointwise MLPs.

In mixture-of-experts models, every expert is exposed as its own neuron layer (`…experts@expert7`). Anything else can be registered with an `ArchitectureAdapter`.

## Install

```bash
pip install -e ".[huggingface,dev]"
```

## Quickstart

```python
from neural_injector import WorkbenchSession, OpenAICompatibleClient, PoleType

s = WorkbenchSession.from_huggingface("HuggingFaceTB/SmolLM2-135M")
layers = list(s.injector.list_targetable_layers())
layer = layers[14]

# 1. Audit: use a broad corpus, since a narrow one makes neurons look like they do several unrelated things
s.run_audit([layer], prompts=my_corpus)

# 2. Inspect a neuron
p = s.injector.get_neuron_profile(layer, 123, unembed_weight=s.unembed_weight, idx_to_token=s.id_to_tok)
p.regime_top_samples["++"]      # strongest gate-open positive activations
p.regime_top_samples["--"]      # strongest double-negative activations, listed separately
p.positive_bands                # examples at fractions of the pole's maximum
p.top_suppressed_tokens         # direct logit attribution of the write vector

# 3. Explain and test it (any OpenAI-compatible endpoint: OpenAI, vLLM, Ollama, ...)
s.injector.set_explainer_llm_client(OpenAICompatibleClient(model="gpt-4o"))
h = s.explain_neuron(layer, 123, steering_prompts=["The"])
h.verification["positive"].outcome                 # "won" / "tied" / "lost"
h.verification["positive"].positive_strong_count

# 4. Audit a random sample of neurons in a layer
summary = s.injector.audit_layer_explanations(layer, n_neurons=50, seed=0,
                                              encode_fn=s.encode_text, decode_fn=s.decode_ids)
```

`examples/hf_workflow.py` runs the complete workflow on any HuggingFace causal LM and dataset: audit, sign regimes, weak-activation removal, double-negative ablation, extreme culling, an optional explanation audit and optional contrastive steering.

## Methods from the paper

| Paper | What is measured | API |
|---|---|---|
| Neurons as verbs: both poles matter | Exact moments, top-k and activation bands, kept separately for the positive and the negative pole | `audit_layers`, `get_neuron_profile` |
| Separating the sign regimes (SwiGLU / GeGLU) | Every activation is tagged `++`, `--`, `+-` or `-+` from the gate sign, with exact top-k, moments and histograms per regime | `NeuronPoleProfile.regime_top_samples`, `layer_regime_band_counts` |
| Activation rings as fractions of a neuron's maximum | Exact counts per fraction-of-extreme band (octave rings by default) and uniform examples inside each band | `activation_bands`, `DEFAULT_BAND_EDGES` |
| Percentiles vs raw activations | Within-pole percentiles from an exact log-magnitude histogram (bin resolution 2^(1/4)) | `percentile_within_pole`, `compute_thresholds` |
| Reading forward, summary motifs, long context | Every example carries its next token, a token window and the full sample text | `ActivationSample` |
| Don't say dot | Direct logit attribution, and the change in next-token log-probability mass of a token set under ablation | `top_suppressed_tokens`, `measure_token_set_effect` |
| Blind maximum; explanations that could lose | Agent 1 explains each pole from the evidence. Agent 2 sees only the explanation and writes a blind maximum, 5 positives, 5 near misses and 5 distant-concept texts. A response is **strong** at ≥ 30% of the pole's corpus extreme. The outcome is **won / tied / lost**, comparing strong positives with strong near misses | `explain_neuron`, `audit_layer_explanations` |
| Clustering the attribution flowing into a neuron | Gradient × activation attribution per activating context, cosine k-means on the attribution profiles, silhouette tested against a permutation null. Each bundle gets its own blind maximum while the other bundles are ablated | `cluster_incoming_attribution_bundles`, `run_bundles=True` |
| Steering as evidence | Greedy continuations with the neuron clamped to its corpus maximum or minimum, given to the explainer | `steering_prompts=` |
| Understanding compounds (crossword) | Explanations of neighbouring neurons, where neighbours come from traced circuit edges or shared traced readers | `solve_circuit_ecology`, `CircuitEcologyMap` |
| Do weak activations matter? | Remove each neuron's weak activations (below a fraction of its extreme or a within-pole percentile) and compare with removing the same number at random: NLL, KL, top-1 agreement and the margins of flipped predictions | `measure_weak_activation_ablation` |
| Double-negative ablation | Zero the `--` regime in any set of gated layers and evaluate held-out next-token prediction | `ablate_double_negative_regime`, `evaluate_intervention_effect` |
| Removing extreme responses | Exact per-neuron gate p99.9 on ordinary text. Culling above it is compared with count-matched and magnitude-matched random controls, optionally on a recitation prompt | `calibrate_tail_thresholds`, `run_extreme_culling_experiment` |
| Direct vs indirect effect (Poe "dreary") | Full ablation, removal of only the neuron's direct write a·W_down[:, i] at the final residual stream, and rescue by restoring downstream neurons | `decompose_direct_vs_indirect_effect` |
| Looking at the residual stream | Per-coordinate share of activation mass vs share of what all downstream readers read | `audit_residual_stream_sinks` |
| Nothing reads that combination | Maximum reader alignment of a neuron combination, compared with its best constituent and with random combinations of the same size; optional causal interaction | `verify_combination_readout` |
| Watching neurons develop | Per-checkpoint moments, write-vector drift and tokens per activation band | `track_checkpoint_evolution` |
| Experts are more specialised | Each expert of a MoE layer is its own neuron layer | layer discovery |

## Neural injection

All interventions act on the neuron activation before the down projection. Every intervention is an `InterventionSpec`, and the same spec runs as a live hook, goes into a preset, or is folded into the weights.

```python
inj = s.injector

# Ablate, clamp, shift or scale chosen neurons (optionally one pole, optionally at chosen positions)
inj.zero_ablate(layer, [123], pole=PoleType.NEGATIVE)
inj.clamp_neurons(layer, 123, value=p.max_val)            # native scale: the neuron's own corpus maximum
inj.steer_neurons(layer, [10, 42], delta=1.5)
inj.steer_neurons(layer, [10, 42], scale=0.5)

# Conditional gate: act only where a detector neuron fires above its own p99 (latched across KV-cached decoding)
inj.add_conditional_gate(target_layer=layer, target_neurons=[10, 42], detector_layer=layers[6], detector_neuron=470,
                         detector_percentile=99.0, value=2.0)

# Contrastive discovery: Welch's t on per-prompt means between two prompt sets, then steer along the contrast
pos = [s.encode_prompt(t)[0] for t in code_prompts]
neg = [s.encode_prompt(t)[0] for t in prose_prompts]
report = inj.discover_contrastive_neurons(pos, neg, layer_names=layers, top_k=10)
inj.inject_contrastive_behavior(report, strength=1.0)      # shift by strength · (mean_pos − mean_neg)

s.generate_comparison("The museum reopened in", max_new_tokens=40)   # clean vs intervened continuation
inj.evaluate_intervention_effect(held_out_batches)                    # NLL, KL, top-1 agreement vs clean
inj.clear_interventions()
```

The contrastive report gives, for each neuron, the t statistic, degrees of freedom, Cohen's d, two-sided p-value and Bonferroni-corrected p-value. `find_synergistic_ablation_set` searches for a small coalition of neurons whose joint ablation gets past backup circuits that absorb single ablations, and reports the joint effect next to the sum of the individual effects.

### Exporting and deploying

```python
inj.export_production_preset("code_style", model_name="SmolLM2-135M",
                             save_json_path="presets/code_style.json")

# Apply a preset to a fresh model without the injector
from neural_injector import ProductionSteeringRuntime
rt = ProductionSteeringRuntime(model, "presets/code_style.json")
with rt.active(use_compiled_residual_vectors=True):   # additive steering as one residual vector per layer
    out = model.generate(...)

# Or bake unconditional affine interventions into W_down and its bias: no hooks at inference
inj.load_production_preset("presets/code_style.json")
inj.fold_interventions_into_weights()
```

Compiled vectors and folding reproduce the hooked model up to floating-point precision. Compiling applies to unconditional additive steering. Folding applies to every unconditional affine intervention (ablate, clamp, scale, shift). Interventions restricted to one pole or to chosen positions stay as hooks, as do gate-conditioned modes and conditional gates.

## CLI and web workbench

```bash
neural-injector --hf-model HuggingFaceTB/SmolLM2-135M repl
```

```bash
neural-injector --hf-model HuggingFaceTB/SmolLM2-135M serve --port 8000
```

REPL commands: `layers`, `audit <layers> @corpus.txt`, `scan`, `profile`, `llm`, `explain`, `trace`, `trace_incoming`, `readout`, `contrastive`, `synergy`, `ablate`, `clamp`, `steer`, `prune`, `dblneg`, `gate`, `eval`, `generate`, `export`, `import`, `fold`. A layer must be audited (`audit`) before its statistics are available.

The explainer endpoint can be set with `--llm-model`, `--llm-base-url` and `--llm-api-key`, or with the environment variables `NEURAL_INJECTOR_LLM_MODEL`, `NEURAL_INJECTOR_LLM_BASE_URL` and `OPENAI_API_KEY`.

## Parameters

These values come from the paper and are used as given:
- the 30% strong-response threshold (`PAPER_STRONG_FRACTION`);
- 5 positive, 5 near-miss and 5 distant texts per test;
- gate p99.9 for culling;
- "keep roughly the strongest quarter" (weak-activation percentile 75 in the example workflow).

Everything else is an explicit argument with a stated meaning:
- histogram bins per octave;
- band edges (octave rings by default);
- examples per band;
- number of random control neurons;
- permutation and null sample counts;
- the significance level (`alpha = 0.05`).

Results are reported as separate measurements, never combined into a weighted score. A neuron is described only through explanations that have been tested, never through labels inferred from its activation statistics.

## Limitations

- Explanations need an LLM, or a human through `CallableLLMClient`.
- Forward-mode Jacobians and gradient attribution are not available for routed MoE expert rows. Ablation-based tracing works there.
- Removing the direct path requires a neuron layer that is hooked at the down-projection input.
- Attention heads are traced only through the neurons they affect.
- Random controls in the weak-activation and culling experiments are drawn over all positions of a batch. Use unpadded batches (batch size 1) so that padding positions are never chosen.

## Reference

Nick Cammarata. *Neural Networks Are Already Clean: A working hypothesis about neurons, generalization, and the ecology of circuits.* Goodfire, October 2026.
