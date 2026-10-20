# Neural Injector (`neural-injector`)

**Universal Bipolar Neuron Auditor, Surgical Activation Injector, Circuit Tracer, Agentic Explainer & Interactive
Workbench for PyTorch.**

`neural-injector` is a mechanistic interpretability and activation-steering framework implementing the concepts and
algorithms from Nick Cammarata’s Goodfire research essay,
***"Neural Networks Are Already Clean: A working hypothesis about neurons, generalization, and the ecology of circuits"
*** (October 2026).

Instead of training external Sparse Autoencoders (SAEs) to disentangle assumed superposition, `neural-injector` operates
directly on the **native neuron basis**—treating individual neurons as bipolar differential routers ("verbs") with
distinct **Positive** and **Negative** poles.

---

## Key Capabilities

1. **Universal Layer Discovery (`discovery.py`)**
    * Automatically resolves targetable neuron bases across HuggingFace causal
      LLMs (`Qwen`, `Llama`, `Mistral`, `GPT-2`), Vision Transformers (`ViT`), CNNs (`Conv2d`), and custom MLPs.
    * Hooks the *input* of `down_proj` / `c_proj` / `fc2` layers to capture exact post-SwiGLU/GELU neuron activations
      before they mix into the residual stream.
2. **O(1) Memory Bipolar Neuron Auditing (`analyzer.py`)**
    * Uses batched GPU Welford online moments ($M_1, M_2, M_4$) to stream arbitrarily large corpora without storing
      activation tensors in VRAM.
    * Tracks separate **Positive Pole ($a > 0$)** and **Negative Pole ($a < 0$)** min-heaps of top-activating tokens and
      contexts, plus **Direct Weight-to-Vocabulary Logit Attribution** ($W_{\text{down}}[:, i] \cdot W_U$).
3. **Batch Layer-Wide Cleanliness Scanner (`analyzer.py`, `injector.py`)**
    * Scans all neurons in a layer in parallel using standardized 4th-moment tail selectivity (kurtosis), bipolar router
      balance, and outgoing residual write norms ($\sigma_i \Vert{}W_{\text{down}}[:, i]\Vert{}_2$).
    * Classifies every neuron into `clean_monosemantic`, `structured_recursive_mixture`, `true_polysemantic`,
      or `dead_or_noisy`.
4. **Vectorized Surgical Neuron Interventions (`interventions.py`)**
    * Zero-ablate (`x = 0`), hard-clamp (`x = c`), scale (`x * alpha`), additively steer (`x + delta`), or prune weak
      background activations (`|a| < threshold`) selectively on the **Positive Pole**, **Negative Pole**, or **Both
      Poles**.
5. **Layer-to-Layer Circuit Tracer (`circuit_tracer.py`)**
    * Combines static residual-stream virtual weights ($W_{\text{up}}$ value coupling and $W_{\text{gate}}$ switch
      coupling) with dynamic causal pole-ablation patching and exact autograd Jacobians ($\partial A_{\text{dst}} /
      \partial A_{\text{src}}$).
6. **Closed-Loop LLM Agent Explainer & Verifier (`explainer.py`)**
    * Synthesizes a unified bipolar "verb" hypothesis from a neuron's profile and downstream circuit graph, then
      automatically designs and executes surgical ablation and pole-flip experiments to empirically verify its own
      hypothesis.
7. **Production Steering Vector Export & Zero-Overhead Weight Folding (`exporter.py`)**
    * Compiles multi-neuron interventions into dense residual-stream steering vectors ($v_{\text{steer}} \in
      \mathbb{R}^{d_{\text{model}}}$), exports portable `.json` and `.pt` presets for `ProductionSteeringRuntime`, or
      permanently folds interventions into `down_proj` weights and biases for **0% runtime hook overhead**.
8. **Interactive CLI REPL & FastAPI Web Workbench (`cli.py`, `server.py`)**
    * Inspect neuron poles, view layer-wide cleanliness heatmaps, inject surgical clamps, trace circuits, and export
      presets from either an interactive terminal shell or a dark-mode browser dashboard.

---

## Project Structure

```text
neural_injector/
├── pyproject.toml
├── requirements.txt
├── README.md
├── neural_injector/
│   ├── __init__.py
│   ├── types.py           # Dataclasses & enums for poles, circuits, hypotheses, and presets
│   ├── discovery.py       # Architecture-agnostic neuron layer & sibling module resolver
│   ├── analyzer.py        # GPU Welford streaming profiler, logit lens & layer-wide scanner
│   ├── interventions.py   # Vectorized pole-aware GPU intervention kernels
│   ├── circuit_tracer.py  # Virtual weights + causal pole patching + local Jacobians
│   ├── explainer.py       # Automated LLM-agent explainer & closed-loop surgical verifier
│   ├── exporter.py        # Residual steering vector compiler, preset runtime & weight folder
│   ├── injector.py        # Main NeuralInjector orchestrator
│   ├── session.py         # Stateful WorkbenchSession (built-in demo & HuggingFace loader)
│   ├── server.py          # FastAPI REST backend & single-page Web Workbench UI
│   └── cli.py             # CLI commands & interactive REPL shell
└── tests/
    └── test_neural_injector.py
```

---

## Installation

```bash
# Core library (requires only PyTorch >= 2.1)
pip install -e .

# With FastAPI Web Workbench, CLI, HuggingFace support, and pytest
pip install -e ".[all,dev]"
```

---

## Quickstart (Python SDK)

```python
import torch
from neural_injector import NeuralInjector, PoleType, WorkbenchSession

# 1. Load a pre-calibrated demo session or any HuggingFace causal LLM:
# session = WorkbenchSession.from_huggingface("Qwen/Qwen2.5-1.5B", device="cuda")
session = WorkbenchSession.create_demo_session()
injector = session.injector
layer = "mlp_block_0.down_proj"

# 2. Scan all neurons in the layer for cleanliness & causal impact
input_ids, _ = session.encode_prompt("the system negotiated AES256 TLS cipher")
scan = injector.scan_layer_cleanliness(layer, sort_by="estimated_causal_impact", limit=5)
print(f"Combined Clean + Structured Neurons: {scan.combined_clean_and_structured_pct}%")

# 3. Inspect a single neuron's Positive & Negative poles + Vocabulary Attribution
profile = injector.get_neuron_profile(
    layer_name=layer,
    neuron_idx=7,
    unembed_weight=session.unembed_weight,
    idx_to_token=session.id_to_tok,
)
print("Promoted by +Pole:", profile.top_promoted_tokens[:3])
print("Promoted by -Pole:", profile.top_suppressed_tokens[:3])

# 4. Surgically Zero-Ablate or Clamp a specific pole during inference
injector.clear_interventions()
injector.zero_ablate(layer, neuron_indices=7, pole=PoleType.POSITIVE)
with injector.active_injection():
    ablated_logits = session.model(input_ids)

# 5. Trace downstream circuit connections into Layer L+1
circuit = injector.trace_circuit(
    src_layer="mlp_block_0.down_proj",
    dst_layer="mlp_block_1.down_proj",
    src_neuron=7,
    inputs=input_ids,
    src_pole=PoleType.POSITIVE,
)

# 6. Run the Closed-Loop Agent Explainer & Empirical Verifier
hypothesis = injector.explain_neuron(
    layer_name="mlp_block_0.down_proj",
    neuron_idx=7,
    tok_to_id=session.tok_to_id,
    idx_to_token=session.id_to_tok,
    unembed_weight=session.unembed_weight,
    dst_layer="mlp_block_1.down_proj",
    circuit_inputs=input_ids,
)
print("Unified Verb Hypothesis:", hypothesis.unified_verb_hypothesis)
print("Verification Score:", hypothesis.verification_report.empirical_verification_score)

# 7. Export Production Steering Preset or Fold Directly into Model Weights
injector.clear_interventions()
injector.steer_neurons(layer, neuron_indices=[7], delta=-15.0)
injector.export_production_preset(
    preset_name="crypto_to_sql",
    save_json_path="preset.json",
    save_pt_vectors_path="vectors.pt",
)
injector.fold_interventions_into_weights()  # 0% runtime hook overhead!
```

---

## Interactive CLI & FastAPI Web Workbench

### Launch the Interactive Terminal REPL

```bash
neural-injector repl
# Or with a HuggingFace model:
neural-injector --hf-model gpt2 --device cuda repl
```

Available REPL commands:

* `layers` — List all targetable neuron bases and hook targets.
* `scan <layer> [sort_by] [top_n]` — Run a batch layer-wide cleanliness audit.
* `profile <layer> <neuron_idx>` — Inspect bipolar contexts and direct logit attribution.
* `ablate <layer> <neuron_idx> [both|positive|negative]` — Set a neuron or pole to `0.0`.
* `clamp <layer> <neuron_idx> <value> [pole]` — Clamp a neuron or pole to a constant scalar.
* `prune <layer> <threshold>` — Prune weak activations (`|a| < threshold`) across the layer.
* `eval <prompt>` — Compare Clean vs. Surgically Intervened token probabilities and KL divergence.
* `trace <src_layer> <dst_layer> <neuron_idx> [pole]` — Trace excitatory, inhibitory, and gating edges.
* `explain <layer> <neuron_idx> [dst_layer]` — Run the LLM agent explainer and surgical verification loop.
* `export <json_path> [pt_path]` / `import <json_path>` / `fold` — Export, load, or weight-fold steering presets.

### Launch the FastAPI Server & Browser Dashboard

```bash
neural-injector serve --host 127.0.0.1 --port 8000
```

* **Web UI Workbench:** `http://127.0.0.1:8000/`
* **OpenAPI / Swagger Docs:** `http://127.0.0.1:8000/docs`

---

## Running the Test Suite

```bash
pytest tests/test_neural_injector.py -v
```