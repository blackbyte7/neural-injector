# Neural Injector (`neural-injector`)

**Universal Bipolar Neuron Auditor, Surgical Activation Injector, Circuit Tracer, Agentic Explainer & Interactive Workbench for PyTorch.**

`neural-injector` is a mechanistic interpretability and activation-steering framework implementing the concepts and algorithms from Nick Cammarata’s Goodfire research essay,
***"Neural Networks Are Already Clean: A working hypothesis about neurons, generalization, and the ecology of circuits"*** (October 2026).

Instead of training external Sparse Autoencoders (SAEs) to disentangle assumed superposition, `neural-injector` operates directly on the **native neuron basis**—treating individual neurons as bipolar differential routers ("verbs") with distinct **Positive** and **Negative** poles.

---

## Key Capabilities

1. **Universal Model-Agnostic Layer Discovery (`discovery.py`)**
    * Automatically resolves targetable neuron bases across HuggingFace causal LLMs (`Qwen`, `Llama`, `Mistral`, `Phi-3`, `GPT-NeoX`, `GPT-2`), Vision Transformers (`ViT`), CNNs (`Conv2d`), and custom MLPs.
    * Hooks the *input* of `down_proj` / `c_proj` / `fc2` layers to capture exact post-SwiGLU/GELU neuron activations before they mix into the residual stream.
    * Supports fused `gate_up_proj` layers (`[2 * d_mlp, d_model]`), 3D Mixture-of-Experts (`MoE`) expert tensors (`[num_experts, d_model, d_mlp]`), `GPT-2` `Conv1D` layers, and disambiguates MLP `c_proj` modules from attention output projections.
    * Resolves pre-MLP and final `RMSNorm` / `LayerNorm` modules to apply exact $\gamma$ weight scaling during virtual weight and logit-lens projections.
2. **O(1) Memory Bipolar & Pole-Specific Neuron Auditing (`analyzer.py`)**
    * Uses batched GPU Welford/Pébay parallel online moments ($M_1, M_2, M_3, M_4$) to stream arbitrarily large corpora without storing activation tensors in VRAM, automatically filtering `<pad>` tokens via `attention_mask`.
    * Tracks conditional pole-specific moments $(M_1^+, M_2^+, M_3^+, M_4^+)$ for $a > 0$ and $(M_1^-, M_2^-, M_3^-, M_4^-)$ for $a < 0$ so **Positive** and **Negative** poles are evaluated and classified independently (`pos_pole_selectivity_score`, `neg_pole_selectivity_score`, `pos_taxonomy_class`, `neg_taxonomy_class`).
    * Tracks separate **Positive Pole ($a > 0$)** and **Negative Pole ($a < 0$)** top-$k$ contexts alongside **Direct Weight-to-Vocabulary Logit Attribution** ($W_{\text{down}}[:, i] \cdot W_U$).
3. **Stratified Quantile Activation Buckets for Recursive Offshoots (`analyzer.py`)**
    * Partitions active firing samples per pole into stratified quantile bands: `peak_trunk` (`[80%..100%]` of max magnitude), `secondary_branch` (`[40%..80%)`), and `tertiary_offshoot` (`[15%..40%)`).
    * Prevents high-magnitude trunk tokens from crowding out secondary and tertiary recursive generalizations when presenting evidence to the Explainer Agent.
4. **Batch Layer-Wide & Pole-Wide Cleanliness Scanner (`analyzer.py`, `injector.py`)**
    * Scans all neurons and all $2 \times N$ poles in a layer in parallel using standardized 4th-moment tail selectivity (kurtosis), bipolar router balance, and outgoing residual write norms ($\sigma_i \Vert W_{\text{down}}[:, i]\Vert_2$).
    * Classifies every neuron and pole into `clean_monosemantic`, `structured_recursive_mixture`, `true_polysemantic`, or `dead_or_noisy`, reporting both whole-neuron (`clean_monosemantic_pct`) and independent pole (`clean_poles_pct`) percentages.
5. **Vectorized Surgical Interventions, Adaptive Thresholds & Closed-Loop Gating (`interventions.py`)**
    * Zero-ablate (`x = 0`), hard-clamp (`x = c`), scale (`x * alpha`), additively steer (`x + delta`), pole-rectify, or prune weak background activations (`|a| < threshold`) selectively on the **Positive Pole**, **Negative Pole**, or **Both Poles**.
    * Supports native **per-neuron adaptive thresholds** (`per_neuron_values`, `std_multiplier`, or `percentile`) derived from each neuron's audited activation distribution.
    * Supports **Closed-Loop Dynamic Concept Gating (`CONDITIONAL_GATE`)**: conditionally steers, clamps, or ablates target neurons in Layer $L_{\text{dst}}$ only when an upstream or same-layer detector neuron in Layer $L_{\text{det}}$ fires above a threshold on a specified pole.
6. **Single-Pass Forward-Mode AD Circuit Tracer & Superposition Readout Auditor (`circuit_tracer.py`)**
    * Combines static residual-stream virtual weights ($W_{\text{up}}$ value coupling and $W_{\text{gate}}$ switch coupling) with dynamic causal pole-ablation patching.
    * Uses **Forward-Mode Automatic Differentiation (`torch.autograd.forward_ad`)** (`trace_forward_ad_all_layers`) to compute exact Jacobians ($\partial A_{\text{dst}} / \partial a_{\text{src}}$) across **every downstream neuron and layer simultaneously** in a single forward pass.
    * Includes `verify_downstream_readout()` to audit Cammarata's *"Nothing Reads That Combination"* finding—testing whether weak cross-activations across a group of upstream neurons actually cross the gating/activation threshold of any downstream neuron.
7. **Crossword Circuit Ecology Solver & Closed-Loop Agent Verifier (`explainer.py`)**
    * Synthesizes a unified bipolar "verb" hypothesis from a neuron's profile, stratified quantile buckets, and downstream circuit graph, then automatically designs and executes surgical ablation and pole-flip experiments to empirically verify its own hypothesis.
    * Implements a persistent **`CircuitEcologyMap`** and two-pass **`solve_layer_ecology()`** solver (*"Understanding Compounds"*): explains high-selectivity monosemantic "anchor" neurons first and injects their verified motifs into the LLM prompts of connected upstream/downstream neurons.
    * Enforces strict Pydantic schema validation (`AgentHypothesisResponseSchema`) and exponential backoff retries on OpenAI-compatible LLM endpoints.
8. **Contrastive Pole Discovery, Checkpoint Evolution & Synergistic Group Ablation (`analyzer.py`, `interventions.py`)**
    * **Contrastive Pole Discovery (`discover_contrastive_neurons`)**: Ranks all native neurons across layers by Welch’s Bipolar Separation $t$-statistic between contrastive prompt sets $D^+$ and $D^-$.
    * **Cross-Checkpoint Recursive Growth Tracker (`track_checkpoint_evolution`)**: Tracks how a neuron’s pole selectivity, outgoing weight geometry, and stratified quantile offshoots sprout across training checkpoints or LoRA fine-tuning stages.
    * **Synergistic Multi-Neuron Ablation (`find_synergistic_ablation_set`)**: Finds minimal $k$-neuron coalitions (via greedy forward search or differentiable $L_1$-sparse continuous mask optimization) whose joint ablation overcomes backup circuits.
9. **Production Steering Vector Export & Zero-Overhead Weight Folding (`exporter.py`)**
    * Compiles unconditional `ADDITIVE_STEER` and `HARD_CLAMP` interventions into dense residual-stream steering vectors ($v_{\text{steer}} \in \mathbb{R}^{d_{\text{model}}}$), tracking `clamped_neurons` so dynamic activations are zeroed before adding $v_{\text{steer}}$.
    * Exports portable `.json` and `.pt` presets for `ProductionSteeringRuntime` (with workspace path-traversal protection), or permanently folds interventions into `down_proj` (`Linear` and `Conv1D`) weights and biases for **0% runtime hook overhead**.
10. **Interactive CLI REPL & Thread-Safe FastAPI Web Workbench (`cli.py`, `server.py`)**
    * Inspect neuron poles and quantile buckets, view layer-wide and pole-wide cleanliness heatmaps, inject surgical clamps or conditional gates, trace circuits, solve crossword ecology, and export presets from an interactive terminal shell or dark-mode browser dashboard.

---

## Project Structure

```text
neural_injector/
├── pyproject.toml
├── requirements.txt
├── README.md
├── main_demo.py               # End-to-end demo: auditing, quantile buckets, adaptive pruning, conditional gates & contrastive discovery
├── demo_circuit_tracer.py     # Demo: single-pass forward-AD Jacobian tracing, pole circuits & downstream readout auditing
├── demo_agent_explainer.py    # Demo: agentic explainer, surgical verification, crossword ecology & checkpoint evolution
├── demo_scan_and_export.py    # Demo: layer/pole cleanliness scan, synergistic group ablation, compiled vectors & weight folding
├── neural_injector/
│   ├── __init__.py
│   ├── types.py               # Dataclasses & enums for poles, circuits, ecology, hypotheses, contrastive/evolution reports & presets
│   ├── discovery.py           # Universal architecture-agnostic layer, norm, Conv1D, fused gate_up & 3D MoE resolver
│   ├── analyzer.py            # GPU Welford/Pébay profiler, pole moments, quantile buckets, logit lens, contrastive & checkpoint trackers
│   ├── interventions.py       # Vectorized pole-aware GPU kernels, adaptive thresholds, conditional gates & AblationMaskOptimizer
│   ├── circuit_tracer.py      # Virtual weights + causal pole patching + single-pass forward-mode AD Jacobians + readout auditor
│   ├── explainer.py           # Automated LLM-agent explainer, Pydantic validator, CircuitEcologyMap & closed-loop surgical verifier
│   ├── exporter.py            # Residual steering vector compiler, path-safe preset runtime & zero-overhead weight folder
│   ├── injector.py            # Thread-safe main NeuralInjector orchestrator
│   ├── session.py             # Stateful WorkbenchSession (built-in demo & HuggingFace loader with pad-masking)
│   ├── server.py              # Async-locked FastAPI REST backend & single-page Web Workbench UI
│   └── cli.py                 # CLI commands & interactive REPL shell
└── tests/
    └── test_neural_injector.py
```

---

## Installation

```bash
# Core library (requires PyTorch >= 2.1, FastAPI, Uvicorn, Pydantic)
pip install -e .

# With HuggingFace Transformers/Accelerate support and pytest/httpx test suite
pip install -e ".[huggingface,dev]"
```

---

## Quickstart (Python SDK)

```python
import torch
from neural_injector import (
    InterventionMode,
    NeuralInjector,
    PoleType,
    ProductionSteeringRuntime,
    WorkbenchSession,
)

# 1. Load a pre-calibrated demo session or any HuggingFace causal LLM:
# session = WorkbenchSession.from_huggingface("Qwen/Qwen2.5-1.5B", device="cuda")
session = WorkbenchSession.create_demo_session()
injector = session.injector
src_layer = "mlp_block_0.down_proj"
dst_layer = "mlp_block_1.down_proj"

# 2. Scan all neurons AND individual poles in the layer for cleanliness & causal impact
input_ids, _ = session.encode_prompt("the system negotiated AES256 TLS cipher")
sql_ids, _ = session.encode_prompt("the system executed SELECT WHERE postgres")

scan = injector.scan_layer_cleanliness(
    src_layer,
    sort_by="estimated_causal_impact",
    compute_kl_for_top_n=3,
    kl_eval_inputs=input_ids,
    limit=5,
)
print(f"Combined Clean + Structured Neurons: {scan.combined_clean_and_structured_pct}%")
print(f"Combined Clean + Structured Poles  : {scan.combined_clean_and_structured_poles_pct}%")

# 3. Inspect a single neuron's Positive & Negative poles, Quantile Buckets & Vocab Attribution
profile = injector.get_neuron_profile(
    layer_name=src_layer,
    neuron_idx=7,
    unembed_weight=session.unembed_weight,
    idx_to_token=session.id_to_tok,
)
print(f"+Pole Selectivity: {profile.pos_pole_selectivity_score} ({profile.pos_taxonomy_class.value})")
print(f"-Pole Selectivity: {profile.neg_pole_selectivity_score} ({profile.neg_taxonomy_class.value})")
print("Promoted by +Pole:", profile.top_promoted_tokens[:3])
print("Promoted by -Pole:", profile.top_suppressed_tokens[:3])
print("Stratified +Pole Quantile Bands:", list(profile.positive_quantile_buckets.keys()))

# 4. Surgically Zero-Ablate, Prune with Native Adaptive Thresholds, or Add a Conditional Gate
injector.clear_interventions()
injector.prune_weak_activations(src_layer, neuron_indices=list(range(128)), std_multiplier=0.75)

# Closed-Loop Conditional Gate: only steer L1 N12 when Detector L0 N7 fires positively (> 1.0)
injector.add_conditional_gate(
    target_layer=dst_layer,
    target_neurons=[12],
    detector_layer=src_layer,
    detector_neuron=7,
    detector_pole=PoleType.POSITIVE,
    threshold=1.0,
    action_mode=InterventionMode.ADDITIVE_STEER,
    value=15.0,
    sequence_scope="any_token",
)
with injector.active_injection():
    gated_logits = session.model(input_ids)

# 5. Trace downstream circuit connections & Single-Pass Forward-Mode AD Jacobians
jac_map = injector.trace_forward_ad_all_layers(
    src_layer=src_layer, src_neuron=7, inputs=input_ids, src_pole=PoleType.POSITIVE
)
circuit = injector.trace_circuit(
    src_layer=src_layer,
    dst_layer=dst_layer,
    src_neuron=7,
    inputs=input_ids,
    src_pole=PoleType.POSITIVE,
)

# 6. Audit Superposition Hiding Places ("Nothing Reads That Combination")
readout = injector.verify_downstream_readout(
    src_layer=src_layer, dst_layer=dst_layer, src_neurons=[0, 1, 2], inputs=input_ids
)
print("Weak Cross-Activation Is Benign Spillover:", readout.is_benign_spillover)

# 7. Run the Closed-Loop Agent Explainer & Crossword Circuit Ecology Solver
ecology_report = injector.solve_layer_ecology(
    layer_name=src_layer,
    tok_to_id=session.tok_to_id,
    idx_to_token=session.id_to_tok,
    unembed_weight=session.unembed_weight,
    dst_layer=dst_layer,
    circuit_inputs=input_ids,
    top_anchors=2,
    top_contextual=2,
)
hypothesis = ecology_report.hypotheses[7]
print("Unified Verb Hypothesis:", hypothesis.unified_verb_hypothesis)
print("Verification Score:", hypothesis.verification_report.empirical_verification_score)

# 8. Contrastive Pole Discovery & Synergistic Multi-Neuron Ablation
contrastive = injector.discover_contrastive_neurons(
    pos_inputs=input_ids, neg_inputs=sql_ids, layer_names=[src_layer], top_k=3
)
synergy = injector.find_synergistic_ablation_set(
    layer_name=src_layer, inputs=input_ids, max_subset_size=2, method="greedy"
)

# 9. Export Production Steering Preset or Fold Directly into Model Weights
injector.clear_interventions()
injector.steer_neurons(src_layer, neuron_indices=[7], delta=-15.0)
injector.export_production_preset(
    preset_name="crypto_to_sql",
    save_json_path="preset.json",
    save_pt_vectors_path="vectors.pt",
)
injector.fold_interventions_into_weights()  # 0% runtime hook overhead!
```

---

## Running the Included Demo Scripts

```bash
# 1. Core pipeline: discovery, pole-specific auditing, quantile bands, adaptive pruning, conditional gating & contrastive discovery
python main_demo.py

# 2. Circuit tracing: single-pass forward-mode AD Jacobians, bipolar edge roles & downstream readout verification
python demo_circuit_tracer.py

# 3. Agentic explainer: closed-loop verification, crossword ecology propagation & cross-checkpoint growth tracking
python demo_agent_explainer.py

# 4. Layer/pole cleanliness scanning, synergistic group ablation, compiled steering vectors & permanent weight folding
python demo_scan_and_export.py
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
* `scan <layer> [sort_by] [top_n]` — Run a batch layer-wide and pole-specific cleanliness audit.
* `profile <layer> <neuron_idx>` — Inspect bipolar contexts, pole selectivities, stratified quantile buckets, and direct logit attribution.
* `ablate <layer> <neuron_idx> [both|positive|negative]` — Set a neuron or pole to `0.0`.
* `clamp <layer> <neuron_idx> <value> [pole]` — Clamp a neuron or pole to a constant scalar.
* `prune <layer> <val> [adaptive]` — Prune weak activations across the layer using a scalar threshold or native per-neuron `std_multiplier`.
* `gate <target_layer> <target_neuron> <det_layer> <det_neuron> <delta> [threshold]` — Register a closed-loop conditional gate.
* `clear` — Remove all active surgical interventions.
* `eval <prompt>` — Compare Clean vs. Surgically Intervened token probabilities and KL divergence.
* `trace <src_layer> <dst_layer> <neuron_idx> [pole] [prompt]` — Trace excitatory, inhibitory, and gating edges with forward-mode AD Jacobians.
* `readout <src_layer> <dst_layer> <neuron_indices_csv> [threshold]` — Audit whether weak cross-activations are read downstream (*"Nothing Reads That Combination"*).
* `explain <layer> <neuron_idx> [dst_layer]` — Run the LLM agent explainer and surgical verification loop.
* `ecology <layer> [dst_layer] [top_anchors] [top_contextual]` — Solve layer circuit ecology via two-pass Crossword propagation.
* `contrastive <pos_prompt> | <neg_prompt>` — Discover native neurons separating two contrastive prompts via Welch's $t$-statistic.
* `synergy <layer> [k] [greedy|l1_optim]` — Find a minimal synergistic coalition of $k$ neurons to ablate.
* `export <json_path> [pt_path]` / `import <json_path>` / `fold` — Export, load, or permanently weight-fold steering presets.

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