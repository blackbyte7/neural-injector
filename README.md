# Neural Injector (`neural-injector`)

**Universal Bipolar Neuron Auditor, Surgical Activation Injector, Circuit Tracer, Agentic Explainer & Interactive Workbench for PyTorch.**

**What is `neural-injector`?** `neural-injector` is a toolkit for looking inside PyTorch and HuggingFace LLMs to find the exact native neurons responsible for specific concepts, reasoning habits, or unsafe behaviors—and surgically steering, clamping, or deactivating them in real time to change how the model thinks and responds. Built on Nick Cammarata’s Goodfire research essay, ***"Neural Networks Are Already Clean: A working hypothesis about neurons, generalization, and the ecology of circuits"*** (October 2026), it bypasses expensive external Sparse Autoencoders (SAEs) to let you audit, trace, and permanently edit a model's native neurons directly.

---

## Why `neural-injector` vs. the Standard Interpretability Stack?

Traditional mechanistic interpretability tools assume that native MLP neurons are hopelessly tangled in superposition, requiring you to train or download massive Sparse Autoencoders (SAEs) for every layer before you can inspect or steer a model. `neural-injector` leverages the finding that **~90% of native neuron poles already act as clean, interpretable differential routers ("verbs")**:

| Capability / Metric             | Standard SAE Stack (`SAELens` / `TransformerLens`)                               | Coarse Vector Steering (`repeng` / PCA Vectors)                      | `neural-injector` (Native Bipolar Basis)                                                      |
|:--------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| **Upfront Training Cost**       | **100s–1,000s of GPU-hours** to train SAEs per model checkpoint                  | None                                                                 | **0 GPU-hours** (operates directly on native weights & activations)                           |
| **VRAM Auditing Overhead**      | **High** ($O(B \cdot S \cdot d_{\text{sae}})$ activation caching; frequent OOMs) | Low                                                                  | **$O(1)$ VRAM** (< 2 MB/layer via GPU Welford/Pébay streaming moments)                        |
| **Granularity & Precision**     | Overcomplete dictionary latents (often splits 1 neuron into 10s of fragments)    | Coarse full-layer residual vectors (no neuron or circuit visibility) | **Exact single-neuron Positive ($+$) & Negative ($-$) poles + SwiGLU gate/up routes**         |
| **Multi-Layer Circuit Tracing** | Slow ($O(N)$ backward passes or expensive attribution patching)                  | Not supported                                                        | **1 Forward Pass** via Forward-Mode AD (`torch.autograd.forward_ad`) + Upstream decomposition |
| **Collateral Damage Control**   | Static clamping across all tokens                                                | Static vector addition across all tokens (degrades general fluency)  | **Closed-Loop Conditional Gating (`CONDITIONAL_GATE`)** with KV-cache latching                |
| **Production Serving Latency**  | Requires running external SAE encoder/decoder hooks at inference                 | Requires runtime residual-stream hooks                               | **0.0% Runtime Overhead** via permanent weight folding into `down_proj`                       |

---

## Key Capabilities

1. **Universal Model-Agnostic Layer Discovery (`discovery.py`)**
    * Automatically resolves targetable neuron bases across HuggingFace causal LLMs (`Qwen`, `Llama`, `Mistral`, `Phi-3`, `GPT-NeoX`, `GPT-2`), Vision Transformers (`ViT`), CNNs (`Conv2d`), and custom MLPs.
    * Hooks the *input* of `down_proj` / `c_proj` / `fc2` layers to capture exact post-SwiGLU/GELU neuron activations before they mix into the residual stream.
    * Supports split `(gate_proj, up_proj)` and fused `gate_up_proj` layers (`[2 * d_mlp, d_model]`), 3D Mixture-of-Experts (`MoE`) expert tensors (`[num_experts, d_model, d_mlp]`), `GPT-2` `Conv1D` layers, and pre-MLP / final `RMSNorm` & `LayerNorm` $\gamma$ scaling.
2. **O(1) Memory Bipolar & Pole-Specific Neuron Auditing (`analyzer.py`)**
    * Uses batched GPU Welford/Pébay parallel online moments ($M_1, M_2, M_3, M_4$) to stream arbitrarily large corpora without storing activation tensors in VRAM, automatically filtering `<pad>` tokens via `attention_mask`.
    * Tracks conditional pole-specific moments $(M_1^+, M_2^+, M_3^+, M_4^+)$ for $a > 0$ and $(M_1^-, M_2^-, M_3^-, M_4^-)$ for $a < 0$ so **Positive** and **Negative** poles are evaluated and classified independently (`pos_pole_selectivity_score`, `neg_pole_selectivity_score`, `pos_taxonomy_class`, `neg_taxonomy_class`).
    * **SwiGLU Double-Negative Route Detection**: Automatically detects when both `gate < 0` and `up < 0` multiply to produce a positive activation ($\text{SiLU}(g) \cdot u > 0$), separating double-negative leakages (`double_negative_positive_rate`, `double_negative_samples`) from true positive (`gate > 0, up > 0`) quantile bands.
    * **Forward Next-Token Context ("Titanic $\to$ Boat")**: Records `next_token_str` ($t_{\text{pos}+1}$) and highlighted local windows (`"... prev [FOCAL] -> next ..."`) on every `ActivationSample` to distinguish backward feature detectors from forward next-token preparers.
3. **Stratified Quantile Activation Buckets & Unbiased Percentiles (`analyzer.py`)**
    * Partitions active firing samples per pole into quota-protected stratified quantile bands across the full batch distribution: `peak_trunk` (`[80%..100%]` of max magnitude), `secondary_branch` (`[40%..80%)`), and `tertiary_offshoot` (`[15%..40%)`).
    * Uses unbiased Algorithm R reservoir sampling over active magnitudes for accurate empirical percentile thresholds (`compute_adaptive_thresholds(percentile=...)`).
4. **Batch Layer-Wide & Pole-Wide Cleanliness Scanner (`analyzer.py`, `injector.py`)**
    * Scans all neurons and all $2 \times N$ poles in a layer in parallel using standardized 4th-moment tail selectivity (kurtosis), bipolar router balance, and outgoing residual write norms ($\sigma_i \Vert W_{\text{down}}[:, i]\Vert_2$) as a fast statistical prior, then promotes neurons that have been semantically verified in `CircuitEcologyMap` (`semantic_verified=True`).
5. **Vectorized Surgical Interventions, Adaptive Thresholds & Latched Conditional Gating (`interventions.py`)**
    * Zero-ablate (`x = 0`), hard-clamp (`x = c`), scale (`x * alpha`), additively steer (`x + delta`), pole-rectify, or prune weak background activations (`|a| < threshold`) selectively on the **Positive Pole**, **Negative Pole**, or **Both Poles**.
    * **Closed-Loop Dynamic Concept Gating (`CONDITIONAL_GATE`)**: Conditionally steers, clamps, or ablates target neurons in Layer $L_{\text{dst}}$ only when an upstream or same-layer detector neuron in Layer $L_{\text{det}}$ fires above a threshold on a specified pole. Includes **KV-Cache Latching (`sticky_on_decode=True`)** so prompt triggers stay latched open across `seq_len == 1` autoregressive generation steps.
6. **Single-Pass Forward-Mode AD & Upstream Circuit Tracer (`circuit_tracer.py`)**
    * Combines static residual-stream virtual weights ($W_{\text{up}}$ value coupling and $W_{\text{gate}}$ switch coupling) with dynamic causal pole-ablation patching.
    * **Downstream Single-Pass Forward-Mode AD (`trace_forward_ad_all_layers`)**: Computes exact Jacobians ($\partial A_{\text{dst}} / \partial a_{\text{src}}$) across **every downstream neuron and layer simultaneously** in 1 forward pass.
    * **Incoming Upstream Decomposition (`trace_incoming_circuit`)**: Traces $L_{\text{src}} \to (L_{\text{dst}}, \text{neuron\_idx})$ to decompose which upstream routers drive a neuron's main trunk vs. recursive offshoots.
    * **Superposition Readout Auditor (`verify_downstream_readout`)**: Tests Cammarata's *"Nothing Reads That Combination"* finding—measuring whether weak cross-activations across background neurons actually cross the gating/activation threshold of any downstream reader.
7. **Crossword Circuit Ecology Solver & Closed-Loop Agent Verifier (`explainer.py`)**
    * Synthesizes a unified bipolar "verb" hypothesis from a neuron's profile, forward next-token clues, stratified quantile buckets, and upstream/downstream circuit graphs, then automatically runs surgical ablation and pole-flip experiments to empirically verify its own hypothesis.
    * Implements a persistent **`CircuitEcologyMap`** and two-pass **`solve_layer_ecology()`** solver (*"Understanding Compounds"*): explains high-selectivity monosemantic "anchor" neurons first and injects their verified motifs into the LLM prompts of connected upstream/downstream neurons.
8. **Contrastive Pole Discovery, Checkpoint Evolution & Synergistic Group Ablation (`analyzer.py`, `interventions.py`)**
    * **Contrastive Pole Discovery (`discover_contrastive_neurons`)**: Ranks all native neurons across layers by regularized Welch’s Bipolar Separation $t$-statistic between contrastive prompt sets $D^+$ and $D^-$.
    * **Cross-Checkpoint Recursive Growth Tracker (`track_checkpoint_evolution`)**: Tracks how a neuron’s pole selectivity, outgoing weight geometry, and stratified quantile offshoots sprout across training checkpoints or LoRA fine-tuning stages.
    * **Synergistic Multi-Neuron Ablation (`find_synergistic_ablation_set`)**: Finds minimal $k$-neuron coalitions (via greedy forward search or differentiable $L_1$-sparse continuous mask optimization) whose joint ablation overcomes backup circuits.
9. **Production Steering Vector Export & Zero-Overhead Weight Folding (`exporter.py`)**
    * Composes sequential same-neuron interventions into exact `(scale, shift)` affine states and compiles unconditional `ADDITIVE_STEER`, `HARD_CLAMP`, and `ZERO_ABLATE` operations into dense residual-stream steering vectors ($v_{\text{steer}} \in \mathbb{R}^{d_{\text{model}}}$).
    * Exports portable `.json` and `.pt` presets for `ProductionSteeringRuntime` (with full cross-layer `CONDITIONAL_GATE` support and workspace path-traversal protection), or permanently folds interventions into `down_proj` (`Linear` and `Conv1D`) weights and biases for **0% runtime hook overhead**.
10. **Interactive CLI REPL, Multi-Token CoT Generator & Thread-Safe FastAPI Web Workbench (`cli.py`, `server.py`, `session.py`)**
    * Compare Clean vs. Surgically Steered **next-token probability distributions (`eval`)** and **multi-token Chain-of-Thought completions (`generate`)** side-by-side from the terminal REPL or dark-mode browser dashboard.

---

## Project Structure

```text
neural_injector/
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── README.md
├── main_demo.py               # End-to-end demo: auditing, quantile buckets, adaptive pruning, conditional gates & contrastive discovery
├── demo_circuit_tracer.py     # Demo: single-pass forward-AD Jacobian tracing, pole circuits & downstream readout auditing
├── demo_agent_explainer.py    # Demo: agentic explainer, surgical verification, crossword ecology & checkpoint evolution
├── demo_scan_and_export.py    # Demo: layer/pole cleanliness scan, synergistic group ablation, compiled vectors & weight folding
├── neural_injector/
│   ├── __init__.py
│   ├── types.py               # Dataclasses & enums for poles, circuits, ecology, hypotheses, contrastive/evolution reports & presets
│   ├── discovery.py           # Universal architecture-agnostic layer, norm, Conv1D, fused gate_up & 3D MoE resolver
│   ├── analyzer.py            # GPU Welford/Pébay profiler, SwiGLU double-neg detector, quantile/percentile reservoirs & trackers
│   ├── interventions.py       # Vectorized pole-aware GPU kernels, adaptive thresholds, latched conditional gates & AblationMaskOptimizer
│   ├── circuit_tracer.py      # Virtual weights + downstream/incoming circuit tracing + single-pass forward-AD + readout auditor
│   ├── explainer.py           # Automated LLM-agent explainer, Pydantic validator, CircuitEcologyMap & closed-loop surgical verifier
│   ├── exporter.py            # Composed affine steering vector compiler, path-safe preset runtime & zero-overhead weight folder
│   ├── injector.py            # Thread-safe main NeuralInjector orchestrator
│   ├── session.py             # Stateful WorkbenchSession (built-in demo, low-mem HuggingFace loader & multi-token CoT generator)
│   ├── server.py              # Non-blocking async FastAPI REST backend & single-page Web Workbench UI
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

# 1. Load a pre-calibrated demo session or any HuggingFace causal LLM (with automatic low-memory dtype):
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

# 3. Inspect a single neuron's Positive & Negative poles, Next-Token Clues, Quantile Buckets & Vocab Attribution
profile = injector.get_neuron_profile(
    layer_name=src_layer,
    neuron_idx=7,
    unembed_weight=session.unembed_weight,
    idx_to_token=session.id_to_tok,
)
print(f"+Pole Selectivity: {profile.pos_pole_selectivity_score} ({profile.pos_taxonomy_class.value})")
print(f"SwiGLU Double-Negative (+Pole) Rate: {profile.double_negative_positive_rate * 100:.1f}%")
print("Top +Pole Sample Window:", profile.top_positive_samples[0].window_str)

# 4. Closed-Loop Conditional Gate with KV-Cache Latching (`sticky_on_decode=True`)
injector.clear_interventions()
injector.add_conditional_gate(
    target_layer=src_layer,
    target_neurons=[7],
    detector_layer=src_layer,
    detector_neuron=7,
    detector_pole=PoleType.POSITIVE,
    threshold=1.0,
    action_mode=InterventionMode.HARD_CLAMP,
    value=-15.0,
    sequence_scope="any_token",
    sticky_on_decode=True,
)

# Compare multi-token Chain-of-Thought / continuation generation side-by-side
gen_comp = session.generate_comparison("the system negotiated AES256", max_new_tokens=6)
print("Clean Multi-Token Completion  :", gen_comp["clean_completion_text"])
print("Steered Multi-Token Completion:", gen_comp["intervened_completion_text"])

# 5. Trace Downstream & Incoming Upstream Circuits
jac_map = injector.trace_forward_ad_all_layers(
    src_layer=src_layer, src_neuron=7, inputs=input_ids, src_pole=PoleType.POSITIVE
)
downstream_circuit = injector.trace_circuit(
    src_layer=src_layer, dst_layer=dst_layer, src_neuron=7, inputs=input_ids, src_pole=PoleType.POSITIVE
)
incoming_circuit = injector.trace_incoming_circuit(
    src_layer=src_layer, dst_layer=dst_layer, dst_neuron=12, inputs=input_ids, dst_pole=PoleType.POSITIVE
)

# 6. Run the Closed-Loop Agent Explainer & Crossword Circuit Ecology Solver
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

# 7. Export Production Steering Preset or Fold Directly into Model Weights
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
* `profile <layer> <neuron_idx>` — Inspect bipolar contexts, forward next-token windows, SwiGLU double-negative rate, stratified quantile buckets, and direct logit attribution.
* `ablate <layer> <neuron_idx> [both|positive|negative]` — Set a neuron or pole to `0.0`.
* `clamp <layer> <neuron_idx> <value> [pole]` — Clamp a neuron or pole to a constant scalar.
* `prune <layer> <val> [adaptive]` — Prune weak activations across the layer using a scalar threshold or native per-neuron `std_multiplier`.
* `gate <target_layer> <target_neuron> <det_layer> <det_neuron> <delta> [threshold]` — Register a closed-loop conditional gate with KV-cache latching.
* `clear` — Remove all active surgical interventions.
* `eval <prompt>` — Compare Clean vs. Surgically Intervened next-token probabilities and KL divergence.
* `generate [max_new_tokens] <prompt>` — Compare Clean vs. Surgically Intervened multi-token completions (Chain-of-Thought steering).
* `trace <src_layer> <dst_layer> <neuron_idx> [pole] [prompt]` — Trace downstream excitatory, inhibitory, and gating edges with forward-mode AD Jacobians.
* `trace_incoming <src_layer> <dst_layer> <dst_neuron> [pole] [prompt]` — Trace incoming upstream neurons in `src_layer` driving `dst_neuron`.
* `readout <src_layer> <dst_layer> <neuron_indices_csv> [threshold]` — Audit whether weak cross-activations are read downstream (*"Nothing Reads That Combination"*).
* `explain <layer> <neuron_idx> [dst_layer] [src_layer]` — Run the LLM agent explainer and surgical verification loop.
* `ecology <layer> [dst_layer] [top_anchors] [top_contextual]` — Solve layer circuit ecology via two-pass Crossword propagation.
* `contrastive <pos_prompt> | <neg_prompt>` — Discover native neurons separating two contrastive prompts via regularized Welch's $t$-statistic.
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
python -m pytest tests/test_neural_injector.py -v

```