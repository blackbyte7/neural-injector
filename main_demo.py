from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_injector import (
    NeuralInjector,
    PoleType,
)


# =============================================================================
# 1. Define a Qwen/Llama-Style SwiGLU Transformer for Realistic Demonstration
# =============================================================================
class SwiGLUMLP(nn.Module):
    """Modern SwiGLU MLP block identical in structure to Qwen3 and Llama-3."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The product `F.silu(gate_proj(x)) * up_proj(x)` forms the bipolar SwiGLU neuron basis
        # right before `down_proj` mixes it back into the residual stream.
        neuron_activations = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(neuron_activations)


class MiniSwiGLUTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, d_mlp: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mlp_block_0 = SwiGLUMLP(d_model, d_mlp)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x + self.mlp_block_0(x)
        x = self.norm(x)
        return self.lm_head(x)


# =============================================================================
# 2. Run the Full Neural Injector Pipeline
# =============================================================================
def run_pipeline():
    torch.manual_seed(42)

    # Vocabulary representing two opposing semantic poles (Security/Encryption vs. SQL/Database)
    # plus neutral background tokens
    vocab = [
        "<pad>", "the", "system", "processed", "request", "with",
        "AES256", "TLS", "cipher", "decrypt",  # Positive Pole Motif: Cryptography (IDs 6..9)
        "SELECT", "WHERE", "JOIN", "postgres",  # Negative Pole Motif: SQL Queries  (IDs 10..13)
        "fast", "slow", "server", "client", "data", "log"
    ]
    tok_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_tok = {i: w for i, w in enumerate(vocab)}

    model = MiniSwiGLUTransformer(vocab_size=len(vocab), d_model=64, d_mlp=128)

    # Align Neuron #7 in `mlp_block_0.down_proj` to act as a clean Bipolar Discriminator ("Verb" Neuron):
    # - Positive Pole (+): Strongly activates on Cryptography tokens (IDs 6..9) and promotes "cipher" / "decrypt"
    # - Negative Pole (-): Strongly activates negatively on SQL tokens (IDs 10..13) and promotes "SELECT" / "postgres"
    with torch.no_grad():
        crypto_dir = F.normalize(torch.randn(64), dim=0)
        sql_dir = F.normalize(torch.randn(64), dim=0)

        for tid in [6, 7, 8, 9]:
            model.embed.weight[tid] = crypto_dir * 3.0
            model.lm_head.weight[tid] = crypto_dir * 4.0
        for tid in [10, 11, 12, 13]:
            model.embed.weight[tid] = sql_dir * 3.0
            model.lm_head.weight[tid] = sql_dir * 4.0

        # Configure Neuron #7 gate & up projections so Crypto -> +Pole and SQL -> -Pole
        model.mlp_block_0.gate_proj.weight[7] = (crypto_dir + sql_dir) * 2.5
        model.mlp_block_0.up_proj.weight[7] = (crypto_dir - sql_dir) * 3.5
        model.mlp_block_0.down_proj.weight[:, 7] = (crypto_dir - sql_dir) * 4.0

    injector = NeuralInjector(model)

    print("=== 1. AUTOMATIC NEURON LAYER DISCOVERY ===")
    layers = injector.list_targetable_layers()
    for name, desc in layers.items():
        print(
            f"Found Layer: {name} | Type: {desc.module_type} | Neurons: {desc.num_neurons} | Hook: {desc.hook_target}")

    target_layer = "mlp_block_0.down_proj"

    # Build a synthetic corpus of sentences
    corpus = [
        ["the", "system", "processed", "AES256", "TLS", "cipher"],
        ["client", "request", "with", "SELECT", "WHERE", "JOIN"],
        ["fast", "server", "decrypt", "AES256", "data", "log"],
        ["slow", "client", "postgres", "SELECT", "JOIN", "data"],
        ["the", "server", "processed", "fast", "client", "request"],
    ]

    batches: List[Dict[str, Any]] = []
    for idx, sent in enumerate(corpus):
        input_ids = torch.tensor([[tok_to_id[w] for w in sent]], dtype=torch.long)
        batches.append({
            "input_ids": input_ids,
            "sample_ids": [idx],
            "tokens_per_sample": [sent],
            "contexts_per_sample": [" ".join(sent)],
        })

    print("\n=== 2. BIPOLAR NEURON AUDITING (POSITIVE & NEGATIVE POLES) ===")
    injector.audit_layers(
        layer_names=[target_layer],
        dataloader_or_batches=batches,
        forward_fn=lambda m, b: m(b["input_ids"]),
        top_k=4,
        tracked_neurons=[7],
    )

    profile = injector.get_neuron_profile(
        layer_name=target_layer,
        neuron_idx=7,
        unembed_weight=model.lm_head.weight,
        idx_to_token=id_to_tok,
    )

    print(f"Neuron #{profile.neuron_idx} in '{profile.layer_name}':")
    print(
        f"  Mean: {profile.mean:.4f} | Std: {profile.std:.4f} | Selectivity Score: {profile.pole_selectivity_score:.2f} | Clean: {profile.is_clean_candidate}")
    print(f"  Positive Pole Firing Rate: {profile.positive_firing_rate * 100:.1f}% | Max: {profile.max_val:.3f}")
    for s in profile.top_positive_samples[:3]:
        print(f"    [+] Val: {s.activation_value:+.3f} | Token: '{s.token_str}' | Context: \"{s.context_str}\"")

    print(f"  Negative Pole Firing Rate: {profile.negative_firing_rate * 100:.1f}% | Min: {profile.min_val:.3f}")
    for s in profile.top_negative_samples[:3]:
        print(f"    [-] Val: {s.activation_value:+.3f} | Token: '{s.token_str}' | Context: \"{s.context_str}\"")

    print(f"  Direct Logit Attribution (+ Pole Promotes): {profile.top_promoted_tokens[:4]}")
    print(f"  Direct Logit Attribution (- Pole Promotes): {profile.top_suppressed_tokens[:4]}")

    # =========================================================================
    # 3. Surgical Interventions: Zero-Ablation, Pole Clamping & Weak Pruning
    # =========================================================================
    test_input = batches[0]["input_ids"]  # ["the", "system", "processed", "AES256", "TLS", "cipher"]
    crypto_token_ids = [tok_to_id["AES256"], tok_to_id["TLS"], tok_to_id["cipher"], tok_to_id["decrypt"]]
    sql_token_ids = [tok_to_id["SELECT"], tok_to_id["WHERE"], tok_to_id["JOIN"], tok_to_id["postgres"]]

    def get_concept_probs(logits: torch.Tensor) -> Tuple[float, float]:
        probs = F.softmax(logits[0, 3, :], dim=-1)  # Inspect predictions at token position 3 ("AES256")
        crypto_prob = probs[crypto_token_ids].sum().item()
        sql_prob = probs[sql_token_ids].sum().item()
        return crypto_prob, sql_prob

    with torch.no_grad():
        base_logits = model(test_input)
        base_crypto, base_sql = get_concept_probs(base_logits)

        # (A) Zero-Ablate Neuron #7 ("set neuron to 0")
        injector.clear_interventions()
        injector.zero_ablate(target_layer, neuron_indices=7, pole=PoleType.BOTH)
        with injector.active_injection():
            zero_logits = model(test_input)
            zero_crypto, zero_sql = get_concept_probs(zero_logits)

        # (B) Weak-Activation Pruning (Zero out weak background activations |a| < 1.5)
        injector.clear_interventions()
        injector.prune_weak_activations(target_layer, neuron_indices=list(range(128)), threshold=1.5)
        with injector.active_injection():
            pruned_logits = model(test_input)
            pruned_crypto, pruned_sql = get_concept_probs(pruned_logits)

        # (C) Negative Pole Injection (Force Neuron #7 to -15.0 to steer from Cryptography -> SQL)
        injector.clear_interventions()
        injector.clamp_neurons(target_layer, neuron_indices=7, value=-15.0)
        with injector.active_injection():
            steered_logits = model(test_input)
            steered_crypto, steered_sql = get_concept_probs(steered_logits)

    kl_impact = injector.measure_ablation_kl_divergence(target_layer, neuron_idx=7, inputs=test_input)

    print("\n=== 3. SURGICAL INJECTION RESULTS AT TOKEN 'AES256' ===")
    print(f"  Baseline (Unmodified)          -> Crypto Prob: {base_crypto:.4f} | SQL Prob: {base_sql:.4f}")
    print(
        f"  Zero-Ablated (Neuron #7 = 0.0) -> Crypto Prob: {zero_crypto:.4f} | SQL Prob: {zero_sql:.4f} (Causal KL: {kl_impact:.4f})")
    print(
        f"  Weak Activations Pruned (<1.5) -> Crypto Prob: {pruned_crypto:.4f} | SQL Prob: {pruned_sql:.4f} (Signal Preserved!)")
    print(
        f"  Clamped to Negative Pole (-15) -> Crypto Prob: {steered_crypto:.4f} | SQL Prob: {steered_sql:.4f} (Concept Flipped!)")


if __name__ == "__main__":
    run_pipeline()
