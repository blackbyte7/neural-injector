from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_injector import NeuralInjector


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TwoLayerSwiGLUTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, d_mlp: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm_0 = nn.LayerNorm(d_model)
        self.mlp_block_0 = SwiGLUMLP(d_model, d_mlp)
        self.norm_1 = nn.LayerNorm(d_model)
        self.mlp_block_1 = SwiGLUMLP(d_model, d_mlp)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x + self.mlp_block_0(self.norm_0(x))
        x = x + self.mlp_block_1(self.norm_1(x))
        return self.lm_head(self.final_norm(x))


def run_agent_explainer_demo():
    torch.manual_seed(7)

    vocab = [
        "<pad>", "the", "system", "negotiated", "executed",
        "AES256", "TLS", "cipher", "handshake",  # Positive Pole: Cryptography (IDs 5..8)
        "SELECT", "WHERE", "postgres", "index_scan",  # Negative Pole: SQL Queries  (IDs 9..12)
        "packet", "buffer", "latencies"
    ]
    tok_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_tok = {i: w for i, w in enumerate(vocab)}

    model = TwoLayerSwiGLUTransformer(vocab_size=len(vocab), d_model=64, d_mlp=128)

    with torch.no_grad():
        crypto_vec = F.normalize(torch.randn(64), dim=0)
        sql_vec = F.normalize(torch.randn(64), dim=0)
        bridge_vec = F.normalize(torch.randn(64), dim=0)

        for tid in [5, 6, 7, 8]:
            model.embed.weight[tid] = crypto_vec * 3.0
            model.lm_head.weight[tid] = (crypto_vec + bridge_vec) * 3.5
        for tid in [9, 10, 11, 12]:
            model.embed.weight[tid] = sql_vec * 3.0
            model.lm_head.weight[tid] = (sql_vec - bridge_vec) * 3.5

        # Layer 0, Neuron #7: Bipolar Verb Router (Crypto -> +Pole, SQL -> -Pole)
        model.mlp_block_0.gate_proj.weight[7] = (crypto_vec + sql_vec) * 2.5
        model.mlp_block_0.up_proj.weight[7] = (crypto_vec - sql_vec) * 3.2
        model.mlp_block_0.down_proj.weight[:, 7] = bridge_vec * 3.5

        # Layer 1, Neuron #12: Excited by L0 N7 (+Pole)
        model.mlp_block_1.gate_proj.weight[12] = bridge_vec * 2.5
        model.mlp_block_1.up_proj.weight[12] = bridge_vec * 3.0
        model.mlp_block_1.down_proj.weight[:, 12] = crypto_vec * 4.0

        # Layer 1, Neuron #19: Inhibited by L0 N7 (+Pole), Excited by (-Pole)
        model.mlp_block_1.gate_proj.weight[19] = (sql_vec - bridge_vec) * 2.2
        model.mlp_block_1.up_proj.weight[19] = (-bridge_vec) * 3.2
        model.mlp_block_1.down_proj.weight[:, 19] = sql_vec * 4.0

    # Initialize NeuralInjector (uses HeuristicSynthesisClient by default, or pass OpenAICompatibleClient)
    injector = NeuralInjector(model)

    src_layer = "mlp_block_0.down_proj"
    dst_layer = "mlp_block_1.down_proj"

    corpus = [
        ["the", "system", "negotiated", "AES256", "TLS", "cipher"],
        ["the", "system", "executed", "SELECT", "WHERE", "postgres"],
        ["packet", "buffer", "TLS", "handshake", "AES256", "cipher"],
        ["packet", "buffer", "postgres", "index_scan", "SELECT", "WHERE"],
        ["the", "system", "packet", "buffer", "latencies", "executed"],
    ]

    batches: List[Dict[str, Any]] = []
    for idx, sent in enumerate(corpus):
        batches.append({
            "input_ids": torch.tensor([[tok_to_id[w] for w in sent]], dtype=torch.long),
            "sample_ids": [idx],
            "tokens_per_sample": [sent],
            "contexts_per_sample": [" ".join(sent)],
        })

    # Step 1: Stream corpus to audit Layer 0
    injector.audit_layers(
        layer_names=[src_layer],
        dataloader_or_batches=batches,
        forward_fn=lambda m, b: m(b["input_ids"]),
        top_k=5,
        tracked_neurons=[7],
    )

    # Step 2: Run the Agentic Explainer + Closed-Loop Surgical Verifier
    hypothesis = injector.explain_neuron(
        layer_name=src_layer,
        neuron_idx=7,
        tok_to_id=tok_to_id,
        idx_to_token=id_to_tok,
        unembed_weight=model.lm_head.weight,
        dst_layer=dst_layer,
        circuit_inputs=batches[0]["input_ids"],
    )

    print("=== AGENTIC NEURON & CIRCUIT EXPLANATION ===")
    print(f"Target Neuron          : {hypothesis.layer_name} [Neuron #{hypothesis.neuron_idx}]")
    print(
        f"Taxonomy Classification: {hypothesis.taxonomy_class.value} (Agent Confidence: {hypothesis.confidence_score:.2f})")
    print(f"Positive Pole (+) Motif: {hypothesis.positive_pole_motif}")
    print(f"Negative Pole (-) Motif: {hypothesis.negative_pole_motif}")
    print(f"Unified 'Verb' Function: {hypothesis.unified_verb_hypothesis}")
    print("Recursive Offshoots    :")
    for branch in hypothesis.recursive_offshoots:
        print(f"  - {branch}")

    if hypothesis.verification_report:
        rep = hypothesis.verification_report
        print("\n=== CLOSED-LOOP SURGICAL VERIFICATION REPORT ===")
        print(f"Empirical Score        : {rep.empirical_verification_score:.3f} / 1.000")
        print(f"Summary                : {rep.summary_notes}")


if __name__ == "__main__":
    run_agent_explainer_demo()
