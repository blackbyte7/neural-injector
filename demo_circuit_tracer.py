import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_injector import NeuralInjector, PoleType


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


def run_circuit_tracer_demo():
    torch.manual_seed(7)

    vocab = [
        "<pad>", "the", "system", "negotiated", "executed",
        "AES256", "TLS", "cipher", "handshake",  # Crypto family (IDs 5..8)
        "SELECT", "WHERE", "postgres", "index_scan",  # SQL family (IDs 9..12)
        "packet", "buffer", "latencies"
    ]
    tok_to_id = {w: i for i, w in enumerate(vocab)}

    model = TwoLayerSwiGLUTransformer(vocab_size=len(vocab), d_model=64, d_mlp=128)

    with torch.no_grad():
        # Orthogonal basis vectors in d_model=64
        crypto_vec = F.normalize(torch.randn(64), dim=0)
        sql_vec = F.normalize(torch.randn(64), dim=0)
        bridge_vec = F.normalize(torch.randn(64), dim=0)

        for tid in [5, 6, 7, 8]:
            model.embed.weight[tid] = crypto_vec * 3.0
        for tid in [9, 10, 11, 12]:
            model.embed.weight[tid] = sql_vec * 3.0

        # --- Layer 0, Neuron #7: Bipolar Router (Crypto -> +Pole, SQL -> -Pole) ---
        model.mlp_block_0.gate_proj.weight[7] = (crypto_vec + sql_vec) * 2.5
        model.mlp_block_0.up_proj.weight[7] = (crypto_vec - sql_vec) * 3.0
        # Writes `bridge_vec` into the residual stream proportional to Neuron #7's signed activation
        model.mlp_block_0.down_proj.weight[:, 7] = bridge_vec * 3.5

        # --- Layer 1, Neuron #12: Excited by Layer 0 Neuron #7 (+Pole) -> TLS Handshake Specialist ---
        model.mlp_block_1.gate_proj.weight[12] = bridge_vec * 2.5
        model.mlp_block_1.up_proj.weight[12] = bridge_vec * 3.0
        model.mlp_block_1.down_proj.weight[:, 12] = crypto_vec * 4.0

        # --- Layer 1, Neuron #19: Inhibited by Layer 0 Neuron #7 (+Pole), Excited by (-Pole) -> SQL Specialist ---
        model.mlp_block_1.gate_proj.weight[19] = (sql_vec - bridge_vec) * 2.2
        model.mlp_block_1.up_proj.weight[19] = (-bridge_vec) * 3.2
        model.mlp_block_1.down_proj.weight[:, 19] = sql_vec * 4.0

        # --- Layer 1, Neuron #33: Gating Modulator (strong gate_proj link, weak up_proj link) ---
        model.mlp_block_1.gate_proj.weight[33] = bridge_vec * 3.5
        model.mlp_block_1.up_proj.weight[33] = bridge_vec * 0.4 + crypto_vec * 1.5

    injector = NeuralInjector(model)
    src_layer = "mlp_block_0.down_proj"
    dst_layer = "mlp_block_1.down_proj"

    # Input batch containing Cryptography tokens (triggers Positive Pole of Layer 0 Neuron #7)
    crypto_prompt = torch.tensor(
        [[tok_to_id["the"], tok_to_id["system"], tok_to_id["negotiated"], tok_to_id["AES256"], tok_to_id["TLS"]]],
        dtype=torch.long,
    )

    print("=== 1. TRACING POSITIVE POLE CIRCUIT: (Layer 0, Neuron #7, +Pole) -> Layer 1 ===")
    pos_circuit = injector.trace_circuit(
        src_layer=src_layer,
        dst_layer=dst_layer,
        src_neuron=7,
        inputs=crypto_prompt,
        src_pole=PoleType.POSITIVE,
        compute_jacobian=True,
        top_k=6,
    )

    print("\n[Top Excited Downstream Neurons in Layer 1]:")
    for edge in pos_circuit.top_excitatory(3):
        print(
            f"  (L0, N{edge.src_neuron}, +) --> (L1, N{edge.dst_neuron:02d}) | "
            f"Role: {edge.role.value:<10} | Causal Delta: {edge.mean_causal_delta:+.4f} "
            f"(Clean: {edge.clean_dst_mean:+.3f} -> Ablated: {edge.ablated_dst_mean:+.3f}) | "
            f"Jacobian: {edge.jacobian_sensitivity:+.4f} | "
            f"W_up: {edge.virtual_weight_up:+.3f}, W_gate: {edge.virtual_weight_gate:+.3f}"
        )

    print("\n[Top Inhibited / Opposing Downstream Neurons in Layer 1]:")
    for edge in pos_circuit.top_inhibitory(3):
        print(
            f"  (L0, N{edge.src_neuron}, +) --> (L1, N{edge.dst_neuron:02d}) | "
            f"Role: {edge.role.value:<10} | Causal Delta: {edge.mean_causal_delta:+.4f} "
            f"(Clean: {edge.clean_dst_mean:+.3f} -> Ablated: {edge.ablated_dst_mean:+.3f}) | "
            f"Jacobian: {edge.jacobian_sensitivity:+.4f} | "
            f"W_up: {edge.virtual_weight_up:+.3f}, W_gate: {edge.virtual_weight_gate:+.3f}"
        )

    # Now trace the NEGATIVE POLE on an SQL prompt
    sql_prompt = torch.tensor(
        [[tok_to_id["the"], tok_to_id["system"], tok_to_id["executed"], tok_to_id["SELECT"], tok_to_id["postgres"]]],
        dtype=torch.long,
    )

    print("\n=== 2. TRACING NEGATIVE POLE CIRCUIT: (Layer 0, Neuron #7, -Pole) -> Layer 1 ===")
    neg_circuit = injector.trace_circuit(
        src_layer=src_layer,
        dst_layer=dst_layer,
        src_neuron=7,
        inputs=sql_prompt,
        src_pole=PoleType.NEGATIVE,
        compute_jacobian=True,
        top_k=6,
    )

    for edge in neg_circuit.top_excitatory(3):
        print(
            f"  (L0, N{edge.src_neuron}, -) --> (L1, N{edge.dst_neuron:02d}) | "
            f"Role: {edge.role.value:<10} | Causal Delta: {edge.mean_causal_delta:+.4f} "
            f"(Clean: {edge.clean_dst_mean:+.3f} -> Ablated: {edge.ablated_dst_mean:+.3f}) | "
            f"W_up: {edge.virtual_weight_up:+.3f}, W_gate: {edge.virtual_weight_gate:+.3f}"
        )

    print("\n=== 3. EXPORTED PAYLOAD FOR UPCOMING LLM-AGENT EXPLAINER ===")
    print(json.dumps(pos_circuit.to_agent_payload(), indent=2))


if __name__ == "__main__":
    run_circuit_tracer_demo()
