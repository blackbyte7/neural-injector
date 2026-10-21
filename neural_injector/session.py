from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .injector import NeuralInjector


class _DemoSwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DemoTwoLayerTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, d_mlp: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm_0 = nn.LayerNorm(d_model)
        self.mlp_block_0 = _DemoSwiGLUMLP(d_model, d_mlp)
        self.norm_1 = nn.LayerNorm(d_model)
        self.mlp_block_1 = _DemoSwiGLUMLP(d_model, d_mlp)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x + self.mlp_block_0(self.norm_0(x))
        x = x + self.mlp_block_1(self.norm_1(x))
        return self.lm_head(self.final_norm(x))


class WorkbenchSession:
    """
    Stateful manager holding the active model, tokenizer mappings, NeuralInjector instance,
    and helper methods for JSON-serializable inspection, intervention, circuit tracing,
    and side-by-side logit comparison.
    """

    def __init__(
        self,
        model: nn.Module,
        tok_to_id: Dict[str, int],
        id_to_tok: Dict[int, str],
        unembed_weight: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
        model_name: str = "custom-model",
    ):
        self.model = model
        self.tok_to_id = tok_to_id
        self.id_to_tok = id_to_tok
        self.unembed_weight = unembed_weight
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.injector = NeuralInjector(model, tokenizer=tokenizer)

    @classmethod
    def create_demo_session(cls) -> WorkbenchSession:
        """Creates a pre-calibrated 2-layer SwiGLU session with pre-audited bipolar circuits."""
        torch.manual_seed(7)
        vocab = [
            "<pad>", "the", "system", "negotiated", "executed",
            "AES256", "TLS", "cipher", "handshake",  # Positive Pole: Cryptography (5..8)
            "SELECT", "WHERE", "postgres", "index_scan",  # Negative Pole: SQL Queries  (9..12)
            "packet", "buffer", "latencies", "request", "server"
        ]
        tok_to_id = {w: i for i, w in enumerate(vocab)}
        id_to_tok = {i: w for i, w in enumerate(vocab)}

        model = _DemoTwoLayerTransformer(vocab_size=len(vocab), d_model=64, d_mlp=128)
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

        session = cls(
            model=model,
            tok_to_id=tok_to_id,
            id_to_tok=id_to_tok,
            unembed_weight=model.lm_head.weight,
            model_name="Demo-2Layer-SwiGLU-64d",
        )
        # Pre-run default corpus audit so the UI & CLI have immediate profiles ready
        default_prompts = [
            "the system negotiated AES256 TLS cipher",
            "the system executed SELECT WHERE postgres",
            "packet buffer TLS handshake AES256 cipher",
            "packet buffer postgres index_scan SELECT WHERE",
            "the server executed packet buffer latencies",
        ]
        session.run_audit(
            layer_names=list(session.injector.list_targetable_layers().keys()),
            prompts=default_prompts,
            tracked_neurons=list(range(128)),
        )
        return session

    @classmethod
    def from_huggingface(cls, model_id: str, device: str = "cpu") -> WorkbenchSession:
        """Loads any HuggingFace AutoModelForCausalLM and AutoTokenizer into the Workbench."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()

        vocab = tokenizer.get_vocab()
        id_to_tok = {v: k for k, v in vocab.items()}
        unembed = (
            model.get_output_embeddings().weight
            if model.get_output_embeddings() is not None
            else None
        )

        return cls(
            model=model,
            tok_to_id=vocab,
            id_to_tok=id_to_tok,
            unembed_weight=unembed,
            tokenizer=tokenizer,
            model_name=model_id,
        )

    def encode_prompt_with_mask(self, text: str) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Tokenizes a text string into `(input_ids, token_strings, attention_mask)`,
        automatically marking `<pad>` tokens as False/0 in `attention_mask`.
        """
        device = next(self.model.parameters()).device
        if self.tokenizer is not None:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            if "attention_mask" in enc:
                attn_mask = enc["attention_mask"].to(device=device, dtype=torch.bool)
            else:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                attn_mask = (
                    (input_ids != pad_id)
                    if pad_id is not None
                    else torch.ones_like(input_ids, dtype=torch.bool)
                )
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
            return input_ids, tokens, attn_mask

        words = [w.strip() for w in text.split() if w.strip()]
        pad_id = self.tok_to_id.get("<pad>", 0)
        ids = [self.tok_to_id.get(w, pad_id) for w in words] or [pad_id]
        tokens = [self.id_to_tok.get(i, "<pad>") for i in ids]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn_mask = input_ids != pad_id
        if not attn_mask.any():
            attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids, tokens, attn_mask

    def encode_prompt(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Tokenizes a text string into an input_ids tensor [1, seq_len] and list of token strings."""
        input_ids, tokens, _ = self.encode_prompt_with_mask(text)
        return input_ids, tokens

    def run_audit(
        self,
        layer_names: List[str],
        prompts: List[str],
        top_k: int = 8,
        tracked_neurons: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Streams a list of text prompts through the model to populate bipolar neuron stats."""
        batches = []
        for idx, text in enumerate(prompts):
            input_ids, tok_strs, attn_mask = self.encode_prompt_with_mask(text)
            batches.append({
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "sample_ids": [idx],
                "tokens_per_sample": [tok_strs],
                "contexts_per_sample": [text],
            })

        self.injector.audit_layers(
            layer_names=layer_names,
            dataloader_or_batches=batches,
            forward_fn=lambda m, b: m(b["input_ids"]),
            top_k=top_k,
            tracked_neurons=tracked_neurons,
        )
        return {
            "status": "audited",
            "layers": layer_names,
            "num_prompts": len(prompts),
        }

    def evaluate_prompt_comparison(self, prompt: str, top_k_logits: int = 6) -> Dict[str, Any]:
        """
        Runs a side-by-side forward pass on `prompt` comparing:
          1. Clean (unmodified) model predictions at the final token position
          2. Surgically intervened model predictions (with currently registered interventions)
          3. Overall output KL-divergence across the sequence
        """
        input_ids, tokens = self.encode_prompt(prompt)
        self.model.eval()

        with torch.no_grad():
            # Clean pass (hooks detached)
            with self.injector.without_interventions():
                clean_out = self.model(input_ids)
                clean_logits = (
                    clean_out.logits if hasattr(clean_out, "logits") else clean_out
                ).float()
                clean_probs = F.softmax(clean_logits[0, -1, :], dim=-1)

            # Intervened pass (hooks attached)
            with self.injector.active_injection():
                mod_out = self.model(input_ids)
                mod_logits = (mod_out.logits if hasattr(mod_out, "logits") else mod_out).float()
                mod_probs = F.softmax(mod_logits[0, -1, :], dim=-1)

            # Compute sequence-wide KL divergence
            p_clean = F.softmax(clean_logits, dim=-1)
            log_p_clean = F.log_softmax(clean_logits, dim=-1)
            log_p_mod = F.log_softmax(mod_logits, dim=-1)
            kl_div = float((p_clean * (log_p_clean - log_p_mod)).sum(dim=-1).mean().item())

        k = min(top_k_logits, clean_probs.numel())
        c_vals, c_ids = torch.topk(clean_probs, k=k)
        m_vals, m_ids = torch.topk(mod_probs, k=k)

        return {
            "prompt": prompt,
            "tokens": tokens,
            "kl_divergence": round(kl_div, 5),
            "clean_top_predictions": [
                {"token": self.id_to_tok.get(i, str(i)), "prob": round(float(p), 4)}
                for p, i in zip(c_vals.tolist(), c_ids.tolist())
            ],
            "intervened_top_predictions": [
                {"token": self.id_to_tok.get(i, str(i)), "prob": round(float(p), 4)}
                for p, i in zip(m_vals.tolist(), m_ids.tolist())
            ],
            "active_interventions": [
                {
                    "layer_name": spec.layer_name,
                    "neuron_indices": spec.neuron_indices,
                    "mode": spec.mode.value,
                    "pole": spec.pole.value,
                    "value": spec.value,
                }
                for spec in self.injector.get_active_interventions()
            ],
        }
