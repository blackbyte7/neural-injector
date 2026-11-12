from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

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
    cached evaluation prompts, and helper methods for JSON-serializable inspection,
    intervention, circuit tracing, side-by-side next-token logit comparison, and
    multi-token Chain-of-Thought autoregressive generation (`generate_comparison`).
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
        self.last_audit_prompts: List[str] = []

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

            # Calibrate neutral background tokens so that after norm_0 they have zero
            # sql_vec/bridge_vec leakage and a weak positive projection (+0.34) along crypto_vec.
            # This produces a_7 ≈ +0.68 on neutral tokens: active (> 1e-5) in the unsaturated
            # regime of norm_1 for Forward-AD Jacobian tracing, yet strictly below the
            # conditional gate threshold (1.0) on SQL prompts.
            ones_vec = torch.ones(64)
            basis = torch.stack([crypto_vec, sql_vec, bridge_vec, ones_vec], dim=1)
            q_basis, _ = torch.linalg.qr(basis)
            s_zero = sql_vec - sql_vec.mean()
            c_zero = crypto_vec - crypto_vec.mean()
            c_zero = c_zero - (torch.dot(c_zero, s_zero) / torch.dot(s_zero, s_zero)) * s_zero
            unit_c = c_zero / torch.dot(c_zero, crypto_vec)

            for tid in range(len(vocab)):
                if tid not in (5, 6, 7, 8, 9, 10, 11, 12):
                    v = model.embed.weight[tid]
                    v_orth = v - q_basis @ (q_basis.T @ v)
                    v_orth = F.normalize(v_orth, dim=0) * 7.8
                    model.embed.weight[tid] = v_orth + 0.34 * unit_c

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
            model.mlp_block_1.up_proj.weight[19] = (-bridge_vec) * 3.8
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
        )
        return session

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        device: str = "cpu",
        torch_dtype: Union[str, torch.dtype] = "auto",
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> WorkbenchSession:
        """
        Loads any HuggingFace AutoModelForCausalLM and AutoTokenizer into the Workbench
        using `torch_dtype="auto"` and `low_cpu_mem_usage=True` to avoid FP32 RAM spikes.
        Supports `device_map="auto"` (Accelerate multi-GPU/CPU offload) and 8-bit/4-bit
        quantized loading without calling `.to(device)` on dispatched models.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        load_kwargs: Dict[str, Any] = {"dtype": torch_dtype}
        if low_cpu_mem_usage:
            load_kwargs["low_cpu_mem_usage"] = True
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        use_auto_dispatch = bool(device_map is not None or load_in_8bit or load_in_4bit)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            if not use_auto_dispatch:
                model = model.to(device)
        except Exception:
            if use_auto_dispatch:
                raise
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = model.to(device)
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

    def _get_input_device(self) -> torch.device:
        """Resolves the input device safely for both single-device and `device_map` models."""
        model_dev = getattr(self.model, "device", None)
        if isinstance(model_dev, torch.device) and model_dev.type != "meta":
            return model_dev
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def _supports_attention_mask(self) -> bool:
        """Checks whether `self.model.forward` accepts `attention_mask` or `**kwargs`."""
        try:
            sig = inspect.signature(self.model.forward)
            if "attention_mask" in sig.parameters:
                return True
            return any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            return False

    def get_default_eval_prompt(self) -> str:
        """
        Returns the most relevant default evaluation prompt for the active session:
        uses the first prompt from the most recent `run_audit` call if available,
        or a natural sentence valid for the active vocabulary.
        """
        if self.last_audit_prompts:
            return self.last_audit_prompts[0]
        if "AES256" in self.tok_to_id:
            return "the system negotiated AES256 TLS cipher"
        return "The quick brown fox jumps over the lazy dog."

    def get_default_eval_inputs(self, custom_prompt: Optional[str] = None) -> torch.Tensor:
        """Encodes `custom_prompt` or `self.get_default_eval_prompt()` into `input_ids`."""
        prompt = (
            custom_prompt.strip()
            if custom_prompt and custom_prompt.strip()
            else self.get_default_eval_prompt()
        )
        input_ids, _ = self.encode_prompt(prompt)
        return input_ids

    def encode_prompt_with_mask(self, text: str) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Tokenizes a text string into `(input_ids, token_strings, attention_mask)`,
        automatically marking `<pad>` tokens as False/0 in `attention_mask`.
        """
        device = self._get_input_device()
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

    def encode_text(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Encoder used for every text measured by the explanation tests (same as the audit encoding)."""
        ids, toks, _ = self.encode_prompt_with_mask(text)
        return ids, toks

    def decode_ids(self, token_ids: List[int]) -> str:
        return self._decode_token_ids(token_ids)[1]

    def explain_neuron(self, layer_name: str, neuron_idx: int, **kwargs: Any):
        """Runs the two-agent explanation protocol with this session's encoder, decoder and vocabulary."""
        kwargs.setdefault("encode_fn", self.encode_text)
        kwargs.setdefault("decode_fn", self.decode_ids)
        kwargs.setdefault("unembed_weight", self.unembed_weight)
        kwargs.setdefault("idx_to_token", self.id_to_tok)
        return self.injector.explain_neuron(layer_name, neuron_idx, **kwargs)

    def encode_prompt(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Tokenizes a text string into an input_ids tensor [1, seq_len] and list of token strings."""
        input_ids, tokens, _ = self.encode_prompt_with_mask(text)
        return input_ids, tokens

    def _decode_token_ids(self, token_ids: List[int]) -> Tuple[List[str], str]:
        """Decodes a list of integer token IDs into token strings and a joined completion string."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            tok_strs = [self.tokenizer.decode([tid]) for tid in token_ids]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return tok_strs, text
        tok_strs = [self.id_to_tok.get(tid, f"token_{tid}") for tid in token_ids]
        return tok_strs, " ".join(tok_strs)

    def _supports_kv_cache(self) -> bool:
        """Checks whether `self.model.forward` accepts `past_key_values` and `use_cache`."""
        try:
            sig = inspect.signature(self.model.forward)
            return "past_key_values" in sig.parameters and "use_cache" in sig.parameters
        except (ValueError, TypeError):
            return False

    @torch.no_grad()
    def _autoregressive_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 12,
        temperature: float = 0.0,
        use_kv_cache: bool = True,
        repetition_penalty: float = 1.0,
    ) -> Tuple[List[int], List[str], str]:
        """
        Runs multi-token autoregressive decoding for `max_new_tokens` steps.
        Supports both HuggingFace KV-cached decoding (`use_cache=True`, passing `[B, 1]`
        tokens on steps >= 1) and custom PyTorch models.
        """
        self.injector.reset_latched_gates()
        curr_ids = input_ids.clone()
        generated_ids: List[int] = []
        eos_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer is not None else None
        can_cache = use_kv_cache and self._supports_kv_cache()
        past_key_values = None

        for step in range(max(1, max_new_tokens)):
            if can_cache:
                step_input = curr_ids if step == 0 else curr_ids[:, -1:]
                out = self.model(
                    step_input, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = getattr(out, "past_key_values", None)
                logits = (out.logits if hasattr(out, "logits") else out[0]).float()
            else:
                out = self.model(curr_ids)
                logits = (out.logits if hasattr(out, "logits") else out).float()

            next_logits = logits[0, -1, :].clone()

            # Optional repetition penalty (1.0 = plain greedy / sampling, the default)
            if repetition_penalty != 1.0 and generated_ids:
                for prev_id in set(generated_ids):
                    if next_logits[prev_id] > 0:
                        next_logits[prev_id] /= repetition_penalty
                    else:
                        next_logits[prev_id] *= repetition_penalty

            if temperature <= 1e-6:
                next_id = int(torch.argmax(next_logits, dim=-1).item())
            else:
                probs = F.softmax(next_logits / float(temperature), dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids.append(next_id)
            next_t = torch.tensor([[next_id]], dtype=torch.long, device=curr_ids.device)
            curr_ids = torch.cat([curr_ids, next_t], dim=1)

            if eos_id is not None and next_id == eos_id:
                break

        tok_strs, completion_text = self._decode_token_ids(generated_ids)
        return generated_ids, tok_strs, completion_text

    def generate_comparison(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        use_kv_cache: bool = True,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Runs a side-by-side multi-token autoregressive generation (Chain-of-Thought /
        continuation comparison) comparing:
          1. Clean (unmodified) multi-token completion
          2. Surgically intervened multi-token completion (with active hooks & latched gates)
        """
        input_ids, prompt_tokens = self.encode_prompt(prompt)
        self.model.eval()

        with torch.no_grad():
            with self.injector.without_interventions():
                clean_ids, clean_toks, clean_text = self._autoregressive_generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_kv_cache=use_kv_cache,
                    repetition_penalty=repetition_penalty,
                )

            with self.injector.active_injection():
                mod_ids, mod_toks, mod_text = self._autoregressive_generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_kv_cache=use_kv_cache,
                    repetition_penalty=repetition_penalty,
                )

        min_len = min(len(clean_ids), len(mod_ids))
        diff_count = sum(1 for i in range(min_len) if clean_ids[i] != mod_ids[i]) + abs(
            len(clean_ids) - len(mod_ids)
        )

        return {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "clean_generated_tokens": clean_toks,
            "clean_completion_text": clean_text,
            "clean_full_text": f"{prompt} {clean_text}".strip(),
            "intervened_generated_tokens": mod_toks,
            "intervened_completion_text": mod_text,
            "intervened_full_text": f"{prompt} {mod_text}".strip(),
            "tokens_changed_count": diff_count,
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

    def run_audit(
        self,
        layer_names: List[str],
        prompts: List[str],
        top_k: int = 16,
        batch_size: int = 1,
        **tracker_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Streams text prompts through the model and records exact streaming statistics for each layer.
        Batched execution right-pads and passes the attention mask, so padded positions are excluded.
        """
        self.last_audit_prompts = list(prompts)
        encoded_items = []
        for idx, text in enumerate(prompts):
            input_ids, tok_strs, attn_mask = self.encode_prompt_with_mask(text)
            encoded_items.append((idx, text, input_ids[0], tok_strs, attn_mask[0]))

        device = self._get_input_device()
        pad_id = (
            getattr(self.tokenizer, "pad_token_id", None)
            if self.tokenizer is not None
            else None
        )
        if pad_id is None:
            pad_id = (
                getattr(self.tokenizer, "eos_token_id", None)
                if self.tokenizer is not None
                else None
            )
        if pad_id is None:
            pad_id = self.tok_to_id.get("<pad>", 0)
        pad_str = self.id_to_tok.get(int(pad_id), "<pad>")

        batches: List[Dict[str, Any]] = []
        eff_bs = max(1, int(batch_size))
        for start in range(0, len(encoded_items), eff_bs):
            chunk = encoded_items[start: start + eff_bs]
            if len(chunk) == 1:
                idx, text, ids_1d, tok_strs, mask_1d = chunk[0]
                batches.append({
                    "input_ids": ids_1d.unsqueeze(0),
                    "attention_mask": mask_1d.unsqueeze(0),
                    "sample_ids": [idx],
                    "tokens_per_sample": [tok_strs],
                    "contexts_per_sample": [text],
                })
            else:
                max_len = max(item[2].numel() for item in chunk)
                batch_ids = torch.full(
                    (len(chunk), max_len), int(pad_id), dtype=torch.long, device=device
                )
                batch_mask = torch.zeros(
                    (len(chunk), max_len), dtype=torch.bool, device=device
                )
                sample_ids: List[int] = []
                tokens_per_sample: List[List[str]] = []
                contexts_per_sample: List[str] = []
                for b_i, (idx, text, ids_1d, tok_strs, mask_1d) in enumerate(chunk):
                    slen = ids_1d.numel()
                    batch_ids[b_i, :slen] = ids_1d
                    batch_mask[b_i, :slen] = mask_1d
                    sample_ids.append(idx)
                    padded_toks = list(tok_strs) + [pad_str] * (max_len - len(tok_strs))
                    tokens_per_sample.append(padded_toks)
                    contexts_per_sample.append(text)
                batches.append({
                    "input_ids": batch_ids,
                    "attention_mask": batch_mask,
                    "sample_ids": sample_ids,
                    "tokens_per_sample": tokens_per_sample,
                    "contexts_per_sample": contexts_per_sample,
                })

        supports_mask = self._supports_attention_mask()

        def _forward_batch(m: nn.Module, b: Dict[str, Any]) -> Any:
            if supports_mask and b.get("attention_mask") is not None and b["input_ids"].shape[0] > 1:
                return m(b["input_ids"], attention_mask=b["attention_mask"].to(dtype=torch.long))
            return m(b["input_ids"])

        self.injector.audit_layers(
            layer_names=layer_names,
            dataloader_or_batches=batches,
            forward_fn=_forward_batch,
            top_k=top_k,
            **tracker_kwargs,
        )
        return {
            "status": "audited",
            "layers": layer_names,
            "num_prompts": len(prompts),
            "batch_size": eff_bs,
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
