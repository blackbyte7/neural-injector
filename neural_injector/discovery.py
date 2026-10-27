from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerDescriptor:
    """Metadata describing a targetable neuron layer inside a PyTorch model."""
    name: str
    module: nn.Module
    module_type: str
    num_neurons: int
    hook_target: str  # "output" (standard Linear/Act) or "input" (e.g., down_proj in SwiGLU LLMs)


class _SlicedLinearView(nn.Module):
    """Lightweight read-only module view over half of a fused gate_up_proj weight matrix."""

    def __init__(self, weight_slice: torch.Tensor):
        super().__init__()
        self.weight = weight_slice


class LayerDiscovery:
    """
    Universal architecture-agnostic layer resolver.
    Automatically detects MLP intermediate neuron bases in Transformers (Llama/Qwen/Mistral/GPT),
    Mixture-of-Experts (MoE) routed & 3D expert layers, Convolutional feature maps in CNNs,
    and standard Linear layers.
    """

    # Common down-projection names in modern Transformer MLPs (Qwen, Llama, Mistral, GPT-NeoX, GPT-2)
    DOWN_PROJ_PATTERNS = ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2", "wo")
    UP_PROJ_PATTERNS = ("up_proj", "c_fc", "fc1", "dense_h_to_4h", "w1", "wi", "wi_1")
    GATE_PROJ_PATTERNS = ("gate_proj", "w3", "wi_0")
    FUSED_GATE_UP_PATTERNS = ("gate_up_proj", "c_fc_fused", "w13")
    ATTN_PATH_SEGMENTS = ("attn", "attention", "self_attn", "mha")
    NORM_PATTERNS = ("norm", "ln", "layernorm", "rmsnorm")

    @classmethod
    def _is_attention_module_path(cls, module_name: str) -> bool:
        """Returns True if `module_name` resides inside an attention block rather than an MLP block."""
        parts = module_name.lower().split(".")
        return any(seg in cls.ATTN_PATH_SEGMENTS for seg in parts[:-1])

    @classmethod
    def resolve_fused_gate_up_module(
        cls, model: nn.Module, layer_name: str
    ) -> Optional[nn.Module]:
        """
        If the MLP block containing `layer_name` uses a single fused `gate_up_proj` module
        (e.g., `gate_up_proj`, `c_fc_fused`, `w13`), returns that executable `nn.Module` so
        forward hooks can split its output into `(gate_activations, up_activations)`.
        """
        _, _, num_neurons = cls.resolve_module(model, layer_name)
        parent_path = ".".join(layer_name.split(".")[:-1])
        if not parent_path:
            return None
        parent_mod = model.get_submodule(parent_path)
        for child_name, child_mod in parent_mod.named_children():
            if any(child_name.endswith(p) for p in cls.FUSED_GATE_UP_PATTERNS) and hasattr(
                child_mod, "weight"
            ):
                w = child_mod.weight
                if (
                    (w.ndim == 3 and w.shape[-2] == 2 * num_neurons)
                    or (child_mod.__class__.__name__ == "Conv1D" and w.shape[1] == 2 * num_neurons)
                    or (w.ndim == 2 and w.shape[0] == 2 * num_neurons)
                ):
                    return child_mod
        return None

    @classmethod
    def resolve_mlp_read_write_modules(
        cls, model: nn.Module, layer_name: str
    ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
        """
        Given a target neuron layer name (typically `...mlp.down_proj`), resolves:
          1. write_module: The module that writes neuron activations to the residual stream (`down_proj`)
          2. up_module: The sibling module that reads linear value projections from the residual stream (`up_proj` / `fc1`)
          3. gate_module: The sibling gating module in SwiGLU blocks (`gate_proj`), if present
        Also supports fused `gate_up_proj` layers by splitting `[2 * d_mlp, d_model]` into `(gate, up)` views,
        and 3D MoE expert tensors `[num_experts, d_model, d_mlp]` or `[num_experts, 2 * d_mlp, d_model]`.
        """
        write_mod, _, num_neurons = cls.resolve_module(model, layer_name)
        parent_path = ".".join(layer_name.split(".")[:-1])
        if not parent_path:
            return write_mod, None, None

        parent_mod = model.get_submodule(parent_path)
        up_mod: Optional[nn.Module] = None
        gate_mod: Optional[nn.Module] = None

        for child_name, child_mod in parent_mod.named_children():
            if any(child_name.endswith(p) for p in cls.FUSED_GATE_UP_PATTERNS) and hasattr(
                child_mod, "weight"
            ):
                w = child_mod.weight.detach()
                if w.ndim == 3 and w.shape[-2] == 2 * num_neurons:
                    # 3D MoE fused gate_up_proj: [num_experts, 2 * d_mlp, d_model] -> preserve 3D [num_experts, d_mlp, d_model]
                    gate_w, up_w = w.chunk(2, dim=-2)
                    gate_mod = _SlicedLinearView(gate_w)
                    up_mod = _SlicedLinearView(up_w)
                elif child_mod.__class__.__name__ == "Conv1D" and w.shape[1] == 2 * num_neurons:
                    gate_w, up_w = w.T.chunk(2, dim=0)
                    gate_mod = _SlicedLinearView(gate_w)
                    up_mod = _SlicedLinearView(up_w)
                elif w.shape[0] == 2 * num_neurons:
                    gate_w, up_w = w.chunk(2, dim=0)
                    gate_mod = _SlicedLinearView(gate_w)
                    up_mod = _SlicedLinearView(up_w)
            elif any(child_name.endswith(p) for p in cls.UP_PROJ_PATTERNS):
                up_mod = child_mod
            elif any(child_name.endswith(p) for p in cls.GATE_PROJ_PATTERNS):
                gate_mod = child_mod

        return write_mod, up_mod, gate_mod

    @classmethod
    def resolve_pre_mlp_norm(cls, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        Resolves the LayerNorm / RMSNorm module immediately preceding the MLP block containing `layer_name`
        (e.g., `post_attention_layernorm`, `norm_0`, `norm_1`, `ln_2`, `ffn_norm`).
        """
        parts = layer_name.split(".")
        # Walk up 1 or 2 levels (e.g. from `layers.0.mlp.down_proj` to `layers.0`)
        for depth in range(len(parts) - 1, -1, -1):
            prefix = ".".join(parts[:depth])
            container = model.get_submodule(prefix) if prefix else model
            norms = [
                (cname, cmod)
                for cname, cmod in container.named_children()
                if any(p in cname.lower() for p in cls.NORM_PATTERNS) and hasattr(cmod, "weight")
            ]
            if norms:
                # 1. If layer_name has a numeric block suffix (e.g., mlp_block_0 -> norm_0), match it first
                for seg in parts:
                    if seg.startswith("mlp_block_"):
                        idx_suffix = seg.replace("mlp_block_", "")
                        for cname, cmod in norms:
                            if cname.endswith(f"_{idx_suffix}") or cname.endswith(idx_suffix):
                                return cmod
                # 2. Prefer post-attention / MLP norm if multiple norms exist in a single transformer block
                for cname, cmod in norms:
                    if any(k in cname.lower() for k in ("post", "mlp", "ffn", "ln_2", "norm_1", "norm2")):
                        return cmod
                return norms[-1][1]
        return None

    @classmethod
    def resolve_final_norm(cls, model: nn.Module) -> Optional[nn.Module]:
        """Resolves the model's final pre-lm_head LayerNorm / RMSNorm module if present."""
        for attr in ("final_norm", "norm", "ln_f", "final_layernorm"):
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if isinstance(cand, nn.Module) and hasattr(cand, "weight"):
                    return cand
        for name, mod in reversed(list(model.named_modules())):
            if any(p in name.lower() for p in cls.NORM_PATTERNS) and hasattr(mod, "weight"):
                return mod
        return None

    @classmethod
    def discover_neuron_layers(
        cls,
        model: nn.Module,
        include_all_linear: bool = False,
    ) -> Dict[str, LayerDescriptor]:
        """
        Scans a model and returns a dictionary of targetable neuron layers.
        For Transformer MLP blocks, hooking the *input* of `down_proj` captures the exact
        post-nonlinearity / post-SwiGLU intermediate neuron activations before they mix back
        into the residual stream.
        """
        descriptors: Dict[str, LayerDescriptor] = {}

        for name, module in model.named_modules():
            if not name:
                continue

            # 1. Check if this is a Transformer MLP down-projection layer (captures true MLP neurons via input hook)
            is_down_proj = any(
                name.endswith(pat) for pat in cls.DOWN_PROJ_PATTERNS
            ) and not cls._is_attention_module_path(name)

            if isinstance(module, nn.Linear):
                if is_down_proj:
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="MLP_Neuron_Basis (Pre-DownProj)",
                        num_neurons=module.in_features,
                        hook_target="input",
                    )
                elif include_all_linear:
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="Linear_Output",
                        num_neurons=module.out_features,
                        hook_target="output",
                    )

            # 2. Check for Conv1D (used in HuggingFace GPT-2 style models)
            elif module.__class__.__name__ == "Conv1D" and hasattr(module, "nf"):
                in_features = module.weight.shape[0]
                out_features = module.nf
                if is_down_proj:
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="GPT2_MLP_Neuron_Basis (Pre-DownProj)",
                        num_neurons=in_features,
                        hook_target="input",
                    )
                elif include_all_linear:
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="GPT2_Conv1D_Output",
                        num_neurons=out_features,
                        hook_target="output",
                    )

            # 3. Check for Conv2d layers in Vision models
            elif isinstance(module, nn.Conv2d) and include_all_linear:
                descriptors[name] = LayerDescriptor(
                    name=name,
                    module=module,
                    module_type="Conv2d_Channel_Basis",
                    num_neurons=module.out_channels,
                    hook_target="output",
                )

            # 4. Check for 3D Mixture-of-Experts (MoE) down-projection modules (e.g., Qwen3-MoE, Mixtral fused experts)
            elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 3:
                if is_down_proj or include_all_linear:
                    w = module.weight
                    # [num_experts, d_model, d_mlp] -> hook input of shape [..., d_mlp]
                    d_mlp = w.shape[-1]
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="MoE_Expert_Neuron_Basis (Pre-DownProj)",
                        num_neurons=d_mlp,
                        hook_target="input",
                    )

        # Fallback: if no specialized `down_proj` layers were found, register all Linear/Conv2d layers
        if not descriptors and not include_all_linear:
            return cls.discover_neuron_layers(model, include_all_linear=True)

        return descriptors

    @staticmethod
    def resolve_module(model: nn.Module, layer_name: str) -> Tuple[nn.Module, str, int]:
        """Resolves a dot-separated layer name to its module, default hook target, and neuron count."""
        discovered = LayerDiscovery.discover_neuron_layers(model, include_all_linear=True)
        if layer_name in discovered:
            desc = discovered[layer_name]
            return desc.module, desc.hook_target, desc.num_neurons

        # Direct lookup if user specified an activation module (e.g., 'mlp.act_fn')
        submodule = model.get_submodule(layer_name)
        num_neurons = getattr(submodule, "out_features", getattr(submodule, "in_features", -1))
        return submodule, "output", num_neurons
