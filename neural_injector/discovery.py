from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ArchitectureAdapter:
    """Declarative specification for a model family's MLP and normalization layout."""
    family: str
    down_proj_names: Tuple[str, ...]
    up_proj_names: Tuple[str, ...]
    gate_proj_names: Tuple[str, ...]
    fused_gate_up_names: Tuple[str, ...] = ()
    pre_mlp_norm_names: Tuple[str, ...] = ()
    norm_unit_offset: bool = False  # True for GemmaRMSNorm where scale is (1.0 + weight)


ARCHITECTURE_REGISTRY: Dict[str, ArchitectureAdapter] = {
    "llama_qwen_mistral": ArchitectureAdapter(
        family="llama_qwen_mistral",
        down_proj_names=("down_proj",),
        up_proj_names=("up_proj",),
        gate_proj_names=("gate_proj",),
        fused_gate_up_names=("gate_up_proj",),
        pre_mlp_norm_names=("post_attention_layernorm", "ffn_norm"),
    ),
    "gemma": ArchitectureAdapter(
        family="gemma",
        down_proj_names=("down_proj",),
        up_proj_names=("up_proj",),
        gate_proj_names=("gate_proj",),
        pre_mlp_norm_names=("pre_feedforward_layernorm", "post_attention_layernorm"),
        norm_unit_offset=True,
    ),
    "mixtral_fairseq": ArchitectureAdapter(
        family="mixtral_fairseq",
        # In Mixtral / original LLaMA: act_fn(w1(x)) * w3(x) -> w1 is GATE, w3 is UP, w2 is DOWN
        down_proj_names=("w2",),
        up_proj_names=("w3",),
        gate_proj_names=("w1",),
        fused_gate_up_names=("w13",),
        pre_mlp_norm_names=("post_attention_layernorm", "ffn_norm"),
    ),
    "phi3_glm": ArchitectureAdapter(
        family="phi3_glm",
        down_proj_names=("down_proj", "dense_4h_to_h"),
        up_proj_names=("up_proj",),
        gate_proj_names=("gate_proj",),
        fused_gate_up_names=("gate_up_proj", "dense_h_to_4h"),
        pre_mlp_norm_names=("post_attention_layernorm",),
    ),
    "t5": ArchitectureAdapter(
        family="t5",
        down_proj_names=("wo",),
        up_proj_names=("wi_1", "wi"),
        gate_proj_names=("wi_0",),
        pre_mlp_norm_names=("layer_norm",),
    ),
    "gpt2": ArchitectureAdapter(
        family="gpt2",
        down_proj_names=("c_proj",),
        up_proj_names=("c_fc",),
        gate_proj_names=(),
        fused_gate_up_names=("c_fc_fused",),
        pre_mlp_norm_names=("ln_2",),
    ),
    "gptj_codegen": ArchitectureAdapter(
        family="gptj_codegen",
        down_proj_names=("fc_out",),
        up_proj_names=("fc_in",),
        gate_proj_names=(),
        pre_mlp_norm_names=("ln_1",),
    ),
    "gpt_neox": ArchitectureAdapter(
        family="gpt_neox",
        down_proj_names=("dense_4h_to_h",),
        up_proj_names=("dense_h_to_4h",),
        gate_proj_names=(),
        pre_mlp_norm_names=("post_attention_layernorm",),
    ),
    "bert_vit_convnext": ArchitectureAdapter(
        family="bert_vit_convnext",
        down_proj_names=("fc2", "pwconv2", "output.dense"),
        up_proj_names=("fc1", "pwconv1", "intermediate.dense"),
        gate_proj_names=(),
        pre_mlp_norm_names=("layernorm_after", "norm2", "norm"),
    ),
}


@dataclass
class LayerDescriptor:
    """Metadata describing a targetable neuron layer inside a PyTorch model."""
    name: str
    module: nn.Module
    module_type: str
    num_neurons: int
    hook_target: str  # "output" (standard Linear/Act) or "input" (e.g., down_proj in SwiGLU/MLP LLMs)
    expert_idx: Optional[int] = None
    is_quantized: bool = False


class _SlicedLinearView(nn.Module):
    """Lightweight read-only module view over half of a fused gate_up_proj weight matrix."""

    def __init__(self, weight_slice: torch.Tensor):
        super().__init__()
        self.weight = weight_slice


class LayerDiscovery:
    """
    Universal architecture-agnostic layer resolver backed by `ARCHITECTURE_REGISTRY`.
    Automatically detects MLP intermediate neuron bases in Transformers (Llama/Qwen/Mistral/Gemma/
    Mixtral/Phi-3/T5/GPT-2/GPT-J/GPT-NeoX/BERT/ViT/ConvNeXt), Mixture-of-Experts (MoE) per-expert
    and 3D expert layers, Convolutional feature maps in CNNs, and standard Linear layers.
    """

    DOWN_PROJ_PATTERNS = (
        "down_proj",
        "c_proj",
        "fc2",
        "fc_out",
        "pwconv2",
        "dense_4h_to_h",
        "w2",
        "wo",
    )
    UP_PROJ_PATTERNS = (
        "up_proj",
        "c_fc",
        "fc1",
        "fc_in",
        "pwconv1",
        "dense_h_to_4h",
        "w3",
        "w1",
        "wi",
        "wi_1",
    )
    GATE_PROJ_PATTERNS = ("gate_proj", "w1", "wi_0")
    FUSED_GATE_UP_PATTERNS = ("gate_up_proj", "c_fc_fused", "w13")
    ATTN_PATH_SEGMENTS = ("attn", "attention", "self_attn", "mha")
    NORM_PATTERNS = ("norm", "ln", "layernorm", "rmsnorm")
    _EXPERT_PATH_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
    _TRAILING_INDEX_RE = re.compile(r"^(.+?)_?(\d+)$")

    @classmethod
    def _is_attention_module_path(cls, module_name: str) -> bool:
        """Returns True if `module_name` resides inside an attention block rather than an MLP block."""
        parts = module_name.lower().split(".")
        return any(seg in cls.ATTN_PATH_SEGMENTS for seg in parts[:-1])

    @classmethod
    def _is_mlp_output_dense(cls, model: nn.Module, module_name: str) -> bool:
        """
        Detects BERT / RoBERTa / ViT style `...output.dense` MLP down-projections
        (which live next to an `...intermediate.dense` module outside attention).
        """
        if cls._is_attention_module_path(module_name):
            return False
        if not module_name.endswith("output.dense"):
            return False
        prefix = module_name[: -len("output.dense")].rstrip(".")
        cand_inter = f"{prefix}.intermediate.dense" if prefix else "intermediate.dense"
        try:
            model.get_submodule(cand_inter)
            return True
        except AttributeError:
            return False

    @classmethod
    def _extract_expert_idx(cls, module_name: str) -> Optional[int]:
        match = cls._EXPERT_PATH_RE.search(module_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def is_quantized_or_offloaded(module: nn.Module) -> bool:
        """
        Returns True if `module` uses quantized integer/4-bit/8-bit weights (e.g., bitsandbytes,
        GPTQ, AWQ) or resides on the PyTorch `meta` device.
        """
        cls_name = module.__class__.__name__.lower()
        if any(q in cls_name for q in ("8bit", "4bit", "quant", "qlinear", "awq", "gptq")):
            return True
        if hasattr(module, "qweight") or hasattr(module, "quant_state"):
            return True
        w = getattr(module, "weight", None)
        if isinstance(w, torch.Tensor):
            if getattr(w, "is_meta", False) or str(w.device).startswith("meta"):
                return True
            if not w.is_floating_point():
                return True
            w_cls = w.__class__.__name__.lower()
            if any(q in w_cls for q in ("4bit", "8bit", "int8", "quant")):
                return True
        return False

    @classmethod
    def resolve_moe_router(cls, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        If `layer_name` is inside an MoE block (e.g. `...mlp.experts.0.down_proj`),
        resolves the sibling router / gate module (`...mlp.gate` or `...mlp.router`).
        """
        parts = layer_name.split(".")
        for depth in range(len(parts) - 1, -1, -1):
            prefix = ".".join(parts[:depth])
            container = model.get_submodule(prefix) if prefix else model
            for rname in ("gate", "router", "shared_expert_gate"):
                if hasattr(container, rname):
                    cand = getattr(container, rname)
                    if isinstance(cand, nn.Module) and cand is not model.get_submodule(layer_name):
                        return cand
        return None

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
        Given a target neuron layer name (typically `...mlp.down_proj`, `...w2`, `...fc_out`), resolves:
          1. write_module: The module that writes neuron activations to the residual stream (`down_proj`)
          2. up_module: The sibling module that reads linear value projections from the residual stream
          3. gate_module: The sibling gating module in SwiGLU/GeGLU blocks, if present.

        Correctly disambiguates Mixtral / original-LLaMA `w1` (gate) vs. `w3` (up) when `w3` is present,
        while treating `w1` as `up` in non-gated 2-layer `w1`/`w2` MLPs. Also resolves BERT/ViT
        `intermediate.dense` -> `output.dense` siblings, fused `gate_up_proj` `[2 * d_mlp, d_model]`,
        and 3D MoE expert tensors.
        """
        write_mod, _, num_neurons = cls.resolve_module(model, layer_name)

        # Handle BERT / RoBERTa / ViT `...output.dense` paired with `...intermediate.dense`
        if cls._is_mlp_output_dense(model, layer_name):
            prefix = layer_name[: -len("output.dense")].rstrip(".")
            cand_inter = f"{prefix}.intermediate.dense" if prefix else "intermediate.dense"
            try:
                inter_mod = model.get_submodule(cand_inter)
                return write_mod, inter_mod, None
            except AttributeError:
                pass

        parent_path = ".".join(layer_name.split(".")[:-1])
        parent_mod = model.get_submodule(parent_path) if parent_path else model
        children = dict(parent_mod.named_children())
        up_mod: Optional[nn.Module] = None
        gate_mod: Optional[nn.Module] = None

        # 1. Check for Mixtral / original-LLaMA (w1 = gate, w3 = up, w2 = down) vs plain (w1 = up, w2 = down)
        has_w1 = "w1" in children
        has_w3 = "w3" in children
        if has_w1 and has_w3:
            gate_mod = children["w1"]
            up_mod = children["w3"]
            return write_mod, up_mod, gate_mod
        elif has_w1 and not has_w3:
            up_mod = children["w1"]

        # 2. Check for standard split or fused gate/up modules
        for child_name, child_mod in children.items():
            if child_mod is write_mod:
                continue
            if any(child_name.endswith(p) for p in cls.FUSED_GATE_UP_PATTERNS) and hasattr(
                child_mod, "weight"
            ):
                w = child_mod.weight.detach()
                if w.ndim == 3 and w.shape[-2] == 2 * num_neurons:
                    gate_w, up_w = w.chunk(2, dim=-2)
                    gate_mod = _SlicedLinearView(gate_w)
                    up_mod = _SlicedLinearView(up_w)
                elif child_mod.__class__.__name__ == "Conv1D" and w.shape[1] == 2 * num_neurons:
                    gate_w, up_w = w.T.chunk(2, dim=0)
                    gate_mod = _SlicedLinearView(gate_w)
                    up_mod = _SlicedLinearView(up_w)
                elif w.ndim == 2 and w.shape[0] == 2 * num_neurons:
                    gate_w, up_w = w.chunk(2, dim=0)
                    gate_mod = _SlicedLinearView(gate_w)
                    up_mod = _SlicedLinearView(up_w)
            elif child_name != "w1" and any(child_name.endswith(p) for p in cls.GATE_PROJ_PATTERNS):
                gate_mod = child_mod
            elif child_name not in ("w1", "w3") and any(
                child_name.endswith(p) for p in cls.UP_PROJ_PATTERNS
            ):
                up_mod = child_mod

        return write_mod, up_mod, gate_mod

    @classmethod
    def resolve_pre_mlp_norm(cls, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        Resolves the LayerNorm / RMSNorm module immediately preceding the MLP block containing `layer_name`
        (e.g., `pre_feedforward_layernorm` in Gemma-2/3, `post_attention_layernorm` in Llama/Qwen/Mistral,
        `ln_2` in GPT-2, `ln_1` in GPT-J, or indexed block norms such as `norm_0`, `norm_1`).
        """
        parts = layer_name.split(".")
        for depth in range(len(parts) - 1, -1, -1):
            prefix = ".".join(parts[:depth])
            container = model.get_submodule(prefix) if prefix else model
            norms = [
                (cname, cmod)
                for cname, cmod in container.named_children()
                if any(p in cname.lower() for p in cls.NORM_PATTERNS) and hasattr(cmod, "weight")
            ]
            if not norms:
                continue

            # 1. General trailing numeric index match between ancestor segment and sibling norm
            #    (e.g., block_0 / mlp_block_0 / layer0 -> norm_0 / ln_0)
            for seg in parts[depth:]:
                m = cls._TRAILING_INDEX_RE.match(seg)
                if m:
                    idx_suffix = m.group(2)
                    for cname, cmod in norms:
                        cm = cls._TRAILING_INDEX_RE.match(cname)
                        if cm and cm.group(2) == idx_suffix:
                            return cmod

            # 2. Prioritize Gemma-style `pre_feedforward_layernorm` / `pre_mlp_layernorm` first
            for cname, cmod in norms:
                clow = cname.lower()
                if any(k in clow for k in ("pre_feedforward", "pre_mlp", "pre_ffn")):
                    return cmod

            # 3. Next, prefer standard post-attention / MLP norm (excluding Gemma's post_feedforward_layernorm)
            for cname, cmod in norms:
                clow = cname.lower()
                if "post_feedforward" in clow or "post_mlp" in clow:
                    continue
                if any(k in clow for k in ("post_attention", "post_attn", "mlp", "ffn", "ln_2", "norm_1", "norm2", "layernorm_after")):
                    return cmod

            # 4. Fallback to the last non-post-feedforward norm in the block
            non_post_ffn = [
                cmod
                for cname, cmod in norms
                if "post_feedforward" not in cname.lower() and "post_mlp" not in cname.lower()
            ]
            return non_post_ffn[-1] if non_post_ffn else norms[-1][1]
        return None

    @classmethod
    def resolve_final_norm(cls, model: nn.Module) -> Optional[nn.Module]:
        """Resolves the model's final pre-lm_head LayerNorm / RMSNorm module if present."""
        for attr in ("final_norm", "norm", "ln_f", "final_layernorm"):
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if isinstance(cand, nn.Module) and hasattr(cand, "weight"):
                    return cand
        for sub_prefix in ("model", "transformer", "gpt_neox"):
            if hasattr(model, sub_prefix):
                sub_mod = getattr(model, sub_prefix)
                for attr in ("norm", "final_layernorm", "ln_f"):
                    if hasattr(sub_mod, attr):
                        cand = getattr(sub_mod, attr)
                        if isinstance(cand, nn.Module) and hasattr(cand, "weight"):
                            return cand
        for name, mod in reversed(list(model.named_modules())):
            if any(p in name.lower() for p in cls.NORM_PATTERNS) and hasattr(mod, "weight"):
                return mod
        return None

    @staticmethod
    def extract_norm_scale(
        norm_module: Optional[nn.Module],
        d_model: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """
        Extracts the multiplicative scaling vector gamma from a LayerNorm / RMSNorm module.
        Automatically accounts for Google Gemma's `GemmaRMSNorm` parameterization where
        the effective scale is `1.0 + norm_module.weight`.
        """
        if norm_module is None or not hasattr(norm_module, "weight"):
            return None
        w = getattr(norm_module, "weight", None)
        if not isinstance(w, torch.Tensor):
            return None
        if d_model is not None and w.numel() != d_model:
            return None
        scale = w.detach().to(device=device or w.device, dtype=dtype)
        cls_name = norm_module.__class__.__name__.lower()
        has_unit_offset = (
            "gemma" in cls_name
            or bool(getattr(norm_module, "add_unit_offset", False))
            or bool(getattr(norm_module, "unit_offset", False))
        )
        if has_unit_offset:
            scale = scale + 1.0
        return scale

    @classmethod
    def discover_neuron_layers(
        cls,
        model: nn.Module,
        include_all_linear: bool = False,
    ) -> Dict[str, LayerDescriptor]:
        """
        Scans a model and returns a dictionary of targetable neuron layers.
        For Transformer MLP blocks, hooking the *input* of `down_proj` / `c_proj` / `fc2` /
        `fc_out` / `w2` / `output.dense` captures the exact post-nonlinearity / post-SwiGLU
        intermediate neuron activations before they mix back into the residual stream.
        """
        descriptors: Dict[str, LayerDescriptor] = {}

        for name, module in model.named_modules():
            if not name:
                continue

            is_down_proj = (
                any(name.endswith(pat) for pat in cls.DOWN_PROJ_PATTERNS)
                and not cls._is_attention_module_path(name)
            ) or cls._is_mlp_output_dense(model, name)

            expert_idx = cls._extract_expert_idx(name)
            quant_flag = cls.is_quantized_or_offloaded(module)

            if isinstance(module, nn.Linear):
                if is_down_proj:
                    m_type = (
                        f"MoE_Expert_{expert_idx}_Neuron_Basis (Pre-DownProj)"
                        if expert_idx is not None
                        else "MLP_Neuron_Basis (Pre-DownProj)"
                    )
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type=m_type,
                        num_neurons=module.in_features,
                        hook_target="input",
                        expert_idx=expert_idx,
                        is_quantized=quant_flag,
                    )
                elif include_all_linear:
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="Linear_Output",
                        num_neurons=module.out_features,
                        hook_target="output",
                        expert_idx=expert_idx,
                        is_quantized=quant_flag,
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
                        expert_idx=expert_idx,
                        is_quantized=quant_flag,
                    )
                elif include_all_linear:
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="GPT2_Conv1D_Output",
                        num_neurons=out_features,
                        hook_target="output",
                        expert_idx=expert_idx,
                        is_quantized=quant_flag,
                    )

            # 3. Check for Conv2d layers in Vision models
            elif isinstance(module, nn.Conv2d) and include_all_linear:
                descriptors[name] = LayerDescriptor(
                    name=name,
                    module=module,
                    module_type="Conv2d_Channel_Basis",
                    num_neurons=module.out_channels,
                    hook_target="output",
                    expert_idx=expert_idx,
                    is_quantized=quant_flag,
                )

            # 4. Check for 3D Mixture-of-Experts (MoE) down-projection modules or fused expert containers
            elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 3:
                if is_down_proj or include_all_linear:
                    w = module.weight
                    d_mlp = w.shape[-1]
                    descriptors[name] = LayerDescriptor(
                        name=name,
                        module=module,
                        module_type="MoE_Expert_Neuron_Basis (Pre-DownProj)",
                        num_neurons=d_mlp,
                        hook_target="input",
                        expert_idx=expert_idx,
                        is_quantized=quant_flag,
                    )
            elif (
                hasattr(module, "down_proj")
                and isinstance(getattr(module, "down_proj"), nn.Parameter)
                and getattr(module, "down_proj").ndim == 3
            ):
                # Newer HuggingFace MoE expert container where `down_proj` is a 3D nn.Parameter
                w_dp = getattr(module, "down_proj")
                d_mlp = w_dp.shape[-1] if w_dp.shape[-1] >= w_dp.shape[-2] else w_dp.shape[-2]
                descriptors[name] = LayerDescriptor(
                    name=name,
                    module=module,
                    module_type="MoE_Parameter_Container_Basis",
                    num_neurons=d_mlp,
                    hook_target="output",
                    expert_idx=expert_idx,
                    is_quantized=quant_flag,
                )

        # Fallback: if no specialized `down_proj` layers were found, register all Linear/Conv2d layers
        if not descriptors and not include_all_linear:
            return cls.discover_neuron_layers(model, include_all_linear=True)

        return descriptors

    @staticmethod
    def resolve_module(model: nn.Module, layer_name: str) -> Tuple[nn.Module, str, int]:
        """Resolves a dot-separated layer name to its module, default hook target, and neuron count."""
        discovered_primary = LayerDiscovery.discover_neuron_layers(model, include_all_linear=False)
        if layer_name in discovered_primary:
            desc = discovered_primary[layer_name]
            return desc.module, desc.hook_target, desc.num_neurons

        discovered_all = LayerDiscovery.discover_neuron_layers(model, include_all_linear=True)
        if layer_name in discovered_all:
            desc = discovered_all[layer_name]
            return desc.module, desc.hook_target, desc.num_neurons

        # Direct lookup if user specified an activation module (e.g., 'mlp.act_fn')
        submodule = model.get_submodule(layer_name)
        num_neurons = getattr(submodule, "out_features", getattr(submodule, "in_features", -1))
        return submodule, "output", num_neurons
