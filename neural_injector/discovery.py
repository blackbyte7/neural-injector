from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch.nn as nn


@dataclass
class LayerDescriptor:
    """Metadata describing a targetable neuron layer inside a PyTorch model."""
    name: str
    module: nn.Module
    module_type: str
    num_neurons: int
    hook_target: str  # "output" (standard Linear/Act) or "input" (e.g., down_proj in SwiGLU LLMs)


class LayerDiscovery:
    """
    Universal architecture-agnostic layer resolver.
    Automatically detects MLP intermediate neuron bases in Transformers (Llama/Qwen/Mistral/GPT),
    Convolutional feature maps in CNNs, and standard Linear layers.
    """

    # Common down-projection names in modern Transformer MLPs (Qwen, Llama, Mistral, GPT-NeoX, GPT-2)
    DOWN_PROJ_PATTERNS = ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2", "wo")
    UP_PROJ_PATTERNS = ("up_proj", "c_fc", "fc1", "dense_h_to_4h", "w1", "wi")
    GATE_PROJ_PATTERNS = ("gate_proj", "w3", "wi_0")

    @classmethod
    def resolve_mlp_read_write_modules(
            cls, model: nn.Module, layer_name: str
    ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
        """
        Given a target neuron layer name (typically `...mlp.down_proj`), resolves:
          1. write_module: The module that writes neuron activations to the residual stream (`down_proj`)
          2. up_module: The sibling module that reads linear value projections from the residual stream (`up_proj` / `fc1`)
          3. gate_module: The sibling gating module in SwiGLU blocks (`gate_proj`), if present
        """
        write_mod, _, _ = cls.resolve_module(model, layer_name)
        parent_path = ".".join(layer_name.split(".")[:-1])
        if not parent_path:
            return write_mod, None, None

        parent_mod = model.get_submodule(parent_path)
        up_mod: Optional[nn.Module] = None
        gate_mod: Optional[nn.Module] = None

        for child_name, child_mod in parent_mod.named_children():
            if any(child_name.endswith(p) for p in cls.UP_PROJ_PATTERNS):
                up_mod = child_mod
            elif any(child_name.endswith(p) for p in cls.GATE_PROJ_PATTERNS):
                gate_mod = child_mod

        return write_mod, up_mod, gate_mod

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
            is_down_proj = any(name.endswith(pat) for pat in cls.DOWN_PROJ_PATTERNS)

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
