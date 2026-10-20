from __future__ import annotations

from typing import List

import torch

from .types import InterventionMode, InterventionSpec, PoleType


class InterventionKernel:
    """
    High-efficiency vectorized tensor operations that manipulate neuron values
    during forward passes with minimal memory or latency overhead.
    """

    @staticmethod
    def apply(tensor: torch.Tensor, specs: List[InterventionSpec], is_conv2d: bool = False) -> torch.Tensor:
        """
        Applies a list of InterventionSpecs to an activation tensor.
        Supports shapes:
          - [batch, seq_len, hidden_dim] (Transformers / Sequence models)
          - [batch, hidden_dim]          (Standard MLPs)
          - [batch, channels, H, W]      (CNNs when is_conv2d=True)
        """
        if not specs:
            return tensor

        # Clone once to avoid corrupting autograd graphs or shared views when modifying in-place
        out = tensor.clone()
        device = out.device
        dtype = out.dtype

        for spec in specs:
            if not spec.enabled or not spec.neuron_indices:
                continue

            idx = torch.as_tensor(spec.neuron_indices, dtype=torch.long, device=device)

            # Extract the slice corresponding to target neurons
            if is_conv2d and out.ndim == 4:
                # Shape: [B, C, H, W] -> index along channel dimension (dim=1)
                sub = out[:, idx, :, :]
                modified = InterventionKernel._transform_slice(sub, spec, dtype)
                out[:, idx, :, :] = modified

            elif out.ndim == 3:
                # Shape: [B, S, D] -> Transformer sequence activation
                if spec.token_positions is not None:
                    pos_idx = torch.as_tensor(spec.token_positions, dtype=torch.long, device=device)
                    # Index both token position and neuron dimension
                    sub = out[:, pos_idx[:, None], idx]
                    modified = InterventionKernel._transform_slice(sub, spec, dtype)
                    out[:, pos_idx[:, None], idx] = modified
                else:
                    sub = out[:, :, idx]
                    modified = InterventionKernel._transform_slice(sub, spec, dtype)
                    out[:, :, idx] = modified

            elif out.ndim == 2:
                # Shape: [B, D] -> Standard 2D batch activation
                sub = out[:, idx]
                modified = InterventionKernel._transform_slice(sub, spec, dtype)
                out[:, idx] = modified

            else:
                # Generic fallback: assume last dimension is the neuron basis
                sub = out[..., idx]
                modified = InterventionKernel._transform_slice(sub, spec, dtype)
                out[..., idx] = modified

        return out

    @staticmethod
    def _transform_slice(sub: torch.Tensor, spec: InterventionSpec, dtype: torch.dtype) -> torch.Tensor:
        """Executes the specific pole-aware mathematical operation on the selected neuron slice."""
        if spec.custom_fn is not None:
            return spec.custom_fn(sub).to(dtype)

        # Build pole mask if targeting only POSITIVE (> 0) or NEGATIVE (< 0) neuron pole
        if spec.pole == PoleType.POSITIVE:
            pole_mask = sub > 0
        elif spec.pole == PoleType.NEGATIVE:
            pole_mask = sub < 0
        else:
            pole_mask = torch.ones_like(sub, dtype=torch.bool)

        mode = spec.mode
        val = float(spec.value)

        if mode == InterventionMode.ZERO_ABLATE:
            return torch.where(pole_mask, torch.zeros_like(sub), sub)

        elif mode == InterventionMode.HARD_CLAMP:
            clamp_tensor = torch.full_like(sub, val)
            return torch.where(pole_mask, clamp_tensor, sub)

        elif mode == InterventionMode.SCALE:
            return torch.where(pole_mask, sub * val, sub)

        elif mode == InterventionMode.ADDITIVE_STEER:
            return torch.where(pole_mask, sub + val, sub)

        elif mode == InterventionMode.WEAK_PRUNE:
            # Cammarata (2026) "Do Weak Activations Matter?" filter:
            # Zero out activations with magnitude smaller than `spec.value`
            weak_mask = pole_mask & (sub.abs() < abs(val))
            return torch.where(weak_mask, torch.zeros_like(sub), sub)

        elif mode == InterventionMode.POLE_RECTIFY:
            # Keep only the selected pole; zero out the opposite pole
            return torch.where(pole_mask, sub, torch.zeros_like(sub))

        return sub
