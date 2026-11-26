"""Attention-layer wrapper for timm DeiT-Tiny.

Load the full-attention checkpoint BEFORE patching, because wrapping changes
state_dict names. All sparse methods reuse the SAME frozen model weights.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .masking import MaskConfig, build_attention_mask


class MaskedTimmAttention(nn.Module):
    """Wrap timm VisionTransformer Attention without changing its weights.

    mode='unmasked': manual dense attention, a timing control.
    mode='masked': construct a 2x2 block mask and apply it BEFORE softmax.
    Neither mode implements sparse matrix multiply skipping.
    """

    def __init__(self, original: nn.Module, mask_config: MaskConfig | None = None):
        super().__init__()
        self.original = original
        self.mask_config = mask_config  # None => unmasked wrapper
        self.collect_diagnostics = False
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._rows = []

    def stats(self) -> dict:
        if not self._rows:
            return {}
        total = sum(weight for weight, _ in self._rows)
        keys = self._rows[0][1].keys()
        return {name: sum(weight * row[name] for weight, row in self._rows) / total
                for name in keys}

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal: bool = False) -> torch.Tensor:
        # Current timm Block.forward may pass these optional keyword arguments.
        # Standard DeiT classification uses neither; fail loudly if that changes.
        if attn_mask is not None or is_causal:
            raise NotImplementedError("This DeiT prototype does not accept upstream/causal masks")
        base = self.original
        b, n, c = x.shape
        heads = base.num_heads
        qkv = base.qkv(x).reshape(b, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(base, "q_norm"):
            q = base.q_norm(q)
        if hasattr(base, "k_norm"):
            k = base.k_norm(k)
        scale = getattr(base, "scale", (c // heads) ** -0.5)
        score = (q * scale) @ k.transpose(-2, -1)
        full_probs = torch.softmax(score, dim=-1) if self.collect_diagnostics else None
        if self.mask_config is not None:
            keep, diagnostic = build_attention_mask(q, k, self.mask_config,
                                                    diagnostics=self.collect_diagnostics)
            score = score.masked_fill(~keep, float("-inf"))
        else:
            keep, diagnostic = None, None
        probs = torch.softmax(score, dim=-1)
        if self.collect_diagnostics:
            with torch.no_grad():
                if keep is None:
                    retained_mass, rel_error, sparsity = 1., 0., 0.
                    entropy, similarity = 0., 0.
                    topk_added, protection_added = 0., 0.
                    protected_query, protected_key = 0., 0.
                else:
                    retained_mass = float((full_probs * keep).sum(-1).mean().item())
                    full_out = full_probs @ v
                    masked_out = probs @ v
                    rel_error = float(((full_out - masked_out).float().norm(dim=-1) /
                                      full_out.float().norm(dim=-1).clamp_min(1e-8)).mean().item())
                    sparsity = float((~keep).float().mean().item())
                    entropy = float(diagnostic["coarse_entropy"].mean().item())
                    similarity = float(((diagnostic["q_similarity"].mean() +
                                         diagnostic["k_similarity"].mean()) / 2).item())
                    topk_added = float(diagnostic["topk_added"].float().mean().item())
                    protection_added = float(diagnostic["protection_added"].float().mean().item())
                    protected_query = (float(diagnostic["protected_q"].float().mean().item())
                                       if diagnostic["protected_q"] is not None else 0.)
                    protected_key = (float(diagnostic["protected_k"].float().mean().item())
                                     if diagnostic["protected_k"] is not None else 0.)
                self._rows.append((b, {"retained_mass": retained_mass, "relative_output_error": rel_error,
                                       "realized_sparsity": sparsity, "coarse_entropy": entropy,
                                       "block_similarity": similarity,
                                       "topk_added_fraction": topk_added,
                                       "protection_added_fraction": protection_added,
                                       "protected_query_block_fraction": protected_query,
                                       "protected_key_block_fraction": protected_key}))
        probs = base.attn_drop(probs)
        out = probs @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        return base.proj_drop(base.proj(out))


def parse_layer_indices(spec: str | None, number_of_blocks: int) -> list[int]:
    """Parse comma-separated 0-based DeiT block indices, preserving order.

    The final project's layer set is selected using validation only. The
    choice is shared across all four masks at both target sparsity levels.
    """
    if spec is None or not spec.strip():
        raise ValueError("Provide --layers as comma-separated 0-based indices, e.g. 8,9,10")
    try:
        indices = [int(item.strip()) for item in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid layer specification: {spec!r}") from exc
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Layer list must be nonempty and contain no duplicates")
    if any(i < 0 or i >= number_of_blocks for i in indices):
        raise ValueError(f"Layer indices must be between 0 and {number_of_blocks - 1}")
    return sorted(indices)


def patch_attention_layers(model: nn.Module, mask_config: MaskConfig | None,
                           layer_indices: list[int] | tuple[int, ...],
                           diagnostics: bool = False):
    """Patch explicit 0-based timm ViT blocks in place; return wrappers.

    Call only on a fresh dense model. The original weight objects are nested
    unchanged in the wrapper, and the classification token remains unmasked.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Expected a timm VisionTransformer with .blocks")
    indices = list(layer_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Specify one or more unique Transformer blocks")
    if any(i < 0 or i >= len(model.blocks) for i in indices):
        raise ValueError(f"Valid layer range is [0, {len(model.blocks) - 1}]")
    if any(isinstance(model.blocks[i].attn, MaskedTimmAttention) for i in indices):
        raise ValueError("One or more requested layers are already patched")
    wrappers = []
    for idx in sorted(indices):
        wrapper = MaskedTimmAttention(model.blocks[idx].attn, mask_config)
        wrapper.collect_diagnostics = diagnostics
        model.blocks[idx].attn = wrapper
        wrappers.append(wrapper)
    return wrappers


def patch_last_blocks(model: nn.Module, mask_config: MaskConfig | None,
                      num_blocks: int = 3, diagnostics: bool = False):
    """Backward-compatible shorthand for patching the final N blocks.

    Research runs should use patch_attention_layers with an explicitly selected
    layer set, so that the precise configuration is recorded and reproducible.
    """
    if not hasattr(model, "blocks") or not 1 <= num_blocks <= len(model.blocks):
        raise ValueError("model must have compatible .blocks list")
    indices = list(range(len(model.blocks) - num_blocks, len(model.blocks)))
    return patch_attention_layers(model, mask_config, indices, diagnostics)
