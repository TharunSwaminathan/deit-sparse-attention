"""Research prototype: 2x2 spatial block masks for DeiT-Tiny 14x14 patch tokens.

All tensors are [batch, heads, tokens, head_dim] unless noted. This module
constructs masks; it does NOT skip the dense QK^T matrix multiplication.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MaskConfig:
    # The method names correspond to the proposal's 2x2 comparison.
    method: str = "top_p"    # "top_p" or "hybrid"
    p: float = 0.75         # cumulative coarse probability mass
    top_k: int = 4          # min number of blocks / query for hybrid
    similarity_threshold: float | None = None  # None = no protection
    grid_size: int = 14
    tile_size: int = 2

    def __post_init__(self):
        if self.method not in ("top_p", "hybrid"):
            raise ValueError("method must be 'top_p' or 'hybrid'")
        if not (0 < self.p <= 1):
            raise ValueError("p must be in (0,1]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.grid_size <= 0 or self.tile_size <= 0 or self.grid_size % self.tile_size:
            raise ValueError("tile_size must divide grid_size")
        if self.similarity_threshold is not None and not (-1 <= self.similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be within [-1, 1]")


def _spatial_blocks(x: torch.Tensor, grid_size: int, tile_size: int) -> torch.Tensor:
    """[B,H,grid^2,D] -> [B,H,num_blocks,tile^2,D] in row-major blocks."""
    b, h, n, d = x.shape
    if n != grid_size * grid_size:
        raise ValueError(f"expected {grid_size**2} patch tokens; received {n}")
    t = tile_size
    g = grid_size // t
    return (x.reshape(b, h, g, t, g, t, d)
             .permute(0, 1, 2, 4, 3, 5, 6)
             .reshape(b, h, g * g, t * t, d))


def _pairwise_similarity(blocks: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal pairwise cosine similarity within each token block."""
    n = blocks.shape[-2]
    if n < 2:
        return torch.ones(blocks.shape[:-2], device=blocks.device, dtype=blocks.dtype)
    unit = F.normalize(blocks.float(), dim=-1, eps=1e-8)
    sim = unit @ unit.transpose(-2, -1)
    return (sim.sum(dim=(-1, -2)) - n) / (n * (n - 1))


def _nucleus_mask(probabilities: torch.Tensor, p: float) -> torch.Tensor:
    """Smallest descending-probability prefix with cumulative mass >= p.

    probabilities: [..., num_keys], normalized over the final dimension.
    """
    if p >= 1.0:
        return torch.ones_like(probabilities, dtype=torch.bool)
    sorted_probs, order = probabilities.sort(dim=-1, descending=True)
    cumulative_before = sorted_probs.cumsum(-1) - sorted_probs
    # Comparing mass before each position guarantees the crossing element stays.
    sorted_kept = cumulative_before < p
    sorted_kept[..., 0] = True
    return torch.zeros_like(sorted_kept).scatter(-1, order, sorted_kept)


def _block_index(grid_size: int, tile_size: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(grid_size, device=device)
    rows = torch.div(x, tile_size, rounding_mode="floor")
    return (rows[:, None] * (grid_size // tile_size) + rows[None, :]).reshape(-1)


def build_attention_mask(q: torch.Tensor, k: torch.Tensor, config: MaskConfig,
                         diagnostics: bool = False) -> tuple[torch.Tensor, dict]:
    """Build bool mask [B,H,197,197] for CLS+196 patches.

    Coarse Q/K pooling predicts spatial block relevance. Class-token row and
    column are always retained. All block-protection costs count in sparsity.
    """
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must be same [batch,heads,tokens,head_dim] shape")
    b, h, n, d = q.shape
    if n != 1 + config.grid_size**2:
        raise ValueError("expected one CLS token followed by grid_size^2 patches")
    q_blocks = _spatial_blocks(q[:, :, 1:, :], config.grid_size, config.tile_size)
    k_blocks = _spatial_blocks(k[:, :, 1:, :], config.grid_size, config.tile_size)
    qbar, kbar = q_blocks.mean(dim=-2), k_blocks.mean(dim=-2)
    block_count = qbar.shape[-2]
    scale = d ** -0.5
    coarse_scores = (qbar.float() @ kbar.float().transpose(-2, -1)) * scale
    coarse_probs = torch.softmax(coarse_scores, dim=-1)
    coarse_keep = _nucleus_mask(coarse_probs, config.p)
    topk_added = torch.zeros_like(coarse_keep) if diagnostics else None
    if config.method == "hybrid":
        k_count = min(config.top_k, block_count)
        keep_idx = coarse_probs.topk(k_count, dim=-1).indices
        topk_mask = torch.zeros_like(coarse_keep).scatter_(-1, keep_idx, True)
        if diagnostics:
            topk_added = topk_mask & ~coarse_keep
        coarse_keep = coarse_keep | topk_mask

    # Unprotected methods should not pay a similarity-computation tax at
    # inference time. Diagnostics can request both similarities separately.
    q_sim, k_sim = None, None
    if config.similarity_threshold is not None or diagnostics:
        q_sim = _pairwise_similarity(q_blocks)
        k_sim = _pairwise_similarity(k_blocks)
    protected_q, protected_k = None, None
    protection_added = torch.zeros_like(coarse_keep) if diagnostics else None
    if config.similarity_threshold is not None:
        protected_q = q_sim < config.similarity_threshold
        protected_k = k_sim < config.similarity_threshold
        protection_keep = protected_q.unsqueeze(-1) | protected_k.unsqueeze(-2)
        if diagnostics:
            protection_added = protection_keep & ~coarse_keep
        coarse_keep = coarse_keep | protection_keep

    mapping = _block_index(config.grid_size, config.tile_size, q.device)
    patch_mask = coarse_keep.index_select(-2, mapping).index_select(-1, mapping)
    mask = torch.ones((b, h, n, n), dtype=torch.bool, device=q.device)
    mask[:, :, 1:, 1:] = patch_mask

    # Scalars returned only when the caller explicitly requests diagnostics.
    diagnostic = {
        "q_similarity": q_sim,
        "k_similarity": k_sim,
        "coarse_entropy": (-(coarse_probs * coarse_probs.clamp_min(1e-12).log()).sum(-1)
                           if diagnostics else None),
        "coarse_keep": coarse_keep,
        "protected_q": protected_q,
        "protected_k": protected_k,
        "topk_added": topk_added,
        "protection_added": protection_added,
    }
    return mask, diagnostic
