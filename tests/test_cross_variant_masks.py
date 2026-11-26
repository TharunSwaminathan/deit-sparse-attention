"""A/B/C/D must be able to select different block connections."""
import torch
from sparse_pet.masking import MaskConfig, build_attention_mask
from sparse_pet.compare_masks import binary_mask_comparison


def test_four_variants_are_not_automatically_identical():
    generator = torch.Generator().manual_seed(207)
    q = torch.randn(1, 2, 197, 12, generator=generator)
    k = torch.randn(1, 2, 197, 12, generator=generator)
    configs = {
        "A": MaskConfig(method="top_p", p=.15),
        "B": MaskConfig(method="top_p", p=.15, similarity_threshold=1.),
        "C": MaskConfig(method="hybrid", p=.15, top_k=24),
        "D": MaskConfig(method="hybrid", p=.15, top_k=24,
                        similarity_threshold=1.),
    }
    masks = {variant: build_attention_mask(q, k, config)[0]
             for variant, config in configs.items()}
    assert all(mask[:, :, 0, :].all() and mask[:, :, :, 0].all()
               for mask in masks.values())
    assert binary_mask_comparison(masks["A"], masks["B"])["changed_patch_edges"] > 0
    assert binary_mask_comparison(masks["A"], masks["C"])["changed_patch_edges"] > 0
    assert masks["B"].all() and masks["D"].all()
    assert binary_mask_comparison(masks["B"], masks["D"])["changed_edges"] == 0
