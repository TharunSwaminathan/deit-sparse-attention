import copy
import pytest
import torch

from sparse_pet.attention import (
    MaskedTimmAttention, parse_layer_indices, patch_attention_layers
)
from sparse_pet.masking import MaskConfig
from test_masks import FakeTransformer


def test_parse_indices():
    assert parse_layer_indices('10,8,9', 12) == [8, 9, 10]
    with pytest.raises(ValueError):
        parse_layer_indices('3,3', 12)
    with pytest.raises(ValueError):
        parse_layer_indices('12', 12)
    with pytest.raises(ValueError):
        parse_layer_indices('-1', 12)
    with pytest.raises(ValueError):
        parse_layer_indices('', 12)


def test_patch_explicit_layers_preserves_others():
    fake = FakeTransformer().eval()
    wrappers = patch_attention_layers(fake, None, [1, 3])
    assert len(wrappers) == 2
    assert isinstance(fake.blocks[1].attn, MaskedTimmAttention)
    assert isinstance(fake.blocks[3].attn, MaskedTimmAttention)
    assert not isinstance(fake.blocks[0].attn, MaskedTimmAttention)
    with pytest.raises(ValueError):
        patch_attention_layers(fake, None, [1])


def test_only_final_layer_cannot_change_cls_logits_when_cls_row_kept():
    torch.manual_seed(21)
    dense = FakeTransformer().eval()
    sparse = copy.deepcopy(dense).eval()
    patch_attention_layers(sparse, MaskConfig(p=.05), [3])
    x = torch.randn(1, 197, 48)
    with torch.inference_mode():
        a, b = dense(x), sparse(x)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_sparse_intermediate_layer_does_change_cls_logits():
    torch.manual_seed(22)
    dense = FakeTransformer().eval()
    sparse = copy.deepcopy(dense).eval()
    patch_attention_layers(sparse, MaskConfig(p=.05), [1])
    x = torch.randn(1, 197, 48)
    with torch.inference_mode():
        a, b = dense(x), sparse(x)
    assert not torch.allclose(a, b, atol=1e-7, rtol=1e-7)
