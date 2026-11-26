import torch
import pytest
from sparse_pet.masking import MaskConfig, build_attention_mask, _nucleus_mask, _block_index
from sparse_pet.attention import MaskedTimmAttention
from torch import nn


def qk(seed=7, b=2, heads=3, dim=12):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, heads, 197, dim, generator=g),
            torch.randn(b, heads, 197, dim, generator=g))


def test_shape_cls_all_and_nonempty():
    q, k = qk()
    mask, _ = build_attention_mask(q, k, MaskConfig(p=.3))
    assert mask.dtype == torch.bool and mask.shape == (2, 3, 197, 197)
    assert mask[:, :, 0].all() and mask[:, :, :, 0].all()
    assert mask.any(-1).all()
    assert (~mask).any()


def test_nucleus_exact_prefix():
    p = torch.tensor([.6, .2, .1, .05, .03, .02])
    assert _nucleus_mask(p, .6).tolist() == [True, False, False, False, False, False]
    assert _nucleus_mask(p, .65).tolist() == [True, True, False, False, False, False]
    assert _nucleus_mask(p, 1.).all()


def test_hybrid_never_keeps_fewer_than_top_p():
    q, k = qk()
    p_mask, _ = build_attention_mask(q, k, MaskConfig(method='top_p', p=.2))
    hybrid_mask, _ = build_attention_mask(q, k, MaskConfig(method='hybrid', p=.2, top_k=6))
    assert (hybrid_mask | p_mask).equal(hybrid_mask)


def test_protection_never_deletes_connections():
    q, k = qk()
    a, _ = build_attention_mask(q, k, MaskConfig(p=.2))
    b, info = build_attention_mask(q, k, MaskConfig(p=.2, similarity_threshold=1.))
    assert (a | b).equal(b)
    assert b.all()  # all non-identical random 2x2 blocks have similarity <1
    assert info['protected_q'].all() and info['protected_k'].all()


def test_patch_to_block_row_major():
    ids = _block_index(14, 2, torch.device('cpu')).reshape(14, 14)
    assert ids[0,0] == ids[0,1] == ids[1,0] == ids[1,1] == 0
    assert ids[0,2] == 1 and ids[2,0] == 7 and ids[-1,-1] == 48


class FakeTimmAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads=3
        self.scale=16**-.5
        self.qkv=nn.Linear(48,48*3)
        self.q_norm=nn.Identity()
        self.k_norm=nn.Identity()
        self.attn_drop=nn.Dropout(0.)
        self.proj=nn.Linear(48,48)
        self.proj_drop=nn.Identity()
    def forward(self,x, attn_mask=None, is_causal=False):
        B,N,C=x.shape
        q,k,v=self.qkv(x).reshape(B,N,3,3,16).permute(2,0,3,1,4).unbind(0)
        a=((q*self.scale)@k.transpose(-2,-1)).softmax(-1)
        return self.proj((a@v).transpose(1,2).reshape(B,N,C))


def test_unmasked_wrapper_matches_dense_and_full_mask():
    torch.manual_seed(3)
    attention=FakeTimmAttention().eval()
    x=torch.randn(1,197,48)
    with torch.no_grad():
        reference=attention(x)
        wrapper=MaskedTimmAttention(attention).eval()(x)
        all_kept=MaskedTimmAttention(attention,MaskConfig(p=1.)).eval()(x)
    assert torch.allclose(reference,wrapper,atol=1e-6)
    assert torch.allclose(reference,all_kept,atol=1e-6)


def test_sparse_output_changes_and_is_finite():
    torch.manual_seed(2)
    attention=FakeTimmAttention().eval()
    x=torch.randn(2,197,48)
    w=MaskedTimmAttention(attention,MaskConfig(p=.15)).eval()
    w.collect_diagnostics=True
    with torch.no_grad():
        out=w(x)
        dense=attention(x)
    assert out.shape==dense.shape and torch.isfinite(out).all()
    assert not torch.allclose(out,dense)
    assert 0 < w.stats()['realized_sparsity'] < 1


def test_invalid_config():
    with pytest.raises(ValueError): MaskConfig(p=0)
    with pytest.raises(ValueError): MaskConfig(grid_size=13, tile_size=2)

class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=FakeTimmAttention()
        self.mlp=nn.Linear(48,48)
    def forward(self,x):
        return x + self.mlp(self.attn(x, attn_mask=None, is_causal=False))


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([FakeBlock() for _ in range(4)])
        self.head=nn.Linear(48,5)
    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return self.head(x[:,0])


def test_model_level_3_blocks_change_cls_predictions():
    from sparse_pet.attention import patch_last_blocks
    import copy
    torch.manual_seed(11)
    dense=FakeTransformer().eval()
    unmasked=copy.deepcopy(dense).eval()
    sparse=copy.deepcopy(dense).eval()
    x=torch.randn(1,197,48)
    patch_last_blocks(unmasked, None, num_blocks=3)
    patch_last_blocks(sparse, MaskConfig(method='hybrid',p=.05,top_k=1), num_blocks=3)
    with torch.inference_mode():
        dense_logits=dense(x)
        unmasked_logits=unmasked(x)
        sparse_logits=sparse(x)
    assert torch.allclose(dense_logits,unmasked_logits,atol=1e-6)
    assert not torch.allclose(dense_logits,sparse_logits,atol=1e-6)


def test_mask_diagnostics_topk_activity():
    q, k = qk(seed=14, b=1, heads=2)
    _, diag = build_attention_mask(q, k, MaskConfig(method="hybrid", p=.12, top_k=20), diagnostics=True)
    assert diag["topk_added"] is not None
    assert diag["topk_added"].float().mean().item() > 0
    assert diag["protected_q"] is None
    assert diag["protection_added"].sum() == 0


def test_mask_diagnostics_protection_activity():
    q, k = qk(seed=14, b=1, heads=2)
    mask, diag = build_attention_mask(q, k, MaskConfig(p=.4, similarity_threshold=1.), diagnostics=True)
    assert mask.all()
    assert diag["protection_added"].float().mean().item() > 0
    assert diag["protected_q"].all() and diag["protected_k"].all()


def test_wrapper_stats_report_mask_activity():
    torch.manual_seed(19)
    attention = FakeTimmAttention().eval()
    x = torch.randn(1, 197, 48)
    w = MaskedTimmAttention(attention, MaskConfig(method="hybrid", p=.2, top_k=24,
                                                   similarity_threshold=.95)).eval()
    w.collect_diagnostics=True
    with torch.inference_mode():
        w(x)
    stats = w.stats()
    for metric in ["topk_added_fraction", "protection_added_fraction",
                   "protected_query_block_fraction", "protected_key_block_fraction"]:
        assert metric in stats and 0 <= stats[metric] <= 1
