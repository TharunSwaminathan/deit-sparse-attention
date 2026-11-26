import numpy as np
import torch
import pytest
from sparse_pet.block_diagnostics import block_metrics, select_indices, describe, FIELDS
from sparse_pet.masking import MaskConfig


def test_full_mask_matches_dense():
    torch.manual_seed(7)
    q,k,v=[torch.randn(2,3,17,4) for _ in range(3)]
    m=block_metrics(q,k,v,MaskConfig(p=1,grid_size=4,tile_size=2),.5)
    assert m.shape==(2,3,4,len(FIELDS))
    torch.testing.assert_close(m[...,3],torch.ones_like(m[...,3]))
    torch.testing.assert_close(m[...,4:],torch.zeros_like(m[...,4:]))


def test_shared_features_and_sparse_errors():
    torch.manual_seed(8)
    q,k,v=[torch.randn(2,3,17,4) for _ in range(3)]
    a=block_metrics(q,k,v,MaskConfig(p=.3,grid_size=4,tile_size=2),.5)
    b=block_metrics(q,k,v,MaskConfig(p=.3,similarity_threshold=1,grid_size=4,tile_size=2),.5)
    torch.testing.assert_close(a[...,:3],b[...,:3])
    assert (a[...,5]>0).any() and (a[...,4]>0).any()
    torch.testing.assert_close(b[...,4:],torch.zeros_like(b[...,4:]))


def test_sampling_and_summary():
    ids=select_indices([1,2,4,5],[0,0,0,1,1,1],2)
    assert ids==[1,2,4,5]
    with pytest.raises(ValueError): select_indices(ids,[0,0,0,1,1,1],3)
    data=np.tile(np.arange(8,dtype=float)[:,None],(1,len(FIELDS)))
    cuts={k:[1.5,3.5,5.5] for k in ('q_similarity','coarse_entropy')}
    r=describe(data,cuts)
    assert r['strata']['q_similarity']['spearman_rho_error']==1
    assert [b['n'] for b in r['strata']['q_similarity']['bins']]==[2]*4
