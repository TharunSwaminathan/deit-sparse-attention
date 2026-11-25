"""Validation-only block fidelity analysis on identical dense layer inputs."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from .common import get_transforms, load_dense_checkpoint, sha256_file, save_json
from .evaluate import make_config
from .masking import build_attention_mask, _spatial_blocks

FIELDS = ('q_similarity','mean_key_similarity','coarse_entropy','retained_mass',
          'relative_output_error','realized_sparsity')


def block_metrics(q, k, v, config, scale):
    """Return [batch, heads, spatial query blocks, metrics]; omit CLS queries.

    CLS keys remain in the denominator and attention outputs. Similarities
    and entropy come directly from the existing masking implementation.
    """
    keep, d = build_attention_mask(q,k,config,diagnostics=True)
    scores = (q * scale) @ k.transpose(-2,-1)
    full = scores.softmax(-1)
    masked = scores.masked_fill(~keep, -torch.inf).softmax(-1)
    ref = full @ v
    out = masked @ v
    error = (ref-out).float().norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(1e-8)
    def pool(x):
        return _spatial_blocks(x[:,:,1:].unsqueeze(-1),config.grid_size,config.tile_size).mean(dim=(-1,-2))
    return torch.stack((d['q_similarity'], d['k_similarity'].mean(-1,keepdim=True).expand_as(d['q_similarity']),
                        d['coarse_entropy'],pool((full*keep).sum(-1)),pool(error),
                        pool((~keep).float().mean(-1))),dim=-1)


def select_indices(indices, labels, per_class=3, seed=42):
    """Select equal counts from each class, only within validation indices."""
    if per_class < 1 or len(set(indices)) != len(indices):
        raise ValueError('Invalid sample request or duplicate validation indices')
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    ix = np.asarray(indices,dtype=int)
    if len(ix)==0 or ix.min()<0 or ix.max()>=len(y):
        raise ValueError('Invalid validation indices')
    picked=[]
    for cls in sorted(np.unique(y)):
        pool=ix[y[ix]==cls]
        if len(pool)<per_class:
            raise ValueError(f'Not enough validation samples in class {cls}')
        picked.extend(rng.choice(pool,per_class,replace=False).tolist())
    return sorted(picked)


def describe(data, cutpoints):
    """Descriptive rank correlations and shared, layer-specific quartile bins."""
    from scipy.stats import spearmanr
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite block metrics')
    result={'n_query_blocks':len(data),'means':dict(zip(FIELDS,data.mean(0).tolist())), 'strata':{}}
    for name in ('q_similarity','coarse_entropy'):
        x=data[:,FIELDS.index(name)]
        y=data[:,FIELDS.index('relative_output_error')]
        rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None
        if rho is not None and not np.isfinite(rho): rho=None
        cuts=np.asarray(cutpoints[name]); bins=np.searchsorted(cuts,x,side='right')
        rows=[]
        for i in range(4):
            chunk=data[bins==i]
            rows.append({'quartile':i+1,'n':len(chunk),'means':dict(zip(FIELDS,chunk.mean(0).tolist())) if len(chunk) else None})
        result['strata'][name]={'cutpoints':cuts.tolist(),'spearman_rho_error':rho,'bins':rows}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--frozen-config',required=True)
    ap.add_argument('--data-root',default='data/downloads')
    ap.add_argument('--manifest',default='outputs/split_manifest.json')
    ap.add_argument('--per-class',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    from torchvision.datasets import OxfordIIITPet
    from torch.utils.data import DataLoader, Subset
    f=json.loads(Path(args.frozen_config).read_text())
    digest=sha256_file(args.checkpoint)
    if digest!=f['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    manifest=json.loads(Path(args.manifest).read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,_=load_dense_checkpoint(args.checkpoint,device)
    _,tf=get_transforms(model)
    ds=OxfordIIITPet(args.data_root,split='trainval',target_types='category',transform=tf,download=False)
    val=manifest['val_indices']; train=manifest['train_indices']
    if manifest['dataset_count']!=len(ds) or len(train)!=2944 or len(val)!=736 or len(set(train+val))!=len(ds) or set(train+val)!=set(range(len(ds))):
        raise ValueError('Unexpected split manifest')
    indices=select_indices(val,ds._labels,args.per_class)
    if len(ds.classes)!=37: raise ValueError('Unexpected number of classes')
    loader=DataLoader(Subset(ds,indices),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    layers=f['layer_indices']; configs={}
    for target in (25,50):
        for variant in 'ABCD':
            label=f'{variant}_{target}'; r=f['selected'][label]
            configs[label]=make_config(r['variant'],r['p'],r['top_k'],r['similarity_threshold'])
    capture={}; handles=[]
    for layer in layers:
        base=model.blocks[layer].attn
        def hook(module,inputs,output,idx=layer,attn=base):
            b,n,c=output.shape; h=attn.num_heads
            q,k,v=output.reshape(b,n,3,h,c//(3*h)).permute(2,0,3,1,4).unbind(0)
            if hasattr(attn,'q_norm'): q=attn.q_norm(q)
            if hasattr(attn,'k_norm'): k=attn.k_norm(k)
            capture[idx]=(q,k,v,attn.scale)
        handles.append(base.qkv.register_forward_hook(hook))
    data={(layer,label):[] for layer in layers for label in configs}
    dest=Path(args.output_dir); dest.mkdir(parents=True,exist_ok=True)
    count=0
    print(f'Device: {device}; validation images: {len(indices)}; configurations: {len(configs)}',flush=True)
    try:
        with torch.inference_mode(), gzip.open(dest/'query_blocks.csv.gz','wt',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream); writer.writerow(['image_id','layer','head','query_block','configuration',*FIELDS])
            for x,_ in loader:
                capture.clear(); model(x.to(device))
                if set(capture)!=set(layers): raise RuntimeError('Missing dense QKV capture')
                for layer in layers:
                    q,k,v,scale=capture[layer]
                    for label,config in configs.items():
                        m=block_metrics(q,k,v,config,scale).cpu().numpy()
                        data[layer,label].append(m.reshape(-1,len(FIELDS)))
                        for bi,head,block in np.ndindex(m.shape[:-1]):
                            writer.writerow([f'trainval:{indices[count+bi]}',layer,head,block,label,*m[bi,head,block].tolist()])
                count+=len(x); print(f'Processed {count}/{len(indices)} validation images',flush=True)
    finally:
        for h in handles: h.remove()
    if count!=len(indices): raise RuntimeError('Unexpected sample count')
    report={'split':'validation_only','device':str(device),'checkpoint_sha256':digest,'images':count,'seed':42,'images_per_class':args.per_class,'image_indices':indices,'layers':layers,'fields':list(FIELDS),'note':'Identical dense layer inputs; patch query blocks only, CLS keys retained. Descriptive block-level correlations, not independent observations or causal effects. Shared quartile cutpoints per layer; ties can produce unequal or empty bins. No masked-model propagation.','results':{}}
    for layer in layers:
        baseline=np.concatenate(data[layer,'A_25'])
        cuts={name:np.quantile(baseline[:,FIELDS.index(name)],[.25,.5,.75]) for name in ('q_similarity','coarse_entropy')}
        report['results'][str(layer)]={label:describe(np.concatenate(data[layer,label]),cuts) for label in configs}
    save_json(dest/'block_diagnostics.json',report)
    print('DIAGNOSTICS COMPLETE:',dest,flush=True)

if __name__=='__main__': main()
