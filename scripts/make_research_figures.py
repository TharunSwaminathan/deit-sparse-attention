"""Generate six validation-only research figures from checked-in V4 outputs.
Run from any directory: python scripts/make_research_figures.py [--root PATH].
No inference, checkpoint, or dataset needed. Requires matplotlib and numpy.
"""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

LABELS=['dense','unmasked']+[f'{v}_{s}' for s in (25,50) for v in 'ABCD']
SPARSE=LABELS[2:]
COLORS=dict(A='#0072B2', B='#D55E00', C='#009E73', D='#CC79A7')
LAYERS=['3','6','9']

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); root=args.root; out=root/'figures'; out.mkdir(exist_ok=True)
    val={k:load(root/'outputs'/'validation_v4'/f'{k}.json') for k in LABELS}
    gpu=load(root/'outputs'/'gpu_latency_v4.json'); diag=load(root/'outputs'/'block_diagnostics.json')
    calibration=load(root/'outputs'/'calibration_64_v4.json')
    assert all(v['split']=='val' and v['scores']['n']==736 for v in val.values())
    assert gpu['gpu']=='Tesla T4' and gpu['layers']==calibration['layer_indices']==diag['layers']==[3,6,9]
    assert gpu['checkpoint_sha256']==diag['checkpoint_sha256']==calibration['checkpoint_sha256']
    assert diag['split']=='validation_only' and diag['images']==111
    times={r['label']:r['median_ms'] for r in gpu['results']}
    assert set(times)==set(LABELS)
    assert all(r['repetitions']==30 and r['timing_batch_size']==1 and r['precision']=='fp32' for r in gpu['results'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white','axes.axisbelow':True})
    def save(fig, name):
        for ext in ('png','pdf'):
            fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.12)
        plt.close(fig)
    def grid(ax):
        ax.grid(axis='y',color='#DCE2E8',lw=.7)
    def note(fig,s):
        fig.text(.06,.015,s,fontsize=8,color='#455261',ha='left')
    # 1: editable vector schematic of implementation; schematic, not observed masks.
    f,ax=plt.subplots(figsize=(10.5,5.8)); ax.set(xlim=(0,10.5),ylim=(0,5.8));ax.axis('off')
    def box(x,y,w,h,txt,fc='#F0F4F8',fs=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.08',fc=fc,ec='#57697B',lw=1.1)); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs)
    def arrow(x,y,x2,y2):
        ax.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1.15,color='#57697B'))
    ax.text(.1,5.55,'Block pooling predicts a token-level attention mask',fontsize=16,weight='bold')
    ax.text(.1,5.24,'DeiT-Tiny • 12 Transformer blocks • only blocks 3, 6, 9 wrapped (zero-indexed)',fontsize=10,color='#455261')
    # token grid and coarse spatial grid
    for i in range(14):
        for j in range(14):
            ax.add_patch(Rectangle((.25+j*.095,3.52+i*.095),.095,.095,fc='#CFE3EF' if ((i//2+j//2)%2==0) else '#EAF1F5',ec='white',lw=.3))
    ax.text(.915,3.34,'14 × 14 patches',ha='center',fontsize=9)
    arrow(1.66,4.18,2.17,4.18)
    for i in range(7):
        for j in range(7):
            ax.add_patch(Rectangle((2.25+j*.17,3.59+i*.17),.17,.17,fc='#CFE3EF' if ((i+j)%2==0) else '#EAF1F5',ec='#FFFFFF',lw=.55))
    ax.text(2.845,3.34,'7 × 7 blocks',ha='center',fontsize=9)
    ax.text(1.93,4.45,'2 × 2\nmean Q/K',ha='center',fontsize=8)
    arrow(3.52,4.18,3.95,4.18)
    box(4,3.68,1.7,1.0,'Coarse scores\nsoftmax\n49 × 49 per head')
    arrow(5.75,4.18,6.1,4.18)
    box(6.15,3.4,2.35,1.55,'A: Top-p\nB: A + protection\nC: A ∪ Top-k\nD: C + protection','#EDF6F1')
    arrow(8.55,4.18,8.9,4.18)
    box(8.95,3.68,1.4,1.0,'Expand mask\n197 × 197\nkeep CLS edges')
    # bottom dense branch
    box(.25,1.55,2.1,.85,'Original token Q, K, V\n197 tokens / head')
    arrow(2.4,1.98,3.0,1.98); box(3.05,1.55,1.75,.85,'Dense QKᵀ scores\n197 × 197')
    arrow(4.85,1.98,5.5,1.98);box(5.55,1.55,1.7,.85,'Apply mask\n(dropped = −∞)','#F9F0E4')
    arrow(7.3,1.98,7.8,1.98);box(7.85,1.55,2.5,.85,'Softmax × V (dense)\nthen output projection')
    ax.plot([9.65,9.65,6.4],[3.65,2.95,2.95],color='#57697B',lw=1.15);arrow(6.4,2.95,6.4,2.43)
    ax.text(.25,.94,'Protection: low mean within-block Q/K cosine similarity keeps entire coarse rows/columns.',fontsize=10)
    ax.text(.25,.57,'Dense score and value products remain: this is a masking prototype, not a sparse CUDA kernel.',fontsize=10,weight='bold',color='#8A4814')
    save(f,'01_attention_architecture')
    # 2 and 3: two budgets, same axes, label by method using colored marker legend
    legend=[Line2D([0],[0],marker='o',color=COLORS[v],lw=0,markersize=7,label=v) for v in 'ABCD']
    for metric,name,title,ylabel in [('accuracy','02_accuracy_vs_sparsity','Validation accuracy at matched sparsity budgets','Accuracy (%)'),('retained_mass','03_retained_mass_vs_sparsity','Retained attention mass at matched sparsity budgets','Retained dense attention mass (%)')]:
        f,axs=plt.subplots(1,2,figsize=(8.2,4.05),sharey=True);f.subplots_adjust(left=.11,right=.98,bottom=.33,top=.78,wspace=.14)
        f.suptitle(title,fontsize=14,weight='bold',y=.98)
        for ax,s in zip(axs,(25,50)):
            ax.axvspan(s-3,s+3,color='#E9EFF3',zorder=0);ax.axvline(s,color='#8B9BA7',lw=.9,ls=':');grid(ax)
            for v in 'ABCD':
                k=f'{v}_{s}';d=val[k];x=100*d['diagnostics']['realized_sparsity'];y=100*(d['scores'][metric] if metric=='accuracy' else d['diagnostics'][metric]); ax.scatter(x,y,s=65,color=COLORS[v],edgecolor='white',linewidth=.7,zorder=3)
            if metric=='accuracy':
                ax.axhline(100*val['dense']['scores']['accuracy'],color='#333333',ls='--',lw=1.1);ax.set_ylim(93.35,94.25)
                ax.set_yticks([93.4,93.6,93.8,94,94.2])
            else: ax.set_ylim(77,101);ax.set_yticks([80,85,90,95,100])
            ax.set_xlim(s-4,s+4);ax.set_xticks([s-3,s,s+3]);ax.set_title(f'{s}% target');ax.set_xlabel('Realized sparsity (%)')
        axs[0].set_ylabel(ylabel)
        handles=legend+([Line2D([0],[0],ls='--',color='#333333',label='Dense / unmasked: 93.75%')] if metric=='accuracy' else [])
        f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),ncol=len(handles),frameon=False,fontsize=9)
        note(f,'736 validation images; shading = target ±3 percentage points. Selected layers only. '+('Zoomed accuracy axis; no uncertainty intervals.' if metric=='accuracy' else 'Mass computed before masking.'))
        save(f,name)
    # 4: GPU timings, zero-origin horizontal bars.
    f,ax=plt.subplots(figsize=(7.7,5.3));f.subplots_adjust(left=.15,right=.86,bottom=.14,top=.86)
    ax.set_title('Sparse masks add GPU inference overhead',loc='left',weight='bold',fontsize=14,pad=13)
    yy=np.arange(len(LABELS));ax.barh(yy,[times[k] for k in LABELS],color=['#455261','#A5B3C1']+[COLORS[k[0]] for k in SPARSE],height=.67)
    for i,k in enumerate(LABELS):ax.text(times[k]+.15,i,f'{times[k]:.2f}',va='center',fontsize=9)
    ax.set_yticks(yy,LABELS);ax.invert_yaxis();ax.set_xlim(0,12.6);ax.set_xlabel('Median forward latency (ms)');ax.axvline(times['dense'],ls='--',lw=1,color='#333333');ax.grid(axis='x',lw=.7,color='#DCE2E8')
    note(f,'Tesla T4; FP32, batch 1, 10 warmups / 30 measured forwards; synthetic GPU-resident inputs.')
    save(f,'04_t4_latency')
    # 5: correlation heatmaps, same centered color scale.
    f,axs=plt.subplots(1,2,figsize=(8.3,5));f.subplots_adjust(left=.12,right=.88,bottom=.12,top=.80,wspace=.30)
    f.suptitle('Approximation-error correlations depend on layer and rule',weight='bold',fontsize=14,y=.98)
    for ax,key,title in zip(axs,['coarse_entropy','q_similarity'],['Coarse attention entropy','Within-query-block similarity']):
        m=np.array([[diag['results'][l][k]['strata'][key]['spearman_rho_error'] for l in LAYERS] for k in SPARSE]);im=ax.imshow(m,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto')
        for i in range(8):
            for j in range(3):ax.text(j,i,f'{m[i,j]:+.2f}',ha='center',va='center',color='white' if abs(m[i,j])>.62 else '#18232D',fontsize=9)
        ax.set_xticks(range(3),LAYERS);ax.set_yticks(range(8),SPARSE);ax.set_xlabel('Layer (zero-indexed)');ax.set_title(title,fontsize=11,pad=9)
        for sp in ax.spines.values():sp.set_visible(False)
    cax=f.add_axes([.91,.18,.018,.57]);f.colorbar(im,cax=cax,label='Spearman ρ')
    note(f,'111 validation images; identical dense layer inputs; descriptive block-level correlations.')
    save(f,'05_error_correlations')
    # 6: all layer/pair comparisons, error and realized sparsity in Q1.
    f,axs=plt.subplots(3,2,figsize=(9,8.3),sharex=True,sharey='col');f.subplots_adjust(left=.13,right=.98,bottom=.13,top=.87,wspace=.24,hspace=.25)
    f.suptitle('Protection reallocates sparsity in low-similarity query blocks',fontsize=14,weight='bold',y=.98)
    pairs=[('A_25','B_25'),('C_25','D_25'),('A_50','B_50'),('C_50','D_50')];x=np.arange(4);w=.34
    for i,l in enumerate(LAYERS):
        for j,(key,title) in enumerate([('relative_output_error','Relative output error (%)'),('realized_sparsity','Local realized sparsity (%)')]):
            ax=axs[i,j]
            a,b=([100*diag['results'][l][p[t]]['strata']['q_similarity']['bins'][0]['means'][key] for p in pairs] for t in (0,1))
            ax.bar(x-w/2,a,w,color='#0072B2',label='Unprotected');ax.bar(x+w/2,b,w,color='#D55E00',label='Protected');grid(ax)
            if i==0:ax.set_title(title,fontsize=11)
            if j==0:ax.set_ylabel(f'Layer {l}')
            ax.set_ylim(0,34 if j==0 else 73)
            if i==2:ax.set_xticks(x,['A/B 25','C/D 25','A/B 50','C/D 50'])
    f.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.052),ncol=2,frameon=False)
    note(f,'Lowest query-similarity quartile per layer; identical dense inputs. Parameters and local budgets differ.')
    save(f,'06_low_similarity_protection')
    print(f'Generated six PNG and six PDF figures in {out}')

if __name__=='__main__': main()
