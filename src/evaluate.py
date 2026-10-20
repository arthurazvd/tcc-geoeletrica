"""Reproduz avaliação reservada para os checkpoints fixos V8.
Atenção: o holdout pode reproduzir a métrica reportada, mas não deve servir
para selecionar novos hiperparâmetros.
"""
import argparse,json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment_utils import sample_groups,forward_batch,summarize
from .neural_core import predict_batch,ROOT

def evaluate(n=400,seed=99103):
    groups=sample_groups(n,seed);report={}
    for gi,(name,params) in enumerate(groups.items()):
        true=np.concatenate([forward_batch(params[j:j+60]) for j in range(0,n,60)])
        report[name]={}
        for noise,label in ((0.,'clean'),(.02,'noise_2pct')):
            rng=np.random.default_rng(seed+300*gi+int(noise*10000)+17)
            obs=true*np.exp(rng.normal(-noise**2/2,noise,true.shape))
            pred=predict_batch(obs,steps=2)
            modeled=np.concatenate([forward_batch(pred[j:j+50]) for j in range(0,n,50)])
            report[name][label]=summarize(modeled,true,pred,params)
    return {'seed':seed,'n_per_scenario':n,'total_distinct_models':4*n,
            'reference':'clean theoretical forward curves for both noise regimes',
            'method':'V8 neural physical-residual correction (2 passes), no TRF or LM',
            'noise_std_pct':2,'results':report}

def plot(rep,dest):
    names=list(rep['results']);labels=['Original','Solo úmido','Seco/rochoso','Condutivo']
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(4)
    for k,(regime,label) in enumerate([('clean','Sem ruído'),('noise_2pct','Ruído 2%')]):
        v=[rep['results'][n][regime]['median_pct'] for n in names]
        bars=ax.bar(x+(-.18 if k==0 else .18),v,.35,label=label)
        for bar,value in zip(bars,v):ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.2f}%',ha='center',fontsize=8)
    ax.axhline(1,ls='--',color='gray',lw=1,label='Meta 1%')
    ax.set_xticks(x,labels);ax.set_ylim(0,2.45);ax.set_ylabel('ERMS mediano (%)')
    ax.set_title('V8 — erro por cenário e condição de ruído');ax.grid(axis='y',alpha=.2);ax.legend();fig.tight_layout();fig.savefig(dest,dpi=190);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=400);ap.add_argument('--seed',type=int,default=99103)
    a=ap.parse_args();rep=evaluate(a.n,a.seed)
    out=ROOT/'results/metricas';out.mkdir(parents=True,exist_ok=True)
    (out/'avaliacao_reproduzida.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf8')
    plot(rep,ROOT/'results/figuras/avaliacao_v8.png')
    for n,r in rep['results'].items():print(n,'limpo=',round(r['clean']['median_pct'],4),'ruido=',round(r['noise_2pct']['median_pct'],4))
if __name__=='__main__':main()
