"""V8, previsão por redes neurais com resíduo físico. Não executa TRF/LM.
python -m src.predict --rho1 100 --rho2 30 --rho3 250 --h1 5 --h2 12
python -m src.predict --csv dados.csv --outdir results/meu_csv
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .models import ThreeLayerModel
from .forward_model import default_ab2,schlumberger_curve
from .neural_core import predict_batch

def run(obs,ab2,outdir='results/minha_previsao',truth=None,steps=2,clean=None):
    obs=np.asarray(obs,float);ab2=np.asarray(ab2,float)
    if len(obs)!=20 or len(ab2)!=20 or not np.isfinite(obs).all() or (obs<=0).any():
        raise ValueError('CSV deve conter 20 valores positivos/finito de resistividade aparente.')
    if not np.allclose(ab2,default_ab2(),rtol=1e-3,atol=1e-8):
        raise ValueError('AB2_m deve corresponder aos 20 espaçamentos logarítmicos entre 1 e 200 m.')
    est=predict_batch(obs,steps=steps)[0]
    calc=schlumberger_curve(ThreeLayerModel(*est),ab2)
    rms_obs=float(100*np.sqrt(np.mean(((calc-obs)/obs)**2)))
    report={'metodo':'V8: base neural + V7 residual + V8 residual físico, 2 passagens por padrão',
            'sem_TRF_ou_LM':True,'numero_passagens_neurais':steps,'ERMS_vs_observed_pct':rms_obs,
            'estimativa':dict(zip(('rho1_ohmm','rho2_ohmm','rho3_ohmm','h1_m','h2_m'),map(float,est)))}
    if truth is not None:
        ref=clean if clean is not None else schlumberger_curve(ThreeLayerModel(*truth),ab2)
        report['ERMS_vs_clean_reference_pct']=float(100*np.sqrt(np.mean(((calc-ref)/ref)**2)))
        report['parameter_relative_errors_pct']=dict(zip(('rho1','rho2','rho3','h1','h2'),map(float,100*np.abs(est-truth)/truth)))
    d=Path(outdir);d.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'AB2_m':ab2,'observado_ohmm':obs,'reconstruido_v8_ohmm':calc,**({'referencia_limpa_ohmm':clean} if clean is not None else {})}).to_csv(d/'curvas.csv',index=False)
    (d/'metricas.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(9,7.2),sharex=True,gridspec_kw={'height_ratios':[2.5,1]})
    ax.loglog(ab2,obs,'ko',label='Entrada observada',ms=5)
    if clean is not None:ax.loglog(ab2,clean,'--',lw=1.5,label='Referência teórica limpa')
    ax.loglog(ab2,calc,lw=2.2,label=f'V8 — ERMS observada {rms_obs:.2f}%')
    ax.set(ylabel='Resistividade aparente (Ω·m)',title='Inversão geoelétrica 1D — V8 (correção neural)');ax.grid(True,which='both',alpha=.2);ax.legend()
    ax2.semilogx(ab2,100*(calc-obs)/obs,'o-',ms=3);ax2.axhline(0,color='gray',ls='--',lw=.7)
    ax2.set(xlabel='AB/2 (m)',ylabel='Resíduo (%)');ax2.grid(True,which='both',alpha=.2)
    fig.tight_layout();fig.savefig(d/'curva_v8.png',dpi=180);plt.close(fig)
    return report

def main():
    p=argparse.ArgumentParser(description='Previsão geoelétrica V8 sem TRF/LM')
    for a in ('rho1','rho2','rho3','h1','h2'):p.add_argument('--'+a,type=float)
    p.add_argument('--csv');p.add_argument('--noise-pct',type=float,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--outdir',default='results/minha_previsao');p.add_argument('--steps',type=int,default=2,choices=(1,2,3));a=p.parse_args()
    if a.csv:
        df=pd.read_csv(a.csv)
        if not {'AB2_m','rho_apparent_ohmm'}.issubset(df.columns):p.error('CSV precisa de AB2_m,rho_apparent_ohmm')
        obs=df['rho_apparent_ohmm'].to_numpy(float);ab2=df['AB2_m'].to_numpy(float);clean=None;truth=None
        if a.noise_pct!=0:p.error('Com --csv o ruído já está nos dados: não use --noise-pct')
    else:
        truth=np.array([getattr(a,v) for v in ('rho1','rho2','rho3','h1','h2')],dtype=float)
        if not np.isfinite(truth).all() or (truth<=0).any():p.error('Informe cinco parâmetros positivos ou --csv')
        ab2=default_ab2();clean=schlumberger_curve(ThreeLayerModel(*truth),ab2);sigma=a.noise_pct/100
        if sigma<0:p.error('--noise-pct não pode ser negativo')
        obs=clean*np.exp(np.random.default_rng(a.seed).normal(-sigma**2/2,sigma,size=20))
    report=run(obs,ab2,a.outdir,truth,a.steps,clean)
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
