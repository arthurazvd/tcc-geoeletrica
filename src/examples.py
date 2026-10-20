"""Cria exemplos executáveis de 4 terrenos, com e sem 2% de ruído."""
import json
from .experiment_utils import sample_groups
from .models import ThreeLayerModel
from .forward_model import default_ab2,schlumberger_curve
from .predict import run
from .neural_core import ROOT
import numpy as np,pandas as pd

def main():
    rng=np.random.default_rng(805)
    cases={'ilustrativo_H':np.array([100,30,250,5,12],dtype=float)}
    for key,values in sample_groups(1,seed=30202).items():cases[key]=values[0]
    table=[];dest=ROOT/'results/exemplos';dest.mkdir(parents=True,exist_ok=True)
    for j,(key,truth) in enumerate(cases.items()):
        clean=schlumberger_curve(ThreeLayerModel(*truth),default_ab2())
        for sigma,suffix in [(0.,'sem_ruido'),(.02,'ruido_2pct')]:
            obs=clean*np.exp(rng.normal(-sigma**2/2,sigma,20));target=dest/key/suffix
            metrics=run(obs,default_ab2(),target,truth=truth,steps=2,clean=clean)
            table.append({'cenario':key,'regime':suffix,**{k:float(v) for k,v in zip(['rho1','rho2','rho3','h1','h2'],truth)},'ERMS_vs_observed_pct':metrics['ERMS_vs_observed_pct'],'ERMS_vs_clean_reference_pct':metrics['ERMS_vs_clean_reference_pct']})
            if key=='ilustrativo_H':
                pd.DataFrame({'AB2_m':default_ab2(),'rho_apparent_ohmm':obs}).to_csv(dest/('entrada_'+suffix+'.csv'),index=False)
    pd.DataFrame(table).to_csv(dest/'resumo_exemplos.csv',index=False)
    print(pd.DataFrame(table).to_string(index=False))
if __name__=='__main__':main()
