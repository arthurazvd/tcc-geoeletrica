"""Cenários sintéticos ilustrativos, NÃO dados medidos em campo.
Faixas são hipóteses escolhidas para simulação, NÃO exigidas ou estabelecidas
pela ASTM D6431 e não permitem identificação litológica unívoca.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .models import ThreeLayerModel
from .forward_model import schlumberger_curve
from .field_protocol import survey_geometry

SCENARIOS={
 'solo_umido': {'rhos':[(25,110),(8,65),(70,450)],'hs':[(2,12),(5,35)]},
 'solo_seco_rochoso': {'rhos':[(150,800),(60,450),(400,2500)],'hs':[(1.5,15),(4,45)]},
 'camada_condutiva': {'rhos':[(80,400),(5,45),(120,1000)],'hs':[(2,16),(3,40)]},
}
def generate_scenarios(n_per_scenario=100,seed=42,noise_pct=2.,du=2e-3):
 rng=np.random.default_rng(seed); geo=survey_geometry(); records=[]
 for scenario, cfg in SCENARIOS.items():
  for model_id in range(n_per_scenario):
   rho=[float(np.exp(rng.uniform(np.log(a),np.log(b)))) for a,b in cfg['rhos']]
   h=[float(rng.uniform(a,b)) for a,b in cfg['hs']]
   m=ThreeLayerModel(*rho,*h)
   clean=schlumberger_curve(m,geo['AB2_m'],du=du)
   if not np.all(np.isfinite(clean)) or np.any(clean<=0): raise ValueError('Resposta direta inválida')
   # lognormal, mantém positividade; sigma ~= coeficiente de variação para ruído pequeno.
   sigma=noise_pct/100
   observed=clean*np.exp(rng.normal(-sigma**2/2,sigma,size=len(clean)))
   for j,(ab2,mn,k,ideal,obs) in enumerate(zip(geo['AB2_m'],geo['MN_m'],geo['K_m'],clean,observed)):
    records.append(dict(scenario=scenario,model_id=f'{scenario}_{model_id:05d}',point=j+1,AB2_m=ab2,AB_m=2*ab2,MN_m=mn,K_m=k,rho_apparent_clean_ohmm=ideal,rho_apparent_observed_ohmm=obs,rho1_ohmm=rho[0],rho2_ohmm=rho[1],rho3_ohmm=rho[2],h1_m=h[0],h2_m=h[1],noise_nominal_pct=noise_pct,synthetic=True))
 return pd.DataFrame(records)

def main():
 p=argparse.ArgumentParser();p.add_argument('--n-per-scenario',type=int,default=100);p.add_argument('--noise-pct',type=float,default=2.);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='data/field_like_synthetic.csv');a=p.parse_args()
 df=generate_scenarios(a.n_per_scenario,a.seed,a.noise_pct)
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False);print(f'{len(df)} leituras sintéticas salvas em {path}')
if __name__=='__main__':main()
