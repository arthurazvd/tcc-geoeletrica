from __future__ import annotations
import argparse
from pathlib import Path
import joblib, numpy as np, matplotlib.pyplot as plt
from .models import ThreeLayerModel
from .forward_model import schlumberger_curve, default_ab2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rho1',type=float,default=100); p.add_argument('--rho2',type=float,default=30); p.add_argument('--rho3',type=float,default=250); p.add_argument('--h1',type=float,default=5); p.add_argument('--h2',type=float,default=12); p.add_argument('--du',type=float,default=5e-4); a=p.parse_args()
    b=joblib.load('models/mlp_bundle.joblib'); true=ThreeLayerModel(a.rho1,a.rho2,a.rho3,a.h1,a.h2); ab2=default_ab2(); obs=schlumberger_curve(true,ab2,du=a.du)
    X=b['scaler_x'].transform(np.log10(obs).reshape(1,-1)); pred=10**b['scaler_y'].inverse_transform(b['model'].predict(X).reshape(1,-1))[0]
    pm=ThreeLayerModel(*pred); calc=schlumberger_curve(pm,ab2,du=a.du); erms=100*np.sqrt(np.mean(((obs-calc)/obs)**2))
    print('\nPARAMETRO       REAL        MLP       ERRO (%)')
    for name,r,q in zip(['rho1','rho2','rho3','h1','h2'],[a.rho1,a.rho2,a.rho3,a.h1,a.h2],pred): print(f'{name:8s} {r:10.3f} {q:10.3f} {abs(q-r)/r*100:10.2f}')
    print(f'\nERMS da curva recalculada: {erms:.2f}%')
    Path('results/figures').mkdir(parents=True,exist_ok=True); plt.figure(figsize=(8,5)); plt.loglog(ab2,obs,'o-',label='Curva original'); plt.loglog(ab2,calc,'s--',label='Curva MLP'); plt.xlabel('AB/2 (m)'); plt.ylabel('Resistividade aparente (ohm.m)'); plt.grid(True,which='both',alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig('results/figures/prediction_demo.png',dpi=180); plt.close(); print('Grafico: results/figures/prediction_demo.png')
if __name__=='__main__': main()
