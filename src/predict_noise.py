from __future__ import annotations
import argparse
import joblib, numpy as np
from .forward_model import schlumberger_curve, default_ab2
from .models import ThreeLayerModel

FEATURES=[f'rhoa_{i:02d}' for i in range(1,21)]

def main():
    p=argparse.ArgumentParser(description='Testa os modelos 0/2/5% no mesmo terreno sintético')
    p.add_argument('--rho1',type=float,default=100); p.add_argument('--rho2',type=float,default=30); p.add_argument('--rho3',type=float,default=250)
    p.add_argument('--h1',type=float,default=5); p.add_argument('--h2',type=float,default=12); p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args(); real=np.array([a.rho1,a.rho2,a.rho3,a.h1,a.h2],float)
    m=ThreeLayerModel(*real); clean=schlumberger_curve(m,default_ab2())
    print('REAL:',dict(zip(['rho1','rho2','rho3','h1','h2'],real)))
    for pct in (0,2,5):
        b=joblib.load(f'models/noise/mlp_noise_{pct}.joblib')
        rng=np.random.default_rng(a.seed+pct); obs=clean.copy() if pct==0 else clean*(1+(pct/100)*rng.normal(size=clean.shape))
        X=b['scaler_x'].transform(np.log10(obs.reshape(1,-1)))
        pred=10**b['scaler_y'].inverse_transform(b['model'].predict(X).reshape(1,-1))[0]
        err=np.abs(pred-real)/real*100
        print(f'\nRuido {pct}%')
        for name,r,pr,e in zip(b['targets'],real,pred,err): print(f'  {name:4s}: real={r:9.3f} | MLP={pr:9.3f} | erro={e:6.2f}%')
if __name__=='__main__': main()
