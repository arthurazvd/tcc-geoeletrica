from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from .ml_pipeline import prepare, inverse_y, TARGETS

def main():
    p=argparse.ArgumentParser(description='Treina a MLP da inversao geoeletrica 1D')
    p.add_argument('--dataset',default='data/dataset_30000.csv'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-iter',type=int,default=200); a=p.parse_args()
    df=pd.read_csv(a.dataset); d=prepare(df,42)
    print(f'Dataset: {len(df)} | treino {len(d["X_train"])} | validacao {len(d["X_val"])} | teste {len(d["X_test"])}')
    model=MLPRegressor(hidden_layer_sizes=(128,128,64),activation='relu',solver='adam',learning_rate_init=1e-3,batch_size=64,max_iter=a.max_iter,early_stopping=True,validation_fraction=0.1764705882,n_iter_no_change=15,alpha=1e-4,random_state=a.seed,verbose=True)
    t=time.perf_counter(); model.fit(d['X_train'],d['y_train']); elapsed=time.perf_counter()-t
    pred=inverse_y(model.predict(d['X_test']),d['sy']); real=inverse_y(d['y_test'],d['sy'])
    errs=np.abs(pred-real)/real*100; med=np.median(errs,axis=0)
    Path('models').mkdir(exist_ok=True); Path('results/metrics').mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'scaler_x':d['sx'],'scaler_y':d['sy'],'features':[f'rhoa_{i:02d}' for i in range(1,21)],'targets':TARGETS},'models/mlp_bundle.joblib')
    out=pd.DataFrame({'parametro':TARGETS,'erro_mediano_pct':med,'erro_medio_pct':errs.mean(axis=0),'p25_pct':np.percentile(errs,25,axis=0),'p75_pct':np.percentile(errs,75,axis=0)})
    out.to_csv('results/metrics/test_metrics.csv',index=False)
    pd.DataFrame(np.c_[real,pred],columns=[f'real_{x}' for x in TARGETS]+[f'pred_{x}' for x in TARGETS]).to_csv('results/metrics/test_predictions.csv',index=False)
    json.dump({'training_seconds':elapsed,'iterations':model.n_iter_,'seed':a.seed},open('results/metrics/training_info.json','w'),indent=2)
    print('\nRESULTADOS NO TESTE'); print(out.to_string(index=False,float_format=lambda x:f'{x:.2f}')); print(f'\nTreino: {elapsed:.1f}s | epocas: {model.n_iter_}'); print('Modelo salvo em models/mlp_bundle.joblib')
if __name__=='__main__': main()
