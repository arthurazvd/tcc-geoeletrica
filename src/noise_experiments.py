from __future__ import annotations
import argparse, copy, json, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .ml_pipeline import FEATURES, TARGETS
from .forward_model import schlumberger_curve, default_ab2
from .models import ThreeLayerModel


def split_indices(n: int, seed: int = 42):
    idx=np.arange(n)
    train_idx,temp_idx=train_test_split(idx,test_size=.30,random_state=seed,shuffle=True)
    val_idx,test_idx=train_test_split(temp_idx,test_size=.50,random_state=seed,shuffle=True)
    return train_idx,val_idx,test_idx


def add_noise(X: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    if sigma == 0:
        return X.copy()
    rng=np.random.default_rng(seed)
    noisy=X*(1.0 + sigma*rng.normal(size=X.shape))
    return np.clip(noisy, np.finfo(float).tiny, None)


def prepare_level(df, sigma, train_idx, val_idx, test_idx, noise_seed):
    X_clean=df[FEATURES].to_numpy(float)
    y=np.log10(df[TARGETS].to_numpy(float))
    X=add_noise(X_clean,sigma,noise_seed)
    Xlog=np.log10(X)
    sx,sy=StandardScaler(),StandardScaler()
    Xtr=sx.fit_transform(Xlog[train_idx]); ytr=sy.fit_transform(y[train_idx])
    return {
        'X_train':Xtr,'y_train':ytr,
        'X_val':sx.transform(Xlog[val_idx]),'y_val':sy.transform(y[val_idx]),
        'X_test':sx.transform(Xlog[test_idx]),'y_test':sy.transform(y[test_idx]),
        'X_test_observed':X[test_idx], 'real_test':10**y[test_idx],
        'sx':sx,'sy':sy
    }


def train_external_validation(d, seed, max_epochs=200, patience=15):
    model=MLPRegressor(hidden_layer_sizes=(128,128,64),activation='relu',solver='adam',
        learning_rate_init=1e-3,batch_size=64,max_iter=1,warm_start=True,
        early_stopping=False,alpha=1e-4,random_state=seed,shuffle=True)
    best=None; best_loss=np.inf; stale=0; history=[]
    start=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        for epoch in range(1,max_epochs+1):
            model.fit(d['X_train'],d['y_train'])
            val_pred=model.predict(d['X_val'])
            val_loss=mean_squared_error(d['y_val'],val_pred)/2.0
            history.append(val_loss)
            if val_loss < best_loss - 1e-8:
                best_loss=val_loss; best=copy.deepcopy(model); stale=0
            else:
                stale+=1
            if epoch==1 or epoch%10==0 or stale>=patience:
                print(f'  epoca {epoch:3d} | perda validacao {val_loss:.6f} | melhor {best_loss:.6f} | paciencia {stale}/{patience}')
            if stale>=patience:
                break
    return best, history, time.perf_counter()-start


def parameter_metrics(real,pred):
    errs=np.abs(pred-real)/real*100
    return pd.DataFrame({
        'parametro':TARGETS,
        'erro_mediano_pct':np.median(errs,axis=0),
        'erro_medio_pct':np.mean(errs,axis=0),
        'p25_pct':np.percentile(errs,25,axis=0),
        'p75_pct':np.percentile(errs,75,axis=0),
    }), errs


def curve_erms(observed, pred_params, du=0.002):
    ab2=default_ab2()
    vals=[]
    for obs,p in zip(observed,pred_params):
        m=ThreeLayerModel(rho1=p[0],rho2=p[1],rho3=p[2],h1=p[3],h2=p[4])
        calc=schlumberger_curve(m,ab2,du=du)
        vals.append(100*np.sqrt(np.mean(((obs-calc)/obs)**2)))
    return np.asarray(vals)


def main():
    ap=argparse.ArgumentParser(description='Etapa 3: treina e compara MLPs com 0%, 2% e 5% de ruido.')
    ap.add_argument('--dataset',default='data/dataset_30000.csv')
    ap.add_argument('--seed',type=int,default=0,help='semente da MLP')
    ap.add_argument('--split-seed',type=int,default=42)
    ap.add_argument('--noise-seed',type=int,default=12345)
    ap.add_argument('--max-epochs',type=int,default=200)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--curve-metrics',action='store_true',help='recalcula curvas do teste para medir ERMS (mais lento)')
    ap.add_argument('--du',type=float,default=0.002,help='passo da integracao para ERMS; use 0.0005 na reproducao final')
    a=ap.parse_args()

    df=pd.read_csv(a.dataset)
    tr,va,te=split_indices(len(df),a.split_seed)
    print(f'Dataset: {len(df)} | treino {len(tr)} | validacao {len(va)} | teste {len(te)}')
    Path('models/noise').mkdir(parents=True,exist_ok=True)
    Path('results/noise').mkdir(parents=True,exist_ok=True)
    Path('results/figures').mkdir(parents=True,exist_ok=True)
    summary=[]

    for pct in (0,2,5):
        sigma=pct/100
        print(f'\n=== RUIDO {pct}% ===')
        d=prepare_level(df,sigma,tr,va,te,a.noise_seed+pct)
        model,hist,seconds=train_external_validation(d,a.seed,a.max_epochs,a.patience)
        pred=10**d['sy'].inverse_transform(model.predict(d['X_test']))
        real=d['real_test']
        metrics,errs=parameter_metrics(real,pred)
        metrics.insert(0,'ruido_pct',pct)
        erms=None
        if a.curve_metrics:
            print('  recalculando 4.500 curvas para ERMS...')
            erms=curve_erms(d['X_test_observed'],pred,a.du)
            print(f'  ERMS mediano: {np.median(erms):.2f}%')
        for _,r in metrics.iterrows(): summary.append(r.to_dict())
        bundle={'model':model,'scaler_x':d['sx'],'scaler_y':d['sy'],'features':FEATURES,'targets':TARGETS,
                'noise_pct':pct,'noise_seed':a.noise_seed+pct,'split_seed':a.split_seed}
        joblib.dump(bundle,f'models/noise/mlp_noise_{pct}.joblib')
        metrics.to_csv(f'results/noise/metrics_noise_{pct}.csv',index=False)
        pred_df=pd.DataFrame(np.c_[real,pred],columns=[f'real_{x}' for x in TARGETS]+[f'pred_{x}' for x in TARGETS])
        if erms is not None: pred_df['erms_pct']=erms
        pred_df.to_csv(f'results/noise/predictions_noise_{pct}.csv',index=False)
        pd.DataFrame({'epoch':np.arange(1,len(hist)+1),'validation_loss':hist}).to_csv(f'results/noise/history_noise_{pct}.csv',index=False)
        json.dump({'noise_pct':pct,'training_seconds':seconds,'epochs':len(hist),'best_validation_loss':float(min(hist)),
                   'seed':a.seed,'split_seed':a.split_seed,'noise_seed':a.noise_seed+pct,
                   'median_erms_pct':None if erms is None else float(np.median(erms))},
                  open(f'results/noise/info_noise_{pct}.json','w'),indent=2)
        print(metrics[['parametro','erro_mediano_pct']].to_string(index=False,float_format=lambda x:f'{x:.2f}'))
        print(f'  treino: {seconds:.1f}s | epocas: {len(hist)}')

    summary_df=pd.DataFrame(summary)
    summary_df.to_csv('results/noise/summary_parameter_metrics.csv',index=False)
    pivot=summary_df.pivot(index='parametro',columns='ruido_pct',values='erro_mediano_pct').reindex(TARGETS)
    print('\n=== RESUMO: ERRO MEDIANO (%) ===')
    print(pivot.to_string(float_format=lambda x:f'{x:.2f}'))

    ax=pivot.T.plot(marker='o')
    ax.set_xlabel('Ruido (%)'); ax.set_ylabel('Erro relativo mediano (%)'); ax.set_title('Robustez da MLP ao ruido')
    ax.grid(True,which='both',alpha=.25); plt.tight_layout(); plt.savefig('results/figures/noise_parameter_errors.png',dpi=180); plt.close()
    print('\nModelos: models/noise/')
    print('Resultados: results/noise/')
    print('Grafico: results/figures/noise_parameter_errors.png')

if __name__=='__main__': main()
