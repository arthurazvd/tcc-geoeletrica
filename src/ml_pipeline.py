from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES=[f'rhoa_{i:02d}' for i in range(1,21)]
TARGETS=['rho1','rho2','rho3','h1','h2']

def prepare(df, seed=42):
    X=np.log10(df[FEATURES].to_numpy(float)); y=np.log10(df[TARGETS].to_numpy(float))
    idx=np.arange(len(df))
    train_idx, temp_idx=train_test_split(idx,test_size=.30,random_state=seed,shuffle=True)
    val_idx, test_idx=train_test_split(temp_idx,test_size=.50,random_state=seed,shuffle=True)
    sx,sy=StandardScaler(),StandardScaler()
    Xtr=sx.fit_transform(X[train_idx]); ytr=sy.fit_transform(y[train_idx])
    return {'X_train':Xtr,'y_train':ytr,'X_val':sx.transform(X[val_idx]),'y_val':sy.transform(y[val_idx]),'X_test':sx.transform(X[test_idx]),'y_test':sy.transform(y[test_idx]),'test_idx':test_idx,'sx':sx,'sy':sy}

def inverse_y(y_scaled, scaler):
    return 10**scaler.inverse_transform(y_scaled)
