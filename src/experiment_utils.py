"""Gerador dos quatro cenários e operador direto em lote da avaliação V8."""
import numpy as np
from scipy.special import j1
from .forward_model import default_ab2
from .realistic_data import SCENARIOS

def sample_groups(n,seed):
    rng=np.random.default_rng(seed);out={}
    out['original']=np.column_stack([10**rng.uniform(1,3,(n,3)),rng.uniform(1,20,(n,1)),rng.uniform(2,50,(n,1))])
    for name,c in SCENARIOS.items():
        r=np.column_stack([np.exp(rng.uniform(np.log(a),np.log(b),n)) for a,b in c['rhos']])
        h=np.column_stack([rng.uniform(a,b,n) for a,b in c['hs']]);out[name]=np.column_stack([r,h])
    return out

def forward_batch(params,ab2=None,du=.002):
    if ab2 is None:ab2=default_ab2()
    u=np.arange(np.log(1e-4),np.log(12.)+du,du);l=np.exp(u)
    c=np.asarray(params,float);t=np.broadcast_to(c[:,2,None],(len(c),len(l))).copy()
    for ix,hx in ((1,4),(0,3)):
        r=c[:,ix,None];th=np.tanh(l[None,:]*c[:,hx,None]);t=(t+r*th)/(1+t/r*th)
    delta=t-c[:,0,None]
    weights=np.ones(len(l));weights[[0,-1]]=.5
    kern=j1(l[:,None]*ab2[None,:])*(l[:,None]**2)*weights[:,None]*du
    return c[:,0,None]+(delta@kern)*(ab2[None,:]**2)

def summarize(curves,ref,params,truth):
    e=100*np.sqrt(np.mean(((curves-ref)/ref)**2,axis=1))
    return {'median_pct':float(np.median(e)),'mean_pct':float(np.mean(e)),
            'p90_pct':float(np.quantile(e,.9)),'below_1pct_fraction':float(np.mean(e<1)),
            'below_2pct_fraction':float(np.mean(e<2)),
            'parameter_median_relative_pct':dict(zip(['rho1','rho2','rho3','h1','h2'],map(float,np.median(100*abs(params-truth)/truth,axis=0))))}
