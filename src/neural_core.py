"""Arquiteturas da V8: MLP base, residual V7 e residual físico V8.
A inferência usa pesos fixos; duas aplicações da V8 são correções neurais,
não passos de otimização por gradiente.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import j1
from .forward_model import default_ab2
ROOT=Path(__file__).resolve().parents[1]

def make_base(width=96):
    return nn.Sequential(nn.Linear(20,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,5))

class ResidualV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(25,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,5))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,z):return z+0.45*torch.tanh(self.net(torch.cat((x,z),dim=1)))

class ResidualV8(nn.Module):
    def __init__(self,width=160):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(45,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width//2),nn.GELU(),nn.Linear(width//2,5))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,z,residual):return z+0.45*torch.tanh(self.head(torch.cat([x,z,residual],dim=-1)))

class PhysicalForward(nn.Module):
    """Operador diferenciável na grade fixa AB/2=1..200 m; entrada log10(params)."""
    def __init__(self,du=.001):
        super().__init__()
        u=np.arange(np.log(1e-4),np.log(12.)+du,du);lam=np.exp(u);s=default_ab2()
        w=np.ones(len(u));w[[0,-1]]=.5
        kernel=(j1(lam[:,None]*s[None,:])*(lam[:,None]**2)*w[:,None]*du)*(s[None,:]**2)
        self.register_buffer('lam',torch.tensor(lam,dtype=torch.float32))
        self.register_buffer('kernel',torch.tensor(kernel,dtype=torch.float32))
    def forward(self,logparams):
        pp=torch.pow(10.,logparams);r1,r2,r3,h1,h2=[pp[:,i:i+1] for i in range(5)]
        th=torch.tanh(h2*self.lam);t=(r3+r2*th)/(1+(r3/r2)*th)
        th=torch.tanh(h1*self.lam);t=(t+r1*th)/(1+(t/r1)*th)
        return r1+(t-r1)@self.kernel

def load_weights():
    ck=torch.load(ROOT/'models/neural_v5.pt',map_location='cpu',weights_only=False)
    base=make_base(ck['width']);base.load_state_dict(ck['weights']);base.eval()
    stage7=ResidualV7();stage7.load_state_dict(torch.load(ROOT/'models/neural_v7_residual.pt',map_location='cpu',weights_only=False)['weights']);stage7.eval()
    ck8=torch.load(ROOT/'models/neural_v8_physics_correction.pt',map_location='cpu',weights_only=False)
    stage8=ResidualV8(ck8['width']);stage8.load_state_dict(ck8['weights']);stage8.eval()
    return ck,base,stage7,stage8

def predict_batch(obs,steps=2,include_stages=False):
    """Retorna estimativas de três camadas, sem TRF/LM e sem backprop no uso."""
    arr=np.asarray(obs,dtype=float)
    if arr.ndim==1:arr=arr[None,:]
    if arr.ndim!=2 or arr.shape[1]!=20 or not np.isfinite(arr).all() or (arr<=0).any():
        raise ValueError('Esperados 20 valores positivos e finitos de resistividade aparente por curva.')
    if steps not in (1,2,3):raise ValueError('steps deve ser 1, 2 ou 3')
    torch.set_num_threads(2)
    ck,base,stage7,stage8=load_weights();phys=PhysicalForward(du=.001)
    log=torch.as_tensor(np.log10(arr),dtype=torch.float32)
    meanx=torch.as_tensor(ck['sx_mean'],dtype=torch.float32);scalex=torch.as_tensor(ck['sx_scale'],dtype=torch.float32)
    meany=torch.as_tensor(ck['sy_mean'],dtype=torch.float32);scaley=torch.as_tensor(ck['sy_scale'],dtype=torch.float32)
    results=[];initial=[]
    with torch.no_grad():
        for chunk in log.split(80):
            x=(chunk-meanx)/scalex; z=stage7(x,base(x));z7=z.clone()
            for _ in range(steps):
                curve=phys(z*scaley+meany).clamp_min(1e-5)
                residual=(chunk-torch.log10(curve))/scalex
                z=stage8(x,z,residual)
            results.append(torch.pow(10.,z*scaley+meany).numpy())
            if include_stages:initial.append(torch.pow(10.,z7*scaley+meany).numpy())
    end=np.concatenate(results)
    return (end,np.concatenate(initial)) if include_stages else end
