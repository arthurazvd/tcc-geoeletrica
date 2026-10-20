"""V8: learned physics-informed correction of frozen V7, with no optimizer at inference.
The trained correction uses the *forward residual* (computed once) as an additional
feature: this is deterministic physical evaluation, NOT numerical inversion.
Reports clean vs 2% noisy performance on independent scenario datasets.
"""
from __future__ import annotations
import argparse, json, time, copy
import numpy as np
import torch
from torch import nn
from .neural_core import ROOT, make_base as net, ResidualV7 as Correction, ResidualV8 as PhysicsCorrection, PhysicalForward
from .experiment_utils import sample_groups, forward_batch
from .forward_model import default_ab2

def forward_groups(groups,batch=70):
    ps=np.concatenate(list(groups.values()))
    curves=np.concatenate([forward_batch(ps[i:i+batch],default_ab2()) for i in range(0,len(ps),batch)])
    return ps,curves

def metric(curves,clean):
    er=100*np.sqrt(np.mean(((curves-clean)/clean)**2,axis=1))
    return {'median_pct':float(np.median(er)),'mean_pct':float(np.mean(er)),
            'p90_pct':float(np.quantile(er,.9)),'below_1pct_fraction':float(np.mean(er<1)),
            'below_2pct_fraction':float(np.mean(er<2))}

def load_base():
    ck=torch.load(ROOT/'models/neural_v5.pt',map_location='cpu',weights_only=False)
    base=net(ck['width']);base.load_state_dict(ck['weights']);base.eval()
    v7=Correction();v7.load_state_dict(torch.load(ROOT/'models/neural_v7_residual.pt',map_location='cpu',weights_only=False)['weights']);v7.eval()
    for m in (base,v7):
        for p in m.parameters():p.requires_grad_(False)
    return ck,base,v7

def predict_normalized(x,ck,base,v7,refiner,physics,sxm,sxs,sym,sys):
    nx=(x-sxm)/sxs
    with torch.no_grad():
        v5=base(nx);z=v7(nx,v5)
        predcurve=physics(z*sys+sym).clamp_min(1e-4)
        diff=(x-torch.log10(predcurve))/sxs
    return refiner(nx,z,diff),z

def evaluate(models, ck, group_n=300,seed=88017):
    groups=sample_groups(group_n,seed)
    rng=np.random.default_rng(seed+100)
    result={};
    for gi,(name,params) in enumerate(groups.items()):
        true=np.concatenate([forward_batch(params[i:i+50],default_ab2()) for i in range(0,len(params),50)])
        result[name]={}
        for sigma,label in [(0,'clean'),(.02,'noise_2pct')]:
            r=np.random.default_rng(seed+gi*110+int(sigma*10000)+1)
            meas=true*np.exp(r.normal(-sigma*sigma/2,sigma,true.shape))
            logs=torch.tensor(np.log10(meas),dtype=torch.float32)
            curves={key:[] for key in ('v7','v8')}
            for xi in logs.split(64):
                with torch.no_grad():
                    z5=models['base']((xi-models['sxm'])/models['sxs']);z7=models['v7']((xi-models['sxm'])/models['sxs'],z5)
                    phys=models['physics'](z7*models['sys']+models['sym']).clamp_min(1e-4)
                    d=(xi-torch.log10(phys))/models['sxs']
                    z8=models['refiner']((xi-models['sxm'])/models['sxs'],z7,d)
                for key,z in [('v7',z7),('v8',z8)]:
                    p=10**(z.cpu().numpy()*ck['sy_scale']+ck['sy_mean']);curves[key].append(p)
            result[name][label]={}
            for key in ['v7','v8']:
                pp=np.concatenate(curves[key]);pred=np.concatenate([forward_batch(pp[i:i+50],default_ab2()) for i in range(0,len(pp),50)])
                result[name][label][key]=metric(pred,true)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--n-scenario',type=int,default=2500)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--lr',type=float,default=2.5e-4)
    parser.add_argument('--width',type=int,default=160)
    parser.add_argument('--seed',type=int,default=8831)
    parser.add_argument('--skip-eval',action='store_true')
    parser.add_argument('--output',default='models/neural_v8_candidate.pt',help='Checkpoint de saída, separado do modelo validado')
    a=parser.parse_args();(ROOT/a.output).parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed);t0=time.time()
    ck,base,v7=load_base();refiner=PhysicsCorrection(a.width)
    physics=PhysicalForward(du=.001)
    sxm=torch.tensor(ck['sx_mean'],dtype=torch.float32);sxs=torch.tensor(ck['sx_scale'],dtype=torch.float32);sym=torch.tensor(ck['sy_mean'],dtype=torch.float32);sys=torch.tensor(ck['sy_scale'],dtype=torch.float32)
    trainp,trainc=forward_groups(sample_groups(a.n_scenario,seed=88301));valp,valc=forward_groups(sample_groups(200,seed=88302))
    ltrain=torch.tensor(np.log10(trainc),dtype=torch.float32);ytrain=torch.tensor((np.log10(trainp)-ck['sy_mean'])/ck['sy_scale'],dtype=torch.float32)
    lval=torch.tensor(np.log10(valc),dtype=torch.float32)
    validation_noisy=lval+0.02*torch.randn(lval.shape,generator=torch.Generator().manual_seed(88001))/np.log(10)
    opt=torch.optim.AdamW(refiner.parameters(),lr=a.lr,weight_decay=1e-5)
    best=float('inf');best_state=None;history=[]
    for epoch in range(a.epochs):
        refiner.train()
        perm=rng.permutation(len(ltrain))
        for ids in np.array_split(perm,max(1,int(np.ceil(len(perm)/a.batch)))):
            clean=ltrain[ids];std=torch.tensor(rng.choice([0.,.005,.01,.02],size=(len(ids),1),p=[.4,.15,.2,.25]),dtype=torch.float32)
            noisy=clean+std*torch.randn_like(clean)/np.log(10)
            z,zold=predict_normalized(noisy,ck,base,v7,refiner,physics,sxm,sxs,sym,sys)
            curve=physics(z*sys+sym).clamp_min(1e-5)
            loss=75*(torch.log10(curve)-clean).square().mean()+.055*(z-ytrain[ids]).square().mean()+.003*(z-zold).square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(refiner.parameters(),3);opt.step()
        refiner.eval();valsum=0
        with torch.no_grad():
            for obs,weight in ((lval,.7),(validation_noisy,.3)):
                for ix in torch.arange(len(lval)).split(100):
                    z,_=predict_normalized(obs[ix],ck,base,v7,refiner,physics,sxm,sxs,sym,sys)
                    pred=physics(z*sys+sym).clamp_min(1e-5)
                    valsum+=weight*(torch.log10(pred)-lval[ix]).square().sum().item()
        val=valsum/(len(lval)*20);history.append({'epoch':epoch+1,'val_curve_log_mse':val})
        if val<best:
            best=val;best_state=copy.deepcopy(refiner.state_dict())
            torch.save({'weights':best_state,'width':a.width,'base':'neural_v5.pt','residual':'neural_v7_residual.pt','method':'learned physics-residual correction, no optimizer at inference','best_validation_log_mse':best},ROOT/a.output)
        print(f'epoch={epoch+1} val_log_mse={val:.9f} best={best:.9f} elapsed={time.time()-t0:.1f}s',flush=True)
    refiner.load_state_dict(best_state);refiner.eval()
    trained={'weights':best_state,'width':a.width,'base':'neural_v5.pt','residual':'neural_v7_residual.pt','method':'one trained correction from log-curves + physics-forward residual. No optimizer at inference','best_validation_log_mse':best}
    torch.save(trained,ROOT/a.output)
    mods=dict(base=base,v7=v7,refiner=refiner,physics=physics,sxm=sxm,sxs=sxs,sym=sym,sys=sys)
    report={} if a.skip_eval else evaluate(mods,ck)
    output={'train':{'epochs':a.epochs,'train_per_scenario':a.n_scenario,'val_per_scenario':200,'seed_train':88301,'seed_val':88302,'seed_test':88017,'best_val_curve_log_mse':best,'history':history},'test':{'n_per_scenario':300,'data_are_synthetic':True,'noise_observation_vs_clean_reference':True,'models':report}}
    path=ROOT/'results/metricas/training_v8_candidate.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print(json.dumps({s:{noise:{ver:round(mm['median_pct'],4) for ver,mm in vals.items()} for noise,vals in ss.items()} for s,ss in report.items()},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
