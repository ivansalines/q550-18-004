"""Sensor-only v18.24 audit. No model, state, ODE or tangent imports.

Comparisons use exactly the same noisy endpoint for all model views. An
independent per-trajectory permutation scrambles ONLY the first 16 samples;
the terminal sample (index 16) remains unchanged, bit for bit.
"""
import numpy as np
from blind_math import HISTORY,HORIZONS,RIDGES,ridge_fit,ridge_predict,rmse,choose_predictive_latent_rank

ANCHORS=(3.5,4.0,4.5)
SIGMAS=(0.,1e-4,1e-3)
PERM_SEEDS=(13,29,47)
MODELS=('endpoint','chronological','recent','past_reversed_global','past_random_per_trial')
LATENT_RANKS=(6,8,10,12,15,20)

def permute_past_only(X,seed,per_trial=True):
    x=np.asarray(X,float)
    if x.ndim!=3 or len(x[0])!=len(HISTORY):raise ValueError('Bad history dimension')
    result=x.copy();rng=np.random.default_rng(seed)
    if per_trial:
        perms=np.array([rng.permutation(len(HISTORY)-1) for _ in range(len(x))])
    else:
        p=rng.permutation(len(HISTORY)-1)
        perms=np.broadcast_to(p,(len(x),len(p)))
    result[:,:-1]=np.take_along_axis(x[:,:-1],perms[:,:,None],axis=1)
    assert np.array_equal(result[:,-1],x[:,-1])
    return result

def features(X,kind,seed=None):
    x=np.asarray(X,float)
    if kind=='endpoint':return x[:,-1,:].copy()
    if kind=='recent':return x[:,-4:,:].reshape(len(x),-1)
    if kind=='chronological':return x.reshape(len(x),-1)
    if kind=='past_reversed_global':
        return np.concatenate((x[:,:-1][:,::-1],x[:,-1:]),axis=1).reshape(len(x),-1)
    if kind=='past_random_per_trial':
        if seed is None:raise ValueError('Seed required for destructive control')
        return permute_past_only(x,seed).reshape(len(x),-1)
    raise ValueError(kind)

def noisy_data(data,sigma,seed):
    rng=np.random.default_rng(seed)
    return (np.asarray(data,float)+rng.normal(0,sigma,np.asarray(data).shape))

def fit_validation_ridge(X,Y,V,W):
    winner=None
    for penalty in RIDGES:
        model=ridge_fit(X,Y,float(penalty))
        score=rmse(ridge_predict(model,V),W)
        if winner is None or score<winner[0]:winner=(score,float(penalty),model)
    return winner

def fit_latent_models(X,Y,V,W,ranks=LATENT_RANKS):
    mean=X.mean(axis=0);std=np.maximum(X.std(axis=0),1e-7)
    Z=(X-mean)/std
    U,s,Vt=np.linalg.svd(Z,full_matrices=False)
    models={};scan=[]
    for k in ranks:
        if k>len(s):continue
        directions=Vt[:k].T
        a=Z@directions;b=(V-mean)/std@directions
        coef=np.linalg.lstsq(np.c_[np.ones(len(a)),a],Y,rcond=None)[0]
        pred=np.c_[np.ones(len(b)),b]@coef
        score=rmse(pred,W)
        models[k]=(mean,std,directions,coef)
        scan.append({'rank':int(k),'validation_rmse':score,
            'variance_fraction':float(np.sum(s[:k]**2)/np.sum(s*s))})
    selected=choose_predictive_latent_rank(scan,W)
    return models,scan,selected

def predict_latent(model,X):
    mean,std,P,coef=model
    Xred=(X-mean)/std@P
    return np.c_[np.ones(len(Xred)),Xred]@coef

def evaluate(datasets,scope,metadata):
    """datasets[observer][anchor][split]=(history[N,17,d], clean_future[N,3*d]).
    4.0 reference train/validation ONLY. Other anchors TEST ONLY. No ODE access.
    """
    import json,csv
    from pathlib import Path
    ROOT=Path(__file__).resolve().parents[1];results=ROOT/'results';results.mkdir(exist_ok=True)
    rows=[];rank_rows=[];checks=[];hidden=[]
    for obsidx,(observer,anchored) in enumerate(sorted(datasets.items())):
        reference=anchored[4.0];Xtr,Ytr=reference['train'];Xva,Yva=reference['val']
        for sigma in SIGMAS:
            noise_seed=1824000+obsidx*10000+int(sigma*1e7)
            TR=noisy_data(Xtr,sigma,noise_seed);VA=noisy_data(Xva,sigma,noise_seed+1)
            allte={a:noisy_data(anchored[a]['test'][0],sigma,noise_seed+int(a*1000)+2) for a in ANCHORS}
            test_truth={a:anchored[a]['test'][1] for a in ANCHORS}
            for mode in MODELS:
                repeats=PERM_SEEDS if mode=='past_random_per_trial' else (0,)
                for rep in repeats:
                    xt=features(TR,mode,seed=noise_seed+rep+11 if rep else None)
                    xv=features(VA,mode,seed=noise_seed+rep+21 if rep else None)
                    score,penalty,model=fit_validation_ridge(xt,Ytr,xv,Yva)
                    for anchor in ANCHORS:
                        XE=features(allte[anchor],mode,seed=noise_seed+rep+int(anchor*1000)+31 if rep else None)
                        pred=ridge_predict(model,XE);truth=test_truth[anchor]
                        rows.append({'scope':scope,'observer':observer,'sigma':sigma,'anchor':anchor,
                             'model':mode,'replicate':rep,'validation_rmse':score,'test_rmse':rmse(pred,truth),
                             'ridge_lambda':penalty,'features':XE.shape[1],
                             'n_test':len(XE),'n_train':len(Xtr),'n_validation':len(Xva)})
                        if len(XE)>36:
                            nout=truth.shape[1]//len(HORIZONS)
                            for j,h in enumerate(HORIZONS):
                                sl=slice(j*nout,(j+1)*nout)
                                hidden.append({'scope':scope,'observer':observer,'sigma':sigma,'anchor':anchor,
                                  'model':mode,'replicate':rep,'horizon':float(h),
                                  'error':float(np.linalg.norm(pred[-1,sl]-truth[-1,sl]))})
                    if mode=='past_reversed_global':
                        chrono_score,chrono_lam,chrono_model=fit_validation_ridge(features(TR,'chronological'),Ytr,
                           features(VA,'chronological'),Yva)
                        p1=ridge_predict(model,features(allte[4.0],mode))
                        p2=ridge_predict(chrono_model,features(allte[4.0],'chronological'))
                        defect=float(np.max(np.abs(p1-p2)))
                        checks.append({'scope':scope,'observer':observer,'sigma':sigma,
                            'shared_reversal_prediction_max_abs_defect':defect,
                            'shared_reversal_lambda_match':bool(penalty==chrono_lam)})
                        if defect>1e-7*max(1.,np.max(np.abs(p2))):
                            raise RuntimeError('Global reversal must preserve linear-ridge predictions')
            latent,scan,winner=fit_latent_models(features(TR,'chronological'),Ytr,
                          features(VA,'chronological'),Yva)
            for k,mdl in latent.items():
                for anchor in ANCHORS:
                    p=predict_latent(mdl,features(allte[anchor],'chronological'))
                    rank_rows.append({'scope':scope,'observer':observer,'sigma':sigma,'anchor':anchor,
                      'rank':k,'validation_rmse':next(s['validation_rmse'] for s in scan if s['rank']==k),
                      'test_rmse':rmse(p,test_truth[anchor]),
                      'validation_selected':bool(k==winner['rank']),
                      'selection_uses_shifted_test':False})
    def save_csv(name,values):
        with (results/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    save_csv(f'{scope}_model_comparison.csv',rows)
    save_csv(f'{scope}_latent_rank_transfer.csv',rank_rows)
    save_csv(f'{scope}_hidden_probe.csv',hidden)
    save_csv(f'{scope}_shared_reversal_checks.csv',checks)
    out={'version':'v18.24','status':scope,'method':'endpoint-preserved independently scrambled past',
      'metadata':metadata,'anchors':ANCHORS,'noise_levels':SIGMAS,'past_samples':len(HISTORY)-1,
      'endpoint_exactly_preserved':True,'global_shared_reversal_prediction_invariant':True,
      'train_val_anchor':4.0,'shifted_anchors_test_only':[3.5,4.5],
      'rank_selection_validation_only':True,'recorded_models':len(rows),
      'max_shared_reversal_defect':max(x['shared_reversal_prediction_max_abs_defect'] for x in checks),
      'caveats':['Synthetic normalized measurement noise, not empirical sensor uncertainty.',
        'Temporal anchor changes both phase and reference posture: not a pure clock-phase intervention.',
        'Material3 diagnostic observer was originally ODE-informed; only predictor fit/test is model blind.',
        'No physical constitutive memory or uniquely identifiable hidden coordinates established.']}
    (results/f'{scope}_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print('AUDIT',scope,'ROWS',len(rows),'LATENT_ROWS',len(rank_rows),'MAX_REVERSED_DEFECT',out['max_shared_reversal_defect'])
    return out
