"""Sensor-history-only cross-posture transport, v18.25.

This file MUST NOT import any Q550 ODE, propagator, Jacobian, or 10D state.
Calibration pairs identify the same training perturbation across sections but
provide ONLY reduced histories. All alignment selection uses histories, never
future labels. Future labels are used to train the reference predictor at xi=4
and only then to score truly held-out section transfers.
"""
from pathlib import Path
import csv,json
import numpy as np
from blind_math import HISTORY,HORIZONS,ridge_fit,ridge_predict,rmse
from robust_math_v18_24 import fit_latent_models,predict_latent

ANCHORS=(3.5,4.0,4.5)
SIGMAS=(0.,1e-4,1e-3)
RANKS=(10,20)
CAL_N=32
CAL_TR=24
SHIFT_GRID=np.linspace(-.125,.125,11)
TRANSPORTS=('none','past_lag','orthogonal','lag_orthogonal','affine_ridge','broken_pair_orthogonal')
RIDGES=(1e-6,1e-4,1e-2,.1,1.,10.)


def past_shift(X,lag):
    """Interpolate past only, with present held bitwise invariant."""
    x=np.asarray(X,float)
    if x.ndim!=3 or x.shape[1]!=len(HISTORY):raise ValueError('Invalid history stencil')
    out=x.copy()
    query=np.clip(HISTORY[:-1]+float(lag),HISTORY[0],HISTORY[-1])
    for j in range(x.shape[2]):
        for i in range(len(x)):
            out[i,:-1,j]=np.interp(query,HISTORY,x[i,:,j])
    assert np.array_equal(out[:,-1,:],x[:,-1,:])
    return out


def asflat(x):return np.asarray(x,float).reshape(len(x),-1)


def center_map_fit(x,y,method,lam=0.):
    """Maps source latent histories to paired reference latent histories."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    mx=x.mean(0);my=y.mean(0);X=x-mx;Y=y-my
    if method=='orthogonal':
        U,s,Vt=np.linalg.svd(X.T@Y,full_matrices=False)
        A=U@Vt
    elif method=='affine_ridge':
        A=np.linalg.solve(X.T@X+max(float(lam),1e-14)*np.eye(x.shape[1]),X.T@Y)
    else:raise ValueError(method)
    return (mx,my,A)


def center_map_predict(model,z):
    mx,my,A=model
    return (np.asarray(z)-mx)@A+my


def latent_encode(model,X):
    m,s,V,b=model
    return ((asflat(X)-m)/s)@V


def latent_target(model,Z):
    return np.c_[np.ones(len(Z)),Z]@model[3]


def choose_lag(cal_shift,cal_ref):
    """Select a single temporal offset using paired PAST histories only.

    Train 24 paired histories; choose on the remaining 8 (no futures).
    The criterion is distance in raw channel/time histories standardized
    using the reference calibration scale; cannot inspect hidden state.
    """
    a=np.asarray(cal_shift);b=np.asarray(cal_ref)
    if len(a)!=CAL_N or len(b)!=CAL_N:raise ValueError('Calibration size mismatch')
    ref_scale=np.maximum(b[:CAL_TR].std(axis=0),1e-4)
    candidates=[]
    for delta in SHIFT_GRID:
        shifted=past_shift(a,float(delta))
        # A fixed linear channel gain and offset is permitted to remove gross
        # amplitude bias. Fit on 24 pairs only; score on 8 holdout pairs.
        inp=(shifted[:CAL_TR]/ref_scale).reshape(CAL_TR,-1)
        tar=(b[:CAL_TR]/ref_scale).reshape(CAL_TR,-1)
        num=float(np.sum(inp*tar));den=float(np.sum(inp*inp))
        gain=num/max(den,1e-16)
        valid=(gain*shifted[CAL_TR:]-b[CAL_TR:])/ref_scale
        err=float(np.sqrt(np.mean(valid**2)))
        candidates.append((err,abs(float(delta)),float(delta)))
    best=min(candidates)
    return best[2],best[0]


def fit_latent_transport(model,shift_cal,ref_cal,kind,wrong_pair=False):
    """Alignment trained solely on paired reduced past/current samples."""
    zs=latent_encode(model,shift_cal);zr=latent_encode(model,ref_cal)
    if wrong_pair:
        perm=np.random.default_rng(182509).permutation(len(zr))
        zr=zr[perm]
    if kind=='orthogonal':
        fit=center_map_fit(zs,zr,'orthogonal')
        recon=center_map_predict(fit,zs[CAL_TR:])
        defect=rmse(recon,zr[CAL_TR:])
        return fit,None,defect
    if kind=='affine_ridge':
        scan=[]
        for lam in RIDGES:
            fit=center_map_fit(zs[:CAL_TR],zr[:CAL_TR],'affine_ridge',lam)
            err=rmse(center_map_predict(fit,zs[CAL_TR:]),zr[CAL_TR:])
            scan.append((err,float(lam)))
        selected=min(scan,key=lambda row:(row[0],row[1]))
        fit=center_map_fit(zs,zr,'affine_ridge',selected[1])
        return fit,selected[1],selected[0]
    raise ValueError(kind)


def rank_models(Xtr,Ytr,Xva,Yva):
    """Rank choice and future predictor fit only at reference xi=4."""
    models,scan,selection=fit_latent_models(asflat(Xtr),Ytr,asflat(Xva),Yva,
                                             ranks=(6,8,10,12,15,20))
    return models,scan,selection


def add_sensor_noise(x,sigma,seed):
    return np.asarray(x,float)+np.random.default_rng(seed).normal(0,float(sigma),size=np.asarray(x).shape)


def evaluate(datasets,scope,metadata):
    """datasets[obs][anchor] = {'train','val','cal','test'}; each (history, clean future).

    At xi=4: train(96), val(32), test(37).
    At xi=3.5/4.5: cal(32 first training directions), test(37).
    Only test histories/futures enter the scoring stage; alignment and model
    selection are fixed BEFORE test scoring.
    """
    root=Path(__file__).resolve().parents[1];rdir=root/'results';rdir.mkdir(exist_ok=True)
    rows=[];lag_rows=[];cal_rows=[];selected_rows=[]
    for oi,obs in enumerate(('material3','raw4','raw6')):
        data=datasets[obs];ref=data[4.0]
        Xtr,Ytr=ref['train'];Xva,Yva=ref['val'];Xte,Yte=ref['test']
        assert len(Xtr)==96 and len(Xva)==32 and len(Xte)==37
        assert len(HISTORY)==17 and len(HORIZONS)==3
        for si,sigma in enumerate(SIGMAS):
            seed=18250000+oi*100000+si*10000
            t=add_sensor_noise(Xtr,sigma,seed+1)
            v=add_sensor_noise(Xva,sigma,seed+2)
            ref_cal=t[:CAL_N]
            models,scan,selected=rank_models(t,Ytr,v,Yva)
            for rank in RANKS:
                assert rank in models
                selected_rows.append({'scope':scope,'observer':obs,'sigma':sigma,'rank':rank,
                   'validation_rmse':next(r['validation_rmse'] for r in scan if r['rank']==rank),
                   'validation_selected_rank':selected['rank'],
                   'rank_selected_without_shifted_future':True})
            for anchor in ANCHORS:
                xt,yt=data[anchor]['test'];test=add_sensor_noise(xt,sigma,seed+int(anchor*1000)+3)
                if anchor==4.0:
                    cal=ref_cal
                else:
                    c,_=data[anchor]['cal']
                    assert len(c)==CAL_N
                    cal=add_sensor_noise(c,sigma,seed+int(anchor*1000)+4)
                lag,lag_score=choose_lag(cal,ref_cal) if anchor!=4 else (0.,0.)
                lag_rows.append({'scope':scope,'observer':obs,'sigma':sigma,'anchor':anchor,
                   'selected_lag':lag,'past_only_calibration_error':lag_score,'endpoint_preserved':True})
                for rank in RANKS:
                    model=models[rank]
                    # Calibrate on paired past/current features: never use test features or future labels.
                    algs={}
                    for mode in ('orthogonal','lag_orthogonal','affine_ridge','broken_pair_orthogonal'):
                        source=past_shift(cal,lag) if mode=='lag_orthogonal' else cal
                        kind='affine_ridge' if mode=='affine_ridge' else 'orthogonal'
                        aligned,reg,err=fit_latent_transport(model,source,ref_cal,kind,
                                                 wrong_pair=(mode=='broken_pair_orthogonal'))
                        algs[mode]=aligned
                        cal_rows.append({'scope':scope,'observer':obs,'sigma':sigma,'anchor':anchor,
                            'rank':rank,'model':mode,'past_only_alignment_error':err,
                            'alignment_ridge_lambda':reg if reg is not None else '',
                            'calibration_n':CAL_N,'selection_uses_future':False})
                    for mode in TRANSPORTS:
                        if mode=='none':
                            prediction=predict_latent(model,asflat(test))
                        elif mode=='past_lag':
                            prediction=predict_latent(model,asflat(past_shift(test,lag)))
                        else:
                            target=past_shift(test,lag) if mode=='lag_orthogonal' else test
                            latent=latent_encode(model,target)
                            latent_aligned=center_map_predict(algs[mode],latent)
                            prediction=latent_target(model,latent_aligned)
                        row={'scope':scope,'observer':obs,'sigma':sigma,'anchor':anchor,'rank':rank,
                             'method':mode,'test_rmse':rmse(prediction,yt),
                             'selected_lag':lag,'validation_selected_rank':selected['rank'],
                             'n_test':len(xt),'calibration_uses_shifted_futures':False,
                             'calibration_uses_test_futures':False,
                             'reference_predictor_uses_shifted_futures':False,
                             'normalized_synthetic_noise':True}
                        # The last 37th test is the original heldout hidden-direction probe.
                        width=xt.shape[-1]
                        for j,h in enumerate(HORIZONS):
                            row[f'probe_error_h{float(h):g}']=float(np.linalg.norm(
                                prediction[-1,j*width:(j+1)*width]-yt[-1,j*width:(j+1)*width]))
                        rows.append(row)
                print(scope,obs,'sigma',sigma,'anchor',anchor,'calibrated lag',lag,flush=True)
    def write_csv(name,data):
        with (rdir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for name,table in (('transport_comparison',rows),('past_lag_selection',lag_rows),
                     ('calibration_diagnostics',cal_rows),('latent_rank_reference_selection',selected_rows)):
        write_csv(f'{scope}_{name}.csv',table)
    result={'version':'v18.25','scope':scope,'metadata':metadata,'tested_anchors':list(ANCHORS),
       'calibration_pairs':CAL_N,'calibration_split_for_lag':{'fit':CAL_TR,'holdout':CAL_N-CAL_TR},
       'reference_training_n':96,'reference_validation_n':32,'test_n':37,
       'rank_values':list(RANKS),'sigma_values':list(SIGMAS),'transport_methods':list(TRANSPORTS),
       'rows':len(rows),'endpoint_fixed_when_past_shifted':True,
       'calibration_uses_no_future_observations':True,
       'calibration_known_pair_correspondence':True,
       'wrong_pair_negative_control':True,
       'rank_selection_uses_only_reference_validation_futures':True,
       'shifted_sections_share_training_directions_only_for_calibration':True,
       'caveats':[
           'Relative history alignment requires matched calibration identities across sections; this is additional information compared with zero-shot transfer.',
           'Rank 20 represents predictive coordinates, not 20 physical degrees of freedom.',
           'Changing xi moves both material posture and clock phase, not a pure phase intervention.',
           'Off-constraint Q550 perturbations; no privileged metric or constitutive hysteresis certified.',
           'Synthetic sensor noise in amplitude-normalized observations, not physical measurement noise.',
           'Parent material3 diagnostic projection was previously designed using the ODE.'
           ]}
    (rdir/f'{scope}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(scope,'transport rows',len(rows),'align records',len(cal_rows))
    return result
