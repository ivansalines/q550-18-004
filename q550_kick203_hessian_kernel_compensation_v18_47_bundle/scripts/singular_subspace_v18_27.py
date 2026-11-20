"""v18.27: genuine nonlinear reduced-history SVD and latent-gauge audit.

No ODE/Jacobian/cocycle is exposed to this script. It authenticates the original
v18.26 server sendback, refits SVD bases using TRAIN histories only, evaluates
predictions on unchanged held-out TEST trajectories and records singular-subspace
angles. Singular vectors are coordinate-dependent; stable projectors and future
predictions are more meaningful than individual vector signs or labels.
"""
from __future__ import annotations
import csv,hashlib,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from robust_math_v18_24 import fit_latent_models,predict_latent
from amplitude_math_v18_26 import AMPLITUDES,OBSERVERS,NOISE_MODES,effective_sigma,sensor_noise,RANKS
from blind_math import rmse
from transport_math_v18_25 import ANCHORS,asflat
PARENT=ROOT/'inputs/v18_26_original_server_sendback.zip'
PREFIX='q550_kick203_vanishing_amplitude_latent_transport_v18_26_bundle/'
RESULTS=ROOT/'results'
NOISE=('none','fixed_normalized','fixed_absolute')


def authentic_data():
    """Verify EVERY authenticated SHA entry and the original-Q550 proof."""
    sha=hashlib.sha256(PARENT.read_bytes()).hexdigest()
    with zipfile.ZipFile(PARENT) as z:
        if z.testzip():raise RuntimeError('Parent CRC mismatch')
        if {name.split('/')[0] for name in z.namelist()}!={PREFIX[:-1]}:
            raise RuntimeError('Unexpected parent archive root')
        ledger=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,digest in ledger.items():
            if hashlib.sha256(z.read(PREFIX+rel)).hexdigest()!=digest:
                raise RuntimeError('Parent v18.26 SHA mismatch: '+rel)
        proof=json.loads(z.read(PREFIX+'results/original_ode_amplitude_proof.json'))
        if proof.get('status')!='ORIGINAL_Q550_NONLINEAR_VANISHING_AMPLITUDE_SCAN_COMPLETED':
            raise RuntimeError('Parent never passed original ODE integration')
        raw=z.read(PREFIX+'results/original_reduced_dataset_v18_26.npz')
        original_rows=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_rank20_vs_10.csv').decode())))
        transfer_rows=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_frozen_reference_transfer.csv').decode())))
        probe_rows=list(csv.DictReader(io.StringIO(z.read(PREFIX+'results/original_even_odd_probe.csv').decode())))
    db=np.load(io.BytesIO(raw))
    return db,{'parent_sha256':sha,'verified_parent_files':len(ledger),
               'proof_status':proof['status'],'parent_integrations':proof['new_original_ode_integrations']},original_rows,transfer_rows,probe_rows


def get(db,e,obs,anchor,split):
    p=f'e{e:.10g}__{obs}__a{anchor:g}__{split}__'
    return db[p+'past'].copy(),db[p+'future'].copy()


def svd_frame(trainX):
    X=asflat(trainX)
    mean=X.mean(axis=0);std=np.maximum(X.std(axis=0),1e-7)
    z=(X-mean)/std
    _,s,Vt=np.linalg.svd(z,full_matrices=False)
    return mean,std,s,Vt.T


def principal_metrics(Vref,V,k):
    U=Vref[:,:k]; W=V[:,:k]
    vals=np.linalg.svd(U.T@W,compute_uv=False)
    vals=np.clip(vals,0,1)
    theta=np.rad2deg(np.arccos(vals))
    return dict(max_principal_angle_deg=float(theta.max()),
                median_principal_angle_deg=float(np.median(theta)),
                projector_distance_frob=float(np.sqrt(max(0.,2*k-2*np.sum(vals**2)))),
                min_principal_cosine=float(vals.min()))


def csvsave(name, rows):
    if not rows:raise ValueError('Cannot save empty rows '+name)
    with (RESULTS/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def matched_original(rows,obs,eps,noise,anchor,method,rank):
    m=[r for r in rows if r['observer']==obs and float(r['amplitude'])==float(eps)
       and r['noise_mode']==noise and float(r['anchor'])==float(anchor)
       and r['method']==method]
    if len(m)!=1:raise AssertionError('Missing parent row '+str((obs,eps,noise,anchor,method)))
    k='rank10_rmse' if rank==10 else 'rank20_rmse'
    return float(m[0][k])


def matched_frozen(rows,obs,eps,noise,anchor,rank):
    m=[r for r in rows if r['observer']==obs and float(r['test_amplitude'])==float(eps)
       and r['noise_mode']==noise and float(r['anchor'])==float(anchor) and int(r['rank'])==rank]
    if len(m)!=1:raise AssertionError('Missing parent frozen row')
    return float(m[0]['test_rmse'])


def analyze():
    RESULTS.mkdir(exist_ok=True)
    db,proof,prior_refit,prior_frozen,prior_probe=authentic_data()
    frames=[];spectral=[];predict=[];gauge=[];basis_storage={}
    refmodels={}
    rng=np.random.default_rng(182701)
    max_reproduce=0.;max_gauge=0.
    for oi,obs in enumerate(OBSERVERS):
      for noise in NOISE:
        for eps in AMPLITUDES:
            Xtr,Ytr=get(db,eps,obs,4.,'train');Xva,Yva=get(db,eps,obs,4.,'val')
            seed=18260000+oi*100000+NOISE_MODES.index(noise)*10000
            sigma=effective_sigma(eps,noise)
            TX=sensor_noise(Xtr,sigma,seed+1);VX=sensor_noise(Xva,sigma,seed+2)
            mean,std,s,V=svd_frame(TX)
            models,scan,winner=fit_latent_models(asflat(TX),Ytr,asflat(VX),Yva,ranks=(6,8,10,12,15,20))
            if eps==AMPLITUDES[0]:refmodels[obs,noise]=(mean,std,s,V,models)
            r_mean,r_std,r_s,r_V,_=refmodels[obs,noise]
            tag=f'{obs}_{noise}_e{eps:.10g}'.replace('-','m').replace('+','p')
            basis_storage[tag+'__s']=s
            basis_storage[tag+'__V20']=V[:,:20]
            basis_storage[tag+'__mean']=mean;basis_storage[tag+'__std']=std
            basis_storage[tag+'__coef10']=models[10][3]
            basis_storage[tag+'__coef20']=models[20][3]
            positive=s[s>np.finfo(float).tiny]
            for k in (10,20):
                met=principal_metrics(r_V,V,k)
                rows={'observer':obs,'noise_mode':noise,'amplitude':eps,'rank':k,
                     'selected_rank_on_reference_validation':winner['rank'],
                     'normalized_sensor_sigma':sigma,
                     'std_relative_change_l2':float(np.linalg.norm(std-r_std)/max(np.linalg.norm(r_std),1e-30)),
                     **met}
                frames.append(rows)
            spectral.append({'observer':obs,'noise_mode':noise,'amplitude':eps,
                's1':float(s[0]),'s9':float(s[8]),'s10':float(s[9]),
                's11':float(s[10]),'s20':float(s[19]),'s21':float(s[20]),
                's10_over_s11':float(s[9]/max(s[10],1e-300)),
                's20_over_s21':float(s[19]/max(s[20],1e-300)),
                'tail_11_20_fraction':float(np.sum(s[10:20]**2)/np.sum(s**2)),
                'tail_above20_fraction':float(np.sum(s[20:]**2)/np.sum(s**2)),
                'numeric_rank_at_1e_minus8_s1':int(np.sum(s>1e-8*s[0])),
                'smallest_positive_singular_value':float(positive[-1])})
            Q,_=np.linalg.qr(rng.normal(size=(20,20)))
            for anchor in ANCHORS:
                Xt,Yt=get(db,eps,obs,anchor,'test')
                noisy=sensor_noise(Xt,sigma,seed+int(anchor*1000)+3)
                for k in (10,20):
                    mdl=models[k];refmdl=refmodels[obs,noise][4][k]
                    own=rmse(predict_latent(mdl,asflat(noisy)),Yt)
                    frozen=rmse(predict_latent(refmdl,asflat(noisy)),Yt)
                    expected_own=matched_original(prior_refit,obs,eps,noise,anchor,'none',k)
                    expected_frozen=matched_frozen(prior_frozen,obs,eps,noise,anchor,k)
                    max_reproduce=max(max_reproduce,abs(own-expected_own),abs(frozen-expected_frozen))
                    if not np.isclose(own,expected_own,rtol=2e-8,atol=2e-11):raise RuntimeError('Refit reproduction differs')
                    if not np.isclose(frozen,expected_frozen,rtol=2e-8,atol=2e-11):raise RuntimeError('Frozen reproduction differs')
                    predict.append({'observer':obs,'noise_mode':noise,'amplitude':eps,'anchor':anchor,
                        'rank':k,'retrained_test_rmse':own,'frozen_test_rmse':frozen,
                        'parent_retrained_test_rmse':expected_own,'parent_frozen_test_rmse':expected_frozen,
                        'retrained_minus_frozen':own-frozen})
                    if k==20 and anchor==4.:
                        m,sd,Vk,coef=mdl
                        transformed=(m,sd,Vk@Q,np.r_[coef[:1],Q.T@coef[1:]])
                        gerr=float(np.max(np.abs(predict_latent(mdl,asflat(noisy))-
                                                    predict_latent(transformed,asflat(noisy)))))
                        max_gauge=max(max_gauge,gerr)
                        gauge.append({'observer':obs,'noise_mode':noise,'amplitude':eps,
                           'gauge_rotation_prediction_abs_max':gerr,
                           'rotation_does_not_change_span':True})
    csvsave('original_singular_spectrum_v18_27.csv',spectral)
    csvsave('original_principal_subspace_v18_27.csv',frames)
    csvsave('original_model_reproduction_v18_27.csv',predict)
    csvsave('original_gauge_invariance_v18_27.csv',gauge)
    np.savez_compressed(RESULTS/'original_latent_bases_v18_27.npz',**basis_storage)
    # Old ±epsilon probes are not themselves a high-precision convergence proof.
    floor=[]
    for r in prior_probe:
        eps=float(r['amplitude']);even=float(r['even_response_rms']);oddres=float(r['odd_vs_tangent_rms'])
        floor.append({'observer':r['observer'],'anchor':float(r['anchor']),'amplitude':eps,
           'even_normalized_rms':even,'even_unnormalized_eps_times_rms':eps*even,
           'odd_vs_tangent_normalized_rms':oddres,
           'odd_residual_unnormalized_eps_times_rms':eps*oddres,
           'not_a_numerical_convergence_certificate':True})
    csvsave('original_parent_even_odd_floor_proxy_v18_27.csv',floor)
    summary={'version':'v18.27','status':'AUTHENTICATED_V18_26_REDUCED_DATA_SINGULAR_ANALYSIS_COMPLETED',
       'parent':proof,'amplitudes':list(AMPLITUDES),'spectrum_rows':len(spectral),
       'subspace_rows':len(frames),'model_replication_rows':len(predict),
       'gauge_rows':len(gauge),'floor_proxy_rows':len(floor),
       'max_parent_prediction_reproduction_difference':max_reproduce,
       'max_exact_latent_gauge_prediction_defect':max_gauge,
       'training_samples_per_frame':96,'test_samples_per_posture':37,
       'information_boundary':'SVD fit from reduced past TRAIN histories only; future labels used for predictive regression. No future TEST data in any fit.',
       'not_yet_completed':'HIGH_PRECISION_ORIGINAL_ODE_FLOOR_CONTROL',
       'caveats':['Rank 20 is an embedding dimension, not a physical degree-of-freedom count.',
                  'Small singular tails may be numerical, finite-amplitude nonlinear effects, or noisy directions.',
                  'Principal angles depend on the standardization gauge of histories: report scale drift.',
                  'Principal-vector rotations within a degenerate cluster do not imply physical torsion.',
                  'Raw epsilon*even response is a floor PROXY, not an actual error bound.',
                  'Local off-constraint diagnostic; no privileged material metric or constitutive memory.']}
    (RESULTS/'original_subspace_summary_v18_27.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('v18.27 cached genuine-data audit:',len(spectral),'spectra,',len(frames),'projectors,',len(predict),'predictions')
    print('parent reproduction max',max_reproduce,'gauge max',max_gauge,flush=True)
    return summary

if __name__=='__main__':analyze()
