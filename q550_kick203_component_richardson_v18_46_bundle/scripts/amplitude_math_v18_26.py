"""Sensor-only vanishing-amplitude latent-rank and cross-posture audit.

No Q550 ODE/Jacobian/cocycle imports. The ODE *oracle* creates reduced
trajectories externally; this module receives ONLY reduced noisy histories
and clean reduced future labels. All calibrations use paired PAST histories;
reference rank is selected using xi=4 validation future only. Test future
labels are used only after model selection, for final scores.
"""
from pathlib import Path
import numpy as np,csv,json
from blind_math import HORIZONS,rmse
from robust_math_v18_24 import fit_latent_models,predict_latent
from transport_math_v18_25 import ANCHORS,CAL_N,asflat,latent_encode,latent_target,fit_latent_transport,center_map_predict

AMPLITUDES=(1.e-5,3.e-6,1.e-6,3.e-7,1.e-7)
REFERENCE_AMPLITUDE=AMPLITUDES[0]
BASE_NOISE=1.e-4
NOISE_MODES=('none','fixed_normalized','fixed_absolute')
RANKS=(10,20)
MODES=('none','orthogonal','affine_ridge','broken_pair_orthogonal')
OBSERVERS=('material3','raw4','raw6')

def effective_sigma(amplitude,noise_mode):
    """Physical sensor noise fixed at ref_eps*1e-4: normalized rises as 1/eps."""
    if noise_mode=='none':return 0.
    if noise_mode=='fixed_normalized':return BASE_NOISE
    if noise_mode=='fixed_absolute':return BASE_NOISE*REFERENCE_AMPLITUDE/float(amplitude)
    raise ValueError(noise_mode)

def sensor_noise(x,sigma,seed):
    return np.asarray(x,float)+np.random.default_rng(seed).normal(0,sigma,size=np.asarray(x).shape)

def predict_with_alignment(model,model_type,cal,reference_cal,test):
    if model_type=='none':return predict_latent(model,asflat(test)),None
    kind='affine_ridge' if model_type=='affine_ridge' else 'orthogonal'
    fit,lam,cal_error=fit_latent_transport(model,cal,reference_cal,kind,
                                          wrong_pair=model_type=='broken_pair_orthogonal')
    pred=latent_target(model,center_map_predict(fit,latent_encode(model,test)))
    return pred,{'calibration_error':float(cal_error),'ridge':lam}

def evaluate(datasets,scope,provenance,amplitudes=AMPLITUDES,noise_modes=NOISE_MODES,
             output_dir=None):
    """datasets[eps][obs][anchor][split] -> (clean X [n,17,d], clean future [n,3*d]).

    Training directions, validation directions, test directions and calibration
    directions are invariant across amplitude. Different seeds ensure no
    sensor-noise reuse BETWEEN cohorts; per-epsilon common draws isolate
    deterministic amplitude dependence.
    """
    if output_dir is None:output_dir=Path(__file__).resolve().parents[1]/'results'
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    rows=[];rank_rows=[];diagnostics=[];frozen_rows=[]
    for oi,obs in enumerate(OBSERVERS):
      for ai,eps in enumerate(amplitudes):
        data=datasets[float(eps)][obs]
        Xtr,Ytr=data[4.]['train'];Xva,Yva=data[4.]['val']
        assert len(Xtr)==96 and len(Xva)==32
        for noise_mode in noise_modes:
            sigma=effective_sigma(eps,noise_mode)
            # The seed is independent of amplitude to make comparisons paired.
            seed=18260000+oi*100000+list(NOISE_MODES).index(noise_mode)*10000
            train=sensor_noise(Xtr,sigma,seed+1)
            val=sensor_noise(Xva,sigma,seed+2)
            ref_cal=train[:CAL_N].copy()
            models,rank_scan,selected=fit_latent_models(asflat(train),Ytr,asflat(val),Yva,
                                                       ranks=(6,8,10,12,15,20))
            for rk in RANKS:
                rank_rows.append({'scope':scope,'observer':obs,'amplitude':eps,'noise_mode':noise_mode,
                   'normalized_noise_sigma':sigma,'rank':rk,
                   'validation_rmse':next(float(r['validation_rmse']) for r in rank_scan if r['rank']==rk),
                   'validation_selected_rank':selected['rank'],
                   'selection_uses_shifted_futures':False})
            for anchor in ANCHORS:
                xt,yt=data[anchor]['test'];test=sensor_noise(xt,sigma,seed+int(anchor*1000)+3)
                if anchor==4.: cal=ref_cal
                else:
                    c,_=data[anchor]['cal']
                    assert len(c)==CAL_N
                    cal=sensor_noise(c,sigma,seed+int(anchor*1000)+4)
                for rank in RANKS:
                    mdl=models[rank]
                    for method in MODES:
                        pred,cal_info=predict_with_alignment(mdl,method,cal,ref_cal,test)
                        score=rmse(pred,yt)
                        width=xt.shape[-1]
                        item={'scope':scope,'observer':obs,'amplitude':eps,'noise_mode':noise_mode,
                              'normalized_noise_sigma':sigma,'anchor':anchor,'rank':rank,
                              'method':method,'test_rmse':score,'n_test':len(xt),
                              'validation_selected_rank':selected['rank'],
                              'calibration_uses_test_future':False,
                              'predictor_knows_ode':False}
                        for j,h in enumerate(HORIZONS):
                            item[f'probe_error_h{float(h):g}']=float(np.linalg.norm(
                                pred[-1,j*width:(j+1)*width]-yt[-1,j*width:(j+1)*width]))
                        rows.append(item)
                        if cal_info:
                            diagnostics.append({'scope':scope,'observer':obs,'amplitude':eps,
                              'noise_mode':noise_mode,'anchor':anchor,'rank':rank,'method':method,
                              **cal_info,'paired_calibration_count':CAL_N,
                              'calibration_uses_futures':False})
            print(scope,obs,'eps',eps,'noise',noise_mode,'rank selection',selected['rank'],flush=True)
    # Frozen-model amplitude-transfer: fit ONCE at epsilon_ref and reuse both
    # latent bases as epsilon decreases. No refitting on small-amplitude future.
    for oi,obs in enumerate(OBSERVERS):
      for noise_mode in noise_modes:
        sigma_ref=effective_sigma(REFERENCE_AMPLITUDE,noise_mode)
        seed=18260000+oi*100000+list(NOISE_MODES).index(noise_mode)*10000
        ref=datasets[REFERENCE_AMPLITUDE][obs][4.]
        tx,ty=ref['train'];vx,vy=ref['val']
        train=sensor_noise(tx,sigma_ref,seed+1)
        valid=sensor_noise(vx,sigma_ref,seed+2)
        ref_models,ref_scan,ref_selected=fit_latent_models(asflat(train),ty,asflat(valid),vy,
                                                       ranks=(6,8,10,12,15,20))
        for eps in amplitudes:
          sigma=effective_sigma(eps,noise_mode)
          for anchor in ANCHORS:
            te,y=datasets[float(eps)][obs][anchor]['test']
            noisy=sensor_noise(te,sigma,seed+int(anchor*1000)+3)
            for rank in RANKS:
                guess=predict_latent(ref_models[rank],asflat(noisy))
                frozen_rows.append({'scope':scope,'observer':obs,'training_amplitude':REFERENCE_AMPLITUDE,
                  'test_amplitude':eps,'noise_mode':noise_mode,'anchor':anchor,'rank':rank,
                  'test_rmse':rmse(guess,y),'selected_at_reference':ref_selected['rank'],
                  'refitted_at_test_amplitude':False,'used_test_futures_for_selection':False})
    # Compare rank20/rank10 on the SAME set of held-out samples and noise draws.
    paired=[]
    lookup={(r['observer'],r['amplitude'],r['noise_mode'],r['anchor'],r['method'],r['rank']):r for r in rows}
    for r in rows:
        if r['rank']!=10:continue
        match=lookup[(*[r[k] for k in ('observer','amplitude','noise_mode','anchor','method')],20)]
        paired.append({'scope':scope,'observer':r['observer'],'amplitude':r['amplitude'],
            'noise_mode':r['noise_mode'],'anchor':r['anchor'],'method':r['method'],
            'rank10_rmse':r['test_rmse'],'rank20_rmse':match['test_rmse'],
            'rank10_over_rank20':r['test_rmse']/max(match['test_rmse'],1e-30),
            'rank20_advantage_absolute':r['test_rmse']-match['test_rmse']})
    def csv_write(name,items):
        with (output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    csv_write(f'{scope}_amplitude_prediction.csv',rows)
    csv_write(f'{scope}_rank20_vs_10.csv',paired)
    csv_write(f'{scope}_rank_selection.csv',rank_rows)
    csv_write(f'{scope}_calibration.csv',diagnostics)
    csv_write(f'{scope}_frozen_reference_transfer.csv',frozen_rows)
    summary={'version':'v18.26','scope':scope,'status':provenance.get('status',scope),
      'provenance':provenance,'amplitudes':[float(e) for e in amplitudes],
      'train_validation_test_counts':[96,32,37],'calibration_paired_count':CAL_N,
      'anchors':list(ANCHORS),'noise_modes':list(noise_modes),'raw_noise_scale':BASE_NOISE*REFERENCE_AMPLITUDE,
      'effective_sigma_at_smallest_epsilon':{
         mode:effective_sigma(min(amplitudes),mode) for mode in noise_modes},
      'model_ranks':list(RANKS),'methods':list(MODES),'rows':len(rows),'frozen_model_transfer_rows':len(frozen_rows),
      'present_sample_left_unchanged_during_alignment':True,
      'reference_training_and_validation_futures_only':True,
      'calibration_uses_only_paired_past_histories':True,
      'no_test_future_leakage':True,
      'same_direction_splits_at_all_amplitudes':True,
      'caveats':['Fixed normalized noise is NOT equivalent to fixed absolute physical sensor noise.',
          'Finite amplitude scan cannot establish an actual limit as epsilon tends to zero.',
          'Numerical integration/cancellation floor must be compared with curvature signal.',
          'Rank 20 is predictive coordinate count, not an assertion of physical dimension.',
          'Off-constraint numerical trajectories; no constitutive memory or privileged metric certified.',
          'Cross posture tests shift both phase and material posture; alignment uses known matched histories.']}
    (output_dir/f'{scope}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
