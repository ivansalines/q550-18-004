"""Common experimental protocol for frozen tangent and original nonlinear Q550 samples.

No model matrices are passed to the estimator; they are used ONLY by separate data
producers and never appear in the predictor module blind_math.py.
"""
from pathlib import Path
import sys,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from blind_math import HISTORY,HORIZONS,evaluate_blind,rmse,delay_latent_rank_scan

RANDOM_SEED=182319
N_TRAIN=96;N_VAL=32;N_TEST=36
NOISES=(0.0,1e-4,1e-3)
OBSERVERS=('material3','raw4','raw6')
MATCHED={'material3':'material3_jet2','raw4':'raw4_jet1','raw6':'raw6_jet1'}


def initial_directions(hidden):
    """Strictly disjoint training, validation and test perturbation draws."""
    rng=np.random.default_rng(RANDOM_SEED)
    U=rng.normal(size=(N_TRAIN+N_VAL+N_TEST,10))
    U/=np.linalg.norm(U,axis=1,keepdims=True)
    result={}
    for obs in OBSERVERS:
        seed=np.asarray(hidden[MATCHED[obs]],float)
        seed/=np.linalg.norm(seed)
        result[obs]={
          'train':U[:N_TRAIN].copy(),
          'val':U[N_TRAIN:N_TRAIN+N_VAL].copy(),
          'test':np.vstack((U[N_TRAIN+N_VAL:].copy(),seed)),
          'matched_seed':seed,
          'provenance':'First 36 tests random, last test previously verified hidden direction'}
    return result


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def run_experiment(datasets,status,provenance,nonlinear=False):
    rows=[];hidden_rows=[];spectra={};latent_rows=[];latent_winners={}
    for obs in OBSERVERS:
        data=datasets[obs]
        training,valid,testing=(data[k] for k in ('train','val','test'))
        Xtr,Ytr=training;Xva,Yva=valid;Xte,Yte=testing
        for sigma in NOISES:
            outcome,pred=evaluate_blind(Xtr,Ytr,Xva,Yva,Xte,Yte,sigma,seed=RANDOM_SEED+int(sigma*1e7)+len(obs))
            spectra[f'{obs}_{sigma}']=outcome.pop('history_singular_values')
            latent,win=delay_latent_rank_scan(Xtr,Ytr,Xva,Yva,Xte,Yte,sigma,seed=RANDOM_SEED+int(sigma*1e7)+len(obs)+500)
            latent_winners[f'{obs}_{sigma}']=win
            for z in latent:
                latent_rows.append(dict(observer=obs,noise_sigma=sigma,**z))
            base=outcome.pop('test_baseline_rmse')
            for model in ('instant','recent_0.10','history_0.50','per_trial_shuffled'):
                perf=outcome[model]
                rows.append({'observer':obs,'noise_sigma':sigma,'model':model,
                  'training_n':len(Xtr),'validation_n':len(Xva),'test_n':len(Xte),
                  'validation_rmse':perf['validation_rmse'],'test_rmse':perf['test_rmse'],
                  'baseline_rmse':base,'ridge_lambda':perf['ridge_lambda'],
                  'n_features':perf['feature_count']})
                for j,h in enumerate(HORIZONS):
                    m=Xte.shape[2]
                    # final test row is the held-out matched hidden direction.
                    a=pred[model][-1,j*m:(j+1)*m];b=Yte[-1,j*m:(j+1)*m]
                    hidden_rows.append({'observer':obs,'noise_sigma':sigma,'model':model,
                        'future_horizon':h,'true_future_norm':float(np.linalg.norm(b)),
                        'prediction_norm':float(np.linalg.norm(a)),
                        'heldout_hidden_error':float(np.linalg.norm(a-b)),
                        'heldout_hidden_baseline_error':float(np.linalg.norm(b))})
    results=ROOT/'results';results.mkdir(exist_ok=True)
    prefix='original' if nonlinear else 'frozen'
    save_csv(results/f'{prefix}_model_comparison.csv',rows)
    save_csv(results/f'{prefix}_hidden_probe_predictions.csv',hidden_rows)
    save_csv(results/f'{prefix}_delay_latent_rank_scan.csv',latent_rows)
    out={'version':'v18.23','status':status,'source':provenance,
      'observer_windows':HISTORY.tolist(),'future_horizons':HORIZONS.tolist(),
      'noise_sigma_normalized':list(NOISES),'n_train':N_TRAIN,'n_val':N_VAL,
      'n_test_random':N_TEST,'n_test_matched_hidden':1,'train_test_contamination':'NONE_BY_CONSTRUCTION',
      'predictor_inputs':['observed current sample','observed past windows'],
      'predictor_target':'future reduced observation only',
      'predictor_access_to_ODE_Jacobian_cocycle_or_true_state':False,
      'reference_oracle_separate_from_predictor':True,
      'singular_spectra':spectra,'delay_latent_selected_ranks':latent_winners,
      'caveats':[
       'Data are synthesized by original Q550 ODE (server) or frozen tangent propagator (preflight); not real sensors.',
       'Noise is synthetic on perturbation observables normalized by perturbation amplitude.',
       'The fixed material3 sensor is a diagnostic basis identified in earlier ODE-aware work; fitting never uses this basis construction.',
       'One point of the full hidden state cannot be certified by predictive success alone.',
       'Model-blind means no physics model at fit/test time, not no prior simulation or calibration.',
       'Nonlinear state estimates and constitutive hysteresis are NOT inferred.']}
    (results/f'{prefix}_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    core=[r for r in rows if r['noise_sigma']==1e-4]
    print(json.dumps({'status':status,'representative_1e-4':core,'hidden_triples':
      [z for z in hidden_rows if z['noise_sigma']==1e-4 and z['future_horizon']==0.5]},indent=2))
    return out
