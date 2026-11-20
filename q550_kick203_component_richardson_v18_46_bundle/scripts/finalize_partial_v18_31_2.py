#!/usr/bin/env python3
"""Recover a v18.31 original run stopped solely by JSON NumPy-bool encoding.

Requires all three *previously written* result intermediates. Recomputes and
cross-verifies the deterministic original Q550 source sampling and singular
spectra. Does NOT rerun the previously completed independent validation flows
or conditional curvature flows. Fails closed on mismatches or missing outputs.
"""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import original_material_invariants_v18_31 as orig
from invariant_math_v18_31 import feature_derivative, select_modes, derivative_score, rms

R=ROOT/'results'
CSV=R/'original_invariant_candidates_v18_31.csv'
COEFF=R/'original_discovered_candidate_coefficients_v18_31.json'
GEOM=R/'original_constraint_normal_curvature_v18_31.json'
PROOF=R/'original_invariant_discovery_proof_v18_31.json'

def _load_partial():
    absent=[str(path) for path in (CSV,COEFF,GEOM) if not path.is_file()]
    if absent:raise RuntimeError('Cannot recover: missing completed original-run intermediates: '+', '.join(absent))
    with CSV.open(newline='') as fp:
        details=list(csv.DictReader(fp))
    coeff=json.loads(COEFF.read_text())
    geometry=json.loads(GEOM.read_text())
    if geometry.get('status')!='CONDITIONAL_ONLY_NO_PHYSICAL_INVARIANT_CERTIFICATE':
        raise RuntimeError('Unrecognized conditional geometry status')
    if 'center' not in coeff or 'scale' not in coeff or 'coefficients' not in coeff:
        raise RuntimeError('Incomplete original fitted coefficient archive')
    return details,coeff,geometry['geometry']

def _close(actual,expected,name):
    if not np.isclose(float(actual),float(expected),rtol=1e-6,atol=1e-13):
        raise RuntimeError(f'Stored original candidate does not match deterministic replay: {name}: {actual} != {expected}')

def run():
    details,coeff,geometry=_load_partial()
    checkpoints,prov,_,_,_=orig.read()
    d=orig.select_original_checkpoint(checkpoints)
    f,_=orig.model(orig.original.DEFAULT_BASE,d)
    state=np.asarray(d['source_state_11D'],float)
    orbit=solve_ivp(f,(0.,float(orig.ANCHORS[-1])),state,t_eval=orig.ANCHORS,method='DOP853',rtol=3e-13,atol=3e-14,max_step=.015)
    if not orbit.success:raise RuntimeError('Deterministic reconstruction failed: '+orbit.message)
    bases=orbit.y.T
    center=bases[3,:10].copy()
    spread=np.ptp(bases[:,:10],axis=0)
    scale=np.maximum.reduce([.025*np.abs(center),.1*spread,np.full(10,1e-6)])
    if not np.allclose(center,coeff['center'],rtol=1e-12,atol=1e-12) or not np.allclose(scale,coeff['scale'],rtol=1e-12,atol=1e-12):
        raise RuntimeError('Original saved normalization disagrees with independently recomputed Q550 anchors')
    tr_y,tr_f,tr_reject=orig.sample_cloud(bases,scale,orig.TRAIN_PER_ANCHOR,np.random.default_rng(orig.SEED),[.15,.4,.8],.25,f,include_anchor=True)
    te_y,te_f,te_reject=orig.sample_cloud(bases,scale,orig.HOLDOUT_PER_ANCHOR,np.random.default_rng(orig.SEED+77),[.6,1.5,2.5],.67,f)
    if len(tr_y)<200 or len(te_y)<100:raise RuntimeError('Too few finite recomputed original states')
    replay=[];scored=[];surviving=[];trajectory_checks=0
    for degree in (1,2):
        A=np.vstack([feature_derivative(y,v,center,scale,degree) for y,v in zip(tr_y,tr_f)])
        B=np.vstack([feature_derivative(y,v,center,scale,degree) for y,v in zip(te_y,te_f)])
        s,modes=select_modes(A,cutoff=2e-6,max_candidates=4)
        scored.append(dict(degree=degree,feature_count=A.shape[1],train_singular_values=s.tolist(),smallest_relative_singular=float(s[-1]/max(s[0],1e-30)),candidate_count=len(modes)))
        for mode in modes:
            q=mode['coeff']
            tr=derivative_score(A,q);te=derivative_score(B,q)
            approved=bool(tr<orig.DEFECT_THRESHOLD and te<orig.DEFECT_THRESHOLD and mode['sratio']<orig.DEFECT_THRESHOLD)
            if approved:trajectory_checks+=4
            replay.append(dict(degree=degree,rank_index=mode['singular_index'],singular_ratio=mode['sratio'],
                train_relative_derivative=tr,holdout_relative_derivative=te,
                train_derivative_rms=rms(A@q),holdout_derivative_rms=rms(B@q),
                candidate_passes_transverse=approved,q=q))
    if len(details)!=len(replay):
        raise RuntimeError(f'Candidate count mismatch: stored {len(details)}, replay {len(replay)}')
    typed=[]
    keys=('singular_ratio','train_relative_derivative','holdout_relative_derivative','train_derivative_rms','holdout_derivative_rms')
    for n,(stored,actual) in enumerate(zip(details,replay)):
        if int(stored['degree'])!=actual['degree'] or int(stored['rank_index'])!=actual['rank_index']:
            raise RuntimeError(f'Candidate index changed at row {n}')
        if stored['candidate_passes_transverse'].strip().lower() != str(actual['candidate_passes_transverse']).lower():
            raise RuntimeError(f'Transverse candidate decision changed at row {n}')
        for name in keys:_close(stored[name],actual[name],f'row {n} {name}')
        status=stored['status']
        allowed={'REJECTED_TRANSVERSE','REJECTED_INDEPENDENT_TRAJECTORY_DRIFT','NUMERIC_CANDIDATE_ONLY_NOT_PHYSICAL_CERTIFICATE'}
        if status not in allowed:raise RuntimeError(f'Unrecognized stored selection status: {status}')
        if (status=='REJECTED_TRANSVERSE')==actual['candidate_passes_transverse']:
            raise RuntimeError('Candidate status and selection disagree')
        if actual['candidate_passes_transverse']:
            for name in ('independent_trajectory_drift_max','transverse_candidate_spread','drift_relative_to_spread'):
                if not stored[name]:raise RuntimeError('Missing completed independent drift test '+name)
            ratio=float(stored['drift_relative_to_spread'])
            spread=float(stored['transverse_candidate_spread'])
            good=ratio<1e-6 and spread>1e-15
            if good != (status=='NUMERIC_CANDIDATE_ONLY_NOT_PHYSICAL_CERTIFICATE'):
                raise RuntimeError('Stored independent drift decision inconsistent')
        entry=dict(degree=int(stored['degree']),rank_index=int(stored['rank_index']),
            singular_ratio=float(stored['singular_ratio']),
            train_relative_derivative=float(stored['train_relative_derivative']),
            holdout_relative_derivative=float(stored['holdout_relative_derivative']),
            train_derivative_rms=float(stored['train_derivative_rms']),
            holdout_derivative_rms=float(stored['holdout_derivative_rms']),
            cross_phase_and_posture_holdout=True,
            candidate_passes_transverse=actual['candidate_passes_transverse'],status=status)
        if actual['candidate_passes_transverse']:
            entry.update(independent_trajectory_drift_max=float(stored['independent_trajectory_drift_max']),
                         transverse_candidate_spread=float(stored['transverse_candidate_spread']),
                         drift_relative_to_spread=float(stored['drift_relative_to_spread']))
        typed.append(entry)
    survivors=[x for x in typed if x['status']=='NUMERIC_CANDIDATE_ONLY_NOT_PHYSICAL_CERTIFICATE']
    if len(survivors)!=len(coeff['coefficients']):raise RuntimeError('Surviving coefficient count mismatch')
    if len(geometry)!=min(2,len(survivors)):raise RuntimeError('Original completed geometry count mismatch')
    for val,sv in zip(coeff['coefficients'],survivors):
        if val['degree']!=sv['degree'] or val['rank_index']!=sv['rank_index']:
            raise RuntimeError('Stored survivor identity mismatch')
        matching=[r for r in replay if r['degree']==sv['degree'] and r['rank_index']==sv['rank_index']]
        if len(matching)!=1:raise RuntimeError('Missing matching candidate direction')
        a=np.asarray(val['q'],float);b=np.asarray(matching[0]['q'],float)
        if min(np.linalg.norm(a-b),np.linalg.norm(a+b))>1e-5:
            raise RuntimeError('Stored survivor coefficients differ from replay')
    for val,sv in zip(geometry,survivors):
        if val['degree']!=sv['degree'] or val['rank_index']!=sv['rank_index']:
            raise RuntimeError('Stored conditional geometry identity mismatch')
    clk=np.r_[tr_f[:,10],te_f[:,10]]
    n_integrations=1+trajectory_checks+len(geometry)
    report=dict(version='v18.31',status='ORIGINAL_Q550_MATERIAL_INVARIANT_DISCOVERY_AND_CONDITIONAL_CURVATURE_COMPLETED',
      parent_sha256=prov['v18_30_server_SHA256'],parent_manifest_members=prov['v18_30_verified_members'],
      pinned_original_ODE_SHA256_verified=True,original_ode_integrations=n_integrations,
      baseline_anchors=orig.ANCHORS.tolist(),train_states=int(len(tr_y)),holdout_states=int(len(te_y)),
      rejected_nonfinite_train=int(tr_reject),rejected_nonfinite_holdout=int(te_reject),
      independent_offtrajectory_states=True,heldout_phase_shifts=True,
      clock_rhs_median=float(np.median(clk)),clock_rhs_spread=float(np.ptp(clk)),
      clock_is_not_material_invariant=True,material_state_dimension=10,
      feature_dictionary='LINEAR_AND_FULL_QUADRATIC_MATERIAL_POLYNOMIALS_IN_DECLARED_NORMALIZED_COORDINATES',
      discovery=scored,candidates_tested=len(typed),candidates_surviving_numeric=len(survivors),
      candidate_details=typed,conditional_curvature=geometry,
      material_constraint_status='NO_PHYSICAL_MATERIAL_INVARIANT_CERTIFIED',
      material_curvature_certificate=False,no_claim_of_global_exhaustiveness=True,
      diagnostic_seeds_off_constraint=True,noise_or_truncation_may_create_false_candidates=True,
      no_physical_flip_or_memory_claim=True,
      serialization_recovery='PARTIAL_ORIGINAL_RESULTS_VALIDATED_WITH_NEW_DETERMINISTIC_FIELD_SAMPLES',
      recovered_completed_original_integration_count=n_integrations,
      recovery_original_anchor_reintegrations=1,
      recovery_prior_intermediate_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (CSV,COEFF,GEOM)})
    # Refuse non-finite JSON instead of silently writing NaNs to a purported proof.
    encoded=json.dumps(report,indent=2,default=orig.numpy_json_default,allow_nan=False)+'\n'
    PROOF.write_text(encoded)
    print('V18_31_2_RECOVERED',n_integrations,'original flows from stored results; 1 anchor replay;',
          'candidates',len(typed),'survivors',len(survivors),flush=True)

if __name__=='__main__':run()
