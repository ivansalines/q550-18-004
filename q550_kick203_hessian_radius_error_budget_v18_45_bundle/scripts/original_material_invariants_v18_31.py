#!/usr/bin/env python3
"""Q550 low-degree first-integral search with independent transverse holdouts.

No discovered candidate is certified as a physical/material invariant.
Only an explicitly verified first integral can define material curvature.
All original Q550 code is loaded from the unchanged SHA-pinned server files.
"""
from __future__ import annotations
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_31 import read

def numpy_json_default(value):
    """Serialize NumPy scalar types, retaining strict JSON-native value semantics."""
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.floating):return float(value)
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
from invariant_math_v18_31 import feature,feature_derivative,gradient_hessian,select_modes,derivative_score,tangent_normal_curvature,rms
from hessian_constraint_math_v18_30 import rhs_hessian,second_variational_rhs
from frozen_preflight import observers
from blind_experiment import initial_directions
from checkpoint_io_v18_21 import PREFIX as PREFIX20
import q550_state_impulse_response_v13 as original
import zipfile

ANCHORS=np.array([3.25,3.5,3.75,4.,4.25,4.5,4.75])
SEED=1831
TRAIN_PER_ANCHOR=56
HOLDOUT_PER_ANCHOR=32
DEFECT_THRESHOLD=1e-7

def select_original_checkpoint(data):
    """Select the original tight Q550 state from the two-tier archive.

    The v18.7/v18.6 checkpoint loader returns a mapping indexed by
    'standard' and 'tight'; metadata such as slope belongs to each tier,
    not to the outer mapping. Fail explicitly for incompatible archives.
    """
    if not isinstance(data,dict) or 'tight' not in data:
        raise RuntimeError('Original Q550 checkpoint tiers missing: expected tight and standard')
    d=data['tight']
    if not isinstance(d,dict) or 'slope' not in d or 'source_state_11D' not in d:
        raise RuntimeError('Original Q550 tight checkpoint missing slope or source_state_11D')
    slope=float(d['slope'])
    state=np.asarray(d['source_state_11D'],float)
    if not np.isfinite(slope) or slope<=0 or state.shape!=(11,) or not np.all(np.isfinite(state)):
        raise RuntimeError('Original Q550 tight checkpoint malformed: slope/state shape or values')
    return d

def model(base,d):
    for role,(filename,sha) in original.DEPENDENCIES.items():
        path=base/filename
        if not path.is_file() or original.hash_file(path)!=sha:
            raise RuntimeError('Pinned Q550 dependency absent or SHA mismatch '+str(path))
    audit=original.load_module(base/original.DEPENDENCIES['audit'][0],'v1831_audit')
    support=original.load_module(base/original.DEPENDENCIES['support'][0],'v1831_support')
    tube=original.load_module(base/original.DEPENDENCIES['tube'][0],'v1831_tube')
    params=audit.ModelParameters(gamma_rho=.3)
    k0=(55./6.)*(10./550.);slope=float(d['slope'])
    def f(t,y):return slope*np.asarray(support.support_rhs_float(np.asarray(y,float),kappa0=k0,params=params,audit=audit),float)
    def jac(y):return slope*np.asarray(tube.float_support_jacobian(np.asarray(y,float),kappa0=k0,params=params,audit=audit,base=support),float)
    return f,jac

def sample_cloud(bases,scale,count,rng,multiplier,phase_width,f,include_anchor=False):
    states=[];fields=[];reject=0
    for yref in bases:
        for i in range(count):
            if i==0 and include_anchor:
                yy=np.asarray(yref,float).copy()
            else:
                yy=yref.copy()
                z=rng.normal(size=10)
                z/=max(np.linalg.norm(z),1e-15)
                # Explicit off-trajectory transverse test of the full material field
                yy[:10]+=scale*float(rng.choice(multiplier))*z
                yy[10]+=rng.uniform(-phase_width,phase_width)
            try:
                ff=np.asarray(f(0.,yy),float)
                if ff.shape!=(11,) or not np.all(np.isfinite(ff)):raise ValueError('Nonfinite')
                states.append(yy);fields.append(ff)
            except (FloatingPointError,ValueError,OverflowError,ZeroDivisionError):reject+=1
    return np.array(states),np.array(fields),reject

def load_seed(d,anchor=4.5):
    seeds=json.loads((ROOT/'.runtime/matched_hidden_seeds_v18_20.json').read_text())
    direction=initial_directions({r['name']:r['seed'] for r in seeds})
    return np.asarray(direction['raw6']['matched_seed'],float)

def profile_conservation(f,starts,q,center,scale,degree,rtol=5e-12):
    # Each independent test interval uses an initial state never seen in training.
    vals=[];integrations=0
    for i,y0 in enumerate(starts):
        sol=solve_ivp(f,(0.,.2),y0,t_eval=[0.,.1,.2],method='DOP853',rtol=rtol,atol=8e-13,max_step=.025)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise RuntimeError('Independent validation flow failed: '+sol.message)
        integrations+=1
        c0=float(feature(sol.y[:,0],center,scale,degree)@q)
        vals.extend(abs(float(feature(sol.y[:,j],center,scale,degree)@q)-c0) for j in (1,2))
    return max(vals,default=float('nan')),integrations

def run(args):
    all_checkpoints,prov,prior,prior_profiles,prior_cases=read()
    d=select_original_checkpoint(all_checkpoints)
    f,jac=model(args.base_root,d)
    output=ROOT/'results';output.mkdir(exist_ok=True)
    yinitial=np.asarray(d['source_state_11D'],float)
    orbit=solve_ivp(f,(0.,float(ANCHORS[-1])),yinitial,t_eval=ANCHORS,method='DOP853',rtol=3e-13,atol=3e-14,max_step=.015)
    if not orbit.success:raise RuntimeError('Original Q550 orbit failed: '+orbit.message)
    bases=orbit.y.T
    center=bases[3,:10].copy()
    empirical_range=np.ptp(bases[:,:10],axis=0)
    scale=np.maximum.reduce([.025*np.abs(center),.1*empirical_range,np.full(10,1e-6)])
    rng1=np.random.default_rng(SEED)
    rng2=np.random.default_rng(SEED+77)
    train_y,train_f,trrej=sample_cloud(bases,scale,TRAIN_PER_ANCHOR,rng1,[.15,.4,.8],.25,f,include_anchor=True)
    test_y,test_f,terej=sample_cloud(bases,scale,HOLDOUT_PER_ANCHOR,rng2,[.6,1.5,2.5],.67,f)
    if len(train_y)<200 or len(test_y)<100:raise RuntimeError('Insufficient finite transverse model states')
    clock=np.r_[train_f[:,10],test_f[:,10]]
    clock_abs_spread=float(np.ptp(clock))
    # Sample f at physically different states; clock constancy is NOT a material integral.
    scored=[];fitted=[];integrations=1;all_candidates=[]
    for degree in (1,2):
        aa=np.vstack([feature_derivative(y,g,center,scale,degree) for y,g in zip(train_y,train_f)])
        bb=np.vstack([feature_derivative(y,g,center,scale,degree) for y,g in zip(test_y,test_f)])
        s,modes=select_modes(aa,cutoff=2e-6,max_candidates=4)
        scored.append(dict(degree=degree,feature_count=aa.shape[1],train_singular_values=s.tolist(),
               smallest_relative_singular=float(s[-1]/max(s[0],1e-30)),candidate_count=len(modes)))
        for mode in modes:
            q=mode['coeff']
            tr=derivative_score(aa,q);te=derivative_score(bb,q)
            train_raw=rms(aa@q);hold_raw=rms(bb@q)
            entry=dict(degree=degree,rank_index=mode['singular_index'],singular_ratio=mode['sratio'],
                       train_relative_derivative=tr,holdout_relative_derivative=te,
                       train_derivative_rms=train_raw,holdout_derivative_rms=hold_raw,
                       cross_phase_and_posture_holdout=True)
            entry['candidate_passes_transverse']=(tr<DEFECT_THRESHOLD and te<DEFECT_THRESHOLD and mode['sratio']<DEFECT_THRESHOLD)
            entry['status']='TRANSVERSE_CANDIDATE_REQUIRES_CONSERVATION_TEST' if entry['candidate_passes_transverse'] else 'REJECTED_TRANSVERSE'
            if entry['candidate_passes_transverse']:
                pick=[test_y[k] for k in np.linspace(0,len(test_y)-1,4,dtype=int)]
                drift,nruns=profile_conservation(f,pick,q,center,scale,degree)
                integrations+=nruns
                values=np.array([feature(y,center,scale,degree)@q for y in test_y])
                spread=float(np.std(values))
                entry.update(independent_trajectory_drift_max=drift,transverse_candidate_spread=spread,
                             drift_relative_to_spread=drift/max(spread,1e-30))
                if drift/max(spread,1e-30)<1e-6 and spread>1e-15:
                    entry['status']='NUMERIC_CANDIDATE_ONLY_NOT_PHYSICAL_CERTIFICATE'
                    all_candidates.append((entry,q))
                else:entry['status']='REJECTED_INDEPENDENT_TRAJECTORY_DRIFT'
            fitted.append(entry)
    # Second-order geometry is computed ONLY for numerically surviving candidates.
    geometry=[]
    seed=load_seed(d)
    for entry,q in all_candidates[:2]:
        a=4.5;y0=bases[np.argmin(abs(ANCHORS-a))]
        degree=entry['degree']
        u,w0,geom=tangent_normal_curvature(y0,seed,center,scale,q,degree)
        Y=np.r_[y0,u,w0]
        # This is a CONDITIONAL level-set calculation, not a material certificate.
        fun=second_variational_rhs(f,jac,.02)
        sol=solve_ivp(fun,(a,a+.2),Y,t_eval=np.linspace(a,a+.2,9),method='DOP853',rtol=3e-13,atol=3e-14,max_step=.015)
        if not sol.success:raise RuntimeError('Conditional curvature variational solve failed')
        integrations+=1
        residuals=[];flow_defect=[];normal_hessian=[]
        for z in sol.y.T:
            yy=z[:11];uu=z[11:21];ww=z[21:31]
            G,HC=gradient_hessian(yy,center,scale,q,degree)
            residuals.append([float(abs(G@uu)),float(abs(G@ww+uu@HC@uu))])
            flow_defect.append(float(abs(G@f(0.,yy)[:10])))
            src=rhs_hessian(f,0.,yy,np.r_[uu,0.],.02)[:10]
            normal_hessian.append(float((G/np.linalg.norm(G))@src))
        geometry.append(dict(degree=degree,rank_index=entry['rank_index'],status='CONDITIONAL_LEVEL_SET_DIAGNOSTIC_NOT_CERTIFIED_MATERIAL',
                     initial_jet=geom,max_first_constraint_defect=float(np.max(np.array(residuals)[:,0])),
                     max_second_constraint_defect=float(np.max(np.array(residuals)[:,1])),
                     max_flow_tangency_defect=float(max(flow_defect)),
                     normal_component_source_rms=rms(normal_hessian)))
    # Store the exact fitted expressions and normalization so future checks can be reproduced.
    coeffs=[]
    for entry,q in all_candidates:
        coeffs.append(dict(degree=entry['degree'],rank_index=entry['rank_index'],q=q.tolist()))
    (output/'original_discovered_candidate_coefficients_v18_31.json').write_text(json.dumps(dict(center=center.tolist(),scale=scale.tolist(),coefficients=coeffs),indent=2)+'\n')
    with (output/'original_invariant_candidates_v18_31.csv').open('w',newline='') as fp:
        fields=['degree','rank_index','singular_ratio','train_relative_derivative','holdout_relative_derivative','train_derivative_rms','holdout_derivative_rms','cross_phase_and_posture_holdout','candidate_passes_transverse','independent_trajectory_drift_max','transverse_candidate_spread','drift_relative_to_spread','status']
        writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader();writer.writerows(fitted)
    (output/'original_constraint_normal_curvature_v18_31.json').write_text(json.dumps(dict(status='CONDITIONAL_ONLY_NO_PHYSICAL_INVARIANT_CERTIFICATE',geometry=geometry),indent=2)+'\n')
    report=dict(version='v18.31',status='ORIGINAL_Q550_MATERIAL_INVARIANT_DISCOVERY_AND_CONDITIONAL_CURVATURE_COMPLETED',
      parent_sha256=prov['v18_30_server_SHA256'],parent_manifest_members=prov['v18_30_verified_members'],
      pinned_original_ODE_SHA256_verified=True,original_ode_integrations=integrations,
      baseline_anchors=ANCHORS.tolist(),train_states=int(len(train_y)),holdout_states=int(len(test_y)),
      rejected_nonfinite_train=int(trrej),rejected_nonfinite_holdout=int(terej),
      independent_offtrajectory_states=True,heldout_phase_shifts=True,
      clock_rhs_median=float(np.median(clock)),clock_rhs_spread=clock_abs_spread,
      clock_is_not_material_invariant=True,material_state_dimension=10,
      feature_dictionary='LINEAR_AND_FULL_QUADRATIC_MATERIAL_POLYNOMIALS_IN_DECLARED_NORMALIZED_COORDINATES',
      discovery= scored,candidates_tested=len(fitted),candidates_surviving_numeric=len(all_candidates),
      candidate_details=fitted,conditional_curvature=geometry,
      material_constraint_status='NO_PHYSICAL_MATERIAL_INVARIANT_CERTIFIED',
      material_curvature_certificate=False,no_claim_of_global_exhaustiveness=True,
      diagnostic_seeds_off_constraint=True,noise_or_truncation_may_create_false_candidates=True,
      no_physical_flip_or_memory_claim=True)
    (output/'original_invariant_discovery_proof_v18_31.json').write_text(json.dumps(report,indent=2,default=numpy_json_default,allow_nan=False)+'\n')
    print('ORIGINAL_V18_31_COMPLETED',integrations,'ODE integrations',len(train_y),len(test_y),'states','surviving',len(all_candidates),flush=True)
    for item in fitted: print('candidate',item['degree'],item['rank_index'],'train',f'{item["train_relative_derivative"]:.3e}','hold',f'{item["holdout_relative_derivative"]:.3e}',item['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    args=p.parse_args();run(args)
