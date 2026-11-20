#!/usr/bin/env python3
"""Discover phase-covariant local invariant hypersurface candidates in original Q550.

Numerical tests of L_F C = mu(theta) C, including independent off-manifold
holdouts, projected zero-level tangency, short flow transport, and phase shifts.
No candidate is promoted to a physical constraint without explicit certificates.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_32 import read
from original_material_invariants_v18_31 import select_original_checkpoint,model,sample_cloud,ANCHORS
from manifold_math_v18_32 import (features,features_flow,features_material_gradient,cofactor,
    scaled_fit,ensemble_score,surface_test,project_to_level)
import q550_state_impulse_response_v13 as original

THRESHOLD=1.e-6
SEED=1832

def grid():
    for phase in (False,True):
        for degree in (1,2):
            for lam in (-1.,-.5,0.,.5,1.):
                for mod in ((0.,-.3,.3) if phase else (0.,)):
                    yield dict(phase=phase,degree=degree,lam=lam,mod=mod)

def anchored_flow(f,y0,anchors):
    sol=solve_ivp(f,(0.,float(max(anchors))),y0,t_eval=anchors,method='DOP853',rtol=3e-13,atol=3e-14,max_step=.015)
    if not sol.success or sol.y.shape!=(11,len(anchors)):raise RuntimeError('Baseline Q550 original ODE integration failed')
    return sol.y.T

def drift_test(f,z,q,center,scale,theta0,degree,phase,dt=.25):
    sol=solve_ivp(f,(0.,dt),z,t_eval=np.linspace(0.,dt,9),method='DOP853',rtol=3e-12,atol=3e-13,max_step=.02)
    if not sol.success:raise RuntimeError('Original Q550 short surface-transport solve failed')
    cs=np.array([features(y,center,scale,theta0,degree,phase)@q for y in sol.y.T])
    return float(np.max(np.abs(cs-cs[0]))), float(max(np.abs(cs))),1

def run(args):
    chk,provenance,prior=read()
    d=select_original_checkpoint(chk)
    f,jac=model(args.base_root,d)
    baseline=anchored_flow(f,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=baseline[3,:10].copy()
    scale=np.maximum.reduce([.025*np.abs(center),.1*np.ptp(baseline[:,:10],axis=0),np.full(10,1e-6)])
    theta0=float(baseline[3,10])
    train,trainf,trrej=sample_cloud(baseline,scale,56,np.random.default_rng(SEED),[.15,.4,.8],.25,f,include_anchor=True)
    test,testf,terej=sample_cloud(baseline,scale,32,np.random.default_rng(SEED+77),[.6,1.5,2.5],.67,f)
    if len(train)!=392 or len(test)!=224:raise RuntimeError('Unexpected sampled-state count; do not mix anchors')
    scores=[];coefficients={}
    # Candidate selection is based strictly on the training objective.
    for opts in grid():
        q,info=scaled_fit(train,trainf,center,scale,theta0,anchors=baseline,**opts)
        tr=ensemble_score(train,trainf,q,center,scale,theta0,**opts)
        name=f"{'phasecov' if opts['phase'] else 'phaseblind'}_d{opts['degree']}_lam{opts['lam']:+.2f}_mod{opts['mod']:+.2f}"
        rec=dict(id=name,**opts,**info,train_relative_residual=tr['relative_residual'],
                 train_residual_rms=tr['residual_rms'],train_C_std=tr['candidate_std'],
                 train_field_lie_rms=tr['field_lie_rms'])
        rec['objective']=float(info['singular_min']/max(info['singular_max'],1e-30))
        scores.append(rec);coefficients[name]=q
    finalists=[]
    for ph in (False,True):
        possibilities=[a for a in scores if a['phase']==ph]
        finalists.extend(sorted(possibilities,key=lambda a:a['objective'])[:2])
    detailed=[];tab=[];integrations=1
    # All phase/posture/transverse tests are strictly out of training.
    for item in finalists:
        opts={k:item[k] for k in ('phase','degree','lam','mod')}
        q=coefficients[item['id']]
        ev=ensemble_score(test,testf,q,center,scale,theta0,**opts)
        anc=np.array([features(y,center,scale,theta0,opts['degree'],opts['phase'])@q for y in baseline])
        anchor_rms=float(np.sqrt(np.mean(anc*anc)))
        groups=[]
        for i,a in enumerate(ANCHORS):
            subset=slice(32*i,32*(i+1)); b=ensemble_score(test[subset],testf[subset],q,center,scale,theta0,**opts)
            groups.append(dict(xi=float(a),relative_residual=b['relative_residual'],residual_rms=b['residual_rms']))
        # Phase offsets are applied to actual Q550 state before re-evaluating the RHS.
        shift_groups=[]
        for shift in (-.6,.6):
            shifted=test[::4].copy();shifted[:,10]+=shift
            ff=np.array([f(0.,y) for y in shifted])
            z=ensemble_score(shifted,ff,q,center,scale,theta0,**opts)
            shift_groups.append(dict(phase_shift=float(shift),relative_residual=z['relative_residual']))
        vals,projected,reject=surface_test(test[::8],f,q,center,scale,theta0,**opts)
        tangent_rms=float(np.sqrt(np.mean([v['surface_residual']**2 for v in vals]))) if vals else None
        tangent_rel=tangent_rms/max(ev['field_lie_rms'],1e-20) if tangent_rms is not None else None
        drift=[];failed_integrations=0
        for z in projected[:6]:
            try:
                dv,mx,n=drift_test(f,z,q,center,scale,theta0,opts['degree'],opts['phase'])
                integrations+=n;drift.append(dv)
            except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError,RuntimeError):
                failed_integrations+=1
        drift_max=float(max(drift)) if drift and failed_integrations==0 else None
        # A conservative three-part numerical screen; not a physical certificate.
        pass_numeric=(ev['relative_residual']<THRESHOLD
            and (tangent_rel is not None and tangent_rel<THRESHOLD)
            and (drift_max is not None and drift_max/max(ev['candidate_std'],1e-20)<THRESHOLD)
            and anchor_rms/max(ev['candidate_std'],1e-20)<THRESHOLD
            and max(x['relative_residual'] for x in shift_groups)<THRESHOLD)
        diag=dict(**{k:item[k] for k in ('id','phase','degree','lam','mod','objective','feature_count','train_relative_residual')},
            holdout=ev,anchor_rms=anchor_rms,anchor_relative=anchor_rms/max(ev['candidate_std'],1e-20),
            cross_posture=groups,phase_shifts=shift_groups,projection_attempts=len(test[::8]),
            projection_accepted=len(projected),projection_failed=reject,
            short_flow_failures=failed_integrations,
            on_surface_tangency_rms=tangent_rms,on_surface_tangency_relative=tangent_rel,
            finite_time_surface_drift_max=drift_max,
            finite_time_drift_relative=None if drift_max is None else drift_max/max(ev['candidate_std'],1e-20),
            numerical_screen_passed=bool(pass_numeric),
            status='NUMERICAL_LOCAL_HYPERSURFACE_CANDIDATE_NOT_PHYSICAL' if pass_numeric else 'REJECTED_OR_UNRESOLVED_LOCAL_HYPERSURFACE')
        # Strictly diagnostic curvature of the fitted level set, not physical.
        if projected:
            z=projected[0]
            grad=features_material_gradient(z,center,scale,theta0,opts['degree'],opts['phase']).T@q
            geomgrad=grad*scale
            diag['material_gradient_norm']=float(np.linalg.norm(geomgrad))
        detailed.append(diag)
        for group in groups:
            tab.append(dict(id=item['id'],phasecov=int(opts['phase']),xi=group['xi'],
                train_rel=item['train_relative_residual'],test_relative=group['relative_residual'],
                holdout_all=ev['relative_residual'],on_surface_tangent=tangent_rel,
                drift_relative=diag['finite_time_drift_relative'],status=diag['status']))
    status='ORIGINAL_Q550_PHASE_COVARIANT_MANIFOLD_DIAGNOSTIC_COMPLETED'
    output=ROOT/'results';output.mkdir(exist_ok=True)
    coeff=dict(center=center.tolist(),scale=scale.tolist(),theta_reference=theta0,
        coefficients=[dict(id=item['id'],q=coefficients[item['id']].tolist()) for item in finalists],
        feature_specification='material linear + quadratic + x*sin(theta-theta0) + x*cos(theta-theta0); no clock-only functions')
    (output/'original_manifold_candidates_v18_32.json').write_text(json.dumps(coeff,indent=2)+'\n')
    with (output/'original_manifold_posture_transfer_v18_32.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(tab[0]));w.writeheader();w.writerows(tab)
    report=dict(version='v18.32',status=status,parent_v18_31_original_sha256=provenance['v18_31_server_SHA256'],
        parent_v18_31_verified_members=provenance['v18_31_verified_members'],
        original_ODE_source_SHA256_verified=True,original_ode_integrations=integrations,
        original_ode_field_evaluations_on_616_sampled_states=True,
        train_states=392,holdout_states=224,train_rejected=int(trrej),holdout_rejected=int(terej),
        initial_state_from_authentic_Q550=True,baseline_anchors=ANCHORS.tolist(),
        phase_model='sin(theta-theta_anchor); exact phase derivative included in Lie operator',
        cofactor_model='mu(theta)=lambda+mod*sin(theta-theta_anchor); parameter grid preregistered',
        dictionary_excludes_phase_only=True,
        anchor_zero_penalty_weight=0.2,
        candidate_selection_training_only=True,candidate_grid_count=len(scores),
        shortlisted_count=len(finalists),numerical_passed=int(sum(d['numerical_screen_passed'] for d in detailed)),
        full_candidate_scan=scores,finalists=detailed,
        numerical_acceptance_relative=THRESHOLD,
        numerical_screen_is_not_material_certificate=True,
        material_constraint_status='NO_PHYSICAL_MATERIAL_MANIFOLD_CERTIFIED',
        candidate_level_set_projection_uses_diagnostic_normalized_coordinate_metric=True,
        cofactor_criterion_is_sufficient_for_level_set_invariance_only_if_exact_on_open_set=True,
        local_flowbox_artifacts_not_excluded_without_global_recurrence_tests=True,
        finite_polynomial_phase_dictionary_not_exhaustive=True,
        no_intrinsic_torsion_flip_or_constitutive_memory_claim=True)
    (output/'original_invariant_manifold_proof_v18_32.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ORIGINAL_V18_32_COMPLETED',integrations,'integrations',len(scores),'grid modes',len(finalists),'finalists','numerical_passed',report['numerical_passed'],flush=True)
    for x in detailed:print(x['id'],'holdout',f"{x['holdout']['relative_residual']:.3e}",x['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    run(p.parse_args())
