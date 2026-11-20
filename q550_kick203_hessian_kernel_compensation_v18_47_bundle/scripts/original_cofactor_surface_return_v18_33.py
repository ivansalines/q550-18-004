#!/usr/bin/env python3
"""Q550 original-ODE cofactor ranking repair and surface-return coherence audit.

No dynamical/physical manifold certificate can be issued by this finite test.
Selection is performed with training scores only; no holdout leakage.
The xi=1.0 clock cycle is explicitly NOT a material Poincare return.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_33 import read
from original_material_invariants_v18_31 import model,select_original_checkpoint,sample_cloud,ANCHORS
from original_invariant_manifold_v18_32 import grid,anchored_flow,SEED
from manifold_math_v18_32 import (features,features_flow,scaled_fit,ensemble_score,
                                    cofactor,project_to_level,surface_test)
from cofactor_return_math_v18_33 import (shortlist,training_rank,normalized_gradient,
    gauge_agnostic_normal_angle,phase_clock_return,surface_accept)
import q550_state_impulse_response_v13 as original

THRESHOLD=1e-6
OUTPUT=ROOT/'results'

def evaluate_samples(f,bases,scale):
    train,trainf,trrej=sample_cloud(bases,scale,56,np.random.default_rng(SEED),[.15,.4,.8],.25,f,include_anchor=True)
    test,testf,terej=sample_cloud(bases,scale,32,np.random.default_rng(SEED+77),[.6,1.5,2.5],.67,f)
    if len(train)!=392 or len(test)!=224:raise RuntimeError('Original original-sampling counts changed')
    return train,trainf,test,testf,trrej,terej

def stream_return(f,start,mode,q,center,scale,theta0,xi0):
    """One original trajectory, three endpoints and a full phase-clock turnover.

    Return is in the phase clock only (if certified modulo 2*pi), not in the
    ten material coordinates. Integrated cofactor uses trapezoidal samples.
    """
    opts=mode; times=np.arange(0.,1.00001,.025)
    sol=solve_ivp(f,(0.,1.),start,t_eval=times,method='DOP853',rtol=4e-12,atol=4e-13,max_step=.02)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise RuntimeError('Original Q550 local time-return solve failed')
    states=sol.y.T
    vals=np.array([features(y,center,scale,theta0,opts['degree'],opts['phase'])@q for y in states])
    mus=np.array([cofactor(y,theta0,opts['lam'],opts['mod']) for y in states])
    area=np.r_[0.,np.cumsum(.5*(mus[1:]+mus[:-1])*np.diff(times))]
    transported=vals[0]*np.exp(area)
    records=[]
    for t in (.25,.5,1.):
        idx=int(round(t/.025));delta=float(states[idx,10]-states[0,10])
        clock=phase_clock_return(delta)
        records.append(dict(xi_start=xi0,delta_xi=t,
            C_start=float(vals[0]),C_end=float(vals[idx]),predicted_C=float(transported[idx]),
            signed_transport_residual=float(vals[idx]-transported[idx]),
            abs_transport_residual=float(abs(vals[idx]-transported[idx])),
            material_displacement_scaled=float(np.linalg.norm((states[idx,:10]-states[0,:10])/scale)),
            delta_phase_rad=delta,**clock))
    return records

def local_overlap(f,opts,q,train,trainf,bases,center,scale,theta0):
    """Fit 3 locally overlapping training windows; compare zero-level normals.

    All local fits use only training states and the same global dictionary.
    They share the same anchor/normalization; no local future/test appears.
    """
    windows=[('left',range(0,4)),('middle',range(2,6)),('right',range(3,7))]
    ref={}; summaries=[]
    for name,inds in windows:
        inds=list(inds)
        ix=np.concatenate([np.arange(i*56,(i+1)*56) for i in inds])
        local,_=scaled_fit(train[ix],trainf[ix],center,scale,theta0,
            anchors=bases[inds],**opts)
        ref[name]=local
    for ix in (2,3,4,5):
        anchor=bases[ix]; z,shift,ok=project_to_level(anchor,q,center,scale,theta0,opts['degree'],opts['phase'])
        if not ok:
            for name in ref:summaries.append(dict(xi=float(ANCHORS[ix]),window=name,
                status='PROJECTION_REJECTED',normal_angle_deg=None,level_offset_scaled=None))
            continue
        g=normalized_gradient(z,q,center,scale,theta0,opts['degree'],opts['phase'])
        for name,local in ref.items():
            gl=normalized_gradient(z,local,center,scale,theta0,opts['degree'],opts['phase'])
            angle=gauge_agnostic_normal_angle(g,gl)
            c=float(features(z,center,scale,theta0,opts['degree'],opts['phase'])@local)
            shiftlen=abs(c)/max(np.linalg.norm(gl),1e-18)
            summaries.append(dict(xi=float(ANCHORS[ix]),window=name,
                status='DIAGNOSTIC_NORMALIZED_COORDINATE_METRIC',
                normal_angle_deg=angle,level_offset_scaled=float(shiftlen),
                global_level_residual=float(features(z,center,scale,theta0,opts['degree'],opts['phase'])@q)))
    return summaries

def main(args):
    data,prov,prior=read()
    d=select_original_checkpoint(data)
    f,_=model(args.base_root,d)
    bases=anchored_flow(f,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=bases[3,:10].copy()
    scale=np.maximum.reduce([.025*np.abs(center),.1*np.ptp(bases[:,:10],axis=0),np.full(10,1e-6)])
    theta0=float(bases[3,10])
    train,trainf,test,testf,trrej,terej=evaluate_samples(f,bases,scale)
    scores=[];qs={};options={};comparison=[]
    history={x['id']:x for x in prior['full_candidate_scan']}
    for opts in grid():
        q,info=scaled_fit(train,trainf,center,scale,theta0,anchors=bases,**opts)
        tr=ensemble_score(train,trainf,q,center,scale,theta0,**opts)
        name=f"{'phasecov' if opts['phase'] else 'phaseblind'}_d{opts['degree']}_lam{opts['lam']:+.2f}_mod{opts['mod']:+.2f}"
        if name not in history:raise RuntimeError('v18.32 training candidate absent: '+name)
        h=history[name]
        discrepancy=abs(float(tr['relative_residual'])-float(h['train_relative_residual']))
        if discrepancy>3e-7:raise RuntimeError('Original v18.32 training replay inconsistency: '+name+' '+str(discrepancy))
        record=dict(id=name,**opts,feature_count=len(q),train_relative_residual=tr['relative_residual'],
            train_residual_rms=tr['residual_rms'],train_field_lie_rms=tr['field_lie_rms'],
            historical_singular_min_max_ratio=h['objective'],historical_training_score_diff=discrepancy,
            selected=False)
        scores.append(record);qs[name]=q;options[name]=opts
    picked=shortlist(scores,per_family=3,include_legacy=True)
    names={p['id'] for p in picked}
    for p in scores:p['selected']=p['id'] in names
    if 'phasecov_d2_lam-1.00_mod+0.00' not in names:raise RuntimeError('Expected overlooked candidate not recovered')
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'original_v18_33_selection_scan.csv').open('w',newline='') as fo:
        w=csv.DictWriter(fo,fieldnames=list(scores[0]));w.writeheader();w.writerows(sorted(scores,key=lambda x:(x['train_relative_residual'],x['id'])))
    overlap=[];flows=[];detail=[];integrations=1;failures=0
    for entry in picked:
        opts=options[entry['id']];q=qs[entry['id']]
        ev=ensemble_score(test,testf,q,center,scale,theta0,**opts)
        anc=np.array([features(y,center,scale,theta0,opts['degree'],opts['phase'])@q for y in bases])
        anchor_rel=float(np.sqrt(np.mean(anc**2))/max(ev['candidate_std'],1e-20))
        posture=[]
        for k,xi in enumerate(ANCHORS):
            sl=slice(32*k,32*(k+1))
            g=ensemble_score(test[sl],testf[sl],q,center,scale,theta0,**opts)
            posture.append(dict(xi=float(xi),relative_residual=g['relative_residual']))
        phase_shift=[]
        for theta_shift in (-.6,.6):
            st=test[::4].copy();st[:,10]+=theta_shift
            ff=np.array([f(0.,s) for s in st])
            g=ensemble_score(st,ff,q,center,scale,theta0,**opts)
            phase_shift.append(dict(delta_theta=float(theta_shift),relative_residual=g['relative_residual']))
        vals,projected,rejected=surface_test(test[::8],f,q,center,scale,theta0,**opts)
        tangent=float(np.sqrt(np.mean([v['surface_residual']**2 for v in vals]))/max(ev['field_lie_rms'],1e-20)) if vals else None
        cross=local_overlap(f,opts,q,train,trainf,bases,center,scale,theta0)
        for x in cross:overlap.append(dict(id=entry['id'],**x))
        # Deterministic sparse projected seeds at three postures, two each.
        successful=0; thisflow=[]
        for k in (1,3,5):
            for idx in (0,1):
                candidate=test[k*32+idx]
                z,shift,ok=project_to_level(candidate,q,center,scale,theta0,opts['degree'],opts['phase'])
                if not ok:
                    failures+=1;continue
                try:
                    transport=stream_return(f,z,opts,q,center,scale,theta0,float(ANCHORS[k]))
                except (FloatingPointError,OverflowError,ValueError,RuntimeError,ZeroDivisionError):
                    failures+=1;continue
                successful+=1;integrations+=1
                for datum in transport:
                    row=dict(id=entry['id'],seed_anchor=float(ANCHORS[k]),seed_index=idx,projection_distance_scaled=shift,**datum)
                    flows.append(row);thisflow.append(row)
        drifts=[r['abs_transport_residual'] for r in thisflow]
        maxdrift=max(drifts) if successful==6 else None
        rel_drift=maxdrift/max(ev['candidate_std'],1e-20) if maxdrift is not None else None
        angles=[r['normal_angle_deg'] for r in cross if r['normal_angle_deg'] is not None]
        offsets=[r['level_offset_scaled'] for r in cross if r['level_offset_scaled'] is not None]
        statuspass=surface_accept(ev['relative_residual'],tangent,rel_drift,anchor_rel,
                   max(x['relative_residual'] for x in phase_shift),max(angles) if angles else None,
                   max(offsets) if offsets else None)
        record=dict(id=entry['id'],opts=opts,training_score=entry['train_relative_residual'],
            historical_singular_objective=entry['historical_singular_min_max_ratio'],
            holdout=ev,anchor_relative=anchor_rel,cross_posture=posture,phase_shift_holdout=phase_shift,
            projected_test_points=len(projected),projected_rejected=rejected,on_surface_relative_tangency=tangent,
            local_window_overlap_max_normal_angle_degrees=max(angles) if angles else None,
            local_window_overlap_max_level_offset_scaled=max(offsets) if offsets else None,
            short_original_integrations_succeeded=successful,
            finite_time_cofactor_return_max_absolute=maxdrift,
            finite_time_cofactor_return_max_relative=rel_drift,
            numerical_local_screen_passed=bool(statuspass),
            clock_cycle_is_not_material_return=True,
            status='NUMERICAL_LOCAL_CANDIDATE_ONLY_NOT_PHYSICAL' if statuspass else 'REJECTED_OR_UNRESOLVED')
        detail.append(record)
        print('CANDIDATE',entry['id'],'train',f'{entry["train_relative_residual"]:.4e}',
              'hold',f'{ev["relative_residual"]:.4e}','overlap',record['local_window_overlap_max_normal_angle_degrees'],
              'return',rel_drift,'pass',statuspass,flush=True)
    def write_json(filename,x):(OUTPUT/filename).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    def csvout(filename,items,empty_columns=None):
        columns=list(items[0]) if items else empty_columns
        if not columns:raise RuntimeError('Diagnostic CSV schema undefined: '+filename)
        with (OUTPUT/filename).open('w',newline='') as fo:
            writer=csv.DictWriter(fo,fieldnames=columns);writer.writeheader();writer.writerows(items)
    write_json('original_v18_33_surface_retests.json',dict(center=center.tolist(),scale=scale.tolist(),theta0=theta0,
        candidates=[dict(id=x['id'],q=qs[x['id']].tolist(),opts=options[x['id']]) for x in picked],diagnostics=detail))
    csvout('original_v18_33_posture_overlap.csv',overlap)
    csvout('original_v18_33_clock_return.csv',flows,empty_columns=[
        'id','seed_anchor','seed_index','projection_distance_scaled','xi_start',
        'delta_xi','C_start','C_end','predicted_C','signed_transport_residual',
        'abs_transport_residual','material_displacement_scaled','delta_phase_rad',
        'phase_turns','phase_wrapping_error_rad','phase_cycle_only',
        'material_recurrence_certified'])
    historypicked=[x['id'] for x in prior['finalists']]
    report=dict(version='v18.33',status='ORIGINAL_Q550_COFACTOR_SELECTION_REPAIR_SURFACE_RETURN_COMPLETED',
        parent_v18_32_sha256=prov['v18_32_server_sha256'],parent_v18_32_manifest_members=prov['v18_32_verified_members'],
        original_ODE_source_SHA256_verified=True,original_ode_integrations=integrations,
        failed_short_flow_attempts=failures,train_states=int(len(train)),holdout_states=int(len(test)),
        candidate_grid_count=len(scores),selection_policy='TRAINING_NORMALIZED_LIE_COFACTOR_RESIDUAL_ONLY',
        no_holdout_in_selection=True,legacy_v18_32_finalists=historypicked,
        repaired_finalists=[x['id'] for x in picked],
        overlooked_phasecov_candidate_included=True,
        candidates_numerically_screened=len(detail),numerically_passing=sum(int(x['numerical_local_screen_passed']) for x in detail),
        candidate_diagnostics=detail,
        posture_window_local_models_train_only=True,
        validation_holdout_excluded_from_fitting=True,
        local_window_models_train_only=True,
        overlapping_window_comparison_not_material_transport_certificate=True,
        xi_1_phase_clock_cycle_is_not_material_recurrence=True,
        numerical_normal_metric_is_diagnostic_only=True,
        no_physically_derived_constraint=True,no_physical_material_manifold_certified=True,
        no_global_surface_return_nor_rigorous_interval_certificate=True,
        off_constraint_probes_only=True,all_models_use_pinned_original_Q550=True)
    write_json('original_v18_33_proof.json',report)
    print('ORIGINAL_V18_33_COMPLETED',integrations,'ODE integrations',len(picked),'finalists',report['numerically_passing'],'numeric passes',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    main(p.parse_args())
