#!/usr/bin/env python3
"""Original Q550 accumulated cofactor-defect audit, no surface certificate.

Reuses the seven *unchanged* v18.33 surface coefficients, choosing the two
training-first degree-2 lambda=-1 surfaces prior to any new holdout analysis.
Original v18.33 projected seeds are replayed deterministically; their
endpoint return values are independently checked before reporting budgets.

The budget identity is algebraic: verifying it does not certify an invariant
surface. Coordinate-normal angles use a nonprivileged diagnostic metric.
"""
import argparse,csv,io,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_34 import read_parent
from original_material_invariants_v18_31 import select_original_checkpoint,model,ANCHORS
from original_invariant_manifold_v18_32 import anchored_flow
from original_cofactor_surface_return_v18_33 import evaluate_samples
from manifold_math_v18_32 import (features,features_flow,cofactor,project_to_level,scaled_fit,features_material_gradient)
from defect_budget_math_v18_34 import integrate_budget,segment_budget,first_crossing,unoriented_angle
import q550_state_impulse_response_v13 as original

SELECTED=['phasecov_d2_lam-1.00_mod+0.00','phaseblind_d2_lam-1.00_mod+0.00']
T=np.linspace(0.,1.,81)
SEGMENTS=[(0.,.1),(.1,.25),(.25,.5),(.5,1.)]

def write_csv(path,rows):
    if not rows:raise RuntimeError('Cannot write empty rows')
    with path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect_angle_table(train,trainf,bases,center,scale,theta0,candidates):
    """Refit overlapping training windows, compare normals under declared metrics."""
    from cofactor_return_math_v18_33 import normalized_gradient
    metrics={'original_uniform':np.ones(10),'alternate_diagonal':np.linspace(.7,1.3,10),
             'inverse_diagonal':1/np.linspace(.7,1.3,10)}
    table=[]
    win={'left':range(0,4),'middle':range(2,6),'right':range(3,7)}
    for it in candidates:
        opts=it['opts'];q=np.asarray(it['q'],float)
        fitted={}
        for name,indices in win.items():
            inds=list(indices); ix=np.concatenate([np.arange(i*56,(i+1)*56) for i in inds])
            fitted[name],_=scaled_fit(train[ix],trainf[ix],center,scale,theta0,anchors=bases[inds],**opts)
        for i in (2,3,4,5):
            z,dist,ok=project_to_level(bases[i],q,center,scale,theta0,opts['degree'],opts['phase'])
            if not ok:raise RuntimeError('Original overlap point projection failed')
            grad=normalized_gradient(z,q,center,scale,theta0,opts['degree'],opts['phase'])
            for name,ql in fitted.items():
                gl=normalized_gradient(z,ql,center,scale,theta0,opts['degree'],opts['phase'])
                for met,w in metrics.items():
                    ang=unoriented_angle(grad,gl,w)
                    table.append(dict(id=it['id'],xi=float(ANCHORS[i]),window=name,
                                      diagnostic_metric=met,angle_deg=ang,
                                      global_sign_flip_angle_deg=unoriented_angle(-grad,gl,w),
                                      local_model_sign_flip_angle_deg=unoriented_angle(grad,-gl,w)))
    return table

def run(args):
    checkpoints,old,model_data,archived_csv,archived_overlap,prov=read_parent()
    d=select_original_checkpoint(checkpoints)
    f,jac=model(args.base_root,d)
    bases=anchored_flow(f,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=np.asarray(model_data['center'],float);scale=np.asarray(model_data['scale'],float)
    theta0=float(model_data['theta0'])
    if not np.allclose(center,bases[3,:10],rtol=5e-9,atol=1e-8):
        raise RuntimeError('Authentic center not reproduced')
    expected_scale=np.maximum.reduce([.025*np.abs(center),.1*np.ptp(bases[:,:10],axis=0),np.full(10,1e-6)])
    if not np.allclose(expected_scale,scale,rtol=5e-9,atol=1e-8):
        raise RuntimeError('Authentic diagnostic scales not reproduced')
    train,trainf,test,testf,_,_=evaluate_samples(f,bases,scale)
    index={(r['id'],float(r['seed_anchor']),int(r['seed_index']),float(r['delta_xi'])):r
           for r in csv.DictReader(io.StringIO(archived_csv))}
    candidates={it['id']:it for it in model_data['candidates']}
    if any(name not in candidates for name in SELECTED):raise RuntimeError('Prespecified v18.33 surface missing')
    samples=[];segments=[];summaries=[];integration_count=1;max_validation_error=0.;max_closure_error=0.
    for name in SELECTED:
        item=candidates[name];q=np.asarray(item['q'],float);opts=item['opts']
        for k in (1,3,5):
            for idx in (0,1):
                start,project_dist,ok=project_to_level(test[k*32+idx],q,center,scale,theta0,opts['degree'],opts['phase'])
                if not ok:raise RuntimeError('Original v18.33 seeded projection not reproducible')
                sol=solve_ivp(f,(0.,1.),start,t_eval=T,method='DOP853',rtol=4e-12,atol=4e-13,max_step=.02)
                if not sol.success or not np.all(np.isfinite(sol.y)):raise RuntimeError('Original pinned Q550 solve failed: '+sol.message)
                integration_count+=1
                states=sol.y.T
                vals=np.array([features(y,center,scale,theta0,opts['degree'],opts['phase'])@q for y in states])
                fields=np.array([f(0.,y) for y in states])
                dvals=np.array([features_flow(y,g,center,scale,theta0,opts['degree'],opts['phase'])@q for y,g in zip(states,fields)])
                mus=np.array([cofactor(y,theta0,opts['lam'],opts['mod']) for y in states])
                budget=integrate_budget(T,vals,dvals,mus)
                distances=np.linalg.norm((states[:,:10]-start[:10])/scale,axis=1)
                for endpoint in (.25,.5,1.):
                    key=(name,float(ANCHORS[k]),idx,endpoint)
                    rec=index[key];jj=int(round(endpoint/.0125))
                    error=abs(float(rec['C_end'])-vals[jj]);max_validation_error=max(max_validation_error,error)
                    if error>1e-6*max(1.,abs(vals[jj])):raise RuntimeError('Original v18.33 trajectory endpoint not reproduced '+str(key)+' '+str(error))
                    endpoint_closure=abs(budget['signed_simpson'][jj]-budget['signed_direct'][jj]);max_closure_error=max(max_closure_error,endpoint_closure)
                    local_segments=[]
                    for a,b in SEGMENTS:
                        if b<=endpoint+1e-10:
                            piece=segment_budget(budget,a,b,endpoint)
                            row=dict(id=name,seed_anchor=float(ANCHORS[k]),seed_index=idx,horizon=endpoint,**piece)
                            segments.append(row);local_segments.append(piece)
                    summaries.append(dict(id=name,seed_anchor=float(ANCHORS[k]),seed_index=idx,horizon=endpoint,
                        projected_start_C=float(vals[0]),C_endpoint=float(vals[jj]),
                        historical_endpoint_error=float(error),direct_transport_defect=float(budget['signed_direct'][jj]),
                        source_budget_simpson=float(budget['signed_simpson'][jj]),
                        closure_error=float(endpoint_closure),quadrature_trapezoid_simpson_gap=float(abs(budget['pred_trap'][jj]-budget['pred_simp'][jj])),
                        budget_abs_over_signed=float(budget['abs_contribution'][jj]/max(abs(budget['signed_simpson'][jj]),1e-20)),
                        max_displacement_up_to_endpoint=float(distances[:jj+1].max()),
                        first_1scale_crossing=first_crossing(T[:jj+1],distances[:jj+1],1.),
                        first_3scale_crossing=first_crossing(T[:jj+1],distances[:jj+1],3.),
                        first_10scale_crossing=first_crossing(T[:jj+1],distances[:jj+1],10.),
                        source_largest_abs_xi=float(T[np.argmax(abs(budget['r'][:jj+1]))]),
                        segments_measured=len(local_segments)))
                for j,t in enumerate(T):
                    samples.append(dict(id=name,seed_anchor=float(ANCHORS[k]),seed_index=idx,delta_xi=float(t),
                        C=float(vals[j]),cofactor_mu=float(mus[j]),dC_dxi=float(dvals[j]),
                        source_r=float(budget['r'][j]),weighted_source=float(budget['source'][j]),
                        transported_initial=float(budget['baseline'][j]),direct_transport_defect=float(budget['signed_direct'][j]),
                        integrated_defect_simpson=float(budget['signed_simpson'][j]),
                        quadrature_closure_error=float(budget['signed_simpson'][j]-budget['signed_direct'][j]),
                        integrated_abs_source=float(budget['abs_contribution'][j]),
                        material_displacement_scaled=float(distances[j]),inside_1scale=bool(distances[j]<=1.),
                        inside_3scale=bool(distances[j]<=3.)))
                print('BUDGET',name,'xi',ANCHORS[k],'seed',idx,'final',float(budget['signed_direct'][-1]),
                      'integral',float(budget['signed_simpson'][-1]),
                      'first exit 1 scale',first_crossing(T,distances,1.),flush=True)
    overlaps=collect_angle_table(train,trainf,bases,center,scale,theta0,[candidates[n] for n in SELECTED])
    old_overlap={(r['id'],float(r['xi']),r['window']):float(r['normal_angle_deg']) for r in csv.DictReader(io.StringIO(archived_overlap))
                 if r['normal_angle_deg']}
    largest_replay_angle_gap=0.
    for row in overlaps:
        if row['diagnostic_metric']!='original_uniform':continue
        key=(row['id'],row['xi'],row['window'])
        if key not in old_overlap:raise RuntimeError('Historical overlap comparison missing')
        largest_replay_angle_gap=max(largest_replay_angle_gap,abs(old_overlap[key]-row['angle_deg']))
    if largest_replay_angle_gap>2e-3:raise RuntimeError('Original v18.33 overlapping normals not reproduced: '+str(largest_replay_angle_gap))
    output=R/'results';output.mkdir(exist_ok=True)
    write_csv(output/'original_v18_34_budget_samples.csv',samples)
    write_csv(output/'original_v18_34_budget_segments.csv',segments)
    write_csv(output/'original_v18_34_budget_endpoints.csv',summaries)
    write_csv(output/'original_v18_34_overlap_gauge.csv',overlaps)
    worst=max(abs(row['quadrature_closure_error']) for row in samples)
    report=dict(version='v18.34',status='ORIGINAL_Q550_TRANSPORT_DEFECT_BUDGET_OVERLAP_GAUGE_COMPLETED',
         original_ode_source_sha256_verified=True,parent_v18_33_sha256=prov['parent_sha'],
         parent_verified_members=prov['parent_verified_members'],original_ode_integrations=integration_count,
         selected_candidates=SELECTED,original_v18_33_candidates_reused_unchanged=True,
         original_v18_33_endpoints_reproduced=True,max_historical_endpoint_absolute_difference=max_validation_error,
         integrated_cofactor_residual_max_absolute_closure_error=max_closure_error,
         all_sample_max_absolute_quadrature_closure_error=worst,
         overlapping_window_max_angle_replay_error_deg=largest_replay_angle_gap,
         total_sample_count=len(samples),segment_count=len(segments),endpoint_count=len(summaries),
         no_new_candidate_fit_or_test_leakage=True,
         flow_integral_is_algebraic_consistency_not_physical_invariant=True,
         scales_are_diagnostic_only=True,no_certified_material_metric=True,
         no_material_poincare_return=True,no_material_constraint_certified=True,
         material_domain_exit_explicitly_tracked=True,
         full_ode_model_pinned=True)
    (output/'original_v18_34_proof.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_34_COMPLETED',integration_count,'ODE integrations',len(samples),'sample rows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    run(p.parse_args())
