#!/usr/bin/env python3
"""Original Q550: jointly admissible seeds, co-moving support and transported normals.

Transporting a surface with the flow is tautologically invariant. The actual
nontrivial test compares its transported cotangent against the same FROZEN
candidate C's gradient at independently evolved states, while recording
support departure and Lie-cofactor residuals. No physical constraint claim.
"""
from pathlib import Path
import argparse,json,csv,sys,hashlib
import numpy as np
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
from checkpoint_io_v18_36 import read_parent_v36
from checkpoint_io_v18_35 import read_parent_v35
from checkpoint_io_v18_34 import read_parent
from original_material_invariants_v18_31 import select_original_checkpoint,model,ANCHORS
from original_invariant_manifold_v18_32 import anchored_flow
from original_cofactor_surface_return_v18_33 import evaluate_samples
from manifold_math_v18_32 import features,features_flow,features_material_gradient,project_to_level,scaled_fit
from near_surface_math_v18_35 import calibration_radius,nearest_training
from flow_normal_math_v18_36 import unit,tangent_to,scaled_jacobian,normal_angle,first_exit,source_budget,transport_rhs
import q550_state_impulse_response_v13 as pinned
NAMES=['phasecov_d2_lam-1.00_mod+0.00','phaseblind_d2_lam-1.00_mod+0.00']
ANCHOR_INDICES=(1,3,5)
T=1.e-5

def write_csv(p,rows):
    if not rows:raise RuntimeError('No data to write '+str(p))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def projected_independent_seed(ref,q,center,scale,theta0,opts,radius,rng,branch):
    """A predetermined rejection sampler; never select by future behavior."""
    attempts=[]
    for k in range(1,257):
        direction=unit(rng.normal(size=10))
        y=np.asarray(ref,float).copy()
        y[:10]+=(0.09 if branch==0 else 0.20)*scale*direction
        y[10]+=(.0 if branch==0 else .08)
        z,shift,ok=project_to_level(y,q,center,scale,theta0,opts['degree'],opts['phase'],max_scaled_step=.7)
        d=float(np.linalg.norm((z[:10]-ref[:10])/scale))
        admissible=bool(ok and d<=.80*radius and abs(z[10]-ref[10])<=.20)
        attempts.append(dict(try_index=k,valid_projection=bool(ok),scaled_step=float(shift),scaled_radius=float(d),in_support=admissible))
        if admissible:return z,attempts
    raise RuntimeError('Could not find certified diagnostic in-support projected seed in 256 fixed attempts')

def run(args):
    proof35,cases35,auth=read_parent_v36()
    p34,stored33,archived,auth34=read_parent_v35()
    checkpoint,proof33,model33,returns,overlap,history=read_parent()
    d=select_original_checkpoint(checkpoint)
    f,jac=model(args.base_root,d)
    bases=anchored_flow(f,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=np.asarray(model33['center'],float);scale=np.asarray(model33['scale'],float);theta0=float(model33['theta0'])
    expect=np.maximum.reduce([.025*np.abs(bases[3,:10]),.1*np.ptp(bases[:,:10],axis=0),np.full(10,1e-6)])
    if not np.allclose(scale,expect,rtol=5e-9,atol=1e-8) or not np.allclose(center,bases[3,:10],rtol=5e-9,atol=1e-8):raise RuntimeError('Authentic calibration reference mismatch')
    train,trainf,test,testf,_,_=evaluate_samples(f,bases,scale)
    candidates={x['id']:x for x in model33['candidates']}
    if any(n not in candidates for n in NAMES):raise RuntimeError('Archived candidate missing')
    times=np.unique(np.r_[0.,np.geomspace(1.e-11,T,65),np.linspace(0.,T,81)])
    trajectory_rows=[];cases=[];attempt_rows=[];integrations=1;max_dual_error=0.
    gauges={'native':np.ones(10),'coordinate_stretch':np.geomspace(.6,1.7,10),'coordinate_inverse':np.geomspace(1.7,.6,10)}
    rng=np.random.default_rng(1836)
    for name in NAMES:
        it=candidates[name];q=np.asarray(it['q'],float);opts=it['opts']
        if float(opts['lam'])!=-1. or float(opts['mod'])!=0.:raise RuntimeError('Unexpected candidate cofactor')
        mu=-1.
        def C(y):return float(features(y,center,scale,theta0,opts['degree'],opts['phase'])@q)
        def gradient_scaled(y):return features_material_gradient(y,center,scale,theta0,opts['degree'],opts['phase']).T@q*scale
        def residual(y,fy):return float(features_flow(y,fy,center,scale,theta0,opts['degree'],opts['phase'])@q-mu*C(y))
        for i in ANCHOR_INDICES:
            a=float(ANCHORS[i]);anchor=bases[i]
            local_train=train[i*56:(i+1)*56]
            calib=calibration_radius(local_train,anchor,scale)
            radius=float(calib['radius_max'])
            for branch in (0,1):
                z,attempts=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                for entry in attempts:attempt_rows.append(dict(id=name,anchor_xi=a,branch=branch,**entry))
                base=anchor.copy();base[10]=z[10]
                n0=unit(gradient_scaled(z))
                probe=np.roll(np.arange(1.,11.),branch+2)
                u0=tangent_to(n0,probe)
                x0=np.r_[z,base,n0,u0,0.]
                def rhs(t,x):
                    ys=x[:11];yr=x[11:22];m=x[22:32];u=x[32:42]
                    fy=f(t,ys);fr=f(t,yr)
                    B=scaled_jacobian(jac(ys),scale)
                    dm,du=transport_rhs(B,m,u)
                    return np.r_[fy,fr,dm,du,np.exp(-mu*t)*residual(ys,fy)]
                sol=solve_ivp(rhs,(0.,T),x0,method='DOP853',dense_output=True,rtol=args.rtol,atol=args.atol,max_step=args.max_step)
                if not sol.success:raise RuntimeError('Original variational-normal solve failed: '+sol.message)
                integrations+=1
                xx=np.asarray(sol.sol(times)).T
                ss=xx[:,:11];refs=xx[:,11:22];normals=xx[:,22:32];tangents=xx[:,32:42];budget=xx[:,42]
                moving=np.linalg.norm((ss[:,:10]-refs[:,:10])/scale,axis=1)
                c0=C(z);cvals=np.asarray([C(y) for y in ss]);rvals=np.asarray([residual(y,f(0.,y)) for y in ss]);diagnostic_normals=np.asarray([unit(gradient_scaled(y)) for y in ss])
                defect=cvals-np.exp(mu*times)*c0;integrated=source_budget(c0,mu,times,budget)
                closure=float(np.max(np.abs(defect-integrated)))
                dual0=float(n0@u0);dual=np.array([m@u for m,u in zip(normals,tangents)])
                dual_error=float(np.max(abs(dual-dual0)));max_dual_error=max(max_dual_error,dual_error)
                phase_band_initial=abs(z[10]-anchor[10])<=.25
                if not phase_band_initial or moving[0]>.8*radius or abs(c0)>1e-8:raise RuntimeError('Inadmissible initial seed')
                def exact_d(t):
                    xt=sol.sol(float(t));return float(np.linalg.norm((xt[:10]-xt[11:21])/scale))
                status,exit_t=first_exit(times,moving,radius,exact_d)
                idx=np.searchsorted(times,exit_t,side='right') if exit_t is not None else len(times)
                within=np.arange(len(times))<idx if exit_t is not None else np.ones(len(times),bool)
                # Once local support is left, angle comparisons are still diagnostics but cannot validate locality.
                for j,(t,y,yref,m,u,nc,dmove,cc,rr,dd,ii) in enumerate(zip(times,ss,refs,normals,tangents,diagnostic_normals,moving,cvals,rvals,defect,integrated)):
                    row=dict(id=name,anchor_xi=a,branch=branch,delta_xi=float(t),candidate_C=float(cc),cofactor_source=float(rr),defect=float(dd),source_integral=float(ii),
                        distance_to_comoving=float(dmove),radius_max=radius,phase_shift_from_training=float(z[10]-anchor[10]),inside_joint_support=bool(dmove<=radius and phase_band_initial),
                        transported_dual_pairing=float(m@u),candidate_normal_tangent_pairing=float(nc@unit(u)),
                        transported_candidate_normal_angle_deg=normal_angle(m,nc))
                    for gauge,weights in gauges.items():row['normal_angle_'+gauge+'_deg']=normal_angle(m,nc,weights)
                    trajectory_rows.append(row)
                cases.append(dict(id=name,anchor_xi=a,branch=branch,initial_C=float(c0),initial_source=float(rvals[0]),initial_distance=float(moving[0]),training_radius_max=radius,
                    initial_phase_shift=float(z[10]-anchor[10]),joint_initial_admissibility=True,first_exit_status=status,first_exit_delta_xi=exit_t,
                    fraction_samples_in_support=float(np.mean(moving<=radius)),max_integrated_defect_within_support=float(np.max(abs(integrated[within]))) if np.any(within) else 0.,
                    max_normal_angle_in_support_deg=float(max(normal_angle(m,n) for m,n,flag in zip(normals,diagnostic_normals,within) if flag)) if np.any(within) else None,
                    final_normal_angle_deg=normal_angle(normals[-1],diagnostic_normals[-1]),
                    max_source_budget_closure_error=closure,max_dual_pairing_error=dual_error,
                    final_defect=float(defect[-1]),final_source_integral=float(integrated[-1]),number_of_seed_attempts=len(attempts)))
                print('FLOW_NORMAL',name,'xi',a,'branch',branch,'exit',status,exit_t,'initial_source',f'{rvals[0]:.5g}','angle_final',f'{cases[-1]["final_normal_angle_deg"]:.5g}',flush=True)
    out=R/'results';out.mkdir(exist_ok=True)
    write_csv(out/'original_v18_36_normal_profiles.csv',trajectory_rows)
    write_csv(out/'original_v18_36_joint_support_cases.csv',cases)
    write_csv(out/'original_v18_36_seed_attempts.csv',attempt_rows)
    report=dict(version='v18.36',status='ORIGINAL_Q550_FLOW_ALIGNED_SUPPORT_NORMAL_TRANSPORT_COMPLETED',original_ode_sha256_verified=True,
        parent_v18_35_sha256=auth['parent_sha256'],parent_v18_35_verified_members=auth['verified_members'],
        original_integrations=integrations,case_count=len(cases),profile_rows=len(trajectory_rows),
        all_seeds_jointly_admissible=bool(all(c['joint_initial_admissibility'] for c in cases)),
        includes_original_jacobian_cotangent_transport=True,checks_transport_against_frozen_candidate_gradient=True,
        no_physical_material_manifold_certified=True,no_privileged_metric_certified=True,no_return_certified=True,
        max_budget_closure_error=max(c['max_source_budget_closure_error'] for c in cases),
        max_dual_pairing_error=max_dual_error,method=dict(T=T,rtol=args.rtol,atol=args.atol,max_step=args.max_step,time_count=len(times),normal_transport='m_prime=-B^T m',tangent_transport='u_prime=B u',B='S^-1 J_material S',normal_gauge='weighted Euclidean diagnostic',calibration_only_training_samples=True,seed_generator='fixed RNG 1836 independent rejection prior to future simulation'))
    (out/'original_v18_36_proof.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ORIGINAL_V18_36_COMPLETED',integrations,len(cases),len(trajectory_rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,default=pinned.DEFAULT_BASE)
    p.add_argument('--rtol',type=float,default=3e-12);p.add_argument('--atol',type=float,default=3e-13)
    p.add_argument('--max-step',type=float,default=2.5e-7);run(p.parse_args())
