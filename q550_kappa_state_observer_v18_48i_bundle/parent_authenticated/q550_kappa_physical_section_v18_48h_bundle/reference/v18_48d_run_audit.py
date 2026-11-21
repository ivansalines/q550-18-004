#!/usr/bin/env python3
"""v18.48d: pinned-Jacobian-derived directional Hessian and numerical floor separation.
The Jacobian is pre-existing; its directional differentiation is numerical, NOT an exact symbolic Hessian.
"""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-30))
def cos(a,b):
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na*nb>1e-28 else None
def vec(x):return [float(t) for t in x]
def mean_safe(x):return float(np.median(x)) if x else None
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def jacobian_hessian_probe(y, fy, jac, scale, g, h):
    """Compute speed*g^T D_u B[:,j] using a 5-point first derivative of the pinned Jacobian.

    B is the normalized state Jacobian; u is unit normalized full speed.
    This avoids O(h^-2) RHS second-difference cancellation, but not J roundoff.
    """
    from dual_hessian_jet_math_v18_44 import normalized_full_speed
    y=np.asarray(y,float); scale=np.asarray(scale,float);g=np.asarray(g,float)
    w=np.r_[scale,1.]
    v=normalized_full_speed(fy,scale);speed=float(np.linalg.norm(v))
    if speed<=1e-30:raise RuntimeError('Zero normalized speed')
    u=v/speed
    def B(a):
        J=np.asarray(jac(y+w*a*h*u),float)
        if J.shape!=(11,11) or not np.all(np.isfinite(J)):raise RuntimeError('invalid pinned Jacobian')
        return J[:10,:10]*scale[None,:]/scale[:,None]
    derivative=(B(-2)-8*B(-1)+8*B(1)-B(2))/(12*h)
    return speed*(g@derivative)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v47-root',type=Path,required=True)
    ap.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
    ap.add_argument('--output-dir',type=Path,default=Path('results'))
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    ap.add_argument('--h',type=float,nargs='+',default=[.048,.032,.024,.016,.012,.008,.006,.004,.003,.002])
    ap.add_argument('--restart',action='store_true',help='also integrate geometric-kappa trajectories from same initial seeds')
    args=ap.parse_args(); R=args.v47_root.resolve();base=args.base_root.resolve();out=args.output_dir.resolve()
    if not (R/'scripts/original_hessian_kernels_v18_47.py').exists():raise SystemExit('Unextracted v18.47 root: '+str(R))
    sys.path[:0]=[str(R/'scripts'),str(R/'reference')]
    from checkpoint_io_v18_47 import read_parent_v47
    from checkpoint_io_v18_43 import read_parent_v43
    from checkpoint_io_v18_37 import read_parent_v37
    from checkpoint_io_v18_34 import read_parent
    from original_material_invariants_v18_31 import select_original_checkpoint,model,ANCHORS
    from original_invariant_manifold_v18_32 import anchored_flow
    from original_cofactor_surface_return_v18_33 import evaluate_samples
    from original_flow_normal_v18_36 import NAMES,ANCHOR_INDICES,projected_independent_seed
    from seed_genealogy_v18_38 import index_archived_attempts,verify_regenerated_attempts
    from near_surface_math_v18_35 import calibration_radius
    from residual_gradient_math_v18_37 import source_geometry
    from kernel_compensation_math_v18_47 import mixed_compensated,polarization_rhs,accel_gradient_identity,shares
    import q550_state_impulse_response_v13 as pinned
    old,_,_,_,auth=read_parent_v47()
    _, old42,*_=read_parent_v43()
    _,cases36,_,attempts36,_=read_parent_v37()
    checkpoints,_,m33,*_=read_parent()
    d=select_original_checkpoint(checkpoints)
    f0,j0=model(base,d) # validates all original SHA-pinned dependencies
    verified={}
    for role,(name,h) in pinned.DEPENDENCIES.items():
        p=base/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=h:raise SystemExit('Dependency checksum failure '+str(p))
        verified[role]={'path':str(p),'sha256':actual}
    # no monkeypatch of the pinned original RHS or reference scripts
    support=pinned.load_module(base/pinned.DEPENDENCIES['support'][0],'v1848b_support')
    audit=pinned.load_module(base/pinned.DEPENDENCIES['audit'][0],'v1848b_audit')
    tube=pinned.load_module(base/pinned.DEPENDENCIES['tube'][0],'v1848b_tube')
    k0=1/6;kg=35*math.pi*10/(12*550);slope=float(d['slope']);params=audit.ModelParameters(gamma_rho=.3)
    def fg(t,y):return slope*np.asarray(support.support_rhs_float(np.asarray(y,float),kappa0=kg,params=params,audit=audit),float)
    def jg(y):return slope*np.asarray(tube.float_support_jacobian(np.asarray(y,float),kappa0=kg,params=params,audit=audit,base=support),float)
    anchors=anchored_flow(f0,np.asarray(d['source_state_11D'],float),ANCHORS)
    center=np.asarray(m33['center'],float);scale=np.asarray(m33['scale'],float);theta0=float(m33['theta0'])
    train,*_=evaluate_samples(f0,anchors,scale)
    candidates={v['id']:v for v in m33['candidates']}
    casesmap={(r['id'],round(float(r['anchor_xi']),8),int(r['branch'])):r for r in cases36}
    attemptsmap=index_archived_attempts(attempts36)
    old42map={(r['id'],round(float(r['anchor_xi']),8),int(r['branch'])):r for r in old42}
    rng=np.random.default_rng(1836)
    rows=[]; cases=[]; integrations=0
    for name in NAMES:
        obj=candidates[name];q=np.asarray(obj['q'],float);opts=obj['opts'];mu=float(opts['lam'])
        if mu!=-1. or float(opts['mod'])!=0:raise RuntimeError('Archived original option drift')
        for i in ANCHOR_INDICES:
            anchor=anchors[i];xi=float(ANCHORS[i]);radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ck=(name,round(xi,8),branch);label=f'{name}|xi={xi:g}|branch={branch}'
                oldrow=old42map[ck]; T=float(oldrow['common_within_support_time'])
                y0,tries=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                verify_regenerated_attempts(ck,tries,casesmap,attemptsmap)
                kw=dict(method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step,dense_output=True)
                sol=solve_ivp(f0,(0,T),y0,**kw)
                if not sol.success:raise RuntimeError(sol.message)
                integrations+=1
                solg=solve_ivp(fg,(0,T),y0,**kw) if args.restart else None
                if solg is not None:
                    if not solg.success:raise RuntimeError(solg.message)
                    integrations+=1
                cases.append(dict(case_id=label,T=T,baseline_steps=len(sol.t)-1,geometric_steps=(len(solg.t)-1 if solg else ''),branch=branch))
                for frac in (.125,.5,1.):
                    t=frac*T;y=np.asarray(sol.sol(t),float);fy=f0(t,y);J=j0(y)
                    geo=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                    g=np.asarray(geo['g'],float)
                    yy=np.asarray(solg.sol(t),float) if solg is not None else None
                    # same reference directions across both kappas: isolates parameter perturbation
                    for h in args.h:
                        # Evaluate both Hessian constructions independently at EACH state.
                        # All directional probes are held at the original chart gauge.
                        A0=np.asarray(mixed_compensated(y,fy,f0,scale,g,h),float)
                        B0=np.asarray(polarization_rhs(y,fy,f0,scale,g,h),float)
                        A1=np.asarray(mixed_compensated(y,fg(t,y),fg,scale,g,h),float)
                        B1=np.asarray(polarization_rhs(y,fg(t,y),fg,scale,g,h),float)
                        J0=np.asarray(jacobian_hessian_probe(y,fy,j0,scale,g,h),float)
                        J1=np.asarray(jacobian_hessian_probe(y,fg(t,y),jg,scale,g,h),float)
                        for mode,yref,ygeo in [('fixed_state_fixed_direction',y,y)]+([('restart_state_common_direction',y,yy)] if yy is not None else []):
                            if mode=='fixed_state_fixed_direction':
                                C1,D1=A1,B1; K1=J1
                            else:
                                C1=np.asarray(mixed_compensated(yy,fg(t,yy),fg,scale,g,h),float)
                                D1=np.asarray(polarization_rhs(yy,fg(t,yy),fg,scale,g,h),float)
                                K1=np.asarray(jacobian_hessian_probe(yy,fg(t,yy),jg,scale,g,h),float)
                            defect0=A0-B0; defect1=C1-D1
                            jac_defect0=A0-J0; jac_defect1=C1-K1
                            scaleH=max(float(np.linalg.norm(A0)),float(np.linalg.norm(C1)),float(np.linalg.norm(B0)),float(np.linalg.norm(D1)),1e-30)
                            rows.append(dict(case_id=label,anchor_xi=xi,branch=branch,fraction=frac,h=h,mode=mode,
                                kappa_ref=k0,kappa_geo=kg,
                                rhs_relative=rel(fy,fg(t,ygeo)),state_relative=rel(yref,ygeo),
                                mixed_relative=rel(A0,C1),mixed_cosine=cos(A0,C1),polar_relative=rel(B0,D1),
                                ref_defect_relative=rel(A0,B0),geo_defect_relative=rel(C1,D1),
                                ref_defect_norm=float(np.linalg.norm(defect0)),geo_defect_norm=float(np.linalg.norm(defect1)),
                                defect_difference_norm=float(np.linalg.norm(defect1-defect0)),
                                defect_change_normalized=float(np.linalg.norm(defect1-defect0)/scaleH),
                                defect_change_self_normalized=rel(defect0,defect1),
                                hessian_scale=scaleH,
                                jacobian_ref_relative=rel(A0,J0),jacobian_geo_relative=rel(C1,K1),
                                jacobian_kappa_relative=rel(J0,K1),jacobian_kappa_cosine=cos(J0,K1),
                                jacobian_ref_norm=float(np.linalg.norm(J0)),jacobian_geo_norm=float(np.linalg.norm(K1)),
                                rhs_jacobian_defect_change=float(np.linalg.norm(jac_defect1-jac_defect0)/scaleH),
                                focus_0_change=float(C1[0]-A0[0]),focus_2_change=float(C1[2]-A0[2]),focus_8_change=float(C1[8]-A0[8]),
                                ref_defect_0=float(defect0[0]),geo_defect_0=float(defect1[0]),
                                ref_defect_2=float(defect0[2]),geo_defect_2=float(defect1[2]),
                                ref_defect_8=float(defect0[8]),geo_defect_8=float(defect1[8]),
                                ref_focus_0_8_share=float(np.sum(shares(defect0)[[0,8]])),
                                geo_focus_0_8_share=float(np.sum(shares(defect1)[[0,8]]))))
                print('v18.48d',label,'snapshots 3',flush=True)
    if len(cases)!=12 or len(rows)!=(36*len(args.h)*(2 if args.restart else 1)):raise RuntimeError('incomplete dataset')
    write_csv(out/'stencil_defects.csv',rows);write_csv(out/'case_integrations.csv',cases)
    summary=dict(version='v18.48d',status='ORIGINAL_Q550_PINNED_JACOBIAN_DERIVATIVE_FLOOR_AUDIT',baseline_kappa=k0,autonomous_kappa=kg,
        delta_kappa=kg-k0, original_case_count=len(cases),snapshot_count=36,rows=len(rows),integrations=integrations,
        restart_enabled=args.restart,original_verified_dependencies=verified,archived_parent_authority=auth,
        scope='36 original snapshots; compare RHS mixed/polarized stencils against independent pinned-Jacobian first-difference pathway. Same original chart directions retained at fixed and restart states.',
        limitations='Binary64 cancellation at small h; no analytic Hessian or interval enclosure. No rescaled tangent frame. Gauge and source_geometry remain original. Convergence checked empirically only.',
        fixed_median_hessian_relative=mean_safe([r['mixed_relative'] for r in rows if r['mode']=='fixed_state_fixed_direction']),
        restart_median_hessian_relative=mean_safe([r['mixed_relative'] for r in rows if r['mode']=='restart_state_common_direction']))
    # Every statistic below is computed from raw per-snapshot rows, not copied across modes.
    convergence=[]
    for mode in sorted(set(r['mode'] for r in rows)):
        for h in sorted(set(args.h),reverse=True):
            subset=[r for r in rows if r['mode']==mode and r['h']==h]
            convergence.append(dict(mode=mode,h=h,rows=len(subset),
                median_mixed_relative=mean_safe([r['mixed_relative'] for r in subset]),
                median_jacobian_kappa_relative=mean_safe([r['jacobian_kappa_relative'] for r in subset]),
                median_rhs_jacobian_ref_defect=mean_safe([r['jacobian_ref_relative'] for r in subset]),
                median_rhs_jacobian_geo_defect=mean_safe([r['jacobian_geo_relative'] for r in subset]),
                median_ref_defect_relative=mean_safe([r['ref_defect_relative'] for r in subset]),
                median_geo_defect_relative=mean_safe([r['geo_defect_relative'] for r in subset]),
                median_normalized_defect_change=mean_safe([r['defect_change_normalized'] for r in subset]),
                median_self_normalized_defect_change=mean_safe([r['defect_change_self_normalized'] for r in subset])))
    write_csv(out/'stencil_convergence.csv',convergence)
    summary['convergence']=convergence
    summary['structural_hessian_status']='NOT_DERIVED: original pinned support RHS source is not bundled; Jacobian pathway remains floating-point differentiated. No exact algebraic component-level decomposition.'
    summary['interpretation_gate']='PINNED_JACOBIAN_DIFFERENTIATION_IS_NUMERICAL: evaluate h-plateau of Jacobian-derived sensitivity against RHS stencil. NO EXACT STRUCTURAL HESSIAN claimed. Relative small-residual changes need absolute context.'
    summary['original_parent_script_sha256']=hashlib.sha256((R/'scripts/original_hessian_kernels_v18_47.py').read_bytes()).hexdigest()
    summary['v18_48b_replayed_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','snapshot_count','rows','integrations','fixed_median_hessian_relative','restart_median_hessian_relative')},indent=2))
if __name__=='__main__':main()
