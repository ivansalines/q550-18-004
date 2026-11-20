#!/usr/bin/env python3
"""v18.48b: authentic fixed-state and restarted-trajectory numerical counterfactual.
No analytic Hessian or certified invariant is asserted.
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v47-root',type=Path,required=True)
    ap.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
    ap.add_argument('--output-dir',type=Path,default=Path('results'))
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    ap.add_argument('--h',type=float,nargs='+',default=[.004,.008,.012])
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
                        c0=np.asarray(mixed_compensated(y,fy,f0,scale,g,h),float)
                        c1=np.asarray(mixed_compensated(y,fg(t,y),fg,scale,g,h),float)
                        p0=np.asarray(polarization_rhs(y,fy,f0,scale,g,h),float)
                        p1=np.asarray(polarization_rhs(y,fg(t,y),fg,scale,g,h),float)
                        record=dict(case_id=label,anchor_xi=xi,branch=branch,fraction=frac,h=h,mode='fixed_state_fixed_direction',kappa_ref=k0,kappa_geo=kg,
                          rhs_relative=rel(fy,fg(t,y)),state_relative=0.,hessian_relative=rel(c0,c1),hessian_cosine=cos(c0,c1),
                          compensated_to_polarized_ref=rel(c0,p0),compensated_to_polarized_geo=rel(c1,p1),
                          change_in_compensation_defect=rel(c0-p0,c1-p1),focus_0_change=float(c1[0]-c0[0]),focus_2_change=float(c1[2]-c0[2]),focus_8_change=float(c1[8]-c0[8]),
                          focus_0_8_share_ref=float(np.sum(shares(c0-p0)[[0,8]])),focus_0_8_share_geo=float(np.sum(shares(c1-p1)[[0,8]])))
                        rows.append(record)
                        if yy is not None:
                            # common transported direction, evaluated on re-integrated geometric state
                            gr=mixed_compensated(yy,fg(t,yy),fg,scale,g,h)
                            rows.append(dict(**{**record,'mode':'restart_state_common_direction','state_relative':rel(y,yy),'rhs_relative':rel(fy,fg(t,yy)),'hessian_relative':rel(c0,gr),'hessian_cosine':cos(c0,gr),'focus_0_change':float(gr[0]-c0[0]),'focus_2_change':float(gr[2]-c0[2]),'focus_8_change':float(gr[8]-c0[8])}))
                print('v18.48b',label,'snapshots 3',flush=True)
    if len(cases)!=12 or len(rows)!=(36*len(args.h)*(2 if args.restart else 1)):raise RuntimeError('incomplete dataset')
    write_csv(out/'transported_kappa_sensitivity.csv',rows);write_csv(out/'case_integrations.csv',cases)
    summary=dict(version='v18.48b',status='ORIGINAL_Q550_NUMERICAL_TRANSPORTED_KAPPA_AUDIT',baseline_kappa=k0,autonomous_kappa=kg,
        delta_kappa=kg-k0, original_case_count=len(cases),snapshot_count=36,rows=len(rows),integrations=integrations,
        restart_enabled=args.restart,original_verified_dependencies=verified,archived_parent_authority=auth,
        scope='Shared original 36 snapshot times and original geometric directions. Optional geometric RHS restarts from identical seeds; restart mode retains original directions, not a newly propagated variational frame.',
        limitations='Binary64 finite differences; numerical Hessian, not exact; original direction construction held fixed. No full parameter-dependent tangent transport, analytic RHS decomposition, or interval certificate.',
        fixed_median_hessian_relative=mean_safe([r['hessian_relative'] for r in rows if r['mode']=='fixed_state_fixed_direction']),
        restart_median_hessian_relative=mean_safe([r['hessian_relative'] for r in rows if r['mode']=='restart_state_common_direction']))
    out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','snapshot_count','rows','integrations','fixed_median_hessian_relative','restart_median_hessian_relative')},indent=2))
if __name__=='__main__':main()
