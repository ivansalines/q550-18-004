#!/usr/bin/env python3
"""v18.48e: exact-formula Q550 RHS decomposition and Hessian identity audit.

Replays original phase-tracked 36 snapshots from SHA-pinned v18.47 ancestors,
without modifying the original system. Compares analytic H against the original
Jacobian finite difference as a diagnostic; no new physical invariant is claimed.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from structural_jet import structural_jet,analytic_delta,chart_kernel,coupling_jet

ROOT=Path(__file__).resolve().parents[1]

def norm(x):return float(np.linalg.norm(x))
def rel(a,b):return norm(a-b)/max(norm(a),norm(b),1e-30)
def cosine(a,b):
    den=norm(a)*norm(b)
    return float(np.dot(a,b)/den) if den>1e-30 else None
def med(vals):return float(np.median(vals)) if vals else None
def write_rows(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def jacobian_kernel(y,fy,jac,scale,g,h):
    """Original v18.48d five-point directional derivative of PINNED Jacobian."""
    w=np.r_[scale,1.]
    v=fy/w
    speed=norm(v)
    if speed<1e-30:raise RuntimeError('zero flow speed')
    u=v/speed
    def B(t):
        J=np.asarray(jac(y+w*t*h*u),float)
        return J[:10,:10]*scale[None,:]/scale[:,None]
    deriv=(B(-2)-8*B(-1)+8*B(1)-B(2))/(12*h)
    return speed*(g@deriv)

def require_reference_pins(base):
    refs={
      'audit':('test_support_mobius_monodromy_audit.py','90792fc9ad144333df521c7c0862193125c4702319fc3aa1a2bcef45491b2864'),
      'support':('test_support_q550_support_ode_interval_residual_pilot_v1.py','5b5938e7920b2ecd242fb8f416a18c7c47ae10aa06e027e0727f15e284c98c61'),
      'tube':('test_support_q550_exact_nonlinear_local_tube_v2_1.py','a9d1ec342c268da8e721dc641cab8cc4cf39f69d821dfd4191163833ba04e881')}
    for role,(name,digest) in refs.items():
        pinned=ROOT/'reference'/name
        origin=base/name
        for p in (pinned,origin):
            if not p.is_file():raise RuntimeError('Missing SHA-pinned source: '+str(p))
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=digest:raise RuntimeError('Original SHA drift: '+str(p))
    return {k:v[1] for k,v in refs.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v47-root',type=Path,required=True)
    ap.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    ap.add_argument('--h',type=float,nargs='+',default=[.048,.024,.012,.008,.004,.002])
    args=ap.parse_args()
    R=args.v47_root.resolve();base=args.base_root.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if not (R/'scripts/original_hessian_kernels_v18_47.py').is_file():raise RuntimeError('Missing v18.47 original source')
    source_hashes=require_reference_pins(base)
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
    import q550_state_impulse_response_v13 as pinned
    # Read all frozen historical seeds/cases and verify their historical provenance.
    original47,_,_,_,auth=read_parent_v47()
    _,old42,*_=read_parent_v43()
    _,cases36,_,attempts36,_=read_parent_v37()
    checkpoints,_,m33,*_=read_parent()
    d=select_original_checkpoint(checkpoints)
    f0,j0=model(base,d)   # SHA-authenticates every original Q550 dependency.
    verified={}
    for role,(name,digest) in pinned.DEPENDENCIES.items():
        file=base/name
        if not file.is_file():raise RuntimeError('Missing dependency '+str(file))
        actual=hashlib.sha256(file.read_bytes()).hexdigest()
        if actual!=digest:raise RuntimeError('SHA failure '+str(file))
        verified[role]={'path':str(file),'sha256':actual}
    support=pinned.load_module(base/pinned.DEPENDENCIES['support'][0],'v1848e_support')
    audit=pinned.load_module(base/pinned.DEPENDENCIES['audit'][0],'v1848e_audit')
    tube=pinned.load_module(base/pinned.DEPENDENCIES['tube'][0],'v1848e_tube')
    params=audit.ModelParameters(gamma_rho=.3)
    slope=float(d['slope']);kr=1/6;kg=35*math.pi*10/(12*550);dk=kg-kr
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
    rows=[];steps=[]; floor=[]; states=[]; gs=[]; frefs=[]; labels=[];integrations=1
    for name in NAMES:
        obj=candidates[name];q=np.asarray(obj['q'],float);opts=obj['opts'];mu=float(opts['lam'])
        if mu!=-1. or float(opts['mod'])!=0:raise RuntimeError('Archived original option drift')
        for i in ANCHOR_INDICES:
            anchor=anchors[i];xi=float(ANCHORS[i])
            radius=float(calibration_radius(train[i*56:(i+1)*56],anchor,scale)['radius_max'])
            for branch in (0,1):
                ck=(name,round(xi,8),branch);label=f'{name}|xi={xi:g}|branch={branch}'
                T=float(old42map[ck]['common_within_support_time'])
                y0,tries=projected_independent_seed(anchor,q,center,scale,theta0,opts,radius,rng,branch)
                verify_regenerated_attempts(ck,tries,casesmap,attemptsmap)
                sol=solve_ivp(f0,(0,T),y0,method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step,dense_output=True)
                if not sol.success:raise RuntimeError(sol.message)
                integrations+=1
                steps.append(dict(case_id=label,T=T,baseline_steps=len(sol.t)-1))
                for frac in (.125,.5,1.):
                    y=np.asarray(sol.sol(frac*T),float);fy=f0(frac*T,y);J=j0(y)
                    geom=source_geometry(y,fy,J,q,center,scale,theta0,opts['degree'],opts['phase'],mu)
                    gg=np.asarray(geom['g'],float)
                    ref,jref,href=structural_jet(y,kr,params,slope)
                    geo,jgeo,hgeo=structural_jet(y,kg,params,slope)
                    dF,dJ,dH=analytic_delta(y,kr,kg,params,slope)
                    fcheck=fg(frac*T,y); jcheck=jg(y)
                    rhs_error=max(rel(ref,fy),rel(geo,fcheck))
                    jac_error=max(rel(jref,J),rel(jgeo,jcheck))
                    if rhs_error>2e-11 or jac_error>2e-10:
                        raise RuntimeError(f'Structural RHS/J mismatch {label}: {rhs_error},{jac_error}')
                    differr=norm((hgeo-href)-dH)/max(norm(dH),1e-30)
                    if differr>2e-8:raise RuntimeError('Analytic Hessian identity failure: '+str(differr))
                    zexact=max(float(np.max(np.abs(dH[:,8,:]))),float(np.max(np.abs(dH[:,:,8]))))
                    if zexact!=0.0:raise RuntimeError('kappa structural z-zero identity violated')
                    if np.max(np.abs(dH[[0,2,4,6,8,9,10]]))!=0:raise RuntimeError('noncoupling row became kappa-sensitive')
                    ker0=chart_kernel(href,fy,scale,gg)
                    ker1=chart_kernel(hgeo,fy,scale,gg)
                    kerd=chart_kernel(dH,fy,scale,gg)
                    kb=chart_kernel(structural_jet(y,0.,params,slope)[2],fy,scale,gg)
                    kc=ker0-kb
                    k_identity=norm((ker1-ker0)-kerd)/max(norm(kerd),1e-30)
                    if k_identity>2e-8:raise RuntimeError('Chart-kernel identity failure: '+str(k_identity))
                    row=dict(case_id=label,anchor_xi=xi,branch=branch,fraction=frac,
                             slope=slope,kappa_ref=kr,kappa_geo=kg,delta_kappa=dk,
                             rhs_vs_pinned_relative=rhs_error,jac_vs_pinned_relative=jac_error,
                             full_hessian_delta_identity_relative=differr,
                             kappa_delta_hessian_norm=norm(dH),original_hessian_norm=norm(href),
                             relative_full_hessian_delta=norm(dH)/max(norm(href),1e-30),
                             predicted_structural_z_hessian_max=zexact,
                             ref_kernel_norm=norm(ker0),geo_kernel_norm=norm(ker1),
                             delta_kernel_norm=norm(kerd),
                             kernel_delta_relative=norm(kerd)/max(norm(ker0),1e-30),
                             kernel_delta_identity_relative=k_identity,
                             exact_kernel_cosine=cosine(ker0,ker1),
                             cancellation_ratio=(norm(kb)+norm(kc))/max(norm(ker0),1e-30),
                             predicted_column_z_delta=float(kerd[8]),
                             kernel_delta_column0=float(kerd[0]),kernel_delta_column2=float(kerd[2]),
                             kernel_delta_column8=float(kerd[8]),
                             ref_kernel_column0=float(ker0[0]),ref_kernel_column2=float(ker0[2]),
                             ref_kernel_column8=float(ker0[8]))
                    rows.append(row);states.append(y);gs.append(gg);frefs.append(fy);labels.append(label+f'|fraction={frac:g}')
                    for h in args.h:
                        nr=jacobian_kernel(y,fy,j0,scale,gg,h)
                        ng=jacobian_kernel(y,fy,jg,scale,gg,h)
                        floor.append(dict(case_id=label,fraction=frac,h=h,
                             original_numeric_vs_analytic_relative=rel(nr,ker0),
                             geometric_numeric_vs_analytic_relative=rel(ng,ker1),
                             kappa_delta_numeric_vs_analytic_relative=norm((ng-nr)-kerd)/max(norm(kerd),1e-30),
                             original_numeric_norm=norm(nr),geometric_numeric_norm=norm(ng),
                             analytic_original_norm=norm(ker0),analytic_delta_norm=norm(kerd),
                             numeric_kappa_delta_z_column=float((ng-nr)[8]),
                             analytic_kappa_delta_z_column=float(kerd[8])))
                print('v18.48e',label,'snapshot 3',flush=True)
    if len(rows)!=36 or len(floor)!=36*len(args.h):raise RuntimeError('Incomplete original dataset')
    write_rows(out/'structural_snapshots.csv',rows)
    write_rows(out/'jacobian_floor.csv',floor)
    write_rows(out/'case_integrations.csv',steps)
    np.savez_compressed(out/'structural_snapshot_states.npz',states=np.array(states),
        g=np.array(gs),f_ref=np.array(frefs),scale=scale,center=center,
        labels=np.array(labels),kappa_ref=kr,kappa_geo=kg,slope=slope)
    scan=[]
    for h in args.h:
        rr=[r for r in floor if r['h']==h]
        scan.append(dict(h=h,snapshots=len(rr),median_original_numeric_vs_analytic=med([r['original_numeric_vs_analytic_relative'] for r in rr]),
          median_geometric_numeric_vs_analytic=med([r['geometric_numeric_vs_analytic_relative'] for r in rr]),
          median_kappa_delta_numeric_vs_analytic=med([r['kappa_delta_numeric_vs_analytic_relative'] for r in rr]),
          median_kappa_delta_numeric_z=med([abs(r['numeric_kappa_delta_z_column']) for r in rr])))
    write_rows(out/'floor_summary.csv',scan)
    report=dict(version='v18.48e',status='ORIGINAL_Q550_ANALYTIC_STRUCTURAL_HESSIAN_FIXED_STATE',
       classification='EXACT_SYMBOLIC_FORMULA_EVALUATED_IN_BINARY64_NOT_INTERVAL_CERTIFIED',
       source_hashes=source_hashes,verified_dependencies=verified,parent_auth=auth,
       original_snapshot_count=len(rows),case_count=len(steps),original_integrations=integrations,
       kappa_ref=kr,kappa_geo=kg,delta_kappa=dk,slope=slope,
       max_rhs_pinned_relative=max(r['rhs_vs_pinned_relative'] for r in rows),
       max_jac_pinned_relative=max(r['jac_vs_pinned_relative'] for r in rows),
       max_delta_tensor_identity_relative=max(r['full_hessian_delta_identity_relative'] for r in rows),
       max_analytic_z_coupling_hessian=max(r['predicted_structural_z_hessian_max'] for r in rows),
       median_exact_full_hessian_delta_relative=med([r['relative_full_hessian_delta'] for r in rows]),
       median_exact_chart_kernel_delta_relative=med([r['kernel_delta_relative'] for r in rows]),
       median_kernel_cancellation_ratio=med([r['cancellation_ratio'] for r in rows]),
       max_chart_kernel_identity_relative=max(r['kernel_delta_identity_relative'] for r in rows),
       floor_scan=scan,
       limitations=('Analytic derivatives of pinned real-number formulas, evaluated in binary64;'
       ' not a formal interval enclosure. Frozen original source-gradient/chart, fixed states;'
       ' not an invariant metric, nor transported frame at geometric-kappa, nor a claim'
       ' about persistence of the full material hysteresis. Original source files unchanged.'))
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')
    print(json.dumps({k:report[k] for k in ('status','original_snapshot_count','case_count','original_integrations','max_rhs_pinned_relative','max_jac_pinned_relative','max_delta_tensor_identity_relative','median_exact_full_hessian_delta_relative','median_exact_chart_kernel_delta_relative')},indent=2))

if __name__=='__main__':main()
