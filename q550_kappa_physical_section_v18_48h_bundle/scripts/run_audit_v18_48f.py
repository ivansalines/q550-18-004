#!/usr/bin/env python3
"""Q550 v18.48f: analytic kernel cancellation + real ODE and 11D cotangent transport.

Data provenance: frozen v18.48e original-run 36 snapshots (SHA-pinned).
The transport starts at the 0.125*T snapshot of each of twelve original cases.
Both kappa trajectories start at the same 11D state, their FULL 11D chart
covectors start at the same [frozen original gradient, zero last component].
Only the original 10D chart projection is used in the kernel observable.
This is a short supported local window, not a full-kick or orbit-level test.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from structural_jet import structural_jet, coupling_jet, chart_kernel

ROOT=Path(__file__).resolve().parents[1]
PINNED={
 'audit':('test_support_mobius_monodromy_audit.py','90792fc9ad144333df521c7c0862193125c4702319fc3aa1a2bcef45491b2864'),
 'support':('test_support_q550_support_ode_interval_residual_pilot_v1.py','5b5938e7920b2ecd242fb8f416a18c7c47ae10aa06e027e0727f15e284c98c61'),
 'tube':('test_support_q550_exact_nonlinear_local_tube_v2_1.py','a9d1ec342c268da8e721dc641cab8cc4cf39f69d821dfd4191163833ba04e881')}

def digest(p):
    hh=hashlib.sha256()
    with open(p,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):hh.update(block)
    return hh.hexdigest()

def norm(x):return float(np.linalg.norm(x))
def med(a):return float(np.median(a)) if a else None
def rel(a,b):return norm(a-b)/max(norm(a),norm(b),1e-30)
def cosine(a,b):
    na,nb=norm(a),norm(b)
    return float(np.dot(a,b)/(na*nb)) if na*nb>1e-30 else None

def write_csv(path,rows):
    if not rows:raise ValueError(f'empty output: {path}')
    with path.open('w',newline='',encoding='utf8') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def verify_parent_inputs():
    pins=json.loads((ROOT/'inputs'/'PARENT_SHA256.json').read_text())
    result={}
    for name,expected in pins['files'].items():
        p=ROOT/'inputs'/name
        if not p.is_file() or digest(p)!=expected:raise RuntimeError(f'v18.48e input SHA drift: {p}')
        result[name]=digest(p)
    report=json.loads((ROOT/'inputs'/'report.json').read_text())
    if report['version']!='v18.48e' or report['original_snapshot_count']!=36 or report['case_count']!=12:
        raise RuntimeError('Parent v18.48e report mismatch')
    if len(report['verified_dependencies'])!=6 or report['max_rhs_pinned_relative']>2e-11:
        raise RuntimeError('Parent provenance not authenticated')
    return report,result

def load_sources(base):
    # Strictly use the historical server sources (no silent fallback to a new physical model).
    for _,(name,expected) in PINNED.items():
        for path in (ROOT/'reference'/name,base/name):
            if not path.is_file() or digest(path)!=expected:
                raise RuntimeError(f'Original source absent or SHA mismatch: {path}')
    import importlib.util
    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    audit=load(base/PINNED['audit'][0],'v1848f_audit')
    support=load(base/PINNED['support'][0],'v1848f_support')
    tube=load(base/PINNED['tube'][0],'v1848f_tube')
    return audit,support,tube

def case_meta(label):
    case,f=label.rsplit('|fraction=',1)
    family,xi_part,branch_part=case.split('|')
    return case,family,float(xi_part.split('=')[1]),int(branch_part.split('=')[1]),float(f)

def cancellation_stats(kb,kc):
    n0,nc=norm(kb),norm(kc)
    kt=kb+kc;nt=norm(kt);den=n0+nc
    return {'base_norm':n0,'coupling_norm':nc,'total_norm':nt,
            'cancellation_ratio':den/max(nt,1e-30),
            'cancellation_fraction':1-nt/max(den,1e-30),
            'opposition_cosine':cosine(kb,kc),
            'near_zero_denominator':bool(nt<1e-7*max(den,1e-30))}

def analytic_parts(y,kappa,slope,params,f,scale,g):
    # H = H_base + kappa H_coupling in the physical 11D internal coordinates.
    hb=structural_jet(y,0.,params,slope)[2]
    hc=slope*kappa*coupling_jet(y,params)[2]
    b=chart_kernel(hb,f,scale,g[:10]);c=chart_kernel(hc,f,scale,g[:10])
    return b,c,hb+hc

def compute_row(yref,ygeo,gref,ggeo,kr,kg,slope,params,scale,fy_ref,fy_geo):
    b_ref,c_ref,hr=analytic_parts(yref,kr,slope,params,fy_ref,scale,gref)
    b_geo,c_geo,hg=analytic_parts(ygeo,kg,slope,params,fy_geo,scale,ggeo)
    r_ref=cancellation_stats(b_ref,c_ref);r_geo=cancellation_stats(b_geo,c_geo)
    kref=b_ref+c_ref;kgeo=b_geo+c_geo
    # Exact telescoping at fixed chart, isolates four mathematically distinct changes:
    # 1. direct kappa curvature 2. state 3. local flow 4. transported cotangent.
    hgeo_on_ref=structural_jet(yref,kg,params,slope)[2]
    A=chart_kernel(hr,fy_ref,scale,gref[:10])
    B=chart_kernel(hgeo_on_ref,fy_ref,scale,gref[:10])
    C=chart_kernel(hg,fy_ref,scale,gref[:10])
    D=chart_kernel(hg,fy_geo,scale,gref[:10])
    E=chart_kernel(hg,fy_geo,scale,ggeo[:10])
    terms=[B-A,C-B,D-C,E-D]
    total=E-A
    budget_err=norm(sum(terms)-total)/max(norm(total),norm(E),1e-30)
    if budget_err>2e-9:raise RuntimeError('Telescoping closure not satisfied')
    _,_,hk=coupling_jet(yref,params)
    closed_direct=chart_kernel(slope*(kg-kr)*hk,fy_ref,scale,gref[:10])
    direct_err=norm(terms[0]-closed_direct)/max(norm(closed_direct),1e-30)
    if direct_err>3e-8:raise RuntimeError('Exact kappa coupling identity violated')
    if np.max(np.abs(hk[:,8,:])) !=0 or np.max(np.abs(hk[:,:,8]))!=0:
        raise RuntimeError('Unexpected z-kappa coupling')
    r={('reference_'+k):v for k,v in r_ref.items()}
    r.update({('geometric_'+k):v for k,v in r_geo.items()})
    r.update(reference_kernel_norm=norm(kref),geometric_kernel_norm=norm(kgeo),
         difference_kernel_norm=norm(total),
         difference_kernel_relative=norm(total)/max(norm(kref),1e-30),
         total_kernel_cosine=cosine(kref,kgeo),
         state_abs_difference=norm(ygeo-yref),state_rel_difference=rel(yref,ygeo),
         covector_difference=norm(ggeo-gref),covector_relative_difference=rel(gref,ggeo),
         direct_hessian_norm=norm(terms[0]),state_hessian_norm=norm(terms[1]),
         flow_change_norm=norm(terms[2]),dual_transport_norm=norm(terms[3]),
         telescope_budget_relative_error=budget_err,direct_analytic_identity_relative_error=direct_err,
         direct_z_kernel_column=float(terms[0][8]),
         total_z_kernel_column=float(total[8]),
         reference_kernel_column0=float(kref[0]),reference_kernel_column2=float(kref[2]),
         reference_kernel_column8=float(kref[8]),
         geometric_kernel_column0=float(kgeo[0]),geometric_kernel_column2=float(kgeo[2]),
         geometric_kernel_column8=float(kgeo[8]),
         total_difference_column0=float(total[0]),total_difference_column2=float(total[2]))
    return r,terms,kref,kgeo,b_ref,c_ref,b_geo,c_geo

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
    ap.add_argument('--samples',type=int,default=33)
    ap.add_argument('--horizon-factor',type=float,default=1.,help='Relative to historical original T; >1 extrapolates beyond validated snapshots')
    ap.add_argument('--rtol',type=float,default=3e-12)
    ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--max-step',type=float,default=2.5e-7)
    ap.add_argument('--kappa-geo',type=float,default=35*math.pi*10/(12*550))
    args=ap.parse_args()
    if args.samples<3 or not math.isfinite(args.horizon_factor) or args.horizon_factor<.5 or args.horizon_factor>16:
        raise ValueError('samples>=3 and 0.5<=horizon-factor<=16 required')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    parent,inputs_sha=verify_parent_inputs()
    audit,support,tube=load_sources(args.base_root.resolve())
    params=audit.ModelParameters(gamma_rho=.3)
    z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
    ys=np.asarray(z['states'],float);gg=np.asarray(z['g'],float)
    sc=np.asarray(z['scale'],float);center=np.asarray(z['center'],float)
    labels=[str(v) for v in z['labels']]
    if ys.shape!=(36,11) or gg.shape!=(36,10) or sc.shape!=(10,):raise ValueError('Frozen NPZ shapes mismatch')
    kr=float(z['kappa_ref']);kg=float(args.kappa_geo);slope=float(z['slope'])
    if abs(kr-1/6)>1e-15 or abs(slope-float(parent['slope']))>1e-15:raise RuntimeError('Parent kappa or slope mismatch')
    if not all(np.all(np.isfinite(v)) for v in (ys,gg,sc)) or np.any(sc<=0):raise RuntimeError('bad source states')
    cases={}
    for i,lbl in enumerate(labels):
        case,family,xi,branch,frac=case_meta(lbl)
        cases.setdefault(case,[]).append((frac,i,xi,branch,family))
    if len(cases)!=12 or any(len(p)!=3 for p in cases.values()):raise RuntimeError('not 12 original cases with 3 snapshots')
    with (ROOT/'inputs'/'case_integrations.csv').open(newline='') as f:
        times={r['case_id']:float(r['T']) for r in csv.DictReader(f)}
    if set(times)!=set(cases):raise RuntimeError('Original case list differs')
    ww=np.r_[sc,1.]
    def make_ode(kappa):
        def fn(t,joined):
            y=joined[:11];g=joined[11:]
            fy=slope*np.asarray(support.support_rhs_float(y,kappa0=kappa,params=params,audit=audit),float)
            jy=slope*np.asarray(tube.float_support_jacobian(y,kappa0=kappa,params=params,audit=audit,base=support),float)
            chart_j=(jy/ww[:,None])*ww[None,:]
            return np.r_[fy,-chart_j.T@g]
        return fn
    f_ref=make_ode(kr);f_geo=make_ode(kg)
    rows=[];summary=[];state_outputs=[];parent_gradient_checks=[];max_match=0.;n_integrations=0
    for case in sorted(cases):
        refs=sorted(cases[case]);frac0,i,xi,branch,family=refs[0]
        if abs(frac0-.125)>1e-10:raise RuntimeError('Missing historical start fraction')
        T=times[case];start=frac0*T;stop=args.horizon_factor*T
        tt=np.unique(np.r_[np.linspace(start,stop,args.samples),[f*T for f,_,_,_,_ in refs if f*T<=stop]])
        joined=np.r_[ys[i],gg[i],0.]
        solr=solve_ivp(f_ref,(start,stop),joined,method='DOP853',rtol=args.rtol,atol=args.atol,
                       max_step=args.max_step,t_eval=tt)
        solg=solve_ivp(f_geo,(start,stop),joined,method='DOP853',rtol=args.rtol,atol=args.atol,
                       max_step=args.max_step,t_eval=tt)
        n_integrations+=2
        if not solr.success or not solg.success or len(solr.t)!=len(tt) or len(solg.t)!=len(tt):
            raise RuntimeError(f'Integration failed: {case} {solr.message} {solg.message}')
        row_ids=[]
        for j,time in enumerate(tt):
            yr,gr=solr.y[:11,j],solr.y[11:,j]
            yg,ggg=solg.y[:11,j],solg.y[11:,j]
            fr=slope*np.asarray(support.support_rhs_float(yr,kappa0=kr,params=params,audit=audit),float)
            fg=slope*np.asarray(support.support_rhs_float(yg,kappa0=kg,params=params,audit=audit),float)
            metrics,terms,ker_r,ker_g,*_=compute_row(yr,yg,gr,ggg,kr,kg,slope,params,sc,fr,fg)
            # Frozen-vs-moving-observer control: held initial source gradient,
            # not a reconstructed optimum at the updated state.
            href=structural_jet(yr,kr,params,slope)[2]
            hgeo=structural_jet(yg,kg,params,slope)[2]
            g_frozen=joined[11:]
            frozen_ref=chart_kernel(href,fr,sc,g_frozen[:10])
            frozen_geo=chart_kernel(hgeo,fg,sc,g_frozen[:10])
            metrics.update(reference_frozen_vs_adjoint_kernel=norm(ker_r-frozen_ref),
                geometric_frozen_vs_adjoint_kernel=norm(ker_g-frozen_geo))
            rr=dict(case_id=case,family=family,anchor_xi=xi,branch=branch,
                    time=time,relative_historical_T=time/T,window_T=T,
                    baseline_support_window=(time<=T+1e-14),kappa_ref=kr,kappa_geo=kg,
                    **metrics)
            rows.append(rr);row_ids.append(len(rows)-1)
            state_outputs.append(dict(case=case,t=time,yref=yr.copy(),ygeo=yg.copy(),gref=gr.copy(),ggeo=ggg.copy(),
                                      terms=np.array(terms),kernel_ref=ker_r,kernel_geo=ker_g))
            for frac,original_i,*_ in refs:
                if abs(time/T-frac)<2e-10:
                    error=norm(yr-ys[original_i])
                    max_match=max(max_match,error)
                    if error>2e-7:raise RuntimeError(f'Original restart does not replay at {case} f={frac:g}, absolute={error}')
                    parent_gradient_checks.append(dict(case_id=case,fraction=frac,
                        historical_source_gradient_norm=norm(gg[original_i]),
                        propagated_adjoint_vs_historical_gradient=norm(gr[:10]-gg[original_i]),
                        relative_observer_difference=rel(gr[:10],gg[original_i]),
                        propagated_phase_covector_component=gr[10]))
        rr=[rows[k] for k in row_ids]
        summary.append(dict(case_id=case,family=family,anchor_xi=xi,branch=branch,
                samples=len(rr),historical_T=T,
                median_reference_cancellation=med([r['reference_cancellation_ratio'] for r in rr]),
                median_geometric_cancellation=med([r['geometric_cancellation_ratio'] for r in rr]),
                maximum_reference_cancellation=max(r['reference_cancellation_ratio'] for r in rr),
                maximum_geometric_cancellation=max(r['geometric_cancellation_ratio'] for r in rr),
                t_at_reference_peak=max(rr,key=lambda r:r['reference_cancellation_ratio'])['relative_historical_T'],
                t_at_geometric_peak=max(rr,key=lambda r:r['geometric_cancellation_ratio'])['relative_historical_T'],
                median_delta_relative=med([r['difference_kernel_relative'] for r in rr]),
                max_delta_absolute=max(r['difference_kernel_norm'] for r in rr),
                max_state_separation=max(r['state_abs_difference'] for r in rr),
                end_state_separation=rr[-1]['state_abs_difference'],
                median_direct_hessian=med([r['direct_hessian_norm'] for r in rr]),
                median_state_hessian=med([r['state_hessian_norm'] for r in rr]),
                median_flow_change=med([r['flow_change_norm'] for r in rr]),
                median_dual_transport=med([r['dual_transport_norm'] for r in rr])))
        print(f'v18.48f {case}: {len(rr)} frames, Cmax ref={summary[-1]["maximum_reference_cancellation"]:.6g} geo={summary[-1]["maximum_geometric_cancellation"]:.6g}',flush=True)
    if len(summary)!=12 or n_integrations!=24:raise RuntimeError('audit not complete')
    write_csv(out/'transported_kernel_ledger.csv',rows)
    write_csv(out/'branch_localization.csv',summary)
    col=lambda key:np.stack([o[key] for o in state_outputs])
    np.savez_compressed(out/'transported_states_and_terms.npz',
        case_id=np.array([o['case'] for o in state_outputs]),time=np.array([o['t'] for o in state_outputs]),
        y_ref=col('yref'),y_geo=col('ygeo'),g_ref=col('gref'),g_geo=col('ggeo'),
        telescope_terms=col('terms'),kernel_ref=col('kernel_ref'),kernel_geo=col('kernel_geo'),
        chart_scale=sc,chart_center=center)
    branch_pairs=[]
    for fam in sorted(set(r['family'] for r in summary)):
        for xi in sorted(set(r['anchor_xi'] for r in summary if r['family']==fam)):
            a,b=sorted([r for r in summary if r['family']==fam and r['anchor_xi']==xi],key=lambda r:r['branch'])
            branch_pairs.append(dict(family=fam,anchor_xi=xi,
                reference_branch0_cmax=a['maximum_reference_cancellation'],reference_branch1_cmax=b['maximum_reference_cancellation'],
                geometric_branch0_cmax=a['maximum_geometric_cancellation'],geometric_branch1_cmax=b['maximum_geometric_cancellation'],
                ratio_of_peak_branch0_to_branch1=a['maximum_reference_cancellation']/max(b['maximum_reference_cancellation'],1e-30),
                absolute_peak_difference=a['maximum_reference_cancellation']-b['maximum_reference_cancellation']))
    write_csv(out/'branch_pairs.csv',branch_pairs)
    write_csv(out/'historical_observer_comparison.csv',parent_gradient_checks)
    focus=[r for r in rows if r['family'].startswith('phaseblind') and r['anchor_xi']==3.5]
    report=dict(version='v18.48f',status='ORIGINAL_Q550_ANALYTIC_LOCAL_CANCELLATION_TRANSPORT',
      provenance='v18.48e 36 authenticated original snapshots; first snapshot in each case used as common restart',
      input_sha256=inputs_sha,source_sha256={k:val[1] for k,val in PINNED.items()},
      parent_report_sha256=inputs_sha['report.json'],parent_auth=parent['parent_auth'],
      cases=len(summary),samples=len(rows),fresh_integrations=n_integrations,
      max_original_restart_snapshot_absolute_error=max_match,
      max_transported_vs_historical_source_gradient_relative=max(r['relative_observer_difference'] for r in parent_gradient_checks),
      max_telescope_relative_error=max(r['telescope_budget_relative_error'] for r in rows),
      max_direct_identity_relative_error=max(r['direct_analytic_identity_relative_error'] for r in rows),
      original_kappa=kr,geometric_kappa=kg,delta_kappa=kg-kr,
      horizon_factor=args.horizon_factor,min_historical_T=min(times.values()),max_historical_T=max(times.values()),
      max_reference_cancellation=max(r['reference_cancellation_ratio'] for r in rows),
      max_geometric_cancellation=max(r['geometric_cancellation_ratio'] for r in rows),
      median_reference_cancellation=med([r['reference_cancellation_ratio'] for r in rows]),
      median_geometric_cancellation=med([r['geometric_cancellation_ratio'] for r in rows]),
      median_kernel_delta_relative=med([r['difference_kernel_relative'] for r in rows]),
      max_state_absolute_difference=max(r['state_abs_difference'] for r in rows),
      focus_phaseblind_xi35=dict(samples=len(focus),
          ref_max_cancellation=max(r['reference_cancellation_ratio'] for r in focus),
          geo_max_cancellation=max(r['geometric_cancellation_ratio'] for r in focus)),
      branch_pair_comparison=branch_pairs,
      limitations=('Times are 0.125*T to horizon_factor*T, historical T roughly 1e-6; local perturbation transport, not long-time hysteresis. '
      '11D tangent-dual transport is a moving covector in the original frozen chart, not a reoptimized observer, '
      'not a mechanical normal invariant. Kernel observable projects first 10 chart dimensions. '
      'Algebraic Hessians evaluated in binary64 without interval enclosures; geometric kappa is a prescribed comparison, not validated physical law. '
      'When horizon_factor>1, times beyond T extrapolate beyond the authenticated historical endpoint.'))
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','samples','fresh_integrations','max_original_restart_snapshot_absolute_error','max_telescope_relative_error','focus_phaseblind_xi35')},indent=2))

if __name__=='__main__':main()
