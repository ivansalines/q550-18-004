#!/usr/bin/env python3
"""v18.48i state/observer/coupling attribution using *authentic* v18.48h IVPs.

Two experiments:
 A) Time 2-cube: y(t0) vs y(t), transported/recomputed covector g(t0)
    vs g(t), at one frozen kappa, on the same genuine Q550 seed IVP.
 B) Kappa 3-cube at a common clock: original vs autonomous state, covector,
    and kappa coefficient, on paired genuine Q550 seed IVPs.

The hybrid evaluations are counterfactual in the ORIGINAL frozen chart, not
true trajectory points or a proof of a canonical cotangent identification.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from factorial_math import shapley,second_interaction,diagnostics
from extract_parent import extract,sha
ROOT=Path(__file__).resolve().parents[1]


def nrm(a):return float(np.linalg.norm(a))

def write_csv(path,rows):
 if not rows:raise RuntimeError('empty output '+str(path))
 keys=list(rows[0])
 if any(set(r)!=set(keys) for r in rows):raise RuntimeError('nonrectangular rows '+str(path))
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def load_paths(p,meta):
 data=np.load(p,allow_pickle=False)
 keys=('families','seed_branches','kappa_kind','xi','y','covectors','chart_scale','clock_speed')
 if any(k not in data for k in keys):raise RuntimeError('Missing parent physical path field')
 N=len(data['xi']);assert N%8==0 and data['y'].shape==(N,11) and data['covectors'].shape==(N,5,11)
 sc=np.asarray(data['chart_scale'],float)
 if sc.shape!=(10,) or np.any(sc<=0):raise RuntimeError('Frozen chart invalid')
 records={}
 for i in range(N):
  key=(str(data['families'][i]),int(data['seed_branches'][i]),str(data['kappa_kind'][i]),round(float(data['xi'][i]),10))
  if key in records:raise RuntimeError('duplicate physical sample '+str(key))
  records[key]=(np.asarray(data['y'][i],float),np.asarray(data['covectors'][i],float))
 xis=sorted({k[-1] for k in records})
 families=sorted({k[0] for k in records})
 if len(families)!=2 or len(records)!=len(families)*2*2*len(xis):raise RuntimeError('Incomplete continuation')
 for a in (3.5,4.,4.25,4.5):
  if not any(abs(x-a)<1e-8 for x in xis):raise RuntimeError('Missing clock section '+str(a))
 return records,xis,families,sc,float(data['clock_speed'])


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--parent-results',type=Path,default=None,help='Existing SHA-authenticated v18.48h output or new genuine continuation')
 ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
 ap.add_argument('--floor-relative',type=float,default=1e-12)
 args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 parent,pin=extract()
 sys.path.insert(0,str(parent/'scripts'))
 from structural_jet import structural_jet,coupling_jet,chart_kernel
 import run_audit as h
 p=(args.parent_results or parent/'results').resolve()
 if not p.is_dir():raise RuntimeError('Parent results missing '+str(p))
 # The parent numerical integration report, not the preflight, is authoritative.
 report=json.loads((p/'report.json').read_text())
 if report.get('version')!='v18.48h' or not report.get('preserved_seed_branch_identity'):
  raise RuntimeError('Not a verified v18.48h identity-preserving IVP report')
 if report.get('max_clock_section_absolute_error',1)>2e-10:
  raise RuntimeError('Clock section drift')
 # The user-server source is authenticated by the parent zip manifest. A new
 # dense integration will be additionally pinned by its newly computed SHA.
 record,xis,families,scale,clock=load_paths(p/'physical_paths.npz',report)
 provenance={'v18_48h_original_server_zip_sha256':pin['zip_sha256'],
             'v18_48h_verified_archive_members':pin['verified_members'],
             'actual_parent_paths_sha256':sha((p/'physical_paths.npz').read_bytes()),
             'actual_parent_report_sha256':sha((p/'report.json').read_bytes()),
             'new_refined_ivp':str(p)!=str((parent/'results').resolve()),
             'parent_classification':report['status']}
 # No rewritten dynamics: analytic jet was pinned in the authenticated parent.
 kvalues={'original':float(report['kappa_original']),
          'autonomous':float(report['kappa_autonomous'])}
 slope=clock/(2*math.pi)
 # Structural ModelParameters: reference to the original imported source
 import importlib.util
 audit_path=parent/'reference/test_support_mobius_monodromy_audit.py'
 sp=importlib.util.spec_from_file_location('v1848i_original_audit',audit_path)
 audit=importlib.util.module_from_spec(sp);sys.modules[sp.name]=audit;sp.loader.exec_module(audit)
 params=audit.ModelParameters(gamma_rho=.3)
 if abs(kvalues['original']-1/6)>1e-13 or abs(kvalues['autonomous']-35*math.pi/660)>1e-13:
  raise RuntimeError('Kappa reference drift')
 # Precompute the exact analytical kernel maps once per physical (state,kappa).
 # The frozen-chart diagnostic is linear in the observer; this matters because
 # the factorial experiment deliberately reuses states with different cotangents.
 matrix_cache={}
 def kernel_matrices(y,kap):
  key=(np.asarray(y,dtype=float).tobytes(),float(kap))
  if key not in matrix_cache:
   f,j,htot=structural_jet(y,kap,params,slope)
   hb=structural_jet(y,0.,params,slope)[2]
   hc=slope*kap*coupling_jet(y,params)[2]
   def matrix(hs):
    # chart_kernel(H,f,scale,g[:10]) == normalized(g[:10]) @ M
    t=np.einsum('iab,a->ib',hs[:10],f)[:,:10]
    return t*scale[None,:]/scale[:,None]
   matrix_cache[key]=(matrix(hb),matrix(hc))
  return matrix_cache[key]
 def terms(y,g,kap):
  gg=np.asarray(g,float)
  if gg.shape!=(11,) or not np.all(np.isfinite(gg)) or nrm(gg)<1e-22:raise RuntimeError('bad observer')
  gg=gg/nrm(gg)
  mb,mc=kernel_matrices(y,kap)
  B=gg[:10]@mb;C=gg[:10]@mc;S=B+C
  stat=diagnostics(B,C,args.floor_relative)
  stat['observer_norm_input']=nrm(g)
  return stat,B,C,S
 # Independent check that the optimized frozen-chart matrices exactly match
 # the historical analytic Hessian contraction (five authentic states).
 kernel_matrix_max_relative=0.
 delta_z_max_abs=0.
 for key,(yy,cc) in list(record.items())[::max(len(record)//5,1)][:5]:
  kap=kvalues[key[2]]
  gtest=cc[0]/nrm(cc[0])
  f0,j0,h0=structural_jet(yy,kap,params,slope)
  hb0=structural_jet(yy,0.,params,slope)[2]
  hc0=slope*kap*coupling_jet(yy,params)[2]
  m0,m1=kernel_matrices(yy,kap)
  refB=chart_kernel(hb0,f0,scale,gtest[:10]);refC=chart_kernel(hc0,f0,scale,gtest[:10])
  err=max(nrm(gtest[:10]@m0-refB),nrm(gtest[:10]@m1-refC))
  kernel_matrix_max_relative=max(kernel_matrix_max_relative,err/max(nrm(refB)+nrm(refC),1e-30))
  delta_z_max_abs=max(delta_z_max_abs,float(np.max(np.abs(hc0[:,8,:]))),float(np.max(np.abs(hc0[:,:,8]))))
 if kernel_matrix_max_relative>3e-11 or delta_z_max_abs>0:
  raise RuntimeError('Original analytic Hessian kernel identity or structural z-null failed')
 def obs(mode,y,cov,kap):
  if mode=='historical_transport':return cov[0]
  if mode=='activity_sin_lie_transport':return cov[3]
  if mode=='activity_cos_lie_transport':return cov[4]
  if mode=='activity_sin_lie_recomputed':return h.activity_lie_gradient(y,params,scale,'sin',kap,slope)
  if mode=='activity_cos_lie_recomputed':return h.activity_lie_gradient(y,params,scale,'cos',kap,slope)
  if mode=='activity_sin_instant_null':return h.activity_gradient(y,params,scale,'sin')
  raise ValueError(mode)
 modes=['historical_transport','activity_sin_lie_transport','activity_cos_lie_transport',
        'activity_sin_lie_recomputed','activity_cos_lie_recomputed','activity_sin_instant_null']
 # Anchored long time windows and all local neighboring pairs; all endpoints
 # are actual IVP solutions. No interpolation of state, observer or covector.
 time_rows=[];peak_rows=[]
 for fam in families:
  for branch in (0,1):
   for kind,kap in kvalues.items():
    for mode in modes:
     for t0,t1,interval_kind in [(3.5,x,'origin_3_5') for x in xis[1:]]+[(4.,x,'crest_4_0') for x in xis if x>4.00000001]+[(a,b,'adjacent') for a,b in zip(xis[:-1],xis[1:])]:
      y0,cv0=record[(fam,branch,kind,round(t0,10))]
      y1,cv1=record[(fam,branch,kind,round(t1,10))]
      g0=obs(mode,y0,cv0,kap);g1=obs(mode,y1,cv1,kap)
      corners=[]
      for s,o in [(0,0),(1,0),(0,1),(1,1)]:
       stat,B,C,S=terms(y1 if s else y0,g1 if o else g0,kap)
       corners.append((stat,B,C,S))
      vals=[v[0]['log_cancellation'] for v in corners]
      contrib=shapley(vals,2)
      err=abs(sum(float(x) for x in contrib)-(vals[3]-vals[0]))
      if err>1e-9:raise RuntimeError('Two-factor scalar Shapley closure')
      if mode=='activity_sin_instant_null' and (not corners[0][0]['is_coupling_null'] or not corners[3][0]['is_coupling_null']):
       raise RuntimeError('Structural null observer invalid')
      vecs=[v[3] for v in corners];veccontrib=shapley(vecs,2)
      verr=nrm(sum(veccontrib)-vecs[3]+vecs[0])/max(nrm(vecs[3]-vecs[0]),nrm(vecs[3]),1e-30)
      if verr>3e-11:raise RuntimeError('Two-factor vector identity')
      base=dict(family=fam,seed_branch=branch,kappa_kind=kind,mode=mode,
                interval_kind=interval_kind,xi_start=t0,xi_end=t1,
                same_genuine_seed_IVP=True,
                frozen_chart_cross_evaluation_not_physical_state=True,
                log_cancel_start=vals[0],log_cancel_state_only=vals[1],
                log_cancel_observer_only=vals[2],log_cancel_end=vals[3],
                cancellation_ratio_start=corners[0][0]['cancellation_ratio'],
                cancellation_ratio_state_only=corners[1][0]['cancellation_ratio'],
                cancellation_ratio_observer_only=corners[2][0]['cancellation_ratio'],
                cancellation_ratio_end=corners[3][0]['cancellation_ratio'],
                log_cancel_change=vals[3]-vals[0],
                log_cancel_state_shapley=float(contrib[0]),
                log_cancel_observer_shapley=float(contrib[1]),
                state_observer_interaction_log=float(second_interaction(*vals)),
                resultant_kernel_change_norm=nrm(vecs[3]-vecs[0]),
                state_kernel_shapley_norm=nrm(veccontrib[0]),
                observer_kernel_shapley_norm=nrm(veccontrib[1]),
                vector_closure_error=verr,
                normalized_covector_cosine=float(np.dot(g0,g1)/(nrm(g0)*nrm(g1))),
                state_distance_euclidean=nrm(y1-y0),
                resultant_norm_end=corners[3][0]['resultant_norm'],
                base_norm_end=corners[3][0]['base_norm'],
                coupling_norm_end=corners[3][0]['coupling_norm'],
                opposition_cosine_end=corners[3][0]['opposition_cosine'],
                structural_coupling_null_end=corners[3][0]['is_coupling_null'],
                near_singular_resultant_end=corners[3][0]['near_singular_resultant'])
      time_rows.append(base)
      if abs(t0-4.)<1e-9 and abs(t1-4.25)<1e-8:
       peak_rows.append(base)
 write_csv(out/'time_state_observer_factorial.csv',time_rows)
 write_csv(out/'crest_attribution_xi400_to_425.csv',peak_rows)
 # Three-corner family: factor 0 state, 1 observer, 2 kappa.
 krows=[]
 for fam in families:
  for branch in (0,1):
   for xi in xis:
    y0,c0=record[(fam,branch,'original',round(xi,10))]
    y1,c1=record[(fam,branch,'autonomous',round(xi,10))]
    for mode in modes:
     g0=obs(mode,y0,c0,kvalues['original']);g1=obs(mode,y1,c1,kvalues['autonomous'])
     corners=[]
     for mask in range(8):
      sy=y1 if (mask&1) else y0
      og=g1 if (mask&2) else g0
      kp=kvalues['autonomous'] if mask&4 else kvalues['original']
      corners.append(terms(sy,og,kp))
     vals=[v[0]['log_cancellation'] for v in corners]
     con=shapley(vals,3)
     rerr=abs(sum(float(x) for x in con)-(vals[7]-vals[0]))
     if rerr>1e-9:raise RuntimeError('Three-factor Shapley nonclosure')
     kvectors=[v[3] for v in corners]
     vcon=shapley(kvectors,3)
     verr=nrm(sum(vcon)-kvectors[7]+kvectors[0])/max(nrm(kvectors[7]),nrm(kvectors[0]),1e-30)
     if verr>3e-11:raise RuntimeError('Three-factor kernel Shapley nonclosure')
     krows.append(dict(family=fam,seed_branch=branch,mode=mode,xi=xi,
        kappa_ref=kvalues['original'],kappa_autonomous=kvalues['autonomous'],
        original_ratio=corners[0][0]['cancellation_ratio'],
        autonomous_ratio=corners[7][0]['cancellation_ratio'],
        original_log_ratio=vals[0],autonomous_log_ratio=vals[7],
        delta_log_ratio=vals[7]-vals[0],
        state_shapley=float(con[0]),observer_shapley=float(con[1]),kappa_shapley=float(con[2]),
        state_kernel_shapley_norm=nrm(vcon[0]),observer_kernel_shapley_norm=nrm(vcon[1]),
        kappa_kernel_shapley_norm=nrm(vcon[2]),
        remainder_scalar_shapley_closure=rerr,
        remainder_vector_shapley_closure=verr,
        observer_cosine=float(np.dot(g0,g1)/(nrm(g0)*nrm(g1))),
        state_distance=nrm(y1-y0),
        coupling_null_at_end=corners[7][0]['is_coupling_null']))
 write_csv(out/'kappa_three_factor.csv',krows)
 # Independent reference check on frozen original y and observer at start.
 summary={}
 for mode in modes:
  rows=[r for r in peak_rows if r['mode']==mode]
  if not rows:raise RuntimeError('Crest sample missing '+mode)
  target=[r for r in rows if r['family'].startswith('phasecov') and r['seed_branch']==0 and r['kappa_kind']=='original'][0]
  summary[mode]={k:target[k] for k in ('cancellation_ratio_start','cancellation_ratio_state_only','cancellation_ratio_observer_only',
                                    'cancellation_ratio_end','log_cancel_change','log_cancel_state_shapley','log_cancel_observer_shapley',
                                    'state_observer_interaction_log','normalized_covector_cosine','resultant_norm_end','near_singular_resultant_end')}
 kp=[r for r in krows if r['family'].startswith('phasecov') and r['seed_branch']==0 and r['mode']=='historical_transport' and abs(r['xi']-4.25)<1e-8][0]
 report_i=dict(version='v18.48i',status='ANALYTIC_HESSIAN_GENUINE_IVP_FROZEN_CHART_FACTORIAL_DIAGNOSTIC',
    parent=provenance,physical_parent_ivp_count=report['genuinely_integrated_IVPs'],
    sections=len(xis),families=families,time_factorial_rows=len(time_rows),kappa_factorial_rows=len(krows),
    tested_kappas=kvalues,crest_phasecov_branch0_original=summary,
    crest_phasecov_branch0_kappa_3factor={k:kp[k] for k in ('original_ratio','autonomous_ratio','delta_log_ratio','state_shapley','observer_shapley','kappa_shapley')},
    clock_speed=clock,
    max_optimized_kernel_vs_independent_analytic_relative=kernel_matrix_max_relative,
    max_direct_kappa_coupling_z_hessian_abs=delta_z_max_abs,
    zero_control='Instantaneous activity gradients have zero projected coupling Hessian by algebra; their cancellation ratio=1 is NOT evidence of material cancellation.',
    attribution_boundary='Two-/three-cube Shapley attribution is exact arithmetic on deliberately cross-evaluated frozen-chart kernels. Cross-evaluations are NOT physical IVP states or canonical dual transports between distinct tangent spaces. Shapley contributions are not causal mechanism strengths or coordinate invariants. Source branches remain independent seeds; no BVP or interval certificate.',
    history_boundary='The evaluated xi3.5..4.5 paths are genuine numerical Q550 IVPs from v18.48h (optionally resampled by its unchanged model); phase clock and historical seed offset caveats remain.',
    numeric_boundary='Analytic formula evaluated in binary64; no interval enclosure. Near-zero resultant is flagged separately from strong antiparallelism.')
 (out/'report.json').write_text(json.dumps(report_i,indent=2,allow_nan=False)+'\n')
 (out/'source_verification.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(json.dumps(dict(status=report_i['status'],sections=len(xis),time_rows=len(time_rows),kappa_rows=len(krows),
                       crest_historical=summary['historical_transport'],
                       crest_kappa=report_i['crest_phasecov_branch0_kappa_3factor']),indent=2))
if __name__=='__main__':main()
