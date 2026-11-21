#!/usr/bin/env python3
"""Q550 v18.48g: phase-gauge-matched branch interpolation and actual local cotangent replay.

This is an EXPLORATORY homotopy, NOT certification of a physical branch continuation.
Only inputs at xi=3.5,4.0,4.5 are original states. All intermediate
xi are constructed by declared interpolation, not observed on Q550.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import run_audit_v18_48f as frozen
from structural_jet import structural_jet, coupling_jet, chart_kernel

ROOT=Path(__file__).resolve().parents[1]
TAU=2*math.pi
ANGULAR=(2,6,10)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cnorm(x):return float(np.linalg.norm(x))
def cos(a,b):
 na,nb=cnorm(a),cnorm(b)
 return float(np.dot(a,b)/(na*nb)) if na*nb>1e-30 else None

def wrap(x):return (float(x)+math.pi)%(2*math.pi)-math.pi

def phase(y,params=None):return float(y[2]+y[6]-(params.phi0 if params is not None else 0.)-y[10])

def gauge_align(anchor,target):
 """Exact symmetry on real arithmetic: (theta1,Theta) -> (theta1+c,Theta+c).
 Keep target angles on shortest 2pi lifts; the two extra shifts preserve RHS.
 """
 a=np.asarray(anchor,dtype=float);b=np.array(target,dtype=float,copy=True)
 shift=a[10]-b[10];b[2]+=shift;b[10]+=shift
 integers={}
 for j in (2,6):
  m=int(np.rint((a[j]-b[j])/TAU)); b[j]+=TAU*m; integers[str(j)]=m
 return b,dict(continuous_gauge_shift=shift,integer_lifts=integers,
               theta_match_error=float(b[10]-a[10]),
               relative_phase_mod_error=wrap(phase(b)-phase(target)))

def interpolate_state(a,b,alpha,method='internal'):
 if method=='internal':return (1-alpha)*a+alpha*b
 if method=='positive_radii':
  out=(1-alpha)*a+alpha*b
  for j in (0,4):
   # rho=e^y; interpolate rho instead of y, then map back.
   out[j]=math.log((1-alpha)*math.exp(a[j])+alpha*math.exp(b[j]))
  return out
 raise ValueError(method)

def interpolate_observer(ga,gb,alpha):
 # Frozen chart cotangents interpolated in the declared chart; no re-optimization.
 return (1-alpha)*ga+alpha*gb

def cancel_at(y,g,kappa,params,slope,sc,audit,support):
 fy=slope*np.asarray(support.support_rhs_float(y,kappa0=kappa,params=params,audit=audit),float)
 hbase=structural_jet(y,0.,params,slope)[2]
 hc=slope*kappa*coupling_jet(y,params)[2]
 kb=chart_kernel(hbase,fy,sc,g[:10]);kc=chart_kernel(hc,fy,sc,g[:10]);total=kb+kc
 out=frozen.cancellation_stats(kb,kc)
 return out,fy,kb,kc,total

def match_distance(ya,ga,yb,gb,sc,params):
 aligned,_=gauge_align(ya,yb)
 # Chart distance uses first-ten frozen chart scales; angle 10 is a phase gauge.
 weighted=float(np.sqrt(np.mean(((aligned[:10]-ya[:10])/sc)**2)))
 pd=abs(wrap(phase(aligned,params)-phase(ya,params)))/math.pi
 c=cos(ga,gb)
 obs=(1-c) if c is not None else float('inf')
 return dict(chart_distance=weighted,relative_phase_distance=pd,observer_misalignment=obs,
             combined=weighted+pd+10*obs)

def assignment(rowsA,rowsB,sc,params):
 choices=[]
 for perm in ((0,1),(1,0)):
  parts=[]
  for i in (0,1):
   ya,ga=rowsA[i];yb,gb=rowsB[perm[i]]
   parts.append(match_distance(ya,ga,yb,gb,sc,params))
  entry=dict(mapping=f'{perm[0]},{perm[1]}',permutation=list(perm))
  for key in ('chart_distance','relative_phase_distance','observer_misalignment','combined'):
   entry[key]=sum(x[key] for x in parts)
  choices.append(entry)
 results={}
 for metric in ('chart_distance','relative_phase_distance','observer_misalignment','combined'):
  sortedchoices=sorted(choices,key=lambda c:c[metric]);gap=sortedchoices[1][metric]-sortedchoices[0][metric]
  results[metric]=dict(preferred=sortedchoices[0]['mapping'],margin=float(gap),
                       relative_margin=float(gap/max(abs(sortedchoices[1][metric]),1e-30)))
 return choices,results

def output_csv(path,rows):
 if not rows:raise ValueError('empty: '+str(path))
 with open(path,'w',newline='',encoding='utf8') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def verify_bundle_sources(base):
 # Exact inherited source and snapshot pins, no silent fallback.
 frozen.verify_parent_inputs()
 for f in ['v18_48f_transport_ledger.csv','v18_48f_branch_pairs.csv','v18_48f_report.json']:
  if digest(ROOT/'inputs'/f)!=json.loads((ROOT/'inputs'/'V18_48F_SHA256.json').read_text())['files'][f]:
   raise RuntimeError('Frozen v18.48f record changed: '+f)
 return frozen.load_sources(base)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
 ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
 ap.add_argument('--alpha-points',type=int,default=9)
 ap.add_argument('--rtol',type=float,default=3e-12)
 ap.add_argument('--atol',type=float,default=3e-13)
 ap.add_argument('--max-step',type=float,default=2.5e-7)
 ap.add_argument('--kappa-geometric',type=float,default=35*math.pi*10/(12*550))
 args=ap.parse_args()
 if args.alpha_points<3 or args.alpha_points>65:raise ValueError('alpha-points must be 3..65')
 out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 audit,support,tube=verify_bundle_sources(args.base_root.resolve())
 params=audit.ModelParameters(gamma_rho=.3)
 d=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
 sc=np.asarray(d['scale'],float); slope=float(d['slope']); kref=float(d['kappa_ref']);kgeo=float(args.kappa_geometric)
 if sc.shape!=(10,) or not np.all(sc>0) or abs(kref-1/6)>1e-15:raise RuntimeError('Wrong chart or original kappa')
 labels=[str(i) for i in d['labels']]
 cases={}
 for i,label in enumerate(labels):
  case,fam,xi,branch,frac=frozen.case_meta(label)
  if abs(frac-.125)<1e-10:
   cases[(fam,xi,branch)]=(np.asarray(d['states'][i],float),np.asarray(d['g'][i],float))
 if len(cases)!=12:raise RuntimeError('Need exactly 12 cases x initial phase')
 with (ROOT/'inputs'/'case_integrations.csv').open(newline='') as f:
  tvalues={r['case_id']:float(r['T']) for r in csv.DictReader(f)}
 def caseT(fam,xi,b):return tvalues[f'{fam}|xi={xi:g}|branch={b}']
 families=sorted(set(k[0] for k in cases));xis=sorted(set(k[1] for k in cases))
 assert len(families)==2 and xis==[3.5,4.0,4.5]
 # Gauge symmetry independently checked against ORIGINAL support RHS and analytic jets.
 max_gauge_rhs=0.;max_gauge_h=0.;max_gauge_delta=0.
 for fam in families:
  for xi in xis:
   a,ga=cases[(fam,xi,0)]; b,gb=cases[(fam,xi,1)]
   t,meta=gauge_align(a,b)
   f0=np.asarray(support.support_rhs_float(b,kappa0=kref,params=params,audit=audit),float)
   f1=np.asarray(support.support_rhs_float(t,kappa0=kref,params=params,audit=audit),float)
   h0=structural_jet(b,kref,params,slope)[2];h1=structural_jet(t,kref,params,slope)[2]
   max_gauge_rhs=max(max_gauge_rhs,cnorm(f1-f0)/max(cnorm(f0),1e-30))
   max_gauge_h=max(max_gauge_h,cnorm(h1-h0)/max(cnorm(h0),1e-30))
   max_gauge_delta=max(max_gauge_delta,abs(wrap(phase(t,params)-phase(b,params))))
 if max_gauge_rhs>1e-8 or max_gauge_h>1e-8:
  raise RuntimeError(f'Gauge shift changed model: RHS={max_gauge_rhs}, Hessian={max_gauge_h}')

 assignrows=[];assignresults=[];seedrows=[];transport=[];payload=[];profiles=[]
 total_integrations=0;max_endpoint_error=0.;max_direct=0.
 ww=np.r_[sc,1.]
 def flow(kappa):
  def fun(t,w):
   y=w[:11];g=w[11:]
   fy=slope*np.asarray(support.support_rhs_float(y,kappa0=kappa,params=params,audit=audit),float)
   jy=slope*np.asarray(tube.float_support_jacobian(y,kappa0=kappa,params=params,audit=audit,base=support),float)
   chartJ=(jy/ww[:,None])*ww[None,:]
   return np.r_[fy,-chartJ.T@g]
  return fun
 f0=flow(kref);f1=flow(kgeo)
 hist={}
 with (ROOT/'inputs'/'v18_48f_transport_ledger.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   if abs(float(r['relative_historical_T'])-.125)<1e-8:
    hist[(r['family'],float(r['anchor_xi']),int(r['branch']))]=r
 for fam in families:
  for xl,xr in zip(xis[:-1],xis[1:]):
   A=[cases[(fam,xl,b)] for b in (0,1)];B=[cases[(fam,xr,b)] for b in (0,1)]
   choices,ranks=assignment(A,B,sc,params)
   for c in choices:assignrows.append(dict(family=fam,xi_start=xl,xi_end=xr,**{k:v for k,v in c.items() if k!='permutation'}))
   chosen=ranks['combined']['preferred'];assignresults.append(dict(family=fam,xi_start=xl,xi_end=xr,chosen=chosen,criteria=ranks))
   for strategy in ('identity','swap'):
    perm=(0,1) if strategy=='identity' else (1,0)
    for track in (0,1):
     ya,ga=A[track];yb,gb=B[perm[track]]
     yb,meta=gauge_align(ya,yb)
     Ta,Tb=caseT(fam,xl,track),caseT(fam,xr,perm[track])
     for ia,alpha in enumerate(np.linspace(0.,1.,args.alpha_points)):
      x=(1-alpha)*xl+alpha*xr
      y=interpolate_state(ya,yb,alpha)
      g=interpolate_observer(ga,gb,alpha)
      if not np.all(np.isfinite(y)) or not np.all(np.isfinite(g)):raise RuntimeError('nonfinite seed')
      # All strategies evaluated instantaneously, two coordinate interpolation rules.
      stats={}
      for method in ('internal','positive_radii'):
       ymethod=interpolate_state(ya,yb,alpha,method)
       cr,_,br,ccr,kr=cancel_at(ymethod,np.r_[g,0.],kref,params,slope,sc,audit,support)
       cg,_,bg,ccg,kg=cancel_at(ymethod,np.r_[g,0.],kgeo,params,slope,sc,audit,support)
       stats[method]=(cr,cg,kr,kg)
      cr,cg,kbr,kbg=stats['internal']
      seedrows.append(dict(family=fam,xi_start=xl,xi_end=xr,matching=strategy,track=track,
       start_branch=track,end_branch=perm[track],alpha=alpha,xi_interpolated=x,
       source_kind='INTERPOLATED_SEED_NOT_ORIGINAL_Q550',
       phase_lift=meta['continuous_gauge_shift'],
       theta1_integer_lift=meta['integer_lifts']['2'],theta2_integer_lift=meta['integer_lifts']['6'],
       reference_C=cr['cancellation_ratio'],geometric_C=cg['cancellation_ratio'],
       reference_base_norm=cr['base_norm'],reference_coupling_norm=cr['coupling_norm'],
       reference_resultant_norm=cr['total_norm'],reference_opposition_cosine=cr['opposition_cosine'],
       geometric_base_norm=cg['base_norm'],geometric_coupling_norm=cg['coupling_norm'],
       geometric_resultant_norm=cg['total_norm'],
       alt_radii_reference_C=stats['positive_radii'][0]['cancellation_ratio'],
       alt_radii_geometric_C=stats['positive_radii'][1]['cancellation_ratio']))
      if strategy != ('identity' if chosen=='0,1' else 'swap'):continue
      # Authentic RHS integration ONLY along the explicitly constructed seed homotopy.
      T=(1-alpha)*Ta+alpha*Tb; t0=.125*T;te=T
      w0=np.r_[y,g,0.]
      for klabel,kap,ode in (('original',kref,f0),('autonomous',kgeo,f1)):
       tt=np.array([t0,(t0+te)/2,te]);sol=solve_ivp(ode,(t0,te),w0,t_eval=tt,method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step)
       total_integrations+=1
       if not sol.success or len(sol.t)!=3:raise RuntimeError('integrator failure: '+sol.message)
       for j,t in enumerate(sol.t):
        yy=sol.y[:11,j];gv=sol.y[11:,j];st,fy,kb,kc,kt=cancel_at(yy,gv,kap,params,slope,sc,audit,support)
        rr=dict(family=fam,xi_start=xl,xi_end=xr,matching=strategy,track=track,end_branch=perm[track],
         alpha=alpha,xi_interpolated=x,kappa_kind=klabel,time_fraction=float(t/T),
         historical_T_interpolated=T,original_anchor=(alpha==0. or alpha==1.),
         state_norm=cnorm(yy),observer_norm=cnorm(gv),
         base_norm=st['base_norm'],coupling_norm=st['coupling_norm'],resultant_norm=st['total_norm'],
         cancellation_ratio=st['cancellation_ratio'],opposition_cosine=st['opposition_cosine'],
         kernel_col0=float(kt[0]),kernel_col2=float(kt[2]),kernel_col8=float(kt[8]),
         relative_phase_wrapped=wrap(phase(yy,params)),global_theta_lifted=float(yy[10]))
        transport.append(rr)
        payload.append((yy.copy(),gv.copy(),kb.copy(),kc.copy(),kt.copy()))
        if j==0 and (alpha==0. or alpha==1.):
         xi=xl if alpha==0 else xr;b=track if alpha==0 else perm[track]
         expected=float(hist[(fam,xi,b)][f'{"reference" if klabel=="original" else "geometric"}_cancellation_ratio'])
         max_endpoint_error=max(max_endpoint_error,abs(st['cancellation_ratio']-expected)/max(expected,1e-30))
       if total_integrations%16==0:print(f'v18.48g completed {total_integrations} integrations',flush=True)
   # Both matching hypotheses remain in the CSV; no claim of a physical branch linkage.

 output_csv(out/'branch_assignment_costs.csv',assignrows)
 output_csv(out/'interpolation_controls.csv',seedrows)
 output_csv(out/'phase_matched_transport.csv',transport)
 np.savez_compressed(out/'transported_arrays.npz',
  y=np.stack([p[0] for p in payload]),g=np.stack([p[1] for p in payload]),
  K_base=np.stack([p[2] for p in payload]),K_coupling=np.stack([p[3] for p in payload]),
  K_total=np.stack([p[4] for p in payload]))
 # Cancellation transfer in each family from the two tracks, with interpolation strategy caveat.
 for strategy in ('identity','swap'):
  for fam in families:
   subset=[r for r in seedrows if r['family']==fam and r['matching']==strategy]
   for xl,xr in zip(xis[:-1],xis[1:]):
    local=[r for r in subset if r['xi_start']==xl and r['xi_end']==xr]
    for kcol in ('reference_C','geometric_C'):
     for method,prefix in ((kcol,'internal'),('alt_radii_'+('reference' if kcol=='reference_C' else 'geometric')+'_C','positive_radii')):
      ordered=sorted(set(r['alpha'] for r in local))
      pairs=[]
      for alpha in ordered:
       r0=next(r for r in local if r['track']==0 and r['alpha']==alpha)
       r1=next(r for r in local if r['track']==1 and r['alpha']==alpha)
       diff=math.log(r0[method])-math.log(r1[method])
       pairs.append((alpha,diff))
      crossings=[]
      for (a,da),(b,db) in zip(pairs[:-1],pairs[1:]):
       if da*db<0:
        alpha=a+(b-a)*abs(da)/(abs(da)+abs(db))
        crossings.append(round(xl+(xr-xl)*alpha,9))
      profiles.append(dict(family=fam,xi_start=xl,xi_end=xr,matching=strategy,
       param=kcol,interpolation=prefix,start_logratio=pairs[0][1],end_logratio=pairs[-1][1],
       crossing_brackets=[(float(pairs[j][0]),float(pairs[j+1][0])) for j in range(len(pairs)-1) if pairs[j][1]*pairs[j+1][1]<0],
       interpolated_zero_estimates=crossings))
 hists=max_endpoint_error
 if hists>1e-7:raise RuntimeError(f'Known anchor mismatch vs v18.48f: {hists}')
 report=dict(version='v18.48g',status='EXPLORATORY_GAUGE_MATCHED_BRANCH_HOMOTOPY_NOT_PHYSICAL_CONTINUATION_CERTIFICATE',
  families=families,original_xi=xis,anchor_cases=12,interpolation_alpha_points=args.alpha_points,
  seed_control_rows=len(seedrows),transport_rows=len(transport),actual_integrations=total_integrations,
  source_pins={key:value[1] for key,value in frozen.PINNED.items()},
  embedded_original_sha=json.loads((ROOT/'inputs'/'PARENT_SHA256.json').read_text())['files'],
  parent_v18_48f_sha=json.loads((ROOT/'inputs'/'V18_48F_SHA256.json').read_text())['files'],
  gauge_transform='(theta1,Theta)->(theta1+c,Theta+c), plus individual integer 2pi angular lifts',
  gauge_preserves_relative_phase=True,max_gauge_original_rhs_rel=max_gauge_rhs,
  max_gauge_analytic_hessian_rel=max_gauge_h,max_gauge_phase_mod_error=max_gauge_delta,
  maximum_known_anchor_ratio_rel_error=max_endpoint_error,
  branch_assignment=assignresults,interpolated_profiles=profiles,
  limitations=('The continuous gauge phase match IS an exact symmetry of the autonomous RHS; this does NOT mean original discrete xi anchors are on one continuous material branch. '
   'The intermediate seeds are introduced by linear interpolation in internal coordinates, plus a positive-radius control, NOT independently authenticated Q550 orbits. '
   'A crossing of the two interpolated cancellation ratios is not a certified branch exchange or dynamic transport event. '
   'The chart is inherited 10D; the adjoint is transported in full 11D for local short windows. '
   'The initial observer is interpolated, not reoptimized; no globally unique continuous branch or gauge invariant normal bundle is proved. '
   'The historical endpoints have T of order 1e-6. No long-time hysteresis or physical coupling-selection law is established.'))
 (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps({k:report[k] for k in ('status','seed_control_rows','transport_rows','actual_integrations','max_gauge_original_rhs_rel','max_gauge_analytic_hessian_rel','maximum_known_anchor_ratio_rel_error')},indent=2))

if __name__=='__main__':main()
