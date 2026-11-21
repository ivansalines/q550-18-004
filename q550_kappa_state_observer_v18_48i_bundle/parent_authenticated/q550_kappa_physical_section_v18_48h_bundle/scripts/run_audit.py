#!/usr/bin/env python3
"""Q550 v18.48h genuine IVP continuation on a common lifted-Theta clock section.

Historical xi values are true reference-flow times. The source branches at each
anchor xi were generated independently, so continuous IVP propagation of xi=3.5
seeds does NOT by itself prove connection to any independently sampled seed at
xi=4.0 or xi=4.5. We keep those questions separate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import run_audit_v18_48f as anc
from structural_jet import structural_jet,coupling_jet,chart_kernel
ROOT=Path(__file__).resolve().parents[1]
PI2=2*math.pi

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def norm(a):return float(np.linalg.norm(a))
def wrap(a):return (float(a)+math.pi)%(2*math.pi)-math.pi

def write_csv(p,rows):
 if not rows:raise ValueError(f'empty CSV {p}')
 with open(p,'w',newline='',encoding='utf8') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def verify_origins(base):
 report,sha=anc.verify_parent_inputs() # historical v18.48e snapshot exact content SHA
 a=json.loads((ROOT/'inputs'/'V18_48G_SENDBACK_SHA256.json').read_text())
 p=ROOT/'inputs'/a['filename']
 if not p.is_file() or digest(p)!=a['sha256']:raise RuntimeError('v18.48g ORIGINAL server sendback SHA drift')
 with zipfile.ZipFile(p) as z:
  if z.testzip():raise RuntimeError('bad original server sendback ZIP')
  manifest=[s for s in z.namelist() if s.endswith('/SHA256SUMS_OUTPUT.json')]
  if len(manifest)!=1:raise RuntimeError('v18.48g manifest path ambiguity')
  zprefix=manifest[0].rsplit('/',1)[0]+'/'
  m=json.loads(z.read(manifest[0]))
  if m.get('version')!='v18.48g' or not m.get('files'):raise RuntimeError('v18.48g bad manifest')
  for name,expected in m['files'].items():
   zname=zprefix+name
   if zname not in z.namelist() or hashlib.sha256(z.read(zname)).hexdigest()!=expected:
    raise RuntimeError('v18.48g member SHA mismatch '+name)
 audit,support,tube=anc.load_sources(base)
 return report,sha,a,audit,support,tube

def canonical(y,theta_target=0.):
 """Exact constant phase-gauge symmetry + independent 2pi angle lifts."""
 y=np.asarray(y,float).copy();c=theta_target-y[10]
 y[2]+=c;y[10]=theta_target
 for j in (2,6):y[j]-=PI2*np.rint(y[j]/PI2)
 return y

def relphase(y,params):return float(y[2]+y[6]-params.phi0-y[10])

def activity_gradient(y,params,scale,kind):
 """Gradient of rho1*rho2*sin(delta) or cos(delta) in frozen 11D chart.
 No interpolation/fit to independently sampled observer gradients.
 """
 a=math.exp(float(y[0]+y[4]));d=relphase(y,params)
 if kind=='sin':val=a*math.sin(d);der=a*math.cos(d)
 elif kind=='cos':val=a*math.cos(d);der=-a*math.sin(d)
 else:raise ValueError(kind)
 gy=np.zeros(11);gy[[0,4]]=val;gy[[2,6]]=der;gy[10]=-der
 g=gy*np.r_[scale,1.]
 mag=norm(g)
 if mag<1e-20:raise FloatingPointError('vanishing observer')
 return g/mag

def activity_lie_gradient(y,params,scale,kind,kappa,slope):
 """State-recomputed chart gradient of L_F[A], A=rho1*rho2*sin/cos(delta).

 Uses the exact structural Jacobian. The Lie gradient sees the acceleration
 channels that the instantaneous scalar gradient does not detect.
 """
 y=np.asarray(y,float);a=math.exp(float(y[0]+y[4]));d=relphase(y,params)
 W=np.zeros(11);W[[0,4]]=1.
 D=np.zeros(11);D[[2,6]]=1.;D[10]=-1.
 if kind=='sin': val=a*math.sin(d);der=a*math.cos(d)
 elif kind=='cos': val=a*math.cos(d);der=-a*math.sin(d)
 else:raise ValueError(kind)
 gy=W*val+D*der
 # Second derivative of A wrt ORIGINAL 11D internal coordinates.
 H=(np.outer(W,W)-np.outer(D,D))*val+(np.outer(W,D)+np.outer(D,W))*der
 f,J,_=structural_jet(y,kappa,params,slope)
 gchart=(H@f+J.T@gy)*np.r_[scale,1.]
 mag=norm(gchart)
 if mag<1e-20:raise FloatingPointError('vanishing Lie-gradient observer')
 return gchart/mag

def field_terms(y,g,kappa,params,slope,scale):
 f,j,h=structural_jet(y,kappa,params,slope)
 hb=structural_jet(y,0.,params,slope)[2]
 hc=slope*kappa*coupling_jet(y,params)[2]
 B=chart_kernel(hb,f,scale,g[:10]); C=chart_kernel(hc,f,scale,g[:10]);S=B+C
 nb,nc,ns=norm(B),norm(C),norm(S)
 ratio=(nb+nc)/max(ns,1e-30)
 oppose=float(B@C/(nb*nc)) if nb*nc>1e-30 else None
 return dict(ratio=ratio,base_norm=nb,coupling_norm=nc,resultant_norm=ns,
             opposition_cosine=oppose,scalar_difference=nb-nc), B,C,S

def matching_cost(y,yhist,scale,params):
 # Gauge invariant modulo-phase distance. Nonangular coordinates scaled by
 # frozen chart; individual angular coordinates mod 2pi, NOT Euclidean lifts.
 d=np.asarray(y,float)-np.asarray(yhist,float)
 for j in (2,6,10):d[j]=wrap(d[j])
 chart=np.r_[scale,1.]
 dx=d/chart
 return float(np.linalg.norm(dx)),abs(wrap(relphase(y,params)-relphase(yhist,params)))

def unit_scaled_distance(y,yhist):
 """Auxiliary size-normalized, phase-modulo state distance, not a material metric."""
 y=np.asarray(y,float);b=np.asarray(yhist,float)
 diff=y-b
 for j in (2,6,10):diff[j]=wrap(diff[j])
 den=np.maximum(1.,np.maximum(abs(y),abs(b)))
 den[[2,6,10]]=math.pi
 return norm(diff/den)

def extract_cases(z):
 cases={}
 for i,lbl in enumerate([str(x) for x in z['labels']]):
  case,fam,xi,branch,frac=anc.case_meta(lbl)
  if abs(frac-.125)<1e-9:cases[(fam,xi,branch)]=(np.asarray(z['states'][i],float),np.asarray(z['g'][i],float))
 if len(cases)!=12:raise RuntimeError(f'Expected 12 actual seeds; got {len(cases)}')
 return cases

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--base-root',type=Path,default=Path('/root/elab1/hysteresis'))
 ap.add_argument('--sections',type=int,default=21,help='Common clock sections over xi=3.5..4.5; >=5')
 ap.add_argument('--rtol',type=float,default=3e-12)
 ap.add_argument('--atol',type=float,default=3e-13)
 ap.add_argument('--max-step',type=float,default=.015)
 ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
 ap.add_argument('--match-threshold',type=float,default=.25,help='Frozen-chart distance; a diagnostic, not a proof criterion')
 args=ap.parse_args()
 if args.sections<5 or args.sections>201 or args.max_step<=0:raise ValueError('invalid resolution')
 out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 parent,pins,gpin,audit,support,tube=verify_origins(args.base_root.resolve())
 params=audit.ModelParameters(gamma_rho=.3)
 z=np.load(ROOT/'inputs'/'structural_snapshot_states.npz')
 scale=np.asarray(z['scale'],float);slope=float(z['slope']);kref=float(z['kappa_ref']);kgeo=float(z['kappa_geo']);cases=extract_cases(z)
 if abs(kref-1/6)>1e-14 or abs(kgeo-35*math.pi*10/(12*550))>1e-14:raise RuntimeError('Kappa provenance drift')
 if not np.all(scale>0) or scale.shape!=(10,):raise RuntimeError('invalid frozen original chart')
 ww=np.r_[scale,1.]
 # The exact support-time clock identity is dTheta/dxi = 2*pi*slope.
 clock_speed=PI2*slope
 families=sorted({x[0] for x in cases}); anchors=[3.5,4.,4.5]
 section_xi=np.linspace(anchors[0],anchors[-1],args.sections)
 if min(abs(section_xi-a).min() for a in anchors)>1e-12:
  raise RuntimeError('Grid must contain 3.5, 4.0, and 4.5; choose odd number of sections')
 # True original IVP forward continuation + cotangent along each of 8 flows.
 rows=[];states=[];endpoint=[];symmetry=[]
 integrations=0;max_clock=0.;max_rhs=0.;max_gauge_rhs=0.;max_instant_coupling=0.
 for fam in families:
  for branch in (0,1):
   seed,gh=cases[(fam,3.5,branch)]
   y0=canonical(seed,0.)
   for j in (0,1):
    if abs(wrap(relphase(y0,params)-relphase(seed,params)))>3e-11:raise RuntimeError('gauge phase invariant failure')
   for klabel,kap in (('original',kref),('autonomous',kgeo)):
    f_original=slope*np.asarray(support.support_rhs_float(y0,kappa0=kap,params=params,audit=audit),float)
    f_analytic=structural_jet(y0,kap,params,slope)[0]
    rhs_err=norm(f_original-f_analytic)/max(norm(f_original),1e-30)
    f_ungauged=slope*np.asarray(support.support_rhs_float(seed,kappa0=kap,params=params,audit=audit),float)
    gauge_err=norm(f_ungauged-f_original)/max(norm(f_ungauged),1e-30)
    max_gauge_rhs=max(max_gauge_rhs,gauge_err)
    if gauge_err>5e-10:raise RuntimeError('Original RHS gauge identity failed')
    max_rhs=max(max_rhs,rhs_err)
    if rhs_err>5e-11:raise RuntimeError('Original RHS and analytic RHS mismatch')
    # Start historical observer in chart, plus two intrinsic material-activity observers.
    g0=np.r_[gh,0.];g0=g0/max(norm(g0),1e-30)
    gsin=activity_gradient(y0,params,scale,'sin')
    gcos=activity_gradient(y0,params,scale,'cos')
    glsin=activity_lie_gradient(y0,params,scale,'sin',kap,slope)
    glcos=activity_lie_gradient(y0,params,scale,'cos',kap,slope)
    init=np.r_[y0,g0,gsin,gcos,glsin,glcos]
    def flow(t,w):
     y=w[:11]
     fy=slope*np.asarray(support.support_rhs_float(y,kappa0=kap,params=params,audit=audit),float)
     jy=slope*np.asarray(tube.float_support_jacobian(y,kappa0=kap,params=params,audit=audit,base=support),float)
     chartJ=(jy/ww[:,None])*ww[None,:]
     covs=w[11:].reshape(5,11)
     return np.r_[fy,(-chartJ.T@covs.T).T.ravel()]
    sol=solve_ivp(flow,(0.,1.),init,t_eval=section_xi-3.5,
                method='DOP853',rtol=args.rtol,atol=args.atol,max_step=args.max_step)
    integrations+=1
    if not sol.success or sol.y.shape!=(66,args.sections):raise RuntimeError('Physical IVP failure '+sol.message)
    for j,xi in enumerate(section_xi):
     y=np.asarray(sol.y[:11,j],float);cov=np.asarray(sol.y[11:,j],float).reshape(5,11)
     ec=abs(y[10]-clock_speed*(xi-3.5));max_clock=max(max_clock,ec)
     # Reconstruct material observers from each authentic propagated state,
     # independently of transported covectors and historical fitted observer.
     freshly=(activity_gradient(y,params,scale,'sin'),activity_gradient(y,params,scale,'cos'),
             activity_lie_gradient(y,params,scale,'sin',kap,slope),activity_lie_gradient(y,params,scale,'cos',kap,slope))
     for mode,g in (('historical_transport',cov[0]),('activity_sin_transport',cov[1]),('activity_cos_transport',cov[2]),
                    ('activity_sin_recomputed',freshly[0]),('activity_cos_recomputed',freshly[1]),
                    ('activity_sin_lie_transport',cov[3]),('activity_cos_lie_transport',cov[4]),
                    ('activity_sin_lie_recomputed',freshly[2]),('activity_cos_lie_recomputed',freshly[3])):
      data,B,C,S=field_terms(y,g,kap,params,slope,scale)
      if mode in ('activity_sin_recomputed','activity_cos_recomputed'):
       max_instant_coupling=max(max_instant_coupling,norm(C))
       if norm(C)>1e-12:raise RuntimeError('Expected activity-gradient structural coupling-null failed')
      r=dict(family=fam,seed_branch=branch,kappa_kind=klabel,xi=float(xi),
             physical_initial_value_continuation=True,common_section='LIFTED_THETA_CLOCK',
             theta_clock_error=ec,observer_mode=mode,phase_relative_wrapped=wrap(relphase(y,params)),
             historical_observer_seed_not_reoptimized=mode=='historical_transport',
             state_derived_observer=mode.endswith('recomputed'),observer_norm=norm(g),
             **data)
      rows.append(r)
     # Endpoint and trajectory arrays are actual original ODE integrations.
     states.append((fam,branch,klabel,float(xi),y.copy(),cov.copy()))
     if min(abs(xi-a) for a in (4.,4.5))<1e-10:
      anchor=float(round(xi*2)/2)
      distances=[]
      for dest_branch in (0,1):
       hist,_=cases[(fam,anchor,dest_branch)];hist=canonical(hist,clock_speed*(anchor-3.5))
       d,dp=matching_cost(y,hist,scale,params);du=unit_scaled_distance(y,hist);distances.append((dest_branch,d,dp,du))
       endpoint.append(dict(family=fam,kappa_kind=klabel,seed_branch=branch,anchor_xi=anchor,
         independent_target_branch=dest_branch,chart_distance=d,phase_relative_abs_error_rad=dp,
         unit_scaled_distance_not_a_metric=du,
         passes_declared_chart_threshold=d<args.match_threshold,threshold=args.match_threshold))
      choice=min(distances,key=lambda t:t[1])
      symmetry.append(dict(family=fam,kappa_kind=klabel,seed_branch=branch,anchor_xi=anchor,
        nearest_independently_seeded_branch=choice[0],minimum_chart_distance=choice[1],
        nearest_phase_difference_rad=choice[2],nearest_unit_scaled_distance=choice[3],
        passes_declared_threshold=choice[1]<args.match_threshold))
    print('v18.48h',fam,'seed_branch',branch,klabel,'physical xi 3.5->4.5','evaluations',sol.nfev,flush=True)
 if max_clock>1e-8:raise RuntimeError('global phase clock identity drift')
 if len(states)!=len(families)*2*2*args.sections:raise RuntimeError('incomplete IVP')
 write_csv(out/'physical_section_cancellation.csv',rows)
 write_csv(out/'original_anchor_matching.csv',endpoint)
 write_csv(out/'best_anchor_matches.csv',symmetry)
 np.savez_compressed(out/'physical_paths.npz',families=np.array([x[0] for x in states]),
   seed_branches=np.array([x[1] for x in states]),kappa_kind=np.array([x[2] for x in states]),
   xi=np.array([x[3] for x in states]),y=np.stack([x[4] for x in states]),
   covectors=np.stack([x[5] for x in states]),chart_scale=scale,
   clock_speed=clock_speed)
 # Paired original-seed paths at the same genuine clock section, with no relabeling.
 comparisons=[]
 for fam in families:
  for kl in ('original','autonomous'):
   for mode in ('historical_transport','activity_sin_transport','activity_cos_transport',
                'activity_sin_recomputed','activity_cos_recomputed','activity_sin_lie_transport','activity_cos_lie_transport',
                'activity_sin_lie_recomputed','activity_cos_lie_recomputed'):
    for xi in section_xi:
     a=next(r for r in rows if r['family']==fam and r['kappa_kind']==kl and r['observer_mode']==mode and r['seed_branch']==0 and abs(r['xi']-xi)<1e-11)
     b=next(r for r in rows if r['family']==fam and r['kappa_kind']==kl and r['observer_mode']==mode and r['seed_branch']==1 and abs(r['xi']-xi)<1e-11)
     comparisons.append(dict(family=fam,kappa_kind=kl,observer_mode=mode,xi=float(xi),
      ratio_branch0=a['ratio'],ratio_branch1=b['ratio'],log_ratio_branch0_over_branch1=math.log(a['ratio']/b['ratio'])))
 write_csv(out/'same_seed_identity_branch_balance.csv',comparisons)
 crossings=[]
 for fam in families:
  for kl in ('original','autonomous'):
   for mode in sorted({r['observer_mode'] for r in rows}):
    rr=[r for r in comparisons if r['family']==fam and r['kappa_kind']==kl and r['observer_mode']==mode]
    roots=[]
    for a,b in zip(rr[:-1],rr[1:]):
     ya=a['log_ratio_branch0_over_branch1'];yb=b['log_ratio_branch0_over_branch1']
     if ya*yb<0:
      xc=a['xi']+(b['xi']-a['xi'])*abs(ya)/(abs(ya)+abs(yb))
      roots.append(float(xc))
    crossings.append(dict(family=fam,kappa_kind=kl,observer_mode=mode,
      crossings_linear_interpolated=roots,sign_change_count=len(roots),
      minimum_log_balance=min(r['log_ratio_branch0_over_branch1'] for r in rr),
      maximum_log_balance=max(r['log_ratio_branch0_over_branch1'] for r in rr)))
 (out/'branch_balance_summary.json').write_text(json.dumps(crossings,indent=2,allow_nan=False)+'\n')
 bestdist=[q['minimum_chart_distance'] for q in symmetry]
 nearest_passed=sum(int(q['passes_declared_threshold']) for q in symmetry)
 report=dict(version='v18.48h',status='GENUINE_Q550_INITIAL_VALUE_SECTION_CONTINUATION_NOT_BRANCH_BVP_CERTIFICATE',
  source_v18_48e_SHA256=pins,parent_v18_48g_zip_sha256=gpin['sha256'],
  original_source_sha256={k:v[1] for k,v in anc.PINNED.items()},
  kappa_original=kref,kappa_autonomous=kgeo,delta_kappa=kgeo-kref,
  genuinely_integrated_IVPs=integrations,common_section_count=args.sections,
  total_genuine_trajectory_states=len(states),observation_rows=len(rows),
  preserved_seed_branch_identity=True,no_intermediate_state_interpolation=True,
  common_section='Theta(xi) - Theta(3.5) = 2pi*slope*(xi-3.5) after exact phase-gauge normalization',
  caveat_seed_phase_offset='Historical xi=3.5 state is recorded at 0.125*T, about 1e-7 after reference xi. End-section comparison is aligned to its own initial clock. No assertion of interval-certified event location.',
  clock_speed=clock_speed,max_clock_section_absolute_error=max_clock,
  max_original_rhs_vs_analytic_relative=max_rhs,
  max_original_rhs_phase_gauge_relative_error=max_gauge_rhs,
  max_instantaneous_activity_coupling_norm=max_instant_coupling,
  instantaneous_activity_coupling_kernel_is_exactly_zero=True,
  independent_target_comparisons=len(endpoint),nearest_target_matches_within_threshold=nearest_passed,
  independent_target_comparisons_per_seed=len(symmetry),
  minimum_best_endpoint_chart_distance=min(bestdist),median_best_endpoint_chart_distance=float(np.median(bestdist)),
  maximum_best_endpoint_chart_distance=max(bestdist),declared_chart_distance_threshold=args.match_threshold,
  median_best_endpoint_unit_scaled_distance=float(np.median([q['nearest_unit_scaled_distance'] for q in symmetry])),
  observer_definitions={
   'historical_transport':'Original fitted covector at xi3.5 only, transported by actual cotangent variational equation; not reoptimized at intermediate sections',
   'activity_sin_recomputed':'Fresh frozen-chart gradient of exp(y1+y2)*sin(delta) at each actual state',
   'activity_cos_recomputed':'Fresh frozen-chart gradient of exp(y1+y2)*cos(delta) at each actual state',
   'activity_sin_transport':'Transported initial activity_sin gradient with -J_chart.T',
   'activity_cos_transport':'Transported initial activity_cos gradient with -J_chart.T',
   'activity_sin_lie_recomputed':'New exact-Jacobian chart gradient of d(activity_sin)/dxi at each actual state',
   'activity_cos_lie_recomputed':'New exact-Jacobian chart gradient of d(activity_cos)/dxi at each actual state',
   'activity_sin_lie_transport':'Initial Lie derivative activity_sin gradient transported by -J_chart.T',
   'activity_cos_lie_transport':'Initial Lie derivative activity_cos gradient transported by -J_chart.T'},
  phase_alignment_original_model_symmetry='(theta1,Theta)->(theta1+c,Theta+c), independent angular 2pi lifts',
  branches_are_independently_seeded_at_each_reference_xi=True,
  crossings_are_sampled_sign_changes_only=True,
  reference_crossings=crossings,
  limitations='The continuation is a genuine Q550 IVP through common lifted clock sections, NOT a physical branch-connected BVP, parameter-continuation proof, invariant observer, or interval certificate. Original independently seeded branches at xi4 and 4.5 may not lie on propagated seed paths. Activity observables are model-defined, not the historical fitted source gradient. Binary64 solver tolerances are numerical diagnostics, not certified enclosures.')
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report[k] for k in ('status','genuinely_integrated_IVPs','common_section_count','observation_rows','max_clock_section_absolute_error','median_best_endpoint_chart_distance','nearest_target_matches_within_threshold')},indent=2))

if __name__=='__main__':main()
