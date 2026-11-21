#!/usr/bin/env python3
"""v18.48l: gauge-aware source-active near-null geometry and observer family.

Numerical rank is tolerance dependent. High-precision focal checks lift the
ORIGINAL double-precision Q550 snapshots, not an exact-real Q550 trajectory.
No new original Q550 integration is claimed.
"""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from pathlib import Path
import numpy as np
from verify_parent import verify
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'scripts')]
from observer_geometry_v18_48k import matrices,ratio_from_maps,unit
from subspace_math import source_active_mode,full_null,subspace_angle
from high_precision import high_precision_matrices,hp_singular_values
from structural_jet import structural_jet,coupling_jet
from test_support_mobius_monodromy_audit import ModelParameters
import mpmath as mp

def write_csv(path,rows):
    if not rows:raise ValueError('empty ledger '+str(path))
    keys=list(rows[0]);assert all(list(x)==keys for x in rows)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def focal(fam,xi,branch):
    if branch!=0:return False
    return ('phaseblind' in fam and abs(xi-3.575)<1e-8) or ('phasecov' in fam and abs(xi-4.25)<1e-8)

def high_precision_report(Lb,Lc,referenceLb,referenceLc,g_observer,dps):
    """Compare independent high-precision model to float maps and singular directions."""
    with mp.workdps(dps):
        T=Lb+Lc; A=mp.matrix(20,10)
        for i in range(10):
            for j in range(10):A[i,j]=Lb[i,j];A[i+10,j]=Lc[i,j]
        u,s,v=mp.svd_r(T)
        sa=hp_singular_values(A)
        smallest=mp.mpf(s[len(s)-1]);largest=mp.mpf(s[0]);source_min=mp.mpf(sa[-1]);source_max=mp.mpf(sa[0]);
        null=mp.matrix([v[9,j] for j in range(10)])
        nb=mp.norm(Lb*null);nc=mp.norm(Lc*null);nt=mp.norm(T*null)
        # Source is demonstrably non-vanishing in the high-precision T-near-null direction.
        align=sum(null[i]*mp.mpf(float(g_observer[i])) for i in range(10))
        valmap=np.asarray(Lb.tolist(),dtype=float);valmapc=np.asarray(Lc.tolist(),dtype=float)
        errb=float(np.linalg.norm(valmap-referenceLb)/max(1.,np.linalg.norm(referenceLb)))
        errc=float(np.linalg.norm(valmapc-referenceLc)/max(1.,np.linalg.norm(referenceLc)))
        return dict(precision_dps=dps,sigma_resultant_max=mp.nstr(largest,25),
          sigma_resultant_second_last=mp.nstr(s[len(s)-2],25),sigma_resultant_last=mp.nstr(smallest,25),
          sigma_resultant_last_relative=mp.nstr(smallest/largest,20),
          sigma_source_last=mp.nstr(source_min,25),sigma_source_last_relative=mp.nstr(source_min/source_max,20),
          null_source_norm_B=mp.nstr(nb,25),null_source_norm_C=mp.nstr(nc,25),
          null_resultant_norm=mp.nstr(nt,25),
          null_cancel_ratio=mp.nstr((nb+nc)/nt,18) if nt else 'infinity',
          observer_null_alignment_abs=mp.nstr(abs(align),18),
          relative_f64_map_error_B=errb,relative_f64_map_error_C=errc)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',default=str(ROOT/'results'),type=Path)
    p.add_argument('--hp-dps',default='45,70,90')
    p.add_argument('--cutoffs',default='1e-4,1e-6,1e-8,1e-10,1e-12')
    args=p.parse_args()
    dps=[int(a) for a in args.hp_dps.split(',')]
    cuts=[float(a) for a in args.cutoffs.split(',')]
    if sorted(dps)!=dps or len(dps)<2 or min(dps)<35:raise ValueError('invalid precision schedule')
    if sorted(cuts,reverse=True)!=cuts or min(cuts)<1e-13 or max(cuts)>1e-2:raise ValueError('invalid cutoff schedule')
    out=args.output_dir.resolve();out.mkdir(exist_ok=True,parents=True)
    provenance=verify()
    pdata=np.load(ROOT/'runtime/parent_k/runtime/parent_j/runtime/physical_paths.npz',allow_pickle=False)
    hist=json.loads((ROOT/'runtime/parent_k/runtime/parent_j/runtime/report_v18_48h.json').read_text())
    sc=np.asarray(pdata['chart_scale'],float);speed=float(pdata['clock_speed'])/(2*math.pi)
    kap={'original':float(hist['kappa_original']),'autonomous':float(hist['kappa_autonomous'])}
    par=ModelParameters(gamma_rho=.3)
    if len(pdata['xi'])!=328:raise ValueError('unexpected Q550 physical states count')
    led=[];active=[];hp=[];families={};fp=[];focal_count=0
    for i in range(len(pdata['xi'])):
        fam=str(pdata['families'][i]);branch=int(pdata['seed_branches'][i]);kind=str(pdata['kappa_kind'][i]);xi=float(pdata['xi'][i])
        group=(fam,branch,kind);y=np.asarray(pdata['y'][i],float);k=kap[kind]
        f=structural_jet(y,k,par,speed)[0]
        HB=structural_jet(y,0,par,speed)[2]
        HC=speed*k*coupling_jet(y,par)[2]
        Lb,Lc=matrices(HB,HC,f,sc)
        # One right-null orientation and one source-active family at conservative cutoff.
        null=full_null(Lb,Lc)
        histg=unit(np.asarray(pdata['covectors'][i,0,:10],float));hr,hb,hc,ht,_=ratio_from_maps(Lb,Lc,histg)
        g=null['g'];
        row=dict(family=fam,branch=branch,kappa=kind,xi=xi,original_observer_R=hr,
          sigma_T_1=null['sigma_max'],sigma_T_9=null['sigma_next'],sigma_T_10=null['sigma_min'],
          sigma_A_1=null['source_sigma_max'],sigma_A_10=null['source_sigma_min'],
          sigma_T_10_over_T_1=null['sigma_min']/null['sigma_max'],
          source_response_null_f64=null['source_norm'],
          resultant_response_null_f64=null['resultant_norm'],
          R_null_f64_untrusted=null['cancellation_ratio'],
          historical_null_abs_cos=abs(float(g@histg)))
        led.append(row)
        famrow=(i,xi,Lb,Lc,histg,g)
        families.setdefault(group,[]).append(famrow)
        for cutoff in cuts:
            a=source_active_mode(Lb,Lc,cutoff)
            active.append(dict(family=fam,branch=branch,kappa=kind,xi=xi,cutoff=cutoff,
                source_rank=a['rank'],generalized_sigma_min=a['gen_min'],
                active_ratio=a['cancel_ratio'],source_activity_relative=a['activity_relative'],
                source_norm=a['source_norm'],resultant_norm=a['resultant_norm'],
                numerical_rank_not_exact=True))
        if focal(fam,xi,branch):
            focal_count+=1
            print('FOCAL',fam[:10],kind,xi,'R',round(hr,6),flush=True)
            for precision in dps:
                with mp.workdps(precision):
                    mpb,mpc=high_precision_matrices(y,mp.mpf(k),par,mp.mpf(speed),sc,precision)
                    checks=high_precision_report(mpb,mpc,Lb,Lc,histg,precision)
                hp.append(dict(crest=('phaseblind_xi3.575' if 'phaseblind' in fam else 'phasecov_xi4.25')+'_'+kind,
                               historical_ratio=hr,**checks))
    if focal_count!=4:raise ValueError('Unexpected focal crest count: '+str(focal_count))
    tracks=[];min_adj=180.;max_adj=0.;source_min=float('inf');active_by={(r['family'],r['branch'],r['kappa'],r['xi'],r['cutoff']):r for r in active}
    # Gauge-consistent sign selection for line-valued subspaces, no claim of physical adjoint continuation.
    for group,rows in sorted(families.items()):
        rows.sort(key=lambda x:x[1]);prev=None;prev_active=None
        for idx,xi,B,C,gh,g0 in rows:
            signed=g0.copy()
            if prev is not None and signed@prev<0:signed=-signed
            ang=None if prev is None else subspace_angle(prev,signed)
            prev=signed.copy()
            for cutoff in (1e-6,1e-10):
                a=source_active_mode(B,C,cutoff)
                act=a['g'];key=(group,cutoff)
                if key in active_by:pass
                # Per cutoff own gauge for each physical family.
                if prev_active is None: prev_active={}
                former=prev_active.get(cutoff)
                if former is not None and act@former<0:act=-act
                act_ang=None if former is None else subspace_angle(former,act)
                prev_active[cutoff]=act
                tracks.append(dict(family=group[0],branch=group[1],kappa=group[2],xi=xi,
                    cutoff=cutoff,source_rank=a['rank'],full_null_intersection_angle_deg=ang,
                    source_active_mode_step_angle_deg=act_ang,
                    full_null_observer_abs_cos=abs(float(gh@signed)),
                    source_active_observer_abs_cos=abs(float(gh@act)),
                    active_R=a['cancel_ratio'],active_source_activity_relative=a['activity_relative']))
            if ang is not None:max_adj=max(max_adj,ang);min_adj=min(min_adj,ang)
    write_csv(out/'full_null_singular_atlas.csv',led)
    write_csv(out/'source_active_cutoff_sweep.csv',active)
    write_csv(out/'source_active_continuation.csv',tracks)
    write_csv(out/'high_precision_focal_checks.csv',hp)
    # Key scientific boundary: last singular in T follows the precision floor; do not certify exact rank.
    summary=[]
    for crest in sorted({x['crest'] for x in hp}):
        c=[x for x in hp if x['crest']==crest];c.sort(key=lambda v:v['precision_dps'])
        summary.append(dict(crest=crest,historical_ratio=c[0]['historical_ratio'],
          last_singular_by_precision={str(z['precision_dps']):z['sigma_resultant_last'] for z in c},
          source_smallest_by_precision={str(z['precision_dps']):z['sigma_source_last'] for z in c},
          largest_f64_map_error=max(max(z['relative_f64_map_error_B'],z['relative_f64_map_error_C']) for z in c)))
    report=dict(version='v18.48l',classification='AUTHENTICATED_FIXED_STATE_NEAR_NULL_OBSERVER_GEOMETRY',
        provenance=provenance,physical_trajectories=8,physical_states=328,new_original_integrations=0,
        focal_crests=4,high_precision_digits=dps,active_rank_cutoffs=cuts,
        maximum_full_null_step_angle_deg=max_adj,
        parameter_values=kap,focal=summary,
        warnings=['SVD minima at precision floors do not by themselves prove an exact algebraic null.',
         'The source-active generalized subspace is cutoff-dependent, not a unique canonical observer.',
         'Adjacency compares instantaneous subspaces along authentic prior Q550 trajectories: no observer adjoint transport is integrated here.',
         'All Q550 state inputs are floating point; arbitrary precision does not recover an exact trajectory.',
         'Full-null cancellation ratio from float64 can be severely inaccurate; prefer high-precision focal checks.',
         'No physical hysteresis or intrinsic material torsion is inferred from observer-space singular geometry.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(classification=report['classification'],states=328,high_precision_checks=len(hp),
         maximum_adjacent_full_null_angle_deg=max_adj,focal=summary),indent=2))
if __name__=='__main__':main()
