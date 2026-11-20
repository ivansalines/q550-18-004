#!/usr/bin/env python3
"""Pinned original Q550: Jacobian-integral compensated pairs vs second variation.

For the line-integral differences only the independent original Jacobian is used;
the second-variation predictor differentiates that Jacobian at fixed displacements.
Independent conventional pair results from v18.28 are used as a third witness.
All perturbed directions are OFF_CONSTRAINT_DIAGNOSTIC.
"""
from __future__ import annotations
import argparse,csv,json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_29 import read as parent_data
from frozen_preflight import observers
from blind_experiment import initial_directions
from blind_math import HISTORY,HORIZONS
from checkpoint_io_v18_21 import PREFIX as PREFIX20
from compensated_math_v18_29 import (compensated_rhs,jacobian_variational_rhs,
    direct_second_from_jac,even_odd_from_deltas,rms,line_increment)
import q550_state_impulse_response_v13 as original

CASES=(('raw6',4.5),('material3',3.5))
EPS=(1e-5,3e-6,1e-6)
TIERS={'standard':(5e-11,5e-12,.035),'tight':(3e-13,3e-14,.0125)}
JSTEPS=(.01,.04)

def integrate(fun,a,b,initial,grid,tier):
    rt,at,mx=TIERS[tier]
    sol=solve_ivp(fun,(float(a),float(b)),np.asarray(initial,float),t_eval=np.asarray(grid,float),
        method='DOP853',rtol=rt,atol=at,max_step=mx)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f'Failed original Q550 {tier} integration: {sol.message}')
    return sol.y.T

def two_sided(fun,a,y,tier):
    back=a+HISTORY[::-1];front=a+np.r_[0.,HORIZONS]
    l=integrate(fun,a,back[-1],y,back,tier)
    r=integrate(fun,a,front[-1],y,front,tier)
    return np.r_[back[::-1],front[1:]],np.vstack((l[::-1],r[1:]))

def build_oracle(base_path):
    data, provenance, proof, old_csv=parent_data()
    d=data['tight']; original.DEFAULT_BASE=base_path
    for role,(file,hash_expected) in original.DEPENDENCIES.items():
        file_path=base_path/file
        if not file_path.is_file() or original.hash_file(file_path)!=hash_expected:
            raise RuntimeError('Pinned original source mismatch '+str(file_path))
    audit=original.load_module(base_path/original.DEPENDENCIES['audit'][0],'v1829_audit')
    support=original.load_module(base_path/original.DEPENDENCIES['support'][0],'v1829_support')
    tube=original.load_module(base_path/original.DEPENDENCIES['tube'][0],'v1829_tube')
    params=audit.ModelParameters(gamma_rho=.3)
    k0=(55./6.)*(10./550.);slope=float(d['slope'])
    def f(t,y):
        return slope*np.asarray(support.support_rhs_float(
                np.asarray(y,float),kappa0=k0,params=params,audit=audit),float)
    def jac(y):
        return slope*np.asarray(tube.float_support_jacobian(
                np.asarray(y,float),kappa0=k0,params=params,audit=audit,base=support),float)
    with zipfile.ZipFile(ROOT/'.runtime/v18_20_original_server_sendback.zip') as z:
        seeds=json.loads(z.read(PREFIX20+'results/matched_hidden_seeds.json'))
    direction=initial_directions({r['name']:r['seed'] for r in seeds})
    H=observers(d)
    oldrows=list(csv.DictReader(old_csv.splitlines()))
    old={(r['observer'],float(r['anchor']),float(r['epsilon'])):r for r in oldrows}
    return d,provenance,proof,H,direction,f,jac,old

def seed_at_anchor(d,v,a):
    if a==4.:return np.asarray(v,float).copy()
    P=CubicSpline(np.asarray(d['xi_nodes']),np.asarray(d['P_full11D'][:,:10,:10]),axis=0)
    vv=P(a)@np.linalg.solve(P(4.),np.asarray(v,float))
    return vv/np.linalg.norm(vv)

def write_csv(path,rows):
    with path.open('w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    args=ap.parse_args()
    d,provenance,parent,Hs,seeds,f,J,old=build_oracle(args.base_root)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[];profiles=[];source_rows=[];diagnostics=[];integrations=0
    centers={}
    for a in sorted(set(a for _,a in CASES)):
        centers[a]=integrate(f,0.,a,np.asarray(d['source_state_11D'],float),[a],'tight')[-1]
        integrations+=1
    for name,a in CASES:
        H=np.asarray(Hs[name],float);v=seed_at_anchor(d,seeds[name]['matched_seed'],a)
        y0=centers[a]
        z0=np.r_[y0,v,np.zeros(10)]
        tang=np.r_[v,0.]
        jac0=J(y0)
        if jac0.shape!=(11,11) or not np.all(np.isfinite(jac0)):
            raise RuntimeError('Pinned original Jacobian has invalid shape/data')
        jac_checks=[]
        for step in [3e-4,1e-3]:
            central=(f(0.,y0+step*tang)-f(0.,y0-step*tang))/(2*step)
            ja=jac0@tang
            error=float(np.linalg.norm(central[:10]-ja[:10])/max(1e-12,np.linalg.norm(ja[:10])))
            jac_checks.append(error)
        if min(jac_checks)>2e-3:
            raise RuntimeError(f'Original pinned Jacobian fails directional RHS audit {name}: {jac_checks}')
        var={}
        for h in JSTEPS:
            fun=jacobian_variational_rhs(f,J,h)
            grid,vec=two_sided(fun,a,z0,'tight');integrations+=2
            var[h]=vec
            for t,z in zip(grid,vec):
                y=z[:11];u=np.r_[z[11:21],0.]
                src=direct_second_from_jac(J,y,u,h)
                source_rows.append(dict(observer=name,anchor=a,xi=float(t),jacobian_h=h,
                    observed_source_rms=rms(H@src[:10]),full10_source_norm=float(np.linalg.norm(src[:10])),
                    tangent_norm=float(np.linalg.norm(u[:10]))))
        ref_even={e:.5*e*(var[JSTEPS[0]][:,21:31]@H.T) for e in EPS}
        h_variation=rms((var[JSTEPS[0]][:,21:31]-var[JSTEPS[1]][:,21:31])@H.T)
        diagnostics.append(dict(observer=name,anchor=a,
           directional_J_residual_3e4=jac_checks[0],directional_J_residual_1e3=jac_checks[1],
           second_variation_source_gap=rms(np.array([r['observed_source_rms'] for r in source_rows if r['observer']==name and r['jacobian_h']==JSTEPS[0]])-
                             np.array([r['observed_source_rms'] for r in source_rows if r['observer']==name and r['jacobian_h']==JSTEPS[1]])),
           second_variation_propagated_w_disagreement=h_variation))
        for eps in EPS:
            state=np.r_[y0,eps*v,-eps*v]
            cases={}
            for tier in TIERS:
                gridc,sol=two_sided(compensated_rhs(f,J,eps,order=3),a,state,tier)
                integrations+=2
                if not np.array_equal(grid,gridc):raise RuntimeError('Compensated and variational grid mismatch')
                even,odd=even_odd_from_deltas(sol,eps,H)
                cases[tier]=(even,odd,sol)
                pred=ref_even[eps]
                for i,t in enumerate(grid):
                    profiles.append(dict(observer=name,anchor=a,epsilon=eps,tier=tier,xi=float(t),delta_xi=float(t-a),
                     compensated_even_norm=float(np.linalg.norm(even[i])),second_variational_even_norm=float(np.linalg.norm(pred[i])),
                     even_residual_norm=float(np.linalg.norm(even[i]-pred[i])),
                     compensated_odd_norm=float(np.linalg.norm(odd[i])),
                     first_variation_odd_norm=float(np.linalg.norm(var[JSTEPS[0]][i,11:21]@H.T))))
            tight_even,tight_odd,_=cases['tight'];coarse_even,coarse_odd,_=cases['standard']
            pred=ref_even[eps];pred_odd=var[JSTEPS[0]][:,11:21]@H.T
            oldrow=old.get((name,a,float(eps)))
            rec=dict(observer=name,anchor=a,epsilon=eps,
                compensation_even_rms=rms(tight_even),second_variational_even_rms=rms(pred),
                compensated_vs_second_variational_rms=rms(tight_even-pred),
                compensated_vs_second_variational_relative=rms(tight_even-pred)/max(rms(pred),1e-30),
                uncompensated_v18_28_even_rms=float(oldrow['paired_even_normalized_rms']) if oldrow else float('nan'),
                uncompensated_v18_28_vs_variation_rms=float(oldrow['paired_even_minus_second_variational_rms']) if oldrow else float('nan'),
                compensated_tier_difference_rms=rms(tight_even-coarse_even),
                odd_tight_vs_tangent_rms=rms(tight_odd-pred_odd),
                odd_tier_difference_rms=rms(tight_odd-coarse_odd),
                second_variation_h_difference_rms=.5*eps*h_variation,
                common_clock_phase_increment=0.,
                valid_off_constraint_diagnostic=True)
            rows.append(rec)
            print('COMP',name,'xi',a,'eps',eps,'new-even',f'{rec["compensation_even_rms"]:.6e}',
               'predict',f'{rec["second_variational_even_rms"]:.6e}',
               'resid',f'{rec["compensated_vs_second_variational_rms"]:.6e}',
               'tier',f'{rec["compensated_tier_difference_rms"]:.6e}',flush=True)
    write_csv(out/'original_compensated_pairs_v18_29.csv',rows)
    write_csv(out/'original_compensated_profiles_v18_29.csv',profiles)
    write_csv(out/'original_jacobian_source_v18_29.csv',source_rows)
    relative=[r['compensated_vs_second_variational_relative'] for r in rows]
    proof=dict(version='v18.29',status='ORIGINAL_Q550_COMPENSATED_PAIR_SECOND_VARIATIONAL_CONVERGENCE_COMPLETED',
     original_source_SHA256_verified=True,parent_sha256=provenance['v18_28_parent_sha256'],
     verified_parent_members=provenance['v18_28_verified_members'],
     original_integrations=integrations,case_count=len(CASES),epsilon_values=list(EPS),
     pair_comparisons=len(rows),solver_tiers={k:dict(rtol=v[0],atol=v[1],max_step=v[2]) for k,v in TIERS.items()},
     jacobian_line_quadrature_order=3,independent_variational_jacobian_steps=list(JSTEPS),
     source_calculation='PINNED_ORIGINAL_ANALYTIC_JACOBIAN_DIRECTIONAL_DERIVATIVE',
     pair_calculation='INTEGRATED_JACOBIAN_LINE_INCREMENTS_FROM_ORIGINAL_Q550',
     conventional_independent_pair_archive='ORIGINAL_Q550_V18_28_SERVER',
     comparisons=rows,anchor_jacobian_checks=diagnostics,
     status_notes='Empirical convergence only. Neither analytic derivative of J nor interval enclosure certified.',
     constraints='OFF_CONSTRAINT_DIAGNOSTIC_NO_PRIVILEGED_MATERIAL_METRIC',
     legacy_pair_response_not_used_in_compensated_integration=True)
    (out/'original_compensated_proof_v18_29.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('ORIGINAL_V18_29_COMPLETED',integrations,'local integrations',len(rows),'comparisons')
if __name__=='__main__':main()
