#!/usr/bin/env python3
"""Non-replay v18.9 measured audit from frozen Q550 kick203 v18.7 checkpoints.

No true constraint/energy function can be established from Jacobians alone.
The algebraic checks below are ODE-tangent diagnostics and are labeled as such.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
from structural_math_v18_9 import (VERSION,roots_bracketed,cosines,transport_invariant,
    euler_strain_birth,lyapunov_null_screen,affine_invariant_screen,unit_change_check)
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'inputs/v18_7_original_sendback.zip'
OUT=ROOT/'results'


def digest(blob):return hashlib.sha256(blob).hexdigest()


def get_data():
    if not ARCHIVE.exists():raise FileNotFoundError(ARCHIVE)
    result={}
    with zipfile.ZipFile(ARCHIVE) as z:
        broken=z.testzip()
        if broken:raise RuntimeError(f'corrupt historical member: {broken}')
        manifest=json.loads(z.read('SHA256SUMS_OUTPUT.json'))['files']
        for name,sha in manifest.items():
            if digest(z.read(name))!=sha:raise RuntimeError('historical provenance hash mismatch: '+name)
        with zipfile.ZipFile(io.BytesIO(z.read('inputs/v18_6_original_sendback.zip'))) as prev:
            source_manifest=json.loads(prev.read('SHA256SUMS_OUTPUT.json'))['files']
            for name,sha in source_manifest.items():
                if digest(prev.read(name))!=sha:raise RuntimeError('v18.6 provenance hash mismatch: '+name)
            for tol in ('standard','tight'):
                with np.load(io.BytesIO(z.read(f'checkpoint/{tol}_sign_crossing.npz')),allow_pickle=False) as d:
                    data={k:d[k].copy() for k in d.files}
                with np.load(io.BytesIO(prev.read(f'checkpoint/{tol}_source_transport.npz')),allow_pickle=False) as d:
                    data.update({'slope':float(d['source_scale'][0]),'s0':float(d['source_s'][0]),
                                 'source_state_11D':d['source_state_11D'].copy(),
                                 'kick_direction_11D':d['kick_direction_11D'].copy()})
                result[tol]=data
    return result,{'original_v18_7_sha256':digest(ARCHIVE.read_bytes()),
                   'v18_7_verified_member_count':len(manifest),
                   'v18_6_verified_member_count':len(source_manifest)}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def audit(data,tol):
    x=data['xi_nodes']; P=data['P_full11D']; J=data['J_full11D'];
    E=data['old_source_material_11D']; K=data['quadratic_total_material_11D'];
    A=data['integral_source_frame_11D']; g=data['source_integrand_11D']
    slope=data['slope']; early=A[80]
    if x.shape!=(641,) or J.shape!=(641,11,11) or P.shape!=(641,11,11):raise ValueError('wrong Q550 checkpoint grid')
    if np.max(abs(np.diff(x)-.0125))>1e-10:raise ValueError('wrong xi sampling')
    if np.max(abs(P[0]-np.eye(11)))>1e-7:raise RuntimeError('initial P is not identity')
    if not np.isfinite(P).all() or not np.isfinite(J).all():raise RuntimeError('bad stored cocycle')
    rE=np.max(np.linalg.norm(E-np.einsum('nij,j->ni',P,early),axis=1))
    rK=np.max(np.linalg.norm(K-np.einsum('nij,nj->ni',P,A),axis=1))
    if rE>2e-9 or rK>2e-9:raise RuntimeError('11D material-source reconstruction failed')
    lab=np.einsum('ni,ni->n',E,K)
    strain,birth=euler_strain_birth(x,E,K,J,P,g,slope)
    labd=strain+birth
    src=np.einsum('j,nj->n',early,A)
    srcd=slope*np.einsum('j,nj->n',early,g)
    rooted_lab=roots_bracketed(x,lab,labd)
    rooted_source=roots_bracketed(x,src,srcd)
    if len(rooted_lab)!=3 or max(abs(a-b) for a,b in zip(rooted_lab,data['xi_roots']))>1e-7:
        raise RuntimeError('v18.7 primary root regression failed')
    src_from_metric=transport_invariant(P,early,A)
    if np.max(abs(src-src_from_metric))>3e-9*max(1,np.max(abs(src))):
        raise RuntimeError('inverse-cocycle geometric identity failed')
    mask=x>=1.
    c_lab=cosines(E[mask],K[mask]);c_src=cosines(np.broadcast_to(early,A[mask].shape),A[mask])
    # Non-collinearity test: if Gram determinant strictly >0 for a given
    # pair, its inner-product sign is not invariant over all SPD choices.
    sine_lab=np.sqrt(np.maximum(0.,1.-c_lab*c_lab))
    sine_src=np.sqrt(np.maximum(0.,1.-c_src*c_src))
    # Necessity screen: continuous time-invariant symmetric quadratic G would
    # satisfy J^T G+G J=0 along linearized homogeneous dynamics.
    gram=lyapunov_null_screen(J,sample_stride=8)
    linear=affine_invariant_screen(J,sample_stride=8)
    Pxi_fd=np.gradient(P,x,axis=0,edge_order=2)
    Pxi_exact=slope*np.einsum('nij,njk->nik',J,P)
    differ=np.linalg.norm((Pxi_fd-Pxi_exact)[1:-1].reshape(-1,121),axis=1)
    scale=np.linalg.norm(Pxi_exact[1:-1].reshape(-1,121),axis=1)
    # coordinate change: new coords z=T y, tensor metric Wz=T^-T Wy T^-1
    units=np.exp(np.linspace(-1.2,1.2,11))
    covariant_error=unit_change_check(P,E[mask],K[mask],units,np.ones(11))
    # Demonstrate a strict coordinate-unit dependence of the naive untransformed I:
    Eunit=E*units;Kunit=K*units
    labunit=np.einsum('ni,ni->n',Eunit,Kunit)
    # Fixed diagonal weights => differentiated exact J and forcing terms.
    ep=slope*np.einsum('nij,nj->ni',J,E)
    kp=slope*(np.einsum('nij,nj->ni',J,K)+np.einsum('nij,nj->ni',P,g))
    unit_d=np.einsum('ni,i,ni->n',ep,units*units,K)+np.einsum('ni,i,ni->n',E,units*units,kp)
    roots_unit=roots_bracketed(x,labunit,unit_d)
    denom=np.maximum(np.linalg.norm(E,axis=1)*np.linalg.norm(K,axis=1),1e-300)
    rows=[]
    for i in np.where(mask)[0]:
        rows.append({'tolerance':tol,'xi':float(x[i]),
            'dot_euclidean':float(lab[i]),'dot_rate_strain':float(strain[i]),
            'dot_rate_new_source':float(birth[i]),
            'cos_euclidean':float(lab[i]/denom[i]),
            'dot_initial_source_frame':float(src[i]),
            'dot_rate_initial_source_frame':float(srcd[i]),
            'cos_initial_source_frame':float(c_src[i-80]),
            'sin_angle_euclidean':float(sine_lab[i-80]),
            'sin_angle_initial_source_frame':float(sine_src[i-80]),
            'dot_after_nondiagonal_units_with_implicit_I':float(labunit[i])})
    return {'tolerance':tol,'state_dimension':11,'xi_range':[float(x[0]),float(x[-1])],
        'xi_step':float(x[1]-x[0]),'ode_slope_ds_dxi':slope,
        'euclidean_bracketed_roots_xi':rooted_lab,'source_frame_bracketed_roots_xi':rooted_source,
        'fixed_nontrivial_unit_rescaling_bracketed_roots_xi':roots_unit,
        'min_source_frame_dot_xi1_to8':float(np.min(src[mask])),
        'min_source_frame_cosine_xi1_to8':float(np.min(c_src)),
        'min_sine_euclidean_xi1_to8':float(np.min(sine_lab)),
        'min_sine_source_frame_xi1_to8':float(np.min(sine_src)),
        'min_sine_near_euclidean_roots':[
          float(np.interp(r,x[mask],sine_lab)) for r in rooted_lab],
        'max_transport_identity_residual':float(np.max(abs(src-src_from_metric))),
        'max_source_early_reconstruction_error':float(rE),
        'max_source_total_reconstruction_error':float(rK),
        'linearized_cocycle_fd_residual_relative_median':float(np.median(differ/np.maximum(scale,1e-300))),
        'linearized_cocycle_fd_residual_relative_max':float(np.max(differ/np.maximum(scale,1e-300))),
        'unit_change_tensor_covariance_error':covariant_error,
        'jacobian_rows_identically_zero_on_archived_nodes':[int(k) for k in np.where(np.max(np.abs(J),axis=(0,2))<1e-12)[0]],
        'jacobian_columns_identically_zero_on_archived_nodes':[int(k) for k in np.where(np.max(np.abs(J),axis=(0,1))<1e-12)[0]],
        'constant_quadratic_tangent_metric_screen':gram,
        'constant_affine_constraint_necessary_screen':linear,
        'status':'DYNAMICAL_TANGENT_DIAGNOSTIC_NO_INTRINSIC_METRIC_OR_CONSTRAINT_CERTIFICATE',
        'notes':['Source-frame metric is pulled back from an INITIAL Euclidean choice, not mechanical energy.',
                 'The sign/number of roots is NOT invariant over all SPD state metrics.',
                 'Finite-difference cocycle diagnostics are not interval validation.',
                 'J^TG+GJ screen is necessary only for a constant tangent quadratic form.',
                 'The archived off-constraint second-order K2 may require a normal curvature component if genuine constraints exist.']},rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=OUT)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sources,integrity=get_data()
    tolout=[];rows=[]
    for tol,d in sources.items():
        q,r=audit(d,tol);tolout.append(q);rows.extend(r)
    maximum_root_discrepancy=max(abs(a-b) for a,b in zip(tolout[0]['euclidean_bracketed_roots_xi'],tolout[1]['euclidean_bracketed_roots_xi']))
    summary={'version':VERSION,'input_provenance':integrity,'by_tolerance':tolout,
     'standard_tight_max_root_disagreement_xi':maximum_root_discrepancy,
     'physical_metric_status':'NOT_IDENTIFIED_FROM_ODE_TANGENT_CHECKPOINTS',
     'physical_constraint_status':'NOT_IDENTIFIED_FROM_ODE_TANGENT_CHECKPOINTS',
     'next_gate':'Replay actual f(y) and independently documented C(y), DC(y), D2C(y), G(y) with physical source provenance; audit must not substitute an arbitrary SPD choice.'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_csv(a.output/'observable_geometry.csv',rows)
    lines=['# Q550 v18.9 — ODE structure, metric, and constraint audit','',
       'Frozen *real* Q550 v18.7/v18.6 full11D source and Jacobian/cocycle arrays; standard and tight independently inspected.',
       '', '## Observed facts (numerical; not rigorous interval certificates)','']
    for d in tolout:
        lines+=['- '+d['tolerance']+': Euclidean crossing xi='+str([round(x,9) for x in d['euclidean_bracketed_roots_xi']])+
               '; inverse-cocycle-source frame crossing xi='+str(d['source_frame_bracketed_roots_xi'])+
               '; source frame minimal cosine='+str(d['min_source_frame_cosine_xi1_to8'])]
    lines+=['','## Coordinate-free statement','',
       'For linearly independent vectors E and K the sign of E^T W K is not intrinsic without choosing a positive-definite metric W. This is mathematical, not a numerical defect. Co-moving pullback W=P^-T P^-1 only preserves an arbitrary initial metric.',
       '', '## Tests of the original ODE tangent structure','',
       'Each archived J gives a local linearization; the audit checks P evolution, the transport-vs-source derivative decomposition, a necessary nullspace screen for a constant linear first integral, and a constant-symmetric-quadratic-form screen. These screens cannot establish or refute nonlinear conserved quantities or a state-dependent material metric.',
       '', '## Constraint gate','',
       'No genuine C(y) or material energy G(y) accompanied the archived checkpoint. The v18.6 source was generated from off-constraint kicks. Even if C(y) is found, projecting K2 onto a tangent plane would incorrectly erase required 1/2 D2C[u,u] normal curvature.',
       '', '## Optional original-ODE replay and candidate verification','',
       'Use `python3 scripts/model_structure_check_v18_9.py --base-root /path/to/original/q550/base` to replay from the **archived kick state** without repeating 400 orbits. To inspect independently sourced constraints and metric also pass `--structure-module /path/to/candidate.py --structure-provenance /path/to/metadata.json`. Report labels remain candidate diagnostics pending external physical verification.',
       '', '## Reproduction','',
       '`bash scripts/run_on_server.sh`; no external Q550 files needed for mandatory frozen-checkpoint audit; full original v18.7→v18.6→v18.5 genealogy retained under inputs.',
       '', '## Important limitation','',
       'The computed crossing count is only grid-bracketed (xi step 0.0125), not an exclusion proof for missed crossing pairs. Absence of roots in one chosen metric is NOT absence of memory, torsion, or another physical mechanism.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'version':VERSION,'root_std':tolout[0]['euclidean_bracketed_roots_xi'],
      'co_moving_std':tolout[0]['source_frame_bracketed_roots_xi'],
      'physical_metric_status':summary['physical_metric_status']},indent=2))

if __name__=='__main__':main()
