"""Training-only cofactor ranking, gauge-aware overlap and return diagnostics.

All normalized-coordinate distances and angles are diagnostic, not material
metrics. Clock phase recurrence is NOT a material Poincare return.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

def training_rank(candidates):
    """Rank by a normalized Lie-cofactor residual, NOT min/max singular value.

    This is a fixed training-only statistic provided by ensemble_score. At
    lambda=mod=0 it is one by its definition, preventing a tiny singular
    ratio caused by a slow drift from automatically winning the selection.
    """
    if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('Duplicate candidates')
    for c in candidates:
        if not np.isfinite(c['train_relative_residual']) or c['train_relative_residual']<0:
            raise ValueError('Invalid cofactor training residual')
    return sorted(candidates,key=lambda c:(c['train_relative_residual'],c['feature_count'],c['id']))

def shortlist(candidates,per_family=3,include_legacy=True):
    ranked=training_rank(candidates)
    selected=[]
    for phase in (False,True):
        selected += [c for c in ranked if c['phase']==phase][:per_family]
    if include_legacy:
        # Two controls chosen from *historical* v18.32 selection, not holdout.
        old=['phaseblind_d2_lam-0.50_mod+0.00','phasecov_d2_lam+0.00_mod+0.00']
        selected += [next(c for c in ranked if c['id']==name) for name in old]
    dedup={c['id']:c for c in selected}
    return sorted(dedup.values(),key=lambda c:(c['train_relative_residual'],c['id']))

def normalized_gradient(y,q,center,scale,theta0,degree,phase):
    from manifold_math_v18_32 import features_material_gradient
    g=features_material_gradient(y,center,scale,theta0,degree,phase).T@q
    return np.asarray(g)*np.asarray(scale)

def gauge_agnostic_normal_angle(g,h):
    a=np.linalg.norm(g); b=np.linalg.norm(h)
    if min(a,b)<1e-14:return None
    dot=np.clip(abs(float(np.dot(g,h)))/(a*b),0.,1.)
    return float(np.degrees(np.arccos(dot)))

def cofactor_predicted_value(thetas,clock_rates,values0,theta0,lam,mod,times):
    mu=(lam+mod*np.sin(np.asarray(thetas)-theta0))*np.asarray(clock_rates)
    # mu is per ξ in the Q550 equations, not per clock angle. Clock rate is
    # intentionally ignored in the scalar transport integral.
    mus=lam+mod*np.sin(np.asarray(thetas)-theta0)
    integral=np.r_[0.,cumulative_trapezoid(mus,np.asarray(times))]
    return values0*np.exp(integral)

def phase_clock_return(delta_theta,tolerance=.03):
    wraps=int(np.rint(delta_theta/(2*np.pi)))
    error=abs(delta_theta-wraps*2*np.pi)
    return {'phase_turns':wraps,'phase_wrapping_error_rad':float(error),
            'phase_cycle_only':bool(wraps!=0 and error<tolerance),
            'material_recurrence_certified':False}

def surface_accept(holdout,normal_rel,drift_rel,anchor_rel,phase_rel,
                   overlap_angle,overlap_offset,threshold=1e-6,
                   max_overlap_angle_deg=5.,max_offset_scaled=.01):
    """Diagnostic screening only; nonprivileged coordinate metric."""
    vals=[holdout,normal_rel,drift_rel,anchor_rel,phase_rel]
    return (all(v is not None and np.isfinite(v) and v<threshold for v in vals)
        and overlap_angle is not None and np.isfinite(overlap_angle)
        and overlap_angle<max_overlap_angle_deg
        and overlap_offset is not None and np.isfinite(overlap_offset)
        and overlap_offset<max_offset_scaled)

