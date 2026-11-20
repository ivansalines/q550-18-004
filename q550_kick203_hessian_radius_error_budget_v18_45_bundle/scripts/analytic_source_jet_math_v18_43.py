"""Model-aware first source jet, with explicit numerical Jacobian directional term.

Exactly differentiated: polynomial phase-candidate C, chain rule for g and grad r,
material transverse projection, normal amplitude and tangent-frame geometry.
NOT exact analytically: dot B = L_F B uses five-point directional differences
of the original analytic Jacobian; it is checked at multiple steps. Second
jet is differentiation of the first jet, NOT an analytic ODE second variation.
All material normalization is a declared Euclidean diagnostic gauge.
"""
import numpy as np
from manifold_math_v18_32 import PAIRS
from residual_gradient_math_v18_37 import source_geometry,unit
from tangent_source_math_v18_39 import minimal_rotation


def candidate_tensors(y,q,center,scale,theta0,degree,phase):
    q=np.asarray(q,float); y=np.asarray(y,float)
    H=np.zeros((10,10))
    if degree==2:
        for k,(i,j) in enumerate(PAIRS):
            H[i,j]+=q[10+k];H[j,i]+=q[10+k]
    gt=np.zeros(10);gtt=np.zeros(10)
    if phase:
        k=10+(55 if degree==2 else 0)
        a=y[10]-theta0
        s,c=np.sin(a),np.cos(a)
        qs=q[k:k+10]; qc=q[k+10:k+20]
        gt=qs*c-qc*s
        gtt=-qs*s-qc*c
    return H,gt,gtt


def normalized_jac(J,scale):
    return J[:10,:10]*scale[None,:]/scale[:,None]


def time_step_from_scale(y,fy,scale,scaled_target):
    speed=max(float(np.linalg.norm(fy[:10]/scale)),float(abs(fy[10])),1.0)
    return float(scaled_target/speed)


def directional_B_dot(y,fy,jac,scale,h):
    if not np.isfinite(h) or h<=0:raise ValueError('Positive directional step required')
    mats=[normalized_jac(np.asarray(jac(np.asarray(y)+k*h*np.asarray(fy)),float),scale)
          for k in (-2.,-1.,1.,2.)]
    return (mats[0]-8*mats[1]+8*mats[2]-mats[3])/(12*h)


def material_source_jet(y,f,jac,q,center,scale,theta0,degree,phase,mu,scaled_target=5e-4):
    y=np.asarray(y,float);fy=np.asarray(f(0.,y),float);J=np.asarray(jac(y),float)
    scale=np.asarray(scale,float)
    G=source_geometry(y,fy,J,q,center,scale,theta0,degree,phase,mu)
    H,gt,gtt=candidate_tensors(y,q,center,scale,theta0,degree,phase)
    g=G['g'];gr=G['gr'];n=unit(g);gn=float(np.linalg.norm(g))
    v=fy[:10]/scale;w=float(fy[10]);vdot=(J[:10,:]@fy)/scale;wdot=float(J[10,:]@fy)
    gdot=H@v+w*gt
    ndot=(gdot-n*(n@gdot))/gn
    h=time_step_from_scale(y,fy,scale,scaled_target)
    Bdot=directional_B_dot(y,fy,jac,scale,h)
    grdot=H@vdot+Bdot.T@g+G['B'].T@gdot+wdot*gt+w*w*gtt-mu*gdot
    p=G['perp']/gn
    pdot=(grdot-n*(n@grdot)-ndot*(n@gr)-n*(ndot@gr))/gn-p*float(n@gdot)/gn
    # Derivative of source, not derivative of projected observer component.
    if abs(float(n@p))>1e-10:raise RuntimeError('Projected source is not tangent')
    if abs(float(n@pdot+ndot@p))>1e-7*max(np.linalg.norm(pdot),1):
        raise RuntimeError('Derivative of tangent condition failed')
    if not np.all(np.isfinite(pdot)):raise RuntimeError('Nonfinite source first jet')
    return dict(p=p,pdot=pdot,n=n,ndot=ndot,g=g,gr=gr,grdot=grdot,
                r=G['r'],C=G['C'],theta=float(y[10]),
                directional_h=h,scaled_target=scaled_target,
                raw_speed=float(np.linalg.norm(fy[:10]/scale)),
                Bdot_norm=float(np.linalg.norm(Bdot)))


def coords(p,own,base,base0,frame0):
    own=unit(own);base=unit(base)
    F=minimal_rotation(base0,base)@frame0
    return F.T@(minimal_rotation(own,base)@(p-own*(own@p)))


def moving_coords_and_jet(src,base,base0,frame0):
    p=src['p'];pdot=src['pdot']
    n=src['n'];ndot=src['ndot']
    b=base['n'];bdot=base['ndot']
    v=coords(p,n,b,base0,frame0)
    # Analytic d/dt p; geometric derivative of the gauge evaluated on normal
    # tangent jets (not from the ODE time samples). This is independently
    # adjustable and is not a material connection certificate.
    h=.0008/max(float(np.linalg.norm(ndot)),float(np.linalg.norm(bdot)),1.)
    right=coords(p,n+h*ndot,b+h*bdot,base0,frame0)
    left=coords(p,n-h*ndot,b-h*bdot,base0,frame0)
    gauge_dot=(right-left)/(2*h)
    out=coords(pdot,n,b,base0,frame0)+gauge_dot
    return v,out,float(np.linalg.norm(gauge_dot)),h


def curvature(vdot,vddot):
    vdot=np.asarray(vdot,float);vddot=np.asarray(vddot,float)
    ss=float(vdot@vdot)
    if ss<1e-50:return float('nan')
    perp=vddot-(float(vddot@vdot)/ss)*vdot
    return float(np.linalg.norm(perp)/ss)


def five_point_d1(vminus2,vminus1,vplus1,vplus2,h):
    return (np.asarray(vminus2)-8*np.asarray(vminus1)+8*np.asarray(vplus1)-np.asarray(vplus2))/(12*h)


def five_point_d2(vminus2,vminus1,vzero,vplus1,vplus2,h):
    return (-np.asarray(vplus2)+16*np.asarray(vplus1)-30*np.asarray(vzero)+16*np.asarray(vminus1)-np.asarray(vminus2))/(12*h*h)
