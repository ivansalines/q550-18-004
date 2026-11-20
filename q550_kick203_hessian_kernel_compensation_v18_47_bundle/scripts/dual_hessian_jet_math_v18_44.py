"""v18.44 independent numerical Hessian contraction, analytic chart differential.

F is the *original* Q550 vector field, J its pinned analytic Jacobian.
There is NO symbolic Hessian of F. The two numeric constructions below use
independent original-model callbacks (field vs Jacobian) and separate stencils.
Coordinates z_i=(y_i-center_i)/scale_i; theta is unscaled and unwrapped.
The Euclidean chart/normal is a declared diagnostic convention, not a metric.
"""
import numpy as np
from residual_gradient_math_v18_37 import source_geometry, unit
from analytic_source_jet_math_v18_43 import candidate_tensors, directional_B_dot
from tangent_source_math_v18_39 import minimal_rotation

COEFF=np.array([1.,-8.,8.,-1.]);OFFSETS=np.array([-2.,-1.,1.,2.])

def normalized_full_speed(fy,scale):
    return np.r_[np.asarray(fy[:10],float)/scale,float(fy[10])]

def dot_B_transpose_g_from_jacobian(y,fy,jac,scale,g,target):
    """(L_F B)^T g, differentiate pinned analytic J in flow direction."""
    v=normalized_full_speed(fy,scale)
    step=target/max(np.linalg.norm(v),1.0)
    return directional_B_dot(y,fy,jac,scale,step).T@g

def mixed_rhs_hessian_contraction(y,fy,f,scale,g,h):
    """D_z[D_{flow} (g . F_z)] in each spatial e_i, independent of J.

    F_z=F_material/scale; g held constant. 4th-order mixed first derivatives:
    sum_{a,b} c_a c_b (g.F_z)(z+a*h*u+b*h*e_i)/(144*h**2).
    The flow direction u is unit in normalized (z,theta) space, with its
    original magnitude restored. Error is O(h**4) plus cancellation noise.
    """
    y=np.asarray(y,float);g=np.asarray(g,float);scale=np.asarray(scale,float)
    v=normalized_full_speed(fy,scale);speed=float(np.linalg.norm(v))
    if speed<=0 or not np.isfinite(speed):raise ValueError('Degenerate flow speed')
    if not(np.isfinite(h) and 0<h<.2):raise ValueError('Unapproved Hessian displacement')
    u=v/speed;out=np.empty(10)
    weights=np.r_[scale,1.]
    for j in range(10):
        e=np.zeros(11);e[j]=1.
        s=0.
        for ii,a in enumerate(OFFSETS):
            for kk,b in enumerate(OFFSETS):
                yp=y+weights*(a*h*u+b*h*e)
                Fz=np.asarray(f(0.,yp),float)[:10]/scale
                s+=COEFF[ii]*COEFF[kk]*float(g@Fz)
        out[j]=speed*s/(144*h*h)
    if not np.all(np.isfinite(out)):raise RuntimeError('Nonfinite original RHS Hessian contraction')
    return out

def source_jet_from_contraction(y,fy,J,q,center,scale,theta0,degree,phase,mu,dbtg):
    """Exact polynomial chain rule, substituted Hessian contraction (DB)^T*g."""
    y=np.asarray(y,float);fy=np.asarray(fy,float);J=np.asarray(J,float)
    G=source_geometry(y,fy,J,q,center,scale,theta0,degree,phase,mu)
    H,gt,gtt=candidate_tensors(y,q,center,scale,theta0,degree,phase)
    g=G['g'];gr=G['gr'];n=unit(g);gn=float(np.linalg.norm(g))
    v=fy[:10]/scale;w=float(fy[10]);vdot=(J[:10,:]@fy)/scale;wdot=float(J[10,:]@fy)
    gdot=H@v+w*gt
    ndot=(gdot-n*(n@gdot))/gn
    grdot=H@vdot+dbtg+G['B'].T@gdot+wdot*gt+w*w*gtt-mu*gdot
    p=G['perp']/gn
    pdot=(grdot-n*(n@grdot)-ndot*(n@gr)-n*(ndot@gr))/gn-p*float(n@gdot)/gn
    if not(np.isfinite(p).all() and np.isfinite(pdot).all()):raise RuntimeError('Invalid jet')
    if abs(n@pdot+ndot@p)>1e-7*max(1.,np.linalg.norm(pdot)):
        raise RuntimeError('Tangent derivative identity failed')
    return dict(p=p,pdot=pdot,n=n,ndot=ndot,g=g,gr=gr,r=G['r'],C=G['C'],theta=float(y[10]))

def minimal_rotation_with_derivative(a,b,da,db):
    a=unit(a);b=unit(b);da=np.asarray(da,float);db=np.asarray(db,float)
    c=float(a@b)
    if c<-1+1e-10:raise ValueError('Unresolved antipodal chart')
    K=np.outer(b,a)-np.outer(a,b)
    dK=np.outer(db,a)+np.outer(b,da)-np.outer(da,b)-np.outer(a,db)
    dc=float(da@b+a@db)
    R=np.eye(len(a))+K+K@K/(1.+c)
    dR=dK+(dK@K+K@dK)/(1.+c)-(K@K)*(dc/(1.+c)**2)
    return R,dR

def exact_chart_jet(source,reference,reference_initial_normal,frame0):
    """Analytic Frechet derivative of two minimal rotations and tangent projection.

    Returns 9D source, 9D first flow jet, and resolved norm of contributions
    from reference-frame transport and own-normal alignment. Does NOT assert
    that this connection is intrinsic to the Q550 material.
    """
    p=source['p'];pd=source['pdot'];n=source['n'];nd=source['ndot']
    b=reference['n'];bd=reference['ndot'];b0=reference_initial_normal
    U,dU=minimal_rotation_with_derivative(b0,b,np.zeros_like(b),bd)
    V,dV=minimal_rotation_with_derivative(n,b,nd,bd)
    F=U@frame0;Fd=dU@frame0
    tang=p-n*(n@p)
    tangdot=pd-nd*(n@p)-n*(nd@p+n@pd)
    mapped=V@tang
    v=F.T@mapped
    part_source=F.T@V@tangdot
    part_own=F.T@dV@tang
    part_base=Fd.T@mapped
    jet=part_source+part_own+part_base
    if not(np.isfinite(v).all() and np.isfinite(jet).all()):raise RuntimeError('Nonfinite chart jet')
    return dict(v=v,jet=jet,source_differential=part_source,own_alignment_differential=part_own,
        frame_transport_differential=part_base,connection_differential=part_own+part_base)
