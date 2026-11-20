"""Original Q550 nonlinear oracle for v18.25; sensor-only learner is separate.

The server authenticates pinned RHS sources. First 32 randomly selected training
perturbations are run at shifted sections to calibrate past-history alignment;
no shifted calibration futures are ever passed to the learned aligner.
"""
from pathlib import Path
import sys,json,zipfile,argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_25 import read
from checkpoint_io_v18_21 import PREFIX as PREFIX20
from frozen_preflight import observers
from blind_experiment import initial_directions,OBSERVERS
from blind_math import HISTORY,HORIZONS
from transport_math_v18_25 import evaluate,ANCHORS,CAL_N
import q550_state_impulse_response_v13 as original

def integrate(rhs,t0,t1,state,grid):
    sol=solve_ivp(rhs,(float(t0),float(t1)),state,t_eval=np.asarray(grid,float),
        method='DOP853',rtol=5e-11,atol=6e-12,max_step=.04)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y.T

def generate(rhs,anchor,center,direction,amplitude):
    past=np.asarray(anchor+HISTORY,float)
    future=np.asarray(anchor+HORIZONS,float)
    init=center.copy();init[:10]+=amplitude*np.asarray(direction,float)
    backward=integrate(rhs,anchor,float(past[0]),init,past[::-1])
    forward=integrate(rhs,anchor,float(future[-1]),init,np.r_[anchor,future])
    states=np.vstack((backward[::-1],forward[1:]))
    assert states.shape==(len(HISTORY)+len(HORIZONS),11)
    return states

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    args=parser.parse_args()
    pinned={}
    for key,(filename,digest) in original.DEPENDENCIES.items():
        source=args.base_root/filename
        if not source.exists():raise FileNotFoundError('Original Q550 source missing: '+str(source))
        if original.hash_file(source)!=digest:raise RuntimeError('ODE dependency SHA256 changed: '+str(source))
        pinned[key]={'path':str(source),'sha256':digest}
    data,prov,proof=read();d=data['tight'];Hs=observers(d)
    with zipfile.ZipFile(ROOT/'.runtime/v18_20_original_server_sendback.zip') as z:
        seeds=json.loads(z.read(PREFIX20+'results/matched_hidden_seeds.json'))
    directions=initial_directions({s['name']:s['seed'] for s in seeds})
    audit=original.load_module(args.base_root/original.DEPENDENCIES['audit'][0],'v1825_audit')
    support=original.load_module(args.base_root/original.DEPENDENCIES['support'][0],'v1825_support')
    params=audit.ModelParameters(gamma_rho=.3)
    kappa0=(55./6.)*(10./550.)
    slope=float(d['slope']);y0=np.asarray(d['source_state_11D'],float)
    def rhs(t,y):
        return slope*np.asarray(support.support_rhs_float(np.asarray(y,float),kappa0=kappa0,
                        params=params,audit=audit),float)
    anchor_states=integrate(rhs,0.,4.5,y0,np.array(ANCHORS))
    baseline={a:generate(rhs,a,center,np.zeros(10),0.) for a,center in zip(ANCHORS,anchor_states)}
    phase={str(a):float(center[-1]) for a,center in zip(ANCHORS,anchor_states)}
    P=CubicSpline(np.asarray(d['xi_nodes']),np.asarray(d['P_full11D'][:,:10,:10]),axis=0)
    out={obs:{} for obs in OBSERVERS}
    n_integrations=len(ANCHORS)
    # Generate each trajectory only once for all three observer projections.
    common=directions['material3']
    for anchor,center in zip(ANCHORS,anchor_states):
        raw={}
        for split in (('train','val','test') if anchor==4. else ('cal','test')):
            U=common['train'][:CAL_N] if split=='cal' else (common['test'][:-1] if split=='test' else common[split])
            eps=1e-5 if split in ('train','val','cal') else 3e-6
            chunks=[]
            for direction in U:
                states=generate(rhs,anchor,center,direction,eps)
                phase_defect=float(np.max(abs(states[:,10]-baseline[anchor][:,10])))
                if phase_defect>1e-8:raise RuntimeError('Clock phase unequal '+str(phase_defect))
                chunks.append((states[:,:10]-baseline[anchor][:,:10])/eps)
                n_integrations+=1
            raw[split]=np.asarray(chunks)
            print('ORIGINAL_Q550_v18_25',anchor,split,len(U),flush=True)
        for obs in OBSERVERS:
            H=Hs[obs];subset={}
            for split,values in raw.items():
                samples=np.einsum('ab,ntb->nta',H,values)
                subset[split]=(samples[:,:len(HISTORY)],samples[:,len(HISTORY):].reshape(len(samples),-1))
            special=directions[obs]['matched_seed'].copy()
            if anchor!=4.:
                transported=P(anchor)@np.linalg.solve(P(4.),special)
                special=transported/np.linalg.norm(transported)
            states=generate(rhs,anchor,center,special,3e-6)
            n_integrations+=1
            if np.max(abs(states[:,10]-baseline[anchor][:,10]))>1e-8:
                raise RuntimeError('Hidden-probe phase mismatch')
            response=((states[:,:10]-baseline[anchor][:,:10])/3e-6)@H.T
            old_x,old_y=subset['test']
            subset['test']=(np.concatenate((old_x,response[:len(HISTORY)][None]),axis=0),
                            np.concatenate((old_y,response[len(HISTORY):].reshape(1,-1)),axis=0))
            out[obs][anchor]=subset
    summary=evaluate(out,'original',{
      'status':'ORIGINAL_Q550_NONLINEAR_PAIRED_DELAY_FRAME_TRANSPORT_COMPLETED',
      'original_rhs_sources':pinned,
      'original_v18_24_server_sha256':prov['v18_24_server_sha256'],
      'train_amplitude':1e-5,'validation_amplitude':1e-5,
      'section_calibration_amplitude':1e-5,'test_amplitude':3e-6,
      'independent_nonlinear_ode_integrations':n_integrations,
      'reference_phase':phase,
      'observed_sensor_data_only_to_alignment_and_prediction':True,
      'paired_calibration_training_histories_only':True,
      'shifted_anchors_change_clock_and_material_posture':True})
    (ROOT/'results/original_ode_transport_proof.json').write_text(json.dumps({
       'version':'v18.25','status':'ORIGINAL_Q550_NONLINEAR_PAIRED_DELAY_FRAME_TRANSPORT_COMPLETED',
       'parent_original_v18_24_status':proof['status'],
       'parent_original_v18_24_sha256':prov['v18_24_server_sha256'],
       'original_rhs_pinned_sha256':pinned,
       'independent_nonlinear_ode_integrations':n_integrations,
       'calibration_uses_only_observed_history':True,
       'no_test_future_leakage':True,
       'off_constraint_diagnostic':True},indent=2)+'\n')
    print('SERVER_NONLINEAR_PROOF',summary['rows'],'rows; independent integrations',n_integrations)

if __name__=='__main__':main()
