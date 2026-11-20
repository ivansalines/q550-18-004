"""Independent original-Q550 nonlinear vanishing amplitude experiment.

For each amplitude, regenerate train / validation / held-out reduced histories
from PINNED original 11D ODE. No tangent propagator or material-state access in
the learning module. Reference baselines are integrated independently.
Positive/negative probe pairs quantify even nonlinearity and cancellation.
"""
from pathlib import Path
import sys,json,zipfile,argparse
import numpy as np
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_26 import read
from checkpoint_io_v18_21 import PREFIX as PREFIX20
from frozen_preflight import observers
from blind_experiment import initial_directions,OBSERVERS
from blind_math import HISTORY,HORIZONS
from transport_math_v18_25 import ANCHORS,CAL_N
from amplitude_math_v18_26 import AMPLITUDES,evaluate
from nonlinear_server_v18_25 import generate,integrate
import q550_state_impulse_response_v13 as original

def export_datasets(output,datasets):
    """Save raw REDUCED observed trajectories, never Q550 hidden full states."""
    arrays={}
    for eps,obs_dict in datasets.items():
      for obs,anchored in obs_dict.items():
        for anchor,split_dict in anchored.items():
          for split,(past,future) in split_dict.items():
            key=f'e{eps:.10g}__{obs}__a{anchor:g}__{split}'
            arrays[key+'__past']=np.asarray(past,dtype=np.float64)
            arrays[key+'__future']=np.asarray(future,dtype=np.float64)
    np.savez_compressed(output,**arrays)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-root',type=Path,default=original.DEFAULT_BASE)
    parser.add_argument('--amplitudes',type=float,nargs='+',default=AMPLITUDES,
                        help='Diagnostic override (default 5 fixed amplitudes)')
    args=parser.parse_args()
    amplitudes=tuple(float(x) for x in args.amplitudes)
    if any(e<=0 or e>1e-4 for e in amplitudes) or len(set(amplitudes))!=len(amplitudes):
        raise ValueError('Expect distinct positive small amplitude samples, <= 1e-4')
    if 1e-5 not in amplitudes:
        raise ValueError('Calibration reference amplitude 1e-5 must be present')
    authenticated={}
    for key,(filename,digest) in original.DEPENDENCIES.items():
        file=args.base_root/filename
        if not file.exists() or original.hash_file(file)!=digest:
            raise RuntimeError('Original Q550 dependency absent or hash changed: '+str(file))
        authenticated[key]={'path':str(file),'sha256':digest}
    data,prov,prior=read();d=data['tight'];Hs=observers(d)
    with zipfile.ZipFile(ROOT/'.runtime/v18_20_original_server_sendback.zip') as z:
        seeds=json.loads(z.read(PREFIX20+'results/matched_hidden_seeds.json'))
    directions=initial_directions({r['name']:r['seed'] for r in seeds})
    audit=original.load_module(args.base_root/original.DEPENDENCIES['audit'][0],'v1826_audit')
    support=original.load_module(args.base_root/original.DEPENDENCIES['support'][0],'v1826_support')
    params=audit.ModelParameters(gamma_rho=.3)
    k0=(55./6.)*(10./550.)
    slope=float(d['slope']); y0=np.asarray(d['source_state_11D'],float)
    def rhs(t,y):
        return slope*np.asarray(support.support_rhs_float(np.asarray(y,float),kappa0=k0,
                        params=params,audit=audit),float)
    center_states=integrate(rhs,0.,4.5,y0,np.asarray(ANCHORS))
    center_map=dict(zip(ANCHORS,center_states))
    baseline={anchor:generate(rhs,anchor,center,np.zeros(10),0.) for anchor,center in center_map.items()}
    P=CubicSpline(np.asarray(d['xi_nodes'],float),np.asarray(d['P_full11D'][:,:10,:10]),axis=0)
    special={}
    for anchor in ANCHORS:
        for obs in OBSERVERS:
            seed=directions[obs]['matched_seed'].copy()
            if anchor!=4.:
                seed=P(anchor)@np.linalg.solve(P(4.),seed)
                seed/=np.linalg.norm(seed)
            special[anchor,obs]=seed
    all_data={};curvature=[];count=len(ANCHORS)+len(ANCHORS)
    for ei,eps in enumerate(amplitudes):
        out={obs:{} for obs in OBSERVERS}
        for anchor in ANCHORS:
            center=center_map[anchor];raw={}
            for split in (('train','val','test') if anchor==4. else ('cal','test')):
                U=(directions['material3']['train'][:CAL_N] if split=='cal'
                     else directions['material3']['test'][:-1] if split=='test'
                     else directions['material3'][split])
                chunks=[]
                for direction in U:
                    states=generate(rhs,anchor,center,direction,eps)
                    defect=float(np.max(np.abs(states[:,10]-baseline[anchor][:,10])))
                    if defect>1e-8:raise RuntimeError('Phase mismatch '+str(defect))
                    chunks.append((states[:,:10]-baseline[anchor][:,:10])/eps)
                    count+=1
                raw[split]=np.asarray(chunks)
                print(f'Q550_v18_26 eps={eps:.2e} anchor={anchor} split={split} n={len(U)}',flush=True)
            for obs in OBSERVERS:
                H=Hs[obs];items={}
                for split,arr in raw.items():
                    reduced=np.einsum('ab,ntb->nta',H,arr)
                    items[split]=(reduced[:,:len(HISTORY)],reduced[:,len(HISTORY):].reshape(len(arr),-1))
                seed=special[anchor,obs]
                plus=generate(rhs,anchor,center,seed,eps)
                minus=generate(rhs,anchor,center,-seed,eps)
                count+=2
                for states in (plus,minus):
                    if np.max(np.abs(states[:,10]-baseline[anchor][:,10]))>1e-8:
                        raise RuntimeError('Hidden probe phase mismatch')
                p=((plus[:,:10]-baseline[anchor][:,:10])/eps)@H.T
                m=((minus[:,:10]-baseline[anchor][:,:10])/eps)@H.T
                odd=(p-m)/2.;even=(p+m)/2.
                full_time=anchor+np.r_[HISTORY,HORIZONS]
                tangent=np.einsum('ab,tbc,c->ta',H,P(full_time)@np.linalg.inv(P(anchor)),seed)
                curvature.append({'amplitude':eps,'anchor':anchor,'observer':obs,
                  'odd_vs_tangent_rms':float(np.sqrt(np.mean((odd-tangent)**2))),
                  'even_response_rms':float(np.sqrt(np.mean(even**2))),
                  'plus_vs_tangent_rms':float(np.sqrt(np.mean((p-tangent)**2))),
                  'odd_response_rms':float(np.sqrt(np.mean(odd**2))),
                  'numerical_differencing_warning':bool(not np.all(np.isfinite(odd)))})
                old_x,old_y=items['test']
                items['test']=(np.concatenate((old_x,p[:len(HISTORY)][None]),axis=0),
                               np.concatenate((old_y,p[len(HISTORY):].reshape(1,-1)),axis=0))
                out[obs][anchor]=items
        all_data[eps]=out
    result_dir=ROOT/'results';result_dir.mkdir(exist_ok=True)
    export_datasets(result_dir/'original_reduced_dataset_v18_26.npz',all_data)
    import csv
    with (result_dir/'original_even_odd_probe.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curvature[0]));writer.writeheader();writer.writerows(curvature)
    proof_meta={'status':'ORIGINAL_Q550_NONLINEAR_VANISHING_AMPLITUDE_SCAN_COMPLETED',
        'source_pinned':authenticated,'parent_v18_25_sha256':prov['v18_25_server_sha256'],
        'parent_v18_25_status':prior['status'],'amplitudes':list(amplitudes),
        'new_original_ode_integrations':count,
        'samples_per_trajectory':len(HISTORY)+len(HORIZONS),
        'plus_minus_probe_pairs':len(curvature),'phase_equal':True,
        'original_ode_used_for_oracle_only':True,
        'paired_calibration_uses_past_only':True}
    summary=evaluate(all_data,'original',proof_meta,amplitudes=amplitudes)
    (result_dir/'original_ode_amplitude_proof.json').write_text(json.dumps(proof_meta,indent=2)+'\n')
    print('ORIGINAL_Q550_v18_26_DONE',count,'independent integrations',summary['rows'], 'prediction rows',flush=True)

if __name__=='__main__':main()
