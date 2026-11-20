"""Authenticated archival preflight; NO new original-Q550 ODE run is claimed."""
import csv,io,json,sys,hashlib,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'reference')]
from checkpoint_io_v18_28 import read

def main():
    d,p,proof,raw=read();r=list(csv.DictReader(io.StringIO(raw)))
    if len(r)!=45:raise RuntimeError('Expected 45 genuine previous epsilon-pair rows')
    for row in r:
        if float(row['tight_even_rms'])<0 or float(row['even_inter_tier_rms_disagreement'])<0:
            raise RuntimeError('Invalid archived proxy')
    out=ROOT/'results';out.mkdir(exist_ok=True)
    selected=[x for x in r if (x['observer'],float(x['anchor'])) in (
               ('material3',3.5),('material3',4.5),('raw6',4.5),('raw4',4.0))]
    summary={'version':'v18.28','status':'AUTHENTICATED_PARENT_PREFLIGHT_ONLY',
      'parent_sha256':p['v18_27_sendback_sha256'],
      'source_proof':proof['status'],'parent_floor_rows':len(r),
      'selected_parent_rows':len(selected),'selected_median_ratio':float(np.median([
        float(q['even_signal_over_tier_difference']) for q in selected])),
      'sample_source_only':True,'new_original_ode_integrations':0,
      'not_a_convergence_or_hysteresis_certificate':True}
    (out/'frozen_preflight_v18_28.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
