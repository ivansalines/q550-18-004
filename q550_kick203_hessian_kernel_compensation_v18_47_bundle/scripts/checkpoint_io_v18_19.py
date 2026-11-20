"""v18.19 original-server v18.18 provenance gate; delegate authenticated v18.17->v18.6 genealogy."""
from pathlib import Path
import hashlib,json,zipfile
import checkpoint_io_v18_18 as old
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs/v18_18_original_server_sendback.zip'
NAME='q550_kick203_recurrence_cancellation_v18_18_bundle/'
def read():
    with zipfile.ZipFile(INPUT) as z:
        broken=z.testzip()
        if broken:raise RuntimeError('v18.18 CRC failure: '+broken)
        roots={n.split('/')[0] for n in z.namelist()}
        if roots!={NAME.rstrip('/')}:raise RuntimeError('v18.18 unexpected root '+str(roots))
        manifest=json.loads(z.read(NAME+'SHA256SUMS_OUTPUT.json'))['files']
        for rel,expected in manifest.items():
            if hashlib.sha256(z.read(NAME+rel)).hexdigest()!=expected:raise RuntimeError('v18.18 SHA mismatch '+rel)
        child=z.read(NAME+'inputs/v18_17_original_sendback.zip')
        summary=json.loads(z.read(NAME+'results/summary.json'))
    tmp=ROOT/'.runtime';tmp.mkdir(exist_ok=True)
    path=tmp/'v18_17_original_sendback.zip'
    if path.exists() and hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(child).digest():
        raise RuntimeError('Stale transient v18.17 input; clear .runtime/ only')
    path.write_bytes(child)
    old.INPUT=path
    data,prior,provenance=old.read()
    provenance.update({'server_v18_18_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'server_v18_18_verified_members':len(manifest),'server_v18_18_status':summary['status']})
    return data,provenance
