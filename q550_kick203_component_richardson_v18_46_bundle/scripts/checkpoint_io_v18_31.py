"""SHA-pin original Q550 v18.30 and follow its nested executable genealogy.

The v18.30 manifest is fully authenticated. Its recursively embedded ZIPs are
cryptographically bound by the pinned root; read only necessary descendants
instead of repeating every already certified historical full ZIP scan.
The original v18.7/v18.6 numerical arrays are independently manifest-checked.
"""
from pathlib import Path
import zipfile,hashlib,json,io,re,sys
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.runtime/v18_30_original_server_sendback.zip'
PREFIX='q550_kick203_independent_hessian_constraint_v18_30_bundle/'
sys.path.insert(0,str(ROOT/'reference'))
import analyze_cached_v18_9 as original_checkpoint

def sha(data):return hashlib.sha256(data).hexdigest()
def read():
    expected=json.loads((ROOT/'ENVIRONMENT.json').read_text())['grandparent_v18_30_sha256']
    blob=SOURCE.read_bytes()
    if sha(blob)!=expected:raise RuntimeError('v18.30 immutable original SHA mismatch')
    rt=ROOT/'.runtime';rt.mkdir(exist_ok=True)
    top_proof=None;top_member_count=None
    for version in list(range(30,9,-1))+[9,7]:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            if version==30:
                if z.testzip():raise RuntimeError('v18.30 ZIP CRC failure')
                if {name.split('/')[0] for name in z.namelist()}!={PREFIX.rstrip('/')}:
                    raise RuntimeError('Wrong original v18.30 ZIP root')
                manifest=json.loads(z.read(PREFIX+'SHA256SUMS_OUTPUT.json'))['files']
                for name,hash_expected in manifest.items():
                    if sha(z.read(PREFIX+name))!=hash_expected:
                        raise RuntimeError('v18.30 SHA member mismatch '+name)
                top_member_count=len(manifest)
                top_proof=json.loads(z.read(PREFIX+'results/original_independent_hessian_proof_v18_30.json'))
                if top_proof.get('status')!='ORIGINAL_Q550_INDEPENDENT_RHS_HESSIAN_DIAGNOSTIC_COMPLETED' or not top_proof.get('original_ODE_input_SHA256_verified') or top_proof.get('original_ode_integrations')!=10:
                    raise RuntimeError('Original v18.30 Q550 server proof not completed')
            members=z.namelist()
            if version==20:
                seeds=[n for n in members if n.endswith('/results/matched_hidden_seeds.json')]
                if len(seeds)!=1:raise RuntimeError('Original v18.20 seeds missing or ambiguous')
                (rt/'matched_hidden_seeds_v18_20.json').write_bytes(z.read(seeds[0]))
            if version==7:
                p=rt/'v18_7_original_sendback.zip';p.write_bytes(blob)
                original_checkpoint.ARCHIVE=p
                data,earliest=original_checkpoint.get_data()  # SHA-check v18.7 and v18.6
                provenance=dict(earliest, v18_30_server_SHA256=expected,
                     v18_30_verified_members=top_member_count,
                     v18_30_status=top_proof['status'],
                     nested_genesis='SHA_PINNED_V18_30_ROOT_WITH_NESTED_UNMODIFIED_GENEALOGY',
                     intermediate_ancestor_manifests_not_individually_recomputed=True)
                return data,provenance,top_proof,None,None
            next_version = 7 if version==9 else version-1
            tag=f'v18_{next_version}_'
            matches=[name for name in members if '/inputs/' in name or name.startswith('inputs/')]
            matches=[name for name in matches if tag in name and name.endswith('.zip')]
            if len(matches)!=1:
                raise RuntimeError(f'Cannot uniquely follow nested v18.{version-1}: {matches}')
            blob=z.read(matches[0])
    raise RuntimeError('Original Q550 v18.7 descendant not reached')
