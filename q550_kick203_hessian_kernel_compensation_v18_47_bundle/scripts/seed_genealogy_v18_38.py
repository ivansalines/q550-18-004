"""v18.38.1: authenticate regenerated seeds against their ORIGINAL v18.36 attempt ledger.

The v18.37 case summary intentionally does not repeat the v18.36
number_of_seed_attempts field. Do not synthesize a count from v18.37.
"""
import math


def seed_key(row):
    return (row['id'],round(float(row['anchor_xi']),8),int(row['branch']))


def index_archived_attempts(rows):
    indexed={}
    for row in rows:
        k=seed_key(row)
        indexed.setdefault(k,[]).append(row)
    if not indexed:
        raise RuntimeError('Authentic v18.36 seed attempt ledger empty')
    for k,items in indexed.items():
        indices=[int(row['try_index']) for row in items]
        if indices!=list(range(1,len(items)+1)):
            raise RuntimeError(f'Non-consecutive or duplicated v18.36 seed attempts: {k}')
    return indexed


def _archived_bool(value):
    if isinstance(value,bool):return value
    if str(value) == 'True':return True
    if str(value) == 'False':return False
    raise RuntimeError('Invalid archived v18.36 seed boolean')


def verify_regenerated_attempts(ref, actual, archived_cases, archived_attempts):
    if ref not in archived_cases or ref not in archived_attempts:
        raise RuntimeError(f'Missing original v18.36 seed genealogy: {ref}')
    original=archived_cases[ref]
    if 'number_of_seed_attempts' not in original:
        raise RuntimeError('Missing number_of_seed_attempts in ORIGINAL v18.36 case')
    expected=int(original['number_of_seed_attempts'])
    saved=archived_attempts[ref]
    if expected<1 or len(saved)!=expected or len(actual)!=expected:
        raise RuntimeError(f'Original v18.36 seed attempt count mismatch: {ref}')
    for n,(a,b) in enumerate(zip(actual,saved),start=1):
        if int(a['try_index'])!=n or int(b['try_index'])!=n:
            raise RuntimeError(f'Original seed try index mismatch: {ref}, attempt={n}')
        for flag in ('valid_projection','in_support'):
            if _archived_bool(a[flag])!=_archived_bool(b[flag]):
                raise RuntimeError(f'Original v18.36 seed replay flag mismatch: {ref}, {flag}')
        for metric in ('scaled_step','scaled_radius'):
            x=float(a[metric]);y=float(b[metric])
            if not (math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-9)):
                raise RuntimeError(f'Original v18.36 seed attempt mismatch: {ref}, {metric}')
    if not _archived_bool(saved[-1]['in_support']):
        raise RuntimeError('Original v18.36 final seed attempt was not accepted')
    return True
