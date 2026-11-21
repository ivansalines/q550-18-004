"""Offline source-equivalence check against the SHA-pinned Q550 originals copied verbatim."""
import unittest,hashlib,sys,importlib.util,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'scripts'))
from structural_jet import structural_jet

EXPECTED={
 'test_support_mobius_monodromy_audit.py':'90792fc9ad144333df521c7c0862193125c4702319fc3aa1a2bcef45491b2864',
 'test_support_q550_support_ode_interval_residual_pilot_v1.py':'5b5938e7920b2ecd242fb8f416a18c7c47ae10aa06e027e0727f15e284c98c61',
 'test_support_q550_exact_nonlinear_local_tube_v2_1.py':'a9d1ec342c268da8e721dc641cab8cc4cf39f69d821dfd4191163833ba04e881',
}

def load(filename,modname):
    path=HERE/'reference'/filename
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED[filename]:raise RuntimeError('reference file checksum drift '+filename)
    spec=importlib.util.spec_from_file_location(modname,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[modname]=module;spec.loader.exec_module(module)
    return module

class OriginalEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=load('test_support_mobius_monodromy_audit.py','audit_q550_e')
        cls.support=load('test_support_q550_support_ode_interval_residual_pilot_v1.py','support_q550_e')
        cls.tube=load('test_support_q550_exact_nonlinear_local_tube_v2_1.py','tube_q550_e')
    def test_original_rhs_and_jacobian(self):
        rng=np.random.default_rng(184847)
        params=self.audit.ModelParameters(gamma_rho=.3)
        for _ in range(24):
            y=np.array([.07,.02,.35,.04,-.11,.01,.21,.05,.0,.03,.14])+rng.normal(0,.12,11)
            for k in (1/6,35*math.pi*10/(12*550)):
                f,j,_=structural_jet(y,k,params)
                F=np.asarray(self.support.support_rhs_float(y,kappa0=k,params=params,audit=self.audit),float)
                J=np.asarray(self.tube.float_support_jacobian(y,kappa0=k,params=params,audit=self.audit,base=self.support),float)
                self.assertLess(np.linalg.norm(f-F)/max(np.linalg.norm(F),1),1e-11)
                self.assertLess(np.linalg.norm(j-J)/max(np.linalg.norm(J),1),1e-10)
if __name__=='__main__':unittest.main()
