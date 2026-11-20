import numpy as np
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'inputs'/'synthetic_smoke.npz'
p.parent.mkdir(exist_ok=True)
rng=np.random.default_rng(1848)
np.savez(p,states=rng.normal(size=(4,11)),directions_u=rng.normal(size=(4,11)),directions_v=rng.normal(size=(4,11)))
print(p)
