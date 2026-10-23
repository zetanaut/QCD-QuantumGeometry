"""Canonical leading six-body excursion on the full non-diagonal free shells."""
import argparse
import json
import time

import numpy as np

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_shell_current import shell_density_between


def run(cutoff):
    start=time.monotonic()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_shell_fock_reference(p,out_cutoff=cutoff)
    obj=CanonicalSixBodyExcursion(ref,out_cutoff=cutoff)
    build=time.monotonic()-start
    a=obj.waves(1.4)
    h=sum(a["six_source"][n].conj().T@v for n,v in a["six_leading"].items())
    prob=sum(v.conj().T@v for v in a["six_leading"].values())
    back=sum(a["five_leading"][n].conj().T@v for n,v in a["five_backflow"].items())
    norm=prob+back+back.conj().T
    def current(delta):
        result=sum(a["six_leading"][n].conj().T@shell_density_between(
            s,s,a["six_leading"][n],delta,b_GeV=.4,charge="electromagnetic")
            for n,s in obj.spaces6.items())
        for n,s in obj.spaces5.items():
            r=shell_density_between(s,s,np.column_stack((
                a["five_leading"][n],a["five_backflow"][n])),delta,b_GeV=.4,charge="electromagnetic")
            k=len(p.matrix)
            result+=a["five_backflow"][n].conj().T@r[:,:k]+a["five_leading"][n].conj().T@r[:,k:]
        return result
    parts=obj.return_components(1.4)
    return dict(scope="complete represented leading six-body excursion; not full g^4 QCD or a QQ55 insertion",
       cutoff=cutoff,build_seconds=build,total_seconds=time.monotonic()-start,
       Q5_dimensions={n:s.dimension for n,s in obj.spaces5.items()},
       Q6_dimensions={n:s.dimension for n,s in obj.spaces6.items()},
       direct_component_norms_GeV2=obj.direct_component_norms,
       H4_norm_GeV2=float(np.linalg.norm(h)),N4_norm=float(np.linalg.norm(norm)),
       six_probability_coefficient_norm=float(np.linalg.norm(prob)),
       five_backflow_coefficient_norm=float(np.linalg.norm(back+back.conj().T)),
       electromagnetic_rho4_norm=float(np.linalg.norm(current((.12,.16)))),
       charge_norm_error=float(np.linalg.norm(current((0,0))-norm)),
       return_component_norms_GeV2={n:float(np.linalg.norm(v)) for n,v in parts.items()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff",type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(run(args.cutoff),sort_keys=True),flush=True)
