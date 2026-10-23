"""Bare five-body free-Q shell study; NOT the complete order-four model.

The retained P and K2,b,m,g remain fixed. No coefficient is refitted.
Exact non-diagonal Q propagation is compared with an intentionally wrong
diagonal-only control. This is a resolution diagnostic, not convergence.
"""
import argparse
import json
import time

import numpy as np
from scipy.linalg import eigh,eigvalsh
from scipy.optimize import brentq

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_current import shell_density_between


def run(cutoff,target):
    start=time.monotonic()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_shell_fock_reference(p,out_cutoff=cutoff)
    built=time.monotonic()
    couplings=[s.source.full_coupling(p.coupling) for s in ref.sectors]
    # Coherent actual coupling, without repeated propagation per source order.
    def effective(e):
        return p.matrix+sum(v.conj().T@s.propagation.resolve(e,v)
                            for s,v in zip(ref.sectors,couplings))
    def objective(e):
        return eigvalsh(effective(e),subset_by_index=[0,0])[0]-e
    floor=min(s.propagation.minimum_energy_GeV2 for s in ref.sectors)
    upper=min(eigvalsh(p.matrix,subset_by_index=[0,0])[0],floor-1e-8)
    if objective(upper)>1e-12:
        raise ValueError("no supported ground root below excluded poles")
    step=max(1.,sum(np.linalg.norm(v) for v in couplings))
    lower=upper-step
    for _ in range(32):
        if objective(lower)>0:break
        step*=2;lower=upper-step
    else:raise ArithmeticError("ground-root lower bracket not found")
    e=float(brentq(objective,lower,upper,xtol=5e-14,rtol=1e-14))
    _,vectors=eigh(effective(e),subset_by_index=[0,0])
    v=vectors[:,0]
    raw=[s.propagation.resolve(e,c@v[:,None]) for s,c in zip(ref.sectors,couplings)]
    norm=np.sqrt(np.vdot(v,v).real+sum(np.linalg.norm(q)**2 for q in raw))
    v=v/norm;raw=[q/norm for q in raw]
    rp=p.matrix@v-e*v
    rq2=0.
    for s,c,q in zip(ref.sectors,couplings,raw):
        rp+=c.conj().T@q[:,0]
        rq=s.propagation.apply(q)-e*q+c@v[:,None]
        rq2+=np.linalg.norm(rq)**2
    delta=(.12,.16)
    current=v.conj()@good_current_density(p,delta,charge="electromagnetic")@v
    for s,q in zip(ref.sectors,raw):
        current+=(q.conj().T@shell_density_between(s.source.space,s.source.space,q,delta,
                      b_GeV=p.width_GeV,charge="electromagnetic"))[0,0]
    diagonal=np.zeros_like(p.matrix)
    for s,c in zip(ref.sectors,couplings):
        space=s.source.space
        en=s.propagation.mass_offsets[space.coordinates[:,0]]+p.width_GeV**2*np.array(
            [4+sum(o) for o in space.orbitals])[space.coordinates[:,1]]
        diagonal+=c.conj().T@(c/(1.4-en[:,None]))
    sigma=effective(1.4)-p.matrix
    return dict(scope="bare canonical gV1+g^2V2, free Q5 only; no QQ/Q6/refit/physical claim",
       target=target,cutoff=cutoff,dimensions=[s.source.space.dimension for s in ref.sectors],
       build_seconds=built-start,total_seconds=time.monotonic()-start,
       lowest_free_Q_GeV2=float(floor),M2_GeV2=e,
       raw_parent_residual_GeV2=float(np.sqrt(np.linalg.norm(rp)**2+rq2)),
       excluded_probability=float(sum(np.linalg.norm(q)**2 for q in raw)),
       electromagnetic_density=[float(current.real),float(current.imag)],
       source_norms_by_power=[{str(n):float(np.linalg.norm(m)) for n,_,m in s.source.components}
                             for s in ref.sectors],
       sigma_at_1p4_norm_GeV2=float(np.linalg.norm(sigma)),
       wrong_diagonal_sigma_difference_GeV2=float(np.linalg.norm(sigma-diagonal)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs",nargs="+",type=int,default=[0,1,2,3])
    parser.add_argument("--target",choices=["PROTON","NEUTRON"],default="PROTON")
    args=parser.parse_args()
    for cutoff in args.cutoffs:
        print(json.dumps(run(cutoff,args.target),sort_keys=True),flush=True)
