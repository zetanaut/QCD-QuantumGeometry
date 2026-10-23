"""Reproduce the named BARE finite H2/excluded-Fock diagnostic, not a fit.

Run from the repository with its installed Python environment. The fixed
numbers below specify only this published exploratory example. No production
Hamiltonian/current is selected. --quantum executes the simulator optimizer.
"""
import argparse

import numpy as np
from scipy import linalg
from scipy.optimize import brentq

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot,baryon_density
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference


def run(*,quantum=False):
    for cutoff in (0,1):
        p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,
            b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
        sources=[build_excluded_source(p,s,out_cutoff=cutoff) for s in ("qqqgg","qqqq-qbar")]
        ref=ExcludedFockReference(p,sources)
        state=ref.reduction.lowest_state()
        print("Q cutoff",cutoff,"parent dimension",len(ref.matrix),flush=True)
        for s in ref.sectors:
            print(s.source.space.sector,"packed/reachable",s.source.space.dimension,len(s.energies_GeV2),
                  "source reconstruction error",s.reconstruction_residual_per_gs,flush=True)
        print("M2",state.energy_GeV2,"independent M2",linalg.eigvalsh(ref.matrix)[0],
              "Q probability",state.excluded_probability,"full residual/probability",
              ref.full_reconstructed_residual(state.energy_GeV2,state.retained),flush=True)
        for charge in ("baryon","electromagnetic","up_number","down_number"):
            rho=ref.density((.12,.16),charge=charge)
            print(charge,"matched density",np.vdot(state.reconstructed,rho@state.reconstructed),flush=True)
        bare=baryon_density(p,(.12,.16))
        print("bare normalized-P baryon density",np.vdot(state.retained,bare@state.retained)/
              np.vdot(state.retained,state.retained),flush=True)
        if cutoff!=1:
            continue
        n=len(ref.matrix); nq=p.bases[0].shape[1]; np_=len(p.matrix)
        a=np.zeros(n); a[:nq]=1
        b=np.zeros(n); b[nq:np_]=1
        anchor=state.energy_GeV2
        rho=ref.density((.12,.16),charge="electromagnetic")
        for shift in (-.5,-.25,0.,.25,.5):
            h=lambda ct:ref.matrix+np.diag(shift*b+ct*a)
            ct=brentq(lambda ct:linalg.eigvalsh(h(ct),subset_by_index=[0,0])[0]-anchor,-1.,1.,xtol=1e-14)
            e,u=linalg.eigh(h(ct),subset_by_index=[0,0]); v=u[:,0]
            print("unselected sector-shift witness",shift,"compensating qqq shift",ct,
                  "M2",e[0],"EM density",np.vdot(v,rho@v),flush=True)
        if quantum:
            from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
            model=ref.reduction.linearized_model(1.4)
            c=SymmetryGraphCircuit(model.matrix,root=int(np.argmin(model.matrix.diagonal().real)),support_threshold=1e-12)
            initial=c.uniform_parameters()
            initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(c.edges))
            fit=c.fit(initial,max_iterations=2200,gradient_tolerance=1e-8)
            e,v=linalg.eigh(model.matrix)
            psi=np.asarray(c.state(fit.x))
            print("quantum",fit.success,str(fit.message),"iterations",fit.nit,"energy",float(fit.fun),
                  "error",float(fit.fun-e[0]),"overlap2",abs(np.vdot(v[:,0],psi[:c.dimension]))**2,
                  "gradient norm",np.linalg.norm(fit.jac),"state residual",
                  np.linalg.norm(model.matrix@psi[:c.dimension]-fit.fun*psi[:c.dimension]),
                  "padding norm",np.linalg.norm(psi[c.dimension:]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantum",action="store_true")
    run(**vars(parser.parse_args()))
