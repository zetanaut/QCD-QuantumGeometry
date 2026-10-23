"""Canonical IF/IG pair-source norms and coherent free-Q return powers."""
import json
import numpy as np
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import (
    build_instantaneous_pair_source,build_adjoint_pair_source,build_combined_pair_source,
)
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source


def main():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    a,b,c=[f(p,out_cutoff=1) for f in (build_instantaneous_pair_source,
                                      build_adjoint_pair_source,build_combined_pair_source)]
    cubic=build_excluded_source(p,"qqqgg",out_cutoff=1)
    np.testing.assert_array_equal(c.space.coordinates,cubic.space.coordinates)
    d=1.4-c.space.free_energies(.4,.35,0.)
    v,w=c.matrix_per_gs2,cubic.matrix_per_gs
    print(json.dumps(dict(scope="canonical pair-source operator norms, not physical masses",
        IF_IG_total_norms_GeV2=[float(np.linalg.norm(s.matrix_per_gs2)) for s in (a,b,c)],
        interference_GeV4=float(2*np.vdot(a.matrix_per_gs2,b.matrix_per_gs2).real),
        Sigma_g4_coefficient_norm_GeV2=float(np.linalg.norm(v.conj().T@(v/d[:,None]))),
        Sigma_g3_cross_block_coefficient_norm_GeV2=float(np.linalg.norm(v.conj().T@(w/d[:,None])))),
        indent=2))


if __name__=="__main__":
    main()
