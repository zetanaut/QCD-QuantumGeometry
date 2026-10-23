"""Canonical fermion QQ insertion, raw wave leakage and matched-current check."""
import argparse
import json
import numpy as np
from scipy import sparse
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_excluded_fermion_scattering import fermion_scattering_per_gs2
from deuteron_wigner.quantum.m2_excluded_annihilation import annihilation_per_gs2
from deuteron_wigner.quantum.m2_excluded_interaction_insertion import ExcludedInteractionInsertion
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scattering-only",action="store_true")
    args=parser.parse_args()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=1)
    operators={s.source.space.sector:(s.source.space,fermion_scattering_per_gs2(s.source.space,b_GeV=.4))
               for s in ref.sectors}
    if not args.scattering_only:
        space,w=operators["qqqq-qbar"]
        operators["qqqq-qbar"]=(space,w+annihilation_per_gs2(space,b_GeV=.4))
    label="canonical fermion t-channel only" if args.scattering_only else "canonical fermion scattering + annihilation"
    obj=ExcludedInteractionInsertion(ref,sector_operators=operators,label=label)
    e=1.4;delta=(.12,.16)
    h,n=obj.hamiltonian_coefficient(e),obj.norm_coefficient(e)
    rho=obj.current_coefficient(e,e,delta,charge="electromagnetic")
    projected=np.zeros_like(h)
    for s in ref.sectors:
        a1,a3=obj.wave_coefficients(s.source.space.sector,e)
        a3=s.lift(s.project(a3))
        r1=intermediate_density_action(s.source.space,a1,delta,b_GeV=.4,charge="electromagnetic")
        r3=intermediate_density_action(s.source.space,a3,delta,b_GeV=.4,charge="electromagnetic")
        projected+=a3.conj().T@r1+a1.conj().T@r3
    root=ref.reduction.lowest_state()
    # First-order energy response to a g^4 effective insertion uses v†Nv=1,
    # not a Euclidean-renormalized retained vector.
    shift=p.coupling**4*np.vdot(root.retained,
        obj.hamiltonian_coefficient(root.energy_GeV2)@root.retained).real
    print(json.dumps(dict(
        scope="single "+label+" QQ insertion, NOT full g^4",
        QQ_raw_dimensions={s.source.space.sector:s.source.space.dimension for s in ref.sectors},
        QQ_frobenius_norms_per_gs2_GeV2={name:float(sparse.linalg.norm(w)) for name,w in obj.operators.items()},
        H4_coefficient_norm_GeV2=float(np.linalg.norm(h)),
        N4_coefficient_norm=float(np.linalg.norm(n)),
        electromagnetic_rho4_norm=float(np.linalg.norm(rho)),
        discarded_current_coefficient_norm=float(np.linalg.norm(rho-projected)),
        discarded_current_norm_at_gs_0p3=float(p.coupling**4*np.linalg.norm(rho-projected)),
        raw_single_W_wave_leakage=obj.free_source_leakage(e),
        norm_charge_error=float(np.linalg.norm(obj.current_coefficient(e,e,(0.,0.),charge="electromagnetic")-n)),
        first_order_mass_squared_response_GeV2=float(shift),
        source_parent_is_modified=False,physical=False),indent=2))


if __name__=="__main__":
    main()
