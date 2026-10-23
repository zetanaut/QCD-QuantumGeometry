"""Complete represented normal-ordered current-square QQ, single-insertion diagnostic."""
import argparse
import json
import numpy as np
from scipy import sparse

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_excluded_fermion_scattering import fermion_scattering_per_gs2
from deuteron_wigner.quantum.m2_excluded_current_scattering import current_scattering_per_gs2
from deuteron_wigner.quantum.m2_excluded_pair_current import (
    build_pair_current_factorization,CoupledPairCurrentInsertion,
)
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cutoff",type=int,default=1,choices=(0,1))
    args=parser.parse_args()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=args.cutoff)
    factor=build_pair_current_factorization(ref)
    operators={}
    separate={}
    for name,s in factor.spaces.items():
        w=fermion_scattering_per_gs2(s,b_GeV=.4)
        separate[name+"_fermion_scattering"]=float(sparse.linalg.norm(w))
        if name=="qqqgg":
            for c in ("qg","gg"):
                extra=current_scattering_per_gs2(s,b_GeV=.4,channel=c)
                separate[c+"_scattering"]=float(sparse.linalg.norm(extra))
                w=w+extra
        # Sea annihilation is in B D B†, never counted a second time here.
        operators[name]=(s,w)
    obj=CoupledPairCurrentInsertion(ref,factorization=factor,sector_operators=operators,
                                   label="full represented normal-ordered instantaneous-gluon current square")
    e=1.4;delta=(.12,.16)
    h,n=obj.hamiltonian_coefficient(e),obj.norm_coefficient(e)
    rho=obj.current_coefficient(e,e,delta,charge="electromagnetic")
    projected=np.zeros_like(h);z={}
    for s in ref.sectors:
        name=s.source.space.sector
        a1,a3=obj.wave_coefficients(name,e)
        z[name]=factor.matrices[name].conj().T@a1
        a3=s.lift(s.project(a3))
        r1=intermediate_density_action(s.source.space,a1,delta,b_GeV=.4,charge="electromagnetic")
        r3=intermediate_density_action(s.source.space,a3,delta,b_GeV=.4,charge="electromagnetic")
        projected+=a3.conj().T@r1+a1.conj().T@r3
    zd=sum(z.values());d=factor.auxiliary.weight
    interference=zd.conj().T@d@zd-sum(a.conj().T@d@a for a in z.values())
    root=ref.reduction.lowest_state()
    shift=p.coupling**4*np.vdot(root.retained,obj.hamiltonian_coefficient(root.energy_GeV2)@root.retained).real
    print(json.dumps(dict(
        scope=obj.label+"; I=1/2 source sector; no IF/A4, six-body or physical matching",
        Q_cutoff=args.cutoff,current_auxiliary_rank=factor.auxiliary.basis.shape[1],
        auxiliary_is_physical_helicity_zero_gluon=False,
        scattering_component_norms_per_gs2_GeV2=separate,
        pair_current_B_norms_GeV={name:float(np.linalg.norm(b)) for name,b in factor.matrices.items()},
        pair_cross_channel_H4_norm_GeV2=float(np.linalg.norm(interference)),
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

