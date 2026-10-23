"""Full represented canonical quartic QQ source action, one W insertion only."""
import argparse
import json
import numpy as np

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction,CanonicalQQInsertion
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cutoff",type=int,choices=(0,1),default=1)
    args=parser.parse_args()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=args.cutoff)
    obj=CanonicalQQInsertion(ref,action=CanonicalQQSourceAction(ref))
    e=1.4;delta=(.12,.16)
    a,_=obj.order_waves(e,1)
    parts=obj.action.component_returns(a)
    h={n:obj.return_power_coefficient(n,e) for n in (4,5,6)}
    norms={n:obj.return_power_coefficient(n,e,norm=True) for n in (4,5,6)}
    currents={n:obj.current_power_coefficient(n,e,e,delta,charge="electromagnetic") for n in (4,5,6)}
    errors={n:float(np.linalg.norm(obj.current_power_coefficient(n,e,e,(0.,0.),charge="electromagnetic")-norms[n]))
            for n in (4,5,6)}
    root=ref.reduction.lowest_state()
    shifts={n:float(p.coupling**n*np.vdot(root.retained,
        obj.return_power_coefficient(n,root.energy_GeV2)@root.retained).real) for n in (4,5,6)}
    projected=np.zeros_like(p.matrix)
    for s in ref.sectors:
        name=s.source.space.sector;a1,a3=obj.wave_coefficients(name,e)
        a3=s.lift(s.project(a3))
        r1=intermediate_density_action(s.source.space,a1,delta,b_GeV=.4,charge="electromagnetic")
        r3=intermediate_density_action(s.source.space,a3,delta,b_GeV=.4,charge="electromagnetic")
        projected+=a3.conj().T@r1+a1.conj().T@r3
    print(json.dumps(dict(
        scope=obj.action.scope+"; one W insertion, not full g^4/g^6 QCD or a nonperturbative QQ solve",
        Q_cutoff=args.cutoff,current_auxiliary_rank=obj.action.pair_current.auxiliary.basis.shape[1],
        component_H4_norms_GeV2={k:float(np.linalg.norm(v)) for k,v in parts.items()},
        H_coefficients_GeV2={n:float(np.linalg.norm(v)) for n,v in h.items()},
        N_coefficients={n:float(np.linalg.norm(v)) for n,v in norms.items()},
        electromagnetic_current_coefficients={n:float(np.linalg.norm(v)) for n,v in currents.items()},
        charge_norm_errors=errors,
        H_single_insertion_weighted_norm_GeV2=float(np.linalg.norm(sum(p.coupling**n*h[n] for n in h))),
        first_order_mass_squared_responses_GeV2=shifts,
        total_first_order_mass_squared_response_GeV2=sum(shifts.values()),
        leading_current_coefficient_lost_by_source_projection=float(np.linalg.norm(currents[4]-projected)),
        leading_raw_wave_leakage=obj.free_source_leakage(e),
        preceding_free_Q_parent_unchanged=True,physical=False),indent=2))


if __name__=="__main__":
    main()

