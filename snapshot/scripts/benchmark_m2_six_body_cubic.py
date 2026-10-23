"""Actual Q5 -> Q6 -> Q5 cubic-path diagnostic; direct quartic W6P absent."""
import argparse,json
import numpy as np
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action
from deuteron_wigner.quantum.m2_six_body_cubic import SixBodyCubicAction
from deuteron_wigner.quantum.m2_six_body_space import density_action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cutoff",type=int,choices=(0,1),default=1)
    args=parser.parse_args()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=args.cutoff)
    spaces={s.source.space.sector:s.source.space for s in ref.sectors}
    v={s.source.space.sector:s.lift(s.coupling_by_power[1]) for s in ref.sectors}
    action=SixBodyCubicAction(spaces,out_cutoff=args.cutoff,b_GeV=.4,mass_GeV=.35)
    e=1.4;delta=(.12,.16)
    e5={n:s.free_energies(.4,.35,0.) for n,s in spaces.items()}
    e6={n:s.free_energies(.4,.35,0.) for n,s in action.outgoing.items()}
    a1={n:vv/(e-e5[n][:,None]) for n,vv in v.items()}
    f=action.apply(a1)
    a6={n:ff/(e-e6[n][:,None]) for n,ff in f.items()}
    back=action.adjoint(a6)
    b5={n:bb/(e-e5[n][:,None]) for n,bb in back.items()}
    h=sum(f[n].conj().T@a6[n] for n in f)
    probability=sum(a.conj().T@a for a in a6.values())
    cross=sum(a1[n].conj().T@b5[n] for n in a1)
    norm=probability+cross+cross.conj().T
    rho=sum(a6[n].conj().T@density_action(s,a6[n],delta,b_GeV=.4,charge="electromagnetic")
            for n,s in action.outgoing.items())
    charge=sum(a6[n].conj().T@density_action(s,a6[n],(0,0),b_GeV=.4,charge="electromagnetic")
            for n,s in action.outgoing.items())
    for n,s in spaces.items():
        joined=intermediate_density_action(s,np.column_stack((a1[n],b5[n])),delta,
            b_GeV=.4,charge="electromagnetic")
        rho+=b5[n].conj().T@joined[:,:len(p.matrix)]+a1[n].conj().T@joined[:,len(p.matrix):]
        joined0=intermediate_density_action(s,np.column_stack((a1[n],b5[n])),(0,0),
            b_GeV=.4,charge="electromagnetic")
        charge+=b5[n].conj().T@joined0[:,:len(p.matrix)]+a1[n].conj().T@joined0[:,len(p.matrix):]
    print(json.dumps(dict(
        scope="canonical cubic-path-only leading six-body diagnostic, W6P not yet supplied",
        K2=9,retained_cutoff=0,Q_cutoff=args.cutoff,
        raw_six_body_dimensions={n:s.dimension for n,s in action.outgoing.items()},
        singlet_multiplicities={n:s.ncolor for n,s in action.outgoing.items()},
        H4_norm_GeV2=float(np.linalg.norm(h)),N4_norm=float(np.linalg.norm(norm)),
        six_body_probability_coefficient_norm=float(np.linalg.norm(probability)),
        five_body_backflow_norm_coefficient=float(np.linalg.norm(cross+cross.conj().T)),
        electromagnetic_rho4_norm=float(np.linalg.norm(rho)),
        charge_norm_error=float(np.linalg.norm(charge-norm)),
        coherent_cubic_source_norms={n:float(np.linalg.norm(ff)) for n,ff in f.items()},
        direct_quartic_Q6P_included=False,complete_g4_six_body=False,
        preceding_parent_unchanged=True,physical=False),indent=2))


if __name__=="__main__":main()
