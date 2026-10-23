"""Complete represented leading six-body excursion, not full g^4 QCD."""
import json
import numpy as np
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action
from deuteron_wigner.quantum.m2_six_body_space import density_action


def main():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=1)
    obj=CanonicalSixBodyExcursion(ref,out_cutoff=1)
    e=1.4;delta=(.12,.16);a=obj.waves(e)
    h=sum(a["six_source"][n].conj().T@v for n,v in a["six_leading"].items())
    prob=sum(v.conj().T@v for v in a["six_leading"].values())
    back=sum(a["five_leading"][n].conj().T@v for n,v in a["five_backflow"].items())
    norm=prob+back+back.conj().T
    def current(transfer):
        result=sum(a["six_leading"][n].conj().T@density_action(
            s,a["six_leading"][n],transfer,b_GeV=.4,charge="electromagnetic") for n,s in obj.spaces6.items())
        for n,s in obj.spaces5.items():
            r=intermediate_density_action(s,np.column_stack((a["five_leading"][n],a["five_backflow"][n])),
                transfer,b_GeV=.4,charge="electromagnetic")
            result+=a["five_backflow"][n].conj().T@r[:,:len(p.matrix)]+a["five_leading"][n].conj().T@r[:,len(p.matrix):]
        return result
    parts=obj.return_components(e)
    root=ref.reduction.lowest_state()
    shift=p.coupling**4*np.vdot(root.retained,obj.hamiltonian_coefficient(root.energy_GeV2)@root.retained).real
    print(json.dumps(dict(scope="leading canonical six-body excursion with free Q5/Q6; not full g^4 QCD",
        direct_component_norms_GeV2=obj.direct_component_norms,
        coherent_direct_source_norms_GeV2={n:float(np.linalg.norm(v)) for n,v in obj.W.items()},
        return_component_norms_GeV2={n:float(np.linalg.norm(v)) for n,v in parts.items()},
        H4_norm_GeV2=float(np.linalg.norm(h)),N4_norm=float(np.linalg.norm(norm)),
        electromagnetic_rho4_norm=float(np.linalg.norm(current(delta))),
        charge_norm_error=float(np.linalg.norm(current((0,0))-norm)),
        six_probability_coefficient_norm=float(np.linalg.norm(prob)),
        backflow_norm_coefficient=float(np.linalg.norm(back+back.conj().T)),
        first_order_mass_squared_response_GeV2=float(shift),
        direct_iterated_interference_included=True,preceding_free_Q_parent_unchanged=True,
        physical=False),indent=2))


if __name__=="__main__":main()
