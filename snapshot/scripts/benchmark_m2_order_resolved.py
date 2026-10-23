"""Both canonical five-body source orders; exact finite H2 matching, not a fit."""
import argparse
import json
import numpy as np
from scipy import linalg

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_spin_resolved_current import (
    parity_aligned_partner,matched_good_current_between,good_form_factor_diagnostic,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--spin",action="store_true")
    args=parser.parse_args()
    rows=[]
    for target in (("PROTON","NEUTRON") if args.spin else ("PROTON",)):
        for cutoff in (0,1):
            p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=1,b_GeV=.4,
                                 quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
            ref=build_order_resolved_reference(p,out_cutoff=cutoff)
            old=ExcludedFockReference(p,[build_excluded_source(p,s,out_cutoff=cutoff)
                                          for s in ("qqqgg","qqqq-qbar")])
            root=ref.reduction.lowest_state();e=root.energy_GeV2
            residual,probability=ref.full_reconstructed_residual(e,root.retained)
            row=dict(target=target,Q_cutoff=cutoff,parent_dimension=len(ref.matrix),
                     cubic_parent_dimension=len(old.matrix),M2_GeV2=e,
                     independent_eigenvalue_error_GeV2=abs(e-linalg.eigvalsh(ref.matrix)[0]),
                     excluded_probability=probability,uncompressed_residual_GeV2=residual,
                     mass_shift_from_cubic_GeV2=e-old.reduction.lowest_state().energy_GeV2,
                     return_order_norms_GeV2={str(n):float(np.linalg.norm(ref.return_coefficient(n,1.4)))
                                            for n in (2,3,4)})
            if args.spin:
                down=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=-1,b_GeV=.4,
                                 quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
                lower=build_order_resolved_reference(down,out_cutoff=cutoff)
                refs=[ref,lower];states=[root.retained,
                    parity_aligned_partner(down,p,root.retained,intrinsic_parity=1)]
                matrix=np.array([[vf.conj()@matched_good_current_between(
                    rf,ri,e,e,(.12,.16),charge="electromagnetic")@vi for ri,vi in zip(refs,states)]
                    for rf,vf in zip(refs,states)])
                ff=good_form_factor_diagnostic(matrix,(.12,.16),mass_GeV=np.sqrt(e),total_pplus=3.)
                row.update(F1=[ff.dirac.real,ff.dirac.imag],F2=[ff.pauli.real,ff.pauli.imag],
                           current_structure_residual=float(np.linalg.norm(ff.residual_matrix)))
            rows.append(row)
    print(json.dumps(dict(scope="canonical cubic/quartic five-body sources with free Q; not full g^4 QCD",
                          transfer_GeV=[.12,.16],rows=rows),indent=2))


if __name__=="__main__":
    main()
