"""Explicit uncalibrated H2 field/radial controls, not renormalization evidence."""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot, GaussianClusterH2


def run(target, width, q5, q6, nb=8, nf=8):
    start=monotonic()
    reg=GaussianClusterRegulator(width_GeV=width,boson_modes=nb,fermion_modes=nf)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    p=build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,regulator=reg,
        total_pplus_GeV=1.3,wick_operators=wick)
    model=GaussianClusterH2(p,excluded_cutoff=q5,six_cutoff=q6)
    snapshot=model.snapshot(1.4)
    h,n=snapshot.effective_hamiltonian,snapshot.metric
    result=dict(target=target,Lambda_GeV=width,Q5_cutoff=q5,Q6_cutoff=q6,
        boson_field_modes=nb,fermion_field_modes=nf,
        retained_dimension=len(p.matrix),
        raw_Q_dimensions={k:s.dimension for k,s in {**model.spaces5,**model.spaces6}.items()},
        self_energy_Frobenius_GeV2=float(np.linalg.norm(h-p.matrix)),
        norm_correction_Frobenius=float(np.linalg.norm(n-np.eye(len(n)))),
        minimum_norm_eigenvalue=float(np.min(np.linalg.eigvalsh(n))),
        H_antihermitian_Frobenius_GeV2=float(np.linalg.norm(h-h.conj().T)),
        norm_antihermitian_Frobenius=float(np.linalg.norm(n-n.conj().T)),
        seconds=monotonic()-start)
    return result,h,n


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cases=[("PROTON",.4,1,1,8,8),("PROTON",.8,1,1,8,8),
           ("PROTON",1.6,1,1,8,8),("PROTON",.8,0,0,8,8),
           ("PROTON",.8,2,1,8,8),("PROTON",.8,1,1,1,2),
           ("NEUTRON",.8,1,1,8,8)]
    records=[]
    for case in cases:
        record,h,n=run(*case)
        if case==cases[1]:
            base_h,base_n=h,n
        if len(records)>=1 and case[0]=="PROTON":
            record["H_difference_from_proton_Lambda08_Q1_Frobenius_GeV2"]=float(np.linalg.norm(h-base_h))
            record["N_difference_from_proton_Lambda08_Q1_Frobenius"]=float(np.linalg.norm(n-base_n))
        records.append(record)
        print(json.dumps(record),flush=True)
    result=dict(scope="common-cluster H2 Hamiltonian/norm sensitivity; no complete field-current evaluation",
        total_k2=9,retained_cutoff=0,jz_twice=1,b_GeV=.4,quark_mass_GeV=.35,
        gluon_mass_GeV=0.,coupling=.3,total_pplus_GeV=1.3,energy_GeV2=1.4,
        quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007,
        finite_conditions_are_uncalibrated_test_values=True,
        all_quantities_are_full_retained_matrix_norms_not_form_factors=True,
        cutoff_convergence_established=False,physical_renormalization=False,
        complete_field_current=False,quantum_execution=False,results=records)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
