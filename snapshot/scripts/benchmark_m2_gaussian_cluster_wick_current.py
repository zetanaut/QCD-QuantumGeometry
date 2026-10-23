"""Uncalibrated direct Wick-current sensitivity, NOT complete loop/current matching."""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot,GaussianClusterH2
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import cluster_wick_minus_h2_coefficients


def run(target,width,q5,q6,nb=8,nf=8):
    start = monotonic()
    reg = GaussianClusterRegulator(width_GeV=width,boson_modes=nb,fermion_modes=nf)
    wick = GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    p = build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,regulator=reg,
        total_pplus_GeV=1.3,wick_operators=wick)
    h2 = GaussianClusterH2(p,excluded_cutoff=q5,six_cutoff=q6)
    f,i = h2.snapshot(1.39),h2.snapshot(1.47)
    pieces,coefficients = {},{}
    for name in ("canonical","regulator","ordered","subtraction","finite","total"):
        c = cluster_wick_minus_h2_coefficients(f,i,(.12,.16),charge="electromagnetic",
            loop_flavor_charges=(2/3,-1/3),prescription=PRESCRIPTION,component=name)
        coefficients[name] = c
        pieces[name] = f.whitener@sum(.3**n*a for n,a in c.items())@i.whitener
    record = dict(target=target,Lambda_GeV=width,Q5_cutoff=q5,Q6_cutoff=q6,
        boson_field_modes=nb,fermion_field_modes=nf,retained_dimension=len(p.matrix),
        raw_Q_dimensions={n:s.dimension for n,s in {**h2.spaces5,**h2.spaces6}.items()},
        normalized_direct_minus_Frobenius={n:float(np.linalg.norm(a)) for n,a in pieces.items()},
        unnormalized_ordered_power_contributions={str(n):float(np.linalg.norm(.3**n*a))
                                                for n,a in coefficients["ordered"].items()},
        H2_norm_minimum_eigenvalues=[float(np.min(np.linalg.eigvalsh(s.metric))) for s in (f,i)],
        subtraction_residual=float(np.linalg.norm(pieces["ordered"]+pieces["subtraction"])),
        finite_pairing_residual=float(np.linalg.norm(pieces["total"]-pieces["finite"])),
        seconds=monotonic()-start)
    return record,pieces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new result path to preserve existing research output")
    cases = [("PROTON",.8,1,1,8,8),("PROTON",.4,1,1,8,8),
             ("PROTON",1.6,1,1,8,8),("PROTON",.8,0,0,8,8),
             ("PROTON",.8,2,1,8,8),("PROTON",.8,1,1,1,2),
             ("NEUTRON",.8,1,1,8,8)]
    records = []
    for case in cases:
        record,pieces = run(*case)
        if not records:
            baseline = pieces
        if case[0]=="PROTON":
            record["matrix_difference_from_proton_Lambda08_Q1_Frobenius"] = {
                n:float(np.linalg.norm(a-baseline[n])) for n,a in pieces.items()}
        records.append(record)
        print(json.dumps(record),flush=True)
    result = dict(scope="Only direct smooth common-kernel Wick-current contributions through H2 g4",
        prescription=PRESCRIPTION,Delta_plus_GeV=0.,transfer_GeV=[.12,.16],
        total_k2=9,retained_cutoff=0,jz_twice=1,b_GeV=.4,quark_mass_GeV=.35,
        gluon_mass_GeV=0.,coupling=.3,total_pplus_GeV=1.3,
        energy_out_GeV2=1.39,energy_in_GeV2=1.47,quark_flavors=2,
        external_charge="electromagnetic",loop_flavor_charges=[2/3,-1/3],
        normal_order_subtraction=True,finite_quark_mass2_per_gs2_GeV2=.011,
        finite_gluon_mass2_per_gs2_GeV2=-.007,
        finite_conditions_are_uncalibrated_test_values=True,
        all_quantities_are_full_retained_matrix_norms_not_form_factors=True,
        smooth_Wick_subtraction_paired=True,longitudinal_envelope_gauged=False,
        complete_field_current=False,complete_loop_current_matching=False,
        cutoff_convergence_established=False,physical_renormalization=False,
        quantum_execution=False,results=records)
    with args.output.open("x") as output:
        json.dump(result,output,indent=2)
        output.write("\n")


if __name__=="__main__":
    main()
