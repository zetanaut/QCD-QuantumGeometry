"""Independent finite-model controls of the assembled H2 four-current.

All numbers below are explicitly identified exploratory test values, not
physical calibration. Each invocation evaluates one named member, so costly
resolution runs can be resumed independently. No quantum execution is claimed.
"""
import argparse
import json
from pathlib import Path
from time import monotonic
import numpy as np
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot,GaussianClusterH2
from deuteron_wigner.quantum.m2_gaussian_cluster_four_current import assemble_cluster_h2_four_current
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from deuteron_wigner.quantum.m2_total_isospin import H2_ISOSPIN_PRESCRIPTION
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between

CASES={
    "baseline":dict(),
    "Q5_shell1":dict(q5=1),
    "Q5_shell2":dict(q5=2),
    "Q6_shell1":dict(q6=1),
    "Q5_Q6_shell1":dict(q5=1,q6=1),
    "retained_shell1":dict(retained=1,q5=1,q6=1),
    # P+ stays fixed, so the longitudinal cell length changes with K2.
    # Both field caps (16 bosonic/15 fermionic doubled units) exceed K11:
    # they do not cut physical external legs, and Wick tails are paired.
    "longitudinal_K11":dict(k2=11),
    "field_tail16":dict(nb=16,nf=16),
    "Gaussian_width04":dict(width=.4),
    "Gaussian_width16":dict(width=1.6),
    "Gaussian_reference12":dict(width=1.2),
    "reference12_degree1":dict(reference_width=1.2,shape_degree=1),
    "reference12_degree2":dict(reference_width=1.2,shape_degree=2),
    "finite_quark_shift":dict(muq=.021),
    "finite_gluon_shift":dict(mug=.003),
    "neutron":dict(target="NEUTRON"),
}


def compare_matrices(reference, candidate):
    """Inspect aligned benchmark matrices; do not fit or change any operator.

    The caller must identify the same retained coordinates. The real scalar
    diagnostic for each raw current coefficient is an unconstrained lower
    bound on its shape mismatch, NOT a simultaneous coupling/mass fit.
    """
    if set(reference) != set(candidate):
        raise ValueError("the same complete set of benchmark matrices is required")
    result={}
    for key, value in reference.items():
        target=np.asarray(value,complex);source=np.asarray(candidate[key],complex)
        if target.shape!=source.shape or not np.all(np.isfinite(target)) or not np.all(np.isfinite(source)):
            raise ValueError("finite matrices in explicitly aligned coordinates required")
        norm=float(np.linalg.norm(target));difference=float(np.linalg.norm(source-target))
        entry=dict(bitwise_identical=bool(np.array_equal(source,target)),
                   difference_Frobenius=difference,
                   relative_difference=difference/norm if norm else None)
        if key.startswith("current_power_"):
            denominator=float(np.vdot(source,source).real)
            scale=float(np.vdot(source,target).real/denominator) if denominator else None
            residual=float(np.linalg.norm(scale*source-target)) if scale is not None else norm
            entry.update(best_real_scalar=scale,
                         best_scalar_residual_Frobenius=residual,
                         best_scalar_relative_residual=residual/norm if norm else None)
        result[key]=entry
    return result


def normalized_current_error_bound(reference, candidate):
    """Finite-reference error carried through BOTH positive H2 norm maps.

    For eta <= the four metric minimum eigenvalues, the inverse-square-root
    integral gives ||N^-1/2-M^-1/2||_2 <= ||N-M||_2/(2 eta^1.5).
    Telescoping Wf J Wi then bounds the stacked-component Frobenius error by
    delta_J/eta + ||J_ref||_F (delta_Nf+delta_Ni)/(2 eta^2).

    This is an analytic finite-matrix bound evaluated numerically, not an
    interval-arithmetic certificate, a cutoff-removal estimate, or a fit.
    Inputs must represent explicitly aligned retained coordinates.
    """
    matrices=[]
    for data in (reference,candidate):
        for name in ('N_out','N_in'):
            value=np.asarray(data[name],complex)
            if (value.ndim!=2 or value.shape[0]!=value.shape[1] or not value.size
                    or not np.all(np.isfinite(value))
                    or not np.allclose(value,value.conj().T,atol=2e-14,rtol=2e-14)):
                raise ValueError('finite Hermitian endpoint metrics required')
            matrices.append(value)
    if len({m.shape for m in matrices})!=1:
        raise ValueError('explicitly aligned retained coordinates required')
    eta=min(float(np.linalg.eigvalsh(m)[0]) for m in matrices)
    if eta<=0:
        raise ValueError('positive endpoint metrics required for inverse-square-root bound')
    currents=[np.asarray(data['raw_current'],complex) for data in (reference,candidate)]
    if any(j.shape!=(4,*matrices[0].shape) or not np.all(np.isfinite(j)) for j in currents):
        raise ValueError('four finite current components in aligned coordinates required')
    delta_current=float(np.linalg.norm(currents[1]-currents[0]))
    delta_norm=[float(np.linalg.norm(matrices[k+2]-matrices[k],ord=2)) for k in range(2)]
    raw_term=delta_current/eta
    norm_term=float(np.linalg.norm(currents[0]))*sum(delta_norm)/(2*eta**2)
    return dict(minimum_endpoint_metric_eigenvalue=eta,
        raw_current_difference_Frobenius=delta_current,
        endpoint_metric_difference_spectral_norm=delta_norm,
        raw_transport_bound_Frobenius=raw_term,
        two_norm_transport_bound_Frobenius=norm_term,
        normalized_current_error_bound_Frobenius=raw_term+norm_term)


def build_model(case):
    values=dict(q5=0,q6=0,nb=8,nf=8,width=.8,muq=.011,mug=-.007,target="PROTON")
    values.update(CASES[case])
    shape=(GaussianReferenceShape(base_width_GeV=values["width"],
        reference_width_GeV=values["reference_width"],degree=values["shape_degree"])
        if "shape_degree" in values else None)
    reg=GaussianClusterRegulator(width_GeV=values["width"],boson_modes=values["nb"],
        fermion_modes=values["nf"],reference_shape=shape)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=values["muq"],finite_gluon_mass2_per_gs2_GeV2=values["mug"])
    pilot=build_cluster_pilot(total_k2=values.get("k2",9),cutoff=values.get("retained",0),target=values["target"],jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        regulator=reg,total_pplus_GeV=1.3,wick_operators=wick)
    model=GaussianClusterH2(pilot,excluded_cutoff=values["q5"],six_cutoff=values["q6"])
    return model,values


def evaluate(case):
    start=monotonic();model,values=build_model(case);pilot=model.pilot
    f,i=model.snapshot(1.39),model.snapshot(1.47);delta=np.array([.12,.16])
    print(case+": constructing all four components",flush=True)
    current=assemble_cluster_h2_four_current(f,i,delta,charge="electromagnetic",
        loop_flavor_charges=(2/3,-1/3),prescription=PRESCRIPTION,quadrature_order=8,
        minus_context=SmoothMinusProjectorContext(mean_total_transverse_GeV=(0.,0.),prescription=SMOOTH_PRESCRIPTION),
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,isospin_prescription=H2_ISOSPIN_PRESCRIPTION)
    raw=current.current(normalized=False);normalized=current.current(normalized=True)
    rho=good_current_density_between(pilot,pilot,delta,charge="electromagnetic")
    expected=f.effective_hamiltonian@rho-rho@i.effective_hamiltonian
    expected+=(f.energy_GeV2-i.energy_GeV2)*(raw[0]-rho)
    ward=2*pilot.total_pplus_GeV*np.einsum("j,jab->ab",delta,raw[2:])-expected
    if np.linalg.norm(ward)>2e-11:raise ArithmeticError("full matrix Ward control failed")
    arrays={"raw_current":raw,"normalized_current":normalized,
        "H_out":f.effective_hamiltonian,"H_in":i.effective_hamiltonian,
        "N_out":f.metric,"N_in":i.metric,
        **{"current_power_"+str(n):v for n,v in current.coefficients.items()}}
    result=dict(case=case,parameters=values,total_k2=pilot.total_k2,retained_cutoff=pilot.cutoff,
        jz_twice=1,b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,
        total_pplus_GeV=1.3,energy_out_GeV2=1.39,energy_in_GeV2=1.47,
        transfer_GeV=delta.tolist(),component_order=["plus","minus","x","y"],
        current_normalization="Jmu/(2Pplus); order g4 selected before both norm maps",
        physical_leg_prescription=PRESCRIPTION,smooth_prescription=SMOOTH_PRESCRIPTION,
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,isospin_prescription=H2_ISOSPIN_PRESCRIPTION,
        raw_Q_dimensions={n:s.dimension for n,s in {**model.spaces5,**model.spaces6}.items()},
        Ward_Frobenius=float(np.linalg.norm(ward)),
        normalized_Ward_Frobenius=float(np.linalg.norm(f.whitener@ward@i.whitener)),
        raw_current_component_Frobenius=[float(np.linalg.norm(v)) for v in raw],
        normalized_current_component_Frobenius=[float(np.linalg.norm(v)) for v in normalized],
        coefficient_component_Frobenius={n:[float(np.linalg.norm(v)) for v in a] for n,a in current.coefficients.items()},
        min_norm_eigenvalue=[float(np.linalg.eigvalsh(s.metric)[0]) for s in (f,i)],
        isospin_wave_residual=max(max(r.values()) for r in current.isospin_residuals),
        finite_conditions_are_uncalibrated_test_values=True,
        continuum_convergence_established=False,physical=False,quantum_execution=False,
        seconds=monotonic()-start)
    return result,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=tuple(CASES),required=True)
    parser.add_argument("--output",type=Path,required=True,help="New JSON report; adjacent NPZ retains full matrices")
    args=parser.parse_args()
    matrices=args.output.with_suffix(".npz")
    if args.output.exists() or matrices.exists():
        parser.error("output already exists; select a new path to preserve previous evidence")
    result,arrays=evaluate(args.case)
    result["full_matrix_file"]=matrices.name
    np.savez_compressed(matrices,**arrays)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
