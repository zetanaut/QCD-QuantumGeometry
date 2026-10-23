"""Reproducible causal/matching qualification; no physical spectral data fit."""
import argparse
import json
from pathlib import Path
import numpy as np

from deuteron_wigner.quantum.causal_spectral_matching import (
    SpectralResponse, gaussian_response, pcac_pole, spectral_null_experiment, resolving_probes,
)


def evaluate():
    result = spectral_null_experiment()
    rows, selected = resolving_probes(result["mass_squared"], result["spacelike_q_squared"], .025)
    spacelike_rows = np.vstack([1/(np.geomspace(.1, 4., 7)[:, None]+result["mass_squared"]),
                               np.ones(9), result["mass_squared"]])
    spacelike_rows /= np.linalg.norm(spacelike_rows, axis=1)[:, None]
    for key, value in tuple(result.items()):
        if isinstance(value, np.ndarray): result[key] = value.tolist()
        if isinstance(value, complex): result[key] = [value.real, value.imag]
    result["resolved_bins"] = dict(selected_mass_squared=np.array(result["mass_squared"])[selected].tolist(),
        row_normalized_joint_condition=float(np.linalg.cond(rows)),
        row_normalized_spacelike_only_condition=float(np.linalg.cond(spacelike_rows)),
        joint_rank=int(np.linalg.matrix_rank(rows)),
        exact_residue_recovery_max_error=float(np.max(abs(np.linalg.solve(rows, rows@result["residues_a"])-result["residues_a"]))),
        scope="Known nine-pole family, noiseless controls, unit row norms; not physical-data identification or universal minimum input count")
    full = SpectralResponse(np.array([.25, 1.2]), np.array([[[.6]], [[.4]]]), "physical_response")
    ret = SpectralResponse(np.array([.25]), np.array([[[1.]]]), "physical_response")
    residual = full.minus(ret)
    misleading = SpectralResponse(np.array([.25, 1.]), np.array([[[-.1]], [[1.1]]]), "signed_remainder")
    times = np.array([.1, 1., 3., 6.])
    old = json.loads((Path(__file__).resolve().parents[1]/"docs/next_level/shared_chiral_dynamics_results.json").read_text())
    p = old["finite_mass_solution"]
    pole = pcac_pole(p["mass"], p["decay_constant"], p["pseudoscalar_residue"], old["inputs"]["current_mass_GeV"], .8)
    pole_ret = SpectralResponse(pole.mass_squared, .35*pole.residues, "physical_response")
    pole_res = pole.minus(pole_ret)
    m = np.diag([.16, .7, 1.8]); b = np.array([[.8, .1], [.2, .9], [.4, -.3]])
    n = np.zeros((3, 3)); n[0, 1] = 1
    c = np.array([[.3, .12, 0.], [.12, -.1, .05], [0., .05, .2]])
    equivalence, frozen, internal = [], [], []
    for z in [-.2, -.7, .8+.03j]:
        t = np.eye(3)+.7*z*n
        reference = gaussian_response(m, b, z)
        equivalence.append(np.linalg.norm(gaussian_response(m, b, z, t)-reference)/np.linalg.norm(reference))
        frozen.append(np.linalg.norm(gaussian_response(m, b, z, t, transform_sources=False)-reference)/np.linalg.norm(reference))
        inverse = np.linalg.inv(m-z*np.eye(3))
        internal.append(np.linalg.norm(np.linalg.inv(t.T@(m-z*np.eye(3))@t)-inverse)/np.linalg.norm(inverse))
    inverse = np.linalg.inv(m+.4*np.eye(3))
    first = -b.T@inverse@c@inverse@b
    other = -b.T@inverse@(c+.25*np.eye(3))@inverse@b
    return dict(scope="Finite causal color-singlet response/matching qualification, not a causal completion of the nonlinear quark functional or new bound-state solve.",
        units="s,z,Q2 GeV2, times GeV-1; normalized C GeV-2 and residues dimensionless; no physical data fit",
        positive_spectra_nonuniqueness=result,
        positivity_and_matching=dict(
            positive_full_reflection_min_eigenvalue=float(np.linalg.eigvalsh(full.reflection_matrix(times)).min()),
            positive_retained_reflection_min_eigenvalue=float(np.linalg.eigvalsh(ret.reflection_matrix(times)).min()),
            residual_reflection_min_eigenvalue=float(np.linalg.eigvalsh(residual.reflection_matrix(times)).min()),
            residual_zero_UV_moment=float(abs(residual.moment(0)[0, 0])),
            residual_Euclidean_Q2_half=float(residual.euclidean(.5)[0, 0].real),
            residual_retarded_negative_time_max=float(np.max(abs(residual.time_response([-1., -.1])))),
            positive_spacelike_but_invalid_reflection_min=float(np.linalg.eigvalsh(misleading.reflection_matrix(times)).min()),
            recombination_max=float(np.max(abs(full.analytic(.2+.1j)-ret.analytic(.2+.1j)-residual.analytic(.2+.1j))))),
        supplied_pion_pole=dict(source="Preserved shared_chiral_dynamics_results.json finite-mass solution; transported, NOT generated in the new functional",
            source_scale_GeV=.8, retained_fraction=.35, pion_mass_squared_GeV2=p["mass_squared"],
            normalized_residue_matrix=pole.residues[0].real.tolist(),
            PCAC_ratio=2*.004/.8, residual_residue_error=float(np.max(abs(pole_res.moment(0)-.65*pole.residues[0])))),
        field_coordinates=dict(max_response_relative_error=float(max(equivalence)),
            frozen_sources_relative_errors=[float(v) for v in frozen],
            internal_propagator_relative_changes=[float(v) for v in internal],
            same_two_point_but_different_scalar_insertion_relative=float(np.linalg.norm(other-first)/np.linalg.norm(first)),
            warning="Field-congruence differences are redundant only with every source/measure partner. Different physical scalar couplings at h=0 are not fixed by the two-point spectrum."),
        decision="Use causal matrix responses and independent scalar/mixed/transverse source matching to constrain ONE shared action. Do not positive-clip residuals, uniquely infer timelike dynamics from sparse spacelike data, or fit redundant off-shell coordinates. Next specify the smallest common-action realization and actual scheme-qualified matching targets; no BLFQ/H2 or spectral-fitting campaign.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(evaluate(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
