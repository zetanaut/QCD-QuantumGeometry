"""Compare ordered and subtracted Wick/H2 operators on complete Q5/Q6 shells.

Finite-model subtraction study only. The Gaussian coincident trace does NOT
regulate the pre-existing propagating vertices; this is not UV convergence,
a physical mass/current fit or a quantum-device benchmark.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from deuteron_wigner.quantum.m2_canonical_quark_self import gaussian_coincident_trace
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import (
    FactorizedCanonicalQQSourceAction, ShellCanonicalQQInsertion,
)
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2
from deuteron_wigner.quantum.m2_wick_one_body import WickOneBody
from deuteron_wigner.quantum.m2_wick_h2_matching import (
    WickMatchedFourthOrderH2, direct_minus_coefficients_between, minus_current_between,
)


def run(target, cutoff, current_kind):
    start = time.monotonic()
    p = build_coupled_pilot(total_k2=9, cutoff=0, target=target, jz_twice=1,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    ref = build_shell_fock_reference(p, out_cutoff=cutoff)
    qq = ShellCanonicalQQInsertion(ref, action=FactorizedCanonicalQQSourceAction(ref))
    base = FourthOrderH2(ref, QQ_insertion=qq,
        six_body=CanonicalSixBodyExcursion(ref, out_cutoff=cutoff))
    original = base.snapshot(1.4)
    print("Common full-shell canonical H2 snapshot constructed", flush=True)
    args = dict(total_pplus_GeV=1.3, charge="electromagnetic", loop_flavor_charges=[2/3, -1/3])
    delta = (.12, .16)
    records, comparisons = [], {}
    for label, subtract, muq, mug in (
        ("ordered", False, 0., 0.), ("normal_order_zero", True, 0., 0.),
        ("normal_order_illustrative_finite", True, .011, -.007),
    ):
        # Independent trace/longitudinal controls plus two joint variations.
        for modes, width in ((8, .4), (12, .4), (16, .4), (8, .8), (8, 1.6),
                             (12, .8), (16, 1.6)):
            controls = WickOneBody(boson_modes=modes, fermion_modes=modes, quark_flavors=2,
                coincident_trace_GeV2=gaussian_coincident_trace(width),
                normal_order_subtraction=subtract, finite_quark_mass2_per_gs2_GeV2=muq,
                finite_gluon_mass2_per_gs2_GeV2=mug)
            m = WickMatchedFourthOrderH2(base, operators=controls)
            s = m.match_snapshot(original)
            energy, vectors = eigh(s.matrix)
            direct = direct_minus_coefficients_between(s, s, delta, **args)
            direct_matrix = sum(.3**n*a for n, a in direct.items())
            w = s.canonical_snapshot.whitener
            direct_matrix = w@direct_matrix@w
            wave_residuals, backflow_norms = {}, {}
            for name, propagation in m.propagation.items():
                a1 = original.five_wave_coefficients[0][name]
                extra = s.canonical_snapshot.five_wave_coefficients[2][name]-original.five_wave_coefficients[2][name]
                residual = 1.4*extra-propagation.apply(extra)-m.qq[name][:, None]*a1
                wave_residuals[name] = float(np.linalg.norm(residual))
                backflow_norms[name] = float(np.linalg.norm(extra))
            full_current = None
            if current_kind == "full":
                value = minus_current_between(s, s, delta, **args, normalized=True)
                z = np.vdot(vectors[:, 0], value@vectors[:, 0])
                full_current = [float(z.real), float(z.imag)]
            reference = comparisons.setdefault(label, s.matrix.copy())
            rho0 = s.current((0., 0.), charge="electromagnetic")
            expected_charge = float(target == "PROTON")
            record = dict(member=label, boson_modes=modes, fermion_modes=modes,
                trace_width_GeV=width, coincident_trace_GeV2=controls.coincident_trace_GeV2,
                finite_quark_mass2_per_gs2_GeV2=muq, finite_gluon_mass2_per_gs2_GeV2=mug,
                M2_GeV2=float(energy[0]), state_residual_GeV2=float(np.linalg.norm(s.matrix@vectors[:, 0]-energy[0]*vectors[:, 0])),
                pp_coefficient_norm_GeV2=float(np.linalg.norm(m.pp)),
                matched_matrix_difference_to_bare=float(np.linalg.norm(s.matrix-original.matrix)),
                matched_matrix_difference_to_first_field_member=float(np.linalg.norm(s.matrix-reference)),
                metric_difference_to_bare=float(np.linalg.norm(s.metric-original.metric)),
                raw_extra_backflow_norms=backflow_norms, raw_Q_wave_equation_residuals=wave_residuals,
                direct_minus_current_order2_norm=float(np.linalg.norm(direct[2])),
                direct_minus_current_order4_norm=float(np.linalg.norm(direct[4])),
                normalized_direct_minus_matrix_norm=float(np.linalg.norm(direct_matrix)),
                full_matched_minus_ground_expectation=full_current,
                normalized_charge_error=float(np.linalg.norm(rho0-expected_charge*np.eye(len(s.matrix)))))
            if max(wave_residuals.values()) > 3e-12 or record["normalized_charge_error"] > 3e-12:
                raise ArithmeticError("paired finite-model identity failed")
            if subtract and record["matched_matrix_difference_to_first_field_member"] > 2e-14:
                raise ArithmeticError("normal-order finite model depends on the cancelled field trace")
            if label == "normal_order_zero" and record["matched_matrix_difference_to_bare"] > 2e-14:
                raise ArithmeticError("zero-finite subtraction changed the preceding canonical model")
            records.append(record)
            print(label, modes, width, "M2", record["M2_GeV2"], flush=True)
    return dict(scope=__doc__, target=target, current_kind=current_kind,
        retained_cutoff=0, excluded_cutoff=cutoff, retained_dimension=len(p.matrix),
        raw_Q5_dimensions={n: s.dimension for n, s in base.spaces5.items()},
        raw_Q6_dimensions={n: s.dimension for n, s in base.spaces6.items()},
        coupling=.3, matching_energy_GeV2=1.4, total_k2=9, total_pplus_GeV=1.3,
        transfer_GeV=list(delta), quark_mass_GeV=.35, gluon_mass_GeV=0., width_GeV=.4,
        preexisting_bare_M2_GeV2=float(eigh(original.matrix, eigvals_only=True)[0]),
        records=records, physical_finite_conditions_selected=False,
        common_continuum_regulator_complete=False, covariance_repaired=False,
        quantum_execution=False, seconds=time.monotonic()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("PROTON", "NEUTRON"), required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--current", choices=("direct", "full"), required=True,
        help="Direct insertion only, or full matched minus current with the changed raw backflow.")
    parser.add_argument("--output", type=Path,
        help="Optional generated JSON results file; parent directory must exist.")
    args = parser.parse_args()
    output = json.dumps(run(args.target, args.cutoff, args.current), indent=2)+"\n"
    if args.output is not None:
        args.output.write_text(output)
    print(output, flush=True)
