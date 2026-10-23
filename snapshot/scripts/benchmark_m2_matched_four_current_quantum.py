"""Execute all four canonical tree-current components on the same H2 state.

Bare common-order g^4 model, NOT full QCD, a physical current, covariance
repair or quantum advantage. The earlier transverse benchmark is unchanged.
"""
import argparse
import json
import time
import numpy as np
from scipy.linalg import eigh

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import FactorizedCanonicalQQSourceAction, ShellCanonicalQQInsertion
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2, current_between
from deuteron_wigner.quantum.m2_fourth_order_transverse_current import transverse_coefficients_between
from deuteron_wigner.quantum.m2_fourth_order_minus_current import minus_coefficients_between
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit
from deuteron_wigner.quantum.m2_current_covariance_diagnostic import current_covariance_diagnostic
from deuteron_wigner.quantum.m2_orientation_current_diagnostic import orientation_current_diagnostic


def run(target, cutoff):
    start = time.monotonic()
    delta, P, g = (.12, .16), 1.3, .3

    def build(jz):
        p = build_coupled_pilot(total_k2=9, cutoff=0, target=target, jz_twice=jz,
            b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=g)
        ref = build_shell_fock_reference(p, out_cutoff=cutoff)
        qq = ShellCanonicalQQInsertion(ref, action=FactorizedCanonicalQQSourceAction(ref))
        return FourthOrderH2(ref, QQ_insertion=qq,
            six_body=CanonicalSixBodyExcursion(ref, out_cutoff=cutoff))

    print("Building the shared H/norm/current snapshots", flush=True)
    models = [build(1), build(-1)]
    snapshots = [m.snapshot(1.4) for m in models]
    s = snapshots[0]
    circuit = SymmetryGraphCircuit(s.matrix,
        root=int(np.argmin(s.matrix.diagonal().real)), support_threshold=1e-12)
    initial = circuit.uniform_parameters()
    initial[1::2] = .03*np.random.default_rng(4901).normal(size=len(circuit.edges))
    fit = quantum_natural_fit(circuit, initial, geometry="tree", metric_rcond=None,
        probability_regularization=1e-20, step_rule="tree_state_retraction",
        residual_tolerance=1e-8, max_iterations=2000)
    print("Quantum state solved; constructing all four matched current components", flush=True)
    # Reference computed AFTER optimization, never supplied as a seed.
    e, u = eigh(s.matrix)
    exact, mass = u[:, 0], np.sqrt(fit.fun)
    maps = [np.eye(len(s.matrix)),
            1j*transverse_parity_map(models[1].pilot, models[0].pilot)]
    operators, keys, coefficient_norms = [], [], {}
    for hf, f in enumerate(snapshots):
        for hi, i in enumerate(snapshots):
            plus = current_between(f, i, delta, charge="electromagnetic", normalized=True)
            tc = transverse_coefficients_between(f, i, delta,
                total_pplus_GeV=P, charge="electromagnetic")
            mc = minus_coefficients_between(f, i, delta,
                total_pplus_GeV=P, charge="electromagnetic")
            transverse = [f.whitener@a@i.whitener for a in sum(g**k*a for k, a in tc.items())]
            minus = f.whitener@sum(g**k*a for k, a in mc.items())@i.whitener
            coefficient_norms[f"{hf}{hi}"] = dict(
                transverse=[float(np.linalg.norm(tc[k])) for k in range(5)],
                minus=[float(np.linalg.norm(mc[k])) for k in range(5)])
            for mu, a in enumerate([plus, minus, *transverse]):
                operators.append(maps[hf].conj().T@a@maps[hi])
                keys.append((mu, hf, hi))
    print("Measuring 16 complex helicity elements as 32 Hermitian quadratures", flush=True)
    hermitian = [part for a in operators for part in ((a+a.conj().T)/2, (a-a.conj().T)/(2j))]
    measured = circuit.measure_operators(fit.x, hermitian).reshape(-1, 2)
    values = measured[:, 0]+1j*measured[:, 1]
    reference = np.array([np.vdot(exact, a@exact) for a in operators])
    diagnostic = circuit.state_diagnostics(fit.x)
    gap = e[1]-diagnostic["energy_GeV2"]
    if gap <= 0:
        raise ArithmeticError("positive exact excited-state gap required for state-error bound")
    part_bounds = [2*np.linalg.norm(a, 2)*diagnostic["residual_GeV2"]/gap for a in hermitian]
    bounds = np.sum(np.array(part_bounds).reshape(-1, 2), axis=1)
    current = np.zeros((4, 2, 2), complex)
    for key, value in zip(keys, values):
        current[key] = value
    cov = current_covariance_diagnostic(current[0], current[2:], delta,
        mass_GeV=mass, total_pplus_GeV=P)
    orientation = orientation_current_diagnostic(current[0], current[2:], delta,
        mass_GeV=mass, total_pplus_GeV=P, minus=current[1])

    def pair(z):
        return [float(np.real(z)), float(np.imag(z))]

    errors = abs(values-reference)
    return dict(scope="bare canonical normal-ordered tree current through common H2 g^4; NOT full QCD/physical four-current",
        target=target, excluded_cutoff=cutoff, retained_dimension=len(s.matrix), qubits=circuit.qubits,
        total_k2=9, retained_cutoff=0, width_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0.,
        coupling=g, matching_energy_GeV2=1.4, total_pplus_GeV=P, transfer_GeV=list(delta),
        M2_GeV2=fit.fun, residual_GeV2=diagnostic["residual_GeV2"],
        residual_tolerance_met=bool(fit.success), iterations=int(fit.nit), evaluations=int(fit.nfev),
        padding_probability=diagnostic["padding_probability"],
        maximum_current_error=float(max(errors)), maximum_current_state_bound=float(max(bounds)),
        every_current_error_within_bound=bool(np.all(errors <= bounds)),
        component_order=["plus", "minus", "x", "y"], helicity_order=[1, -1],
        measured_current=[[[pair(z) for z in row] for row in component] for component in current],
        current_errors=[float(x) for x in errors], current_state_bounds=[float(x) for x in bounds],
        F1_from_plus=pair(cov.F1_from_plus), F2_from_plus=pair(cov.F2_from_plus),
        GM_plus=pair(cov.magnetic_from_plus), GM_transverse=pair(cov.magnetic_from_transverse),
        F1_separated=pair(orientation.F1_separated), F2_separated=pair(orientation.F2_separated),
        B1=pair(orientation.B1), B2=pair(orientation.B2), B3=pair(orientation.B3),
        full_five_structure_tested=orientation.full_five_structure_tested,
        orientation_structure_residual_norm=float(np.linalg.norm(orientation.structure_residual)),
        elastic_continuity_defect_norm=float(np.linalg.norm(cov.elastic_continuity_defect)),
        coefficient_norms=coefficient_norms, same_controlled_gate_state_for_all_components=True,
        exact_state_seed=False, physical=False, covariance_repaired=False,
        quantum_advantage=False, seconds=time.monotonic()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("PROTON", "NEUTRON"), required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    args = parser.parse_args()
    result = run(args.target, args.cutoff)
    print(json.dumps(result, indent=2), flush=True)
    if not result["residual_tolerance_met"] or not result["every_current_error_within_bound"]:
        raise SystemExit("finite-model quantum acceptance failed")
