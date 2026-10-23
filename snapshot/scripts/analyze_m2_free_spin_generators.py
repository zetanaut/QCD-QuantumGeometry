"""Replay free-source spin/projection diagnostics; not an interacting fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from deuteron_wigner.quantum.m2_free_spin_generators import (
    free_cell_galerkin, free_jacobi_tangent, free_pauli_lubanski,
    longitudinal_cell_operators, projected_commutator,
)


def analyze():
    cases = {}
    for particle, mass, modes in (("quark", .3, (1, 3, 5)), ("gluon", 0., (2, 4, 6))):
        low = free_cell_galerkin(modes, 2.7, 1, .4, mass, particle=particle)
        high = free_cell_galerkin(modes, 2.7, 4, .4, mass, particle=particle)
        rows = [((k*15+n)*2+s) for k in range(3) for n in range(3) for s in range(2)]
        u = np.eye(90)[:, rows]
        split = projected_commutator(u, high["F1"], high["F2"])
        f, p = low["F1"], low["Pplus"]
        boundary = np.linalg.norm(f@p-p@f+1j*low["Px"]-1j*low["cell_boundary_x"])
        wp, wm, wx, wy = free_pauli_lubanski([.73, .27, -.16], mass, particle=particle)
        casimir = np.linalg.norm(wp@wm+wm@wp-wx@wx-wy@wy+.75*mass*mass*np.eye(2))
        cases[particle] = {
            "mass_GeV": mass, "doubled_modes": modes, "half_length_GeVinv": 2.7,
            "width_GeV": .4, "retained_cutoff": 1, "parent_cutoff": 4,
            "retained_dimension": len(f), "parent_dimension": len(high["F1"]),
            "cell_plus_commutator_residual_GeV": float(boundary),
            "free_casimir_residual_GeV2": float(casimir),
            "projected_F1_F2_commutator_norm": float(np.linalg.norm(split["commutator"])),
            "parent_cell_defect_projected_norm": float(np.linalg.norm(split["parent_commutator"])),
            "transverse_projection_leakage_norm": float(np.linalg.norm(split["leakage"])),
            "leakage_identity_residual": split["identity_residual"],
        }
        if max(boundary, casimir, split["identity_residual"]) > 1e-12:
            raise ArithmeticError("free-source identity failed")
    cell = []
    for n in (3, 5, 9):
        x, p, b = longitudinal_cell_operators(tuple(range(1, 2*n, 2)), 2.7)
        cell.append({"mode_count": n, "boundary_spectral_norm": float(np.linalg.norm(b, 2)),
                     "commutator_identity_residual": float(np.linalg.norm(x@p-p@x-1j*(np.eye(n)-b)))})
    pp_dot, dx, dq = free_jacobi_tangent((1, 3, 5), np.array([[.07, -.12], [.19, .04], [0., 0.]]),
                                        2.1, [.3]*3, 0)
    return {"scope": "source-derived free one-particle and finite-cell/Galerkin references; not the interacting many-body H2 spin action",
            "cases": cases, "APBC_cell_resolution": cell,
            "qqq_K2_9_zero_total_transverse_F1_tangent": {
                "modes": [1, 3, 5], "total_pplus_GeV": 2.1, "masses_GeV": [.3]*3,
                "coordinates_GeV": [[.07, -.12], [.19, .04], [0., 0.]],
                "total_pplus_derivative_GeV": pp_dot, "fraction_derivative": dx.tolist(),
                "Jacobi_derivative_GeV": dq.tolist()},
            "physical_prediction": False, "interacting_spin_qualified": False,
            "current_covariance_repaired": False,
            "next": "derive regulated interaction-dependent generators and many-body domain action, then common-order H2/norm transport and multiplet/current tests"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(analyze(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
