"""Reproduce numerical M2o quantum parity; print a compact JSON result.

Run with the installed project [quantum] extra. No optimization or physical
parameter inference is performed. All action entries come from actual
PennyLane expectation measurements, not sparse matvecs inside the adapter.
"""
from __future__ import annotations

import json
import platform
from importlib.metadata import version

import numpy as np

from deuteron_wigner.quantum.m2_hamiltonian_gauged_current import (
    build_hamiltonian_gauged_baryon_current,
    build_m2_baryon_density_operator,
)
from deuteron_wigner.quantum.m2_pennylane_contract import build_m2_pennylane_contract
from deuteron_wigner.quantum.m2_pennylane_oracle import M2PennyLaneOracle
from deuteron_wigner.quantum.m2_total_sign_h2_hamiltonian import build_total_sign_h2_hamiltonian


def maximum(value):
    return float(np.max(np.abs(value)))


def compare(target, dx, dy):
    h = build_total_sign_h2_hamiltonian(target)
    def current(sign):
        return build_hamiltonian_gauged_baryon_current(build_m2_baryon_density_operator(
            h, delta_x_GeV=sign * dx, delta_y_GeV=sign * dy,
        ))
    positive, reverse = current(1), current(-1)
    oracle = M2PennyLaneOracle(build_m2_pennylane_contract(h, positive, reverse))
    rng = np.random.default_rng(20260905)
    probes = rng.normal(size=(2, 266)) + 1j * rng.normal(size=(2, 266))
    probes /= np.linalg.norm(probes, axis=1)[:, None]
    probe, other = probes
    actual_state = oracle.prepare(probe)
    actions, measured = oracle.actions(probe), oracle.expectations(probe)
    compact = {"hamiltonian": h.matrix}
    for name, matrix in (("density", positive.density.matrix),
                         ("current_x", positive.current_x), ("current_y", positive.current_y)):
        compact[name + "_cosine"] = (matrix + matrix.getH()) / 2
        compact[name + "_sine"] = (matrix - matrix.getH()) / (2j)
    transitions = oracle.transitions(probe, other)
    record = {
        "target": target, "transfer_GeV": [dx, dy],
        "state_residual": maximum(actual_state[:266] - probe),
        "state_padding_probability": float(np.vdot(actual_state[266:], actual_state[266:]).real),
        "hamiltonian_action_residual_GeV2": maximum(actions["hamiltonian"][:266] - h.matrix @ probe),
        "quadrature_action_residual": max(maximum(actions[n][:266] - a @ probe)
                                          for n, a in compact.items() if n != "hamiltonian"),
        "expectation_residual": max(abs(measured[n] - np.vdot(probe, a @ probe))
                                    for n, a in compact.items()),
        "complex_transition_residual": max(abs(transitions[n] - np.vdot(probe, a @ other))
                                           for n, a in compact.items()),
        "action_padding_residual": max(maximum(a[266:]) for a in actions.values()),
        "directed_current_action_norms": {n: float(np.linalg.norm(a))
                                          for n, a in oracle.directed(actions).items() if n != "density"},
    }
    continuity, reversal = [], []
    for reversed_flag, c in ((False, positive), (True, reverse)):
        directed = oracle.directed(actions, reverse=reversed_flag)
        for name, matrix in (("density", c.density.matrix),
                             ("current_x", c.current_x), ("current_y", c.current_y)):
            reversal.append(maximum(directed[name][:266] - matrix @ probe))
        lhs = c.density.delta_x_GeV * directed["current_x"] + c.density.delta_y_GeV * directed["current_y"]
        rho = c.density.matrix
        continuity.append(maximum(lhs[:266] - (h.matrix @ (rho @ probe) - rho @ (h.matrix @ probe))))
    record["directed_and_reversed_action_residual"] = max(reversal)
    record["continuity_action_residual_GeV2"] = max(continuity)
    energies, states = np.linalg.eigh(h.matrix.toarray())
    tx = states.conj().T @ (positive.current_x @ states)
    np.fill_diagonal(tx, 0)
    m, n = np.unravel_index(np.argmax(np.abs(tx)), tx.shape)
    spectral = oracle.directed(oracle.transitions(states[:, m], states[:, n]))
    rho_mn = np.vdot(states[:, m], positive.density.matrix @ states[:, n])
    record["nonzero_spectral_Jx_magnitude_GeV"] = float(abs(spectral["current_x"]))
    record["spectral_transition_residual_GeV"] = max(
        abs(spectral[name] - delta / (dx*dx + dy*dy) * (energies[m] - energies[n]) * rho_mn)
        for name, delta in (("current_x", dx), ("current_y", dy))
    )
    record["ground_energy_residual_GeV2"] = max(
        abs(oracle.expectations(s.state)["hamiltonian"] - s.energy_GeV2)
        for s in oracle.contract.helicity_pair.states
    )
    if max(v for k, v in record.items() if "residual" in k) > 8e-13:
        raise RuntimeError(f"Quantum parity failed for {target}, {(dx, dy)}")
    if record["nonzero_spectral_Jx_magnitude_GeV"] < .1:
        raise RuntimeError("The selected transition is not a nontrivial current probe")
    return record


if __name__ == "__main__":
    result = {
        "scope": "Exact analytic simulator parity for the represented M2 system; nonphysical",
        "python": platform.python_version(),
        "platform": platform.system() + "-" + platform.machine(),
        "packages": {name: version(name) for name in (
            "numpy", "scipy", "sympy", "mpmath", "PennyLane", "PennyLane-Lightning", "autoray",
        )},
        "device": "lightning.qubit", "qubits": 9, "shots": None,
        "seed": 20260905, "absolute_tolerance": 8e-13,
        "actions": "Polarization of backend-measured quadratic forms, including all 512 output rows",
        "physical_activation": False,
        "cases": [compare(target, dx, dy) for target in ("PROTON", "NEUTRON")
                  for dx, dy in ((.2, 0.), (.12, .16))],
    }
    print(json.dumps(result, indent=2))
