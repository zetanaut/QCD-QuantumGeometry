"""Exact analytic PennyLane oracle for the represented M2o contract.

This is simulator validation, not an eigensolver, ansatz, hardware algorithm,
or physical activation. SparseHamiltonian is a measured observable, not a
nonunitary state-update gate. Actions and complex transitions are recovered
from its expectation values by polarization, independently of SciPy matvecs.
The full nine-wire basis is retained even for helicity eigenstates.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Mapping, Sequence

import numpy as np

from .m2_pennylane_contract import (
    M2PennyLaneContract,
    compact_to_padded_state,
    padding_projector,
)


class M2PennyLaneOracle:
    """Measure a caller-supplied M2o contract on analytic lightning.qubit.

    Public state inputs must already be normalized; they are never silently
    repaired or assigned physical meaning. ``*_padded`` methods deliberately
    accept padding amplitude for isolation/penalty diagnostics. Ordinary
    methods accept only compact represented-subspace vectors.
    """

    physical = False
    shots = None

    def __init__(self, contract: M2PennyLaneContract) -> None:
        if not isinstance(contract, M2PennyLaneContract):
            raise TypeError("contract must be an M2PennyLaneContract")
        # Keep PennyLane optional for the classical scientific modules.
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Install the declared project [quantum] extra") from exc

        self.contract = contract
        self.basis = contract.full_basis
        self.wires = tuple(range(self.basis.qubits))
        self.device = qml.device(
            "lightning.qubit", wires=self.wires, shots=None, c_dtype=np.complex128
        )
        matrices = {
            "hamiltonian": contract.penalized_hamiltonian,
            **contract.hermitian_observables,
            "padding_probability": padding_projector(self.basis),
        }
        self.observable_names = tuple(matrices)
        observables = tuple(
            qml.SparseHamiltonian(matrix.copy(), wires=self.wires)
            for matrix in matrices.values()
        )

        @qml.qnode(self.device, interface=None, diff_method=None)
        def prepare(state):
            qml.StatePrep(state, wires=self.wires, normalize=False)
            return qml.state()

        @qml.qnode(self.device, interface=None, diff_method=None)
        def measure(state):
            qml.StatePrep(state, wires=self.wires, normalize=False)
            return tuple(qml.expval(observable) for observable in observables)

        self._prepare = prepare
        self._measure = measure

    @staticmethod
    def _state(values: Sequence[complex], dimension: int) -> np.ndarray:
        state = np.asarray(values, dtype=np.complex128)
        if state.shape != (dimension,) or not np.all(np.isfinite(state)):
            raise ValueError(f"state must be finite with shape ({dimension},)")
        norm_squared = float(np.vdot(state, state).real)
        if not np.isclose(norm_squared, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("state must already be normalized")
        return state.copy()

    def _encode(self, compact_state: Sequence[complex]) -> np.ndarray:
        return compact_to_padded_state(
            self.basis, self._state(compact_state, self.basis.compact_dimension)
        )

    def prepare(self, compact_state: Sequence[complex]) -> np.ndarray:
        """Return the actual backend state, in the declared big-endian order."""
        return self.prepare_padded(self._encode(compact_state))

    def prepare_padded(self, padded_state: Sequence[complex]) -> np.ndarray:
        state = self._state(padded_state, self.basis.padded_dimension)
        return np.asarray(self._prepare(state), dtype=np.complex128).copy()

    def expectations(self, compact_state: Sequence[complex]) -> Mapping[str, float]:
        return self.expectations_padded(self._encode(compact_state))

    def expectations_padded(self, padded_state: Sequence[complex]) -> Mapping[str, float]:
        state = self._state(padded_state, self.basis.padded_dimension)
        measured = np.asarray(self._measure(state), dtype=float)
        return dict(zip(self.observable_names, measured.tolist()))

    def _quadratic_form(self, unnormalized: np.ndarray) -> np.ndarray:
        norm_squared = float(np.vdot(unnormalized, unnormalized).real)
        if norm_squared == 0.0:
            # The quadratic form on the zero vector is exactly zero; no
            # normalized quantum state exists for this polarization term.
            return np.zeros(len(self.observable_names))
        return norm_squared * np.asarray(
            self._measure(unnormalized / np.sqrt(norm_squared)), dtype=float
        )

    def _transition(self, bra: np.ndarray, ket: np.ndarray) -> np.ndarray:
        # For q(v)=v^dagger O v (O Hermitian), recover bra^dagger O ket.
        # The minus sign before i is essential for imaginary transitions.
        return 0.25 * (
            self._quadratic_form(bra + ket)
            - self._quadratic_form(bra - ket)
            - 1j * (
                self._quadratic_form(bra + 1j * ket)
                - self._quadratic_form(bra - 1j * ket)
            )
        )

    def transitions(
        self, bra: Sequence[complex], ket: Sequence[complex]
    ) -> Mapping[str, complex]:
        """Return all Hermitian-observable transitions via four preparations.

        The endpoints may be nonorthogonal or identical, including endpoints
        differing only by phase. Norms of the four sums are retained exactly.
        """
        measured = self._transition(self._encode(bra), self._encode(ket))
        return dict(zip(self.observable_names, measured.tolist()))

    def actions(self, compact_state: Sequence[complex]) -> Mapping[str, np.ndarray]:
        """Reconstruct all 512 components using measured basis transitions.

        This O(2**n) exact-oracle diagnostic is intentionally not a scalable
        circuit for applying a nonunitary operator. Every padding component
        is measured too; no physical-subspace isolation is assumed.
        """
        ket = self._encode(compact_state)
        result = np.empty(
            (len(self.observable_names), self.basis.padded_dimension),
            dtype=np.complex128,
        )
        bra = np.zeros(self.basis.padded_dimension, dtype=np.complex128)
        for index in range(self.basis.padded_dimension):
            bra[index] = 1.0
            result[:, index] = self._transition(bra, ket)
            bra[index] = 0.0
        return dict(zip(self.observable_names, result))

    @staticmethod
    def directed(
        measured: Mapping[str, complex | np.ndarray], *, reverse: bool = False
    ) -> Mapping[str, complex | np.ndarray]:
        """Reconstruct density/Jx/Jy at either direction of the same transfer."""
        sign = -1j if reverse else 1j
        return {
            name: measured[f"{name}_cosine"] + sign * measured[f"{name}_sine"]
            for name in ("density", "current_x", "current_y")
        }

    def execution_metadata(self) -> Mapping[str, object]:
        return {
            "device": "lightning.qubit",
            "pennylane": version("PennyLane"),
            "lightning": version("PennyLane-Lightning"),
            "qubits": self.basis.qubits,
            "compact_dimension": self.basis.compact_dimension,
            "padded_dimension": self.basis.padded_dimension,
            "wire_order": self.wires,
            "shots": None,
            "precision": "complex128",
            "action_route": "expectation-value polarization; all padded rows",
            "variational_algorithm_selected": False,
            "physical_activation": False,
        }


__all__ = ["M2PennyLaneOracle"]
