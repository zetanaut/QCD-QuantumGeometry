"""Exact finite-parent reduction and matched operators for the H2 program.

This is a finite-matrix reference, not a higher-Fock matching calculation.
The supplied parent Hamiltonian and operators define all of its physics.
No missing QCD block is inferred as zero. H and the energy variable are M^2
in GeV^2; wave operators and the induced norm metric are dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import brentq

from .m2_coupled_longitudinal_jacobi import oscillator_shell


def _matrix(value, name):
    a = np.array(value, dtype=complex, copy=True)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite matrix")
    return a


def _hermitian(a, name):
    if a.shape[0] != a.shape[1] or not np.allclose(
        a, a.conj().T, rtol=0, atol=2e-11 * max(1., np.linalg.norm(a))
    ):
        raise ValueError(f"{name} must be Hermitian")


def _inverse_sqrt(metric):
    # A norm close to I has nearly degenerate eigenvalues. Divide-and-conquer
    # preserves orthogonality more accurately here than the default MRRR
    # eigenvectors; no physical norm eigenvalue or conservation tolerance changes.
    values, vectors = linalg.eigh(metric, driver="evd")
    if values[0] <= 0:
        raise ArithmeticError("effective norm metric is not positive")
    return (vectors / np.sqrt(values)) @ vectors.conj().T


@dataclass(frozen=True)
class ReducedEigenstate:
    energy_GeV2: float
    retained: np.ndarray
    reconstructed: np.ndarray
    excluded_probability: float
    parent_residual_GeV2: float


@dataclass(frozen=True)
class LinearizedModel:
    """Energy-linearized generalized eigenproblem in orthonormal coordinates."""
    expansion_energy_GeV2: float
    matrix: np.ndarray
    metric: np.ndarray
    parent_isometry: np.ndarray

    def operator(self, parent_operator):
        op = _matrix(parent_operator, "parent operator")
        n = self.parent_isometry.shape[0]
        if op.shape != (n, n):
            raise ValueError("operator must act on the explicit parent space")
        return self.parent_isometry.conj().T @ op @ self.parent_isometry


class FiniteParentReduction:
    """Feshbach/Schur elimination with exact finite-parent current matching.

    retained_isometry carries the actual small basis into the parent basis;
    equal dimension/order is never sufficient. A dense complement and its
    eigendecomposition make this a small-reference tool, not a speed claim.
    Only real energies away from ALL excluded-block poles are supported.
    """
    def __init__(self, parent_hamiltonian, retained_isometry, *, scope,
                 pole_rtol=1e-11):
        h = _matrix(parent_hamiltonian, "parent Hamiltonian")
        p = _matrix(retained_isometry, "retained isometry")
        _hermitian(h, "parent Hamiltonian")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("explicit physical/diagnostic scope is required")
        if not np.isfinite(pole_rtol) or pole_rtol <= 0:
            raise ValueError("positive finite pole tolerance required")
        n = h.shape[0]
        if p.shape[0] != n or not 0 < p.shape[1] < n:
            raise ValueError("retained space must be a nonempty proper subspace")
        if not np.allclose(p.conj().T @ p, np.eye(p.shape[1]), rtol=0, atol=2e-11):
            raise ValueError("retained columns must be an isometry")
        q = linalg.qr(p, mode="full")[0][:, p.shape[1]:]
        self.h, self.p, self.q = h, p, q
        self.scope, self.pole_rtol = scope, float(pole_rtol)
        self.hpp = p.conj().T @ h @ p
        self.hqp = q.conj().T @ h @ p
        self.hqq = q.conj().T @ h @ q
        self.poles, self.q_eigenvectors = linalg.eigh(self.hqq)
        self.spectral_coupling = self.q_eigenvectors.conj().T @ self.hqp

    def _denominators(self, energy):
        if np.ndim(energy) != 0 or np.iscomplexobj(energy) or not np.isfinite(energy):
            raise ValueError("finite real mass-squared energy required")
        delta = float(energy) - self.poles
        scale = max(1., abs(float(energy)), float(np.max(np.abs(self.poles))))
        if np.min(np.abs(delta)) <= self.pole_rtol * scale:
            raise ValueError("energy at/too near an excluded-space pole")
        return delta

    def wave_operator(self, energy):
        """Omega(E)=P+Q(E-H_QQ)^-1 H_QP, in parent coordinates."""
        delta = self._denominators(energy)
        tail = self.q_eigenvectors @ (self.spectral_coupling / delta[:, None])
        return self.p + self.q @ tail

    def effective_hamiltonian(self, energy):
        delta = self._denominators(energy)
        b = self.spectral_coupling
        return self.hpp + b.conj().T @ (b / delta[:, None])

    def energy_derivative(self, energy):
        """dH_eff/dE = -H_PQ(E-H_QQ)^-2 H_QP, NOT positive."""
        delta = self._denominators(energy)
        x = self.spectral_coupling / delta[:, None]
        return -x.conj().T @ x

    def norm_metric(self, energy):
        return np.eye(self.p.shape[1]) - self.energy_derivative(energy)

    def matched_operator(self, parent_operator, energy_out, energy_in):
        """Omega(E_out)^dag O Omega(E_in), including all P/Q cross blocks."""
        op = _matrix(parent_operator, "parent operator")
        if op.shape != self.h.shape:
            raise ValueError("operator must act on the explicit parent space")
        left, right = self.wave_operator(energy_out), self.wave_operator(energy_in)
        return left.conj().T @ op @ right

    def linearized_model(self, expansion_energy):
        """Linearize H_eff(E) and retain the induced metric.

        A=H_eff(E0)-E0 H_eff'(E0), N=I-H_eff'(E0);
        H_lin=N^-1/2 A N^-1/2. Equivalently it is the parent Rayleigh-Ritz
        Hamiltonian in U=Omega(E0) N^-1/2. Matching EVERY operator with U
        preserves charge. This is one energy-local approximation, not an
        exact single operator for all transitions at different energies.
        """
        metric = self.norm_metric(expansion_energy)
        invsqrt = _inverse_sqrt(metric)
        u = self.wave_operator(expansion_energy) @ invsqrt
        a = self.effective_hamiltonian(expansion_energy) - (
            expansion_energy * self.energy_derivative(expansion_energy))
        return LinearizedModel(float(expansion_energy), invsqrt @ a @ invsqrt,
                               metric, u)

    def linearization_remainder_bound(self, energy, expansion_energy):
        """Spectral-norm bound for the unnormalized H_eff Taylor remainder."""
        d0 = np.min(np.abs(self._denominators(expansion_energy)))
        self._denominators(energy)
        shift = abs(float(energy) - float(expansion_energy))
        if shift >= d0:
            raise ValueError("energy lies outside the pole-free Taylor disk")
        return float(np.linalg.norm(self.hqp, 2)**2 * shift**2 /
                     (d0*d0*(d0-shift)))

    def lowest_state(self):
        """Find the lowest root below all Q poles, without the parent eigensolve.

        States with zero retained projection cannot be reconstructed. If the
        ground state is an excluded intruder at a pole, no root in this
        supported interval is returned; do not shift/clip a pole to hide it.
        """
        def objective(energy):
            return float(linalg.eigvalsh(self.effective_hamiltonian(energy),
                         subset_by_index=[0, 0])[0] - energy)
        scale = max(1., np.linalg.norm(self.h, np.inf))
        lower = float(np.min(self.h.diagonal().real -
                            (np.abs(self.h).sum(axis=1)-np.abs(self.h.diagonal()))) - scale)
        margin = 100 * self.pole_rtol * scale
        upper = min(float(linalg.eigvalsh(self.hpp, subset_by_index=[0, 0])[0]),
                    float(self.poles[0] - margin))
        if objective(lower) < 0 or objective(upper) > 1e-12*scale:
            raise ValueError("no supported ground root below excluded poles; enlarge retained space")
        if abs(objective(upper)) <= 1e-13*scale:
            energy = upper
        else:
            energy = brentq(objective, lower, upper, xtol=5e-14, rtol=1e-14)
        _, vectors = linalg.eigh(self.effective_hamiltonian(energy), subset_by_index=[0, 0])
        v = vectors[:, 0]
        v /= np.sqrt(np.vdot(v, self.norm_metric(energy) @ v).real)
        full = self.wave_operator(energy) @ v
        probability = float(np.linalg.norm(self.q.conj().T @ full)**2)
        residual = float(np.linalg.norm(self.h @ full - energy*full))
        return ReducedEigenstate(float(energy), v, full, probability, residual)


def nested_pilot_isometry(smaller, parent):
    """Physical occupation-label embedding between two common-width H2 shells.

    Q here consists ONLY of higher oscillator modes of qqq+qqqg. It is NOT
    qqqgg or sea-quark space. Compact symmetry eigenvectors may rotate.
    """
    for name in ("total_k2", "target", "jz_twice", "width_GeV",
                 "quark_mass_GeV", "gluon_mass_GeV", "coupling"):
        if getattr(smaller, name) != getattr(parent, name):
            raise ValueError(f"nested pilots disagree on {name}")
    if smaller.cutoff >= parent.cutoff:
        raise ValueError("parent must have a strictly larger oscillator cutoff")
    blocks = []
    for d, nc, low, high in zip((2, 3, 3), (1, 2, 2), smaller.bases, parent.bases):
        small_shell = oscillator_shell(d, smaller.cutoff)
        large_shell = oscillator_shell(d, parent.cutoff)
        lookup = {state: i for i, state in enumerate(large_shell)}
        orbital = np.array([lookup[state] for state in small_shell])
        cols = np.arange(low.shape[0])
        slots = nc*64
        lo = cols // (len(small_shell)*slots)
        rem = cols % (len(small_shell)*slots)
        rows = (lo*len(large_shell) + orbital[rem//slots])*slots + rem % slots
        embedding = sparse.csr_matrix((np.ones(len(cols)), (rows, cols)),
                                     shape=(high.shape[0], low.shape[0]))
        lifted = embedding @ low
        block = high.getH() @ lifted
        if sparse.linalg.norm(lifted - high @ block) > 2e-10:
            raise ArithmeticError("nested state leaves the parent symmetry sector")
        blocks.append(block)
    result = sparse.block_diag(blocks, format="csr").toarray()
    if not np.allclose(result.conj().T @ result, np.eye(result.shape[1]), rtol=0, atol=3e-11):
        raise ArithmeticError("nested compact map is not an isometry")
    return result
