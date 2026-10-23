"""Finite-reference matching across explicitly identified excluded cutoffs.

This transports an operator family, not physical renormalization conditions.
The larger/reference parent is supplied explicitly; calling it a reference
does not make it continuum QCD. The retained coordinate map is mandatory.
"""
from dataclasses import dataclass

import numpy as np

from .m2_matched_h2_reduction import _matrix


@dataclass(frozen=True)
class EnergyLocalCounterterm:
    anchor_GeV2: float
    value_GeV2: np.ndarray
    slope: np.ndarray

    def at(self, energy_GeV2):
        e = float(energy_GeV2)
        if not np.isfinite(e):
            raise ValueError("finite real energy required")
        return self.value_GeV2 + (e-self.anchor_GeV2)*self.slope

    @property
    def norm_counterterm(self):
        return -self.slope


class FiniteReferenceTransport:
    """Delta H(E), -Delta H'(E), and two-energy Delta O on a common P space.

    map_to_reference maps lower-parent retained coordinates into reference
    retained coordinates. It must be an explicitly supplied unitary, even
    when both compact dimensions coincide. Parent Q spaces may differ.
    """
    def __init__(self, lower, reference, *, map_to_reference):
        t = _matrix(map_to_reference, "retained coordinate map")
        n = lower.p.shape[1]
        if t.shape != (reference.p.shape[1], n) or t.shape[0] != n:
            raise ValueError("equally sized retained spaces and a square map required")
        if not np.allclose(t.conj().T@t, np.eye(n), rtol=0, atol=2e-11):
            raise ValueError("retained coordinate map must be unitary")
        self.lower, self.reference, self.map = lower, reference, t

    def _back(self, value):
        return self.map.conj().T@value@self.map

    def hamiltonian_counterterm(self, energy):
        return (self._back(self.reference.effective_hamiltonian(energy))
                - self.lower.effective_hamiltonian(energy))

    def slope_counterterm(self, energy):
        return (self._back(self.reference.energy_derivative(energy))
                - self.lower.energy_derivative(energy))

    def norm_counterterm(self, energy):
        return -self.slope_counterterm(energy)

    def local_counterterm(self, anchor):
        return EnergyLocalCounterterm(float(anchor),
            self.hamiltonian_counterterm(anchor), self.slope_counterterm(anchor))

    def hamiltonian_remainder_bound(self, energy, anchor):
        """Conservative spectral-norm Taylor bound from both declared parents."""
        return (self.lower.linearization_remainder_bound(energy, anchor)
                + self.reference.linearization_remainder_bound(energy, anchor))

    def operator_counterterm(self, lower_operator, reference_operator, energy_out, energy_in):
        """Two independently supplied parent operators; neither is inferred."""
        return (self._back(self.reference.matched_operator(reference_operator, energy_out, energy_in))
                - self.lower.matched_operator(lower_operator, energy_out, energy_in))

    @staticmethod
    def _wave_derivative(reduction, energy):
        delta = reduction._denominators(energy)
        return -reduction.q@reduction.q_eigenvectors@(
            reduction.spectral_coupling/delta[:, None]**2)

    @classmethod
    def _operator_jet(cls, reduction, operator, anchor):
        op = _matrix(operator, "parent operator")
        if op.shape != reduction.h.shape:
            raise ValueError("parent operator dimensions do not match")
        w = reduction.wave_operator(anchor)
        dw = cls._wave_derivative(reduction, anchor)
        return w.conj().T@op@w, dw.conj().T@op@w, w.conj().T@op@dw

    def operator_jet_counterterm(self, lower_operator, reference_operator, anchor):
        """Return (Delta O, d_Eout Delta O, d_Ein Delta O) at (E0,E0).

        The two slopes differ for general complex transition operators;
        a single-energy density correction is insufficient for transitions.
        These are unnormalized effective operators, with the induced metric.
        """
        low = self._operator_jet(self.lower, lower_operator, anchor)
        high = self._operator_jet(self.reference, reference_operator, anchor)
        return tuple(self._back(b)-a for a, b in zip(low, high))

    def generalized_lower_parent(self, anchor):
        """(A,B) giving lower Heff(E)+Delta H(E0)+(E-E0)Delta H'(E0).

        The full lower-parent generalized eigenproblem is A psi=E B psi.
        Its Schur complement reproduces that energy-local transport. B must
        be positive; no eigenvalue clipping or negative-metric repair.
        Operator/current matching is separate: this routine does not invent
        a full-parent current for this approximate generalized model.
        """
        ct = self.local_counterterm(anchor)
        p = self.lower.p
        a = self.lower.h + p@(ct.value_GeV2-anchor*ct.slope)@p.conj().T
        b = np.eye(len(a))-p@ct.slope@p.conj().T
        if np.linalg.eigvalsh(b)[0] <= 0:
            raise ValueError("energy-local reference transport has a nonpositive metric")
        return a, b
