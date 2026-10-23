"""Embedded exact inner-shell free resolvent for gauged-cutoff comparisons.

This is Pi (E-Pi H0 Pi)^-1 Pi on the supplied OUTER packed coordinates.
It is not Pi (E-H0_outer)^-1 Pi. The difference matters beginning at
the non-diagonal radial shell. Pole clipping and source compression are absent.
"""
import numpy as np

from .m2_shell_projector_current import inner_mask, _columns
from .m2_shell_free_propagation import ShellFreePropagation


class ProjectedShellFreeResolvent:
    def __init__(self, outer, *, inner_cutoff, b_GeV, quark_mass_GeV, gluon_mass_GeV):
        self.space = outer
        mask = inner_mask(outer, inner_cutoff)
        self.inner = type(outer)(outer.sector, outer.total_k2, inner_cutoff, outer.target, outer.jz_twice)
        self.rows = np.flatnonzero(mask)
        if (outer.longitudinal != self.inner.longitudinal
                or tuple(outer.orbitals[:len(self.inner.orbitals)]) != self.inner.orbitals
                or not np.array_equal(outer.coordinates[self.rows], self.inner.coordinates)):
            raise ValueError("exact nested raw coordinate embedding is missing")
        self.free = ShellFreePropagation(self.inner, b_GeV=b_GeV,
            quark_mass_GeV=quark_mass_GeV, gluon_mass_GeV=gluon_mass_GeV)

    def _act(self, values, operation):
        a = _columns(self.space, values)
        result = np.zeros_like(a)
        result[self.rows] = operation(a[self.rows])
        return result

    def apply(self, values):
        """Pi H0 Pi, embedded on the outer coordinate set."""
        return self._act(values, self.free.apply)

    def resolve(self, energy, values):
        return self._act(values, lambda a: self.free.resolve(energy, a))
