"""Q5 free-propagation loop regulator current with actual canonical sources.

For V=g V1+g^2 V2, this transports V_f^dag R_reg^T(Ef,Ei) V_i at orders
g^2,g^3,g^4. Both unprojected OUTER source tails and moving-projector endpoint
terms are essential. It is a finite-outer-Q regulator correction, not the
complete H2 g^4 current: W55/Q6 loop insertions, P-regulator terms, common
field regulation and finite physical matching remain separate.
"""
import numpy as np

from .m2_shell_fock_reference import ShellFockReference
from .m2_shell_projector_current import QShellProjectorFamily
from .m2_projected_shell_resolvent import ProjectedShellFreeResolvent


def _sources(reference):
    result = {1: {}, 2: {}}
    for entry in reference.sectors:
        name = entry.source.space.sector
        for power in (1, 2):
            pieces = [(cols, v) for p, cols, v in entry.source.components if p == power]
            if len(pieces) != 1:
                raise ValueError("one exact outer cubic and quartic source per named sector required")
            cols, v = pieces[0]
            a = np.zeros((entry.source.space.dimension, len(reference.pilot.matrix)), complex)
            a[:, cols] = v
            result[power][name] = a
    return result


class Q5LoopRegulatorCurrent:
    physical = False
    full_H2_order4_regulator_matching_complete = False

    def __init__(self, outgoing, incoming, *, inner_cutoff, energy_out, energy_in,
                 transfer, charge, total_pplus_GeV, quadrature_order):
        if not isinstance(outgoing, ShellFockReference) or not isinstance(incoming, ShellFockReference):
            raise ValueError("actual uncompressed outer canonical ShellFockReference endpoints required")
        pf, pi = outgoing.pilot, incoming.pilot
        for key in ("total_k2", "cutoff", "target", "width_GeV", "quark_mass_GeV",
                    "gluon_mass_GeV", "coupling"):
            if getattr(pf, key) != getattr(pi, key):
                raise ValueError("loop-current endpoints disagree on "+key)
        ef, ei, pp = map(float, (energy_out, energy_in, total_pplus_GeV))
        if not np.all(np.isfinite([ef, ei, pp])) or pp <= 0:
            raise ValueError("finite energies and positive total P+ required")
        self.outgoing, self.incoming = outgoing, incoming
        self.energy_out, self.energy_in, self.total_pplus_GeV = ef, ei, pp
        sf = {s.source.space.sector: s.source.space for s in outgoing.sectors}
        si = {s.source.space.sector: s.source.space for s in incoming.sectors}
        self.projector = QShellProjectorFamily(sf, si, inner_cutoffs={n: inner_cutoff for n in si},
            transfer=transfer, b_GeV=pi.width_GeV, charge=charge, quadrature_order=quadrature_order)
        self.sources_out, self.sources_in = _sources(outgoing), _sources(incoming)
        args = dict(inner_cutoff=inner_cutoff, b_GeV=pi.width_GeV,
                    quark_mass_GeV=pi.quark_mass_GeV, gluon_mass_GeV=pi.gluon_mass_GeV)
        self.resolvents_out = {n: ProjectedShellFreeResolvent(s, **args) for n, s in sf.items()}
        self.resolvents_in = {n: ProjectedShellFreeResolvent(s, **args) for n, s in si.items()}
        self.free_out = {s.source.space.sector: s.propagation for s in outgoing.sectors}
        self.free_in = {s.source.space.sector: s.propagation for s in incoming.sectors}

    def regulator_action(self, values, *, component):
        return self.projector.resolvent_boundary(values, component=component,
            resolve_incoming=lambda a: {n: self.resolvents_in[n].resolve(self.energy_in, v) for n, v in a.items()},
            resolve_outgoing=lambda a: {n: self.resolvents_out[n].resolve(self.energy_out, v) for n, v in a.items()},
            parent_incoming=lambda a: {n: self.free_in[n].apply(v) for n, v in a.items()},
            parent_outgoing=lambda a: {n: self.free_out[n].apply(v) for n, v in a.items()},
            total_pplus_GeV=self.total_pplus_GeV)

    def coefficients(self):
        """Two transverse components per order, BEFORE either retained norm."""
        nf, ni = len(self.outgoing.pilot.matrix), len(self.incoming.pilot.matrix)
        result = {p: np.zeros((2, nf, ni), complex) for p in (2, 3, 4)}
        joined = {n: np.column_stack([self.sources_in[p][n] for p in (1, 2)])
                  for n in self.projector.incoming}
        for j in (0, 1):
            acted = self.regulator_action(joined, component=j)
            for left in (1, 2):
                for right in (1, 2):
                    result[left+right][j] += sum(self.sources_out[left][n].conj().T
                        @a[:, (right-1)*ni:right*ni] for n, a in acted.items())
        return result
