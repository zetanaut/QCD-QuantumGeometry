"""Paired one-body Wick/counterterm transport through the common H2 order g^4.

The same explicit operators alter PP at g^2, the full Q5 backflow at g^3,
the H/norm/current returns at g^4, and direct minus-current insertions.
Q6 one-body insertions first contribute at g^6 and are not imported here.
Pairs the known ORDERED H/J composites. The additional gauged-contraction
and propagating-loop vertices, especially at nonzero longitudinal transfer,
are not supplied by a normal-order subtraction identity.
This is NOT common continuum renormalization or physical finite matching.
"""
from dataclasses import dataclass, replace

import numpy as np

from .m2_fourth_order_h2 import FourthOrderH2, FourthOrderSnapshot, _norm_inverse_sqrt
from .m2_shell_free_propagation import ShellFreePropagation
from .m2_wick_one_body import WickOneBody
from .m2_wick_current_projection import retained_wick_minus_between, shell_wick_minus_between


@dataclass(frozen=True)
class WickMatchedSnapshot:
    owner: object
    canonical_snapshot: object

    @property
    def energy_GeV2(self):
        return self.canonical_snapshot.energy_GeV2

    @property
    def effective_hamiltonian(self):
        return self.canonical_snapshot.effective_hamiltonian

    @property
    def metric(self):
        return self.canonical_snapshot.metric

    @property
    def matrix(self):
        return self.canonical_snapshot.matrix

    def current(self, transfer, *, charge):
        """Matched plus density; the one-body Wick/kinetic source has no direct J+."""
        return self.canonical_snapshot.current(transfer, charge=charge)


class WickMatchedFourthOrderH2:
    physical = False
    common_regulator_renormalization_complete = False

    def __init__(self, base, *, operators):
        if not isinstance(base, FourthOrderH2) or not isinstance(operators, WickOneBody):
            raise ValueError("explicit common-order canonical H2 and paired WickOneBody required")
        self.base, self.operators, self.pilot = base, operators, base.pilot
        self.pp = operators.retained_mass_coefficient(self.pilot)
        self.qq = {name: operators.shell_mass_diagonal(space)
                   for name, space in base.spaces5.items()}
        self.propagation = {
            name: ShellFreePropagation(space, b_GeV=self.pilot.width_GeV,
                quark_mass_GeV=self.pilot.quark_mass_GeV, gluon_mass_GeV=self.pilot.gluon_mass_GeV)
            for name, space in base.spaces5.items()
        }

    def snapshot(self, energy):
        return self.match_snapshot(self.base.snapshot(energy))

    def match_snapshot(self, old):
        """Reuse an exact-source snapshot for explicit subtraction comparisons."""
        if not isinstance(old, FourthOrderSnapshot) or old.owner is not self.base:
            raise ValueError("snapshot must belong to this exact canonical base")
        e, g = old.energy_GeV2, self.pilot.coupling
        a1, a2, c3 = old.five_wave_coefficients
        extra = {}
        for name, propagation in self.propagation.items():
            v1 = self.base.qq.sources_by_power[1][name]
            checked = propagation.resolve(e, v1)
            if not np.allclose(checked, a1[name], rtol=2e-13, atol=2e-15):
                raise ArithmeticError("base and Wick reference disagree on the full free-Q resolvent")
            extra[name] = propagation.resolve(e, self.qq[name][:, None]*a1[name])
        h = old.effective_hamiltonian + g*g*self.pp
        h += g**4*sum(self.base.qq.sources_by_power[1][n].conj().T@extra[n] for n in extra)
        metric = old.metric + g**4*sum(
            a1[n].conj().T@extra[n]+extra[n].conj().T@a1[n] for n in extra)
        for name, value in (("Hamiltonian", h), ("metric", metric)):
            if not np.all(np.isfinite(value)) or np.linalg.norm(value-value.conj().T) > 3e-11:
                raise ArithmeticError("paired Wick "+name+" failed Hermiticity")
        whiten = _norm_inverse_sqrt(metric)
        local = whiten@(h+e*(metric-np.eye(len(h))))@whiten
        tree = replace(old, effective_hamiltonian=h, metric=metric, matrix=local,
            whitener=whiten, five_wave_coefficients=(a1, a2, {n: c3[n]+extra[n] for n in c3}))
        return WickMatchedSnapshot(self, tree)


def _endpoints(outgoing, incoming):
    if not isinstance(outgoing, WickMatchedSnapshot) or not isinstance(incoming, WickMatchedSnapshot):
        raise ValueError("paired Wick H2 snapshots required; bare snapshots omit direct partners")
    if outgoing.owner.operators != incoming.owner.operators:
        raise ValueError("identical field trace and finite subtraction conditions at both endpoints required")
    # Tree-current implementations additionally validate physical parameters and coordinates.
    return outgoing.canonical_snapshot, incoming.canonical_snapshot


def direct_minus_coefficients_between(outgoing, incoming, transfer, *, total_pplus_GeV,
                                       charge, loop_flavor_charges):
    """Direct insertion coefficients at orders 2 and 4, before both norms.

    W_PP at g^2 and A1_f-dagger W_55 A1_i at g^4. These supplement the
    canonical tree current evaluated with the MODIFIED C3 backflow.
    They must not replace that backflow or be normalized at only one end.
    """
    f, i = _endpoints(outgoing, incoming)
    pf, pi = f.owner.pilot, i.owner.pilot
    operators = incoming.owner.operators
    args = dict(operators=operators, total_pplus_GeV=total_pplus_GeV,
                charge=charge, loop_flavor_charges=loop_flavor_charges)
    result = {2: retained_wick_minus_between(pf, pi, transfer, **args),
              4: np.zeros((len(pf.matrix), len(pi.matrix)), complex)}
    for name, space in i.owner.spaces5.items():
        acted = shell_wick_minus_between(f.owner.spaces5[name], space,
            i.five_wave_coefficients[0][name], transfer, b_GeV=pi.width_GeV, **args)
        result[4] += f.five_wave_coefficients[0][name].conj().T@acted
    return result


def minus_coefficients_between(outgoing, incoming, transfer, *, total_pplus_GeV,
                                charge, loop_flavor_charges, mean_total_transverse=(0., 0.)):
    from .m2_fourth_order_minus_current import minus_coefficients_between as tree_coefficients
    f, i = _endpoints(outgoing, incoming)
    result = tree_coefficients(f, i, transfer, total_pplus_GeV=total_pplus_GeV,
                              charge=charge, mean_total_transverse=mean_total_transverse)
    direct = direct_minus_coefficients_between(outgoing, incoming, transfer,
        total_pplus_GeV=total_pplus_GeV, charge=charge, loop_flavor_charges=loop_flavor_charges)
    for order, value in direct.items():
        result[order] += value
    return result


def minus_current_between(outgoing, incoming, transfer, *, total_pplus_GeV,
                           charge, loop_flavor_charges, normalized,
                           mean_total_transverse=(0., 0.)):
    if not isinstance(normalized, (bool, np.bool_)):
        raise ValueError("explicit normalization choice required")
    f, i = _endpoints(outgoing, incoming)
    coefficients = minus_coefficients_between(outgoing, incoming, transfer,
        total_pplus_GeV=total_pplus_GeV, charge=charge, loop_flavor_charges=loop_flavor_charges,
        mean_total_transverse=mean_total_transverse)
    g = incoming.owner.pilot.coupling
    result = sum(g**n*value for n, value in coefficients.items())
    return f.whitener@result@i.whitener if normalized else result


def plus_current_between(outgoing, incoming, transfer, *, charge, normalized):
    from .m2_fourth_order_h2 import current_between
    f, i = _endpoints(outgoing, incoming)
    return current_between(f, i, transfer, charge=charge, normalized=normalized)


def transverse_current_between(outgoing, incoming, transfer, *, total_pplus_GeV, charge, normalized):
    from .m2_fourth_order_transverse_current import transverse_current_between as tree_current
    f, i = _endpoints(outgoing, incoming)
    return tree_current(f, i, transfer, total_pplus_GeV=total_pplus_GeV,
                        charge=charge, normalized=normalized)
