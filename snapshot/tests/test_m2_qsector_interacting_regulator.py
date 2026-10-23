"""Interacting cutoff boundary on the actual coupled canonical Q5/Q6 actions.

Each source coefficient is checked separately. This is an identity for the
declared finite outer parent, not common continuum loop renormalization.
"""
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_shell_projector_current import QShellProjectorFamily
from deuteron_wigner.quantum.m2_shell_free_propagation import ShellFreePropagation
from deuteron_wigner.quantum.m2_five_body_factorized import FiveBodyContactAction
from deuteron_wigner.quantum.m2_pair_current_factorized import FactorizedPairCurrentAction
from deuteron_wigner.quantum.m2_six_body_factorized import FactorizedSixBodyCubicAction
from test_m2_shell_projector_current import space, random_columns
from test_m2_wick_one_body import operators


class Parent:
    def __init__(self, jz):
        self.five = {n: space(n, 2, jz) for n in ("qqqgg", "qqqq-qbar")}
        self.contacts = FiveBodyContactAction(self.five, b_GeV=.4)
        self.pair = FactorizedPairCurrentAction(self.five, b_GeV=.4)
        self.cubic = FactorizedSixBodyCubicAction(self.five, out_cutoff=2, b_GeV=.4, mass_GeV=.35)
        self.spaces = {**self.five, **self.cubic.outgoing}
        self.free = {n: ShellFreePropagation(s, b_GeV=.4, quark_mass_GeV=.35,
                     gluon_mass_GeV=0.) for n, s in self.spaces.items()}
        wick = operators()
        self.self_diagonal = {n: wick.shell_mass_diagonal(s) for n, s in self.spaces.items()}

    def action(self, a, component):
        if component == "free":
            return {n: self.free[n].apply(v) for n, v in a.items()}
        if component == "wick":
            return {n: self.self_diagonal[n][:, None]*v for n, v in a.items()}
        result = {n: np.zeros_like(v) for n, v in a.items()}
        if component == "quartic":
            x = {n: a[n] for n in self.five}
            c, p = self.contacts.apply(x), self.pair.apply(x)
            result.update({n: c[n]+p[n] for n in self.five})
        elif component == "cubic":
            result.update(self.cubic.apply({n: a[n] for n in self.five}))
            result.update(self.cubic.adjoint({n: a[n] for n in self.cubic.outgoing}))
        else:
            raise ValueError(component)
        return result


@lru_cache(maxsize=2)
def parent(jz):
    return Parent(jz)


@pytest.fixture(scope="module", params=[1, -1])
def endpoints(request):
    i, f = parent(1), parent(request.param)
    q = QShellProjectorFamily(f.spaces, i.spaces,
        inner_cutoffs={n: 0 for n in i.spaces}, transfer=(.12, .16),
        b_GeV=.4, charge="electromagnetic", quadrature_order=16)
    return f, i, q


@pytest.mark.parametrize("component", ["free", "quartic", "cubic", "wick"])
def test_coupled_projector_product_rule_for_each_actual_source(endpoints, component):
    f, i, q = endpoints
    a = {n: random_columns(s, 963) for n, s in i.spaces.items()}
    v = q.project(a, endpoint="incoming")
    Hf, Hi = lambda a: f.action(a, component), lambda a: i.action(a, component)
    rho_v = q.density(v)
    rho_p = q.project(rho_v, endpoint="outgoing")
    h_rho = q.project(Hf(rho_p), endpoint="outgoing")
    rho_h = q.project(q.density(q.project(Hi(v), endpoint="incoming")), endpoint="outgoing")
    left, right = Hf(rho_v), q.density(Hi(v))
    parent_defect = q.project({n: left[n]-right[n] for n in v}, endpoint="outgoing")
    expected = {n: h_rho[n]-rho_h[n]-parent_defect[n] for n in v}
    boundary = [q.hamiltonian_boundary(v, parent_incoming=Hi, parent_outgoing=Hf,
                component=j, total_pplus_GeV=1.3) for j in (0, 1)]
    for n in v:
        actual = 2*1.3*(.12*boundary[0][n]+.16*boundary[1][n])
        np.testing.assert_allclose(actual, expected[n], atol=5e-16)
        np.testing.assert_array_equal(boundary[0][n][~q.vertices[n].mask_out], 0.)
    norm = sum(np.linalg.norm(b[n]) for b in boundary for n in v)
    if component == "cubic" or (component == "quartic" and f.spaces["qqqgg"].jz_twice == 1):
        assert norm > 1e-6
    if component in ("free", "wick"):
        assert norm < 2e-14


@pytest.mark.parametrize("component", ["quartic", "cubic"])
def test_conversions_cannot_be_replaced_with_diagonal_sector_actions(endpoints, component):
    f, i, q = endpoints
    # A real Q5 gluonic input has regulator-current outputs in sea/Q6 spaces.
    a = {n: np.zeros((s.dimension, 1), complex) for n, s in i.spaces.items()}
    a["qqqgg"] = random_columns(i.spaces["qqqgg"], 54)
    b = [q.hamiltonian_boundary(a,
        parent_incoming=lambda v: i.action(v, component),
        parent_outgoing=lambda v: f.action(v, component),
        component=j, total_pplus_GeV=1.3) for j in (0, 1)]
    targets = ("qqqq-qbar",) if component == "quartic" else tuple(i.cubic.outgoing)
    norm = sum(np.linalg.norm(x[n]) for x in b for n in targets)
    if component == "cubic" or f.spaces["qqqgg"].jz_twice == 1:
        assert norm > 1e-8
    # No additional physical Q6 g^2 insertion has been added to the H2 model.


def test_family_rejects_missing_sectors_and_wrong_shapes():
    s = space("qqqgg", 1, 1)
    with pytest.raises(ValueError):
        QShellProjectorFamily({"qqqgg": s}, {"qqqgg": s}, inner_cutoffs={},
            transfer=(0., 0.), b_GeV=.4, charge="baryon", quadrature_order=16)
    f = QShellProjectorFamily({"qqqgg": s}, {"qqqgg": s}, inner_cutoffs={"qqqgg": 0},
        transfer=(0., 0.), b_GeV=.4, charge="baryon", quadrature_order=16)
    with pytest.raises(ValueError):
        f.apply({}, component=0)
    with pytest.raises(ValueError):
        f.project({"qqqgg": np.ones(s.dimension)}, endpoint="incoming")
    with pytest.raises(ValueError):
        f.project({"qqqgg": np.ones((s.dimension, 1))}, endpoint="undefined")
