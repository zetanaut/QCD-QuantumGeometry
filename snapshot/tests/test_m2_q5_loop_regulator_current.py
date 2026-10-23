"""Actual canonical-source Q5 return loops, not arbitrary matrix surrogates."""
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_q5_loop_regulator_current import Q5LoopRegulatorCurrent
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between


@lru_cache(maxsize=2)
def reference(jz):
    p = build_coupled_pilot(total_k2=9, cutoff=0, target="PROTON", jz_twice=jz,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    return build_shell_fock_reference(p, out_cutoff=2)


@pytest.fixture(scope="module", params=[1, -1])
def control(request):
    m = Q5LoopRegulatorCurrent(reference(request.param), reference(1), inner_cutoff=0,
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    return m, m.coefficients()


def test_each_order_against_unexpanded_source_polynomial(control):
    m, coefficients = control
    rng = np.random.default_rng(894)
    vf, vi = rng.normal(size=37)+1j*rng.normal(size=37), rng.normal(size=37)+1j*rng.normal(size=37)
    vf, vi = vf/np.linalg.norm(vf), vi/np.linalg.norm(vi)
    out = {p: {n: a@vf[:, None] for n, a in sources.items()} for p, sources in m.sources_out.items()}
    inc = {p: {n: a@vi[:, None] for n, a in sources.items()} for p, sources in m.sources_in.items()}
    nodes = np.linspace(-.7, .7, 7)
    values = []
    for g in nodes:
        left = {n: g*out[1][n]+g*g*out[2][n] for n in m.projector.outgoing}
        right = {n: g*inc[1][n]+g*g*inc[2][n] for n in m.projector.incoming}
        values.append([sum(np.vdot(left[n], a) for n, a in m.regulator_action(right, component=j).items())
                       for j in (0, 1)])
    polynomial = np.polynomial.polynomial.polyfit(nodes, np.asarray(values), 4)
    np.testing.assert_allclose(polynomial[:2], 0., atol=2e-18)
    for p in (2, 3, 4):
        np.testing.assert_allclose(polynomial[p],
            [np.vdot(vf, a@vi) for a in coefficients[p]], atol=8e-18)
    assert not m.physical and not m.full_H2_order4_regulator_matching_complete


def test_reversed_transfer_energy_and_both_spin_endpoints(control):
    m, c = control
    reverse = Q5LoopRegulatorCurrent(m.incoming, m.outgoing, inner_cutoff=0,
        energy_out=m.energy_in, energy_in=m.energy_out, transfer=(-.12, -.16),
        charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16).coefficients()
    for p in c:
        np.testing.assert_allclose(c[p], reverse[p].conj().swapaxes(1, 2), atol=5e-17)


def test_unprojected_outer_source_tails_are_required(control):
    m, c = control
    q = m.projector
    a = q.project(m.sources_in[1], endpoint="incoming")
    left = q.project(m.sources_out[1], endpoint="outgoing")
    wrong = np.array([sum(left[n].conj().T@v for n, v in m.regulator_action(a, component=j).items())
                      for j in (0, 1)])
    assert np.linalg.norm(c[2]-wrong) > 1e-7


def test_cubic_loop_ward_residual_with_independent_parent_current(control):
    m, coefficients = control
    q, P = m.projector, m.total_pplus_GeV
    v = m.sources_in[1]
    ri = {n: m.resolvents_in[n].resolve(m.energy_in, a) for n, a in v.items()}
    rho_ri, rho_v = q.density(ri), q.density(v)
    jp = {n: m.resolvents_out[n].resolve(m.energy_out, a) for n, a in rho_ri.items()}
    rhs = {n: rho_ri[n]-m.resolvents_out[n].resolve(m.energy_out, rho_v[n]) for n in v}
    jt = []
    for j in (0, 1):
        parent = {n: m.resolvents_out[n].resolve(m.energy_out,
            shell_transverse_current_between(q.outgoing[n], q.incoming[n], a, (.12, .16),
                b_GeV=.4, total_pplus_GeV=P, charge="electromagnetic", component=j))
            for n, a in ri.items()}
        jt.append(sum(m.sources_out[1][n].conj().T@a for n, a in parent.items())+coefficients[2][j])
    lhs = (m.energy_out-m.energy_in)*sum(m.sources_out[1][n].conj().T@a for n, a in jp.items())
    lhs -= 2*P*(.12*jt[0]+.16*jt[1])
    expected = sum(m.sources_out[1][n].conj().T@a for n, a in rhs.items())
    np.testing.assert_allclose(lhs, expected, atol=8e-17)
