"""Actual canonical H2 word transport, not a source-image or toy-model test."""
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import (
    ShellCanonicalQQInsertion, FactorizedCanonicalQQSourceAction)
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2, current_between
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import (
    InteractingQLoopRegulatorCurrent, RETURN_WORDS)
from deuteron_wigner.quantum.m2_q5_loop_regulator_current import Q5LoopRegulatorCurrent


@lru_cache(maxsize=6)
def model(cutoff, jz=1, target="PROTON"):
    p = build_coupled_pilot(total_k2=9, cutoff=0, target=target, jz_twice=jz,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    r = build_shell_fock_reference(p, out_cutoff=cutoff)
    qq = ShellCanonicalQQInsertion(r, action=FactorizedCanonicalQQSourceAction(r))
    return FourthOrderH2(r, QQ_insertion=qq, six_body=CanonicalSixBodyExcursion(r, out_cutoff=cutoff))


@lru_cache(maxsize=2)
def control(jz):
    f, i = model(1, jz), model(1)
    return InteractingQLoopRegulatorCurrent(f, i,
        inner_cutoffs={n: 0 for n in (*i.spaces5, *i.spaces6)},
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)


def scalar_vectors(seed=823):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(37, 1))+1j*rng.normal(size=(37, 1))
    return a/np.linalg.norm(a)


def dot(a, b):
    return sum(a[n].conj().T@b[n] for n in a)


def add(a, b):
    return {n: a[n]+b[n] for n in a}


def scale(c, a):
    return {n: c*v for n, v in a.items()}


@pytest.fixture(scope="module", params=[1, -1])
def m(request):
    return control(request.param)


@pytest.mark.parametrize("endpoint", ["incoming", "outgoing"])
def test_unexpanded_H_and_complete_backflow_norm_match_inner_H2(m, endpoint):
    jz = m._model(endpoint).pilot.jz_twice
    e = m.energy_in if endpoint == "incoming" else m.energy_out
    inner = model(0, jz)
    snapshot = inner.snapshot(e)
    h = inner.pilot.matrix.copy()
    for p, a in m.return_coefficients(endpoint=endpoint).items():
        h += .3**p*a
    np.testing.assert_allclose(h, snapshot.effective_hamiltonian, atol=2e-15)
    np.testing.assert_allclose(m.metric(endpoint=endpoint), snapshot.metric, atol=3e-15)


def test_two_energy_density_words_match_independent_full_H2_backflow(m):
    vf, vi = scalar_vectors(51), scalar_vectors(63)
    source_f = m._source_columns("outgoing", vf)
    source_i = m._source_columns("incoming", vi)
    def plus(a):
        ri = m.action("R", a, endpoint="incoming")
        return m.action("R", m.projector.density(ri), endpoint="outgoing")
    value = np.zeros((1, 1), complex)
    for l, r, order, word in RETURN_WORDS:
        a = m.gauged_word(word, source_i[r], vertices={"R": plus})
        value += .3**(l+r+order)*dot(source_f[l], a)
    f, i = model(0, m.outgoing.pilot.jz_twice).snapshot(m.energy_out), model(0).snapshot(m.energy_in)
    expected = current_between(f, i, (.12, .16), charge="electromagnetic", normalized=False)
    expected -= good_current_density_between(f.owner.pilot, i.owner.pilot, (.12, .16), charge="electromagnetic")
    np.testing.assert_allclose(value, vf.conj().T@expected@vi, atol=1e-16)


@pytest.mark.parametrize("word", [("R",), ("R", "U", "R"), ("R", "W", "R"),
                                  ("R", "U", "R", "U", "R")])
def test_finite_parent_two_energy_Ward_for_each_interacting_word(m, word):
    # Independently use H_f rho-rho H_i for the CONTRACTED outer-parent
    # vertices, including U/W. This is an algebra check, not a prescription
    # for a transverse physical parent current.
    q, pp = m.projector, m.total_pplus_GeV
    a = m._source_columns("incoming", scalar_vectors())[2]
    def defect(op, v):
        return add(m.action(op, q.density(v), endpoint="outgoing"),
                   scale(-1, q.density(m.action(op, v, endpoint="incoming"))))
    def plus(v):
        return m.action("R", q.density(m.action("R", v, endpoint="incoming")), endpoint="outgoing")
    def rdiv(v):
        parent = m.action("R", defect("H0", m.action("R", v, endpoint="incoming")), endpoint="outgoing")
        reg = add(scale(.12, m.regulator_action(v, component=0)),
                  scale(.16, m.regulator_action(v, component=1)))
        return add(parent, scale(2*pp, reg))
    jplus = m.gauged_word(word, a, vertices={"R": plus})
    div = m.gauged_word(word, a, vertices={"R": rdiv,
        "U": lambda v: defect("U", v), "W": lambda v: defect("W", v)})
    lhs = add(scale(m.energy_out-m.energy_in, jplus), scale(-1, div))
    rhs = add(q.density(m.word_action(word, a, endpoint="incoming")),
              scale(-1, m.word_action(word, q.density(a), endpoint="outgoing")))
    for n in a:
        np.testing.assert_allclose(lhs[n], rhs[n], atol=2e-16, rtol=2e-11)


@pytest.mark.parametrize("word", [("R", "U", "R"), ("R", "W", "R"),
                                  ("R", "U", "R", "U", "R")])
def test_explicit_projected_vertex_gauging_equals_unprojected_word(m, word):
    q, pp = m.projector, m.total_pplus_GeV
    a = m._source_columns("incoming", scalar_vectors())[2]
    def projected(op, a, endpoint):
        if op == "R":
            return m.action(op, a, endpoint=endpoint)
        return q.project(m.action(op, q.project(a, endpoint=endpoint), endpoint=endpoint), endpoint=endpoint)
    def vertex(op, a, j):
        # Full (Pi X Pi)' = K X_i Pi_i + Pi_f X_f K.
        if op == "R":
            return m.regulator_action(a, component=j)
        left = q.apply(m.action(op, q.project(a, endpoint="incoming"), endpoint="incoming"), component=j)
        right = q.project(m.action(op, q.apply(a, component=j), endpoint="outgoing"), endpoint="outgoing")
        return scale(1/(2*pp), add(left, right))
    for j in (0, 1):
        v = a
        tangent = {n: np.zeros((s.dimension, 1), complex) for n, s in q.outgoing.items()}
        for op in reversed(word):
            tangent = add(projected(op, tangent, "outgoing"), vertex(op, v, j))
            v = projected(op, v, "incoming")
        actual = m.gauged_word(word, a, vertices={"R": lambda v: m.regulator_action(v, component=j)})
        for n in a:
            np.testing.assert_allclose(actual[n], tangent[n], atol=2e-17, rtol=2e-11)


def test_low_orders_reduce_to_preceding_actual_Q5_loop(m):
    old = Q5LoopRegulatorCurrent(m.outgoing.reference, m.incoming.reference,
        inner_cutoff=0, energy_out=m.energy_out, energy_in=m.energy_in,
        transfer=(.12, .16), charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16)
    vf, vi = scalar_vectors(44), scalar_vectors(45)
    new = m.coefficients(outgoing_vectors=vf, incoming_vectors=vi)
    previous = old.coefficients()
    for p in (2, 3):
        np.testing.assert_allclose(new[p], [vf.conj().T@a@vi for a in previous[p]], atol=2e-17)
    # Order four is compared through the full H2 identities, not assumed equal.
    assert m.Q_only_H2_order4_transport_complete
    assert not m.full_H2_order4_regulator_matching_complete
    assert not m.common_field_regulator_complete and not m.physical


def test_reverse_transfer_spin_and_energy_reciprocity(m):
    rev = InteractingQLoopRegulatorCurrent(m.incoming, m.outgoing,
        inner_cutoffs={n: 0 for n in m.projector.incoming},
        energy_out=m.energy_in, energy_in=m.energy_out, transfer=(-.12, -.16),
        charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16)
    vf, vi = scalar_vectors(23), scalar_vectors(85)
    a = m.coefficients(outgoing_vectors=vf, incoming_vectors=vi)
    b = rev.coefficients(outgoing_vectors=vi, incoming_vectors=vf)
    for p in a:
        np.testing.assert_allclose(a[p], b[p].conj().swapaxes(1, 2), atol=2e-17)


def test_two_local_norms_are_applied_once_and_keep_distinct_energies(m):
    vf, vi = scalar_vectors(23), scalar_vectors(85)
    sf = model(0, m.outgoing.pilot.jz_twice).snapshot(m.energy_out)
    si = model(0).snapshot(m.energy_in)
    expected = m.current(normalized=False, outgoing_vectors=sf.whitener@vf, incoming_vectors=si.whitener@vi)
    actual = m.current(normalized=True, outgoing_vectors=vf, incoming_vectors=vi)
    np.testing.assert_allclose(actual, expected, atol=2e-17)
    assert np.linalg.norm(sf.metric-si.metric) > 1e-7


def test_invalid_inputs_fail_without_silent_current_or_inner_cutoff_choice(m):
    for kwargs in ({"normalized": None}, {"normalized": True, "incoming_vectors": np.ones(37)}):
        with pytest.raises(ValueError):
            m.current(**kwargs)
    with pytest.raises(ValueError):
        m.return_coefficients(endpoint="missing")
    with pytest.raises(ValueError):
        m.return_coefficients(endpoint="incoming", norm=None)
    a = m.sources["incoming"][1]
    with pytest.raises(ValueError):
        m.gauged_word(("R", "unknown"), a, vertices={})
    with pytest.raises(ValueError):
        m.gauged_word(("R",), a, vertices={"unknown": lambda x: x})

@pytest.fixture(scope="module", params=[0, 1])
def radial(request):
    i = model(2)
    cuts = {n: 0 for n in i.spaces5}
    cuts.update({n: request.param for n in i.spaces6})
    return InteractingQLoopRegulatorCurrent(i, i, inner_cutoffs=cuts,
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)


@pytest.mark.parametrize("word", [("R", "U", "R"), ("R", "W", "R"),
                                  ("R", "U", "R", "U", "R")])
def test_radial_outer_shell_and_independent_Q6_cutoff(radial, word):
    test_finite_parent_two_energy_Ward_for_each_interacting_word(radial, word)
    test_explicit_projected_vertex_gauging_equals_unprojected_word(radial, word)


def test_nonzero_radial_interaction_and_double_counting_witness(radial):
    m, q = radial, radial.projector
    a = m._source_columns("incoming", scalar_vectors(904))[1]
    word = ("R", "W", "R")
    b = m.gauged_word(word, a, vertices={"R": lambda v: m.regulator_action(v, component=0)})
    # Incorrectly add the projected-W boundary ON TOP of the unprojected
    # word derivative. Its nonzero extra term proves a real double count.
    extra = m.gauged_word(word, a, vertices={"W": lambda v:
        q.hamiltonian_boundary(v,
            parent_incoming=lambda x: m.action("W", x, endpoint="incoming"),
            parent_outgoing=lambda x: m.action("W", x, endpoint="outgoing"),
            component=0, total_pplus_GeV=1.3)})
    norm = sum(np.linalg.norm(v) for v in b.values())
    excess = sum(np.linalg.norm(v) for v in extra.values())
    assert norm > 1e-10
    assert excess > 512*np.finfo(float).eps*norm

def test_cached_coefficients_equal_direct_unfactorized_Leibniz(m):
    vf, vi = scalar_vectors(15), scalar_vectors(16)
    left, right = m._source_columns("outgoing", vf), m._source_columns("incoming", vi)
    direct = {p: np.zeros((2, 1, 1), complex) for p in (2, 3, 4)}
    for l, r, order, word in RETURN_WORDS:
        for j in (0, 1):
            a = m.gauged_word(word, right[r],
                vertices={"R": lambda v, j=j: m.regulator_action(v, component=j)})
            direct[l+r+order][j] += dot(left[l], a)
    cached = m.coefficients(outgoing_vectors=vf, incoming_vectors=vi)
    for p in direct:
        np.testing.assert_allclose(cached[p], direct[p], atol=2e-17, rtol=2e-11)
