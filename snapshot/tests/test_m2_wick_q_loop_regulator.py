"""Paired ordered Wick operators in the same canonical H2 Q-regulator engine."""
from functools import lru_cache
from dataclasses import replace

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent
from deuteron_wigner.quantum.m2_wick_h2_matching import WickMatchedFourthOrderH2
from test_m2_interacting_q_loop_regulator import model, scalar_vectors, dot, add, scale
from test_m2_wick_one_body import operators


def make(basef, basei, op, *, transfer=(.12, .16)):
    return InteractingQLoopRegulatorCurrent(
        WickMatchedFourthOrderH2(basef, operators=op),
        WickMatchedFourthOrderH2(basei, operators=op),
        inner_cutoffs={n: 0 for n in (*basei.spaces5, *basei.spaces6)},
        energy_out=1.39, energy_in=1.47, transfer=transfer, charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)


def bare(m):
    return InteractingQLoopRegulatorCurrent(m.outgoing, m.incoming,
        inner_cutoffs={n: v.inner_cutoff for n, v in m.projector.vertices.items()},
        energy_out=m.energy_out, energy_in=m.energy_in,
        transfer=m.projector.vertices["qqqgg"].transfer,
        charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16)


@lru_cache(maxsize=3)
def matched(target, jz, cutoff):
    return make(model(cutoff, jz, target), model(cutoff, 1, target), operators())


@pytest.fixture(scope="module", params=[("PROTON", 1, 2), ("PROTON", -1, 1), ("NEUTRON", -1, 1)])
def m(request):
    return matched(*request.param)


def insertion(m, a, endpoint):
    return {n: (m.wick_Q5[endpoint][n][:, None]*v if n in m.wick_Q5[endpoint] else np.zeros_like(v))
            for n, v in a.items()}


def increment(m, vf, vi):
    # Independent two placements, no full-W differencing or cached words.
    f, i = m._source_columns("outgoing", vf)[1], m._source_columns("incoming", vi)[1]
    ai = m.action("R", i, endpoint="incoming")
    af = m.action("R", f, endpoint="outgoing")
    si, sf = insertion(m, ai, "incoming"), insertion(m, af, "outgoing")
    return np.array([dot(f, m.regulator_action(si, component=j))
                     +dot(sf, m.regulator_action(i, component=j)) for j in (0, 1)])


@pytest.fixture(scope="module")
def currents(m):
    vf, vi = scalar_vectors(972), scalar_vectors(124)
    old = bare(m)
    return vf, vi, old, old.coefficients(outgoing_vectors=vf, incoming_vectors=vi), m.coefficients(
        outgoing_vectors=vf, incoming_vectors=vi)


def test_same_full_H2_Hamiltonian_and_both_norms_as_independent_Wick_model(m):
    for endpoint, energy in (("outgoing", 1.39), ("incoming", 1.47)):
        outer = m._model(endpoint)
        inner = WickMatchedFourthOrderH2(model(0, outer.pilot.jz_twice, outer.pilot.target),
                                         operators=m.wick_operators)
        expected = inner.snapshot(energy)
        np.testing.assert_allclose(m.effective_hamiltonian(endpoint=endpoint),
                                   expected.effective_hamiltonian, atol=3e-15)
        np.testing.assert_allclose(m.metric(endpoint=endpoint), expected.metric, atol=3e-15)


def test_only_order_four_return_current_changes_and_both_insertions_are_needed(m, currents):
    vf, vi, old, before, after = currents
    for p in (2, 3):
        np.testing.assert_array_equal(after[p], before[p])
    expected = increment(m, vf, vi)
    np.testing.assert_allclose(after[4]-before[4], expected, atol=3e-17)
    assert np.linalg.norm(expected) > 1e-10
    f, i = m._source_columns("outgoing", vf)[1], m._source_columns("incoming", vi)[1]
    ri = m.action("R", i, endpoint="incoming")
    left_only = np.array([dot(f, m.regulator_action(insertion(m, ri, "incoming"), component=j))
                          for j in (0, 1)])
    assert np.linalg.norm(expected-left_only) > 1e-11


def test_one_body_insertion_commutes_with_density_and_projector_but_is_not_zero(m):
    a = m._source_columns("incoming", scalar_vectors())[2]
    s = insertion(m, a, "incoming")
    assert sum(np.linalg.norm(v) for v in s.values()) > 1e-6
    for f in (m.projector.density, lambda v: m.projector.apply(v, component=0),
              lambda v: m.projector.apply(v, component=1)):
        lhs, rhs = f(s), insertion(m, f(a), "outgoing")
        for n in a:
            np.testing.assert_allclose(lhs[n], rhs[n], atol=2e-16)
    # The inner sandwich of (K S + S K) vanishes; this does NOT make
    # the moving-endpoint derivative of R S R vanish.
    for j in (0, 1):
        boundary = m.projector.hamiltonian_boundary(a,
            parent_incoming=lambda x: insertion(m, x, "incoming"),
            parent_outgoing=lambda x: insertion(m, x, "outgoing"),
            component=j, total_pplus_GeV=1.3)
        for v in boundary.values():
            np.testing.assert_allclose(v, 0., atol=2e-16)


def test_Q6_one_body_and_vertex_mass_not_silently_resummed(m):
    a = m._source_columns("incoming", scalar_vectors())[2]
    only_six = {n: np.zeros_like(v) if n in m.incoming.spaces5 else v for n, v in a.items()}
    for v in m.action("W", only_six, endpoint="incoming").values():
        np.testing.assert_array_equal(v, 0.)
    old = bare(m)
    for op in ("H0", "R", "U"):
        lhs, rhs = m.action(op, a, endpoint="incoming"), old.action(op, a, endpoint="incoming")
        for n in lhs:
            np.testing.assert_array_equal(lhs[n], rhs[n])
    assert m.incoming.pilot.quark_mass_GeV == .35
    assert m.paired_Wick_in_regulator_wrapper and not m.physical
    assert not m.full_H2_order4_regulator_matching_complete


def test_reversed_current_uses_identical_finite_conditions_at_both_ends(m, currents):
    vf, vi, _, _, c = currents
    reverse = InteractingQLoopRegulatorCurrent(
        m.input_models["incoming"], m.input_models["outgoing"],
        inner_cutoffs={n: v.inner_cutoff for n, v in m.projector.vertices.items()},
        energy_out=1.47, energy_in=1.39, transfer=(-.12, -.16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    r = reverse.coefficients(outgoing_vectors=vi, incoming_vectors=vf)
    for p in c:
        np.testing.assert_allclose(c[p], r[p].conj().swapaxes(1, 2), atol=3e-17)


def test_once_only_Wick_norms_agree_with_independent_matched_snapshots():
    m = matched("PROTON", -1, 1)
    vf, vi = scalar_vectors(927), scalar_vectors(33)
    f = WickMatchedFourthOrderH2(model(0, -1), operators=m.wick_operators).snapshot(1.39)
    i = WickMatchedFourthOrderH2(model(0, 1), operators=m.wick_operators).snapshot(1.47)
    expected = m.current(normalized=False,
        outgoing_vectors=f.canonical_snapshot.whitener@vf,
        incoming_vectors=i.canonical_snapshot.whitener@vi)
    np.testing.assert_allclose(m.current(normalized=True, outgoing_vectors=vf, incoming_vectors=vi),
                               expected, atol=2e-17)


def test_exact_zero_finite_subtraction_returns_the_preceding_regulator_model():
    basef, basei = model(1, -1), model(1)
    new = make(basef, basei, operators(subtract=True, muq=0., mug=0.))
    old = bare(new)
    vf, vi = scalar_vectors(645), scalar_vectors(887)
    a, b = new.coefficients(outgoing_vectors=vf, incoming_vectors=vi), old.coefficients(
        outgoing_vectors=vf, incoming_vectors=vi)
    for p in a:
        np.testing.assert_array_equal(a[p], b[p])
    for v in new.wick_Q5["incoming"].values():
        np.testing.assert_array_equal(v, 0.)


def test_finite_subtracted_regulator_increment_is_trace_independent_not_zero():
    basef, basei = model(1, -1), model(1)
    a = make(basef, basei, operators(subtract=True, width=.4))
    b = make(basef, basei, operators(subtract=True, width=1.6))
    vf, vi = scalar_vectors(645), scalar_vectors(887)
    for endpoint in ("outgoing", "incoming"):
        for n in a.wick_Q5[endpoint]:
            np.testing.assert_array_equal(a.wick_Q5[endpoint][n], b.wick_Q5[endpoint][n])
    ia, ib = increment(a, vf, vi), increment(b, vf, vi)
    np.testing.assert_array_equal(ia, ib)
    assert np.linalg.norm(ia) > 1e-10


def test_ordered_regulator_increment_scales_with_common_field_trace():
    basef, basei = model(1, -1), model(1)
    op = operators(muq=0., mug=0.)
    a = make(basef, basei, op)
    b = make(basef, basei, replace(op, coincident_trace_GeV2=3*op.coincident_trace_GeV2))
    vf, vi = scalar_vectors(48), scalar_vectors(884)
    np.testing.assert_allclose(increment(b, vf, vi), 3*increment(a, vf, vi), atol=3e-18)


def test_reject_mixed_or_disagreeing_Wick_prescriptions():
    base = model(1)
    paired = WickMatchedFourthOrderH2(base, operators=operators())
    other = WickMatchedFourthOrderH2(base, operators=operators(subtract=True))
    args = dict(inner_cutoffs={n: 0 for n in (*base.spaces5, *base.spaces6)},
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    for left, right in ((base, paired), (paired, base), (paired, other)):
        with pytest.raises(ValueError):
            InteractingQLoopRegulatorCurrent(left, right, **args)

def test_Wick_return_word_two_energy_Ward_with_independent_parent_vertices(m):
    q = m.projector
    a = m._source_columns("incoming", scalar_vectors(649))[1]
    def R(v, endpoint):
        return m.action("R", v, endpoint=endpoint)
    def C(v, endpoint):
        return R(insertion(m, R(v, endpoint), endpoint), endpoint)
    def plus(v):
        return R(q.density(R(v, "incoming")), "outgoing")
    def divergence(v):
        ri = R(v, "incoming")
        defect = add(m.action("H0", q.density(ri), endpoint="outgoing"),
                     scale(-1, q.density(m.action("H0", ri, endpoint="incoming"))))
        parent = R(defect, "outgoing")
        regulator = add(scale(.12, m.regulator_action(v, component=0)),
                        scale(.16, m.regulator_action(v, component=1)))
        return add(parent, scale(2*1.3, regulator))
    sr = insertion(m, R(a, "incoming"), "incoming")
    cp = add(plus(sr), R(insertion(m, plus(a), "outgoing"), "outgoing"))
    cd = add(divergence(sr), R(insertion(m, divergence(a), "outgoing"), "outgoing"))
    lhs = add(scale(1.39-1.47, cp), scale(-1, cd))
    rhs = add(q.density(C(a, "incoming")), scale(-1, C(q.density(a), "outgoing")))
    for n in a:
        np.testing.assert_allclose(lhs[n], rhs[n], atol=2e-16, rtol=2e-11)
