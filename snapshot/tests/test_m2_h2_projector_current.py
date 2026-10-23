"""P/source plus Q cutoff transport on actual canonical H2 spaces."""
from functools import lru_cache
from dataclasses import replace

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_h2_projector_current import H2ProjectorCurrent
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent
from deuteron_wigner.quantum.m2_wick_h2_matching import WickMatchedFourthOrderH2
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_galerkin_projector_current import interacting_galerkin_boundary_between
from test_m2_interacting_q_loop_regulator import model, scalar_vectors
from test_m2_interacting_projector_current import pilot
from test_m2_wick_one_body import operators


@lru_cache(maxsize=3)
def control(inner_Q=1, wick=False):
    f, i = model(2, -1), model(2, 1)
    if wick:
        op = operators(subtract=True)
        f, i = WickMatchedFourthOrderH2(f, operators=op), WickMatchedFourthOrderH2(i, operators=op)
    q = InteractingQLoopRegulatorCurrent(f, i,
        inner_cutoffs={n: inner_Q for n in ("qqqgg", "qqqq-qbar", "qqqggg", "qqqq-qbar-g")},
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    return H2ProjectorCurrent(q, parent_outgoing=pilot(1, jz=-1),
                             parent_incoming=pilot(1), quadrature_order=16)


@pytest.fixture(scope="module", params=[False, True])
def c(request):
    return control(1, request.param)


@pytest.fixture(scope="module")
def coefficients(c):
    vf, vi = scalar_vectors(31), scalar_vectors(772)
    return vf, vi, c.loop_coefficients(outgoing_vectors=vf, incoming_vectors=vi)


def test_return_boundary_is_nonzero_and_retains_all_orders(c, coefficients):
    vf, vi, result = coefficients
    assert np.linalg.norm(result[2]) > 1e-9
    assert np.linalg.norm(result[4]) > 1e-10
    assert c.finite_P_Q_oscillator_projector_transport_complete
    assert not c.common_field_regulator_complete and not c.complete_physical_four_current
    assert not c.physical


def test_finite_parent_Ward_identity_for_each_complete_return_order(c, coefficients):
    vf, vi, result = coefficients
    pf, pi = c.embeddings["outgoing"], c.embeddings["incoming"]
    rho = good_current_density_between(c.parents["outgoing"], c.parents["incoming"],
                                       (.12, .16), charge="electromagnetic")
    rp = pf.conj().T@rho@pi
    # Four independent bilinears of the actual outer self-energy. This
    # avoids defining a physical current from a divided continuity defect.
    hf_rho = c.parent_return_bilinears(endpoint="outgoing", left=pf@vf, right=pf@rp@vi)
    rho_hi = c.parent_return_bilinears(endpoint="incoming", left=pi@rp.conj().T@vf, right=pi@vi)
    parent_f = c.parent_return_bilinears(endpoint="outgoing", left=pf@vf, right=rho@pi@vi)
    parent_i = c.parent_return_bilinears(endpoint="incoming", left=rho.conj().T@pf@vf, right=pi@vi)
    for p in result:
        expected = hf_rho[p]-rho_hi[p]-parent_f[p]+parent_i[p]
        actual = 2*1.3*(.12*result[p][0]+.16*result[p][1])
        np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=3e-11)


def test_retained_H_boundary_agrees_with_existing_exact_nested_implementation(c, coefficients):
    vf, vi, _ = coefficients
    q = c.Q_current
    f, i = c.parents["outgoing"], c.parents["incoming"]
    if q.paired_Wick_in_regulator_wrapper:
        f = replace(f, matrix=f.matrix+.3**2*q.wick_operators.retained_mass_coefficient(f))
        i = replace(i, matrix=i.matrix+.3**2*q.wick_operators.retained_mass_coefficient(i))
    full = interacting_galerkin_boundary_between(q.outgoing.pilot, q.incoming.pilot,
        parent_outgoing=f, parent_incoming=i, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    expected = np.array([vf.conj().T@a@vi for a in full])
    np.testing.assert_allclose(c.retained_current(outgoing_vectors=vf, incoming_vectors=vi),
                               expected, atol=2e-15)


def test_reversed_P_source_current_energy_spin_and_transfer(c, coefficients):
    vf, vi, result = coefficients
    q = c.Q_current
    r = InteractingQLoopRegulatorCurrent(q.input_models["incoming"], q.input_models["outgoing"],
        inner_cutoffs={n: v.inner_cutoff for n, v in q.projector.vertices.items()},
        energy_out=1.47, energy_in=1.39, transfer=(-.12, -.16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=16)
    rev = H2ProjectorCurrent(r, parent_outgoing=c.parents["incoming"],
        parent_incoming=c.parents["outgoing"], quadrature_order=16)
    actual = rev.loop_coefficients(outgoing_vectors=vi, incoming_vectors=vf)
    for p in result:
        np.testing.assert_allclose(actual[p], result[p].conj().swapaxes(1, 2), atol=3e-16)


def test_zero_inner_P_Q_selection_does_not_delete_raw_source_tails():
    c = control(0)
    vf, vi = scalar_vectors(63), scalar_vectors(903)
    result = c.loop_coefficients(outgoing_vectors=vf, incoming_vectors=vi)
    for value in result.values():
        np.testing.assert_allclose(value, 0., atol=3e-17)
    high = c.vertices[0]@c.embeddings["incoming"]@vi
    sources = c.source_columns("outgoing", high)
    assert sum(np.linalg.norm(a) for src in sources.values() for a in src.values()) > 1e-6
    for src in sources.values():
        projected = c.Q_current.projector.project(src, endpoint="outgoing")
        for a in projected.values():
            np.testing.assert_allclose(a, 0., atol=3e-14)


def test_all_cutoff_terms_are_added_before_the_two_independent_norms():
    c = control(1, True)
    q = c.Q_current
    vf, vi = scalar_vectors(156), scalar_vectors(984)
    from deuteron_wigner.quantum.m2_fourth_order_h2 import _norm_inverse_sqrt
    wf, wi = _norm_inverse_sqrt(q.metric(endpoint="outgoing")), _norm_inverse_sqrt(q.metric(endpoint="incoming"))
    args = dict(outgoing_vectors=wf@vf, incoming_vectors=wi@vi)
    pc, qc = c.loop_coefficients(**args), q.coefficients(**args)
    expected = c.retained_current(**args)+sum(.3**p*(pc[p]+qc[p]) for p in pc)
    np.testing.assert_allclose(c.current(normalized=True, outgoing_vectors=vf, incoming_vectors=vi),
                               expected, atol=3e-16)


def test_parent_parameter_and_endpoint_validation():
    c = control()
    with pytest.raises(ValueError):
        H2ProjectorCurrent(c.Q_current, parent_outgoing=pilot(1, "NEUTRON", -1),
                           parent_incoming=pilot(1), quadrature_order=16)
    with pytest.raises(ValueError):
        c.parent_return_bilinears(endpoint="undefined", left=np.ones((204, 1)), right=np.ones((204, 1)))
    with pytest.raises(ValueError):
        c.current(normalized=None, outgoing_vectors=scalar_vectors(), incoming_vectors=scalar_vectors())

def test_inner_source_return_and_energy_derivative_binding_with_unequal_probes(c):
    from copy import copy
    q = c.Q_current
    left = np.column_stack((scalar_vectors(324), scalar_vectors(512)))
    right = scalar_vectors(416)
    for endpoint in ("outgoing", "incoming"):
        p = c.embeddings[endpoint]
        src = c.source_columns(endpoint, p@left)
        expected_src = q._source_columns(endpoint, left)
        for power in src:
            for n in src[power]:
                np.testing.assert_allclose(src[power][n], expected_src[power][n], atol=2e-14)
        h = c.parent_return_bilinears(endpoint=endpoint, left=p@left, right=p@right)
        norm = c.parent_return_bilinears(endpoint=endpoint, left=p@left, right=p@right, norm=True)
        hc = q.return_coefficients(endpoint=endpoint)
        nc = q.return_coefficients(endpoint=endpoint, norm=True)
        eps = 2e-5
        shifted = []
        for sign in (-1, 1):
            member, cq = copy(c), copy(q)
            name = "energy_out" if endpoint == "outgoing" else "energy_in"
            setattr(cq, name, getattr(cq, name)+sign*eps)
            member.Q_current = cq
            shifted.append(member.parent_return_bilinears(endpoint=endpoint, left=p@left, right=p@right))
        for power in h:
            np.testing.assert_allclose(h[power], left.conj().T@hc[power]@right, atol=3e-16)
            np.testing.assert_allclose(norm[power], left.conj().T@nc[power]@right, atol=3e-16)
            np.testing.assert_allclose((shifted[1][power]-shifted[0][power])/(2*eps),
                                       -norm[power], atol=5e-12, rtol=2e-7)
    # The stacked incoming/outgoing probe slices have genuinely unequal sizes.
    actual = c.loop_coefficients(outgoing_vectors=left, incoming_vectors=right)
    columns = [c.loop_coefficients(outgoing_vectors=left[:, k:k+1], incoming_vectors=right)
               for k in range(2)]
    for power in actual:
        np.testing.assert_allclose(actual[power], np.concatenate([x[power] for x in columns], axis=1),
                                   atol=3e-16)


def test_reject_different_parent_H_and_different_Q_source_without_mutation():
    from copy import copy
    c = control()
    parent = c.parents["incoming"]
    changed = replace(parent, matrix=parent.matrix+1e-3*np.eye(len(parent.matrix)))
    with pytest.raises(ValueError, match="same inner Hamiltonian"):
        H2ProjectorCurrent(c.Q_current, parent_incoming=changed,
            parent_outgoing=c.parents["outgoing"], quadrature_order=16)
    q = copy(c.Q_current)
    q.sources = {e: {p: dict(s) for p, s in values.items()} for e, values in q.sources.items()}
    q.sources["incoming"][1]["qqqgg"] = 1.01*q.sources["incoming"][1]["qqqgg"]
    changed = H2ProjectorCurrent(q, parent_incoming=parent,
        parent_outgoing=c.parents["outgoing"], quadrature_order=16)
    with pytest.raises(ValueError, match="source"):
        changed.loop_coefficients(outgoing_vectors=scalar_vectors(37), incoming_vectors=scalar_vectors(53))
