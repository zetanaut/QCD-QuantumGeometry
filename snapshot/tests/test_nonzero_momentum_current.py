import json
import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.canonical_lf_fields import GAMMA, slash
from deuteron_wigner.quantum.nonzero_momentum_current import (
    CollectiveChannel, QuarkCollectiveReference, fermion_propagators,
    benchmark_reference, benchmark_momenta, radiative_current, contract,
    external_states, lf_coordinates, dot,
    canonical_radiative_current,
)


@pytest.mark.parametrize('p', [[.4,.2,.1,.3], [-.4,.2,-.1,-.3], [.1,.3,.1,-.5]])
def test_independent_signed_LF_spin_sums(p):
    parts = fermion_propagators(np.array(p), .23)
    np.testing.assert_allclose(parts['light_front'], parts['covariant'], atol=5e-14, rtol=2e-14)
    np.testing.assert_allclose((slash(p)-.23*np.eye(4))@parts['light_front'], np.eye(4), atol=2e-14)
    assert np.linalg.norm(parts['propagating']-parts['covariant']) > 1


@pytest.mark.parametrize('variation', [0., .5, 2., -1.])
def test_all_attachments_Ward_and_independent_LF_components(variation):
    model = benchmark_reference(); out, inc = benchmark_momenta(model.mass, variation)
    cov = radiative_current(model, out, inc)
    lf = radiative_current(model, out, inc, propagation='light_front')
    prop = radiative_current(model, out, inc, propagation='propagating')
    inst = radiative_current(model, out, inc, propagation='instantaneous')
    np.testing.assert_allclose(lf['total'], cov['total'], atol=3e-10, rtol=2e-12)
    np.testing.assert_allclose(prop['total']+inst['total'], lf['total'], atol=3e-10, rtol=2e-12)
    np.testing.assert_allclose(contract(cov['total'], cov['photon']), 0, atol=2e-10)
    assert np.linalg.norm(contract(cov['endpoint'], cov['photon'])) > 1
    assert np.linalg.norm(prop['total']-cov['total'])/np.linalg.norm(cov['total']) > .01
    explicit = radiative_current(model, out, inc, explicit_modes=True)
    np.testing.assert_allclose(explicit['total'], cov['total'], atol=5e-11, rtol=1e-13)


def test_tree_channel_Ward_transverse_sensitivity_and_mode_subtraction():
    model = benchmark_reference(); out, inc = benchmark_momenta(model.mass)
    r, s = out-inc
    pion = model.pion
    for p in [r,-s,-r,s]:
        np.testing.assert_allclose(pion.response(p,representation='light_front'),pion.response(p),atol=2e-14)
    np.testing.assert_allclose(dot(r+s, pion.current(r, -s)), pion.weight(r)-pion.weight(s), atol=2e-13)
    minimal = CollectiveChannel(pion.hessian, pion.couplings, np.zeros_like(pion.c))
    delta = pion.current(r, -s)-minimal.current(r, -s)
    assert np.linalg.norm(delta) > .1
    assert abs(dot(r+s, delta)) < 2e-13
    # Eigenbasis decomposition: the true remainder includes cross-mode currents.
    masses, e = np.linalg.eigh(pion.hessian)
    heavy = CollectiveChannel(masses[1:, None], e[:, 1:].T@pion.couplings, e[:, 1:].T@pion.c@e[:, 1:])
    light = pion.retain_lightest()
    remainder = pion.current(r, -s)-light.current(r, -s)
    assert np.linalg.norm(remainder-heavy.current(r, -s)) > .01
    np.testing.assert_allclose(pion.weight(r), light.weight(r)+heavy.weight(r), atol=2e-13)
    full = radiative_current(model, out, inc)
    retained = radiative_current(model.retain_lightest(), out, inc)
    residual = full['total']-retained['total']
    np.testing.assert_allclose(contract(residual, full['photon']), 0, atol=2e-11)
    np.testing.assert_allclose(retained['total']+residual, full['total'], atol=1e-12)
    assert np.linalg.norm(full['interaction']-retained['interaction']) > 1


def boost_data(direction):
    k = np.zeros((4,4));k[0,1:]=k[1:,0]=direction
    spin = expm(sum(direction[i]*GAMMA[0]@GAMMA[i+1]/2 for i in range(3)))
    return expm(k), np.kron(spin, np.eye(6))


@pytest.mark.parametrize('direction', [[.4,0,0], [0,0,.7], [-.3,.2,.15]])
def test_all_four_components_with_actual_external_spin_transport(direction):
    model = benchmark_reference(); out, inc = benchmark_momenta(model.mass)
    a = radiative_current(model, out, inc)
    lorentz, spin = boost_data(direction)
    boosted_out, boosted_inc = out@lorentz.T, inc@lorentz.T
    b = radiative_current(model, boosted_out, boosted_inc, propagation='light_front')
    def rotation(p):
        changed = external_states(lorentz@p, model.mass)
        original = external_states(p, model.mass)
        w = changed.conj().T@np.kron(GAMMA[0], np.eye(6))@spin@original/(2*model.mass)
        np.testing.assert_allclose(w@w.conj().T, np.eye(12), atol=2e-14)
        return w
    left = np.kron(rotation(out[0]), rotation(out[1]))
    right = np.kron(rotation(inc[0]), rotation(inc[1]))
    expected = np.einsum('mn,nij->mij', lorentz, a['total'])
    expected = np.array([left@component@right.conj().T for component in expected])
    np.testing.assert_allclose(b['total'], expected, atol=4e-10, rtol=2e-12)
    # A transverse current modification cannot be detected by Ward alone.
    zero = CollectiveChannel(model.pion.hessian, model.pion.couplings, np.zeros_like(model.pion.c))
    minimal = radiative_current(QuarkCollectiveReference(model.scalar, zero, model.mass), out, inc)
    difference = a['total']-minimal['total']
    np.testing.assert_allclose(contract(difference, a['photon']), 0, atol=2e-11)
    assert np.linalg.norm(difference) > 1


def test_exchange_labels_and_fermion_antisymmetry():
    model = benchmark_reference(); out, inc = benchmark_momenta(model.mass)
    full = radiative_current(model, out, inc)['total']
    swapped = radiative_current(model, out[::-1], inc[::-1])['total']
    swapped = swapped.reshape(4,12,12,12,12).transpose(0,2,1,4,3).reshape(4,144,144)
    np.testing.assert_allclose(swapped, full, atol=1e-11)
    # Direct-minus-outgoing-exchanged graph, not a deletion of flavor channels.
    crossed = radiative_current(model, out[::-1], inc)['total']
    crossed = crossed.reshape(4,12,12,12,12).swapaxes(1,2).reshape(4,144,144)
    physical = full-crossed
    q = np.sum(out-inc, axis=0)
    np.testing.assert_allclose(contract(physical, q), 0, atol=1e-10)
    assert np.linalg.norm(physical) > 1


def test_rejects_undefined_domains_instead_of_inventing_zero_modes():
    with pytest.raises(ValueError): fermion_propagators(np.array([.3,0,0,-.3]), .2)
    with pytest.raises(ValueError): fermion_propagators(np.array([.2,0,0,0]), .2)
    with pytest.raises(ValueError): CollectiveChannel(np.diag([1.,-1.]), [1.,1.], np.eye(2))
    model = benchmark_reference()
    with pytest.raises(ValueError): model.pion.response(np.array([1.,0,0,0]))


@pytest.mark.parametrize('variation', [0., .7, -1.])
def test_canonical_color_vector_both_LF_decompositions_and_negatives(variation):
    model = benchmark_reference(); out, inc = benchmark_momenta(model.mass, variation)
    cov = canonical_radiative_current(model.mass,out,inc,1.2)
    lf = canonical_radiative_current(model.mass,out,inc,1.2,
                                     quark_propagation='light_front',gluon_representation='light_front')
    no_instant_gluon = canonical_radiative_current(model.mass,out,inc,1.2,gluon_representation='propagating')
    no_instant_quark = canonical_radiative_current(model.mass,out,inc,1.2,quark_propagation='propagating')
    np.testing.assert_allclose(lf['total'],cov['total'],atol=3e-10,rtol=2e-12)
    np.testing.assert_allclose(contract(cov['total'],cov['photon']),0,atol=2e-11)
    assert lf['max_radiating_gluon_divergence'] < 1e-12
    assert np.linalg.norm(no_instant_gluon['total']-cov['total'])/np.linalg.norm(cov['total']) > .01
    assert np.linalg.norm(no_instant_quark['total']-cov['total'])/np.linalg.norm(cov['total']) > .01
    assert no_instant_quark['max_radiating_gluon_divergence'] > .01
    soft = radiative_current(model,out,inc)['total']
    # Full color trace on both legs kills t_a tensor t_a but not singlet S/P.
    def trace_colors(a):
        a=a.reshape(4,4,3,4,3,4,3,4,3)
        return np.einsum('mabcdebfd->macef',a, optimize=True)
    # Explicit index spelling avoids reordering spin/flavor with color.
    np.testing.assert_allclose(trace_colors(cov['total']),0,atol=2e-12)
    assert np.linalg.norm(trace_colors(soft)) > 1


def test_combined_current_boost_reversal_and_signed_pair_route():
    model=benchmark_reference(); out,inc=benchmark_momenta(model.mass)
    def current(a,b):
        return (radiative_current(model,a,b,propagation='light_front')['total']
                +canonical_radiative_current(model.mass,a,b,1.2,
                    quark_propagation='light_front',gluon_representation='light_front')['total'])
    a=current(out,inc)
    reversed_current=current(inc,out)
    np.testing.assert_allclose(reversed_current,a.conj().transpose(0,2,1),atol=5e-10,rtol=2e-12)
    lorentz,spin=boost_data([-.2,.4,.15])
    b=current(out@lorentz.T,inc@lorentz.T)
    def w(p):
        return external_states(lorentz@p,model.mass).conj().T@np.kron(GAMMA[0],np.eye(6))@spin@external_states(p,model.mass)/(2*model.mass)
    left,right=np.kron(w(out[0]),w(out[1])),np.kron(w(inc[0]),w(inc[1]))
    expected=np.array([left@j@right.conj().T for j in np.einsum('mn,nij->mij',lorentz,a)])
    np.testing.assert_allclose(b,expected,atol=5e-10,rtol=2e-12)
    # A genuine negative internal-plus route from future external particles.
    spatial=np.array([[.9,-.2,-.8],[-.05,.09,-.2],[.04,.02,.1],[.2,-.1,1.2]])
    momenta=np.c_[np.sqrt(model.mass**2+np.sum(spatial**2,axis=1)),spatial]
    out,inc=momenta[:2],momenta[2:];q=np.sum(out-inc,axis=0)
    assert min(lf_coordinates(p)[0] for p in [*(inc+q),*(out-q)]) < 0
    cov=radiative_current(model,out,inc)['total']
    lf=radiative_current(model,out,inc,propagation='light_front')['total']
    np.testing.assert_allclose(cov,lf,atol=5e-11,rtol=2e-12)
    np.testing.assert_allclose(contract(lf,q),0,atol=5e-11)


def test_replay_is_serializable_and_narrow():
    from scripts.evaluate_nonzero_momentum_current import evaluate
    data = evaluate(); json.dumps(data, allow_nan=False)
    assert not data['claims']['physical_nucleon']
    assert not data['claims']['quantum_rematched_coefficients']
    assert data['maxima']['LF_component_relative'] < 1e-11
