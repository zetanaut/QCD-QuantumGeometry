"""Same-regulator oscillator transport; not complete physical field currents."""
from functools import lru_cache
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import (
    build_cluster_pilot, require_same_cluster_definition)
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent
from deuteron_wigner.quantum.m2_h2_projector_current import H2ProjectorCurrent, canonical_sources_on_vectors
from deuteron_wigner.quantum.m2_fourth_order_h2 import _norm_inverse_sqrt
from test_m2_gaussian_cluster_h2 import model
from test_m2_interacting_q_loop_regulator import (
    scalar_vectors,
    test_finite_parent_two_energy_Ward_for_each_interacting_word as word_ward,
    test_explicit_projected_vertex_gauging_equals_unprojected_word as projected_word,
    test_cached_coefficients_equal_direct_unfactorized_Leibniz as direct_leibniz)
from test_m2_h2_projector_current import (
    test_finite_parent_Ward_identity_for_each_complete_return_order as parent_ward,
    test_retained_H_boundary_agrees_with_existing_exact_nested_implementation as retained_boundary,
    test_reversed_P_source_current_energy_spin_and_transfer as reverse_parent)


@lru_cache(maxsize=1)
def qcontrol(target="PROTON", jz=-1, inner=0):
    f, i = model(inner+1,inner+1,target,jz), model(inner+1,inner+1,target,1)
    return InteractingQLoopRegulatorCurrent(f,i,
        inner_cutoffs={n:inner for n in (*i.spaces5,*i.spaces6)},
        energy_out=1.39,energy_in=1.47,transfer=(.12,.16),charge="electromagnetic",
        total_pplus_GeV=1.3,quadrature_order=16)


@pytest.fixture(scope="module",params=[("PROTON",-1),("NEUTRON",1)])
def q(request):
    return qcontrol(*request.param)


def test_inner_H_and_both_norms_match_independently_rebuilt_cluster_model(q):
    assert not q.paired_Wick_in_regulator_wrapper
    assert q.wick_operators==q.incoming.wick_operators
    for endpoint in ("outgoing","incoming"):
        outer=q._model(endpoint)
        inner=model(0,0,outer.pilot.target,outer.pilot.jz_twice)
        e=q.energy_out if endpoint=="outgoing" else q.energy_in
        snap=inner.snapshot(e)
        h=inner.pilot.matrix+sum(.3**p*a for p,a in q.return_coefficients(endpoint=endpoint).items())
        np.testing.assert_allclose(h,snap.effective_hamiltonian,atol=3e-14)
        np.testing.assert_allclose(q.metric(endpoint=endpoint),snap.metric,atol=5e-15)
    assert not q.common_field_regulator_complete and not q.physical


@pytest.mark.parametrize("word",[("R",),("R","U","R"),("R","W","R"),("R","U","R","U","R")])
def test_two_energy_Ward_of_actual_regulated_Wick_and_interacting_words(q,word):
    word_ward(q,word)
    if len(word)>1:
        projected_word(q,word)


def test_current_coefficients_equal_independent_direct_product_rule(q):
    direct_leibniz(q)


def parent(p):
    return build_cluster_pilot(total_k2=p.total_k2,cutoff=1,target=p.target,jz_twice=p.jz_twice,
        b_GeV=p.width_GeV,quark_mass_GeV=p.quark_mass_GeV,gluon_mass_GeV=p.gluon_mass_GeV,
        coupling=p.coupling,regulator=p.field_regulator,total_pplus_GeV=p.total_pplus_GeV,
        wick_operators=p.wick_operators)


@pytest.fixture(scope="module",params=["PROTON","NEUTRON"])
def c(request):
    q=qcontrol(request.param,-1,1)
    return H2ProjectorCurrent(q,parent_outgoing=parent(q.outgoing.pilot),
                             parent_incoming=parent(q.incoming.pilot),quadrature_order=16)


@pytest.fixture(scope="module")
def coefficients(c):
    vf,vi=scalar_vectors(31),scalar_vectors(772)
    return vf,vi,c.loop_coefficients(outgoing_vectors=vf,incoming_vectors=vi)


def test_regulated_source_parent_current_has_nonzero_returns_and_finite_parent_Ward(c,coefficients):
    assert np.linalg.norm(coefficients[2][2])>1e-9
    assert np.linalg.norm(coefficients[2][4])>1e-11
    parent_ward(c,coefficients)
    retained_boundary(c,coefficients)
    assert not c.complete_physical_four_current and not c.common_field_regulator_complete


def test_reverse_energy_spin_and_transfer_with_the_same_field_definition(c,coefficients):
    reverse_parent(c,coefficients)


def test_all_13_regulated_source_actions_bind_on_unequal_requested_probe_columns(c):
    left=np.column_stack((scalar_vectors(42),scalar_vectors(73)))
    for endpoint in ("outgoing","incoming"):
        actual=c.source_columns(endpoint,c.embeddings[endpoint]@left)
        expected=c.Q_current._source_columns(endpoint,left)
        for power in actual:
            for n in actual[power]:
                np.testing.assert_allclose(actual[power][n],expected[power][n],atol=3e-14)
        p=c.parents[endpoint]
        # A same-sized bare pilot demonstrably changes the source. The
        # definition check must not rely solely on a Hamiltonian subblock.
        from deuteron_wigner.quantum.m2_coupled_pilot import CoupledPilot
        from dataclasses import fields
        bare=CoupledPilot(**{f.name:getattr(p,f.name) for f in fields(CoupledPilot)})
        wrong=canonical_sources_on_vectors(bare,c.embeddings[endpoint]@left,
                                          spaces=getattr(c.Q_current.projector,endpoint))
        assert sum(np.linalg.norm(actual[o][n]-wrong[o][n]) for o in actual for n in actual[o])>1e-5


def test_retained_return_and_Q_terms_share_both_norms_once(c):
    q=c.Q_current
    vf,vi=scalar_vectors(512),scalar_vectors(781)
    wf,wi=_norm_inverse_sqrt(q.metric(endpoint="outgoing")),_norm_inverse_sqrt(q.metric(endpoint="incoming"))
    args=dict(outgoing_vectors=wf@vf,incoming_vectors=wi@vi)
    p,r=c.loop_coefficients(**args),q.coefficients(**args)
    expected=c.retained_current(**args)+sum(.3**n*(p[n]+r[n]) for n in p)
    actual=c.current(normalized=True,outgoing_vectors=vf,incoming_vectors=vi)
    np.testing.assert_allclose(actual,expected,atol=5e-16)


def test_reject_inconsistent_regulator_Wick_and_Pplus_even_at_equal_matrix_dimensions():
    from copy import copy
    q=qcontrol()
    p=q.incoming.pilot
    wrong=[replace(p,field_regulator=replace(p.field_regulator,width_GeV=1.1)),
           replace(p,total_pplus_GeV=1.4),
           replace(p,wick_operators=replace(p.wick_operators,finite_quark_mass2_per_gs2_GeV2=.012))]
    for bad in wrong:
        with pytest.raises(ValueError,match="identical cluster"):
            require_same_cluster_definition(p,bad)
        changed=copy(q.incoming)
        changed.pilot=bad
        with pytest.raises(ValueError,match="identical cluster"):
            InteractingQLoopRegulatorCurrent(q.outgoing,changed,
                inner_cutoffs={n:0 for n in q.projector.incoming},energy_out=1.39,energy_in=1.47,
                transfer=(.12,.16),charge="electromagnetic",total_pplus_GeV=1.3,quadrature_order=16)
        with pytest.raises(ValueError,match="identical cluster"):
            H2ProjectorCurrent(q,parent_outgoing=p,parent_incoming=bad,quadrature_order=16)
    with pytest.raises(ValueError,match="total P"):
        require_same_cluster_definition(p,p,total_pplus_GeV=1.4)
