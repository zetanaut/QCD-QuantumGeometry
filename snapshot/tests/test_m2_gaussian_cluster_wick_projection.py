"""Actual common-kernel Wick currents: raw tensors, H2 words and paired norms."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import (
    ClusterWickCurrentWeights,cluster_wick_retained_minus_between,
    cluster_wick_shell_minus_between,cluster_wick_minus_h2_coefficients,
    cluster_wick_minus_h2_between)
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_wick_current_projection import retained_wick_minus_between
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace
from test_m2_wick_current_projection import _tensor_reference,SECTORS,FIVE
from test_m2_gaussian_cluster_wick import operators,regulator
from test_m2_wick_one_body import operators as constant_operators
from test_m2_gaussian_cluster_current_actions import pilot
from test_m2_gaussian_cluster_h2 import model


ARGS = dict(charge="electromagnetic",loop_flavor_charges=(2/3,-1/3),prescription=PRESCRIPTION)
DELTA = np.array([.12,.16])


@pytest.mark.parametrize("sector",SECTORS)
@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_all_raw_Q5_Q6_coordinates_against_independent_full_tensor(sector,target,jz):
    cls = IntermediateSpace if sector in FIVE else SixBodySpace
    i,f = cls(sector,9,1,target,1),cls(sector,9,1,target,jz)
    rng = np.random.default_rng(773)
    x = (rng.normal(size=(i.dimension,2))+1j*rng.normal(size=(i.dimension,2)))/np.sqrt(i.dimension)
    y = (rng.normal(size=(f.dimension,2))+1j*rng.normal(size=(f.dimension,2)))/np.sqrt(f.dimension)
    op = operators(regulator(),subtract=True)
    kwargs = dict(operators=op,b_GeV=.4,total_pplus_GeV=1.3,**ARGS)
    pieces = {}
    for component in ("canonical","regulator","ordered","subtraction","finite","total"):
        weights = ClusterWickCurrentWeights(operators=op,transfer_GeV=DELTA,
                                           prescription=PRESCRIPTION,component=component)
        actual = cluster_wick_shell_minus_between(f,i,x,DELTA,component=component,**kwargs)
        np.testing.assert_allclose(actual,_tensor_reference(f,i,x,DELTA,weights),atol=2e-16)
        pieces[component] = actual
    np.testing.assert_allclose(pieces["ordered"],pieces["canonical"]+pieces["regulator"],atol=2e-16)
    np.testing.assert_array_equal(pieces["subtraction"],-pieces["ordered"])
    np.testing.assert_array_equal(pieces["total"],pieces["finite"])
    assert np.linalg.norm(pieces["regulator"])>1e-6
    reverse = cluster_wick_shell_minus_between(i,f,y,-DELTA,component="ordered",**kwargs)
    np.testing.assert_allclose(y.conj().T@pieces["ordered"],reverse.conj().T@x,atol=3e-17)
    # The paired scalar H diagonal commutes with this mode-preserving action.
    rhs = cluster_wick_shell_minus_between(f,i,op.shell_mass_diagonal(i)[:,None]*x,
                                         DELTA,component="ordered",**kwargs)
    np.testing.assert_allclose(op.shell_mass_diagonal(f)[:,None]*pieces["ordered"],rhs,atol=3e-17)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_retained_components_flavor_linearity_and_same_finite_kinetic_partner(target):
    p = pilot(target,1)
    def value(component,charge="electromagnetic",loops=(2/3,-1/3)):
        return cluster_wick_retained_minus_between(p,p,DELTA,component=component,
            charge=charge,loop_flavor_charges=loops,prescription=PRESCRIPTION)
    ordered,sub,finite,total = (value(c) for c in ("ordered","subtraction","finite","total"))
    np.testing.assert_array_equal(sub,-ordered)
    np.testing.assert_array_equal(total,finite)
    np.testing.assert_allclose(ordered,value("canonical")+value("regulator"),atol=2e-15)
    np.testing.assert_allclose(ordered,(2*value("ordered","up_number",(1,0))-
        value("ordered","down_number",(0,1)))/3,atol=3e-15)
    old = constant_operators(subtract=True)
    expected = retained_wick_minus_between(p,p,DELTA,operators=old,total_pplus_GeV=1.3,
                        charge="electromagnetic",loop_flavor_charges=(2/3,-1/3))
    np.testing.assert_array_equal(finite,expected)
    assert np.linalg.norm(ordered)>1e-2 and np.linalg.norm(finite)>1e-3
    # A changed total P+ changes current normalization, never M² itself.
    boosted = replace(p,total_pplus_GeV=2.6)
    a = cluster_wick_retained_minus_between(boosted,boosted,DELTA,component="ordered",**ARGS)
    np.testing.assert_array_equal(a,ordered/4)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_actual_H2_direct_order2_order4_independent_raw_words_and_both_norms(target,jz):
    owner_i,owner_f = model(1,0,target),model(1,0,target,jz=jz)
    incoming,outgoing = owner_i.snapshot(1.47),owner_f.snapshot(1.39)
    saved = [(s.effective_hamiltonian.copy(),s.metric.copy()) for s in (incoming,outgoing)]
    op = owner_i.wick_operators
    pieces = {}
    for component in ("ordered","subtraction","finite","total"):
        coefficients = cluster_wick_minus_h2_coefficients(outgoing,incoming,DELTA,
                                                        component=component,**ARGS)
        assert set(coefficients)=={2,4}
        weights = ClusterWickCurrentWeights(operators=op,transfer_GeV=DELTA,
                                            prescription=PRESCRIPTION,component=component)
        independent = np.zeros_like(coefficients[4])
        for name,space in owner_i.spaces5.items():
            # Direct free resolvent on the raw source, independent of the
            # snapshot wave cache and the production packed current action.
            ai = owner_i.qq.propagation[name].resolve(1.47,owner_i.qq.sources_by_power[1][name])
            af = owner_f.qq.propagation[name].resolve(1.39,owner_f.qq.sources_by_power[1][name])
            independent += af.conj().T@_tensor_reference(owner_f.spaces5[name],space,ai,DELTA,weights)
        np.testing.assert_allclose(coefficients[4],independent,atol=4e-18)
        raw = .3**2*coefficients[2]+.3**4*independent
        actual = cluster_wick_minus_h2_between(outgoing,incoming,DELTA,
                                     component=component,normalized=False,**ARGS)
        normed = cluster_wick_minus_h2_between(outgoing,incoming,DELTA,
                                     component=component,normalized=True,**ARGS)
        np.testing.assert_allclose(actual,raw,atol=4e-18)
        np.testing.assert_allclose(normed,outgoing.whitener@raw@incoming.whitener,atol=4e-18)
        reverse = cluster_wick_minus_h2_between(incoming,outgoing,-DELTA,
                                     component=component,normalized=True,**ARGS)
        np.testing.assert_allclose(normed,reverse.conj().T,atol=2e-17)
        pieces[component] = normed
    np.testing.assert_array_equal(pieces["subtraction"],-pieces["ordered"])
    np.testing.assert_array_equal(pieces["total"],pieces["finite"])
    assert np.linalg.norm(pieces["ordered"])>1e-8
    for snapshot,(h,n) in zip((incoming,outgoing),saved):
        np.testing.assert_array_equal(snapshot.effective_hamiltonian,h)
        np.testing.assert_array_equal(snapshot.metric,n)
    with pytest.raises(ValueError,match="four-current"):
        incoming.current(DELTA,charge="electromagnetic")


def test_cluster_definition_cannot_be_silently_changed():
    p = pilot("PROTON",1)
    for f in (replace(p,total_pplus_GeV=2.),
              replace(p,wick_operators=replace(p.wick_operators,normal_order_subtraction=False))):
        with pytest.raises(ValueError):
            cluster_wick_retained_minus_between(f,p,DELTA,component="total",**ARGS)
    s = model().snapshot(1.4)
    with pytest.raises(ValueError,match="Boolean"):
        cluster_wick_minus_h2_between(s,s,DELTA,component="total",normalized="yes",**ARGS)
