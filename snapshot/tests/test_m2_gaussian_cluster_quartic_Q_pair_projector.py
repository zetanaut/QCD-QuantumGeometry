"""Quartic Q projector checks: strong source domain and independent raw Ward."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_Q_pair_projector import ClusterFiveQuarticProjector
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_pair_actions import (
    ClusterFiveQuarticCurrent,CHARGED_COMPONENTS,NEUTRAL_COMPONENTS)
from deuteron_wigner.quantum.m2_five_body_factorized import apply_terms
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from test_m2_gaussian_cluster_quartic_pair_actions import hamiltonian_terms
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS,raw,rho
from test_m2_gaussian_cluster_h2 import model


def action_spaces(cutoff,jz):
    return {n:IntermediateSpace(n,9,cutoff,"PROTON",jz) for n in ("qqqgg","qqqq-qbar")}


def action(cls,outgoing,incoming):
    return cls(outgoing,incoming,D,regulator=model().pilot.field_regulator,
        b_GeV=.4,total_pplus_GeV=PP,mean_total_transverse_GeV=(0.,0.),**ARGS)


@pytest.mark.parametrize("cutoff",[0,1])
def test_neutral_Hamiltonians_match_the_original_on_H2_source_waves(cutoff):
    m=model(q5=cutoff);x=m.snapshot(1.4).five_wave_coefficients[0]
    a=ClusterFiveQuarticProjector.from_models(m,m,D,**ARGS)
    for component,label in (("gg_number","gg_number_scattering"),("A4","A4")):
        got=hamiltonian_terms(a,x,[component])
        n,terms=m.qq.action.contacts.components[label]
        expected=apply_terms(terms,x[n],m.spaces5[n].dimension)
        np.testing.assert_allclose(got[n],expected,atol=2e-14)
        assert np.linalg.norm(expected)>1e-8
    only={n:x[n] if n=="qqqgg" else np.zeros_like(v) for n,v in x.items()}
    got=hamiltonian_terms(a,only,["gg_annihilation"])
    expected=m.qq.action.pair_current.apply(only)
    np.testing.assert_allclose(got["qqqgg"],expected["qqqgg"],atol=2e-14)
    assert np.linalg.norm(expected["qqqgg"])>1e-8


@pytest.mark.parametrize("component",CHARGED_COMPONENTS+NEUTRAL_COMPONENTS)
def test_full_physical_raw_Q5_Ward_every_quartic_component_and_adjoint(component):
    # Neutral constant pair H changes its relative radial quantum number
    # by two. The first nonzero spectator/cutoff boundary needs shell2.
    cutoff=2 if component in NEUTRAL_COMPONENTS else 1
    inc,out=action_spaces(cutoff,1),action_spaces(cutoff,-1)
    a=action(ClusterFiveQuarticProjector,out,inc)
    hi=action(ClusterFiveQuarticProjector,inc,inc)
    hf=action(ClusterFiveQuarticProjector,out,out)
    j=action(ClusterFiveQuarticCurrent,out,inc)
    probes=raw(inc)
    x={n:five_statistics_project(s,probes[n]) for n,s in inc.items()}
    y=raw(out,701)
    shifted={n:rho(out[n],s,x[n]) for n,s in inc.items()}
    expected=hamiltonian_terms(hf,shifted,[component])
    hx=hamiltonian_terms(hi,x,[component])
    for n in expected:expected[n]-=rho(out[n],inc[n],hx[n])
    full={n:np.zeros_like(v) for n,v in expected.items()};norm=0.
    for mu in range(2):
        v=a.apply(x,component=mu,components=[component])
        back=a.adjoint(y,component=mu,components=[component])
        np.testing.assert_allclose(sum(np.vdot(y[n],v[n]) for n in v),
                                   sum(np.vdot(back[n],x[n]) for n in x),atol=4e-15)
        physical=j.apply(x,component=mu+2,part="total",components=[component])
        for n in full:
            full[n]+=2*PP*D[mu]*(v[n]+physical[n])
            norm+=np.linalg.norm(v[n])
    for n in full:np.testing.assert_allclose(full[n],expected[n],atol=2e-14)
    assert norm>1e-8


@pytest.mark.parametrize("component",NEUTRAL_COMPONENTS)
def test_neutral_pair_shell_one_selection_zero_is_not_an_absent_Hamiltonian(component):
    inc,out=action_spaces(1,1),action_spaces(1,-1)
    a=action(ClusterFiveQuarticProjector,out,inc)
    x=raw(inc)
    h=hamiltonian_terms(action(ClusterFiveQuarticProjector,inc,inc),x,[component])
    assert sum(np.linalg.norm(v) for v in h.values())>1e-8
    for j in range(2):
        for v in a.apply(x,component=j,components=[component]).values():
            np.testing.assert_allclose(v,0.,atol=3e-15)


def test_only_explicit_transverse_components_and_complete_families_are_allowed():
    m=model();a=ClusterFiveQuarticProjector.from_models(m,m,D,**ARGS)
    with pytest.raises(ValueError,match="transverse"):a.apply(raw(m.spaces5),component=2)
    with pytest.raises(ValueError):a.apply(raw(m.spaces5),component=0,components=["unknown"])
    assert not a.minus_projector_supplied and not a.complete_H2_matched_current
