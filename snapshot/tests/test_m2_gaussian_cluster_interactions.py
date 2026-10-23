"""Common cluster interaction orbitals: independent integration and bare limits."""
from itertools import product
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_interactions import (
    cluster_pair_contact, cluster_pair_integral_reference, pair_coordinate_map,
    cluster_retained_emission, cluster_six_cubic, cluster_pair_current_orbital)
from deuteron_wigner.quantum.m2_coupled_pair_action import pair_contact_block
from deuteron_wigner.quantum.m2_excluded_current_scattering import five_body_contact
from deuteron_wigner.quantum.m2_excluded_pair_conversion import conversion_contact
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import emission_routes, oscillator_shell
from deuteron_wigner.quantum.m2_canonical_emission import _route_orbital
from deuteron_wigner.quantum.m2_six_body_vertices import split_routes as six_routes, cubic_orbital
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import split_routes, SPECIES
from deuteron_wigner.quantum.m2_excluded_pair_current import pair_current_orbital
from test_m2_gaussian_cluster_regulator import regulator

Q3 = ("q",)*3
CASES = [
    ((3,1,5),(1,3,5),(0,1),Q3,Q3),
    ((3,1,3,2),(1,1,3,4),(0,3),Q3+("g",),Q3+("g",)),
    ((1,1,1,2,4),(1,1,1,4,2),(3,4),SPECIES["qqqgg"],SPECIES["qqqgg"]),
    ((3,1,1,2,2),(1,1,1,4,2),(0,3),SPECIES["qqqgg"],SPECIES["qqqgg"]),
    ((1,1,1,3,3),(3,1,1,3,1),(0,4),SPECIES["qqqq-qbar"],SPECIES["qqqq-qbar"]),
    ((1,1,1,2,4),(1,1,1,3,3),(3,4),SPECIES["qqqgg"],SPECIES["qqqq-qbar"]),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("width", [.2, .8])
@pytest.mark.parametrize("ci,co", [(0,0),(1,2),(2,1),(2,2)])
def test_all_contact_types_and_radial_signs_against_unfactorized_continuum_Gaussian(case,width,ci,co):
    incoming,outgoing,pair,si,so=case
    args=dict(regulator=regulator(width),species=si,out_species=so)
    got=cluster_pair_contact(incoming,outgoing,pair,.4,ci,co,**args)
    reference=cluster_pair_integral_reference(incoming,outgoing,pair,.4,ci,co,
        **args,total_transverse_GeV=(.23,-.17))
    np.testing.assert_allclose(got,reference,atol=4e-15,rtol=3e-12)
    reverse=cluster_pair_contact(outgoing,incoming,pair,.4,co,ci,
        regulator=regulator(width),species=so,out_species=si)
    np.testing.assert_allclose(got,reverse.conj().T,atol=4e-15)
    if ci != co:
        high=cluster_pair_contact(incoming,outgoing,pair,.4,max(ci,co),max(ci,co),**args)
        np.testing.assert_allclose(got,high[:got.shape[0],:got.shape[1]],atol=4e-15)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cutoff",[0,1,2])
def test_contact_bare_limit_matches_existing_complete_radial_kernel(case,cutoff):
    incoming,outgoing,pair,si,so=case
    got=cluster_pair_contact(incoming,outgoing,pair,.4,cutoff,
        regulator=regulator(1e7),species=si,out_species=so)
    if len(si)<5:
        expected=pair_contact_block(incoming,outgoing,pair,.4,cutoff)
    elif si != so:
        expected=conversion_contact(incoming,outgoing,.4,cutoff)
    else:
        sector="qqqgg" if si==SPECIES["qqqgg"] else "qqqq-qbar"
        expected=five_body_contact(sector,incoming,outgoing,pair,.4,cutoff)
    np.testing.assert_allclose(got,expected,atol=4e-15,rtol=3e-12)


@pytest.mark.parametrize("cutoff",[0,1,2])
def test_retained_cubic_bare_limit_all_emitters_and_helicities(cutoff):
    for route in (emission_routes(9)[0],emission_routes(9)[-1]):
        for hq,hi,hg in product((-1,1),repeat=3):
            got=cluster_retained_emission(route,cutoff,.4,.35,hq,hi,hg,
                regulator=regulator(1e7),total_pplus_GeV=1.3)
            np.testing.assert_allclose(got,_route_orbital(route,cutoff,.4,.35,hq,hi,hg),atol=4e-15,rtol=3e-12)


SIX=[("qqqgg","q_to_qg"),("qqqgg","g_to_gg"),("qqqgg","g_to_pair"),
     ("qqqq-qbar","q_to_qg"),("qqqq-qbar","anti_to_antig")]


@pytest.mark.parametrize("sector,channel",SIX)
@pytest.mark.parametrize("cutoff",[0,1,2])
def test_U65_every_cubic_channel_and_signed_antiquark_bare_limit(sector,channel,cutoff):
    routes=six_routes(9,sector,channel)
    for route in (routes[0],routes[-1]):
        for hs in product((-1,1),repeat=3):
            got=cluster_six_cubic(route,cutoff,cutoff,.4,.35,hs,
                regulator=regulator(1e7),total_pplus_GeV=1.3)
            np.testing.assert_allclose(got,cubic_orbital(route,cutoff,cutoff,.4,.35,hs),atol=4e-15,rtol=3e-12)


@pytest.mark.parametrize("channel",["g_to_gg","g_to_pair"])
@pytest.mark.parametrize("cutoff",[0,1,2])
def test_pair_current_factorization_bare_limit_and_auxiliary_is_not_regulated_field(channel,cutoff):
    route=split_routes(9,channel)[0]
    got=cluster_pair_current_orbital(route,cutoff,.4,regulator=regulator(1e7))
    np.testing.assert_allclose(got,pair_current_orbital(route,cutoff,.4),atol=4e-15)
    if channel=="g_to_gg":
        # Two permitted k2=2 gluons can have an auxiliary current mode4
        # larger than the boson field envelope. That auxiliary is not a field.
        route=next(r for r in split_routes(9,channel) if r.outgoing[3:]==(2,2))
        small=replace(regulator(),boson_modes=1)
        assert route.incoming[3]>2*small.boson_modes
        assert np.linalg.norm(cluster_pair_current_orbital(route,cutoff,.4,regulator=small))>0


def test_contact_field_envelope_applies_only_to_actual_participating_legs():
    reg=replace(regulator(),boson_modes=1,fermion_modes=2)
    args=dict(regulator=reg,species=Q3)
    # Spectator mode5 lies outside the field envelope but is not in this vertex.
    assert np.linalg.norm(cluster_pair_contact((3,1,5),(1,3,5),(0,1),.4,1,**args))>0
    # Moving that mode into an active leg forbids the interaction exactly.
    np.testing.assert_array_equal(cluster_pair_contact((3,1,5),(5,1,3),(0,2),.4,1,**args),0.)
