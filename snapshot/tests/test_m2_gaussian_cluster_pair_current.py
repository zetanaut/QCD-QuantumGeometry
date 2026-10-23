"""Unfactorized four-physical-leg current against Hamiltonian Ward endpoints."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_pair_current import (
    pair_current_geometry,cluster_pair_current_orbitals,ordered_qg_coefficients)
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_gaussian_field_regulator import gaussian_shell_integral
from deuteron_wigner.quantum.m2_two_gluon_minus_current import displaced_pair_contact
from test_m2_gaussian_cluster_current import regulator

Q3 = ("q",)*3
CASES = [
    ((3,1,5),(1,3,5),(0,1),Q3,Q3,(.7,-.2,.7,-.2)),
    ((3,1,3,2),(1,1,3,4),(0,3),Q3+("g",),Q3+("g",),(.7,0.,.7,0.)),
    ((3,1,1,2,2),(1,1,1,4,2),(0,3),Q3+("g","g"),Q3+("g","g"),(.7,0.,.7,0.)),
    ((1,1,1,3,3),(3,1,1,3,1),(0,4),("q",)*4+("qbar",),("q",)*4+("qbar",),(.7,-.7,.7,-.7)),
    ((1,1,1,2,4),(1,1,1,3,3),(3,4),Q3+("g","g"),("q",)*4+("qbar",),(.7,-.7,0.,0.)),
    ((1,1,1,2,2,2),(1,1,1,2,2,2),(3,5),Q3+("g",)*3,Q3+("g",)*3,(0.,0.,0.,0.)),
]


def pair_current(case,*,delta=(.17,-.11),mean=(.04,.03),ci=1,co=1,reg=None,
                 h=.32+.14j,j=.21-.1j,order=12):
    inc,out,pair,si,so,charges = case
    return cluster_pair_current_orbitals(inc,out,pair,ci,co,.4,species=si,out_species=so,
        charges=charges,regulator=regulator() if reg is None else reg,total_pplus_GeV=1.3,
        transfer_GeV=delta,mean_total_transverse_GeV=mean,hamiltonian_coefficient=h,
        canonical_current_coefficient=(0.,j,0.,0.),prescription=PRESCRIPTION,quadrature_order=order)


def quartic_endpoint_reference(case,delta,mean,reg,ci,co,h):
    inc,out,pair,si,so,charges = case
    g = pair_current_geometry(inc,out,pair,species=si,out_species=so,
        transfer_GeV=delta,mean_total_transverse_GeV=mean)
    B = reg.leg_transform(g["modes"])
    result = 0.
    for a,sigma in enumerate((-1,-1,1,1)):
        shifts = g["physical_field_shifts"].copy();shifts[a] += sigma*np.asarray(delta)
        result += sigma*charges[a]*gaussian_shell_integral(out_cutoff=co,in_cutoff=ci,
            out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],b_GeV=.4,
            field_maps=B@g["physical_field_maps"],field_shifts=B@shifts,
            field_precision=1/reg.width_GeV**2,constant=h)
    return g["jacobian"]*result/(2*1.3)


@pytest.mark.parametrize("case",CASES)
@pytest.mark.parametrize("ci,co",[(0,1),(1,0),(1,2)])
def test_four_physical_leg_Ward_and_Hermitian_reversal(case,ci,co):
    delta,mean = np.array([.17,-.11]),np.array([.04,.03])
    a = pair_current(case,ci=ci,co=co)
    expected = quartic_endpoint_reference(case,delta,mean,regulator(),ci,co,.32+.14j)
    np.testing.assert_allclose(delta@a.total[2:].reshape(2,-1),expected.reshape(-1),atol=3e-16,rtol=4e-12)
    inc,out,pair,si,so,charges = case
    reverse = (out,inc,pair,so,si,charges[2:]+charges[:2])
    back = pair_current(reverse,delta=-delta,mean=mean,ci=co,co=ci,h=.32-.14j,j=.21+.1j)
    for forward,backward in ((a.canonical,back.canonical),(a.regulator,back.regulator)):
        np.testing.assert_allclose(forward,backward.conj().transpose(0,2,1),atol=2e-15,rtol=4e-12)
    if si!=so and out[pair[0]]==out[pair[1]] and co==0:
        # Equal-mode q/qbar exchange reverses the regulator charge sum.
        # The outgoing ground orbital is even and the incoming gg side
        # is independent of that relative coordinate: the integral is zero.
        np.testing.assert_allclose(a.regulator,0.,atol=2e-18)
    elif any(charges):
        assert np.linalg.norm(a.regulator[2:])>1e-5
    else:
        np.testing.assert_array_equal(a.regulator,0.)


@pytest.mark.parametrize("case",CASES)
def test_four_leg_current_boost_law_and_path_refinement(case):
    beta,mean = np.array([.23,-.16]),np.array([.04,.03])
    a = pair_current(case,mean=mean,order=8)
    b = pair_current(case,mean=mean+1.3*beta,order=16)
    np.testing.assert_allclose(a.total[2:],b.total[2:],atol=2e-15)
    np.testing.assert_allclose(b.total[1],a.total[1]+np.einsum("j,jab->ab",beta,a.total[2:]),atol=3e-15)


@pytest.mark.parametrize("crossed",[False,True])
@pytest.mark.parametrize("hq,hg",[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_actual_ordered_qg_canonical_minus_and_signed_Hamiltonian_partner(crossed,hq,hg):
    case = CASES[1]
    modes = (1,4,3,2)
    h,j = ordered_qg_coefficients(modes,total_k2=9,total_pplus_GeV=1.3,
        quark_helicity=hq,gluon_helicity=hg,crossed=crossed,flavor_charge=.7)
    r = -1 if crossed else 5
    # Independent constrained-denominator derivative, not an absolute mode.
    np.testing.assert_allclose(j[1],.7*9/(2*1.3**2)*h/r,atol=4e-16)
    a = pair_current(case,ci=1,co=2,reg=regulator(1e7),h=h,j=j[1],order=2)
    old = displaced_pair_contact(case[0],case[1],case[2],.4,1,2,(.17,-.11))
    np.testing.assert_allclose(a.total,j[:,None,None]*old,atol=4e-16)
    if h!=0:
        actual = pair_current(case,h=h,j=j[1])
        assert np.linalg.norm(actual.regulator)>1e-5


def test_pair_conversion_kernel_contains_four_physical_fields_not_an_auxiliary_gluon():
    case = CASES[4]
    g = pair_current_geometry(*case[:3],species=case[3],out_species=case[4],
        transfer_GeV=(.17,-.11),mean_total_transverse_GeV=(.04,.03))
    assert g["modes"]==(3,3,2,4) and g["species"]==("q","qbar","g","g")
    # The exchanged spin-zero current has mode6, beyond the allowed gluon4.
    # Only the physical gluons2,4 enter its common field envelope.
    a = pair_current(case,reg=replace(regulator(),boson_modes=2))
    assert np.linalg.norm(a.total)>1e-5
    np.testing.assert_array_equal(pair_current(case,reg=replace(regulator(),boson_modes=1)).total,0.)
