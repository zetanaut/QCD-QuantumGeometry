"""Actual three/four-leg routes and noncommuting finite-projector partners."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
from deuteron_wigner.quantum.m2_gaussian_cluster_current import (
    PRESCRIPTION,cubic_current_geometry,cubic_physical_kernel,cluster_cubic_current_orbitals)
from deuteron_wigner.quantum.m2_gaussian_cluster_pair_current import cluster_pair_current_orbitals
from deuteron_wigner.quantum.m2_gaussian_cluster_projector_kernel import exact_transverse_projector_boundary
from deuteron_wigner.quantum.m2_gaussian_cluster_quartic_source_current import (
    quartic_source_geometry,quartic_source_charges,cluster_quartic_source_current)
from test_m2_gaussian_cluster_current import ROUTES as CUBIC_ROUTES,KINDS,regulator
from test_m2_gaussian_cluster_projector_kernel import CASES,data,density,D,B,PP
from test_m2_gaussian_cluster_quartic_source_current import ROUTES as QUARTIC_ROUTES,hs_nonzero,coefficients

SHAPE=GaussianReferenceShape(base_width_GeV=.8,reference_width_GeV=1.2,degree=1)


@pytest.mark.parametrize("route",CUBIC_ROUTES)
def test_actual_cubic_shape_current_matches_independent_shifted_Hamiltonian_ends(route):
    hs=(1,-1,-1) if getattr(route,"channel",None)=="g_to_pair" else (1,-1,1)
    mean=(.04,.03);g=cubic_current_geometry(route,transfer_GeV=D,mean_total_transverse_GeV=mean)
    modes,channel=g["modes"],g["channel"];reg=regulator()
    current=cluster_cubic_current_orbitals(route,0,1,B,.35,hs,regulator=reg,
        total_pplus_GeV=PP,transfer_GeV=D,mean_total_transverse_GeV=mean,
        flavor_charge=.7,prescription=PRESCRIPTION,quadrature_order=12,reference_shape=SHAPE)
    transform=reg.leg_transform(modes);expected=0.
    args=dict(channel=channel,helicities=hs,total_k2=g["total_k2"],total_pplus_GeV=PP,mass_GeV=.35)
    for a,sign in enumerate((-1,-1,1)):
        shifted=g["physical_field_shifts"].copy();shifted[a]+=sign*D
        c=cubic_physical_kernel(modes,shifted,**args);lin=np.zeros((2,g["out_map"].shape[1]),complex)
        for j in range(2):
            for k in range(lin.shape[1]):
                unit=shifted.copy();unit[:,j]+=g["physical_field_maps"][:,k]
                lin[j,k]=cubic_physical_kernel(modes,unit,**args)-c
        expected+=sign*KINDS[channel][2][a]*SHAPE.integrate(out_cutoff=1,in_cutoff=0,
            out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],b_GeV=B,
            field_precision=2*SHAPE.a,field_maps=transform@g["physical_field_maps"],
            field_shifts=transform@shifted,constant=c,linear=lin)
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,current.total[2:]),g["jacobian"]*expected,atol=6e-15)
    assert current.reference_shape is SHAPE and not current.physical


@pytest.mark.parametrize("case",CASES)
def test_shaped_four_leg_current_and_full_projector_close_same_finite_Hamiltonian(case):
    inc,out,pair,si,so,ei,ef=case;h,wi,wf=data(case);h=replace(h,reference_shape=SHAPE)
    ci,co=0,1;projected=h.integrate(co,ci)
    args=dict(species=si,out_species=so,charges=tuple(ef[j] for j in pair)+tuple(ei[j] for j in pair),
        regulator=regulator(),total_pplus_GeV=PP,transfer_GeV=D,hamiltonian_coefficient=1.,
        canonical_current_coefficient=(0.,0.,0.,0.),prescription=PRESCRIPTION,quadrature_order=12,reference_shape=SHAPE)
    physical=cluster_pair_current_orbitals(inc,out,pair,ci,co,B,mean_total_transverse_GeV=(0.,0.),**args)
    boundary=exact_transverse_projector_boundary(h,out_cutoff=co,in_cutoff=ci,
        out_weights=wf,in_weights=wi,out_charges=ef,in_charges=ei,transfer_GeV=D,total_pplus_GeV=PP,quadrature_order=12)
    expected=projected@density(wi,ei,ci,ci)-density(wf,ef,co,co)@projected
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical.total[2:]+boundary),expected,atol=3e-15)
    # This keeps charged spectator attachments to the finite HO boundary.
    full=sum(ch*h.integrate(co,ci,in_shift=-w[:,None]*D) for w,ch in zip(wi,ei))
    full-=sum(ch*h.integrate(co,ci,out_shift=w[:,None]*D) for w,ch in zip(wf,ef))
    np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,physical.total[2:]),full,atol=3e-15)
    beta=np.array([.23,-.16])
    boosted=cluster_pair_current_orbitals(inc,out,pair,ci,co,B,mean_total_transverse_GeV=PP*beta,**args)
    np.testing.assert_allclose(boosted.total[2:],physical.total[2:],atol=3e-15)
    np.testing.assert_allclose(boosted.total[1],physical.total[1]+np.einsum("j,jab->ab",beta,physical.total[2:]),atol=3e-15)


@pytest.mark.parametrize("route",QUARTIC_ROUTES)
def test_every_quartic_source_family_shares_whole_vertex_shape_and_current_derivative(route):
    hs=hs_nonzero(route);mean=(.04,.03);reg=regulator()
    current=cluster_quartic_source_current(route,0,1,B,hs,regulator=reg,total_pplus_GeV=PP,
        transfer_GeV=D,mean_total_transverse_GeV=mean,emitter_charge=.7,pair_charge=-.2,
        prescription=PRESCRIPTION,quadrature_order=12,reference_shape=SHAPE)
    g=quartic_source_geometry(route,transfer_GeV=D,mean_total_transverse_GeV=mean)
    charges=quartic_source_charges(route,emitter_charge=.7,pair_charge=-.2)
    transform=reg.leg_transform(g["modes"]);expected=0.
    for a,sign in enumerate((-1,-1,-1,1)):
        shifted=g["physical_field_shifts"].copy();shifted[a]+=sign*D
        expected+=sign*charges[a]*SHAPE.integrate(out_cutoff=1,in_cutoff=0,
            out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],b_GeV=B,
            field_precision=2*SHAPE.a,field_maps=transform@g["physical_field_maps"],field_shifts=transform@shifted)
    h,_=coefficients(route,hs)
    np.testing.assert_allclose(2*PP*np.einsum("j,joiab->oiab",D,current.total[2:]),
        g["jacobian"]*np.einsum("oi,ab->oiab",expected,h),atol=3e-15)
    assert current.reference_shape is SHAPE and not current.physical
