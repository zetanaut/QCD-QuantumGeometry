"""Canonical J+ normalization, proton/neutron charges and matched sea current."""
from math import sqrt

import numpy as np
import pytest

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density,parton_charge
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot,baryon_density
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference,intermediate_density_action

@pytest.mark.parametrize("factory",[lf_spinor,antiquark_spinor])
def test_good_current_is_spin_identity_with_physical_field_normalization(factory):
    a=(.7,.21,-.1); b=(2.1,-.5,.4)
    u=np.column_stack([factory(*a,.35,h)/sqrt(2*a[0]) for h in (-1,1)])
    v=np.column_stack([factory(*b,.35,h)/sqrt(2*b[0]) for h in (-1,1)])
    np.testing.assert_allclose(v.conj().T@GAMMA[0]@GAMMA_PLUS@u,np.eye(2),atol=4e-15)
    # The minus for an antiparticle charge comes from normal-ordering d d†,
    # not from a negative v-bar gamma+ v or a helicity relabeling.

@pytest.mark.parametrize("species",["q","qbar","g"])
def test_charge_flavor_decomposition_and_antiquark_sign(species):
    spins=np.arange(2 if species=="g" else 4)
    u=parton_charge(spins,species,"up_number")
    d=parton_charge(spins,species,"down_number")
    np.testing.assert_allclose(parton_charge(spins,species,"baryon"),(u+d)/3,atol=0)
    np.testing.assert_allclose(parton_charge(spins,species,"electromagnetic"),(2*u-d)/3,atol=0)
    if species!="g":
        np.testing.assert_allclose(parton_charge(spins,"q","electromagnetic"),
                                   -parton_charge(spins,"qbar","electromagnetic"),atol=0)

@pytest.fixture(scope="module",params=["PROTON","NEUTRON"])
def reference(request):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=request.param,jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    return ExcludedFockReference(p,[build_excluded_source(p,s,out_cutoff=1)
                                   for s in ("qqqgg","qqqq-qbar")])

def test_exact_charges_in_both_h2_and_actual_excluded_sectors(reference):
    p=reference.pilot
    charges={"baryon":1,"electromagnetic":int(p.target=="PROTON"),
             "up_number":2 if p.target=="PROTON" else 1,
             "down_number":1 if p.target=="PROTON" else 2}
    for charge,expected in charges.items():
        np.testing.assert_allclose(good_current_density(p,(0.,0.),charge=charge),
                                   expected*np.eye(len(p.matrix)),atol=1e-14)
        for sector in reference.sectors:
            s=sector.source
            actual=intermediate_density_action(s.space,s.matrix_per_gs,(0.,0.),b_GeV=.4,charge=charge)
            np.testing.assert_allclose(actual,expected*s.matrix_per_gs,atol=1e-14)

def test_finite_transfer_flavor_resolved_good_current_and_existing_baryon_oracle(reference):
    p=reference.pilot
    delta=(.12,.16)
    u=good_current_density(p,delta,charge="up_number")
    d=good_current_density(p,delta,charge="down_number")
    np.testing.assert_allclose((u+d)/3,baryon_density(p,delta),atol=2e-15)
    np.testing.assert_allclose((2*u-d)/3,good_current_density(p,delta,charge="electromagnetic"),atol=2e-15)
    assert np.linalg.norm(u-d)>.1
    for sector in reference.sectors:
        s=sector.source
        v=s.matrix_per_gs[:,:3]
        action=lambda charge:intermediate_density_action(s.space,v,delta,b_GeV=.4,charge=charge)
        up,down=action("up_number"),action("down_number")
        np.testing.assert_allclose((up+down)/3,action("baryon"),atol=1e-14)
        np.testing.assert_allclose((2*up-down)/3,action("electromagnetic"),atol=1e-14)

def test_matched_electromagnetic_good_current_charge_norm_and_flow(reference):
    r=reference.reduction
    state=r.lowest_state()
    e=state.energy_GeV2
    expected=int(reference.pilot.target=="PROTON")
    charge=reference.density((0.,0.),charge="electromagnetic")
    np.testing.assert_allclose(r.matched_operator(charge,e,e),expected*r.norm_metric(e),atol=2e-14)
    rho,jx,jy=reference.flow_current((.12,.16),charge="electromagnetic")
    effective=r.matched_operator(rho,e,e)
    assert np.vdot(state.retained,effective@state.retained)==pytest.approx(
        np.vdot(state.reconstructed,rho@state.reconstructed),abs=2e-14)
    np.testing.assert_allclose(.12*jx+.16*jy,r.h@rho-rho@r.h,atol=2e-14)
    assert np.linalg.norm(jx[len(state.retained):,:len(state.retained)])>.001
    np.testing.assert_allclose(reference.density((-.12,-.16),charge="electromagnetic"),
                               rho.conj().T,atol=2e-14)

def test_unsupported_current_is_not_assigned_a_default_charge():
    with pytest.raises(ValueError):
        parton_charge(np.arange(4),"q","axial")
