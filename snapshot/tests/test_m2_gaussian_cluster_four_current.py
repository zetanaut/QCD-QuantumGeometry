"""Combined four-current: independently evaluated H Ward identity and boosts."""
from dataclasses import replace
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_cluster_four_current import assemble_cluster_h2_four_current,ClusterH2FourCurrent
from deuteron_wigner.quantum.m2_total_isospin import (
    H2_ISOSPIN_PRESCRIPTION,require_half_isospin_H2_snapshot)
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_Q_projector_h2 import H_coefficients
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS,raw
from test_m2_gaussian_cluster_smooth_minus_actions import ZERO,BOOST,BETA


def assemble(f,i,d=D,context=ZERO):
    return assemble_cluster_h2_four_current(f,i,d,minus_context=context,
        loop_flavor_charges=(2/3,-1/3),field_envelope_prescription=SPECTRAL_PRESCRIPTION,
        isospin_prescription=H2_ISOSPIN_PRESCRIPTION,**ARGS)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_all_four_current_components_two_energies_norms_reversal_boost_and_Ward(target,jz):
    f=model(target=target,jz=jz).snapshot(1.39)
    i=model(target=target).snapshot(1.47)
    a,b,c=assemble(f,i),assemble(i,f,-D),assemble(f,i,context=BOOST)
    hf,hi=H_coefficients(f.owner,f.energy_GeV2),H_coefficients(i.owner,i.energy_GeV2)
    rho=good_current_density_between(f.owner.pilot,i.owner.pilot,D,charge="electromagnetic")
    ward=[];boost=[]
    for n,v in a.coefficients.items():
        np.testing.assert_allclose(v,b.coefficients[n].conj().transpose(0,2,1),atol=5e-14)
        transformed=v.copy()
        transformed[1]+=np.einsum("j,jab->ab",BETA,v[2:])+.5*np.dot(BETA,BETA)*v[0]
        transformed[2:]+=BETA[:,None,None]*v[0]
        boost.append(float(np.linalg.norm(c.coefficients[n]-transformed)))
        np.testing.assert_allclose(c.coefficients[n],transformed,atol=5e-14)
        expected=hf[n]@rho-rho@hi[n]
        if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
        residue=2*PP*np.einsum("j,jab->ab",D,v[2:])-expected
        ward.append(float(np.linalg.norm(residue)))
        np.testing.assert_allclose(residue,0.,atol=5e-14)
        np.testing.assert_allclose(f.whitener@residue@i.whitener,0.,atol=5e-14)
        boosted=c.coefficients[n]
        r=2*PP*np.einsum("j,jab->ab",D,boosted[2:])-2*np.dot(PP*BETA,D)*boosted[0]-expected
        np.testing.assert_allclose(r,0.,atol=5e-14)
    v=a.current(normalized=False)
    np.testing.assert_allclose(a.current(normalized=True),
        np.einsum("ab,mbc,cd->mad",f.whitener,v,i.whitener),atol=5e-14)
    assert a.all_represented_four_components_through_g4 and a.retained_isospin_H2_pullback_closed
    assert a.zero_plus_transfer_field_envelope_bound
    assert a.complete_H2_matched_current
    assert not a.raw_isospin_minus_supplied and not a.regulator_independence_established
    assert not a.physical_finite_conditions_selected and not a.physical
    assert np.linalg.norm(v[1]-a.physical_numerator.current(normalized=False)[1])>1e-8
    print(target,jz,"combined four-current Ward",ward,"boost",boost,flush=True)


def test_completion_is_finite_assembly_only_and_does_not_promote_partial_layers():
    from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import GaussianClusterH2,GaussianClusterSnapshot
    from deuteron_wigner.quantum.m2_gaussian_cluster_h2_current import ClusterH2CurrentAssembly
    from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape
    assert ClusterH2FourCurrent.complete_H2_matched_current
    assert 'finite H2 g4; Delta+=0' in ClusterH2FourCurrent.matching_scope
    assert not ClusterH2FourCurrent.physical and not ClusterH2FourCurrent.regulator_independence_established
    assert not ClusterH2FourCurrent.physical_finite_conditions_selected
    assert not ClusterH2FourCurrent.raw_isospin_minus_supplied
    assert not GaussianClusterH2.complete_regulated_H2_matching
    assert not ClusterH2CurrentAssembly.complete_H2_matched_current
    assert not GaussianReferenceShape.H2_transport_complete
    # The Hamiltonian snapshot still cannot silently choose a current.
    with pytest.raises(ValueError,match='explicit prescriptions'):
        GaussianClusterSnapshot.current(None)


def test_isospin_binding_rejects_impure_wave_and_unknown_representative():
    m=model();s=m.snapshot(1.39)
    residuals=require_half_isospin_H2_snapshot(s,prescription=H2_ISOSPIN_PRESCRIPTION)
    assert max(residuals.values())<1e-13
    with pytest.raises(ValueError,match="prescription"):
        require_half_isospin_H2_snapshot(s,prescription=None)
    waves=[{n:v.copy() for n,v in w.items()} for w in s.five_wave_coefficients]
    probe=raw(m.spaces5)
    for n in probe:waves[0][n][:,:probe[n].shape[1]]+=probe[n]
    changed=replace(s,five_wave_coefficients=tuple(waves))
    with pytest.raises(ValueError,match="outside total isospin"):
        require_half_isospin_H2_snapshot(changed,prescription=H2_ISOSPIN_PRESCRIPTION)
    assert max(require_half_isospin_H2_snapshot(s,prescription=H2_ISOSPIN_PRESCRIPTION).values())<1e-13


def test_combined_current_never_selects_unknown_prescriptions():
    m=model();f,i=m.snapshot(1.39),m.snapshot(1.47)
    args=dict(ARGS,loop_flavor_charges=(2/3,-1/3),minus_context=ZERO,
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,isospin_prescription=H2_ISOSPIN_PRESCRIPTION)
    for key in ("minus_context","field_envelope_prescription","isospin_prescription"):
        with pytest.raises(ValueError):assemble_cluster_h2_four_current(f,i,D,**{**args,key:None})
