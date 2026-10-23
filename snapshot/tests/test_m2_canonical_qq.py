"""Coherent canonical QQ source action and all single-insertion source orders."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction,CanonicalQQInsertion
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference


@pytest.fixture(scope="module")
def insertion():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    ref=build_order_resolved_reference(p,out_cutoff=0)
    return CanonicalQQInsertion(ref,action=CanonicalQQSourceAction(ref))


def test_complete_component_sum_on_actual_source_and_old_leading_alias(insertion):
    obj=insertion;a,_=obj.order_waves(1.4,1)
    parts=obj.action.component_returns(a)
    assert set(parts)=={"qqqgg_fermion_scattering","qqqq-qbar_fermion_scattering",
        "qg_number_scattering","gg_number_scattering","IF_scattering","A4",
        "pair_current_square","IF_pair_conversion"}
    total=sum(parts.values())
    np.testing.assert_allclose(total,total.conj().T,atol=3e-14)
    np.testing.assert_allclose(total,obj.hamiltonian_coefficient(1.4),atol=3e-14)
    np.testing.assert_allclose(total,obj.return_power_coefficient(4,1.4),atol=3e-14)
    assert np.linalg.norm(parts["IF_pair_conversion"])>1e-7


@pytest.mark.parametrize("power",[4,5,6])
def test_each_order_norm_energy_derivative_and_four_charge_identities(insertion,power):
    obj=insertion;e=1.4
    n=obj.return_power_coefficient(power,e,norm=True)
    slope=(obj.return_power_coefficient(power,e+1e-5)-obj.return_power_coefficient(power,e-1e-5))/2e-5
    np.testing.assert_allclose(slope,-n,atol=3e-12)
    assert np.linalg.norm(obj.return_power_coefficient(power,e))>1e-7
    for charge,total in (("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)):
        np.testing.assert_allclose(obj.current_power_coefficient(power,e,e,(0.,0.),charge=charge),
                                   total*n,atol=3e-14)


@pytest.mark.parametrize("power",[4,5,6])
def test_each_order_two_energy_current_reversal(insertion,power):
    obj=insertion
    a=obj.current_power_coefficient(power,1.46,1.31,(.12,.16),charge="electromagnetic")
    b=obj.current_power_coefficient(power,1.31,1.46,(-.12,-.16),charge="electromagnetic")
    np.testing.assert_allclose(a,b.conj().T,atol=3e-14)


def test_all_powers_recombine_full_g_weighted_single_W_response(insertion):
    obj=insertion;e=1.4
    for g in (-.3,0.,.3):
        v={name:sum(g**p*obj.sources_by_power[p][name] for p in (1,2)) for name in obj.sources}
        a={name:source/(e-obj.energies[name][:,None]) for name,source in v.items()}
        w=obj.action.apply(a)
        expected=g*g*sum(a[name].conj().T@w[name] for name in a)
        actual=sum(g**power*obj.return_power_coefficient(power,e) for power in (4,5,6))
        np.testing.assert_allclose(actual,expected,atol=3e-14)
    # A sign change isolates g^5; it is not an extra g^4 vertex.
    assert np.linalg.norm(obj.return_power_coefficient(5,e))>1e-7


@pytest.mark.parametrize("power",[True,3,7])
def test_unrepresented_single_insertion_orders_rejected(insertion,power):
    with pytest.raises(ValueError):insertion.return_power_coefficient(power,1.4)

