from dataclasses import replace
import numpy as np
import pytest
from scipy import linalg
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import build_combined_pair_source
from deuteron_wigner.quantum.m2_instantaneous_sea_exit import build_instantaneous_sea_source
from deuteron_wigner.quantum.m2_order_resolved_reference import OrderResolvedFockReference
from deuteron_wigner.quantum.m2_spin_resolved_current import (
    parity_aligned_partner,matched_good_current_between,good_form_factor_diagnostic,
)


def make(target="PROTON",jz=1,cutoff=1,coupling=.3):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=coupling)
    cubic=[build_excluded_source(p,s,out_cutoff=cutoff) for s in ("qqqgg","qqqq-qbar")]
    quartic=[build_combined_pair_source(p,out_cutoff=cutoff),build_instantaneous_sea_source(p,out_cutoff=cutoff)]
    return OrderResolvedFockReference(p,cubic_sources=cubic,quartic_sources=quartic),cubic,quartic


@pytest.fixture(scope="module")
def system():
    return make()


def test_each_power_reconstructs_before_g_weighting_and_keeps_new_source_directions(system):
    ref,cubic,quartic=system
    assert len(ref.matrix)>len(ExcludedFockReference(ref.pilot,cubic).matrix)
    for sector,a,b in zip(ref.sectors,cubic,quartic):
        for power,source,raw in ((1,a,a.matrix_per_gs),(2,b,b.matrix_per_gs2)):
            actual=sector.lift(sector.coupling_by_power[power])
            expected=np.zeros_like(actual);expected[:,source.retained_columns]=raw
            np.testing.assert_allclose(actual,expected,atol=3e-14)
            assert sector.reconstruction_residual_by_power[power]<3e-14
        for g in (0.,1e-8,-.3,.3):
            np.testing.assert_allclose(sector.lift(sector.coupling(g)),
                                       sector.source.full_coupling(g),atol=3e-14)


def test_coherent_return_polynomial_norm_derivative_and_g3_failure_control(system):
    ref,_,_=system;g=ref.pilot.coupling;e=1.4
    coefficients=[ref.return_coefficient(n,e) for n in (2,3,4)]
    sigma=sum(g**n*a for n,a in zip((2,3,4),coefficients))
    np.testing.assert_allclose(ref.reduction.effective_hamiltonian(e),ref.pilot.matrix+sigma,atol=3e-14)
    assert np.linalg.norm(coefficients[1])>1e-5
    nq=ref.pilot.bases[0].shape[1]
    np.testing.assert_allclose(coefficients[1][:nq,:nq],0.,atol=1e-17)
    np.testing.assert_allclose(coefficients[1][nq:,nq:],0.,atol=1e-17)
    assert np.linalg.norm(sigma-(g*g*coefficients[0]+g**4*coefficients[2]))>1e-6
    norm=np.eye(len(ref.pilot.matrix))+sum(g**n*ref.return_coefficient(n,e,norm=True) for n in (2,3,4))
    np.testing.assert_allclose(norm,ref.reduction.norm_metric(e),atol=3e-14)
    for n in (2,3,4):
        slope=(ref.return_coefficient(n,e+1e-5)-ref.return_coefficient(n,e-1e-5))/2e-5
        np.testing.assert_allclose(slope,-ref.return_coefficient(n,e,norm=True),atol=2e-11)


@pytest.mark.parametrize("charge,total",[("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)])
def test_current_powers_match_both_energy_endpoints_and_exact_zero_transfer_charge(system,charge,total):
    ref,_,_=system;g=ref.pilot.coupling;ef,ei=1.46,1.31;delta=(.12,.16)
    op=sum(g**n*ref.current_coefficient(n,ef,ei,delta,charge=charge) for n in (0,2,3,4))
    direct=ref.reduction.matched_operator(ref.density(delta,charge=charge),ef,ei)
    np.testing.assert_allclose(op,direct,atol=4e-14)
    for n in (0,2,3,4):
        reverse=ref.current_coefficient(n,ei,ef,(-.12,-.16),charge=charge)
        np.testing.assert_allclose(ref.current_coefficient(n,ef,ei,delta,charge=charge),reverse.conj().T,atol=4e-14)
        expected=total*(np.eye(len(ref.pilot.matrix)) if n==0 else ref.return_coefficient(n,1.4,norm=True))
        np.testing.assert_allclose(ref.current_coefficient(n,1.4,1.4,(0.,0.),charge=charge),expected,atol=3e-14)


def test_parent_and_uncompressed_equations_match_solved_h2_state(system):
    ref,_,_=system
    root=ref.reduction.lowest_state()
    assert root.energy_GeV2==pytest.approx(linalg.eigvalsh(ref.matrix)[0],abs=4e-14)
    residual,probability=ref.full_reconstructed_residual(root.energy_GeV2,root.retained)
    assert residual<4e-14
    assert probability==pytest.approx(root.excluded_probability,abs=2e-16)


def test_both_canonical_helicities_parity_and_induced_pauli(system):
    up,_,_=system;down,_,_=make(jz=-1)
    root=up.reduction.lowest_state();e=root.energy_GeV2
    states=[root.retained,parity_aligned_partner(down.pilot,up.pilot,root.retained,intrinsic_parity=1)]
    full=down.reduction.wave_operator(e)@states[1]
    assert np.linalg.norm(down.matrix@full-e*full)<4e-14
    refs=[up,down];delta=(.12,.16)
    matrix=np.array([[vf.conj()@matched_good_current_between(rf,ri,e,e,delta,charge="electromagnetic")@vi
                      for ri,vi in zip(refs,states)] for rf,vf in zip(refs,states)])
    ff=good_form_factor_diagnostic(matrix,delta,mass_GeV=np.sqrt(e),total_pplus=3.)
    assert np.linalg.norm(ff.residual_matrix)<4e-14
    assert abs(ff.pauli.real)>1e-9


def test_missing_mismatched_and_wrong_column_sources_are_not_inferred(system):
    ref,cubic,quartic=system
    with pytest.raises(ValueError,match="sectors"):
        OrderResolvedFockReference(ref.pilot,cubic_sources=cubic,quartic_sources=quartic[:1])
    wrong=replace(quartic[0],retained_columns=np.arange(len(quartic[0].retained_columns))+1)
    with pytest.raises(ValueError,match="columns"):
        OrderResolvedFockReference(ref.pilot,cubic_sources=cubic,quartic_sources=[wrong,quartic[1]])
    wrong=replace(quartic[0],retained_pilot=replace(ref.pilot,coupling=.4))
    with pytest.raises(ValueError,match="exact retained"):
        OrderResolvedFockReference(ref.pilot,cubic_sources=cubic,quartic_sources=[wrong,quartic[1]])


def test_zero_coupling_preserves_the_independent_source_range_but_decouples_the_parent(system):
    zero,_,_=make(coupling=0.)
    assert len(zero.matrix)==len(system[0].matrix)
    n=len(zero.pilot.matrix)
    np.testing.assert_allclose(zero.matrix[n:,:n],0.,atol=0.)
    np.testing.assert_allclose(zero.reduction.norm_metric(1.4),np.eye(n),atol=0.)
    assert zero.reduction.lowest_state().energy_GeV2==pytest.approx(
        linalg.eigvalsh(zero.pilot.matrix)[0],abs=3e-14)


@pytest.mark.parametrize("power",[1,5,True])
def test_unrepresented_return_orders_are_rejected(system,power):
    with pytest.raises(ValueError):system[0].return_coefficient(power,1.4)
