import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator,gmres

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace,build_excluded_source
from deuteron_wigner.quantum.m2_six_body_space import SixBodySpace,density_action_between
from deuteron_wigner.quantum.m2_excluded_fock_matching import intermediate_density_action_between
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import build_combined_pair_source
from deuteron_wigner.quantum.m2_instantaneous_sea_exit import build_instantaneous_sea_source
from deuteron_wigner.quantum.m2_order_resolved_reference import OrderResolvedFockReference
from deuteron_wigner.quantum.m2_shell_current import shell_density_between
from deuteron_wigner.quantum.m2_shell_fock_reference import ShellFockReference


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar","qqqggg","qqqq-qbar-g"])
@pytest.mark.parametrize("jz",[-1,1])
def test_factored_density_matches_original_full_tensor_with_both_spin_endpoints(sector,jz):
    cls=IntermediateSpace if sector in ("qqqgg","qqqq-qbar") else SixBodySpace
    old=intermediate_density_action_between if cls is IntermediateSpace else density_action_between
    s=cls(sector,9,1,"NEUTRON",1);f=cls(sector,9,1,"NEUTRON",jz)
    rng=np.random.default_rng(687)
    v=(rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2)))/np.sqrt(s.dimension)
    for charge in ("electromagnetic","up_number"):
        expected=old(f,s,v,(.12,.16),b_GeV=.4,charge=charge)
        actual=shell_density_between(f,s,v,(.12,.16),b_GeV=.4,charge=charge)
        np.testing.assert_allclose(actual,expected,atol=4e-17)


@pytest.mark.parametrize("sector",["qqqgg","qqqq-qbar"])
def test_second_shell_factored_density_matches_original_and_preserves_statistics(sector):
    s=IntermediateSpace(sector,9,2,"PROTON",1)
    rng=np.random.default_rng(775)
    v=s.statistics_project((rng.normal(size=(s.dimension,1))+1j*rng.normal(size=(s.dimension,1)))/np.sqrt(s.dimension))
    expected=intermediate_density_action_between(s,s,v,(.12,.16),b_GeV=.4,charge="electromagnetic")
    actual=shell_density_between(s,s,v,(.12,.16),b_GeV=.4,charge="electromagnetic")
    np.testing.assert_allclose(actual,expected,atol=8e-17)
    np.testing.assert_allclose(s.statistics_project(actual),actual,atol=2e-16)


@pytest.fixture(scope="module",params=[0,1,2])
def systems(request):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    cutoff=request.param
    cubic=[build_excluded_source(p,s,out_cutoff=cutoff) for s in ("qqqgg","qqqq-qbar")]
    quartic=[build_combined_pair_source(p,out_cutoff=cutoff),
             build_instantaneous_sea_source(p,out_cutoff=cutoff)]
    ref=ShellFockReference(p,cubic_sources=cubic,quartic_sources=quartic)
    old=OrderResolvedFockReference(p,cubic_sources=cubic,quartic_sources=quartic) if cutoff<2 else None
    return ref,old,cutoff


def test_source_returns_recover_previous_reference_or_independent_full_resolvent(systems):
    ref,old,cutoff=systems
    for n in (2,3,4):
        h=ref.return_coefficient(n,1.4)
        norm=ref.return_coefficient(n,1.4,norm=True)
        np.testing.assert_allclose(h,h.conj().T,atol=2e-16)
        slope=(ref.return_coefficient(n,1.4+1e-5)-ref.return_coefficient(n,1.4-1e-5))/2e-5
        np.testing.assert_allclose(slope,-norm,atol=4e-11)
        if old is not None:
            np.testing.assert_allclose(h,old.return_coefficient(n,1.4),atol=2e-16)
            np.testing.assert_allclose(norm,old.return_coefficient(n,1.4,norm=True),atol=2e-16)
    if cutoff==2:
        rng=np.random.default_rng(769)
        v=rng.normal(size=len(ref.pilot.matrix))+1j*rng.normal(size=len(ref.pilot.matrix))
        v/=np.linalg.norm(v)
        returned=np.zeros_like(v);probability=0.
        for s in ref.sectors:
            source=s.source.full_coupling(ref.pilot.coupling)
            right=source@v
            op=LinearOperator((len(right),len(right)),dtype=complex,
                 matvec=lambda x,s=s:1.4*x-s.propagation.apply(x[:,None])[:,0])
            wave,info=gmres(op,right,rtol=2e-13,atol=0)
            assert info==0
            returned+=source.conj().T@wave
            probability+=np.vdot(wave,wave).real
        np.testing.assert_allclose(returned,(ref.effective_hamiltonian(1.4)-ref.pilot.matrix)@v,atol=3e-15)
        assert probability==pytest.approx(np.vdot(v,(ref.norm_metric(1.4)-np.eye(len(v)))@v).real,abs=2e-16)


@pytest.mark.parametrize("charge,q",[("baryon",1),("electromagnetic",1),("up_number",2),("down_number",1)])
def test_charge_norm_all_orders_at_each_shell(systems,charge,q):
    ref,_,_=systems
    for n in (2,3,4):
        np.testing.assert_allclose(ref.current_coefficient(n,1.4,1.4,(0,0),charge=charge),
                                  q*ref.return_coefficient(n,1.4,norm=True),atol=3e-16)


def test_two_energy_current_and_reverse_with_previous_reference_control(systems):
    ref,old,_=systems
    for n in (2,3,4):
        forward=ref.current_coefficient(n,1.45,1.32,(.12,.16),charge="electromagnetic")
        reverse=ref.current_coefficient(n,1.32,1.45,(-.12,-.16),charge="electromagnetic")
        np.testing.assert_allclose(forward,reverse.conj().T,atol=4e-16)
        if old is not None:
            np.testing.assert_allclose(forward,old.current_coefficient(n,1.45,1.32,(.12,.16),charge="electromagnetic"),atol=4e-16)


def test_missing_sources_and_mismatched_current_spaces_are_not_assumed():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                         quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    with pytest.raises(ValueError,match="source sectors"):
        ShellFockReference(p,cubic_sources=[],quartic_sources=[])
    s=IntermediateSpace("qqqgg",9,0,"PROTON",1)
    f=IntermediateSpace("qqqgg",9,1,"PROTON",1)
    with pytest.raises(ValueError,match="cutoff"):
        shell_density_between(f,s,np.zeros((s.dimension,1)),(0,0),b_GeV=.4,charge="baryon")
