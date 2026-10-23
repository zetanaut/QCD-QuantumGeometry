"""Two-energy/order matching against whole raw parent-current matrix elements."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between
from deuteron_wigner.quantum.m2_shell_transverse_matching import FreeQ5TransverseMatching
from deuteron_wigner.quantum.m2_matched_h2_reduction import _inverse_sqrt


def reference(target,jz,cutoff):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    return build_shell_fock_reference(p,out_cutoff=cutoff)


@pytest.fixture(scope="module",params=[
    (t,j,n) for t in ("PROTON","NEUTRON") for j in (-1,1) for n in (0,1)])
def model(request):
    target,jz,cutoff=request.param
    incoming=reference(target,1,cutoff)
    outgoing=incoming if jz==1 else reference(target,jz,cutoff)
    return FreeQ5TransverseMatching(outgoing,incoming,(.12,.16),
                                    total_pplus_GeV=1.3,charge="electromagnetic")


def whole_parent_element(model,vf,vi,ef,ei,g):
    """Lift full coupling columns directly, not the order-coefficient waves."""
    m=model
    value=np.array([np.vdot(vf,j@vi) for j in m.retained_free+g*m.retained_contact])
    contact=np.zeros(2,complex)
    for sf,si,forward,backward in m.sectors:
        qf=sf.propagation.resolve(ef,sf.source.full_coupling(g)@vf[:,None])
        qi=si.propagation.resolve(ei,si.source.full_coupling(g)@vi[:,None])
        for j in (0,1):
            left=np.vdot(vf[backward.retained_columns],
                         g*backward.current_per_gs[j].conj().T@qi[:,0])
            right=np.vdot(qf[:,0],g*forward.current_per_gs[j]@vi[forward.retained_columns])
            qcurrent=shell_transverse_current_between(sf.source.space,si.source.space,qi,
                m.transfer,component=j,b_GeV=m.incoming.pilot.width_GeV,
                total_pplus_GeV=m.total_pplus_GeV,charge=m.charge)
            value[j]+=left+right+np.vdot(qf,qcurrent)
            contact[j]+=left+right
    return value,contact


def test_all_orders_against_raw_parent_at_distinct_energies_and_couplings(model):
    m=model;rng=np.random.default_rng(887)
    nf,ni=len(m.outgoing.pilot.matrix),len(m.incoming.pilot.matrix)
    vf=rng.normal(size=nf)+1j*rng.normal(size=nf);vf/=np.linalg.norm(vf)
    vi=rng.normal(size=ni)+1j*rng.normal(size=ni);vi/=np.linalg.norm(vi)
    ef,ei=1.39,1.47
    coefs=m.coefficients(ef,ei)
    assert set(coefs)==set(range(5))
    for g in (0.,-.17,.3):
        expected,_=whole_parent_element(m,vf,vi,ef,ei,g)
        actual=np.array([np.vdot(vf,j@vi) for j in sum(g**n*a for n,a in coefs.items())])
        np.testing.assert_allclose(actual,expected,atol=3e-15)
    np.testing.assert_allclose(m.effective_current(ef,ei),sum(.3**n*a for n,a in coefs.items()),atol=0.)
    assert m.complete_physical_four_current is False
    assert m.interacting_QQ_Q6_current is False


def test_each_coupling_order_obeys_reversed_transfer_and_two_energy_adjoint(model):
    m=model
    reverse=FreeQ5TransverseMatching(m.incoming,m.outgoing,tuple(-x for x in m.transfer),
                                      total_pplus_GeV=1.3,charge=m.charge)
    f=m.coefficients(1.39,1.47);b=reverse.coefficients(1.47,1.39)
    for n in f:
        np.testing.assert_allclose(f[n],b[n].conj().swapaxes(1,2),atol=4e-15)


def test_both_norm_factors_preserve_whole_parent_matrix_element(model):
    m=model;rng=np.random.default_rng(886)
    vf=rng.normal(size=len(m.outgoing.pilot.matrix))+1j*rng.normal(size=len(m.outgoing.pilot.matrix))
    vi=rng.normal(size=len(m.incoming.pilot.matrix))+1j*rng.normal(size=len(m.incoming.pilot.matrix))
    vf/=np.linalg.norm(vf);vi/=np.linalg.norm(vi)
    ef,ei=1.39,1.47
    left=_inverse_sqrt(m.outgoing.norm_metric(ef))@vf
    right=_inverse_sqrt(m.incoming.norm_metric(ei))@vi
    expected,_=whole_parent_element(m,left,right,ef,ei,.3)
    actual=np.array([np.vdot(vf,j@vi) for j in m.normalized_current(ef,ei)])
    np.testing.assert_allclose(actual,expected,atol=3e-15)


def test_second_shell_contact_contribution_is_not_lost_in_good_density_matching():
    i=reference("PROTON",1,2);f=reference("PROTON",-1,2)
    m=FreeQ5TransverseMatching(f,i,(.12,.16),total_pplus_GeV=1.3,charge="electromagnetic")
    rng=np.random.default_rng(894)
    vi=rng.normal(size=37)+1j*rng.normal(size=37);vi/=np.linalg.norm(vi)
    vf=rng.normal(size=37)+1j*rng.normal(size=37);vf/=np.linalg.norm(vf)
    expected,contact=whole_parent_element(m,vf,vi,1.39,1.47,.3)
    actual=np.array([np.vdot(vf,j@vi) for j in m.effective_current(1.39,1.47)])
    np.testing.assert_allclose(actual,expected,atol=3e-15)
    assert np.linalg.norm(contact)>1e-6


def test_wrong_reference_or_endpoint_labels_are_not_assumed():
    with pytest.raises(ValueError):
        FreeQ5TransverseMatching(None,None,(0,0),total_pplus_GeV=1.,charge="baryon")
    i=reference("PROTON",1,0);f=reference("NEUTRON",1,0)
    with pytest.raises(ValueError):
        FreeQ5TransverseMatching(f,i,(0,0),total_pplus_GeV=1.,charge="electromagnetic")
