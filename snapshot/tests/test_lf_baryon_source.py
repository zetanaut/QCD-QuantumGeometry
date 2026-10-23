from math import sqrt

import numpy as np
import pytest

from deuteron_wigner.quantum.finite_lf_coupling import reference_inputs
from deuteron_wigner.quantum.lf_baryon_emission import norm2,difference,occupied
from deuteron_wigner.quantum.lf_pair_electromagnetic import phi_source_coefficients,PHI_CHARGE
from deuteron_wigner.quantum.lf_baryon_source import BaryonSource,gluon_contact,covariant_phi_source,source_harmonics
from deuteron_wigner.quantum.modular_gauge_network import generators


P=(.375,.2,0.);Q=(0,-.2,.2);C=np.array([[.14,.037],[.037,-.06]])


def make(*,representation='good',gauge=False,species='p',sign=1,shell=1,epsilon=(1,0,0,0)):
    cell,m,g,_=reference_inputs()
    return BaryonSource(cell,m,C,g,.43,.27,[P],sign*np.array(Q),shell,12,epsilon,
                        lambda l:l.interpolator(P,species,1),pure_gauge=gauge,representation=representation)


@pytest.fixture(scope='module')
def ordinary(): return make()


@pytest.fixture(scope='module')
def gauge(): return make(gauge=True)


@pytest.fixture(scope='module')
def ordinary_wave(ordinary): return ordinary.response(nodes=64)


@pytest.mark.parametrize('component',range(4))
def test_gluon_contact_derived_from_constraint_and_independent_instantaneous(component):
    cell,*_=reference_inputs();a=(.125,.2,-.2);k=(.25,-.2,.2);p=(.375,.2,.2);e=np.eye(4)[component]
    good=gluon_contact(a,k,p,.27,e,cell.volume)
    cov=gluon_contact(a,k,p,.27,e,cell.volume,covariant=True)
    assert np.linalg.norm(good-cov)<3e-16
    assert np.linalg.norm(good)>0 if component!=1 else np.linalg.norm(good)==0


@pytest.mark.parametrize('omega',[-.17,.11,.38])
@pytest.mark.parametrize('component',range(4))
def test_collective_source_and_outgoing_endpoint_from_actual_covariant_vertex(omega,component):
    _,m,_,_=reference_inputs();a=(.25,.2,0);b=(.25,0,-.2);e=np.eye(4)[component]
    j0,j1,j2,w=phi_source_coefficients(m,C,a,b,e)
    jref,wref=covariant_phi_source(m,C,a,b,e,omega)
    assert np.linalg.norm(j0+omega*j1+omega**2*j2-jref)<2e-15
    assert np.linalg.norm(w-wref)<2e-15
    reverse,wr=covariant_phi_source(m,C,b,a,e,-omega)
    assert np.linalg.norm(reverse-jref.conj().T)<2e-15
    assert np.linalg.norm(wr-wref.conj().T)<2e-15


def test_complete_baryon_fourier_wave_refined_independent_Dirac_ODE(ordinary,ordinary_wave):
    reference=make(representation='dirac')
    # Different formulae, different CAR implementation, physical-basis ODE;
    # no diagonalization/spectral quadrature is reused by this reference.
    coarse,_=reference.ode_response()
    fine,_=reference.ode_response(rtol=3e-12,atol=3e-14)
    assert difference(ordinary_wave,fine)<5e-11
    assert difference(ordinary_wave,fine)<difference(ordinary_wave,coarse)/20
    assert difference(ordinary_wave,ordinary.response(nodes=96))<2e-14
    assert norm2(ordinary_wave)>.01


def test_pure_gauge_whole_baryon_wave_and_required_contact_routes(gauge):
    wave=gauge.response()
    assert sqrt(norm2(wave))<2e-14
    for omission in ('before','after_quark','collective','fermion_contact','outside_intermediates'):
        assert sqrt(norm2(gauge.response(omit=omission)))>.001
    assert gauge.diagnostics()['outside_intermediate_blocks']>0


def test_spectator_omission_passes_gauge_but_changes_ordinary_wave(ordinary,ordinary_wave,gauge):
    missing=ordinary.response(omit='spectators')
    assert difference(ordinary_wave,missing)>.01
    # Spectator response is separately gauge invariant. A gauge-only test
    # cannot certify that this actual physical contribution was retained.
    assert sqrt(norm2(gauge.response(omit='spectators')))<2e-14


def test_collective_endpoint_independently_needed_for_general_source(ordinary,ordinary_wave):
    assert difference(ordinary_wave,ordinary.response(omit='collective_endpoint'))>.001


def check_charge_and_momentum(obj,wave,charge):
    error=0.;target=3*np.array(P)+obj.q
    for (kind,k,mask),v in wave.items():
        quarks=occupied(mask)
        assert len(quarks)==3
        total=np.sum([obj.layer.momenta[i//12] for i in quarks],axis=0)+k
        assert np.linalg.norm(total-target)<1e-13
        q=sum(2/3 if (i%12)//3%2==0 else -1/3 for i in quarks)
        qb=np.kron(np.eye(2),PHI_CHARGE) if kind=='phi' else np.zeros((16,16))
        error+=np.linalg.norm((q-charge)*v+qb@v)**2
    assert sqrt(error)<2e-14


def test_full_state_charge_and_momentum_not_only_selected_components(ordinary,ordinary_wave):
    check_charge_and_momentum(ordinary,ordinary_wave,1)


def test_neutron_opposite_real_source_branch_and_all_source_components():
    obj=make(species='n',sign=-1,epsilon=(.37,-.22,.19,.11))
    wave=obj.response();check_charge_and_momentum(obj,wave,0)
    ref=make(species='n',sign=-1,representation='dirac',epsilon=(.37,-.22,.19,.11))
    assert difference(wave,ref.response())<3e-14
    assert norm2(wave)>1e-5


def test_source_created_pairs_not_assumed_or_source_domain_hidden():
    cell,m,g,_=reference_inputs()
    seed=lambda l:l.interpolator(P,'p',1)
    with pytest.raises(ValueError):
        BaryonSource(cell,m,C,g,.43,.27,[P],(.25,.2,0),1,12,(1,0,0,0),seed)
    with pytest.raises(ValueError):
        BaryonSource(cell,m,C,g,.43,.27,[P],Q,1,12,(1,1,0,0),seed,pure_gauge=True)


def test_unequal_momenta_source_routes_keep_all_quarks():
    cell,m,g,_=reference_inputs();p=(.375,0.,0.);s=(.125,0.,0.)
    def seed(l):
        packet=l.wavepacket_interpolator({p:1/sqrt(2),s:1j/sqrt(2)},'n',-1)
        return {mask:v for mask,v in packet.items() if sum(i//12==l.lookup[p] for i in occupied(mask))==1}
    obj=BaryonSource(cell,m,C,g,.43,.27,[p,s],(0,.2,0),1,12,(1,0,0,0),seed,pure_gauge=True)
    assert obj.nquarks==3
    assert abs(norm2(obj.seed)-.375)<1e-14
    assert sqrt(norm2(obj.response()))<2e-14


def test_mixed_gluon_contact_retains_full_color_intertwiner():
    cell,*_=reference_inputs();a=(.125,.2,-.2);k=(.25,-.2,.2);p=(.375,.2,.2)
    v=gluon_contact(a,k,p,.27,(.37,-.22,.19,.11),cell.volume)
    ts=generators()
    for t in ts:
        tq=np.kron(np.eye(4),t)
        adj=np.array([[-2*np.trace((t@b-b@t)@c) for c in ts] for b in ts])
        tg=np.kron(np.eye(2),adj)
        defect=np.einsum('ij,jak->iak',tq,v)+np.einsum('ab,ibk->iak',tg,v)-np.einsum('iaj,jk->iak',v,tq)
        assert np.linalg.norm(defect)<1e-16


def test_source_envelope_and_derivatives_independent_of_shared_solver_inputs():
    T=12.;eps=np.array([.37,-.22,.19,.11]);omega,modes=source_harmonics(T,Q,eps)
    wg,mg=source_harmonics(T,Q,(.5,0,0,0),pure_gauge=True)
    for t in np.linspace(0,T,13):
        s=np.sin(np.pi*t/T);c=np.cos(np.pi*t/T)
        f=s**4;df=4*np.pi/T*s**3*c;ddf=4*(np.pi/T)**2*(3*s*s*c*c-s**4)
        for order,expected in enumerate((f,df,ddf)):
            actual=np.einsum('k,kj->j',(-1j*omega)**order*np.exp(-1j*omega*t),modes)
            assert np.linalg.norm(actual-expected*eps)<4e-16
        pure=np.einsum('k,kj->j',np.exp(-1j*wg*t),mg)
        expected=.5*np.array([0,df,-1j*Q[1]*f,-1j*Q[2]*f])
        assert np.linalg.norm(pure-expected)<2e-16
