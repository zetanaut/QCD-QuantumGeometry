import json
import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.quantum_pair_threshold import (
    analytic_spectral,phase_space_spectral,pair_amplitudes,collective_cut,
    feynman_nonlocal,dispersive_nonlocal,feynman_upper_boundary,
    tree_width_diagnostics,dot,
)
from deuteron_wigner.quantum.nonzero_momentum_current import (
    benchmark_reference,CollectiveChannel,QuarkCollectiveReference,
)


@pytest.mark.parametrize('q', [[1.,0,0,0],[1.,.1,-.2,.3],[1.4,-.3,.4,.6]])
@pytest.mark.parametrize('geometry',['rest','light_front'])
def test_twenty_source_quantum_cut_from_resolved_spinors(q,geometry):
    expected=analytic_spectral(q,.28)
    actual=phase_space_spectral(q,.28,geometry=geometry,order=4)
    np.testing.assert_allclose(actual,expected,atol=3e-15,rtol=1e-12)
    assert np.linalg.eigvalsh(actual).min()>-2e-15
    np.testing.assert_allclose(actual,actual.conj().T,atol=2e-15)


def test_four_component_source_boost_and_both_Ward_relations():
    q=np.array([1.,.1,-.2,.3]);mass=.28
    a=analytic_spectral(q,mass)
    contraction=q*np.array([1,-1,-1,-1])
    np.testing.assert_allclose(contraction@a[16:20],0,atol=2e-15)
    for flavor in range(3):
        p=1+flavor;sl=slice(4+4*flavor,8+4*flavor)
        np.testing.assert_allclose(contraction@a[sl],-1j*mass*a[p],atol=2e-15)
    erased=a.copy();erased[1:4,4:16]=0;erased[4:16,1:4]=0
    # All diagonal rates and positivity survive, but source interference is lost.
    np.testing.assert_array_equal(np.diag(erased),np.diag(a))
    assert np.linalg.eigvalsh(erased).min()>-1e-14
    assert np.linalg.norm(contraction@erased[4:8]+1j*mass*erased[1])>.01
    gen=np.zeros((4,4));gen[0,1:]=gen[1:,0]=[.3,-.2,.15]
    lorentz=expm(gen);source=np.eye(20)
    for offset in [4,8,12,16]: source[offset:offset+4,offset:offset+4]=lorentz
    boosted=phase_space_spectral(lorentz@q,mass,geometry='light_front',order=5)
    np.testing.assert_allclose(boosted,source@a@source.T,atol=3e-15,rtol=1e-12)


def test_pair_amplitudes_keep_color_flavor_and_vector_axial_phases():
    m=.3;p=np.array([.5,.3,0,.2645751311064591]);anti=p*np.array([1,-1,-1,-1])
    q=p+anti;amplitude=pair_amplitudes(p,anti,m)
    np.testing.assert_allclose(np.einsum('m,mij->ij',q*np.array([1,-1,-1,-1]),amplitude[16:]),0,atol=2e-15)
    for flavor in range(3):
        np.testing.assert_allclose(np.einsum('m,mij->ij',q*np.array([1,-1,-1,-1]),amplitude[4+4*flavor:8+4*flavor]),
                                   -1j*m*amplitude[1+flavor],atol=2e-15)
    blocks=amplitude.reshape(20,2,2,3,2,2,3)
    assert np.linalg.norm(blocks[1,:,0,:, :,1,:])>0  # off-diagonal isovector flavor
    for color1 in range(3):
        for color2 in range(3):
            if color1!=color2: np.testing.assert_array_equal(blocks[:,:,:,color1,:,:,color2],0)


@pytest.mark.parametrize('s',[.36,.3601,.8])
def test_threshold_support_and_scalar_versus_pion_power(s):
    m=.3;q=[np.sqrt(s),0,0,0];a=analytic_spectral(q,m)
    if s<=4*m*m:
        np.testing.assert_array_equal(a,0)
        np.testing.assert_array_equal(phase_space_spectral(q,m),0)
    else:
        beta=np.sqrt(1-4*m*m/s)
        assert a[0,0].real/a[1,1].real==pytest.approx(beta*beta)
        assert a[1,1].real==pytest.approx(3*s*beta/(4*np.pi))


@pytest.mark.parametrize('index',[0,1])
def test_zero_momentum_scalar_mass_insertion_includes_threshold_measure(index):
    m=.28;s=.8;h=2e-6;q=[np.sqrt(s),0,0,0]
    plus=phase_space_spectral(q,m+h,geometry='light_front',order=4)[index,index].real
    minus=phase_space_spectral(q,m-h,geometry='light_front',order=4)[index,index].real
    derivative=(plus-minus)/(2*h);beta=np.sqrt(1-4*m*m/s)
    expected=-9*m*beta/np.pi if index==0 else -3*m/(np.pi*beta)
    assert derivative==pytest.approx(expected,rel=2e-9)
    # Freezing phase-space beta omits part of the same mass-source response.
    a=analytic_spectral(q,m)[index,index].real
    frozen=derivative-a*(-4*m/(s*beta))/beta
    assert abs(frozen-expected)>.1


@pytest.mark.parametrize('z',[-1.,-.1,.1,1e-6])
@pytest.mark.parametrize('channel',['scalar','pion'])
def test_independent_Feynman_and_dispersive_nonlocal_quantum_parts(z,channel):
    m=.28
    a=feynman_nonlocal(m,z,channel);b=dispersive_nonlocal(m,z,channel)
    assert a==pytest.approx(b,rel=1e-11,abs=1e-20)
    assert a<0
    low=dispersive_nonlocal(m,z,channel,beta_interval=(0.,.63))
    high=dispersive_nonlocal(m,z,channel,beta_interval=(.63,1.))
    assert low+high==pytest.approx(b,rel=2e-13)
    # Twice-subtracted quantities alone do not pick the unknown local polynomial.
    assert feynman_nonlocal(m,0.,channel)==0


@pytest.mark.parametrize('s',[.5,1.])
@pytest.mark.parametrize('channel',['scalar','pion'])
def test_independent_complex_loop_discontinuity(s,channel):
    m=.28;index=0 if channel=='scalar' else 1
    cut=analytic_spectral([np.sqrt(s),0,0,0],m)[index,index].real
    value=feynman_upper_boundary(m,s,channel,imaginary_part=1e-8)
    assert -value.imag==pytest.approx(cut,rel=2e-7)


def test_common_continuum_interference_and_unfitted_mass_preserving_direction():
    model=benchmark_reference();s=1.
    a=collective_cut(model,s,'scalar');g=model.scalar.couplings
    dark=np.array([-g[1],g[0]])/np.linalg.norm(g)
    np.testing.assert_allclose(a@dark,0,atol=1e-15)
    assert dark@np.diag(np.diag(a))@dark>.1
    assert np.linalg.matrix_rank(a,tol=1e-13)==1
    widths=tree_width_diagnostics(model)
    assert widths[1]['projected_Gamma_over_M']==pytest.approx(.809723059255486,rel=1e-12)
    assert widths[0]['projected_Gamma_over_M']<.003
    assert all(not r['channel_open'] for r in widths[2:])
    # Preserve the protected light-pion coupling (and aligned tree vacuum mass).
    _,pion_modes=np.linalg.eigh(model.pion.hessian);v=pion_modes[:,0]
    normal=np.array([-v[1],v[0]])
    _,scalar_modes=np.linalg.eigh(model.scalar.hessian);heavy=scalar_modes[:,1]
    changed=g-(heavy@g)/(heavy@normal)*normal
    assert changed@v==pytest.approx(g@v,rel=1e-14)
    alternative=QuarkCollectiveReference(CollectiveChannel(model.scalar.hessian,changed,model.scalar.c),
        CollectiveChannel(model.pion.hessian,changed,model.pion.c),model.mass)
    assert tree_width_diagnostics(alternative)[1]['projected_Gamma_over_M']<1e-25
    assert np.linalg.norm(changed)>np.linalg.norm(g)


def test_invalid_domains_do_not_silently_invent_a_prescription():
    with pytest.raises(ValueError): analytic_spectral([0.,1.,0.,0.],.3)
    with pytest.raises(ValueError): feynman_nonlocal(.3,.5,'pion')
    with pytest.raises(ValueError): dispersive_nonlocal(.3,-.5,'pion',beta_interval=(.7,.2))
    with pytest.raises(ValueError): phase_space_spectral([1.,0,0,0],.3,geometry='invented')


def test_replay_does_not_promote_a_quark_cut_to_physical_hadrons():
    from scripts.evaluate_quantum_pair_threshold import evaluate
    data=evaluate();json.dumps(data,allow_nan=False)
    assert not data['claims']['physical_hadron_widths']
    assert not data['claims']['finite_local_matching_selected']
    assert data['checks']['max_LF_spectral_relative']<1e-12
