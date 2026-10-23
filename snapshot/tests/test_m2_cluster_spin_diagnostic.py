import numpy as np
import pytest
from scipy.optimize import brentq

from deuteron_wigner.quantum.m2_cluster_spin_diagnostic import (
    finite_elastic_pole,elastic_spin_pair,elastic_orientation_analysis,
    rotate_current_blocks,free_spin_fiber,free_current_parts,
    complete_elastic_spin_amplitudes)
from deuteron_wigner.quantum.m2_orientation_current_diagnostic import orientation_current_structures
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION


@pytest.mark.parametrize('angle',(.37,-.81,np.pi))
def test_kinematic_rotation_phase_is_checked_against_external_spinors(angle):
    d=np.array([.12,.16]);c,s=np.cos(angle),np.sin(angle)
    r=np.array([[c,-s],[s,c]])
    original=orientation_current_structures(d,mass_GeV=1.2,total_pplus_GeV=1.3)
    rotated=orientation_current_structures(r@d,mass_GeV=1.2,total_pplus_GeV=1.3)
    for f,hf in enumerate((1,-1)):
        for i,hi in enumerate((1,-1)):
            predicted=rotate_current_blocks(original[:,:,f:f+1,i:i+1],angle,
                                            jz_out_twice=hf,jz_in_twice=hi)
            np.testing.assert_allclose(predicted,rotated[:,:,f:f+1,i:i+1],atol=2e-15,rtol=0)


@pytest.mark.parametrize('seed',range(4))
def test_covariance_residual_is_not_a_change_of_spin_coordinates(seed):
    rng=np.random.default_rng(seed);d=(.12,.16)
    b=orientation_current_structures(d,mass_GeV=1.2,total_pplus_GeV=1.3)
    coeff=np.array([.8,.3,-.9,.2,.4])
    current=np.einsum('k,kmab->mab',coeff,b)
    current[1,0,1]+=.02+.01j
    q,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    transformed=np.einsum('ab,mbc,cd->mad',q.conj().T,current,q)
    a=elastic_orientation_analysis(current,d,energy_GeV2=1.44,total_pplus_GeV=1.3)
    z=elastic_orientation_analysis(transformed,d,energy_GeV2=1.44,total_pplus_GeV=1.3,helicity_basis=q)
    for n in ('two','five'):
        np.testing.assert_allclose(a[n]['coefficients'],z[n]['coefficients'],atol=1e-13,rtol=0)
        assert a[n]['residual_norm']==pytest.approx(z[n]['residual_norm'],abs=2e-15)
    assert a['five']['residual_norm']>1e-3
    assert not a['physical'] and not a['covariance_repaired']


def test_order_family_decomposition_is_linear_and_keeps_outside_residuals():
    rng=np.random.default_rng(82)
    pieces=rng.normal(size=(3,4,2,2))+1j*rng.normal(size=(3,4,2,2))
    parts=[elastic_orientation_analysis(p,(.12,.16),energy_GeV2=1.4,total_pplus_GeV=1.3) for p in pieces]
    full=elastic_orientation_analysis(sum(pieces),(.12,.16),energy_GeV2=1.4,total_pplus_GeV=1.3)
    for n in ('two','five'):
        np.testing.assert_allclose(sum(p[n]['coefficients'] for p in parts),full[n]['coefficients'],atol=2e-13)
        np.testing.assert_allclose(sum(p[n]['residual'] for p in parts),full[n]['residual'],atol=2e-14)
    assert full['five']['residual_norm']>1.


@pytest.mark.parametrize('delta',((.12,.16),(.2,0.),(-.1,.17)))
def test_spin_completion_against_independent_external_dirac_structures(delta):
    basis=orientation_current_structures(delta,mass_GeV=1.2,total_pplus_GeV=1.3)
    predicted=complete_elastic_spin_amplitudes(basis[...,0,0],basis[...,1,0],delta)
    np.testing.assert_allclose(predicted,basis,atol=2e-15,rtol=0)


def test_fixed_frame_reference_error_propagates_without_changing_current():
    rng=np.random.default_rng(241)
    basis=orientation_current_structures((.12,.16),mass_GeV=1.2,total_pplus_GeV=1.3)
    b=np.column_stack([v.ravel() for v in basis]);error=rng.normal(size=16)+1j*rng.normal(size=16)
    for n in (2,5):
        inverse=np.linalg.pinv(b[:,:n]);projector=b[:,:n]@inverse
        assert np.linalg.norm(inverse@error)<=np.linalg.norm(inverse,2)*np.linalg.norm(error)+1e-13
        assert np.linalg.norm((np.eye(16)-projector)@error)<=np.linalg.norm(error)+1e-13


@pytest.mark.parametrize('seed',range(3))
def test_numerator_norm_and_state_attribution_is_exact_for_complex_inputs(seed):
    from scripts.analyze_m2_cluster_spin_resolution import block_change
    rng=np.random.default_rng(seed)
    def matrix(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    def packet():
        return dict(family_coefficients=matrix((2,5,4,3,3)),
            out_W=matrix((3,3)),in_W=matrix((3,3)),
            out_state=matrix((3,)),in_state=matrix((3,)))
    a,b=packet(),packet();parts=block_change(a,b)
    assert parts.shape==(3,2,5,4)
    assert min(np.linalg.norm(p) for p in parts)>1.


@pytest.mark.parametrize('kind',('shape','nan','zero_mass','zero_transfer','nonunitary'))
def test_invalid_elastic_diagnostic_inputs_are_rejected(kind):
    cur=np.ones((4,2,2),complex);d=(.12,.16);e=1.4;u=None
    if kind=='shape':cur=cur[:3]
    if kind=='nan':cur[0,0,0]=np.nan
    if kind=='zero_mass':e=0.
    if kind=='zero_transfer':d=(0.,0.)
    if kind=='nonunitary':u=2*np.eye(2)
    with pytest.raises(ValueError):
        elastic_orientation_analysis(cur,d,energy_GeV2=e,total_pplus_GeV=1.3,helicity_basis=u)


@pytest.fixture(scope='module')
def pair():
    from scripts.benchmark_m2_cluster_spin import models
    return elastic_spin_pair(*models('PROTON'),bracket_GeV2=(1.3,1.5),intrinsic_parity=1)


def test_elastic_pole_independent_generalized_root_and_derivative(pair):
    from scipy.linalg import eigvalsh
    model=pair['snapshots'][0].owner
    # Solve the generalized linearized eigenproblem independently.
    def alternate(e):
        s=model.snapshot(e);n=s.metric
        return eigvalsh(s.effective_hamiltonian+e*(n-np.eye(len(n))),n)[0]-e
    root=brentq(alternate,1.42,1.44,xtol=5e-14)
    assert root==pytest.approx(pair['energy_GeV2'],abs=1e-12)
    e=root;eps=1e-4;s=model.snapshot(e)
    derivative=(model.snapshot(e+eps).effective_hamiltonian-model.snapshot(e-eps).effective_hamiltonian)/(2*eps)
    np.testing.assert_allclose(np.eye(len(s.matrix))-derivative,s.metric,atol=3e-10,rtol=0)
    for ss,u,v in zip(pair['snapshots'],pair['raw_states'],pair['states']):
        np.testing.assert_allclose(ss.effective_hamiltonian@u,e*u,atol=3e-13,rtol=0)
        np.testing.assert_allclose(ss.matrix@v,e*v,atol=3e-13,rtol=0)
        assert np.vdot(u,ss.metric@u)==pytest.approx(1.,abs=3e-13)


@pytest.mark.parametrize('bracket',((1.5,1.3),(1.1,1.2),(1.3,20.),(-1.,2.)))
def test_invalid_or_unbracketed_or_cross_pole_intervals_rejected(pair,bracket):
    with pytest.raises(ValueError):finite_elastic_pole(pair['snapshots'][0].owner,bracket_GeV2=bracket)


def test_parity_maps_both_norms_without_arbitrary_eigensolver_phases(pair):
    f,i=pair['snapshots'][1],pair['snapshots'][0];t=pair['parity_map']
    back=transverse_parity_map(i.owner.pilot,f.owner.pilot)
    np.testing.assert_allclose(back@t,-np.eye(len(t)),atol=4e-15)
    np.testing.assert_allclose(pair['raw_states'][1],1j*t@pair['raw_states'][0],atol=0)
    np.testing.assert_allclose(pair['states'][1],1j*t@pair['states'][0],atol=4e-15)
    assert max(pair['parity_errors'].values())<3e-11


def test_free_carrier_is_exact_h0_not_an_interaction_default(pair):
    p=pair['snapshots'][0].owner.pilot
    f=free_spin_fiber(total_k2=9,cutoff=0,target='PROTON',jz_twice=1,
                      b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.)
    np.testing.assert_array_equal(f.h0,p.h0)
    assert f.coupling==0 and not hasattr(f,'cross_pairs_per_gs2')


def test_free_full_current_is_original_saved_raw_g0(pair):
    from pathlib import Path
    f=free_spin_fiber(total_k2=9,cutoff=0,target='PROTON',jz_twice=1,
                      b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.)
    context=SmoothMinusProjectorContext(mean_total_transverse_GeV=(0.,0.),prescription=SMOOTH_PRESCRIPTION)
    physical,projector=free_current_parts(f,f,(.12,.16),total_pplus_GeV=1.3,
        charge='electromagnetic',quadrature_order=8,minus_context=context)
    path=Path(__file__).resolve().parents[1]/'docs/next_level/m2_gaussian_cluster_four_current_baseline.npz'
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_allclose(physical+projector,data['current_power_0'],atol=1e-14,rtol=0)
