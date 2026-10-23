import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.linalg import expm

from deuteron_wigner.quantum.nonstationary_lf_source import (
    SourceFrame,canonical_mixed_map,mapped_jet,inverse_jet,evolve_weak_wave,canonical_wave_jet,
    plus,minus,ENERGY_DIRECTION,tensor,anticommutator,commutator,
)


def cart(P,w,x,y): return np.array([(P+w)/np.sqrt(2),x,y,(P-w)/np.sqrt(2)])


def parameters():
    return (np.array([[.12,.02],[.02,.25]]),np.array([[.4,.05],[.05,.2]]),
        np.array([[-.1,.03],[.03,.06]]),np.array([[.7,.1],[.1,.4]]))


@pytest.fixture(scope='module')
def frame():
    return SourceFrame(cart(.8,0,.1,.2),cart(.13,.07,.05,-.03),cart(.07,-.04,-.02,.04),
        np.array([.2,.4,-.1,.3]),np.array([.1,-.2,.35,-.15]))


@pytest.fixture(scope='module')
def expanded(frame): return frame.weak_extension()


@pytest.mark.parametrize('omega',[-.27,-.15,.31])
def test_actual_action_nonstationary_polynomial_and_complete_map(frame,omega):
    args=parameters();actual=frame.inverse_jets(*args,omega,actual=True);poly=frame.inverse_jets(*args,omega)
    for a,b in zip(actual,poly): np.testing.assert_allclose(a,b,atol=2e-13)
    result=mapped_jet(frame.mapping(*args),frame.frequency(2,omega))
    for a,b in zip(result,inverse_jet(actual)): np.testing.assert_allclose(a,b,atol=3e-11)
    # Source plus/minus momentum transfers and all reverse routes survive.
    np.testing.assert_allclose(minus(frame.momenta(omega))-minus(frame.momenta(0)),omega,atol=1e-15)
    assert min(plus(frame.momenta(omega)))>0


@pytest.mark.parametrize('omission',['connection','endpoints','direct','one_order','right_adjoint'])
def test_required_canonical_source_partners_have_visible_negative_controls(frame,omission):
    m=frame.mapping(*parameters());p=frame.frequency(2,-.15)
    good=inverse_jet(frame.inverse_jets(*parameters(),-.15))[3]
    assert torch.linalg.norm(good-mapped_jet(m,p,omit=omission)[3])>1e-5


def test_second_source_map_is_actual_mixed_inverse_derivative(frame):
    jets=frame.inverse_jets(*parameters(),-.15);rng=np.random.default_rng(28)
    left=tensor(rng.normal(size=8)+1j*rng.normal(size=8));right=tensor(rng.normal(size=8)+1j*rng.normal(size=8))
    def functional(x):
        k,ka,kb,kab=jets;value=left.conj()@torch.linalg.solve(k+x[0]*ka+x[1]*kb+x[0]*x[1]*kab,right)
        return torch.stack([value.real,value.imag])
    z=torch.zeros(2,dtype=torch.float64)
    actual=torch.autograd.functional.jacobian(lambda x:torch.autograd.functional.jacobian(functional,x,create_graph=True),z)[:,0,1]
    result=mapped_jet(frame.mapping(*parameters()),frame.frequency(2,-.15))[3]
    expected=left.conj()@result@right
    np.testing.assert_allclose(actual,[expected.real,expected.imag],atol=3e-13)


def test_complex_noncommuting_frequency_map_and_frame_connection():
    rng=np.random.default_rng(27);p=np.diag([-.2,.13,.31,.48]);h=np.diag([.12,.3,.7,1.1]);x=[]
    for _ in range(7):
        a=rng.normal(0,.2,(4,4))+1j*rng.normal(0,.2,(4,4));x.append((a+a.conj().T)/2)
    ga,va,gb,vb,gab,vab,dab=x;m=canonical_mixed_map(h,p,*x)
    for H in m['H']: assert torch.allclose(H,H.conj().T,rtol=0,atol=2e-14)
    k=h-p;ka=va-anticommutator(ga,p)/2;kb=vb-anticommutator(gb,p)/2
    kab=vab-anticommutator(gab,p)/2-p@dab@p
    for a,b in zip(inverse_jet((k,ka,kb,kab)),mapped_jet(m,p)): np.testing.assert_allclose(a,b,atol=4e-10)
    assert torch.linalg.norm(m['connection'])>.01
    r=torch.linalg.inv(tensor(k))
    defect=mapped_jet(m,p,omit='connection')[3]-inverse_jet((k,ka,kb,kab))[3]
    np.testing.assert_allclose(defect,r@m['connection']@r,atol=3e-13)
    assert torch.linalg.norm(defect)>1000*torch.linalg.norm(mapped_jet(m,p)[3]-inverse_jet((k,ka,kb,kab))[3])
    # Frequency differences, not a common frequency offset, generate it.
    shifted=canonical_mixed_map(h,p+2.3*np.eye(4),*x)
    np.testing.assert_allclose(shifted['connection'],m['connection'],atol=2e-15)


@pytest.mark.parametrize('polarization',range(4))
def test_all_four_current_components_and_four_source_partner_components(frame,polarization):
    for other in range(4):
        f=SourceFrame(frame.p0,frame.qa,frame.qb,np.eye(4)[polarization],np.eye(4)[other])
        actual=inverse_jet(f.inverse_jets(*parameters(),-.15,actual=True))
        mapped=mapped_jet(f.mapping(*parameters()),f.frequency(2,-.15))
        np.testing.assert_allclose(mapped[3],actual[3],atol=3e-12)


@pytest.mark.parametrize('which',[0,1])
def test_independent_offshell_Ward_difference_not_imposed_to_build_current(frame,which):
    eps=[frame.epsa,frame.epsb];eps[which]=[frame.qa,frame.qb][which]
    f=SourceFrame(frame.p0,frame.qa,frame.qb,*eps)
    result=[f.unnormalize_response(x).numpy() for x in mapped_jet(f.mapping(*parameters()),f.frequency(2,-.15))]
    generator=np.zeros((4,4))
    pairs=[(1,0),(3,2)] if which==0 else [(2,0),(3,1)]
    for i,j in pairs: generator[i,j]=1;generator[j,i]=-1
    t=np.kron(generator,np.eye(2));partner=result[2 if which==0 else 1]
    np.testing.assert_allclose(result[3],commutator(partner,t),atol=3e-12)
    assert np.linalg.norm(result[3])>1e-2  # off-shell endpoint difference, not zero


@pytest.mark.parametrize('beta',[[.17,-.12,.08],[-.13,.09,.19]])
def test_all_external_kinetic_factors_restore_scalar_Lorentz_response(frame,beta):
    a=np.zeros((4,4));a[0,1:]=a[1:,0]=beta;boost=expm(a);p0=frame.momenta(-.15)[0]
    moved=SourceFrame(boost@p0,boost@frame.qa,boost@frame.qb,boost@frame.epsa,boost@frame.epsb)
    original=mapped_jet(frame.mapping(*parameters()),frame.frequency(2,-.15))[3]
    transformed=mapped_jet(moved.mapping(*parameters()),moved.frequency(2,0))[3]
    np.testing.assert_allclose(frame.unnormalize_response(original),moved.unnormalize_response(transformed),atol=3e-12)
    assert torch.linalg.norm(original-transformed)>1e-2


def test_node_rotations_and_mixed_interaction_source_gradients(frame):
    args=parameters();r=expm(np.array([[0.,.31],[-.31,0.]]));u=np.kron(np.eye(4),r)
    original=mapped_jet(frame.mapping(*args),frame.frequency(2,-.15))[3]
    moved=mapped_jet(frame.mapping(*(r@x@r.T for x in args)),frame.frequency(2,-.15))[3]
    np.testing.assert_allclose(moved,u@original.numpy()@u.T,atol=3e-12)
    def value(theta):
        mass,c,d,e=[tensor(x) for x in args];v=tensor([[1.,.2],[.2,0.]])
        m=frame.mapping(mass+theta[0]*v,c+theta[1]*v,d,e+theta[2]*v)
        return mapped_jet(m,frame.frequency(2,-.15))[3][6,0].real
    z=torch.zeros(3,dtype=torch.float64);gradient=torch.autograd.functional.jacobian(value,z)
    assert min(abs(gradient))>1e-5
    for i in range(3):
        delta=torch.eye(3,dtype=torch.float64)[i]*1e-4
        finite=(-value(2*delta)+8*value(delta)-8*value(-delta)+value(-2*delta))/(12e-4)
        assert gradient[i]==pytest.approx(float(finite),abs=2e-9)


def test_actual_LF_weak_wave_response_and_Noether_norm(expanded):
    frame=expanded;rng=np.random.default_rng(39);seed=rng.normal(size=8)+1j*rng.normal(size=8);seed/=np.linalg.norm(seed)
    u=np.zeros(2*len(frame.routes),complex);u[:8]=seed
    result=evolve_weak_wave(frame.matrices(*parameters()),11.,u)
    np.testing.assert_allclose(result['original'],result['mapped_original'],atol=3e-9)
    np.testing.assert_allclose(result['canonical_ab_norm'],0,atol=3e-9)
    np.testing.assert_allclose(result['original_ab_Noether_norm'],0,atol=3e-9)
    matrices=frame.matrices(*parameters());h,p,*_,dab=[x.numpy() for x in matrices]
    predicted=[]
    for t,phi in zip(result['times'],result['original']):
        U=expm(-1j*t*p);D=U@dab@U.conj().T
        predicted.append(-np.vdot(phi[0],(D@h+h@D)@phi[0]).real)
    np.testing.assert_allclose(result['omitting_D_velocity_norm'],predicted,atol=3e-9)
    assert max(np.abs(predicted))>1000*max(abs(result['original_ab_Noether_norm']))
    assert np.linalg.norm(result['canonical'][-1,3])>.01
    exact=canonical_wave_jet(frame.mapping(*parameters()),frame.frequency(2,0),11.,u)
    np.testing.assert_allclose(exact,result['canonical'][-1],atol=3e-9)


def test_shared_wave_and_source_parameters_have_exact_Torch_derivatives(expanded):
    frame=expanded;rng=np.random.default_rng(39);seed=rng.normal(size=8)+1j*rng.normal(size=8);seed/=np.linalg.norm(seed)
    u=np.zeros(32,complex);u[:8]=seed
    def response(theta):
        h,c,d,e=[tensor(x) for x in parameters()];direction=tensor([[1.,.2],[.2,0.]])
        m=frame.mapping(h+theta[0]*direction,c+theta[1]*direction,d,e+theta[2]*direction)
        state=canonical_wave_jet(m,frame.frequency(2,0),11.,u)
        # Full complex final-mode amplitude derivative, not an observable fit.
        return state[3,6].real+.3*state[3,6].imag
    z=torch.zeros(3,dtype=torch.float64);gradient=torch.autograd.functional.jacobian(response,z)
    assert min(abs(gradient))>1e-6
    for i in range(3):
        delta=torch.eye(3,dtype=torch.float64)[i]*1e-4
        finite=(-response(2*delta)+8*response(delta)-8*response(-delta)+response(-2*delta))/(12e-4)
        assert gradient[i]==pytest.approx(float(finite),abs=2e-8)


def test_complete_weak_route_carrier_and_projected_state_negative(frame,expanded):
    assert len(expanded.routes)==16
    for i,j in frame.routes:
        for di in [-1,0,1]:
            for dj in [-1,0,1]: assert (i+di,j+dj) in expanded.routes
    rng=np.random.default_rng(39);u=rng.normal(size=8)+1j*rng.normal(size=8);u/=np.linalg.norm(u)
    full=np.zeros(32,complex);full[:8]=u
    small=canonical_wave_jet(frame.mapping(*parameters()),frame.frequency(2,0),11.,u).numpy()
    large=canonical_wave_jet(expanded.mapping(*parameters()),expanded.frequency(2,0),11.,full).numpy()
    np.testing.assert_allclose(small,large[:,:8],atol=3e-12)
    # New first-order routes carry real information despite exact norm in
    # the projected submodel. Retaining only the four old corners loses it.
    assert np.linalg.norm(large[1:,8:])>.1
    # All old-corner amplitudes agree, despite the missing full-wave content.
    spatial=expanded.momenta(0)[:,1:3]
    grid=np.column_stack([plus(expanded.momenta(0)),spatial])/.01
    np.testing.assert_allclose(grid,np.round(grid),atol=1e-12)


def test_zero_plus_and_undeclared_map_domains_are_not_filled_in(frame):
    with pytest.raises(ValueError): SourceFrame(cart(-.1,0,.1,.2),frame.qa,frame.qb,frame.epsa,frame.epsb)
    with pytest.raises(ValueError): SourceFrame(frame.p0,frame.qa,frame.qa,frame.epsa,frame.epsb)
    with pytest.raises(ValueError): mapped_jet(frame.mapping(*parameters()),frame.frequency(2,-.15),omit='anything')


def test_saved_complete_wave_routes_and_full_Fock_scope():
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'docs/next_level/nonstationary_lf_source_results.json').read_text())
    wave=record['weak_wave_response']
    assert len(wave['all_weak_routes'])==16
    assert len(wave['final_mixed_canonical_amplitudes_GeV_minus2'])==32
    assert wave['previously_unrepresented_wave_amplitude_norms_by_order'][3]>3
    assert max(wave['all_retained_four_corner_amplitude_errors_by_order'])<1e-12
    assert 'Not a complete sourced Fock state' in record['scope']
    assert not record['reassessment']['zero_plus_or_pair_creation_crossings_resolved']
    assert not record['reassessment']['physical_nucleon_or_nucleus_generated']
    assert not record['reassessment']['known_loops_requantized']
    assert not record['reassessment']['production_model_selected']
