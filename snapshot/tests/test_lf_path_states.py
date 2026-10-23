from math import pi
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad_vec

from deuteron_wigner.quantum.lf_path_states import (
    DressingPath,PathCarrier,baryon_seed,pullback,reference,
)
from deuteron_wigner.quantum.finite_lf_coupling import LFCell,fermion_good_mode_metric
from deuteron_wigner.quantum.canonical_lf_fields import lf_spinor,LAMBDA_PLUS
from deuteron_wigner.quantum.modular_gauge_network import generators,su3,adjoint
from deuteron_wigner.quantum.graded_fock_network import Mode,Ladder,Word,FockRegister


@pytest.fixture(scope='module')
def setup():
    c,p=reference()
    return c,p,{name:c.wave(path) for name,path in p.items()}


def test_normalized_modes_and_complete_order_two_recoil(setup):
    c,p,w=setup
    assert len(c.labels)==32 and len(c.configs)==561
    initial=c.quarks.sum(axis=0)
    for config in c.configs:
        q,k=c.recoil(config)
        c.cell.mode(q,'fermion')
        assert q[0]>0 and np.linalg.norm(q+k+c.quarks[1:].sum(axis=0)-initial)<1e-15
        assert np.linalg.norm(fermion_good_mode_metric(q,c.mass)-np.eye(2))<2e-15
        u=np.column_stack([lf_spinor(*q,c.mass,h) for h in (-1,1)])
        parent=np.column_stack([lf_spinor(*c.quarks[0],c.mass,h) for h in (-1,1)])
        metric=np.sqrt(2)*u.conj().T@LAMBDA_PLUS@parent/(2*np.sqrt(q[0]*c.quarks[0,0]))
        assert np.linalg.norm(metric-np.eye(2))<2e-15
    assert np.linalg.norm(baryon_seed())==pytest.approx(1.)
    for t in generators():
        total=np.kron(t,np.eye(9))+np.kron(np.eye(3),np.kron(t,np.eye(3)))+np.kron(np.eye(9),t)
        assert np.linalg.norm(total@baryon_seed())<2e-16


def test_single_color_tensor_embeds_in_actual_fermion_exterior_space(setup):
    c,paths,w=setup
    # Three distinct spin/flavor labels: u-, d-, u+. Momentum-major order
    # makes exchange signs nontrivial and independent of the tensor axes.
    momenta={tuple(c.recoil(x)[0]) for x in c.configs}|set(map(tuple,c.quarks[1:]))
    labels=sorted((p,f,a) for p in momenta for f in range(3) for a in range(3))
    register=FockRegister([Mode(str(label),'fermion',1) for label in labels])
    lookup={label:i for i,label in enumerate(labels)}
    vac={(0,)*len(labels):1}
    # All 27 tensor coordinates, including zeros of the epsilon seed.
    for config in ((),(0,),(0,0),(0,17),(17,31)):
        quarks=c.quarks.copy();quarks[0]=c.recoil(config)[0]
        images=[]
        for colors in product(range(3),repeat=3):
            modes=[lookup[(tuple(quarks[f]),f,colors[f])] for f in range(3)]
            word=Word(1,tuple(Ladder(i,True) for i in modes))
            result=register.occupation_action((word,),vac)
            assert len(result)==1
            key,amplitude=next(iter(result.items()))
            inversions=sum(modes[i]>modes[j] for i in range(3) for j in range(i+1,3))
            assert amplitude==(-1)**inversions
            images.append(key)
        assert len(set(images))==27


@pytest.mark.parametrize('bad',[(),((0,0),),((0,float('nan'),0),)])
def test_invalid_path_rejected(bad):
    with pytest.raises(ValueError):DressingPath(bad)


def test_recoil_failure_is_explicit_not_missing_sector_zero(setup):
    c,_,_=setup
    p=c.quarks.copy();p[0,0]=.875
    with pytest.raises(ValueError,match='recoil'):PathCarrier(c.cell,[c.momenta[0],c.momenta[2]],p,.27)
    with pytest.raises(ValueError,match='Distinct'):PathCarrier(c.cell,[c.momenta[0]]*2,c.quarks,.27)


def test_path_profile_sign_phase_and_analytic_integral(setup):
    c,paths,w=setup
    start=np.array([.7,-.4,.8]);end=np.array([1.9,.3,-.2]);d=end-start
    for s in (0.,.21,1.):
        r=start+s*d
        expected=[]
        for k,h in zip(c.momenta,c.helicities):
            # eps_T=(-h,-i)/sqrt(2); minus spatial metric contraction.
            expected.append((h*d[1]-1j*d[2])/np.sqrt(2)*
                            np.exp(1j*(k[0]*r[0]-k[1:]@r[1:]))/
                            np.sqrt(2*k[0]*c.cell.volume))
        assert np.linalg.norm(c.segment(start,end,s)-expected)<2e-18
    numeric=quad_vec(lambda s:c.segment(start,end,s),0,1,epsabs=1e-13)[0]
    f,_,_=c.integrals(DressingPath((start,end)))
    assert np.linalg.norm(f-numeric)<2e-17


def test_ordered_quadrature_refinement_and_vacuum_contraction(setup):
    c,paths,w=setup
    for path in paths.values():
        f,i,j=c.integrals(path)
        assert np.linalg.norm(c.wave(path,16)-c.wave(path,32))<2e-16
        assert np.linalg.norm(i+i.T-np.outer(f,f))<2e-17
        assert np.linalg.norm(2*j.real-abs(f)**2)<2e-17
    # Unequal x-minus retains the imaginary ordered vacuum contribution.
    assert abs(sum(c.integrals(paths['LF_detour'])[2]).imag)>1e-5


def test_independent_full_oscillator_Dyson_wave(setup):
    c,paths,w=setup
    reference_wave,n=c.ode_wave(paths['LF_detour'])
    assert n>0
    for order in range(3):
        assert np.linalg.norm(reference_wave[order]-w['LF_detour'][order])<1e-10


def test_same_order_vacuum_is_required_but_two_gluon_wave_invisible_to_norm(setup):
    c,paths,w=setup;v=w['LF_detour'];norm=pullback(v,v)
    assert np.linalg.norm(norm-[1,0,0])<2e-17+5e-16
    omitted=v.copy();omitted[2,0]=0
    assert pullback(omitted,omitted)[2].real>1e-4
    omitted=v.copy();omitted[2,1+len(c.labels):]=0
    assert np.linalg.norm(pullback(omitted,omitted)-norm)==0
    assert np.linalg.norm(omitted-v)>1e-4
    # A Hermitian same-carrier quadratic field probe DOES see that wave.
    f=c.integrals(paths['different_loop'])[0];creators=c.oscillator_creators()
    action=lambda x:c.field_action(f,c.field_action(f,x,creators),creators)
    assert abs(pullback(v,v,action)[2]-pullback(omitted,omitted,action)[2])>1e-8


def test_nonabelian_ordering_is_not_exponential_of_integrated_field(setup):
    c,paths,w=setup;path=paths['LF_detour'];f,_,_=c.integrals(path)
    zero=w['bare'][0];ops=c.oscillator_creators()
    unordered=-.5*c.field_action(f,c.field_action(f,zero,ops),ops)
    assert np.linalg.norm(w['LF_detour'][2]-unordered)>1e-4
    # Subtracting only a scalar vacuum phase cannot repair the two-gluon part.
    assert np.linalg.norm((w['LF_detour'][2]-unordered)[1+len(c.labels):])>1e-4


def test_path_reversal_keeps_operator_adjoint_and_order(setup):
    c,paths,w=setup
    for name,path in paths.items():
        reverse=c.wave(path.reversed());f,_,j=c.integrals(path)
        assert np.linalg.norm(reverse[1]+w[name][1])<1e-16
        assert np.linalg.norm(reverse[2,0]-w[name][2,0].conj())<1e-16
        # U_reverse^(2)+U^(2)=-(integral B)^2 on vacuum.
        ops=c.oscillator_creators()
        square=c.field_action(f,c.field_action(f,w[name][0],ops),ops)
        assert np.linalg.norm(reverse[2]+w[name][2]+square)<2e-16


def test_pure_longitudinal_path_is_trivial_only_in_declared_A_plus_zero_carrier(setup):
    c,_,w=setup
    long=DressingPath(((0,1,2),(7,1,2),(-2,1,2)))
    assert np.linalg.norm(c.wave(long)-w['bare'])==0


def test_common_Gram_operator_and_duplicate_path_are_not_independent_labels(setup):
    c,paths,w=setup;names=list(paths)+['LF_detour']
    e=c.free_energies();bare=e[0]
    g=np.array([[pullback(w[a],w[b]) for b in names] for a in names])
    h=np.array([[pullback(w[a],w[b],lambda x:e[:,None]*x) for b in names] for a in names])
    assert np.linalg.norm(g-g.transpose(1,0,2).conj())<2e-17
    assert np.linalg.norm(h-h.transpose(1,0,2).conj())<2e-17
    for a,name in enumerate(names):
        fa,_,ja=c.integrals(paths[name])
        for b,other in enumerate(names):
            fb,_,jb=c.integrals(paths[other])
            assert abs(g[a,b,2]-(4/3)*(np.vdot(fa,fb)-sum(ja).conjugate()-sum(jb)))<1e-17
            k=c.momenta;x=k[:,0]/c.quarks[0,0]
            delta=(np.sum((k[:,1:]-x[:,None]*c.quarks[0,1:])**2,axis=1)+x*x*c.mass*c.mass)/(
                2*c.quarks[0,0]*x*(1-x))
            assert abs(h[a,b,2]-bare*g[a,b,2]-(4/3)*np.sum(fa.conj()*fb*delta))<1e-17
    assert np.linalg.norm(g[2]-g[4])==0 and np.linalg.norm(h[2]-h[4])==0
    # Direct full polynomial Gram is PSD, but its g^4 terms are NOT a
    # qualified physical fourth-order calculation.
    columns=np.column_stack([sum(.6**n*w[name][n] for n in range(3)).ravel() for name in names])
    gram=columns.conj().T@columns
    assert np.linalg.eigvalsh(gram).min()>-2e-15
    assert np.linalg.svd(columns,compute_uv=False)[-1]<1e-15
    assert np.linalg.svd(columns[:,:4],compute_uv=False)[-1]>1e-4
    for name in paths:
        v=w[name]
        shift=pullback(v,v,lambda x:e[:,None]*x)[2]-bare*pullback(v,v)[2]
        independent=np.sum(abs(v[1])**2*(e-bare)[:,None])
        assert abs(shift-independent)<1e-17
        assert shift.real>=0


def test_shared_free_energy_is_actual_four_momentum_not_static_path_cost(setup):
    c,_,_=setup;e=c.free_energies()
    for i in (0,1,17,42,560):
        config=c.configs[i];p=c.quarks.copy();p[0]=c.recoil(config)[0]
        particles=[(q,c.mass) for q in p]+[(c.momenta[c.labels[j][0]],0) for j in config]
        p0=p3=0
        for q,m in particles:
            minus=(m*m+q[1:]@q[1:])/(2*q[0])
            p0+=(q[0]+minus)/np.sqrt(2);p3+=(q[0]-minus)/np.sqrt(2)
        assert abs((p0-p3)/np.sqrt(2)-e[i])<2e-15


def test_global_color_covariance_of_zero_one_and_two_gluon_amplitudes(setup):
    c,_,w=setup;v=w['LF_detour']
    u=su3(np.array([.2,-.1,.35,.08,-.17,.3,.2,-.4]));r=adjoint(u)
    uc=np.kron(u,np.kron(u,u));rg=np.kron(np.eye(len(c.momenta)),r)
    assert np.linalg.norm(uc@v[0,0]-v[0,0])<5e-16
    one=v[1,1:1+len(c.labels)]
    assert np.linalg.norm(rg@one@uc.T-one)<2e-16
    two=np.zeros((len(c.labels),len(c.labels),27),complex)
    for n,(j,k) in enumerate(c.pairs,1+len(c.labels)):
        two[j,k]=two[k,j]=v[2,n]/(1 if j==k else np.sqrt(2))
    transformed=np.einsum('ij,kl,ac,jlc->ika',rg,rg,uc,two,optimize=True)
    assert np.linalg.norm(transformed-two)<2e-17


def test_LF_kinematical_boost_and_relative_path_shift(setup):
    c,paths,w=setup;path=paths['LF_detour'];f,i,j=c.integrals(path)
    for boost in (.4,2.7):
        scale=np.array([boost,1,1]);length=np.array(c.cell.lengths)/scale
        other=PathCarrier(LFCell(tuple(length)),c.momenta[::2]*scale,c.quarks*scale,c.mass)
        shifted=DressingPath(tuple(np.array(path.vertices)/scale))
        assert np.linalg.norm(other.wave(shifted)-w['LF_detour'])<2e-16
        assert np.linalg.norm(other.free_energies()*boost-c.free_energies())<2e-14
    offset=np.array([.9,-.3,.4])
    shifted=DressingPath(tuple(np.array(path.vertices)+offset))
    fp,ip,jp=c.integrals(shifted)
    phase=np.exp(1j*(c.momenta*np.array([1,-1,-1]))@offset)
    assert np.linalg.norm(fp-phase*f)<2e-17
    assert np.linalg.norm(ip-phase[:,None]*phase[None,:]*i)<2e-17
    assert np.linalg.norm(jp-j)<2e-17


def test_concatenation_retains_cross_path_vacuum_and_two_gluon_terms(setup):
    c,p,w=setup;a=p['transverse_loop'];b=p['LF_detour']
    combined=c.wave(DressingPath(a.vertices+b.vertices[1:]))
    f=c.integrals(b)[0];ops=c.oscillator_creators()
    assert np.linalg.norm(combined[1]-w['transverse_loop'][1]-w['LF_detour'][1])<2e-16
    cross=1j*c.field_action(f,w['transverse_loop'][1],ops)
    assert np.linalg.norm(combined[2]-w['transverse_loop'][2]-w['LF_detour'][2]-cross)<2e-16
    assert np.linalg.norm(cross)>1e-4


def test_mode_enlargement_keeps_resolved_creation_waves_but_changes_vacuum(setup):
    c,p,w=setup
    extended=PathCarrier(c.cell,[*c.momenta[::2],(.25,.2,.2)],c.quarks,c.mass)
    a=w['LF_detour'];b=extended.wave(p['LF_detour'])
    assert np.linalg.norm(b[1,1:33]-a[1,1:33])<16*np.finfo(float).eps*np.linalg.norm(a[1])
    lookup={pair:n for n,pair in enumerate(extended.configs)}
    old_two=np.array([b[2,lookup[pair]] for pair in c.pairs])
    assert np.linalg.norm(old_two-a[2,33:])<16*np.finfo(float).eps*np.linalg.norm(a[2,33:])
    assert np.linalg.norm(b[2,0]-a[2,0])>1e-5
    assert np.linalg.norm(pullback(b,b)-[1,0,0])<6e-16
    added=np.linalg.norm(b[1])**2-np.linalg.norm(a[1])**2
    assert added>0
    assert abs(2*np.vdot(c.seed,b[2,0]-a[2,0]).real+added)<1e-17
