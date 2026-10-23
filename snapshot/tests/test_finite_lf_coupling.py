from itertools import product
from math import pi

import numpy as np
import pytest
import torch

from deuteron_wigner.quantum.finite_lf_coupling import (
    LFCell, PairReference, collective_pair_leaf, direct_time_wave,
    fermion_good_mode_metric, first_order_wave, pair_routes, pulse_integral,
    reference_inputs, spinors,
)
from deuteron_wigner.quantum.canonical_lf_network import cubic_vertex_tensor
from deuteron_wigner.quantum.canonical_lf_fields import (
    GAMMA, lf_spinor, polarization, polarization_cartesian, slash,
)
from deuteron_wigner.quantum.modular_gauge_network import generators
from deuteron_wigner.quantum.graded_fock_network import Mode,Ladder,Word,FockRegister


@pytest.mark.parametrize('bad',[(1,2),(-1,2,3),(1,np.inf,3)])
def test_cell_invalid(bad):
    with pytest.raises(ValueError): LFCell(bad)


def test_boundary_grid_and_exact_mode_measure():
    c,*_=reference_inputs()
    assert np.prod(c.spacing)==pytest.approx((2*pi)**3/c.volume)
    with pytest.raises(ValueError): c.mode((.25,0,0),'fermion')
    with pytest.raises(ValueError): c.mode((.125,0,0),'boson')
    with pytest.raises(ValueError): c.mode((0,0,0),'boson')
    assert c.phase_integral((0,0,0))==pytest.approx(c.volume)
    assert abs(c.phase_integral(c.spacing*np.array([1,-2,3])))<1e-30
    p=c.mode((.375,.2,-.2),'fermion')
    np.testing.assert_allclose(fermion_good_mode_metric(p,.27),np.eye(2),atol=4e-16)
    # Three normalized continuum quadrature weights times the spatial delta.
    dp=np.prod(c.spacing)
    assert dp**1.5/(2*pi)**1.5/dp==pytest.approx(1/np.sqrt(c.volume))


@pytest.mark.parametrize('channel', ['q_to_qg','qbar_to_qbarg','g_to_pair','g_to_gg'])
def test_all_canonical_cubic_cell_factors(channel):
    c,*_=reference_inputs()
    if channel in ('q_to_qg','qbar_to_qbarg'):
        a=np.array([.125,.2,0]);b=np.array([.25,-.2,.2])
        stats=('fermion','boson','fermion')
    elif channel=='g_to_pair':
        a=np.array([.125,.2,0]);b=np.array([.375,-.2,.2])
        stats=('fermion','fermion','boson')
    else:
        a=np.array([.25,.2,0]);b=np.array([.25,-.2,.2])
        stats=('boson','boson','boson')
    p=a+b;g=.31
    leaf=cubic_vertex_tensor(channel,a,b,p,.27)
    numerator=leaf*np.sqrt(8*a[0]*b[0]*p[0])
    factors=np.prod([c.field_factor(x,s) for x,s in zip((a,b,p),stats)])
    direct=g*numerator*factors*c.phase_integral(a+b-p)
    np.testing.assert_allclose(c.cubic(channel,a,b,p,.27,g),direct,atol=2e-18)
    # Scaling all GeV momenta and inverse lengths gives the proper P-minus unit.
    scale=1.7;scaled=LFCell(tuple(np.array(c.lengths)/scale))
    np.testing.assert_allclose(scaled.cubic(channel,scale*a,scale*b,scale*p,scale*.27,g),
                               scale*direct,atol=8e-18)
    # Actual kinematic longitudinal boost (m and pT do not change).
    boost=np.array([scale,1,1]);bc=LFCell(tuple(np.array(c.lengths)/boost))
    boosted=bc.cubic(channel,boost*a,boost*b,boost*p,.27,g)
    np.testing.assert_allclose(boosted,direct/scale,atol=8e-18)
    # delta M^2=2 P+ delta P- is invariant, not P+ delta P-.
    np.testing.assert_allclose(2*(scale*p[0])*boosted,2*p[0]*direct,atol=8e-18)


def test_full_spatial_wavepacket_integral_against_mode_sum():
    c,*_=reference_inputs();m=.27;g=.31
    qa=[np.array([p,.2,0]) for p in (.125,.375)]
    gg=[np.array([p,-.2,.2]) for p in (.25,.50)]
    qp=[np.array([p,0,.2]) for p in (.375,.625,.875)]
    aa=np.array([1,.4j]);aa/=np.linalg.norm(aa)
    bb=np.array([.2j,1]);bb/=np.linalg.norm(bb)
    cc=np.array([.3,1j,.2-.4j]);cc/=np.linalg.norm(cc)
    points=np.array(list(product(*[(np.arange(8)+.5)*L/8-L/2 for L in c.lengths])))
    def phase(p): return np.exp(-1j*(points[:,0]*p[0]-points[:,1:]@p[1:]))
    left=sum(z*c.field_factor(p,'fermion')*phase(p)[:,None]*lf_spinor(*p,m,-1)
             for z,p in zip(aa,qa))
    parent=sum(z*c.field_factor(p,'fermion')*phase(p)[:,None]*lf_spinor(*p,m,1)
               for z,p in zip(cc,qp))
    gluon=sum(z*c.field_factor(p,'boson')*phase(p)[:,None]*polarization_cartesian(polarization(*p,1))
              for z,p in zip(bb,gg))
    density=np.array([-g*u.conj()@GAMMA[0]@slash(e.conj())@v for u,e,v in zip(left,gluon,parent)])
    # Fix a nonzero t^a color entry, independently of the field contraction.
    t=generators()[0,0,1]
    direct=c.volume*np.mean(density)*t
    assembled=0j
    for i,a in enumerate(qa):
        for j,b in enumerate(gg):
            for k,p in enumerate(qp):
                if np.allclose(a+b,p):
                    assembled+=aa[i].conjugate()*bb[j].conjugate()*cc[k]*c.cubic('q_to_qg',a,b,p,m,g)[0,8,4]
    assert abs(direct-assembled)<2e-17
    assert abs(direct)>1e-5


def test_complete_pair_routes_are_distinguishable_not_halved():
    c,*_=reference_inputs();p=(.5,0,0)
    routes=pair_routes(c,p,1)
    assert len(routes)==18
    keys={tuple(a) for a,b in routes}
    for a,b in routes:
        np.testing.assert_allclose(a+b,p,atol=1e-16)
        assert tuple(b) in keys
    assert len(pair_routes(c,(.5,.2,0),1))==12
    with pytest.raises(ValueError): pair_routes(c,p,True)


def test_collective_resolved_color_and_isospin_intertwiners():
    a=np.array([.125,.2,-.2]);b=np.array([.375,-.2,.2]);p=a+b
    v=collective_pair_leaf(a,b,p,.27)
    for t in generators():
        t=np.kron(np.eye(4),t)
        defect=np.einsum('ij,jkA->ikA',t,v)-np.einsum('ijA,jk->ikA',v,t)
        assert np.linalg.norm(defect)<3e-15
    tau=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    for tb in tau/2:
        t=np.kron(np.kron(np.eye(2),tb),np.eye(3))
        j=np.zeros((4,4),complex)
        j[1:,1:]=np.array([[np.trace(tc@(tb@ta-ta@tb))/2 for ta in tau] for tc in tau])
        defect=(np.einsum('ij,jkA->ikA',t,v)-np.einsum('ijA,jk->ikA',v,t)
                -np.einsum('ijB,BA->ijA',v,j))
        assert np.linalg.norm(defect)<3e-15


def test_collective_complete_graded_pair_wave_and_spectator_sign():
    c,m,g,seed=reference_inputs();a=np.array([.125,.2,0]);b=np.array([.375,-.2,0])
    leaf=collective_pair_leaf(a,b,a+b,.27)/np.sqrt(c.volume)
    # One extra occupied fermion BETWEEN q and anti modes exposes JW signs.
    modes=[Mode('q'+str(i),'fermion',1) for i in range(12)]
    modes+=[Mode('spectator','fermion',1)]
    modes+=[Mode('anti'+str(i),'fermion',1) for i in range(12)]
    modes+=[Mode('phi'+str(i),'boson',2) for i in range(8)]
    reg=FockRegister(modes);words=[];initial={}
    for n,A in product(range(2),range(4)):
        s=[0]*33;s[12]=1;s[25+4*n+A]=1;initial[tuple(s)]=seed[n,A]
        for i,j in product(range(12),repeat=2):
            z=g[n]*leaf[i,j,A]
            if z!=0:
                words.append(Word(z,(Ladder(i,True),Ladder(13+j,True),Ladder(25+4*n+A,False))))
    occ=reg.occupation_action(words,initial);tens=reg.tensor_action(words,initial)
    assert occ.keys()==tens.keys()
    np.testing.assert_allclose(list(occ.values()),[tens[k] for k in occ],atol=1e-18)
    direct=-np.einsum('ijA,n,nA->ij',leaf,g,seed)
    actual=np.zeros((12,12),complex)
    for s,z in occ.items():
        assert sum(s)==3 and s[12]==1
        actual[np.flatnonzero(s[:12])[0],np.flatnonzero(s[13:25])[0]]=z
    np.testing.assert_allclose(actual,direct,atol=2e-18)
    assert np.linalg.norm(actual+direct)>1e-3  # ungraded sign is wrong
    # Adjoint and Bose factors on a single nonzero monomial.
    w=words[0];s=[0]*33;s[w.factors[-1].mode]=2
    out=reg.occupation_action([w],{tuple(s):1})
    assert np.linalg.norm(list(out.values()))==pytest.approx(np.sqrt(2)*abs(w.coefficient))
    back=reg.occupation_action([w.adjoint()],out)
    assert back[tuple(s)]==pytest.approx(2*abs(w.coefficient)**2)


@pytest.mark.parametrize('shell',[0,1,3])
def test_first_order_full_complex_wave_against_independent_time_integral(shell):
    c,m,g,v=reference_inputs();r=PairReference(c,(.5,0,0),.27,shell)
    parent,pair=first_order_wave(r,m,g,v,12.)
    direct,error=direct_time_wave(r,m,g,v,12.)
    np.testing.assert_allclose(pair.numpy(),direct,rtol=2e-11,atol=2e-14)
    assert error<2e-10
    assert torch.linalg.vector_norm(parent).item()==pytest.approx(1,abs=7e-16)
    assert np.linalg.norm(direct)>1e-4
    assert pair.shape==(2*(2*shell+1)**2,144)


def test_spectral_pulse_exact_resonances():
    from scipy.integrate import quad
    t=12.
    for d in (0.,2*pi/t,-4*pi/t,.273):
        actual=pulse_integral(torch.tensor(d,dtype=torch.float64),t).item()
        expected=quad(lambda x: np.sin(pi*x/t)**4*np.cos(d*x),0,t,epsabs=1e-12)[0]
        expected-=1j*quad(lambda x: np.sin(pi*x/t)**4*np.sin(d*x),0,t,epsabs=1e-12)[0]
        assert abs(actual-expected)<4e-15


def test_shared_noncommuting_mass_source_and_yukawa_gradients():
    c,m,g,v=reference_inputs();r=PairReference(c,(.5,0,0),.27,1)
    direction=np.tile(np.array([[.2,1],[1,-.3]]),(4,1,1))
    # Static source s has GeV^2 units and alters SAME collective H0.
    x=torch.tensor([0.,g[0],g[1]],dtype=torch.float64,requires_grad=True)
    def response(parameters):
        mm=torch.tensor(m)+parameters[0]*torch.tensor(direction)
        _,pair=first_order_wave(r,mm,parameters[1:],v,12.)
        probe=torch.tensor(np.random.default_rng(42).normal(size=pair.shape)
                           +1j*np.random.default_rng(43).normal(size=pair.shape))
        probe=probe/torch.linalg.vector_norm(probe)
        return torch.real(torch.sum(probe.conj()*pair))
    grad=torch.autograd.grad(response(x),x)[0].numpy()
    h=2e-5;fd=[];base=x.detach().numpy()
    for i in range(3):
        e=np.eye(3)[i]*h
        vals=[response(torch.tensor(base+k*e)).item() for k in (-2,-1,1,2)]
        fd.append((vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h))
    np.testing.assert_allclose(grad,fd,rtol=2e-7,atol=2e-10)
    assert np.min(abs(grad))>1e-6
    assert np.linalg.norm(m[0]@direction[0]-direction[0]@m[0])>.1


def test_node_rotation_keeps_entire_wave_not_only_probability():
    c,m,g,v=reference_inputs();r=PairReference(c,(.5,0,0),.27,1)
    a=.43;u=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    p,w=first_order_wave(r,m,g,v,12.)
    pr,wr=first_order_wave(r,np.array([u@x@u.T for x in m]),u@g,u@v,12.)
    np.testing.assert_allclose(w,wr,atol=2e-17)
    np.testing.assert_allclose(u@p.numpy(),pr,atol=8e-16)


def test_reversing_momentum_assignments_retains_nonzero_missing_sector():
    c,m,g,v=reference_inputs();r=PairReference(c,(.5,0,0),.27,1)
    _,w=first_order_wave(r,m,g,v,12.)
    a=w.numpy()
    assert np.linalg.norm(a[9:])>.01
    assert np.linalg.norm(a[:9])==pytest.approx(np.linalg.norm(a[9:]),rel=1e-12)
    # Halving an amplitude is not counting distinguishable species correctly.
    assert np.linalg.norm(a/2-a)>.01


def test_full_first_order_wave_longitudinal_boost_and_time():
    c,m,g,v=reference_inputs();r=PairReference(c,(.5,0,0),.27,1)
    p,w=first_order_wave(r,m,g,v,12.)
    scale=1.7
    bc=LFCell(tuple(np.array(c.lengths)/[scale,1,1]))
    br=PairReference(bc,(.5*scale,0,0),.27,1)
    bp,bw=first_order_wave(br,m,g,v,12.*scale)
    np.testing.assert_allclose(p,bp,atol=5e-16)
    np.testing.assert_allclose(w,bw,atol=3e-17)


@pytest.mark.parametrize('charge',[-1,0,1])
def test_full_pair_wave_stays_in_declared_electric_charge_block(charge):
    c,m,g,v=reference_inputs();r=PairReference(c,(.5,0,0),.27,1)
    if charge:
        v=np.zeros((2,4),complex)
        v[:,1]=[1,.3j];v[:,2]=1j*charge*v[:,1]
        v/=np.linalg.norm(v)
    _,w=first_order_wave(r,m,g,v,12.)
    q=np.tile(np.repeat([2/3,-1/3],3),2)
    qpair=(q[:,None]-q[None,:]).reshape(144)
    residual=(qpair[None,:]-charge)*w.numpy()
    assert np.linalg.norm(residual)<2e-17
    assert torch.linalg.vector_norm(w)>.1
    direct,_=direct_time_wave(r,m,g,v,12.)
    np.testing.assert_allclose(w.numpy(),direct,atol=2e-14,rtol=2e-11)
