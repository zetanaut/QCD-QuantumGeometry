"""Polar-wave quadrature and Lorentz/charge checks of the minus contact."""
from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from deuteron_wigner.bridge.modes.core import ho_momentum,polarization,GAMMA
from deuteron_wigner.quantum.m2_constrained_fermion_current import (
    GOOD_AXES,ANTI_AXES,one_gluon_current_spin,pair_creation_current_spin,
)
from deuteron_wigner.quantum.m2_antiquark_contact_current import antiquark_one_gluon_current_spin
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    emission_routes,jacobi_frame,emission_coordinate_map,oscillator_shell,
)
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame,split_frame,split_routes
from deuteron_wigner.quantum.m2_six_body_vertices import split_frame as six_frame,split_routes as six_routes
from deuteron_wigner.quantum.m2_six_body_kinematics import frame as six_body_frame
from deuteron_wigner.quantum.m2_one_gluon_minus_contact import (
    local_linear_moments,one_gluon_minus_orbital,minus_contact_between,
)
from deuteron_wigner.quantum.m2_transverse_contact_current import transverse_contact_between
from test_m2_transverse_contact_current import endpoint


def route_data(kind):
    if kind=="retained":
        r=emission_routes(9)[0];fi=jacobi_frame(r.incoming);fo=jacobi_frame(r.outgoing)
        return fi,r.emitter,*fo.fractions[[r.emitter,3]],emission_coordinate_map(r),"q_to_qg"
    if kind in ("q_to_qg","g_to_pair"):
        r=split_routes(9,kind)[0];fi=frame("qqqg",r.incoming);fo=frame(r.sector,r.outgoing)
        return fi,r.parent,*fo.fractions[list(r.daughters)],split_frame(r),kind
    r=six_routes(9,"qqqq-qbar","anti_to_antig")[0]
    fi=frame(r.source_sector,r.incoming);fo=six_body_frame(r.sector,r.outgoing)
    return fi,r.parent,*fo.fractions[list(r.daughters)],six_frame(r),"anti_to_antig"


@pytest.mark.parametrize("kind",["retained","q_to_qg","g_to_pair","anti_to_antig"])
@pytest.mark.parametrize("hg",[-1,1])
def test_full_linear_spin_contact_against_independent_polar_quadrature(kind,hg):
    fi,active,x,y,rotation,channel=route_data(kind);d=len(fi.fractions)-1
    b=.4;mass=.35;pp=1.3;delta=np.array([.12,-.16]);mean=np.array([.07,.03])
    shift=fi.forward[:-1,active,None]*delta;xp=x+y;scale=sqrt(x*y/xp)
    nodes,w=hermgauss(3);idx=np.array(list(product(range(3),repeat=2*(d+1))))
    t=nodes[idx].reshape(-1,d+1,2)
    weights=np.prod(w[idx],axis=1)*2*b**(2*(d+1))*np.exp(np.sum(t*t,axis=(1,2)))
    q=b*t[:,:d]-shift/2;r=sqrt(2)*b*t[:,d]
    adapted=np.concatenate((q+shift,r[:,None,:]),axis=1)
    qo=np.einsum("ab,nbj->naj",rotation.T,adapted)
    def waves(dim,cutoff,mom):
        result=[]
        for occupation in oscillator_shell(dim,cutoff):
            value=np.ones(len(mom),complex)
            for a in range(dim):
                n,m,phase=circular_to_c45(occupation[a],occupation[dim+a])
                value*=phase*ho_momentum(n,m,mom[:,a,0],mom[:,a,1],b)/(2*pi)
            result.append(value)
        return np.array(result)
    wi,wo=waves(d,0,q),waves(d+1,1,qo)
    pin=np.einsum("a,naj->nj",fi.inverse[active,:-1],q)+xp*(mean-delta/2)
    pout=x/xp*(pin+delta)+scale*r
    pa=y/xp*(pin+delta)-scale*r
    alpha=np.array([GAMMA[0]@GAMMA[j] for j in (1,2)])
    eps=polarization((xp if channel=="g_to_pair" else y)*pp,0,0,hg)[2:]
    if channel!="g_to_pair":eps=eps.conj()
    def bad(sign,plus,momentum):
        return (np.einsum("nj,jab->nab",sign*momentum,alpha)+GAMMA[0]*mass)/(sqrt(2)*sign*plus)
    if channel=="q_to_qg":
        bf,bi=bad(1,x*pp,pout),bad(1,xp*pp,pin);rp,lp=x*pp,xp*pp
        left_axes=right_axes=GOOD_AXES
    elif channel=="g_to_pair":
        bf,bi=bad(1,x*pp,pout),bad(-1,y*pp,pa);rp,lp=x*pp,-y*pp
        left_axes,right_axes=GOOD_AXES,ANTI_AXES
    else:
        bf,bi=bad(-1,xp*pp,pin),bad(-1,x*pp,pout);rp,lp=-xp*pp,-x*pp
        left_axes=right_axes=ANTI_AXES
    vertex=np.einsum("j,jab->ab",eps,alpha)/sqrt(2)
    matrix=bf.conj().swapaxes(1,2)@(vertex/rp)+(vertex/lp)@bi
    spin=np.einsum("ha,nab,bk->nhk",left_axes.conj().T,matrix,right_axes)
    if channel=="anti_to_antig":spin=-spin.swapaxes(1,2)
    expected=scale/(2*pi)*np.einsum("on,nhk,in,n->hkoi",wo.conj(),spin,wi,weights,optimize=True)
    actual=one_gluon_minus_orbital(fi,active,x,y,rotation,0,1,delta,channel=channel,
        b_GeV=b,total_pplus_GeV=pp,mass_GeV=mass,gluon_helicity=hg,mean_total_transverse=mean)
    np.testing.assert_allclose(actual,expected,atol=3e-15)
    assert np.linalg.norm(actual)>1e-3


@pytest.mark.parametrize("kind",["retained","q_to_qg","g_to_pair","anti_to_antig"])
def test_exact_contact_longitudinal_and_transverse_boosts(kind):
    fi,active,x,y,rotation,channel=route_data(kind);d=len(fi.fractions)-1
    delta=np.array([.12,.16]);boost=np.array([.09,-.05]);pp=1.3;hg=1
    def j(p=pp,mean=(0.,0.)):
        return one_gluon_minus_orbital(fi,active,x,y,rotation,0,1,delta,channel=channel,
            b_GeV=.4,total_pplus_GeV=p,mass_GeV=.35,gluon_helicity=hg,mean_total_transverse=mean)
    actual=j()
    np.testing.assert_allclose(j(2*pp),actual/4,atol=5e-16)
    xp=x+y;eps=polarization((xp if channel=="g_to_pair" else y)*pp,0,0,hg)[2:]
    if channel!="g_to_pair":eps=eps.conj()
    if channel=="g_to_pair":
        spin=pair_creation_current_spin((x*pp,0,0),(y*pp,0,0),mass_GeV=.35,
            polarization=eps,right_plus=x*pp,left_plus=-y*pp)[2:]
    else:
        fn=one_gluon_current_spin if channel=="q_to_qg" else antiquark_one_gluon_current_spin
        spin=fn((x*pp,0,0),(xp*pp,0,0),mass_GeV=.35,polarization=eps,
            right_plus=x*pp if channel=="q_to_qg" else -xp*pp,
            left_plus=xp*pp if channel=="q_to_qg" else -x*pp)[2:]
    base,_,_=local_linear_moments(d,0,1,fi.forward[:-1,active,None]*delta,
                                 fi.inverse[active,:-1],b_GeV=.4)
    from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_rotation
    orbital=sqrt(x*y/xp)*oscillator_rotation(rotation,1).T@base
    expected=np.einsum("j,jhk->hk",boost/pp,spin)[:,:,None,None]*orbital
    np.testing.assert_allclose(j(mean=boost)-actual,expected,atol=2e-15)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("PROTON",-1),("NEUTRON",1),("NEUTRON",-1)])
def test_retained_minus_reversal_flavor_and_boost(target,jz):
    i=endpoint(1,target,1);f=endpoint(1,target,jz);delta=np.array([.12,.16]);pp=1.3
    def j(out,inc,d,charge="electromagnetic",mean=(0.,0.),p=pp):
        return minus_contact_between(out,inc,d,total_pplus_GeV=p,charge=charge,mean_total_transverse=mean)
    actual=j(f,i,delta)
    np.testing.assert_allclose(actual,j(i,f,-delta).conj().T,atol=3e-15)
    np.testing.assert_allclose((2*j(f,i,delta,"up_number")-j(f,i,delta,"down_number"))/3,actual,atol=3e-15)
    np.testing.assert_allclose(j(f,i,delta,p=2*pp),actual/4,atol=0.)
    jt=transverse_contact_between(f,i,delta,total_pplus_GeV=pp,charge="electromagnetic")
    boost=np.array([.09,-.05])
    np.testing.assert_allclose(j(f,i,delta,mean=boost)-actual,np.einsum("j,joi->oi",boost/pp,jt),atol=3e-15)
    assert np.linalg.norm(actual)>1e-4


def test_relative_moment_retains_outside_bra_neighbor():
    zero,parent,relative=local_linear_moments(1,0,1,np.zeros((1,2)),np.array([1.]),b_GeV=.4)
    # Every first-excited relative mode receives vacuum AND radial-excited
    # m=0 bra neighbors, even though the latter exceeds external cutoff1.
    shell=oscillator_shell(2,1)
    for rel_index in (1,3):
        row=next(i for i,q in enumerate(shell) if q[rel_index]==1)
        assert abs(relative[0,row,0]-.4**2/sqrt(pi))<1e-16
    assert np.linalg.norm(parent)>0


def test_invalid_contact_kinematics():
    fi,a,x,y,rot,ch=route_data("retained")
    with pytest.raises(ValueError):
        one_gluon_minus_orbital(fi,a,x,y+.1,rot,0,1,(0,0),channel=ch,b_GeV=.4,
            total_pplus_GeV=1.3,mass_GeV=.35,gluon_helicity=1)
