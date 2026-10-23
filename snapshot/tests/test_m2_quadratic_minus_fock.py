"""Quadratic number-changing current: Dirac numerators and exact coincidence."""
from itertools import product
from math import sqrt,pi
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS,ho_momentum
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell,jacobi_frame
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import frame as frame5
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import pair_contact_orbital,pair_ordered_color,pair_emission_frame
from deuteron_wigner.quantum.m2_excluded_pair_conversion import conversion_contact,conversion_ordered_color
from deuteron_wigner.quantum.m2_excluded_current_scattering import five_body_contact
from deuteron_wigner.quantum.m2_six_body_quartic import (
    direct_routes,contact_orbital,valence_ordered_color,gluon_pair_ordered_color,
)
from deuteron_wigner.quantum.m2_quadratic_minus_fock import (
    pair_emission_minus_kernel,conversion_minus_kernel,direct_six_minus_kernel,
    pair_emission_minus_orbital,direct_six_minus_orbital,five_displaced_pair_contact,
    five_pair_adapted_rows,
)


def dirac_numerator(hf,hi,left,right,pair):
    # Arbitrary unequal plus/transverse endpoints and nonzero mass, rather
    # than the numerator helpers' fixed reference spinors.
    pf,pi_=1.7,.8
    u=lf_spinor(pf,.13,-.21,.47,hf)/sqrt(2*pf)
    fn=antiquark_spinor if pair else lf_spinor
    v=fn(pi_,-.04,.17,.47,hi)/sqrt(2*pi_)
    a=sum(GAMMA[j+1]*left[j] for j in (0,1));b=sum(GAMMA[j+1]*right[j] for j in (0,1))
    return u.conj()@GAMMA[0]@a@GAMMA_PLUS@b@v


@pytest.mark.parametrize("kind",["pair","conversion","valence_IF","gluon_IF_pair"])
@pytest.mark.parametrize("pp",[.7,1.3,2.6])
def test_all_spin_kernels_against_full_Dirac_ordered_insertions(kind,pp):
    if kind=="pair":inc=(5,1,1);out=(1,1,1,2,2);q=0;K=7
    elif kind=="conversion":inc=(1,1,1,2,4);out=(1,1,1,3,3);K=9
    else:route=direct_routes(9,kind)[0];inc,out=route.incoming,route.outgoing;K=9
    for hs in product((-1,1),repeat=4):
        hf,hi,h1,h2=hs
        eps=lambda h:-np.array([h,1j])/sqrt(2)
        if kind in ("pair","valence_IF"):
            q=0 if kind=="pair" else route.emitter
            k1,k2=out[-2:];r1=(inc[q]-k2)*pp/K;r2=(inc[q]-k1)*pp/K
            c1=pair_ordered_color(q) if kind=="pair" else valence_ordered_color(q)
            c2=pair_ordered_color(q,True) if kind=="pair" else valence_ordered_color(q,True)
            s1=dirac_numerator(hf,hi,eps(h1).conj(),eps(h2).conj(),False)
            s2=dirac_numerator(hf,hi,eps(h2).conj(),eps(h1).conj(),False)
            expected=(s1*c1/r1**2+s2*c2/r2**2)/(4*pi*sqrt(k1*k2))
            if kind=="pair":actual=pair_emission_minus_kernel(inc,out,q,hs,total_pplus_GeV=pp)
            else:
                expected*=sqrt(6)/2
                actual=direct_six_minus_kernel(route,hs,total_pplus_GeV=pp)
        elif kind=="conversion":
            k1,k2=inc[-2:];r1=(k2-out[4])*pp/K;r2=(k1-out[4])*pp/K
            s1=dirac_numerator(hf,hi,eps(h1),eps(h2),True)
            s2=dirac_numerator(hf,hi,eps(h2),eps(h1),True)
            expected=(s1*conversion_ordered_color()/r1**2+s2*conversion_ordered_color(True)/r2**2)/(4*pi*sqrt(k1*k2))
            actual=conversion_minus_kernel(inc,out,hs,total_pplus_GeV=pp)
        else:
            kin,kout=inc[3],out[5];r1=(kin-out[4])*pp/K;r2=(-kout-out[4])*pp/K
            s1=dirac_numerator(hf,hi,eps(h1).conj(),eps(h2),True)
            s2=dirac_numerator(hf,hi,eps(h2),eps(h1).conj(),True)
            expected=2*(s1*gluon_pair_ordered_color()/r1**2+s2*gluon_pair_ordered_color(True)/r2**2)/(4*pi*sqrt(kin*kout))
            actual=direct_six_minus_kernel(route,hs,total_pplus_GeV=pp)
        np.testing.assert_allclose(actual,expected,atol=3e-14)


@pytest.mark.parametrize("cutoff",[0,1,2])
def test_zero_transfer_new_orbitals_equal_independent_Hamiltonian_coincidences(cutoff):
    inc=(5,1,1);out=(1,1,1,2,2)
    actual=pair_emission_minus_orbital(inc,out,0,cutoff,cutoff,.4,(0.,0.))
    np.testing.assert_allclose(actual,pair_contact_orbital(inc,out,0,cutoff,cutoff,.4),atol=0.)
    for kind in ("valence_IF","gluon_IF_pair"):
        route=direct_routes(9,kind)[0]
        np.testing.assert_allclose(direct_six_minus_orbital(route,cutoff,cutoff,.4,(0.,0.)),
                                   contact_orbital(route,cutoff,cutoff,.4),atol=0.)
    inc=(1,1,1,2,4);out=(1,1,1,3,3)
    c=five_displaced_pair_contact("qqqgg","qqqq-qbar",inc,out,(3,4),cutoff,cutoff,.4,(0.,0.))
    # Real Hamiltonian and complex current products use different BLAS paths.
    # Exact structural zeros are not exact floating-point cancellations.
    roundoff=32*np.finfo(float).eps*np.linalg.norm(c,2)
    np.testing.assert_allclose(c,conversion_contact(inc,out,.4,cutoff),rtol=0.,atol=roundoff)
    out=(3,1,1,2,2)
    c=five_displaced_pair_contact("qqqgg","qqqgg",inc,out,(0,4),cutoff,cutoff,.4,(0.,0.))
    roundoff=32*np.finfo(float).eps*np.linalg.norm(c,2)
    np.testing.assert_allclose(c,five_body_contact("qqqgg",inc,out,(0,4),.4,cutoff),rtol=0.,atol=roundoff)


def waves(d,cutoff,q,b):
    result=[]
    for occupation in oscillator_shell(d,cutoff):
        v=np.ones(len(q),complex)
        for a in range(d):
            n,m,phase=circular_to_c45(occupation[a],occupation[d+a])
            v*=phase*ho_momentum(n,m,q[:,a,0],q[:,a,1],b)/(2*pi)
        result.append(v)
    return np.array(result)


def test_double_emission_recoil_against_eight_dimensional_polar_integral():
    inc=(5,1,1);out=(1,1,1,2,2);b=.4;delta=np.array([.12,-.16])
    fi=jacobi_frame(inc);fo=frame5("qqqgg",out);shift=fi.forward[:-1,0,None]*delta
    nodes,w=hermgauss(3);idx=np.array(list(product(range(3),repeat=8)));t=nodes[idx].reshape(-1,4,2)
    weights=4*b**8*np.prod(w[idx],axis=1)*np.exp(np.sum(t*t,axis=(1,2)))
    q=b*t[:,:2]-shift/2;adapted=np.concatenate((q+shift,sqrt(2)*b*t[:,2:]),axis=1)
    global_out=np.einsum("ab,nbj->naj",pair_emission_frame(inc,out,0).T,adapted)
    x=fo.fractions[[0,3,4]]
    expected=sqrt(np.prod(x)/x.sum())/(2*pi)**2*(waves(4,1,global_out,b).conj()*weights)@waves(2,1,q,b).T
    actual=pair_emission_minus_orbital(inc,out,0,1,1,b,tuple(delta))
    np.testing.assert_allclose(actual,expected,atol=2e-15)


@pytest.mark.parametrize("conversion",[False,True])
def test_five_pair_recoil_against_ten_dimensional_polar_integral(conversion):
    inc=(1,1,1,2,4);out=(1,1,1,3,3) if conversion else (3,1,1,2,2)
    sector="qqqq-qbar" if conversion else "qqqgg";pair=(3,4) if conversion else (0,4)
    fi,fo=frame5("qqqgg",inc),frame5(sector,out);ri,ro=five_pair_adapted_rows(fi,pair),five_pair_adapted_rows(fo,pair)
    b=.4;delta=np.array([.12,-.16]);shift=ri[1:,pair[0],None]*delta
    nodes,w=hermgauss(3);idx=np.array(list(product(range(3),repeat=10)));t=nodes[idx].reshape(-1,5,2)
    weights=4*b**10*np.prod(w[idx],axis=1)*np.exp(np.sum(t*t,axis=(1,2)))
    spectator=b*t[:,:3]-shift/2
    qi=np.concatenate((sqrt(2)*b*t[:,3:4],spectator),axis=1)
    qo=np.concatenate((sqrt(2)*b*t[:,4:5],spectator+shift),axis=1)
    qi=np.einsum("ab,nbj->naj",(ri@fi.inverse[:,:-1]).T,qi)
    qo=np.einsum("ab,nbj->naj",(ro@fo.inverse[:,:-1]).T,qo)
    ai=np.prod(fi.fractions[list(pair)])/sum(fi.fractions[list(pair)])
    ao=np.prod(fo.fractions[list(pair)])/sum(fo.fractions[list(pair)])
    expected=sqrt(ai*ao)/(2*pi)**2*(waves(4,1,qo,b).conj()*weights)@waves(4,1,qi,b).T
    actual=five_displaced_pair_contact("qqqgg",sector,inc,out,pair,1,1,b,tuple(delta))
    np.testing.assert_allclose(actual,expected,atol=2e-15)
    reverse=five_displaced_pair_contact(sector,"qqqgg",out,inc,pair,1,1,b,tuple(-delta))
    np.testing.assert_allclose(actual,reverse.conj().T,atol=2e-15)


def test_unrelated_quartic_Hamiltonian_is_not_an_implicit_current():
    route=direct_routes(9,"gluon_A4")[0]
    with pytest.raises(ValueError):
        direct_six_minus_kernel(route,(1,1,1,1),total_pplus_GeV=1.3)
