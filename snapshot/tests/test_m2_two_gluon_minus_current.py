"""Independent coincidence quadrature and Hamiltonian/current denominator checks."""
from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy import sparse
from deuteron_wigner.bridge.modes.core import ho_momentum,polarization
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    jacobi_frame,longitudinal_basis,oscillator_shell,
)
from deuteron_wigner.quantum.m2_coupled_pair_action import pair_contact_block,circular_to_c45
from deuteron_wigner.quantum.m2_canonical_instantaneous_fermion import (
    ordered_qg_color,transverse_spin_factor,
)
from deuteron_wigner.quantum.m2_constrained_fermion_current import two_gluon_current_spin
from deuteron_wigner.quantum.m2_canonical_good_current import parton_charge
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
from deuteron_wigner.quantum.m2_two_gluon_minus_current import (
    displaced_pair_contact,qg_two_gluon_minus,retained_two_gluon_minus_between,
)
from test_m2_transverse_contact_current import endpoint


@pytest.mark.parametrize("quark",range(3))
@pytest.mark.parametrize("cutoffs",[(0,0),(1,2)])
def test_displaced_pair_against_eight_dimensional_wave_quadrature(quark,cutoffs):
    incoming=(1,1,1,6);o=list(incoming);o[quark]=5;o[3]=2;outgoing=tuple(o)
    rest=tuple(i for i in range(4) if i not in (quark,3));tree=((quark,3),rest)
    fi,fo=jacobi_frame(incoming,tree),jacobi_frame(outgoing,tree)
    gi,go=jacobi_frame(incoming),jacobi_frame(outgoing)
    ri=fi.forward[:-1]@gi.inverse[:,:-1];ro=fo.forward[:-1]@go.inverse[:,:-1]
    b=.4;delta=np.array([.12,-.16]);shift=fi.forward[1:-1,quark,None]*delta
    nodes,w=hermgauss(3);indices=np.array(list(product(range(3),repeat=8)))
    t=nodes[indices].reshape(-1,4,2)
    weights=4*b**8*np.prod(w[indices],axis=1)*np.exp(np.sum(t*t,axis=(1,2)))
    spectators=b*t[:,:2]-shift/2
    qi=np.concatenate((sqrt(2)*b*t[:,2:3],spectators),axis=1)
    qo=np.concatenate((sqrt(2)*b*t[:,3:4],spectators+shift),axis=1)
    qi=np.einsum("ab,nbj->naj",ri.T,qi);qo=np.einsum("ab,nbj->naj",ro.T,qo)
    def waves(cutoff,q):
        out=[]
        for occupation in oscillator_shell(3,cutoff):
            v=np.ones(len(q),complex)
            for a in range(3):
                n,m,phase=circular_to_c45(occupation[a],occupation[a+3])
                v*=phase*ho_momentum(n,m,q[:,a,0],q[:,a,1],b)/(2*pi)
            out.append(v)
        return np.array(out)
    wi,wo=waves(cutoffs[0],qi),waves(cutoffs[1],qo)
    ai=fi.fractions[quark]*fi.fractions[3]/(fi.fractions[quark]+fi.fractions[3])
    af=fo.fractions[quark]*fo.fractions[3]/(fo.fractions[quark]+fo.fractions[3])
    expected=sqrt(ai*af)/(2*pi)**2*(wo.conj()*weights)@wi.T
    actual=displaced_pair_contact(incoming,outgoing,(quark,3),b,*cutoffs,tuple(delta))
    np.testing.assert_allclose(actual,expected,atol=2e-15)
    zero=displaced_pair_contact(incoming,outgoing,(quark,3),b,*cutoffs,(0.,0.))
    np.testing.assert_allclose(zero,pair_contact_block(incoming,outgoing,(quark,3),b,*cutoffs),atol=0.)
    reverse=displaced_pair_contact(outgoing,incoming,(quark,3),b,*cutoffs[::-1],tuple(-delta))
    np.testing.assert_allclose(actual,reverse.conj().T,atol=2e-15)


@pytest.mark.parametrize("r2",[-5,-3,-1,1,3,5])
@pytest.mark.parametrize("crossed",[False,True])
def test_ordered_current_is_fixed_field_Hamiltonian_constraint_derivative(r2,crossed):
    pp=1.3;total=9;hg=1;r=pp*r2/total
    eps=polarization(pp,0,0,hg)[2:];left,right=(eps,eps.conj()) if crossed else (eps.conj(),eps)
    current=two_gluon_current_spin(left_polarization=left,right_polarization=right,left_plus=r,right_plus=r)[1]
    # The independently normalized Hamiltonian numerator has twice the
    # current numerator at r+=1; P-minus has its additional 1/2.
    spin=np.diag([transverse_spin_factor(h,hg,crossed=crossed) for h in (-1,1)])
    epsilon=abs(r)*1e-4
    def h(z):return spin/(2*(r+z))
    derivative=(h(-2*epsilon)-8*h(-epsilon)+8*h(epsilon)-h(2*epsilon))/(12*epsilon)
    np.testing.assert_allclose(current,-derivative,rtol=3e-12,atol=1e-13)


@pytest.mark.parametrize("hg",[-1,1])
def test_ground_raw_qg_matrix_against_independent_ordered_Hamiltonian_numerators(hg):
    modes=longitudinal_basis("qqqg",9);lookup={s:i for i,s in enumerate(modes)}
    expected=np.zeros((len(modes)*128,)*2,complex);sf=np.array(SPIN_FLAVOR_BASIS);pp=1.3
    for c,state in enumerate(modes):
        for q in range(3):
            charge=parton_charge(sf[:,q],"q","electromagnetic")
            for kg in range(2,state[q]+state[3],2):
                out=list(state);out[q]=state[q]+state[3]-kg;out[3]=kg;out=tuple(out)
                contact=pair_contact_block(state,out,(q,3),.4,0)[0,0]
                block=np.zeros((128,128),complex)
                for crossed,r2 in ((False,state[q]+state[3]),(True,state[q]-kg)):
                    spin=np.array([transverse_spin_factor(2*(s[q]%2)-1,hg,crossed=crossed) for s in sf])
                    block+=np.kron(ordered_qg_color(q,crossed),np.diag(charge*spin))/r2**2
                expected[lookup[out]*128:(lookup[out]+1)*128,c*128:(c+1)*128]+=(
                    contact*9**2/(4*pi*pp**2*sqrt(kg*state[3]))*block)
    actual=qg_two_gluon_minus(9,0,0,(0,0),b_GeV=.4,total_pplus_GeV=pp,
                              gluon_helicity=hg,charge="electromagnetic").toarray()
    np.testing.assert_allclose(actual,expected,atol=3e-15)
    assert np.linalg.norm(actual)>1e-3


@pytest.mark.parametrize("target,jz",[("PROTON",1),("PROTON",-1),("NEUTRON",1),("NEUTRON",-1)])
def test_retained_quadratic_minus_reversal_boost_flavor_and_exact_support(target,jz):
    i=endpoint(1,target,1);f=endpoint(1,target,jz);delta=np.array([.12,.16])
    def j(out,inc,d,charge="electromagnetic",pp=1.3):
        return retained_two_gluon_minus_between(out,inc,d,total_pplus_GeV=pp,charge=charge)
    actual=j(f,i,delta)
    np.testing.assert_allclose(actual,j(i,f,-delta).conj().T,atol=5e-15)
    np.testing.assert_allclose(j(f,i,delta,pp=2.6),actual/4,atol=0.)
    np.testing.assert_allclose((2*j(f,i,delta,"up_number")-j(f,i,delta,"down_number"))/3,actual,atol=5e-15)
    np.testing.assert_array_equal(actual[:f.bases[0].shape[1]],0.)
    np.testing.assert_array_equal(actual[:,:i.bases[0].shape[1]],0.)
    if jz==1:assert np.linalg.norm(actual)>1e-3


def test_invalid_quadratic_current():
    with pytest.raises(ValueError):
        qg_two_gluon_minus(9,0,0,(0,0),b_GeV=.4,total_pplus_GeV=0,
                           gluon_helicity=1,charge="baryon")
