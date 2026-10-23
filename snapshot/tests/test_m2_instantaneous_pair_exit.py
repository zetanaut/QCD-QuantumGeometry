from itertools import product
from math import pi,sqrt
import numpy as np
import pytest
from scipy.special import roots_hermite
from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS,ho_momentum
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell,oscillator_rotation
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_permutation,frame
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import (
    pair_spin_factor,pair_ordered_color,pair_mass_kernel,pair_emission_frame,
    pair_contact_orbital,build_instantaneous_pair_source,
    adjoint_pair_mass_kernel,build_adjoint_pair_source,build_combined_pair_source,
)


@pytest.mark.parametrize("h1,h2",list(product((-1,1),repeat=2)))
def test_full_spinor_and_finite_cell_coefficient_without_reusing_normalized_kernel(h1,h2):
    inc,out,slot=(7,1,1),(3,1,1,2,2),0
    for pp,mass,hq in product((.7,3.9),(.17,.8),(-1,1)):
        cell=pi*9/pp
        p0,pf,k1,k2=pp*np.array([7,3,2,2])/9
        u=lf_spinor(pf,.23,-.19,mass,hq)
        v=lf_spinor(p0,-.17,.31,mass,hq)
        def gamma_eps(h):
            eps=(-np.array([h,1j])/sqrt(2)).conj()
            return GAMMA[1]*eps[0]+GAMMA[2]*eps[1]
        # SB's -g²/2 ubar gamma+ gamma.A 1/(i d+) gamma.A u.
        # Both creation assignments are present, all four field norms explicit.
        norm=sqrt((2*pf)*(2*p0)*(2*k1)*(2*k2))
        actual=np.zeros((6,1),complex)
        for left,right,kr,reverse in ((h1,h2,k2,False),(h2,h1,k1,True)):
            spin=u.conj()@GAMMA[0]@GAMMA_PLUS@gamma_eps(left)@gamma_eps(right)@v
            actual+=(-pp/(cell*norm))*spin/(p0-kr)*pair_ordered_color(slot,reverse)
        np.testing.assert_allclose(pair_mass_kernel(inc,out,slot,(hq,hq,h1,h2)),actual,atol=4e-16)
        assert pair_spin_factor(-hq,hq,h1,h2)==0.


@pytest.mark.parametrize("slot",[0,1,2])
def test_ordered_colors_and_two_assignment_bose_exchange(slot):
    out=[1,1,1,2,4];out[slot]=3
    inc=out[:3];inc[slot]+=6
    exchanged=tuple(out[:3]+[4,2]);out=tuple(out);inc=tuple(inc)
    c=color_permutation("qqqgg",(0,1,2,4,3))
    np.testing.assert_allclose(c@pair_ordered_color(slot),pair_ordered_color(slot,True),atol=2e-14)
    for h1,h2 in product((-1,1),repeat=2):
        a=pair_mass_kernel(inc,out,slot,(1,1,h1,h2))
        b=pair_mass_kernel(inc,exchanged,slot,(1,1,h2,h1))
        np.testing.assert_allclose(c@a,b,atol=3e-16)


@pytest.mark.parametrize("slot",[0,1,2])
def test_two_local_jacobi_measure_ground_integral_and_retained_excitation(slot):
    out=[1,1,1,2,2];out[slot]=3
    inc=out[:3];inc[slot]+=4;out=tuple(out);inc=tuple(inc)
    rotation=pair_emission_frame(inc,out,slot)
    np.testing.assert_allclose(rotation@rotation.T,np.eye(4),atol=1e-14)
    b=.43
    x,y,z=frame("qqqgg",out).fractions[[slot,3,4]]
    measure=sqrt(x*y*z/(x+y+z))
    nodes,weights=roots_hermite(10)
    xx,yy=np.meshgrid(sqrt(2)*b*nodes,sqrt(2)*b*nodes,indexing="ij")
    phi=ho_momentum(0,0,xx,yy,b)*np.exp((xx*xx+yy*yy)/(2*b*b))
    integral=np.sum(np.outer(weights,weights)*phi)*2*b*b/(2*pi)**2
    ground=pair_contact_orbital(inc,out,slot,0,0,b)
    assert ground[0,0]==pytest.approx(measure*integral**2,abs=3e-16)
    np.testing.assert_allclose(pair_contact_orbital(inc,out,slot,1,1,2*b),
                               4*pair_contact_orbital(inc,out,slot,1,1,b),atol=2e-16)
    adapted=oscillator_rotation(rotation,1)@pair_contact_orbital(inc,out,slot,1,1,b)
    source=oscillator_shell(2,1)
    for row,s in enumerate(oscillator_shell(4,1)):
        inherited=s[:2]+s[4:6]
        for col,t in enumerate(source):
            expected=measure*b*b/pi if inherited==t and s[2]==s[6] and s[3]==s[7] else 0.
            assert adapted[row,col]==pytest.approx(expected,abs=2e-16)


@pytest.mark.parametrize("target,jz,cutoff",[("PROTON",1,0),("NEUTRON",-1,0),("PROTON",1,1)])
def test_actual_source_statistics_nonzero_self_return_and_higher_shell_embedding(target,jz,cutoff):
    p=build_coupled_pilot(total_k2=9,cutoff=cutoff,target=target,jz_twice=jz,b_GeV=.4,
                          quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    source=build_instantaneous_pair_source(p,out_cutoff=max(cutoff,1))
    v=source.matrix_per_gs2;s=source.space
    assert np.linalg.norm(v)>1e-4
    assert np.array_equal(source.retained_columns,np.arange(p.bases[0].shape[1]))
    np.testing.assert_allclose(s.permute(v,(1,0,2,3,4)),-v,atol=3e-14)
    np.testing.assert_allclose(s.permute(v,(0,1,2,4,3)),v,atol=3e-14)
    energies=s.free_energies(.4,.35,0.);e=1.4
    sigma=v.conj().T@(v/(e-energies[:,None]))
    assert np.linalg.eigvalsh(sigma)[-1]<2e-16
    assert np.linalg.norm(sigma)>1e-6
    # This separate contact squared is g^4, not a newly relabeled cubic g^2 term.
    assert np.linalg.norm((.6**4)*sigma)/np.linalg.norm((.3**4)*sigma)==pytest.approx(16.)
    if cutoff==0:
        low=build_instantaneous_pair_source(p,out_cutoff=0)
        rows=s.coordinates[:,1]==0
        np.testing.assert_allclose(v[rows],low.matrix_per_gs2,atol=1e-16)
        np.testing.assert_allclose(v[~rows],0.,atol=1e-16)


@pytest.mark.parametrize("inc,out,slot",[((3,3,3),(1,1,1,2,2),0),
                                       ((7,1,1),(1,3,1,2,2),0),
                                       ((7,1,1),(3,1,1,2,2),True)])
def test_mismatched_fraction_routes_rejected(inc,out,slot):
    with pytest.raises(ValueError):
        pair_mass_kernel(inc,out,slot,(1,1,-1,1))


@pytest.mark.parametrize("modes",[(3,1,1,2,2),(1,1,1,2,4),(1,1,1,4,2)])
def test_mixed_adjoint_current_from_explicit_derivatives_and_full_field_norm(modes):
    from deuteron_wigner.microscopic.h0.color import ColorSingletBasis,representation_generators
    from deuteron_wigner.pilot.color import structure_constants
    from deuteron_wigner.quantum.m2_excluded_fock_kinematics import color_basis
    inc=(modes[0]+modes[3]+modes[4],)+modes[1:3]
    pp=2.7;cell=pi*sum(modes)/pp
    pf,p0,k1,k2=pp*np.array([modes[0],inc[0],modes[3],modes[4]])/sum(modes)
    u=lf_spinor(pf,.13,.28,.41,1);v=lf_spinor(p0,-.24,.08,.41,1)
    jq=u.conj()@GAMMA[0]@GAMMA_PLUS@v
    original=ColorSingletBasis.construct("qqq").tensors.reshape(3,3,3,1)
    t=representation_generators("3");f=structure_constants()
    raw=np.zeros((3,3,3,8,8,1),complex)
    for a,b in product(range(8),repeat=2):
        # h1=-1,h2=+1 gives eps1* dot eps2*=-1.
        matrix=sum(-1j*(k1-k2)*(-1)*f[c,a,b]*t[c] for c in range(8))
        raw[:,:,:,a,b,:]=np.tensordot(matrix,original,axes=(1,0))
    norm=sqrt((2*pf)*(2*p0)*(2*k1)*(2*k2))
    expected=2*pp/cell*jq/norm/(k1+k2)**2*(
        color_basis("qqqgg").conj().T@raw.reshape(1728,1))
    actual=adjoint_pair_mass_kernel(inc,modes,0,(1,1,-1,1))
    np.testing.assert_allclose(actual,expected,atol=4e-16)
    reverse=modes[:3]+(modes[4],modes[3])
    c=color_permutation("qqqgg",(0,1,2,4,3))
    np.testing.assert_allclose(c@actual,adjoint_pair_mass_kernel(inc,reverse,0,(1,1,1,-1)),atol=4e-16)
    if k1==k2:
        assert np.linalg.norm(actual)==0.


def test_combined_actual_pair_source_keeps_interference_and_statistics():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
                          quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    a=build_instantaneous_pair_source(p,out_cutoff=1)
    b=build_adjoint_pair_source(p,out_cutoff=1)
    c=build_combined_pair_source(p,out_cutoff=1)
    np.testing.assert_allclose(c.matrix_per_gs2,a.matrix_per_gs2+b.matrix_per_gs2,atol=2e-16)
    assert np.linalg.norm(b.matrix_per_gs2)>1e-4
    assert abs(np.linalg.norm(c.matrix_per_gs2)**2-np.linalg.norm(a.matrix_per_gs2)**2
               -np.linalg.norm(b.matrix_per_gs2)**2)>1e-7
    np.testing.assert_allclose(c.space.permute(c.matrix_per_gs2,(1,0,2,3,4)),-c.matrix_per_gs2,atol=3e-14)
