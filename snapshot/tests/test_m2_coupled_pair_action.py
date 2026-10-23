"""Independent Fourier, contact quadrature and coupled qq symmetry checks."""
from itertools import product
from math import comb, factorial, pi, sqrt

import numpy as np
import pytest
from scipy import sparse
from scipy.special import eval_hermite, roots_hermite

from deuteron_wigner.bridge.modes.core import ho_coordinate, ho_momentum
from deuteron_wigner.quantum.m2_c117_pair_current import color_permutation_matrix
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    S3, jacobi_frame, longitudinal_basis, longitudinal_orbital_action,
    oscillator_shell, total_sign_project,
)
from deuteron_wigner.quantum.m2_coupled_pair_action import (
    circular_to_c45, ground_width_tail, qq_contact_block,
    qq_pair_mass_per_gs2, radial_width_overlap,
)


def maxabs(a):
    v = a.data if sparse.issparse(a) else np.asarray(a)
    return float(np.max(np.abs(v))) if v.size else 0.


@pytest.mark.parametrize("p,m", list(product(range(4),repeat=2)))
def test_circular_c45_phase_from_cartesian_creation_expansion(p,m):
    # Independent Cartesian Hermite functions, not polar Laguerre identities.
    x,y=np.array([.03,.17,-.22]),np.array([-.12,.29,.11])
    b=.43
    cartesian=np.zeros_like(x,dtype=complex)
    for ny in range(p+m+1):
        nx=p+m-ny
        coeff=sum(comb(p,k)*comb(m,ny-k)*1j**k*(-1j)**(ny-k)
                  for k in range(p+1) if 0<=ny-k<=m)
        coeff*=sqrt(factorial(nx)*factorial(ny)/(factorial(p)*factorial(m)))/2**((p+m)/2)
        cartesian+=coeff*eval_hermite(nx,x/b)*eval_hermite(ny,y/b)/sqrt(
            2**(nx+ny)*factorial(nx)*factorial(ny))
    cartesian*=np.exp(-(x*x+y*y)/(2*b*b))/(sqrt(pi)*b)
    n,angular,phase=circular_to_c45(p,m)
    np.testing.assert_allclose(cartesian,phase*ho_momentum(n,angular,x,y,b)/(2*pi),
                               atol=3e-14)


@pytest.mark.parametrize("m",[-2,-1,0,1,2])
def test_width_overlap_independent_cartesian_quadrature_and_adjoint(m):
    bo,bi=.4,.4*sqrt(3)
    t,w=roots_hermite(14)
    scale=(bo**-2+bi**-2)/2
    x,y=np.meshgrid(t/sqrt(scale),t/sqrt(scale),indexing="ij")
    weights=np.outer(w,w)/scale*np.exp(scale*(x*x+y*y))
    for no,ni in product(range(4),repeat=2):
        a=(-1)**no*ho_momentum(no,m,x,y,bo)/(2*pi)
        b=(-1)**ni*ho_momentum(ni,m,x,y,bi)/(2*pi)
        direct=np.sum(weights*a.conj()*b)
        value=radial_width_overlap(no,ni,m,bo,bi)
        assert value==pytest.approx(direct,abs=2e-13)
        assert value==pytest.approx(radial_width_overlap(ni,no,m,bi,bo),abs=2e-13)


def test_old_qg_ground_squeeze_tail_not_a_finite_isometry():
    for cutoff in range(9):
        amps=np.array([radial_width_overlap(n,0,0,.4,sqrt(3)*.4)
                       for n in range(cutoff//2+1)])
        # Circular phases remove the alternating sign of the C45 Laguerres.
        np.testing.assert_allclose(amps,sqrt(3)/2*.5**np.arange(len(amps)),atol=2e-14)
        tail=ground_width_tail(.4,sqrt(3)*.4,cutoff)
        assert np.sum(amps**2)+tail==pytest.approx(1,abs=2e-14)
        assert tail==pytest.approx(4.**(-(cutoff//2+1)))
    assert ground_width_tail(.4,.4,0)==0


def direct_coordinate_contact(incoming,outgoing,pair,b,cutoff):
    """Integrate the physical r_i=r_j surface, independent of pair-tree maps."""
    d=len(incoming)-1
    i,j=pair
    constraint=np.zeros(d)
    if i<d: constraint[i]+=1
    if j<d: constraint[j]-=1
    pivot=np.flatnonzero(constraint)[0]
    free=[k for k in range(d) if k!=pivot]
    z=np.eye(d)[:,free]
    z[pivot]=-constraint[free]/constraint[pivot]
    frames=[jacobi_frame(k) for k in (incoming,outgoing)]
    maps=[f.inverse[:-1,:-1].T@z for f in frames]
    precision=b*b*sum(a.T@a for a in maps)
    transform=sqrt(2)*np.linalg.inv(np.linalg.cholesky(precision).T)
    t,w=roots_hermite(cutoff+2)
    labels=np.array(list(product(range(len(t)),repeat=2*(d-1))))
    h=t[labels].reshape(-1,d-1,2)
    v=np.einsum("ij,njk->nik",transform,h)
    points=[np.einsum("ij,njk->nik",a,v) for a in maps]
    weight=np.prod(w[labels],axis=1)*2**(d-1)/np.linalg.det(precision)
    # Remove the combined Gaussian already supplied by the Hermite weights.
    weight*=np.exp(np.sum(h*h,axis=(1,2)))
    shell=oscillator_shell(d,cutoff)
    values=[]
    for frame,r in zip(frames,points):
        modes=[]
        for s in shell:
            value=np.ones(len(r),complex)*sqrt(frame.transverse_measure_factor)
            for k in range(d):
                n=min(s[k],s[d+k]); m=s[k]-s[d+k]
                value*=(-1)**n*ho_coordinate(n,m,r[:,k,0],r[:,k,1],b)
            modes.append(value)
        values.append(np.array(modes))
    return (values[1].conj()*weight)@values[0].T


@pytest.mark.parametrize("incoming,outgoing,pair",[
    ((9,9,9),(1,17,9),(0,1)),
    ((1,9,17),(1,15,11),(1,2)),
    ((3,7,9,8),(5,5,9,8),(0,1)),
    ((1,3,9,14),(7,3,3,14),(0,2)),
])
def test_all_excited_contact_elements_against_physical_coordinate_quadrature(incoming,outgoing,pair):
    expected=direct_coordinate_contact(incoming,outgoing,pair,.4,2)
    actual=qq_contact_block(incoming,outgoing,pair,.4,2)
    np.testing.assert_allclose(actual,expected,atol=3e-15)
    np.testing.assert_allclose(actual.T,qq_contact_block(outgoing,incoming,pair,.4,2),
                               atol=3e-15)


def test_ground_contact_equal_share_and_new_fraction_factor():
    same=qq_contact_block((9,9,9),(9,9,9),(0,1),.4,0)[0,0]
    assert same==pytest.approx(.16/(6*pi))
    different=qq_contact_block((9,9,9),(1,17,9),(0,1),.4,0)[0,0]
    assert different/same==pytest.approx(sqrt(17)/9)
    assert different!=pytest.approx(same)


@pytest.mark.parametrize("sector",["qqq","qqqg"])
@pytest.mark.parametrize("cutoff",[0,1,2])
def test_coupled_mass_hermitian_s3_covariant_lz_and_b_squared(sector,cutoff):
    h=qq_pair_mass_per_gs2(sector,9,.4,cutoff)
    assert maxabs(h-h.getH())<5e-15
    assert maxabs(qq_pair_mass_per_gs2(sector,9,.8,cutoff)-4*h)<5e-15
    colors=1 if sector=="qqq" else 2
    d=2 if sector=="qqq" else 3
    shell=oscillator_shell(d,cutoff)
    lz=np.tile(np.repeat([sum(s[:d])-sum(s[d:]) for s in shell],colors),
               len(longitudinal_basis(sector,9)))
    l=sparse.diags(lz,dtype=float)
    assert maxabs(l@h-h@l)<5e-15
    for p in S3:
        u=sparse.kron(longitudinal_orbital_action(sector,9,cutoff,p),
                      color_permutation_matrix(sector,p),format="csr")
        assert maxabs(u@h-h@u)<8e-15
    # This current has no zero-transfer qq block, including transverse changes.
    block=len(shell)*colors
    for k in range(len(longitudinal_basis(sector,9))):
        assert h[k*block:(k+1)*block,k*block:(k+1)*block].nnz==0


@pytest.mark.parametrize("sector",["qqq","qqqg"])
def test_spin_flavor_total_sign_carrier_is_preserved(sector):
    h=qq_pair_mass_per_gs2(sector,9,.4,1)
    nc=1 if sector=="qqq" else 2
    shape=(h.shape[0]//nc,nc,64)
    rng=np.random.default_rng(117)
    v=rng.normal(size=shape)+1j*rng.normal(size=shape)
    projected=total_sign_project(v,sector,9,1)
    acted=(h@projected.reshape(h.shape[0],64)).reshape(shape)
    np.testing.assert_allclose(total_sign_project(acted,sector,9,1),acted,atol=2e-14)


def test_parent_shell_leakage_is_nonzero_and_bare_contact_not_norm_convergent():
    # One pair and one incoming ground; each additional relative radial state
    # receives the same amplitude. This is a distribution, not a bounded
    # continuum operator on this Gaussian, despite finite Galerkin matrices.
    norms=[]
    for cutoff in (0,2,4,6):
        block=qq_contact_block((9,9,9),(1,17,9),(1,0),.4,0,cutoff)
        norms.append(np.linalg.norm(block[:,0])**2)
    np.testing.assert_allclose(np.array(norms)/norms[0],[1,2,3,4],atol=2e-13)
    h0=qq_pair_mass_per_gs2("qqq",9,.4,0)
    parent=qq_pair_mass_per_gs2("qqq",9,.4,0,2)
    outdim=len(oscillator_shell(2,2))
    np.testing.assert_allclose(parent[::outdim].toarray(),h0.toarray(),atol=5e-15)
    assert np.linalg.norm(parent.toarray())>np.linalg.norm(h0.toarray())


@pytest.mark.parametrize("args",[
    ((9,9,9),(1,9,17),(0,1),.4,0),
    ((9,9,9),(9,9,9),(0,0),.4,0),
    ((9,9,9),(9,9,9),(0,1),0.,0),
    ((9,9,9),(9,9,9),(0,1),.4,-1),
])
def test_invalid_contact_requests_are_rejected(args):
    with pytest.raises((ValueError,TypeError)):
        qq_contact_block(*args)
