"""Independent source, statistics, color and intermediate-space checks."""
from itertools import permutations, product
from math import factorial, pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad

from deuteron_wigner.bridge.modes.core import (
    GAMMA, LAMBDA_PLUS, ho_momentum, lf_dot, polarization, slash,
)
from deuteron_wigner.quantum.m2_canonical_emission import (
    emission_mass_kernel_per_gs, emission_mode_per_gs,
)
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    EmissionRoute, jacobi_frame, longitudinal_basis,
)
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_kinematics import (
    SPECIES, SplitRoute, color_basis, color_generator_residual, color_permutation,
    color_source, frame, longitudinal_states, permutation_frame, split_frame, split_routes,
)
from deuteron_wigner.quantum.m2_excluded_fock_vertices import (
    antiquark_spinor, cubic_mass_kernel, cubic_mode, daughter_momenta,
    three_gluon_numerator,
)
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source

CHANNELS = ("q_to_qg","g_to_gg","g_to_pair")

@pytest.fixture(scope="module")
def pilot():
    return build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)

@pytest.fixture(scope="module",params=("qqqgg","qqqq-qbar"))
def source(request,pilot):
    return build_excluded_source(pilot,request.param,out_cutoff=1)

@pytest.mark.parametrize("total",[9,11])
def test_counts_frames_and_all_routings(total):
    assert longitudinal_states("qqqg",total)==longitudinal_basis("qqqg",total)
    for s in longitudinal_states("qqqg",total):
        np.testing.assert_allclose(frame("qqqg",s).forward,jacobi_frame(s).forward,atol=8e-16)
    for sector,number in (("qqqgg",5 if total==9 else 15),("qqqq-qbar",15 if total==9 else 35)):
        assert len(longitudinal_states(sector,total))==number
        for s in longitudinal_states(sector,total):
            f=frame(sector,s)
            assert sum(s)==total
            assert all(v%2==int(kind!="g") for v,kind in zip(s,SPECIES[sector]))
            np.testing.assert_allclose(f.forward@f.inverse,np.eye(5),atol=1e-15)
            np.testing.assert_allclose(f.inverse.T@np.diag(1/f.fractions)@f.inverse,
                                       np.eye(5),atol=1e-15)
    for channel in CHANNELS:
        for r in split_routes(total,channel):
            rotation=split_frame(r)
            np.testing.assert_allclose(rotation@rotation.T,np.eye(4),atol=1e-15)
            a,b,p=map(np.array,daughter_momenta(r,(.23,-.16),2.7,(.31,-.22)))
            np.testing.assert_allclose(a+b,p,atol=5e-16)

@pytest.mark.parametrize("sector,rank",[("qqqgg",6),("qqqq-qbar",3)])
def test_complete_color_singlets_and_generators(sector,rank):
    basis=color_basis(sector)
    assert basis.shape[1]==rank
    np.testing.assert_allclose(basis.conj().T@basis,np.eye(rank),atol=8e-15)
    assert color_generator_residual(sector)<8e-15

@pytest.mark.parametrize("channel,emitter,casimir",[
    ("q_to_qg",0,4/3),("q_to_qg",1,4/3),("q_to_qg",2,4/3),
    ("g_to_gg",3,3),("g_to_pair",3,.5),
])
def test_local_source_color_normalization(channel,emitter,casimir):
    c=color_source(channel,emitter)
    np.testing.assert_allclose(c.conj().T@c,casimir*np.eye(2),atol=8e-15)

def test_gluon_color_antisymmetry():
    c=color_source("g_to_gg")
    np.testing.assert_allclose(color_permutation("qqqgg",(0,1,2,4,3))@c,-c,atol=8e-15)

@pytest.mark.parametrize("pp,px,py,m",[(.2,.1,-.3,.35),(3.,-.2,.7,1.2),(1.3,.4,.2,0.)])
def test_antiquark_dirac_completeness_normalization_and_physical_spin(pp,px,py,m):
    pm=(m*m+px*px+py*py)/(2*pp)
    p=np.array([(pp+pm)/sqrt(2),px,py,(pp-pm)/sqrt(2)])
    v=np.column_stack([antiquark_spinor(pp,px,py,m,h) for h in (-1,1)])
    np.testing.assert_allclose((slash(p)+m*np.eye(4))@v,0,atol=5e-15)
    np.testing.assert_allclose(v.conj().T@GAMMA[0]@v,-2*m*np.eye(2),atol=5e-15)
    np.testing.assert_allclose(v@v.conj().T@GAMMA[0],slash(p)-m*np.eye(4),atol=6e-15)
    # Antiparticle physical helicity is NOT its spinor good-component eigenvalue.
    good=LAMBDA_PLUS@v
    np.testing.assert_allclose(1j*GAMMA[1]@GAMMA[2]@good,good@np.diag([1,-1]),atol=4e-15)

@pytest.mark.parametrize("channel",CHANNELS)
@pytest.mark.parametrize("h",list(product((-1,1),repeat=3)))
def test_local_kernel_boost_invariance_linearity_and_dimensions(channel,h):
    r=split_routes(11,channel)[-2]
    f=lambda q,**kw:cubic_mass_kernel(r,q,.35,h,**kw)
    value=f((.23,-.16))
    for pp,pt in ((.3,(0.,0.)),(9.,(.31,-.22))):
        assert f((.23,-.16),total_pplus=pp,parent_transverse=pt)==pytest.approx(value,abs=8e-15)
    assert f((.23,-.16))==pytest.approx(f((0.,0.))+.23*(f((1.,0.))-f((0.,0.)))
                                      -.16*(f((0.,1.))-f((0.,0.))),abs=8e-15)
    for np_,nm in ((0,0),(1,0),(0,1)):
        assert cubic_mode(r,np_,nm,.8,.7,h)==pytest.approx(4*cubic_mode(r,np_,nm,.4,.35,h),abs=8e-15)

@pytest.mark.parametrize("h",list(product((-1,1),repeat=3)))
def test_emission_same_active_momenta_as_independent_existing_three_quark_source(h):
    r=SplitRoute("q_to_qg",(5,1,1,2),(3,1,1,2,2),0)
    old=EmissionRoute((5,3,1),0,(3,3,1,2))  # same active pair; spectators grouped
    h1,h2,h0=h
    assert cubic_mass_kernel(r,(.23,-.16),.35,h)==pytest.approx(
        emission_mass_kernel_per_gs(old,(.23,-.16),.35,h1,h0,h2),abs=3e-15)
    for np_,nm in ((0,0),(1,0),(0,1),(1,1)):
        assert cubic_mode(r,np_,nm,.4,.35,h)==pytest.approx(
            emission_mode_per_gs(old,np_,nm,.4,.35,h1,h0,h2),abs=3e-15)

@pytest.mark.parametrize("h",list(product((-1,1),repeat=3)))
def test_three_gluon_six_derivative_contractions_and_daughter_exchange(h):
    r=split_routes(11,"g_to_gg")[-2]
    left,right,parent=daughter_momenta(r,(.23,-.16),3.,(.31,-.22))
    h1,h2,h0=h
    eps=[polarization(*parent,h0),polarization(*left,h1).conj(),polarization(*right,h2).conj()]
    def lf(p):
        pp,x,y=p
        return np.array([pp,(x*x+y*y)/(2*pp),x,y])
    # g f^{abc} (partial_mu A_nu^a) A^{b mu} A^{c nu}; outgoing phases +i k.x.
    k=[-lf(parent),lf(left),lf(right)]
    direct=0j
    for p in permutations(range(3)):
        sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        a,b,c=p
        direct+=1j*sign*lf_dot(k[a],eps[b])*lf_dot(eps[a],eps[c])
    value=three_gluon_numerator(left,right,parent,h1,h2,h0)
    assert value==pytest.approx(direct,abs=5e-15)
    assert three_gluon_numerator(right,left,parent,h2,h1,h0)==pytest.approx(-value,abs=5e-15)

@pytest.mark.parametrize("channel,h",[("g_to_gg",(1,-1,1)),("g_to_pair",(1,1,1)),
    ("g_to_pair",(1,-1,-1))])
def test_new_modes_against_direct_radial_angular_integration(channel,h):
    route=split_routes(9,channel)[-1]
    theta=np.arange(12)*2*pi/12
    a,c=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
    jac=sqrt(a*c/(a+c))
    for np_,nm in ((0,0),(1,0),(0,1),(1,1),(2,0)):
        n,m=min(np_,nm),np_-nm
        def radial(r):
            xy=np.column_stack((r*np.cos(theta),r*np.sin(theta)))
            f=np.array([cubic_mass_kernel(route,q,.35,h) for q in xy])
            phi=(-1)**n*ho_momentum(n,m,xy[:,0],xy[:,1],.4)
            return jac*r/(2*pi)*np.mean(phi.conj()*f)
        expected=quad(lambda r:radial(r).real,0,np.inf,epsabs=2e-12)[0]+1j*quad(
            lambda r:radial(r).imag,0,np.inf,epsabs=2e-12)[0]
        assert cubic_mode(route,np_,nm,.4,.35,h)==pytest.approx(expected,abs=3e-12)

def test_ordered_slot_creation_factors_against_normalized_fock_states():
    # Three-fermion Slater; append an orthogonal fourth mode.
    a=np.zeros((4,)*4)
    for p in permutations(range(3)):
        sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        a[p+(3,)]=sign/sqrt(6)
    out=np.zeros_like(a)
    for p in permutations(range(4)):
        sign=(-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))
        out+=2*sign*a.transpose(p)/factorial(4)
    assert np.linalg.norm(out)==pytest.approx(1)
    assert out[0,1,2,3]==pytest.approx(1/sqrt(24))
    # a†_j |i>: distinct modes have norm 1; repeated mode has norm sqrt(2).
    for i,j,expected in ((0,1,1),(0,0,sqrt(2))):
        raw=np.zeros((2,2)); raw[i,j]=1
        assert np.linalg.norm((raw+raw.T)/sqrt(2))==pytest.approx(expected)
    # (1/2) M_ij a†_i a†_j: distinguished symmetric M, ket tensor M/sqrt(2).
    m=np.array([[2.,3.],[3.,5.]])
    occupation=np.array([m[0,0]/sqrt(2),m[0,1],m[1,1]/sqrt(2)])
    assert np.linalg.norm(m/sqrt(2))==pytest.approx(np.linalg.norm(occupation))

def test_many_body_statistics_and_free_operator_covariance(source):
    s=source.space
    v=source.matrix_per_gs
    energy=s.free_energies(.4,.35,0.)
    np.testing.assert_allclose(s.statistics_project(v),v,atol=2e-14)
    # Adjacent generators certify the full quark permutation group.
    for i in range(2 if s.sector=="qqqgg" else 3):
        p=list(range(5)); p[i],p[i+1]=p[i+1],p[i]
        pv=s.permute(v,tuple(p))
        np.testing.assert_allclose(pv,-v,atol=2e-14)
        np.testing.assert_allclose(s.permute(energy[:,None]*v,tuple(p)),energy[:,None]*pv,atol=5e-14)
    assert len(source.retained_columns)==27
    assert source.retained_columns[0]==10  # no qqq -> retained qqqg counted in Q
    assert np.linalg.norm(v)>.3

def test_propagating_self_energy_sign_norm_and_energy_derivative(source):
    e=source.space.free_energies(.4,.35,0.)
    sigma=source.self_energy(1.4,e)
    norm=source.norm_correction(1.4,e)
    np.testing.assert_allclose(sigma,sigma.conj().T,atol=3e-17)
    assert np.linalg.eigvalsh(sigma)[-1]<1e-15
    assert np.linalg.eigvalsh(norm)[0]>-1e-15
    derivative=(source.self_energy(1.4+1e-5,e)-source.self_energy(1.4-1e-5,e))/2e-5
    np.testing.assert_allclose(derivative,-norm,atol=2e-12)
    with pytest.raises(ValueError,match="pole"):
        source.self_energy(e[0],e)


def test_proton_source_is_highest_weight_isospin_half_not_just_fixed_iz(source):
    s=source.space
    v=source.matrix_per_gs
    # On Iz=+1/2, I_+ annihilation proves I=1/2. Antiquark generator is -T*.
    for li in range(len(s.longitudinal)):
        rows=np.flatnonzero(s.coordinates[:,0]==li)
        orb,col,spin=s.coordinates[rows,1:].T
        tensor=np.zeros((len(s.orbitals),s.ncolor,s.nspin,v.shape[1]),complex)
        tensor[orb,col,spin]=v[rows]
        raised=np.zeros_like(tensor)
        for particle,species in enumerate(SPECIES[s.sector]):
            if species=="g":
                continue
            selected=np.flatnonzero(s.spins[:,particle]>=2 if species=="q" else s.spins[:,particle]<2)
            changed=s.spins[selected].copy()
            changed[:,particle]+=(-2 if species=="q" else 2)
            target=np.ravel_multi_index(changed.T,s.spin_dims)
            raised[:,:,target]+=(1 if species=="q" else -1)*tensor[:,:,selected]
        assert np.linalg.norm(raised)<5e-14

@pytest.mark.parametrize("bad",[-1,.5,True])
def test_noninteger_negative_modes_rejected(bad):
    with pytest.raises((ValueError,TypeError)):
        cubic_mode(split_routes(9,"g_to_gg")[0],bad,0,.4,.35,(1,1,1))

def test_invalid_mass_rejected_even_for_angular_zero():
    with pytest.raises(ValueError):
        cubic_mode(split_routes(9,"g_to_gg")[0],0,0,.4,-1.,(1,1,1))


@pytest.mark.parametrize("channel,h,m",[("g_to_gg",(1,-1,1),1),
    ("g_to_pair",(1,-1,1),1),("g_to_pair",(1,1,1),0),("q_to_qg",(-1,1,1),0)])
def test_bare_radial_tails_do_not_give_a_finite_norm_continuum_source(channel,h,m):
    route=split_routes(9,channel)[-1]
    base=cubic_mode(route,m,0,.4,.35,h)
    assert abs(base)>1e-3
    for n in range(6):
        expected=base*(sqrt(n+1) if m else 1.)
        assert cubic_mode(route,n+m,n,.4,.35,h)==pytest.approx(expected,abs=5e-14)
    # Thus sum |c_n|^2 grows as R+1 (m=0) or (R+1)(R+2)/2 (|m|=1).
    # This is a source-norm witness, not a computed renormalized loop divergence.
