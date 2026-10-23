"""Source normalization, conserved momenta, boosts and coherent emission."""
from fractions import Fraction
from itertools import product
from math import pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad

from deuteron_wigner.bridge.modes.core import (
    GAMMA, GAMMA_PLUS, LAMBDA_PLUS, cartesian_momentum, ho_momentum, slash, spinor,
)
from deuteron_wigner.bridge.vsrc.core import finite_box_pminus_kernel
from deuteron_wigner.quantum.m2_canonical_emission import (
    apply_canonical_emission, emission_mass_kernel_per_gs, emission_mode_per_gs,
    emission_momenta, lf_spinor,
)
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import (
    emission_routes, longitudinal_basis, oscillator_shell, total_sign_project,
)
from deuteron_wigner.quantum.m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


@pytest.mark.parametrize("pp,px,py,mass",[
    (.2,.1,-.3,.35),(3.,-.2,.7,1.2),(1.3,.4,.2,0.),
])
def test_lf_spinors_dirac_completeness_and_good_spin(pp,px,py,mass):
    pm=(mass*mass+px*px+py*py)/(2*pp)
    p=np.array([(pp+pm)/sqrt(2),px,py,(pp-pm)/sqrt(2)])
    u=np.column_stack([lf_spinor(pp,px,py,mass,h) for h in (-1,1)])
    np.testing.assert_allclose((slash(p)-mass*np.eye(4))@u,0,atol=4e-15)
    np.testing.assert_allclose(u.conj().T@GAMMA[0]@u,2*mass*np.eye(2),atol=4e-15)
    np.testing.assert_allclose(u@u.conj().T@GAMMA[0],slash(p)+mass*np.eye(4),atol=5e-15)
    sigma3=1j*GAMMA[1]@GAMMA[2]
    good=LAMBDA_PLUS@u
    np.testing.assert_allclose(sigma3@good,good@np.diag([-1,1]),atol=4e-15)


def test_canonical_field_factors_give_unit_good_current_between_different_momenta():
    a=(.7,.21,-.1); b=(2.1,-.5,.4)
    u=np.column_stack([lf_spinor(*a,.35,h)/sqrt(2*a[0]) for h in (-1,1)])
    v=np.column_stack([lf_spinor(*b,.35,h)/sqrt(2*b[0]) for h in (-1,1)])
    np.testing.assert_allclose(v.conj().T@GAMMA[0]@GAMMA_PLUS@u,np.eye(2),atol=4e-15)
    # Instant-spin labels require a momentum-dependent two-state rotation.
    inst=np.column_stack([spinor(*a,.35,h) for h in (-1,1)])
    front=np.column_stack([lf_spinor(*a,.35,h) for h in (-1,1)])
    rotation=inst.conj().T@GAMMA[0]@front/(2*.35)
    np.testing.assert_allclose(rotation.conj().T@rotation,np.eye(2),atol=4e-15)
    np.testing.assert_allclose(inst@rotation,front,atol=4e-15)
    assert abs(rotation[0,1])>.05


def test_all_k27_emission_routes_conserve_physical_plus_and_transverse():
    for route in emission_routes(27):
        pq,pg,p=map(np.array,emission_momenta(route,(.23,-.16),3.,(.13,.27)))
        np.testing.assert_allclose(pq+pg,p,atol=5e-16)
        xq=route.outgoing[route.emitter]/27
        xg=route.outgoing[3]/27
        assert np.linalg.norm(pq[1:]-(xq/(xq+xg))*p[1:])==pytest.approx(
            sqrt(xq*xg/(xq+xg))*np.hypot(.23,.16))


def test_historical_c50_witness_is_not_a_lawful_physical_momentum_inverse():
    # Document why the OLD numerical oracle cannot normalize the new source.
    r=finite_box_pminus_kernel(kq=Fraction(5,2),kg=Fraction(2),
        K=Fraction(9,2),qrel=(.2,-.1),mass=1.2,h_out=1,h_in=1,h_g=1)
    k=r["kinematics"]
    defect=np.array(k["pout"])+np.array(k["gluon"])-np.array(k["pin"])
    assert np.linalg.norm(defect[1:])>.01
    assert defect[0]==pytest.approx(0)


@pytest.mark.parametrize("h_out,h_in,h_g",list(product((-1,1),repeat=3)))
def test_canonical_kernel_boost_invariance_and_exact_linear_polynomial(h_out,h_in,h_g):
    route=emission_routes(27)[391]
    args=(route,(.23,-.16),.35,h_out,h_in,h_g)
    value=emission_mass_kernel_per_gs(*args,total_pplus=3.)
    for pp in (.3,1.,9.):
        for pt in ((0.,0.),(.31,-.22)):
            assert emission_mass_kernel_per_gs(*args,total_pplus=pp,parent_transverse=pt
                )==pytest.approx(value,abs=2e-15)
    f=lambda xy:emission_mass_kernel_per_gs(route,xy,.35,h_out,h_in,h_g)
    assert value==pytest.approx(f((0.,0.))+.23*(f((1.,0.))-f((0.,0.)))
                               -.16*(f((0.,1.))-f((0.,0.))),abs=2e-15)


@pytest.mark.parametrize("h_out,h_in,h_g",list(product((-1,1),repeat=3)))
def test_all_low_modes_against_direct_spinor_radial_angular_integration(h_out,h_in,h_g):
    route=emission_routes(9)[8]
    b=.4; mass=.35
    theta=np.arange(16)*2*pi/16
    i=route.emitter
    xq,xg=route.outgoing[i]/9,route.outgoing[3]/9
    jac=sqrt(xq*xg/(xq+xg))
    for nplus,nminus in ((0,0),(1,0),(0,1),(1,1),(2,0)):
        n=min(nplus,nminus); m=nplus-nminus
        def radial(r):
            xy=np.column_stack((r*np.cos(theta),r*np.sin(theta)))
            f=np.array([emission_mass_kernel_per_gs(route,q,mass,h_out,h_in,h_g) for q in xy])
            phi=(-1)**n*ho_momentum(n,m,xy[:,0],xy[:,1],b)
            return jac*r/(2*pi)*np.mean(phi.conj()*f)
        expected=quad(lambda r:radial(r).real,0,np.inf,epsabs=2e-12)[0]+1j*quad(
            lambda r:radial(r).imag,0,np.inf,epsabs=2e-12)[0]
        actual=emission_mode_per_gs(route,nplus,nminus,b,mass,h_out,h_in,h_g)
        assert actual==pytest.approx(expected,abs=3e-12)


@pytest.mark.parametrize("hg",[-1,1])
def test_coherent_emission_preserves_total_sign_and_jz_flavor(hg):
    total=9; cutoff=1
    nq=len(longitudinal_basis("qqq",total))*len(oscillator_shell(2,cutoff))
    rng=np.random.default_rng(53)
    source=rng.normal(size=(nq,1,64))+1j*rng.normal(size=(nq,1,64))
    sf=np.array(SPIN_FLAVOR_BASIS)
    sz=(2*(sf%2)-1).sum(axis=1)
    iz=np.where(sf<2,1,-1).sum(axis=1)
    angular=np.tile([sum(s[:2])-sum(s[2:]) for s in oscillator_shell(2,cutoff)],
                     len(longitudinal_basis("qqq",total)))
    mask=(2*angular[:,None]+sz[None,:]==1)&(iz[None,:]==1)
    source*=mask[:,None,:]
    source=total_sign_project(source,"qqq",total,cutoff)
    out=apply_canonical_emission(source,total,cutoff,.4,.35,hg)
    assert np.linalg.norm(out)>1e-4
    np.testing.assert_allclose(total_sign_project(out,"qqqg",total,cutoff),out,atol=2e-14)
    angles=np.tile([sum(s[:3])-sum(s[3:]) for s in oscillator_shell(3,cutoff)],
                  len(longitudinal_basis("qqqg",total)))
    forbidden=(2*angles[:,None]+sz[None,:]+2*hg!=1)|(iz[None,:]!=1)
    assert np.max(np.abs(out*forbidden[:,None,:]))<2e-15


def test_full_nucleon_mass_kernel_has_no_double_parent_conversion():
    route=emission_routes(27)[300]
    value=emission_mode_per_gs(route,1,0,.4,.35,1,1,-1)
    scaled=emission_mode_per_gs(route,1,0,.8,.7,1,1,-1)
    assert abs(value)>1e-6
    assert scaled==pytest.approx(4*value,abs=2e-14)
