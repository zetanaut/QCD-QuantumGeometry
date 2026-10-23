"""Continuum quadrature, Dirac spin kernels and explicit Galerkin Ward boundary."""
from functools import lru_cache
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy import sparse

from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_constrained_fermion_current import free_fermion_current
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis,free_sector,free_orbital
from deuteron_wigner.quantum.m2_transverse_current_projection import (
    displacement_and_moments,free_transverse_current_between,
)


@pytest.mark.parametrize("cutoff",range(4))
@pytest.mark.parametrize("shift",[(0.,0.),(.19,-.27)])
def test_density_and_momentum_against_independent_polar_wave_quadrature(cutoff,shift):
    b=.43;s=np.array(shift)
    nodes,weights=hermgauss(20)
    tx,ty=np.meshgrid(nodes,nodes,indexing="ij")
    x,y=b*tx-s[0]/2,b*ty-s[1]/2
    weight=b*b*np.outer(weights,weights)*np.exp(tx*tx+ty*ty)
    shell=oscillator_shell(1,cutoff)
    def wave(q,x,y):
        n,m,phase=circular_to_c45(*q)
        return phase*ho_momentum(n,m,x,y,b)/(2*pi)
    incoming=np.array([wave(q,x,y) for q in shell])
    outgoing=np.array([wave(q,x+s[0],y+s[1]) for q in shell])
    density=np.einsum("oab,iab,ab->oi",outgoing.conj(),incoming,weight)
    moments=np.array([np.einsum("oab,iab,ab->oi",outgoing.conj(),incoming,weight*z)
                      for z in (x,y)])
    actual_d,actual_m=displacement_and_moments(1,cutoff,s[None,:],b_GeV=b)
    np.testing.assert_allclose(actual_d,density,atol=2e-15)
    np.testing.assert_allclose(actual_m[0],moments,atol=2e-15)


@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("cutoff",[0,1,2])
def test_exact_momentum_uses_outside_ladder_and_has_correct_midpoint_adjoint(dimension,cutoff):
    rng=np.random.default_rng(24)
    s=rng.normal(size=(dimension,2))*.2
    density,moments=displacement_and_moments(dimension,cutoff,s,b_GeV=.4)
    backward,reverse_m=displacement_and_moments(dimension,cutoff,-s,b_GeV=.4)
    np.testing.assert_allclose(density,backward.conj().T,atol=1e-15)
    midpoint=moments+s[:,:,None,None]*density/2
    reverse_mid=reverse_m-s[:,:,None,None]*backward/2
    np.testing.assert_allclose(midpoint,reverse_mid.conj().swapaxes(-1,-2),atol=1e-15)
    _,q=displacement_and_moments(dimension,cutoff,np.zeros_like(s),b_GeV=.4)
    assert np.linalg.norm(moments-density@q)>1e-3
    np.testing.assert_allclose(moments[:,:,0,0],-s*density[0,0]/2,atol=1e-15)


@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("cutoff",[0,1,2])
def test_finite_free_ward_defect_equals_independent_nplus2_boundary(dimension,cutoff):
    s=np.random.default_rng(7).normal(size=(dimension,2))*.13
    d,m=displacement_and_moments(dimension,cutoff,s,b_GeV=.4)
    h=free_orbital(dimension,cutoff,.4).toarray()
    full_d,_=displacement_and_moments(dimension,cutoff+2,s,b_GeV=.4)
    full_h=free_orbital(dimension,cutoff+2,.4).toarray()
    n=len(d)
    continuum=2*np.einsum("aj,ajoi->oi",s,m)+np.sum(s*s)*d
    defect=h@d-d@h-continuum
    boundary=-full_h[:n,n:]@full_d[n:,:n]+full_d[:n,n:]@full_h[n:,:n]
    np.testing.assert_allclose(defect,boundary,atol=2e-15)
    if cutoff:
        assert np.linalg.norm(defect)>1e-3


@pytest.mark.parametrize("species",["q","qbar"])
def test_equal_plus_analytic_transverse_kernel_matches_full_constraint(species):
    pf=np.array([.7,.31,-.24]);pi_=np.array([.7,-.12,.18])
    delta=pf[1:]-pi_[1:]
    expected=np.array([np.diag((pf[j+1]+pi_[j+1]
        +1j*np.array([-1,1])*np.array([-delta[1],delta[0]])[j])/(2*pf[0]))
        for j in range(2)])
    for mass in (0.,.35,1.2):
        np.testing.assert_allclose(free_fermion_current(pf,pi_,mass_GeV=mass,species=species)[2:],
                                   expected,atol=1e-15)


@lru_cache(maxsize=8)
def endpoint(cutoff,target,jz):
    """Actual symmetry basis; no need to build interactions for a free test."""
    labels=(("qqq",0),("qqqg",-1),("qqqg",1))
    bases=tuple(symmetry_basis(s,9,cutoff,target,jz,hg) for s,hg in labels)
    return SimpleNamespace(total_k2=9,cutoff=cutoff,target=target,jz_twice=jz,
        width_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,bases=bases)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("final_jz",[1,-1])
def test_projected_transverse_reversal_boost_and_flavor(target,final_jz):
    pi_=endpoint(1,target,1);pf=endpoint(1,target,final_jz)
    delta=np.array([.12,.16]);mean=np.array([.03,-.07]);pplus=1.6
    def j(out,inc,d,charge="electromagnetic",mean=(0.,0.),p=pplus):
        return free_transverse_current_between(out,inc,d,total_pplus_GeV=p,
            charge=charge,mean_total_transverse=mean)
    base=j(pf,pi_,delta)
    np.testing.assert_allclose(base,j(pi_,pf,-delta).conj().swapaxes(1,2),atol=3e-15)
    rho=good_current_density_between(pf,pi_,delta,charge="electromagnetic")
    np.testing.assert_allclose(j(pf,pi_,delta,mean=mean)-base,
                               mean[:,None,None]*rho/pplus,atol=3e-15)
    np.testing.assert_allclose(j(pf,pi_,delta,p=2*pplus),base/2,atol=2e-15)
    np.testing.assert_allclose((2*j(pf,pi_,delta,"up_number")-j(pf,pi_,delta,"down_number"))/3,
                               base,atol=3e-15)
    if target=="PROTON" and final_jz==1:
        # At N=1 fixed spin and Jz force fixed orbital excitation: this
        # particular reduced block hides the raw-shell Ward boundary.
        blocks=[]
        for s,u in zip(("qqq","qqqg","qqqg"),pi_.bases):
            blocks.append((u.getH()@free_sector(s,9,1,.4,.35,0.)@u).toarray())
        h=sparse.block_diag(blocks).toarray()
        defect=h@rho-rho@h-2*pplus*np.einsum("j,joi->oi",delta,base)
        np.testing.assert_allclose(defect,0.,atol=3e-14)


def test_next_shell_exposes_projected_boundary_hidden_at_cutoff_one():
    p=endpoint(2,"PROTON",1);delta=np.array([.12,.16]);pplus=1.6
    rho=good_current_density_between(p,p,delta,charge="electromagnetic")
    current=free_transverse_current_between(p,p,delta,total_pplus_GeV=pplus,
                                            charge="electromagnetic")
    h=sparse.block_diag([(u.getH()@free_sector(s,9,2,.4,.35,0.)@u).toarray()
                         for s,u in zip(("qqq","qqqg","qqqg"),p.bases)]).toarray()
    defect=h@rho-rho@h-2*pplus*np.einsum("j,joi->oi",delta,current)
    assert np.linalg.norm(defect)>1e-3


def test_invalid_projection_parameters_are_rejected():
    with pytest.raises(ValueError):
        displacement_and_moments(2,1,[[0,0]],b_GeV=.4)
    with pytest.raises(ValueError):
        displacement_and_moments(1,0,[[0,0]],b_GeV=0.)
    p=endpoint(0,"PROTON",1)
    with pytest.raises(ValueError):
        free_transverse_current_between(p,p,[0,0],total_pplus_GeV=0.,charge="baryon")
    with pytest.raises(ValueError):
        free_transverse_current_between(p,endpoint(0,"NEUTRON",1),[0,0],
                                        total_pplus_GeV=1.,charge="baryon")
