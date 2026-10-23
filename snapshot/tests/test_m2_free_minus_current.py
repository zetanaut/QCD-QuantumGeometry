from math import pi
from types import SimpleNamespace
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from deuteron_wigner.bridge.modes.core import ho_momentum,GAMMA,GAMMA_MINUS
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_excluded_fock_vertices import antiquark_spinor
from deuteron_wigner.quantum.m2_coupled_pair_action import circular_to_c45
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_free_minus_current import (
    displaced_intrinsic_moments,projected_free_minus,free_minus_current_between,
)
from deuteron_wigner.quantum.m2_transverse_current_projection import displacement_and_moments


@pytest.mark.parametrize("cutoff",[0,1,2])
@pytest.mark.parametrize("species",["q","qbar"])
def test_projected_minus_against_independent_polar_wave_and_Dirac_quadrature(cutoff,species):
    b=.43;s=np.array([.19,-.27]);w=.7;pp=.61;m=.35
    out=np.array([.12,-.09]);inc=np.array([-.03,.04])
    nodes,weights=hermgauss(16);tx,ty=np.meshgrid(nodes,nodes,indexing="ij")
    x,y=b*tx-s[0]/2,b*ty-s[1]/2
    weight=b*b*np.outer(weights,weights)*np.exp(tx*tx+ty*ty)
    shell=oscillator_shell(1,cutoff)
    def wave(q,x,y):
        n,ell,phase=circular_to_c45(*q)
        return phase*ho_momentum(n,ell,x,y,b)/(2*pi)
    incoming=np.array([wave(q,x,y) for q in shell])
    outgoing=np.array([wave(q,x+s[0],y+s[1]) for q in shell])
    def direct_spin(xx,yy):
        spinor=lf_spinor if species=="q" else antiquark_spinor
        vf=np.column_stack([spinor(pp,*(w*np.array([xx,yy])+out),m,h) for h in (-1,1)])
        vi=np.column_stack([spinor(pp,*(w*np.array([xx,yy])+inc),m,h) for h in (-1,1)])
        return (vf.conj().T@GAMMA[0]@GAMMA_MINUS@vi if species=="q" else
                (vi.conj().T@GAMMA[0]@GAMMA_MINUS@vf).T)/(2*pp)
    spin=np.array([direct_spin(xx,yy) for xx,yy in zip(x.ravel(),y.ravel())]).reshape(16,16,2,2)
    exact=np.einsum("oab,iab,abhk,ab->hkoi",outgoing.conj(),incoming,spin,weight)
    actual=projected_free_minus(1,cutoff,s[None],[w],b_GeV=b,pplus_GeV=pp,
        outgoing_offset=out,incoming_offset=inc,mass_GeV=m,species=species)
    np.testing.assert_allclose(actual,exact,atol=4e-15)


@pytest.mark.parametrize("d",[1,2,3])
def test_complete_shell_linear_and_quadratic_moments_keep_outside_neighbors(d):
    rng=np.random.default_rng(333);s=rng.normal(size=(d,2))*.1;w=rng.normal(size=d);b=.4
    density,first,second=displaced_intrinsic_moments(d,2,s,w,b_GeV=b)
    old,m=displacement_and_moments(d,2,s,b_GeV=b)
    np.testing.assert_allclose(density,old,atol=1e-15)
    np.testing.assert_allclose(first,np.einsum("a,ajoi->joi",w,m),atol=2e-15)
    np.testing.assert_allclose(second[0,1],second[1,0],atol=3e-15)
    _,q,_=displaced_intrinsic_moments(d,2,np.zeros_like(s),w,b_GeV=b)
    assert np.linalg.norm(second[0,0]-density@q[0]@q[0])>1e-3
    # Independent Gaussian ground moments; covariance b²/2 per Cartesian mode.
    shift=w@s
    for j in (0,1):
        for k in (0,1):
            expected=(shift[j]*shift[k]/4+(b*b*np.dot(w,w)/2 if j==k else 0))*density[0,0]
            assert second[j,k,0,0]==pytest.approx(expected,abs=2e-15)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz",[1,-1])
def test_retained_minus_reversal_longitudinal_boost_and_flavor(target,jz):
    from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis
    def endpoint(j):
        bases=tuple(symmetry_basis(s,9,1,target,j,hg) for s,hg in (("qqq",0),("qqqg",-1),("qqqg",1)))
        return SimpleNamespace(total_k2=9,cutoff=1,target=target,jz_twice=j,width_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,bases=bases)
    f,i=endpoint(jz),endpoint(1);delta=np.array([.12,.16])
    def op(f,i,d,p=1.3,charge="electromagnetic"):
        return free_minus_current_between(f,i,d,total_pplus_GeV=p,charge=charge)
    current=op(f,i,delta)
    np.testing.assert_allclose(current,op(i,f,-delta).conj().T,atol=4e-15)
    np.testing.assert_allclose(op(f,i,delta,p=2.6),current/4,atol=2e-15)
    np.testing.assert_allclose((2*op(f,i,delta,charge="up_number")
        -op(f,i,delta,charge="down_number"))/3,current,atol=4e-15)
    assert np.linalg.norm(current)>1e-3


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_free_control_all_four_components_resolve_all_five_structures(target):
    from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
    from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
    from deuteron_wigner.quantum.m2_transverse_current_projection import free_transverse_current_between
    from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
    from deuteron_wigner.quantum.m2_orientation_current_diagnostic import orientation_current_diagnostic
    def endpoint(jz):
        return build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=0.)
    pilots=[endpoint(1),endpoint(-1)]
    e,u=np.linalg.eigh(pilots[0].matrix)
    v=[u[:,0],1j*transverse_parity_map(pilots[1],pilots[0])@u[:,0]]
    delta=(.12,.16);P=1.3;current=np.zeros((4,2,2),complex)
    for hf in range(2):
        for hi in range(2):
            f,i=pilots[hf],pilots[hi]
            matrices=[good_current_density_between(f,i,delta,charge="electromagnetic"),
                free_minus_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic"),
                *free_transverse_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic")]
            current[:,hf,hi]=[np.vdot(v[hf],j@v[hi]) for j in matrices]
    partial=orientation_current_diagnostic(current[0],current[2:],delta,
        mass_GeV=np.sqrt(e[0]),total_pplus_GeV=P)
    full=orientation_current_diagnostic(current[0],current[2:],delta,
        mass_GeV=np.sqrt(e[0]),total_pplus_GeV=P,minus=current[1])
    np.testing.assert_allclose([full.F1_separated,full.F2_separated,full.B1],
        [partial.F1_separated,partial.F2_separated,partial.B1],atol=5e-14)
    np.testing.assert_allclose(full.structure_residual,0.,atol=5e-14)
    assert full.full_five_structure_tested and full.B2 is not None and full.B3 is not None
    assert not full.physical and not full.covariance_repaired and abs(full.B1)>.8
