import numpy as np
import pytest
from scipy import sparse
from scipy.special import roots_hermite

from deuteron_wigner.bridge.modes.core import ho_momentum
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import oscillator_shell
from deuteron_wigner.quantum.m2_coupled_pilot import (
    _displacement, baryon_density, build_coupled_pilot, free_orbital, mass_flow_current,
)


@pytest.fixture(scope="module",params=[0,1])
def pilot(request):
    return build_coupled_pilot(total_k2=9,cutoff=request.param,target="PROTON",jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)


def test_symmetry_isometries_and_hermitian_partial_hamiltonian(pilot):
    expected=[10,7,20] if pilot.cutoff==0 else [37,67,100]
    assert [u.shape[1] for u in pilot.bases]==expected
    for u in pilot.bases:
        np.testing.assert_allclose((u.getH()@u).toarray(),np.eye(u.shape[1]),atol=5e-14)
    np.testing.assert_allclose(pilot.matrix,pilot.matrix.conj().T,atol=3e-14)
    np.testing.assert_allclose(pilot.matrix,pilot.h0+.3*pilot.emission_per_gs+
                               .09*(pilot.cross_pairs_per_gs2+
                                    pilot.instantaneous_fermion_per_gs2),atol=3e-14)
    assert len(pilot.omitted_terms)>=4 and pilot.physical is False


def test_mass_squared_free_matrix_against_cartesian_quadrature():
    shell=oscillator_shell(1,4)
    t,w=roots_hermite(7)
    x,y=np.meshgrid(.4*t,.4*t,indexing="ij")
    weights=.16*np.outer(w,w)*np.exp((x*x+y*y)/.16)
    modes=[]
    for p,m in shell:
        n=min(p,m)
        modes.append((-1)**n*ho_momentum(n,p-m,x,y,.4).ravel()/(2*np.pi))
    phi=np.array(modes)
    expected=(phi.conj()*(weights*(x*x+y*y)).ravel())@phi.T
    np.testing.assert_allclose(free_orbital(1,4,.4).toarray(),expected,atol=5e-15)


def test_displacement_density_direct_momentum_recoil_integral():
    b=.4; delta=np.array([.12,.16])
    t,w=roots_hermite(7)
    x,y=np.meshgrid(b*t+delta[0]/2,b*t+delta[1]/2,indexing="ij")
    weights=b*b*np.outer(w,w)*np.exp(((x-delta[0]/2)**2+(y-delta[1]/2)**2)/b**2)
    shell=oscillator_shell(1,2)
    alpha=(delta[0]-1j*delta[1])/(2*b)
    for bra in shell:
        for ket in shell:
            a=(-1)**min(bra)*ho_momentum(min(bra),bra[0]-bra[1],x,y,b)/(2*np.pi)
            c=(-1)**min(ket)*ho_momentum(min(ket),ket[0]-ket[1],x-delta[0],y-delta[1],b)/(2*np.pi)
            expected=np.sum(weights*a.conj()*c)
            result=_displacement(bra[0],ket[0],alpha)*_displacement(bra[1],ket[1],alpha.conjugate())
            assert result==pytest.approx(expected,abs=4e-15)


def test_baryon_charge_reversal_and_nonzero_continuity_current(pilot):
    np.testing.assert_allclose(baryon_density(pilot,(0.,0.)),np.eye(len(pilot.matrix)),atol=5e-14)
    r,jx,jy=mass_flow_current(pilot,(.12,.16))
    rev,rx,ry=mass_flow_current(pilot,(-.12,-.16))
    np.testing.assert_allclose(rev,r.conj().T,atol=3e-14)
    np.testing.assert_allclose(rx,jx.conj().T,atol=3e-14)
    np.testing.assert_allclose(ry,jy.conj().T,atol=3e-14)
    np.testing.assert_allclose(.12*jx+.16*jy,pilot.matrix@r-r@pilot.matrix,atol=3e-14)
    nq=pilot.bases[0].shape[1]
    assert np.linalg.norm(jx[nq:,:nq])>1e-4
    assert np.linalg.norm(jx[:nq,:nq])>1e-6  # new longitudinal pair current too


def test_projected_spectra_target_and_helicity_degenerate():
    spectra=[]
    for target,jz in (("PROTON",1),("PROTON",-1),("NEUTRON",1)):
        p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
        spectra.append(np.linalg.eigvalsh(p.matrix))
    np.testing.assert_allclose(spectra[0],spectra[1],atol=3e-14)
    np.testing.assert_allclose(spectra[0],spectra[2],atol=3e-14)


def test_controlled_zero_coupling():
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=0.)
    np.testing.assert_array_equal(p.matrix,p.h0)
    assert np.linalg.norm(mass_flow_current(p,(.12,.16))[1])<2e-13
