"""Independent eigenspace perturbation of the finite field envelope."""
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import (
    SPECTRAL_PRESCRIPTION,finite_cell_envelope_spectrum,finite_cell_envelope_minus_tangent)


def args(species="q",count=3):
    return dict(regulator=GaussianClusterRegulator(width_GeV=.8,boson_modes=count,
        fermion_modes=count),species=species,total_k2=9,total_pplus_GeV=1.3,
        prescription=SPECTRAL_PRESCRIPTION)


def spectral_projector(matrix,edges):
    p,u = np.linalg.eigh(matrix)
    if len(edges)==2:
        keep = (p>edges[0])&(p<edges[1])
    else:
        keep = ((p>edges[0])&(p<edges[1]))|((p>edges[2])&(p<edges[3]))
    return (u*keep)@u.conj().T


@pytest.mark.parametrize("species",["q","qbar"])
@pytest.mark.parametrize("count",[1,3])
def test_envelope_tangent_matches_independent_covariant_spectral_variation(species,count):
    modes = np.arange(-11,12,2)
    p,member,edges = finite_cell_envelope_spectrum(modes,**args(species,count))
    rng = np.random.default_rng(317)
    a = rng.normal(size=(len(p),len(p)))+1j*rng.normal(size=(len(p),len(p)))
    a = (a+a.conj().T)/15
    charge = 2/3 if species=="q" else -2/3
    result = finite_cell_envelope_minus_tangent(modes,a,charge=charge,**args(species,count))
    eps = 2e-5
    values = {k:spectral_projector(np.diag(p)-k*eps*charge*a,edges) for k in (-2,-1,1,2)}
    derivative = (values[-2]-8*values[-1]+8*values[1]-values[2])/(12*eps)
    np.testing.assert_allclose(result,derivative,atol=3e-10,rtol=0.)
    np.testing.assert_allclose(result,result.conj().T,atol=2e-15)
    projector = np.diag(member)
    np.testing.assert_allclose(result,projector@result+result@projector,atol=2e-15)
    np.testing.assert_allclose((p[:,None]-p[None,:])*result,
        -charge*(projector@a-a@projector),atol=2e-15)
    assert np.linalg.norm(result)>1e-1
    # Any cut positions in the same spectral gaps give the same derivative.
    shifted_edges = edges+np.array([.03,-.02])
    values = {k:spectral_projector(np.diag(p)-k*eps*charge*a,shifted_edges)
              for k in (-2,-1,1,2)}
    moved = (values[-2]-8*values[-1]+8*values[1]-values[2])/(12*eps)
    np.testing.assert_allclose(moved,result,atol=3e-10,rtol=0.)


@pytest.mark.parametrize("species",["q","qbar","g"])
@pytest.mark.parametrize("count",[1,3])
def test_zero_plus_transfer_envelope_response_is_exactly_zero_with_transverse_multiplicity(species,count):
    labels = np.arange(-10,11,2) if species=="g" else np.arange(-11,12,2)
    modes = np.repeat(labels,3)
    p,member,edges = finite_cell_envelope_spectrum(modes,**args(species,count))
    rng = np.random.default_rng(523)
    probe = np.zeros((len(modes),)*2,complex)
    for k in range(len(labels)):
        b = rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
        probe[3*k:3*k+3,3*k:3*k+3] = (b+b.conj().T)/10
    charge = 0 if species=="g" else (-1/3 if species=="qbar" else 1/3)
    current = finite_cell_envelope_minus_tangent(modes,probe,charge=charge,**args(species,count))
    np.testing.assert_array_equal(current,0.)
    # Verify the actual projector, not just the zero numerator in the formula.
    for perturbation in (-.03,.03):
        value = spectral_projector(np.diag(p)-perturbation*charge*probe,edges)
        np.testing.assert_allclose(value,np.diag(member),atol=2e-15)
    assert np.any(member) and not np.all(member)


def test_signed_field_membership_matches_original_envelope_and_keeps_zero_mode_separate():
    for species in ("q","qbar","g"):
        labels = np.arange(-10,11,2) if species=="g" else np.arange(-11,12,2)
        p,member,edges = finite_cell_envelope_spectrum(labels,**args(species))
        reg = args(species)["regulator"]
        for k,m in enumerate(labels):
            assert member[k] == (m!=0 and reg.contains(species,abs(int(m))))
        # All selected and omitted lattice eigenvalues are strictly away from cuts.
        assert np.min(abs(p[:,None]-edges[None,:]))>0.14
    modes = np.arange(-11,12,2)
    probe = np.ones((len(modes),)*2)
    q = finite_cell_envelope_minus_tangent(modes,probe,charge=2/3,**args("q"))
    anti = finite_cell_envelope_minus_tangent(modes,probe,charge=-2/3,**args("qbar"))
    np.testing.assert_array_equal(q,-anti)


def test_envelope_requires_explicit_spectral_scope_physical_charge_and_fourier_modes():
    a = args()
    for update in (dict(prescription=None),dict(species="aux"),dict(total_k2=True),
                   dict(total_pplus_GeV=0.)):
        with pytest.raises(ValueError):finite_cell_envelope_spectrum([-1,1],**{**a,**update})
    with pytest.raises(ValueError):finite_cell_envelope_spectrum([0,2],**a)
    with pytest.raises(ValueError):finite_cell_envelope_spectrum([True,1],**a)
    with pytest.raises(ValueError):finite_cell_envelope_minus_tangent([-2,0,2],np.eye(3),charge=1.,**args("g"))
    with pytest.raises(ValueError):finite_cell_envelope_minus_tangent([-1,1],np.eye(3),charge=1.,**a)
