"""Independent CAR/Wick, continuum-sum and coupled-basis checks."""
import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import quad

from deuteron_wigner.quantum.m2_canonical_quark_self import (
    CF, gaussian_coincident_trace, quark_wick_sums,
    quark_wick_infinite_sums, quark_self_contractions,
    quark_mass_squared_direction,
)
from deuteron_wigner.quantum.m2_canonical_instantaneous_fermion import transverse_spin_factor
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot


def _annihilators(n):
    """Independent occupation-bit CAR representation, including antiquarks."""
    dim = 2**n
    result = []
    for i in range(n):
        cols = np.array([c for c in range(dim) if c >> i & 1])
        signs = [(-1)**((int(c) & ((1 << i)-1)).bit_count()) for c in cols]
        result.append(sparse.csr_matrix((signs, (cols ^ (1 << i), cols)), shape=(dim, dim)))
    return result


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_gluon_instantaneous_wick_sum_against_full_fermion_fock_space(n):
    # One good-field component / Abelian unit charge. Color restoration
    # is the independently known sum_a T^a T^a=C_F I.
    ops = _annihilators(2*n)
    b, d = ops[:n], ops[n:]
    size = 2**(2*n)
    currents = {w: sparse.csr_matrix((size, size)) for w in range(-4*n, 4*n+1, 2)}
    for i, p in enumerate(range(1, 2*n, 2)):
        for j, r in enumerate(range(1, 2*n, 2)):
            currents[p-r] += b[i].T @ b[j] - d[j].T @ d[i]
            currents[p+r] += b[i].T @ d[j].T
            currents[-p-r] += d[j] @ b[i]
    for w, c in currents.items():
        assert sparse.linalg.norm(c.getH() - currents[-w]) < 1e-14
    h = sum(.5 * c @ currents[-w] / w**2 for w, c in currents.items() if w)
    for i, p in enumerate(range(1, 2*n, 2)):
        vacuum_subtracted = h[1 << i, 1 << i] - h[0, 0]
        ig = quark_wick_sums(p, boson_modes=n, fermion_modes=n)[1]
        assert vacuum_subtracted == pytest.approx(ig/2, abs=3e-15)


@pytest.mark.parametrize("p", [1, 3, 5, 9])
def test_if_boson_commutator_with_signed_resolvent(p):
    nb = 7
    # Vacuum a a†=1, whereas :a a†:=a†a has zero vacuum expectation.
    a = np.diag(np.sqrt(np.arange(1, 4)), 1)
    vacuum = np.array([1., 0., 0., 0.])
    contraction = sum(vacuum @ (a @ a.T - a.T @ a) @ vacuum / (k*(p-k))
                      for k in range(2, 2*nb+1, 2))
    actual = quark_wick_sums(p, boson_modes=nb, fermion_modes=6)[0]
    assert actual == pytest.approx(contraction, abs=1e-15)
    assert any(p-k < 0 for k in range(2, 2*nb+1, 2))


@pytest.mark.parametrize("h", [-1, 1])
def test_if_helicity_sum_canonical_normalization(h):
    assert sum(transverse_spin_factor(h, g) for g in (-1, 1)) == pytest.approx(2.)
    assert sum(transverse_spin_factor(h, g, crossed=True) for g in (-1, 1)) == pytest.approx(2.)


@pytest.mark.parametrize("p", [1, 3, 5, 7, 13])
def test_infinite_field_trace_sums(p):
    nb = nf = 100_000
    finite = np.array(quark_wick_sums(p, boson_modes=nb, fermion_modes=nf))
    limit = np.array(quark_wick_infinite_sums(p))
    assert abs(finite[0]-limit[0]) < 2.501e-6  # explicit 1/(4N) tail
    assert abs(finite[1]-limit[1]) < 4e-10


def test_zero_transfer_exclusion_does_not_delete_pair_subtraction():
    assert quark_wick_sums(1, boson_modes=1, fermion_modes=1) == (-.5, -.25)
    assert quark_wick_infinite_sums(1) == (-np.log(2), 0.)


@pytest.mark.parametrize("width", [.2, .7, 1.3])
def test_gaussian_trace_by_independent_radial_quadrature(width):
    integral = quad(lambda k: k*np.exp(-k*k/width**2)/(2*np.pi), 0, np.inf)[0]
    assert gaussian_coincident_trace(width) == pytest.approx(integral, abs=2e-13)


@pytest.mark.parametrize("values", [(True, 2, 3), (2, 2, 3), (9, 2, 3), (1, 0, 3), (1, 2, 1.0)])
def test_reject_invalid_field_modes(values):
    p, nb, nf = values
    with pytest.raises(ValueError):
        quark_wick_sums(p, boson_modes=nb, fermion_modes=nf)


@pytest.fixture(scope="module", params=["PROTON", "NEUTRON"])
def pilot(request):
    return build_coupled_pilot(total_k2=9, cutoff=0, target=request.param, jz_twice=1,
                              b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)


def _self(pilot, width=.8):
    return quark_self_contractions(pilot, boson_modes=12, fermion_modes=12,
                                  coincident_trace_GeV2=gaussian_coincident_trace(width))


def test_coupled_self_is_hermitian_non_scalar_and_not_universal_mass(pilot):
    s = _self(pilot).total_per_gs2
    mass = quark_mass_squared_direction(pilot)
    assert np.linalg.norm(s-s.conj().T) < 2e-13
    assert np.linalg.norm(s-np.trace(s)/len(s)*np.eye(len(s))) > .1
    coefficient = np.vdot(mass, s)/np.vdot(mass, mass)
    assert np.linalg.norm(s-coefficient*mass) > .1
    assert np.linalg.eigvalsh(mass)[0] >= 9-1e-12


@pytest.mark.parametrize("charge", ["baryon", "electromagnetic", "up_number", "down_number"])
def test_local_zero_skewness_density_commutes_with_quark_self(pilot, charge):
    s = _self(pilot).total_per_gs2
    rho = good_current_density(pilot, (.12, .16), charge=charge)
    assert np.linalg.norm(s@rho-rho@s) < 3e-13


def test_trace_scaling_and_explicit_normal_order_subtraction(pilot):
    first, second = _self(pilot), _self(pilot, 1.6)
    np.testing.assert_allclose(second.total_per_gs2, 4*first.total_per_gs2, atol=1e-14)
    finite = -.017
    for obj in (first, second):
        unsubtracted = pilot.matrix + pilot.coupling**2 * obj.total_per_gs2
        subtracted = unsubtracted + obj.normal_order_counterterm(finite_quark_mass_squared_GeV2=finite)
        np.testing.assert_allclose(subtracted, pilot.matrix + finite*quark_mass_squared_direction(pilot), atol=1e-14)
    with pytest.raises(TypeError):
        first.normal_order_counterterm()


def test_canonical_box_conversion_matches_retained_contact_coefficient(pilot):
    # Sum over gluon helicities=2; each ordered retained IF contact has
    # K2/(2pi*k2*r2). Its aa† Wick trace must therefore have K2/(pi*k2*r2).
    p, k = 3, 2
    ordered = sum(transverse_spin_factor(1, hg, crossed=True) for hg in (-1, 1))
    canonical = CF*pilot.total_k2/(2*np.pi*k*(p-k))*ordered
    assert canonical == pytest.approx(CF*pilot.total_k2/(np.pi*k*(p-k)))
