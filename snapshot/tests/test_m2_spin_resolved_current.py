import numpy as np
import pytest
from scipy import linalg, sparse

from deuteron_wigner.bridge.modes.core import GAMMA, polarization, polarization_cartesian
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density, good_current_density_between
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_spin_resolved_current import (
    transverse_parity_map, parity_aligned_partner, dirac_pauli_good_structures,
    good_form_factor_diagnostic,
)


@pytest.mark.parametrize("h", [-1, 1])
@pytest.mark.parametrize("pplus,px,py,m", [(1., 0., 0., .35), (.7, .23, -.31, 0.)])
def test_independent_dirac_transverse_parity(h, pplus, px, py, m):
    sx = np.array([[0., 1.], [1., 0.]])
    transform = -1j*linalg.block_diag(sx, sx)@GAMMA[0]
    np.testing.assert_allclose(transform@lf_spinor(pplus, px, py, m, h),
                               -1j*lf_spinor(pplus, -px, py, m, -h), atol=2e-15)
    eps = polarization_cartesian(polarization(pplus, px, py, h))
    reflected = polarization_cartesian(polarization(pplus, -px, py, -h))
    np.testing.assert_allclose(np.diag([1., -1., 1., 1.])@eps, reflected, atol=2e-15)


@pytest.mark.parametrize("delta", [(0., 0.), (.2, 0.), (0., .2), (.12, .16)])
@pytest.mark.parametrize("pp", [.6, 3.])
def test_external_spinor_dirac_pauli_structures(delta, pp):
    mass = .94
    f1, f2 = dirac_pauli_good_structures(delta, mass_GeV=mass, total_pplus=pp)
    x, y = delta
    expected = np.array([[0., -x+1j*y], [x+1j*y, 0.]])/(2*mass)
    np.testing.assert_allclose(f1, np.eye(2), atol=2e-15)
    np.testing.assert_allclose(f2, expected, atol=2e-15)


def test_structure_diagnostic_retains_incompatible_matrix_residual():
    delta = (.12, .16)
    f1, f2 = dirac_pauli_good_structures(delta, mass_GeV=.94, total_pplus=3.)
    value = .83*f1 + 1.7*f2 + .03*np.diag([1., -1.])
    result = good_form_factor_diagnostic(value, delta, mass_GeV=.94, total_pplus=3.)
    assert result.dirac == pytest.approx(.83)
    assert result.pauli == pytest.approx(1.7)
    assert np.linalg.norm(result.residual_matrix) == pytest.approx(.03*np.sqrt(2))
    with pytest.raises(ValueError):
        good_form_factor_diagnostic(np.eye(2), (0., 0.), mass_GeV=.94, total_pplus=3.)


@pytest.fixture(scope="module", params=["PROTON", "NEUTRON"])
def pair(request):
    ps = [build_coupled_pilot(total_k2=9, cutoff=1, target=request.param, jz_twice=j, b_GeV=.4,
                              quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3) for j in (1,-1)]
    e, v = linalg.eigh(ps[0].matrix, subset_by_index=[0,0])
    up = v[:,0]
    down = parity_aligned_partner(ps[1], ps[0], up, intrinsic_parity=1)
    return ps, (up, down), float(e[0])


def _spin_current(pair, delta, charge):
    ps, vectors, _ = pair
    return np.array([[vf.conj()@good_current_density_between(pf, pi, delta, charge=charge)@vi
                     for pi, vi in zip(ps, vectors)] for pf, vf in zip(ps, vectors)])


def test_exact_parity_intertwines_every_retained_hamiltonian_term(pair):
    (up, down), _, _ = pair
    t = transverse_parity_map(down, up)
    reverse = transverse_parity_map(up, down)
    np.testing.assert_allclose(reverse@t, -np.eye(len(up.matrix)), atol=1e-13)
    for name in ("h0", "cross_pairs_per_gs2", "emission_per_gs",
                 "instantaneous_fermion_per_gs2", "matrix"):
        np.testing.assert_allclose(getattr(down, name)@t, t@getattr(up, name), atol=1e-13)


@pytest.mark.parametrize("charge", ["baryon", "electromagnetic", "up_number", "down_number"])
def test_spin_resolved_zero_charge_reversal_and_existing_same_jz_api(pair, charge):
    ps, _, _ = pair
    delta = np.array([.12, .16])
    charges = dict(baryon=1., electromagnetic=1., up_number=2., down_number=1.)
    if ps[0].target == "NEUTRON":
        charges.update(electromagnetic=0., up_number=1., down_number=2.)
    np.testing.assert_allclose(_spin_current(pair, (0., 0.), charge), charges[charge]*np.eye(2), atol=3e-14)
    for out in ps:
        for inc in ps:
            forward = good_current_density_between(out, inc, delta, charge=charge)
            reverse = good_current_density_between(inc, out, -delta, charge=charge)
            np.testing.assert_allclose(forward, reverse.conj().T, atol=2e-14)
    np.testing.assert_allclose(good_current_density(ps[0], delta, charge=charge),
                               good_current_density_between(ps[0], ps[0], delta, charge=charge), atol=0)


def test_retained_ground_phase_alignment_is_eigenstate_not_arbitrary_degeneracy_pick(pair):
    ps, vectors, energy = pair
    assert np.linalg.norm(ps[1].matrix@vectors[1]-energy*vectors[1]) < 8e-14
    assert np.linalg.norm(vectors[1]) == pytest.approx(1., abs=2e-14)
    with pytest.raises(TypeError):
        parity_aligned_partner(ps[1], ps[0], vectors[0])
    with pytest.raises(ValueError):
        parity_aligned_partner(ps[1], ps[0], vectors[0], intrinsic_parity=True)


def test_nonzero_pauli_signal_transverse_orientation_and_reversal(pair):
    values = []
    for delta in ((.2, 0.), (0., .2), (.12, .16), (-.12, -.16)):
        rho = _spin_current(pair, delta, "electromagnetic")
        result = good_form_factor_diagnostic(rho, delta, mass_GeV=np.sqrt(pair[2]), total_pplus=3.)
        assert np.linalg.norm(result.residual_matrix) < 4e-14
        assert abs(result.pauli.imag) < 2e-14
        values.append([result.dirac, result.pauli])
    np.testing.assert_allclose(values, [values[0]]*4, atol=4e-14)
    assert abs(values[0][1]) > 1e-4


def test_fock_resolved_quantum_helicity_interference_current(pair):
    # Executed exact oracle, not a variational ansatz or scalability claim.
    import pennylane as qml
    ps, vectors, energy = pair
    delta = np.array([.12, .16]); q = np.linalg.norm(delta)
    phase = (delta[1]-1j*delta[0])/q
    state = np.r_[vectors[0], phase*vectors[1]]/np.sqrt(2)
    rho = np.block([[good_current_density_between(pf, pi, delta, charge="electromagnetic")
                     for pi in ps] for pf in ps])
    cosine, sine = (rho+rho.conj().T)/2, (rho-rho.conj().T)/(2j)
    qubits = int(np.ceil(np.log2(len(state))))
    size = 2**qubits
    padded = np.pad(state, (0,size-len(state)))
    ops = [sparse.block_diag((sparse.csr_matrix(op), sparse.csr_matrix((size-len(state),)*2)), format="csr")
           for op in (cosine, sine)]
    device = qml.device("lightning.qubit", wires=qubits)
    @qml.qnode(device)
    def circuit():
        qml.StatePrep(padded, wires=range(qubits))
        return tuple(qml.expval(qml.SparseHamiltonian(op, wires=range(qubits))) for op in ops)
    measured = np.array(circuit())
    direct = np.array([np.vdot(state, op@state).real for op in (cosine, sine)])
    np.testing.assert_allclose(measured, direct, atol=3e-14)
    fit = good_form_factor_diagnostic(_spin_current(pair, delta, "electromagnetic"), delta,
                                      mass_GeV=np.sqrt(energy), total_pplus=3.)
    np.testing.assert_allclose(measured, [fit.dirac.real, q/(2*np.sqrt(energy))*fit.pauli.real], atol=3e-14)
    assert qubits == 9 and np.linalg.norm(padded[len(state):]) == 0
