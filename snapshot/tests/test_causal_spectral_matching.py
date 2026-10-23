import numpy as np
import pytest
from scipy.integrate import quad

from deuteron_wigner.quantum.causal_spectral_matching import (
    SpectralResponse, pcac_pole, gaussian_response, spectral_null_experiment, resolving_probes,
)


def example():
    rng = np.random.default_rng(100926)
    b = rng.normal(size=(3, 2, 2))+1j*rng.normal(size=(3, 2, 2))
    return SpectralResponse(np.array([.16, .7, 1.8]), b@b.conj().transpose(0, 2, 1), "physical_response")


@pytest.mark.parametrize("z", [-.4, .3+.1j, 2.+.7j])
def test_analytic_reflection_and_physical_matrix_positivity(z):
    a = example()
    np.testing.assert_allclose(a.analytic(np.conj(z)), a.analytic(z).conj().T, atol=1e-13)
    if np.imag(z) > 0:
        c = a.analytic(z)
        spectral = (c-c.conj().T)/(2j)
        assert np.min(np.linalg.eigvalsh(spectral)) > 0
    k = a.reflection_matrix([.1, .4, 1., 2.])
    assert np.min(np.linalg.eigvalsh(k)) > -1e-12


def test_positive_spacelike_values_are_not_reflection_positivity():
    a = SpectralResponse(np.array([.25, 1.]), np.array([[[-.1]], [[1.1]]]), "signed_remainder")
    assert np.min(a.euclidean(np.geomspace(.001, 100.)).real) > 0
    assert np.min(np.linalg.eigvalsh(a.reflection_matrix([.1, 1., 3., 6.]))) < -.001
    with pytest.raises(ValueError, match="PSD"):
        SpectralResponse(a.mass_squared, a.residues, "physical_response")


@pytest.mark.parametrize("omega", [.3+.4j, -.8+.6j, 1.3+.3j])
def test_retarded_transform_independent_time_integral(omega):
    a = example()
    assert np.max(abs(a.time_response([-4., -1., -.001], .2))) == 0
    expected = sum(r*complex(
        quad(lambda t: (np.exp(1j*omega*t)*np.sin(np.sqrt(s+.2)*t)/np.sqrt(s+.2)).real, 0, np.inf, epsabs=2e-11)[0],
        quad(lambda t: (np.exp(1j*omega*t)*np.sin(np.sqrt(s+.2)*t)/np.sqrt(s+.2)).imag, 0, np.inf, epsabs=2e-11)[0])
        for s, r in zip(a.mass_squared, a.residues))
    np.testing.assert_allclose(a.retarded(omega, .2), expected, atol=2e-8, rtol=2e-9)


def test_signed_remainder_is_causal_not_an_independent_positive_response():
    full = SpectralResponse(np.array([.25, 1.2]), np.array([[[.6]], [[.4]]]), "physical_response")
    retained = SpectralResponse(np.array([.25]), np.array([[[1.]]]), "physical_response")
    residual = full.minus(retained)
    assert residual.role == "signed_remainder"
    assert residual.euclidean(.5).real[0, 0] < 0
    np.testing.assert_allclose(residual.moment(0), 0., atol=1e-15)
    for z in [-.3, .2+.02j, 1.1+.1j]:
        np.testing.assert_allclose(retained.analytic(z)+residual.analytic(z), full.analytic(z), atol=1e-14)
    assert np.max(abs(residual.time_response([-1., -.1]))) == 0


def test_fixed_UV_budget_cannot_add_nonzero_positive_soft_measure():
    canonical = SpectralResponse(np.array([0.]), np.array([[[1.2]]]), "physical_response")
    positive_soft = SpectralResponse(np.array([.3, 1.]), np.array([[[.2]], [[.1]]]), "physical_response")
    q = 1e7
    assert (q*(canonical.euclidean(q)+positive_soft.euclidean(q)))[0, 0].real == pytest.approx(1.5, abs=3e-8)
    # A signed zero-total measure has a Q^-4, not Q^-2, leading correction.
    signed = SpectralResponse(np.array([.3, 1.]), np.array([[[.2]], [[-.2]]]), "signed_remainder")
    assert signed.moment(0)[0, 0] == 0
    assert (q*q*signed.euclidean(q))[0, 0].real == pytest.approx(-signed.moment(1)[0, 0].real, rel=1e-6)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_subtracted_dispersion_restores_only_supplied_Taylor_contacts(order):
    a = example()
    q0, q = .4, 1.1
    contacts = sum((-(q-q0))**k*np.einsum("n,nab->ab", 1/(a.mass_squared+q0)**(k+1), a.residues) for k in range(order))
    np.testing.assert_allclose(a.subtracted_remainder(q, q0, order)+contacts, a.euclidean(q), atol=2e-14)
    np.testing.assert_allclose(a.subtracted_remainder(q0, q0, order), 0, atol=1e-15)


def test_PCAC_pole_transports_existing_generated_reference_without_new_solve():
    import json
    from pathlib import Path
    old = json.loads((Path(__file__).resolve().parents[1]/"docs/next_level/shared_chiral_dynamics_results.json").read_text())
    p = old["finite_mass_solution"]
    pole = pcac_pole(p["mass"], p["decay_constant"], p["pseudoscalar_residue"], old["inputs"]["current_mass_GeV"], .8)
    ratio = 2*.004/.8
    np.testing.assert_allclose(pole.residues[0, 1], ratio*pole.residues[0, 0], atol=2e-14)
    assert np.linalg.matrix_rank(pole.residues[0]) == 1
    retained = SpectralResponse(pole.mass_squared, .35*pole.residues, "physical_response")
    residual = pole.minus(retained)
    # Verify the whole PP, PA, AA pole, not just its mass or PP element.
    np.testing.assert_allclose(residual.moment(0), .65*pole.residues[0], atol=2e-16)
    z = p["mass_squared"]+1e-8j
    np.testing.assert_allclose((p["mass_squared"]-z)*residual.analytic(z), .65*pole.residues[0], atol=1e-10)
    with pytest.raises(ValueError, match="PCAC"):
        pcac_pole(p["mass"], p["decay_constant"], 2*p["pseudoscalar_residue"], .004, .8)


def test_nonunique_positive_spectra_despite_spacelike_and_two_UV_constraints():
    r = spectral_null_experiment()
    assert r["constraint_rank"] == 6 and r["nullity"] == 3
    assert r["constraint_max_error"] < 1e-14
    assert min(r["residues_a"].min(), r["residues_b"].min()) > 0
    assert r["smeared_spectral_relative_difference"] > .1


def test_independent_spectral_bins_resolve_declared_finite_family_not_all_dynamics():
    r = spectral_null_experiment()
    rows, selected = resolving_probes(r["mass_squared"], r["spacelike_q_squared"], .025)
    assert len(selected) == 3 and np.linalg.matrix_rank(rows) == 9
    np.testing.assert_allclose(np.linalg.solve(rows, rows@r["residues_a"]), r["residues_a"], atol=1e-12)


def test_replay_serializable_and_source_insertion_is_independent_information():
    import json
    from scripts.evaluate_causal_spectral_matching import evaluate
    r = evaluate()
    json.dumps(r, allow_nan=False)
    assert r["field_coordinates"]["max_response_relative_error"] < 1e-13
    assert r["field_coordinates"]["same_two_point_but_different_scalar_insertion_relative"] > .1
    assert r["supplied_pion_pole"]["residual_residue_error"] < 1e-14


@pytest.mark.parametrize("z", [-.2, -.7, .8+.03j])
def test_field_redefinition_preserves_sources_but_not_internal_propagator(z):
    m = np.diag([.16, .7, 1.8]); b = np.array([[.8, .1], [.2, .9], [.4, -.3]])
    n = np.zeros((3, 3)); n[0, 1] = 1
    t = np.eye(3)+.7*z*n  # nilpotent N, det T=1; inverse I-.7*z*N
    reference = gaussian_response(m, b, z)
    np.testing.assert_allclose(gaussian_response(m, b, z, t), reference, atol=1e-13)
    wrong = gaussian_response(m, b, z, t, transform_sources=False)
    assert np.linalg.norm(wrong-reference)/np.linalg.norm(reference) > .01
    inverse = np.linalg.inv(m-z*np.eye(3))
    changed = np.linalg.inv(t.T@(m-z*np.eye(3))@t)
    assert np.linalg.norm(changed-inverse) > .01


def test_shared_scalar_insertions_and_second_sources_survive_field_transport_Torch():
    import torch
    m = torch.diag(torch.tensor([.16, .7, 1.8], dtype=torch.float64))
    b = torch.tensor([[.8, .1], [.2, .9], [.4, -.3]], dtype=torch.float64)
    c = torch.tensor([[.3, .12, 0.], [.12, -.1, .05], [0., .05, .2]], dtype=torch.float64)
    n = torch.zeros((3, 3), dtype=torch.float64); n[0, 1] = 1
    h = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    z = -.4
    a = m+h*c-z*torch.eye(3, dtype=torch.float64)
    t = torch.eye(3, dtype=torch.float64)+(.7*z+.3*h)*n
    def derivatives(value):
        first = torch.autograd.grad(value, h, create_graph=True, retain_graph=True)[0]
        second = torch.autograd.grad(first, h, retain_graph=True)[0]
        return np.array([value.item(), first.item(), second.item()])
    base = derivatives((b.T@torch.linalg.solve(a, b))[0, 1])
    moved_b = t.T@b
    moved = derivatives((moved_b.T@torch.linalg.solve(t.T@a@t, moved_b))[0, 1])
    frozen = derivatives((b.T@torch.linalg.solve(t.T@a@t, b))[0, 1])
    np.testing.assert_allclose(moved, base, atol=1e-14)
    inverse = np.linalg.inv(m.detach().numpy()-z*np.eye(3))
    bb, cc = b.numpy(), c.numpy()
    np.testing.assert_allclose(base[1:], [(-bb.T@inverse@cc@inverse@bb)[0, 1],
                                        (2*bb.T@inverse@cc@inverse@cc@inverse@bb)[0, 1]], atol=1e-14)
    assert np.linalg.norm(frozen-base) > .01


def test_diagonal_sources_miss_relative_sign_information():
    s = np.array([.25, 1.])
    a, b = np.array([.8, .3]), np.array([.8, -.3])
    common = np.array([.2, .6])
    x = SpectralResponse(s, np.array([np.outer(a, a), np.outer(common, common)]), "physical_response")
    y = SpectralResponse(s, np.array([np.outer(b, b), np.outer(common, common)]), "physical_response")
    for q in [.1, .5, 2.]:
        np.testing.assert_allclose(np.diag(x.euclidean(q)), np.diag(y.euclidean(q)))
        assert abs((x.euclidean(q)-y.euclidean(q))[0, 1]) > .1


@pytest.mark.parametrize("bad", ["negative_mass", "nonhermitian", "role", "retarded", "pole", "zero_energy"])
def test_invalid_inputs_do_not_become_physical_zeros(bad):
    with pytest.raises(ValueError):
        if bad == "negative_mass": SpectralResponse(np.array([-.1]), np.ones((1, 1, 1)), "physical_response")
        if bad == "nonhermitian": SpectralResponse(np.array([.1]), np.array([[[1j]]]), "signed_remainder")
        if bad == "role": SpectralResponse(np.array([.1]), np.ones((1, 1, 1)), "colored_positive_by_default")
        if bad == "retarded": example().retarded(.2-.1j, 0.)
        if bad == "pole": example().analytic(.16)
        if bad == "zero_energy": SpectralResponse(np.array([0.]), np.ones((1, 1, 1)), "physical_response").time_response([1.])
