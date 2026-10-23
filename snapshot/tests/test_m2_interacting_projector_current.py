"""Nested interacting-cutoff Ward tests on actual canonical symmetry bases."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
from deuteron_wigner.quantum.m2_galerkin_projector_current import (
    retained_projector_vertex_between, interacting_galerkin_boundary_between,
)


@lru_cache(None)
def pilot(n, target="PROTON", jz=1):
    return build_coupled_pilot(total_k2=9, cutoff=n, target=target, jz_twice=jz,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)


@pytest.mark.parametrize("target,jz,outer", [("PROTON", 1, 2), ("PROTON", -1, 1), ("NEUTRON", -1, 1)])
def test_interacting_boundary_closes_actual_nested_hamiltonian_defect(target, jz, outer):
    small_i, small_f = pilot(0, target, 1), pilot(0, target, jz)
    parent_i, parent_f = pilot(outer, target, 1), pilot(outer, target, jz)
    pi = nested_pilot_isometry(small_i, parent_i)
    pf = nested_pilot_isometry(small_f, parent_f)
    Pi, Pf = pi@pi.conj().T, pf@pf.conj().T
    delta = np.array([.12, .16])
    rho = good_current_density_between(parent_f, parent_i, delta, charge="electromagnetic")
    k = retained_projector_vertex_between(parent_f, parent_i, inner_cutoff=0,
        transfer=delta, charge="electromagnetic", quadrature_order=16)
    np.testing.assert_allclose(np.einsum("j,joi->oi", delta, k), Pf@rho-rho@Pi, atol=8e-15)
    np.testing.assert_allclose(k, Pf@k+k@Pi, atol=6e-14)
    np.testing.assert_allclose(pf.conj().T@parent_f.matrix@pf, small_f.matrix, atol=7e-14)
    np.testing.assert_allclose(pi.conj().T@parent_i.matrix@pi, small_i.matrix, atol=7e-14)
    boundary = interacting_galerkin_boundary_between(small_f, small_i,
        parent_outgoing=parent_f, parent_incoming=parent_i, transfer=delta,
        charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16)
    inner_rho = pf.conj().T@rho@pi
    residual = small_f.matrix@inner_rho-inner_rho@small_i.matrix
    residual -= pf.conj().T@(parent_f.matrix@rho-rho@parent_i.matrix)@pi
    np.testing.assert_allclose(residual, 2*1.3*np.einsum("j,joi->oi", delta, boundary), atol=1e-13)
    assert np.linalg.norm(boundary) > 1e-6
    reverse = interacting_galerkin_boundary_between(small_i, small_f,
        parent_outgoing=parent_i, parent_incoming=parent_f, transfer=-delta,
        charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16)
    np.testing.assert_allclose(boundary, reverse.conj().swapaxes(1, 2), atol=7e-14)


@pytest.mark.parametrize("delta", [(0., 0.), (1e-10, -2e-10), (.42, -.31)])
def test_interaction_partner_regular_and_path_converged(delta):
    low, high = pilot(0), pilot(1)
    low_f, high_f = pilot(0, "PROTON", -1), pilot(1, "PROTON", -1)
    def partner(order, high=high, high_f=high_f):
        return interacting_galerkin_boundary_between(low_f, low,
            parent_outgoing=high_f, parent_incoming=high, transfer=delta,
            charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=order)
    a, b = partner(16), partner(32)
    np.testing.assert_allclose(a, b, atol=2e-14)
    assert np.all(np.isfinite(a))
    # The free N=0 orbital boundary vanishes, the actual cubic source does not.
    free = partner(16, replace(high, matrix=high.h0), replace(high_f, matrix=high_f.h0))
    np.testing.assert_allclose(free, 0., atol=2e-14)
    assert np.linalg.norm(a-free) > 1e-6


def test_all_source_powers_are_carried_linearly_not_only_free_H():
    low, high = pilot(0), pilot(2)
    k = retained_projector_vertex_between(high, high, inner_cutoff=0,
        transfer=(.12, .16), charge="up_number", quadrature_order=16)
    p = nested_pilot_isometry(low, high)
    def boundary(h):
        return np.array([p.conj().T@(v@h+h@v)@p/(2*1.3) for v in k])
    full = boundary(high.matrix)
    pieces = (boundary(high.h0), boundary(high.emission_per_gs),
              boundary(high.cross_pairs_per_gs2), boundary(high.instantaneous_fermion_per_gs2))
    np.testing.assert_allclose(full, pieces[0]+.3*pieces[1]+.09*(pieces[2]+pieces[3]), atol=3e-14)
    assert np.linalg.norm(pieces[1]) > 1e-4
    assert np.linalg.norm(pieces[2]) > 1e-4
    assert np.linalg.norm(pieces[3]) > 1e-4


@pytest.mark.parametrize("target", ["PROTON", "NEUTRON"])
def test_first_shell_same_helicity_zero_is_selection_not_convergence(target):
    low, high = pilot(0, target), pilot(1, target)
    k = retained_projector_vertex_between(high, high, inner_cutoff=0,
        transfer=(.12, .16), charge="electromagnetic", quadrature_order=16)
    # A scalar density preserves individual helicities. At fixed Jz the
    # N=1 orbital states cannot connect to N=0 without changing orbital Lz.
    np.testing.assert_array_equal(k, 0.)
    boundary = interacting_galerkin_boundary_between(low, low,
        parent_outgoing=high, parent_incoming=high, transfer=(.12, .16),
        charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=16)
    np.testing.assert_array_equal(boundary, 0.)


def test_explicit_outer_model_and_matching_endpoints_required():
    a, b = pilot(0), pilot(1)
    with pytest.raises(ValueError):
        retained_projector_vertex_between(b, b, inner_cutoff=1, transfer=(0., 0.),
            charge="baryon", quadrature_order=16)
    with pytest.raises(ValueError):
        interacting_galerkin_boundary_between(a, a, parent_outgoing=b, parent_incoming=b,
            transfer=(.1, .2), charge="baryon", total_pplus_GeV=0., quadrature_order=16)
