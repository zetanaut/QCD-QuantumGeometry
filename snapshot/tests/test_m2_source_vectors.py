"""All canonical source components: matrix parity and rectangular-shell actions."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_source_vectors import SourceImage, source_tensor
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source, apply_excluded_source
from deuteron_wigner.quantum.m2_instantaneous_pair_exit import (
    build_instantaneous_pair_source, build_adjoint_pair_source, build_combined_pair_source, apply_pair_source)
from deuteron_wigner.quantum.m2_instantaneous_sea_exit import (
    build_instantaneous_sea_source, apply_instantaneous_sea_source)
from deuteron_wigner.quantum.m2_six_body_quartic import COMPONENTS, build_direct_source, apply_direct_source
from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
from test_m2_interacting_projector_current import pilot

CASES = ("qqqgg", "qqqq-qbar", "IF", "IG", "combined", "sea")+tuple(COMPONENTS)


def source(p, kind, out, vectors=None):
    if kind in ("qqqgg", "qqqq-qbar"):
        return (build_excluded_source(p, kind, out_cutoff=out) if vectors is None else
                apply_excluded_source(p, kind, vectors, out_cutoff=out))
    if kind in ("IF", "IG", "combined"):
        builds = {"IF": build_instantaneous_pair_source, "IG": build_adjoint_pair_source,
                  "combined": build_combined_pair_source}
        return (builds[kind](p, out_cutoff=out) if vectors is None else
                apply_pair_source(p, vectors, out_cutoff=out, component=kind))
    if kind == "sea":
        return (build_instantaneous_sea_source(p, out_cutoff=out) if vectors is None else
                apply_instantaneous_sea_source(p, vectors, out_cutoff=out))
    return (build_direct_source(p, out_cutoff=out, component=kind) if vectors is None else
            apply_direct_source(p, vectors, out_cutoff=out, component=kind))


def probes(p, count=2):
    rng = np.random.default_rng(716)
    return (rng.normal(size=(len(p.matrix), count))+1j*rng.normal(size=(len(p.matrix), count)))/np.sqrt(len(p.matrix))


@pytest.mark.parametrize("target,jz", [("PROTON", 1), ("NEUTRON", -1)])
@pytest.mark.parametrize("kind", CASES)
def test_every_vector_action_matches_legacy_constructor_and_complex_linearity(kind, target, jz):
    p = pilot(0, target, jz)
    v = probes(p)
    v = np.column_stack((v, (.2+.3j)*v[:, 0]-(.6-.4j)*v[:, 1], np.zeros(len(v))))
    old, new = source(p, kind, 1), source(p, kind, 1, v)
    matrix = old.matrix_per_gs if kind in ("qqqgg", "qqqq-qbar") else old.matrix_per_gs2
    expected = matrix@v[old.retained_columns]
    assert isinstance(new, SourceImage) and not hasattr(new, "retained_columns")
    assert new.coupling_power == (1 if kind in ("qqqgg", "qqqq-qbar") else 2)
    np.testing.assert_array_equal(new.space.coordinates, old.space.coordinates)
    np.testing.assert_allclose(new.values, expected, atol=2e-14, rtol=3e-12)
    np.testing.assert_allclose(new.values[:, 2],
        (.2+.3j)*new.values[:, 0]-(.6-.4j)*new.values[:, 1], atol=2e-14)
    np.testing.assert_array_equal(new.values[:, 3], 0.)
    np.testing.assert_allclose(sum(new.components.values()), new.values, atol=0.)


@pytest.mark.parametrize("kind", CASES)
def test_rectangular_cutoffs_equal_outer_restriction_and_nested_low_source(kind):
    p, small = pilot(1), pilot(0)
    v = probes(p, 1)
    low, high = source(p, kind, 0, v), source(p, kind, 1, v)
    rows = np.flatnonzero(np.sum(np.asarray(high.space.orbitals), axis=1)[high.space.coordinates[:, 1]] == 0)
    np.testing.assert_array_equal(high.space.coordinates[rows], low.space.coordinates)
    np.testing.assert_allclose(low.values, high.values[rows], atol=2e-14)
    embedding = nested_pilot_isometry(small, p)
    outside = v-embedding@(embedding.conj().T@v)
    np.testing.assert_allclose(source(p, kind, 0, outside).values, 0., atol=3e-15)
    small_v = probes(small, 1)
    a, b = source(p, kind, 0, embedding@small_v), source(small, kind, 0, small_v)
    np.testing.assert_allclose(a.values, b.values, atol=2e-14)
    # Historical constructor contract stays unchanged.
    with pytest.raises(ValueError):
        source(p, kind, 0)


@pytest.mark.parametrize("sector", ["qqq", "qqqg"])
def test_tensor_uses_actual_sparse_isometries_and_shared_probe_axis(sector):
    p = pilot(1)
    v = probes(p, 3)
    t = source_tensor(p, sector, v)
    nq = p.bases[0].shape[1]
    if sector == "qqq":
        expected = p.bases[0].toarray()@v[:nq]
        np.testing.assert_allclose(t.reshape(-1, 3), expected, atol=1e-15)
    else:
        start = nq
        for hg, u in enumerate(p.bases[1:]):
            stop = start+u.shape[1]
            np.testing.assert_allclose(t[:, :, :, :, hg].reshape(-1, 3),
                                       u.toarray()@v[start:stop], atol=1e-15)
            start = stop


@pytest.mark.parametrize("kind", ["qqqgg", "IF"])
def test_large_input_norm_scales_validation_not_amplitude_support(kind):
    p = pilot(0)
    v = probes(p, 1)
    a, b = source(p, kind, 1, v), source(p, kind, 1, 1e8*v)
    np.testing.assert_allclose(b.values, 1e8*a.values, atol=5e-8, rtol=3e-12)


def test_coherent_IF_IG_sum_remains_inside_the_action():
    p = pilot(1)
    v = probes(p, 2)
    np.testing.assert_allclose(source(p, "combined", 0, v).values,
        source(p, "IF", 0, v).values+source(p, "IG", 0, v).values, atol=2e-15)


@pytest.mark.parametrize("values", [None, np.ones(37), np.ones((36, 1)), np.ones((37, 0)), np.full((37, 1), np.nan)])
def test_action_inputs_cannot_be_missing_or_relabelled_as_basis_columns(values):
    p = pilot(0)
    with pytest.raises(ValueError):
        apply_excluded_source(p, "qqqgg", values, out_cutoff=0)
    with pytest.raises(ValueError):
        apply_pair_source(p, values, out_cutoff=0, component="combined")
    with pytest.raises(ValueError):
        apply_instantaneous_sea_source(p, values, out_cutoff=0)
    with pytest.raises(ValueError):
        apply_direct_source(p, values, out_cutoff=0, component=COMPONENTS[0])
