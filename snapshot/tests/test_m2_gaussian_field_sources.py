"""Common Gaussian sources on full raw color/statistics coordinates, not toys."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_gaussian_field_sources import apply_field_source, FieldRegulatedSourceImage
from deuteron_wigner.quantum.m2_matched_h2_reduction import nested_pilot_isometry
from test_m2_gaussian_field_regulator import regulator
from test_m2_source_vectors import CASES, source, probes
from test_m2_interacting_projector_current import pilot


def apply(p, v, kind, out, width=.8):
    return apply_field_source(p, v, kind=kind, out_cutoff=out, regulator=regulator(width),
                              total_pplus_GeV=1.3, total_transverse_GeV=(0., 0.))


@pytest.mark.parametrize("target,jz", [("PROTON", 1), ("NEUTRON", -1)])
@pytest.mark.parametrize("kind", CASES)
def test_actual_all_field_sources_recover_bare_limit_keep_statistics_and_complex_linearity(kind, target, jz):
    p = pilot(0, target, jz)
    a = probes(p, 2)
    v = np.column_stack((a, (.2+.3j)*a[:, 0]-(.6-.4j)*a[:, 1], np.zeros(len(a))))
    field = apply(p, v, kind, 1)
    near_bare = apply(p, v, kind, 1, 1e7)
    old = source(p, kind, 1, v)
    assert isinstance(field, FieldRegulatedSourceImage)
    assert not field.physical and not field.complete_regulated_H2_matching
    assert field.field_regulator == regulator()
    np.testing.assert_array_equal(field.space.coordinates, old.space.coordinates)
    np.testing.assert_allclose(near_bare.values, old.values, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(field.values[:, 2],
        (.2+.3j)*field.values[:, 0]-(.6-.4j)*field.values[:, 1], atol=2e-14)
    np.testing.assert_array_equal(field.values[:, 3], 0.)
    np.testing.assert_allclose(sum(field.components.values()), field.values, atol=0.)
    np.testing.assert_allclose(field.space.statistics_project(field.values), field.values, atol=3e-14)


@pytest.mark.parametrize("kind", CASES)
def test_rectangular_field_action_has_exact_nested_coordinate_restriction(kind):
    p = pilot(1)
    v = probes(p, 1)
    low, high = apply(p, v, kind, 0), apply(p, v, kind, 1)
    rows = np.flatnonzero(np.sum(np.asarray(high.space.orbitals), axis=1)[high.space.coordinates[:, 1]] == 0)
    np.testing.assert_array_equal(high.space.coordinates[rows], low.space.coordinates)
    np.testing.assert_allclose(low.values, high.values[rows], atol=3e-14)


def test_regulated_P_complement_reaches_ground_Q_and_coherent_contact_still_sums():
    p, small = pilot(1), pilot(0)
    embedding = nested_pilot_isometry(small, p)
    v = probes(p, 1)
    outside = v-embedding@(embedding.conj().T@v)
    # Exact no-lowering selection from the bare common-width source is lost.
    np.testing.assert_allclose(source(p, "qqqgg", 0, outside).values, 0., atol=3e-15)
    assert np.linalg.norm(apply(p, outside, "qqqgg", 0).values) > 1e-6
    np.testing.assert_allclose(apply(p, v, "combined", 1).values,
        apply(p, v, "IF", 1).values+apply(p, v, "IG", 1).values, atol=3e-14)


def test_regulated_sources_cannot_omit_their_field_or_frame_definition():
    p, v = pilot(0), probes(pilot(0), 1)
    args = dict(out_cutoff=0, kind="qqqgg", regulator=regulator(),
                total_pplus_GeV=1.3, total_transverse_GeV=(0., 0.))
    for bad in (dict(regulator=None), dict(total_pplus_GeV=0.),
                dict(total_transverse_GeV=(np.nan, 0.)), dict(kind="unknown")):
        with pytest.raises(ValueError):
            apply_field_source(p, v, **{**args, **bad})
