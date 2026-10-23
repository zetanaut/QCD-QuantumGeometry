from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest
from scipy import sparse

from deuteron_wigner.bridge.basis1 import core as c47
from deuteron_wigner.bridge.c401_c396_mass_directions import (
    D_DELTA_MU_G_SQ,
    D_MU_Q_SQ,
    RESOLUTION_LABELS,
    canonical_partitions,
    coordinate_operator_csr,
    resolution_record,
    source_mass_component_action,
)
from deuteron_wigner.bridge.free2 import core as c128
from deuteron_wigner.bridge.c411_c117_i2_finite_c43_adapter import (
    ExploratoryC1171Parameters,
)
from deuteron_wigner.microscopic.h0 import (
    build_exploratory_k_local_h0,
    direct_target_kinetic_csr,
    k_local_h0_record,
)
from deuteron_wigner.quantum.operator_bundle import (
    build_mapped_exploratory_hamiltonian,
    bundle_record,
)


def _c47_public_labels(resolution: str) -> tuple[tuple[object, ...], ...]:
    full = resolution_record(resolution)["full_resolution_id"]
    c47_resolution = next(item for item in c47.RESOLUTIONS if item.label == full)
    q_labels = tuple(("q", row[3], row[4]) for row in c47.q_basis(c47_resolution))
    qg_rows, _, _ = c47.qg_basis(c47_resolution)
    qg_labels = tuple(
        ("qg", row[0], row[5], row[6], row[9], row[10], row[11])
        for row in qg_rows
    )
    return q_labels + qg_labels


_C128_HELICITY_PAIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _independent_c128_entry_root(
    resolution: str,
    bra: int,
    ket: int,
    value: str,
    branch: str,
) -> str:
    """Encode the source record root without calling a C128 helper.

    C128's record root covers the recurrence branch string.  Checking it here
    makes the source raising/lowering orientation part of the independent
    cross-check, while the M2 route remains numerical and free of C128's
    longitudinal/free-matrix implementation.
    """

    payload = (resolution, "qg", bra, ket, value, branch)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode()).hexdigest()


def _c128_qg_label(full_resolution: str, local_index: int) -> tuple[object, ...]:
    """Decode a C128 qg coordinate without consulting an M2 basis helper."""

    partition, mode, quark_helicity, gluon_helicity, color, _ = c128._qg_decode(  # noqa: SLF001 - source-order audit
        full_resolution, local_index
    )
    n, m = c128._ho_modes(full_resolution)[mode]  # noqa: SLF001 - source-order audit
    return ("qg", partition, n, m, quark_helicity, gluon_helicity, color)


def _c128_qg_index(
    full_resolution: str,
    *,
    partition: int,
    n: int,
    m: int,
    quark_helicity: int,
    gluon_helicity: int,
    color: int,
) -> int:
    """Encode a C128 qg coordinate independently of M2 label lookups."""

    modes = c128._ho_modes(full_resolution)  # noqa: SLF001 - source-order audit
    mode = modes.index((n, m))
    spin = _C128_HELICITY_PAIRS.index((quark_helicity, gluon_helicity))
    return ((partition * len(modes) + mode) * 4 + spin) * 3 + color


@pytest.mark.parametrize("resolution", RESOLUTION_LABELS)
def test_k_local_h0_is_a_full_isometric_map_into_live_target_space(resolution: str) -> None:
    supply = build_exploratory_k_local_h0(resolution)
    dimension = resolution_record(resolution)["direct_sum_dimension"]
    record = k_local_h0_record(supply)

    assert supply.source_operator.shape == supply.target_operator.shape == (dimension, dimension)
    assert supply.basis_map.embedding.shape == (dimension, dimension)
    assert supply.basis_map.nonzero_count == dimension
    assert supply.basis_map.target_support_count == dimension
    assert supply.basis_map.isometry_residual == 0.0
    assert supply.validation["pass"] is True
    assert supply.validation["minimum_qg_eigenvalue_GeV2"] > 0.0
    assert supply.validation["mass_terms_in_h0"] is False
    assert supply.validation["C7_C8_dimension_assumption_used"] is False
    assert supply.validation["physical"] is False
    assert record["basis_map_id"] == supply.basis_map.map_id
    assert record["claim_tier"] == "EXPLORATORY"
    assert record["hamiltonian_activation"] is False


def test_k9_source_order_matches_public_c47_basis_and_functional_diagonal() -> None:
    supply = build_exploratory_k_local_h0("K9")
    assert supply.source_basis_labels == _c47_public_labels("K9")

    full = resolution_record("K9")["full_resolution_id"]
    c47_resolution = next(item for item in c47.RESOLUTIONS if item.label == full)
    functional, rows = c47.free_functional(c47_resolution, mass=0.0)
    assert len(rows) + 6 == supply.dimension
    np.testing.assert_allclose(
        supply.source_operator.diagonal()[6:], functional, rtol=0.0, atol=1.0e-14
    )


def test_k9_m2_off_diagonal_recurrence_cross_checks_c128_pperp2_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhaustively compare K9 M2 radial neighbors to C128 ``pperp2`` only.

    The C128 index decoder and ``pperp2_entry`` are the independent source
    route.  Its historical free entries, sparse matrix, and matrix-free free
    action are deliberately poisoned: neither longitudinal fractions nor an
    evaluated C128 free matrix can participate in this M2 recurrence check.
    """

    def forbidden_c128_free_route(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("C128 historical free route must not enter the M2 recurrence check")

    monkeypatch.setattr(c128, "free_entry", forbidden_c128_free_route)
    monkeypatch.setattr(c128, "free_sparse_matrix", forbidden_c128_free_route)
    monkeypatch.setattr(c128, "apply_free_m2", forbidden_c128_free_route)

    supply = build_exploratory_k_local_h0("K9")
    full = resolution_record("K9")["full_resolution_id"]
    q_dimension = int(resolution_record("K9")["q_dimension"])
    b_squared = float(resolution_record("K9")["b_HO"]) ** 2
    local_dimension = int(resolution_record("K9")["qg_dimension"])
    source_labels = tuple(_c128_qg_label(full, local) for local in range(local_dimension))
    assert supply.target_basis_labels[q_dimension:] == source_labels

    target = supply.target_operator.tocsr()
    modes = c128._ho_modes(full)  # noqa: SLF001 - source-order audit
    partition_count = len(canonical_partitions("K9"))

    for local, label in enumerate(source_labels):
        _, partition, n, m, quark_helicity, gluon_helicity, color = label
        row = q_dimension + local
        diagonal = c128.pperp2_entry(full, "qg", local, local)
        assert diagonal["value"] == "b_HO^2*(2*n+abs(m)+1)"
        assert diagonal["route_a"] == diagonal["route_b"] == diagonal["value"]
        assert diagonal["status"] == "AVAILABLE_SOURCE_QUALIFIED"
        assert diagonal["zero_rule"] is None
        assert diagonal["hermitian_partner"] == (local, local)
        assert target[row, row] == pytest.approx(
            b_squared * (2 * n + abs(m) + 1), abs=1.0e-14
        )

        expected_locals = {local}
        for target_n, orientation in ((n + 1, "raising"), (n - 1, "lowering")):
            if (target_n, m) not in modes:
                continue
            partner_local = _c128_qg_index(
                full,
                partition=partition,
                n=target_n,
                m=m,
                quark_helicity=quark_helicity,
                gluon_helicity=gluon_helicity,
                color=color,
            )
            expected_locals.add(partner_local)
            entry = c128.pperp2_entry(full, "qg", local, partner_local)
            lower_n = min(n, target_n)
            expected_expression = (
                f"-b_HO^2*sqrt({lower_n + 1}*{lower_n + abs(m) + 1})"
            )
            expected_branch = (
                "HO ladder raising" if orientation == "raising" else "HO ladder lowering"
            )
            expected_value = -b_squared * np.sqrt(
                (lower_n + 1) * (lower_n + abs(m) + 1)
            )
            assert entry["value"] == expected_expression
            assert entry["route_a"] == entry["route_b"] == expected_expression
            assert entry["status"] == "AVAILABLE_SOURCE_QUALIFIED"
            assert entry["zero_rule"] is None
            assert entry["hermitian_partner"] == (partner_local, local)
            assert entry["root"] == _independent_c128_entry_root(
                full, local, partner_local, expected_expression, expected_branch
            )
            assert target[row, q_dimension + partner_local] == pytest.approx(
                expected_value, abs=1.0e-14
            )
            reverse = c128.pperp2_entry(full, "qg", partner_local, local)
            assert reverse["value"] == expected_expression
            assert reverse["hermitian_partner"] == (local, partner_local)
            assert reverse["root"] == _independent_c128_entry_root(
                full,
                partner_local,
                local,
                expected_expression,
                "HO ladder lowering" if orientation == "raising" else "HO ladder raising",
            )
            assert target[q_dimension + partner_local, row] == pytest.approx(
                expected_value, abs=1.0e-14
            )
            assert target[row, q_dimension + partner_local] == target[
                q_dimension + partner_local, row
            ]

        observed_local_columns = {
            int(column - q_dimension)
            for column in target.getrow(row).indices
            if column >= q_dimension
        }
        assert observed_local_columns == expected_locals
        assert not np.any(target.getrow(row).indices < q_dimension)

        foreign_partition = (partition + 1) % partition_count
        if foreign_partition != partition:
            forbidden_local = _c128_qg_index(
                full,
                partition=foreign_partition,
                n=n,
                m=m,
                quark_helicity=quark_helicity,
                gluon_helicity=gluon_helicity,
                color=color,
            )
            forbidden = c128.pperp2_entry(full, "qg", local, forbidden_local)
            assert forbidden["value"] == "0"
            assert forbidden["zero_rule"] == "longitudinal/internal identity orthogonality"
            assert target[row, q_dimension + forbidden_local] == 0.0

        flipped_quark = _c128_qg_index(
            full,
            partition=partition,
            n=n,
            m=m,
            quark_helicity=-quark_helicity,
            gluon_helicity=gluon_helicity,
            color=color,
        )
        forbidden_spin = c128.pperp2_entry(full, "qg", local, flipped_quark)
        assert forbidden_spin["value"] == "0"
        assert forbidden_spin["zero_rule"] == "longitudinal/internal identity orthogonality"
        assert target[row, q_dimension + flipped_quark] == 0.0

        radial_non_neighbor = n + 2
        if (radial_non_neighbor, m) in modes:
            forbidden_radial = _c128_qg_index(
                full,
                partition=partition,
                n=radial_non_neighbor,
                m=m,
                quark_helicity=quark_helicity,
                gluon_helicity=gluon_helicity,
                color=color,
            )
            forbidden = c128.pperp2_entry(full, "qg", local, forbidden_radial)
            assert forbidden["value"] == "0"
            assert forbidden["zero_rule"] == "Laguerre radial selection"
            assert target[row, q_dimension + forbidden_radial] == 0.0

    assert supply.validation["C47_sparse_hamiltonian_matrix_supplied"] is False
    assert supply.validation["historical_C128_longitudinal_fractions_used"] is False


def test_embedded_and_independently_built_target_kinetic_routes_are_exact() -> None:
    supply = build_exploratory_k_local_h0("K9")
    direct = direct_target_kinetic_csr("K9")
    residual = (supply.target_operator - direct).tocsr()
    assert residual.nnz == 0
    assert supply.validation["embedded_vs_direct_target_max_abs_residual"] == 0.0
    assert max(supply.validation["commutator_residuals"].values()) == 0.0
    assert supply.validation["source_and_target_orders_differ"] is True


def test_c401_mass_directions_complete_the_free_split_without_double_counting() -> None:
    resolution = "K9"
    supply = build_exploratory_k_local_h0(resolution)
    mu_q_sq = 0.37
    delta_mu_g_sq = -0.04
    full = (
        supply.target_operator
        + mu_q_sq * coordinate_operator_csr(resolution, D_MU_Q_SQ)
        + delta_mu_g_sq * coordinate_operator_csr(resolution, D_DELTA_MU_G_SQ)
    ).tocsr()
    rng = np.random.default_rng(2002)
    vector = rng.normal(size=supply.dimension) + 1j * rng.normal(size=supply.dimension)
    independent = supply.target_operator @ vector + source_mass_component_action(
        resolution,
        vector,
        mu_q_sq=mu_q_sq,
        delta_mu_g_sq=delta_mu_g_sq,
    )
    np.testing.assert_allclose(full @ vector, independent, rtol=0.0, atol=2.0e-13)


def test_historical_c128_numeric_kinetic_is_preserved_but_not_reused() -> None:
    resolution = "K9"
    supply = build_exploratory_k_local_h0(resolution)
    full = resolution_record(resolution)["full_resolution_id"]
    historical = c128.free_sparse_matrix(
        full, parameter_point={"m_q_sq": 0.0, "m_g_sq": 0.0}
    )
    old = sparse.csr_matrix(
        (
            np.asarray(historical["evaluated_data"], dtype=np.complex128),
            (
                np.asarray(historical["rows"], dtype=np.int64),
                np.asarray(historical["cols"], dtype=np.int64),
            ),
        ),
        shape=historical["shape"],
    )
    defect = (old - supply.target_operator).tocsr()
    assert defect.nnz > 0
    assert np.max(np.abs(defect.data)) > 0.1
    assert supply.validation["historical_C128_numeric_operator_used"] is False
    assert supply.validation[
        "historical_C128_qg_transverse_kinetic_denominator_affected"
    ] is True


def test_k9_h0_contract_binds_to_live_c396_c117_bundle_without_mass_double_counting() -> None:
    supply = build_exploratory_k_local_h0("K9")
    coefficients = {D_MU_Q_SQ: 0.20, D_DELTA_MU_G_SQ: 0.10}
    c117 = ExploratoryC1171Parameters(
        resolution="K9", residual_normalization=0.50, mixing_coefficient=0.80
    )
    bundle = build_mapped_exploratory_hamiltonian(
        "K9",
        h0_supply=supply,
        c396_coefficients=coefficients,
        c117_parameters=c117,
        c117_coefficient=0.07,
    )
    record = bundle_record(bundle)

    assert bundle.h0_source == (
        "C47_X_SCALED_BASIS_DIAGONAL_PLUS_M2_HO_RECURRENCE_MAPPED_TO_C401_C410:K9"
    )
    assert record["h0_basis_map_supplied"] is True
    assert record["h0_basis_map_id"] == supply.basis_map.map_id
    assert record["h0_operator_units"] == "GeV^2"
    assert record["h0_mass_terms_included"] is False
    assert "UNIMPLEMENTED_NOT_ZERO" in record["h0_omitted_sector_treatment"]
    assert record["h0_basis_ordering"]["direct_sum"] == "q sector followed by qg sector"
    assert "C401/C396" in record["h0_normalization_ownership"]["mass_terms"]
    assert "C411" in record["h0_normalization_ownership"]["C117"]
    assert record["c117_source_basis"].endswith("C47 shell-major qg")
    assert record["c117_target_basis"].endswith("C128_TRANSVERSE_ORDER")
    assert record["c117_basis_permutation_applied"] is True
    assert record["h0_claim_tier"] == "EXPLORATORY"
    assert record["h0_physical"] is False
    expected = supply.target_operator.copy()
    for term in bundle.terms:
        expected = expected + term.coefficient * term.matrix
    residual = (bundle.matrix() - expected).tocsr()
    assert residual.nnz == 0


def test_mapped_bundle_rejects_a_cross_resolution_h0_supply() -> None:
    with pytest.raises(ValueError, match="resolution disagree"):
        build_mapped_exploratory_hamiltonian(
            "K9",
            h0_supply=build_exploratory_k_local_h0("K11"),
            c396_coefficients={D_MU_Q_SQ: 0.20, D_DELTA_MU_G_SQ: 0.10},
            c117_parameters=ExploratoryC1171Parameters("K9", 0.50, 0.80),
            c117_coefficient=0.07,
        )


def test_k_local_h0_rejects_an_unknown_resolution() -> None:
    with pytest.raises(KeyError):
        build_exploratory_k_local_h0("K15")
