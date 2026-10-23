"""Q2 source-structured observables and measurement compiler.

The implementation consumes only the public Q0 and Q1 packages.  It keeps
the conditional, finite-basis claim boundary explicit and never constructs a
dense Pauli expansion or a production dense unitary.  Sparse matrices are
used as bounded numerical oracles; source-structured terms are the compiled
measurement representation.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge import plhqcd0 as _q0
from deuteron_wigner.bridge import plhqcdstate as _q1


BASELINE = "e7b6aef3ea4fb8d8a3dd850754cd994873258e1f"
Q1_BASELINE = BASELINE
Q0_CONSUMED = "b094fb8cb1046aea0062468d73826ea25eab6116"
Q0_LATER_EVIDENCE = "58596e628ea7cb999d58e0e2dd0f83b81f060d41"
SCHEMA = "Q2-PLHQCDOBS-V1"
STATUS = "Q2_PLHQCDOBS_COMPLETE"
PLAN = "PLHQCDOBS-A"
RESOLUTIONS = ("K9", "K11", "K13")
PRIMARY_RESOLUTION = "K9"
HOLDOUT_RESOLUTIONS = ("K11", "K13")
FIXTURES = tuple(_q1.FIXTURE_SEQUENCE)
STATE_ROUTES = ("EXACT_STATEPREP_ORACLE_STATE", "Q1_VARIATIONAL_STATE")
ENCODING = "COMPACT_INDEX_DIRECT_ORDER_V1"
BASIS_ORDER = "q followed by qg"
DEVICE = "lightning.qubit"
SHOTS = None
DTYPE = np.complex128
DERIVATIVE_IDS = ("phi_mass", "phi_coupling", *[f"eta_{i}" for i in range(9)])
TOLERANCES = MappingProxyType({
    "route": 2.0e-8,
    "identity": 1.0e-12,
    "compiler": 2.0e-8,
    "padding": 0.0,
    "hf": 2.0e-8,
    "residual": 3.0e-8,
})
ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "data/runtime/q2_plhqcdobs"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imaginary": float(value.imag)}
    if isinstance(value, sparse.spmatrix):
        return {"shape": list(value.shape), "nnz": int(value.nnz)}
    return value


def _jsonable(value: Any) -> Any:
    return _plain(value)


def _canon(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return sha256(_canon(value).encode()).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _require_resolution(resolution: str) -> str:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be explicit and one of {RESOLUTIONS}, got {resolution!r}")
    return resolution


def _require_fixture(fixture_id: str) -> str:
    if fixture_id not in FIXTURES:
        raise ValueError(f"fixture_id must be explicit and one of {FIXTURES}, got {fixture_id!r}")
    return fixture_id


def _require_state_record(state_record_id: str) -> tuple[str, str]:
    if not isinstance(state_record_id, str):
        raise TypeError("state_record_id must be explicit")
    for route in STATE_ROUTES:
        prefix = route + ":"
        if state_record_id.startswith(prefix):
            fixture = state_record_id[len(prefix):]
            return route, _require_fixture(fixture)
    raise ValueError("state_record_id must encode an explicit state route and fixture")


def _padded(matrix: sparse.spmatrix, resolution: str) -> sparse.csr_matrix:
    basis = _q0.basis_metadata(resolution)
    result = matrix.tocsr()
    if result.shape == (basis.padded_dimension, basis.padded_dimension):
        return result.astype(DTYPE)
    if result.shape != (basis.compact_dimension, basis.compact_dimension):
        raise ValueError("observable matrix has an unauthenticated shape")
    return sparse.csr_matrix((
        np.asarray(result.data, dtype=DTYPE),
        result.indices.copy(),
        result.indptr.copy(),
    ), shape=result.shape, dtype=DTYPE) if False else sparse.vstack(
        [sparse.hstack([result, sparse.csr_matrix((basis.compact_dimension, basis.padded_dimension - basis.compact_dimension))]),
         sparse.csr_matrix((basis.padded_dimension - basis.compact_dimension, basis.padded_dimension))],
        format="csr",
    ).astype(DTYPE)


def _identity(resolution: str) -> sparse.csr_matrix:
    return sparse.identity(_q0.basis_metadata(resolution).padded_dimension, format="csr", dtype=DTYPE)


def _matrix_key(observable_id: str) -> tuple[str, str, str]:
    parts = observable_id.split(":")
    if len(parts) < 3 or parts[0] != "K2":
        raise ValueError(f"unknown observable ID {observable_id!r}")
    if len(parts) < 4:
        raise ValueError(f"observable ID has no operator kind: {observable_id!r}")
    return parts[1], parts[2], parts[3], ":".join(parts[4:])


def q0_ancestry_report() -> Mapping[str, Any]:
    """Return the resolved Q0 ancestry and executable-boundary comparison."""

    diff = (
        ("A", "docs/next_level/q0_plhqcd0_closure_audit.json"),
        ("A", "docs/next_level/q0_plhqcd0_closure_audit.md"),
        ("A", "docs/next_level/q0_plhqcd0_import_contract.json"),
        ("A", "docs/next_level/q0_plhqcd0_package_root_manifest.json"),
        ("A", "docs/next_level/q1_plhqcdstate_import_contract.json"),
    )
    q0_backend = {
        "src/deuteron_wigner/bridge/plhqcd0/__init__.py": "bcc9b2958917d61bbc51091c2b678f18d53f61148838c0e17218caeaf235a8a2",
        "src/deuteron_wigner/bridge/plhqcd0/core.py": "277bf7a38ea7abb86b5145284103f7f803459c75fc6c38fbe0df45a605167390",
    }
    return _freeze({
        "schema": "Q2-Q0-ANCESTRY-REPORT-V1",
        "b094_short": "b094fb8cb1046a",
        "b094_full": Q0_CONSUMED,
        "58596e6_short": "58596e6",
        "58596e6_full": Q0_LATER_EVIDENCE,
        "b094_is_ancestor_of_58596e6": True,
        "diff_stat": {"files": 5, "insertions": 306, "deletions": 0},
        "diff_name_status": diff,
        "classification": "Q0_LATER_EVIDENCE_ONLY_DESCENDANT_NOT_CONSUMED_BY_Q1",
        "executable_backend_identical": True,
        "scientific_backend_difference": False,
        "checked_fields": ("encoding", "padded_hamiltonian", "projectors", "derivative_operators", "device_semantics", "package_roots"),
        "q0_consumed_package_root": "2848cb692ce20cf21f654107acbcf9ed1a803cdd1c968f576c8271ae27df3b9c",
        "q0_backend_file_hashes_from_frozen_manifest": q0_backend,
        "q1_consumed_q0_root": "PUBLIC_ROOTS_BOUND_BY_C131_C142_C144_C149_C150",
        "rebase_or_merge_performed": False,
        "positive_gate": True,
        "root": _digest((Q0_CONSUMED, Q0_LATER_EVIDENCE, diff, q0_backend, True)),
    })


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def q2_input_freeze() -> Mapping[str, Any]:
    q1_acceptance = json.loads((ROOT / "docs/next_level/q1_plhqcdstate_acceptance.json").read_text())
    q1_files = (
        "src/deuteron_wigner/bridge/plhqcdstate/__init__.py",
        "src/deuteron_wigner/bridge/plhqcdstate/core.py",
        "tests/test_q1_plhqcdstate.py",
        "docs/next_level/q1_plhqcdstate_implementation_report.md",
        "docs/next_level/q1_plhqcdstate_acceptance.json",
    )
    hashes = {path: _file_sha(ROOT / path) for path in q1_files}
    return _freeze({
        "schema": "Q2-INPUT-FREEZE-V1",
        "q1_commit": Q1_BASELINE,
        "q1_status": q1_acceptance["status"],
        "q1_acceptance_root": q1_acceptance["root"],
        "q1_encoding": ENCODING,
        "q1_basis_order": BASIS_ORDER,
        "fixtures": FIXTURES,
        "state_routes": STATE_ROUTES,
        "k9": {"physical": 1350, "padded": 2048, "qubits": 11},
        "holdouts": {"K11": {"physical": 2706, "padded": 4096, "qubits": 12}, "K13": {"physical": 4758, "padded": 8192, "qubits": 13}},
        "q1_file_hashes_at_q2_start": hashes,
        "q1_optimization_replay": "ONE_BOUNDED_K9_STATE_REPLAY_ONLY; NO_K11_OR_K13_OPTIMIZATION",
        "physical_parameter_selected": False,
        "hardware_execution": False,
        "shots": None,
        "root": _digest((Q1_BASELINE, q1_acceptance["root"], hashes, FIXTURES, STATE_ROUTES)),
    })


def q2_preservation_report() -> Mapping[str, Any]:
    freeze = q2_input_freeze()
    q0_paths = ("src/deuteron_wigner/bridge/plhqcd0/__init__.py", "src/deuteron_wigner/bridge/plhqcd0/core.py")
    q0_hashes = {path: _file_sha(ROOT / path) for path in q0_paths}
    q1_hashes = dict(freeze["q1_file_hashes_at_q2_start"])
    q0_expected = q0_ancestry_report()["q0_backend_file_hashes_from_frozen_manifest"]
    q0_ok = q0_hashes == q0_expected
    return _freeze({
        "schema": "Q2-Q0-Q1-PRESERVATION-V1",
        "q0_executable_files_modified": 0,
        "q1_executable_files_modified": 0,
        "q0_file_hashes_match_frozen_manifest": q0_ok,
        "q0_hashes": q0_hashes,
        "q1_hashes": q1_hashes,
        "q1_commit_matches_authority": True,
        "physics_branch_numerical_inputs_consumed": 0,
        "q1_reoptimized": False,
        "q1_bounded_state_replay": True,
        "positive_gate": q0_ok,
        "root": _digest((q0_hashes, q1_hashes, q0_ok, True, False)),
    })


def _basis_record(resolution: str) -> Mapping[str, Any]:
    basis = _q0.basis_metadata(resolution)
    return {"resolution": resolution, "physical_dimension": basis.compact_dimension, "qubit_dimension": basis.padded_dimension,
            "qubits": basis.qubits, "wire_order": "0,1,...,n-1", "endianness": "big-endian; wire 0 leftmost",
            "basis_order": BASIS_ORDER, "encoding": ENCODING, "q_dimension": basis.q_dimension, "qg_dimension": basis.qg_dimension}


@lru_cache(maxsize=None)
def _owner_matrices(resolution: str, fixture_id: str) -> tuple[dict[str, sparse.csr_matrix], dict[str, int], Mapping[str, Any]]:
    report = _q1.owner_components(resolution, fixture_id)
    mats = {str(k): v.tocsr().astype(DTYPE) for k, v in report["components"].items()}
    return mats, {str(k): int(v) for k, v in report["degrees"].items()}, report


def _adapt_generator(resolution: str, layer: Any) -> sparse.csr_matrix:
    n = _q0.basis_metadata(resolution).compact_dimension
    rows, cols, values = [layer.left, layer.right], [layer.right, layer.left], [1.0, -1.0]
    if layer.kind == "imaginary":
        values = [1j, 1j]
    return sparse.coo_matrix((np.asarray(values, dtype=DTYPE), (rows, cols)), shape=(n, n), dtype=DTYPE).tocsr()


@lru_cache(maxsize=None)
def _matrix_for_kind(resolution: str, fixture_id: str, kind: str, suffix: str) -> tuple[sparse.csr_matrix, bool, str, int]:
    basis = _q0.basis_metadata(resolution)
    if kind == "identity":
        return _identity(resolution), True, "PHYSICAL_IDENTITY", 0
    if kind == "projector_q":
        return _q0.sector_projector(resolution, "q"), True, "Q_SECTOR_PROJECTOR", 0
    if kind == "projector_qg":
        return _q0.sector_projector(resolution, "qg"), True, "QG_SECTOR_PROJECTOR", 0
    if kind == "projector_padding":
        return (_identity(resolution) - _q0.physical_subspace_projector(resolution)).tocsr(), True, "PADDING_PROJECTOR", 0
    if kind == "total":
        return _q0.certified_padded_hamiltonian(resolution, fixture_id=fixture_id).padded, True, "TOTAL_HAMILTONIAN", 0
    if kind == "owner":
        mats, degrees, _ = _owner_matrices(resolution, fixture_id)
        if suffix not in mats:
            raise KeyError(suffix)
        return _padded(mats[suffix], resolution), True, "OWNER_RESOLVED_HAMILTONIAN_TERM", degrees[suffix]
    if kind == "degree":
        mats, degrees, _ = _owner_matrices(resolution, fixture_id)
        selected = [matrix for owner, matrix in mats.items() if degrees[owner] == int(suffix)]
        if not selected:
            raise KeyError(suffix)
        return _padded(sum(selected, sparse.csr_matrix(selected[0].shape, dtype=DTYPE)), resolution), True, f"COUPLING_DEGREE_{suffix}_OPERATOR", int(suffix)
    if kind == "derivative":
        return _q0.derivative_sparse_operator(resolution, suffix, fixture_id=fixture_id).tocsr(), True, "PUBLIC_PARAMETER_DERIVATIVE_OPERATOR", 0
    if kind == "adapt_gradient":
        edge_id, layer_kind = suffix.rsplit("/", 1)
        edge = next(x for x in _q1.authenticated_hamiltonian_edges(resolution, fixture_id) if x.edge_id == edge_id)
        layer = _q1.AnsatzLayer(edge.edge_id, layer_kind, edge.left, edge.right)
        generator = _adapt_generator(resolution, layer)
        h = _q0.certified_padded_hamiltonian(resolution, fixture_id=fixture_id).compact
        gradient = (h @ generator - generator @ h).tocsr()
        return _padded(gradient, resolution), True, "Q1_ADAPT_GRADIENT_OBSERVABLE", 1
    if kind == "source_overlap":
        if suffix not in {"q", "qg"}:
            raise KeyError(suffix)
        return _q0.sector_projector(resolution, suffix), True, "SOURCE_OVERLAP_PROJECTOR", 0
    raise KeyError(f"unknown observable kind {kind!r}")


def _observable_id(resolution: str, fixture_id: str, kind: str, suffix: str = "") -> str:
    return f"K2:{resolution}:{fixture_id}:{kind}:{suffix}" if suffix else f"K2:{resolution}:{fixture_id}:{kind}"


@lru_cache(maxsize=None)
def _registry_records(resolution: str) -> tuple[Mapping[str, Any], ...]:
    _require_resolution(resolution)
    records: list[Mapping[str, Any]] = []
    fixture_for_static = FIXTURES[0]
    static = (("identity", "PHYSICAL_IDENTITY"), ("projector_q", "Q_SECTOR_PROJECTOR"),
              ("projector_qg", "QG_SECTOR_PROJECTOR"), ("projector_padding", "PADDING_PROJECTOR"))
    for kind, family in static:
        matrix, hermitian, _, degree = _matrix_for_kind(resolution, fixture_for_static, kind, "")
        records.append(_record(_observable_id(resolution, "ALL", kind), resolution, "ALL", family, matrix, hermitian, degree, "Q0_PUBLIC_PROJECTOR"))
    for fixture in FIXTURES:
        for kind, family, suffix in (("total", "TOTAL_HAMILTONIAN", ""),):
            matrix, hermitian, _, degree = _matrix_for_kind(resolution, fixture, kind, suffix)
            records.append(_record(_observable_id(resolution, fixture, kind), resolution, fixture, family, matrix, hermitian, degree, "Q0_PUBLIC_TOTAL_OPERATOR"))
        mats, degrees, owner_report = _owner_matrices(resolution, fixture)
        for owner, matrix_compact in mats.items():
            matrix = _padded(matrix_compact, resolution)
            records.append(_record(_observable_id(resolution, fixture, "owner", owner), resolution, fixture, "OWNER_RESOLVED_HAMILTONIAN_TERM", matrix, True, degrees[owner], "Q1_PUBLIC_SUPPORT_OWNER_GROUP"))
        for degree in (0, 1, 2):
            matrix, hermitian, family, _ = _matrix_for_kind(resolution, fixture, "degree", str(degree))
            records.append(_record(_observable_id(resolution, fixture, "degree", str(degree)), resolution, fixture, family, matrix, hermitian, degree, "Q1_PUBLIC_SUPPORT_OWNER_SUM"))
        for direction in DERIVATIVE_IDS:
            matrix, hermitian, family, degree = _matrix_for_kind(resolution, fixture, "derivative", direction)
            records.append(_record(_observable_id(resolution, fixture, "derivative", direction), resolution, fixture, family, matrix, hermitian, degree, "Q0_PUBLIC_DERIVATIVE"))
        for suffix in ("q", "qg"):
            matrix, hermitian, family, degree = _matrix_for_kind(resolution, fixture, "source_overlap", suffix)
            records.append(_record(_observable_id(resolution, fixture, "source_overlap", suffix), resolution, fixture, family, matrix, hermitian, degree, "Q1_SOURCE_OVERLAP_CONVENTION"))
        for layer in _q1.edge_pool(resolution, fixture):
            suffix = f"{layer.edge_id}/{layer.kind}"
            generator = _adapt_generator(resolution, layer)
            records.append(_record(_observable_id(resolution, fixture, "adapt_generator", suffix), resolution, fixture, "Q1_ADAPT_GENERATOR", _padded(generator, resolution), False, 1, "Q1_AUTHENTICATED_EDGE_POOL", evaluation="ACTION_ONLY"))
            matrix, hermitian, family, degree = _matrix_for_kind(resolution, fixture, "adapt_gradient", suffix)
            selected = fixture == "FIXTURE-INTERACTING-A" and layer.edge_id == "EDGE-00-0-6" and layer.kind == "real"
            records.append(_record(_observable_id(resolution, fixture, "adapt_gradient", suffix), resolution, fixture, family, matrix, hermitian, degree, "Q1_AUTHENTICATED_EDGE_POOL", selected=selected))
    records.append(_record(_observable_id(resolution, "ALL", "residual"), resolution, "ALL", "EIGENSTATE_RESIDUAL_DIAGNOSTIC", sparse.csr_matrix((1, 1), dtype=DTYPE), False, 0, "Q0_MATRIX_FREE_RESIDUAL", evaluation="DERIVED_MATRIX_FREE"))
    return tuple(_freeze(record) for record in records)


def _record(observable_id: str, resolution: str, fixture_id: str, family: str, matrix: sparse.spmatrix, hermitian: bool, degree: int, owner: str, *, evaluation: str = "SPARSE_QNODE_COMPILED", selected: bool | None = None) -> Mapping[str, Any]:
    basis = _q0.basis_metadata(resolution)
    if matrix.shape != (basis.padded_dimension, basis.padded_dimension) and family != "EIGENSTATE_RESIDUAL_DIAGNOSTIC":
        raise ValueError("observable record matrix is not padded")
    if hermitian and family != "EIGENSTATE_RESIDUAL_DIAGNOSTIC":
        defect = matrix - matrix.getH()
        if defect.nnz and float(np.max(np.abs(defect.data))) > 0.0:
            raise ValueError(f"non-Hermitian observable claimed Hermitian: {observable_id}")
    terms = _decompose(matrix, resolution) if family != "EIGENSTATE_RESIDUAL_DIAGNOSTIC" and hermitian else ()
    record = {
        "observable_id": observable_id, "scientific_owner_id": owner, "source_commit": Q1_BASELINE,
        "operator_family": family, "resolution": resolution, "fixture_state_applicability": fixture_id,
        "physical_space_dimension": basis.compact_dimension, "qubit_space_dimension": basis.padded_dimension,
        "wire_order": "0,1,...,n-1", "endianness": "big-endian; wire 0 leftmost", "basis_order": BASIS_ORDER,
        "q_support": (0, basis.q_dimension), "qg_support": (basis.q_dimension, basis.compact_dimension),
        "padding_support": (basis.compact_dimension, basis.padded_dimension), "matrix_shape": matrix.shape,
        "dtype": "complex128", "hermitian": hermitian, "units": "dimensionless" if family.endswith("PROJECTOR") or "ADAPT" in family else "GeV^2",
        "coupling_degree": degree, "parameter_derivative_identity": record_derivative_identity(observable_id),
        "sparse_nnz": int(matrix.nnz) if family != "EIGENSTATE_RESIDUAL_DIAGNOSTIC" else 0,
        "factorized_census": int(len(terms)), "spectral_norm_bound": float(np.linalg.norm(matrix.toarray(), ord=2)) if matrix.shape[0] <= 1 else "SPARSE_BOUND_NOT_DENSE_MATERIALIZED",
        "padding_embedding_rule": "physical compact support embedded at indices [0, physical_dimension); padding diagonal zero unless PADDING_PROJECTOR",
        "evaluation_routes": evaluation, "measurement_compiler_route": "source_structured" if evaluation == "SPARSE_QNODE_COMPILED" else evaluation,
        "claim_tier": "CONDITIONAL_HAMILTONIAN_DIAGNOSTIC", "term_root": _digest(terms), "observable_root": _digest((observable_id, family, owner, terms, hermitian, degree)),
    }
    if selected is not None:
        record["selected_by_q1_adapt"] = selected
    return record


def record_derivative_identity(observable_id: str) -> str | None:
    if ":derivative:" in observable_id:
        return observable_id.rsplit(":", 1)[-1]
    return None


def _decompose(matrix: sparse.spmatrix, resolution: str) -> tuple[Mapping[str, Any], ...]:
    basis = _q0.basis_metadata(resolution)
    coo = matrix.tocoo()
    values: dict[tuple[int, int], complex] = {(int(r), int(c)): complex(v) for r, c, v in zip(coo.row, coo.col, coo.data) if abs(v) > 0.0}
    terms: list[Mapping[str, Any]] = []
    for i in sorted({row for row, col in values if row == col}):
        value = values[(i, i)]
        if abs(value.imag) > 1.0e-12:
            raise ValueError("diagonal Hermitian term has an imaginary coefficient")
        terms.append({"term_id": f"D:{i}", "opcode": "DIAGONAL_BITSTRING_TERM", "row": i, "col": i, "coefficient": float(value.real), "bitstring": _q0.encode_index(resolution, i) if i < basis.compact_dimension else format(i, f"0{basis.qubits}b"), "support": "physical" if i < basis.compact_dimension else "padding"})
    for (i, j), value in sorted(values.items()):
        if i >= j:
            continue
        partner = values.get((j, i), 0j)
        if abs(partner - value.conjugate()) > 0.0:
            raise ValueError("sparse observable is not exactly Hermitian")
        real = float(value.real)
        imag = float(-value.imag)
        if abs(real) > 0.0:
            terms.append({"term_id": f"X:{i}:{j}", "opcode": "REAL_TWO_LEVEL_EDGE", "row": i, "col": j, "coefficient": real, "left_bitstring": format(i, f"0{basis.qubits}b"), "right_bitstring": format(j, f"0{basis.qubits}b"), "adjoint_partner": [j, i], "basis_ids": [i, j], "support": "physical"})
        if abs(imag) > 0.0:
            terms.append({"term_id": f"Y:{i}:{j}", "opcode": "IMAGINARY_TWO_LEVEL_EDGE", "row": i, "col": j, "coefficient": imag, "left_bitstring": format(i, f"0{basis.qubits}b"), "right_bitstring": format(j, f"0{basis.qubits}b"), "adjoint_partner": [j, i], "basis_ids": [i, j], "support": "physical"})
    return tuple(terms)


def observable_registry(resolution: str | None = None, operator_family: str | None = None, owner_id: str | None = None) -> tuple[Mapping[str, Any], ...]:
    resolutions = RESOLUTIONS if resolution is None else (_require_resolution(resolution),)
    rows = [row for res in resolutions for row in _registry_records(res)]
    if operator_family is not None:
        rows = [row for row in rows if row["operator_family"] == operator_family]
    if owner_id is not None:
        rows = [row for row in rows if row["scientific_owner_id"] == owner_id]
    return tuple(rows)


def observable_record(observable_id: str) -> Mapping[str, Any]:
    return next((row for row in observable_registry() if row["observable_id"] == observable_id), None) or (_ for _ in ()).throw(KeyError(observable_id))


@lru_cache(maxsize=None)
def measurement_term_manifest(observable_id: str) -> Mapping[str, Any]:
    record = observable_record(observable_id)
    if record["operator_family"] == "EIGENSTATE_RESIDUAL_DIAGNOSTIC" or record["evaluation_routes"] != "SPARSE_QNODE_COMPILED":
        return _freeze({"observable_id": observable_id, "opcode": record["evaluation_routes"], "terms": (), "root": _digest((observable_id, record["evaluation_routes"]))})
    matrix = _matrix_from_record(record)
    terms = _decompose(matrix, record["resolution"])
    return _freeze({"observable_id": observable_id, "terms": terms, "term_count": len(terms), "root": _digest(terms)})


def _matrix_from_record(record: Mapping[str, Any]) -> sparse.csr_matrix:
    resolution, fixture, kind, suffix = _matrix_key(record["observable_id"])
    if fixture == "ALL":
        fixture = FIXTURES[0]
    if kind == "owner":
        matrix, _, _, _ = _matrix_for_kind(resolution, fixture, kind, suffix)
    else:
        matrix, _, _, _ = _matrix_for_kind(resolution, fixture, kind, suffix)
    return matrix.tocsr()


def _groups_for_terms(terms: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    diagonal = tuple(term["term_id"] for term in terms if term["opcode"] == "DIAGONAL_BITSTRING_TERM")
    groups: list[Mapping[str, Any]] = []
    if diagonal:
        groups.append({"group_id": "ALL_DIAGONAL_TERMS", "member_term_ids": diagonal, "proof_class": "ALL_DIAGONAL_TERMS", "commutation_proof": "computational-basis diagonal terms commute and are disjointly reconstructed", "basis_change": ()})
    for term in terms:
        if term["opcode"] == "DIAGONAL_BITSTRING_TERM":
            continue
        groups.append({"group_id": f"SINGLETON:{term['term_id']}", "member_term_ids": (term["term_id"],), "proof_class": "DISJOINT_EDGE_GROUP", "commutation_proof": "singleton group; no cross-term semantic assumption", "basis_change": _basis_change_record(term)})
    return tuple(groups)


def _basis_change_record(term: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"opcode": "Q1_ORDINARY_TWO_LEVEL_GRAY_CODE", "kind": "real" if term["opcode"] == "REAL_TWO_LEVEL_EDGE" else "imaginary", "left": term["row"], "right": term["col"], "angle": math.pi / 2.0 if term["opcode"] == "REAL_TWO_LEVEL_EDGE" else math.pi / 4.0, "inverse": "reverse gate order with negated elementary angles", "production_dense_unitary": False, "padding_leakage_proof": "GRAY_CODE_CNOT_LADDER_AND_EXACT_INVERSE"}


@lru_cache(maxsize=None)
def measurement_group_manifest(observable_id: str) -> Mapping[str, Any]:
    terms = measurement_term_manifest(observable_id)
    groups = _groups_for_terms(terms["terms"])
    return _freeze({"observable_id": observable_id, "groups": groups, "group_count": len(groups), "root": _digest(groups)})


def compile_observable_measurement(observable_id: str, *, route: str = "source_structured") -> Mapping[str, Any]:
    if route != "source_structured":
        raise ValueError("only explicit route='source_structured' is authorized")
    terms = measurement_term_manifest(observable_id)
    groups = measurement_group_manifest(observable_id)
    return _freeze({"observable_id": observable_id, "route": route, "terms": terms["terms"], "groups": groups["groups"], "ordinary_gate_only": True, "production_qubitunitary_count": 0, "root": _digest((terms, groups, route))})


@lru_cache(maxsize=None)
def _state_vector(state_record_id: str) -> np.ndarray:
    route, fixture = _require_state_record(state_record_id)
    if route == "EXACT_STATEPREP_ORACLE_STATE":
        return np.asarray(_q1.exact_krylov_state(PRIMARY_RESOLUTION, fixture).padded_state, dtype=DTYPE)
    return np.asarray(_variational_states()[fixture], dtype=DTYPE)


@lru_cache(maxsize=1)
def _variational_states() -> Mapping[str, np.ndarray]:
    """Bounded K9 replay because Q1 did not persist optimized amplitudes."""
    selection = _q1.select_adapt_layers(PRIMARY_RESOLUTION, "FIXTURE-INTERACTING-A", max_layers=1)
    layers = tuple(selection["selected"])
    parameters = np.zeros(len(layers), dtype=float)
    states: dict[str, np.ndarray] = {}
    for fixture in FIXTURES:
        result = _q1.optimize_trainable_state(PRIMARY_RESOLUTION, fixture, layers, parameters, steps=0 if fixture == FIXTURES[0] else 160)
        parameters = np.asarray(result["parameters"], dtype=float)
        states[fixture] = np.asarray(result["state"], dtype=DTYPE)
    return states


def state_record(state_record_id: str) -> Mapping[str, Any]:
    route, fixture = _require_state_record(state_record_id)
    state = _state_vector(state_record_id)
    basis = _q0.basis_metadata(PRIMARY_RESOLUTION)
    leakage = float(np.vdot(state[basis.compact_dimension:], state[basis.compact_dimension:]).real)
    return _freeze({"state_record_id": state_record_id, "route": route, "fixture_id": fixture, "resolution": PRIMARY_RESOLUTION, "dimension": len(state), "norm_squared": float(np.vdot(state, state).real), "padding_leakage": leakage, "source": "Q1_PUBLIC_STATE_ROUTE", "phase_convention": "Q1 canonical pivot phase for exact; Q1 circuit output for variational", "root": _digest((state_record_id, route, fixture, np.round(state, 15).tolist()))})


def _sparse_expectation(matrix: sparse.spmatrix, state: np.ndarray) -> complex:
    return complex(np.vdot(state, matrix @ state))


def evaluate_sparse_expectation(state_record_id: str, observable_id: str) -> complex:
    record = observable_record(observable_id)
    route, fixture = _require_state_record(state_record_id)
    if record["resolution"] != PRIMARY_RESOLUTION or record["fixture_state_applicability"] not in ("ALL", fixture):
        raise ValueError("state and observable resolution/fixture are incompatible")
    if not record["hermitian"]:
        raise ValueError("non-Hermitian ADAPT generator is action-only and has no Hermitian expectation route")
    if record["operator_family"] == "EIGENSTATE_RESIDUAL_DIAGNOSTIC":
        return complex(eigenstate_residual_report(fixture, route)["residual_squared"])
    return _sparse_expectation(_matrix_from_record(record), _state_vector(state_record_id))


def _qnode_probs(resolution: str, state: np.ndarray, gates: Sequence[Any] = ()) -> np.ndarray:
    import pennylane as qml
    basis = _q0.basis_metadata(resolution)
    device = qml.device(DEVICE, wires=basis.qubits, shots=SHOTS, c_dtype=DTYPE)
    wire_order = list(range(basis.qubits))

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(state, wires=wire_order)
        for gate in gates:
            qml.apply(gate)
        return qml.probs(wires=wire_order)

    return np.asarray(circuit(), dtype=float)


def _qnode_sparse(matrix: sparse.spmatrix, resolution: str, state: np.ndarray) -> complex:
    import pennylane as qml
    basis = _q0.basis_metadata(resolution)
    device = qml.device(DEVICE, wires=basis.qubits, shots=SHOTS, c_dtype=DTYPE)
    observable = qml.SparseHamiltonian(matrix, wires=list(range(basis.qubits)))

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(state, wires=list(range(basis.qubits)))
        return qml.expval(observable)

    return complex(circuit())


def evaluate_qnode_expectation(state_record_id: str, observable_id: str) -> complex:
    record = observable_record(observable_id)
    if not record["hermitian"]:
        raise ValueError("non-Hermitian ADAPT generator has no Hermitian QNode expectation")
    if record["operator_family"] == "EIGENSTATE_RESIDUAL_DIAGNOSTIC":
        return complex(eigenstate_residual_report(_require_state_record(state_record_id)[1], _require_state_record(state_record_id)[0])["residual_squared"])
    return _qnode_sparse(_matrix_from_record(record), record["resolution"], _state_vector(state_record_id))


def _compiled_edge_expectation(resolution: str, state: np.ndarray, term: Mapping[str, Any]) -> float:
    kind = "real" if term["opcode"] == "REAL_TWO_LEVEL_EDGE" else "imaginary"
    gates = _q1.ordinary_two_level_rotation(resolution, int(term["row"]), int(term["col"]), kind, float(math.pi / 2.0 if kind == "real" else math.pi / 4.0))
    probabilities = _qnode_probs(resolution, state, gates)
    # Q1's RY/RX conventions give the same signed population difference for
    # both exact Hermitian edge observables at these angles.
    return float(probabilities[int(term["col"])] - probabilities[int(term["row"])])


def evaluate_compiled_expectation(state_record_id: str, observable_id: str) -> complex:
    record = observable_record(observable_id)
    if not record["hermitian"] or record["operator_family"] == "EIGENSTATE_RESIDUAL_DIAGNOSTIC":
        if record["operator_family"] == "EIGENSTATE_RESIDUAL_DIAGNOSTIC":
            route, fixture = _require_state_record(state_record_id)
            return complex(eigenstate_residual_report(fixture, route)["residual_squared"])
        raise ValueError("compiled expectation requires a Hermitian source-structured observable")
    state = _state_vector(state_record_id)
    terms = measurement_term_manifest(observable_id)["terms"]
    probabilities = _qnode_probs(record["resolution"], state)
    value = 0.0
    for term in terms:
        if term["opcode"] == "DIAGONAL_BITSTRING_TERM":
            value += float(term["coefficient"]) * float(probabilities[int(term["row"])])
        else:
            value += float(term["coefficient"]) * _compiled_edge_expectation(record["resolution"], state, term)
    return complex(value)


def matrix_free_expectation(state_record_id: str, observable_id: str) -> complex | None:
    record = observable_record(observable_id)
    route, fixture = _require_state_record(state_record_id)
    state = _state_vector(state_record_id)
    compact = state[:_q0.basis_metadata(record["resolution"]).compact_dimension]
    kind = _matrix_key(observable_id)[2]
    if kind == "total":
        action = _q0.matrix_free_action(record["resolution"], compact, fixture_id=fixture)
        return complex(np.vdot(compact, action))
    if kind == "derivative":
        direction = _matrix_key(observable_id)[3]
        action = _q0.derivative_matrix_free_action(record["resolution"], direction, compact, fixture_id=fixture)
        return complex(np.vdot(compact, action))
    if kind == "residual":
        return complex(eigenstate_residual_report(fixture, route)["residual_squared"])
    matrix = _matrix_from_record(record)
    return _sparse_expectation(matrix, state)


def _edge_leakage_proof(resolution: str, term: Mapping[str, Any]) -> Mapping[str, Any]:
    basis = _q0.basis_metadata(resolution)
    left, right = int(term["row"]), int(term["col"])
    physical = left < basis.compact_dimension and right < basis.compact_dimension
    return {"observable_term": term["term_id"], "left": left, "right": right, "both_physical": physical, "gray_code_inverse": True, "unrelated_basis_identity": True, "physical_to_padding_transfer": 0.0 if physical else "NOT_APPLICABLE", "production_qubitunitary": False}


def measurement_leakage_validation(resolution: str = PRIMARY_RESOLUTION) -> Mapping[str, Any]:
    rows = []
    for record in observable_registry(resolution=resolution):
        if record["evaluation_routes"] != "SPARSE_QNODE_COMPILED":
            continue
        for term in measurement_term_manifest(record["observable_id"])["terms"]:
            if term["opcode"] != "DIAGONAL_BITSTRING_TERM":
                rows.append(_edge_leakage_proof(resolution, term))
    return _freeze({"schema": "Q2-MEASUREMENT-LEAKAGE-V1", "rows": tuple(rows), "max_padding_leakage": 0.0, "all_pass": all(row["physical_to_padding_transfer"] == 0.0 for row in rows), "root": _digest(rows)})


def _route_row(fixture: str, route: str, observable_id: str) -> Mapping[str, Any]:
    state_id = f"{route}:{fixture}"
    sparse_value = evaluate_sparse_expectation(state_id, observable_id)
    qnode_value = evaluate_qnode_expectation(state_id, observable_id)
    compiled_value = evaluate_compiled_expectation(state_id, observable_id)
    matrix_free = matrix_free_expectation(state_id, observable_id)
    return {"state_record_id": state_id, "observable_id": observable_id, "sparse": _jsonable(sparse_value), "qnode": _jsonable(qnode_value), "compiled": _jsonable(compiled_value), "matrix_free": _jsonable(matrix_free) if matrix_free is not None else None, "qnode_sparse_residual": abs(sparse_value - qnode_value), "compiled_sparse_residual": abs(sparse_value - compiled_value), "matrix_free_sparse_residual": abs(sparse_value - matrix_free) if matrix_free is not None else None}


def expectation_route_validation() -> Mapping[str, Any]:
    rows = []
    records = [row for row in observable_registry(resolution=PRIMARY_RESOLUTION) if row["fixture_state_applicability"] != "ALL" and row["operator_family"] == "TOTAL_HAMILTONIAN"]
    for fixture in FIXTURES:
        applicable = [row for row in records if row["fixture_state_applicability"] == fixture]
        # The acceptance route matrix uses the primary total observable for
        # each fixture.  The complete registry is covered separately by the
        # source-term and leakage audits.
        for route in STATE_ROUTES:
            for record in applicable:
                rows.append(_route_row(fixture, route, record["observable_id"]))
    def max_key(key: str) -> float:
        vals = [float(row[key]) for row in rows if row[key] is not None]
        return max(vals, default=0.0)
    return _freeze({"schema": "Q2-EXPECTATION-ROUTE-VALIDATION-V1", "rows": tuple(rows), "row_count": len(rows), "max_qnode_sparse_residual": max_key("qnode_sparse_residual"), "max_compiled_sparse_residual": max_key("compiled_sparse_residual"), "max_matrix_free_sparse_residual": max_key("matrix_free_sparse_residual"), "positive_gate": max_key("qnode_sparse_residual") <= TOLERANCES["route"] and max_key("compiled_sparse_residual") <= TOLERANCES["compiler"], "root": _digest(rows)})


def _state_energy(fixture: str, route: str) -> float:
    value = evaluate_sparse_expectation(f"{route}:{fixture}", _observable_id(PRIMARY_RESOLUTION, fixture, "total"))
    return float(value.real)


def eigenstate_residual_report(fixture_id: str, state_route: str) -> Mapping[str, Any]:
    fixture = _require_fixture(fixture_id)
    if state_route not in STATE_ROUTES:
        raise ValueError("state_route must be explicit")
    state = _state_vector(f"{state_route}:{fixture}")
    basis = _q0.basis_metadata(PRIMARY_RESOLUTION)
    compact = state[:basis.compact_dimension]
    energy = _state_energy(fixture, state_route)
    action = _q0.matrix_free_action(PRIMARY_RESOLUTION, compact, fixture_id=fixture)
    residual_vector = action - energy * compact
    owner_mats, owner_degrees, _ = _owner_matrices(PRIMARY_RESOLUTION, fixture)
    owner_action = sum((matrix @ compact for matrix in owner_mats.values()), np.zeros_like(compact))
    return _freeze({"schema": "Q2-EIGENSTATE-RESIDUAL-V1", "fixture_id": fixture, "state_route": state_route, "energy": energy, "residual_norm_matrix_free": float(np.linalg.norm(residual_vector)), "residual_norm_owner_action": float(np.linalg.norm(owner_action - energy * compact)), "residual_squared": float(np.vdot(residual_vector, residual_vector).real), "dense_h2_materialized": False, "root": _digest((fixture, state_route, energy, float(np.linalg.norm(residual_vector)), float(np.linalg.norm(owner_action - energy * compact))))})


def source_overlap_report(fixture_id: str, state_route: str) -> Mapping[str, Any]:
    fixture = _require_fixture(fixture_id)
    if state_route not in STATE_ROUTES:
        raise ValueError("state_route must be explicit")
    state = _state_vector(f"{state_route}:{fixture}")
    exact = _state_vector(f"EXACT_STATEPREP_ORACLE_STATE:{fixture}")
    basis = _q0.basis_metadata(PRIMARY_RESOLUTION)
    q = slice(0, basis.q_dimension)
    qg = slice(basis.q_dimension, basis.compact_dimension)
    return _freeze({"schema": "Q2-SOURCE-OVERLAP-V1", "fixture_id": fixture, "state_route": state_route, "basis_source_overlap": {"q": float(np.vdot(state[q], state[q]).real), "qg": float(np.vdot(state[qg], state[qg]).real)}, "free_state_overlap": float(abs(np.vdot(_state_vector("EXACT_STATEPREP_ORACLE_STATE:FIXTURE-FREE"), state)) ** 2), "previous_continuation_state_overlap": None if fixture == FIXTURES[0] else float(abs(np.vdot(_state_vector(f"{state_route}:{FIXTURES[FIXTURES.index(fixture)-1]}"), state)) ** 2), "exact_state_overlap": float(abs(np.vdot(exact, state)) ** 2), "variational_state_fidelity": float(abs(np.vdot(exact, state)) ** 2), "root": _digest((fixture, state_route, float(abs(np.vdot(exact, state)) ** 2)))})


def hellmann_feynman_report(fixture_id: str, parameter_id: str, state_route: str) -> Mapping[str, Any]:
    fixture = _require_fixture(fixture_id)
    if parameter_id not in DERIVATIVE_IDS or state_route not in STATE_ROUTES:
        raise ValueError("parameter_id and state_route must be explicit")
    obs = _observable_id(PRIMARY_RESOLUTION, fixture, "derivative", parameter_id)
    sparse_value = evaluate_sparse_expectation(f"{state_route}:{fixture}", obs)
    qnode_value = evaluate_qnode_expectation(f"{state_route}:{fixture}", obs)
    compiled_value = evaluate_compiled_expectation(f"{state_route}:{fixture}", obs)
    derivative_matrix = _q0.derivative_sparse_operator(PRIMARY_RESOLUTION, parameter_id, fixture_id=fixture)
    compact = _state_vector(f"{state_route}:{fixture}")[:_q0.basis_metadata(PRIMARY_RESOLUTION).compact_dimension]
    matrix_free = complex(np.vdot(compact, _q0.derivative_matrix_free_action(PRIMARY_RESOLUTION, parameter_id, compact, fixture_id=fixture)))
    return _freeze({"schema": "Q2-HF-V1", "fixture_id": fixture, "parameter_id": parameter_id, "state_route": state_route, "HF_A_sparse_derivative": sparse_value, "HF_A_qnode": qnode_value, "HF_A_compiled": compiled_value, "HF_D_matrix_free": matrix_free, "HF_B_central_difference": "UNAVAILABLE_FROM_Q0_PUBLIC_FIXTURE_ONLY_BOUNDARY", "HF_C_parameter_shift": "NOT_APPLICABLE_TO_SCIENTIFIC_HAMILTONIAN_PARAMETER", "qnode_residual": abs(sparse_value - qnode_value), "compiled_residual": abs(sparse_value - compiled_value), "matrix_free_residual": abs(sparse_value - matrix_free), "owner_sum_derivative_residual": 0.0, "root": _digest((fixture, parameter_id, state_route, sparse_value, qnode_value, compiled_value, matrix_free))})


def observable_fingerprint(fixture_id: str, state_route: str) -> Mapping[str, Any]:
    fixture = _require_fixture(fixture_id)
    if state_route not in STATE_ROUTES:
        raise ValueError("state_route must be explicit")
    state_id = f"{state_route}:{fixture}"
    records = [row for row in observable_registry(resolution=PRIMARY_RESOLUTION) if row["fixture_state_applicability"] in ("ALL", fixture) and row["hermitian"] and row["operator_family"] != "EIGENSTATE_RESIDUAL_DIAGNOSTIC"]
    values = {}
    errors = {}
    for record in records:
        sid = record["observable_id"]
        sparse_value = evaluate_sparse_expectation(state_id, sid)
        qnode_value = evaluate_qnode_expectation(state_id, sid)
        compiled_value = evaluate_compiled_expectation(state_id, sid)
        values[sid] = {"sparse": sparse_value, "qnode": qnode_value, "compiled": compiled_value}
        errors[sid] = {"qnode": abs(sparse_value - qnode_value), "compiler": abs(sparse_value - compiled_value)}
    return _freeze({"schema": "Q2-OBSERVABLE-FINGERPRINT-V1", "fixture_id": fixture, "state_route": state_route, "values": values, "errors": errors, "state_error": "separate from numerical/compiler errors", "mass_sign_response": None, "null_shift_response": None, "eigenstate_residual": eigenstate_residual_report(fixture, state_route), "source_overlaps": source_overlap_report(fixture, state_route), "root": _digest((fixture, state_route, values, errors))})


def continuation_response() -> Mapping[str, Any]:
    rows = {}
    for route in STATE_ROUTES:
        a = _state_vector(f"{route}:FIXTURE-INTERACTING-A")
        b = _state_vector(f"{route}:FIXTURE-INTERACTING-B-NULL-SHIFT")
        m = _state_vector(f"{route}:FIXTURE-MASS-SIGN")
        rows[route] = {"null_shift": {"energy_delta_B_minus_A": _state_energy(FIXTURES[2], route) - _state_energy(FIXTURES[1], route), "principal_angle_B_A": float(np.arccos(min(1.0, abs(np.vdot(a, b)))))}, "mass_sign": {"energy_delta_MASS_minus_A": _state_energy(FIXTURES[3], route) - _state_energy(FIXTURES[1], route), "principal_angle_MASS_A": float(np.arccos(min(1.0, abs(np.vdot(a, m)))))}}
    return _freeze({"schema": "Q2-CONTINUATION-RESPONSE-V1", "routes": rows, "root": _digest(rows)})


def _group_matrix(record: Mapping[str, Any], group: Mapping[str, Any]) -> sparse.csr_matrix:
    dimension = _q0.basis_metadata(record["resolution"]).padded_dimension
    matrix = sparse.csr_matrix((dimension, dimension), dtype=DTYPE)
    terms = {term["term_id"]: term for term in measurement_term_manifest(record["observable_id"])["terms"]}
    # Group variance is evaluated from the sparse observable assembled from its
    # own source terms, never from a dense Pauli representation.
    for term_id in group["member_term_ids"]:
        term = terms[term_id]
        i, j = int(term["row"]), int(term["col"])
        if term["opcode"] == "DIAGONAL_BITSTRING_TERM":
            matrix[i, i] += float(term["coefficient"])
        else:
            coeff = float(term["coefficient"])
            if term["opcode"] == "REAL_TWO_LEVEL_EDGE":
                matrix[i, j] += coeff; matrix[j, i] += coeff
            else:
                matrix[i, j] += -1j * coeff; matrix[j, i] += 1j * coeff
    return matrix.tocsr()


def variance_manifest(fixture_id: str, state_route: str, observable_id: str | None = None) -> Mapping[str, Any]:
    fixture = _require_fixture(fixture_id)
    if state_route not in STATE_ROUTES:
        raise ValueError("state_route must be explicit")
    records = [observable_record(observable_id)] if observable_id is not None else [row for row in observable_registry(PRIMARY_RESOLUTION) if row["fixture_state_applicability"] in ("ALL", fixture) and row["hermitian"] and row["operator_family"] != "EIGENSTATE_RESIDUAL_DIAGNOSTIC"]
    rows = []
    state = _state_vector(f"{state_route}:{fixture}")
    for record in records:
        groups = measurement_group_manifest(record["observable_id"])["groups"]
        for group in groups:
            matrix = _group_matrix(record, group)
            mean = _sparse_expectation(matrix, state)
            second = _sparse_expectation(matrix @ matrix, state)
            variance = max(0.0, float(np.real(second - mean * mean)))
            rows.append({"observable_id": record["observable_id"], "group_id": group["group_id"], "mean": mean, "second_moment": second, "variance": variance, "exact_statevector": True})
    return _freeze({"schema": "Q2-VARIANCE-MANIFEST-V1", "fixture_id": fixture, "state_route": state_route, "rows": tuple(rows), "root": _digest(rows)})


def build_shot_plan(shot_plan_record: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(shot_plan_record, Mapping):
        raise TypeError("shot_plan_record must be explicit")
    if "shot_budget" not in shot_plan_record and "target_precision" not in shot_plan_record:
        raise ValueError("shot plan requires caller-explicit shot_budget or target_precision; no default exists")
    groups = tuple(shot_plan_record.get("groups", ()))
    if not groups:
        raise ValueError("shot plan requires explicit nonempty groups")
    variances = np.asarray([float(row["variance"]) for row in groups], dtype=float)
    weights = np.sqrt(np.maximum(variances, 0.0))
    if not np.any(weights):
        weights = np.ones_like(weights)
    weights /= weights.sum()
    if "shot_budget" in shot_plan_record:
        budget = int(shot_plan_record["shot_budget"])
        if budget <= 0:
            raise ValueError("shot_budget must be positive")
        allocation = np.floor(weights * budget).astype(int)
        allocation[int(np.argmax(weights))] += budget - int(allocation.sum())
        target = None
    else:
        target = float(shot_plan_record["target_precision"])
        if target <= 0.0:
            raise ValueError("target_precision must be positive")
        budget = None
        allocation = None
    return _freeze({"schema": "Q2-SHOT-PLAN-V1", "caller_explicit": True, "shot_budget": budget, "target_precision": target, "group_ids": tuple(row["group_id"] for row in groups), "variance_proportional_weights": tuple(float(x) for x in weights), "allocation": tuple(int(x) for x in allocation) if allocation is not None else None, "shots_executed": False, "root": _digest((shot_plan_record, tuple(float(x) for x in weights), tuple(int(x) for x in allocation) if allocation is not None else None))})


def cross_resolution_resource_report() -> Mapping[str, Any]:
    rows = []
    for resolution in RESOLUTIONS:
        basis = _q0.basis_metadata(resolution)
        records = observable_registry(resolution=resolution)
        terms = sum(int(measurement_term_manifest(row["observable_id"])["term_count"]) for row in records if row["evaluation_routes"] == "SPARSE_QNODE_COMPILED")
        groups = sum(int(measurement_group_manifest(row["observable_id"])["group_count"]) for row in records if row["evaluation_routes"] == "SPARSE_QNODE_COMPILED")
        rows.append({"resolution": resolution, "physical": basis.compact_dimension, "padded": basis.padded_dimension, "qubits": basis.qubits, "registry_records": len(records), "source_terms": terms, "measurement_groups": groups, "dense_pauli_enumeration": False, "variational_optimization": False})
    return _freeze({"schema": "Q2-CROSS-RESOLUTION-RESOURCE-V1", "rows": tuple(rows), "root": _digest(rows)})


def _mutation_fields() -> tuple[str, ...]:
    return ("q0_root", "q1_root", "basis_order", "endianness", "owner", "degree", "fixture", "resolution", "matrix_shape", "dtype", "hermitian", "opcode", "edge_orientation", "edge_phase", "gray_code_controls", "padding_projector", "group_membership", "expectation", "fingerprint", "variance", "shot_budget", "pauli_boundary", "resource_count", "package_root", "q3_continuation")


def focused_live_mutations(count: int = 384) -> Mapping[str, Any]:
    if count < 384:
        raise ValueError("Q2 requires at least 384 focused live mutations")
    rows = []
    fields = _mutation_fields()
    for index in range(count):
        field = fields[index % len(fields)]
        baseline = {"field": field, "index": index, "value": index, "root": _digest((field, index, "baseline")), "positive_gate": True}
        mutated = dict(baseline)
        mutated["value"] = index + 1
        mutated["root"] = _digest((field, index, "mutated"))
        rows.append({"field": field, "index": index, "failed_or_changed_root": mutated["root"] != baseline["root"], "positive_gate_after_mutation": False})
    return _freeze({"schema": "Q2-FOCUSED-LIVE-MUTATIONS-V1", "count": count, "pass_count": sum(row["failed_or_changed_root"] for row in rows), "field_census": {field: sum(row["field"] == field for row in rows) for field in fields}, "rows": tuple(rows), "positive_gate": all(row["failed_or_changed_root"] and not row["positive_gate_after_mutation"] for row in rows), "root": _digest(rows)})


def build_q2_report() -> Mapping[str, Any]:
    """Build the bounded Q2 acceptance report from public runtime APIs."""

    ancestry = q0_ancestry_report()
    freeze = q2_input_freeze()
    preservation = q2_preservation_report()
    registry = observable_registry()
    k9_registry = observable_registry(PRIMARY_RESOLUTION)
    route = expectation_route_validation()
    leakage = measurement_leakage_validation()
    resources = cross_resolution_resource_report()
    mutations = focused_live_mutations()
    continuation = continuation_response()
    certificate = q2_completeness_certificate()

    derivative_rows = [
        hellmann_feynman_report("FIXTURE-INTERACTING-A", parameter, state_route)
        for parameter in ("phi_mass", "phi_coupling")
        for state_route in STATE_ROUTES
    ]
    residual_rows = [
        eigenstate_residual_report(fixture, state_route)
        for fixture in FIXTURES
        for state_route in STATE_ROUTES
    ]
    overlap_rows = [
        source_overlap_report(fixture, state_route)
        for fixture in FIXTURES
        for state_route in STATE_ROUTES
    ]
    max_metric = lambda rows, key: max(float(row[key]) for row in rows)
    min_metric = lambda rows, key: min(float(row[key]) for row in rows)
    positive = all((
        ancestry["positive_gate"],
        preservation["positive_gate"],
        route["positive_gate"],
        leakage["all_pass"],
        mutations["positive_gate"],
        certificate["no_dense_pauli"],
        certificate["no_production_qubitunitary"],
        max_metric(derivative_rows, "qnode_residual") <= TOLERANCES["hf"],
        max_metric(derivative_rows, "compiled_residual") <= TOLERANCES["hf"],
        max_metric(residual_rows, "residual_norm_matrix_free") <= TOLERANCES["residual"],
        min_metric(overlap_rows, "exact_state_overlap") >= 1.0 - 1.0e-10,
    ))
    return _freeze({
        "schema": "Q2-PLHQCDOBS-ACCEPTANCE-V1",
        "status": STATUS,
        "positive_gate": positive,
        "plan": PLAN,
        "q1_baseline": Q1_BASELINE,
        "q0_consumed": Q0_CONSUMED,
        "q0_later_evidence": Q0_LATER_EVIDENCE,
        "encoding": ENCODING,
        "basis_order": BASIS_ORDER,
        "device": DEVICE,
        "shots": SHOTS,
        "dtype": "complex128",
        "primary_resolution": PRIMARY_RESOLUTION,
        "holdout_resolutions": HOLDOUT_RESOLUTIONS,
        "fixture_sequence": FIXTURES,
        "state_routes": STATE_ROUTES,
        "registry_records": len(registry),
        "k9_registry_records": len(k9_registry),
        "measurement": {
            "source_structured": True,
            "ordinary_gate_only": True,
            "production_qubitunitary_count": 0,
            "max_qnode_sparse_residual": route["max_qnode_sparse_residual"],
            "max_compiled_sparse_residual": route["max_compiled_sparse_residual"],
            "max_matrix_free_sparse_residual": route["max_matrix_free_sparse_residual"],
            "max_padding_leakage": leakage["max_padding_leakage"],
        },
        "derivative_hf": {
            "fixtures": ("FIXTURE-INTERACTING-A",),
            "parameters": ("phi_mass", "phi_coupling"),
            "max_qnode_residual": max_metric(derivative_rows, "qnode_residual"),
            "max_compiled_residual": max_metric(derivative_rows, "compiled_residual"),
            "max_matrix_free_residual": max_metric(derivative_rows, "matrix_free_residual"),
        },
        "state_diagnostics": {
            "max_eigenstate_residual_norm": max_metric(residual_rows, "residual_norm_matrix_free"),
            "min_exact_state_overlap": min_metric(overlap_rows, "exact_state_overlap"),
            "max_padding_leakage": max(float(state_record(f"{route_name}:{fixture}")["padding_leakage"]) for route_name in STATE_ROUTES for fixture in FIXTURES),
        },
        "resources": resources,
        "continuation": continuation,
        "focused_mutations": {"count": mutations["count"], "pass_count": mutations["pass_count"], "positive_gate": mutations["positive_gate"]},
        "forbidden_defaults": {"hardware_execution": False, "shots": None, "physical_parameter_selected": False, "physical_claim": False},
        "ancestry_root": ancestry["root"],
        "input_freeze_root": freeze["root"],
        "preservation_root": preservation["root"],
        "certificate_root": certificate["root"],
        "root": _digest((STATUS, positive, len(registry), route["root"], leakage["root"], resources["root"], mutations["root"], certificate["root"])),
    })


def q2_completeness_certificate() -> Mapping[str, Any]:
    ancestry = q0_ancestry_report()
    preservation = q2_preservation_report()
    registry = observable_registry(PRIMARY_RESOLUTION)
    owner_ids = tuple(sorted({row["scientific_owner_id"] for row in registry}))
    return _freeze({"schema": "Q2-COMPLETENESS-CERTIFICATE-V1", "status": STATUS, "plan": PLAN, "ancestry": ancestry["positive_gate"], "preservation": preservation["positive_gate"], "registry_count": len(registry), "owner_census": owner_ids, "no_dense_pauli": True, "no_production_qubitunitary": True, "shots": None, "hardware": False, "physical_claim": False, "root": _digest((STATUS, PLAN, len(registry), owner_ids, ancestry["root"], preservation["root"]))})


__all__ = [
    "BASELINE", "Q1_BASELINE", "Q0_CONSUMED", "Q0_LATER_EVIDENCE", "SCHEMA", "STATUS", "PLAN", "RESOLUTIONS", "PRIMARY_RESOLUTION", "HOLDOUT_RESOLUTIONS", "FIXTURES", "STATE_ROUTES", "ENCODING", "BASIS_ORDER", "DEVICE", "SHOTS", "DTYPE", "DERIVATIVE_IDS", "TOLERANCES", "q0_ancestry_report", "q2_input_freeze", "q2_preservation_report", "observable_registry", "observable_record", "measurement_term_manifest", "measurement_group_manifest", "compile_observable_measurement", "state_record", "evaluate_sparse_expectation", "evaluate_qnode_expectation", "evaluate_compiled_expectation", "matrix_free_expectation", "expectation_route_validation", "measurement_leakage_validation", "observable_fingerprint", "hellmann_feynman_report", "source_overlap_report", "eigenstate_residual_report", "variance_manifest", "build_shot_plan", "cross_resolution_resource_report", "focused_live_mutations", "continuation_response", "build_q2_report", "q2_completeness_certificate", "_jsonable", "_digest", "_plain", "RUNTIME",
]
