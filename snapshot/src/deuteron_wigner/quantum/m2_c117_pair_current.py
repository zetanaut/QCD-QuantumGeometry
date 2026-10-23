"""Source-derived C117 quark-pair primitives, separate from the M2m Hamiltonian.

This module fixes the ordered-Fourier sign and supplies unreduced color-pair
operators and the number-preserving qq cross-pair kernel. Couplings and finite
C43/RI-SMOM matching are NOT selected. Existing C410/M2m matrices are not changed.

The 91-state K2=27 qqq diagnostic uses all positive APBC longitudinal triples
and a fixed physical transverse Gaussian. Its equal-share member is exactly
the M2l intrinsic global ground: b_single=b_lambda/sqrt(3). Away from equal
share this defines an explicit fixed-physical-width regulator, not a claim
of an M2m embedding or a complete transverse/Fock truncation.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb, factorial, isfinite, pi, sqrt
from operator import index
from typing import Mapping

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.c403_c117_i2_numerical_primitive.spatial import HOMode
from deuteron_wigner.microscopic.h0.color import (
    ColorSingletBasis, representation_generators,
)

_REPS = {"qqq": ("3", "3", "3"), "qqqg": ("3", "3", "3", "8")}
QUARK_PAIRS = tuple(combinations(range(3), 2))


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(name + " must be an integer, not bool")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(name + " must be an exact integer") from exc


def _positive(value: float, name: str) -> float:
    result = float(value)
    if isinstance(value, (bool, np.bool_)) or not isfinite(result) or result <= 0:
        raise ValueError(name + " must be finite and positive")
    return result


def inverse_i_derivative(transfer: int, half_length_GeVinv: float) -> float:
    """Q0/(i partial+) eigenvalue on exp(-i*pi*n*xminus/L); zero is removed."""
    n = _integer(transfer, "transfer")
    length = _positive(half_length_GeVinv, "half_length_GeVinv")
    return 0.0 if n == 0 else length / (pi * n)


def source_current_square(
    current_modes: Mapping[int, np.ndarray], half_length_GeVinv: float
) -> np.ndarray:
    """Integral (R j)(R j), NOT integral (R j)^dagger (R j).

    Coefficients are j_n in j(x)=sum_n j_n exp(-i*pi*n*x/L); include their
    field/box normalization before calling. Require j_-n=j_n^dagger.
    This is a finite-mode numerical identity, not a complete field projection.
    """
    length = _positive(half_length_GeVinv, "half_length_GeVinv")
    if not current_modes:
        raise ValueError("at least one current mode is required")
    modes = {}
    shape = None
    for raw_n, value in current_modes.items():
        n = _integer(raw_n, "transfer")
        matrix = np.asarray(value, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
            raise ValueError("current coefficients must be nonempty square matrices")
        if shape is None:
            shape = matrix.shape
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise ValueError("current coefficients must have common shape and be finite")
        modes[n] = matrix
    for n, value in modes.items():
        if -n not in modes:
            raise ValueError("both transfer orientations are required")
        residual = np.linalg.norm(modes[-n] - value.conj().T)
        if residual > 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(value)):
            raise ValueError("current coefficients violate j_-n = j_n^dagger")
    result = np.zeros(shape, dtype=np.complex128)
    for n, value in modes.items():
        result += (2 * length * inverse_i_derivative(n, length)
                   * inverse_i_derivative(-n, length)) * (value @ modes[-n])
    return result


def instantaneous_pminus_per_gs2(
    current_modes: Mapping[int, np.ndarray], half_length_GeVinv: float
) -> np.ndarray:
    """Bare source coefficient -1/2 applied ONCE, with g_s^2 factored."""
    return -0.5 * source_current_square(current_modes, half_length_GeVinv)


def _slot_action(matrix: np.ndarray, tensors: np.ndarray, slot: int) -> np.ndarray:
    return np.moveaxis(np.tensordot(matrix, tensors, axes=(1, slot)), 0, slot)


@lru_cache(maxsize=None)
def _color_pair_cached(sector: str, left: int, right: int) -> np.ndarray:
    reps = _REPS[sector]
    basis = ColorSingletBasis.construct(sector).tensors
    dims = tuple(3 if rep == "3" else 8 for rep in reps)
    tensor = basis.reshape(dims + (basis.shape[1],))
    acted = np.zeros_like(tensor)
    for a, b in zip(representation_generators(reps[left]),
                    representation_generators(reps[right])):
        acted += _slot_action(a, _slot_action(b, tensor, right), left)
    result = basis.conj().T @ acted.reshape(basis.shape)
    result.setflags(write=False)
    return result


def color_pair_matrix(sector: str, left: int, right: int) -> np.ndarray:
    """Complete C7 singlet matrix of sum_a T_left^a T_right^a, including self."""
    if sector not in _REPS:
        raise ValueError("sector must be qqq or qqqg")
    left, right = _integer(left, "left"), _integer(right, "right")
    if not (0 <= left < len(_REPS[sector]) and 0 <= right < len(_REPS[sector])):
        raise IndexError("constituent slot outside sector")
    return np.array(_color_pair_cached(sector, left, right), copy=True)


def color_permutation_matrix(sector: str, permutation: tuple[int, int, int]) -> np.ndarray:
    """C7 color action with tensor axes transposed as in the M2 S3 carrier."""
    if sector not in _REPS or tuple(sorted(permutation)) != (0, 1, 2):
        raise ValueError("expected qqq/qqqg sector and an S3 permutation")
    permutation = tuple(_integer(p, "permutation") for p in permutation)
    basis = ColorSingletBasis.construct(sector).tensors
    dims = tuple(3 if rep == "3" else 8 for rep in _REPS[sector])
    axes = permutation + tuple(range(3, len(dims))) + (len(dims),)
    transformed = basis.reshape(dims + (basis.shape[1],)).transpose(axes)
    return basis.conj().T @ transformed.reshape(basis.shape)


def four_ho_contact(
    bra: tuple[HOMode, HOMode], ket: tuple[HOMode, HOMode], b_GeV: float
) -> complex:
    """Integral phi_a* phi_b* phi_c phi_d d2r in the exact C45 convention.

    One common physical-coordinate HO width; no density-member sum or
    intermediate contraction is implicit. Includes all C45 Fourier phases.
    """
    b = _positive(b_GeV, "b_GeV")
    if len(bra) != 2 or len(ket) != 2 or not all(
        isinstance(mode, HOMode) for mode in (*bra, *ket)
    ):
        raise TypeError("bra and ket must each contain two HOMode values")
    if sum(m.m for m in bra) != sum(m.m for m in ket):
        return 0j
    polynomial = [Fraction(1)]
    norm = Fraction(1)
    phase = 1 + 0j
    for position, mode in enumerate((*bra, *ket)):
        a = abs(mode.m)
        coefficients = [
            Fraction((-1)**d * comb(mode.n + a, mode.n - d), factorial(d))
            for d in range(mode.n + 1)
        ]
        result = [Fraction(0)] * (len(polynomial) + len(coefficients) - 1)
        for i, x in enumerate(polynomial):
            for j, y in enumerate(coefficients):
                result[i+j] += x*y
        polynomial = result
        norm *= Fraction(factorial(mode.n), factorial(mode.n + a))
        factor = (-1)**mode.n * (1j)**a
        phase *= factor.conjugate() if position < 2 else factor
    power = sum(abs(m.m) for m in (*bra, *ket)) // 2
    radial = sum(
        c * Fraction(factorial(power+d), 2**(power+d+1))
        for d, c in enumerate(polynomial)
    )
    return phase * (b*b/pi) * sqrt(float(norm)) * float(radial)


def ordered_quark_pminus_factor(transfer: int, half_length_GeVinv: float) -> float:
    """One ordered qq current product, canonical density 1/(2L), g_s^2 factored."""
    length = _positive(half_length_GeVinv, "half_length_GeVinv")
    n = _integer(transfer, "transfer")
    if n == 0:
        return 0.0
    # -1/2 * (2L) * [L/(pi*n)] [L/(-pi*n)] * (2L)^-2.
    return length / (4 * pi*pi * n*n)


def ordered_quark_mass_factor(total_k2: int) -> float:
    """Coefficient of 1/n^2 times transverse/color product for ONE ordered term."""
    total = _integer(total_k2, "total_k2")
    if total <= 0:
        raise ValueError("total_k2 must be positive")
    # 2P+ = pi*K2/L; this factor is not an active-subsystem 1/y embedding.
    return total / (4*pi)


def qqq_longitudinal_basis(total_k2: int) -> tuple[tuple[int, int, int], ...]:
    """All ordered positive APBC quark triples at fixed doubled total momentum."""
    total = _integer(total_k2, "total_k2")
    if total < 3 or total % 2 != 1:
        raise ValueError("three APBC quarks require odd total_k2 >= 3")
    return _qqq_longitudinal_basis_cached(total)


@lru_cache(maxsize=None)
def _qqq_longitudinal_basis_cached(total: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (a, b, total-a-b)
        for a in range(1, total-1, 2)
        for b in range(1, total-a, 2)
        if total-a-b > 0
    )


def qq_longitudinal_pair_kernel(total_k2: int, pair: tuple[int, int]) -> sparse.csr_matrix:
    """1/n^2 exchange; unchanged spectator, exact momentum conservation and Q0."""
    if len(pair) != 2:
        raise ValueError("pair must contain two distinct quark slots")
    i, j = (_integer(p, "pair slot") for p in pair)
    if i == j or i not in range(3) or j not in range(3):
        raise ValueError("pair must contain two distinct quark slots")
    basis = qqq_longitudinal_basis(total_k2)
    lookup = {state: n for n, state in enumerate(basis)}
    rows, columns, values = [], [], []
    for column, state in enumerate(basis):
        for out_i in range(1, state[i]+state[j], 2):
            transfer = (out_i-state[i]) // 2
            if transfer == 0:
                continue
            out = list(state)
            out[i], out[j] = out_i, state[i]+state[j]-out_i
            rows.append(lookup[tuple(out)])
            columns.append(column)
            values.append(1.0/(transfer*transfer))
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(basis),)*2)


def qqq_longitudinal_permutation(total_k2: int, permutation: tuple[int, int, int]) -> sparse.csr_matrix:
    if tuple(sorted(permutation)) != (0, 1, 2):
        raise ValueError("permutation must be in S3")
    permutation = tuple(_integer(p, "permutation") for p in permutation)
    basis = qqq_longitudinal_basis(total_k2)
    lookup = {state: n for n, state in enumerate(basis)}
    rows = [lookup[tuple(state[i] for i in permutation)] for state in basis]
    return sparse.csr_matrix((np.ones(len(basis)), (rows, np.arange(len(basis)))),
                             shape=(len(basis),)*2)


def qqq_fixed_width_ground_pair_mass_per_gs2(
    total_k2: int, pair: tuple[int, int], b_lambda_GeV: float
) -> sparse.csr_matrix:
    """Bare qq cross-pair matrix, NOT a C117 renormalized coordinate binding.

    Intrinsic physical-coordinate Gaussian from the M2l equal-share ground:
    b_single=b_lambda/sqrt(3). It is kept fixed when longitudinal modes change.
    CM factors separate under an orthogonal equal-mass Jacobi transform.
    Pair contact = b_lambda^2/(6*pi). Both ordered constituent terms are included.
    Spin/flavor identity is implicit; antisymmetry is imposed on the tensor
    product with C7 color and the existing nucleon spin-flavor state, not by
    discarding longitudinal coordinates.
    """
    width = _positive(b_lambda_GeV, "b_lambda_GeV")
    longitudinal = qq_longitudinal_pair_kernel(total_k2, pair)
    ground = HOMode(0, 0)
    contact = four_ho_contact((ground, ground), (ground, ground), width/sqrt(3))
    color = color_pair_matrix("qqq", *pair)[0, 0]
    return (2 * ordered_quark_mass_factor(total_k2) * contact * color * longitudinal).tocsr()


def qqq_fixed_width_ground_mass_per_gs2(total_k2: int, b_lambda_GeV: float) -> sparse.csr_matrix:
    """Sum all THREE qq cross pairs; excludes one-body contractions and qg/gg."""
    matrices = [qqq_fixed_width_ground_pair_mass_per_gs2(total_k2, p, b_lambda_GeV)
                for p in QUARK_PAIRS]
    return sum(matrices[1:], matrices[0]).tocsr()
