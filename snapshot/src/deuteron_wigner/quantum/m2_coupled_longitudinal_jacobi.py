"""Coupled longitudinal/Jacobi kinematics, NOT an enlarged interacting M2 H.

Positive APBC quarks and a positive PBC gluon share one finite cell.
Jacobi modes use global mass-weighted coordinates and a common oscillator
scale. Complete excitation shells have exact quark-permutation closure.
C53 routing is kinematic only: no new amplitude, coupling or physical state
is selected. C45 polar-HO excited-state phase matching remains downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import factorial, prod, sqrt
from operator import index

import numpy as np
from scipy import sparse

from .m2_c117_pair_current import (
    color_permutation_matrix, qqq_longitudinal_basis,
)
from .m2_spin_flavor_s3_completion import spin_flavor_permutation_representation

S3 = tuple(permutations(range(3)))
_DEFAULT_TREES = {"qqq": (0, (1, 2)), "qqqg": ((0, 3), (1, 2))}


def _int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(name + " cannot be bool")
    return index(value)


def _sector(sector):
    if sector not in _DEFAULT_TREES:
        raise ValueError("sector must be qqq or qqqg")
    return 3 if sector == "qqq" else 4


def _modes(modes):
    result = tuple(_int(v, "mode") for v in modes)
    if len(result) not in (3, 4) or any(v <= 0 for v in result):
        raise ValueError("expected three/four positive doubled modes")
    if any(v % 2 != 1 for v in result[:3]) or (len(result) == 4 and result[3] % 2):
        raise ValueError("quarks require odd modes and the gluon an even mode")
    return result


def _permutation(p):
    p = tuple(_int(v, "permutation") for v in p)
    if p not in S3:
        raise ValueError("expected an S3 permutation")
    return p


def longitudinal_basis(sector, total_k2):
    """Ordered slot tuples, before spin/color/transverse/antisymmetry reduction."""
    _sector(sector)
    total = _int(total_k2, "total_k2")
    qqq = qqq_longitudinal_basis(total)
    if sector == "qqq":
        return qqq
    return tuple(sorted(
        state + (g,) for g in range(2, total-2, 2)
        for state in qqq_longitudinal_basis(total-g)
    ))


@dataclass(frozen=True)
class EmissionRoute:
    incoming: tuple[int, int, int]
    emitter: int
    outgoing: tuple[int, int, int, int]


def emission_routes(total_k2):
    """Allowed momentum routes, not a list of nonzero C53 amplitudes."""
    routes = []
    for incoming in longitudinal_basis("qqq", total_k2):
        for emitter in range(3):
            for gluon in range(2, incoming[emitter], 2):
                outgoing = list(incoming)
                outgoing[emitter] -= gluon
                routes.append(EmissionRoute(incoming, emitter, tuple(outgoing)+(gluon,)))
    return tuple(routes)


def emission_incidence(total_k2, emitter):
    """Dimensionless unit routing; its transpose is absorption routing ONLY."""
    emitter = _int(emitter, "emitter")
    if emitter not in range(3):
        raise ValueError("emitter outside quark slots")
    incoming, outgoing = (longitudinal_basis(s, total_k2) for s in ("qqq", "qqqg"))
    qi, gi = ({v: i for i, v in enumerate(b)} for b in (incoming, outgoing))
    selected = [r for r in emission_routes(total_k2) if r.emitter == emitter]
    return sparse.csr_matrix(
        (np.ones(len(selected)), ([gi[r.outgoing] for r in selected],
                                  [qi[r.incoming] for r in selected])),
        shape=(len(outgoing), len(incoming)),
    )


def _leaves(tree):
    if isinstance(tree, tuple):
        if len(tree) != 2:
            raise ValueError("each internal tree node must have two children")
        return _leaves(tree[0]) + _leaves(tree[1])
    return (_int(tree, "tree leaf"),)


@dataclass(frozen=True)
class JacobiFrame:
    modes: tuple[int, ...]
    tree: object
    fractions: np.ndarray
    forward: np.ndarray
    inverse: np.ndarray

    @property
    def transverse_measure_factor(self):
        """dmu_kappa = prod(x_i) prod_a d^2 Q_a (flat measures, no 2pi factors)."""
        return float(np.prod(self.fractions))

    def to_jacobi(self, momenta):
        """Physical parton momenta -> Q_1,...,Q_(N-1),P_T."""
        p = np.asarray(momenta, dtype=float)
        if p.shape != (len(self.modes), 2) or not np.all(np.isfinite(p)):
            raise ValueError("momenta must be a finite (N,2) array")
        return self.forward @ p

    def from_jacobi(self, coordinates):
        q = np.asarray(coordinates, dtype=float)
        if q.shape != (len(self.modes), 2) or not np.all(np.isfinite(q)):
            raise ValueError("coordinates must be a finite (N,2) array")
        return self.inverse @ q


def jacobi_frame(modes, tree=None):
    """Postorder binary-tree Q's with CM last; Q's diagonalize the free metric.

    At a split A|B of weight X=X_A+X_B:
      Q = (X_B p_A-X_A p_B)/sqrt(X_A X_B X).
    This is the daughter-normalized relative momentum divided by sqrt(X).
    """
    modes = _modes(modes)
    size = len(modes)
    tree = _DEFAULT_TREES["qqq" if size == 3 else "qqqg"] if tree is None else tree
    if sorted(_leaves(tree)) != list(range(size)):
        raise ValueError("tree must contain every constituent exactly once")
    x = np.asarray(modes, dtype=float) / sum(modes)
    rows = []

    def visit(node):
        if not isinstance(node, tuple):
            return (index(node),)
        left, right = visit(node[0]), visit(node[1])
        xa, xb = x[list(left)].sum(), x[list(right)].sum()
        row = np.zeros(size)
        row[list(left)] = sqrt(xb / (xa * (xa+xb)))
        row[list(right)] = -sqrt(xa / (xb * (xa+xb)))
        rows.append(row)
        return left+right

    visit(tree)
    forward = np.vstack(rows + [np.ones(size)])
    inverse = x[:, None] * forward.T
    for a in (x, forward, inverse):
        a.setflags(write=False)
    return JacobiFrame(modes, tree, x, forward, inverse)


def tree_change(modes, incoming_tree, outgoing_tree):
    """Orthogonal same-fraction intrinsic map; not a momentum-transfer kernel."""
    a, b = jacobi_frame(modes, incoming_tree), jacobi_frame(modes, outgoing_tree)
    return b.forward[:-1] @ a.inverse[:, :-1]


def permutation_jacobi_map(modes, permutation):
    """Q_out=O Q_in when physical partons and their fractions are relabeled."""
    modes, p = _modes(modes), _permutation(permutation)
    slots = p + ((3,) if len(modes) == 4 else ())
    outgoing = tuple(modes[i] for i in slots)
    a, b = jacobi_frame(modes), jacobi_frame(outgoing)
    return outgoing, b.forward[:-1] @ a.inverse[list(slots), :-1]


def emission_coordinate_map(route):
    """Q_out -> (parent's two Q's, global mass-weighted qg relative Q).

    This orthogonal kinematic map preserves both spectators and the free
    transverse metric. It neither evaluates nor normalizes a C53 amplitude.
    """
    incoming, outgoing = _modes(route.incoming), _modes(route.outgoing)
    emitter = _int(route.emitter, "emitter")
    if len(incoming) != 3 or len(outgoing) != 4 or emitter not in range(3):
        raise ValueError("invalid emission route")
    parent = list(outgoing[:3])
    parent[emitter] += outgoing[3]
    if tuple(parent) != incoming:
        raise ValueError("route fails spectator/parent momentum conservation")
    spectators = tuple(i for i in range(3) if i != emitter)
    a, b = jacobi_frame(incoming), jacobi_frame(outgoing)
    collapse = np.column_stack((np.eye(3), np.eye(3)[:, emitter]))
    split = jacobi_frame(outgoing, ((emitter, 3), spectators)).forward[0]
    return np.vstack((a.forward[:-1] @ collapse, split)) @ b.inverse[:, :-1]


def oscillator_shell(dimension, max_excitation):
    """Complete circular-HO occupations (+ per Jacobi mode, then -).

    sum(occupations)<=max_excitation; zero-point energy is NOT in this cutoff.
    Lz=sum(n_plus)-sum(n_minus). One common scale across all coordinates is
    required. No C45 excited-state phase crosswalk is implicit.
    """
    d, cutoff = _int(dimension, "dimension"), _int(max_excitation, "max_excitation")
    if d < 1 or cutoff < 0:
        raise ValueError("dimension must be positive and cutoff nonnegative")

    def compositions(total, length):
        if length == 1:
            yield (total,)
        else:
            for first in range(total+1):
                for rest in compositions(total-first, length-1):
                    yield (first,)+rest

    return tuple(state for n in range(cutoff+1) for state in compositions(n, 2*d))


def oscillator_rotation(orthogonal, max_excitation):
    """Normalized symmetric powers of O direct-sum O on complete HO shells."""
    o = np.asarray(orthogonal)
    if (o.ndim != 2 or o.shape[0] != o.shape[1] or not o.size
            or np.iscomplexobj(o) or not np.all(np.isfinite(o))):
        raise ValueError("expected a finite real square orthogonal matrix")
    if np.max(np.abs(o.T @ o - np.eye(len(o)))) > 2e-12:
        raise ValueError("a changed-fraction metric is not an orthogonal tree map")
    modes = oscillator_shell(len(o), max_excitation)
    lookup = {s: i for i, s in enumerate(modes)}
    transform = np.kron(np.eye(2), o)
    answer = np.zeros((len(modes), len(modes)))
    vacuum = (0,)*(2*len(o))
    for column, state in enumerate(modes):
        polynomial = {vacuum: 1.0}
        for j, count in enumerate(state):
            for _ in range(count):
                updated = {}
                for power, coefficient in polynomial.items():
                    for i in range(len(transform)):
                        if transform[i, j] == 0:
                            continue  # exact block/selection zero, no magnitude pruning
                        out = list(power)
                        out[i] += 1
                        key = tuple(out)
                        updated[key] = updated.get(key, 0.0) + coefficient*transform[i, j]
                polynomial = updated
        denominator = prod(factorial(v) for v in state)
        for power, coefficient in polynomial.items():
            answer[lookup[power], column] = coefficient*sqrt(
                prod(factorial(v) for v in power)/denominator)
    return answer


@lru_cache(maxsize=32)
def _longitudinal_orbital_action(sector, total_k2, max_excitation, permutation):
    basis = longitudinal_basis(sector, total_k2)
    dimension = _sector(sector)-1
    nmodes = len(oscillator_shell(dimension, max_excitation))
    lookup = {s: i for i, s in enumerate(basis)}
    rows, columns, values = [], [], []
    for column, state in enumerate(basis):
        outgoing, transform = permutation_jacobi_map(state, permutation)
        rotation = oscillator_rotation(transform, max_excitation)
        ri, ci = np.nonzero(rotation)
        rows.extend(lookup[outgoing]*nmodes+ri)
        columns.extend(column*nmodes+ci)
        values.extend(rotation[ri, ci])
    return sparse.csr_matrix((values, (rows, columns)),
                             shape=(len(basis)*nmodes,)*2)


def longitudinal_orbital_action(sector, total_k2, max_excitation, permutation):
    """Joint fraction-dependent S3 action, not a fixed tensor product."""
    _sector(sector)
    total = _int(total_k2, "total_k2")
    cutoff = _int(max_excitation, "max_excitation")
    p = _permutation(permutation)
    return _longitudinal_orbital_action(sector, total, cutoff, p).copy()


def total_sign_project(tensor, sector, total_k2, max_excitation):
    """Apply full S3 sign projector without constructing a dense huge matrix.

    Input axes: (longitudinal*orbital, C7 singlet color, 64 spin/flavor).
    Gluon helicity, when present, is a separately held fixed label.
    Does not select target, total Jz or I; those commute with this projector.
    """
    colors = 1 if _sector(sector) == 3 else 2
    n = len(longitudinal_basis(sector, total_k2))*len(
        oscillator_shell(_sector(sector)-1, max_excitation))
    value = np.asarray(tensor, dtype=np.complex128)
    if value.shape != (n, colors, 64) or not np.all(np.isfinite(value)):
        raise ValueError("tensor must have finite (longitudinal*orbital,color,64) axes")
    result = np.zeros_like(value)
    for p in S3:
        orbital = longitudinal_orbital_action(sector, total_k2, max_excitation, p)
        rotated = (orbital @ value.reshape(n, -1)).reshape(value.shape)
        color, sf = color_permutation_matrix(sector, p), spin_flavor_permutation_representation(p)
        sign = (-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        result += (sign/6)*np.einsum("ab,cs,nbs->nac", color, sf, rotated, optimize=True)
    return result


def intrinsic_ground_precision(modes, b_GeV):
    """Precision G in independent kappa_0..kappa_(N-2), flat d^2kappa measure."""
    b = float(b_GeV)
    if isinstance(b_GeV, (bool, np.bool_)) or not np.isfinite(b) or b <= 0:
        raise ValueError("b_GeV must be finite and positive")
    frame = jacobi_frame(modes)
    d = len(frame.modes)-1
    constrain = np.vstack((np.eye(d), -np.ones(d)))
    return (constrain.T / frame.fractions) @ constrain / b**2


def intrinsic_ground_overlap(incoming, outgoing, b_GeV):
    """Normalized transverse ground overlap at fixed physical intrinsic momenta.

    This strips off orthogonal longitudinal kets. It is NOT their Hilbert
    overlap, a C117 contact kernel, or an emission amplitude.
    """
    a, b = intrinsic_ground_precision(incoming, b_GeV), intrinsic_ground_precision(outgoing, b_GeV)
    if a.shape != b.shape:
        raise ValueError("ground overlap requires the same constituent count")
    _, la = np.linalg.slogdet(a)
    _, lb = np.linalg.slogdet(b)
    _, lab = np.linalg.slogdet(a+b)
    return float(np.exp(len(a)*np.log(2)+(la+lb)/2-lab))


def total_sign_weight_rank(sector, total_k2, max_excitation, target, jz_twice, gluon_helicity):
    """Rank of P_sign at fixed target Iz, I=1/2, total Jz and gluon helicity.

    This counts symmetry-allowed directions, not populated source amplitudes,
    an interacting Hamiltonian's dimension contract, or a selected state.
    Any half-integer Jz is allowed so vector generators can reach neighboring
    weights. Weights outside the finite shell have rank zero, not a fitted
    or silently truncated spin assignment.
    """
    from .m2_spin_flavor_s3_completion import (
        SPIN_FLAVOR_BASIS, build_spin_flavor_s3_algebra,
    )
    size = _sector(sector)
    jz2, hg = _int(jz_twice, "jz_twice"), _int(gluon_helicity, "gluon_helicity")
    if target not in ("PROTON", "NEUTRON") or jz2 % 2 != 1:
        raise ValueError("expected proton/neutron and odd integer twice-Jz")
    if hg not in ((0,) if size == 3 else (-1, 1)):
        raise ValueError("gluon helicity must be 0 in qqq, +/-1 in qqqg")
    basis = longitudinal_basis(sector, total_k2)
    shell = oscillator_shell(size-1, max_excitation)
    states = np.asarray(SPIN_FLAVOR_BASIS)
    iz2 = np.where(states < 2, 1, -1).sum(axis=1)
    sz2 = (2*(states % 2)-1).sum(axis=1)
    p_iso = (3.75*np.eye(64)-build_spin_flavor_s3_algebra().isospin_squared)/3
    target_iz = 1 if target == "PROTON" else -1
    rank = 0j
    for p in S3:
        sign = (-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        action = longitudinal_orbital_action(sector, total_k2, max_excitation, p)
        traces = action.diagonal().reshape(len(basis), len(shell)).sum(axis=0)
        sf_diag = np.diag(p_iso @ spin_flavor_permutation_representation(p))
        color_trace = np.trace(color_permutation_matrix(sector, p))
        for a, occupation in enumerate(shell):
            m = sum(occupation[:size-1])-sum(occupation[size-1:])
            mask = (iz2 == target_iz) & (2*m+sz2+2*hg == jz2)
            rank += sign*traces[a]*color_trace*sf_diag[mask].sum()/6
    result = int(round(rank.real))
    if abs(rank-result) > 1e-8:
        raise ArithmeticError("projector trace is not an integer at numerical precision")
    return result
