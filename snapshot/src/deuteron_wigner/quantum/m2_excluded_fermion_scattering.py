"""Canonical number-preserving fermion-current scattering within five-body Q.

All distinct qq and q-antiquark t-channel pairs, per g_s², on the actual
packed coordinates. Antiquark charge is -T^T, not an extra fitted sign.
This is NOT the annihilation channel, full QQ Hamiltonian, or renormalization.
No projection back onto the old free-source image is applied here.
"""
from functools import lru_cache
from itertools import combinations
from math import pi, sqrt

import numpy as np
from scipy import sparse

from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_coupled_longitudinal_jacobi import oscillator_rotation
from .m2_coupled_pair_action import _contact_shell, _pair, _width
from .m2_excluded_fock_kinematics import SPECIES, color_basis, frame, _act


@lru_cache(maxsize=20, typed=True)
def fermion_pair_color(sector, left, right):
    """U† T_i.T_j U, with the anti-fundamental generator from canonical CAR."""
    species = SPECIES[sector]
    pair, _ = _pair((left, right), len(species))
    if any(species[i] == "g" for i in pair):
        raise ValueError("two fermion slots required")
    dims = tuple(8 if s == "g" else 3 for s in species)
    u = color_basis(sector)
    tensor = u.reshape(dims + (u.shape[1],))
    reps = ["3bar" if species[i] == "qbar" else "3" for i in pair]
    raw = sum(_act(a, _act(b, tensor, right), left) for a, b in zip(
        representation_generators(reps[0]), representation_generators(reps[1])))
    raw = raw.reshape(u.shape)
    reduced = u.conj().T @ raw
    if np.linalg.norm(raw-u@reduced) > 2e-12:
        raise ArithmeticError("fermion scattering leaves complete singlet carrier")
    reduced.setflags(write=False)
    return reduced


@lru_cache(maxsize=8192)
def _pair_rotation(sector, modes, pair, cutoff):
    f = frame(sector, modes)
    pair, _ = _pair(pair, len(modes))
    rest = tuple(i for i in range(len(modes)) if i not in pair)
    # First coordinate is the interacting pair; the other three describe
    # the unchanged spectators and total pair versus spectator cluster.
    tree = (pair, ((rest[0], rest[1]), rest[2]))
    rows = []

    def visit(node):
        if not isinstance(node, tuple):
            return (node,)
        a, b = map(visit, node)
        xa, xb = f.fractions[list(a)].sum(), f.fractions[list(b)].sum()
        row = np.zeros(5)
        row[list(a)], row[list(b)] = sqrt(xb/(xa*(xa+xb))), -sqrt(xa/(xb*(xa+xb)))
        rows.append(row)
        return a+b

    visit(tree)
    rotation = np.array(rows) @ f.inverse[:, :-1]
    if np.linalg.norm(rotation@rotation.T-np.eye(4)) > 2e-12:
        raise ArithmeticError("five-body pair Jacobi map is not orthogonal")
    return oscillator_rotation(rotation, cutoff)


def fermion_contact_block(sector, incoming, outgoing, pair, b_GeV, cutoff):
    """Five-body transverse delta contact with conserved longitudinal spectators."""
    fi, fo = frame(sector, tuple(incoming)), frame(sector, tuple(outgoing))
    pair, _ = _pair(pair, 5)
    if sum(incoming) != sum(outgoing) or any(incoming[i] != outgoing[i]
                                            for i in range(5) if i not in pair):
        raise ValueError("pair sum and spectators must be conserved")
    if any(SPECIES[sector][i] == "g" for i in pair):
        raise ValueError("two fermion slots required")
    width = _width(b_GeV)
    i, j = pair
    ai = fi.fractions[i]*fi.fractions[j]/(fi.fractions[i]+fi.fractions[j])
    ao = fo.fractions[i]*fo.fractions[j]/(fo.fractions[i]+fo.fractions[j])
    ri = _pair_rotation(sector, fi.modes, pair, cutoff)
    ro = _pair_rotation(sector, fo.modes, pair, cutoff)
    return width**2/pi*sqrt(ai*ao)*(ro.T@_contact_shell(4, cutoff, cutoff)@ri)


def fermion_scattering_per_gs2(space, *, b_GeV):
    """Sparse t-channel normal-ordered j_q^+ j_q^+ on exact Iz/Jz rows.

    Both current orders give 2 K2/[pi (Delta k2)^2]. Q0 is removed at
    exactly zero longitudinal transfer, including transverse rearrangement.
    Spin/flavor is an identity in ordered slots; summing all distinct pairs
    preserves inherited quark antisymmetry without an additional factorial.
    """
    _width(b_GeV)
    fermions = [i for i, s in enumerate(SPECIES[space.sector]) if s != "g"]
    lookup = {m: i for i, m in enumerate(space.longitudinal)}
    norb, nc, ns = len(space.orbitals), space.ncolor, space.nspin
    rows, cols, values = [], [], []
    spins = np.arange(ns)
    for li, incoming in enumerate(space.longitudinal):
        for i, j in combinations(fermions, 2):
            color = fermion_pair_color(space.sector, i, j)
            for ki in range(1, incoming[i]+incoming[j], 2):
                transfer = ki-incoming[i]
                if transfer == 0:
                    continue
                outgoing = list(incoming)
                outgoing[i], outgoing[j] = ki, incoming[i]+incoming[j]-ki
                outgoing = tuple(outgoing)
                contact = fermion_contact_block(
                    space.sector, incoming, outgoing, (i, j), b_GeV, space.cutoff)
                block = np.kron(contact, color)*2*space.total_k2/(pi*transfer**2)
                a, b = np.nonzero(block)
                raw_rows = (lookup[outgoing]*norb*nc+a[:, None])*ns+spins
                raw_cols = (li*norb*nc+b[:, None])*ns+spins
                packed_rows, packed_cols = space.lookup[raw_rows], space.lookup[raw_cols]
                keep = (packed_rows >= 0) & (packed_cols >= 0)
                outside = (packed_rows < 0) & (packed_cols >= 0)
                elements = np.broadcast_to(block[a, b][:, None], keep.shape)
                if np.linalg.norm(elements[outside]) > 2e-12:
                    raise ArithmeticError("scattering violates exact Iz/Jz")
                rows.append(packed_rows[keep])
                cols.append(packed_cols[keep])
                values.append(elements[keep])
    if not values:
        return sparse.csr_matrix((space.dimension, space.dimension), dtype=complex)
    return sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(space.dimension, space.dimension)).tocsr()

