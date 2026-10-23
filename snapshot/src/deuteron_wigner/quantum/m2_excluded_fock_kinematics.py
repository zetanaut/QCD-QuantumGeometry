"""Explicit qqqgg/qqqq-qbar intermediate kinematics and complete color spaces.

These are intermediate states for matched H2, not a selected higher-Fock
physical eigenproblem. Positive doubled modes, common-width global Jacobi
coordinates and full ordered slots precede Bose/Fermi projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import sqrt
from operator import index

import numpy as np

from deuteron_wigner.microscopic.h0.color import (
    ColorSingletBasis, representation_generators,
)
from deuteron_wigner.pilot.color import structure_constants
from .m2_coupled_longitudinal_jacobi import JacobiFrame

SPECIES = {
    "qqqg": ("q", "q", "q", "g"),
    "qqqgg": ("q", "q", "q", "g", "g"),
    "qqqq-qbar": ("q", "q", "q", "q", "qbar"),
}
TREES = {"qqqg": ((0, 3), (1, 2)),
         "qqqgg": (((0, 3), 4), (1, 2)),
         "qqqq-qbar": (((0, 3), 4), (1, 2))}


def _integer(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("exact integer required")
    return index(value)


def _modes(sector, modes):
    species = SPECIES[sector]
    modes = tuple(_integer(v) for v in modes)
    if len(modes) != len(species) or any(
        v <= 0 or v % 2 != int(s != "g") for v, s in zip(modes, species)
    ):
        raise ValueError("positive odd fermion/even gluon doubled modes required")
    return modes


@lru_cache(maxsize=None)
def longitudinal_states(sector, total_k2):
    species = SPECIES[sector]
    total = _integer(total_k2)
    minimum = tuple(2 if s == "g" else 1 for s in species)
    extra = total-sum(minimum)
    if extra < 0 or extra % 2:
        return ()

    def compositions(n, d):
        if d == 1:
            yield (n,)
        else:
            for a in range(n+1):
                for rest in compositions(n-a, d-1):
                    yield (a,)+rest

    return tuple(tuple(a+2*b for a,b in zip(minimum, rest))
                 for rest in compositions(extra//2, len(species)))


@lru_cache(maxsize=4096)
def frame(sector, modes):
    modes = _modes(sector, modes)
    x = np.array(modes, float)/sum(modes)
    rows = []

    def visit(tree):
        if not isinstance(tree, tuple):
            return (tree,)
        left, right = map(visit, tree)
        a,b = x[list(left)].sum(), x[list(right)].sum()
        row = np.zeros(len(x))
        row[list(left)] = sqrt(b/(a*(a+b)))
        row[list(right)] = -sqrt(a/(b*(a+b)))
        rows.append(row)
        return left+right

    visit(TREES[sector])
    forward = np.vstack(rows+[np.ones(len(x))])
    inverse = x[:,None]*forward.T
    for a in (x,forward,inverse):
        a.setflags(write=False)
    return JacobiFrame(modes,TREES[sector],x,forward,inverse)


@dataclass(frozen=True)
class SplitRoute:
    channel: str
    incoming: tuple
    outgoing: tuple
    parent: int

    @property
    def sector(self):
        return "qqqq-qbar" if self.channel == "g_to_pair" else "qqqgg"

    @property
    def daughters(self):
        return (self.parent, 4)


@lru_cache(maxsize=32)
def split_routes(total_k2, channel):
    if channel not in ("q_to_qg", "g_to_gg", "g_to_pair"):
        raise ValueError("unknown cubic source channel")
    routes = []
    for incoming in longitudinal_states("qqqg", total_k2):
        slots = range(3) if channel == "q_to_qg" else (3,)
        for parent in slots:
            # Appended particle is a gluon except for pair creation.
            first = 1 if channel == "g_to_pair" else 2
            for appended in range(first,incoming[parent],2):
                out = list(incoming)
                out[parent] -= appended
                routes.append(SplitRoute(channel,incoming,tuple(out)+(appended,),parent))
    return tuple(routes)


def split_frame(route):
    """Q_out -> (three unchanged parent Q coordinates, new pair-relative Q)."""
    a,b = frame("qqqg",route.incoming),frame(route.sector,route.outgoing)
    if route not in split_routes(sum(route.incoming),route.channel):
        raise ValueError("invalid route")
    collapse = np.column_stack((np.eye(4),np.eye(4)[:,route.parent]))
    i,j = route.daughters
    xi,xj = b.fractions[[i,j]]
    split = np.zeros(5)
    split[i],split[j] = sqrt(xj/(xi*(xi+xj))),-sqrt(xi/(xj*(xi+xj)))
    return np.vstack((a.forward[:-1]@collapse,split))@b.inverse[:,:-1]


def permutation_frame(sector, modes, permutation):
    p = tuple(_integer(v) for v in permutation)
    species = SPECIES[sector]
    if sorted(p) != list(range(len(species))) or any(species[i] != species[j] for i,j in enumerate(p)):
        raise ValueError("permutation must preserve species")
    outgoing = tuple(modes[i] for i in p)
    a,b = frame(sector,modes),frame(sector,outgoing)
    return outgoing,b.forward[:-1]@a.inverse[list(p),:-1]


def _act(matrix,tensor,slot):
    return np.moveaxis(np.tensordot(matrix,tensor,axes=(1,slot)),0,slot)


@lru_cache(maxsize=3)
def quark_emission_tensor(slot):
    if slot not in range(3):
        raise ValueError("quark emitter outside 0,1,2")
    source = ColorSingletBasis.construct("qqqg").tensors.reshape(3,3,3,8,2)
    t = np.array(representation_generators("3"))
    raw = np.moveaxis(np.tensordot(t,source,axes=(2,slot)),1,slot+1)
    return np.moveaxis(raw,0,-2).reshape(1728,2)


@lru_cache(maxsize=2)
def color_basis(sector):
    if sector == "qqqq-qbar":
        return ColorSingletBasis.construct(sector).tensors
    if sector != "qqqgg":
        raise ValueError("expected excluded sector")
    # qqq=1+8+8+10; 8x8=1+8+8+10+10bar+27 -> six singlets.
    candidates = np.column_stack([quark_emission_tensor(i) for i in range(3)])
    u,s,_ = np.linalg.svd(candidates,full_matrices=False)
    if sum(s>1e-11) != 6:
        raise ArithmeticError("qqqgg invariant source span is incomplete")
    # Canonical projector pivots without a 1728-by-1728 dense projector.
    basis = []
    for row in range(len(u)):
        v = u@u[row,:].conj()
        for previous in basis:
            v -= previous*np.vdot(previous,v)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v /= norm
            pivot = np.flatnonzero(abs(v)>1e-10)[0]
            v *= np.exp(-1j*np.angle(v[pivot]))
            basis.append(v)
        if len(basis) == 6:
            break
    result = np.column_stack(basis)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=5)
def color_source(channel, emitter=3):
    source = ColorSingletBasis.construct("qqqg").tensors.reshape(3,3,3,8,2)
    if channel == "q_to_qg":
        raw = quark_emission_tensor(emitter)
        sector = "qqqgg"
    elif channel == "g_to_gg":
        # REAL f(out_g1,out_g2,in_g); the derivative i belongs to the vertex.
        raw = np.einsum("abc,ijkcn->ijkabn",structure_constants(),source).reshape(1728,2)
        sector = "qqqgg"
    elif channel == "g_to_pair":
        raw = np.einsum("aIJ,ijkan->ijkIJn",
                        np.array(representation_generators("3")),source).reshape(243,2)
        sector = "qqqq-qbar"
    else:
        raise ValueError("unknown cubic source")
    basis = color_basis(sector)
    reduced = basis.conj().T@raw
    if np.linalg.norm(raw-basis@reduced)>2e-11:
        raise ArithmeticError("cubic source leaves complete color-singlet space")
    return reduced


@lru_cache(maxsize=32)
def color_permutation(sector, permutation):
    species = SPECIES[sector]
    permutation_frame(sector,tuple(2 if s=="g" else 1 for s in species),permutation)
    dims = tuple(8 if s=="g" else 3 for s in species)
    basis = color_basis(sector)
    transformed = basis.reshape(dims+(basis.shape[1],)).transpose(tuple(permutation)+(len(dims),))
    return basis.conj().T@transformed.reshape(basis.shape)


def color_generator_residual(sector):
    species = SPECIES[sector]
    dims = tuple(8 if s=="g" else 3 for s in species)
    reps = tuple("8" if s=="g" else "3bar" if s=="qbar" else "3" for s in species)
    u = color_basis(sector).reshape(dims+(-1,))
    return max(np.linalg.norm(sum(_act(representation_generators(rep)[a],u,i)
               for i,rep in enumerate(reps))) for a in range(8))
