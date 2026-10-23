"""Bare qq cross-pair action in a common-global-width Jacobi shell.

This is a Galerkin matrix of the number-preserving, normal-ordered distinct
quark pairs, per g_s^2. It is NOT the full C117 operator, a matched finite-C43
coordinate, or a replacement for the historical represented M2 Hamiltonian.
Spin/flavor and gluon helicity are identities; both C7 qqqg colors are kept.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial, pi, sqrt
from operator import index

import numpy as np
from scipy import sparse
from scipy.special import eval_genlaguerre, roots_genlaguerre

from .m2_c117_pair_current import QUARK_PAIRS, color_pair_matrix
from .m2_coupled_longitudinal_jacobi import (
    jacobi_frame, longitudinal_basis, oscillator_rotation, oscillator_shell,
)


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(name + " cannot be bool")
    result = index(value)
    if result < 0:
        raise ValueError(name + " must be nonnegative")
    return result


def _width(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("width must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("width must be finite and positive")
    return result


def circular_to_c45(n_plus, n_minus):
    """Return (radial n, angular m, phase).

    Unit flat-d2p circular wave = phase * C45 ho_momentum / (2*pi).
    a_+^dagger=(a_x^dagger+i*a_y^dagger)/sqrt(2); similarly minus.
    """
    p, m = _nonnegative(n_plus, "n_plus"), _nonnegative(n_minus, "n_minus")
    n = min(p, m)
    return n, p-m, (-1)**n


def radial_width_overlap(n_out, n_in, angular_m, b_out, b_in):
    """Circular-HO <out,b_out|in,b_in>, with one common angular momentum.

    Generalized Laguerre quadrature integrates the finite polynomial exactly
    (up to floating arithmetic), after absorbing BOTH Gaussian widths.
    This is a transverse overlap, not overlap of distinct longitudinal kets.
    """
    no, ni = _nonnegative(n_out, "n_out"), _nonnegative(n_in, "n_in")
    if isinstance(angular_m, (bool, np.bool_)):
        raise TypeError("angular_m cannot be bool")
    a = abs(index(angular_m))
    bo, bi = _width(b_out), _width(b_in)
    if bo == bi:
        return float(no == ni)
    scale = (bo**-2 + bi**-2)/2
    t, w = roots_genlaguerre((no+ni)//2+1, a)
    # Product polynomial has degree no+ni; 2*nodes-1 >= degree.
    norm = sqrt(factorial(no)*factorial(ni) /
                (factorial(no+a)*factorial(ni+a)))
    value = np.dot(w, eval_genlaguerre(no, a, t/(scale*bo*bo)) *
                   eval_genlaguerre(ni, a, t/(scale*bi*bi)))
    return float((-1)**(no+ni)*norm*value / ((bo*bi*scale)**(a+1)))


def ground_width_tail(b_out, b_in, max_excitation):
    """Exact omitted squared norm for a squeezed m=0 ground state.

    Only n_plus=n_minus=n contributes; excitation is 2*n.
    No finite-shell renormalization is performed.
    """
    bo, bi = _width(b_out), _width(b_in)
    cutoff = _nonnegative(max_excitation, "max_excitation")
    ratio = (bi*bi-bo*bo)/(bi*bi+bo*bo)
    return float(ratio**(2*(cutoff//2+1)))


def _pair(pair, size):
    p = tuple(_nonnegative(v, "pair slot") for v in pair)
    if len(p) != 2 or p[0] == p[1] or any(i >= size for i in p):
        raise ValueError("pair must contain two distinct constituent slots")
    rest = tuple(i for i in range(size) if i not in p)
    return p, (p, rest[0] if len(rest) == 1 else rest)


@lru_cache(maxsize=8192)
def _pair_rotation(modes, pair, cutoff):
    f = jacobi_frame(modes)
    _, tree = _pair(pair, len(modes))
    adapted = jacobi_frame(modes, tree)
    return oscillator_rotation(adapted.forward[:-1] @ f.inverse[:, :-1], cutoff)


@lru_cache(maxsize=64)
def _contact_shell(d, out_cutoff, in_cutoff):
    """Relative mode is first. Its Fourier-transformed circular HO at zero
    is b/sqrt(pi) for every n+=n-, and zero otherwise. Spectators are equal.
    The non-square form exposes finite-parent leakage without claiming closure.
    """
    outgoing = oscillator_shell(d, out_cutoff)
    incoming = oscillator_shell(d, in_cutoff)
    result = np.zeros((len(outgoing), len(incoming)))
    for i, a in enumerate(outgoing):
        if a[0] != a[d]:
            continue
        spectators = a[1:d] + a[d+1:]
        for j, b in enumerate(incoming):
            if b[0] == b[d] and b[1:d]+b[d+1:] == spectators:
                result[i, j] = 1.
    return result


def pair_contact_block(incoming, outgoing, pair, b_GeV, in_excitation,
                       out_excitation=None):
    """Exact transverse delta^2(r_i-r_j) Galerkin block, units GeV^2.

    Longitudinal pair sum and all spectators must be conserved. Transfer zero
    is allowed HERE to test the contact itself; the current kernel removes it.
    Labels use the default global Jacobi trees and circular oscillator shell.
    """
    a, b = jacobi_frame(incoming), jacobi_frame(outgoing)
    if len(a.modes) != len(b.modes) or sum(a.modes) != sum(b.modes):
        raise ValueError("pair contact requires the same sector and total K2")
    p, _ = _pair(pair, len(a.modes))
    if any(a.modes[i] != b.modes[i] for i in range(len(a.modes)) if i not in p):
        raise ValueError("pair contact cannot change spectator longitudinal modes")
    ci = _nonnegative(in_excitation, "in_excitation")
    co = ci if out_excitation is None else _nonnegative(out_excitation, "out_excitation")
    width = _width(b_GeV)
    i, j = p
    x, y = a.fractions, b.fractions
    alpha_in, alpha_out = x[i]*x[j]/(x[i]+x[j]), y[i]*y[j]/(y[i]+y[j])
    ri = _pair_rotation(a.modes, p, ci)
    ro = _pair_rotation(b.modes, p, co)
    return (width**2/pi * sqrt(alpha_in*alpha_out) *
            (ro.T @ _contact_shell(len(a.modes)-1, co, ci) @ ri))


def qq_contact_block(incoming, outgoing, pair, b_GeV, in_excitation,
                     out_excitation=None):
    """Quark-only entry point to the constituent-independent transverse contact."""
    p,_=_pair(pair,len(incoming))
    if any(i>=3 for i in p):
        raise ValueError("qq contact requires two quark slots")
    return pair_contact_block(incoming,outgoing,p,b_GeV,in_excitation,out_excitation)


def qq_pair_mass_per_gs2(sector, total_k2, b_GeV, in_excitation,
                         out_excitation=None, *, pair=None, _contact_kernel=None):
    """Sparse bare cross-pair matrix on (longitudinal, orbital, C7 color).

    Both ordered current products give K2/(2*pi*n^2); n=Delta(k_i,2)/2.
    No extra 1/x_parent, self-contraction, physical coupling or matching is
    implicit. A larger output cutoff is an explicitly regulated parent probe.
    Spin/flavor and fixed gluon helicity tensor factors remain identities.
    """
    basis = longitudinal_basis(sector, total_k2)
    ci = _nonnegative(in_excitation, "in_excitation")
    co = ci if out_excitation is None else _nonnegative(out_excitation, "out_excitation")
    width = _width(b_GeV)
    size = 3 if sector == "qqq" else 4
    nc = size-2
    ni, no = (len(oscillator_shell(size-1, n)) for n in (ci, co))
    pairs = QUARK_PAIRS if pair is None else (_pair(pair, size)[0],)
    if any(i>=3 for p in pairs for i in p):
        raise ValueError("qq mass requires two quark slots")
    lookup = {s: i for i, s in enumerate(basis)}
    rows, columns, values = [], [], []
    for col, state in enumerate(basis):
        for p in pairs:
            i, j = p
            color = color_pair_matrix(sector, i, j)
            for ki in range(1, state[i]+state[j], 2):
                transfer = (ki-state[i])//2
                if transfer == 0:
                    continue  # Q0, exact zero-transfer removal
                out = list(state)
                out[i], out[j] = ki, state[i]+state[j]-ki
                out = tuple(out)
                contact = (qq_contact_block if _contact_kernel is None else _contact_kernel)(state, out, p, width, ci, co)
                block = np.kron(contact, color)*sum(state)/(2*pi*transfer**2)
                r, c = np.nonzero(block)
                rows.extend(lookup[out]*no*nc+r)
                columns.extend(col*ni*nc+c)
                values.extend(block[r,c])
    return sparse.csr_matrix((values, (rows, columns)),
                             shape=(len(basis)*no*nc, len(basis)*ni*nc))
