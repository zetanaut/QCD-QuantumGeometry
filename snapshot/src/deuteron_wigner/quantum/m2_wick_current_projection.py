"""Exact local Wick/kinetic-minus-current projection on retained and raw shells.

The same declared WickOneBody determines both H and these current weights.
No Wick term is identified with a tree two-gluon contact; no source tail is
compressed. The returned matrices/actions are coefficients per g_s^2.
"""
import numpy as np
from scipy import sparse

from .m2_coupled_longitudinal_jacobi import oscillator_shell, longitudinal_basis, jacobi_frame
from .m2_galerkin_projector_current import _rectangular_displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS
from .m2_excluded_fock_kinematics import SPECIES as FIVE, frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX, frame as frame6


def retained_wick_minus_between(outgoing, incoming, transfer, *, operators,
                                 total_pplus_GeV, charge, loop_flavor_charges):
    for key in ("total_k2", "cutoff", "target", "width_GeV", "quark_mass_GeV",
                "gluon_mass_GeV", "coupling"):
        if getattr(outgoing, key) != getattr(incoming, key):
            raise ValueError("Wick-current endpoints disagree on "+key)
    delta = np.asarray(transfer, float)
    if delta.shape != (2,) or not np.all(np.isfinite(delta)):
        raise ValueError("finite transverse transfer required")
    sf = np.array(SPIN_FLAVOR_BASIS)
    p = incoming
    reduced = []
    for sector, uf, ui in zip(("qqq", "qqqg", "qqqg"), outgoing.bases, incoming.bases):
        d = 2 if sector == "qqq" else 3
        nc, n = d-1, len(oscillator_shell(d, p.cutoff))
        species = ("q",)*3 + (("g",) if sector == "qqqg" else ())
        blocks = []
        for modes in longitudinal_basis(sector, p.total_k2):
            frame = jacobi_frame(modes)
            block = sparse.csr_matrix((n*nc*64,)*2, dtype=complex)
            for active, (kind, mode) in enumerate(zip(species, modes)):
                spins = sf[:, active] if kind != "g" else np.zeros(64, int)
                weight = operators.minus_particle_weight(kind, mode, spins,
                    total_k2=p.total_k2, total_pplus_GeV=total_pplus_GeV,
                    charge=charge, loop_flavor_charges=loop_flavor_charges)
                shift = frame.forward[:-1, active, None]*delta[None, :]
                orbital = _rectangular_displacement(d, p.cutoff, p.cutoff, shift, p.width_GeV)
                block += sparse.kron(orbital, sparse.diags(np.tile(weight, nc)), format="csr")
            blocks.append(block)
        full = sparse.block_diag(blocks, format="csr")
        reduced.append((uf.getH()@full@ui).toarray())
    return sparse.block_diag(reduced, format="csr").toarray()


def shell_wick_minus_between(outgoing, incoming, values, transfer, *, operators,
                              b_GeV, total_pplus_GeV, charge, loop_flavor_charges):
    """Apply the exact one-body current to all supplied raw Q5/Q6 columns."""
    for key in ("sector", "total_k2", "cutoff", "target"):
        if getattr(outgoing, key) != getattr(incoming, key):
            raise ValueError("Wick-current endpoints disagree on "+key)
    if outgoing.orbitals != incoming.orbitals:
        raise ValueError("identical ordered oscillator coordinates required")
    s = incoming
    a = np.asarray(values, complex)
    delta, b = np.asarray(transfer, float), float(b_GeV)
    if a.ndim != 2 or a.shape[0] != s.dimension or not np.all(np.isfinite(a)):
        raise ValueError("finite raw incoming columns required")
    if delta.shape != (2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b <= 0:
        raise ValueError("finite transfer and positive width required")
    if s.sector in FIVE:
        species, frame = FIVE[s.sector], frame5
    elif s.sector in SIX:
        species, frame = SIX[s.sector], frame6
    else:
        raise ValueError("explicit canonical Q5/Q6 coordinates required")
    d = len(species)-1
    occupations = np.array(s.orbitals)
    lz = occupations[:, :d].sum(axis=1)-occupations[:, d:].sum(axis=1)
    result = np.zeros((outgoing.dimension, a.shape[1]), complex)
    for li, modes in enumerate(s.longitudinal):
        f = frame(s.sector, modes)
        for active, kind in enumerate(species):
            weight = operators.minus_particle_weight(kind, modes[active], s.spins[:, active],
                total_k2=s.total_k2, total_pplus_GeV=total_pplus_GeV,
                charge=charge, loop_flavor_charges=loop_flavor_charges)
            shift = f.forward[:-1, active, None]*delta[None, :]
            density = _rectangular_displacement(d, s.cutoff, s.cutoff, shift, b)
            for m in np.unique(lz):
                oi = np.flatnonzero(lz == m)
                rows = np.flatnonzero((s.coordinates[:, 0] == li)
                                     &np.isin(s.coordinates[:, 1], oi))
                if not len(rows):
                    continue
                spins = np.unique(s.coordinates[rows, 3])
                mf = m+(outgoing.jz_twice-incoming.jz_twice)/2
                of = np.flatnonzero(lz == mf)
                target = np.flatnonzero((outgoing.coordinates[:, 0] == li)
                    &np.isin(outgoing.coordinates[:, 1], of)
                    &np.isin(outgoing.coordinates[:, 3], spins))
                if not len(target):
                    continue
                if len(rows) != len(oi)*s.ncolor*len(spins) or len(target) != len(of)*s.ncolor*len(spins):
                    raise ValueError("incomplete exact orbital/color/spin block")
                block = a[rows].reshape(len(oi), s.ncolor, len(spins), a.shape[1])
                block = block*weight[spins][None, None, :, None]
                value = density[np.ix_(of, oi)]@block.reshape(len(oi), -1)
                result[target] += value.reshape(len(target), a.shape[1])
    return result
