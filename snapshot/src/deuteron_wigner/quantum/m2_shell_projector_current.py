"""Straight-path oscillator-projector currents on full packed Q5/Q6 spaces.

The finite outer shell reports continuum translation matrix elements; it does
not truncate their internal path. This supplies an explicit transverse path
representative, not a physical regulator selection or continuum matching.
No raw orbital/color/spin amplitude is removed by a numerical threshold.
"""
from dataclasses import dataclass

import numpy as np

from .m2_canonical_good_current import parton_charge
from .m2_excluded_fock_kinematics import SPECIES as FIVE, frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX, frame as frame6
from .m2_galerkin_projector_current import straight_path_projector_vertex
from .m2_shell_current import shell_density_between


def inner_mask(space, cutoff):
    if (isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer))
            or not 0 <= cutoff < space.cutoff):
        raise ValueError("explicit inner cutoff strictly below the outer shell required")
    shell_number = np.sum(np.asarray(space.orbitals), axis=1)
    return shell_number[space.coordinates[:, 1]] <= cutoff


def _columns(space, values):
    a = np.asarray(values, complex)
    if a.ndim != 2 or a.shape[0] != space.dimension or not np.all(np.isfinite(a)):
        raise ValueError("finite packed columns on the named space required")
    return a


def _component(j):
    if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or j not in (0, 1):
        raise ValueError("explicit transverse component 0 or 1 required")
    return int(j)


@dataclass(frozen=True)
class _Block:
    incoming_rows: np.ndarray
    outgoing_rows: np.ndarray
    kernel: np.ndarray
    nspin: int


class ShellProjectorVertex:
    """K_j with Delta.K = Pi_f rho - rho Pi_i and K = Pi_f K + K Pi_i.

    Both external spaces are full outer shells, possibly with different Jz.
    Charge and path quadrature are explicit. K has units GeV^-1.
    """

    def __init__(self, outgoing, incoming, *, inner_cutoff, transfer, b_GeV,
                 charge, quadrature_order):
        for key in ("sector", "total_k2", "cutoff", "target"):
            if getattr(outgoing, key) != getattr(incoming, key):
                raise ValueError("projector endpoints disagree on "+key)
        if (outgoing.orbitals != incoming.orbitals
                or outgoing.longitudinal != incoming.longitudinal
                or not np.array_equal(outgoing.spins, incoming.spins)
                or outgoing.ncolor != incoming.ncolor):
            raise ValueError("identical ordered orbital/color/spin coordinates required")
        self.outgoing, self.incoming = outgoing, incoming
        self.mask_out, self.mask_in = inner_mask(outgoing, inner_cutoff), inner_mask(incoming, inner_cutoff)
        self.inner_cutoff = inner_cutoff
        delta, b = np.asarray(transfer, float), float(b_GeV)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b <= 0:
            raise ValueError("finite transfer and positive width required")
        self.transfer, self.width_GeV, self.charge = delta.copy(), b, charge
        s = incoming
        if s.sector in FIVE:
            species, frame = FIVE[s.sector], frame5
        elif s.sector in SIX:
            species, frame = SIX[s.sector], frame6
        else:
            raise ValueError("explicit five- or six-particle coordinates required")
        d, n = len(species)-1, len(s.orbitals)
        occ = np.asarray(s.orbitals)
        lz = occ[:, :d].sum(axis=1)-occ[:, d:].sum(axis=1)
        self.blocks = []
        for li, modes in enumerate(s.longitudinal):
            fr = frame(s.sector, modes)
            # Spin diagonal, but keep all flavor weights before projection.
            kernels = []
            for active, kind in enumerate(species):
                if kind == "g":
                    continue
                q = parton_charge(s.spins[:, active], kind, charge)
                k = straight_path_projector_vertex(d, inner_cutoff, s.cutoff,
                    fr.forward[:-1, active], delta, b_GeV=b, quadrature_order=quadrature_order)
                kernels.append((k, q))
            for m, oi, spins in s.symmetry_groups:
                rows = s.group_rows[li, m]
                mf = m+(outgoing.jz_twice-incoming.jz_twice)/2
                of = np.flatnonzero(lz == mf)
                target = np.flatnonzero((outgoing.coordinates[:, 0] == li)
                    &np.isin(outgoing.coordinates[:, 1], of)
                    &np.isin(outgoing.coordinates[:, 3], spins))
                if not len(target):
                    continue
                expected = np.array([(o, c, spin) for o in of
                                     for c in range(s.ncolor) for spin in spins])
                if not np.array_equal(outgoing.coordinates[target, 1:], expected):
                    raise ValueError("incomplete or reordered exact output spin block")
                # Only orbital/spin kernel storage, never a full raw matrix.
                kernel = sum(k[:, of][:, :, oi, None]*q[spins][None, None, None, :]
                             for k, q in kernels)
                self.blocks.append(_Block(rows, target, kernel, len(spins)))

    def apply(self, values, *, component):
        j = _component(component)
        a = _columns(self.incoming, values)
        result = np.zeros((self.outgoing.dimension, a.shape[1]), complex)
        nc = self.incoming.ncolor
        for block in self.blocks:
            v = a[block.incoming_rows].reshape(-1, nc, block.nspin, a.shape[1])
            out = np.einsum("ois,icsk->ocsk", block.kernel[j], v, optimize=True)
            result[block.outgoing_rows] += out.reshape(len(block.outgoing_rows), a.shape[1])
        return result

    def density(self, values):
        return shell_density_between(self.outgoing, self.incoming, values, self.transfer,
                                     b_GeV=self.width_GeV, charge=self.charge)


class QShellProjectorFamily:
    """Product-rule boundary for a named, potentially coupled Q5/Q6 parent.

    Incoming/outgoing Hamiltonian actions act on dictionaries containing ALL
    named sectors. Thus conversion and Q5<->Q6 terms are not replaced with
    diagonal-sector approximations. All outputs remain on full outer rows.
    """

    def __init__(self, outgoing, incoming, *, inner_cutoffs, transfer,
                 b_GeV, charge, quadrature_order):
        if not incoming or set(outgoing) != set(incoming) or set(inner_cutoffs) != set(incoming):
            raise ValueError("matching complete named spaces and explicit inner cutoffs required")
        self.outgoing, self.incoming = dict(outgoing), dict(incoming)
        for spaces in (self.outgoing, self.incoming):
            first = next(iter(spaces.values()))
            for name, space in spaces.items():
                if space.sector != name or any(getattr(space, key) != getattr(first, key)
                        for key in ("total_k2", "target", "jz_twice")):
                    raise ValueError("one common physical endpoint per named family required")
        self.vertices = {n: ShellProjectorVertex(outgoing[n], s,
            inner_cutoff=inner_cutoffs[n], transfer=transfer, b_GeV=b_GeV,
            charge=charge, quadrature_order=quadrature_order) for n, s in incoming.items()}

    def columns(self, values, *, endpoint):
        if endpoint not in ("incoming", "outgoing"):
            raise ValueError("explicit incoming or outgoing endpoint required")
        spaces = getattr(self, endpoint)
        if set(values) != set(spaces):
            raise ValueError("all and only named parent sectors required")
        a = {n: _columns(s, values[n]) for n, s in spaces.items()}
        if len({x.shape[1] for x in a.values()}) != 1:
            raise ValueError("same number of columns in every sector required")
        return a

    def project(self, values, *, endpoint):
        a = self.columns(values, endpoint=endpoint)
        masks = {n: (v.mask_in if endpoint == "incoming" else v.mask_out)
                 for n, v in self.vertices.items()}
        return {n: masks[n][:, None]*x for n, x in a.items()}

    def apply(self, values, *, component):
        a = self.columns(values, endpoint="incoming")
        return {n: self.vertices[n].apply(x, component=component) for n, x in a.items()}

    def density(self, values):
        a = self.columns(values, endpoint="incoming")
        return {n: self.vertices[n].density(x) for n, x in a.items()}

    def hamiltonian_boundary(self, values, *, parent_incoming, parent_outgoing,
                             component, total_pplus_GeV):
        """Pi_f (K H_i + H_f K) Pi_i /(2P+), including inter-sector H."""
        _component(component)
        pp = float(total_pplus_GeV)
        if not np.isfinite(pp) or pp <= 0:
            raise ValueError("finite positive common total P+ required")
        a = self.project(values, endpoint="incoming")
        hi = self.columns(parent_incoming(a), endpoint="incoming")
        kh = self.apply(hi, component=component)
        hk = self.columns(parent_outgoing(self.apply(a, component=component)), endpoint="outgoing")
        return self.project({n: (kh[n]+hk[n])/(2*pp) for n in kh}, endpoint="outgoing")

    def resolvent_boundary(self, values, *, resolve_incoming, resolve_outgoing,
                            parent_incoming, parent_outgoing, component, total_pplus_GeV):
        """Regulator part of the embedded resolvent's transverse current.

        [K R_i + R_f K]/(2P+) + R_f B_H R_i.
        R_a is the supplied INNER projected resolvent at its own energy.
        B_H is hamiltonian_boundary for the SAME parent. This omits the
        separately projected physical/parent current R_f J_parent R_i.
        Endpoint K terms must not be discarded. This does not resum or
        automatically change the existing order-g^4 H2 model.
        """
        _component(component)
        pp = float(total_pplus_GeV)
        if not np.isfinite(pp) or pp <= 0:
            raise ValueError("finite positive common total P+ required")
        a = self.columns(values, endpoint="incoming")
        ri = self.columns(resolve_incoming(a), endpoint="incoming")
        k_ri = self.apply(ri, component=component)
        rf_k = self.columns(resolve_outgoing(self.apply(a, component=component)), endpoint="outgoing")
        b_ri = self.hamiltonian_boundary(ri, parent_incoming=parent_incoming,
            parent_outgoing=parent_outgoing, component=component, total_pplus_GeV=pp)
        rf_b_ri = self.columns(resolve_outgoing(b_ri), endpoint="outgoing")
        return {n: (k_ri[n]+rf_k[n])/(2*pp)+rf_b_ri[n] for n in ri}
