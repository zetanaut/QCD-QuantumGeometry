"""Canonical quark Wick self terms, separately from propagating loops.

The many-body Galerkin cutoff is NOT a coincident-field trace regulator.
Here that trace and positive APBC/PBC field-mode counts are explicit inputs.
Only the quark one-body descendants of instantaneous fermion/gluon products
are included. Gluon tadpoles, propagating self energies, finite counterterms,
zero modes and a common renormalized QCD regulator are NOT completed here.

Source: Srivastava--Brodsky hep-ph/0011372 Eq.24, with symmetric LF fields.
The independent continuum check is 2504.07162 Appendix B, Eqs.101--104;
that paper's nonsymmetric p+ convention must be converted before comparison.
"""
from dataclasses import dataclass
from math import pi

import numpy as np
from scipy import sparse

from .m2_coupled_longitudinal_jacobi import longitudinal_basis, oscillator_shell

CF = 4 / 3


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def gaussian_coincident_trace(width_GeV):
    """Integral d²k/(2pi)² exp(-k²/Lambda²) = Lambda²/(4pi), GeV².

    This defines a translation-invariant contraction test member, not a
    claim that all canonical vertices already carry this same regulator.
    """
    width = float(width_GeV)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("positive finite contraction width required")
    return width * width / (4 * pi)


def quark_wick_sums(quark_k2, *, boson_modes, fermion_modes):
    """Dimensionless ordered Wick sums (instantaneous fermion, gluon).

    Bosons k2=2,4,...,2*N_b; fermions r2=1,3,...,2*N_f-1.
    IF: sum 1/[k2*(p2-k2)], including negative denominators.
    IG: sum (Q0/(p2-r2)² - 1/(p2+r2)²).
    Q0 removes only the exactly zero transfer; it does NOT remove the
    antiparticle subtraction at r2=p2. Vacuum identity is subtracted.
    No corresponding-propagating-graph restriction k2<p2 is imposed.
    """
    p = _integer(quark_k2, "quark_k2")
    nb = _integer(boson_modes, "boson_modes")
    nf = _integer(fermion_modes, "fermion_modes")
    if p % 2 != 1 or p > 2 * nf - 1:
        raise ValueError("external APBC quark must belong to the field regulator")
    k = 2 * np.arange(1, nb + 1, dtype=float)
    r = 2 * np.arange(nf, dtype=float) + 1
    inst_f = np.sum(1 / (k * (p - k)))
    direct = np.zeros(nf)
    nonzero = r != p
    direct[nonzero] = 1 / (p - r[nonzero]) ** 2
    inst_g = np.sum(direct - 1 / (p + r) ** 2)
    return float(inst_f), float(inst_g)


def antiquark_wick_sums(antiquark_k2, *, boson_modes, fermion_modes):
    """Ordered antiquark IF/IG self sums; vacuum identity subtracted.

    CAR reverses the negative IF constrained mode -(p2+k2), leaving
    +1/[k2*(p2+k2)]. The instantaneous-gluon current square has the same
    quark/antiquark one-body descendant, checked in the full CAR Fock space.
    This is the ordered composite Hamiltonian, not a C-symmetrized scheme.
    """
    p = _integer(antiquark_k2, "antiquark_k2")
    _, inst_g = quark_wick_sums(p, boson_modes=boson_modes, fermion_modes=fermion_modes)
    k = 2*np.arange(1, boson_modes+1, dtype=float)
    return float(np.sum(1/(k*(p+k)))), inst_g


def quark_wick_infinite_sums(quark_k2):
    """Analytic infinite-longitudinal-trace check at fixed external p2.

    Transverse UV divergence remains: these are NOT renormalized masses.
    IF=(sum_{j=1}^a 1/(2j-1)-log(2))/p2; IG=H_a^(2)/2,
    with p2=2a+1. In particular IF(p2=1)=-log(2), IG(p2=1)=0.
    """
    p = _integer(quark_k2, "quark_k2")
    if p % 2 != 1:
        raise ValueError("odd APBC quark mode required")
    a = (p - 1) // 2
    j = np.arange(1, a + 1, dtype=float)
    return float((np.sum(1 / (2 * j - 1)) - np.log(2)) / p), float(np.sum(1 / j**2) / 2)


def _project_longitudinal_scalar(pilot, value):
    """Project an exchange-symmetric scalar function of each ordered tuple."""
    blocks = []
    for sector, u in zip(("qqq", "qqqg", "qqqg"), pilot.bases):
        d = 2 if sector == "qqq" else 3
        nc = d - 1
        repeats = len(oscillator_shell(d, pilot.cutoff)) * nc * 64
        entries = np.repeat([value(s) for s in longitudinal_basis(sector, pilot.total_k2)], repeats)
        blocks.append((u.getH() @ sparse.diags(entries, format="csr") @ u).toarray())
    return sparse.block_diag(blocks, format="csr").toarray()


def quark_mass_squared_direction(pilot):
    """Exact one-body delta(m_q²) direction sum_i 1/x_i, both H2 sectors.

    This shifts the kinetic mass-squared operator only. It does not shift
    the independent vertex mass or assert a renormalization relation.
    """
    return _project_longitudinal_scalar(pilot, lambda s: sum(s) * sum(1 / p for p in s[:3]))


@dataclass(frozen=True)
class QuarkSelfContractions:
    pilot: object
    boson_modes: int
    fermion_modes: int
    coincident_trace_GeV2: float
    instantaneous_fermion_per_gs2: np.ndarray
    instantaneous_gluon_per_gs2: np.ndarray

    @property
    def total_per_gs2(self):
        return self.instantaneous_fermion_per_gs2 + self.instantaneous_gluon_per_gs2

    def normal_order_counterterm(self, *, finite_quark_mass_squared_GeV2):
        """Explicit subtraction scheme, with caller-supplied finite mass².

        Returns -g²*S_trace + delta(m_q²)*sum_i 1/x_i. Cancelling the
        displayed Wick trace defines normal ordering; it does NOT determine
        the remaining physical finite mass, vertex, or current conditions.
        Zero finite mass is an explicit test choice, not a default.
        """
        finite = float(finite_quark_mass_squared_GeV2)
        if not np.isfinite(finite):
            raise ValueError("finite, signed mass-squared coefficient required")
        return -self.pilot.coupling**2 * self.total_per_gs2 + finite * quark_mass_squared_direction(self.pilot)


def quark_self_contractions(pilot, *, boson_modes, fermion_modes, coincident_trace_GeV2):
    """Normal-order first, then exact coupled-basis Galerkin projection.

    delta M²/g² = C_F*K2/pi * I_trace * sum_quarks (F_IF+F_IG).
    Each displayed piece has GeV² units. No scalar sector shift is inferred.
    I_trace is the same explicit transverse trace for both Wick descendants.
    The omitted gluon self terms must not be inferred as zero.
    """
    nb = _integer(boson_modes, "boson_modes")
    nf = _integer(fermion_modes, "fermion_modes")
    trace = float(coincident_trace_GeV2)
    if not np.isfinite(trace) or trace < 0:
        raise ValueError("nonnegative finite coincident trace required")
    possible = {p for sector in ("qqq", "qqqg")
                for s in longitudinal_basis(sector, pilot.total_k2) for p in s[:3]}
    sums = {p: quark_wick_sums(p, boson_modes=nb, fermion_modes=nf) for p in possible}
    scale = CF * pilot.total_k2 / pi * trace
    matrices = [_project_longitudinal_scalar(pilot, lambda s, a=a: scale * sum(sums[p][a] for p in s[:3]))
                for a in range(2)]
    return QuarkSelfContractions(pilot, nb, nf, trace, *matrices)
