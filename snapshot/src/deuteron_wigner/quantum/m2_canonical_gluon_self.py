"""Gluon one-body Wick descendants of the canonical constrained Hamiltonian.

Together with m2_canonical_quark_self these exhaust the quadratic, vacuum-
subtracted Wick descendants of the three O(g²) quartic sources in SB Eq.24
for two transverse polarizations. This is a contraction-regulated calculation,
NOT a complete common-regulator propagating loop or renormalized QCD model.
"""
from dataclasses import dataclass
from math import pi

import numpy as np

from .m2_canonical_quark_self import (
    _integer, _project_longitudinal_scalar, quark_mass_squared_direction,
    quark_self_contractions,
)
from .m2_coupled_longitudinal_jacobi import longitudinal_basis

CA = 3.
TF = .5


def gluon_wick_sums(gluon_k2, *, boson_modes, fermion_modes, quark_flavors):
    """Dimensionless (IF sea trace, IG adjoint current, local A^4 tadpole).

    All color/flavor factors included. Common multiplier K2*I_trace/pi
    gives M²/g². Positive internal fermions include both creation orderings
    through 1/(p-r)-1/(p+r); their difference is not a positive probability.
    The adjoint current has both number and Bose-pair Wick terms.
    """
    p = _integer(gluon_k2, "gluon_k2")
    nb = _integer(boson_modes, "boson_modes")
    nf = _integer(fermion_modes, "fermion_modes")
    if isinstance(quark_flavors, (bool, np.bool_)) or not isinstance(quark_flavors, (int, np.integer)) or quark_flavors < 0:
        raise ValueError("nonnegative integer quark flavor count required")
    if p % 2 != 0 or p > 2*nb:
        raise ValueError("external PBC gluon must belong to the field regulator")
    k = 2*np.arange(1, nb+1, dtype=float)
    r = 2*np.arange(nf, dtype=float)+1
    inst_f = TF*quark_flavors/p * np.sum(1/(p-r)-1/(p+r))
    direct = np.zeros(nb)
    nz = k != p
    direct[nz] = (p+k[nz])**2 / (4*p*k[nz]*(p-k[nz])**2)
    pair = (p-k)**2 / (4*p*k*(p+k)**2)
    inst_g = CA*np.sum(direct+pair)
    # Six Wick choices in A_b^i A_d^i A_c^j A_e^j give
    # 2*(d_T-1)*C_A*C_trace*A², with d_T=2 and
    # C_trace=I_trace/(2pi)*sum_k 1/k2.
    quartic = CA/(2*p)*np.sum(1/k)
    return float(inst_f), float(inst_g), float(quartic)


def gluon_mass_squared_direction(pilot):
    """Kinetic delta(m_g²)/x_g; zero in qqq, no vertex-mass identification."""
    return _project_longitudinal_scalar(pilot, lambda s: sum(s)/s[3] if len(s) == 4 else 0.)


@dataclass(frozen=True)
class GluonSelfContractions:
    pilot: object
    boson_modes: int
    fermion_modes: int
    quark_flavors: int
    coincident_trace_GeV2: float
    instantaneous_fermion_per_gs2: np.ndarray
    instantaneous_gluon_per_gs2: np.ndarray
    local_quartic_per_gs2: np.ndarray

    @property
    def total_per_gs2(self):
        return (self.instantaneous_fermion_per_gs2 +
                self.instantaneous_gluon_per_gs2 + self.local_quartic_per_gs2)


def gluon_self_contractions(pilot, *, boson_modes, fermion_modes, quark_flavors, coincident_trace_GeV2):
    trace = float(coincident_trace_GeV2)
    if not np.isfinite(trace) or trace < 0:
        raise ValueError("nonnegative finite coincident trace required")
    sums = {s[3]: gluon_wick_sums(s[3], boson_modes=boson_modes, fermion_modes=fermion_modes,
                                quark_flavors=quark_flavors)
            for s in longitudinal_basis("qqqg", pilot.total_k2)}
    scale = pilot.total_k2/pi*trace
    matrices = [_project_longitudinal_scalar(pilot, lambda s, a=a: scale*sums[s[3]][a] if len(s)==4 else 0.)
                for a in range(3)]
    return GluonSelfContractions(pilot, boson_modes, fermion_modes, quark_flavors, trace, *matrices)


@dataclass(frozen=True)
class CanonicalWickSelf:
    quark: object
    gluon: GluonSelfContractions

    @property
    def total_per_gs2(self):
        return self.quark.total_per_gs2 + self.gluon.total_per_gs2

    def normal_order_counterterm(self, *, finite_quark_mass_squared_GeV2, finite_gluon_mass_squared_GeV2):
        """Declare a subtraction, not a physical fit. Both finite terms explicit."""
        q, g = map(float, (finite_quark_mass_squared_GeV2, finite_gluon_mass_squared_GeV2))
        if not np.all(np.isfinite([q, g])):
            raise ValueError("finite signed mass-squared coefficients required")
        p = self.quark.pilot
        return (-p.coupling**2*self.total_per_gs2
                + q*quark_mass_squared_direction(p) + g*gluon_mass_squared_direction(p))


def canonical_wick_self(pilot, *, boson_modes, fermion_modes, quark_flavors, coincident_trace_GeV2):
    """Common contraction inputs; the original partial pilot remains unchanged."""
    args = dict(boson_modes=boson_modes, fermion_modes=fermion_modes,
                coincident_trace_GeV2=coincident_trace_GeV2)
    return CanonicalWickSelf(quark_self_contractions(pilot, **args),
                            gluon_self_contractions(pilot, quark_flavors=quark_flavors, **args))
