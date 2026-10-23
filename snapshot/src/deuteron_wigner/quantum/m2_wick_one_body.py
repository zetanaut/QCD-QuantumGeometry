"""One common set of ordered Wick/kinetic subtraction operators for H and J.

These are explicit finite-field-trace composites, not a continuum QCD
regulator or physical finite-condition selection. Finite mass parameters
below multiply g_s^2 and modify ONLY the kinetic mass-squared directions.
The vertex mass in the canonical constrained field is not changed.
"""
from dataclasses import dataclass

import numpy as np

from .m2_canonical_quark_self import (
    CF, _integer, _project_longitudinal_scalar, quark_wick_sums, antiquark_wick_sums,
)
from .m2_canonical_gluon_self import gluon_wick_sums
from .m2_current_wick_self import ordered_wick_coefficient
from .m2_canonical_good_current import parton_charge
from .m2_excluded_fock_kinematics import SPECIES as FIVE
from .m2_six_body_kinematics import SPECIES as SIX


@dataclass(frozen=True, kw_only=True)
class WickOneBody:
    boson_modes: int
    fermion_modes: int
    quark_flavors: int
    coincident_trace_GeV2: float
    normal_order_subtraction: bool
    finite_quark_mass2_per_gs2_GeV2: float
    finite_gluon_mass2_per_gs2_GeV2: float

    def __post_init__(self):
        # This validates all field counts including an explicit sea flavor count.
        gluon_wick_sums(2, boson_modes=self.boson_modes,
            fermion_modes=self.fermion_modes, quark_flavors=self.quark_flavors)
        if not isinstance(self.normal_order_subtraction, (bool, np.bool_)):
            raise ValueError("explicit Boolean normal-order subtraction choice required")
        values = [self.coincident_trace_GeV2, self.finite_quark_mass2_per_gs2_GeV2,
                  self.finite_gluon_mass2_per_gs2_GeV2]
        if not np.all(np.isfinite(values)) or self.coincident_trace_GeV2 < 0:
            raise ValueError("nonnegative finite trace and explicit signed finite mass² terms required")

    def _constituent(self, mode, total_k2):
        total = _integer(total_k2, "total_k2")
        p = _integer(mode, "constituent mode")
        if p >= total:
            raise ValueError("external constituent must have 0<p2<K2")
        return p, total

    def mass_components(self, species, mode, *, total_k2):
        """(ordered Wick, normal-order counterterm, finite kinetic) M²/g_s²."""
        p, total = self._constituent(mode, total_k2)
        args = dict(boson_modes=self.boson_modes, fermion_modes=self.fermion_modes)
        if species in ("q", "qbar"):
            function = quark_wick_sums if species == "q" else antiquark_wick_sums
            bare = CF*sum(function(p, **args))
            finite = self.finite_quark_mass2_per_gs2_GeV2*total/p
        elif species == "g":
            bare = sum(gluon_wick_sums(p, **args, quark_flavors=self.quark_flavors))
            finite = self.finite_gluon_mass2_per_gs2_GeV2*total/p
        else:
            raise ValueError("quark, antiquark or gluon required")
        bare *= total*self.coincident_trace_GeV2/np.pi
        return float(bare), float(-bare if self.normal_order_subtraction else 0.), float(finite)

    def retained_mass_coefficient(self, pilot):
        """Project the summed one-body M²/g_s² on the actual retained isometries."""
        def value(modes):
            species = ("q",)*3 + (("g",) if len(modes) == 4 else ())
            return sum(sum(self.mass_components(s, p, total_k2=pilot.total_k2))
                       for s, p in zip(species, modes))
        return _project_longitudinal_scalar(pilot, value)

    def shell_mass_diagonal(self, space):
        """Exact packed Q5/Q6 diagonal; no source-space/rank projection."""
        species = FIVE.get(space.sector, SIX.get(space.sector))
        if species is None:
            raise ValueError("explicit canonical five- or six-body sector required")
        values = np.array([sum(sum(self.mass_components(s, p, total_k2=space.total_k2))
                              for s, p in zip(species, modes))
                           for modes in space.longitudinal])
        return values[space.coordinates[:, 0]]

    def _loop_charges(self, values):
        charges = np.asarray(values, float)
        if charges.shape != (self.quark_flavors,) or not np.all(np.isfinite(charges)):
            raise ValueError("one explicit charge per Hamiltonian trace flavor required")
        return charges

    def minus_particle_weight(self, species, mode, spins, *, total_k2,
                               total_pplus_GeV, charge, loop_flavor_charges):
        """Spin-diagonal J-/(2P+) weight per g_s², physical charges included.

        Gauging the explicit quark kinetic mass² counterterm gives
        e_q*mu_q²/[2(p_q+)²]. A gluon kinetic mass has no bare Abelian charge.
        This selects no vertex-mass counterterm or transverse physical term.
        """
        p, total = self._constituent(mode, total_k2)
        loops = self._loop_charges(loop_flavor_charges)
        bare = ordered_wick_coefficient(species, p, total_k2=total,
            total_pplus_GeV=total_pplus_GeV, boson_modes=self.boson_modes,
            fermion_modes=self.fermion_modes, coincident_trace_GeV2=self.coincident_trace_GeV2,
            loop_flavor_charges=loops)
        weight = bare + (-bare if self.normal_order_subtraction else 0.)
        physical = parton_charge(spins, species, charge)  # validate charge even for gluons
        if species == "g":
            return np.full(np.shape(physical), weight, float)
        finite = self.finite_quark_mass2_per_gs2_GeV2*total**2/(2*p*p*float(total_pplus_GeV)**2)
        return physical*(weight+finite)
