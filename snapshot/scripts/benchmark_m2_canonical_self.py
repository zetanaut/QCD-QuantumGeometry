"""Explicit nonphysical canonical Wick-trace and subtraction diagnostics."""
import json

import numpy as np

from deuteron_wigner.quantum.m2_canonical_gluon_self import (
    canonical_wick_self, gluon_mass_squared_direction,
)
from deuteron_wigner.quantum.m2_canonical_quark_self import (
    gaussian_coincident_trace, quark_mass_squared_direction,
)
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot


def main():
    pilot = build_coupled_pilot(total_k2=9, cutoff=0, target="PROTON", jz_twice=1,
                               b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    mass = quark_mass_squared_direction(pilot)
    gmass = gluon_mass_squared_direction(pilot)
    directions = np.column_stack([mass.ravel(), gmass.ravel()])
    rho = good_current_density(pilot, (.12, .16), charge="electromagnetic")
    rows = []
    for modes, width in ((4, .8), (12, .4), (12, .8), (12, 1.6), (48, .8)):
        obj = canonical_wick_self(pilot, boson_modes=modes, fermion_modes=modes,
                                  quark_flavors=2, coincident_trace_GeV2=gaussian_coincident_trace(width))
        self = obj.total_per_gs2
        coefficients = np.linalg.lstsq(directions, self.ravel(), rcond=None)[0]
        residual = np.linalg.norm(self.ravel()-directions@coefficients)
        shifted = pilot.matrix+pilot.coupling**2*self
        # Explicit zero finite terms here only test the normal-order identity.
        ct = obj.normal_order_counterterm(finite_quark_mass_squared_GeV2=0., finite_gluon_mass_squared_GeV2=0.)
        rows.append(dict(field_modes_each=modes, gaussian_trace_width_GeV=width,
                         quark_IF_norm_GeV2=float(np.linalg.norm(obj.quark.instantaneous_fermion_per_gs2)),
                         quark_IG_norm_GeV2=float(np.linalg.norm(obj.quark.instantaneous_gluon_per_gs2)),
                         gluon_IF_norm_GeV2=float(np.linalg.norm(obj.gluon.instantaneous_fermion_per_gs2)),
                         gluon_IG_norm_GeV2=float(np.linalg.norm(obj.gluon.instantaneous_gluon_per_gs2)),
                         gluon_A4_norm_GeV2=float(np.linalg.norm(obj.gluon.local_quartic_per_gs2)),
                         total_self_norm_GeV2=float(np.linalg.norm(self)),
                         two_kinetic_mass_span_residual_GeV2=float(residual),
                         unsubtracted_lowest_M2_GeV2=float(np.linalg.eigvalsh(shifted)[0]),
                         zero_finite_subtraction_residual_GeV2=float(np.linalg.norm(shifted+ct-pilot.matrix)),
                         good_current_commutator_norm_GeV2=float(np.linalg.norm(self@rho-rho@self))))
    print(json.dumps(dict(scope="contraction-regulated one-body Wick diagnostic only; not complete renormalized QCD",
                          baseline_lowest_M2_GeV2=float(np.linalg.eigvalsh(pilot.matrix)[0]),
                          rows=rows), indent=2))


if __name__ == "__main__":
    main()
