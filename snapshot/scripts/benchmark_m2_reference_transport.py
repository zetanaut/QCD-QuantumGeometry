"""Finite-reference cutoff matching, no fitted or activated physical inputs."""
import json

import numpy as np
from scipy import linalg

from deuteron_wigner.quantum.m2_canonical_quark_self import quark_mass_squared_direction
from deuteron_wigner.quantum.m2_canonical_gluon_self import gluon_mass_squared_direction
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
from deuteron_wigner.quantum.m2_reference_transport import FiniteReferenceTransport


def main():
    p = build_coupled_pilot(total_k2=9, cutoff=0, target="PROTON", jz_twice=1, b_GeV=.4,
                            quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    parents = [ExcludedFockReference(p, [build_excluded_source(p, sector, out_cutoff=n)
                for sector in ("qqqgg", "qqqq-qbar")]) for n in (0, 1)]
    low, high = parents
    flow = FiniteReferenceTransport(low.reduction, high.reduction, map_to_reference=np.eye(len(p.matrix)))
    state = high.reduction.lowest_state()
    energy = state.energy_GeV2
    directions = [np.eye(len(p.matrix)), quark_mass_squared_direction(p), gluon_mass_squared_direction(p),
                  p.emission_per_gs, p.cross_pairs_per_gs2, p.instantaneous_fermion_per_gs2]
    axes = np.column_stack([a.ravel()/np.linalg.norm(a) for a in directions])
    norm_axes = np.concatenate([axes.real, axes.imag])
    rows = []
    for anchor in (1., 1.4, 1.42):
        local = flow.local_counterterm(anchor)
        bare = local.value_GeV2.ravel()
        coefficients, _, rank, _ = np.linalg.lstsq(norm_axes, np.r_[bare.real, bare.imag], rcond=None)
        projection_residual = float(np.linalg.norm(bare-axes@coefficients))
        constant = low.matrix+low.reduction.p@local.value_GeV2@low.reduction.p.conj().T
        a, b = flow.generalized_lower_parent(anchor)
        approximate = float(linalg.eigvalsh(a, b)[0])
        rows.append(dict(anchor_GeV2=anchor, delta_H_norm_GeV2=float(np.linalg.norm(local.value_GeV2)),
                         delta_N_norm=float(np.linalg.norm(local.norm_counterterm)),
                         generous_six_direction_rank=int(rank),
                         counterterm_outside_six_directions_GeV2=projection_residual,
                         constant_only_energy_error_GeV2=float(linalg.eigvalsh(constant)[0]-energy),
                         energy_local_with_metric_error_GeV2=approximate-energy,
                         H_counterterm_remainder_at_reference_GeV2=float(np.linalg.norm(flow.hamiltonian_counterterm(energy)-local.at(energy), 2)),
                         conservative_remainder_bound_GeV2=flow.hamiltonian_remainder_bound(energy, anchor)))
    v = state.retained
    charge_without_correction = float(np.vdot(v, low.reduction.norm_metric(energy)@v).real)
    rho = [h.density((.12, .16), charge="electromagnetic") for h in parents]
    drho = flow.operator_counterterm(*rho, energy, energy)
    dh = flow.hamiltonian_counterterm(energy)
    dn = flow.norm_counterterm(energy)
    print(json.dumps(dict(scope="finite free-Q reference matching only; not continuum renormalization",
        parent_dimensions=[len(h.matrix) for h in parents], reference_M2_GeV2=energy,
        lower_M2_GeV2=float(linalg.eigvalsh(low.matrix)[0]), rows=rows,
        charge_with_high_metric_but_untransported_current=charge_without_correction,
        charge_missing_correction=float(np.vdot(v, dn@v).real),
        electromagnetic_missing_correction=float(np.vdot(v, drho@v).real),
        delta_H_eigenvalue_range=np.linalg.eigvalsh(dh)[[0, -1]].tolist(),
        delta_N_eigenvalue_range=np.linalg.eigvalsh(dn)[[0, -1]].tolist()), indent=2))


if __name__ == "__main__":
    main()
