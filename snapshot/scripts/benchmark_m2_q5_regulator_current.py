"""Finite-outer-shell Q5 loop regulator controls, without a physical-current claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_q5_loop_regulator_current import Q5LoopRegulatorCurrent
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between


def run(target, outer_cutoff):
    start = time.monotonic()
    def reference(jz):
        p = build_coupled_pilot(total_k2=9, cutoff=0, target=target, jz_twice=jz,
            b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
        return build_shell_fock_reference(p, out_cutoff=outer_cutoff)
    incoming = reference(1)
    records = []
    for jz in (1, -1):
        outgoing = incoming if jz == 1 else reference(jz)
        m = Q5LoopRegulatorCurrent(outgoing, incoming, inner_cutoff=0,
            energy_out=1.39, energy_in=1.47, transfer=(.12, .16),
            charge="electromagnetic", total_pplus_GeV=1.3, quadrature_order=20)
        reg = m.coefficients()
        plus = {p: np.zeros((37, 37), complex) for p in reg}
        tree = {p: np.zeros((2, 37, 37), complex) for p in reg}
        rhs = {p: np.zeros((37, 37), complex) for p in reg}
        for right in (1, 2):
            ri = {n: m.resolvents_in[n].resolve(1.47, a) for n, a in m.sources_in[right].items()}
            rho_ri, rho_v = m.projector.density(ri), m.projector.density(m.sources_in[right])
            rfrho = {n: m.resolvents_out[n].resolve(1.39, a) for n, a in rho_ri.items()}
            ward = {n: rho_ri[n]-m.resolvents_out[n].resolve(1.39, rho_v[n]) for n in ri}
            currents = [{n: m.resolvents_out[n].resolve(1.39,
                shell_transverse_current_between(m.projector.outgoing[n], m.projector.incoming[n],
                    a, (.12, .16), b_GeV=.4, total_pplus_GeV=1.3,
                    charge="electromagnetic", component=j)) for n, a in ri.items()} for j in (0, 1)]
            for left in (1, 2):
                v = m.sources_out[left]
                plus[left+right] += sum(v[n].conj().T@a for n, a in rfrho.items())
                rhs[left+right] += sum(v[n].conj().T@a for n, a in ward.items())
                for j in (0, 1):
                    tree[left+right][j] += sum(v[n].conj().T@a for n, a in currents[j].items())
        errors = {}
        missing = {}
        for p in reg:
            lhs = (1.39-1.47)*plus[p]-2*1.3*np.einsum("j,joi->oi", [.12, .16], tree[p]+reg[p])
            errors[p] = float(np.linalg.norm(lhs-rhs[p]))
            missing[p] = float(np.linalg.norm(lhs+2*1.3*np.einsum("j,joi->oi", [.12, .16], reg[p])-rhs[p]))
            if errors[p] > 3e-13:
                raise ArithmeticError("actual-source two-energy loop Ward identity failed")
        record = dict(outgoing_jz_twice=jz, incoming_jz_twice=1,
            raw_Q5_dimensions={n: s.dimension for n, s in m.projector.incoming.items()},
            regulator_coefficient_norms={p: [float(np.linalg.norm(a)) for a in reg[p]] for p in reg},
            normalized_by_retained_metric=False,
            summed_at_gs_point_norms=[float(np.linalg.norm(a)) for a in sum(.3**p*reg[p] for p in reg)],
            loop_Ward_residuals=errors, loop_Ward_residuals_if_regulator_is_omitted=missing)
        records.append(record)
        print("checked", target, "outer", outer_cutoff, "jz", jz, errors, flush=True)
    return dict(target=target, inner_Q5_cutoff=0, outer_Q5_cutoff=outer_cutoff,
        retained_cutoff=0, retained_dimension=37, total_k2=9,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3,
        energy_out_GeV2=1.39, energy_in_GeV2=1.47, total_pplus_GeV=1.3,
        transfer_GeV=[.12, .16], quadrature_order=20, records=records,
        scope="Q5 free-propagation V1/V2 loop regulator coefficients, before both retained norms",
        full_H2_order4_regulator_matching_complete=False,
        continuum_regulator_convergence_established=False,
        covariance_repaired=False, physical=False, quantum_execution=False,
        seconds=time.monotonic()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("PROTON", "NEUTRON"), required=True)
    parser.add_argument("--outer-cutoff", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.target, args.outer_cutoff)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
