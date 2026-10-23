"""Finite P/Q oscillator-current controls; no eigenstate or physical-current claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmark_m2_interacting_q_loop_regulator import build
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_h2_projector_current import H2ProjectorCurrent
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import InteractingQLoopRegulatorCurrent
from deuteron_wigner.quantum.m2_wick_h2_matching import WickMatchedFourthOrderH2
from deuteron_wigner.quantum.m2_wick_one_body import WickOneBody
from deuteron_wigner.quantum.m2_canonical_quark_self import gaussian_coincident_trace
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between


def run(target):
    start = time.monotonic()
    op = WickOneBody(boson_modes=8, fermion_modes=8, quark_flavors=2,
        coincident_trace_GeV2=gaussian_coincident_trace(.8), normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011, finite_gluon_mass2_per_gs2_GeV2=-.007)
    models = {e: WickMatchedFourthOrderH2(build(target, 2, j), operators=op)
              for e, j in (("outgoing", -1), ("incoming", 1))}
    q = InteractingQLoopRegulatorCurrent(models["outgoing"], models["incoming"],
        inner_cutoffs={n: 1 for n in (*models["incoming"].base.spaces5, *models["incoming"].base.spaces6)},
        energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
        total_pplus_GeV=1.3, quadrature_order=20)
    parents = {e: build_coupled_pilot(total_k2=9, cutoff=1, target=target, jz_twice=j,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
        for e, j in (("outgoing", -1), ("incoming", 1))}
    c = H2ProjectorCurrent(q, parent_outgoing=parents["outgoing"], parent_incoming=parents["incoming"],
                             quadrature_order=20)
    rng = np.random.default_rng(20260907)
    def probes(count):
        return np.linalg.qr(rng.normal(size=(37, count))+1j*rng.normal(size=(37, count)))[0]
    vf, vi = probes(2), probes(1)
    snaps, errors = {}, {}
    for e, j, energy, v in (("outgoing", -1, 1.39, vf), ("incoming", 1, 1.47, vi)):
        inner = WickMatchedFourthOrderH2(build(target, 1, j), operators=op)
        snap = inner.snapshot(energy).canonical_snapshot
        snaps[e] = snap
        p = c.embeddings[e]
        h = c.parent_return_bilinears(endpoint=e, left=p@v, right=p@v)
        n = c.parent_return_bilinears(endpoint=e, left=p@v, right=p@v, norm=True)
        expected_h = v.conj().T@(snap.effective_hamiltonian-inner.pilot.matrix-.3**2*inner.pp)@v
        expected_n = v.conj().T@(snap.metric-np.eye(37))@v
        errors[e] = dict(H_probe=float(np.linalg.norm(sum(.3**k*a for k, a in h.items())-expected_h)),
                         norm_probe=float(np.linalg.norm(sum(.3**k*a for k, a in n.items())-expected_n)))
        if max(errors[e].values()) > 3e-13:
            raise ArithmeticError("independent inner H2 reference mismatch")
        print("inner reference", target, e, errors[e], flush=True)
    records = {}
    for normalized in (False, True):
        f = snaps["outgoing"].whitener@vf if normalized else vf
        i = snaps["incoming"].whitener@vi if normalized else vi
        args = dict(outgoing_vectors=f, incoming_vectors=i)
        pc, qc, retained = c.loop_coefficients(**args), q.coefficients(**args), c.retained_current(**args)
        total = retained+sum(.3**k*(pc[k]+qc[k]) for k in pc)
        records["two_norm" if normalized else "raw"] = dict(
            retained_H_probe_norms=[float(np.linalg.norm(a)) for a in retained],
            P_return_coefficient_probe_norms={k: [float(np.linalg.norm(a)) for a in v] for k, v in pc.items()},
            Q_return_coefficient_probe_norms={k: [float(np.linalg.norm(a)) for a in v] for k, v in qc.items()},
            summed_probe_norms=[float(np.linalg.norm(a)) for a in total])
        if not normalized:
            raw_pc = pc
        print("current", target, normalized, records["two_norm" if normalized else "raw"], flush=True)
    pf, pi = c.embeddings["outgoing"], c.embeddings["incoming"]
    rho = good_current_density_between(parents["outgoing"], parents["incoming"], (.12, .16),
                                       charge="electromagnetic")
    rp = pf.conj().T@rho@pi
    a = c.parent_return_bilinears(endpoint="outgoing", left=pf@vf, right=pf@rp@vi)
    b = c.parent_return_bilinears(endpoint="incoming", left=pi@rp.conj().T@vf, right=pi@vi)
    d = c.parent_return_bilinears(endpoint="outgoing", left=pf@vf, right=rho@pi@vi)
    e = c.parent_return_bilinears(endpoint="incoming", left=rho.conj().T@pf@vf, right=pi@vi)
    ward = {k: float(np.linalg.norm(2*1.3*(.12*v[0]+.16*v[1])-a[k]+b[k]+d[k]-e[k]))
            for k, v in raw_pc.items()}
    if max(ward.values()) > 3e-13:
        raise ArithmeticError("finite-parent P-return Ward failure")
    return dict(target=target, total_k2=9, retained_inner_cutoff=0, retained_outer_cutoff=1,
        Q_inner_cutoffs={n: 1 for n in q.projector.incoming},
        Q_outer_cutoffs={n: s.cutoff for n, s in q.projector.incoming.items()},
        raw_Q_dimensions={n: s.dimension for n, s in q.projector.incoming.items()},
        incoming_jz_twice=1, outgoing_jz_twice=-1, b_GeV=.4, quark_mass_GeV=.35,
        gluon_mass_GeV=0., coupling=.3, energy_out_GeV2=1.39, energy_in_GeV2=1.47,
        transfer_GeV=[.12, .16], total_pplus_GeV=1.3, quadrature_order=20,
        Wick=dict(boson_modes=8, fermion_modes=8, quark_flavors=2,
            coincident_trace_GeV2=op.coincident_trace_GeV2, Gaussian_width_GeV=.8,
            normal_order_subtraction=True, finite_quark_mass2_per_gs2_GeV2=.011,
            finite_gluon_mass2_per_gs2_GeV2=-.007,
            field_regulation_of_propagating_vertices=False),
        probes=dict(seed=20260907, outgoing_count=2, incoming_count=1, eigenstates=False,
                    full_retained_matrix_norms=False),
        inner_reference_errors=errors, P_return_Ward_residual_norms=ward, currents=records,
        scope="paired-Wick H2 finite oscillator P-H/P-return/Q-return correction through g4",
        finite_P_Q_oscillator_projector_transport_complete=True,
        common_field_regulator_complete=False, complete_physical_four_current=False,
        physical=False, continuum_convergence_established=False, quantum_execution=False,
        seconds=time.monotonic()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("PROTON", "NEUTRON"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.target)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
