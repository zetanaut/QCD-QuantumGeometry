"""Reproducible H2 Q-regulator probe matrices; not form factors or quantum runs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import (
    FactorizedCanonicalQQSourceAction, ShellCanonicalQQInsertion)
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2
from deuteron_wigner.quantum.m2_interacting_q_loop_regulator import (
    InteractingQLoopRegulatorCurrent, RETURN_WORDS)


def build(target, cutoff, jz):
    p = build_coupled_pilot(total_k2=9, cutoff=0, target=target, jz_twice=jz,
        b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3)
    r = build_shell_fock_reference(p, out_cutoff=cutoff)
    qq = ShellCanonicalQQInsertion(r, action=FactorizedCanonicalQQSourceAction(r))
    return FourthOrderH2(r, QQ_insertion=qq, six_body=CanonicalSixBodyExcursion(r, out_cutoff=cutoff))


def dot(a, b):
    return sum(a[n].conj().T@b[n] for n in a)


def run(target, outer_cutoff):
    if outer_cutoff < 1:
        raise ValueError("outer shell must be strictly larger than inner shell zero")
    start = time.monotonic()
    incoming = build(target, outer_cutoff, 1)
    inner_i = build(target, 0, 1)
    rng = np.random.default_rng(20260907)
    # Independent reproducible retained probes, not eigenstates or truncated
    # source-image bases. Every raw intermediate coordinate is still used.
    vi = np.linalg.qr(rng.normal(size=(37, 3))+1j*rng.normal(size=(37, 3)))[0]
    vf = np.linalg.qr(rng.normal(size=(37, 3))+1j*rng.normal(size=(37, 3)))[0]
    records = []
    for jz in (1, -1):
        outgoing = incoming if jz == 1 else build(target, outer_cutoff, jz)
        inner_f = inner_i if jz == 1 else build(target, 0, jz)
        m = InteractingQLoopRegulatorCurrent(outgoing, incoming,
            inner_cutoffs={n: 0 for n in (*incoming.spaces5, *incoming.spaces6)},
            energy_out=1.39, energy_in=1.47, transfer=(.12, .16), charge="electromagnetic",
            total_pplus_GeV=1.3, quadrature_order=20)
        # Verify H and norm on these probes against independently built inner
        # H2. Reuse its verified local whiteners for the reported current.
        errors = {}
        snaps = {}
        for endpoint, inner, vectors, energy in (
                ("outgoing", inner_f, vf, 1.39), ("incoming", inner_i, vi, 1.47)):
            snap = inner.snapshot(energy)
            snaps[endpoint] = snap
            src = m._source_columns(endpoint, vectors)
            h, norm = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
            for l, r, order, word in RETURN_WORDS:
                h += .3**(l+r+order)*dot(src[l], m.word_action(word, src[r], endpoint=endpoint))
                for k, op in enumerate(word):
                    if op == "R":
                        norm += .3**(l+r+order)*dot(src[l],
                            m.word_action(word[:k]+("R",)+word[k:], src[r], endpoint=endpoint))
            errors[endpoint] = dict(
                H=float(np.linalg.norm(h-vectors.conj().T@(snap.effective_hamiltonian-inner.pilot.matrix)@vectors)),
                norm=float(np.linalg.norm(norm-vectors.conj().T@(snap.metric-np.eye(37))@vectors)))
            if max(errors[endpoint].values()) > 2e-13:
                raise ArithmeticError("outer-word H/norm lost an inner canonical contribution")
        c = m.coefficients(outgoing_vectors=vf, incoming_vectors=vi)
        cw = m.coefficients(outgoing_vectors=snaps["outgoing"].whitener@vf,
                            incoming_vectors=snaps["incoming"].whitener@vi)
        # Separate order-four interacting insertion from the R-only returns,
        # including coherent direct/iterated Q6 terms and both backflow legs.
        sf, si = m._source_columns("outgoing", vf), m._source_columns("incoming", vi)
        parts = {}
        for word in (("R", "U", "R"), ("R", "W", "R"), ("R", "U", "R", "U", "R")):
            result = np.zeros((2, 3, 3), complex)
            for l, r, order, w in RETURN_WORDS:
                if w != word or l+r+order != 4:
                    continue
                for j in (0, 1):
                    result[j] += dot(sf[l], m.gauged_word(word, si[r],
                        vertices={"R": lambda a, j=j: m.regulator_action(a, component=j)}))
            parts[" ".join(word)] = [float(np.linalg.norm(x)) for x in result]
        records.append(dict(incoming_jz_twice=1, outgoing_jz_twice=jz,
            raw_dimensions={n: s.dimension for n, s in m.projector.incoming.items()},
            H_norm_reference_errors=errors,
            raw_coefficient_probe_norms={p: [float(np.linalg.norm(x)) for x in a] for p, a in c.items()},
            order4_interacting_word_probe_norms=parts,
            summed_raw_probe_norms=[float(np.linalg.norm(x)) for x in sum(.3**p*a for p, a in c.items())],
            summed_two_norm_probe_norms=[float(np.linalg.norm(x)) for x in sum(.3**p*a for p, a in cw.items())]))
        print("checked", target, outer_cutoff, jz, errors, parts, flush=True)
    return dict(target=target, outer_cutoff=outer_cutoff, inner_Q5_Q6_cutoff=0,
        retained_cutoff=0, retained_dimension=37, total_k2=9, b_GeV=.4,
        quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3,
        energy_out_GeV2=1.39, energy_in_GeV2=1.47, total_pplus_GeV=1.3,
        transfer_GeV=[.12, .16], quadrature_order=20,
        probes=dict(seed=20260907, count_per_endpoint=3,
                    full_retained_matrix_norms=False, eigenstates=False),
        records=records, seconds=time.monotonic()-start,
        scope="canonical H2 Q-only projector return current through g4 with both local norms",
        Q_only_H2_order4_transport_complete=True,
        full_H2_order4_regulator_matching_complete=False,
        common_field_regulator_complete=False, physical=False,
        continuum_convergence_established=False, quantum_execution=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("PROTON", "NEUTRON"), required=True)
    parser.add_argument("--outer-cutoff", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.target, args.outer_cutoff)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
