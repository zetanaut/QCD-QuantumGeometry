"""Spin-resolved retained J+ and optional exact PennyLane interference oracle."""
import argparse
import json

import numpy as np
from scipy import linalg, sparse

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_spin_resolved_current import (
    transverse_parity_map, parity_aligned_partner, good_form_factor_diagnostic,
    normalized_matched_good_current_between,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum", action="store_true")
    parser.add_argument("--matched", action="store_true", help="retain N_exc=0 and match both free-Q sectors at N_exc=1")
    args = parser.parse_args()
    rows = []
    delta = np.array([.12, .16]); q = np.linalg.norm(delta)
    for target in ("PROTON", "NEUTRON"):
        for cutoff in ((0,) if args.matched else (0, 1)):
            ps = [build_coupled_pilot(total_k2=9, cutoff=cutoff, target=target, jz_twice=j, b_GeV=.4,
                                      quark_mass_GeV=.35, gluon_mass_GeV=0., coupling=.3) for j in (1, -1)]
            if args.matched:
                from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
                from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
                from deuteron_wigner.quantum.m2_matched_h2_reduction import _inverse_sqrt
                refs=[ExcludedFockReference(p,[build_excluded_source(p,s,out_cutoff=1)
                      for s in ("qqqgg","qqqq-qbar")]) for p in ps]
                root=refs[0].reduction.lowest_state()
                e=np.array([root.energy_GeV2])
                retained=(root.retained,parity_aligned_partner(ps[1],ps[0],root.retained,intrinsic_parity=1))
                states=tuple(np.linalg.solve(_inverse_sqrt(r.reduction.norm_metric(e[0])),v) for r,v in zip(refs,retained))
                density=[[normalized_matched_good_current_between(rf,ri,e[0],e[0],delta,charge="electromagnetic")
                          for ri in refs] for rf in refs]
                hamiltonians=[r.reduction.linearized_model(e[0]).matrix for r in refs]
            else:
                e, v = linalg.eigh(ps[0].matrix, subset_by_index=[0, 0])
                states = (v[:, 0], parity_aligned_partner(ps[1], ps[0], v[:, 0], intrinsic_parity=1))
                density = [[good_current_density_between(pf, pi, delta, charge="electromagnetic") for pi in ps] for pf in ps]
                hamiltonians=[p.matrix for p in ps]
            t = transverse_parity_map(ps[1], ps[0])
            matrix = np.array([[states[f].conj()@density[f][i]@states[i] for i in range(2)] for f in range(2)])
            fit = good_form_factor_diagnostic(matrix, delta, mass_GeV=np.sqrt(e[0]), total_pplus=3.)
            row = dict(target=target, retained_excitation=cutoff, helicity_dimensions=[len(p.matrix) for p in ps],
                       M2_GeV2=float(e[0]), F1=[fit.dirac.real, fit.dirac.imag], F2=[fit.pauli.real, fit.pauli.imag],
                       structure_residual=float(np.linalg.norm(fit.residual_matrix)),
                       model="matched two free-Q sectors" if args.matched else "bare retained",
                       parity_H_residual_GeV2=float(np.linalg.norm(hamiltonians[1]@t-t@hamiltonians[0])),
                       lower_helicity_state_residual_GeV2=float(np.linalg.norm(hamiltonians[1]@states[1]-e[0]*states[1])))
            if args.matched:
                row["reference_parent_dimensions"]=[len(r.matrix) for r in refs]
            if args.quantum:
                import pennylane as qml
                wave = np.r_[states[0], (delta[1]-1j*delta[0])/q*states[1]]/np.sqrt(2)
                rho = np.block(density); ops = [(rho+rho.conj().T)/2, (rho-rho.conj().T)/(2j)]
                qubits = int(np.ceil(np.log2(len(wave)))); padding = 2**qubits-len(wave)
                encoded_ops = [sparse.block_diag((sparse.csr_matrix(op), sparse.csr_matrix((padding,padding))), format="csr") for op in ops]
                encoded_wave = np.pad(wave, (0,padding))
                device = qml.device("lightning.qubit", wires=qubits)
                @qml.qnode(device)
                def circuit():
                    qml.StatePrep(encoded_wave, wires=range(qubits))
                    return tuple(qml.expval(qml.SparseHamiltonian(op, wires=range(qubits))) for op in encoded_ops)
                measured = np.array(circuit())
                direct = [np.vdot(wave, op@wave).real for op in ops]
                row["quantum"] = dict(qubits=qubits, cosine_sine=measured.tolist(),
                                     classical_difference=float(np.linalg.norm(measured-direct)),
                                     F2_from_sine=float(measured[1]*2*np.sqrt(e[0])/q),
                                     padding_norm=float(np.linalg.norm(encoded_wave[len(wave):])),
                                     scope="exact StatePrep oracle, not variational or scalable execution")
            rows.append(row)
    print(json.dumps(dict(scope="explicit finite-reference local J+ spin diagnostic; not physical nucleon form factors",
                          transfer_GeV=delta.tolist(), rows=rows), indent=2))


if __name__ == "__main__":
    main()
