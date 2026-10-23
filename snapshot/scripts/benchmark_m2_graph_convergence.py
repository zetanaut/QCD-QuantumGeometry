"""Actual controlled-gate convergence, independent of an exact-state seed."""
import argparse
import json
import time

import numpy as np
from scipy import linalg

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import ExcludedFockReference
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations",type=int,default=5000)
    parser.add_argument("--function-tolerance",type=float,default=0.)
    parser.add_argument("--gradient-tolerance",type=float,default=1e-10)
    parser.add_argument("--natural",action="store_true")
    parser.add_argument("--metric-rcond",type=float,default=1e-14)
    parser.add_argument("--residual-tolerance",type=float,default=1e-8)
    parser.add_argument("--spin-current",action="store_true")
    parser.add_argument("--quartic",action="store_true",help="include both canonical order-gs² pair exits")
    parser.add_argument("--target",choices=("PROTON","NEUTRON"),default="PROTON")
    args=parser.parse_args()
    p=build_coupled_pilot(total_k2=9,cutoff=0,target=args.target,jz_twice=1,
                         b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    def reference(pilot):
        if args.quartic:
            from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
            return build_order_resolved_reference(pilot,out_cutoff=1)
        return ExcludedFockReference(pilot,[build_excluded_source(pilot,s,out_cutoff=1)
                                            for s in ("qqqgg","qqqq-qbar")])
    ref=reference(p)
    model=ref.reduction.linearized_model(1.4)
    circuit=SymmetryGraphCircuit(model.matrix,root=int(np.argmin(model.matrix.diagonal().real)),
                                support_threshold=1e-12)
    initial=circuit.uniform_parameters()
    initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(circuit.edges))
    print("Starting actual six-qubit, 72-parameter graph optimization",flush=True)
    start=time.monotonic()
    if args.natural:
        from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit
        fit=quantum_natural_fit(circuit,initial,max_iterations=args.iterations,metric_rcond=args.metric_rcond,
                               residual_tolerance=args.residual_tolerance)
    else:
        fit=circuit.fit(initial,max_iterations=args.iterations,gradient_tolerance=args.gradient_tolerance,
                        function_tolerance=args.function_tolerance)
    # The independent exact eigensolve is AFTER optimization; never an initializer.
    energies,vectors=linalg.eigh(model.matrix)
    psi=np.asarray(circuit.state(fit.x))[:circuit.dimension]
    diagnostics=circuit.state_diagnostics(fit.x)
    diagnostics.update(optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
                       iterations=int(fit.nit),evaluations=int(fit.nfev),
                       gradient_inf=float(np.linalg.norm(fit.jac,np.inf)),
                       requested_gradient_tolerance_met=bool(np.linalg.norm(fit.jac,np.inf)<=args.gradient_tolerance),
                       energy_error_GeV2=float(fit.fun-energies[0]),
                       ground_overlap_squared=float(abs(np.vdot(vectors[:,0],psi))**2),
                       spectral_gap_GeV2=float(energies[1]-energies[0]),
                       algorithm="full projective metric" if args.natural else "centered L-BFGS-B",
                       residual_tolerance_met=bool(diagnostics["residual_GeV2"]<=args.residual_tolerance),
                       target=args.target,quartic_sources=args.quartic,parent_dimension=len(ref.matrix),
                       seconds=time.monotonic()-start,
                       scope="finite-model full-tree baseline; not scalable ansatz or quantum advantage")
    if args.spin_current:
        from deuteron_wigner.quantum.m2_spin_resolved_current import (
            transverse_parity_map,normalized_matched_good_current_between,
        )
        down=build_coupled_pilot(total_k2=9,cutoff=0,target=args.target,jz_twice=-1,
                         b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
        lower=reference(down)
        t=transverse_parity_map(down,p);delta=(.12,.16)
        diagonal=model.operator(ref.density(delta,charge="electromagnetic"))
        off=normalized_matched_good_current_between(lower,ref,1.4,1.4,delta,charge="electromagnetic")
        # Explicit positive-parity phase down=i T up. Reduce the transition
        # onto the up fiber before encoding; this is not a diagonal-only F2.
        pauli=-1j*2*np.sqrt(fit.fun)/(delta[0]+1j*delta[1])*t.conj().T@off
        ops=[(op+op.conj().T)/2 for op in (diagonal,pauli)]
        measured=circuit.measure_operators(fit.x,ops)
        direct=[float(np.vdot(psi,op@psi).real) for op in ops]
        reference=[float(np.vdot(vectors[:,0],op@vectors[:,0]).real) for op in ops]
        distance=energies[1]-diagnostics["energy_GeV2"]
        # Exact-arithmetic spectral residual bound, numerically evaluated
        # against this finite reference; not a physical uncertainty band.
        bounds=[2*np.linalg.norm(op,2)*diagnostics["residual_GeV2"]/distance for op in ops]
        diagnostics["spin_current"]=dict(qubits=circuit.qubits,F1_F2=measured.tolist(),
                   exact_model_F1_F2=reference,absolute_errors=np.abs(measured-reference).tolist(),
                   compact_measurement_difference=float(np.linalg.norm(measured-direct)),
                   spectral_state_error_bounds=bounds,
                   scope="same 1.4 GeV² energy-local reference; six-qubit graph gates, no StatePrep")
    print(json.dumps(diagnostics,indent=2),flush=True)


if __name__=="__main__":
    main()
