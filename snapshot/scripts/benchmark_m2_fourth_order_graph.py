"""Actual graph solve and matched spin currents for the fourth-order H2 model."""
import argparse,json,time
import numpy as np
from scipy import linalg

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_order_resolved_reference import build_order_resolved_reference
from deuteron_wigner.quantum.m2_canonical_qq import CanonicalQQSourceAction,CanonicalQQInsertion
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2,current_between
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target",choices=("PROTON","NEUTRON"),default="PROTON")
    parser.add_argument("--shell-cutoff",type=int,default=None,
        help="Explicit complete-shell raw reference; omission preserves the cutoff1 compressed control.")
    parser.add_argument("--geometry",choices=("dense","tree"),default="dense")
    parser.add_argument("--probability-regularization",type=float,default=None,
        help="Required for tree geometry; zero selects its exact interior inverse.")
    args=parser.parse_args()
    def build(jz):
        p=build_coupled_pilot(total_k2=9,cutoff=0,target=args.target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
        if args.shell_cutoff is None:
            cutoff=1
            ref=build_order_resolved_reference(p,out_cutoff=cutoff)
            qq=CanonicalQQInsertion(ref,action=CanonicalQQSourceAction(ref))
        else:
            from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
            from deuteron_wigner.quantum.m2_shell_canonical_qq import (
                FactorizedCanonicalQQSourceAction,ShellCanonicalQQInsertion,
            )
            cutoff=args.shell_cutoff
            ref=build_shell_fock_reference(p,out_cutoff=cutoff)
            qq=ShellCanonicalQQInsertion(ref,action=FactorizedCanonicalQQSourceAction(ref))
        six=CanonicalSixBodyExcursion(ref,out_cutoff=cutoff)
        return FourthOrderH2(ref,QQ_insertion=qq,six_body=six)
    start=time.monotonic()
    print("Constructing the order-consistent H2 snapshot",flush=True)
    upper=build(1);s=upper.snapshot(1.4)
    construction=time.monotonic()-start
    circuit=SymmetryGraphCircuit(s.matrix,root=int(np.argmin(s.matrix.diagonal().real)),support_threshold=1e-12)
    initial=circuit.uniform_parameters()
    initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(circuit.edges))
    print("Solving with actual graph gates; no exact-state seed",flush=True)
    start=time.monotonic()
    fit=quantum_natural_fit(circuit,initial,max_iterations=5000,
        metric_rcond=1e-18 if args.geometry=="dense" else None,residual_tolerance=1e-8,
        geometry=args.geometry,probability_regularization=args.probability_regularization)
    fit_seconds=time.monotonic()-start
    # Independent exact solve occurs AFTER optimization, not as an initializer.
    ev,vec=linalg.eigh(s.matrix)
    psi=np.asarray(circuit.state(fit.x))[:circuit.dimension]
    diagnostics=circuit.state_diagnostics(fit.x)
    print("Matching the opposite-helicity current",flush=True)
    lower=build(-1);sf=lower.snapshot(1.4)
    t=transverse_parity_map(lower.pilot,upper.pilot)
    delta=(.12,.16)
    diagonal=s.current(delta,charge="electromagnetic")
    off=current_between(sf,s,delta,charge="electromagnetic",normalized=True)
    # Explicit positive-parity phase down=i T up; no eigensolver phase alignment.
    pauli=-1j*2*np.sqrt(fit.fun)/(delta[0]+1j*delta[1])*t.conj().T@off
    operators=[(op+op.conj().T)/2 for op in (diagonal,pauli)]
    measured=circuit.measure_operators(fit.x,operators)
    exact=[float(np.vdot(vec[:,0],op@vec[:,0]).real) for op in operators]
    direct=[float(np.vdot(psi,op@psi).real) for op in operators]
    distance=ev[1]-diagnostics["energy_GeV2"]
    bounds=[2*np.linalg.norm(op,2)*diagnostics["residual_GeV2"]/distance for op in operators]
    diagnostics.update(target=args.target,scope="bare fourth-order finite-sector H2 model, energy-local at1.4; not renormalized QCD",
        shell_cutoff=1 if args.shell_cutoff is None else args.shell_cutoff,
        complete_shell_propagation=args.shell_cutoff is not None,
        geometry=args.geometry,probability_regularization=args.probability_regularization,
        five_body_dimensions={n:s.dimension for n,s in upper.spaces5.items()},
        six_body_dimensions={n:s.dimension for n,s in upper.spaces6.items()},
        construction_seconds=construction,fit_seconds=fit_seconds,
        iterations=int(fit.nit),evaluations=int(fit.nfev),optimizer_message=str(fit.message),
        residual_tolerance_met=bool(diagnostics["residual_GeV2"]<=1e-8),
        energy_error_GeV2=float(fit.fun-ev[0]),overlap_squared=float(abs(np.vdot(vec[:,0],psi))**2),
        metric_min_eigenvalue=float(np.linalg.eigvalsh(s.metric)[0]),qubits=circuit.qubits,parameters=len(fit.x),
        F1_F2=measured.tolist(),exact_model_F1_F2=exact,
        current_errors=np.abs(measured-exact).tolist(),
        compact_measurement_difference=float(np.linalg.norm(measured-direct)),
        finite_reference_state_error_bounds=bounds,
        parity_Hamiltonian_error=float(np.linalg.norm(sf.effective_hamiltonian-t@s.effective_hamiltonian@t.conj().T)),
        parity_metric_error=float(np.linalg.norm(sf.metric-t@s.metric@t.conj().T)),
        exact_state_seed=False,physical=False,scalable_quantum_advantage=False)
    print(json.dumps(diagnostics,indent=2),flush=True)


if __name__=="__main__":main()
