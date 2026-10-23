"""Executed graph solve with linear-storage geometry on a retained-pilot control.

This is a solver scaling control, not the separate matched interacting-Q5/Q6
model and not a physical nucleon calculation. Exact states are never seeds.
"""
import argparse,json,time
import numpy as np
from scipy import linalg

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cutoff",type=int,required=True)
    parser.add_argument("--probability-regularization",type=float,required=True)
    parser.add_argument("--max-iterations",type=int,default=2000)
    parser.add_argument("--step-rule",choices=("parameters","tree_state_retraction"),default="parameters")
    args=parser.parse_args()
    start=time.monotonic()
    p=build_coupled_pilot(total_k2=9,cutoff=args.cutoff,target="PROTON",jz_twice=1,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    c=SymmetryGraphCircuit(p.matrix,root=int(np.argmin(p.matrix.diagonal().real)),support_threshold=1e-12)
    initial=c.uniform_parameters()
    initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(c.edges))
    print("Executing tree-geometry solver on",c.dimension,"retained states",flush=True)
    begin=time.monotonic()
    def progress(value):
        if value["iteration"]%25==0:
            print(json.dumps(dict(progress=value,seconds=time.monotonic()-begin)),flush=True)
    fit=quantum_natural_fit(c,initial,geometry="tree",metric_rcond=None,
        probability_regularization=args.probability_regularization,residual_tolerance=1e-8,
        max_iterations=args.max_iterations,step_rule=args.step_rule,progress=progress)
    fit_seconds=time.monotonic()-begin
    e,u=linalg.eigh(p.matrix)  # only after quantum optimization
    rho=good_current_density(p,(.12,.16),charge="electromagnetic")
    op=(rho+rho.conj().T)/2
    measured=float(c.measure_operators(fit.x,[op])[0]);exact=float(np.vdot(u[:,0],op@u[:,0]).real)
    d=c.state_diagnostics(fit.x)
    gap=e[1]-fit.fun
    bound=2*np.linalg.norm(op,2)*d["residual_GeV2"]/gap if gap>0 else None
    d.update(scope="bare retained coupled-pilot solver scaling control; NOT matched QQ/Q6 H2",
        cutoff=args.cutoff,dimension=c.dimension,qubits=c.qubits,parameters=c.parameter_count,
        iterations=int(fit.nit),evaluations=int(fit.nfev),success=bool(fit.success),
        message=str(fit.message),fit_seconds=fit_seconds,total_seconds=time.monotonic()-start,
        energy_error_GeV2=float(fit.fun-e[0]),current=measured,exact_current=exact,
        current_error=abs(measured-exact),finite_reference_state_error_bound=bound,
        geometry="tree",probability_regularization=args.probability_regularization,
        step_rule=args.step_rule,
        exact_state_seed=False,dense_parameter_metric=False,physical=False,quantum_advantage=False)
    print(json.dumps(d,indent=2),flush=True)


if __name__=="__main__":main()
