"""Executed plus/transverse current tomography on the same order-four H2 state.

Bare finite model, NOT a physical form factor, covariance repair or complete
four-current. Both spin endpoints and raw QQ/Q6 contact/backflow are retained.
"""
import argparse,json,time
import numpy as np
from scipy.linalg import eigh

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_canonical_qq import FactorizedCanonicalQQSourceAction,ShellCanonicalQQInsertion
from deuteron_wigner.quantum.m2_canonical_six_body import CanonicalSixBodyExcursion
from deuteron_wigner.quantum.m2_fourth_order_h2 import FourthOrderH2,current_between
from deuteron_wigner.quantum.m2_fourth_order_transverse_current import transverse_coefficients_between
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit
from deuteron_wigner.quantum.m2_current_covariance_diagnostic import current_covariance_diagnostic
from deuteron_wigner.quantum.m2_orientation_current_diagnostic import orientation_current_diagnostic


def run(target,cutoff):
    start=time.monotonic();delta=(.12,.16);P=1.3
    def build(jz):
        p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
        ref=build_shell_fock_reference(p,out_cutoff=cutoff)
        qq=ShellCanonicalQQInsertion(ref,action=FactorizedCanonicalQQSourceAction(ref))
        return FourthOrderH2(ref,QQ_insertion=qq,six_body=CanonicalSixBodyExcursion(ref,out_cutoff=cutoff))
    print("Building the shared H/norm/current snapshots",flush=True)
    models=[build(1),build(-1)];snapshots=[m.snapshot(1.4) for m in models]
    s=snapshots[0]
    circuit=SymmetryGraphCircuit(s.matrix,root=int(np.argmin(s.matrix.diagonal().real)),support_threshold=1e-12)
    initial=circuit.uniform_parameters();initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(circuit.edges))
    fit=quantum_natural_fit(circuit,initial,geometry="tree",metric_rcond=None,
        probability_regularization=1e-20,step_rule="tree_state_retraction",
        residual_tolerance=1e-8,max_iterations=2000)
    print("Quantum state solved; constructing both matched current spin endpoints",flush=True)
    # The reference is computed after optimization and never supplied as a seed.
    e,u=eigh(s.matrix);exact=u[:,0];mass=np.sqrt(fit.fun)
    maps=[np.eye(len(s.matrix)),1j*transverse_parity_map(models[1].pilot,models[0].pilot)]
    operators=[];keys=[];coefficient_norms={}
    for hf,f in enumerate(snapshots):
        for hi,i in enumerate(snapshots):
            plus=current_between(f,i,delta,charge="electromagnetic",normalized=True)
            coefficients=transverse_coefficients_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic")
            transverse=np.array([f.whitener@a@i.whitener for a in sum(.3**k*a for k,a in coefficients.items())])
            coefficient_norms[str(hf)+str(hi)]=[float(np.linalg.norm(coefficients[k])) for k in range(5)]
            for mu,a in enumerate([plus,*transverse]):
                operators.append(maps[hf].conj().T@a@maps[hi]);keys.append((mu,hf,hi))
    print("Measuring all complex current matrix elements with the same controlled gates",flush=True)
    hermitian=[part for a in operators for part in ((a+a.conj().T)/2,(a-a.conj().T)/(2j))]
    measured=circuit.measure_operators(fit.x,hermitian).reshape(-1,2)
    values=measured[:,0]+1j*measured[:,1]
    reference=np.array([np.vdot(exact,a@exact) for a in operators])
    diagnostic=circuit.state_diagnostics(fit.x)
    gap=e[1]-diagnostic["energy_GeV2"]
    # Bound real and imaginary parts separately, then combine by the triangle inequality.
    part_bounds=[2*np.linalg.norm(a,2)*diagnostic["residual_GeV2"]/gap for a in hermitian]
    bounds=np.sum(np.array(part_bounds).reshape(-1,2),axis=1)
    current=np.zeros((3,2,2),complex)
    for key,value in zip(keys,values):current[key]=value
    cov=current_covariance_diagnostic(current[0],current[1:],delta,mass_GeV=mass,total_pplus_GeV=P)
    orientation=orientation_current_diagnostic(current[0],current[1:],delta,mass_GeV=mass,total_pplus_GeV=P)
    def pair(z):return [float(np.real(z)),float(np.imag(z))]
    return dict(scope="bare common-order g^4 H2 with QQ/Q6 transverse contact/backflow; NOT full QCD/four-current",
        target=target,excluded_cutoff=cutoff,retained_dimension=len(s.matrix),qubits=circuit.qubits,
        M2_GeV2=fit.fun,residual_GeV2=diagnostic["residual_GeV2"],residual_tolerance_met=bool(fit.success),
        iterations=int(fit.nit),evaluations=int(fit.nfev),padding_probability=diagnostic["padding_probability"],
        maximum_current_error=float(max(abs(values-reference))),maximum_current_state_bound=float(max(bounds)),
        every_current_error_within_bound=bool(np.all(abs(values-reference)<=bounds)),
        F1_from_plus=pair(cov.F1_from_plus),F2_from_plus=pair(cov.F2_from_plus),
        GM_plus=pair(cov.magnetic_from_plus),GM_transverse=pair(cov.magnetic_from_transverse),
        B1=pair(orientation.B1),B2=None,B3=None,
        orientation_structure_residual_norm=float(np.linalg.norm(orientation.structure_residual)),
        elastic_continuity_defect_norm=float(np.linalg.norm(cov.elastic_continuity_defect)),
        transverse_coefficient_norms=coefficient_norms,
        same_controlled_gate_state_for_all_components=True,exact_state_seed=False,
        physical=False,covariance_repaired=False,quantum_advantage=False,seconds=time.monotonic()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",choices=("PROTON","NEUTRON"),required=True)
    parser.add_argument("--cutoff",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.target,args.cutoff),indent=2),flush=True)
