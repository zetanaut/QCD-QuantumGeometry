"""Resumable finite common-regulated H2 quantum benchmark, one member per run.

Saved members rebuild only H/N and source coordinates, never expensive currents.
Live members use explicitly identified baseline test parameters, not calibration.
All outputs are new files; old scientific/quantum reference outputs are untouched.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from deuteron_wigner.quantum.m2_cluster_quantum_reference import (
    load_saved_cluster_reference,reference_from_current,COMPONENTS)
from deuteron_wigner.quantum.m2_cluster_quantum_readout import (
    ClusterCurrentOracle,controlled_transition_readout,endpoint_error_certificate)
from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot,GaussianClusterH2
from deuteron_wigner.quantum.m2_gaussian_cluster_four_current import assemble_cluster_h2_four_current
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from deuteron_wigner.quantum.m2_total_isospin import H2_ISOSPIN_PRESCRIPTION
from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
from deuteron_wigner.quantum.m2_graph_natural_gradient import quantum_natural_fit


def assemble(f,i,transfer):
    return assemble_cluster_h2_four_current(f,i,transfer,charge='electromagnetic',
        loop_flavor_charges=(2/3,-1/3),prescription=PRESCRIPTION,quadrature_order=8,
        minus_context=SmoothMinusProjectorContext(mean_total_transverse_GeV=(0.,0.),
                                                  prescription=SMOOTH_PRESCRIPTION),
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,isospin_prescription=H2_ISOSPIN_PRESCRIPTION)


def live_reference(target,helicity_out,helicity_in,*,coupling,transfer):
    """Named baseline finite inputs, with explicit target/spin/coupling/transfer."""
    reg=GaussianClusterRegulator(width_GeV=.8,boson_modes=8,fermion_modes=8)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    def model(jz):
        p=build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=coupling,regulator=reg,
            total_pplus_GeV=1.3,wick_operators=wick)
        return GaussianClusterH2(p,excluded_cutoff=0,six_cutoff=0)
    f=model(helicity_out).snapshot(1.39)
    i=(f.owner if helicity_out==helicity_in else model(helicity_in)).snapshot(1.47)
    print('Assembling full regulated current at explicit baseline test parameters',flush=True)
    return reference_from_current(assemble(f,i,transfer))


def run(reference,*,check_reversal,residual_tolerance,max_iterations):
    import pennylane as qml
    start=time.monotonic()
    r=reference;operators=r.operators
    rng=np.random.default_rng(20260908)
    vectors=[]
    for s in (r.outgoing,r.incoming):
        v=rng.normal(size=len(s.matrix))+1j*rng.normal(size=len(s.matrix))
        vectors.append(v/np.linalg.norm(v))
    oracle=ClusterCurrentOracle(r)
    with qml.Tracker(oracle.device) as tracker:
        measured=oracle.transitions(*vectors)
        actions=oracle.actions(vectors[1])
    oracle_errors={n:float(abs(measured[n]-np.vdot(vectors[0],a@vectors[1])))
                   for n,a in operators.items()}
    action_errors={n:float(np.max(abs(actions[n]-a@vectors[1]))) for n,a in operators.items()}
    if max(*oracle_errors.values(),*action_errors.values())>8e-13:
        raise ArithmeticError('measured exact oracle parity failed')
    print('Measured all 24 complex operators; solving both endpoints',flush=True)
    circuits=[];fits=[];states=[]
    for label,s in (('out',r.outgoing),('in',r.incoming)):
        c=SymmetryGraphCircuit(s.matrix,root=int(np.argmin(s.matrix.diagonal().real)),support_threshold=1e-12)
        initial=c.uniform_parameters()
        initial[1::2]=.03*np.random.default_rng(4901).normal(size=len(c.edges))
        def progress(value):
            if value['iteration']%20==0:
                print(label+' '+json.dumps(value),flush=True)
        fit=quantum_natural_fit(c,initial,geometry='tree',metric_rcond=None,
            probability_regularization=1e-20,step_rule='tree_state_retraction',
            residual_tolerance=residual_tolerance,max_iterations=max_iterations,progress=progress)
        if not fit.success:raise ArithmeticError(label+' solver: '+fit.message)
        circuits.append(c);fits.append(fit);states.append(np.asarray(c.state(fit.x)))
    # Exact eigensystems are first accessed AFTER both actual gate optimizations.
    certificates=[endpoint_error_certificate(s.matrix,v[:len(s.matrix)])
                  for s,v in zip((r.outgoing,r.incoming),states)]
    readout=controlled_transition_readout(circuits[0],fits[0].x,circuits[1],fits[1].x,operators)
    np.testing.assert_allclose(readout['state'],np.r_[states[0],states[1]]/np.sqrt(2),atol=5e-14,rtol=0)
    checks={}
    for name,a in operators.items():
        actual=readout['transitions'][name]
        classical=np.vdot(states[0][:len(a)],a@states[1][:a.shape[1]])
        exact=np.vdot(certificates[0]['reference'],a@certificates[1]['reference'])
        bound=float(np.linalg.norm(a,2)*sum(c['state_distance_bound'] for c in certificates))
        distance_bound=float(np.linalg.norm(a,2)*sum(c['phase_aligned_distance'] for c in certificates))
        checks[name]=dict(measured=[float(actual.real),float(actual.imag)],
            executed_state_error=float(abs(actual-classical)),ground_reference_error=float(abs(actual-exact)),
            residual_gap_bound=bound,direct_state_distance_bound=distance_bound,
            roundoff_budget=8e-13)
        if abs(actual-classical)>8e-13 or abs(actual-exact)>bound+8e-13:
            raise ArithmeticError('controlled-gate measurement/state bound failed: '+name)
    arrays={"current_coefficients":r.coefficients,
        "normalized_current":np.stack([operators['total_'+mu] for mu in COMPONENTS]),
        "executed_flagged_state":readout['state']}
    for label,s,c,fit,v,cert in zip(('out','in'),(r.outgoing,r.incoming),circuits,fits,states,certificates):
        arrays.update({label+'_H':s.effective_hamiltonian,label+'_N':s.metric,
            label+'_L':s.matrix,label+'_state':v,label+'_parameters':fit.x,
            label+'_history':fit.history,label+'_reference':cert['reference']})
        for j,basis in enumerate(s.owner.pilot.bases):
            # scipy sparse objects become pickle-bearing object arrays under
            # np.savez. Keep exact CSR coordinates as ordinary numeric arrays.
            for key,value in (('data',basis.data),('indices',basis.indices),
                              ('indptr',basis.indptr),('shape',np.array(basis.shape))):
                arrays[f'{label}_basis_{j}_{key}']=value
    reversal=None
    if check_reversal:
        print('Independently assembling endpoint-exchanged reversed transfer',flush=True)
        b=reference_from_current(assemble(r.incoming,r.outgoing,-np.array(r.metadata['transfer_GeV'])))
        errors=[float(np.linalg.norm(r.coefficients[n]-b.coefficients[n].conj().transpose(0,2,1)))
                for n in range(5)]
        if max(errors)>5e-12:raise ArithmeticError('independent reversal identity failed')
        reverse_measurement=controlled_transition_readout(circuits[1],fits[1].x,circuits[0],fits[0].x,b.operators)
        measured_errors={name:float(abs(reverse_measurement['transitions'][name]-
            readout['transitions'][name].conjugate())) for name in operators}
        if max(measured_errors.values())>2e-12:raise ArithmeticError('measured reversed transition failed')
        arrays['independent_reverse_coefficients']=b.coefficients
        reversal=dict(coefficient_Frobenius_errors=errors,measured_errors=measured_errors,
                      independent_assembly=True,endpoint_energies_exchanged=True)
    p=r.incoming.owner.pilot
    endpoint_records=[]
    for c,fit,cert in zip(circuits,fits,certificates):
        record={k:v for k,v in cert.items() if k!='reference'}
        record.update(iterations=int(fit.nit),evaluations=int(fit.nfev),
            state_qubits=c.qubits,padding_probability=c.state_diagnostics(fit.x)['padding_probability'],
            root=c.root,support_threshold=c.support_threshold,padding_penalty_GeV2=c.padding_penalty)
        endpoint_records.append(record)
    result=dict(scope='finite common-regulated H2 g4 at Delta+=0; explicit test conditions',
        endpoint_current_metadata=dict(r.metadata),
        solver_controls=dict(geometry='tree',step_rule='tree_state_retraction',
            metric_rcond=None,probability_regularization=1e-20,max_iterations=max_iterations,
            initializer='uniform tree; seeded relative phases',phase_seed=4901,phase_sigma=.03,
            reference_phase='each exact endpoint aligned to its executed gate state'),
        model_parameters=dict(b_GeV=p.width_GeV,quark_mass_GeV=p.quark_mass_GeV,
            gluon_mass_GeV=p.gluon_mass_GeV,coupling=p.coupling,total_pplus_GeV=p.total_pplus_GeV,
            regulator=asdict(p.field_regulator),wick=asdict(p.wick_operators),
            excluded_cutoffs={n:s.cutoff for n,s in {**r.incoming.owner.spaces5,**r.incoming.owner.spaces6}.items()}),
        oracle=oracle.execution_metadata(),oracle_executions=int(tracker.totals['executions']),
        oracle_errors=oracle_errors,oracle_action_errors=action_errors,
        controlled_readout_executions=readout['executions'],controlled_readout_qubits=readout['readout_qubits'],
        controlled_readout_padding=readout['padding_probability'],
        controlled_readout_stateprep=False,exact_state_seed=False,measurements=checks,
        endpoints=endpoint_records,independent_reversal=reversal,
        Ward_coefficient_Frobenius=r.ward_residuals,
        residual_tolerance_GeV2=residual_tolerance,all_24_measurements_within_bounds=True,
        physical=False,quantum_advantage=False,shots=None,seconds=time.monotonic()-start)
    return result,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--saved',type=Path)
    group.add_argument('--target',choices=('PROTON','NEUTRON'))
    parser.add_argument('--helicity-out',type=int,choices=(-1,1))
    parser.add_argument('--helicity-in',type=int,choices=(-1,1))
    parser.add_argument('--coupling',type=float)
    parser.add_argument('--transfer',type=float,nargs=2)
    parser.add_argument('--check-reversal',action='store_true')
    parser.add_argument('--residual-tolerance',type=float,default=1e-8)
    parser.add_argument('--max-iterations',type=int,default=2000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.suffix!='.json':parser.error('output must be a new JSON path')
    matrix_file=args.output.with_suffix('.npz')
    if args.output.exists() or matrix_file.exists():parser.error('preserve existing output; select a new path')
    start=time.monotonic()
    if args.saved:
        if any(v is not None for v in (args.target,args.helicity_out,args.helicity_in,args.coupling,args.transfer)):
            parser.error('saved endpoint parameters cannot be overridden')
        reference=load_saved_cluster_reference(args.saved,charge='electromagnetic',
            loop_flavor_charges=(2/3,-1/3),mean_total_transverse_GeV=(0.,0.),quadrature_order=8)
    else:
        if any(v is None for v in (args.helicity_out,args.helicity_in,args.coupling,args.transfer)):
            parser.error('live named baseline requires both helicities, coupling and transfer explicitly')
        reference=live_reference(args.target,args.helicity_out,args.helicity_in,
                                 coupling=args.coupling,transfer=args.transfer)
    result,arrays=run(reference,check_reversal=args.check_reversal,
        residual_tolerance=args.residual_tolerance,max_iterations=args.max_iterations)
    result.update(total_seconds=time.monotonic()-start,source_report=str(args.saved) if args.saved else None,
                  full_matrix_file=matrix_file.name)
    # Exclusive creation protects old evidence even if another process races us.
    with matrix_file.open('xb') as stream:np.savez_compressed(stream,**arrays)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),seconds=result['total_seconds'],
        endpoint_residuals=[c['residual_GeV2'] for c in result['endpoints']],
        all_24_measurements_within_bounds=result['all_24_measurements_within_bounds'],
        independent_reversal=bool(result['independent_reversal']))),flush=True)


if __name__=='__main__':main()
