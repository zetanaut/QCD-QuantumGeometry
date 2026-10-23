"""Joint state/coefficient inference with exact and PennyLane references.

Finite synthetic source-family closure only; no physical nucleon calibration.
The matching rank, failed controls, withheld currents and state equations are
reported separately. No energy minimization with respect to free dynamics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deuteron_wigner.quantum.joint_state_dynamics import (
    GraphState,SourceFamily,AmplitudeMatching,state_metric,REAL,COMPLEX,
)


def fixture():
    h=np.diag([0.,.7,1.3,2.])
    t=np.zeros((3,4,4),complex)
    for i in range(3):t[0,i,i+1]=t[0,i+1,i]=-.4
    t[1]=np.diag([.1,-.2,.3,-.1])
    t[2,0,3]=.2j;t[2,3,0]=-.2j
    return SourceFamily(h,t,['hopping','diagonal','loop'],units='dimensionless_fixture')


def joint_fit(*, use_state_equation=True, seed=7, backend='torch', initial_loop=.2, phase_matching=False):
    torch.manual_seed(seed)
    model=fixture()
    truth=torch.tensor([.85,-.45,.55],dtype=REAL)
    eig,states=torch.linalg.eigh(model.matrix(truth))
    target=states[:,0]
    # Independent matching amplitudes identify only first two coefficients.
    design=np.array([[1.,.2,0],[.3,1.,0],[1.2,-.4,0]])
    retained=np.array([.3,-.2,.7])
    if phase_matching:
        # Im <0|H|3> / .2 is an independent phase-sensitive amplitude
        # in this FIXTURE's declared basis. It is not a QCD datum.
        design=np.vstack((design,[0.,0.,1.]))
        retained=np.append(retained,0.)
    matching=AmplitudeMatching(design,retained+design@truth.numpy(),retained,
                               np.eye(len(retained))*.02**2,specification='Synthetic same-carrier source-amplitude matching; not QCD Wilson coefficients')
    graph=GraphState(['channel_0','channel_1','channel_2','channel_3'],[(0,1),(1,2),(2,3)])
    statefn=graph.torch_state if backend=='torch' else graph.pennylane_state()
    # Independent diagonal measurements; complex transition/current elements withheld.
    probes=[torch.diag(torch.tensor(v,dtype=COMPLEX)) for v in ([1.,0.,0.,0.],[0.,1.,0.,-1.])]
    targets=torch.stack([torch.vdot(target,o@target).real for o in probes])
    parameters=torch.nn.Parameter(torch.tensor([[-.8,.1],[-.7,-.1],[-.6,.2]],dtype=REAL)+.06*torch.randn((3,2),dtype=REAL))
    if initial_loop<0:
        with torch.no_grad(): parameters[:,1].neg_()
    phi=torch.nn.Parameter(torch.tensor([.7,-.2,initial_loop],dtype=REAL))
    optimizer=torch.optim.LBFGS([parameters,phi],lr=.7,max_iter=500,tolerance_grad=2e-11,tolerance_change=1e-15,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        calls+=1
        optimizer.zero_grad()
        state=statefn(parameters)
        energy,residual=model.eigenstate_residual(phi,state)
        predicted=torch.stack([torch.vdot(state,o@state).real for o in probes])
        loss=matching.loss(phi)+torch.sum(((predicted-targets)/.01)**2)+((energy-eig[0])/.01)**2
        if use_state_equation:loss=loss+residual/.01**2
        loss.backward()
        return loss
    optimizer.step(closure)
    state=statefn(parameters)
    energy,residual=model.eigenstate_residual(phi,state)
    learned_eigs,learned_states=torch.linalg.eigh(model.matrix(phi))
    # Full density-matrix/source-current comparison is withheld from the fit.
    rho=torch.outer(state,state.conj());rho0=torch.outer(target,target.conj())
    predicted=torch.stack([torch.vdot(state,o@state).real for o in probes])
    current=model.currents(phi)
    current0=model.currents(truth)
    return {
        'backend':backend,'seed':seed,'use_state_equation':use_state_equation,'closure_calls':calls,
        'initial_loop':initial_loop,'phase_sensitive_matching':phase_matching,
        'truth':truth.tolist(),'learned':phi.detach().tolist(),
        'matching_rank':matching.rank,'matching_nullspace':matching.nullspace.tolist(),
        'matching_chi2':float(matching.loss(phi).detach()),
        'training_probe_max_error':float(torch.max(torch.abs(predicted-targets)).detach()),
        'training_energy_error':float(torch.abs(energy-eig[0]).detach()),
        'state_equation_norm':float(torch.sqrt(residual).detach()),
        'ground_energy_excess':float((energy-learned_eigs[0]).detach()),
        'fidelity_to_own_ground':float(torch.abs(torch.vdot(learned_states[:,0],state)).square().detach()),
        'withheld_density_matrix_error':float(torch.linalg.norm(rho-rho0).detach()),
        'withheld_current_operator_error':float(torch.linalg.norm(current-current0).detach()),
        'state_metric_eigenvalues':torch.linalg.eigvalsh(state_metric(statefn,parameters)).detach().tolist(),
        'matching_metric_eigenvalues':torch.linalg.eigvalsh(matching.design.T@matching.design).tolist(),
        'physical_nucleon_fit':False,
    }


def evaluate():
    full=[joint_fit(seed=s) for s in (7,11,29)]
    quantum=joint_fit(seed=7,backend='pennylane')
    negative=joint_fit(seed=7,use_state_equation=False)
    conjugate_fit=joint_fit(initial_loop=-.2)
    phase_resolved=[joint_fit(initial_loop=sign*.2,phase_matching=True) for sign in (1,-1)]
    # Sign of imaginary loop is not identifiable from these real diagonal
    # probes and energies alone. Report it; never select the sign by a hidden prior.
    return {'scope':'joint-learning/source/matching synthetic closure, not physical nucleon dynamics',
            'full_fits':full,'quantum_fit':quantum,'no_state_equation_negative':negative,
            'conjugate_branch_fit':conjugate_fit,'phase_resolved_fits':phase_resolved,
            'sign_identifiability':'real diagonal probes and eigenvalues cannot distinguish complex conjugate dynamics; a phase-sensitive observable or matching condition is required'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=evaluate()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
