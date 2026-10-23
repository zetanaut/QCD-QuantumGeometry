"""Dense and tree-factorized metric baselines for the controlled-gate circuit.

Actual PennyLane energies and adjoint gradients drive every update. Geometry
is calculated classically from the same gate tree and independently tested
against executed states. The optional tree inverse uses O(D) classical storage;
neither it nor the full O(D)-gate ansatz establishes quantum scalability.
No exact Hamiltonian eigenvector enters the initialization or update.
See Stokes et al., Quantum 4,269 (2020), doi:10.22331/q-2020-05-25-269.
"""
import numpy as np
from scipy.optimize import OptimizeResult


def graph_state_jacobian(circuit,parameters):
    """Analytic state/parameter Jacobian of the same complex two-level gates."""
    x=np.asarray(parameters,float)
    if x.shape!=(circuit.parameter_count,) or not np.all(np.isfinite(x)):
        raise ValueError("finite circuit parameter vector required")
    psi=np.zeros(circuit.dimension,complex);psi[circuit.root]=1
    jac=np.zeros((circuit.dimension,circuit.parameter_count),complex)
    for k,(a,b) in enumerate(circuit.edges):
        c,s=np.cos(x[2*k]/2),np.sin(x[2*k]/2)
        phase=np.exp(1j*x[2*k+1])
        aa,bb=psi[a],psi[b];ja,jb=jac[a].copy(),jac[b].copy()
        psi[a],psi[b]=c*aa-s*bb,(s*aa+c*bb)*phase
        jac[a],jac[b]=c*ja-s*jb,(s*ja+c*jb)*phase
        jac[a,2*k]+=(-s*aa-c*bb)/2
        jac[b,2*k]+=(c*aa-s*bb)*phase/2
        jac[b,2*k+1]+=1j*psi[b]
    return psi,jac


def graph_projective_metric(circuit,parameters):
    psi,jac=graph_state_jacobian(circuit,parameters)
    projected=jac-psi[:,None]*(psi.conj()@jac)[None,:]
    metric=(projected.conj().T@projected).real
    return (metric+metric.T)/2


def quantum_natural_fit(circuit,initial,*,metric_rcond,residual_tolerance,
                        max_iterations=1000,initial_step=.5,geometry="dense",
                        probability_regularization=None,step_rule="parameters",progress=None):
    """Full Fubini-Study geometry with executed energy line search.

    A caller-visible relative metric cutoff removes numerically unresolved
    tangent directions, never Hamiltonian/Fock states. A small parameter
    gradient alone is not the stopping criterion. 'success' means the
    executed state's eigenpair residual meets the requested tolerance;
    ground-state identification still requires an independent reference.

    geometry="tree" requires metric_rcond=None and an explicit nonnegative
    probability_regularization. Zero is the exact interior inverse; positive
    values regularize the inverse preconditioner without truncating the state.
    No dense Jacobian or metric is constructed on that path.
    Optional tree_state_retraction retracts the gradient-generated horizontal
    tangent on the normalized state sphere and converts it back to the SAME
    gate coordinates. It is not an exact-state seed or a compressed ansatz.
    """
    from pennylane import numpy as pnp
    from .m2_tree_projective_geometry import (
        tree_inverse_metric_action,tree_state_tangent,tree_parameters_from_state,
    )
    if (not np.all(np.isfinite([residual_tolerance,initial_step]))
            or min(residual_tolerance,initial_step)<=0):
        raise ValueError("explicit positive residual and step controls required")
    if geometry=="dense":
        if (metric_rcond is None or not np.isfinite(metric_rcond) or not 0<metric_rcond<1
                or probability_regularization is not None):
            raise ValueError("explicit dense metric controls required; no probability regularization")
    elif geometry=="tree":
        if metric_rcond is not None or probability_regularization is None:
            raise ValueError("tree controls require metric_rcond=None and explicit probability regularization")
    else:
        raise ValueError("geometry must be explicitly dense or tree")
    if step_rule not in ("parameters","tree_state_retraction") or (
            step_rule=="tree_state_retraction" and geometry!="tree"):
        raise ValueError("tree_state_retraction requires tree geometry")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    x=np.asarray(initial,float).copy()
    if geometry=="dense":
        graph_state_jacobian(circuit,x)
    else:
        tree_inverse_metric_action(circuit,x,np.zeros_like(x),
                                   probability_regularization=probability_regularization)
    history=[];evaluations=0;message="iteration limit";success=False
    for iteration in range(max_iterations+1):
        theta=pnp.array(x,requires_grad=True)
        value=float(circuit.centered_energy(theta));evaluations+=1
        gradient=np.asarray(circuit._centered_gradient(theta),float)
        diagnostics=circuit.state_diagnostics(x)
        history.append((value,diagnostics["residual_GeV2"]))
        if progress is not None:
            progress(dict(iteration=iteration,evaluations=evaluations,
                energy_GeV2=value+circuit.optimizer_energy_origin,
                residual_GeV2=diagnostics["residual_GeV2"]))
        if diagnostics["residual_GeV2"]<=residual_tolerance:
            success=True;message="executed eigenpair residual tolerance";break
        if iteration==max_iterations:
            break
        # Factor the real projective Jacobian directly. Forming G=B.T@B
        # first squares the condition number and obscures weak-amplitude
        # tangent directions. rcond still refers to eigenvalues of G.
        if geometry=="dense":
            psi,jacobian=graph_state_jacobian(circuit,x)
            projected=jacobian-psi[:,None]*(psi.conj()@jacobian)[None,:]
            factor=np.vstack((projected.real,projected.imag))
            _,singular,vh=np.linalg.svd(factor,full_matrices=False)
            eig=singular**2
            keep=eig>metric_rcond*eig[0]
            if not np.any(keep):
                message="no resolved metric directions";break
            direction=-.5*vh[keep].T@((vh[keep]@gradient)/eig[keep])
        else:
            direction=-.5*tree_inverse_metric_action(circuit,x,gradient,
                probability_regularization=probability_regularization)
        descent=float(gradient@direction)
        if not descent<0:
            message="no resolved descent direction";break
        if step_rule=="tree_state_retraction":
            psi,tangent=tree_state_tangent(circuit,x,direction)
        step=initial_step
        for trial in range(40):
            candidate=(x+step*direction if step_rule=="parameters" else
                       tree_parameters_from_state(circuit,psi+step*tangent))
            proposed=float(circuit.centered_energy(candidate));evaluations+=1
            if proposed<=value+1e-4*step*descent:
                x=candidate;break
            # Near an eigenstate, energy is second order in state error and
            # may no longer resolve improvement. Permit only a roundoff-size
            # energy rise AND an independently executed residual decrease.
            if (diagnostics["residual_GeV2"]<1e-6
                    and proposed<=value+16*np.finfo(float).eps*max(1.,abs(value))
                    and circuit.state_diagnostics(candidate)["residual_GeV2"]
                        < diagnostics["residual_GeV2"]):
                x=candidate;break
            step/=2
        else:
            message="line search cannot resolve an energy decrease";break
    return OptimizeResult(x=x,fun=value+circuit.optimizer_energy_origin,
                          centered_fun=value,jac=gradient,nit=iteration,nfev=evaluations,
                          success=success,message=message,diagnostics=diagnostics,
                          history=np.asarray(history),metric_rcond=metric_rcond,geometry=geometry,
                          probability_regularization=probability_regularization,step_rule=step_rule)
