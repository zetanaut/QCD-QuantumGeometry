"""Linear-storage exact projective geometry for the full complex tree circuit.

Each child is fresh when first rotated. The angle metric is diagonal with
G_theta[e,e]=w_e/4, where w_e is the parent's probability just before that
split. A phase edge moves its entire descendant subtree. If B is the oriented
edge/vertex incidence matrix, then G_phi^{-1}=B diag(1/p_v) B.T at an interior
state. Angle/phase cross terms vanish. Both inverse actions cost O(D), with
no state Jacobian, dense metric, eigendecomposition, or Hamiltonian eigenstate.

Positive caller-selected epsilon replaces w,p by w+epsilon,p+epsilon in
this inverse preconditioner. That is explicit optimization regularization,
not the exact metric, a Fock-state cutoff, or a physical parameter.
"""
import numpy as np


def tree_state_weights(circuit,parameters):
    x=np.asarray(parameters,float)
    if x.shape!=(circuit.parameter_count,) or not np.all(np.isfinite(x)):
        raise ValueError("finite circuit parameter vector required")
    if len(circuit.edges)!=circuit.dimension-1 or circuit.parameter_count!=2*len(circuit.edges):
        raise ValueError("complete two-parameter spanning tree required")
    psi=np.zeros(circuit.dimension,complex);psi[circuit.root]=1
    before=np.empty(len(circuit.edges));seen={circuit.root}
    for k,(parent,child) in enumerate(circuit.edges):
        if parent not in seen or child in seen or not 0<=child<circuit.dimension:
            raise ValueError("each tree child must be fresh and its parent already present")
        seen.add(child)
        before[k]=abs(psi[parent])**2
        c,s=np.cos(x[2*k]/2),np.sin(x[2*k]/2)
        amplitude=psi[parent]
        psi[parent]=c*amplitude
        psi[child]=s*amplitude*np.exp(1j*x[2*k+1])
    return psi,before


def tree_inverse_metric_action(circuit,parameters,covector,*,probability_regularization):
    """Apply the exact interior inverse (epsilon=0), or declared regularization.

    A singular chart at epsilon=0 is rejected, not silently regularized.
    Every probability/amplitude remains in the executed quantum state.
    """
    if isinstance(probability_regularization,(bool,np.bool_)):
        raise ValueError("explicit finite nonnegative probability regularization required")
    epsilon=float(probability_regularization)
    if not np.isfinite(epsilon) or epsilon<0:
        raise ValueError("explicit finite nonnegative probability regularization required")
    g=np.asarray(covector,float)
    if g.shape!=(circuit.parameter_count,) or not np.all(np.isfinite(g)):
        raise ValueError("finite parameter covector required")
    psi,w=tree_state_weights(circuit,parameters)
    p=abs(psi)**2
    if np.any(w+epsilon<=0) or np.any(p+epsilon<=0):
        raise ValueError("singular tree coordinate chart; choose an explicit regularization")
    result=np.empty_like(g)
    result[0::2]=4*g[0::2]/(w+epsilon)
    # B.T g_phi, diagonal inverse, then B. Never form the matrices.
    vertex=np.zeros(circuit.dimension)
    for k,(parent,child) in enumerate(circuit.edges):
        vertex[child]+=g[2*k+1];vertex[parent]-=g[2*k+1]
    vertex/=p+epsilon
    for k,(parent,child) in enumerate(circuit.edges):
        result[2*k+1]=vertex[child]-vertex[parent]
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("tree inverse metric overflow; unresolved coordinate chart")
    return result


def tree_state_tangent(circuit,parameters,direction):
    """State and horizontal Jacobian-vector action in O(D) storage/time.

    This differentiates the gate tree, not the Hamiltonian. It consumes a
    caller-supplied direction (in the solver, from the executed gradient).
    """
    x=np.asarray(parameters,float);v=np.asarray(direction,float)
    # Includes full tree and parameter validation; no dense Jacobian.
    tree_state_weights(circuit,x)
    if v.shape!=x.shape or not np.all(np.isfinite(v)):
        raise ValueError("finite tree parameter direction required")
    psi=np.zeros(circuit.dimension,complex);psi[circuit.root]=1
    tangent=np.zeros_like(psi)
    for k,(parent,child) in enumerate(circuit.edges):
        c,s=np.cos(x[2*k]/2),np.sin(x[2*k]/2);phase=np.exp(1j*x[2*k+1])
        a,da=psi[parent],tangent[parent]
        psi[parent]=c*a;psi[child]=phase*s*a
        tangent[parent]=c*da-s*a*v[2*k]/2
        tangent[child]=phase*(s*da+c*a*v[2*k]/2+1j*s*a*v[2*k+1])
    return psi,tangent-psi*np.vdot(psi,tangent)


def tree_parameters_from_state(circuit,state):
    """O(D) gate coordinates for a supplied state, modulo global phase.

    Used to retract a gradient-generated tangent, NOT to import an exact
    Hamiltonian eigenstate. No amplitude is clipped. Reverse accumulation
    avoids subtracting nearly equal subtree probabilities at chart boundaries.
    The executed ansatz remains the original controlled-gate tree.
    """
    psi=np.asarray(state,complex)
    if psi.shape!=(circuit.dimension,) or not np.all(np.isfinite(psi)):
        raise ValueError("finite tree state required")
    norm=np.linalg.norm(psi)
    if not np.isfinite(norm) or norm==0:
        raise ValueError("nonzero finite tree state norm required")
    tree_state_weights(circuit,np.zeros(circuit.parameter_count))
    probability=abs(psi/norm)**2
    remaining=probability.copy();subtree=np.empty(len(circuit.edges));after=np.empty_like(subtree)
    for k in reversed(range(len(circuit.edges))):
        parent,child=circuit.edges[k]
        subtree[k]=remaining[child];after[k]=remaining[parent]
        remaining[parent]+=remaining[child]
    phases=np.angle(psi)
    result=np.empty(circuit.parameter_count)
    for k,(parent,child) in enumerate(circuit.edges):
        result[2*k]=2*np.arctan2(np.sqrt(subtree[k]),np.sqrt(after[k]))
        result[2*k+1]=phases[child]-phases[parent]
    return result
