"""Differentiable state/dynamics blocks and amplitude matching.

This supplies an inference mechanism, not a production nucleon interaction.
The caller supplies a symmetry-qualified carrier and source-dependent operator
family. State parameters and shared dynamics coefficients remain distinct.
PyTorch is optional; PennyLane is imported only for the requested backend.
"""
import numpy as np
import torch

REAL = torch.float64
COMPLEX = torch.complex128


class GraphState:
    """Complex normalized state from occupation-preserving edge rotations.

    A one-excitation encoding of caller-labelled basis states. This small
    reference uses D wires for D carrier vectors, NOT a scalable many-body
    encoding. Edges describe solver connectivity, not physical topological
    order. Physical symmetry comes from the supplied carrier/isometry.
    """
    def __init__(self, labels, edges):
        self.labels, self.edges = tuple(labels),tuple(tuple(e) for e in edges)
        self.dimension = len(self.labels)
        if self.dimension < 2 or len(set(self.labels)) != self.dimension:
            raise ValueError('Distinct physical-carrier labels required.')
        if not self.edges or any(len(e)!=2 or e[0]==e[1] or min(e)<0 or max(e)>=self.dimension for e in self.edges):
            raise ValueError('Valid distinct-node edges required.')
        reached = {0}
        for i,j in self.edges:
            if i in reached or j in reached:
                reached.update((i,j))
        if len(reached) != self.dimension:
            raise ValueError('Ordered gates must reach the complete declared carrier.')

    def _angles(self, angles):
        if angles.shape != (len(self.edges),2):
            raise ValueError('One rotation and phase per edge required.')
        if torch.is_complex(angles) or not torch.isfinite(angles).all():
            raise ValueError('Finite real gate parameters required.')

    def torch_state(self, angles):
        self._angles(angles)
        eye = torch.eye(self.dimension,dtype=COMPLEX,device=angles.device)
        state = eye[:,0]
        for (i,j),(rotation,phase) in zip(self.edges,angles):
            a,b = eye[:,i],eye[:,j]
            c,s = torch.cos(rotation/2),torch.sin(rotation/2)
            gate = eye+(c-1)*(torch.outer(a,a)+torch.outer(b,b))+s*(torch.outer(a,b)-torch.outer(b,a))
            state = gate@state
            state = state+(torch.exp(1j*phase)-1)*state[j]*b
        return state

    def pennylane_state(self):
        import pennylane as qml
        device = qml.device('default.qubit',wires=self.dimension,shots=None)
        @qml.qnode(device,interface='torch',diff_method='backprop')
        def circuit(angles):
            self._angles(angles)
            qml.PauliX(0)
            for (i,j),(rotation,phase) in zip(self.edges,angles):
                qml.SingleExcitation(rotation,wires=[i,j])
                qml.PhaseShift(phase,wires=j)
            return qml.state()
        indices = [1 << (self.dimension-1-i) for i in range(self.dimension)]
        def state(angles):
            return circuit(angles)[indices]
        return state


class SourceFamily:
    """Hermitian affine dynamics gauged by independent oriented edge sources.

    H_ij(phi,a)=H_ij(phi,0) exp(i a_ij). No independent current weights.
    This exact finite-source identity does NOT establish a continuum EM
    four-current or fix transverse nonlocal-path dependence.
    """
    def __init__(self, baseline, terms, names, *, units):
        base,ops = np.asarray(baseline,complex),np.asarray(terms,complex)
        if base.ndim!=2 or base.shape[0]!=base.shape[1] or ops.ndim!=3 or ops.shape[1:]!=base.shape:
            raise ValueError('Square baseline and same-carrier operator terms required.')
        if not units or len(names)!=len(ops) or len(set(names))!=len(names):
            raise ValueError('Explicit units and distinct coefficient names required.')
        if not np.isfinite(base).all() or not np.isfinite(ops).all():
            raise ValueError('Finite operators required.')
        if not np.allclose(base,base.conj().T,rtol=0,atol=1e-12) or not np.allclose(ops,ops.conj().transpose(0,2,1),rtol=0,atol=1e-12):
            raise ValueError('Hermitian source-free dynamics required.')
        self.baseline,self.terms = torch.tensor(base,dtype=COMPLEX),torch.tensor(ops,dtype=COMPLEX)
        self.names,self.units = tuple(names),units
        self.dimension = len(base)
        self.edges = tuple((i,j) for i in range(len(base)) for j in range(i+1,len(base))
                           if base[i,j]!=0 or np.any(ops[:,i,j]!=0))
        incidence = np.zeros((len(base),len(self.edges)))
        for e,(i,j) in enumerate(self.edges):
            incidence[i,e],incidence[j,e] = 1,-1
        self.incidence = torch.tensor(incidence,dtype=REAL)

    def matrix(self, coefficients, sources=None):
        if coefficients.shape!=(len(self.names),):
            raise ValueError('Explicit coefficient for every interaction term required.')
        if torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
            raise ValueError('Finite real coefficients preserve the declared Hermitian family.')
        h = self.baseline.to(coefficients.device)+torch.einsum('a,aij->ij',coefficients.to(COMPLEX),self.terms.to(coefficients.device))
        if sources is None:
            return h
        if sources.shape!=(len(self.edges),):
            raise ValueError('One real source per oriented edge required.')
        if torch.is_complex(sources) or not torch.isfinite(sources).all():
            raise ValueError('Finite real link phases required.')
        basis = torch.eye(self.dimension,dtype=COMPLEX,device=coefficients.device)
        result = h
        for (i,j),a in zip(self.edges,sources):
            result = result+h[i,j]*(torch.exp(1j*a)-1)*torch.outer(basis[i],basis[j])
            result = result+h[j,i]*(torch.exp(-1j*a)-1)*torch.outer(basis[j],basis[i])
        return result

    def currents(self, coefficients):
        h = self.matrix(coefficients)
        if not self.edges:
            return h.unsqueeze(0)[:0]
        basis = torch.eye(self.dimension,dtype=COMPLEX,device=coefficients.device)
        return torch.stack([1j*(h[i,j]*torch.outer(basis[i],basis[j])-h[j,i]*torch.outer(basis[j],basis[i])) for i,j in self.edges])

    def eigenstate_residual(self, coefficients, state):
        if state.shape!=(self.dimension,) or not torch.isfinite(state).all():
            raise ValueError('Finite state in the same carrier required.')
        if not torch.allclose(torch.vdot(state,state).real,torch.ones((),dtype=REAL,device=state.device),atol=1e-10,rtol=0):
            raise ValueError('Normalized state required; zero/scaled states are not solutions.')
        h = self.matrix(coefficients)
        energy = torch.vdot(state,h@state).real
        residual = h@state-energy*state
        return energy,torch.vdot(residual,residual).real


class AmplitudeMatching:
    """Match full minus explicitly retained amplitudes to an operator basis.

    Real coefficients, optionally complex amplitudes; covariance describes
    the stacked real/imaginary residuals. No missing coefficient is set to a
    physical zero. A least-norm solution is a representative if rank deficient.
    Scales/scheme/order and source-insertion definitions are caller metadata,
    not inferred from equal array shapes. Correlated residuals are whitened.
    """
    def __init__(self, design, full, retained, covariance, *, specification, rtol=1e-10):
        a,f,r = np.asarray(design),np.asarray(full),np.asarray(retained)
        if a.ndim!=2 or f.shape!=(a.shape[0],) or r.shape!=f.shape or not specification or not 0<rtol<1:
            raise ValueError('Common-domain amplitudes, basis and explicit matching specification required.')
        complex_rows = np.iscomplexobj(a) or np.iscomplexobj(f) or np.iscomplexobj(r)
        if complex_rows:
            a = np.concatenate((a.real,a.imag),axis=0)
            y = np.concatenate(((f-r).real,(f-r).imag))
        else:
            y = f-r
        cov = np.asarray(covariance,float)
        if cov.shape!=(len(y),len(y)) or not np.allclose(cov,cov.T,rtol=0,atol=1e-14):
            raise ValueError('Symmetric residual covariance with the declared real/imaginary ordering required.')
        if not all(np.isfinite(x).all() for x in (a,y,cov)):
            raise ValueError('Finite matching inputs required.')
        try:
            factor = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as e:
            raise ValueError('Positive definite covariance required; exact constraints need an explicit separate treatment.') from e
        aw,yw = np.linalg.solve(factor,a),np.linalg.solve(factor,y)
        _,singular,vh = np.linalg.svd(aw,full_matrices=True)
        self.rank = int(np.sum(singular>rtol*singular[0])) if len(singular) and singular[0]>0 else 0
        self.nullspace = vh[self.rank:].T
        self.singular_values = singular
        self.representative = np.linalg.lstsq(aw,yw,rcond=rtol)[0]
        self.design,self.target = torch.tensor(aw,dtype=REAL),torch.tensor(yw,dtype=REAL)
        self.specification = specification

    def loss(self, coefficients):
        residual = self.design.to(coefficients.device)@coefficients-self.target.to(coefficients.device)
        return torch.dot(residual,residual)


def state_metric(state_function, parameters):
    """Projective pure-state metric only; not a metric for effective coefficients."""
    shape = parameters.shape
    def real_output(flat):
        z = state_function(flat.reshape(shape))
        return torch.cat((z.real,z.imag))
    jac = torch.autograd.functional.jacobian(real_output,parameters.reshape(-1),create_graph=False)
    state = state_function(parameters)
    d = len(state)
    tangent = jac[:d]+1j*jac[d:]
    vertical = state.conj()@tangent
    return (tangent.conj().T@tangent-torch.outer(vertical.conj(),vertical)).real
