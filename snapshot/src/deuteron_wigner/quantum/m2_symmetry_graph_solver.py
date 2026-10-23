"""Exploratory symmetry-sector graph circuit; no hardware/advantage claim.

The compact basis already has exact I, Iz, Jz, color and fermion sign. A
caller-qualified Hamiltonian graph supplies a spanning tree of two-level
rotations. Gray-code controlled gates implement them without dense unitary
insertion or StatePrep. Each COMPLETE logical rotation is identity on padding.
Gate intermediates may visit padding; this is not a noise-protection code.
"""
from __future__ import annotations

from collections import deque
from math import ceil,log2
from operator import index

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh


def _matrix(matrix):
    h=sparse.csr_matrix(matrix,dtype=complex)
    if h.shape[0]!=h.shape[1] or h.shape[0]<2 or not np.all(np.isfinite(h.data)):
        raise ValueError("finite square Hamiltonian of dimension >=2 required")
    residual=h-h.getH()
    if residual.nnz and np.max(abs(residual.data))>2e-12:
        raise ValueError("Hamiltonian must be Hermitian")
    return h


def graph_tree(matrix,root,*,support_threshold):
    """Threshold defines circuit proposals ONLY; the measured H is unchanged.

    Reject disconnected support rather than silently ignoring components.
    """
    h=_matrix(matrix)
    if isinstance(root,(bool,np.bool_)):
        raise TypeError("root must be an index")
    root=index(root)
    tol=float(support_threshold)
    if not 0<=root<h.shape[0] or not np.isfinite(tol) or tol<0:
        raise ValueError("valid root and finite nonnegative graph threshold required")
    visited={root}; queue=deque([root]); edges=[]
    while queue:
        parent=queue.popleft()
        row=h.getrow(parent)
        candidates=sorted(zip(row.indices,row.data),key=lambda item:(-abs(item[1]),item[0]))
        for child,value in candidates:
            if child not in visited and abs(value)>tol:
                visited.add(int(child)); queue.append(int(child)); edges.append((parent,int(child)))
    if len(visited)!=h.shape[0]:
        raise ValueError("proposal graph disconnected: solve every component or lower the explicit threshold")
    return tuple(edges)


def lanczos_ground(matrix,*,tolerance=1e-11):
    """Sparse classical reference; eigenstate is NOT fed into the circuit."""
    h=_matrix(matrix)
    if h.shape[0]<=3:
        e,v=np.linalg.eigh(h.toarray()); return float(e[0]),v[:,0]
    # Fixed non-eigenstate starting vector, explicitly separate from VQE.
    start=np.cos(np.arange(h.shape[0])+0.3)+1j*np.sin(1.7*np.arange(h.shape[0])+.2)
    values,vectors=eigsh(h,k=1,which="SA",v0=start,tol=tolerance)
    return float(values[0]),vectors[:,0]


class SymmetryGraphCircuit:
    physical=False
    shots=None

    def __init__(self,matrix,*,root,support_threshold):
        import pennylane as qml
        self.hamiltonian=_matrix(matrix)
        self.dimension=self.hamiltonian.shape[0]
        self.root=index(root)
        self.edges=graph_tree(self.hamiltonian,root,support_threshold=support_threshold)
        self.support_threshold=float(support_threshold)
        self.qubits=ceil(log2(self.dimension))
        self.padded_dimension=2**self.qubits
        self.parameter_count=2*len(self.edges)
        self.wires=tuple(range(self.qubits))
        # Certified upper bound; padding cannot become the variational ground.
        bound=float(np.max(np.asarray(abs(self.hamiltonian).sum(axis=1))))
        self.padding_penalty=bound+max(1.,.1*bound)
        tail=self.padded_dimension-self.dimension
        self.padded_hamiltonian=sparse.block_diag((self.hamiltonian,
            self.padding_penalty*sparse.eye(tail)),format="csr")
        observable=qml.SparseHamiltonian(self.padded_hamiltonian,wires=self.wires)
        # Center INSIDE the measured operator, not after evaluating a large
        # total energy. A post-measurement subtraction cannot restore lost
        # precision in tiny energy changes. The shift is eigenstate-free.
        self.optimizer_energy_origin=float(np.min(self.hamiltonian.diagonal().real))
        centered=qml.SparseHamiltonian(
            self.padded_hamiltonian-self.optimizer_energy_origin*sparse.eye(self.padded_dimension),
            wires=self.wires)
        dev=qml.device("lightning.qubit",wires=self.wires,shots=None)

        def bits(label):
            return [(label>>(self.qubits-1-wire))&1 for wire in self.wires]

        def adjacent(a,b,theta=None):
            aa,bb=bits(a),bits(b)
            target=next(i for i in self.wires if aa[i]!=bb[i])
            controls=[i for i in self.wires if i!=target]
            pattern=[aa[i] for i in controls]
            if theta is None:
                qml.ctrl(qml.PauliX,control=controls,control_values=pattern)(wires=target)
            else:
                qml.ctrl(qml.RY,control=controls,control_values=pattern)(
                    theta if aa[target]==0 else -theta,wires=target)

        def rotation(a,b,theta):
            path=[a]; current=a
            for wire in self.wires:
                mask=1<<(self.qubits-1-wire)
                if (a^b)&mask:
                    current^=mask; path.append(current)
            swaps=[(path[k],path[k-1]) for k in range(len(path)-1,1,-1)]
            for u,v in swaps: adjacent(u,v)
            adjacent(a,path[1],theta)
            for u,v in reversed(swaps): adjacent(u,v)

        def phase(label,angle):
            pattern=bits(label)
            target=self.wires[-1]
            if pattern[-1]==0: qml.PauliX(wires=target)
            qml.ctrl(qml.PhaseShift,control=self.wires[:-1],control_values=pattern[:-1])(
                angle,wires=target)
            if pattern[-1]==0: qml.PauliX(wires=target)

        def ansatz(parameters):
            for wire,bit in enumerate(bits(self.root)):
                if bit: qml.PauliX(wires=wire)
            for k,(a,b) in enumerate(self.edges):
                rotation(a,b,parameters[2*k])
                phase(b,parameters[2*k+1])

        @qml.qnode(dev,interface="autograd",diff_method="adjoint")
        def energy(parameters):
            ansatz(parameters)
            return qml.expval(observable)

        @qml.qnode(dev,interface="autograd",diff_method="adjoint")
        def centered_energy(parameters):
            ansatz(parameters)
            return qml.expval(centered)

        @qml.qnode(dev,interface=None,diff_method=None)
        def state(parameters):
            ansatz(parameters)
            return qml.state()

        self.energy=energy
        self.centered_energy=centered_energy
        self.state=state
        self._apply_ansatz=ansatz
        self._gradient=qml.grad(energy)
        self._centered_gradient=qml.grad(centered_energy)

    def uniform_parameters(self):
        """Tree-only initializer with nonzero amplitude on every compact state.

        No Hamiltonian eigenvector is used. Near-zero edge angles can starve
        whole subtrees and create stationary but inaccurate variational runs.
        """
        sizes=np.ones(self.dimension,dtype=int)
        for parent,child in reversed(self.edges):
            sizes[parent]+=sizes[child]
        remaining=sizes.copy()
        theta=np.zeros(self.parameter_count)
        for k,(parent,child) in enumerate(self.edges):
            theta[2*k]=2*np.arcsin(np.sqrt(sizes[child]/remaining[parent]))
            remaining[parent]-=sizes[child]
        return theta

    def fit(self,initial,*,max_iterations=200,gradient_tolerance=1e-8,function_tolerance=1e-14):
        """Actual simulator energy/adjoint-gradient optimization, no exact-state seed.

        This full tree prototype has O(dimension) parameters and controlled
        gates. It is a scaling baseline, NOT an efficient many-body ansatz.
        """
        from pennylane import numpy as pnp
        x=np.asarray(initial,float)
        if x.shape!=(self.parameter_count,) or not np.all(np.isfinite(x)):
            raise ValueError("finite parameter vector required")
        if (not np.isfinite(gradient_tolerance) or gradient_tolerance<=0
                or not np.isfinite(function_tolerance) or function_tolerance<0):
            raise ValueError("positive gradient and nonnegative function tolerances required")
        def objective(values):
            theta=pnp.array(values,requires_grad=True)
            return float(self.centered_energy(theta)),np.asarray(self._centered_gradient(theta),float)
        result=minimize(objective,x,jac=True,method="L-BFGS-B",options={
            "maxiter":max_iterations,"gtol":gradient_tolerance,"ftol":function_tolerance,"maxls":40})
        result.centered_fun=float(result.fun)
        result.energy_origin=self.optimizer_energy_origin
        result.fun+=self.optimizer_energy_origin
        return result

    def state_diagnostics(self,parameters):
        """Executed-state residual/variance; optimizer success is not physics convergence.

        The residual bounds distance to SOME eigenvalue, not automatically the
        ground eigenvalue. A separately validated spectral gap/reference is
        needed to bound a ground-state overlap or an observable error.
        """
        psi=np.asarray(self.state(parameters))
        action=self.padded_hamiltonian@psi
        energy=float(np.vdot(psi,action).real)
        residual=float(np.linalg.norm(action-energy*psi))
        return dict(energy_GeV2=energy,residual_GeV2=residual,variance_GeV4=residual**2,
                    norm=float(np.vdot(psi,psi).real),
                    padding_probability=float(np.linalg.norm(psi[self.dimension:])**2))

    def measure_operators(self,parameters,operators):
        """Execute the SAME controlled-gate state for caller-supplied Hermitian observables."""
        import pennylane as qml
        x=np.asarray(parameters,float)
        if x.shape!=(self.parameter_count,) or not np.all(np.isfinite(x)):
            raise ValueError("finite parameter vector required")
        observables=[]
        for value in operators:
            op=_matrix(value)
            if op.shape!=self.hamiltonian.shape:
                raise ValueError("observable must act on this compact symmetry fiber")
            tail=self.padded_dimension-self.dimension
            padded=sparse.block_diag((op,sparse.csr_matrix((tail,tail))),format="csr")
            observables.append(qml.SparseHamiltonian(padded,wires=self.wires))
        if not observables:
            raise ValueError("at least one explicit observable required")
        dev=qml.device("lightning.qubit",wires=self.wires,shots=None)
        @qml.qnode(dev,interface=None,diff_method=None)
        def measure():
            self._apply_ansatz(x)
            return tuple(qml.expval(op) for op in observables)
        # Some supported QNode versions unwrap a one-element return tuple.
        # The public measurement API always returns one entry per observable.
        return np.asarray(measure()).reshape(len(observables))

    def classical_state(self,parameters):
        """Independent two-component updates to audit Gray-code gate execution."""
        x=np.asarray(parameters,float)
        if x.shape!=(self.parameter_count,) or not np.all(np.isfinite(x)):
            raise ValueError("finite parameter vector required")
        state=np.zeros(self.dimension,complex); state[self.root]=1
        for k,(a,b) in enumerate(self.edges):
            c,s=np.cos(x[2*k]/2),np.sin(x[2*k]/2)
            aa,bb=state[a],state[b]
            state[a],state[b]=c*aa-s*bb,(s*aa+c*bb)*np.exp(1j*x[2*k+1])
        return state
