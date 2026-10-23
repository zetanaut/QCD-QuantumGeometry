"""Tagged two-endpoint quantum readout for the explicit finite H2 reference.

The flag distinguishes coordinate spaces; equal dimensions do not identify
two spin fibers. StatePrep is confined to the exact oracle. Variational
transitions use coherent controls of the actual solved gate circuits.
Analytic simulator diagnostics, not hardware/resource advantage evidence.
"""
from importlib.metadata import version
import numpy as np
from scipy import sparse

from .m2_cluster_quantum_reference import ClusterQuantumReference


def _state(values, dimension):
    a = np.asarray(values,complex)
    if a.shape != (dimension,) or not np.all(np.isfinite(a)):
        raise ValueError("finite endpoint state with the declared dimension required")
    if not np.isclose(np.vdot(a,a).real,1.,atol=1e-12,rtol=0):
        raise ValueError("endpoint state must already be normalized")
    return a


def _transition_blocks(operators, df, di, size):
    names, observables = [], []
    for name, value in operators.items():
        a = np.asarray(value,complex)
        if a.shape != (df,di) or not np.all(np.isfinite(a)):
            raise ValueError("finite directed operator on the declared endpoint bases required")
        padded = sparse.bmat([[sparse.csr_matrix(a),sparse.csr_matrix((df,size-di))],
            [sparse.csr_matrix((size-df,di)),sparse.csr_matrix((size-df,size-di))]],format="csr")
        zero = sparse.csr_matrix((size,size),dtype=complex)
        names.append(name)
        observables.extend((sparse.bmat([[zero,padded],[padded.getH(),zero]],format="csr"),
            sparse.bmat([[zero,-1j*padded],[1j*padded.getH(),zero]],format="csr")))
    if not names:
        raise ValueError("at least one directed operator required")
    return tuple(names), tuple(observables)


class ClusterCurrentOracle:
    """Exact flagged-state measurements, with both endpoint bases retained."""
    physical = False
    shots = None

    def __init__(self, reference):
        import pennylane as qml
        if not isinstance(reference,ClusterQuantumReference):
            raise TypeError("explicit bound common-regulated reference required")
        self.reference = reference
        self.df,self.di = len(reference.outgoing.matrix),len(reference.incoming.matrix)
        self.endpoint_qubits = (max(self.df,self.di)-1).bit_length()
        self.size = 2**self.endpoint_qubits
        self.qubits = self.endpoint_qubits+1
        self.wires = tuple(range(self.qubits))
        self.names,blocks = _transition_blocks(reference.operators,self.df,self.di,self.size)
        self.blocks = blocks
        self.padding_penalty = max(float(np.max(np.sum(abs(s.matrix),axis=1)))
            for s in (reference.outgoing,reference.incoming))+1.
        h = []
        for s in (reference.outgoing,reference.incoming):
            h.append(sparse.block_diag((s.matrix,self.padding_penalty*sparse.eye(
                self.size-len(s.matrix))),format="csr"))
        padding = np.r_[np.zeros(self.df),np.ones(self.size-self.df),
                        np.zeros(self.di),np.ones(self.size-self.di)]
        self.hamiltonian = sparse.block_diag(h,format="csr")
        self.padding = sparse.diags(padding,format="csr")
        self.device = qml.device("lightning.qubit",wires=self.wires,shots=None,c_dtype=np.complex128)
        obs = tuple(qml.SparseHamiltonian(a,wires=self.wires) for a in
                    (*blocks,self.hamiltonian,self.padding))

        @qml.qnode(self.device,interface=None,diff_method=None)
        def measure(state):
            qml.StatePrep(state,wires=self.wires,normalize=False)
            return tuple(qml.expval(o) for o in obs)
        self._measure = measure

    def measure_padded(self, state):
        """Deliberately permit flagged padding probes to test isolation/penalty."""
        values = np.asarray(self._measure(_state(state,2*self.size)),float)
        current = values[:-2].reshape(-1,2)
        return dict(transitions=dict(zip(self.names,current[:,0]+1j*current[:,1])),
                    hamiltonian=float(values[-2]),padding_probability=float(values[-1]))

    def transitions(self, outgoing, incoming):
        f,i = _state(outgoing,self.df),_state(incoming,self.di)
        state = np.zeros(2*self.size,complex)
        state[:self.df],state[self.size:self.size+self.di] = f/np.sqrt(2),i/np.sqrt(2)
        return self.measure_padded(state)["transitions"]

    def actions(self, incoming):
        ket = _state(incoming,self.di)
        result = {name:np.empty(self.df,complex) for name in self.names}
        for j in range(self.df):
            bra = np.eye(1,self.df,k=j,dtype=complex)[0]
            for name,value in self.transitions(bra,ket).items():
                result[name][j] = value
        return result

    def execution_metadata(self):
        return dict(device="lightning.qubit",pennylane=version("PennyLane"),
            lightning=version("PennyLane-Lightning"),state_qubits=self.endpoint_qubits,
            readout_qubits=self.qubits,readout="tagged direct-sum Hermitian quadratures",
            exact_oracle_stateprep=True,shots=None,physical=False)


def controlled_transition_readout(outgoing_circuit, outgoing_parameters,
        incoming_circuit, incoming_parameters, operators):
    """Execute a coherent flag control of BOTH real gate preparations.

    A common qubit register is required here; the endpoint compact labels
    remain separate. No statevector is supplied as a gate or StatePrep seed.
    """
    import pennylane as qml
    f,i = outgoing_circuit,incoming_circuit
    if f.wires != i.wires:
        raise ValueError("explicit common qubit-register embeddings required")
    xf,xi = np.asarray(outgoing_parameters,float),np.asarray(incoming_parameters,float)
    for x,c in ((xf,f),(xi,i)):
        if x.shape != (c.parameter_count,) or not np.all(np.isfinite(x)):
            raise ValueError("finite circuit parameters required")
    names,blocks = _transition_blocks(operators,f.dimension,i.dimension,f.padded_dimension)
    flag = f.qubits
    wires = (flag,*f.wires)
    dev = qml.device("lightning.qubit",wires=wires,shots=None,c_dtype=np.complex128)
    obs = tuple(qml.SparseHamiltonian(a,wires=wires) for a in blocks)
    def prepare():
        qml.Hadamard(wires=flag)
        qml.ctrl(f._apply_ansatz,control=flag,control_values=[0])(xf)
        qml.ctrl(i._apply_ansatz,control=flag,control_values=[1])(xi)

    @qml.qnode(dev,interface=None,diff_method=None)
    def measure():
        prepare()
        return tuple(qml.expval(o) for o in obs)

    @qml.qnode(dev,interface=None,diff_method=None)
    def state():
        prepare()
        return qml.state()

    with qml.Tracker(dev) as tracker:
        values = np.asarray(measure(),float).reshape(-1,2)
        executed = np.asarray(state(),complex)
    gates = tuple(o.name for o in measure.qtape.operations)
    if any("StatePrep" in g or "QubitUnitary" in g for g in gates):
        raise ArithmeticError("non-gate state insertion in controlled readout")
    pad = np.r_[executed[f.dimension:f.padded_dimension],
                executed[f.padded_dimension+i.dimension:]]
    return dict(transitions=dict(zip(names,values[:,0]+1j*values[:,1])),
        state=executed,padding_probability=float(np.vdot(pad,pad).real),
        executions=int(tracker.totals["executions"]),readout_qubits=len(wires),
        gate_names=gates,exact_state_seed=False,stateprep_used=False)


def endpoint_error_certificate(matrix, state):
    """Independent finite-spectrum, phase-aligned ground-state distance bound.

    For Rayleigh value r below lambda_1, sin(angle) <= residual/(lambda_1-r).
    This requires explicit isolated-ground identification, not small residual
    alone. Reference vectors returned here must not be supplied to a solver.
    """
    h = np.asarray(matrix,complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or len(h)<2 or not np.all(np.isfinite(h)):
        raise ValueError("finite square Hamiltonian required")
    if not np.allclose(h,h.conj().T,rtol=0,atol=2e-12):
        raise ValueError("Hermitian Hamiltonian required")
    v = _state(state,len(h))
    eig,vec = np.linalg.eigh(h)
    rayleigh = float(np.vdot(v,h@v).real)
    residual = float(np.linalg.norm(h@v-rayleigh*v))
    separation = float(eig[1]-rayleigh)
    if separation <= 0 or eig[1]-eig[0] <= 1e-12:
        raise ValueError("isolated ground state and positive excited-state separation required")
    u = vec[:,0]
    overlap = np.vdot(u,v)
    if abs(overlap)==0:
        raise ValueError("nonzero ground overlap required for the phase convention")
    u = u*overlap/abs(overlap)
    bound = np.sqrt(2)*min(1.,residual/separation)
    distance = float(np.linalg.norm(v-u))
    if distance > bound+5e-13:
        raise ArithmeticError("ground-state error bound failed")
    return dict(reference=u,energy_GeV2=rayleigh,ground_energy_GeV2=float(eig[0]),
        residual_GeV2=residual,separation_GeV2=separation,
        phase_aligned_distance=distance,state_distance_bound=float(bound))
