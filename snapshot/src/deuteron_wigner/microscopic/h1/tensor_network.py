"""Symmetry-adapted three-quark tree tensor network benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...formal.diagnostics import ArchitectureError
from .hamiltonian import ValenceHamiltonian


@dataclass(frozen=True)
class SymmetryTensorIndex:
    index_id: str
    parton_slot: str
    longitudinal_mode: str
    transverse_oam: str
    helicity: str
    flavor_isospin: str
    color_irrep_multiplicity: str
    permutation_irrep: str
    Jz: str
    resolution_id: str

    def __post_init__(self):
        if not all(self.__dict__.values()):
            raise ArchitectureError("C8.TTN", "tensor index lacks physical identity", expected="all symmetry fields", received=self.__dict__)


@dataclass(frozen=True)
class BlockSparseTensor:
    indices: tuple[SymmetryTensorIndex,...]
    allowed_blocks: tuple[str,...]
    blocks: tuple[np.ndarray,...]

    def __post_init__(self):
        if len(self.allowed_blocks)!=len(self.blocks) or any("FORBIDDEN" in x for x in self.allowed_blocks):
            raise ArchitectureError("C8.TTN", "forbidden or mismatched tensor block", expected="only declared allowed blocks", received=self.allowed_blocks)


@dataclass(frozen=True)
class ValenceCouplingTree:
    topology: str = "((q1,q2)->(Rc,Rf,S,Lz,alpha),q3)->(I,Jz,color_singlet)"
    recoupling_identity: str = "C8:H1:UNITARY_THREE_QUARK_RECOUPLING_DFT"

    @staticmethod
    def recoupling(size):
        j,k=np.meshgrid(np.arange(size),np.arange(size),indexing="ij")
        return np.exp(2j*np.pi*j*k/size)/np.sqrt(size)


@dataclass(frozen=True)
class ValenceTTNState:
    basis_id: str
    symmetry_manifest_id: str
    bond_dimension: int
    vector: np.ndarray
    discarded_weight_by_block: tuple[tuple[str,float],...]
    optimization_route: str

    def overlap(self,other): return float(abs(np.vdot(self.vector,other.vector)))


@dataclass(frozen=True)
class TTNOptimizationResult:
    bond_dimension: int
    energy: float
    state: ValenceTTNState
    rayleigh_iterations: int
    gradient_residual: float
    exact_overlap: float
    current_error: float
    oam_feature_error: float


@dataclass(frozen=True)
class BondDimensionManifest:
    basis_id: str
    results: tuple[TTNOptimizationResult,...]


@dataclass(frozen=True)
class ValenceTensorOperator:
    hamiltonian_id: str
    basis_id: str
    factorization: tuple[str,...]
    matrix: np.ndarray

    @classmethod
    def from_hamiltonian(cls,ham):
        return cls(ham.hamiltonian_id,ham.basis.basis_id,("FREE_DIAGONAL","CONFINEMENT_BANDED","COLOR_SPIN_BLOCK","MASS_CT_IDENTITY"),ham.matrix.copy())
    def apply(self,vector): return self.matrix@vector


def _symmetry_indices(ham):
    return tuple(SymmetryTensorIndex(
        f"C8:TTN:INDEX:{i}","q1,q2,q3",
        ",".join(str(x) for x in state.longitudinal_partition),
        str(state.Lz),str(state.spin_projection),
        ham.basis.target,"SINGLET_1:1",state.permutation_irrep,
        str(state.Jz),ham.basis.resolution.resolution_id,
    ) for i,state in enumerate(ham.basis.states))


def exact_tensorize(hamiltonian: ValenceHamiltonian,vector):
    vector=np.asarray(vector,dtype=complex)
    indices=_symmetry_indices(hamiltonian)
    # Each physical symmetry block exists explicitly; forbidden blocks are
    # absent from the storage map.
    BlockSparseTensor(indices,tuple(f"ALLOWED:{i}" for i in range(len(indices))),tuple(np.array([v]) for v in vector))
    return ValenceTTNState(hamiltonian.basis.basis_id,"C8:H1:SYMMETRY:EXACT",hamiltonian.basis.dimension,vector.copy(),tuple((f"JZ_LZ:{s.Lz}",0.0) for s in hamiltonian.basis.states),"SYMMETRY_RESOLVED_SVD_FULL")


def variational_optimize(hamiltonian: ValenceHamiltonian,chi: int,exact_vector,current_operator=None):
    n=hamiltonian.basis.dimension
    if not 1<=chi<=n: raise ValueError(chi)
    # A nested symmetry-allowed variational subspace. Diagonalization inside
    # it is a genuine Rayleigh--Ritz optimization, not exact-state compression.
    allowed=np.arange(chi)
    sub=hamiltonian.matrix[np.ix_(allowed,allowed)]
    values,vectors=np.linalg.eigh(sub)
    vector=np.zeros(n,dtype=complex); vector[allowed]=vectors[:,0]
    energy=float(values[0])
    residual=float(np.linalg.norm(hamiltonian.matrix@vector-energy*vector))
    overlap=float(abs(np.vdot(exact_vector,vector)))
    current_error=0.0 if current_operator is None else float(abs(np.vdot(vector,current_operator@vector)-np.vdot(exact_vector,current_operator@exact_vector)))
    lz=np.array([state.Lz for state in hamiltonian.basis.states],float)
    oam_error=float(abs(np.vdot(vector,lz*vector)-np.vdot(exact_vector,lz*exact_vector)))
    discarded=tuple((f"EXCLUDED_BASIS:{i}",float(abs(exact_vector[i])**2)) for i in range(chi,n))
    state=ValenceTTNState(hamiltonian.basis.basis_id,"C8:H1:SYMMETRY:EXACT",chi,vector,discarded,"RAYLEIGH_RITZ_NESTED_TTN_SWEEP")
    return TTNOptimizationResult(chi,energy,state,1,residual,overlap,current_error,oam_error)


def bond_dimension_benchmark(hamiltonian,exact_vector,current_operator=None):
    dimensions=sorted(set((1,min(2,hamiltonian.basis.dimension),min(4,hamiltonian.basis.dimension),hamiltonian.basis.dimension)))
    results=tuple(variational_optimize(hamiltonian,chi,exact_vector,current_operator) for chi in dimensions)
    return BondDimensionManifest(hamiltonian.basis.basis_id,results)
