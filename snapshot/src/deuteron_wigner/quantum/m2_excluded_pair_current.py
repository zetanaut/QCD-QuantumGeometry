"""Coupled fermion/adjoint pair-current QQ factorization.

The auxiliary qqq + color-adjoint slot is a spin-zero CURRENT coordinate,
not a physical helicity-zero gluon or additional propagating Fock sector.
B_s D B_t† implements both pair-current orders of normal-ordered (j_q+j_g)^2
on the total-I=1/2 quark-antisymmetric source sector. Includes sea/gg pair
annihilation and their mixed IG cross channel; NOT IF or A^4 partners.
"""
from dataclasses import dataclass
from math import pi, sqrt
from operator import index
import numpy as np
from scipy import sparse

from .m2_c117_pair_current import color_permutation_matrix
from .m2_coupled_longitudinal_jacobi import (
    S3,longitudinal_basis,longitudinal_orbital_action,oscillator_shell,oscillator_rotation,
)
from .m2_spin_flavor_s3_completion import (
    SPIN_FLAVOR_BASIS,build_spin_flavor_s3_algebra,spin_flavor_permutation_representation,
)
from .m2_excluded_fock_action import IntermediateSpace
from .m2_excluded_fock_kinematics import color_source,frame,split_frame,split_routes
from .m2_instantaneous_sea_exit import sea_pair_spin_factor
from .m2_excluded_interaction_insertion import ExcludedInteractionInsertion


@dataclass
class PairCurrentAuxiliary:
    """Complete I=1/2/S3-sign current coordinate basis at its finite shell."""
    total_k2: int
    cutoff: int
    target: str
    jz_twice: int

    def __post_init__(self):
        if self.target not in ("PROTON","NEUTRON") or self.jz_twice not in (-1,1):
            raise ValueError("explicit target and twice-Jz required")
        if self.cutoff not in (0,1):
            raise ValueError("this pair-current reference supports cutoffs0/1")
        self.longitudinal=longitudinal_basis("qqqg",self.total_k2)
        shell=oscillator_shell(3,self.cutoff)
        sf=np.asarray(SPIN_FLAVOR_BASIS)
        angular=np.tile(np.repeat([sum(s[:3])-sum(s[3:]) for s in shell],2),len(self.longitudinal))
        sz=(2*(sf%2)-1).sum(axis=1);iz=np.where(sf<2,1,-1).sum(axis=1)
        # NO physical gluon helicity: this is J+ under transverse rotations.
        mask=(2*angular[:,None]+sz[None,:]==self.jz_twice)&(
            iz[None,:]==(1 if self.target=="PROTON" else -1))
        rows=np.flatnonzero(mask.ravel())
        iso=(3.75*np.eye(64)-build_spin_flavor_s3_algebra().isospin_squared)/3
        proj=sparse.csr_matrix((len(rows),len(rows)),dtype=complex)
        for p in S3:
            sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
            lc=sparse.kron(longitudinal_orbital_action("qqqg",self.total_k2,self.cutoff,p),
                           color_permutation_matrix("qqqg",p),format="csr")
            full=sparse.kron(lc,spin_flavor_permutation_representation(p)@iso,format="csr")
            proj+=sign/6*full[rows][:,rows]
        dense=proj.toarray()
        if np.linalg.norm(dense-dense.conj().T)>2e-11 or np.linalg.norm(dense@dense-dense)>2e-11:
            raise ArithmeticError("auxiliary current sign/isospin projector is not closed")
        ev,u=np.linalg.eigh(dense);u=u[:,ev>.5]
        if abs(np.trace(dense)-u.shape[1])>2e-10:
            raise ArithmeticError("auxiliary rank does not match projector trace")
        ri,ci=np.nonzero(u)
        self.basis=sparse.csr_matrix((u[ri,ci],(rows[ri],ci)),shape=(mask.size,u.shape[1]))
        raw_weight=np.repeat([2*self.total_k2/(pi*m[3]**2) for m in self.longitudinal],len(shell)*2*64)
        self.weight=(self.basis.getH()@self.basis.multiply(raw_weight[:,None])).toarray()
        if np.linalg.norm(self.basis.multiply(raw_weight[:,None]).toarray()-self.basis@self.weight)>2e-12:
            raise ArithmeticError("positive pair-transfer weight leaves auxiliary range")


def pair_current_orbital(route,cutoff,b_GeV):
    b=float(b_GeV)
    if not np.isfinite(b) or b<=0:
        raise ValueError("finite positive width required")
    incoming=oscillator_shell(3,cutoff);outgoing=oscillator_shell(4,cutoff)
    lookup={s:i for i,s in enumerate(incoming)}
    adapted=np.zeros((len(outgoing),len(incoming)))
    for row,s in enumerate(outgoing):
        inherited=s[:3]+s[4:7]
        if s[3]==s[7] and inherited in lookup:
            adapted[row,lookup[inherited]]=b/sqrt(pi)
    f=frame(route.sector,route.outgoing);x,y=f.fractions[3:]
    return sqrt(x*y/(x+y))*oscillator_rotation(split_frame(route),cutoff).T@adapted


def adjoint_pair_current_factor(k1,k2,h1,h2):
    """Two creation assignments of +f A dA, then normalized Bose 1/sqrt(2!)."""
    if any(isinstance(k,(bool,np.bool_)) or index(k)<=0 or index(k)%2 for k in (k1,k2)):
        raise ValueError("positive even doubled gluon modes required")
    if any(isinstance(h,(bool,np.bool_)) or h not in (-1,1) for h in (h1,h2)):
        raise ValueError("signed physical helicities required")
    eps_dot=(h1*h2-1)/2
    return -1j*(k1-k2)/(2*sqrt(k1*k2))*eps_dot/sqrt(2)


@dataclass
class PairCurrentFactorization:
    auxiliary: PairCurrentAuxiliary
    spaces: dict
    matrices: dict

    def apply(self,columns):
        if set(columns)!=set(self.spaces):
            raise ValueError("both pair-current sector column sets required")
        count=None
        for name,space in self.spaces.items():
            a=np.asarray(columns[name],complex)
            if a.ndim!=2 or a.shape[0]!=space.dimension or not np.all(np.isfinite(a)):
                raise ValueError("finite columns on exact pair-current coordinates required")
            if count is not None and count!=a.shape[1]:
                raise ValueError("common source-column count required")
            count=a.shape[1]
        auxiliary=sum(self.matrices[name].conj().T@columns[name] for name in self.spaces)
        weighted=self.auxiliary.weight@auxiliary
        return {name:b@weighted for name,b in self.matrices.items()}


def build_pair_current_factorization(reference):
    spaces={s.source.space.sector:s.source.space for s in reference.sectors}
    if set(spaces)!={"qqqgg","qqqq-qbar"}:
        raise ValueError("both actual five-body channels required")
    p=reference.pilot;cutoff=spaces["qqqgg"].cutoff
    if spaces["qqqq-qbar"].cutoff!=cutoff:
        raise ValueError("pair-current sectors must share the declared cutoff")
    aux=PairCurrentAuxiliary(p.total_k2,cutoff,p.target,p.jz_twice)
    no=len(oscillator_shell(3,cutoff));rank=aux.basis.shape[1]
    source=aux.basis.toarray().reshape(len(aux.longitudinal),no,2,64,rank)
    ilookup={m:i for i,m in enumerate(aux.longitudinal)}
    matrices={}
    for sector,space in spaces.items():
        channel="g_to_gg" if sector=="qqqgg" else "g_to_pair"
        color=color_source(channel)
        ol={m:i for i,m in enumerate(space.longitudinal)}
        image=np.zeros((space.dimension,rank),complex)
        for route in split_routes(p.total_k2,channel):
            orbital=pair_current_orbital(route,cutoff,p.width_GeV)
            amplitude=np.einsum("oi,ab,ibsk->oask",orbital,color,source[ilookup[route.incoming]],optimize=True)
            for h in (-1,1):
                if sector=="qqqgg":
                    k1,k2=route.outgoing[3:]
                    factor=adjoint_pair_current_factor(k1,k2,h,-h)
                    spins=(np.arange(64)*2+(h+1)//2)*2+(-h+1)//2
                    space.add(image,ol[route.outgoing],spins,factor*amplitude)
                else:
                    for flavor in range(2):
                        q,anti=2*flavor+(h+1)//2,2*flavor+(-h+1)//2
                        spins=(np.arange(64)*4+q)*4+anti
                        space.add(image,ol[route.outgoing],spins,sea_pair_spin_factor(h,-h)*amplitude)
        if sector=="qqqq-qbar":
            image=2*space.statistics_project(image)
        elif np.linalg.norm(image-space.statistics_project(image))>2e-11:
            raise ArithmeticError("adjoint pair current is not Bose symmetric")
        matrices[sector]=image
    return PairCurrentFactorization(aux,spaces,matrices)


class CoupledPairCurrentInsertion(ExcludedInteractionInsertion):
    """Single insertion of ALL pair-current channels, optional explicit block terms.

    Do not also supply sea annihilation in the block operators: its diagonal
    is already present in B D B†. Block terms should be independently named
    number-current scattering or other non-overlapping operators.
    """
    def __init__(self,reference,*,factorization,sector_operators,label):
        super().__init__(reference,sector_operators=sector_operators,label=label)
        for sector in reference.sectors:
            if factorization.spaces[sector.source.space.sector] is not sector.source.space:
                raise ValueError("factorized current must bind the exact reference spaces")
        self.factorization=factorization

    def wave_coefficients(self,sector_name,energy):
        # R W R V must sum incoming channels coherently BEFORE returning
        # the desired outgoing Q channel. No cross term is a separate fit.
        a1={}
        for name in self.sources:
            d=float(energy)-self.energies[name]
            if not np.all(np.isfinite(d)) or np.min(abs(d))<1e-10:
                raise ValueError("finite off-pole energy required")
            a1[name]=self.sources[name]/d[:,None]
        pair=self.factorization.apply(a1)
        d=float(energy)-self.energies[sector_name]
        a3=(self.operators[sector_name]@a1[sector_name]+pair[sector_name])/d[:,None]
        return a1[sector_name],a3
