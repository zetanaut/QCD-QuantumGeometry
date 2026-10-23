"""Explicit finite-shell H0 + bare qq/qg + canonical emission pilot.

Partial dynamics ONLY: every parameter is supplied by the caller. Omitted
operators are listed, not asserted to vanish. This does not replace M2m,
identify a physical nucleon/deuteron, or bind finite-C43 matching.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial, sqrt

import numpy as np
from scipy import sparse
from scipy.special import eval_genlaguerre

from .m2_c117_pair_current import color_permutation_matrix
from .m2_coupled_longitudinal_jacobi import (
    S3, jacobi_frame, longitudinal_basis, longitudinal_orbital_action,
    oscillator_shell, total_sign_weight_rank,
)
from .m2_coupled_pair_action import _width, qq_pair_mass_per_gs2
from .m2_canonical_mixed_current import qg_cross_mass_per_gs2
from .m2_canonical_emission import apply_canonical_emission
from .m2_canonical_instantaneous_fermion import instantaneous_fermion_mass_per_gs2
from .m2_spin_flavor_s3_completion import (
    SPIN_FLAVOR_BASIS, build_spin_flavor_s3_algebra,
    spin_flavor_permutation_representation,
)

OMITTED_TERMS=(
    "unsubtracted Wick traces and physical finite counterterms (see m2_canonical_gluon_self)",
    "higher-Fock transitions and induced effective interactions",
    "finite-C43 operator mixing and renormalization conditions",
    "zero modes, complete physical currents, and nuclear dynamics",
)


@lru_cache(maxsize=32)
def symmetry_basis(sector,total,cutoff,target,jz_twice,hg):
    """Sparse isometry onto I=1/2, fixed Iz/Jz, joint S3 sign.

    Dense diagonalization is confined to the fixed-weight projector. This
    pilot routine is an exact small-resolution reference, not a scalable
    construction for arbitrary cutoffs.
    All half-integer Jz weights are supported; a weight alone is not a
    physical spin multiplet. Unrepresented finite-shell weights are empty.
    """
    rank=total_sign_weight_rank(sector,total,cutoff,target,jz_twice,hg)
    nc=1 if sector=="qqq" else 2
    d=2 if sector=="qqq" else 3
    nl=len(longitudinal_basis(sector,total))
    shell=oscillator_shell(d,cutoff)
    sf=np.array(SPIN_FLAVOR_BASIS)
    sz=(2*(sf%2)-1).sum(axis=1)
    iz=np.where(sf<2,1,-1).sum(axis=1)
    angular=np.tile(np.repeat([sum(s[:d])-sum(s[d:]) for s in shell],nc),nl)
    mask=(2*angular[:,None]+sz[None,:]+2*hg==jz_twice)&(
        iz[None,:]==(1 if target=="PROTON" else -1))
    rows=np.flatnonzero(mask.ravel())
    iso=(3.75*np.eye(64)-build_spin_flavor_s3_algebra().isospin_squared)/3
    proj=sparse.csr_matrix((len(rows),len(rows)),dtype=complex)
    for p in S3:
        sign=(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        lc=sparse.kron(longitudinal_orbital_action(sector,total,cutoff,p),
                        color_permutation_matrix(sector,p),format="csr")
        op=sparse.kron(lc,spin_flavor_permutation_representation(p)@iso,format="csr")
        proj+=sign/6*op[rows][:,rows]
    a=proj.toarray()
    if np.linalg.norm(a-a.conj().T)>2e-11 or np.linalg.norm(a@a-a)>2e-11:
        raise ArithmeticError("joint fixed-weight projector did not close")
    values,vectors=np.linalg.eigh(a)
    selected=vectors[:,values>.5]
    if selected.shape[1]!=rank:
        raise ArithmeticError("projector eigen-rank disagrees with character count")
    # The isometry within degenerate multiplicities has no physical meaning.
    ri,ci=np.nonzero(selected)
    return sparse.csr_matrix((selected[ri,ci],(rows[ri],ci)),shape=(mask.size,rank))


def free_orbital(d,cutoff,b_GeV):
    """P_shell sum Q_a^2 P_shell in circular momentum-oscillator phases."""
    b=_width(b_GeV)
    shell=oscillator_shell(d,cutoff)
    lookup={s:i for i,s in enumerate(shell)}
    result=np.diag([b*b*(sum(s)+d) for s in shell])
    for j,s in enumerate(shell):
        for k in range(d):
            out=list(s)
            out[k]+=1; out[d+k]+=1
            key=tuple(out)
            if key in lookup:
                i=lookup[key]
                result[i,j]=result[j,i]=b*b*sqrt((s[k]+1)*(s[d+k]+1))
    return sparse.csr_matrix(result)


def free_sector(sector,total,cutoff,b,mq,mg):
    nc=1 if sector=="qqq" else 2
    d=2 if sector=="qqq" else 3
    masses=np.array([mq]*3+([mg] if sector=="qqqg" else []))
    if np.any(masses<0) or not np.all(np.isfinite(masses)):
        raise ValueError("finite nonnegative masses required")
    orbital=free_orbital(d,cutoff,b)
    blocks=[]
    for state in longitudinal_basis(sector,total):
        mass=float(np.sum(masses**2/jacobi_frame(state).fractions))
        blocks.append(orbital+mass*sparse.eye(orbital.shape[0],format="csr"))
    lo=sparse.block_diag(blocks,format="csr")
    return sparse.kron(lo,sparse.eye(nc*64),format="csr")


@dataclass(frozen=True)
class CoupledPilot:
    total_k2: int
    cutoff: int
    target: str
    jz_twice: int
    width_GeV: float
    quark_mass_GeV: float
    gluon_mass_GeV: float
    coupling: float
    bases: tuple
    h0: np.ndarray
    cross_pairs_per_gs2: np.ndarray
    emission_per_gs: np.ndarray
    instantaneous_fermion_per_gs2: np.ndarray
    matrix: np.ndarray
    omitted_terms: tuple=OMITTED_TERMS
    physical: bool=False


def build_coupled_pilot(*,total_k2,cutoff,target,jz_twice,b_GeV,
                         quark_mass_GeV,gluon_mass_GeV,coupling,_contact_kernel=None,_emission_orbital=None):
    """No default physical parameters and no implicit calibration."""
    b=_width(b_GeV)
    mq,mg,g=map(float,(quark_mass_GeV,gluon_mass_GeV,coupling))
    if not np.all(np.isfinite([mq,mg,g])) or min(mq,mg)<0:
        raise ValueError("finite coupling and nonnegative masses required")
    labels=(("qqq",0),("qqqg",-1),("qqqg",1))
    bases=tuple(symmetry_basis(s,total_k2,cutoff,target,jz_twice,hg) for s,hg in labels)
    n=sum(u.shape[1] for u in bases)
    if n>3000:
        raise ValueError("dense pilot exceeds 3000 directions; use a matrix-free scaled construction")
    h0=[]; pairs=[]; instants=[]
    for (sector,hg),u in zip(labels,bases):
        free=free_sector(sector,total_k2,cutoff,b,mq,mg)
        pair=qq_pair_mass_per_gs2(sector,total_k2,b,cutoff,_contact_kernel=_contact_kernel)
        if sector=="qqqg":
            pair+=qg_cross_mass_per_gs2(total_k2,b,cutoff,_contact_kernel=_contact_kernel)
        pair=sparse.kron(pair,sparse.eye(64),format="csr")
        h0.append((u.getH()@free@u).toarray())
        pairs.append((u.getH()@pair@u).toarray())
        if sector=="qqqg":
            contact=instantaneous_fermion_mass_per_gs2(total_k2,b,cutoff,hg,_contact_kernel=_contact_kernel)
            instants.append((u.getH()@contact@u).toarray())
        else:
            instants.append(np.zeros((u.shape[1],u.shape[1]),complex))
    kinetic=sparse.block_diag(h0).toarray()
    cross=sparse.block_diag(pairs).toarray()
    emission=np.zeros((n,n),complex)
    qbase=bases[0].toarray()
    nq=bases[0].shape[1]
    offset=nq
    for (_,hg),u in zip(labels[1:],bases[1:]):
        columns=[]
        for column in qbase.T:
            image=apply_canonical_emission(column.reshape(-1,1,64),total_k2,cutoff,b,mq,hg,_orbital_kernel=_emission_orbital)
            columns.append(u.getH()@image.ravel())
        block=np.column_stack(columns)
        emission[offset:offset+u.shape[1],:nq]=block
        emission[:nq,offset:offset+u.shape[1]]=block.conj().T
        offset+=u.shape[1]
    instant=sparse.block_diag(instants).toarray()
    matrix=kinetic+g*emission+g*g*(cross+instant)
    if np.linalg.norm(matrix-matrix.conj().T)>2e-10:
        raise ArithmeticError("derived pilot Hamiltonian is not Hermitian")
    return CoupledPilot(total_k2,cutoff,target,jz_twice,b,mq,mg,g,bases,
                         kinetic,cross,emission,instant,matrix)


def _displacement(out,inc,alpha):
    z=abs(alpha)**2
    if out>=inc:
        return np.exp(-z/2)*sqrt(factorial(inc)/factorial(out))*alpha**(out-inc)*eval_genlaguerre(inc,out-inc,z)
    return np.exp(-z/2)*sqrt(factorial(out)/factorial(inc))*(-alpha.conjugate())**(inc-out)*eval_genlaguerre(out,inc-out,z)


def baryon_density(pilot,transfer):
    """Finite-shell normalized baryon U(1) density, not physical EM current.

    Quarks carry 1/3, gluon zero. Each physical momentum recoil is mapped
    with the actual fraction-dependent J_x; no equal-share form factor.
    """
    delta=np.asarray(transfer,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError("finite transverse transfer pair required")
    reduced=[]
    for sector,u in zip(("qqq","qqqg","qqqg"),pilot.bases):
        d=2 if sector=="qqq" else 3
        nc=d-1
        shell=oscillator_shell(d,pilot.cutoff)
        blocks=[]
        for state in longitudinal_basis(sector,pilot.total_k2):
            f=jacobi_frame(state)
            block=np.zeros((len(shell),len(shell)),complex)
            for i in range(3):
                shift=f.forward[:-1,i,None]*delta[None,:]
                plus=(shift[:,0]-1j*shift[:,1])/(2*pilot.width_GeV)
                alpha=np.r_[plus,plus.conj()]
                for row,bra in enumerate(shell):
                    for col,ket in enumerate(shell):
                        block[row,col]+=np.prod([_displacement(a,b,z) for a,b,z in zip(bra,ket,alpha)])/3
            blocks.append(block)
        full=sparse.kron(sparse.block_diag(blocks,format="csr"),sparse.eye(nc*64),format="csr")
        reduced.append((u.getH()@full@u).toarray())
    return sparse.block_diag(reduced).toarray()


def mass_flow_current(pilot,transfer):
    """Minimum-norm mass-squared-flow current: Delta.J=[M^2,rho].

    Units GeV. Divide by 2P+ for the P-minus-flow convention. This
    construction does NOT determine transverse or non-Abelian physical
    currents, zero modes, counterterms, or Ward identities of full QCD.
    """
    delta=np.asarray(transfer,float)
    rho=baryon_density(pilot,delta)
    norm=float(delta@delta)
    if norm==0:
        raise ValueError("nonzero transfer required for longitudinal flow current")
    comm=pilot.matrix@rho-rho@pilot.matrix
    return rho,delta[0]/norm*comm,delta[1]/norm*comm
