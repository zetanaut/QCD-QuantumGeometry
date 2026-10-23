"""Packed exact-Iz/Jz six-particle spaces and local good-density action.

This is a finite global-Jacobi representation, not an interacting physical
state. All color multiplicities and spin/flavor labels precede statistics.
"""
from dataclasses import dataclass
from itertools import permutations,product
from math import factorial

import numpy as np

from .m2_coupled_longitudinal_jacobi import oscillator_rotation,oscillator_shell
from .m2_six_body_kinematics import (
    SPECIES,frame,longitudinal_states,color_basis,color_permutation,permutation_frame,
)
from .m2_coupled_pilot import _displacement
from .m2_canonical_good_current import parton_charge


@dataclass
class SixBodySpace:
    sector: str
    total_k2: int
    cutoff: int
    target: str
    jz_twice: int

    def __post_init__(self):
        if (self.sector not in SPECIES or isinstance(self.cutoff,(bool,np.bool_))
                or not isinstance(self.cutoff,(int,np.integer)) or self.cutoff<0):
            raise ValueError("six-particle sector and nonnegative integer N_exc required")
        if self.target not in ("PROTON","NEUTRON"):
            raise ValueError("PROTON or NEUTRON target required")
        if isinstance(self.jz_twice,(bool,np.bool_)) or not isinstance(self.jz_twice,(int,np.integer)):
            raise ValueError("exact doubled Jz required")
        species=SPECIES[self.sector]
        self.longitudinal=longitudinal_states(self.sector,self.total_k2)
        self.orbitals=oscillator_shell(5,self.cutoff)
        self.spin_dims=tuple(2 if s=="g" else 4 for s in species)
        self.spins=np.array(list(product(*(range(d) for d in self.spin_dims))))
        self.nspin=len(self.spins)
        self.ncolor=color_basis(self.sector).shape[1]
        self.shape=(len(self.longitudinal),len(self.orbitals),self.ncolor,self.nspin)
        sz=sum((2*self.spins[:,i]-1)*2 if s=="g" else 2*(self.spins[:,i]%2)-1
               for i,s in enumerate(species))
        iz=sum(np.where(self.spins[:,i]<2,1,-1)*(1 if s=="q" else -1)
               for i,s in enumerate(species) if s!="g")
        lz=np.array([sum(o[:5])-sum(o[5:]) for o in self.orbitals])
        allowed=((2*lz[:,None]+sz[None,:]==self.jz_twice)
                 &(iz[None,:]==(1 if self.target=="PROTON" else -1)))
        self.rows=np.flatnonzero(np.broadcast_to(allowed[None,:,None,:],self.shape).ravel())
        self.lookup=np.full(np.prod(self.shape),-1,dtype=int)
        self.lookup[self.rows]=np.arange(len(self.rows))
        self.coordinates=np.array(np.unravel_index(self.rows,self.shape)).T
        # Species permutations preserve Lz, Iz and total spin. Keep only the
        # corresponding exact spin block instead of allocating forbidden rows.
        self.symmetry_groups=[]
        self.group_rows={}
        for m in sorted(set(lz)):
            orbitals=np.flatnonzero(lz==m)
            spins=np.flatnonzero(allowed[orbitals[0]])
            if not len(spins):continue
            self.symmetry_groups.append((int(m),orbitals,spins))
            for li in range(len(self.longitudinal)):
                rows=np.flatnonzero((self.coordinates[:,0]==li)
                     &np.isin(self.coordinates[:,1],orbitals))
                if len(rows)!=len(orbitals)*self.ncolor*len(spins):
                    raise ArithmeticError("exact quantum-number block is incomplete")
                self.group_rows[li,int(m)]=rows

    @property
    def dimension(self):
        return len(self.rows)

    def add(self,output,longitudinal_index,spins,amplitudes):
        raw=(((longitudinal_index*len(self.orbitals)+np.arange(len(self.orbitals))[:,None,None])
              *self.ncolor+np.arange(self.ncolor)[None,:,None])*self.nspin
             +np.asarray(spins)[None,None,:])
        rows=self.lookup[raw]
        outside=rows<0
        if np.linalg.norm(amplitudes[outside])>3e-11:
            raise ArithmeticError("six-particle image violates exact Iz/Jz")
        output[rows[~outside]]+=amplitudes[~outside]

    def permute(self,values,permutation):
        a=np.asarray(values,complex)
        if a.ndim!=2 or a.shape[0]!=self.dimension or not np.all(np.isfinite(a)):
            raise ValueError("finite packed six-particle columns required")
        p=tuple(permutation)
        c=color_permutation(self.sector,p)
        spin_lookup=np.arange(self.nspin).reshape(self.spin_dims).transpose(p).ravel()
        long_lookup={s:i for i,s in enumerate(self.longitudinal)}
        result=np.zeros_like(a)
        for li,state in enumerate(self.longitudinal):
            target,r=permutation_frame(self.sector,state,p)
            orb=oscillator_rotation(r,self.cutoff)
            for m,orbitals,spins in self.symmetry_groups:
                inverse_spin=np.full(self.nspin,-1,dtype=int)
                inverse_spin[spins]=np.arange(len(spins))
                order=inverse_spin[spin_lookup[spins]]
                if np.any(order<0):
                    raise ArithmeticError("species permutation changed exact spin/charge block")
                outside=np.setdiff1d(np.arange(len(self.orbitals)),orbitals)
                if np.linalg.norm(orb[np.ix_(outside,orbitals)])>3e-12:
                    raise ArithmeticError("species permutation changed total orbital projection")
                rows=self.group_rows[li,m]
                block=a[rows].reshape(len(orbitals),self.ncolor,len(spins),a.shape[1])
                block=np.einsum("oi,ab,ibsk->oask",orb[np.ix_(orbitals,orbitals)],c,block,optimize=True)
                target_rows=self.group_rows[long_lookup[target],m]
                result[target_rows]=block[:,:,order].reshape(len(target_rows),a.shape[1])
        return result

    def statistics_project(self,values):
        nq=3 if self.sector=="qqqggg" else 4
        result=np.zeros_like(values,dtype=complex)
        for p in permutations(range(nq)):
            sign=(-1)**sum(p[i]>p[j] for i in range(nq) for j in range(i+1,nq))
            result+=sign*self.permute(values,p+tuple(range(nq,6)))/factorial(nq)
        if self.sector=="qqqggg":
            quark=result
            result=sum(self.permute(quark,(0,1,2)+p) for p in permutations((3,4,5)))/6
        return result

    def free_energies(self,b_GeV,quark_mass_GeV,gluon_mass_GeV):
        """Diagonal only at N_exc <=1; higher shells require the full operator."""
        if self.cutoff>1:
            raise ValueError("non-diagonal free shell: use ShellFreePropagation")
        b,mq,mg=map(float,(b_GeV,quark_mass_GeV,gluon_mass_GeV))
        if not np.all(np.isfinite([b,mq,mg])) or b<=0 or min(mq,mg)<0:
            raise ValueError("positive width and nonnegative finite masses required")
        masses=np.array([mg if s=="g" else mq for s in SPECIES[self.sector]])
        longitudinal=[np.sum(masses**2/frame(self.sector,s).fractions) for s in self.longitudinal]
        orbital=[b*b*(5+sum(o)) for o in self.orbitals]
        return np.array(longitudinal)[self.coordinates[:,0]]+np.array(orbital)[self.coordinates[:,1]]


def density_action(space,values,transfer,*,b_GeV,charge):
    """Same-Jz local J+/(2P+), Wilson order zero; other blocks not asserted zero."""
    return density_action_between(space,space,values,transfer,b_GeV=b_GeV,charge=charge)


def density_action_between(outgoing,incoming,values,transfer,*,b_GeV,charge):
    """Same-regulator density between explicit Jz fibers, without phase choices."""
    for key in ("sector","total_k2","cutoff","target"):
        if getattr(outgoing,key)!=getattr(incoming,key):
            raise ValueError("six-body current endpoints disagree on "+key)
    space=incoming
    a=np.asarray(values,complex);delta=np.asarray(transfer,float);b=float(b_GeV)
    if a.ndim!=2 or a.shape[0]!=space.dimension or not np.all(np.isfinite(a)):
        raise ValueError("finite packed six-particle columns required")
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b<=0:
        raise ValueError("finite transfer and positive width required")
    result=np.zeros((outgoing.dimension,a.shape[1]),complex);n=len(space.orbitals)
    for li,state in enumerate(space.longitudinal):
        density=np.zeros((n,n,space.nspin),complex)
        f=frame(space.sector,state)
        for particle,species in enumerate(SPECIES[space.sector]):
            if species=="g":continue
            shift=f.forward[:-1,particle,None]*delta[None,:]
            plus=(shift[:,0]-1j*shift[:,1])/(2*b)
            alpha=np.r_[plus,plus.conj()]
            weights=parton_charge(space.spins[:,particle],species,charge)
            for row,bra in enumerate(space.orbitals):
                for col,ket in enumerate(space.orbitals):
                    density[row,col]+=weights*np.prod([
                        _displacement(o,i,z) for o,i,z in zip(bra,ket,alpha)])
        rows=np.flatnonzero(space.coordinates[:,0]==li)
        orb,color,spin=space.coordinates[rows,1:].T
        block=np.zeros((n,space.ncolor,space.nspin,a.shape[1]),complex)
        block[orb,color,spin]=a[rows]
        out=np.einsum("ois,icsk->ocsk",density,block,optimize=True)
        target_rows=np.flatnonzero(outgoing.coordinates[:,0]==li)
        target_orb,target_color,target_spin=outgoing.coordinates[target_rows,1:].T
        result[target_rows]=out[target_orb,target_color,target_spin]
    return result
