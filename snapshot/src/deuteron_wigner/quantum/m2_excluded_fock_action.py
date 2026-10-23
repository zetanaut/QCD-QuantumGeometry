"""Coherent, statistics-correct cubic images into explicit excluded sectors.

Packed rows carry exact Iz and Jz but retain all color multiplicities,
longitudinal tuples, oscillator modes and spin/flavor labels. No numerical
amplitude threshold defines support. Higher-Fock intermediate states are
not identified with oscillator-only matching from the preceding milestone.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial, sqrt

import numpy as np

from .m2_coupled_longitudinal_jacobi import oscillator_rotation, oscillator_shell
from .m2_excluded_fock_kinematics import (
    SPECIES, color_basis, color_permutation, color_source, frame,
    longitudinal_states, permutation_frame, split_routes,
)
from .m2_excluded_fock_vertices import cubic_orbital


@dataclass
class IntermediateSpace:
    sector: str
    total_k2: int
    cutoff: int
    target: str
    jz_twice: int

    def __post_init__(self):
        if (self.sector not in ("qqqgg","qqqq-qbar")
                or isinstance(self.cutoff,(bool,np.bool_))
                or not isinstance(self.cutoff,(int,np.integer)) or self.cutoff<0):
            raise ValueError("five-particle sector and nonnegative integer N_exc required")
        if self.target not in ("PROTON","NEUTRON"):
            raise ValueError("PROTON or NEUTRON target required")
        self.longitudinal = longitudinal_states(self.sector,self.total_k2)
        self.orbitals = oscillator_shell(4,self.cutoff)
        self.spin_dims = (4,4,4,2,2) if self.sector=="qqqgg" else (4,4,4,4,4)
        self.spins = np.array(list(product(*(range(d) for d in self.spin_dims))))
        self.ncolor = color_basis(self.sector).shape[1]
        self.nspin = len(self.spins)
        self.shape = (len(self.longitudinal),len(self.orbitals),self.ncolor,self.nspin)
        twice_sz = sum((2*self.spins[:,i]-1)*2 if s=="g" else
                       2*(self.spins[:,i]%2)-1 for i,s in enumerate(SPECIES[self.sector]))
        twice_iz = sum(np.where(self.spins[:,i]<2,1,-1)*(1 if s=="q" else -1)
                       for i,s in enumerate(SPECIES[self.sector]) if s!="g")
        lz = np.array([sum(o[:4])-sum(o[4:]) for o in self.orbitals])
        allowed = ((2*lz[:,None]+twice_sz[None,:]==self.jz_twice)&
                   (twice_iz[None,:]==(1 if self.target=="PROTON" else -1)))
        mask = np.broadcast_to(allowed[None,:,None,:],self.shape)
        self.rows = np.flatnonzero(mask.ravel())
        self.lookup = np.full(np.prod(self.shape),-1,dtype=int)
        self.lookup[self.rows] = np.arange(len(self.rows))
        self.coordinates = np.array(np.unravel_index(self.rows,self.shape)).T
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
        """Add (orbital,color,selected spin,source column) amplitudes."""
        raw = (((longitudinal_index*len(self.orbitals)+np.arange(len(self.orbitals))[:,None,None])
                *self.ncolor+np.arange(self.ncolor)[None,:,None])*self.nspin+
                np.asarray(spins)[None,None,:])
        rows = self.lookup[raw]
        outside = rows<0
        if np.linalg.norm(amplitudes[outside])>2e-11:
            raise ArithmeticError("cubic image violates exact Iz/Jz")
        output[rows[~outside]] += amplitudes[~outside]

    def permute(self,values,permutation):
        """Simultaneously permute fractions, HO coordinates, color and spin."""
        value = np.asarray(values,complex)
        if value.ndim!=2 or value.shape[0]!=self.dimension or not np.all(np.isfinite(value)):
            raise ValueError("packed intermediate matrix required")
        p = tuple(permutation)
        c = color_permutation(self.sector,p)
        spin_lookup = np.arange(self.nspin).reshape(self.spin_dims).transpose(p).ravel()
        long_lookup = {s:i for i,s in enumerate(self.longitudinal)}
        result = np.zeros_like(value)
        for li,state in enumerate(self.longitudinal):
            target,rotation = permutation_frame(self.sector,state,p)
            orbital = oscillator_rotation(rotation,self.cutoff)
            for m,orbitals,spins in self.symmetry_groups:
                inverse_spin=np.full(self.nspin,-1,dtype=int)
                inverse_spin[spins]=np.arange(len(spins))
                order=inverse_spin[spin_lookup[spins]]
                if np.any(order<0):
                    raise ArithmeticError("species permutation changed exact spin/charge block")
                outside=np.setdiff1d(np.arange(len(self.orbitals)),orbitals)
                if np.linalg.norm(orbital[np.ix_(outside,orbitals)])>3e-12:
                    raise ArithmeticError("species permutation changed orbital Lz")
                rows=self.group_rows[li,m]
                block=value[rows].reshape(len(orbitals),self.ncolor,len(spins),value.shape[1])
                block=np.einsum("oi,ab,ibsk->oask",orbital[np.ix_(orbitals,orbitals)],
                                c,block,optimize=True)
                target_rows=self.group_rows[long_lookup[target],m]
                result[target_rows]=block[:,:,order].reshape(len(target_rows),value.shape[1])
        return result

    def statistics_project(self,values):
        if self.sector=="qqqgg":
            return (values+self.permute(values,(0,1,2,4,3)))/2
        result = np.zeros_like(values)
        for p in permutations(range(4)):
            sign = (-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))
            result += sign*self.permute(values,p+(4,))/factorial(4)
        return result

    def free_energies(self,b_GeV,quark_mass_GeV,gluon_mass_GeV):
        """Finite-shell H0 diagonal for N_exc <=1; not valid at cutoff 2."""
        if self.cutoff>1:
            raise ValueError("non-diagonal free shell: use ShellFreePropagation")
        b,mq,mg = map(float,(b_GeV,quark_mass_GeV,gluon_mass_GeV))
        if not np.all(np.isfinite([b,mq,mg])) or b<=0 or min(mq,mg)<0:
            raise ValueError("positive width and nonnegative finite masses required")
        masses = np.array([mg if s=="g" else mq for s in SPECIES[self.sector]])
        longitudinal = [np.sum(masses**2/frame(self.sector,s).fractions) for s in self.longitudinal]
        orbital = [b*b*(4+sum(o)) for o in self.orbitals]
        return np.array(longitudinal)[self.coordinates[:,0]]+np.array(orbital)[self.coordinates[:,1]]


@dataclass
class ExcludedSource:
    retained_pilot: object
    space: IntermediateSpace
    retained_columns: np.ndarray
    components: dict
    matrix_per_gs: np.ndarray

    def self_energy(self,energy,free_energies):
        """Leading g_s² coefficient V_PQ (E-H0_QQ)^-1 V_QP.

        Only propagating cubic intermediates. The canonical Wick one-body
        partners are derived separately in m2_canonical_gluon_self; common
        loop-regulator completion and physical finite matching remain open.
        """
        denominators = float(energy)-np.asarray(free_energies,float)
        if denominators.shape!=(self.space.dimension,) or not np.all(np.isfinite(denominators)):
            raise ValueError("one finite energy per intermediate row required")
        if np.min(np.abs(denominators))<1e-10:
            raise ValueError("intermediate pole; do not clip denominators")
        v = self.matrix_per_gs
        return v.conj().T@(v/denominators[:,None])

    def norm_correction(self,energy,free_energies):
        denominators = float(energy)-np.asarray(free_energies,float)
        self.self_energy(energy,free_energies)  # common validation
        image = self.matrix_per_gs/denominators[:,None]
        return image.conj().T@image


def build_excluded_source(pilot,sector,*,out_cutoff,_retained_vectors=None,_orbital_kernel=None):
    """Bare cubic source from the caller's H2 qqqg columns, not a physical fit."""
    space = IntermediateSpace(sector,pilot.total_k2,out_cutoff,pilot.target,pilot.jz_twice)
    if _retained_vectors is None and pilot.cutoff>out_cutoff:
        raise ValueError("intermediate cutoff must include parent shell")
    incoming = longitudinal_states("qqqg",pilot.total_k2)
    ni = len(oscillator_shell(3,pilot.cutoff))
    ranks = [u.shape[1] for u in pilot.bases]
    columns = np.arange(ranks[0],sum(ranks))
    from .m2_source_vectors import SourceImage, source_tensor, image_symmetry_tolerance
    if _retained_vectors is None:
        count = len(columns)
        source = np.zeros((len(incoming),ni,2,64,2,count),complex)
        offset = 0
        for hg,u in enumerate(pilot.bases[1:]):
            n = u.shape[1]
            source[:,:,:,:,hg,offset:offset+n] = u.toarray().reshape(len(incoming),ni,2,64,n)
            offset += n
    else:
        source = source_tensor(pilot, "qqqg", _retained_vectors)
        count = source.shape[-1]
    ins,outs = ({s:i for i,s in enumerate(states)} for states in (incoming,space.longitudinal))
    sf = np.array(list(product(range(4),repeat=3)))
    b,m = pilot.width_GeV,pilot.quark_mass_GeV
    components = {}
    channels = ("q_to_qg","g_to_gg") if sector=="qqqgg" else ("g_to_pair",)
    for channel in channels:
        image = np.zeros((space.dimension,count),complex)
        for route in split_routes(pilot.total_k2,channel):
            color = color_source(channel,route.parent)
            for h1,h2,h0 in product((-1,1),repeat=3):
                orbital = (cubic_orbital if _orbital_kernel is None else _orbital_kernel)(
                    route,pilot.cutoff,out_cutoff,b,m,(h1,h2,h0))
                if not np.any(orbital):
                    continue  # exact polynomial/Jz selection, not tolerance pruning
                if channel=="q_to_qg":
                    selected = np.flatnonzero(2*(sf[:,route.parent]%2)-1==h0)
                    changed = sf[selected].copy()
                    changed[:,route.parent] = 2*(changed[:,route.parent]//2)+(h1+1)//2
                    out_sf = np.ravel_multi_index(changed.T,(4,4,4))
                    for old_hg in range(2):
                        value = np.take(source[ins[route.incoming],:,:,:,old_hg],selected,axis=2)
                        amp = np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                        spins = (out_sf*2+old_hg)*2+(h2+1)//2
                        space.add(image,outs[route.outgoing],spins,amp)
                else:
                    value = source[ins[route.incoming],:,:,:,(h0+1)//2]
                    amp = np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                    if channel=="g_to_gg":
                        spins = (np.arange(64)*2+(h1+1)//2)*2+(h2+1)//2
                        space.add(image,outs[route.outgoing],spins,amp)
                    else:
                        # Coherent flavor singlet u ubar + d dbar; no fitted weights.
                        for flavor in range(2):
                            q,anti = 2*flavor+(h1+1)//2,2*flavor+(h2+1)//2
                            spins = (np.arange(64)*4+q)*4+anti
                            space.add(image,outs[route.outgoing],spins,amp)
        if channel=="q_to_qg":
            # a† on a one-gluon state: sqrt(2) S_2 on ordered-slot wavefunctions.
            components[channel] = sqrt(2)*space.statistics_project(image)
        elif channel=="g_to_gg":
            # Distinguished-daughter M includes both contractions: M/sqrt(2!).
            if np.linalg.norm(space.statistics_project(image)-image)>image_symmetry_tolerance(
                    image, vector_action=_retained_vectors is not None):
                raise ArithmeticError("three-gluon image violates Bose symmetry")
            components[channel] = image/sqrt(2)
        else:
            # Four identical quarks: sqrt(4) A_4 applied to inherited A_3 state.
            components[channel] = 2*space.statistics_project(image)
    total = sum(components.values())
    if _retained_vectors is not None:
        return SourceImage(pilot, space, 1, total, components)
    return ExcludedSource(pilot,space,columns,components,total)


def apply_excluded_source(pilot, sector, vectors, *, out_cutoff):
    """Apply both canonical cubic channels before raw-image allocation.

    Rectangular cutoffs are allowed: inherited parent excitations above the
    outgoing shell have exactly zero image in these common-width kernels.
    No amplitude is pruned and no source/basis column labels are invented.
    """
    from .m2_source_vectors import validate_source_vectors
    return build_excluded_source(pilot, sector, out_cutoff=out_cutoff,
                                 _retained_vectors=validate_source_vectors(pilot, vectors))
