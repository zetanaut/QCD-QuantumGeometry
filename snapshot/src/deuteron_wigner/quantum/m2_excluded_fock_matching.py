"""Source-reachable free-Q reference and actual higher-Fock H2 matching.

Q contains qqqgg and qqqq-qbar, never the retained qqqg propagation.
Only the newly derived cubic exits and free Q Hamiltonian are included.
This is a bare finite regulator diagnostic, NOT renormalized QCD. Source
compression within exactly degenerate free-energy spaces changes no
resolvent or reconstructed state beyond reported roundoff; it is not
a variational ansatz, amplitude cutoff, or physical Fock truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import linalg

from .m2_coupled_pilot import _displacement
from .m2_canonical_good_current import good_current_density,parton_charge
from .m2_excluded_fock_kinematics import SPECIES, frame
from .m2_matched_h2_reduction import FiniteParentReduction


def _energy_groups(space):
    """Exact rational mass coefficients and oscillator occupations, not float bins."""
    keys=[]
    for modes in space.longitudinal:
        q=sum((Fraction(space.total_k2,k) for k,s in zip(modes,SPECIES[space.sector])
               if s!="g"),Fraction(0))
        g=sum((Fraction(space.total_k2,k) for k,s in zip(modes,SPECIES[space.sector])
               if s=="g"),Fraction(0))
        keys.append((q,g))
    groups={}
    for row,(li,orb,_,_) in enumerate(space.coordinates):
        key=keys[li]+(sum(space.orbitals[orb]),)
        groups.setdefault(key,[]).append(row)
    return tuple(np.array(rows,dtype=int) for rows in groups.values())


def intermediate_density_action(space,values,transfer,*,b_GeV,charge="baryon"):
    """P_Q rho(Delta) P_Q, resolved flavor charges, opposite antiquark sign.

    A finite transfer can change Jz. Both endpoints here use this space's
    declared Jz, so only the same-Jz block is returned; no other blocks are
    asserted to vanish. Operator is number preserving at Wilson order zero.
    """
    return intermediate_density_action_between(space,space,values,transfer,b_GeV=b_GeV,charge=charge)


def intermediate_density_action_between(outgoing,incoming,values,transfer,*,b_GeV,charge):
    """Bare local good current between distinct same-regulator Jz fibers."""
    for name in ("sector","total_k2","cutoff","target"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("intermediate current endpoints disagree on "+name)
    space=incoming
    a=np.asarray(values,complex)
    delta=np.asarray(transfer,float)
    b=float(b_GeV)
    if a.ndim!=2 or a.shape[0]!=space.dimension or not np.all(np.isfinite(a)):
        raise ValueError("finite packed intermediate column matrix required")
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b<=0:
        raise ValueError("finite transverse transfer and positive width required")
    result=np.zeros((outgoing.dimension,a.shape[1]),complex)
    n=len(space.orbitals)
    for li,state in enumerate(space.longitudinal):
        density=np.zeros((n,n,space.nspin),complex)
        f=frame(space.sector,state)
        for particle,species in enumerate(SPECIES[space.sector]):
            if species=="g":
                continue
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


@dataclass(frozen=True)
class FreeEnergySlice:
    rows: np.ndarray
    isometry: np.ndarray
    compressed_columns: slice
    energy_GeV2: float


@dataclass(frozen=True)
class ReachableSector:
    source: object
    slices: tuple
    coupling_per_gs: np.ndarray
    energies_GeV2: np.ndarray
    reconstruction_residual_per_gs: float

    def lift(self,values):
        a=np.asarray(values,complex)
        if a.ndim!=2 or a.shape[0]!=len(self.energies_GeV2):
            raise ValueError("compressed sector columns required")
        out=np.zeros((self.source.space.dimension,a.shape[1]),complex)
        for group in self.slices:
            out[group.rows]=group.isometry@a[group.compressed_columns]
        return out

    def project(self,values):
        a=np.asarray(values,complex)
        if a.ndim!=2 or a.shape[0]!=self.source.space.dimension:
            raise ValueError("packed sector columns required")
        return np.vstack([group.isometry.conj().T@a[group.rows] for group in self.slices])

    def density(self,transfer,*,b_GeV,charge="baryon"):
        basis=self.lift(np.eye(len(self.energies_GeV2)))
        return self.project(intermediate_density_action(
            self.source.space,basis,transfer,b_GeV=b_GeV,charge=charge))


def compress_free_source(source,*,b_GeV,quark_mass_GeV,gluon_mass_GeV):
    """Exact source-reachable free-Q invariant range, with numerical-rank audit."""
    energy=source.space.free_energies(b_GeV,quark_mass_GeV,gluon_mass_GeV)
    v=source.matrix_per_gs
    slices=[]; blocks=[]; energies=[]; offset=0; error2=0.
    for rows in _energy_groups(source.space):
        block=v[rows]
        u,s,_=linalg.svd(block,full_matrices=False)
        # Resolve only floating-point null singular values; measure what is lost.
        tolerance=8*np.finfo(float).eps*max(block.shape)*s[0]
        rank=int(np.sum(s>tolerance))
        if rank==0:
            error2+=float(np.linalg.norm(block)**2)
            continue
        u=u[:,:rank]
        reduced=u.conj().T@block
        error2+=float(np.linalg.norm(block-u@reduced)**2)
        if np.ptp(energy[rows])>3e-14:
            raise ArithmeticError("exact source labels do not share free energy")
        e=float(energy[rows[0]])
        slices.append(FreeEnergySlice(rows,u,slice(offset,offset+rank),e))
        blocks.append(reduced); energies.extend([e]*rank); offset+=rank
    error=np.sqrt(error2)
    if error>2e-12*max(1.,np.linalg.norm(v)):
        raise ArithmeticError("free-Q source compression loses resolved physics")
    if not blocks:
        raise ValueError("empty cubic source; no source-reachable reference")
    return ReachableSector(source,tuple(slices),np.vstack(blocks),np.array(energies),float(error))


class ExcludedFockReference:
    """Independent finite parent with explicit source-qualified higher-Fock exits.

    H_PP is the caller's partial H2 pilot. H_QQ is the finite free shell;
    H_QP is g_s times the coherent cubic source. Solving this finite parent
    resums returns through P but does NOT generate omitted QQ interactions,
    counterterms, unsubtracted instantaneous Wick traces, or higher-order
    matching. The Wick traces and their explicit normal-order subtraction
    are now derived separately in m2_canonical_gluon_self.
    """
    def __init__(self,pilot,sources):
        self.pilot=pilot
        sources=tuple(sources)
        if not sources or len({s.space.sector for s in sources})!=len(sources):
            raise ValueError("distinct explicitly supplied excluded sectors required")
        for s in sources:
            if s.retained_pilot is not pilot:
                raise ValueError("source must belong to this exact retained pilot/basis")
            for attr,expected in (("total_k2",pilot.total_k2),("target",pilot.target),
                                  ("jz_twice",pilot.jz_twice)):
                if getattr(s.space,attr)!=expected:
                    raise ValueError("excluded source does not match retained "+attr)
            if not np.array_equal(s.retained_columns,np.arange(pilot.bases[0].shape[1],len(pilot.matrix))):
                raise ValueError("actual retained qqqg source columns required")
        self.sectors=tuple(compress_free_source(s,b_GeV=pilot.width_GeV,
            quark_mass_GeV=pilot.quark_mass_GeV,gluon_mass_GeV=pilot.gluon_mass_GeV) for s in sources)
        np_=len(pilot.matrix)
        poles=np.concatenate([s.energies_GeV2 for s in self.sectors])
        coupling=np.zeros((len(poles),np_),complex)
        offset=0
        for s in self.sectors:
            n=len(s.energies_GeV2)
            coupling[offset:offset+n,s.source.retained_columns]=pilot.coupling*s.coupling_per_gs
            offset+=n
        self.matrix=np.block([[pilot.matrix,coupling.conj().T],[coupling,np.diag(poles)]])
        self.retained_isometry=np.eye(len(self.matrix),dtype=complex)[:,:np_]
        self.reduction=FiniteParentReduction(self.matrix,self.retained_isometry,
            scope="bare H2 plus cubic qqqgg/qqqq-qbar exits with free finite Q; NOT renormalized QCD")

    def density(self,transfer,*,charge="baryon"):
        return linalg.block_diag(good_current_density(self.pilot,transfer,charge=charge),*(
            s.density(transfer,b_GeV=self.pilot.width_GeV,charge=charge) for s in self.sectors))

    def flow_current(self,transfer,*,charge="baryon"):
        delta=np.asarray(transfer,float)
        rho=self.density(delta,charge=charge)
        norm=float(delta@delta)
        if norm==0:
            raise ValueError("nonzero transfer required")
        comm=self.matrix@rho-rho@self.matrix
        return rho,delta[0]*comm/norm,delta[1]*comm/norm

    def full_reconstructed_residual(self,energy,retained):
        """Check the UNCOMPRESSED Q equations as well as the retained equation."""
        v=np.asarray(retained,complex)
        if v.shape!=(len(self.pilot.matrix),):
            raise ValueError("one retained state vector required")
        rp=self.pilot.matrix@v-float(energy)*v
        normq=0.; residualq=0.
        for sector in self.sectors:
            s=sector.source
            energies=s.space.free_energies(self.pilot.width_GeV,
                        self.pilot.quark_mass_GeV,self.pilot.gluon_mass_GeV)
            s.self_energy(energy,energies)  # reject poles, no denominator clipping
            coupling=self.pilot.coupling*s.matrix_per_gs
            q=coupling@v[s.retained_columns]/(energy-energies)
            rp[s.retained_columns]+=coupling.conj().T@q
            rq=(energies-energy)*q+coupling@v[s.retained_columns]
            normq+=float(np.vdot(q,q).real)
            residualq+=float(np.vdot(rq,rq).real)
        return float(np.sqrt(np.vdot(rp,rp).real+residualq)),normq
