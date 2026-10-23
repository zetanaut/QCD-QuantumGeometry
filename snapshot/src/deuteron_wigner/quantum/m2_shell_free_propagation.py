"""Non-diagonal free propagation on complete global-Jacobi Q5/Q6 shells.

This is P_N (sum_a Q_a^2 + sum_i m_i^2/x_i) P_N, not a diagonal
oscillator Hamiltonian and not P_N (E-H_continuum)^-1 P_N. It changes
neither the accepted regulator nor a physical finite matching condition.
Only orbital Lz blocks are diagonalized; all color/spin/source columns
are carried without amplitude compression.
"""
from dataclasses import dataclass

import numpy as np

from .m2_coupled_longitudinal_jacobi import oscillator_shell
from .m2_coupled_pilot import free_orbital
from .m2_excluded_fock_kinematics import SPECIES as FIVE, frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX, frame as frame6


@dataclass(frozen=True)
class _Block:
    rows: np.ndarray
    orbital: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mass: float


class ShellFreePropagation:
    """Exact finite-shell H0 action and real off-pole resolvent.

    In momentum circular phases the radial raising/lowering entries are
    +b^2 sqrt((n_+ + 1)(n_- + 1)); they first enter at N_exc=2.
    """

    def __init__(self,space,*,b_GeV,quark_mass_GeV,gluon_mass_GeV):
        b,mq,mg=map(float,(b_GeV,quark_mass_GeV,gluon_mass_GeV))
        if not np.all(np.isfinite([b,mq,mg])) or b<=0 or min(mq,mg)<0:
            raise ValueError("positive width and nonnegative finite masses required")
        if space.sector in FIVE:
            species,frame=FIVE[space.sector],frame5
        elif space.sector in SIX:
            species,frame=SIX[space.sector],frame6
        else:
            raise ValueError("explicit five- or six-particle space required")
        self.space=space
        self.width_GeV=b
        self.quark_mass_GeV=mq
        self.gluon_mass_GeV=mg
        d=len(species)-1
        if tuple(space.orbitals)!=oscillator_shell(d,space.cutoff):
            raise ValueError("complete ordered circular oscillator shell required")
        orbital=free_orbital(d,space.cutoff,b).toarray()
        lz=np.array([sum(o[:d])-sum(o[d:]) for o in space.orbitals])
        masses=np.array([mg if s=="g" else mq for s in species])
        self.mass_offsets=np.array([np.sum(masses**2/frame(space.sector,s).fractions)
                                    for s in space.longitudinal])
        coordinates=np.asarray(space.coordinates)
        self.blocks=[]
        count=0
        for m in np.unique(lz):
            indices=np.flatnonzero(lz==m)
            t=orbital[np.ix_(indices,indices)]
            outside=np.flatnonzero(lz!=m)
            if np.any(orbital[np.ix_(outside,indices)]!=0):
                raise ArithmeticError("free propagation changed exact orbital Lz")
            w,u=np.linalg.eigh(t)
            for li,mass in enumerate(self.mass_offsets):
                rows=np.flatnonzero((coordinates[:,0]==li)
                                    &np.isin(coordinates[:,1],indices))
                if not len(rows):
                    continue
                spin=np.unique(coordinates[rows,3])
                expected=np.array([(o,c,s) for o in indices
                                   for c in range(space.ncolor) for s in spin])
                if not np.array_equal(coordinates[rows,1:],expected):
                    raise ValueError("incomplete or reordered orbital/color/spin block")
                self.blocks.append(_Block(rows,t,w,u,float(mass)))
                count+=len(rows)
        if count!=space.dimension or not self.blocks:
            raise ValueError("packed free blocks do not cover the supplied space")
        self.minimum_energy_GeV2=min(b.mass+b.eigenvalues[0] for b in self.blocks)
        self.maximum_energy_GeV2=max(b.mass+b.eigenvalues[-1] for b in self.blocks)

    def _columns(self,values):
        a=np.asarray(values,complex)
        if a.ndim!=2 or a.shape[0]!=self.space.dimension or not np.all(np.isfinite(a)):
            raise ValueError("finite packed column matrix required")
        return a

    def apply(self,values):
        a=self._columns(values)
        result=np.zeros_like(a)
        for block in self.blocks:
            v=a[block.rows].reshape(len(block.eigenvalues),-1)
            result[block.rows]=(block.orbital@v+block.mass*v).reshape(len(block.rows),a.shape[1])
        return result

    def resolve(self,energy,values):
        """(E-H0_N)^-1 on every supplied raw column; never clip a pole."""
        e=float(energy)
        if not np.isfinite(e):
            raise ValueError("finite off-pole energy required")
        a=self._columns(values)
        result=np.zeros_like(a)
        for block in self.blocks:
            denominators=e-(block.mass+block.eigenvalues)
            if np.min(abs(denominators))<1e-10:
                raise ValueError("free intermediate pole; do not clip denominators")
            v=a[block.rows].reshape(len(block.eigenvalues),-1)
            u=block.eigenvectors
            result[block.rows]=(u@((u.T@v)/denominators[:,None])).reshape(len(block.rows),a.shape[1])
        return result
