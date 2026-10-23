"""Exact packed complete-shell local good density for Q5/Q6.

The same projected continuum displacement as the preceding current is
factored by particle and exact spin/Lz blocks. No density, wavefunction
tail, or source is thresholded. This is Wilson-order-zero J+, not the
unconstructed complete physical four-current.
"""
import numpy as np

from .m2_coupled_pilot import _displacement
from .m2_canonical_good_current import parton_charge
from .m2_excluded_fock_kinematics import SPECIES as FIVE,frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX,frame as frame6


def shell_density_between(outgoing,incoming,values,transfer,*,b_GeV,charge):
    for name in ("sector","total_k2","cutoff","target"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    if incoming.orbitals!=outgoing.orbitals:
        raise ValueError("ordered oscillator coordinates must agree")
    s=incoming
    a=np.asarray(values,complex);delta=np.asarray(transfer,float);b=float(b_GeV)
    if a.ndim!=2 or a.shape[0]!=s.dimension or not np.all(np.isfinite(a)):
        raise ValueError("finite packed incoming columns required")
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or not np.isfinite(b) or b<=0:
        raise ValueError("finite transfer and positive width required")
    if s.sector in FIVE:
        species,frame=FIVE[s.sector],frame5
    elif s.sector in SIX:
        species,frame=SIX[s.sector],frame6
    else:
        raise ValueError("five- or six-particle current space required")
    d=len(species)-1;n=len(s.orbitals)
    occupations=np.array(s.orbitals)
    lz=occupations[:,:d].sum(axis=1)-occupations[:,d:].sum(axis=1)
    result=np.zeros((outgoing.dimension,a.shape[1]),complex)
    weights={i:parton_charge(s.spins[:,i],parton,charge)
             for i,parton in enumerate(species) if parton!="g"}
    for li,state in enumerate(s.longitudinal):
        f=frame(s.sector,state)
        for particle,weight in weights.items():
            shift=f.forward[:-1,particle,None]*delta[None,:]
            plus=(shift[:,0]-1j*shift[:,1])/(2*b)
            alpha=np.r_[plus,plus.conj()]
            density=np.ones((n,n),complex)
            # Factor each circular coordinate exactly, before restricting Jz.
            for k,z in enumerate(alpha):
                table=np.array([[_displacement(o,i,z) for i in range(s.cutoff+1)]
                                for o in range(s.cutoff+1)])
                density*=table[occupations[:,k,None],occupations[None,:,k]]
            for m in np.unique(lz):
                oi=np.flatnonzero(lz==m)
                rows=np.flatnonzero((s.coordinates[:,0]==li)
                                    &np.isin(s.coordinates[:,1],oi))
                if not len(rows):
                    continue
                spins=np.unique(s.coordinates[rows,3])
                # J+ preserves individual parton helicities. Exact outgoing
                # Jz determines the new orbital Lz for these same spin labels.
                mf=m+(outgoing.jz_twice-incoming.jz_twice)/2
                of=np.flatnonzero(lz==mf)
                target=np.flatnonzero((outgoing.coordinates[:,0]==li)
                     &np.isin(outgoing.coordinates[:,1],of)
                     &np.isin(outgoing.coordinates[:,3],spins))
                if not len(target):
                    continue
                if len(rows)!=len(oi)*s.ncolor*len(spins) or len(target)!=len(of)*s.ncolor*len(spins):
                    raise ValueError("incomplete exact orbital/color/spin block")
                block=a[rows].reshape(len(oi),s.ncolor,len(spins),a.shape[1])
                block=block*weight[spins][None,None,:,None]
                value=density[np.ix_(of,oi)]@block.reshape(len(oi),-1)
                result[target]+=value.reshape(len(target),a.shape[1])
    return result
