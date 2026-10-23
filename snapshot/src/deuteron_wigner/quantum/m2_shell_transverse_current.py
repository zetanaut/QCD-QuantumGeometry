"""Packed Q5/Q6 canonical free transverse-current action, not full matching."""
import numpy as np

from .m2_canonical_good_current import parton_charge
from .m2_excluded_fock_kinematics import SPECIES as FIVE,frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX,frame as frame6
from .m2_transverse_current_projection import displacement_and_moments


def shell_transverse_current_between(outgoing,incoming,values,transfer,*,
                                     component,b_GeV,total_pplus_GeV,charge,
                                     mean_total_transverse=(0.,0.)):
    """One component of J_T/(2P+) on every supplied raw Q5/Q6 column.

    component=0 or 1 means x or y. The same exact displaced-momentum kernel
    as the retained current preserves every color, helicity and orbital
    tail. Antiquark charge includes its CAR minus; its physical helicity
    uses the independently checked charge-stripped free Dirac kernel.
    This one-body part alone is NOT a conserved current for finite H2.
    """
    for name in ("sector","total_k2","cutoff","target"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    if incoming.orbitals!=outgoing.orbitals or not np.array_equal(incoming.spins,outgoing.spins):
        raise ValueError("ordered orbital and spin coordinates must agree")
    s=incoming
    a=np.asarray(values,complex);delta=np.asarray(transfer,float)
    mean=np.asarray(mean_total_transverse,float);b=float(b_GeV);pplus=float(total_pplus_GeV)
    if (a.ndim!=2 or a.shape[0]!=s.dimension or not np.all(np.isfinite(a))
            or delta.shape!=(2,) or mean.shape!=(2,)
            or not np.all(np.isfinite(np.r_[delta,mean,b,pplus])) or min(b,pplus)<=0
            or isinstance(component,(bool,np.bool_)) or component not in (0,1)):
        raise ValueError("finite packed columns, vectors, positive scales and explicit x/y required")
    j=int(component)
    if s.sector in FIVE:
        species,frame=FIVE[s.sector],frame5
    elif s.sector in SIX:
        species,frame=SIX[s.sector],frame6
    else:
        raise ValueError("five- or six-particle current space required")
    d=len(species)-1
    occupations=np.array(s.orbitals)
    lz=occupations[:,:d].sum(axis=1)-occupations[:,d:].sum(axis=1)
    result=np.zeros((outgoing.dimension,a.shape[1]),complex)
    cross=np.array([-delta[1],delta[0]])
    charges={i:parton_charge(s.spins[:,i],parton,charge)
             for i,parton in enumerate(species) if parton!="g"}
    for li,state in enumerate(s.longitudinal):
        f=frame(s.sector,state)
        groups=[]
        for m in np.unique(lz):
            oi=np.flatnonzero(lz==m)
            rows=np.flatnonzero((s.coordinates[:,0]==li)&np.isin(s.coordinates[:,1],oi))
            if not len(rows):
                continue
            spins=np.unique(s.coordinates[rows,3])
            of=np.flatnonzero(lz==m+(outgoing.jz_twice-incoming.jz_twice)/2)
            target=np.flatnonzero((outgoing.coordinates[:,0]==li)
                &np.isin(outgoing.coordinates[:,1],of)&np.isin(outgoing.coordinates[:,3],spins))
            if not len(target):
                continue
            if len(rows)!=len(oi)*s.ncolor*len(spins) or len(target)!=len(of)*s.ncolor*len(spins):
                raise ValueError("incomplete exact orbital/color/spin block")
            groups.append((oi,of,rows,target,spins))
        for particle,weights in charges.items():
            x=f.fractions[particle]
            shift=f.forward[:-1,particle,None]*delta[None,:]
            density,moments=displacement_and_moments(d,s.cutoff,shift,b_GeV=b)
            convection=np.einsum("a,aoi->oi",f.inverse[particle,:-1],moments[:,j])
            convection+=((1-x)*delta[j]/2+x*mean[j])*density
            for oi,of,rows,target,spins in groups:
                block=a[rows].reshape(len(oi),s.ncolor,len(spins),a.shape[1])
                weighted=block*weights[spins][None,None,:,None]
                helicity=2*(s.spins[spins,particle]%2)-1
                polarized=weighted*helicity[None,None,:,None]
                value=(convection[np.ix_(of,oi)]@weighted.reshape(len(oi),-1)
                    +.5j*cross[j]*density[np.ix_(of,oi)]@polarized.reshape(len(oi),-1))/(x*pplus)
                result[target]+=value.reshape(len(target),a.shape[1])
    return result
