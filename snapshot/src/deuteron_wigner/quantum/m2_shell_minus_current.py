"""Complete raw Q5/Q6 free minus-current action, including spin flips.

The free constrained-field kernel retains both oscillator momenta before
projection, including all N+2 outside neighbors. No interaction, Wick,
regulator or loop current is implicitly supplied.
"""
import numpy as np
from .m2_canonical_good_current import parton_charge
from .m2_excluded_fock_kinematics import SPECIES as FIVE,frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX,frame as frame6
from .m2_free_minus_current import projected_free_minus
from .m2_six_body_transverse_contact import _block,_add


def shell_minus_current_between(outgoing,incoming,values,transfer,*,b_GeV,
                                total_pplus_GeV,mass_GeV,charge,
                                mean_total_transverse=(0.,0.)):
    """J-minus/(2P+) on every supplied raw column; both Jz endpoints explicit."""
    for name in ("sector","total_k2","cutoff","target"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    if incoming.orbitals!=outgoing.orbitals or not np.array_equal(incoming.spins,outgoing.spins):
        raise ValueError("ordered orbital/spin coordinates must agree")
    s=incoming;a=np.asarray(values,complex);delta=np.asarray(transfer,float)
    mean=np.asarray(mean_total_transverse,float);b=float(b_GeV);pp=float(total_pplus_GeV)
    mass=float(mass_GeV)
    if (a.ndim!=2 or a.shape[0]!=s.dimension or not np.all(np.isfinite(a))
            or delta.shape!=(2,) or mean.shape!=(2,)
            or not np.all(np.isfinite(np.r_[delta,mean,b,pp,mass]))
            or min(b,pp)<=0 or mass<0):
        raise ValueError("finite columns/kinematics, positive width/P+ and nonnegative mass required")
    if s.sector in FIVE:species,frame=FIVE[s.sector],frame5
    elif s.sector in SIX:species,frame=SIX[s.sector],frame6
    else:raise ValueError("explicit five- or six-body space required")
    d=len(species)-1;result=np.zeros((outgoing.dimension,a.shape[1]),complex)
    for li,modes in enumerate(s.longitudinal):
        f=frame(s.sector,modes)
        for active,parton in enumerate(species):
            if parton=="g":continue
            x=f.fractions[active]
            kernel=projected_free_minus(d,s.cutoff,f.forward[:-1,active,None]*delta,
                f.inverse[active,:-1],b_GeV=b,pplus_GeV=x*pp,
                outgoing_offset=x*mean+(1-x/2)*delta,incoming_offset=x*mean-x*delta/2,
                mass_GeV=mass,species=parton)
            for hi in (0,1):
                selected=np.flatnonzero(s.spins[:,active]%2==hi)
                block=_block(s,li,selected,a)
                block*=parton_charge(s.spins[selected,active],parton,charge)[None,None,:,None]
                for hf in (0,1):
                    changed=s.spins[selected].copy()
                    changed[:,active]=2*(changed[:,active]//2)+hf
                    target_spins=np.ravel_multi_index(changed.T,s.spin_dims)
                    output=np.einsum("oi,icsk->ocsk",kernel[hf,hi],block,optimize=True)
                    _add(outgoing,li,target_spins,output,result)
    return result
