"""Linear-momentum constrained J-minus contacts with exact oscillator moments.

Same signed Dirac kernels as the canonical transverse current. These are
tree-level one-gluon contacts, not two-gluon/Wick/loop completion. Color,
physical charge, field and Fock factors stay external to the orbital kernel.
"""
from functools import lru_cache
from math import pi,sqrt
import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.modes.core import polarization
from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes
from .m2_antiquark_contact_current import antiquark_one_gluon_current_spin
from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import (
    one_gluon_current_spin,pair_creation_current_spin,discrete_gluon_field_factor,
)
from .m2_coupled_longitudinal_jacobi import (
    oscillator_shell,oscillator_rotation,jacobi_frame,emission_coordinate_map,
    emission_routes,longitudinal_basis,
)
from .m2_coupled_pilot import _displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def local_linear_moments(d,in_cutoff,out_cutoff,shift,parent_weights,*,b_GeV):
    """Adapted-coordinate <1>, <kappa_parent^j>, <r^j> coincidence integrals.

    Output is in d parent plus one relative circular modes; the relative
    coordinate is integrated, not matched to an incoming oscillator.
    Each m=0 radial has integral b/sqrt(pi). Relative r uses BRA neighbors
    through N_out+1; parent momentum uses KET neighbors through N_in+1.
    """
    incoming=oscillator_shell(d,in_cutoff);outgoing=oscillator_shell(d+1,out_cutoff)
    shift=np.asarray(shift,float);w=np.asarray(parent_weights,float);b=float(b_GeV)
    if shift.shape!=(d,2) or w.shape!=(d,) or not np.all(np.isfinite(np.r_[shift.ravel(),w,b])) or b<=0:
        raise ValueError("finite shift/parent weights and positive width required")
    a=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[a,a.conj()]
    tables=np.array([[[_displacement(o,i,z) for i in range(in_cutoff+2)]
                      for o in range(out_cutoff+1)] for z in alpha])
    po=np.array([q[:d]+q[d+1:2*d+1] for q in outgoing]);ket=np.array(incoming)
    def overlap(q):
        value=np.ones((len(outgoing),len(incoming)),complex)
        for k in range(2*d):value*=tables[k,po[:,k,None],q[None,:,k]]
        return value
    density=overlap(ket);parent=np.zeros((2,*density.shape),complex)
    for k in range(2*d):
        for step in (-1,1):
            neighbor=ket.copy();neighbor[:,k]=np.maximum(ket[:,k]+step,0)
            value=overlap(neighbor)*b*w[k%d]*np.sqrt(ket[:,k]+(step==1))[None,:]/2
            parent[0]+=value
            parent[1]+=(-1j if k<d else 1j)*step*value
    relative=np.array([(q[d],q[2*d+1]) for q in outgoing])
    coincidence=(relative[:,0]==relative[:,1])*b/sqrt(pi)
    first=np.zeros((2,len(outgoing)),complex)
    for row,q in enumerate(relative):
        for k in (0,1):
            for step in (-1,1):
                neighbor=q.copy();neighbor[k]+=step
                if neighbor[k]<0 or neighbor[0]!=neighbor[1]:continue
                v=b*np.sqrt(q[k]+(step==1))/2*b/sqrt(pi)
                first[0,row]+=v
                first[1,row]+=np.conj((-1j if k==0 else 1j)*step)*v
    return density*coincidence[:,None],parent*coincidence[None,:,None],first[:,:,None]*density


def one_gluon_minus_orbital(parent_frame,active,x,y,rotation,in_cutoff,out_cutoff,
                            transfer,*,channel,b_GeV,total_pplus_GeV,mass_GeV,
                            gluon_helicity,mean_total_transverse=(0.,0.)):
    """Charge/color/field/Fock-stripped (h_out,h_in,orb_out,orb_in) contact.

    x,y are outgoing daughter fractions. Their sum is the active parent.
    q_to_qg and anti_to_antig use x for the fermion; g_to_pair uses x for
    the quark and y for the antiquark. All momenta are physical; signed
    Fourier denominators are supplied explicitly to their Dirac kernels.
    Antiquark emission factors BOTH physical charge and Tbar separately.
    """
    if channel not in ("q_to_qg","anti_to_antig","g_to_pair"):
        raise ValueError("explicit charged one-gluon contact channel required")
    delta=np.asarray(transfer,float);mean=np.asarray(mean_total_transverse,float)
    pp=float(total_pplus_GeV);x=float(x);y=float(y);xp=x+y
    if (delta.shape!=(2,) or mean.shape!=(2,) or not np.all(np.isfinite(np.r_[delta,mean,pp,x,y]))
            or min(pp,x,y)<=0 or gluon_helicity not in (-1,1)):
        raise ValueError("finite kinematics, positive fractions/P+ and physical helicity required")
    if abs(xp-parent_frame.fractions[active])>2e-14:
        raise ValueError("daughter fractions must sum to the explicit active parent")
    d=len(parent_frame.fractions)-1
    shift=parent_frame.forward[:-1,active,None]*delta
    zeroth,parent,relative=local_linear_moments(d,in_cutoff,out_cutoff,shift,
        parent_frame.inverse[active,:-1],b_GeV=b_GeV)
    scale=sqrt(x*y/xp)
    pg=(xp if channel=="g_to_pair" else y)*pp
    eps=polarization(pg,0.,0.,gluon_helicity)[2:]
    if channel!="g_to_pair":eps=eps.conj()
    def spin(kappa,r):
        pin=kappa+xp*(mean-delta/2)
        pout=x/xp*(pin+delta)+scale*r
        if channel=="g_to_pair":
            anti=y/xp*(pin+delta)-scale*r
            return pair_creation_current_spin((x*pp,*pout),(y*pp,*anti),
                mass_GeV=mass_GeV,polarization=eps,right_plus=x*pp,left_plus=-y*pp)[1]
        kernel=one_gluon_current_spin if channel=="q_to_qg" else antiquark_one_gluon_current_spin
        right=x*pp if channel=="q_to_qg" else -xp*pp
        left=xp*pp if channel=="q_to_qg" else -x*pp
        return kernel((x*pp,*pout),(xp*pp,*pin),mass_GeV=mass_GeV,
                      polarization=eps,right_plus=right,left_plus=left)[1]
    zero=np.zeros(2);base=spin(zero,zero)
    result=base[:,:,None,None]*zeroth
    for j in (0,1):
        unit=np.eye(2)[j]
        result+=(spin(unit,zero)-base)[:,:,None,None]*parent[j]
        result+=(spin(zero,unit)-base)[:,:,None,None]*relative[j]
    # The relative coincidence Jacobian is independent of polynomial degree.
    return scale*np.einsum("ao,hkoi->hkai",oscillator_rotation(rotation,out_cutoff).T,result)


@lru_cache(maxsize=1024)
def retained_route_minus(route,in_cutoff,out_cutoff,b,mass,pp,hg,delta,mean):
    fi=jacobi_frame(route.incoming);fo=jacobi_frame(route.outgoing)
    x,y=fo.fractions[[route.emitter,3]]
    value=one_gluon_minus_orbital(fi,route.emitter,x,y,emission_coordinate_map(route),
        in_cutoff,out_cutoff,delta,channel="q_to_qg",b_GeV=b,total_pplus_GeV=pp,
        mass_GeV=mass,gluon_helicity=hg,mean_total_transverse=mean)
    value.setflags(write=False)
    return value


def emission_minus_contact(total_k2,in_cutoff,out_cutoff,transfer,*,b_GeV,mass_GeV,
                            total_pplus_GeV,gluon_helicity,charge,
                            mean_total_transverse=(0.,0.)):
    """Raw qqq→qqqg J-minus contact per gs, with all spin flips and emitters."""
    q,g=(longitudinal_basis(s,total_k2) for s in ("qqq","qqqg"))
    nq,ng=len(oscillator_shell(2,in_cutoff)),len(oscillator_shell(3,out_cutoff))
    qi,gi=({s:i for i,s in enumerate(states)} for states in (q,g))
    sf=SPIN_FLAVOR_BASIS;lookup={state:i for i,state in enumerate(sf)}
    rows=[];cols=[];values=[]
    for route in emission_routes(total_k2):
        orbital=retained_route_minus(route,in_cutoff,out_cutoff,b_GeV,mass_GeV,
            total_pplus_GeV,gluon_helicity,tuple(transfer),tuple(mean_total_transverse))
        color=emitted_gluon_color_amplitudes(route.emitter)[:,None]
        field=discrete_gluon_field_factor(route.outgoing[3])
        for hf in (0,1):
            for hi in (0,1):
                sr=[];sc=[];sv=[]
                for col,state in enumerate(sf):
                    if state[route.emitter]%2!=hi:continue
                    target=list(state);target[route.emitter]=2*(state[route.emitter]//2)+hf
                    sr.append(lookup[tuple(target)]);sc.append(col)
                    sv.append(float(parton_charge(state[route.emitter],"q",charge)))
                spin=sparse.csr_matrix((sv,(sr,sc)),shape=(64,64))
                block=sparse.kron(orbital[hf,hi],sparse.kron(color,spin),format="coo")
                rows.extend(gi[route.outgoing]*ng*128+block.row)
                cols.extend(qi[route.incoming]*nq*64+block.col)
                values.extend(field*block.data)
    return sparse.coo_matrix((values,(rows,cols)),shape=(len(g)*ng*128,len(q)*nq*64)).tocsr()


def minus_contact_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge,
                           mean_total_transverse=(0.,0.)):
    """Retained one-gluon J-minus per gs; absorption uses reversed transfer.

    No two-gluon, Wick, regulator, loop or excluded-wave contribution is
    implicitly supplied by this standalone operator.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    p=incoming;delta=np.asarray(transfer,float)
    nf=[u.shape[1] for u in outgoing.bases];ni=[u.shape[1] for u in incoming.bases]
    of=np.cumsum([0]+nf);oi=np.cumsum([0]+ni)
    result=np.zeros((sum(nf),sum(ni)),complex)
    for sector,hg in ((1,-1),(2,1)):
        def raw(d):
            return emission_minus_contact(p.total_k2,p.cutoff,p.cutoff,d,b_GeV=p.width_GeV,
                mass_GeV=p.quark_mass_GeV,total_pplus_GeV=total_pplus_GeV,
                gluon_helicity=hg,charge=charge,mean_total_transverse=mean_total_transverse)
        result[of[sector]:of[sector+1],:oi[1]]=(
            outgoing.bases[sector].getH()@raw(delta)@incoming.bases[0]).toarray()
        result[:of[1],oi[sector]:oi[sector+1]]=(
            outgoing.bases[0].getH()@raw(-delta).getH()@incoming.bases[sector]).toarray()
    return result
