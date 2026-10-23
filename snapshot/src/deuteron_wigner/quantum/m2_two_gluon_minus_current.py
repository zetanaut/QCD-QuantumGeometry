"""Normal-ordered retained qg two-gluon J-minus contact, per gs^2.

The two constrained fermion denominators are both retained. At Delta+=0
they coincide, including the negative crossed mode. This is not a
propagating loop, nor all quadratic-current number-changing/Wick partners.
"""
from functools import lru_cache
from math import pi,sqrt
import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.modes.core import polarization
from .m2_canonical_good_current import parton_charge
from .m2_constrained_fermion_current import two_gluon_current_spin,discrete_gluon_field_factor
from .m2_canonical_instantaneous_fermion import ordered_qg_color
from .m2_coupled_longitudinal_jacobi import jacobi_frame,oscillator_shell,longitudinal_basis
from .m2_coupled_pair_action import _pair,_pair_rotation,pair_contact_block
from .m2_coupled_pilot import _displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


@lru_cache(maxsize=4096)
def displaced_pair_contact(incoming,outgoing,pair,b_GeV,in_cutoff,out_cutoff,transfer):
    """Local pair coincidence with full photon recoil on the shared vertex.

    In the pair-adapted frame the first relative coordinate is integrated
    on BOTH sides; all other coordinates overlap with the cluster shift.
    Every relative m=0 radial mode contributes b/sqrt(pi). This is not
    multiplication of separately truncated density/contact matrices.
    """
    # Reuse the exact route/fraction validation, not its zero-transfer answer.
    pair_contact_block(incoming,outgoing,pair,b_GeV,in_cutoff,out_cutoff)
    delta=np.asarray(transfer,float);b=float(b_GeV)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError("finite transverse transfer required")
    p,tree=_pair(pair,len(incoming));fi=jacobi_frame(incoming,tree);fo=jacobi_frame(outgoing,tree)
    d=len(incoming)-1;i,j=p
    shift=fi.forward[1:-1,i,None]*delta
    a=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[a,a.conj()]
    inc=oscillator_shell(d,in_cutoff);out=oscillator_shell(d,out_cutoff)
    adapted=np.zeros((len(out),len(inc)),complex)
    for row,o in enumerate(out):
        if o[0]!=o[d]:continue
        po=o[1:d]+o[d+1:]
        for col,k in enumerate(inc):
            if k[0]!=k[d]:continue
            pi_=k[1:d]+k[d+1:]
            adapted[row,col]=np.prod([_displacement(a,c,z) for a,c,z in zip(po,pi_,alpha)])
    ai=fi.fractions[i]*fi.fractions[j]/(fi.fractions[i]+fi.fractions[j])
    af=fo.fractions[i]*fo.fractions[j]/(fo.fractions[i]+fo.fractions[j])
    result=b*b/pi*sqrt(ai*af)*_pair_rotation(outgoing,p,out_cutoff).T@adapted@_pair_rotation(incoming,p,in_cutoff)
    result.setflags(write=False)
    return result


def qg_two_gluon_minus(total_k2,in_cutoff,out_cutoff,transfer,*,b_GeV,
                       total_pplus_GeV,gluon_helicity,charge):
    """Raw qqqg→qqqg ordered direct+crossed J-minus/(2P+), per gs^2.

    Per ordering Jminus = f_g' f_g chi† alpha.eps_L alpha.eps_R chi /(2r+^2).
    r_direct=p_q+k_g, r_cross=p_q-k_g'. Both signed denominators
    occur BEFORE multiplication; no absolute-value prescription is used.
    Only this normal-ordered qg channel is constructed.
    """
    pp=float(total_pplus_GeV);delta=tuple(float(x) for x in transfer)
    if len(delta)!=2 or not np.all(np.isfinite(np.r_[delta,pp])) or pp<=0 or gluon_helicity not in (-1,1):
        raise ValueError("finite transfer, positive P+ and physical gluon helicity required")
    basis=longitudinal_basis("qqqg",total_k2);lookup={s:i for i,s in enumerate(basis)}
    ni,no=len(oscillator_shell(3,in_cutoff)),len(oscillator_shell(3,out_cutoff))
    sf=np.array(SPIN_FLAVOR_BASIS);rows=[];cols=[];values=[]
    eps=polarization(pp,0,0,gluon_helicity)[2:]
    for column,state in enumerate(basis):
        for quark in range(3):
            charge_weights=parton_charge(sf[:,quark],"q",charge)
            for kg in range(2,state[quark]+state[3],2):
                out=list(state);out[3]=kg;out[quark]=state[quark]+state[3]-kg;out=tuple(out)
                orbital=displaced_pair_contact(state,out,(quark,3),b_GeV,in_cutoff,out_cutoff,delta)
                field=discrete_gluon_field_factor(kg)*discrete_gluon_field_factor(state[3])
                spin_color=np.zeros((128,128),complex)
                for crossed,r2 in ((False,state[quark]+state[3]),(True,state[quark]-kg)):
                    if r2%2!=1:raise ArithmeticError("constrained fermion mode must remain odd")
                    left,right=(eps,eps.conj()) if crossed else (eps.conj(),eps)
                    spin=two_gluon_current_spin(left_polarization=left,right_polarization=right,
                        left_plus=pp*r2/total_k2,right_plus=pp*r2/total_k2)[1]
                    if np.any(spin-np.diag(np.diag(spin))):
                        raise ArithmeticError("unexpected quadratic quark helicity flip")
                    diagonal=charge_weights*spin[sf[:,quark]%2,sf[:,quark]%2]
                    spin_color+=np.kron(ordered_qg_color(quark,crossed),np.diag(diagonal))
                block=(field*sparse.kron(orbital,spin_color,format="coo")).tocoo()
                rows.extend(lookup[out]*no*128+block.row)
                cols.extend(column*ni*128+block.col);values.extend(block.data)
    return sparse.coo_matrix((values,(rows,cols)),shape=(len(basis)*no*128,len(basis)*ni*128)).tocsr()


def retained_two_gluon_minus_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge):
    """Retained qg block; unchanged gluon helicity is an exact spin selection.

    The free and one-gluon current, Wick contraction and higher-sector
    quadratic contacts are separate; no complete physical current is implied.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    p=incoming;nf=[u.shape[1] for u in outgoing.bases];ni=[u.shape[1] for u in incoming.bases]
    blocks=[np.zeros((nf[0],ni[0]),complex)]
    for k,hg in ((1,-1),(2,1)):
        raw=qg_two_gluon_minus(p.total_k2,p.cutoff,p.cutoff,transfer,b_GeV=p.width_GeV,
                               total_pplus_GeV=total_pplus_GeV,gluon_helicity=hg,charge=charge)
        blocks.append((outgoing.bases[k].getH()@raw@incoming.bases[k]).toarray())
    return sparse.block_diag(blocks).toarray()
