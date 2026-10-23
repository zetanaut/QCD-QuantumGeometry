"""Normal-ordered canonical J-minus through the common H2 order g^4.

Every free, one-gluon and quadratic-current block contributing through
this order on the declared P/Q5/Q6 wave map is explicit. This is NOT all
order-g^4 QCD, a complete physical current or common-regulator matching:
Wick/loop/counterterm, zero-mode and gauged-projector partners remain separate.
"""
import numpy as np
from .m2_fourth_order_h2 import FourthOrderSnapshot
from .m2_free_minus_current import free_minus_current_between
from .m2_one_gluon_minus_contact import minus_contact_between
from .m2_shell_minus_current import shell_minus_current_between
from .m2_excluded_minus_contact import build_excluded_minus_contact,SixBodyMinusContact
from .m2_two_gluon_minus_current import retained_two_gluon_minus_between
from .m2_quadratic_minus_actions import (
    build_pair_minus_endpoint,build_six_minus_endpoint,FiveBodyQuadraticMinus,
)


def minus_coefficients_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge,
                                mean_total_transverse=(0.,0.)):
    """Raw Jminus[0..4], before BOTH snapshots' declared norm maps."""
    if not isinstance(outgoing,FourthOrderSnapshot) or not isinstance(incoming,FourthOrderSnapshot):
        raise ValueError("explicit common-order H2 snapshots required")
    left,right=outgoing.owner,incoming.owner;pf,pi=left.pilot,right.pilot
    for k in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(pf,k)!=getattr(pi,k):raise ValueError("current endpoints disagree on "+k)
    if left.spaces5.keys()!=right.spaces5.keys() or left.spaces6.keys()!=right.spaces6.keys():
        raise ValueError("same declared Fock sectors required")
    delta=tuple(float(x) for x in transfer);mean=tuple(float(x) for x in mean_total_transverse)
    P=float(total_pplus_GeV)
    if len(delta)!=2 or len(mean)!=2 or not np.all(np.isfinite(np.r_[delta,mean,P])) or P<=0:
        raise ValueError("finite kinematics and positive P+ required")
    nP=len(pi.matrix);shape=(len(pf.matrix),nP);result={n:np.zeros(shape,complex) for n in range(5)}
    common=dict(total_pplus_GeV=P,charge=charge,mean_total_transverse=mean)
    result[0]=free_minus_current_between(pf,pi,delta,**common)
    result[1]=minus_contact_between(pf,pi,delta,**common)
    result[2]=retained_two_gluon_minus_between(pf,pi,delta,total_pplus_GeV=P,charge=charge)
    af,ai=outgoing.five_wave_coefficients,incoming.five_wave_coefficients
    reverse=tuple(-x for x in delta)
    for name,s in right.spaces5.items():
        f=build_excluded_minus_contact(pi,left.spaces5[name],delta,**common)
        b=build_excluded_minus_contact(pf,s,reverse,**common)
        joined=shell_minus_current_between(left.spaces5[name],s,
            np.column_stack(tuple(a[name] for a in ai)),delta,b_GeV=pi.width_GeV,
            mass_GeV=pi.quark_mass_GeV,**common)
        r1,r2,r3=(joined[:,k*nP:(k+1)*nP] for k in range(3))
        result[2]+=af[0][name].conj().T@r1
        result[3]+=af[0][name].conj().T@r2+af[1][name].conj().T@r1
        result[4]+=(af[1][name].conj().T@r2+af[0][name].conj().T@r3+af[2][name].conj().T@r1)
        for k in range(3):
            result[k+2][:,f.retained_columns]+=af[k][name].conj().T@f.current_per_gs
            result[k+2][b.retained_columns,:]+=b.current_per_gs.conj().T@ai[k][name]
    f=build_pair_minus_endpoint(pi,left.spaces5["qqqgg"],delta,total_pplus_GeV=P,charge=charge)
    b=build_pair_minus_endpoint(pf,right.spaces5["qqqgg"],reverse,total_pplus_GeV=P,charge=charge)
    for k in (0,1):
        result[k+3][:,f.retained_columns]+=af[k]["qqqgg"].conj().T@f.current_per_gs2
        result[k+3][b.retained_columns,:]+=b.current_per_gs2.conj().T@ai[k]["qqqgg"]
    qq=FiveBodyQuadraticMinus(left.spaces5,right.spaces5,delta,b_GeV=pi.width_GeV,
                              total_pplus_GeV=P,charge=charge).apply(ai[0])
    result[4]+=sum(af[0][s].conj().T@qq[s] for s in qq)
    f6=SixBodyMinusContact(right.spaces5,left.spaces6,delta,b_GeV=pi.width_GeV,
                           mass_GeV=pi.quark_mass_GeV,**common).apply(ai[0])
    b6=SixBodyMinusContact(left.spaces5,right.spaces6,reverse,b_GeV=pi.width_GeV,
                           mass_GeV=pi.quark_mass_GeV,**common).apply(af[0])
    for name,s in right.spaces6.items():
        a6f=outgoing.six_wave_coefficient[name];a6i=incoming.six_wave_coefficient[name]
        free=shell_minus_current_between(left.spaces6[name],s,a6i,delta,b_GeV=pi.width_GeV,
                                         mass_GeV=pi.quark_mass_GeV,**common)
        result[4]+=a6f.conj().T@free+a6f.conj().T@f6[name]+b6[name].conj().T@a6i
        component="valence_IF" if name=="qqqggg" else "gluon_IF_pair"
        f=build_six_minus_endpoint(pi,left.spaces6[name],delta,component=component,total_pplus_GeV=P,charge=charge)
        b=build_six_minus_endpoint(pf,s,reverse,component=component,total_pplus_GeV=P,charge=charge)
        result[4][:,f.retained_columns]+=a6f.conj().T@f.current_per_gs2
        result[4][b.retained_columns,:]+=b.current_per_gs2.conj().T@a6i
    return result


def minus_current_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge,normalized,
                           mean_total_transverse=(0.,0.)):
    """Apply the same finite-model norm at both ends; no physical activation."""
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit normalization choice required")
    coefficients=minus_coefficients_between(outgoing,incoming,transfer,total_pplus_GeV=total_pplus_GeV,
                                            charge=charge,mean_total_transverse=mean_total_transverse)
    g=incoming.owner.pilot.coupling;value=sum(g**n*a for n,a in coefficients.items())
    return outgoing.whitener@value@incoming.whitener if normalized else value
