"""Canonical transverse current through g^4 on the common interacting H2.

Includes PP, P-Q5, Q5-Q5, Q5-Q6 and Q6-Q6, with all raw QQ/backflow
endpoints. Minus/Wick/current-loop and gauged-regulator completion remain
open. These are finite-model canonical coefficients, not physical form factors.
"""
import numpy as np

from .m2_fourth_order_h2 import FourthOrderSnapshot
from .m2_transverse_current_projection import free_transverse_current_between
from .m2_transverse_contact_current import transverse_contact_between
from .m2_shell_transverse_current import shell_transverse_current_between
from .m2_excluded_transverse_contact import build_excluded_transverse_contact
from .m2_six_body_transverse_contact import SixBodyTransverseContact


def transverse_coefficients_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge):
    """Unwhitened two-energy coefficients J[0],...,J[4], each (x/y,Pf,Pi).

    The same snapshots contain A1,A2,C3 in Q5 and A6 in Q6. Products above
    order four are excluded explicitly, not included by an all-order norm.
    """
    if not isinstance(outgoing,FourthOrderSnapshot) or not isinstance(incoming,FourthOrderSnapshot):
        raise ValueError("explicit fourth-order H2 snapshots required")
    left,right=outgoing.owner,incoming.owner;pf,pi=left.pilot,right.pilot
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(pf,key)!=getattr(pi,key):raise ValueError("current endpoints disagree on "+key)
    if left.spaces5.keys()!=right.spaces5.keys() or left.spaces6.keys()!=right.spaces6.keys():
        raise ValueError("same declared Fock sectors required")
    delta=tuple(float(x) for x in transfer);P=float(total_pplus_GeV)
    if len(delta)!=2 or not np.all(np.isfinite([*delta,P])) or P<=0:
        raise ValueError("finite transfer and positive total P+ required")
    nP=len(pi.matrix);shape=(2,len(pf.matrix),nP)
    result={n:np.zeros(shape,complex) for n in range(5)}
    result[0]=free_transverse_current_between(pf,pi,delta,total_pplus_GeV=P,charge=charge)
    result[1]=transverse_contact_between(pf,pi,delta,total_pplus_GeV=P,charge=charge)
    af,ai=outgoing.five_wave_coefficients,incoming.five_wave_coefficients
    reverse=tuple(-x for x in delta)
    for name,s in right.spaces5.items():
        forward=build_excluded_transverse_contact(pi,left.spaces5[name],delta,
                                                   total_pplus_GeV=P,charge=charge)
        backward=build_excluded_transverse_contact(pf,s,reverse,total_pplus_GeV=P,charge=charge)
        for j in (0,1):
            joined=shell_transverse_current_between(left.spaces5[name],s,
                np.column_stack(tuple(a[name] for a in ai)),delta,b_GeV=pi.width_GeV,
                total_pplus_GeV=P,charge=charge,component=j)
            r1,r2,r3=(joined[:,k*nP:(k+1)*nP] for k in range(3))
            result[2][j]+=af[0][name].conj().T@r1
            result[3][j]+=af[0][name].conj().T@r2+af[1][name].conj().T@r1
            result[4][j]+=(af[1][name].conj().T@r2
                +af[0][name].conj().T@r3+af[2][name].conj().T@r1)
            for k in range(3):
                result[k+2][j][:,forward.retained_columns]+=af[k][name].conj().T@forward.current_per_gs[j]
                result[k+2][j][backward.retained_columns,:]+=backward.current_per_gs[j].conj().T@ai[k][name]
    forward6=SixBodyTransverseContact(right.spaces5,left.spaces6,delta,b_GeV=pi.width_GeV,
                                      total_pplus_GeV=P,charge=charge)
    backward6=SixBodyTransverseContact(left.spaces5,right.spaces6,reverse,b_GeV=pi.width_GeV,
                                       total_pplus_GeV=P,charge=charge)
    for j in (0,1):
        forward=forward6.apply(ai[0],component=j)
        backward=backward6.apply(af[0],component=j)
        for name,s in right.spaces6.items():
            a6f=outgoing.six_wave_coefficient[name];a6i=incoming.six_wave_coefficient[name]
            value=shell_transverse_current_between(left.spaces6[name],s,a6i,delta,
                b_GeV=pi.width_GeV,total_pplus_GeV=P,charge=charge,component=j)
            result[4][j]+=(a6f.conj().T@value+a6f.conj().T@forward[name]
                           +backward[name].conj().T@a6i)
    return result


def transverse_current_between(outgoing,incoming,transfer,*,total_pplus_GeV,charge,normalized):
    """Use BOTH snapshots' existing whiteners, never a one-endpoint shortcut."""
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit normalization choice required")
    coefficients=transverse_coefficients_between(outgoing,incoming,transfer,
        total_pplus_GeV=total_pplus_GeV,charge=charge)
    g=incoming.owner.pilot.coupling
    value=sum(g**n*a for n,a in coefficients.items())
    return np.array([outgoing.whitener@a@incoming.whitener for a in value]) if normalized else value
