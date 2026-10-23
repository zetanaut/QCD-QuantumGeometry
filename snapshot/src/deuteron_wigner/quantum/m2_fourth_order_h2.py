"""One order-consistent bare H2 model through g^4 for the declared finite sectors.

Combines five-body V1/V2 returns, one canonical QQ insertion and the
complete leading six-body excursion. PP is the caller's retained pilot.
No higher-order subset (g^5/g^6) is mixed into this fourth-order model.
This does not establish common-regulator renormalization, full QCD,
a physical four-current or an uncertainty-controlled prediction.
"""
from dataclasses import dataclass

import numpy as np

from .m2_canonical_good_current import good_current_density_between
from .m2_shell_current import shell_density_between


def _norm_inverse_sqrt(metric):
    """Stable symmetric whitening for a norm close to identity.

    Reconstruct the small correction, not I through nearly degenerate
    eigenvectors. The rationalized scalar expression also avoids subtracting
    1 from (1+delta)^(-1/2). No eigenvalue clipping or modified norm is used.
    """
    identity=np.eye(len(metric))
    delta,vectors=np.linalg.eigh(metric-identity)
    if np.min(1+delta)<=0:
        raise ArithmeticError("fourth-order norm metric is not positive")
    roots=np.sqrt(1+delta)
    correction=-delta/(roots*(1+roots))
    return identity+(vectors*correction)@vectors.conj().T


@dataclass(frozen=True)
class FourthOrderSnapshot:
    owner: object
    energy_GeV2: float
    effective_hamiltonian: np.ndarray
    metric: np.ndarray
    matrix: np.ndarray
    whitener: np.ndarray
    five_wave_coefficients: tuple
    six_wave_coefficient: dict

    def current(self,transfer,*,charge):
        return current_between(self,self,transfer,charge=charge,normalized=True)


class FourthOrderH2:
    physical=False
    common_regulator_renormalization_complete=False

    def __init__(self,reference,*,QQ_insertion,six_body):
        if QQ_insertion.reference is not reference or six_body.reference is not reference:
            raise ValueError("every insertion must bind the exact same source reference")
        self.reference=reference;self.qq=QQ_insertion;self.six=six_body
        self.pilot=reference.pilot
        self.spaces5=six_body.spaces5;self.spaces6=six_body.spaces6
        for name,space in self.spaces5.items():
            if QQ_insertion.action.spaces[name] is not space:
                raise ValueError("five-body actions need the identical coordinate object")

    def snapshot(self,energy):
        """Linearize H_eff about E with N=I-d_E H_eff; preserve raw waves once."""
        e=float(energy);g=self.pilot.coupling
        a1,qq_tail=self.qq.order_waves(e,1)
        a2=self.qq.leading_waves(e,2)
        six=self.six.waves(e)
        for n in a1:
            if np.linalg.norm(a1[n]-six["five_leading"][n])>3e-12:
                raise ArithmeticError("shared leading five-body source changed")
        c3={n:qq_tail[n]+six["five_backflow"][n] for n in a1}
        a6=six["six_leading"]
        h=self.pilot.matrix.copy()
        for power in (2,3,4):
            h+=g**power*self.reference.return_coefficient(power,e)
        h4=sum(self.qq.sources_by_power[1][n].conj().T@qq_tail[n] for n in a1)
        h4+=sum(six["six_source"][n].conj().T@a6[n] for n in a6)
        h+=g**4*h4
        metric=np.eye(len(h),dtype=complex)
        for n in a1:
            metric+=g*g*(a1[n].conj().T@a1[n])
            c=a1[n].conj().T@a2[n]
            metric+=g**3*(c+c.conj().T)
            c=a1[n].conj().T@c3[n]
            metric+=g**4*(a2[n].conj().T@a2[n]+c+c.conj().T)
        metric+=g**4*sum(v.conj().T@v for v in a6.values())
        for label,m in (("Hamiltonian",h),("norm",metric)):
            if not np.all(np.isfinite(m)) or np.linalg.norm(m-m.conj().T)>3e-11:
                raise ArithmeticError("fourth-order "+label+" failed Hermiticity")
        whiten=_norm_inverse_sqrt(metric)
        # H_eff(E)=H_eff(e)+(I-N)(E-e); generalized A v=E N v.
        linearized=whiten@(h+e*(metric-np.eye(len(h))))@whiten
        return FourthOrderSnapshot(self,e,h,metric,linearized,whiten,(a1,a2,c3),a6)


def current_between(outgoing,incoming,transfer,*,charge,normalized):
    """Consistent local good current through g^4, including different Jz fibers.

    The wave orders are g A1 + g^2 A2 + g^3 C3 in Q5 and g^2 A6 in Q6.
    Their products are truncated by order, not formed into an all-order norm.
    """
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit normalization choice required")
    left,right=outgoing.owner,incoming.owner
    pf,pi=left.pilot,right.pilot
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(pf,key)!=getattr(pi,key):
            raise ValueError("fourth-order current endpoints disagree on "+key)
    if left.spaces5.keys()!=right.spaces5.keys() or left.spaces6.keys()!=right.spaces6.keys():
        raise ValueError("same declared Fock sectors required")
    result=good_current_density_between(pf,pi,transfer,charge=charge)
    g=pi.coupling
    # Exact same snapshot stores both endpoint tails; no repeated source solve.
    af=outgoing.five_wave_coefficients;ai=incoming.five_wave_coefficients
    for n,s in right.spaces5.items():
        nP=len(pi.matrix)
        joined=shell_density_between(left.spaces5[n],s,
            np.column_stack(tuple(a[n] for a in ai)),transfer,b_GeV=pi.width_GeV,charge=charge)
        r1,r2,r3=(joined[:,j*nP:(j+1)*nP] for j in range(3))
        result+=g*g*(af[0][n].conj().T@r1)
        result+=g**3*(af[0][n].conj().T@r2+af[1][n].conj().T@r1)
        result+=g**4*(af[1][n].conj().T@r2+af[0][n].conj().T@r3+af[2][n].conj().T@r1)
    for n,s in right.spaces6.items():
        r=shell_density_between(left.spaces6[n],s,incoming.six_wave_coefficient[n],
                                 transfer,b_GeV=pi.width_GeV,charge=charge)
        result+=g**4*outgoing.six_wave_coefficient[n].conj().T@r
    return outgoing.whitener@result@incoming.whitener if normalized else result
