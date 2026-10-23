"""Orientation-dependent spin-half current decomposition, not a physical fix.

Carbonell et al., Phys. Rept. 300 (1998), Eqs. (6.25),(6.33), arXiv:nucl-th/9804029.
The B1 subtraction convention is their orthogonality convention, not a
renormalization prescription. Full current/state rotational covariance is
not established by representing a finite set of matrix elements in this basis.
"""
from dataclasses import dataclass

import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS
from .m2_canonical_emission import lf_spinor
from .m2_constrained_propagators import slash
from .m2_current_covariance_diagnostic import current_covariance_diagnostic


def orientation_current_structures(transfer, *, mass_GeV, total_pplus_GeV):
    """Five external tensors (F1,F2,B1,B2,B3), components (+,-,x,y).

    On-shell elastic Delta+=0, mean P_T=0, omega=(1,0,0,-1) in Cartesian
    coordinates. Every tensor is ubar Gamma^mu u/(2P+), helicity order (+,-).
    This is kinematics only; it does NOT supply a microscopic minus current.
    """
    delta=np.asarray(transfer,float);m=float(mass_GeV);pp=float(total_pplus_GeV)
    if (delta.shape!=(2,) or not np.all(np.isfinite(np.r_[delta,m,pp]))
            or min(m,pp)<=0 or np.linalg.norm(delta)==0):
        raise ValueError("nonzero finite transfer and positive mass/P+ required")
    eta=float(delta@delta/(4*m*m))
    qs=slash([0.,*delta,0.])
    omega_slash=GAMMA[0]+GAMMA[3]
    omega_dot_p=np.sqrt(2.)*pp
    omega=np.array([0.,np.sqrt(2.),0.,0.])
    momentum_sum=np.array([2*pp,(m*m+delta@delta/4)/pp,0.,0.])
    gammas=(GAMMA_PLUS,(GAMMA[0]-GAMMA[3])/np.sqrt(2.),GAMMA[1],GAMMA[2])
    identity=np.eye(4)
    b1=omega_slash/omega_dot_p-identity/((1+eta)*m)
    bra=[lf_spinor(pp,*(delta/2),m,h) for h in (1,-1)]
    ket=[lf_spinor(pp,*(-delta/2),m,h) for h in (1,-1)]
    result=np.empty((5,4,2,2),complex)
    for mu,gamma in enumerate(gammas):
        operators=(gamma,-(gamma@qs-qs@gamma)/(4*m),b1*momentum_sum[mu],
            m*omega[mu]/omega_dot_p*identity,
            m*m*omega[mu]/omega_dot_p**2*omega_slash)
        for k,op in enumerate(operators):
            result[k,mu]=[[u.conj()@GAMMA[0]@op@v/(2*pp) for v in ket] for u in bra]
    return result


@dataclass(frozen=True)
class OrientationCurrentDiagnostic:
    F1_separated: complex
    F2_separated: complex
    B1: complex
    B2: complex | None
    B3: complex | None
    electric_from_plus: complex
    electric_separated: complex
    component_indices: tuple
    structure_residual: np.ndarray
    unresolved_coefficients: tuple
    full_five_structure_tested: bool
    covariance_repaired: bool = False
    physical: bool = False


def orientation_current_diagnostic(plus,transverse,transfer,*,mass_GeV,total_pplus_GeV,minus=None):
    """Retain B1 and all residuals; absent J- leaves B2/B3 unknown, NOT zero.

    Plus/T uniquely resolve F1,F2,B1 in this stated convention at nonzero
    transfer. An optional caller-supplied minus component completes the
    five-structure test. No input operator or state is changed.
    """
    basis=orientation_current_structures(transfer,mass_GeV=mass_GeV,total_pplus_GeV=total_pplus_GeV)
    cross=current_covariance_diagnostic(plus,transverse,transfer,
        mass_GeV=mass_GeV,total_pplus_GeV=total_pplus_GeV)
    delta=np.asarray(transfer,float);eta=float(delta@delta/(4*mass_GeV**2))
    b1=(cross.magnetic_from_plus-cross.magnetic_from_transverse)/2
    f1=cross.F1_from_plus-2*eta*b1/(1+eta)
    f2=cross.F2_from_plus-2*b1/(1+eta)
    if minus is None:
        indices=(0,2,3);coefficients=(f1,f2,b1);b2=b3=None
        value=np.concatenate((np.asarray(plus,complex)[None],np.asarray(transverse,complex)))
    else:
        jm=np.asarray(minus,complex)
        if jm.shape!=(2,2) or not np.all(np.isfinite(jm)):
            raise ValueError("finite 2x2 helicity minus current required")
        indices=(0,1,2,3)
        value=np.concatenate((np.asarray(plus,complex)[None],jm[None],np.asarray(transverse,complex)))
        axes=np.column_stack([b.ravel() for b in basis])
        coefficients,_,rank,_=np.linalg.lstsq(axes,value.ravel(),rcond=None)
        if rank!=5:
            raise ArithmeticError("five current structures numerically unresolved at these kinematics")
        f1,f2,b1,b2,b3=map(complex,coefficients)
    predicted=sum(c*basis[k,list(indices)] for k,c in enumerate(coefficients))
    return OrientationCurrentDiagnostic(complex(f1),complex(f2),complex(b1),b2,b3,
        cross.F1_from_plus-eta*cross.F2_from_plus,complex(f1-eta*f2),indices,
        value-predicted,("B2","B3") if minus is None else (),minus is not None)
