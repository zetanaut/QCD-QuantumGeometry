"""Cross-component spin-half current diagnostic, never a covariance repair."""
from dataclasses import dataclass

import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA
from .m2_canonical_emission import lf_spinor
from .m2_constrained_propagators import slash
from .m2_spin_resolved_current import good_form_factor_diagnostic


def transverse_dirac_pauli_structures(transfer,*,mass_GeV,total_pplus_GeV):
    """Independent external on-shell tensors, helicity (+1/2,-1/2).

    At Delta+=0 and P_mean,T=0 the two transverse tensors coincide by the
    Gordon identity. Hence J_T tests G_M=F1+F2, not two independent fits.
    """
    delta=np.asarray(transfer,float);m=float(mass_GeV);pp=float(total_pplus_GeV)
    if (delta.shape!=(2,) or not np.all(np.isfinite(np.r_[delta,m,pp]))
            or min(m,pp)<=0 or np.linalg.norm(delta)==0):
        raise ValueError("nonzero finite transfer and positive mass/P+ required")
    qs=slash([0.,*delta,0.])
    bra=[lf_spinor(pp,*(delta/2),m,h) for h in (1,-1)]
    ket=[lf_spinor(pp,*(-delta/2),m,h) for h in (1,-1)]
    structures=[]
    for j in (1,2):
        gamma=GAMMA[j]
        pauli=-(gamma@qs-qs@gamma)/(4*m)
        structures.append(np.array([[[u.conj()@GAMMA[0]@op@v/(2*pp)
                                      for v in ket] for u in bra] for op in (gamma,pauli)]))
    return np.array(structures).transpose(1,0,2,3)


@dataclass(frozen=True)
class CurrentCovarianceDiagnostic:
    F1_from_plus: complex
    F2_from_plus: complex
    magnetic_from_plus: complex
    magnetic_from_transverse: complex
    plus_structure_residual: np.ndarray
    transverse_covariance_residual: np.ndarray
    transverse_nonmagnetic_residual: np.ndarray
    elastic_continuity_defect: np.ndarray


def current_covariance_diagnostic(plus,transverse,transfer,*,mass_GeV,total_pplus_GeV):
    """Compare supplied, identically phased current components; do not alter them."""
    t=np.asarray(transverse,complex)
    if t.shape!=(2,2,2) or not np.all(np.isfinite(t)):
        raise ValueError("finite two-component 2x2 helicity transverse current required")
    ff=good_form_factor_diagnostic(plus,transfer,mass_GeV=mass_GeV,total_pplus=total_pplus_GeV)
    d,p=transverse_dirac_pauli_structures(transfer,mass_GeV=mass_GeV,total_pplus_GeV=total_pplus_GeV)
    gm=complex(np.vdot(d,t)/np.vdot(d,d))
    predicted=ff.dirac*d+ff.pauli*p
    return CurrentCovarianceDiagnostic(ff.dirac,ff.pauli,ff.dirac+ff.pauli,gm,
        ff.residual_matrix,t-predicted,t-gm*d,np.einsum("j,jab->ab",transfer,t))
