"""First variation of the declared finite-cell field-mode envelope.

Explicit covariant spectral representative: E[A+]=chi(D+ - e A+), with
cut edges in the gaps between APBC/PBC eigenvalues. Its first variation
is independent of where inside the same gaps those edges are placed.
This is a FIELD envelope, not the inverse-derivative zero-mode sector,
the Fock-space/isospin projector, or a physical renormalization condition.

In the Fourier basis, dE[B]_mn = (chi(p_m)-chi(p_n))/(p_m-p_n) B_mn.
The divided difference is zero for equal eigenvalues in a spectral gap.
At Delta+=0, B connects only equal p+ and the envelope minus tangent is
therefore exactly zero. Nonzero-transfer edges are generally nonzero.
This zero does NOT apply to the smooth Gaussian or Jacobi projector.
"""
from operator import index
import numpy as np
from .m2_gaussian_field_regulator import GaussianClusterRegulator, _array

SPECTRAL_PRESCRIPTION = "finite-cell-covariant-spectral-gapped-envelope"


def finite_cell_envelope_spectrum(modes, *, regulator, species, total_k2,
        total_pplus_GeV, prescription):
    """Signed Fourier eigenvalues, envelope membership and midpoint cut edges.

    Repeated modes may carry arbitrary transverse/color/spin multiplicity.
    Fermion APBC bands include both quark/antiquark Fourier signs. Boson
    PBC mode zero is not a represented positive field oscillator; excluding
    it here does not solve or discard its separate constraint dynamics.
    """
    if prescription != SPECTRAL_PRESCRIPTION or not isinstance(regulator,GaussianClusterRegulator):
        raise ValueError("explicit common regulator and covariant spectral prescription required")
    if species not in ("q","qbar","g"):
        raise ValueError("explicit physical field species required")
    if isinstance(total_k2,(bool,np.bool_)) or index(total_k2)<=0:
        raise ValueError("positive exact total K2 required")
    pp = float(total_pplus_GeV)
    if not np.isfinite(pp) or pp<=0:
        raise ValueError("positive finite total P+ required")
    labels = tuple(modes)
    if not labels or any(isinstance(m,(bool,np.bool_)) for m in labels):
        raise ValueError("nonempty exact signed doubled Fourier modes required")
    labels = np.array([index(m) for m in labels],dtype=int)
    if np.any(labels%2 != int(species!="g")):
        raise ValueError("APBC fermions require odd and PBC gluons even modes")
    c = pp/index(total_k2)
    if species=="g":
        end = 2*regulator.boson_modes+1
        edges = c*np.array([-end,-1,1,end],float)
        member = (labels!=0)&(abs(labels)<end)
    else:
        end = 2*regulator.fermion_modes
        edges = c*np.array([-end,end],float)
        member = abs(labels)<end
    return c*labels,member,edges


def finite_cell_envelope_minus_tangent(modes, photon_matrix, *, regulator,
        species, total_k2, total_pplus_GeV, charge, prescription):
    """-e d chi(D+)[A+], an envelope tangent, not a complete current.

    photon_matrix supplies the explicit Fourier/transverse matrix of the
    probe. No Delta+=0 restriction is inferred from array shape. The
    function handles repeated eigenvalues without a 0/0 regularization.
    At nonzero transfer, q+ K- = -e [E,A+] element by element.
    """
    momenta,member,_ = finite_cell_envelope_spectrum(modes,regulator=regulator,
        species=species,total_k2=total_k2,total_pplus_GeV=total_pplus_GeV,
        prescription=prescription)
    e = float(charge)
    if not np.isfinite(e) or (species=="g" and e!=0):
        raise ValueError("finite physical charge required; EM/flavor gluons are neutral")
    photon = _array(photon_matrix,(len(momenta),)*2,"explicit Fourier probe",complex_values=True)
    difference = momenta[:,None]-momenta[None,:]
    numerator = member.astype(float)[:,None]-member.astype(float)[None,:]
    divided = np.zeros_like(difference)
    np.divide(numerator,difference,out=divided,where=difference!=0)
    return -e*divided*photon
