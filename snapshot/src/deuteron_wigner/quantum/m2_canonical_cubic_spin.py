"""Canonical Legendre density and Noether spin for the bare q->qg source.

Resolve coordinate-density improvements BEFORE projecting. The canonical
local Hamiltonian and the convenient Dirac vertex agree on spatial momentum
conservation, but their first moments differ. The combined local spin and
coordinate-density contact agrees with the independently derived Ward
representative. This is not the full regulated/cell/many-body spin action.
"""
from math import sqrt

import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_MINUS, polarization
from .m2_constrained_fermion_current import GOOD_AXES, ALPHA, bad_field_kernel
from .m2_cubic_lorentz_ward import (
    _momentum, _helicity, _axis, emission_minus_vertex,
    fermion_noether_one_gluon,
)


def _free_field(p, mass):
    return (np.eye(4)+bad_field_kernel(p, mass))@GOOD_AXES/2**.25


def _spatial_dirac_h(outgoing, incoming, mass):
    """Local Legendre H quadratic form, before integrating by parts.

    Hermitized L_q=i/2 bar(psi) gamma.D<-> psi - m bar(psi)psi,
    D=partial-igA. Remove the gamma+ time derivative, not a spatial
    divergence. These plus/transverse arguments can be signed field modes;
    their minus energies are not used in this spatial quadratic form.
    """
    pf, pi = np.asarray(outgoing), np.asarray(incoming)
    return GAMMA[0]@(-GAMMA_MINUS*(pf[0]+pi[0])/2
        +sum(GAMMA[i+1]*(pf[i+1]+pi[i+1])/2 for i in (0, 1))+mass*np.eye(4))


def _data(outgoing_quark, gluon, incoming_quark, mass, helicity):
    pf, k, pi = map(_momentum, (outgoing_quark, gluon, incoming_quark))
    h = _helicity(helicity)
    mass = float(mass)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError("finite nonnegative quark mass required")
    r, l, q = pi-k, pf+k, pi-pf
    if r[0] == 0 or l[0] == 0 or q[0] == 0:
        raise ValueError("zero modes of the fermion/Gauss constraints are not supplied")
    eps = polarization(*k, h).conj()
    vertex = sum(a*e for a, e in zip(ALPHA, eps[2:]))
    # The same emitted Fourier leg appears in both insertions. l/r are
    # constraint momenta, not inferred from off-conserving external labels.
    right = vertex@GOOD_AXES/(sqrt(2)*r[0]*2**.25)
    left = GOOD_AXES.conj().T@vertex/(sqrt(2)*l[0]*2**.25)
    return pf, k, pi, r, l, q, eps, right, left, mass


def canonical_emission_density(outgoing_quark, gluon, incoming_quark, mass, helicity):
    """C_can=C_Dirac+D_fermion+D_gluon, per g/color and sqrt(2k+) field.

    The source D=partial-igA gives (partial+)² A^-_[q]=g J+,
    hence A^-_[q]/g= -J+/(q+)², q=pi-pf. The local gauge Legendre
    density is -(partial+ A^-)²/2 - partial+ A^i partial_i A^-
    + F_ij²/4. Its q->qg cross term is retained even though its integral
    vanishes on momentum conservation. No time derivative or Fock equation
    is replaced with a fitted pole. All three returned differences are
    BEFORE the finite Gaussian, cell and oscillator projection.
    """
    pf, k, pi, r, l, q, eps, right, left, mass = _data(
        outgoing_quark, gluon, incoming_quark, mass, helicity)
    uf, ui = _free_field(pf, mass), _free_field(pi, mass)
    fermion = (uf.conj().T@_spatial_dirac_h(pf, r, mass)@right
                +left@_spatial_dirac_h(l, pi, mass)@ui)/sqrt(2*k[0])
    gluon = -k[0]/q[0]**2*(eps[2:]@q[1:]-q[0]*eps[1])*np.eye(2)/sqrt(2*k[0])
    dirac = emission_minus_vertex(pf, k, pi, mass, helicity)
    return {"canonical": dirac+fermion+gluon, "Dirac": dirac,
            "fermion_density_difference": fermion, "gluon_density_difference": gluon}


def quark_gluon_noether_spin(outgoing_quark, gluon, incoming_quark, mass, helicity):
    """Separate local fermion/gluon S^{+,-i}, including Gauss constraint.

    Normalized per g/color with the physical gluon field 1/sqrt(2k+).
    Neither is the complete F_i, nor are they additional pieces to be
    added on top of the Ward representative's already combined contact.
    """
    pf, k, pi, r, l, q, eps, _, _, mass = _data(
        outgoing_quark, gluon, incoming_quark, mass, helicity)
    fermion = fermion_noether_one_gluon(pf, pi, mass, eps[2:],
                                        right_plus=r[0], left_plus=l[0])/sqrt(2*k[0])
    gluon = np.array([-1j*(k[0]+q[0])*eps[i+2]/q[0]**2*np.eye(2)/sqrt(2*k[0]) for i in (0, 1)])
    return {"fermion": fermion, "gluon": gluon}


def canonical_density_contact(outgoing_quark, gluon, incoming_quark, mass, helicity, axis):
    """Reduce -i C_can partial_delta_i(delta)+S_can delta independently.

    Return analytic derivatives of D=C_can-C_Dirac with respect to
    Delta=pf+k-pi at fixed pf,k, on Delta=0, and their combined Noether
    contact S_can+i partial_Delta D. No free Lorentz Ward matrices or
    energy-gap inversion are used to compute it. Spatial T^{+i} remains
    kinematical in this A+=0 canonical density; its x^- moment adds no
    cubic q->qg contact. This comparison precedes cell projection; it makes
    no assertion that the eventual finite-cell surface terms vanish.
    """
    a = _axis(axis)
    pf, k, pi, r, l, q, eps, right, left, mass = _data(
        outgoing_quark, gluon, incoming_quark, mass, helicity)
    if not np.allclose(pf+k, pi, rtol=0, atol=1e-12*max(1., np.linalg.norm(pi))):
        raise ValueError("spatial conservation required for the declared delta-contact reduction")
    uf, ui = _free_field(pf, mass), _free_field(pi, mass)
    dkernel = -GAMMA[0]@GAMMA[a+1]/2
    dfield = -ALPHA[a]@GOOD_AXES/(sqrt(2)*pi[0]*2**.25)
    df = (uf.conj().T@dkernel@right+left@dkernel@ui
          +left@_spatial_dirac_h(l, pi, mass)@dfield)/sqrt(2*k[0])
    dg = eps[a+2]/k[0]*np.eye(2)/sqrt(2*k[0])
    spin = quark_gluon_noether_spin(pf, k, pi, mass, helicity)
    sf, sg = spin["fermion"][a], spin["gluon"][a]
    return {"fermion_density_derivative": df, "gluon_density_derivative": dg,
            "fermion_spin": sf, "gluon_spin": sg,
            "fermion_improved_contact": sf+1j*df,
            "gluon_improved_contact": sg+1j*dg,
            "combined_contact": sf+sg+1j*(df+dg)}
