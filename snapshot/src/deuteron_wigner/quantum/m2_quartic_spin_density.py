"""Bare quartic Legendre/Noether density before Wick and finite projection.

The constrained fermion field is linear in g A_T and A^- is at most
quadratic in good fields (linear in g). Thus the only candidate local
quartic fermion spin is B1^dagger H_i B1, which vanishes by the bad-projector
identity; gluon S^{+,-i} has no quartic term. Canonical T^{+i} is kinematical.
The NONZERO quartic generator is the coordinate moment -x_i H4. Its bare
normal delta_i contact is zero, not the generator or the quartic current.

These functions derive the density before spatial partial integration and
compare it with the source IF/IG/A4 form. Longitudinal density differences
are explicit: they must not be dropped on a finite domain by assumption.
No higher-order Lorentz commutator, regulator partner or H2 spin is supplied.
"""
from math import sqrt
import numpy as np

from deuteron_wigner.bridge.modes.core import LAMBDA_MINUS, GAMMA
from .m2_constrained_fermion_current import (
    GOOD_AXES, ANTI_AXES, ALPHA, bad_field_kernel, _momentum, _mass,
)
from .m2_canonical_cubic_spin import _spatial_dirac_h
from .m2_cubic_lorentz_ward import fermion_noether_matrix
from .m2_excluded_fock_vertices import _slash_epsilon


def _axes(kind):
    if kind not in ("q", "qbar"):
        raise ValueError("explicit q/qbar signed-field spin axes required")
    return GOOD_AXES if kind == "q" else ANTI_AXES


def _transverse(e):
    value = np.asarray(e, complex)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("finite transverse Fourier polarization required")
    return value


def bad_projected_noether(axis):
    """Exact 4x4 algebraic B1^dagger H_i B1 witness, not an assumed zero."""
    return LAMBDA_MINUS@fermion_noether_matrix(axis)@LAMBDA_MINUS


def instantaneous_fermion_density(outgoing_field, incoming_field, left_gluon,
        right_gluon, mass, left_polarization, right_polarization, *, out_axes, in_axes):
    """One ordered IF coefficient per g²/color, with physical field norms.

    All four momenta are SIGNED Fourier labels: pf+kL+kR-pi=Delta.
    Creation gluons have positive k; annihilation gluons negative k with
    their own unconjugated polarization. Negative fermionic field labels
    use qbar axes. No CAR sign, physical-axis transpose or Bose assignment
    is guessed. r=pi-kR and l=pf+kL retain signed nonzero plus components.

    C_IF uses the source right inverse 1/r+. The local hermitized Legendre
    density instead contains (1/r+ + 1/l+)/2; their difference is exactly
    -Delta+ times the supplied coefficient. Both are transverse-constant
    even off conservation, so their normal transverse improvement is zero.
    """
    pf, pi, kl, kr = map(_momentum, (outgoing_field, incoming_field, left_gluon, right_gluon))
    mass = _mass(mass)
    oa, ia = _axes(out_axes), _axes(in_axes)
    el, er = map(_transverse, (left_polarization, right_polarization))
    vl, vr = [sum(a*e for a, e in zip(ALPHA, eps)) for eps in (el, er)]
    l, r = pf+kl, pi-kr
    if l[0] == 0 or r[0] == 0:
        raise ValueError("zero constrained fermion modes are not supplied")
    uf = (np.eye(4)+bad_field_kernel(pf, mass))@oa/2**.25
    ui = (np.eye(4)+bad_field_kernel(pi, mass))@ia/2**.25
    left = oa.conj().T@vl/(sqrt(2)*l[0]*2**.25)
    right = vr@ia/(sqrt(2)*r[0]*2**.25)
    eps = [np.array([0., k[1:]@e/k[0], *e]) for k, e in ((kl, el), (kr, er))]
    norm = sqrt(4*abs(kl[0]*kr[0]))
    kinetic = left@_spatial_dirac_h(l, r, mass)@right/norm
    interaction = (-uf.conj().T@GAMMA[0]@_slash_epsilon(eps[0])@right
                   -left@GAMMA[0]@_slash_epsilon(eps[1])@ui)/norm
    spin = np.array([left@fermion_noether_matrix(a)@right/norm for a in (0, 1)])
    ordered = oa.conj().T@vl@vr@ia
    source = ordered/(2*r[0]*norm)
    delta = pf+kl+kr-pi
    difference = -delta[0]*ordered/(4*l[0]*r[0]*norm)
    return {"canonical": kinetic+interaction, "source": source,
            "quadratic_field_kinetic": kinetic, "explicit_interaction": interaction,
            "density_difference": difference, "local_spin": spin,
            "delta_plus": float(delta[0]),
            "transverse_contact": np.zeros_like(spin)}


def instantaneous_gluon_density(left_current_plus, right_current_plus, current_product):
    """Ordered Gauss density per g², keeping its longitudinal improvement.

    Currents have phases exp(i Q+ x^- - i Q_T.x_T); A1=-J/(Q+)².
    Include current spin/color and physical field normalizations in the
    explicit ordered product J_a J_b. It is not symmetrized or Wick-traced
    here. The local terms are -(partial+ A1)²/2 - J A1. Compare with the
    source +J_a J_b/[2(Q_b+)²]. Both current orderings still belong in H.
    Products of even good-field currents can differ in lower Wick degree;
    no claim about their quantum reordering is inferred from this top-degree
    local identity. It applies to qq, qg and gg current products alike.
    """
    qa, qb = float(left_current_plus), float(right_current_plus)
    value = np.asarray(current_product, complex)
    if not np.all(np.isfinite([qa, qb])) or qa == 0 or qb == 0:
        raise ValueError("finite signed nonzero Gauss transfer modes required")
    if not np.all(np.isfinite(value)):
        raise ValueError("finite explicitly ordered current product required")
    kinetic = value/(2*qa*qb)
    interaction = value/qb**2
    source = value/(2*qb**2)
    return {"canonical": kinetic+interaction, "source": source,
            "quadratic_constraint_kinetic": kinetic, "current_constraint_interaction": interaction,
            "density_difference": (qa+qb)*value/(2*qa*qb**2),
            "delta_plus": qa+qb,
            "local_spin": np.zeros((2,)+value.shape, complex),
            "transverse_contact": np.zeros((2,)+value.shape, complex)}


def four_gluon_density(physical_plus, polarizations, ordered_color):
    """One ordered A4 coefficient; sum all 24 field assignments separately.

    The caller supplies the color tensor f^(xab) f^(xcd), in this order,
    and already-conjugated creation polarizations. The source 1/4 and all
    four physical 1/sqrt(2p+) factors occur once; Fock factors are not here.
    There are no longitudinal constraint inverses or derivatives in A4.
    """
    pp = np.asarray(physical_plus, float)
    eps = np.asarray([_transverse(e) for e in polarizations])
    color = np.asarray(ordered_color, complex)
    if pp.shape != (4,) or eps.shape != (4, 2) or not np.all(np.isfinite(pp)) or np.any(pp <= 0):
        raise ValueError("four finite positive physical gluon plus momenta required")
    if not np.all(np.isfinite(color)):
        raise ValueError("finite explicitly ordered four-gluon color required")
    c = (eps[0]@eps[2])*(eps[1]@eps[3])*color/(4*sqrt(np.prod(2*pp)))
    return {"canonical": c, "source": c,
            "local_spin": np.zeros((2,)+c.shape, complex),
            "transverse_contact": np.zeros((2,)+c.shape, complex)}
