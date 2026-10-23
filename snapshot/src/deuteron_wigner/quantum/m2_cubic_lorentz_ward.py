"""Source q->qg Lorentz identity and explicit common-regulator variation.

Continuum physical-leg kernels before cell/Fock/Galerkin projection, per g_s
and color. This is NOT a full interacting spin operator. In particular the
Noether coordinate-density extension, gluonic/quartic/Wick terms, finite
domains and common-order H2 reduction still require their own construction.
No Ward residual is divided by an energy gap to manufacture a generator.
"""
from math import sqrt
from functools import lru_cache

import numpy as np

from deuteron_wigner.bridge.modes.core import (
    GAMMA, GAMMA_PLUS, GAMMA_MINUS, polarization, polarization_cartesian,
)
from .m2_canonical_emission import lf_spinor, _helicity, apply_canonical_emission
from .m2_constrained_fermion_current import (
    GOOD_AXES, bad_field_kernel, ALPHA, discrete_gluon_field_factor,
)
from .m2_coupled_longitudinal_jacobi import jacobi_frame, emission_coordinate_map
from .m2_gaussian_field_vertices import _field_data, _integrate
from .m2_free_spin_generators import free_generator
from .m2_gaussian_field_regulator import GaussianClusterRegulator


def _momentum(p):
    p = np.asarray(p, float)
    if p.shape != (3,) or not np.all(np.isfinite(p)) or p[0] <= 0:
        raise ValueError("finite physical (p+,px,py) with p+>0 required")
    return p


def _axis(axis):
    if isinstance(axis, (bool, np.bool_)) or axis not in (0, 1):
        raise ValueError("transverse axis 0 or 1 required")
    return int(axis)


def _minus(p, mass):
    return (mass*mass+p[1:]@p[1:])/(2*p[0])


def fermion_noether_matrix(axis):
    """gamma0 {gamma+,sigma^{-i}}/4, with sigma=i[gamma,gamma]/2."""
    i = _axis(axis)+1
    sigma = .5j*(GAMMA_MINUS@GAMMA[i]-GAMMA[i]@GAMMA_MINUS)
    return GAMMA[0]@(GAMMA_PLUS@sigma+sigma@GAMMA_PLUS)/4


def free_fermion_noether_spin(outgoing, incoming, mass):
    """Local spin density S^{+,-i}; does not include coordinate moments.

    Two component good-field axes and 2^(-1/4) normalization are source-owned.
    At equal endpoints this recovers the previous free Fi spin matrices.
    """
    pf, pi = _momentum(outgoing), _momentum(incoming)
    uf = (np.eye(4)+bad_field_kernel(pf, mass))@GOOD_AXES/2**.25
    ui = (np.eye(4)+bad_field_kernel(pi, mass))@GOOD_AXES/2**.25
    return np.array([uf.conj().T@fermion_noether_matrix(a)@ui for a in (0, 1)])


def fermion_noether_one_gluon(outgoing, incoming, mass, epsilon, *, right_plus, left_plus):
    """Linear constrained-field spin-density insertion, not full Fi[1].

    Strip g_s, color and the gluon field normalization. epsilon is the same
    physical Fourier leg on both sides (epsilon* for emission). Signed
    constraint denominators are explicit, not replaced by absolute values.
    Coordinate-Hamiltonian-density and gluon-constraint terms are NOT zero.
    """
    pf, pi = _momentum(outgoing), _momentum(incoming)
    eps = np.asarray(epsilon, complex)
    if (eps.shape != (2,) or not np.all(np.isfinite(eps))
            or not np.all(np.isfinite([right_plus, left_plus])) or right_plus == 0 or left_plus == 0):
        raise ValueError("finite transverse field and signed nonzero constraint denominators required")
    vertex = ALPHA[0]*eps[0]+ALPHA[1]*eps[1]
    right, left = vertex/(sqrt(2)*right_plus), vertex/(sqrt(2)*left_plus)
    bf, bi = bad_field_kernel(pf, mass), bad_field_kernel(pi, mass)
    return np.array([GOOD_AXES.conj().T@((np.eye(4)+bf.conj().T)@fermion_noether_matrix(a)@right
                    +left@fermion_noether_matrix(a)@(np.eye(4)+bi))@GOOD_AXES/sqrt(2) for a in (0, 1)])


def free_gluon_noether_spin(outgoing, incoming):
    """Normal-ordered free S^{+,-i}=F^{+i}A^- - F^{+-}A^i.

    Rows/columns have helicities (-,+). The constrained A^- polarization is
    retained. Nonzero interaction terms in A^- are not supplied here.
    """
    pf, pi = _momentum(outgoing), _momentum(incoming)
    out = np.array([polarization(*pf, h) for h in (-1, 1)]).conj()
    inc = np.array([polarization(*pi, h) for h in (-1, 1)])
    factor = 1j*(pf[0]+pi[0])/(2*sqrt(pf[0]*pi[0]))
    return np.array([factor*(out[:, a+2, None]*inc[None, :, 1]-out[:, 1, None]*inc[None, :, a+2]) for a in (0, 1)])


@lru_cache(maxsize=4096)
def retained_fermion_spin_orbital(route, cutoff, b, mass, hq, hi, hg, *,
                                 axis, regulator, total_pplus_GeV):
    """Actual qqq->qqqg Galerkin block of the cubic FERMION spin density.

    Per g_s and color, with all physical field legs in the existing common
    regulator. The instantaneous denominators are the outgoing/incoming
    quark p+. The gluon carries the source finite-cell field factor; NO
    2P+ mass conversion belongs to a Lorentz generator. This labelled piece
    omits coordinate-density, gluon-constraint, regulator-variation and
    boundary contributions. It is NOT a proposed full regulated generator.
    """
    a = _axis(axis)
    hq, hi, hg = map(_helicity, (hq, hi, hg))
    if not isinstance(regulator, GaussianClusterRegulator):
        raise ValueError("explicit existing common-cluster regulator required")
    f, tree = jacobi_frame(route.outgoing), emission_coordinate_map(route)
    fields, shifts, precision, eligible = _field_data(
        f, (route.emitter, 3), route.incoming[route.emitter], "q",
        regulator=regulator, total_pplus_GeV=total_pplus_GeV,
        total_transverse_GeV=(0., 0.), species=("q",)*3+("g",))
    x, y = f.fractions[[route.emitter, 3]]
    pf, pi = np.array([x*total_pplus_GeV, 0., 0.]), np.array([(x+y)*total_pplus_GeV, 0., 0.])
    eps = polarization(y*total_pplus_GeV, 0., 0., hg).conj()[2:]
    spin = fermion_noether_one_gluon(pf, pi, mass, eps, right_plus=pf[0], left_plus=pi[0])
    coefficient = spin[a, (hq+1)//2, (hi+1)//2]*discrete_gluon_field_factor(route.outgoing[3])
    result = sqrt(x*y/(x+y))*_integrate(
        tree[:2], in_cutoff=cutoff, out_cutoff=cutoff, b=b,
        fields=fields, shifts=shifts, precision=precision, eligible=eligible,
        constant=coefficient, reference_shape=regulator.reference_shape)
    result.setflags(write=False)
    return result


def apply_retained_fermion_spin(tensor, total_k2, max_excitation, b_GeV, mass_GeV,
                                h_g, *, axis, regulator, total_pplus_GeV):
    """Coherent three-emitter action of the labelled Noether fermion piece.

    Uses the unchanged source color/CAR, flavor and circular-Jacobi maps.
    Input/output shapes are those of apply_canonical_emission. The adjoint
    supplies absorption. It connects different Jz sectors, not an eigenstate
    spin assignment, and is not added to the Ward representative contact.
    """
    a = _axis(axis)
    def kernel(route, cutoff, b, mass, hq, hi, hg):
        return retained_fermion_spin_orbital(route, cutoff, b, mass, hq, hi, hg,
            axis=a, regulator=regulator, total_pplus_GeV=total_pplus_GeV)
    return apply_canonical_emission(tensor, total_k2, max_excitation, b_GeV,
                                    mass_GeV, h_g, _orbital_kernel=kernel)


def emission_minus_vertex(outgoing_quark, gluon, incoming_quark, mass, helicity):
    """Actual Dirac q->qg source extended to independent physical momenta.

    C=-ubar gamma.eps* u / sqrt(8 p_out+ p_in+ k+), per g_s and color,
    in source (-,+) order. Multiplying by 2P+/sqrt(2L) recovers the discrete
    source mass kernel. The off-conserving extension is explicit and is
    used ONLY for the distributional Ward identity, not a new Hamiltonian.
    """
    pf, k, pi = map(_momentum, (outgoing_quark, gluon, incoming_quark))
    helicity = _helicity(helicity)
    out = np.column_stack([lf_spinor(*pf, mass, h) for h in (-1, 1)])
    inc = np.column_stack([lf_spinor(*pi, mass, h) for h in (-1, 1)])
    eps = polarization_cartesian(polarization(*k, helicity).conj())
    slash = sum(g*e*(1 if n == 0 else -1) for n, (g, e) in enumerate(zip(GAMMA, eps)))
    return -out.conj().T@GAMMA[0]@slash@inc/sqrt(8*pf[0]*pi[0]*k[0])


def emission_lorentz_contact(gluon, helicity, axis):
    """Gauge-restoration contact of the declared Dirac-kernel extension.

    Z_i=-i eps_i*/[k+ sqrt(2k+)] I, finite without dividing by Delta E.
    With -i C d_delta_i this is a first-order distributional q->qg Ward
    representative. It is NOT equal by definition to the local fermion spin
    density alone; Noether-density first moments must be matched separately.
    Absorption is the adjoint, with its reversed delta derivative.
    """
    k = _momentum(gluon)
    helicity = _helicity(helicity)
    eps = polarization(*k, helicity).conj()
    return -1j*eps[_axis(axis)+2]/(k[0]*sqrt(2*k[0]))*np.eye(2)


def cluster_quadratic_lorentz(momenta, masses, name):
    """Q_T of the existing positive-physical-leg kernel and its Lorentz drift.

    The comparison Q_L=Q_T+W sum(m_a²/p_a+)/2=(sum p_a)^2/2 is a Lorentz
    scalar on on-shell legs. It is returned as an analytic diagnostic only;
    no production regulator or normalization is changed. Longitudinal cell
    membership/boundaries are not differentiated by this smooth kernel.
    """
    p = np.asarray(momenta, float)
    m = np.asarray(masses, float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or m.shape != (len(p),)
            or not np.all(np.isfinite(p)) or np.any(p[:, 0] <= 0)
            or not np.all(np.isfinite(m)) or np.any(m < 0)):
        raise ValueError("positive physical momenta and nonnegative masses for all legs required")
    v = np.array([free_generator(q, mass, name).velocity for q, mass in zip(p, m)])
    pp, pt = p[:, 0], p[:, 1:]
    w, total = pp.sum(), pt.sum(axis=0)
    dw, dtotal = v[:, 0].sum(), v[:, 1:].sum(axis=0)
    s = np.sum(pt*pt, axis=1)
    r = float(np.sum(m*m/pp))
    # Weighted variance equals the source leg_transform squared, but avoids
    # subtracting two large nearly equal positive quadratic forms.
    relative = pt-pp[:, None]*total[None, :]/w
    transverse = float(w*np.sum(relative*relative/pp[:, None])/2)
    ds = np.sum(2*pt*v[:, 1:], axis=1)
    derivative = float((dw*np.sum(s/pp)+w*np.sum(ds/pp-s*v[:, 0]/pp**2)-2*total@dtotal)/2)
    correction = w*r/2
    dcorrection = float((dw*r-w*np.sum(m*m*v[:, 0]/pp**2))/2)
    return {"transverse_GeV2": transverse, "transverse_derivative_GeV2": derivative,
            "mass_completion_GeV2": float(correction), "mass_completion_derivative_GeV2": dcorrection,
            "scalar_comparison_GeV2": transverse+correction,
            "scalar_comparison_derivative_GeV2": derivative+dcorrection}


def source_regulator_lorentz(momenta, masses, name, regulator):
    """Smooth kernel and tangent of the ACTUAL Gaussian/reference-shape family.

    Membership is deliberately not inferred from noninteger flowed momenta.
    The corresponding finite-envelope/cell terms must be kept separately.
    """
    if not isinstance(regulator, GaussianClusterRegulator):
        raise ValueError("explicit existing common-cluster regulator required")
    data = cluster_quadratic_lorentz(momenta, masses, name)
    q = data["transverse_GeV2"]
    if regulator.reference_shape is None:
        value = np.exp(-q/(2*regulator.width_GeV**2))
        slope = -value/(2*regulator.width_GeV**2)
    else:
        value, slope = regulator.reference_shape.value_and_derivative(q)
    return float(value), float(slope*data["transverse_derivative_GeV2"])


def emission_regular_ward(outgoing_quark, gluon, incoming_quark, mass, helicity, axis, directional_derivative, *, regulator=None):
    """Residual of the first-order dynamical Ward identity on Delta+,DeltaT=0.

    directional_derivative is the independently evaluated simultaneous
    free-Lorentz derivative of the BARE or regulated C. Return the explicit
    smooth regulator defect -i (D R) C, not an invented contact counterterm.
    The derivative-of-delta term cancels between [F0,V] and [F1,P0^-].
    No equality here establishes [F1,F2] through g² or the many-body algebra.
    """
    pf, k, pi = map(_momentum, (outgoing_quark, gluon, incoming_quark))
    if not np.allclose(pf+k, pi, rtol=0, atol=1e-12*max(1., np.linalg.norm(pi))):
        raise ValueError("physical plus/transverse momentum conservation required")
    derivative = np.asarray(directional_derivative, complex)
    if derivative.shape != (2, 2) or not np.all(np.isfinite(derivative)):
        raise ValueError("finite independently evaluated 2x2 derivative required")
    name = "F"+str(_axis(axis)+1)
    af = free_generator(pf, mass, name)
    ag = free_generator(k, 0., name, particle="gluon")
    ai = free_generator(pi, mass, name)
    # Validate helicity through the physical polarization, without integer coercion.
    contact = emission_lorentz_contact(k, helicity, axis)
    hg = 0 if helicity == -1 else 1
    bare = emission_minus_vertex(pf, k, pi, mass, helicity)
    value, tangent = (1., 0.) if regulator is None else source_regulator_lorentz(
        [pf, k, pi], [mass, 0., mass], name, regulator)
    vertex = value*bare
    gap = _minus(pf, mass)+_minus(k, 0.)-_minus(pi, mass)
    residual = (-1j*(derivative+(af.divergence+ag.divergence+ai.divergence)*vertex/2)
                +af.spin@vertex+ag.spin[hg, hg]*vertex-vertex@ai.spin-gap*value*contact)
    return {"residual": residual, "smooth_regulator_defect": -1j*tangent*bare,
            "gap_GeV": gap, "contact": value*contact}
