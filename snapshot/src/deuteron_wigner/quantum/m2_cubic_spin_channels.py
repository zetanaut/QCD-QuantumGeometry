"""Bare canonical cubic spin in all source Fock channels, before projection.

Signed constraint Fourier modes are not negative-energy particles. External
momenta are positive physical (p+,px,py) in daughter,daughter,parent order.
Strip g_s and the source color matrix: T for q/pair, Tbar=-T^T for qbar,
f^{a_parent a_left a_right} for gg. No additional CAR/Bose factors belong
to these labelled kernels. Multiplying C by 2P+/sqrt(2L) gives the source
mass kernel; spin/contact kernels receive no 2P+ conversion.

Canonical local Legendre densities and Noether spin are constructed before
discarding a spatial divergence. Their delta-contact reduction is compared
with separately defined gauge-restoration contacts, never inferred by
dividing a Ward residual by a free energy gap. This is only order g on
continuum physical legs: no quartic, loop, domain, Galerkin or H2 closure.
"""
from itertools import permutations
from math import sqrt

import numpy as np

from deuteron_wigner.bridge.modes.core import (
    GAMMA, polarization, polarization_cartesian, lf_dot,
)
from .m2_constrained_fermion_current import GOOD_AXES, ANTI_AXES, ALPHA, bad_field_kernel
from .m2_canonical_cubic_spin import _spatial_dirac_h
from .m2_cubic_lorentz_ward import (
    _momentum, _axis, _helicity, _minus, fermion_noether_matrix,
    source_regulator_lorentz,
)
from .m2_free_spin_generators import free_generator
from .m2_excluded_fock_vertices import three_gluon_numerator


FERMION_CHANNELS = ("q_to_qg", "anti_to_antig", "g_to_pair")


def _conserved(p):
    if not np.allclose(p[0]+p[1], p[2], rtol=0,
                       atol=1e-12*max(1., np.linalg.norm(p[2]))):
        raise ValueError("physical plus/transverse conservation required")


def _fermion_data(channel, momenta, mass, helicity):
    if channel not in FERMION_CHANNELS:
        raise ValueError("explicit source fermionic cubic channel required")
    p = np.array([_momentum(q) for q in momenta])
    if p.shape != (3, 3):
        raise ValueError("two daughters and one parent required")
    mass = float(mass)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError("finite nonnegative quark mass required")
    h = _helicity(helicity)
    if channel == "q_to_qg":
        pf, k, pi = p
        out_axes, in_axes = GOOD_AXES, GOOD_AXES
    elif channel == "anti_to_antig":
        pf, k, pi = -p[2], p[1], -p[0]
        out_axes, in_axes = ANTI_AXES, ANTI_AXES
    else:
        pf, k, pi = p[0], -p[2], -p[1]
        out_axes, in_axes = GOOD_AXES, ANTI_AXES
    eps = (polarization(*p[2], h) if channel == "g_to_pair"
           else polarization(*p[1], h).conj())
    r, l, q = pi-k, pf+k, pi-pf
    if r[0] == 0 or l[0] == 0 or q[0] == 0:
        raise ValueError("zero modes of the fermion/Gauss constraints are not supplied")
    uf = (np.eye(4)+bad_field_kernel(pf, mass))@out_axes/2**.25
    ui = (np.eye(4)+bad_field_kernel(pi, mass))@in_axes/2**.25
    vertex = sum(a*e for a, e in zip(ALPHA, eps[2:]))
    right = vertex@in_axes/(sqrt(2)*r[0]*2**.25)
    left = out_axes.conj().T@vertex/(sqrt(2)*l[0]*2**.25)
    return p, pf, k, pi, r, l, q, eps, out_axes, in_axes, uf, ui, right, left, mass


def _physical_order(channel, matrix):
    # Normal ordering's minus is already in Tbar=-T^T. This transpose maps
    # signed field (incoming anti bra,outgoing anti ket) to physical states.
    return matrix.swapaxes(-1, -2) if channel == "anti_to_antig" else matrix


def fermionic_cubic_density(channel, momenta, mass, helicity):
    """Local canonical H and S^{+,-i} from signed constrained Dirac fields.

    Return C_can=C_Dirac+D_f+D_g and the TWO separate local spin pieces.
    These off-conservation density differences vanish on spatial delta=0;
    their derivatives do not. Pair output axes are (q helicity,qbar helicity).
    """
    (p, pf, k, pi, r, l, q, eps, oa, ia, uf, ui, right, left, mass) = _fermion_data(
        channel, momenta, mass, helicity)
    norm = sqrt(2*abs(k[0]))
    cart = polarization_cartesian(eps)
    slash = sum(g*e*(1 if n == 0 else -1) for n, (g, e) in enumerate(zip(GAMMA, cart)))
    dirac = -uf.conj().T@GAMMA[0]@slash@ui/norm
    df = (uf.conj().T@_spatial_dirac_h(pf, r, mass)@right
          +left@_spatial_dirac_h(l, pi, mass)@ui)/norm
    jp = oa.conj().T@ia
    dg = -k[0]*(eps[2:]@q[1:]-q[0]*eps[1])*jp/(q[0]**2*norm)
    sf = np.array([(uf.conj().T@fermion_noether_matrix(a)@right
                    +left@fermion_noether_matrix(a)@ui)/norm for a in (0, 1)])
    sg = np.array([-1j*(k[0]+q[0])*eps[a+2]*jp/(q[0]**2*norm) for a in (0, 1)])
    return {key: _physical_order(channel, value) for key, value in {
        "canonical": dirac+df+dg, "Dirac": dirac,
        "fermion_density_difference": df, "gluon_density_difference": dg,
        "fermion_spin": sf, "gluon_spin": sg}.items()}


def fermionic_cubic_contact(channel, momenta, mass, helicity, axis):
    """Analytic density first moment, not the Lorentz/Ward residual.

    Delta=pf+k-pi is the physical daughters-parent mismatch for each signed
    channel. Differentiate at fixed signed pf,k (pi -> pi-Delta). This may
    move a daughter in the crossed channel; normal derivatives of D|Delta=0
    are independent of the chosen tangential partition of that displacement.
    """
    a = _axis(axis)
    (p, pf, k, pi, r, l, q, eps, oa, ia, uf, ui, right, left, mass) = _fermion_data(
        channel, momenta, mass, helicity)
    _conserved(p)
    norm = sqrt(2*abs(k[0]))
    dkernel = -GAMMA[0]@GAMMA[a+1]/2
    dfield = -ALPHA[a]@ia/(sqrt(2)*pi[0]*2**.25)
    df = (uf.conj().T@dkernel@right+left@dkernel@ui
          +left@_spatial_dirac_h(l, pi, mass)@dfield)/norm
    dg = eps[a+2]*(oa.conj().T@ia)/(k[0]*norm)
    df, dg = (_physical_order(channel, v) for v in (df, dg))
    spin = fermionic_cubic_density(channel, p, mass, helicity)
    sf, sg = spin["fermion_spin"][a], spin["gluon_spin"][a]
    return {"fermion_density_derivative": df, "gluon_density_derivative": dg,
            "fermion_spin": sf, "gluon_spin": sg,
            "fermion_improved_contact": sf+1j*df,
            "gluon_improved_contact": sg+1j*dg,
            "combined_contact": sf+sg+1j*(df+dg)}


def fermionic_gauge_contact(channel, momenta, helicity, axis):
    """Independent Dirac gauge-restoration Z, regular even at zero gap.

    Mass does not enter this contact. The pair plus-current is G^dagger A,
    not I; the incoming gauge polarization and its signed Fourier k+ reverse
    the emission sign. No local Noether or Legendre function is called here.
    """
    p = np.array([_momentum(q) for q in momenta])
    if p.shape != (3, 3) or channel not in FERMION_CHANNELS:
        raise ValueError("explicit channel and three physical momenta required")
    a, h = _axis(axis), _helicity(helicity)
    if channel == "g_to_pair":
        k = p[2]
        return 1j*polarization(*k, h)[a+2]*(GOOD_AXES.conj().T@ANTI_AXES)/(k[0]*sqrt(2*k[0]))
    k = p[1]
    return -1j*polarization(*k, h).conj()[a+2]*np.eye(2)/(k[0]*sqrt(2*k[0]))


def _gluon_data(momenta, helicities):
    p = np.array([_momentum(q) for q in momenta])
    h = tuple(_helicity(v) for v in helicities)
    if p.shape != (3, 3) or len(h) != 3:
        raise ValueError("two gluon daughters and one parent with three helicities required")
    k = np.array([-p[2], p[0], p[1]])
    e = np.array([polarization(*p[2], h[2]), polarization(*p[0], h[0]).conj(),
                  polarization(*p[1], h[1]).conj()])
    return p, h, k, e, sqrt(8*np.prod(p[:, 0]))


def gluonic_cubic_density(momenta, helicities):
    """Local YM H3, quadratic-H constraint cross term, and gluon spin.

    Cyclic color order is (parent,left,right). The Gauss term includes both
    ordered gluons in f A_T partial+ A_T. Canonical local H3 is evaluated
    directly from -g f partial+Ai_a A^-_b Ai_c + g f Fij0_a Ai_b Aj_c/2,
    NOT from the three-gluon source numerator. That numerator is the
    independent comparison used by cubic_regular_ward and the tests.
    """
    p, h, k, e, norm = _gluon_data(momenta, helicities)
    local = 0j
    for a, b, c in permutations(range(3)):
        order = (a, b, c)
        sign = (-1)**sum(x > y for j, x in enumerate(order) for y in order[j+1:])
        local += sign*(-1j*k[a, 0]*e[b, 1]*(e[a, 2:]@e[c, 2:])
            +.5j*((k[a, 1:]@e[b, 2:])*(e[a, 2:]@e[c, 2:])
                  -(e[a, 2:]@e[b, 2:])*(k[a, 1:]@e[c, 2:])))
    difference, spin = 0j, np.zeros(2, complex)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        q = k[b]+k[c]
        if q[0] == 0:
            raise ValueError("zero mode of the gluonic Gauss constraint is not supplied")
        a1 = -1j*(k[c, 0]-k[b, 0])*(e[b, 2:]@e[c, 2:])/q[0]**2
        difference += k[a, 0]*(q[0]*e[a, 1]-e[a, 2:]@q[1:])*a1
        spin += 1j*(k[a, 0]-q[0])*e[a, 2:]*a1
    return {"canonical": (local+difference)/norm, "local_YM": local/norm,
            "gluon_density_difference": difference/norm, "gluon_spin": spin/norm}


def gluonic_cubic_contact(momenta, helicities, axis):
    """Noether spin plus analytic normal derivative of the YM density."""
    a = _axis(axis)
    p, h, k, e, norm = _gluon_data(momenta, helicities)
    _conserved(p)
    derivative = 0j
    for j, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        q = k[b]+k[c]
        a1 = -1j*(k[c, 0]-k[b, 0])*(e[b, 2:]@e[c, 2:])/q[0]**2
        derivative -= k[j, 0]*e[j, a+2]*a1/norm
    spin = gluonic_cubic_density(p, h)["gluon_spin"][a]
    return {"gluon_density_derivative": derivative, "gluon_spin": spin,
            "combined_contact": spin+1j*derivative}


def gluonic_gauge_contact(momenta, helicities, axis):
    """Independent contracted three-gluon gauge-restoration identity.

    On spatial conservation, replacing e_a by k_a in the covariant numerator
    F gives gap*(k_b+-k_c+) (e_b.e_c). This polynomial identity fixes Z
    without any division by gap, including massless collinear configurations.
    """
    _, _, k, e, norm = _gluon_data(momenta, helicities)
    a = _axis(axis)
    return sum(e[j, a+2]/k[j, 0]*(k[b, 0]-k[c, 0])*lf_dot(e[b], e[c])
               for j, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))/norm


def cubic_regular_ward(channel, momenta, mass, helicities, axis,
                       directional_derivative, *, regulator=None):
    """Order-g regular Ward residual; derivative must be supplied independently.

    Fermionic helicities is the single physical gluon helicity; for g_to_gg
    it is the three physical helicities (left,right,parent), with mass=0.
    With the whole existing smooth regulator, the predicted uncancelled
    defect is -i(DR)C. This function supplies neither its repair nor the
    derivative-of-delta, finite-cell and Fock/projection commutator terms.
    """
    p = np.array([_momentum(q) for q in momenta])
    if p.shape != (3, 3):
        raise ValueError("three physical momenta required")
    _conserved(p)
    name = "F"+str(_axis(axis)+1)
    if channel == "g_to_gg":
        if mass != 0:
            raise ValueError("physical gluons must be massless")
        _, h, _, _, norm = _gluon_data(p, helicities)
        masses, species = (0., 0., 0.), ("gluon",)*3
        bare = three_gluon_numerator(*p, *h)/norm
        contact = gluonic_gauge_contact(p, h, axis)
        shape = ()
    else:
        bare = fermionic_cubic_density(channel, p, mass, helicities)["Dirac"]
        contact = fermionic_gauge_contact(channel, p, helicities, axis)
        masses = (mass, mass, 0.) if channel == "g_to_pair" else (mass, 0., mass)
        species = (("quark", "antiquark", "gluon") if channel == "g_to_pair" else
                   (("antiquark", "gluon", "antiquark") if channel == "anti_to_antig"
                    else ("quark", "gluon", "quark")))
        h, shape = _helicity(helicities), (2, 2)
    derivative = np.asarray(directional_derivative, complex)
    if derivative.shape != shape or not np.all(np.isfinite(derivative)):
        raise ValueError("finite independently evaluated derivative of the channel kernel required")
    gs = [free_generator(q, m, name, particle=s) for q, m, s in zip(p, masses, species)]
    r, dr = (1., 0.) if regulator is None else source_regulator_lorentz(p, masses, name, regulator)
    c = r*bare
    if channel == "g_to_gg":
        spin_action = sum((1 if j < 2 else -1)*g.spin[(hh+1)//2, (hh+1)//2]*c
                          for j, (g, hh) in enumerate(zip(gs, h)))
    elif channel == "g_to_pair":
        spin_action = gs[0].spin@c+c@gs[1].spin.T-gs[2].spin[(h+1)//2, (h+1)//2]*c
    else:
        spin_action = gs[0].spin@c+gs[1].spin[(h+1)//2, (h+1)//2]*c-c@gs[2].spin
    gap = _minus(p[0], masses[0])+_minus(p[1], masses[1])-_minus(p[2], masses[2])
    residual = -1j*(derivative+sum(g.divergence for g in gs)*c/2)+spin_action-gap*r*contact
    return {"residual": residual, "smooth_regulator_defect": -1j*dr*bare,
            "gap_GeV": gap, "contact": r*contact}
