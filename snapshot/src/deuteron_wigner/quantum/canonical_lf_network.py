"""Canonical LF operator leaves and complete bare quark--gluon tree network.

Reusable explicit-momentum kernels, NOT the old BLFQ/H2 finite Hamiltonian.
No oscillator basis, fixed K, regulator, pole ansatz, model fit or elimination
of Fock sectors is selected. Tree amplitudes strip g_s^2 and an overall
S-matrix phase; external quark normalization is ubar u=2m. Cubic P-minus
kernels strip the spatial delta/measure, with units GeV^(-1/2), and cannot
be added to a free-energy matrix without a declared wavepacket/cell map.
"""
from math import sqrt
import numpy as np

from .canonical_lf_fields import (
    GAMMA, GAMMA_PLUS, GAMMA_MINUS, lf_spinor, antiquark_spinor,
    polarization, polarization_cartesian, slash, three_gluon_numerator,
)
from .m2_constrained_fermion_current import free_fermion_current
from .modular_gauge_network import generators
from .graded_fock_network import Ladder, Word


def momentum(value):
    p = np.asarray(value, float)
    if p.shape != (3,) or not np.all(np.isfinite(p)) or p[0] <= 0:
        raise ValueError('Finite physical (p+,px,py), p+>0 required.')
    return p


def four_vector(value, mass):
    p = momentum(value)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError('Finite nonnegative mass required.')
    minus = (mass*mass+p[1:]@p[1:])/(2*p[0])
    return np.array([(p[0]+minus)/sqrt(2), p[1], p[2], (p[0]-minus)/sqrt(2)])


def dot(a, b):
    return a[0]*b[0]-a[1:]@b[1:]


def cubic_kernel(channel, left, right, parent, mass, helicities):
    """Bare color/g_s-stripped cubic P-minus coefficient in flat mode measure.

    q_to_qg, qbar_to_qbarg, g_to_pair, g_to_gg. For qbar emission
    the NORMAL-ORDERED color factor is -t^T, supplied separately. For a
    pair the color is t[quark,antiquark]; three gluons use f[parent,left,right].
    Daughter Bose/Fermi factors belong to the operator action, not this leaf.
    """
    a, b, p = map(momentum, (left, right, parent))
    if not np.allclose(a+b, p, rtol=0, atol=1e-12*max(1., np.linalg.norm(p))):
        raise ValueError('Physical plus/transverse conservation required.')
    if not np.isfinite(mass) or mass < 0 or len(helicities) != 3:
        raise ValueError('Mass and three helicities required.')
    ha, hb, hp = helicities
    if channel == 'q_to_qg':
        u, v = lf_spinor(*a, mass, ha), lf_spinor(*p, mass, hp)
        eps = polarization_cartesian(polarization(*b, hb).conj())
        numerator = -u.conj()@GAMMA[0]@slash(eps)@v
    elif channel == 'qbar_to_qbarg':
        u, v = antiquark_spinor(*p, mass, hp), antiquark_spinor(*a, mass, ha)
        eps = polarization_cartesian(polarization(*b, hb).conj())
        numerator = -u.conj()@GAMMA[0]@slash(eps)@v
    elif channel == 'g_to_pair':
        u, v = lf_spinor(*a, mass, ha), antiquark_spinor(*b, mass, hb)
        eps = polarization_cartesian(polarization(*p, hp))
        numerator = -u.conj()@GAMMA[0]@slash(eps)@v
    elif channel == 'g_to_gg':
        numerator = three_gluon_numerator(a, b, p, ha, hb, hp)
    else:
        raise ValueError('Unknown canonical cubic channel.')
    return complex(numerator/sqrt(8*a[0]*b[0]*p[0]))


def cubic_word(channel, coefficient, left_mode, right_mode, parent_mode):
    """Ordered normal monomial for distinguished daughter modes.

    A g->gg enumeration must count an unordered physical daughter pair once
    (or include the standard 1/2 in a fully ordered enumeration). This helper
    does not enumerate modes or silently choose a global Hamiltonian domain.
    """
    if channel not in ('q_to_qg', 'qbar_to_qbarg', 'g_to_pair', 'g_to_gg'):
        raise ValueError('Unknown cubic channel.')
    return Word(coefficient, (Ladder(left_mode, True), Ladder(right_mode, True), Ladder(parent_mode, False)))


def cubic_vertex_tensor(channel, left, right, parent, mass):
    """Resolved (left spin/color, right spin/color, parent spin/color) leaf.

    Axes have size 6 for q/qbar and 16 for gluons; spin/helicity order (-,+).
    Uses the modular network's SU(3) generator order throughout, including
    the conjugate antiquark representation. Coupling and mode measure are
    still stripped. Distinct daughter momenta: no global Bose counting here.
    """
    t = generators()
    if channel == 'q_to_qg':
        color = t.transpose(1, 0, 2)
    elif channel == 'qbar_to_qbarg':
        color = -t.transpose(2, 0, 1)
    elif channel == 'g_to_pair':
        color = t.transpose(1, 2, 0)
    elif channel == 'g_to_gg':
        # [t_a,t_b]=i f_abc t_c and tr(t_a t_b)=delta_ab/2.
        commutator = np.einsum('aij,bjk->abik', t, t)-np.einsum('bij,ajk->abik', t, t)
        color = (-2j*np.einsum('abij,cji->abc', commutator, t)).real
    else:
        raise ValueError('Unknown canonical cubic channel.')
    spin = np.empty((2, 2, 2), complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                spin[i,j,k] = cubic_kernel(channel, left, right, parent, mass, (2*i-1,2*j-1,2*k-1))
    return np.einsum('ijk,abc->iajbkc', spin, color).reshape(tuple(2*n for n in color.shape))


def electromagnetic_current(outgoing, incoming, mass, *, species, quark_charge):
    """Reused full four-component FREE fermion current, not an interacting current.

    Inputs/outputs (+,-,x,y); canonical field factors already included.
    quark_charge is the flavor's quark charge; qbar receives its negative.
    """
    if not np.isfinite(quark_charge):
        raise ValueError('Finite explicit flavor charge required.')
    sign = 1 if species == 'q' else -1
    return sign*quark_charge*free_fermion_current(outgoing, incoming, mass_GeV=mass, species=species)


def fermion_propagator_parts(r, mass):
    """Exact off-shell identity, signed r+ retained; strip the common factor i.

    (slash(r)+m)/(r²-m²)=(slash(r_on)+m)/(r²-m²)+gamma+/(2r+).
    Negative r+ is allowed; no positive-only or absolute-value replacement.
    This identity is not an H2 reduction or a loop renormalization scheme.
    """
    r = np.asarray(r, float)
    if r.shape != (4,) or not np.all(np.isfinite(r)) or not np.isfinite(mass) or mass < 0:
        raise ValueError('Finite Cartesian momentum and nonnegative mass required.')
    plus = (r[0]+r[3])/sqrt(2)
    den = dot(r, r)-mass*mass
    if abs(plus) < 1e-12 or abs(den) < 1e-12:
        raise ValueError('Zero-plus and on-shell internal domains require a separate prescription.')
    instant = GAMMA_PLUS/(2*plus)
    minus_on = (mass*mass+r[1]**2+r[2]**2)/(2*plus)
    on = np.array([(plus+minus_on)/sqrt(2), r[1], r[2], (plus-minus_on)/sqrt(2)])
    return (slash(on)+mass*np.eye(4))/den, instant


def gluon_current(k, outgoing_k, incoming_epsilon, outgoing_epsilon_conjugate):
    """Three-gluon Lorentz tensor with all-outgoing momenta (q,-k,k')."""
    q = k-outgoing_k
    e, f = incoming_epsilon, outgoing_epsilon_conjugate
    return (e*dot(q+k, f)-(k+outgoing_k)*dot(e, f)
            +f*dot(outgoing_k-q, e))


def quark_gluon_tree(outgoing_q, outgoing_g, incoming_q, incoming_g, mass,
                     *, replace_incoming=False, replace_outgoing=False):
    """Full bare tree in (quark spin,color;gluon helicity,color) order.

    Three graph channels use fixed t_b t_a, t_a t_b, [t_b,t_a] couplers.
    Independent covariant and LF propagator-plus-instantaneous contractions.
    Polarization replacement supplies an external gluon Ward test. Physical
    gluon polarization labels are (-,+), not internal gauge-dependent modes.
    Returns 96x96 colored partonic amplitude matrices, not a nucleon H.
    """
    pf, kf, pi, ki = map(momentum, (outgoing_q, outgoing_g, incoming_q, incoming_g))
    p, l, r, k = four_vector(pf, mass), four_vector(kf, 0), four_vector(pi, mass), four_vector(ki, 0)
    if not np.allclose(p+l, r+k, rtol=0, atol=2e-12):
        raise ValueError('On-shell four-momentum conservation required for a scattering amplitude.')
    q = k-l
    transfer = dot(q, q)
    plus_q = (q[0]+q[3])/sqrt(2)
    if abs(transfer) < 1e-12 or abs(plus_q) < 1e-12:
        raise ValueError('Nonzero t and exchanged plus required in this LF control.')
    sout = np.column_stack([lf_spinor(*pf, mass, h) for h in (-1, 1)])
    sin = np.column_stack([lf_spinor(*pi, mass, h) for h in (-1, 1)])
    bar = sout.conj().T@GAMMA[0]
    quark_current = np.array([bar@g@sin for g in GAMMA])
    sp, si = fermion_propagator_parts(r+k, mass)
    up, ui = fermion_propagator_parts(r-l, mass)
    # Independently evaluated covariant denominators, not the sum of the LF pieces.
    sc = (slash(r+k)+mass*np.eye(4))/(dot(r+k, r+k)-mass*mass)
    uc = (slash(r-l)+mass*np.eye(4))/(dot(r-l, r-l)-mass*mass)
    names = ('s_covariant', 'u_covariant', 't_covariant', 's_propagating',
             's_instantaneous', 'u_propagating', 'u_instantaneous',
             't_propagating', 't_instantaneous')
    spin = {name: np.zeros((2, 2, 2, 2), complex) for name in names}
    residual = 0.
    for go, ho in enumerate((-1, 1)):
        f = l.astype(complex) if replace_outgoing else polarization_cartesian(polarization(*kf, ho).conj())
        for gi, hi in enumerate((-1, 1)):
            e = k.astype(complex) if replace_incoming else polarization_cartesian(polarization(*ki, hi))
            for name, left, propagator, right in (
                ('s_covariant', f, sc, e), ('u_covariant', e, uc, f),
                ('s_propagating', f, sp, e), ('s_instantaneous', f, si, e),
                ('u_propagating', e, up, f), ('u_instantaneous', e, ui, f)):
                spin[name][:, go, :, gi] = bar@slash(left)@propagator@slash(right)@sin
            c = gluon_current(k, l, e, f)
            residual = max(residual, abs(dot(q, c)))
            contraction = quark_current[0]*c[0]-sum(quark_current[j]*c[j] for j in (1, 2, 3))
            spin['t_covariant'][:, go, :, gi] = contraction/transfer
            mode = np.sign(plus_q)*np.array([plus_q, q[1], q[2]])
            propagating = np.zeros((2, 2), complex)
            for h in (-1, 1):
                eps = polarization_cartesian(polarization(*mode, h))
                first = quark_current[0]*eps[0]-sum(quark_current[j]*eps[j] for j in (1, 2, 3))
                propagating -= first*dot(eps.conj(), c)/transfer
            plus_current = (quark_current[0]+quark_current[3])/sqrt(2)
            plus_c = (c[0]+c[3])/sqrt(2)
            spin['t_propagating'][:, go, :, gi] = propagating
            spin['t_instantaneous'][:, go, :, gi] = -plus_current*plus_c/plus_q**2
    t = generators()
    colors_s = np.einsum('bix,axj->ibja', t, t)
    colors_u = np.einsum('aix,bxj->ibja', t, t)
    colors = {'s': colors_s, 'u': colors_u, 't': colors_s-colors_u}
    matrices = {name: np.einsum('ohpi,cbda->ochbpdia', value,
                                colors[name[0]]).reshape(96, 96) for name, value in spin.items()}
    covariant = sum(matrices[n] for n in names[:3])
    propagating = sum(matrices[n] for n in ('s_propagating','u_propagating','t_propagating'))
    instantaneous = sum(matrices[n] for n in ('s_instantaneous','u_instantaneous','t_instantaneous'))
    return dict(matrices, covariant=covariant, propagating=propagating,
                instantaneous=instantaneous, light_front=propagating+instantaneous,
                three_gluon_divergence=float(residual),
                exchange_GeV2=float(transfer), crossed_fermion_plus=float(((r-l)[0]+(r-l)[3])/sqrt(2)))
