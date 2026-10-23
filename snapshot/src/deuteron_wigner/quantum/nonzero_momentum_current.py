"""Resolved tree qq + external photon control, not a nucleon/QCD model.

Local scalar/isovector-pseudoscalar Yukawa exchange, canonical external LF
spinors, off-shell Dirac propagators and the same-action collective current.
Metric +---; p+/-=(p0+/-p3)/sqrt(2). Overall S-matrix i and electric e are
stripped. External states have ubar u=2m. No loops, finite Fock projection,
spacetime continuation of the homogeneous quantum cell, or fitted masses.
"""
from dataclasses import dataclass
import numpy as np

from .canonical_lf_fields import (GAMMA, GAMMA_PLUS, slash, lf_spinor, antiquark_spinor,
                                  polarization, polarization_cartesian)
from .collective_quark_matching import bilinear_matrices
from .transverse_source_matching import dot, transverse_vertex
from .modular_gauge_network import generators


def lf_coordinates(p):
    p = np.asarray(p, float)
    if p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError('Finite Cartesian four-momentum required.')
    return np.array([(p[0]+p[3])/np.sqrt(2), p[1], p[2]])


def fermion_propagators(p, mass):
    """i-stripped covariant, independent LF spin-sum, instantaneous parts.

    Negative k+ uses -sum v(-k_on) vbar(-k_on), not a discarded pair route.
    Values are away from poles and k+=0; no zero-mode prescription inferred.
    Each matrix has units GeV^-1. The instantaneous matrix is gamma+/(2k+).
    """
    pp, x, y = lf_coordinates(p)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError('Finite nonnegative common quark mass required.')
    denominator = dot(p, p)-mass*mass
    if abs(pp) < 1e-10 or abs(denominator) < 1e-10:
        raise ValueError('Control excludes internal poles and k+=0.')
    sign = np.sign(pp)
    spinor = lf_spinor if sign > 0 else antiquark_spinor
    states = np.column_stack([spinor(sign*pp, sign*x, sign*y, mass, h) for h in [-1, 1]])
    propagating = sign*states@states.conj().T@GAMMA[0]/denominator
    instantaneous = GAMMA_PLUS/(2*pp)
    return dict(covariant=(slash(p)+mass*np.eye(4))/denominator,
                propagating=propagating, instantaneous=instantaneous,
                light_front=propagating+instantaneous)


@dataclass(frozen=True)
class CollectiveChannel:
    """Declared tree inverse response H-t, endpoint g, transverse C.

    H GeV^2, g dimensionless, C GeV^-2. Finite positive masses allow a
    pole-free spacelike test. This is not an arbitrary quantum spectral fit.
    """
    hessian: np.ndarray
    couplings: np.ndarray
    c: np.ndarray

    def __post_init__(self):
        h, g, c = (np.array(v, float, copy=True) for v in (self.hessian, self.couplings, self.c))
        if g.ndim != 1 or not len(g) or h.shape != (len(g), len(g)) or c.shape != h.shape:
            raise ValueError('Consistent nonempty mode matrices/endpoints required.')
        if not all(np.isfinite(x).all() for x in (h, g, c)):
            raise ValueError('Finite channel data required.')
        if not np.allclose(h, h.T, atol=1e-13, rtol=0) or not np.allclose(c, c.T, atol=1e-13, rtol=0):
            raise ValueError('Real symmetric H and C required.')
        if np.linalg.eigvalsh(h).min() <= 0:
            raise ValueError('Positive tree masses required in this control.')
        for name, value in [('hessian', h), ('couplings', g), ('c', c)]:
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def response(self, p, *, representation='covariant'):
        t = dot(p, p)
        if not np.isfinite(t) or t >= -1e-12:
            raise ValueError('Nonzero spacelike exchange required.')
        if representation == 'covariant':
            return np.linalg.inv(self.hessian-t*np.eye(len(self.couplings)))
        if representation != 'light_front':
            raise ValueError('Unknown collective representation.')
        pp,x,y = lf_coordinates(p)
        if abs(pp) < 1e-10:
            raise ValueError('LF collective energy denominators exclude k+=0.')
        pm = (p[0]-p[3])/np.sqrt(2)
        masses,modes = np.linalg.eigh(self.hessian)
        on_minus = (masses+x*x+y*y)/(2*pp)
        # Real response K=H-k^2 has the opposite sign to 1/(k^2-M^2).
        return (modes*(-1/(2*pp*(pm-on_minus))))@modes.T

    def weight(self, p, *, representation='covariant'):
        return float(self.couplings@self.response(p,representation=representation)@self.couplings)

    def current(self, outgoing, incoming, *, explicit_modes=False, representation='covariant'):
        """g^T R(out) Gamma^mu(out,in) R(in) g, GeV^-3.

        q_mu current^mu = W(out)-W(in). Explicit summation is the
        uneliminated mode-index control, not an independent physics ansatz.
        """
        left, right = (self.response(p,representation=representation) for p in (outgoing,incoming))
        gamma = np.einsum('m,ij->mij', np.asarray(outgoing)+incoming, np.eye(len(left)))
        gamma += np.einsum('m,ij->mij', transverse_vertex(outgoing, incoming, 1.), self.c)
        g = self.couplings
        if explicit_modes:
            return np.array([sum(g[i]*left[i, j]*gamma[mu, j, k]*right[k, l]*g[l]
                                 for i in range(len(g)) for j in range(len(g))
                                 for k in range(len(g)) for l in range(len(g))) for mu in range(4)])
        return np.einsum('i,ij,mjk,kl,l->m', g, left, gamma, right, g)

    def retain_lightest(self):
        """Identify one full eigenmode, no momentum/UV truncation or QCD scale.

        The missing response includes heavy-light source transitions. Taking
        only the heavy diagonal is therefore NOT the full matched remainder.
        """
        _, modes = np.linalg.eigh(self.hessian)
        e = modes[:, :1]
        return CollectiveChannel(e.T@self.hessian@e, e.T@self.couplings, e.T@self.c@e)


@dataclass(frozen=True)
class QuarkCollectiveReference:
    scalar: CollectiveChannel
    pion: CollectiveChannel
    mass: float

    def __post_init__(self):
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError('Positive explicit common external quark mass required.')

    def retain_lightest(self):
        return QuarkCollectiveReference(self.scalar.retain_lightest(), self.pion.retain_lightest(), self.mass)


def external_states(p, mass):
    """24x12 Dirac embedding: spin(-,+) x flavor(u,d) x color(0,1,2)."""
    p = np.asarray(p, float)
    if p[0] <= 0 or abs(dot(p, p)-mass*mass) > 2e-11:
        raise ValueError('Future on-shell external quarks required.')
    u = np.column_stack([lf_spinor(*lf_coordinates(p), mass, h) for h in [-1, 1]])
    return np.kron(u, np.eye(6))


def product(left, right):
    """Resolved outgoing1,outgoing2,incoming1,incoming2 matrix."""
    return np.einsum('ab,cd->acbd', left, right).reshape(144, 144)


def radiative_current(model, outgoing, incoming, *, propagation='covariant', explicit_modes=False):
    """All tree O(e g_i g_j) qq+gamma*->qq attachments, e/i stripped.

    q=p1'+p2'-p1-p2 is incoming. Both exchanged momenta r=p1'-p1,
    s=p2'-p2 are spacelike; photon virtuality need not be zero. This is the
    labeled direct amplitude, not yet identical-fermion antisymmetrized.
    Output currents (4,144,144) have units GeV^-1 with this spinor convention.
    D starts at two photons and is absent at this order, NOT set to zero.
    """
    if propagation not in ('covariant', 'propagating', 'instantaneous', 'light_front'):
        raise ValueError('Unknown propagator control.')
    out, inc = np.asarray(outgoing, float), np.asarray(incoming, float)
    if out.shape != (2, 4) or inc.shape != (2, 4) or not np.isfinite([out, inc]).all():
        raise ValueError('Two incoming/outgoing finite four-momenta required.')
    q = np.sum(out-inc, axis=0)
    r, s = out-inc
    u = [external_states(p, model.mass) for p in inc]
    bar = [external_states(p, model.mass).conj().T@np.kron(GAMMA[0], np.eye(6)) for p in out]
    b = bilinear_matrices()
    charge = np.kron(np.eye(4), np.kron(np.diag([2/3, -1/3]), np.eye(3)))
    em = [np.kron(g, np.eye(6))@charge for g in GAMMA]
    lines = [[bar[j]@a@u[j] for a in b] for j in range(2)]
    endpoint = np.zeros((4, 144, 144), complex)
    collective_representation = 'light_front' if propagation == 'light_front' else 'covariant'
    for leg in range(2):
        before = np.kron(fermion_propagators(inc[leg]+q, model.mass)[propagation], np.eye(6))
        after = np.kron(fermion_propagators(out[leg]-q, model.mass)[propagation], np.eye(6))
        transfer = (s, r)[leg]
        weights = [model.scalar.weight(transfer,representation=collective_representation)]
        weights += [model.pion.weight(transfer,representation=collective_representation)]*3
        for mu in range(4):
            for a, weight in enumerate(weights):
                dressed = bar[leg]@(b[a]@before@em[mu]+em[mu]@after@b[a])@u[leg]
                endpoint[mu] += weight*(product(dressed, lines[1][a]) if leg == 0
                                        else product(lines[0][a], dressed))
    flavor_exchange = 1j*(product(lines[0][1], lines[1][2])-product(lines[0][2], lines[1][1]))
    coefficient = model.pion.current(r, -s, explicit_modes=explicit_modes,representation=collective_representation)
    interaction = coefficient[:, None, None]*flavor_exchange
    # Instantaneous-only is a diagnostic partition of the endpoint terms.
    # Never count the collective current twice when assembling the LF sum.
    total = endpoint+interaction if propagation != 'instantaneous' else endpoint
    return dict(total=total, endpoint=endpoint, interaction=interaction, photon=q,
                exchange=np.array([r, s]))


def contract(current, four_vector):
    return np.einsum('m,mij->ij', np.asarray(four_vector)*[1, -1, -1, -1], current)


def gluon_exchange(left, right, transfer, representation):
    """J1.J2/t or independent two transverse gluons + instantaneous piece.

    Currents must both be conserved at this transfer (the photon attachments
    on a radiating quark line are summed first). Color factors are in the
    open endpoints. No Fock regulator, sector truncation or H2 is introduced.
    """
    t = dot(transfer, transfer)
    pp, x, y = lf_coordinates(transfer)
    if t >= -1e-12 or abs(pp) < 1e-10:
        raise ValueError('Spacelike transfer with nonzero signed plus required.')
    if representation == 'covariant':
        return sum(eta*product(left[a], right[a]) for a, eta in enumerate([1,-1,-1,-1]))/t
    if representation not in ('light_front', 'propagating'):
        raise ValueError('Unknown gluon representation.')
    result = np.zeros((144,144), complex)
    sign = np.sign(pp)
    for h in [-1,1]:
        eps = polarization_cartesian(polarization(sign*pp, sign*x, sign*y, h))
        j1 = np.einsum('m,mij->ij', eps*np.array([1,-1,-1,-1]), left)
        j2 = np.einsum('m,mij->ij', eps.conj()*np.array([1,-1,-1,-1]), right)
        result -= product(j1,j2)/t
    if representation == 'light_front':
        result -= product((left[0]+left[3])/np.sqrt(2), (right[0]+right[3])/np.sqrt(2))/pp**2
    return result


def canonical_radiative_current(mass, outgoing, incoming, gs_squared,
                                *, quark_propagation='covariant', gluon_representation='covariant'):
    """Bare order e*g_s^2 color-vector current, distinct from soft S/P.

    The sign follows the preserved canonical J1.J2/t convention. Photon
    couples to quark lines; gluons are electrically neutral. Each complete
    radiating line is conserved in the gluon index only after both photon
    attachments AND instantaneous quark terms are included.
    """
    if not np.isfinite(gs_squared) or gs_squared <= 0:
        raise ValueError('Explicit positive finite g_s^2 required.')
    if quark_propagation not in ('covariant', 'light_front', 'propagating'):
        raise ValueError('Unknown quark representation.')
    out, inc = np.asarray(outgoing, float), np.asarray(incoming, float)
    if out.shape != (2,4) or inc.shape != (2,4) or not np.isfinite([out,inc]).all():
        raise ValueError('Two finite incoming/outgoing four-momenta required.')
    q = np.sum(out-inc,axis=0); r,s = out-inc
    u = [external_states(p,mass) for p in inc]
    bar = [external_states(p,mass).conj().T@np.kron(GAMMA[0],np.eye(6)) for p in out]
    em = [np.kron(g,np.kron(np.diag([2/3,-1/3]),np.eye(3))) for g in GAMMA]
    result = np.zeros((4,144,144),complex); gluon_divergence = 0.
    for leg in range(2):
        before = np.kron(fermion_propagators(inc[leg]+q,mass)[quark_propagation],np.eye(6))
        after = np.kron(fermion_propagators(out[leg]-q,mass)[quark_propagation],np.eye(6))
        transfer = (s,r)[leg]
        for color in generators():
            vertices = [np.kron(g,np.kron(np.eye(2),color)) for g in GAMMA]
            spectator = np.array([bar[1-leg]@v@u[1-leg] for v in vertices])
            for mu in range(4):
                radiation = np.array([bar[leg]@(v@before@em[mu]+em[mu]@after@v)@u[leg] for v in vertices])
                gluon_divergence = max(gluon_divergence,float(np.linalg.norm(contract(radiation,transfer))))
                left,right = (radiation,spectator) if leg == 0 else (spectator,radiation)
                result[mu] += gs_squared*gluon_exchange(left,right,transfer,gluon_representation)
    return dict(total=result,photon=q,max_radiating_gluon_divergence=gluon_divergence)


def benchmark_reference():
    """Old tree coefficients solely as a declared amplitude control.

    This is NOT the quantum-cell susceptibility or a quantum-rematched
    action. Shared source C has unfitted directions; D is not a one-photon
    coefficient. No pion calibration is transferred to nucleon observables.
    """
    from .common_action_pilot import calibrated_action
    base = calibrated_action(.13957039, .0922, .515, .004, .6, 1.4, .12, [3., 1.])
    vacuum = base.solve(.004)['vacuum']
    hp, hs = base.matrices(vacuum)
    # Explicit non-diagonal source direction, not a new physical fit.
    c = np.array([[1.7, .35], [.35, -.2]])
    return QuarkCollectiveReference(CollectiveChannel(hs, base.yukawa, np.zeros((2, 2))),
                                   CollectiveChannel(hp, base.yukawa, c),
                                   float(.004+base.yukawa@vacuum))


def benchmark_momenta(mass, variation=0.):
    """Noncollinear on-shell routes in GeV; no empirical data."""
    spatial = np.array([[.17, -.11, .23], [-.09, .16, -.18],
                        [.31, .04, -.08], [-.21, -.06, .13]])
    spatial[0] += variation*np.array([.13, -.07, .11])
    carts = np.c_[np.sqrt(mass*mass+np.sum(spatial*spatial, axis=1)), spatial]
    return carts[:2], carts[2:]
