"""Ordered canonical quadratic-current Wick descendants at Delta+=0.

The A-annihilator/A-creator contraction in the ordered quark J^-[2]
uses the SAME longitudinal envelope and transverse coincidence trace as
the canonical instantaneous-fermion Hamiltonian self term. Normal ordering
must subtract both partners. Antiquark blocking and the fermion-vacuum
gluon descendant are included separately below. These are descendants of
the ORDERED product psi_-^dagger psi_-, not a charge-conjugation-symmetrized
or renormalized composite current. The vacuum identity is subtracted.
Propagating loops, zero modes and physical finite conditions remain open.
"""
from math import pi

import numpy as np

from .m2_canonical_quark_self import CF,_integer,quark_wick_sums


def _trace_scale(total_k2, total_pplus_GeV, coincident_trace_GeV2):
    total = _integer(total_k2, "total_k2")
    pp, trace = float(total_pplus_GeV), float(coincident_trace_GeV2)
    if not np.all(np.isfinite([pp, trace])) or pp <= 0 or trace < 0:
        raise ValueError("positive P+ and explicit nonnegative contraction trace required")
    return trace * total**2 / (2*pi*pp**2)


def quark_minus_current_wick(quark_k2,*,total_k2,total_pplus_GeV,
                             boson_modes,fermion_modes,coincident_trace_GeV2):
    """Charge-stripped quark J^-/(2P+) spin scalar, per g_s^2.

    At Delta+=0 this local contraction has no transverse-momentum dependence:
      C_F I_trace K2^2/(2 pi P+^2) sum_k 1/[k2 (p2-k2)^2].
    Multiply the quark's charge and displaced orbital density separately.
    The corresponding P^- self term has one power of (p2-k2)^-1.
    At fixed cell and field regulator this current is its -d/dp_q+.
    In particular k2>p2 contributions must not be dropped.
    """
    total=_integer(total_k2,"total_k2")
    quark_wick_sums(quark_k2,boson_modes=boson_modes,fermion_modes=fermion_modes)
    if quark_k2>=total:
        raise ValueError("external constituent must have 0<p2<K2")
    pplus=float(total_pplus_GeV);trace=float(coincident_trace_GeV2)
    if not np.all(np.isfinite([pplus,trace])) or pplus<=0 or trace<0:
        raise ValueError("positive P+ and explicit nonnegative contraction trace required")
    k=2*np.arange(1,boson_modes+1,dtype=float)
    return float(CF*trace*total**2/(2*pi*pplus**2)*np.sum(1/(k*(quark_k2-k)**2)))


def antiquark_minus_current_wick(antiquark_k2, *, total_k2, total_pplus_GeV,
                                 boson_modes, fermion_modes, coincident_trace_GeV2):
    """Physical-antiquark-charge-stripped ordered J-/(2P+), per g_s².

    The vacuum A-annihilator/A-creator pair surrounds d d-dagger.
    CAR gives the physical antiquark charge minus; the constrained mode
    is -(p2+k2), NOT p2-k2. Multiply the physical antiquark charge once.
    Equating this to the quark result would silently change the ordering.
    """
    p = _integer(antiquark_k2, "antiquark_k2")
    quark_wick_sums(p, boson_modes=boson_modes, fermion_modes=fermion_modes)
    scale = _trace_scale(total_k2, total_pplus_GeV, coincident_trace_GeV2)
    if p >= total_k2:
        raise ValueError("external constituent must have 0<p2<K2")
    k = 2*np.arange(1, boson_modes+1, dtype=float)
    return float(CF*scale*np.sum(1/(k*(p+k)**2)))


def gluon_minus_current_wick(gluon_k2, *, total_k2, total_pplus_GeV,
                             boson_modes, fermion_modes, coincident_trace_GeV2,
                             loop_flavor_charges):
    """Connected gluon descendant of the ordered fermion vacuum contraction.

    Returns J-/(2P+) per g_s² INCLUDING the supplied quark flavor charges.
    The two Bose number orderings give (k-r)^-2 + (k+r)^-2. T_F=1/2;
    the two good-spin traces cancel the constrained-field factor 1/2.
    A gluon's bare charge is zero but this ordered composite descendant
    is not: it is not the complete two-gluon/one-current loop amplitude.
    Normal-order subtraction removes it. Do not apply Furry cancellation
    to this isolated equal-LF-time term or infer a physical gluon charge.
    """
    k = _integer(gluon_k2, "gluon_k2")
    nb = _integer(boson_modes, "boson_modes")
    nf = _integer(fermion_modes, "fermion_modes")
    scale = _trace_scale(total_k2, total_pplus_GeV, coincident_trace_GeV2)
    if k % 2 or k > 2*nb or k >= total_k2:
        raise ValueError("external PBC gluon must belong to the field regulator and system")
    charges = np.asarray(loop_flavor_charges, dtype=float)
    if charges.ndim != 1 or not np.all(np.isfinite(charges)):
        raise ValueError("explicit finite list of quark loop flavor charges required")
    r = 2*np.arange(nf, dtype=float)+1
    return float(.5*sum(charges)*scale/k*np.sum(1/(k-r)**2+1/(k+r)**2))


def ordered_wick_coefficient(species, mode, *, total_k2, total_pplus_GeV,
                             boson_modes, fermion_modes, coincident_trace_GeV2,
                             loop_flavor_charges):
    """One scalar coefficient; q/qbar charges external, gluon loop charges included."""
    args = dict(total_k2=total_k2, total_pplus_GeV=total_pplus_GeV,
                boson_modes=boson_modes, fermion_modes=fermion_modes,
                coincident_trace_GeV2=coincident_trace_GeV2)
    if species == "q":
        return quark_minus_current_wick(mode, **args)
    if species == "qbar":
        return antiquark_minus_current_wick(mode, **args)
    if species == "g":
        return gluon_minus_current_wick(mode, **args, loop_flavor_charges=loop_flavor_charges)
    raise ValueError("quark, antiquark or gluon required")
