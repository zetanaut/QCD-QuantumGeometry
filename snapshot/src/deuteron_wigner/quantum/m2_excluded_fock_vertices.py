"""Canonical cubic qqqg exits: quark emission, gluon splitting and sea pairs.

Source: Srivastava--Brodsky hep-ph/0011372v2 Eq.(24), Appendix A/B.
Returns distinguished-daughter amplitudes. Bose/Fermi creation factors and
many-body symmetrization belong to the action, NOT these local kernels.
"""
from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt
from operator import index

import numpy as np
from deuteron_wigner.quantum.canonical_lf_fields import (
    antiquark_spinor, _lf_momentum, three_gluon_numerator,
)
from scipy.special import roots_hermite

from deuteron_wigner.bridge.modes.core import (
    GAMMA, ho_momentum, lf_dot, polarization,
    polarization_cartesian,
)
from .m2_canonical_emission import lf_spinor, _helicity
from .m2_coupled_longitudinal_jacobi import oscillator_rotation, oscillator_shell
from .m2_excluded_fock_kinematics import frame, split_frame


def daughter_momenta(route,relative,total_pplus=3.,parent_transverse=(0.,0.)):
    split_frame(route)
    q,pt = np.asarray(relative,float),np.asarray(parent_transverse,float)
    pp = float(total_pplus)
    if q.shape != (2,) or pt.shape != (2,) or not np.all(np.isfinite([q,pt])) or not np.isfinite(pp) or pp <= 0:
        raise ValueError("finite momenta and positive total P+ required")
    i,j = route.daughters
    x,y = frame(route.sector,route.outgoing).fractions[[i,j]]
    X = x+y
    r = sqrt(x*y/X)*q
    return (x*pp,*(x/X*pt+r)),(y*pp,*(y/X*pt-r)),(X*pp,*pt)


def _slash_epsilon(eps):
    e = polarization_cartesian(eps)
    return sum(g*value*(1 if mu==0 else -1) for mu,(g,value) in enumerate(zip(GAMMA,e)))


def cubic_mass_kernel(route,relative,mass_GeV,helicities,*,
                      total_pplus=3.,parent_transverse=(0.,0.)):
    """M²/g_s kernel in GeV before HO integration; all finite-cell factors once.

    helicities=(left daughter, right daughter, parent), physical h=+/-1
    for gluons and doubled physical helicity for fermions. The common field
    factor is 2P+ / sqrt(ell_- (2p0+)(2p1+)(2p2+)),
    ell_-=pi K2/P+. All quarks/antiquarks here share the caller's mass.
    """
    h1,h2,h0 = tuple(_helicity(h) for h in helicities)
    mass = float(mass_GeV)
    if not np.isfinite(mass) or mass<0:
        raise ValueError("finite nonnegative quark mass required")
    left,right,parent = daughter_momenta(route,relative,total_pplus,parent_transverse)
    if route.channel == "q_to_qg":
        u,v = lf_spinor(*left,mass,h1),lf_spinor(*parent,mass,h0)
        numerator = -u.conj()@GAMMA[0]@_slash_epsilon(polarization(*right,h2).conj())@v
    elif route.channel == "g_to_pair":
        u,v = lf_spinor(*left,mass,h1),antiquark_spinor(*right,mass,h2)
        numerator = -u.conj()@GAMMA[0]@_slash_epsilon(polarization(*parent,h0))@v
    elif route.channel == "g_to_gg":
        numerator = three_gluon_numerator(left,right,parent,h1,h2,h0)
    else:
        raise ValueError("unknown source channel")
    cell = pi*sum(route.incoming)/total_pplus
    denominator = sqrt(cell*8*left[0]*right[0]*parent[0])
    return complex(2*total_pplus*numerator/denominator)


@lru_cache(maxsize=8192,typed=True)
def cubic_mode(route,n_plus,n_minus,b_GeV,mass_GeV,helicities):
    if any(isinstance(n,(bool,np.bool_)) for n in (n_plus,n_minus)):
        raise ValueError("exact nonnegative occupations required")
    n_plus,n_minus = index(n_plus),index(n_minus)
    if min(n_plus,n_minus)<0:
        raise ValueError("nonnegative occupations required")
    b = float(b_GeV)
    if not np.isfinite(b) or b<=0:
        raise ValueError("finite positive width required")
    if not np.isfinite(mass_GeV) or mass_GeV<0:
        raise ValueError("finite nonnegative quark mass required")
    split_frame(route)  # Validate even when the angular selection gives zero.
    h1,h2,h0 = tuple(_helicity(h) for h in helicities)
    m = n_plus-n_minus
    required = ((h0-h1)//2-h2 if route.channel=="q_to_qg" else
                h0-(h1+h2)//2 if route.channel=="g_to_pair" else h0-h1-h2)
    if m != required or abs(m)>1:
        return 0j
    n = min(n_plus,n_minus)
    f0 = cubic_mass_kernel(route,(0.,0.),mass_GeV,helicities)
    fx = cubic_mass_kernel(route,(1.,0.),mass_GeV,helicities)-f0
    fy = cubic_mass_kernel(route,(0.,1.),mass_GeV,helicities)-f0
    nodes,weights = roots_hermite(n+abs(m)+2)
    x,y = np.meshgrid(sqrt(2)*b*nodes,sqrt(2)*b*nodes,indexing="ij")
    phi = (-1)**n*ho_momentum(n,m,x,y,b)
    polynomial_phi = phi*np.exp((x*x+y*y)/(2*b*b))
    result = np.sum(np.outer(weights,weights)*polynomial_phi.conj()*(f0+fx*x+fy*y))
    fractions = frame(route.sector,route.outgoing).fractions
    a,c = fractions[list(route.daughters)]
    return complex(sqrt(a*c/(a+c))*result*2*b*b/(2*pi)**2)


@lru_cache(maxsize=4096)
def cubic_orbital(route,in_cutoff,out_cutoff,b,mass,helicities):
    """Default five-body tree rows; inherited parent modes preserved exactly."""
    incoming,outgoing = oscillator_shell(3,in_cutoff),oscillator_shell(4,out_cutoff)
    lookup = {s:i for i,s in enumerate(incoming)}
    adapted = np.zeros((len(outgoing),len(incoming)),complex)
    for row,s in enumerate(outgoing):
        inherited = s[:3]+s[4:7]
        if inherited in lookup:
            adapted[row,lookup[inherited]] = cubic_mode(route,s[3],s[7],b,mass,helicities)
    return oscillator_rotation(split_frame(route),out_cutoff).T@adapted
