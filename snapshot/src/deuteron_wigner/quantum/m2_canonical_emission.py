"""Canonical bare q -> qg source, with physical momenta and LF spin.

Separate from C50/C52's historical numerical oracle: those routines pass
x-scaled momenta as physical momenta and omit canonical fermion field factors.
Here the source is -g psibar gamma.T psi A (Srivastava--Brodsky Eq.24),
u-bar u=2m, fields u/sqrt(2p+), A/sqrt(2k+), unit discrete longitudinal CAR.
No finite-C43 matching or production activation is implied.
"""
from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt

import numpy as np
from deuteron_wigner.quantum.canonical_lf_fields import (
    _helicity, lf_spinor,
)
from scipy.special import roots_hermite

from deuteron_wigner.bridge.modes.core import (
    GAMMA, ho_momentum, polarization, polarization_cartesian,
)
from deuteron_wigner.microscopic.h0.color import emitted_gluon_color_amplitudes
from .m2_coupled_longitudinal_jacobi import (
    emission_coordinate_map, emission_routes, jacobi_frame, longitudinal_basis,
    oscillator_rotation, oscillator_shell,
)
from .m2_coupled_pair_action import _nonnegative, _width, circular_to_c45
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def emission_momenta(route, global_relative, total_pplus, parent_transverse=(0.,0.)):
    """Physical (p+,px,py) for daughter quark, gluon and incoming parent."""
    emission_coordinate_map(route)  # validates routing and spectators
    frame=jacobi_frame(route.outgoing)
    xq,xg=frame.fractions[route.emitter],frame.fractions[3]
    xp=xq+xg
    pp=_width(total_pplus)
    q=np.asarray(global_relative,float)
    pt=np.asarray(parent_transverse,float)
    if q.shape!=(2,) or pt.shape!=(2,) or not np.all(np.isfinite([q,pt])):
        raise ValueError("finite transverse pairs required")
    rel=sqrt(xq*xg/xp)*q
    pq=(xq/xp)*pt+rel
    pg=(xg/xp)*pt-rel
    return (xq*pp,*pq),(xg*pp,*pg),(xp*pp,*pt)


def emission_mass_kernel_per_gs(route, global_relative, mass_GeV,
                                 h_out, h_in, h_g, *,
                                 total_pplus=3., parent_transverse=(0.,0.)):
    """Bare full-nucleon M^2 plane-wave kernel, color stripped, units GeV.

    2P_N+ times -ubar u_gamma_epsilon /
       [sqrt(2p_q+) sqrt(2p_parent+) sqrt(4*pi*k_g)].
    k_g is the UNDoubled integer. The longitudinal box cancels exactly.
    The transverse Jacobi sqrt(alpha) and HO integration are not included.
    """
    hq,hi,hg=(_helicity(h) for h in (h_out,h_in,h_g))
    pq,pg,pin=emission_momenta(route,global_relative,total_pplus,parent_transverse)
    uo,ui=lf_spinor(*pq,mass_GeV,hq),lf_spinor(*pin,mass_GeV,hi)
    eps=polarization_cartesian(polarization(*pg,hg).conj())
    gamma=sum(g*e*(1 if mu==0 else -1) for mu,(g,e) in enumerate(zip(GAMMA,eps)))
    numerator=uo.conj()@GAMMA[0]@gamma@ui
    kg=route.outgoing[3]/2
    return complex(-2*total_pplus*numerator/(2*sqrt(pq[0]*pin[0])*sqrt(4*pi*kg)))


@lru_cache(maxsize=8192)
def emission_mode_per_gs(route, n_plus, n_minus, b_GeV, mass_GeV,
                          h_out, h_in, h_g):
    """Exact polynomial-Gaussian projection, units GeV^2, circular phases.

    Includes sqrt(alpha), alpha=x_q*x_g/x_parent, from the normalized
    transverse measure. Integrates full R^2, not a +/-3.5b rectangle.
    Longitudinal and color factors must not be inserted a second time.
    """
    n,m,phase=circular_to_c45(n_plus,n_minus)
    b=_width(b_GeV)
    hq,hi,hg=(_helicity(h) for h in (h_out,h_in,h_g))
    emission_coordinate_map(route)
    # Exact Jz rule and degree-one canonical numerator; no magnitude pruning.
    if 2*m != hi-hq-2*hg or abs(m)>1:
        return 0j
    # LF spinor algebra yields a constant plus linear polynomial in Q.
    f0=emission_mass_kernel_per_gs(route,(0.,0.),mass_GeV,hq,hi,hg)
    fx=emission_mass_kernel_per_gs(route,(1.,0.),mass_GeV,hq,hi,hg)-f0
    fy=emission_mass_kernel_per_gs(route,(0.,1.),mass_GeV,hq,hi,hg)-f0
    nodes=n+abs(m)+2
    t,w=roots_hermite(nodes)
    x,y=np.meshgrid(sqrt(2)*b*t,sqrt(2)*b*t,indexing="ij")
    phi=phase*ho_momentum(n,m,x,y,b)
    poly_phi=phi*np.exp((x*x+y*y)/(2*b*b))
    integral=np.sum(np.outer(w,w)*poly_phi.conj()*(f0+fx*x+fy*y))*2*b*b/(2*pi)**2
    frame=jacobi_frame(route.outgoing)
    xq,xg=frame.fractions[route.emitter],frame.fractions[3]
    return complex(sqrt(xq*xg/(xq+xg))*integral)


@lru_cache(maxsize=2048)
def _route_orbital(route, cutoff, b, mass, hq, hi, hg):
    incoming=oscillator_shell(2,cutoff)
    outgoing=oscillator_shell(3,cutoff)
    lookup={s:i for i,s in enumerate(incoming)}
    adapted=np.zeros((len(outgoing),len(incoming)),complex)
    for row,s in enumerate(outgoing):
        parent=(s[0],s[1],s[3],s[4])
        if parent in lookup:
            adapted[row,lookup[parent]]=emission_mode_per_gs(
                route,s[2],s[5],b,mass,hq,hi,hg)
    return oscillator_rotation(emission_coordinate_map(route),cutoff).T@adapted


def apply_canonical_emission(tensor,total_k2,max_excitation,b_GeV,mass_GeV,h_g,*,_orbital_kernel=None):
    """Matrix-free coherent sum of all three emitters, no selected coupling.

    Input: (qqq longitudinal*orbital, 1 color, 64 spin/flavor).
    Output: (qqqg longitudinal*orbital, 2 C7 colors, 64 spin/flavor).
    Fixed gluon helicity is an explicit separate argument. It preserves
    spectators, flavor and the full joint S3 sign carrier.
    """
    cutoff=_nonnegative(max_excitation,"max_excitation")
    b=_width(b_GeV)
    hg=_helicity(h_g)
    mass=float(mass_GeV)
    if not np.isfinite(mass) or mass<0:
        raise ValueError("mass must be finite and nonnegative")
    q,g=(longitudinal_basis(s,total_k2) for s in ("qqq","qqqg"))
    nq,ng=len(oscillator_shell(2,cutoff)),len(oscillator_shell(3,cutoff))
    value=np.asarray(tensor,complex)
    if value.shape!=(len(q)*nq,1,64) or not np.all(np.isfinite(value)):
        raise ValueError("expected finite (qqq longitudinal*orbital,1,64) tensor")
    source=value.reshape(len(q),nq,64)
    result=np.zeros((len(g),ng,2,64),complex)
    qi,gi=({s:i for i,s in enumerate(states)} for states in (q,g))
    sf={s:i for i,s in enumerate(SPIN_FLAVOR_BASIS)}
    for route in emission_routes(total_k2):
        color=emitted_gluon_color_amplitudes(route.emitter)
        for hi,hq in ((-1,-1),(-1,1),(1,-1),(1,1)):
            block=(_route_orbital if _orbital_kernel is None else _orbital_kernel)(route,cutoff,b,mass,hq,hi,hg)
            for col,s in enumerate(SPIN_FLAVOR_BASIS):
                if 2*(s[route.emitter]%2)-1!=hi:
                    continue
                out=list(s)
                out[route.emitter]=2*(s[route.emitter]//2)+(hq+1)//2
                row=sf[tuple(out)]
                amplitude=block@source[qi[route.incoming],:,col]
                result[gi[route.outgoing],:,:,row]+=amplitude[:,None]*color[None,:]
    return result.reshape(len(g)*ng,2,64)
