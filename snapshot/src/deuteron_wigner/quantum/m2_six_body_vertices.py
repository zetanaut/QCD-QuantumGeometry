"""Canonical cubic Q5 -> Q6 kernels and exact coordinate routing.

All amplitudes are distinguished-slot coefficients, before identical-particle
creation factors. The physical antiquark color is -T^T; its vector spinor
bilinear has reversed particle endpoints. No quartic Q6P source is supplied.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import pi,sqrt
from operator import index

import numpy as np
from scipy.special import roots_hermite

from deuteron_wigner.bridge.modes.core import GAMMA,ho_momentum,polarization
from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_canonical_emission import lf_spinor,_helicity
from .m2_excluded_fock_vertices import antiquark_spinor,_slash_epsilon,three_gluon_numerator
from .m2_excluded_fock_kinematics import (
    longitudinal_states as five_states,frame as five_frame,color_basis as five_color,
)
from .m2_coupled_longitudinal_jacobi import oscillator_shell,oscillator_rotation
from .m2_six_body_kinematics import frame,color_basis,color_emission

@dataclass(frozen=True)
class SixSplitRoute:
    source_sector: str
    channel: str
    incoming: tuple
    outgoing: tuple
    parent: int

    @property
    def sector(self):
        return "qqqggg" if self.source_sector=="qqqgg" and self.channel!="g_to_pair" else "qqqq-qbar-g"

    @property
    def daughters(self):
        return (3,4) if self.channel=="g_to_pair" else (self.parent,5)


@lru_cache(maxsize=16)
def split_routes(total_k2,source_sector,channel):
    valid={"qqqgg":("q_to_qg","g_to_gg","g_to_pair"),
           "qqqq-qbar":("q_to_qg","anti_to_antig")}
    if channel not in valid.get(source_sector,()):
        raise ValueError("valid canonical Q5 -> Q6 channel required")
    routes=[]
    for incoming in five_states(source_sector,total_k2):
        slots=range(3 if source_sector=="qqqgg" else 4) if channel=="q_to_qg" else (4,)
        for parent in slots:
            for k in range(1 if channel=="g_to_pair" else 2,incoming[parent],2):
                if channel=="g_to_pair":
                    outgoing=incoming[:3]+(incoming[4]-k,k,incoming[3])
                else:
                    outgoing=list(incoming);outgoing[parent]-=k
                    outgoing=tuple(outgoing)+(k,)
                routes.append(SixSplitRoute(source_sector,channel,incoming,outgoing,parent))
    return tuple(routes)


def split_frame(route):
    if route not in split_routes(sum(route.incoming),route.source_sector,route.channel):
        raise ValueError("valid six-particle split route required")
    a,b=five_frame(route.source_sector,route.incoming),frame(route.sector,route.outgoing)
    collapse=np.zeros((5,6))
    if route.channel=="g_to_pair":
        collapse[:3,:3]=np.eye(3);collapse[3,5]=1;collapse[4,3:5]=1
    else:
        collapse[:,:5]=np.eye(5);collapse[route.parent,5]=1
    i,j=route.daughters;xi,xj=b.fractions[[i,j]]
    split=np.zeros(6)
    split[i],split[j]=sqrt(xj/(xi*(xi+xj))),-sqrt(xi/(xj*(xi+xj)))
    return np.vstack((a.forward[:-1]@collapse,split))@b.inverse[:,:-1]


@lru_cache(maxsize=1)
def pair_color():
    source=five_color("qqqgg").reshape(3,3,3,8,8,6)
    # Old gluon slot3 becomes outgoing slot5; old slot4 creates q3,anti4.
    raw=np.einsum("bIJ,ijkabn->ijkIJ an",
                  np.array(representation_generators("3")),source).reshape(1944,6)
    u=color_basis("qqqq-qbar-g")
    reduced=u.conj().T@raw
    if np.linalg.norm(raw-u@reduced)>3e-11:
        raise ArithmeticError("pair conversion leaves complete color space")
    return reduced


def color_source(route):
    split_frame(route)
    if route.channel=="g_to_pair":
        return pair_color()
    c=color_emission(route.sector,route.parent)
    # F^(new)_(left,parent) = i f^(left,new,parent).
    return -1j*c if route.channel=="g_to_gg" else c


def daughter_momenta(route,relative,total_pplus=3.,parent_transverse=(0.,0.)):
    split_frame(route)
    q,pt=np.asarray(relative,float),np.asarray(parent_transverse,float)
    pp=float(total_pplus)
    if q.shape!=(2,) or pt.shape!=(2,) or not np.all(np.isfinite([q,pt])) or not np.isfinite(pp) or pp<=0:
        raise ValueError("finite momenta and positive total P+ required")
    x,y=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
    X=x+y;r=sqrt(x*y/X)*q
    return (x*pp,*(x/X*pt+r)),(y*pp,*(y/X*pt-r)),(X*pp,*pt)


def cubic_mass_kernel(route,relative,mass_GeV,helicities,*,total_pplus=3.,parent_transverse=(0.,0.)):
    h1,h2,h0=tuple(_helicity(h) for h in helicities)
    mass=float(mass_GeV)
    if not np.isfinite(mass) or mass<0:
        raise ValueError("finite nonnegative mass required")
    left,right,parent=daughter_momenta(route,relative,total_pplus,parent_transverse)
    if route.channel=="q_to_qg":
        u,v=lf_spinor(*left,mass,h1),lf_spinor(*parent,mass,h0)
        numerator=-u.conj()@GAMMA[0]@_slash_epsilon(polarization(*right,h2).conj())@v
    elif route.channel=="anti_to_antig":
        # The CAR sign is already in Tbar=-T^T, not inserted again here.
        vi,vf=antiquark_spinor(*parent,mass,h0),antiquark_spinor(*left,mass,h1)
        numerator=-vi.conj()@GAMMA[0]@_slash_epsilon(polarization(*right,h2).conj())@vf
    elif route.channel=="g_to_pair":
        u,v=lf_spinor(*left,mass,h1),antiquark_spinor(*right,mass,h2)
        numerator=-u.conj()@GAMMA[0]@_slash_epsilon(polarization(*parent,h0))@v
    else:
        numerator=three_gluon_numerator(left,right,parent,h1,h2,h0)
    cell=pi*sum(route.incoming)/total_pplus
    return complex(2*total_pplus*numerator/sqrt(cell*8*left[0]*right[0]*parent[0]))


@lru_cache(maxsize=32768,typed=True)
def cubic_mode(route,n_plus,n_minus,b_GeV,mass_GeV,helicities):
    if any(isinstance(n,(bool,np.bool_)) for n in (n_plus,n_minus)):
        raise ValueError("exact nonnegative occupations required")
    n_plus,n_minus=index(n_plus),index(n_minus)
    b=float(b_GeV)
    if min(n_plus,n_minus)<0 or not np.isfinite(b) or b<=0 or not np.isfinite(mass_GeV) or mass_GeV<0:
        raise ValueError("valid occupations, width and mass required")
    split_frame(route)
    h1,h2,h0=tuple(_helicity(h) for h in helicities)
    m=n_plus-n_minus
    required=((h0-h1)//2-h2 if route.channel in ("q_to_qg","anti_to_antig") else
              h0-(h1+h2)//2 if route.channel=="g_to_pair" else h0-h1-h2)
    if m!=required or abs(m)>1:return 0j
    n=min(n_plus,n_minus)
    f0=cubic_mass_kernel(route,(0.,0.),mass_GeV,helicities)
    fx=cubic_mass_kernel(route,(1.,0.),mass_GeV,helicities)-f0
    fy=cubic_mass_kernel(route,(0.,1.),mass_GeV,helicities)-f0
    nodes,weights=roots_hermite(n+abs(m)+2)
    x,y=np.meshgrid(sqrt(2)*b*nodes,sqrt(2)*b*nodes,indexing="ij")
    phi=(-1)**n*ho_momentum(n,m,x,y,b)*np.exp((x*x+y*y)/(2*b*b))
    result=np.sum(np.outer(weights,weights)*phi.conj()*(f0+fx*x+fy*y))
    a,c=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
    return complex(sqrt(a*c/(a+c))*result*2*b*b/(2*pi)**2)


@lru_cache(maxsize=8192)
def cubic_orbital(route,in_cutoff,out_cutoff,b,mass,helicities):
    incoming,outgoing=oscillator_shell(4,in_cutoff),oscillator_shell(5,out_cutoff)
    lookup={s:i for i,s in enumerate(incoming)}
    adapted=np.zeros((len(outgoing),len(incoming)),complex)
    for row,s in enumerate(outgoing):
        inherited=s[:4]+s[5:9]
        if inherited in lookup:
            adapted[row,lookup[inherited]]=cubic_mode(route,s[4],s[9],b,mass,helicities)
    return oscillator_rotation(split_frame(route),out_cutoff).T@adapted
