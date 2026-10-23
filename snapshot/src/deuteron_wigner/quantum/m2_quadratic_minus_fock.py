"""Number-changing canonical two-gluon J-minus kernels and coincidence maps.

Per gs^2, before physical flavor charge. Normal-ordered IF descendants only;
no propagating loop, Wick contraction or physical finite current is supplied.
The constrained denominators appear squared at Delta+=0 because BOTH
eliminated fermion components are present in J-minus.
"""
from functools import lru_cache
from math import pi,sqrt
import numpy as np

from .m2_coupled_longitudinal_jacobi import jacobi_frame,oscillator_shell,oscillator_rotation
from .m2_coupled_pilot import _displacement
from .m2_instantaneous_pair_exit import (
    _route,pair_spin_factor,pair_ordered_color,pair_emission_frame,
)
from .m2_excluded_pair_conversion import (
    _route as conversion_route,conversion_spin,conversion_ordered_color,
)
from .m2_six_body_quartic import (
    double_split_frame,valence_ordered_color,gluon_pair_ordered_color,pair_scattering_spin,
)
from .m2_excluded_fock_kinematics import frame as frame5
from .m2_six_body_kinematics import frame as frame6


def _plus(value):
    p=float(value)
    if not np.isfinite(p) or p<=0:raise ValueError("explicit positive total P+ required")
    return p


def pair_emission_minus_kernel(incoming,outgoing,quark,helicities,*,total_pplus_GeV):
    """qqq→qqqgg distinguished assignments; divide by sqrt(2!) in Fock map."""
    _route(incoming,outgoing,quark);pp=_plus(total_pplus_GeV)
    ho,hi,h1,h2=helicities;p=incoming[quark];k1,k2=outgoing[3:]
    core=(pair_spin_factor(ho,hi,h1,h2)*pair_ordered_color(quark)/(p-k2)**2
          +pair_spin_factor(ho,hi,h2,h1)*pair_ordered_color(quark,True)/(p-k1)**2)
    return sum(incoming)**2/(4*pi*pp**2*sqrt(k1*k2))*core


def conversion_minus_kernel(incoming,outgoing,helicities,*,total_pplus_GeV):
    """qqqgg→sea distinguished gluons; 2 A4/sqrt(2!) is external."""
    conversion_route(incoming,outgoing);pp=_plus(total_pplus_GeV)
    hq,ha,h1,h2=helicities;k1,k2=incoming[3:];anti=outgoing[4]
    core=(conversion_spin(hq,ha,h1,h2)*conversion_ordered_color()/(k2-anti)**2
          +conversion_spin(hq,ha,h2,h1)*conversion_ordered_color(True)/(k1-anti)**2)
    return sum(incoming)**2/(4*pi*pp**2*sqrt(k1*k2))*core


def direct_six_minus_kernel(route,helicities,*,total_pplus_GeV):
    """qqqg→Q6 IF current, INCLUDING the source's exact Fock factors.

    valence_IF has sqrt(6)/2; gluon_IF_pair has 2. No IG/A4 current
    is inferred from replacing a Hamiltonian coefficient.
    """
    double_split_frame(route);pp=_plus(total_pplus_GeV);K=sum(route.incoming)
    if route.component=="valence_IF":
        ho,hi,h1,h2=helicities;p=route.incoming[route.emitter];k1,k2=route.outgoing[4:]
        core=(pair_spin_factor(ho,hi,h1,h2)*valence_ordered_color(route.emitter)/(p-k2)**2
            +pair_spin_factor(ho,hi,h2,h1)*valence_ordered_color(route.emitter,True)/(p-k1)**2)
        return sqrt(6)/2*K**2/(4*pi*pp**2*sqrt(k1*k2))*core
    if route.component=="gluon_IF_pair":
        hq,ha,hout,hin=helicities;k=route.outgoing;kin=route.incoming[3]
        direct,reverse=pair_scattering_spin(hq,ha,hout,hin)
        core=(direct*gluon_pair_ordered_color()/(kin-k[4])**2
               +reverse*gluon_pair_ordered_color(True)/(-k[5]-k[4])**2)
        return 2*K**2/(4*pi*pp**2*sqrt(k[5]*kin))*core
    raise ValueError("explicit valence_IF or gluon_IF_pair current source required")


def double_local_orbital(parent,active,daughter_fractions,rotation,in_cutoff,out_cutoff,
                         b_GeV,transfer):
    """Two local relative coordinates, with the full parent photon recoil."""
    d=len(parent.fractions)-1;b=float(b_GeV);delta=np.asarray(transfer,float)
    x=np.asarray(daughter_fractions,float)
    if (x.shape!=(3,) or delta.shape!=(2,) or not np.all(np.isfinite(np.r_[x,b,delta]))
            or min(*x,b)<=0 or abs(x.sum()-parent.fractions[active])>2e-14):
        raise ValueError("three positive daughters must sum to the explicit active parent")
    shift=parent.forward[:-1,active,None]*delta
    a=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[a,a.conj()]
    incoming=oscillator_shell(d,in_cutoff);outgoing=oscillator_shell(d+2,out_cutoff)
    result=np.zeros((len(outgoing),len(incoming)),complex)
    for row,o in enumerate(outgoing):
        if o[d]!=o[2*d+2] or o[d+1]!=o[2*d+3]:continue
        po=o[:d]+o[d+2:2*d+2]
        for col,i in enumerate(incoming):
            result[row,col]=np.prod([_displacement(a,c,z) for a,c,z in zip(po,i,alpha)])
    return b*b/pi*sqrt(np.prod(x)/x.sum())*oscillator_rotation(rotation,out_cutoff).T@result


@lru_cache(maxsize=2048)
def pair_emission_minus_orbital(incoming,outgoing,quark,in_cutoff,out_cutoff,b_GeV,transfer):
    fo=_route(incoming,outgoing,quark)
    value=double_local_orbital(jacobi_frame(incoming),quark,fo.fractions[[quark,3,4]],
        pair_emission_frame(incoming,outgoing,quark),in_cutoff,out_cutoff,b_GeV,transfer)
    value.setflags(write=False)
    return value


@lru_cache(maxsize=2048)
def direct_six_minus_orbital(route,in_cutoff,out_cutoff,b_GeV,transfer):
    if route.component not in ("valence_IF","gluon_IF_pair"):
        raise ValueError("explicit canonical quadratic current component required")
    fo=frame6(route.sector,route.outgoing)
    value=double_local_orbital(frame5("qqqg",route.incoming),route.emitter,
        fo.fractions[list(route.daughters)],double_split_frame(route),in_cutoff,out_cutoff,b_GeV,transfer)
    value.setflags(write=False)
    return value


def five_pair_adapted_rows(f,pair):
    """Real pair-first Jacobi rows, same tree as the five-body Hamiltonian."""
    pair=tuple(pair)
    if len(pair)!=2 or len(set(pair))!=2 or min(pair)<0 or max(pair)>=5:
        raise ValueError("two distinct five-body slots required")
    rest=tuple(i for i in range(5) if i not in pair);tree=(pair,((rest[0],rest[1]),rest[2]));rows=[]
    def visit(node):
        if not isinstance(node,tuple):return (node,)
        left,right=map(visit,node);a,b=f.fractions[list(left)].sum(),f.fractions[list(right)].sum()
        row=np.zeros(5);row[list(left)]=sqrt(b/(a*(a+b)));row[list(right)]=-sqrt(a/(b*(a+b)))
        rows.append(row);return left+right
    visit(tree)
    return np.array(rows)


@lru_cache(maxsize=4096)
def five_displaced_pair_contact(in_sector,out_sector,incoming,outgoing,pair,
                                in_cutoff,out_cutoff,b_GeV,transfer):
    """qqqgg qg scattering or gg→sea coincidence, with explicit endpoints."""
    fi,fo=frame5(in_sector,incoming),frame5(out_sector,outgoing)
    if sum(incoming)!=sum(outgoing) or any(incoming[k]!=outgoing[k] for k in range(5) if k not in pair):
        raise ValueError("pair sum and all spectators must be conserved")
    ri,ro=five_pair_adapted_rows(fi,pair),five_pair_adapted_rows(fo,pair)
    b=float(b_GeV);delta=np.asarray(transfer,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(np.r_[delta,b])) or b<=0:
        raise ValueError("finite transfer and positive width required")
    shift=ri[1:,pair[0],None]*delta
    a=(shift[:,0]-1j*shift[:,1])/(2*b);alpha=np.r_[a,a.conj()]
    inc,out=oscillator_shell(4,in_cutoff),oscillator_shell(4,out_cutoff)
    adapted=np.zeros((len(out),len(inc)),complex)
    for row,o in enumerate(out):
        if o[0]!=o[4]:continue
        po=o[1:4]+o[5:]
        for col,i in enumerate(inc):
            if i[0]!=i[4]:continue
            pi_=i[1:4]+i[5:]
            adapted[row,col]=np.prod([_displacement(a,c,z) for a,c,z in zip(po,pi_,alpha)])
    ai=np.prod(fi.fractions[list(pair)])/sum(fi.fractions[list(pair)])
    af=np.prod(fo.fractions[list(pair)])/sum(fo.fractions[list(pair)])
    result=b*b/pi*sqrt(ai*af)*oscillator_rotation(ro@fo.inverse[:,:-1],out_cutoff).T@adapted@oscillator_rotation(
        ri@fi.inverse[:,:-1],in_cutoff)
    result.setflags(write=False)
    return result
