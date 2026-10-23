"""Canonical direct quartic qqqg -> Q6 source kernels, per g_s^2.

Seven normal-ordered components share finite-cell normalization, complete
six-body colors, and the global double-split Jacobi contact. Wick terms
are separate. No physical finite counterterm or production state is chosen.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import pi,sqrt

import numpy as np

from deuteron_wigner.bridge.modes.core import GAMMA,GAMMA_PLUS
from deuteron_wigner.microscopic.h0.color import ColorSingletBasis,representation_generators
from deuteron_wigner.pilot.color import structure_constants
from .m2_canonical_emission import lf_spinor
from .m2_excluded_fock_vertices import antiquark_spinor
from .m2_excluded_fock_kinematics import frame as parent_frame,longitudinal_states as parent_states,_act
from .m2_coupled_longitudinal_jacobi import oscillator_shell,oscillator_rotation
from .m2_instantaneous_pair_exit import pair_spin_factor
from .m2_instantaneous_sea_exit import sea_pair_spin_factor
from .m2_excluded_four_gluon import ORDERS
from .m2_six_body_kinematics import frame,longitudinal_states,color_basis
from .m2_six_body_space import SixBodySpace

COMPONENTS=("valence_IF","valence_IG","valence_sea","gluon_A4",
            "gluon_IF_pair","gluon_IG_pair","gluon_IG_ggg")


@dataclass(frozen=True)
class DirectRoute:
    component: str
    incoming: tuple
    outgoing: tuple
    emitter: int

    @property
    def sector(self):
        return "qqqq-qbar-g" if self.component in ("valence_sea","gluon_IF_pair","gluon_IG_pair") else "qqqggg"

    @property
    def daughters(self):
        if self.component=="valence_sea":return (self.emitter,3,4)
        if self.component.startswith("valence"):return (self.emitter,4,5)
        return (3,4,5)


@lru_cache(maxsize=32)
def direct_routes(total_k2,component):
    if component not in COMPONENTS:raise ValueError("explicit canonical quartic component required")
    sector="qqqq-qbar-g" if component in ("valence_sea","gluon_IF_pair","gluon_IG_pair") else "qqqggg"
    incoming_set=set(parent_states("qqqg",total_k2));result=[]
    for out in longitudinal_states(sector,total_k2):
        for emitter in (range(3) if component.startswith("valence") else (3,)):
            if component=="valence_sea":
                inc=list(out[:3]) + [out[5]];inc[emitter]+=out[3]+out[4]
            elif component.startswith("valence"):
                inc=list(out[:4]);inc[emitter]+=out[4]+out[5]
            else:
                inc=list(out[:3])+[sum(out[3:])]
            if tuple(inc) not in incoming_set:raise ArithmeticError("quartic route left parent modes")
            result.append(DirectRoute(component,tuple(inc),out,emitter))
    return tuple(result)


def double_split_frame(route):
    if route not in direct_routes(sum(route.incoming),route.component):
        raise ValueError("valid direct quartic route required")
    a,b=parent_frame("qqqg",route.incoming),frame(route.sector,route.outgoing)
    collapse=np.zeros((4,6))
    if route.component=="valence_sea":
        collapse[:3,:3]=np.eye(3);collapse[3,5]=1;collapse[route.emitter,3:5]=1
    elif route.component.startswith("valence"):
        collapse[:,:4]=np.eye(4);collapse[route.emitter,4:6]=1
    else:
        collapse[:3,:3]=np.eye(3);collapse[3,3:6]=1
    i,j,k=route.daughters;x,y,z=b.fractions[[i,j,k]]
    first=np.zeros(6);first[i]=sqrt(y/(x*(x+y)));first[j]=-sqrt(x/(y*(x+y)))
    second=np.zeros(6);second[[i,j]]=sqrt(z/((x+y)*(x+y+z)))
    second[k]=-sqrt((x+y)/(z*(x+y+z)))
    return np.vstack((a.forward[:-1]@collapse,first,second))@b.inverse[:,:-1]


def contact_orbital(route,in_cutoff,out_cutoff,b_GeV):
    b=float(b_GeV)
    if not np.isfinite(b) or b<=0:raise ValueError("positive finite width required")
    inc,out=oscillator_shell(3,in_cutoff),oscillator_shell(5,out_cutoff)
    lookup={s:i for i,s in enumerate(inc)}
    adapted=np.zeros((len(out),len(inc)))
    for row,s in enumerate(out):
        inherited=s[:3]+s[5:8]
        if inherited in lookup and s[3]==s[8] and s[4]==s[9]:
            adapted[row,lookup[inherited]]=b*b/pi
    x,y,z=frame(route.sector,route.outgoing).fractions[list(route.daughters)]
    return sqrt(x*y*z/(x+y+z))*oscillator_rotation(double_split_frame(route),out_cutoff).T@adapted


def _source_color():
    return ColorSingletBasis.construct("qqqg").tensors.reshape(3,3,3,8,2)


def _reduce(sector,raw):
    u=color_basis(sector);raw=raw.reshape(u.shape[0],2)
    c=u.conj().T@raw
    if np.linalg.norm(raw-u@c)>4e-11:
        raise ArithmeticError("quartic source leaves complete six-particle color carrier")
    return c


@lru_cache(maxsize=6,typed=True)
def valence_ordered_color(emitter,reverse=False):
    if emitter not in (0,1,2) or isinstance(emitter,bool) or not isinstance(reverse,bool):
        raise ValueError("valence emitter and explicit order required")
    source=_source_color();t=representation_generators("3")
    raw=np.zeros((3,3,3,8,8,8,2),complex)
    for a,ta in enumerate(t):
        for b,tb in enumerate(t):
            raw[:,:,:,:,a,b]=_act(tb@ta if reverse else ta@tb,source,emitter)
    return _reduce("qqqggg",raw)


@lru_cache(maxsize=3)
def valence_sea_color(emitter):
    source=_source_color();raw=np.zeros((3,3,3,3,3,8,2),complex)
    for t in representation_generators("3"):
        raw+=np.einsum("ijkan,bc->ijkbcan",_act(t,source,emitter),t)
    return _reduce("qqqq-qbar-g",raw)


@lru_cache(maxsize=2,typed=True)
def gluon_pair_ordered_color(reverse=False):
    if not isinstance(reverse,bool):raise ValueError("explicit color order required")
    source=_source_color().reshape(27,8,2)
    t=np.asarray(representation_generators("3"))
    raw=(np.einsum("cij,ajk,rcn->rikan",t,t,source) if reverse else
         np.einsum("aij,cjk,rcn->rikan",t,t,source))
    return _reduce("qqqq-qbar-g",raw)


@lru_cache(maxsize=1)
def gluon_pair_current_color():
    source=_source_color().reshape(27,8,2)
    raw=np.einsum("dij,dac,rcn->rijan",np.array(representation_generators("3")),
                  np.array(representation_generators("8")),source)
    return _reduce("qqqq-qbar-g",raw)


@lru_cache(maxsize=1)
def gluon_three_current_color():
    # Distinguished surviving gluon a (slot3), pair b,c (slots4,5).
    raw=np.einsum("dbc,dae,ijken->ijkabcn",structure_constants(),
                  np.array(representation_generators("8")),_source_color())
    return _reduce("qqqggg",raw)


@lru_cache(maxsize=24)
def A4_ordered_color(order):
    if tuple(order) not in ORDERS:raise ValueError("four external-leg assignment required")
    f=structure_constants()
    tensor=np.einsum("xab,xcd->abcd",f,f).transpose(np.argsort(order))
    raw=np.einsum("abcd,ijkdn->ijkabcn",tensor,_source_color())
    return _reduce("qqqggg",raw)


@lru_cache(maxsize=16,typed=True)
def pair_scattering_spin(hq,ha,hout,hin):
    if any(h not in (-1,1) or isinstance(h,(bool,np.bool_)) for h in (hq,ha,hout,hin)):
        raise ValueError("signed physical helicities required")
    u=lf_spinor(1.,0.,0.,.35,hq)/sqrt(2)
    v=antiquark_spinor(1.,0.,0.,.35,ha)/sqrt(2)
    eout=(-np.array([hout,1j])/sqrt(2)).conj()
    ein=-np.array([hin,1j])/sqrt(2)
    left,right=[sum(GAMMA[i+1]*e[i] for i in range(2)) for e in (eout,ein)]
    return (complex(u.conj()@GAMMA[0]@left@GAMMA_PLUS@right@v),
            complex(u.conj()@GAMMA[0]@right@GAMMA_PLUS@left@v))


def mass_kernel(route,helicities):
    """Fully normalized ordered-slot coefficient, INCLUDING Fock factors.

    Valence: (h_quark_out,h_quark_in,h_new1,h_new2).
    Gluon parent: (h_out_slot3,h_out_slot4,h_out_slot5,h_gluon_in).
    """
    double_split_frame(route)
    if any(h not in (-1,1) or isinstance(h,(bool,np.bool_)) for h in helicities) or len(helicities)!=4:
        raise ValueError("four signed physical helicities required")
    k=route.outgoing;K=sum(k);name=route.component
    if name in ("valence_IF","valence_IG"):
        ho,hi,h1,h2=helicities;p=route.incoming[route.emitter];k1,k2=k[4:]
        c,d=valence_ordered_color(route.emitter),valence_ordered_color(route.emitter,True)
        if name=="valence_IF":
            core=K/(2*pi*sqrt(k1*k2))*(pair_spin_factor(ho,hi,h1,h2)*c/(p-k2)
                    +pair_spin_factor(ho,hi,h2,h1)*d/(p-k1))
        else:
            core=-K/(pi*sqrt(k1*k2))*(k1-k2)/(k1+k2)**2*(h1*h2-1)/2*int(ho==hi)*(c-d)
        return sqrt(6)/2*core
    if name=="valence_sea":
        ho,hi,hq,ha=helicities
        return 2*(2*K/(pi*(k[3]+k[4])**2)*int(ho==hi)
                  *sea_pair_spin_factor(hq,ha)*valence_sea_color(route.emitter))
    h1,h2,h3,hin=helicities;kin=route.incoming[3]
    if name=="gluon_A4":
        eps=[-np.array([h,1j])/sqrt(2) for h in helicities]
        eps[:3]=[e.conj() for e in eps[:3]]
        tensor=sum(np.dot(eps[p[0]],eps[p[2]])*np.dot(eps[p[1]],eps[p[3]])*A4_ordered_color(p) for p in ORDERS)
        return K/(8*pi*sqrt(6*kin*np.prod(k[3:])))*tensor
    if name=="gluon_IF_pair":
        direct,reverse=pair_scattering_spin(h1,h2,h3,hin)
        # Incoming-gluon momentum is positive; outgoing-gluon momentum
        # crossing the constrained line is negative. Both odd signs retained.
        return 2*K/(2*pi*sqrt(k[5]*kin))*(direct*gluon_pair_ordered_color()/(kin-k[4])
            +reverse*gluon_pair_ordered_color(True)/(-k[5]-k[4]))
    if name=="gluon_IG_pair":
        return 2*(2*K/(pi*(k[3]+k[4])**2)*sea_pair_spin_factor(h1,h2)
                  *int(h3==hin)*(kin+k[5])/(2*sqrt(kin*k[5]))*gluon_pair_current_color())
    # j_g number times j_g pair: each current order once.
    pair=-1j*(k[4]-k[5])/(2*sqrt(k[4]*k[5]))*(h2*h3-1)/2
    return sqrt(6)/2*(2*K/(pi*(k[4]+k[5])**2)*int(h1==hin)
        *(kin+k[3])/(2*sqrt(kin*k[3]))*pair*gluon_three_current_color())


@dataclass
class DirectSixSource:
    retained_pilot: object
    space: object
    retained_columns: np.ndarray
    matrix_per_gs2: np.ndarray
    component: str


def build_direct_source(pilot,*,out_cutoff,component,_retained_vectors=None,_orbital_kernel=None):
    routes=direct_routes(pilot.total_k2,component)
    sector="qqqq-qbar-g" if component in ("valence_sea","gluon_IF_pair","gluon_IG_pair") else "qqqggg"
    space=SixBodySpace(sector,pilot.total_k2,out_cutoff,pilot.target,pilot.jz_twice)
    if _retained_vectors is None and pilot.cutoff>out_cutoff:
        raise ValueError("outgoing shell must contain parent shell")
    incoming=parent_states("qqqg",pilot.total_k2);ni=len(oscillator_shell(3,pilot.cutoff))
    nq=pilot.bases[0].shape[1];count=len(pilot.matrix)-nq
    from .m2_source_vectors import SourceImage, source_tensor
    if _retained_vectors is None:
        source=np.zeros((len(incoming),ni,2,64,2,count),complex)
        offset=0
        for hg,u in enumerate(pilot.bases[1:]):
            n=u.shape[1]
            source[:,:,:,:,hg,offset:offset+n]=u.toarray().reshape(len(incoming),ni,2,64,n)
            offset+=n
    else:
        source=source_tensor(pilot,"qqqg",_retained_vectors)
        count=source.shape[-1]
    lookup={s:i for i,s in enumerate(incoming)}
    target={s:i for i,s in enumerate(space.longitudinal)}
    sf=np.array(list(product(range(4),repeat=3)))
    image=np.zeros((space.dimension,count),complex)
    for route in routes:
        orbital=(contact_orbital if _orbital_kernel is None else _orbital_kernel)(
            route,pilot.cutoff,out_cutoff,pilot.width_GeV)
        for hs in product((-1,1),repeat=4):
            color=mass_kernel(route,hs)
            if not np.any(color):continue
            if component.startswith("valence"):
                ho,hi,h1,h2=hs
                selected=np.flatnonzero(2*(sf[:,route.emitter]%2)-1==hi)
                changed=sf[selected].copy();changed[:,route.emitter]=2*(changed[:,route.emitter]//2)+(ho+1)//2
                for hg in range(2):
                    value=np.take(source[lookup[route.incoming],:,:,:,hg],selected,axis=2)
                    amp=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                    if component=="valence_sea":
                        outspins=[np.column_stack((changed,np.full(len(selected),2*f+(h1+1)//2),
                            np.full(len(selected),2*f+(h2+1)//2),np.full(len(selected),hg))) for f in range(2)]
                    else:
                        outspins=[np.column_stack((changed,np.full(len(selected),hg),
                            np.full(len(selected),(h1+1)//2),np.full(len(selected),(h2+1)//2)))]
                    for outsf in outspins:
                        space.add(image,target[route.outgoing],np.ravel_multi_index(outsf.T,space.spin_dims),amp)
            else:
                h1,h2,h3,hin=hs
                value=source[lookup[route.incoming],:,:,:,(hin+1)//2]
                amp=np.einsum("oi,ab,ibsk->oask",orbital,color,value,optimize=True)
                if sector=="qqqggg":
                    outspins=[np.column_stack((sf,np.full(64,(h1+1)//2),np.full(64,(h2+1)//2),np.full(64,(h3+1)//2)))]
                else:
                    outspins=[np.column_stack((sf,np.full(64,2*f+(h1+1)//2),
                        np.full(64,2*f+(h2+1)//2),np.full(64,(h3+1)//2))) for f in range(2)]
                for outsf in outspins:
                    space.add(image,target[route.outgoing],np.ravel_multi_index(outsf.T,space.spin_dims),amp)
    image=space.statistics_project(image)
    if _retained_vectors is not None:
        return SourceImage(pilot,space,2,image,{component:image})
    return DirectSixSource(pilot,space,np.arange(nq,len(pilot.matrix)),image,component)


def apply_direct_source(pilot,vectors,*,out_cutoff,component):
    """One of all seven exact Q6P quartic actions on arbitrary retained columns."""
    from .m2_source_vectors import validate_source_vectors
    return build_direct_source(pilot,out_cutoff=out_cutoff,component=component,
                               _retained_vectors=validate_source_vectors(pilot,vectors))
