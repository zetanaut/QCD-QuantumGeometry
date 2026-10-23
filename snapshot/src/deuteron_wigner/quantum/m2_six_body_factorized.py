"""Exact tensor-factorized cubic Q5 <-> Q6 transitions.

This is the same normal-ordered U65 as m2_six_body_cubic, without assembly
of its many-body sparse matrix. Routing, orbital, color and spin factors
remain resolved; neither ranks nor amplitudes are truncated.
"""
from dataclasses import dataclass
from itertools import product
from math import sqrt

import numpy as np

from .m2_excluded_fock_kinematics import SPECIES as FIVE
from .m2_six_body_kinematics import SPECIES as SIX
from .m2_six_body_space import SixBodySpace
from .m2_six_body_vertices import split_routes,color_source,cubic_orbital
from .m2_six_body_cubic import SixBodyCubicAction,five_statistics_project


@dataclass(frozen=True)
class _TensorTerm:
    incoming_rows: np.ndarray
    outgoing_rows: np.ndarray
    orbital: np.ndarray
    color: np.ndarray

    def forward(self,values):
        return np.einsum("oi,ab,ibsk->oask",self.orbital,self.color,
                         values[self.incoming_rows],optimize=True)

    def reverse(self,values):
        return np.einsum("io,ba,oask->ibsk",self.orbital.conj().T,self.color.conj().T,
                         values[self.outgoing_rows],optimize=True)


def _twice_spin(spins,species):
    return sum(2*(2*spins[:,i]-1) if s=="g" else 2*(spins[:,i]%2)-1
               for i,s in enumerate(species))


def _rows(space,li,orbitals,spins):
    raw=(((li*len(space.orbitals)+orbitals[:,None,None])*space.ncolor
          +np.arange(space.ncolor)[None,:,None])*space.nspin+spins[None,None,:])
    rows=space.lookup[raw]
    if np.any(rows<0):
        raise ArithmeticError("factorized transition leaves exact Iz/Jz coordinates")
    return rows


def _terms(incoming,outgoing,channel,factor,b,mass,_orbital_kernel=None):
    if incoming.cutoff>outgoing.cutoff and _orbital_kernel is None:
        raise ValueError("outgoing shell must include incoming shell")
    ins={s:i for i,s in enumerate(incoming.longitudinal)}
    outs={s:i for i,s in enumerate(outgoing.longitudinal)}
    lz_out=np.array([sum(o[:5])-sum(o[5:]) for o in outgoing.orbitals])
    terms=[]
    for route in split_routes(incoming.total_k2,incoming.sector,channel):
        if route.sector!=outgoing.sector:
            raise ValueError("channel and outgoing sector disagree")
        color=factor*color_source(route)
        for h1,h2,h0 in product((-1,1),repeat=3):
            orbital=(cubic_orbital if _orbital_kernel is None else _orbital_kernel)(route,incoming.cutoff,outgoing.cutoff,b,mass,(h1,h2,h0))
            if not np.any(orbital):
                continue
            for m,oi,allowed in incoming.symmetry_groups:
                states=incoming.spins[allowed]
                helicity=(2*states[:,route.parent]-1 if channel.startswith("g_to")
                          else 2*(states[:,route.parent]%2)-1)
                selected=allowed[helicity==h0]
                if not len(selected):
                    continue
                changed=incoming.spins[selected].copy()
                if channel=="g_to_pair":
                    all_out=[np.column_stack((changed[:,:3],
                        np.full(len(selected),2*f+(h1+1)//2),
                        np.full(len(selected),2*f+(h2+1)//2),changed[:,3]))
                        for f in range(2)]
                else:
                    changed[:,route.parent]=((h1+1)//2 if channel=="g_to_gg" else
                        2*(changed[:,route.parent]//2)+(h1+1)//2)
                    all_out=[np.column_stack((changed,np.full(len(selected),(h2+1)//2)))]
                for spin_out in all_out:
                    change=(_twice_spin(incoming.spins[selected],FIVE[incoming.sector])
                            -_twice_spin(spin_out,SIX[outgoing.sector]))
                    if np.any(change!=change[0]) or change[0]%2:
                        raise ArithmeticError("inconsistent exact orbital spin transfer")
                    mf=m+int(change[0])//2
                    of=np.flatnonzero(lz_out==mf)
                    forbidden=np.flatnonzero(lz_out!=mf)
                    if np.linalg.norm(orbital[np.ix_(forbidden,oi)])>3e-12:
                        raise ArithmeticError("canonical orbital kernel changes the wrong Jz")
                    if not len(of):
                        continue
                    o=orbital[np.ix_(of,oi)]
                    if not np.any(o):
                        continue
                    mapped=np.ravel_multi_index(spin_out.T,outgoing.spin_dims)
                    if len(np.unique(mapped))!=len(mapped):
                        raise ArithmeticError("ordered-spin routing is not injective")
                    terms.append(_TensorTerm(
                        _rows(incoming,ins[route.incoming],oi,selected),
                        _rows(outgoing,outs[route.outgoing],of,mapped),o,color))
    return tuple(terms)


class FactorizedSixBodyCubicAction:
    physical=False
    direct_quartic_Q6P_supplied=False
    _validate=staticmethod(SixBodyCubicAction._validate)

    def __init__(self,spaces,*,out_cutoff,b_GeV,mass_GeV,_orbital_kernel=None):
        if set(spaces)!={"qqqgg","qqqq-qbar"}:
            raise ValueError("both explicit five-body coordinate spaces required")
        example=spaces["qqqgg"]
        for name,s in spaces.items():
            if s.sector!=name or any(getattr(s,k)!=getattr(example,k)
                                    for k in ("total_k2","cutoff","target","jz_twice")):
                raise ValueError("five-body source spaces disagree")
        b,mass=float(b_GeV),float(mass_GeV)
        if not np.all(np.isfinite([b,mass])) or b<=0 or mass<0:
            raise ValueError("positive width and finite nonnegative mass required")
        self.incoming=dict(spaces)
        self.outgoing={name:SixBodySpace(name,example.total_k2,out_cutoff,example.target,example.jz_twice)
                       for name in ("qqqggg","qqqq-qbar-g")}
        routes=[("qqqgg","qqqggg","q_to_qg",sqrt(3)),
                ("qqqgg","qqqggg","g_to_gg",sqrt(3)),
                ("qqqgg","qqqq-qbar-g","g_to_pair",2*sqrt(2)),
                ("qqqq-qbar","qqqq-qbar-g","q_to_qg",1),
                ("qqqq-qbar","qqqq-qbar-g","anti_to_antig",1)]
        self.components={source+":"+channel:(source,target,_terms(
            spaces[source],self.outgoing[target],channel,factor,b,mass,_orbital_kernel))
            for source,target,channel,factor in routes}

    def component_action(self,label,values,*,adjoint=False):
        """Unprojected distinguished source; paired with its identical adjoint."""
        if not isinstance(adjoint,(bool,np.bool_)):
            raise ValueError("explicit Boolean adjoint choice required")
        source,target,terms=self.components[label]
        incoming,outgoing=self.incoming[source],self.outgoing[target]
        if adjoint:
            incoming,outgoing=outgoing,incoming
        a=np.asarray(values,complex)
        if a.ndim!=2 or a.shape[0]!=incoming.dimension or not np.all(np.isfinite(a)):
            raise ValueError("finite columns on the component input space required")
        result=np.zeros((outgoing.dimension,a.shape[1]),complex)
        for term in terms:
            rows=term.incoming_rows if adjoint else term.outgoing_rows
            result[rows]+=(term.reverse(a) if adjoint else term.forward(a))
        return result

    def apply(self,columns):
        n=self._validate(self.incoming,columns)
        source={name:five_statistics_project(s,columns[name]) for name,s in self.incoming.items()}
        out={name:np.zeros((s.dimension,n),complex) for name,s in self.outgoing.items()}
        for label,(s,t,_) in self.components.items():
            out[t]+=self.component_action(label,source[s])
        return {name:s.statistics_project(out[name]) for name,s in self.outgoing.items()}

    def adjoint(self,columns):
        n=self._validate(self.outgoing,columns)
        source={name:s.statistics_project(columns[name]) for name,s in self.outgoing.items()}
        out={name:np.zeros((s.dimension,n),complex) for name,s in self.incoming.items()}
        for label,(s,t,_) in self.components.items():
            out[s]+=self.component_action(label,source[t],adjoint=True)
        return {name:five_statistics_project(s,out[name]) for name,s in self.incoming.items()}
