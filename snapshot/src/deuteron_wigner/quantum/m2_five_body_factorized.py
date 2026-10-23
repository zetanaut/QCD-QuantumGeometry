"""Tensor-factorized canonical two-body contacts within Q5, per g_s^2.

Reuses the independently normalized canonical kernels. No many-body sparse
matrix is assembled, no amplitude or color multiplicity is truncated, and
the pair-current square remains a separate (non-overlapping) contribution.
"""
from itertools import combinations,product
from math import pi,sqrt
import numpy as np

from .m2_excluded_fock_kinematics import SPECIES
from .m2_excluded_fermion_scattering import fermion_pair_color,fermion_contact_block
from .m2_excluded_current_scattering import current_pair_color,five_body_contact,canonical_gluon_current_factor
from .m2_excluded_instantaneous_fermion import ordered_five_qg_color,transverse_spin_factor
from .m2_excluded_four_gluon import quartic_spin_color
from .m2_excluded_pair_conversion import conversion_contact,instantaneous_conversion_kernel
from .m2_six_body_factorized import _TensorTerm,_rows,_twice_spin
from .m2_six_body_cubic import SixBodyCubicAction
from .m2_coupled_pair_action import _width


def contact_tensor_terms(inside,outside,li,lo,orbital,color,selected,mapped):
    """Factor one exact ordered-spin map, with physical packed endpoint labels."""
    if not np.any(color) or not np.any(orbital):
        return ()
    d=len(outside.orbitals[0])//2
    out_lz=np.array([sum(o[:d])-sum(o[d:]) for o in outside.orbitals])
    inverse=np.full(inside.nspin,-1,dtype=int);inverse[selected]=np.arange(len(selected))
    terms=[]
    def spin(space,rows):
        return (space.twice_spin[rows] if hasattr(space,"twice_spin")
                else _twice_spin(space.spins[rows],SPECIES[space.sector]))
    for m,oi,allowed in inside.symmetry_groups:
        si=allowed[inverse[allowed]>=0]
        if not len(si):continue
        so=mapped[inverse[si]]
        change=(outside.jz_twice-inside.jz_twice)+spin(inside,si)-spin(outside,so)
        if np.any(change!=change[0]) or change[0]%2:
            raise ArithmeticError("ordered spin map has inconsistent angular transfer")
        mf=m+int(change[0])//2
        of=np.flatnonzero(out_lz==mf)
        forbidden=np.flatnonzero(out_lz!=mf)
        if np.linalg.norm(orbital[np.ix_(forbidden,oi)])*np.linalg.norm(color)*sqrt(len(si))>3e-12:
            raise ArithmeticError("canonical contact violates exact Iz/Jz")
        if not len(of):continue
        o=orbital[np.ix_(of,oi)]
        if not np.any(o):continue
        if len(np.unique(so))!=len(so):
            raise ArithmeticError("ordered spin map must be injective")
        terms.append(_TensorTerm(_rows(inside,li,oi,si),_rows(outside,lo,of,so),o,color))
    return tuple(terms)


def apply_terms(terms,values,dimension,*,adjoint=False):
    result=np.zeros((dimension,values.shape[1]),complex)
    for t in terms:
        result[t.incoming_rows if adjoint else t.outgoing_rows]+=(
            t.reverse(values) if adjoint else t.forward(values))
    return result


class FiveBodyContactAction:
    physical=False
    pair_current_square_included=False

    def __init__(self,spaces,*,b_GeV,_contact_kernel=None,_conversion_kernel=None):
        if set(spaces)!={"qqqgg","qqqq-qbar"}:
            raise ValueError("both exact five-body spaces required")
        example=spaces["qqqgg"];b=_width(b_GeV)
        for name,s in spaces.items():
            if s.sector!=name or any(getattr(s,k)!=getattr(example,k)
                                    for k in ("total_k2","cutoff","target","jz_twice")):
                raise ValueError("five-body contact coordinates disagree")
        self.spaces=dict(spaces)
        self.components={}
        for name,s in spaces.items():
            self.components[name+"_fermion_scattering"]=(name,self._number(s,b,"fermion",_contact_kernel))
        s=spaces["qqqgg"]
        for channel in ("qg","gg"):
            self.components[channel+"_number_scattering"]=("qqqgg",self._number(s,b,channel,_contact_kernel))
        self.components["IF_scattering"]=("qqqgg",self._instantaneous(s,b,_contact_kernel))
        self.components["A4"]=("qqqgg",self._quartic(s,b,_contact_kernel))
        self.conversion=self._conversion(b,_conversion_kernel)

    @staticmethod
    def _number(s,b,channel,_contact_kernel=None):
        species=SPECIES[s.sector];lookup={m:i for i,m in enumerate(s.longitudinal)}
        if channel=="fermion":
            pairs=list(combinations([i for i,x in enumerate(species) if x!="g"],2))
        else:
            pairs=[(i,j) for i,j in combinations(range(5),2)
                   if "".join((species[i],species[j]))==channel]
        terms=[];spins=np.arange(s.nspin)
        for li,incoming in enumerate(s.longitudinal):
            for i,j in pairs:
                color=(fermion_pair_color(s.sector,i,j) if channel=="fermion"
                       else current_pair_color(s.sector,i,j))
                for ki in range(2 if species[i]=="g" else 1,incoming[i]+incoming[j],2):
                    transfer=ki-incoming[i]
                    if transfer==0:continue  # exact Q0 removal, not a cutoff threshold
                    outgoing=list(incoming);outgoing[i]=ki;outgoing[j]=incoming[i]+incoming[j]-ki
                    outgoing=tuple(outgoing)
                    factor=2*s.total_k2/(pi*transfer**2)
                    for k in (i,j):
                        if species[k]=="g":
                            factor*=canonical_gluon_current_factor(outgoing[k],incoming[k])
                    orbital=((fermion_contact_block if channel=="fermion" else five_body_contact)
                             if _contact_kernel is None else _contact_kernel)(
                                 s.sector,incoming,outgoing,(i,j),b,s.cutoff)
                    terms.extend(contact_tensor_terms(s,s,li,lookup[outgoing],orbital,factor*color,spins,spins))
        return tuple(terms)

    @staticmethod
    def _instantaneous(s,b,_contact_kernel=None):
        lookup={m:i for i,m in enumerate(s.longitudinal)};terms=[]
        for li,incoming in enumerate(s.longitudinal):
            for q in range(3):
                for g in (3,4):
                    for kg in range(2,incoming[q]+incoming[g],2):
                        outgoing=list(incoming);outgoing[g]=kg;outgoing[q]=incoming[q]+incoming[g]-kg
                        outgoing=tuple(outgoing)
                        orbital=(five_body_contact if _contact_kernel is None else _contact_kernel)(s.sector,incoming,outgoing,(q,g),b,s.cutoff)
                        prefactor=s.total_k2/(2*pi*sqrt(incoming[g]*kg))
                        for hq,hg,reverse in product((-1,1),(-1,1),(False,True)):
                            selected=np.flatnonzero((2*(s.spins[:,q]%2)-1==hq)&(2*s.spins[:,g]-1==hg))
                            spin=transverse_spin_factor(hq,hg,crossed=reverse)
                            denominator=incoming[q]-kg if reverse else incoming[q]+incoming[g]
                            if denominator%2!=1:raise ArithmeticError("constrained denominator must be signed odd")
                            color=ordered_five_qg_color(q,g,reverse)*prefactor*spin/denominator
                            terms.extend(contact_tensor_terms(s,s,li,lookup[outgoing],orbital,color,selected,selected))
        return tuple(terms)

    @staticmethod
    def _quartic(s,b,_contact_kernel=None):
        lookup={m:i for i,m in enumerate(s.longitudinal)};terms=[]
        for li,incoming in enumerate(s.longitudinal):
            for k1 in range(2,sum(incoming[3:]),2):
                outgoing=incoming[:3]+(k1,sum(incoming[3:])-k1)
                orbital=(five_body_contact if _contact_kernel is None else _contact_kernel)(s.sector,incoming,outgoing,(3,4),b,s.cutoff)
                factor=s.total_k2/(16*pi*sqrt(np.prod(incoming[3:]+outgoing[3:])))
                for ho1,ho2,hi1,hi2 in product((-1,1),repeat=4):
                    color=factor*quartic_spin_color(ho1,ho2,hi1,hi2)
                    selected=np.flatnonzero((2*s.spins[:,3]-1==hi1)&(2*s.spins[:,4]-1==hi2))
                    changed=s.spins[selected].copy()
                    changed[:,3],changed[:,4]=(ho1+1)//2,(ho2+1)//2
                    mapped=np.ravel_multi_index(changed.T,s.spin_dims)
                    terms.extend(contact_tensor_terms(s,s,li,lookup[outgoing],orbital,color,selected,mapped))
        return tuple(terms)

    def _conversion(self,b,_conversion_kernel=None):
        inside,outside=self.spaces["qqqgg"],self.spaces["qqqq-qbar"]
        lookup={m:i for i,m in enumerate(outside.longitudinal)};terms=[]
        for li,incoming in enumerate(inside.longitudinal):
            for kq in range(1,sum(incoming[3:]),2):
                outgoing=incoming[:3]+(kq,sum(incoming[3:])-kq)
                orbital=(conversion_contact if _conversion_kernel is None else _conversion_kernel)(incoming,outgoing,b,inside.cutoff)
                for hq,ha,h1,h2 in product((-1,1),repeat=4):
                    color=instantaneous_conversion_kernel(incoming,outgoing,(hq,ha,h1,h2))
                    selected=np.flatnonzero((2*inside.spins[:,3]-1==h1)&(2*inside.spins[:,4]-1==h2))
                    for flavor in range(2):
                        changed=np.column_stack((inside.spins[selected,:3],
                            np.full(len(selected),2*flavor+(hq+1)//2),
                            np.full(len(selected),2*flavor+(ha+1)//2)))
                        mapped=np.ravel_multi_index(changed.T,outside.spin_dims)
                        terms.extend(contact_tensor_terms(inside,outside,li,lookup[outgoing],orbital,color,selected,mapped))
        return tuple(terms)

    def apply(self,columns):
        count=SixBodyCubicAction._validate(self.spaces,columns)
        result={n:np.zeros((s.dimension,count),complex) for n,s in self.spaces.items()}
        for name,terms in self.components.values():
            result[name]+=apply_terms(terms,columns[name],self.spaces[name].dimension)
        sea=self.spaces["qqqq-qbar"];gg=self.spaces["qqqgg"]
        result["qqqq-qbar"]+=sqrt(2)*sea.statistics_project(
            apply_terms(self.conversion,columns["qqqgg"],sea.dimension))
        result["qqqgg"]+=sqrt(2)*apply_terms(self.conversion,
            sea.statistics_project(columns["qqqq-qbar"]),gg.dimension,adjoint=True)
        return result
