"""Matrix-free canonical pair-current square on complete Q5 shells.

The auxiliary color-adjoint slot is a spin-zero CURRENT coordinate, not a
physical gluon. Its exact I=1/2 and quark-sign projector is retained as an
action. E P_aux D P_aux E† replaces E U (U† D U) U† E† using U U†=P_aux
and [P_aux,D]=0, not by dropping an auxiliary symmetry restriction.
"""
from math import pi
import numpy as np

from .m2_coupled_longitudinal_jacobi import (
    S3,longitudinal_basis,oscillator_shell,oscillator_rotation,permutation_jacobi_map,
)
from .m2_c117_pair_current import color_permutation_matrix
from .m2_spin_flavor_s3_completion import (
    SPIN_FLAVOR_BASIS,build_spin_flavor_s3_algebra,spin_flavor_permutation_representation,
)
from .m2_excluded_fock_kinematics import split_routes,color_source
from .m2_excluded_pair_current import pair_current_orbital,adjoint_pair_current_factor
from .m2_instantaneous_sea_exit import sea_pair_spin_factor
from .m2_five_body_factorized import contact_tensor_terms,apply_terms
from .m2_six_body_cubic import SixBodyCubicAction
from .m2_coupled_pair_action import _width


class AuxiliaryCurrentSpace:
    def __init__(self,total_k2,cutoff,target,jz_twice):
        if target not in ("PROTON","NEUTRON") or jz_twice not in (-1,1) or isinstance(jz_twice,(bool,np.bool_)):
            raise ValueError("explicit target and doubled half-integer Jz required")
        self.sector="qqq-adjoint-current"
        self.total_k2,self.cutoff,self.target,self.jz_twice=total_k2,cutoff,target,jz_twice
        self.longitudinal=longitudinal_basis("qqqg",total_k2)
        self.orbitals=oscillator_shell(3,cutoff)
        self.spins=np.asarray(SPIN_FLAVOR_BASIS)
        self.nspin,self.ncolor=64,2
        self.twice_spin=(2*(self.spins%2)-1).sum(axis=1)
        iz=np.where(self.spins<2,1,-1).sum(axis=1)
        lz=np.array([sum(o[:3])-sum(o[3:]) for o in self.orbitals])
        allowed=((2*lz[:,None]+self.twice_spin[None,:]==jz_twice)
                 &(iz[None,:]==(1 if target=="PROTON" else -1)))
        shape=(len(self.longitudinal),len(self.orbitals),2,64)
        self.rows=np.flatnonzero(np.broadcast_to(allowed[None,:,None,:],shape).ravel())
        self.dimension=len(self.rows)
        self.lookup=np.full(np.prod(shape),-1,dtype=int);self.lookup[self.rows]=np.arange(self.dimension)
        self.coordinates=np.array(np.unravel_index(self.rows,shape)).T
        self.symmetry_groups=[];self.group_rows={}
        for m in np.unique(lz):
            orbitals=np.flatnonzero(lz==m);spins=np.flatnonzero(allowed[orbitals[0]])
            if not len(spins):continue
            self.symmetry_groups.append((int(m),orbitals,spins))
            for li in range(len(self.longitudinal)):
                rows=np.flatnonzero((self.coordinates[:,0]==li)&np.isin(self.coordinates[:,1],orbitals))
                if len(rows)!=len(orbitals)*2*len(spins):raise ArithmeticError("incomplete auxiliary block")
                self.group_rows[li,int(m)]=rows
        self.isospin_projector=(3.75*np.eye(64)-build_spin_flavor_s3_algebra().isospin_squared)/3
        self.weight=np.array([2*total_k2/(pi*m[3]**2) for m in self.longitudinal])[self.coordinates[:,0]]

    def _columns(self,values):
        a=np.asarray(values,complex)
        if a.ndim!=2 or a.shape[0]!=self.dimension or not np.all(np.isfinite(a)):
            raise ValueError("finite packed auxiliary columns required")
        return a

    def permute(self,values,p):
        a=self._columns(values);c=color_permutation_matrix("qqqg",p)
        sf=spin_flavor_permutation_representation(p)
        lookup={state:i for i,state in enumerate(self.longitudinal)}
        out=np.zeros_like(a)
        for li,state in enumerate(self.longitudinal):
            target,r=permutation_jacobi_map(state,p);o=oscillator_rotation(r,self.cutoff)
            for m,orbitals,spins in self.symmetry_groups:
                block=a[self.group_rows[li,m]].reshape(len(orbitals),2,len(spins),a.shape[1])
                value=np.einsum("oi,ab,st,ibtk->oask",o[np.ix_(orbitals,orbitals)],c,
                                sf[np.ix_(spins,spins)],block,optimize=True)
                rows=self.group_rows[lookup[target],m]
                out[rows]=value.reshape(len(rows),a.shape[1])
        return out

    def project(self,values):
        a=self._columns(values);iso=np.zeros_like(a)
        for li in range(len(self.longitudinal)):
            for m,orbitals,spins in self.symmetry_groups:
                rows=self.group_rows[li,m];block=a[rows].reshape(len(orbitals),2,len(spins),a.shape[1])
                iso[rows]=np.einsum("st,octk->ocsk",self.isospin_projector[np.ix_(spins,spins)],
                                   block,optimize=True).reshape(len(rows),a.shape[1])
        return sum((-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
                   *self.permute(iso,p) for p in S3)/6


class FactorizedPairCurrentAction:
    physical=False

    def __init__(self,spaces,*,b_GeV,_orbital_kernel=None):
        if set(spaces)!={"qqqgg","qqqq-qbar"}:
            raise ValueError("both explicit five-body pair-current spaces required")
        example=spaces["qqqgg"];b=_width(b_GeV)
        for name,s in spaces.items():
            if s.sector!=name or any(getattr(s,k)!=getattr(example,k)
                                    for k in ("total_k2","cutoff","target","jz_twice")):
                raise ValueError("pair-current coordinates disagree")
        self.spaces=dict(spaces)
        a=self.auxiliary=AuxiliaryCurrentSpace(example.total_k2,example.cutoff,example.target,example.jz_twice)
        ilookup={m:i for i,m in enumerate(a.longitudinal)}
        self.terms={}
        for name,s in spaces.items():
            channel="g_to_gg" if name=="qqqgg" else "g_to_pair"
            color=color_source(channel);lookup={m:i for i,m in enumerate(s.longitudinal)}
            terms=[]
            for route in split_routes(example.total_k2,channel):
                orbital=(pair_current_orbital if _orbital_kernel is None else _orbital_kernel)(route,example.cutoff,b)
                for h in (-1,1):
                    if name=="qqqgg":
                        factor=adjoint_pair_current_factor(*route.outgoing[3:],h,-h)
                        mapped=(np.arange(64)*2+(h+1)//2)*2+(-h+1)//2
                        terms.extend(contact_tensor_terms(a,s,ilookup[route.incoming],lookup[route.outgoing],
                                                          orbital,factor*color,np.arange(64),mapped))
                    else:
                        for flavor in range(2):
                            q,anti=2*flavor+(h+1)//2,2*flavor+(-h+1)//2
                            mapped=(np.arange(64)*4+q)*4+anti
                            terms.extend(contact_tensor_terms(a,s,ilookup[route.incoming],lookup[route.outgoing],
                                orbital,sea_pair_spin_factor(h,-h)*color,np.arange(64),mapped))
            self.terms[name]=tuple(terms)

    def apply(self,columns):
        count=SixBodyCubicAction._validate(self.spaces,columns)
        a=self.auxiliary
        contracted=np.zeros((a.dimension,count),complex)
        for name,s in self.spaces.items():
            factor=2 if name=="qqqq-qbar" else 1
            contracted+=factor*apply_terms(self.terms[name],s.statistics_project(columns[name]),
                                          a.dimension,adjoint=True)
        weighted=a.weight[:,None]*a.project(contracted)
        # D commutes with the exact auxiliary projector; one P is sufficient
        # between E and E†. No reduced eigenspace or numerical rank is chosen.
        return {name:(2 if name=="qqqq-qbar" else 1)*s.statistics_project(
            apply_terms(self.terms[name],weighted,s.dimension)) for name,s in self.spaces.items()}
