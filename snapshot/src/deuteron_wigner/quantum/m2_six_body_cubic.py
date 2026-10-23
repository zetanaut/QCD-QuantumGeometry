"""Sparse canonical cubic Q5 -> Q6 action with normalized Fock statistics.

Forward and reverse bind the same exact packed coordinate objects. This
supplies U65, not the separate direct quartic W6P required for a complete
leading six-body excursion.
"""
from itertools import permutations,product
from math import sqrt
import numpy as np
from scipy import sparse

from .m2_six_body_space import SixBodySpace
from .m2_six_body_vertices import split_routes,color_source,cubic_orbital


def five_statistics_project(space,values):
    a=space.statistics_project(values)
    if space.sector=="qqqq-qbar":return a
    return sum((-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
               *space.permute(a,p+(3,4)) for p in permutations(range(3)))/6


def distinguished_matrix(incoming,outgoing,channel,*,b_GeV,mass_GeV):
    if incoming.cutoff>outgoing.cutoff:
        raise ValueError("outgoing shell must include incoming shell")
    rows=[];cols=[];data=[]
    ins={s:i for i,s in enumerate(incoming.longitudinal)}
    outs={s:i for i,s in enumerate(outgoing.longitudinal)}
    for route in split_routes(incoming.total_k2,incoming.sector,channel):
        if route.sector!=outgoing.sector:
            raise ValueError("channel and outgoing sector disagree")
        color=color_source(route)
        for h1,h2,h0 in product((-1,1),repeat=3):
            orbital=cubic_orbital(route,incoming.cutoff,outgoing.cutoff,b_GeV,mass_GeV,(h1,h2,h0))
            if not np.any(orbital):continue
            spins=incoming.spins
            parent_helicity=2*spins[:,route.parent]-1 if channel.startswith("g_to") else 2*(spins[:,route.parent]%2)-1
            selected=np.flatnonzero(parent_helicity==h0)
            changed=spins[selected].copy()
            if channel=="g_to_pair":
                # Old g3 -> outgoing g5, pair daughters -> q3,anti4.
                all_out=[np.column_stack((changed[:,:3],
                    np.full(len(selected),2*f+(h1+1)//2),
                    np.full(len(selected),2*f+(h2+1)//2),changed[:,3])) for f in range(2)]
            else:
                changed[:,route.parent]=((h1+1)//2 if channel=="g_to_gg" else
                    2*(changed[:,route.parent]//2)+(h1+1)//2)
                all_out=[np.column_stack((changed,np.full(len(selected),(h2+1)//2)))]
            for out_sf in all_out:
                out_spins=np.ravel_multi_index(out_sf.T,outgoing.spin_dims)
                for oo,oi in zip(*np.nonzero(orbital)):
                    for co,ci in zip(*np.nonzero(color)):
                        coeff=orbital[oo,oi]*color[co,ci]
                        raw_in=(((ins[route.incoming]*len(incoming.orbitals)+oi)
                            *incoming.ncolor+ci)*incoming.nspin+selected)
                        raw_out=(((outs[route.outgoing]*len(outgoing.orbitals)+oo)
                            *outgoing.ncolor+co)*outgoing.nspin+out_spins)
                        ii,jj=incoming.lookup[raw_in],outgoing.lookup[raw_out]
                        if np.any((ii>=0)&(jj<0)) and abs(coeff)>2e-13:
                            raise ArithmeticError("canonical cubic source violates exact Iz/Jz")
                        valid=(ii>=0)&(jj>=0)
                        rows.extend(jj[valid]);cols.extend(ii[valid]);data.extend([coeff]*np.count_nonzero(valid))
    return sparse.coo_matrix((data,(rows,cols)),shape=(outgoing.dimension,incoming.dimension),dtype=complex).tocsr()


class SixBodyCubicAction:
    physical=False
    direct_quartic_Q6P_supplied=False

    def __init__(self,spaces,*,out_cutoff,b_GeV,mass_GeV):
        if set(spaces)!={"qqqgg","qqqq-qbar"}:
            raise ValueError("both explicit five-body coordinate spaces required")
        self.incoming=dict(spaces)
        example=spaces["qqqgg"]
        for name,s in spaces.items():
            if s.sector!=name or any(getattr(s,k)!=getattr(example,k) for k in ("total_k2","cutoff","target","jz_twice")):
                raise ValueError("five-body source spaces disagree")
        b,m=float(b_GeV),float(mass_GeV)
        if not np.all(np.isfinite([b,m])) or b<=0 or m<0:
            raise ValueError("positive width and finite nonnegative mass required")
        self.outgoing={name:SixBodySpace(name,example.total_k2,out_cutoff,example.target,example.jz_twice)
                       for name in ("qqqggg","qqqq-qbar-g")}
        self.components={}
        routes=[("qqqgg","qqqggg","q_to_qg",sqrt(3)),
                ("qqqgg","qqqggg","g_to_gg",sqrt(3)),
                ("qqqgg","qqqq-qbar-g","g_to_pair",2*sqrt(2)),
                ("qqqq-qbar","qqqq-qbar-g","q_to_qg",1),
                ("qqqq-qbar","qqqq-qbar-g","anti_to_antig",1)]
        for source,target,channel,factor in routes:
            w=distinguished_matrix(spaces[source],self.outgoing[target],channel,b_GeV=b,mass_GeV=m)
            self.components[source+":"+channel]=(source,target,factor*w)

    @staticmethod
    def _validate(spaces,columns):
        if set(columns)!=set(spaces):
            raise ValueError("all declared sectors must be supplied")
        n=None
        for name,s in spaces.items():
            a=np.asarray(columns[name])
            if a.ndim!=2 or a.shape[0]!=s.dimension or not np.all(np.isfinite(a)):
                raise ValueError("finite columns in the exact packed coordinates required")
            if n is not None and a.shape[1]!=n:
                raise ValueError("common column count required")
            n=a.shape[1]
        return n

    def apply(self,columns):
        n=self._validate(self.incoming,columns)
        source={name:five_statistics_project(s,columns[name]) for name,s in self.incoming.items()}
        out={name:np.zeros((s.dimension,n),complex) for name,s in self.outgoing.items()}
        for _,(s,t,w) in self.components.items():out[t]+=w@source[s]
        return {name:s.statistics_project(out[name]) for name,s in self.outgoing.items()}

    def adjoint(self,columns):
        n=self._validate(self.outgoing,columns)
        source={name:s.statistics_project(columns[name]) for name,s in self.outgoing.items()}
        out={name:np.zeros((s.dimension,n),complex) for name,s in self.incoming.items()}
        for _,(s,t,w) in self.components.items():out[s]+=w.getH()@source[t]
        return {name:five_statistics_project(s,out[name]) for name,s in self.incoming.items()}
