"""Second-order path-dressed LF quark/gluon states, NOT a binding action.

A translated path acts on one distinguished quark color slot, with its
momentum recoiling against every created gluon. The other two quark slots
have different spin/flavor labels. The raw color tensor embeds isometrically
in exterior Fock space; the seed is not a physical p/n spin wavefunction.
Only the perturbative transverse Fock vacuum and positive-plus modes are
used. Local Gauss law, zero modes, a flux vacuum and GTMD process links are
NOT supplied. There is no second independent flux register.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations
from math import pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix

from .canonical_lf_fields import polarization
from .finite_lf_coupling import LFCell
from .modular_gauge_network import generators


@dataclass(frozen=True)
class DressingPath:
    """Oriented polyline r=(x-minus,x,y), GeV^-1, at x-plus=0.

    Vertices run from source to destination (unlike GaugeGraph.Path's
    transport-index ordering). No process-dependent staple semantics.
    """
    vertices: tuple

    def __post_init__(self):
        r=np.asarray(self.vertices,float)
        if r.ndim!=2 or r.shape[1]!=3 or len(r)<1 or not np.isfinite(r).all():
            raise ValueError('One or more finite LF spatial vertices required.')
        object.__setattr__(self,'vertices',tuple(map(tuple,r)))

    def reversed(self):
        return DressingPath(self.vertices[::-1])


def baryon_seed():
    """epsilon/sqrt(6) in three ordered, distinct spin/flavor quark slots."""
    v=np.zeros((3,3,3),complex)
    for p in permutations(range(3)):
        inversions=sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        v[p]=(-1)**inversions/sqrt(6)
    return v.ravel()


class PathCarrier:
    """Complete color/helicity and <=2 bosons of declared spatial modes.

    Not a continuum cutoff or dynamical Fock truncation. The parent quark
    must stay positive-plus for EVERY pair of retained gluons, including
    repeated modes, to qualify the order-two recoil image.
    """
    def __init__(self,cell,gluon_momenta,quarks,mass):
        self.cell=cell
        k=np.asarray([cell.mode(p,'boson') for p in gluon_momenta])
        p=np.asarray([cell.mode(q,'fermion') for q in quarks])
        if k.ndim!=2 or len(k)<1 or len({tuple(x) for x in k})!=len(k):
            raise ValueError('Distinct nonempty gluon spatial modes required.')
        if p.shape!=(3,3) or not np.isfinite(mass) or mass<0:
            raise ValueError('Three quark momenta and finite nonnegative mass required.')
        if p[0,0]<=2*np.max(k[:,0]):
            raise ValueError('Order-two recoil reaches nonpositive quark plus momentum.')
        self.quarks=p;self.mass=float(mass)
        self.momenta=np.repeat(k,2,axis=0)
        self.helicities=np.tile([-1,1],len(k))
        self.eps=np.array([polarization(*q,h)[2:] for q,h in zip(self.momenta,self.helicities)])
        self.factor=np.array([cell.field_factor(q,'boson') for q in self.momenta])
        self.labels=tuple((m,a) for m in range(len(self.momenta)) for a in range(8))
        self.pairs=tuple(combinations_with_replacement(range(len(self.labels)),2))
        self.configs=((),)+tuple((j,) for j in range(len(self.labels)))+self.pairs
        self.color=np.array([np.kron(t,np.eye(9)) for t in generators()])
        self.seed=baryon_seed()

    def segment(self,start,end,t):
        """Creation f_m(t)=-drT.eps*_m exp(+ik.r)/sqrt(2k+ V).

        D=partial-i gs t.A -> U=P exp(+i gs integral A_mu dr^mu).
        At fixed x+, A+=0 leaves the MINUS transverse contraction.
        t is a scalar or array in [0,1]; result has (..., mode,helicity).
        """
        start,end=np.asarray(start,float),np.asarray(end,float)
        d=end-start;kt=self.momenta*np.array([1,-1,-1])
        phase=np.asarray(t)[...,None]*(kt@d)+(kt@start)
        return -np.exp(1j*phase)*(self.eps.conj()@d[1:])*self.factor

    def integrals(self,path,order=24):
        """F_m, I_mn=int_late f_m int_early f_n, J_m=int f*_m int f_m.

        Inner straight-segment integrals are analytic; only the outer
        integral uses Gauss quadrature. All preceding segments are included.
        """
        if isinstance(order,bool) or int(order)!=order or order<2:
            raise ValueError('Integer quadrature order >=2 required.')
        nodes,weights=np.polynomial.legendre.leggauss(int(order))
        nodes=(nodes+1)/2;weights=weights/2
        n=len(self.momenta);f=np.zeros(n,complex);i=np.zeros((n,n),complex);j=np.zeros(n,complex)
        for a,b in zip(path.vertices[:-1],path.vertices[1:]):
            delta=np.array(b)-a
            omega=(self.momenta*np.array([1,-1,-1]))@delta
            base=self.segment(a,b,0.)
            phase=nodes[:,None]*omega
            inner=f+base*nodes[:,None]*np.exp(.5j*phase)*np.sinc(phase/(2*pi))
            late=self.segment(a,b,nodes)
            i+=np.einsum('s,sm,sn->mn',weights,late,inner)
            j+=np.einsum('s,sm,sm->m',weights,late.conj(),inner)
            f+=base*np.exp(.5j*omega)*np.sinc(omega/(2*pi))
        return f,i,j

    def wave(self,path,order=24):
        """Coefficient vectors of 1,gs,gs^2, shape (3,Fock_configs,27)."""
        f,i,j=self.integrals(path,order)
        wave=np.zeros((3,len(self.configs),27),complex);wave[0,0]=self.seed
        tv=self.color@self.seed
        wave[2,0]=-sum(j)*sum(t@t@self.seed for t in self.color)
        for n,(m,a) in enumerate(self.labels):
            wave[1,1+n]=1j*f[m]*tv[a]
        for n,(l,r) in enumerate(self.pairs,1+len(self.labels)):
            m,a=self.labels[l];p,b=self.labels[r]
            if l==r:
                wave[2,n]=-sqrt(2)*i[m,m]*self.color[a]@tv[a]
            else:
                wave[2,n]=-i[m,p]*self.color[a]@tv[b]-i[p,m]*self.color[b]@tv[a]
        return wave

    def recoil(self,config):
        k=sum((self.momenta[self.labels[j][0]] for j in config),np.zeros(3))
        return self.quarks[0]-k,k

    def free_energies(self):
        """Actual total P-minus including recoiled quark, not path lengths."""
        def e(p,m):return (m*m+p[1:]@p[1:])/(2*p[0])
        spectator=sum(e(p,self.mass) for p in self.quarks[1:])
        return np.array([spectator+e(self.recoil(c)[0],self.mass)
                         +sum(e(self.momenta[self.labels[j][0]],0.) for j in c)
                         for c in self.configs])

    def oscillator_creators(self):
        """Independent normalized-occupation sparse operators, 0..2 bosons."""
        index={c:i for i,c in enumerate(self.configs)};ops=[]
        for j in range(len(self.labels)):
            rows=[];cols=[];values=[]
            for n,c in enumerate(self.configs):
                if len(c)==2:continue
                rows.append(index[tuple(sorted((*c,j)))]);cols.append(n)
                values.append(sqrt(c.count(j)+1))
            ops.append(coo_matrix((values,(rows,cols)),shape=(len(index),len(index))).tocsr())
        return tuple(ops)

    def field_action(self,profile,state,creators=None):
        """Hermitian full color/Fock B(profile), with implicit exact recoil."""
        if creators is None:creators=self.oscillator_creators()
        result=np.zeros_like(state,dtype=complex)
        for n,(m,a) in enumerate(self.labels):
            field=profile[m]*(creators[n]@state)+profile[m].conjugate()*(creators[n].T@state)
            result+=field@self.color[a].T
        return result

    def ode_wave(self,path,rtol=2e-10,atol=2e-12):
        """Independent physical-basis Dyson jets; no ordered-integral input."""
        creators=self.oscillator_creators();shape=(2,len(self.configs),27)
        zero=np.zeros(shape[1:],complex);zero[0]=self.seed
        y=np.zeros(shape,complex);evaluations=0
        for start,end in zip(path.vertices[:-1],path.vertices[1:]):
            start=np.array(start);d=np.array(end)-start
            def rhs(s,v):
                # Build the actual field from separate annihilation/creation
                # plane waves, independently of segment()/integrals().
                x=start+s*d
                phase=self.momenta[:,0]*x[0]-self.momenta[:,1:]@x[1:]
                f=-np.einsum('mi,i->m',self.eps.conj(),d[1:])*np.exp(1j*phase)/np.sqrt(
                    2*self.momenta[:,0]*self.cell.volume)
                first,second=v.reshape(shape)
                return np.stack((1j*self.field_action(f,zero,creators),
                                 1j*self.field_action(f,first,creators))).ravel()
            sol=solve_ivp(rhs,(0,1),y.ravel(),method='DOP853',rtol=rtol,atol=atol)
            if not sol.success:raise RuntimeError(sol.message)
            y=sol.y[:,-1].reshape(shape);evaluations+=sol.nfev
        return np.concatenate((zero[None],y)),evaluations


def pullback(left,right,operator=None):
    """Common coefficient matrices through gs^2; no independent path energies.

    operator acts on (Fock_configs,27); None denotes identity.
    """
    if operator is None:operator=lambda x:x
    return np.array([sum(np.vdot(left[j],operator(right[n-j])) for j in range(n+1))
                     for n in range(3)])


def reference():
    cell=LFCell(tuple(2*pi/np.array([.25,.2,.2])))
    carrier=PathCarrier(cell,[(.25,.2,0),(.5,0,.2)],
                        [(1.875,.2,-.2),(.375,0,.2),(.625,-.2,0)],.27)
    paths={
        'bare':DressingPath(((0,0,0),)),
        'transverse_loop':DressingPath(((0,0,0),(0,4,0),(0,4,3),(0,0,3),(0,0,0))),
        'LF_detour':DressingPath(((0,0,0),(2,4,0),(3,4,3),(1,0,3),(0,0,0))),
        'different_loop':DressingPath(((0,0,0),(1,3,1),(2,2,5),(0,0,0))),
    }
    return carrier,paths
