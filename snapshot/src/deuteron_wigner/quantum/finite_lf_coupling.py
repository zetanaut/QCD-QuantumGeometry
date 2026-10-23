"""Normalized canonical LF couplings and a first-Yukawa-order Fock experiment.

The cell is a declared numerical reference, not confinement, a matching scale,
or the old BLFQ/H2 model. Canonical QCD leaves are unchanged. No old loop is
reused as a bare interaction. Only the coefficient of ONE Yukawa insertion
is evolved; its norm squared is a leading transition coefficient, not a
unitarily normalized all-order state or a physical meson decay prediction.
"""
from dataclasses import dataclass
from itertools import product
from math import pi, sqrt

import numpy as np
import torch
from scipy.integrate import quad_vec

from .canonical_lf_fields import GAMMA, LAMBDA_PLUS, lf_spinor, antiquark_spinor
from .canonical_lf_network import cubic_vertex_tensor, momentum
from .collective_quark_matching import bilinear_matrices


@dataclass(frozen=True)
class LFCell:
    """Lengths (x-minus,x,y), GeV^-1; AP longitudinal fermions, periodic bosons.

    Transverse boundary conditions are periodic for both. Positive-plus modes
    only; no prescription for the constrained zero mode is selected here.
    """
    lengths: tuple

    def __post_init__(self):
        x = np.asarray(self.lengths, float)
        if x.shape != (3,) or not np.isfinite(x).all() or np.any(x <= 0):
            raise ValueError('Three finite positive cell lengths required.')
        object.__setattr__(self, 'lengths', tuple(x))

    @property
    def volume(self):
        return float(np.prod(self.lengths))

    @property
    def spacing(self):
        return 2*pi/np.asarray(self.lengths)

    def mode(self, p, statistics):
        p = momentum(p)
        if statistics not in ('fermion', 'boson'):
            raise ValueError('Explicit fermion/boson statistics required.')
        index = p/self.spacing
        index[0] -= .5 if statistics == 'fermion' else 0
        if not np.allclose(index, np.rint(index), rtol=0, atol=1e-10):
            raise ValueError('Momentum is not on the declared boundary-condition grid.')
        return p

    def phase_integral(self, mismatch):
        """Exact integral exp(i*(dp+ x- - dpT.xT)) on the centered cell."""
        d = np.asarray(mismatch, float)
        if d.shape != (3,) or not np.isfinite(d).all():
            raise ValueError('Finite three-momentum mismatch required.')
        return self.volume*float(np.prod(np.sinc(d*np.asarray(self.lengths)/(2*pi))))

    def field_factor(self, p, statistics):
        return 1/sqrt(2*self.mode(p, statistics)[0]*self.volume)

    def cubic(self, channel, left, right, parent, mass, coupling):
        stats = {'q_to_qg': ('fermion','boson','fermion'),
                 'qbar_to_qbarg': ('fermion','boson','fermion'),
                 'g_to_pair': ('fermion','fermion','boson'),
                 'g_to_gg': ('boson','boson','boson')}
        if channel not in stats or not np.isfinite(coupling):
            raise ValueError('Known cubic channel and finite coupling required.')
        for p, s in zip((left,right,parent), stats[channel]):
            self.mode(p, s)
        return coupling*cubic_vertex_tensor(channel,left,right,parent,mass)/sqrt(self.volume)


def spinors(p, mass, *, anti=False):
    """Dirac/flavor/color by helicity/flavor/color, 24 by 12."""
    fn = antiquark_spinor if anti else lf_spinor
    return np.kron(np.column_stack([fn(*p,mass,h) for h in (-1,1)]),np.eye(6))


def collective_pair_leaf(left, right, parent, mass):
    """+qbar (sigma+i gamma5 tau.pi) q: (q12,anti12,O4), GeV^-1/2.

    All flavors, colors, physical antiparticle helicities and complex phases
    survive. Multiplet Yukawa couplings and finite-cell factor are separate.
    This comes from the preserved reference action, NOT canonical QCD itself.
    """
    a,b,p = map(momentum,(left,right,parent))
    if not np.allclose(a+b,p,rtol=0,atol=1e-12):
        raise ValueError('Conserved plus/transverse momentum required.')
    u,v = spinors(a,mass),spinors(b,mass,anti=True)
    bar = u.conj().T@np.kron(GAMMA[0],np.eye(6))
    return np.einsum('ij,Ajk,kl->ilA',bar,bilinear_matrices(),v)/sqrt(8*a[0]*b[0]*p[0])


def fermion_good_mode_metric(p, mass):
    """sqrt(2) u+^dagger u+/(2p+) = identity in physical helicity space."""
    u = np.column_stack([lf_spinor(*p,mass,h) for h in (-1,1)])
    return sqrt(2)*u.conj().T@LAMBDA_PLUS@u/(2*p[0])


def pair_routes(cell, parent, transverse_shell):
    """All AP q/anti partitions at fixed P, within a square transverse domain.

    Both species' transverse indices must lie in [-shell,shell]^2. Quark and
    antiquark are distinguishable: reversed momentum assignments are distinct,
    not divided by two. This enumerates a finite domain, not the continuum.
    """
    p=cell.mode(parent,'boson')
    if (isinstance(transverse_shell,bool) or int(transverse_shell)!=transverse_shell
            or transverse_shell<0):
        raise ValueError('Nonnegative integer transverse shell required.')
    k=int(round(p[0]/cell.spacing[0])); n=int(transverse_shell)
    routes=[]
    for j,ix,iy in product(range(k),range(-n,n+1),range(-n,n+1)):
        a=cell.spacing*np.array([j+.5,ix,iy]);b=p-a
        if np.any(abs(b[1:]/cell.spacing[1:])>n+1e-10):
            continue
        cell.mode(b,'fermion')
        routes.append((a,b))
    return tuple(routes)


@dataclass(frozen=True)
class PairReference:
    """Two scalar/isovector nodes, one parent, all declared q/anti routes."""
    cell: LFCell
    parent: tuple
    quark_mass: float
    transverse_shell: int

    def __post_init__(self):
        self.cell.mode(self.parent,'boson')
        if not np.isfinite(self.quark_mass) or self.quark_mass<0:
            raise ValueError('Finite nonnegative reference quark mass required.')
        pair_routes(self.cell,self.parent,self.transverse_shell)

    def arrays(self):
        """Return route energies and normalized full pair couplers, R x 144 x 4."""
        energy=[];leaves=[]
        for a,b in pair_routes(self.cell,self.parent,self.transverse_shell):
            energy.append(sum((self.quark_mass**2+x[1:]@x[1:])/(2*x[0]) for x in (a,b)))
            leaves.append(collective_pair_leaf(a,b,self.parent,self.quark_mass).reshape(144,4)
                          /sqrt(self.cell.volume))
        if not leaves:
            raise ValueError('The declared transverse domain contains no pair routes.')
        return np.array(energy),np.array(leaves)


def _tensor(value):
    if isinstance(value,torch.Tensor):
        return value.to(dtype=torch.complex128)
    return torch.tensor(np.array(value,copy=True),dtype=torch.complex128)


def _parameters(mass_squared,couplings,initial):
    """M2[A,node,node]; equal pion blocks preserve vector isospin in this test."""
    m,g,v=map(_tensor,(mass_squared,couplings,initial))
    if m.shape!=(4,2,2) or g.shape!=(2,) or v.shape!=(2,4):
        raise ValueError('Four 2x2 mass-squared blocks, two couplings, 2x4 seed required.')
    if not all(torch.isfinite(x).all() for x in (m,g,v)):
        raise ValueError('Finite input required.')
    if (not torch.allclose(m,m.mH,rtol=0,atol=1e-12)
            or not torch.allclose(m.imag,torch.zeros_like(m.imag),rtol=0,atol=1e-12)
            or not torch.allclose(g.imag,torch.zeros_like(g.imag),rtol=0,atol=1e-12)
            or not torch.allclose(m[1:],m[1].expand(3,-1,-1),rtol=0,atol=1e-12)
            or torch.any(torch.linalg.eigvalsh(m)<=0)):
        raise ValueError('Positive real symmetric masses, degenerate isovectors and real Yukawas required.')
    return m,g,v


def pulse_integral(delta,duration):
    """Integral_0^T sin^4(pi*t/T) exp(-i*delta*t) dt, stable at resonances."""
    result=torch.zeros_like(delta,dtype=torch.complex128)
    for k,c in ((0,3/8),(1,-1/4),(-1,-1/4),(2,1/16),(-2,1/16)):
        x=(delta-2*pi*k/duration)*duration
        result=result+c*duration*torch.exp(-.5j*x)*torch.sinc(x/(2*pi))
    return result


def first_order_wave(reference,mass_squared,couplings,initial,duration):
    """Zeroth parent wave and coefficient of lambda in full q/anti Fock wave.

    H_int(x+)=lambda*sin^4(pi*x+/T)*V_Y. The smoothly switched interaction
    is a numerical test pulse, not a fitted physical time dependence. Its
    first-order UV convergence does not renormalize the second-order phase.
    The differentiable source is a perturbation of the shared M2 or Yukawas,
    NOT an electromagnetic current or a source-dependent fitted output head.
    """
    if not np.isfinite(duration) or duration<=0:
        raise ValueError('Finite positive LF duration required.')
    m,g,v=_parameters(mass_squared,couplings,initial)
    energy,leaf=reference.arrays()
    e=_tensor(energy).real;l=_tensor(leaf)
    parent=[];weights=[]
    for a in range(4):
        w,u=torch.linalg.eigh(m[a]/(2*reference.parent[0]))
        if torch.min(abs(w[1:]-w[:-1]))<1e-10:
            raise ValueError('This eigenderivative control requires distinct node eigenvalues.')
        c=u.mH@v[:,a]
        parent.append(u@(torch.exp(-1j*w*duration)*c))
        f=pulse_integral(w[None,:]-e[:,None],duration)
        weights.append(-1j*torch.exp(-1j*e*duration)*(f@((g@u)*c)))
    pair=torch.einsum('rjA,rA->rj',l,torch.stack(weights,dim=-1))
    return torch.stack(parent,dim=-1),pair


def direct_time_wave(reference,mass_squared,couplings,initial,duration):
    """Independent adaptive time integral; matrix exponential, no spectral filter."""
    from scipy.linalg import expm
    _parameters(mass_squared,couplings,initial)
    m,g,v=map(np.asarray,(mass_squared,couplings,initial))
    e,l=reference.arrays()
    h=m/(2*reference.parent[0])
    def integrand(t):
        amplitudes=np.array([g@expm(-1j*h[a]*t)@v[:,a] for a in range(4)])
        return -1j*np.sin(pi*t/duration)**4*np.exp(-1j*e[:,None]*(duration-t))*amplitudes
    weights,error=quad_vec(integrand,0,duration,epsabs=2e-12,epsrel=2e-12)
    return np.einsum('rjA,rA->rj',l,weights),float(error)


def reference_inputs():
    """Explicit illustrative GeV inputs, not calibrated or inherited loop fits."""
    cell=LFCell(tuple(2*pi/np.array([.25,.20,.20])))
    sigma=np.array([[.65**2,.027],[.027,.91**2]])
    pion=np.array([[.35**2,-.018],[-.018,.55**2]])
    m=np.array([sigma,pion,pion,pion])
    g=np.array([.31,-.19])
    # A physical benchmark stays in ONE electric-charge block (Q=0 here).
    # Charged pi1 +/- i*pi2 seeds are tested separately, not coherently mixed.
    v=np.array([[1,0,0,-.4],[-.2j,0,0,.2]])
    v=v/np.linalg.norm(v)
    return cell,m,g,v
