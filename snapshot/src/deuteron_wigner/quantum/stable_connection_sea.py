"""Constructive stability/sea controls, not a physical QCD vacuum regulator.

Full SU(3) links and Dirac/flavor matrices on an algebraic eight-node carrier.
Homogeneous O(4) chiral fields permit an exact positive two-flavor determinant.
All numbers are dimensionless reference-cell quantities, not physical fits.
No continuum/anomaly, confinement, LF reconstruction or baryon claim is made.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, expm, lu_factor

from .canonical_lf_fields import GAMMA
from .chiral_soft_matching import G5
from .gluon_surface_consistency import cube_graph
from .modular_gauge_network import GaugeGraph, Path, generators


EGAMMA = (GAMMA[0],) + tuple(1j*g for g in GAMMA[1:])
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
VERTICES = np.array(((0,0,0),(1,0,0),(1,1,0),(0,1,0),
                     (0,0,1),(1,0,1),(1,1,1),(0,1,1)))
FACES = tuple(Path(p) for p in ((0,1,2,3,0),(4,5,6,7,4),
              (0,1,5,4,0),(1,2,6,5,1),(2,3,7,6,2),(3,0,4,7,3)))


def _finite(value, name, positive=False, nonnegative=False):
    if (not np.isfinite(value) or (positive and value <= 0)
            or (nonnegative and value < 0)):
        raise ValueError(f'Invalid finite {name}.')


@dataclass(frozen=True)
class BoundedKineticGate:
    """kappa=exp(eta [rho/(L²+rho)-rho_ref/(L²+rho_ref)]).

    Strictly positive upper/lower bounds; the reference kinetic term is one.
    This multiplies full F[A]^2, NOT an independent flux-density coordinate.
    """
    eta: float = 1.
    scale: float = 1.
    rho_reference: float = .5

    def __post_init__(self):
        _finite(self.eta, 'eta')
        _finite(self.scale, 'scale', positive=True)
        _finite(self.rho_reference, 'reference invariant', nonnegative=True)
        # Explicit numerical domain, not an unlimited floating-point promise.
        if abs(self.eta) > 50 or not 1e-8 <= self.scale <= 1e8:
            raise ValueError('Reference numerical domain exceeded.')

    def bounds(self):
        z0 = self.rho_reference/(self.scale**2+self.rho_reference)
        return tuple(sorted(np.exp(self.eta*np.array([-z0, 1-z0]))))

    def jet(self, rho):
        """Value, first and second derivatives with respect to rho."""
        _finite(rho, 'chiral invariant', nonnegative=True)
        l2 = self.scale**2
        z0 = self.rho_reference/(l2+self.rho_reference)
        k = np.exp(self.eta*(rho/(l2+rho)-z0))
        dz, ddz = l2/(l2+rho)**2, -2*l2/(l2+rho)**3
        return np.array([k, k*self.eta*dz, k*((self.eta*dz)**2+self.eta*ddz)])

    def energy(self, rho, electric_momentum, magnetic_field):
        """Canonical electric variable is Pi=kappa E; all color/spatial axes."""
        p, b = np.asarray(electric_momentum), np.asarray(magnetic_field)
        if not np.isfinite(p).all() or not np.isfinite(b).all():
            raise ValueError('Finite canonical fields required.')
        k = self.jet(rho)[0]
        return float(np.vdot(p, p).real/(2*k)+k*np.vdot(b, b).real/2)


def connection_profile(theta, seed=17, edge=(0,1), generator=0):
    """One periodic direction in a noncommuting connection, not a vacuum solve."""
    _finite(theta, 'connection angle')
    if generator not in range(6):
        raise ValueError('Use a 4pi-periodic off-diagonal generator for this profile.')
    graph = cube_graph(seed)
    if edge not in graph.links:
        raise ValueError('Oriented cube edge required.')
    links = dict(graph.links)
    links[edge] = expm(1j*theta*generators()[generator])@links[edge]
    return GaugeGraph(links)


def dirac_jet(graph, edge=(0,1), generator=0, hopping=.4):
    """D0, d_theta D0, d²_theta D0; node-spin-color order, U_ab:b->a.

    D_ab=h gamma_axis U_ab, D_ba=-D_ab^dagger. Exact anticommutation
    with gamma5 is an algebraic control only (NOT anomaly-correct fermions).
    """
    _finite(hopping, 'hopping', positive=True)
    if tuple(graph.nodes) != tuple(range(8)) or edge not in graph.links:
        raise ValueError('Full eight-node cube carrier and stored source edge required.')
    if generator not in range(8):
        raise ValueError('SU(3) generator index required.')
    out = [np.zeros((96,96), complex) for _ in range(3)]
    t = generators()[generator]
    for (a,b), u in graph.links.items():
        delta = np.flatnonzero(VERTICES[a] != VERTICES[b])
        if len(delta) != 1:
            raise ValueError('Cube nearest-neighbor edges required.')
        jets = (u, 1j*t@u, -t@t@u) if (a,b) == edge else (u, 0*u, 0*u)
        for matrix, uj in zip(out, jets):
            block = hopping*np.kron(EGAMMA[1+delta[0]], uj)
            matrix[12*a:12*(a+1),12*b:12*(b+1)] = block
            matrix[12*b:12*(b+1),12*a:12*(a+1)] = -block.conj().T
    return tuple(out)


def face_action_jet(graph, edge=(0,1), generator=0):
    """Sum(1-Re tr W/3) and exact connection derivatives, with all links kept."""
    t = generators()[generator]
    total = np.zeros(3)
    for path in FACES:
        w, dw, ddw = np.eye(3, dtype=complex), np.zeros((3,3),complex), np.zeros((3,3),complex)
        for a,b in zip(path.vertices[:-1], path.vertices[1:]):
            u = graph.link(a,b)
            if (a,b) == edge:
                du, ddu = 1j*t@u, -t@t@u
            elif (b,a) == edge:
                du, ddu = -1j*u@t, -u@t@t
            else:
                du, ddu = 0*u, 0*u
            w, dw, ddw = w@u, dw@u+w@du, ddw@u+2*dw@du+w@ddu
        total += [1-np.trace(w).real/3, -np.trace(dw).real/3, -np.trace(ddw).real/3]
    return total


def chiral_matrix(d0, phi, mass=0., yukawa=1.):
    """Explicit full two-flavor D, homogeneous phi=(sigma,pi1,pi2,pi3)."""
    phi = np.asarray(phi, float)
    if phi.shape != (4,) or not np.isfinite(phi).all():
        raise ValueError('Finite homogeneous O(4) multiplet required.')
    _finite(mass, 'current mass')
    _finite(yukawa, 'Yukawa coupling', positive=True)
    nodes = d0.shape[0]//12
    if d0.shape != (12*nodes,12*nodes) or not np.isfinite(d0).all():
        raise ValueError('Finite node-spin-color Dirac matrix required.')
    g5 = np.kron(np.eye(nodes),np.kron(G5,np.eye(3)))
    return (np.kron(d0+(mass+yukawa*phi[0])*np.eye(len(d0)),np.eye(2))
            +1j*yukawa*np.kron(g5,np.einsum('a,aij->ij',phi[1:],PAULI)))


def full_logdet(matrix):
    """Independent complex LU determinant: log modulus and unit phase."""
    lu, piv = lu_factor(matrix)
    diagonal = np.diag(lu)
    if np.any(np.abs(diagonal) == 0):
        raise ValueError('Singular Dirac matrix.')
    logabs = float(np.log(np.abs(diagonal)).sum())
    phase = np.exp(1j*(np.angle(diagonal).sum()+np.pi*np.count_nonzero(piv != np.arange(len(piv)))))
    return logabs, complex(phase)


def sea_kernel(d0, phi, mass=0., yukawa=1.):
    """D^dagger D without the duplicated flavor identity; det D_Nf2=det K.

    Valid for homogeneous multiplets, anti-Hermitian chirally anticommuting
    D0 and zero chemical potential. Never used for arbitrary pion textures.
    """
    phi = np.asarray(phi, float)
    if phi.shape != (4,) or not np.isfinite(phi).all():
        raise ValueError('Finite homogeneous O(4) multiplet required.')
    _finite(mass, 'current mass')
    _finite(yukawa, 'Yukawa coupling', positive=True)
    nodes = len(d0)//12
    g5 = np.kron(np.eye(nodes),np.kron(G5,np.eye(3)))
    if (d0.shape != g5.shape or not np.isfinite(d0).all()
            or np.linalg.norm(d0+d0.conj().T) > 1e-10
            or np.linalg.norm(d0@g5+g5@d0) > 1e-10):
        raise ValueError('Anti-Hermitian chirally paired Dirac carrier required.')
    r2 = (mass+yukawa*phi[0])**2+yukawa**2*np.dot(phi[1:],phi[1:])
    return d0.conj().T@d0+r2*np.eye(len(d0))


def negative_logdet_jet(kernel, first=(), second=None, weight=1.):
    """-w log det K, its linked sources and Hessian, including K_st contact.

    Weight is a declared cell normalization, NOT an extra flavor factor.
    Nf=2 is already included in det K. Sea loops must be counted once.
    """
    _finite(weight, 'sea weight', nonnegative=True)
    if not np.isfinite(kernel).all() or np.linalg.norm(kernel-kernel.conj().T) > 1e-10:
        raise ValueError('Finite Hermitian positive kernel required.')
    factor = cho_factor(kernel, lower=True)
    value = -2*weight*np.log(np.diag(factor[0]).real).sum()
    inverse = cho_solve(factor, np.eye(len(kernel)))
    n = len(first)
    if second is None:
        second = [[np.zeros_like(kernel) for _ in range(n)] for _ in range(n)]
    products = [inverse@d for d in first]
    gradient = np.array([-weight*np.trace(p).real for p in products])
    hessian = np.array([[weight*(np.trace(products[i]@products[j])
                        -np.trace(inverse@second[i][j])).real
                        for j in range(n)] for i in range(n)]).reshape(n,n)
    return float(value), gradient, hessian


@dataclass(frozen=True)
class StableConnectionSea:
    """Reference shared-action slice; no fitting, no extra PV sea determinant.

    S=beta*kappa(rho)*sum_faces(1-Re tr W/3)+8 U(rho)-w logdet K.
    U=a rho+lambda rho²/2, lambda>0; a may have either sign. On compact
    links the logarithmic sea cannot beat quartic coercivity at large |phi|.
    The graph face term is only a gauge-invariant source control for F².
    """
    gate: BoundedKineticGate = BoundedKineticGate()
    a: float = .25
    quartic: float = .6
    beta: float = .7
    sea_weight: float = .04
    mass: float = .06
    yukawa: float = 1.
    hopping: float = .4
    seed: int = 17

    def __post_init__(self):
        _finite(self.a, 'quadratic coefficient')
        for name in ('quartic','beta','yukawa','hopping'):
            _finite(getattr(self,name), name, positive=True)
        _finite(self.sea_weight, 'sea weight', nonnegative=True)
        _finite(self.mass, 'current mass', nonnegative=True)

    def jet(self, sigma, theta):
        """Shared value, gradient and Hessian in (sigma, one link angle)."""
        _finite(sigma, 'sigma')
        graph = connection_profile(theta, self.seed)
        d, dt, dtt = dirac_jet(graph, hopping=self.hopping)
        b, bt, btt = face_action_jet(graph)
        rho = sigma*sigma/2
        k, kr, krr = self.gate.jet(rho)
        ks, kss = kr*sigma, krr*sigma*sigma+kr
        potential = self.a*rho+self.quartic*rho*rho/2
        gradient = np.array([8*sigma*(self.a+self.quartic*rho)+self.beta*ks*b,
                             self.beta*k*bt])
        hessian = np.array([[8*(self.a+3*self.quartic*rho)+self.beta*kss*b,self.beta*ks*bt],
                            [self.beta*ks*bt,self.beta*k*btt]])
        kernel = sea_kernel(d,[sigma,0,0,0],self.mass,self.yukawa)
        eye, zero = np.eye(len(d)), np.zeros_like(d)
        km = 2*self.yukawa*(self.mass+self.yukawa*sigma)*eye
        kt = dt.conj().T@d+d.conj().T@dt
        ktt = dtt.conj().T@d+2*dt.conj().T@dt+d.conj().T@dtt
        sea, sg, sh = negative_logdet_jet(kernel,[km,kt],
                      [[2*self.yukawa**2*eye,zero],[zero,ktt]],self.sea_weight)
        return float(8*potential+self.beta*k*b+sea), gradient+sg, hessian+sh

    def homogeneous_value(self, phi, theta):
        phi = np.asarray(phi,float)
        d = dirac_jet(connection_profile(theta,self.seed),hopping=self.hopping)[0]
        kernel = sea_kernel(d,phi,self.mass,self.yukawa)
        rho = float(np.dot(phi,phi)/2)
        b = face_action_jet(connection_profile(theta,self.seed))[0]
        sea = negative_logdet_jet(kernel,weight=self.sea_weight)[0]
        return float(8*(self.a*rho+self.quartic*rho*rho/2)+self.beta*self.gate.jet(rho)[0]*b+sea)


def scalar_texture_matrix(d0, masses):
    """One-flavor scalar texture; two degenerate flavors have (det D)^2>=0.

    Positive masses imply a positive Hermitian part and no zero determinant.
    This does not assert positivity for arbitrary inhomogeneous pion fields.
    """
    masses = np.asarray(masses,float)
    if (masses.shape != (len(d0)//12,) or not np.isfinite(masses).all()
            or np.any(masses <= 0)):
        raise ValueError('One positive scalar mass per node required.')
    return d0+block_diag(*(m*np.eye(12) for m in masses))


def _product_jet(left, right):
    a, da, dda = left
    b, db, ddb = right
    return a@b, da@b+a@db, dda@b+2*da@db+a@ddb


def retained_dirac_jet(matrices, retained):
    """Exact fermionic Gaussian matching with one source through second order.

    D_eff=D_PP-D_PQ D_QQ^-1 D_QP, det D=det D_QQ det D_eff.
    Return (D_eff jet, D_QQ jet). A retained solver uses D_eff, while only
    det D_QQ enters its sea influence. Never also attach det D to that solver.
    This source-dependent Schur identity reuses the existing elimination
    principle; it is not a gluonic integration or full LF/Fock match.
    """
    if len(matrices) != 3:
        raise ValueError('Dirac value, first and second source matrices required.')
    n = len(matrices[0])
    if any(m.shape != (n,n) or not np.isfinite(m).all() for m in matrices):
        raise ValueError('Finite square jets of the same size required.')
    p = np.asarray(retained)
    if (p.ndim != 1 or p.dtype.kind not in 'iu' or len(p) == 0 or len(p) >= n
            or len(np.unique(p)) != len(p) or np.any(p < 0) or np.any(p >= n)):
        raise ValueError('Nonempty unique proper retained index set required.')
    q = np.setdiff1d(np.arange(n),p)
    pp,pq,qp,qq = [tuple(m[np.ix_(a,b)] for m in matrices)
                   for a,b in ((p,p),(p,q),(q,p),(q,q))]
    inverse = np.linalg.inv(qq[0])
    inverse_jet = (inverse,-inverse@qq[1]@inverse,
                   2*inverse@qq[1]@inverse@qq[1]@inverse-inverse@qq[2]@inverse)
    induced = _product_jet(_product_jet(pq,inverse_jet),qp)
    return tuple(a-b for a,b in zip(pp,induced)), qq


def dirac_determinant_jet(matrices):
    """Complex logdet source derivatives; no discarded determinant phase."""
    d, ds, dss = matrices
    logabs, phase = full_logdet(d)
    inv = np.linalg.inv(d)
    first = np.trace(inv@ds)
    second = np.trace(inv@dss-inv@ds@inv@ds)
    return logabs, phase, complex(first), complex(second)


def texture_hamiltonian(graph, fields, mass=0., yukawa=1., hopping=.4):
    """Hermitian instantaneous Dirac control for arbitrary real O(4) textures.

    h_ab=-i h alpha_axis U_ab; h_ba=h_ab^dagger. Onsite term is
    beta(m+y sigma)+i y beta gamma5 tau.pi. This is a finite background
    Hamiltonian, NOT an LF Hamiltonian or a reconstructed Euclidean ensemble.
    It supplies a canonical-operator alternative to assuming a positive
    determinant for every pion texture. Gauge-field quantization is not run.
    """
    fields = np.asarray(fields,float)
    if (fields.shape != (8,4) or not np.isfinite(fields).all()
            or tuple(graph.nodes) != tuple(range(8))):
        raise ValueError('Eight finite real O(4) site fields on the cube required.')
    _finite(mass,'mass',nonnegative=True)
    _finite(yukawa,'Yukawa',positive=True)
    _finite(hopping,'hopping',positive=True)
    result = np.zeros((192,192),complex)
    beta = GAMMA[0]
    for n,phi in enumerate(fields):
        onsite = (np.kron(np.kron(beta,np.eye(3)),np.eye(2))*(mass+yukawa*phi[0])
                  +1j*yukawa*np.kron(np.kron(beta@G5,np.eye(3)),
                                     np.einsum('a,aij->ij',phi[1:],PAULI)))
        result[24*n:24*(n+1),24*n:24*(n+1)] = onsite
    for (a,b),u in graph.links.items():
        delta = np.flatnonzero(VERTICES[a] != VERTICES[b])
        if len(delta) != 1:
            raise ValueError('Cube nearest-neighbor edges required.')
        block = -1j*hopping*np.kron(np.kron(beta@GAMMA[1+delta[0]],u),np.eye(2))
        result[24*a:24*(a+1),24*b:24*(b+1)] = block
        result[24*b:24*(b+1),24*a:24*(a+1)] = block.conj().T
    return result
