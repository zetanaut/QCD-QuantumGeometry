"""Canonical action versus spin-blind gauge-path coordinates.

This is a one-emission SHORT-DISTANCE-CORRESPONDENCE test, not a binding
model, loop self-energy or higher-order decorated-Wilson construction.
Canonical quark/gluon dynamics is unchanged. All map columns (both parent
helicities) are tested, not merely one fitted input state.
"""
from itertools import product
from math import sqrt

import numpy as np

from .canonical_lf_fields import GAMMA,lf_spinor,polarization
from .canonical_lf_network import cubic_kernel
from .lf_path_states import DressingPath,PathCarrier,reference
from .modular_gauge_network import generators


PAULI=np.array([np.eye(2),[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)


def canonical_map(carrier):
    """V[m=(k,hg),hout,hin], GeV, gs/color stripped but cell normalized."""
    p=carrier.quarks[0];v=np.empty((len(carrier.momenta),2,2),complex)
    for m,(k,hg) in enumerate(zip(carrier.momenta,carrier.helicities)):
        for a,b in product(range(2),repeat=2):
            v[m,a,b]=cubic_kernel('q_to_qg',p-k,k,p,carrier.mass,
                                  (2*a-1,int(hg),2*b-1))/sqrt(carrier.cell.volume)
    return v


def direct_dirac_map(carrier):
    """Independent Cartesian component contraction; no cubic/slash helper."""
    p=carrier.quarks[0];v=np.empty((len(carrier.momenta),2,2),complex)
    for m,(k,hg) in enumerate(zip(carrier.momenta,carrier.helicities)):
        a=p-k;eps=polarization(*k,int(hg)).conj()
        e0=(eps[0]+eps[1])/sqrt(2);e3=(eps[0]-eps[1])/sqrt(2)
        operator=e0*GAMMA[0]-eps[2]*GAMMA[1]-eps[3]*GAMMA[2]-e3*GAMMA[3]
        u=np.column_stack([lf_spinor(*a,carrier.mass,h) for h in (-1,1)])
        parent=np.column_stack([lf_spinor(*p,carrier.mass,h) for h in (-1,1)])
        v[m]=-u.conj().T@GAMMA[0]@operator@parent/sqrt(8*a[0]*k[0]*p[0]*carrier.cell.volume)
    return v


def constrained_good_map(carrier):
    """Independent two-component LF constraint, including epsilon-minus.

    chi axes (-,+) are constant good-spinor columns. With alpha=gamma0 gammaT,
    h_g=-eps-minus + (B0(out) alpha.eps/out+ + alpha.eps B0(in)/in+)/2.
    No full Dirac spinor or canonical cubic helper is used in this route.
    """
    good=np.array([[0,1],[1,0],[0,1],[-1,0]],complex)/sqrt(2)
    alpha=np.array([GAMMA[0]@GAMMA[i] for i in (1,2)])
    p=carrier.quarks[0];out=[]
    b0=lambda q:np.einsum('i,ijk->jk',q[1:],alpha)+carrier.mass*GAMMA[0]
    for k,h in zip(carrier.momenta,carrier.helicities):
        a=p-k;eps=polarization(*k,int(h)).conj()
        vertex=np.einsum('i,ijk->jk',eps[2:],alpha)
        hgood=-eps[1]*np.eye(2)+good.conj().T@(b0(a)@vertex/a[0]+vertex@b0(p)/p[0])@good/2
        out.append(hgood*carrier.cell.field_factor(k,'boson'))
    return np.array(out)


def gap(carrier):
    p=carrier.quarks[0];k=carrier.momenta;x=k[:,0]/p[0]
    return (np.sum((k[:,1:]-x[:,None]*p[1:])**2,axis=1)+x*x*carrier.mass**2)/(
        2*p[0]*x*(1-x))


def spin_components(matrix):
    """Complete Hilbert-Schmidt Pauli coordinates, not independent couplings."""
    return np.einsum('aij,mij->ma',PAULI.conj(),matrix)/2


def path_dictionary():
    """Four declared closed LF paths; no dynamically selected flux labels."""
    c,p=reference()
    p.pop('bare')
    p['additional_detour']=DressingPath(((0,0,0),(3,-2,1),(1,1,4),(2,3,-1),(0,0,0)))
    return c,p


def coordinate_image(carrier,paths,spin_indices=(0,)):
    """L[m,hout,hin,column]=i F_path[m] sigma_alpha[hout,hin].

    These are tangent coordinates (U_path-I)*sigma_alpha, with zero bare
    term. Higher-order normalized decorated states are NOT defined here.
    Spin tensors transform as operator intertwiners; their Pauli labels are
    basis coordinates, not separate physical couplings or scalar invariants.
    """
    spin_indices=tuple(spin_indices)
    if (not spin_indices or len(set(spin_indices))!=len(spin_indices)
            or any(isinstance(i,bool) or i not in range(4) for i in spin_indices)):
        raise ValueError('Distinct Pauli indices 0..3 required.')
    fs=np.column_stack([carrier.integrals(path)[0] for path in paths.values()])
    return np.stack([1j*fs[:,j,None,None]*PAULI[a]
                     for j in range(fs.shape[1]) for a in spin_indices],axis=-1)


def fit_tangent(target,delta,image):
    """Gap-weighted projection of full map; returns represented chi and residual.

    chi_exact=-V/Delta. The Frobenius metric is an equal-spin average after
    multiplying by C_F/2. SVD threshold resolves numerical linear dependence
    only; all retained singular values and residuals are returned. No physics
    direction is deleted from target or relabelled as zero.
    """
    target=np.asarray(target,complex);delta=np.asarray(delta,float)
    image=np.asarray(image,complex)
    if (target.ndim!=3 or target.shape[1:]!=(2,2) or delta.shape!=(len(target),)
            or image.ndim!=4 or image.shape[:3]!=target.shape
            or not all(np.isfinite(x).all() for x in (target,delta,image)) or np.any(delta<=0)):
        raise ValueError('Finite matching map/image and strictly positive free gaps required.')
    exact=-target/delta[:,None,None]
    weighted=(np.sqrt(delta)[:,None,None,None]*image).reshape(-1,image.shape[-1])
    rhs=(np.sqrt(delta)[:,None,None]*exact).ravel()
    u,s,vh=np.linalg.svd(weighted,full_matrices=False)
    threshold=max(weighted.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    active=s>threshold
    coefficients=vh[active].conj().T@((u[:,active].conj().T@rhs)/s[active])
    represented=np.einsum('mijk,k->mij',image,coefficients)
    error=represented-exact
    return dict(coefficients=coefficients,wave=represented,exact=exact,
                singular_values=s,rank=int(sum(active)),threshold=threshold,
                wave_error=float(np.linalg.norm(error)),
                relative_wave_error=float(np.linalg.norm(error)/np.linalg.norm(exact)) if np.linalg.norm(exact) else 0.,
                energy_deficit_GeV=float((4/3)/2*np.sum(delta[:,None,None]*abs(error)**2)),
                stationary_residual=delta[:,None,None]*represented+target)


def maximal_spin_blind(target,delta):
    """Best possible arbitrary mode coefficients times identity, regardless of paths."""
    chi=-target/delta[:,None,None]
    return np.trace(chi,axis1=1,axis2=2)[:,None,None]*np.eye(2)[None]/2


def energy_coefficient(target,delta,chi):
    """Spin-averaged normalized E2 minus a COMMON unknown bare H2 expectation.

    E2-C0 = C_F/2 Tr[chi† Delta chi + chi† V + V† chi]. This difference
    between trial tangents needs no new counterterm choice. It is NOT a
    physical mass shift, a bound-state solve or a complete QCD self-energy.
    """
    return float((4/3)/2*(np.sum(delta[:,None,None]*abs(chi)**2)+2*np.vdot(chi,target).real))


def color_map(spin_map):
    """Full open quark color map, m,hout,cout,a,hin,cin (no singlet projection)."""
    return np.einsum('mij,acb->micajb',spin_map,generators())


def spin_loss(chi):
    """Order-two emission-induced loss of parent quark helicity alignment.

    For each parent h, include its norm-restoring vacuum term. The output
    is 1 - <sigma3_out>/sigma3_parent coefficient = 2*C_F*flip norm^2.
    It is a constituent LF spin diagnostic, NOT a nucleon axial observable.
    """
    return 2*(4/3)*np.array([np.sum(abs(chi[:,1,0])**2),np.sum(abs(chi[:,0,1])**2)])


def reduced_spin_coefficient(chi,rho):
    """Order-two reduced spin map with common Hermitian norm restoration.

    Not a phenomenological noise channel or complete QCD spin observable.
    An undetermined order-two unitary spin-frame rotation is not fitted here.
    Opposite-input averaged SAME-axis retention below is insensitive to it.
    """
    a=np.einsum('mji,mjk->ik',chi.conj(),chi)
    return (4/3)*(np.einsum('mij,jk,mlk->il',chi,rho,chi.conj())-(a@rho+rho@a)/2)


def spin_retention_loss(chi):
    """Opposite-input averaged losses along Pauli x,y,z, coefficient of gs^2."""
    return np.array([-np.trace(s@reduced_spin_coefficient(chi,s)).real/2 for s in PAULI[1:]])
