"""Continuum free transverse current projected onto complete Jacobi shells.

Components are J^(x,y)/(2P+), in the same normalization as the good density.
This is the canonical one-body part, NOT a conserved finite-Hamiltonian
current by itself. Interaction contacts and the gauged Galerkin boundary
must be supplied coherently. No physical matching or counterterm is chosen.
"""
import numpy as np
from scipy import sparse

from .m2_canonical_good_current import parton_charge
from .m2_coupled_longitudinal_jacobi import (
    jacobi_frame, longitudinal_basis, oscillator_shell,
)
from .m2_coupled_pilot import _displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def displacement_and_moments(dimension,cutoff,shift,*,b_GeV):
    """D=<out|D(s)|in> and M[a,j]=<out|D(s) Q_a^j|in>, GeV.

    D(s)=integral phi_out*(Q+s) phi_in(Q). Ladder neighbors of the ket
    through N+1 are retained even when they are outside the external shell.
    Consequently M is NOT the product of two truncated matrices D_N Q_N.
    """
    shell=oscillator_shell(dimension,cutoff)
    d=len(shell[0])//2
    s=np.asarray(shift,float);b=float(b_GeV)
    if s.shape!=(d,2) or not np.all(np.isfinite(s)) or not np.isfinite(b) or b<=0:
        raise ValueError("finite (dimension,2) shift and positive width required")
    plus=(s[:,0]-1j*s[:,1])/(2*b)
    alpha=np.r_[plus,plus.conj()]
    tables=np.array([[[_displacement(o,i,z) for i in range(cutoff+2)]
                       for o in range(cutoff+1)] for z in alpha])
    occupations=np.array(shell)
    def overlap(kets):
        value=np.ones((len(shell),len(shell)),complex)
        for k in range(2*d):
            value*=tables[k,occupations[:,k,None],kets[None,:,k]]
        return value
    density=overlap(occupations)
    moments=np.zeros((d,2,len(shell),len(shell)),complex)
    for k in range(2*d):
        for step in (-1,1):
            kets=occupations.copy()
            kets[:,k]=np.maximum(occupations[:,k]+step,0)
            # For vacuum lowering the coefficient is exactly zero. The
            # clamped array index only avoids looking up occupation -1.
            x=b*np.sqrt(occupations[:,k]+(step==1))/2
            value=overlap(kets)*x[None,:]
            moments[k%d,0]+=value
            moments[k%d,1]+=(-1j if k<d else 1j)*step*value
    return density,moments


def free_transverse_current_between(outgoing,incoming,transfer,*,
                                    total_pplus_GeV,charge,
                                    mean_total_transverse=(0.,0.)):
    """Retained qqq/qqqg canonical free transverse current at Delta+=0.

    External transverse momenta are P_mean +/- Delta/2. Endpoints can have
    different Jz but must share the same raw regulator and parameters.
    At fixed positive p+ the quark and charge-stripped antiquark kernel is
      (p_fT+p_iT+i h z_cross_Delta)/(2p+).
    Its mass terms cancel; no helicity-flip free kernel survives at Delta+=0.
    This retained implementation has quarks only; no sea endpoint is implied.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                 "gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    delta=np.asarray(transfer,float)
    mean=np.asarray(mean_total_transverse,float)
    pplus=float(total_pplus_GeV)
    if (delta.shape!=(2,) or mean.shape!=(2,)
            or not np.all(np.isfinite(np.r_[delta,mean,pplus])) or pplus<=0):
        raise ValueError("finite transverse vectors and explicit positive P+ required")
    sf=np.array(SPIN_FLAVOR_BASIS)
    charges=[parton_charge(sf[:,i],"q",charge) for i in range(3)]
    cross=np.array([-delta[1],delta[0]])
    reduced=[[],[]]
    p=incoming
    for sector,uf,ui in zip(("qqq","qqqg","qqqg"),outgoing.bases,incoming.bases):
        d=2 if sector=="qqq" else 3
        nc=d-1
        norb=len(oscillator_shell(d,p.cutoff))
        blocks=[[],[]]
        for modes in longitudinal_basis(sector,p.total_k2):
            frame=jacobi_frame(modes)
            current=[sparse.csr_matrix((norb*nc*64,)*2,dtype=complex) for _ in range(2)]
            for active in range(3):
                x=frame.fractions[active]
                shift=frame.forward[:-1,active,None]*delta[None,:]
                density,moments=displacement_and_moments(d,p.cutoff,shift,b_GeV=p.width_GeV)
                intrinsic=np.einsum("a,ajoi->joi",frame.inverse[active,:-1],moments)
                q=sparse.diags(np.tile(charges[active],nc))
                qh=sparse.diags(np.tile(charges[active]*(2*(sf[:,active]%2)-1),nc))
                for j in range(2):
                    convection=intrinsic[j]+((1-x)*delta[j]/2+x*mean[j])*density
                    current[j]+=(sparse.kron(convection,q,format="csr")
                        +.5j*cross[j]*sparse.kron(density,qh,format="csr"))/(x*pplus)
            for j in range(2):
                blocks[j].append(current[j])
        for j in range(2):
            full=sparse.block_diag(blocks[j],format="csr")
            reduced[j].append((uf.getH()@full@ui).toarray())
    return np.array([sparse.block_diag(blocks).toarray() for blocks in reduced])
