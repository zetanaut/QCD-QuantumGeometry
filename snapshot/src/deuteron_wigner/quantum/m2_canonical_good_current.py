"""Flavor-resolved local good current at Delta+=0 in the declared finite basis.

The normalized matrix strips 2P+ from J+; charges are in units of e.
Positive longitudinal modes give a number-preserving canonical local current;
antiquark charge reverses by normal-order CAR. This is not a full current
four-vector, a nuclear form factor, or a zero-mode/renormalization completion.
Sources: canonical LF field expansion (SB hep-ph/0011372 Appendix B),
Drell--Yan--West overlap; no physical state or current fit is chosen here.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .m2_coupled_longitudinal_jacobi import jacobi_frame,longitudinal_basis,oscillator_shell
from .m2_coupled_pilot import _displacement
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def parton_charge(spins,species,charge):
    if charge not in ("baryon","electromagnetic","up_number","down_number"):
        raise ValueError("explicit supported conserved charge required")
    if species not in ("q","qbar","g"):
        raise ValueError("quark, antiquark or gluon required")
    spins=np.asarray(spins)
    if species=="g":
        return np.zeros_like(spins,dtype=float)
    sign=1 if species=="q" else -1
    if charge=="baryon":
        return sign*np.ones_like(spins,dtype=float)/3
    up=(spins<2)
    if charge=="electromagnetic":
        return sign*np.where(up,2/3,-1/3)
    return sign*(up if charge=="up_number" else ~up).astype(float)


def good_current_density(pilot,transfer,*,charge):
    """Same-Jz block of the local quark/antiquark J+/(2P+), not full J^mu."""
    return good_current_density_between(pilot,pilot,transfer,charge=charge)


def good_current_density_between(outgoing,incoming,transfer,*,charge):
    """All retained same-/opposite-Jz blocks, with the identical raw regulator.

    Endpoints may have different Jz, but not different physical parameters
    or raw truncations. This is still the bare local good current, not a
    complete current four-vector or an implicit physical-state selection.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                 "gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("current endpoints disagree on "+name)
    pilot=incoming
    delta=np.asarray(transfer,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
        raise ValueError("finite transverse transfer required")
    sf=np.array(SPIN_FLAVOR_BASIS)
    charges=[parton_charge(sf[:,i],"q",charge) for i in range(3)]
    reduced=[]
    for sector,uf,ui in zip(("qqq","qqqg","qqqg"),outgoing.bases,incoming.bases):
        d=2 if sector=="qqq" else 3
        nc=d-1
        shell=oscillator_shell(d,pilot.cutoff)
        blocks=[]
        for modes in longitudinal_basis(sector,pilot.total_k2):
            f=jacobi_frame(modes)
            block=sparse.csr_matrix((len(shell)*nc*64,)*2,dtype=complex)
            for particle in range(3):
                shift=f.forward[:-1,particle,None]*delta[None,:]
                plus=(shift[:,0]-1j*shift[:,1])/(2*pilot.width_GeV)
                alpha=np.r_[plus,plus.conj()]
                orbital=np.array([[np.prod([_displacement(o,i,z) for o,i,z in zip(bra,ket,alpha)])
                    for ket in shell] for bra in shell])
                block+=sparse.kron(orbital,sparse.diags(np.tile(charges[particle],nc)),format="csr")
            blocks.append(block)
        full=sparse.block_diag(blocks,format="csr")
        reduced.append((uf.getH()@full@ui).toarray())
    return sparse.block_diag(reduced).toarray()
