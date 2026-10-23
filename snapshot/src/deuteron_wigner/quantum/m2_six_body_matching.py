"""Coherent leading six-body H2 excursion and its five-body backflow.

Finite-matrix algebra with EXPLICIT H5,H6,V5P,U65,W6P inputs. This does
not supply the canonical six-body vertices, choose physical couplings,
or pretend that a missing quartic source is zero. E and H are in GeV^2;
V,U,W are coefficients in H5P=g V, H65=g U, H6P=g^2 W.
"""
from dataclasses import dataclass

import numpy as np


def _matrix(value,shape=None,*,hermitian=False):
    a=np.array(value,dtype=complex,copy=True)
    if a.ndim!=2 or not np.all(np.isfinite(a)) or (shape is not None and a.shape!=shape):
        raise ValueError("finite matrix in the declared coordinates required")
    if hermitian and (a.shape[0]!=a.shape[1] or not np.allclose(a,a.conj().T,rtol=0,atol=1e-12)):
        raise ValueError("Hermitian finite Hamiltonian required")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class SixBodyWaves:
    five_leading: np.ndarray
    six_source: np.ndarray
    six_leading: np.ndarray
    five_backflow: np.ndarray


class SixBodyExcursion:
    physical=False
    canonical_six_body_vertices_supplied=False

    def __init__(self,*,H5,H6,V5P,U65,W6P):
        self.H5=_matrix(H5,hermitian=True)
        self.H6=_matrix(H6,hermitian=True)
        self.V=_matrix(V5P)
        n5,n6=len(self.H5),len(self.H6)
        if min(n5,n6,self.V.shape[1])==0 or self.V.shape[0]!=n5:
            raise ValueError("nonempty compatible spaces required")
        self.U=_matrix(U65,(n6,n5))
        self.W=_matrix(W6P,(n6,self.V.shape[1]))
        self.nP=self.V.shape[1]
        self._spectra=tuple(np.linalg.eigh(h) for h in (self.H5,self.H6))

    def _resolvent_action(self,sector,energy,columns):
        e=float(energy)
        d,u=self._spectra[sector]
        if not np.isfinite(e) or np.min(abs(e-d))<1e-10:
            raise ValueError("finite off-pole energy required; no pole clipping")
        return u@((u.conj().T@columns)/(e-d)[:,None])

    def waves(self,energy):
        a1=self._resolvent_action(0,energy,self.V)
        f=self.W+self.U@a1
        a6=self._resolvent_action(1,energy,f)
        b5=self._resolvent_action(0,energy,self.U.conj().T@a6)
        return SixBodyWaves(a1,f,a6,b5)

    def hamiltonian_coefficient(self,energy):
        """Coefficient of g^4: (W+UR5V)^dag R6 (W+UR5V)."""
        a=self.waves(energy)
        return a.six_source.conj().T@a.six_leading

    def norm_coefficient(self,energy):
        """-d_E Sigma4, including both five-body backflow endpoints."""
        a=self.waves(energy)
        c=a.five_leading.conj().T@a.five_backflow
        return a.six_leading.conj().T@a.six_leading+c+c.conj().T

    def current_coefficient(self,energy_out,energy_in,*,rho5,rho6):
        """Number-preserving two-energy density, coefficient of g^4.

        Both rho blocks are explicit. This is not a number-changing or
        complete physical four-current prescription.
        """
        r5=_matrix(rho5,self.H5.shape)
        r6=_matrix(rho6,self.H6.shape)
        af,ai=self.waves(energy_out),self.waves(energy_in)
        return (af.six_leading.conj().T@r6@ai.six_leading
                +af.five_backflow.conj().T@r5@ai.five_leading
                +af.five_leading.conj().T@r5@ai.five_backflow)
