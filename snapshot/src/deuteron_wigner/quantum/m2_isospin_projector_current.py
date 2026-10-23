"""Straight-path gauging of the retained three-quark I=1/2 projector.

This finite internal projector is distinct from the oscillator cutoff.
Its flavor-exchange terms carry a current between the two physical
quark positions. No photon is assigned to an auxiliary field.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np
from .m2_coupled_longitudinal_jacobi import oscillator_shell,jacobi_frame,longitudinal_basis
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS,build_spin_flavor_s3_algebra
from .m2_galerkin_projector_current import _rectangular_displacement
from .m2_canonical_good_current import parton_charge


@lru_cache(maxsize=512)
def exchange_path_orbital(d,cutoff,wa,wb,delta,b,order):
    """Delta.K = D(wa Delta)-D(wb Delta), without inverse Delta².

    Differentiate exact continuum oscillator overlaps along the straight
    interpolation from the b-quark position to the a-quark position.
    The derivative generator needs exactly N+1, not a finite outer H.
    """
    wa,wb=np.asarray(wa,float),np.asarray(wb,float);delta=np.asarray(delta,float)
    if (wa.shape!=(d,) or wb.shape!=(d,) or delta.shape!=(2,)
            or not np.all(np.isfinite(np.r_[wa,wb,delta,b])) or b<=0
            or isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or order<2):
        raise ValueError("finite recoil/scales and explicit path quadrature required")
    outer=oscillator_shell(d,cutoff+1);n=len(oscillator_shell(d,cutoff));lookup={o:j for j,o in enumerate(outer)}
    g=np.zeros((2,n,len(outer)),complex);w=wa-wb
    derivative=np.array([np.r_[w,w],np.r_[-1j*w,1j*w]])/(2*b)
    for col,state in enumerate(outer):
        for mode in range(2*d):
            for step in (-1,1):
                out=list(state);out[mode]+=step
                if out[mode]<0 or sum(out)>cutoff:continue
                row=lookup[tuple(out)]
                value=derivative[:,mode] if step==1 else -derivative[:,mode].conj()
                g[:,row,col]+=np.sqrt(state[mode]+(step==1))*value
    nodes,weights=np.polynomial.legendre.leggauss(order)
    result=np.zeros((2,n,n),complex)
    for t,weight in zip((nodes+1)/2,weights/2):
        shift=(wb+t*(wa-wb))[:,None]*delta
        result+=weight*np.einsum("jab,bc->jac",g,
            _rectangular_displacement(d,cutoff+1,cutoff,shift,b),optimize=True)
    result.setflags(write=False);return result


def retained_isospin_projector():
    return (3.75*np.eye(64)-build_spin_flavor_s3_algebra().isospin_squared)/3


def isospin_outside_tangent(pilot,source,transfer,*,charge,quadrature_order,sector="qqqg"):
    """(1-I) X_I I on a full qqq or qqqg tensor in the retained I space.

    Tensor layout (longitudinal,orbital,color,64 quark spins,gluon spin,column)
    for qqqg; the gluon-spin axis is absent in qqq.
    I=1-(S01+S02+S12)/3 with S swapping FLAVORS only, not helicities.
    X=-sum S_ab (e_a-e_b) K_ab/3; Delta.X=[I,rho].
    Both external I blocks of X are removed by the projector tangent.
    This function supplies its incoming-I to outgoing-complement block.
    """
    source=np.asarray(source,complex);sf=np.array(SPIN_FLAVOR_BASIS)
    if sector not in ("qqq","qqqg"):
        raise ValueError("explicit retained qqq or qqqg sector required")
    d=2 if sector=="qqq" else 3
    modes=longitudinal_basis(sector,pilot.total_k2);n=len(oscillator_shell(d,pilot.cutoff))
    shape=(len(modes),n,1,64) if sector=="qqq" else (len(modes),n,2,64,2)
    if source.ndim!=len(shape)+1 or source.shape[:-1]!=shape:
        raise ValueError("actual full retained "+sector+" source tensor required")
    # A trivial gluon-spin axis lets both physical sectors share the same
    # three-quark flavor algebra without inventing a gluon in qqq.
    if sector=="qqq":source=source[...,None,:]
    iso=retained_isospin_projector()
    projected=np.einsum("st,loathk->loashk",iso,source,optimize=True)
    if not np.allclose(projected,source,atol=3e-13,rtol=3e-13):
        raise ValueError("incoming tensor must lie in the retained I=1/2 space")
    result=np.zeros((2,*source.shape),complex)
    delta=tuple(float(x) for x in transfer)
    for li,state in enumerate(modes):
        fr=jacobi_frame(state)
        for a,b in combinations(range(3),2):
            changed=sf.copy()
            changed[:,a]=2*(sf[:,b]//2)+sf[:,a]%2
            changed[:,b]=2*(sf[:,a]//2)+sf[:,b]%2
            mapped=np.ravel_multi_index(changed.T,(4,4,4))
            weight=parton_charge(sf[:,a],"q",charge)-parton_charge(sf[:,b],"q",charge)
            orbital=exchange_path_orbital(d,pilot.cutoff,tuple(fr.forward[:-1,a]),
                tuple(fr.forward[:-1,b]),delta,pilot.width_GeV,quadrature_order)
            value=-np.einsum("joi,iasnk,s->joasnk",orbital,source[li],weight,optimize=True)/3
            block=result[:,li]
            block[:,:,:,mapped,:,:]+=value
    result-=np.einsum("st,jloathk->jloashk",iso,result,optimize=True)
    return result[...,0,:] if sector=="qqq" else result
