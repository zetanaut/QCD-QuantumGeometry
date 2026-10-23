"""Total SU(2) isospin on the actual packed Q5/Q6 spaces.

Quarks carry tau/2, antiquarks -tau^T/2, and gluons are singlets.
The Casimir is not the isospin of the three labeled valence slots when
a sea pair is present. This distinction is essential for H2 source domains.
No state is projected or deleted by this diagnostic operator.
"""
from functools import lru_cache
from itertools import combinations,product
import numpy as np
from scipy import sparse
from .m2_excluded_fock_kinematics import SPECIES as FIVE
from .m2_six_body_kinematics import SPECIES as SIX
from .m2_excluded_fock_action import IntermediateSpace
from .m2_six_body_space import SixBodySpace
from .m2_shell_projector_current import _columns

SPECIES={"qqq":("q",)*3,**FIVE,**SIX}
H2_ISOSPIN_PRESCRIPTION="flavor-degenerate-total-isospin-half-H2-source-domain"


@lru_cache(maxsize=8)
def _spin_casimir(sector):
    species=SPECIES[sector]
    dims=tuple(2 if s=="g" else 4 for s in species)
    spins=np.array(list(product(*(range(d) for d in dims))))
    fermions=[a for a,s in enumerate(species) if s!="g"]
    signs=np.array([1 if species[a]=="q" else -1 for a in fermions])
    z=signs[None,:]*(1-2*(spins[:,fermions]//2))/2
    diagonal=len(fermions)/2+z.sum(axis=1)**2
    rows=list(range(len(spins)));cols=rows.copy();values=list(diagonal)
    for j,k in combinations(range(len(fermions)),2):
        selected=np.flatnonzero(z[:,j]+z[:,k]==0)
        changed=spins[selected].copy()
        changed[:,fermions[j]]^=2
        changed[:,fermions[k]]^=2
        mapped=np.ravel_multi_index(changed.T,dims)
        rows.extend(mapped);cols.extend(selected)
        values.extend([signs[j]*signs[k]]*len(selected))
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(spins),)*2),spins,dims


def spin_total_isospin_squared(sector):
    """Sparse spin/flavor Casimir in the canonical physical slot order."""
    if sector not in SPECIES:raise ValueError("explicit physical P/Q sector required")
    return _spin_casimir(sector)[0].copy()


def total_isospin_squared_apply(space,values):
    """Apply I_total² without allocating the omitted spin/longitudinal rows."""
    if not isinstance(space,(IntermediateSpace,SixBodySpace)):
        raise ValueError("actual physical Q5/Q6 space required")
    v=_columns(space,values)
    casimir,spins,dims=_spin_casimir(space.sector)
    if tuple(space.spin_dims)!=dims or not np.array_equal(space.spins,spins):
        raise ValueError("physical spin/flavor order does not match the named sector")
    coords=space.coordinates
    result=casimir.diagonal()[coords[:,3],None]*v
    # Pair ladder terms are injective. Their mapped rows keep Iz and Jz,
    # so they remain inside the same packed target space.
    species=SPECIES[space.sector]
    fermions=[a for a,s in enumerate(species) if s!="g"]
    for a,b in combinations(fermions,2):
        sa,sb=(1 if species[k]=="q" else -1 for k in (a,b))
        za=sa*(1-2*(spins[:,a]//2));zb=sb*(1-2*(spins[:,b]//2))
        selected=np.flatnonzero(za[coords[:,3]]+zb[coords[:,3]]==0)
        changed=spins[coords[selected,3]].copy();changed[:,a]^=2;changed[:,b]^=2
        mapped=np.ravel_multi_index(changed.T,dims)
        c=coords[selected]
        raw_index=((c[:,0]*len(space.orbitals)+c[:,1])*space.ncolor+c[:,2])*space.nspin+mapped
        rows=space.lookup[raw_index]
        if np.any(rows<0):raise ArithmeticError("isospin ladder escaped its conserved physical target")
        result[rows]+=sa*sb*v[selected]
    return result


def require_half_isospin_H2_snapshot(snapshot,*,prescription):
    """Check the actual source-domain assumption; never project an impure wave.

    In the flavor-degenerate GaussianClusterH2 definition, all strong kernels,
    free resolvents, statistics and flavor-blind regulators commute with total
    SU(2). A retained-isospin projector tangent maps P(I=1/2) to I=3/2;
    its strong source image is orthogonal to every checked H2 wave. Thus its
    H2 pullback vanishes for any tangent representative, including minus.
    This is not a construction of the raw tangent, and is not valid after
    isospin breaking or for arbitrary external Q vectors.
    """
    from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot,GaussianClusterH2
    if prescription!=H2_ISOSPIN_PRESCRIPTION:
        raise ValueError("explicit flavor-degenerate H2 isospin prescription required")
    if not isinstance(snapshot,GaussianClusterSnapshot) or type(snapshot.owner) is not GaussianClusterH2:
        raise ValueError("the proved flavor-degenerate GaussianClusterH2 definition is required")
    if snapshot.owner.wick_operators.quark_flavors!=2:
        raise ValueError("same two degenerate strong flavors required")
    residuals={}
    for power,waves in zip((1,2,3),snapshot.five_wave_coefficients):
        for sector,space in snapshot.owner.spaces5.items():
            v=waves[sector]
            residual=total_isospin_squared_apply(space,v)-.75*v
            norm=float(np.linalg.norm(residual))
            if not np.isfinite(norm) or norm>3e-12*max(1.,float(np.linalg.norm(v))):
                raise ValueError("actual H2 wave is outside total isospin 1/2")
            residuals[(sector,power)]=norm
    for sector,space in snapshot.owner.spaces6.items():
        v=snapshot.six_wave_coefficient[sector]
        residual=total_isospin_squared_apply(space,v)-.75*v
        norm=float(np.linalg.norm(residual))
        if not np.isfinite(norm) or norm>3e-12*max(1.,float(np.linalg.norm(v))):
            raise ValueError("actual H2 wave is outside total isospin 1/2")
        residuals[(sector,2)]=norm
    return residuals
