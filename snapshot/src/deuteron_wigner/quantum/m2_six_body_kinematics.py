"""Six-particle color completeness and global-Jacobi kinematics.

No interaction, selected state or probability is inferred from these spaces.
Their 22/8 singlet multiplicities are obtained independently by SU(3)
tensor-product recursion, not by declaring the numerical source rank complete.
"""
from collections import Counter
from functools import lru_cache
from math import sqrt
from operator import index

import numpy as np

from deuteron_wigner.microscopic.h0.color import representation_generators
from .m2_coupled_longitudinal_jacobi import JacobiFrame
from .m2_excluded_fock_kinematics import color_basis as five_color_basis, _act

SPECIES = {
    "qqqggg": ("q","q","q","g","g","g"),
    "qqqq-qbar-g": ("q","q","q","q","qbar","g"),
}
PARENTS = {"qqqggg":"qqqgg", "qqqq-qbar-g":"qqqq-qbar"}
TREE = ((((0,3),4),5),(1,2))


def _integer(n):
    if isinstance(n,(bool,np.bool_)):
        raise ValueError("exact integer required")
    return index(n)


def _fundamental_product(decomposition, conjugate=False):
    result=Counter()
    for (p,q),n in decomposition.items():
        if conjugate:
            terms=[(p,q+1)]+([(p+1,q-1)] if q else [])+([(p-1,q)] if p else [])
        else:
            terms=[(p+1,q)]+([(p-1,q+1)] if p else [])+([(p,q-1)] if q else [])
        for rep in terms:
            result[rep]+=n
    return result


def color_decomposition(species):
    """Dynkin (p,q) products, using 8=3 tensor 3bar minus 1 exactly."""
    result=Counter({(0,0):1})
    for s in species:
        if s=="q":
            result=_fundamental_product(result)
        elif s=="qbar":
            result=_fundamental_product(result,True)
        elif s=="g":
            product=_fundamental_product(_fundamental_product(result),True)
            for rep,n in result.items():
                product[rep]-=n
            if any(n<0 for n in product.values()):
                raise ArithmeticError("negative SU(3) multiplicity")
            result=+product
        else:
            raise ValueError("unknown species")
    return dict(result)


def representation_dimension(p,q):
    return (p+1)*(q+1)*(p+q+2)//2


@lru_cache(maxsize=16)
def longitudinal_states(sector,total_k2):
    species=SPECIES[sector]
    minimum=tuple(2 if s=="g" else 1 for s in species)
    extra=_integer(total_k2)-sum(minimum)
    if extra<0 or extra%2:
        return ()
    def compositions(n,d):
        if d==1:
            yield (n,)
        else:
            for a in range(n+1):
                for rest in compositions(n-a,d-1):
                    yield (a,)+rest
    return tuple(tuple(a+2*b for a,b in zip(minimum,c))
                 for c in compositions(extra//2,6))


@lru_cache(maxsize=2048)
def frame(sector,modes):
    species=SPECIES[sector]
    modes=tuple(_integer(n) for n in modes)
    if len(modes)!=6 or any(n<=0 or n%2!=int(s!="g") for n,s in zip(modes,species)):
        raise ValueError("six positive odd fermion/even gluon modes required")
    x=np.array(modes,float)/sum(modes)
    rows=[]
    def visit(tree):
        if not isinstance(tree,tuple):
            return (tree,)
        left,right=map(visit,tree)
        a,b=x[list(left)].sum(),x[list(right)].sum()
        row=np.zeros(6)
        row[list(left)]=sqrt(b/(a*(a+b)))
        row[list(right)]=-sqrt(a/(b*(a+b)))
        rows.append(row)
        return left+right
    visit(TREE)
    forward=np.vstack(rows+[np.ones(6)])
    inverse=x[:,None]*forward.T
    for a in (x,forward,inverse):
        a.setflags(write=False)
    return JacobiFrame(modes,TREE,x,forward,inverse)


def permutation_frame(sector,modes,permutation):
    p=tuple(_integer(n) for n in permutation)
    species=SPECIES[sector]
    if sorted(p)!=list(range(6)) or any(species[i]!=species[j] for i,j in enumerate(p)):
        raise ValueError("species-preserving permutation required")
    outgoing=tuple(modes[i] for i in p)
    a,b=frame(sector,modes),frame(sector,outgoing)
    return outgoing,b.forward[:-1]@a.inverse[list(p),:-1]


@lru_cache(maxsize=10)
def emission_tensor(sector,slot):
    """Raw color generator emission, appended gluon, BEFORE vertex/Fock factors."""
    species=SPECIES[sector]
    slot=_integer(slot)
    if slot not in range(5):
        raise ValueError("five-body emitter required")
    dims=tuple(8 if s=="g" else 3 for s in species[:-1])
    parent=five_color_basis(PARENTS[sector])
    tensor=parent.reshape(dims+(parent.shape[1],))
    rep="8" if species[slot]=="g" else "3bar" if species[slot]=="qbar" else "3"
    raw=np.stack([_act(g,tensor,slot) for g in representation_generators(rep)],axis=-2)
    return raw.reshape(np.prod(dims)*8,parent.shape[1])


@lru_cache(maxsize=2)
def color_basis(sector):
    expected=color_decomposition(SPECIES[sector])[(0,0)]
    candidates=np.column_stack([emission_tensor(sector,i) for i in range(5)])
    u,s,_=np.linalg.svd(candidates,full_matrices=False)
    if np.count_nonzero(s>1e-11)!=expected:
        raise ArithmeticError("canonical emission span is not the complete singlet space")
    u=u[:,:expected]
    # Deterministic lexicographic projector pivots, performed in coefficient
    # space to avoid ever allocating the huge raw-space square projector.
    coefficients=[]
    for row in u:
        v=row.conj().copy()
        for old in coefficients:
            v-=old*np.vdot(old,v)
        norm=np.linalg.norm(v)
        if norm>1e-10:
            coefficients.append(v/norm)
        if len(coefficients)==expected:
            break
    result=u@np.column_stack(coefficients)
    if np.linalg.norm(result.conj().T@result-np.eye(expected))>1e-9:
        raise ArithmeticError("color basis lost orthonormality")
    result.setflags(write=False)
    return result


@lru_cache(maxsize=10)
def color_emission(sector,slot):
    u,raw=color_basis(sector),emission_tensor(sector,slot)
    reduced=u.conj().T@raw
    if np.linalg.norm(raw-u@reduced)>3e-10:
        raise ArithmeticError("emission leaves the complete singlet space")
    return reduced


@lru_cache(maxsize=64)
def color_permutation(sector,permutation):
    species=SPECIES[sector]
    permutation_frame(sector,tuple(2 if s=="g" else 1 for s in species),permutation)
    dims=tuple(8 if s=="g" else 3 for s in species)
    u=color_basis(sector)
    moved=u.reshape(dims+(-1,)).transpose(tuple(permutation)+(6,)).reshape(u.shape)
    result=u.conj().T@moved
    if np.linalg.norm(moved-u@result)>3e-10:
        raise ArithmeticError("permutation leaves complete color space")
    return result


def color_generator_residual(sector):
    species=SPECIES[sector]
    dims=tuple(8 if s=="g" else 3 for s in species)
    reps=tuple("8" if s=="g" else "3bar" if s=="qbar" else "3" for s in species)
    u=color_basis(sector).reshape(dims+(-1,))
    return max(np.linalg.norm(sum(_act(representation_generators(rep)[a],u,i)
               for i,rep in enumerate(reps))) for a in range(8))
