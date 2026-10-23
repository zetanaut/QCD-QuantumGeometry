"""Whole-vertex reference-shape pair action with an exact auxiliary projector.

The shape degree bounds u+v inside one physical quartic vertex. Separately
truncating the two halves would import higher-degree terms. The auxiliary
coordinate is only a Hamiltonian factorization device, never a photon leg.
"""
from dataclasses import replace
from functools import lru_cache, partial
from math import factorial
import numpy as np
from .m2_gaussian_field_regulator import GaussianClusterRegulator
from .m2_gaussian_cluster_interactions import cluster_pair_current_orbital
from .m2_pair_current_factorized import FactorizedPairCurrentAction
from .m2_five_body_factorized import apply_terms
from .m2_six_body_cubic import SixBodyCubicAction


class CorrelatedReferencePairCurrentAction:
    physical=False

    def __init__(self,spaces,*,b_GeV,regulator):
        if not isinstance(regulator,GaussianClusterRegulator) or regulator.reference_shape is None:
            raise ValueError("an explicitly bound whole-vertex reference shape is required")
        self.regulator=regulator
        self.shape=regulator.shape_for()
        self.spaces=dict(spaces)
        self.delta=self.shape.a-self.shape.a_reference
        degree=self.shape.degree if self.delta else 0
        base=replace(regulator,reference_shape=None)
        self.factors=tuple(FactorizedPairCurrentAction(spaces,b_GeV=b_GeV,
            _orbital_kernel=lru_cache(maxsize=8192)(partial(cluster_pair_current_orbital,
                regulator=base,derivative_order=u))) for u in range(degree+1))
        self.auxiliary=self.factors[0].auxiliary
        # Derivatives change only radial matrix elements, never the exact
        # auxiliary coordinate domain, statistical or isospin projector.
        a=self.auxiliary
        for factor in self.factors[1:]:
            other=factor.auxiliary
            if (any(getattr(a,k)!=getattr(other,k) for k in
                    ("sector","total_k2","cutoff","target","jz_twice","longitudinal","orbitals"))
                    or any(not np.array_equal(getattr(a,k),getattr(other,k))
                           for k in ("coordinates","rows","weight","isospin_projector"))):
                raise ArithmeticError("reference derivatives must preserve the exact auxiliary projector")

    def apply(self,columns):
        count=SixBodyCubicAction._validate(self.spaces,columns)
        if len(self.factors)==1:
            return self.factors[0].apply(columns)
        a=self.auxiliary
        projected={n:s.statistics_project(columns[n]) for n,s in self.spaces.items()}
        weighted=[]
        for factor in self.factors:
            contracted=np.zeros((a.dimension,count),complex)
            for name in self.spaces:
                contracted+=(2 if name=="qqqq-qbar" else 1)*apply_terms(
                    factor.terms[name],projected[name],a.dimension,adjoint=True)
            weighted.append(a.weight[:,None]*a.project(contracted))
        result={n:np.zeros((s.dimension,count),complex) for n,s in self.spaces.items()}
        for u,factor in enumerate(self.factors):
            combined=sum(self.delta**(u+v)/(factorial(u)*factorial(v))*weighted[v]
                         for v in range(len(self.factors)-u))
            for name,s in self.spaces.items():
                result[name]+=apply_terms(factor.terms[name],combined,s.dimension)
        return {n:(2 if n=="qqqq-qbar" else 1)*s.statistics_project(result[n])
                for n,s in self.spaces.items()}
