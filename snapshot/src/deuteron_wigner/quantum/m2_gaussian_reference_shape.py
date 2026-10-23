"""Finite-reference Gaussian shape operators, before projection or Wick.

F_M(Q)=exp(-a Q) sum_{m=0}^M ((a-a_ref)Q)^m/m!, with a_ref<=a.
The degree is independent of coupling order. These are a named finite-model
transport family, not calibrated counterterms or a completed H2 assembly.
Use the WHOLE physical-leg quadratic form; auxiliary halves do not factorize.
"""
from dataclasses import dataclass
from operator import index
from math import comb,factorial
import numpy as np
from scipy.special import gammaincc
from .m2_gaussian_field_regulator import _array,gaussian_shell_integral


def radial_half_moment(n,t,order,scale):
    """(-d_a)^order [(1-t)^n/(1+t)^(n+1)], with t=scale*a.

    Leibniz form stays regular at t=1, including exact radial zeros. This
    is an unweighted derivative half, not a separately truncated vertex.
    """
    if any(isinstance(k,(bool,np.bool_)) for k in (n,order)):
        raise ValueError("exact nonnegative radial and derivative orders required")
    try:n,order=index(n),index(order)
    except TypeError as exc:
        raise ValueError("exact nonnegative radial and derivative orders required") from exc
    t,scale=float(t),float(scale)
    if min(n,order)<0 or min(t,scale)<0 or not np.all(np.isfinite([t,scale])):
        raise ValueError("nonnegative finite Gaussian radial data required")
    return scale**order*sum(comb(order,j)*factorial(n+order-j)/factorial(n-j)
        *(1-t)**(n-j)/(1+t)**(n+1+order-j) for j in range(min(n,order)+1))


@dataclass(frozen=True,kw_only=True)
class GaussianReferenceShape:
    base_width_GeV: float
    reference_width_GeV: float
    degree: int
    physical=False
    H2_transport_complete=False

    def __post_init__(self):
        b,r=float(self.base_width_GeV),float(self.reference_width_GeV)
        if not np.all(np.isfinite([b,r])) or b<=0 or r<b:
            raise ValueError("explicit finite widths with reference >= base > 0 required")
        if isinstance(self.degree,(bool,np.bool_)):
            raise ValueError("nonnegative exact reference expansion degree required")
        try:m=index(self.degree)
        except TypeError as exc:raise ValueError("nonnegative exact reference expansion degree required") from exc
        if m<0:raise ValueError("nonnegative exact reference expansion degree required")
        object.__setattr__(self,"base_width_GeV",b)
        object.__setattr__(self,"reference_width_GeV",r)
        object.__setattr__(self,"degree",m)

    @property
    def a(self):return 1/(2*self.base_width_GeV**2)

    @property
    def a_reference(self):return 1/(2*self.reference_width_GeV**2)

    def value_and_derivative(self,Q):
        """Stable scalar kernel and dF/dQ, not a physical current by itself."""
        q=np.asarray(Q,float)
        if not np.all(np.isfinite(q)) or np.any(q<0):
            raise ValueError("nonnegative finite physical quadratic form required")
        delta=self.a-self.a_reference;z=delta*q
        reference=np.exp(-self.a_reference*q)
        value=reference*gammaincc(self.degree+1,z)
        previous=reference*gammaincc(self.degree,z) if self.degree else np.zeros_like(q)
        return value,-self.a*value+delta*previous

    def weight_polynomial(self,field_maps,field_shifts,*,gauged=False):
        """Polynomial multiplying the BASE Gaussian or its first derivative.

        If gauged, return S_M-(a-a_ref)/a S_(M-1), not S_M. Multiplying
        an old Gaussian current by S_M omits the new polynomial's derivative.
        The field maps here already include the physical-leg boost projector.
        """
        from .m2_gaussian_cluster_current import _affine,_multiply,_add_scaled
        if not isinstance(gauged,(bool,np.bool_)):raise ValueError("Boolean gauged selection required")
        fields=np.asarray(field_maps,float)
        if fields.ndim!=2 or not fields.shape[1] or not np.all(np.isfinite(fields)):
            raise ValueError("finite field maps on nonempty integration coordinates required")
        shifts=_array(field_shifts,(len(fields),2),"field shifts")
        d=fields.shape[1];zero=(0,)*(2*d);q={}
        for field,shift in zip(fields,shifts):
            for j in range(2):
                linear=np.zeros((2,d));linear[j]=field
                term=_affine(shift[j],linear)
                _add_scaled(q,_multiply(term,term),1.)
        term={zero:1.};series=term.copy();previous={}
        delta=self.a-self.a_reference
        for m in range(1,self.degree+1):
            previous=series.copy()
            term={p:c*delta/m for p,c in _multiply(term,q).items() if c*delta!=0}
            _add_scaled(series,term,1.)
        if gauged:_add_scaled(series,previous,-delta/self.a)
        return {p:c for p,c in series.items() if c!=0}

    def integrate(self,*,gauged=False,**arguments):
        """Exact joint HO/Gaussian/polynomial moments; no projected products.

        `gauged=True` weights a supplied BASE-Gaussian first derivative.
        The caller still supplies the actual physical-leg derivative and
        endpoint bare vertex. It is not permission to infer J-minus from Ward.
        """
        from .m2_gaussian_cluster_current import _affine,_multiply,_add_scaled
        if not isinstance(gauged,(bool,np.bool_)):raise ValueError("Boolean gauged selection required")
        precision=float(arguments["field_precision"])
        if not np.isclose(precision,2*self.a,atol=0.,rtol=1e-14):
            raise ValueError("reference shape must bind its actual base Gaussian width")
        if self.degree==0 or self.a==self.a_reference:
            return gaussian_shell_integral(**arguments)
        args=dict(arguments);fields=np.asarray(args["field_maps"],float)
        if fields.ndim!=2:raise ValueError("two-dimensional field map required")
        d=fields.shape[1];constant=complex(args.pop("constant",1.))
        if not np.isfinite(constant):raise ValueError("finite numerator required")
        linear=args.pop("linear",None)
        linear=np.zeros((2,d),complex) if linear is None else _array(linear,(2,d),"linear numerator",complex_values=True)
        numerator=_affine(constant,linear)
        quadratic=args.pop("quadratic",None)
        if quadratic is not None:
            matrix=_array(quadratic,(2*d,2*d),"quadratic numerator",complex_values=True)
            if not np.allclose(matrix,matrix.T,atol=1e-14,rtol=1e-13):raise ValueError("symmetric quadratic numerator required")
            for j in range(2*d):
                for k in range(2*d):
                    powers=[0]*(2*d);powers[j]+=1;powers[k]+=1
                    _add_scaled(numerator,{tuple(powers):matrix[j,k]/2},1.)
        supplied=args.pop("polynomial_terms",None)
        if supplied is not None:
            if not hasattr(supplied,"items"):raise ValueError("explicit polynomial mapping required")
            for powers,value in supplied.items():
                if len(powers)!=2*d or any(isinstance(n,(bool,np.bool_)) for n in powers):
                    raise ValueError("one exact nonnegative polynomial power per coordinate required")
                try:powers=tuple(index(n) for n in powers)
                except TypeError as exc:raise ValueError("exact polynomial powers required") from exc
                value=complex(value)
                if min(powers)<0 or not np.isfinite(value):raise ValueError("finite polynomial coefficients and nonnegative powers required")
                _add_scaled(numerator,{powers:value},1.)
        weight=self.weight_polynomial(fields,args["field_shifts"],gauged=gauged)
        return gaussian_shell_integral(**args,constant=0.,polynomial_terms=_multiply(numerator,weight))
