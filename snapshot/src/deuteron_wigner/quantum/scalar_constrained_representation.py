"""Exact endpoint conditions for the numerical scalar reference only.

L=(1-t)/a+t, w=(1-a)^2*t*(1-t)/a.
Lift: g_L=L-w*f, with exact residual f=g.
Projection: g_P=L+H_t[f(a,t)-f(1,t)], where H_t removes endpoint interpolation.
Both fix g(a,0)=1/a, g(a,1)=1, g(1,t)=1 for arbitrary residual coefficients.
These are representations of a known core, not new physical dynamics or a
proof that arbitrary trainable coefficients generate a conserved current.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.chebyshev import chebval

from .scalar_joint_representation import PolynomialCore, RadialSlice


@dataclass(frozen=True)
class ConstrainedSlice(RadialSlice):
    mode: str = "lift"

    def g(self, t):
        # Parent validates and evaluates the stored polynomial, not exact g.
        represented = super().g(t)
        t = np.asarray(t)
        endpoint_line = (1-t)/self.a + t
        if self.mode == "lift":
            multiplier = (1-self.a)**2/self.a * t*(1-t)
            return endpoint_line - multiplier*represented
        if self.mode == "projection":
            return endpoint_line + represented
        raise ValueError("Unknown constrained slice mode.")


class ConstrainedCore:
    """Wrap an immutable polynomial core. No new fitted parameters.

    The projection caches its a=1 radial coefficients; this derived cache is
    included in stored_bytes/counts. Temporary bound slices are not included,
    just as in the earlier benchmark. Neither mode evaluates exact g inside
    the current; the known endpoint lift L is explicit fixed information.
    """

    def __init__(self, core, mode="lift"):
        if not isinstance(core, PolynomialCore):
            raise TypeError("A stored polynomial core is required for this experiment.")
        if mode not in ("lift", "projection"):
            raise ValueError("Choose lift or projection explicitly.")
        self.core = core
        self.mode = mode
        self.anchor = None
        self.stored_coefficients = core.stored_coefficients
        self.stored_bytes = core.stored_bytes
        if mode == "projection":
            self.anchor = np.array(core.bind(1.0).coefficients, copy=True)
            self.anchor.setflags(write=False)
            self.stored_coefficients += self.anchor.size
            self.stored_bytes += self.anchor.nbytes

    def bind(self, a):
        original = self.core.bind(a)  # includes virtuality-domain validation
        if self.mode == "lift":
            return ConstrainedSlice(a, original.coefficients, "lift")
        difference = original.coefficients - self.anchor
        left, right = chebval(-1, difference), chebval(1, difference)
        # Subtract the endpoint line in coefficient space. This is the
        # homogeneous boundary map, not a fitted per-observable correction.
        difference[0] -= (left+right)/2
        difference[1] -= (right-left)/2
        return ConstrainedSlice(a, difference, "projection")
