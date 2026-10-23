"""Exact Gaussian interaction integrals on complete circular-oscillator shells.

The regulator multiplies PARTICIPATING fields, not spectator Fock legs.
Independent field-leg smearing is a finite-frame diagnostic; the correlated
active-cluster prescription preserves light-front kinematic boosts. Neither
is a proved Lorentz/BRST-preserving continuum regulator or physical fit.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import factorial, pi
from operator import index

import numpy as np

from .m2_coupled_longitudinal_jacobi import oscillator_shell


@dataclass(frozen=True, kw_only=True)
class GaussianFieldRegulator:
    width_GeV: float
    boson_modes: int
    fermion_modes: int
    reference_transverse_velocity: tuple

    def __post_init__(self):
        width = float(self.width_GeV)
        velocity = np.asarray(self.reference_transverse_velocity, float)
        if not np.isfinite(width) or width <= 0 or velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("positive finite width and explicit finite two-component reference velocity required")
        for n in (self.boson_modes, self.fermion_modes):
            if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
                raise ValueError("positive exact longitudinal field-mode counts required")
        object.__setattr__(self, "width_GeV", width)
        object.__setattr__(self, "reference_transverse_velocity", tuple(velocity))

    def contains(self, species, mode):
        if isinstance(mode, (bool, np.bool_)):
            raise ValueError("exact positive doubled field mode required")
        p = index(mode)
        if species not in ("q", "qbar", "g") or p <= 0 or p % 2 != int(species != "g"):
            raise ValueError("positive APBC fermion or PBC gluon field mode required")
        return p <= (2*self.boson_modes if species == "g" else 2*self.fermion_modes-1)

    @property
    def coincident_trace_GeV2(self):
        """Integral d²k/(2pi)² r(k)², not an unrelated Wick regulator."""
        return self.width_GeV**2/(4*pi)

    def amplitude(self, species, mode, transverse_momentum, *, pplus_GeV):
        k, pp = np.asarray(transverse_momentum, float), float(pplus_GeV)
        if k.shape[-1:] != (2,) or not np.all(np.isfinite(k)) or not np.isfinite(pp) or pp <= 0:
            raise ValueError("finite transverse momenta and positive p+ required")
        eligible = self.contains(species, mode)
        relative = k-pp*np.asarray(self.reference_transverse_velocity)
        return eligible*np.exp(-np.sum(relative**2, axis=-1)/(2*self.width_GeV**2))




@dataclass(frozen=True, kw_only=True)
class GaussianClusterRegulator:
    """One correlated ordered-field kernel, invariant under LF transverse boosts.

    With positive physical leg modes w (creation/annihilation both included),
    F=exp[-Q/(2 Lambda²)], Q=(sum(w)*sum(pT²/w)-|sum(pT)|²)/2.
    On momentum-conserving vertices Q=M_in,T²+M_out,T² of active legs.
    This is not a product of separate one-field smearing amplitudes.
    """
    width_GeV: float
    boson_modes: int
    fermion_modes: int
    reference_shape: object=None

    def __post_init__(self):
        checked = GaussianFieldRegulator(width_GeV=self.width_GeV,
            boson_modes=self.boson_modes, fermion_modes=self.fermion_modes,
            reference_transverse_velocity=(0., 0.))
        object.__setattr__(self, "width_GeV", checked.width_GeV)
        self.shape_for()

    contains = GaussianFieldRegulator.contains

    def shape_for(self,explicit=None):
        """Bind one whole-vertex shape across H, Wick and physical currents."""
        shape=self.reference_shape if explicit is None else explicit
        if shape is not None:
            from .m2_gaussian_reference_shape import GaussianReferenceShape
            if not isinstance(shape,GaussianReferenceShape) or shape.base_width_GeV!=self.width_GeV:
                raise ValueError("reference shape must bind the actual base Gaussian width")
            if self.reference_shape is not None and shape!=self.reference_shape:
                raise ValueError("current and Hamiltonian reference shapes must be identical")
        return shape

    def integrate(self,**arguments):
        """The explicitly selected whole kernel before oscillator projection."""
        integral=gaussian_shell_integral if self.reference_shape is None else self.reference_shape.integrate
        return integral(**arguments)

    def leg_transform(self, modes):
        """Q=|B pT|², with the collective transverse-boost direction removed."""
        w = np.asarray(modes, float)
        if w.ndim != 1 or len(w) < 2 or not np.all(np.isfinite(w)) or np.min(w) <= 0:
            raise ValueError("positive physical modes of every participating field leg required")
        u = np.sqrt(w/w.sum())
        return np.sqrt(w.sum()/2)*(np.eye(len(w))-np.outer(u, u))/np.sqrt(w)[None, :]

    def contraction_trace_GeV2(self, *, external_mode, loop_mode):
        """Same quartic kernel after Wick: integral F(p,k;p,k)=Lambda² k/(4pi p).

        The loop Gaussian is centered at kT=(k+/p+) pT. The coincident
        trace is mode dependent; substituting Lambda²/(4pi) is incorrect.
        Mode eligibility and signed constrained denominators remain separate.
        """
        p, k = float(external_mode), float(loop_mode)
        if not np.all(np.isfinite([p, k])) or min(p, k) <= 0:
            raise ValueError("positive external and loop modes required")
        trace=self.width_GeV**2*k/(4*pi*p)
        if self.reference_shape is not None:
            s=self.reference_shape;r=1-s.a_reference/s.a
            trace*=sum(r**m for m in range(s.degree+1))
        return trace

    def contracted_kernel_trace(self, *, external_mode, loop_mode,
                                outgoing_transverse_GeV, incoming_transverse_GeV):
        """Integrate F(p_out,k;p_in,k) before imposing the momentum delta.

        This specifies the off-conservation extension of the ordered kernel,
        not an electromagnetic current or a new physical form factor.
        Longitudinal modes at the two external endpoints are equal here.
        """
        trace = self.contraction_trace_GeV2(external_mode=external_mode, loop_mode=loop_mode)
        out = _array(outgoing_transverse_GeV, (2,), "outgoing transverse momentum")
        inc = _array(incoming_transverse_GeV, (2,), "incoming transverse momentum")
        p, k = float(external_mode), float(loop_mode)
        if self.reference_shape is not None:
            from .m2_gaussian_cluster_wick_current import _shape_path_moment
            q0=(p+k)*np.sum((out-inc)**2)/(2*p)
            return float(k/(8*pi*p)*_shape_path_moment(0,q0,0.,self.reference_shape,inverse_a=True))
        return float(trace*np.exp(-(p+k)*np.sum((out-inc)**2)/(4*p*self.width_GeV**2)))

    def contracted_kernel_derivatives(self, *, external_mode, loop_mode,
                                      outgoing_transverse_GeV, incoming_transverse_GeV):
        """Analytic derivatives of the same integrated off-conservation kernel.

        Differentiate at fixed loop/external physical momenta and fixed
        envelope membership. These are smooth kernel derivatives, NOT the
        gauging of the finite longitudinal envelope or a complete J-minus.
        """
        value = self.contracted_kernel_trace(external_mode=external_mode, loop_mode=loop_mode,
            outgoing_transverse_GeV=outgoing_transverse_GeV,
            incoming_transverse_GeV=incoming_transverse_GeV)
        p, k = float(external_mode), float(loop_mode)
        delta = np.asarray(outgoing_transverse_GeV)-np.asarray(incoming_transverse_GeV)
        if self.reference_shape is not None:
            D=float(delta@delta);q0=(p+k)*D/(2*p)
            field=self.reference_shape.value_and_derivative(q0)[0]
            coefficient=k/(8*pi*p)*field
            transverse=-coefficient*(p+k)/p*delta
            return dict(external_mode=-value/p+coefficient*k*D/(2*p*p),
                        loop_mode=value/k-coefficient*D/(2*p),
                        outgoing_transverse=transverse,incoming_transverse=-transverse)
        a = np.sum(delta**2)/(4*self.width_GeV**2)
        transverse = -value*(p+k)/(2*p*self.width_GeV**2)*delta
        return dict(external_mode=value*(-1/p+k*a/p**2),
                    loop_mode=value*(1/k-a/p),
                    outgoing_transverse=transverse, incoming_transverse=-transverse)

def _array(value, shape, name, *, complex_values=False):
    a = np.asarray(value, complex if complex_values else float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError("finite "+name+" with shape "+str(shape)+" required")
    return a


def gaussian_shell_integral(*, out_cutoff, in_cutoff, out_map, in_map,
                            b_GeV, field_precision, field_maps, field_shifts,
                            out_shift=None, in_shift=None, constant=1.,
                            linear=None, quadratic=None, polynomial_terms=None):
    """Integrate bra HO(Lo q+so)* polynomial(q) ket HO(Li q+si) times fields.

    q=(qx,qy) has d integration coordinates in EACH transverse component.
    Measure is product d²q/(2pi)²; every HO is normalized in that measure.
    Field factor: exp[-precision/2 sum_legs |Lleg q+sleg|²].
    Polynomial: constant + linear.flatten()·(qx,qy)
                + (qx,qy)^T quadratic (qx,qy)/2.
    polynomial_terms adds explicit {power_tuple: coefficient} monomials
    in flattened (qx,qy). Exact joint Gaussian moments retain higher
    degree insertions from gauging the regulator, before projection.
    No truncated multiplication of separately projected operators is used.
    """
    lo, li = np.asarray(out_map, float), np.asarray(in_map, float)
    if lo.ndim != 2 or li.ndim != 2 or not lo.shape[0] or not li.shape[0] or lo.shape[1] != li.shape[1]:
        raise ValueError("nonempty outgoing/incoming maps on one integration coordinate system required")
    no, d, ni = lo.shape[0], lo.shape[1], li.shape[0]
    _array(lo, (no, d), "outgoing map")
    _array(li, (ni, d), "incoming map")
    b, precision = float(b_GeV), float(field_precision)
    if not np.all(np.isfinite([b, precision])) or b <= 0 or precision < 0:
        raise ValueError("positive oscillator width and nonnegative explicit field precision required")
    fields = np.asarray(field_maps, float)
    if fields.ndim != 2 or fields.shape[1] != d or not np.all(np.isfinite(fields)):
        raise ValueError("field leg maps on the common integration coordinates required")
    shifts = _array(field_shifts, (len(fields), 2), "field shifts")
    so = np.zeros((no, 2)) if out_shift is None else _array(out_shift, (no, 2), "outgoing shift")
    si = np.zeros((ni, 2)) if in_shift is None else _array(in_shift, (ni, 2), "incoming shift")
    states_o, states_i = oscillator_shell(no, out_cutoff), oscillator_shell(ni, in_cutoff)
    # The Gaussian covariance is exact on the continuum integration space.
    c = (lo.T@lo+li.T@li)/b**2+precision*(fields.T@fields)
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the declared integration has an unconstrained Gaussian direction") from exc
    cv = np.linalg.solve(c, np.eye(d))
    h = -(lo.T@so+li.T@si)/b**2-precision*fields.T@shifts
    mu = cv@h
    exponent = -(np.sum(so**2)+np.sum(si**2))/(2*b*b)-precision*np.sum(shifts**2)/2
    exponent += np.sum(h*mu)/2
    lognorm = ((no+ni)*np.log(np.sqrt(4*pi)/b)-d*np.log(2*pi)
               -2*np.sum(np.log(np.diag(chol)))+exponent)
    prefactor = np.exp(lognorm)
    # Variable order matches oscillator_shell: out+, out-, in+, in-.
    bx = np.column_stack((lo.T, lo.T, li.T, li.T))/b
    by = np.column_stack((-1j*lo.T, 1j*lo.T, 1j*li.T, -1j*li.T))/b
    m = bx.T@cv@bx+by.T@cv@by
    for offset, count in ((0, no), (2*no, ni)):
        for j in range(count):
            m[offset+j, offset+count+j] -= 1
            m[offset+count+j, offset+j] -= 1
    ell = np.concatenate((so[:, 0]-1j*so[:, 1], so[:, 0]+1j*so[:, 1],
                          si[:, 0]+1j*si[:, 1], si[:, 0]-1j*si[:, 1]))/b
    ell += bx.T@mu[:, 0]+by.T@mu[:, 1]
    value = complex(constant)
    lin = np.zeros(2*d, complex) if linear is None else _array(
        linear, (2, d), "linear polynomial", complex_values=True).reshape(-1)
    quad = np.zeros((2*d, 2*d), complex) if quadratic is None else _array(
        quadratic, (2*d, 2*d), "quadratic polynomial", complex_values=True)
    if not np.isfinite(value) or not np.allclose(quad, quad.T, atol=1e-14, rtol=1e-13):
        raise ValueError("finite constant and symmetric quadratic polynomial required")
    v = np.kron(np.eye(2), cv)
    mean = mu.T.reshape(-1)
    cb = np.vstack((cv@bx, cv@by))
    p0 = value+lin@mean+(mean@quad@mean+np.trace(quad@v))/2
    p1 = (lin+mean@quad)@cb
    p2 = cb.T@quad@cb
    insertions = []
    if polynomial_terms is not None:
        if not hasattr(polynomial_terms, "items"):
            raise ValueError("explicit polynomial power-to-coefficient mapping required")
        for powers, coefficient in polynomial_terms.items():
            if len(powers) != 2*d or any(isinstance(n, (bool, np.bool_)) for n in powers):
                raise ValueError("one nonnegative exact power per integration coordinate required")
            powers = tuple(index(n) for n in powers)
            coefficient = complex(coefficient)
            if min(powers) < 0 or not np.isfinite(coefficient):
                raise ValueError("nonnegative powers and finite monomial coefficient required")
            insertions.append((tuple(j for j,n in enumerate(powers) for _ in range(n)), coefficient))

    @lru_cache(None)
    def moment(power):
        if not any(power):
            return 1.+0j
        j = next(k for k, n in enumerate(power) if n)
        lower = list(power)
        lower[j] -= 1
        result = ell[j]*moment(tuple(lower))
        for k, n in enumerate(lower):
            if n:
                rest = lower.copy()
                rest[k] -= 1
                result += n*m[j, k]*moment(tuple(rest))
        return result

    inserted_coefficients = None
    if insertions and len(states_o)*len(states_i) < 16:
        # For tiny matrices the identical scalar recurrence has less Python/
        # NumPy setup overhead. This selects an evaluation algorithm, not an
        # oscillator cutoff or a tolerance: every requested entry is retained.
        @lru_cache(None)
        def inserted_moment(power, coordinates):
            if not coordinates:
                return moment(power)
            j, rest = coordinates[0], coordinates[1:]
            value = mean[j]*inserted_moment(power, rest)
            for k, n in enumerate(power):
                if n:
                    lower = list(power); lower[k] -= 1
                    value += n*cb[j,k]*inserted_moment(tuple(lower), rest)
            for pos,k in enumerate(rest):
                value += v[j,k]*inserted_moment(power, rest[:pos]+rest[pos+1:])
            return value
        inserted_coefficients = np.array([
            sum(c*inserted_moment(a+bst,q) for q,c in insertions)
            for a in states_o for bst in states_i])
    elif insertions:
        # Evaluate the SAME Gaussian insertion recurrence over all requested
        # bra/ket powers together. Complete rectangular shells are downward
        # closed under lowering any one oscillator power. Thus every lower
        # index refers to an actual requested matrix element; no extra shell,
        # polynomial truncation, covariance approximation or tolerance enters.
        powers = tuple(a+bst for a in states_o for bst in states_i)
        lookup = {p:i for i,p in enumerate(powers)}
        base_moments = np.array([moment(p) for p in powers])
        lowered = []
        for k in range(len(ell)):
            rows, columns, weights = [], [], []
            for row,p in enumerate(powers):
                if p[k]:
                    low = list(p); low[k] -= 1
                    rows.append(row); columns.append(lookup[tuple(low)]); weights.append(p[k])
            if rows:
                lowered.append((k,np.array(rows),np.array(columns),np.array(weights)))

        @lru_cache(None)
        def inserted_moments(coordinates):
            if not coordinates:
                return base_moments
            j, rest = coordinates[0], coordinates[1:]
            previous = inserted_moments(rest)
            value = mean[j]*previous
            for k,rows,columns,weights in lowered:
                value[rows] += weights*cb[j,k]*previous[columns]
            for pos,k in enumerate(rest):
                value += v[j,k]*inserted_moments(rest[:pos]+rest[pos+1:])
            return value

        inserted_coefficients = sum(c*inserted_moments(q) for q,c in insertions)

    result = np.zeros((len(states_o), len(states_i)), complex)
    for row, a in enumerate(states_o):
        for col, bstate in enumerate(states_i):
            power = a+bstate
            coefficient = p0*moment(power)
            if inserted_coefficients is not None:
                coefficient += inserted_coefficients[row*len(states_i)+col]
            occupied = [j for j, n in enumerate(power) if n]
            for j in occupied:
                lower = list(power)
                lower[j] -= 1
                coefficient += power[j]*p1[j]*moment(tuple(lower))
                for k in occupied:
                    multiplicity = power[j]*lower[k]
                    if multiplicity:
                        second = lower.copy()
                        second[k] -= 1
                        coefficient += .5*multiplicity*p2[j, k]*moment(tuple(second))
            result[row, col] = prefactor*coefficient/np.sqrt(float(np.prod([factorial(n) for n in power])))
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("nonfinite Gaussian shell integral")
    return result


def gaussian_field_direction_vertex(*, direction, transfer_GeV, quadrature_order,
                              out_cutoff, in_cutoff, out_map, in_map,
                              b_GeV, field_precision, field_maps, field_shifts,
                              out_shift=None, in_shift=None, constant=1., linear=None):
    """Straight-path derivative along an explicitly declared field direction.

    Contracting with transfer gives the difference of the complete
    Gaussian source integrals with shifts+s*direction*transfer. The
    canonical polynomial is held fixed here: its current is separate.
    Physical charge and the external 1/(2P+) are not included.
    """
    delta = _array(transfer_GeV, (2,), "transverse transfer")
    if isinstance(quadrature_order, (bool, np.bool_)):
        raise ValueError("exact path quadrature order required")
    order = index(quadrature_order)
    fields, shifts = np.asarray(field_maps, float), np.asarray(field_shifts, float)
    if fields.ndim != 2 or order < 2:
        raise ValueError("field maps and path order at least two required")
    direction = _array(direction, (len(fields),), "field shift direction")
    args = dict(out_cutoff=out_cutoff, in_cutoff=in_cutoff, out_map=out_map, in_map=in_map,
        b_GeV=b_GeV, field_precision=field_precision, field_maps=fields, field_shifts=shifts,
        out_shift=out_shift, in_shift=in_shift)
    base = gaussian_shell_integral(**args, constant=constant, linear=linear)
    d = fields.shape[1]
    lin = np.zeros(2*d, complex) if linear is None else _array(
        linear, (2, d), "linear polynomial", complex_values=True).reshape(-1)
    precision = float(field_precision)
    result = np.zeros((2, *base.shape), complex)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for node, weight in zip((nodes+1)/2, weights/2):
        shifted = shifts.copy()
        shifted += node*direction[:, None]*delta
        for component in (0, 1):
            u = np.zeros(2*d)
            u[component*d:(component+1)*d] = direction@fields
            offset = direction@shifted[:, component]
            c0 = -precision*offset*constant
            c1 = -precision*(offset*lin+constant*u)
            c2 = -precision*(np.outer(u, lin)+np.outer(lin, u))
            result[component] += weight*gaussian_shell_integral(
                **{**args, "field_shifts":shifted}, constant=c0,
                linear=c1.reshape(2, d), quadratic=c2)
    return result


def gaussian_field_leg_vertex(*, leg, **kwargs):
    """One independent field occurrence; correlated physical legs use B[:,leg]."""
    if isinstance(leg, (bool, np.bool_)):
        raise ValueError("exact field leg index required")
    k = index(leg)
    fields = np.asarray(kwargs["field_maps"])
    if fields.ndim != 2 or not 0 <= k < len(fields):
        raise ValueError("existing field leg required")
    direction = np.zeros(len(fields))
    direction[k] = 1.
    return gaussian_field_direction_vertex(direction=direction, **kwargs)
