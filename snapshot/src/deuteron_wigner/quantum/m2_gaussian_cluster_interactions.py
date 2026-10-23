"""Exact common-cluster orbitals for retained and excluded interactions.

Number-preserving contacts factorize ONLY on the momentum-conserving
Hamiltonian slice. Gauge their original physical four-leg kernel, not
auxiliary factorization legs. Spectators are never field-regulated.
"""
from functools import lru_cache
from math import pi, sqrt,factorial
import numpy as np

from .m2_gaussian_field_regulator import GaussianClusterRegulator, gaussian_shell_integral
from .m2_coupled_longitudinal_jacobi import (
    jacobi_frame, oscillator_shell, oscillator_rotation, emission_coordinate_map)
from .m2_coupled_pair_action import _pair, _nonnegative, _width, _contact_shell
from .m2_excluded_fock_kinematics import SPECIES as FIVE, frame as five_frame, split_frame as five_split
from .m2_six_body_kinematics import SPECIES as SIX, frame as six_frame
from .m2_six_body_vertices import split_frame as six_split, cubic_mass_kernel as six_kernel
from .m2_canonical_emission import emission_mass_kernel_per_gs
from .m2_gaussian_field_vertices import _field_data, _integrate


def _regulator(value):
    if not isinstance(value, GaussianClusterRegulator):
        raise ValueError("explicit boost-invariant active-cluster regulator required")
    return value


def _frame(modes, species):
    modes, species = tuple(modes), tuple(species)
    if len(modes) != len(species):
        raise ValueError("one physical species per mode required")
    if species in (("q",)*3, ("q",)*3+("g",)):
        return jacobi_frame(modes)
    for name, kinds in FIVE.items():
        if species == kinds:
            return five_frame(name, modes)
    for name, kinds in SIX.items():
        if species == kinds:
            return six_frame(name, modes)
    raise ValueError("canonical ordered retained/five/six-body sector required")


@lru_cache(maxsize=8192)
def pair_coordinate_map(modes, species, pair):
    f = _frame(modes, species)
    p, _ = _pair(pair, len(modes))
    rest = tuple(i for i in range(len(modes)) if i not in p)
    other = rest[0]
    for j in rest[1:]:
        other = (other, j)
    rows = []
    def visit(node):
        if not isinstance(node, tuple):
            return (node,)
        left, right = map(visit, node)
        x, y = f.fractions[list(left)].sum(), f.fractions[list(right)].sum()
        row = np.zeros(len(modes))
        row[list(left)], row[list(right)] = sqrt(y/(x*(x+y))), -sqrt(x/(y*(x+y)))
        rows.append(row)
        return left+right
    visit((p, other))
    r = np.array(rows)@f.inverse[:, :-1]
    if np.linalg.norm(r@r.T-np.eye(len(modes)-1)) > 3e-12:
        raise ArithmeticError("pair/spectator coordinate map is not orthogonal")
    r.setflags(write=False)
    return r


def _pair_data(incoming, outgoing, pair, b_GeV, in_excitation, out_excitation,
               regulator, species, out_species):
    reg = _regulator(regulator)
    si = tuple(species)
    so = si if out_species is None else tuple(out_species)
    fi, fo = _frame(incoming, si), _frame(outgoing, so)
    p, _ = _pair(pair, len(incoming))
    if len(incoming) != len(outgoing) or sum(incoming) != sum(outgoing):
        raise ValueError("common particle count and total K2 required")
    if any((incoming[j], si[j]) != (outgoing[j], so[j])
           for j in range(len(incoming)) if j not in p):
        raise ValueError("longitudinal and species spectators must be conserved")
    b = _width(b_GeV)
    ci = _nonnegative(in_excitation, "input shell")
    co = ci if out_excitation is None else _nonnegative(out_excitation, "output shell")
    eligible = all(reg.contains(kinds[j], modes[j])
        for kinds, modes in ((si, incoming), (so, outgoing)) for j in p)
    i, j = p
    x = fi.fractions[i]+fi.fractions[j]
    jac = sqrt(fi.fractions[i]*fi.fractions[j]*fo.fractions[i]*fo.fractions[j])/x
    ri = pair_coordinate_map(tuple(incoming), si, p)
    ro = pair_coordinate_map(tuple(outgoing), so, p)
    return reg, fi, fo, si, so, p, b, ci, co, eligible, x, jac, ri, ro


@lru_cache(maxsize=512)
def _weighted_contact_shell(d, co, ci, t):
    def weights(cutoff):
        return np.array([((1-t)/(1+t))**state[0]/(1+t)
                         for state in oscillator_shell(d, cutoff)])
    result = _contact_shell(d, co, ci)*weights(co)[:, None]*weights(ci)[None, :]
    result.setflags(write=False)
    return result


@lru_cache(maxsize=512)
def _shape_contact_shell(d,co,ci,t,scale,shape):
    """Correlated derivative sum of the WHOLE physical quartic kernel."""
    from .m2_gaussian_reference_shape import radial_half_moment
    delta=shape.a-shape.a_reference;M=shape.degree
    if M==0 or delta==0:return _weighted_contact_shell(d,co,ci,t)
    def moments(cutoff):
        return [np.array([radial_half_moment(state[0],t,u,scale) for state in oscillator_shell(d,cutoff)])
                for u in range(M+1)]
    out,inc=moments(co),moments(ci)
    weights=sum(delta**(u+v)/(factorial(u)*factorial(v))*out[u][:,None]*inc[v][None,:]
                for u in range(M+1) for v in range(M+1-u))
    result=_contact_shell(d,co,ci)*weights;result.setflags(write=False)
    return result


def cluster_pair_contact(incoming, outgoing, pair, b_GeV, in_excitation,
                         out_excitation=None, *, regulator, species, out_species=None):
    """Exact continuum-projected quartic contact, including the Jacobi measure."""
    reg, fi, fo, si, so, p, b, ci, co, eligible, x, jac, ri, ro = _pair_data(
        incoming, outgoing, pair, b_GeV, in_excitation, out_excitation, regulator, species, out_species)
    d = len(incoming)-1
    if not eligible:
        return np.zeros((len(oscillator_shell(d, co)), len(oscillator_shell(d, ci))))
    contact = _weighted_contact_shell(d, co, ci, x*b*b/reg.width_GeV**2)
    if reg.reference_shape is not None:
        contact=_shape_contact_shell(d,co,ci,x*b*b/reg.width_GeV**2,2*x*b*b,reg.reference_shape)
    return b*b/pi*jac*(oscillator_rotation(ro, co).T@contact@oscillator_rotation(ri, ci))


def cluster_pair_integral_reference(incoming, outgoing, pair, b_GeV, in_excitation,
                                    out_excitation=None, *, regulator, species,
                                    out_species=None, total_transverse_GeV):
    """Independent direct Gaussian integration before radial factorization."""
    reg, fi, fo, si, so, p, b, ci, co, eligible, x, jac, ri, ro = _pair_data(
        incoming, outgoing, pair, b_GeV, in_excitation, out_excitation, regulator, species, out_species)
    pt = np.asarray(total_transverse_GeV, float)
    if pt.shape != (2,) or not np.all(np.isfinite(pt)):
        raise ValueError("finite explicit transverse pair required")
    d = len(incoming)-1
    outgoing_map = np.column_stack((np.eye(d), np.zeros(d)))
    incoming_map = outgoing_map.copy()
    incoming_map[0] = 0.
    incoming_map[0, -1] = 1.
    lo, li = ro.T@outgoing_map, ri.T@incoming_map
    fields = np.vstack((fo.inverse[list(p), :-1]@lo, fi.inverse[list(p), :-1]@li))
    modes = tuple(outgoing[j] for j in p)+tuple(incoming[j] for j in p)
    transform = reg.leg_transform(modes)
    fractions = np.r_[fo.fractions[list(p)], fi.fractions[list(p)]]
    if not eligible:
        return np.zeros((len(oscillator_shell(d, co)), len(oscillator_shell(d, ci))))
    return jac*reg.integrate(out_cutoff=co, in_cutoff=ci, out_map=lo, in_map=li,
        b_GeV=b, field_precision=1/reg.width_GeV**2,
        field_maps=transform@fields, field_shifts=transform@(fractions[:, None]*pt))


def cluster_retained_emission(route, cutoff, b, mass, hq, hi, hg, *,
                              regulator, total_pplus_GeV):
    reg = _regulator(regulator)
    f, tree = jacobi_frame(route.outgoing), emission_coordinate_map(route)
    fields, shifts, precision, eligible = _field_data(f, (route.emitter, 3),
        route.incoming[route.emitter], "q", regulator=reg,
        total_pplus_GeV=total_pplus_GeV, total_transverse_GeV=(0., 0.), species=("q",)*3+("g",))
    def kernel(q):
        return emission_mass_kernel_per_gs(route, q, mass, hq, hi, hg, total_pplus=total_pplus_GeV)
    c = kernel((0., 0.))
    linear = np.array([(kernel((1., 0.))-c)*tree[-1], (kernel((0., 1.))-c)*tree[-1]])
    x, y = f.fractions[[route.emitter, 3]]
    return sqrt(x*y/(x+y))*_integrate(tree[:2], in_cutoff=cutoff, out_cutoff=cutoff,
        b=b, fields=fields, shifts=shifts, precision=precision, eligible=eligible, constant=c, linear=linear,
        reference_shape=reg.reference_shape)


def cluster_six_cubic(route, in_cutoff, out_cutoff, b, mass, helicities, *,
                      regulator, total_pplus_GeV):
    reg = _regulator(regulator)
    f, tree = six_frame(route.sector, route.outgoing), six_split(route)
    fields, shifts, precision, eligible = _field_data(f, route.daughters,
        route.incoming[route.parent], FIVE[route.source_sector][route.parent],
        regulator=reg, total_pplus_GeV=total_pplus_GeV, total_transverse_GeV=(0., 0.),
        species=SIX[route.sector])
    def kernel(q):
        return six_kernel(route, q, mass, helicities, total_pplus=total_pplus_GeV)
    c = kernel((0., 0.))
    linear = np.array([(kernel((1., 0.))-c)*tree[-1], (kernel((0., 1.))-c)*tree[-1]])
    x, y = f.fractions[list(route.daughters)]
    return sqrt(x*y/(x+y))*_integrate(tree[:4], in_cutoff=in_cutoff, out_cutoff=out_cutoff,
        b=b, fields=fields, shifts=shifts, precision=precision, eligible=eligible, constant=c, linear=linear,
        reference_shape=reg.reference_shape)


def cluster_pair_current_orbital(route, cutoff, b_GeV, *, regulator,derivative_order=0):
    """Hamiltonian factorization only: auxiliary current is NOT a field leg."""
    reg = _regulator(regulator)
    if reg.reference_shape is not None:
        raise ValueError("a whole reference-shape vertex requires the correlated auxiliary pair sum")
    derivative_order=_nonnegative(derivative_order,"half-kernel derivative order")
    f, tree = five_frame(route.sector, route.outgoing), five_split(route)
    x, y = f.fractions[3:]
    b = _width(b_GeV)
    outgoing, incoming = oscillator_shell(4, cutoff), oscillator_shell(3, cutoff)
    result = np.zeros((len(outgoing), len(incoming)))
    # Only the two physical daughters belong to this half of the quartic
    # envelope. The spin-zero auxiliary mode is not bounded as a gluon.
    if not all(reg.contains(FIVE[route.sector][j], route.outgoing[j]) for j in (3, 4)):
        return result
    lookup = {s:i for i,s in enumerate(incoming)}
    t = (x+y)*b*b/reg.width_GeV**2
    for row,s in enumerate(outgoing):
        inherited = s[:3]+s[4:7]
        if s[3] == s[7] and inherited in lookup:
            if derivative_order:
                from .m2_gaussian_reference_shape import radial_half_moment
                weight=radial_half_moment(s[3],t,derivative_order,2*(x+y)*b*b)
            else:weight=((1-t)/(1+t))**s[3]/(1+t)
            result[row,lookup[inherited]] = b/sqrt(pi)*weight
    return sqrt(x*y/(x+y))*oscillator_rotation(tree, cutoff).T@result
