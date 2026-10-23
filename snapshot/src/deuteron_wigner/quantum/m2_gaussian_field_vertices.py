"""Participating-field Gaussian factors inside canonical source integrals.

The diagnostic family multiplies independent field amplitudes. The active-
cluster family applies one correlated kernel to all participating legs.
Its full Wick trace depends on external/loop longitudinal modes. Spectators
carry identity, not smearing factors. These functions supply source ORBITALS,
not a complete regulated Hamiltonian/current model.
"""
from math import sqrt
import numpy as np

from .m2_gaussian_field_regulator import GaussianFieldRegulator, GaussianClusterRegulator, gaussian_shell_integral
from .m2_coupled_longitudinal_jacobi import oscillator_shell
from .m2_excluded_fock_kinematics import SPECIES as FIVE, frame as five_frame, split_frame
from .m2_excluded_fock_vertices import cubic_mass_kernel
from .m2_instantaneous_pair_exit import _route, pair_emission_frame
from .m2_six_body_kinematics import SPECIES as SIX, frame as six_frame
from .m2_six_body_quartic import double_split_frame


def _field_data(frame, daughters, parent_mode, parent_species, *, regulator,
                total_pplus_GeV, total_transverse_GeV, species):
    pp = float(total_pplus_GeV)
    pt = np.asarray(total_transverse_GeV, float)
    if not np.isfinite(pp) or pp <= 0 or pt.shape != (2,) or not np.all(np.isfinite(pt)):
        raise ValueError("positive total p+ and explicit finite total transverse momentum required")
    children = frame.inverse[list(daughters), :-1]
    maps = np.vstack((children, children.sum(axis=0)))
    fractions = frame.fractions[list(daughters)]
    fractions = np.append(fractions, fractions.sum())
    if regulator is None:
        return maps, np.zeros((len(maps), 2)), 0., True
    if not isinstance(regulator, (GaussianFieldRegulator, GaussianClusterRegulator)):
        raise ValueError("explicit common Gaussian field regulator or bare None comparison required")
    modes = tuple(frame.modes[j] for j in daughters)+(parent_mode,)
    kinds = tuple(species[j] for j in daughters)+(parent_species,)
    eligible = all(regulator.contains(s, n) for s, n in zip(kinds, modes))
    if isinstance(regulator, GaussianClusterRegulator):
        transform = regulator.leg_transform(modes)
        return transform@maps, transform@(fractions[:, None]*pt), 1/regulator.width_GeV**2, eligible
    shift = fractions[:, None]*(pt-pp*np.asarray(regulator.reference_transverse_velocity))
    return maps, shift, 1/regulator.width_GeV**2, eligible


def _integrate(in_map, *, in_cutoff, out_cutoff, b, fields, shifts, precision,
               eligible, constant=1., linear=None,reference_shape=None):
    no = in_map.shape[1]
    if not eligible:
        return np.zeros((len(oscillator_shell(no, out_cutoff)),
                         len(oscillator_shell(len(in_map), in_cutoff))), complex)
    integral=gaussian_shell_integral if reference_shape is None else reference_shape.integrate
    return integral(out_cutoff=out_cutoff, in_cutoff=in_cutoff,
        out_map=np.eye(no), in_map=in_map, b_GeV=b, field_precision=precision,
        field_maps=fields, field_shifts=shifts, constant=constant, linear=linear)


def cubic_field_orbital(route, in_cutoff, out_cutoff, b, mass, helicities, *,
                         regulator, total_pplus_GeV, total_transverse_GeV):
    f, tree = five_frame(route.sector, route.outgoing), split_frame(route)
    data = _field_data(f, route.daughters, route.incoming[route.parent],
        "q" if route.channel == "q_to_qg" else "g", regulator=regulator,
        total_pplus_GeV=total_pplus_GeV, total_transverse_GeV=total_transverse_GeV,
        species=FIVE[route.sector])
    fields, shifts, precision, eligible = data
    args = dict(total_pplus=total_pplus_GeV)
    c0 = cubic_mass_kernel(route, (0., 0.), mass, helicities, **args)
    cx = cubic_mass_kernel(route, (1., 0.), mass, helicities, **args)-c0
    cy = cubic_mass_kernel(route, (0., 1.), mass, helicities, **args)-c0
    linear = np.array([cx*tree[-1], cy*tree[-1]])
    x, y = f.fractions[list(route.daughters)]
    return sqrt(x*y/(x+y))*_integrate(tree[:3], in_cutoff=in_cutoff,
        out_cutoff=out_cutoff, b=b, fields=fields, shifts=shifts,
        precision=precision, eligible=eligible, constant=c0, linear=linear,
        reference_shape=getattr(regulator,"reference_shape",None))


def pair_field_orbital(incoming, outgoing, quark, in_cutoff, out_cutoff, b, *,
                        sector, regulator, total_pplus_GeV, total_transverse_GeV):
    f = _route(incoming, outgoing, quark, sector)
    tree = pair_emission_frame(incoming, outgoing, quark, sector=sector)
    daughters = (quark, 3, 4)
    fields, shifts, precision, eligible = _field_data(f, daughters, incoming[quark], "q",
        regulator=regulator, total_pplus_GeV=total_pplus_GeV,
        total_transverse_GeV=total_transverse_GeV, species=FIVE[sector])
    x, y, z = f.fractions[list(daughters)]
    return sqrt(x*y*z/(x+y+z))*_integrate(tree[:2], in_cutoff=in_cutoff,
        out_cutoff=out_cutoff, b=b, fields=fields, shifts=shifts,
        precision=precision, eligible=eligible,reference_shape=getattr(regulator,"reference_shape",None))


def direct_field_orbital(route, in_cutoff, out_cutoff, b, *,
                          regulator, total_pplus_GeV, total_transverse_GeV):
    f, tree = six_frame(route.sector, route.outgoing), double_split_frame(route)
    fields, shifts, precision, eligible = _field_data(f, route.daughters,
        route.incoming[route.emitter], "q" if route.emitter < 3 else "g",
        regulator=regulator, total_pplus_GeV=total_pplus_GeV,
        total_transverse_GeV=total_transverse_GeV, species=SIX[route.sector])
    x, y, z = f.fractions[list(route.daughters)]
    return sqrt(x*y*z/(x+y+z))*_integrate(tree[:3], in_cutoff=in_cutoff,
        out_cutoff=out_cutoff, b=b, fields=fields, shifts=shifts,
        precision=precision, eligible=eligible,reference_shape=getattr(regulator,"reference_shape",None))
