"""Charged physical-leg endpoint product rule for one common cluster kernel.

At Delta+=0 choose explicitly J_Lambda=F0 J_can + sum_a e_a V_a K_a/(2P+).
V_a is the bare Hamiltonian at the leg's shifted ENDPOINT, while K_a
gauges F along that physical leg. K_T=integral partial_pT F and
K_minus=-integral partial_p+ F. This recovers the canonical spin current
in the bare limit, not just its longitudinal Ward projection.

This is a specified nonlocal-current representative, not unique physical
matching. Smooth plus derivatives hold envelope membership fixed; the
finite longitudinal envelope and complete gauged contractions are separate.
"""
from dataclasses import dataclass
from math import pi, sqrt
from operator import index
import numpy as np
from deuteron_wigner.bridge.modes.core import GAMMA, polarization
from .m2_gaussian_field_regulator import GaussianClusterRegulator, gaussian_shell_integral, _array
from .m2_canonical_emission import lf_spinor, _helicity
from .m2_excluded_fock_vertices import antiquark_spinor, _slash_epsilon, three_gluon_numerator
from .m2_constrained_fermion_current import (
    one_gluon_current_spin, pair_creation_current_spin, discrete_gluon_field_factor)
from .m2_antiquark_contact_current import antiquark_one_gluon_current_spin


PRESCRIPTION = "physical-leg-endpoint-product"


def _affine(constant, linear):
    lin = np.asarray(linear, complex).reshape(-1)
    zero = (0,)*len(lin)
    result = {zero:complex(constant)} if constant != 0 else {}
    for k,v in enumerate(lin):
        if v != 0:
            powers = list(zero)
            powers[k] = 1
            result[tuple(powers)] = v
    return result


def _multiply(left, right):
    result = {}
    for a,c in left.items():
        for b,d in right.items():
            key = tuple(x+y for x,y in zip(a,b))
            result[key] = result.get(key, 0)+c*d
    return result


def _add_scaled(target, source, scale):
    for key,value in source.items():
        target[key] = target.get(key, 0)+scale*value


@dataclass(frozen=True)
class ClusterCurrentIntegral:
    canonical: np.ndarray
    regulator: np.ndarray
    prescription: str
    reference_shape: object=None
    complete_gauged_contractions = False
    longitudinal_envelope_gauged = False
    physical = False

    @property
    def total(self):
        return self.canonical+self.regulator


def cluster_current_integrals(*, regulator, modes, species, outgoing_legs, charges,
        transfer_GeV, total_k2, total_pplus_GeV, prescription, quadrature_order,
        out_cutoff, in_cutoff, out_map, in_map, b_GeV, physical_field_maps,
        physical_field_shifts, hamiltonian_constant, hamiltonian_linear,
        hamiltonian_leg_derivatives, canonical_current_constant, canonical_current_linear,
        out_shift=None, in_shift=None, reference_shape=None):
    """Four (+,-,x,y) current orbitals with explicit physical-leg charge flow.

    H coefficients are M²/g^n; canonical current coefficients already are
    J^mu/(2P+)/g^n INCLUDING charges. Color/Fock/Jacobi measures remain external.
    H is affine in physical transverse momenta, as in cubic/quartic vertices.
    Leg derivatives hold all OTHER physical momenta fixed, not conservation.
    No particle/spectator or auxiliary-field identity is inferred from size.
    """
    if prescription != PRESCRIPTION or not isinstance(regulator, GaussianClusterRegulator):
        raise ValueError("explicit common cluster and physical-leg endpoint prescription required")
    reference_shape=regulator.shape_for(reference_shape)
    integral=gaussian_shell_integral if reference_shape is None else reference_shape.integrate
    def gauged_integral(**arguments):
        return (gaussian_shell_integral(**arguments) if reference_shape is None
                else reference_shape.integrate(gauged=True,**arguments))
    pp = float(total_pplus_GeV)
    if isinstance(total_k2, (bool,np.bool_)) or isinstance(quadrature_order, (bool,np.bool_)):
        raise ValueError("exact K2 and path order required")
    total, order = index(total_k2), index(quadrature_order)
    if total <= 0 or order < 2 or not np.isfinite(pp) or pp <= 0:
        raise ValueError("positive K2/P+ and path order at least two required")
    modes, species, outgoing_legs = tuple(modes), tuple(species), tuple(outgoing_legs)
    if len(modes)<2 or len(species)!=len(modes) or len(outgoing_legs)!=len(modes):
        raise ValueError("one physical species and orientation for each field occurrence required")
    if any(not isinstance(x,(bool,np.bool_)) for x in outgoing_legs):
        raise ValueError("explicit Boolean field orientations required")
    eligible = [regulator.contains(s,n) for s,n in zip(species,modes)]
    sigma = np.array([-1 if x else 1 for x in outgoing_legs])
    e = _array(charges,(len(modes),),"physical field charges")
    if any(e[a]!=0 for a,s in enumerate(species) if s=="g"):
        raise ValueError("physical electromagnetic/flavor gluon charge must be zero")
    if abs(sigma@e)>2e-14 or sigma@np.asarray(modes)!=0:
        raise ValueError("the ordered monomial must conserve charge and plus momentum at Delta+=0")
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    fields = np.asarray(physical_field_maps,float)
    if fields.ndim!=2 or fields.shape[0]!=len(modes):
        raise ValueError("one physical field map per occurrence required")
    d = fields.shape[1]
    fields = _array(fields,(len(modes),d),"physical field maps")
    shifts = _array(physical_field_shifts,(len(modes),2),"physical field shifts")
    if not np.allclose(sigma@fields,0.,atol=3e-13,rtol=0.) or not np.allclose(
            sigma@shifts,-delta,atol=3e-13,rtol=0.):
        raise ValueError("physical active legs must carry the full transverse photon recoil")
    h0 = complex(hamiltonian_constant)
    h1 = _array(hamiltonian_linear,(2,d),"Hamiltonian polynomial",complex_values=True)
    dh = _array(hamiltonian_leg_derivatives,(len(modes),2),"independent physical-leg H derivatives",complex_values=True)
    j0 = _array(canonical_current_constant,(4,),"canonical four-current constant",complex_values=True)
    j1 = _array(canonical_current_linear,(4,2,d),"canonical four-current polynomial",complex_values=True)
    if not np.isfinite(h0):
        raise ValueError("finite Hamiltonian constant required")
    transform = regulator.leg_transform(modes)
    precision = 1/regulator.width_GeV**2
    args = dict(out_cutoff=out_cutoff,in_cutoff=in_cutoff,out_map=out_map,in_map=in_map,
        b_GeV=b_GeV,field_precision=precision,field_maps=transform@fields,
        field_shifts=transform@shifts,out_shift=out_shift,in_shift=in_shift)
    zero = gaussian_shell_integral(**args,constant=0.)
    canonical, gauged = np.zeros((4,*zero.shape),complex), np.zeros((4,*zero.shape),complex)
    if not all(eligible):
        return ClusterCurrentIntegral(canonical,gauged,prescription,reference_shape)
    for mu in range(4):
        if j0[mu]!=0 or np.any(j1[mu]!=0):
            canonical[mu] = integral(**args,constant=j0[mu],linear=j1[mu])
    gram, w = transform.T@transform, np.asarray(modes,float)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    if np.all(delta==0):
        nodes, weights = np.array([0.]), np.array([1.])
    for a,charge in enumerate(e):
        if charge==0:
            continue
        # Endpoint, not an interpolated bare V: this is the stated product rule.
        endpoint = _affine(h0+sigma[a]*(dh[a]@delta),h1)
        if not endpoint:
            continue
        for t,weight in zip(nodes,weights):
            shifted = shifts.copy()
            shifted[a] += sigma[a]*t*delta
            current_args = {**args,"field_shifts":transform@shifted}
            for j in (0,1):
                linear = np.zeros((2,d))
                linear[j] = -precision*(gram[a]@fields)
                derivative = _affine(-precision*(gram[a]@shifted[:,j]),linear)
                gauged[j+2] += charge*weight/(2*pp)*gauged_integral(
                    **current_args,constant=0.,polynomial_terms=_multiply(endpoint,derivative))
            # -partial_p+ F = K2/(4 P+ Lambda²)
            # * [sum_b p_bT²/w_b - S p_aT²/w_a²] F, at fixed field modes.
            derivative = {}
            mode_weights = 1/w
            mode_weights = mode_weights.copy()
            mode_weights[a] -= w.sum()/w[a]**2
            for b,coefficient in enumerate(mode_weights):
                for j in (0,1):
                    linear = np.zeros((2,d))
                    linear[j] = fields[b]
                    component = _affine(shifted[b,j],linear)
                    _add_scaled(derivative,_multiply(component,component),
                                total*precision/(4*pp)*coefficient)
            gauged[1] += charge*weight/(2*pp)*gauged_integral(
                **current_args,constant=0.,polynomial_terms=_multiply(endpoint,derivative))
    return ClusterCurrentIntegral(canonical,gauged,prescription,reference_shape)


def cubic_physical_kernel(modes, transverse, *, channel, helicities,
                          total_k2, total_pplus_GeV, mass_GeV):
    """Actual normalized Dirac cubic kernel WITHOUT transverse conservation.

    Physical leg order: left daughter, right daughter, parent. Color is
    stripped, including the physical -T^T representation for antiquarks.
    The independent momenta fix the off-conservation extension explicitly.
    """
    w = np.asarray(modes,float)
    t = _array(transverse,(3,2),"physical transverse cubic momenta")
    pp = float(total_pplus_GeV)
    if w.shape!=(3,) or np.any(w<=0) or not np.all(np.isfinite(w)) or not np.isfinite(pp) or pp<=0 or total_k2<=0:
        raise ValueError("positive finite physical modes, K2 and P+ required")
    p = np.column_stack((w*pp/total_k2,t))
    h1,h2,h0 = tuple(_helicity(h) for h in helicities)
    mass = float(mass_GeV)
    if not np.isfinite(mass) or mass<0:
        raise ValueError("finite nonnegative fermion mass required")
    if channel=="q_to_qg":
        u,v = lf_spinor(*p[0],mass,h1),lf_spinor(*p[2],mass,h0)
        numerator = -u.conj()@GAMMA[0]@_slash_epsilon(polarization(*p[1],h2).conj())@v
    elif channel=="anti_to_antig":
        u,v = antiquark_spinor(*p[2],mass,h0),antiquark_spinor(*p[0],mass,h1)
        numerator = -u.conj()@GAMMA[0]@_slash_epsilon(polarization(*p[1],h2).conj())@v
    elif channel=="g_to_pair":
        u,v = lf_spinor(*p[0],mass,h1),antiquark_spinor(*p[1],mass,h2)
        numerator = -u.conj()@GAMMA[0]@_slash_epsilon(polarization(*p[2],h0))@v
    elif channel=="g_to_gg":
        numerator = three_gluon_numerator(*p,h1,h2,h0)
    else:
        raise ValueError("explicit canonical cubic channel required")
    cell = pi*total_k2/pp
    return complex(2*pp*numerator/sqrt(cell*8*np.prod(p[:,0])))


def cubic_canonical_current(modes, transverse, *, channel, helicities,
                            total_k2, total_pplus_GeV, mass_GeV, flavor_charge):
    """Established canonical spin current, not a derivative/Ward surrogate."""
    t = _array(transverse,(3,2),"physical transverse cubic momenta")
    w = np.asarray(modes)
    p = np.column_stack((w*float(total_pplus_GeV)/total_k2,t))
    h1,h2,h0 = tuple(_helicity(h) for h in helicities)
    if channel=="g_to_gg":
        return np.zeros(4,complex)
    if channel=="g_to_pair":
        spin = pair_creation_current_spin(p[0],p[1],mass_GeV=mass_GeV,
            polarization=polarization(*p[2],h0)[2:],right_plus=p[0,0],left_plus=-p[1,0])
        return flavor_charge*discrete_gluon_field_factor(modes[2])*spin[:,(h1+1)//2,(h2+1)//2]
    eps = polarization(*p[1],h2).conj()[2:]
    if channel=="q_to_qg":
        spin = one_gluon_current_spin(p[0],p[2],mass_GeV=mass_GeV,
            polarization=eps,right_plus=p[0,0],left_plus=p[2,0])
        charge = flavor_charge
    elif channel=="anti_to_antig":
        spin = antiquark_one_gluon_current_spin(p[0],p[2],mass_GeV=mass_GeV,
            polarization=eps,right_plus=-p[2,0],left_plus=-p[0,0])
        charge = -flavor_charge
    else:
        raise ValueError("explicit canonical cubic channel required")
    return charge*discrete_gluon_field_factor(modes[1])*spin[:,(h1+1)//2,(h0+1)//2]


def cubic_current_geometry(route, *, transfer_GeV, mean_total_transverse_GeV):
    """Physical three-leg recoil maps on an actual P/Q5/Q6 cubic route.

    Integrate in the global outgoing Jacobi coordinates. All spectator
    physical momenta remain unchanged. The incoming parent's momentum is
    the sum of the outgoing daughter momenta minus Delta. No auxiliary
    pair-current coordinate is interpreted as a physical gluon here.
    """
    from .m2_coupled_longitudinal_jacobi import (
        EmissionRoute, jacobi_frame, emission_coordinate_map)
    from .m2_excluded_fock_kinematics import (
        SplitRoute, frame as five_frame, split_frame as five_split)
    from .m2_six_body_vertices import SixSplitRoute, split_frame as six_split
    from .m2_six_body_kinematics import frame as six_frame
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    mean = _array(mean_total_transverse_GeV,(2,),"mean transverse momentum")
    if isinstance(route,EmissionRoute):
        fi,fo = jacobi_frame(route.incoming),jacobi_frame(route.outgoing)
        tree,children,parent = emission_coordinate_map(route),(route.emitter,3),route.emitter
        channel = "q_to_qg"
    elif isinstance(route,SplitRoute):
        fi,fo = five_frame("qqqg",route.incoming),five_frame(route.sector,route.outgoing)
        tree,children,parent = five_split(route),route.daughters,route.parent
        channel = route.channel
    elif isinstance(route,SixSplitRoute):
        fi,fo = five_frame(route.source_sector,route.incoming),six_frame(route.sector,route.outgoing)
        tree,children,parent = six_split(route),route.daughters,route.parent
        channel = route.channel
    else:
        raise ValueError("actual retained, Q5 or Q6 canonical cubic route required")
    maps = fo.inverse[list(children),:-1]
    shifts = fo.fractions[list(children),None]*(mean+delta/2)
    x,y = fo.fractions[list(children)]
    return dict(channel=channel,total_k2=sum(route.incoming),
        modes=tuple(route.outgoing[j] for j in children)+(route.incoming[parent],),
        out_map=np.eye(len(route.outgoing)-1),in_map=tree[:-1],
        in_shift=-fi.forward[:-1,parent,None]*delta,
        physical_field_maps=np.vstack((maps,maps.sum(axis=0))),
        physical_field_shifts=np.vstack((shifts,shifts.sum(axis=0)-delta)),
        jacobian=sqrt(x*y/(x+y)))


def cluster_cubic_current_orbitals(route, in_cutoff, out_cutoff, b_GeV, mass_GeV,
        helicities, *, regulator, total_pplus_GeV, transfer_GeV,
        mean_total_transverse_GeV, flavor_charge, prescription, quadrature_order, reference_shape=None):
    """Actual color-stripped four-current route, including its Jacobi measure.

    All helicity combinations, physical flavors/antiquark signs and both
    endpoint shells remain explicit. Creation statistics and color action
    belong to the surrounding raw Fock assembly, not this orbital kernel.
    The independent-momentum Dirac kernels are exactly affine; unit samples
    extract polynomial coefficients algebraically, not finite differences.
    """
    geometry = cubic_current_geometry(route,transfer_GeV=transfer_GeV,
        mean_total_transverse_GeV=mean_total_transverse_GeV)
    modes,channel = geometry["modes"],geometry["channel"]
    args = dict(channel=channel,helicities=helicities,total_k2=geometry["total_k2"],
                total_pplus_GeV=total_pplus_GeV,mass_GeV=mass_GeV)
    charge = float(flavor_charge)
    if not np.isfinite(charge):
        raise ValueError("explicit finite flavor charge required")
    kinds,charges = {
        "q_to_qg":(("q","g","q"),(charge,0.,charge)),
        "anti_to_antig":(("qbar","g","qbar"),(-charge,0.,-charge)),
        "g_to_pair":(("q","qbar","g"),(charge,-charge,0.)),
        "g_to_gg":(("g","g","g"),(0.,0.,0.)),
    }[channel]
    zero = np.zeros((3,2))
    constant = cubic_physical_kernel(modes,zero,**args)
    derivatives = np.zeros((3,2),complex)
    for leg in range(3):
        for j in range(2):
            unit = zero.copy()
            unit[leg,j] = 1.
            derivatives[leg,j] = cubic_physical_kernel(modes,unit,**args)-constant
    fields,shifts = geometry["physical_field_maps"],geometry["physical_field_shifts"]
    h0 = cubic_physical_kernel(modes,shifts,**args)
    h1 = derivatives.T@fields
    j0 = cubic_canonical_current(modes,shifts,**args,flavor_charge=charge)
    j1 = np.zeros((4,2,fields.shape[1]),complex)
    for j in range(2):
        for k in range(fields.shape[1]):
            unit = shifts.copy()
            unit[:,j] += fields[:,k]
            j1[:,j,k] = cubic_canonical_current(modes,unit,**args,flavor_charge=charge)-j0
    result = cluster_current_integrals(regulator=regulator,modes=modes,species=kinds,
        outgoing_legs=(True,True,False),charges=charges,transfer_GeV=transfer_GeV,
        total_k2=geometry["total_k2"],total_pplus_GeV=total_pplus_GeV,
        prescription=prescription,quadrature_order=quadrature_order,
        out_cutoff=out_cutoff,in_cutoff=in_cutoff,out_map=geometry["out_map"],
        in_map=geometry["in_map"],in_shift=geometry["in_shift"],b_GeV=b_GeV,
        physical_field_maps=fields,physical_field_shifts=shifts,
        hamiltonian_constant=h0,hamiltonian_linear=h1,hamiltonian_leg_derivatives=derivatives,
        canonical_current_constant=j0,canonical_current_linear=j1,reference_shape=reference_shape)
    return ClusterCurrentIntegral(geometry["jacobian"]*result.canonical,
                                  geometry["jacobian"]*result.regulator,prescription,result.reference_shape)
