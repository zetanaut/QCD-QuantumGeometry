"""Ordered physical-leg Wick currents of the SAME common cluster kernel.

The transverse loop is integrated only after physical external/internal
photon attachments. External and internal plus paths are different at
finite transverse transfer. This module keeps canonical, regulator and
normal-order subtraction pieces separate, with the H trace of every term.

These smooth Delta+=0 kernels hold field-envelope membership fixed.
Longitudinal-envelope partners, full propagating loops and physical finite
matching are not supplied by normal ordering.
"""
from dataclasses import dataclass
from math import pi,comb,factorial
import numpy as np
from scipy.special import hyp1f1
from .m2_gaussian_field_regulator import GaussianClusterRegulator, _array
from .m2_gaussian_cluster_current import PRESCRIPTION
from .m2_gaussian_cluster_wick import GaussianClusterWick
from .m2_canonical_quark_self import CF, _integer
from .m2_canonical_gluon_self import CA, TF, gluon_wick_sums
from .m2_canonical_good_current import parton_charge


def _gaussian_moment(n,a):
    """Integral_0^1 t^n exp(-a t²) dt, analytic and regular at a=0."""
    return float(hyp1f1((n+1)/2,(n+3)/2,-a)/(n+1))


@dataclass(frozen=True)
class ClusterContractionPaths:
    coincident_trace_GeV2: float
    base_trace_GeV2: float
    external_mode_path_GeV2: float
    loop_mode_path_GeV2: float


def _shape_path_moment(n,c0,c2,shape,*,inverse_a):
    """Apply sum delta^m (-d_a)^m/m! to a^-k integral t^n exp[-a(c0+c2t²)].

    k is zero or one here. Coefficients delta are held fixed during the
    derivative. This performs the same WHOLE-vertex insertion before the
    loop integral, not separate expansions of the two contracted halves.
    """
    a=shape.a;delta=a-shape.a_reference;ratio=delta/a
    value=0.
    for j in range(shape.degree+1):
        polynomial=sum(comb(j,l)*c0**(j-l)*c2**l*_gaussian_moment(n+2*l,a*c2)
                       for l in range(j+1))
        coefficient=delta**j/factorial(j)
        if inverse_a:coefficient*=sum(ratio**m for m in range(shape.degree-j+1))/a
        value+=coefficient*polynomial
    return np.exp(-a*c0)*value


def cluster_contraction_paths(*,regulator,external_mode,loop_mode,transfer_GeV,reference_shape=None):
    """Closed loop integrals of both one-leg plus-mode paths.

    The returned paths contain sum_a integral dt partial_(w_a) F for the
    two external or the two loop occurrences. Multiply -K2/P+ for K-minus.
    At the base the two loop momenta coincide. Each photon path moves ONE
    of them (or one external leg) before integration. Their transverse
    derivative sums cancel exactly, separately for external and loop pairs.
    """
    if not isinstance(regulator,GaussianClusterRegulator):
        raise ValueError("explicit common cluster regulator required")
    p,k = float(external_mode),float(loop_mode)
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    if not np.all(np.isfinite([p,k])) or min(p,k)<=0:
        raise ValueError("positive finite external and loop modes required")
    reference_shape=regulator.shape_for(reference_shape)
    L,D = regulator.width_GeV**2,float(delta@delta)
    if reference_shape is not None:
        s=reference_shape
        C=k/(8*pi*p);c0=(p+k)*D/(2*p);c2=(p+k)*D/(2*k)
        moment=lambda n,x,y,inv:_shape_path_moment(n,x,y,s,inverse_a=inv)
        trace=C*moment(0,0.,0.,True)
        base=C*moment(0,c0,0.,True)
        external=-C/p*moment(0,0.,c0,True)
        external+=C*D/(2*p*p)*((2*p+3*k)*moment(2,0.,c0,False)-2*(p+k)*moment(1,0.,c0,False))
        loop=C/k*moment(0,c0,c2,True)-C*D/(2*p)*moment(0,c0,c2,False)
        loop+=C*p*D/(2*k*k)*moment(2,c0,c2,False)
        if not np.all(np.isfinite([trace,base,external,loop])):
            raise ArithmeticError("nonfinite reference-shape Wick path")
        return ClusterContractionPaths(float(trace),float(base),float(external),float(loop))
    trace = L*k/(4*pi*p)
    a,b = (p+k)*D/(4*p*L),(p+k)*D/(4*k*L)
    # On an external path the two possible active-leg moves have different
    # external means. Their combined correction to the common-mean path is
    # -(p+k)t(1-t)D/(2 Lambda² p²), not partial_p of the contracted trace.
    external = trace*(-_gaussian_moment(0,a)/p+D/(4*L*p*p)*(
        (2*p+3*k)*_gaussian_moment(2,a)-2*(p+k)*_gaussian_moment(1,a)))
    loop = trace*np.exp(-a)*((1/k-D/(4*L*p))*_gaussian_moment(0,b)
                           +p*D/(4*L*k*k)*_gaussian_moment(2,b))
    base = trace*np.exp(-a)
    if not np.all(np.isfinite([trace,base,external,loop])):
        raise ArithmeticError("nonfinite common-kernel contraction integral")
    return ClusterContractionPaths(float(trace),float(base),float(external),float(loop))


@dataclass(frozen=True)
class OrderedClusterWickTerm:
    source: str
    ordering: str
    loop_species: str
    loop_mode: int
    loop_flavor: int | None
    hamiltonian_per_gs2_GeV2: float
    canonical_minus: float
    external_leg_minus: float
    internal_leg_minus: float

    @property
    def regulator_minus(self):
        return self.external_leg_minus+self.internal_leg_minus

    @property
    def minus(self):
        return self.canonical_minus+self.regulator_minus


def ordered_cluster_wick_terms(species,mode,*,regulator,quark_flavors,total_k2,
        total_pplus_GeV,transfer_GeV,external_charge,loop_flavor_charges,prescription,reference_shape=None):
    """All connected quadratic IF/IG/A4 descendants, before subtraction.

    H values are M²/gs² on the conserved slice. Currents are J-/(2P+)/gs²
    at the supplied transverse transfer; J+ and J_T vanish for these traces.
    A loop flavor remains separately identified in every gluon IF term.
    IG same-flavor direct and pair contractions have opposite loop charges.
    The vacuum fermion in a gluon IF contraction is a physical antiquark.
    No k<p restriction or absolute constrained-mode prescription is used.
    """
    if prescription!=PRESCRIPTION or not isinstance(regulator,GaussianClusterRegulator):
        raise ValueError("explicit common cluster and physical-leg endpoint prescription required")
    reference_shape=regulator.shape_for(reference_shape)
    p,total = _integer(mode,"external mode"),_integer(total_k2,"total K2")
    pp,e = float(total_pplus_GeV),float(external_charge)
    if not np.all(np.isfinite([pp,e])) or pp<=0 or p>=total:
        raise ValueError("positive total P+, constituent p2<K2 and finite charge required")
    gluon_wick_sums(2,boson_modes=regulator.boson_modes,fermion_modes=regulator.fermion_modes,
                   quark_flavors=quark_flavors)
    loops = _array(loop_flavor_charges,(quark_flavors,),"Hamiltonian loop flavor charges")
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    eligible = regulator.contains(species,p)
    if species=="g" and e!=0:
        raise ValueError("a physical gluon has no bare electromagnetic/flavor charge")
    if not eligible:
        return ()
    result = []
    def add(source,ordering,kind,k,h,j,loop_charge,flavor=None):
        path = cluster_contraction_paths(regulator=regulator,external_mode=p,
                                        loop_mode=k,transfer_GeV=delta,reference_shape=reference_shape)
        hm = total/pi*h
        gauge = -hm*total/(2*pp*pp)
        result.append(OrderedClusterWickTerm(source,ordering,kind,k,flavor,
            hm*path.coincident_trace_GeV2,
            total*total/(2*pi*pp*pp)*j*path.base_trace_GeV2,
            gauge*e*path.external_mode_path_GeV2,
            gauge*loop_charge*path.loop_mode_path_GeV2))
    bosons = range(2,2*regulator.boson_modes+1,2)
    fermions = range(1,2*regulator.fermion_modes,2)
    if species in ("q","qbar"):
        for k in bosons:
            r = p-k if species=="q" else p+k
            add("IF","ordered-boson-vacuum","g",k,CF/(k*r),e*CF/(k*r*r),0.)
        for k in fermions:
            if k!=p:  # only the exact zero IG transfer is excluded
                add("IG","direct",species,k,CF/(p-k)**2,0.,e)
            opposite = "qbar" if species=="q" else "q"
            add("IG","pair-blocking",opposite,k,-CF/(p+k)**2,0.,-e)
    else:
        for flavor,charge in enumerate(loops):
            for k in fermions:
                for label,sign in (("difference",-1),("sum",1)):
                    r = p+sign*k
                    add("IF",label,"qbar",k,-sign*TF/(p*r),
                        TF*charge/(p*r*r),-charge,flavor)
        for k in bosons:
            if k!=p:
                add("IG","direct","g",k,CA*(p+k)**2/(4*p*k*(p-k)**2),0.,0.)
            add("IG","pair","g",k,CA*(p-k)**2/(4*p*k*(p+k)**2),0.,0.)
            add("A4","local","g",k,CA/(2*p*k),0.,0.)
    return tuple(result)


@dataclass(frozen=True,kw_only=True)
class ClusterWickCurrentWeights:
    """Transfer-bound projection adapter; never mistaken for bare WickOneBody."""
    operators: GaussianClusterWick
    transfer_GeV: tuple
    prescription: str
    component: str
    complete_gauged_contractions = False
    longitudinal_envelope_gauged = False
    physical = False

    def __post_init__(self):
        if not isinstance(self.operators,GaussianClusterWick) or self.prescription!=PRESCRIPTION:
            raise ValueError("same common-cluster Wick operators and explicit current prescription required")
        delta = _array(self.transfer_GeV,(2,),"transverse transfer")
        if self.component not in ("canonical","regulator","ordered","subtraction","finite","total"):
            raise ValueError("explicit canonical/regulator/ordered/subtraction/finite/total component required")
        object.__setattr__(self,"transfer_GeV",tuple(delta))

    def minus_particle_weight(self,species,mode,spins,*,total_k2,total_pplus_GeV,
                              charge,loop_flavor_charges):
        """Physical flavor-resolved weight, including finite kinetic partner.

        Unit external-charge response is computed once for q/qbar, then
        multiplied by that constituent's actual signed flavor charge.
        Gluon IF currents instead retain the specified loop charges.
        The finite mass term changes kinetic mass only, not vertex mass.
        """
        physical = parton_charge(spins,species,charge)
        op = self.operators
        terms = ordered_cluster_wick_terms(species,mode,regulator=op.regulator,
            quark_flavors=op.quark_flavors,total_k2=total_k2,total_pplus_GeV=total_pplus_GeV,
            transfer_GeV=self.transfer_GeV,external_charge=0. if species=="g" else 1.,
            loop_flavor_charges=loop_flavor_charges,prescription=self.prescription)
        canonical,reg = sum(t.canonical_minus for t in terms),sum(t.regulator_minus for t in terms)
        ordered = canonical+reg
        finite = 0. if species=="g" else op.finite_quark_mass2_per_gs2_GeV2*total_k2**2/(
            2*mode*mode*float(total_pplus_GeV)**2)
        subtraction = -ordered if op.normal_order_subtraction else 0.
        value = dict(canonical=canonical,regulator=reg,ordered=ordered,subtraction=subtraction,
                     finite=finite,total=ordered+subtraction+finite)[self.component]
        return np.full(np.shape(physical),value,float) if species=="g" else physical*value


def cluster_wick_retained_minus_between(outgoing,incoming,transfer,*,charge,
        loop_flavor_charges,prescription,component):
    """Actual PP current coefficient per gs², tied to its existing Wick H.

    There is no second Hamiltonian insertion. The two pilots must already
    contain the identical common-kernel Wick/finite definition.
    """
    from .m2_gaussian_cluster_h2 import require_same_cluster_definition
    from .m2_wick_current_projection import retained_wick_minus_between
    if not require_same_cluster_definition(outgoing,incoming):
        raise ValueError("actual common-cluster regulated current endpoints required")
    weights = ClusterWickCurrentWeights(operators=incoming.wick_operators,
        transfer_GeV=transfer,prescription=prescription,component=component)
    return retained_wick_minus_between(outgoing,incoming,transfer,operators=weights,
        total_pplus_GeV=incoming.total_pplus_GeV,charge=charge,loop_flavor_charges=loop_flavor_charges)


def cluster_wick_shell_minus_between(outgoing,incoming,values,transfer,*,operators,
        b_GeV,total_pplus_GeV,charge,loop_flavor_charges,prescription,component):
    """Exact packed Q5/Q6 one-body current action, no source-rank projection."""
    from .m2_wick_current_projection import shell_wick_minus_between
    weights = ClusterWickCurrentWeights(operators=operators,transfer_GeV=transfer,
                                       prescription=prescription,component=component)
    return shell_wick_minus_between(outgoing,incoming,values,transfer,operators=weights,
        b_GeV=b_GeV,total_pplus_GeV=total_pplus_GeV,charge=charge,loop_flavor_charges=loop_flavor_charges)


def cluster_wick_minus_h2_coefficients(outgoing,incoming,transfer,*,charge,
        loop_flavor_charges,prescription,component):
    """Direct common-kernel Wick-current insertions at powers two and four.

    The common H2 snapshot ALREADY contains its paired Wick H, changed
    third-order raw backflow and both norms. This function adds only the
    explicit one-body current: W_PP at g² and A1_f† W55 A1_i at g⁴.
    Q6 one-body currents first return at g⁶ and are not mixed into H2.
    This is one contribution, not the full matched four-current.
    """
    from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot,require_same_cluster_definition
    if not isinstance(outgoing,GaussianClusterSnapshot) or not isinstance(incoming,GaussianClusterSnapshot):
        raise ValueError("both actual common-cluster H2 snapshots required")
    f,i = outgoing.owner,incoming.owner
    if not require_same_cluster_definition(f.pilot,i.pilot):
        raise ValueError("identical common-cluster H2 definitions required")
    if f.spaces5.keys()!=i.spaces5.keys() or f.spaces6.keys()!=i.spaces6.keys():
        raise ValueError("same declared physical Q sectors required")
    for n in (*f.spaces5,*f.spaces6):
        fs,ins = f.spaces5.get(n,f.spaces6.get(n)),i.spaces5.get(n,i.spaces6.get(n))
        if fs.cutoff!=ins.cutoff:
            raise ValueError("same declared Q-sector resolution at both current endpoints required")
    args = dict(charge=charge,loop_flavor_charges=loop_flavor_charges,
                prescription=prescription,component=component)
    result = {2:cluster_wick_retained_minus_between(f.pilot,i.pilot,transfer,**args),
              4:np.zeros((len(f.pilot.matrix),len(i.pilot.matrix)),complex)}
    for name,space in i.spaces5.items():
        acted = cluster_wick_shell_minus_between(f.spaces5[name],space,
            incoming.five_wave_coefficients[0][name],transfer,operators=i.wick_operators,
            b_GeV=i.pilot.width_GeV,total_pplus_GeV=i.pilot.total_pplus_GeV,**args)
        result[4] += outgoing.five_wave_coefficients[0][name].conj().T@acted
    return result


def cluster_wick_minus_h2_between(outgoing,incoming,transfer,*,charge,
        loop_flavor_charges,prescription,component,normalized):
    """Selected direct Wick-current contribution with both existing H2 norms."""
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit Boolean two-endpoint normalization choice required")
    coefficients = cluster_wick_minus_h2_coefficients(outgoing,incoming,transfer,charge=charge,
        loop_flavor_charges=loop_flavor_charges,prescription=prescription,component=component)
    g = incoming.owner.pilot.coupling
    value = sum(g**n*a for n,a in coefficients.items())
    return outgoing.whitener@value@incoming.whitener if normalized else value
