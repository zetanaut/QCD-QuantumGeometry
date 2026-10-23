"""Physical four-leg 1→3 source currents of the same common H2 kernel.

All ten normal-ordered quartic source families are retained. The transverse-
constant Hamiltonian may be summed over its ordered color terms because
every ordering has the SAME four physical fields and common scalar kernel.
Canonical J-minus is independently supplied by the constrained-fermion
bilinear. IG contacts have zero canonical current, not zero regulator current.
No auxiliary leg is gauged; no Wick or full H2 matching is claimed here.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import sqrt
import numpy as np
from .m2_coupled_longitudinal_jacobi import jacobi_frame,longitudinal_basis
from .m2_excluded_fock_kinematics import SPECIES as FIVE,frame as frame5
from .m2_six_body_kinematics import SPECIES as SIX,frame as frame6
from .m2_instantaneous_pair_exit import (
    _route,pair_emission_frame,pair_mass_kernel,adjoint_pair_mass_kernel)
from .m2_instantaneous_sea_exit import sea_pair_mass_kernel
from .m2_six_body_quartic import DirectRoute,direct_routes,double_split_frame,mass_kernel
from .m2_quadratic_minus_fock import pair_emission_minus_kernel,direct_six_minus_kernel
from .m2_gaussian_field_regulator import _array
from .m2_gaussian_cluster_current import cluster_current_integrals,ClusterCurrentIntegral


@dataclass(frozen=True)
class PairSourceRoute:
    """Actual retained qqq→Q5 source, not a fictitious factorization route."""
    component: str
    incoming: tuple
    outgoing: tuple
    emitter: int

    def __post_init__(self):
        if isinstance(self.emitter,(bool,np.bool_)) or not isinstance(self.emitter,(int,np.integer)):
            raise ValueError("explicit exact valence emitter required")
        object.__setattr__(self,"incoming",tuple(self.incoming))
        object.__setattr__(self,"outgoing",tuple(self.outgoing))
        _route(self.incoming,self.outgoing,self.emitter,self.sector)

    @property
    def sector(self):
        if self.component not in ("valence_IF","valence_IG","valence_sea"):
            raise ValueError("explicit IF, IG or sea pair source required")
        return "qqqq-qbar" if self.component=="valence_sea" else "qqqgg"

    @property
    def daughters(self):
        return (self.emitter,3,4)


def quartic_source_geometry(route,*,transfer_GeV,mean_total_transverse_GeV):
    """Three outgoing fields and the incoming parent in global Jacobi frames.

    Incoming parent = sum(outgoing daughters) - Delta. Incoming inherited
    coordinates shift by -L_in[:,parent] Delta; the two new relative
    coordinates are integrated independently. The surviving old gluon is
    relocated to slot5 in valence-sea emission, but is NOT a participating leg.
    """
    delta = _array(transfer_GeV,(2,),"transverse transfer")
    mean = _array(mean_total_transverse_GeV,(2,),"mean transverse momentum")
    if isinstance(route,PairSourceRoute):
        fi,fo = jacobi_frame(route.incoming),_route(route.incoming,route.outgoing,route.emitter,route.sector)
        tree = pair_emission_frame(route.incoming,route.outgoing,route.emitter,sector=route.sector)
        species = FIVE[route.sector]
    elif isinstance(route,DirectRoute):
        fi,fo = frame5("qqqg",route.incoming),frame6(route.sector,route.outgoing)
        tree,species = double_split_frame(route),SIX[route.sector]
    else:
        raise ValueError("actual qqq→Q5 or qqqg→Q6 quartic route required")
    children,parent = route.daughters,route.emitter
    maps = fo.inverse[list(children),:-1]
    shifts = fo.fractions[list(children),None]*(mean+delta/2)
    fractions = fo.fractions[list(children)]
    return dict(total_k2=sum(route.incoming),
        modes=tuple(route.outgoing[j] for j in children)+(route.incoming[parent],),
        species=tuple(species[j] for j in children)+("q" if parent<3 else "g",),
        out_map=np.eye(len(route.outgoing)-1),in_map=tree[:-2],
        in_shift=-fi.forward[:-1,parent,None]*delta,
        physical_field_maps=np.vstack((maps,maps.sum(axis=0))),
        physical_field_shifts=np.vstack((shifts,shifts.sum(axis=0)-delta)),
        jacobian=sqrt(np.prod(fractions)/fractions.sum()))


def quartic_source_charges(route,*,emitter_charge,pair_charge):
    """Physical charges: created q/anti are distinct from the valence flavor."""
    if not isinstance(route,(PairSourceRoute,DirectRoute)):
        raise ValueError("actual canonical quartic source route required")
    e,f = float(emitter_charge),float(pair_charge)
    if not np.all(np.isfinite([e,f])):
        raise ValueError("explicit finite emitter and created-pair charges required")
    if route.component in ("valence_IF","valence_IG"):
        return (e,0.,0.,e)
    if route.component=="valence_sea":
        return (e,f,-f,e)
    if route.component in ("gluon_IF_pair","gluon_IG_pair"):
        return (f,-f,0.,0.)
    return (0.,0.,0.,0.)


def quartic_source_coefficients(route,helicities,*,total_pplus_GeV,emitter_charge,pair_charge):
    """Actual M²/gs² and J-minus/(2P+ gs²) COLOR matrices, Fock included.

    Q5 gg creation divides by sqrt(2!); sea creation multiplies by2.
    All Q6 factors already belong to mass_kernel/direct_six_minus_kernel.
    The sum over two constrained-field orders retains signed denominators
    in H and independently squared denominators in canonical J-minus.
    """
    pp = float(total_pplus_GeV)
    quartic_source_charges(route,emitter_charge=emitter_charge,pair_charge=pair_charge)
    if not np.isfinite(pp) or pp<=0:
        raise ValueError("positive finite total P+ required")
    if isinstance(route,PairSourceRoute):
        kernels = {"valence_IF":pair_mass_kernel,"valence_IG":adjoint_pair_mass_kernel,
                   "valence_sea":sea_pair_mass_kernel}
        if route.component not in kernels:
            raise ValueError("explicit IF, IG or sea pair source required")
        args = (route.incoming,route.outgoing,route.emitter,helicities)
        fock = 2. if route.component=="valence_sea" else 1/sqrt(2)
        h = fock*kernels[route.component](*args)
        j = (fock*float(emitter_charge)*pair_emission_minus_kernel(*args,total_pplus_GeV=pp)
             if route.component=="valence_IF" else np.zeros_like(h))
    elif isinstance(route,DirectRoute):
        h = mass_kernel(route,helicities)
        if route.component in ("valence_IF","gluon_IF_pair"):
            e = emitter_charge if route.component=="valence_IF" else pair_charge
            j = float(e)*direct_six_minus_kernel(route,helicities,total_pplus_GeV=pp)
        else:
            j = np.zeros_like(h)
    else:
        raise ValueError("actual canonical quartic source route required")
    return h,j


@lru_cache(maxsize=2048)
def _source_orbitals(route,ci,co,b,regulator,pp,delta,mean,charges,prescription,order,reference_shape=None):
    """Unit transverse-constant H and unit canonical minus, color external."""
    g = quartic_source_geometry(route,transfer_GeV=delta,mean_total_transverse_GeV=mean)
    d = g["out_map"].shape[1]
    result = cluster_current_integrals(regulator=regulator,modes=g["modes"],species=g["species"],
        outgoing_legs=(True,True,True,False),charges=charges,transfer_GeV=delta,
        total_k2=g["total_k2"],total_pplus_GeV=pp,prescription=prescription,quadrature_order=order,
        out_cutoff=co,in_cutoff=ci,out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],
        b_GeV=b,physical_field_maps=g["physical_field_maps"],physical_field_shifts=g["physical_field_shifts"],
        hamiltonian_constant=1.,hamiltonian_linear=np.zeros((2,d)),hamiltonian_leg_derivatives=np.zeros((4,2)),
        canonical_current_constant=(0.,1.,0.,0.),canonical_current_linear=np.zeros((4,2,d)),
        reference_shape=reference_shape)
    arrays = [g["jacobian"]*a for a in (result.canonical,result.regulator)]
    for a in arrays:a.setflags(write=False)
    return ClusterCurrentIntegral(*arrays,prescription,result.reference_shape)


def cluster_quartic_source_current(route,in_cutoff,out_cutoff,b_GeV,helicities,*,
        regulator,total_pplus_GeV,transfer_GeV,mean_total_transverse_GeV,
        emitter_charge,pair_charge,prescription,quadrature_order,reference_shape=None):
    """(+,-,x,y) × outgoing/incoming orbitals × outgoing/incoming colors.

    Includes all source Fock factors. Physical reverse absorption is the
    reversed-transfer emission's adjoint. This is the normal-ordered source
    contact only, with no propagators, Wick insertions or normalization added.
    """
    from .m2_gaussian_cluster_excluded_current import _definition
    delta,mean,order,b,_,pp = _definition(regulator,transfer_GeV,mean_total_transverse_GeV,
        prescription,quadrature_order,b_GeV,0.,total_pplus_GeV)
    charges = quartic_source_charges(route,emitter_charge=emitter_charge,pair_charge=pair_charge)
    h,j = quartic_source_coefficients(route,helicities,total_pplus_GeV=pp,
        emitter_charge=emitter_charge,pair_charge=pair_charge)
    orbit = _source_orbitals(route,in_cutoff,out_cutoff,b,regulator,pp,delta,mean,charges,prescription,order,reference_shape)
    return ClusterCurrentIntegral(np.einsum("moi,ab->moiab",orbit.canonical,j),
        np.einsum("moi,ab->moiab",orbit.regulator,h),prescription,orbit.reference_shape)


@dataclass(frozen=True)
class ClusterQuarticSourceCurrent:
    incoming_pilot: object
    outgoing_space: object
    retained_columns: np.ndarray
    transfer: tuple
    component: str
    canonical: np.ndarray
    regulator: np.ndarray
    prescription: str
    complete_H2_matched_current = False
    longitudinal_envelope_gauged = False
    physical = False

    @property
    def total(self):
        return self.canonical+self.regulator


def build_cluster_quartic_source_current(pilot,space,transfer,*,component,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """Actual retained P→Q5/Q6 source action, all four components, per gs².

    All raw color/orbital/spin coordinates and both created-pair flavors are
    kept before the statistics projection. The endpoint may have a different
    Jz or oscillator cutoff. No extra creation factor is applied by this
    assembly: coefficients already contain the exact Hamiltonian factors.
    """
    from .m2_canonical_good_current import parton_charge
    from .m2_excluded_fock_action import IntermediateSpace
    from .m2_six_body_space import SixBodySpace
    from .m2_gaussian_cluster_h2 import GaussianClusterPilot
    from .m2_gaussian_cluster_excluded_current import _definition
    from .m2_source_vectors import source_tensor
    from .m2_excluded_transverse_contact import _add_current
    from .m2_six_body_transverse_contact import _add
    if not isinstance(pilot,GaussianClusterPilot) or not isinstance(space,(IntermediateSpace,SixBodySpace)):
        raise ValueError("actual common-cluster pilot and explicit physical Q5/Q6 space required")
    if any(getattr(space,k)!=getattr(pilot,k) for k in ("total_k2","target")):
        raise ValueError("current endpoints disagree on total momentum/target")
    if pilot.field_regulator!=pilot.wick_operators.regulator:
        raise ValueError("pilot field and Wick definitions disagree")
    delta,mean,order,b,_,pp = _definition(pilot.field_regulator,transfer,
        mean_total_transverse_GeV,prescription,quadrature_order,
        pilot.width_GeV,pilot.quark_mass_GeV,pilot.total_pplus_GeV)
    flavor_charges = parton_charge(np.array([0,2]),"q",charge)
    pair_source = isinstance(space,IntermediateSpace)
    if pair_source:
        if component not in ("valence_IF","valence_IG","valence_sea"):
            raise ValueError("explicit Q5 IF, IG or sea source component required")
        expected = "qqqq-qbar" if component=="valence_sea" else "qqqgg"
        if space.sector!=expected:
            raise ValueError("explicit matching quartic component and physical endpoint sector required")
        routes = []
        for out in space.longitudinal:
            for q in range(3):
                inc = list(out[:3]);inc[q] += out[3]+out[4]
                routes.append(PairSourceRoute(component,tuple(inc),out,q))
        columns = np.arange(pilot.bases[0].shape[1])
        parent_sector = "qqq"
    else:
        routes = direct_routes(pilot.total_k2,component)
        expected = routes[0].sector if routes else None
        columns = np.arange(pilot.bases[0].shape[1],len(pilot.matrix))
        parent_sector = "qqqg"
    if space.sector!=expected:
        raise ValueError("explicit matching quartic component and physical endpoint sector required")
    source = source_tensor(pilot,parent_sector,np.eye(len(pilot.matrix))[:,columns])
    incoming = longitudinal_basis(parent_sector,pilot.total_k2)
    ins,outs = ({s:i for i,s in enumerate(states)} for states in (incoming,space.longitudinal))
    sf = np.array(list(product(range(4),repeat=3)))
    images = [np.zeros((4,space.dimension,len(columns)),complex) for _ in range(2)]
    # All participating fields are gluons, independently of source helicity/color.
    if component in ("gluon_A4","gluon_IG_ggg"):
        return ClusterQuarticSourceCurrent(pilot,space,columns,delta,component,*images,prescription)
    for route in routes:
        for hs in product((-1,1),repeat=4):
            h,junit = quartic_source_coefficients(route,hs,total_pplus_GeV=pp,emitter_charge=1.,pair_charge=1.)
            if not np.any(h) and not np.any(junit):
                continue
            blocks = []
            if component.startswith("valence"):
                ho,hi,h1,h2 = hs;q = route.emitter
                for flavor in range(2):
                    selected = np.flatnonzero((2*(sf[:,q]%2)-1==hi)&(sf[:,q]//2==flavor))
                    changed = sf[selected].copy();changed[:,q] = 2*flavor+(ho+1)//2
                    for old_hg in (range(1) if pair_source else range(2)):
                        tensor = source[ins[route.incoming]] if pair_source else source[ins[route.incoming],:,:,:,old_hg]
                        value = np.take(tensor,selected,axis=2)
                        if component=="valence_sea":
                            for created in range(2):
                                pair_spins = [np.full(len(selected),2*created+(hh+1)//2) for hh in (h1,h2)]
                                labels = [changed,*pair_spins]
                                if not pair_source:labels.append(np.full(len(selected),old_hg))
                                blocks.append((value,np.column_stack(labels),flavor_charges[flavor],flavor_charges[created]))
                        else:
                            labels = [changed]
                            if not pair_source:labels.append(np.full(len(selected),old_hg))
                            labels.extend(np.full(len(selected),(hh+1)//2) for hh in (h1,h2))
                            blocks.append((value,np.column_stack(labels),flavor_charges[flavor],0.))
            else:
                hq,ha,hg,hin = hs
                value = source[ins[route.incoming],:,:,:,(hin+1)//2]
                for created in range(2):
                    spins = np.column_stack((sf,np.full(64,2*created+(hq+1)//2),
                        np.full(64,2*created+(ha+1)//2),np.full(64,(hg+1)//2)))
                    blocks.append((value,spins,0.,flavor_charges[created]))
            for value,labels,e,f in blocks:
                physical = quartic_source_charges(route,emitter_charge=e,pair_charge=f)
                orbit = _source_orbitals(route,pilot.cutoff,space.cutoff,b,pilot.field_regulator,
                    pp,delta,mean,physical,prescription,order)
                j = junit*(e if component=="valence_IF" else f)
                spins = np.ravel_multi_index(labels.T,space.spin_dims)
                for image,orb,color in ((images[0],orbit.canonical,j),(images[1],orbit.regulator,h)):
                    if not np.any(color):continue
                    for mu,matrix in enumerate(orb):
                        if not np.any(matrix):continue
                        amp = np.einsum("oi,ab,ibsk->oask",matrix,color,value,optimize=True)
                        if pair_source:_add_current(space,image[mu],outs[route.outgoing],spins,amp)
                        else:_add(space,outs[route.outgoing],spins,amp,image[mu])
    images = [np.array([space.statistics_project(a) for a in image]) for image in images]
    return ClusterQuarticSourceCurrent(pilot,space,columns,delta,component,*images,prescription)


def cluster_quartic_source_h2_coefficients(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """All quartic SOURCE-current return words through gs⁴, not PP or Q5Q5.

    J2_5P returns against A1 at gs³ and A2 at gs⁴; J2_6P returns
    against A6 at gs⁴. Both absorption and emission endpoints are explicit.
    C3† J2_5P first enters at gs⁵ and is deliberately not mixed in.
    """
    from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot,require_same_cluster_definition
    if not all(isinstance(x,GaussianClusterSnapshot) for x in (outgoing,incoming)):
        raise ValueError("both actual common-cluster H2 snapshots required")
    f,i = outgoing.owner,incoming.owner
    if not require_same_cluster_definition(f.pilot,i.pilot):
        raise ValueError("identical common-cluster H2 definitions required")
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(f.pilot,key)!=getattr(i.pilot,key):
            raise ValueError("H2 source-current endpoints disagree on "+key)
    for fs,ins in ((f.spaces5,i.spaces5),(f.spaces6,i.spaces6)):
        if fs.keys()!=ins.keys():
            raise ValueError("same declared physical Q sectors required")
        if any(fs[n].cutoff!=ins[n].cutoff for n in ins):
            raise ValueError("same declared Q-sector resolution at both H2 endpoints required")
    delta = tuple(_array(transfer,(2,),"transverse transfer"));reverse = tuple(-x for x in delta)
    args = dict(charge=charge,mean_total_transverse_GeV=mean_total_transverse_GeV,
                prescription=prescription,quadrature_order=quadrature_order)
    shape = (4,len(f.pilot.matrix),len(i.pilot.matrix))
    result = {part:{n:np.zeros(shape,complex) for n in (3,4)} for part in ("canonical","regulator")}
    groups = [
        (f.spaces5,i.spaces5,
         {"valence_IF":"qqqgg","valence_IG":"qqqgg","valence_sea":"qqqq-qbar"},
         [(3,outgoing.five_wave_coefficients[0],incoming.five_wave_coefficients[0]),
          (4,outgoing.five_wave_coefficients[1],incoming.five_wave_coefficients[1])]),
        (f.spaces6,i.spaces6,
         {c:("qqqq-qbar-g" if c in ("valence_sea","gluon_IF_pair","gluon_IG_pair") else "qqqggg")
          for c in ("valence_IF","valence_IG","valence_sea","gluon_A4","gluon_IF_pair","gluon_IG_pair","gluon_IG_ggg")},
         [(4,outgoing.six_wave_coefficient,incoming.six_wave_coefficient)]),
    ]
    for fs,ins,components,waves in groups:
        for c,name in components.items():
            jf = build_cluster_quartic_source_current(i.pilot,fs[name],delta,component=c,**args)
            jb = build_cluster_quartic_source_current(f.pilot,ins[name],reverse,component=c,**args)
            for part in result:
                for n,af,ai in waves:
                    result[part][n][:,:,jf.retained_columns] += np.einsum(
                        "ab,mbc->mac",af[name].conj().T,getattr(jf,part),optimize=True)
                    result[part][n][:,jb.retained_columns,:] += np.einsum(
                        "mba,bc->mac",getattr(jb,part).conj(),ai[name],optimize=True)
    return {n:ClusterCurrentIntegral(result["canonical"][n],result["regulator"][n],prescription) for n in (3,4)}


def cluster_quartic_source_h2_current_between(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order,normalized):
    """Normal-ordered quartic source contribution only, with both H2 norms."""
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit Boolean two-endpoint normalization choice required")
    coeff = cluster_quartic_source_h2_coefficients(outgoing,incoming,transfer,charge=charge,
        mean_total_transverse_GeV=mean_total_transverse_GeV,prescription=prescription,
        quadrature_order=quadrature_order)
    values = []
    for part in ("canonical","regulator"):
        value = sum(incoming.owner.pilot.coupling**n*getattr(c,part) for n,c in coeff.items())
        if normalized:
            value = np.einsum("ab,mbc,cd->mad",outgoing.whitener,value,incoming.whitener,optimize=True)
        values.append(value)
    return ClusterCurrentIntegral(*values,prescription)
