"""Common physical-leg cubic currents on every represented charged P/Q route.

The orbital contains the discrete gluon factor and signed physical charge
for a unit underlying flavor. This file supplies only color, flavor and
normalized Fock statistics, not a second charge sign or M² conversion.
Canonical and regulator parts remain separate. These are order-g contacts,
not a full H2 current or a gauged finite longitudinal envelope.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import sqrt
from operator import index
import numpy as np
from .m2_canonical_good_current import parton_charge
from .m2_coupled_longitudinal_jacobi import oscillator_shell
from .m2_excluded_fock_action import IntermediateSpace
from .m2_excluded_fock_kinematics import split_routes,longitudinal_states,color_source
from .m2_excluded_transverse_contact import _add_current
from .m2_six_body_vertices import color_source as color_source6
from .m2_six_body_cubic import five_statistics_project,SixBodyCubicAction
from .m2_six_body_transverse_contact import SixBodyTransverseContact,_block,_add
from .m2_gaussian_field_regulator import GaussianClusterRegulator,_array
from .m2_gaussian_cluster_current import PRESCRIPTION,cluster_cubic_current_orbitals
from .m2_gaussian_cluster_h2 import GaussianClusterPilot,require_same_cluster_definition


def _definition(regulator,transfer,mean,prescription,order,b,mass,pp):
    if not isinstance(regulator,GaussianClusterRegulator) or prescription!=PRESCRIPTION:
        raise ValueError("explicit common cluster and physical-leg endpoint prescription required")
    delta = tuple(_array(transfer,(2,),"transverse transfer"))
    mean = tuple(_array(mean,(2,),"mean transverse momentum"))
    b,mass,pp = float(b),float(mass),float(pp)
    if not np.all(np.isfinite([b,mass,pp])) or min(b,pp)<=0 or mass<0:
        raise ValueError("positive width/P+ and nonnegative finite vertex mass required")
    if isinstance(order,(bool,np.bool_)) or index(order)<2:
        raise ValueError("explicit exact path quadrature order at least two required")
    return delta,mean,index(order),b,mass,pp


@lru_cache(maxsize=1024)
def _orbital(route,ci,co,b,mass,helicities,regulator,pp,delta,mean,prescription,order):
    value = cluster_cubic_current_orbitals(route,ci,co,b,mass,helicities,
        regulator=regulator,total_pplus_GeV=pp,transfer_GeV=delta,
        mean_total_transverse_GeV=mean,flavor_charge=1.,
        prescription=prescription,quadrature_order=order)
    value.canonical.setflags(write=False)
    value.regulator.setflags(write=False)
    return value


@dataclass(frozen=True)
class ClusterExcludedCubicCurrent:
    incoming_pilot: object
    outgoing_space: object
    retained_columns: np.ndarray
    transfer: tuple
    canonical: np.ndarray
    regulator: np.ndarray
    prescription: str
    complete_H2_matched_current = False
    longitudinal_envelope_gauged = False
    physical = False

    @property
    def total(self):
        return self.canonical+self.regulator


def build_cluster_excluded_cubic_current(pilot,space,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """Actual P's qqqg columns to all Q5 coordinates, per gs, (+,-,x,y).

    The P Hamiltonian already owns its common field/Wick definition.
    Both outgoing helicities and all requested oscillator states are kept.
    The pure three-gluon electromagnetic/flavor contact is exactly zero.
    Physical absorption is the reversed-transfer emission's adjoint.
    """
    if not isinstance(pilot,GaussianClusterPilot) or not isinstance(space,IntermediateSpace):
        raise ValueError("actual common-cluster pilot and explicit physical Q5 space required")
    if any(getattr(space,k)!=getattr(pilot,k) for k in ("total_k2","target")):
        raise ValueError("current endpoints disagree on total momentum/target")
    if pilot.field_regulator!=pilot.wick_operators.regulator:
        raise ValueError("pilot field and Wick definitions disagree")
    delta,mean,order,b,mass,pp = _definition(pilot.field_regulator,transfer,
        mean_total_transverse_GeV,prescription,quadrature_order,
        pilot.width_GeV,pilot.quark_mass_GeV,pilot.total_pplus_GeV)
    parton_charge(np.arange(4),"q",charge)
    incoming = longitudinal_states("qqqg",pilot.total_k2)
    ni = len(oscillator_shell(3,pilot.cutoff))
    ranks = [u.shape[1] for u in pilot.bases]
    columns = np.arange(ranks[0],sum(ranks));count = len(columns)
    source = np.zeros((len(incoming),ni,2,64,2,count),complex)
    offset = 0
    for hg,u in enumerate(pilot.bases[1:]):
        n = u.shape[1]
        source[:,:,:,:,hg,offset:offset+n] = u.toarray().reshape(len(incoming),ni,2,64,n)
        offset += n
    ins,outs = ({s:i for i,s in enumerate(states)} for states in (incoming,space.longitudinal))
    sf = np.array(list(product(range(4),repeat=3)))
    channel = "q_to_qg" if space.sector=="qqqgg" else "g_to_pair"
    images = [np.zeros((4,space.dimension,count),complex) for _ in range(2)]
    for route in split_routes(pilot.total_k2,channel):
        color = color_source(channel,route.parent)
        for hg,hf,hi in product((-1,1),range(2),range(2)):
            hs = (2*hf-1,hg,2*hi-1) if channel=="q_to_qg" else (2*hf-1,2*hi-1,hg)
            orbital = _orbital(route,pilot.cutoff,space.cutoff,b,mass,hs,
                pilot.field_regulator,pp,delta,mean,prescription,order)
            if channel=="q_to_qg":
                selected = np.flatnonzero(sf[:,route.parent]%2==hi)
                changed = sf[selected].copy()
                changed[:,route.parent] = 2*(changed[:,route.parent]//2)+hf
                out_sf = np.ravel_multi_index(changed.T,(4,4,4))
                weight = parton_charge(sf[selected,route.parent],"q",charge)
                blocks = []
                for old_hg in range(2):
                    value = np.take(source[ins[route.incoming],:,:,:,old_hg],selected,axis=2)
                    spins = (out_sf*2+old_hg)*2+(hg+1)//2
                    blocks.append((value,spins,weight))
            else:
                value = source[ins[route.incoming],:,:,:,(hg+1)//2]
                blocks = []
                for flavor in range(2):
                    q,anti = 2*flavor+hf,2*flavor+hi
                    spins = (np.arange(64)*4+q)*4+anti
                    blocks.append((value,spins,np.full(64,float(parton_charge(q,"q",charge)))))
            for image,components in zip(images,(orbital.canonical,orbital.regulator)):
                for mu,matrix in enumerate(components):
                    if not np.any(matrix):
                        continue
                    for value,spins,weight in blocks:
                        amp = np.einsum("oi,ab,ibsk->oask",matrix,color,value,optimize=True)
                        amp *= weight[None,None,:,None]
                        _add_current(space,image[mu],outs[route.outgoing],spins,amp)
    fock = sqrt(2) if channel=="q_to_qg" else 2.
    images = [np.array([fock*space.statistics_project(a) for a in image]) for image in images]
    return ClusterExcludedCubicCurrent(pilot,space,columns,delta,*images,prescription)


class ClusterSixCubicCurrent:
    """Factorized Q5→Q6 four-current action with exact raw Fock statistics.

    The common regulator and all physical parameters are explicit. Use
    from_models to bind this action to both actual common-H2 definitions.
    No giant Q6-by-Q5 matrix or source-rank truncation is constructed.
    """
    physical = False
    complete_H2_matched_current = False
    longitudinal_envelope_gauged = False

    def __init__(self,incoming,outgoing,transfer,*,regulator,b_GeV,mass_GeV,
            total_pplus_GeV,charge,mean_total_transverse_GeV,prescription,quadrature_order):
        delta,mean,order,b,mass,pp = _definition(regulator,transfer,mean_total_transverse_GeV,
            prescription,quadrature_order,b_GeV,mass_GeV,total_pplus_GeV)
        # Reuse only physical route/color/statistics inventory and space
        # validation. No bare transverse current kernel is used here.
        routing = SixBodyTransverseContact(incoming,outgoing,delta,
            b_GeV=b,total_pplus_GeV=pp,charge=charge)
        self.incoming,self.outgoing = routing.incoming,routing.outgoing
        self.routes = tuple(routing.routes)
        self.transfer,self.mean,self.order = delta,mean,order
        self.regulator,self.b,self.mass,self.P = regulator,b,mass,pp
        self.charge,self.prescription = charge,prescription

    @classmethod
    def from_models(cls,outgoing,incoming,transfer,*,charge,
            mean_total_transverse_GeV,prescription,quadrature_order):
        from .m2_gaussian_cluster_h2 import GaussianClusterH2
        if not all(isinstance(x,GaussianClusterH2) for x in (outgoing,incoming)):
            raise ValueError("both actual common-cluster H2 models required")
        f,i = outgoing.pilot,incoming.pilot
        if not require_same_cluster_definition(f,i):
            raise ValueError("identical common-cluster H2 definitions required")
        for key in ("total_k2","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
            if getattr(f,key)!=getattr(i,key):
                raise ValueError("current endpoints disagree on "+key)
        return cls(incoming.spaces5,outgoing.spaces6,transfer,regulator=i.field_regulator,
            b_GeV=i.width_GeV,mass_GeV=i.quark_mass_GeV,total_pplus_GeV=i.total_pplus_GeV,
            charge=charge,mean_total_transverse_GeV=mean_total_transverse_GeV,
            prescription=prescription,quadrature_order=quadrature_order)

    def _terms(self,component,part):
        for s,t,route,fock in self.routes:
            source,target = self.incoming[s],self.outgoing[t]
            li,lo = source.longitudinal.index(route.incoming),target.longitudinal.index(route.outgoing)
            color,spins = color_source6(route),source.spins
            for hg,hf,hi in product((-1,1),range(2),range(2)):
                pair = route.channel=="g_to_pair"
                hs = (2*hf-1,2*hi-1,hg) if pair else (2*hf-1,hg,2*hi-1)
                value = _orbital(route,source.cutoff,target.cutoff,self.b,self.mass,hs,
                    self.regulator,self.P,self.transfer,self.mean,self.prescription,self.order)
                orbital = getattr(value,part)[component]*fock
                if not np.any(orbital):
                    continue
                if not pair:
                    selected = np.flatnonzero(spins[:,route.parent]%2==hi)
                    changed = spins[selected].copy()
                    changed[:,route.parent] = 2*(changed[:,route.parent]//2)+hf
                    out = np.column_stack((changed,np.full(len(selected),(hg+1)//2)))
                    out_spins = np.ravel_multi_index(out.T,target.spin_dims)
                    # The orbital ALREADY supplies the antiquark charge
                    # sign. Multiply the underlying flavor charge only.
                    weight = parton_charge(spins[selected,route.parent],"q",self.charge)
                    yield s,t,li,lo,orbital,color,selected,out_spins,weight
                else:
                    selected = np.flatnonzero(2*spins[:,4]-1==hg)
                    changed = spins[selected]
                    for flavor in range(2):
                        q,anti = 2*flavor+hf,2*flavor+hi
                        out = np.column_stack((changed[:,:3],np.full(len(selected),q),
                            np.full(len(selected),anti),changed[:,3]))
                        out_spins = np.ravel_multi_index(out.T,target.spin_dims)
                        weight = np.full(len(selected),float(parton_charge(q,"q",self.charge)))
                        yield s,t,li,lo,orbital,color,selected,out_spins,weight

    @staticmethod
    def _selection(component,part):
        if isinstance(component,(bool,np.bool_)) or index(component) not in range(4):
            raise ValueError("explicit (+,-,x,y) component 0 through 3 required")
        if part not in ("canonical","regulator","total"):
            raise ValueError("explicit canonical/regulator/total current part required")

    def apply(self,columns,*,component,part):
        self._selection(component,part)
        n = SixBodyCubicAction._validate(self.incoming,columns)
        result = {s:np.zeros((sp.dimension,n),complex) for s,sp in self.outgoing.items()}
        if component==0:  # exact zero plus contact, not the free plus current
            return result
        source = {s:five_statistics_project(sp,columns[s]) for s,sp in self.incoming.items()}
        for s,t,li,lo,orb,color,si,sf,weight in self._terms(component,part):
            block = _block(self.incoming[s],li,si,source[s])
            out = np.einsum("oi,ab,ibsk->oask",orb,color,block,optimize=True)
            out *= weight[None,None,:,None]
            _add(self.outgoing[t],lo,sf,out,result[t])
        return {s:sp.statistics_project(result[s]) for s,sp in self.outgoing.items()}

    def adjoint(self,columns,*,component,part):
        """Same-transfer adjoint; physical absorption uses reversed Delta."""
        self._selection(component,part)
        n = SixBodyCubicAction._validate(self.outgoing,columns)
        result = {s:np.zeros((sp.dimension,n),complex) for s,sp in self.incoming.items()}
        if component==0:
            return result
        source = {s:sp.statistics_project(columns[s]) for s,sp in self.outgoing.items()}
        for s,t,li,lo,orb,color,si,sf,weight in self._terms(component,part):
            block = _block(self.outgoing[t],lo,sf,source[t])
            block *= weight[None,None,:,None].conj()
            back = np.einsum("oi,ab,oask->ibsk",orb.conj(),color.conj(),block,optimize=True)
            _add(self.incoming[s],li,si,back,result[s])
        return {s:five_statistics_project(sp,result[s]) for s,sp in self.incoming.items()}


def cluster_cubic_h2_coefficients(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """All represented CUBIC contact-current words through H2 order gs^4.

    Power one: PP. Powers two/three/four: both P↔Q5 endpoints against
    A1/A2/C3 respectively. Power four also has both A6† J65 A1 returns.
    Sources, Wick H, raw backflow and norms already belong to the snapshots.
    No free, quartic, Wick-direct or projector current is substituted here.
    """
    from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot
    from .m2_gaussian_cluster_current import ClusterCurrentIntegral
    from .m2_gaussian_cluster_current_actions import cluster_cubic_current_between
    if not all(isinstance(x,GaussianClusterSnapshot) for x in (outgoing,incoming)):
        raise ValueError("both actual common-cluster H2 snapshots required")
    f,i = outgoing.owner,incoming.owner
    if not require_same_cluster_definition(f.pilot,i.pilot):
        raise ValueError("identical common-cluster H2 definitions required")
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(f.pilot,key)!=getattr(i.pilot,key):
            raise ValueError("H2 current endpoints disagree on "+key)
    if f.spaces5.keys()!=i.spaces5.keys() or f.spaces6.keys()!=i.spaces6.keys():
        raise ValueError("same declared physical Q sectors required")
    for name in (*i.spaces5,*i.spaces6):
        fs,ins = f.spaces5.get(name,f.spaces6.get(name)),i.spaces5.get(name,i.spaces6.get(name))
        if fs.cutoff!=ins.cutoff:
            raise ValueError("same declared Q-sector resolution at both H2 endpoints required")
    delta = tuple(_array(transfer,(2,),"transverse transfer"));reverse = tuple(-x for x in delta)
    args = dict(charge=charge,mean_total_transverse_GeV=mean_total_transverse_GeV,
                prescription=prescription,quadrature_order=quadrature_order)
    retained = cluster_cubic_current_between(f.pilot,i.pilot,delta,**args)
    parts = {part:{n:np.zeros_like(retained.canonical) for n in (1,2,3,4)}
             for part in ("canonical","regulator")}
    for part in parts:
        parts[part][1] = getattr(retained,part)
    af,ai = outgoing.five_wave_coefficients,incoming.five_wave_coefficients
    for name,s in i.spaces5.items():
        forward = build_cluster_excluded_cubic_current(i.pilot,f.spaces5[name],delta,**args)
        backward = build_cluster_excluded_cubic_current(f.pilot,s,reverse,**args)
        for part,result in parts.items():
            jf,jb = getattr(forward,part),getattr(backward,part)
            for k in range(3):
                result[k+2][:,:,forward.retained_columns] += np.einsum(
                    "ab,mbc->mac",af[k][name].conj().T,jf,optimize=True)
                result[k+2][:,backward.retained_columns,:] += np.einsum(
                    "mba,bc->mac",jb.conj(),ai[k][name],optimize=True)
    forward = ClusterSixCubicCurrent.from_models(f,i,delta,**args)
    backward = ClusterSixCubicCurrent.from_models(i,f,reverse,**args)
    for part,result in parts.items():
        for mu in (1,2,3):  # plus contact is exactly zero
            jf = forward.apply(ai[0],component=mu,part=part)
            jb = backward.apply(af[0],component=mu,part=part)
            for name in i.spaces6:
                result[4][mu] += outgoing.six_wave_coefficient[name].conj().T@jf[name]
                result[4][mu] += jb[name].conj().T@incoming.six_wave_coefficient[name]
    return {n:ClusterCurrentIntegral(parts["canonical"][n],parts["regulator"][n],prescription)
            for n in (1,2,3,4)}


def cluster_cubic_h2_current_between(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order,normalized):
    """The complete cubic contribution only, with both existing H2 norms."""
    from .m2_gaussian_cluster_current import ClusterCurrentIntegral
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit Boolean two-endpoint normalization choice required")
    coefficients = cluster_cubic_h2_coefficients(outgoing,incoming,transfer,charge=charge,
        mean_total_transverse_GeV=mean_total_transverse_GeV,prescription=prescription,
        quadrature_order=quadrature_order)
    g = incoming.owner.pilot.coupling
    values = []
    for part in ("canonical","regulator"):
        value = sum(g**n*getattr(c,part) for n,c in coefficients.items())
        if normalized:
            value = np.einsum("ab,mbc,cd->mad",outgoing.whitener,value,incoming.whitener,optimize=True)
        values.append(value)
    return ClusterCurrentIntegral(*values,prescription)
