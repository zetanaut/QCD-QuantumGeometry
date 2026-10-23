"""One common-cluster regulated H2 Hamiltonian and norm through g_s^4.

All represented PP/W55/U65 and source words share the same ordered kernel
and Wick H subtraction. Complete electromagnetic regulator/loop currents
are still being derived: this is not physical renormalization or a chosen
four-current. The plus density is an explicitly named wave-map diagnostic.
"""
from dataclasses import dataclass, fields
from functools import partial, lru_cache
import numpy as np

from .m2_gaussian_field_regulator import GaussianClusterRegulator
from .m2_gaussian_cluster_wick import GaussianClusterWick
from .m2_gaussian_cluster_interactions import (
    cluster_pair_contact, cluster_retained_emission, cluster_six_cubic,
    cluster_pair_current_orbital)
from .m2_gaussian_field_sources import apply_field_source
from .m2_coupled_pilot import CoupledPilot, build_coupled_pilot
from .m2_excluded_fock_kinematics import SPECIES
from .m2_order_resolved_reference import OrderResolvedSource
from .m2_shell_fock_reference import ShellFockReference, _RawSector
from .m2_shell_free_propagation import ShellFreePropagation
from .m2_five_body_factorized import FiveBodyContactAction
from .m2_pair_current_factorized import FactorizedPairCurrentAction
from .m2_shell_canonical_qq import FactorizedCanonicalQQSourceAction, ShellCanonicalQQInsertion
from .m2_six_body_factorized import FactorizedSixBodyCubicAction
from .m2_six_body_quartic import COMPONENTS
from .m2_canonical_six_body import CanonicalSixBodyExcursion
from .m2_fourth_order_h2 import FourthOrderH2, FourthOrderSnapshot, current_between


def _cached(function):
    return lru_cache(maxsize=8192)(function)


def _retained_contact(incoming, outgoing, pair, b, ci, co=None, *, regulator):
    species = ("q",)*3+(("g",) if len(incoming)==4 else ())
    return cluster_pair_contact(incoming, outgoing, pair, b, ci, co,
                                 regulator=regulator, species=species)


def _five_contact(sector, incoming, outgoing, pair, b, cutoff, *, regulator):
    return cluster_pair_contact(incoming, outgoing, pair, b, cutoff,
                                 regulator=regulator, species=SPECIES[sector])


def _conversion_contact(incoming, outgoing, b, cutoff, *, regulator):
    return cluster_pair_contact(incoming, outgoing, (3,4), b, cutoff,
        regulator=regulator, species=SPECIES["qqqgg"], out_species=SPECIES["qqqq-qbar"])


@dataclass(frozen=True, kw_only=True)
class GaussianClusterPilot(CoupledPilot):
    field_regulator: GaussianClusterRegulator
    total_pplus_GeV: float
    wick_operators: GaussianClusterWick
    wick_mass_per_gs2: np.ndarray


def require_same_cluster_definition(left, right, *, total_pplus_GeV=None):
    """Bind field/Wick conventions before combining otherwise nested bases.

    Bare endpoints return False; a mixed or different cluster prescription
    is not an alternative representation of the same finite Hamiltonian.
    Resolution and physical base parameters are checked by the caller.
    """
    typed = [isinstance(p, GaussianClusterPilot) for p in (left, right)]
    if any(typed) and not all(typed):
        raise ValueError("both endpoints must share the explicit cluster regulator definition")
    if not any(typed):
        return False
    if (left.field_regulator != right.field_regulator
            or left.wick_operators != right.wick_operators
            or left.total_pplus_GeV != right.total_pplus_GeV
            or (total_pplus_GeV is not None and left.total_pplus_GeV != total_pplus_GeV)):
        raise ValueError("identical cluster regulator, Wick conditions and total P+ required")
    return True


def build_cluster_pilot(*, total_k2, cutoff, target, jz_twice, b_GeV,
                        quark_mass_GeV, gluon_mass_GeV, coupling,
                        regulator, total_pplus_GeV, wick_operators):
    if not isinstance(regulator, GaussianClusterRegulator):
        raise ValueError("explicit common active-cluster regulator required")
    if not isinstance(wick_operators, GaussianClusterWick) or wick_operators.regulator != regulator:
        raise ValueError("Wick H operator must use the identical common regulator")
    if wick_operators.quark_flavors != 2:
        raise ValueError("the represented up/down sea source requires the same two Wick trace flavors")
    pp = float(total_pplus_GeV)
    if not np.isfinite(pp) or pp <= 0:
        raise ValueError("positive finite explicit total P+ required")
    base = build_coupled_pilot(total_k2=total_k2, cutoff=cutoff, target=target,
        jz_twice=jz_twice, b_GeV=b_GeV, quark_mass_GeV=quark_mass_GeV,
        gluon_mass_GeV=gluon_mass_GeV, coupling=coupling,
        _contact_kernel=_cached(partial(_retained_contact, regulator=regulator)),
        _emission_orbital=_cached(partial(cluster_retained_emission,
                                          regulator=regulator, total_pplus_GeV=pp)))
    wick = wick_operators.retained_mass_coefficient(base)
    values = {f.name:getattr(base,f.name) for f in fields(CoupledPilot)}
    values["matrix"] = base.matrix+base.coupling**2*wick
    values["omitted_terms"] = ("four-current is a separate explicit-prescription assembly: m2_gaussian_cluster_four_current",
        "physical finite conditions and regulator-independent matching",
        "dynamical covariance, higher Fock sectors, zero modes and nuclear/GTMD construction")
    return GaussianClusterPilot(**values, field_regulator=regulator,
        total_pplus_GeV=pp, wick_operators=wick_operators, wick_mass_per_gs2=wick)


def _source(pilot, kind, cutoff):
    return apply_field_source(pilot, np.eye(len(pilot.matrix)), kind=kind, out_cutoff=cutoff,
        regulator=pilot.field_regulator, total_pplus_GeV=pilot.total_pplus_GeV,
        total_transverse_GeV=(0.,0.))


def _bind_space(left, right):
    for key in ("sector","total_k2","cutoff","target","jz_twice"):
        if getattr(left,key) != getattr(right,key):
            raise ValueError("common regulator sources disagree on "+key)
    if (left.longitudinal != right.longitudinal or left.orbitals != right.orbitals
            or not np.array_equal(left.coordinates,right.coordinates)):
        raise ValueError("exact common-regulator coordinate map is missing")


class ClusterFockReference(ShellFockReference):
    def __init__(self, pilot, *, out_cutoff):
        if not isinstance(pilot, GaussianClusterPilot):
            raise ValueError("explicit cluster-regulated retained pilot required")
        self.pilot = pilot
        self.source_images, sectors = {}, []
        columns = np.arange(len(pilot.matrix))
        for name,quartic in (("qqqgg","combined"),("qqqq-qbar","sea")):
            c, q = _source(pilot,name,out_cutoff), _source(pilot,quartic,out_cutoff)
            _bind_space(c.space,q.space)
            source = OrderResolvedSource(pilot,c.space,((1,columns,c.values),(2,columns,q.values)))
            propagation = ShellFreePropagation(c.space,b_GeV=pilot.width_GeV,
                quark_mass_GeV=pilot.quark_mass_GeV,gluon_mass_GeV=pilot.gluon_mass_GeV)
            sectors.append(_RawSector(source,propagation))
            self.source_images[name] = (c,q)
        self.sectors = tuple(sectors)


class ClusterQQAction(FactorizedCanonicalQQSourceAction):
    scope = "common-cluster regulated ordered W55, including explicitly paired Wick H subtraction"
    def __init__(self, reference):
        if not isinstance(reference, ClusterFockReference):
            raise ValueError("common regulated source reference required")
        self.reference = reference
        self.spaces = {s.source.space.sector:s.source.space for s in reference.sectors}
        p, reg = reference.pilot, reference.pilot.field_regulator
        self.contacts = FiveBodyContactAction(self.spaces,b_GeV=p.width_GeV,
            _contact_kernel=_cached(partial(_five_contact,regulator=reg)),
            _conversion_kernel=_cached(partial(_conversion_contact,regulator=reg)))
        if reg.reference_shape is None:
            self.pair_current = FactorizedPairCurrentAction(self.spaces,b_GeV=p.width_GeV,
                _orbital_kernel=_cached(partial(cluster_pair_current_orbital,regulator=reg)))
        else:
            from .m2_gaussian_reference_pair import CorrelatedReferencePairCurrentAction
            self.pair_current = CorrelatedReferencePairCurrentAction(self.spaces,
                b_GeV=p.width_GeV,regulator=reg)
        self.wick_diagonal = {n:p.wick_operators.shell_mass_diagonal(s) for n,s in self.spaces.items()}

    def apply(self, columns):
        result = super().apply(columns)
        return {n:result[n]+self.wick_diagonal[n][:,None]*columns[n] for n in result}

    def component_returns(self, waves):
        result = super().component_returns(waves)
        result["paired_Wick_H"] = sum(waves[n].conj().T@(w[:,None]*waves[n])
                                      for n,w in self.wick_diagonal.items())
        return result


class ClusterSixExcursion(CanonicalSixBodyExcursion):
    def __init__(self, reference, *, out_cutoff):
        if not isinstance(reference, ClusterFockReference):
            raise ValueError("common regulated source reference required")
        self.reference = reference
        p = reference.pilot
        self.spaces5 = {s.source.space.sector:s.source.space for s in reference.sectors}
        self.propagation5 = {s.source.space.sector:s.propagation for s in reference.sectors}
        self.V = {s.source.space.sector:next(v for power,_,v in s.source.components if power==1)
                  for s in reference.sectors}
        self.U = FactorizedSixBodyCubicAction(self.spaces5,out_cutoff=out_cutoff,
            b_GeV=p.width_GeV,mass_GeV=p.quark_mass_GeV,
            _orbital_kernel=_cached(partial(cluster_six_cubic,
                regulator=p.field_regulator,total_pplus_GeV=p.total_pplus_GeV)))
        self.spaces6 = self.U.outgoing
        self.propagation6 = {n:ShellFreePropagation(s,b_GeV=p.width_GeV,
            quark_mass_GeV=p.quark_mass_GeV,gluon_mass_GeV=p.gluon_mass_GeV)
            for n,s in self.spaces6.items()}
        self.W = {n:np.zeros((s.dimension,len(p.matrix)),complex) for n,s in self.spaces6.items()}
        self.direct_component_norms = {}
        for component in COMPONENTS:
            image = _source(p,component,out_cutoff)
            _bind_space(image.space,self.spaces6[image.space.sector])
            self.W[image.space.sector] += image.values
            self.direct_component_norms[component] = float(np.linalg.norm(image.values))


@dataclass(frozen=True)
class GaussianClusterSnapshot(FourthOrderSnapshot):
    def current(self, *args, **kwargs):
        raise ValueError("four-current requires assemble_cluster_h2_four_current with explicit prescriptions; request plus_density for the good-current-only map")

    def plus_density(self, transfer, *, charge, normalized):
        """The wave-map plus density, not a completed physical four-current."""
        return cluster_plus_density_between(self,self,transfer,charge=charge,normalized=normalized)


def cluster_plus_density_between(outgoing, incoming, transfer, *, charge, normalized):
    """Both H2 norms and energy/spin endpoints in one common field definition."""
    if not isinstance(outgoing, GaussianClusterSnapshot) or not isinstance(incoming, GaussianClusterSnapshot):
        raise ValueError("both explicitly regulated cluster snapshots required")
    f, i = outgoing.owner, incoming.owner
    if (f.field_regulator != i.field_regulator or f.wick_operators != i.wick_operators
            or f.pilot.total_pplus_GeV != i.pilot.total_pplus_GeV):
        raise ValueError("identical field regulator, Wick conditions and total P+ at both endpoints required")
    for name in (*f.spaces5, *f.spaces6):
        fs = f.spaces5.get(name, f.spaces6.get(name))
        ins = i.spaces5.get(name, i.spaces6.get(name))
        if fs.cutoff != ins.cutoff:
            raise ValueError("same declared Q-sector resolution at both endpoints required")
    return current_between(outgoing,incoming,transfer,charge=charge,normalized=normalized)


class GaussianClusterH2(FourthOrderH2):
    # Hamiltonian/waves only. The separately bound finite four-current may
    # be complete without promoting this object or selecting its current.
    complete_regulated_H2_matching = False
    complete_gauged_contractions = False
    def __init__(self, pilot, *, excluded_cutoff, six_cutoff):
        reference = ClusterFockReference(pilot,out_cutoff=excluded_cutoff)
        action = ClusterQQAction(reference)
        qq = ShellCanonicalQQInsertion(reference,action=action)
        six = ClusterSixExcursion(reference,out_cutoff=six_cutoff)
        super().__init__(reference,QQ_insertion=qq,six_body=six)
        self.field_regulator = pilot.field_regulator
        self.wick_operators = pilot.wick_operators

    def snapshot(self, energy):
        raw = super().snapshot(energy)
        return GaussianClusterSnapshot(**{f.name:getattr(raw,f.name) for f in fields(FourthOrderSnapshot)})
