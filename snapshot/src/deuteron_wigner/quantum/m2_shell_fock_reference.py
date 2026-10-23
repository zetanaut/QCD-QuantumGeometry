"""Raw complete-shell free-Q matching with coherent g V1 + g^2 V2 sources.

This is a separate resolution reference. Canonical QQ and Q6 contributions
of the earlier order-four benchmark are NOT implicitly transferred across
cutoffs. No source compression or diagonal free-energy assumption is used.
"""
from dataclasses import dataclass

import numpy as np

from .m2_order_resolved_reference import _source
from .m2_canonical_good_current import good_current_density
from .m2_shell_free_propagation import ShellFreePropagation
from .m2_shell_current import shell_density_between


@dataclass(frozen=True)
class _RawSector:
    source: object
    propagation: ShellFreePropagation


class ShellFockReference:
    physical=False
    complete_gs4_QCD=False

    def __init__(self,pilot,*,cubic_sources,quartic_sources):
        cubic_sources,quartic_sources=tuple(cubic_sources),tuple(quartic_sources)
        cubic={s.space.sector:s for s in cubic_sources}
        quartic={s.space.sector:s for s in quartic_sources}
        if (not cubic or cubic.keys()!=quartic.keys() or len(cubic)!=len(cubic_sources)
                or len(quartic)!=len(quartic_sources)):
            raise ValueError("matching distinct explicitly supplied source sectors required")
        self.pilot=pilot
        self.sectors=[]
        for name,c in cubic.items():
            source=_source(pilot,c,quartic[name])
            for power,columns,v in source.components:
                if v.shape!=(source.space.dimension,len(columns)) or not np.all(np.isfinite(v)):
                    raise ValueError("finite source on explicit raw coordinates required")
            propagation=ShellFreePropagation(source.space,b_GeV=pilot.width_GeV,
                       quark_mass_GeV=pilot.quark_mass_GeV,gluon_mass_GeV=pilot.gluon_mass_GeV)
            self.sectors.append(_RawSector(source,propagation))
        self.sectors=tuple(self.sectors)

    def waves(self,energy):
        return tuple({power:s.propagation.resolve(energy,v)
                      for power,_,v in s.source.components} for s in self.sectors)

    def return_coefficient(self,power,energy,*,norm=False):
        if isinstance(power,(bool,np.bool_)) or power not in (2,3,4):
            raise ValueError("return power must be 2,3,4")
        if not isinstance(norm,(bool,np.bool_)):
            raise ValueError("norm selection must be Boolean")
        result=np.zeros_like(self.pilot.matrix)
        for s,wave in zip(self.sectors,self.waves(energy)):
            for left,cl,vl in s.source.components:
                for right,cr,vr in s.source.components:
                    if left+right==power:
                        result[np.ix_(cl,cr)]+=(wave[left] if norm else vl).conj().T@wave[right]
        return result

    def current_coefficient(self,power,energy_out,energy_in,transfer,*,charge):
        if isinstance(power,(bool,np.bool_)) or power not in (0,2,3,4):
            raise ValueError("current power must be 0,2,3,4")
        if power==0:
            return good_current_density(self.pilot,transfer,charge=charge)
        result=np.zeros_like(self.pilot.matrix)
        for s,wf,wi in zip(self.sectors,self.waves(energy_out),self.waves(energy_in)):
            space=s.source.space
            for right,cr,_ in s.source.components:
                rho=shell_density_between(space,space,wi[right],transfer,
                                         b_GeV=self.pilot.width_GeV,charge=charge)
                for left,cl,_ in s.source.components:
                    if left+right==power:
                        result[np.ix_(cl,cr)]+=wf[left].conj().T@rho
        return result

    def effective_hamiltonian(self,energy):
        g=self.pilot.coupling
        return self.pilot.matrix+sum(g**n*self.return_coefficient(n,energy) for n in (2,3,4))

    def norm_metric(self,energy):
        g=self.pilot.coupling
        return np.eye(len(self.pilot.matrix))+sum(
            g**n*self.return_coefficient(n,energy,norm=True) for n in (2,3,4))


def build_shell_fock_reference(pilot,*,out_cutoff):
    """Explicitly request both canonical five-body source families, with free Q."""
    from .m2_excluded_fock_action import build_excluded_source
    from .m2_instantaneous_pair_exit import build_combined_pair_source
    from .m2_instantaneous_sea_exit import build_instantaneous_sea_source
    return ShellFockReference(pilot,
        cubic_sources=[build_excluded_source(pilot,s,out_cutoff=out_cutoff)
                       for s in ("qqqgg","qqqq-qbar")],
        quartic_sources=[build_combined_pair_source(pilot,out_cutoff=out_cutoff),
                         build_instantaneous_sea_source(pilot,out_cutoff=out_cutoff)])
