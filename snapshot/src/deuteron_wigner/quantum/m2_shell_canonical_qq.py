"""Canonical single-QQ insertion on complete non-diagonally propagated shells.

The same represented normal-ordered quartic action as the assembled control,
now factorized on raw coordinates. No source-image projection of the W tail,
full QQ resummation, Wick completion, or physical renormalization is implied.
"""
import numpy as np

from .m2_canonical_qq import CanonicalQQInsertion,CanonicalQQSourceAction
from .m2_five_body_factorized import FiveBodyContactAction,apply_terms
from .m2_pair_current_factorized import FactorizedPairCurrentAction


class FactorizedCanonicalQQSourceAction:
    physical=False
    scope=CanonicalQQSourceAction.scope

    def __init__(self,reference):
        self.reference=reference
        self.spaces={s.source.space.sector:s.source.space for s in reference.sectors}
        b=reference.pilot.width_GeV
        self.contacts=FiveBodyContactAction(self.spaces,b_GeV=b)
        self.pair_current=FactorizedPairCurrentAction(self.spaces,b_GeV=b)

    def apply(self,columns):
        contact=self.contacts.apply(columns)
        pair=self.pair_current.apply(columns)
        return {name:contact[name]+pair[name] for name in self.spaces}

    def component_returns(self,waves):
        pair=self.pair_current.apply(waves)  # validates all raw coordinates
        result={label:waves[name].conj().T@apply_terms(terms,waves[name],self.spaces[name].dimension)
                for label,(name,terms) in self.contacts.components.items()}
        result["pair_current_square"]=sum(waves[n].conj().T@pair[n] for n in self.spaces)
        # Isolate the IF conversion; the pair square already owns the IG cross term.
        c=self.contacts;gg,sea="qqqgg","qqqq-qbar"
        converted=np.sqrt(2)*self.spaces[sea].statistics_project(
            apply_terms(c.conversion,waves[gg],self.spaces[sea].dimension))
        cross=waves[sea].conj().T@converted
        result["IF_pair_conversion"]=cross+cross.conj().T
        return result


class ShellCanonicalQQInsertion(CanonicalQQInsertion):
    """Reuse order bookkeeping, replacing only raw source/propagator ownership.

    The assembled sparse/diagonal superclass initializer is deliberately not
    used. Ar=R_N Vr and Br=R_N W Ar retain every raw shell coordinate.
    """
    def __init__(self,reference,*,action):
        if action.reference is not reference:
            raise ValueError("canonical QQ action must bind this exact reference")
        self.reference,self.action,self.label=reference,action,action.scope
        self.propagation={}
        self.sources_by_power={p:{} for p in (1,2)}
        for sector in reference.sectors:
            s=sector.source.space;name=s.sector
            if action.spaces[name] is not s or not hasattr(sector,"propagation"):
                raise ValueError("complete-shell reference with identical action coordinates required")
            self.propagation[name]=sector.propagation
            for p in (1,2):
                entries=[(cols,v) for power,cols,v in sector.source.components if power==p]
                if len(entries)!=1:
                    raise ValueError("one explicitly identified source per coupling order required")
                cols,v=entries[0]
                source=np.zeros((s.dimension,len(reference.pilot.matrix)),complex)
                source[:,cols]=v
                self.sources_by_power[p][name]=source
        self.sources=self.sources_by_power[1]

    def leading_waves(self,energy,source_power):
        if source_power not in (1,2) or isinstance(source_power,(bool,np.bool_)):
            raise ValueError("source power must be1 or2")
        return {n:self.propagation[n].resolve(energy,v)
                for n,v in self.sources_by_power[source_power].items()}

    def order_waves(self,energy,source_power):
        waves=self.leading_waves(energy,source_power)
        acted=self.action.apply(waves)
        return waves,{n:self.propagation[n].resolve(energy,v) for n,v in acted.items()}

    def current_coefficient(self,energy_out,energy_in,transfer,*,charge):
        return self.current_power_coefficient(4,energy_out,energy_in,transfer,charge=charge)

    def free_source_leakage(self,energy):
        raise ValueError("raw complete-shell reference has no compressed free-source image")
