"""All normal-ordered canonical quartic QQ source actions in the five-body model.

Includes the current square, qg constrained-fermion scattering, A^4 and
constrained-fermion gg/sea conversion. Source symmetry is I=1/2 with
quark sign and gluon Bose statistics. This is NOT six-body closure,
unsubtracted Wick terms, renormalized QCD, or a chosen physical state.
"""
import numpy as np

from .m2_excluded_fermion_scattering import fermion_scattering_per_gs2
from .m2_excluded_current_scattering import current_scattering_per_gs2
from .m2_excluded_pair_current import build_pair_current_factorization
from .m2_excluded_instantaneous_fermion import instantaneous_fermion_QQ_per_gs2
from .m2_excluded_four_gluon import four_gluon_QQ_per_gs2
from .m2_excluded_pair_conversion import FiveBodyPairConversion
from .m2_excluded_interaction_insertion import ExcludedInteractionInsertion


class CanonicalQQSourceAction:
    physical=False
    scope="normal-ordered canonical quartic QQ on represented I=1/2, quark-sign/Bose source sector"

    def __init__(self,reference):
        self.reference=reference
        self.pair_current=build_pair_current_factorization(reference)
        self.spaces=self.pair_current.spaces
        b=reference.pilot.width_GeV
        self.block_components={}
        for name,space in self.spaces.items():
            self.block_components[name+"_fermion_scattering"]=(name,fermion_scattering_per_gs2(space,b_GeV=b))
        gg=self.spaces["qqqgg"]
        for channel in ("qg","gg"):
            self.block_components[channel+"_number_scattering"]=(
                "qqqgg",current_scattering_per_gs2(gg,b_GeV=b,channel=channel))
        self.block_components["IF_scattering"]=("qqqgg",instantaneous_fermion_QQ_per_gs2(gg,b_GeV=b))
        self.block_components["A4"]=("qqqgg",four_gluon_QQ_per_gs2(gg,b_GeV=b))
        self.blocks={name:sum(w for target,w in self.block_components.values() if target==name)
                     for name in self.spaces}
        # IG conversion and sea annihilation are already in the pair square.
        self.conversion=FiveBodyPairConversion(self.spaces,b_GeV=b,channel="IF")

    def apply(self,columns):
        pair=self.pair_current.apply(columns)  # validates every coordinate/column
        conversion=self.conversion.apply(columns)
        return {name:self.blocks[name]@columns[name]+pair[name]+conversion[name] for name in self.spaces}

    def component_returns(self,waves):
        """A† W_component A on supplied raw wave columns, BEFORE any g weights."""
        self.pair_current.apply(waves)  # common validation
        result={label:waves[name].conj().T@(op@waves[name])
                for label,(name,op) in self.block_components.items()}
        for label,action in (("pair_current_square",self.pair_current.apply),
                             ("IF_pair_conversion",self.conversion.apply)):
            acted=action(waves)
            result[label]=sum(waves[name].conj().T@acted[name] for name in self.spaces)
        return result


class CanonicalQQInsertion(ExcludedInteractionInsertion):
    """Single canonical QQ insertion. Leading g^4 aliases preserve old API."""
    def __init__(self,reference,*,action):
        if action.reference is not reference:
            raise ValueError("canonical QQ action must bind this exact reference")
        super().__init__(reference,sector_operators={name:(space,action.blocks[name])
            for name,space in action.spaces.items()},label=action.scope)
        self.action=action
        self.sources_by_power={power:{s.source.space.sector:s.lift(s.coupling_by_power[power])
                                      for s in reference.sectors} for power in (1,2)}

    def leading_waves(self,energy,source_power):
        if source_power not in (1,2) or isinstance(source_power,(bool,np.bool_)):
            raise ValueError("source power must be1 or2")
        waves={}
        for name,source in self.sources_by_power[source_power].items():
            d=float(energy)-self.energies[name]
            if not np.all(np.isfinite(d)) or np.min(abs(d))<1e-10:
                raise ValueError("finite off-pole energy required")
            waves[name]=source/d[:,None]
        return waves

    def order_waves(self,energy,source_power):
        waves=self.leading_waves(energy,source_power)
        acted=self.action.apply(waves)
        tails={name:acted[name]/(float(energy)-self.energies[name][:,None]) for name in waves}
        return waves,tails

    def wave_coefficients(self,sector_name,energy):
        waves,tails=self.order_waves(energy,1)
        return waves[sector_name],tails[sector_name]

    def return_power_coefficient(self,power,energy,*,norm=False):
        """Single-W coefficient of g^4,g^5,g^6 from both explicit source orders."""
        if power not in (4,5,6) or isinstance(power,(bool,np.bool_)):
            raise ValueError("single-insertion power must be4,5,6")
        if not isinstance(norm,(bool,np.bool_)):
            raise ValueError("norm selection must be Boolean")
        data={p:self.order_waves(energy,p) for p in (1,2)}
        result=np.zeros_like(self.reference.pilot.matrix)
        for r in (1,2):
            for s in (1,2):
                if r+s+2!=power:continue
                ar,br=data[r];aa,bb=data[s]
                for name in self.sources:
                    if norm:
                        result+=ar[name].conj().T@bb[name]+br[name].conj().T@aa[name]
                    else:
                        result+=self.sources_by_power[r][name].conj().T@bb[name]
        return result

    def current_power_coefficient(self,power,energy_out,energy_in,transfer,*,charge):
        """Same single insertion and both source orders in the two-energy current."""
        from .m2_shell_current import shell_density_between
        if power not in (4,5,6) or isinstance(power,(bool,np.bool_)):
            raise ValueError("single-insertion current power must be4,5,6")
        outgoing={p:self.order_waves(energy_out,p) for p in (1,2)}
        incoming={p:self.order_waves(energy_in,p) for p in (1,2)}
        result=np.zeros_like(self.reference.pilot.matrix)
        for s in (1,2):
            ai,bi=incoming[s]
            for name,space in self.action.spaces.items():
                joined=shell_density_between(space,space,np.column_stack((ai[name],bi[name])),transfer,
                    b_GeV=self.reference.pilot.width_GeV,charge=charge)
                n=ai[name].shape[1]
                for r in (1,2):
                    if r+s+2!=power:continue
                    af,bf=outgoing[r]
                    result+=bf[name].conj().T@joined[:,:n]+af[name].conj().T@joined[:,n:]
        return result
