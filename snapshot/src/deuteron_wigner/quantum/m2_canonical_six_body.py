"""Leading canonical six-particle H2 excursion: direct plus iterated sources.

The three- and four-particle retained pilot remains unchanged. This new
order-g^4 contribution includes both Q6 channels, all seven normal-ordered
quartic Q6P source components, cubic Q5-to-Q6 paths, five-body backflow,
and raw two-energy local good currents. It is not full order-g^4 QCD,
an interacting-Q resummation, physical calibration, or a full four-current.
"""
import numpy as np
from functools import cached_property

from .m2_six_body_cubic import SixBodyCubicAction
from .m2_six_body_factorized import FactorizedSixBodyCubicAction
from .m2_six_body_quartic import COMPONENTS,build_direct_source
from .m2_shell_free_propagation import ShellFreePropagation
from .m2_shell_current import shell_density_between


class CanonicalSixBodyExcursion:
    physical=False
    complete_gs4_QCD=False

    def __init__(self,reference,*,out_cutoff):
        self.reference=reference
        p=reference.pilot
        self.spaces5={s.source.space.sector:s.source.space for s in reference.sectors}
        self.V={}
        for s in reference.sectors:
            if hasattr(s,"lift"):
                v=s.lift(s.coupling_by_power[1])  # preserve the old compressed control
            else:
                components=[(c,a) for power,c,a in s.source.components if power==1]
                if len(components)!=1:
                    raise ValueError("one explicit raw cubic source order required")
                columns,a=components[0]
                v=np.zeros((s.source.space.dimension,len(p.matrix)),complex)
                v[:,columns]=a
            self.V[s.source.space.sector]=v
        cubic_type=SixBodyCubicAction if out_cutoff<=1 else FactorizedSixBodyCubicAction
        self.U=cubic_type(self.spaces5,out_cutoff=out_cutoff,
            b_GeV=p.width_GeV,mass_GeV=p.quark_mass_GeV)
        self.spaces6=self.U.outgoing
        def propagators(spaces):
            return {n:ShellFreePropagation(s,b_GeV=p.width_GeV,
                quark_mass_GeV=p.quark_mass_GeV,gluon_mass_GeV=p.gluon_mass_GeV) for n,s in spaces.items()}
        self.propagation5=propagators(self.spaces5)
        self.propagation6=propagators(self.spaces6)
        self.W={n:np.zeros((s.dimension,len(p.matrix)),complex) for n,s in self.spaces6.items()}
        self.direct_component_norms={}
        for component in COMPONENTS:
            source=build_direct_source(p,out_cutoff=out_cutoff,component=component)
            target=self.spaces6[source.space.sector]
            # Coordinate binding is verified in full, not inferred from shape.
            for key in ("sector","total_k2","cutoff","target","jz_twice"):
                if getattr(source.space,key)!=getattr(target,key):
                    raise ValueError("direct and iterated six-body coordinates disagree")
            if (source.space.longitudinal!=target.longitudinal
                    or source.space.orbitals!=target.orbitals
                    or not np.array_equal(source.space.coordinates,target.coordinates)):
                raise ValueError("exact six-body coordinate map is missing")
            self.W[target.sector][:,source.retained_columns]+=source.matrix_per_gs2
            self.direct_component_norms[component]=float(np.linalg.norm(source.matrix_per_gs2))

    @cached_property
    def energies5(self):
        """Historical diagonal-only diagnostic; rejects higher shells."""
        p=self.reference.pilot
        return {n:s.free_energies(p.width_GeV,p.quark_mass_GeV,p.gluon_mass_GeV)
                for n,s in self.spaces5.items()}

    @cached_property
    def energies6(self):
        """Historical diagonal-only diagnostic; rejects higher shells."""
        p=self.reference.pilot
        return {n:s.free_energies(p.width_GeV,p.quark_mass_GeV,p.gluon_mass_GeV)
                for n,s in self.spaces6.items()}

    @staticmethod
    def _resolvent(propagators,energy,columns):
        if set(propagators)!=set(columns):
            raise ValueError("every declared propagation sector is required")
        return {n:propagators[n].resolve(energy,a) for n,a in columns.items()}

    def waves(self,energy):
        a1=self._resolvent(self.propagation5,energy,self.V)
        iterated=self.U.apply(a1)
        f={n:self.W[n]+iterated[n] for n in self.W}
        a6=self._resolvent(self.propagation6,energy,f)
        b5=self._resolvent(self.propagation5,energy,self.U.adjoint(a6))
        return dict(five_leading=a1,iterated_source=iterated,six_source=f,
                    six_leading=a6,five_backflow=b5)

    def hamiltonian_coefficient(self,energy):
        a=self.waves(energy)
        return sum(a["six_source"][n].conj().T@v for n,v in a["six_leading"].items())

    def norm_coefficient(self,energy):
        a=self.waves(energy)
        norm=sum(v.conj().T@v for v in a["six_leading"].values())
        cross=sum(a["five_leading"][n].conj().T@b for n,b in a["five_backflow"].items())
        return norm+cross+cross.conj().T

    def current_coefficient(self,energy_out,energy_in,transfer,*,charge):
        p=self.reference.pilot
        af,ai=self.waves(energy_out),self.waves(energy_in)
        result=sum(af["six_leading"][n].conj().T@shell_density_between(
            s,s,ai["six_leading"][n],transfer,b_GeV=p.width_GeV,charge=charge)
            for n,s in self.spaces6.items())
        nP=len(p.matrix)
        for n,s in self.spaces5.items():
            joined=shell_density_between(s,s,np.column_stack((
                ai["five_leading"][n],ai["five_backflow"][n])),transfer,b_GeV=p.width_GeV,charge=charge)
            result+=(af["five_backflow"][n].conj().T@joined[:,:nP]
                     +af["five_leading"][n].conj().T@joined[:,nP:])
        return result

    def return_components(self,energy):
        """Separate W†RW, (URV)†R(URV), and their coherent cross term."""
        a=self.waves(energy);it=a["iterated_source"]
        rw=self._resolvent(self.propagation6,energy,self.W)
        ri=self._resolvent(self.propagation6,energy,it)
        direct=sum(self.W[n].conj().T@v for n,v in rw.items())
        iterated=sum(it[n].conj().T@v for n,v in ri.items())
        cross=sum(self.W[n].conj().T@v for n,v in ri.items())
        return dict(direct=direct,iterated=iterated,cross=cross+cross.conj().T)
