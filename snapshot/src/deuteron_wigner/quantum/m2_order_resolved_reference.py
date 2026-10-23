"""Coherent cubic/quartic P-Q sources with a power-resolved finite free-Q parent.

Both actual five-body channels are caller supplied. Compression preserves the
UNION of each coupling-order image independently of g, including at g=0.
The parent is not complete order-g^4 QCD: QQ interactions, six-body exits,
loop/current renormalization and physical finite conditions remain open.
"""
from dataclasses import dataclass
import numpy as np
from scipy import linalg

from .m2_canonical_good_current import good_current_density
from .m2_excluded_fock_matching import _energy_groups,FreeEnergySlice,ReachableSector
from .m2_matched_h2_reduction import FiniteParentReduction


@dataclass(frozen=True)
class OrderResolvedSource:
    retained_pilot: object
    space: object
    components: tuple  # (power, explicit retained columns, raw source matrix)

    def full_coupling(self,coupling):
        g=float(coupling)
        if not np.isfinite(g):
            raise ValueError("finite coupling required")
        result=np.zeros((self.space.dimension,len(self.retained_pilot.matrix)),complex)
        for power,columns,matrix in self.components:
            result[:,columns]+=g**power*matrix
        return result


@dataclass(frozen=True)
class OrderResolvedSector:
    source: OrderResolvedSource
    slices: tuple
    coupling_by_power: dict
    energies_GeV2: np.ndarray
    reconstruction_residual_by_power: dict

    # These existing coordinate-only methods depend on source.space, slices
    # and energies, not the cubic coupling convention.
    lift=ReachableSector.lift
    project=ReachableSector.project
    density=ReachableSector.density

    def coupling(self,g):
        g=float(g)
        if not np.isfinite(g):
            raise ValueError("finite coupling required")
        return sum(g**power*matrix for power,matrix in self.coupling_by_power.items())


def _source(pilot,cubic,quartic):
    if cubic.retained_pilot is not pilot or quartic.retained_pilot is not pilot:
        raise ValueError("each source must bind this exact retained pilot")
    for name in ("sector","total_k2","cutoff","target","jz_twice"):
        if getattr(cubic.space,name)!=getattr(quartic.space,name):
            raise ValueError("source regulators/basis labels disagree on "+name)
    if not np.array_equal(cubic.space.coordinates,quartic.space.coordinates):
        raise ValueError("raw source coordinates disagree")
    nq=pilot.bases[0].shape[1];n=len(pilot.matrix)
    if (not np.array_equal(cubic.retained_columns,np.arange(nq,n))
            or not np.array_equal(quartic.retained_columns,np.arange(nq))):
        raise ValueError("cubic qqqg and quartic qqq columns must retain their explicit identities")
    return OrderResolvedSource(pilot,cubic.space,(
        (1,cubic.retained_columns,cubic.matrix_per_gs),
        (2,quartic.retained_columns,quartic.matrix_per_gs2)))


def _compress(source):
    p=source.retained_pilot
    energy=source.space.free_energies(p.width_GeV,p.quark_mass_GeV,p.gluon_mass_GeV)
    for _,columns,matrix in source.components:
        if matrix.shape!=(source.space.dimension,len(columns)) or not np.all(np.isfinite(matrix)):
            raise ValueError("finite raw matrices on explicit source columns required")
    slices=[];energies=[];blocks={power:[] for power,_,_ in source.components}
    lost={power:0. for power,_,_ in source.components};offset=0
    for rows in _energy_groups(source.space):
        # Do not let a tiny g² weight define numerical support or suppress
        # a quartic direction already absent from the cubic reachable space.
        atlas=np.column_stack([matrix[rows] for _,_,matrix in source.components])
        u,s,_=linalg.svd(atlas,full_matrices=False)
        threshold=8*np.finfo(float).eps*max(atlas.shape)*s[0]
        rank=int(np.sum(s>threshold))
        if not rank:
            for power,_,matrix in source.components:
                lost[power]+=np.linalg.norm(matrix[rows])**2
            continue
        u=u[:,:rank]
        if np.ptp(energy[rows])>3e-14:
            raise ArithmeticError("source range does not share an exact free energy")
        e=float(energy[rows[0]])
        slices.append(FreeEnergySlice(rows,u,slice(offset,offset+rank),e))
        for power,columns,matrix in source.components:
            reduced=u.conj().T@matrix[rows]
            lost[power]+=np.linalg.norm(matrix[rows]-u@reduced)**2
            full=np.zeros((rank,len(p.matrix)),complex);full[:,columns]=reduced
            blocks[power].append(full)
        energies.extend([e]*rank);offset+=rank
    if not slices:
        raise ValueError("empty order-resolved source range")
    for power,_,matrix in source.components:
        lost[power]=float(np.sqrt(lost[power]))
        if lost[power]>2e-12*max(1.,np.linalg.norm(matrix)):
            raise ArithmeticError("compression loses a resolved coupling-order source")
    return OrderResolvedSector(source,tuple(slices),
        {power:np.vstack(parts) for power,parts in blocks.items()},np.array(energies),lost)


class OrderResolvedFockReference:
    """Explicit g V1+g² V2 exits, full cross terms, matched norm/current.

    Both source collections are required; no missing quartic is inferred
    to be zero. The supplied quartic qqqgg source can be a declared component
    or coherent sum; construction does not certify its physical completeness.
    """
    physical=False

    def __init__(self,pilot,*,cubic_sources,quartic_sources):
        cubic_sources,quartic_sources=tuple(cubic_sources),tuple(quartic_sources)
        cubic={s.space.sector:s for s in cubic_sources}
        quartic={s.space.sector:s for s in quartic_sources}
        if (not cubic or cubic.keys()!=quartic.keys() or len(cubic)!=len(cubic_sources)
                or len(quartic)!=len(quartic_sources)):
            raise ValueError("matching distinct explicitly supplied cubic/quartic sectors required")
        self.pilot=pilot
        self.sectors=tuple(_compress(_source(pilot,c,quartic[name])) for name,c in cubic.items())
        poles=np.concatenate([s.energies_GeV2 for s in self.sectors])
        coupling=np.vstack([s.coupling(pilot.coupling) for s in self.sectors])
        self.matrix=np.block([[pilot.matrix,coupling.conj().T],[coupling,np.diag(poles)]])
        self.retained_isometry=np.eye(len(self.matrix),dtype=complex)[:,:len(pilot.matrix)]
        self.reduction=FiniteParentReduction(self.matrix,self.retained_isometry,
            scope="canonical cubic/quartic five-body exits, free Q, coherent orders; NOT complete g^4 QCD")

    def density(self,transfer,*,charge="baryon"):
        return linalg.block_diag(good_current_density(self.pilot,transfer,charge=charge),*(
            s.density(transfer,b_GeV=self.pilot.width_GeV,charge=charge) for s in self.sectors))

    @staticmethod
    def _denominators(sector,energy):
        d=float(energy)-sector.energies_GeV2
        if not np.all(np.isfinite(d)) or np.min(abs(d))<1e-10:
            raise ValueError("finite off-pole energy required")
        return d

    def return_coefficient(self,power,energy,*,norm=False):
        """Coefficient of g^power in H return or induced norm (powers2,3,4)."""
        if power not in (2,3,4) or isinstance(power,(bool,np.bool_)):
            raise ValueError("return power must be 2,3,4")
        if not isinstance(norm,(bool,np.bool_)):
            raise ValueError("norm selection must be Boolean")
        result=np.zeros_like(self.pilot.matrix)
        for sector in self.sectors:
            d=self._denominators(sector,energy)
            for left,a in sector.coupling_by_power.items():
                for right,b in sector.coupling_by_power.items():
                    if left+right==power:
                        result+=a.conj().T@(b/(d[:,None]**(2 if norm else 1)))
        return result

    def current_coefficient(self,power,energy_out,energy_in,transfer,*,charge):
        """Number-preserving local good current; both energy endpoints explicit."""
        if isinstance(power,(bool,np.bool_)):
            raise ValueError("local current power must be 0,2,3,4")
        if power==0:
            return good_current_density(self.pilot,transfer,charge=charge)
        if power not in (2,3,4) or isinstance(power,(bool,np.bool_)):
            raise ValueError("local current power must be 0,2,3,4")
        result=np.zeros_like(self.pilot.matrix)
        for sector in self.sectors:
            df=self._denominators(sector,energy_out)
            di=self._denominators(sector,energy_in)
            density=sector.density(transfer,b_GeV=self.pilot.width_GeV,charge=charge)
            for left,a in sector.coupling_by_power.items():
                for right,b in sector.coupling_by_power.items():
                    if left+right==power:
                        result+=(a/df[:,None]).conj().T@density@(b/di[:,None])
        return result

    def full_reconstructed_residual(self,energy,retained):
        v=np.asarray(retained,complex)
        if v.shape!=(len(self.pilot.matrix),) or not np.all(np.isfinite(v)):
            raise ValueError("finite retained state required")
        rp=self.pilot.matrix@v-float(energy)*v
        normq=residualq=0.
        for sector in self.sectors:
            self._denominators(sector,energy)
            source=sector.source
            energies=source.space.free_energies(self.pilot.width_GeV,self.pilot.quark_mass_GeV,
                                                self.pilot.gluon_mass_GeV)
            coupling=source.full_coupling(self.pilot.coupling)
            q=coupling@v/(energy-energies)
            rp+=coupling.conj().T@q
            rq=(energies-energy)*q+coupling@v
            normq+=float(np.vdot(q,q).real);residualq+=float(np.vdot(rq,rq).real)
        return float(np.sqrt(np.vdot(rp,rp).real+residualq)),normq


def build_order_resolved_reference(pilot,*,out_cutoff):
    """Explicitly request both implemented canonical five-body source families."""
    from .m2_excluded_fock_action import build_excluded_source
    from .m2_instantaneous_pair_exit import build_combined_pair_source
    from .m2_instantaneous_sea_exit import build_instantaneous_sea_source
    return OrderResolvedFockReference(pilot,
        cubic_sources=[build_excluded_source(pilot,s,out_cutoff=out_cutoff)
                       for s in ("qqqgg","qqqq-qbar")],
        quartic_sources=[build_combined_pair_source(pilot,out_cutoff=out_cutoff),
                         build_instantaneous_sea_source(pilot,out_cutoff=out_cutoff)])
