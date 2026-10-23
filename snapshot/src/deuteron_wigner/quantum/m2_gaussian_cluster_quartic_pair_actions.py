"""Physical four-leg quartic pair currents on P and Q5, per gs².

Number-current, constrained-fermion and pair-annihilation/conversion
monomials are gauged before projection. The pair-annihilation kernel is
the physical extension of the auxiliary I=1/2 source representation; its
agreement on H2 source waves must be checked, not inferred on arbitrary
isospin columns. No auxiliary gluon is assigned a charge or envelope.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations,product
from math import pi,sqrt
from operator import index
import numpy as np
from .m2_canonical_good_current import parton_charge
from .m2_c117_pair_current import color_pair_matrix
from .m2_canonical_instantaneous_fermion import ordered_qg_color
from .m2_excluded_instantaneous_fermion import ordered_five_qg_color
from .m2_excluded_current_scattering import current_pair_color,canonical_gluon_current_factor
from .m2_excluded_annihilation import annihilation_color
from .m2_excluded_pair_conversion import instantaneous_conversion_kernel,current_conversion_kernel
from .m2_quadratic_minus_fock import conversion_minus_kernel
from .m2_excluded_fock_kinematics import SPECIES as FIVE,longitudinal_states
from .m2_coupled_longitudinal_jacobi import longitudinal_basis
from .m2_gaussian_cluster_pair_current import cluster_pair_current_orbitals,ordered_qg_coefficients
from .m2_gaussian_cluster_current import ClusterCurrentIntegral
from .m2_gaussian_cluster_excluded_current import _definition
from .m2_gaussian_cluster_h2 import GaussianClusterPilot,require_same_cluster_definition
from .m2_source_vectors import source_tensor

SPECIES = {"qqq":("q",)*3,"qqqg":("q",)*3+("g",),**FIVE}
CHARGED_COMPONENTS = ("fermion_number","qg_number","IF_scattering","IF_conversion",
                       "IG_conversion","sea_annihilation")
NEUTRAL_COMPONENTS = ("gg_number","gg_annihilation","A4")


@dataclass(frozen=True)
class PhysicalPairTerm:
    component: str
    incoming_sector: str
    outgoing_sector: str
    incoming: tuple
    outgoing: tuple
    pair: tuple
    hamiltonian: np.ndarray
    canonical_minus: np.ndarray
    selected: np.ndarray
    mapped: np.ndarray
    charges: tuple

    def reverse(self):
        return PhysicalPairTerm(self.component,self.outgoing_sector,self.incoming_sector,
            self.outgoing,self.incoming,self.pair,self.hamiltonian.conj().T,self.canonical_minus.conj().T,
            self.mapped,self.selected,self.charges[2:]+self.charges[:2])


@lru_cache(maxsize=4)
def _spins(sector):
    dims = tuple(2 if s=="g" else 4 for s in SPECIES[sector])
    spins=np.array(list(product(*(range(n) for n in dims))))
    spins.setflags(write=False)
    return spins,dims


def _charged_terms(component,si,so,inc,out,pair,h,junit,selected,mapped,charge):
    """Group equal PHYSICAL charges, not amplitudes, before orbital evaluation.

    junit is the independent canonical minus matrix for unit outgoing-quark
    flavor. Only IF terms use it; their charged fermion is first in pair.
    """
    if not np.any(h) and not np.any(junit):return
    isp,_ = _spins(si);osp,_ = _spins(so)
    selected,mapped = np.asarray(selected),np.asarray(mapped)
    # Annihilation/recreation is many-to-one in the active spin labels.
    # Each tensor term must be injective so packed accumulation retains
    # every incoming helicity amplitude, not an advanced-index overwrite.
    if len(np.unique(mapped))!=len(mapped):
        active=isp[selected][:,pair]
        for labels in np.unique(active,axis=0):
            keep=np.all(active==labels,axis=1)
            if len(np.unique(mapped[keep]))!=keep.sum():
                raise ArithmeticError("physical pair map changed spectator spin labels")
            yield from _charged_terms(component,si,so,inc,out,pair,h,junit,
                selected[keep],mapped[keep],charge)
        return
    e = np.column_stack([parton_charge(osp[mapped,j],SPECIES[so][j],charge) for j in pair]
                        +[parton_charge(isp[selected,j],SPECIES[si][j],charge) for j in pair])
    for values in np.unique(e,axis=0):
        take = np.all(e==values,axis=1)
        yield PhysicalPairTerm(component,si,so,tuple(inc),tuple(out),tuple(pair),np.asarray(h,complex),
            values[0]*np.asarray(junit,complex),selected[take],mapped[take],tuple(values))


def physical_pair_terms(sector,total_k2,*,total_pplus_GeV,charge,include_conversion):
    """Actual charged quartic Hamiltonian/current monomials in a source sector.

    Pure-gluon gg number, A4 and pair-annihilation currents are exactly zero
    because every participating field has zero electromagnetic/flavor charge.
    Their nonzero Hamiltonians are unchanged in the supplied H2 owner.
    """
    if sector not in SPECIES or sector not in ("qqq","qqqg","qqqgg","qqqq-qbar"):
        raise ValueError("actual retained or Q5 pair-current sector required")
    kinds = SPECIES[sector];spins,dims = _spins(sector);all_spins=np.arange(len(spins))
    if not isinstance(include_conversion,(bool,np.bool_)):
        raise ValueError("explicit Boolean pair-conversion inventory required")
    parton_charge(np.arange(4),"q",charge)
    states=(longitudinal_basis if len(kinds)<5 else longitudinal_states)(sector,total_k2)
    for inc in states:
        for pair in combinations(range(len(kinds)),2):
            a,b=pair
            if kinds[a]==kinds[b]=="g":continue
            color=(color_pair_matrix(sector,a,b) if len(kinds)<5 else current_pair_color(sector,a,b))
            label="qg_number" if "g" in (kinds[a],kinds[b]) else "fermion_number"
            for ka in range(2 if kinds[a]=="g" else 1,inc[a]+inc[b],2):
                dk=ka-inc[a]
                if dk==0:continue  # exact instantaneous-gluon Q0 exclusion
                out=list(inc);out[a],out[b]=ka,inc[a]+inc[b]-ka;out=tuple(out)
                h=2*total_k2/(pi*dk**2)*color
                for j in pair:
                    if kinds[j]=="g":h=h*canonical_gluon_current_factor(out[j],inc[j])
                yield from _charged_terms(label,sector,sector,inc,out,pair,h,np.zeros_like(h),
                    all_spins,all_spins,charge)
        if "g" in kinds:
            for q in range(3):
                for g in (j for j,s in enumerate(kinds) if s=="g"):
                    for kg in range(2,inc[q]+inc[g],2):
                        out=list(inc);out[q],out[g]=inc[q]+inc[g]-kg,kg;out=tuple(out)
                        modes=(out[q],out[g],inc[q],inc[g])
                        for hq,hg in product((-1,1),repeat=2):
                            selected=np.flatnonzero((2*(spins[:,q]%2)-1==hq)&(2*spins[:,g]-1==hg))
                            h=0.;j=0.
                            for crossed in (False,True):
                                ch,cj=ordered_qg_coefficients(modes,total_k2=total_k2,total_pplus_GeV=total_pplus_GeV,
                                    quark_helicity=hq,gluon_helicity=hg,crossed=crossed,flavor_charge=1.)
                                color=ordered_qg_color(q,crossed) if len(kinds)==4 else ordered_five_qg_color(q,g,crossed)
                                h=h+ch*color;j=j+cj[1]*color
                            yield from _charged_terms("IF_scattering",sector,sector,inc,out,(q,g),h,j,selected,selected,charge)
        if sector=="qqqq-qbar":
            for q in range(4):
                selected=np.flatnonzero((spins[:,q]//2==spins[:,4]//2)&(spins[:,q]%2!=spins[:,4]%2))
                total=inc[q]+inc[4]
                h=2*total_k2/(pi*total**2)*annihilation_color(q)
                for kq in range(1,total,2):
                    out=list(inc);out[q],out[4]=kq,total-kq;out=tuple(out)
                    for f,hq in product(range(2),range(2)):
                        changed=spins[selected].copy();changed[:,q]=2*f+hq;changed[:,4]=2*f+1-hq
                        mapped=np.ravel_multi_index(changed.T,dims)
                        yield from _charged_terms("sea_annihilation",sector,sector,inc,out,(q,4),h,np.zeros_like(h),
                            selected,mapped,charge)
        if sector=="qqqgg" and include_conversion:
            so="qqqq-qbar";_,outdims=_spins(so)
            for kq in range(1,sum(inc[3:]),2):
                out=inc[:3]+(kq,sum(inc[3:])-kq)
                for hs in product((-1,1),repeat=4):
                    hq,ha,h1,h2=hs
                    selected=np.flatnonzero((2*spins[:,3]-1==h1)&(2*spins[:,4]-1==h2))
                    for f in range(2):
                        changed=np.column_stack((spins[selected,:3],np.full(len(selected),2*f+(hq+1)//2),
                                                 np.full(len(selected),2*f+(ha+1)//2)))
                        mapped=np.ravel_multi_index(changed.T,outdims)
                        for label,kernel in (("IF_conversion",instantaneous_conversion_kernel),("IG_conversion",current_conversion_kernel)):
                            h=sqrt(2)*kernel(inc,out,hs)
                            j=(sqrt(2)*conversion_minus_kernel(inc,out,hs,total_pplus_GeV=total_pplus_GeV)
                               if label=="IF_conversion" else np.zeros_like(h))
                            terms=_charged_terms(label,sector,so,inc,out,(3,4),h,j,selected,mapped,charge)
                            for term in terms:
                                yield term
                                yield term.reverse()


@lru_cache(maxsize=4096)
def _orbitals(si,so,inc,out,pair,ci,co,b,regulator,pp,delta,mean,charges,prescription,order):
    value=cluster_pair_current_orbitals(inc,out,pair,ci,co,b,species=SPECIES[si],out_species=SPECIES[so],
        charges=charges,regulator=regulator,total_pplus_GeV=pp,transfer_GeV=delta,
        mean_total_transverse_GeV=mean,hamiltonian_coefficient=1.,
        canonical_current_coefficient=(0.,1.,0.,0.),prescription=prescription,quadrature_order=order)
    value.canonical.setflags(write=False);value.regulator.setflags(write=False)
    return value


def _part_and_component(part,component):
    if part not in ("canonical","regulator","total"):
        raise ValueError("explicit canonical, regulator or total current part required")
    if isinstance(component,(bool,np.bool_)) or index(component) not in range(4):
        raise ValueError("explicit (+,-,x,y) component index required")
    return index(component)


def _current_factors(term,orbital,mu,part):
    if part in ("canonical","total") and np.any(term.canonical_minus):
        yield orbital.canonical[mu],term.canonical_minus
    if part in ("regulator","total") and np.any(term.hamiltonian):
        yield orbital.regulator[mu],term.hamiltonian


def cluster_quartic_pair_current_between(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """Complete retained PP quartic current, both actual spin endpoints."""
    if not all(isinstance(p,GaussianClusterPilot) for p in (outgoing,incoming)):
        raise ValueError("both actual common-cluster retained pilots required")
    require_same_cluster_definition(outgoing,incoming)
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(outgoing,key)!=getattr(incoming,key):
            raise ValueError("retained pair-current endpoints disagree on "+key)
    p=incoming
    delta,mean,order,b,_,pp=_definition(p.field_regulator,transfer,mean_total_transverse_GeV,
        prescription,quadrature_order,p.width_GeV,p.quark_mass_GeV,p.total_pplus_GeV)
    images=[np.zeros((4,len(outgoing.matrix),len(incoming.matrix)),complex) for _ in range(2)]
    for sector in ("qqq","qqqg"):
        uf,ui=[source_tensor(owner,sector,np.eye(len(owner.matrix))) for owner in (outgoing,incoming)]
        if sector=="qqqg":
            uf=uf.reshape(*uf.shape[:3],128,uf.shape[-1]);ui=ui.reshape(*ui.shape[:3],128,ui.shape[-1])
        lookup={m:j for j,m in enumerate(longitudinal_basis(sector,p.total_k2))}
        for term in physical_pair_terms(sector,p.total_k2,total_pplus_GeV=pp,charge=charge,include_conversion=False):
            orbit=_orbitals(sector,sector,term.incoming,term.outgoing,term.pair,p.cutoff,outgoing.cutoff,
                b,p.field_regulator,pp,delta,mean,term.charges,prescription,order)
            bra=np.take(uf[lookup[term.outgoing]],term.mapped,axis=2)
            ket=np.take(ui[lookup[term.incoming]],term.selected,axis=2)
            for image,part in zip(images,("canonical","regulator")):
                for mu in (1,2,3):
                    for orb,color in _current_factors(term,orbit,mu,part):
                        if not np.any(orb):continue
                        image[mu]+=np.einsum("oasf,oi,ab,ibsn->fn",bra.conj(),orb,color,ket,optimize=True)
    return ClusterCurrentIntegral(*images,prescription)


class ClusterFiveQuarticCurrent:
    """All physical normal-ordered Q5 quartic contacts on raw packed columns.

    The electromagnetic operator is not restricted to auxiliary I=1/2.
    Its strong-H source-sector equivalence is separately verified before
    binding its A1_f† J55 A1_i return to common H2. No full QCD claim.
    """
    physical=False
    complete_H2_matched_current=False
    longitudinal_envelope_gauged=False
    exact_neutral_current_components=NEUTRAL_COMPONENTS

    def __init__(self,outgoing,incoming,transfer,*,regulator,b_GeV,total_pplus_GeV,charge,
            mean_total_transverse_GeV,prescription,quadrature_order):
        from .m2_excluded_fock_action import IntermediateSpace
        if set(incoming)!=set(outgoing) or set(incoming)!={"qqqgg","qqqq-qbar"}:
            raise ValueError("both physical Q5 sectors at both current endpoints required")
        for group in (incoming,outgoing):
            example=group["qqqgg"]
            if any(not isinstance(s,IntermediateSpace) or s.sector!=n or
                   any(getattr(s,k)!=getattr(example,k) for k in ("total_k2","cutoff","target","jz_twice"))
                   for n,s in group.items()):
                raise ValueError("actual consistent physical Q5 endpoint spaces required")
        if any(getattr(incoming["qqqgg"],k)!=getattr(outgoing["qqqgg"],k) for k in ("total_k2","target")):
            raise ValueError("Q5 pair-current endpoints disagree on total momentum/target")
        self.delta,self.mean,self.order,self.b,_,self.P=_definition(regulator,transfer,
            mean_total_transverse_GeV,prescription,quadrature_order,b_GeV,0.,total_pplus_GeV)
        self.incoming,self.outgoing=dict(incoming),dict(outgoing)
        self.regulator,self.charge,self.prescription=regulator,charge,prescription
        self.terms=tuple(term for n in incoming for term in physical_pair_terms(n,incoming[n].total_k2,
            total_pplus_GeV=self.P,charge=charge,include_conversion=True))

    @classmethod
    def from_models(cls,outgoing,incoming,transfer,*,charge,
            mean_total_transverse_GeV,prescription,quadrature_order):
        from .m2_gaussian_cluster_h2 import GaussianClusterH2
        if not all(isinstance(x,GaussianClusterH2) for x in (outgoing,incoming)):
            raise ValueError("both actual common-cluster H2 models required")
        f,i=outgoing.pilot,incoming.pilot
        if not require_same_cluster_definition(f,i):
            raise ValueError("identical common-cluster H2 definitions required")
        for key in ("total_k2","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
            if getattr(f,key)!=getattr(i,key):
                raise ValueError("current endpoints disagree on "+key)
        return cls(outgoing.spaces5,incoming.spaces5,transfer,regulator=i.field_regulator,
            b_GeV=i.width_GeV,total_pplus_GeV=i.total_pplus_GeV,charge=charge,
            mean_total_transverse_GeV=mean_total_transverse_GeV,
            prescription=prescription,quadrature_order=quadrature_order)

    def apply(self,columns,*,component,part,components=None):
        from .m2_six_body_cubic import SixBodyCubicAction,five_statistics_project
        from .m2_six_body_transverse_contact import _block,_add
        mu=_part_and_component(part,component)
        if components is not None and (not components or not set(components)<=set(CHARGED_COMPONENTS+NEUTRAL_COMPONENTS)):
            raise ValueError("explicit nonempty physical quartic component set required")
        count=SixBodyCubicAction._validate(self.incoming,columns)
        result={n:np.zeros((s.dimension,count),complex) for n,s in self.outgoing.items()}
        if mu==0 or (part=="canonical" and mu!=1):
            # Quartic canonical contacts have only minus; plus density
            # and transverse regulator partners are separate operators.
            return result
        source={n:five_statistics_project(s,columns[n]) for n,s in self.incoming.items()}
        lookup={side:{n:{m:j for j,m in enumerate(s.longitudinal)} for n,s in group.items()}
                for side,group in (("i",self.incoming),("f",self.outgoing))}
        for term in self.terms:
            if components is not None and term.component not in components:continue
            si,so=term.incoming_sector,term.outgoing_sector
            isp,osp=self.incoming[si],self.outgoing[so]
            orbit=_orbitals(si,so,term.incoming,term.outgoing,term.pair,isp.cutoff,osp.cutoff,
                self.b,self.regulator,self.P,self.delta,self.mean,term.charges,self.prescription,self.order)
            block=_block(isp,lookup["i"][si][term.incoming],term.selected,source[si])
            for orb,color in _current_factors(term,orbit,mu,part):
                if not np.any(orb):continue
                value=np.einsum("oi,ab,ibsk->oask",orb,color,block,optimize=True)
                _add(osp,lookup["f"][so][term.outgoing],term.mapped,value,result[so])
        return {n:five_statistics_project(s,result[n]) for n,s in self.outgoing.items()}

    def adjoint(self,columns,*,component,part,components=None):
        reverse=ClusterFiveQuarticCurrent(self.incoming,self.outgoing,tuple(-x for x in self.delta),
            regulator=self.regulator,b_GeV=self.b,total_pplus_GeV=self.P,charge=self.charge,
            mean_total_transverse_GeV=self.mean,prescription=self.prescription,quadrature_order=self.order)
        return reverse.apply(columns,component=component,part=part,components=components)


def cluster_quartic_pair_h2_coefficients(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """Quartic PAIR-current words: J2_PP at g² and A1_f† J2_55 A1_i at g⁴.

    Source currents P↔Q5/Q6 and Wick contractions are separate contributions.
    A2/C3 with J2_55 start at g⁵/g⁶; Q6 diagonal J2_66 starts at g⁶.
    These higher-order subsets must not be imported into an order-four result.
    Both endpoints use the actual common-regulated H2 waves, including their
    source-domain pair-annihilation restriction; no arbitrary raw isospin
    Hamiltonian equivalence is assumed.
    """
    from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot
    if not all(isinstance(x,GaussianClusterSnapshot) for x in (outgoing,incoming)):
        raise ValueError("both actual common-cluster H2 snapshots required")
    f,i=outgoing.owner,incoming.owner
    args=dict(charge=charge,mean_total_transverse_GeV=mean_total_transverse_GeV,
              prescription=prescription,quadrature_order=quadrature_order)
    action=ClusterFiveQuarticCurrent.from_models(f,i,transfer,**args)
    if f.pilot.cutoff!=i.pilot.cutoff:
        raise ValueError("same declared retained resolution at both H2 endpoints required")
    for fs,ins in ((f.spaces5,i.spaces5),(f.spaces6,i.spaces6)):
        if fs.keys()!=ins.keys():
            raise ValueError("same declared physical Q sectors required")
        if any(fs[n].cutoff!=ins[n].cutoff for n in ins):
            raise ValueError("same declared Q-sector resolution at both H2 endpoints required")
    retained=cluster_quartic_pair_current_between(f.pilot,i.pilot,transfer,**args)
    af,ai=outgoing.five_wave_coefficients[0],incoming.five_wave_coefficients[0]
    values=[]
    for part in ("canonical","regulator"):
        value=np.zeros_like(retained.canonical)
        for mu in (1,2,3):
            applied=action.apply(ai,component=mu,part=part)
            value[mu]=sum(af[n].conj().T@applied[n] for n in af)
        values.append(value)
    return {2:retained,4:ClusterCurrentIntegral(*values,prescription)}


def cluster_quartic_pair_h2_current_between(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order,normalized):
    """Normal-ordered quartic pair contribution, with both H2 norms."""
    if not isinstance(normalized,(bool,np.bool_)):
        raise ValueError("explicit Boolean two-endpoint normalization choice required")
    coeff=cluster_quartic_pair_h2_coefficients(outgoing,incoming,transfer,charge=charge,
        mean_total_transverse_GeV=mean_total_transverse_GeV,prescription=prescription,
        quadrature_order=quadrature_order)
    values=[]
    for part in ("canonical","regulator"):
        value=sum(incoming.owner.pilot.coupling**n*getattr(c,part) for n,c in coeff.items())
        if normalized:
            value=np.einsum("ab,mbc,cd->mad",outgoing.whitener,value,incoming.whitener,optimize=True)
        values.append(value)
    return ClusterCurrentIntegral(*values,prescription)
