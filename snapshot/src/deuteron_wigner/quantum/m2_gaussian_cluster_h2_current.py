"""Common H2 four-current numerator: physical free/contact/Wick terms.

All wave and current products are truncated at gs^4 before applying both
existing norm maps. Oscillator-projector transport, finite longitudinal
envelope partners and combined Ward matching are not silently supplied.
This assembly is an inspectable finite representative, not physical QCD.
"""
from dataclasses import dataclass
import numpy as np
from .m2_gaussian_cluster_h2 import GaussianClusterSnapshot,require_same_cluster_definition
from .m2_gaussian_cluster_excluded_current import _definition,cluster_cubic_h2_coefficients
from .m2_gaussian_cluster_quartic_source_current import cluster_quartic_source_h2_coefficients
from .m2_gaussian_cluster_quartic_pair_actions import cluster_quartic_pair_h2_coefficients
from .m2_gaussian_cluster_wick_current import cluster_wick_minus_h2_coefficients
from .m2_canonical_good_current import good_current_density_between,parton_charge
from .m2_free_minus_current import free_minus_current_between
from .m2_transverse_current_projection import free_transverse_current_between
from .m2_shell_current import shell_density_between
from .m2_shell_minus_current import shell_minus_current_between
from .m2_shell_transverse_current import shell_transverse_current_between


def _endpoints(outgoing,incoming,transfer,mean,prescription,quadrature_order):
    if not all(isinstance(x,GaussianClusterSnapshot) for x in (outgoing,incoming)):
        raise ValueError("both actual common-cluster H2 snapshots required")
    f,i=outgoing.owner,incoming.owner
    require_same_cluster_definition(f.pilot,i.pilot)
    for key in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV","gluon_mass_GeV","coupling"):
        if getattr(f.pilot,key)!=getattr(i.pilot,key):
            raise ValueError("H2 current endpoints disagree on "+key)
    for fs,ins in ((f.spaces5,i.spaces5),(f.spaces6,i.spaces6)):
        if fs.keys()!=ins.keys() or any(fs[n].cutoff!=ins[n].cutoff for n in ins):
            raise ValueError("same declared physical Q sectors and resolutions required")
    p=i.pilot
    d,m,q,_,_,_=_definition(p.field_regulator,transfer,mean,prescription,
        quadrature_order,p.width_GeV,p.quark_mass_GeV,p.total_pplus_GeV)
    return d,m,q


def cluster_free_h2_coefficients(outgoing,incoming,transfer,*,charge,
        mean_total_transverse_GeV,prescription,quadrature_order):
    """Every free one-body P/Q5/Q6 insertion on the SAME regulated H2 waves.

    Power zero is PP. Q5 products have powers 2,3,4; Q6 has power 4.
    Free orbital/Dirac kernels are unmodified: the regulator and paired Wick
    H already enter the wave recurrence. Their contact currents are separate.
    """
    delta,mean,_=_endpoints(outgoing,incoming,transfer,mean_total_transverse_GeV,
                             prescription,quadrature_order)
    f,i=outgoing.owner,incoming.owner;p=i.pilot
    args=dict(total_pplus_GeV=p.total_pplus_GeV,charge=charge,mean_total_transverse=mean)
    pp=np.zeros((4,len(f.pilot.matrix),len(p.matrix)),complex)
    pp[0]=good_current_density_between(f.pilot,p,delta,charge=charge)
    pp[1]=free_minus_current_between(f.pilot,p,delta,**args)
    pp[2:]=free_transverse_current_between(f.pilot,p,delta,**args)
    result={n:np.zeros_like(pp) for n in range(5)}
    result[0]=pp
    def apply(fs,ins,v,mu):
        if mu==0:
            return shell_density_between(fs,ins,v,delta,b_GeV=p.width_GeV,charge=charge)
        if mu==1:
            return shell_minus_current_between(fs,ins,v,delta,b_GeV=p.width_GeV,
                mass_GeV=p.quark_mass_GeV,**args)
        return shell_transverse_current_between(fs,ins,v,delta,b_GeV=p.width_GeV,
            component=mu-2,**args)
    af,ai=outgoing.five_wave_coefficients,incoming.five_wave_coefficients
    ni=len(p.matrix)
    for n,ins in i.spaces5.items():
        joined=np.column_stack([a[n] for a in ai])
        for mu in range(4):
            values=apply(f.spaces5[n],ins,joined,mu)
            for left in range(3):
                for right in range(3):
                    power=left+right+2
                    if power>4:continue
                    result[power][mu]+=af[left][n].conj().T@values[:,right*ni:(right+1)*ni]
    for n,ins in i.spaces6.items():
        for mu in range(4):
            value=apply(f.spaces6[n],ins,incoming.six_wave_coefficient[n],mu)
            result[4][mu]+=outgoing.six_wave_coefficient[n].conj().T@value
    return result


@dataclass(frozen=True)
class ClusterH2CurrentAssembly:
    outgoing: GaussianClusterSnapshot
    incoming: GaussianClusterSnapshot
    transfer_GeV: tuple
    mean_total_transverse_GeV: tuple
    charge: str
    loop_flavor_charges: tuple
    prescription: str
    quadrature_order: int
    contributions: dict
    wick_components: dict

    free_contact_Wick_numerator_complete=True
    oscillator_projector_transport_included=False
    longitudinal_envelope_gauged=False
    complete_H2_matched_current=False
    physical=False

    @property
    def coefficients(self):
        return {n:sum(block[n] for block in self.contributions.values()) for n in range(5)}

    def current(self,*,normalized):
        if not isinstance(normalized,(bool,np.bool_)):
            raise ValueError("explicit Boolean two-endpoint normalization choice required")
        g=self.incoming.owner.pilot.coupling
        value=sum(g**n*a for n,a in self.coefficients.items())
        if normalized:
            value=np.einsum("ab,mbc,cd->mad",self.outgoing.whitener,value,self.incoming.whitener,optimize=True)
        return value


def assemble_cluster_h2_current(outgoing,incoming,transfer,*,charge,
        loop_flavor_charges,mean_total_transverse_GeV,prescription,quadrature_order):
    """Free+cubic+quartic+ordered/subtracted/finite Wick, not cutoff completion.

    Every non-overlapping family stays separately inspectable. Wick
    subtraction is paired at each constituent weight BEFORE projection;
    separately projected large divergent terms are diagnostics, not inputs
    to a numerically catastrophic subtract-after-projection calculation.
    The loop
    flavors must describe the same explicit U(1) charge as the external
    fields, not independent electromagnetic weights hidden in a loop.
    No paired Hamiltonian is inserted again, and no full-current flag changes.
    """
    delta,mean,order=_endpoints(outgoing,incoming,transfer,mean_total_transverse_GeV,
                               prescription,quadrature_order)
    loop=np.asarray(loop_flavor_charges,float)
    expected=parton_charge(np.array([0,2]),"q",charge)
    if loop.shape!=(2,) or not np.all(np.isfinite(loop)) or not np.array_equal(loop,expected):
        raise ValueError("explicit two loop flavors must match the same physical U(1) charges")
    args=dict(charge=charge,mean_total_transverse_GeV=mean,
              prescription=prescription,quadrature_order=order)
    free=cluster_free_h2_coefficients(outgoing,incoming,delta,**args)
    contributions={"free":free}
    shape=free[0].shape
    for family,builder in (("cubic",cluster_cubic_h2_coefficients),
                           ("quartic_source",cluster_quartic_source_h2_coefficients),
                           ("quartic_pair",cluster_quartic_pair_h2_coefficients)):
        blocks=builder(outgoing,incoming,delta,**args)
        for part in ("canonical","regulator"):
            contributions[family+"_"+part]={n:(getattr(blocks[n],part).copy() if n in blocks
                                               else np.zeros(shape,complex)) for n in range(5)}
    wick={}
    for part in ("canonical","regulator","subtraction","finite","total"):
        values=cluster_wick_minus_h2_coefficients(outgoing,incoming,delta,charge=charge,
            loop_flavor_charges=tuple(loop),prescription=prescription,component=part)
        block={n:np.zeros(shape,complex) for n in range(5)}
        for n,v in values.items():block[n][1]=v
        if part=="total":contributions["Wick_paired"]=block
        else:wick[part]=block
    for blocks in (*contributions.values(),*wick.values()):
        for value in blocks.values():value.setflags(write=False)
    return ClusterH2CurrentAssembly(outgoing,incoming,delta,mean,charge,tuple(loop),
                                    prescription,order,contributions,wick)


def cluster_transverse_projector_h2_coefficients(outgoing,incoming,projector,*,
        outgoing_vectors,incoming_vectors):
    """Bind the existing finite-parent P/source/Q tangent to actual snapshots.

    This returns TRANSVERSE coefficients only. It supplies no independently
    derived minus component and makes no assertion that a finite reported
    parent contains every Gaussian interaction tail. Both limitations must
    remain visible when testing the assembled physical current.
    """
    from .m2_h2_projector_current import H2ProjectorCurrent
    from .m2_source_vectors import validate_source_vectors
    if not isinstance(projector,H2ProjectorCurrent):
        raise ValueError("explicit actual P/source/Q projector current required")
    if not all(isinstance(s,GaussianClusterSnapshot) for s in (outgoing,incoming)):
        raise ValueError("both actual common-cluster H2 snapshots required")
    q=projector.Q_current
    if q.paired_Wick_in_regulator_wrapper:
        raise ValueError("common H2 already contains Wick H; no second wrapper")
    for endpoint,snapshot in (("outgoing",outgoing),("incoming",incoming)):
        owner=snapshot.owner;p=owner.pilot;outer=q._model(endpoint)
        require_same_cluster_definition(p,outer.pilot)
        for key in ("total_k2","cutoff","target","jz_twice","width_GeV",
                    "quark_mass_GeV","gluon_mass_GeV","coupling"):
            if getattr(p,key)!=getattr(outer.pilot,key):
                raise ValueError("projector endpoint disagrees on "+key)
        energy=q.energy_out if endpoint=="outgoing" else q.energy_in
        if snapshot.energy_GeV2!=energy:
            raise ValueError("same explicit energy at each projector endpoint required")
        spaces={**owner.spaces5,**owner.spaces6}
        if spaces.keys()!=q.projector.vertices.keys() or any(
                spaces[n].cutoff!=v.inner_cutoff for n,v in q.projector.vertices.items()):
            raise ValueError("projector must target the actual H2 inner Q resolutions")
        # These are mathematical bindings, not equality of dimensions.
        for name,actual,expected in (
                ("Hamiltonian",q.effective_hamiltonian(endpoint=endpoint),snapshot.effective_hamiltonian),
                ("norm",q.metric(endpoint=endpoint),snapshot.metric)):
            if not np.allclose(actual,expected,atol=3e-11,rtol=3e-12):
                raise ValueError("projector does not represent the same H2 "+name)
    vf=validate_source_vectors(outgoing.owner.pilot,outgoing_vectors)
    vi=validate_source_vectors(incoming.owner.pilot,incoming_vectors)
    f=projector.embeddings["outgoing"]@vf
    i=projector.embeddings["incoming"]@vi
    pp=q.total_pplus_GeV
    h={}
    for endpoint,p in projector.parents.items():
        h[endpoint]={0:p.h0,1:p.emission_per_gs,
            2:p.cross_pairs_per_gs2+p.instantaneous_fermion_per_gs2+p.wick_mass_per_gs2}
    shape=(2,vf.shape[1],vi.shape[1])
    retained={n:np.zeros(shape,complex) for n in range(5)}
    for n in (0,1,2):
        for j,k in enumerate(projector.vertices):
            retained[n][j]=(f.conj().T@k@h["incoming"][n]@i+
                             f.conj().T@h["outgoing"][n]@k@i)/(2*pp)
    args=dict(outgoing_vectors=vf,incoming_vectors=vi)
    source=projector.loop_coefficients(**args)
    excluded=q.coefficients(**args)
    return {
        "retained_projector":retained,
        "source_projector":{n:source.get(n,np.zeros(shape,complex)) for n in range(5)},
        "Q_projector":{n:excluded.get(n,np.zeros(shape,complex)) for n in range(5)},
    }
