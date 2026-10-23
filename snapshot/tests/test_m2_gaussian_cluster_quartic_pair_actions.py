"""Physical pair monomials versus source-owned H, bare currents and Ward ends."""
from dataclasses import replace
from functools import lru_cache
import numpy as np
import pytest
from deuteron_wigner.quantum import m2_gaussian_cluster_quartic_pair_actions as current
from deuteron_wigner.quantum.m2_gaussian_cluster_pair_current import pair_current_geometry
from deuteron_wigner.quantum.m2_gaussian_field_regulator import gaussian_shell_integral
from deuteron_wigner.quantum.m2_gaussian_cluster_interactions import cluster_pair_contact
from deuteron_wigner.quantum.m2_two_gluon_minus_current import retained_two_gluon_minus_between
from deuteron_wigner.quantum.m2_quadratic_minus_actions import FiveBodyQuadraticMinus
from deuteron_wigner.quantum.m2_six_body_cubic import five_statistics_project
from deuteron_wigner.quantum.m2_six_body_transverse_contact import _block,_add
from deuteron_wigner.quantum.m2_five_body_factorized import apply_terms
from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_source_vectors import source_tensor
from deuteron_wigner.quantum.m2_coupled_longitudinal_jacobi import longitudinal_basis
from test_m2_gaussian_cluster_excluded_current import pilot,raw
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_current import regulator
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

DELTA=(.12,.16)
MEAN=(.03,.07)
ARGS=dict(charge="electromagnetic",mean_total_transverse_GeV=MEAN,prescription=PRESCRIPTION,quadrature_order=8)


@lru_cache(maxsize=8)
def spaces(cutoff=0,target="PROTON",jz=1):
    return {n:IntermediateSpace(n,9,cutoff,target,jz) for n in ("qqqgg","qqqq-qbar")}


def action(inc=None,out=None,*,width=.8,delta=DELTA,mean=MEAN,charge="electromagnetic",pp=1.3,order=8):
    inc=spaces() if inc is None else inc;out=inc if out is None else out
    return current.ClusterFiveQuarticCurrent(out,inc,delta,regulator=regulator(width),
        b_GeV=.4,total_pplus_GeV=pp,charge=charge,mean_total_transverse_GeV=mean,
        prescription=PRESCRIPTION,quadrature_order=order)


def hamiltonian_terms(action,x,components):
    """Direct original four-leg H integral, not the factorized Q operator."""
    x={n:five_statistics_project(s,x[n]) for n,s in action.incoming.items()}
    result={n:np.zeros((s.dimension,next(iter(x.values())).shape[1]),complex) for n,s in action.outgoing.items()}
    for term in action.terms:
        if term.component not in components:continue
        si,so=action.incoming[term.incoming_sector],action.outgoing[term.outgoing_sector]
        orb=cluster_pair_contact(term.incoming,term.outgoing,term.pair,.4,si.cutoff,so.cutoff,
            regulator=action.regulator,species=current.SPECIES[si.sector],out_species=current.SPECIES[so.sector])
        block=_block(si,si.longitudinal.index(term.incoming),term.selected,x[si.sector])
        val=np.einsum("oi,ab,ibsk->oask",orb,term.hamiltonian,block,optimize=True)
        _add(so,so.longitudinal.index(term.outgoing),term.mapped,val,result[so.sector])
    return {n:five_statistics_project(s,result[n]) for n,s in action.outgoing.items()}


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_all_retained_quartic_H_terms_equal_actual_same_regulated_pilot(target):
    p=pilot(target);got=np.zeros_like(p.matrix)
    for sector in ("qqq","qqqg"):
        source=source_tensor(p,sector,np.eye(len(p.matrix)))
        if sector=="qqqg":source=source.reshape(*source.shape[:3],128,source.shape[-1])
        lookup={m:j for j,m in enumerate(longitudinal_basis(sector,9))}
        for t in current.physical_pair_terms(sector,9,total_pplus_GeV=1.3,charge="electromagnetic",include_conversion=False):
            orbital=cluster_pair_contact(t.incoming,t.outgoing,t.pair,.4,0,regulator=p.field_regulator,species=current.SPECIES[sector])
            bra=np.take(source[lookup[t.outgoing]],t.mapped,axis=2)
            ket=np.take(source[lookup[t.incoming]],t.selected,axis=2)
            got+=np.einsum("oasf,oi,ab,ibsn->fn",bra.conj(),orbital,t.hamiltonian,ket,optimize=True)
    np.testing.assert_allclose(got,p.cross_pairs_per_gs2+p.instantaneous_fermion_per_gs2,atol=2e-15)


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_retained_bare_limit_and_both_current_endpoints(target,jz):
    from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot
    p=pilot(target,1e7)
    f=build_cluster_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
        quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3,regulator=p.field_regulator,
        total_pplus_GeV=1.3,wick_operators=p.wick_operators)
    new=current.cluster_quartic_pair_current_between(f,p,DELTA,**{**ARGS,"quadrature_order":2})
    old=retained_two_gluon_minus_between(f,p,DELTA,total_pplus_GeV=1.3,charge="electromagnetic")
    np.testing.assert_allclose(new.total[1],old,atol=5e-15)
    np.testing.assert_allclose(new.regulator,0.,atol=4e-15)
    np.testing.assert_array_equal(new.total[0],0.)


@pytest.mark.parametrize("cutoff",[0,1])
def test_physical_Q5_Hamiltonian_number_IF_and_pair_square_agree_on_actual_H2_waves(cutoff):
    m=model(q5=cutoff)
    s=m.spaces5;a=action(s)
    x=m.snapshot(1.4).five_wave_coefficients[0]
    components={"fermion_number":("qqqgg_fermion_scattering","qqqq-qbar_fermion_scattering"),
                "qg_number":("qg_number_scattering",),"IF_scattering":("IF_scattering",)}
    for key,labels in components.items():
        got=hamiltonian_terms(a,x,[key])
        expected={n:np.zeros_like(v) for n,v in x.items()}
        for label in labels:
            name,terms=m.qq.action.contacts.components[label]
            expected[name]+=apply_terms(terms,x[name],s[name].dimension)
        for name in s:
            np.testing.assert_allclose(got[name],expected[name],atol=2e-14)
    got=hamiltonian_terms(a,x,["IF_conversion"])
    gg,sea=s["qqqgg"],s["qqqq-qbar"]
    terms=m.qq.action.contacts.conversion
    forward=np.sqrt(2)*sea.statistics_project(apply_terms(terms,x["qqqgg"],sea.dimension))
    backward=np.sqrt(2)*apply_terms(terms,sea.statistics_project(x["qqqq-qbar"]),gg.dimension,adjoint=True)
    np.testing.assert_allclose(got["qqqq-qbar"],forward,atol=2e-14)
    np.testing.assert_allclose(got["qqqgg"],backward,atol=2e-14)
    # Independent all-four-quark annihilator and physical IG conversion
    # reproduce the auxiliary source-sector operator, not a fake gluon current.
    for origin,target,key in (("qqqq-qbar","qqqq-qbar","sea_annihilation"),
                              ("qqqgg","qqqq-qbar","IG_conversion"),
                              ("qqqq-qbar","qqqgg","IG_conversion")):
        probe={n:x[n] if n==origin else np.zeros_like(x[n]) for n in s}
        got=hamiltonian_terms(a,probe,[key])
        expected=m.qq.action.pair_current.apply(probe)
        np.testing.assert_allclose(got[target],expected[target],atol=2e-14)
        assert np.linalg.norm(expected[target])>1e-8


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_all_Q5_bare_canonical_currents_equal_original_quadratic_action(target,jz):
    # Old reference demands equal cutoffs, so compare the matching shell0
    # action here; rectangular finite transfers are tested independently.
    inc=out=spaces(0,target,jz);x=raw(inc,989);new=action(inc,out,width=1e7,order=2)
    old=FiveBodyQuadraticMinus(out,inc,DELTA,b_GeV=.4,total_pplus_GeV=1.3,charge="electromagnetic")
    got=new.apply(x,component=1,part="total");expected=old.apply(x)
    for n in out:
        np.testing.assert_allclose(got[n],expected[n],atol=3e-15)
    assert sum(np.linalg.norm(v) for v in got.values())>1e-6


def test_raw_physical_annihilation_is_not_mislabeled_as_full_auxiliary_isospin_operator():
    # On arbitrary statistics-projected raw isospin columns, the physical
    # extension need not equal the auxiliary I=1/2 restriction. The actual
    # H2-wave equivalence is tested separately above.
    m=model();s=m.spaces5;a=action(s)
    x=raw(s,987);x["qqqgg"]=np.zeros_like(x["qqqgg"])
    phys=hamiltonian_terms(a,x,["sea_annihilation"])["qqqq-qbar"]
    restricted=m.qq.action.pair_current.apply(x)["qqqq-qbar"]
    assert np.linalg.norm(phys-restricted)>1e-5


def ward_orbital(term,ci,co,reg,delta=DELTA,mean=MEAN):
    """Independent full shifted-H Gaussian endpoints, no current derivatives."""
    g=pair_current_geometry(term.incoming,term.outgoing,term.pair,
        species=current.SPECIES[term.incoming_sector],out_species=current.SPECIES[term.outgoing_sector],
        transfer_GeV=delta,mean_total_transverse_GeV=mean)
    B=reg.leg_transform(g["modes"]);result=0.
    for a,sigma in enumerate((-1,-1,1,1)):
        shifts=g["physical_field_shifts"].copy();shifts[a]+=sigma*np.asarray(delta)
        result+=sigma*term.charges[a]*gaussian_shell_integral(out_cutoff=co,in_cutoff=ci,
            out_map=g["out_map"],in_map=g["in_map"],in_shift=g["in_shift"],b_GeV=.4,
            field_maps=B@g["physical_field_maps"],field_shifts=B@shifts,field_precision=1/reg.width_GeV**2)
    return result*g["jacobian"]/(2*1.3)


def raw_ward_reference(a,x,component):
    source={n:five_statistics_project(s,x[n]) for n,s in a.incoming.items()}
    result={n:np.zeros((s.dimension,next(iter(x.values())).shape[1]),complex) for n,s in a.outgoing.items()}
    @lru_cache(maxsize=None)
    def cached(si,so,inc,out,pair,charges):
        # Color/spin have no part in the independent scalar Ward integral.
        term=next(t for t in a.terms if (t.incoming_sector,t.outgoing_sector,t.incoming,t.outgoing,t.pair,t.charges)
                  ==(si,so,inc,out,pair,charges))
        return ward_orbital(term,a.incoming[si].cutoff,a.outgoing[so].cutoff,a.regulator,a.delta,a.mean)
    for term in a.terms:
        if term.component!=component:continue
        si,so=a.incoming[term.incoming_sector],a.outgoing[term.outgoing_sector]
        orb=cached(si.sector,so.sector,term.incoming,term.outgoing,term.pair,term.charges)
        li,lo=si.longitudinal.index(term.incoming),so.longitudinal.index(term.outgoing)
        # One full physical spin/color slot at a time, independent of
        # the production packed multi-spin gather/accumulation.
        for spin,outspin in zip(term.selected,term.mapped):
            ids=np.flatnonzero((si.coordinates[:,0]==li)&(si.coordinates[:,3]==spin))
            block=np.zeros((len(si.orbitals),si.ncolor,source[si.sector].shape[1]),complex)
            block[si.coordinates[ids,1],si.coordinates[ids,2]]=source[si.sector][ids]
            value=np.einsum("oi,ab,ibk->oak",orb,term.hamiltonian,block,optimize=True)
            rows=np.flatnonzero((so.coordinates[:,0]==lo)&(so.coordinates[:,3]==outspin))
            result[so.sector][rows]+=value[so.coordinates[rows,1],so.coordinates[rows,2]]
    return {n:five_statistics_project(s,result[n]) for n,s in a.outgoing.items()}


@pytest.mark.parametrize("component",current.CHARGED_COMPONENTS)
def test_every_raw_Q5_charged_quartic_component_independent_slot_Ward_and_adjoint(component):
    inc,out=spaces(0,"NEUTRON",1),spaces(1,"NEUTRON",-1)
    a=action(inc,out);x,y=raw(inc,992),raw(out,993)
    got={n:np.zeros((s.dimension,1),complex) for n,s in out.items()}
    for j in range(2):
        v=a.apply(x,component=j+2,part="total",components=(component,))
        for n in out:got[n]+=DELTA[j]*v[n]
    expected=raw_ward_reference(a,x,component)
    for n in out:np.testing.assert_allclose(got[n],expected[n],atol=3e-16,rtol=4e-10)
    for part in ("canonical","regulator"):
        for mu in (1,2,3):
            f=a.apply(x,component=mu,part=part,components=(component,))
            b=a.adjoint(y,component=mu,part=part,components=(component,))
            np.testing.assert_allclose(sum(np.vdot(y[n],f[n]) for n in out),
                sum(np.vdot(b[n],x[n]) for n in inc),atol=3e-16)
    assert sum(np.linalg.norm(v) for v in got.values())>1e-9


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_retained_common_pair_current_Ward_reversal_charge_and_both_boost_laws(target):
    p=pilot(target)
    def value(mean=MEAN,pp=1.3,charge="electromagnetic",delta=DELTA):
        a=replace(p,total_pplus_GeV=pp)
        return current.cluster_quartic_pair_current_between(a,a,delta,
            **{**ARGS,"mean_total_transverse_GeV":mean,"charge":charge})
    a=value();back=value(delta=tuple(-x for x in DELTA))
    beta=np.array([.04,-.07])
    boost=value(mean=tuple(np.array(MEAN)+1.3*beta));long=value(pp=2.6)
    up,down=value(charge="up_number"),value(charge="down_number")
    for part in ("canonical","regulator"):
        aa=getattr(a,part)
        np.testing.assert_allclose(aa,getattr(back,part).conj().transpose(0,2,1),atol=4e-15)
        np.testing.assert_allclose(aa,(2*getattr(up,part)-getattr(down,part))/3,atol=4e-15)
        np.testing.assert_allclose(getattr(boost,part)[2:],aa[2:],atol=4e-15)
        np.testing.assert_allclose(getattr(boost,part)[1],aa[1]+np.einsum("j,joi->oi",beta,aa[2:]),atol=4e-15)
        np.testing.assert_allclose(getattr(long,part)[1],aa[1]/4,atol=4e-15)
        np.testing.assert_allclose(getattr(long,part)[2:],aa[2:]/2,atol=4e-15)
    expected=np.zeros_like(p.matrix)
    for sector in ("qqq","qqqg"):
        source=source_tensor(p,sector,np.eye(len(p.matrix)))
        if sector=="qqqg":source=source.reshape(*source.shape[:3],128,source.shape[-1])
        lookup={m:j for j,m in enumerate(longitudinal_basis(sector,9))}
        for t in current.physical_pair_terms(sector,9,total_pplus_GeV=1.3,charge="electromagnetic",include_conversion=False):
            orb=ward_orbital(t,0,0,p.field_regulator)
            bra=np.take(source[lookup[t.outgoing]],t.mapped,axis=2)
            ket=np.take(source[lookup[t.incoming]],t.selected,axis=2)
            expected+=np.einsum("oasf,oi,ab,ibsn->fn",bra.conj(),orb,t.hamiltonian,ket,optimize=True)
    np.testing.assert_allclose(np.einsum("j,joi->oi",DELTA,a.total[2:]),expected,atol=4e-15)
    assert np.linalg.norm(a.regulator)>1e-5


def test_full_Q5_charge_linearity_boosts_and_exact_pure_gluon_zeros():
    inc,out=spaces(0),spaces(1,jz=-1);x=raw(inc,994)
    a,u,d=[action(inc,out,charge=q) for q in ("electromagnetic","up_number","down_number")]
    beta=np.array([.04,-.07])
    boost=action(inc,out,mean=tuple(np.array(MEAN)+1.3*beta));long=action(inc,out,pp=2.6)
    for part in ("canonical","regulator"):
        base={mu:a.apply(x,component=mu,part=part) for mu in (1,2,3)}
        for mu in (1,2,3):
            up,down=u.apply(x,component=mu,part=part),d.apply(x,component=mu,part=part)
            sh,lg=boost.apply(x,component=mu,part=part),long.apply(x,component=mu,part=part)
            for n in out:
                np.testing.assert_allclose(base[mu][n],(2*up[n]-down[n])/3,atol=2e-15)
                expected=base[mu][n]+sum(beta[j]*base[j+2][n] for j in range(2)) if mu==1 else base[mu][n]
                np.testing.assert_allclose(sh[n],expected,atol=3e-15)
                np.testing.assert_allclose(lg[n],base[mu][n]/(4 if mu==1 else 2),atol=2e-15)
                np.testing.assert_allclose(five_statistics_project(out[n],base[mu][n]),base[mu][n],atol=2e-15)
    for c in current.NEUTRAL_COMPONENTS:
        for v in a.apply(x,component=1,part="total",components=(c,)).values():
            np.testing.assert_array_equal(v,0.)
    for v in a.apply(x,component=0,part="total").values():np.testing.assert_array_equal(v,0.)


def test_pair_action_validation_and_injective_annihilator_inventory():
    a=action();x=raw(spaces())
    assert all(len(np.unique(t.mapped))==len(t.mapped) for t in a.terms)
    for kwargs in (dict(part="auto",component=1),dict(part="total",component=True),
                   dict(part="total",component=1,components=("anything",))):
        with pytest.raises(ValueError):a.apply(x,**kwargs)
    with pytest.raises(ValueError,match="sectors"):
        action({"qqqgg":spaces()["qqqgg"]})
    with pytest.raises(ValueError,match="target"):
        action(spaces(),spaces(target="NEUTRON"))
    with pytest.raises(ValueError,match="pilots"):
        current.cluster_quartic_pair_current_between(object(),pilot(),DELTA,**ARGS)
