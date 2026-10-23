"""Combined physical current: independent wave powers and canonical limits."""
import numpy as np
import pytest
from deuteron_wigner.quantum import m2_gaussian_cluster_h2_current as current
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_fourth_order_h2 import current_between as plus_between
from deuteron_wigner.quantum.m2_fourth_order_minus_current import minus_coefficients_between
from deuteron_wigner.quantum.m2_fourth_order_transverse_current import transverse_coefficients_between
from deuteron_wigner.quantum.m2_gaussian_cluster_wick_current import cluster_wick_minus_h2_coefficients
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_cubic_h2_current import independent_waves
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION

D=(.12,.16)
ARGS=dict(charge="electromagnetic",mean_total_transverse_GeV=(.03,.07),
          prescription=PRESCRIPTION,quadrature_order=8)
LOOP=(2/3,-1/3)


def polynomial_free(f,i):
    af,a6f=independent_waves(f.owner,f.energy_GeV2)
    ai,a6i=independent_waves(i.owner,i.energy_GeV2)
    p=i.owner.pilot;nf,ni=len(f.owner.pilot.matrix),len(p.matrix)
    args=dict(charge="electromagnetic",total_pplus_GeV=1.3,mean_total_transverse=ARGS["mean_total_transverse_GeV"])
    pp=np.zeros((4,nf,ni),complex)
    pp[0]=good_current_density_between(f.owner.pilot,p,D,charge="electromagnetic")
    pp[1]=current.free_minus_current_between(f.owner.pilot,p,D,**args)
    pp[2:]=current.free_transverse_current_between(f.owner.pilot,p,D,**args)
    def apply(fs,ins,v,mu):
        if mu==0:return current.shell_density_between(fs,ins,v,D,b_GeV=.4,charge="electromagnetic")
        if mu==1:return current.shell_minus_current_between(fs,ins,v,D,b_GeV=.4,mass_GeV=.35,**args)
        return current.shell_transverse_current_between(fs,ins,v,D,b_GeV=.4,component=mu-2,**args)
    acted5={mu:{n:apply(f.owner.spaces5[n],s,np.column_stack([a[n] for a in ai]),mu)
                   for n,s in i.owner.spaces5.items()} for mu in range(4)}
    acted6={mu:{n:apply(f.owner.spaces6[n],s,a6i[n],mu)
                   for n,s in i.owner.spaces6.items()} for mu in range(4)}
    points=np.exp(2j*np.pi*np.arange(7)/7);samples=[]
    for z in points:
        value=pp.copy()
        for mu in range(4):
            for n in ai[0]:
                bra=sum(z**(k+1)*a[n].conj().T for k,a in enumerate(af))
                ket=sum(z**(k+1)*acted5[mu][n][:,k*ni:(k+1)*ni] for k in range(3))
                value[mu]+=bra@ket
            value[mu]+=z**4*sum(a6f[n].conj().T@v for n,v in acted6[mu].items())
        samples.append(value)
    return {n:sum(a/z**n for z,a in zip(points,samples))/7 for n in range(7)}


@pytest.mark.parametrize("target,jz",[("PROTON",1),("NEUTRON",-1)])
def test_free_all_components_on_common_waves_independent_polynomial_and_boosts(target,jz):
    f=model(target=target,jz=jz).snapshot(1.39)
    i=model(target=target).snapshot(1.47)
    got=current.cluster_free_h2_coefficients(f,i,D,**ARGS)
    ref=polynomial_free(f,i)
    for n in range(5):np.testing.assert_allclose(got[n],ref[n],atol=5e-15)
    np.testing.assert_array_equal(got[1],0.)
    assert np.linalg.norm(ref[6][1])>1e-10
    plus=sum(.3**n*v[0] for n,v in got.items())
    np.testing.assert_allclose(plus,plus_between(f,i,D,charge="electromagnetic",normalized=False),atol=3e-15)
    # Exact common boost of every term; H2 intrinsic waves do not change.
    beta=np.array([.04,-.07])
    changed=current.cluster_free_h2_coefficients(f,i,D,**{**ARGS,
        "mean_total_transverse_GeV":tuple(np.array(ARGS["mean_total_transverse_GeV"])+1.3*beta)})
    for n,a in got.items():
        np.testing.assert_allclose(changed[n][0],a[0],atol=3e-15)
        np.testing.assert_allclose(changed[n][2:],a[2:]+beta[:,None,None]*a[0],atol=3e-15)
        np.testing.assert_allclose(changed[n][1],a[1]+np.einsum("j,jab->ab",beta,a[2:])+.5*(beta@beta)*a[0],atol=5e-15)


def test_assembled_bare_limit_equals_independent_existing_full_current_at_each_order():
    p=model(width=1e7,q=0.,g=0.)
    f,i=p.snapshot(1.39),p.snapshot(1.47)
    args={**ARGS,"mean_total_transverse_GeV":(0.,0.),"quadrature_order":2}
    a=current.assemble_cluster_h2_current(f,i,D,loop_flavor_charges=LOOP,**args)
    minus=minus_coefficients_between(f,i,D,total_pplus_GeV=1.3,charge="electromagnetic")
    trans=transverse_coefficients_between(f,i,D,total_pplus_GeV=1.3,charge="electromagnetic")
    for n,value in a.coefficients.items():
        np.testing.assert_allclose(value[1],minus[n],atol=2e-14)
        np.testing.assert_allclose(value[2:],trans[n],atol=2e-14)
    np.testing.assert_allclose(a.current(normalized=False)[0],
        plus_between(f,i,D,charge="electromagnetic",normalized=False),atol=3e-15)
    for name,blocks in a.contributions.items():
        if name.endswith("_regulator") and not name.startswith("Wick"):
            for value in blocks.values():np.testing.assert_allclose(value,0.,atol=2e-14)
    # Exact paired subtraction happens BEFORE projection, even though the
    # separately inspectable Wick pieces grow quadratically with Lambda.
    assert np.linalg.norm(a.wick_components["canonical"][2])>1e10
    for value in a.contributions["Wick_paired"].values():
        np.testing.assert_array_equal(value,0.)


def test_assembled_finite_current_preserves_all_families_Wick_subtraction_and_both_norms():
    f,i=model(target="NEUTRON").snapshot(1.39),model(target="NEUTRON").snapshot(1.47)
    a=current.assemble_cluster_h2_current(f,i,D,loop_flavor_charges=LOOP,**ARGS)
    assert set(a.contributions)=={"free","cubic_canonical","cubic_regulator",
        "quartic_source_canonical","quartic_source_regulator","quartic_pair_canonical",
        "quartic_pair_regulator","Wick_paired"}
    assert set(a.wick_components)=={"canonical","regulator","subtraction","finite"}
    ref=cluster_wick_minus_h2_coefficients(f,i,D,charge="electromagnetic",
        loop_flavor_charges=LOOP,prescription=PRESCRIPTION,component="total")
    for n in (2,4):
        combined=a.contributions["Wick_paired"][n]
        np.testing.assert_allclose(combined[1],ref[n],atol=4e-16)
        np.testing.assert_allclose(sum(a.wick_components[part][n] for part in ("canonical","regulator","subtraction")),0.,atol=3e-16)
    raw=a.current(normalized=False)
    np.testing.assert_allclose(a.current(normalized=True),
        np.einsum("ab,mbc,cd->mad",f.whitener,raw,i.whitener),atol=5e-16)
    np.testing.assert_allclose(raw[0],plus_between(f,i,D,charge="electromagnetic",normalized=False),atol=4e-15)
    assert a.free_contact_Wick_numerator_complete
    assert not a.complete_H2_matched_current and not a.oscillator_projector_transport_included
    assert not a.physical and not a.longitudinal_envelope_gauged
    assert np.linalg.norm(a.contributions["quartic_source_regulator"][4])>1e-10
    assert np.linalg.norm(a.contributions["quartic_pair_regulator"][4])>1e-9
    with pytest.raises(ValueError,match="Boolean"):a.current(normalized="auto")
    with pytest.raises(ValueError,match="four-current"):i.current(D,charge="electromagnetic")


def test_combined_definition_guards_precede_expensive_assembly():
    a=model().snapshot(1.4);b=model(width=.4).snapshot(1.4)
    with pytest.raises(ValueError,match="identical cluster"):
        current.assemble_cluster_h2_current(a,b,D,loop_flavor_charges=LOOP,**ARGS)
    with pytest.raises(ValueError,match="both actual"):
        current.assemble_cluster_h2_current(a.owner,b.owner,D,loop_flavor_charges=LOOP,**ARGS)
    for loop in ((2/3,),(-1/3,2/3),(2/3,float("nan"))):
        with pytest.raises(ValueError,match="two loop flavors"):
            current.assemble_cluster_h2_current(a,a,D,loop_flavor_charges=loop,**ARGS)
