"""Independent finite-parameter and noncommuting norm derivative controls."""
import numpy as np
import pytest
import json
import importlib.util
from pathlib import Path
from functools import lru_cache
from deuteron_wigner.quantum.m2_gaussian_cluster_finite_response import (
    inverse_sqrt_frechet,finite_mass_snapshot_response,finite_mass_current_coefficients,
    UnitFiniteKineticMass,normalized_finite_mass_current_derivative)
from deuteron_wigner.quantum.m2_gaussian_cluster_four_current import assemble_cluster_h2_four_current
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from deuteron_wigner.quantum.m2_total_isospin import H2_ISOSPIN_PRESCRIPTION
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_Q_projector import D,PP,ARGS
from test_m2_gaussian_cluster_smooth_minus_actions import ZERO,BOOST,BETA


@lru_cache(maxsize=1)
def benchmark_module():
    path=Path(__file__).resolve().parents[1]/"scripts"/"benchmark_m2_gaussian_cluster_four_current.py"
    spec=importlib.util.spec_from_file_location("four_current_benchmark",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_saved_matrix_scalar_diagnostic_matches_independent_real_least_squares():
    rng=np.random.default_rng(491)
    x=rng.normal(size=(4,3,3))+1j*rng.normal(size=(4,3,3))
    y=rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)
    compare=benchmark_module().compare_matrices
    result=compare({"current_power_1":y},{"current_power_1":x})["current_power_1"]
    real=lambda a:np.concatenate((a.real.ravel(),a.imag.ravel()))
    independent=np.linalg.lstsq(real(x)[:,None],real(y),rcond=None)[0][0]
    np.testing.assert_allclose(result["best_real_scalar"],independent,atol=2e-15)
    np.testing.assert_allclose(result["best_scalar_residual_Frobenius"],np.linalg.norm(real(y)-independent*real(x)))
    assert compare({"a":y},{"a":y})["a"]["bitwise_identical"]
    zero=compare({"current_power_0":np.zeros(2)},{"current_power_0":np.zeros(2)})["current_power_0"]
    assert zero["relative_difference"] is None and zero["best_real_scalar"] is None
    with pytest.raises(ValueError):compare({"a":y},{"b":y})
    with pytest.raises(ValueError):compare({"a":y},{"a":np.array([np.nan])})


def test_Gaussian_width_change_has_order_one_shape_not_absorbable_by_mass_terms_or_single_scalar():
    root=Path(__file__).resolve().parents[1]/"docs"/"next_level"
    prefix="m2_gaussian_cluster_four_current_"
    infos=[json.loads((root/(prefix+name+".json")).read_text()) for name in ("baseline","Gaussian_width04")]
    assert {k for k,v in infos[0]["parameters"].items() if v!=infos[1]["parameters"][k]}=={"width"}
    assert all(i["Ward_Frobenius"]<3e-15 and not i["physical"] for i in infos)
    # These two named benchmark constructors use the same fixed global
    # retained basis. Changing only the field width does not reorder it.
    with np.load(root/infos[0]["full_matrix_file"]) as a,np.load(root/infos[1]["full_matrix_file"]) as b:
        comparison=benchmark_module().compare_matrices(a,b)
        assert comparison["current_power_0"]["bitwise_identical"]
        assert comparison["current_power_1"]["relative_difference"]>.27
        assert comparison["current_power_1"]["best_scalar_relative_residual"]>.08
        assert comparison["current_power_3"]["difference_Frobenius"]>2e-4
        # The independently checked analytic mass directions have zero
        # order-one and order-three coefficients. This is a span obstruction
        # for those directions, not a lower bound on every possible EFT fit.


def inverse_sqrt(a):
    n,u=np.linalg.eigh(a)
    return (u/np.sqrt(n))@u.conj().T


@pytest.mark.parametrize("degenerate",[True,False])
def test_inverse_sqrt_derivative_against_independent_perturbed_eigensystems(degenerate):
    rng=np.random.default_rng(981)
    z=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7));u=np.linalg.qr(z)[0]
    n=(u*(np.ones(7)*1.2 if degenerate else np.arange(1,8)))@u.conj().T
    z=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7));d=(z+z.conj().T)/20
    step=1e-4
    ref=(-inverse_sqrt(n+2*step*d)+8*inverse_sqrt(n+step*d)-8*inverse_sqrt(n-step*d)+inverse_sqrt(n-2*step*d))/(12*step)
    actual=inverse_sqrt_frechet(n,d)
    np.testing.assert_allclose(actual,ref,atol=1e-11)
    w=inverse_sqrt(n)
    np.testing.assert_allclose(actual@n@w+w@d@w+w@n@actual,0.,atol=3e-15)
    if not degenerate:assert np.linalg.norm(n@d-d@n)>.1


@pytest.mark.parametrize("target,cutoff",[("PROTON",0),("NEUTRON",0),("PROTON",1),("NEUTRON",1)])
@pytest.mark.parametrize("parameter",["quark","gluon"])
def test_analytic_H_norm_backflow_and_energy_derivatives_against_independent_rebuilt_models(target,cutoff,parameter):
    args=dict(target=target,q5=cutoff,q6=cutoff);s=model(**args).snapshot(1.39)
    direction=finite_mass_snapshot_response(s,parameter=parameter)
    eps=1e-4;key="q" if parameter=="quark" else "g";base=.011 if key=="q" else -.007
    f=model(**args,**{key:base+eps}).snapshot(1.39)
    b=model(**args,**{key:base-eps}).snapshot(1.39)
    for name in ("effective_hamiltonian","metric","whitener","matrix"):
        np.testing.assert_allclose(getattr(direction,name),(getattr(f,name)-getattr(b,name))/(2*eps),atol=3e-10)
    for n,dc in direction.C3_direction.items():
        np.testing.assert_allclose(dc,(f.five_wave_coefficients[2][n]-b.five_wave_coefficients[2][n])/(2*eps),atol=2e-13)
        for k in (0,1):np.testing.assert_array_equal(f.five_wave_coefficients[k][n],b.five_wave_coefficients[k][n])
    e=lambda x:finite_mass_snapshot_response(model(**args).snapshot(x),parameter=parameter).effective_hamiltonian
    dh=(-e(1.39+2*eps)+8*e(1.39+eps)-8*e(1.39-eps)+e(1.39-2*eps))/(12*eps)
    np.testing.assert_allclose(dh,-direction.metric,atol=2e-10)
    assert np.linalg.norm(direction.effective_hamiltonian)>0
    assert np.linalg.norm(direction.metric)>0


def coefficients(f,i,parameter,delta=D,context=ZERO):
    return finite_mass_current_coefficients(f,i,delta,parameter=parameter,
        loop_flavor_charges=(2/3,-1/3),minus_context=context,
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,
        isospin_prescription=H2_ISOSPIN_PRESCRIPTION,**ARGS)


@pytest.mark.parametrize("parameter",["quark","gluon"])
def test_analytic_finite_current_direction_Ward_boost_reversal_and_exact_order_support(parameter):
    f,i=model().snapshot(1.39),model().snapshot(1.47)
    a,b,c=coefficients(f,i,parameter),coefficients(i,f,parameter,-D),coefficients(f,i,parameter,context=BOOST)
    rf,ri=[finite_mass_snapshot_response(s,parameter=parameter) for s in (f,i)]
    rho=good_current_density_between(f.owner.pilot,i.owner.pilot,D,charge="electromagnetic")
    for n,v in a.items():
        np.testing.assert_allclose(v,b[n].conj().transpose(0,2,1),atol=2e-14)
        boosted=v.copy();boosted[1]+=np.einsum("j,jab->ab",BETA,v[2:])+.5*np.dot(BETA,BETA)*v[0]
        boosted[2:]+=BETA[:,None,None]*v[0]
        np.testing.assert_allclose(c[n],boosted,atol=2e-14)
        h=(rf.H_coefficients[n]@rho-rho@ri.H_coefficients[n]) if n in (2,4) else np.zeros_like(rho)
        expected=h+(f.energy_GeV2-i.energy_GeV2)*v[0]
        np.testing.assert_allclose(2*PP*np.einsum("j,jab->ab",D,v[2:]),expected,atol=3e-14)
        if n not in (2,4):np.testing.assert_array_equal(v,0.)
    assert np.linalg.norm(a[4])>1e-8
    if parameter=="gluon":np.testing.assert_array_equal(a[2],0.)
    else:assert np.linalg.norm(a[2][1])>1.
    print(parameter,"finite-current derivative norms",[np.linalg.norm(a[n]) for n in (2,4)],flush=True)


def test_finite_response_rejects_undefined_parameter_and_nonpositive_norm():
    with pytest.raises(ValueError):UnitFiniteKineticMass("vertex_mass")
    with pytest.raises(ValueError):UnitFiniteKineticMass("quark").mass_components("ghost",1,total_k2=9)
    with pytest.raises(ValueError):UnitFiniteKineticMass("quark").mass_components("q",0,total_k2=9)
    with pytest.raises(ValueError):finite_mass_snapshot_response(None,parameter="quark")
    with pytest.raises(ValueError):inverse_sqrt_frechet(np.diag([0.,1.]),np.eye(2))
    with pytest.raises(ValueError):inverse_sqrt_frechet(np.eye(2),np.array([[0.,1.],[0.,0.]]))


@pytest.mark.parametrize("parameter",["quark","gluon"])
def test_analytic_response_matches_saved_independently_rebuilt_full_current_matrices(parameter):
    # The reproducible benchmark reruns the complete current/H2 construction,
    # independently of this derivative. Keep the full complex matrices, not
    # merely a norm or scalar probe. These are explicitly nonphysical members.
    root=Path(__file__).resolve().parents[1]/"docs"/"next_level"
    prefix="m2_gaussian_cluster_four_current_"
    base_info=json.loads((root/(prefix+"baseline.json")).read_text())
    shifted_info=json.loads((root/(prefix+"finite_"+parameter+"_shift.json")).read_text())
    assert not base_info["physical"] and not shifted_info["physical"]
    key="muq" if parameter=="quark" else "mug"
    assert {k for k,v in base_info["parameters"].items() if v!=shifted_info["parameters"][k]}=={key}
    step=shifted_info["parameters"][key]-base_info["parameters"][key]
    m=model();f,i=m.snapshot(1.39),m.snapshot(1.47)
    a=coefficients(f,i,parameter)
    with np.load(root/base_info["full_matrix_file"]) as base,np.load(root/shifted_info["full_matrix_file"]) as changed:
        for endpoint,s in (("out",f),("in",i)):
            r=finite_mass_snapshot_response(s,parameter=parameter)
            for name,attr in (("H_","effective_hamiltonian"),("N_","metric")):
                np.testing.assert_allclose(base[name+endpoint],getattr(s,attr),atol=3e-14)
                np.testing.assert_allclose((changed[name+endpoint]-base[name+endpoint])/step,getattr(r,attr),atol=3e-12)
        for n,v in a.items():
            ref=(changed["current_power_"+str(n)]-base["current_power_"+str(n)])/step
            np.testing.assert_allclose(v,ref,atol=3e-13)
            if n in (0,1,3):np.testing.assert_array_equal(changed["current_power_"+str(n)],base["current_power_"+str(n)])
        raw=base["raw_current"];derivative=sum(.3**n*v for n,v in a.items())
        np.testing.assert_allclose(derivative,(changed["raw_current"]-raw)/step,atol=3e-13)


@lru_cache(maxsize=1)
def full_assembly():
    m=model()
    return assemble_cluster_h2_four_current(m.snapshot(1.39),m.snapshot(1.47),D,
        loop_flavor_charges=(2/3,-1/3),minus_context=ZERO,
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,
        isospin_prescription=H2_ISOSPIN_PRESCRIPTION,**ARGS)


@pytest.mark.parametrize("parameter",["quark","gluon"])
def test_normalized_analytic_current_derivative_keeps_both_noncommuting_norm_responses(parameter):
    assembly=full_assembly()
    actual=normalized_finite_mass_current_derivative(assembly,parameter=parameter)
    root=Path(__file__).resolve().parents[1]/"docs"/"next_level"
    prefix="m2_gaussian_cluster_four_current_"
    with np.load(root/(prefix+"baseline.npz")) as a,np.load(root/(prefix+"finite_"+parameter+"_shift.npz")) as b:
        np.testing.assert_allclose(assembly.current(normalized=False),a["raw_current"],atol=3e-14)
        # Independently rebuilt finite-shift matrices fix the affine raw
        # response. Perturb and rediagonalize each metric independently.
        raw=a["raw_current"];dj=(b["raw_current"]-raw)/.01
        nf,ni=a["N_out"],a["N_in"]
        df,di=(b["N_out"]-nf)/.01,(b["N_in"]-ni)/.01
        def shifted(t):
            return np.einsum("ab,mbc,cd->mad",inverse_sqrt(nf+t*df),raw+t*dj,inverse_sqrt(ni+t*di))
        h=1e-3
        reference=(-shifted(2*h)+8*shifted(h)-8*shifted(-h)+shifted(-2*h))/(12*h)
        np.testing.assert_allclose(actual,reference,atol=3e-10)
        frozen=np.einsum("ab,mbc,cd->mad",inverse_sqrt(nf),dj,inverse_sqrt(ni))
        assert np.linalg.norm(actual-frozen)>1e-10
