"""Full shaped H2 current: each power/component, two norms and kinematics."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_gaussian_cluster_four_current import assemble
from test_m2_gaussian_cluster_Q_projector_h2 import H_coefficients
from test_m2_gaussian_cluster_Q_projector import D,PP
from test_m2_gaussian_cluster_smooth_minus_actions import BOOST,BETA

ROOT=Path(__file__).resolve().parents[1]/"docs/next_level"
PREFIX="m2_gaussian_cluster_four_current_"


def saved(case):
    info=json.loads((ROOT/(PREFIX+case+".json")).read_text())
    with np.load(ROOT/info["full_matrix_file"]) as archive:
        arrays={k:archive[k] for k in archive}
    return info,arrays


@pytest.fixture(scope="module")
def member():
    m,values=build_model("reference12_degree1")
    info,arrays=saved("reference12_degree1")
    assert info["parameters"]==values
    assert not info["physical"] and not info["continuum_convergence_established"]
    return m.snapshot(info["energy_out_GeV2"]),m.snapshot(info["energy_in_GeV2"]),arrays


def test_saved_degree_one_current_closes_each_H2_Ward_coefficient_and_both_norms(member):
    verify_saved_current(member)


def test_saved_degree_two_current_closes_each_H2_Ward_coefficient_and_both_norms():
    m, values = build_model('reference12_degree2')
    info, arrays = saved('reference12_degree2')
    assert info['parameters'] == values
    assert not info['physical'] and not info['continuum_convergence_established']
    verify_saved_current((m.snapshot(info['energy_out_GeV2']), m.snapshot(info['energy_in_GeV2']), arrays))


def verify_saved_current(member):
    f,i,arrays=member
    for label,s in (("out",f),("in",i)):
        np.testing.assert_allclose(arrays["H_"+label],s.effective_hamiltonian,atol=2e-14)
        np.testing.assert_allclose(arrays["N_"+label],s.metric,atol=2e-15)
    hf,hi=H_coefficients(f.owner,f.energy_GeV2),H_coefficients(i.owner,i.energy_GeV2)
    rho=good_current_density_between(f.owner.pilot,i.owner.pilot,D,charge="electromagnetic")
    for n in range(5):
        v=arrays["current_power_"+str(n)]
        expected=hf[n]@rho-rho@hi[n]
        if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
        residue=2*PP*np.einsum("j,jab->ab",D,v[2:])-expected
        np.testing.assert_allclose(residue,0.,atol=5e-14)
        np.testing.assert_allclose(f.whitener@residue@i.whitener,0.,atol=5e-14)
    summed=sum(i.owner.pilot.coupling**n*arrays["current_power_"+str(n)] for n in range(5))
    np.testing.assert_array_equal(summed,arrays["raw_current"])
    np.testing.assert_allclose(arrays["normalized_current"],
        np.einsum("ab,mbc,cd->mad",f.whitener,summed,i.whitener),atol=5e-14)


def test_degree_one_improves_every_nonzero_current_component_against_named_reference():
    ref,r=saved("Gaussian_reference12");base,a=saved("baseline");shape,b=saved("reference12_degree1")
    for key in ("total_k2","retained_cutoff","jz_twice","b_GeV","quark_mass_GeV","gluon_mass_GeV",
                "coupling","total_pplus_GeV","energy_out_GeV2","energy_in_GeV2","transfer_GeV"):
        assert ref[key]==base[key]==shape[key]
    errors={}
    for n in range(5):
        key="current_power_"+str(n);errors[n]=[]
        for mu,target in enumerate(r[key]):
            norm=np.linalg.norm(target)
            if n==0 or norm==0:
                np.testing.assert_array_equal(a[key][mu],target)
                np.testing.assert_array_equal(b[key][mu],target)
            else:
                old=np.linalg.norm(a[key][mu]-target)/norm
                new=np.linalg.norm(b[key][mu]-target)/norm
                # A measured improvement of this named member, not a
                # universal truncation bound or an input calibration.
                assert 0<new<.2*old
                errors[n].append(float(new))
    print("degree-one per-component relative errors",errors,flush=True)


def test_degree_two_improves_every_nonzero_component_against_unchanged_finite_reference():
    ref, r = saved('Gaussian_reference12')
    first, a = saved('reference12_degree1')
    second, b = saved('reference12_degree2')
    assert second['parameters'] == {**first['parameters'], 'shape_degree':2}
    for key in ('total_k2', 'retained_cutoff', 'jz_twice', 'b_GeV', 'quark_mass_GeV',
                'gluon_mass_GeV', 'coupling', 'total_pplus_GeV', 'energy_out_GeV2',
                'energy_in_GeV2', 'transfer_GeV', 'physical_leg_prescription',
                'smooth_prescription', 'field_envelope_prescription', 'isospin_prescription'):
        assert ref[key] == first[key] == second[key]
    maximum = 0.
    recorded = second['finite_reference_comparison']
    assert recorded['reference_report'] == 'm2_gaussian_cluster_four_current_Gaussian_reference12.json'
    for n in range(5):
        key = 'current_power_'+str(n)
        stacked = np.linalg.norm(b[key]-r[key])/np.linalg.norm(r[key])
        np.testing.assert_allclose(recorded['relative_stacked_current_errors_g0_through_g4'][n], stacked, atol=2e-15, rtol=3e-12)
        for mu, target in enumerate(r[key]):
            norm = np.linalg.norm(target)
            if n == 0 or norm == 0:
                np.testing.assert_array_equal(b[key][mu], target)
                continue
            old = np.linalg.norm(a[key][mu]-target)/norm
            new = np.linalg.norm(b[key][mu]-target)/norm
            np.testing.assert_allclose(recorded['relative_component_errors_g0_through_g4'][n][mu], new, atol=2e-15, rtol=3e-12)
            assert 0 < new < .15*old
            maximum = max(maximum, new)
    np.testing.assert_allclose(maximum, .0024345542220830015, rtol=3e-12)
    normalized = np.linalg.norm(b['normalized_current']-r['normalized_current'])/np.linalg.norm(r['normalized_current'])
    np.testing.assert_allclose(normalized, 6.942042171040344e-6, rtol=3e-12)
    # Fixed finite reference/cutoffs only, not physical calibration or
    # regulator removal.


def test_degree_two_original_scalar_and_batched_full_arrays_are_bitwise_identical():
    scalar, a = saved('reference12_degree2_scalar')
    batched, b = saved('reference12_degree2')
    expected = {'raw_current', 'normalized_current', 'H_out', 'H_in', 'N_out', 'N_in'}
    expected.update('current_power_'+str(n) for n in range(5))
    assert set(a) == set(b) == expected
    for key in expected:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        assert np.all(np.isfinite(a[key])) and np.all(np.isfinite(b[key]))
        # Check actual payload bytes as well as values, including signed
        # zeros; equal norms or array_equal alone do not prove this claim.
        assert a[key].tobytes(order='C') == b[key].tobytes(order='C'), key
    excluded = {'seconds', 'full_matrix_file', 'moment_evaluation_variant',
                'numerical_equivalence', 'finite_reference_comparison'}
    assert {k:v for k,v in scalar.items() if k not in excluded} == {
        k:v for k,v in batched.items() if k not in excluded}
    record = scalar['numerical_equivalence']
    assert record['compared_with'] == PREFIX+'reference12_degree2.json'
    assert record['all_eleven_array_payloads_bitwise_identical']
    assert record['all_common_physical_metadata_identical']
    assert not record['timing_is_controlled_performance_benchmark']
    assert record['scalar_seconds'] == scalar['seconds']
    assert record['batched_seconds'] == batched['seconds']
    assert record['observed_scalar_to_batched_walltime_ratio'] == scalar['seconds']/batched['seconds']
    assert not scalar['physical'] and not scalar['continuum_convergence_established']


def test_full_degree_one_current_reverses_and_boosts_with_both_energy_norm_endpoints(member):
    f,i,arrays=member
    reverse=assemble(i,f,-D)
    boosted=assemble(f,i,context=BOOST)
    for n in range(5):
        v=arrays["current_power_"+str(n)]
        np.testing.assert_allclose(v,reverse.coefficients[n].conj().transpose(0,2,1),atol=5e-14)
        expected=v.copy()
        expected[1]+=np.einsum("j,jab->ab",BETA,v[2:])+.5*np.dot(BETA,BETA)*v[0]
        expected[2:]+=BETA[:,None,None]*v[0]
        np.testing.assert_allclose(boosted.coefficients[n],expected,atol=5e-14)
    raw=arrays["raw_current"].copy()
    raw[1]+=np.einsum("j,jab->ab",BETA,raw[2:])+.5*np.dot(BETA,BETA)*raw[0]
    raw[2:]+=BETA[:,None,None]*raw[0]
    np.testing.assert_allclose(boosted.current(normalized=True),
        np.einsum("ab,mbc,cd->mad",f.whitener,raw,i.whitener),atol=5e-14)
    np.testing.assert_allclose(arrays["normalized_current"],
        reverse.current(normalized=True).conj().transpose(0,2,1),atol=5e-14)
