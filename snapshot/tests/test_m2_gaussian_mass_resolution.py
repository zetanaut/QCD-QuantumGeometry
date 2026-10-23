"""Independent finite-condition attribution, never a production calibration."""
import json
from pathlib import Path
import numpy as np
import pytest
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_gaussian_cluster_finite_response import UnitFiniteKineticMass
from test_m2_gaussian_cluster_h2 import model
from test_m2_gaussian_cluster_Q_projector_h2 import H_coefficients
from test_m2_gaussian_plus_resolution import word_data
from test_m2_gaussian_reference_shape_four_current import saved

ROOT=Path(__file__).resolve().parents[1]/'docs/next_level'


def mass_data(case):
    info=json.loads((ROOT/('m2_gaussian_mass_resolution_'+case+'.json')).read_text())
    with np.load(ROOT/info['full_matrix_file']) as data:return info,{k:data[k] for k in data}


@pytest.fixture(scope='module',params=['baseline','Q5_shell1'])
def member(request):
    case=request.param;info,data=mass_data(case);base,full=saved(case)
    assert info['parameters']==base['parameters']
    return case,info,data,base,full


def test_intercept_H_N_and_each_current_Ward_order_against_independent_zero_parameter_model(member):
    _,info,data,base,_=member
    m=model(q5=info['parameters']['q5'],q6=info['parameters']['q6'],q=0.,g=0.)
    f,i=m.snapshot(info['energy_out_GeV2']),m.snapshot(info['energy_in_GeV2'])
    for label,s in (('out',f),('in',i)):
        for key,attribute in (('H','effective_hamiltonian'),('N','metric')):
            np.testing.assert_allclose(data['intercept_'+key+'_'+label],getattr(s,attribute),atol=2e-14,rtol=2e-12)
    hf,hi=H_coefficients(m,f.energy_GeV2),H_coefficients(m,i.energy_GeV2)
    rho=good_current_density_between(m.pilot,m.pilot,info['transfer_GeV'],charge='electromagnetic')
    for n in range(5):
        v=data['intercept_current_g'+str(n)]
        expected=hf[n]@rho-rho@hi[n]
        if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
        residue=2*base['total_pplus_GeV']*np.einsum('j,jab->ab',info['transfer_GeV'],v[2:])-expected
        np.testing.assert_allclose(residue,0.,atol=3e-14)
        np.testing.assert_allclose(f.whitener@residue@i.whitener,0.,atol=3e-14)
    raw=sum(base['coupling']**n*data['intercept_current_g'+str(n)] for n in range(5))
    np.testing.assert_array_equal(raw,data['intercept_raw_current'])
    np.testing.assert_allclose(data['intercept_normalized_current'],
        np.einsum('ab,mbc,cd->mad',f.whitener,raw,i.whitener),atol=2e-14,rtol=2e-12)


def test_affine_mass_family_reconstructs_original_current_and_both_endpoint_metrics(member):
    _,info,data,_,full=member
    for n in range(5):
        intercept=data['intercept_current_g'+str(n)]
        if n in (2,4):
            finite=sum(info['parameters'][key]*data[parameter+'_current_g'+str(n)] for parameter,key in (('quark','muq'),('gluon','mug')))
            np.testing.assert_array_equal(finite,data['finite_current_g'+str(n)])
            np.testing.assert_allclose(intercept+finite,full['current_power_'+str(n)],atol=2e-16,rtol=2e-12)
        else:np.testing.assert_array_equal(intercept,full['current_power_'+str(n)])
    for key in ('H_out','H_in','N_out','N_in'):
        reconstructed=data['intercept_'+key]+sum(info['parameters'][mu]*data[parameter+'_'+key] for parameter,mu in (('quark','muq'),('gluon','mug')))
        np.testing.assert_allclose(reconstructed,full[key],atol=2e-14,rtol=2e-12)
    assert info['original_benchmark_unchanged'] and not info['physical']


def test_finite_current_response_agrees_with_independent_W55_decomposition_and_signed_mass_weights(member):
    case,info,data,_,_=member;_,words=word_data(case)
    finite=sum(v for k,v in words.items() if k.endswith('_finite_Wick'))
    np.testing.assert_allclose(data['finite_current_g4'][0],finite,atol=2e-17,rtol=2e-12)
    m=model(q5=info['parameters']['q5'],q6=info['parameters']['q6'])
    assert m.wick_operators.normal_order_subtraction
    for n,space in m.spaces5.items():
        expected=sum(info['parameters'][key]*UnitFiniteKineticMass(parameter).shell_mass_diagonal(space)
                     for parameter,key in (('quark','muq'),('gluon','mug')))
        np.testing.assert_allclose(m.qq.action.wick_diagonal[n],expected,atol=2e-16,rtol=2e-14)


def test_large_relative_resolution_errors_remain_in_the_mass_independent_current():
    _,a=mass_data('baseline');_,b=mass_data('Q5_shell1')
    _,x=saved('baseline');_,y=saved('Q5_shell1')
    for n,expected in ((2,.5103967829774788),(3,.16129908041170188),(4,.7766235723075793)):
        key='intercept_current_g'+str(n)
        np.testing.assert_allclose(np.linalg.norm(b[key]-a[key])/np.linalg.norm(a[key]),expected,rtol=2e-12)
    change=y['current_power_4']-x['current_power_4']
    finite=b['finite_current_g4']-a['finite_current_g4']
    np.testing.assert_allclose(np.vdot(change,finite).real/np.vdot(change,change).real,.9740119352227086,atol=2e-14)
    assert np.linalg.norm(change-finite)>1e-4
    # A signed matrix projection, not a probability or a fitted condition.
