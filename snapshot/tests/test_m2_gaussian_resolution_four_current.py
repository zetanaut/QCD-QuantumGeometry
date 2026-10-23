"""Independent Q5/Q6 resolution controls of saved full H2 currents.

Compare raw coupling coefficients before the two resummed norm maps.
Conservation and small total changes do not certify resolution convergence.
"""
import numpy as np
import pytest
import json
from pathlib import Path
from scripts.benchmark_m2_gaussian_cluster_four_current import build_model
from scripts.benchmark_m2_gaussian_cluster_four_current import normalized_current_error_bound
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from test_m2_gaussian_reference_shape_four_current import saved
from test_m2_gaussian_cluster_Q_projector_h2 import H_coefficients


def test_Q6_resolution_enters_only_fourth_raw_order_not_lower_orders():
    base,a=saved('baseline');changed,b=saved('Q6_shell1')
    assert changed['parameters']=={**base['parameters'],'q6':1}
    for key in ('total_k2','retained_cutoff','jz_twice','b_GeV','quark_mass_GeV',
                'gluon_mass_GeV','coupling','total_pplus_GeV','energy_out_GeV2',
                'energy_in_GeV2','transfer_GeV','component_order',
                'physical_leg_prescription','smooth_prescription',
                'field_envelope_prescription','isospin_prescription'):
        assert changed[key]==base[key]
    assert len(a)==len(b)==11 and set(a)==set(b)
    for n in range(4):
        np.testing.assert_array_equal(a['current_power_'+str(n)],b['current_power_'+str(n)])
    difference=b['current_power_4']-a['current_power_4']
    assert all(np.linalg.norm(v)>1e-8 for v in difference)
    np.testing.assert_allclose(np.linalg.norm(difference)/np.linalg.norm(a['current_power_4']),
                               .018269393625301413,rtol=2e-12)
    assert not changed['continuum_convergence_established'] and not changed['physical']


@pytest.mark.parametrize('case',['Q5_shell1','Q5_shell2','Q6_shell1','Q5_Q6_shell1','longitudinal_K11'])
def test_separate_and_joint_shell_currents_match_independent_H2_coefficients_and_both_norms(case):
    info,a=saved(case);m,parameters=build_model(case)
    assert parameters==info['parameters']
    f,i=m.snapshot(info['energy_out_GeV2']),m.snapshot(info['energy_in_GeV2'])
    for label,s in (('out',f),('in',i)):
        np.testing.assert_allclose(a['H_'+label],s.effective_hamiltonian,atol=3e-14)
        np.testing.assert_allclose(a['N_'+label],s.metric,atol=3e-15)
    hf,hi=H_coefficients(m,f.energy_GeV2),H_coefficients(m,i.energy_GeV2)
    d=np.array(info['transfer_GeV']);p=info['total_pplus_GeV']
    rho=good_current_density_between(m.pilot,m.pilot,d,charge='electromagnetic')
    for n in range(5):
        v=a['current_power_'+str(n)]
        expected=hf[n]@rho-rho@hi[n]
        if n:expected+=(f.energy_GeV2-i.energy_GeV2)*v[0]
        residue=2*p*np.einsum('j,jab->ab',d,v[2:])-expected
        np.testing.assert_allclose(residue,0.,atol=5e-14)
        np.testing.assert_allclose(f.whitener@residue@i.whitener,0.,atol=5e-14)
    raw=sum(info['coupling']**n*a['current_power_'+str(n)] for n in range(5))
    np.testing.assert_array_equal(raw,a['raw_current'])
    np.testing.assert_allclose(a['normalized_current'],
        np.einsum('ab,mbc,cd->mad',f.whitener,raw,i.whitener),atol=5e-14)


def test_shell_two_full_current_agrees_with_independent_nested_plus_and_both_norm_matrices():
    info, a = saved('Q5_shell2')
    root = Path(__file__).resolve().parents[1]/'docs/next_level'
    record = info['resolution_comparison']
    other = json.loads((root/record['independent_plus_report']).read_text())
    assert other['Q5_cutoffs'][0] == info['parameters']['q5'] == 2
    assert other['Q6_cutoffs'][0] == info['parameters']['q6'] == 0
    assert other['shared_benchmark_parameters'] == {
        k:v for k,v in info['parameters'].items() if k not in ('q5','q6')}
    assert other['endpoint_energies_GeV2'] == [info['energy_out_GeV2'],info['energy_in_GeV2']]
    for key in ('total_k2','retained_cutoff','b_GeV','quark_mass_GeV','gluon_mass_GeV',
                'coupling','total_pplus_GeV','transfer_GeV'):
        assert other[key] == info[key]
    with np.load(root/other['full_matrix_file']) as b:
        pairs = [(str(n),a['current_power_'+str(n)][0],b['lower_plus_g'+str(n)]) for n in range(5)]
        pairs += [(k,a[k],b['lower_'+k]) for k in ('N_out','N_in')]
        pairs += [(k,a[k+'_current'][0],b['lower_'+k+'_plus']) for k in ('raw','normalized')]
        for key, x, y in pairs:
            np.testing.assert_allclose(x,y,atol=3e-17,rtol=3e-12)
            np.testing.assert_allclose(np.linalg.norm(x-y),
                record['independent_plus_matrix_differences'][key],atol=1e-30,rtol=3e-12)


@pytest.mark.parametrize('reference',['baseline','Q5_shell1'])
def test_shell_two_each_matrix_component_and_two_norm_bound_against_aligned_reference(reference):
    info,b = saved('Q5_shell2');lower,a = saved(reference)
    assert info['parameters'] == {**lower['parameters'],'q5':2}
    for key in ('total_k2','retained_cutoff','jz_twice','b_GeV','quark_mass_GeV',
                'gluon_mass_GeV','coupling','total_pplus_GeV','energy_out_GeV2',
                'energy_in_GeV2','transfer_GeV','component_order','physical_leg_prescription',
                'smooth_prescription','field_envelope_prescription','isospin_prescription'):
        assert info[key] == lower[key]
    record = info['resolution_comparison'][reference]
    assert set(a) == set(b) == set(record['matrices']) and len(a) == 11
    for key,x in a.items():
        diff = np.linalg.norm(b[key]-x)
        np.testing.assert_allclose(diff,record['matrices'][key]['difference_Frobenius'],atol=1e-30,rtol=3e-12)
        np.testing.assert_allclose(diff/np.linalg.norm(x),record['matrices'][key]['relative_difference'],atol=1e-30,rtol=3e-12)
    for n in range(5):
        key='current_power_'+str(n)
        np.testing.assert_allclose([np.linalg.norm(x) for x in b[key]],info['coefficient_component_Frobenius'][str(n)],atol=3e-17,rtol=3e-12)
        for mu,(x,y) in enumerate(zip(a[key],b[key])):
            value=record['relative_component_changes'][str(n)][mu]
            if np.linalg.norm(x):
                np.testing.assert_allclose(np.linalg.norm(y-x)/np.linalg.norm(x),value,atol=1e-30,rtol=3e-12)
            else:
                assert value is None
                np.testing.assert_array_equal(x,y)
    bound=normalized_current_error_bound(a,b)
    for key,value in bound.items():
        np.testing.assert_allclose(value,record['two_norm_bound'][key],rtol=3e-12)
    actual=np.linalg.norm(b['normalized_current']-a['normalized_current'])
    assert actual < bound['normalized_current_error_bound_Frobenius']


def test_shell_two_resolution_effect_is_not_hidden_by_lower_orders_or_total_norm():
    info,b=saved('Q5_shell2');_,a=saved('Q5_shell1')
    for n in (0,1):
        np.testing.assert_array_equal(a['current_power_'+str(n)],b['current_power_'+str(n)])
    for n,expected in ((2,.0139613676104754),(3,.5895431341999058),(4,.20693010178581325)):
        key='current_power_'+str(n)
        np.testing.assert_allclose(np.linalg.norm(b[key]-a[key])/np.linalg.norm(a[key]),expected,rtol=3e-12)
        assert all(np.linalg.norm(y-x)>1e-8 for x,y in zip(a[key],b[key]))
    np.testing.assert_allclose(np.linalg.norm(b['normalized_current']-a['normalized_current'])/
        np.linalg.norm(a['normalized_current']),.00010052027177639382,rtol=3e-12)
    assert not info['physical'] and not info['continuum_convergence_established']


def test_two_resolution_axes_have_nonadditive_fourth_order_current_change():
    infos,arrays=zip(*(saved(k) for k in ('baseline','Q5_shell1','Q6_shell1','Q5_Q6_shell1')))
    for info,q5,q6 in zip(infos,(0,1,0,1),(0,0,1,1)):
        assert info['parameters']=={**infos[0]['parameters'],'q5':q5,'q6':q6}
    a,b,c,d=arrays
    for n in range(4):
        k='current_power_'+str(n)
        np.testing.assert_array_equal(c[k],a[k])
        np.testing.assert_array_equal(d[k],b[k])
    k='current_power_4'
    mixed=d[k]-b[k]-c[k]+a[k]
    changes=[np.linalg.norm(x-a[k])/np.linalg.norm(a[k]) for x in (b[k],c[k],d[k])]
    np.testing.assert_allclose(changes,[1.0248026130114076,.018269393625301413,.9053418556828291],rtol=3e-12)
    np.testing.assert_allclose(np.linalg.norm(mixed)/np.linalg.norm(a[k]),.12079975669530348,rtol=3e-12)
    assert all(np.linalg.norm(v)>1e-6 for v in mixed)
    # The joint effect is not the sum of separate matrix effects. It is
    # already present in the raw g^4 coefficient, before either norm map.
    assert np.linalg.norm(mixed)>.1*np.linalg.norm(d[k]-a[k])
    assert all(not info['continuum_convergence_established'] for info in infos)


@pytest.mark.parametrize('sector,numerator',[
    ('qqqgg',1029),('qqqq-qbar',1617),('qqqggg',1323),('qqqq-qbar-g',1911)])
def test_named_K9_resolution_study_has_a_shell_uniform_free_resolvent_gap(sector,numerator):
    from fractions import Fraction
    from deuteron_wigner.quantum import m2_excluded_fock_kinematics as five
    from deuteron_wigner.quantum import m2_six_body_kinematics as six
    from deuteron_wigner.quantum.m2_coupled_pilot import free_orbital
    module=five if sector in five.SPECIES else six
    species=module.SPECIES[sector]
    # Enumerate every positive APBC/PBC longitudinal composition exactly.
    # mq=7/20 GeV, mg=0. Positive compressed transverse kinetic energy
    # cannot lower this floor at ANY oscillator cutoff.
    floors=[9*sum((Fraction(49,400*p) for s,p in zip(species,modes) if s!='g'),Fraction())
            for modes in module.longitudinal_states(sector,9)]
    floor=min(floors)
    assert floor==Fraction(numerator,400)
    assert floor-Fraction(147,100)>=Fraction(441,400)
    # The executable non-diagonal radial propagation first changes at N=2.
    lows=[np.linalg.eigvalsh(free_orbital(len(species)-1,n,.4).toarray())[0] for n in range(4)]
    assert all(x>0 for x in lows)
    assert lows[2]<lows[1]
    assert all(float(floor)+x-1.47>1.1025 for x in lows)
