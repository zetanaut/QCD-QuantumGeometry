import json
from pathlib import Path
import numpy as np
import pytest
from scripts.analyze_m2_cluster_spin import read_block,analyze
from deuteron_wigner.quantum.m2_cluster_spin_diagnostic import (
    elastic_orientation_analysis,rotate_current_blocks,complete_elastic_spin_amplitudes)

ROOT=Path(__file__).resolve().parents[1]/'docs/next_level'


@pytest.mark.parametrize('target',('p','n'))
def test_all_four_fresh_spin_blocks_and_independent_kinematic_completion(target):
    paths=[ROOT/f'm2_cluster_spin_{target}_{h}.json' for h in ('pp','pm','mp','mm')]
    report,a=analyze(paths)
    predicted=complete_elastic_spin_amplitudes(a['family_order_amplitudes'][...,0,0],
                                              a['family_order_amplitudes'][...,1,0],report['metadata']['transfer_GeV'])
    # Test each family and each raw order, not just the small weighted total.
    np.testing.assert_allclose(predicted,a['family_order_amplitudes'],atol=3e-13,rtol=0)
    reduced,b=analyze([paths[0],paths[2]],symmetry_complete=True)
    for key in ('current','five_coefficients','two_residual','five_residual'):
        np.testing.assert_allclose(a[key],b[key],atol=3e-13,rtol=0)
    assert report['independent_spin_blocks']==4 and reduced['independent_spin_blocks']==2
    assert reduced['kinematically_completed_blocks']==['pm','mm']
    assert report['two_structure_residual_norm']>1e-2
    assert np.linalg.norm(a['five_residual'])<1e-10
    assert report['continuity_norm']<1e-12
    assert not report['physical'] and not report['covariance_repaired']
    with pytest.raises(ValueError):analyze([paths[0],paths[2]])


@pytest.mark.parametrize('target',('p','n'))
@pytest.mark.parametrize('channel',('pp','pm','mp','mm'))
def test_saved_finite_pole_current_uses_the_actual_both_norm_states(target,channel):
    info,a=read_block(ROOT/f'm2_cluster_spin_{target}_{channel}.json')
    e=info['energy_GeV2']
    for label in ('out','in'):
        u,v=a[label+'_raw_state'],a[label+'_state'];n,h,w=a[label+'_N'],a[label+'_H'],a[label+'_W']
        assert min(np.linalg.eigvalsh(n))>0
        np.testing.assert_allclose(w@n@w,np.eye(len(n)),atol=4e-14,rtol=0)
        np.testing.assert_allclose(w@v,u,atol=4e-14,rtol=0)
        np.testing.assert_allclose(h@u,e*u,atol=3e-13,rtol=0)
        assert np.vdot(u,n@u)==pytest.approx(1,abs=4e-14)
    currents=np.einsum('ab,nmbc,cd->nmad',a['out_W'],a['current_coefficients'],a['in_W'])
    measured=np.einsum('a,nmab,b->nm',a['out_state'].conj(),currents,a['in_state'])
    np.testing.assert_allclose(measured,a['family_amplitudes'].sum(axis=0),atol=3e-14,rtol=0)


def test_fresh_axis_current_kinematic_rotation_every_family_and_order():
    axis,x=read_block(ROOT/'m2_cluster_spin_p_mp_axis.json')
    oblique,y=read_block(ROOT/'m2_cluster_spin_p_mp.json')
    angle=np.arctan2(.16,.12)
    for key in ('current_coefficients','family_coefficients'):
        predicted=rotate_current_blocks(x[key],angle,jz_out_twice=-1,jz_in_twice=1)
        np.testing.assert_allclose(predicted,y[key],atol=8e-13,rtol=0)
    assert axis['energy_GeV2']==oblique['energy_GeV2']


def test_fresh_transverse_boost_every_family_and_order():
    boosted,b=read_block(ROOT/'m2_cluster_spin_p_mp_boost.json')
    _,a=read_block(ROOT/'m2_cluster_spin_p_mp.json')
    beta=np.array(boosted['metadata']['mean_total_transverse_GeV'])/boosted['parameters']['Pplus_GeV']
    for key in ('current_coefficients','family_coefficients'):
        v=a[key];expected=v.copy()
        expected[...,1,:,:]+=np.einsum('j,...jab->...ab',beta,v[...,2:,:,:])+.5*(beta@beta)*v[...,0,:,:]
        expected[...,2:,:,:]+=beta[:,None,None]*v[...,0:1,:,:]
        np.testing.assert_allclose(expected,b[key],atol=8e-13,rtol=0)


@pytest.mark.parametrize('member',('K9_P0_p','K9_P0_n','K9_P1_p','K9_P2_p','K11_P0_p'))
def test_free_resolution_reports_replay_without_chosen_degenerate_state(member):
    info,a=read_block(ROOT/f'm2_cluster_free_spin_{member}.json')
    rank=info['ground_multiplicity'];assert a['up_subspace'].shape[1]==rank
    for key in ('up_subspace','down_subspace'):
        np.testing.assert_allclose(a[key].conj().T@a[key],np.eye(rank),atol=6e-14,rtol=0)
    assert a['subspace_current_parts'].shape==(2,4,2,2,rank,rank)
    if rank!=1:
        assert info['orientation'] is None and rank==3
        assert 'no arbitrary' in info['state_selection']
    else:
        c=a['spin_current_parts'].sum(axis=0)
        q=elastic_orientation_analysis(c,info['transfer_GeV'],energy_GeV2=info['energy_GeV2'],total_pplus_GeV=info['Pplus_GeV'])
        np.testing.assert_allclose(q['five']['coefficients'],[complex(*z) for z in info['orientation']['full']['coefficients']],atol=3e-13)
        assert q['two']['residual_norm']>1e-2
        if info['target']=='PROTON':
            np.testing.assert_allclose(q['five']['coefficients'][2],-c[0,0,0],atol=4e-14)
        assert np.linalg.norm(a['spin_current_parts'][1,0])<2e-13
        assert np.linalg.norm(a['spin_current_parts'][1,2:])<2e-13


@pytest.mark.parametrize('target',('p','n'))
def test_raw_g0_matches_independent_free_limit_in_every_spin_channel(target):
    _,free=read_block(ROOT/f'm2_cluster_free_spin_K9_P0_{target}.json')
    for f,hf in enumerate(('p','m')):
        for i,hi in enumerate(('p','m')):
            _,current=read_block(ROOT/f'm2_cluster_spin_{target}_{hf}{hi}.json')
            np.testing.assert_allclose(current['current_coefficients'][0],free[f'raw_g0_{f}{i}'],atol=2e-14,rtol=0)


@pytest.mark.parametrize('name',('proton','neutron','Q5_shell1'))
def test_saved_analysis_packets_reconstruct_all_arrays_and_diagnostics(name):
    saved,a=read_block(ROOT/f'm2_cluster_spin_{name}_analysis.json')
    paths=[ROOT/Path(p).name for p in saved['input_blocks']]
    rebuilt,b=analyze(paths,symmetry_complete=bool(saved['kinematically_completed_blocks']))
    for key in b:
        np.testing.assert_allclose(a[key],b[key],atol=4e-14,rtol=0)
    for key in ('two_structure_residual_norm','five_structure_residual_norm',
                'continuity_norm','energy_GeV2','maximum_pole_residual'):
        assert saved[key]==pytest.approx(rebuilt[key],abs=3e-14)
    assert rebuilt['five_structure_residual_norm']<1e-10
    assert rebuilt['two_structure_residual_norm']>1e-2


def test_Q_resolution_numerator_both_norm_state_and_mass_basis_attribution():
    from scripts.analyze_m2_cluster_spin_resolution import compare
    saved,a=read_block(ROOT/'m2_cluster_spin_Q5_resolution.json')
    old=[ROOT/Path(p).name for p in saved['old_blocks']]
    new=[ROOT/Path(p).name for p in saved['new_blocks']]
    report,b=compare(old,new)
    for key in b:np.testing.assert_allclose(a[key],b[key],atol=4e-14,rtol=0)
    np.testing.assert_allclose(a['total_change_pieces'].sum(axis=0),
        a['new_current']-a['old_current'],atol=5e-14,rtol=0)
    assert report['energy_change_GeV2']!=0
    assert report['independent_blocks_per_resolution']==2
    assert report['kinematically_completed_blocks']==['pm','mm']
    assert report['split_is_convention_not_unique_physics']
    assert not report['covariance_repaired'] and not report['physical']
    for label in report['attribution']:
        assert saved['attribution'][label]['amplitude_norm']==pytest.approx(
            report['attribution'][label]['amplitude_norm'],abs=4e-14)


@pytest.mark.parametrize('channel',('pp','mp'))
def test_larger_Q_pole_norm_and_raw_order_Ward_records(channel):
    info,a=read_block(ROOT/f'm2_cluster_spin_Q5_shell1_{channel}.json')
    assert info['parameters']['q5']==1 and info['parameters']['q6']==0
    assert max(info['Ward_order_norms'])<1e-11
    assert max(info['pole_residuals_GeV2'])<1e-12
    for label in ('out','in'):
        u,v=a[label+'_raw_state'],a[label+'_state']
        np.testing.assert_allclose(a[label+'_H']@u,info['energy_GeV2']*u,atol=3e-13,rtol=0)
        np.testing.assert_allclose(a[label+'_W']@v,u,atol=4e-14,rtol=0)
        assert np.vdot(u,a[label+'_N']@u)==pytest.approx(1,abs=4e-14)
        assert np.vdot(v,v)==pytest.approx(1,abs=4e-14)
