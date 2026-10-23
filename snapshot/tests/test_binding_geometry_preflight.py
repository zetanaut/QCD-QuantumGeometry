from itertools import combinations
from math import sqrt

import numpy as np
import pytest

from deuteron_wigner.quantum.binding_geometry_preflight import (
    free_invariant_mass,longitudinal_threshold,singlet_carrier,
    pair_action,generator_pair_action,local_color_action,geometry,
    pair_potential,partition_energies,connected_tree_length,cluster_case,
)
from deuteron_wigner.quantum.modular_gauge_network import generators


def test_free_threshold_from_cartesian_LF_vectors_and_Cauchy_minimum():
    masses=np.array([.22,.31,.38]);x=masses/masses.sum()
    transverse=np.outer(x,[.14,-.23])
    for boost in (.3,1.,4.):
        assert abs(free_invariant_mass(masses,boost*x,transverse)-masses.sum())<2e-15
    rng=np.random.default_rng(308)
    for _ in range(30):
        x=rng.dirichlet([2,2,2]);k=rng.normal(size=(3,2))*.1;k-=k.mean(axis=0)
        mass=free_invariant_mass(masses,x,k)
        assert abs(mass**2-np.sum((masses*masses+np.sum(k*k,axis=1))/x))<2e-14
        assert mass>=masses.sum()


def test_odd_grid_threshold_is_not_a_physical_binding_certificate():
    for k in (3,5,7,9,11,13,15,29,31,61,121):
        m,n=longitudinal_threshold([.27]*3,k)
        assert sum(n)==k and all(x>0 and x%2==1 for x in n)
        assert abs(m-free_invariant_mass([.27]*3,np.array(n)/k,np.zeros((3,2))))<3e-15
        assert m>=.81-1e-15
        if k%3==0: assert abs(m-.81)<1e-15
    # A mock .9 level can be below one grid's free minimum and above the
    # actual free continuum threshold. This is not a solved eigenstate.
    assert longitudinal_threshold([.27]*3,5)[0]>.9>.81
    assert longitudinal_threshold([.27]*3,121)[0]<.811
    assert longitudinal_threshold([.27]*3,5.)==longitudinal_threshold([.27]*3,5)


@pytest.mark.parametrize('k2',[0,2,4,5.1,True])
def test_invalid_longitudinal_carriers_are_rejected(k2):
    with pytest.raises(ValueError): longitudinal_threshold([.27]*3,k2)


def test_all_epsilon_partitions_have_exact_rank_and_common_Gram():
    parts,e,g,u=singlet_carrier()
    assert len(parts)==10 and e.shape==(729,10) and u.shape==(729,5)
    assert np.linalg.norm(u.T@u-np.eye(5))<2e-15
    assert np.max(abs(np.linalg.eigvalsh(g)-np.array([0]*5+[2]*5)))<2e-15
    assert np.max(abs(g*3-np.rint(g*3)))<2e-15
    assert np.max(abs(abs(g)-(np.eye(10)*2/3+np.ones((10,10))/3)))<1e-15
    assert np.linalg.norm(e-u@u.T@e)<3e-15
    # Distinct static slots embed one-to-one into fixed-occupancy exterior
    # states. No additional orthogonal topology register is introduced.
    assert np.count_nonzero(e[:,0])==36
    assert abs(g[0,1])==pytest.approx(1/3)
    assert not any(x.flags.writeable for x in (e,g,u))


def test_raw_local_generators_annihilate_all_five_global_singlets():
    _,e,_,u=singlet_carrier()
    for t in generators():
        value=sum((local_color_action(e,i,t) for i in range(6)),np.zeros_like(e,dtype=complex))
        assert np.linalg.norm(value)<2e-15
    # Complete global-singlet Casimir sum, not one chosen baryon partition.
    terms=[pair_action(u,i,j) for i,j in combinations(range(6),2)]
    scale=sum(np.linalg.norm(t) for t in terms)+4*np.linalg.norm(u)
    assert np.linalg.norm(sum(terms)+4*u)<16*np.finfo(float).eps*scale


def test_pair_Fierz_action_matches_independent_full_color_generators():
    rng=np.random.default_rng(394);v=rng.normal(size=(729,2))+1j*rng.normal(size=(729,2))
    for i,j in combinations(range(6),2):
        assert np.linalg.norm(pair_action(v,i,j)-generator_pair_action(v,i,j))<2e-14


def test_hidden_color_channels_are_not_absent_or_new_topology_states():
    _,e,_,u=singlet_carrier()
    ca=4*np.eye(5)+2*sum(u.T@pair_action(u,i,j) for i,j in combinations(range(3),2))
    assert np.linalg.norm(np.linalg.eigvalsh(ca)-[0,3,3,3,3])<5e-15
    b=u.T@e[:,0]
    assert np.linalg.norm(ca@b)<3e-15


def test_partition_energy_labels_fail_necessary_Hermiticity_condition():
    _,e,g,u=singlet_carrier();d=partition_energies(geometry(4.))
    assert np.linalg.norm(g*d[None,:]-d[:,None]*g)>1
    # Explicitly construct the best Hermitian least-squares operator for
    # this tight frame. It is Hermitian but cannot satisfy H E = E D.
    b=u.T@e;h=(b*d[None,:])@b.T/2
    assert np.linalg.norm(h-h.T)<2e-14
    assert np.linalg.norm(h@b-b*d[None,:])>1
    equal=np.full(10,6.)
    assert np.linalg.norm(g*equal[None,:]-equal[:,None]*g)==0
    assert np.linalg.norm(6*np.eye(5)@b-b*equal[None,:])==0


def test_connected_tree_confines_separated_singlets_spuriously():
    for r in (2.,4.,8.,16.):
        assert abs(connected_tree_length(geometry(r))-(4+r))<1e-13
    # Two independently connected three-quark clusters have total MST 4.
    h=1e-4
    derivative=(connected_tree_length(geometry(8+h))-connected_tree_length(geometry(8-h)))/(2*h)
    assert abs(derivative-1)<1e-10


def test_color_pair_ground_state_retains_virtual_hidden_color_and_raw_residual():
    _,e,_,u=singlet_carrier();r=geometry(4.);h=pair_potential(r)
    w,v=np.linalg.eigh(h);b=u.T@e[:,0];raw=u@v[:,0]
    assert abs(b@h@b-6)<1e-13
    assert w[0]<6-.001
    assert 1-abs(b@v[:,0])**2>1e-5
    independent=sum(-1.5*np.linalg.norm(r[i]-r[j])*generator_pair_action(raw,i,j)
                    for i,j in combinations(range(6),2))
    assert np.linalg.norm(independent-w[0]*raw)<4e-14


def test_geometry_response_is_derivative_of_same_operator_not_a_current_head():
    for sep in (2.,4.,8.):
        velocity=np.zeros((6,3));velocity[3:,0]=1
        derivative=pair_potential(geometry(sep),derivative=velocity)
        step=2e-5
        delta=(pair_potential(geometry(sep+step))-pair_potential(geometry(sep-step)))/(2*step)
        assert np.linalg.norm(delta-derivative)<1e-9
        force=cluster_case(sep)['force']
        energy=lambda r:np.linalg.eigvalsh(pair_potential(geometry(r)))[0]
        assert abs(force+(energy(sep+step)-energy(sep-step))/(2*step))<1e-9


def test_static_geometry_rotation_translation_and_slot_relabeling():
    _,_,_,u=singlet_carrier();r=geometry(3.4);h=pair_potential(r)
    rotation=np.array([[0,0,1],[1,0,0],[0,1,0]])
    assert np.linalg.norm(pair_potential(r@rotation+[2,-1,.7])-h)<2e-14
    assert np.linalg.norm(pair_potential(2*r,coefficient=.3)-.6*h)<2e-14
    permutation=[3,1,5,0,4,2]
    new=pair_potential(r[permutation]);up=u.reshape((3,)*6+(5,)).transpose(*permutation,6).reshape(729,5)
    recoupling=u.T@up
    assert np.linalg.norm(recoupling.T@recoupling-np.eye(5))<3e-15
    assert np.linalg.norm(new-recoupling@h@recoupling.T)<5e-14


def test_full_cluster_shift_decays_but_is_not_zero_by_singlet_projection():
    cases=[cluster_case(r) for r in (8,16,32)]
    shift=[-x['pair_ground_shift_from_two_singlets'] for x in cases]
    assert all(s>0 for s in shift)
    assert shift[0]>5*shift[1]>25*shift[2]
    assert cases[0]['hidden_color_weight']>cases[1]['hidden_color_weight']>cases[2]['hidden_color_weight']>0
    # This static algebraic tail is not identified with physical nuclear
    # forces or declared a completed cluster/covariance prescription.


def test_aligned_geometry_has_derived_not_fitted_large_distance_tail():
    _,e,_,u=singlet_carrier();b=u.T@e[:,0]
    s=sum(u.T@generator_pair_action(u,i,i+3) for i in range(3))
    assert abs(b@s@b)<1e-15
    assert abs(np.linalg.norm(s@b)**2-1)<3e-15
    ca=4*np.eye(5)+2*sum(u.T@generator_pair_action(u,i,j) for i,j in combinations(range(3),2))
    for r in (4.,8.):
        d=sqrt(r*r+1)
        exact=6*np.eye(5)+1.5*(d-1)*ca+1.5*(d-r)*s
        assert np.linalg.norm(exact-pair_potential(geometry(r)))<16*np.finfo(float).eps*np.linalg.norm(exact)
    # Coupling 3/(4R), octet gap 9R/2, ||S b||^2=1 -> -1/(8R^3).
    error=[]
    for r in (32.,64.):
        error.append(abs(cluster_case(r)['pair_ground_shift_from_two_singlets']*r**3+1/8))
    assert error[1]<error[0]/1.8 and error[1]<.02/8
