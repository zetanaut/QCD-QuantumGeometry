import numpy as np
import pytest
from scipy.linalg import expm,eigh

from deuteron_wigner.quantum.path_dirac_matching import (
    PAULI,canonical_map,direct_dirac_map,constrained_good_map,gap,
    spin_components,path_dictionary,coordinate_image,fit_tangent,
    maximal_spin_blind,energy_coefficient,color_map,spin_loss,
    reduced_spin_coefficient,spin_retention_loss,
)
from deuteron_wigner.quantum.lf_path_states import PathCarrier,DressingPath
from deuteron_wigner.quantum.finite_lf_coupling import LFCell
from deuteron_wigner.quantum.canonical_lf_network import cubic_vertex_tensor
from deuteron_wigner.quantum.modular_gauge_network import generators


@pytest.fixture(scope='module')
def setup():
    c,p=path_dictionary();return c,p,canonical_map(c),gap(c)


@pytest.mark.parametrize('mass',[0.,.27,.54])
def test_full_Dirac_and_independent_good_constraint_match_every_spin_mode(mass,setup):
    c,p,_,_=setup;c=PathCarrier(c.cell,c.momenta[::2],c.quarks,mass)
    v=canonical_map(c)
    assert np.linalg.norm(v-direct_dirac_map(c))<1e-17
    assert np.linalg.norm(v-constrained_good_map(c))<1e-17
    components=spin_components(v)
    assert np.linalg.norm(v-np.einsum('ma,aij->mij',components,PAULI))<3e-18
    if mass==0:
        assert np.linalg.norm(components[:,1:3])<1e-18
        assert np.linalg.norm(components[:,3])>1e-4
    else:
        assert np.linalg.norm(components[:,1:3])>1e-5


def test_open_color_map_matches_preserved_cell_normalized_cubic_tensor(setup):
    c,p,v,d=setup;full=color_map(v)
    for j,k in enumerate(c.momenta[::2]):
        expected=cubic_vertex_tensor('q_to_qg',c.quarks[0]-k,k,c.quarks[0],c.mass)/np.sqrt(c.cell.volume)
        actual=full[2*j:2*j+2].transpose(1,2,0,3,4,5).reshape(6,16,6)
        assert np.linalg.norm(actual-expected)<5e-18
    # Explicit color trace agrees with C_F/2 spin-average metric.
    assert abs(np.linalg.norm(full)**2/6-(4/3)/2*np.linalg.norm(v)**2)<1e-19


def test_arbitrarily_many_spin_blind_paths_cannot_match_canonical_map(setup):
    c,p,v,d=setup;blind=maximal_spin_blind(v,d);exact=-v/d[:,None,None]
    result=fit_tangent(v,d,coordinate_image(c,p))
    assert result['rank']==4
    assert np.linalg.norm(result['wave']-blind)<1e-14
    assert .09<np.linalg.norm(blind-exact)/np.linalg.norm(exact)<.10
    # Residual is orthogonal to ALL arbitrary mode scalars times identity.
    assert np.linalg.norm(np.trace(exact-blind,axis1=1,axis2=2))<1e-17
    assert result['energy_deficit_GeV']>1e-5


def test_geometric_rank_is_separate_from_spin_information(setup):
    c,p,v,d=setup
    old=dict(list(p.items())[:3]);old_image=coordinate_image(c,old,(0,1,2,3))
    result=fit_tangent(v,d,old_image)
    assert result['rank']==12 and result['relative_wave_error']>.01
    full=fit_tangent(v,d,coordinate_image(c,p,(0,1,2,3)))
    assert full['rank']==16 and full['relative_wave_error']<1e-12
    assert full['singular_values'][-1]>1e10*full['threshold']
    assert np.linalg.norm(full['stationary_residual'])<2e-15
    diagonal=fit_tangent(v,d,coordinate_image(c,p,(0,3)))
    assert diagonal['rank']==8 and .03<diagonal['relative_wave_error']<.031


def test_full_gap_variational_deficit_and_independent_small_matrix_limit(setup):
    c,p,v,d=setup;exact=-v/d[:,None,None]
    optimum=energy_coefficient(v,d,exact)
    for spins in ((0,),(0,3),(0,1,2,3)):
        result=fit_tangent(v,d,coordinate_image(c,p,spins))
        difference=energy_coefficient(v,d,result['wave'])-optimum
        assert abs(difference-result['energy_deficit_GeV'])<2e-18
    # Independent finite reference eigenvalue: arbitrary common C0 shifts
    # both fits; no physical counterterm, order-two completion or mass is selected.
    for parent in range(2):
        vector=np.sqrt(4/3)*v[:,:,parent].ravel()
        energies=np.repeat(d,2)
        expected=-np.sum(abs(vector)**2/energies)
        errors=[]
        for coupling in (.02,.01):
            h=np.diag(np.r_[0.,energies]).astype(complex)
            h[1:,0]=coupling*vector;h[0,1:]=h[1:,0].conj()
            errors.append(abs(eigh(h,eigvals_only=True)[0]/coupling**2-expected))
        assert errors[1]<errors[0]/3.9


def test_normalized_spin_channel_sees_defect_while_charge_does_not(setup):
    c,p,v,d=setup;exact=-v/d[:,None,None]
    blind=maximal_spin_blind(v,d)
    for chi in (exact,blind):
        roundoff=16*np.finfo(float).eps*(4/3)*np.linalg.norm(chi)**2
        for rho in (np.eye(2)/2,(np.eye(2)+PAULI[1])/2,(np.eye(2)+PAULI[3])/2):
            response=reduced_spin_coefficient(chi,rho)
            assert abs(np.trace(response))<roundoff*np.linalg.norm(rho)
            assert np.linalg.norm(response-response.conj().T)<roundoff*np.linalg.norm(rho)
    assert np.linalg.norm(spin_retention_loss(blind))<roundoff
    losses=spin_retention_loss(exact)
    assert np.all(losses>1e-5)
    assert abs(losses[2]-np.mean(spin_loss(exact)))<roundoff
    for a,s in enumerate(PAULI[1:]):
        comm=s@exact-exact@s
        assert abs(losses[a]-(4/3)/4*np.linalg.norm(comm)**2)<roundoff
        # Arbitrary common second-order unitary spin rotation does not
        # alter the same-axis, opposite-input average.
        h=.7*PAULI[1]+.4*PAULI[2]+.8*PAULI[3]
        assert abs(np.trace(s@(-1j*(h@s-s@h))))<1e-15


def test_spin_coefficient_from_exactly_normalized_finite_isometry(setup):
    c,p,v,d=setup;chi=-v/d[:,None,None]
    a=(4/3)*np.einsum('mji,mjk->ik',chi.conj(),chi)
    rho=(np.eye(2)+.6*PAULI[1]+.4*PAULI[3])/2
    target=reduced_spin_coefficient(chi,rho)
    errors=[]
    for g in (.2,.1):
        values,u=eigh(np.eye(2)+g*g*a);n=(u/np.sqrt(values))@u.conj().T
        state=n@rho@n
        actual=state+g*g*(4/3)*np.einsum('mij,jk,mlk->il',chi,state,chi.conj())
        assert abs(np.trace(actual)-1)<1e-15
        errors.append(np.linalg.norm((actual-rho)/g**2-target))
    assert errors[1]<errors[0]/3.9


def test_spin_basis_change_transforms_both_map_and_coordinate_image(setup):
    c,p,v,d=setup;image=coordinate_image(c,p)
    incoming=expm(1j*(.2*PAULI[1]-.3*PAULI[2]))
    outgoing=np.array([expm(1j*(.1*(m+1)*PAULI[1]+.2*PAULI[3])) for m in range(len(v))])
    changed=outgoing.conj().transpose(0,2,1)@v@incoming
    changed_image=np.einsum('mij,mjka,kl->mila',outgoing.conj().transpose(0,2,1),image,incoming)
    old=fit_tangent(v,d,image);new=fit_tangent(changed,d,changed_image)
    assert abs(old['energy_deficit_GeV']-new['energy_deficit_GeV'])<1e-18
    assert abs(old['relative_wave_error']-new['relative_wave_error'])<2e-15


def test_kinematic_LF_boosts_preserve_matching_and_deficit_units(setup):
    c,p,v,d=setup
    for boost in (.4,2.7):
        scale=np.array([boost,1,1])
        other=PathCarrier(LFCell(tuple(np.array(c.cell.lengths)/scale)),c.momenta[::2]*scale,c.quarks*scale,c.mass)
        paths={name:DressingPath(tuple(np.array(path.vertices)/scale)) for name,path in p.items()}
        assert np.linalg.norm(canonical_map(other)*boost-v)<2e-17
        assert np.linalg.norm(gap(other)*boost-d)<2e-16
        result=fit_tangent(canonical_map(other),gap(other),coordinate_image(other,paths))
        base=fit_tangent(v,d,coordinate_image(c,p))
        assert abs(result['energy_deficit_GeV']*boost-base['energy_deficit_GeV'])<2e-18
    beta=np.array([1.6,-1.6])
    boost=lambda a:np.column_stack((a[:,0],a[:,1:]+a[:,0,None]*beta))
    other=PathCarrier(c.cell,boost(c.momenta[::2]),boost(c.quarks),c.mass)
    paths={}
    for name,path in p.items():
        r=np.array(path.vertices);r[:,0]+=r[:,1:]@beta;paths[name]=DressingPath(tuple(r))
    assert np.linalg.norm(canonical_map(other)-v)<2e-16
    assert np.linalg.norm(gap(other)-d)<2e-16
    assert np.linalg.norm(coordinate_image(other,paths)-coordinate_image(c,p))<2e-16


def test_duplicate_path_does_not_invent_rank_or_repair_spin_blindness(setup):
    c,p,v,d=setup;p={**p,'duplicate':p['LF_detour']}
    result=fit_tangent(v,d,coordinate_image(c,p))
    assert result['rank']==4 and result['singular_values'][-1]<1e-17
    assert result['relative_wave_error']>.09


def test_finite_path_interpolation_does_not_prove_enlarged_domain_completion(setup):
    c,p,v,d=setup
    other=PathCarrier(c.cell,[*c.momenta[::2],(.25,.2,.2)],c.quarks,c.mass)
    v=canonical_map(other);d=gap(other)
    result=fit_tangent(v,d,coordinate_image(other,p,(0,1,2,3)))
    assert result['rank']==16 and v.size==24
    assert result['relative_wave_error']>.01
    # A known canonical node retains its full spin matrix for every new
    # route. This is reuse of physics, NOT a learned one-parameter solution
    # of the full nucleon or a claim that the soft residual is fixed.
    chi=-constrained_good_map(other)/d[:,None,None]
    assert np.linalg.norm(d[:,None,None]*chi+v)<2e-17


def test_zero_target_remains_zero_without_rank_or_normalization_fictions(setup):
    c,p,v,d=setup
    result=fit_tangent(np.zeros_like(v),d,coordinate_image(c,p))
    assert result['relative_wave_error']==0 and result['energy_deficit_GeV']==0
    assert np.linalg.norm(result['wave'])==0


@pytest.mark.parametrize('bad',[(True,),(),(0,0),(4,)])
def test_invalid_spin_coordinates_are_not_accepted_as_physics(bad,setup):
    c,p,_,_=setup
    with pytest.raises(ValueError):coordinate_image(c,p,bad)
