"""Actual cubic higher-Fock matching: energy, norm, current and quantum parity."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import linalg

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot,baryon_density
from deuteron_wigner.quantum.m2_excluded_fock_action import build_excluded_source
from deuteron_wigner.quantum.m2_excluded_fock_matching import (
    ExcludedFockReference,intermediate_density_action,
)

@pytest.fixture(scope="module",params=[0,1])
def reference(request):
    p=build_coupled_pilot(total_k2=9,cutoff=0,target="PROTON",jz_twice=1,
        b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=.3)
    sources=[build_excluded_source(p,s,out_cutoff=request.param) for s in ("qqqgg","qqqq-qbar")]
    return ExcludedFockReference(p,sources)

@pytest.fixture(scope="module")
def density_current(reference):
    return reference.flow_current((.12,.16))

def test_exact_source_reachable_free_reference(reference):
    for sector in reference.sectors:
        v=sector.source.matrix_per_gs
        np.testing.assert_allclose(sector.lift(sector.coupling_per_gs),v,atol=2e-14)
        assert sector.reconstruction_residual_per_gs<4e-14
        energies=sector.source.space.free_energies(.4,.35,0.)
        test=sector.coupling_per_gs.astype(complex)
        np.testing.assert_allclose(sector.lift(sector.energies_GeV2[:,None]*test),
            energies[:,None]*sector.lift(test),atol=6e-14)
        np.testing.assert_allclose(sector.project(sector.lift(test)),test,atol=2e-14)
    assert reference.matrix.shape[0] in (92,165)

def test_full_packed_self_energy_matches_compressed_parent(reference):
    n=len(reference.pilot.matrix)
    expected=reference.pilot.matrix.copy()
    metric=np.eye(n,dtype=complex)
    for sector in reference.sectors:
        s=sector.source
        e=s.space.free_energies(.4,.35,0.)
        ix=np.ix_(s.retained_columns,s.retained_columns)
        expected[ix]+=.09*s.self_energy(1.4,e)
        metric[ix]+=.09*s.norm_correction(1.4,e)
    np.testing.assert_allclose(reference.reduction.effective_hamiltonian(1.4),expected,atol=2e-14)
    np.testing.assert_allclose(reference.reduction.norm_metric(1.4),metric,atol=2e-14)

def test_ground_root_independent_diagonalization_and_full_reconstruction(reference):
    state=reference.reduction.lowest_state()
    e,u=linalg.eigh(reference.matrix)
    assert state.energy_GeV2==pytest.approx(e[0],abs=3e-13)
    assert abs(np.vdot(u[:,0],state.reconstructed))**2==pytest.approx(1.,abs=3e-13)
    residual,probability=reference.full_reconstructed_residual(state.energy_GeV2,state.retained)
    assert residual<3e-13
    assert probability==pytest.approx(state.excluded_probability,abs=1e-14)
    assert 5e-7<probability<2e-6
    assert e[0]<linalg.eigvalsh(reference.pilot.matrix)[0]-1e-6

def test_baryon_charge_requires_negative_antiquark_partner(reference):
    np.testing.assert_allclose(reference.density((0.,0.)),np.eye(len(reference.matrix)),atol=2e-14)
    for sector in reference.sectors:
        s=sector.source.space
        v=sector.source.matrix_per_gs
        np.testing.assert_allclose(intermediate_density_action(s,v,(0.,0.),b_GeV=.4),v,atol=2e-14)
        # For sea, omitting antiquark would give 4/3, not conserved baryon 1.
        if s.sector=="qqqq-qbar":
            assert np.linalg.norm(4/3*v-v)>.1

def test_two_energy_density_reversal_and_nonzero_transfer_matching(reference,density_current):
    rho,_,_=density_current
    np.testing.assert_allclose(reference.density((-.12,-.16)),rho.conj().T,atol=2e-14)
    r=reference.reduction
    state=r.lowest_state()
    e=state.energy_GeV2
    matched=r.matched_operator(rho,e,e)
    value=np.vdot(state.retained,matched@state.retained)
    assert value==pytest.approx(np.vdot(state.reconstructed,rho@state.reconstructed),abs=2e-13)
    bare=baryon_density(reference.pilot,(.12,.16))
    naive=np.vdot(state.retained,bare@state.retained)/np.vdot(state.retained,state.retained)
    assert abs(value-naive)>5e-8  # real omitted-operator effect, not just norm
    # Check non-Hermitian off-energy matrix reversal, not just a ground scalar.
    np.testing.assert_allclose(r.matched_operator(rho,1.3,1.6).conj().T,
        r.matched_operator(rho.conj().T,1.6,1.3),atol=2e-14)

def test_projected_density_respects_particle_statistics(reference):
    for sector in reference.sectors:
        s=sector.source.space
        v=sector.source.matrix_per_gs[:,:3]
        rho=lambda a:intermediate_density_action(s,a,(.12,.16),b_GeV=.4)
        out=rho(v)
        np.testing.assert_allclose(s.permute(out,(1,0,2,3,4)),-out,atol=2e-14)
        if s.sector=="qqqq-qbar":
            np.testing.assert_allclose(s.permute(out,(0,1,3,2,4)),-out,atol=2e-14)
        else:
            np.testing.assert_allclose(s.permute(out,(0,1,2,4,3)),out,atol=2e-14)

def test_induced_number_changing_current_and_off_shell_identity(reference,density_current):
    rho,jx,jy=density_current
    r=reference.reduction
    divergence=.12*jx+.16*jy
    np.testing.assert_allclose(divergence,r.h@rho-rho@r.h,atol=3e-15)
    n=len(reference.pilot.matrix)
    assert np.linalg.norm(jx[n:,:n])>1e-3
    ef,ei=1.3,1.6
    wo,wi=r.wave_operator(ef),r.wave_operator(ei)
    deff=r.effective_hamiltonian(ef)-ef*np.eye(n)
    deifi=r.effective_hamiltonian(ei)-ei*np.eye(n)
    effective=r.matched_operator(rho,ef,ei)
    rhs=(ef-ei)*effective+deff@r.p.conj().T@rho@wi-wo.conj().T@rho@r.p@deifi
    np.testing.assert_allclose(r.matched_operator(divergence,ef,ei),rhs,atol=3e-14)
    # A bare commutator of one Heff and an arbitrary two-energy operator fails.
    naive=r.effective_hamiltonian(ef)@effective-effective@r.effective_hamiltonian(ei)
    assert np.linalg.norm(naive-r.matched_operator(divergence,ef,ei))>1e-6

def test_gluon_and_quark_emission_interfere_before_matching(reference):
    s=reference.sectors[0].source
    q,g=s.components.values()
    d=1.4-s.space.free_energies(.4,.35,0.)
    incoherent=q.conj().T@(q/d[:,None])+g.conj().T@(g/d[:,None])
    difference=s.self_energy(1.4,1.4-d)-incoherent
    if s.space.cutoff==0:
        np.testing.assert_allclose(g,0,atol=0)  # linear-Q gg vertex has no ground HO mode
        np.testing.assert_allclose(difference,0,atol=1e-16)
    else:
        assert np.linalg.norm(difference)>.01

def test_energy_linearization_remains_hermitian_ritz_and_current_matched(reference,density_current):
    r=reference.reduction
    rho,jx,jy=density_current
    e0=linalg.eigvalsh(reference.pilot.matrix)[0]
    model=r.linearized_model(e0)
    np.testing.assert_allclose(model.matrix,model.matrix.conj().T,atol=2e-14)
    np.testing.assert_allclose(model.operator(np.eye(len(r.h))),np.eye(len(model.matrix)),atol=2e-14)
    e=linalg.eigvalsh(model.matrix)[0]
    exact=linalg.eigvalsh(r.h)[0]
    assert exact-1e-13<=e<exact+1e-10
    u=model.parent_isometry
    leakage=np.eye(len(r.h))-u@u.conj().T
    correction=u.conj().T@(r.h@leakage@rho-rho@leakage@r.h)@u
    comm=model.matrix@model.operator(rho)-model.operator(rho)@model.matrix
    np.testing.assert_allclose(model.operator(.12*jx+.16*jy),comm+correction,atol=2e-14)

def test_actual_matched_h2_controlled_gate_energy_and_state_parity(reference):
    from deuteron_wigner.quantum.m2_symmetry_graph_solver import SymmetryGraphCircuit
    model=reference.reduction.linearized_model(1.4)
    c=SymmetryGraphCircuit(model.matrix,root=0,support_threshold=1e-14)
    angles=c.uniform_parameters()+.15*np.random.default_rng(142).normal(size=c.parameter_count)
    classical=c.classical_state(angles)
    actual=np.asarray(c.state(angles))
    np.testing.assert_allclose(actual[:c.dimension],classical,atol=2e-13)
    assert np.linalg.norm(actual[c.dimension:])<1e-13
    assert c.energy(angles)==pytest.approx(np.vdot(classical,model.matrix@classical).real,abs=2e-13)

def test_same_dimension_does_not_qualify_different_source_pilot(reference):
    other=replace(reference.pilot,width_GeV=.41)
    with pytest.raises(ValueError,match="exact retained"):
        ExcludedFockReference(other,[s.source for s in reference.sectors])


def test_equal_mass_charge_and_continuity_do_not_identify_finite_sector_counterterms(reference):
    from scipy.optimize import brentq
    h=reference.matrix
    nq=reference.pilot.bases[0].shape[1]
    np_=len(reference.pilot.matrix)
    a=np.zeros(len(h)); a[:nq]=1
    b=np.zeros(len(h)); b[nq:np_]=1
    anchor=linalg.eigvalsh(h,subset_by_index=[0,0])[0]
    rho=reference.density((.12,.16),charge="electromagnetic")
    values=[]
    for shift in (-.5,.5):
        family=lambda ct:h+np.diag(ct*a+shift*b)
        ct=brentq(lambda ct:linalg.eigvalsh(family(ct),subset_by_index=[0,0])[0]-anchor,-1.,1.,xtol=1e-14)
        energy,state=linalg.eigh(family(ct),subset_by_index=[0,0])
        assert energy[0]==pytest.approx(anchor,abs=1e-13)
        assert np.vdot(state[:,0],state[:,0])==pytest.approx(1.,abs=1e-14)
        # Exact charge I and the gauged commutator construction close for BOTH
        # members, yet the independent finite-transfer matrix element differs.
        current=(family(ct)@rho-rho@family(ct))/.2
        np.testing.assert_allclose(.2*current,family(ct)@rho-rho@family(ct),atol=1e-15)
        values.append(np.vdot(state[:,0],rho@state[:,0]).real)
    assert abs(values[1]-values[0])>4e-4
    # This is an identifiability witness, NOT a selected counterterm or prior.
