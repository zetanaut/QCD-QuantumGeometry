from dataclasses import replace

import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h1 import *
from deuteron_wigner.microscopic.h1.injections import INJECTIONS
from deuteron_wigner.microscopic.h1.tensor_network import (
    bond_dimension_benchmark, exact_tensorize,
)


def plans():
    return tuple(compile_plan(H1AssumptionBundle(c,s)) for c,s in (
        ("INDUCED_REFIT","EFFECTIVE_COLOR_SPIN"),
        ("ZERO_CONFINEMENT","EFFECTIVE_COLOR_SPIN"),
        ("INDUCED_REFIT","NONE"),
    ))


def test_assumption_plan_stability_and_scope():
    for plan in plans():
        assert plan==compile_plan(plan.assumption)
        assert plan.plan_id.startswith("C8:H1:PLAN:")
        with pytest.raises(ArchitectureError): plan.require_output("TMD")
    with pytest.raises(ArchitectureError):
        compile_plan(plans()[0].assumption,explicit_qqqg=True)
    with pytest.raises(ArchitectureError):
        compile_plan(plans()[0].assumption,frozen_confinement=True)


@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
@pytest.mark.parametrize("jz",[pytest.param(__import__("fractions").Fraction(1,2)),pytest.param(__import__("fractions").Fraction(-1,2))])
def test_nontrivial_nested_basis(target,jz):
    tower=build_basis_tower(target=target,Jz=jz)
    assert [b.dimension for b in tower.bases]==[4,7,10]
    for basis in tower.bases:
        assert {s.Lz for s in basis.states}=={-1,0,1}
        assert all(s.Jz==jz and s.center_of_mass_quantum==0 for s in basis.states)
    for mapping in tower.comparison_maps:
        assert np.allclose(mapping.conj().T@mapping,np.eye(mapping.shape[1]))


@pytest.mark.parametrize("plan",plans())
def test_renormalization_flow_and_hamiltonian(plan):
    trajectory=fit_trajectory(plan,build_basis_tower())
    assert len(trajectory.members)==3
    for member,ham in zip(trajectory.members,trajectory.hamiltonians):
        assert abs(np.linalg.eigvalsh(ham.matrix)[0]-0.88**2)<2e-12
        assert np.max(np.abs(ham.matrix-ham.matrix.conj().T))<1e-14
        assert member.hamiltonian_id==ham.hamiltonian_id
        assert ham.discrepancy.interpretation=="UNIMPLEMENTED_NOT_ZERO"
    kappas=[dict(m.parameters)["kappa4"] for m in trajectory.members]
    if plan.assumption.confinement_route=="INDUCED_REFIT":
        assert len(set(kappas))==3
    else:
        assert kappas==[0,0,0]


def test_exact_krylov_and_current_compatibility():
    trajectory=fit_trajectory(plans()[0],build_basis_tower())
    for ham in trajectory.hamiltonians:
        exact=exact_solve(ham); krylov=krylov_solve(ham)
        assert abs(exact.eigenvalues[0]-krylov.eigenvalues[0])<2e-11
        assert max(exact.residuals)<2e-13
        assert max(krylov.residuals)<2e-11
        current=ValenceVectorCurrent.for_hamiltonian(ham)
        psi=exact.eigenvectors[:,0]
        assert abs(current.expectation(ham,psi)-1)<2e-12
    wrong=trajectory.hamiltonians[1]
    with pytest.raises(ArchitectureError):
        ValenceVectorCurrent.for_hamiltonian(trajectory.hamiltonians[0]).matrix(wrong)


def test_neutron_charge_is_correlated_prediction():
    tower=build_basis_tower(target="NEUTRON")
    ham=fit_trajectory(plans()[0],tower).hamiltonians[0]
    psi=exact_solve(ham).eigenvectors[:,0]
    current=ValenceVectorCurrent.for_hamiltonian(ham)
    assert current.expectation(ham,psi,Q2=0)==0
    assert current.expectation(ham,psi,Q2=0.3)!=0


def test_ttn_exact_variational_and_operator():
    ham=fit_trajectory(plans()[0],build_basis_tower()).hamiltonians[-1]
    exact=exact_solve(ham); psi=exact.eigenvectors[:,0]
    current=ValenceVectorCurrent.for_hamiltonian(ham).matrix(ham,0.3)
    state=exact_tensorize(ham,psi)
    assert state.overlap(state)==pytest.approx(1)
    manifest=bond_dimension_benchmark(ham,psi,current)
    energies=[r.energy for r in manifest.results]
    assert all(a>=b-1e-13 for a,b in zip(energies,energies[1:]))
    assert all(e>=exact.eigenvalues[0]-2e-12 for e in energies)
    assert manifest.results[-1].exact_overlap>1-2e-12
    assert abs(manifest.results[-1].energy-exact.eigenvalues[0])<2e-12
    assert manifest.results[0].oam_feature_error>1e-5
    tensor_op=ValenceTensorOperator.from_hamiltonian(ham)
    rng=np.random.default_rng(8); vector=rng.normal(size=ham.basis.dimension)
    assert np.allclose(tensor_op.apply(vector),ham.apply(vector))
    U=ValenceCouplingTree.recoupling(5)
    assert np.max(np.abs(U.conj().T@U-np.eye(5)))<2e-15


def test_mandatory_controlled_benchmarks():
    toy=renormalization_toy_benchmark()
    assert all(abs(row["pole_mass2"]-0.81)<1e-14 and abs(row["renormalized_charge"]-1)<1e-14 for row in toy)
    tracking=state_tracking_benchmark()
    assert tracking["eigenvalue_order_fails"]
    assert tracking["overlap_chain"][-1]==tracking["intended_end"]
    flow=confinement_flow_benchmark()
    assert set(flow)=={"PLAN_A","PLAN_B","PLAN_C"}
    assert len({flow[x]["plan_id"] for x in flow})==3


def test_state_bundle_is_valence_only():
    ham=fit_trajectory(plans()[0],build_basis_tower()).hamiltonians[0]
    sol=exact_solve(ham); cur=ValenceVectorCurrent.for_hamiltonian(ham)
    bundle=ValenceMicroscopicStateBundle("C8:BUNDLE:1",ham.hamiltonian_id,ham.plan_id,ham.basis.resolution.resolution_id,ham.basis.basis_id,float(sol.eigenvalues[0]),tuple(sol.eigenvectors[:,0]),cur.current_id,sol.residuals[0],sol.residuals[0],0.0,"LARGEST_COMPONENT_REAL_POSITIVE",ham.discrepancy.omitted)
    assert bundle.scope=="C8_H1_VALIDATION_ONLY" and bundle.sector_scope=="VALENCE_ONLY"


def test_injection_catalogue_complete_and_stable():
    assert len(INJECTIONS)==56
    assert len({x[0] for x in INJECTIONS})==56
