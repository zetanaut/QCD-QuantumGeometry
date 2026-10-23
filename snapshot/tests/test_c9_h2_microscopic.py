import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h2 import *
from deuteron_wigner.microscopic.h2.diagnostics import solve
from deuteron_wigner.microscopic.h2.injections import INJECTIONS

def plans():
    return tuple(compile_h2_plan(H2AssumptionBundle(x)) for x in ("INDUCED_REFIT","ZERO_CONFINEMENT"))

def test_compiler_exclusivity_and_stability():
    for p in plans(): assert p==compile_h2_plan(p.bundle)
    assert len({p.plan_id for p in plans()})==2
    with pytest.raises(ArchitectureError): compile_h2_plan(plans()[0].bundle,h1_color_spin=True)
    with pytest.raises(ArchitectureError): compile_h2_plan(plans()[0].bundle,analytic_c4=True)

@pytest.mark.parametrize("target",["PROTON","NEUTRON"])
def test_complete_growing_basis(target):
    tower=build_coupled_basis_tower(target)
    assert [(b.qqq_dimension,b.qqqg_dimension) for b in tower]==[(4,6),(7,10),(10,14)]
    for b in tower:
        assert {x.color_multiplicity for x in b.gluon_states}=={1,2}
        assert {x.gluon_helicity for x in b.gluon_states}=={-1,1}
        assert all(sum(x.longitudinal_partition)==1 for x in b.gluon_states)

@pytest.mark.parametrize("plan",plans())
def test_coupled_hamiltonian_flow_and_solvers(plan):
    trajectory=fit_h2_trajectory(plan,build_coupled_basis_tower())
    assert len(trajectory.hamiltonians)==3
    for member,h in zip(trajectory.members,trajectory.hamiltonians):
        assert abs(member["mass2"]-.7744)<2e-12
        assert np.max(abs(h.matrix-h.matrix.conj().T))<2e-14
        rng=np.random.default_rng(9);x=rng.normal(size=h.basis.dimension)
        assert np.allclose(h.apply(x),h.matrix@x)
        e,v=solve(h)
        from scipy.sparse.linalg import eigsh,LinearOperator
        ke,kv=eigsh(LinearOperator(h.matrix.shape,matvec=h.apply,dtype=float),k=2,which="SA",v0=np.ones(h.basis.dimension))
        assert abs(min(ke)-e[0])<2e-11
        assert {t.term_id for t in h.instantaneous_terms}=={"C9:H2:INSTANT_FERMION","C9:H2:INSTANT_GLUON"}

def test_current_ward_and_neutron():
    h=fit_h2_trajectory(plans()[0],build_coupled_basis_tower())[0] if False else fit_h2_trajectory(plans()[0],build_coupled_basis_tower()).hamiltonians[0]
    e,v=solve(h);psi=v[:,0];current=H2VectorCurrent.for_hamiltonian(h)
    assert np.vdot(psi,current.matrix(h)@psi).real==pytest.approx(1)
    ward=ward_benchmark(h)
    assert abs(ward["residual"])<1e-14
    assert all(abs(x)>0 for x in ward["omission_residuals"].values())
    nh=fit_h2_trajectory(plans()[0],build_coupled_basis_tower("NEUTRON")).hamiltonians[0]
    ne,nv=solve(nh);nc=H2VectorCurrent.for_hamiltonian(nh)
    assert np.vdot(nv[:,0],nc.matrix(nh)@nv[:,0]).real==0
    with pytest.raises(ArchitectureError): current.matrix(nh)

def test_ledgers_ttn_feshbach_tracking():
    h=fit_h2_trajectory(plans()[0],build_coupled_basis_tower()).hamiltonians[-1]
    e,v=solve(h); ledger=gluon_oam_ledger(h,v[:,0])
    assert abs(ledger["probability_residual"])<1e-14
    assert abs(ledger["momentum_residual"])<1e-14 and abs(ledger["Jz_residual"])<1e-14
    assert 0<ledger["P_qqqg"]<1 and ledger["color_multiplicities"]==[1,2]
    ttn=coupled_ttn_benchmark(h);rows=ttn["rows"]
    assert all(a["energy"]>=b["energy"]-1e-13 for a,b in zip(rows,rows[1:]))
    assert all(x["energy"]>=ttn["exact_energy"]-2e-12 for x in rows)
    assert ttn["full_bond_residual"]<2e-12 and rows[0]["P_qqqg_error"]>1e-5
    f=feshbach_comparison(h);assert f["equivalence_residual"]<1e-14 and f["remainder_norm"]>0
    assert sector_tracking_benchmark()["eigenvalue_order_fails"]

def test_wilson_adapter_cut_support_and_coupling():
    data=MicroscopicRescatteringInput("C9:BUNDLE","C9:H")
    adapter=MicroscopicWilsonInputAdapter()
    assert adapter.absorption(data)==0 and adapter.absorption(data,epsilon=1e-3)==0
    assert adapter.absorption(data,spectral_rule=.2)==.2
    assert adapter.status=="MICROSCOPIC_WILSON_INPUT_INTERFACE_VALIDATED"
    with pytest.raises(ArchitectureError):adapter.absorption(data,separate_coupling=.2)

def test_injections():
    assert len(INJECTIONS)==83 and len({x[0] for x in INJECTIONS})==83
