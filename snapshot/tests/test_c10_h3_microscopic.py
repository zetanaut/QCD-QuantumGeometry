import numpy as np,pytest
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h0.color import ColorSingletBasis
from deuteron_wigner.microscopic.h0.permutation import PermutationBasis
from deuteron_wigner.microscopic.h3.core import *
from deuteron_wigner.microscopic.h3.diagnostics import *
from deuteron_wigner.microscopic.h3.injections import INJECTIONS
def test_plans_and_exclusion():
 assert len({p.plan_id for p in plans()})==2
 with pytest.raises(ArchitectureError):compile_h3_plan(H3AssumptionBundle("H3-PLAN-A"),induced_sea=True)
def test_color_and_antisymmetry():
 c=ColorSingletBasis.construct("qqqq-qbar");assert c.multiplicity==3 and c.generator_residual()<2e-14 and c.orthonormality_residual()<2e-14
 p=PermutationBasis(4);r=p.residuals();assert r["idempotence"]<2e-14 and r["hermiticity"]<2e-14
def test_basis_growth_flavor_and_multiplicity():
 t=build_h3_basis_tower();assert [b.dimensions for b in t]==[(4,6,9,9),(7,10,15,15),(10,14,21,21)]
 for b in t:
  assert {x.flavor for x in b.uubar}=={"u"} and {x.flavor for x in b.ddbar}=={"d"}
  assert {x.color_multiplicity for x in b.uubar}=={1,2,3}
def test_hamiltonian_flow_hermiticity_krylov():
 for plan in plans():
  tr=fit_trajectory(plan,build_h3_basis_tower())
  for m,h in zip(tr.members,tr.hamiltonians):
   assert abs(m["mass2"]-.7744)<2e-11 and np.max(abs(h.matrix-h.matrix.T))<2e-13
   from scipy.sparse.linalg import eigsh,LinearOperator
   e=np.linalg.eigvalsh(h.matrix)[0];ke=eigsh(LinearOperator(h.matrix.shape,matvec=h.apply,dtype=float),k=1,which="SA",v0=np.ones(h.basis.dimension),return_eigenvectors=False)[0]
   assert abs(e-ke)<2e-10
def test_pcac_ledgers_ttn_common_parent_feshbach():
 h=fit_trajectory(plans()[0],build_h3_basis_tower()).hamiltonians[-1];e,v=solve(h);psi=v[:,0]
 pc=pcac_report(h);assert abs(pc["residual"])<2e-14 and all(abs(x)>0 for x in pc["omission_residuals"].values())
 L=ledger(h,psi);assert abs(L["probability_residual"])<2e-13 and abs(L["momentum_residual"])<2e-13 and abs(L["Jz_residual"])<2e-13
 assert L["valence_u"]==2 and L["valence_d"]==1 and L["charge"]==1 and L["baryon"]==1
 T=ttn_benchmark(h);assert T["full_residual"]<2e-12 and T["rows"][0]["sea_probability_error"]>1e-5
 assert all(a["energy"]>=b["energy"]-1e-12 for a,b in zip(T["rows"],T["rows"][1:]))
 C=common_parent(h,psi);assert C["closure_residual"]==0 and all(x["positive_x_direct"] for x in C["rows"])
 F=feshbach(h);assert F["equivalence_residual"]<2e-14 and F["remainder_norm"]>0
 W=antiquark_wilson_handoff(h);assert W["absorption"]==0 and not W["WILSON_READY"]
def test_positive_x_direct():
 e=AntiquarkOverlapEvaluator("state")
 assert e.evaluate("ubar",.2)>0 and e.evaluate("dbar",.2)>0
 with pytest.raises(ArchitectureError):e.evaluate("ubar",0)
def test_injections():assert len(INJECTIONS)==90 and len({x[0] for x in INJECTIONS})==90
