from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.microscopic.h0.basis import (
    PartonBasisState, reference_basis,
)
from deuteron_wigner.microscopic.h0.cm import CenterOfMassPolicy
from deuteron_wigner.microscopic.h0.color import (
    ColorSingletBasis, representation_generators,
)
from deuteron_wigner.microscopic.h0.injections import (
    INJECTIONS, detect_injected_violation,
)
from deuteron_wigner.microscopic.h0.permutation import PermutationBasis
from deuteron_wigner.microscopic.h0.readiness import (
    H0Readiness, provenance_graph, require_isolation,
)
from deuteron_wigner.microscopic.h0.resolution import (
    HamiltonianScale, OscillatorScale, lf_invariant_mass_squared,
    reference_resolution,
)
from deuteron_wigner.microscopic.h0.terms import (
    FreeInvariantMassTerm, ReducedCanonicalVertexTerm,
)


def test_light_front_factor_two_and_resolution_scale_separation():
    assert lf_invariant_mass_squared(3,4,2)==22
    resolution=reference_resolution()
    assert isinstance(resolution.oscillator_scale_b,OscillatorScale)
    assert isinstance(resolution.hamiltonian_resolution_lambda,HamiltonianScale)
    assert resolution.oscillator_scale_b.type_id!=resolution.hamiltonian_resolution_lambda.type_id
    assert resolution.to_dict()["K"]==[9,2]
    assert resolution.resolution_id==reference_resolution().resolution_id
    assert resolution.resolution_id!=reference_resolution(N_max=10).resolution_id


def test_exact_longitudinal_modes_and_physical_blocks():
    resolution=reference_resolution()
    for sector in ("qqq","qqqg","qqqq-qbar"):
        for proton in (True,False):
            for jz in (Fraction(1,2),Fraction(-1,2)):
                basis=reference_basis(resolution,sector,proton=proton,Jz=jz)
                assert basis.dimension=={"qqq":1,"qqqg":2,"qqqq-qbar":3}[sector]
                for state in basis.states:
                    assert state.total_longitudinal_mode==resolution.K
                    assert state.nmax_usage<=resolution.N_max
                    assert state.Jz==jz
                    assert state.baryon_number==1
                    assert all(isinstance(item.longitudinal_fraction_exact,Fraction) for item in state.creation_labels)


def test_gluon_zero_mode_and_floating_fermion_modes_fail():
    r=reference_resolution()
    with pytest.raises(ArchitectureError,match="C7.MODE"):
        PartonBasisState("GLUON","NOT_APPLICABLE",Fraction(0),0,0,Fraction(1),"a",r.resolution_id,r.K)
    with pytest.raises(ArchitectureError,match="C7.MODE"):
        PartonBasisState("QUARK","u",Fraction(1),0,0,Fraction(1,2),"c",r.resolution_id,r.K)


@pytest.mark.parametrize("sector,expected",(("qqq",1),("qqqg",2),("qqqq-qbar",3)))
def test_hc_complete_color_multiplicity_generator_and_recoupling(sector,expected):
    color=ColorSingletBasis.construct(sector)
    assert color.multiplicity==expected
    assert color.invariant_dimension_from_rank()==expected
    assert color.generator_residual()<2e-14
    assert color.orthonormality_residual()<1e-14
    assert color.recoupling_unitarity_residual()<1e-14
    assert len(set(color.content_hashes()))==expected


def test_antiquark_generator_is_antifundamental():
    fundamental=representation_generators("3")
    anti=representation_generators("3bar")
    for t,tbar in zip(fundamental,anti):
        np.testing.assert_allclose(tbar,-t.T,atol=0)


@pytest.mark.parametrize("count",(2,3,4))
def test_exact_fermion_antisymmetrizer_before_assembly(count):
    permutation=PermutationBasis(count)
    residuals=permutation.residuals()
    assert residuals["idempotence"]<2e-16
    assert residuals["hermiticity"]==0
    assert permutation.exchange_sign(0,count-1)==-1


def test_center_of_mass_factorization_and_lawson_gate():
    policy=CenterOfMassPolicy()
    intrinsic=np.asarray((2.0,3.0))
    policy.require_ready((0,0),intrinsic)
    assert policy.factorization_residual((0,0))==0
    assert policy.intrinsic_drift(intrinsic)==0
    spectra=policy.lawson_spectra(intrinsic,np.asarray((20.0,)))
    assert spectra[-1][-1]>spectra[0][-1]
    with pytest.raises(ArchitectureError,match="C7.CM"):
        policy.require_ready((0,1),intrinsic)


@pytest.mark.parametrize("nmax,b",((8,0.4),(8,0.45),(10,0.5)))
@pytest.mark.parametrize("sector",("qqq","qqqg","qqqq-qbar"))
def test_ha_free_operator_assembled_matrix_free_quadrature_and_hermiticity(nmax,b,sector):
    resolution=reference_resolution(N_max=nmax,b=b)
    basis=reference_basis(resolution,sector)
    term=FreeInvariantMassTerm.for_sector(sector)
    matrix=term.assemble(basis)
    rng=np.random.default_rng(7100+nmax+len(sector))
    vector=rng.normal(size=basis.dimension)+1j*rng.normal(size=basis.dimension)
    np.testing.assert_allclose(term.apply(vector,basis),matrix@vector,atol=1e-14)
    assert np.max(np.abs(matrix-matrix.conj().T))==0
    assert term.quadrature_residual(basis)<5e-12
    assert np.count_nonzero(matrix)==basis.dimension
    if basis.dimension>1:
        assert np.max(np.abs(np.diag(matrix)-matrix[0,0]))==0


def test_hb_vertex_and_generated_adjoint_are_hermitian_for_basis_and_random_vectors():
    resolution=reference_resolution()
    qqq=reference_basis(resolution,"qqq")
    qqqg=reference_basis(resolution,"qqqg")
    emission=ReducedCanonicalVertexTerm.emission()
    absorption=emission.adjoint()
    V=emission.matrix(qqq,qqqg)
    Vdag=absorption.matrix(qqqg,qqq)
    np.testing.assert_allclose(Vdag,V.conj().T,atol=0)
    rng=np.random.default_rng(7123)
    source=rng.normal(size=qqq.dimension)+1j*rng.normal(size=qqq.dimension)
    target=rng.normal(size=qqqg.dimension)+1j*rng.normal(size=qqqg.dimension)
    lhs=np.vdot(target,V@source)
    rhs=np.vdot(Vdag@target,source)
    assert abs(lhs-rhs)<1e-14
    assert emission.parameter_block_id==absorption.parameter_block_id
    assert emission.regulator_identity==absorption.regulator_identity
    assert emission.emitter_index==1
    assert emission.approximation_status=="REDUCED_CANONICAL_INTERFACE_BENCHMARK"


def test_h0_readiness_ceiling_and_isolated_provenance():
    readiness=H0Readiness()
    readiness.require("FREE_OPERATOR_VALIDATED")
    for unavailable in readiness.unavailable:
        with pytest.raises(ArchitectureError,match="C7.READINESS"):
            readiness.require(unavailable)
    graph=provenance_graph()
    assert graph["production_reachable"] is False
    require_isolation({"production:root","volume_v:evolution","volume_vi:inference"})
    with pytest.raises(ArchitectureError,match="C7.ISOLATION"):
        require_isolation({"C7:H0:FREE_OPERATOR"})


@pytest.mark.parametrize("stable_id,description,diagnostic",INJECTIONS)
def test_all_48_c7_injections_have_structured_diagnostics(stable_id,description,diagnostic):
    with pytest.raises(ArchitectureError) as caught:
        detect_injected_violation(stable_id)
    assert caught.value.requirement_id==diagnostic


def test_c7_injection_ledger_is_complete_and_ordered():
    assert len(INJECTIONS)==48
    assert [row[0] for row in INJECTIONS]==[f"C7.INJECT.{i:02d}" for i in range(1,49)]
