"""C4 reuses the C3 fibers, recoil authority, and overlap evaluator."""

from dataclasses import replace

import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.gauge_path import ColorClass, StapleOrientation
from deuteron_wigner.formal.legacy_adapters import registry_operator_identity
from deuteron_wigner.formal.operator_identity import IdentityState
from deuteron_wigner.gtmd import Species
from deuteron_wigner.kinematics import MomentumTransfer
from deuteron_wigner.pilot.active import PositiveXActiveSelector
from deuteron_wigner.pilot.fibers import ZeroSkewnessFrame
from deuteron_wigner.pilot.overlap import AnalyticOverlapEvaluator, OverlapKernel
from deuteron_wigner.pilot.recoil import SymmetricXiZeroRecoil
from deuteron_wigner.pilot.sectors import gluon_state, sea_state
from deuteron_wigner.registry import (
    leading_twist_gluon_registry, leading_twist_quark_registry,
)
from deuteron_wigner.tmd_scheme import (
    DELTA_COLLINS_ZETA_SCHEME, TMDScalePoint,
)


def operator(species):
    registry = (
        leading_twist_gluon_registry()
        if species == Species.GLUON else leading_twist_quark_registry()
    )
    registry_species = (
        Species.QUARK if species == Species.ANTIQUARK else species
    )
    entry = registry.get(registry_species, "f1")
    value = registry_operator_identity(
        entry,
        flavor=(
            IdentityState.NOT_APPLICABLE
            if species == Species.GLUON else "d"
        ),
        scale=TMDScalePoint.canonical(5),
        scheme=DELTA_COLLINS_ZETA_SCHEME,
        orientation=StapleOrientation.FUTURE,
        gluon_color_class=(
            ColorClass.DIAGONAL_ADJOINT
            if species == Species.GLUON else ColorClass.NOT_APPLICABLE
        ),
    )
    return (
        replace(value, parton_species=Species.ANTIQUARK.value)
        if species == Species.ANTIQUARK else value
    )


def evaluate_member(state, species, delta):
    sector = state.sectors[1]
    selection = PositiveXActiveSelector().select(
        sector.configuration, species
    )[0]
    config = sector.configuration.with_active(selection.slot_index)
    frame = ZeroSkewnessFrame.symmetric(
        p_plus=2.0, mass_gev=.94, delta_t=MomentumTransfer(*delta),
        sector_scope=config.sector.basis_id, member="C4_VALIDATION_ONLY",
    )
    recoil = SymmetricXiZeroRecoil().apply(config, frame)
    SymmetricXiZeroRecoil().verify_physical_assignment(recoil)
    kernel = OverlapKernel(
        f"C4:KERNEL:{species.value}", species.value,
        "NOT_APPLICABLE" if species == Species.GLUON else selection.flavor,
        selection.slot_index, config.sector.basis_id, config.sector.basis_id,
        "trace" if species == Species.GLUON else "vector",
        "DIAGONAL_ADJOINT" if species == Species.GLUON else "identity",
        "exact spectator delta", 1.0, frame.incoming, frame.outgoing,
        "SYMMETRIC_XI0", operator(species),
    )
    return AnalyticOverlapEvaluator().evaluate(
        sector.state, config, recoil, kernel
    ), kernel


@pytest.mark.parametrize(
    "state,species",
    ((sea_state(.2), Species.ANTIQUARK),
     (gluon_state(.3), Species.GLUON)),
)
def test_c4_members_use_common_evaluator_forward_and_hermiticity(state, species):
    forward, _ = evaluate_member(state, species, (0.0, 0.0))
    positive, _ = evaluate_member(state, species, (0.2, -0.1))
    reversed_result, _ = evaluate_member(state, species, (-0.2, 0.1))
    assert forward.value.real > 0
    assert positive.value.conjugate() == pytest.approx(reversed_result.value)
    assert positive.provenance_trace[1] == "SYMMETRIC_XI0"
    assert AnalyticOverlapEvaluator.stable_id == "C3:COMMON_DIAGONAL_OVERLAP"


def test_common_evaluator_rejects_wrong_species_and_off_diagonal_sector():
    state = sea_state(.2)
    _, kernel = evaluate_member(state, Species.ANTIQUARK, (0.0, 0.0))
    sector = state.sectors[1]
    selection = PositiveXActiveSelector().select(
        sector.configuration, Species.ANTIQUARK
    )[0]
    config = sector.configuration.with_active(selection.slot_index)
    frame = ZeroSkewnessFrame.symmetric(
        p_plus=2, mass_gev=.94, delta_t=MomentumTransfer(0, 0),
        sector_scope=config.sector.basis_id,
    )
    recoil = SymmetricXiZeroRecoil().apply(config, frame)
    with pytest.raises(ArchitectureError, match="C4.ACTIVE.SPECIES"):
        AnalyticOverlapEvaluator().evaluate(
            sector.state, config, recoil, replace(kernel, active_species="g")
        )
    with pytest.raises(ArchitectureError, match="C3.KERNEL.SECTOR"):
        replace(kernel, target_sector_id="different-sector")
