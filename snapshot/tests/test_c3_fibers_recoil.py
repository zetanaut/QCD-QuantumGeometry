"""C3 fiber, configuration, recoil algebra, and mismatch injections."""

from __future__ import annotations

from dataclasses import replace

import pytest

from deuteron_wigner.formal.coordinates import CoordinateKind, coordinate_spec
from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.gtmd import Species
from deuteron_wigner.kinematics import MomentumTransfer, PartonMomentum
from deuteron_wigner.pilot.configuration import ColorLabel, Constituent, IntrinsicConfiguration
from deuteron_wigner.pilot.fibers import ZeroSkewnessFrame
from deuteron_wigner.pilot.recoil import SymmetricXiZeroRecoil
from deuteron_wigner.pilot.states import pilot_sector


def two_body(active=0):
    return IntrinsicConfiguration((
        Constituent("active", 0.4, PartonMomentum(0.2, -0.1), Species.QUARK, "u", ColorLabel.RED, 1, 0, "toy"),
        Constituent("spectator", 0.6, PartonMomentum(-0.2, 0.1), Species.GLUON, "scalar", ColorLabel.NONE, 0, 0, "toy"),
    ), active, pilot_sector("two_body", ("u",)), "member", "real", "distinguishable")


def frame(delta=(0.3, -0.2)):
    return ZeroSkewnessFrame.symmetric(p_plus=2.0, mass_gev=0.94, delta_t=MomentumTransfer(*delta), sector_scope="two_body")


def test_fibers_are_symmetric_serializable_and_xi_zero():
    value = frame()
    assert value.incoming.p_transverse == value.delta_t.scale(-0.5)
    assert value.outgoing.p_transverse == value.delta_t.scale(0.5)
    assert value.average.p_transverse.norm_squared() == 0
    with pytest.raises(ArchitectureError, match="C3.FIBER.XI"):
        replace(value, xi=0.1)
    with pytest.raises(ArchitectureError, match="C1.COORD"):
        replace(value, delta_coordinate=coordinate_spec(CoordinateKind.B_TMD))


def test_configuration_support_closure_duplicate_and_active_failures():
    base = two_body()
    with pytest.raises(ArchitectureError, match="C3.CONFIG.SUPPORT"):
        replace(base, constituents=(replace(base.constituents[0], x=0.5), base.constituents[1]))
    with pytest.raises(ArchitectureError, match="C3.CONFIG.CLOSURE"):
        replace(base, constituents=(replace(base.constituents[0], k_t=PartonMomentum(0.3, 0)), base.constituents[1]))
    with pytest.raises(ArchitectureError, match="C3.CONFIG.DUPLICATE"):
        replace(base, constituents=(base.constituents[0], replace(base.constituents[1], stable_id="active")))
    with pytest.raises(ArchitectureError, match="C3.CONFIG.ACTIVE"):
        replace(base, active_index=2)


def test_recoil_closure_physical_assignment_jacobian_and_involution():
    recoil = SymmetricXiZeroRecoil()
    result = recoil.apply(two_body(), frame())
    recoil.verify_physical_assignment(result)
    assert result.jacobian == 1
    reversed_result = recoil.apply(two_body(), frame((-0.3, 0.2)))
    assert result.incoming.constituents == reversed_result.outgoing.constituents
    assert result.outgoing.constituents == reversed_result.incoming.constituents
    identity = recoil.apply(two_body(), frame((0, 0)))
    assert identity.incoming.constituents == two_body().constituents == identity.outgoing.constituents
    with pytest.raises(ArchitectureError, match="C3.RECOIL.JACOBIAN"):
        recoil.apply(two_body(), frame(), jacobian=0.5)


def test_recoil_permutation_covariance():
    original = two_body(0)
    permuted = IntrinsicConfiguration(tuple(reversed(original.constituents)), 1, original.sector, original.member_id, original.phase_id, original.permutation_class)
    a = SymmetricXiZeroRecoil().apply(original, frame())
    b = SymmetricXiZeroRecoil().apply(permuted, frame())
    assert a.incoming.constituents == tuple(reversed(b.incoming.constituents))
    assert a.outgoing.constituents == tuple(reversed(b.outgoing.constituents))
