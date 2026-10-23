"""C3 common evaluator and analytic Benchmarks A--D."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from deuteron_wigner.formal.diagnostics import ArchitectureError
from deuteron_wigner.formal.gauge_path import StapleOrientation
from deuteron_wigner.formal.legacy_adapters import registry_operator_identity
from deuteron_wigner.formal.maps import MapClass
from deuteron_wigner.formal.operator_identity import IdentityState
from deuteron_wigner.gtmd import Species
from deuteron_wigner.kinematics import MomentumTransfer, PartonMomentum
from deuteron_wigner.pilot.configuration import ColorLabel, Constituent, IntrinsicConfiguration
from deuteron_wigner.pilot.fibers import ZeroSkewnessFrame
from deuteron_wigner.pilot.overlap import AnalyticOverlapEvaluator, OverlapKernel
from deuteron_wigner.pilot.recoil import SymmetricXiZeroRecoil
from deuteron_wigner.pilot.states import GaussianScalarState, PointState, SpinorOAMState, ThreeQuarkColorState, neutron_from_proton, pilot_sector
from deuteron_wigner.registry import leading_twist_quark_registry
from deuteron_wigner.tmd_scheme import DELTA_COLLINS_ZETA_SCHEME, TMDScalePoint


def operator():
    entry = leading_twist_quark_registry().get(Species.QUARK, "f1")
    return registry_operator_identity(entry, flavor="u", scale=TMDScalePoint.canonical(5), scheme=DELTA_COLLINS_ZETA_SCHEME, orientation=StapleOrientation.FUTURE)


def kernel(frame, sector, active=0, **changes):
    value = OverlapKernel("kernel", "q", "u", active, sector.basis_id, sector.basis_id, "vector", "identity", "exact spectator delta", 1.0, frame.incoming, frame.outgoing, "SYMMETRIC_XI0", operator())
    return replace(value, **changes)


def point_config():
    sector = pilot_sector("point", ("u",))
    return IntrinsicConfiguration((Constituent("q", 1, PartonMomentum(0,0), Species.QUARK, "u", ColorLabel.RED, 1, 0, "point"),), 0, sector, "point", "real", "one")


def scalar_config(x=.4, k=(.2,-.1)):
    sector = pilot_sector("scalar", ("u",))
    return IntrinsicConfiguration((
        Constituent("q", x, PartonMomentum(*k), Species.QUARK, "u", ColorLabel.RED, 1, 0, "scalar"),
        Constituent("s", 1-x, PartonMomentum(-k[0],-k[1]), Species.GLUON, "scalar", ColorLabel.NONE, 0, 0, "scalar"),
    ), 0, sector, "gaussian", "real", "distinguishable")


def frame(sector, delta=(.3,-.2)):
    return ZeroSkewnessFrame.symmetric(p_plus=2, mass_gev=.94, delta_t=MomentumTransfer(*delta), sector_scope=sector.basis_id)


def test_benchmark_a_point_state_exact_for_any_transfer():
    config = point_config(); frm = frame(config.sector)
    recoil = SymmetricXiZeroRecoil().apply(config, frm)
    assert recoil.incoming.constituents == recoil.outgoing.constituents == config.constituents
    result = AnalyticOverlapEvaluator().evaluate(PointState(), config, recoil, kernel(frm, config.sector))
    assert result.value == 1
    with pytest.raises(ArchitectureError, match="C3.ISOLATE.PROMOTION"):
        result.authorize_production()


@pytest.mark.parametrize("x,k,delta", [(.2,(.1,.0),(.2,.0)),(.4,(.2,-.1),(.3,-.2)),(.7,(-.3,.2),(-.1,.4))])
def test_benchmark_b_common_evaluator_matches_independent_gaussian_oracle(x,k,delta):
    config = scalar_config(x,k); frm = frame(config.sector, delta)
    state = GaussianScalarState(.45)
    result = AnalyticOverlapEvaluator().evaluate(state, config, SymmetricXiZeroRecoil().apply(config, frm), kernel(frm, config.sector))
    expected = state.analytic_overlap(x, k[0]**2+k[1]**2, delta[0]**2+delta[1]**2)
    assert result.value.real == pytest.approx(expected, rel=2e-15)
    reverse = frame(config.sector, (-delta[0],-delta[1]))
    partner = AnalyticOverlapEvaluator().evaluate(state, config, SymmetricXiZeroRecoil().apply(config, reverse), kernel(reverse, config.sector))
    assert result.value.conjugate() == pytest.approx(partner.value, rel=2e-15)


def test_benchmark_c_oam_real_zero_complex_activation_and_matrix_closure():
    real = SpinorOAMState((1,.2,.2), .94)
    assert real.phase_odd() == 0
    assert real.rank_one_interference() == 0
    complex_member = SpinorOAMState((1,.2j,-.1j), .94)
    assert complex_member.phase_odd() != 0
    matrix = complex_member.helicity_matrix()
    assert matrix.shape == (4,4)
    assert np.allclose(matrix, matrix.conj().T)
    assert np.trace(matrix) == pytest.approx(1)
    assert SpinorOAMState((1,0,0), .94).rank_one_interference() == 0
    assert SpinorOAMState((0,.2j,-.1j), .94).rank_one_interference() == 0
    config=scalar_config(); frm=frame(config.sector)
    result=AnalyticOverlapEvaluator().evaluate(complex_member,config,SymmetricXiZeroRecoil().apply(config,frm),kernel(frm,config.sector))
    assert np.isfinite(result.value)


def test_benchmark_d_color_singlet_counts_and_reversible_isospin():
    proton = ThreeQuarkColorState()
    assert proton.color_norm() == pytest.approx(1, abs=3e-16)
    assert proton.total_color_generator_residual() < 2e-16
    assert proton.counts() == {"d":1,"u":2}
    neutron = neutron_from_proton(proton)
    assert neutron.counts() == {"d":2,"u":1}
    assert neutron_from_proton(neutron).counts() == proton.counts()
    assert np.array_equal(neutron.color_tensor(), proton.color_tensor())
    sector=pilot_sector("color",("u","u","d"))
    colors=(ColorLabel.RED,ColorLabel.GREEN,ColorLabel.BLUE)
    config=IntrinsicConfiguration(tuple(
        Constituent(f"q{i}",1/3,PartonMomentum(0,0),Species.QUARK,flavor,color,1,0,"color")
        for i,(flavor,color) in enumerate(zip(proton.flavors,colors))
    ),0,sector,"color","real","S3")
    frm=frame(sector)
    result=AnalyticOverlapEvaluator().evaluate(proton,config,SymmetricXiZeroRecoil().apply(config,frm),kernel(frm,sector))
    permuted=IntrinsicConfiguration((config.constituents[1],config.constituents[0],config.constituents[2]),1,sector,"color","real","S3")
    p_result=AnalyticOverlapEvaluator().evaluate(proton,permuted,SymmetricXiZeroRecoil().apply(permuted,frm),kernel(frm,sector,active=1))
    assert abs(result.value)==pytest.approx(abs(p_result.value))


def test_kernel_rejects_wrong_class_sector_wilson_and_fiber():
    config=point_config(); frm=frame(config.sector); base=kernel(frm,config.sector)
    with pytest.raises(ArchitectureError, match="C3.KERNEL.CLASS"):
        replace(base,map_class=MapClass.RED)
    with pytest.raises(ArchitectureError, match="C3.KERNEL.SECTOR"):
        replace(base,target_sector_id="other")
    with pytest.raises(ArchitectureError, match="C3.KERNEL.WILSON"):
        replace(base,wilson_order=1)
    bad_frame=ZeroSkewnessFrame.symmetric(p_plus=3,mass_gev=.94,delta_t=MomentumTransfer(.3,-.2),sector_scope=config.sector.basis_id)
    with pytest.raises(ArchitectureError, match="C3.FIBER"):
        replace(base,target_fiber=bad_frame.outgoing)
