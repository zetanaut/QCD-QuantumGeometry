"""Two-energy regulated-propagator Ward identity, including endpoint currents."""
import numpy as np
import pytest

from deuteron_wigner.quantum.m2_shell_projector_current import QShellProjectorFamily
from deuteron_wigner.quantum.m2_projected_shell_resolvent import ProjectedShellFreeResolvent
from deuteron_wigner.quantum.m2_shell_free_propagation import ShellFreePropagation
from deuteron_wigner.quantum.m2_shell_transverse_current import shell_transverse_current_between
from test_m2_shell_projector_current import SECTORS, space, random_columns


@pytest.fixture(scope="module", params=[(sector, jz) for sector in SECTORS for jz in (1, -1)])
def control(request):
    sector, jz = request.param
    s, f = space(sector, 2, 1), space(sector, 2, jz)
    family = QShellProjectorFamily({sector: f}, {sector: s}, inner_cutoffs={sector: 0},
        transfer=(.12, .16), b_GeV=.4, charge="electromagnetic", quadrature_order=16)
    args = dict(b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0.)
    hf, hi = ShellFreePropagation(f, **args), ShellFreePropagation(s, **args)
    rf, ri = ProjectedShellFreeResolvent(f, inner_cutoff=0, **args), ProjectedShellFreeResolvent(s, inner_cutoff=0, **args)
    ef, ei, P = 1.39, 1.47, 1.3
    Rf, Ri = lambda v: {sector: rf.resolve(ef, v[sector])}, lambda v: {sector: ri.resolve(ei, v[sector])}
    Hf, Hi = lambda v: {sector: hf.apply(v[sector])}, lambda v: {sector: hi.apply(v[sector])}
    a = {sector: random_columns(s, 596)}
    kw = dict(resolve_incoming=Ri, resolve_outgoing=Rf, parent_incoming=Hi, parent_outgoing=Hf, total_pplus_GeV=P)
    reg = [family.resolvent_boundary(a, component=j, **kw)[sector] for j in (0, 1)]
    rin = Ri(a)[sector]
    parent = [rf.resolve(ef, shell_transverse_current_between(f, s, rin, (.12, .16),
        b_GeV=.4, total_pplus_GeV=P, charge="electromagnetic", component=j)) for j in (0, 1)]
    return sector, family, a, reg, parent, kw, rf, ri, ef, ei, P


def test_two_energy_ward_identity_on_entire_outer_coordinate_set(control):
    n, family, a, reg, parent, kw, rf, ri, ef, ei, P = control
    Ri, Rf = kw["resolve_incoming"], kw["resolve_outgoing"]
    jp = Rf(family.density(Ri(a)))[n]
    lhs = (ef-ei)*jp-2*P*(.12*(parent[0]+reg[0])+.16*(parent[1]+reg[1]))
    rhs = family.density(Ri(a))[n]-Rf(family.density(a))[n]
    np.testing.assert_allclose(lhs, rhs, atol=5e-17)
    # Keeping R_f B R_i alone misses the moving projector endpoints.
    inner_only = []
    for j in (0, 1):
        b = family.hamiltonian_boundary(Ri(a), component=j,
            parent_incoming=kw["parent_incoming"], parent_outgoing=kw["parent_outgoing"],
            total_pplus_GeV=P)
        inner_only.append(Rf(b)[n])
    omitted = 2*P*(.12*(reg[0]-inner_only[0])+.16*(reg[1]-inner_only[1]))
    assert np.linalg.norm(omitted) > 1e-4


def test_regulator_current_has_reversed_energy_and_spin_endpoints(control):
    n, family, a, reg, parent, kw, rf, ri, ef, ei, P = control
    reverse = QShellProjectorFamily(family.incoming, family.outgoing, inner_cutoffs={n: 0},
        transfer=(-.12, -.16), b_GeV=.4, charge="electromagnetic", quadrature_order=16)
    b = {n: random_columns(family.outgoing[n], 469)}
    for j in (0, 1):
        value = reverse.resolvent_boundary(b, component=j,
            resolve_incoming=kw["resolve_outgoing"], resolve_outgoing=kw["resolve_incoming"],
            parent_incoming=kw["parent_outgoing"], parent_outgoing=kw["parent_incoming"],
            total_pplus_GeV=P)[n]
        scale = max(np.linalg.norm(b[n])*np.linalg.norm(reg[j]), np.linalg.norm(a[n])*np.linalg.norm(value))
        assert np.vdot(b[n], reg[j]) == pytest.approx(np.vdot(value, a[n]),
            abs=64*np.finfo(float).eps*scale, rel=0.)


@pytest.mark.parametrize("sector", SECTORS)
def test_projection_of_inverse_is_not_inverse_of_projected_H_and_energy_derivative(sector):
    s = space(sector, 2, 1)
    args = dict(b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0.)
    outer = ShellFreePropagation(s, **args)
    r = ProjectedShellFreeResolvent(s, inner_cutoff=0, **args)
    a = random_columns(s, 727)
    mask = np.zeros(s.dimension, bool)
    mask[r.rows] = True
    projected = mask[:, None]*a
    exact = r.resolve(1.4, a)
    np.testing.assert_allclose(1.4*exact-r.apply(exact), projected, atol=3e-17)
    wrong = mask[:, None]*outer.resolve(1.4, projected)
    # Independent N=0 -> N=2 Schur complement: d equal radial neighbors
    # have coupling b^2 and diagonal (d+2)b^2. The sea's larger mass makes
    # the absolute difference smaller, not zero or a permission to equate them.
    d, b2 = len(s.orbitals[0])//2, .4**2
    mass = outer.mass_offsets[s.coordinates[r.rows, 0]]
    denominator = 1.4-mass-d*b2-d*b2**2/(1.4-mass-(d+2)*b2)
    np.testing.assert_allclose(wrong[r.rows], a[r.rows]/denominator[:, None], atol=4e-18)
    scale = max(np.linalg.norm(exact), np.linalg.norm(wrong))
    assert np.linalg.norm(exact-wrong) > 1e6*np.finfo(float).eps*scale
    eps = 2e-5
    derivative = (r.resolve(1.4+eps, a)-r.resolve(1.4-eps, a))/(2*eps)
    np.testing.assert_allclose(derivative, -r.resolve(1.4, exact), atol=2e-11)
    np.testing.assert_array_equal(exact[~mask], 0.)


def test_zero_transfer_resolvent_charge_and_pole_not_clipped():
    n = "qqqgg"
    s = space(n, 2, 1)
    r = ProjectedShellFreeResolvent(s, inner_cutoff=0, b_GeV=.4,
                                    quark_mass_GeV=.35, gluon_mass_GeV=0.)
    f = QShellProjectorFamily({n: s}, {n: s}, inner_cutoffs={n: 0},
        transfer=(0., 0.), b_GeV=.4, charge="electromagnetic", quadrature_order=16)
    a = random_columns(s, 78)
    ei, ef = 1.39, 1.47
    density = f.density({n: r.resolve(ei, a)})[n]
    np.testing.assert_allclose((ef-ei)*r.resolve(ef, density),
                               r.resolve(ei, a)-r.resolve(ef, a), atol=2e-17)
    with pytest.raises(ValueError, match="pole"):
        r.resolve(r.free.minimum_energy_GeV2, a)


@pytest.mark.parametrize("jz", [1, -1])
def test_non_diagonal_inner_shell_resolves_the_projected_outer_equation(jz):
    s = space("qqqgg", 3, jz)
    args = dict(b_GeV=.4, quark_mass_GeV=.35, gluon_mass_GeV=0.)
    outer = ShellFreePropagation(s, **args)
    r = ProjectedShellFreeResolvent(s, inner_cutoff=2, **args)
    a = random_columns(s, 491)
    mask = np.zeros((s.dimension, 1), bool)
    mask[r.rows] = True
    answer = r.resolve(1.4, a)
    np.testing.assert_allclose(r.apply(a), mask*outer.apply(mask*a), atol=3e-17)
    np.testing.assert_allclose(1.4*answer-mask*outer.apply(answer), mask*a, atol=4e-17)
    # Keep actual radial off-diagonal contributions, not diagonal energies.
    diagonal_answer = np.zeros_like(a)
    occ = np.asarray(s.orbitals)
    number = occ.sum(axis=1)[s.coordinates[r.rows, 1]]
    mass = outer.mass_offsets[s.coordinates[r.rows, 0]]
    diagonal_answer[r.rows] = a[r.rows]/(1.4-mass-(4+number)*.4**2)[:, None]
    assert np.linalg.norm(answer-diagonal_answer) > 1e-3
